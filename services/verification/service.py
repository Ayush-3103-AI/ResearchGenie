"""Public entry point for the Citation Verification Service.

Takes a WritingResult (Service 2's output) and checks each rendered
reference-list entry's DOI against free, keyless scholarly APIs (Crossref,
then OpenAlex as a fallback — the same "no external key" pattern Service 1
uses, see DECISIONS.md D-007), up to 100 DOIs per filter request rather
than one request each (#7). Both lookups go through the scholarly gateway
(#4), so they share Discovery's pacing, retry, identity and breaker: a
throttled lookup is retried, and a lookup that still gets no answer is
"unverifiable", never "invalid". This service never invents or silently
repairs a reference: an entry with no extractable DOI, or a DOI that
resolves nowhere, is flagged in the validation report rather than dropped,
corrected, or hidden — see PROJECT_NOTES.md's "must not silently invent or
repair references" rule and DECISIONS.md D-012.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import AsyncIterator

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.contracts.verification_contract import (  # noqa: E402
    ReferenceCheck, VerificationRequest, VerificationResult,
)
from shared.utilities import scholarly_gateway  # noqa: E402

CROSSREF_URL = "https://api.crossref.org/works"
OPENALEX_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 10.0

# Matches a DOI trailed by "doi: 10.x/y" (IEEE/Vancouver style) or
# "https://doi.org/10.x/y" (author-year/MLA styles) at the end of a
# rendered reference entry (see writing/modules/citation_formatter.py).
_DOI_PATTERNS = (
    re.compile(r"doi:\s*(10\.\d{4,9}/\S+?)[.\s]*$", re.IGNORECASE),
    re.compile(r"https?://doi\.org/(10\.\d{4,9}/\S+?)[.\s]*$", re.IGNORECASE),
)

NO_DOI_ISSUE = "No DOI found in this reference entry — existence could not be checked automatically."
PROVIDER_EXHAUSTED_ISSUE = "unverifiable (provider exhausted):"


def extract_doi(reference_entry: str) -> str | None:
    """Return the DOI embedded in a rendered reference entry, if any."""
    stripped = reference_entry.rstrip()
    for pattern in _DOI_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return match.group(1).rstrip(".,;").lower()
    return None


def _bare_doi(doi: str | None) -> str:
    """OpenAlex answers "https://doi.org/10.x/y", Crossref "10.X/Y"."""
    return (doi or "").lower().removeprefix("https://doi.org/")


async def _lookup(client: httpx.AsyncClient, source: str, dois: list[str]) -> dict[str, bool | str]:
    """Each DOI's outcome at one provider, asked with one OR-filter request
    per DOI_BATCH DOIs rather than one request per DOI (#7). True: found.
    False: the provider answered without it. A string: no answer, saying
    why — "exhausted" when the gateway skipped the provider, else "failed"
    (retries used up, network error)."""
    # ponytail: a DOI holding "," or "|" breaks its batch's filter (HTTP 400,
    # the batch reads "failed"); none is expected from extract_doi's patterns.
    outcomes: dict[str, bool | str] = {}
    for start in range(0, len(dois), scholarly_gateway.DOI_BATCH):
        batch = dois[start:start + scholarly_gateway.DOI_BATCH]
        if source == "Crossref":  # repeated doi filters are OR-ed
            url, params = CROSSREF_URL, {"filter": ",".join(f"doi:{d}" for d in batch),
                                         "rows": len(batch)}
        else:
            url, params = OPENALEX_URL, {"filter": "doi:" + "|".join(batch), "per-page": len(batch)}
        try:
            resp = await scholarly_gateway.get(client, url, params=params, timeout=REQUEST_TIMEOUT)
            body = resp.json() if resp.status_code == 200 else None
        except scholarly_gateway.ProviderExhausted:
            outcomes.update(dict.fromkeys(batch, "exhausted"))
            continue
        except (httpx.HTTPError, ValueError):
            body = None
        if body is None:
            outcomes.update(dict.fromkeys(batch, "failed"))
            continue
        works = (body.get("message") or {}).get("items") if source == "Crossref" else body.get("results")
        found = {_bare_doi(w.get("DOI") or w.get("doi")) for w in works or []}
        outcomes.update({d: d in found for d in batch})
    return outcomes


async def check_references(client: httpx.AsyncClient, entries: list[str]) -> list[ReferenceCheck]:
    """Every entry's check: Crossref first, then OpenAlex for the DOIs
    Crossref did not confirm, each asked in batches (#7)."""
    dois = [extract_doi(entry) for entry in entries]
    pending = list(dict.fromkeys(d for d in dois if d))
    outcomes: dict[str, list[bool | str]] = {d: [] for d in pending}
    verified_by: dict[str, str] = {}
    for source in ("Crossref", "OpenAlex"):
        for doi, outcome in (await _lookup(client, source, pending)).items():
            if outcome is True:
                verified_by[doi] = source
            else:
                outcomes[doi].append(outcome)
        pending = [d for d in pending if d not in verified_by]
    return [_check(entry, doi, verified_by.get(doi), outcomes.get(doi, []))
            for entry, doi in zip(entries, dois)]


async def check_reference(client: httpx.AsyncClient, reference_entry: str) -> ReferenceCheck:
    return (await check_references(client, [reference_entry]))[0]


def _check(reference_entry: str, doi: str | None, source: str | None,
           outcomes: list[bool | str]) -> ReferenceCheck:
    if doi is None:
        return ReferenceCheck(reference_entry=reference_entry, verified=False, issue=NO_DOI_ISSUE)
    if source is not None:
        return ReferenceCheck(reference_entry=reference_entry, doi=doi, verified=True,
                              verification_source=source)
    if "exhausted" in outcomes:
        return ReferenceCheck(
            reference_entry=reference_entry, doi=doi, verified=False,
            issue=f"{PROVIDER_EXHAUSTED_ISSUE} DOI {doi} could not be checked because a "
                  "lookup provider is out of budget or rate-limited for this run — "
                  "this says nothing about whether the DOI is real.")
    if "failed" in outcomes:
        return ReferenceCheck(
            reference_entry=reference_entry, doi=doi, verified=False,
            issue=f"unverifiable (lookup failed): DOI {doi} got no answer from Crossref or "
                  "OpenAlex after retries — check it by hand.")
    return ReferenceCheck(
        reference_entry=reference_entry, doi=doi, verified=False,
        issue=f"DOI {doi} was not found in Crossref or OpenAlex — it may be incorrect or fabricated.",
    )


def build_validation_report(
    *, checked: int, verified: list[ReferenceCheck], invalid: list[ReferenceCheck],
    unverifiable: list[ReferenceCheck], missing_references: list[str],
    citation_issues: list[str],
) -> str:
    lines = [
        "# Citation Verification Report", "",
        f"- References checked: {checked}",
        f"- Verified: {len(verified)}",
        f"- Invalid (DOI not found): {len(invalid)}",
        f"- Unverifiable (no DOI, or no answer from the lookup providers): {len(unverifiable)}",
        "",
    ]
    if invalid:
        lines.append("## Invalid references")
        lines += [f"- {c.reference_entry}\n  - {c.issue}" for c in invalid]
        lines.append("")
    if unverifiable:
        lines.append("## Unverifiable references")
        lines += [f"- {c.reference_entry}\n  - {c.issue}" for c in unverifiable]
        lines.append("")
    if missing_references:
        lines.append("## Missing references")
        lines += [f"- {m}" for m in missing_references]
        lines.append("")
    if citation_issues:
        lines.append("## Citation issues")
        lines += [f"- {c}" for c in citation_issues]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def run_verification(request: VerificationRequest, *,
                           transport: httpx.AsyncBaseTransport | None = None,
                           ) -> AsyncIterator[dict]:
    """Yields progress events (the gateway's rate_limited / provider_exhausted
    among them), then a final {"type": "result", "result": VerificationResult}
    event. `transport` is a test seam (httpx.MockTransport)."""
    writing = request.writing
    references = writing.reference_candidates

    yield {"type": "status", "stage": "verify", "state": "running",
           "message": f"Verifying {len(references)} reference(s) against Crossref/OpenAlex…"}

    async def check_all() -> list[ReferenceCheck]:
        async with httpx.AsyncClient(transport=transport) as client:
            return await check_references(client, references)

    gateway_run = scholarly_gateway.Run()
    checking = gateway_run.task(check_all())
    async for event in gateway_run.relay(checking):
        yield event
    if (cached := gateway_run.cache_event()) is not None:
        yield cached

    verified: list[ReferenceCheck] = []
    invalid: list[ReferenceCheck] = []
    unverifiable: list[ReferenceCheck] = []
    for check in checking.result():
        if check.verified:
            verified.append(check)
        elif check.doi is None or (check.issue or "").startswith("unverifiable"):
            unverifiable.append(check)
        else:
            invalid.append(check)

    missing_references: list[str] = []
    cited_count = writing.draft_metadata.papers_cited
    if cited_count and cited_count != len(references):
        missing_references.append(
            f"Draft metadata reports {cited_count} paper(s) cited, but the reference "
            f"list has {len(references)} entr{'y' if len(references) == 1 else 'ies'} "
            "— counts should match; check for citations without a reference entry."
        )

    citation_issues: list[str] = [
        f"Carried over from the writing stage: {error}" for error in writing.draft_metadata.errors
    ]

    validation_report = build_validation_report(
        checked=len(references), verified=verified, invalid=invalid,
        unverifiable=unverifiable, missing_references=missing_references,
        citation_issues=citation_issues,
    )

    yield {"type": "status", "stage": "verify", "state": "done",
           "message": f"Verification complete — {len(verified)} verified, "
                      f"{len(invalid)} invalid, {len(unverifiable)} unverifiable."}

    validated_draft_markdown = writing.research_draft_markdown
    if invalid or unverifiable or missing_references:
        validated_draft_markdown += (
            "\n> **Citation verification found issues — see the validation report "
            "before using this draft.**\n"
        )

    result = VerificationResult(
        validated_draft_markdown=validated_draft_markdown,
        verified_references=verified,
        invalid_references=invalid,
        unverifiable_references=unverifiable,
        missing_references=missing_references,
        citation_issues=citation_issues,
        validation_report=validation_report,
    )
    yield {"type": "result", "result": result}
