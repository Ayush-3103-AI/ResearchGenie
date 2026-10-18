"""Public entry point for the Quality Assurance Service.

Takes Services 1-3's outputs (DiscoveryResult, WritingResult,
VerificationResult) and runs a final, independent audit: citation
integrity, claim-to-evidence alignment, required-section presence, and an
overall quality score. Deliberately does NOT call the local LLM for this
pass — grading a draft with the same model that wrote it is a weaker
independence guarantee than deterministic, rule-based checks, and this
hardware is already slow for LLM calls (see DECISIONS.md D-009/D-010). See
DECISIONS.md D-013 for the full adaptation record.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.contracts.qa_contract import (  # noqa: E402
    MissingCitation, QualityAssuranceRequest, QualityAssuranceResult,
    QualityScores, UnsupportedClaim,
)

_CITATION = re.compile(r"\[(\d+)\]")
_REFERENCE_NUMBER = re.compile(r"^\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*])")
_SKIP_LINE_CHARS = set("-|: >*")

_HIGH_RISK_FACTUAL = re.compile(
    r"(?:\b\d+(?:\.\d+)?%?\b|\b(?:shows?|showed|reports?|reported|"
    r"introduces?|introduced|proposes?|proposed|evaluates?|evaluated|"
    r"improves?|improved|reduces?|reduced|increases?|increased|"
    r"decreases?|decreased|demonstrates?|demonstrated|finds?|found|"
    r"outperforms?|outperformed|achieves?|achieved)\b)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "such",
    "that", "the", "their", "these", "this", "to", "with",
}
REQUIRED_HEADINGS = [
    "## Introduction", "## Literature Review", "## Research Gap",
    "## Proposed Novelty and Contribution", "## Limitations", "## Conclusion",
    "## References",
]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _tokens(text: str) -> set[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text.lower())
    return {_stem(token) for token in raw if len(token) >= 4 and token not in _STOPWORDS}


def _is_skippable(stripped_line: str) -> bool:
    return not stripped_line or set(stripped_line) <= _SKIP_LINE_CHARS


def find_missing_citations(markdown: str) -> list[MissingCitation]:
    """Sentences that read as factual/empirical claims but cite no source."""
    body = markdown.split("## References", 1)[0]
    section = "Preamble"
    missing: list[MissingCitation] = []
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            section = stripped.lstrip("#").strip() or section
            continue
        if _is_skippable(stripped):
            continue
        for sentence in _SENTENCE_SPLIT.split(stripped):
            sentence = sentence.strip()
            if sentence and not _CITATION.search(sentence) and _HIGH_RISK_FACTUAL.search(sentence):
                missing.append(MissingCitation(sentence=sentence, section=section))
    return missing


def build_reference_profiles(discovery, verification) -> dict[int, set[str]]:
    """Map each numbered reference (as it appears in the draft, `[N]`) to a
    token profile built from the evidence Discovery actually held for that
    paper — title, abstract, venue, and the retrieved Discussion excerpt —
    matched by DOI. References whose DOI has no Discovery match are left
    out — an honest gap, not a false claim of support (see DECISIONS.md D-013).

    The profile must cover exactly the evidence Service 2 was actually given,
    so it includes `discussion_excerpt` wherever Discovery retrieved one
    (D-039). Checking a Discussion-sourced claim against the abstract alone
    would report an unsupported claim for evidence that is genuinely present —
    the same class of false positive as D-021."""
    doi_to_paper = {
        paper.doi.lower(): paper for paper in discovery.selected_papers if paper.doi
    }
    all_checks = (
        verification.verified_references
        + verification.invalid_references
        + verification.unverifiable_references
    )
    profiles: dict[int, set[str]] = {}
    for check in all_checks:
        match = _REFERENCE_NUMBER.match(check.reference_entry.strip())
        if not match:
            continue
        paper = doi_to_paper.get((check.doi or "").lower())
        if paper is None:
            continue
        profile_text = " ".join(filter(None, [
            paper.title, paper.abstract, paper.venue, paper.discussion_excerpt,
        ]))
        profiles[int(match.group(1))] = _tokens(profile_text)
    return profiles


def find_unsupported_claims(markdown: str, profiles: dict[int, set[str]]) -> list[UnsupportedClaim]:
    """Cited sentences whose wording does not overlap enough with any of
    their cited papers' evidence profile (see build_reference_profiles).
    Only flags a claim when at least one of its citation numbers has a known
    profile to check against — an unmatched DOI is a coverage gap
    (known_limitations), not evidence of an unsupported claim.

    Sensitivity trade-off, accepted knowingly (D-039): a profile that now
    includes a retrieved Discussion excerpt holds several times more tokens
    than title+abstract alone, so this check clears more claims than it used
    to against the same 3-token threshold. That is the cost of not reporting
    every correctly Discussion-grounded claim as unsupported, which is what
    the narrower profile would have done once Service 2 gained that evidence.
    The threshold is deliberately left at 3: changing it in the same pass
    would be an unmeasured change to a check whose false-positive behaviour
    was the actual problem. Revisit with real-run data, not by guessing."""
    body = markdown.split("## References", 1)[0]
    unsupported: list[UnsupportedClaim] = []
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#") or _is_skippable(stripped):
            continue
        numbers = [int(n) for n in _CITATION.findall(stripped)]
        if not numbers or not any(number in profiles for number in numbers):
            continue
        claim_tokens = _tokens(_CITATION.sub("", stripped))
        if len(claim_tokens) < 4:
            continue
        supported = any(
            len(claim_tokens & profiles[number]) >= 3
            for number in numbers
            if number in profiles
        )
        if not supported:
            unsupported.append(UnsupportedClaim(sentence=stripped, citation_numbers=numbers))
    return unsupported


def find_missing_sections(markdown: str) -> list[str]:
    return [heading for heading in REQUIRED_HEADINGS if heading not in markdown]


def compute_scores(
    *, verification, writing, missing_citations: list[MissingCitation],
    unsupported_claims: list[UnsupportedClaim], missing_sections: list[str],
) -> QualityScores:
    citation_integrity = 5.0
    if verification.invalid_references or missing_citations:
        citation_integrity = 2.0
    elif verification.unverifiable_references:
        citation_integrity = 4.0

    claim_source_alignment = 5.0 if not unsupported_claims else 3.0

    # Prefer the writing service's own explicit completeness classification
    # (DECISIONS.md D-019) over inferring severity from warning/error text —
    # a "partial" draft is scored by how much of it actually failed, not by
    # merely whether any warning exists at all.
    total = writing.draft_metadata.total_sections
    failed = writing.draft_metadata.failed_sections
    if writing.draft_metadata.draft_status == "complete" and not writing.draft_metadata.errors:
        process_control = 5.0
    elif writing.draft_metadata.errors:
        process_control = 2.0
    elif total > 0:
        failed_fraction = failed / total
        process_control = round(max(2.0, 5.0 - failed_fraction * 3.0), 2)
    elif writing.draft_metadata.warnings:
        process_control = 4.0
    else:
        process_control = 5.0

    literature_coverage = 5.0 if not missing_sections else 3.0

    overall = round(
        (citation_integrity + claim_source_alignment + process_control + literature_coverage) / 4,
        2,
    )
    return QualityScores(
        citation_integrity=citation_integrity,
        claim_source_alignment=claim_source_alignment,
        process_control=process_control,
        literature_coverage=literature_coverage,
        overall=overall,
    )


def build_known_limitations(
    *, discovery, verification, profiles: dict[int, set[str]],
) -> list[str]:
    limitations: list[str] = []
    if discovery.limitations:
        limitations.append(discovery.limitations)
    if verification.unverifiable_references:
        limitations.append(
            f"{len(verification.unverifiable_references)} reference(s) had no DOI, or got no "
            "answer from the lookup providers, and could not be automatically verified "
            "against Crossref/OpenAlex."
        )
    all_checks = (
        verification.verified_references
        + verification.invalid_references
        + verification.unverifiable_references
    )
    unmatched = sum(
        1 for check in all_checks
        if _REFERENCE_NUMBER.match(check.reference_entry.strip())
        and int(_REFERENCE_NUMBER.match(check.reference_entry.strip()).group(1)) not in profiles
    )
    if unmatched:
        limitations.append(
            f"{unmatched} reference(s) could not be matched back to a Discovery paper record "
            "by DOI, so claims citing them could not be checked against original evidence."
        )
    return limitations


def build_quality_report(
    *, scores: QualityScores, missing_citations: list[MissingCitation],
    unsupported_claims: list[UnsupportedClaim], missing_sections: list[str],
    known_limitations: list[str],
) -> str:
    lines = [
        "# Quality Assurance Report", "",
        "## Scores",
        f"- Citation integrity: {scores.citation_integrity:.1f}/5.0",
        f"- Claim-source alignment: {scores.claim_source_alignment:.1f}/5.0",
        f"- Process control: {scores.process_control:.1f}/5.0",
        f"- Literature coverage: {scores.literature_coverage:.1f}/5.0",
        f"- **Overall: {scores.overall:.1f}/5.0**",
        "",
    ]
    if missing_citations:
        lines.append("## Missing citations")
        lines += [f"- ({item.section}) {item.sentence}" for item in missing_citations]
        lines.append("")
    if unsupported_claims:
        lines.append("## Unsupported claims")
        lines += [
            f"- {item.sentence}  _(citations: {', '.join(map(str, item.citation_numbers))})_"
            for item in unsupported_claims
        ]
        lines.append("")
    if missing_sections:
        lines.append("## Missing sections")
        lines += [f"- {heading}" for heading in missing_sections]
        lines.append("")
    if known_limitations:
        lines.append("## Known limitations")
        lines += [f"- {item}" for item in known_limitations]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def run_quality_assurance(request: QualityAssuranceRequest) -> AsyncIterator[dict]:
    """Yields progress events, then a final {"type": "result", "result":
    QualityAssuranceResult} event. Purely deterministic — no LLM calls,
    so this runs fast regardless of hardware."""
    discovery = request.discovery
    writing = request.writing
    verification = request.verification
    markdown = verification.validated_draft_markdown

    yield {"type": "status", "stage": "audit", "state": "running",
           "message": "Running independent citation and evidence audit…"}

    missing_citations = find_missing_citations(markdown)
    profiles = build_reference_profiles(discovery, verification)
    unsupported_claims = find_unsupported_claims(markdown, profiles)
    missing_sections = find_missing_sections(markdown)
    scores = compute_scores(
        verification=verification, writing=writing, missing_citations=missing_citations,
        unsupported_claims=unsupported_claims, missing_sections=missing_sections,
    )
    known_limitations = build_known_limitations(
        discovery=discovery, verification=verification, profiles=profiles,
    )
    quality_report_markdown = build_quality_report(
        scores=scores, missing_citations=missing_citations,
        unsupported_claims=unsupported_claims, missing_sections=missing_sections,
        known_limitations=known_limitations,
    )

    yield {"type": "status", "stage": "audit", "state": "done",
           "message": f"Audit complete — overall quality score {scores.overall:.1f}/5.0."}

    audit_metadata = {
        "run_id": writing.draft_metadata.run_id,
        "model_name": writing.draft_metadata.model_name,
        "writing_generated_at": writing.draft_metadata.generated_at,
        "audited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "references_checked": (
            len(verification.verified_references)
            + len(verification.invalid_references)
            + len(verification.unverifiable_references)
        ),
    }

    result = QualityAssuranceResult(
        final_draft_markdown=markdown,
        final_references=writing.reference_candidates,
        final_validation_report=verification.validation_report,
        quality_report_markdown=quality_report_markdown,
        scores=scores,
        unsupported_claims=unsupported_claims,
        missing_citations=missing_citations,
        missing_sections=missing_sections,
        known_limitations=known_limitations,
        audit_metadata=audit_metadata,
    )
    yield {"type": "result", "result": result}
