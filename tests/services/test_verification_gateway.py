"""Citation Verification through the scholarly gateway (#4).

Crossref and OpenAlex DOI lookups share Discovery's pacing, retry and
breaker. The conftest gateway never really sleeps, so retries are instant.
"""

import asyncio
import importlib.util
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.contracts.verification_contract import VerificationRequest  # noqa: E402
from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verification_gateway_service", ROOT / "services" / "verification" / "service.py")
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)

REAL = '[1] A. Author, "Real Paper," *Journal*, 2020, doi: 10.1111/real.'


def _answer(request, known):
    """A Crossref or OpenAlex DOI-filter answer (#7) listing the asked DOIs in `known`."""
    asked = [d.removeprefix("doi:") for d in re.split(r"[,|]", request.url.params["filter"])]
    hits = [d for d in asked if d in known]
    if request.url.host == "api.crossref.org":
        return httpx.Response(200, json={"message": {"items": [{"DOI": d.upper()} for d in hits]}})
    return httpx.Response(200, json={"results": [{"doi": f"https://doi.org/{d}"} for d in hits]})


def _verify(references, handler):
    writing = WritingResult(
        research_outline="# Q", research_draft_markdown="text", reference_candidates=references,
        run_directory="/tmp/run",
        draft_metadata=DraftMetadata(model_name="m", run_id="r", generated_at="2026-10-18T00:00:00Z",
                                     papers_cited=len(references)),
    )

    async def run():
        events, result = [], None
        async for event in service.run_verification(
                VerificationRequest(writing=writing), transport=httpx.MockTransport(handler)):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result

    return asyncio.run(run())


def test_a_real_doi_throttled_once_by_crossref_is_verified_not_unverifiable():
    crossref_calls = []

    def handler(request):
        if request.url.host == "api.crossref.org":
            crossref_calls.append(request)
            return httpx.Response(429) if len(crossref_calls) == 1 else _answer(request, {"10.1111/real"})
        return _answer(request, set())

    events, result = _verify([REAL], handler)

    assert [c.doi for c in result.verified_references] == ["10.1111/real"]
    assert result.verified_references[0].verification_source == "Crossref"
    assert not result.unverifiable_references and not result.invalid_references
    assert len(crossref_calls) == 2
    assert any(e["type"] == "rate_limited" for e in events)


def test_an_exhausted_provider_is_reported_as_unverifiable_not_as_a_missing_doi():
    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(429, text='{"error": "not_enough_credits"}')
        return httpx.Response(503)

    events, result = _verify([REAL], handler)

    assert not result.invalid_references and not result.verified_references
    [check] = result.unverifiable_references
    assert check.doi == "10.1111/real"
    assert check.issue.startswith("unverifiable (provider exhausted)")
    assert "does not resolve" not in check.issue and "fabricated" not in check.issue
    assert any(e["type"] == "provider_exhausted" for e in events)
    assert "provider exhausted" in result.validation_report


def test_a_doi_both_providers_answer_without_is_still_invalid():
    _, result = _verify([REAL], lambda request: _answer(request, set()))
    assert [c.doi for c in result.invalid_references] == ["10.1111/real"]


def test_n_dois_take_one_filter_request_per_100_not_one_each():
    dois = [f"10.1000/w.{i}" for i in range(250)]
    references = [f'[{i}] A. Author, "Paper {i}," 2020, doi: {doi}.' for i, doi in enumerate(dois)]
    seen = []

    def handler(request):
        seen.append(request)
        return _answer(request, set(dois[:200]))

    _, result = _verify(references + references[:5], handler)  # a repeated DOI is asked once

    hosts = [r.url.host for r in seen]
    assert hosts.count("api.crossref.org") == 3  # ceil(250 / 100)
    assert hosts.count("api.openalex.org") == 1  # only the 50 Crossref did not confirm
    assert len(result.verified_references) == 205 and len(result.invalid_references) == 50


def test_crossref_gets_the_polite_pool_mailto_when_an_email_is_configured(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "student@example.edu")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200)

    _verify([REAL], handler)
    assert seen[0].url.host == "api.crossref.org"
    assert seen[0].url.params["mailto"] == "student@example.edu"
