"""Verification-candidate searches within the budget check's allowance (#7):
one search per distinct question, OpenAlex only up to the allowed count,
Crossref (no credits) for the rest."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402


def test_verification_searches_are_deduplicated_and_capped_by_the_budget():
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [
                {"display_name": "Fasting and memory", "doi": "https://doi.org/10.5555/fm"}]})
        return httpx.Response(200, json={"message": {"items": [
            {"title": ["Sleep and memory"], "DOI": "10.5555/SM", "issued": {"date-parts": [[2021]]},
             "author": [{"given": "A.", "family": "Author"}]}]}})

    candidates = [{"question": "Does fasting help memory?"}, {"question": "does fasting help memory"},
                  {"question": "Does sleep help memory?"}, {"question": "Does diet help mood?"}]
    papers, warnings = asyncio.run(pipeline._verify_candidates(
        candidates, [], httpx.MockTransport(handler), openalex_searches=1))

    hosts = [r.url.host for r in seen]
    assert hosts.count("api.openalex.org") == 1  # what the budget allowed
    assert hosts.count("api.crossref.org") == 2  # the other distinct questions, at no credit cost
    assert not warnings
    assert candidates[0]["evidence_paper_ids"] == candidates[1]["evidence_paper_ids"] != []
    assert {p.doi for p in papers} == {"10.5555/fm", "10.5555/sm"}
    assert next(p for p in papers if p.source == "Crossref").year == 2021
