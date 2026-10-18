"""Regression: '3 of 4 sources failed at once' (PAPER_OUTPUT_FINAL_DIAGNOSTIC.md
finding #3, DECISIONS.md D-028), replayed through the scholarly gateway.

A real diagnostic run saw Semantic Scholar, PubMed and arXiv all fail on the
same query while OpenAlex answered. Replayed here with a mock transport —
S2 keeps answering 429, NCBI and arXiv keep answering 503 — Discovery must
still finish its search stage with a corpus from OpenAlex alone, and say why
the other three were dropped: rate_limited events while it waited,
provider_exhausted when S2's breaker tripped, and one source_error per
failed source. The conftest's gateway never really sleeps.
"""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402

QUESTION = "Effects of intermittent fasting on cognitive performance"
QUERIES = {"keyword": "intermittent fasting cognitive performance",
           "pubmed": "intermittent fasting AND cognition",
           "arxiv": "intermittent fasting cognition",
           "interpretation": "How intermittent fasting affects cognition."}


def _openalex_work(i: int) -> dict:
    return {
        "id": f"https://openalex.org/W{i}",
        "display_name": f"Intermittent fasting and cognitive performance in adults, study {i}",
        "abstract_inverted_index": {"Intermittent": [0], "fasting": [1], "improved": [2],
                                    "cognitive": [3], "performance": [4]},
        "publication_year": 2022,
        "cited_by_count": 10 + i,
        "doi": f"https://doi.org/10.1000/if.{i}",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"landing_page_url": f"https://example.org/{i}"},
    }


def _handler(request: httpx.Request) -> httpx.Response:
    host = request.url.host
    if host == "api.openalex.org":
        return httpx.Response(200, json={"results": [_openalex_work(i) for i in range(3)]})
    if host == "api.semanticscholar.org":
        return httpx.Response(429, json={"message": "Too Many Requests"})
    return httpx.Response(503)  # NCBI and arXiv


def _run_until_corpus(monkeypatch) -> tuple[list[dict], list[str]]:
    async def fake_queries(question, model, feedback=None):
        return dict(QUERIES)

    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return _handler(request)

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ["s2", "openalex", "pubmed", "arxiv"],
                                       per_source=10, corpus_size=5, fulltext_enabled=False,
                                       transport=httpx.MockTransport(handler))
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "corpus":
                    break  # the search stage is done; the rest needs a model
        finally:
            await stream.aclose()
        return events

    return asyncio.run(go()), hosts


def test_three_failed_sources_still_leave_a_corpus_from_the_fourth(monkeypatch):
    events, _ = _run_until_corpus(monkeypatch)
    corpus = events[-1]
    assert corpus["type"] == "corpus"
    assert corpus["analyzed"] == 3
    assert {p["source"] for p in corpus["papers"]} == {"OpenAlex"}

    results = {e["source"]: e["count"] for e in events if e["type"] == "source_result"}
    assert results == {"OpenAlex": 3}
    failed = {e["source"] for e in events if e["type"] == "source_error"}
    assert failed == {"Semantic Scholar", "PubMed", "arXiv"}


def test_the_failures_are_explained_with_rate_limit_events(monkeypatch):
    events, hosts = _run_until_corpus(monkeypatch)

    waited = {e["host"] for e in events if e["type"] == "rate_limited"}
    assert waited == {"api.semanticscholar.org", "eutils.ncbi.nlm.nih.gov", "export.arxiv.org"}
    assert all(e["wait_seconds"] > 0 and e["message"] for e in events
               if e["type"] == "rate_limited")

    # S2's keyless breaker trips on the first 429 burst: two requests, not four.
    [exhausted] = [e for e in events if e["type"] == "provider_exhausted"]
    assert exhausted["host"] == "api.semanticscholar.org"
    assert "keyless" in exhausted["remedy"]
    assert hosts.count("api.semanticscholar.org") == 2

    # The why comes before the source is reported as dropped.
    types = [(e["type"], e.get("source") or e.get("host")) for e in events]
    assert types.index(("provider_exhausted", "api.semanticscholar.org")) < \
        types.index(("source_error", "Semantic Scholar"))


def test_gateway_events_reach_the_standard_progress_shape(monkeypatch):
    """The CLI progress view and the web event log render any standard
    progress event by its message and status; host, wait and remedy ride
    along in details."""
    sys.path.insert(0, str(ROOT))
    from orchestrator.pipeline import _adapt_inner_event

    events, _ = _run_until_corpus(monkeypatch)
    rate_limited = next(e for e in events if e["type"] == "rate_limited")
    exhausted = next(e for e in events if e["type"] == "provider_exhausted")

    waiting = _adapt_inner_event("run1", "discovery", "research_discovery", rate_limited)
    assert waiting["type"] == "progress" and waiting["status"] == "running"
    assert waiting["message"] == rate_limited["message"]
    assert waiting["details"] == {"raw_type": "rate_limited", "host": rate_limited["host"],
                                  "wait_seconds": rate_limited["wait_seconds"]}

    skipped = _adapt_inner_event("run1", "discovery", "research_discovery", exhausted)
    assert skipped["status"] == "warning"
    assert skipped["details"] == {"raw_type": "provider_exhausted",
                                  "host": "api.semanticscholar.org",
                                  "remedy": exhausted["remedy"]}
