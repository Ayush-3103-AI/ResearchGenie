"""Re-plan the search when the corpus is too thin (#15).

Round 1 finds fewer on-topic papers than the corpus size, so Discovery
regenerates its queries once (fed the rejected titles and reasons), searches
again through the gateway, merges, dedupes and screens again. Never more
than two rounds; never on a full corpus; never against an exhausted provider.
"""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402

QUESTION = "Effects of intermittent fasting on cognitive performance"


def _work(i: int, on_topic: bool = True) -> dict:
    title = (f"Intermittent fasting and cognitive performance in adults, study {i}" if on_topic
             else f"Soil erosion in alpine river basins, survey {i}")
    words = ["Intermittent", "fasting", "improved", "cognitive", "performance"] if on_topic \
        else ["Soil", "erosion", "rivers", "alpine", "basins"]
    return {
        "id": f"https://openalex.org/W{i}", "display_name": title,
        "abstract_inverted_index": {w: [n] for n, w in enumerate(words)},
        "publication_year": 2022, "cited_by_count": 10,
        "doi": f"https://doi.org/10.1000/w.{i}",
        "authorships": [{"author": {"display_name": "A. Author"}}],
    }


def _queries(keyword: str) -> dict:
    return {"keyword": keyword, "pubmed": keyword, "arxiv": keyword, "interpretation": "i"}


def _run(monkeypatch, handler, corpus_size=5):
    calls = []

    async def fake_queries(question, model, feedback=None):
        calls.append(feedback)
        return _queries("round two query" if feedback else "round one query")

    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ["openalex"], per_source=10,
                                       corpus_size=corpus_size, fulltext_enabled=False,
                                       transport=httpx.MockTransport(handler))
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "corpus":
                    break
        finally:
            await stream.aclose()
        return events

    return asyncio.run(go()), calls


def _two_rounds(request):
    """Round 1: two on-topic papers and an off-topic one. Round 2: one of
    the same papers again (a duplicate) plus three new on-topic ones."""
    if request.url.params["search"] == "round one query":
        works = [_work(1), _work(2), _work(90, on_topic=False)]
    else:
        works = [_work(2), _work(3), _work(4), _work(5)]
    return httpx.Response(200, json={"results": works})


def test_a_thin_corpus_re_plans_once_and_merges_the_second_round(monkeypatch):
    events, calls = _run(monkeypatch, _two_rounds)

    [replan] = [e for e in events if e["type"] == "replan"]
    assert replan["round"] == 2 and replan["reason"]
    assert replan["accepted_before"] == 2 and replan["accepted_after"] == 5
    assert "replan" not in replan["message"].lower() and replan["message"]

    # Fed the rejected titles and the reason they were rejected.
    assert calls[0] is None and "Soil erosion" in calls[1]
    corpus = events[-1]["papers"]
    assert len(corpus) == 5 and len({p["doi"] for p in corpus}) == 5  # study 2 deduped


def test_never_more_than_two_rounds(monkeypatch):
    def always_thin(request):
        return httpx.Response(200, json={"results": [_work(1)]})

    events, calls = _run(monkeypatch, always_thin)
    assert len(calls) == 2
    assert len([e for e in events if e["type"] == "replan"]) == 1


def test_a_full_corpus_never_re_plans(monkeypatch):
    def plenty(request):
        return httpx.Response(200, json={"results": [_work(i) for i in range(8)]})

    events, calls = _run(monkeypatch, plenty)
    assert calls == [None]
    assert not [e for e in events if e["type"] == "replan"]


def test_no_re_plan_against_an_exhausted_provider(monkeypatch, fresh_scholarly_gateway):
    def round_one_only(request):
        assert request.url.params["search"] == "round one query", "round 2 hit an exhausted provider"
        return httpx.Response(200, json={"results": [_work(1)]})

    real_search = pipeline.SOURCES["openalex"][1]

    async def search_then_budget_gone(http, queries, limit):
        papers = await real_search(http, queries, limit)
        # As if a parallel call had just got OpenAlex's not_enough_credits 429.
        fresh_scholarly_gateway._exhausted_until["api.openalex.org"] = 4e9
        return papers

    monkeypatch.setitem(pipeline.SOURCES, "openalex", ("OpenAlex", search_then_budget_gone))
    events, calls = _run(monkeypatch, round_one_only)
    assert calls == [None]
    assert not [e for e in events if e["type"] == "replan"]
    assert any("second search" in (e.get("message") or "") for e in events)
