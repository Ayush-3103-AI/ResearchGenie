"""The scholarly response cache (#6), seen through the gateway's seam: the
mock transport's request count, a fake clock for TTLs, and the cache file
itself. The root conftest points RESEARCHGENIE_CACHE at a temporary file."""

import asyncio
import sqlite3
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import fulltext, pipeline  # noqa: E402
from discovery.models import Paper  # noqa: E402
from shared.utilities import response_cache, scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import Gateway, ProviderExhausted, Run  # noqa: E402

SEARCH = "https://api.openalex.org/works"
LOOKUP = "https://api.crossref.org/works/10.1000/abc"
DAY = 24 * 3600


class FakeTime:
    def __init__(self):
        self.now = 1_800_000_000.0

    def wall(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(0)


def _counting(response=None):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return response or httpx.Response(200, json={"results": []})

    return httpx.MockTransport(handler), seen


def _get(gateway, transport, url, run=None, **kwargs):
    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await gateway.get(http, url, **kwargs)

    async def main():
        return await (run or Run()).task(go())

    return asyncio.run(main())


def _gateway(fake=None):
    fake = fake or FakeTime()
    return Gateway(wall_clock=fake.wall, clock=lambda: 0.0, sleep=fake.sleep), fake


def test_a_repeated_request_is_answered_from_the_cache():
    gateway, _ = _gateway()
    transport, seen = _counting(httpx.Response(200, json={"results": [{"id": "W1"}]}))
    first = _get(gateway, transport, SEARCH, params={"search": "fasting"})
    run = Run()
    second = _get(gateway, transport, SEARCH, run=run, params={"search": "fasting"})
    assert len(seen) == 1
    assert second.json() == first.json() == {"results": [{"id": "W1"}]}
    assert run.cache_event()["hits"] == 1
    assert Run().cache_event() is None


def test_ttl_is_honored_per_request_kind():
    gateway, fake = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    _get(gateway, transport, LOOKUP)
    fake.now += 8 * DAY  # past a search's 7 days, inside a lookup's 30
    _get(gateway, transport, SEARCH, params={"search": "q"})
    _get(gateway, transport, LOOKUP)
    assert [r.url.host for r in seen] == ["api.openalex.org", "api.crossref.org", "api.openalex.org"]
    fake.now += 23 * DAY  # 31 days after the first lookup
    _get(gateway, transport, LOOKUP)
    assert len(seen) == 4


def test_cache_keys_never_contain_a_key_or_email(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "sekret-openalex-key")
    monkeypatch.setenv("CONTACT_EMAIL", "student@uni.example")
    gateway, _ = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert "sekret-openalex-key" in str(seen[0].url)  # the key did go to OpenAlex

    rows = sqlite3.connect(response_cache.cache_path()).execute("SELECT key FROM responses").fetchall()
    assert rows
    for (key,) in rows:
        assert "sekret" not in key and "student" not in key and "uni.example" not in key

    # Without the key it is the same request, so the same entry answers it.
    monkeypatch.delenv("OPENALEX_API_KEY")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 1


def test_only_successful_responses_are_stored():
    gateway, _ = _gateway()
    transport, seen = _counting(httpx.Response(404))
    _get(gateway, transport, LOOKUP)
    _get(gateway, transport, LOOKUP)
    assert len(seen) == 2


def test_no_cache_skips_reads_and_clear_empties_the_cache(monkeypatch):
    gateway, _ = _gateway()
    transport, seen = _counting()
    _get(gateway, transport, SEARCH, params={"search": "q"})
    monkeypatch.setenv("RESEARCHGENIE_NO_CACHE", "1")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 2
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 2
    assert response_cache.clear() == 1
    _get(gateway, transport, SEARCH, params={"search": "q"})
    assert len(seen) == 3


def test_exhaustion_and_balance_survive_a_process_restart():
    fake = FakeTime()
    first, _ = _gateway(fake)
    rate = {"X-RateLimit-Remaining": "40", "X-RateLimit-Limit": "1000"}
    transport, _ = _counting(httpx.Response(200, headers=rate, json={"results": []}))
    _get(first, transport, SEARCH, params={"search": "a"})
    gone = httpx.Response(429, text='{"error": "not_enough_credits"}')
    transport, _ = _counting(gone)
    with pytest.raises(ProviderExhausted):
        _get(first, transport, SEARCH, params={"search": "b"})

    restarted, _ = _gateway(fake)  # a new process: a fresh Gateway, the same cache file
    transport, seen = _counting()
    with pytest.raises(ProviderExhausted):
        _get(restarted, transport, SEARCH, params={"search": "c"})
    assert seen == []
    assert restarted.provider_status()[0]["health"] == "degraded"
    balance = response_cache.balance("api.openalex.org")
    assert balance["x-ratelimit-remaining"] == "40" and balance["seen_at"] == fake.now
    assert restarted.provider_status()[0]["budget_remaining"] == "40"  # known before a run starts
    # The cached search still answers, budget or not.
    assert _get(restarted, transport, SEARCH, params={"search": "a"}).status_code == 200


def test_extracted_full_text_is_cached_with_no_expiry(monkeypatch):
    discussion = "Discussion. " + "Future work should test this in older adults. " * 20
    xml = (f"<article><body><sec sec-type='discussion'><title>Discussion</title>"
           f"<p>{discussion}</p></sec></body></article>")
    transport, seen = _counting(httpx.Response(200, text=xml))

    async def fetch():
        paper = Paper(title="T", source="OpenAlex", pmcid="PMC1")
        async with httpx.AsyncClient(transport=transport) as http:
            run = Run()
            ok = await run.task(fulltext.fetch_future_sections(http, paper))
        return ok, paper.future_text, run.cache_hits

    first = asyncio.run(fetch())
    second = asyncio.run(fetch())
    assert first[0] and second[0] and second[1] == first[1]
    assert len(seen) == 1 and second[2] == 1


def _work(i: int) -> dict:
    return {"id": f"https://openalex.org/W{i}",
            "display_name": f"Intermittent fasting and cognitive performance, study {i}",
            "abstract_inverted_index": {w: [n] for n, w in enumerate(
                ["Intermittent", "fasting", "improved", "cognitive", "performance"])},
            "publication_year": 2022, "cited_by_count": 10,
            "doi": f"https://doi.org/10.1000/w.{i}",
            "authorships": [{"author": {"display_name": "A. Author"}}]}


def test_a_second_identical_discovery_run_makes_no_network_call(monkeypatch):
    async def fake_queries(question, model, feedback=None):
        return {"keyword": "fasting cognition", "pubmed": "x", "arxiv": "x", "interpretation": "i"}

    async def fake_analysis(question, corpus, candidates, verification_papers, model):
        yield {"type": "report", "report": {}}

    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)
    monkeypatch.setattr(pipeline, "analyze_gaps", fake_analysis)
    transport, seen = _counting(httpx.Response(200, json={"results": [_work(i) for i in range(8)]}))

    async def discover():
        return [e async for e in pipeline.run_pipeline(
            "Effects of intermittent fasting on cognitive performance", ["openalex"],
            per_source=10, corpus_size=6, fulltext_enabled=False, transport=transport)]

    first = asyncio.run(discover())
    requests_after_first = len(seen)
    second = asyncio.run(discover())
    assert requests_after_first == 1 and len(seen) == 1
    assert not [e for e in first if e["type"] == "cache_hit"]
    [hit] = [e for e in second if e["type"] == "cache_hit"]
    assert hit["hits"] == 1 and "cache" in hit["message"]
    corpus = lambda events: [e for e in events if e["type"] == "corpus"][0]["papers"]
    assert corpus(second) == corpus(first)
