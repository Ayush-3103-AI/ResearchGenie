"""Discovery reads the research library (#19): a stored excerpt replaces a
full-text fetch, past papers matching the question join the candidates and
still face the relevance gate, and memory off never reads the library."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import fulltext, pipeline  # noqa: E402
from shared.utilities import library  # noqa: E402
from tests.services.test_discovery_replan import _work  # noqa: E402

QUESTION = "A study of intermittent fasting and cognitive performance"
EXCERPT = "Discussion. Future work should test intermittent fasting in older adults."
PAST = {"title": "Intermittent fasting and cognitive performance in older adults",
        "abstract": "Intermittent fasting improved cognitive performance in older adults.",
        "year": 2019, "venue": "J. Aging", "doi": "10.9/past", "authors": ["B. Past"],
        "citations": 12, "discussion_excerpt": EXCERPT}
# Found by the library's full-text search ("study") but sharing no topic term.
OFF_TOPIC = {"title": "A study of soil erosion in alpine river basins",
             "abstract": "Erosion rates were measured across alpine catchments.",
             "year": 2018, "doi": "10.9/soil"}


def _store(tmp_path, papers):
    folder = tmp_path / "past-run"
    (folder / "01_discovery").mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps(
        {"run_id": "past", "research_question": "Intermittent fasting and memory?"}), encoding="utf-8")
    (folder / "01_discovery" / "result.json").write_text(
        json.dumps({"selected_papers": papers}), encoding="utf-8")
    assert library.import_run(folder)


def _discover(monkeypatch, works, corpus_size, memory=True):
    """Events up to the end of the full-text stage, and the DOIs fetched."""
    async def fake_queries(question, model, feedback=None):
        return {"keyword": "intermittent fasting cognition", "pubmed": "x", "arxiv": "x",
                "interpretation": "i"}

    fetched = []

    async def fake_fetch(http, paper):
        fetched.append(paper.doi)
        return False

    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)
    monkeypatch.setattr(fulltext, "fetch_future_sections", fake_fetch)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"results": works}))

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ["openalex"], per_source=10,
                                       corpus_size=corpus_size, transport=transport, memory=memory)
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "fulltext_done":
                    break
        finally:
            await stream.aclose()
        return events

    return asyncio.run(go()), fetched


def _one(events, kind):
    [event] = [e for e in events if e["type"] == kind]
    return event


def test_a_stored_excerpt_is_reused_not_fetched(tmp_path, monkeypatch):
    _store(tmp_path, [{**PAST, "doi": "10.1000/w.1"}])
    events, fetched = _discover(monkeypatch, [_work(i) for i in range(1, 6)], corpus_size=5)

    assert "10.1000/w.1" not in fetched and len(fetched) == 4
    corpus = _one(events, "corpus")["papers"]
    reused_id = next(p["id"] for p in corpus if p["doi"] == "10.1000/w.1")
    assert _one(events, "fulltext_done")["excerpts"] == {reused_id: EXCERPT}
    hit = _one(events, "memory_hit")
    assert (hit["count"], hit["reused_excerpts"], hit["seeded_papers"]) == (1, 1, 0)


def test_seeded_past_papers_go_through_the_relevance_gate(tmp_path, monkeypatch):
    _store(tmp_path, [PAST, OFF_TOPIC])
    events, fetched = _discover(monkeypatch, [_work(i) for i in range(1, 6)], corpus_size=6)

    rejected = _one(events, "relevance_filter")["rejected_examples"]
    assert [r["title"] for r in rejected] == [OFF_TOPIC["title"]]
    corpus = _one(events, "corpus")["papers"]
    assert OFF_TOPIC["doi"] not in {p["doi"] for p in corpus}
    [seeded] = [p for p in corpus if p["doi"] == PAST["doi"]]
    assert seeded["source"] == pipeline.LIBRARY_SOURCE and seeded["authors"] == ["B. Past"]
    assert PAST["doi"] not in fetched  # its stored excerpt came with it
    hit = _one(events, "memory_hit")
    assert (hit["count"], hit["reused_excerpts"], hit["seeded_papers"]) == (2, 1, 1)


def test_memory_off_never_reads_the_library_and_behaves_as_before(tmp_path, monkeypatch):
    works = [_work(i) for i in range(1, 6)]
    baseline, baseline_fetched = _discover(monkeypatch, works, corpus_size=5)  # empty library
    _store(tmp_path, [{**PAST, "doi": "10.1000/w.1"}, PAST])

    reads = []
    monkeypatch.setattr(library, "connect", lambda *a, **k: reads.append(a) or 1 / 0)
    events, fetched = _discover(monkeypatch, works, corpus_size=5, memory=False)

    assert reads == []
    assert not [e for e in events if e["type"] in ("memory_hit", "source_error")]
    assert sorted(fetched) == sorted(baseline_fetched) and "10.1000/w.1" in fetched
    assert _one(events, "corpus") == _one(baseline, "corpus")
    assert [e["type"] for e in events] == [e["type"] for e in baseline]


def test_an_unreadable_library_never_fails_discovery(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(library, "connect", broken)

    events, fetched = _discover(monkeypatch, [_work(i) for i in range(1, 6)], corpus_size=5)
    assert len(fetched) == 5 and _one(events, "fulltext_done")
    warnings = [e for e in events if e["type"] == "source_error"]
    assert len(warnings) == 2 and all(e["source"] == pipeline.LIBRARY_SOURCE for e in warnings)
