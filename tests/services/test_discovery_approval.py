"""The paper approval step (#17): the run pauses after search and screening,
before full text, analysis and Writing; dropped papers are backfilled from
the ranked candidates, so the corpus keeps its size."""

import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402

QUESTION = "Effects of intermittent fasting on cognitive performance"
QUERIES = {"keyword": "intermittent fasting cognitive performance", "pubmed": "fasting",
           "arxiv": "fasting", "interpretation": "How intermittent fasting affects cognition."}


def _work(i: int) -> dict:
    return {
        "id": f"https://openalex.org/W{i}",
        "display_name": f"Intermittent fasting and cognitive performance in adults, study {i}",
        "abstract_inverted_index": {"Intermittent": [0], "fasting": [1], "improved": [2],
                                    "cognitive": [3], "performance": [4]},
        "publication_year": 2022, "cited_by_count": 100 - i,
        "doi": f"https://doi.org/10.1000/if.{i}",
        "authorships": [{"author": {"display_name": "A. Author"}}],
        "primary_location": {"source": {"display_name": "Nutrients"}},
    }


def _transport():
    def handler(request):
        if request.url.host == "api.openalex.org":
            return httpx.Response(200, json={"results": [_work(i) for i in range(8)]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


async def _fake_queries(question, model):
    return dict(QUERIES)


def _until_corpus(monkeypatch, approve=None):
    monkeypatch.setattr(pipeline, "generate_queries", _fake_queries)

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ["openalex"], per_source=10, corpus_size=5,
                                       fulltext_enabled=False, transport=_transport(),
                                       approve=approve)
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "corpus":
                    break
        finally:
            await stream.aclose()
        return events

    return asyncio.run(go())


def test_dropped_papers_are_replaced_from_the_ranked_candidates(monkeypatch):
    shown = {}

    async def approve(papers, reserve):
        shown["papers"], shown["reserve"] = papers, reserve
        return {"approved_by": "cli", "drop": [2, 5]}

    events = _until_corpus(monkeypatch, approve)
    assert {"title", "relevance_score", "year", "venue"} <= set(shown["papers"][0])
    assert shown["reserve"] == 3
    dropped = {shown["papers"][1]["title"], shown["papers"][4]["title"]}

    corpus = events[-1]["papers"]
    assert len(corpus) == 5 and not dropped & {p["title"] for p in corpus}
    assert [p["id"] for p in corpus] == [1, 2, 3, 4, 5]

    [approval] = [e for e in events if e["type"] == "approval"]
    assert approval["approved_by"] == "cli" and approval["approved_at"]
    assert {d["title"] for d in approval["dropped"]} == dropped
    assert len(approval["replacements"]) == 2
    types = [e["type"] for e in events]
    assert types.index("approval_request") < types.index("approval")


def test_without_an_approver_the_corpus_is_approved_automatically(monkeypatch):
    events = _until_corpus(monkeypatch)
    assert "approval_request" not in [e["type"] for e in events]
    [approval] = [e for e in events if e["type"] == "approval"]
    assert approval["approved_by"] == "auto" and approval["dropped"] == []
    assert len(events[-1]["papers"]) == 5


def test_the_final_draft_never_cites_a_dropped_paper(tmp_path, monkeypatch):
    """Real Discovery service and orchestrator; fake network, model and later stages."""
    from orchestrator import pipeline as orchestrator_pipeline
    from shared.contracts.pipeline_contract import ResearchRequest
    sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
    from test_pipeline import _fake_qa_result, _fake_verification_result, _fake_writing_result

    service = orchestrator_pipeline.discovery_service
    real = service.run_pipeline

    async def fake_analysis(question, papers, candidates, verification_papers, model_name):
        yield {"type": "report", "report": {
            "field_overview": "o", "themes": [], "limitations": "l", "gaps": [],
            "author_flagged_gaps": [], "confidence_notes": "c",
            "novelty_analysis": {"novelty_summary": "s", "confidence": "low", "caveats": "c"}}}

    monkeypatch.setattr(pipeline, "generate_queries", _fake_queries)
    monkeypatch.setattr(pipeline, "analyze_gaps", fake_analysis)
    monkeypatch.setattr(service, "run_pipeline", lambda *a, **k: real(
        *a, **{**k, "transport": _transport(), "fulltext_enabled": False}))
    seen_by_writing = []

    async def writing(request, output_root=None):
        seen_by_writing.extend(p.title for p in request.discovery.selected_papers)
        draft = " ".join(f"{p.title} [{p.id}]." for p in request.discovery.selected_papers)
        yield {"type": "result", "result": _fake_writing_result().model_copy(
            update={"research_draft_markdown": draft})}

    async def verification(request):
        yield {"type": "result", "result": _fake_verification_result()}

    async def qa(request):
        yield {"type": "result", "result": _fake_qa_result().model_copy(
            update={"final_draft_markdown": request.writing.research_draft_markdown})}

    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", qa)
    dropped = []

    async def approve(papers, reserve):
        dropped.append(papers[0]["title"])
        return {"approved_by": "web", "drop": [1]}

    async def run():
        return [e async for e in orchestrator_pipeline.run_pipeline(
            ResearchRequest(research_question=QUESTION, corpus_size=6), output_root=tmp_path,
            approve=approve)]

    result = asyncio.run(run())[-1]["result"]
    assert len(seen_by_writing) == 6 and dropped[0] not in seen_by_writing
    draft = (Path(result.run_directory) / "final" / "draft.md").read_text(encoding="utf-8")
    assert dropped[0] not in draft
    logged = (Path(result.run_directory) / "events.jsonl").read_text(encoding="utf-8")
    assert '"approved_by": "web"' in logged and dropped[0] in logged
