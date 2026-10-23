"""The research library (shared/utilities/library.py), its pipeline hooks
and its CLI/API surfaces, against a temporary library (tests/conftest.py)
and fixture run folders."""

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import api as orchestrator_api  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import cli  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.utilities import library  # noqa: E402
from tests.orchestrator import test_pipeline as fakes  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_services(monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", fakes._fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", fakes._fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", fakes._fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", fakes._fake_run_qa)


def _run_folder(root: Path, run_id: str, question: str, papers: list[dict],
                quality: float = 4.2, discovery: bool = True) -> Path:
    folder = root / f"{run_id}-folder"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({
        "run_id": run_id, "research_question": question, "overall_quality_score": quality,
        "draft_status": "complete",
    }), encoding="utf-8")
    (folder / "draft.md").write_text("past draft", encoding="utf-8")
    if discovery:
        (folder / "01_discovery").mkdir()
        (folder / "01_discovery" / "result.json").write_text(
            json.dumps({"selected_papers": papers}), encoding="utf-8")
    return folder


_PAPER = {"title": "Fasting and memory in adults", "abstract": "Intermittent fasting improved recall.",
          "year": 2021, "venue": "J. Nutr.", "doi": "https://doi.org/10.1/ABC",
          "discussion_excerpt": "Future work should test older adults."}


def _counts() -> dict[str, int]:
    conn = library.connect()
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("papers", "runs", "run_papers", "papers_fts", "runs_fts")}
    finally:
        conn.close()


def test_library_path_is_the_temporary_one(tmp_path):
    assert library.library_path() == tmp_path / "library.sqlite"


def test_write_then_search_round_trip(tmp_path):
    folder = _run_folder(tmp_path, "run1", "What are the effects of intermittent fasting on cognition?",
                         [_PAPER, {"title": "No DOI paper"}])
    assert library.import_run(folder)

    runs = library.search_runs("Does intermittent fasting help memory?")
    assert [r["run_id"] for r in runs] == ["run1"]
    assert runs[0]["overall_quality"] == 4.2 and runs[0]["run_dir"] == str(folder.resolve())

    papers = library.search_papers("fasting recall")
    assert [p["doi"] for p in papers] == ["10.1/abc"]  # normalised; the DOI-less paper is skipped
    assert papers[0]["fulltext_status"] == "excerpt"
    assert library.search_runs("quantum chromodynamics") == []


def test_import_is_idempotent(tmp_path):
    _run_folder(tmp_path, "run1", "Intermittent fasting and cognition?", [_PAPER])
    _run_folder(tmp_path, "run2", "Remote work and productivity?", [], discovery=False)
    (tmp_path / "advisor_only" / "00_trend_advisor").mkdir(parents=True)  # not a finished run

    assert library.import_runs([tmp_path]) == 2
    first = _counts()
    assert library.import_runs([tmp_path]) == 2
    assert _counts() == first == {"papers": 1, "runs": 2, "run_papers": 1, "papers_fts": 1, "runs_fts": 2}


def test_a_later_run_without_an_excerpt_keeps_the_stored_one(tmp_path):
    library.import_run(_run_folder(tmp_path, "run1", "Fasting?", [_PAPER]))
    library.import_run(_run_folder(tmp_path, "run2", "Fasting again?",
                                   [{**_PAPER, "discussion_excerpt": None, "abstract": None}]))
    paper = library.search_papers("fasting")[0]
    assert paper["discussion_excerpt"] == _PAPER["discussion_excerpt"]
    assert paper["abstract"] == _PAPER["abstract"]


def test_stored_excerpts_by_doi_and_an_older_library_file_gains_the_new_columns(tmp_path):
    old = sqlite3.connect(library.library_path())  # the schema before #19: no authors, citations
    old.execute("CREATE TABLE papers (doi TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, "
                "venue TEXT, discussion_excerpt TEXT, fulltext_status TEXT, first_seen TEXT)")
    old.execute("INSERT INTO papers VALUES ('10.2/old', 'Old', NULL, 2010, NULL, 'Old excerpt.', "
                "'excerpt', '2020-01-01')")
    old.commit()
    old.close()

    library.import_run(_run_folder(tmp_path, "run1", "Fasting?",
                                   [{**_PAPER, "authors": ["A. One"], "citations": 7}]))
    assert library.stored_excerpts(["10.2/OLD", "https://doi.org/10.1/abc", "10.3/none"]) == {
        "10.2/OLD": "Old excerpt.", "https://doi.org/10.1/abc": _PAPER["discussion_excerpt"]}
    paper = library.search_papers("fasting")[0]
    assert json.loads(paper["authors"]) == ["A. One"] and paper["citations"] == 7


def test_discovery_memory_and_time_reach_metadata(tmp_path, monkeypatch):
    """A later script compares two runs' Discovery from metadata.json (#19)."""
    seen = []

    async def discovery(request):
        seen.append(request.memory)
        yield {"type": "memory_hit", "count": 3, "reused_excerpts": 2, "seeded_papers": 1,
               "message": "Research library: reused 2 stored excerpt(s)."}
        yield {"type": "result", "result": fakes._fake_discovery_result()}
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", discovery)

    events = _run(ResearchRequest(research_question="Remote work?", corpus_size=6), tmp_path)
    run_directory = Path(events[-1]["result"].run_directory)
    logged = [json.loads(line) for line in
              (run_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    [hit] = [e for e in logged if e.get("status") == "memory_hit"]
    assert hit["stage"] == "discovery" and hit["details"] == {
        "raw_type": "memory_hit", "count": 3, "reused_excerpts": 2, "seeded_papers": 1}
    metadata = json.loads((run_directory / "metadata.json").read_text())
    assert metadata["memory"] == {"enabled": True, "count": 3, "reused_excerpts": 2, "seeded_papers": 1}
    assert isinstance(metadata["timings"]["discovery_seconds"], float)

    _run(ResearchRequest(research_question="Remote work?", corpus_size=6, memory=False), tmp_path)
    assert seen == [True, False]


def test_query_syntax_in_the_question_is_never_passed_to_match():
    assert library.fts_query('What "is" NOT fasting-related (AND) OR*?') == '"not" OR "fasting" OR "related"'
    assert library.search_runs('"unbalanced quote NEAR( -x *') == []
    assert library.fts_query("What is the?") == ""


def _run(request, output_root):
    async def go():
        return [e async for e in orchestrator_pipeline.run_pipeline(request, output_root=output_root)]
    return asyncio.run(go())


def test_finished_run_is_written_and_related_runs_are_surfaced_before_discovery(tmp_path):
    first = _run(ResearchRequest(research_question="What is the impact of remote work on productivity?",
                                 corpus_size=6), tmp_path / "out")
    first_id = first[-1]["result"].run_id
    assert library.get_run(first_id)["overall_quality"] == 5.0
    assert [p["doi"] for p in library.search_papers("Paper One")] == ["10.1/x"]
    assert not any(e.get("details", {}).get("raw_type") == "related_runs" for e in first)

    events = _run(ResearchRequest(research_question="Remote work and employee productivity?",
                                  corpus_size=6), tmp_path / "out")
    related = [e for e in events if e.get("details", {}).get("raw_type") == "related_runs"]
    assert len(related) == 1
    event = related[0]
    assert event["type"] == "progress" and event["stage"] == "discovery"
    assert [r["run_id"] for r in event["details"]["runs"]] == [first_id]
    assert set(event["details"]["runs"][0]) >= {"question", "created", "overall_quality", "run_dir"}
    assert "QA 5.0" in event["message"]
    # Surfaced before Discovery's own events.
    assert events.index(event) < next(i for i, e in enumerate(events) if e.get("message") == "searching")


def test_a_library_failure_never_fails_the_run(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(library, "connect", broken)

    events = _run(ResearchRequest(research_question="Remote work?", corpus_size=6), tmp_path)
    assert events[-1]["type"] == "result"
    warnings = [e for e in events if e.get("details", {}).get("raw_type") == "library_error"]
    assert len(warnings) == 2 and all(e["status"] == "warning" for e in warnings)
    assert "disk full" in warnings[0]["message"]


def test_cli_library_import_and_search(tmp_path):
    _run_folder(tmp_path, "run1", "Intermittent fasting and cognition?", [_PAPER])
    console = Console(record=True, width=300)
    cli._library_command(console, ["import", str(tmp_path)])
    cli._library_command(console, ["search", "fasting"])
    text = console.export_text()
    assert "Imported 1 run(s)" in text
    assert 'You researched a similar question on' in text and "QA 4.2" in text
    assert "Fasting and memory in adults" in text and "doi:10.1/abc" in text


def test_api_serves_a_past_runs_file_from_the_library(tmp_path):
    library.import_run(_run_folder(tmp_path, "pastrun", "Fasting?", []))
    client = TestClient(orchestrator_api.app)
    response = client.get("/research/runs/pastrun/files/draft.md")
    assert response.status_code == 200 and response.text == "past draft"
    assert client.get("/research/runs/pastrun/files/paper.pdf").status_code == 404
    assert client.get("/research/runs/unknown/files/draft.md").status_code == 404
