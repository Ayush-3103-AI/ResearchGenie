"""The decision brief (#12): one test per rule row over hand-built run
folders, then the brief written at the end of a real (faked-service) run,
completed and failed, without a model call, and shown by every surface."""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import decision_brief  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless, tui  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.verification_contract import ReferenceCheck  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_qa_result, _fake_run_discovery, _fake_run_qa, _fake_run_verification,
    _fake_run_writing, _fake_verification_result, _fake_writing_result,
)

REFERENCES = [f'[{n}] A. Author, "Paper {n}," 2020, doi: 10.1/{n}.' for n in (1, 2, 3)]


def _event(stage, status="running", **details):
    return {"type": "progress", "run_id": "r1", "stage": stage, "status": status,
            "message": "m", "details": details}


def _folder(tmp_path, events=None, draft=None, invalid=(), unverifiable=(), failed=False):
    """A run folder: the given events (else a completed run's), the Writing,
    Verification and QA results, the draft metadata overridden by `draft`."""
    folder = tmp_path / "what-is-x_r1"
    folder.mkdir()
    events = events or [_event("quality_assurance", "done")]
    events = events + [{"type": "error", "message": "Ollama went away"} if failed
                       else {"type": "result", "result": {}}]
    (folder / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    writing = _fake_writing_result()
    writing = writing.model_copy(update={
        "reference_candidates": REFERENCES,
        "draft_metadata": writing.draft_metadata.model_copy(update=draft or {})})
    check = lambda n, issue: ReferenceCheck(reference_entry=REFERENCES[n - 1], verified=False, issue=issue)  # noqa: E731
    verification = _fake_verification_result().model_copy(update={
        "verified_references": [ReferenceCheck(reference_entry=REFERENCES[0], verified=True)],
        "invalid_references": [check(n, "DOI not found") for n in invalid],
        "unverifiable_references": [check(n, "no DOI") for n in unverifiable]})
    for name, result in (("02_writing", writing), ("03_verification", verification),
                         ("04_quality_assurance", _fake_qa_result())):
        (folder / name).mkdir()
        (folder / name / "result.json").write_text(result.model_dump_json(), encoding="utf-8")
    return folder


ROWS = [
    # events, draft metadata, invalid, unverifiable, failed, rule, words in the action
    ([_event("writing")], {"broken_sections": ["Discussion"]}, (), (), True,
     "run_failed", ("researchgenie doctor r1", "Writing", "Ollama went away")),
    (None, {"draft_status": "partial", "broken_sections": ["Discussion", "Methods"]}, (2,), (), False,
     "broken_sections", ("--from writing", "2 section(s) are broken: Discussion, Methods")),
    ([_event("verification", "warning", raw_type="provider_exhausted", host="api.openalex.org")],
     {"evidence_limited_sections": ["Results"]}, (), (3,), False,
     "reduced_plan", ("Add your OpenAlex key", "--from verification", "reduced plan")),
    ([{"status": "budget_check", "details": {"plan": "reduced"}}], {}, (), (), False,
     "reduced_plan", ("Add your OpenAlex key", "--from verification")),
    (None, {"evidence_limited_sections": ["Methods", "Results"]}, (2,), (), False,
     "evidence_limited", ("--from writing", "2 section(s) were evidence-limited: Methods, Results")),
    (None, {}, (2,), (3,), False, "check_references", ("Check reference(s) 2, 3 by hand",)),
    (None, {}, (), (), False, "ready", ("evidence_table.md",)),
]


@pytest.mark.parametrize("events, draft, invalid, unverifiable, failed, rule, words", ROWS,
                         ids=[f"{row[5]}-{i}" for i, row in enumerate(ROWS)])
def test_each_rule_row(tmp_path, events, draft, invalid, unverifiable, failed, rule, words):
    brief = decision_brief.build_brief(_folder(tmp_path, events, draft, invalid, unverifiable, failed))
    assert brief["rule"] == rule
    assert all(word in brief["next_action"] for word in words), brief["next_action"]
    # Exactly one next action, in the markdown under its own heading.
    assert brief["markdown"].count("## Next action") == 1
    assert f"## Next action\n\n{brief['next_action']}\n" in brief["markdown"]


def test_every_flagged_reference_and_section_is_listed(tmp_path):
    folder = _folder(tmp_path, draft={"broken_sections": ["Discussion"],
                                      "evidence_limited_sections": ["Methods"]},
                     invalid=(3,), unverifiable=(2,))
    brief = decision_brief.build_brief(folder)
    assert brief["check_references"] == [2, 3]
    assert brief["sections"] == ["Discussion", "Methods"]
    for line in ("- [2] unverifiable: no DOI", "- [3] invalid: DOI not found",
                 "- Discussion: broken", "- Methods: evidence-limited"):
        assert line in brief["markdown"]
    assert "Overall quality 5.0/5" in brief["strong"]
    assert "1 invalid reference(s)" in brief["weak"] and "1 unverifiable reference(s)" in brief["weak"]


def test_cache_and_memory_hits_are_read_when_logged(tmp_path):
    events = [_event("verification", raw_type="cache_hit", hits=4),
              {"status": "memory_hit", "details": {"count": 2}}]
    brief = decision_brief.build_brief(_folder(tmp_path, events))
    assert "4 scholarly request(s) answered from the local cache" in brief["strong"]
    assert "2 paper excerpt(s) reused from the research library" in brief["strong"]


# ------------------------------------------------- at the end of every run

@pytest.fixture
def services(tmp_path, monkeypatch):
    """Faked services, and a model that must never be asked."""
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    def refuse(*args, **kwargs):
        raise AssertionError("the decision brief must not call the model")
    for name in ("stream_json", "complete_json", "complete_tools", "health_check"):
        monkeypatch.setattr(llm_provider, name, refuse)
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    return monkeypatch


async def _drain(events):
    async for _ in events:
        pass


def test_a_completed_run_writes_final_decision_brief_without_the_model(tmp_path, services):
    asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
        ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    folder = orchestrator_pipeline.find_run_directory("r1", tmp_path)
    written = (folder / "final" / "decision_brief.md").read_text(encoding="utf-8")
    assert written == decision_brief.build_brief(folder)["markdown"]
    assert "Run r1 · completed" in written

    report = headless.result_of(folder)
    assert report["decision_brief"]["path"] == str(folder / "final" / "decision_brief.md")
    assert report["decision_brief"]["markdown"] == written
    assert report["decision_brief"]["next_action"] in written
    assert "decision_brief.md" in report["final_files"]


def test_a_failed_run_gets_a_brief_that_points_to_the_doctor_until_a_resume_completes(tmp_path, services):
    async def failing_writing(request, output_root=None):
        raise orchestrator_pipeline.PipelineError("Research Writing Service produced no result.")
        yield  # pragma: no cover
    services.setattr(orchestrator_pipeline.writing_service, "run_writing", failing_writing)
    with pytest.raises(orchestrator_pipeline.PipelineError):
        asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
            ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    folder = orchestrator_pipeline.find_run_directory("r1", tmp_path)
    assert not (folder / "final").exists()
    report = headless.result_of(folder)
    assert report["decision_brief"]["path"] == str(folder / "decision_brief.md")
    assert report["decision_brief"]["next_action"].startswith("Run `researchgenie doctor r1`")

    services.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    asyncio.run(_drain(orchestrator_pipeline.resume_pipeline(folder)))
    assert (folder / "final" / "decision_brief.md").is_file()
    assert not (folder / "decision_brief.md").exists()


def test_the_api_cli_and_web_page_show_the_brief(tmp_path, services):
    asyncio.run(_drain(orchestrator_pipeline.run_pipeline(
        ResearchRequest(research_question="What is X?", corpus_size=6), output_root=tmp_path, run_id="r1")))
    brief = decision_brief.build_brief(orchestrator_pipeline.find_run_directory("r1", tmp_path))

    from orchestrator import api
    client = TestClient(api.app)
    body = client.get("/research/runs/r1/brief").json()
    assert body["next_action"] == brief["next_action"] and body["path"].endswith("decision_brief.md")
    assert client.get("/research/runs/nope/brief").status_code == 404
    assert '"Decision brief": "_Loading the decision brief…_"' in client.get("/").text

    console = Console(record=True, width=400)
    tui.render_brief(console, brief)
    assert brief["next_action"] in console.export_text()
