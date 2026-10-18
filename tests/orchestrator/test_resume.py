"""Resume a failed run from the failed stage (#9), with every service faked."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

REQUEST = ResearchRequest(research_question="What is X?", corpus_size=6)


async def _failing_verification(request):
    yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
    raise orchestrator_pipeline.PipelineError("Crossref went away.")


async def _must_not_run(*args, **kwargs):
    raise AssertionError("an earlier stage was re-run on resume")
    yield  # pragma: no cover


def _drain(events):
    async def run():
        return [event async for event in events]
    return asyncio.run(run())


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    """A run that finished Discovery and Writing, then failed in Verification."""
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _failing_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)
    with pytest.raises(orchestrator_pipeline.PipelineError):
        _drain(orchestrator_pipeline.run_pipeline(REQUEST, output_root=tmp_path, run_id="abc123"))
    # From here on Discovery and Writing must not be called again.
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _must_not_run)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _must_not_run)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    return tmp_path / "what-is-x_abc123"


def test_resuming_a_run_that_failed_in_verification_skips_discovery_and_writing(failed_run):
    events = _drain(orchestrator_pipeline.resume_pipeline(failed_run))

    result = events[-1]["result"]
    assert result.run_id == "abc123" and Path(result.run_directory) == failed_run
    assert result.discovery.selected_papers[0].title == "Paper One"  # loaded, not re-run
    for name in ("draft.md", "references.md", "evidence_table.md", "paper.tex"):
        assert (failed_run / "final" / name).is_file()


def test_the_resume_is_recorded_in_the_event_log_and_metadata(failed_run):
    _drain(orchestrator_pipeline.resume_pipeline(failed_run))

    logged = [json.loads(line) for line in (failed_run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    resume = [e for e in logged if e.get("details", {}).get("raw_type") == "resume"]
    assert len(resume) == 1 and resume[0]["details"]["from_stage"] == "verification"
    assert logged.index(resume[0]) > [e["type"] for e in logged].index("error")
    assert logged[-1]["type"] == "result"

    metadata = json.loads((failed_run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_from_stage"] == "verification"
    assert metadata["final_versions"] == []


def test_an_existing_final_folder_is_kept_as_a_numbered_version(failed_run):
    _drain(orchestrator_pipeline.resume_pipeline(failed_run))
    (failed_run / "final" / "draft.md").write_text("first draft", encoding="utf-8")

    _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="quality_assurance"))
    _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="quality_assurance"))

    assert (failed_run / "final_v1" / "draft.md").read_text(encoding="utf-8") == "first draft"
    assert (failed_run / "final_v2" / "draft.md").is_file()
    assert (failed_run / "final" / "draft.md").is_file()
    metadata = json.loads((failed_run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_from_stage"] == "quality_assurance"
    assert metadata["final_versions"] == ["final_v1", "final_v2"]


def test_a_corrupt_stage_result_fails_with_a_message_naming_the_stage(failed_run):
    (failed_run / "02_writing" / "result.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(orchestrator_pipeline.PipelineError, match="writing"):
        _drain(orchestrator_pipeline.resume_pipeline(failed_run))


def test_a_missing_earlier_stage_result_fails_with_a_message_naming_the_stage(failed_run):
    (failed_run / "01_discovery" / "result.json").unlink()
    with pytest.raises(orchestrator_pipeline.PipelineError, match="discovery"):
        _drain(orchestrator_pipeline.resume_pipeline(failed_run, from_stage="verification"))


def test_resume_finds_a_run_by_its_id(failed_run):
    assert orchestrator_pipeline.find_run_directory("abc123", failed_run.parent) == failed_run
    assert orchestrator_pipeline.find_run_directory("nope", failed_run.parent) is None


# ------------------------------------------------------------------ API + CLI

def _api_client(monkeypatch, output_root):
    from fastapi.testclient import TestClient
    from orchestrator import api, run_registry
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", output_root)
    monkeypatch.setattr(run_registry, "_runs", {})
    return TestClient(api.app)


def test_the_api_resumes_a_failed_run_and_streams_its_events(failed_run, monkeypatch):
    import time
    client = _api_client(monkeypatch, failed_run.parent)
    assert client.get("/research/runs/abc123").json()["status"] == "failed"

    response = client.post("/research/runs/abc123/resume", json={})
    assert response.status_code == 202 and response.json() == {"run_id": "abc123"}
    for _ in range(200):
        body = client.get("/research/runs/abc123").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] == "completed"
    with client.stream("GET", "/research/runs/abc123/events") as stream:
        replay = "".join(stream.iter_text())
    assert '"raw_type": "resume"' in replay and replay.endswith("event: end\ndata: {}\n\n")


def test_the_api_refuses_a_resume_it_cannot_do(failed_run, monkeypatch):
    client = _api_client(monkeypatch, failed_run.parent)
    assert client.post("/research/runs/nope/resume", json={}).status_code == 404
    (failed_run / "02_writing" / "result.json").write_text("{not json", encoding="utf-8")
    response = client.post("/research/runs/abc123/resume", json={})
    assert response.status_code == 422 and "writing" in response.json()["detail"]


def test_the_cli_accepts_a_run_folder_or_a_run_id(failed_run, monkeypatch):
    from researchgenie import cli
    monkeypatch.setattr(cli, "DEFAULT_OUTPUT_ROOT", failed_run.parent)
    assert cli._resolve_run(str(failed_run)) == failed_run
    assert cli._resolve_run("abc123") == failed_run
    assert cli._resolve_run("nope") is None


def test_the_web_page_offers_resume_only_for_failed_runs():
    page = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="resume"' in page and "/resume" in page
