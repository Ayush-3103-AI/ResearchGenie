"""Fast unit tests for the FastAPI backend, with the pipeline faked out."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from orchestrator import api as orchestrator_api  # noqa: E402
from shared.contracts.discovery_contract import (  # noqa: E402
    DiscoveryRequest, DiscoveryResult, NoveltyAssessment, PaperMetadata, ResearchGap,
)
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest, StageTimings  # noqa: E402
from shared.contracts.qa_contract import QualityAssuranceResult, QualityScores  # noqa: E402
from shared.contracts.verification_contract import VerificationResult  # noqa: E402
from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402
from orchestrator.pipeline import PipelineError  # noqa: E402


def _fake_pipeline_result() -> PipelineResult:
    discovery = DiscoveryResult(
        research_request=DiscoveryRequest(research_question="Q?", corpus_size=6),
        research_interpretation="interp", search_queries={"keyword": "x"},
        sources_searched=["OpenAlex"],
        selected_papers=[PaperMetadata(id=1, title="Paper One", has_abstract=True, abstract="abstract")],
        field_overview="overview", limitations="none",
        research_gaps=[ResearchGap(
            title="Gap", gap_type="methodological", impact="high",
            description="desc", evidence="evi", supporting_paper_ids=[1], research_questions=["Q1?"],
        )],
        novelty_analysis=NoveltyAssessment(novelty_summary="summary", confidence="medium", caveats="caveat"),
        confidence_notes="notes",
    )
    writing = WritingResult(
        research_outline="# Q\n## Introduction", research_draft_markdown="draft [1].",
        reference_candidates=['[1] A. Author, "Paper One," 2020.'],
        run_directory="/tmp/writing-run",
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="run1", generated_at="2026-08-27T00:00:00Z",
            sections_written=2, papers_cited=1,
        ),
    )
    verification = VerificationResult(
        validated_draft_markdown="draft [1].",
        validation_report="# Citation Verification Report\n",
    )
    qa = QualityAssuranceResult(
        final_draft_markdown="draft [1].", final_references=['[1] A. Author, "Paper One," 2020.'],
        final_validation_report="# Citation Verification Report\n",
        quality_report_markdown="# Quality Assurance Report\n",
        scores=QualityScores(
            citation_integrity=5.0, claim_source_alignment=5.0,
            process_control=5.0, literature_coverage=5.0, overall=5.0,
        ),
    )
    return PipelineResult(
        run_id="run1", run_directory="/tmp/run1",
        request=ResearchRequest(research_question="Q?"),
        discovery=discovery, writing=writing, verification=verification, quality_assurance=qa,
        timings=StageTimings(
            discovery_seconds=1.0, writing_seconds=1.0,
            verification_seconds=1.0, quality_assurance_seconds=1.0,
        ),
    )


async def _fake_run_pipeline_success(request, output_root=None, run_id=None):
    yield {"type": "progress", "run_id": run_id, "message": "working"}
    yield {"type": "result", "result": _fake_pipeline_result()}


async def _fake_run_pipeline_failure(request, output_root=None, run_id=None):
    yield {"type": "progress", "run_id": run_id, "message": "working"}
    raise PipelineError("Research Discovery Service produced no result.")


@pytest.fixture
def client():
    return TestClient(orchestrator_api.app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_research_endpoint_returns_pipeline_result(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research", json={"research_question": "What is X?"})
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "run1"
    assert body["quality_assurance"]["scores"]["overall"] == 5.0


def test_research_endpoint_rejects_invalid_request(client):
    response = client.post("/research", json={
        "research_question": "Q?", "target_format": "Other",
    })
    assert response.status_code == 422


def test_research_endpoint_returns_502_on_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_failure)
    response = client.post("/research", json={"research_question": "What is X?"})
    assert response.status_code == 502


def _wait_for_status(client, run_id, target_statuses, timeout=5.0):
    deadline = time.monotonic() + timeout
    body = None
    while time.monotonic() < deadline:
        response = client.get(f"/research/runs/{run_id}")
        body = response.json()
        if body["status"] in target_statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} never reached {target_statuses}, last saw {body}")


def test_start_research_run_returns_run_id_immediately(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    assert run_id

    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "completed"
    assert body["result"]["run_id"] == "run1"
    assert body["error"] is None


def test_start_research_run_marks_failed_on_pipeline_error(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_failure)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]

    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "failed"
    assert body["result"] is None
    assert "produced no result" in body["error"]


def test_get_research_run_returns_404_for_unknown_run_id(client):
    response = client.get("/research/runs/does-not-exist")
    assert response.status_code == 404


def test_research_run_events_stream_returns_progress_then_end(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api, "run_pipeline", _fake_run_pipeline_success)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]
    _wait_for_status(client, run_id, {"completed", "failed"})

    with client.stream("GET", f"/research/runs/{run_id}/events") as stream:
        assert stream.status_code == 200
        body = "".join(stream.iter_text())

    assert '"message": "working"' in body or '"message":"working"' in body
    assert '"run_id": "run1"' in body or '"run_id":"run1"' in body
    assert "event: end" in body


def test_stream_research_run_events_returns_404_for_unknown_run_id(client):
    response = client.get("/research/runs/does-not-exist/events")
    assert response.status_code == 404


@pytest.mark.parametrize("fake_stages, expected_status", [
    (_fake_run_pipeline_success, "completed"),
    (_fake_run_pipeline_failure, "failed"),
])
def test_past_run_survives_restart_from_events_jsonl(client, monkeypatch, tmp_path, fake_stages, expected_status):
    from orchestrator import pipeline as orchestrator_pipeline
    from orchestrator import run_registry

    # Real run_pipeline wrapper (so events.jsonl gets written), fake stages.
    monkeypatch.setattr(orchestrator_pipeline, "_run_stages", fake_stages)
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    response = client.post("/research/runs", json={"research_question": "What is X?"})
    run_id = response.json()["run_id"]
    before = _wait_for_status(client, run_id, {"completed", "failed"})

    monkeypatch.setattr(run_registry, "_runs", {})  # simulated restart

    body = client.get(f"/research/runs/{run_id}").json()
    assert body == before
    assert body["status"] == expected_status

    with client.stream("GET", f"/research/runs/{run_id}/events") as stream:
        assert stream.status_code == 200
        replay = "".join(stream.iter_text())
    assert '"message": "working"' in replay
    assert replay.endswith("event: end\ndata: {}\n\n")


def test_advisor_shortlist_hands_off_to_pipeline_under_same_run_id(client, monkeypatch):
    from shared.contracts.trend_contract import TopicCandidate, TrendAdvisorResult

    shortlist = TrendAdvisorResult(
        domain="MECHANICAL", generated_at="2026-09-16T00:00:00Z",
        shortlist=[TopicCandidate(topic="Topic A", growth_metric=2.0, paper_count=100,
                                  gap_evidence=["no work tests A under load"], gap_signal=True)],
    )

    async def fake_advisor(request, output_root=None, run_id=None):
        yield {"type": "progress", "run_id": run_id, "message": "ranking"}
        yield {"type": "result", "result": shortlist, "run_directory": f"/nowhere/x_{run_id}"}

    seen = {}

    async def fake_pipeline(request, output_root=None, run_id=None, **kwargs):
        seen.update(request=request, run_id=run_id, **kwargs)
        yield {"type": "result", "result": _fake_pipeline_result()}

    async def fake_question(candidate):
        return f"How does {candidate.topic} behave under load?"

    monkeypatch.setattr(orchestrator_api, "run_trend_advisor_stage", fake_advisor)
    monkeypatch.setattr(orchestrator_api, "run_pipeline", fake_pipeline)
    monkeypatch.setattr(orchestrator_api.advisor_cli, "_research_question_for", fake_question)

    run_id = client.post("/advisor/runs", json={"domain": "MECHANICAL"}).json()["run_id"]
    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["result"]["shortlist"][0]["topic"] == "Topic A"

    assert client.post(f"/advisor/runs/{run_id}/question", json={"topic": "Nope"}).status_code == 422
    drafted = client.post(f"/advisor/runs/{run_id}/question", json={"topic": "Topic A"}).json()
    assert drafted == {"research_question": "How does Topic A behave under load?", "drafted": True}

    response = client.post(f"/advisor/runs/{run_id}/pipeline", json={
        "topic": "Topic A", "research_question": drafted["research_question"]})
    assert response.status_code == 202
    body = _wait_for_status(client, run_id, {"completed", "failed"})
    assert body["status"] == "completed"
    assert seen["run_id"] == run_id
    assert seen["request"].research_question == drafted["research_question"]
    assert seen["trend_advisor"].chosen_topic == "Topic A"
    assert seen["advisor_directory"] == f"/nowhere/x_{run_id}"


def test_index_serves_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ResearchGenie" in response.text


def test_evidence_table_is_downloadable_and_has_a_tab(client, monkeypatch, tmp_path):
    (tmp_path / "final").mkdir()
    (tmp_path / "final" / "evidence_table.md").write_text("# Evidence table\n", encoding="utf-8")
    result = _fake_pipeline_result().model_copy(update={"run_directory": str(tmp_path)})
    monkeypatch.setattr(orchestrator_api.run_registry, "get_run",
                        lambda run_id: SimpleNamespace(result=result))

    response = client.get("/research/runs/run1/files/evidence_table.md")
    assert response.status_code == 200
    assert response.text.startswith("# Evidence table")
    page = client.get("/").text
    assert '["evidence_table.md", "Evidence table", ".md"]' in page and '"Evidence": ""' in page


def test_system_endpoint_reports_each_scholarly_provider_without_secrets(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api.llm_provider, "health_check", _async_true)
    monkeypatch.setenv("OPENALEX_API_KEY", "oa-secret-0123456789")
    monkeypatch.setenv("CONTACT_EMAIL", "student@example.edu")
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    body = client.get("/system").json()
    providers = {p["name"]: p for p in body["providers"]}
    assert providers["OpenAlex"]["access"] == "keyed"
    assert providers["PubMed (NCBI)"]["access"] == "identified"
    assert providers["Semantic Scholar"]["access"] == "anonymous"
    assert all(p["health"] in ("healthy", "degraded") for p in body["providers"])
    assert "oa-secret-0123456789" not in str(body) and "student@example.edu" not in str(body)


async def _async_true():
    return True


def test_the_budget_endpoint_tells_the_page_when_to_offer_the_key_or_the_reduced_plan(client, monkeypatch):
    monkeypatch.setattr(orchestrator_api.rg_config, "load", lambda: orchestrator_api.rg_config.Config())
    monkeypatch.setattr(orchestrator_api.scholarly_gateway, "balance", lambda: 30)
    body = client.post("/research/budget", json={"research_question": "What is X?"}).json()
    assert body["plan"] == "reduced" and body["balance"] == 30 and body["estimated_credits"] == 71
    chosen = client.post("/research/budget", json={"research_question": "What is X?",
                                                   "budget_plan": "full"}).json()
    assert chosen["plan"] == "full"


def test_a_web_run_waits_for_approval_and_approve_continues_it(monkeypatch):
    # One event loop for the whole test (the run waits on a future across requests);
    # the lifespan then runs, so keep it off the real ~/.researchgenie config.
    monkeypatch.setattr(orchestrator_api.rg_config, "load", lambda: orchestrator_api.rg_config.Config())
    with TestClient(orchestrator_api.app) as client:
        _approval_round_trip(client, monkeypatch)


def _approval_round_trip(client, monkeypatch):
    papers = [{"id": 1, "title": "Kept"}, {"id": 2, "title": "Dropped"}]
    decisions = []

    async def pipeline_with_approval(request, output_root=None, run_id=None, approve=None):
        yield {"type": "progress", "run_id": run_id, "message": "review the papers",
               "details": {"raw_type": "approval_request", "papers": papers}}
        decisions.append(await approve(papers, 0))
        yield {"type": "result", "result": _fake_pipeline_result()}

    monkeypatch.setattr(orchestrator_api, "run_pipeline", pipeline_with_approval)
    run_id = client.post("/research/runs", json={"research_question": "Q?", "approval": "ask"}).json()["run_id"]
    _wait_for_status(client, run_id, {"awaiting_approval"})
    assert client.post(f"/research/runs/{run_id}/approve", json={"drop": [2]}).status_code == 200
    _wait_for_status(client, run_id, {"completed"})
    assert decisions == [{"approved_by": "web", "drop": [2]}]
    assert client.post(f"/research/runs/{run_id}/approve", json={"drop": []}).status_code == 409


def test_an_api_run_approves_automatically_by_default(client, monkeypatch):
    seen = {}

    async def pipeline(request, output_root=None, run_id=None, approve=None):
        seen["approve"] = approve
        yield {"type": "result", "result": _fake_pipeline_result()}

    monkeypatch.setattr(orchestrator_api, "run_pipeline", pipeline)
    run_id = client.post("/research/runs", json={"research_question": "Q?"}).json()["run_id"]
    _wait_for_status(client, run_id, {"completed"})
    assert seen["approve"] is None
