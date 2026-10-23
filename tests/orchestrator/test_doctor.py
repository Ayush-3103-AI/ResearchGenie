"""The Doctor (#13): one test per rule row over recorded event-log lines
(tests/fixtures/doctor/), then the confirmation rule and the fix → resume →
report loop through the headless CLI and the API, with every service faked."""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import doctor  # noqa: E402
from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless  # noqa: E402
from researchgenie import ollama_manager as om  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

FIXTURES = ROOT / "tests" / "fixtures" / "doctor"
HEALTHY = {"provider": "ollama", "model": "qwen3.5:9b", "ollama_running": True,
           "model_installed": True, "latex": True, "openalex_key": True, "gemini_key": False}


def _run_folder(tmp_path, fixture, metadata=None, pdf=False):
    folder = tmp_path / "what-is-x_r1"
    folder.mkdir()
    shutil.copy(FIXTURES / f"{fixture}.jsonl", folder / "events.jsonl")
    if metadata is not None:
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (folder / "final").mkdir()
    if pdf:
        (folder / "final" / "paper.pdf").write_bytes(b"%PDF")
    return folder


ROWS = [
    # fixture, live-status changes, metadata, pdf, rule, fix
    ("ollama_unreachable", {"ollama_running": False}, None, False, "ollama_unreachable", "start_ollama"),
    ("writing_model_failed", {"ollama_running": False, "model_installed": None}, None, False,
     "ollama_stopped", "start_ollama"),
    ("model_missing", {}, None, False, "model_missing", "pull_model"),
    ("writing_model_failed", {"model_installed": False}, None, False, "model_missing_now", "pull_model"),
    ("gemini_unavailable", {"provider": "gemini", "ollama_running": None}, None, False,
     "gemini_unavailable", "gemini_setup"),
    ("providers_exhausted", {"openalex_key": False}, None, False, "openalex_budget_no_key", "openalex_key"),
    ("providers_exhausted", {}, None, False, "providers_exhausted", "wait_for_reset"),
    ("all_sources_failed", {}, None, False, "all_sources_failed", "resume"),
    ("no_relevant_paper", {}, None, False, "no_relevant_paper", "rephrase"),
    ("completed", {}, {"draft_status": "partial", "broken_sections": ["Discussion"]}, True,
     "partial_draft", "resume_writing"),
    ("completed", {"latex": False}, {"draft_status": "complete"}, False, "missing_latex", "install_latex"),
]


@pytest.mark.parametrize("fixture, system, metadata, pdf, rule, fix", ROWS,
                         ids=[row[4] for row in ROWS])
def test_each_rule_row(tmp_path, fixture, system, metadata, pdf, rule, fix):
    folder = _run_folder(tmp_path, fixture, metadata, pdf)
    diagnosis = doctor.diagnose(folder, {**HEALTHY, **system})
    assert diagnosis["rule"] == rule and diagnosis["fix"]["id"] == fix
    assert diagnosis["cause"] and diagnosis["fix"]["label"]
    error = doctor.last_error(doctor.read_events(folder))
    if error is not None:
        assert f"error: {error}" in diagnosis["evidence"]
    if rule in ("providers_exhausted", "openalex_budget_no_key"):
        assert any("api.openalex.org" in line for line in diagnosis["evidence"])
    if rule in ("ollama_stopped", "model_missing_now", "missing_latex", "partial_draft"):
        assert any(line.startswith(("Live check", "metadata.json")) for line in diagnosis["evidence"])


def test_a_known_cause_is_not_blamed_on_a_model_that_is_merely_off(tmp_path):
    folder = _run_folder(tmp_path, "providers_exhausted")
    down = {**HEALTHY, "ollama_running": False, "model_installed": None}
    assert doctor.diagnose(folder, down)["rule"] == "providers_exhausted"


def test_a_healthy_completed_run_and_an_unknown_failure_are_reported_honestly(tmp_path):
    completed = _run_folder(tmp_path, "completed", {"draft_status": "complete"}, pdf=True)
    assert doctor.diagnose(completed, HEALTHY)["rule"] is None

    shutil.rmtree(completed)
    folder = _run_folder(tmp_path, "unknown")
    diagnosis = doctor.diagnose(folder, HEALTHY)
    assert diagnosis["rule"] is None and diagnosis["fix"] is None
    assert diagnosis["failure_class"] == "unknown"
    assert "No known cause" in diagnosis["cause"] and diagnosis["last_error"] == "Crossref went away."
    assert diagnosis["evidence"] == ["error: Crossref went away."]


def test_the_rows_without_a_live_status_give_the_headless_exit_class():
    assert doctor.match([]) is None
    for fixture, failure_class in (("ollama_unreachable", "model_unavailable"),
                                   ("providers_exhausted", "providers_exhausted"),
                                   ("no_relevant_paper", "no_relevant_papers")):
        lines = [json.loads(l) for l in (FIXTURES / f"{fixture}.jsonl").read_text().splitlines()]
        assert doctor.match(lines).failure_class == failure_class


# ------------------------------------------------- confirmation, fix, resume, report

@pytest.fixture
def ollama_died_in_verification(tmp_path, monkeypatch):
    """A real (faked-service) run that failed in Verification with Ollama
    unreachable, and an Ollama that is down until the Doctor starts it."""
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    async def ollama_gone(request):
        yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
        raise llm_provider.LLMProviderError(
            "Could not reach the local Ollama server at http://127.0.0.1:11434. Start it with: ollama serve")

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", ollama_gone)
    monkeypatch.setattr(headless, "_ready", lambda: True)
    assert headless.main(["run", "-q", "What is X?", "--json"]) == headless.EXIT_MODEL_UNAVAILABLE
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        _fake_run_verification)

    ollama = {"running": False, "started": 0}

    def start():
        ollama["started"] += 1
        ollama["running"] = True
        return True

    async def running():
        return ollama["running"]

    async def installed(model):
        return True

    monkeypatch.setattr(om, "start_ollama_server", start)
    monkeypatch.setattr(om, "is_ollama_server_running", running)
    monkeypatch.setattr(om, "is_model_installed", installed)
    folder = next(tmp_path.glob("*_*"))
    return folder, ollama


def _report(folder):
    return json.loads((folder / "doctor_report.json").read_text(encoding="utf-8"))


def test_headless_needs_yes_before_any_fix_is_applied(ollama_died_in_verification, capsys):
    folder, ollama = ollama_died_in_verification
    capsys.readouterr()
    code = headless.main(["doctor", folder.name.partition("_")[2], "--json"])
    out = json.loads(capsys.readouterr().out)
    assert ollama["started"] == 0 and out["applied"] is False and "--yes" in out["outcome"]
    assert code == headless.EXIT_MODEL_UNAVAILABLE  # the run is still failed

    report = _report(folder)
    assert report["rule"] == "ollama_unreachable" and report["fix"]["id"] == "start_ollama"
    assert report["evidence"] and report["applied"] is False and report["outcome"]


def test_doctor_yes_starts_ollama_resumes_and_records_the_outcome(ollama_died_in_verification, capsys):
    folder, ollama = ollama_died_in_verification
    capsys.readouterr()
    code = headless.main(["doctor", str(folder), "--yes", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == headless.EXIT_OK and ollama["started"] == 1
    assert headless.status_of(folder)["state"] == "completed"
    report = _report(folder)
    assert report["applied"] is True and report["outcome"] == "Fix applied; the resumed run completed."
    assert out["outcome"] == report["outcome"]


def test_a_fix_for_you_is_never_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    folder = _run_folder(tmp_path, "no_relevant_paper")
    monkeypatch.setattr(doctor, "system_status", _async(HEALTHY))
    headless.main(["doctor", str(folder), "--yes", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["fix"]["id"] == "rephrase" and out["applied"] is False
    assert out["outcome"].startswith("No automatic fix")


def _async(value):
    async def get():
        return value
    return get


def test_the_api_diagnoses_then_applies_only_when_asked(ollama_died_in_verification, monkeypatch):
    from fastapi.testclient import TestClient
    from orchestrator import api, run_registry
    folder, ollama = ollama_died_in_verification
    monkeypatch.setattr(run_registry, "_runs", {})
    client = TestClient(api.app)
    run_id = folder.name.partition("_")[2]

    shown = client.post(f"/research/runs/{run_id}/doctor", json={}).json()
    assert shown["rule"] == "ollama_unreachable" and shown["applied"] is False
    assert ollama["started"] == 0 and _report(folder)["applied"] is False

    stale = client.post(f"/research/runs/{run_id}/doctor", json={"apply": True, "fix": "pull_model"})
    assert stale.status_code == 409 and ollama["started"] == 0  # not the fix that was shown
    applied = client.post(f"/research/runs/{run_id}/doctor",
                          json={"apply": True, "fix": shown["fix"]["id"]}).json()
    assert applied["applied"] is True and applied["resumed"] is True and ollama["started"] == 1
    deadline = time.monotonic() + 5
    while client.get(f"/research/runs/{run_id}").json()["status"] != "completed":
        assert time.monotonic() < deadline, "the resumed run never completed"
        time.sleep(0.02)
    for _ in range(100):  # the report is written as the resume's stream ends
        if _report(folder)["outcome"] == "Fix applied; the resumed run completed.":
            break
        time.sleep(0.02)
    assert _report(folder)["outcome"] == "Fix applied; the resumed run completed."
    assert client.post("/research/runs/nope/doctor", json={}).status_code == 404
