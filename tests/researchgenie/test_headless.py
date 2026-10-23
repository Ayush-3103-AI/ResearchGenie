"""The headless CLI (#10), driven through its own surface (argv in, stdout
JSON and an exit code out) over faked pipeline services."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import headless  # noqa: E402
from shared.utilities import llm_provider, response_cache  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)


@pytest.fixture(autouse=True)
def outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(headless, "DEFAULT_OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery)
    monkeypatch.setattr(orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing)
    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification)
    monkeypatch.setattr(orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)
    monkeypatch.setattr(headless, "_ready", lambda: True)
    return tmp_path


def _cli(capsys, *argv):
    code = headless.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip().startswith("{") else out


def _failing_discovery(message, exhausted=False):
    async def discovery(request, approve=None):
        if exhausted:
            yield {"type": "provider_exhausted", "host": "api.openalex.org",
                   "remedy": "Add a key", "message": "OpenAlex is out of budget"}
        if "Ollama" in message:
            raise llm_provider.LLMProviderError(message)
        raise orchestrator_pipeline.PipelineError(message)
        yield  # pragma: no cover
    return discovery


def test_run_takes_every_request_field_and_returns_the_result_as_json(capsys, outputs):
    code, report = _cli(
        capsys, "run", "-q", "What is X?", "--corpus-size", "7", "--format", "Springer",
        "--publication-type", "journal", "--target-venue", "Nature", "--deadline", "2026-12-01",
        "--domain", "biology", "--year-range", "2015-2024", "--preferred-databases", "pubmed",
        "--language", "en", "--length", "short", "--keywords", "fasting", "--keywords", "memory",
        "--excluded-topics", "mice", "--clarify", "skip", "--approval", "auto", "--no-memory",
        "--json")
    assert code == headless.EXIT_OK
    assert report["state"] == "completed" and report["draft_status"] == "complete"
    assert report["scores"]["overall"] == 5.0
    assert report["verification"] == {"verified": 0, "invalid": 0, "unverifiable": 0}
    assert Path(report["final_files"]["draft.md"]).is_file()

    saved = json.loads((Path(report["run_directory"]) / "00_request.json").read_text())
    assert saved["corpus_size"] == 7 and saved["target_format"] == "Springer"
    assert saved["year_range"] == [2015, 2024] and saved["keywords"] == ["fasting", "memory"]
    assert saved["excluded_topics"] == ["mice"] and saved["preferred_databases"] == ["pubmed"]
    assert saved["publication_type"] == "journal" and saved["target_venue"] == "Nature"
    assert saved["max_draft_length"] > 0 and saved["approval"] == "auto"
    assert saved["memory"] is False

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["state"] == "completed"
    assert status["stage"] == "quality_assurance" and status["last_event"]["message"]


def test_a_headless_run_short_of_openalex_budget_takes_the_reduced_plan_and_says_so(capsys, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline.scholarly_gateway, "balance", lambda: 30)
    code = headless.main(["run", "-q", "What is X?", "--json"])
    err = capsys.readouterr().err
    assert code == headless.EXIT_OK
    assert "Running the reduced plan" in err and "OpenAlex key" in err


@pytest.mark.parametrize("message, exhausted, exit_code, failure_class", [
    ("Could not reach the local Ollama server at http://127.0.0.1:11434. Start it with: ollama serve",
     False, headless.EXIT_MODEL_UNAVAILABLE, "model_unavailable"),
    ("No papers found in any source. Try rephrasing the question or enabling more sources.",
     True, headless.EXIT_PROVIDERS_EXHAUSTED, "providers_exhausted"),
    ("No paper in the search results was judged relevant to the research question. Try again.",
     False, headless.EXIT_NO_RELEVANT_PAPERS, "no_relevant_papers"),
    ("Crossref went away.", False, headless.EXIT_FAILED, "unknown"),
])
def test_each_failure_class_has_its_own_exit_code(capsys, monkeypatch, message, exhausted,
                                                  exit_code, failure_class):
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "run_discovery",
                        _failing_discovery(message, exhausted))
    code, report = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == exit_code
    assert report["state"] == "failed" and report["failure_class"] == failure_class
    assert report["error"] == message

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["failure_class"] == failure_class
    code, _ = _cli(capsys, "result", report["run_id"], "--json")
    assert code == exit_code


def test_an_unready_model_stops_before_the_run(capsys, monkeypatch):
    monkeypatch.setattr(headless, "_ready", lambda: False)
    code, report = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == headless.EXIT_MODEL_UNAVAILABLE and report["failure_class"] == "model_unavailable"


def test_detach_prints_the_run_id_and_starts_a_fixed_background_command(capsys, monkeypatch):
    started = []

    class FakePopen:
        def __init__(self, argv, **options):
            started.append((argv, options))

    monkeypatch.setattr(headless.subprocess, "Popen", FakePopen)
    code, report = _cli(capsys, "run", "-q", "What is X?; rm -rf /", "--detach", "--json")
    assert code == 0 and report["state"] == "started" and report["run_id"]
    [(argv, options)] = started
    assert argv[:3] == [sys.executable, "-m", "researchgenie.cli"]
    assert "--detach" not in argv and argv[-2:] == ["--run-id", report["run_id"]]
    assert "What is X?; rm -rf /" in argv  # one argv item, never parsed by a shell
    assert not options.get("shell")

    code, status = _cli(capsys, "status", report["run_id"], "--json")
    assert code == 0 and status["state"] == "starting"


def test_a_chosen_run_id_is_plain_and_used_once(capsys):
    code, report = _cli(capsys, "run", "-q", "What is X?", "--run-id", "rg-20261023-1530", "--json")
    assert code == 0 and report["run_id"] == "rg-20261023-1530"
    code, status = _cli(capsys, "status", "rg-20261023-1530", "--json")
    assert code == 0 and status["state"] == "completed"
    code, refused = _cli(capsys, "run", "-q", "What is Y?", "--run-id", "rg-20261023-1530", "--json")
    assert code == headless.EXIT_USAGE and "already used" in refused["error"]
    assert headless.main(["run", "-q", "What is X?", "--run-id", "../escape", "--json"]) == headless.EXIT_USAGE


def test_resume_is_available_headless(capsys, monkeypatch):
    async def verification_fails(request):
        yield {"type": "status", "stage": "verify", "state": "running", "message": "verifying"}
        raise orchestrator_pipeline.PipelineError("Crossref went away.")

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        verification_fails)
    code, failed = _cli(capsys, "run", "-q", "What is X?", "--json")
    assert code == headless.EXIT_FAILED and failed["stage"] == "verification"

    monkeypatch.setattr(orchestrator_pipeline.verification_service, "run_verification",
                        _fake_run_verification)
    code, report = _cli(capsys, "resume", failed["run_id"], "--json")
    assert code == 0 and report["state"] == "completed"

    code, refused = _cli(capsys, "resume", failed["run_id"], "--json")  # nothing left to resume
    assert code == headless.EXIT_FAILED and "already has a saved result" in refused["error"]


def test_unknown_and_unfinished_runs(capsys, outputs):
    code, report = _cli(capsys, "status", "nope", "--json")
    assert code == headless.EXIT_NOT_FOUND and "nope" in report["error"]

    running = outputs / "what-is-x_r1"
    running.mkdir()
    (running / "events.jsonl").write_text(json.dumps(
        {"type": "progress", "stage": "writing", "status": "running", "message": "writing"}) + "\n")
    code, status = _cli(capsys, "status", "r1", "--json")
    assert code == 0 and status["state"] == "running" and status["stage"] == "writing"
    code, report = _cli(capsys, "result", str(running), "--json")
    assert code == headless.EXIT_NOT_FINISHED and report["final_files"] == {}


@pytest.mark.parametrize("argv", [
    ["run", "--json"],                                    # no question
    ["run", "-q", "Q?", "--approval", "ask"],             # nobody to ask
    ["run", "-q", "Q?", "--clarify", "ask"],
    ["run", "-q", "Q?", "--corpus-size", "40", "--json"],  # out of the contract's range
    ["run", "-q", "Q?", "--year-range", "2015"],
])
def test_bad_arguments_exit_2(capsys, argv):
    assert headless.main(argv) == headless.EXIT_USAGE


def test_library_search_speaks_json(capsys, outputs):
    code, report = _cli(capsys, "run", "-q", "Does sleep aid memory?", "--json")
    code, found = _cli(capsys, "library", "search", "sleep", "memory", "--json")
    assert code == 0 and [r["run_id"] for r in found["runs"]] == [report["run_id"]]


def test_which_command_lines_are_headless():
    assert headless.is_headless(["run", "-q", "Q"]) and headless.is_headless(["status", "x"])
    assert headless.is_headless(["resume", "x", "--json"])
    assert not headless.is_headless(["resume", "x"]) and not headless.is_headless([])
    assert not headless.is_headless(["library", "search", "x"])
    assert headless.is_headless(["library", "search", "x", "--json"])


def test_no_cache_and_cache_clear(capsys, monkeypatch):
    _cli(capsys, "status", "nope", "--no-cache")
    assert response_cache.reads_enabled() is False
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")

    from researchgenie import cli
    monkeypatch.setattr(sys, "argv", ["researchgenie", "--no-cache", "status", "nope", "--json"])
    with pytest.raises(SystemExit) as stop:  # headless, not the interactive app
        cli.main()
    assert stop.value.code == headless.EXIT_NOT_FOUND and not response_cache.reads_enabled()
    monkeypatch.delenv("RESEARCHGENIE_NO_CACHE")

    response_cache.put("GET x?", 200, "text/plain", b"x", None, 0.0)
    from researchgenie import cli
    monkeypatch.setattr(sys, "argv", ["researchgenie", "cache", "clear"])
    cli.main()
    assert response_cache.get("GET x?", 0.0) is None
