"""Clarify step (issue #14): one structured model call before Discovery, the
user's accept/edit/keep choice on the CLI and the web, and the record of it
in 00_request.json and the event stream. No real model calls."""

import asyncio
import io
import json
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
from shared.contracts.pipeline_contract import Clarification, ResearchRequest  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402
from test_pipeline import _patch_services  # noqa: E402, F401 — autouse: fakes all four services

analysis = sys.modules["discovery.analysis"]

VAGUE = "AI in healthcare"
SHARPER = "How accurate are deep-learning models at detecting diabetic retinopathy from fundus images?"
VAGUE_ANSWER = {"specific": False, "missing": ["task", "population"], "suggested_question": SHARPER}
SPECIFIC_ANSWER = {"specific": True, "missing": [], "suggested_question": SHARPER}


def _fake_clarify(answer):
    calls = []

    async def fake(question, model_name=None):
        calls.append(question)
        return answer
    return fake, calls


# ---- analysis.clarify_question ----

def test_clarify_question_makes_one_structured_call_without_thinking(monkeypatch):
    calls = []

    async def fake_complete_json(**kwargs):
        calls.append(kwargs)
        return llm_provider.Completion(text=json.dumps(VAGUE_ANSWER))
    monkeypatch.setattr(analysis.llm, "complete_json", fake_complete_json)

    assert asyncio.run(analysis.clarify_question(VAGUE)) == VAGUE_ANSWER
    assert len(calls) == 1
    assert calls[0]["think"] is False
    assert calls[0]["schema"] is analysis.CLARIFY_SCHEMA
    assert set(analysis.CLARIFY_SCHEMA["required"]) == {"specific", "missing", "suggested_question"}
    assert VAGUE in calls[0]["user"]


@pytest.mark.parametrize("outcome", ["raise", "garbage", "cut_off"])
def test_clarify_question_model_failure_returns_none(monkeypatch, outcome):
    async def fake_complete_json(**kwargs):
        if outcome == "raise":
            raise llm_provider.LLMProviderError("Could not reach the local Ollama server")
        if outcome == "garbage":
            return llm_provider.Completion(text="not json")
        return llm_provider.Completion(text='{"specific": fal', stop="length")
    monkeypatch.setattr(analysis.llm, "complete_json", fake_complete_json)

    assert asyncio.run(analysis.clarify_question(VAGUE)) is None


def test_a_suggestion_equal_to_the_question_counts_as_specific(monkeypatch):
    async def fake_complete_json(**kwargs):
        return llm_provider.Completion(text=json.dumps(
            {"specific": False, "missing": [], "suggested_question": VAGUE}))
    monkeypatch.setattr(analysis.llm, "complete_json", fake_complete_json)

    assert asyncio.run(analysis.clarify_question(VAGUE))["specific"] is True


# ---- pipeline: request file + event ----

def _run(request, tmp_path):
    async def go():
        events, result = [], None
        async for event in orchestrator_pipeline.run_pipeline(request, output_root=tmp_path):
            if event["type"] == "result":
                result = event["result"]
            else:
                events.append(event)
        return events, result
    events, result = asyncio.run(go())
    saved = json.loads((Path(result.run_directory) / "00_request.json").read_text(encoding="utf-8"))
    clarify_events = [e for e in events if e["details"].get("raw_type") == "clarify"]
    return result, saved, clarify_events


def test_default_mode_skips_the_clarify_step(tmp_path, monkeypatch):
    fake, calls = _fake_clarify(VAGUE_ANSWER)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "clarify_question", fake)

    result, saved, clarify_events = _run(ResearchRequest(research_question=VAGUE), tmp_path)

    assert calls == []
    assert clarify_events == []
    assert saved["clarify"] == "skip" and saved["clarification"] is None
    assert result.request.research_question == VAGUE


def test_auto_mode_takes_the_suggestion_and_records_it(tmp_path, monkeypatch):
    fake, calls = _fake_clarify(VAGUE_ANSWER)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "clarify_question", fake)

    result, saved, clarify_events = _run(
        ResearchRequest(research_question=VAGUE, clarify="auto"), tmp_path)

    assert calls == [VAGUE]
    assert saved["research_question"] == SHARPER
    assert saved["clarification"]["original_question"] == VAGUE
    assert saved["clarification"]["decision"] == "accept"
    [event] = clarify_events
    assert event["type"] == "progress"
    assert event["details"]["original"] == VAGUE
    assert event["details"]["chosen"] == SHARPER
    assert event["details"]["decision"] == "accept"
    assert result.request.research_question == SHARPER
    # The event log (#8) lands in the folder named after the chosen question.
    assert (Path(result.run_directory) / "events.jsonl").exists()
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [Path(result.run_directory).name]


def test_auto_mode_model_failure_keeps_the_original_and_the_run_completes(tmp_path, monkeypatch):
    fake, _ = _fake_clarify(None)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "clarify_question", fake)

    result, saved, clarify_events = _run(
        ResearchRequest(research_question=VAGUE, clarify="auto"), tmp_path)

    assert result.quality_assurance.scores.overall == 5.0
    assert saved["research_question"] == VAGUE
    assert saved["clarification"]["decision"] == "keep"
    assert saved["clarification"]["specific"] is None
    assert clarify_events[0]["details"]["chosen"] == VAGUE


def test_ask_mode_records_the_surfaces_choice_without_a_second_call(tmp_path, monkeypatch):
    fake, calls = _fake_clarify(VAGUE_ANSWER)
    monkeypatch.setattr(orchestrator_pipeline.discovery_service, "clarify_question", fake)
    edited = "How accurate is AI at grading diabetic retinopathy in rural clinics?"
    request = ResearchRequest(
        research_question=edited, clarify="ask",
        clarification=Clarification(original_question=VAGUE, decision="edit", **VAGUE_ANSWER))

    _, saved, clarify_events = _run(request, tmp_path)

    assert calls == []
    assert saved["research_question"] == edited
    assert saved["clarification"]["decision"] == "edit"
    assert {"original": VAGUE, "chosen": edited, "decision": "edit"}.items() <= clarify_events[0]["details"].items()


# ---- API ----

def test_clarify_endpoint_returns_the_suggestion_for_a_vague_question(monkeypatch):
    fake, calls = _fake_clarify(VAGUE_ANSWER)
    monkeypatch.setattr(orchestrator_api.discovery_service, "clarify_question", fake)

    response = TestClient(orchestrator_api.app).post(
        "/research/clarify", json={"research_question": VAGUE})

    assert response.status_code == 200
    assert response.json() == {"available": True, **VAGUE_ANSWER}
    assert calls == [VAGUE]


def test_clarify_endpoint_reports_a_model_failure_as_unavailable(monkeypatch):
    fake, _ = _fake_clarify(None)
    monkeypatch.setattr(orchestrator_api.discovery_service, "clarify_question", fake)

    body = TestClient(orchestrator_api.app).post(
        "/research/clarify", json={"research_question": VAGUE}).json()

    assert body["available"] is False


def test_api_request_accepts_a_clarify_mode():
    request = ResearchRequest.model_validate({"research_question": VAGUE, "clarify": "auto"})
    assert request.clarify == "auto"
    with pytest.raises(ValueError):
        ResearchRequest.model_validate({"research_question": VAGUE, "clarify": "sometimes"})


# ---- CLI ----

def _console():
    return Console(file=io.StringIO(), width=200, legacy_windows=False)


def _answers(monkeypatch, *answers):
    queue = list(answers)
    asked = []

    def fake_ask(prompt, *args, **kwargs):
        asked.append(prompt)
        return queue.pop(0)
    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(fake_ask))
    return asked


@pytest.mark.parametrize("choice, extra, chosen, decision", [
    ("1", [], SHARPER, "accept"),
    ("2", ["My own sharper question?"], "My own sharper question?", "edit"),
    ("3", [], VAGUE, "keep"),
])
def test_cli_shows_the_suggestion_for_a_vague_question(monkeypatch, choice, extra, chosen, decision):
    monkeypatch.setattr(cli.discovery_service, "clarify_question", _fake_clarify(VAGUE_ANSWER)[0])
    _answers(monkeypatch, choice, *extra)
    console = _console()

    question, clarification = cli._clarify(console, VAGUE)

    assert SHARPER in console.file.getvalue()
    assert question == chosen
    assert clarification.decision == decision
    assert clarification.original_question == VAGUE


@pytest.mark.parametrize("answer", [SPECIFIC_ANSWER, None])
def test_cli_asks_nothing_for_a_specific_question_or_a_failed_check(monkeypatch, answer):
    monkeypatch.setattr(cli.discovery_service, "clarify_question", _fake_clarify(answer)[0])
    asked = _answers(monkeypatch)

    question, clarification = cli._clarify(_console(), VAGUE)

    assert asked == []
    assert question == VAGUE
    assert clarification.decision == "keep"
