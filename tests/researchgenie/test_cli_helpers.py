"""Tests for ResearchGenie CLI helper logic that doesn't require a live
terminal or interactive input."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie.cli import _provider_label  # noqa: E402
from researchgenie.config import Config  # noqa: E402


def test_provider_label_for_ollama():
    config = Config(provider="ollama", ollama_model="qwen3.5:9b")
    assert _provider_label(config) == "Ollama · qwen3.5:9b"


def test_provider_label_for_gemini():
    config = Config(provider="gemini", gemini_model="gemini-2.5-flash")
    assert _provider_label(config) == "Gemini · gemini-2.5-flash"


def test_the_approval_answer_accepts_enter_or_a_drop_list():
    from researchgenie.cli import parse_approval
    assert parse_approval("", 8) == []
    assert parse_approval("drop 2,5", 8) == [2, 5]
    assert parse_approval("Drop 2, 5 7", 8) == [2, 5, 7]
    assert parse_approval("drop 9", 8) is None
    assert parse_approval("maybe", 8) is None


def test_a_short_budget_offers_the_key_or_the_reduced_plan(monkeypatch):
    from rich.console import Console

    from researchgenie import cli
    from shared.contracts.pipeline_contract import ResearchRequest
    request = ResearchRequest(research_question="Q?")
    console = Console(quiet=True)
    monkeypatch.setattr(cli, "check_budget", lambda request: {"plan": "reduced", "reason": "short"})
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "2")
    assert cli._budget_choice(console, Config(), request).budget_plan == "reduced"

    added = []
    monkeypatch.setattr(cli.Prompt, "ask", lambda *a, **k: "1")
    monkeypatch.setattr(cli, "setup_scholarly_access", lambda console, config: added.append(config))
    monkeypatch.setattr(cli.rg_config, "save", lambda config: None)
    assert cli._budget_choice(console, Config(), request).budget_plan == "auto"  # the run checks again
    assert added
