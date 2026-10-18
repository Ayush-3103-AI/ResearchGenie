"""Why a run failed, from its event log: one data-driven rule table.

Deterministic, never a model call (spec #1, "Doctor"). Each rule matches the
run's last error message, optionally with a check over the logged events,
and names a failure class the headless CLI turns into an exit code (#10).
The first matching rule wins; no match is an honest "no known cause".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Events = list[dict[str, Any]]


def read_events(run_directory: Path) -> Events:
    """Every line of <run>/events.jsonl; [] when the run has no log yet."""
    log = Path(run_directory) / "events.jsonl"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_error(events: Events) -> str | None:
    """The message of the last error line, unless a later attempt (a resume)
    has logged progress since."""
    error = None
    for event in events:
        if event.get("type") == "error":
            error = event.get("message") or ""
        elif event.get("type") in ("progress", "result"):
            error = None
    return error


def exhausted_hosts(events: Events) -> list[str]:
    """Hosts the gateway reported as exhausted, in the order first reported."""
    hosts = [e["details"]["host"] for e in events
             if (e.get("details") or {}).get("raw_type") == "provider_exhausted"
             and e["details"].get("host")]
    return list(dict.fromkeys(hosts))


@dataclass(frozen=True)
class Rule:
    id: str
    failure_class: str            # model_unavailable | providers_exhausted | no_relevant_papers
    pattern: str                  # regex searched in the last error message
    cause: str                    # plain language, for the student
    when: Callable[[Events], bool] = field(default=lambda events: True)


RULES: tuple[Rule, ...] = (
    Rule("ollama_unreachable", "model_unavailable", r"Could not reach the local Ollama server",
         "The local model server (Ollama) is not running, so the model could not be asked."),
    Rule("model_missing", "model_unavailable", r"is not installed in Ollama",
         "The model this run uses is not downloaded in Ollama yet."),
    Rule("gemini_unavailable", "model_unavailable",
         r"GEMINI_API_KEY is not set|Gemini rejected the configured API key|Could not reach the Gemini API",
         "The Gemini provider could not be used: its API key is missing or rejected, or it is unreachable."),
    Rule("providers_exhausted", "providers_exhausted", r"No papers found in any source",
         "Every literature source was out of budget or rate-limited, so the search found nothing.",
         when=lambda events: bool(exhausted_hosts(events))),
    Rule("all_sources_failed", "providers_exhausted", r"No papers found in any source",
         "Every literature source failed to answer (network trouble or an outage)."),
    Rule("no_relevant_paper", "no_relevant_papers", r"judged relevant to the research question",
         "The search worked, but no paper it found was on topic for the question."),
)


def match(events: Events) -> Rule | None:
    """The first rule matching the run's last error, or None (no known cause,
    or no error at all)."""
    error = last_error(events)
    if error is None:
        return None
    return next((rule for rule in RULES
                 if re.search(rule.pattern, error) and rule.when(events)), None)
