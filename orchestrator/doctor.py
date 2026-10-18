"""The Doctor (#13): why a run failed, and the one fix that recovers it.

Deterministic, never a model call (spec #1, "Doctor"). RULES is the whole
diagnosis: each row matches the run's last error message and/or a check over
its logged events, the live system status and its final folder, and names a
plain-language cause, a failure class (the headless CLI's exit code, #10)
and a fix. The first matching row wins; no match is an honest "no known
cause" with the last error, never a guess.

A fix reuses what already exists: start Ollama or pull the model
(researchgenie/ollama_manager.py), the key setup (researchgenie config),
waiting out a provider's reset, and resume (#9). Nothing is applied here
without the caller's confirmation; every surface (CLI, headless --yes, the
API and the web button) asks first. Each diagnosis is saved as
<run>/doctor_report.json with the evidence, the chosen fix and its outcome.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from shared.utilities.scholarly_gateway import redact

Events = list[dict[str, Any]]
REPORT_NAME = "doctor_report.json"


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


def _completed(events: Events) -> bool:
    return bool(events) and events[-1].get("type") == "result"


def exhausted_hosts(events: Events) -> list[str]:
    """Hosts the gateway reported as exhausted, in the order first reported."""
    hosts = [e["details"]["host"] for e in events
             if (e.get("details") or {}).get("raw_type") == "provider_exhausted"
             and e["details"].get("host")]
    return list(dict.fromkeys(hosts))


@dataclass
class Context:
    """What a rule may look at."""
    events: Events
    system: dict[str, Any] | None = None   # None: classify from the log alone (headless exit codes)
    run_directory: Path | None = None

    def metadata(self) -> dict[str, Any]:
        path = self.run_directory / "metadata.json" if self.run_directory else None
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path else {}
        except (OSError, ValueError):
            return {}

    def sys(self, key: str) -> Any:
        return (self.system or {}).get(key)


@dataclass(frozen=True)
class Fix:
    id: str
    label: str                      # shown before the user confirms
    automatic: bool = True          # False: the report says what to do; nothing is run
    resume_from: str | None = "failed"   # resume afterwards: "failed" = the failed stage, or a stage name
    interactive_only: bool = False  # needs a prompt (key entry); headless and web report it instead


FIXES = {fix.id: fix for fix in (
    Fix("start_ollama", "Start Ollama, then resume the run from the failed stage."),
    Fix("pull_model", "Download the model in Ollama, then resume the run from the failed stage."),
    Fix("gemini_setup", "Re-enter the Gemini API key with `researchgenie config`, then resume.",
        interactive_only=True),
    Fix("openalex_key", "Add a free OpenAlex API key (takes a minute, `researchgenie config`), "
                        "then resume.", interactive_only=True),
    Fix("wait_for_reset", "Resume the run once the provider's budget has reset (checked first; "
                          "OpenAlex resets at midnight UTC)."),
    Fix("resume", "Resume the run from the failed stage."),
    Fix("resume_writing", "Write the draft again (Writing and every stage after it).",
        resume_from="writing"),
    Fix("rephrase", "Start a new run with a sharper question or added keywords "
                    "(`--clarify auto` suggests one).", automatic=False, resume_from=None),
    Fix("install_latex", "Install tectonic (https://tectonic-typesetting.github.io), then re-run "
                         "Quality Assurance to compile paper.pdf; paper.tex is already in final/.",
        automatic=False, resume_from=None),
)}


@dataclass(frozen=True)
class Rule:
    id: str
    failure_class: str | None  # model_unavailable | providers_exhausted | no_relevant_papers | None
    cause: str                 # plain language, for the student
    fix: str                   # a FIXES key
    pattern: str | None = None  # regex the last error must match; None: any error
    when: Callable[[Context], bool] = field(default=lambda ctx: True)
    completed: bool = False    # a rule for a run that finished (no error), e.g. a partial draft
    quote: tuple[str, ...] = ()  # event raw_types quoted as evidence besides the error line
    fact: Callable[[Context], str] | None = None  # the live-status evidence


def _ollama_down(ctx: Context) -> bool:
    return ctx.sys("provider") == "ollama" and ctx.sys("ollama_running") is False


def _model_missing(ctx: Context) -> bool:
    return ctx.sys("provider") == "ollama" and ctx.sys("ollama_running") and \
        ctx.sys("model_installed") is False


_OLLAMA_FACT = lambda ctx: "Live check: Ollama is not answering right now."  # noqa: E731
_MODEL_FACT = lambda ctx: f"Live check: the model {ctx.sys('model')} is not installed in Ollama."  # noqa: E731

RULES: tuple[Rule, ...] = (
    Rule("ollama_unreachable", "model_unavailable",
         "The local model server (Ollama) is not running, so the model could not be asked.",
         "start_ollama", pattern=r"Could not reach the local Ollama server",
         fact=lambda ctx: _OLLAMA_FACT(ctx) if _ollama_down(ctx) else ""),
    Rule("model_missing", "model_unavailable",
         "The model this run uses is not downloaded in Ollama yet.",
         "pull_model", pattern=r"is not installed in Ollama"),
    Rule("gemini_unavailable", "model_unavailable",
         "The Gemini provider could not be used: its API key is missing or rejected, or it is unreachable.",
         "gemini_setup",
         pattern=r"GEMINI_API_KEY is not set|Gemini rejected the configured API key|Could not reach the Gemini API"),
    Rule("openalex_budget_no_key", "providers_exhausted",
         "OpenAlex's free daily budget (about 100 searches) is spent and no OpenAlex key is set, "
         "and no other source found papers.",
         "openalex_key", pattern=r"No papers found in any source",
         when=lambda ctx: "api.openalex.org" in exhausted_hosts(ctx.events)
         and ctx.sys("openalex_key") is False,
         quote=("provider_exhausted", "source_error"),
         fact=lambda ctx: "Live check: no OpenAlex API key is configured."),
    Rule("providers_exhausted", "providers_exhausted",
         "Every literature source was out of budget or rate-limited, so the search found nothing.",
         "wait_for_reset", pattern=r"No papers found in any source",
         when=lambda ctx: bool(exhausted_hosts(ctx.events)), quote=("provider_exhausted", "source_error")),
    Rule("all_sources_failed", "providers_exhausted",
         "Every literature source failed to answer (network trouble or an outage).",
         "resume", pattern=r"No papers found in any source", quote=("source_error",)),
    Rule("no_relevant_paper", "no_relevant_papers",
         "The search worked, but no paper it found was on topic for the question.",
         "rephrase", pattern=r"judged relevant to the research question", quote=("relevance_filter",)),
    # Any other error while the model is down now: after every specific cause
    # above, so a known cause is never blamed on a model that is merely off.
    Rule("ollama_stopped", "model_unavailable",
         "Ollama is not running now; the run most likely failed because it stopped mid-run.",
         "start_ollama", when=_ollama_down, fact=_OLLAMA_FACT),
    Rule("model_missing_now", "model_unavailable",
         "Ollama is running, but the model this run uses is not installed.",
         "pull_model", when=_model_missing, fact=_MODEL_FACT),
    Rule("partial_draft", None,
         "The run finished, but the draft is incomplete: some sections are missing or broken.",
         "resume_writing", completed=True,
         when=lambda ctx: ctx.metadata().get("draft_status") in ("partial", "failed"),
         fact=lambda ctx: f"metadata.json: draft_status = {ctx.metadata().get('draft_status')}, "
                          f"broken sections = {ctx.metadata().get('broken_sections') or []}"),
    Rule("missing_latex", None,
         "The run finished, but no PDF was compiled: no LaTeX toolchain (tectonic or pdflatex) "
         "is installed.",
         "install_latex", completed=True,
         when=lambda ctx: ctx.run_directory is not None
         and not (ctx.run_directory / "final" / "paper.pdf").is_file() and ctx.sys("latex") is False,
         fact=lambda ctx: "Live check: neither tectonic nor pdflatex is on PATH; final/paper.pdf is missing."),
)


def match(events: Events, system: dict[str, Any] | None = None,
          run_directory: Path | None = None) -> Rule | None:
    """The first rule matching the run, or None (no known cause)."""
    ctx = Context(events, system, run_directory)
    error = last_error(events)
    completed = _completed(events)
    for rule in RULES:
        if rule.completed:
            applies = completed
        else:
            applies = error is not None and (rule.pattern is None or re.search(rule.pattern, error))
        if applies and rule.when(ctx):
            return rule
    return None


def _line(event: dict[str, Any]) -> str:
    if event.get("type") == "error":
        return f"error: {event.get('message')}"
    return f"{event.get('timestamp', '')} [{event.get('stage', '')}] {event.get('message', '')}".strip()


async def system_status() -> dict[str, Any]:
    """The live facts the rules check. Never carries a key, only whether one is set."""
    from researchgenie import ollama_manager as om
    provider = os.getenv("AI_PROVIDER", "ollama")
    model = os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")
    running = await om.is_ollama_server_running() if provider == "ollama" else None
    return {
        "provider": provider, "model": model, "ollama_running": running,
        "model_installed": await om.is_model_installed(model) if running else None,
        "latex": bool(shutil.which("tectonic") or shutil.which("pdflatex")),
        "openalex_key": bool(os.getenv("OPENALEX_API_KEY")),
        "gemini_key": bool(os.getenv("GEMINI_API_KEY")),
    }


def diagnose(run_directory: Path, system: dict[str, Any]) -> dict[str, Any]:
    """The diagnosis of one run folder (also the body of doctor_report.json)."""
    run_directory = Path(run_directory)
    events = read_events(run_directory)
    rule = match(events, system, run_directory)
    error = last_error(events)
    diagnosis: dict[str, Any] = {
        "run_id": run_directory.name.partition("_")[2],
        "run_directory": str(run_directory),
        "diagnosed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "state": "completed" if _completed(events) else "failed" if error is not None else "running",
        "last_error": error, "system": system,
    }
    if rule is None:
        diagnosis.update(
            rule=None, failure_class="unknown", fix=None,
            cause=("No known cause matches this run's last error; nothing is guessed. "
                   "The last error is quoted below." if error is not None else
                   "This run has not failed: it is still going, or it stopped without logging "
                   "an error." if not _completed(events) else
                   "This run completed and no known problem was found."),
            evidence=[f"error: {error}"] if error is not None else [])
        return diagnosis
    ctx = Context(events, system, run_directory)
    evidence = [_line(e) for e in events
                if (e.get("details") or {}).get("raw_type") in rule.quote][-8:]
    if error is not None:
        evidence.append(f"error: {error}")
    if rule.fact and (fact := rule.fact(ctx)):
        evidence.append(fact)
    fix = FIXES[rule.fix]
    diagnosis.update(rule=rule.id, failure_class=rule.failure_class, cause=rule.cause,
                     evidence=evidence,
                     fix={"id": fix.id, "label": fix.label, "automatic": fix.automatic,
                          "interactive_only": fix.interactive_only, "resume_from": fix.resume_from})
    return diagnosis


def write_report(run_directory: Path, diagnosis: dict[str, Any], *, applied: bool,
                 outcome: str) -> Path:
    path = Path(run_directory) / REPORT_NAME
    report = {**diagnosis, "applied": applied, "outcome": outcome}
    path.write_text(redact(json.dumps(report, indent=2, ensure_ascii=False)), encoding="utf-8")
    return path


def resume_stage(diagnosis: dict[str, Any]) -> str | None:
    """The stage a fix resumes from: None means the failed stage (resume's default)."""
    stage = (diagnosis.get("fix") or {}).get("resume_from")
    return None if stage == "failed" else stage


async def apply_fix(diagnosis: dict[str, Any], *, key_setup: Callable[[], bool] | None = None,
                    wait_seconds: int = 15) -> str | None:
    """Carries out the fix's own step (before any resume). Returns None when
    it worked, else why not. The caller has already asked for confirmation.
    key_setup is the interactive key prompt; without one (headless, web) a
    key fix is reported, not run."""
    from researchgenie import ollama_manager as om
    from shared.utilities import scholarly_gateway
    fix = diagnosis.get("fix") or {}
    if not fix:
        return "There is no fix for an unknown cause."
    if not fix["automatic"]:
        return f"This fix is a step for you: {fix['label']}"
    if fix["id"] == "start_ollama":
        if not om.start_ollama_server():
            return "Ollama could not be started (is it installed and on PATH?)."
        for _ in range(wait_seconds):
            if await om.is_ollama_server_running():
                return None
            await asyncio.sleep(1)
        return f"Ollama was started but did not answer within {wait_seconds} s."
    if fix["id"] == "pull_model":
        model = (diagnosis.get("system") or {}).get("model") or os.getenv("LOCAL_AI_MODEL", "qwen3.5:9b")
        return None if await om.pull_model(model) else f"Downloading {model} failed."
    if fix["interactive_only"]:
        if key_setup is None:
            return f"This fix needs you at a prompt: {fix['label']}"
        return None if key_setup() else "Key setup did not finish."
    if fix["id"] == "wait_for_reset":
        still = [h for h in exhausted_hosts(read_events(Path(diagnosis["run_directory"])))
                 if scholarly_gateway._gateway.exhausted(h)]
        if still:
            return f"Still out of budget: {', '.join(still)}. Try again after the reset (midnight UTC)."
        return None
    return None  # resume / resume_writing: the resume is the fix


async def resume_and_record(run_directory: Path, diagnosis: dict[str, Any],
                            resume: Callable[..., AsyncIterator[dict[str, Any]]],
                            approve: Callable | None = None) -> AsyncIterator[dict[str, Any]]:
    """The fix's resume: yields its events and writes the outcome into
    doctor_report.json when it ends. `resume` is orchestrator.pipeline.resume_pipeline."""
    try:
        async for event in resume(run_directory, resume_stage(diagnosis), approve=approve):
            yield event
    except Exception as error:
        write_report(run_directory, diagnosis, applied=True,
                     outcome=f"Fix applied; the resumed run failed again: {redact(str(error)) or type(error).__name__}")
        raise
    write_report(run_directory, diagnosis, applied=True, outcome="Fix applied; the resumed run completed.")
