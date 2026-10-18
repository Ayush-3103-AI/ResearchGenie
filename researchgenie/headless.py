"""Headless, JSON-speaking CLI (#10), for agents, scripts and the skill.

    researchgenie run -q "question" [--every-request-field ...] [--length short]
                      [--no-cache] [--detach] [--json]
    researchgenie status <run id or folder> [--json]
    researchgenie result <run id or folder> [--json]
    researchgenie resume <run id or folder> [--from STAGE] (--json | --detach)
    researchgenie doctor <run id or folder> [--yes] [--json]

Never prompts. Progress goes to stderr, the answer to stdout (JSON with
--json). `run --detach` starts the same command in a background process and
prints the run id at once. Command lines are fixed: no shell, no inline
code, user input only ever travels as argv items or pydantic-validated
fields. Exit codes (EXIT_CODES) name the failure class from
orchestrator/doctor.py's rule table.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import typing
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from pydantic import ValidationError
from rich.console import Console
from rich.prompt import Confirm

from orchestrator import doctor
from orchestrator.pipeline import (
    DEFAULT_OUTPUT_ROOT, FINAL_FILES, STAGE_NAMES, find_run_directory, resume_pipeline,
    run_pipeline,
)
from researchgenie import config as rg_config
from shared.contracts.pipeline_contract import ResearchRequest
from shared.utilities.scholarly_gateway import redact

EXIT_OK = 0
EXIT_FAILED = 1            # a failure with no more specific class
EXIT_USAGE = 2             # bad arguments (argparse uses 2 as well)
EXIT_MODEL_UNAVAILABLE = 3
EXIT_PROVIDERS_EXHAUSTED = 4
EXIT_NO_RELEVANT_PAPERS = 5
EXIT_NOT_FOUND = 6         # no run with that id or folder
EXIT_NOT_FINISHED = 7      # `result` asked of a run still going
EXIT_CODES = {"model_unavailable": EXIT_MODEL_UNAVAILABLE,
              "providers_exhausted": EXIT_PROVIDERS_EXHAUSTED,
              "no_relevant_papers": EXIT_NO_RELEVANT_PAPERS}

COMMANDS = ("run", "status", "result", "doctor")
# ResearchRequest fields a flag can't set: the clarify outcome is recorded by
# an interactive surface, never typed in.
_NOT_FLAGS = {"clarification"}


def _year_range(text: str) -> list[int]:
    start, sep, end = text.partition("-")
    if not (sep and start.isdigit() and end.isdigit()):
        raise argparse.ArgumentTypeError("use START-END, e.g. 2015-2024")
    return [int(start), int(end)]


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    """One flag per ResearchRequest field, so the CLI never falls behind the
    contract; pydantic validates the values."""
    for name, info in ResearchRequest.model_fields.items():
        if name in _NOT_FLAGS:
            continue
        flags = ["--" + name.replace("_", "-")]
        options: dict[str, Any] = {"dest": name, "default": None}
        if name == "research_question":
            flags = ["-q", "--question", *flags]
        elif name == "target_format":
            flags = ["--format", *flags]
        if name == "year_range":
            options.update(type=_year_range, metavar="START-END")
        elif typing.get_origin(info.annotation) is list:
            options.update(action="append", metavar=name.rstrip("s").upper(),
                           help="repeat for more than one")
        elif typing.get_origin(info.annotation) is typing.Literal:
            # "ask" (clarify, approval) needs someone to answer; never headless.
            options["choices"] = [c for c in typing.get_args(info.annotation) if c != "ask"]
        parser.add_argument(*flags, **options)
    parser.add_argument("--length", choices=("short", "standard", "detailed"),
                        help="a paper length preset (sets --max-draft-length)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchgenie", description="Headless ResearchGenie.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start a research run without prompts")
    _add_request_options(run)
    run.add_argument("--detach", action="store_true", help="print the run id and return at once")
    run.add_argument("--run-id", help=argparse.SUPPRESS)  # set by --detach for the child

    for name, text in (("status", "stage, state and last event"),
                       ("result", "final files, scores, verification counts, draft status")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("run", help="run id or run folder")

    resume = commands.add_parser("resume", help="resume a failed run from the failed stage")
    resume.add_argument("run", help="run id or run folder")
    resume.add_argument("--from", dest="from_stage", choices=STAGE_NAMES)
    resume.add_argument("--detach", action="store_true")

    doctor_cmd = commands.add_parser("doctor", help="diagnose a run; apply the fix once confirmed")
    doctor_cmd.add_argument("run", help="run id or run folder")
    doctor_cmd.add_argument("--yes", action="store_true",
                            help="apply the proposed fix without asking (required with --json)")

    for sub in commands.choices.values():
        sub.add_argument("--json", action="store_true", help="machine-readable output")
        sub.add_argument("--no-cache", action="store_true",
                         help="don't answer scholarly requests from the local cache")
    return parser


def is_headless(args: list[str]) -> bool:
    """run/status/result always; resume only with --json or --detach (a bare
    `resume` stays the interactive one)."""
    if args[:1] == ["resume"]:
        return "--json" in args or "--detach" in args
    return bool(args) and args[0] in COMMANDS


def _request(args: argparse.Namespace) -> ResearchRequest:
    fields = {name: value for name, value in vars(args).items()
              if name in ResearchRequest.model_fields and value is not None}
    if args.length and "max_draft_length" not in fields:
        from researchgenie.cli import _LENGTH_PRESETS
        fields["max_draft_length"] = _LENGTH_PRESETS[args.length]
    return ResearchRequest(**fields)


def _resolve(target: str) -> Path | None:
    path = Path(target)
    return path if (path / "events.jsonl").is_file() or (path / "00_request.json").is_file() \
        else find_run_directory(target, DEFAULT_OUTPUT_ROOT)


def _detached_log(run_id: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f".detached-{run_id}.log"


def status_of(run_directory: Path) -> dict[str, Any]:
    events = doctor.read_events(run_directory)
    state = "running"
    for event in events:  # the last word wins: a resume logs progress after an error
        if event.get("type") == "result":
            state = "completed"
        elif event.get("type") == "error":
            state = "failed"
        elif event.get("type") == "progress":
            state = "running"
    last = next((e for e in reversed(events) if e.get("type") in ("progress", "error")), {})
    # An error line has no stage of its own: it failed in the last one that ran.
    stage = next((e["stage"] for e in reversed(events) if e.get("stage")), None)
    status: dict[str, Any] = {
        "run_id": run_directory.name.partition("_")[2], "state": state,
        "stage": stage, "run_directory": str(run_directory),
        "last_event": {k: last[k] for k in ("type", "stage", "status", "message", "timestamp")
                       if k in last},
    }
    if state == "failed":
        rule = doctor.match(events)
        status["error"] = doctor.last_error(events)
        status["failure_class"] = rule.failure_class if rule else "unknown"
    return status


def result_of(run_directory: Path) -> dict[str, Any]:
    status = status_of(run_directory)
    events = doctor.read_events(run_directory)
    final = next((e["result"] for e in reversed(events) if e.get("type") == "result"), None)
    out = {**status, "final_files": {name: str(run_directory / "final" / name)
                                     for name in FINAL_FILES
                                     if (run_directory / "final" / name).is_file()}}
    if status["state"] == "completed" and final:
        draft = final["writing"]["draft_metadata"]
        verification = final["verification"]
        out.update(
            scores=final["quality_assurance"]["scores"],
            verification={"verified": len(verification["verified_references"]),
                          "invalid": len(verification["invalid_references"]),
                          "unverifiable": len(verification["unverifiable_references"])},
            draft_status=draft["draft_status"], length_status=draft.get("length_status"),
            actual_words=draft.get("actual_words"), target_words=draft.get("target_words"),
        )
    return out


def _exit_code(report: dict[str, Any]) -> int:
    if report["state"] == "completed":
        return EXIT_OK
    if report["state"] == "failed":
        return EXIT_CODES.get(report.get("failure_class"), EXIT_FAILED)
    return EXIT_NOT_FINISHED


def _print(data: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return
    for key, value in data.items():
        shown = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        print(f"{key}: {shown}")


async def _drive(events) -> str | None:
    """Runs an event stream to the end, progress to stderr. The error message, or None."""
    try:
        async for event in events:
            if event.get("type") == "progress":
                print(f"[{event['stage']}] {event['message']}", file=sys.stderr, flush=True)
    except Exception as error:  # PipelineError, a model error, or anything a stage raised
        return redact(str(error)) or type(error).__name__
    return None


def _detach(argv: list[str], run_id: str) -> None:
    """Starts this same command in the background, without --detach."""
    rest = [a for a in argv if a != "--detach"]
    if rest[:1] == ["run"]:
        rest += ["--run-id", run_id]
    elif "--json" not in rest:
        rest.append("--json")  # without it the child would be the interactive resume
    child = [sys.executable, "-m", "researchgenie.cli", *rest]
    log = _detached_log(run_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log.open("ab"),
                               "stderr": subprocess.STDOUT}
    if sys.platform == "win32":
        options["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                                    | subprocess.CREATE_NO_WINDOW)
    else:
        options["start_new_session"] = True
    subprocess.Popen(child, **options)


def _ready() -> bool:
    """The model is usable now (starting Ollama if it isn't running), with
    the checks' messages on stderr."""
    from researchgenie.cli import _ensure_ready
    return _ensure_ready(Console(stderr=True, legacy_windows=False), rg_config.load())


def _key_setup(fix_id: str) -> Callable[[], bool]:
    """The interactive key prompt a key fix runs (the same steps as `researchgenie config`)."""
    def setup() -> bool:
        from researchgenie.setup_wizard import run_setup_wizard, setup_scholarly_access
        console = Console(legacy_windows=False)
        if fix_id == "gemini_setup":
            config = run_setup_wizard(console)
        else:
            config = rg_config.load()
            setup_scholarly_access(console, config)
            rg_config.save(config)
        rg_config.apply_to_environment(config)
        return config.setup_complete or fix_id != "gemini_setup"
    return setup


async def _doctor(directory: Path, args: argparse.Namespace) -> int:
    """Diagnose, ask (or take --yes), apply, resume; the report records each step."""
    diagnosis = doctor.diagnose(directory, await doctor.system_status())
    fix = diagnosis["fix"]
    if not args.json:
        print(f"Cause: {diagnosis['cause']}")
        for line in diagnosis["evidence"]:
            print(f"  {line}")
        if fix:
            print(f"Proposed fix: {fix['label']}")
    if fix is None:
        applied, outcome = False, "No fix proposed."
    elif not fix["automatic"]:
        applied, outcome = False, f"No automatic fix; to do: {fix['label']}"
    elif not (args.yes or (not args.json and Confirm.ask("Apply this fix?", default=False))):
        applied, outcome = False, "Not applied: not confirmed" + (" (headless needs --yes)." if args.json else ".")
    else:
        problem = await doctor.apply_fix(
            diagnosis, key_setup=None if args.json else _key_setup(fix["id"]))
        if problem:
            applied, outcome = False, problem
        else:
            await _drive(doctor.resume_and_record(directory, diagnosis, resume_pipeline))
            report = json.loads((directory / doctor.REPORT_NAME).read_text(encoding="utf-8"))
            _print({**report, "report_path": str(directory / doctor.REPORT_NAME)}, args.json)
            return _exit_code(result_of(directory))
    path = doctor.write_report(directory, diagnosis, applied=applied, outcome=outcome)
    _print({**diagnosis, "applied": applied, "outcome": outcome, "report_path": str(path)}, args.json)
    return _exit_code(result_of(directory))


def main(argv: list[str]) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as stop:  # argparse already printed why
        return EXIT_USAGE if stop.code else EXIT_OK
    if args.no_cache:
        os.environ["RESEARCHGENIE_NO_CACHE"] = "1"
    rg_config.apply_scholarly(rg_config.load())

    def fail(code: int, message: str, **extra) -> int:
        _print({"error": message, **extra}, args.json)
        return code

    if args.command == "run":
        try:
            request = _request(args)
        except ValidationError as error:
            return fail(EXIT_USAGE, str(error))
        run_id = args.run_id or uuid4().hex
        if args.detach:
            _detach(argv, run_id)
            _print({"run_id": run_id, "state": "started",
                    "log": str(_detached_log(run_id))}, args.json)
            return EXIT_OK
        if not _ready():
            return fail(EXIT_MODEL_UNAVAILABLE, "The configured model is not ready.",
                        run_id=run_id, failure_class="model_unavailable")
        error = asyncio.run(_drive(run_pipeline(request, run_id=run_id)))
        directory = find_run_directory(run_id, DEFAULT_OUTPUT_ROOT)
        if directory is None:  # failed before the run folder existed
            return fail(EXIT_FAILED, error or "The run left no folder.", run_id=run_id)
        report = result_of(directory)
        _print(report, args.json)
        return _exit_code(report)

    directory = _resolve(args.run)
    if directory is None:
        if args.command == "status" and _detached_log(args.run).is_file():
            _print({"run_id": args.run, "state": "starting",
                    "log": str(_detached_log(args.run))}, args.json)
            return EXIT_OK
        return fail(EXIT_NOT_FOUND, f"No run {args.run!r} under {DEFAULT_OUTPUT_ROOT}.")

    if args.command == "doctor":
        return asyncio.run(_doctor(directory, args))
    if args.command == "status":
        _print(status_of(directory), args.json)
        return EXIT_OK
    if args.command == "result":
        report = result_of(directory)
        _print(report, args.json)
        return _exit_code(report)

    # resume
    if args.detach:
        _detach(argv, directory.name.partition("_")[2])
        _print({"run_id": directory.name.partition("_")[2], "state": "resuming"}, args.json)
        return EXIT_OK
    if not _ready():
        return fail(EXIT_MODEL_UNAVAILABLE, "The configured model is not ready.",
                    failure_class="model_unavailable")
    logged = len(doctor.read_events(directory))
    error = asyncio.run(_drive(resume_pipeline(directory, args.from_stage)))
    if error and len(doctor.read_events(directory)) == logged:
        return fail(EXIT_FAILED, error)  # refused (e.g. a corrupt saved stage) before it started
    report = result_of(directory)
    _print(report, args.json)
    return _exit_code(report)
