"""ResearchGenie — console entry point.

    researchgenie          Run setup (first time only), then the research app.
    researchgenie config   Force the setup wizard to run again.
    researchgenie library import [paths...]   Backfill the research library
                                              from existing run folders.
    researchgenie library search <query>      Search past runs and papers.
    researchgenie resume <run folder or id> [--from STAGE]
                                              Resume a failed run from the
                                              failed stage (or STAGE).
    researchgenie cache clear                 Empty the scholarly response cache.
    researchgenie run|status|result ...       Headless, JSON-speaking commands
                                              (researchgenie/headless.py).
    --no-cache (any command)                  Fetch scholarly answers fresh.
    --no-memory (a research run)              Don't let Discovery read the
                                              research library (#19).

Everything downstream of "ready" reuses the existing, already-tested
pipeline (orchestrator.pipeline.run_pipeline) unchanged — this module is
purely the product/UX layer: config, provider readiness, and presentation.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

# Windows' legacy console API can't render some Unicode Rich uses (box-
# drawing characters, ✦, em dashes) — reconfigure stdout and steer Rich away
# from its legacy-console code path so a modern terminal renders correctly.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.prompt import Confirm, IntPrompt, Prompt  # noqa: E402

from orchestrator import decision_brief  # noqa: E402
from orchestrator.pipeline import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT, STAGE_NAMES, PipelineError, find_run_directory, resume_pipeline,
    run_pipeline,
)
from orchestrator.pipeline import discovery_service  # noqa: E402 — for the clarify step
from researchgenie import config as rg_config  # noqa: E402
from researchgenie import ollama_manager as om  # noqa: E402
from researchgenie.setup_wizard import run_setup_wizard  # noqa: E402
from researchgenie.theme import ACCENT, ERROR, MUTED, OK, WARN  # noqa: E402
from researchgenie.tui import (  # noqa: E402
    PipelineView, render_approval_table, render_banner, render_brief, render_completion,
    render_failure, render_provider_status, render_summary,
)
from shared.contracts.pipeline_contract import Clarification, ResearchRequest  # noqa: E402
from shared.utilities import library, response_cache, scholarly_gateway  # noqa: E402

# IEEE and Springer are numbered styles; APA is APA 7 author-date (#26).
_TARGET_FORMATS = ["IEEE", "Springer", "APA"]


def _load_word_budget():
    """word_budget.py is a standalone module inside services/research-writing/
    (not a package) — loaded by explicit file path, the same pattern
    orchestrator/pipeline.py uses for that service, so this CLI never has to
    put services/research-writing on sys.path (and risk colliding with a
    same-named module elsewhere in the project — see DECISIONS.md D-010)."""
    path = ROOT / "services" / "research-writing" / "word_budget.py"
    spec = importlib.util.spec_from_file_location("researchgenie_word_budget", path)
    module = importlib.util.module_from_spec(spec)
    # Register under its own name before exec: word_budget.py's @dataclass
    # usage needs to resolve its own module via sys.modules[__module__]
    # while it is executing, which module_from_spec() alone does not set up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_word_budget = _load_word_budget()
_LENGTH_PRESETS = _word_budget.LENGTH_PRESETS
_DEFAULT_LENGTH_PRESET = _word_budget.DEFAULT_LENGTH_PRESET
# Display order or 1/2/3 prompt choices; kept separate from the dict's
# iteration order so this stays readable regardless of dict definition order.
_LENGTH_CHOICES = ["short", "standard", "detailed"]


def _ensure_ready(console: Console, config: rg_config.Config) -> bool:
    """Re-verifies the saved provider is actually ready right now (the
    Ollama server may not be running yet even though setup previously
    succeeded — e.g. after a reboot) and starts it automatically if not."""
    rg_config.apply_to_environment(config)
    if config.provider == "gemini":
        if not config.gemini_api_key:
            console.print(f"[{WARN}]No Gemini API key configured.[/{WARN}]")
            return False
        return True

    async def check_and_start() -> bool:
        if await om.is_ollama_server_running():
            return await om.is_model_installed(config.ollama_model)
        console.print(f"[{MUTED}]Starting Ollama...[/{MUTED}]")
        om.start_ollama_server()
        for _ in range(15):
            await asyncio.sleep(1)
            if await om.is_ollama_server_running():
                return await om.is_model_installed(config.ollama_model)
        return False

    ready = asyncio.run(check_and_start())
    if not ready:
        console.print(
            f"[{WARN}]Ollama isn't ready. Run `researchgenie config` to fix your setup.[/{WARN}]"
        )
    return ready


def _collect_request(console: Console, memory: bool = True) -> ResearchRequest:
    console.print()
    question = Prompt.ask("[bold]What are you researching?[/bold]")
    while not question.strip():
        question = Prompt.ask("[bold]What are you researching?[/bold]")
    question, clarification = _clarify(console, question)

    corpus_size = IntPrompt.ask(
        "How many papers should the research corpus include? (6-9)", default=8,
    )
    corpus_size = max(6, min(9, corpus_size))

    console.print("Paper format: " + "  ".join(
        f"[{i}] {name}" for i, name in enumerate(_TARGET_FORMATS, start=1)))
    format_choice = Prompt.ask("Select", choices=[str(i) for i in range(1, len(_TARGET_FORMATS) + 1)],
                               default="1")
    target_format = _TARGET_FORMATS[int(format_choice) - 1]

    console.print(
        "\nPaper length: "
        f"[1] Short (~{_LENGTH_PRESETS['short']} words)  "
        f"[2] Standard (~{_LENGTH_PRESETS['standard']} words)  "
        f"[3] Detailed (~{_LENGTH_PRESETS['detailed']} words)"
    )
    length_choice = Prompt.ask("Select", choices=["1", "2", "3"], default="2")
    length_preset = _LENGTH_CHOICES[int(length_choice) - 1]
    target_words = _LENGTH_PRESETS[length_preset]

    return ResearchRequest(
        research_question=question, corpus_size=corpus_size, target_format=target_format,
        max_draft_length=target_words, clarify="ask", clarification=clarification,
        approval="ask", memory=memory,
    )


def parse_approval(answer: str, count: int) -> list[int] | None:
    """The approval step's answer (#17): [] approves as shown, "drop 2,5"
    gives the paper numbers to replace, anything else is None (ask again)."""
    words = answer.strip().lower()
    if words in ("", "y", "yes", "ok", "approve"):
        return []
    numbers = words.removeprefix("drop").replace(",", " ").split() if words.startswith("drop") else []
    if not numbers or not all(n.isdigit() and 1 <= int(n) <= count for n in numbers):
        return None
    return sorted({int(n) for n in numbers})


def _clarify(console: Console, question: str) -> tuple[str, Clarification]:
    """One sharpening question before a 15-minute run. A specific question,
    or a model that gives no usable answer, keeps the question unasked."""
    with console.status(f"[{MUTED}]Checking your question...[/{MUTED}]"):
        clarified = asyncio.run(discovery_service.clarify_question(question))
    if clarified is None or clarified["specific"]:
        return question, Clarification(original_question=question, decision="keep", **(clarified or {}))

    suggested = clarified["suggested_question"]
    console.print(f"\n[{WARN}]This question may be too broad to search well.[/{WARN}]")
    if clarified["missing"]:
        console.print(f"[{MUTED}]Missing: {escape(', '.join(clarified['missing']))}[/{MUTED}]")
    console.print(f"Suggested: [bold]{escape(suggested)}[/bold]")
    console.print("[1] Use the suggestion  [2] Edit it  [3] Keep mine")
    choice = Prompt.ask("Select", choices=["1", "2", "3"], default="1")
    if choice == "1":
        chosen, decision = suggested, "accept"
    elif choice == "2":
        chosen, decision = Prompt.ask("Your question", default=suggested).strip() or suggested, "edit"
    else:
        chosen, decision = question, "keep"
    return chosen, Clarification(original_question=question, decision=decision, **clarified)


def _provider_label(config: rg_config.Config) -> str:
    if config.provider == "gemini":
        return f"Gemini · {config.gemini_model}"
    return f"Ollama · {config.ollama_model}"


async def _run_research(console: Console, request: ResearchRequest | None,
                        trend_advisor=None, run_id: str | None = None,
                        advisor_directory: str | None = None, start=None) -> None:
    """trend_advisor is the optional Stage 0 result, passed only by the
    `researchgenie-advise` flow — see orchestrator/pipeline.py. run_id and
    advisor_directory come with it, so Stage 0's folder becomes this run's.
    start(approve) replaces the pipeline run with another event stream (a resume)."""
    view = PipelineView(include_trend_advisor=trend_advisor is not None)
    result = None
    error: PipelineError | None = None

    async def approve(papers: list[dict], reserve: int) -> dict:
        """The paper approval step (#17), outside the live view while it asks."""
        live.stop()
        render_approval_table(console, papers, reserve)
        while (drop := parse_approval(Prompt.ask(
                "Press Enter to approve, or type e.g. [bold]drop 2,5[/bold]",
                default="", show_default=False), len(papers))) is None:
            console.print(f"[{WARN}]Press Enter, or type drop and the paper numbers.[/{WARN}]")
        live.start()
        return {"approved_by": "cli", "drop": drop}

    start = start or (lambda approve: run_pipeline(
        request, trend_advisor=trend_advisor, run_id=run_id,
        advisor_directory=advisor_directory, approve=approve))
    events = start(approve)

    with Live(view.render(), console=console, refresh_per_second=6) as live:
        try:
            async for event in events:
                run_id = event.get("run_id") or run_id
                if event["type"] == "result":
                    result = event["result"]
                elif event.get("details", {}).get("raw_type") == "related_runs":
                    # Printed above the live view so the folder paths stay on screen.
                    for run in event["details"]["runs"]:
                        live.console.print(f"[{ACCENT}]{escape(library.describe_run(run))}[/{ACCENT}]")
                else:
                    view.ingest(event)
                    live.update(view.render())
        except PipelineError as exc:
            error = exc

    if error is not None:
        # Each completed stage's artifacts are already on disk (see
        # orchestrator/pipeline.py), so the run can pick up where it stopped.
        directory = find_run_directory(run_id) if run_id else None
        render_failure(
            console, stage="unknown", reason=scholarly_gateway.redact(str(error)),
            run_directory=str(directory) if directory
            else str(ROOT / "outputs") + " (see the most recent run folder)",
        )
        if directory:  # the brief's one next action points to the Doctor (#12)
            render_brief(console, decision_brief.build_brief(directory))
        return

    if result is None:
        render_failure(
            console, stage="unknown",
            reason="The pipeline finished without producing a result.",
            run_directory=None,
        )
        return

    artifacts = ["Research draft (Markdown)", "Quality report", "Citation validation report",
                "Reference list", "Run metadata"]
    if Path(result.run_directory, "final", "paper.tex").exists():
        artifacts.insert(0, "LaTeX source (paper.tex)")
    if Path(result.run_directory, "final", "paper.pdf").exists():
        artifacts.insert(0, "Compiled research paper (paper.pdf)")

    draft_meta = result.writing.draft_metadata
    render_completion(
        console, run_directory=result.run_directory,
        overall_score=result.quality_assurance.scores.overall, artifacts=artifacts,
        draft_status=draft_meta.draft_status, length_status=draft_meta.length_status,
        actual_words=draft_meta.actual_words, target_words=draft_meta.target_words,
        evidence_limited_sections=draft_meta.evidence_limited_sections,
        broken_sections=draft_meta.broken_sections,
    )
    render_brief(console, decision_brief.build_brief(Path(result.run_directory)))


def _run_app(console: Console, config: rg_config.Config, memory: bool = True) -> None:
    render_banner(console)
    render_provider_status(console, scholarly_gateway.provider_status())
    console.print(f"[{OK}]Ready to research.[/{OK}]\n")
    request = _collect_request(console, memory)
    render_summary(
        console, question=request.research_question, corpus_size=request.corpus_size,
        target_format=request.target_format, provider_label=_provider_label(config),
        target_words=request.max_draft_length,
    )
    if not Confirm.ask("\n[bold]Start research?[/bold]", default=True):
        console.print(f"[{MUTED}]Cancelled.[/{MUTED}]")
        return
    asyncio.run(_run_research(console, request))


def _library_command(console: Console, args: list[str]) -> None:
    if args[:1] == ["import"]:
        paths = args[1:] or [DEFAULT_OUTPUT_ROOT, ROOT / "docs" / "research-papers"]
        count = library.import_runs(paths)
        console.print(f"[{OK}]Imported {count} run(s) into {escape(str(library.library_path()))}[/{OK}]")
    elif args[:1] == ["search"] and args[1:]:
        query = " ".join(args[1:])
        runs, papers = library.search_runs(query), library.search_papers(query)
        if not runs and not papers:
            console.print(f"[{MUTED}]Nothing in the research library matches that.[/{MUTED}]")
        for run in runs:
            console.print(escape(library.describe_run(run)))
        for paper in papers:
            console.print(f"{paper['year'] or 'n.d.'}  {escape(paper['title'] or '')}  "
                          f"[{MUTED}]doi:{escape(paper['doi'])}[/{MUTED}]")
    else:
        console.print("Usage: researchgenie library import \\[paths...]\n"
                      "       researchgenie library search <query>")
        sys.exit(2)


def _resolve_run(target: str) -> Path | None:
    """A run folder path, or a run id looked up under outputs/."""
    path = Path(target)
    return path if path.is_dir() else find_run_directory(target, DEFAULT_OUTPUT_ROOT)


def _resume_command(console: Console, config: rg_config.Config, args: list[str]) -> None:
    from_stage = None
    if "--from" in args:
        index = args.index("--from")
        from_stage = args[index + 1] if index + 1 < len(args) else ""
        args = args[:index] + args[index + 2:]
    directory = _resolve_run(args[0]) if len(args) == 1 else None
    if directory is None or from_stage == "":
        console.print("Usage: researchgenie resume <run folder or run id> [--from STAGE]\n"
                      f"       STAGE is one of: {', '.join(STAGE_NAMES)}")
        sys.exit(2)
    if not _ensure_ready(console, config):
        sys.exit(1)
    asyncio.run(_run_research(console, None, start=lambda approve: resume_pipeline(
        directory, from_stage, approve=approve)))


def main() -> None:
    console = Console(legacy_windows=False)
    args = sys.argv[1:]

    if "--no-cache" in args:  # any command, anywhere on the line
        args.remove("--no-cache")
        os.environ["RESEARCHGENIE_NO_CACHE"] = "1"
    from researchgenie import headless  # imports this module's helpers
    if headless.is_headless(args):
        sys.exit(headless.main(args))
    if args[:2] == ["cache", "clear"]:
        count = response_cache.clear()
        console.print(f"[{OK}]Removed {count} cached response(s) from "
                      f"{escape(str(response_cache.cache_path()))}[/{OK}]")
        return
    if args[:1] == ["library"]:
        _library_command(console, args[1:])
        return
    if args[:1] == ["resume"]:
        _resume_command(console, rg_config.load(), args[1:])
        return

    force_setup = bool(args) and args[0] == "config"
    if force_setup or not rg_config.is_setup_complete():
        config = run_setup_wizard(console)
        if not config.setup_complete:
            console.print(f"[{ERROR}]Setup did not complete. Run `researchgenie` to try again.[/{ERROR}]")
            sys.exit(1)
        if force_setup:
            return
    else:
        config = rg_config.load()

    if not _ensure_ready(console, config):
        sys.exit(1)

    _run_app(console, config, memory="--no-memory" not in args)


if __name__ == "__main__":
    main()
