"""Pipeline orchestrator: runs Services 1-4 in sequence for one research
request, persisting each stage's artifacts under outputs/<slug>_<run_id>/
and emitting standardized progress events.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.contracts.discovery_contract import DiscoveryRequest, DiscoveryResult  # noqa: E402
from shared.contracts.pipeline_contract import (  # noqa: E402
    Clarification, PipelineResult, ResearchRequest, StageTimings,
)
from shared.contracts.qa_contract import QualityAssuranceRequest, QualityAssuranceResult  # noqa: E402
from shared.contracts.trend_contract import (  # noqa: E402
    TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.contracts.verification_contract import VerificationRequest, VerificationResult  # noqa: E402
from shared.contracts.writing_contract import WritingRequest, WritingResult  # noqa: E402
from shared.utilities import evidence_table, latex_export, library, scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import redact  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402


def _load_service(name: str, relative_path: str):
    """Each service's entry point is named service.py and reached only via
    sys.path insertion (see DECISIONS.md D-010/D-012/D-013's recurring
    module-name-collision note) — load every one by explicit file path
    under a unique name so the orchestrator can safely import all four
    together in one process."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


discovery_service = _load_service(
    "orchestrator_discovery_service", "services/research-discovery/service.py")
writing_service = _load_service(
    "orchestrator_writing_service", "services/research-writing/service.py")
verification_service = _load_service(
    "orchestrator_verification_service", "services/verification/service.py")
qa_service = _load_service(
    "orchestrator_qa_service", "services/quality-assurance/service.py")

# Loaded on first use rather than beside the four above: Stage 0 is optional,
# and its module pulls in the gap-mining swarm and the local model adapter
# that a normal four-stage run never touches.
trend_advisor_service = None


def _trend_advisor_service():
    global trend_advisor_service
    if trend_advisor_service is None:
        trend_advisor_service = _load_service(
            "orchestrator_trend_advisor_service", "services/trend-advisor/service.py")
    return trend_advisor_service


DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
# The files the pipeline writes to <run>/final/: the only ones the API,
# the headless CLI and the MCP server hand out.
FINAL_FILES = ("draft.md", "quality_report.md", "validation_report.md",
               "references.md", "evidence_table.md", "paper.tex", "paper.pdf")

_STAGE_EMOJI = {
    "trend_advisor": "📈",
    "discovery": "🔍",
    "writing": "✍️",
    "verification": "🔗",
    "quality_assurance": "🛡️",
}


class PipelineError(RuntimeError):
    """Raised when a stage fails to produce output; the orchestrator stops
    rather than passing partial/fabricated data to the next stage."""


# Each stage, the folder its result.json is saved in, and the contract that
# reads it back on resume (#9), in pipeline order.
STAGES = (
    ("discovery", "01_discovery", DiscoveryResult),
    ("writing", "02_writing", WritingResult),
    ("verification", "03_verification", VerificationResult),
    ("quality_assurance", "04_quality_assurance", QualityAssuranceResult),
)
STAGE_NAMES = tuple(name for name, _, _ in STAGES)


def find_run_directory(run_id: str, output_root: Path | None = None) -> Path | None:
    """The run folder <slug>_<run_id> under output_root, if there is one.
    Slugs never contain "_", so the id is everything after the first one."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    if not output_root.is_dir():
        return None
    return next((d for d in output_root.iterdir()
                 if d.is_dir() and d.name.partition("_")[2] == run_id), None)


def _load_saved(path: Path, contract, what: str):
    try:
        return contract.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PipelineError(f"Cannot resume: the saved {what} result is missing ({path}).") from None
    except (OSError, ValueError) as error:  # pydantic's ValidationError is a ValueError
        reason = (str(error).splitlines() or [type(error).__name__])[0]
        raise PipelineError(
            f"Cannot resume: the saved {what} result is corrupt ({path}): {reason}") from None


def plan_resume(run_directory: Path, from_stage: str | None = None) -> dict[str, Any]:
    """Everything a resume needs, loaded and checked up front so a bad run
    folder fails before anything is moved or re-run. from_stage defaults to
    the first stage without a saved result."""
    run_directory = Path(run_directory)
    request = _load_saved(run_directory / "00_request.json", ResearchRequest, "request")
    if from_stage is None:
        from_stage = next((name for name, folder, _ in STAGES
                           if not (run_directory / folder / "result.json").is_file()), None)
        if from_stage is None:
            raise PipelineError("Every stage of this run already has a saved result; "
                                "name the stage to re-run (e.g. quality_assurance).")
    if from_stage not in STAGE_NAMES:
        raise PipelineError(f"Unknown stage {from_stage!r}; choose one of {', '.join(STAGE_NAMES)}.")
    loaded = {name: _load_saved(run_directory / folder / "result.json", contract, name)
              for name, folder, contract in STAGES[:STAGE_NAMES.index(from_stage)]}
    advisor = run_directory / "00_trend_advisor" / "result.json"
    return {
        "request": request, "from_stage": from_stage, "loaded": loaded,
        "run_id": run_directory.name.partition("_")[2], "output_root": run_directory.parent,
        "trend_advisor": _load_saved(advisor, TrendAdvisorResult, "trend advisor")
        if advisor.is_file() else None,
    }


async def resume_pipeline(run_directory: Path | str, from_stage: str | None = None,
                          approve: Callable | None = None) -> AsyncIterator[dict[str, Any]]:
    """Resumes a run folder from `from_stage` (default: the first stage
    without a saved result). Earlier stages are loaded through their
    contracts, not re-run; the events append to the same events.jsonl.
    approve is the approval step's callback, used when Discovery re-runs."""
    plan = plan_resume(Path(run_directory), from_stage)
    async for event in run_pipeline(
        plan["request"], output_root=plan["output_root"], run_id=plan["run_id"],
        trend_advisor=plan["trend_advisor"], resume=(plan["from_stage"], plan["loaded"]),
        approve=approve,
    ):
        yield event


def _final_versions(run_directory: Path) -> list[str]:
    numbered = [p.name for p in run_directory.glob("final_v*") if p.name[7:].isdigit()]
    return sorted(numbered, key=lambda name: int(name[7:]))


def _keep_final_as_version(run_directory: Path) -> str | None:
    """Moves an existing final/ to final_vN (the next free N) so a resume
    never overwrites output; returns the new name, or None."""
    final = run_directory / "final"
    if not final.is_dir():
        return None
    versions = _final_versions(run_directory)
    name = f"final_v{int(versions[-1][7:]) + 1 if versions else 1}"
    final.rename(run_directory / name)
    return name


def _loaded_event(run_id: str, stage: str, service: str) -> dict[str, Any]:
    return _progress_event(
        run_id=run_id, stage=stage, service=service, status="done",
        title=f"{stage.replace('_', ' ').title()} loaded",
        message=f"{stage.replace('_', ' ').capitalize()} result loaded from disk (not re-run).",
        details={"raw_type": "stage_loaded"},
    )


def _progress_event(
    *, run_id: str, stage: str, service: str, status: str, title: str,
    message: str, details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "progress",
        "run_id": run_id,
        "stage": stage,
        "service": service,
        "status": status,
        "emoji": _STAGE_EMOJI.get(stage, "•"),
        "title": title,
        "message": message,
        "details": details or {},
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


_PASSED_THROUGH = ("host", "wait_seconds", "remedy", "papers", "reserve", "approved_by",
                   "approved_at", "dropped", "replacements", "round", "reason",
                   "accepted_before", "accepted_after", "hits", "mode", "plan")


def _adapt_inner_event(run_id: str, stage: str, service: str, inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap one of a service's own progress events (each service emits its
    own ad hoc shape — see each service.py) into the standard format."""
    inner_type = inner.get("type", "status")
    state = inner.get("state")
    if state in ("running", "done", "failed"):
        status = state
    elif inner_type in ("source_error", "provider_exhausted"):
        status = "warning"
    else:
        status = "running"
    message = inner.get("message") or inner_type
    title = message if len(message) <= 60 else f"{stage.replace('_', ' ').title()} update"
    # The scholarly gateway's rate_limited / provider_exhausted events carry
    # their host, wait and remedy through to the standard shape; the approval
    # step (#17) its papers and decision, so the event log records them.
    details = {"raw_type": inner_type,
               **{k: inner[k] for k in _PASSED_THROUGH if k in inner}}
    return _progress_event(
        run_id=run_id, stage=stage, service=service, status=status,
        title=title, message=message, details=details,
    )


def _library_warning(run_id: str, stage: str, error: Exception) -> dict[str, Any]:
    return _progress_event(
        run_id=run_id, stage=stage, service="research_library", status="warning",
        title="Research library unavailable",
        message=f"The research library could not be used ({str(error) or type(error).__name__}); "
                "the run continues without it.",
        details={"raw_type": "library_error"},
    )


REDUCED_PLAN = ("verification searches use Crossref instead of OpenAlex. Add your free OpenAlex "
                "key (`researchgenie config`, takes 1 minute) for the full plan.")


def check_budget(request: ResearchRequest) -> dict[str, Any]:
    """The budget check before Discovery (#7): the run's planned OpenAlex
    credits — the first search and the re-plan's, one verification search
    per candidate, one DOI filter for Citation Verification — against the
    balance known from earlier requests. The plan is "reduced" when the user
    chose it, or (budget_plan "auto") when the balance can't cover the run;
    verification_searches is how many OpenAlex verification searches it allows."""
    cost = scholarly_gateway.CREDIT_COSTS
    candidates = discovery_service.MAX_CANDIDATES_FOR_ANALYSIS
    fixed = 2 * cost["search"] + cost["list"]
    estimated = fixed + candidates * cost["search"]
    balance = scholarly_gateway.balance()
    short = balance is not None and balance < estimated
    plan = "reduced" if request.budget_plan == "reduced" or (
        short and request.budget_plan == "auto") else "full"
    reason = ("No OpenAlex balance is known yet; the first request reports it." if balance is None
              else f"OpenAlex has {balance} credit(s) left today; this run may use up to {estimated}.")
    if request.budget_plan != "auto":
        reason += f" The {plan} plan was chosen."
    allowed = (0 if plan == "reduced" else candidates if balance is None
               else max(0, min(candidates, (balance - fixed) // cost["search"])))
    return {"estimated_credits": estimated, "balance": balance, "plan": plan, "reason": reason,
            "verification_searches": allowed}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_pipeline(
    request: ResearchRequest, output_root: Path | None = None, run_id: str | None = None,
    **stage_options: Any,
) -> AsyncIterator[dict[str, Any]]:
    """The entry point every surface (CLI, API, web) uses: runs _run_stages
    (see its docstring for the events and stage_options) and appends each
    event to <run_dir>/events.jsonl before passing it on unchanged, plus an
    {"type": "error"} line if it raises. One line per event, written and
    closed as it arrives, so a crash keeps everything logged so far;
    run_registry rebuilds past runs from this file after a restart.

    The log is opened per event rather than up front: _run_stages may
    rename the Stage 0 folder into the run folder, which it only does if
    the run folder doesn't exist yet, and it has created the folder by the
    time it yields its first event."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = run_id or uuid4().hex
    # Clarify: an interactive surface already asked ("ask" + clarification);
    # "auto" asks the model here and takes its suggestion. Before the folder
    # and the log path are named, so it carries the question actually researched.
    if request.clarify == "auto" and request.clarification is None:
        clarified = await discovery_service.clarify_question(request.research_question)
        accept = clarified is not None and not clarified["specific"]
        request = request.model_copy(update={
            "research_question": clarified["suggested_question"] if accept else request.research_question,
            "clarification": Clarification(original_question=request.research_question,
                                           decision="accept" if accept else "keep", **(clarified or {})),
        })
    log_path = output_root / f"{slugify(request.research_question)}_{run_id}" / "events.jsonl"

    def append(event: dict[str, Any]) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=lambda o: o.model_dump(mode="json"))
        with log_path.open("a", encoding="utf-8") as log:
            log.write(redact(line) + "\n")  # no key or email in a run folder (#5, D-023)

    try:
        async for event in _run_stages(request, output_root=output_root, run_id=run_id, **stage_options):
            append(event)
            yield event
    except Exception as error:
        # Same message run_registry reports for a failed run.
        append({"type": "error", "message": str(error) or type(error).__name__})
        raise


async def _run_stages(
    request: ResearchRequest, output_root: Path | None = None, run_id: str | None = None,
    trend_advisor: TrendAdvisorResult | None = None,
    advisor_directory: Path | str | None = None,
    resume: tuple[str, dict[str, Any]] | None = None,
    approve: Callable | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yields standardized progress events, then a final
    {"type": "result", "result": PipelineResult} event. Raises
    PipelineError if any stage fails to produce a structured result.

    run_id is normally left to be generated here, but callers that need to
    know the run_id before the first event arrives (e.g. the async
    POST /research/runs API endpoint, which must return a run_id to the
    client immediately) may pass one in explicitly.

    advisor_directory is the folder Stage 0 already wrote under this same
    run_id (the advisor names it after the domain, since only the caller
    knows the chosen topic); it is renamed to this run's directory so one
    run leaves one folder (D-034).

    resume is (from_stage, {stage: saved result}) from resume_pipeline (#9):
    the loaded stages are not re-run, and an existing final/ is kept as
    final_vN before anything new is written.

    approve is the paper approval step's callback (#17), handed to
    Discovery; without one the corpus is approved automatically, so
    headless runs, scripts and tests never stall."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = run_id or uuid4().hex
    run_directory = output_root / f"{slugify(request.research_question)}_{run_id}"
    if advisor_directory:
        advisor_dir = Path(advisor_directory)
        # Skipped unless the reported directory exists and carries this
        # run_id, so a caller-supplied path is never moved by mistake.
        if (advisor_dir.is_dir() and advisor_dir.name.endswith(f"_{run_id}")
                and advisor_dir != run_directory and not run_directory.exists()):
            advisor_dir.rename(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    credits_used = scholarly_gateway.count_credits()  # this run's, into metadata.json (#7)
    loaded: dict[str, Any] = resume[1] if resume else {}
    if resume:
        kept = _keep_final_as_version(run_directory)
        yield _progress_event(
            run_id=run_id, stage=resume[0], service="orchestrator", status="running",
            title="Run resumed",
            message=f"Resuming from {resume[0].replace('_', ' ')}; "
                    f"{len(loaded)} earlier stage(s) loaded from disk"
                    + (f", previous final output kept as {kept}." if kept else "."),
            details={"raw_type": "resume", "from_stage": resume[0],
                     "loaded_stages": list(loaded), "kept_final_as": kept},
        )
    _write_json(run_directory / "00_request.json", request.model_dump(mode="json"))
    if request.clarification is not None and not resume:
        c = request.clarification
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_discovery", status="running",
            title="Research question checked",
            message=f"Clarify: {c.decision} — researching \"{request.research_question}\".",
            details={"raw_type": "clarify", "original": c.original_question,
                     "chosen": request.research_question, "decision": c.decision,
                     "specific": c.specific, "missing": c.missing,
                     "suggested": c.suggested_question},
        )

    # ---- Stage 0 (optional): Trend & Gap Advisor ----
    if trend_advisor is not None:
        _write_json(run_directory / "00_trend_advisor" / "result.json",
                    trend_advisor.model_dump(mode="json"))
        flagged = trend_advisor.gap_flagged_count
        yield _progress_event(
            run_id=run_id, stage="trend_advisor", service="trend_advisor", status="done",
            title="Topic chosen from trend analysis",
            message=f"Shortlisted {len(trend_advisor.shortlist)} rising topic(s) in "
                    f"{trend_advisor.domain_label}, "
                    f"{flagged} flagged as recurring gaps.",
            details={"domain": trend_advisor.domain},
        )

    # ---- Research library: past runs on a similar question ----
    try:
        # Already shown when the run first started.
        related = [] if resume else library.search_runs(
            request.research_question, limit=3, exclude_run_id=run_id)
    except Exception as error:  # the library is a convenience; never fail a run over it
        related = []
        yield _library_warning(run_id, "discovery", error)
    if related:
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_library", status="running",
            title="Related past runs", message=library.describe_run(related[0])
            + (f" (and {len(related) - 1} more)" if len(related) > 1 else ""),
            details={"raw_type": "related_runs", "runs": related},
        )

    # ---- Stage 1: Research Discovery ----
    t0 = time.monotonic()
    if "discovery" in loaded:
        discovery_result = loaded["discovery"]
        yield _loaded_event(run_id, "discovery", "research_discovery")
    else:
        budget = check_budget(request)
        yield _progress_event(
            run_id=run_id, stage="discovery", service="scholarly_gateway",
            status="warning" if budget["plan"] == "reduced" else "running",
            title="Budget check",
            message=f"Budget check: {budget['reason']} "
                    + (f"Running the reduced plan: {REDUCED_PLAN}" if budget["plan"] == "reduced"
                       else "Running the full plan."),
            details={"raw_type": "budget_check", **budget},
        )
        discovery_request = DiscoveryRequest(
            research_question=request.research_question, domain=request.domain,
            year_range=request.year_range, preferred_databases=request.preferred_databases,
            keywords=request.keywords, excluded_topics=request.excluded_topics,
            corpus_size=request.corpus_size,
            verification_searches=budget["verification_searches"],
        )
        discovery_result = None
        async for event in discovery_service.run_discovery(
                discovery_request, **({"approve": approve} if approve else {})):
            if event["type"] == "result":
                discovery_result = event["result"]
            else:
                yield _adapt_inner_event(run_id, "discovery", "research_discovery", event)
        if discovery_result is None:
            raise PipelineError("Research Discovery Service produced no result.")
        _write_json(run_directory / "01_discovery" / "result.json", discovery_result.model_dump(mode="json"))
    t1 = time.monotonic()
    if "discovery" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="discovery", service="research_discovery", status="done",
            title="Research discovery complete",
            message=f"Selected {len(discovery_result.selected_papers)} paper(s), "
                    f"found {len(discovery_result.research_gaps)} research gap(s).",
            details={"seconds": round(t1 - t0, 1)},
        )

    # ---- Stage 2: Research Writing ----
    if "writing" in loaded:
        writing_result = loaded["writing"]
        yield _loaded_event(run_id, "writing", "research_writing")
    else:
        writing_request = WritingRequest(
            discovery=discovery_result, target_format=request.target_format,
            format_other_name=request.format_other_name, output_language=request.language,
            target_words=request.max_draft_length,
        )
        writing_result = None
        async for event in writing_service.run_writing(
            writing_request, output_root=run_directory / "02_writing",
        ):
            if event["type"] == "result":
                writing_result = event["result"]
            else:
                yield _adapt_inner_event(run_id, "writing", "research_writing", event)
        if writing_result is None:
            raise PipelineError("Research Writing Service produced no result.")
        _write_json(run_directory / "02_writing" / "result.json", writing_result.model_dump(mode="json"))
    t2 = time.monotonic()
    if "writing" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="writing", service="research_writing", status="done",
            title="Draft generation complete",
            message=f"{writing_result.draft_metadata.sections_written} section(s) written, "
                    f"{len(writing_result.reference_candidates)} reference(s) cited.",
            details={"seconds": round(t2 - t1, 1)},
        )

    # ---- Stage 3: Citation Verification ----
    if "verification" in loaded:
        verification_result = loaded["verification"]
        yield _loaded_event(run_id, "verification", "citation_verification")
    else:
        verification_result = None
        async for event in verification_service.run_verification(
            VerificationRequest(writing=writing_result),
        ):
            if event["type"] == "result":
                verification_result = event["result"]
            else:
                yield _adapt_inner_event(run_id, "verification", "citation_verification", event)
        if verification_result is None:
            raise PipelineError("Citation Verification Service produced no result.")
        _write_json(run_directory / "03_verification" / "result.json", verification_result.model_dump(mode="json"))
    t3 = time.monotonic()
    if "verification" not in loaded:
        yield _progress_event(
            run_id=run_id, stage="verification", service="citation_verification", status="done",
            title="Citation verification complete",
            message=f"{len(verification_result.verified_references)} verified, "
                    f"{len(verification_result.invalid_references)} invalid, "
                    f"{len(verification_result.unverifiable_references)} unverifiable.",
            details={"seconds": round(t3 - t2, 1)},
        )

    # ---- Stage 4: Quality Assurance (always the last stage, so never loaded) ----
    qa_result = None
    async for event in qa_service.run_quality_assurance(
        QualityAssuranceRequest(
            discovery=discovery_result, writing=writing_result, verification=verification_result,
        ),
    ):
        if event["type"] == "result":
            qa_result = event["result"]
        else:
            yield _adapt_inner_event(run_id, "quality_assurance", "quality_assurance", event)
    if qa_result is None:
        raise PipelineError("Quality Assurance Service produced no result.")
    t4 = time.monotonic()
    _write_json(run_directory / "04_quality_assurance" / "result.json", qa_result.model_dump(mode="json"))
    yield _progress_event(
        run_id=run_id, stage="quality_assurance", service="quality_assurance", status="done",
        title="Quality assurance complete",
        message=f"Overall quality score: {qa_result.scores.overall:.1f}/5.0.",
        details={"seconds": round(t4 - t3, 1)},
    )

    # ---- Final artifacts ----
    final_dir = run_directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "draft.md").write_text(qa_result.final_draft_markdown, encoding="utf-8")
    (final_dir / "quality_report.md").write_text(qa_result.quality_report_markdown, encoding="utf-8")
    (final_dir / "validation_report.md").write_text(qa_result.final_validation_report, encoding="utf-8")
    (final_dir / "references.md").write_text(
        "\n".join(f"- {entry}" for entry in qa_result.final_references) + "\n", encoding="utf-8",
    )
    # Deterministic, no LLM call — see shared/utilities/evidence_table.py.
    (final_dir / "evidence_table.md").write_text(
        evidence_table.build_evidence_table(Path(writing_result.run_directory), verification_result),
        encoding="utf-8",
    )

    # LaTeX source is always generated (deterministic, no LLM call — see
    # shared/utilities/latex_export.py). PDF compilation is attempted only
    # if a LaTeX toolchain (pdflatex or tectonic) is actually available on
    # this machine; never claimed as produced otherwise.
    tex_path = final_dir / "paper.tex"
    tex_path.write_text(
        latex_export.markdown_to_latex(
            qa_result.final_draft_markdown, title=request.research_question,
            target_format=request.target_format,
        ),
        encoding="utf-8",
    )
    latex_export.compile_pdf(tex_path, final_dir)

    timings = StageTimings(
        discovery_seconds=round(t1 - t0, 1), writing_seconds=round(t2 - t1, 1),
        verification_seconds=round(t3 - t2, 1), quality_assurance_seconds=round(t4 - t3, 1),
    )
    result = PipelineResult(
        run_id=run_id, run_directory=str(run_directory), request=request,
        discovery=discovery_result, writing=writing_result,
        verification=verification_result, quality_assurance=qa_result, timings=timings,
        trend_advisor=trend_advisor,
    )
    draft_meta = writing_result.draft_metadata
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "research_question": request.research_question,
        "timings": timings.model_dump(),
        "overall_quality_score": qa_result.scores.overall,
        # Length/completeness analytics (Fix 8, DECISIONS.md D-026) — the
        # same fields as draft_metadata, surfaced at the top level of the
        # run's metadata.json so a final-status report never has to dig
        # through the writing stage's own result to answer "did this run
        # produce a complete, correctly-sized paper?".
        "draft_status": draft_meta.draft_status,
        "target_words": draft_meta.target_words,
        "actual_words": draft_meta.actual_words,
        "length_status": draft_meta.length_status,
        "evidence_limited_sections": draft_meta.evidence_limited_sections,
        "broken_sections": draft_meta.broken_sections,
        "duplicate_sections": draft_meta.duplicate_sections,
        # Credits each scholarly provider charged this run (#7); stages
        # loaded on a resume spent nothing this time.
        "credits_used": credits_used,
    }
    if resume:
        metadata["resumed_from_stage"] = resume[0]
        metadata["final_versions"] = _final_versions(run_directory)
    if trend_advisor is not None and trend_advisor.chosen_topic:
        metadata["chosen_topic"] = trend_advisor.chosen_topic
        chosen = trend_advisor.chosen_candidate
        metadata["chosen_topic_gap_flagged"] = chosen is not None and chosen.gap_signal
    _write_json(run_directory / "metadata.json", metadata)
    try:
        library.import_run(run_directory)
    except Exception as error:  # the run itself is complete and on disk
        yield _library_warning(run_id, "quality_assurance", error)
    yield {"type": "result", "result": result}


async def run_trend_advisor_stage(
    request: TrendAdvisorRequest, output_root: Path | None = None,
    run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Runs Stage 0 on its own, adapting the advisor service's own progress
    events into the same standardized shape as Stages 1-4, then yielding its
    {"type": "result", "result": TrendAdvisorResult} event unchanged.

    Which shortlisted topic to pick is not decided here — see
    run_advisor_then_pipeline."""
    run_id = run_id or uuid4().hex
    async for event in _trend_advisor_service().run_trend_advisor(
        request, output_root=output_root or DEFAULT_OUTPUT_ROOT, run_id=run_id,
    ):
        if event["type"] == "result":
            yield event
        else:
            yield _adapt_inner_event(run_id, "trend_advisor", "trend_advisor", event)


async def run_advisor_then_pipeline(
    advisor_request: TrendAdvisorRequest,
    select_topic: Callable[[TrendAdvisorResult], str | None],
    request: ResearchRequest | None = None,
    output_root: Path | None = None, run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stage 0 followed immediately by the four-stage pipeline, under one run_id.

    select_topic is handed the finished shortlist and returns the topic that
    was locked in, or None if it was not (user story 10). Choosing is the
    caller's decision — the same hand-off the advisor CLI makes after its
    converging chat — so nothing here picks a topic or asks a model to.

    The chosen topic becomes ResearchRequest.research_question; `request`
    supplies the remaining pipeline settings when the defaults are not
    wanted, and its own research_question is overwritten."""
    run_id = run_id or uuid4().hex
    advisor_result = None
    advisor_directory = None
    async for event in run_trend_advisor_stage(advisor_request, output_root, run_id):
        if event["type"] == "result":
            advisor_result = event["result"]
            advisor_directory = event.get("run_directory")
        else:
            yield event
    if advisor_result is None:
        raise PipelineError("Trend & Gap Advisor Service produced no result.")

    topic = select_topic(advisor_result)
    if topic is None:
        yield _progress_event(
            run_id=run_id, stage="trend_advisor", service="trend_advisor", status="done",
            title="No topic locked in",
            message="The shortlist was produced but no topic was chosen, so the "
                    "research pipeline was not started.",
        )
        return

    request = (request or ResearchRequest(research_question=topic)).model_copy(
        update={"research_question": topic})

    async for event in run_pipeline(
        request, output_root=output_root, run_id=run_id,
        trend_advisor=advisor_result.model_copy(update={"chosen_topic": topic}),
        advisor_directory=advisor_directory,
    ):
        yield event
