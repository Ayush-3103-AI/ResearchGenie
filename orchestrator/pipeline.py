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

from shared.contracts.discovery_contract import DiscoveryRequest  # noqa: E402
from shared.contracts.pipeline_contract import (  # noqa: E402
    Clarification, PipelineResult, ResearchRequest, StageTimings,
)
from shared.contracts.qa_contract import QualityAssuranceRequest  # noqa: E402
from shared.contracts.trend_contract import (  # noqa: E402
    TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.contracts.verification_contract import VerificationRequest  # noqa: E402
from shared.contracts.writing_contract import WritingRequest  # noqa: E402
from shared.utilities import latex_export  # noqa: E402
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


def _adapt_inner_event(run_id: str, stage: str, service: str, inner: dict[str, Any]) -> dict[str, Any]:
    """Wrap one of a service's own progress events (each service emits its
    own ad hoc shape — see each service.py) into the standard format."""
    inner_type = inner.get("type", "status")
    state = inner.get("state")
    if state in ("running", "done", "failed"):
        status = state
    elif inner_type == "source_error":
        status = "warning"
    else:
        status = "running"
    message = inner.get("message") or inner_type
    title = message if len(message) <= 60 else f"{stage.replace('_', ' ').title()} update"
    return _progress_event(
        run_id=run_id, stage=stage, service=service, status=status,
        title=title, message=message, details={"raw_type": inner_type},
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_pipeline(
    request: ResearchRequest, output_root: Path | None = None, run_id: str | None = None,
    trend_advisor: TrendAdvisorResult | None = None,
    advisor_directory: Path | str | None = None,
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
    run leaves one folder (D-034)."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    run_id = run_id or uuid4().hex
    # Clarify: an interactive surface already asked ("ask" + clarification);
    # "auto" asks the model here and takes its suggestion. Before the folder
    # is named, so it carries the question actually researched.
    if request.clarify == "auto" and request.clarification is None:
        clarified = await discovery_service.clarify_question(request.research_question)
        accept = clarified is not None and not clarified["specific"]
        request = request.model_copy(update={
            "research_question": clarified["suggested_question"] if accept else request.research_question,
            "clarification": Clarification(original_question=request.research_question,
                                           decision="accept" if accept else "keep", **(clarified or {})),
        })
    run_directory = output_root / f"{slugify(request.research_question)}_{run_id}"
    if advisor_directory:
        advisor_dir = Path(advisor_directory)
        # Skipped unless the reported directory exists and carries this
        # run_id, so a caller-supplied path is never moved by mistake.
        if (advisor_dir.is_dir() and advisor_dir.name.endswith(f"_{run_id}")
                and advisor_dir != run_directory and not run_directory.exists()):
            advisor_dir.rename(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    _write_json(run_directory / "00_request.json", request.model_dump(mode="json"))
    if request.clarification is not None:
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

    # ---- Stage 1: Research Discovery ----
    t0 = time.monotonic()
    discovery_request = DiscoveryRequest(
        research_question=request.research_question, domain=request.domain,
        year_range=request.year_range, preferred_databases=request.preferred_databases,
        keywords=request.keywords, excluded_topics=request.excluded_topics,
        corpus_size=request.corpus_size,
    )
    discovery_result = None
    async for event in discovery_service.run_discovery(discovery_request):
        if event["type"] == "result":
            discovery_result = event["result"]
        else:
            yield _adapt_inner_event(run_id, "discovery", "research_discovery", event)
    if discovery_result is None:
        raise PipelineError("Research Discovery Service produced no result.")
    t1 = time.monotonic()
    _write_json(run_directory / "01_discovery" / "result.json", discovery_result.model_dump(mode="json"))
    yield _progress_event(
        run_id=run_id, stage="discovery", service="research_discovery", status="done",
        title="Research discovery complete",
        message=f"Selected {len(discovery_result.selected_papers)} paper(s), "
                f"found {len(discovery_result.research_gaps)} research gap(s).",
        details={"seconds": round(t1 - t0, 1)},
    )

    # ---- Stage 2: Research Writing ----
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
    t2 = time.monotonic()
    _write_json(run_directory / "02_writing" / "result.json", writing_result.model_dump(mode="json"))
    yield _progress_event(
        run_id=run_id, stage="writing", service="research_writing", status="done",
        title="Draft generation complete",
        message=f"{writing_result.draft_metadata.sections_written} section(s) written, "
                f"{len(writing_result.reference_candidates)} reference(s) cited.",
        details={"seconds": round(t2 - t1, 1)},
    )

    # ---- Stage 3: Citation Verification ----
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
    t3 = time.monotonic()
    _write_json(run_directory / "03_verification" / "result.json", verification_result.model_dump(mode="json"))
    yield _progress_event(
        run_id=run_id, stage="verification", service="citation_verification", status="done",
        title="Citation verification complete",
        message=f"{len(verification_result.verified_references)} verified, "
                f"{len(verification_result.invalid_references)} invalid, "
                f"{len(verification_result.unverifiable_references)} unverifiable.",
        details={"seconds": round(t3 - t2, 1)},
    )

    # ---- Stage 4: Quality Assurance ----
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
    }
    if trend_advisor is not None and trend_advisor.chosen_topic:
        metadata["chosen_topic"] = trend_advisor.chosen_topic
        chosen = trend_advisor.chosen_candidate
        metadata["chosen_topic_gap_flagged"] = chosen is not None and chosen.gap_signal
    _write_json(run_directory / "metadata.json", metadata)
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
