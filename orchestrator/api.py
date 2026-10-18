"""FastAPI backend exposing the pipeline as a set of research endpoints.

POST /research runs the full four-stage pipeline synchronously and returns
the final result directly — the original MVP endpoint, kept unchanged for
callers happy to block for the run's full duration (see DECISIONS.md D-015).

POST /research/runs, GET /research/runs/{run_id}, and
GET /research/runs/{run_id}/events add the asynchronous path deferred at
that point: start a run in the background, poll its status, or stream its
progress events over SSE as they happen (see DECISIONS.md D-017).

The /advisor/* endpoints expose the optional Stage 0 (Trend & Gap Advisor)
the same way, mirroring researchgenie-advise step for step: shortlist, chat,
draft a question, then hand off to the pipeline under the same run_id.
GET / serves the single-page demo UI in frontend/index.html.

Run locally with:
    uvicorn orchestrator.api:app --reload
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator import run_registry  # noqa: E402
from orchestrator.pipeline import (  # noqa: E402
    PipelineError, find_run_directory, plan_resume, resume_pipeline, run_pipeline,
    run_trend_advisor_stage,
)
from orchestrator.pipeline import discovery_service  # noqa: E402 — for the clarify step
from researchgenie import advisor_cli  # noqa: E402
from researchgenie import config as rg_config  # noqa: E402
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest, TrendAdvisorResult  # noqa: E402
from shared.utilities import library, llm_provider, scholarly_gateway  # noqa: E402

FRONTEND = ROOT / "frontend" / "index.html"
# Only the files the pipeline itself writes to <run>/final/ are downloadable.
FINAL_FILES = {"draft.md", "quality_report.md", "validation_report.md",
               "references.md", "evidence_table.md", "paper.tex", "paper.pdf"}

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # The scholarly keys and email saved by `researchgenie config` (#5);
    # environment variables already set win.
    rg_config.apply_scholarly(rg_config.load())
    yield


app = FastAPI(
    title="AI Research Assistant",
    description="Local-first pipeline: research request -> evidence-grounded, "
                "citation-verified, quality-audited draft.",
    version="0.1.0",
    lifespan=_lifespan,
)


@app.post("/research", response_model=PipelineResult)
async def create_research_run(request: ResearchRequest) -> PipelineResult:
    """Validates the request, runs Services 1-4 in sequence, and returns
    the final PipelineResult. FastAPI/Pydantic reject a malformed request
    body with a 422 before this function runs."""
    result: PipelineResult | None = None
    try:
        async for event in run_pipeline(request):
            if event["type"] == "result":
                result = event["result"]
    except PipelineError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline finished without producing a result.")
    return result


@app.post("/research/runs", status_code=202)
async def start_research_run(request: ResearchRequest) -> dict[str, str]:
    """Starts a pipeline run in the background and returns its run_id
    immediately, instead of blocking until the run finishes. Poll
    GET /research/runs/{run_id} for status, or GET
    /research/runs/{run_id}/events for a live SSE progress stream."""
    run_id = uuid4().hex
    # approval "ask": pause after Discovery until POST .../approve (#17).
    approve = {"approve": run_registry.approver(run_id)} if request.approval == "ask" else {}
    await run_registry.start_run(
        request, run_id, lambda req, run_id: run_pipeline(req, run_id=run_id, **approve))
    return {"run_id": run_id}


class Approval(BaseModel):
    drop: list[int] = []  # ids of the shown papers to replace from the ranked candidates


@app.post("/research/runs/{run_id}/approve")
async def approve_research_run(run_id: str, approval: Approval) -> dict[str, str]:
    """Continues a run that is awaiting_approval, dropping the given papers."""
    if not run_registry.approve(run_id, approval.drop):
        raise HTTPException(status_code=409, detail="This run is not waiting for approval.")
    return {"run_id": run_id}


class ResumeRequest(BaseModel):
    from_stage: str | None = None  # default: the first stage without a saved result


@app.post("/research/runs/{run_id}/resume", status_code=202)
async def resume_research_run(run_id: str, body: ResumeRequest) -> dict[str, str]:
    """Resumes a failed run in the background under its own run_id (#9):
    earlier stages are loaded from disk, an existing final/ is kept as
    final_vN, and the events stream from /research/runs/{run_id}/events."""
    state = run_registry.get_run(run_id)
    if state is not None and state.status == "running":
        raise HTTPException(status_code=409, detail="This run is still running.")
    directory = find_run_directory(run_id)
    if directory is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    try:
        plan = plan_resume(directory, body.from_stage)  # fail now, not in the background
    except PipelineError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    approve = run_registry.approver(run_id) if plan["request"].approval == "ask" else None
    await run_registry.start_run(
        None, run_id,
        lambda _request, run_id: resume_pipeline(directory, body.from_stage, approve=approve))
    return {"run_id": run_id}


class QuestionCheck(BaseModel):
    research_question: str


@app.post("/research/clarify")
async def clarify_research_question(check: QuestionCheck) -> dict[str, object]:
    """The clarify step on its own, called by the page before it starts a run
    so the user can accept, edit or keep the suggestion. available is false
    when the model gave no usable answer: the page then keeps the question."""
    clarified = await discovery_service.clarify_question(check.research_question)
    if clarified is None:
        return {"available": False, "specific": None, "missing": [], "suggested_question": ""}
    return {"available": True, **clarified}


@app.get("/research/runs/{run_id}")
async def get_research_run(run_id: str) -> dict[str, object]:
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return {
        "run_id": state.run_id,
        "status": state.status,
        "result": jsonable_encoder(state.result) if state.result is not None else None,
        "error": state.error,
    }


@app.get("/research/runs/{run_id}/events")
async def stream_research_run_events(run_id: str) -> StreamingResponse:
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

    async def event_stream():
        queue = run_registry.subscribe(state)
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "end":
                    yield "event: end\ndata: {}\n\n"
                    break
                payload = json.dumps(jsonable_encoder(event))
                yield f"data: {payload}\n\n"
        finally:
            run_registry.unsubscribe(state, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/system")
async def system_status() -> dict[str, object]:
    """Whether the configured model is ready — shown before a run starts, so
    an unreachable Ollama is a visible red light, not a stage failing later."""
    return {
        "provider": llm_provider.AI_PROVIDER,
        "model": llm_provider.GEMINI_MODEL if llm_provider.AI_PROVIDER == "gemini"
        else llm_provider.DEFAULT_MODEL,
        "model_ready": await llm_provider.health_check(),
        "domains": [{"key": key, "label": label} for key, label in advisor_cli.DOMAINS],
        # keyed / identified / anonymous and healthy / degraded per source (#5)
        "providers": scholarly_gateway.provider_status(),
    }


@app.get("/research/runs/{run_id}/files/{name}")
async def get_run_file(run_id: str, name: str) -> FileResponse:
    state = run_registry.get_run(run_id)
    if state is not None and isinstance(state.result, PipelineResult):
        directory = Path(state.result.run_directory)
    else:
        # A past run this server process never ran (e.g. one listed by a
        # related_runs event) is found through the research library.
        past = library.get_run(run_id)
        directory = Path(past["run_dir"]) if past else None
    if directory is None or name not in FINAL_FILES:
        raise HTTPException(status_code=404, detail="No such file for this run.")
    # Imported copies under docs/research-papers/ keep their files at the top level.
    path = next((p for p in (directory / "final" / name, directory / name) if p.is_file()), None)
    if path is None:
        raise HTTPException(status_code=404, detail=f"{name} was not produced by this run.")
    return FileResponse(path, filename=name)


# ---- Stage 0: Trend & Gap Advisor ----

class ChatTurn(BaseModel):
    question: str


class TopicPick(BaseModel):
    topic: str


class AdvisorHandoff(ResearchRequest):
    """A ResearchRequest plus the shortlisted topic it was drafted from."""
    topic: str


def _advisor_result(run_id: str) -> tuple[TrendAdvisorResult, str | None]:
    """The finished shortlist of an advisor run and the folder it saved to."""
    state = run_registry.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    if not isinstance(state.result, TrendAdvisorResult):
        raise HTTPException(status_code=409, detail="This run has no finished shortlist.")
    directory = next((e.get("run_directory") for e in state.events
                      if e.get("type") == "result"), None)
    return state.result, directory


def _candidate(result: TrendAdvisorResult, topic: str):
    candidate = next((c for c in result.shortlist if c.topic == topic), None)
    if candidate is None:
        raise HTTPException(status_code=422, detail="Topic is not on this run's shortlist.")
    return candidate


@app.post("/advisor/runs", status_code=202)
async def start_advisor_run(request: TrendAdvisorRequest) -> dict[str, str]:
    """Starts Stage 0 in the background; poll or stream it with the same
    /research/runs/{run_id} endpoints as a pipeline run."""
    run_id = uuid4().hex
    await run_registry.start_run(
        request, run_id, lambda req, run_id: run_trend_advisor_stage(req, run_id=run_id))
    return {"run_id": run_id}


@app.post("/advisor/runs/{run_id}/chat")
async def advisor_chat(run_id: str, turn: ChatTurn) -> dict[str, str]:
    """One grounded chat turn about the shortlist (advisor_cli._chat_turn)."""
    result, _ = _advisor_result(run_id)
    try:
        reply = await advisor_cli._chat_turn(turn.question, advisor_cli._shortlist_context(result))
    except llm_provider.LLMProviderError as error:
        raise HTTPException(status_code=502, detail=f"The model didn't answer: {error}") from error
    return {"reply": reply}


@app.post("/advisor/runs/{run_id}/question")
async def advisor_question(run_id: str, pick: TopicPick) -> dict[str, object]:
    """Drafts a research question for the picked topic. Falls back to the
    topic itself, flagged, exactly as the CLI does when no question comes back."""
    result, _ = _advisor_result(run_id)
    candidate = _candidate(result, pick.topic)
    try:
        question = await advisor_cli._research_question_for(candidate)
    except llm_provider.LLMProviderError:
        question = ""
    return {"research_question": question or candidate.topic, "drafted": bool(question)}


@app.post("/advisor/runs/{run_id}/pipeline", status_code=202)
async def advisor_to_pipeline(run_id: str, handoff: AdvisorHandoff) -> dict[str, str]:
    """Runs the four-stage pipeline on the locked-in topic under the advisor's
    own run_id, so the chained run leaves one folder (D-034). The registry
    entry is replaced by the pipeline run; its stream starts afresh."""
    result, directory = _advisor_result(run_id)
    chosen = advisor_cli.with_chosen_topic(result, _candidate(result, handoff.topic))
    request = ResearchRequest(**handoff.model_dump(exclude={"topic"}))
    approve = {"approve": run_registry.approver(run_id)} if request.approval == "ask" else {}
    await run_registry.start_run(
        request, run_id,
        lambda req, run_id: run_pipeline(req, run_id=run_id, trend_advisor=chosen,
                                         advisor_directory=directory, **approve))
    return {"run_id": run_id}
