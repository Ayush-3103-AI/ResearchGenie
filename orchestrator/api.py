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
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from orchestrator import run_registry  # noqa: E402
from orchestrator.pipeline import PipelineError, run_pipeline, run_trend_advisor_stage  # noqa: E402
from orchestrator.pipeline import discovery_service  # noqa: E402 — for the clarify step
from researchgenie import advisor_cli  # noqa: E402
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import TrendAdvisorRequest, TrendAdvisorResult  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402

FRONTEND = ROOT / "frontend" / "index.html"
# Only the files the pipeline itself writes to <run>/final/ are downloadable.
FINAL_FILES = {"draft.md", "quality_report.md", "validation_report.md",
               "references.md", "paper.tex", "paper.pdf"}

app = FastAPI(
    title="AI Research Assistant",
    description="Local-first pipeline: research request -> evidence-grounded, "
                "citation-verified, quality-audited draft.",
    version="0.1.0",
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
    await run_registry.start_run(request, run_id, run_pipeline)
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
    }


@app.get("/research/runs/{run_id}/files/{name}")
async def get_run_file(run_id: str, name: str) -> FileResponse:
    state = run_registry.get_run(run_id)
    if state is None or not isinstance(state.result, PipelineResult) or name not in FINAL_FILES:
        raise HTTPException(status_code=404, detail="No such file for this run.")
    path = Path(state.result.run_directory) / "final" / name
    if not path.is_file():
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
    await run_registry.start_run(
        request, run_id,
        lambda req, run_id: run_pipeline(req, run_id=run_id, trend_advisor=chosen,
                                         advisor_directory=directory))
    return {"run_id": run_id}
