"""In-memory registry for asynchronous pipeline runs, backing
GET /research/runs/{run_id} and its SSE event stream (orchestrator/api.py).

Deliberately in-memory and single-process: this is an MVP addition with no
job queue of its own — a run's live status is only queryable from the same
server process that started it (see DECISIONS.md D-017). A pipeline run id
not in memory (after a restart, or a run started from the CLI) is rebuilt
read-only from the events.jsonl log run_pipeline writes into its run folder.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from orchestrator import pipeline
from shared.contracts.pipeline_contract import PipelineResult, ResearchRequest
from shared.utilities.scholarly_gateway import redact

RunPipelineFn = Callable[..., Any]


@dataclass
class RunState:
    run_id: str
    status: str = "running"  # running | awaiting_approval | completed | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    result: PipelineResult | None = None
    error: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    approval: asyncio.Future | None = None  # set while the run waits at the approval step


_runs: dict[str, RunState] = {}


def get_run(run_id: str) -> RunState | None:
    return _runs.get(run_id) or _load_run_from_disk(run_id)


def _load_run_from_disk(run_id: str) -> RunState | None:
    """Rebuilds a past run from outputs/<slug>_<run_id>/events.jsonl: the
    same events (the result re-validated into a PipelineResult) plus the
    end marker, so status and SSE replay behave as they did live. A log
    with neither a result nor an error line means the process stopped
    mid-run, reported as failed. Not cached: the log stays the source."""
    run_directory = pipeline.find_run_directory(run_id)
    log_path = run_directory / "events.jsonl" if run_directory else None
    if log_path is None or not log_path.is_file():
        return None
    state = RunState(run_id=run_id, status="failed", error="Run stopped before finishing.")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["type"] == "result":
            event["result"] = PipelineResult.model_validate(event["result"])
            state.status, state.result, state.error = "completed", event["result"], None
        elif event["type"] == "error":
            state.status, state.error = "failed", event["message"]
        state.events.append(event)
    state.events.append({"type": "end"})
    return state


def approver(run_id: str):
    """The paper approval step's callback for a web run (#17): the run shows
    as awaiting_approval until approve(run_id, drop) is called."""
    async def wait_for_approval(papers: list[dict], reserve: int) -> dict:
        state = _runs[run_id]
        state.approval = asyncio.get_running_loop().create_future()
        state.status = "awaiting_approval"
        try:
            drop = await state.approval
        finally:
            state.approval, state.status = None, "running"
        return {"approved_by": "web", "drop": drop}
    return wait_for_approval


def approve(run_id: str, drop: list[int]) -> bool:
    """Continues a run waiting at the approval step; False if none is waiting."""
    state = _runs.get(run_id)
    if state is None or state.approval is None or state.approval.done():
        return False
    state.approval.set_result(drop)
    return True


def _publish(state: RunState, event: dict[str, Any]) -> None:
    state.events.append(event)
    for queue in list(state.subscribers):
        queue.put_nowait(event)


def subscribe(state: RunState) -> asyncio.Queue:
    """Returns a queue pre-loaded with every event so far, then kept live
    for any events published after subscribing — so a client connecting
    to the SSE stream after the run has already started still sees the
    full history, not just what happens from then on."""
    queue: asyncio.Queue = asyncio.Queue()
    for event in state.events:
        queue.put_nowait(event)
    state.subscribers.append(queue)
    return queue


def unsubscribe(state: RunState, queue: asyncio.Queue) -> None:
    if queue in state.subscribers:
        state.subscribers.remove(queue)


async def start_run(
    request: ResearchRequest, run_id: str, run_pipeline_fn: RunPipelineFn,
) -> RunState:
    """Registers a new run and launches it as a background task, returning
    immediately with the (running) RunState so the caller can hand the
    run_id back to its client without waiting for the pipeline itself."""
    state = RunState(run_id=run_id)
    _runs[run_id] = state
    asyncio.create_task(_drive_run(request, run_id, run_pipeline_fn, state))
    return state


async def _drive_run(
    request: ResearchRequest, run_id: str, run_pipeline_fn: RunPipelineFn, state: RunState,
) -> None:
    try:
        async for event in run_pipeline_fn(request, run_id=run_id):
            if event["type"] == "result":
                state.result = event["result"]
                state.status = "completed"
            _publish(state, event)
    except Exception as error:  # PipelineError, or any unexpected stage failure
        state.status = "failed"
        # Some exceptions (e.g. httpx timeouts) stringify to "" — never report a blank failure.
        state.error = redact(str(error)) or type(error).__name__
        _publish(state, {"type": "error", "message": state.error})
    finally:
        _publish(state, {"type": "end"})
