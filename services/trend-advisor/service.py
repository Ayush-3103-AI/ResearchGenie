"""Public entry point for the Trend & Gap Advisor Service (pipeline stage 0).

Runs the two evidence steps — publication velocity (trend_advisor/velocity.py)
and the cross-paper future-work gap-mining swarm (trend_advisor/gap_mining.py)
— and adapts them to the shared TrendAdvisorResult contract, in the same
progress-events-then-final-result shape as the four existing pipeline
services, so the orchestrator and the advisor CLI can treat this stage
uniformly with the rest.

This stage is optional and strictly additive: nothing in Services 1-4 calls
it, and a normal `researchgenie` run never reaches it.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.contracts.trend_contract import (  # noqa: E402
    TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.utilities import llm_provider, scholarly_gateway  # noqa: E402
from shared.utilities.run_paths import slugify  # noqa: E402

sys.path.insert(0, str(ROOT / "services" / "research-writing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trend_advisor.gap_mining import mine_gaps  # noqa: E402
from trend_advisor.velocity import (  # noqa: E402
    ARXIV_DOMAINS, VelocityUnavailable, find_rising_topics,
)
from writing.adapters.language_model import close_language_model  # noqa: E402
from writing.adapters.ollama_model import create_ollama_model  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

# Each gap-mining worker only reads one abstract, so it needs nothing like
# the writing service's budget — see gap_mining.WORKER_MAX_TOKENS for why a
# small budget is also the fast choice on this hardware (D-009).
_GAP_MODEL_MAX_TOKENS = 1200


class TrendAdvisorError(RuntimeError):
    """The stage could not produce an honest shortlist. Raised rather than
    yielding an empty or padded result — the whole point of this stage is
    evidence the student can check, so "no evidence" must be visible."""


async def run_trend_advisor(
    request: TrendAdvisorRequest, *, output_root: Path | None = None,
    run_id: str | None = None, model=None,
) -> AsyncIterator[dict]:
    """Yields progress events, then a final
    {"type": "result", "result": TrendAdvisorResult} event.

    Raises TrendAdvisorError when no honest shortlist can be produced (most
    often OpenAlex unreachable, or a free-text domain too thin to rank).

    `model` is a test/advanced seam for the gap-mining swarm's LanguageModel;
    production callers leave it None and get the local Ollama adapter.
    """
    run_id = run_id or uuid4().hex
    warnings: list[str] = []

    yield {"type": "status", "stage": "velocity", "state": "running",
           "message": f"Analysing publication velocity for {request.domain_label}..."}
    # Its own gateway run (#4): the advisor's OpenAlex and arXiv calls share
    # the process-wide pacing with any pipeline run beside it, trip their own
    # breaker, and report rate_limited / provider_exhausted here as they happen.
    gateway_run = scholarly_gateway.Run()
    velocity = gateway_run.task(find_rising_topics(
        request.domain, limit=request.shortlist_size,
        domain_other_name=request.domain_other_name,
    ))
    async for event in gateway_run.relay(velocity):
        yield event
    if (cached := gateway_run.cache_event()) is not None:
        yield cached
    try:
        candidates = velocity.result()
    except VelocityUnavailable as error:
        yield {"type": "status", "stage": "velocity", "state": "failed",
               "message": str(error)}
        raise TrendAdvisorError(str(error)) from None
    yield {"type": "status", "stage": "velocity", "state": "done",
           "message": f"Ranked {len(candidates)} rising topic(s) by publication growth."}

    if request.domain == "Other":
        warnings.append(
            f"'{request.domain_other_name}' is a free-text domain, so it has no "
            "curated OpenAlex subfield mapping — topics were matched by text "
            "search and may be less precise than for a listed domain."
        )
    # Only the CS-adjacent domains query arXiv at all, so a missing count
    # elsewhere means "never asked", not "asked and failed" — warning on it
    # would report an outage that did not happen.
    if request.domain in ARXIV_DOMAINS and any(
            c.arxiv_recent_count is None for c in candidates):
        warnings.append("arXiv was unreachable, so the preprint signal is missing "
                        "for some topics. The OpenAlex ranking is unaffected.")

    owns_model = model is None
    if owns_model:
        # Checked before the swarm rather than inferred from failed workers:
        # with the provider down, every worker fails, every topic comes back
        # with no gap evidence, and "no recurring gaps found" would read as a
        # finding instead of an outage. Degrade visibly, not silently.
        if not await llm_provider.health_check():
            warnings.append(
                "The local model was unavailable, so no future-work/limitations "
                "mining ran — topics are ranked on publication velocity alone, "
                "and the absence of gap flags below means 'not checked', not "
                "'no gaps exist'."
            )
            yield {"type": "status", "stage": "gap_mining", "state": "failed",
                   "message": warnings[-1]}
            model = None
        else:
            model = create_ollama_model(max_tokens=_GAP_MODEL_MAX_TOKENS)

    if model is not None:
        yield {"type": "status", "stage": "gap_mining", "state": "running",
               "message": f"Mining future-work and limitations across "
                          f"{len(candidates)} topic(s) in parallel..."}
        try:
            candidates = await mine_gaps(candidates, model=model)
        finally:
            if owns_model:
                await close_language_model(model)
        flagged = sum(1 for c in candidates if c.gap_signal)
        yield {"type": "status", "stage": "gap_mining", "state": "done",
               "message": f"{flagged} topic(s) flagged as recurring gaps by "
                          "multiple independent papers."}

    result = TrendAdvisorResult(
        domain=request.domain,
        domain_other_name=request.domain_other_name,
        # Gap-flagged topics first: a recurring, publicly-admitted unmet need
        # is a stronger reason to pick a topic than raw growth, which is why
        # this stage mines gaps at all. Within each group the velocity
        # ranking is preserved.
        shortlist=sorted(candidates, key=lambda c: c.gap_signal, reverse=True),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        warnings=warnings,
    )

    # Step 2 (memory): every run keeps its own full, timestamped record, the
    # same convention the four pipeline stages already follow.
    root = output_root or DEFAULT_OUTPUT_ROOT
    run_directory = root / f"{slugify(request.domain_label)}_{run_id}"
    result_path = run_directory / "00_trend_advisor" / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    yield {"type": "result", "result": result, "run_directory": str(run_directory)}
