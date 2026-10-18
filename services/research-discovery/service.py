"""Public entry point for the Research Discovery Service.

Wraps the internal pipeline (query generation -> multi-source search -> dedupe
-> rank -> full-text mining -> gap/novelty analysis) and adapts its output to
the shared DiscoveryResult contract so the orchestrator and Service 2 never
depend on this service's internal event/report shape.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.contracts.discovery_contract import (  # noqa: E402
    AuthorFlaggedGap, DiscoveryRequest, DiscoveryResult, NoveltyAssessment,
    PaperMetadata, ResearchGap,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discovery.analysis import clarify_question  # noqa: E402, F401 — the orchestrator's clarify step
from discovery.models import PipelineError  # noqa: E402
from discovery.pipeline import run_pipeline  # noqa: E402

DEFAULT_SOURCES = ["s2", "openalex", "pubmed", "arxiv"]
PER_SOURCE_RESULTS = 40


def _paper_metadata(p: dict, discussion_excerpt: str | None = None) -> PaperMetadata:
    return PaperMetadata(
        id=p["id"], title=p["title"], authors=[a for a in p["authors"] if a != "et al."],
        year=p["year"], venue=p["venue"], citations=p["citations"], doi=p.get("doi"),
        url=p.get("link"), source=p["source"], has_abstract=p["has_abstract"],
        abstract=p.get("abstract"),
        relevance_score=p.get("relevance_score"), relevance_reason=p.get("relevance_reason"),
        discussion_excerpt=discussion_excerpt,
    )


async def run_discovery(request: DiscoveryRequest,
                        sources: list[str] | None = None,
                        approve=None) -> AsyncIterator[dict]:
    """Yields progress events (same shape as the underlying pipeline's), then a
    final {"type": "result", "result": DiscoveryResult} event. `approve` is
    the paper approval step's callback (discovery.pipeline.Approver, #17).

    Raises PipelineError (discovery.models.PipelineError) on unrecoverable failure —
    the orchestrator should catch it, log it to the run's audit trail, and stop
    the pipeline at this stage rather than pass partial/fabricated data onward.
    """
    source_keys = sources or DEFAULT_SOURCES
    queries: dict = {}
    corpus_papers: list[dict] = []
    # Populated by the full-text stage, which runs after the corpus event —
    # see discovery.pipeline.fulltext_done_event().
    excerpts: dict[int, str] = {}
    report: dict = {}

    async for event in run_pipeline(request.research_question, source_keys,
                                    PER_SOURCE_RESULTS, request.corpus_size,
                                    fulltext_enabled=True, quality="balanced",
                                    keywords=request.keywords, approve=approve,
                                    domain=request.domain,
                                    preferred_databases=request.preferred_databases):
        if event["type"] == "queries":
            queries = event["queries"]
        elif event["type"] == "tool_plan":
            source_keys = [p["key"] for p in event["plan"]]
        elif event["type"] == "corpus":
            corpus_papers = event["papers"]
        elif event["type"] == "fulltext_done":
            excerpts = event.get("excerpts", {})
        elif event["type"] == "report":
            report = event["report"]
        yield event

    if not report:
        raise PipelineError("Discovery pipeline finished without producing a report.")

    novelty = report["novelty_analysis"]
    result = DiscoveryResult(
        research_request=request,
        research_interpretation=queries.get("interpretation", ""),
        search_queries={k: v for k, v in queries.items() if k != "interpretation"},
        sources_searched=[s for s in source_keys],
        selected_papers=[_paper_metadata(p, excerpts.get(p["id"]))
                         for p in corpus_papers],
        field_overview=report["field_overview"],
        important_findings=[t["summary"] for t in report.get("themes", [])],
        limitations=report["limitations"],
        future_research_directions=[g["question"] for g in report.get("author_flagged_gaps", [])],
        research_gaps=[ResearchGap(**g) for g in report["gaps"]],
        author_flagged_gaps=[AuthorFlaggedGap(**g) for g in report.get("author_flagged_gaps", [])],
        novelty_analysis=NoveltyAssessment(**novelty),
        confidence_notes=report["confidence_notes"],
    )
    yield {"type": "result", "result": result}
