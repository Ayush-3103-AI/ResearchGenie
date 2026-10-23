"""Input/output contract for the Research Discovery Service (pipeline stage 1).

Defined independently of any particular internal implementation so the
orchestrator, the adapter, and Service 2 can all depend on a stable shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GapType = Literal[
    "theoretical", "methodological", "population", "contextual", "temporal",
    "measurement", "evidence_contradiction", "interdisciplinary", "practical",
]
Impact = Literal["high", "medium", "low"]
GapStatus = Literal["open", "partially_addressed", "already_answered"]


class DiscoveryRequest(BaseModel):
    """Input to the Research Discovery Service."""
    research_question: str
    domain: str | None = None
    year_range: tuple[int, int] | None = None
    preferred_databases: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    corpus_size: int = Field(ge=6, le=9)
    # OpenAlex verification searches the budget check allowed (#7); the
    # rest search Crossref. None: no cap.
    verification_searches: int | None = Field(default=None, ge=0)


class PaperMetadata(BaseModel):
    id: int
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citations: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str = ""
    has_abstract: bool = False
    abstract: str | None = None
    relevance_score: float | None = None
    relevance_reason: str | None = None
    # The authors' own Discussion/Limitations/Future-research text, when an
    # open-access full text was actually retrieved (Discovery already fetches
    # this for its own gap mining). Forwarded so Service 2 can write from the
    # authors' stated reasoning and not only from the abstract — see
    # DECISIONS.md D-029/D-039. None whenever no full text was available.
    discussion_excerpt: str | None = None


class ResearchGap(BaseModel):
    title: str
    gap_type: GapType
    impact: Impact
    description: str
    evidence: str
    supporting_paper_ids: list[int] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)


class AuthorFlaggedGap(BaseModel):
    question: str
    status: GapStatus
    verdict: str
    source_paper_ids: list[int] = Field(default_factory=list)
    evidence_paper_ids: list[int] = Field(default_factory=list)
    recommendation: str


class NoveltyAssessment(BaseModel):
    """Whether/how the research question offers something not already covered
    by the selected corpus. Must be evidence-grounded, not asserted."""
    novelty_summary: str
    supporting_gap_titles: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    caveats: str


class DiscoveryResult(BaseModel):
    """Output of the Research Discovery Service — direct input to Service 2."""
    research_request: DiscoveryRequest
    research_interpretation: str
    search_queries: dict[str, str]
    sources_searched: list[str]
    selected_papers: list[PaperMetadata]
    field_overview: str
    important_findings: list[str] = Field(default_factory=list)
    limitations: str
    future_research_directions: list[str] = Field(default_factory=list)
    research_gaps: list[ResearchGap]
    author_flagged_gaps: list[AuthorFlaggedGap] = Field(default_factory=list)
    novelty_analysis: NoveltyAssessment
    confidence_notes: str
