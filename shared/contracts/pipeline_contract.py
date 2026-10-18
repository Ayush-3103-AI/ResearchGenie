"""Top-level input/output contract for the full four-stage research pipeline.

This is the orchestrator's own contract, one level above the four
per-service contracts — see PROJECT_NOTES.md's "Inputs" section for the
user-facing fields this maps from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from shared.contracts.discovery_contract import DiscoveryResult
from shared.contracts.qa_contract import QualityAssuranceResult
from shared.contracts.verification_contract import VerificationResult
from shared.contracts.trend_contract import TrendAdvisorResult
from shared.contracts.writing_contract import TargetFormat, WritingResult

PublicationType = Literal["conference", "journal", "other"]
# ask: an interactive surface (CLI, web page) asked the user before the run
# and recorded the outcome in `clarification`. auto: the pipeline asks the
# model itself and takes the suggestion. skip: no clarify step — the default,
# so headless callers and scripts keep their question unchanged.
ClarifyMode = Literal["ask", "skip", "auto"]


class Clarification(BaseModel):
    """What the clarify step saw and decided. The chosen question is the
    request's research_question; this keeps the original next to it."""
    original_question: str
    decision: Literal["accept", "edit", "keep"]
    # None when the model gave no usable answer (the original was kept).
    specific: bool | None = None
    missing: list[str] = Field(default_factory=list)
    suggested_question: str = ""


class ResearchRequest(BaseModel):
    """The single request that starts a pipeline run."""
    research_question: str
    publication_type: PublicationType = "conference"
    target_venue: str | None = None
    target_format: TargetFormat = "IEEE"
    format_other_name: str | None = None
    deadline: str | None = None
    corpus_size: int = Field(default=8, ge=6, le=9)
    domain: str | None = None
    year_range: tuple[int, int] | None = None
    preferred_databases: list[str] = Field(default_factory=list)
    language: str = "en"
    max_draft_length: int | None = Field(default=None, gt=0)
    keywords: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    clarify: ClarifyMode = "skip"
    clarification: Clarification | None = None

    @model_validator(mode="after")
    def _require_format_other_name(self) -> "ResearchRequest":
        if self.target_format == "Other" and not self.format_other_name:
            raise ValueError("format_other_name is required when target_format is 'Other'")
        return self


class StageTimings(BaseModel):
    discovery_seconds: float
    writing_seconds: float
    verification_seconds: float
    quality_assurance_seconds: float


class PipelineResult(BaseModel):
    """Aggregate output of one full pipeline run."""
    run_id: str
    run_directory: str
    request: ResearchRequest
    # Present only when the run was started from the optional Trend & Gap
    # Advisor (pipeline stage 0) rather than from a question the user typed.
    # Carried so a finished run answers "why this topic?" as well as "what
    # did it produce?" — see DECISIONS.md D-031.
    trend_advisor: TrendAdvisorResult | None = None
    discovery: DiscoveryResult
    writing: WritingResult
    verification: VerificationResult
    quality_assurance: QualityAssuranceResult
    timings: StageTimings
