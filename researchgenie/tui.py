"""Terminal presentation layer: translates the orchestrator's real progress
events into a clean, live, section-based display. Never fabricates progress
— every line shown here is derived from an actual event the pipeline emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from researchgenie.theme import ACCENT, ERROR, MUTED, OK, SECONDARY, WARN

# Maps the orchestrator's internal stage keys to display titles, in the
# fixed order the pipeline actually runs them.
_STAGE_TITLES = {
    "trend_advisor": "Trend & Gap Advisor",
    "discovery": "Research Discovery",
    "writing": "Research Writing",
    "verification": "Citation Verification",
    "quality_assurance": "Quality Assessment",
}
_STAGE_ORDER = ["discovery", "writing", "verification", "quality_assurance"]
# Stage 0 is optional and only ever runs via `researchgenie-advise`, so it is
# opt-in here — a plain run must not show a row that will stay empty forever.
_TREND_ADVISOR_STAGE = "trend_advisor"

_STATUS_GLYPH = {
    "done": ("✓", OK),
    "running": ("◐", ACCENT),
    "warning": ("!", WARN),
    "failed": ("✗", ERROR),
    "pending": ("○", MUTED),
}


@dataclass
class _StageLog:
    lines: list[tuple[str, str]] = field(default_factory=list)  # (message, status)
    status: str = "pending"
    had_warning: bool = False  # tracks a warning/failed line seen anywhere in this stage

    def add(self, message: str, status: str) -> None:
        # Collapse consecutive duplicate messages (e.g. repeated "running"
        # pings for the same underlying step) rather than spamming the log.
        if self.lines and self.lines[-1][0] == message:
            self.lines[-1] = (message, status)
        else:
            self.lines.append((message, status))
        if status in ("warning", "failed"):
            self.had_warning = True


class PipelineView:
    """Consumes orchestrator progress events and renders a live tree of
    stage → message lines, each with a real status glyph."""

    def __init__(self, include_trend_advisor: bool = False) -> None:
        order = ([_TREND_ADVISOR_STAGE] if include_trend_advisor else []) + _STAGE_ORDER
        self.stages: dict[str, _StageLog] = {key: _StageLog() for key in order}

    def ingest(self, event: dict) -> None:
        stage = event.get("stage")
        if stage not in self.stages:
            return
        status = event.get("status", "running")
        message = event.get("message", "")
        log = self.stages[stage]
        log.add(message, status)
        if status == "failed":
            log.status = "failed"
        elif status == "done":
            # A stage can report its own overall "done" even after one of
            # its own messages was a warning/failure (e.g. Writing finishes
            # with some sections unresolved) — never let that overwrite the
            # honest degraded state with a plain, all-clear "done".
            log.status = "warning" if log.had_warning else "done"
        elif log.status == "pending":
            log.status = "running"

    def render(self) -> Group:
        blocks = []
        for key, log in self.stages.items():
            glyph, color = _STATUS_GLYPH[log.status]
            title = Text(f"{glyph} {_STAGE_TITLES[key]}", style=f"bold {color}")
            body = Table.grid(padding=(0, 1))
            body.add_column()
            if not log.lines:
                body.add_row(Text("○ waiting", style=MUTED))
            for message, status in log.lines[-6:]:  # keep the visible log compact
                line_glyph, line_color = _STATUS_GLYPH.get(status, _STATUS_GLYPH["running"])
                body.add_row(Text(f"{line_glyph} {message}", style=line_color))
            blocks.append(Panel(body, title=title, border_style=color, padding=(0, 1)))
        return Group(*blocks)


def render_banner(console: Console) -> None:
    console.print(Panel.fit(
        Text.from_markup(
            "[bold]✦ RESEARCHGENIE[/bold]\n"
            f"[{MUTED}]Local AI Research Assistant[/{MUTED}]"
        ),
        border_style=ACCENT, padding=(1, 4),
    ))


def render_provider_status(console: Console, providers: list[dict]) -> None:
    """One line per literature source: keyed / identified / anonymous, and
    healthy / degraded (#5). `providers` is scholarly_gateway.provider_status()."""
    table = Table.grid(padding=(0, 2))
    for p in providers:
        health = f"[{OK}]healthy[/{OK}]" if p["health"] == "healthy" else f"[{WARN}]degraded[/{WARN}]"
        note = f"[{MUTED}]{p['reason']}[/{MUTED}]" if p["reason"] else ""
        table.add_row(p["name"], p["access"], health, note)
    console.print(Panel(table, title="Literature sources", border_style=SECONDARY))


def render_approval_table(console: Console, papers: list[dict], reserve: int) -> None:
    """The selected papers, numbered, before Writing starts (#17)."""
    table = Table(title="Selected papers", title_style=f"bold {ACCENT}", border_style=SECONDARY)
    table.add_column("#", justify="right", style=MUTED)
    table.add_column("Title", overflow="fold")
    table.add_column("Year", justify="right")
    table.add_column("Venue", overflow="fold", style=MUTED)
    table.add_column("Relevance", justify="right")
    for p in papers:
        relevance = p.get("relevance_score")
        table.add_row(str(p["id"]), p["title"], str(p.get("year") or "n.d."), p.get("venue") or "",
                      f"{relevance:.2f}" if isinstance(relevance, (int, float)) else "")
    console.print(table)
    console.print(f"[{MUTED}]Dropped papers are replaced from the next {reserve} ranked "
                  f"candidate(s), so the corpus keeps its size.[/{MUTED}]")


def render_summary(console: Console, *, question: str, corpus_size: int,
                   target_format: str, provider_label: str,
                   target_words: int | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=f"bold {MUTED}")
    table.add_column()
    table.add_row("Research Topic", question)
    table.add_row("Papers", str(corpus_size))
    table.add_row("Format", target_format)
    if target_words is not None:
        table.add_row("Target length", f"~{target_words} words")
    table.add_row("AI Engine", provider_label)
    console.print(Panel(table, title="Research Summary", border_style=SECONDARY))


def render_completion(console: Console, *, run_directory: str, overall_score: float,
                      artifacts: list[str], draft_status: str | None = None,
                      length_status: str | None = None, actual_words: int | None = None,
                      target_words: int | None = None,
                      evidence_limited_sections: list[str] | None = None,
                      broken_sections: list[str] | None = None) -> None:
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(f"Overall quality score: {overall_score:.1f}/5.0", style=ACCENT))
    if draft_status is not None:
        status_style = {"complete": OK, "partial": WARN, "failed": ERROR}.get(draft_status, MUTED)
        body.add_row(Text(f"Draft status: {draft_status}", style=status_style))
    if actual_words is not None:
        length_line = f"Length: {actual_words} words"
        if target_words is not None:
            length_line += f" (target ~{target_words}, {length_status or 'n/a'})"
        body.add_row(Text(length_line, style=MUTED))
    if evidence_limited_sections:
        body.add_row(Text(
            f"Evidence-limited sections (shorter but honest): {', '.join(evidence_limited_sections)}",
            style=WARN,
        ))
    if broken_sections:
        body.add_row(Text(f"Broken sections: {', '.join(broken_sections)}", style=ERROR))
    body.add_row("")
    body.add_row(Text("Output folder:", style=f"bold {MUTED}"))
    body.add_row(Text(run_directory, style=ACCENT))
    body.add_row("")
    for artifact in artifacts:
        body.add_row(Text(f"✓ {artifact}", style=OK))
    console.print(Panel.fit(
        Group(Text("✓ RESEARCH COMPLETE", style=f"bold {OK}"), Text(""), body),
        border_style=OK, padding=(1, 3),
    ))


def render_failure(console: Console, *, stage: str, reason: str, run_directory: str | None) -> None:
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(Text(f"Stage: {_STAGE_TITLES.get(stage, stage)}", style=MUTED))
    body.add_row("")
    body.add_row(Text("Reason:", style=f"bold {MUTED}"))
    body.add_row(Text(reason, style=ERROR))
    if run_directory:
        body.add_row("")
        body.add_row(Text("Partial work has been preserved at:", style=MUTED))
        body.add_row(Text(run_directory, style=ACCENT))
    console.print(Panel.fit(
        Group(Text("✗ RESEARCH COULD NOT COMPLETE", style=f"bold {ERROR}"), Text(""), body),
        border_style=ERROR, padding=(1, 3),
    ))
