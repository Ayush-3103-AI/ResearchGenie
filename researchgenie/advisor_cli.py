"""ResearchGenie Trend & Gap Advisor — console entry point.

    researchgenie-advise   Pick a domain, get an evidence-backed shortlist of
                           rising topics, converge on one, and optionally run
                           the full research pipeline on it.

Stage 0 only: everything after the student locks in a topic is the existing,
already-tested pipeline (orchestrator.pipeline.run_pipeline) called unchanged
with an auto-filled ResearchRequest. The plain `researchgenie` entry point is
untouched by this module.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.prompt import Confirm, Prompt  # noqa: E402
from rich.table import Table  # noqa: E402

from researchgenie import config as rg_config  # noqa: E402
from researchgenie.cli import _ensure_ready, _run_research  # noqa: E402
from researchgenie.theme import ACCENT, ERROR, MUTED, OK, SECONDARY, WARN  # noqa: E402
from researchgenie.tui import render_banner  # noqa: E402
from shared.contracts.pipeline_contract import ResearchRequest  # noqa: E402
from shared.contracts.trend_contract import (  # noqa: E402
    TopicCandidate, TrendAdvisorRequest, TrendAdvisorResult,
)
from shared.utilities import llm_provider  # noqa: E402

# (picklist key, what a student would call it)
DOMAINS: list[tuple[str, str]] = [
    ("CS_AI_ML", "Computer Science / AI / ML"),
    ("ECE", "Electronics & Communication Engineering"),
    ("EEE", "Electrical & Electronics Engineering"),
    ("MECHANICAL", "Mechanical Engineering"),
    ("CIVIL", "Civil Engineering"),
    ("CLOUD_DEVOPS", "Cloud / DevOps"),
    ("Other", "Other (type your own)"),
]

_PICK_PATTERNS = [
    # "go with #2", "pick 2", "number 2", "option 2", "let's do 2", "2"
    re.compile(r"(?:^|\b)(?:go\s+with|let'?s\s+do|pick|choose|select|option|number|take)\s*"
               r"#?\s*(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"^\s*#?\s*(\d{1,2})\s*$"),
]

# A question is never a pick: "what do 3 papers say?" mentions a number, it
# does not choose one.
_QUESTION = re.compile(r"^(?:wh(?:at|ich|y|en|o|ere)|how|is|are|was|were|do|does|did|can|"
                       r"could|should|would|will|tell|explain|compare)\b", re.IGNORECASE)

CHAT_SYSTEM_PROMPT = """You are a research-topic advisor for an engineering student.

You are given a shortlist of topics, each with real publication counts and
real papers behind it. Answer the student's question about those topics
using ONLY the evidence given to you.

Rules:
- Never invent a paper, a number, or a topic that is not in the shortlist.
- If the evidence does not answer the question, say so plainly.
- Never tell the student which topic to choose — compare, and let them decide.
- Two short paragraphs at most."""

QUESTION_SYSTEM_PROMPT = """You turn a research topic into one research question.

Write a single, specific, answerable research question for the topic and the
unmet needs given to you. It must be narrow enough for one student paper,
and grounded in the gap evidence provided — not a broad survey prompt.
Return the question only."""


def _load_advisor_service():
    """services/trend-advisor/service.py sits in a hyphenated directory, so it
    is loaded by explicit file path — the same pattern orchestrator/pipeline.py
    uses for the other four services (see DECISIONS.md D-010)."""
    path = ROOT / "services" / "trend-advisor" / "service.py"
    spec = importlib.util.spec_from_file_location("researchgenie_trend_advisor", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_pick(text: str, candidates: list[TopicCandidate]) -> int | None:
    """The 1-based shortlist position the student just chose, or None.

    Parsed here rather than by the model (spec user story 7): "go with #2"
    must mean entry 2 every single time, not usually. A number outside the
    shortlist is not a pick — better to fall through to the chat turn and
    let the student see the list again than to silently lock in the wrong
    topic. An exact topic name is accepted too, since students copy titles.
    A question that happens to contain a number is never a pick.
    """
    stripped = text.strip()
    asking = stripped.endswith("?") or bool(_QUESTION.match(stripped))
    for pattern in _PICK_PATTERNS:
        match = pattern.search(stripped)
        if match:
            if asking:
                return None
            index = int(match.group(1))
            if 1 <= index <= len(candidates):
                return index
            return None
    for position, candidate in enumerate(candidates, start=1):
        if candidate.topic.lower() == stripped.lower():
            return position
    return None


def with_chosen_topic(result: TrendAdvisorResult,
                      chosen: TopicCandidate) -> TrendAdvisorResult:
    """The same result, naming the topic the student locked in.

    The shortlist keeps its own ranking untouched — it is the context the
    decision was made against, and re-sorting it would misrepresent what the
    student was actually shown.
    """
    return result.model_copy(update={"chosen_topic": chosen.topic})


def _shortlist_context(result: TrendAdvisorResult) -> str:
    """The shortlist as structured text for the model — every number it is
    allowed to cite, and nothing else."""
    lines = []
    for position, c in enumerate(result.shortlist, start=1):
        lines.append(
            f"{position}. {c.topic}\n"
            f"   growth: {c.growth_metric}x ({c.prior_papers} -> {c.paper_count} papers)\n"
            f"   volume: {c.paper_count} recent papers "
            f"({'crowded' if c.is_crowded else 'under-explored'})\n"
            f"   recurring gap flagged by multiple papers: {'yes' if c.gap_signal else 'no'}"
        )
        for evidence in c.gap_evidence[:3]:
            lines.append(f"   gap evidence: {evidence}")
        for paper in c.example_papers[:3]:
            lines.append(f"   example paper: {paper.title} ({paper.doi or 'no DOI'})")
    return "\n".join(lines)


def _render_shortlist(console: Console, result: TrendAdvisorResult) -> None:
    table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None, pad_edge=False)
    table.add_column("#", width=3)
    table.add_column("Topic")
    table.add_column("Growth", justify="right")
    table.add_column("Recent papers", justify="right")
    table.add_column("Signal")
    for position, c in enumerate(result.shortlist, start=1):
        style = SECONDARY if c.gap_signal else MUTED
        signal = f"[{style}]{c.signal_label}[/{style}]"
        table.add_row(str(position), c.topic, f"{c.growth_metric}x",
                      f"{c.paper_count:,}", signal)
    console.print()
    console.print(table)
    for warning in result.warnings:
        console.print(f"[{WARN}]Note:[/{WARN}] {warning}")


def _render_evidence(console: Console, candidate: TopicCandidate, position: int) -> None:
    console.print(f"\n[bold]{position}. {candidate.topic}[/bold]")
    console.print(
        f"[{MUTED}]{candidate.prior_papers:,} papers in the earlier window -> "
        f"{candidate.paper_count:,} in the recent one "
        f"({candidate.growth_metric}x).[/{MUTED}]"
    )
    if candidate.arxiv_recent_count is not None:
        console.print(f"[{MUTED}]arXiv preprints among recent top hits: "
                      f"{candidate.arxiv_recent_count}.[/{MUTED}]")
    for evidence in candidate.gap_evidence:
        console.print(f"  [{SECONDARY}]gap:[/{SECONDARY}] {evidence}")
    for paper in candidate.example_papers:
        doi = f" — doi:{paper.doi}" if paper.doi else ""
        console.print(f"  [{MUTED}]paper:[/{MUTED}] {paper.title}{doi}")


def _collect_domain(console: Console) -> TrendAdvisorRequest:
    console.print("\n[bold]Which field are you working in?[/bold]")
    for position, (_, label) in enumerate(DOMAINS, start=1):
        console.print(f"  [{ACCENT}]{position}[/{ACCENT}]  {label}")
    choice = Prompt.ask("Select", choices=[str(i) for i in range(1, len(DOMAINS) + 1)],
                        default="1")
    domain = DOMAINS[int(choice) - 1][0]

    other_name = None
    if domain == "Other":
        other_name = Prompt.ask("[bold]Name your field[/bold]")
        while not other_name.strip():
            other_name = Prompt.ask("[bold]Name your field[/bold]")
        console.print(
            f"[{WARN}]Heads up:[/{WARN}] a free-text field has no curated OpenAlex "
            "subfield mapping, so topics are matched by text search and may be "
            "less precise than for a listed field."
        )
    return TrendAdvisorRequest(domain=domain, domain_other_name=other_name)


async def _chat_turn(question: str, context: str) -> str:
    """One model call per student turn, no hidden multi-step reasoning —
    this loop has to stay responsive on this hardware (DECISIONS.md D-009)."""
    comp = await llm_provider.complete_json(
        system=CHAT_SYSTEM_PROMPT,
        user=f"Shortlist:\n{context}\n\nStudent's question: {question}",
        schema={"type": "object", "properties": {"reply": {"type": "string"}},
                "required": ["reply"]},
        max_tokens=600, think=False,
    )
    try:
        return json.loads(comp.text)["reply"]
    except (ValueError, KeyError):
        return comp.text.strip() or "I couldn't answer that from the shortlist evidence."


async def _research_question_for(candidate: TopicCandidate) -> str:
    evidence = "\n".join(f"- {e}" for e in candidate.gap_evidence) or "- (none recorded)"
    comp = await llm_provider.complete_json(
        system=QUESTION_SYSTEM_PROMPT,
        user=f"Topic: {candidate.topic}\n\nUnmet needs stated by papers in this area:\n{evidence}",
        schema={"type": "object", "properties": {"research_question": {"type": "string"}},
                "required": ["research_question"]},
        max_tokens=300, think=False,
    )
    try:
        return json.loads(comp.text)["research_question"].strip()
    except (ValueError, KeyError):
        return ""


def _converge(console: Console, result: TrendAdvisorResult) -> TopicCandidate | None:
    """Refine by conversation until the student explicitly picks one topic.

    The agent never commits on the student's behalf: a pick is always their
    typed choice, and always confirmed before anything downstream runs.
    """
    context = _shortlist_context(result)
    console.print(
        f"\n[{MUTED}]Ask about any topic, or say \"go with #2\" to choose one. "
        f"Type 'quit' to stop.[/{MUTED}]"
    )
    while True:
        answer = Prompt.ask("\n[bold]You[/bold]").strip()
        if not answer:
            continue
        if answer.lower() in {"quit", "exit", "q"}:
            return None

        position = parse_pick(answer, result.shortlist)
        if position is not None:
            candidate = result.shortlist[position - 1]
            _render_evidence(console, candidate, position)
            if Confirm.ask(f"\n[bold]Lock in \"{candidate.topic}\"?[/bold]", default=False):
                return candidate
            continue

        console.print(f"[{MUTED}]Thinking...[/{MUTED}]")
        try:
            console.print(asyncio.run(_chat_turn(answer, context)))
        except llm_provider.LLMProviderError as error:
            console.print(f"[{ERROR}]The model didn't answer: {error}[/{ERROR}]")
            console.print(f"[{MUTED}]Your shortlist and its evidence are still here — "
                          f"ask again, or pick a topic.[/{MUTED}]")


def _run_advisor(console: Console, request: TrendAdvisorRequest,
                 run_id: str) -> tuple[TrendAdvisorResult, str] | None:
    """The shortlist and the folder it was saved in, or None on failure."""
    service = _load_advisor_service()

    async def collect() -> tuple[TrendAdvisorResult, str] | None:
        found = None
        async for event in service.run_trend_advisor(request, run_id=run_id):
            if event["type"] == "result":
                found = (event["result"], event["run_directory"])
            else:
                console.print(f"[{MUTED}]{event['message']}[/{MUTED}]")
        return found

    try:
        return asyncio.run(collect())
    except service.TrendAdvisorError as error:
        console.print(f"\n[{ERROR}]{error}[/{ERROR}]")
        return None


def main() -> None:
    console = Console(legacy_windows=False)

    if not rg_config.is_setup_complete():
        console.print(f"[{WARN}]Run `researchgenie` once first to set up your "
                      f"model provider.[/{WARN}]")
        sys.exit(1)
    config = rg_config.load()
    if not _ensure_ready(console, config):
        sys.exit(1)

    render_banner(console)
    console.print(f"[{OK}]Trend & Gap Advisor.[/{OK}]")

    # One run_id from Stage 0 onward, so a chained run leaves one folder.
    run_id = uuid4().hex
    found = _run_advisor(console, _collect_domain(console), run_id)
    if found is None:
        sys.exit(1)
    result, advisor_directory = found

    _render_shortlist(console, result)
    chosen = _converge(console, result)
    if chosen is None:
        console.print(f"[{MUTED}]No topic locked in. Your shortlist is saved under "
                      f"outputs/.[/{MUTED}]")
        return

    console.print(f"\n[{MUTED}]Drafting a research question...[/{MUTED}]")
    try:
        question = asyncio.run(_research_question_for(chosen))
    except llm_provider.LLMProviderError as error:
        console.print(f"[{ERROR}]The model didn't answer: {error}[/{ERROR}]")
        question = ""
    if not question:
        question = chosen.topic
        console.print(f"[{WARN}]No question was drafted. Below is your topic, not a "
                      f"research question — edit it into one before running.[/{WARN}]")
    console.print(f"\n[bold]Research question:[/bold] {question}")
    question = Prompt.ask("[bold]Use this, or type your own[/bold]", default=question)

    if not Confirm.ask("\n[bold]Run the full research pipeline on it now?[/bold]",
                       default=False):
        console.print(f"[{MUTED}]Saved. Run `researchgenie` when you're ready and "
                      f"paste the question above.[/{MUTED}]")
        return

    asyncio.run(_run_research(
        console,
        # The topic also travels as a keyword: Discovery's relevance gate
        # scores candidates against user keywords as its highest-confidence
        # terms (D-018), so the pick keeps steering the search even if the
        # student rewrote the question above into something looser.
        ResearchRequest(research_question=question, keywords=[chosen.topic], approval="ask"),
        trend_advisor=with_chosen_topic(result, chosen),
        run_id=run_id, advisor_directory=advisor_directory,
    ))


if __name__ == "__main__":
    main()
