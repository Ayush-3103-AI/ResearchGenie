"""The decision brief (#12): where the student's judgement is needed, and nothing else.

Deterministic, never a model call. Built from what a run folder already
holds: the event log (failure, exhausted providers, cache hits and, when
#7/#19 log them, budget_check and memory_hit), the Writing result's draft
metadata (draft status, broken and evidence-limited sections, the numbered
reference list), Verification's invalid and unverifiable references and the
QA scores. RULES picks exactly one next action: the first matching row wins.

A completed run's brief is final/decision_brief.md, beside the rest of its
output (so a resume keeps it in final_vN with that output). A run that
failed has no final/ of its own: its brief is <run>/decision_brief.md, next
to doctor_report.json, and is removed once a resume completes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from orchestrator import doctor

BRIEF_NAME = "decision_brief.md"
_SCORES = (("overall", "Overall quality"), ("citation_integrity", "Citation integrity"),
           ("claim_source_alignment", "Claim-source alignment"),
           ("process_control", "Process control"), ("literature_coverage", "Literature coverage"))
_STRONG, _WEAK = 3.5, 2.5   # the web page's score tones: ok from 3.5, bad under 2.5
_LEADING_NUMBER = re.compile(r"^\s*\[?(\d+)[\].]")


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _of(event: dict[str, Any], kind: str) -> bool:
    """A gateway or memory event, by its status (#7, #19) or its raw_type."""
    return event.get("status") == kind or (event.get("details") or {}).get("raw_type") == kind


@dataclass
class Facts:
    """Everything a rule or a line of the brief looks at."""
    run_id: str
    state: str                               # completed | failed | stopped
    error: str | None
    stage: str | None                        # the last stage that logged anything
    draft: dict[str, Any]                    # the Writing result's draft_metadata
    scores: dict[str, float]
    references: list[tuple[int, str, str]]   # (number, kind, issue): invalid / unverifiable
    verified: int
    unsupported_claims: int
    missing_citations: int
    exhausted: list[tuple[str, str]]         # (host, stage), first report first
    reduced_plan: bool                       # a budget_check took the reduced plan (#7)
    cache_hits: int
    memory_hits: int
    broken: list[str] = field(default_factory=list)
    evidence_limited: list[str] = field(default_factory=list)


def _number(entry: str, numbered: list[str]) -> int | None:
    """A reference's number: its place in the Writing result's list, else a leading [N] or N."""
    if entry in numbered:
        return numbered.index(entry) + 1
    found = _LEADING_NUMBER.match(entry)
    return int(found.group(1)) if found else None


def gather(run_directory: Path) -> Facts:
    run_directory = Path(run_directory)
    events = doctor.read_events(run_directory)
    error = doctor.last_error(events)
    completed = bool(events) and events[-1].get("type") == "result"
    writing = _read(run_directory / "02_writing" / "result.json")
    verification = _read(run_directory / "03_verification" / "result.json")
    qa = _read(run_directory / "04_quality_assurance" / "result.json") if completed else {}
    draft = writing.get("draft_metadata") or {}
    numbered = writing.get("reference_candidates") or []
    references = sorted(
        (number, kind, check.get("issue") or "")
        for kind in ("invalid", "unverifiable")
        for check in verification.get(f"{kind}_references") or []
        if (number := _number(check.get("reference_entry", ""), numbered)) is not None)
    exhausted: dict[str, str] = {}  # doctor.exhausted_hosts, keeping the stage
    for e in events:
        if _of(e, "provider_exhausted") and (host := (e.get("details") or {}).get("host")):
            exhausted.setdefault(host, e.get("stage") or "discovery")
    budget = [e.get("details") or {} for e in events if _of(e, "budget_check")]
    return Facts(
        run_id=run_directory.name.partition("_")[2],
        state="completed" if completed else "failed" if error is not None else "stopped",
        error=error, stage=next((e["stage"] for e in reversed(events) if e.get("stage")), None),
        draft=draft, scores=(qa.get("scores") or {}), references=references,
        verified=len(verification.get("verified_references") or []),
        unsupported_claims=len(qa.get("unsupported_claims") or []),
        missing_citations=len(qa.get("missing_citations") or []),
        exhausted=list(exhausted.items()),
        reduced_plan=any(d.get("plan") == "reduced" or d.get("reduced_plan") for d in budget),
        cache_hits=sum((e.get("details") or {}).get("hits") or 0 for e in events if _of(e, "cache_hit")),
        memory_hits=sum((e.get("details") or {}).get("count") or 0 for e in events if _of(e, "memory_hit")),
        broken=list(draft.get("broken_sections") or []),
        evidence_limited=list(draft.get("evidence_limited_sections") or []),
    )


def _title(stage: str | None) -> str:
    return (stage or "an unknown stage").replace("_", " ").capitalize()


def _names(items: list) -> str:
    return ", ".join(str(item) for item in items)


def _openalex_stage(f: Facts) -> str | None:
    """The stage that ran without OpenAlex: where its budget ran out, else
    verification when a budget check chose the reduced plan (#7 skips
    OpenAlex verification searches)."""
    stage = next((stage for host, stage in f.exhausted if host == "api.openalex.org"), None)
    return stage or ("verification" if f.reduced_plan else None)


@dataclass(frozen=True)
class Rule:
    id: str
    when: Callable[[Facts], bool]
    action: Callable[[Facts], str]


RULES: tuple[Rule, ...] = (
    Rule("run_failed", lambda f: f.state != "completed",
         lambda f: f"Run `researchgenie doctor {f.run_id}`: the run stopped in {_title(f.stage)}"
                   + (f" ({f.error})" if f.error else "") + ". The Doctor names the cause and "
                   "resumes from the failed stage once you confirm."),
    Rule("broken_sections", lambda f: bool(f.broken) or f.draft.get("draft_status") in ("partial", "failed"),
         lambda f: f"Resume from Writing (`researchgenie resume {f.run_id} --from writing`): "
                   + (f"{len(f.broken)} section(s) are broken: {_names(f.broken)}." if f.broken
                      else f"the draft is {f.draft.get('draft_status')}.")),
    Rule("reduced_plan", lambda f: _openalex_stage(f) is not None,
         lambda f: f"Add your OpenAlex key (`researchgenie config`), then resume from "
                   f"{_title(_openalex_stage(f))} (`researchgenie resume {f.run_id} --from "
                   f"{_openalex_stage(f)}`): {_title(_openalex_stage(f))} ran on a reduced plan."),
    Rule("evidence_limited", lambda f: bool(f.evidence_limited),
         lambda f: f"Resume from Writing (`researchgenie resume {f.run_id} --from writing`): "
                   f"{len(f.evidence_limited)} section(s) were evidence-limited: {_names(f.evidence_limited)}."),
    Rule("check_references", lambda f: bool(f.references),
         lambda f: f"Check reference(s) {_names([n for n, _, _ in f.references])} by hand "
                   "against their DOI before you submit: Verification could not confirm them."),
    Rule("ready", lambda f: True,
         lambda f: "Read the draft through and check each claim against its source in "
                   "final/evidence_table.md; nothing else needs your judgement."),
)


def match(facts: Facts) -> Rule:
    """The first matching row; the last row always matches."""
    return next(rule for rule in RULES if rule.when(facts))


def _strong_and_weak(f: Facts) -> tuple[list[str], list[str]]:
    strong, weak = [], []
    for key, label in _SCORES:
        if (score := f.scores.get(key)) is not None:
            if score >= _STRONG:
                strong.append(f"{label} {score:.1f}/5")
            elif score < _WEAK:
                weak.append(f"{label} {score:.1f}/5")
    total = f.verified + len(f.references)
    if f.verified:
        strong.append(f"{f.verified} of {total} reference(s) verified against Crossref/OpenAlex")
    if f.draft.get("draft_status") == "complete" and not f.broken:
        strong.append(f"Draft complete: {f.draft.get('sections_written', 0)} section(s) written")
    if f.draft.get("length_status") == "on_target":
        strong.append(f"Length on target ({f.draft.get('actual_words')} words)")
    if f.cache_hits:
        strong.append(f"{f.cache_hits} scholarly request(s) answered from the local cache")
    if f.memory_hits:
        strong.append(f"{f.memory_hits} paper excerpt(s) reused from the research library")

    if f.state != "completed":
        weak.append(f"The run stopped in {_title(f.stage)}" + (f": {f.error}" if f.error else ""))
    if f.draft.get("draft_status") in ("partial", "failed"):
        weak.append(f"Draft status: {f.draft['draft_status']}")
    for kind in ("invalid", "unverifiable"):
        if count := sum(1 for _, k, _ in f.references if k == kind):
            weak.append(f"{count} {kind} reference(s)")
    if f.unsupported_claims:
        weak.append(f"{f.unsupported_claims} claim(s) not supported by the cited paper's abstract")
    if f.missing_citations:
        weak.append(f"{f.missing_citations} factual sentence(s) without a citation")
    if f.draft.get("length_status") in ("short", "over_target"):
        weak.append(f"Length {f.draft['length_status'].replace('_', ' ')}: {f.draft.get('actual_words')} "
                    f"of ~{f.draft.get('target_words')} words")
    for host, stage in f.exhausted:
        weak.append(f"{host} ran out of budget during {_title(stage)}")
    if f.reduced_plan:
        weak.append("The OpenAlex budget check chose the reduced plan")
    return strong, weak


def build_brief(run_directory: Path) -> dict[str, Any]:
    """The brief of one run folder: reads it, writes nothing."""
    f = gather(run_directory)
    rule = match(f)
    next_action = rule.action(f)
    strong, weak = _strong_and_weak(f)
    sections = list(dict.fromkeys(f.broken + f.evidence_limited))
    lines = ["# Decision brief", "",
             f"Run {f.run_id} · {f.state}"
             + (f" · overall quality {f.scores['overall']:.1f}/5" if "overall" in f.scores else ""), "",
             "## Next action", "", next_action, "", "## Strong", ""]
    lines += [f"- {line}" for line in strong] or ["- Nothing stands out yet."]
    lines += ["", "## Weak", ""]
    lines += [f"- {line}" for line in weak] or ["- Nothing."]
    lines += ["", "## References to check by hand", ""]
    lines += [f"- [{n}] {kind}" + (f": {issue}" if issue else "") for n, kind, issue in f.references] \
        or ["- None: every reference was verified." if f.verified else "- None."]
    lines += ["", "## Sections to look at", ""]
    lines += [f"- {name}: " + ("broken" if name in f.broken else "evidence-limited") for name in sections] \
        or ["- None."]
    return {"rule": rule.id, "next_action": next_action, "strong": strong, "weak": weak,
            "check_references": [n for n, _, _ in f.references], "sections": sections,
            "state": f.state, "markdown": "\n".join(lines) + "\n"}


def brief_path(run_directory: Path, completed: bool) -> Path:
    run_directory = Path(run_directory)
    return run_directory / "final" / BRIEF_NAME if completed else run_directory / BRIEF_NAME


def write_brief(run_directory: Path) -> Path:
    """Writes the brief where it belongs (see the module docstring) and returns its path."""
    brief = build_brief(run_directory)
    path = brief_path(run_directory, brief["state"] == "completed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief["markdown"], encoding="utf-8")
    if brief["state"] == "completed":
        brief_path(run_directory, False).unlink(missing_ok=True)  # an earlier failure's brief
    return path


def find_brief(run_directory: Path) -> Path | None:
    """The brief written for this run, if any."""
    return next((p for p in (brief_path(run_directory, True), brief_path(run_directory, False))
                 if p.is_file()), None)
