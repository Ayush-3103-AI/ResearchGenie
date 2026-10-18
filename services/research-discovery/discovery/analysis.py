"""LLM-powered steps: search-query generation, full-text mining, and the gap
analysis. The actual provider (local Ollama model) lives in
shared/utilities/llm_provider.py; this module owns the prompts, schemas, and
result validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared.utilities import llm_provider as llm  # noqa: E402

from .models import Paper, PipelineError


def model() -> str:
    """Model for the gap analysis (judgment-heavy step)."""
    return llm.DEFAULT_MODEL


def economy_model() -> str:
    """Model for query generation + full-text mining (cheap/easy steps).

    A single local model handles both — there is no separate economy tier
    the way there is for metered cloud APIs."""
    return llm.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Step 1: turn the research question into database-appropriate search queries
# ---------------------------------------------------------------------------

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "research_interpretation": {
            "type": "string",
            "description": "1-2 sentences restating what the researcher is actually asking, "
                           "in precise field terminology — how the system understood the "
                           "question before searching.",
        },
        "keyword_query": {
            "type": "string",
            "description": "3-8 core terms for general scholarly search engines "
                           "(Semantic Scholar, OpenAlex). No boolean operators.",
        },
        "pubmed_query": {
            "type": "string",
            "description": "A PubMed query. May use OR between synonyms and AND between "
                           "concepts. Keep it broad enough to return results.",
        },
        "arxiv_query": {
            "type": "string",
            "description": "2-6 keywords suitable for arXiv full-text search. "
                           "No boolean operators, no quotes.",
        },
    },
    "required": ["research_interpretation", "keyword_query", "pubmed_query", "arxiv_query"],
    "additionalProperties": False,
}

QUERYGEN_SYSTEM = (
    "You convert a researcher's natural-language question into effective literature-search "
    "queries. Favor recall over precision: the results will be filtered later, so broad, "
    "well-chosen terms beat narrow exact phrases. Use standard terminology from the field "
    "the question belongs to."
)


async def generate_queries(question: str, model_name: str | None = None,
                           feedback: str | None = None) -> dict:
    """`feedback` is the re-plan's account of a thin first search (#15)."""
    comp = await llm.complete_json(
        system=QUERYGEN_SYSTEM,
        user=f"Research question: {question}" + (f"\n\n{feedback}" if feedback else ""),
        schema=QUERY_SCHEMA,
        model=model_name or economy_model(),
        max_tokens=1500,
        think=False,
    )
    if comp.stop == "refusal":
        raise PipelineError("The model declined to generate search queries. Try rephrasing the question.")
    if comp.stop == "length":
        raise PipelineError("Query generation was cut off before finishing. Try again.")
    if not comp.text.strip():
        raise PipelineError("Query generation returned no output. Try again.")
    try:
        q = json.loads(comp.text)
    except json.JSONDecodeError:
        raise PipelineError("Query generation returned unparseable output. Try again.")
    return {"interpretation": q["research_interpretation"], "keyword": q["keyword_query"],
            "pubmed": q["pubmed_query"], "arxiv": q["arxiv_query"]}


# ---------------------------------------------------------------------------
# Step 0 (optional): ask whether the question is specific enough to search
# ---------------------------------------------------------------------------

CLARIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "specific": {
            "type": "boolean",
            "description": "True if the question is specific enough to search the literature "
                           "and write a focused paper on as it stands.",
        },
        "missing": {
            "type": "array",
            "description": "What a sharper question would pin down (e.g. population, method, "
                           "outcome, setting). Empty when specific.",
            "items": {"type": "string"},
        },
        "suggested_question": {
            "type": "string",
            "description": "One sharper version of the question, keeping the researcher's intent.",
        },
    },
    "required": ["specific", "missing", "suggested_question"],
    "additionalProperties": False,
}

CLARIFY_SYSTEM = (
    "You check whether a research question is specific enough to drive a literature search "
    "and a focused paper. A question naming no concrete subject, method, population or outcome "
    "is not specific. If it is not specific, say what is missing and suggest one sharper "
    "question that keeps the researcher's intent. Never change the topic."
)


async def clarify_question(question: str, model_name: str | None = None) -> dict | None:
    """{specific, missing, suggested_question} for the question, or None when
    the model gives no usable answer — the caller then keeps the original
    question, so this step can never fail a run."""
    try:
        comp = await llm.complete_json(
            system=CLARIFY_SYSTEM,
            user=f"Research question: {question}",
            schema=CLARIFY_SCHEMA,
            model=model_name or economy_model(),
            max_tokens=800,
            think=False,
        )
        if comp.stop != "end":
            return None
        c = json.loads(comp.text)
        suggested = str(c["suggested_question"]).strip()
        return {
            # Nothing to offer without a different suggestion, so treat it as specific.
            "specific": bool(c["specific"]) or not suggested or suggested == question.strip(),
            "missing": [str(m) for m in c["missing"]],
            "suggested_question": suggested,
        }
    except Exception:  # noqa: BLE001 — any model/parse failure means "keep the original"
        return None


# ---------------------------------------------------------------------------
# Step 2: mine author-flagged future-research statements from full texts
# ---------------------------------------------------------------------------

MINE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "description": "At most 10 consolidated future-research candidates.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The candidate gap, phrased as one specific, "
                                       "researchable question.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why the authors flagged this (1-2 sentences).",
                    },
                    "source_paper_ids": {"type": "array", "items": {"type": "integer"}},
                    "quotes": {
                        "type": "array",
                        "description": "1-3 short verbatim quotes (max ~40 words each) "
                                       "from the papers' text supporting this candidate.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "integer"},
                                "quote": {"type": "string"},
                            },
                            "required": ["paper_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["question", "rationale", "source_paper_ids", "quotes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

MINE_SYSTEM = """You extract future-research statements from the Discussion, Limitations, Future Directions, and Conclusion sections of academic papers.

You will receive excerpts of these sections from several papers, each numbered [id]. Identify the places where the AUTHORS THEMSELVES point to open questions, limitations that future work should address, or explicit calls for further research.

Rules:
- Only include what authors actually flagged — do not invent gaps yourself. Every candidate needs at least one supporting verbatim quote.
- Quotes must be copied exactly from the provided text (light truncation with ellipses is fine), each at most ~40 words, with the correct paper id.
- Consolidate: when several papers flag essentially the same open question, merge them into one candidate listing all source papers.
- Phrase each candidate as a single specific, researchable question — not "more research is needed on X" but a question a study could answer.
- Ignore boilerplate ("future research should replicate these findings") unless the authors give it specific substance.
- Return at most 10 candidates, ordered from most to least substantive. If the text contains no substantive future-research statements, return an empty list.
- The text comes from PDF extraction and may contain artifacts (broken words, page headers); read through the noise."""

MAX_MINE_CHARS_PER_PAPER = 6500


async def mine_future_research(papers: list[Paper],
                               model_name: str | None = None) -> tuple[list[dict], str | None]:
    """papers: corpus entries that have future_text.
    Returns (consolidated candidates, warning message or None)."""
    m = model_name or economy_model()
    blocks = []
    for p in papers:
        text = p.future_text or ""
        if len(text) > MAX_MINE_CHARS_PER_PAPER:
            half = MAX_MINE_CHARS_PER_PAPER // 2
            text = text[:half] + "\n[…]\n" + text[-half:]
        blocks.append(f"[{p.id}] {p.title} ({p.year or 'n.d.'})\n{text}\n")
    user_message = (
        "Extract and consolidate the author-flagged future-research statements from "
        f"these {len(blocks)} papers:\n\n" + "\n---\n".join(blocks)
    )

    comp = await llm.complete_json(
        system=MINE_SYSTEM,
        user=user_message,
        schema=MINE_SCHEMA,
        model=m,
        max_tokens=6000,
        think=False,
    )

    # Mining is best-effort — the main analysis still runs — but say so honestly.
    if comp.stop == "refusal":
        return [], "The model declined to mine the full texts; skipping author-flagged analysis."
    if comp.stop == "length":
        return [], ("Full-text mining hit its output limit and was discarded; "
                    "author-flagged analysis skipped for this run.")
    try:
        candidates = json.loads(comp.text).get("candidates", [])
    except json.JSONDecodeError:
        return [], "Full-text mining returned unparseable output; author-flagged analysis skipped."

    valid_ids = {p.id for p in papers}
    cleaned = []
    dropped = 0
    for c in candidates[:10]:
        c["source_paper_ids"] = [i for i in c.get("source_paper_ids", []) if i in valid_ids]
        c["quotes"] = [q for q in c.get("quotes", []) if q.get("paper_id") in valid_ids][:3]
        if c["source_paper_ids"]:
            cleaned.append(c)
        else:
            dropped += 1
    warning = (f"{dropped} mined candidate(s) cited papers outside the corpus and were "
               "dropped." if dropped else None)
    return cleaned, warning


# ---------------------------------------------------------------------------
# Step 3: synthesize the corpus into a structured gap report
# ---------------------------------------------------------------------------

GAP_TYPES = [
    "theoretical", "methodological", "population", "contextual", "temporal",
    "measurement", "evidence_contradiction", "interdisciplinary", "practical",
]

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "field_overview": {
            "type": "string",
            "description": "2-3 paragraphs summarizing the state of research on this "
                           "question as reflected in the corpus.",
        },
        "themes": {
            "type": "array",
            "description": "The 2-4 major research themes/clusters in the corpus. "
                           "Fewer, well-supported themes are better than padding "
                           "this list for a small corpus.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "paper_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "summary", "paper_ids"],
                "additionalProperties": False,
            },
        },
        "gaps": {
            "type": "array",
            "description": "2-4 distinct, well-evidenced research gaps. Fewer,"
                           " well-evidenced gaps are better than padding this"
                           " list for a small corpus.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short, specific gap statement."},
                    "gap_type": {"type": "string", "enum": GAP_TYPES},
                    "impact": {"type": "string", "enum": ["high", "medium", "low"]},
                    "description": {
                        "type": "string",
                        "description": "What is missing, and why it matters.",
                    },
                    "evidence": {
                        "type": "string",
                        "description": "How the corpus supports this being a gap: what the "
                                       "cited papers did and did not cover, contradictions, "
                                       "or explicit calls for future research.",
                    },
                    "supporting_paper_ids": {"type": "array", "items": {"type": "integer"}},
                    "research_questions": {
                        "type": "array",
                        "description": "2-4 concrete, answerable research questions that "
                                       "would address this gap.",
                        "items": {"type": "string"},
                    },
                },
                "required": ["title", "gap_type", "impact", "description", "evidence",
                             "supporting_paper_ids", "research_questions"],
                "additionalProperties": False,
            },
        },
        "author_flagged_gaps": {
            "type": "array",
            "description": "One entry per author-flagged future-research candidate you were "
                           "given, in the same order. Empty array if no candidates were given.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "The candidate question, possibly sharpened."},
                    "status": {
                        "type": "string",
                        "enum": ["open", "partially_addressed", "already_answered"],
                        "description": "Verdict after checking the corpus and the "
                                       "verification searches.",
                    },
                    "verdict": {
                        "type": "string",
                        "description": "The reasoning: what the verification papers show, "
                                       "citing them by id. Be concrete about what has and "
                                       "has not been done since the authors flagged this.",
                    },
                    "source_paper_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Papers whose authors flagged this.",
                    },
                    "evidence_paper_ids": {
                        "type": "array", "items": {"type": "integer"},
                        "description": "Papers (from the corpus or verification searches) "
                                       "that informed the verdict.",
                    },
                    "recommendation": {
                        "type": "string",
                        "description": "What a researcher should do with this: pursue it, "
                                       "narrow it, or skip it — and why.",
                    },
                },
                "required": ["question", "status", "verdict", "source_paper_ids",
                             "evidence_paper_ids", "recommendation"],
                "additionalProperties": False,
            },
        },
        "limitations": {
            "type": "string",
            "description": "Honest caveats about this analysis: corpus coverage, databases "
                           "searched, abstract-only reading, etc.",
        },
        "novelty_analysis": {
            "type": "object",
            "description": "Evidence-grounded assessment of what, if anything, the research "
                           "question could newly contribute given this corpus. Must be "
                           "hedged as an assessment, never asserted as proven novelty.",
            "properties": {
                "novelty_summary": {
                    "type": "string",
                    "description": "1-2 paragraphs on what would be new relative to the "
                                   "corpus, grounded in specific gaps — not a generic claim.",
                },
                "supporting_gap_titles": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Titles of entries in `gaps` that this novelty claim rests on.",
                },
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "caveats": {
                    "type": "string",
                    "description": "Why this novelty assessment could be wrong: retrieval "
                                   "limits, corpus size, fast-moving subfield, etc.",
                },
            },
            "required": ["novelty_summary", "supporting_gap_titles", "confidence", "caveats"],
            "additionalProperties": False,
        },
        "confidence_notes": {
            "type": "string",
            "description": "Overall uncertainty notes for this whole report: corpus size "
                           "relative to the field, abstract-only vs full-text coverage, "
                           "any stages that were skipped or failed.",
        },
    },
    "required": ["field_overview", "themes", "gaps", "author_flagged_gaps", "limitations",
                 "novelty_analysis", "confidence_notes"],
    "additionalProperties": False,
}

ANALYSIS_SYSTEM = """You are an expert research methodologist who identifies gaps in academic literature — the kind of analysis a senior scholar performs when writing the "future directions" section of a systematic review.

You will receive a research question and a numbered corpus of papers (titles, metadata, abstracts) retrieved from scholarly databases. Produce a research-gap report as structured JSON.

Rules:
- Ground every claim in the corpus. Cite papers by their [number] via the paper_ids fields. Never cite a number that is not in the corpus.
- A gap must be a genuine absence, contradiction, or weakness evidenced by the corpus — not a generic "more research is needed." Be specific about populations, methods, contexts, measures, or theory.
- Distinguish carefully between "absent from this corpus" and "absent from the literature." When a gap might simply reflect retrieval limits, say so in the evidence field and in limitations.
- Prefer gaps that are researchable: a competent PhD student or research team could act on your suggested research questions.
- Weigh contradictory findings across papers highly — resolving them is often the most valuable kind of gap.
- Use plain prose (no markdown formatting) inside JSON string fields.
- Write for a reader who has not seen the corpus: spell out constructs and abbreviations on first use.
- For novelty_analysis: only claim novelty that is directly grounded in specific gaps you found — never assert novelty as proven fact, always as a hedged assessment with an honest confidence level and caveats. If nothing in the corpus supports a novelty claim, say so plainly (confidence "low", explaining why) rather than inventing one.
- For confidence_notes: be concrete about what could make this report wrong or incomplete (small corpus, abstract-only reading, single-database bias, etc.) — do not write a generic disclaimer.

You may also receive AUTHOR-FLAGGED FUTURE-RESEARCH CANDIDATES — open questions that individual papers' authors raised in their Discussion/Limitations/Future-research sections — each accompanied by a targeted verification search of the wider literature. A statement from a single study is a lead, not a gap: it may have been answered since, or by work the authors did not know. For each candidate, in the author_flagged_gaps field and in the same order:
- Judge its status against BOTH the main corpus and its verification search results: "already_answered" if published work substantially answers it, "partially_addressed" if some but not all of it has been studied, "open" if it remains genuinely unaddressed.
- Justify the verdict citing specific paper ids, and give a concrete recommendation.
- Keep each candidate's verdict and recommendation to 1-3 sentences each — a terse, specific judgment, not a re-analysis of the candidate. You are given every candidate the extraction step found; do not expand any single one at length merely because it was included.
- If a candidate is confirmed open and important, you may also feature it (reworked) in your main gaps list — note the connection in that gap's evidence.
If no candidates are provided, return an empty author_flagged_gaps array.

Overall output budget: this report is generated with a fixed maximum output size. Every field above already tells you to prefer fewer, well-evidenced entries over padding — treat that as a hard constraint, not a suggestion: if you are covering many gaps, themes, and author-flagged candidates in one report, write EACH entry more concisely (still specific and evidence-grounded) rather than risk being cut off before finishing the report. A complete, concise report is always correct over a longer one that gets truncated."""

MAX_ABSTRACT_CHARS = 1600


def _format_corpus(papers: list[Paper]) -> str:
    lines = []
    for p in papers:
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        meta = " | ".join(str(x) for x in [authors or None, p.year, p.venue,
                          f"{p.citations} citations" if p.citations is not None else None,
                          p.source] if x)
        abstract = (p.abstract or "").strip()
        if len(abstract) > MAX_ABSTRACT_CHARS:
            abstract = abstract[:MAX_ABSTRACT_CHARS] + "…"
        lines.append(f"[{p.id}] {p.title}\n({meta})\nAbstract: {abstract or 'Not available.'}\n")
    return "\n".join(lines)


def _format_candidates(candidates: list[dict], verification_papers: list[Paper]) -> str:
    if not candidates:
        return ""
    parts = ["\n\nAUTHOR-FLAGGED FUTURE-RESEARCH CANDIDATES\n"
             "(mined from the papers' Discussion/Limitations/Future-research sections; "
             "judge each against the corpus and the verification search results):\n"]
    for i, c in enumerate(candidates, start=1):
        quotes = "; ".join(f"[{q['paper_id']}] \"{q['quote']}\"" for q in c.get("quotes", []))
        parts.append(
            f"Candidate {i}: {c['question']}\n"
            f"  Flagged by papers: {c['source_paper_ids']}\n"
            f"  Rationale: {c['rationale']}\n"
            f"  Quotes: {quotes or 'n/a'}\n"
            f"  Verification search found papers: {c.get('evidence_paper_ids') or 'none'}\n"
        )
    if verification_papers:
        parts.append(f"\nVERIFICATION SEARCH RESULTS ({len(verification_papers)} additional "
                     "papers, retrieved specifically to test the candidates above):\n")
        parts.append(_format_corpus(verification_papers))
    return "\n".join(parts)


async def analyze_gaps(question: str, papers: list[Paper],
                       candidates: list[dict] | None = None,
                       verification_papers: list[Paper] | None = None,
                       model_name: str | None = None) -> AsyncIterator[dict]:
    """Yields progress events, then a final {"type": "report", ...} event."""
    m = model_name or model()
    candidates = candidates or []
    verification_papers = verification_papers or []
    user_message = (
        f"Research question:\n{question}\n\n"
        f"Corpus ({len(papers)} papers):\n\n{_format_corpus(papers)}"
        f"{_format_candidates(candidates, verification_papers)}"
    )

    yield {"type": "analysis_note", "message": "The model is reading and reasoning over the corpus…"}

    chars_seen = 0
    last_reported = 0
    comp = None
    async for event in llm.stream_json(system=ANALYSIS_SYSTEM, user=user_message,
                                       schema=REPORT_SCHEMA, model=m, max_tokens=8000,
                                       think=False):
        if event["type"] == "delta":
            chars_seen += len(event["text"])
            if chars_seen - last_reported >= 2000:
                last_reported = chars_seen
                yield {"type": "analysis_progress", "chars": chars_seen}
        elif event["type"] == "done":
            comp = event["completion"]

    if comp is None or comp.stop == "refusal":
        raise PipelineError("The model declined to analyze this topic. "
                            "Try rephrasing the research question.")
    if comp.stop == "length":
        raise PipelineError("The analysis exceeded the output limit. Try analyzing fewer papers.")

    try:
        report = json.loads(comp.text)
    except json.JSONDecodeError:
        raise PipelineError("The analysis output could not be parsed. Try again.")

    valid_ids = {p.id for p in papers} | {p.id for p in verification_papers}
    for theme in report.get("themes", []):
        theme["paper_ids"] = [i for i in theme["paper_ids"] if i in valid_ids]
    for gap in report.get("gaps", []):
        gap["supporting_paper_ids"] = [i for i in gap["supporting_paper_ids"] if i in valid_ids]
    for flagged in report.get("author_flagged_gaps", []):
        flagged["source_paper_ids"] = [i for i in flagged["source_paper_ids"] if i in valid_ids]
        flagged["evidence_paper_ids"] = [i for i in flagged["evidence_paper_ids"] if i in valid_ids]
    # Attach the miners' verbatim quotes so the UI can show provenance. The schema
    # asks for one entry per candidate in order, but don't trust that: pair by
    # index and give any surplus entries an empty quotes list rather than
    # truncating silently (zip would drop them).
    flagged_list = report.get("author_flagged_gaps", [])
    for i, flagged in enumerate(flagged_list):
        flagged["quotes"] = candidates[i].get("quotes", []) if i < len(candidates) else []

    yield {"type": "report", "report": report,
           "usage": {"input_tokens": comp.prompt_tokens,
                     "output_tokens": comp.completion_tokens}}
