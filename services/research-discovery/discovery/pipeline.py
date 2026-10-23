"""Orchestrates the full run: query gen → multi-source search → dedupe/rank → analysis."""

from __future__ import annotations

import asyncio
import datetime
import json
import math
from typing import AsyncIterator, Awaitable, Callable

import httpx

from shared.utilities import library, scholarly_gateway

from . import fulltext
from .analysis import (analyze_gaps, economy_model, generate_queries,
                       mine_future_research, model)
from .models import Paper, PipelineError, normalize_title
from .relevance import build_topic_profile, screen_papers
from .source_plan import plan_sources
from .sources import SOURCE_URLS, SOURCES, crossref, openalex

FULLTEXT_MAX_ATTEMPTS = 40   # papers we try to fetch full text for
VERIFY_RESULTS_PER_CANDIDATE = 8
MAX_CANDIDATES_FOR_ANALYSIS = 5  # caps final-synthesis output size; see D-009
LIBRARY_SOURCE = "Research library"  # the source of a past paper seeded from the library (#19)
LIBRARY_SEED_LIMIT = 10
# The query each source reads from a queries dict (see discovery/sources/).
_QUERY_FIELD = {"s2": "keyword", "openalex": "keyword", "pubmed": "pubmed", "arxiv": "arxiv"}


def _merge(existing: Paper, incoming: Paper) -> None:
    """Fold a duplicate into the paper we already have, keeping the richer fields."""
    if not existing.abstract and incoming.abstract:
        existing.abstract = incoming.abstract
    if (incoming.citations or 0) > (existing.citations or 0):
        existing.citations = incoming.citations
    if not existing.doi and incoming.doi:
        existing.doi = incoming.doi
    if not existing.url and incoming.url:
        existing.url = incoming.url
    if not existing.venue and incoming.venue:
        existing.venue = incoming.venue
    if not existing.pdf_url and incoming.pdf_url:
        existing.pdf_url = incoming.pdf_url
    if not existing.pmcid and incoming.pmcid:
        existing.pmcid = incoming.pmcid
    if not existing.year and incoming.year:
        existing.year = incoming.year
    if len(incoming.authors) > len(existing.authors):
        existing.authors = incoming.authors
    existing.relevance_rank = min(existing.relevance_rank, incoming.relevance_rank)
    if incoming.source not in existing.source:
        existing.source = f"{existing.source}, {incoming.source}"


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Collapse duplicates, matching on DOI *or* normalized title.

    A paper can arrive with a DOI from one source (OpenAlex/S2) and without one
    from another (arXiv always has doi=None), so we index every kept paper by
    both keys and check both — otherwise the same work survives twice and the
    model double-counts its evidence.
    """
    by_doi: dict[str, Paper] = {}
    by_title: dict[str, Paper] = {}
    kept: list[Paper] = []

    def index(p: Paper) -> None:
        if p.doi:
            by_doi.setdefault(p.doi, p)
        tkey = normalize_title(p.title)
        if tkey:
            by_title.setdefault(tkey, p)

    for p in papers:
        tkey = normalize_title(p.title)
        existing = (by_doi.get(p.doi) if p.doi else None) or (by_title.get(tkey) if tkey else None)
        if existing is not None:
            _merge(existing, p)
            index(existing)  # a merged-in DOI/title makes `existing` findable both ways
        else:
            kept.append(p)
            index(p)
    return kept


def rank(papers: list[Paper], per_source_limit: int) -> list[Paper]:
    this_year = datetime.date.today().year

    def score(p: Paper) -> float:
        s = 0.0
        if p.abstract:
            s += 4.0  # the analysis reads abstracts; papers without one are near-useless
        # Cap the citation term so hyper-cited classics can't outrank topical papers.
        s += min(math.log1p(p.citations or 0), 8.0) * 0.6
        if p.year:
            s += max(0.0, min(1.0, (p.year - (this_year - 15)) / 15)) * 2.0
        s += (1.0 - p.relevance_rank / max(per_source_limit, 1)) * 4.0
        # Topical relevance to the actual research question (relevance.py /
        # DECISIONS.md D-018) is applied as a MULTIPLIER, not an additive
        # bonus. An additive term can still be out-weighed by raw citations
        # and recency — a highly-cited, recent paper that only weakly
        # overlaps the topic would still out-rank a clearly on-topic one.
        # Multiplying means a paper's citation/recency/abstract "quality"
        # only counts for as much as it is actually relevant.
        return s * p.relevance_score

    return sorted(papers, key=score, reverse=True)


def papers_with_fulltext(corpus: list[Paper]) -> list[Paper]:
    """The papers the full-text stage actually retrieved text for."""
    return [p for p in corpus if p.future_text]


def fulltext_done_event(corpus: list[Paper]) -> dict:
    """The end-of-stage event for the full-text stage.

    Carries the retrieved text itself, not just which papers have it: the
    `corpus` event is emitted before this stage runs, so `to_client_dict()`
    cannot see `future_text` and this event is the only place downstream
    consumers can pick it up (see DECISIONS.md D-039).
    """
    with_text = papers_with_fulltext(corpus)
    return {"type": "fulltext_done",
            "paper_ids": [p.id for p in with_text],
            "excerpts": {p.id: p.future_text for p in with_text}}


async def _collect_fulltext(corpus: list[Paper], gateway_run: scholarly_gateway.Run,
                            transport: httpx.AsyncBaseTransport | None = None,
                            ) -> AsyncIterator[dict]:
    """Fetch Discussion/Future sections for the best full-text candidates.
    Yields fulltext_progress events, plus the gateway's own rate_limited /
    provider_exhausted events as they queue up."""
    # A paper already holding text (reused from the library, #19) is not fetched again.
    targets = [p for p in corpus
               if not p.future_text and fulltext.is_candidate(p)][:FULLTEXT_MAX_ATTEMPTS]
    if not targets:
        return
    # Every fetch starts at Europe PMC, so its policy sets the fan-out width;
    # the gateway's per-host slots then cap each PDF host on its own.
    sem = asyncio.Semaphore(scholarly_gateway.concurrency(fulltext.EPMC_SEARCH))
    done = ok = 0

    async def one(p: Paper) -> bool:
        async with sem:
            return await fulltext.fetch_future_sections(http, p)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True, transport=transport) as http:
        tasks = [gateway_run.task(one(p)) for p in targets]
        try:
            for fut in asyncio.as_completed(tasks):
                ok += 1 if await fut else 0
                done += 1
                for event in gateway_run.drain():
                    yield event
                if done % 5 == 0 or done == len(targets):
                    yield {"type": "fulltext_progress", "done": done,
                           "ok": ok, "total": len(targets)}
        finally:
            # If the client disconnected mid-stage, don't leave tasks running
            # against a client that's about to close.
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def _verify_candidates(candidates: list[dict], corpus: list[Paper],
                             transport: httpx.AsyncBaseTransport | None = None,
                             openalex_searches: int | None = None,
                             ) -> tuple[list[Paper], list[dict]]:
    """Search the literature for each candidate question; return new verification
    papers (with ids continuing after the corpus) and per-candidate warnings.
    One search per distinct question; the first `openalex_searches` of them
    (None: all) search OpenAlex and the rest Crossref, so they never spend
    more credits than the budget check allowed (#7)."""
    # Index the corpus by both DOI and title so a verification hit that lacks a
    # DOI still matches a DOI-keyed corpus paper (and vice versa) instead of being
    # re-added as a duplicate reference.
    known_by_doi: dict[str, Paper] = {p.doi: p for p in corpus if p.doi}
    known_by_title: dict[str, Paper] = {normalize_title(p.title): p for p in corpus
                                        if normalize_title(p.title)}
    verification_papers: list[Paper] = []
    warnings: list[dict] = []
    next_id = max((p.id for p in corpus), default=0) + 1
    sem = asyncio.Semaphore(4)

    def key(cand: dict) -> str:
        return normalize_title(cand["question"]) or cand["question"]

    questions: dict[str, str] = {}
    for cand in candidates:
        questions.setdefault(key(cand), cand["question"])
    allowed = len(questions) if openalex_searches is None else openalex_searches

    async def search_one(index: int, question: str) -> list[Paper]:
        async with sem:
            if index >= allowed:
                return await crossref.search(http, {"keyword": question},
                                             VERIFY_RESULTS_PER_CANDIDATE)
            # require_abstract=False: an answering paper must not be hidden just
            # because OpenAlex has no indexed abstract for it.
            return await openalex.search(http, {"keyword": question},
                                         VERIFY_RESULTS_PER_CANDIDATE,
                                         require_abstract=False)

    async with httpx.AsyncClient(timeout=45, follow_redirects=True, transport=transport) as http:
        found = await asyncio.gather(*[search_one(i, q) for i, q in enumerate(questions.values())],
                                     return_exceptions=True)
    by_question = dict(zip(questions, found))

    for cand in candidates:
        result = by_question[key(cand)]
        cand["evidence_paper_ids"] = []
        if isinstance(result, BaseException):
            warnings.append({"type": "source_error", "source": "Verification search",
                             "message": f"Verification search failed for one candidate "
                                        f"({type(result).__name__}) — it will be judged "
                                        "on the main corpus only."})
            continue
        for p in result:
            tkey = normalize_title(p.title)
            existing = (known_by_doi.get(p.doi) if p.doi else None) or \
                (known_by_title.get(tkey) if tkey else None)
            if existing is not None:
                if existing.id and existing.id not in cand["evidence_paper_ids"]:
                    cand["evidence_paper_ids"].append(existing.id)
                continue
            p.id = next_id
            p.role = "verification"
            next_id += 1
            if p.doi:
                known_by_doi.setdefault(p.doi, p)
            if tkey:
                known_by_title.setdefault(tkey, p)
            verification_papers.append(p)
            cand["evidence_paper_ids"].append(p.id)
    return verification_papers, warnings


async def _search_round(selected: list, queries: dict[str, dict], per_source: int,
                        gateway_run: scholarly_gateway.Run,
                        transport: httpx.AsyncBaseTransport | None) -> AsyncIterator:
    """One search of `selected` sources through the gateway, each with its
    own queries (queries[source key]). Yields the gateway's events and each
    source's result/error event, plus a (source key, papers) tuple for every
    source that answered."""
    async def search_all() -> list:
        async with httpx.AsyncClient(timeout=45, follow_redirects=True,
                                     transport=transport) as http:
            return await asyncio.gather(
                *[fn(http, queries[key], per_source) for key, _, fn in selected],
                return_exceptions=True,
            )

    search = gateway_run.task(search_all())
    async for event in gateway_run.relay(search):
        yield event
    for (key, label, _), result in zip(selected, search.result()):
        if isinstance(result, BaseException):
            yield {"type": "source_error", "source": label,
                   "message": f"{label} search failed ({type(result).__name__}) — continuing without it."}
        else:
            yield {"type": "source_result", "source": label, "count": len(result)}
            yield key, result


def _library_papers(question: str) -> list[Paper]:
    """Papers past runs selected whose title or abstract matches the question
    (#19), best match first. They join the candidate pool before the
    relevance gate, which screens them like any search result."""
    return [Paper(title=row["title"] or "", abstract=row["abstract"], year=row["year"],
                  venue=row["venue"], doi=row["doi"], citations=row["citations"],
                  authors=json.loads(row["authors"] or "[]"), source=LIBRARY_SOURCE,
                  relevance_rank=i)
            for i, row in enumerate(library.search_papers(question, LIBRARY_SEED_LIMIT))]


def _library_error(error: Exception) -> dict:
    # The library is a convenience; never fail a run over it.
    return {"type": "source_error", "source": LIBRARY_SOURCE,
            "message": f"The research library could not be read ({type(error).__name__}) "
                       "— continuing without it."}


def _replan_feedback(rejected: list[Paper], accepted: int, wanted: int) -> str:
    """What the second query generation is told about the first search (#15)."""
    lines = [f"The first search found only {accepted} on-topic paper(s); {wanted} are needed."]
    if rejected:
        lines.append("These results were rejected as off-topic, with the reason:")
        lines += [f"- {p.title} — {p.relevance_reason or 'off-topic'}"
                  for p in sorted(rejected, key=lambda p: p.relevance_score)[:10]]
    lines.append("Write different queries that would find papers on the research question itself.")
    return "\n".join(lines)


# approve(papers, reserve_size) -> {"approved_by": str, "drop": [paper ids]} (#17)
Approver = Callable[[list[dict], int], Awaitable[dict]]


def _paper_ref(p: Paper) -> dict:
    return {"title": p.title, "doi": p.doi, "year": p.year}


async def _approve_corpus(corpus: list[Paper], reserve: list[Paper],
                          approve: Approver | None) -> AsyncIterator:
    """The approval step: asks `approve` (or approves automatically when
    there is none), replaces each dropped paper with the next ranked
    candidate, and renumbers. Yields its events, then the approved corpus."""
    if approve is None:
        decision = {"approved_by": "auto", "drop": []}
    else:
        yield {"type": "approval_request", "reserve": len(reserve),
               "message": f"Waiting for approval of the {len(corpus)} selected paper(s) "
                          "before full text, analysis and writing.",
               "papers": [p.to_client_dict() for p in corpus]}
        decision = await approve([p.to_client_dict() for p in corpus], len(reserve))
    drop = set(decision.get("drop") or [])
    dropped = [p for p in corpus if p.id in drop]
    replacements = reserve[:len(dropped)]
    approved = [p for p in corpus if p.id not in drop] + replacements
    for i, p in enumerate(approved, start=1):
        p.id = i
    short = len(dropped) - len(replacements)
    message = (f"Corpus approved ({decision['approved_by']})"
               + (f": dropped {len(dropped)}, replaced {len(replacements)} from the ranked "
                  "candidates" if dropped else "")
               + (f" — {short} fewer paper(s), no candidates left" if short else "") + ".")
    yield {"type": "approval", "approved_by": decision["approved_by"],
           "approved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "dropped": [_paper_ref(p) for p in dropped],
           "replacements": [_paper_ref(p) for p in replacements], "message": message}
    yield approved


def _models(quality: str) -> tuple[str, str]:
    """(fast model for queries+mining, premium model for the gap analysis)."""
    premium, economy = model(), economy_model()
    return {
        "best": (premium, premium),
        "budget": (economy, economy),
    }.get(quality, (economy, premium))  # "balanced" default


async def run_pipeline(question: str, source_keys: list[str],
                       per_source: int, corpus_size: int,
                       fulltext_enabled: bool = True,
                       quality: str = "balanced",
                       keywords: list[str] | None = None,
                       transport: httpx.AsyncBaseTransport | None = None,
                       approve: Approver | None = None,
                       domain: str | None = None,
                       preferred_databases: list[str] | None = None,
                       memory: bool = True,
                       verification_searches: int | None = None) -> AsyncIterator[dict]:
    """`transport` is a test seam (httpx.MockTransport); production leaves it None.
    `approve` pauses the run after screening for the paper approval step
    (#17); without it the corpus is approved automatically. `domain` and
    `preferred_databases` steer which of `source_keys` are searched (#16).
    `memory` reads the research library (#19): past papers matching the
    question join the candidates, and a stored excerpt replaces a full-text
    fetch. Off, the library is never read.

    `verification_searches` caps the OpenAlex verification searches at what
    the budget check allowed (#7); None: no cap."""
    selected = [(k, *SOURCES[k]) for k in source_keys if k in SOURCES]
    if not selected:
        raise PipelineError("Select at least one literature source.")

    fast_model, analysis_model = _models(quality)
    # This run's breaker state and the queue of the gateway's rate_limited /
    # provider_exhausted events, relayed below as each stage runs.
    gateway_run = scholarly_gateway.Run()
    yield {"type": "log",
           "message": f"Models — queries & mining: {fast_model}; "
                      f"gap analysis: {analysis_model}"}

    all_papers: list[Paper] = []
    yield {"type": "status", "stage": "queries", "state": "running",
           "message": "Generating search queries with the local model…"}
    queries = await generate_queries(question, fast_model)
    yield {"type": "queries", "queries": queries}
    yield {"type": "status", "stage": "queries", "state": "done",
           "message": f"Search queries ready — “{queries['keyword']}”"}

    # Which databases fit the question, and the query for each (#16).
    plan = await plan_sources(question, queries, [k for k, _, _ in selected], fast_model,
                              domain, preferred_databases)
    selected = [s for s in selected if s[0] in plan["queries"]]
    shown = [{"key": key, "source": label, "query": plan["queries"][key][_QUERY_FIELD[key]]}
             for key, label, _ in selected]
    yield {"type": "tool_plan", "mode": plan["mode"], "reason": plan["reason"], "plan": shown,
           "message": "Searching " + ", ".join(f"{s['source']} (“{s['query']}”)" for s in shown)
                      + (" — chosen by the model." if plan["mode"] == "model"
                         else f" — default sources, because {plan['reason']}.")}

    yield {"type": "status", "stage": "search", "state": "running",
           "message": f"Searching {len(selected)} database(s)…"}
    answered: list[str] = []
    async for item in _search_round(selected, plan["queries"], per_source, gateway_run, transport):
        if isinstance(item, tuple):
            answered.append(item[0])
            all_papers.extend(item[1])
        else:
            yield item

    if not answered:
        raise PipelineError("No papers found in any source. Try rephrasing the question "
                            "or enabling more sources.")

    seeded: list[Paper] = []
    if memory:
        try:
            seeded = _library_papers(question)
        except Exception as error:
            yield _library_error(error)
    # Search results first, so a past paper found again keeps the search's fields.
    unique = dedupe(all_papers + seeded)

    # Hard relevance gate (DECISIONS.md D-018): screen every deduped candidate
    # against the research question + LLM-generated keyword query + any
    # user-supplied keywords *before* ranking. Without this, ranking alone
    # (citations/recency/abstract presence/search-engine position) can let a
    # topically unrelated paper into the final corpus if it merely looks
    # "strong" by those other signals — the exact failure this closes.
    profile = build_topic_profile(question, queries.get("keyword", ""), keywords)
    accepted, rejected = screen_papers(unique, profile)

    # Re-plan (#15): a thin corpus gets one second search with new queries,
    # told what was rejected and why — at most 2 rounds in total, and only on
    # sources that answered and the gateway has not marked exhausted.
    if len(accepted) < corpus_size:
        before = len(accepted)
        thin = f"Only {before} of the {corpus_size} papers needed were on topic"
        usable = [s for s in selected if s[0] in answered
                  and gateway_run.available(SOURCE_URLS[s[0]])]
        if not usable:
            yield {"type": "log", "message": f"{thin}, but every source that answered is out of "
                                             "budget or rate-limited, so no second search was run."}
        else:
            yield {"type": "status", "stage": "search", "state": "running",
                   "message": f"{thin}; trying a second search with new queries…"}
            try:
                retry_queries = await generate_queries(
                    question, fast_model, feedback=_replan_feedback(rejected, before, corpus_size))
            except PipelineError as error:
                yield {"type": "source_error", "source": "Re-plan",
                       "message": f"{error} Continuing with the first search's papers."}
            else:
                found: list[Paper] = []
                async for item in _search_round(usable, {k: retry_queries for k, _, _ in usable},
                                                per_source, gateway_run, transport):
                    if isinstance(item, tuple):
                        found.extend(item[1])
                    else:
                        yield item
                all_papers.extend(found)
                unique = dedupe(unique + found)
                accepted, rejected = screen_papers(unique, profile)
                yield {"type": "replan", "round": 2, "reason": f"{thin} after the first search.",
                       "queries": retry_queries, "accepted_before": before,
                       "accepted_after": len(accepted),
                       "message": f"{thin}, so the search ran a second time with new queries: "
                                  f"{before} → {len(accepted)} on-topic paper(s)."}

    if not unique:
        raise PipelineError("No papers found in any source. Try rephrasing the question "
                            "or enabling more sources.")
    if rejected:
        examples = sorted(rejected, key=lambda p: p.relevance_score)[:5]
        yield {"type": "relevance_filter", "accepted": len(accepted), "rejected": len(rejected),
               "rejected_examples": [
                   {"title": p.title, "relevance_score": p.relevance_score, "reason": p.relevance_reason}
                   for p in examples
               ]}
        yield {"type": "status", "stage": "search", "state": "running",
               "message": f"Relevance screening rejected {len(rejected)} of {len(unique)} "
                          "unique paper(s) as off-topic."}
    if not accepted:
        raise PipelineError(
            "No paper in the search results was judged relevant to the research question. "
            "Try rephrasing the question or providing more specific keywords."
        )

    # Past papers only the library supplied, and the gate let through.
    seeded_count = sum(p.source == LIBRARY_SOURCE for p in accepted)
    ranked = rank(accepted, per_source)
    corpus = ranked[:corpus_size]
    for i, p in enumerate(corpus, start=1):
        p.id = i
    # Before full text and analysis, so no later step ever reads a dropped paper.
    async for item in _approve_corpus(corpus, ranked[corpus_size:], approve):
        if isinstance(item, list):
            corpus = item
        else:
            yield item
    yield {"type": "status", "stage": "search", "state": "done",
           "message": f"Found {len(all_papers)} papers → {len(unique)} unique → "
                      f"{len(accepted)} passed relevance screening → analyzing top {len(corpus)}"}
    yield {"type": "corpus", "total_found": len(all_papers), "unique": len(unique),
           "relevant": len(accepted), "analyzed": len(corpus),
           "papers": [p.to_client_dict() for p in corpus]}

    reused = 0
    if memory and fulltext_enabled:
        try:
            stored = library.stored_excerpts([p.doi for p in corpus if p.doi])
        except Exception as error:
            stored = {}
            yield _library_error(error)
        for p in corpus:
            if p.doi in stored:
                p.future_text = stored[p.doi]
                reused += 1
    if reused or seeded_count:
        # count: library reads this run used, i.e. reused_excerpts + seeded_papers.
        yield {"type": "memory_hit", "count": reused + seeded_count,
               "reused_excerpts": reused, "seeded_papers": seeded_count,
               "message": f"Research library: reused {reused} stored excerpt(s); "
                          f"{seeded_count} past paper(s) passed the relevance gate."}

    candidates: list[dict] = []
    verification_papers: list[Paper] = []
    if fulltext_enabled:
        yield {"type": "status", "stage": "fulltext", "state": "running",
               "message": "Retrieving full texts (open access)…"}
        async for event in _collect_fulltext(corpus, gateway_run, transport):
            if event["type"] != "fulltext_progress":
                yield event  # the gateway's rate_limited / provider_exhausted
                continue
            yield {"type": "status", "stage": "fulltext", "state": "running",
                   "message": f"Retrieving full texts… {event['ok']} of {event['done']} "
                              f"tried (of {event['total']} candidates)"}
        with_text = papers_with_fulltext(corpus)
        yield {"type": "status", "stage": "fulltext", "state": "done",
               "message": f"Discussion/Future-research sections retrieved for "
                          f"{len(with_text)} papers"}
        yield fulltext_done_event(corpus)

        # --- Stage: mine author-flagged future-research statements ----------
        if with_text:
            yield {"type": "status", "stage": "mine", "state": "running",
                   "message": "Extracting author-flagged future-research statements…"}
            try:
                candidates, mine_warning = await mine_future_research(with_text, fast_model)
                # Every candidate gets a full entry in the final gap-analysis
                # JSON (question/status/verdict/evidence/recommendation), so an
                # uncapped candidate count scales the final synthesis call's
                # output size directly. Mining already orders candidates most-
                # to-least substantive, so keep the strongest ones.
                candidates = candidates[:MAX_CANDIDATES_FOR_ANALYSIS]
            except PipelineError as e:
                # Mining is best-effort. A rate-limit/connection error to the
                # local model here must not discard the search/full-text work
                # already done — fall through to the main gap analysis with
                # no author-flagged candidates.
                candidates = []
                yield {"type": "source_error", "source": "Full-text mining",
                       "message": f"{e} Continuing without author-flagged analysis."}
                yield {"type": "status", "stage": "mine", "state": "skipped",
                       "message": "Full-text mining unavailable — skipped"}
            else:
                if mine_warning:
                    yield {"type": "source_error", "source": "Full-text mining",
                           "message": mine_warning}
                if candidates:
                    yield {"type": "status", "stage": "mine", "state": "done",
                           "message": f"Found {len(candidates)} author-flagged candidate(s)"}
                    yield {"type": "candidates",
                           "candidates": [{"question": c["question"],
                                           "source_paper_ids": c["source_paper_ids"]}
                                          for c in candidates]}
                else:
                    yield {"type": "status", "stage": "mine", "state": "done",
                           "message": "No substantive author-flagged statements found"}
        else:
            yield {"type": "status", "stage": "mine", "state": "skipped",
                   "message": "Skipped — no full texts retrieved"}

        # --- Stage: verify candidates against the wider literature ----------
        if candidates:
            yield {"type": "status", "stage": "verify", "state": "running",
                   "message": f"Checking whether {len(candidates)} candidate(s) have "
                              "already been answered…"}
            verify = gateway_run.task(_verify_candidates(candidates, corpus, transport,
                                                         verification_searches))
            async for event in gateway_run.relay(verify):
                yield event
            verification_papers, warnings = verify.result()
            for w in warnings:
                yield w
            if verification_papers:
                yield {"type": "verification_corpus",
                       "papers": [p.to_client_dict() for p in verification_papers]}
            yield {"type": "status", "stage": "verify", "state": "done",
                   "message": f"Verification searches added {len(verification_papers)} "
                              "papers to check against"}
        else:
            yield {"type": "status", "stage": "verify", "state": "skipped",
                   "message": "Skipped — no candidates to verify"}

    if (cached := gateway_run.cache_event()) is not None:
        yield cached  # searches, full text and verification answered from disk (#6)
    yield {"type": "status", "stage": "analyze", "state": "running",
           "message": "Synthesizing the corpus and identifying gaps (this takes a few minutes)…"}
    async for event in analyze_gaps(question, corpus, candidates, verification_papers,
                                    analysis_model):
        yield event
    yield {"type": "status", "stage": "analyze", "state": "done", "message": "Analysis complete."}
