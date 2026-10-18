"""OpenAlex API connector (free, no key; the gateway adds the contact mailto)."""

from __future__ import annotations

import httpx

from shared.utilities import scholarly_gateway

from ..models import Paper, normalize_doi

SEARCH_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, indices in inverted_index.items():
        for i in indices:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


async def search(http: httpx.AsyncClient, queries: dict, limit: int,
                 require_abstract: bool = True) -> list[Paper]:
    params = {
        "search": queries["keyword"],
        "per-page": min(limit, 100),
    }
    if require_abstract:
        # Corpus papers with no indexed abstract are near-useless to the analysis,
        # but a *verification* search must not hide an answering paper merely because
        # OpenAlex lacks its abstract — callers pass require_abstract=False there.
        params["filter"] = "has_abstract:true"

    # OpenAlex is the sole backend for the per-candidate verification searches,
    # so a transient 429/5xx must not silently lose a candidate's evidence —
    # the gateway retries it (and never retries a not_enough_credits 429).
    resp = await scholarly_gateway.get(http, SEARCH_URL, params=params)
    resp.raise_for_status()

    papers = []
    for rank, work in enumerate(resp.json().get("results") or []):
        title = work.get("display_name")
        if not title:
            continue
        venue = None
        primary = work.get("primary_location") or {}
        if primary.get("source"):
            venue = primary["source"].get("display_name")

        best_oa = work.get("best_oa_location") or {}
        pmcid = None
        pmcid_url = (work.get("ids") or {}).get("pmcid")  # e.g. ".../articles/PMC123/"
        if pmcid_url:
            digits = "".join(ch for ch in pmcid_url.rsplit("PMC", 1)[-1] if ch.isdigit())
            if digits:
                pmcid = f"PMC{digits}"
        papers.append(Paper(
            title=title,
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            year=work.get("publication_year"),
            venue=venue,
            authors=[
                name
                for a in (work.get("authorships") or [])
                # authorship["author"] can be present but null — `.get("author", {})`
                # would then return None and crash; coerce None → {} first.
                if (name := (a.get("author") or {}).get("display_name"))
            ],
            citations=work.get("cited_by_count"),
            doi=normalize_doi(work.get("doi")),
            url=(primary.get("landing_page_url") or work.get("id")),
            source="OpenAlex",
            relevance_rank=rank,
            pdf_url=best_oa.get("pdf_url"),
            pmcid=pmcid,
        ))
    return papers


# ---------------------------------------------------------------------------
# Works-count-by-year queries (Trend & Gap Advisor, pipeline stage 0).
#
# Added here rather than in a second OpenAlex client so the advisor inherits
# this module's gateway-routed pacing, retry/backoff, and
# abstract reconstruction (see DECISIONS.md D-028) instead of re-implementing
# three things that have each already been a real bug once.
# ---------------------------------------------------------------------------

# group_by returns at most 200 buckets per response and does not paginate;
# 200 topics is far more than any shortlist needs, so this is a ceiling we
# deliberately accept rather than a page size to iterate.
GROUP_BY_LIMIT = 200


def _topic_key(group_key: str) -> str:
    """OpenAlex group keys are full URLs ("https://openalex.org/T10028")."""
    return group_key.rsplit("/", 1)[-1]


async def topic_counts(http: httpx.AsyncClient, *, year_range: tuple[int, int],
                       subfield_ids: list[str] | None = None,
                       search: str | None = None) -> dict[str, tuple[str, int]]:
    """Works published per OpenAlex topic within `year_range`, restricted
    either to a set of subfields (the curated domain mapping) or to a
    free-text search (the "Other" domain fallback).

    Returns {topic_id: (topic_display_name, works_count)}. An empty result
    is returned as an empty dict — callers must treat that as a failure to
    surface, never as "no topics are growing".
    """
    start, end = year_range
    filters = [f"publication_year:{start}-{end}"]
    if subfield_ids:
        filters.append(
            "primary_topic.subfield.id:"
            + "|".join(f"subfields/{sid}" for sid in subfield_ids)
        )
    params = {
        "filter": ",".join(filters),
        "group_by": "primary_topic.id",
        "per-page": GROUP_BY_LIMIT,
    }
    if search:
        params["search"] = search

    resp = await scholarly_gateway.get(http, SEARCH_URL, params=params)
    resp.raise_for_status()
    return {
        _topic_key(group["key"]): (group.get("key_display_name") or "", group.get("count", 0))
        for group in (resp.json().get("group_by") or [])
        if group.get("key")
    }


async def works_for_topic(http: httpx.AsyncClient, topic_id: str, *,
                          year_range: tuple[int, int], limit: int = 5) -> list[Paper]:
    """The most-cited recent works whose primary topic is `topic_id`.

    These are both the student-facing evidence for a ranking and the corpus
    the gap-mining swarm reads, so abstracts are required — a paper with no
    indexed abstract can serve neither purpose.
    """
    start, end = year_range
    params = {
        "filter": (f"primary_topic.id:{topic_id},publication_year:{start}-{end},"
                   "has_abstract:true"),
        "sort": "cited_by_count:desc",
        "per-page": min(limit, 100),
    }
    resp = await scholarly_gateway.get(http, SEARCH_URL, params=params)
    resp.raise_for_status()

    papers = []
    for rank, work in enumerate(resp.json().get("results") or []):
        title = work.get("display_name")
        if not title:
            continue
        primary = work.get("primary_location") or {}
        venue = (primary.get("source") or {}).get("display_name")
        papers.append(Paper(
            title=title,
            abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
            year=work.get("publication_year"),
            venue=venue,
            authors=[
                name
                for a in (work.get("authorships") or [])
                if (name := (a.get("author") or {}).get("display_name"))
            ],
            citations=work.get("cited_by_count"),
            doi=normalize_doi(work.get("doi")),
            url=(primary.get("landing_page_url") or work.get("id")),
            source="OpenAlex",
            relevance_rank=rank,
            pdf_url=(work.get("best_oa_location") or {}).get("pdf_url"),
        ))
    return papers
