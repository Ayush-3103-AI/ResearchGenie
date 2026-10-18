"""Semantic Scholar Graph API connector (free, keyless, best effort)."""

from __future__ import annotations

import httpx

from shared.utilities import scholarly_gateway

from ..models import Paper, normalize_doi

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,year,venue,citationCount,authors,externalIds,url,openAccessPdf"


async def search(http: httpx.AsyncClient, queries: dict, limit: int) -> list[Paper]:
    params = {"query": queries["keyword"], "limit": min(limit, 100), "fields": FIELDS}

    # Keyless by decision: no API key is sent. The shared anonymous pool 429s
    # often, so the gateway paces S2 at 1 request per ~3 s and trips its
    # breaker on the first 429 burst (ProviderExhausted) — Discovery then
    # continues on the other sources.
    resp = await scholarly_gateway.get(http, SEARCH_URL, params=params)
    resp.raise_for_status()

    papers = []
    for rank, item in enumerate(resp.json().get("data") or []):
        if not item.get("title"):
            continue
        external_ids = item.get("externalIds") or {}
        papers.append(Paper(
            title=item["title"],
            abstract=item.get("abstract"),
            year=item.get("year"),
            venue=item.get("venue") or None,
            authors=[a.get("name", "") for a in (item.get("authors") or []) if a.get("name")],
            citations=item.get("citationCount"),
            doi=normalize_doi(external_ids.get("DOI")),
            url=item.get("url"),
            source="Semantic Scholar",
            relevance_rank=rank,
            pdf_url=(item.get("openAccessPdf") or {}).get("url"),
        ))
    return papers
