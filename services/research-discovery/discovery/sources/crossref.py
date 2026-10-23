"""Crossref works search, for the verification-candidate searches the
OpenAlex budget can't cover (#7): free and keyless, with no credit budget;
the gateway adds the contact mailto. Not a corpus source (not in SOURCES)."""

from __future__ import annotations

import re

import httpx

from shared.utilities import scholarly_gateway

from ..models import Paper, normalize_doi

SEARCH_URL = "https://api.crossref.org/works"
_TAGS = re.compile(r"<[^>]+>")  # Crossref abstracts are JATS XML


async def search(http: httpx.AsyncClient, queries: dict, limit: int) -> list[Paper]:
    resp = await scholarly_gateway.get(http, SEARCH_URL,
                                       params={"query": queries["keyword"], "rows": min(limit, 100)})
    resp.raise_for_status()

    papers = []
    for rank, item in enumerate((resp.json().get("message") or {}).get("items") or []):
        title = " ".join(item.get("title") or []).strip()
        if not title:
            continue
        date = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
        papers.append(Paper(
            title=title,
            abstract=_TAGS.sub("", item.get("abstract") or "").strip() or None,
            year=date[0] if date else None,
            venue=(item.get("container-title") or [None])[0],
            authors=[" ".join(filter(None, (a.get("given"), a.get("family"))))
                     for a in item.get("author") or [] if a.get("family")],
            citations=item.get("is-referenced-by-count"),
            doi=normalize_doi(item.get("DOI")),
            url=item.get("URL"),
            source="Crossref",
            relevance_rank=rank,
        ))
    return papers
