"""arXiv Atom API connector (free, no key)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from shared.utilities import scholarly_gateway

from ..models import Paper

QUERY_URL = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


async def search(http: httpx.AsyncClient, queries: dict, limit: int) -> list[Paper]:
    # Paced at arXiv's 1 request per 3 s on one connection, with retry — see
    # shared/utilities/scholarly_gateway.py (DECISIONS.md D-028 for the history).
    resp = await scholarly_gateway.get(http, QUERY_URL, params={
        "search_query": f"all:{queries['arxiv']}",
        "max_results": min(limit, 100),
        "sortBy": "relevance",
    })
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    papers = []
    for rank, entry in enumerate(root.findall(f"{ATOM}entry")):
        title_el = entry.find(f"{ATOM}title")
        if title_el is None or not (title_el.text or "").strip():
            continue
        summary_el = entry.find(f"{ATOM}summary")
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        abs_url = entry.findtext(f"{ATOM}id")
        pdf_url = abs_url.replace("/abs/", "/pdf/") if abs_url and "/abs/" in abs_url else None

        papers.append(Paper(
            title=" ".join(title_el.text.split()),
            abstract=" ".join((summary_el.text or "").split()) or None if summary_el is not None else None,
            year=year,
            venue="arXiv (preprint)",
            authors=[
                name.text for name in entry.findall(f"{ATOM}author/{ATOM}name") if name.text
            ],
            citations=None,
            doi=None,
            url=abs_url,
            source="arXiv",
            relevance_rank=rank,
            pdf_url=pdf_url,
        ))
    return papers
