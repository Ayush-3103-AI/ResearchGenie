"""PubMed E-utilities connector (esearch → efetch, free, no key)."""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET

import httpx

from shared.utilities import scholarly_gateway

from ..models import Paper, normalize_doi

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _text(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    return "".join(el.itertext()).strip() or None


async def search(http: httpx.AsyncClient, queries: dict, limit: int) -> list[Paper]:
    # Both calls go through the gateway, which paces NCBI at 3 req/s, retries
    # transient errors and adds the required `tool` and `email` parameters.
    resp = await scholarly_gateway.get(http, ESEARCH_URL, params={
        "db": "pubmed",
        "term": queries["pubmed"],
        "retmax": min(limit, 100),
        "retmode": "json",
        "sort": "relevance",
    })
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    resp = await scholarly_gateway.get(http, EFETCH_URL, params={
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    })
    resp.raise_for_status()
    # Parse off the event loop: efetch returns a large XML document and the walk is
    # pure-Python CPU work that would otherwise block the other sources' requests.
    return await asyncio.to_thread(_parse_efetch, resp.text)


def _parse_efetch(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)

    papers = []
    for rank, article in enumerate(root.iter("PubmedArticle")):
        title = _text(article.find(".//ArticleTitle"))
        if not title:
            continue

        abstract_parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            part = _text(node)
            if part:
                abstract_parts.append(f"{label}: {part}" if label else part)
        abstract = " ".join(abstract_parts) or None

        year = None
        year_el = article.find(".//JournalIssue/PubDate/Year")
        if year_el is not None and year_el.text and year_el.text.isdigit():
            year = int(year_el.text)
        else:
            medline_date = _text(article.find(".//JournalIssue/PubDate/MedlineDate"))
            if medline_date and medline_date[:4].isdigit():
                year = int(medline_date[:4])

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last, fore = _text(author.find("LastName")), _text(author.find("ForeName"))
            if last:
                authors.append(f"{fore} {last}" if fore else last)

        # Scope to the article's OWN id list (a direct child of PubmedData).
        # `.//ArticleIdList` also matches the lists inside PubmedData/ReferenceList,
        # so the loop would otherwise pick up a cited reference's DOI/PMCID.
        doi = None
        pmcid = None
        for aid in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = normalize_doi(aid.text)
            elif aid.get("IdType") == "pmc" and aid.text:
                pmcid = aid.text.strip()
        if not doi:  # some records carry the DOI only as an ELocationID
            for eloc in article.findall("MedlineCitation/Article/ELocationID"):
                if eloc.get("EIdType") == "doi" and eloc.text:
                    doi = normalize_doi(eloc.text)
                    break
        pmid = _text(article.find(".//PMID"))

        papers.append(Paper(
            title=title.rstrip("."),
            abstract=abstract,
            year=year,
            venue=_text(article.find(".//Journal/Title")),
            authors=authors,
            citations=None,  # PubMed doesn't expose citation counts
            doi=doi,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            source="PubMed",
            relevance_rank=rank,
            pmcid=pmcid,
        ))
    return papers
