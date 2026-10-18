"""Open-access full-text retrieval and Discussion/Future-research section extraction.

Two routes, tried in order of reliability:
  1. Europe PMC full-text JATS XML (by PMCID, or by DOI lookup) — clean structured
     sections, no PDF parsing.
  2. A direct open-access PDF URL from Semantic Scholar / OpenAlex / arXiv, parsed
     with pypdf and sliced with heading heuristics.

The goal is the paper's Discussion / Limitations / Future-research / Conclusion
text, where authors explicitly flag what remains to be studied.
"""

from __future__ import annotations

import asyncio
import io
import re
import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DET
import httpx
from defusedxml.common import DefusedXmlException
from pypdf import PdfReader

from shared.utilities import scholarly_gateway

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_XML_BYTES = 25 * 1024 * 1024
MAX_LOCAL_TEXT_BYTES = 25 * 1024 * 1024  # cap on a local Zotero PDF / ft-cache read
MAX_PDF_PAGES = 60
MAX_PDF_TEXT_CHARS = 400_000  # stop extracting once we clearly have enough
MAX_SECTION_CHARS = 7000

SECTION_KEYWORDS = (
    "discussion", "limitation", "future direction", "future research",
    "future work", "conclusion", "concluding remarks", "implications",
)

_HEADING_RE = re.compile(
    r"\n[ \t]{0,8}(?:\d{1,2}[.\d]*[ \t]{1,4})?"
    r"(general\s+discussion|discussion|limitations?(?:\s+and\s+future\s+(?:directions|research|work))?"
    r"|(?:directions\s+for\s+)?future\s+(?:directions|research|work)"
    r"|conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks)"
    r"[ \t]*\n",
    re.IGNORECASE,
)
_REFS_RE = re.compile(r"\n[ \t]{0,8}(references|bibliography|literature\s+cited)[ \t]*\n", re.IGNORECASE)


def _cap(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= MAX_SECTION_CHARS:
        return text
    # Future-research statements cluster at the end of the discussion — keep both ends.
    half = MAX_SECTION_CHARS // 2
    return text[:half] + "\n[…]\n" + text[-half:]


# ---------------------------------------------------------------- PDF route

def _sections_from_pdf_text(text: str) -> str | None:
    if len(text) < 2000:
        return None  # extraction likely failed (scanned/encoded PDF)
    # Skip the first quarter so an intro mention of "discussion" doesn't match.
    matches = list(_HEADING_RE.finditer(text, len(text) // 4))
    # Prefer the LAST discussion-type heading (multi-study papers have a per-study
    # "Discussion" before the closing "General Discussion"); otherwise the first
    # end-matter heading of any kind, so the slice still covers everything after it.
    discussion = [m for m in matches if "discussion" in m.group(1).lower()]
    match = discussion[-1] if discussion else (matches[0] if matches else None)
    refs = _REFS_RE.search(text, match.end() if match else len(text) // 4)
    if match:
        end = refs.start() if refs else len(text)
        section = text[match.start():end]
    elif refs:
        section = text[max(0, refs.start() - 12000):refs.start()]  # tail before references
    else:
        section = text[-10000:]  # last resort: the tail of the paper
    section = section.strip()
    return _cap(section) if len(section) > 400 else None


def _extract_pdf_text(content: bytes) -> str:
    # Page and character budgets bound the work per PDF: asyncio.wait_for can't
    # cancel pypdf once it's running in a worker thread, so a parse-heavy PDF would
    # otherwise pin a thread past the timeout. (A process pool with a hard kill
    # would be the fuller fix; the budgets keep this good enough for a local tool.)
    try:
        reader = PdfReader(io.BytesIO(content))
        parts = []
        total = 0
        for page in reader.pages[:MAX_PDF_PAGES]:
            try:
                text = page.extract_text() or ""
            except Exception:
                continue  # a corrupt page shouldn't discard the rest of the PDF
            parts.append(text)
            total += len(text)
            if total >= MAX_PDF_TEXT_CHARS:
                break
        return "\n".join(parts)
    except Exception:
        return ""


async def _try_pdf(http: httpx.AsyncClient, url: str) -> str | None:
    # Stream so we can abort an oversized or non-PDF body mid-download instead of
    # buffering the whole thing (these URLs point at arbitrary publisher hosts and
    # several of them run at once, each paced by its host's gateway policy).
    async with scholarly_gateway.stream(http, url) as resp:
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "").lower()
        if "html" in ctype or "text/plain" in ctype:
            return None  # a landing/redirect page, not a PDF
        declared = resp.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_PDF_BYTES:
            return None
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > MAX_PDF_BYTES:
                return None  # stop pulling bytes; don't grow the buffer further
    content = bytes(buf)
    # The %PDF signature may sit behind up to ~1024 bytes of leading whitespace/junk.
    if b"%PDF" not in content[:1024]:
        return None
    # Parse AND slice off the event loop — the heading regex over the extracted text
    # is itself CPU work, and several of these run at once.
    return await asyncio.to_thread(_pdf_to_sections, content)


def _pdf_to_sections(content: bytes) -> str | None:
    return _sections_from_pdf_text(_extract_pdf_text(content))


# ---------------------------------------------------------- Europe PMC route

def _walk_jats(sec: ET.Element, collected: list[str]) -> None:
    title = ("".join((sec.find("title").itertext())) if sec.find("title") is not None else "").strip()
    sec_type = (sec.get("sec-type") or "").lower()
    matched = any(k in title.lower() for k in SECTION_KEYWORDS) or \
        any(k in sec_type for k in ("discussion", "conclusion"))
    if matched:
        parts = [title] if title else []
        parts += ["".join(p.itertext()).strip() for p in sec.iter("p")]
        collected.append("\n".join(x for x in parts if x))
        return  # don't also collect nested subsections separately
    for child in sec.findall("sec"):
        _walk_jats(child, collected)


def _sections_from_jats(xml_text: str) -> str | None:
    # defusedxml: Europe PMC XML is external, so guard against billion-laughs /
    # entity-expansion (stdlib ElementTree expands internal entities).
    root = DET.fromstring(xml_text)
    collected: list[str] = []
    body = root.find(".//body")
    if body is None:
        return None
    for sec in body.findall("sec"):
        _walk_jats(sec, collected)
    text = "\n\n".join(collected).strip()
    return _cap(text) if len(text) > 400 else None


async def _resolve_pmcid(http: httpx.AsyncClient, doi: str) -> str | None:
    resp = await scholarly_gateway.get(http, EPMC_SEARCH, params={
        "query": f'DOI:"{doi}" AND OPEN_ACCESS:y',
        "format": "json",
        "pageSize": 1,
    })
    if resp.status_code != 200:
        return None
    results = resp.json().get("resultList", {}).get("result", [])
    return results[0].get("pmcid") if results else None


async def _try_europe_pmc(http: httpx.AsyncClient, pmcid: str | None, doi: str | None) -> str | None:
    if not pmcid and doi:
        pmcid = await _resolve_pmcid(http, doi)
    if not pmcid:
        return None
    # Stream with a size cap — full-article JATS XML is unbounded and this fetch
    # otherwise buffers whatever the endpoint returns.
    async with scholarly_gateway.stream(http, EPMC_FULLTEXT.format(pmcid=pmcid)) as resp:
        if resp.status_code != 200:
            return None
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > MAX_XML_BYTES:
                return None
    xml_text = bytes(buf).decode("utf-8", errors="ignore")
    try:
        # Parse off the event loop; DefusedXmlException fires on a hostile document.
        return await asyncio.to_thread(_sections_from_jats, xml_text)
    except (ET.ParseError, DefusedXmlException):
        return None


# ------------------------------------------------------------------ public API

def is_candidate(paper) -> bool:
    """Can we plausibly get full text for this paper?"""
    return bool(paper.pmcid or paper.pdf_url or paper.doi)


async def _fetch(http: httpx.AsyncClient, paper) -> str | None:
    try:
        text = await _try_europe_pmc(http, paper.pmcid, paper.doi)
    except scholarly_gateway.ProviderExhausted:
        text = None  # Europe PMC is skipped for this run; the PDF route still works
    if not text and paper.pdf_url:
        text = await _try_pdf(http, paper.pdf_url)
    return text


async def fetch_future_sections(http: httpx.AsyncClient, paper) -> bool:
    """Try to populate paper.future_text. Returns True on success. Never raises."""
    try:
        # wait_for (not asyncio.timeout) so Python 3.10 works too
        text = await asyncio.wait_for(_fetch(http, paper), timeout=75)
    except Exception:
        return False
    if text:
        paper.future_text = text
        return True
    return False
