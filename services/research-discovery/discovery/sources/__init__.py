"""Registry of literature sources: key -> (display label, async search function)."""

from . import arxiv, openalex, pubmed, semantic_scholar

SOURCES = {
    "s2": ("Semantic Scholar", semantic_scholar.search),
    "openalex": ("OpenAlex", openalex.search),
    "pubmed": ("PubMed", pubmed.search),
    "arxiv": ("arXiv", arxiv.search),
}

# The URL each source searches, so a caller can ask the gateway whether its
# host is still available in this run (the re-plan does, #15).
SOURCE_URLS = {
    "s2": semantic_scholar.SEARCH_URL,
    "openalex": openalex.SEARCH_URL,
    "pubmed": pubmed.ESEARCH_URL,
    "arxiv": arxiv.QUERY_URL,
}
