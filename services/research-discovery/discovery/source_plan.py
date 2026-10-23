"""Which databases to search, and with what query (#16).

The model sees the four sources as tools that take one query each and calls
the ones that fit the question. Its calls are validated: a call to an
unknown tool, or without a usable query string, rejects the whole plan.
Whenever the plan is rejected, the model makes no call, or the provider
can't call tools (Gemini), a deterministic fallback is used: the request's
preferred databases if it names any, else the default sources filtered by
domain (no PubMed for mechanical, civil, electrical/electronics and
cloud/devops; computer science keeps it for health informatics). The fallback is
never worse than searching everything, which is what Discovery did before.
"""

from __future__ import annotations

import re

from shared.utilities import llm_provider as llm

from .sources import SOURCES

# What each source is good for, shown to the model in its tool description.
_STRENGTHS = {
    "s2": "broad coverage of every discipline, strong in computer science",
    "openalex": "the broadest index of scholarly works in every discipline",
    "pubmed": "biomedical, clinical, health and life-science literature only",
    "arxiv": "preprints in physics, mathematics, computer science, engineering and quantitative fields",
}
# Domains whose papers PubMed doesn't index; spending its quota there finds nothing.
_NOT_PUBMED = re.compile(r"\b(mechanical|civil|electrical|electronics|ece|eee|cloud|devops)\b")
MAX_QUERY_CHARS = 300

PLAN_SYSTEM = (
    "You choose which literature databases to search for a research question. Call the "
    "search tool of every database likely to hold relevant papers, each with a search query "
    "suited to that database, and skip databases outside the question's field."
)


def tools(source_keys: list[str]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": f"search_{key}",
        "description": f"Search {SOURCES[key][0]}: {_STRENGTHS[key]}.",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string",
                                                "description": "the search query for this database"}}},
    }} for key in source_keys]


def _valid(calls: list[dict], source_keys: list[str]) -> dict[str, str] | None:
    """{source key: query} when every call is a known tool with a usable query, else None."""
    plan: dict[str, str] = {}
    for call in calls:
        name, arguments = call.get("name"), call.get("arguments")
        key = name.removeprefix("search_") if isinstance(name, str) else None
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if key not in source_keys or not isinstance(query, str) \
                or not query.strip() or len(query) > MAX_QUERY_CHARS:
            return None
        plan.setdefault(key, query.strip())
    return plan or None


def _named(source_keys: list[str], preferred: list[str] | None) -> list[str]:
    """The sources the request names (by key or label), among those available."""
    return [k for k in source_keys
            if any(p.strip().lower() in (k, SOURCES[k][0].lower()) for p in preferred or [])]


def fallback(source_keys: list[str], domain: str | None,
             preferred: list[str] | None = None) -> list[str]:
    """The deterministic plan: preferred databases, else the domain filter."""
    if named := _named(source_keys, preferred):
        return named
    domain_words = (domain or "").lower().replace("_", " ")
    return [k for k in source_keys if not (k == "pubmed" and _NOT_PUBMED.search(domain_words))]


async def plan_sources(question: str, queries: dict, source_keys: list[str], model: str | None,
                       domain: str | None = None, preferred: list[str] | None = None) -> dict:
    """{"mode": "model" | "fallback", "reason": str, "queries": {key: query dict}}.
    Never raises: any model failure is a fallback with its reason."""
    if _named(source_keys, preferred):  # databases the user named win over the model's choice
        reason = "the request names its databases"
    else:
        try:
            calls = await llm.complete_tools(
                system=PLAN_SYSTEM, user=f"Research question: {question}"
                + (f"\nField: {domain}" if domain else ""),
                tools=tools(source_keys), model=model)
        except Exception as error:  # LLMProviderError, a malformed reply: never crash the run
            calls, reason = [], f"the model could not be asked ({type(error).__name__})"
        else:
            reason = "the model called no tool"
        if calls is None:
            reason = "this model provider does not support tool calls"
        elif calls:
            chosen = _valid(calls, source_keys)
            if chosen:
                return {"mode": "model", "reason": "chosen by the model",
                        "queries": {k: {**queries, "keyword": q, "pubmed": q, "arxiv": q}
                                    for k, q in chosen.items()}}
            reason = "the model's tool calls were invalid (unknown tool or bad arguments)"
    keys = fallback(source_keys, domain, preferred) or list(source_keys)
    return {"mode": "fallback", "reason": reason, "queries": {k: queries for k in keys}}
