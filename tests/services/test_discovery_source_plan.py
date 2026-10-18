"""Model-chosen sources via tool calling, with a fallback by domain (#16).

Discovery runs end to end over an httpx.MockTransport that records which
hosts were searched with which query; the model's tool calls are faked."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from discovery import pipeline  # noqa: E402
from shared.utilities import llm_provider  # noqa: E402

ALL = ["s2", "openalex", "pubmed", "arxiv"]
QUESTION = "Fatigue life of welded steel joints under variable amplitude loading"


def _work(i: int) -> dict:
    return {"id": f"https://openalex.org/W{i}",
            "display_name": f"Fatigue life of welded steel joints under variable loading, study {i}",
            "abstract_inverted_index": {w: [n] for n, w in enumerate(
                ["Fatigue", "life", "welded", "steel", "joints", "variable", "amplitude", "loading"])},
            "publication_year": 2022, "cited_by_count": 10, "doi": f"https://doi.org/10.1000/f.{i}",
            "authorships": [{"author": {"display_name": "A. Author"}}]}


def _handler(seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen.append((host, dict(request.url.params)))
        if host == "api.openalex.org":
            return httpx.Response(200, json={"results": [_work(i) for i in range(8)]})
        if host == "api.semanticscholar.org":
            return httpx.Response(200, json={"data": []})
        if host == "eutils.ncbi.nlm.nih.gov":
            return httpx.Response(200, json={"esearchresult": {"idlist": []}})
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    return handle


def _discover(monkeypatch, tool_calls, domain=None, preferred=None, real=None):
    """tool_calls: what the fake model returns (a list, None, or an exception);
    `real` is the unpatched complete_tools to use instead of a fake."""
    asked = []

    async def fake_tools(**kwargs):
        asked.append(kwargs)
        if real is not None:
            return await real(**kwargs)
        if isinstance(tool_calls, Exception):
            raise tool_calls
        return tool_calls

    async def fake_queries(question, model, feedback=None):
        return {"keyword": "welded joint fatigue", "pubmed": "fatigue[tiab]",
                "arxiv": "fatigue welded", "interpretation": "i"}

    monkeypatch.setattr(llm_provider, "complete_tools", fake_tools)
    monkeypatch.setattr(pipeline, "generate_queries", fake_queries)
    seen: list = []

    async def go():
        events = []
        stream = pipeline.run_pipeline(QUESTION, ALL, per_source=10, corpus_size=6,
                                       fulltext_enabled=False, domain=domain,
                                       preferred_databases=preferred,
                                       transport=httpx.MockTransport(_handler(seen)))
        try:
            async for event in stream:
                events.append(event)
                if event["type"] == "corpus":
                    break
        finally:
            await stream.aclose()
        return events

    events = asyncio.run(go())
    [plan] = [e for e in events if e["type"] == "tool_plan"]
    return plan, {host for host, _ in seen}, seen, asked


def _call(name, query):
    return {"name": name, "arguments": {"query": query}}


def test_the_model_chooses_the_sources_and_their_queries(monkeypatch):
    plan, hosts, seen, asked = _discover(monkeypatch, [
        _call("search_openalex", "welded joint fatigue variable amplitude"),
        _call("search_arxiv", "fatigue crack welded steel")])
    assert plan["mode"] == "model"
    assert [(p["source"], p["query"]) for p in plan["plan"]] == [
        ("OpenAlex", "welded joint fatigue variable amplitude"), ("arXiv", "fatigue crack welded steel")]
    assert "chosen by the model" in plan["message"]
    assert hosts == {"api.openalex.org", "export.arxiv.org"}
    assert ("api.openalex.org", {"search": "welded joint fatigue variable amplitude",
                                 "per-page": "10", "filter": "has_abstract:true"}) in seen
    # The model was offered all four sources as tools that take a query.
    assert {t["function"]["name"] for t in asked[0]["tools"]} == {f"search_{k}" for k in ALL}


def test_no_tool_calls_fall_back_by_domain_and_never_query_pubmed_for_mechanical(monkeypatch):
    plan, hosts, _, _ = _discover(monkeypatch, [], domain="MECHANICAL")
    assert plan["mode"] == "fallback" and "called no tool" in plan["reason"]
    assert "eutils.ncbi.nlm.nih.gov" not in hosts
    assert hosts == {"api.openalex.org", "api.semanticscholar.org", "export.arxiv.org"}
    assert "default sources" in plan["message"]


@pytest.mark.parametrize("domain", ["Civil Engineering", "EEE", "Electrical & Electronics Engineering"])
def test_other_engineering_domains_skip_pubmed_too(monkeypatch, domain):
    _, hosts, _, _ = _discover(monkeypatch, None, domain=domain)
    assert "eutils.ncbi.nlm.nih.gov" not in hosts


def test_a_biomedical_or_unset_domain_keeps_pubmed(monkeypatch):
    _, hosts, _, _ = _discover(monkeypatch, None, domain="Neuroscience")
    assert "eutils.ncbi.nlm.nih.gov" in hosts


@pytest.mark.parametrize("calls", [
    [_call("search_scopus", "fatigue")],                                  # unknown tool
    [{"name": "search_openalex", "arguments": {"q": "fatigue"}}],         # bad argument name
    [{"name": "search_openalex", "arguments": {"query": 42}}],            # bad argument type
    [{"name": "search_openalex", "arguments": "fatigue"}],                # not an object
    [_call("search_openalex", "  ")],                                     # empty query
    [_call("search_openalex", "ok"), {"name": None, "arguments": None}],  # one bad call rejects the plan
])
def test_invalid_tool_calls_are_rejected_and_fall_back(monkeypatch, calls):
    plan, hosts, _, _ = _discover(monkeypatch, calls, domain="MECHANICAL")
    assert plan["mode"] == "fallback" and "invalid" in plan["reason"]
    assert "eutils.ncbi.nlm.nih.gov" not in hosts and "api.openalex.org" in hosts


def test_a_model_error_falls_back_instead_of_crashing(monkeypatch):
    plan, hosts, _, _ = _discover(monkeypatch, llm_provider.LLMProviderError("Ollama went away"))
    assert plan["mode"] == "fallback" and "could not be asked" in plan["reason"]
    assert len(hosts) == 4


def test_preferred_databases_win_without_asking_the_model(monkeypatch):
    plan, hosts, _, asked = _discover(monkeypatch, [_call("search_arxiv", "x")], preferred=["OpenAlex"])
    assert asked == [] and plan["mode"] == "fallback"
    assert hosts == {"api.openalex.org"}


def test_gemini_goes_through_the_fallback(monkeypatch, real_complete_tools):
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "gemini")

    async def ask():
        return await real_complete_tools(system="s", user="u", tools=[])
    assert asyncio.run(ask()) is None  # no request is made at all

    plan, hosts, _, _ = _discover(monkeypatch, None, domain="CIVIL", real=real_complete_tools)
    assert plan["mode"] == "fallback" and "does not support tool calls" in plan["reason"]
    assert "eutils.ncbi.nlm.nih.gov" not in hosts


def _ollama_answers(monkeypatch, response: httpx.Response):
    real = httpx.AsyncClient
    sent = []

    def handler(request):
        sent.append(request)
        return response

    monkeypatch.setattr(llm_provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return sent


def test_the_ollama_tool_completion_returns_the_calls(monkeypatch, real_complete_tools):
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "ollama")
    sent = _ollama_answers(monkeypatch, httpx.Response(200, json={"message": {"tool_calls": [
        {"function": {"name": "search_openalex", "arguments": {"query": "q"}}}]}}))
    calls = asyncio.run(real_complete_tools(system="s", user="u", tools=[{"type": "function"}]))
    assert calls == [{"name": "search_openalex", "arguments": {"query": "q"}}]
    body = json.loads(sent[0].content)
    assert body["tools"] == [{"type": "function"}] and body["stream"] is False


def test_a_model_without_tool_support_is_none_not_an_error(monkeypatch, real_complete_tools):
    monkeypatch.setattr(llm_provider, "AI_PROVIDER", "ollama")
    _ollama_answers(monkeypatch, httpx.Response(400, json={"error": "qwen does not support tools"}))
    assert asyncio.run(real_complete_tools(system="s", user="u", tools=[])) is None
