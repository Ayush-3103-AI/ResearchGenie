"""Service-boundary tests for the Trend & Gap Advisor.

Exercises external behaviour at `run_trend_advisor()` — the same level
tests/services/test_discovery_service.py works at — with OpenAlex mocked at
the transport layer and the gap-mining model faked, so no network and no
Ollama are required.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))
sys.path.insert(0, str(ROOT / "services" / "trend-advisor"))

from shared.contracts.trend_contract import TrendAdvisorRequest, TrendAdvisorResult  # noqa: E402
from trend_advisor import velocity  # noqa: E402


def _load_service():
    """Every service's entry point is named service.py, so a plain
    `import service` resolves to whichever one a sibling test module imported
    first (see DECISIONS.md D-010). Load this one by explicit file path under
    a unique name, exactly as orchestrator/pipeline.py does."""
    path = ROOT / "services" / "trend-advisor" / "service.py"
    spec = importlib.util.spec_from_file_location("test_trend_advisor_service_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service()
TrendAdvisorError = service.TrendAdvisorError
run_trend_advisor = service.run_trend_advisor

RECENT = {"T1": ("Rising Topic", 900), "T2": ("Flat Topic", 800), "T3": ("Third Topic", 700)}
PRIOR = {"T1": ("Rising Topic", 100), "T2": ("Flat Topic", 790), "T3": ("Third Topic", 400)}


def _openalex_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "arxiv.org" in str(request.url):
            # The CS-adjacent domains also query arXiv for a secondary signal.
            return httpx.Response(200, text=(
                '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
                "<title>A Preprint</title><summary>Some text.</summary>"
                "<published>2024-01-01T00:00:00Z</published>"
                "<id>http://arxiv.org/abs/2401.00001</id>"
                "</entry></feed>"
            ))
        if "group_by" in params:
            start = int(params["filter"].split("publication_year:")[1].split("-")[0])
            counts = RECENT if start >= 2023 else PRIOR
            return httpx.Response(200, json={"group_by": [
                {"key": f"https://openalex.org/{tid}", "key_display_name": name, "count": n}
                for tid, (name, n) in counts.items()
            ]})
        topic = params["filter"].split("primary_topic.id:")[1].split(",")[0]
        return httpx.Response(200, json={"results": [
            {
                "display_name": f"{topic} Paper {i}", "publication_year": 2024,
                "cited_by_count": 10, "doi": f"https://doi.org/10.1000/{topic}-{i}",
                "abstract_inverted_index": {"Future": [0], "work": [1], "is": [2], "needed": [3]},
                "authorships": [{"author": {"display_name": "A. Author"}}],
                "primary_location": {"source": {"display_name": "A Journal"}},
            }
            for i in range(2)
        ]})
    return httpx.MockTransport(handler)


class _FakeModel:
    """Reports one gap per paper, so every topic clears the recurrence bar."""

    def __init__(self):
        self.calls = 0

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        self.calls += 1
        title = user_prompt.split("Paper title: ")[1].split("\n")[0]
        return response_model.model_validate(
            {"gap_statements": [f"Gap stated by {title}."]}
        )


@pytest.fixture(autouse=True)
def _mock_openalex(monkeypatch):
    """Pin the advisor's OpenAlex calls to the mock transport and to a fixed
    'today', so the year windows the assertions rely on never drift."""
    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        kwargs.setdefault("transport", _openalex_transport())
        kwargs.setdefault("current_year", 2026)
        return await real(domain, **kwargs)

    monkeypatch.setattr(velocity, "find_rising_topics", patched)
    monkeypatch.setattr(service, "find_rising_topics", patched)


async def _collect(request, **kwargs):
    events, result = [], None
    async for event in run_trend_advisor(request, **kwargs):
        if event["type"] == "result":
            result = event["result"]
        else:
            events.append(event)
    return events, result


def test_produces_a_ranked_shortlist_with_evidence(tmp_path):
    events, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CS_AI_ML", shortlist_size=6),
        output_root=tmp_path, model=_FakeModel(),
    ))

    assert isinstance(result, TrendAdvisorResult)
    assert result.domain == "CS_AI_ML"
    assert len(result.shortlist) == 3
    top = result.shortlist[0]
    assert top.growth_metric > 1
    assert top.example_papers and top.example_papers[0].doi
    # Both stages ran and both reported done.
    assert [e["stage"] for e in events if e["state"] == "done"] == ["velocity", "gap_mining"]


def test_gap_flagged_topics_are_listed_before_ungapped_ones(tmp_path):
    class _OneTopicOnly(_FakeModel):
        async def generate_structured(self, *, system_prompt, user_prompt, response_model):
            title = user_prompt.split("Paper title: ")[1].split("\n")[0]
            gaps = [f"Gap from {title}."] if title.startswith("T3") else []
            return response_model.model_validate({"gap_statements": gaps})

    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CIVIL"), output_root=tmp_path, model=_OneTopicOnly(),
    ))

    assert result.shortlist[0].topic == "Third Topic"
    assert result.shortlist[0].gap_signal is True
    assert result.shortlist[0].source == "both"
    assert all(c.gap_signal is False for c in result.shortlist[1:])


def test_run_output_is_saved_to_its_own_timestamped_folder(tmp_path):
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="MECHANICAL"), output_root=tmp_path,
        run_id="abc123", model=_FakeModel(),
    ))

    saved = tmp_path / "mechanical_abc123" / "00_trend_advisor" / "result.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8"))["shortlist"][0]["topic"] == \
        result.shortlist[0].topic


def test_two_runs_keep_two_separate_records(tmp_path):
    """Step 2 (memory) claims every run keeps its own full record. Without a
    distinct run id per run the second call would overwrite the first, and a
    student comparing this week's shortlist with last week's would find only
    one of them."""
    request = TrendAdvisorRequest(domain="MECHANICAL")
    _, first = asyncio.run(_collect(request, output_root=tmp_path, model=_FakeModel()))
    _, second = asyncio.run(_collect(request, output_root=tmp_path, model=_FakeModel()))

    saved = sorted(tmp_path.glob("mechanical_*/00_trend_advisor/result.json"))
    assert len(saved) == 2, "the second run overwrote the first run's record"

    # Run ids are random, so the two files sort in no meaningful order —
    # each is checked on its own rather than paired with a call by position.
    expected = {len(first.shortlist), len(second.shortlist)}
    for path in saved:
        record = json.loads(path.read_text(encoding="utf-8"))
        # Each record is the full shortlist with its evidence, not a stub.
        assert len(record["shortlist"]) in expected
        assert all(topic["example_papers"] for topic in record["shortlist"])
        assert record["generated_at"]


def test_free_text_domain_warns_that_results_are_less_precise(tmp_path):
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="Other", domain_other_name="synthetic biology"),
        output_root=tmp_path, model=_FakeModel(),
    ))

    assert any("free-text domain" in w for w in result.warnings)
    assert result.domain_other_name == "synthetic biology"


def test_unreachable_openalex_raises_rather_than_returning_an_empty_shortlist(tmp_path, monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(refuse), **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)

    with pytest.raises(TrendAdvisorError):
        asyncio.run(_collect(TrendAdvisorRequest(domain="EEE"),
                             output_root=tmp_path, model=_FakeModel()))


def test_no_result_file_is_written_when_the_stage_fails(tmp_path, monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("openalex unreachable", request=request)

    real = velocity.find_rising_topics

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(refuse), **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)

    with pytest.raises(TrendAdvisorError):
        asyncio.run(_collect(TrendAdvisorRequest(domain="EEE"),
                             output_root=tmp_path, model=_FakeModel()))
    assert list(tmp_path.iterdir()) == []


def test_model_outage_is_reported_rather_than_read_as_no_gaps_found(tmp_path, monkeypatch):
    """With the provider down every worker fails, so every topic comes back
    with no gap evidence. That must not be presented as a finding."""

    async def down():
        return False

    monkeypatch.setattr(service.llm_provider, "health_check", down)

    events, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="CIVIL"), output_root=tmp_path,
    ))

    assert any("model was unavailable" in w for w in result.warnings)
    assert any(e["stage"] == "gap_mining" and e["state"] == "failed" for e in events)
    # The velocity half of the stage still delivered.
    assert len(result.shortlist) == 3
    assert all(c.source == "velocity" for c in result.shortlist)


def test_no_arxiv_warning_for_a_domain_that_never_queries_arxiv(tmp_path):
    """Only the CS-adjacent domains use arXiv. Warning that it was
    "unreachable" for Mechanical reports an outage that never happened."""
    _, result = asyncio.run(_collect(
        TrendAdvisorRequest(domain="MECHANICAL"), output_root=tmp_path,
        model=_FakeModel(),
    ))

    assert all(c.arxiv_recent_count is None for c in result.shortlist)
    assert not any("arXiv" in w for w in result.warnings)


def test_a_throttled_openalex_call_is_retried_and_shown_as_a_progress_event(tmp_path, monkeypatch):
    """The advisor's OpenAlex calls run inside its own gateway run (#4): a 429
    is retried rather than failing the stage, and the wait is reported."""
    real = velocity.find_rising_topics
    inner = _openalex_transport()
    throttled = []

    def handler(request):
        if "group_by" in request.url.params and not throttled:
            throttled.append(request)
            return httpx.Response(429, headers={"Retry-After": "1"})
        return inner.handle_request(request)

    async def patched(domain, **kwargs):
        return await real(domain, transport=httpx.MockTransport(handler), current_year=2026, **kwargs)

    monkeypatch.setattr(service, "find_rising_topics", patched)
    events, result = asyncio.run(_collect(TrendAdvisorRequest(domain="MECHANICAL"),
                                          output_root=tmp_path, model=_FakeModel()))

    assert throttled and result.shortlist
    assert any(e["type"] == "rate_limited" for e in events)
