"""The read-only MCP server (#20), through a real MCP client session (the
SDK's in-memory transport) against a temporary run folder and library."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from orchestrator import pipeline as orchestrator_pipeline  # noqa: E402
from researchgenie import mcp_server  # noqa: E402
from shared.utilities import library  # noqa: E402
from test_pipeline import (  # noqa: E402
    _fake_run_discovery, _fake_run_qa, _fake_run_verification, _fake_run_writing,
)

TOOLS = {"list_runs", "read_run_artifact", "search_library", "summarize_run",
         "search_literature", "verify_doi"}
REAL_DOI = "10.1038/nature14539"


@pytest.fixture
def finished_run(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_pipeline, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    for service, name, fake in (
            (orchestrator_pipeline.discovery_service, "run_discovery", _fake_run_discovery),
            (orchestrator_pipeline.writing_service, "run_writing", _fake_run_writing),
            (orchestrator_pipeline.verification_service, "run_verification", _fake_run_verification),
            (orchestrator_pipeline.qa_service, "run_quality_assurance", _fake_run_qa)):
        monkeypatch.setattr(service, name, fake)

    async def run():
        request = orchestrator_pipeline.ResearchRequest(research_question="Does sleep aid memory?")
        return [e async for e in orchestrator_pipeline.run_pipeline(request, run_id="run42")]

    asyncio.run(run())  # also writes the run into the (temporary) library
    return tmp_path / "outputs" / "does-sleep-aid-memory_run42"


def _doi_transport(request: httpx.Request) -> httpx.Response:
    if request.url.host == "api.crossref.org":
        return httpx.Response(200 if REAL_DOI in request.url.path else 404, json={})
    if "doi:" in request.url.path:  # OpenAlex fallback
        return httpx.Response(404)
    return httpx.Response(200, json={"results": [{
        "id": "https://openalex.org/W1", "display_name": "Sleep and memory",
        "publication_year": 2020, "cited_by_count": 7, "doi": "https://doi.org/10.1/sleep",
        "authorships": [{"author": {"display_name": "A. Author"}}]}]})


def _call(tool: str, **arguments):
    async def go():
        async with create_connected_server_and_client_session(mcp_server.server) as client:
            if tool == "list_tools":
                return await client.list_tools()
            return await client.call_tool(tool, arguments)
    return asyncio.run(go())


def _data(result):
    assert not result.isError, result.content
    if result.structuredContent is not None:
        data = result.structuredContent
        return data.get("result", data) if set(data) == {"result"} else data
    return json.loads(result.content[0].text)


def _snapshot(*paths):
    return {str(p): p.stat().st_mtime_ns for root in paths for p in sorted(Path(root).rglob("*"))
            if p.is_file()}


def test_the_six_read_only_tools_return_the_right_data(finished_run, monkeypatch):
    monkeypatch.setattr(mcp_server, "_transport", httpx.MockTransport(_doi_transport))
    before = _snapshot(finished_run, library.library_path().parent)

    assert {t.name for t in _call("list_tools").tools} == TOOLS

    [run] = _data(_call("list_runs"))
    assert run["run_id"] == "run42" and run["state"] == "completed"
    assert run["question"] == "Does sleep aid memory?"

    artifact = _data(_call("read_run_artifact", run_id="run42", name="draft.md"))
    assert artifact["text"] == (finished_run / "final" / "draft.md").read_text(encoding="utf-8")

    found = _data(_call("search_library", query="sleep memory paper"))
    assert [r["run_id"] for r in found["runs"]] == ["run42"]
    assert found["papers"][0]["doi"] == "10.1/x"

    summary = _data(_call("summarize_run", run_id="run42"))
    assert summary["state"] == "completed" and summary["scores"]["overall"] == 5.0
    assert summary["papers"][0]["title"] == "Paper One" and "draft.md" in summary["final_files"]

    papers = _data(_call("search_literature", query="sleep memory"))
    assert papers[0]["title"] == "Sleep and memory" and papers[0]["doi"] == "10.1/sleep"

    assert _data(_call("verify_doi", doi=REAL_DOI))["status"] == "verified"
    made_up = _data(_call("verify_doi", doi="10.9999/made.up.2026"))
    assert made_up["status"] == "not_found" and "fabricated" in made_up["detail"]

    # Nothing in the run folder or the library changed.
    assert _snapshot(finished_run, library.library_path().parent) == before


@pytest.mark.parametrize("name", ["../../events.jsonl", "../00_request.json", "metadata.json",
                                  "paper.pdf", "draft.md/../../00_request.json", "/etc/passwd"])
def test_artifact_reads_outside_the_final_files_are_refused(finished_run, name):
    result = _call("read_run_artifact", run_id="run42", name=name)
    assert result.isError and "Only these files" in result.content[0].text


def test_unknown_runs_are_refused(finished_run):
    assert _call("summarize_run", run_id="nope").isError
    assert _call("read_run_artifact", run_id="../run42", name="draft.md").isError


def test_no_tool_starts_a_run_or_writes(monkeypatch):
    """The server only exposes the six tools above; none takes a path to
    write or a request to start, and searching a missing library never
    creates one."""
    tools = {t.name: t for t in _call("list_tools").tools}
    for tool in tools.values():
        assert not any(word in tool.name for word in ("run_pipeline", "start", "write", "delete"))
        assert "path" not in tool.inputSchema.get("properties", {})
    assert _data(_call("search_library", query="anything")) == {"runs": [], "papers": []}
    assert not library.library_path().exists()
