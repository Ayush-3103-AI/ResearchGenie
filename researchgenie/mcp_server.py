"""Read-only MCP server (#20): ResearchGenie's runs, library and DOI checker
for any MCP client, over stdio, on the official MCP SDK.

    researchgenie-mcp

Six tools, none of which writes a run folder, the library or the config,
or starts a run (MCP request timeouts are short; long runs go through the
headless CLI / skill). The two tools that reach the network go through the
scholarly gateway, so they share its pacing, keys and response cache.
Config snippets for OpenClaw, Claude Code and Claude Desktop are in
docs/mcp-server.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-discovery"))

from orchestrator import pipeline  # noqa: E402
from researchgenie import config as rg_config  # noqa: E402
from researchgenie import headless  # noqa: E402
from shared.utilities import library  # noqa: E402

# Text files a client may read; paper.pdf is binary and stays a path in summaries.
READABLE_FILES = tuple(name for name in pipeline.FINAL_FILES if not name.endswith(".pdf"))
MAX_ARTIFACT_CHARS = 200_000
_transport: httpx.AsyncBaseTransport | None = None  # test seam (httpx.MockTransport)

server = FastMCP("researchgenie")


def _run_directory(run_id: str) -> Path | None:
    """A run folder under the output root, else a past run the library knows."""
    found = pipeline.find_run_directory(run_id, pipeline.DEFAULT_OUTPUT_ROOT)
    if found is None and library.library_path().is_file():
        past = library.get_run(run_id)
        found = Path(past["run_dir"]) if past and Path(past["run_dir"]).is_dir() else None
    return found


@server.tool()
def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Recent research runs in the output folder, newest first: run id,
    question, state (running / failed / completed) and folder."""
    root = pipeline.DEFAULT_OUTPUT_ROOT
    folders = sorted((d for d in root.iterdir() if d.is_dir() and (d / "events.jsonl").is_file()),
                     key=lambda d: (d / "events.jsonl").stat().st_mtime, reverse=True) \
        if root.is_dir() else []
    runs = []
    for folder in folders[:max(1, min(limit, 200))]:
        status = headless.status_of(folder)
        try:
            question = json.loads((folder / "00_request.json").read_text(encoding="utf-8"))["research_question"]
        except (OSError, ValueError, KeyError):
            question = None
        runs.append({"run_id": status["run_id"], "question": question, "state": status["state"],
                     "stage": status["stage"], "run_directory": status["run_directory"]})
    return runs


@server.tool()
def read_run_artifact(run_id: str, name: str) -> dict[str, Any]:
    """The text of one final file of a run. `name` is one of: draft.md,
    quality_report.md, validation_report.md, references.md,
    evidence_table.md, paper.tex. Anything else is refused."""
    if name not in READABLE_FILES:
        raise ValueError(f"Only these files can be read: {', '.join(READABLE_FILES)}.")
    directory = _run_directory(run_id)
    if directory is None:
        raise ValueError(f"No run {run_id!r}.")
    # Imported copies under docs/research-papers/ keep their files at the top level.
    for path in (directory / "final" / name, directory / name):
        resolved = path.resolve()
        if resolved.is_relative_to(directory.resolve()) and resolved.is_file():
            text = resolved.read_text(encoding="utf-8", errors="replace")
            return {"run_id": run_id, "name": name, "truncated": len(text) > MAX_ARTIFACT_CHARS,
                    "text": text[:MAX_ARTIFACT_CHARS]}
    raise ValueError(f"{name} was not produced by run {run_id!r}.")


@server.tool()
def search_library(query: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Past runs and stored papers in the local research library whose
    question, title or abstract matches the query."""
    if not library.library_path().is_file():  # never create one just to search it
        return {"runs": [], "papers": []}
    limit = max(1, min(limit, 50))
    return {"runs": library.search_runs(query, limit=limit),
            "papers": [{k: p[k] for k in ("doi", "title", "year", "venue", "abstract")}
                       for p in library.search_papers(query, limit=limit)]}


@server.tool()
def summarize_run(run_id: str) -> dict[str, Any]:
    """One run at a glance: question, state, quality scores, verification
    counts, draft status, the papers it used and its final files."""
    directory = _run_directory(run_id)
    if directory is None:
        raise ValueError(f"No run {run_id!r}.")
    summary = headless.result_of(directory)
    try:
        summary["question"] = json.loads(
            (directory / "00_request.json").read_text(encoding="utf-8"))["research_question"]
    except (OSError, ValueError, KeyError):
        summary["question"] = None
    try:
        discovery = json.loads((directory / "01_discovery" / "result.json").read_text(encoding="utf-8"))
        summary["papers"] = [{k: p.get(k) for k in ("title", "year", "doi")}
                             for p in discovery.get("selected_papers", [])]
        summary["research_gaps"] = [g["title"] for g in discovery.get("research_gaps", [])]
    except (OSError, ValueError):
        summary["papers"], summary["research_gaps"] = [], []
    return summary


@server.tool()
async def search_literature(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search the scholarly literature (OpenAlex, through ResearchGenie's
    gateway and cache). Returns title, authors, year, venue, DOI, citations."""
    from discovery.sources import openalex
    async with httpx.AsyncClient(timeout=45, follow_redirects=True, transport=_transport) as http:
        papers = await openalex.search(http, {"keyword": query}, max(1, min(limit, 25)),
                                       require_abstract=False)
    return [{"title": p.title, "authors": p.authors[:6], "year": p.year, "venue": p.venue,
             "doi": p.doi, "citations": p.citations, "url": p.url} for p in papers]


@server.tool()
async def verify_doi(doi: str) -> dict[str, Any]:
    """Whether a DOI is real: 'verified' (Crossref or OpenAlex knows it),
    'not_found' (neither does — possibly made up) or 'unverifiable' (no
    answer right now, which says nothing about the DOI)."""
    check_reference = pipeline.verification_service.check_reference
    async with httpx.AsyncClient(transport=_transport) as http:
        check = await check_reference(http, f"doi: {doi.strip()}")
    if check.doi is None:
        return {"doi": doi, "status": "unverifiable", "detail": "That is not a DOI (10.xxxx/...)."}
    status = "verified" if check.verified else \
        "unverifiable" if (check.issue or "").startswith("unverifiable") else "not_found"
    return {"doi": check.doi, "status": status, "source": check.verification_source,
            "detail": check.issue}


def main() -> None:
    rg_config.apply_scholarly(rg_config.load())  # the keys and email `researchgenie config` saved
    server.run()  # stdio


if __name__ == "__main__":
    main()
