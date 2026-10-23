"""The local research library: every completed run and the papers it
selected, in one SQLite file (standard library sqlite3 + FTS5) next to the
CLI's config, at ~/.researchgenie/library.sqlite.

    papers     (doi PK, title, abstract, year, venue, discussion_excerpt,
                fulltext_status, first_seen, authors (JSON list), citations)
    runs       (run_id PK, question, created, overall_quality, draft_status, run_dir)
    run_papers (run_id, doi, selected)
    papers_fts (title, abstract)   runs_fts (question)

Written once at the end of every run by orchestrator/pipeline.py, which
calls import_run() on the run folder it just finished, the same function
`researchgenie library import` uses to backfill old run folders. Re-importing
a folder replaces its rows, so a backfill can be repeated safely. A paper
with no DOI is not stored: the DOI is the only key that is the same across
runs. Folders without 01_discovery/result.json (the flattened copies under
docs/research-papers/) give a run row with no papers.

Discovery reads it back (#19): stored_excerpts() spares a paper's full-text
fetch, and search_papers() seeds past papers into the candidate pool.

Literature cards are deliberately NOT stored. Their cache key includes the
question-specific tag definitions (see build_cached_literature_card() in
services/research-writing/writing/modules/literature_card.py), so a card
built for one question is not valid for another.

RESEARCHGENIE_LIBRARY overrides the file's location (tests point it at a
temporary file so the real library is never touched).
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from researchgenie.config import CONFIG_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    doi TEXT PRIMARY KEY, title TEXT, abstract TEXT, year INTEGER, venue TEXT,
    discussion_excerpt TEXT, fulltext_status TEXT, first_seen TEXT,
    authors TEXT, citations INTEGER);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, question TEXT, created TEXT, overall_quality REAL,
    draft_status TEXT, run_dir TEXT);
CREATE TABLE IF NOT EXISTS run_papers (
    run_id TEXT, doi TEXT, selected INTEGER, PRIMARY KEY (run_id, doi));
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(doi UNINDEXED, title, abstract);
CREATE VIRTUAL TABLE IF NOT EXISTS runs_fts USING fts5(run_id UNINDEXED, question);
"""
# Added to `papers` after the first release (#19); connect() adds them to an
# older library file, so a seeded past paper can be cited with its authors.
_ADDED_PAPER_COLUMNS = {"authors": "TEXT", "citations": "INTEGER"}

# Words every research question shares; left in, they would make every past
# run "related" to every new one.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "when", "which", "who", "why", "with", "between", "effect", "effects",
    "impact", "role", "their", "there", "this", "that", "these", "affect",
}


def library_path() -> Path:
    return Path(os.getenv("RESEARCHGENIE_LIBRARY", str(CONFIG_DIR / "library.sqlite")))


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(path) if path else library_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    have = {row["name"] for row in conn.execute("PRAGMA table_info(papers)")}
    for column, kind in _ADDED_PAPER_COLUMNS.items():
        if column not in have:
            conn.execute(f"ALTER TABLE papers ADD COLUMN {column} {kind}")
    return conn


def fts_query(text: str) -> str:
    """Builds a MATCH expression from free text: each content word quoted,
    OR'd together. Raw user text is never passed to MATCH, where quotes,
    hyphens or words like NOT would be parsed as query syntax."""
    words = dict.fromkeys(w for w in re.findall(r"\w+", text.lower())
                          if len(w) > 1 and w not in _STOPWORDS)
    return " OR ".join(f'"{w}"' for w in words)


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip().lower()) or None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def import_run(run_dir: Path | str, conn: sqlite3.Connection | None = None) -> bool:
    """Writes one run folder (metadata.json, plus 01_discovery/result.json
    when present) into the library. Returns False for a folder that is not
    a finished run."""
    run_dir = Path(run_dir).resolve()
    metadata = _read_json(run_dir / "metadata.json")
    if not isinstance(metadata, dict) or not metadata.get("run_id"):
        return False
    question = metadata.get("research_question") or ""
    created = metadata.get("created") or datetime.datetime.fromtimestamp(
        (run_dir / "metadata.json").stat().st_mtime, datetime.timezone.utc).isoformat()
    discovery = _read_json(run_dir / "01_discovery" / "result.json") or {}

    own_conn = conn is None
    conn = conn or connect()
    try:
        with conn:
            run_id = metadata["run_id"]
            conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)", (
                run_id, question, created, metadata.get("overall_quality_score"),
                metadata.get("draft_status"), str(run_dir)))
            conn.execute("DELETE FROM runs_fts WHERE run_id = ?", (run_id,))
            conn.execute("INSERT INTO runs_fts VALUES (?, ?)", (run_id, question))
            for paper in discovery.get("selected_papers") or []:
                doi = _normalise_doi(paper.get("doi"))
                if not doi:
                    continue
                excerpt = paper.get("discussion_excerpt")
                # A later run without the excerpt must not erase one an
                # earlier run already extracted.
                conn.execute("""
                    INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doi) DO UPDATE SET
                        title = excluded.title,
                        abstract = COALESCE(excluded.abstract, papers.abstract),
                        year = COALESCE(excluded.year, papers.year),
                        venue = COALESCE(excluded.venue, papers.venue),
                        discussion_excerpt = COALESCE(excluded.discussion_excerpt, papers.discussion_excerpt),
                        fulltext_status = CASE WHEN COALESCE(excluded.discussion_excerpt,
                            papers.discussion_excerpt) IS NULL THEN 'none' ELSE 'excerpt' END,
                        first_seen = MIN(papers.first_seen, excluded.first_seen),
                        authors = COALESCE(excluded.authors, papers.authors),
                        citations = COALESCE(excluded.citations, papers.citations)
                """, (doi, paper.get("title"), paper.get("abstract"), paper.get("year"),
                      paper.get("venue"), excerpt, "excerpt" if excerpt else "none", created,
                      json.dumps(paper["authors"]) if paper.get("authors") else None,
                      paper.get("citations")))
                row = conn.execute("SELECT title, abstract FROM papers WHERE doi = ?", (doi,)).fetchone()
                conn.execute("DELETE FROM papers_fts WHERE doi = ?", (doi,))
                conn.execute("INSERT INTO papers_fts VALUES (?, ?, ?)", (doi, row["title"], row["abstract"]))
                # Discovery's result lists only the papers it selected.
                conn.execute("INSERT OR REPLACE INTO run_papers VALUES (?, ?, 1)", (run_id, doi))
    finally:
        if own_conn:
            conn.close()
    return True


def import_runs(paths: list[Path | str]) -> int:
    """Imports every run folder (a folder holding metadata.json) under the
    given paths. Returns how many runs were written."""
    conn = connect()
    try:
        return sum(import_run(meta.parent, conn)
                   for path in paths if Path(path).is_dir()
                   for meta in Path(path).rglob("metadata.json"))
    finally:
        conn.close()


def get_run(run_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def search_runs(text: str, limit: int = 10, exclude_run_id: str | None = None) -> list[dict[str, Any]]:
    """Past runs whose question matches, best full-text score first."""
    query = fts_query(text)
    if not query:
        return []
    conn = connect()
    try:
        rows = conn.execute("""
            SELECT runs.* FROM runs_fts JOIN runs USING (run_id)
            WHERE runs_fts MATCH ? AND run_id != ? ORDER BY bm25(runs_fts) LIMIT ?
        """, (query, exclude_run_id or "", limit)).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def search_papers(text: str, limit: int = 10) -> list[dict[str, Any]]:
    """Stored papers whose title or abstract matches, best score first."""
    query = fts_query(text)
    if not query:
        return []
    conn = connect()
    try:
        rows = conn.execute("""
            SELECT papers.* FROM papers_fts JOIN papers USING (doi)
            WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts) LIMIT ?
        """, (query, limit)).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def stored_excerpts(dois: list[str]) -> dict[str, str]:
    """{doi: discussion excerpt} for each of `dois` the library holds an
    excerpt for, keyed by the DOI as given."""
    wanted = {doi: _normalise_doi(doi) for doi in dois}
    keys = sorted({key for key in wanted.values() if key})
    if not keys:
        return {}
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT doi, discussion_excerpt FROM papers WHERE discussion_excerpt IS NOT NULL "
            f"AND doi IN ({', '.join('?' * len(keys))})", keys).fetchall()
    finally:
        conn.close()
    found = {row["doi"]: row["discussion_excerpt"] for row in rows}
    return {doi: found[key] for doi, key in wanted.items() if key in found}


def describe_run(run: dict[str, Any]) -> str:
    """'You researched a similar question on 13 Sep ("...") — QA 4.2 → <folder>'"""
    try:
        created = datetime.datetime.fromisoformat(run["created"])
        when = f"{created.day} {created:%b %Y}"
    except (TypeError, ValueError):
        when = "an earlier date"
    quality = run.get("overall_quality")
    qa = f"QA {quality:.1f}" if isinstance(quality, (int, float)) else "QA n/a"
    return (f'You researched a similar question on {when} ("{run["question"]}") '
            f'— {qa} → {run["run_dir"]}')
