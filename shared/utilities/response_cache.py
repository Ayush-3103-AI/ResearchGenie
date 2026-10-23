"""Persistent cache for scholarly responses and provider state (#6).

One SQLite file next to the CLI's config, ~/.researchgenie/cache.sqlite
(RESEARCHGENIE_CACHE overrides it; tests point it at a temporary file):

    responses      (key PK, status, content_type, body, expires)
    provider_state (host PK, exhausted_until, balance)

The scholarly gateway reads and writes `responses` on every GET: a repeat or
similar run answers its searches and DOI lookups from disk, at no quota. The
key is the method, URL and sorted parameters with the identity parameters
(keys, email, NCBI's tool name) left out, so it never holds a secret and the
same request made with or without a key hits the same entry. TTLs: searches
7 days, DOI and entity lookups 30 days, extracted full text never expires.
Only 200 responses are stored.

`provider_state` keeps what the next run must know before it starts: the
time a credit-metered host (OpenAlex) is exhausted until, and its last
rate-limit headers (the balance).

RESEARCHGENIE_NO_CACHE=1 (the CLI's `--no-cache`) skips reads; fresh answers
still refresh the cache. A broken or locked cache file never fails a run:
every operation degrades to a miss.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from researchgenie.config import CONFIG_DIR

SEARCH_TTL = 7 * 24 * 3600
LOOKUP_TTL = 30 * 24 * 3600
# The parameters that identify the user; never part of a cache key.
IDENTITY_PARAMS = frozenset({"api_key", "mailto", "email", "tool"})
# A single-record path: Crossref /works/<doi>, OpenAlex /works/doi:<doi> or /works/W123
# (a DOI has slashes of its own, so match the start of the id, not the path's end).
_ENTITY_PATH = re.compile(r"/works/(?:10\.|doi:|W\d)", re.IGNORECASE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY, status INTEGER, content_type TEXT, body BLOB, expires REAL);
CREATE TABLE IF NOT EXISTS provider_state (
    host TEXT PRIMARY KEY, exhausted_until REAL, balance TEXT);
"""


def cache_path() -> Path:
    return Path(os.getenv("RESEARCHGENIE_CACHE", str(CONFIG_DIR / "cache.sqlite")))


def reads_enabled() -> bool:
    return os.getenv("RESEARCHGENIE_NO_CACHE", "") in ("", "0")


def _connect() -> sqlite3.Connection:
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.executescript(_SCHEMA)
    return conn


def _run(sql: str, args: tuple = ()) -> list[tuple]:
    """One statement in its own connection; [] on any SQLite or file error."""
    # ponytail: synchronous SQLite on the event loop; a local file answers in
    # well under a millisecond. Move to a thread if profiling ever says otherwise.
    try:
        conn = _connect()
    except (sqlite3.Error, OSError):
        return []
    try:
        with conn:
            return conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def key(url: str, params: dict | None = None) -> str:
    """'GET <url>?<sorted params>' without the identity parameters."""
    kept = sorted((k, str(v)) for k, v in (params or {}).items() if k not in IDENTITY_PARAMS)
    return f"GET {url}?{urlencode(kept)}"


def ttl_for(url: str) -> float:
    """30 days for a DOI/entity lookup or a PubMed fetch by id, else 7 (a search)."""
    path = urlsplit(url).path
    return LOOKUP_TTL if path.endswith("efetch.fcgi") or _ENTITY_PATH.search(path) else SEARCH_TTL


def get(cache_key: str, now: float) -> tuple[int, str, bytes] | None:
    """(status, content type, body) of a fresh entry, else None."""
    if not reads_enabled():
        return None
    rows = _run("SELECT status, content_type, body FROM responses "
                "WHERE key = ? AND (expires IS NULL OR expires > ?)", (cache_key, now))
    return (rows[0][0], rows[0][1], bytes(rows[0][2])) if rows else None


def put(cache_key: str, status: int, content_type: str, body: bytes,
        ttl: float | None, now: float) -> None:
    """ttl None: never expires."""
    _run("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
         (cache_key, status, content_type, body, None if ttl is None else now + ttl))


def get_text(cache_key: str) -> str | None:
    """Extracted full text stored with put_text (never expires)."""
    hit = get(cache_key, 0.0)
    return hit[2].decode("utf-8") if hit else None


def put_text(cache_key: str, text: str) -> None:
    put(cache_key, 200, "text/plain; charset=utf-8", text.encode("utf-8"), None, 0.0)


def clear() -> int:
    """Removes every cached response (provider state stays). Returns how many."""
    rows = _run("SELECT COUNT(*) FROM responses")
    _run("DELETE FROM responses")
    return rows[0][0] if rows else 0


def save_exhausted(host: str, until: float) -> None:
    _run("INSERT INTO provider_state (host, exhausted_until) VALUES (?, ?) "
         "ON CONFLICT(host) DO UPDATE SET exhausted_until = excluded.exhausted_until",
         (host, until))


def exhausted_until() -> dict[str, float]:
    return {host: until for host, until in
            _run("SELECT host, exhausted_until FROM provider_state "
                 "WHERE exhausted_until IS NOT NULL")}


def save_balance(host: str, balance: dict) -> None:
    _run("INSERT INTO provider_state (host, balance) VALUES (?, ?) "
         "ON CONFLICT(host) DO UPDATE SET balance = excluded.balance",
         (host, json.dumps(balance)))


def balance(host: str) -> dict | None:
    """The host's last rate-limit headers, with 'seen_at' (epoch seconds)."""
    rows = _run("SELECT balance FROM provider_state WHERE host = ? AND balance IS NOT NULL", (host,))
    return json.loads(rows[0][0]) if rows else None
