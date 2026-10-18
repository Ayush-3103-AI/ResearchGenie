"""One cheap live request per scholarly key or email, for the setup wizard (#5).

Each check returns (ok, message): True = works, False = rejected (the
message says how to fix it), None = could not check right now (saved anyway).
Messages never contain the key or the email. These calls go straight to the
provider rather than through the gateway, because the value under test is
not in the environment yet.
"""

from __future__ import annotations

import re

import httpx

OPENALEX_KEY_URL = "https://openalex.org/settings/api"
NCBI_KEY_URL = "https://account.ncbi.nlm.nih.gov/settings/"
TIMEOUT = 15.0
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _probe(url: str, params: dict, transport: httpx.BaseTransport | None) -> int | None:
    """The HTTP status of one GET, or None when the provider could not be reached."""
    try:
        with httpx.Client(timeout=TIMEOUT, transport=transport) as http:
            return http.get(url, params=params).status_code
    except httpx.HTTPError:
        return None


def _verdict(status: int | None, name: str, fix: str) -> tuple[bool | None, str]:
    if status == 200:
        return True, f"{name} works."
    if status in (400, 401, 403):
        return False, f"{name} was rejected. {fix}"
    reached = "provider unreachable" if status is None else f"provider answered HTTP {status}"
    return None, f"Could not check the {name} right now ({reached}); saved anyway."


def check_openalex_key(key: str, *, transport: httpx.BaseTransport | None = None):
    status = _probe("https://api.openalex.org/works", {"per-page": 1, "api_key": key}, transport)
    return _verdict(status, "OpenAlex key",
                    f"Copy it again from {OPENALEX_KEY_URL} (no spaces), or press Enter to skip.")


def check_ncbi_key(key: str, *, transport: httpx.BaseTransport | None = None):
    status = _probe("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    {"db": "pubmed", "term": "test", "retmax": 0, "api_key": key}, transport)
    return _verdict(status, "NCBI key",
                    f"Copy it again from {NCBI_KEY_URL} (API Key Management), or press Enter to skip.")


def check_contact_email(email: str, *, transport: httpx.BaseTransport | None = None):
    if not _EMAIL.match(email):
        return False, "That doesn't look like an email address (name@example.edu)."
    status = _probe("https://api.crossref.org/works", {"rows": 0, "mailto": email}, transport)
    return _verdict(status, "Contact email",
                    "Use an address you read; Crossref and NCBI only use it to contact you.")
