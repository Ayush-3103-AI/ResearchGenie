"""Live checks and the setup step for the scholarly keys and email (#5)."""

import io
import sys
from pathlib import Path

import httpx
from rich.console import Console

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from researchgenie import scholarly_setup  # noqa: E402
from researchgenie import setup_wizard  # noqa: E402
from researchgenie.config import Config  # noqa: E402

KEY = "oa-secret-0123456789"


def _transport(status: int, seen: list | None = None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json={})
    return httpx.MockTransport(handler)


def test_a_valid_openalex_key_passes_with_one_request():
    seen = []
    ok, message = scholarly_setup.check_openalex_key(KEY, transport=_transport(200, seen))
    assert ok and len(seen) == 1 and seen[0].url.params["api_key"] == KEY
    assert KEY not in message


def test_an_invalid_key_is_reported_with_the_fix_and_without_the_key():
    for check in (scholarly_setup.check_openalex_key, scholarly_setup.check_ncbi_key):
        ok, message = check(KEY, transport=_transport(403))
        assert not ok and "https://" in message and KEY not in message


def test_an_unreachable_provider_is_not_called_an_invalid_key():
    def refuse(request):
        raise httpx.ConnectError("no network")
    ok, message = scholarly_setup.check_ncbi_key(KEY, transport=httpx.MockTransport(refuse))
    assert ok is None and "could not check" in message.lower()


def test_a_malformed_email_fails_without_a_request():
    seen = []
    ok, _ = scholarly_setup.check_contact_email("not-an-email", transport=_transport(200, seen))
    assert ok is False and not seen
    ok, _ = scholarly_setup.check_contact_email("me@uni.edu", transport=_transport(200, seen))
    assert ok and seen[0].url.params["mailto"] == "me@uni.edu"


def test_the_setup_step_asks_in_priority_order_and_every_value_can_be_skipped(monkeypatch):
    asked = []
    answers = iter([KEY, "", "ncbi-key-0123456789"])

    def ask(prompt, **kwargs):
        asked.append(prompt)
        return next(answers)

    monkeypatch.setattr(setup_wizard.Prompt, "ask", ask)
    monkeypatch.setattr(scholarly_setup, "check_openalex_key", lambda key, **k: (True, "OpenAlex key works."))
    monkeypatch.setattr(scholarly_setup, "check_ncbi_key", lambda key, **k: (True, "NCBI key works."))
    config = Config()
    setup_wizard.setup_scholarly_access(Console(file=io.StringIO()), config)

    assert ["OpenAlex" in asked[0], "email" in asked[1].lower(), "NCBI" in asked[2]] == [True] * 3
    assert config.openalex_api_key == KEY and config.contact_email is None
    assert config.ncbi_api_key == "ncbi-key-0123456789"


def test_the_setup_step_never_asks_for_a_semantic_scholar_key(monkeypatch):
    asked = []
    monkeypatch.setattr(setup_wizard.Prompt, "ask", lambda prompt, **k: asked.append(prompt) or "")
    setup_wizard.setup_scholarly_access(Console(file=io.StringIO()), Config())
    assert len(asked) == 3 and not any("semantic" in a.lower() for a in asked)
