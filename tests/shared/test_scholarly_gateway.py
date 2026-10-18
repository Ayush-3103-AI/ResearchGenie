"""Tests for the scholarly HTTP gateway (shared/utilities/scholarly_gateway.py).

The gateway replaces Discovery's `_http_retry.get_with_retry()` (DECISIONS.md
D-028); the first group of tests is that helper's suite, carried over so the
old guarantees still hold. Every test runs on a fake clock whose sleeps
return at once and are recorded, and on an httpx.MockTransport — no real
network, no real waiting.
"""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.utilities import scholarly_gateway  # noqa: E402
from shared.utilities.scholarly_gateway import Gateway, ProviderExhausted, Run  # noqa: E402

OPENALEX = "https://api.openalex.org/works"
S2 = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV = "https://export.arxiv.org/api/query"
NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EPOCH = 1_800_000_000.0


class FakeTime:
    """A clock that only moves when a test moves it. Sleeps are recorded and
    return at once, so a sleep's length is exactly the wait the gateway chose."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def wall(self) -> float:
        return EPOCH + self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        await asyncio.sleep(0)


def _gateway(fake: FakeTime) -> Gateway:
    return Gateway(clock=fake.clock, wall_clock=fake.wall, sleep=fake.sleep,
                   jitter=lambda: 0.5)  # jitter factor 1.0: backoff is exactly base * 2**n


def _replies(*responses: httpx.Response):
    """A MockTransport that answers with `responses` in order (the last one
    repeats) and records each request it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return httpx.MockTransport(handler), seen


def _get(gateway: Gateway, transport, url: str, run: Run | None = None, **kwargs):
    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await gateway.get(http, url, **kwargs)

    async def main():
        return await (run or Run()).task(go())

    return asyncio.run(main())


# --------------------------------------------------- carried over from _http_retry

def test_immediate_success_makes_exactly_one_request():
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(200))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 1
    assert fake.sleeps == []


def test_a_single_transient_5xx_recovers_on_retry():
    transport, seen = _replies(httpx.Response(503), httpx.Response(200))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 2


def test_429_is_retried_the_same_as_5xx():
    transport, seen = _replies(httpx.Response(429), httpx.Response(429), httpx.Response(200))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 200
    assert len(seen) == 3


def test_a_non_retryable_status_returns_immediately_without_delay():
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(404))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert resp.status_code == 404
    assert len(seen) == 1  # a real client error, not transient
    assert fake.sleeps == []


def test_exhausting_all_attempts_returns_the_last_response_not_an_exception():
    transport, seen = _replies(httpx.Response(503))
    resp = _get(_gateway(FakeTime()), transport, OPENALEX)
    assert resp.status_code == 503
    assert len(seen) == scholarly_gateway.POLICIES["api.openalex.org"].attempts


def test_backoff_without_a_server_hint_doubles_each_time():
    fake = FakeTime()
    transport, _ = _replies(httpx.Response(503), httpx.Response(503), httpx.Response(200))
    _get(_gateway(fake), transport, OPENALEX)
    assert fake.sleeps == [1.0, 2.0]


# ------------------------------------------------------------------- pacing

def test_two_runs_at_once_share_one_token_bucket_per_host():
    """arXiv allows 1 request per 3 s. Two runs firing three requests each at
    the same instant must be spaced 3 s apart as one queue, not two."""
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, seen = _replies(httpx.Response(200))

    async def three_requests():
        async with httpx.AsyncClient(transport=transport) as http:
            return await asyncio.gather(*[gateway.get(http, ARXIV) for _ in range(3)])

    async def main():
        run_a, run_b = Run(), Run()
        await asyncio.gather(run_a.task(three_requests()), run_b.task(three_requests()))

    asyncio.run(main())
    assert len(seen) == 6
    # The first request takes the idle bucket's token; each later one waits
    # one more 3-second slot, whichever run it came from.
    assert sorted(fake.sleeps) == pytest.approx([3, 6, 9, 12, 15])


def test_bucket_refills_with_time():
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, _ = _replies(httpx.Response(200))
    _get(gateway, transport, ARXIV)
    fake.now += 3.5  # more than a full slot later
    _get(gateway, transport, ARXIV)
    assert fake.sleeps == []


# ------------------------------------------------------------- server hints

def test_retry_after_is_honored():
    fake = FakeTime()
    run = Run()
    transport, seen = _replies(httpx.Response(429, headers={"Retry-After": "10"}),
                               httpx.Response(200))
    resp = _get(_gateway(fake), transport, OPENALEX, run=run)
    assert resp.status_code == 200
    assert fake.sleeps == pytest.approx([10.0])
    assert run.drain() == [{
        "type": "rate_limited", "host": "api.openalex.org", "wait_seconds": 10.0,
        "message": "OpenAlex answered HTTP 429; waiting 10 s before retrying.",
    }]


def test_rate_limit_reset_header_is_honored():
    fake = FakeTime()
    transport, _ = _replies(httpx.Response(429, headers={"X-RateLimit-Reset": "7"}),
                            httpx.Response(200))
    _get(_gateway(fake), transport, NCBI)
    assert fake.sleeps == pytest.approx([7.0])


def test_a_long_retry_after_waits_only_up_to_the_host_cap():
    """OpenAlex's cap is 60 s of waiting per request: asked for 500 s, the
    gateway waits 60 s once, then hands back the 429 instead of waiting on."""
    fake = FakeTime()
    transport, seen = _replies(httpx.Response(429, headers={"Retry-After": "500"}))
    resp = _get(_gateway(fake), transport, OPENALEX)
    assert fake.sleeps == pytest.approx([60.0])
    assert len(seen) == 2
    assert resp.status_code == 429


# ------------------------------------------------------------- credit budget

def _no_credits(reset: str | None = "3600") -> httpx.Response:
    headers = {"X-RateLimit-Reset": reset} if reset else {}
    return httpx.Response(429, headers=headers,
                          json={"error": "not_enough_credits",
                                "message": "Daily budget exhausted."})


def test_not_enough_credits_is_never_retried_and_marks_the_host_exhausted():
    fake = FakeTime()
    gateway = _gateway(fake)
    run = Run()
    transport, seen = _replies(_no_credits(), httpx.Response(200))

    with pytest.raises(ProviderExhausted) as raised:
        _get(gateway, transport, OPENALEX, run=run)
    assert len(seen) == 1
    assert fake.sleeps == []
    assert raised.value.host == "api.openalex.org"
    assert raised.value.until == pytest.approx(EPOCH + 3600)
    [event] = run.drain()
    assert event["type"] == "provider_exhausted"
    assert event["host"] == "api.openalex.org"
    assert event["remedy"] == scholarly_gateway.POLICIES["api.openalex.org"].remedy

    # Exhaustion is host state, not run state: a second run fails fast too.
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, OPENALEX)
    assert len(seen) == 1

    # Once the reset time passes, the host is used again.
    fake.now += 3601
    assert _get(gateway, transport, OPENALEX).status_code == 200


def test_not_enough_credits_without_a_reset_header_lasts_until_utc_midnight():
    fake = FakeTime()
    transport, _ = _replies(_no_credits(reset=None))
    with pytest.raises(ProviderExhausted) as raised:
        _get(_gateway(fake), transport, OPENALEX)
    assert raised.value.until == pytest.approx(EPOCH - EPOCH % 86400 + 86400)


def test_a_plain_openalex_429_is_still_retried():
    transport, seen = _replies(httpx.Response(429, json={"error": "rate_limited"}),
                               httpx.Response(200))
    assert _get(_gateway(FakeTime()), transport, OPENALEX).status_code == 200
    assert len(seen) == 2


# ----------------------------------------------------------- circuit breaker

def test_semantic_scholar_breaker_trips_on_the_first_429_burst_and_then_fails_fast():
    fake = FakeTime()
    gateway = _gateway(fake)
    run = Run()
    transport, seen = _replies(httpx.Response(429))

    with pytest.raises(ProviderExhausted) as raised:
        _get(gateway, transport, S2, run=run)
    assert len(seen) == 2  # the 429 and one retry, a few seconds apart
    assert sum(fake.sleeps) < 10
    assert raised.value.host == "api.semanticscholar.org"

    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, S2, run=run)
    assert len(seen) == 2  # failed fast: no request sent

    events = run.drain()
    assert [e["type"] for e in events] == ["rate_limited", "provider_exhausted"]
    assert events[1]["remedy"] == scholarly_gateway.POLICIES["api.semanticscholar.org"].remedy


def test_repeated_5xx_across_requests_trips_the_breaker_for_the_rest_of_the_run():
    gateway = _gateway(FakeTime())
    run = Run()
    transport, seen = _replies(httpx.Response(503))
    policy = scholarly_gateway.POLICIES["export.arxiv.org"]

    # One request alone uses up its attempts without tripping the breaker...
    assert _get(gateway, transport, ARXIV, run=run).status_code == 503
    assert len(seen) == policy.attempts
    # ...but the next one in the same run reaches the threshold.
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, ARXIV, run=run)
    assert len(seen) == policy.breaker_threshold
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, ARXIV, run=run)
    assert len(seen) == policy.breaker_threshold


def test_a_success_resets_the_breaker_count():
    gateway = _gateway(FakeTime())
    run = Run()
    transport, seen = _replies(httpx.Response(503), httpx.Response(200),
                               httpx.Response(503), httpx.Response(200))
    assert _get(gateway, transport, S2, run=run).status_code == 200
    assert _get(gateway, transport, S2, run=run).status_code == 200
    assert len(seen) == 4


def test_a_tripped_breaker_belongs_to_its_run_only():
    gateway = _gateway(FakeTime())
    transport, _ = _replies(httpx.Response(429), httpx.Response(429), httpx.Response(200))
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, S2, run=Run())
    assert _get(gateway, transport, S2, run=Run()).status_code == 200


def test_provider_exhausted_is_an_httpx_error_so_existing_handlers_degrade():
    assert issubclass(ProviderExhausted, httpx.HTTPError)


# ------------------------------------------------------------------ identity

def test_identity_is_injected_per_host_policy(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "me@example.org")
    monkeypatch.setenv("S2_API_KEY", "should-never-be-sent")
    gateway = _gateway(FakeTime())
    transport, seen = _replies(httpx.Response(200))
    for url in (OPENALEX, NCBI, S2):
        _get(gateway, transport, url, params={"q": "x"})
    openalex, ncbi, s2 = seen

    assert openalex.url.params["mailto"] == "me@example.org"
    assert ncbi.url.params["tool"] == scholarly_gateway.TOOL_NAME
    assert ncbi.url.params["email"] == "me@example.org"
    assert "mailto" not in s2.url.params and "email" not in s2.url.params
    assert "x-api-key" not in s2.headers  # Semantic Scholar stays keyless
    assert all("mailto:me@example.org" in r.headers["User-Agent"] for r in seen)
    assert all(r.url.params["q"] == "x" for r in seen)


# ------------------------------------------------------------------- stream

def test_stream_retries_a_transient_error_before_handing_over_the_body():
    gateway = _gateway(FakeTime())
    transport, seen = _replies(httpx.Response(503), httpx.Response(200, content=b"%PDF-1.7"))

    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            async with gateway.stream(http, "https://journals.example.org/a.pdf") as resp:
                return resp.status_code, b"".join([c async for c in resp.aiter_bytes()])

    status, body = asyncio.run(go())
    assert (status, body) == (200, b"%PDF-1.7")
    assert len(seen) == 2


def test_requests_in_flight_per_host_never_exceed_its_policy():
    gateway = _gateway(FakeTime())
    in_flight = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        for _ in range(3):
            await asyncio.sleep(0)
        in_flight["now"] -= 1
        return httpx.Response(200)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            await asyncio.gather(*[gateway.get(http, "https://www.ebi.ac.uk/europepmc/x")
                                   for _ in range(12)])

    asyncio.run(go())
    assert in_flight["max"] == scholarly_gateway.POLICIES["www.ebi.ac.uk"].concurrency


def test_host_policy_sets_fan_out_concurrency():
    assert scholarly_gateway.concurrency("https://export.arxiv.org/api/query") == 1
    assert scholarly_gateway.concurrency("https://www.ebi.ac.uk/europepmc/x") == \
        scholarly_gateway.POLICIES["www.ebi.ac.uk"].concurrency
    assert scholarly_gateway.concurrency("https://unknown.example.org/p.pdf") == \
        scholarly_gateway.DEFAULT_POLICY.concurrency


# ------------------------------------------------------------- keys (#5)

OPENALEX_KEY, NCBI_KEY, EMAIL = "oa-secret-key-123", "ncbi-secret-key-456", "student@example.edu"


def _keys(monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", OPENALEX_KEY)
    monkeypatch.setenv("NCBI_API_KEY", NCBI_KEY)
    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)


def test_each_provider_gets_its_own_identity_and_semantic_scholar_never_a_key(monkeypatch):
    _keys(monkeypatch)
    fake = FakeTime()
    gateway = _gateway(fake)
    transport, seen = _replies(httpx.Response(200))
    for url in (OPENALEX, NCBI, "https://api.crossref.org/works/10.1/x", S2):
        _get(gateway, transport, url)
    openalex, ncbi, crossref, s2 = (r.url.params for r in seen)

    assert openalex["api_key"] == OPENALEX_KEY and openalex["mailto"] == EMAIL
    assert ncbi["api_key"] == NCBI_KEY and ncbi["tool"] and ncbi["email"] == EMAIL
    assert crossref["mailto"] == EMAIL and "api_key" not in crossref
    assert not s2 and not any(k.lower() in ("x-api-key", "api_key") for k in seen[3].headers)


def test_a_keyed_ncbi_is_paced_at_its_keyed_rate(monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    fake = FakeTime()
    unkeyed = _gateway(fake)
    transport, _ = _replies(httpx.Response(200))
    for _ in range(4):
        _get(unkeyed, transport, NCBI)
    assert fake.sleeps and fake.sleeps[-1] == pytest.approx(1 / 3)

    monkeypatch.setenv("NCBI_API_KEY", NCBI_KEY)
    fake = FakeTime()
    keyed = _gateway(fake)
    for _ in range(11):
        _get(keyed, transport, NCBI)
    assert fake.sleeps == [pytest.approx(1 / 10)]


def test_no_key_or_email_leaks_into_errors_tracebacks_or_logs(monkeypatch, caplog):
    """D-023's lesson: the keys travel as query parameters (neither OpenAlex
    nor NCBI accepts a header), so every message that echoes a URL must not."""
    import logging
    import traceback
    _keys(monkeypatch)
    caplog.set_level(logging.DEBUG)
    gateway = _gateway(FakeTime())
    transport, _ = _replies(httpx.Response(404))
    resp = _get(gateway, transport, OPENALEX, params={"search": "x"})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        resp.raise_for_status()

    shown = str(caught.value) + "".join(traceback.format_exception(caught.value)) + caplog.text
    assert "api.openalex.org" in shown  # the URL is still there, just without its secrets
    for secret in (OPENALEX_KEY, EMAIL):
        assert secret not in shown
    assert scholarly_gateway.redact(f"key={OPENALEX_KEY}&email={EMAIL}") == "key=***&email=***"


def test_provider_status_shows_access_and_health_per_provider(monkeypatch):
    for name in ("OPENALEX_API_KEY", "NCBI_API_KEY", "CONTACT_EMAIL"):
        monkeypatch.delenv(name, raising=False)
    fake = FakeTime()
    gateway = _gateway(fake)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["access"] == "anonymous"
    assert status["api.openalex.org"]["health"] == "healthy"

    monkeypatch.setenv("CONTACT_EMAIL", EMAIL)
    monkeypatch.setenv("OPENALEX_API_KEY", OPENALEX_KEY)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["access"] == "keyed"
    assert status["api.crossref.org"]["access"] == "identified"
    assert status["api.semanticscholar.org"]["access"] == "anonymous"  # never keyed (no S2 plan)

    transport, _ = _replies(httpx.Response(429, text="not_enough_credits"))
    with pytest.raises(ProviderExhausted):
        _get(gateway, transport, OPENALEX)
    status = {p["host"]: p for p in gateway.provider_status()}
    assert status["api.openalex.org"]["health"] == "degraded"
    assert OPENALEX_KEY not in str(status) and EMAIL not in str(status)
