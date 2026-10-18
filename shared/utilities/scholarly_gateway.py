"""One gateway for every outbound scholarly HTTP request.

Replaces Discovery's `_http_retry.py` (DECISIONS.md D-028). That helper
retried each request on its own, so four sources and the full-text fan-out
hammered the same hosts in parallel with no shared pace, waited at most
~7 s in total, and retried OpenAlex's "daily budget gone" 429 as if it were
transient. The gateway owns four things, all driven by the POLICIES table:

- pacing: one token bucket per host, shared by the whole process (two runs at
  once draw from the same bucket). Requests wait for a token; they are never
  fired first and retried later.
- retry: jittered exponential backoff on 429/5xx that honors Retry-After and
  rate-limit reset headers, up to a per-host cap on total waiting.
- identity: the contact email (CONTACT_EMAIL) as `mailto`, NCBI's `tool` and
  `email` parameters, the free OpenAlex and NCBI keys (OPENALEX_API_KEY,
  NCBI_API_KEY, #5) and a User-Agent that names the tool. Neither provider
  accepts its key in a header, so keys travel as query parameters and every
  URL the gateway hands back is redacted (`redact`, D-023's lesson).
- circuit breaker: repeated 429/5xx responses within one run, or an OpenAlex
  `not_enough_credits` 429, mark the host exhausted. Later calls fail fast
  with ProviderExhausted and the run continues on the other sources.

Callers keep passing their own httpx.AsyncClient. Per-run state (breaker
counts, the event queue for `rate_limited` / `provider_exhausted`) lives in a
Run; start the run's work with `run.task(coro)` and relay its events with
`run.relay(task)`. A call made outside any Run still gets pacing, retry and
identity, but its breaker state lasts only for that one call.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime
import email.utils
import logging
import os
import random
import threading
import time
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Coroutine
from urllib.parse import urlsplit

import httpx

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TOOL_NAME = "lit-gap"  # NCBI's required `tool` parameter, kept from the old sources
PACING_NOTICE_SECONDS = 10.0  # a self-imposed pacing wait this long is worth telling the user


@dataclass(frozen=True)
class HostPolicy:
    name: str                 # shown to the user
    rate: float               # sustained requests per second
    burst: int                # tokens the bucket holds when idle
    concurrency: int          # requests in flight at once
    identity: str             # "mailto" | "ncbi" | "none" (every host also gets the User-Agent)
    remedy: str               # what the user can do once the host is exhausted
    key_env: str | None = None     # env var holding the provider's free API key, sent as `api_key`
    keyed_rate: float | None = None  # sustained rate once that key is set (burst follows it)
    attempts: int = 4         # tries per request, the first one included
    max_retry_wait: float = 60.0   # cap on total waiting per request, seconds
    backoff: float = 1.0      # first backoff step when the server gives no wait
    breaker_threshold: int = 6     # consecutive 429/5xx responses in a run that trip the breaker
    credits_marker: str | None = None  # 429 body text meaning "budget gone until reset"


# Starting values from each provider's published limits (spec #1, Further
# Notes, checked 2026-10-18) — deliberately conservative.
POLICIES: dict[str, HostPolicy] = {
    "api.openalex.org": HostPolicy(
        name="OpenAlex", rate=10, burst=10, concurrency=4, identity="mailto",
        credits_marker="not_enough_credits", key_env="OPENALEX_API_KEY",
        # The key raises the daily budget (about 100 -> 1,000 searches), not the pace.
        remedy="Add a free OpenAlex API key (`researchgenie config`) for a larger daily "
               "budget, or wait for it to reset; Discovery continues on the other sources."),
    # Keyless only: the project will not get an S2 key, so no key header is
    # ever sent. The shared anonymous pool 429s often; one slow connection and
    # a breaker that trips on the first 429 burst keep a busy pool to seconds.
    "api.semanticscholar.org": HostPolicy(
        name="Semantic Scholar", rate=1 / 3, burst=1, concurrency=1, identity="none",
        attempts=3, max_retry_wait=15, backoff=2.0, breaker_threshold=2,
        remedy="Semantic Scholar's shared keyless pool is busy; try again later. "
               "Discovery continues on the other sources."),
    "eutils.ncbi.nlm.nih.gov": HostPolicy(
        name="PubMed (NCBI)", rate=3, burst=3, concurrency=3, identity="ncbi",
        key_env="NCBI_API_KEY", keyed_rate=10,
        remedy="Add a contact email and a free NCBI API key (`researchgenie config`), "
               "or try again later."),
    "export.arxiv.org": HostPolicy(
        name="arXiv", rate=1 / 3, burst=1, concurrency=1, identity="none",
        backoff=3.0,
        remedy="arXiv is throttling this machine; try again in a few minutes."),
    # Single-record pace without the polite pool; Citation Verification's DOI lookups (#4).
    "api.crossref.org": HostPolicy(
        name="Crossref", rate=5, burst=5, concurrency=1, identity="mailto",
        remedy="Add a contact email (`researchgenie config`) to use Crossref's polite "
               "pool, or try again later."),
    # No published limit — be conservative.
    "www.ebi.ac.uk": HostPolicy(
        name="Europe PMC", rate=5, burst=5, concurrency=4, identity="none",
        remedy="Europe PMC is overloaded; full text falls back to open-access PDFs."),
}

# Any other host: open-access PDF links point at arbitrary publisher sites.
DEFAULT_POLICY = HostPolicy(
    name="Open-access host", rate=2, burst=2, concurrency=2, identity="none",
    attempts=2, max_retry_wait=10,
    remedy="This host is refusing requests; other full-text routes are still tried.")


# Environment variables whose values must never be shown to the user.
SECRET_ENVS = ("OPENALEX_API_KEY", "NCBI_API_KEY", "CONTACT_EMAIL", "GEMINI_API_KEY")
DEGRADED_SECONDS = 15 * 60  # a tripped breaker shows the provider as degraded this long


def redact(text: str) -> str:
    """`text` with every configured key and the contact email masked."""
    for name in SECRET_ENVS:
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    return text


class _RedactLogs(logging.Filter):
    """httpx logs each request URL at INFO; mask the secrets in it."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, None
        return True


for _logger in ("httpx", "httpcore"):
    logging.getLogger(_logger).addFilter(_RedactLogs())


def _keyed(policy: "HostPolicy") -> bool:
    return bool(policy.key_env and os.getenv(policy.key_env))


def host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def policy_for(url_or_host: str) -> HostPolicy:
    host = host_of(url_or_host) if "/" in url_or_host else url_or_host.lower()
    return POLICIES.get(host, DEFAULT_POLICY)


def concurrency(url_or_host: str) -> int:
    """How many requests a caller should fan out to this host at once."""
    return policy_for(url_or_host).concurrency


class ProviderExhausted(httpx.HTTPError):
    """A host is skipped for the rest of the run (breaker tripped) or until its
    reset time (credit budget gone). Subclasses httpx.HTTPError so callers that
    already degrade on HTTP failures (the Trend Advisor, Verification) treat it
    the same way without new except clauses."""

    def __init__(self, host: str, reason: str, remedy: str, until: float | None = None):
        self.host = host
        self.reason = reason
        self.remedy = remedy
        self.until = until  # wall-clock epoch seconds, for credit exhaustion
        super().__init__(f"{policy_for(host).name} ({host}) is unavailable: {reason}. {remedy}")


_current_run: contextvars.ContextVar["Run | None"] = contextvars.ContextVar(
    "scholarly_gateway_run", default=None)


class Run:
    """One run's breaker state and its queue of gateway progress events."""

    def __init__(self) -> None:
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.failures: dict[str, int] = {}
        self.tripped: dict[str, ProviderExhausted] = {}
        self._reported: set[str] = set()

    def task(self, coro: Coroutine) -> asyncio.Task:
        """Start `coro` as a task whose gateway calls (and any tasks it spawns)
        count against this run."""
        context = contextvars.copy_context()
        context.run(_current_run.set, self)
        return asyncio.create_task(coro, context=context)

    async def relay(self, task: asyncio.Task) -> AsyncIterator[dict]:
        """Yield this run's gateway events as they happen while `task` runs.
        When the loop ends, `task.result()` is ready."""
        getter = None
        try:
            while not task.done():
                getter = asyncio.ensure_future(self.events.get())
                await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield getter.result()
                else:
                    getter.cancel()
            for event in self.drain():
                yield event
        finally:
            if getter is not None and not getter.done():
                getter.cancel()
            if not task.done():  # the consumer stopped listening mid-run
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def drain(self) -> list[dict]:
        """The events queued so far, without waiting."""
        drained = []
        while not self.events.empty():
            drained.append(self.events.get_nowait())
        return drained

    def emit(self, event: dict) -> None:
        self.events.put_nowait(event)

    def exhausted(self, error: ProviderExhausted) -> ProviderExhausted:
        """Report a host as exhausted (once per run) and return the error to raise."""
        if error.host not in self._reported:
            self._reported.add(error.host)
            self.emit({"type": "provider_exhausted", "host": error.host,
                       "remedy": error.remedy, "message": str(error)})
        return error


class _Bucket:
    """Token bucket that hands out future slots: a caller reserves a token and
    is told how long to sleep, so concurrent callers queue in order without
    holding a lock across the wait. Tokens go negative while slots are booked
    ahead. A threading lock keeps it correct across event loops and threads."""

    def __init__(self, rate: float, burst: int, clock: Callable[[], float]):
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._tokens = float(burst)
        self._updated = clock()  # may lie in the future while the host is paused
        self._lock = threading.Lock()

    def reserve(self) -> float:
        """Take a token; return the seconds to wait before using it."""
        with self._lock:
            now = self._clock()
            start = max(now, self._updated)
            self._tokens = min(self.burst, self._tokens + (start - self._updated) * self.rate)
            self._updated = start
            self._tokens -= 1
            return start - now + max(0.0, -self._tokens) / self.rate

    def pause(self, seconds: float) -> None:
        """The server asked us to slow down: no new slot before now+seconds,
        and a single request (not a burst) when the pause ends."""
        with self._lock:
            self._updated = max(self._updated, self._clock() + seconds)
            if self._tokens >= 0:  # negative means callers already booked slots
                self._tokens = 1.0


def _header_wait(resp: httpx.Response, now: float) -> float | None:
    """Seconds the server asked us to wait: Retry-After (delta or HTTP date),
    else a rate-limit reset header (delta seconds or an epoch timestamp)."""
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                return max(0.0, email.utils.parsedate_to_datetime(retry_after).timestamp() - now)
            except (TypeError, ValueError):
                pass
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            value = float(reset)
        except ValueError:
            return None
        return max(0.0, value - now) if value > 1e9 else max(0.0, value)
    return None


def _next_utc_midnight(now: float) -> float:
    today = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()
    midnight = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time(),
                                         datetime.timezone.utc)
    return midnight.timestamp()


def _identify(policy: HostPolicy, params: dict | None, headers: dict | None) -> tuple[dict, dict]:
    params = dict(params or {})
    headers = dict(headers or {})
    if _keyed(policy):
        params.setdefault("api_key", os.environ[policy.key_env])
    email_address = os.getenv("CONTACT_EMAIL", "")
    if policy.identity == "mailto" and email_address:
        params.setdefault("mailto", email_address)
    elif policy.identity == "ncbi":
        params.setdefault("tool", TOOL_NAME)
        if email_address:
            params.setdefault("email", email_address)
    contact = f"; mailto:{email_address}" if email_address else ""
    headers.setdefault("User-Agent",
                       f"Mozilla/5.0 (compatible; {TOOL_NAME}/1.0; research tool{contact})")
    return params, headers


class Gateway:
    """Process-wide host state: buckets, connection slots and credit exhaustion.
    `clock`, `wall_clock`, `sleep` and `jitter` are test seams."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Coroutine] = asyncio.sleep,
                 jitter: Callable[[], float] = random.random):
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._jitter = jitter
        self._buckets: dict[str, _Bucket] = {}
        self._buckets_lock = threading.Lock()
        # Semaphores belong to one event loop, so keep a set per loop.
        self._slots: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        # ponytail: in-process only; #6 persists this in the cache DB so the
        # next run knows before it starts.
        self._exhausted_until: dict[str, float] = {}
        self._tripped_at: dict[str, float] = {}  # wall clock of the last breaker trip, any run

    def _bucket(self, host: str, policy: HostPolicy) -> _Bucket:
        # Keyed and keyless traffic pace separately, so adding a key mid-process takes effect.
        keyed = _keyed(policy) and policy.keyed_rate is not None
        with self._buckets_lock:
            if (host, keyed) not in self._buckets:
                rate = policy.keyed_rate if keyed else policy.rate
                burst = max(policy.burst, int(rate)) if keyed else policy.burst
                self._buckets[(host, keyed)] = _Bucket(rate, burst, self._clock)
            return self._buckets[(host, keyed)]

    def provider_status(self) -> list[dict]:
        """Each named provider's access (keyed / identified / anonymous) and
        health (healthy / degraded), for the CLI and the web header. Never
        carries a key or the email itself."""
        now = self._wall_clock()
        identified = bool(os.getenv("CONTACT_EMAIL"))
        status = []
        for host, policy in POLICIES.items():
            if _keyed(policy):
                access = "keyed"
            elif identified and policy.identity in ("mailto", "ncbi"):
                access = "identified"
            else:
                access = "anonymous"
            until = self._exhausted_until.get(host)
            if until is not None and until > now:
                health, reason = "degraded", "daily budget used up"
            elif now - self._tripped_at.get(host, -DEGRADED_SECONDS) < DEGRADED_SECONDS:
                health, reason = "degraded", "recent rate-limit or server errors"
            else:
                health, reason = "healthy", ""
            if host == "api.semanticscholar.org":
                reason = reason or "keyless shared pool, best-effort"
            status.append({"name": policy.name, "host": host, "access": access,
                           "health": health, "reason": reason})
        return status

    def _slot(self, host: str, policy: HostPolicy) -> asyncio.Semaphore:
        per_loop = self._slots.setdefault(asyncio.get_running_loop(), {})
        if host not in per_loop:
            per_loop[host] = asyncio.Semaphore(policy.concurrency)
        return per_loop[host]

    async def get(self, http: httpx.AsyncClient, url: str, *, params: dict | None = None,
                  headers: dict | None = None, timeout: float | None = None) -> httpx.Response:
        """GET through the gateway. Returns the final response — a successful
        one, a non-retryable error (the caller's raise_for_status reports it),
        or the last 429/5xx once retries are used up. Raises ProviderExhausted
        when the host is skipped."""
        host, policy = host_of(url), policy_for(url)
        async with self._slot(host, policy):
            return await self._send(http, url, params, headers, timeout, stream=False)

    @asynccontextmanager
    async def stream(self, http: httpx.AsyncClient, url: str, *, params: dict | None = None,
                     headers: dict | None = None,
                     timeout: float | None = None) -> AsyncIterator[httpx.Response]:
        """Streaming GET through the gateway; the host's connection slot is
        held until the body has been read or abandoned."""
        host, policy = host_of(url), policy_for(url)
        async with self._slot(host, policy):
            resp = await self._send(http, url, params, headers, timeout, stream=True)
            try:
                yield resp
            finally:
                await resp.aclose()

    async def _send(self, http: httpx.AsyncClient, url: str, params: dict | None,
                    headers: dict | None, timeout: float | None, *, stream: bool) -> httpx.Response:
        host, policy = host_of(url), policy_for(url)
        run = _current_run.get() or Run()
        params, headers = _identify(policy, params, headers)
        extra = {} if timeout is None else {"timeout": timeout}
        waited = 0.0
        for attempt in range(policy.attempts):
            self._check_available(run, host, policy)
            pace = self._bucket(host, policy).reserve()
            # Only a first attempt's wait is pacing; a retry's was reported as rate_limited.
            if attempt == 0 and pace >= PACING_NOTICE_SECONDS:
                run.emit({"type": "rate_limited", "host": host, "wait_seconds": round(pace, 1),
                          "message": f"Pacing requests to {policy.name}: waiting "
                                     f"{pace:.0f} s for a free slot."})
            if pace > 0:
                await self._sleep(pace)

            request = http.build_request("GET", url, params=params, headers=headers, **extra)
            resp = await http.send(request, stream=stream)
            shown = redact(str(request.url))
            if shown != str(request.url):
                # raise_for_status and error reprs echo resp.request.url.
                resp.request = httpx.Request("GET", shown)
            if resp.status_code not in RETRY_STATUSES:
                run.failures[host] = 0
                return resp

            if stream:
                await resp.aread()  # small error body; needed for the credits check
            if (resp.status_code == 429 and policy.credits_marker
                    and policy.credits_marker in resp.text):
                await resp.aclose()
                now = self._wall_clock()
                wait = _header_wait(resp, now)
                until = now + wait if wait is not None else _next_utc_midnight(now)
                self._exhausted_until[host] = until
                raise run.exhausted(self._credits_error(host, policy, until))

            run.failures[host] = run.failures.get(host, 0) + 1
            if run.failures[host] >= policy.breaker_threshold:
                await resp.aclose()
                self._tripped_at[host] = self._wall_clock()
                run.tripped[host] = ProviderExhausted(
                    host, f"{run.failures[host]} rate-limit/server errors in a row "
                          f"(last HTTP {resp.status_code})", policy.remedy)
                raise run.exhausted(run.tripped[host])

            remaining = policy.max_retry_wait - waited
            if attempt == policy.attempts - 1 or remaining <= 0:
                return resp  # retries used up; the caller's raise_for_status reports it
            await resp.aclose()
            wait = _header_wait(resp, self._wall_clock())
            if wait is None:
                wait = policy.backoff * 2 ** attempt * (0.5 + self._jitter())
            wait = min(wait, remaining)
            waited += wait
            self._bucket(host, policy).pause(wait)
            run.emit({"type": "rate_limited", "host": host, "wait_seconds": round(wait, 1),
                      "message": f"{policy.name} answered HTTP {resp.status_code}; "
                                 f"waiting {wait:.0f} s before retrying."})
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def _check_available(self, run: Run, host: str, policy: HostPolicy) -> None:
        if host in run.tripped:
            raise run.exhausted(run.tripped[host])
        until = self._exhausted_until.get(host)
        if until is not None:
            if until > self._wall_clock():
                raise run.exhausted(self._credits_error(host, policy, until))
            self._exhausted_until.pop(host, None)  # the budget has reset

    @staticmethod
    def _credits_error(host: str, policy: HostPolicy, until: float) -> ProviderExhausted:
        reset = datetime.datetime.fromtimestamp(until, datetime.timezone.utc)
        return ProviderExhausted(
            host, f"its daily credit budget is used up until {reset:%H:%M} UTC",
            policy.remedy, until=until)


_gateway = Gateway()


async def get(http: httpx.AsyncClient, url: str, *, params: dict | None = None,
              headers: dict | None = None, timeout: float | None = None) -> httpx.Response:
    """GET `url` through the process-wide gateway (see Gateway.get)."""
    return await _gateway.get(http, url, params=params, headers=headers, timeout=timeout)


def provider_status() -> list[dict]:
    """Per-provider access and health from the process-wide gateway."""
    return _gateway.provider_status()


def stream(http: httpx.AsyncClient, url: str, *, params: dict | None = None,
           headers: dict | None = None, timeout: float | None = None):
    """`async with stream(http, url) as resp:` through the process-wide gateway."""
    return _gateway.stream(http, url, params=params, headers=headers, timeout=timeout)
