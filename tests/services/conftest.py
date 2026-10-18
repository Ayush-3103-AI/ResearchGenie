"""Every service test gets its own scholarly gateway that never really sleeps.

The gateway's state is process-wide by design (one token bucket per host for
the whole process), so without this a test would wait out real pacing (arXiv
is 1 request per 3 s) and inherit host state left behind by earlier tests.
"""

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.utilities import scholarly_gateway  # noqa: E402


async def _no_wait(seconds: float) -> None:
    await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def fresh_scholarly_gateway(monkeypatch):
    gateway = scholarly_gateway.Gateway(sleep=_no_wait)
    monkeypatch.setattr(scholarly_gateway, "_gateway", gateway)
    return gateway
