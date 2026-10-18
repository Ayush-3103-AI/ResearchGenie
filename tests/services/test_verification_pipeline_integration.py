"""End-to-end integration test for the Citation Verification Service.

Requires internet access to Crossref/OpenAlex (both free, keyless). Skips
cleanly if neither is reachable. Does not require Ollama — verification is
pure HTTP lookups, no LLM calls.
"""

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "verification"))
sys.path.insert(0, str(ROOT))

from shared.contracts.writing_contract import DraftMetadata, WritingResult  # noqa: E402
from shared.contracts.verification_contract import VerificationRequest  # noqa: E402

# A bare `import service` would collide with the other pipeline services'
# own service.py modules — see the same fix in test_writing_pipeline_integration.py.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "research_verification_service", ROOT / "services" / "verification" / "service.py",
)
_service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_service)
run_verification = _service.run_verification


def _network_available() -> bool:
    try:
        resp = httpx.get("https://api.crossref.org/works/10.1038/s41467-020-14676-4", timeout=5)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Live network test: skipped on CI, where shared runner IPs exhaust "
               "OpenAlex's keyless daily credits. Run it locally.",
    ),
    pytest.mark.skipif(not _network_available(), reason="Crossref not reachable"),
]


def test_verification_service_checks_a_real_and_a_fabricated_doi():
    references = [
        '[1] Z. Liu, X. Dai, and H. Zhang, "Gut microbiota mediates intermittent-fasting '
        'alleviation of diabetes-induced cognitive impairment," *Nature Communications*, '
        "2020, doi: 10.1038/s41467-020-14676-4.",
        '[2] A. Nobody, "A Paper That Does Not Exist," *Journal of Nothing*, 2099, '
        "doi: 10.9999/does-not-exist-12345.",
    ]
    writing = WritingResult(
        research_outline="# Q\n## Introduction",
        research_draft_markdown="Draft citing [1] and [2].",
        reference_candidates=references,
        run_directory=str(ROOT / "outputs" / ".verification_test"),
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="test", generated_at="2026-08-27T00:00:00Z",
            papers_cited=2,
        ),
    )
    request = VerificationRequest(writing=writing)

    async def run():
        result = None
        async for event in run_verification(request):
            if event["type"] == "result":
                result = event["result"]
        return result

    result = asyncio.run(run())

    assert len(result.verified_references) == 1
    assert result.verified_references[0].doi == "10.1038/s41467-020-14676-4"
    assert len(result.invalid_references) == 1
    assert "found issues" in result.validated_draft_markdown
