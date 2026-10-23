import asyncio
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "verification"))

from shared.contracts.writing_contract import DraftMetadata, WritingResult
from shared.contracts.verification_contract import VerificationRequest

# A bare `import service` would collide with the other pipeline services'
# own service.py modules (all reached only via sys.path insertion and all
# sharing the generic name "service") — whichever test file imports first
# wins the sys.modules cache slot. Load by explicit file path instead — see
# the same fix in tests/services/test_writing_pipeline_integration.py.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "research_verification_service", ROOT / "services" / "verification" / "service.py",
)
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)
build_validation_report = service.build_validation_report
extract_doi = service.extract_doi
run_verification = service.run_verification


def test_extract_doi_handles_ieee_and_author_year_styles():
    ieee = '[1] A. Author, "Title," *Journal*, 2020, doi: 10.1038/s41467-020-14676-4.'
    apa = "Author, A. (2020). Title. *Journal*. https://doi.org/10.3390/nu13093166"
    no_doi = "Author, A. (2020). Title with no identifier at all."

    assert extract_doi(ieee) == "10.1038/s41467-020-14676-4"
    assert extract_doi(apa) == "10.3390/nu13093166"
    assert extract_doi(no_doi) is None


def _writing_result(references: list[str], papers_cited: int, errors: list[str] | None = None) -> WritingResult:
    return WritingResult(
        research_outline="# Q\n## Introduction",
        research_draft_markdown="Some draft text.",
        reference_candidates=references,
        run_directory="/tmp/run",
        draft_metadata=DraftMetadata(
            model_name="qwen3.5:9b", run_id="abc", generated_at="2026-08-27T00:00:00Z",
            papers_cited=papers_cited, errors=errors or [],
        ),
    )


_RealAsyncClient = httpx.AsyncClient  # captured before any test monkeypatches httpx.AsyncClient


def _mock_client(responder) -> httpx.AsyncClient:
    transport = httpx.MockTransport(responder)
    return _RealAsyncClient(transport=transport)


def test_run_verification_classifies_verified_invalid_and_unverifiable(monkeypatch):
    references = [
        "[1] A. Author, \"Real Paper,\" *Journal*, 2020, doi: 10.1111/real.",
        "[2] B. Author, \"Fake Paper,\" *Journal*, 2021, doi: 10.2222/fake.",
        "[3] C. Author, \"No DOI Paper,\" *Journal*, 2019.",
    ]
    writing = _writing_result(references, papers_cited=3)
    request = VerificationRequest(writing=writing)

    def responder(req: httpx.Request) -> httpx.Response:
        # One DOI-filter request per provider (#7); each lists the works it has.
        if req.url.host == "api.crossref.org":
            return httpx.Response(200, json={"message": {"items": [{"DOI": "10.1111/REAL"}]}})
        return httpx.Response(200, json={"results": []})

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            self._client = _mock_client(responder)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *exc):
            await self._client.aclose()

    monkeypatch.setattr(service.httpx, "AsyncClient", _FakeAsyncClient)

    async def run():
        result = None
        async for event in run_verification(request):
            if event["type"] == "result":
                result = event["result"]
        return result

    result = asyncio.run(run())

    assert len(result.verified_references) == 1
    assert result.verified_references[0].doi == "10.1111/real"
    assert len(result.invalid_references) == 1
    assert result.invalid_references[0].doi == "10.2222/fake"
    assert len(result.unverifiable_references) == 1
    assert result.unverifiable_references[0].doi is None
    assert not result.missing_references
    assert "Citation Verification Report" in result.validation_report
    assert "found issues" in result.validated_draft_markdown


def test_run_verification_flags_missing_references_on_count_mismatch(monkeypatch):
    writing = _writing_result(["[1] A. Author, \"Paper,\" *Journal*, 2020, doi: 10.1111/real."], papers_cited=3)
    request = VerificationRequest(writing=writing)

    class _FakeAsyncClient:
        def __init__(self, *a, **k):
            self._client = _mock_client(lambda req: httpx.Response(200))

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *exc):
            await self._client.aclose()

    monkeypatch.setattr(service.httpx, "AsyncClient", _FakeAsyncClient)

    async def run():
        result = None
        async for event in run_verification(request):
            if event["type"] == "result":
                result = event["result"]
        return result

    result = asyncio.run(run())
    assert result.missing_references
    assert "3 paper" in result.missing_references[0]


def test_build_validation_report_is_empty_sections_free_when_all_verified():
    from shared.contracts.verification_contract import ReferenceCheck

    report = build_validation_report(
        checked=1,
        verified=[ReferenceCheck(reference_entry="ref", doi="10.1/x", verified=True, verification_source="Crossref")],
        invalid=[], unverifiable=[], missing_references=[], citation_issues=[],
    )
    assert "Verified: 1" in report
    assert "Invalid references" not in report
