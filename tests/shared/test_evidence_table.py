"""The evidence table is built only from a run's own artifacts — tested
against a small fixture run folder shaped like a real one."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.contracts.verification_contract import VerificationResult  # noqa: E402
from shared.utilities import evidence_table, llm_provider  # noqa: E402

RUN = ROOT / "tests" / "fixtures" / "evidence_run"
WRITING = RUN / "02_writing" / "0a1b2c3d"


def _verification() -> VerificationResult:
    return VerificationResult.model_validate_json(
        (RUN / "03_verification" / "result.json").read_text(encoding="utf-8"))


def _rows(markdown: str) -> list[list[str]]:
    lines = [line for line in markdown.splitlines() if line.startswith("| ")]
    return [[cell.strip() for cell in line.strip("|").split(" | ")] for line in lines[1:]]


@pytest.fixture
def no_llm(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the evidence table must not call the model")
    for name in ("stream_json", "complete_json", "health_check"):
        monkeypatch.setattr(llm_provider, name, refuse)


def test_rows_join_used_points_to_sections_and_verification(no_llm):
    rows = _rows(evidence_table.build_evidence_table(WRITING, _verification()))

    assert [row[1] for row in rows] == [
        "Time-restricted eating improved working memory \\| recall scores.",
        "Ketosis was linked to faster reaction times.",
        "Self-reported focus rose during fasting.",
    ]  # the unused P002-POINT-02 is left out
    assert rows[0][0] == "Fasting and memory in adults"
    assert rows[0][2] == "Introduction; Background"
    assert rows[2][2] == "Findings"


def test_every_doi_and_status_matches_the_run(no_llm):
    verification = _verification()
    references = (RUN / "final" / "references.md").read_text(encoding="utf-8").lower()
    status_by_doi = {
        check.doi.lower(): status
        for status, checks in (("verified", verification.verified_references),
                               ("invalid", verification.invalid_references),
                               ("unverifiable", verification.unverifiable_references))
        for check in checks if check.doi
    }
    rows = _rows(evidence_table.build_evidence_table(WRITING, verification))

    assert [row[4] for row in rows] == ["verified", "invalid", "unverifiable"]
    for paper, _, _, doi, status in rows:
        if doi == "—":
            assert any(paper in check.reference_entry for check in verification.unverifiable_references)
            assert status == "unverifiable"
        else:
            assert doi.lower() in references
            assert status == status_by_doi[doi.lower()]


def test_missing_writing_artifacts_still_give_a_table(tmp_path, no_llm):
    markdown = evidence_table.build_evidence_table(tmp_path / "absent", _verification())
    assert markdown.startswith("# Evidence table")
    assert _rows(markdown) == []
