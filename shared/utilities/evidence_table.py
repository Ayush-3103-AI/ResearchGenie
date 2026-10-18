"""Deterministic evidence table for a finished run (no LLM call).

Joins three artifacts the pipeline already writes:
  - the Writing run's literature_cards/<paper_id>.json (evidence points),
  - its tag_tree.json + section_drafts/<tag_id>.json (which section used
    which point — used_point_ids is checked by the section audit),
  - Citation Verification's result, matched to each paper by DOI.
One row per evidence point a section actually used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from shared.contracts.verification_contract import VerificationResult

HEADER = (
    "# Evidence table\n\n"
    "Every evidence point the draft uses, the paper it comes from, the "
    "section(s) where it is used, and that paper's citation-verification "
    "status. Check each claim against its source before you submit.\n\n"
)


_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def _bare_doi(doi: str | None) -> str:
    """Card metadata may carry 'https://doi.org/10.x/y' where the reference
    entry carries '10.x/y' — compare bare DOIs, case-folded."""
    return _DOI_PREFIX.sub("", (doi or "").strip()).rstrip(". ")


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence_table(writing_run_directory: Path, verification: VerificationResult) -> str:
    status_by_doi = {
        _bare_doi(check.doi).lower(): status
        for status, checks in (("verified", verification.verified_references),
                               ("invalid", verification.invalid_references),
                               ("unverifiable", verification.unverifiable_references))
        for check in checks if check.doi
    }

    points = {}  # point_id -> (card, point)
    for path in sorted((writing_run_directory / "literature_cards").glob("*.json")):
        card = _read(path)
        for point in card.get("points", {}).values():
            points[point["point_id"]] = (card, point)

    tree_path = writing_run_directory / "tag_tree.json"
    tree = _read(tree_path) if tree_path.is_file() else []
    sections = ([("INTRODUCTION", "Introduction")]
                + [(node["tag_id"], node["title"]) for node in tree if node.get("parent_id")]
                + [("CONCLUSION", "Conclusion")])

    used_in: dict[str, list[str]] = {}  # point_id -> section titles, first use first
    for tag_id, title in sections:
        draft_path = writing_run_directory / "section_drafts" / f"{tag_id}.json"
        if draft_path.is_file():
            for point_id in _read(draft_path).get("used_point_ids", []):
                if point_id in points and title not in used_in.setdefault(point_id, []):
                    used_in[point_id].append(title)

    if not used_in:
        return HEADER + "_No evidence points were recorded for this run._\n"

    lines = ["| Paper | Evidence point | Section used | DOI | Verification |",
             "|---|---|---|---|---|"]
    for point_id, titles in used_in.items():
        card, point = points[point_id]
        doi = _bare_doi(card["metadata"].get("doi"))
        # No DOI means Verification could not check it (its own rule);
        # a DOI it never saw means the paper is missing from the references.
        status = status_by_doi.get(doi.lower(), "not in references") if doi else "unverifiable"
        lines.append(" | ".join([
            "", _cell(card["metadata"]["title"]), _cell(point["content"]),
            _cell("; ".join(titles)), _cell(doi or "—"), status, "",
        ]).strip())
    return HEADER + "\n".join(lines) + "\n"
