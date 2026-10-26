"""APA 7 undated-work suffixes and en-dash page ranges (#26), per
docs/tickets/26-apa-7.md. Other author-year styles must render unchanged."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))

from writing.modules.citation_formatter import (  # noqa: E402
    build_citation_map, citation_format_errors,
)
from writing.schemas import PaperMetadata  # noqa: E402


def test_undated_works_get_suffixes_in_title_order():
    papers = {
        "P001": PaperMetadata(title="Undated work two", authors=["Zed Zhou"], year=None),
        "P002": PaperMetadata(title="Undated work", authors=["Zed Zhou"], year=None),
    }
    citations = build_citation_map(["P001", "P002"], papers, "apa-7")

    assert citations["P002"].in_text_citation == "(Zhou, n.d.-a)"
    assert citations["P001"].in_text_citation == "(Zhou, n.d.-b)"
    assert "(n.d.-a). Undated work." in citations["P002"].reference_entry
    assert "(n.d.-b). Undated work two." in citations["P001"].reference_entry
    assert citation_format_errors(citations["P002"], "apa-7") == []
    assert citation_format_errors(citations["P001"], "apa-7") == []


def test_single_undated_work_stays_plain():
    papers = {
        "P001": PaperMetadata(title="Single undated work", authors=["Zed Zhou"], year=None),
    }
    citations = build_citation_map(["P001"], papers, "apa-7")

    assert citations["P001"].in_text_citation == "(Zhou, n.d.)"
    assert "(n.d.). Single undated work." in citations["P001"].reference_entry
    assert citation_format_errors(citations["P001"], "apa-7") == []


def test_page_range_uses_en_dash_in_apa_but_not_ieee():
    papers = {
        "P001": PaperMetadata(title="Ranged pages", authors=["Zed Zhou"], year=2020,
                              journal="Journal", volume="1", pages="121-133"),
    }
    apa = build_citation_map(["P001"], papers, "apa-7")
    ieee = build_citation_map(["P001"], papers, "ieee")

    assert "121–133" in apa["P001"].reference_entry
    assert "121-133" in ieee["P001"].reference_entry
    assert "–" not in ieee["P001"].reference_entry
    assert citation_format_errors(apa["P001"], "apa-7") == []


def test_single_page_is_unchanged():
    papers = {
        "P001": PaperMetadata(title="Single page", authors=["Zed Zhou"], year=2020,
                              journal="Journal", volume="1", pages="e1234"),
    }
    citations = build_citation_map(["P001"], papers, "apa-7")

    assert "e1234" in citations["P001"].reference_entry
    assert "–" not in citations["P001"].reference_entry
