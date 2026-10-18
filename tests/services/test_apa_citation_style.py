"""APA 7 author-date citations (#26) on a fixture paper set: in-text and
narrative citations, the reference list, and how they render in Markdown
and LaTeX. Two rules the formatter does not follow yet (n.d.-a/n.d.-b and
en-dash page ranges) are the Dev Agent's ticket, docs/tickets/26-apa-7.md."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services" / "research-writing"))

from writing.modules.citation_formatter import (  # noqa: E402
    build_citation_map, build_reference_list, citation_format_errors, format_citations,
)
from writing.schemas import PaperMetadata  # noqa: E402

from shared.contracts.writing_contract import WritingRequest  # noqa: E402
from shared.utilities.latex_export import markdown_to_latex  # noqa: E402

PAPERS = {
    "P001": PaperMetadata(title="Sleep-dependent memory consolidation", year=2005,
                          authors=["Matthew P. Walker", "Robert Stickgold"], journal="Nature",
                          volume="437", issue="7063", pages="1272-1278",
                          doi="10.1038/nature04286"),
    "P002": PaperMetadata(title="Memory and sleep in adolescents", year=2020,
                          authors=["Ann Lee", "Bo Kim", "Cy Park"], journal="Sleep", volume="43",
                          doi="https://doi.org/10.1093/sleep/zsz001"),
    "P003": PaperMetadata(title="Napping and recall", year=2020,
                          authors=["Ann Lee", "Bo Kim", "Cy Park"]),
    "P004": PaperMetadata(title="A consortium study", year=2021,
                          authors=[f"Given{i} Family{i:02d}" for i in range(1, 24)]),
    "P005": PaperMetadata(title="Dreams", year=2019, authors=["Jan Adams"]),
}
APA = build_citation_map(list(PAPERS), PAPERS, "apa-7")


def test_the_apa_format_maps_to_apa_7():
    from shared.contracts.writing_contract import _FORMAT_TO_CITATION_STYLE
    assert _FORMAT_TO_CITATION_STYLE["APA"] == "apa-7"


def test_in_text_citations_are_author_date():
    assert APA["P005"].in_text_citation == "(Adams, 2019)"
    assert APA["P001"].in_text_citation == "(Walker & Stickgold, 2005)"  # & inside parentheses
    assert APA["P001"].narrative_citation == "Walker and Stickgold (2005)"  # "and" in running text
    assert APA["P004"].in_text_citation == "(Family01 et al., 2021)"  # three or more: et al.
    # Same authors, same year: a and b, in reference-list (title) order.
    assert APA["P002"].in_text_citation == "(Lee et al., 2020a)"
    assert APA["P003"].in_text_citation == "(Lee et al., 2020b)"
    for citation in APA.values():
        assert citation_format_errors(citation, "apa-7") == []


def test_reference_entries_follow_apa_7():
    assert APA["P001"].reference_entry == (
        "Walker, M. P., & Stickgold, R. (2005). Sleep-dependent memory consolidation. "
        "*Nature, 437*(7063), 1272-1278. https://doi.org/10.1038/nature04286")
    assert APA["P002"].reference_entry == (
        "Lee, A., Kim, B., & Park, C. (2020a). Memory and sleep in adolescents. *Sleep, 43*. "
        "https://doi.org/10.1093/sleep/zsz001")
    # 21 or more authors: the first 19, an ellipsis, then the last.
    entry = APA["P004"].reference_entry
    assert "Family19, G., ... Family23, G. (2021)" in entry and "Family20" not in entry


def test_the_reference_list_is_alphabetical_by_first_author():
    references = build_reference_list(list(APA), APA)
    assert [r.split(",")[0] for r in references] == ["Adams", "Family01", "Lee", "Lee", "Walker"]


def test_apa_renders_in_markdown_and_latex():
    prose = "Sleep helps memory [@P001], especially in teens [@P002] [@P003]."
    markdown = format_citations(prose, APA, "apa-7")
    assert "(Walker & Stickgold, 2005)" in markdown and "(Lee et al., 2020a)" in markdown

    draft = f"# Sleep and memory\n\n## Introduction\n\n{markdown}\n\n## References\n\n" + \
        "\n".join(f"- {r}" for r in build_reference_list(list(APA), APA))
    tex = markdown_to_latex(draft, title="Sleep and memory", target_format="APA")
    assert "\\documentclass" in tex and "(Walker \\& Stickgold, 2005)" in tex
    assert "\\textit{Nature, 437}(7063)" in tex
    assert "Lee, A., Kim, B., \\& Park, C. (2020a)" in tex


def test_apa_is_a_request_option_everywhere(capsys, monkeypatch):
    from researchgenie import cli
    assert "APA" in cli._TARGET_FORMATS
    assert WritingRequest.model_fields["target_format"].annotation.__args__.count("APA") == 1
    frontend = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert '["IEEE", "Springer", "ACM", "APA"]' in frontend

    from researchgenie import headless
    args = headless.build_parser().parse_args(["run", "-q", "Q?", "--format", "APA"])
    assert headless._request(args).target_format == "APA"
