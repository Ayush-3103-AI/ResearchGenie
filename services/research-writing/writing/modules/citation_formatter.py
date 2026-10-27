"""Deterministically render paper-level citations from bibliographic metadata."""

import re
from dataclasses import dataclass

from writing.schemas import CitationInfo, CitationStyle, PaperMetadata

# The prompts ask for `[@P001]`, but a model occasionally drops the "@"
# (`[P001]`) in prose while still correctly restricting itself to allowed
# papers (see DECISIONS.md D-021) — the mirror image of the cited_paper_ids
# "@P001" mistake schemas.py normalizes. "@" is optional here so a bare
# `[P001]` still resolves as the same citation instead of silently vanishing
# from extraction (which would otherwise make a fully valid citation look
# unsupported and, worse, could leave an un-replaced literal `[P001]` in the
# rendered output). "P" + 3+ digits is a paper-ID-specific pattern, so this
# does not risk matching unrelated bracketed text.
#
# A model occasionally invents a range/list shorthand instead of repeating
# the placeholder — `[P003-P006]` for "P003, P004, P005, P006" — even though
# editorial_standard.md now explicitly tells it not to (see DECISIONS.md
# D-026 "bookend audit investigation": this was found to be the ROOT CAUSE
# of the bookend Introduction/Conclusion sections being discarded on a real
# run — the shorthand form parsed as zero valid citations, so it looked like
# the model cited papers in `cited_paper_ids` that never appeared in the
# prose at all, failing the audit even though every paper in the range was
# genuinely allowed). The optional `-@?P\d{3,}` end group tolerates the
# shorthand as a defense-in-depth safety net, the same "prompt fix + tolerant
# parsing" pattern D-021 already established for the missing-"@" case.
_CITATION = re.compile(r"\[@?(?P<paper_id>P\d{3,})(?:-@?(?P<range_end>P\d{3,}))?\]")


def _expand_citation_match(match: re.Match[str]) -> list[str]:
    """Returns the one or more paper IDs one `_CITATION` match covers."""

    start = match.group("paper_id")
    end = match.group("range_end")
    if end is None:
        return [start]
    start_digits, end_digits = start[1:], end[1:]
    if len(start_digits) != len(end_digits):
        # Different digit widths (e.g. "P003-P0006") isn't a genuine
        # sequential range — treat the two ends literally rather than guess.
        return [start, end]
    lo, hi = int(start_digits), int(end_digits)
    # A garbled or absurdly wide "range" (hi < lo, or spanning 50+ IDs) is
    # more likely a formatting error than a real citation list — fall back
    # to just the two literal endpoints rather than expanding it.
    if hi < lo or hi - lo > 50:
        return [start, end]
    width = len(start_digits)
    return [f"P{n:0{width}d}" for n in range(lo, hi + 1)]
_AUTHOR_YEAR = re.compile(
    r"^\([^(),]+(?:\s+(?:and|&)\s+[^(),]+|\s+et\s+al\.)?,\s*"
    r"(?:\d{4}[a-z]?|n\.d\.(?:-[a-z])?)\)$"
)
_NARRATIVE_YEAR = re.compile(r"^.+\s\((?:\d{4}[a-z]?|n\.d\.(?:-[a-z])?)\)$")
_NUMERIC = re.compile(r"^(?:\[\d+\]|\(\d+\))$")
# The optional `-[a-z]` suffix is APA 7's n.d.-a/n.d.-b (see `_apa`); the other
# author-year styles never render it, so widening this is safe for them too.
_YEAR = re.compile(r"(?:\d{4}[a-z]?|n\.d\.(?:-[a-z])?)")
_APA_PAGE_RANGE = re.compile(r"(\d)-(\d)")

SUPPORTED_CITATION_STYLES: tuple[CitationStyle, ...] = (
    "elsevier-harvard",
    "apa-7",
    "chicago-author-date",
    "mla-9",
    "ieee",
    "vancouver",
)
_NUMERIC_STYLES = {"ieee", "vancouver"}


class CitationFormatError(ValueError):
    """Raised when prose contains an unknown or unsupported citation."""


@dataclass(frozen=True)
class _Author:
    family: str
    given: str

    @property
    def initials(self) -> str:
        parts = re.findall(r"[^\W\d_]+", self.given, flags=re.UNICODE)
        return " ".join(f"{part[0].upper()}." for part in parts)

    @property
    def compact_initials(self) -> str:
        return "".join(part.rstrip(".") for part in self.initials.split())


def _require_style(citation_style: str) -> CitationStyle:
    if citation_style not in SUPPORTED_CITATION_STYLES:
        raise CitationFormatError(f"unsupported citation style: {citation_style}")
    return citation_style


def _parse_author(value: str) -> _Author:
    cleaned = " ".join(value.split())
    if "," in cleaned:
        family, given = cleaned.split(",", maxsplit=1)
        return _Author(family.strip(), given.strip())
    parts = cleaned.split()
    if len(parts) == 1:
        return _Author(parts[0], "")
    return _Author(parts[-1], " ".join(parts[:-1]))


def _authors(metadata: PaperMetadata) -> list[_Author]:
    return [_parse_author(value) for value in metadata.authors]


def _join(values: list[str], *, final: str = "and") -> str:
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} {final} {values[1]}"
    return f"{', '.join(values[:-1])}, {final} {values[-1]}"


def _join_serial(values: list[str], *, final: str = "and") -> str:
    """Join bibliography authors with a serial comma, including two authors."""

    if len(values) == 1:
        return values[0]
    return f"{', '.join(values[:-1])}, {final} {values[-1]}"


def _year(metadata: PaperMetadata, suffix: str = "") -> str:
    return f"{metadata.year}{suffix}" if metadata.year is not None else "n.d."


def _doi_url(doi: str | None) -> str | None:
    if not doi:
        return None
    cleaned = re.sub(r"^(?:https?://doi\.org/|doi:\s*)", "", doi.strip(), flags=re.I)
    return f"https://doi.org/{cleaned}"


def _journal_parts(metadata: PaperMetadata) -> tuple[str, str, str, str]:
    return (
        metadata.journal or "",
        metadata.volume or "",
        metadata.issue or "",
        metadata.pages or "",
    )


def _author_year_names(authors: list[_Author], two_author_joiner: str) -> str:
    families = [author.family for author in authors]
    if len(families) == 1:
        return families[0]
    if len(families) == 2:
        return f"{families[0]} {two_author_joiner} {families[1]}"
    return f"{families[0]} et al."


def _elsevier(metadata: PaperMetadata, suffix: str) -> CitationInfo:
    authors = _authors(metadata)
    year = _year(metadata, suffix)
    names = _author_year_names(authors, "and")
    reference_names = ", ".join(
        filter(None, (f"{author.family}, {author.initials}".rstrip() for author in authors))
    )
    journal, volume, _issue, pages = _journal_parts(metadata)
    publication = journal
    if volume:
        publication += f" {volume}"
    if pages:
        publication += f", {pages}"
    parts = [f"{reference_names}, {year}. {metadata.title}."]
    if publication:
        parts.append(f"{publication}.")
    doi = _doi_url(metadata.doi)
    if doi:
        parts.append(doi)
    return CitationInfo(
        citation_key=f"{authors[0].family.casefold()}-{year}",
        in_text_citation=f"({names}, {year})",
        narrative_citation=f"{names} ({year})",
        reference_entry=" ".join(parts),
    )


def _apa(metadata: PaperMetadata, suffix: str) -> CitationInfo:
    authors = _authors(metadata)
    # APA 7's undated-work suffix is hyphenated (n.d.-a, n.d.-b), unlike the
    # plain letter other author-year styles append to a real year (2020a) —
    # `_year` stays shared and unchanged for those; this is apa-7 only.
    year = f"n.d.-{suffix}" if metadata.year is None and suffix else _year(metadata, suffix)
    names = _author_year_names(authors, "&")
    rendered_authors = [f"{author.family}, {author.initials}".rstrip() for author in authors]
    reference_names = (
        f"{', '.join(rendered_authors[:19])}, ... {rendered_authors[-1]}"
        if len(rendered_authors) > 20
        else _join_serial(rendered_authors, final="&")
    )
    journal, volume, issue, pages = _journal_parts(metadata)
    publication = ""
    if journal and volume:
        publication = f"*{journal}, {volume}*"
    elif journal:
        publication = f"*{journal}*"
    elif volume:
        publication = f"*{volume}*"
    if issue:
        publication += f"({issue})"
    if pages:
        # APA 7 renders a page range with an en dash (U+2013); a single page
        # like "e1234" has no digit-hyphen-digit to match, so it is unchanged.
        en_dash_pages = _APA_PAGE_RANGE.sub("\\1\u2013\\2", pages)
        publication += f", {en_dash_pages}"
    parts = [f"{reference_names} ({year}). {metadata.title}."]
    if publication:
        parts.append(f"{publication}.")
    doi = _doi_url(metadata.doi)
    if doi:
        parts.append(doi)
    narrative_names = _author_year_names(authors, "and")
    return CitationInfo(
        citation_key=f"{authors[0].family.casefold()}-{year}",
        in_text_citation=f"({names}, {year})",
        narrative_citation=f"{narrative_names} ({year})",
        reference_entry=" ".join(parts),
    )


def _chicago(metadata: PaperMetadata, suffix: str) -> CitationInfo:
    authors = _authors(metadata)
    year = _year(metadata, suffix)
    names = _author_year_names(authors, "and")
    full_names = [
        f"{author.family}, {author.given}" if index == 0 and author.given else
        " ".join(filter(None, [author.given, author.family]))
        for index, author in enumerate(authors)
    ]
    journal, volume, issue, pages = _journal_parts(metadata)
    publication = f"*{journal}*" if journal else ""
    if volume:
        publication += f" {volume}"
    if issue:
        publication += f", no. {issue}"
    if pages:
        publication += f": {pages}"
    parts = [f"{_join_serial(full_names)}. {year}. \"{metadata.title}.\""]
    if publication:
        parts.append(f"{publication}.")
    doi = _doi_url(metadata.doi)
    if doi:
        parts.append(doi)
    return CitationInfo(
        citation_key=f"{authors[0].family.casefold()}-{year}",
        in_text_citation=f"({names} {year})",
        narrative_citation=f"{names} ({year})",
        reference_entry=" ".join(parts),
    )


def _mla(metadata: PaperMetadata) -> CitationInfo:
    authors = _authors(metadata)
    first = authors[0]
    if len(authors) == 1:
        reference_names = f"{first.family}, {first.given}".rstrip(", ")
        in_text_name = first.family
    elif len(authors) == 2:
        second = " ".join(filter(None, [authors[1].given, authors[1].family]))
        reference_names = f"{first.family}, {first.given}, and {second}"
        in_text_name = f"{first.family} and {authors[1].family}"
    else:
        reference_names = f"{first.family}, {first.given}, et al."
        in_text_name = f"{first.family} et al."
    journal, volume, issue, pages = _journal_parts(metadata)
    segments = [f'{reference_names}. "{metadata.title}."']
    publication: list[str] = []
    if journal:
        publication.append(f"*{journal}*")
    if volume:
        publication.append(f"vol. {volume}")
    if issue:
        publication.append(f"no. {issue}")
    publication.append(str(metadata.year) if metadata.year is not None else "n.d.")
    if pages:
        publication.append(f"pp. {pages}")
    segments.append(", ".join(publication) + ".")
    doi = _doi_url(metadata.doi)
    if doi:
        segments.append(doi)
    return CitationInfo(
        citation_key=f"{first.family.casefold()}-{_year(metadata)}",
        in_text_citation=f"({in_text_name})",
        narrative_citation=in_text_name,
        reference_entry=" ".join(segments),
    )


def _numeric(metadata: PaperMetadata, number: int, style: CitationStyle) -> CitationInfo:
    authors = _authors(metadata)
    journal, volume, issue, pages = _journal_parts(metadata)
    year = str(metadata.year) if metadata.year is not None else "n.d."
    doi = _doi_url(metadata.doi)
    if style == "ieee":
        names = _join(
            [" ".join(filter(None, [author.initials, author.family])) for author in authors]
        )
        details = f"*{journal}*" if journal else ""
        if volume:
            details += f", vol. {volume}"
        if issue:
            details += f", no. {issue}"
        if pages:
            details += f", pp. {pages}"
        details += f", {year}"
        if doi:
            details += f", doi: {doi.removeprefix('https://doi.org/')}"
        reference = f'[{number}] {names}, "{metadata.title}," {details}.'
        in_text = f"[{number}]"
    elif style == "vancouver":
        names = ", ".join(
            " ".join(filter(None, [author.family, author.compact_initials]))
            for author in authors
        )
        details = f"{journal}. {year}" if journal else year
        if volume:
            details += f";{volume}"
        if issue:
            details += f"({issue})"
        if pages:
            details += f":{pages}"
        if doi:
            details += f". doi:{doi.removeprefix('https://doi.org/')}"
        reference = f"{number}. {names}. {metadata.title}. {details}."
        in_text = f"({number})"
    else:
        raise CitationFormatError(f"unsupported numeric citation style: {style}")
    return CitationInfo(
        citation_key=f"reference-{number}",
        in_text_citation=in_text,
        narrative_citation=in_text,
        reference_entry=reference,
    )


def build_citation_map(
    paper_ids: list[str],
    metadata_by_paper_id: dict[str, PaperMetadata],
    citation_style: str = "elsevier-harvard",
) -> dict[str, CitationInfo]:
    """Build citations in bibliography order with stable numeric numbering."""

    style = _require_style(citation_style)
    ordered_ids = list(dict.fromkeys(paper_ids))
    missing = [paper_id for paper_id in ordered_ids if paper_id not in metadata_by_paper_id]
    if missing:
        raise CitationFormatError("missing citation metadata for: " + ", ".join(missing))
    citation_numbers = {
        paper_id: number for number, paper_id in enumerate(ordered_ids, start=1)
    }
    bibliography_ids = (
        ordered_ids
        if style in _NUMERIC_STYLES
        else sorted(
            ordered_ids,
            key=lambda paper_id: (
                tuple(
                    author.family.casefold()
                    for author in _authors(metadata_by_paper_id[paper_id])
                ),
                -(metadata_by_paper_id[paper_id].year or 0),
                metadata_by_paper_id[paper_id].title.casefold(),
            ),
        )
    )
    suffixes = {paper_id: "" for paper_id in bibliography_ids}
    if style in {"elsevier-harvard", "apa-7", "chicago-author-date"}:
        collisions: dict[tuple[str, int | None], list[str]] = {}
        for paper_id in bibliography_ids:
            metadata = metadata_by_paper_id[paper_id]
            joiner = "&" if style == "apa-7" else "and"
            label = _author_year_names(_authors(metadata), joiner).casefold()
            collisions.setdefault((label, metadata.year), []).append(paper_id)
        suffixes = {
            paper_id: (chr(ord("a") + index) if len(group) > 1 else "")
            for group in collisions.values()
            for index, paper_id in enumerate(group)
        }
    rendered: dict[str, CitationInfo] = {}
    for paper_id in bibliography_ids:
        metadata = metadata_by_paper_id[paper_id]
        if style == "elsevier-harvard":
            rendered[paper_id] = _elsevier(metadata, suffixes[paper_id])
        elif style == "apa-7":
            rendered[paper_id] = _apa(metadata, suffixes[paper_id])
        elif style == "chicago-author-date":
            rendered[paper_id] = _chicago(metadata, suffixes[paper_id])
        elif style == "mla-9":
            rendered[paper_id] = _mla(metadata)
        else:
            rendered[paper_id] = _numeric(
                metadata, citation_numbers[paper_id], style
            )
    return rendered


def citation_format_errors(
    citation: CitationInfo, citation_style: str = "elsevier-harvard"
) -> list[str]:
    """Return deterministic formatting errors for one rendered citation record."""

    style = _require_style(citation_style)
    errors: list[str] = []
    if style in _NUMERIC_STYLES:
        if _NUMERIC.fullmatch(citation.in_text_citation) is None:
            errors.append("invalid numeric in_text_citation")
        if citation.narrative_citation != citation.in_text_citation:
            errors.append("invalid numeric narrative_citation")
    elif style == "mla-9":
        if not citation.in_text_citation.startswith(
            "("
        ) or not citation.in_text_citation.endswith(")"):
            errors.append("invalid MLA in_text_citation")
    else:
        if style == "chicago-author-date":
            valid_parenthetical = bool(
                re.fullmatch(
                    r"\(.+\s(?:\d{4}[a-z]?|n\.d\.)\)",
                    citation.in_text_citation,
                )
            )
        else:
            valid_parenthetical = _AUTHOR_YEAR.fullmatch(citation.in_text_citation) is not None
        if not valid_parenthetical:
            errors.append("invalid author-year in_text_citation")
        if _NARRATIVE_YEAR.fullmatch(citation.narrative_citation) is None:
            errors.append("invalid author-year narrative_citation")
    if _YEAR.search(citation.reference_entry) is None:
        errors.append("reference_entry does not contain a year marker")
    return errors


def extract_cited_paper_ids(content: str) -> list[str]:
    """Return citation placeholder IDs in first-appearance order. A
    range/list shorthand like `[P003-P006]` expands to every ID it covers
    (see `_expand_citation_match`) rather than vanishing from extraction."""

    ids: list[str] = []
    for match in _CITATION.finditer(content):
        ids.extend(_expand_citation_match(match))
    return list(dict.fromkeys(ids))


def format_citations(
    content: str,
    citations: dict[str, CitationInfo],
    citation_style: str = "elsevier-harvard",
) -> str:
    """Replace validated paper placeholders with configured in-text citations."""

    style = _require_style(citation_style)

    def render_one(paper_id: str) -> str:
        try:
            rendered = citations[paper_id].in_text_citation
        except KeyError as error:
            raise CitationFormatError(f"unknown citation placeholder: {paper_id}") from error
        format_errors = citation_format_errors(citations[paper_id], style)
        if format_errors:
            raise CitationFormatError(
                f"invalid {style} citation for {paper_id}: " + "; ".join(format_errors)
            )
        return rendered

    def replace(match: re.Match[str]) -> str:
        # A range/list shorthand (see `_expand_citation_match`) renders as
        # each covered paper's own citation, joined — never silently
        # collapsed to just the first or last paper in the range.
        return "; ".join(render_one(paper_id) for paper_id in _expand_citation_match(match))

    return _CITATION.sub(replace, content)


def build_reference_list(
    paper_ids: list[str], citations: dict[str, CitationInfo]
) -> list[str]:
    """Build a deduplicated reference list in first-citation order."""

    entries: list[str] = []
    for paper_id in dict.fromkeys(paper_ids):
        try:
            entries.append(citations[paper_id].reference_entry)
        except KeyError as error:
            raise CitationFormatError(f"missing reference entry for: {paper_id}") from error
    return entries
