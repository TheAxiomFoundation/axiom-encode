"""Titled reference lists in complete-source numeric recall.

UK instruments print a title after each provision they refer to. Regulation 14
of the Council Tax Reduction (Scotland) Regulations 2012 reads "regulation 15
(persons not entitled to council tax reduction: absentees), regulation 16
(...) or 19 (...)". The plain English citation pass removes each keyword-led
item, but its number list stops at the title, so the keyword-less 19 stayed in
the recall text as an amount. Signed re-encode run 41745442025 ended with that
demand among its issues ("Authoritative corpus numeric value 19 has no named
scalar representation"), and any re-encode of regulation 14 meets it.

Invariants pinned here:

* every item number of a titled list leaves the recall inventory;
* a title's own amounts stay, as do a following untitled quantity and amounts
  outside the list;
* a parenthesis that does not name a provision keeps its item out of the list:
  a bare marker, a glued or wrapped parenthesis, one that opens with a number
  word or with a word carrying on the amount or limiting the reference, one
  that holds a sentence break, and one that runs past 300 characters;
* a list on a line the extractor reads as a heading stays whole, and that
  check agrees with the extractor's own heading pattern;
* on the generated lists, which include all of those shapes, the pass adds no
  demand;
* the pass keeps every title verbatim and every line break, and blanks the
  rest of the list in place, so the recall text keeps its length there.
"""

from __future__ import annotations

import contextlib
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    _STRUCTURAL_SOURCE_HEADING_PATTERN,
    _STRUCTURAL_SOURCE_QUOTE_CHARS,
    _numeric_profile_for_citation_path,
    extract_typed_numeric_inventory_occurrences_from_text,
)

CITATION = "uk/regulation/ssi/2012/319/14"
# The corpus text of regulation 14 byte for byte: its SHA-256 equals the
# generation_input_sha256 that run 41745442025 attested (corpus release
# uk-rulespec-2026-10-25-pe-child-definitions).
REGULATION_14 = (
    Path(__file__).parent
    / "fixtures"
    / "source_completeness"
    / "uk_ssi_2012_319_14_source.txt"
)
_NEVER = re.compile(r"(?!)")


def recall_values(text: str) -> list[float]:
    """The values complete-source recall demands, as the validator reads them."""
    recall = sc.authoritative_numeric_recall_text(text, corpus_citation_path=CITATION)
    return sorted(
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            recall, profile=_numeric_profile_for_citation_path(CITATION)
        )
    )


@contextlib.contextmanager
def titled_list_pass_disabled() -> Iterator[None]:
    saved = sc._ENGLISH_TITLED_REFERENCE_LIST
    sc._ENGLISH_TITLED_REFERENCE_LIST = _NEVER
    try:
        yield
    finally:
        sc._ENGLISH_TITLED_REFERENCE_LIST = saved


def test_regulation_14_demands_only_its_rate() -> None:
    text = REGULATION_14.read_text(encoding="utf-8")

    # 2 6/7 per cent, the only quantity regulation 14 states.
    assert recall_values(text) == [pytest.approx(2 / 70)]
    with titled_list_pass_disabled():
        assert recall_values(text) == [pytest.approx(2 / 70), 19.0]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            "is not a person to whom regulation 15 (persons not entitled to "
            "council tax reduction: absentees), regulation 16 (persons not "
            "entitled to council tax reduction: persons treated as not being in "
            "Great Britain) or 19 (persons not entitled to council tax "
            "reduction: persons subject to immigration control) applies; and",
            [],
            id="regulation-14-3-b",
        ),
        pytest.param(
            "in accordance with Part 9 (applications) or 10 (decisions on "
            "applications)",
            [],
            id="part-keyword",
        ),
        pytest.param(
            "regulation 57 (residents of a dwelling to whom regulation 14(6) "
            "does not apply) or 58 (date on which entitlement begins)",
            [],
            id="nested-title",
        ),
        pytest.param(
            "if regulation 15 (persons aged 65 or over) or 19 (persons subject "
            "to immigration control) applies",
            [65.0],
            id="title-amount-stays",
        ),
        pytest.param(
            "regulations 14A (council tax reduction: alternative calculation) "
            "and 20 (amount of reduction) apply, reduced by 25 per cent",
            [0.25],
            id="amount-outside-list-stays",
        ),
        pytest.param(
            "A person to whom regulation 5 (persons), 6 (persons) or 7 persons apply.",
            [7.0],
            id="untitled-tail-after-one-word-titles",
        ),
        pytest.param(
            # A keyword left before "(persons)" would read as a reference to
            # that title, and the reference would run on into the untitled 7.
            "A person to whom regulation 5 (persons), regulation 6 (persons) or "
            "7 persons apply.",
            [7.0],
            id="untitled-tail-after-repeated-keyword",
        ),
        pytest.param(
            "under regulation 15 (absentees), 16 (persons) or 19 persons",
            [19.0],
            id="titled-items-leave-untitled-item-stays",
        ),
        pytest.param(
            "A person to whom regulation 15 (absentees) or\n"
            "19 (persons subject to immigration control) applies for 5 persons.",
            [5.0],
            id="list-across-a-line-break",
        ),
    ],
)
def test_titled_list_items_leave_recall(text: str, expected: list[float]) -> None:
    assert recall_values(text) == expected


_NON_TITLE_OPENINGS = (
    "or more",
    "and over",
    "but not more than 20",
    "plus costs",
    "minus fees",
    "less tax",
    "per annum",
    "for couples",
    "up to 30",
    "at least 30",
    "at most 30",
    "not more than 30",
    "not less than 30",
    "not fewer than 30",
    "not exceeding 30",
    "no more than 30",
    "no less than 30",
    "no fewer than 30",
    "more than 30",
    "except paragraph 2",
    "excluding paragraph 2",
    "with the exception of paragraph 2",
    "other than paragraph 2",
    "only if paragraph 2 applies",
    "provided that paragraph 2 applies",
    "unless paragraph 2 applies",
    "if paragraph 2 applies",
    "subject to paragraph 2",
    "notwithstanding paragraph 2",
    "save paragraph 2",
    "apart from paragraph 2",
    "read with paragraph 2",
    "as modified by paragraph 2",
    "which does not apply if paragraph 2 applies",
    "so far as paragraph 2 applies",
    "in so far as paragraph 2 applies",
    "insofar as paragraph 2 applies",
    "to the extent paragraph 2 applies",
    "where paragraph 2 applies",
    "nineteen",
    "nineteen per cent",
    "nineteen and one-half",
    "twenty-one",
    "one hundred and five",
    "two million",
    "two billion",
)


@pytest.mark.parametrize("opening", _NON_TITLE_OPENINGS)
def test_a_non_title_parenthesis_keeps_its_item(opening: str) -> None:
    text = f"if regulation 15 (absentees) or 19 ({opening}) weeks have passed"
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert 19.0 in before
    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text) is None
    assert recall_values(text) == before


@pytest.mark.parametrize(
    "title",
    [
        "order of the court",
        "forfeiture",
        "periodic payments",
        "lessees",
        "savings",
        "notices",
        "tenants",
        "onerous property",
        "wherewithal",
    ],
)
def test_a_title_may_begin_with_the_letters_of_a_refused_word(title: str) -> None:
    text = f"if regulation 15 (absentees) or 19 ({title}) applies"

    assert recall_values(text) == []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "if regulation 15 (absentees) or 19 weeks have passed",
            id="untitled-quantity",
        ),
        pytest.param(
            "if regulation 15 (absentees) or " + chr(0xA3) + "19 (nineteen "
            "pounds) is paid",
            id="currency-amount",
        ),
        pytest.param(
            "if regulation 15 (3) or 19 (2) applies",
            id="digit-parenthesis-is-no-title",
        ),
        pytest.param(
            "under paragraph 3 (earnings), 16 (or more) hours",
            id="amount-after-a-keyword-led-title",
        ),
        pytest.param(
            "regulations 15 (absentees) to 19 (persons) apply",
            id="a-range-is-no-list",
        ),
        pytest.param(
            "if regulation 15\n(absentees) or 19\n(persons) applies",
            id="a-title-on-the-next-line-is-no-title",
        ),
        pytest.param(
            "if regulation 15 (absentees) or 19 (a (b (c) d) e) applies",
            id="two-nested-levels-are-no-title",
        ),
        pytest.param(
            "under subregulation 15 (absentees) or 19 (persons)",
            id="the-keyword-must-start-a-word",
        ),
    ],
)
def test_non_reference_numbers_keep_their_obligations(text: str) -> None:
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert 19.0 in before or 16.0 in before
    assert recall_values(text) == before


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("A person to whom Part 5 (a), 6 (b) or 7 applies.", id="markers"),
        pytest.param(
            "A person to whom Schedule 5 (a) and 6 (b), 7 and 8 apply.",
            id="markers-schedule",
        ),
        pytest.param(
            "A person to whom subsection 5 (i), 6 (ii) or 7 applies.",
            id="roman-markers",
        ),
        pytest.param(
            "A person to whom Part 5 (A), 6 (IV) or 7 applies.",
            id="upper-case-markers",
        ),
        pytest.param(
            "A person to whom subsection 5 (ii), 6 (iv) or 7 applies.",
            id="roman-markers-of-several-letters",
        ),
        pytest.param(
            "A person to whom subsection 5(a), 6(b) or 7 applies.", id="glued-markers"
        ),
        pytest.param(
            "A person to whom subparagraphs 5(a), 6(b) and 7 apply.",
            id="glued-markers-plural",
        ),
        pytest.param(
            "A person to whom Part 5(transitional), 6 (b) or 7 applies.",
            id="glued-title",
        ),
        pytest.param(
            "A person to whom regulation 15(absentees) or 19(persons) applies "
            "for 7 weeks.",
            id="glued-titles-throughout",
        ),
    ],
)
def test_marker_and_glued_parentheses_stay_as_they_were(text: str) -> None:
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text) is None
    assert recall_values(text) == before


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "See regulation 5 (average income. The rate is 7 pounds.) or 19 "
            "(absentees).",
            id="sentence-break-before-a-capital",
        ),
        pytest.param(
            "See regulation 5 (average income; The rate is 7 pounds) or 19 "
            "(absentees).",
            id="semicolon-before-a-capital",
        ),
        pytest.param(
            "See regulation 5 (average income. (1) The rate is 7 pounds) or 19 "
            "(absentees).",
            id="sentence-break-before-a-marker",
        ),
        pytest.param(
            "See regulation 5 (income (see the rate. It is 7 pounds) of a person) "
            "or 19 (absentees).",
            id="sentence-break-inside-a-nested-pair",
        ),
        pytest.param(
            "See paragraph 50 (average income\n(1) The amount is 7 pounds.\n"
            "(2) The rate applies to ( B x C )) or 19 (absentees).",
            id="a-parenthesis-left-open-reaches-a-later-close",
        ),
    ],
)
def test_a_sentence_break_keeps_a_parenthesis_from_being_a_title(text: str) -> None:
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert 19.0 in before
    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text) is None
    assert recall_values(text) == before


@pytest.mark.parametrize(
    "title",
    [
        pytest.param("persons etc. in receipt of benefit", id="stop-before-lower-case"),
        pytest.param("income; capital", id="semicolon-before-lower-case"),
        pytest.param("sums payable under s. 5", id="stop-before-a-number"),
        pytest.param("persons treated as absent etc.", id="stop-before-the-close"),
        pytest.param("residents of St.Andrews", id="stop-with-no-space-after-it"),
    ],
)
def test_a_stop_that_ends_no_sentence_stays_in_a_title(title: str) -> None:
    text = f"if regulation 15 (absentees) or 19 ({title}) applies"

    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text) is not None
    assert 19.0 not in recall_values(text)


def test_a_title_ends_within_300_characters() -> None:
    def text(length: int) -> str:
        return f"if regulation 15 (absentees) or 19 ({'q' * length}) applies"

    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text(300)) is not None
    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text(301)) is None


def test_a_single_titled_reference_is_no_list() -> None:
    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search("regulation 15 (absentees)") is None


def test_a_month_in_parentheses_reads_as_a_title() -> None:
    # Intended limit: only the listed openings refuse a parenthesis, so any
    # other word-led parenthesis after a number reads as a title.
    assert recall_values("section 5 (commencement), 1 (January) 2020") == [2020.0]


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "Part 9 (applications) or 10 (decisions) applies to 7 persons.",
            id="opens-with-part",
        ),
        pytest.param(
            "Part 9 (applications), 10 (decisions) and 11",
            id="untitled-last-item",
        ),
        pytest.param(
            "Schedule 1 (persons aged 65 or over) and 2 (students) apply.",
            id="title-amount-on-a-heading-line",
        ),
        pytest.param(
            "Chapter 1 (persons aged 65 or over) and 2 (students) apply.",
            id="opens-with-chapter",
        ),
        pytest.param(
            "Part 5 (paragraph 21 & 22), Part 12 (chapter 3, paragraph 105), Part "
            "14 (except paragraph 112)",
            id="repeated-keyword",
        ),
        pytest.param(
            "(1) Part 9 (applications) and 10 (decisions) apply to 7 persons.",
            id="behind-a-paragraph-marker",
        ),
        pytest.param(
            "(1) 1. Chapter 9 (applications) and 10 (decisions) apply to 7 persons.",
            id="behind-a-marker-and-a-number",
        ),
        pytest.param(
            "(1) bis (2) Part 9 (applications) and 10 (decisions) apply to 7 persons.",
            id="behind-a-marker-range",
        ),
        pytest.param(
            "A person to whom regulation 15 (absentees) or\nPart 19 (persons) "
            "applies for 5 weeks.",
            id="wrapped-onto-a-heading",
        ),
        pytest.param(
            "A person to whom regulation 15 (absentees) or\nPART 19 (persons) "
            "applies and 21 others.",
            id="wrapped-onto-an-upper-case-heading",
        ),
        pytest.param(
            "page one\x0cPart 9 (applications) or 10 (decisions) apply to 7 persons.",
            id="after-a-form-feed",
        ),
        pytest.param(
            "A person to whom regulation 15 (absentees) or\x0cPart 19 (persons) "
            "applies for 5 weeks.",
            id="wrapped-onto-a-heading-by-a-form-feed",
        ),
        pytest.param(
            "\xa0(1) Part 9 (applications) or 10 (decisions) apply to 7 persons.",
            id="guard-errs-toward-leaving-the-list-alone",
        ),
    ],
)
def test_a_heading_line_stays_as_it_was(text: str) -> None:
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert sc._ENGLISH_TITLED_REFERENCE_LIST.search(text) is not None
    assert recall_values(text) == before


def test_a_keyword_before_the_list_does_not_guard_a_later_line() -> None:
    # "Part 3" closes its own line, so the list that follows still leaves.
    assert (
        recall_values(
            "Part 3\nA person to whom regulation 15 (absentees) or 19 (persons) applies."
        )
        == []
    )


_LINE_OPENINGS = (
    "",
    " ",
    "\t",
    "(1) ",
    "(1)",
    "(a) ",
    "(A) ",
    "(12b) ",
    "(ii) ",
    "(iv)",
    "1. ",
    "12a. ",
    "1.",
    "1.2.3 ",
    "12 ",
    "A. ",
    "- ",
    chr(0x2022) + " ",
    "(1) bis (2) ",
    '"',
    "'",
    "`",
    chr(0x201C),
    chr(0x201D),
    chr(0x2018),
    chr(0x2019),
    chr(0x201C) + " ",
    "A person to whom ",
)
_HEADING_KEYWORDS = ("Part", "PART", "part", "Chapter", "Schedule")


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(st.sampled_from(_LINE_OPENINGS), max_size=3).map("".join),
    st.sampled_from(_HEADING_KEYWORDS),
)
def test_the_heading_check_agrees_with_the_extractor(
    opening: str, keyword: str
) -> None:
    text = f"{opening}{keyword} 9 (applications) or 10 (decisions) apply to 7 persons."
    match = sc._ENGLISH_TITLED_REFERENCE_LIST.search(text)
    assert match is not None
    with titled_list_pass_disabled():
        recall = sc.authoritative_numeric_recall_text(
            text, corpus_citation_path=CITATION
        )
    reads_as_heading = bool(
        _STRUCTURAL_SOURCE_HEADING_PATTERN.match(
            recall.strip().strip(_STRUCTURAL_SOURCE_QUOTE_CHARS)
        )
    )

    assert sc._titled_list_touches_heading_line(match) is reads_as_heading


def test_the_heading_check_mirrors_the_extractors_quote_characters() -> None:
    pattern = sc._ENGLISH_HEADING_LINE_OPENING
    for quote in _STRUCTURAL_SOURCE_QUOTE_CHARS:
        assert pattern.match(f"{quote}Part 9") is not None


def test_unclosed_title_finishes_and_changes_nothing() -> None:
    text = "regulation 15 (absentees) or 19 (" + "persons not entitled " * 2000
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert recall_values(text) == before


def test_the_heading_check_stays_linear_on_a_long_blank_run() -> None:
    # Only the pass: the extractor's own cost grows faster than the text here.
    blank = " " * 200_000
    text = f"{blank}\n(1) {blank}Part 9 (x y) or 10 (y z) applies to 7."

    assert (
        sc._ENGLISH_TITLED_REFERENCE_LIST.sub(sc._blank_titled_reference_list, text)
        == text
    )


_KEYWORDS = (
    "regulation",
    "regulations",
    "section",
    "sections",
    "sec.",
    "secs.",
    "paragraph",
    "paragraphs",
    "article",
    "articles",
    "part",
    "parts",
    "schedule",
    "schedules",
    "chapter",
    "chapters",
    "sub-paragraph",
    "sub-paragraphs",
    "subparagraph",
    "subparagraphs",
    "subsection",
    "subsections",
)
_LINE_HEADING_KEYWORDS = frozenset({"part", "schedule", "chapter"})
_SEPARATORS = (
    ", ",
    ", and ",
    ", or ",
    " and ",
    " or ",
    ",\n",
    " and\n",
    " or\n",
    " or\r\n",
)
_TITLE_WORDS = (
    "persons",
    "not",
    "entitled",
    "to",
    "council",
    "tax",
    "reduction:",
    "absentees",
    "immigration",
    "control",
    "date",
    "on",
    "which",
    "entitlement",
    "begins",
    "applications",
)


TitledList = tuple[str, list[str], list[str], list[float]]


@st.composite
def titled_lists(draw: st.DrawFn) -> TitledList:
    """A titled reference list, its item numbers and titles, and title amounts."""
    keyword = draw(st.sampled_from(_KEYWORDS))
    keyword = draw(st.sampled_from((keyword, keyword.capitalize())))
    count = draw(st.integers(min_value=2, max_value=5))
    numbers: list[str] = []
    titles: list[str] = []
    amounts: list[float] = []
    parts: list[str] = []
    for index in range(count):
        number = str(draw(st.integers(min_value=1, max_value=99))) + draw(
            st.sampled_from(("", "", "A", "ZB"))
        )
        words = draw(st.lists(st.sampled_from(_TITLE_WORDS), min_size=1, max_size=6))
        if draw(st.booleans()):
            # Title amounts never collide with item numbers (1-99).
            amount = draw(st.integers(min_value=100, max_value=999))
            amounts.append(float(amount))
            words = [*words, "aged", str(amount), "or", "over"]
        title = f"({' '.join(words)})"
        item = f"{number} {title}"
        if index == 0:
            parts.append(f"{keyword} {item}")
        else:
            separator = draw(st.sampled_from(_SEPARATORS))
            # A singular Part, Chapter or Schedule that opens a line reads as
            # a heading, which test_a_heading_line_stays_as_it_was covers.
            may_repeat = not (
                separator.endswith("\n") and keyword.lower() in _LINE_HEADING_KEYWORDS
            )
            repeat = f"{keyword} " if may_repeat and draw(st.booleans()) else ""
            parts.append(f"{separator}{repeat}{item}")
        numbers.append(number)
        titles.append(title)
    return "".join(parts), numbers, titles, sorted(amounts)


_SETTINGS = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@_SETTINGS
@given(titled_lists())
def test_only_item_numbers_leave_recall(case: TitledList) -> None:
    reference_list, numbers, _titles, amounts = case
    text = f"A person to whom {reference_list} applies is not entitled."
    with titled_list_pass_disabled():
        before = recall_values(text)
    after = recall_values(text)

    assert after == amounts
    assert Counter(after) <= Counter(before)
    item_numbers = {float(re.match(r"\d+", number).group()) for number in numbers}
    assert set((Counter(before) - Counter(after)).elements()) <= item_numbers


@_SETTINGS
@given(titled_lists(), st.integers(min_value=1, max_value=99))
def test_untitled_tail_keeps_its_obligation(case: TitledList, tail: int) -> None:
    reference_list, _numbers, _titles, amounts = case
    text = f"A person to whom {reference_list} or {tail} persons apply."

    assert recall_values(text) == sorted([*amounts, float(tail)])


@_SETTINGS
@given(titled_lists())
def test_blanking_keeps_only_titles(case: TitledList) -> None:
    reference_list, _numbers, titles, _amounts = case
    text = f"A person to whom {reference_list} applies."
    matches = list(sc._ENGLISH_TITLED_REFERENCE_LIST.finditer(text))

    assert len(matches) == 1
    match = matches[0]
    assert match.group(0) == reference_list
    blanked = sc._blank_titled_reference_list(match)
    assert len(blanked) == len(reference_list)
    assert all(
        replacement in (" ", original)
        for original, replacement in zip(reference_list, blanked)
    )
    assert blanked.split() == " ".join(titles).split()
    assert [c for c in blanked if c in "\r\n"] == [
        c for c in reference_list if c in "\r\n"
    ]


_HOSTILE_KEYWORDS = (*_KEYWORDS, "Part", "PART", "Chapter", "Schedule", "SCHEDULE")
_HOSTILE_TITLES = (
    "a",
    "b",
    "i",
    "ii",
    "iv",
    "x",
    "persons",
    "absentees",
    "applications",
    "persons aged 65 or over",
    "see regulation 14(6)",
    "rate of 10%",
    "or more",
    "per week",
    "for couples",
    "January",
    "nineteen",
    "nineteen per cent",
    "twenty-one",
    "one hundred and five",
    "paragraph 21 & 22",
    "Part 3",
)
_HOSTILE_GAPS = (" ", " ", "", "\t", "  ", "\n")
_HOSTILE_SEPARATORS = (*_SEPARATORS, ",\n\n")
_HOSTILE_PREFIXES = (
    "",
    "A person to whom ",
    "\n",
    "(1) ",
    "(1) 1. ",
    "1. ",
    chr(0x201C),
    "under ",
    "See ",
    "Part 3\n",
    "12 months and ",
    "25 per cent of ",
    "  ",
)
_HOSTILE_TAILS = (
    "",
    " applies.",
    " or 7",
    ", 7",
    " and 8 persons",
    " to 9",
    " applies for 5 weeks",
    "\nPart 4 applies for 6 weeks",
    " 12",
    " per cent",
    " weeks",
    " or 7 persons",
    ", 7 and 8",
    ", 7 (a)",
    " (a)",
    "\n7",
)


@st.composite
def hostile_texts(draw: st.DrawFn) -> str:
    """A text built to tempt the pass: every shape a review found adding a demand."""

    def item() -> str:
        return (
            str(draw(st.integers(min_value=1, max_value=99)))
            + draw(st.sampled_from(("", "", "A", "ZB")))
            + draw(st.sampled_from(_HOSTILE_GAPS))
            + f"({draw(st.sampled_from(_HOSTILE_TITLES))})"
        )

    body = draw(st.sampled_from(_HOSTILE_KEYWORDS)) + " " + item()
    for _ in range(draw(st.integers(min_value=1, max_value=3))):
        body += draw(st.sampled_from(_HOSTILE_SEPARATORS))
        if draw(st.booleans()):
            body += draw(st.sampled_from(_HOSTILE_KEYWORDS)) + " "
        body += item()
    return (
        draw(st.sampled_from(_HOSTILE_PREFIXES))
        + body
        + draw(st.sampled_from(_HOSTILE_TAILS))
    )


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(hostile_texts())
def test_the_pass_adds_no_demand_on_generated_lists(text: str) -> None:
    with titled_list_pass_disabled():
        before = Counter(recall_values(text))

    assert not Counter(recall_values(text)) - before
