"""Titled reference lists in complete-source numeric recall.

UK instruments print a title after each provision they refer to. Regulation 14
of the Council Tax Reduction (Scotland) Regulations 2012 reads "regulation 15
(persons not entitled to council tax reduction: absentees), regulation 16
(...) or 19 (...)". The plain English citation pass removes each keyword-led
item, but a title ends its number list, so the keyword-less 19 stayed in the
recall text as an amount. That demand ("Authoritative corpus numeric value 19
has no named scalar representation") blocked the final candidate of signed
re-encode run 41745442025, and any re-encode of regulation 14 meets it.

Invariants pinned here:

* every item number of a titled list leaves the recall inventory;
* nothing else does: a title's own amounts, a following quantity without a
  title, a number that a parenthesis only spells out, and amounts outside the
  list all stay obligations;
* the pass only removes obligations: with it disabled the inventory is a
  superset, and the difference holds only item numbers;
* the pass keeps every title verbatim and blanks the rest of the list in
  place, so the recall text keeps its length there.
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
            # A keyword left before "(persons)" would read as a lettered
            # subdivision reference and carry on into the untitled 7.
            "A person to whom regulation 5 (persons), 6 (persons) or 7 persons apply.",
            [7.0],
            id="untitled-tail-after-one-word-titles",
        ),
    ],
)
def test_titled_list_items_leave_recall(text: str, expected: list[float]) -> None:
    assert recall_values(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "if regulation 15 (absentees) or 19 weeks have passed",
            id="untitled-quantity",
        ),
        pytest.param(
            "if regulation 15 (absentees) or 19 (nineteen) weeks have passed",
            id="spelled-number-is-no-title",
        ),
        pytest.param(
            "if regulation 15 (absentees) or 21 (twenty-one) weeks have passed",
            id="hyphenated-spelled-number-is-no-title",
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
    ],
)
def test_non_reference_numbers_keep_their_obligations(text: str) -> None:
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert 19.0 in before or 21.0 in before
    assert recall_values(text) == before


def test_unclosed_title_finishes_and_changes_nothing() -> None:
    text = "regulation 15 (absentees) or 19 (" + "persons not entitled " * 2000
    with titled_list_pass_disabled():
        before = recall_values(text)

    assert recall_values(text) == before


_KEYWORDS = (
    "regulation",
    "regulations",
    "section",
    "paragraph",
    "article",
    "part",
    "schedule",
    "chapter",
    "sub-paragraph",
    "subsection",
)
_SEPARATORS = (", ", ", and ", ", or ", " and ", " or ")
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
            repeat = f"{keyword} " if draw(st.booleans()) else ""
            parts.append(f"{draw(st.sampled_from(_SEPARATORS))}{repeat}{item}")
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
