"""Grounding English month names, as #1745 grounds Hebrew ones.

#1745 taught the validator to read a dated Hebrew month name ("ספטמבר 1939")
as its month number, so an encoding of National Insurance Law schedule A1
part D that compares a numeric birth month (``birth_month <= 6``) grounds. A
faithful English rendering of the same schedule ("September 1939 to April
1940") still reported 6, 6, 7, 5, 12 and 9 as ungrounded (finding N8 of the
axiom-encode#1951 review): the reading existed only in Hebrew, so the same
schedule passed in one language and failed in the other.

English month names now ground behind the same guards. The name must be dated
on its own line and table cell. A month number grounds only a literal that
forms one whole side of a comparison with a calendar-month operand, and only
when every literal of that value in the module is such a comparison. The
operand-name rules, including the ``insured``/``required`` tokens of
axiom-encode#2045, are shared with Hebrew and unchanged.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.validator_pipeline import (
    _ENGLISH_GREGORIAN_MONTH_NUMBERS,
    GROUNDING_ALLOWED_VALUES,
    _month_comparison_literal_values,
    extract_numeric_occurrences_from_text,
    extract_typed_month_occurrences_from_text,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)
from tests.test_hebrew_month_and_index_selector_grounding import (
    EXPECTED_DISQUALIFYING_TOKENS,
    PART_D_CITATION,
    PART_D_MONTH_NUMBERS,
    PART_D_TEXT,
    SIXTEEN_ROWS,
    _cohort_selector_module,
    _month_module,
    _part_d_birth_month_module,
)

# PART_D_TEXT rendered faithfully in English: the same rows, years and ages,
# with the Hebrew month names and connectives in English ("עד" = "until" or
# "to", the prefix "ו" = "and", "ואילך" = "onwards", "ו־4 חודשים" = "and 4
# months").
ENGLISH_PART_D_TEXT = """Part D
(Sections 245(a), 342(c), 351(b) and 406(a)(4)(a))
Age of eligibility for a senior citizen's pension for women by month of birth
Month of birth | Age of eligibility (in years)
Until June 1939 | 65
July and August 1939 | 65 and 4 months
September 1939 to April 1940 | 65 and 8 months
May to December 1940 | 66
January to August 1941 | 66 and 4 months
September 1941 to April 1942 | 66 and 8 months
May 1942 to December 1944 | 67
January to August 1945 | 67 and 4 months
September 1945 to April 1946 | 67 and 8 months
May to December 1946 | 68
January to August 1947 | 68 and 4 months
September 1947 to April 1948 | 68 and 8 months
May to December 1948 | 69
January to August 1949 | 69 and 4 months
September 1949 to April 1950 | 69 and 8 months
May 1950 onwards | 70"""

# The months Part D names, in English.
ENGLISH_PART_D_MONTHS = {
    "January": 1.0,
    "April": 4.0,
    "May": 5.0,
    "June": 6.0,
    "July": 7.0,
    "August": 8.0,
    "September": 9.0,
    "December": 12.0,
}

UNGROUNDED_PREFIX = "Ungrounded generated numeric literal: "


def _ungrounded_values(issues: list[str]) -> list[str]:
    """The reported literals, in order and with repeats."""
    return [
        issue.removeprefix(UNGROUNDED_PREFIX).split(" ", 1)[0]
        for issue in issues
        if issue.startswith(UNGROUNDED_PREFIX)
    ]


def _reported(content: str, source: str) -> set[str]:
    return set(
        _ungrounded_values(
            find_ungrounded_numeric_issues(
                content, source_text=source, source_citation_path=PART_D_CITATION
            )
        )
    )


def _month_values(text: str) -> set[float]:
    """The month numbers the text's dated month names ground."""
    return {
        occurrence.value
        for occurrence in extract_typed_month_occurrences_from_text(text)
    }


def _english_month_occurrences(text: str) -> set[tuple[str, float, bool]]:
    """Month occurrences read from English names: raw word, value, temporal."""
    return {
        (occurrence.raw, occurrence.value, occurrence.has_temporal_context)
        for occurrence in extract_typed_month_occurrences_from_text(text)
        if occurrence.raw.capitalize() in _ENGLISH_GREGORIAN_MONTH_NUMBERS
    }


@pytest.fixture(autouse=True)
def _cold_tokenizer_cache():
    # Tests below monkeypatch the recognizers; never reuse a cached reading.
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    yield
    vp._tokenize_numeric_occurrences_from_text.cache_clear()


# --- (a) English Part D ------------------------------------------------------


def test_english_part_d_grounds_the_birth_months_an_encoding_compares():
    # main reported 6, 6, 7, 5, 12 and 9 here, in that order.
    assert (
        find_ungrounded_numeric_issues(
            _part_d_birth_month_module(),
            source_text=ENGLISH_PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
        == []
    )


def test_english_part_d_reported_the_birth_months_without_the_reading(monkeypatch):
    # The counterexample, kept as the contrast with the test above.
    monkeypatch.setattr(vp, "_iter_english_gregorian_month_matches", lambda *_a: [])

    assert _ungrounded_values(
        find_ungrounded_numeric_issues(
            _part_d_birth_month_module(),
            source_text=ENGLISH_PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
    ) == ["6", "6", "7", "5", "12", "9"]


def test_minimal_counterexample_now_grounds():
    # The smallest pair the delta-debugger found on main: one row, one
    # comparison. Hebrew grounded it; English reported 5.
    content = _month_module("if birth_month >= 5: 1 else: 0")

    assert _reported(content, "מאי 1950 ואילך") == set()
    assert _reported(content, "May 1950 onwards") == set()


# The month values Part D also prints as numbers, as months of age.
PART_D_PRINTED_MONTH_VALUES = {4.0, 8.0}


def test_english_part_d_grounds_exactly_the_months_it_names():
    assert _month_values(ENGLISH_PART_D_TEXT) == PART_D_MONTH_NUMBERS
    assert set(ENGLISH_PART_D_MONTHS.values()) == PART_D_MONTH_NUMBERS


@pytest.mark.parametrize(
    ("month", "value"),
    sorted(ENGLISH_PART_D_MONTHS.items(), key=lambda item: item[1]),
)
def test_each_month_english_part_d_names_grounds_its_number(month, value):
    content = _month_module(f"if birth_month == {value:g}: 1 else: 0")
    # The same schedule with this month renamed to October, which Part D
    # never names: the reading must come from this month's own name.
    renamed = re.sub(rf"\b{month}\b", "October", ENGLISH_PART_D_TEXT)

    assert value in _month_values(ENGLISH_PART_D_TEXT)
    assert value not in _month_values(renamed)
    assert _reported(content, ENGLISH_PART_D_TEXT) == set()
    if value in GROUNDING_ALLOWED_VALUES | PART_D_PRINTED_MONTH_VALUES:
        # January is 1, which never needs grounding; Part D prints 4 and 8 as
        # months of age ("and 4 months"), which ground the literal anyway.
        assert _reported(content, renamed) == set()
    else:
        assert _reported(content, renamed) == {f"{value:g}"}


def test_a_month_english_part_d_does_not_name_stays_ungrounded():
    # October and November are named in no Part D row, in either language.
    content = _part_d_birth_month_module().replace(
        "birth_month >= 9: 8", "birth_month >= 10: 8"
    )

    assert _reported(content, ENGLISH_PART_D_TEXT) == {"10"}
    assert _reported(content, PART_D_TEXT) == {"10"}


def test_english_month_numbers_ground_but_are_never_recall_obligations(monkeypatch):
    # Part D prints 4 and 8 as months of age ("and 4 months"); those stay
    # obligations. The birth months it names add none.
    recall = extract_numeric_occurrences_from_text(ENGLISH_PART_D_TEXT)
    inventory = [
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            ENGLISH_PART_D_TEXT
        )
    ]
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    monkeypatch.setattr(vp, "_iter_english_gregorian_month_matches", lambda *_a: [])

    assert recall == extract_numeric_occurrences_from_text(ENGLISH_PART_D_TEXT)
    assert inventory == recall
    assert set(recall) == {4.0, 8.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0}


# --- (b) Which English month names count ------------------------------------

# Each sentence dates the month name "{m}" in a different way.
ENGLISH_DATED_CONTEXTS = (
    "{m} 1942",
    "{m}, 1942",
    "{m} of 1942",
    "{m} of the year 2011",
    "{m} in the year 2011",
    "{m} 30, 2011",
    "{m} 1st 1950",
    "on 4 {m}",
    "on {m} 4",
    "on the 4th of {m}",
    "on the first day of {m} of the tax year",
    "on the thirty-first day of {m}",
    "in the month of {m} immediately following",
    "of the months {m}, July",
    "{m} of each year",
    "{m} 15 of each year",
    "{m} of the following calendar year",
    "{m} through December",
    "(5 {m} 2008)",
)
# Contexts dated by a day alone. "May" is also the modal verb, so it needs a
# year, the year it recurs in, or "month of": a bare day beside it is too
# often a section or an age ("Section 4 May Be Waived").
ENGLISH_DAY_ONLY_CONTEXTS = frozenset(
    {"on 4 {m}", "on {m} 4", "on the 4th of {m}", "on the thirty-first day of {m}"}
)
# Each sentence uses "{m}" with nothing that dates it.
ENGLISH_UNDATED_CONTEXTS = (
    "{m}",
    "in {m}",
    "the word {m} is written here",
    "the months of {m}",
    "65 and 4 months\n{m} sufficiency",
    "| 4 | {m} sufficiency",
    "1939\n{m} sufficiency",
    # A month heading a column, with the year in the row under it (#1822).
    "{m}\n1939",
    "  {m}\n  30, 1939",
    "{m} | 1939",
    # Every break str.splitlines knows ends the line, not only "\n".
    "{m}\r1939",
    "{m}\u20281939",
)
ENGLISH_CASES = (str, str.upper)


@pytest.mark.parametrize("context", ENGLISH_DATED_CONTEXTS)
def test_every_english_month_grounds_its_number_when_dated(context):
    for name, value in _ENGLISH_GREGORIAN_MONTH_NUMBERS.items():
        for case in ENGLISH_CASES:
            word = case(name)
            text = context.format(m=word)

            occurrences = _english_month_occurrences(text)

            if name == "May" and context in ENGLISH_DAY_ONLY_CONTEXTS:
                assert occurrences == set(), (text, occurrences)
            else:
                assert (word, value, True) in occurrences, (text, occurrences)


@pytest.mark.parametrize("context", ENGLISH_UNDATED_CONTEXTS)
def test_no_english_month_grounds_a_number_undated(context):
    for name in _ENGLISH_GREGORIAN_MONTH_NUMBERS:
        for case in ENGLISH_CASES:
            text = context.format(m=case(name))

            assert _english_month_occurrences(text) == set(), text


@pytest.mark.parametrize(
    "text",
    [
        # Title-case headings, where the modal verb is capitalised.
        "Section 4 May Be Waived",
        "Persons Aged 18 May Apply",
        "Benefits Under Paragraph 2 May Be Reduced",
        "THE BOARD MAY 4 TIMES A YEAR",
        "MAY 4",
        "May 4",
        "on the 4th of May",
        "the first of May",
        "May or may not",
        "Two May",
    ],
)
def test_may_the_modal_verb_grounds_nothing(text):
    assert 5.0 not in _month_values(text)


@pytest.mark.parametrize(
    "text",
    [
        "May 1950",
        "MAY 1950",
        "May, 1950",
        "May 1, 1950",
        "1 May 2020",
        "the 1st of May, 2020",
        "May 1 of each year",
        "May of the following year",
        "in the month of May",
        "May and June",
        "April, May",
    ],
)
def test_may_the_month_grounds_five(text):
    assert 5.0 in _month_values(text)


@pytest.mark.parametrize(
    "text",
    [
        # Legislation capitalises a month; lowercase is an ordinary word.
        "june 1939",
        "may 1950",
        "the march 1990",
        "august 1, 1990",
        # Abbreviations mostly date citations and source credits.
        "Jun. 1939",
        "Jun 1939",
        "Pub. L. 99-514, Oct. 22, 1986",
        "Sept. 1, 2020",
        "Dec 31, 2020",
        # A month name inside a longer word.
        "Mayor 1950",
        "Marshall 1950",
        "Juneau 1950",
        "Augustine 1950",
        "Junes 1950",
        "June_1950",
    ],
)
def test_only_full_capitalised_month_names_count(text):
    assert _month_values(text) == set()


def test_an_english_list_of_months_is_dated_as_a_whole():
    # July is dated by the year after August, and a list after "the months"
    # is dated by its own length however long it runs.
    assert {7.0, 8.0} <= _month_values("July and August 1939")
    assert {1.0, 4.0, 7.0, 10.0} <= _month_values(
        "in each of the months April, July, October and January, for the quarter"
    )
    assert {1.0, 4.0, 7.0, 10.0} <= _month_values(
        "in each of the months April, July, October, and January"
    )
    assert {2.0, 11.0} <= _month_values(
        "on the fifteenth day of each of the ten months February through November"
    )
    assert {5.0, 12.0} <= _month_values("May–December 1940")


def test_an_english_month_and_its_dating_words_share_a_line_and_cell():
    assert _month_values("June\n30") == set()
    assert _month_values("30\nJune") == set()
    assert _month_values("30 | June") == set()
    assert _month_values("the month of\nJune") == set()
    assert _month_values("June\nof each year") == set()
    assert _month_values("June and\nJuly") == set()
    assert _month_values("June | July") == set()


def test_a_day_or_month_word_too_far_back_dates_nothing():
    filler = " the amount payable under this paragraph"  # 40 characters
    assert len(filler) == vp._ENGLISH_MONTH_CONTEXT_WINDOW
    assert 6.0 in _month_values("the month of June")
    assert 6.0 not in _month_values("the month of" + filler + " June")
    assert 6.0 in _month_values("30 June")
    assert 6.0 not in _month_values("30" + filler + " June")


# A day the window reaches in full dates the name; the tail of a longer number
# the window cuts ("1230") is no day. The patterns search the text by position,
# so they see the digit before the window, as they would with no window.
WINDOW_EDGE_PADDING = " " * 38


@pytest.mark.parametrize(
    ("name", "value"),
    (("June", 6.0), ("JUNE", 6.0), ("ביוני", 6.0)),
)
def test_a_number_the_window_cuts_is_no_day(name, value):
    window = (
        vp._ENGLISH_MONTH_CONTEXT_WINDOW
        if name.isascii()
        else vp._HEBREW_MONTH_CONTEXT_WINDOW
    )
    assert len("30" + WINDOW_EDGE_PADDING) == window

    assert _month_values("30" + WINDOW_EDGE_PADDING + name) == {value}
    assert _month_values("1230" + WINDOW_EDGE_PADDING + name) == set()
    assert _month_values("1.30" + WINDOW_EDGE_PADDING + name) == set()
    assert _month_values("1230 " + name) == set()
    # A cell break just before the window leaves the day whole.
    assert _month_values("12|30" + WINDOW_EDGE_PADDING + name) == {value}


def test_a_word_the_window_cuts_is_no_month_of():
    padding = " " * (vp._ENGLISH_MONTH_CONTEXT_WINDOW - len("month of"))
    assert _month_values("month of" + padding + "May") == {5.0}
    assert _month_values("semimonth of" + padding + "May") == set()


# --- (c) Month numbers ground only month comparisons, in English -------------

DATED_DECEMBER_TEXT_EN = (
    "The pension is payable from December 1, 1940 in the amount of 500 dollars a month."
)


def _ungrounded_for(formula: str) -> set[str]:
    return _reported(_month_module(formula), DATED_DECEMBER_TEXT_EN)


def test_a_dated_english_month_does_not_ground_an_unrelated_literal():
    assert _ungrounded_for("monthly_amount * 12 + 500") == {"12"}


@pytest.mark.parametrize(
    "formula",
    (
        "if birth_month >= 12: 500 else: 0",
        "if 12 <= birth_month: 500 else: 0",
        "if person.birth_month == 12: 500 else: 0",
        "if month_of_birth != 12: 0 else: 500",
    ),
)
def test_a_dated_english_month_grounds_a_month_comparison(formula):
    assert _ungrounded_for(formula) == set()


@pytest.mark.parametrize(
    "formula",
    (
        # The same value also used as a quantity: no month grounding at all.
        "if birth_month >= 12: monthly_amount * 12 else: 500",
        # A count of months is a quantity, not a month.
        "if months_employed >= 12: 500 else: 0",
        "if month_count >= 12: 500 else: 0",
        "if age_in_months >= 12: 500 else: 0",
        # Arithmetic on the month operand is not a plain month comparison.
        "if birth_month + 1 >= 12: 500 else: 0",
        "if birth_month >= 12 * 1: 500 else: 0",
        "if 1 * 12 <= birth_month: 500 else: 0",
    ),
)
def test_a_dated_english_month_grounds_nothing_outside_a_plain_month_comparison(
    formula,
):
    assert "12" in _ungrounded_for(formula)


def test_english_month_occurrences_ground_nothing_by_default():
    (month,) = extract_typed_month_occurrences_from_text(DATED_DECEMBER_TEXT_EN)
    assert (month.raw, month.value, month.requires_month_context) == (
        "December",
        12.0,
        True,
    )
    assert not numeric_value_is_grounded(12.0, [month])
    assert numeric_value_is_grounded(
        12.0, [month], month_context_values=frozenset({12.0})
    )
    # The stream every other consumer reads never holds the name.
    shared = extract_typed_numeric_occurrences_from_text(DATED_DECEMBER_TEXT_EN)
    assert {occurrence.value for occurrence in shared} >= {1940.0, 500.0}
    assert not any(
        occurrence.requires_month_context
        or occurrence.value == 12.0
        or "December" in occurrence.raw
        for occurrence in shared
    )


def test_english_proof_scoped_and_flat_paths_ground_only_month_comparisons(
    monkeypatch,
):
    def scoped(formula: str) -> set[str]:
        return set(
            _ungrounded_values(
                find_ungrounded_numeric_issues_scoped(
                    _month_module(formula),
                    module_source_text=DATED_DECEMBER_TEXT_EN,
                    module_citation_path=PART_D_CITATION,
                )
            )
        )

    assert scoped("if birth_month >= 12: 500 else: 0") == set()
    assert scoped("monthly_amount * 12 + 500") == {"12"}
    # Force evaluate_numeric_grounding_values onto its flat return path.
    monkeypatch.setattr(vp, "_rule_grounding_values_by_path", lambda *_a, **_k: [])
    assert _ungrounded_for("if birth_month >= 12: 500 else: 0") == set()
    assert _ungrounded_for("monthly_amount * 12 + 500") == {"12"}


HALF_YEAR_TEXT_EN = (
    "The insured person is entitled to the benefit if contributions were paid "
    "for at least half a year, beginning June 1, 2010."
)
ONE_YEAR_TEXT_EN = (
    "The insured person is entitled to the benefit if contributions were paid "
    "for a year, beginning December 31, 2010."
)


@pytest.mark.parametrize(
    ("formula", "source", "literal"),
    (
        # A half year or a year beside a commencement date: the 6 or 12 is the
        # encoder's own conversion and must stay reported.
        ("if qualifying_month_threshold >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if waiting_month_limit >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if month_cap >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if elapsed_month >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if insurance_period_month >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if contribution_month_floor >= 12: 1 else: 0", ONE_YEAR_TEXT_EN, "12"),
        ("if age_month >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if month_index >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
        ("if month_of_birth_date >= 6: 1 else: 0", HALF_YEAR_TEXT_EN, "6"),
    ),
)
def test_an_english_month_named_quantity_or_span_reports_the_literal(
    formula, source, literal
):
    assert literal in _reported(_month_module(formula), source)


@pytest.mark.parametrize(
    ("formula", "source"),
    (
        ("if birth_month >= 6: 1 else: 0", HALF_YEAR_TEXT_EN),
        ("if payment_month == 6: 1 else: 0", HALF_YEAR_TEXT_EN),
        ("if month < 6: 1 else: 0", HALF_YEAR_TEXT_EN),
        ("if person.month_of_birth <= 12: 1 else: 0", ONE_YEAR_TEXT_EN),
    ),
)
def test_an_english_calendar_month_operand_grounds_its_dated_month(formula, source):
    assert _reported(_month_module(formula), source) == set()


# axiom-encode#2045: a calendar-month name with "insured" or "required" before
# "month" still reports its literal. This change leaves that as it is, in
# English as in Hebrew.
ISSUE_2045_NAMES = (
    "insured_birth_month",
    "insured_person_birth_month",
    "insured_start_month",
    "required_payment_month",
    "required_filing_month",
)


@pytest.mark.parametrize("name", ISSUE_2045_NAMES)
@pytest.mark.parametrize(
    "source",
    (
        HALF_YEAR_TEXT_EN,
        "המבוטח זכאי לגמלה אם שילם דמי ביטוח במשך חצי שנה לפחות, החל מיום 1 ביוני 2010",
    ),
)
def test_issue_2045_names_still_report_the_literal_in_both_languages(name, source):
    assert "6" in _reported(_month_module(f"if {name} <= 6: 1 else: 0"), source)


@pytest.mark.parametrize("token", sorted(EXPECTED_DISQUALIFYING_TOKENS))
def test_every_disqualifying_token_reports_the_literal_in_english(token):
    assert "6" in _reported(
        _month_module(f"if {token}_month >= 6: 1 else: 0"), HALF_YEAR_TEXT_EN
    )


def test_disqualifying_tokens_are_unchanged():
    assert vp._MONTH_OPERAND_DISQUALIFYING_TOKENS == EXPECTED_DISQUALIFYING_TOKENS


# A month name beside a rate word takes rate context from it. It still grounds
# only its own month number: "July to December percent" prints no rate of
# 0.12. Before this change the Hebrew "%" row grounded 0.12 that way.
RATE_NEIGHBOUR_TEXTS = (
    "יולי ודצמבר %",
    "July to December percent",
    "July to December per cent",
    "July to December %",
)


@pytest.mark.parametrize("text", RATE_NEIGHBOUR_TEXTS)
def test_a_month_beside_a_rate_grounds_no_scaled_value(text):
    december = [
        occurrence
        for occurrence in extract_typed_month_occurrences_from_text(text)
        if occurrence.value == 12.0
    ]
    assert [occurrence.has_rate_context for occurrence in december] == [True]
    assert _reported(_month_module("birth_month >= 0.12"), text) == {"0.12"}
    assert _reported(_month_module("birth_month >= 12"), text) == set()


def test_a_month_occurrence_grounds_its_own_number_alone():
    december = vp.NumericOccurrence(
        value=12.0,
        start=0,
        end=8,
        raw="December",
        has_rate_context=True,
        requires_month_context=True,
    )
    december_context = frozenset({12.0})

    assert numeric_value_is_grounded(
        12.0, [december], month_context_values=december_context
    )
    assert not numeric_value_is_grounded(12.0, [december])
    assert not numeric_value_is_grounded(
        12.0, [december], month_context_values=frozenset({11.0})
    )
    assert not numeric_value_is_grounded(
        0.12, [december], month_context_values=frozenset({0.12})
    )


# --- (d) Hebrew and faithful English Part D agree ----------------------------

PART_D_MODULES = {
    "birth-month comparisons": _part_d_birth_month_module(),
    "an unnamed month": _part_d_birth_month_module().replace(
        "birth_month >= 9: 8", "birth_month >= 10: 8"
    ),
    "a month also used as a quantity": _part_d_birth_month_module().replace(
        "      else: 0\n", "      else: monthly_amount * 12\n", 1
    ),
    "single comparisons": _month_module(
        " or ".join(f"birth_month == {value}" for value in range(4, 13))
    ),
    "an index selector": _cohort_selector_module(SIXTEEN_ROWS),
    "an unrelated literal": _month_module("monthly_amount * 12 + 500"),
}


def _entry_point_verdicts(content: str, text: str) -> dict[str, object]:
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    return {
        "find_ungrounded": _ungrounded_values(
            find_ungrounded_numeric_issues(
                content, source_text=text, source_citation_path=PART_D_CITATION
            )
        ),
        "find_ungrounded_authoritative": _ungrounded_values(
            find_ungrounded_numeric_issues(
                content, text, authoritative_source_text=text
            )
        ),
        "find_ungrounded_scoped": _ungrounded_values(
            find_ungrounded_numeric_issues_scoped(
                content,
                module_source_text=text,
                module_citation_path=PART_D_CITATION,
            )
        ),
        "month_values": sorted(_month_values(text)),
    }


@pytest.mark.parametrize("variant", sorted(PART_D_MODULES))
def test_hebrew_and_english_part_d_agree_on_every_entry_point(variant):
    content = PART_D_MODULES[variant]

    hebrew = _entry_point_verdicts(content, PART_D_TEXT)
    english = _entry_point_verdicts(content, ENGLISH_PART_D_TEXT)

    assert english == hebrew


def test_the_part_d_variants_exercise_both_outcomes():
    # Guard against a vacuous agreement: some variants report literals.
    outcomes = {
        variant: _entry_point_verdicts(content, ENGLISH_PART_D_TEXT)["find_ungrounded"]
        for variant, content in PART_D_MODULES.items()
    }
    assert outcomes["birth-month comparisons"] == []
    assert outcomes["an unnamed month"] == ["10"]
    assert "12" in outcomes["a month also used as a quantity"]
    assert outcomes["single comparisons"] == ["10", "11"]
    # 12 is arithmetic, not a month; the schedule prints no 500 at all.
    assert outcomes["an unrelated literal"] == ["12", "500"]


def test_march_spelled_merets_is_the_one_intended_asymmetry():
    # The corpus spells March "מרס", which dates like any month, and "מרץ"
    # ("vigour") in a few later amendments, which Hebrew treats as a
    # homograph. English "March" is no homograph when capitalised, so a day
    # alone dates it in English and "מרס" but not "מרץ": Hebrew stays the
    # stricter side there.
    assert 3.0 in _month_values("ביום 4 במרס")
    assert 3.0 in _month_values("on 4 March")
    assert 3.0 not in _month_values("ביום 4 במרץ")


# --- (e) Properties ----------------------------------------------------------

MONTH_NAMES_BY_VALUE = {
    int(value): name for name, value in _ENGLISH_GREGORIAN_MONTH_NUMBERS.items()
}
HEBREW_NAMES_BY_VALUE = {
    1: "ינואר",
    2: "פברואר",
    # The corpus's usual spelling; "מרץ" is the homograph (see above).
    3: "מרס",
    4: "אפריל",
    5: "מאי",
    6: "יוני",
    7: "יולי",
    8: "אוגוסט",
    9: "ספטמבר",
    10: "אוקטובר",
    11: "נובמבר",
    12: "דצמבר",
}
# Values that need grounding: 1 to 3 are always allowed.
GROUNDED_MONTH_VALUES = range(4, 13)
# Days and years drawn from ranges that never print a month value 4 to 12.
DAYS = st.integers(min_value=13, max_value=28)
YEARS = st.integers(min_value=1900, max_value=2099)
MONTHS = st.integers(min_value=1, max_value=12)

# English snippets, each with the months it dates by construction. A snippet
# is one line, so no two snippets can date each other.
#   kind -> (template, dates the first month, dates the second month)
# where "dated" may depend on whether a month is May (the homograph).


def _english_snippet(kind: str, m: int, m2: int, day: int, year: int):
    name, name2 = MONTH_NAMES_BY_VALUE[m], MONTH_NAMES_BY_VALUE[m2]
    may, may2 = m == 5, m2 == 5
    snippets = {
        "year": (f"{name} {year}", {m}),
        "comma-year": (f"{name}, {year}", {m}),
        "of-year": (f"{name} of {year}", {m}),
        "day-year": (f"{name} {day}, {year}", {m}),
        "day-before-year": (f"{day} {name} {year}", {m}),
        "recurs": (f"{name} of each year", {m}),
        "day-recurs": (f"{name} {day} of each year", {m}),
        "month-of": (f"in the month of {name}", {m}),
        "day-after": (f"on {name} {day}", set() if may else {m}),
        "day-before": (f"on {day} {name}", set() if may else {m}),
        "list": (
            f"{name} and {name2}",
            set() if may and may2 else {m, m2},
        ),
        "list-year": (f"{name} to {name2} {year}", {m, m2}),
        "bare": (name, set()),
        "in": (f"in {name}", set()),
        "header": (f"{name}\n{year}", set()),
        "cell": (f"{name} | {year}", set()),
        "modal": (f"Persons Aged {day} {name} Apply", set() if may else {m}),
        "lowercase": (f"{name.lower()} {year}", set()),
    }
    return snippets[kind]


ENGLISH_SNIPPET_KINDS = (
    "year",
    "comma-year",
    "of-year",
    "day-year",
    "day-before-year",
    "recurs",
    "day-recurs",
    "month-of",
    "day-after",
    "day-before",
    "list",
    "list-year",
    "bare",
    "in",
    "header",
    "cell",
    "modal",
    "lowercase",
)

english_snippets = st.builds(
    _english_snippet,
    st.sampled_from(ENGLISH_SNIPPET_KINDS),
    MONTHS,
    MONTHS,
    DAYS,
    YEARS,
)

ALL_MONTH_COMPARISONS = _month_module(
    " or ".join(f"birth_month == {value}" for value in GROUNDED_MONTH_VALUES)
)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.lists(english_snippets, min_size=1, max_size=6))
def test_property_a_literal_grounds_only_when_its_month_is_dated(snippets):
    """A month comparison grounds exactly the months the source dates."""
    text = "\n".join(snippet_text for snippet_text, _dated in snippets)
    dated = set().union(*(dated for _text, dated in snippets))
    vp._tokenize_numeric_occurrences_from_text.cache_clear()

    reported = _reported(ALL_MONTH_COMPARISONS, text)

    assert reported == {
        str(value) for value in GROUNDED_MONTH_VALUES if value not in dated
    }, text
    assert _month_values(text) == {float(value) for value in dated}, text


def _bilingual_snippet(kind: str, m: int, m2: int, day: int, year: int, age: int):
    en, en2 = MONTH_NAMES_BY_VALUE[m], MONTH_NAMES_BY_VALUE[m2]
    he, he2 = HEBREW_NAMES_BY_VALUE[m], HEBREW_NAMES_BY_VALUE[m2]
    pairs = {
        # Schedule rows, as Part D writes them.
        "until": (f"עד {he} {year} | {age}", f"Until {en} {year} | {age}"),
        "pair": (
            f"{he} ו{he2} {year} | {age} ו־4 חודשים",
            f"{en} and {en2} {year} | {age} and 4 months",
        ),
        "span": (
            f"{he} {year} עד {he2} {year + 1} | {age} ו־8 חודשים",
            f"{en} {year} to {en2} {year + 1} | {age} and 8 months",
        ),
        "span-same-year": (
            f"{he} עד {he2} {year} | {age}",
            f"{en} to {en2} {year} | {age}",
        ),
        "onwards": (f"{he} {year} ואילך | {age}", f"{en} {year} onwards | {age}"),
        # Other dated forms.
        "recurs": (f"{he} של כל שנה", f"{en} of each year"),
        "day-only": (f"ביום {day} ב{he}", f"on {day} {en}"),
        "day-year": (f"{day} ב{he} {year}", f"{day} {en} {year}"),
        "month-of": (f"בחודש {he}", f"in the month of {en}"),
        "months-list": (f"מהחודשים {he}, {he2}", f"of the months {en}, {en2}"),
        # Undated.
        "bare": (he, en),
        "in": (f"ב{he}", f"in {en}"),
    }
    return pairs[kind]


BILINGUAL_KINDS = (
    "until",
    "pair",
    "span",
    "span-same-year",
    "onwards",
    "recurs",
    "day-only",
    "day-year",
    "month-of",
    "months-list",
    "bare",
    "in",
)

bilingual_schedules = st.lists(
    st.builds(
        _bilingual_snippet,
        st.sampled_from(BILINGUAL_KINDS),
        MONTHS,
        MONTHS,
        DAYS,
        YEARS,
        st.integers(min_value=60, max_value=75),
    ),
    min_size=1,
    max_size=8,
)


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(bilingual_schedules)
def test_property_hebrew_and_english_renderings_ground_the_same_months(rows):
    """The same schedule grounds the same months in Hebrew and in English."""
    hebrew = "\n".join(he for he, _en in rows)
    english = "\n".join(en for _he, en in rows)
    vp._tokenize_numeric_occurrences_from_text.cache_clear()

    assert _month_values(english) == _month_values(hebrew), (hebrew, english)
    assert _reported(ALL_MONTH_COMPARISONS, english) == _reported(
        ALL_MONTH_COMPARISONS, hebrew
    ), (hebrew, english)


# A literal's uses in a module: a plain month comparison, a comparison with a
# quantity named with "month", or arithmetic.
LITERAL_USES = {
    "month-comparison": "birth_month == {v}",
    "reversed-month-comparison": "{v} <= payment_month",
    "quantity-comparison": "months_employed >= {v}",
    "arithmetic": "monthly_amount * {v} > 0",
    "operand-arithmetic": "birth_month + 1 >= {v}",
}
MONTH_USES = frozenset({"month-comparison", "reversed-month-comparison"})


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    st.dictionaries(
        st.sampled_from(list(GROUNDED_MONTH_VALUES)),
        st.lists(st.sampled_from(sorted(LITERAL_USES)), min_size=1, max_size=3),
        min_size=1,
        max_size=5,
    ),
    st.sets(st.sampled_from(list(GROUNDED_MONTH_VALUES))),
)
def test_property_a_dated_month_grounds_only_month_comparisons(uses, dated):
    """A literal grounds from a month name only if every use is a month."""
    formula = " or ".join(
        LITERAL_USES[use].format(v=value)
        for value, value_uses in sorted(uses.items())
        for use in value_uses
    )
    # A header line keeps the source non-empty: with no source text at all the
    # check has nothing to compare against and reports nothing.
    text = "\n".join(
        [
            "Month of birth | Age of eligibility",
            *(f"{MONTH_NAMES_BY_VALUE[value]} 1950" for value in sorted(dated)),
        ]
    )
    content = _month_module(formula)
    vp._tokenize_numeric_occurrences_from_text.cache_clear()

    expected_month_values = {
        float(value)
        for value, value_uses in uses.items()
        if set(value_uses) <= MONTH_USES
    }
    assert _month_comparison_literal_values(content) == expected_month_values
    assert _reported(content, text) == {
        str(value)
        for value in uses
        if not (value in dated and float(value) in expected_month_values)
    }, (formula, text)


MONTH_STREAM_TOKENS = st.lists(
    st.one_of(
        st.sampled_from(
            [
                *_ENGLISH_GREGORIAN_MONTH_NUMBERS,
                *(name.upper() for name in _ENGLISH_GREGORIAN_MONTH_NUMBERS),
                *(name.lower() for name in _ENGLISH_GREGORIAN_MONTH_NUMBERS),
                *HEBREW_NAMES_BY_VALUE.values(),
                *(f"ב{name}" for name in HEBREW_NAMES_BY_VALUE.values()),
                "מרץ",
                "of each year",
                "the month of",
                "בחודש",
                "שנת",
                "and",
                "to",
                "עד",
                ",",
                "|",
                "\n",
                "Section",
                "may",
                "percent",
                "%",
                "$",
                "₪",
            ]
        ),
        st.integers(min_value=0, max_value=2100).map(str),
    ),
    max_size=14,
)


def _cold_tokenization(text: str, profile: str = "legacy"):
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    return vp._tokenize_numeric_occurrences_from_text(text, profile=profile)


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(MONTH_STREAM_TOKENS, st.sampled_from(sorted(vp._NUMERIC_EXTRACTION_PROFILES)))
def test_property_month_names_reach_only_the_month_stream(tokens, profile):
    """No month name, Hebrew or English, changes what other readers see.

    Source completeness, proof excerpts, reanchoring and worksheet
    certification read the grounding and inventory streams, so a month name
    must leave both exactly as they are without it, under every profile.
    """
    text = " ".join(tokens)
    on = _cold_tokenization(text, profile)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vp, "_iter_gregorian_month_name_matches", lambda *_a: [])
        off = _cold_tokenization(text, profile)
    vp._tokenize_numeric_occurrences_from_text.cache_clear()

    assert on.grounding == off.grounding
    assert on.inventory == off.inventory
    assert off.month_grounding == ()
    if profile != "legacy":
        # Only the legacy reader, which every non-German, non-Danish source
        # uses, reads month names, as on main.
        assert on.month_grounding == ()
        return
    assert sorted(
        (occurrence.start, occurrence.end, occurrence.value)
        for occurrence in on.month_grounding
    ) == sorted(
        (start, end, value)
        for (start, end), value in vp._iter_gregorian_month_name_matches(text)
    )
    for occurrence in on.month_grounding:
        assert occurrence.requires_month_context
        assert text[occurrence.start : occurrence.end] == occurrence.raw


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(MONTH_STREAM_TOKENS)
def test_property_english_month_names_only_ground_their_month(tokens):
    """An English name grounds its own month number, in a month comparison only."""
    text = " ".join(tokens)
    on = _cold_tokenization(text)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(vp, "_iter_english_gregorian_month_matches", lambda *_a: [])
        off = _cold_tokenization(text)
    vp._tokenize_numeric_occurrences_from_text.cache_clear()

    assert on.grounding == off.grounding
    assert on.inventory == off.inventory
    assert [
        occurrence
        for occurrence in off.month_grounding
        if occurrence not in on.month_grounding
    ] == []
    added = [
        occurrence
        for occurrence in on.month_grounding
        if occurrence not in off.month_grounding
    ]
    for occurrence in added:
        name = occurrence.raw
        assert name in (name.capitalize(), name.upper()), name
        month = _ENGLISH_GREGORIAN_MONTH_NUMBERS[name.capitalize()]
        assert month == occurrence.value
        # It grounds its month number where the module compares a month, and
        # nothing else: not without month context, and never a value scaled
        # from it, even when a neighbouring "percent" gives it rate context.
        assert numeric_value_is_grounded(
            month, [occurrence], month_context_values=frozenset({month})
        )
        assert not numeric_value_is_grounded(month, [occurrence])
        for scaled in (month / 100, month * 100):
            assert not numeric_value_is_grounded(
                scaled, [occurrence], month_context_values=frozenset({scaled})
            ), (text, occurrence)


# --- (f) Only the grounding decisions read a month number --------------------
#
# Source completeness, proof excerpts, reanchoring and worksheet certification
# read the shared grounding stream for intervals, boundaries, formula matching
# and counts, where a month name is no number. The month numbers live in a
# stream of their own (``_NumericTokenization.month_grounding``), which only
# the two grounding decisions read.

PACKAGE_ROOT = Path(vp.__file__).resolve().parents[1]
MONTH_EVIDENCE_NAMES = frozenset(
    {
        "month_grounding",
        "_grounding_decision_occurrences",
        "extract_typed_month_occurrences_from_text",
    }
)
# (module, enclosing class or function, name) for every reference to the
# month stream in the package. A new reader must be added here on purpose.
EXPECTED_MONTH_EVIDENCE_REFERENCES = {
    ("harness/validator_pipeline.py", "_NumericTokenization", "month_grounding"),
    (
        "harness/validator_pipeline.py",
        "_tokenize_numeric_occurrences_from_text",
        "month_grounding",
    ),
    (
        "harness/validator_pipeline.py",
        "extract_typed_month_occurrences_from_text",
        "month_grounding",
    ),
    (
        "harness/validator_pipeline.py",
        "_grounding_decision_occurrences",
        "month_grounding",
    ),
    (
        "harness/validator_pipeline.py",
        "_numeric_evidence_index",
        "_grounding_decision_occurrences",
    ),
    (
        "harness/validator_pipeline.py",
        "evaluate_numeric_grounding_values",
        "_grounding_decision_occurrences",
    ),
}


def _month_evidence_references() -> set[tuple[str, str, str]]:
    references: set[tuple[str, str, str]] = set()

    def visit(node: ast.AST, module: str, scope: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = node.name
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.keyword):
            name = node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        elif isinstance(node, ast.alias):
            name = node.name
        if name in MONTH_EVIDENCE_NAMES:
            references.add((module, scope, name))
        for child in ast.iter_child_nodes(node):
            visit(child, module, scope)

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(PACKAGE_ROOT).as_posix()
        visit(ast.parse(path.read_text(encoding="utf-8")), module, "<module>")
    return references


def test_only_the_grounding_decisions_read_the_month_stream():
    assert _month_evidence_references() == EXPECTED_MONTH_EVIDENCE_REFERENCES


# How a caller may take a tokenization: by naming the one stream it reads, or
# whole, for the grounding decisions only. A reader that kept or unpacked the
# whole tokenization (``dataclasses.asdict``, a stored variable) could reach
# the month stream without naming it.
TOKENIZATION_READS = frozenset(
    {"grounding", "inventory", "month_grounding", "_grounding_decision_occurrences"}
)


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _tokenization_reads() -> list[tuple[str, str, str]]:
    """(module, enclosing function, what it reads) for each tokenizer call."""
    reads = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        module = path.relative_to(PACKAGE_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }

        def scope_of(node: ast.AST) -> str:
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return node.name
            return "<module>"

        for node in ast.walk(tree):
            if _node_name(node) != "_tokenize_numeric_occurrences_from_text":
                continue
            call = parents.get(node)
            reader = parents.get(call)
            if not (isinstance(call, ast.Call) and call.func is node):
                read = "<not a call>"
            elif isinstance(reader, ast.Attribute) and reader.value is call:
                read = reader.attr
            elif (
                isinstance(reader, ast.Call)
                and reader.args == [call]
                and _node_name(reader.func) == "_grounding_decision_occurrences"
            ):
                read = "_grounding_decision_occurrences"
            else:
                read = f"<{type(reader).__name__}>"
            reads.append((module, scope_of(node), read))
    return reads


def test_every_tokenizer_call_names_the_stream_it_reads():
    reads = _tokenization_reads()
    assert len(reads) >= 10
    assert [read for read in reads if read[2] not in TOKENIZATION_READS] == []
    assert {scope for _module, scope, read in reads if read == "month_grounding"} == {
        "extract_typed_month_occurrences_from_text"
    }


def test_the_grounding_decisions_still_ground_part_d_through_the_month_stream():
    # The guard above would also pass if no one read the month stream; Part D
    # grounds only because the decisions do.
    content = _part_d_birth_month_module()
    for text in (ENGLISH_PART_D_TEXT, PART_D_TEXT):
        assert _reported(content, text) == set()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                vp, "_grounding_decision_occurrences", lambda tokens: tokens.grounding
            )
            vp._tokenize_numeric_occurrences_from_text.cache_clear()
            assert _reported(content, text) != set()
        vp._tokenize_numeric_occurrences_from_text.cache_clear()


# What the readers of the shared stream returned before any month name was
# read, in Hebrew (#1745) or English. An audit of those readers found #1745's
# Hebrew month numbers changing each of these.
MONTH_CONSUMER_TEXTS = {
    "Hebrew": ("יוני 1939", "חודש יוני"),
    "English": ("June 1939", "the month of June"),
}


@pytest.mark.parametrize("language", sorted(MONTH_CONSUMER_TEXTS))
def test_proof_excerpt_readers_never_count_a_month_name(language):
    from axiom_encode import cli

    dated, month_of = MONTH_CONSUMER_TEXTS[language]
    assert _month_values(dated) == {6.0}
    assert _month_values(month_of) == {6.0}

    assert dict(
        cli._proof_excerpt_numeric_occurrence_counter(dated, profile="legacy")
    ) == {1939.0: 1}
    assert (
        cli._closest_exact_source_excerpt(
            source_text="The allowance applies in 1939.",
            excerpt=f"The allowance applies in {dated}.",
        )
        == "The allowance applies in 1939."
    )
    assert (
        cli._legacy_exact_dependent_reanchor_query(
            "... percent",
            rule_source=f"{month_of} percent",
            numeric_profile="legacy",
        )
        is None
    )


@pytest.mark.parametrize("language", sorted(MONTH_CONSUMER_TEXTS))
def test_source_completeness_never_reads_a_month_name_as_a_bound(language):
    from axiom_encode.harness import source_completeness

    dated, _month_of = MONTH_CONSUMER_TEXTS[language]
    text = f"at least {dated}"
    branch = source_completeness.SourceStructureBranch(
        (), "source-unit", "source", text, 0, len(text)
    )

    assert (
        source_completeness._formula_interval_from_text(
            text,
            extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        )
        is None
    )
    assert (
        list(
            source_completeness._source_boundary_obligations(
                (branch,),
                extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
            )
        )
        == []
    )


@pytest.mark.parametrize("language", sorted(MONTH_CONSUMER_TEXTS))
def test_complete_source_analysis_asks_no_test_for_a_month_name(language):
    from axiom_encode.harness import source_completeness

    _dated, month_of = MONTH_CONSUMER_TEXTS[language]
    citation = "us/statute/example"
    source = f"The amount is 2 percent of income in {month_of}."

    def rule(name: str, formula: str, kind: str = "derived") -> dict:
        return {
            "name": name,
            "kind": kind,
            "dtype": "Decimal",
            "entity": "Person",
            "period": "Year",
            "versions": [{"effective_from": "2020-01-01", "formula": formula}],
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "formula",
                            "source": {
                                "corpus_citation_path": citation,
                                "excerpt": source,
                            },
                        }
                    ]
                }
            },
        }

    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "inputs": [
            {"name": "income", "dtype": "Decimal", "entity": "Person", "period": "Year"}
        ],
        "rules": [
            rule("result", "income * 0.02"),
            rule("rate", "0.02", "parameter"),
        ],
    }
    case = {
        "name": "case",
        "period": "2024",
        "input": {"income": 1},
        "output": {"result": "0.02"},
    }

    analysis = source_completeness.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=citation,
        test_cases=[case],
        extract_numeric_occurrences=extract_typed_numeric_inventory_occurrences_from_text,
        extract_numeric_grounding_occurrences=extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    assert [str(issue) for issue in analysis.issues] == []


WORKSHEET_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "source_path_shadow"


def _certify_worksheet_with_formula(formula: str):
    from axiom_encode.harness.source_path_evidence import certify_worksheet_paths

    candidate = yaml.safe_load(
        (WORKSHEET_FIXTURE / "candidate.yaml").read_text(encoding="utf-8")
    )
    source = (WORKSHEET_FIXTURE / "source.txt").read_text(encoding="utf-8")
    (rule,) = [
        rule
        for rule in candidate["rules"]
        if rule["name"] == "multiple_foreign_country_count_threshold"
    ]
    rule["versions"][0]["formula"] = formula
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source
    vp._tokenize_numeric_occurrences_from_text.cache_clear()
    return certify_worksheet_paths(
        candidate,
        source_text=source,
        corpus_citation_path=(
            "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
        ),
        test_cases=yaml.safe_load(
            (WORKSHEET_FIXTURE / "cases.yaml").read_text(encoding="utf-8")
        ),
        numeric_value_is_grounded=numeric_value_is_grounded,
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
    )


@pytest.mark.parametrize("formula", ("1 # יוני 2025", "1 # June 2025"))
def test_worksheet_certification_reads_no_month_in_a_formula_comment(formula):
    # A month named in a parameter's formula comment once left the parameter
    # "ungrounded selected parameter", with no certified condition.
    with_month = _certify_worksheet_with_formula(formula)
    without_month = _certify_worksheet_with_formula("1 # 2025")

    assert without_month.unresolved == ()
    assert without_month.condition_spans
    assert with_month.unresolved == without_month.unresolved
    assert with_month.condition_spans == without_month.condition_spans
