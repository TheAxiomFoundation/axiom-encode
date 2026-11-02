"""A month name alone on its line never swallows the next line's number (#1822).

A month standing alone on its line is a table column header, and the number on
the next line is a row value. A month inside wrapped prose still pairs with the
day or year on the next line, as on main.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import re
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as pipeline_module
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures"
MONTHS = (
    "January",
    "Feb",
    "March",
    "Apr.",
    "May",
    "June",
    "Jul",
    "August",
    "Sept",
    "October",
    "Nov.",
    "December",
)


def _values(
    text: str,
    profile: str = "en-US",
    extractor=extract_typed_numeric_inventory_occurrences_from_text,
) -> list[float]:
    return [occurrence.value for occurrence in extractor(text, profile=profile)]


def _grounding(text: str, profile: str = "en-US") -> list[float]:
    return _values(text, profile, extract_typed_numeric_occurrences_from_text)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0.01", [0.01]),
        ("7.65", [7.65]),
        ("2026", [2026.0]),
        ("1-2", [1.0, 2.0]),
    ],
)
@pytest.mark.parametrize("month", ["May", "Jan", "Feb", "Mai"])
def test_issue_probes_keep_the_number_after_a_month_line(month, value, expected):
    text = f"Benefit amounts, in dollars:\n{month}\n{value} Monthly Allowance"

    assert _values(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The amount changes on May 15 of each year.",
        "The amount changes in May 2026.",
        "The amount changes in Jan 2026.",
        "Effective May\n15, 2026, the allowance applies.",
        "Effective 15\nMay 2026, the allowance applies.",
    ],
)
def test_dates_with_their_own_evidence_stay_dates(text):
    """Same-line dates, and line-broken dates carrying day and year, stay dates."""

    assert _values(text) == []


@pytest.mark.parametrize("month", ["May", "Jan", "Feb", "Mai"])
@pytest.mark.parametrize("indent", ["", "  ", "\t"])
def test_indented_month_header_keeps_the_row_value(month, indent):
    text = f"Benefit amounts, in dollars:\n{indent}{month}\n7.65 Monthly Allowance"

    assert _values(text) == [7.65]


WRAPPED_PROSE_DATES = [
    ("en-US", "The rate applies in May\n2026 to every claim."),
    ("en-US", "The credit is paid on May\n15 of each year."),
    ("en-US", "Beginning January\n2027, the rate applies."),
    ("de-DE", "Der Betrag gilt ab Januar\n2026 für jedes Kind."),
    ("de-DE", "Der Betrag galt im Mai\n2025 und danach."),
]


@pytest.mark.parametrize("profile,text", WRAPPED_PROSE_DATES)
def test_wrapped_prose_dates_stay_dates(profile, text):
    """A month inside wrapped prose pairs with the next line, as on main."""

    assert _values(text, profile) == []
    with _main_patterns():
        assert _values(text, profile) == []


_CREDIT_ARTIFACT = """format: rulespec/v1
module:
  summary: Credit.
rules:
  - name: credit_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '40'
"""


def _credit_issues(source: str):
    return analyze_complete_source_unit(
        _CREDIT_ARTIFACT,
        source,
        corpus_citation_path="us/statute/26/21",
        test_cases=None,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


@pytest.mark.parametrize(
    "source",
    [
        "Beginning in January 2026, the credit is $40 per taxpayer.\n",
        "Beginning in January\n2026, the credit is $40 per taxpayer.\n",
    ],
)
def test_wrapped_date_does_not_refuse_a_correct_encoding(source):
    """The review's end-to-end repro: a wrapped date is not a numeric demand."""

    assert _credit_issues(source) == ()


@pytest.mark.parametrize("profile", ["en-US", "de-DE"])
@pytest.mark.parametrize("month", ["Mai", "May"])
@pytest.mark.parametrize("separator", ["\n", "\r\n", "\n  "])
def test_header_year_removed_as_a_line_label_is_still_grounded(
    profile, month, separator
):
    """``Mai\n2026.`` keeps 2026 for grounding, as on main (review P3)."""

    text = f"Betrag\n{month}{separator}2026. Weiter"

    assert 2026.0 in _grounding(text, profile)
    with _main_patterns():
        assert 2026.0 in _grounding(text, profile)


LINE_SEPARATORS = ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x85", "\u2028", "\u2029"]


@pytest.mark.parametrize("separator", LINE_SEPARATORS, ids=repr)
def test_every_line_separator_ends_a_month_header(separator):
    text = f"Benefit amounts, in dollars:{separator}May{separator}7.65 Allowance"

    assert _values(text) == [7.65]


@pytest.mark.parametrize("separator", LINE_SEPARATORS, ids=repr)
def test_every_line_separator_inside_prose_keeps_the_date(separator):
    text = f"The rate applies in May{separator}2026 to every claim."

    assert _values(text) == []


_AMOUNTS = st.one_of(
    st.integers(min_value=1, max_value=99_999).map(str),
    st.builds(
        lambda whole, cents: f"{whole}.{cents:02d}",
        st.integers(min_value=0, max_value=9_999),
        st.integers(min_value=1, max_value=99),
    ),
)


@settings(max_examples=150, deadline=None)
@given(
    month=st.sampled_from(MONTHS),
    amounts=st.lists(_AMOUNTS, min_size=1, max_size=4),
)
def test_property_month_header_does_not_change_row_recall(month, amounts):
    """A table whose column header is a month name recalls every row."""

    rows = "\n".join(f"{amount} Monthly Allowance" for amount in amounts)
    with_header = _values(f"Benefit amounts, in dollars:\n{month}\n{rows}")
    without_header = _values(f"Benefit amounts, in dollars:\n{rows}")

    assert with_header == without_header
    assert Counter(with_header) == Counter(float(amount) for amount in amounts)


# The three patterns as main shipped them: a month paired with the next line
# whatever stood around it. With them in place the header-year helper finds
# nothing, so the extractors behave exactly as on main.
_LEGACY_MONTH_NAME_DAY_PATTERN = re.compile(
    rf"\b{pipeline_module._MONTH_NAME_BODY}\s+\d{{1,2}}\b",
    re.IGNORECASE,
)
_LEGACY_GERMAN_MONTH_YEAR_PATTERN = re.compile(
    rf"\b{pipeline_module._GERMAN_MONTH_NAME_BODY}\s+(?:des\s+Jahres\s+)?"
    rf"(?P<year>{pipeline_module._TEMPORAL_YEAR_BODY})"
    rf"{pipeline_module._TEMPORAL_YEAR_END}",
    re.IGNORECASE,
)
_LEGACY_ENGLISH_MONTH_YEAR_PATTERN = re.compile(
    rf"\b{pipeline_module._MONTH_NAME_BODY}\s+"
    rf"(?P<year>{pipeline_module._TEMPORAL_YEAR_BODY})"
    rf"{pipeline_module._TEMPORAL_YEAR_END}",
    re.IGNORECASE,
)


@contextlib.contextmanager
def _main_patterns():
    saved = {
        name: getattr(pipeline_module, name)
        for name in (
            "_MONTH_NAME_DAY_PATTERN",
            "_GERMAN_MONTH_YEAR_PATTERN",
            "_ENGLISH_MONTH_YEAR_PATTERN",
        )
    }
    pipeline_module._MONTH_NAME_DAY_PATTERN = _LEGACY_MONTH_NAME_DAY_PATTERN
    pipeline_module._GERMAN_MONTH_YEAR_PATTERN = _LEGACY_GERMAN_MONTH_YEAR_PATTERN
    pipeline_module._ENGLISH_MONTH_YEAR_PATTERN = _LEGACY_ENGLISH_MONTH_YEAR_PATTERN
    pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(pipeline_module, name, value)
        pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()


def _losses(text: str, profile: str, extractor) -> Counter[float]:
    pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()
    after = Counter(_values(text, profile, extractor))
    with _main_patterns():
        before = Counter(_values(text, profile, extractor))
    lost = before - after
    # The issue's range shape: main read a header-broken ``1-2`` as ``-2``.
    return Counter(
        {
            value: count
            for value, count in lost.items()
            if not (value < 0 and -value in after)
        }
    )


_SWEEP_MONTHS = ("May", "Jan", "Feb.", "September", "Mai", "Januar", "März")
_SWEEP_SEPARATORS = ("\n", "\r\n", "\n  ", "\n\t", "\u2028")
_SWEEP_NUMBERS = ("2026", "2026.", "2026,", "15", "15,", "7.65", "1-2", "0.01")
_SWEEP_LAYOUTS = (
    # (before the month, after the number); a leading line break makes the
    # month a header, prose before it makes the month part of a sentence.
    ("Amounts:\n", " Allowance"),
    ("Amounts:\n", ". Weiter"),
    ("The rate applies in ", " to every claim."),
    ("Der Betrag gilt ab ", " für jedes Kind."),
)


@pytest.mark.parametrize("profile", ["en-US", "de-DE"])
@pytest.mark.parametrize(
    "extractor",
    [
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
    ],
    ids=["inventory", "grounding"],
)
def test_generated_sweep_never_loses_a_value_main_kept(profile, extractor):
    """The review's loss sweep: months × separators × numbers × layouts.

    Nothing main extracted is lost, except the issue's ``1-2`` → ``-2`` shape,
    and prose layouts extract exactly what main extracted.
    """

    for month, separator, number, (lead, tail) in itertools.product(
        _SWEEP_MONTHS, _SWEEP_SEPARATORS, _SWEEP_NUMBERS, _SWEEP_LAYOUTS
    ):
        text = f"{lead}{month}{separator}{number}{tail}"
        assert not _losses(text, profile, extractor), (text, profile)
        if not lead.endswith("\n"):
            pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()
            after = _values(text, profile, extractor)
            with _main_patterns():
                assert _values(text, profile, extractor) == after, (text, profile)


def _fixture_texts():
    paths = sorted(
        path
        for path in FIXTURES.rglob("*")
        if path.is_file()
        and (path.name == "source.txt" or path.suffix == ".txt")
        and path.stat().st_size < 2_000_000
    )
    return [pytest.param(path, id=str(path.relative_to(FIXTURES))) for path in paths]


@pytest.mark.parametrize("profile", ["en-US", "de-DE"])
@pytest.mark.parametrize("path", _fixture_texts())
def test_differential_fixture_texts_never_lose_a_value(path, profile):
    """Over every fixture text, both extractors keep every value main kept.

    The review found no fixture with a month at the end of a line followed by a
    digit, so this guards against regressions rather than exercising #1822; the
    generated sweep above exercises it.
    """

    text = path.read_text(errors="replace")
    for extractor in (
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
    ):
        assert not _losses(text, profile, extractor), (path.name, profile)
