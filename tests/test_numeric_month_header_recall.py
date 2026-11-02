"""A bare month-name line never swallows the number on the next line (#1822)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as pipeline_module
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
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


def _values(text: str) -> list[float]:
    return [
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile="en-US"
        )
    ]


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


# The two patterns as main shipped them: their whitespace crossed line breaks.
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


def _fixture_texts():
    paths = sorted(
        path
        for path in FIXTURES.rglob("*")
        if path.is_file()
        and (path.name == "source.txt" or path.suffix == ".txt")
        and path.stat().st_size < 2_000_000
    )
    return [pytest.param(path, id=str(path.relative_to(FIXTURES))) for path in paths]


@pytest.mark.parametrize("path", _fixture_texts())
def test_differential_fixture_recall_only_gains(monkeypatch, path):
    """Over every fixture text, the change only adds recalled numbers.

    The one allowed exception is the issue's range shape, where main read a
    line-broken ``May`` / ``1-2`` as ``-2``: any value main recalled that the
    fix does not must be the negation of a value the fix now recalls.
    """

    text = path.read_text(errors="replace")
    pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()
    after = Counter(_values(text))
    with monkeypatch.context() as patch:
        patch.setattr(
            pipeline_module,
            "_MONTH_NAME_DAY_PATTERN",
            _LEGACY_MONTH_NAME_DAY_PATTERN,
        )
        patch.setattr(
            pipeline_module,
            "_GERMAN_MONTH_YEAR_PATTERN",
            _LEGACY_GERMAN_MONTH_YEAR_PATTERN,
        )
        patch.setattr(
            pipeline_module,
            "_ENGLISH_MONTH_YEAR_PATTERN",
            _LEGACY_ENGLISH_MONTH_YEAR_PATTERN,
        )
        pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()
        before = Counter(_values(text))
    pipeline_module._tokenize_numeric_occurrences_from_text.cache_clear()

    lost = before - after
    assert all(value < 0 and -value in after for value in lost), lost
