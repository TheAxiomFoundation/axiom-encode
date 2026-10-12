"""Bare outlines must retain operative numbers and reject ambiguous labels."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.bare_structure import (
    bare_outline,
    mask_bare_labels,
    slice_bare_path,
)


@given(
    amount=st.integers(min_value=10, max_value=999_999),
    count=st.integers(min_value=3, max_value=99),
    year=st.integers(min_value=1900, max_value=2099),
    percentage=st.integers(min_value=1, max_value=99),
    table_cell=st.integers(min_value=3, max_value=999),
)
@settings(max_examples=80, deadline=None)
def test_generated_mixed_outline_preserves_every_operative_number(
    amount, count, year, percentage, table_cell
):
    source = (
        "1\nA person is entitled if—\n"
        f"a\nthe amount is ${amount} for {count} children; and\n"
        f"b\nthe rate is {percentage}% from {year}.\n"
        "Table of annual rates\n"
        f"{table_cell}\nAnnual family entitlement\n"
        "2\nThe entitlement ends on death.\n"
    )
    outline = bare_outline(source)
    assert outline is not None
    assert tuple(label.path for label in outline) == (
        ("1",), ("1", "a"), ("1", "b"), ("2",)
    )
    masked = mask_bare_labels(source)
    expected = source
    for label in reversed(outline):
        expected = (
            expected[: label.start]
            + " " * (label.marker_end - label.start)
            + expected[label.marker_end :]
        )
    assert masked == expected
    assert f"${amount}" in masked
    assert f"{count} children" in masked
    assert f"{percentage}%" in masked
    assert str(year) in masked
    assert f"\n{table_cell}\nAnnual family entitlement" in masked
    assert len(source) == len(masked)
    assert source.count("\n") == masked.count("\n")


@pytest.mark.parametrize("token", ("a", "c", "2"))
def test_unpunctuated_repeated_or_out_of_sequence_label_fails_closed(token):
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person meets the first requirement;\n"
        f"{token}\nthe person meets another requirement\n"
        "b\nthe person meets the final requirement.\n"
        "2\nThe entitlement ends on death.\n"
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert slice_bare_path(source, ("1", "a")) is None


def test_table_cell_that_coincides_with_next_sibling_stays_operative():
    source = (
        "1\nThe entitlement is payable.\n"
        "2\nThe annual rate follows.\n"
        "Table of dependent children\n"
        "3\nNumber of dependent children.\n"
    )
    masked = mask_bare_labels(source)
    assert "\n3\nNumber of dependent children." in masked


def test_nested_roman_and_numeric_suffix_paths():
    source = (
        "1\nA person is entitled if—\n"
        "a\nthe person meets either requirement—\n"
        "i\nthe first condition is fulfilled; or\n"
        "ii\nthe second condition is fulfilled; and\n"
        "b\nthe person meets the final requirement.\n"
        "1A\nAn additional restriction is applicable.\n"
        "2\nThe entitlement ends on death.\n"
    )
    assert slice_bare_path(source, ("1", "a", "ii")) == (
        "ii\nthe second condition is fulfilled; and"
    )
    assert slice_bare_path(source, ("1A",)) == (
        "1A\nAn additional restriction is applicable."
    )


def test_parenthetical_body_has_no_bare_outline_or_mask_changes():
    source = (
        "(1) A person is entitled if—\n"
        "(a) the amount is $25 for 3 children; and\n"
        "(b) the rate is 20% from 2026.\n"
        "(2) The entitlement ends on death."
    )
    assert bare_outline(source) is None
    assert mask_bare_labels(source) == source
    assert re.findall(r"\d+", mask_bare_labels(source)) == re.findall(r"\d+", source)
