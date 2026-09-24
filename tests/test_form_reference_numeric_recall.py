"""Printed form coordinates must not become financial scalar requirements."""

import pytest

from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
)


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize(
    "reference",
    [
        "lines 70 and 72",
        "lines 43 and 45",
        "lines 70, 72, and 81",
        "lines 70 or 72",
        "lines 70 through 72",
        "lines 70 to 72",
    ],
)
def test_form_line_coordinates_preserve_adjacent_operands(profile, reference):
    text = f"Enter 200 dollars on {reference} of Form ON428 and multiply by 15%."
    for extract in (
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
    ):
        occurrences = extract(text, profile=profile)
        assert {o.value for o in occurrences} == {
            200,
            0.15 if profile == "legacy" else 15,
        }
        assert all(text[o.start : o.end] == o.raw for o in occurrences)


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize("counter", ["Page 1 of 2", "Page 2 of 2", "PAGE 43 OF 45"])
def test_terminal_page_counter_is_not_a_scalar(profile, counter):
    text = f"Credit of 200 dollars. {counter}\nDeduct 15%."
    assert {
        o.value
        for o in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )
    } == {200, 0.15 if profile == "legacy" else 15}


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Credit lines 70 and 72 dollars.", 72),
        ("Credit lines 70 and 72%.", 72),
        ("Credit lines 70 and 72.5 dollars.", 72.5),
        ("Page 43 of 45 dollars is not a footer.", 45),
        ("Pay 43 dollars per page out of 45 dollars.", 43),
    ],
)
def test_substantive_numeric_suffix_is_preserved(text, expected, profile):
    if profile == "legacy" and "72%" in text:
        expected = 0.72
    if profile == "da-DK":
        text = text.replace("72.5", "72,5")
    assert expected in {
        o.value
        for o in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )
    }
