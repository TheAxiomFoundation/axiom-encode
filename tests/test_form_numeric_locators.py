"""Numeric inventory must distinguish complete locators from actual amounts."""

import pytest

from axiom_encode.harness import validator_pipeline as v


def values(text):
    return [
        float(x.value)
        for x in v.extract_typed_numeric_inventory_occurrences_from_text(text)
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Complete form TP-21.4.39-V. Tax is 21.4 dollars.", [21.4]),
        ("Enter 100 dollars on line 16.1.", [100]),
        ("Add lines 22 through 26.1.", []),
        ("Enter the amount from box 14-7 of your RL-15 slip on line 16.1.", []),
        ("Enter the amount from box 1-12 of the RL-15 slip on line 26.1.", []),
    ],
)
def test_complete_locators_do_not_emit_fractional_fragments(text, expected):
    assert values(text) == expected


@pytest.mark.parametrize(
    "text,amount",
    [
        ("The amount is 21.4 dollars.", 21.4),
        ("line 16.1 dollars", 16.1),
        ("line 16.1 percent", 0.161),
        ("lines 22 through 26.1 percent", 0.261),
        ("line 16.1 CAD", 16.1),
        ("line 16.1 + 2", 16.1),
        ("The amount is -26.1.", 26.1),
        ("Subtract 14 - 7.", 14),
        ("box 14-7 without a slip", 14),
        ("box 14-7 of your unrelated slip", 14),
        ("box 14-7 dollars of your RL-15 slip", 14),
        ("form TP-21.4.39-V. Tax is 100 percent.", 1),
    ],
)
def test_financial_and_ambiguous_tokens_remain(text, amount):
    assert amount in [abs(x) for x in values(text)]


@pytest.mark.parametrize(
    "extractor",
    [
        v.extract_typed_numeric_inventory_occurrences_from_text,
        v.extract_typed_numeric_occurrences_from_text,
    ],
)
@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize("unit", ["years", "months", "days", "kg", "kilograms"])
def test_measurement_suffix_is_never_a_locator(extractor, profile, unit):
    token = "16,1" if profile == "da-DK" else "16.1"
    text = f"line {token} {unit}"
    found = extractor(text, profile=profile)
    assert any(
        x.value == 16.1 and x.raw == token and (x.start, x.end) == (5, 5 + len(token))
        for x in found
    )


@pytest.mark.parametrize(
    "extractor",
    [
        v.extract_typed_numeric_inventory_occurrences_from_text,
        v.extract_typed_numeric_occurrences_from_text,
    ],
)
def test_comma_decimal_range_amount_retains_exact_offsets(extractor):
    text = "lines 22 through 26,1 dollars"
    assert any(
        x.value == 26.1 and x.raw == "26,1" and (x.start, x.end) == (17, 21)
        for x in extractor(text, profile="da-DK")
    )
