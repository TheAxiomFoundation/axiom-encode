"""Numeric-only printed locators; actual admitted bodies remain immutable."""

from pathlib import Path

import pytest

from axiom_encode.harness import validator_pipeline as v

FIXTURES = Path(__file__).parent / "fixtures/form-numeric-locators"
EXTRACTORS = [
    v.extract_typed_numeric_inventory_occurrences_from_text,
    v.extract_typed_numeric_occurrences_from_text,
]


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize("extract", EXTRACTORS)
@pytest.mark.parametrize(
    "text",
    [
        "box 14 of all T4 slips",
        "box 52 of all T4 slips and box 034 of all T4A slips",
        "box 034 of all T4A slips",
        "box 18 of the T4A(OAS) slip",
        "box 20 of the T4A(P) slip",
        "box 12 of all RC62 slips",
        "boxes 39, 41, 91 and 92 of all T4 slips",
    ],
)
def test_complete_slip_references(profile, extract, text):
    source = text + ". Deduct 1074 dollars."
    found = extract(source, profile=profile)
    assert [x.value for x in found] == [1074]
    assert all(source[x.start : x.end] == x.raw for x in found)


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize("extract", EXTRACTORS)
@pytest.mark.parametrize(
    "text",
    [
        "box 14",
        "boxes 14 and 16",
        "box 14 dollars of all T4 slips",
        "box 14 of all unknown slips",
        "box 14 of all T4",
        "box 14 of all T4 slips dollars",
        "box 14 of all T4 slips + 2",
        "box 14 of all T4 slips years",
        "box 14 of all T4 slips kg",
        "boxes 14, 14 of all T4 slips",
    ],
)
def test_incomplete_or_value_bearing_referral_keeps_amount(profile, extract, text):
    found = extract(text, profile=profile)
    assert found
    assert all(text[x.start : x.end] == x.raw for x in found)


def test_actual_three_bodies_inventory():
    expected = {2: set(), 3: {1074, 82125, 93454}, 4: set()}
    for step, values in expected.items():
        text = (FIXTURES / f"step-{step}.txt").read_text()
        actual = {
            x.value
            for x in v.extract_typed_numeric_inventory_occurrences_from_text(text)
        }
        assert actual == values


def test_ambiguous_decimal_box_does_not_create_locator():
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    for text in ["box 14.5 of all T4 slips", "box 14,5 of all T4 slips"]:
        assert form_numeric_locator_spans(text) == ()


@pytest.mark.parametrize("step", [2, 3])
def test_exact_span_mask_preserves_prose_operators_and_offsets(step):
    from axiom_encode.form_numeric_structure import (
        form_numeric_locator_spans,
        mask_form_numeric_locators,
    )

    text = (FIXTURES / f"step-{step}.txt").read_text()
    spans = form_numeric_locator_spans(text)
    masked = mask_form_numeric_locators(text)
    assert len(masked) == len(text)
    positions = {i for start, end in spans for i in range(start, end)}
    assert all(
        text[i].isdigit() and masked[i] == " "
        if i in positions
        else text[i] == masked[i]
        for i in range(len(text))
    )
    assert text.count("Add lines") == masked.count("Add lines")
    assert text.count("if negative") == masked.count("if negative")
    assert text.count("Federal Worksheet") == masked.count("Federal Worksheet")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_footer",
        "wrong_page",
        "missing_header",
        "duplicate_row",
        "missing_row",
        "amount_footer",
        "quoted_footer",
    ],
)
def test_uncorroborated_footer_retains_code(mutation):
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    text = (FIXTURES / "step-3.txt").read_text()
    footer = "5000-R (Page 5 of 8)"
    if mutation == "duplicate_footer":
        text = text + footer + "\n"
    elif mutation == "wrong_page":
        text = text.replace(footer, "5000-R (Page 9 of 8)")
    elif mutation == "missing_header":
        text = text.replace("Step 3 - Net income", "Some income")
    elif mutation == "duplicate_row":
        text = text.replace("Line 37 and 20805:", "Line 36 and 20805:")
    elif mutation == "missing_row":
        text = text.replace("Line 37 and 20805:", "Not a printed row:")
    elif mutation == "amount_footer":
        text = text.replace(footer, footer + " dollars")
    else:
        text = text.replace(footer, '"' + footer + '"')
    spans = form_numeric_locator_spans(text)
    starts = [i for i in range(len(text)) if text.startswith("5000-R", i)]
    assert starts and all(not any(a <= i < b for a, b in spans) for i in starts)


@pytest.mark.parametrize(
    "mutation", ["duplicate_field", "duplicate_row", "amount_suffix", "missing_caption"]
)
def test_coordinated_row_refuses_ambiguous_grid(mutation):
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    text = (FIXTURES / "step-3.txt").read_text()
    if mutation == "duplicate_field":
        text = text.replace("Line 37 and 20805:", "Line 37 and 20800:")
    elif mutation == "duplicate_row":
        text = text.replace("Line 37 and 20805:", "Line 36 and 20805:")
    elif mutation == "amount_suffix":
        text = text.replace("Line 36 and 20800:", "Line 36 and 20800 dollars:")
    else:
        text = text.replace(
            "Line 36 and 20800: RRSP deduction (complete Schedule 7 and attach receipts) ^",
            "Line 36 and 20800:",
        )
    i = text.index("20800")
    spans = form_numeric_locator_spans(text)
    assert not any(a <= i < b for a, b in spans)


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize("extract", EXTRACTORS)
def test_actual_layout_preserves_adjacent_amount_offsets(profile, extract):
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    text = (FIXTURES / "step-3.txt").read_text()
    amount = "765,25" if profile == "da-DK" else "765.25"
    text = text.replace("RRSP deduction (", f"RRSP deduction {amount} dollars (")
    start = text.index(amount)
    assert all(not (a <= start < b) for a, b in form_numeric_locator_spans(text))
    assert any(
        x.value == 765.25
        and x.raw == amount
        and (x.start, x.end) == (start, start + len(amount))
        for x in extract(text, profile=profile)
    )


@pytest.mark.parametrize(
    "text",
    ['"box 14 of all T4 slips"', "“box 14 of all T4 slips”", '"box 14 of all T4 slips'],
)
def test_quoted_or_unbalanced_referral_is_not_structural(text):
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    assert form_numeric_locator_spans(text) == ()


@pytest.mark.parametrize(
    "tail",
    ["and box 034", "and box 034 dollars", "and box 034 of all T4A slips dollars"],
)
def test_coordinated_referral_requires_complete_second_member(tail):
    from axiom_encode.form_numeric_structure import form_numeric_locator_spans

    text = "box 52 of all T4 slips " + tail
    assert form_numeric_locator_spans(text) == ()
