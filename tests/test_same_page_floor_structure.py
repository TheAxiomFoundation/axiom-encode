"""Closed source operands remain descriptive, ordered and tied to raw text."""

from pathlib import Path

import pytest

from axiom_encode.printed_source_rows import recognize_printed_rows

FIXTURES = Path(__file__).parent / "fixtures"
SOURCE = (FIXTURES / "detached_schedule2_source.txt").read_text()


def test_local_floor_operands_have_exact_separate_raw_owners():
    structure = recognize_printed_rows(SOURCE, allow_detached=True)
    assert structure is not None
    first, final = structure.same_page_floors
    assert (first.target, first.left, first.right) == (12, 7, 11)
    assert (final.target, final.left, final.right) == (13, 6, 12)
    assert SOURCE[slice(*first.operation_span)].startswith(
        "Spouse's or common-law partner's adjusted taxable income:\nLine 7 minus line 11"
    )
    assert SOURCE[slice(*final.operation_span)].startswith("Line 6 minus line 12")
    assert [SOURCE[slice(*span)] for span in final.operand_spans[0]] == [
        "=\n\n6",
        "Add lines 1 to 5.",
    ]
    assert first.operation_span == final.operand_spans[1][0]
    assert [SOURCE[slice(*span)] for span in first.operand_spans[1]] == [
        "Add lines 8 to 10."
    ]
    assert "57,375" in SOURCE[slice(*first.operand_spans[0][0])]


@pytest.mark.parametrize(
    "before,after",
    [
        ("Line 7 minus line 11", "Line 11 minus line 7"),
        ("Line 6 minus line 12", "Line 12 minus line 6"),
        ("Add lines 8 to 10.", "Add lines 8 to 11."),
        ("Add lines 1 to 5.", "Add lines 1 to 6."),
        ('if negative, enter "0"', 'if negative, enter "1"'),
        ("under 18 years", "under 19 years"),
    ],
)
def test_unknown_source_cannot_gain_floor_descriptors(before, after):
    assert (
        recognize_printed_rows(SOURCE.replace(before, after), allow_detached=True)
        is None
    )


def test_nonlocal_layout_and_default_call_have_no_new_route():
    assert recognize_printed_rows(SOURCE) is None
    step = (FIXTURES / "form-numeric-locators/step-3.txt").read_text()
    structure = recognize_printed_rows(step)
    assert structure is not None
    assert structure.same_page_floors == ()
