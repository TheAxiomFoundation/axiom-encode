import pytest

from axiom_encode.form_source_structure import form_display_spans, mask_form_display
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v

SOURCE = """Revenu Quebec Schedule M 2025 - Interest Paid on a Student Loan

T P -1. D. M -V (2025-12)

Complete this schedule if interest was paid in 2025 on a student loan.
Also complete this schedule if, for 2024, you calculated an amount.
If interest was paid after 1997, enter interest paid for 1998 through 2024 minus interest used.
Unused portion (line 62 of Schedule M for 2024).
Add lines 46 and 48.
Carry this amount to line 385.
Subtract line 60 from line 52.

46
48
52
60
62

M"""


def inventory(text):
    return v.extract_typed_numeric_inventory_occurrences_from_text(
        text, profile="legacy"
    )


def test_correlated_metadata_preserves_offsets_and_substantive_years():
    masked = mask_form_display(SOURCE)
    assert len(masked) == len(SOURCE)
    assert [i for i, c in enumerate(masked) if c == "\n"] == [
        i for i, c in enumerate(SOURCE) if c == "\n"
    ]
    assert len(form_display_spans(SOURCE)) == 4
    values = {o.value for o in inventory(SOURCE)}
    assert not values & {-1, 46, 48, 52, 60, 62}
    assert {1997, 2024, 2025} <= values
    for occurrence in inventory(SOURCE):
        assert SOURCE[occurrence.start : occurrence.end] == occurrence.raw


@pytest.mark.parametrize(
    "old,new",
    [
        ("Schedule M 2025 -", "Schedule M 2025 +"),
        ("T P -1. D. M -V", "T P -1"),
        ("(2025-12)", "(2024-12)"),
        ("(2025-12)", "(2025-13)"),
        ("D. M -V", "D. N -V"),
        ("\n\nT P", "\nIncome = A - 1\nT P"),
        ("T P -1. D. M -V (2025-12)", "T P -1. D. M -V (2025-12) + 5"),
    ],
)
def test_incomplete_or_inconsistent_header_is_unchanged(old, new):
    changed = SOURCE.replace(old, new)
    assert changed != SOURCE
    assert mask_form_display(changed) == changed


@pytest.mark.parametrize(
    "old,new",
    [
        ("\n\nM", "\n\nN"),
        ("\n48\n", "\n46\n"),
        ("\n48\n", "\n47\n"),
        ("\n48\n", "\n$48\n"),
        ("\n48\n", "\n-48\n"),
        ("\n48\n", "\n48%\n"),
        ("\n48\n", "\n48.5\n"),
        ("\n46\n48\n", "\n48\n46\n"),
        ("Add lines 46 and 48.", "Add the amounts."),
    ],
)
def test_ambiguous_tail_not_masked(old, new):
    changed = SOURCE.replace(old, new)
    assert changed != SOURCE
    assert len(form_display_spans(changed)) == 3
    assert mask_form_display(changed).endswith(changed[changed.rfind("\n\n") :])


@pytest.mark.parametrize(
    "text",
    ["A - 1", "amount = 2025 - income", "2025 - 2024", "-48 dollars", "46 48 52 60 62"],
)
def test_uncontextual_arithmetic_and_values_unchanged(text):
    assert mask_form_display(text) == text


def test_formula_inventory_no_title_computation_but_keeps_operations():
    branches = s._source_formula_branches(
        SOURCE,
        corpus_citation_path="ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest",
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )
    assert not any(b.start == 0 for b in branches)
    masked = mask_form_display(SOURCE)
    for clause in (
        "Add lines 46 and 48.",
        "Subtract line 60 from line 52.",
        "1998 through 2024 minus interest used.",
    ):
        assert clause in masked
    assert s._has_substantive_arithmetic_expression("A - 1")
    assert s._has_substantive_arithmetic_expression("2025 - 2024")


def test_duplicate_form_identifier_refuses_metadata():
    code = "T P -1. D. M -V (2025-12)"
    text = SOURCE.replace(code, code + "\n" + code)
    assert mask_form_display(text) == text


def test_substantive_same_numbers_elsewhere_survive():
    text = SOURCE.replace(
        "Complete this schedule",
        "Amount = A - 1. Tax is 46 dollars.\nComplete this schedule",
    )
    values = {o.value for o in inventory(text)}
    assert 1 in values and 46 in values  # subtraction has positive operand 1
    assert "A - 1" in mask_form_display(text)
