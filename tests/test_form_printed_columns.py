from __future__ import annotations

import pytest

from axiom_encode.form_source_structure import (
    _schedule_l_printed_columns,
    mask_form_display,
)
from axiom_encode.harness import source_completeness as s
from tests.test_partnership_reporting_evidence import material


def test_exact_closed_layout_masks_only_redundant_glyphs():
    _, _, source = material()
    result = _schedule_l_printed_columns(source)
    assert result is not None
    masked = mask_form_display(source)
    assert len(masked) == len(source)
    assert [i for i, x in enumerate(masked) if x == "\n"] == [
        i for i, x in enumerate(source) if x == "\n"
    ]
    for quote in [
        "Add lines 22 through 26.1.",
        "Add lines 27 through 30.",
        "Carry the result to line 164 of your return.",
        "or an intermediate resource (line 8 of form LM-53-V)",
    ]:
        assert quote in masked
    start = source.index("Carry the result")
    end = source.index("Enclose this schedule", start)
    assert s.source_states_explicit_computation(source[start:end])
    assert not s.source_states_explicit_computation(masked[start:end])
    assert all(set(source[a:b]) <= set("0123456789.+=") for a, b in result.spans)


@pytest.mark.parametrize(
    "old,new",
    [
        ("Schedule L 2025", "Schedule L 2024"),
        ("L-V (2025-12)", "L-V (2024-12)"),
        ("Gross income\nNet income", "Gross income"),
        ("Farming\n13", "Farming\n14"),
        ("Farming\n13", "Farming\n13\n13"),
        ("Farming\n13", "Farming\n13 dollars"),
        ("\n+ 26.1\n", "\n+ 26.1 percent\n"),
        ("\n23\n", "\n23\n23\n"),
        ("\n40\n", "\n40 dollars\n"),
        ("line 8 of form LM-53-V", "line 9 of form LM-53-V"),
        (
            "Carry the result to line 164 of your return.",
            "Carry the result to line 164 of your return up to 100 dollars.",
        ),
        ("Add lines 27 through 30.", "Add lines 27 through 30 and 100 dollars."),
        ("Net business income = 34", "Net business income = 34 + 100"),
        ("\nL", "\nQ"),
    ],
)
def test_ambiguous_or_value_bearing_layout_refuses(old, new):
    _, _, source = material()
    changed = source.replace(old, new)
    assert changed != source
    assert _schedule_l_printed_columns(changed) is None


def test_quoted_or_duplicated_layout_is_not_metadata():
    _, _, source = material()
    assert _schedule_l_printed_columns('"' + source + '"') is None
    assert _schedule_l_printed_columns(source + source) is None


def test_combined_actual_diagnostic_has_no_normal_source_findings():
    from tests.test_partnership_reporting_evidence import issues

    doc, cases, source = material()
    assert issues(doc, cases, source) == ()


@pytest.mark.parametrize(
    "old,new",
    [
        ("\n+ 28\n", "\n+ 28\n+ 100 dollars\n"),
        ("\n23\n", "\n-23\n"),
        ("\n40\n", "\n40\n40\n"),
        ("T P -1. D. L-V", "A B -2. D. L-V"),
    ],
)
def test_extra_or_changed_numeric_rows_preserve_unresolved_layout(old, new):
    _, _, source = material()
    changed = source.replace(old, new)
    assert changed != source
    assert _schedule_l_printed_columns(changed) is None
