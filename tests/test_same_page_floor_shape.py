"""Local floor syntax cannot collapse a conditional's false zero into proof."""

import pytest

from axiom_encode.harness.actual_floor_evidence import _same_page_floor_shape


def test_ordered_local_subtraction_and_named_selected_gate():
    assert _same_page_floor_shape("max(0, capacity - deductions)") == (
        None,
        "capacity",
        "deductions",
    )
    assert _same_page_floor_shape(
        "if eligible:\n  max(0, total - adjusted)\nelse:\n  0"
    ) == ("eligible", "total", "adjusted")
    assert _same_page_floor_shape("max(0, deductions - capacity)") == (
        None,
        "deductions",
        "capacity",
    )  # Binding must compare this order to the source, not sort names.


@pytest.mark.parametrize(
    "formula",
    [
        "max(False, capacity - deductions)",
        "max(0.0, capacity - deductions)",
        "min(0, capacity - deductions)",
        "max(0, capacity - capacity)",
        "max(0, capacity + deductions)",
        "max(0, 4 - deductions)",
        "if eligible:\n  0\nelse:\n  max(0, total - adjusted)",
        "if eligible:\n  max(0, total - adjusted)\nelse:\n  False",
        "if eligible:\n  max(0, total - adjusted)\nelse:\n  0.0",
        "if eligible and another:\n  max(0, total - adjusted)\nelse:\n  0",
        "if eligible:\n  max(0, total - adjusted)",
        "if eligible:\n  max(0, total - adjusted)\n  unrelated\nelse:\n  0",
        "if eligible:\n  max(0, total - adjusted)\nelse:\n  0\nunrelated",
    ],
)
def test_other_shapes_do_not_match(formula):
    assert _same_page_floor_shape(formula) is None
