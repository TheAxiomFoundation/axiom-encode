import pytest

from axiom_encode.harness.source_completeness import (
    _concessive_form_completion_condition,
    _source_conjunctive_fact_gates,
)

PREFIX = (
    "Even if you are not claiming a capital gains deduction in 2025, "
    "you should still complete this form if "
)


def test_complete_frame_preserves_actual_condition_and_soft_wrap():
    condition = "you had any investment\nincome or expenses in 2025"
    assert _concessive_form_completion_condition(PREFIX + condition + ".") == (
        "If you had any investment income or expenses in 2025."
    )
    assert _source_conjunctive_fact_gates(PREFIX + condition + ".") == ()


@pytest.mark.parametrize(
    "tail",
    [
        "you had investment income and you are resident",
        "you are resident and you are disabled",
        "you had income only if you are approved",
        "you had income unless you are disqualified",
        "you had income provided that you are resident",
        "you had income\nand your spouse is disabled",
    ],
)
def test_condition_and_additional_requirements_are_preserved(tail):
    assert (
        _concessive_form_completion_condition(PREFIX + tail + ".") == "If " + tail + "."
    )
    assert _source_conjunctive_fact_gates(PREFIX + tail + ".") == (
        _source_conjunctive_fact_gates(
            "If " + tail + ".", care_source_text=PREFIX + tail + "."
        )
    )


@pytest.mark.parametrize(
    "text",
    [
        PREFIX.replace("Even if", "Only if") + "you had income.",
        PREFIX.replace("Even if", "Unless") + "you had income.",
        PREFIX.replace("you should still", "your spouse should still")
        + "you had income.",
        PREFIX.replace("complete this form", "qualify for this deduction")
        + "you had income.",
        PREFIX + "your spouse had income.",
        PREFIX + "you had income",
        '"' + PREFIX + 'you had income."',
        PREFIX + "you meet these conditions:\n• income\n• residency.",
        PREFIX + "you had income. You must also be approved.",
    ],
)
def test_other_or_incomplete_frames_fall_through(text):
    assert _concessive_form_completion_condition(text) is None
