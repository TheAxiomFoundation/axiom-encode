"""A printed chart result cannot supply predicates to a prior instruction."""

import pytest

from axiom_encode.harness import source_completeness as c

HEADING = "105 WORK CHART – Correction of employment income Amount from box A 1 "
LABEL = "\b Correction of employment income = 3"
INSTRUCTION = (
    "If the result is negative, enter a minus sign before the amount "
    "and subtract it instead of adding it."
)


def test_completed_chart_label_does_not_create_factual_gate():
    assert c._source_conjunctive_fact_gates(INSTRUCTION) == ()
    assert c._source_conjunctive_fact_gates(HEADING + INSTRUCTION + " " + LABEL) == ()


@pytest.mark.parametrize(
    "body,context",
    [
        (INSTRUCTION + " Correction of employment income = 3", HEADING),
        (INSTRUCTION + " " + LABEL, "See the chart elsewhere."),
        (INSTRUCTION + "\b Unrelated income = 3", HEADING),
        ("If income is below $100.50 and disabled " + LABEL, HEADING),
        (INSTRUCTION + " " + LABEL + ". If disabled, also include income.", HEADING),
    ],
)
def test_uncorroborated_or_nonterminal_text_is_unchanged(body, context):
    assert c._without_completed_chart_result_label(body, context + body) == body


@pytest.mark.parametrize(
    "condition",
    [
        "If income is below $100.50 and the person is disabled, enter the amount.",
        "If income is below $100.50\nand the person is disabled, enter the amount.",
        "If the person is disabled and dependent on the claimant, enter the amount.",
        "If income is below $100.50 and the person is disabled, enter the amount unless the person is employed.",
        "Unless the person is employed and the spouse is disabled, enter the amount.",
    ],
)
def test_genuine_conditions_are_preserved(condition):
    expected = c._source_conjunctive_fact_gates(condition)
    assert expected
    assert (
        c._source_conjunctive_fact_gates(HEADING + condition + " " + LABEL) == expected
    )


def test_no_general_sentence_or_decimal_truncation():
    body = "If income exceeds $100.50, see e.g. the instructions. And the spouse is disabled."
    assert c._without_completed_chart_result_label(body, HEADING + body) == body


@pytest.mark.parametrize(
    "context",
    [
        "See " + HEADING,
        "The quotation is “" + HEADING + "”. ",
        '"\n' + HEADING + '\n" ',
        "“\n" + HEADING + "\n” ",
    ],
)
def test_reference_or_quoted_heading_cannot_corroborate_label(context):
    body = INSTRUCTION + " " + LABEL
    assert c._without_completed_chart_result_label(body, context + body) == body


def test_real_heading_after_publisher_introductory_sentence():
    body = INSTRUCTION + " " + LABEL
    assert (
        c._without_completed_chart_result_label(
            body, "Do not enclose these pages with your return. " + HEADING + body
        )
        == INSTRUCTION
    )
