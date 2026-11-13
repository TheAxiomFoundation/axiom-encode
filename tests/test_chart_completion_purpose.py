import pytest

from axiom_encode.harness.source_completeness import (
    _chart_completion_purpose_antecedent,
    _source_conjunctive_fact_gates,
    _without_worksheet_imperative_consequent,
)

ANTECEDENT = (
    ", in 2025, you have capital gains other than from the disposition of "
    "qualified farm or fishing property or qualified small\nbusiness corporation shares"
)
COMMAND = (
    ", first complete Chart A on page 4 of this form to determine if you have additional "
    "investment\nincome to include when you calculate your CNIL."
)


def test_closed_computational_purpose_is_not_a_factual_prerequisite():
    expected = ANTECEDENT.replace("small\nbusiness", "small business")
    assert _chart_completion_purpose_antecedent(ANTECEDENT + COMMAND) == expected
    assert _source_conjunctive_fact_gates("If" + ANTECEDENT + COMMAND) == ()
    assert "other than" in expected
    assert " or " in expected


def test_real_antecedent_conjunct_remains_at_caller():
    body = ANTECEDENT + " and you are resident" + COMMAND
    gates = _source_conjunctive_fact_gates("If" + body)
    assert len(gates) == 2
    assert any("resident" in predicates for _, predicates in gates)


@pytest.mark.parametrize(
    "extra",
    [
        " only if you are approved",
        " unless you are disqualified",
        " provided that you are resident",
        " when you are approved",
    ],
)
def test_extra_conditional_introducers_use_original_analysis(extra):
    body = ANTECEDENT + extra + COMMAND
    assert _chart_completion_purpose_antecedent(body) is None
    assert _without_worksheet_imperative_consequent(body) == body


def test_only_if_approval_gate_is_not_lost():
    gates = _source_conjunctive_fact_gates(
        "If" + ANTECEDENT + " only if you are approved" + COMMAND
    )
    assert any("approved" in predicates for _, predicates in gates)


@pytest.mark.parametrize(
    "command",
    [
        COMMAND.replace("to determine if", "only if approved to determine if"),
        COMMAND.replace("your CNIL.", "your CNIL, provided that you are resident."),
        COMMAND.replace("you have additional", "your spouse has additional"),
        COMMAND.replace("first complete", "claim a deduction using"),
        COMMAND.replace("investment\nincome", "eligibility"),
        COMMAND[:-1],
        COMMAND + " You must be resident.",
    ],
)
def test_changed_or_incomplete_commands_fall_through(command):
    assert _chart_completion_purpose_antecedent(ANTECEDENT + command) is None


@pytest.mark.parametrize("prefix", ['"', "“", "● "])
def test_quoted_or_list_context_is_not_flattened(prefix):
    assert _chart_completion_purpose_antecedent(prefix + ANTECEDENT + COMMAND) is None
