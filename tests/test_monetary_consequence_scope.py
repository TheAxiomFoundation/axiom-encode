"""Closed source-consequence handling; source and non-selector passes stay intact."""

import pytest

from axiom_encode.harness import source_completeness as completeness

EQUALITY = (
    "the amount of the federal foreign non-business income tax credit you are entitled "
    "to deduct is equal to the foreign non-business\n"
    "tax you paid, your provincial or territorial foreign tax credit would be zero. "
    "As a result, you do not have to complete this form."
)
SEQUENCE = (
    'you were a resident of Ontario, calculate this amount by entering "0" on lines '
    "70 and 72 of Form ON428 and continue the\n"
    "calculation. The result from line 81 is your provincial or territorial tax "
    "otherwise payable. If you paid tax to more than one\n"
    'jurisdiction in 2025, calculate this amount by entering "0" on lines 43 and 45 '
    "in Part 4 of Section ON428MJ of Form T2203\n"
    "and continue the calculation. The amount from line 58 is your provincial or "
    "territorial tax otherwise payable."
)


@pytest.mark.parametrize("whitespace", [" ", "\n", "\t", "\r\n"])
def test_closed_equality_retains_amount_comparison(whitespace):
    body = EQUALITY.replace("\n", whitespace)
    original = body
    facts = completeness._without_worksheet_imperative_consequent(body)
    assert facts == (
        "the amount of the federal foreign non-business income tax credit you are "
        "entitled to deduct is equal to the foreign non-business tax you paid"
    )
    assert body == original
    assert completeness._source_conjunctive_fact_gates("If " + body) == ()


def test_complete_zero_line_sequence_preserves_both_antecedents():
    facts = completeness._without_worksheet_imperative_consequent(SEQUENCE)
    assert facts == (
        "you were a resident of Ontario\n"
        "If you paid tax to more than one jurisdiction in 2025"
    )
    gates = completeness._source_conjunctive_fact_gates("If " + SEQUENCE)
    # Existing gate vocabulary does not classify "paid" as a conjunctive
    # predicate. Preserve its factual text without changing that independent API.
    assert gates == completeness._source_conjunctive_fact_gates("If " + facts)
    segments = completeness._source_gate_split_conjunctive_conditions(facts)
    assert len(segments) == 2
    assert "Ontario" in segments[0][1]
    assert "paid tax to more than one jurisdiction" in segments[1][1]


@pytest.mark.parametrize(
    "suffix",
    [
        " Only if your spouse is disabled.",
        " Unless the claimant is eligible.",
        " And the claimant must be resident.",
        " Enter another amount.",
        " If income exceeds 100, do not claim the credit.",
        " (for disabled claimants)",
    ],
)
@pytest.mark.parametrize("body", [EQUALITY, SEQUENCE])
def test_additional_restrictive_or_unknown_tail_is_not_removed(body, suffix):
    text = body + suffix
    assert completeness._without_worksheet_imperative_consequent(text) == text


@pytest.mark.parametrize(
    "body",
    [
        EQUALITY.replace("would be zero.", "would be zero only if eligible."),
        EQUALITY.replace(
            "you are entitled to deduct", "your disabled spouse may deduct"
        ),
        EQUALITY.replace("tax you paid", "tax you paid if eligible"),
        SEQUENCE.replace('"0"', '"42"'),
        SEQUENCE.replace('"0"', "“0”"),
        SEQUENCE.replace('"0"', '"0'),
        SEQUENCE.replace("lines 70 and 72", "lines 70 and 72 ("),
        SEQUENCE.replace("lines 70 and 72", "lines 70 and 72 [)"),
        SEQUENCE.replace(
            "and continue the calculation", "and continue only if eligible"
        ),
        SEQUENCE.replace("Section ON428MJ", "Section ON428MJ for disabled spouses"),
        SEQUENCE.replace("jurisdiction in 2025", "jurisdiction unless exempt in 2025"),
        SEQUENCE.replace("If you paid", "If your disabled spouse paid"),
        '"' + SEQUENCE + '"',
        "(" + EQUALITY,
        EQUALITY + ")",
    ],
)
def test_malformed_or_person_restricted_operations_fail_closed(body):
    assert completeness._without_worksheet_imperative_consequent(body) == body


def test_real_ontario_threshold_condition_remains_unchanged():
    body = (
        "you were a resident of Ontario at the end of the year, follow the "
        "instructions that apply to your situation:\n"
        "– If the total non-business income taxes you paid to all foreign "
        "countries is $200 or less:"
    )
    assert completeness._without_worksheet_imperative_consequent(body) == body
    assert len(completeness._source_conjunctive_fact_gates("If " + body)) == 2


def test_existing_single_operation_contract_is_retained():
    body = SEQUENCE.split(" If you paid", 1)[0]
    assert completeness._without_worksheet_imperative_consequent(body) == (
        "you were a resident of Ontario"
    )
