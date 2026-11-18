"""A hypothetical return instruction must not invent additional factual gates."""

import pytest

from axiom_encode.harness import source_completeness as sc

ANTECEDENT = " your spouse or common-law partner is not filing a return"
COMMAND = (
    ", use the amounts they would have entered on their return, schedules,\n"
    "and worksheets as if they were filing a return, and attach their information slips."
)


def test_exact_source_nonfiler_instruction_is_not_an_extra_fact():
    body = ANTECEDENT + COMMAND
    assert sc._without_worksheet_imperative_consequent(body) == ANTECEDENT
    assert sc._source_conjunctive_fact_gates("If" + body) == ()


def test_all_same_actor_antecedent_conjuncts_preserved_byte_exact():
    antecedent = (
        ANTECEDENT + " and your spouse or common-law partner is disabled and dependent"
    )
    expected = sc._source_conjunctive_fact_gates("If" + antecedent)
    assert len(expected) == 3
    assert (
        sc._without_worksheet_imperative_consequent(antecedent + COMMAND) == antecedent
    )
    assert sc._source_conjunctive_fact_gates("If" + antecedent + COMMAND) == expected


@pytest.mark.parametrize(
    "body",
    [
        ANTECEDENT
        + COMMAND.replace("use the amounts", "use the amounts only if resident"),
        ANTECEDENT
        + COMMAND.replace(
            "attach their information slips",
            "attach their information slips if requested",
        ),
        ANTECEDENT
        + COMMAND.replace("were filing a return", "were filing a return and eligible"),
        ANTECEDENT + COMMAND.replace("as if they were", "as if your child were"),
        ANTECEDENT + " and your child is dependent" + COMMAND,
        ANTECEDENT + ' and the notice says "eligible"' + COMMAND,
        ANTECEDENT
        + COMMAND.replace(
            "as if they were filing a return", 'as if "they were filing a return"'
        ),
        ANTECEDENT
        + COMMAND.replace(
            "as if they were filing a return",
            "as if they were filing a return (unless resident)",
        ),
        ANTECEDENT
        + COMMAND.replace(
            "as if they were filing a return",
            "as if they were filing a return (provided that resident)",
        ),
        ANTECEDENT + COMMAND + " If your child is dependent, claim the amount.",
        ANTECEDENT + COMMAND + " Your child must be dependent.",
        " your spouse is filing a return, use their actual amounts. If"
        + ANTECEDENT
        + COMMAND,
        "not (" + ANTECEDENT + COMMAND + ")",
        "unless" + ANTECEDENT + COMMAND,
        '"' + ANTECEDENT + COMMAND + '"',
        ANTECEDENT
        + COMMAND.replace("slips.", "slips after the spouse becomes resident."),
    ],
)
def test_unresolved_or_additional_real_conditions_are_not_trimmed(body):
    assert sc._nonfiler_return_instruction_antecedent(body) is None
    assert sc._without_worksheet_imperative_consequent(body) == body


def test_ordinary_existing_imperative_behavior_is_unchanged():
    body = " the applicant is disabled and the spouse is resident"
    assert (
        sc._without_worksheet_imperative_consequent(
            body + ", enter the amount and carry the result."
        )
        == body
    )


def test_document_duty_source_clause_inventory_is_not_rewritten():
    source = "If" + ANTECEDENT + COMMAND
    clauses = list(sc._source_clause_spans(source, branches=()))
    assert clauses
    assert any("attach their information slips" in text for _, _, text in clauses)
    assert all(source[start:end] == text for start, end, text in clauses)


@pytest.mark.parametrize(
    "first,second",
    [
        ("your spouse", "your common-law partner"),
        ("your common-law partner", "your spouse"),
        ("your spouse or common-law partner", "your spouse"),
        ("your spouse", "your spouse or common-law partner"),
        ("your spouse or common-law partner", "your common-law partner"),
        ("your common-law partner", "your spouse or common-law partner"),
    ],
)
def test_distinct_explicit_actor_phrases_are_not_equated(first, second):
    body = f" {first} is not filing a return and {second} is disabled" + COMMAND
    assert sc._nonfiler_return_instruction_antecedent(body) is None
    assert sc._without_worksheet_imperative_consequent(body) == body


@pytest.mark.parametrize(
    "actor",
    ["your spouse", "your common-law partner", "your spouse or common-law partner"],
)
def test_identical_explicit_actor_phrase_is_preserved(actor):
    antecedent = f" {actor} is not filing a return and {actor} is disabled"
    assert (
        sc._nonfiler_return_instruction_antecedent(antecedent + COMMAND) == antecedent
    )
