"""Document instructions do not assert that the expenses' family is eligible."""

import pytest

from axiom_encode.harness import validator_pipeline as v

CLAUSE = (
    "If applicable, complete the following table and include all of your family's\n"
    "eligible physical activity expenses."
)


@pytest.mark.parametrize(
    "text", [CLAUSE, CLAUSE.replace("\n", " "), CLAUSE.replace("\n", "\r\n")]
)
def test_instruction_is_not_family_eligibility_and_preserves_offsets(text):
    masked = v._mask_family_expense_table_instruction(text)
    start = text.index("family")
    assert masked == text[:start] + " " * 6 + text[start + 6 :]
    assert v._classify_source_scope(text) is None
    assert v._classify_source_unit_entity(text) is None


@pytest.mark.parametrize(
    "predicate",
    [
        "The family is eligible for benefits.",
        "The family income must be below the limit.",
        "The family resources must meet the test.",
    ],
)
@pytest.mark.parametrize("before", [False, True])
def test_independent_family_predicate_is_preserved(predicate, before):
    parts = [predicate, CLAUSE] if before else [CLAUSE, predicate]
    text = "\n\n".join(parts)
    assert predicate in v._mask_family_expense_table_instruction(text)
    assert v._classify_source_scope(text) == "unit"
    assert v._classify_source_unit_entity(text) == "family"


def test_person_and_mixed_scope_are_not_reassigned():
    person = "The individual is eligible for a credit."
    assert v._classify_source_scope(CLAUSE + "\n\n" + person) == "person"
    mixed = CLAUSE + "\n\n" + person + "\n\nThe family is eligible for benefits."
    assert v._classify_source_scope(mixed) is None


@pytest.mark.parametrize(
    "text",
    [
        "Do not: " + CLAUSE,
        "Do not:\n" + CLAUSE,
        "Do not:\n\n" + CLAUSE,
        "Do not:\r\n\r\n" + CLAUSE,
        CLAUSE + " unless another instruction applies.",
        CLAUSE + "\nunless another instruction applies.",
        CLAUSE + "\n\nunless another instruction applies.",
        CLAUSE + "\r\n\r\nExcept where otherwise stated.",
        CLAUSE + "\n\nProvided that the family is eligible.",
        CLAUSE.replace("include", "exclude"),
        CLAUSE.replace("eligible physical activity expenses", "income"),
        CLAUSE.replace("family's\n", "family's\n\n"),
        CLAUSE.removesuffix("."),
    ],
)
def test_qualified_negated_or_different_instruction_is_not_masked(text):
    assert v._mask_family_expense_table_instruction(text) == text


def test_original_source_remains_immutable():
    text = CLAUSE
    assert v._classify_source_scope(text) is None
    assert v._classify_source_unit_entity(text) is None
    assert text == CLAUSE
