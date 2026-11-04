"""Binary instructions require source computations without inventing operands."""

import json
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as sc


@pytest.mark.parametrize(
    "text,kind,left,right",
    [
        ("Add income to allowance.", "add", "income", "allowance"),
        ("Subtract expenses from income.", "subtract", "income", "expenses"),
        ("Multiply income by 2.", "multiply", "income", "2"),
        ("Divide income by 2.", "divide", "income", "2"),
        ("Subtract line 4 from line 1.", "subtract", "line 1", "line 4"),
        (
            "Subtract the amount on line 2 from the amount on line 1.",
            "subtract",
            "the amount on line 1",
            "the amount on line 2",
        ),
        ("Multiply 2.5 by 1.25.", "multiply", "2.5", "1.25"),
        ("Add $1,000 to income.", "add", "$1,000", "income"),
        ("Divide the income by the divisor.", "divide", "the income", "the divisor"),
    ],
)
def test_binary_instruction_has_owned_ordered_operand_spans(text, kind, left, right):
    operations = sc._source_imperative_arithmetic_operations(text)
    assert len(operations) == 1
    operation = operations[0]
    assert operation.kind == kind
    assert text[slice(*operation.left_operand)] == left
    assert text[slice(*operation.right_operand)] == right
    assert sc.source_states_explicit_computation(text)
    assert sc._source_states_nonrounding_computation(text)
    assert kind in sc._formula_operation_kinds(text)
    branches = sc._source_formula_branches(
        text, branches=(), active_branches=(), deferred_paths=set()
    )
    assert branches
    assert any(branch.text == text for branch in branches)


@pytest.mark.parametrize(
    "verb,connector,kind",
    [
        ("Add", "to", "add"),
        ("Subtract", "from", "subtract"),
        ("Multiply", "by", "multiply"),
        ("Divide", "by", "divide"),
    ],
)
def test_semicolon_consumer_retains_actual_arithmetic_clause(verb, connector, kind):
    instruction = f"{verb} income {connector} allowance."
    text = "Go to example.ca/info; " + instruction
    branches = sc._source_formula_branches(
        text, branches=(), active_branches=(), deferred_paths=set()
    )
    branch = next(branch for branch in branches if branch.text == instruction)
    assert kind in sc._formula_operation_kinds(branch.text)
    assert text[branch.start : branch.end] == instruction
    # The prior false URL obligation has not been removed by this change.
    assert sc.source_states_explicit_computation("Go to example.ca/info")


@pytest.mark.parametrize(
    "text",
    [
        "Add your name to the form.",
        "Add your address to the mailing list.",
        "Divide the document by section.",
        "Multiply applications by department.",
        "Add telephone number to contact record.",
        "Add telephone number to income.",
        "Divide account number by 2.",
        "Add income to tax number.",
        "Divide benefit number by 2.",
        "Divide the whole number by 2.",
        'The title is "Divide income by 2".',
        'The title is "Instructions; Divide income by 2."',
        "Do not divide income by 2.",
        "Do not:\nDivide income by 2.",
        "If resident:\nDivide income by 2.",
        "Subtract expenses by income.",
        "Add form TP-1 to income.",
        "Add line 0 to income.",
        "Add line 12a to income.",
        "Add example.ca/info to income.",
        "Divide income by 2 unless disabled.",
        "If resident, divide income by 2.",
        "Subtract expenses from income, except for persons aged 65.",
        "Add income and expenses to allowance.",
        "Add lines 1 and 2.",
        "Deduct expenses from income.",
        "Halve the income.",
        "Total the two amounts.",
        "Apply a rate of 3%.",
    ],
)
def test_administrative_ambiguous_or_out_of_scope_forms_are_not_newly_admitted(text):
    assert not sc._source_imperative_arithmetic_operations(text)


def test_independent_sentence_does_not_get_swallowed_by_invalid_instruction():
    text = "Add your name. Subtract expenses from income."
    operations = sc._source_imperative_arithmetic_operations(text)
    assert len(operations) == 1
    assert operations[0].kind == "subtract"
    assert (
        text[operations[0].start : operations[0].end] == "Subtract expenses from income"
    )


def test_multiple_raw_spans_preserve_distinct_operations():
    text = "Subtract expenses from income; Divide the amount by 2."
    operations = sc._source_imperative_arithmetic_operations(text)
    assert [operation.kind for operation in operations] == ["subtract", "divide"]
    assert sc._formula_operation_kinds(text) == {"subtract", "divide"}


def test_direct_formula_ast_keeps_original_operation_contract():
    assert sc._formula_operation_kinds("income - expenses / divisor") == {
        "subtract",
        "divide",
    }


def test_header_typography_is_not_exempted():
    assert sc.source_states_explicit_computation("14 2025 – GUIDE-V Contact us.")
    assert "subtract" in sc._formula_operation_kinds("14 2025 – GUIDE-V Contact us.")


def test_explicit_count_operand_remains_numeric():
    text = "Multiply income by the number of qualifying children."
    operations = sc._source_imperative_arithmetic_operations(text)
    assert len(operations) == 1
    assert operations[0].kind == "multiply"
    assert (
        text[slice(*operations[0].right_operand)] == "the number of qualifying children"
    )


ACTUAL_INSTRUCTIONS = json.loads(
    (
        Path(__file__).parent
        / "fixtures/source_completeness/quebec_binary_instructions.json"
    ).read_text()
)["samples"]


@pytest.mark.parametrize("sample", ACTUAL_INSTRUCTIONS)
def test_authenticated_instructions_have_correct_operations_without_source_rewriting(
    sample,
):
    text = sample["text"]
    operations = sc._source_imperative_arithmetic_operations(text)
    actual = [
        {
            "kind": op.kind,
            "left": text[slice(*op.left_operand)],
            "right": text[slice(*op.right_operand)],
        }
        for op in operations
    ]
    assert actual == sample["new_descriptors"]
    assert sc.source_states_explicit_computation(text)
    assert {op.kind for op in operations} <= sc._formula_operation_kinds(text)
    branches = sc._source_formula_branches(
        text, branches=(), active_branches=(), deferred_paths=set()
    )
    assert branches
    for branch in branches:
        assert text[branch.start : branch.end] == branch.text


def test_independent_sentence_break_permits_newline_but_not_arbitrary_context():
    text = "Subtract line 4 from line 1.\nDivide income by 2."
    operations = sc._source_imperative_arithmetic_operations(text)
    assert [op.kind for op in operations] == ["subtract", "divide"]
    assert text[operations[1].start : operations[1].end] == "Divide income by 2"
