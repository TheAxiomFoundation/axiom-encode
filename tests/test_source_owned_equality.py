"""A source-owned monetary equality controls its own zero consequence."""

import ast
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c
from axiom_encode.harness import validator_pipeline as v

SOURCE = (Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt").read_text()
CLAUSE = SOURCE[624:834]
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
CREDIT = "t2209_federal_non_business_foreign_tax_credit_line_3"
PAID = "t2209_non_business_income_tax_paid_to_foreign_country_line_1"


def fixture():
    rule = {
        "name": "principal",
        "kind": "derived",
        "dtype": "Money",
        "entity": "Person",
        "period": "Year",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": f"if {CREDIT} == {PAID}: 0 else: 1600",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": CLAUSE},
                    }
                ]
            }
        },
    }
    cases = [
        {
            "name": str(value),
            "period": "2025",
            "input": {CREDIT: value, PAID: 5000},
            "output": {"principal": 0 if value == 5000 else 1600},
        }
        for value in (3000, 5000)
    ]
    inputs = {
        name: {"name": name, "entity": "Person", "dtype": "Money", "period": "Year"}
        for name in (CREDIT, PAID)
    }
    return rule, cases, inputs


def run(rule, cases, inputs, source=SOURCE, clause=CLAUSE):
    start = source.index(clause)
    branch = c.SourceStructureBranch(
        (), "exception-clause", "root", clause, start, start + len(clause)
    )
    witnesses = c._toggled_formula_numeric_selectors(
        {"principal": rule},
        asserted_by_rule={"principal": cases},
        formula_environment={},
    )
    return c._exception_witnesses_for_branch(
        branch,
        source_text=source,
        corpus_citation_path=CITATION,
        principal_rules={"principal": rule},
        principal_rule_paths={"principal": {()}},
        asserted_by_rule={"principal": cases},
        toggled_exception_selectors=witnesses,
        extract_numeric_occurrences=v.extract_typed_numeric_occurrences_from_text,
        input_declarations=inputs,
        formula_environment={},
    )


def test_exact_actual_clause_has_equality_active_zero_witness():
    rule, cases, inputs = fixture()
    found = run(rule, cases, inputs)
    assert len(found) == 1
    witness = next(iter(found))
    assert witness.zeroes
    assert witness.relational_transitions == ((CREDIT, "==", PAID),)


def test_reversed_formula_operands_are_same_equality():
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = f"if {PAID} == {CREDIT}: 0 else: 1600"
    assert run(rule, cases, inputs)


def test_reversed_source_operands_are_same_equality():
    rule, cases, inputs = fixture()
    clause = "If the foreign non-business tax you paid is equal to the amount of the federal foreign non-business income tax credit you are entitled to deduct, your provincial or territorial foreign tax credit would be zero."
    source = SOURCE.replace(CLAUSE, clause)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    assert run(rule, cases, inputs, source, clause)


@pytest.mark.parametrize(
    "formula",
    [
        f"if {CREDIT} >= {PAID}: 0 else: 1600",
        f"if {CREDIT} <= {PAID}: 0 else: 1600",
        f"if {CREDIT} != {PAID}: 1600 else: 0",
        f"if not ({CREDIT} == {PAID}): 1600 else: 0",
        f"if {CREDIT} == {PAID} == other: 0 else: 1600",
        f"if {CREDIT} == {PAID} and eligible: 0 else: 1600",
        f"if True: 1600 else: if {CREDIT} == {PAID}: 0 else: 1600",
    ],
)
def test_unsupported_or_dead_comparison_is_not_positive_eq(formula):
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = formula
    for case in cases:
        case["input"].update(other=5000, eligible=True)
    assert not run(rule, cases, inputs)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_operand",
        "generic_alias",
        "missing_proof",
        "wrong_citation",
        "wrong_version",
        "wrong_assertion",
        "wrong_entity",
        "wrong_dtype",
        "different_period",
        "extra_input",
        "nonzero_equal",
        "zero_unequal",
    ],
)
def test_missing_identity_provenance_or_effect_rejects(mutation):
    rule, cases, inputs = fixture()
    atom = rule["metadata"]["proof"]["atoms"][0]
    if mutation in ("wrong_operand", "generic_alias"):
        name = "unrelated_credit" if mutation == "wrong_operand" else "credit"
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            CREDIT, name
        )
        inputs[name] = inputs.pop(CREDIT)
        inputs[name]["name"] = name
        for case in cases:
            case["input"][name] = case["input"].pop(CREDIT)
    elif mutation == "missing_proof":
        rule["metadata"]["proof"]["atoms"] = []
    elif mutation == "wrong_citation":
        atom["source"]["corpus_citation_path"] = "ca/policy/other"
    elif mutation == "wrong_version":
        atom["path"] = "versions[1].formula"
    elif mutation == "wrong_assertion":
        cases[1]["output"]["principal"] = 1
    elif mutation == "wrong_entity":
        inputs[CREDIT]["entity"] = "Household"
    elif mutation == "wrong_dtype":
        inputs[CREDIT]["dtype"] = "Boolean"
    elif mutation == "different_period":
        cases[1]["period"] = "2026"
    elif mutation == "extra_input":
        cases[1]["input"][PAID] = 6000
    elif mutation == "nonzero_equal":
        rule["versions"][0]["formula"] = f"if {CREDIT} == {PAID}: 1 else: 1600"
        cases[1]["output"]["principal"] = 1
    elif mutation == "zero_unequal":
        rule["versions"][0]["formula"] = f"if {CREDIT} == {PAID}: 1600 else: 0"
        cases[0]["output"]["principal"] = 0
        cases[1]["output"]["principal"] = 1600
    assert not run(rule, cases, inputs)


@pytest.mark.parametrize(
    "phrase",
    [
        "is not equal to",
        "is equal to or greater than",
        "is equal to or less than",
        "equals or exceeds",
    ],
)
def test_negated_or_inclusive_source_is_not_partial_equality(phrase):
    rule, cases, inputs = fixture()
    clause = CLAUSE.replace("is equal to", phrase)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    assert not run(rule, cases, inputs, SOURCE.replace(CLAUSE, clause), clause)
    assert not c._source_relational_exception_matches(
        clause, left_name=CREDIT, relation="==", right_name=PAID
    )


@pytest.mark.parametrize(
    "tail",
    [
        " Unless the claimant is disabled.",
        " If the credit exceeds the tax, the credit is zero.",
        " However, the credit is not zero.",
    ],
)
def test_additional_or_conflicting_effect_clause_cannot_lend_zero(tail):
    rule, cases, inputs = fixture()
    clause = CLAUSE + tail
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    assert not run(rule, cases, inputs, SOURCE.replace(CLAUSE, clause), clause)


@pytest.mark.parametrize("prefix,suffix", [('"', '"'), ('"', ""), ("“", "”")])
def test_quoted_or_malformed_source_rejects(prefix, suffix):
    rule, cases, inputs = fixture()
    assert not run(
        rule, cases, inputs, SOURCE.replace(CLAUSE, prefix + CLAUSE + suffix)
    )


def test_shared_relation_indices_preserve_mixed_effect_ownership():
    equality = c._collapse_text(CLAUSE).lower()
    text = "If credit exceeds liability, the refund is payable. " + equality
    indices = c._source_relational_exception_match_indices(
        text, left_name=CREDIT, relation="==", right_name=PAID
    )
    assert indices == (1,)
    assert c._source_relational_effect_text(text, relation_index=1).strip() == (
        "your provincial or territorial foreign tax credit would be zero."
    )
    assert "refund is payable" in c._source_relational_effect_text(
        text, relation_index=0
    )
    text = equality + " If credit exceeds liability, the refund is payable."
    assert c._source_relational_exception_match_indices(
        text, left_name="credit", relation=">", right_name="liability"
    ) == (1,)


def test_eq_extractor_does_not_walk_nested_negated_or_chained_nodes():
    for text in ("not (a == b)", "a == b == c", "(a == b) and eligible"):
        assert not [
            r
            for r in c._formula_relational_expressions(
                ast.parse(text, mode="eval").body
            )
            if r[1] == "=="
        ]


@pytest.mark.parametrize(
    "replacement",
    [
        "t9999_federal_non_business_foreign_tax_credit_line_3",
        "t2209_federal_non_business_foreign_tax_credit_line_4",
    ],
)
def test_wrong_external_form_or_line_cannot_use_semantic_core_alias(replacement):
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        CREDIT, replacement
    )
    inputs[replacement] = inputs.pop(CREDIT)
    inputs[replacement]["name"] = replacement
    for case in cases:
        case["input"][replacement] = case["input"].pop(CREDIT)
    assert not run(rule, cases, inputs)


def test_known_external_alias_requires_actual_source_line_reference():
    rule, cases, inputs = fixture()
    source = SOURCE.replace(
        "Enter the amount from line 3 of Form T2209",
        "Enter the amount from line 9 of Form T2209",
    )
    assert not run(rule, cases, inputs, source)
