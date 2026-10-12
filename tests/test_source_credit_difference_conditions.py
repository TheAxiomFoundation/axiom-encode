"""Credit subtraction operands are computations, not claimant fact gates."""

from functools import partial

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CLAUSE = (
    "If the resident taxpayer has a federal tax liability, the taxpayer shall "
    "claim from the state not more than the difference between the amount of "
    "the state child care credit for which the taxpayer is eligible and the "
    "federal credit for child and dependent care expenses the taxpayer is able "
    "to deduct from federal tax liability for the same taxable year;"
)
CITATION = "us-nm/statute/7-2-18.1"


def test_nm_credit_difference_has_no_conjunctive_operand_facts():
    gates = sc._source_conjunctive_fact_gates(CLAUSE)
    assert gates == ()
    payload = {
        "inputs": [
            {
                "name": "resident_has_federal_tax_liability",
                "description": "The resident taxpayer has federal tax liability.",
            },
            {
                "name": "state_child_care_credit",
                "description": "The amount of the state child care credit for which the taxpayer is eligible.",
            },
            {
                "name": "federal_cdcc_able_to_deduct",
                "description": "Federal credit for child and dependent care expenses the taxpayer is able to deduct from federal tax liability.",
            },
        ]
    }
    assert sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in payload["inputs"]),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(payload),
    )


def test_exact_nm_credit_clause_corroborates_formula_at_explicit_conditions_gate():
    rule = {
        "name": "nm_cdcc",
        "kind": "derived",
        "entity": "TaxUnit",
        "dtype": "Money",
        "unit": "USD",
        "period": "Year",
        "source": CITATION + "(e)",
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": "if resident_has_federal_tax_liability: state_child_care_credit - federal_cdcc_able_to_deduct else: state_child_care_credit",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": CLAUSE,
                        },
                    }
                ]
            }
        },
    }
    source = "E. " + CLAUSE
    payload = {
        "inputs": [
            {"name": name, "description": description}
            for name, description in (
                (
                    "resident_has_federal_tax_liability",
                    "The resident taxpayer has federal tax liability.",
                ),
                ("state_child_care_credit", "The state child care credit amount."),
                (
                    "federal_cdcc_able_to_deduct",
                    "The federal credit for child and dependent care expenses deductible from federal tax liability.",
                ),
            )
        ],
        "rules": [rule],
    }
    assert (
        sc._opaque_same_source_condition_input_issues(
            payload,
            source_text=source,
            branches=sc.recognize_source_structure(
                source, corpus_citation_path=CITATION
            ),
            principal_rules={"nm_cdcc": rule},
            corpus_citation_path=CITATION,
        )
        == []
    )


def test_dependent_care_subject_matter_is_not_dependency_polarity():
    phrase = "the federal credit for child and dependent care expenses"
    assert "dependent" not in sc._source_gate_predicate_polarities(phrase)
    assert not {
        "dependent",
        "not_dependent",
        "unknown_dependent",
    } & sc._source_gate_semantic_tokens(phrase)


@pytest.mark.parametrize("punctuation", [",", ";", ":", ".", " —"])
def test_punctuation_keeps_dependency_status_separate_from_care(punctuation):
    phrase = f"the child is dependent{punctuation} care expenses are deductible"
    assert sc._source_gate_predicate_polarities(phrase) == {"dependent": False}
    assert "dependent" in sc._source_gate_semantic_tokens(phrase)


def test_credit_difference_retains_complete_computation_demand():
    source = CLAUSE
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    obligations = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(obligations) == 1
    assert obligations[0].text == CLAUSE
    assert source[obligations[0].start : obligations[0].end] == CLAUSE


def test_credit_difference_still_requires_a_formula_output():
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []}),
        "E. " + CLAUSE,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert any(
        "[complete-source-unit:formula-output]" in issue for issue in result.issues
    )


@given(
    st.sampled_from(
        [
            "the applicant is resident",
            "the taxpayer is disabled",
            "the spouse is employed",
        ]
    ),
    st.sampled_from(
        [
            "the spouse is blind",
            "the applicant has income",
            "the child is dependent on the taxpayer",
        ]
    ),
    st.sampled_from([" ", "\n", "\t", "  "]),
)
@settings(deadline=None, max_examples=80)
def test_credit_difference_preserves_all_antecedent_fact_demands(first, second, space):
    antecedent = f"If {first} and {second}"
    expected = sc._source_conjunctive_fact_gates(antecedent)
    assert len(expected) == 2
    consequence = CLAUSE.split(",", 1)[1].replace(" ", space)
    source = antecedent + "," + consequence
    assert sc._source_conjunctive_fact_gates(source) == expected
    assert sc.source_states_explicit_computation(source)


@pytest.mark.parametrize(
    "restriction",
    [
        "if the spouse is disabled",
        "and the spouse is disabled",
        "only when the applicant is employed",
        "provided that the taxpayer is resident",
    ],
)
def test_unrecognized_credit_consequence_conditions_remain_demanded(restriction):
    restricted = CLAUSE.rstrip(";") + " " + restriction + ";"
    assert sc._source_conjunctive_fact_gates(restricted)


@given(
    st.sampled_from(
        [
            "may be dependent",
            "is sometimes dependent",
            "is neither dependent nor independent",
        ]
    )
)
@settings(deadline=None, max_examples=10)
def test_dependent_care_does_not_hide_ambiguous_dependency_status(status):
    text = f"If the child {status} on the taxpayer and the spouse is disabled, claim the dependent care credit."
    gates = sc._source_conjunctive_fact_gates(text)
    assert any("unknown_dependent" in predicates for _, predicates in gates)
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset({"child_dependent_fact", "spouse_is_disabled"}),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": "child_dependent_fact"},
                    {"name": "spouse_is_disabled"},
                ]
            }
        ),
    )


@given(
    st.integers(min_value=0, max_value=20),
    st.sampled_from(["the taxpayer is dependent", "the taxpayer has income"]),
)
@settings(deadline=None, max_examples=80)
def test_care_subject_preserves_other_predicate_negation_windows(padding, predicate):
    prefix = "not eligible for a credit for {} care expenses "
    suffix = "qualifier " * padding + predicate
    neutral = prefix.format("ancillary") + suffix
    subject = prefix.format("dependent") + suffix
    assert sc._source_gate_predicate_polarities(
        subject
    ) == sc._source_gate_predicate_polarities(neutral)


@given(
    st.integers(min_value=0, max_value=120),
    st.sampled_from(["the spouse is dependent", "the spouse has income"]),
)
@example(51, "the spouse is dependent")
@settings(deadline=None, max_examples=80)
def test_care_subject_preserves_guarded_yet_character_windows(padding, predicate):
    prefix = "the applicant is disabled yet eligible for {} care expenses "
    suffix = "q" * padding + " " + predicate
    neutral = prefix.format("ancillary") + suffix
    subject = prefix.format("dependent") + suffix
    assert sc._source_gate_predicate_polarities(
        subject
    ) == sc._source_gate_predicate_polarities(neutral)
