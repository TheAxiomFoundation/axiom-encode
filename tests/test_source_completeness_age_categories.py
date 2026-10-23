"""Unresolved numeric-plus phrases preserve base arithmetic obligations.

A definition reference alone cannot authenticate a category rather than an
addition operand, so this withdrawn relaxation keeps the original source text.
"""

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "nz/statute/act/public/2018/0032/section/72"
# Social Security Act 2018, s 72(2)(ca). The age interval is independent of
# the externally defined category and the contribution condition.
SOURCE = (
    "the person is aged 50 to 64 years, is a 50+ single person as defined in "
    "section 21 of the Residential Care and Disability Support Services Act "
    "2018 , and is under section 22 of that Act liable to make a contribution "
    "based on income towards the cost of the person’s LTR contracted care "
    "that is less than the maximum contribution (as defined in section 15(2) "
    "of that Act)"
)


def test_unresolved_single_person_category_retains_base_computation_obligation():
    assert [m.group() for m in sc._ARITHMETIC_EXPRESSION.finditer(SOURCE)] == [
        "50+ single"
    ]
    assert sc.source_states_explicit_computation(SOURCE)
    assert sc._source_states_nonrounding_computation(SOURCE)
    assert sc._formula_operation_kinds(SOURCE) == {"add"}
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    assert sc._source_formula_branches(
        SOURCE,
        corpus_citation_path=CITATION,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )


def test_boolean_care_eligibility_keeps_age_bounds_and_unresolved_formula_gate():
    rule = {
        "name": "single_residential_care_disqualification",
        "kind": "derived",
        "dtype": "Judgment",
        "period": "Year",
        "source": CITATION,
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": SOURCE},
                    }
                ]
            }
        },
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": (
                    "age >= minimum_age and age <= maximum_age "
                    "and single_person_under_section_21 "
                    "and liable_contribution_below_maximum_under_section_22"
                ),
            }
        ],
    }
    parameters = [
        {
            "name": name,
            "kind": "parameter",
            "dtype": "Count",
            "period": "Year",
            "source": CITATION,
            "versions": [{"effective_from": "2026-01-01", "formula": value}],
        }
        for name, value in (("minimum_age", 50), ("maximum_age", 64))
    ]
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "age", "dtype": "Count"},
            {"name": "single_person_under_section_21", "dtype": "Bool"},
            {
                "name": "liable_contribution_below_maximum_under_section_22",
                "dtype": "Bool",
            },
        ],
        "rules": [*parameters, rule],
    }
    cases = []
    for name, age, single, contribution, expected in (
        ("positive", 60, True, True, True),
        ("below age", 49, True, True, False),
        ("above age", 65, True, True, False),
        ("not single", 60, False, True, False),
        ("no contribution", 60, True, False, False),
    ):
        case = {
            "name": name,
            "period": "2026",
            "input": {
                "age": age,
                "single_person_under_section_21": single,
                "liable_contribution_below_maximum_under_section_22": contribution,
            },
            "output": {rule["name"]: "holds" if expected else "not_holds"},
        }
        execution = sc._case_formula_execution(
            rule,
            case,
            formula_environment={"minimum_age": 50, "maximum_age": 64},
        )
        assert execution is not None
        assert execution.evaluated_value == ("bool", str(expected))
        cases.append(case)
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    assert [issue for issue in result.issues if "formula branch" in issue]


@given(
    threshold=st.integers(min_value=1, max_value=999),
    category=st.sampled_from(("single person", "household", "resident claimant")),
    definition=st.sampled_from(
        ("as defined in", "as defined by", "within the meaning of")
    ),
)
@settings(max_examples=35, deadline=None)
def test_unresolved_category_preserves_scalars_and_separate_arithmetic(
    threshold, category, definition
):
    source = f"The person is a {threshold}+ {category} {definition} section 21."
    assert str(threshold) in sc.authoritative_numeric_recall_text(source)
    assert sc.source_states_explicit_computation(source)
    assert sc._formula_operation_kinds(source) == {"add"}
    arithmetic = " The amount is (income + 2) * 3."
    assert sc.source_states_explicit_computation(source + arithmetic)
    assert sc._formula_operation_kinds(source + arithmetic) == {"add", "multiply"}
    assert sc._explicit_source_arithmetic_topology(arithmetic) is not None


@pytest.mark.parametrize(
    "source",
    (
        "The amount is 50 + single_person_income.",
        "The amount is 50+ single_person_income.",
        "The amount is 50+ income as defined in section 21.",
        "The amount is a total of 50 + income as defined in section 21.",
        "The amount is the 50 + income as defined in section 21.",
        "The amount is a 50 + 20 supplement.",
        "The amount is 50 + 20.",
    ),
)
def test_unresolved_category_keeps_actual_addition(source):
    assert sc.source_states_explicit_computation(source)
    assert "add" in sc._formula_operation_kinds(source)


def test_unresolved_category_preserves_independently_stated_age_conditions():
    assert "the person is aged 50 to 64 years" in SOURCE
    assert sc.authoritative_numeric_recall_text(SOURCE).count("50") == 2
    assert "64" in sc.authoritative_numeric_recall_text(SOURCE)


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "phrase",
    (
        "a 50+ income as defined in section 21",
        "a 50+ supplement as defined by section 21",
        "the 50+ increment within the meaning of section 21",
    ),
)
def test_definition_reference_does_not_resolve_numeric_phrase_as_category(
    jurisdiction, phrase
):
    source = f"The amount is {phrase}."
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    assert sc.source_states_explicit_computation(source)
    assert "add" in sc._formula_operation_kinds(source)
    assert "50" in sc.authoritative_numeric_recall_text(source)
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    assert sc._source_formula_branches(
        source,
        corpus_citation_path=citation,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
