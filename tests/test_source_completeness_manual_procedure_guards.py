"""Procedure exemptions must preserve main's factual and operative verdicts.

Main admits US-manual analyzer gates only for stated numeric limits. The
non-numeric fixtures therefore compare every issue with the empty exemption
list; direct witness checks still retain their absolute obligations.
"""

from __future__ import annotations

import functools

import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.manual_procedure_differential import (
    analyze_with_manual_procedure_differential,
    manual_procedure_requires_paired_witness,
    unchanged_without_manual_procedure_exemptions,
)

CITATION = "us-fl/manual/test/section-1"
PAGE8_SEQUENCE = (
    "When the eligibility specialist has determined available income as per "
    "Chapters 1800 and 2400, the policy in this chapter must be used to perform "
    "the budgets and tests to determine eligibility for benefits and the benefit amount."
)


def _analyze(
    source: str, *, list_items: bool = False, item_rules=None, formulas=None, cases=None
):
    rules = [("gross_income_test_applies", "")]
    if list_items:
        rules += item_rules or [
            ("elderly_exception", "(1)"),
            ("disabled_exception", "(2)"),
        ]
    content = (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {CITATION}\nrules:\n"
    )
    for name, path in rules:
        formula = (formulas or {}).get(name, "true")
        content += (
            f"  - name: {name}\n    kind: derived\n    source: {CITATION}{path}\n"
            f"    versions: [{{formula: '{formula}'}}]\n"
        )
    return analyze_with_manual_procedure_differential(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=cases
        or [
            {
                "name": "only enabled case",
                "input": {"is_disabled": True, "countable_income": 1000},
                "output": {name: True for name, _ in rules},
            }
        ],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


@pytest.mark.parametrize(
    "source",
    (
        "When the eligibility specialist has determined that the applicant is disabled, "
        "apply the earned income deduction.",
        "When the eligibility specialist has determined income and the household "
        "includes a member over 60, subtract $100 from countable income.",
        "When the applicant is determined eligible, apply the earned income deduction.",
        "When the worker has determined income, apply the earned income deduction.",
    ),
)
def test_determined_facts_keep_their_gate_obligation(source):
    result = _analyze(source)
    if "subtract $100" in source:
        assert any("paired positive/blocking" in issue for issue in result.issues)
    else:
        assert result.issues == result.empty_list_issues


def test_incomplete_value_determination_keeps_its_paired_gate_obligation():
    assert manual_procedure_requires_paired_witness(PAGE8_SEQUENCE)
    result = _analyze(PAGE8_SEQUENCE)
    assert result.issues == result.empty_list_issues


GROSS_INCOME_SEQUENCE = (
    "when gross income is determined, that income is compared to the allowable "
    "gross income limit for the assistance group's size."
)


@pytest.mark.parametrize(
    "prefix",
    (
        "Subject to the household having a disabled member, ",
        "Only for households with an elderly member, ",
        "For residents, ",
        "Provided that the household has a disabled member, ",
        "The gross income standard applies only to households with an elderly member, and ",
        "The gross income standard applies only to households with an elderly member, or ",
        "The gross income standard applies only to households with an elderly member; ",
        "Subject to the household having a disabled member; ",
        "For residents; ",
        "(Subject to the household having a disabled member), ",
        "If the household has a disabled member, ",
    ),
)
def test_qualification_before_quantity_sequence_keeps_paired_obligation(prefix):
    result = _analyze(prefix + GROSS_INCOME_SEQUENCE)
    assert result.issues == result.empty_list_issues


@pytest.mark.parametrize(
    "tail",
    (
        " Only for households with an elderly member.",
        " The household must have a disabled member.",
        " If the applicant is resident, the test applies.",
        " The benefit amount is zero.",
        " Review the application.",
    ),
)
def test_extra_text_after_quantity_instruction_keeps_paired_obligation(tail):
    assert manual_procedure_requires_paired_witness(GROSS_INCOME_SEQUENCE + tail)


def _exception_cases():
    return [
        {
            "name": str((elderly, resident)),
            "period": "2026",
            "input": {
                "is_elderly": elderly,
                "is_resident": resident,
                "is_disabled": True,
            },
            "output": {
                "gross_income_test_applies": True,
                "elderly_exception": elderly,
                "resident_exception": resident,
                "disabled_exception": True,
            },
        }
        for elderly, resident in ((False, False), (True, False), (False, True))
    ]


def test_inline_parenthesized_exception_keeps_header_with_only_first_item_pair():
    source = (
        "Assistance groups must meet the standards with the following exceptions:\n"
        "1. Households with an elderly member; and (2) Households with a disabled member."
    )
    result = _analyze(
        source,
        list_items=True,
        formulas={"elderly_exception": "is_elderly"},
        cases=_exception_cases(),
    )
    assert result.issues == result.empty_list_issues


def test_later_item_qualification_keeps_header_despite_independent_item_pairs():
    source = (
        "Assistance groups must meet the standards with the following exceptions:\n"
        "1. Households with an elderly member. That member must also be disabled.\n"
        "2. Households with a resident member."
    )
    result = _analyze(
        source,
        list_items=True,
        item_rules=[("elderly_exception", "(1)"), ("resident_exception", "(2)")],
        formulas={
            "elderly_exception": "is_elderly",
            "resident_exception": "is_resident",
        },
        cases=_exception_cases(),
    )
    assert result.issues == result.empty_list_issues


@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(
    prefix=st.sampled_from(
        (
            "Subject to the household having a disabled member, ",
            "Only for households with an elderly member, ",
            "For residents, ",
            "Provided that the household has a disabled member, ",
        )
    ),
    sequence=st.sampled_from((PAGE8_SEQUENCE, GROSS_INCOME_SEQUENCE)),
    delimiter=st.sampled_from((",", ";")),
)
def test_quantity_sequence_never_erases_a_leading_qualification(
    prefix, sequence, delimiter
):
    prefix = prefix.rstrip(" ,") + delimiter + " "
    result = _analyze(prefix + sequence)
    assert result.issues == result.empty_list_issues


@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(
    marker=st.sampled_from(("2.", "(2)", "a.", "3.")),
    qualification=st.sampled_from(("", " That member must also be disabled.")),
)
def test_exception_header_obligation_survives_list_shape_and_item_qualifications(
    marker, qualification
):
    source = (
        "Assistance groups must meet the standards with the following exceptions:\n"
        f"1. Households with an elderly member.{qualification}\n"
        f"{marker} Households with a resident member."
    )
    result = _analyze(
        source,
        list_items=True,
        item_rules=[("elderly_exception", "(1)"), ("resident_exception", "(2)")],
        formulas={
            "elderly_exception": "is_elderly",
            "resident_exception": "is_resident",
        },
        cases=_exception_cases(),
    )
    assert result.issues == result.empty_list_issues


@pytest.mark.parametrize(
    "subject",
    (
        "a person who qualifies only if disabled",
        "a person who qualifies if disabled",
        "a person who qualifies",
        "a person unless resident",
        "a person when eligible",
        "the elderly or disabled person and their spouse",
    ),
)
def test_income_exclusions_keep_nested_conditions_and_operative_effect(subject):
    source = (
        f"When applying the gross income test, the income of {subject} "
        "is excluded from this calculation."
    )
    result = _analyze(source)
    assert result.issues == result.empty_list_issues


@pytest.mark.parametrize("subject", ("boarders", "the others"))
@pytest.mark.parametrize("scope", ("determination", "test", "calculation"))
def test_income_exclusion_without_subject_condition_keeps_operative_obligation(
    subject, scope
):
    source = (
        f"When applying the gross income test, the income of {subject} "
        f"is excluded from this {scope}."
    )
    assert sc._source_exception_effect_requirement(source) == "exclude"
    assert not sc._source_condition_is_manual_procedure(source)
    result = _analyze(source)
    assert result.issues == result.empty_list_issues


@pytest.mark.parametrize("separator", (" ", "\n", "\t"))
def test_semicolon_walkback_stops_at_the_containing_branch(separator):
    preceding = "Only for households with an elderly member;"
    source = preceding + separator + GROSS_INCOME_SEQUENCE
    current_start = len(preceding + separator)
    branches = (
        sc.SourceStructureBranch(("a",), "letter", "(a)", preceding, 0, len(preceding)),
        sc.SourceStructureBranch(
            ("b",),
            "letter",
            "(b)",
            GROSS_INCOME_SEQUENCE,
            current_start,
            len(source),
        ),
    )
    obligations = unchanged_without_manual_procedure_exemptions(
        sc._source_exception_branches,
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    (obligation,) = obligations
    assert obligation.path == ("b",)
    assert obligation.start == current_start
    assert obligation.text == GROSS_INCOME_SEQUENCE
    assert manual_procedure_requires_paired_witness(obligation.text)


@pytest.mark.parametrize("transition", ((99, 100), (100, 99)))
def test_inclusive_relation_keeps_legacy_numeric_witness_admission(transition):
    source = (
        "If income is greater than or equal to the income threshold, "
        "the applicant is ineligible."
    )
    rule = {
        "name": "gross_income_test_applies",
        "kind": "derived",
        "versions": [{"formula": "if income >= income_threshold: false else: true"}],
    }
    cases = [
        {
            "name": f"income={income}",
            "period": "2026",
            "input": {"income": income, "income_threshold": 100},
            "output": {"gross_income_test_applies": income < 100},
        }
        for income in transition
    ]
    witness = sc._ExceptionWitness(
        rule_name="gross_income_test_applies",
        selector_name="income",
        active_value=True,
        blocks=True,
        boolean_effect=True,
        zeroes=False,
        numeric_transition=transition,
        relational_transitions=(("income", ">=", "income_threshold"),),
        case_pair_identity=tuple(id(case) for case in cases),
    )
    branch = sc.SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )
    assert manual_procedure_requires_paired_witness(source)
    assert unchanged_without_manual_procedure_exemptions(
        sc._numeric_exception_witness_matches_source,
        branch,
        witness,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
    )
    assert all(
        sc._formula_execution_runtime_value(sc._case_formula_execution(rule, case))
        == case["output"]["gross_income_test_applies"]
        for case in cases
    )


@settings(max_examples=12, deadline=None, phases=(Phase.generate,))
@given(
    quantity=st.sampled_from(("income", "available income", "the benefit amount")),
    fact=st.sampled_from(
        (
            "the applicant is disabled",
            "the household includes a member over 60",
            "the applicant is a resident",
            "the household is eligible",
        )
    ),
)
def test_joining_a_fact_to_a_quantity_never_exempts_it(quantity, fact):
    source = (
        f"When the worker has determined {quantity} and {fact}, "
        "apply the earned income deduction."
    )
    assert manual_procedure_requires_paired_witness(source)


@settings(max_examples=8, deadline=None, phases=(Phase.generate,))
@given(cue=st.sampled_from(("if", "only if", "unless", "when")))
def test_nested_income_scope_condition_never_exempts_it(cue):
    source = (
        "When applying the gross income test, the income of a person "
        f"who qualifies {cue} disabled is excluded from this calculation."
    )
    assert manual_procedure_requires_paired_witness(source)
