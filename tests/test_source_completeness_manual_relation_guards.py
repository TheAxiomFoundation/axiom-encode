"""Retain relation/proviso probes and compare known witness limits with baseline.

Every nonexempt analyzer verdict must equal its empty-exemption verdict.
Absolute rejection of the historical witness limitations is tracked in #1920.
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
    numeric_value_is_grounded,
)
from tests.manual_procedure_differential import (
    analyze_with_manual_procedure_differential,
    manual_procedure_requires_paired_witness,
    unchanged_without_manual_procedure_exemptions,
)

CITATION = "de/statute/estg/32a"
EXTRACT_NUMBERS = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)


def _analyze(source: str, formula: str, cases: list[dict]):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
rules:
  - name: result
    kind: derived
    source: {CITATION}(1)
    versions:
      - formula: '{formula}'
"""
    return analyze_with_manual_procedure_differential(
        content,
        f"(1) {source}",
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT_NUMBERS,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _assert_empty_exemption_issues(result, source, formula, cases):
    # The analyzer helper already ran this exact fixture with the list emptied.
    assert result.issues == result.empty_list_issues


def _cases(values: tuple[int, ...], outcomes: tuple[bool, ...], *, threshold=100):
    return [
        {
            "name": str(value),
            "period": "2026",
            "input": {"income": value, "income_threshold": threshold},
            "output": {"result": outcome},
        }
        for value, outcome in zip(values, outcomes, strict=True)
    ]


def _assert_cases_follow_formula(formula, cases):
    rule = {"name": "result", "kind": "derived", "versions": [{"formula": formula}]}
    assert all(
        sc._formula_execution_runtime_value(sc._case_formula_execution(rule, case))
        == case["output"]["result"]
        for case in cases
    )


def _qualified_cases(*, threshold=100, permission=True):
    return [
        {
            "name": str((qualifies, income)),
            "period": "2026",
            "input": {
                "qualifies": qualifies,
                "income": income,
                "income_threshold": threshold,
            },
            "output": {
                "result": (qualifies and income <= threshold)
                if permission
                else not (qualifies and income <= threshold)
            },
        }
        for qualifies in (False, True)
        for income in (threshold - 1, threshold, threshold + 1)
    ]


@pytest.mark.parametrize(
    "proposition",
    (
        "If the applicant qualifies, the credit is not allowed",
        "The individual (and spouse or children if required) is ineligible",
    ),
)
def test_negative_proviso_rejects_reversed_effect_through_analyzer(proposition):
    source = f"{proposition}, provided income does not exceed the income threshold."
    result = _analyze(
        source,
        "if income > income_threshold: false else: true",
        _cases((99, 101), (True, False)),
    )
    assert any("paired" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "proposition", ("the credit is not allowed", "the applicant is ineligible")
)
def test_negative_proviso_accepts_withdrawal_of_prohibition(proposition):
    source = f"If the applicant qualifies, {proposition}, provided income does not exceed the income threshold."
    _assert_cases_follow_formula(
        "if qualifies and income <= income_threshold: false else: true",
        _qualified_cases(permission=False),
    )
    assert manual_procedure_requires_paired_witness(source)
    result = _analyze(
        source,
        "if qualifies and income <= income_threshold: false else: true",
        _qualified_cases(permission=False),
    )
    assert not result.issues


@pytest.mark.parametrize("reverse", (False, True))
def test_affirmative_proviso_requires_withdrawal_of_permission(reverse):
    source = "If the applicant qualifies, the credit is allowed, provided income does not exceed the income threshold."
    formula = (
        "if qualifies and income <= income_threshold: false else: true"
        if reverse
        else "if qualifies and income <= income_threshold: true else: false"
    )
    _assert_cases_follow_formula(formula, _qualified_cases(permission=not reverse))
    assert manual_procedure_requires_paired_witness(source)
    result = _analyze(source, formula, _qualified_cases(permission=not reverse))
    assert bool(result.issues) is reverse


@pytest.mark.parametrize(
    "proposition",
    (
        "the worker reviews the record",
        "the applicant who is ineligible receives the credit",
        "the credit is not allowed to expire",
    ),
)
def test_unreadable_proviso_effect_keeps_paired_obligation(proposition):
    result = _analyze(
        f"If the applicant qualifies, {proposition}, provided income does not exceed the income threshold.",
        "if income > income_threshold: false else: true",
        _cases((99, 101), (True, False)),
    )
    assert any("paired" in issue for issue in result.issues)


INCLUSIVE_SOURCE = "If income is equal to or exceeds the income threshold, the applicant is ineligible."


def test_inclusive_dynamic_bound_demands_an_authenticated_equality_case():
    result = _analyze(
        INCLUSIVE_SOURCE,
        "if income >= income_threshold and income != income_threshold: false else: true",
        _cases((99, 101), (True, False)),
    )
    assert any("paired" in issue for issue in result.issues)


@pytest.mark.parametrize("values", ((99, 100), (99, 100, 101)))
def test_unsupported_inclusive_wording_keeps_issue_with_correct_equality(values):
    result = _analyze(
        INCLUSIVE_SOURCE,
        "if income >= income_threshold: false else: true",
        _cases(values, tuple(value < 100 for value in values)),
    )
    assert manual_procedure_requires_paired_witness(INCLUSIVE_SOURCE)
    assert any("paired" in issue for issue in result.issues)


@pytest.mark.parametrize("equality_assertion", (False, True))
def test_inclusive_dynamic_bound_rejects_wrong_or_forged_equality(equality_assertion):
    result = _analyze(
        INCLUSIVE_SOURCE,
        "if income >= income_threshold and income != income_threshold: false else: true",
        _cases((99, 100, 101), (True, equality_assertion, False)),
    )
    assert any("paired" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "condition, values, outcomes",
    (
        ("income >= income_threshold + 1", (100, 101), (True, False)),
        ("income_threshold + 1 <= income", (100, 101), (True, False)),
        ("income - 1 >= income_threshold", (100, 101), (True, False)),
        ("income >= income_threshold - 1", (98, 100), (True, False)),
        ("income + 1 >= income_threshold", (98, 100), (True, False)),
    ),
)
def test_inclusive_dynamic_bound_rejects_arithmetic_operand_projection(
    condition, values, outcomes
):
    result = _analyze(
        INCLUSIVE_SOURCE,
        f"if {condition}: false else: true",
        _cases(values, outcomes),
    )
    assert any("paired" in issue for issue in result.issues)


def test_inclusive_dynamic_bound_rejects_a_different_source_operand():
    cases = _cases((99, 100), (True, False), threshold=200)
    for case in cases:
        case["input"]["other_income_threshold"] = 100
    result = _analyze(
        INCLUSIVE_SOURCE,
        "if income >= other_income_threshold: false else: true",
        cases,
    )
    assert any("paired" in issue for issue in result.issues)


@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(
    threshold=st.integers(min_value=2, max_value=10_000),
    shift=st.integers(min_value=1, max_value=20),
)
def test_inclusive_equality_cannot_be_the_ordinary_side_of_a_shifted_bound(
    threshold, shift
):
    result = _analyze(
        INCLUSIVE_SOURCE,
        f"if income >= income_threshold + {shift}: false else: true",
        _cases((threshold, threshold + shift), (True, False), threshold=threshold),
    )
    assert any("paired" in issue for issue in result.issues)


def _boolean_cases(selector: str):
    return [
        {
            "name": str(active),
            "period": "2026",
            "input": {selector: active},
            "output": {"result": not active},
        }
        for active in (False, True)
    ]


@pytest.mark.parametrize(
    "selector",
    (
        "income_is_equal_to_or_exceeds_income_threshold",
        "income_equals_or_exceeds_income_threshold",
        "income_exceeds_income_threshold",
        "income_is_above_income_threshold",
    ),
)
def test_inclusive_dynamic_bound_rejects_boolean_shortcut_without_equality(selector):
    _assert_cases_follow_formula(f"not {selector}", _boolean_cases(selector))
    assert manual_procedure_requires_paired_witness(INCLUSIVE_SOURCE)
    result = _analyze(INCLUSIVE_SOURCE, f"not {selector}", _boolean_cases(selector))
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(
        result, INCLUSIVE_SOURCE, f"not {selector}", _boolean_cases(selector)
    )


def test_unreadable_proviso_effect_rejects_boolean_shortcut():
    selector = "income_exceeds_income_threshold"
    source = "If the applicant qualifies, the worker reviews the record, provided income does not exceed the income threshold."
    _assert_cases_follow_formula(
        f"not {selector}",
        _boolean_cases(selector),
    )
    assert manual_procedure_requires_paired_witness(source)
    result = _analyze(source, f"not {selector}", _boolean_cases(selector))
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(
        result, source, f"not {selector}", _boolean_cases(selector)
    )


@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(threshold=st.integers(min_value=2, max_value=10_000))
def test_proviso_polarity_is_preserved_across_dynamic_thresholds(threshold):
    source = "If the applicant qualifies, the credit is not allowed, provided income does not exceed the income threshold."
    _assert_cases_follow_formula(
        "if qualifies and income <= income_threshold: false else: true",
        _qualified_cases(threshold=threshold, permission=False),
    )
    correct = _analyze(
        source,
        "if qualifies and income <= income_threshold: false else: true",
        _qualified_cases(threshold=threshold, permission=False),
    )
    reversed_effect = _analyze(
        source,
        "if qualifies and income <= income_threshold: true else: false",
        _qualified_cases(threshold=threshold, permission=True),
    )
    assert manual_procedure_requires_paired_witness(source)
    assert not correct.issues
    assert any("paired" in issue for issue in reversed_effect.issues)


@settings(max_examples=8, deadline=None, phases=(Phase.generate,), database=None)
@given(threshold=st.integers(min_value=2, max_value=10_000))
def test_unsupported_inclusive_relation_keeps_issue_with_or_without_equality(threshold):
    formula = "if income >= income_threshold: false else: true"
    missing = _analyze(
        INCLUSIVE_SOURCE,
        formula,
        _cases((threshold - 1, threshold + 1), (True, False), threshold=threshold),
    )
    complete = _analyze(
        INCLUSIVE_SOURCE,
        formula,
        _cases((threshold - 1, threshold), (True, False), threshold=threshold),
    )
    assert any("paired" in issue for issue in missing.issues)
    assert manual_procedure_requires_paired_witness(INCLUSIVE_SOURCE)
    assert any("paired" in issue for issue in complete.issues)


COMPOUND_INCLUSIVE_SHAPES = (
    (
        "conjunct_after",
        "income is greater than or equal to the income threshold and the applicant is a resident",
        "income >= income_threshold and is_resident",
        "and",
    ),
    (
        "conjunct_before",
        "the applicant is a resident and income is greater than or equal to the income threshold",
        "is_resident and income >= income_threshold",
        "and",
    ),
    (
        "exceeds_or_equals",
        "income exceeds or equals the income threshold and the applicant is a resident",
        "income >= income_threshold and is_resident",
        "and",
    ),
    (
        "equals_or_exceeds",
        "income equals or exceeds the income threshold and the applicant is a resident",
        "income >= income_threshold and is_resident",
        "and",
    ),
    (
        "disjunct",
        "income is greater than or equal to the income threshold or assets are greater than or equal to the asset threshold",
        "income >= income_threshold or assets >= asset_threshold",
        "or",
    ),
)


def _compound_cases(shape, *, threshold=100, shift=0, equality=True, strict=False):
    values = (threshold - 1, threshold, threshold + shift + 1)
    if not equality:
        values = (values[0], values[-1])
    cases = []
    for income in values:
        for other in values if shape == "or" else (False, True):
            income_active = (
                income > threshold if strict else income >= threshold + shift
            )
            other_active = other >= threshold if shape == "or" else other
            blocked = (
                income_active or other_active
                if shape in {"or", "or_boolean"}
                else income_active and other_active
            )
            cases.append(
                {
                    "name": str((income, other)),
                    "period": "2026",
                    "input": {
                        "income": income,
                        "income_threshold": threshold,
                        "assets": other if shape == "or" else 0,
                        "asset_threshold": threshold,
                        "is_resident": other if shape != "or" else False,
                    },
                    "output": {"result": not blocked},
                }
            )
    return cases


@pytest.mark.parametrize(
    "name, condition, formula_condition, shape", COMPOUND_INCLUSIVE_SHAPES
)
@pytest.mark.parametrize("mirrored", (False, True))
def test_compound_inclusive_comparison_accepts_honest_paired_cases(
    name, condition, formula_condition, shape, mirrored
):
    if mirrored:
        formula_condition = formula_condition.replace(
            "income >= income_threshold", "income_threshold <= income"
        ).replace("assets >= asset_threshold", "asset_threshold <= assets")
    _assert_cases_follow_formula(
        f"if {formula_condition}: false else: true",
        _compound_cases(shape),
    )
    assert manual_procedure_requires_paired_witness(
        f"If {condition}, the applicant is ineligible."
    )
    result = _analyze(
        f"If {condition}, the applicant is ineligible.",
        f"if {formula_condition}: false else: true",
        _compound_cases(shape),
    )
    assert not result.issues


@pytest.mark.parametrize("operand", ("income", "assets"))
def test_inclusive_disjuncts_each_require_their_own_equality_case(operand):
    _name, condition, formula_condition, _shape = COMPOUND_INCLUSIVE_SHAPES[-1]
    cases = [case for case in _compound_cases("or") if case["input"][operand] != 100]
    source = f"If {condition}, the applicant is ineligible."
    result = _analyze(
        source,
        f"if {formula_condition}: false else: true",
        cases,
    )
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(
        result, source, f"if {formula_condition}: false else: true", cases
    )


def test_other_conjunct_does_not_exempt_the_source_comparison():
    _name, condition, _formula_condition, _shape = COMPOUND_INCLUSIVE_SHAPES[0]
    source = f"If {condition}, the applicant is ineligible."
    branch = sc.SourceStructureBranch(("1",), "number", "1.", source, 0, len(source))
    witness = sc._ExceptionWitness(
        rule_name="result",
        selector_name="is_resident",
        active_value=True,
        blocks=True,
        boolean_effect=True,
        zeroes=False,
        numeric_transition=None,
    )
    assert manual_procedure_requires_paired_witness(source)
    assert not unchanged_without_manual_procedure_exemptions(
        sc._numeric_exception_witness_matches_source,
        branch,
        witness,
        extract_numeric_occurrences=EXTRACT_NUMBERS,
    )


@pytest.mark.parametrize(
    "name, condition, formula_condition, shape", COMPOUND_INCLUSIVE_SHAPES
)
@pytest.mark.parametrize(
    "defect", ("missing_equality", "shifted", "strict", "unrelated")
)
def test_compound_inclusive_comparison_rejects_incorrect_boundary(
    name, condition, formula_condition, shape, defect
):
    cases = _compound_cases(
        shape,
        shift=1 if defect == "shifted" else 0,
        equality=defect != "missing_equality",
        strict=defect == "strict",
    )
    if defect == "shifted":
        formula_condition = formula_condition.replace(
            ">= income_threshold", ">= income_threshold + 1"
        )
    elif defect == "strict":
        formula_condition = formula_condition.replace("income >=", "income >")
    elif defect == "unrelated":
        formula_condition = formula_condition.replace(
            "income_threshold", "other_income_threshold"
        )
        for case in cases:
            case["input"]["other_income_threshold"] = 100
            case["input"]["income_threshold"] = 200
    source = f"If {condition}, the applicant is ineligible."
    result = _analyze(
        source,
        f"if {formula_condition}: false else: true",
        cases,
    )
    assert manual_procedure_requires_paired_witness(source)
    # #1920 restores absolute rejection for the 19 baseline boundary limitations.
    if defect == "strict" and name == "disjunct":
        assert any("paired" in issue for issue in result.issues)
    else:
        _assert_empty_exemption_issues(
            result, source, f"if {formula_condition}: false else: true", cases
        )


@settings(max_examples=20, deadline=None, phases=(Phase.generate,), database=None)
@given(
    phrase=st.sampled_from(
        ("is greater than or equal to", "exceeds or equals", "equals or exceeds")
    ),
    connector=st.sampled_from(("and", "or")),
    comparison_first=st.booleans(),
    threshold=st.integers(min_value=2, max_value=10_000),
    shift=st.integers(min_value=1, max_value=20),
)
def test_compound_inclusive_equality_is_satisfiable_and_cannot_shift(
    phrase, connector, comparison_first, threshold, shift
):
    comparison = f"income {phrase} the income threshold"
    other = "the applicant is a resident"
    conditions = (comparison, other) if comparison_first else (other, comparison)
    condition = f" {connector} ".join(conditions)
    expressions = ("income >= income_threshold", "is_resident")
    if not comparison_first:
        expressions = expressions[::-1]
    formula_condition = f" {connector} ".join(expressions)
    shape = "and" if connector == "and" else "or_boolean"
    source = f"If {condition}, the applicant is ineligible."
    _assert_cases_follow_formula(
        f"if {formula_condition}: false else: true",
        _compound_cases(shape, threshold=threshold),
    )
    honest = _analyze(
        source,
        f"if {formula_condition}: false else: true",
        _compound_cases(shape, threshold=threshold),
    )
    shifted_condition = formula_condition.replace(
        ">= income_threshold", f">= income_threshold + {shift}"
    )
    shifted = _analyze(
        source,
        f"if {shifted_condition}: false else: true",
        _compound_cases(shape, threshold=threshold, shift=shift),
    )
    assert manual_procedure_requires_paired_witness(source)
    assert not honest.issues
    # #1920 restores absolute rejection when the historical witness is fixed.
    _assert_empty_exemption_issues(
        shifted,
        source,
        f"if {shifted_condition}: false else: true",
        _compound_cases(shape, threshold=threshold, shift=shift),
    )
