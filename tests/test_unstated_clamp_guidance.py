"""The validators do not ask for a zero floor on a bare source amount.

The encoder prompt tells the model not to wrap a source-stated amount in
``max(0, ...)`` when the source states no floor. These tests pin the validator
behavior that guidance relies on: the unfloored shape is clean, the unstated
floor costs a below-zero companion case, and the floors the validators do
require (a reduction subtracted from a maximum) are unchanged.

The prompt keeps the floor on every difference. The floor check sees only some
differences, so for the others the prompt is the only guard; a floor on a
difference needs no below-zero companion case.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_nonnegative_amount_reduction_issues,
    numeric_value_is_grounded,
)

CITATION = "us/guidance/example/fee"
MODULE = "us:policies/example/fee"
SOURCE = "Mandatory association fees are an allowable expense deduction."
STATED_AMOUNT = "if fee_is_mandatory: fee_amount else: 0"
UNSTATED_FLOOR = "if fee_is_mandatory: max(0, fee_amount) else: 0"


def _payload(formula: str) -> dict:
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "fee_is_mandatory",
                "entity": "Household",
                "dtype": "Boolean",
                "period": "Month",
            },
            {
                "name": "fee_amount",
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
            },
        ],
        "rules": [
            {
                "name": "allowable_fee_expense_deduction",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": SOURCE,
                                },
                            }
                        ]
                    }
                },
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
        ],
    }


def _case(name: str, mandatory: bool, amount: int, expected: int) -> dict:
    return {
        "name": name,
        "period": "2026-01",
        "input": {
            f"{MODULE}#input.fee_is_mandatory": mandatory,
            f"{MODULE}#input.fee_amount": amount,
        },
        "output": {f"{MODULE}#allowable_fee_expense_deduction": expected},
    }


CASES = [
    _case("mandatory_fee", True, 250, 250),
    _case("optional_fee", False, 250, 0),
]


def _issues(formula: str, cases: list[dict]) -> list[str]:
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(_payload(formula), sort_keys=False),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return list(result.issues)


def test_stated_amount_without_a_floor_is_clean():
    assert _issues(STATED_AMOUNT, CASES) == []


def test_unstated_floor_needs_a_below_zero_case():
    issues = _issues(UNSTATED_FLOOR, CASES)
    assert len(issues) == 1
    assert "Direct nonnegative clamp companion evidence is missing" in issues[0]
    # The same candidate is clean once a case sets the amount below zero.
    assert _issues(UNSTATED_FLOOR, [*CASES, _case("negative", True, -50, 0)]) == []


@pytest.mark.parametrize("formula", [STATED_AMOUNT, UNSTATED_FLOOR])
def test_floor_check_does_not_ask_for_a_floor_on_a_bare_amount(formula):
    content = yaml.safe_dump(_payload(formula), sort_keys=False)
    assert find_nonnegative_amount_reduction_issues(content) == []


def test_floor_check_still_requires_the_floor_on_a_reduction():
    payload = _payload("maximum_fee_deduction - household_contribution")
    issues = find_nonnegative_amount_reduction_issues(
        yaml.safe_dump(payload, sort_keys=False)
    )
    assert len(issues) == 1
    assert "Nonnegative amount reduction missing floor" in issues[0]
    payload = _payload("max(0, maximum_fee_deduction - household_contribution)")
    assert (
        find_nonnegative_amount_reduction_issues(
            yaml.safe_dump(payload, sort_keys=False)
        )
        == []
    )


UNCHECKED_DIFFERENCES = [
    "fee_amount - fee_reimbursement",
    "total_fee_amount - fee_threshold",
    "fee_payment_standard - countable_amount",
    "tentative_fee_amount - phaseout_amount",
]


@pytest.mark.parametrize("difference", UNCHECKED_DIFFERENCES)
def test_floor_check_does_not_see_these_differences(difference):
    # No validator asks for the floor here; the prompt rule is the guard.
    content = yaml.safe_dump(_payload(difference), sort_keys=False)
    assert find_nonnegative_amount_reduction_issues(content) == []
    content = yaml.safe_dump(_payload(f"max(0, {difference})"), sort_keys=False)
    assert find_nonnegative_amount_reduction_issues(content) == []


def test_floor_on_a_difference_needs_no_below_zero_case():
    payload = _payload("max(0, fee_amount - fee_reimbursement)")
    payload["inputs"][0] = {
        "name": "fee_reimbursement",
        "entity": "Household",
        "dtype": "Money",
        "unit": "USD",
        "period": "Month",
    }
    source = "The fee less its reimbursement is an allowable expense deduction."
    payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source
    cases = [
        {
            "name": "fee_above_reimbursement",
            "period": "2026-01",
            "input": {
                f"{MODULE}#input.fee_amount": 250,
                f"{MODULE}#input.fee_reimbursement": 100,
            },
            "output": {f"{MODULE}#allowable_fee_expense_deduction": 150},
        }
    ]
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert not [i for i in result.issues if "Direct nonnegative clamp" in i]
