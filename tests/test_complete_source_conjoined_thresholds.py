"""Every source-stated threshold of a conjoined condition is a boundary demand.

`less than $500 and whose net income is less than $300` composes into one
interval that keeps only the tighter `$300`, so a wrong `<=` on the `$500`
comparison used to pass the companion-test analysis. Each dropped threshold is
now demanded with its own comparator's direction and inclusivity.
"""

from __future__ import annotations

import functools

import pytest

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    collect_artifact_numeric_bindings,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us/statute/7/9999"
TARGET = "us:statutes/7/9999"
EXTRACT = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="en-US"
)
SOURCE = (
    "(a) A household whose countable income is less than $500 and whose net "
    "income is less than $300 is eligible.\n"
    "(b) The net income is the countable income minus the household deduction.\n"
)
ELIGIBILITY = (
    "A household whose countable income is less than $500 and whose net income "
    "is less than $300 is eligible."
)


def _proof(excerpt: str, kind: str = "formula") -> str:
    return (
        "    metadata:\n      proof:\n        atoms:\n"
        "          - path: versions[0].formula\n"
        f"            kind: {kind}\n            source:\n"
        f"              corpus_citation_path: {CITATION}\n"
        f"              excerpt: {excerpt}\n"
    )


def _module(income_comparison: str) -> str:
    return (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {CITATION}\ninputs:\n"
        "  - name: countable_income\n    entity: Household\n    dtype: Money\n"
        "    period: Month\n"
        "  - name: household_deduction\n    entity: Household\n    dtype: Money\n"
        "    period: Month\n"
        "rules:\n"
        "  - name: income_limit\n    kind: parameter\n    dtype: Money\n    unit: USD\n"
        + _proof("less than $500", "amount")
        + "    versions:\n      - effective_from: '2026-01-01'\n        formula: '500'\n"
        "  - name: net_limit\n    kind: parameter\n    dtype: Money\n    unit: USD\n"
        + _proof("less than $300", "amount")
        + "    versions:\n      - effective_from: '2026-01-01'\n        formula: '300'\n"
        "  - name: net_income\n    kind: derived\n    entity: Household\n"
        "    dtype: Money\n    unit: USD\n    period: Month\n"
        + _proof(
            "The net income is the countable income minus the household deduction."
        )
        + "    versions:\n      - effective_from: '2026-01-01'\n"
        "        formula: countable_income - household_deduction\n"
        "  - name: household_eligible\n    kind: derived\n    entity: Household\n"
        "    dtype: Judgment\n    period: Month\n"
        + _proof(ELIGIBILITY)
        + "    versions:\n      - effective_from: '2026-01-01'\n"
        f"        formula: {income_comparison} and net_income < net_limit\n"
    )


def _case(name: str, income: int, deduction: int, eligible: str, net: int):
    return {
        "name": name,
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": income,
            f"{TARGET}#input.household_deduction": deduction,
        },
        "output": {
            f"{TARGET}#household_eligible": eligible,
            f"{TARGET}#net_income": net,
        },
    }


BASE_CASES = [
    _case("low", 100, 0, "holds", 100),
    _case("high", 900, 0, "not_holds", 900),
    _case("mid", 450, 200, "holds", 250),
    # Net income exactly at the $300 boundary: not eligible.
    _case("net at limit", 400, 100, "not_holds", 300),
]


def _boundary_issues(content: str, cases) -> list[str]:
    bindings = collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    result = analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=EXTRACT,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return [issue for issue in result.issues if "boundary input" in issue]


def test_the_threshold_the_composed_interval_drops_is_demanded():
    [issue] = _boundary_issues(_module("countable_income < income_limit"), BASE_CASES)

    assert "(a)=500" in issue
    assert "(a)=300" not in issue


def test_a_case_at_the_dropped_threshold_with_the_source_value_closes_it():
    cases = [*BASE_CASES, _case("income at limit", 500, 400, "not_holds", 100)]

    assert _boundary_issues(_module("countable_income < income_limit"), cases) == []


def test_a_wrong_inclusive_comparison_on_the_dropped_threshold_keeps_its_demand():
    """The source says "less than $500"; `<=` admits 500. The model's case
    agrees with its own wrong encoding, so only the source can expose it."""

    cases = [*BASE_CASES, _case("income at limit", 500, 400, "holds", 100)]

    [issue] = _boundary_issues(_module("countable_income <= income_limit"), cases)

    assert "(a)=500" in issue


def test_an_inverted_comparison_on_the_dropped_threshold_keeps_its_demand():
    cases = [
        _case("low", 100, 0, "not_holds", 100),
        _case("mid", 450, 200, "not_holds", 250),
        _case("income at limit", 500, 400, "holds", 100),
        _case("income above limit", 501, 400, "holds", 101),
    ]

    [issue] = _boundary_issues(_module("countable_income >= income_limit"), cases)

    assert "(a)=500" in issue


def _new_thresholds(text: str):
    branches = recognize_source_structure(text, corpus_citation_path=CITATION)
    roots = completeness_module._active_or_root_source_branches(
        text, branches=branches, active_branches=branches, deferred_paths=set()
    )
    existing = completeness_module._source_boundary_obligations(
        roots, extract_numeric_occurrences=EXTRACT
    )
    return [
        (
            occurrence.raw,
            "lower" if interval.lower is not None else "upper",
            interval.lower_inclusive
            if interval.lower is not None
            else interval.upper_inclusive,
        )
        for _branch, occurrence, interval, _polarity in (
            completeness_module._source_conjoined_threshold_obligations(
                roots,
                branches=branches,
                existing=existing,
                extract_numeric_occurrences=EXTRACT,
            )
        )
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        (
            "A household whose countable income is less than $500 and whose net "
            "income is less than $300 is eligible.",
            [("500", "upper", False)],
        ),
        (
            "A household whose income is not more than $500 and whose resources "
            "are not more than $2,000 is eligible.",
            [("2,000", "upper", True)],
        ),
        (
            "A household whose income is less than $500 or more than $1,000 is "
            "ineligible.",
            [("1,000", "lower", False)],
        ),
        (
            "A household whose income is less than $500, whose net income is less "
            "than $300, and whose resources are less than $2,000 is eligible.",
            [("300", "upper", False), ("2,000", "upper", False)],
        ),
        # Real operative text (us-co HB24-1052): a second filing-status limit.
        (
            "has a federal adjusted gross income less than or equal to seventy-five "
            "thousand dollars for an individual filing a single return, or has a "
            "federal adjusted gross income less than or equal to one hundred "
            "twenty-five thousand dollars for an individual filing a joint return",
            [("one hundred twenty-five thousand", "upper", True)],
        ),
    ),
)
def test_each_conjoined_threshold_keeps_its_own_direction(text, expected):
    assert _new_thresholds(text) == expected


@pytest.mark.parametrize(
    "text",
    (
        # One comparator, not two clauses.
        "A household whose income is less than or equal to $500 is eligible.",
        "A household whose income is between $100 and $500 is eligible.",
        # A permissive modal scopes over the whole sentence.
        "The agency may approve a household whose income is less than $500 and "
        "whose resources are less than $2,000.",
        # The same value again is not a second demand.
        "A household that has at least 30 quarters of coverage, or is married to "
        "someone with at least 30 quarters of coverage, may buy into Part A.",
        "Custody is for more than one-half of the calendar year if a parent has "
        "physical custody for more than one-half of the calendar year.",
        # A clause joined by `or` after a negation shares that negation
        # (review of #1937): reading it alone would invert its direction.
        "The payment shall not exceed $500 or be less than $100.",
        "The payment shall not be more than $500 or be less than $100.",
        "A household whose income does not exceed $500 or fall below $100 is eligible.",
        "A household whose income is not below $100 or above $500 is eligible.",
        # Real operative text (us-ok/statute/68-3017).
        "The amount of uncollected taxes for the previous year or years; provided "
        "that the reserve so added shall not exceed twenty percent (20%) or be "
        "less than five percent (5%); and provided, further, that the reserve so "
        "added shall not be subject to review.",
        # One fragment, two opposite effects: no single polarity fits both.
        "A household is ineligible if its income is more than $500, and is "
        "eligible if its resources are less than $2,000.",
        "A household is eligible if income is less than $500, and the household "
        "is ineligible if resources are more than $2,000.",
        # `more or less than` is not a threshold (us-ga SNAP 3605).
        "In some instances, more or less than one month's income or expenses OR "
        "more or less than the four (4) current consecutive weeks of pay for "
        "fluctuating income is used if one month's income is not representative.",
    ),
)
def test_no_new_threshold_without_a_second_source_comparison(text):
    assert _new_thresholds(text) == []
