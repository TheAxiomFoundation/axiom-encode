"""Companion-test findings: no duplicate demands, exact boundary repair detail,
and structured obligations for deterministic case synthesis."""

from __future__ import annotations

import functools

import pytest

from axiom_encode.harness.source_completeness import (
    MISSING_BOUNDARY_DETAIL_MARKER,
    CompanionTestObligation,
    analyze_complete_source_unit,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us-zz/statute/9"
TESTS = "[complete-source-unit:tests]"

SOURCE = """\
(a) A household whose countable income is less than $500 is eligible.
(b) A member who is less than 18 years of age is a minor.
"""

HEADER = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
inputs:
  - name: countable_income
    entity: Household
    dtype: Money
    period: Month
  - name: member_age
    entity: Household
    dtype: Integer
    period: Month
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: less than $500
    versions:
      - effective_from: '2026-01-01'
        formula: 500
  - name: household_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: A household whose countable income is less than $500 is eligible.
    versions:
      - effective_from: '2026-01-01'
        formula: __FORMULA__
"""

MINOR_RULES = f"""\
  - name: minor_age_limit
    kind: parameter
    dtype: Integer
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: less than 18 years of age
    versions:
      - effective_from: '2026-01-01'
        formula: 18
  - name: member_is_minor
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: A member who is less than 18 years of age is a minor.
    versions:
      - effective_from: '2026-01-01'
        formula: member_age < minor_age_limit
"""


def _case(name: str, output: dict[str, str], **inputs: int) -> dict[str, object]:
    return {"name": name, "period": "2026-01", "input": inputs, "output": output}


ELIGIBILITY_CASES = [
    _case("low", {"household_eligible": "holds"}, countable_income=100),
    _case("high", {"household_eligible": "not_holds"}, countable_income=900),
]


def _analyze(content: str, cases: list[dict[str, object]]):
    return analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _boundary_issues(result) -> list[str]:
    return [issue for issue in result.issues if "boundary input; missing:" in issue]


def _listed(issue: str) -> list[str]:
    listed = issue.split("missing:", 1)[1]
    listed = listed.split(MISSING_BOUNDARY_DETAIL_MARKER, 1)[0]
    return [item.strip() for item in listed.strip().rstrip(".").split(",")]


def test_boundary_finding_names_source_text_covering_output_and_input():
    result = _analyze(
        HEADER.replace("__FORMULA__", "countable_income < income_limit"),
        ELIGIBILITY_CASES,
    )

    [issue] = _boundary_issues(result)
    assert _listed(issue) == ["(a)=500"]
    detail = issue.split(MISSING_BOUNDARY_DETAIL_MARKER, 1)[1]
    assert "less than $500 is eligible" in detail
    assert "covering principal output(s) `household_eligible`" in detail
    assert "input(s) compared directly with 500 `countable_income`" in detail
    assert "to exactly the listed value" in detail
    assert result.companion_test_obligations == (
        CompanionTestObligation(
            "boundary",
            ("household_eligible",),
            values=(500.0,),
            input_names=("countable_income",),
            label="(a)=500",
        ),
    )


def test_case_at_the_boundary_clears_the_finding_and_the_obligation():
    result = _analyze(
        HEADER.replace("__FORMULA__", "countable_income < income_limit"),
        [
            *ELIGIBILITY_CASES,
            _case("at", {"household_eligible": "not_holds"}, countable_income=500),
        ],
    )

    assert _boundary_issues(result) == []
    assert not any(
        obligation.kind == "boundary"
        for obligation in result.companion_test_obligations
    )


def test_unencoded_branch_boundary_is_reported_once_as_structure():
    result = _analyze(
        HEADER.replace("__FORMULA__", "countable_income < income_limit"),
        ELIGIBILITY_CASES,
    )

    assert any(
        issue.startswith("[complete-source-unit:structure] Source branch (b)")
        for issue in result.issues
    )
    assert all("(b)=18" not in _listed(issue) for issue in _boundary_issues(result))


def test_encoded_branch_boundary_is_still_demanded():
    """Negative control: the same boundary returns once a principal output covers it."""

    content = HEADER.replace("__FORMULA__", "countable_income < income_limit")
    result = _analyze(
        content + MINOR_RULES,
        [
            *ELIGIBILITY_CASES,
            _case("child", {"member_is_minor": "holds"}, member_age=10),
            _case("adult", {"member_is_minor": "not_holds"}, member_age=40),
        ],
    )

    [issue] = _boundary_issues(result)
    assert "(b)=18" in _listed(issue)
    assert not any(
        issue.startswith("[complete-source-unit:structure] Source branch (b)")
        for issue in result.issues
    )
    assert (
        CompanionTestObligation(
            "boundary",
            ("member_is_minor",),
            values=(18.0,),
            input_names=("member_age",),
            label="(b)=18",
        )
        in result.companion_test_obligations
    )


@pytest.mark.parametrize(
    "formula",
    [
        # The compared quantity is a sum, so placing 500 in one addend is not
        # the source boundary and must never be proposed.
        "countable_income + 0 < income_limit",
        "countable_income * 2 < income_limit * 2",
    ],
)
def test_no_synthesis_input_when_the_input_is_not_compared_directly(formula):
    result = _analyze(HEADER.replace("__FORMULA__", formula), ELIGIBILITY_CASES)

    assert not any(
        obligation.kind == "boundary"
        for obligation in result.companion_test_obligations
    )
    [issue] = _boundary_issues(result)
    assert "no input is compared directly with 500" in issue


def test_fractional_boundary_is_never_proposed_for_an_integer_input():
    content = HEADER.replace(
        "__FORMULA__", "countable_income < income_limit"
    ) + MINOR_RULES.replace("formula: 18\n", "formula: 18.5\n").replace(
        "less than 18 years", "less than 18.5 years"
    ).replace("excerpt: less than 18 years of age", "excerpt: less than 18.5")
    source = SOURCE.replace("less than 18 years", "less than 18.5 years")
    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[
            *ELIGIBILITY_CASES,
            _case("child", {"member_is_minor": "holds"}, member_age=10),
            _case("adult", {"member_is_minor": "not_holds"}, member_age=40),
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

    assert any("(b)=18.5" in _listed(issue) for issue in _boundary_issues(result))
    assert not any(
        obligation.label == "(b)=18.5"
        for obligation in result.companion_test_obligations
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("more than 1 qualifying child", (2.0,)),
        ("at least 3 qualifying children", (3.0, 4.0)),
        ("more than 2.5 hours", (2.525,)),
    ],
)
def test_range_probes_stay_inside_the_source_range(text, expected):
    from axiom_encode.harness.source_completeness import (
        _formula_interval_from_text,
        _interval_probe_values,
    )

    interval = _formula_interval_from_text(
        text,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )

    assert interval is not None
    assert _interval_probe_values(interval) == expected


def test_principal_output_and_clamp_obligations_name_their_targets():
    content = HEADER.replace(
        "__FORMULA__", "countable_income < income_limit"
    ) + MINOR_RULES.replace(
        "formula: member_age < minor_age_limit",
        "formula: max(0, member_age) < minor_age_limit",
    )

    result = _analyze(content, ELIGIBILITY_CASES)

    assert any(
        issue == f"{TESTS} Principal output `member_is_minor` is never asserted by "
        "a companion test."
        for issue in result.issues
    )
    assert (
        CompanionTestObligation(
            "principal-output", ("member_is_minor",), label="member_is_minor"
        )
        in result.companion_test_obligations
    )


def test_unbound_computation_is_one_finding_that_keeps_the_binding_hint():
    """The tests finding for an unbound computation folds into formula-output."""

    source = (
        "(a) The credit is equal to 10 percent of the earned income of the taxpayer.\n"
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
inputs:
  - name: earned_income
    entity: TaxUnit
    dtype: Money
    period: Year
rules:
  - name: credit_rate
    kind: parameter
    dtype: Rate
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: 10 percent
    versions:
      - effective_from: '2026-01-01'
        formula: 0.1
  - name: credit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: earned_income * credit_rate
"""
    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[
            {
                "name": "credit",
                "period": "2026",
                "input": {"earned_income": 1000},
                "output": {"credit": 100},
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

    [formula_output] = [
        issue
        for issue in result.issues
        if issue.startswith("[complete-source-unit:formula-output]")
    ]
    assert "Explicit source computation (a) formula clause 1" in formula_output
    assert (
        "Asserted principal formula(s) already execute this computation but are "
        "excluded from source-bound evidence: `credit`." in formula_output
    )
    assert not any("do not demonstrate formula branch" in i for i in result.issues)


GUARDED_CLAMP_RULES = """\
  - name: unmet_need_considered
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if household_eligible: max(0, unmet_need)
          else: 0
"""


@pytest.mark.parametrize(
    ("countable_income", "eligible", "credited"),
    [
        # The case takes `else: 0`: a negative input there never reaches the
        # clamp, so it is no evidence for it (accuracy fix on main).
        (900, "not_holds", False),
        (100, "holds", True),
    ],
)
def test_clamp_evidence_must_be_on_the_owner_path(countable_income, eligible, credited):
    content = (
        HEADER.replace("__FORMULA__", "countable_income < income_limit").replace(
            "  - name: member_age\n",
            "  - name: unmet_need\n    entity: Household\n    dtype: Money\n"
            "    period: Month\n  - name: member_age\n",
        )
        + GUARDED_CLAMP_RULES
    )
    case = _case(
        "negative need",
        {"household_eligible": eligible, "unmet_need_considered": "0"},
        countable_income=countable_income,
        unmet_need=-1,
    )
    case["output"]["unmet_need_considered"] = 0

    result = _analyze(content, [*ELIGIBILITY_CASES, case])

    clamp_missing = any(
        "`max(0, unmet_need)` in `unmet_need_considered`" in issue
        for issue in result.issues
    )
    assert clamp_missing is not credited
