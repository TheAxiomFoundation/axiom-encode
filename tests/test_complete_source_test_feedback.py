"""Companion-test findings: no duplicate demands, exact boundary repair detail,
and structured obligations for deterministic case synthesis."""

from __future__ import annotations

import functools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import cli as cli_module
from axiom_encode.checks import ValidationIssue, retype
from axiom_encode.harness.source_completeness import (
    MISSING_BOUNDARY_DETAIL_MARKER,
    CompanionTestObligation,
    _expression_reaches_direct_clamp,
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
    # The folded tests demand still counts as residual work for ranking.
    assert formula_output.absorbed_obligations == 1


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


@pytest.mark.parametrize(
    ("formula", "income", "expected", "reached"),
    [
        (
            "if household_eligible and max(0, unmet_need) > 10: 1\nelse: 0",
            900,
            0,
            False,
        ),
        ("if household_eligible and max(0, unmet_need) > 10: 1\nelse: 0", 100, 0, True),
        (
            "if not household_eligible or max(0, unmet_need) > 10: 0\nelse: 1",
            900,
            0,
            False,
        ),
        (
            "if not household_eligible or max(0, unmet_need) > 10: 0\nelse: 1",
            100,
            1,
            True,
        ),
        # RuleSpec refuses chained comparisons; neither path can earn credit.
        ("if 0 > 1 > max(0, unmet_need): 1\nelse: 0", 100, 0, False),
        ("if 0 < 1 < max(0, unmet_need): 1\nelse: 0", 100, 0, False),
        # Reading this input elsewhere does not execute the max call.
        ("if unmet_need < 0 or max(0, unmet_need) > 10: 0\nelse: 1", 100, 0, False),
        (
            "if countable_income < 100.00000000000000000001 or max(0, unmet_need) > 10: 0\n"
            "else: 1",
            100,
            0,
            False,
        ),
        (
            "if household_eligible and max(0, unmet_need) > 10: 1\nelse: unmet_need + 1",
            900,
            0,
            False,
        ),
        ("if opaque_status and max(0, unmet_need) > 10: 1\nelse: 0", 100, 0, False),
    ],
)
def test_clamp_call_itself_must_be_reached(formula, income, expected, reached):
    content = HEADER.replace("__FORMULA__", "countable_income < income_limit").replace(
        "  - name: member_age\n",
        "  - name: unmet_need\n    entity: Household\n    dtype: Money\n"
        "    period: Month\n  - name: member_age\n",
    ) + GUARDED_CLAMP_RULES.replace(
        "if household_eligible: max(0, unmet_need)\n          else: 0",
        formula.replace("\n", "\n          "),
    )
    case = _case(
        "negative need",
        {
            "household_eligible": "holds" if income < 500 else "not_holds",
            "unmet_need_considered": expected,
        },
        countable_income=income,
        unmet_need=-1,
    )
    result = _analyze(content, [*ELIGIBILITY_CASES, case])
    missing = any(
        "`max(0, unmet_need)` in `unmet_need_considered`" in issue
        for issue in result.issues
    )
    assert missing is not reached


@pytest.mark.parametrize(
    "text",
    [
        "opaque(max(0, unmet_need))",
        "unknown + max(0, unmet_need)",
        "unknown and max(0, unmet_need) > 0",
        "max(0, unmet_need) if unknown else 0",
        "0 > 1 > max(0, unmet_need)",
        "0 < 1 < max(0, unmet_need)",
        "not valid RuleSpec syntax!",
    ],
)
def test_opaque_or_unresolved_clamp_expressions_fail_closed(text):
    assert not _expression_reaches_direct_clamp(
        text, input_name="unmet_need", environment={"unmet_need": -1}
    )


_CLAMP_EXPRESSIONS = st.recursive(
    st.sampled_from(
        [
            "gate",
            "other",
            "unmet_need < 0",
            "max(0, unmet_need) > 0",
        ]
    ),
    lambda children: st.one_of(
        st.tuples(children, children).map(lambda pair: f"({pair[0]}) and ({pair[1]})"),
        st.tuples(children, children).map(lambda pair: f"({pair[0]}) or ({pair[1]})"),
        children.map(lambda child: f"not ({child})"),
    ),
    max_leaves=12,
)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(_CLAMP_EXPRESSIONS, st.booleans(), st.booleans(), st.integers(-100, -1))
def test_clamp_reachability_matches_executed_boolean_expressions(
    expression, gate, other, unmet_need
):
    calls = []

    def observed_max(zero, value):
        calls.append((zero, value))
        return max(zero, value)

    environment = {"gate": gate, "other": other, "unmet_need": unmet_need}
    # Closed generated expressions only; observe actual operator evaluation
    # independently of the AST walker, including nested short circuits.
    eval(expression, {"__builtins__": {}}, {**environment, "max": observed_max})
    assert _expression_reaches_direct_clamp(
        expression, input_name="unmet_need", environment=environment
    ) is bool(calls)


RANKING_SOURCE = """\
(a) A household whose countable income is less than $500 is eligible.
(b) A member who is less than 18 years of age is a minor.
(c) A member who works less than 30 hours per week is part-time.
(d) A household whose resources are less than $2,000 is resource-eligible.
(e) A household whose rent is less than $700 is low-rent.
"""

RANKING_INPUTS = """\
  - name: weekly_hours
    entity: Household
    dtype: Integer
    period: Month
  - name: resources
    entity: Household
    dtype: Money
    period: Month
  - name: rent
    entity: Household
    dtype: Money
    period: Month
"""


def _threshold_rules(
    output: str, parameter: str, value: int, excerpt: str, input_name: str
) -> str:
    line = next(text[4:] for text in RANKING_SOURCE.splitlines() if excerpt in text)
    dtype = "Integer" if input_name in {"member_age", "weekly_hours"} else "Money"
    return f"""\
  - name: {parameter}
    kind: parameter
    dtype: {dtype}
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: {excerpt}
    versions:
      - effective_from: '2026-01-01'
        formula: {value}
  - name: {output}
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
              excerpt: {line}
    versions:
      - effective_from: '2026-01-01'
        formula: {input_name} < {parameter}
"""


RANKING_BRANCHES = {
    "b": (
        "member_is_minor",
        "minor_age_limit",
        18,
        "less than 18 years of age",
        "member_age",
    ),
    "c": (
        "part_time",
        "part_time_hours",
        30,
        "less than 30 hours per week",
        "weekly_hours",
    ),
    "d": ("resource_eligible", "resource_limit", 2000, "less than $2,000", "resources"),
    "e": ("low_rent", "rent_limit", 700, "less than $700", "rent"),
}


def _ranking_candidate(*, encoded: str, boundary_tested: str):
    content = HEADER.replace("__FORMULA__", "countable_income < income_limit").replace(
        "rules:\n", RANKING_INPUTS + "rules:\n", 1
    )
    cases = [
        *ELIGIBILITY_CASES,
        _case("income_at", {"household_eligible": "not_holds"}, countable_income=500),
    ]
    for label in encoded:
        output, parameter, value, excerpt, input_name = RANKING_BRANCHES[label]
        content += _threshold_rules(output, parameter, value, excerpt, input_name)
        cases.append(_case(f"{label}_in", {output: "holds"}, **{input_name: 1}))
        cases.append(
            _case(f"{label}_out", {output: "not_holds"}, **{input_name: value * 3})
        )
        if label in boundary_tested:
            cases.append(
                _case(f"{label}_at", {output: "not_holds"}, **{input_name: value})
            )
    result = analyze_complete_source_unit(
        content,
        RANKING_SOURCE,
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
    issues = tuple(result.issues)
    return issues, cli_module._FailedEncodeAttempt(
        result=None,
        error="Generated RuleSpec failed CI validation",
        validation_issues=issues,
        validation_issue_count=len(issues),
        full_validation_issues=issues,
    )


def test_deduplicated_feedback_does_not_outrank_a_more_complete_candidate():
    """A candidate that leaves (b) unencoded must not beat one that encodes
    every branch but misses its boundary cases only because the unencoded
    branch's boundary demand is folded into the structure finding."""

    less_issues, less_complete = _ranking_candidate(
        encoded="cde", boundary_tested="cde"
    )
    more_issues, more_complete = _ranking_candidate(encoded="bcde", boundary_tested="")

    # The feedback stays de-duplicated: (b)=18 is not listed for (b).
    assert any(
        issue.startswith("[complete-source-unit:structure] Source branch (b)")
        for issue in less_issues
    )
    assert not any("(b)=18" in issue for issue in less_issues)
    [boundary] = _boundary_issues_from(more_issues)
    assert _listed(boundary) == ["(b)=18", "(c)=30", "(d)=2000", "(e)=700"]
    # The folded (b)=18 demand still counts on the finding that absorbed it.
    assert sum(getattr(issue, "absorbed_obligations", 0) for issue in less_issues) == 1

    # Residual work as main counted it before the fold: structure (b),
    # control (b), numeric recall 18 and the folded (b)=18, against four
    # missing boundary items. Without the folded demand the less complete
    # candidate counts 3 and wins; with it the tie goes to fewer messages.
    count = cli_module._validation_residual_obligation_count
    assert count(less_issues) == 4
    assert count(more_issues) == 4
    assert len(less_issues) > len(more_issues)
    for order in ((less_complete, more_complete), (more_complete, less_complete)):
        assert cli_module._best_validation_retry_attempt(order) is more_complete


def _boundary_issues_from(issues) -> list[str]:
    return [issue for issue in issues if "boundary input; missing:" in issue]


def test_absorbed_obligations_survive_rewrites_and_pickling():
    import pickle

    issue = ValidationIssue(
        "[complete-source-unit:structure] Source branch (b) is unencoded.",
        check="complete-source-unit.structure.branch-unencoded",
        absorbed_obligations=2,
    )
    prefixed = retype(issue, f"rules.yaml: ci: {issue}")

    assert prefixed.absorbed_obligations == 2
    assert pickle.loads(pickle.dumps(prefixed)).absorbed_obligations == 2
    assert cli_module._validation_residual_obligation_count((prefixed,)) == 3
    # Equality and the typed document stay the message's.
    assert prefixed == f"rules.yaml: ci: {issue}"
