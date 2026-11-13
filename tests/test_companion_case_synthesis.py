"""Deterministic companion-case synthesis for complete-source test demands."""

from __future__ import annotations

import pytest

from axiom_encode.harness.companion_case_synthesis import (
    SYNTHESIZED_CASE_PREFIX,
    companion_obligation_keys,
    expected_value_from_engine_output,
    has_synthesizable_test_issue,
    propose_companion_cases,
    synthesize_companion_cases,
)
from axiom_encode.harness.source_completeness import (
    MISSING_BOUNDARY_DETAIL_MARKER,
    CompanionTestObligation,
)

BASE = "us:statutes/7/2015/f"
TESTS = "[complete-source-unit:tests]"

RULES = {
    "format": "rulespec/v1",
    "rules": [
        {
            "name": "income_considered",
            "kind": "derived",
            "entity": "Person",
            "dtype": "Money",
            "versions": [
                {"effective_from": "2025-07-01", "formula": "max(0, income)"},
            ],
        },
        {
            "name": "income_limit_met",
            "kind": "derived",
            "entity": "Person",
            "dtype": "Judgment",
            "versions": [
                {"effective_from": "2024-01-01", "formula": "income <= limit"},
                {"effective_from": "2025-07-01", "formula": "income < limit"},
            ],
        },
        {
            "name": "household_total",
            "kind": "derived",
            "entity": "Household",
            "dtype": "Money",
            "versions": [{"effective_from": "2025-07-01", "formula": "total"}],
        },
    ],
}


def _engine(case: dict, keys: tuple[str, ...]) -> dict[str, object]:
    """A stand-in for the rules engine that executes RULES faithfully."""

    income = case["input"].get(f"{BASE}#input.income", 0)
    values = {
        "income_considered": max(0, income),
        "income_limit_met": "holds" if income < 500 else "not_holds",
        "household_total": case["input"].get(f"{BASE}#input.total", 0),
    }
    return {key: values[key.rsplit("#", 1)[-1]] for key in keys}


def _case(name: str, **inputs: object) -> dict[str, object]:
    case = {
        "name": name,
        "period": "2025-07",
        "input": {f"{BASE}#input.{key}": value for key, value in inputs.items()},
    }
    case["output"] = _engine(
        case, (f"{BASE}#income_considered", f"{BASE}#income_limit_met")
    )
    return case


def _boundary_issue(*items: str) -> str:
    return (
        f"{TESTS} Companion tests do not exercise every source-stated boundary "
        f"input; missing: {', '.join(items)}.{MISSING_BOUNDARY_DETAIL_MARKER} "
        "(i)=500 at us/x(i); source `at least $500, more, text`."
    )


def test_obligation_keys_split_grouped_findings_into_demands():
    issues = [
        _boundary_issue("(i)=500", "12. formula clause 61=500"),
        f"{TESTS} Companion tests do not demonstrate formula branch (A) formula "
        "clause 4 at us/x(b) [Absatz b]. Exact source computation (characters "
        "1:2): `x`. Observed asserted candidate-case periods: 2026-01-01.",
        f"{TESTS} Direct nonnegative clamp companion evidence is missing for "
        "`max(0, a)` in `r`, `max(0, b)` in `s`. Each requires ...",
        f"{TESTS} Principal output `shelter_costs` is never asserted by a "
        "companion test.",
        f"{TESTS} Source-stated exceptions or applicability conditions require "
        "paired positive/blocking cases ...; missing: us/x(1) [Absatz 1] "
        "[change]: `one`; us/x(2) [Absatz 2] [block]: `two`. The evaluator "
        "recognized nothing.",
        "[complete-source-unit:structure] Source branch (b) is unencoded.",
    ]

    keys = companion_obligation_keys(issues)

    assert ("boundary", "(i)=500") in keys
    assert ("boundary", "12. formula clause 61=500") in keys
    assert ("formula-branch", "(A) formula clause 4 at us/x(b) [Absatz b]") in keys
    assert ("clamp", "`max(0, a)` in `r`") in keys
    assert ("clamp", "`max(0, b)` in `s`") in keys
    assert ("principal-output", "shelter_costs") in keys
    assert sum(1 for key in keys if key[0] == "paired-condition") == 2
    assert ("issue", issues[-1]) in keys
    # The details after the marker never become demands.
    assert not any("source `" in key[1] for key in keys if key[0] == "boundary")


def test_clamp_keys_come_from_complete_obligations_when_the_list_is_truncated():
    shown = ", ".join(f"`max(0, x{index})` in `r{index}`" for index in range(8))
    issue = (
        f"{TESTS} Direct nonnegative clamp companion evidence is missing for "
        f"{shown} (2 additional clamps omitted). Each requires ..."
    )
    obligations = [
        CompanionTestObligation(
            "clamp",
            (f"r{index}",),
            values=(-1.0,),
            input_names=(f"x{index}",),
            label=f"`max(0, x{index})` in `r{index}`",
        )
        for index in range(10)
    ]

    keys = companion_obligation_keys([issue], obligations)
    after_one_fix = companion_obligation_keys([issue], obligations[1:])

    assert keys == {("clamp", obligation.label) for obligation in obligations}
    # Covering one clamp is a strict reduction even though the shown list and
    # the omitted count both change.
    assert after_one_fix < keys


def test_obligation_keys_ignore_observed_period_detail():
    first = (
        f"{TESTS} Companion tests do not demonstrate formula branch (B) at us/y. "
        "Exact source computation (characters 1:2): `x`. Observed asserted "
        "candidate-case periods: 2026-01-01."
    )
    second = first.replace("2026-01-01.", "2025-01-01, 2026-01-01.")

    assert companion_obligation_keys([first]) == companion_obligation_keys([second])


@pytest.mark.parametrize(
    "issue",
    [
        f"{TESTS} Source-stated exceptions ... require paired positive/blocking",
        f"{TESTS} Companion tests do not demonstrate formula branch (b) at us/x(b). "
        "Exact source computation (characters 1:2): `x`.",
        f"{TESTS} Direct nonnegative clamp companion evidence is missing for "
        "`max(0, a)` in `r`.",
        f"{TESTS} Principal output `r` is never asserted by a companion test.",
        "[complete-source-unit:structure] Source branch (b) is unencoded.",
    ],
)
def test_only_boundary_demands_are_synthesizable(issue):
    """Branch, clamp, output and exception demands exist so the model states a
    value; an engine-computed expectation would pin the encoding (A1, A13)."""

    assert has_synthesizable_test_issue([_boundary_issue("(i)=500")])
    assert not has_synthesizable_test_issue([issue])


@pytest.mark.parametrize(
    "obligation",
    [
        CompanionTestObligation(
            "clamp",
            ("income_considered",),
            values=(-1.0,),
            input_names=("income",),
            label="`max(0, income)` in `income_considered`",
        ),
        CompanionTestObligation(
            "formula-branch",
            ("income_limit_met",),
            values=(600.0,),
            input_names=("income",),
            label="(b) at us/x(b)",
        ),
        CompanionTestObligation(
            "principal-output", ("income_considered",), label="income_considered"
        ),
    ],
)
def test_non_boundary_obligations_are_never_proposed(obligation):
    assert (
        propose_companion_cases(
            [obligation],
            rules_payload=RULES,
            test_cases=[_case("base", income=300, eligible=True)],
        )
        == []
    )


def test_boundary_proposal_changes_only_the_directly_compared_input():
    obligation = CompanionTestObligation(
        "boundary",
        ("income_limit_met",),
        values=(500.0,),
        input_names=("income",),
        label="(i)=500",
    )
    case = _case("base", income=100, other=7)

    proposals = propose_companion_cases(
        [obligation], rules_payload=RULES, test_cases=[case]
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.case["name"].startswith(SYNTHESIZED_CASE_PREFIX)
    assert proposal.case["input"] == {
        f"{BASE}#input.income": 500,
        f"{BASE}#input.other": 7,
    }
    assert proposal.case["period"] == "2025-07"
    assert "output" not in proposal.case
    # Only the targeted rule (and what it reads) is re-asserted, but every
    # same-entity value the model stated in the source case must reproduce.
    assert proposal.output_keys == (f"{BASE}#income_limit_met",)
    assert proposal.source_case is case
    assert proposal.source_output_keys == tuple(case["output"])
    # The source case is never mutated.
    assert case["input"][f"{BASE}#input.income"] == 100


def test_clones_assert_only_outputs_on_the_covering_rule_entity():
    obligation = CompanionTestObligation(
        "boundary",
        ("income_limit_met",),
        values=(500.0,),
        input_names=("income",),
        label="(i)=500",
    )
    mixed = {
        "name": "mixed entities",
        "period": "2025-07",
        "input": {f"{BASE}#input.income": 100},
        "output": {
            f"{BASE}#household_total": 3,
            f"{BASE}#income_limit_met": "holds",
            f"{BASE}#some_parameter": 7,
        },
    }

    [proposal] = propose_companion_cases(
        [obligation], rules_payload=RULES, test_cases=[mixed]
    )

    assert proposal.output_keys == (f"{BASE}#income_limit_met",)


def test_unusable_cases_are_never_cloned():
    obligation = CompanionTestObligation(
        "boundary",
        ("income_limit_met",),
        values=(500.0,),
        input_names=("income",),
        label="(i)=500",
    )
    tables_case = {**_case("tables", income=100), "tables": {"Person": []}}
    list_case = {
        **_case("rows", income=100),
        "output": {f"{BASE}#income_limit_met": ["holds"]},
    }

    assert (
        propose_companion_cases(
            [obligation], rules_payload=RULES, test_cases=[tables_case, list_case]
        )
        == []
    )


def _proposal_pair():
    boundary = CompanionTestObligation(
        "boundary",
        ("income_limit_met",),
        values=(500.0,),
        input_names=("income",),
        label="(i)=500",
    )
    case = _case("base", income=100)
    return (
        boundary,
        case,
        propose_companion_cases([boundary], rules_payload=RULES, test_cases=[case]),
    )


def test_synthesis_keeps_only_executed_cases_that_shrink_demands():
    _boundary, case, proposals = _proposal_pair()
    analyzed: list[list[dict]] = []

    def analyze(cases):
        analyzed.append(cases)
        at_boundary = any(
            item["input"].get(f"{BASE}#input.income") == 500 for item in cases
        )
        return [] if at_boundary else [_boundary_issue("(i)=500")]

    def evaluate(proposed_case, output_keys):
        assert "output" not in proposed_case
        return _engine(proposed_case, output_keys)

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=evaluate,
        analyze_cases=analyze,
    )

    assert len(accepted) == 1
    assert accepted[0]["output"] == {f"{BASE}#income_limit_met": "not_holds"}
    assert accepted[0]["input"][f"{BASE}#input.income"] == 500


def test_no_clone_of_a_case_whose_stated_expectation_the_engine_rejects():
    """If the model's own case disagrees with execution, its clone would pin
    the encoding's side of that conflict, so nothing is synthesized."""

    _boundary, case, proposals = _proposal_pair()
    case["output"][f"{BASE}#income_limit_met"] = "not_holds"

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=lambda cases: (
            [_boundary_issue("(i)=500")] if len(cases) == 1 else []
        ),
    )

    assert accepted == []


def test_source_agreement_covers_outputs_the_clone_does_not_reassert():
    """`income_considered` is outside the clone's keys, but the model stated
    it in the source case; a conflict there still blocks every clone."""

    _boundary, case, proposals = _proposal_pair()
    case["output"][f"{BASE}#income_considered"] = 999

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=lambda cases: (
            [_boundary_issue("(i)=500")] if len(cases) == 1 else []
        ),
    )

    assert accepted == []


def test_source_agreement_fails_closed_without_stated_values():
    from axiom_encode.harness.companion_case_synthesis import (
        CompanionCaseProposal,
        _source_case_agrees_with_engine,
    )

    boundary, case, _proposals = _proposal_pair()
    proposal = CompanionCaseProposal(
        boundary, dict(case), (f"{BASE}#income_limit_met",), source_case=case
    )

    assert not _source_case_agrees_with_engine(proposal, _engine)


def test_synthesis_rejects_a_case_that_adds_any_other_finding():
    _boundary, case, proposals = _proposal_pair()

    def analyze(cases):
        if len(cases) == 1:
            return [_boundary_issue("(i)=500")]
        return ["[complete-source-unit:tests] Companion case `x` supplies outputs."]

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=analyze,
    )

    assert accepted == []


@pytest.mark.parametrize(
    "evaluated",
    [None, {}, {f"{BASE}#income_considered": 0}],
)
def test_synthesis_never_invents_values_the_engine_did_not_return(evaluated):
    _boundary, case, proposals = _proposal_pair()
    calls = []

    def analyze(cases):
        calls.append(len(cases))
        return [_boundary_issue("(i)=500")] if len(cases) == 1 else []

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=lambda _case, _keys: evaluated,
        analyze_cases=analyze,
    )

    assert accepted == []
    assert calls == [1]


def test_synthesis_credits_only_the_targeted_demand():
    """A clone that closes some other demand (here a paired condition) is not
    credit for its own target, even though the findings strictly shrink."""

    _boundary, case, proposals = _proposal_pair()
    paired = (
        f"{TESTS} Source-stated exceptions or applicability conditions require "
        "paired positive/blocking cases; missing: us/x(a) [change]: `x`."
    )

    def analyze(cases):
        if len(cases) == 1:
            return [_boundary_issue("(i)=500"), paired]
        return [_boundary_issue("(i)=500")]

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=analyze,
    )

    assert accepted == []


def test_synthesis_never_closes_an_unasserted_output_on_the_way():
    _boundary, case, proposals = _proposal_pair()
    never = f"{TESTS} Principal output `household_total` is never asserted by a companion test."

    def analyze(cases):
        if len(cases) == 1:
            return [_boundary_issue("(i)=500"), never]
        return []

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=analyze,
    )

    assert accepted == []


def test_rejected_proposals_do_not_consume_case_names():
    _boundary, case, proposals = _proposal_pair()
    duplicate = [proposals[0], proposals[0]]
    calls = []

    def analyze(cases):
        calls.append(len(cases))
        if len(cases) == 1:
            return [_boundary_issue("(i)=500")]
        return [] if len(calls) > 2 else [_boundary_issue("(i)=500"), "new finding"]

    [accepted] = synthesize_companion_cases(
        test_cases=[case],
        proposals=duplicate,
        evaluate_case=_engine,
        analyze_cases=analyze,
    )

    assert accepted["name"] == "auto_source_boundary_income_500"


def test_synthesis_respects_the_analysis_budget():
    _boundary, case, proposals = _proposal_pair()

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=_engine,
        analyze_cases=lambda cases: [_boundary_issue("(i)=500")],
        max_analyses=0,
    )

    assert accepted == []


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"kind": "judgment", "outcome": "holds"}, "holds"),
        ({"kind": "judgment", "outcome": "not_holds"}, "not_holds"),
        ({"kind": "judgment", "outcome": "undetermined"}, None),
        ({"kind": "scalar", "value": {"kind": "bool", "value": True}}, True),
        ({"kind": "scalar", "value": {"kind": "integer", "value": 42}}, 42),
        ({"kind": "scalar", "value": {"kind": "decimal", "value": "250.00"}}, 250),
        ({"kind": "scalar", "value": {"kind": "decimal", "value": "0.15"}}, 0.15),
        (
            {"kind": "scalar", "value": {"kind": "decimal", "value": "0.1000000001"}},
            0.1000000001,
        ),
        (
            {
                "kind": "scalar",
                "value": {"kind": "decimal", "value": "1.00000000000000000001"},
            },
            None,
        ),
        ({"kind": "scalar", "value": {"kind": "text", "value": "x"}}, None),
        ({"kind": "scalar", "value": {"kind": "integer", "value": True}}, None),
        ("holds", None),
    ],
)
def test_engine_outputs_render_as_exact_expected_values(output, expected):
    assert expected_value_from_engine_output(output) == expected
