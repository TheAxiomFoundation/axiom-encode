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


def _case(name: str, **inputs: object) -> dict[str, object]:
    return {
        "name": name,
        "period": "2025-07",
        "input": {f"{BASE}#input.{key}": value for key, value in inputs.items()},
        "output": {
            f"{BASE}#income_considered": 100,
            f"{BASE}#income_limit_met": "holds",
        },
    }


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


def test_obligation_keys_ignore_observed_period_detail():
    first = (
        f"{TESTS} Companion tests do not demonstrate formula branch (B) at us/y. "
        "Exact source computation (characters 1:2): `x`. Observed asserted "
        "candidate-case periods: 2026-01-01."
    )
    second = first.replace("2026-01-01.", "2025-01-01, 2026-01-01.")

    assert companion_obligation_keys([first]) == companion_obligation_keys([second])


def test_has_synthesizable_test_issue_requires_a_supported_tests_demand():
    assert has_synthesizable_test_issue([_boundary_issue("(i)=500")])
    assert not has_synthesizable_test_issue(
        [f"{TESTS} Source-stated exceptions ... require paired positive/blocking"]
    )
    assert not has_synthesizable_test_issue(
        ["[complete-source-unit:structure] Source branch (b) is unencoded."]
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
    assert proposal.output_keys == tuple(case["output"])
    # The source case is never mutated.
    assert case["input"][f"{BASE}#input.income"] == 100


def test_clamp_proposal_uses_a_negative_input_and_asserts_the_clamped_rule():
    obligation = CompanionTestObligation(
        "clamp",
        ("income_considered",),
        values=(-1.0,),
        input_names=("income",),
        label="`max(0, income)` in `income_considered`",
    )
    case = {
        "name": "limit only",
        "period": "2025-07",
        "input": {f"{BASE}#input.income": 300},
        "output": {f"{BASE}#income_limit_met": "holds"},
    }

    proposals = propose_companion_cases(
        [obligation], rules_payload=RULES, test_cases=[case]
    )

    assert [proposal.case["input"] for proposal in proposals] == [
        {f"{BASE}#input.income": -1}
    ]
    assert proposals[0].output_keys == (
        f"{BASE}#income_limit_met",
        f"{BASE}#income_considered",
    )
    assert "minus_1" in proposals[0].case["name"]


def test_principal_output_proposal_asserts_only_that_output_on_same_entity_cases():
    obligation = CompanionTestObligation(
        "principal-output", ("income_considered",), label="income_considered"
    )
    person_case = {
        "name": "person",
        "period": "2025-07",
        "input": {f"{BASE}#input.income": 300},
        "output": {f"{BASE}#income_limit_met": "holds"},
    }
    household_case = {
        "name": "household",
        "period": "2025-07",
        "input": {f"{BASE}#input.total": 3},
        "output": {f"{BASE}#household_total": 3},
    }

    proposals = propose_companion_cases(
        [obligation],
        rules_payload=RULES,
        test_cases=[household_case, person_case],
    )

    assert len(proposals) == 1
    assert proposals[0].case["input"] == person_case["input"]
    assert proposals[0].output_keys == (f"{BASE}#income_considered",)


def test_formula_branch_proposals_flip_selectors_and_move_to_version_periods():
    obligation = CompanionTestObligation(
        "formula-branch",
        ("income_limit_met",),
        toggle_input_names=("eligible",),
        label="(b) at us/x(b)",
    )
    case = _case("base", income=100, eligible=True)

    proposals = propose_companion_cases(
        [obligation], rules_payload=RULES, test_cases=[case]
    )

    flipped = [
        proposal
        for proposal in proposals
        if proposal.case["input"][f"{BASE}#input.eligible"] is False
    ]
    moved = [proposal for proposal in proposals if proposal.case["period"] != "2025-07"]
    assert len(flipped) == 1
    assert flipped[0].case["period"] == "2025-07"
    assert [proposal.case["period"] for proposal in moved] == ["2024-01"]
    assert moved[0].case["input"] == case["input"]


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
        return {key: "not_holds" if "met" in key else 0 for key in output_keys}

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=evaluate,
        analyze_cases=analyze,
    )

    assert len(accepted) == 1
    assert accepted[0]["output"] == {
        f"{BASE}#income_considered": 0,
        f"{BASE}#income_limit_met": "not_holds",
    }
    assert accepted[0]["input"][f"{BASE}#input.income"] == 500


def test_synthesis_rejects_a_case_that_adds_any_other_finding():
    _boundary, case, proposals = _proposal_pair()

    def analyze(cases):
        if len(cases) == 1:
            return [_boundary_issue("(i)=500")]
        return ["[complete-source-unit:tests] Companion case `x` supplies outputs."]

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=lambda _case, keys: dict.fromkeys(keys, 0),
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


def test_synthesis_respects_the_analysis_budget():
    _boundary, case, proposals = _proposal_pair()

    accepted = synthesize_companion_cases(
        test_cases=[case],
        proposals=proposals,
        evaluate_case=lambda _case, keys: dict.fromkeys(keys, 0),
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
