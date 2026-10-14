from copy import deepcopy

import pytest

from axiom_encode.harness import source_completeness as c


def rule(name, formula, dtype="Decimal"):
    return {
        "name": name,
        "kind": "derived",
        "dtype": dtype,
        "versions": [{"formula": formula}],
    }


def fixture():
    rules = {
        "helper": rule("helper", "if applies: amount * 2 else: 0"),
        "result": rule("result", "if applies: helper else: baseline"),
    }
    cases = [
        {
            "name": "ordinary",
            "period": "2025",
            "input": {"applies": False, "amount": 50, "baseline": 200},
            "output": {"result": 200},
        },
        {
            "name": "exception",
            "period": "2025",
            "input": {"applies": True, "amount": 50, "baseline": 200},
            "output": {"helper": 100, "result": 100},
        },
    ]
    return rules, cases


def witnesses(rules, cases):
    return {
        w
        for w in c._toggled_formula_boolean_selectors(
            rules,
            asserted_by_rule={name: cases for name in rules},
            formula_environment={},
        )
        if w.rule_name == "result" and w.selector_name == "applies"
    }


def test_direct_fact_replays_the_newly_reached_corroborated_closure():
    rules, cases = fixture()
    found = witnesses(rules, cases)
    assert {w.active_value for w in found} == {False, True}
    # The ordinary helper is zero, not the new world's 100; no wrong assertion.
    deps = c._case_dependency_environment(
        rules, cases[0], formula_environment={}, require_asserted_value=False
    )
    assert deps["helper"] == 0


def test_multilevel_closure_uses_each_asserted_actual_dependency():
    rules, cases = fixture()
    rules["intermediate"] = rule("intermediate", "helper + 10")
    rules["result"] = rule("result", "if applies: intermediate else: baseline")
    cases[1]["output"].update(intermediate=110, result=110)
    assert {w.active_value for w in witnesses(rules, cases)} == {False, True}


@pytest.mark.parametrize(
    "change",
    [
        "wrong_helper",
        "missing_helper",
        "wrong_result",
        "missing_fact",
        "cycle",
        "second_input",
        "period",
        "missing_period",
        "duplicate_alias",
        "different_entity_key",
    ],
)
def test_unsupported_counterfactual_closure_is_not_a_positive_witness(change):
    rules, cases = fixture()
    if change == "wrong_helper":
        cases[1]["output"]["helper"] = 999
    elif change == "missing_helper":
        cases[1]["output"].pop("helper")
    elif change == "wrong_result":
        cases[1]["output"]["result"] = 999
    elif change == "missing_fact":
        for case in cases:
            case["input"].pop("amount")
    elif change == "cycle":
        rules["helper"] = rule("helper", "result")
    elif change == "second_input":
        cases[1]["input"]["amount"] = 60
        cases[1]["output"].update(helper=120, result=120)
    elif change == "period":
        cases[1]["period"] = "2026"
    elif change == "missing_period":
        for case in cases:
            case.pop("period")
    elif change == "duplicate_alias":
        for case in cases:
            case["input"]["other#input.applies"] = case["input"]["applies"]
    elif change == "different_entity_key":
        cases[1]["input"]["other#input.applies"] = cases[1]["input"].pop("applies")
    assert not any(w.active_value for w in witnesses(rules, cases))


def test_derived_selector_does_not_borrow_independently_changed_amounts():
    rules = {
        "applies": rule("applies", "income > 15", "Judgment"),
        "helper": rule("helper", "income * 2"),
        "result": rule("result", "if applies: helper else: helper"),
    }
    cases = [
        {
            "period": "2025",
            "input": {"income": income},
            "output": {
                "applies": income > 15,
                "helper": income * 2,
                "result": income * 2,
            },
        }
        for income in [10, 20]
    ]
    assert not witnesses(rules, cases)


def test_no_dependency_toggle_is_unchanged():
    rules, cases = fixture()
    rules.pop("helper")
    rules["result"] = rule("result", "if applies: amount else: baseline")
    cases[1]["output"] = {"result": 50}
    assert {w.active_value for w in witnesses(rules, cases)} == {False, True}


def test_shortcut_checks_period_and_input_identity_locally():
    _, cases = fixture()
    kwargs = dict(
        selector_name="applies",
        active_value=True,
        ordinary_dependencies={},
        exception_dependencies={"helper": 100},
    )
    assert c._direct_boolean_intervention_matches_case(*cases, **kwargs)
    altered = deepcopy(cases)
    altered[1]["input"]["amount"] = 60
    assert not c._direct_boolean_intervention_matches_case(*altered, **kwargs)
    altered = deepcopy(cases)
    altered[0]["period"] = {"start": "2025-01-01", "end": "2025-06-30"}
    altered[1]["period"] = {"start": "2025-01-01", "end": "2025-12-31"}
    assert not c._direct_boolean_intervention_matches_case(*altered, **kwargs)


def test_normalized_period_strings_do_not_imply_identical_runtime_coordinates():
    rules, cases = fixture()
    rules["helper"] = rule(
        "helper", "if period_start == period_start: amount * 2 else: 0"
    )
    cases[0]["period"] = "2025"
    cases[1]["period"] = "2025-01"
    assert c._normalized_case_period(cases[0]) == c._normalized_case_period(cases[1])
    assert not any(w.active_value for w in witnesses(rules, cases))
