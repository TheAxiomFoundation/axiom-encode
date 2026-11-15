"""Run real repair helpers without adding source-boundary expected values."""

from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import evals, validator_pipeline
from tests.test_complete_source_model_cases_engine import (
    TARGET,
    _boundary_payload,
    _declare_engine_pin,
    _model_case,
    _pipeline,
)


def _guarded_wrong_operand():
    source = "(a) A household whose countable income is less than $500 is eligible.\n"
    payload = _boundary_payload(
        source,
        "rent < income_limit and countable_income >= 0",
        extra_inputs=(("rent", "Money"),),
    )
    cases = [
        _model_case("low", 100, "holds", rent=100),
        _model_case("high", 900, "not_holds", rent=900),
    ]
    return source, payload, cases


def _polarity_unknown():
    source = (
        "(a) A household whose countable income is not more than $500 "
        "receives the supplement.\n"
    )
    payload = _boundary_payload(
        source, "countable_income <= income_limit and countable_income != income_limit"
    )
    cases = [_model_case("low", 100, "holds"), _model_case("high", 900, "not_holds")]
    return source, payload, cases


def _cross_boundary():
    source = (
        "(a) A household whose countable income is not more than $500 is eligible "
        "if the household is a resident of the State.\n"
    )
    payload = _boundary_payload(
        source,
        "countable_income <= income_limit and "
        "(household_is_state_resident or household_is_territory_resident)",
        extra_inputs=(
            ("household_is_state_resident", "Boolean"),
            ("household_is_territory_resident", "Boolean"),
        ),
    )
    cases = [
        _model_case(
            "high territory",
            900,
            "not_holds",
            household_is_state_resident=False,
            household_is_territory_resident=True,
        ),
        _model_case(
            "low state",
            100,
            "holds",
            household_is_state_resident=True,
            household_is_territory_resident=False,
        ),
        _model_case(
            "low neither",
            100,
            "not_holds",
            household_is_state_resident=False,
            household_is_territory_resident=False,
        ),
    ]
    return source, payload, cases


def _equal_rent_threshold():
    source = (
        "(a) A household whose countable income is less than $500 and whose "
        "housing cost is low is eligible.\n"
        "(b) Housing cost is low if the rent is less than $500.\n"
    )
    payload = _boundary_payload(
        source,
        "countable_income <= income_limit and housing_cost_low",
        extra_inputs=(("rent", "Money"),),
    )
    root = payload["rules"][-1]
    root["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source.splitlines()[
        0
    ].removeprefix("(a) ")
    rent_limit = copy.deepcopy(payload["rules"][0])
    rent_limit["name"] = "rent_limit"
    rent_limit["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        "rent is less than $500"
    )
    helper = copy.deepcopy(root)
    helper["name"] = "housing_cost_low"
    helper["versions"][0]["formula"] = "rent < rent_limit"
    helper["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source.splitlines()[
        1
    ].removeprefix("(b) ")
    payload["rules"].extend([rent_limit, helper])

    def case(name, income, rent, root_value, helper_value):
        item = _model_case(name, income, root_value, rent=rent)
        item["output"][f"{TARGET}#housing_cost_low"] = helper_value
        return item

    cases = [
        case("low", 100, 100, "holds", "holds"),
        case("high", 900, 900, "not_holds", "not_holds"),
        _model_case(
            "model rent boundary", 100, "not_holds", output="housing_cost_low", rent=500
        ),
        case("rent blocks", 100, 900, "not_holds", "not_holds"),
        case("income blocks", 900, 100, "not_holds", "holds"),
    ]
    return source, payload, cases


def _numeric_r1():
    source = (
        "(a) The benefit of a household whose countable income is not more than $500 "
        "is its unmet need minus its countable income, but not less than zero.\n"
    )
    payload = _boundary_payload(
        source,
        "if countable_income <= income_limit: max(0, unmet_need) - countable_income\nelse: 0",
        extra_inputs=(("unmet_need", "Money"),),
    )
    rule = payload["rules"][-1]
    rule["name"], rule["dtype"], rule["unit"] = "benefit", "Money", "USD"

    def case(name, income, need, value):
        return _model_case(name, income, value, output="benefit", unmet_need=need)

    cases = [
        case("low", 100, 300, 200),
        case("zero", 0, 40, 40),
        case("high", 900, 40, 0),
    ]
    return source, payload, cases


SCENARIOS = {
    "strict_guarded_wrong_operand": _guarded_wrong_operand,
    "r2_p1_2_unknown_polarity": _polarity_unknown,
    "r2_p1_1_cross_boundary": _cross_boundary,
    "r2_p1_3_equal_rent_threshold": _equal_rent_threshold,
    "r1_numeric_clamp": _numeric_r1,
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_real_repair_path_preserves_model_cases_and_source_boundary_demand(
    real_engine, tmp_path, monkeypatch, scenario
):
    source, payload, cases = SCENARIOS[scenario]()
    _declare_engine_pin(tmp_path, real_engine)
    root = tmp_path.resolve() / "rulespec-us" / "us"
    module = root / "statutes" / "7" / "9999.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(yaml.safe_dump(payload, sort_keys=False))
    companion = module.with_name("9999.test.yaml")
    companion.write_text("# Model-stated expected values.\n" + yaml.safe_dump(cases))
    before = (module.read_bytes(), companion.read_bytes())

    # The shared fixture supplies three corpus/proof readers from this source.
    # This fixture has no named release, so it needs no corpus binding scope.
    pipeline = _pipeline(root, real_engine.root, monkeypatch, source)
    monkeypatch.setattr(
        validator_pipeline,
        "_authoritative_corpus_scope",
        lambda release: contextlib.nullcontext(release),
    )

    boundary_label = (
        "(a) formula clause 1=500" if scenario == "r1_numeric_clamp" else "(a)=500"
    )

    def boundary_demands(issues):
        return [
            str(issue)
            for issue in issues
            if f"source-stated boundary input; missing: {boundary_label}" in str(issue)
        ]

    ci = pipeline._run_ci(module)
    assert not ci.passed
    assert boundary_demands(ci.issues)

    # Keep the repair helper real: a restored authoring path must reach these
    # assertions rather than disappear behind a mock of the helper itself.
    repairs = evals._apply_generated_eval_repairs(
        rulespec_file=module,
        policy_repo_root=root,
        axiom_rules_path=real_engine.root,
        issues=list(ci.issues),
        local_corpus_release=object(),
    )
    assert repairs == []
    assert (module.read_bytes(), companion.read_bytes()) == before

    # Adapt only evaluation to real CI on the current bytes. The repair loop
    # and its repair helper remain real; this does not run full artifact eval.
    evaluated_issues = []

    def evaluate(**kwargs):
        result = pipeline._run_ci(kwargs["rulespec_file"])
        evaluated_issues.append(list(result.issues))
        return SimpleNamespace(ci_issues=list(result.issues))

    monkeypatch.setattr(evals, "evaluate_artifact", evaluate)
    evals._evaluate_generated_artifact_with_repairs(
        rulespec_file=module,
        policy_repo_root=root,
        axiom_rules_path=real_engine.root,
        source_text=source,
        local_corpus_release=object(),
        require_complete_source_unit=True,
    )
    assert evaluated_issues
    assert boundary_demands(evaluated_issues[-1])
    assert (module.read_bytes(), companion.read_bytes()) == before
