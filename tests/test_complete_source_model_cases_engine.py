"""Model-authored expected values remain authoritative in real-engine checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

CITATION = "us/statute/7/9999"


SOURCE = (
    "(a) A household whose countable income is less than $500 is eligible.\n"
    "(b) The countable shortfall is the household's unmet need, but not less "
    "than zero.\n"
)


MODULE = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CITATION}
inputs:
  - name: countable_income
    entity: Household
    dtype: Money
    period: Month
  - name: unmet_need
    entity: Household
    dtype: Money
    period: Month
rules:
  - name: income_limit
    kind: parameter
    dtype: Money
    unit: USD
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
        formula: '500'
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
        formula: countable_income < income_limit
  - name: countable_shortfall
    kind: derived
    entity: Household
    dtype: Money
    unit: USD
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: >-
                The countable shortfall is the household's unmet need, but not
                less than zero.
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, unmet_need)
"""


TARGET = "us:statutes/7/9999"


TESTS = [
    {
        "name": "low income",
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": 100,
            f"{TARGET}#input.unmet_need": 40,
        },
        "output": {
            f"{TARGET}#household_eligible": "holds",
            f"{TARGET}#countable_shortfall": 40,
        },
    },
    {
        "name": "high income",
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": 900,
            f"{TARGET}#input.unmet_need": 0,
        },
        "output": {
            f"{TARGET}#household_eligible": "not_holds",
            f"{TARGET}#countable_shortfall": 0,
        },
    },
]


def _pipeline(
    root: Path, engine_root: Path, monkeypatch, source: str = SOURCE
) -> ValidatorPipeline:
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        source_citation_path=CITATION,
    )
    monkeypatch.setattr(
        pipeline,
        "_source_texts_for_rulespec_content",
        lambda _content: {CITATION: source},
    )
    monkeypatch.setattr(
        pipeline,
        "_proof_source_texts_for_rulespec_content",
        lambda _content, source_texts: dict(source_texts or {}),
    )
    # Production binds a local corpus release around synthesis; serve the
    # fixture body the same way to the static companion checks.
    monkeypatch.setattr(
        validator_pipeline,
        "_fetch_corpus_source_text",
        lambda citation_path: source if citation_path == CITATION else None,
    )
    return pipeline


def _tests_issues(
    pipeline: ValidatorPipeline, module: Path, cases, source: str = SOURCE
) -> list[str]:
    return [
        issue
        for issue in pipeline._complete_source_unit_issues(
            module.read_text(),
            validation_source_texts={CITATION: source},
            test_cases=cases,
            rules_file=module,
        )
        if issue.startswith("[complete-source-unit:tests]")
    ]


def _declare_engine_pin(tmp_path, real_engine):
    receipt = json.loads(
        real_engine.binary.with_name("axiom-rules-engine.provenance.json").read_text()
    )
    assert receipt["binary_sha256"] == real_engine.sha256
    config = tmp_path / "rulespec-us" / ".axiom" / "toolchain.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f'[toolchain]\naxiom_rules_engine_ref = "{receipt["engine_commit"]}"\n'
    )


def _model_case(name, income, expected, *, output="household_eligible", **inputs):
    return {
        "name": name,
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": income,
            **{f"{TARGET}#input.{key}": value for key, value in inputs.items()},
        },
        "output": {f"{TARGET}#{output}": expected},
    }


def _boundary_payload(source, formula, *, extra_inputs=()):
    payload = yaml.safe_load(MODULE)
    payload["inputs"] = payload["inputs"][:1] + [
        {"name": name, "entity": "Household", "dtype": dtype, "period": "Month"}
        for name, dtype in extra_inputs
    ]
    payload["rules"] = payload["rules"][:2]
    payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        "not more than $500" if "not more than" in source else "less than $500"
    )
    payload["rules"][-1]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        source.removeprefix("(a) ").strip()
    )
    payload["rules"][-1]["versions"][0]["formula"] = formula
    return payload


def _validate_model_cases(tmp_path, real_engine, monkeypatch, payload, cases, source):
    _declare_engine_pin(tmp_path, real_engine)
    root = tmp_path / "rulespec-us" / "us"
    module = root / "statutes" / "7" / "9999.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(yaml.safe_dump(payload, sort_keys=False))
    companion = module.with_name("9999.test.yaml")
    companion.write_text("# Model-stated expected values.\n" + yaml.safe_dump(cases))
    before = (module.read_bytes(), companion.read_bytes())
    pipeline = _pipeline(root, real_engine.root, monkeypatch, source)
    compiled_path = tmp_path / "compiled.json"
    result, compiled = pipeline._compile_rulespec_to_artifact(module, compiled_path)
    assert result.returncode == 0, result.stderr
    execution_issues = pipeline._run_rulespec_test_cases(
        rules_file=module,
        compiled_path=compiled_path,
        compiled_payload=compiled,
        cases=cases,
    )
    source_issues = pipeline._complete_source_unit_issues(
        module.read_text(),
        validation_source_texts={CITATION: source},
        test_cases=cases,
        rules_file=module,
    )
    assert (module.read_bytes(), companion.read_bytes()) == before
    return execution_issues, source_issues


def _missing_boundary(issues):
    return any("boundary input; missing: (a)=500" in issue for issue in issues)


def _assert_model_boundary_mismatch(issues, name="model source boundary"):
    assert any(
        f"Test case `{name}` output `" in issue and " expected " in issue
        for issue in issues
    ), issues


def test_wrong_operand_stays_unresolved_until_model_supplies_source_boundary(
    real_engine, tmp_path, monkeypatch
):
    source = "(a) A household whose countable income is less than $500 is eligible.\n"
    wrong = _boundary_payload(
        source,
        "rent < income_limit and countable_income >= 0",
        extra_inputs=(("rent", "Money"),),
    )
    cases = [
        _model_case("low", 100, "holds", rent=100),
        _model_case("high", 900, "not_holds", rent=900),
    ]
    executed, issues = _validate_model_cases(
        tmp_path / "original", real_engine, monkeypatch, wrong, cases, source
    )
    assert executed == []
    assert _missing_boundary(issues)
    boundary = _model_case("model source boundary", 500, "not_holds", rent=100)
    executed, _issues = _validate_model_cases(
        tmp_path / "model_boundary",
        real_engine,
        monkeypatch,
        wrong,
        [*cases, boundary],
        source,
    )
    _assert_model_boundary_mismatch(executed)
    correct = _boundary_payload(source, "countable_income < income_limit")
    correct_cases = [
        {
            **case,
            "input": {
                f"{TARGET}#input.countable_income": case["input"][
                    f"{TARGET}#input.countable_income"
                ]
            },
        }
        for case in [*cases, boundary]
    ]
    executed, issues = _validate_model_cases(
        tmp_path / "correct", real_engine, monkeypatch, correct, correct_cases, source
    )
    assert executed == []
    assert issues == []


@pytest.mark.parametrize("wording", ["is eligible", "receives the supplement"])
def test_boolean_boundary_demands_need_model_stated_expectations(
    real_engine, tmp_path, monkeypatch, wording
):
    source = (
        f"(a) A household whose countable income is not more than $500 {wording}.\n"
    )
    wrong = _boundary_payload(
        source, "countable_income <= income_limit and countable_income != income_limit"
    )
    cases = [_model_case("low", 100, "holds"), _model_case("high", 900, "not_holds")]
    executed, issues = _validate_model_cases(
        tmp_path / "original", real_engine, monkeypatch, wrong, cases, source
    )
    assert executed == []
    assert _missing_boundary(issues)
    boundary = _model_case("model source boundary", 500, "holds")
    executed, _issues = _validate_model_cases(
        tmp_path / "model_boundary",
        real_engine,
        monkeypatch,
        wrong,
        [*cases, boundary],
        source,
    )
    _assert_model_boundary_mismatch(executed, "model source boundary")
    correct = _boundary_payload(source, "countable_income <= income_limit")
    executed, issues = _validate_model_cases(
        tmp_path / "correct",
        real_engine,
        monkeypatch,
        correct,
        [*cases, boundary],
        source,
    )
    assert executed == []
    assert issues == []


def test_cross_boundary_boolean_logic_needs_a_model_boundary_case(
    real_engine, tmp_path, monkeypatch
):
    source = (
        "(a) A household whose countable income is not more than $500 is eligible "
        "if the household is a resident of the State.\n"
    )
    wrong = _boundary_payload(
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
    executed, issues = _validate_model_cases(
        tmp_path / "original", real_engine, monkeypatch, wrong, cases, source
    )
    assert executed == []
    assert _missing_boundary(issues)
    boundary = _model_case(
        "model state-residency boundary",
        500,
        "not_holds",
        household_is_state_resident=False,
        household_is_territory_resident=True,
    )
    executed, _issues = _validate_model_cases(
        tmp_path / "model_boundary",
        real_engine,
        monkeypatch,
        wrong,
        [*cases, boundary],
        source,
    )
    _assert_model_boundary_mismatch(executed, "model state-residency boundary")

    correct = _boundary_payload(
        source,
        "countable_income <= income_limit and household_is_state_resident",
        extra_inputs=(("household_is_state_resident", "Boolean"),),
    )
    correct_cases = [
        _model_case("low state", 100, "holds", household_is_state_resident=True),
        _model_case("low neither", 100, "not_holds", household_is_state_resident=False),
        _model_case("high state", 900, "not_holds", household_is_state_resident=True),
        _model_case(
            "model source boundary", 500, "holds", household_is_state_resident=True
        ),
        _model_case(
            "model nonresident boundary",
            500,
            "not_holds",
            household_is_state_resident=False,
        ),
    ]
    executed, issues = _validate_model_cases(
        tmp_path / "correct", real_engine, monkeypatch, correct, correct_cases, source
    )
    assert executed == []
    assert issues == []


def test_equal_rent_threshold_does_not_create_an_income_boundary_case(
    real_engine, tmp_path, monkeypatch
):
    source = (
        "(a) A household whose countable income is less than $500 and whose "
        "housing cost is low is eligible.\n"
        "(b) Housing cost is low if the rent is less than $500.\n"
    )
    wrong = _boundary_payload(
        source,
        "countable_income <= income_limit and housing_cost_low",
        extra_inputs=(("rent", "Money"),),
    )
    root = wrong["rules"][-1]
    root["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = source.splitlines()[
        0
    ].removeprefix("(a) ")
    rent_limit = copy.deepcopy(wrong["rules"][0])
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
    wrong["rules"].extend([rent_limit, helper])

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
    executed, issues = _validate_model_cases(
        tmp_path / "original", real_engine, monkeypatch, wrong, cases, source
    )
    assert executed == []
    assert _missing_boundary(issues)
    boundary = case("model income boundary", 500, 100, "not_holds", "holds")
    executed, _issues = _validate_model_cases(
        tmp_path / "model_boundary",
        real_engine,
        monkeypatch,
        wrong,
        [*cases, boundary],
        source,
    )
    _assert_model_boundary_mismatch(executed, "model income boundary")
    correct = copy.deepcopy(wrong)
    correct["rules"][1]["versions"][0]["formula"] = (
        "countable_income < income_limit and housing_cost_low"
    )
    executed, issues = _validate_model_cases(
        tmp_path / "correct",
        real_engine,
        monkeypatch,
        correct,
        [*cases, boundary],
        source,
    )
    assert executed == []
    assert issues == []


@pytest.mark.parametrize("subtrahend", ["countable_income", "household_contribution"])
def test_numeric_boundary_cannot_gain_an_engine_authored_expected_value(
    real_engine, tmp_path, monkeypatch, subtrahend
):
    source = (
        "(a) The benefit of a household whose countable income is not more than $500 "
        f"is its unmet need minus its {subtrahend.replace('_', ' ')}, but not less than zero.\n"
    )
    wrong = _boundary_payload(
        source,
        f"if countable_income <= income_limit: max(0, unmet_need) - {subtrahend}\nelse: 0",
        extra_inputs=(("unmet_need", "Money"),)
        + (
            (("household_contribution", "Money"),)
            if subtrahend == "household_contribution"
            else ()
        ),
    )
    rule = wrong["rules"][-1]
    rule["name"], rule["dtype"], rule["unit"] = "benefit", "Money", "USD"

    def case(name, income, need, value, contribution=0):
        inputs = {"unmet_need": need}
        if subtrahend == "household_contribution":
            inputs["household_contribution"] = contribution
        return _model_case(name, income, value, output="benefit", **inputs)

    cases = (
        [case("low", 100, 300, 200), case("zero", 0, 40, 40), case("high", 900, 40, 0)]
        if subtrahend == "countable_income"
        else [
            case("low", 100, 300, 200, 100),
            case("zero", 0, 40, 40),
            case("high", 900, 40, 0),
        ]
    )
    executed, issues = _validate_model_cases(
        tmp_path / "original", real_engine, monkeypatch, wrong, cases, source
    )
    assert executed == []
    assert any(
        "boundary input; missing:" in issue and "=500" in issue and "`benefit`" in issue
        for issue in issues
    )
    boundary = case("model source boundary", 500, 40, 0, 100)
    executed, _issues = _validate_model_cases(
        tmp_path / "model_boundary",
        real_engine,
        monkeypatch,
        wrong,
        [*cases, boundary],
        source,
    )
    _assert_model_boundary_mismatch(executed, "model source boundary")


@pytest.mark.parametrize(
    ("formula", "income", "expected", "clamp_reached", "input_read"),
    [
        (
            "if household_eligible and max(0, unmet_need) > 10: 1\nelse: 0",
            900,
            0,
            False,
            False,
        ),
        (
            "if household_eligible and max(0, unmet_need) > 10: 1\nelse: 0",
            100,
            0,
            True,
            True,
        ),
        (
            "if not household_eligible or max(0, unmet_need) > 10: 0\nelse: 1",
            900,
            0,
            False,
            False,
        ),
        (
            "if not household_eligible or max(0, unmet_need) > 10: 0\nelse: 1",
            100,
            1,
            True,
            True,
        ),
        (
            "if unmet_need < 0 or max(0, unmet_need) > 10: 0\nelse: 1",
            100,
            0,
            False,
            True,
        ),
        (
            "if countable_income < 100.00000000000000000001 or max(0, unmet_need) > 10: 0\n"
            "else: 1",
            100,
            0,
            False,
            False,
        ),
        (
            "if household_eligible and max(0, unmet_need) > 10: 1\nelse: unmet_need + 1",
            900,
            0,
            False,
            True,
        ),
    ],
)
def test_clamp_credit_follows_actual_short_circuit_execution(
    real_engine,
    tmp_path,
    monkeypatch,
    formula,
    income,
    expected,
    clamp_reached,
    input_read,
):
    _declare_engine_pin(tmp_path, real_engine)
    root = tmp_path.resolve() / "rulespec-us" / "us"
    module = root / "statutes/7/9999.yaml"
    module.parent.mkdir(parents=True)
    payload = yaml.safe_load(MODULE)
    payload["rules"][-1]["versions"][0]["formula"] = formula
    module.write_text(yaml.safe_dump(payload, sort_keys=False))
    case = {
        "name": "negative shortfall",
        "period": "2026-01",
        "input": {
            f"{TARGET}#input.countable_income": income,
            f"{TARGET}#input.unmet_need": -1,
        },
        "output": {
            f"{TARGET}#household_eligible": "holds" if income < 500 else "not_holds",
            f"{TARGET}#countable_shortfall": expected,
        },
    }
    pipeline = _pipeline(root, real_engine.root, monkeypatch)
    compiled_path = tmp_path / "compiled.json"
    compiled_result, compiled = pipeline._compile_rulespec_to_artifact(
        module, compiled_path
    )
    assert compiled_result.returncode == 0, compiled_result.stderr

    def execute(proposed):
        return pipeline._run_rulespec_test_cases(
            rules_file=module,
            compiled_path=compiled_path,
            compiled_payload=compiled,
            cases=[proposed],
        )

    assert execute(case) == []
    absent = {
        **case,
        "input": {
            key: value
            for key, value in case["input"].items()
            if key != f"{TARGET}#input.unmet_need"
        },
    }
    assert bool(execute(absent)) is input_read
    missing = any(
        "`max(0, unmet_need)` in `countable_shortfall`" in issue
        for issue in _tests_issues(pipeline, module, [case])
    )
    assert missing is not clamp_reached
