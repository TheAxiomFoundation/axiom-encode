"""Historical actual failures validate a separate, non-authoritative record type."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_capital_gift_operands import bind_capital_gift
from axiom_encode.harness.capital_gift_execution import (
    ENGINE_PIN,
    _compiled,
    _direct_refusal,
    _record,
    direct_refusal_pairs,
)

F = Path(__file__).parent / "fixtures"
STATIC = json.loads((F / "t1170-static.json").read_text())
EXECUTION = json.loads((F / "t1170-execution.json").read_text())
REFUSALS = json.loads((F / "t1170-direct-refusals.json").read_text())


@pytest.fixture(scope="module")
def bound():
    result = bind_capital_gift(
        STATIC["policy_yaml"],
        (F / "t1170-authenticated-body.txt").read_text(),
        STATIC["p113_body"],
    )
    assert result is not None
    return result


@pytest.mark.parametrize("index", range(18))
def test_actual_direct_failure_has_independently_false_selector(bound, index):
    case, call = REFUSALS["cases"][index], REFUSALS["calls"][index]
    result = _direct_refusal(bound, EXECUTION["artifact"], ENGINE_PIN, case, call)
    assert result.root == case["entity_id"]
    assert result.principal == case["expected_refusal"]["output"].split("#")[1]
    assert not hasattr(result, "outputs")


@pytest.mark.parametrize(
    "field,value",
    [
        ("returncode", 0),
        ("returncode", 2),
        ("returncode", True),
        ("stdout", "{}"),
        ("stderr", "division by zero\n"),
        ("stderr", "missing input\n"),
        ("stderr", ""),
    ],
)
def test_arbitrary_process_failure_is_not_refusal(bound, field, value):
    call = copy.deepcopy(REFUSALS["calls"][0])
    call[field] = value
    with pytest.raises(ValueError):
        _direct_refusal(
            bound, EXECUTION["artifact"], ENGINE_PIN, REFUSALS["cases"][0], call
        )


@pytest.mark.parametrize("suffix", ["\n", "extra diagnostic", " "])
def test_error_profile_is_exact(bound, suffix):
    call = copy.deepcopy(REFUSALS["calls"][0])
    call["stderr"] += suffix
    with pytest.raises(ValueError):
        _direct_refusal(
            bound, EXECUTION["artifact"], ENGINE_PIN, REFUSALS["cases"][0], call
        )


def test_forged_error_cannot_replace_true_selector(bound):
    case = copy.deepcopy(REFUSALS["cases"][0])
    call = copy.deepcopy(REFUSALS["calls"][0])
    name = next(n for n in case["input"] if n.endswith("personal_servitude_term_years"))
    case["input"][name] = 100
    fact = next(f for f in call["request"]["dataset"]["inputs"] if f["name"] == name)
    fact["value"]["value"] = 100
    with pytest.raises(ValueError, match="independently be false"):
        _direct_refusal(bound, EXECUTION["artifact"], ENGINE_PIN, case, call)


def test_mixed_success_and_refusal_contract_rejected(bound):
    case = copy.deepcopy(REFUSALS["cases"][0])
    case["output"] = {case["expected_refusal"]["output"]: 0}
    with pytest.raises(ValueError):
        _direct_refusal(
            bound, EXECUTION["artifact"], ENGINE_PIN, case, REFUSALS["calls"][0]
        )


def test_unknown_engine_cannot_interpret_error(bound):
    with pytest.raises(ValueError):
        _direct_refusal(
            bound,
            EXECUTION["artifact"],
            "unknown",
            REFUSALS["cases"][0],
            REFUSALS["calls"][0],
        )


@pytest.fixture(scope="module")
def paired(bound):
    cases = yaml.safe_load(EXECUTION["companion_yaml"])
    cases = cases["cases"] if isinstance(cases, dict) else cases
    cases = cases + [REFUSALS["separately_executed_positive"]]
    calls = EXECUTION["calls"] + [REFUSALS["separately_executed_positive_call"]]
    derived, parameters = _compiled(bound, EXECUTION["artifact"], ENGINE_PIN)
    positives = tuple(
        _record(bound, derived, parameters, c, r)
        for c, r in zip(cases, calls, strict=True)
    )
    refusals = tuple(
        _direct_refusal(bound, EXECUTION["artifact"], ENGINE_PIN, c, r)
        for c, r in zip(REFUSALS["cases"], REFUSALS["calls"], strict=True)
    )
    return cases, positives, refusals


def test_all_direct_refusals_have_typed_one_fact_root_pairs(bound, paired):
    cases, positives, refusals = paired
    pairs = direct_refusal_pairs(bound, positives, cases, refusals, REFUSALS["cases"])
    assert {p[3] for p in pairs} == {r.case_key for r in refusals}


def test_missing_partnership_positive_is_not_pair_credit(bound, paired):
    cases, positives, refusals = paired
    with pytest.raises(ValueError, match="lacks an exact one-fact"):
        direct_refusal_pairs(
            bound, positives[:-1], cases[:-1], refusals, REFUSALS["cases"]
        )


def test_pair_case_identity_is_not_rewritten(bound, paired):
    cases, positives, refusals = paired
    altered = copy.deepcopy(REFUSALS["cases"])
    altered[0]["entity_id"] = "borrowed"
    with pytest.raises(ValueError, match="case identity"):
        direct_refusal_pairs(bound, positives, cases, refusals, altered)
