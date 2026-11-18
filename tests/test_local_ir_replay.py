"""Closed arithmetic/path checks; fixture data is not capture authority."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from axiom_encode.harness.local_ir_replay import _arithmetic, check_compiled_replay
from tests.test_actual_local_relation_evidence import _actual_case


def artifact():
    return json.loads(
        (Path(__file__).parent / "fixtures/t936_provider_compiled.json").read_text()
    )


def test_actual31_nodes_with_exact_paths_and_arithmetic():
    assert len(check_compiled_replay(artifact(), *_actual_case())) == 31


@pytest.mark.parametrize(
    "change", ["reason", "missing", "extra", "duplicate", "wrong_entity"]
)
def test_skip_metadata_cannot_borrow_equal_final_value(change):
    policy, request, response = _actual_case()
    node = next(
        n
        for n in response["results"][0]["trace"].values()
        if n.get("not_evaluated_dependencies")
    )
    skipped = node["not_evaluated_dependencies"]
    if change == "reason":
        skipped[0]["reason"] = "invented"
    elif change == "missing":
        skipped.pop()
    elif change == "extra":
        skipped.append({"dependency": "invented", "reason": "short_circuit"})
    elif change == "duplicate":
        skipped.append(dict(skipped[0]))
    else:
        skipped[0]["dependency"] += "@entity:77726f6e67"
    with pytest.raises(ValueError):
        check_compiled_replay(artifact(), policy, request, response)


def test_decimal_promotion_and_ambient_independence():
    with localcontext() as context:
        context.prec = 2
        assert _arithmetic("add", [1, Decimal("0.0001")]) == Decimal("1.0001")
        assert _arithmetic("mul", [Decimal("440.20"), Decimal("0.5")]) == Decimal(
            "220.100"
        )


def test_intermediate_overflow_cannot_be_cancelled():
    maximum = Decimal((1 << 96) - 1)
    with pytest.raises(ValueError):
        _arithmetic("add", [maximum, 1, -1])


def test_multiplication_requiring_scale_reduction_is_not_exact_profile():
    with pytest.raises(ValueError):
        _arithmetic("mul", [Decimal("1e-28"), Decimal("0.1")])


@pytest.mark.parametrize("value", [True, 0.1, float("inf")])
def test_nonexact_numeric_operands_decline(value):
    with pytest.raises(ValueError):
        _arithmetic("add", [value])


@pytest.mark.parametrize(
    "expr",
    [
        {"kind": []},
        {"kind": "literal", "value": {"kind": "integer", "value": True}},
        {"kind": "literal", "value": {"kind": "decimal", "value": "invalid"}},
        {"kind": "derived", "name": []},
        {
            "kind": "sum_related",
            "relation": "r",
            "current_slot": 1,
            "related_slot": 0,
            "value": None,
            "where": None,
        },
        {"kind": "comparison", "op": "bogus", "left": {}, "right": {}},
    ],
)
def test_unexecuted_malformed_rule_declines_consistently(expr):
    compiled = artifact()
    rule = next(
        r
        for r in compiled["program"]["derived"]
        if r["name"] == "t936_zepv_cost_excluded"
    )
    rule["expr"] = expr
    rule["versions"][0]["expr"] = expr
    with pytest.raises(ValueError):
        check_compiled_replay(compiled, *_actual_case())


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", {"kind": "integer", "value": 37999}),
        ("source", "different source"),
        ("effective_from", "2024-01-01"),
        ("unit", "different unit"),
    ],
)
def test_compiled_parameter_must_match_read_even_when_result_unchanged(field, value):
    compiled = artifact()
    param = next(
        p
        for p in compiled["program"]["parameters"]
        if p["name"] == "t936_vehicle_cost_38000"
    )
    if field == "value":
        param["versions"][0]["values"]["0"] = value
    elif field == "effective_from":
        param["versions"][0][field] = value
    else:
        param[field] = value
    with pytest.raises(ValueError, match="IR selected parameter mismatch"):
        check_compiled_replay(compiled, *_actual_case())
