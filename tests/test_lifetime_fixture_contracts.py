"""Lifetime workflow contracts bind exact facts through final apply admission."""

import argparse
import copy
import json

import pytest
import yaml

from axiom_encode.cli import (
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)
from axiom_encode.harness.evals import (
    _format_required_test_case_contracts,
    _preserves_companion_test_cases,
)

CITATION = "us/statute/42/415/b"
RULE_PATH = "us/statutes/42/415/b.yaml"
INPUT = "us:statutes/42/415/b#input.synthetic_amount"
OUTPUT = "us:statutes/42/415/b#synthetic_total"


@pytest.fixture
def history():
    periods = [
        {"period_kind": "tax_year", "start": f"{year}-01-01", "end": f"{year}-12-31"}
        for year in (1998, 1999)
    ]
    return {
        "name": "required history",
        "description": "Synthetic engine fixture, not a policy calculation.",
        "period": periods[-1],
        "lifetime": {
            "entity": "Person",
            "periods": periods,
            "batches": [
                {
                    "row_count": 2,
                    "entity_ids": ["worker-1,234", "worker-2"],
                    "inputs": {
                        INPUT: {"kind": "decimal", "values": [str(value), "2.00"]}
                    },
                }
                for value in (3, 4)
            ],
        },
        "required_output": {OUTPUT: ["7.00", "4.00"]},
    }


def raw_contract(cases):
    return json.dumps(
        {
            "schema": "axiom-encode/review-contract/v2",
            "citation": CITATION,
            "rulespec_path": RULE_PATH,
            "required_deferred_outputs": [],
            "required_test_cases": cases,
        }
    )


def as_case(contract):
    result = copy.deepcopy(contract)
    result["output"] = result.pop("required_output")
    return result


def issues(tmp_path, contract, candidate):
    source = tmp_path / "b.yaml"
    source.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    source.with_suffix(".test.yaml").write_text(yaml.safe_dump([candidate]))
    return _required_deferred_output_contract_issues(
        source, contract, citation=CITATION, rulespec_path=RULE_PATH
    )


@pytest.mark.parametrize("calculation", [False, True])
def test_lifetime_contract_survives_cli_prompt_and_final_admission(
    tmp_path, history, calculation
):
    if calculation:
        history["period"] = {
            "period_kind": "month",
            "start": "2026-01-01",
            "end": "2026-01-31",
        }
        history["lifetime"]["calculation_period"] = copy.deepcopy(history["period"])
    parsed = _parse_deferred_output_review_contract_json(raw_contract([history]))
    typed = parsed.required_test_cases[0]
    assert typed.as_mapping() == history
    assert json.dumps(history, separators=(",", ":"), sort_keys=True) in (
        _format_required_test_case_contracts([typed.as_mapping()])
    )
    assert _preserves_companion_test_cases(
        "[]", yaml.safe_dump([as_case(history)]), [typed.as_mapping()]
    )
    assert issues(tmp_path, parsed, as_case(history)) == []
    exported = typed.as_mapping()
    exported["lifetime"]["batches"][0]["inputs"][INPUT]["values"][0] = "999"
    assert typed.as_mapping() == history


def test_final_admission_preserves_explicit_calculation_date(tmp_path, history):
    history["period"] = {
        "period_kind": "month",
        "start": "2026-01-01",
        "end": "2026-01-31",
    }
    history["lifetime"]["calculation_period"] = copy.deepcopy(history["period"])
    parsed = _parse_deferred_output_review_contract_json(raw_contract([history]))
    changed = as_case(history)
    changed["period"] = {
        "period_kind": "month",
        "start": "2027-01-01",
        "end": "2027-01-31",
    }
    changed["lifetime"]["calculation_period"] = copy.deepcopy(changed["period"])
    assert issues(tmp_path, parsed, changed)
    assert not _preserves_companion_test_cases(
        "[]", yaml.safe_dump([changed]), [history]
    )


@pytest.mark.parametrize(
    "change",
    [
        "period",
        "fact",
        "id",
        "row_order",
        "column_kind",
        "value_type",
        "bool_integer_alias",
        "output_type",
        "extra_top",
        "extra_history",
        "extra_batch",
        "extra_column",
        "scalar_input",
        "description",
    ],
)
def test_final_admission_refuses_any_history_contract_drift(tmp_path, history, change):
    parsed = _parse_deferred_output_review_contract_json(raw_contract([history]))
    case = as_case(history)
    batch = case["lifetime"]["batches"][0]
    if change == "period":
        case["period"]["end"] = "2000-12-31"
    elif change == "fact":
        batch["inputs"][INPUT]["values"][0] = "999"
    elif change == "id":
        batch["entity_ids"][0] = "someone-else"
    elif change == "row_order":
        for observation in case["lifetime"]["batches"]:
            observation["entity_ids"].reverse()
    elif change == "column_kind":
        batch["inputs"][INPUT]["kind"] = "text"
    elif change == "value_type":
        batch["inputs"][INPUT]["values"][0] = 3
    elif change == "bool_integer_alias":
        batch["row_count"] = True
    elif change == "output_type":
        case["output"][OUTPUT][0] = 7
    elif change == "extra_top":
        case["unexpected"] = {}
    elif change == "extra_history":
        case["lifetime"]["unexpected"] = {}
    elif change == "extra_batch":
        batch["unexpected"] = {}
    elif change == "extra_column":
        batch["inputs"][INPUT]["default"] = 0
    elif change == "scalar_input":
        case["input"] = {}
    else:
        case["description"] = "Changed description"
    assert issues(tmp_path, parsed, case)
    assert not _preserves_companion_test_cases(
        "[]", yaml.safe_dump([case]), [parsed.required_test_cases[0].as_mapping()]
    )


@pytest.mark.parametrize(
    "change",
    [
        "extra_top",
        "extra_period",
        "extra_history",
        "extra_batch",
        "extra_column",
        "scalar_input",
        "missing_batch",
        "invalid_decimal",
        "invalid_integer",
        "invalid_bool",
        "invalid_date",
        "unsupported_kind",
        "judgment_input",
        "mixed_period_kind",
        "mixed_custom_name",
        "column_length",
        "reversed_periods",
        "float_expected",
        "unknown_period",
        "duplicate_ids",
    ],
)
def test_cli_and_prompt_refuse_malformed_history_contracts(history, change):
    column = history["lifetime"]["batches"][0]["inputs"][INPUT]
    if change == "extra_top":
        history["unknown"] = True
    elif change == "extra_period":
        history["period"]["unknown"] = True
    elif change == "extra_history":
        history["lifetime"]["unknown"] = True
    elif change == "extra_batch":
        history["lifetime"]["batches"][0]["unknown"] = True
    elif change == "extra_column":
        column["unknown"] = True
    elif change == "scalar_input":
        history["input"] = {}
    elif change == "missing_batch":
        history["lifetime"]["batches"].pop()
    elif change == "invalid_decimal":
        column["values"][0] = 3.0
    elif change == "invalid_integer":
        column.update(kind="integer", values=[True, 2])
    elif change == "invalid_bool":
        column.update(kind="bool", values=[1, False])
    elif change == "invalid_date":
        column.update(kind="date", values=["1999-02-30", "1999-01-01"])
    elif change == "unsupported_kind":
        column["kind"] = "float"
    elif change == "judgment_input":
        column.update(kind="judgment", values=["holds", "not_holds"])
    elif change == "mixed_period_kind":
        history["lifetime"]["periods"][0]["period_kind"] = "month"
    elif change == "mixed_custom_name":
        for period in history["lifetime"]["periods"]:
            period.update(period_kind="custom", name="first")
        history["lifetime"]["periods"][-1]["name"] = "second"
    elif change == "column_length":
        column["values"].pop()
    elif change == "reversed_periods":
        history["lifetime"]["periods"].reverse()
        history["period"] = history["lifetime"]["periods"][-1]
    elif change == "float_expected":
        history["required_output"][OUTPUT][0] = 7.0
    elif change == "unknown_period":
        history["period"]["period_kind"] = "year"
    else:
        history["lifetime"]["batches"][0]["entity_ids"] = ["same", "same"]
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_deferred_output_review_contract_json(raw_contract([history]))
    with pytest.raises(ValueError):
        _format_required_test_case_contracts([history])


def test_scalar_contract_round_trip_is_unchanged_and_cannot_authorize_history(
    tmp_path, history
):
    scalar = {
        "name": "existing scalar",
        "period": history["period"],
        "input": {INPUT: True},
        "required_output": {OUTPUT: 1},
    }
    parsed = _parse_deferred_output_review_contract_json(raw_contract([scalar]))
    assert parsed.required_test_cases[0].as_mapping() == scalar
    assert parsed.required_test_cases[0].lifetime is None
    assert issues(tmp_path, parsed, as_case(scalar)) == []
    candidate = as_case(scalar)
    candidate["lifetime"] = history["lifetime"]
    assert "unsigned runtime input field(s): lifetime" in "\n".join(
        issues(tmp_path, parsed, candidate)
    )


def test_nested_bool_cannot_match_contracted_integer_output(tmp_path, history):
    history["required_output"][OUTPUT] = [1, 0]
    parsed = _parse_deferred_output_review_contract_json(raw_contract([history]))
    candidate = as_case(history)
    candidate["output"][OUTPUT] = [True, False]
    assert "missing or changes required output" in "\n".join(
        issues(tmp_path, parsed, candidate)
    )
    assert not _preserves_companion_test_cases(
        "[]", yaml.safe_dump([candidate]), [parsed.required_test_cases[0].as_mapping()]
    )
