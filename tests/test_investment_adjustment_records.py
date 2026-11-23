import copy
import json
from pathlib import Path

import pytest

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_execution import ENGINE_PIN, PREFIX
from axiom_encode.harness.investment_adjustment_records import (
    observe_investment_adjustment_record,
)

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
ARTIFACT = json.loads((F / "schedule-n-compiled.json").read_text())
RECORD = json.loads((F / "schedule-n-actual-record.json").read_text())
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def observe(record):
    return observe_investment_adjustment_record(
        BOUND,
        CONTENT,
        PARENT,
        GUIDE,
        ARTIFACT,
        ENGINE_PIN,
        record["case"],
        record["call"],
        caller="pipeline",
    )


def test_actual_full_typed_record_recomputed_with_complete_trace():
    result = observe(RECORD)
    assert len(result.outputs) == 18
    assert len(result.reached) == 18
    assert len(result.inputs) == 21


@pytest.mark.parametrize("index", range(18))
def test_coherent_output_and_trace_value_forgery_rejected(index):
    record = copy.deepcopy(RECORD)
    response = json.loads(record["call"]["stdout"])
    result = response["results"][0]
    name = list(result["outputs"])[index]
    for node in (result["outputs"][name], result["trace"][name]):
        if node["kind"] == "judgment":
            node["outcome"] = "holds" if node["outcome"] == "not_holds" else "not_holds"
        elif node["dtype"] == "bool":
            node["value"]["value"] = not node["value"]["value"]
        else:
            node["value"] = {"kind": "decimal", "value": "987654321"}
    record["call"]["stdout"] = json.dumps(response)
    with pytest.raises(ValueError):
        observe(record)


@pytest.mark.parametrize("field", ["entity_id", "interval", "entity", "name", "value"])
def test_actual_input_identity_and_typed_value_must_match_raw_case(field):
    record = copy.deepcopy(RECORD)
    record["call"]["request"]["dataset"]["inputs"][0][field] = None
    with pytest.raises(ValueError, match="actual request"):
        observe(record)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_relation",
        "duplicate_input",
        "extra_query",
        "wrong_root",
        "missing_fact",
        "bool_money",
        "year",
        "false_exit",
        "stderr",
        "duplicate_json",
    ],
)
def test_request_and_execution_boundary_rejects_malformed_evidence(mutation):
    record = copy.deepcopy(RECORD)
    call = record["call"]
    request = call["request"]
    if mutation == "extra_relation":
        request["dataset"]["relations"] = [{}]
    elif mutation == "duplicate_input":
        request["dataset"]["inputs"].append(request["dataset"]["inputs"][0])
    elif mutation == "extra_query":
        request["queries"].append(request["queries"][0])
    elif mutation == "wrong_root":
        request["queries"][0]["entity_id"] = "other"
    elif mutation == "missing_fact":
        record["case"]["input"].pop(next(iter(record["case"]["input"])))
    elif mutation == "bool_money":
        record["case"]["input"][next(iter(record["case"]["input"]))] = True
    elif mutation == "year":
        record["case"]["period"]["start"] = "2024-01-01"
    elif mutation == "false_exit":
        call["returncode"] = False
    elif mutation == "stderr":
        call["stderr"] = "failed"
    elif mutation == "duplicate_json":
        call["stdout"] = '{"metadata":{},"metadata":{}}'
    with pytest.raises(ValueError):
        observe(record)


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "expression",
        "unit",
        "dtype",
        "reached",
        "skipped",
        "parameter",
        "extra_trace",
        "missing_trace",
    ],
)
def test_trace_contract_is_not_just_matching_returned_amounts(mutation):
    record = copy.deepcopy(RECORD)
    response = json.loads(record["call"]["stdout"])
    trace = response["results"][0]["trace"]
    node = trace[PREFIX + "schedule_n_resource_deduction_recovery_line_28"]
    if mutation == "source":
        node["source"] = "unrelated"
    elif mutation == "expression":
        node["executed_expression"] = "500"
    elif mutation == "unit":
        node["unit"] = "USD"
    elif mutation == "dtype":
        node["dtype"] = "integer"
    elif mutation == "reached":
        node["dependencies"] = [PREFIX + "foreign"]
    elif mutation == "skipped":
        trace[PREFIX + "schedule_n_entered_return252_within_source_upper_conditions"][
            "not_evaluated_dependencies"
        ] = []
    elif mutation == "parameter":
        node["parameter_reads"][0]["value"]["value"] = "0.4"
    elif mutation == "extra_trace":
        trace[PREFIX + "foreign"] = node
    elif mutation == "missing_trace":
        trace.pop(next(iter(trace)))
    record["call"]["stdout"] = json.dumps(response)
    with pytest.raises(ValueError):
        observe(record)
