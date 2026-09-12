"""Lifetime fixture transport is exact, complete and fails closed."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from axiom_encode.harness.lifetime_fixtures import (
    RESPONSE_SCHEMA,
    build_lifetime_request,
    compare_lifetime_response,
    run_lifetime_fixture,
)

OUTPUT = "us:statutes/test#total"
INPUT = "us:statutes/test#input.amount"


@pytest.fixture
def case():
    periods = [
        {"period_kind": "tax_year", "start": f"{year}-01-01", "end": f"{year}-12-31"}
        for year in (2020, 2021)
    ]
    return {
        "name": "two people with distinct histories",
        "period": periods[-1],
        "output": {OUTPUT: ["3.000000000000000001", "7.50"]},
        "lifetime": {
            "entity": "Person",
            "periods": periods,
            "batches": [
                {
                    "row_count": 2,
                    "entity_ids": ["first", "second"],
                    "inputs": {INPUT: {"kind": "decimal", "values": values}},
                }
                for values in (["1.000000000000000001", "3.25"], ["2.0", "4.25"])
            ],
        },
    }


def response_for(case):
    request = build_lifetime_request(case, case["period"])
    return {
        "schema": RESPONSE_SCHEMA,
        "engine_version": "synthetic-transport-test",
        "artifact_format_version": 2,
        "entity": request["entity"],
        "arithmetic": "decimal",
        "row_count": 2,
        "entity_ids": ["first", "second"],
        "periods": request["periods"],
        "reference_period": request["output_period"],
        "output_period": request["output_period"],
        "outputs": {
            OUTPUT: {
                "id": OUTPUT,
                "name": "total",
                "dtype": "decimal",
                "unit": None,
                "column": {"kind": "decimal", "values": case["output"][OUTPUT].copy()},
            }
        },
    }


def test_exact_decimal_comparison_distinguishes_beyond_float_precision(case):
    request = build_lifetime_request(case, case["period"])
    response = response_for(case)
    assert compare_lifetime_response(request, case["output"], response) == []
    response["outputs"][OUTPUT]["column"]["values"][0] = "3.000000000000000002"
    issues = compare_lifetime_response(request, case["output"], response)
    assert len(issues) == 1 and "first" in issues[0]


@pytest.mark.parametrize("field", ["input", "tables", "oracle_inputs", "unknown"])
def test_history_rejects_silently_ignored_scalar_fields(case, field):
    case[field] = {}
    with pytest.raises(ValueError, match="unsupported fields"):
        build_lifetime_request(case, case["period"])


@pytest.mark.parametrize("mutation", ["rows", "order", "period", "batches", "float"])
def test_invalid_history_is_rejected_before_execution(case, mutation):
    if mutation == "rows":
        case["output"][OUTPUT] = "3.0"
    elif mutation == "order":
        case["lifetime"]["batches"][1]["entity_ids"].reverse()
    elif mutation == "period":
        case["period"] = case["lifetime"]["periods"][0]
    elif mutation == "batches":
        case["lifetime"]["batches"].pop()
    else:
        case["output"][OUTPUT][0] = 3.0
    with pytest.raises(ValueError):
        build_lifetime_request(case, case["period"])


@pytest.mark.parametrize(
    "mutation",
    [
        "order",
        "rows",
        "period",
        "arithmetic",
        "missing",
        "extra",
        "float",
        "nan",
        "kind",
    ],
)
def test_response_contract_is_checked_before_comparison(case, mutation):
    request = build_lifetime_request(case, case["period"])
    response = copy.deepcopy(response_for(case))
    if mutation == "order":
        response["entity_ids"].reverse()
    elif mutation == "rows":
        response["row_count"] = True
    elif mutation == "period":
        response["reference_period"] = request["periods"][0]
    elif mutation == "arithmetic":
        response["arithmetic"] = "f64"
    elif mutation == "missing":
        response["outputs"] = {}
    elif mutation == "extra":
        response["outputs"]["extra"] = {}
    elif mutation == "kind":
        response["outputs"][OUTPUT]["column"]["kind"] = []
    else:
        response["outputs"][OUTPUT]["column"]["values"][0] = (
            3.0 if mutation == "float" else "NaN"
        )
    with pytest.raises(ValueError):
        compare_lifetime_response(request, case["output"], response)


def test_run_uses_real_cli_contract_without_calculation(case, monkeypatch, tmp_path):
    response = response_for(case)
    calls = []

    def capture(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "")

    monkeypatch.setattr(
        "axiom_encode.harness.lifetime_fixtures.subprocess.run", capture
    )
    artifact = tmp_path / "compiled.json"
    assert (
        run_lifetime_fixture(
            binary=Path("/test/engine"),
            compiled_path=artifact,
            case=case,
            period=case["period"],
            cwd=tmp_path,
            env={},
        )
        == []
    )
    command, options = calls[0]
    assert command == ["/test/engine", "run-lifetime", "--artifact", str(artifact)]
    assert options["timeout"] == 60
    request = json.loads(options["input"])
    assert request["batches"] == case["lifetime"]["batches"]
    assert request["outputs"] == [OUTPUT]


def test_duplicate_response_keys_fail_closed(case, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "axiom_encode.harness.lifetime_fixtures.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, '{"outputs":{},"outputs":{}}', ""
        ),
    )
    with pytest.raises(ValueError, match="duplicate JSON keys"):
        run_lifetime_fixture(
            binary=Path("/test/engine"),
            compiled_path=tmp_path / "compiled.json",
            case=case,
            period=case["period"],
            cwd=tmp_path,
            env={},
        )


@pytest.mark.parametrize(
    "mutation", ["version", "engine", "dtype", "unit", "name", "id"]
)
def test_response_metadata_must_match_the_wire(case, mutation):
    response = response_for(case)
    if mutation == "version":
        response["artifact_format_version"] = 3
    elif mutation == "engine":
        response["engine_version"] = ""
    elif mutation == "id":
        del response["outputs"][OUTPUT]["id"]
    else:
        response["outputs"][OUTPUT][mutation] = {
            "dtype": "integer",
            "unit": [],
            "name": "",
        }[mutation]
    with pytest.raises(ValueError):
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )


@pytest.mark.parametrize("value", ["2020-02-30", "20200101", "not-a-date"])
def test_dates_are_valid_canonical_iso_strings_even_when_equal(case, value):
    case["output"][OUTPUT] = [value, value]
    response = response_for(case)
    response["outputs"][OUTPUT]["dtype"] = "date"
    response["outputs"][OUTPUT]["column"]["kind"] = "date"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )


@pytest.mark.parametrize("value", ["1_000", "１２", "1e2", "+1"])
def test_decimal_wire_rejects_nondecimal_spellings(case, value):
    case["output"][OUTPUT] = [value, value]
    response = response_for(case)
    with pytest.raises(ValueError, match="decimal strings"):
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )


def test_canonical_output_requires_retained_identity(case):
    response = response_for(case)
    response["outputs"][OUTPUT]["id"] = None
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )
