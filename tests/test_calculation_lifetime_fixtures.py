"""V2 transports a separate legal date without changing historical facts."""

import copy

import pytest

from axiom_encode.harness.lifetime_fixture_contracts import (
    validate_lifetime_test_contract,
)
from axiom_encode.harness.lifetime_fixtures import (
    CALCULATION_REQUEST_SCHEMA,
    CALCULATION_RESPONSE_SCHEMA,
    build_lifetime_request,
    compare_lifetime_response,
)
from axiom_encode.harness.validator_pipeline import find_test_input_assignment_issues

OUTPUT = "us:statutes/99/1#total"
INPUT = "us:statutes/99/1#input.amount"


def year(y):
    return {"period_kind": "tax_year", "start": f"{y}-01-01", "end": f"{y}-12-31"}


@pytest.fixture
def case():
    return {
        "name": "completed history",
        "period": year(2026),
        "output": {OUTPUT: "3.000000000000000001"},
        "lifetime": {
            "entity": "Person",
            "calculation_period": year(2026),
            "periods": [year(1990), year(1991)],
            "batches": [
                {
                    "row_count": 1,
                    "entity_ids": ["001"],
                    "inputs": {INPUT: {"kind": "decimal", "values": [value]}},
                }
                for value in ["1.000000000000000001", "2"]
            ],
        },
    }


def response_for(case):
    request = build_lifetime_request(case, case["period"])
    return {
        "schema": CALCULATION_RESPONSE_SCHEMA,
        "engine_version": "synthetic-wire-test",
        "artifact_format_version": 2,
        "arithmetic": "decimal",
        "entity": "Person",
        "row_count": 1,
        "entity_ids": ["001"],
        "periods": request["periods"],
        "calculation_period": request["calculation_period"],
        "reference_period": request["calculation_period"],
        "output_period": request["calculation_period"],
        "selected_versions": [
            {
                "kind": "derived",
                "name": "total",
                "id": OUTPUT,
                "version_index": 2,
                "effective_from": "2020-01-01",
                "effective_to": None,
            },
            {
                "kind": "parameter",
                "name": "factor",
                "id": None,
                "version_index": 1,
                "effective_from": "2026-01-01",
                "effective_to": "2026-12-31",
            },
        ],
        "outputs": {
            OUTPUT: {
                "id": OUTPUT,
                "name": "total",
                "dtype": "decimal",
                "unit": None,
                "column": {"kind": "decimal", "values": [case["output"][OUTPUT]]},
            }
        },
    }


def test_explicit_calculation_date_selects_v2_and_preserves_every_fact(case):
    before = copy.deepcopy(case)
    request = build_lifetime_request(case, case["period"])
    assert request["schema"] == CALCULATION_REQUEST_SCHEMA
    assert request["periods"] == before["lifetime"]["periods"]
    assert request["batches"] == before["lifetime"]["batches"]
    assert case == before
    assert compare_lifetime_response(request, case["output"], response_for(case)) == []
    response = response_for(case)
    response["outputs"][OUTPUT]["column"]["values"] = ["3.000000000000000002"]
    assert len(compare_lifetime_response(request, case["output"], response)) == 1


@pytest.mark.parametrize("mutation", ["absent", "null", "different"])
def test_calculation_date_cannot_be_inferred_or_silently_downgraded(case, mutation):
    if mutation == "absent":
        del case["lifetime"]["calculation_period"]
    else:
        case["lifetime"]["calculation_period"] = (
            None if mutation == "null" else year(2021)
        )
    with pytest.raises(ValueError):
        build_lifetime_request(case, case["period"])


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "calculation",
        "reference",
        "missing",
        "empty",
        "unknown",
        "duplicate",
        "index",
        "future",
        "expired",
        "date",
        "identity",
        "name",
        "extra",
    ],
)
def test_malformed_or_mismatched_provenance_fails_before_numeric_comparison(
    case, mutation
):
    response = response_for(case)
    selection = response["selected_versions"][0]
    if mutation == "schema":
        response["schema"] = "axiom-rules-engine/lifetime-response/v1"
    elif mutation in {"calculation", "reference"}:
        response[mutation + "_period"] = year(1991)
    elif mutation == "missing":
        del response["selected_versions"]
    elif mutation == "empty":
        response["selected_versions"] = []
    elif mutation == "unknown":
        selection["kind"] = "unknown"
    elif mutation == "duplicate":
        response["selected_versions"].append(copy.deepcopy(selection))
    elif mutation == "index":
        selection["version_index"] = True
    elif mutation == "future":
        selection["effective_from"] = "2027-01-01"
    elif mutation == "expired":
        selection["effective_to"] = "2025-12-31"
    elif mutation == "date":
        selection["effective_to"] = "20260230"
    elif mutation == "identity":
        selection["id"] = None
    elif mutation == "name":
        selection["name"] = "different"
    else:
        selection["ignored"] = 1
    with pytest.raises(ValueError):
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )


def test_unversioned_output_is_explicit_and_nullable_dependency_ids_are_allowed(case):
    response = response_for(case)
    response["selected_versions"][0] = {
        "kind": "unversioned_derived",
        "name": "total",
        "id": OUTPUT,
    }
    assert (
        compare_lifetime_response(
            build_lifetime_request(case, case["period"]), case["output"], response
        )
        == []
    )


def test_required_contract_rejects_observation_on_calculation_start(case):
    contract = copy.deepcopy(case)
    contract["required_output"] = contract.pop("output")
    validate_lifetime_test_contract(contract)
    contract["lifetime"]["periods"][-1]["end"] = "2026-01-01"
    with pytest.raises(ValueError, match="end before calculation"):
        validate_lifetime_test_contract(contract)


def test_input_obligations_use_calculation_start_for_every_historical_batch(case):
    rules = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: total
    kind: derived
    entity: Person
    versions:
      - effective_from: '1980-01-01'
        effective_to: '2025-12-31'
        formula: sum_over_periods(old_amount)
      - effective_from: '2026-01-01'
        effective_to: '2026-06-30'
        formula: sum_over_periods(amount)
      - effective_from: '2026-07-01'
        formula: sum_over_periods(future_amount)
"""
    # The calculation interval crosses a version boundary; only its start selects law.
    assert find_test_input_assignment_issues(rules, [case]) == []
    case["lifetime"]["batches"][0]["inputs"] = {}
    issues = find_test_input_assignment_issues(rules, [case])
    assert len(issues) == 1 and "batch #1" in issues[0] and "#input.amount" in issues[0]
    assert "old_amount" not in issues[0] and "future_amount" not in issues[0]
