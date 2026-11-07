"""Exact result mutations; fixtures never stand in for private execution."""

import copy
from decimal import Decimal

import pytest

from axiom_encode.harness import mhrtc_results as m

KEY = m._PREFIX + "multigenerational_home_renovation_tax_credit_line_45355"
PERIOD = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}


@pytest.fixture
def sample():
    rule = m._RULES[KEY]
    declaration = {
        "id": KEY,
        "name": rule["name"],
        "entity": rule["entity"],
        "dtype": "decimal",
        "unit": "CAD",
        "period": "Year",
        "semantics": "scalar",
        "versions": [{"effective_from": "2025-01-01", "effective_to": "2025-12-31"}],
    }
    request = {
        "mode": "explain",
        "queries": [{"entity_id": "p", "period": PERIOD, "outputs": [KEY]}],
    }
    output = {
        "kind": "scalar",
        "name": rule["name"],
        "id": KEY,
        "dtype": "decimal",
        "unit": "CAD",
        "value": {"kind": "decimal", "value": "7250"},
    }
    response = {
        "metadata": {
            "requested_mode": "explain",
            "actual_mode": "explain",
            "fallback_reason": None,
        },
        "results": [
            {
                "entity_id": "p",
                "period": PERIOD,
                "outputs": {KEY: output},
                "trace": {KEY: {**copy.deepcopy(output), "entity_id": "p"}},
            }
        ],
    }
    return {"program": {"derived": [declaration]}}, request, response, {KEY: 7250}


def test_exact_result(sample):
    assert m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize("kind,value", [("integer", 7250), ("decimal", "7250.000")])
def test_supported_numeric_wires(sample, kind, value):
    sample[2]["results"][0]["outputs"][KEY]["value"] = {"kind": kind, "value": value}
    sample[2]["results"][0]["trace"][KEY]["value"] = {"kind": kind, "value": value}
    assert m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "integer"),
        ("unit", "USD"),
        ("id", "wrong"),
        ("name", "wrong"),
        ("kind", "judgment"),
        ("extra", 1),
    ],
)
def test_output_metadata_mutation(sample, field, value):
    sample[2]["results"][0]["outputs"][KEY][field] = value
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "decimal", "value": 7250.0},
        {"kind": "integer", "value": True},
        {"kind": "decimal", "value": "NaN"},
        {"kind": "decimal", "value": "Infinity"},
        {"kind": "decimal", "value": "7250.000000000000000001"},
    ],
)
def test_invalid_or_distinct_wire(sample, wire):
    sample[2]["results"][0]["outputs"][KEY]["value"] = wire
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize("expected", [True, 7250.0, Decimal("7250.000000000000000001")])
def test_expected_precision_and_types(sample, expected):
    sample[3][KEY] = expected
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize(
    "mutation",
    [
        "entity",
        "period",
        "assessment",
        "missing",
        "extra",
        "duplicate",
        "fallback",
        "version",
        "declaration",
        "duplicate_declaration",
    ],
)
def test_request_or_declaration_mutation(sample, mutation):
    payload, request, response, _ = sample
    row = response["results"][0]
    if mutation == "entity":
        row["entity_id"] = "other"
    elif mutation == "period":
        row["period"] = {**PERIOD, "end": "2026-12-31"}
    elif mutation == "assessment":
        row["assessment_date"] = None
    elif mutation == "missing":
        row["outputs"].clear()
    elif mutation == "extra":
        row["outputs"]["unrequested"] = row["outputs"][KEY]
    elif mutation == "duplicate":
        request["queries"][0]["outputs"].append(KEY)
    elif mutation == "fallback":
        response["metadata"]["fallback_reason"] = "fallback"
    elif mutation == "version":
        payload["program"]["derived"][0]["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "declaration":
        payload["program"]["derived"][0]["entity"] = "OtherClaim"
    else:
        payload["program"]["derived"].append(
            copy.deepcopy(payload["program"]["derived"][0])
        )
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


def parameters():
    expected = {
        m._PREFIX
        + "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation": 50000,
        m._PREFIX + "federal_2025_mhrtc_rate": Decimal("0.145"),
        m._PREFIX + "federal_2025_mhrtc_maximum_credit_per_renovation": 7250,
    }
    rows = []
    for key, value in expected.items():
        rows.append(
            {
                "id": key,
                "name": m._RULES[key]["name"],
                "unit": m._RULES[key].get("unit"),
                "indexed_by": None,
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "values": {
                            "0": {
                                "kind": "integer" if type(value) is int else "decimal",
                                "value": value if type(value) is int else str(value),
                            }
                        },
                    }
                ],
            }
        )
    return {"program": {"parameters": rows}}, expected


def test_three_parameters_are_static_assertions():
    assert m.validate_mhrtc_parameters(*parameters())


@pytest.mark.parametrize(
    "mutation", ["date", "unit", "index", "value", "missing", "extra", "duplicate"]
)
def test_parameter_mutation(mutation):
    payload, expected = parameters()
    rows = payload["program"]["parameters"]
    if mutation == "date":
        rows[0]["versions"][0]["effective_from"] = "2024-01-01"
    elif mutation == "unit":
        rows[0]["unit"] = "USD"
    elif mutation == "index":
        rows[0]["indexed_by"] = "age"
    elif mutation == "value":
        rows[0]["versions"][0]["values"]["0"]["value"] = 50001
    elif mutation == "missing":
        expected.pop(next(iter(expected)))
    elif mutation == "extra":
        expected["unrelated"] = 0
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        m.validate_mhrtc_parameters(payload, expected)


@pytest.mark.parametrize(
    "mutation",
    [
        "trace_missing",
        "trace_entity",
        "trace_value",
        "envelope_extra",
        "row_extra",
        "semantics",
        "version_type",
    ],
)
def test_closed_envelope_and_root_trace(sample, mutation):
    payload, _, response, _ = sample
    row = response["results"][0]
    if mutation == "trace_missing":
        row["trace"].clear()
    elif mutation == "trace_entity":
        row["trace"][KEY]["entity_id"] = "unrelated"
    elif mutation == "trace_value":
        row["trace"][KEY]["value"]["value"] = "0"
    elif mutation == "envelope_extra":
        response["unreviewed"] = True
    elif mutation == "row_extra":
        row["unreviewed"] = True
    elif mutation == "semantics":
        payload["program"]["derived"][0]["semantics"] = "judgment"
    else:
        payload["program"]["derived"][0]["versions"] = [None]
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


@pytest.mark.parametrize("wire", [" 7250", "7250 ", "7_250", "７２５０"])
def test_noncanonical_numeric_wire(sample, wire):
    sample[2]["results"][0]["outputs"][KEY]["value"]["value"] = wire
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


def test_numeric_expected_string_rejected(sample):
    sample[3][KEY] = "7250"
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


def test_extra_related_trace_allowed(sample):
    sample[2]["results"][0]["trace"]["nested"] = {"entity_id": "related-record"}
    assert m.validate_mhrtc_outputs(*sample)


def test_parameter_wire_swap_rejected():
    payload, expected = parameters()
    payload["program"]["parameters"][0]["versions"][0]["values"]["0"] = {
        "kind": "decimal",
        "value": "50000",
    }
    with pytest.raises(ValueError):
        m.validate_mhrtc_parameters(payload, expected)


@pytest.mark.parametrize(
    "expected,wire,passes",
    [
        (0, "0", True),
        (0, "0.0", True),
        (0, "0.5", False),
        (Decimal("0"), "0", False),
        (False, "0", False),
    ],
)
def test_integer_declared_decimal_wire(expected, wire, passes):
    key = m._PREFIX + "prior_distinct_renovation_count"
    declaration = {"name": m._RULES[key]["name"], "dtype": "integer", "unit": None}
    node = {
        "id": key,
        "name": declaration["name"],
        "kind": "scalar",
        "dtype": "integer",
        "unit": None,
        "value": {"kind": "decimal", "value": wire},
    }
    if passes:
        m._assert_output(key, node, declaration, expected)
    else:
        with pytest.raises(ValueError):
            m._assert_output(key, node, declaration, expected)


@pytest.mark.parametrize("expected", ["2025-01-01", "2025-1-1", "2025-02-30"])
def test_canonical_date(expected):
    key = m._PREFIX + "record_paid_date"
    declaration = {"name": "record_paid_date", "dtype": "date", "unit": None}
    node = {
        "id": key,
        "name": declaration["name"],
        "kind": "scalar",
        "dtype": "date",
        "unit": None,
        "value": {"kind": "date", "value": expected},
    }
    if expected == "2025-01-01":
        m._assert_output(key, node, declaration, expected)
    else:
        with pytest.raises(ValueError):
            m._assert_output(key, node, declaration, expected)


@pytest.mark.parametrize("expected", ["holds", "not_holds", True, "unknown"])
def test_judgment_expected(expected):
    key = m._PREFIX + "attach_schedule_to_return"
    declaration = {
        "name": "attach_schedule_to_return",
        "dtype": "judgment",
        "unit": None,
    }
    node = {
        "id": key,
        "name": declaration["name"],
        "kind": "judgment",
        "unit": None,
        "outcome": expected,
    }
    if type(expected) is str and expected in ("holds", "not_holds"):
        m._assert_output(key, node, declaration, expected)
    else:
        with pytest.raises(ValueError):
            m._assert_output(key, node, declaration, expected)


@pytest.mark.parametrize("position", [1, 2])
def test_nonmapping_request_response_rejects_as_validation_error(sample, position):
    args = list(sample)
    args[position] = []
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*args)


def test_unrepresentable_decimal_exponent_rejects_as_validation_error(sample):
    sample[2]["results"][0]["outputs"][KEY]["value"]["value"] = (
        "1e999999999999999999999999999999999999"
    )
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


def test_observation_is_exact_actual_value_not_expected_assertion(sample):
    payload, request, response, _ = sample
    response["results"][0]["outputs"][KEY]["value"]["value"] = "7249.855"
    response["results"][0]["trace"][KEY]["value"]["value"] = "7249.855"
    assert m.observe_mhrtc_outputs(payload, request, response) == (
        (KEY, Decimal("7249.855")),
    )
    with pytest.raises(ValueError):
        m.validate_mhrtc_outputs(*sample)


def test_observation_still_requires_exact_trace_and_declaration(sample):
    payload, request, response, _ = sample
    response["results"][0]["trace"][KEY]["entity_id"] = "different"
    with pytest.raises(ValueError):
        m.observe_mhrtc_outputs(payload, request, response)
