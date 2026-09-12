"""Strict history facts for workflow contracts, separate from engine semantics."""

from datetime import date
from typing import Any, Mapping

from .lifetime_fixtures import _typed_value, build_lifetime_request


def exact_fixture_value_equal(actual: object, expected: object) -> bool:
    """Compare every nested scalar type, including bool versus integer."""
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            exact_fixture_value_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            exact_fixture_value_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _normalized_string(value: Any, label: str, *, multiline: bool = False) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(
            (ord(char) < 32 and (not multiline or char not in {"\n", "\t"}))
            or ord(char) == 127
            for char in value
        )
    ):
        raise ValueError(f"{label} must be a nonempty normalized string")


def _period(value: Any) -> tuple[date, date]:
    if not isinstance(value, dict):
        raise ValueError("lifetime contract periods must be explicit mappings")
    fields = {"period_kind", "start", "end"}
    if value.get("period_kind") == "custom":
        fields.add("name")
    if set(value) != fields or value.get("period_kind") not in {
        "month",
        "benefit_week",
        "tax_year",
        "custom",
    }:
        raise ValueError("lifetime contract period has unsupported fields or kind")
    for field in fields:
        _normalized_string(value[field], f"period {field}")
    start, end = date.fromisoformat(value["start"]), date.fromisoformat(value["end"])
    if (
        start.isoformat() != value["start"]
        or end.isoformat() != value["end"]
        or start > end
    ):
        raise ValueError("lifetime contract period must have ordered ISO dates")
    return start, end


def validate_lifetime_test_contract(contract: Mapping[str, Any]) -> None:
    """Check exact contract syntax; the engine still validates and executes laws."""
    if not isinstance(contract, Mapping):
        raise ValueError("lifetime test contract must be a mapping")
    fields = {"name", "period", "lifetime", "required_output"}
    if "description" in contract:
        fields.add("description")
        _normalized_string(contract["description"], "description", multiline=True)
    if set(contract) != fields:
        raise ValueError(
            "lifetime test contract contains unsupported or missing fields"
        )
    _normalized_string(contract["name"], "case name")
    case = {key: value for key, value in contract.items() if key != "required_output"}
    case["output"] = contract["required_output"]
    request = build_lifetime_request(case, case["period"])
    output_start, _ = _period(case["period"])
    calculation = "calculation_period" in request
    if calculation:
        _period(request["calculation_period"])
    _normalized_string(request["entity"], "entity")
    previous_end = None
    period_identity = None
    for period, batch in zip(request["periods"], request["batches"], strict=True):
        start, end = _period(period)
        if calculation and end >= output_start:
            raise ValueError("lifetime observations must end before calculation starts")
        identity = (period["period_kind"], period.get("name"))
        if period_identity is not None and identity != period_identity:
            raise ValueError(
                "lifetime contract periods must share kind and custom name"
            )
        period_identity = identity
        if previous_end is not None and start <= previous_end:
            raise ValueError(
                "lifetime contract periods must be ordered and nonoverlapping"
            )
        previous_end = end
        for entity_id in batch["entity_ids"]:
            _normalized_string(entity_id, "entity ID")
        if len(batch["inputs"]) > 64:
            raise ValueError(
                "lifetime contract batch inputs must have at most 64 fields"
            )
        for reference, column in batch["inputs"].items():
            _normalized_string(reference, "input reference")
            if not isinstance(column, dict) or set(column) != {"kind", "values"}:
                raise ValueError(
                    "lifetime input columns require exactly kind and values"
                )
            if column["kind"] not in ("decimal", "integer", "bool", "text", "date"):
                raise ValueError("unsupported lifetime input column kind")
            values = column["values"]
            if not isinstance(values, list) or len(values) != batch["row_count"]:
                raise ValueError("lifetime input columns must cover every entity row")
            for value in values:
                _typed_value(value, column["kind"])
    if len(case["output"]) > 64:
        raise ValueError(
            "lifetime contract required_output must have at most 64 fields"
        )
    for reference, expected in case["output"].items():
        _normalized_string(reference, "output reference")
        for value in expected if isinstance(expected, list) else [expected]:
            if isinstance(value, str):
                _normalized_string(value, "expected output", multiline=True)
