"""Exact, row-aligned companion fixtures for the Rust lifetime executor.

This adapter transports facts and compares typed results. All formula evaluation
and temporal/input validation belongs to `axiom-rules-engine run-lifetime`.
"""

import json
import re
import subprocess
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

REQUEST_SCHEMA = "axiom-rules-engine/lifetime-request/v1"
RESPONSE_SCHEMA = "axiom-rules-engine/lifetime-response/v1"
CALCULATION_REQUEST_SCHEMA = "axiom-rules-engine/lifetime-request/v2"
CALCULATION_RESPONSE_SCHEMA = "axiom-rules-engine/lifetime-response/v2"
MAX_REQUEST_BYTES = 16 * 1024 * 1024


def _keys(value: Any, allowed: set[str], required: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if set(value) - allowed:
        raise ValueError(f"{label} contains unsupported fields")
    if required - set(value):
        raise ValueError(f"{label} is missing required fields")
    return value


def build_lifetime_request(case: dict, period: dict) -> dict:
    """Build the public CLI request without coercing or inventing observations."""
    _keys(
        case,
        {"name", "description", "period", "output", "lifetime"},
        {"name", "period", "output", "lifetime"},
        "lifetime test case",
    )
    lifetime = _keys(
        case["lifetime"],
        {"entity", "arithmetic", "periods", "batches", "calculation_period"},
        {"entity", "periods", "batches"},
        "lifetime",
    )
    if not isinstance(lifetime["entity"], str) or not lifetime["entity"].strip():
        raise ValueError("lifetime.entity must be a nonempty string")
    if lifetime.get("arithmetic", "decimal") != "decimal":
        raise ValueError("lifetime fixtures require decimal arithmetic")
    periods, batches = lifetime["periods"], lifetime["batches"]
    if not isinstance(periods, list) or not periods or len(periods) > 512:
        raise ValueError("lifetime.periods must contain 1 to 512 periods")
    if not isinstance(batches, list) or len(batches) != len(periods):
        raise ValueError("lifetime requires one batch per period")
    calculation = "calculation_period" in lifetime
    if calculation:
        if not isinstance(lifetime["calculation_period"], dict):
            raise ValueError("lifetime.calculation_period must be an explicit mapping")
        if lifetime["calculation_period"] != period:
            raise ValueError(
                "lifetime calculation period must equal the test output period"
            )
    elif periods[-1] != period:
        raise ValueError("lifetime final period must equal the test output period")
    row_count = None
    entity_ids = None
    for batch in batches:
        _keys(
            batch,
            {"row_count", "entity_ids", "inputs"},
            {"row_count", "entity_ids", "inputs"},
            "lifetime batch",
        )
        count, ids = batch["row_count"], batch["entity_ids"]
        if type(count) is not int or not 1 <= count <= 100_000:
            raise ValueError("lifetime row_count must be an integer from 1 to 100000")
        if (
            not isinstance(ids, list)
            or len(ids) != count
            or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != count
        ):
            raise ValueError("lifetime entity_ids must identify every row uniquely")
        if row_count is not None and (count != row_count or ids != entity_ids):
            raise ValueError("lifetime batches must preserve entity row order")
        row_count, entity_ids = count, ids
        if not isinstance(batch["inputs"], dict):
            raise ValueError("lifetime batch inputs must be a mapping")
    outputs = case["output"]
    if (
        not isinstance(outputs, dict)
        or not outputs
        or any(not isinstance(key, str) or not key for key in outputs)
    ):
        raise ValueError("lifetime output must be a nonempty reference mapping")
    for expected in outputs.values():
        values = expected if isinstance(expected, list) else [expected]
        if len(values) != row_count:
            raise ValueError("lifetime expected outputs must assert every row")
        if any(type(value) not in (str, int, bool) for value in values):
            raise ValueError("lifetime expected decimals must be quoted strings")
    return {
        "schema": CALCULATION_REQUEST_SCHEMA if calculation else REQUEST_SCHEMA,
        **lifetime,
        "arithmetic": "decimal",
        "outputs": list(outputs),
        "output_period": period,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("lifetime response contains duplicate JSON keys")
        result[key] = value
    return result


def _typed_value(value: Any, kind: str) -> Any:
    if not isinstance(kind, str):
        raise ValueError("lifetime output kind must be a string")
    if kind == "decimal":
        if not isinstance(value, str) or not re.fullmatch(
            r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value
        ):
            raise ValueError("decimal values must be quoted decimal strings")
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("invalid decimal value") from exc
        if not parsed.is_finite():
            raise ValueError("decimal values must be finite")
        return parsed
    if kind == "integer":
        if type(value) is not int or not -(2**63) <= value < 2**63:
            raise ValueError("integer values must be exact signed 64-bit integers")
    elif kind == "bool":
        if type(value) is not bool:
            raise ValueError("bool values must be booleans")
    elif kind in {"text", "date", "judgment"}:
        if not isinstance(value, str):
            raise ValueError(f"{kind} values must be strings")
        if kind == "judgment" and value not in {"holds", "not_holds", "undetermined"}:
            raise ValueError("invalid judgment value")
        if kind == "date":
            try:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("noncanonical date")
            except ValueError as exc:
                raise ValueError(
                    "date values must be valid YYYY-MM-DD strings"
                ) from exc
    else:
        raise ValueError(f"unsupported lifetime output kind: {kind}")
    return value


def _validate_selected_versions(request: dict, response: dict) -> None:
    """Check version provenance shape and output identity, without selecting law."""
    selected = response["selected_versions"]
    if not isinstance(selected, list) or not selected:
        raise ValueError("lifetime selected_versions must be a nonempty list")
    calculation_date = _typed_value(request["calculation_period"]["start"], "date")
    identities = set()
    derived_ids = {}
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("lifetime selected version must be a mapping")
        kind = item.get("kind")
        if kind not in ("derived", "unversioned_derived", "parameter"):
            raise ValueError("lifetime selected version has unsupported kind")
        fields = {"kind", "name", "id"}
        if kind != "unversioned_derived":
            fields |= {"version_index", "effective_from", "effective_to"}
        _keys(item, fields, fields, "lifetime selected version")
        name, identity = item["name"], item["id"]
        if (
            not isinstance(name, str)
            or not name
            or (
                identity is not None and (not isinstance(identity, str) or not identity)
            )
        ):
            raise ValueError("lifetime selected version has invalid identity")
        key = ("parameter" if kind == "parameter" else "derived", name)
        if key in identities:
            raise ValueError("lifetime selected version identity is duplicated")
        identities.add(key)
        if kind != "parameter" and identity is not None:
            if identity in derived_ids:
                raise ValueError("lifetime selected derived ID is duplicated")
            derived_ids[identity] = name
        if kind != "unversioned_derived":
            if type(item["version_index"]) is not int or item["version_index"] < 0:
                raise ValueError(
                    "lifetime selected version index must be nonnegative integer"
                )
            start = _typed_value(item["effective_from"], "date")
            end = item["effective_to"]
            if end is not None:
                end = _typed_value(end, "date")
            if start > calculation_date or (end is not None and end < calculation_date):
                raise ValueError(
                    "lifetime selected version does not cover calculation date"
                )
    for reference, output in response["outputs"].items():
        if not isinstance(output, dict) or derived_ids.get(reference) != output.get(
            "name"
        ):
            raise ValueError("lifetime selected versions do not identify each output")


def compare_lifetime_response(
    request: dict, expected: dict, response: Any
) -> list[str]:
    """Reject malformed/misaligned results before comparing exact typed columns."""
    schemas = {
        REQUEST_SCHEMA: RESPONSE_SCHEMA,
        CALCULATION_REQUEST_SCHEMA: CALCULATION_RESPONSE_SCHEMA,
    }
    response_schema = schemas.get(request.get("schema"))
    if (
        response_schema is None
        or not isinstance(response, dict)
        or response.get("schema") != response_schema
    ):
        raise ValueError("invalid lifetime response schema")
    fields = {
        "schema",
        "engine_version",
        "artifact_format_version",
        "arithmetic",
        "entity",
        "row_count",
        "entity_ids",
        "periods",
        "reference_period",
        "output_period",
        "outputs",
    }
    calculation = request["schema"] == CALCULATION_REQUEST_SCHEMA
    if calculation:
        fields |= {"calculation_period", "selected_versions"}
    _keys(response, fields, fields, "lifetime response")
    if (
        not isinstance(response["engine_version"], str)
        or not response["engine_version"]
        or type(response["artifact_format_version"]) is not int
        or response["artifact_format_version"] != 2
    ):
        raise ValueError("lifetime response has unsupported version metadata")
    batch = request["batches"][-1]
    expected_metadata = {
        "arithmetic": "decimal",
        "entity": request["entity"],
        "entity_ids": batch["entity_ids"],
        "periods": request["periods"],
        "reference_period": request["output_period"],
        "output_period": request["output_period"],
    }
    if calculation:
        expected_metadata["calculation_period"] = request["calculation_period"]
    for field, wanted in expected_metadata.items():
        if response.get(field) != wanted:
            raise ValueError(f"lifetime response {field} does not match the request")
    if (
        type(response.get("row_count")) is not int
        or response["row_count"] != batch["row_count"]
    ):
        raise ValueError("lifetime response row_count does not match the request")
    outputs = response.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected):
        raise ValueError("lifetime response must contain exactly the requested outputs")
    if calculation:
        _validate_selected_versions(request, response)
    issues = []
    for reference, expected_value in expected.items():
        fields = {"id", "name", "dtype", "unit", "column"}
        output = _keys(outputs[reference], fields, fields, "lifetime output")
        if output["id"] != reference:
            raise ValueError("lifetime response output identity mismatch")
        if (
            not isinstance(output["name"], str)
            or not output["name"]
            or (output["unit"] is not None and not isinstance(output["unit"], str))
        ):
            raise ValueError("invalid lifetime output metadata")
        column = _keys(
            output.get("column"),
            {"kind", "values"},
            {"kind", "values"},
            "lifetime output column",
        )
        values = column["values"]
        if output["dtype"] != column["kind"]:
            raise ValueError("lifetime output dtype does not match its column kind")
        if not isinstance(values, list) or len(values) != batch["row_count"]:
            raise ValueError("lifetime output column must contain every row")
        wanted = (
            expected_value if isinstance(expected_value, list) else [expected_value]
        )
        for index, (actual, target) in enumerate(zip(values, wanted, strict=True)):
            kind = column["kind"]
            if _typed_value(actual, kind) != _typed_value(target, kind):
                issues.append(
                    f"output `{reference}` row `{batch['entity_ids'][index]}` "
                    f"expected {target!r}, got {actual!r}."
                )
    return issues


def run_lifetime_fixture(
    *,
    binary: Path,
    compiled_path: Path,
    case: dict,
    period: dict,
    cwd: Path | None,
    env: Mapping[str, str],
) -> list[str]:
    """Execute a synthetic companion history using the real Rust CLI."""
    request = build_lifetime_request(case, period)
    try:
        payload = json.dumps(request, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "lifetime inputs must use JSON types; quote dates/decimals"
        ) from exc
    if len(payload.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValueError("lifetime request exceeds the 16 MiB limit")
    result = subprocess.run(
        [str(binary), "run-lifetime", "--artifact", str(compiled_path)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=cwd,
        env=env,
    )
    if result.returncode:
        raise ValueError(
            "lifetime execution failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        response = json.loads(result.stdout, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError("lifetime response is not valid JSON") from exc
    return compare_lifetime_response(request, case["output"], response)
