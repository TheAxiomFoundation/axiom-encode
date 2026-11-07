"""Exact MHRtC result assertions, separate from private execution authority.

Call only with independently reconstructed requests and freshly verified IR.
These checks do not authenticate a response, source, case or compiled program.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from .mhrtc_execution import same_typed_json
from .mhrtc_program_contract import POLICY_PATH, RULES
from .same_case_execution import strict_explain_metadata

_PREFIX = "ca:" + POLICY_PATH.removeprefix("ca/").removesuffix(".yaml") + "#"
_RULES = {_PREFIX + rule["name"]: rule for rule in RULES}
_PUBLIC_PARAMETERS = frozenset(
    _PREFIX + name
    for name in (
        "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
        "federal_2025_mhrtc_rate",
        "federal_2025_mhrtc_maximum_credit_per_renovation",
    )
)


def _number(value):
    if type(value) not in (int, Decimal):
        raise ValueError("exact number required; no float or boolean coercion")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("finite decimal required")
    return result


def _numeric_wire(value):
    if type(value) is not dict or set(value) != {"kind", "value"}:
        raise ValueError("closed numeric wire required")
    if value["kind"] == "integer" and type(value["value"]) is int:
        return Decimal(value["value"])
    if value["kind"] == "decimal" and type(value["value"]) is str:
        raw = value["value"]
        if (
            re.fullmatch(
                r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", raw
            )
            is None
        ):
            raise ValueError("ordinary finite decimal string required")
        try:
            parsed = Decimal(raw)
        except InvalidOperation as exc:
            raise ValueError("invalid decimal wire") from exc
        return _number(parsed)
    raise ValueError("exact integer or decimal-string wire required")


def _declarations(payload, role):
    rows = payload["program"][role]
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("compiled declarations required")
    result = {}
    for row in rows:
        key = row.get("id")
        if type(key) is not str or key in result:
            raise ValueError("unique compiled IDs required")
        result[key] = row
    return result


def _derived(key, declarations):
    rule = _RULES.get(key)
    row = declarations.get(key)
    if rule is None or rule["kind"] != "derived" or row is None:
        raise ValueError("declared MHRtC derived output required")
    dtype = {"Money": "decimal", "Rate": "decimal"}.get(
        rule["dtype"], rule["dtype"].lower()
    )
    if any(
        row.get(field) != expected
        for field, expected in (
            ("name", rule["name"]),
            ("entity", rule["entity"]),
            ("dtype", dtype),
            ("unit", rule.get("unit")),
            ("period", "Year"),
            ("semantics", "judgment" if dtype == "judgment" else "scalar"),
        )
    ):
        raise ValueError("derived declaration differs from bound role")
    versions = row.get("versions")
    if (
        type(versions) is not list
        or len(versions) != 1
        or type(versions[0]) is not dict
        or versions[0].get("effective_from") != "2025-01-01"
        or versions[0].get("effective_to") != "2025-12-31"
    ):
        raise ValueError("exact derived effective interval required")
    return row


def _assert_output(key, node, declaration, expected):
    if type(node) is not dict:
        raise ValueError("closed output object required")
    common = {"kind", "name", "id", "unit"}
    dtype = declaration["dtype"]
    if (
        node.get("id") != key
        or node.get("name") != declaration["name"]
        or node.get("unit") != declaration.get("unit")
    ):
        raise ValueError("output identity or unit mismatch")
    if dtype == "judgment":
        if (
            set(node) != common | {"outcome"}
            or node["kind"] != "judgment"
            or type(expected) is not str
            or expected not in ("holds", "not_holds")
            or node["outcome"] != expected
        ):
            raise ValueError("judgment assertion mismatch")
        return
    if (
        set(node) != common | {"dtype", "value"}
        or node["kind"] != "scalar"
        or node["dtype"] != dtype
    ):
        raise ValueError("scalar declaration mismatch")
    if dtype in ("decimal", "integer"):
        actual = _numeric_wire(node["value"])
        if dtype == "integer" and type(expected) is not int:
            raise ValueError("exact Integer expectation required")
        wanted = _number(expected)
        if dtype == "integer" and actual != actual.to_integral_value():
            raise ValueError("nonintegral Integer result")
        if actual != wanted:
            raise ValueError("exact numeric assertion mismatch")
    elif dtype == "date":
        if (
            type(expected) is not str
            or date.fromisoformat(expected).isoformat() != expected
            or not same_typed_json(node["value"], {"kind": dtype, "value": expected})
        ):
            raise ValueError("text/date assertion mismatch")
    else:
        raise ValueError("unsupported declared result type")


def validate_mhrtc_outputs(payload, request, response, expected):
    """Check a single reconstructed query's exact requested derived assertions."""
    if (
        type(request) is not dict
        or type(response) is not dict
        or type(expected) is not dict
        or not expected
        or not strict_explain_metadata(request, response)
    ):
        raise ValueError("strict Explain assertions required")
    if set(response) != {"metadata", "results"}:
        raise ValueError("closed Explain response required")
    queries, rows = request.get("queries"), response.get("results")
    if (
        type(queries) is not list
        or len(queries) != 1
        or type(rows) is not list
        or len(rows) != 1
    ):
        raise ValueError("one reconstructed query and result required")
    query, row = queries[0], rows[0]
    if type(query) is not dict or type(row) is not dict:
        raise ValueError("query/result mappings required")
    if (
        set(row) != {"entity_id", "period", "outputs", "trace"}
        or type(row["trace"]) is not dict
    ):
        raise ValueError("closed result with trace required")
    if not same_typed_json(
        query.get("period"),
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    ):
        raise ValueError("exact 2025 query period required")
    outputs = query.get("outputs")
    if (
        type(outputs) is not list
        or any(type(key) is not str for key in outputs)
        or len(set(outputs)) != len(outputs)
        or set(outputs) != set(expected)
    ):
        raise ValueError("exact expected output set required")
    for key in ("entity_id", "period", "assessment_date"):
        if (key in query) != (key in row) or not same_typed_json(
            query.get(key), row.get(key)
        ):
            raise ValueError("result does not echo reconstructed query")
    if type(row.get("outputs")) is not dict or set(row["outputs"]) != set(expected):
        raise ValueError("missing or extra result output")
    declarations = _declarations(payload, "derived")
    for key in outputs:
        node = row["outputs"][key]
        _assert_output(key, node, _derived(key, declarations), expected[key])
        trace = row["trace"].get(key)
        if (
            type(trace) is not dict
            or trace.get("entity_id") != query["entity_id"]
            or not set(node) <= set(trace)
            or not same_typed_json({field: trace[field] for field in node}, node)
        ):
            raise ValueError("root trace does not match result projection")
    return True


def validate_mhrtc_parameters(payload, expected):
    """Static assertions from fresh rebuilt IR, never a runtime execution record."""
    if type(expected) is not dict or set(expected) != _PUBLIC_PARAMETERS:
        raise ValueError("exact three public parameter assertions required")
    declarations = _declarations(payload, "parameters")
    for key in _PUBLIC_PARAMETERS:
        row, rule = declarations.get(key), _RULES[key]
        if (
            row is None
            or row.get("name") != rule["name"]
            or row.get("unit") != rule.get("unit")
            or row.get("indexed_by") is not None
        ):
            raise ValueError("public parameter declaration mismatch")
        versions = row.get("versions")
        if (
            type(versions) is not list
            or len(versions) != 1
            or type(versions[0]) is not dict
        ):
            raise ValueError("single parameter version required")
        version = versions[0]
        if (
            set(version) != {"effective_from", "effective_to", "values"}
            or version["effective_from"] != "2025-01-01"
            or version["effective_to"] != "2025-12-31"
            or type(version["values"]) is not dict
            or set(version["values"]) != {"0"}
        ):
            raise ValueError("exact parameter interval and scalar value required")
        if _numeric_wire(version["values"]["0"]) != _number(expected[key]):
            raise ValueError("parameter expectation mismatch")
        wire = version["values"]["0"]
        canonical = (
            {"kind": "decimal", "value": "0.145"}
            if rule["dtype"] == "Rate"
            else {
                "kind": "integer",
                "value": 50000 if "expenditure_limit" in key else 7250,
            }
        )
        if not same_typed_json(wire, canonical):
            raise ValueError("parameter literal differs from source-bound contract")
    return True


def observe_mhrtc_outputs(payload, request, response):
    """Decode actual typed auxiliary results without inventing fixture assertions.

    Shared result checks still apply, but these are observations, not independent
    expected-value assertions. Only the private fresh execution can authenticate
    the response; the caller must apply qualification and capacity obligations.
    """
    if type(request) is not dict or type(response) is not dict:
        raise ValueError("request and response mappings required")
    rows = response.get("results")
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        raise ValueError("one result required")
    outputs = rows[0].get("outputs")
    if type(outputs) is not dict:
        raise ValueError("typed output mapping required")
    declarations = _declarations(payload, "derived")
    observed = {}
    for key, node in outputs.items():
        declaration = _derived(key, declarations)
        if type(node) is not dict:
            raise ValueError("typed output required")
        dtype = declaration["dtype"]
        if dtype in ("decimal", "integer"):
            value = _numeric_wire(node.get("value"))
            if dtype == "integer":
                if value != value.to_integral_value():
                    raise ValueError("nonintegral Integer observation")
                value = int(value)
        elif dtype == "judgment":
            value = node.get("outcome")
        elif dtype == "date":
            wire = node.get("value")
            if type(wire) is not dict:
                raise ValueError("date wire required")
            value = wire.get("value")
        else:
            raise ValueError("unsupported observed type")
        observed[key] = value
    # Reuse closed envelope, declaration, wire and root-trace checks; this
    # self-projection is intentionally not claimed as an expected-value test.
    validate_mhrtc_outputs(payload, request, response, observed)
    return tuple(observed.items())
