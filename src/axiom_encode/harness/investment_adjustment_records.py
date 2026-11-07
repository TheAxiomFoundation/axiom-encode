"""Independent Schedule N typed record checks. No protected capture/source authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import SimpleNamespace

from .bound_investment_adjustment_operands import FLAGS, INPUT_ROWS
from .investment_adjustment_execution import (
    PREFIX,
    _same,
    verify_investment_adjustment_compiled,
)

PERIOD = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}


def _number(value):
    if type(value) not in (int, float, str) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError("finite number required")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid number") from exc
    if not result.is_finite():
        raise ValueError("finite number required")
    return result


def _typed(value, dtype, expected=False):
    if dtype == "Money":
        return _number(value)
    if dtype == "Judgment" and expected:
        if type(value) is not str or value not in ("holds", "not_holds"):
            raise ValueError("Judgment expectation required")
        return value == "holds"
    if dtype in ("Judgment", "Boolean") and type(value) is bool:
        return value
    raise ValueError("invalid typed value")


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _request(case, request, derived, caller):
    if caller not in ("cli", "pipeline"):
        raise ValueError("explicit caller required")
    if type(case) is not dict or set(case) != {
        "name",
        "entity_id",
        "period",
        "input",
        "output",
    }:
        raise ValueError("closed case required")
    if (
        type(case["name"]) is not str
        or not case["name"]
        or type(case["entity_id"]) is not str
        or not case["entity_id"]
    ):
        raise ValueError("case identity required")
    if not _same(case["period"], PERIOD):
        raise ValueError("closed2025 period required")
    money = {"schedule_n_" + n for n in INPUT_ROWS.values()} | {
        "schedule_n_resource_deduction_recovery_base"
    }
    names = money | set(FLAGS)
    if type(case["input"]) is not dict or set(case["input"]) != {
        PREFIX + "input." + n for n in names
    }:
        raise ValueError("complete explicit input contract required")
    if (
        type(case["output"]) is not dict
        or not case["output"]
        or not set(case["output"]) <= {PREFIX + n for n in derived}
    ):
        raise ValueError("known queried outputs required")
    root = "case" if caller == "cli" else case["entity_id"]
    inputs = []
    values = {}
    for canonical, value in case["input"].items():
        name = canonical.removeprefix(PREFIX + "input.")
        if name in money:
            if type(value) not in (int, float):
                raise ValueError("numeric case input required")
            parsed = _number(value)
            wire = (
                {"kind": "integer", "value": value}
                if type(value) is int
                else {"kind": "decimal", "value": str(value)}
            )
        else:
            parsed = _typed(value, "Boolean")
            wire = {"kind": "bool", "value": value}
        values[(root, name)] = parsed
        inputs.append(
            {
                "name": canonical,
                "entity": "Person",
                "entity_id": root,
                "interval": {"start": PERIOD["start"], "end": PERIOD["end"]},
                "value": wire,
            }
        )
    expected = {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": [
            {"entity_id": root, "period": PERIOD, "outputs": list(case["output"])}
        ],
    }
    if not _same(request, expected):
        raise ValueError("actual request differs from typed raw case")
    return values, root


def _trace_id(name, entity, root):
    if entity != root:
        raise ValueError("unexpected related entity")
    return PREFIX + name


@dataclass(frozen=True)
class InvestmentAdjustmentRecordObservation:
    case_sha256: str
    root: str
    outputs: tuple
    reached: tuple
    inputs: tuple


def _render(node):
    kind = node["kind"]
    if kind == "literal":
        value = node["value"]
        raw = value["value"]
        if value["kind"] == "bool":
            return "true" if raw else "false"
        if value["kind"] in ("text", "date"):
            return json.dumps(raw, ensure_ascii=False)
        return str(raw)
    if kind == "input":
        return f"input({json.dumps(node['name'])})"
    if kind == "derived":
        return PREFIX + node["name"]
    if kind == "parameter_lookup":
        return PREFIX + node["parameter"] + "[" + _render(node["index"]) + "]"
    if kind in ("add", "and", "or"):
        symbol = {"add": "+", "and": "and", "or": "or"}[kind]
        return "(" + f" {symbol} ".join(_render(x) for x in node["items"]) + ")"
    if kind in ("sub", "mul", "div", "comparison"):
        symbol = {"sub": "-", "mul": "*", "div": "/"}.get(kind) or {
            "eq": "==",
            "gte": ">=",
            "gt": ">",
            "lte": "<=",
        }[node["op"]]
        return f"({_render(node['left'])} {symbol} {_render(node['right'])})"
    if kind == "if":
        return f"(if {_render(node['condition'])} then {_render(node['then_expr'])} else {_render(node['else_expr'])})"
    if kind == "no_match":
        return f"(no match arm for {_render(node['subject'])} among [true])"
    if kind in ("min", "max"):
        return kind + "(" + ", ".join(_render(x) for x in node["items"]) + ")"
    if kind == "days_between":
        return f"days_between({_render(node['from'])}, {_render(node['to'])})"
    if kind == "sum_related":
        return f"sum_related({json.dumps(node['relation'])}, 1, 0, {_render(node['value'])})"
    raise ValueError("unknown render node")


def _wire_output(node, dtype):
    if dtype == "Judgment":
        if (
            node.get("kind") != "judgment"
            or node.get("outcome") not in ("holds", "not_holds")
            or node.get("unit") is not None
        ):
            raise ValueError("malformed Judgment output")
        return node["outcome"] == "holds"
    expected_dtype = {
        "Money": "decimal",
        "Integer": "integer",
        "Date": "date",
        "Text": "text",
        "Boolean": "bool",
    }[dtype]
    if (
        node.get("kind") != "scalar"
        or node.get("dtype") != expected_dtype
        or node.get("unit") != ("CAD" if dtype == "Money" else None)
    ):
        raise ValueError("actual scalar type/unit differs")
    wire = node.get("value")
    if (
        type(wire) is not dict
        or set(wire) != {"kind", "value"}
        or wire["kind"]
        not in (("integer", "decimal") if dtype == "Money" else (expected_dtype,))
    ):
        raise ValueError("actual wire scalar kind differs")
    raw = wire["value"]
    if dtype == "Money":
        if (wire["kind"] == "decimal" and type(raw) is not str) or (
            wire["kind"] == "integer" and type(raw) is not int
        ):
            raise ValueError("Money wire must be typed integer/decimal")
        return _number(raw)
    return _typed(raw, dtype)


def _references(node):
    if isinstance(node, dict):
        if node.get("kind") == "derived":
            yield ("derived", node["name"])
        if node.get("kind") == "parameter_lookup":
            yield ("parameter", node["parameter"])
        for value in node.values():
            yield from _references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _references(value)


def _record(derived, parameters, case, call, caller):
    if (
        type(call) is not dict
        or set(call) != {"request", "returncode", "stdout", "stderr"}
        or type(call["returncode"]) is not int
        or call["returncode"] != 0
        or call["stderr"] != ""
        or type(call["stdout"]) is not str
    ):
        raise ValueError("positive actual execution required")
    values, root = _request(case, call["request"], derived, caller)
    response = json.loads(call["stdout"], object_pairs_hook=_unique_json)
    if (
        type(response) is not dict
        or set(response) != {"metadata", "results"}
        or not _same(
            response["metadata"],
            {
                "requested_mode": "explain",
                "actual_mode": "explain",
                "fallback_reason": None,
            },
        )
        or type(response["results"]) is not list
        or len(response["results"]) != 1
    ):
        raise ValueError("strict explain response required")
    result = response["results"][0]
    if (
        type(result) is not dict
        or set(result) != {"entity_id", "period", "outputs", "trace"}
        or result["entity_id"] != root
        or not _same(result["period"], case["period"])
    ):
        raise ValueError("response query identity mismatch")
    outputs, trace = result["outputs"], result["trace"]
    if (
        type(outputs) is not dict
        or type(trace) is not dict
        or set(outputs) != set(case["output"])
    ):
        raise ValueError("actual outputs differ from required output set")
    specs = {
        name: SimpleNamespace(
            dtype={"decimal": "Money", "bool": "Boolean", "judgment": "Judgment"}[
                rule["dtype"]
            ]
        )
        for name, rule in derived.items()
    }
    computed, visiting = {}, set()
    expected_trace = {}

    def evaluate_rule(name, entity):
        key = (entity, name)
        if key in computed:
            return computed[key]
        if key in visiting:
            raise ValueError("execution cycle")
        visiting.add(key)
        deps, skipped, reads = {}, {}, {}

        def skip(node, reason):
            skipped.update(
                {
                    (
                        (
                            _trace_id(n, entity, root)
                            if kind == "derived"
                            else PREFIX + n
                        ),
                        reason,
                    ): None
                    for kind, n in _references(node)
                }
            )

        def ev(node):
            kind = node["kind"]
            if kind == "literal":
                wire = node["value"]
                return (
                    _number(wire["value"])
                    if wire["kind"] == "decimal"
                    else wire["value"]
                )
            if kind == "input":
                return values[(entity, node["name"])]
            if kind == "derived":
                deps[_trace_id(node["name"], entity, root)] = None
                return evaluate_rule(node["name"], entity)
            if kind == "parameter_lookup":
                index = ev(node["index"])
                if type(index) is not int:
                    raise ValueError("parameter index not integer")
                parameter = parameters[node["parameter"]]
                wire = parameter["versions"][0]["values"][str(index)]
                read = {
                    "parameter": PREFIX + node["parameter"],
                    "name": node["parameter"],
                    "id": PREFIX + node["parameter"],
                    "index": index,
                    "value": wire,
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "source": parameter["source"],
                }
                reads[(node["parameter"], index)] = read
                if wire["kind"] == "decimal":
                    return _number(wire["value"])
                return wire["value"]
            if kind == "comparison":
                left, right = ev(node["left"]), ev(node["right"])
                return {
                    "eq": lambda: left == right,
                    "gte": lambda: left >= right,
                    "gt": lambda: left > right,
                    "lte": lambda: left <= right,
                }[node["op"]]()
            if kind in ("and", "or"):
                for index, child in enumerate(node["items"]):
                    value = ev(child)
                    if type(value) is not bool:
                        raise ValueError("non-Boolean condition")
                    if (kind == "and" and not value) or (kind == "or" and value):
                        for rest in node["items"][index + 1 :]:
                            skip(rest, "short_circuit")
                        return value
                return kind == "and"
            if kind == "if":
                condition = ev(node["condition"])
                if type(condition) is not bool:
                    raise ValueError("non-Boolean if")
                skip(
                    node["else_expr"] if condition else node["then_expr"],
                    "branch_not_selected",
                )
                return ev(node["then_expr"] if condition else node["else_expr"])
            if kind == "no_match":
                raise ValueError("NoMatchingArm is not positive evidence")
            if kind in ("add", "min", "max"):
                items = [ev(x) for x in node["items"]]
                return (
                    sum(items)
                    if kind == "add"
                    else min(items)
                    if kind == "min"
                    else max(items)
                )
            if kind in ("sub", "mul", "div"):
                left, right = ev(node["left"]), ev(node["right"])
                return (
                    left - right
                    if kind == "sub"
                    else left * right
                    if kind == "mul"
                    else left / right
                )
            raise ValueError("unknown compiled node")

        spec = specs[name]
        value = ev(derived[name]["expr"])
        node_key = _trace_id(name, entity, root)
        node = trace.get(node_key)
        basic = (
            {"kind", "name", "id", "unit", "outcome"}
            if spec.dtype == "Judgment"
            else {"kind", "name", "id", "dtype", "unit", "value"}
        )
        keys = basic | {"entity_id", "source", "executed_expression", "dependencies"}
        if skipped:
            keys.add("not_evaluated_dependencies")
        if reads:
            keys.add("parameter_reads")
        if (
            type(node) is not dict
            or set(node) != keys
            or node["name"] != name
            or node["id"] != PREFIX + name
            or node["entity_id"] != entity
            or node["source"] != derived[name]["source"]
            or node["executed_expression"] != _render(derived[name]["expr"])
        ):
            raise ValueError(f"trace node identity/expression mismatch: {name}")
        if _wire_output(node, spec.dtype) != value:
            raise ValueError("trace value differs from typed evaluated graph")
        if not _same(node["dependencies"], list(deps)):
            raise ValueError(
                f"trace reached dependency closure differs: {name}: {node['dependencies']} != {list(deps)}"
            )
        expected_skipped = [
            {"dependency": dep, "reason": reason} for dep, reason in skipped
        ]
        if not _same(node.get("not_evaluated_dependencies", []), expected_skipped):
            raise ValueError(f"trace skipped branches differ: {name}")
        if not _same(node.get("parameter_reads", []), list(reads.values())):
            raise ValueError(f"trace parameter reads differ: {name}")
        computed[key] = value
        visiting.remove(key)
        expected_trace[node_key] = True
        return value

    for output, expected in case["output"].items():
        name = output.removeprefix(PREFIX)
        value = evaluate_rule(name, root)
        spec = specs[name]
        node = outputs[output]
        fields = (
            {"kind", "name", "id", "unit", "outcome"}
            if spec.dtype == "Judgment"
            else {"kind", "name", "id", "dtype", "unit", "value"}
        )
        if (
            type(node) is not dict
            or set(node) != fields
            or node["name"] != name
            or node["id"] != output
            or _wire_output(node, spec.dtype) != value
            or _typed(expected, spec.dtype, expected=True) != value
        ):
            raise ValueError("actual queried output or expected value differs")
    if set(trace) != set(expected_trace):
        raise ValueError("extra or absent reached trace instance")
    return InvestmentAdjustmentRecordObservation(
        hashlib.sha256(
            json.dumps(case, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        root,
        tuple(
            (name, computed[(root, name.removeprefix(PREFIX))])
            for name in case["output"]
        ),
        tuple(
            (entity, name, value) for (entity, name), value in sorted(computed.items())
        ),
        tuple(
            (entity, name, value) for (entity, name), value in sorted(values.items())
        ),
    )


def observe_investment_adjustment_record(
    bound, content, parent, guide, artifact, engine_pin, case, call, *, caller
):
    derived = verify_investment_adjustment_compiled(
        bound, content, parent, guide, artifact, engine_pin
    )
    parameters = {p["name"]: p for p in artifact["program"]["parameters"]}
    return _record(derived, parameters, case, call, caller)
