"""Standalone exact T1170 compiled/typed observation, not capture authority.

The closed lowering profile is pinned to official engine 89571cc. No generic
same-case guard, error acceptance or source-completeness consumer uses this.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from .bound_capital_gift_operands import CapitalGiftBinding

ENGINE_PIN = "89571cc2a938707fd60a5489b951135345697296"
PREFIX = "ca:policies/cra/t1-2025/capital-gains-on-gifts#"


def _same(left, right):
    return json.dumps(
        left, sort_keys=True, allow_nan=False, separators=(",", ":")
    ) == json.dumps(right, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _units():
    # Official formula.rs1125-1189, exact895 Git blob checked independently.
    return [
        {
            "name": n,
            "kind": "currency",
            "minor_units": 0 if n in {"GHp", "UGX", "RWF"} else 2,
        }
        for n in (
            "GBP",
            "USD",
            "EUR",
            "CHF",
            "NOK",
            "GHS",
            "GHp",
            "NGN",
            "UGX",
            "ZMW",
            "ETB",
            "RWF",
            "TZS",
            "DKK",
            "AMD",
            "ILS",
        )
    ] + [
        {"name": "count", "kind": "count"},
        {"name": "person", "kind": "count"},
        {"name": "ratio", "kind": "ratio"},
        {"name": "status", "kind": "custom", "label": "judgment"},
        {"name": "CAD", "kind": "currency", "minor_units": 2},
    ]


def _literal(value, decimal=False):
    if type(value) is bool:
        wire = {"kind": "bool", "value": value}
    elif type(value) is str:
        wire = {"kind": "text", "value": value}
    elif decimal:
        wire = {"kind": "decimal", "value": str(Decimal(str(value)).normalize())}
    elif type(value) is int:
        wire = {"kind": "integer", "value": value}
    else:
        wire = {"kind": "decimal", "value": str(Decimal(str(value)).normalize())}
    return {"kind": "literal", "value": wire}


def _lower(node, inputs, rules, *, condition=False, decimal=False):
    kind, *parts = node
    if kind == "name":
        name = parts[0]
        if name in ("true", "false"):
            value = _literal(name == "true")
        elif name in inputs:
            value = {"kind": "input", "name": name}
        elif rules[name].kind == "parameter":
            value = {
                "kind": "parameter_lookup",
                "parameter": name,
                "index": _literal(0),
            }
        else:
            value = {"kind": "derived", "name": name}
        if condition and name not in ("true", "false"):
            return {
                "kind": "comparison",
                "left": value,
                "op": "eq",
                "right": _literal(True),
            }
        return value
    if kind == "literal":
        return _literal(parts[1], decimal=decimal)
    if kind == "indexed":
        return {
            "kind": "parameter_lookup",
            "parameter": parts[0],
            "index": _lower(("name", parts[1]), inputs, rules),
        }
    if kind in ("Add", "Sub", "Mult", "Div"):
        operands = [_lower(p, inputs, rules, decimal=decimal) for p in parts]
        if kind == "Add":
            return {"kind": "add", "items": operands}
        return {
            "kind": {"Sub": "sub", "Mult": "mul", "Div": "div"}[kind],
            "left": operands[0],
            "right": operands[1],
        }
    if kind in ("Eq", "GtE", "LtE"):
        return {
            "kind": "comparison",
            "left": _lower(parts[0], inputs, rules),
            "op": {"Eq": "eq", "GtE": "gte", "LtE": "lte"}[kind],
            "right": _lower(parts[1], inputs, rules),
        }
    if kind in ("And", "Or"):
        # Official parser lowers left-associative binary Boolean expressions.
        items = [_lower(p, inputs, rules, condition=True) for p in parts]
        value = items[0]
        for item in items[1:]:
            value = {"kind": kind.lower(), "items": [value, item]}
        return value
    if kind == "if":
        return {
            "kind": "if",
            "condition": _lower(parts[0], inputs, rules, condition=True),
            "then_expr": _lower(parts[1], inputs, rules, decimal=decimal),
            "else_expr": _lower(parts[2], inputs, rules, decimal=decimal),
        }
    if kind == "match_true":
        subject = _lower(("name", parts[0]), inputs, rules)
        return {
            "kind": "if",
            "condition": {
                "kind": "comparison",
                "left": subject,
                "op": "eq",
                "right": _literal(True),
            },
            "then_expr": _lower(parts[1], inputs, rules, decimal=decimal),
            "else_expr": {
                "kind": "no_match",
                "subject": subject,
                "patterns": [
                    {"kind": "literal", "value": {"kind": "bool", "value": True}}
                ],
            },
        }
    if kind == "call":
        name, *args = parts
        if name == "sum":
            relation, provider = args[0][1:]
            return {
                "kind": "sum_related",
                "relation": PREFIX + "relation." + relation,
                "current_slot": 1,
                "related_slot": 0,
                "value": {"kind": "derived", "name": provider},
                "where": None,
            }
        if name == "days_between":
            return {
                "kind": name,
                "from": _lower(args[0], inputs, rules),
                "to": _lower(args[1], inputs, rules),
            }
        return {
            "kind": name,
            "items": [_lower(a, inputs, rules, decimal=decimal) for a in args],
        }
    raise ValueError("unsupported bound AST")


def _indexed(items, key):
    if not isinstance(items, list):
        raise ValueError("list required")
    result = {}
    for item in items:
        if (
            not isinstance(item, dict)
            or type(item.get(key)) is not str
            or item[key] in result
        ):
            raise ValueError("duplicate or malformed declaration")
        result[item[key]] = item
    return result


def _compiled(binding, artifact, engine_pin):
    if (
        engine_pin != ENGINE_PIN
        or not isinstance(artifact, dict)
        or set(artifact)
        != {"artifact_format_version", "engine_version", "program", "metadata"}
    ):
        raise ValueError("unknown compiled profile")
    if (
        type(artifact["artifact_format_version"]) is not int
        or artifact["artifact_format_version"] != 2
        or artifact["engine_version"] != "0.2.2"
    ):
        raise ValueError("unknown artifact format/version")
    program = artifact["program"]
    if (
        not isinstance(program, dict)
        or set(program) != {"module", "units", "relations", "parameters", "derived"}
        or not _same(program["units"], _units())
    ):
        raise ValueError("compiled union/unit profile mismatch")
    inputs = {n: (entity, dtype) for n, entity, dtype in binding.inputs}
    rules = {r.name: r for r in binding.rules}
    derived = _indexed(program["derived"], "name")
    parameters = _indexed(program["parameters"], "name")
    relations = _indexed(program["relations"], "name")
    expected_sets = (
        {r.name for r in binding.rules if r.kind == "derived"},
        {r.name for r in binding.rules if r.kind == "parameter"},
        {
            PREFIX + "relation." + r.name
            for r in binding.rules
            if r.kind == "data_relation"
        },
    )
    if (set(derived), set(parameters), set(relations)) != expected_sets:
        raise ValueError("complete compiled declaration partition mismatch")
    source_title = binding.source.source[slice(*binding.source.records[0].span)]
    for rule in binding.rules:
        name = rule.name
        if rule.kind == "data_relation":
            expected = {
                "name": PREFIX + "relation." + name,
                "arity": 2,
                "slot_entities": ["SourceEntry", "Person"],
            }
            if not _same(relations[expected["name"]], expected):
                raise ValueError("relation type/direction mismatch")
            continue
        expected = {
            "id": PREFIX + name,
            "name": name,
            "unit": "CAD" if rule.dtype == "Money" else None,
            "source": source_title,
            "source_url": None,
            "corpus_citation_path": binding.source.citation,
        }
        version = {"effective_from": "2025-01-01", "effective_to": "2025-12-31"}
        if rule.kind == "parameter":
            if rule.dtype == "Date":
                _, year, day, index = rule.expression
                expected["indexed_by"] = index
                version["values"] = {str(year): {"kind": "date", "value": day}}
            else:
                expected["indexed_by"] = None
                value = rule.expression[2]
                version["values"] = {
                    "0": _literal(value, decimal=rule.dtype == "Rate")["value"]
                }
            expected["versions"] = [version]
            actual = parameters[name]
        else:
            dtype = {
                "Money": "decimal",
                "Integer": "integer",
                "Date": "date",
                "Text": "text",
                "Boolean": "bool",
                "Judgment": "judgment",
            }[rule.dtype]
            semantics = "judgment" if rule.dtype == "Judgment" else "scalar"
            expr = _lower(
                rule.expression,
                inputs,
                rules,
                condition=rule.dtype == "Judgment",
                decimal=rule.dtype == "Money",
            )
            expected.update(
                entity=rule.entity,
                dtype=dtype,
                period="Year",
                semantics=semantics,
                expr=expr,
            )
            version.update(semantics=semantics, expr=expr)
            expected["versions"] = [version]
            actual = derived[name]
        if not _same(actual, expected):
            raise ValueError(f"compiled declaration differs: {name}")
    metadata = artifact["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "evaluation_order",
        "fast_path",
        "input_catalog",
    }:
        raise ValueError("compiled metadata schema mismatch")
    # Indexed-by source_year appears in official catalog, not a supplied fact.
    slots = set(inputs) | {"t1170_form_year"}
    expected_catalog = [
        {
            "slot": n,
            "canonical_request_name": PREFIX + "input." + n,
            "request_names": [PREFIX + "input." + n],
        }
        for n in sorted(slots)
    ]
    if not _same(metadata["input_catalog"], expected_catalog):
        raise ValueError("input catalog or synthetic index slot mismatch")
    order = metadata["evaluation_order"]
    if (
        not isinstance(order, list)
        or len(order) != len(derived)
        or set(order) != set(derived)
    ):
        raise ValueError("evaluation order does not partition derived rules")
    positions = {name: i for i, name in enumerate(order)}
    if any(
        positions[d] >= positions[r.name]
        for r in binding.rules
        if r.kind == "derived"
        for d in r.dependencies
        if d in derived
    ):
        raise ValueError("evaluation order violates dependencies")
    # Fast-path declarations do not grant actual explain metadata permission.
    fast = metadata["fast_path"]
    if (
        not isinstance(fast, dict)
        or set(fast) != {"strategy", "compatible", "blockers"}
        or fast["strategy"] != "generic_bulk"
        or fast["compatible"] is not False
        or not isinstance(fast["blockers"], list)
        or any(type(x) is not str for x in fast["blockers"])
    ):
        raise ValueError("unreviewed fast metadata profile")
    expected_blockers = []

    def blockers(node, name):
        if isinstance(node, dict):
            if node.get("kind") == "sum_related":
                expected_blockers.append(
                    name
                    + ": fast mode does not yet support sum_related over related derived values"
                )
            if node.get("kind") == "days_between":
                expected_blockers.append(
                    name
                    + ": bulk fast mode does not yet support days_between; explain mode and the generic dense path do"
                )
            for value in node.values():
                blockers(value, name)
        elif isinstance(node, list):
            for value in node:
                blockers(value, name)

    for name, declaration in derived.items():
        blockers(declaration["expr"], name)
        blockers(declaration["versions"][0]["expr"], name)
    if Counter(fast["blockers"]) != Counter(expected_blockers):
        raise ValueError("fast metadata blockers differ from compiled graph")
    module = program["module"]
    if (
        not isinstance(module, dict)
        or set(module) != {"summary", "source_verification"}
        or type(module["summary"]) is not str
        or not _same(
            module["source_verification"],
            {
                "corpus_citation_path": binding.source.citation,
                "source_sha256": binding.source.source_sha256,
            },
        )
    ):
        raise ValueError("compiled source identity mismatch")
    return derived, parameters


def _number(value):
    if type(value) not in (int, float, str):
        raise ValueError("not a number")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("nonfinite number")
    return result


def _typed(value, dtype, *, expected=False):
    if dtype == "Money":
        if type(value) not in (int, float):
            raise ValueError("money fact must be explicit numeric scalar")
        return _number(value)
    if dtype == "Integer":
        if type(value) is not int:
            raise ValueError("integer fact must not be bool/float")
        return value
    if dtype == "Boolean":
        if type(value) is not bool:
            raise ValueError("Boolean fact must be bool")
        return value
    if dtype == "Date":
        if expected:
            if type(value) is not date:
                raise ValueError("Date expected output must retain YAML Date type")
            return value
        if type(value) is not str:
            raise ValueError("Date fact must be ISO string")
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("noncanonical Date")
        return parsed
    if dtype == "Text":
        if type(value) is not str:
            raise ValueError("Text fact required")
        return value
    if dtype == "Judgment" and expected:
        if value not in ("holds", "not_holds") or type(value) is not str:
            raise ValueError("Judgment expectation required")
        return value == "holds"
    raise ValueError("unsupported factual type")


def _wire_input(raw, dtype):
    _typed(raw, dtype)
    if dtype == "Date":
        return {"kind": "date", "value": raw}
    if type(raw) is bool:
        return {"kind": "bool", "value": raw}
    if type(raw) is int:
        return {"kind": "integer", "value": raw}
    if type(raw) is float:
        return {"kind": "decimal", "value": str(raw)}
    return {"kind": "text", "value": raw}


def capital_gift_case_key(case, binding):
    """Private type-preserving identity; never changes generic case_key.

    Date tags are allowed only in declared Date expected-output paths. All node
    kinds are tagged, so a string/list that looks like a Date tag cannot collide.
    Raw companion bytes and ordinal must additionally bind future live capture.
    """
    rules = {PREFIX + r.name: r for r in binding.rules}
    active = set()

    def tag(value, path):
        if type(value) is date:
            if (
                len(path) != 2
                or path[0] != "output"
                or path[1] not in rules
                or rules[path[1]].dtype != "Date"
            ):
                raise ValueError("Date outside compiled Date expectation")
            return ["date", value.isoformat()]
        if type(value) is bool:
            return ["bool", value]
        if type(value) is int:
            return ["int", str(value)]
        if type(value) is float:
            _number(value)
            return ["float", value.hex()]
        if type(value) is str:
            return ["str", value]
        if type(value) not in (dict, list) or id(value) in active:
            raise ValueError("unsupported case value/cycle")
        active.add(id(value))
        try:
            if isinstance(value, list):
                return ["list", [tag(v, path + (i,)) for i, v in enumerate(value)]]
            if any(type(k) is not str for k in value):
                raise ValueError("case object keys must be strings")
            return ["object", [[k, tag(value[k], path + (k,))] for k in sorted(value)]]
        finally:
            active.remove(id(value))

    encoded = json.dumps(
        ["capital-gift-case/v1", tag(case, ())],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _request(case, request, binding, *, refusal=False):
    result_field = "expected_refusal" if refusal else "output"
    if (
        type(case) is not dict
        or set(case) != {"name", "entity_id", "period", "input", result_field}
        or any(type(case[k]) is not str or not case[k] for k in ("name", "entity_id"))
    ):
        raise ValueError("closed explicit case identity required")
    if not _same(
        case["period"],
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    ):
        raise ValueError("unsupported case frame")
    if (
        not isinstance(case["input"], dict)
        or not isinstance(case[result_field], dict)
        or not case[result_field]
    ):
        raise ValueError("explicit input/output mappings required")
    if refusal:
        contract = case[result_field]
        if (
            set(contract) != {"output", "reason"}
            or contract["reason"] != "NoMatchingArm"
            or type(contract["output"]) is not str
        ):
            raise ValueError("closed direct refusal contract required")
        output_names = [contract["output"]]
    else:
        output_names = list(case["output"])
    root = case["entity_id"]
    specs = {
        PREFIX + "input." + n: (n, entity, dtype) for n, entity, dtype in binding.inputs
    }
    rules = {PREFIX + r.name: r for r in binding.rules if r.kind == "derived"}
    relations = {
        PREFIX + "relation." + r.name
        for r in binding.rules
        if r.kind == "data_relation"
    }
    if not set(output_names) <= set(rules):
        raise ValueError("unknown queried output")
    entities = {rules[n].entity for n in output_names}
    if len(entities) != 1:
        raise ValueError("mixed root entity kinds")
    root_kind = next(iter(entities))
    interval = {"start": "2025-01-01", "end": "2025-12-31"}
    facts, edges, values = [], [], {}

    def add(name, raw, entity_id, kind):
        if name not in specs or specs[name][1] != kind:
            raise ValueError("unknown fact or wrong actual entity kind")
        local, _, dtype = specs[name]
        key = (entity_id, local)
        if key in values:
            raise ValueError("duplicate factual assignment")
        values[key] = _typed(raw, dtype)
        facts.append(
            {
                "name": name,
                "entity": kind,
                "entity_id": entity_id,
                "interval": interval,
                "value": _wire_input(raw, dtype),
            }
        )

    for name, value in case["input"].items():
        if name in relations:
            if refusal or root_kind != "Person" or not isinstance(value, list):
                raise ValueError("relation list requires claimant")
            for i, row in enumerate(value, 1):
                if type(row) is not dict:
                    raise ValueError("explicit related fact object required")
                provider = f"{root}-{name}-{i}"
                if provider == root:
                    raise ValueError("row identity collision")
                edges.append(
                    {"name": name, "tuple": [provider, root], "interval": interval}
                )
                for key, raw in row.items():
                    add(key, raw, provider, "SourceEntry")
        else:
            add(name, value, root, root_kind)
    expected = {
        "mode": "explain",
        "dataset": {"inputs": facts, "relations": edges},
        "queries": [
            {
                "entity_id": root,
                "period": case["period"],
                "outputs": output_names,
            }
        ],
    }
    if not _same(request, expected):
        raise ValueError("actual request differs from typed case/transport")
    if not refusal:
        for name, value in case["output"].items():
            _typed(value, rules[name].dtype, expected=True)
    return values, edges, root


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


def _trace_id(name, entity, root):
    return (
        PREFIX + name + ("" if entity == root else "@entity:" + entity.encode().hex())
    )


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


@dataclass(frozen=True)
class CapitalGiftRecordObservation:
    case_key: str
    root_entity_id: str
    outputs: tuple[tuple[str, Any], ...]
    reached: tuple[tuple[str, str, Any], ...]
    facts: tuple[tuple[str, str, Any], ...]


def _record(binding, derived, parameters, case, call):
    if (
        type(call) is not dict
        or set(call) != {"request", "returncode", "stdout", "stderr"}
        or type(call["returncode"]) is not int
        or call["returncode"] != 0
        or call["stderr"] != ""
        or type(call["stdout"]) is not str
    ):
        raise ValueError("positive actual execution required")
    values, edges, root = _request(case, call["request"], binding)
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
    specs = {r.name: r for r in binding.rules}
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
                if wire["kind"] == "date":
                    return date.fromisoformat(wire["value"])
                if wire["kind"] == "decimal":
                    return _number(wire["value"])
                return wire["value"]
            if kind == "comparison":
                left, right = ev(node["left"]), ev(node["right"])
                return {
                    "eq": lambda: left == right,
                    "gte": lambda: left >= right,
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
            if kind == "days_between":
                return (ev(node["to"]) - ev(node["from"])).days
            if kind == "sum_related":
                total = Decimal(0)
                for edge in edges:
                    if edge["name"] == node["relation"] and edge["tuple"][1] == entity:
                        provider = edge["tuple"][0]
                        target = node["value"]["name"]
                        deps[_trace_id(target, provider, root)] = None
                        total += evaluate_rule(target, provider)
                return total
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
    return CapitalGiftRecordObservation(
        capital_gift_case_key(case, binding),
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


@dataclass(frozen=True)
class CapitalGiftRefusalObservation:
    """Validated raw direct failure, never a successful trace or capture grant."""

    case_key: str
    principal: str
    selector: str
    root: str
    facts: tuple


def direct_refusal_pairs(
    binding, positive_records, positive_cases, refusals, refusal_cases
):
    """Pair isolated roots only, after both individual record validators ran.

    This does not establish provider lineage or source obligation coverage.
    Raw case keys/identities remain intact; only comparison uses root roles.
    """
    if len(positive_records) != len(positive_cases) or len(refusals) != len(
        refusal_cases
    ):
        raise ValueError("record/case partition mismatch")
    if any(
        record.case_key != capital_gift_case_key(case, binding)
        for records, cases in (
            (positive_records, positive_cases),
            (refusals, refusal_cases),
        )
        for record, case in zip(records, cases, strict=True)
    ):
        raise ValueError("pair case identity differs from validated record")
    pairs = []
    for refusal, case in zip(refusals, refusal_cases, strict=True):
        matches = []
        for positive, other in zip(positive_records, positive_cases, strict=True):
            if (
                not _same(case["period"], other["period"])
                or PREFIX + refusal.principal not in dict(positive.outputs)
                or any(
                    entity != positive.root_entity_id for entity, _, _ in positive.facts
                )
                or any(isinstance(value, list) for value in other["input"].values())
            ):
                continue
            left = {name: value for _, name, value in positive.facts}
            right = dict(refusal.facts)
            if set(left) != set(right):
                continue
            changed = [
                name
                for name in left
                if type(left[name]) is not type(right[name])
                or left[name] != right[name]
            ]
            if len(changed) == 1 and type(left[changed[0]]) is type(right[changed[0]]):
                matches.append(
                    (refusal.principal, changed[0], positive.case_key, refusal.case_key)
                )
        if not matches:
            raise ValueError("direct refusal lacks an exact one-fact positive pair")
        pairs.extend(matches)
    return tuple(pairs)


def _direct_refusal(binding, artifact, engine_pin, case, call):
    """Derive the official error from a false, source-bound direct selector.

    This standalone validator supplies no freshness or paired-witness authority.
    The private materialized context must establish both before consumer use.
    """
    derived, parameters = _compiled(binding, artifact, engine_pin)
    if (
        type(call) is not dict
        or set(call) != {"request", "returncode", "stdout", "stderr"}
        or type(call["returncode"]) is not int
        or call["returncode"] != 1
        or call["stdout"] != ""
        or type(call["stderr"]) is not str
    ):
        raise ValueError("exact raw refusal process required")
    values, edges, root = _request(case, call["request"], binding, refusal=True)
    if edges:
        raise ValueError("provider refusal requires separate lineage authority")
    principal = case["expected_refusal"]["output"].removeprefix(PREFIX)
    # These are the three direct principal roles in the complete static contract.
    # A matching name alone cannot pass _compiled's independent source lowering.
    if principal not in {
        "federal_2025_t1170_gift_capital_gain_column_5",
        "t1170_prior_exchange_gain_at_zero_inclusion",
        "t1170_partnership_special_capital_gain_per_entry",
    }:
        raise ValueError("unsupported direct principal refusal")
    rule = derived[principal]
    branch = rule["expr"]
    if branch["kind"] != "if" or branch["else_expr"]["kind"] != "no_match":
        raise ValueError("source-bound true-only principal required")
    subject = branch["else_expr"]["subject"]
    if subject["kind"] != "derived":
        raise ValueError("source-owned Boolean selector required")
    computed, visiting = {}, set()

    def evaluate(name):
        if name in computed:
            return computed[name]
        if name in visiting:
            raise ValueError("selector dependency cycle")
        visiting.add(name)
        value = ev(derived[name]["expr"])
        visiting.remove(name)
        computed[name] = value
        return value

    def ev(node):
        kind = node["kind"]
        if kind == "literal":
            wire = node["value"]
            return (
                _number(wire["value"]) if wire["kind"] == "decimal" else wire["value"]
            )
        if kind == "input":
            return values[(root, node["name"])]
        if kind == "derived":
            return evaluate(node["name"])
        if kind == "parameter_lookup":
            index = ev(node["index"])
            if type(index) is not int:
                raise ValueError("parameter index must be integer")
            wire = parameters[node["parameter"]]["versions"][0]["values"][str(index)]
            if wire["kind"] == "date":
                return date.fromisoformat(wire["value"])
            return (
                _number(wire["value"]) if wire["kind"] == "decimal" else wire["value"]
            )
        if kind == "comparison":
            left, right = ev(node["left"]), ev(node["right"])
            return {
                "eq": lambda: left == right,
                "gte": lambda: left >= right,
                "lte": lambda: left <= right,
            }[node["op"]]()
        if kind in ("and", "or"):
            for child in node["items"]:
                value = ev(child)
                if type(value) is not bool:
                    raise ValueError("Boolean selector operand required")
                if (kind == "and" and not value) or (kind == "or" and value):
                    return value
            return kind == "and"
        if kind == "if":
            condition = ev(node["condition"])
            if type(condition) is not bool:
                raise ValueError("Boolean selector condition required")
            return ev(node["then_expr"] if condition else node["else_expr"])
        if kind == "days_between":
            return (ev(node["to"]) - ev(node["from"])).days
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
        # An error in a helper or a relation traversal is not the requested direct
        # false-selector outcome, even if the process also exited with code one.
        raise ValueError("unsupported direct selector evaluation")

    selector = subject["name"]
    if evaluate(selector) is not False or ev(branch["condition"]) is not False:
        raise ValueError("actual typed selector must independently be false")
    expected = (
        f"no `match` arm in `{PREFIX}{principal}` covers `{selector}` = false "
        "(arms: true); add an arm for it or a final `_ =>` arm\n"
    )
    if call["stderr"] != expected:
        raise ValueError("actual error differs from source-derived official profile")
    return CapitalGiftRefusalObservation(
        capital_gift_case_key(case, binding),
        principal,
        selector,
        root,
        tuple(sorted((name, value) for (entity, name), value in values.items())),
    )


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate actual response key")
        result[key] = value
    return result


@dataclass(frozen=True)
class CapitalGiftPositiveObservation:
    """Positive typed records and role witnesses, never full-source authority.

    Direct NoMatchingArm and provider refusal evidence is intentionally absent.
    A later source consumer must not infer blocking principal observations here.
    """

    content_sha256: str
    artifact_sha256: str
    companion_sha256: str
    records: tuple[CapitalGiftRecordObservation, ...]
    root_role_pairs: tuple[tuple[str, str, str, str], ...]
    pending_refusal_contract: tuple[str, ...] = (
        "direct_gift_principal",
        "direct_prior_exchange_principal",
        "direct_partnership_principal",
        "provider_refusal_lineage",
    )


def _root_pairs(records, cases):
    """Root-only scenario renaming; individual actual identities stay exact.

    The closed static AST contains no ID-inspection primitive. Related-row
    scenarios are excluded: no provider graph alpha-equivalence is asserted.
    """
    pairs = []
    for i, left in enumerate(cases):
        if any(type(v) in (dict, list) for v in left["input"].values()):
            continue
        if any(
            entity != records[i].root_entity_id for entity, _, _ in records[i].facts
        ):
            continue
        for j in range(i + 1, len(cases)):
            right = cases[j]
            if any(type(v) in (dict, list) for v in right["input"].values()) or any(
                entity != records[j].root_entity_id for entity, _, _ in records[j].facts
            ):
                continue
            if not _same(left["period"], right["period"]) or set(left["input"]) != set(
                right["input"]
            ):
                continue
            changed = [
                name
                for name in left["input"]
                if not _same(left["input"][name], right["input"][name])
            ]
            if len(changed) != 1:
                continue
            a, b = dict(records[i].outputs), dict(records[j].outputs)
            for name in sorted(set(a) & set(b)):
                if type(a[name]) is type(b[name]) and a[name] != b[name]:
                    pairs.append(
                        (changed[0], name, records[i].case_key, records[j].case_key)
                    )
    return tuple(pairs)


def _positive_coverage(binding, records, cases):
    pairs = _root_pairs(records, cases)
    witnessed = {
        (fact.removeprefix(PREFIX + "input."), output.removeprefix(PREFIX))
        for fact, output, _, _ in pairs
    }
    required = {
        ("completing_t1170", "t1170_enclosure_required"),
        ("needs_t1170_additional_space", "t1170_additional_sheet_required"),
        (
            "t1170_fts_schedule3_additional_gain_applies",
            "t1170_fts_t691_reporting_required",
        ),
        (
            "t1170_nonprescribed_partnership_exchange_applies",
            "t1170_partnership_special_calculation_required",
        ),
        (
            "t1170_nonprescribed_partnership_exchange_applies",
            "t1170_partnership_schedule3_reporting_required",
        ),
        (
            "t1170_eligible_amount_of_gift_column_6",
            "federal_2025_t1170_gain_eligible_zero_inclusion_column_7",
        ),
        (
            "t1170_eligible_amount_of_gift_column_6",
            "federal_2025_t1170_gain_subject_to_50_percent_inclusion_column_8",
        ),
        (
            "t1170_eligible_amount_of_gift_column_6",
            "federal_2025_t1170_taxable_capital_gain_from_gift",
        ),
        (
            "t1170_partnership_exchange_gain_otherwise_determined",
            "t1170_partnership_special_capital_gain_per_entry",
        ),
        (
            "t1170_partnership_adjusted_basis_without_distributions",
            "t1170_partnership_special_capital_gain_per_entry",
        ),
        (
            "t1170_prior_exchange_capital_gain",
            "t1170_prior_exchange_gain_at_zero_inclusion",
        ),
    }
    required.update(
        (name, "t1170_gift_within_form_scope")
        for name in (
            "t1170_qualified_donee",
            "t1170_property_source_class",
            "t1170_property_instrument_kind",
            "t1170_disposition_date",
            "t1170_ecological_donee_not_private_foundation",
        )
    )
    required.update(
        (name, "t1170_exchange_extension_conditions_met")
        for name in (
            "t1170_exchanged_asset_kind",
            "t1170_gift_is_of_securities_received_in_exchange",
            "t1170_exchange_right_at_issue",
            "t1170_exchange_right_at_disposition",
            "t1170_securities_only_exchange_consideration",
            "t1170_exchange_date",
        )
    )
    if not required <= witnessed:
        raise ValueError("missing root-role positive/blocking helper or money witness")
    reached_names = {name for r in records for _, name, _ in r.reached}
    if reached_names != {r.name for r in binding.rules if r.kind == "derived"}:
        raise ValueError("missing whole graph actual derived observation")
    # One-case multirow checks require exact actual edges and provider traces
    # already validated; they do not compare/relabel providers across cases.
    pooling = set()
    for record, case in zip(records, cases, strict=True):
        outputs = dict(record.outputs)
        for category, line in (
            ("shares_units", "68230"),
            ("bonds_other", "68250"),
            ("ecological_land", "68252"),
        ):
            key = PREFIX + "relation.t1170_" + category + "_row_for_return"
            rows = case["input"].get(key, [])
            query = PREFIX + "t1170_" + category + "_zero_inclusion_line_" + line
            if len(rows) < 2 or query not in outputs:
                continue

            def amount(row, role):
                return _number(row[PREFIX + "input.t1170_" + role])

            gains = [
                amount(r, "proceeds_of_disposition_column_2")
                - amount(r, "adjusted_cost_base_column_3")
                - amount(r, "outlays_and_expenses_column_4")
                for r in rows
            ]
            proceeds = [amount(r, "proceeds_of_disposition_column_2") for r in rows]
            eligible = [amount(r, "eligible_amount_of_gift_column_6") for r in rows]
            if (
                sum(proceeds)
                and sum(gains) * sum(eligible) / sum(proceeds) != outputs[query]
            ):
                pooling.add(category)
        rows = case["input"].get(
            PREFIX + "relation.t1170_partnership_exchange_for_return", []
        )
        query = PREFIX + "t1170_partnership_gain_for_schedule3_line17400"
        if len(rows) >= 2 and query in outputs:

            def total(name):
                return sum(_number(row[PREFIX + "input." + name]) for row in rows)

            wrong = min(
                total("t1170_partnership_exchange_gain_otherwise_determined"),
                max(
                    total("t1170_exchanged_partnership_interest_cost")
                    + total("t1170_donor_partnership_capital_contributions")
                    - total("t1170_partnership_adjusted_basis_without_distributions"),
                    0,
                ),
            )
            if wrong != outputs[query]:
                pooling.add("partnership")
    if pooling != {"shares_units", "bonds_other", "ecological_land", "partnership"}:
        raise ValueError("missing per-entry-before-aggregate distinction")
    return pairs


def observe_capital_gift(
    binding, content, source, guide, artifact, calls, companion, engine_pin
):
    """Validate unsigned standalone records; private fresh capture is future work."""
    import yaml

    from .bound_capital_gift_operands import _load, _UniqueLoader

    try:
        if (
            not isinstance(binding, CapitalGiftBinding)
            or not binding.bound(content, source, guide)
            or type(companion) is not str
            or len(companion) > 1_000_000
        ):
            return None
        raw = yaml.load(companion, Loader=_UniqueLoader)
        cases = raw["cases"] if type(raw) is dict and set(raw) == {"cases"} else raw
        if (
            type(cases) is not list
            or type(calls) is not list
            or not cases
            or len(cases) != len(calls)
        ):
            return None
        derived, parameters = _compiled(binding, artifact, engine_pin)
        if (
            artifact["program"]["module"]["summary"]
            != _load(content)["module"]["summary"]
        ):
            return None
        records = tuple(
            _record(binding, derived, parameters, case, call)
            for case, call in zip(cases, calls, strict=True)
        )
        if len({r.case_key for r in records}) != len(records):
            return None
        pairs = _positive_coverage(binding, records, cases)
        return CapitalGiftPositiveObservation(
            binding.content_sha256,
            hashlib.sha256(
                json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
            hashlib.sha256(companion.encode()).hexdigest(),
            records,
            pairs,
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        RecursionError,
        ArithmeticError,
        yaml.YAMLError,
    ):
        return None
