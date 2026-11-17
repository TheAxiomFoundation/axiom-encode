"""Standalone observation of the complete retirement-transfer graph.

The private capture context checks this closed graph; it grants no source
coverage or publication authority.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

import yaml

from axiom_encode.bound_retirement_transfer_form import bind_retirement_transfer_form

from .bound_form_operands import _wire_number
from .same_case_execution import case_key, strict_explain_metadata

# Official 89571cc2 src/formula.rs::lower_module, SHA256
# 7d7a8c89fb41e984094f809243e3d426d44aba0cf62dd1e614b3c8403696c305.
# This immutable observer-local profile does not change global engine policy.
_PROFILE_PIN = "89571cc2a938707fd60a5489b951135345697296"
_CURRENCIES = (
    ("GBP", 2),
    ("USD", 2),
    ("EUR", 2),
    ("CHF", 2),
    ("NOK", 2),
    ("GHS", 2),
    ("GHp", 0),
    ("NGN", 2),
    ("UGX", 0),
    ("ZMW", 2),
    ("ETB", 2),
    ("RWF", 0),
    ("TZS", 2),
    ("DKK", 2),
    ("AMD", 2),
    ("ILS", 2),
)


def _units():
    return [
        {"name": n, "kind": "currency", "minor_units": v} for n, v in _CURRENCIES
    ] + [
        {"name": "count", "kind": "count"},
        {"name": "person", "kind": "count"},
        {"name": "ratio", "kind": "ratio"},
        {"name": "status", "kind": "custom", "label": "judgment"},
        {"name": "CAD", "kind": "currency", "minor_units": 2},
    ]


def _equal(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False
    )


def _number(v):
    if type(v) not in (str, int, float):
        raise ValueError("numeric value required")
    n = Decimal(str(v))
    if not n.is_finite():
        raise ValueError("finite value required")
    return n


def _lower(node, inputs, rules, prefix, boolean=False, integer=False):
    """Lower only AST shapes already accepted by the closed static binder."""
    if isinstance(node, ast.Name):
        n = node.id
        if n in inputs:
            result = {"kind": "input", "name": n}
            if boolean:
                return {
                    "kind": "comparison",
                    "left": result,
                    "op": "eq",
                    "right": {
                        "kind": "literal",
                        "value": {"kind": "bool", "value": True},
                    },
                }
            return result
        if rules[n]["kind"] == "parameter":
            return {
                "kind": "parameter_lookup",
                "parameter": n,
                "index": {"kind": "literal", "value": {"kind": "integer", "value": 0}},
            }
        return {"kind": "derived", "name": n}
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return {
            "kind": "literal",
            "value": (
                {"kind": "integer", "value": node.value}
                if integer and type(node.value) is int
                else {"kind": "decimal", "value": str(node.value)}
            ),
        }
    if isinstance(node, ast.BinOp):
        a, b = (_lower(x, inputs, rules, prefix) for x in (node.left, node.right))
        if isinstance(node.op, ast.Add):
            return {"kind": "add", "items": [a, b]}
        return {
            "kind": {ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(node.op)],
            "left": a,
            "right": b,
        }
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return {
            "kind": "parameter_lookup",
            "parameter": node.value.id,
            "index": _lower(node.slice, inputs, rules, prefix, integer=True),
        }
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        values = [_lower(n, inputs, rules, prefix, True) for n in node.values]
        result = values[0]
        for value in values[1:]:
            result = {"kind": "and", "items": [result, value]}
        return result
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return {
            "kind": "comparison",
            "left": _lower(node.left, inputs, rules, prefix),
            "op": {ast.Gt: "gt", ast.Lt: "lt", ast.Eq: "eq"}[type(node.ops[0])],
            "right": _lower(node.comparators[0], inputs, rules, prefix, integer=True),
        }
    if isinstance(node, ast.If) and len(node.body) == len(node.orelse) == 1:
        return {
            "kind": "if",
            "condition": _lower(node.test, inputs, rules, prefix, True),
            "then_expr": _lower(node.body[0].value, inputs, rules, prefix),
            "else_expr": _lower(node.orelse[0].value, inputs, rules, prefix),
        }
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("min", "max"):
            return {
                "kind": node.func.id,
                "items": [_lower(n, inputs, rules, prefix) for n in node.args],
            }
        if (
            node.func.id == "sum"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Attribute)
        ):
            n = node.args[0]
            return {
                "kind": "sum_related",
                "relation": prefix + "relation." + n.value.id,
                "current_slot": 1,
                "related_slot": 0,
                "value": {"kind": "derived", "name": n.attr},
                "where": None,
            }
    raise ValueError("unsupported static AST")


def _compiled(binding, content, artifact, prefix):
    if (
        type(artifact.get("artifact_format_version")) is not int
        or artifact["artifact_format_version"] != 2
    ):
        raise ValueError("artifact format differs")
    data = yaml.safe_load(content)
    inputs = {i["name"]: i for i in data["inputs"]}
    rules = {r["name"]: r for r in data["rules"]}
    expressions = {}
    parameters = []
    derived = []
    relations = []
    for name, r in rules.items():
        if r["kind"] == "data_relation":
            relations.append(
                {
                    "name": prefix + "relation." + name,
                    "arity": 2,
                    "slot_entities": ["Person", "Person"],
                }
            )
            continue
        version = r["versions"][0]
        shared = dict(
            id=prefix + name,
            name=name,
            unit=r.get("unit"),
            source=r["source"],
            source_url=None,
            corpus_citation_path=binding.source.citation,
        )
        dates = {k: version[k] for k in ("effective_from", "effective_to")}
        if r["kind"] == "parameter":
            if "values" in version:
                wire_values = {
                    str(k): {"kind": "date", "value": v}
                    for k, v in version["values"].items()
                }
            else:
                value = version["formula"]
                wire_values = {
                    "0": {"kind": "integer", "value": value}
                    if type(value) is int
                    else {"kind": "decimal", "value": str(value)}
                }
            parameters.append(
                dict(
                    **shared,
                    indexed_by=r.get("indexed_by"),
                    versions=[dict(**dates, values=wire_values)],
                )
            )
            continue
        tree = ast.parse(version["formula"], mode="exec").body[0]
        expr = _lower(
            tree.value if isinstance(tree, ast.Expr) else tree,
            inputs,
            rules,
            prefix,
            r["dtype"] == "Judgment",
        )
        expressions[name] = expr
        semantics = "judgment" if r["dtype"] == "Judgment" else "scalar"
        derived.append(
            dict(
                **shared,
                entity=r["entity"],
                dtype={"Money": "decimal", "Judgment": "judgment", "Text": "text"}[
                    r["dtype"]
                ],
                period="Year",
                semantics=semantics,
                expr=expr,
                versions=[dict(**dates, semantics=semantics, expr=expr)],
            )
        )
    parameters.sort(key=lambda p: p["indexed_by"] is not None)
    expected = dict(
        module={k: data["module"][k] for k in ("summary", "source_verification")},
        units=_units(),
        relations=relations,
        parameters=parameters,
        derived=derived,
    )
    if not _equal(artifact["program"], expected):
        raise ValueError("compiled declaration union differs")
    catalog = [
        {
            "slot": n,
            "canonical_request_name": prefix + "input." + n,
            "request_names": [prefix + "input." + n],
        }
        for n in sorted(
            set(inputs)
            | {p["indexed_by"] for p in parameters if p["indexed_by"] is not None}
        )
    ]
    if not _equal(artifact["metadata"]["input_catalog"], catalog):
        raise ValueError("input catalog differs")
    return inputs, rules, expressions, {p["name"]: p for p in parameters}


def _render(n, prefix):
    k = n["kind"]
    if k == "input":
        return f'input("{n["name"]}")'
    if k == "derived":
        return prefix + n["name"]
    if k == "literal":
        return "true" if n["value"]["value"] is True else str(n["value"]["value"])
    if k == "parameter_lookup":
        return prefix + n["parameter"] + "[" + _render(n["index"], prefix) + "]"
    if k == "sum_related":
        return f'sum_related("{n["relation"]}", 1, 0, {_render(n["value"], prefix)})'
    if k == "if":
        return f"(if {_render(n['condition'], prefix)} then {_render(n['then_expr'], prefix)} else {_render(n['else_expr'], prefix)})"
    if k in ("sub", "mul", "div", "comparison"):
        op = {
            "sub": "-",
            "mul": "*",
            "div": "/",
            "comparison": {"eq": "==", "lt": "<", "gt": ">"}.get(n.get("op")),
        }[k]
        return f"({_render(n['left'], prefix)} {op} {_render(n['right'], prefix)})"
    values = [_render(x, prefix) for x in n["items"]]
    if k in ("min", "max"):
        return k + "(" + ", ".join(values) + ")"
    return "(" + {"add": " + ", "and": " and "}[k].join(values) + ")"


def _wire_value(value, dtype):
    if dtype == "Boolean":
        if (
            not isinstance(value, dict)
            or set(value) != {"kind", "value"}
            or value["kind"] != "bool"
            or type(value["value"]) is not bool
        ):
            raise ValueError("Boolean fact required")
        return value["value"]
    if dtype == "Date":
        if (
            not isinstance(value, dict)
            or set(value) != {"kind", "value"}
            or value["kind"] != "date"
            or type(value["value"]) is not str
        ):
            raise ValueError("Date fact required")
        result = date.fromisoformat(value["value"])
        if result.isoformat() != value["value"]:
            raise ValueError("canonical Date required")
        return result
    if dtype == "Text":
        if (
            not isinstance(value, dict)
            or set(value) != {"kind", "value"}
            or value["kind"] != "text"
            or type(value["value"]) is not str
        ):
            raise ValueError("Text fact required")
        return value["value"]
    result = _wire_number(value)
    if result is None:
        raise ValueError("numeric wire value required")
    return result


def _raw_value(value, dtype):
    if dtype == "Boolean" and type(value) is bool:
        return value
    if dtype == "Text" and type(value) is str:
        return value
    if dtype == "Date" and type(value) is str:
        result = date.fromisoformat(value)
        if result.isoformat() != value:
            raise ValueError("canonical raw Date required")
        return result
    if dtype == "Money":
        return _number(value)
    raise ValueError("raw fact type differs")


def _dataset(case, request, inputs, prefix, relation, query_kind, entity, interval):
    expected = {}
    tuples = []
    for key, value in case["input"].items():
        if key == relation:
            if not isinstance(value, list) or len(value) != 1:
                raise ValueError("selected-pair evidence required")
            for i, row in enumerate(value, 1):
                row_id = f"{entity}-{relation}-{i}"
                tuples.append(
                    {"name": relation, "tuple": [row_id, entity], "interval": interval}
                )
                for fact, raw in row.items():
                    name = fact.removeprefix(prefix + "input.")
                    if (
                        fact != prefix + "input." + name
                        or inputs[name]["entity"] != "Person"
                    ):
                        raise ValueError("row fact role differs")
                    expected[(row_id, name)] = _raw_value(raw, inputs[name]["dtype"])
        else:
            name = key.removeprefix(prefix + "input.")
            if key != prefix + "input." + name or inputs[name]["entity"] != query_kind:
                raise ValueError("query fact role differs")
            expected[(entity, name)] = _raw_value(value, inputs[name]["dtype"])
    dataset = request["dataset"]
    if set(dataset) != {"inputs", "relations"}:
        raise ValueError("unsupported dataset transport")
    if not _equal(dataset["relations"], tuples):
        raise ValueError("relation tuple/slot/frame differs")
    facts = {}
    for fact in dataset["inputs"]:
        name = fact["name"].removeprefix(prefix + "input.")
        key = (fact["entity_id"], name)
        if (
            fact["name"] != prefix + "input." + name
            or fact["entity"] != inputs[name]["entity"]
            or fact["interval"] != interval
            or key in facts
        ):
            raise ValueError("actual fact identity differs")
        facts[key] = _wire_value(fact["value"], inputs[name]["dtype"])
    if facts != expected or len(facts) != len(expected):
        raise ValueError("actual raw facts differ")
    return facts, [t["tuple"][0] for t in tuples]


def _trace_key(prefix, name, entity, query_entity):
    return (
        prefix
        + name
        + ("" if entity == query_entity else "@entity:" + entity.encode("utf-8").hex())
    )


def _branch_references(node):
    """Declared branch dependencies, including parameters, before selection."""
    if isinstance(node, dict):
        refs = set()
        if node.get("kind") == "derived":
            refs.add(node["name"])
        elif node.get("kind") == "parameter_lookup":
            refs.add(node["parameter"])
        for value in node.values():
            refs.update(_branch_references(value))
        return refs
    if isinstance(node, list):
        return set().union(*(_branch_references(value) for value in node))
    return set()


@dataclass(frozen=True)
class RetirementTransferObservation:
    content_sha256: str
    source_sha256: str
    cases: tuple[str, ...]
    witness_pairs: tuple[tuple[str, str, str], ...]
    provider_instances: tuple[tuple[str, str], ...]
    zero_transfer_cases: tuple[str, ...]
    executions: tuple[tuple, ...]


def _record(
    case, record, index, inputs, rules, expressions, parameters, prefix, owners, period
):
    request, response = (
        json.loads(record.request_json),
        json.loads(record.response_json),
    )
    if (
        not strict_explain_metadata(request, response)
        or len(request["queries"]) != 1
        or len(response["results"]) != 1
    ):
        raise ValueError("strict single query required")
    if set(request) - {"mode", "dataset", "queries", "relation_binding"}:
        raise ValueError("unsupported execution override")
    q, r = request["queries"][0], response["results"][0]
    if set(q) - {"entity_id", "period", "outputs", "assessment_date"}:
        raise ValueError("unsupported query override")
    entity = case.get("entity_id", f"case-{index}")
    if (
        not isinstance(entity, str)
        or not entity
        or q["entity_id"] != entity
        or q["period"] != period
        or case["period"] != period
        or q.get("assessment_date") is not None
    ):
        raise ValueError("query frame differs")
    if any(r.get(k) != q.get(k) for k in ("entity_id", "period", "assessment_date")):
        raise ValueError("response frame differs")
    outputs = case["output"]
    if (
        not outputs
        or set(outputs) != set(q["outputs"])
        or len(outputs) != len(q["outputs"])
        or set(outputs) != set(r["outputs"])
    ):
        raise ValueError("output inventory differs")
    names = [n.removeprefix(prefix) for n in outputs]
    if any(prefix + n not in outputs or n not in expressions for n in names):
        raise ValueError("unknown output")
    kinds = {rules[n]["entity"] for n in names}
    if len(kinds) != 1:
        raise ValueError("mixed query kinds")
    relation = prefix + "relation." + owners["selected_relation"]
    interval = {k: period[k] for k in ("start", "end")}
    facts, rows = _dataset(
        case, request, inputs, prefix, relation, kinds.pop(), entity, interval
    )
    values = {}
    pending = set()
    details = {}
    used = set()
    read_parameters = set()

    def derived(name, who):
        key = (name, who)
        if key in pending:
            raise ValueError("cyclic provider")
        if key in values:
            return values[key]
        pending.add(key)
        deps = []
        reads = []
        skipped = set()

        def evaluate(node):
            kind = node["kind"]
            if kind == "input":
                fact = (who, node["name"])
                used.add(fact)
                return facts[fact]
            if kind == "derived":
                deps.append((node["name"], who))
                return derived(node["name"], who)
            if kind == "parameter_lookup":
                parameter_name = node["parameter"]
                p = parameters[parameter_name]
                v = p["versions"][0]
                index = evaluate(node["index"])
                if type(index) is not int and (
                    not isinstance(index, Decimal) or index != index.to_integral_value()
                ):
                    raise ValueError("parameter index not integer")
                index = int(index)
                wire = v["values"][str(index)]
                read = dict(
                    parameter=prefix + parameter_name,
                    name=parameter_name,
                    id=prefix + parameter_name,
                    index=index,
                    value=wire,
                    **({"unit": p["unit"]} if p["unit"] else {}),
                    effective_from=v["effective_from"],
                    effective_to=v["effective_to"],
                    source=p["source"],
                )
                if read not in reads:
                    reads.append(read)
                read_parameters.add(parameter_name)
                return (
                    date.fromisoformat(wire["value"])
                    if wire["kind"] == "date"
                    else _number(wire["value"])
                )
            if kind == "literal":
                v = node["value"]["value"]
                return v if type(v) is bool else _number(v)
            if kind == "sum_related":
                if who != entity or rules[name]["entity"] != "Person":
                    raise ValueError("invalid relation consumer")
                total = Decimal(0)
                for row in rows:
                    child = node["value"]["name"]
                    deps.append((child, row))
                    total += derived(child, row)
                return total
            if kind == "if":
                selected = evaluate(node["condition"])
                skipped.update(
                    (n, "branch_not_selected")
                    for n in _branch_references(
                        node["else_expr"] if selected else node["then_expr"]
                    )
                )
                return evaluate(node["then_expr"] if selected else node["else_expr"])
            if kind == "comparison":
                a, b = evaluate(node["left"]), evaluate(node["right"])
                return {"eq": lambda: a == b, "lt": lambda: a < b, "gt": lambda: a > b}[
                    node["op"]
                ]()
            if kind in ("sub", "mul", "div"):
                a, b = evaluate(node["left"]), evaluate(node["right"])
                if kind == "div" and b == 0:
                    raise ValueError("positive transfer has zero denominator")
                return {
                    "sub": lambda: a - b,
                    "mul": lambda: a * b,
                    "div": lambda: a / b,
                }[kind]()
            if kind == "and":
                for i, item in enumerate(node["items"]):
                    if not evaluate(item):
                        skipped.update(
                            (n, "short_circuit")
                            for rest in node["items"][i + 1 :]
                            for n in _branch_references(rest)
                        )
                        return False
                return True
            vals = [evaluate(x) for x in node["items"]]
            return {"add": sum, "min": min, "max": max}[kind](vals)

        value = evaluate(expressions[name])
        pending.remove(key)
        values[key] = value
        details[key] = (deps, reads, skipped)
        return value

    for n in names:
        derived(n, entity)
    # Evidence domain only, not a global API input constraint or formula floor.
    if any(
        value < 0 for (name, _), value in values.items() if name == owners["net_income"]
    ):
        raise ValueError("negative reached net income outside source-valid evidence")
    expected_keys = {_trace_key(prefix, n, e, entity) for n, e in values}
    if set(r["trace"]) != expected_keys:
        raise ValueError("provider trace closure differs")
    for (name, who), value in values.items():
        rule = rules[name]
        trace = r["trace"][_trace_key(prefix, name, who, entity)]
        deps, reads, skipped = details[(name, who)]
        expected_deps = {_trace_key(prefix, n, e, entity) for n, e in deps}
        if (
            trace.get("entity_id") != who
            or trace.get("source") != rule["source"]
            or trace.get("executed_expression") != _render(expressions[name], prefix)
            or set(trace.get("dependencies", [])) != expected_deps
            or len(trace.get("dependencies", [])) != len(expected_deps)
            or not _equal(trace.get("parameter_reads", []), reads)
            or not _equal(
                sorted(
                    trace.get("not_evaluated_dependencies", []),
                    key=lambda d: (d["dependency"], d["reason"]),
                ),
                [
                    {
                        "dependency": prefix + n
                        if n in parameters
                        else _trace_key(prefix, n, who, entity),
                        "reason": reason,
                    }
                    for n, reason in sorted(skipped)
                ],
            )
        ):
            raise ValueError("trace lineage/AST/parameter differs")
        for item in [trace] + (
            [r["outputs"][prefix + name]] if name in names and who == entity else []
        ):
            if any(
                item.get(k) != v
                for k, v in dict(
                    id=prefix + name,
                    name=name,
                    kind="judgment" if rule["dtype"] == "Judgment" else "scalar",
                    unit=rule.get("unit"),
                ).items()
            ):
                raise ValueError("value owner differs")
            if rule["dtype"] == "Judgment":
                if type(value) is not bool or item.get("outcome") != (
                    "holds" if value else "not_holds"
                ):
                    raise ValueError("judgment differs")
            else:
                if (
                    item.get("dtype")
                    != ("text" if rule["dtype"] == "Text" else "decimal")
                    or _wire_value(item["value"], rule["dtype"]) != value
                ):
                    raise ValueError("typed output differs")
        if name in names and who == entity:
            expected = outputs[prefix + name]
            if rule["dtype"] == "Judgment":
                if expected != ("holds" if value else "not_holds"):
                    raise ValueError("expected judgment differs")
            elif _raw_value(expected, rule["dtype"]) != value:
                raise ValueError("expected scalar differs")
    return (
        entity,
        facts,
        {n: values[n, entity] for n in names},
        set(values),
        read_parameters,
        values,
    )


def _required_pairs(records, owners):
    """Independent one-fact effects; no unrelated duty may substitute."""
    requirements = {
        "permission_birth": ("permission", "transferor_birth_date"),
        "permission_spouse": ("permission", "has_selected_spouse_at_reference_date"),
        "permission_date": ("permission", "spouse_reference_date"),
        "permission_decision": (
            "permission",
            "jointly_decided_retirement_income_transfer",
        ),
        "withholding_inclusion": (
            "entered_withholding",
            "compensation_included_in_line20",
        ),
        "withholding_duty": (
            "withholding_duty",
            "schedule_q_requested_retirement_income_transfer_line_22",
        ),
        "enclosure": ("enclosure", "is_completing_retirement_transfer_schedule"),
        "identification": (
            "identification",
            "is_completing_retirement_transfer_schedule",
        ),
        "compensation_guide": (
            "compensation_guide",
            "received_retirement_compensation_amounts",
        ),
        "copy_guide": ("copy_guide", "wants_retirement_transfer_schedule_copy"),
    }
    pairs = []
    for label, (role, fact) in requirements.items():
        matches = []
        owner = owners[role]
        for a in records:
            for b in records:
                if (
                    a[0] >= b[0]
                    or a[1] != b[1]
                    or owner not in a[3]
                    or owner not in b[3]
                    or a[2].keys() != b[2].keys()
                ):
                    continue
                changed = {k for k in a[2] if a[2][k] != b[2][k]}
                if changed == {(a[1], fact)} and a[3][owner] != b[3][owner]:
                    matches.append((a[0], b[0]))
        if not matches:
            raise ValueError("missing required witness: " + label)
        # Retain all complete pairs, not a last-wins claimant bucket.
        pairs.extend((label, a, b) for a, b in sorted(set(matches)))
    return pairs


def _provider_pairs(records, owners):
    pairs = []
    requirements = [
        (role, "schedule_q_requested_retirement_income_transfer_line_22")
        for role in (
            "recipient_income",
            "recipient_withholding",
            "recipient_income_duty",
            "recipient_withholding_duty",
        )
    ]
    requirements += [
        (role, fact)
        for role in ("recipient_income", "recipient_withholding")
        for fact in (
            "transferor_birth_date",
            "has_selected_spouse_at_reference_date",
            "spouse_reference_date",
            "jointly_decided_retirement_income_transfer",
        )
    ]
    for role, fact in requirements:
        owner = owners[role]
        found = []
        for a in records:
            for b in records:
                if (
                    a[0] >= b[0]
                    or a[1] != b[1]
                    or owner not in a[3]
                    or owner not in b[3]
                ):
                    continue
                providers_a = {e for n, e in a[4] if e != a[1]}
                providers_b = {e for n, e in b[4] if e != b[1]}
                if (
                    len(providers_a) != 1
                    or providers_a != providers_b
                    or a[2].keys() != b[2].keys()
                ):
                    continue
                provider = next(iter(providers_a))
                changed = {k for k in a[2] if a[2][k] != b[2][k]}
                if changed == {(provider, fact)} and a[3][owner] != b[3][owner]:
                    found.append((a[0], b[0]))
        if not found:
            raise ValueError("missing provider witness: " + role + ":" + fact)
        pairs.extend((role + ":" + fact, a, b) for a, b in sorted(set(found)))
    return pairs


def _normalization_and_zero(records, owners):
    groups = {}
    zeros = []
    zero_roles = set()
    request_name = "schedule_q_requested_retirement_income_transfer_line_22"
    for key, entity, facts, outputs, instances, _, values in records:
        reqkey = (entity, request_name)
        netkey = (owners["net_income"], entity)
        if (
            reqkey in facts
            and owners["authorized_transfer"] in outputs
            and netkey in values
        ):
            requested = facts[reqkey]
            net = values[netkey]
            source_facts = tuple(
                sorted((k, str(v)) for k, v in facts.items() if k != reqkey)
            )
            group = groups.setdefault((entity, source_facts), set())
            if requested < 0 and outputs.get(owners["normalized_request"]) == 0:
                group.add("negative")
            elif requested == 0:
                group.add("zero")
            elif requested < net * Decimal("0.5"):
                group.add("below")
            elif requested == net * Decimal("0.5"):
                group.add("equal")
            else:
                group.add("above")
        # Both permission states locally, and a genuine recipient-provider zero,
        # establish no-transfer semantics without defining zero division.
        for name, who in instances:
            if name != owners["net_income"] or values[(name, who)] != 0:
                continue
            permission = values.get((owners["permission"], who))
            if (
                type(permission) is bool
                and (who, "quebec_rl2_box_j_from_line20") in facts
                and facts[(who, "quebec_rl2_box_j_from_line20")] > 0
                and values.get((owners["authorized_transfer"], who)) == 0
                and values.get((owners["transferred_withholding"], who)) == 0
            ):
                zero_roles.add(("local" if who == entity else "recipient", permission))
                zeros.append(key)
    if not any(
        {"negative", "zero", "below", "equal", "above"} <= g for g in groups.values()
    ):
        raise ValueError("missing chosen-amount boundary observations")
    if not {("local", True), ("local", False), ("recipient", True)} <= zero_roles:
        raise ValueError("missing zero-net no-transfer observation")
    return tuple(sorted(set(zeros)))


def observe_retirement_transfer(binding, evidence, content, cases):
    """Observe the whole graph, without capture or source-coverage authority."""
    try:
        if (
            evidence.engine_pin != _PROFILE_PIN
            or not evidence.bound(content, ())
            or bind_retirement_transfer_form(
                content, binding.source.source, binding.source.citation
            )
            != binding
        ):
            return None
        path = PurePosixPath(evidence.root_relative)
        if (
            path.is_absolute()
            or path.suffix != ".yaml"
            or len(path.parts) < 3
            or path.parts[1] != "policies"
            or ".." in path.parts
            or str(path) != evidence.root_relative
        ):
            return None
        citation = str(
            PurePosixPath(path.parts[0], "policy", *path.parts[2:]).with_suffix("")
        )
        if citation != binding.source.citation:
            return None
        companion = str(path.with_suffix(".test.yaml"))
        if {f.relative for f in evidence.files} != {str(path), companion} or len(
            evidence.files
        ) != 2:
            return None
        from .validator_pipeline import _safe_load_unique_keys

        raw_cases = _safe_load_unique_keys(
            next(f.raw for f in evidence.files if f.relative == companion).decode()
        )
        if not _equal(raw_cases, cases):
            return None
        prefix = (
            path.parts[0]
            + ":"
            + str(PurePosixPath(*path.parts[1:]).with_suffix(""))
            + "#"
        )
        inputs, rules, expressions, parameters = _compiled(
            binding, content, json.loads(evidence.artifact_bytes), prefix
        )
        keys = [case_key(c) for c in cases]
        if (
            not keys
            or None in keys
            or len(set(keys)) != len(keys)
            or len(evidence.records) != len(cases)
        ):
            return None
        owners = dict(binding.rule_roles)
        records = []
        all_instances, all_parameters, all_outputs = set(), set(), set()
        year = binding.source.year
        period = dict(
            period_kind="tax_year", start=f"{year}-01-01", end=f"{year}-12-31"
        )
        for index, (case, key) in enumerate(zip(cases, keys, strict=True), 1):
            matches = [
                r
                for r in evidence.records
                if r.case_sha256 == key and r.case_index == index
            ]
            if len(matches) != 1:
                return None
            result = _record(
                case,
                matches[0],
                index,
                inputs,
                rules,
                expressions,
                parameters,
                prefix,
                owners,
                period,
            )
            records.append((key, *result))
            all_instances.update(result[3])
            all_parameters.update(result[4])
            all_outputs.update(result[2])
        if all_outputs != set(expressions) or all_parameters != set(parameters):
            return None
        pairs = _required_pairs(records, owners) + _provider_pairs(records, owners)
        zeros = _normalization_and_zero(records, owners)
        return RetirementTransferObservation(
            binding.content_sha256,
            binding.source.source_sha256,
            tuple(keys),
            tuple(pairs),
            tuple(sorted(all_instances)),
            zeros,
            tuple(
                (
                    key,
                    entity,
                    tuple(sorted(facts.items())),
                    tuple(sorted(outputs.items())),
                    tuple(sorted(instances)),
                    tuple(sorted(values.items())),
                )
                for key, entity, facts, outputs, instances, _, values in records
            ),
        )
    except (
        AttributeError,
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        InvalidOperation,
        RecursionError,
        yaml.YAMLError,
    ):
        return None


def _capture_source_binding(content, sources, root_relative):
    """Called with ordinary resolver-owned source bodies, never YAML grants."""
    try:
        payload = yaml.safe_load(content)
        citation = payload["module"]["source_verification"]["corpus_citation_path"]
        source = sources.get(citation)
        if not isinstance(source, str):
            return None
        path = PurePosixPath(root_relative)
        if (
            path.is_absolute()
            or path.suffix != ".yaml"
            or len(path.parts) < 3
            or path.parts[1] != "policies"
            or ".." in path.parts
            or str(path) != root_relative
            or citation
            != str(
                PurePosixPath(path.parts[0], "policy", *path.parts[2:]).with_suffix("")
            )
        ):
            return None
        return bind_retirement_transfer_form(content, source, citation)
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


@dataclass(frozen=True)
class _RetirementTransferCaptureContext:
    """Private compiled context; no source coverage or independent admission."""

    binding: object
    content: str
    root_relative: str
    artifact_sha256: str
    engine_pin: str
    companion_bytes: bytes
    case_keys: tuple[str, ...]

    def accepts(self, artifact_bytes, case, request, response, case_index):
        try:
            if (
                type(case_index) is not int
                or case_index <= 0
                or case_index > len(self.case_keys)
                or case_key(case) != self.case_keys[case_index - 1]
                or hashlib.sha256(artifact_bytes).hexdigest() != self.artifact_sha256
                or _bind_capture_context(
                    self.binding,
                    self.content,
                    self.root_relative,
                    self.engine_pin,
                    artifact_bytes,
                    self.companion_bytes,
                )
                != self
                or case_key(case) is None
            ):
                return False
            prefix = (
                self.root_relative.split("/", 1)[0]
                + ":"
                + self.root_relative.split("/", 1)[1].removesuffix(".yaml")
                + "#"
            )
            inputs, rules, expressions, parameters = _compiled(
                self.binding, self.content, json.loads(artifact_bytes), prefix
            )
            year = self.binding.source.year
            from .same_case_execution import CaseExecution

            record = CaseExecution(
                case_key(case), json.dumps(request), json.dumps(response), case_index
            )
            _record(
                case,
                record,
                case_index,
                inputs,
                rules,
                expressions,
                parameters,
                prefix,
                dict(self.binding.rule_roles),
                dict(
                    period_kind="tax_year", start=f"{year}-01-01", end=f"{year}-12-31"
                ),
            )
            return True
        except (
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
            IndexError,
            InvalidOperation,
            RecursionError,
            yaml.YAMLError,
        ):
            return False


def _bind_capture_context(
    binding, content, root_relative, engine_pin, artifact_bytes, companion_bytes
):
    try:
        if engine_pin != _PROFILE_PIN or binding is None:
            return None
        rebound = _capture_source_binding(
            content, {binding.source.citation: binding.source.source}, root_relative
        )
        if rebound != binding:
            return None
        # Use the same duplicate-key rejection as ordinary companion loading.
        # Ordinary companions may share acyclic YAML aliases. Policy-source
        # alias restrictions do not apply here; cyclic values cannot case_key.
        from .validator_pipeline import _safe_load_unique_keys

        cases = _safe_load_unique_keys(companion_bytes.decode())
        if (
            not isinstance(cases, list)
            or not cases
            or not all(isinstance(c, dict) for c in cases)
        ):
            return None
        keys = tuple(case_key(c) for c in cases)
        if None in keys or len(set(keys)) != len(keys):
            return None
        artifact = json.loads(artifact_bytes)
        if (
            type(artifact.get("artifact_format_version")) is not int
            or artifact["artifact_format_version"] != 2
        ):
            return None
        prefix = (
            root_relative.split("/", 1)[0]
            + ":"
            + root_relative.split("/", 1)[1].removesuffix(".yaml")
            + "#"
        )
        _compiled(binding, content, artifact, prefix)
        return _RetirementTransferCaptureContext(
            binding,
            content,
            root_relative,
            hashlib.sha256(artifact_bytes).hexdigest(),
            engine_pin,
            companion_bytes,
            keys,
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        InvalidOperation,
        RecursionError,
        yaml.YAMLError,
    ):
        return None
