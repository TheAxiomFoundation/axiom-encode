"""Exact compiled and actual-case observation for the closed student-loan graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

import yaml

from .bound_form_operands import _wire_number
from .same_case_execution import case_key, strict_explain_metadata
from .student_loan_schedule_binding import bind_student_loan_schedule

# Immutable compiler source: 89571cc2 src/formula.rs::lower_module,
# SHA256 7d7a8c89fb41e984094f809243e3d426d44aba0cf62dd1e614b3c8403696c305.
# This observer refuses unknown pins; it does not change global engine policy.
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


def _expected_units():
    return [
        *(
            {"name": name, "kind": "currency", "minor_units": digits}
            for name, digits in _CURRENCIES
        ),
        {"name": "count", "kind": "count"},
        {"name": "person", "kind": "count"},
        {"name": "ratio", "kind": "ratio"},
        {"name": "status", "kind": "custom", "label": "judgment"},
        {"name": "CAD", "kind": "currency", "minor_units": 2},
    ]


def _equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _decimal(value):
    if type(value) not in (str, int, float):
        raise ValueError("Numeric fact required")
    parsed = Decimal(str(value))
    if not parsed.is_finite():
        raise ValueError("Finite fact required")
    return parsed


def _expressions(binding):
    facts, owners = dict(binding.input_roles), dict(binding.rule_roles)

    def inp(role):
        return {"kind": "input", "name": facts[role]}

    def ref(role):
        return {"kind": "derived", "name": owners[role]}

    def boolean(role):
        return {
            "kind": "comparison",
            "left": inp(role),
            "op": "eq",
            "right": {"kind": "literal", "value": {"kind": "bool", "value": True}},
        }

    return {
        "line46": {
            "kind": "if",
            "condition": boolean("calculated"),
            "then_expr": inp("prior_unused"),
            "else_expr": {
                "kind": "sub",
                "left": inp("history_paid"),
                "right": inp("history_used"),
            },
        },
        "line52": {"kind": "add", "items": [ref("line46"), inp("current_paid")]},
        "line385": {
            "kind": "min",
            "items": [
                ref("line52"),
                {
                    "kind": "max",
                    "items": [
                        {"kind": "literal", "value": {"kind": "decimal", "value": "0"}},
                        inp("requested"),
                    ],
                },
            ],
        },
        "line62": {"kind": "sub", "left": ref("line52"), "right": ref("line385")},
        "completion": {
            "kind": "or",
            "items": [
                {
                    "kind": "comparison",
                    "left": inp("current_paid"),
                    "op": "gt",
                    "right": {
                        "kind": "literal",
                        "value": {"kind": "integer", "value": 0},
                    },
                },
                {"kind": "and", "items": [boolean("calculated"), boolean("decision")]},
            ],
        },
        "enclosure": boolean("form_use"),
        "guide": boolean("copy_wish"),
    }


def _references(node):
    result = set()
    if isinstance(node, dict):
        if node.get("kind") == "derived":
            result.add(node["name"])
        for value in node.values():
            result.update(_references(value))
    elif isinstance(node, list):
        for value in node:
            result.update(_references(value))
    return result


def _render(node, prefix):
    kind = node["kind"]
    if kind == "input":
        return f'input("{node["name"]}")'
    if kind == "derived":
        return prefix + node["name"]
    if kind == "literal":
        v = node["value"]["value"]
        return "true" if v is True else str(v)
    if kind == "comparison":
        return f"({_render(node['left'], prefix)} {'==' if node['op'] == 'eq' else '>'} {_render(node['right'], prefix)})"
    if kind == "if":
        return f"(if {_render(node['condition'], prefix)} then {_render(node['then_expr'], prefix)} else {_render(node['else_expr'], prefix)})"
    if kind == "sub":
        return f"({_render(node['left'], prefix)} - {_render(node['right'], prefix)})"
    items = [_render(n, prefix) for n in node["items"]]
    if kind in {"min", "max"}:
        return kind + "(" + ", ".join(items) + ")"
    return "(" + {"add": " + ", "and": " and ", "or": " or "}[kind].join(items) + ")"


def _evaluate(node, expressions, facts, used):
    kind = node["kind"]
    if kind == "input":
        used.add(node["name"])
        return facts[node["name"]]
    if kind == "derived":
        return _evaluate(expressions[node["name"]], expressions, facts, used)
    if kind == "literal":
        value = node["value"]["value"]
        return value if type(value) is bool else _decimal(value)
    if kind == "if":
        condition = _evaluate(node["condition"], expressions, facts, used)
        return _evaluate(
            node["then_expr"] if condition else node["else_expr"],
            expressions,
            facts,
            used,
        )
    if kind == "comparison":
        a, b = (_evaluate(node[k], expressions, facts, used) for k in ("left", "right"))
        return a == b if node["op"] == "eq" else a > b
    if kind == "sub":
        return _evaluate(node["left"], expressions, facts, used) - _evaluate(
            node["right"], expressions, facts, used
        )
    values = (_evaluate(n, expressions, facts, used) for n in node["items"])
    if kind == "and":
        return all(values)
    if kind == "or":
        return any(values)
    return {"add": sum, "min": min, "max": max}[kind](values)


@dataclass(frozen=True)
class StudentLoanScheduleObservation:
    content_sha256: str
    source_sha256: str
    cases: tuple[str, ...]
    duty_pairs: tuple[tuple[str, str, tuple[str, str]], ...]
    exclusive_provider_cases: tuple[str, str]
    equal_provider_pair: tuple[str, str]


def observe_student_loan_schedule(binding, evidence, content, cases):
    """Rebind the complete graph and all actual records; mint no generic waiver."""
    try:
        if (
            evidence.engine_pin != _PROFILE_PIN
            or not evidence.bound(content, ())
            or bind_student_loan_schedule(binding.source, content) != binding
        ):
            return None
        path = PurePosixPath(evidence.root_relative)
        if (
            path.suffix != ".yaml"
            or len(path.parts) < 3
            or path.parts[1] != "policies"
            or ".." in path.parts
        ):
            return None
        prefix = (
            path.parts[0]
            + ":"
            + str(PurePosixPath(*path.parts[1:]).with_suffix(""))
            + "#"
        )
        data = yaml.safe_load(content)
        inputs = {i["name"]: i for i in data["inputs"]}
        rules = {r["name"]: r for r in data["rules"]}
        owners, facts_by_role = dict(binding.rule_roles), dict(binding.input_roles)
        expressions = {owners[k]: v for k, v in _expressions(binding).items()}
        artifact = json.loads(evidence.artifact_bytes)
        program = artifact["program"]
        if set(program) != {
            "module",
            "units",
            "relations",
            "parameters",
            "derived",
        } or not _equal(
            program["module"],
            {k: data["module"][k] for k in ("summary", "source_verification")},
        ):
            return None
        catalog = [
            {
                "slot": n,
                "canonical_request_name": prefix + "input." + n,
                "request_names": [prefix + "input." + n],
            }
            for n in sorted(inputs)
        ]
        if not _equal(artifact["metadata"]["input_catalog"], catalog):
            return None
        declarations = {d["name"]: d for d in program["derived"]}
        if (
            program["relations"] != []
            or program["parameters"] != []
            or len(declarations) != 7
            or len(program["derived"]) != 7
            or set(declarations) != set(rules)
        ):
            return None
        if not _equal(program["units"], _expected_units()):
            return None
        period = {
            "period_kind": "tax_year",
            "start": f"{binding.source.year}-01-01",
            "end": f"{binding.source.year}-12-31",
        }
        interval = {k: period[k] for k in ("start", "end")}
        for name, rule in rules.items():
            semantics = "judgment" if rule["dtype"] == "Judgment" else "scalar"
            expr = expressions[name]
            expected = dict(
                id=prefix + name,
                name=name,
                entity="Person",
                dtype="judgment" if semantics == "judgment" else "decimal",
                unit=rule.get("unit"),
                period="Year",
                source=rule["source"],
                source_url=None,
                corpus_citation_path=binding.source.citation,
                semantics=semantics,
                expr=expr,
                versions=[
                    dict(
                        effective_from=period["start"],
                        effective_to=period["end"],
                        semantics=semantics,
                        expr=expr,
                    )
                ],
            )
            if not _equal(declarations[name], expected):
                return None
        keys = [case_key(c) for c in cases]
        if (
            None in keys
            or len(set(keys)) != len(cases)
            or len(evidence.records) != len(cases)
        ):
            return None
        observed = set()
        valid = []
        exclusive = {True: [], False: []}
        domains = set()
        for index, case in enumerate(cases, 1):
            key = case_key(case)
            records = [r for r in evidence.records if r.case_sha256 == key]
            if (
                len(records) != 1
                or records[0].case_index != index
                or case["period"] != period
            ):
                return None
            request, response = (
                json.loads(records[0].request_json),
                json.loads(records[0].response_json),
            )
            if (
                not strict_explain_metadata(request, response)
                or len(request["queries"]) != 1
                or len(response["results"]) != 1
            ):
                return None
            query, result = request["queries"][0], response["results"][0]
            entity_id = case.get("entity_id", f"case-{index}")
            if (
                not isinstance(entity_id, str)
                or not entity_id
                or query.get("entity_id") != entity_id
                or query.get("period") != period
                or query.get("assessment_date") is not None
                or any(
                    result.get(k) != query.get(k)
                    for k in ("entity_id", "period", "assessment_date")
                )
            ):
                return None
            outputs = case["output"]
            if (
                not outputs
                or set(outputs) != set(query["outputs"])
                or len(outputs) != len(query["outputs"])
                or set(outputs) != set(result["outputs"])
            ):
                return None
            names = {n.removeprefix(prefix) for n in outputs}
            if (
                any(n != prefix + n.removeprefix(prefix) for n in outputs)
                or not names <= rules.keys()
            ):
                return None
            raw_facts = case["input"]
            if any(not n.startswith(prefix + "input.") for n in raw_facts):
                return None
            local = {n[len(prefix + "input.") :]: v for n, v in raw_facts.items()}
            if not local.keys() <= inputs.keys():
                return None
            wire = request["dataset"]["inputs"]
            if (
                request["dataset"].get("relations") != []
                or len(wire) != len(local)
                or {f["name"] for f in wire} != set(raw_facts)
            ):
                return None
            facts = {}
            for fact in wire:
                name = fact["name"][len(prefix + "input.") :]
                if (
                    fact.get("entity") != "Person"
                    or fact.get("entity_id") != entity_id
                    or fact.get("interval") != interval
                ):
                    return None
                value = fact["value"]
                if inputs[name]["dtype"] == "Boolean":
                    if (
                        set(value) != {"kind", "value"}
                        or value["kind"] != "bool"
                        or type(value["value"]) is not bool
                        or type(local[name]) is not bool
                        or value["value"] is not local[name]
                    ):
                        return None
                    facts[name] = value["value"]
                else:
                    number = _wire_number(value)
                    if number is None or number != _decimal(local[name]):
                        return None
                    facts[name] = number
            if (
                facts_by_role["current_paid"] in facts
                and facts[facts_by_role["current_paid"]] < 0
            ):
                return None
            closure = set(names)
            pending = list(names)
            while pending:
                for dependency in _references(expressions[pending.pop()]):
                    if dependency not in closure:
                        closure.add(dependency)
                        pending.append(dependency)
            if set(result["trace"]) != {prefix + n for n in closure}:
                return None
            values = {}
            for name in closure:
                used = set()
                value = _evaluate(expressions[name], expressions, facts, used)
                values[name] = value
                # The reviewed claim normalization only applies within a
                # nonnegative available-interest domain, even on partial queries.
                if name == owners["line52"] and value < 0:
                    return None
                trace = result["trace"][prefix + name]
                judgment = rules[name]["dtype"] == "Judgment"
                for item in [trace] + (
                    [result["outputs"][prefix + name]] if name in names else []
                ):
                    if any(
                        item.get(k) != v
                        for k, v in dict(
                            id=prefix + name,
                            name=name,
                            kind="judgment" if judgment else "scalar",
                            unit=rules[name].get("unit"),
                        ).items()
                    ):
                        return None
                    if judgment:
                        if (
                            type(value) is not bool
                            or item.get("outcome")
                            != ("holds" if value else "not_holds")
                            or (
                                name in names
                                and outputs[prefix + name] != item["outcome"]
                            )
                        ):
                            return None
                    elif (
                        item.get("dtype") != "decimal"
                        or _wire_number(item.get("value")) != value
                        or (name in names and _decimal(outputs[prefix + name]) != value)
                    ):
                        return None
                deps = {prefix + n for n in _references(expressions[name])}
                if (
                    trace.get("entity_id") != entity_id
                    or trace.get("source") != rules[name]["source"]
                    or trace.get("executed_expression")
                    != _render(expressions[name], prefix)
                    or set(trace.get("dependencies", [])) != deps
                    or len(trace.get("dependencies", [])) != len(deps)
                ):
                    return None
            if owners["line46"] in names:
                selected = facts[facts_by_role["calculated"]]
                unselected = (
                    {facts_by_role["history_paid"], facts_by_role["history_used"]}
                    if selected
                    else {facts_by_role["prior_unused"]}
                )
                if not unselected & facts.keys():
                    exclusive[selected].append(key)
            if {owners["line385"], owners["line52"]} <= names:
                request_amount, available = (
                    facts[facts_by_role["requested"]],
                    values[owners["line52"]],
                )
                domains.add(
                    "negative"
                    if request_amount < 0
                    else "above"
                    if request_amount > available
                    else "within"
                )
            observed.update(names)
            valid.append((key, case.get("entity_id"), local, names, values))
        if (
            observed != set(rules)
            or not all(exclusive.values())
            or domains != {"negative", "above", "within"}
        ):
            return None
        pairs = []
        required = {
            "completion": ("current_paid", "calculated", "decision"),
            "enclosure": ("form_use",),
            "guide": ("copy_wish",),
        }
        for role, inputs_required in required.items():
            owner = owners[role]
            for fact_role in inputs_required:
                found = []
                for a in valid:
                    for b in valid:
                        if (
                            a[0] >= b[0]
                            or a[1] != b[1]
                            or owner not in a[3] & b[3]
                            or a[2].keys() != b[2].keys()
                        ):
                            continue
                        changed = {k for k in a[2] if not _equal(a[2][k], b[2][k])}
                        if (
                            changed == {facts_by_role[fact_role]}
                            and a[4][owner] != b[4][owner]
                        ):
                            found.append((a[0], b[0]))
                if not found:
                    return None
                pairs.append((role, fact_role, min(found)))
        equal_pairs = []
        money = {owners[k] for k in ("line46", "line52", "line385", "line62")}
        for a in valid:
            for b in valid:
                if (
                    a[0] >= b[0]
                    or a[1] != b[1]
                    or not money <= a[3] & b[3]
                    or a[2].keys() != b[2].keys()
                ):
                    continue
                changed = {k for k in a[2] if not _equal(a[2][k], b[2][k])}
                if changed == {facts_by_role["calculated"]} and all(
                    a[4][n] == b[4][n] for n in money
                ):
                    equal_pairs.append((a[0], b[0]))
        if not equal_pairs:
            return None
        return StudentLoanScheduleObservation(
            binding.content_sha256,
            binding.source.source_sha256,
            tuple(sorted(keys)),
            tuple(sorted(pairs)),
            (min(exclusive[True]), min(exclusive[False])),
            min(equal_pairs),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
        InvalidOperation,
    ):
        return None
