"""Strict complete-graph Schedule observation; no caller-supplied certificate."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .bound_form_operands import _compiled_expression, _number, _tree, _wire_number
from .bound_schedule_operands import ScheduleBinding, bind_schedule
from .imported_export_evidence import _named, _payload
from .same_case_execution import case_key, strict_explain_metadata


def _equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _compiled(node, rules, inputs, judgment=False):
    if isinstance(node, ast.Name):
        if not judgment and node.id in {"period_start", "period_end"}:
            return {"kind": node.id}
        if judgment and node.id in inputs:
            return {
                "kind": "comparison",
                "left": {"kind": "input", "name": node.id},
                "op": "eq",
                "right": {"kind": "literal", "value": {"kind": "bool", "value": True}},
            }
        return _compiled_expression(node, rules, set(inputs))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        result = _compiled(node.values[0], rules, inputs, True)
        for v in node.values[1:]:
            result = {
                "kind": "and",
                "items": [result, _compiled(v, rules, inputs, True)],
            }
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return {"kind": "not", "item": _compiled(node.operand, rules, inputs, True)}
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    ):
        return {
            "kind": "comparison",
            "left": _compiled(node.left, rules, inputs),
            "op": "eq",
            "right": _compiled(node.comparators[0], rules, inputs),
        }
    return _compiled_expression(node, rules, set(inputs))


def _render(node, rules, inputs, prefix, judgment=False):
    if isinstance(node, ast.Name):
        if node.id in inputs:
            text = 'input("' + node.id + '")'
            return "(" + text + " == true)" if judgment else text
        if node.id in {"period_start", "period_end"}:
            return node.id
        return (
            prefix + node.id + ("[0]" if rules[node.id]["kind"] == "parameter" else "")
        )
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        value = _render(node.values[0], rules, inputs, prefix, True)
        for child in node.values[1:]:
            value = (
                "("
                + value
                + " and "
                + _render(child, rules, inputs, prefix, True)
                + ")"
            )
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return "not (" + _render(node.operand, rules, inputs, prefix, True) + ")"
    if isinstance(node, ast.Compare):
        return (
            "("
            + _render(node.left, rules, inputs, prefix)
            + " == "
            + _render(node.comparators[0], rules, inputs, prefix)
            + ")"
        )
    if isinstance(node, ast.BinOp):
        return (
            "("
            + _render(node.left, rules, inputs, prefix)
            + (" - " if isinstance(node.op, ast.Sub) else " * ")
            + _render(node.right, rules, inputs, prefix)
            + ")"
        )
    if isinstance(node, ast.Call) and node.func.id == "min":
        return (
            "min("
            + ", ".join(_render(n, rules, inputs, prefix) for n in node.args)
            + ")"
        )
    raise ValueError("Unsupported execution expression")


def _evaluate(node, rules, facts, period, active=()):
    if isinstance(node, ast.Name):
        if node.id in facts:
            return facts[node.id]
        if node.id in {"period_start", "period_end"}:
            return date.fromisoformat(
                period["start" if node.id == "period_start" else "end"]
            )
        if node.id in active:
            raise ValueError("Cyclic graph")
        return _evaluate(
            _tree(rules[node.id]["versions"][0]["formula"]),
            rules,
            facts,
            period,
            active + (node.id,),
        )
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        values = [_evaluate(n, rules, facts, period, active) for n in node.values]
        if any(type(v) is not bool for v in values):
            raise ValueError("Boolean facts required")
        return all(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _evaluate(node.operand, rules, facts, period, active)
        if type(value) is not bool:
            raise ValueError("Boolean fact required")
        return not value
    if isinstance(node, ast.Compare):
        return _evaluate(node.left, rules, facts, period, active) == _evaluate(
            node.comparators[0], rules, facts, period, active
        )
    if isinstance(node, ast.BinOp):
        a, b = (
            _evaluate(n, rules, facts, period, active) for n in (node.left, node.right)
        )
        return a - b if isinstance(node.op, ast.Sub) else a * b
    if isinstance(node, ast.Call) and node.func.id == "min":
        return min(_evaluate(n, rules, facts, period, active) for n in node.args)
    raise ValueError("Unsupported value")


@dataclass(frozen=True)
class ScheduleObservation:
    content_sha256: str
    source_sha256: str
    row_cases: tuple[str, ...]
    pairs: tuple[tuple[str, str, tuple[str, str]], ...]
    medical_claimants: tuple[str, ...]


def observe_schedule(
    binding: ScheduleBinding, evidence, content, cases
) -> ScheduleObservation | None:
    """Rebind source, every compiled declaration and exact typed actual cases."""
    try:
        if (
            binding.namespace is None
            or not evidence.bound(content, ())
            or bind_schedule(
                binding.form,
                content,
                {binding.form.citation: binding.form.source},
                root_relative=evidence.root_relative,
            )
            != binding
        ):
            return None
        payload = _payload(content)
        rules = _named(payload, "rules")
        inputs = _named(payload, "inputs")
        prefix = binding.namespace + "#"
        program = json.loads(evidence.artifact_bytes)["program"]
        derived = {n: r for n, r in rules.items() if r["kind"] == "derived"}
        parameters = {n: r for n, r in rules.items() if r["kind"] == "parameter"}
        declarations = {r["name"]: r for r in program["derived"]}
        constants = {r["name"]: r for r in program["parameters"]}
        if (
            program.get("relations") != []
            or [u for u in program.get("units", []) if u.get("name") == "CAD"]
            != [{"name": "CAD", "kind": "currency", "minor_units": 2}]
            or set(declarations) != set(derived)
            or len(declarations) != len(program["derived"])
            or set(constants) != set(parameters)
            or len(constants) != len(program["parameters"])
        ):
            return None
        frame = {
            "effective_from": binding.period["start"],
            "effective_to": binding.period["end"],
        }
        for name, rule in rules.items():
            declaration = (declarations if name in derived else constants)[name]
            common_keys = {
                "id",
                "name",
                "source",
                "source_url",
                "corpus_citation_path",
                "unit",
                "versions",
            }
            semantic_keys = (
                {"entity", "dtype", "period", "semantics", "expr"}
                if name in derived
                else {"indexed_by"}
            )
            if set(declaration) != common_keys | semantic_keys:
                return None
            if any(
                declaration.get(k) != v
                for k, v in {
                    "id": prefix + name,
                    "name": name,
                    "source": rule["source"],
                    "source_url": None,
                    "corpus_citation_path": binding.form.citation,
                    "unit": rule.get("unit"),
                }.items()
            ):
                return None
            if name in derived:
                judgment = rule["dtype"] == "Judgment"
                semantics = "judgment" if judgment else "scalar"
                expr = _compiled(
                    _tree(rule["versions"][0]["formula"]), rules, inputs, judgment
                )
                expected = {
                    "entity": rule["entity"],
                    "dtype": "judgment" if judgment else "decimal",
                    "period": "Year",
                    "semantics": semantics,
                    "expr": expr,
                    "versions": [{**frame, "semantics": semantics, "expr": expr}],
                }
                if any(not _equal(declaration.get(k), v) for k, v in expected.items()):
                    return None
            else:
                versions = declaration["versions"]
                if (
                    declaration.get("indexed_by") is not None
                    or len(versions) != 1
                    or set(versions[0]) != {"effective_from", "effective_to", "values"}
                    or any(versions[0][k] != v for k, v in frame.items())
                    or set(versions[0]["values"]) != {"0"}
                    or _wire_number(versions[0]["values"]["0"])
                    != _number(rule["versions"][0]["formula"])
                ):
                    return None
        valid = []
        roles = dict(binding.roles)
        row_derived = {n for _, n in binding.row_owners if n in derived}
        observed_names = set()
        row_cases = []
        claimants = set()
        if (
            not isinstance(cases, (list, tuple))
            or len({case_key(c) for c in cases}) != len(cases)
            or len(evidence.records) != len(cases)
        ):
            return None
        for case in cases:
            if case.get("period") != binding.period:
                return None
            key = case_key(case)
            records = [r for r in evidence.records if r.case_sha256 == key]
            if len(records) != 1:
                return None
            record = records[0]
            request = json.loads(record.request_json)
            response = json.loads(record.response_json)
            if (
                not strict_explain_metadata(request, response)
                or len(request["queries"]) != 1
                or len(response["results"]) != 1
            ):
                return None
            query = request["queries"][0]
            result = response["results"][0]
            entity_id = case.get("entity_id")
            outputs = case.get("output")
            facts = binding.case_facts(case)
            if (
                not isinstance(entity_id, str)
                or not entity_id
                or not isinstance(outputs, dict)
                or not outputs
                or facts is None
            ):
                return None
            if (
                query.get("entity_id") != entity_id
                or query.get("period") != binding.period
                or query.get("assessment_date") is not None
                or any(
                    result.get(k) != query.get(k)
                    for k in ("entity_id", "period", "assessment_date")
                )
            ):
                return None
            if (
                set(query["outputs"]) != set(outputs)
                or len(query["outputs"]) != len(outputs)
                or set(result["outputs"]) != set(outputs)
            ):
                return None
            names = []
            for ref in outputs:
                if not ref.startswith(prefix) or ref[len(prefix) :] not in derived:
                    return None
                names.append(ref[len(prefix) :])
            entities = {derived[n]["entity"] for n in names}
            if len(entities) != 1:
                return None
            entity = next(iter(entities))
            trace_names = set(names)
            pending = list(names)
            while pending:
                owner = pending.pop()
                for dependency in ast.walk(
                    _tree(rules[owner]["versions"][0]["formula"])
                ):
                    if (
                        isinstance(dependency, ast.Name)
                        and dependency.id in derived
                        and dependency.id not in trace_names
                    ):
                        trace_names.add(dependency.id)
                        pending.append(dependency.id)
            if set(result["trace"]) != {prefix + n for n in trace_names}:
                return None
            needed = set().union(*(binding.reached_inputs(n, rules) for n in names))
            if set(facts) != needed:
                return None
            wire = request["dataset"]["inputs"]
            if (
                request["dataset"].get("relations") != []
                or len(wire) != len(needed)
                or {f["name"] for f in wire} != {prefix + "input." + n for n in needed}
            ):
                return None
            values = {}
            for fact in wire:
                n = fact["name"][len(prefix + "input.") :]
                decl = inputs[n]
                value = fact.get("value")
                if (
                    fact.get("entity") != decl["entity"]
                    or decl["entity"] != entity
                    or fact.get("entity_id") != entity_id
                    or fact.get("interval")
                    != {"start": binding.period["start"], "end": binding.period["end"]}
                    or not isinstance(value, dict)
                    or set(value) != {"kind", "value"}
                ):
                    return None
                if decl["dtype"] == "Boolean":
                    if (
                        value["kind"] != "bool"
                        or type(value["value"]) is not bool
                        or type(facts[n]) is not bool
                        or value["value"] is not facts[n]
                    ):
                        return None
                    values[n] = value["value"]
                elif decl["dtype"] == "Date":
                    if (
                        value["kind"] != "date"
                        or type(value["value"]) is not str
                        or value["value"] != facts[n]
                    ):
                        return None
                    values[n] = date.fromisoformat(value["value"])
                else:
                    v = _wire_number(value)
                    if v is None or v != _number(facts[n]) or v < 0:
                        return None
                    values[n] = v
            for name in trace_names:
                rule = derived[name]
                ref = prefix + name
                node = _tree(rule["versions"][0]["formula"])
                expected = _evaluate(node, rules, values, binding.period)
                judgment = rule["dtype"] == "Judgment"
                trace = result["trace"][ref]
                observations = [trace]
                if name in names:
                    observations.append(result["outputs"][ref])
                for observed in observations:
                    if any(
                        observed.get(k) != v
                        for k, v in {
                            "id": ref,
                            "name": name,
                            "kind": "judgment" if judgment else "scalar",
                            "unit": rule.get("unit"),
                        }.items()
                    ):
                        return None
                    if judgment:
                        outcome = "holds" if expected else "not_holds"
                        if (
                            type(expected) is not bool
                            or observed.get("outcome") != outcome
                            or (name in names and outputs[ref] != outcome)
                        ):
                            return None
                    elif (
                        observed.get("dtype") != "decimal"
                        or _wire_number(observed.get("value")) != expected
                        or (name in names and _number(outputs[ref]) != expected)
                    ):
                        return None
                dependencies = {
                    prefix + n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Name) and n.id in derived
                }
                if (
                    trace.get("entity_id") != entity_id
                    or trace.get("source") != rule["source"]
                    or trace.get("executed_expression")
                    != _render(node, rules, inputs, prefix, judgment)
                    or set(trace.get("dependencies", [])) != dependencies
                    or len(trace.get("dependencies", [])) != len(dependencies)
                ):
                    return None
                observed_names.add(name)
            by_row = dict(binding.row_owners)
            entered = dict(binding.entered)
            total, assistance, other = (entered[by_row[n]] for n in (1, 2, 4))
            if {total, assistance} <= values.keys():
                a, b = values[total], values[assistance]
                if not 0 <= b <= a:
                    return None
                if other in values:
                    cap = _number(rules[roles["cap"]]["versions"][0]["formula"])
                    if not 0 <= values[other] <= min(cap, a - b):
                        return None
            if row_derived <= set(names):
                row_cases.append(key)
            if (
                roles["medical"] in names
                and result["outputs"][prefix + roles["medical"]]["outcome"] == "holds"
            ):
                claimants.add(entity_id)
            valid.append((key, entity_id, facts, names, outputs))
        if observed_names != set(derived) or not row_cases or len(claimants) < 2:
            return None
        pairs = []
        for role, name in roles.items():
            if name not in derived or derived[name]["dtype"] != "Judgment":
                continue
            required = binding.reached_inputs(name, rules)
            for fact_name in required:
                found = []
                for a in valid:
                    for b in valid:
                        if (
                            a[0] >= b[0]
                            or a[1] != b[1]
                            or name not in a[3]
                            or name not in b[3]
                            or a[2].keys() != b[2].keys()
                        ):
                            continue
                        changed = {k for k in a[2] if not _equal(a[2][k], b[2][k])}
                        if (
                            changed == {fact_name}
                            and a[4][prefix + name] != b[4][prefix + name]
                        ):
                            found.append((a[0], b[0]))
                if not found:
                    return None
                pairs.append((name, fact_name, min(found)))
        return ScheduleObservation(
            binding.content_sha256,
            binding.form.source_sha256,
            tuple(sorted(row_cases)),
            tuple(sorted(pairs)),
            tuple(sorted(claimants)),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
    ):
        return None


def observed_schedule_condition_witnesses(
    binding,
    evidence,
    content,
    cases,
    branches,
    *,
    source_text,
    corpus_citation_path,
    principal_rules,
):
    """Exact Schedule effects, minted only from complete actual observation."""
    result = {branch: set() for branch in branches}
    observed = observe_schedule(binding, evidence, content, cases)
    if observed is None:
        return result
    raw = binding.form.source
    rules = _named(_payload(content), "rules")
    if (
        source_text not in (raw, raw.rstrip())
        or corpus_citation_path != binding.form.citation
        or {n: r for n, r in rules.items() if r["kind"] == "derived"} != principal_rules
    ):
        return result
    roles = dict(binding.roles)
    medical = binding.conditions.medical
    residence = binding.conditions.residence
    before = raw[: medical.span[0]].rstrip()
    heading_start = before.rfind("\n\n") + 2
    medical_spans = {
        (medical.span[0], medical.permission_span[1]),
        (heading_start, medical.permission_span[1]),
    }
    residence_spans = {(residence.yes_span[0], residence.no_span[1])}
    suffix = raw[residence.no_span[1] :]
    if suffix.strip() == "See the privacy notice on your return.":
        residence_spans.add((residence.yes_span[0], len(raw.rstrip())))
    groups = [
        ((roles["medical"],), medical_spans),
        ((roles["residence_yes"], roles["residence_no"]), residence_spans),
    ]
    for branch in branches:
        if (
            branch.kind != "exception-clause"
            or branch.path
            or source_text[branch.start : branch.end] != branch.text
        ):
            continue
        for owners, spans in groups:
            if (
                branch.start,
                branch.end,
            ) not in spans or branch.required_principal_output not in (None, *owners):
                continue
            pairs = tuple(
                sorted(
                    (name, fact, keys)
                    for name, fact, keys in observed.pairs
                    if name in owners
                )
            )
            if {name for name, _, _ in pairs} != set(owners):
                continue
            result[branch].add(
                (binding.content_sha256, binding.form.source_sha256, owners, pairs)
            )
    return result
