"""Standalone observation of the closed child-credit graph.

This module has no capture, snapshot or source-coverage callers. A successful
observation checks diagnostic evidence; it does not grant publication authority.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

import yaml

from axiom_encode.bound_child_credit_form import bind_child_credit_form

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


def _lower(node, inputs, rules, prefix, boolean=False):
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
            "value": {"kind": "decimal", "value": str(node.value)},
        }
    if isinstance(node, ast.BinOp):
        a, b = (_lower(x, inputs, rules, prefix) for x in (node.left, node.right))
        if isinstance(node.op, ast.Add):
            return {"kind": "add", "items": [a, b]}
        return {
            "kind": {ast.Sub: "sub", ast.Mult: "mul"}[type(node.op)],
            "left": a,
            "right": b,
        }
    if isinstance(node, ast.BoolOp):
        return {
            "kind": "and",
            "items": [_lower(n, inputs, rules, prefix, True) for n in node.values],
        }
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Gt)
    ):
        return {
            "kind": "comparison",
            "left": _lower(node.left, inputs, rules, prefix),
            "op": "gt",
            "right": _lower(node.comparators[0], inputs, rules, prefix),
        }
    if isinstance(node, ast.If) and len(node.body) == len(node.orelse) == 1:
        return {
            "kind": "if",
            "condition": _lower(node.test, inputs, rules, prefix, True),
            "then_expr": _lower(node.body[0].value, inputs, rules, prefix),
            "else_expr": _lower(node.orelse[0].value, inputs, rules, prefix),
        }
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "min":
            return {
                "kind": "min",
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
                    "slot_entities": ["SourceEntry", "Person"],
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
            value = version["formula"]
            wire = (
                {"kind": "integer", "value": value}
                if type(value) is int
                else {"kind": "decimal", "value": str(value)}
            )
            parameters.append(
                dict(
                    **shared,
                    indexed_by=None,
                    versions=[dict(**dates, values={"0": wire})],
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
        for n in sorted(inputs)
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
        return prefix + n["parameter"] + "[0]"
    if k == "sum_related":
        return f'sum_related("{n["relation"]}", 1, 0, {_render(n["value"], prefix)})'
    if k == "if":
        return f"(if {_render(n['condition'], prefix)} then {_render(n['then_expr'], prefix)} else {_render(n['else_expr'], prefix)})"
    if k in ("sub", "mul", "comparison"):
        op = {
            "sub": "-",
            "mul": "*",
            "comparison": "==" if n.get("op") == "eq" else ">",
        }[k]
        return f"({_render(n['left'], prefix)} {op} {_render(n['right'], prefix)})"
    values = [_render(x, prefix) for x in n["items"]]
    if k == "min":
        return "min(" + ", ".join(values) + ")"
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
    if dtype == "Money":
        return _number(value)
    raise ValueError("raw fact type differs")


def _dataset(case, request, inputs, prefix, relation, query_kind, entity, interval):
    expected = {}
    tuples = []
    for key, value in case["input"].items():
        if key == relation:
            if not isinstance(value, list):
                raise ValueError("table rows required")
            for i, row in enumerate(value, 1):
                row_id = f"{entity}-{relation}-{i}"
                tuples.append(
                    {"name": relation, "tuple": [row_id, entity], "interval": interval}
                )
                for fact, raw in row.items():
                    name = fact.removeprefix(prefix + "input.")
                    if (
                        fact != prefix + "input." + name
                        or inputs[name]["entity"] != "SourceEntry"
                    ):
                        raise ValueError("row fact role differs")
                    expected[(row_id, name)] = _raw_value(raw, inputs[name]["dtype"])
        else:
            name = key.removeprefix(prefix + "input.")
            if key != prefix + "input." + name or inputs[name]["entity"] != query_kind:
                raise ValueError("query fact role differs")
            expected[(entity, name)] = _raw_value(value, inputs[name]["dtype"])
    dataset = request["dataset"]
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
class ChildCreditObservation:
    content_sha256: str
    source_sha256: str
    cases: tuple[str, ...]
    duty_pairs: tuple[tuple[str, str, str, str], ...]
    refund_cases: tuple[str, str, str]
    provider_instances: tuple[tuple[str, str], ...]
    per_child_cap_pair: tuple[str, str]


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
    relation = prefix + "relation." + owners["row_membership"]
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
                reads.append(
                    dict(
                        parameter=prefix + parameter_name,
                        name=parameter_name,
                        id=prefix + parameter_name,
                        index=0,
                        value=v["values"]["0"],
                        **({"unit": p["unit"]} if p["unit"] else {}),
                        effective_from=v["effective_from"],
                        effective_to=v["effective_to"],
                        source=p["source"],
                    )
                )
                read_parameters.add(parameter_name)
                return _number(v["values"]["0"]["value"])
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
                    _branch_references(
                        node["else_expr"] if selected else node["then_expr"]
                    )
                )
                return evaluate(node["then_expr"] if selected else node["else_expr"])
            if kind == "comparison":
                a, b = evaluate(node["left"]), evaluate(node["right"])
                return a == b if node["op"] == "eq" else a > b
            if kind in ("sub", "mul"):
                a, b = evaluate(node["left"]), evaluate(node["right"])
                return a - b if kind == "sub" else a * b
            if kind == "and":
                return all(evaluate(x) for x in node["items"])
            vals = [evaluate(x) for x in node["items"]]
            return {"add": sum, "min": min}[kind](vals)

        value = evaluate(expressions[name])
        pending.remove(key)
        values[key] = value
        details[key] = (deps, reads, skipped)
        return value

    for n in names:
        derived(n, entity)
    # Inclusion is a reached-fact contract, including partial identity queries.
    for who in {e for e, n in used}:
        b, c, e = (
            (who, n)
            for n in (
                "eligible_expenses_b",
                "assistance_received_or_receivable_c",
                "amount_from_d_claimed_by_others_e",
            )
        )
        for key in (b, c, e):
            if key in used and facts[key] < 0:
                raise ValueError("negative entered amount")
        if b in used and c in used and facts[c] > facts[b]:
            raise ValueError("assistance not included")
        if (
            b in used
            and c in used
            and e in used
            and facts[e] > min(facts[b] - facts[c], Decimal(500))
        ):
            raise ValueError("other claim not from D")
    if ("taxes_to_pay" in {n for e, n in used}) and facts[(entity, "taxes_to_pay")] < 0:
        raise ValueError("negative tax domain")
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
                trace.get("not_evaluated_dependencies", []),
                [
                    {"dependency": prefix + n, "reason": "branch_not_selected"}
                    for n in sorted(skipped)
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
    )


def observe_child_credit(binding, evidence, content, cases):
    """Require complete compiled identity and actual typed provider observations."""
    try:
        if (
            evidence.engine_pin != _PROFILE_PIN
            or not evidence.bound(content, ())
            or bind_child_credit_form(
                content, binding.source.source, binding.source.citation
            )
            != binding
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
        if binding.source.citation != str(
            PurePosixPath(path.parts[0], "policy", *path.parts[2:]).with_suffix("")
        ):
            return None
        inputs, rules, expressions, parameters = _compiled(
            binding, content, json.loads(evidence.artifact_bytes), prefix
        )
        keys = [case_key(c) for c in cases]
        if (
            None in keys
            or len(set(keys)) != len(keys)
            or len(evidence.records) != len(cases)
        ):
            return None
        owners = dict(binding.rule_roles)
        records = []
        all_instances = set()
        all_parameters = set()
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
        if {n for n, e in all_instances} != set(expressions) or all_parameters != set(
            parameters
        ):
            return None
        # Observe different distributions with equal entered totals: the cap
        # must be applied per child before the claimant sum.
        cap_groups = {}
        for key, entity, facts, outputs, instances, _ in records:
            if owners["aggregate_d_export"] not in outputs:
                continue
            row_ids = {
                who
                for name, who in instances
                if name == owners["column_d"] and who != entity
            }
            if len(row_ids) < 2:
                continue
            totals = tuple(
                sum((facts[(who, name)] for who in row_ids), Decimal(0))
                for name in (
                    "eligible_expenses_b",
                    "assistance_received_or_receivable_c",
                )
            )
            cap_groups.setdefault((entity, totals), []).append(
                (key, outputs[owners["aggregate_d_export"]])
            )
        cap_pairs = [
            (a[0], b[0])
            for group in cap_groups.values()
            for a in group
            for b in group
            if a[0] < b[0] and a[1] != b[1]
        ]
        if not cap_pairs:
            return None
        pairs = []
        requirements = {
            "line1": (
                "ns479_volunteer_firefighters_ground_search_rescue_credit_eligible",
            ),
            "paper_attachment": ("completing_ns479", "filing_paper_return"),
            "table_completion": ("using_children_credit_section",),
            "receipts_retention": ("using_children_credit_section",),
            "receipts_non_enclosure": ("using_children_credit_section",),
            "additional_sheet": ("needs_additional_space",),
        }
        for role, fact_names in requirements.items():
            owner = owners[role]
            for fact in fact_names:
                found = []
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
                            found.append((a[0], b[0]))
                if not found:
                    return None
                left, right = min(found)
                pairs.append((role, fact, left, right))
        groups = {}
        excess, line5 = owners["conditional_excess"], owners["line5"]
        for key, entity, facts, outputs, _, _ in records:
            if excess not in outputs or line5 not in outputs:
                continue
            taxes = facts[(entity, "taxes_to_pay")]
            credit = outputs[line5]
            group = (
                entity,
                tuple(
                    sorted(
                        (k, str(v))
                        for k, v in facts.items()
                        if k != (entity, "taxes_to_pay")
                    )
                ),
            )
            groups.setdefault(group, {})[
                "below" if taxes < credit else "equal" if taxes == credit else "above"
            ] = key
        complete = [g for g in groups.values() if set(g) == {"below", "equal", "above"}]
        if not complete:
            return None
        group = min(
            complete, key=lambda g: tuple(g[k] for k in ("below", "equal", "above"))
        )
        return ChildCreditObservation(
            binding.content_sha256,
            binding.source.source_sha256,
            tuple(keys),
            tuple(pairs),
            tuple(group[k] for k in ("below", "equal", "above")),
            tuple(sorted(all_instances)),
            min(cap_pairs),
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
        return bind_child_credit_form(content, source, citation)
    except (KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


@dataclass(frozen=True)
class _ChildCreditCaptureContext:
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
        return _ChildCreditCaptureContext(
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
