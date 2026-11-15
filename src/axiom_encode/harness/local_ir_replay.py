"""Closed pinned-IR replay; plain return values never confer source authority."""

from __future__ import annotations

from datetime import date
from decimal import Context, Decimal, InvalidOperation, localcontext
from typing import Any

from .actual_local_relation_evidence import (
    _closed_interval,
    _exact_engine_decimal,
    _related_ids,
    _replay_values,
    _wire_value,
)

_FIELDS = {
    "literal": {"kind", "value"},
    "input": {"kind", "name"},
    "derived": {"kind", "name"},
    "parameter_lookup": {"kind", "parameter", "index"},
    "add": {"kind", "items"},
    "min": {"kind", "items"},
    "max": {"kind", "items"},
    "and": {"kind", "items"},
    "or": {"kind", "items"},
    "sub": {"kind", "left", "right"},
    "mul": {"kind", "left", "right"},
    "comparison": {"kind", "left", "op", "right"},
    "if": {"kind", "condition", "then_expr", "else_expr"},
    "sum_related": {
        "kind",
        "relation",
        "current_slot",
        "related_slot",
        "value",
        "where",
    },
}


def _children(expr: dict) -> list[dict]:
    kind = expr["kind"]
    if kind in ("literal", "input", "derived"):
        return []
    if kind == "parameter_lookup":
        return [expr["index"]]
    if "items" in expr:
        return expr["items"]
    if kind in ("sub", "mul", "comparison"):
        return [expr["left"], expr["right"]]
    if kind == "if":
        return [expr["condition"], expr["then_expr"], expr["else_expr"]]
    return [expr["value"]]


def _shape(expr: Any, *, root: bool = True) -> None:
    if (
        not isinstance(expr, dict)
        or type(expr.get("kind")) is not str
        or expr["kind"] not in _FIELDS
        or set(expr) != _FIELDS[expr["kind"]]
    ):
        raise ValueError("unsupported IR shape")
    kind = expr["kind"]
    if kind == "literal":
        try:
            _wire_value(expr["value"])
        except (InvalidOperation, TypeError, OverflowError) as exc:
            raise ValueError("invalid IR literal") from exc
    for field in ("name", "parameter", "relation"):
        if field in expr:
            value = expr[field]
            if type(value) is not str or not value:
                raise ValueError("invalid IR identity")
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("invalid IR identity") from exc
    if kind == "comparison" and (
        type(expr["op"]) is not str
        or expr["op"] not in {"lt", "lte", "gt", "gte", "eq", "ne"}
    ):
        raise ValueError("unsupported comparison")
    if "items" in expr and (not isinstance(expr["items"], list) or not expr["items"]):
        raise ValueError("nonempty operand list required")
    if expr["kind"] == "sum_related" and (
        not root
        or expr["where"] is not None
        or type(expr["current_slot"]) is not int
        or type(expr["related_slot"]) is not int
        or (expr["current_slot"], expr["related_slot"]) != (1, 0)
        or not isinstance(expr["value"], dict)
        or expr["value"].get("kind") != "derived"
    ):
        raise ValueError("unsupported aggregate profile")
    for child in _children(expr):
        _shape(child, root=False)


def _number(value: Any) -> Decimal:
    result = _exact_engine_decimal(value)
    if result is None:
        raise ValueError("nonexact numeric operand")
    return result


def _arithmetic(kind: str, values: list[Any]) -> Decimal:
    numbers = [_number(v) for v in values]
    with localcontext(Context(prec=80, Emin=-999999, Emax=999999)):
        if kind == "add":
            value = Decimal(0)
            for operand in numbers:
                value = _number(value + operand)
        elif kind == "sub":
            value = _number(numbers[0] - numbers[1])
        elif kind == "mul":
            value = _number(numbers[0] * numbers[1])
        elif kind == "min":
            value = min(numbers)
        elif kind == "max":
            value = max(numbers)
        else:
            raise ValueError("unsupported arithmetic")
    return value


def _selected(versions: Any, start: str) -> dict:
    if not isinstance(versions, list) or not versions:
        raise ValueError("versions required")
    selected = []
    intervals = []
    for version in versions:
        if not isinstance(version, dict):
            raise ValueError("invalid version")
        frame = _closed_interval(
            {
                "start": version.get("effective_from"),
                "end": version.get("effective_to") or "9999-12-31",
            }
        )
        if frame is None or any(
            frame[0] <= old[1] and old[0] <= frame[1] for old in intervals
        ):
            raise ValueError("ambiguous version interval")
        intervals.append(frame)
        if frame[0] <= date.fromisoformat(start) <= frame[1]:
            selected.append(version)
    if len(selected) != 1:
        raise ValueError("no unique active version")
    return selected[0]


def check_compiled_replay(
    artifact: dict, policy: dict, request: dict, response: dict
) -> dict:
    """Compare authored values and exact IR paths; requires custody at consumer."""
    authored = _replay_values(policy, request, response)
    program = artifact["program"]
    rules = {r["name"]: r for r in program["derived"]}
    params = {r["name"]: r for r in program["parameters"]}
    if (
        len(rules) != len(program["derived"])
        or len(params) != len(program["parameters"])
        or rules.keys() & params.keys()
    ):
        raise ValueError("ambiguous compiled names")
    query = request["queries"][0]
    root = query["entity_id"]
    period = query["period"]
    interval = {k: period[k] for k in ("start", "end")}
    selected_interval = _closed_interval(interval)
    trace = response["results"][0]["trace"]
    namespace = next(iter(trace.values()))["id"].split("#")[0]
    for name, item in {**rules, **params}.items():
        if item.get("id") != namespace + "#" + name:
            raise ValueError("compiled identity mismatch")
    for rule in rules.values():
        _shape(rule["expr"])
        for version in rule["versions"]:
            _shape(version["expr"])
    facts = {}
    for fact in request["dataset"]["inputs"]:
        frame = _closed_interval(fact["interval"])
        if (
            frame is not None
            and frame[0] <= selected_interval[0] <= selected_interval[1] <= frame[1]
        ):
            key = (fact["name"].split("#input.", 1)[1], fact["entity_id"])
            if key in facts:
                raise ValueError("duplicate fact")
            facts[key] = _wire_value(fact["value"])
    cache = {}
    active = set()

    def reference(name, entity):
        return (
            namespace
            + "#"
            + name
            + ("" if entity == root else "@entity:" + entity.encode().hex())
        )

    def references(expr, entity):
        kind = expr["kind"]
        result = set()
        if kind == "derived":
            result.add(reference(expr["name"], entity))
        if kind == "parameter_lookup":
            result.add(namespace + "#" + expr["parameter"])
        for child in _children(expr):
            result.update(references(child, entity))
        return result

    def evaluate_rule(name, entity):
        key = (name, entity)
        if key in active:
            raise ValueError("compiled cycle")
        if key in cache:
            return cache[key]
        if name not in rules or key not in authored:
            raise ValueError("missing reached instance")
        active.add(key)
        dependencies = set()
        skipped = set()
        reads = {}

        def skip(expr, reason):
            skipped.update((ref, reason) for ref in references(expr, entity))

        def evaluate(expr, current_entity=entity):
            kind = expr["kind"]
            if kind == "literal":
                return _wire_value(expr["value"])
            if kind == "input":
                return facts[(expr["name"], current_entity)]
            if kind == "derived":
                dependencies.add(reference(expr["name"], current_entity))
                return evaluate_rule(expr["name"], current_entity)
            if kind == "parameter_lookup":
                index = evaluate(expr["index"], current_entity)
                if type(index) is not int:
                    raise ValueError("noninteger parameter index")
                parameter = params[expr["parameter"]]
                version = _selected(parameter["versions"], period["start"])
                value = _wire_value(version["values"][str(index)])
                expected_read = {
                    "parameter": parameter["id"],
                    "id": parameter["id"],
                    "name": parameter["name"],
                    "index": index,
                    "value": value,
                    "effective_from": version["effective_from"],
                    "effective_to": version.get("effective_to"),
                    "source": parameter.get("source"),
                }
                if parameter.get("unit") is not None:
                    expected_read["unit"] = parameter["unit"]
                reads[(parameter["name"], index)] = expected_read
                return value
            if kind in ("add", "min", "max"):
                return _arithmetic(
                    kind, [evaluate(x, current_entity) for x in expr["items"]]
                )
            if kind in ("sub", "mul"):
                return _arithmetic(
                    kind,
                    [
                        evaluate(expr["left"], current_entity),
                        evaluate(expr["right"], current_entity),
                    ],
                )
            if kind == "comparison":
                left = evaluate(expr["left"], current_entity)
                right = evaluate(expr["right"], current_entity)
                if type(left) in (int, Decimal) and type(right) in (int, Decimal):
                    left, right = _number(left), _number(right)
                elif type(left) is not type(right) or type(left) not in (bool, date):
                    raise ValueError("comparison type mismatch")
                op = expr["op"]
                if op == "eq":
                    return left == right
                if op == "ne":
                    return left != right
                if type(left) is bool:
                    raise ValueError("ordered Boolean comparison")
                if op == "lt":
                    return left < right
                if op == "lte":
                    return left <= right
                if op == "gt":
                    return left > right
                if op == "gte":
                    return left >= right
                raise ValueError("unsupported comparison")
            if kind in ("and", "or"):
                for i, item in enumerate(expr["items"]):
                    value = evaluate(item, current_entity)
                    if type(value) is not bool:
                        raise ValueError("nonboolean selector")
                    if (kind == "and" and not value) or (kind == "or" and value):
                        for later in expr["items"][i + 1 :]:
                            skip(later, "short_circuit")
                        return value
                return kind == "and"
            if kind == "if":
                condition = evaluate(expr["condition"], current_entity)
                if type(condition) is not bool:
                    raise ValueError("nonboolean condition")
                chosen, other = (
                    ("then_expr", "else_expr")
                    if condition
                    else ("else_expr", "then_expr")
                )
                skip(expr[other], "branch_not_selected")
                return evaluate(expr[chosen], current_entity)
            if kind == "sum_related":
                ids = _related_ids(
                    request["dataset"].get("relations"),
                    relation=expr["relation"],
                    root=current_entity,
                    interval=interval,
                    current_slot=expr["current_slot"],
                    related_slot=expr["related_slot"],
                )
                if ids is None:
                    raise ValueError("invalid relation roster")
                return _arithmetic("add", [evaluate(expr["value"], i) for i in ids])
            raise ValueError("unsupported expression")

        rule = rules[name]
        version = _selected(rule["versions"], period["start"])
        value = evaluate(version["expr"])
        node = trace[reference(name, entity)]
        actual = (
            _wire_value(node["value"])
            if node["kind"] == "scalar"
            else node["outcome"] == "holds"
        )
        if (
            type(value) is bool
            and type(actual) is not bool
            or type(actual) is bool
            and type(value) is not bool
            or value != actual
        ):
            raise ValueError("IR value mismatch")
        reported = node.get("not_evaluated_dependencies", [])
        if not isinstance(reported, list) or any(
            not isinstance(x, dict)
            or set(x) != {"dependency", "reason"}
            or type(x["dependency"]) is not str
            or type(x["reason"]) is not str
            for x in reported
        ):
            raise ValueError("invalid skipped trace")
        actual_skips = [(x["dependency"], x["reason"]) for x in reported]
        if len(set(actual_skips)) != len(actual_skips) or set(actual_skips) != skipped:
            raise ValueError("skipped path mismatch: " + name)
        if set(node.get("dependencies", [])) != dependencies:
            raise ValueError("IR dependency mismatch")
        actual_reads = {}
        for reported_read in node.get("parameter_reads", []):
            read = dict(reported_read)
            read["value"] = _wire_value(read["value"])
            read_key = (read["name"], read["index"])
            if read_key in actual_reads:
                raise ValueError("duplicate IR parameter read")
            actual_reads[read_key] = read
        if actual_reads.keys() != reads.keys():
            raise ValueError("IR parameter path mismatch")
        for read_key, expected_read in reads.items():
            actual_read = actual_reads[read_key]
            if actual_read != expected_read or type(actual_read["value"]) is not type(
                expected_read["value"]
            ):
                raise ValueError("IR selected parameter mismatch")
        active.remove(key)
        cache[key] = value
        return value

    for output in query["outputs"]:
        evaluate_rule(output.split("#", 1)[1], root)
    if set(cache) != set(authored):
        raise ValueError("different reached closure")
    return cache
