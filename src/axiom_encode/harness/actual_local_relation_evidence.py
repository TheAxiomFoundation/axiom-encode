"""Private local-relation replay primitives; these do not mint execution evidence.

The observer integration must establish materializer/compile/process custody
before using these results. Plain records and historical JSON are not authority.
"""

from __future__ import annotations

from datetime import date
from decimal import Context, Decimal, localcontext
from typing import Any, Iterable

_MAX_COEFFICIENT = (1 << 96) - 1


def _exact_engine_decimal(value: Any) -> Decimal | None:
    """Accept the exact representable subset of rust_decimal, never round."""
    if type(value) is int:
        value = Decimal(value)
    if type(value) is not Decimal or not value.is_finite():
        return None
    sign, digits, exponent = value.as_tuple()
    if not isinstance(exponent, int) or exponent < -28 or exponent > 28:
        return None
    if len(digits) > 29:
        return None
    coefficient = int("".join(map(str, digits)))
    if exponent > 0:
        coefficient *= 10**exponent
    if coefficient > _MAX_COEFFICIENT:
        return None
    return value


def _checked_sum(values: Iterable[Decimal | int]) -> Decimal | None:
    """Exact bounded addition; reject precision reduction rather than imitate it."""
    total = Decimal(0)
    with localcontext(Context(prec=80, Emin=-999999, Emax=999999)) as context:
        # At most 29 coefficient digits plus 28 fractional digits; no ambient
        # Decimal context can round away a provider amount before the check.
        context.prec = 80
        for value in values:
            operand = _exact_engine_decimal(value)
            if operand is None:
                return None
            total = total + operand
            if _exact_engine_decimal(total) is None:
                return None
    return total


def _closed_interval(value: Any) -> tuple[date, date] | None:
    if not isinstance(value, dict) or set(value) != {"start", "end"}:
        return None
    points = []
    for key in ("start", "end"):
        raw = value[key]
        if not isinstance(raw, str) or len(raw) != 10:
            return None
        try:
            point = date.fromisoformat(raw)
        except ValueError:
            return None
        if point.isoformat() != raw:
            return None
        points.append(point)
    return (points[0], points[1]) if points[0] <= points[1] else None


def _valid_identity(value: Any) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _related_ids(
    edges: Any,
    *,
    relation: str,
    root: str,
    interval: dict[str, str],
    current_slot: int,
    related_slot: int,
) -> tuple[str, ...] | None:
    """895 full containment and sorted identities for a closed binary edge set.

    Duplicate tuples decline this observer's evidence. This does not change or
    describe ordinary engine admission, which deduplicates selected identities.
    The caller separately authenticates declaration slots, types and transport.
    """
    period = _closed_interval(interval)
    if (
        period is None
        or type(edges) is not list
        or not _valid_identity(relation)
        or not _valid_identity(root)
        or type(current_slot) is not int
        or type(related_slot) is not int
        or {current_slot, related_slot} != {0, 1}
    ):
        return None
    seen: set[tuple[str, str]] = set()
    selected: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {"name", "tuple", "interval"}:
            return None
        if edge["name"] != relation:
            return None
        ids = edge["tuple"]
        frame = _closed_interval(edge["interval"])
        if (
            type(ids) is not list
            or len(ids) != 2
            or any(not _valid_identity(i) for i in ids)
            or frame is None
        ):
            return None
        pair = (ids[0], ids[1])
        if pair in seen:
            return None
        seen.add(pair)
        if (
            ids[current_slot] == root
            and frame[0] <= period[0]
            and period[1] <= frame[1]
        ):
            selected.add(ids[related_slot])
    return tuple(sorted(selected))


def _wire_value(value: Any) -> Any:
    """Decode the closed scalar wire subset; raise for unsupported values."""
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        raise ValueError("unsupported scalar wire")
    kind, raw = value["kind"], value["value"]
    if kind == "bool" and type(raw) is bool:
        return raw
    if kind == "date" and isinstance(raw, str):
        point = date.fromisoformat(raw)
        if point.isoformat() == raw:
            return point
    if kind == "integer" and type(raw) is int and -(1 << 63) <= raw < (1 << 63):
        number = _exact_engine_decimal(raw)
        if number is not None:
            return raw
    if kind == "decimal" and isinstance(raw, str) and len(raw) <= 100:
        number = _exact_engine_decimal(Decimal(raw))
        if number is not None:
            return number
    raise ValueError("unsupported scalar wire")


def _value_matches_declaration(value: Any, declaration: dict[str, Any]) -> bool:
    dtype = declaration.get("dtype")
    if dtype in ("Judgment", "Boolean"):
        return type(value) is bool and declaration.get("unit") is None
    if dtype == "Date":
        return type(value) is date and declaration.get("unit") is None
    if dtype == "Integer":
        return type(value) is int and -(1 << 63) <= value < (1 << 63)
    if dtype in ("Money", "Decimal"):
        return _exact_engine_decimal(value) is not None
    return False


def _derived_order(rules: dict[str, Any]) -> tuple[str, ...]:
    """Topological authored order, never solve equations from observed peers."""
    import re

    from .same_case_execution import _formula_names

    edges = {}
    for name, rule in rules.items():
        dependencies = set()
        versions = rule.get("versions")
        if not isinstance(versions, list) or not versions:
            raise ValueError("versions required")
        for version in versions:
            formula = version.get("formula") if isinstance(version, dict) else None
            if not isinstance(formula, str):
                raise ValueError("formula required")
            names = _formula_names(formula)
            if names is None:
                raise ValueError("unsupported formula")
            dependencies.update(names & rules.keys())
            dependencies.update(
                re.findall(
                    r"sum\([A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\)", formula
                )
            )
        if not dependencies <= rules.keys():
            raise ValueError("unknown provider")
        edges[name] = dependencies
    visiting, done, order = set(), set(), []

    def visit(name):
        if name in visiting:
            raise ValueError("cyclic authored declarations")
        if name in done:
            return
        visiting.add(name)
        for dependency in sorted(edges[name]):
            visit(dependency)
        visiting.remove(name)
        done.add(name)
        order.append(name)

    for name in sorted(rules):
        visit(name)
    return tuple(order)


def _check_parameter_reads(
    node: dict[str, Any],
    names: set[str],
    declarations: dict[str, dict],
    environment: dict[str, Any],
    namespace: str,
    start: str,
) -> None:
    """Match reached parameter values, indexes and versions; not source credit."""
    from .source_completeness import _DeclaredDateTable

    expected = names & declarations.keys()
    reads = node.get("parameter_reads", [])
    if not isinstance(reads, list) or any(not isinstance(r, dict) for r in reads):
        raise ValueError("malformed parameter reads")
    actual = [r.get("name") for r in reads]
    if (
        any(not isinstance(n, str) for n in actual)
        or len(set(actual)) != len(actual)
        or set(actual) != expected
    ):
        raise ValueError("selected parameter read set mismatch")
    for read in reads:
        name = read["name"]
        rule = declarations[name]
        versions = rule.get("versions", [])
        if not isinstance(versions, list) or any(
            not isinstance(v, dict) for v in versions
        ):
            raise ValueError("invalid parameter versions")
        selected = []
        for version in versions:
            lower, upper = version.get("effective_from"), version.get("effective_to")
            if (
                not isinstance(lower, str)
                or _closed_interval({"start": lower, "end": upper or "9999-12-31"})
                is None
            ):
                raise ValueError("invalid parameter interval")
            if lower <= start <= (upper or "9999-12-31"):
                selected.append(version)
        if len(selected) != 1 or name not in environment:
            raise ValueError("ambiguous selected parameter")
        version = selected[0]
        value = environment[name]
        index = 0
        if isinstance(value, _DeclaredDateTable):
            index = value.selected_index_value
            cells = dict(value.cells)
            if type(index) is not int or index not in cells:
                raise ValueError("missing Date table index")
            value = cells[index]
        reference = namespace + "#" + name
        allowed = {
            "parameter",
            "name",
            "id",
            "index",
            "value",
            "effective_from",
            "effective_to",
            "source",
        }
        if rule.get("unit") is not None:
            allowed.add("unit")
        if (
            set(read) != allowed
            or read.get("id") != reference
            or read.get("parameter") != reference
            or type(read.get("index")) is not int
            or read["index"] != index
            or read.get("effective_from") != version["effective_from"]
            or read.get("effective_to") != version.get("effective_to")
            or read.get("source") != rule.get("source")
            or read.get("unit") != rule.get("unit")
        ):
            raise ValueError("parameter lineage mismatch")
        actual_value = _wire_value(read["value"])
        value_declaration = rule
        if rule.get("dtype") in ("Rate", "Decimal", "Money", "Integer"):
            # The older formula environment may contain binary floats. Bind the
            # exact authored literal instead; never convert a float into evidence.
            import ast
            from decimal import InvalidOperation

            try:
                if rule.get("dtype") == "Integer":
                    value = ast.literal_eval(str(version["formula"]))
                    if type(value) is not int:
                        raise ValueError("noninteger parameter")
                else:
                    value = Decimal(str(version["formula"]))
                    if _exact_engine_decimal(value) is None:
                        raise ValueError("unrepresentable parameter")
            except (KeyError, ValueError, SyntaxError, InvalidOperation) as exc:
                raise ValueError("nonliteral parameter outside replay profile") from exc
            if rule.get("dtype") == "Rate":
                value_declaration = {**rule, "dtype": "Decimal"}
        if (
            not _value_matches_declaration(actual_value, value_declaration)
            or type(actual_value) is bool
            or actual_value != value
        ):
            raise ValueError("parameter value mismatch")


def _replay_values(
    payload: dict[str, Any],
    request: dict[str, Any],
    response: dict[str, Any],
) -> dict[tuple[str, str], Any]:
    """Pure authored value replay, NOT an observation or corroboration grant.

    Used to establish the finite replay contract before capture integration.
    Process custody, compiled binding, exact trace metadata/parameter reads and
    companion outcome checks are separate required gates. No caller may treat
    this dictionary as source evidence.
    """
    import re

    from . import source_completeness as sc
    from .same_case_execution import strict_explain_metadata

    if payload.get("imports") != [] or not strict_explain_metadata(request, response):
        raise ValueError("unsupported module or execution mode")
    queries, results = request.get("queries"), response.get("results")
    if (
        not isinstance(queries, list)
        or len(queries) != 1
        or not isinstance(results, list)
        or len(results) != 1
    ):
        raise ValueError("single root required")
    query, result = queries[0], results[0]
    root, period = query["entity_id"], query["period"]
    interval = {k: period[k] for k in ("start", "end")}
    if (
        not _valid_identity(root)
        or _closed_interval(interval) is None
        or any(
            result.get(k) != query.get(k)
            for k in ("entity_id", "period", "assessment_date")
        )
    ):
        raise ValueError("query identity mismatch")
    declarations = payload.get("rules", [])
    inputs = payload.get("inputs", [])
    if any(
        not isinstance(r, dict) or not _valid_identity(r.get("name"))
        for r in declarations + inputs
    ):
        raise ValueError("malformed declarations")
    names = [r["name"] for r in declarations + inputs]
    if len(set(names)) != len(names):
        raise ValueError("ambiguous declarations")
    parameters = {r["name"]: r for r in declarations if r.get("kind") == "parameter"}
    rules = {r["name"]: r for r in declarations if r.get("kind") == "derived"}
    order = _derived_order(rules)
    relation_rules = {
        r["name"]: r for r in declarations if r.get("kind") == "data_relation"
    }
    input_rules = {r["name"]: r for r in inputs}
    trace = result.get("trace")
    if not isinstance(trace, dict) or not trace:
        raise ValueError("trace required")
    namespaces = {
        n.get("id", "").rsplit("#", 1)[0] for n in trace.values() if isinstance(n, dict)
    }
    if len(namespaces) != 1:
        raise ValueError("one local namespace required")
    namespace = next(iter(namespaces))
    observed = {}
    for key, node in trace.items():
        name, entity = node.get("name"), node.get("entity_id")
        if name not in rules or not _valid_identity(entity):
            raise ValueError("unknown trace instance")
        public = namespace + "#" + name
        expected_key = (
            public if entity == root else public + "@entity:" + entity.encode().hex()
        )
        if (
            key != expected_key
            or node.get("id") != public
            or (name, entity) in observed
        ):
            raise ValueError("trace identity mismatch")
        declaration = rules[name]
        if declaration.get("dtype") == "Judgment":
            if (
                node.get("kind") != "judgment"
                or node.get("unit") is not None
                or node.get("dtype") is not None
                or node.get("outcome")
                not in (
                    "holds",
                    "not_holds",
                )
            ):
                raise ValueError("judgment mismatch")
            value = node["outcome"] == "holds"
        else:
            expected_dtype = {
                "Money": "decimal",
                "Decimal": "decimal",
                "Integer": "integer",
                "Boolean": "bool",
            }.get(declaration.get("dtype"))
            if (
                node.get("kind") != "scalar"
                or expected_dtype is None
                or node.get("dtype") != expected_dtype
                or node.get("unit") != declaration.get("unit")
            ):
                raise ValueError("scalar declaration mismatch")
            value = _wire_value(node.get("value"))
            if (expected_dtype == "bool") != (type(value) is bool):
                raise ValueError("boolean is not a number")
        if not _value_matches_declaration(value, declaration):
            raise ValueError("trace value type mismatch")
        observed[(name, entity)] = value
    outputs = query.get("outputs")
    if (
        not isinstance(outputs, list)
        or not outputs
        or len(set(outputs)) != len(outputs)
    ):
        raise ValueError("unique query outputs required")
    if any(ref not in trace or trace[ref].get("entity_id") != root for ref in outputs):
        raise ValueError("queried root trace missing")
    dataset = request["dataset"]
    facts: dict[str, dict[str, Any]] = {}
    for fact in dataset.get("inputs", []):
        prefix = namespace + "#input."
        reference = fact.get("name", "")
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise ValueError("foreign input")
        name = reference[len(prefix) :]
        declaration = input_rules.get(name)
        entity = fact.get("entity_id")
        frame = _closed_interval(fact.get("interval"))
        selected = _closed_interval(interval)
        if (
            declaration is None
            or fact.get("entity") != declaration.get("entity")
            or not _valid_identity(entity)
            or frame is None
            or selected is None
        ):
            raise ValueError("input declaration mismatch")
        if not frame[0] <= selected[0] <= selected[1] <= frame[1]:
            continue
        row = facts.setdefault(entity, {})
        if name in row:
            raise ValueError("duplicate selected input")
        value = _wire_value(fact.get("value"))
        if not _value_matches_declaration(value, declaration):
            raise ValueError("input value type mismatch")
        row[name] = value
    constants = sc._constant_rule_environment(payload)
    replayed = {}
    reached_edges: dict[str, set[str]] = {}
    with localcontext(Context(prec=80, Emin=-999999, Emax=999999)):
        ordered_instances = sorted(
            observed, key=lambda item: (order.index(item[0]), item[1])
        )
        for name, entity in ordered_instances:
            actual = observed[(name, entity)]
            rule = rules[name]
            case = {"period": period, "input": facts.get(entity, {})}
            formula = sc._rule_formula_text_for_case(rule, case)
            if not isinstance(formula, str):
                raise ValueError("no selected formula")
            aggregate = re.fullmatch(
                r"sum\(([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\)",
                formula.strip(),
            )
            reached_names: set[str] = set()
            parameter_environment = {}
            if aggregate:
                relation, provider = aggregate.groups()
                declaration = relation_rules.get(relation, {}).get("data_relation", {})
                provider_rule = rules.get(provider, {})
                if (
                    declaration
                    != {
                        "arity": 2,
                        "arguments": [provider_rule.get("entity"), rule.get("entity")],
                    }
                    or provider_rule.get("unit") != rule.get("unit")
                    or provider_rule.get("dtype") != rule.get("dtype")
                ):
                    raise ValueError("unsupported relation slots/types")
                ids = _related_ids(
                    dataset.get("relations"),
                    relation=namespace + "#relation." + relation,
                    root=entity,
                    interval=interval,
                    current_slot=1,
                    related_slot=0,
                )
                if ids is None or any((provider, i) not in replayed for i in ids):
                    raise ValueError("provider closure missing")
                expected_dependencies = {
                    namespace
                    + "#"
                    + provider
                    + ("" if i == root else "@entity:" + i.encode().hex())
                    for i in ids
                }
                value = _checked_sum(replayed[(provider, i)] for i in ids)
                if value is None:
                    raise ValueError("unsupported sum")
            else:
                dependencies = {
                    n: v for (n, e), v in replayed.items() if e == entity and n != name
                }
                execution = sc._case_formula_execution(
                    rule,
                    case,
                    formula_environment=constants,
                    dependency_environment=dependencies,
                )
                if execution is None:
                    raise ValueError("unreplayable formula: " + name)
                environment = sc._case_formula_identifier_environment(
                    case,
                    formula_environment=constants,
                    dependency_environment=dependencies,
                )
                if environment is None:
                    raise ValueError("unresolved identifier environment")
                parameter_environment = execution.constant_environment
                for expression in (
                    execution.leaf,
                    *(
                        selector
                        for step in execution.trace
                        for selector in step.selectors
                    ),
                ):
                    names = sc._evaluated_formula_identifier_names(
                        expression, environment=environment
                    )
                    if names is None:
                        raise ValueError("unsupported selected expression")
                    reached_names.update(names)
                expected_dependencies = {
                    namespace
                    + "#"
                    + n
                    + ("" if entity == root else "@entity:" + entity.encode().hex())
                    for n in reached_names
                    if n in rules
                }
                value = sc._formula_execution_runtime_value(execution)
                if value is sc._UNRESOLVED_CONDITION_VALUE:
                    raise ValueError("unresolved formula: " + name)
            key = (
                namespace
                + "#"
                + name
                + ("" if entity == root else "@entity:" + entity.encode().hex())
            )
            _check_parameter_reads(
                trace[key],
                reached_names,
                parameters,
                parameter_environment,
                namespace,
                period["start"],
            )
            reported_dependencies = trace[key].get("dependencies", [])
            if (
                not isinstance(reported_dependencies, list)
                or any(not isinstance(ref, str) for ref in reported_dependencies)
                or len(set(reported_dependencies)) != len(reported_dependencies)
                or set(reported_dependencies) != expected_dependencies
                or any(ref not in trace for ref in expected_dependencies)
            ):
                raise ValueError("selected dependency trace mismatch: " + name)
            reached_edges[key] = expected_dependencies
            if not sc._formula_runtime_values_equal(value, actual):
                raise ValueError("replayed value mismatch: " + name)
            replayed[(name, entity)] = value
    reachable: set[str] = set()
    pending = list(outputs)
    while pending:
        key = pending.pop()
        if key not in reachable:
            reachable.add(key)
            pending.extend(reached_edges[key])
    if reachable != set(trace):
        raise ValueError("trace contains unreachable instances")
    return replayed


def local_relation_snapshot_eligible(payload: Any) -> bool:
    """Select ordinary snapshot recording only; grant no execution/source credit."""
    import ast

    if (
        not isinstance(payload, dict)
        or payload.get("format") != "rulespec/v1"
        or payload.get("imports", []) != []
    ):
        return False
    module = payload.get("module")
    if not isinstance(module, dict) or module.get("kind") == "composition":
        return False
    rules, inputs = payload.get("rules"), payload.get("inputs", [])
    if not isinstance(rules, list) or not isinstance(inputs, list):
        return False
    declarations = rules + inputs
    if any(
        not isinstance(r, dict) or not _valid_identity(r.get("name"))
        for r in declarations
    ):
        return False
    if len({r["name"] for r in declarations}) != len(declarations):
        return False
    by_name = {r["name"]: r for r in rules}
    for aggregate in rules:
        if (
            aggregate.get("kind") != "derived"
            or aggregate.get("dtype") not in ("Money", "Decimal")
            or aggregate.get("period") != "Year"
            or not _valid_identity(aggregate.get("unit"))
        ):
            continue
        versions = aggregate.get("versions")
        if not isinstance(versions, list) or not versions:
            continue
        accepted = True
        for version in versions:
            try:
                expression = ast.parse(version["formula"], mode="eval").body
            except (ValueError, TypeError, KeyError, SyntaxError):
                accepted = False
                break
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Name)
                and expression.func.id == "sum"
                and len(expression.args) == 1
                and not expression.keywords
                and isinstance(expression.args[0], ast.Attribute)
                and isinstance(expression.args[0].value, ast.Name)
            ):
                accepted = False
                break
            argument = expression.args[0]
            relation = by_name.get(argument.value.id, {})
            provider = by_name.get(argument.attr, {})
            slots = relation.get("data_relation", {})
            args = slots.get("arguments") if isinstance(slots, dict) else None
            if not (
                relation.get("kind") == "data_relation"
                and isinstance(slots, dict)
                and type(slots.get("arity")) is int
                and slots["arity"] == 2
                and isinstance(args, list)
                and len(args) == 2
                and all(_valid_identity(x) for x in args)
                and args[0] != args[1]
                and args == [provider.get("entity"), aggregate.get("entity")]
                and provider.get("kind") == "derived"
                and provider.get("period") == "Year"
                and provider.get("dtype") == aggregate.get("dtype")
                and provider.get("unit") == aggregate.get("unit")
            ):
                accepted = False
                break
        if accepted:
            return True
    return False
