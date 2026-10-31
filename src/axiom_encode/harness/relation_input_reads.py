"""Follow factual input reads in native compiled relation and scalar expressions.

This inventory reads the compiler's IR. It neither evaluates formulas nor
reconstructs lowering from source names; unknown executable nodes fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping

# These child positions follow ScalarExprSpec and JudgmentExprSpec. Literal
# values and entity terms do not contain executable expressions.
_CHILDREN = {
    "literal": (),
    "input": (),
    "input_for_year": (),
    "input_or_else": (),
    "derived": (),
    "role_input": (),
    "role_derived": (),
    "parameter_lookup": ("index",),
    "add": ("items",),
    "sub": ("left", "right"),
    "mul": ("left", "right"),
    "div": ("left", "right"),
    "max": ("items",),
    "min": ("items",),
    "ceil": ("value",),
    "floor": ("value",),
    "period_start": (),
    "period_end": (),
    "date_add_days": ("date", "days"),
    "date_add_months": ("date", "months"),
    "date_add_years": ("date", "years"),
    "days_between": ("from", "to"),
    "only_related": ("value",),
    "count_related": ("where",),
    "sum_related": ("value", "where"),
    "extreme_related": ("value", "where"),
    "single_related": ("value", "where"),
    "if": ("condition", "then_expr", "else_expr"),
    "no_match": ("subject", "patterns"),
    "over_periods": ("value", "n"),
    "comparison": ("left", "right"),
    "relation_member": (),
    "relation_atom": (),
    "exists": ("body",),
    "same_entity": (),
    "and": ("items",),
    "or": ("items",),
    "not": ("item",),
    "exactly_one": ("items",),
}
_OPTIONAL = frozenset({"where", "n"})
_LISTS = frozenset({"items", "patterns"})
_RELATION_KINDS = frozenset(
    {
        "relation_member",
        "relation_atom",
        "exists",
        "only_related",
        "count_related",
        "sum_related",
        "extreme_related",
        "single_related",
    }
)


def compiled_relation_input_reads(
    payload: object, *, source_rule_ids: Mapping[str, str], relation_roots: set[str]
) -> tuple[dict[str, set[tuple[str, str]]], list[str]]:
    """Return each relation's transitive ``(owning module, input)`` reads.

    ``source_rule_ids`` supplies canonical owners when an older artifact omits
    derived-rule IDs. Ambiguous source names must be absent from that mapping.
    Any malformed or unknown executable node in a requested relation's
    dependency closure produces an analysis finding. Unrelated scalar rules
    do not become relation-validation failures.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("program"), dict):
        return {}, ["native artifact has no program mapping"]
    program = payload["program"]
    if not all(isinstance(program.get(key), list) for key in ("derived", "relations")):
        return {}, ["native program has no derived/relation inventories"]
    nodes: dict[str, tuple[str, list[object], set[str]]] = {}
    derived_names: dict[str, list[str]] = {}
    relation_ids: set[str] = set()
    node_issues: dict[str, list[str]] = {}
    for rule in program["derived"]:
        if not isinstance(rule, dict) or not isinstance(rule.get("name"), str):
            # A reachable reference to an absent/malformed declaration fails
            # resolution below. Unrelated scalar rules do not widen this check.
            continue
        name = rule["name"]
        identity = rule.get("id") or source_rule_ids.get(name)
        if not isinstance(identity, str) or "#" not in identity:
            continue
        issues = node_issues.setdefault(identity, [])
        expressions: list[object] = [rule.get("expr")]
        # DerivedSpec serializes empty versions by omitting the field. Its
        # top-level semantics/expr remains executable in formats 2 and 3.
        versions = rule.get("versions", [])
        if not isinstance(versions, list):
            issues.append(f"derived rule `{identity}` has no version inventory")
        else:
            expressions.extend(
                version.get("expr") if isinstance(version, dict) else None
                for version in versions
            )
        nodes[identity] = (identity.partition("#")[0], expressions, set())
        derived_names.setdefault(name, []).append(identity)
    for relation in program["relations"]:
        if not isinstance(relation, dict) or not isinstance(relation.get("name"), str):
            continue
        identity = relation["name"]
        issues = node_issues.setdefault(identity, [])
        relation_ids.add(identity)
        derivation = relation.get("derivation")
        if derivation is None:
            nodes[identity] = (identity.partition("#")[0], [], set())
            continue
        if not isinstance(derivation, dict):
            issues.append(f"relation `{identity}` has malformed derivation")
            continue
        sources: set[str] = set()
        source = derivation.get("source_relation")
        path = derivation.get("source_path")
        if isinstance(source, str) and path is None:
            sources.add(source)
        elif source is None and isinstance(path, list) and path:
            for step in path:
                if not isinstance(step, dict) or not isinstance(
                    step.get("relation"), str
                ):
                    issues.append(f"relation `{identity}` has malformed source path")
                else:
                    sources.add(step["relation"])
        else:
            issues.append(f"relation `{identity}` has no unique source")
        nodes[identity] = (
            identity.partition("#")[0],
            [derivation.get("predicate")],
            sources,
        )
    direct_inputs: dict[str, set[tuple[str, str]]] = {}
    dependencies: dict[str, set[str]] = {}
    for identity, (owner, expressions, sources) in nodes.items():
        issues = node_issues.setdefault(identity, [])
        reads: set[tuple[str, str]] = set()
        deps = set(sources)
        pending = list(expressions)
        while pending:
            expression = pending.pop()
            if (
                not isinstance(expression, dict)
                or not isinstance(expression.get("kind"), str)
                or expression["kind"] not in _CHILDREN
            ):
                issues.append(
                    f"`{identity}` has an unknown or malformed native expression"
                )
                continue
            kind = expression["kind"]
            if kind in {"input", "role_input", "input_or_else", "input_for_year"}:
                name = expression.get("input" if kind == "input_for_year" else "name")
                if isinstance(name, str) and name:
                    reads.add((owner, name))
                else:
                    issues.append(f"`{identity}` has a malformed native input read")
            elif kind in {"derived", "role_derived"}:
                name = expression.get("name")
                candidates = (
                    [name]
                    if isinstance(name, str) and name in nodes
                    else derived_names.get(name, [])
                    if isinstance(name, str)
                    else []
                )
                if len(candidates) == 1:
                    deps.add(candidates[0])
                else:
                    issues.append(
                        f"`{identity}` has an unresolved native derived read `{name}`"
                    )
            if kind in _RELATION_KINDS:
                relation = expression.get("relation")
                if isinstance(relation, str) and relation in relation_ids:
                    deps.add(relation)
                else:
                    issues.append(
                        f"`{identity}` has an unresolved native relation read"
                    )
            for field in _CHILDREN[kind]:
                child = expression.get(field)
                if field in _OPTIONAL and child is None:
                    continue
                if field in _LISTS:
                    if isinstance(child, list) and child:
                        pending.extend(child)
                    else:
                        issues.append(
                            f"`{identity}` has malformed native `{field}` children"
                        )
                else:
                    pending.append(child)
        direct_inputs[identity] = reads
        dependencies[identity] = deps
        if deps - nodes.keys():
            issues.append(f"`{identity}` has an unresolved native source dependency")
    result: dict[str, set[tuple[str, str]]] = {}
    issues = []
    for identity in sorted(relation_roots):
        if identity not in relation_ids:
            issues.append(f"relation `{identity}` has no native declaration")
            continue
        inputs: set[tuple[str, str]] = set()
        seen: set[str] = set()
        pending = [identity]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            issues.extend(node_issues.get(node, []))
            inputs.update(direct_inputs.get(node, set()))
            pending.extend(dependencies.get(node, set()) - seen)
        result[identity] = inputs
    return result, list(dict.fromkeys(issues))
