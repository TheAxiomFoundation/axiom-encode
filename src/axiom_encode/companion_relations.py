"""Resolve companion relation direction from reachable executable expressions."""

from __future__ import annotations

from typing import Any

_RELATION_ROLE_KINDS = frozenset(
    {"role_input", "role_derived", "relation_atom", "exists", "same_entity"}
)


def uses_relation_roles(relations: list[dict[str, Any]] | None) -> bool:
    """Recognize the engine's relation-role IR, including path sources."""

    def walk(value: Any) -> bool:
        if isinstance(value, list):
            return any(walk(item) for item in value)
        if not isinstance(value, dict):
            return False
        derivation = value.get("derivation")
        # Constant anchored views need the same pair-fact contract even when
        # their predicates lower to ordinary IR. The native artifact retains
        # an entity field for a legacy filtered entity, with or without an alias.
        anchored = isinstance(derivation, dict) and not derivation.get("entity")
        return (
            anchored
            or value.get("kind") in _RELATION_ROLE_KINDS
            or "source_path" in value
            or any(walk(item) for item in value.values())
        )

    return walk(relations)


def pair_relation_assignment_issues(
    relations: list[dict[str, Any]] | None,
    case: dict[str, Any],
    case_name: str,
) -> list[str]:
    """Require explicit pair facts, including empty sets, for role programs.

    The current engine treats an absent tuple as false and exposes no relation
    completeness declaration. Assigning every base pair relation makes that
    closed-world choice explicit in each companion case.
    """
    if not uses_relation_roles(relations):
        return []
    inputs = case.get("input")
    assigned = set(map(str, inputs)) if isinstance(inputs, dict) else set()
    issues = []
    for relation in relations or []:
        if relation.get("derivation") or relation.get("arity") != 2:
            continue
        name = str(relation.get("name") or "")
        if not name:
            continue
        aliases = {name}
        if "#relation." not in name:
            aliases.update(key for key in assigned if key.endswith(f"#relation.{name}"))
        if not aliases & assigned:
            issues.append(
                f"Test case `{case_name}` must assign pair relation `{name}`, "
                "including `[]` when no tuples hold."
            )
    return issues


def explicit_relation_tuples(
    rows: list[Any],
    *,
    relation: str,
    slots: tuple[str, ...] | None,
) -> list[list[str]] | None:
    """Read tuple-only companion rows in the relation's declared slot order.

    Return None for the existing compact relation-member mappings. Named table
    rows supply facts for the ids in explicit tuples; the engine checks kinds.
    """
    if not any(isinstance(row, dict) and "tuple" in row for row in rows):
        return None
    if slots is None:
        raise ValueError(
            f"relation `{relation}` tuple rows require declared slot kinds"
        )
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != {"tuple"}:
            raise ValueError(
                f"relation `{relation}` tuple row #{index} must contain only `tuple`; "
                "supply entity facts in named table rows"
            )
        value = row["tuple"]
        if (
            not isinstance(value, list)
            or len(value) != len(slots)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError(
                f"relation `{relation}` tuple row #{index} must contain "
                f"{len(slots)} string ids in declared slot order"
            )
        result.append(list(value))
    return result


def explicit_tuple_relation_names(case_input: Any) -> set[str]:
    """Find supplied relations whose rows need no implicit anchor direction."""
    if not isinstance(case_input, dict):
        return set()
    return {
        str(name)
        for name, rows in case_input.items()
        if isinstance(rows, list)
        and (not rows or any(isinstance(row, dict) and "tuple" in row for row in rows))
    }


def executable_relation_directions(
    derived: dict[str, dict[str, Any]],
    outputs: list[str],
    period: dict[str, Any],
    query_entity: str,
    declared: dict[str, tuple[str, ...]],
    relations: list[dict[str, Any]] | None = None,
    *,
    explicit_relations: set[str] | None = None,
) -> dict[str, tuple[int, str | None]]:
    """Return current slot and related entity, rejecting ambiguous evidence.

    Declarations identify entity kinds; executable expressions own direction.
    Only versions active at period start and reachable outputs count.
    """
    aliases: dict[str, list[dict[str, Any]]] = {}
    for key, rule in derived.items():
        for alias in {key, str(rule.get("id", "")), str(rule.get("name", ""))} - {""}:
            entries = aliases.setdefault(alias, [])
            if not any(entry is rule for entry in entries):
                entries.append(rule)
    directions: dict[str, int] = {}
    explicit_hints: dict[str, set[str]] = {}
    fallback_hints: dict[str, set[str]] = {}
    visited: set[tuple[int, str, bool]] = set()
    schemas = {str(r["name"]): r for r in relations or [] if r.get("name")}
    active_relations: set[tuple[str, bool]] = set()
    explicit_relations = explicit_relations or set()

    def resolve(name: str) -> dict[str, Any]:
        matches = aliases.get(name, [])
        if len(matches) != 1:
            raise ValueError(
                f"ambiguous or missing compiled derived reference {name!r}"
            )
        return matches[0]

    def expressions(rule: dict[str, Any]) -> list[Any]:
        versions = rule.get("versions")
        if not versions:
            return [rule.get("expr")]
        active = [
            v
            for v in versions
            if str(v.get("effective_from", "0001-01-01")) <= str(period["start"])
            and str(v.get("effective_to") or "9999-12-31") >= str(period["start"])
        ]
        if not active:
            return []
        latest = max(str(v.get("effective_from", "0001-01-01")) for v in active)
        selected = [
            v for v in active if str(v.get("effective_from", "0001-01-01")) == latest
        ]
        if len(selected) != 1:
            raise ValueError("ambiguous effective compiled expression")
        return [selected[0].get("expr")]

    def visit_rule(rule: dict[str, Any], entity: str, root: bool) -> None:
        # Ordinary derived calls retain entity_id even across declared kinds.
        # A derived body is evaluated without its caller's relation context.
        hint = str(rule.get("entity") or entity)
        if hint == "Scalar":
            hint = entity
        marker = (id(rule), hint, root)
        if marker in visited:
            return
        visited.add(marker)
        for expr in expressions(rule):
            walk(expr, hint, root)

    def relation_derivation(relation: str, entity: str, root: bool) -> None:
        schema = schemas.get(relation)
        if schema is None and "#relation." in relation:
            schema = schemas.get(relation.rsplit("#relation.", 1)[1])
        derivation = (schema or {}).get("derivation")
        if not derivation:
            return
        marker = (relation, root)
        if marker in active_relations:
            raise ValueError("cyclic compiled relation derivation")
        active_relations.add(marker)
        try:
            current, related = (
                derivation.get("current_slot"),
                derivation.get("related_slot"),
            )
            if (
                type(current) is not int
                or type(related) is not int
                or {current, related} != {0, 1}
            ):
                raise ValueError("invalid executable binary relation coordinates")
            slots = derivation.get("slot_entities") or []
            current_kind = slots[current] if len(slots) == 2 else None
            related_kind = slots[related] if len(slots) == 2 else None
            source = derivation.get("source_relation")
            path = derivation.get("source_path")
            if (source is None) == (path is None):
                raise ValueError("compiled relation needs exactly one source")
            if source is not None:
                if root:
                    record(source, (current, related_kind))
                relation_derivation(source, entity, root)
            else:
                if not isinstance(path, list) or not path:
                    raise ValueError("compiled relation source path must have steps")
                for index, step in enumerate(path):
                    step_relation = step.get("relation")
                    start, end = step.get("from_slot"), step.get("to_slot")
                    if (
                        not isinstance(step_relation, str)
                        or type(start) is not int
                        or type(end) is not int
                        or {start, end} != {0, 1}
                    ):
                        raise ValueError(
                            "invalid executable binary relation path coordinates"
                        )
                    step_slots = declared.get(step_relation) or ()
                    step_kind = step_slots[end] if len(step_slots) == 2 else None
                    step_root = root and index == 0
                    if step_root:
                        record(step_relation, (start, step_kind))
                    relation_derivation(step_relation, entity, step_root)
            # Only derived-relation predicates install RelationEvalContext.
            context = (current_kind, root, related_kind, False)
            walk(derivation["predicate"], related_kind or entity, False, context)
        finally:
            active_relations.remove(marker)

    def record(
        relation: str, direction: tuple[int, str | None], *, explicit: bool = True
    ) -> None:
        if relation in explicit_relations or (
            "#relation." not in relation
            and any(
                name.endswith(f"#relation.{relation}") for name in explicit_relations
            )
        ):
            return
        current, hint = direction
        if relation in directions and directions[relation] != current:
            raise ValueError(f"conflicting executable directions for {relation}")
        directions[relation] = current
        if hint is not None:
            hints = explicit_hints if explicit else fallback_hints
            hints.setdefault(relation, set()).add(hint)

    def term_target(
        term: Any, context: tuple | None, variables: tuple
    ) -> tuple[str | None, bool]:
        if context is None:
            raise ValueError("relation role requires derived relation context")
        if term == "anchor":
            return context[0], context[1]
        if term == "candidate":
            return context[2], context[3]
        if isinstance(term, dict) and type(term.get("var")) is int:
            index = term["var"]
            if 0 <= index < len(variables):
                return variables[-1 - index]
        raise ValueError("invalid or unbound compiled relation term")

    def walk(
        node: Any,
        entity: str,
        root: bool,
        context: tuple | None = None,
        variables: tuple = (),
    ) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, entity, root, context, variables)
            return
        if not isinstance(node, dict):
            return
        if node.get("kind") == "role_derived":
            kind, target_root = term_target(node.get("term"), context, variables)
            visit_rule(resolve(str(node.get("name", ""))), kind or entity, target_root)
            return
        if node.get("kind") in {"relation_atom", "exists"}:
            relation = str(node.get("relation", ""))
            schema = (
                schemas.get(relation)
                or schemas.get(relation.rsplit("#relation.", 1)[-1])
                or {}
            )
            derivation = schema.get("derivation")
            terms = node.get("terms") or []
            if derivation:
                current = derivation.get("current_slot")
                if type(current) is not int or not 0 <= current < len(terms):
                    raise ValueError("invalid executable relation atom coordinates")
                kind, target_root = term_target(terms[current], context, variables)
                relation_derivation(relation, kind or entity, target_root)
            else:
                slots = schema.get("slot_entities") or declared.get(relation) or ()
                variable_slot = (
                    node.get("variable_slot") if node.get("kind") == "exists" else None
                )
                if len(slots) == 2 and len(terms) == 2:
                    for index, term in enumerate(terms):
                        if index == variable_slot:
                            continue
                        _kind, target_root = term_target(term, context, variables)
                        if target_root:
                            record(relation, (index, slots[1 - index]))
            if node.get("kind") == "exists":
                variable_slot = node.get("variable_slot")
                slots = schema.get("slot_entities") or declared.get(relation) or ()
                if type(variable_slot) is not int or not 0 <= variable_slot < len(
                    slots
                ):
                    raise ValueError("invalid executable relation variable slot")
                walk(
                    node.get("body"),
                    entity,
                    root,
                    context,
                    (*variables, (slots[variable_slot], False)),
                )
            return
        if node.get("kind") == "derived":
            rule = resolve(str(node.get("name", "")))
            target_root = root
            if context:
                current_kind, current_root, related_kind, related_root = context
                if rule.get("entity") == current_kind and current_kind is not None:
                    target_root = current_root
                elif rule.get("entity") == related_kind and related_kind is not None:
                    target_root = related_root
            visit_rule(rule, entity, target_root)
            return
        if "current_slot" in node or "related_slot" in node:
            relation = node.get("relation")
            current, related = node.get("current_slot"), node.get("related_slot")
            if (
                not isinstance(relation, str)
                or not relation
                or type(current) is not int
                or type(related) is not int
                or {current, related} != {0, 1}
            ):
                raise ValueError("invalid executable binary relation coordinates")
            if node.get("kind") == "relation_member":
                if context is None:
                    raise ValueError(
                        "relation member requires derived relation context"
                    )
                current_kind, current_root, related_kind, _ = context
                if current_root:
                    record(relation, (current, related_kind))
                relation_derivation(relation, current_kind or entity, current_root)
                return
            slots = declared.get(relation)
            if slots is None and "#relation." in relation:
                slots = declared.get(relation.rsplit("#relation.", 1)[1])
            child_entities: set[str] = set()

            def child_refs(value: Any) -> None:
                if isinstance(value, list):
                    for item in value:
                        child_refs(item)
                elif isinstance(value, dict):
                    if "current_slot" in value or "related_slot" in value:
                        return  # Nested aggregations establish their own context.
                    if value.get("kind") == "derived":
                        child = resolve(str(value.get("name", "")))
                        if child.get("entity") == "Scalar":
                            return
                        child_entities.add(
                            str(
                                resolve(str(value.get("name", ""))).get("entity")
                                or entity
                            )
                        )
                    else:
                        for item in value.values():
                            child_refs(item)

            for key, value in node.items():
                if key not in {"current_slot", "related_slot", "relation", "kind"}:
                    child_refs(value)
            # Sum value and predicate execute on the same related ID even
            # when their ordinary derived annotations differ. Mixed kinds are
            # unavailable typing evidence, not a coordinate conflict.
            if len(child_entities) == 1:
                other = next(iter(child_entities))
            elif slots and len(slots) == 2 and entity in slots:
                other = slots[1 - slots.index(entity)]
            else:
                other = None
            if root:
                record(relation, (current, other), explicit=len(child_entities) == 1)
            relation_derivation(relation, entity, root)
            for key, value in node.items():
                if key not in {"current_slot", "related_slot", "relation", "kind"}:
                    walk(value, other or "Entity", False)
            return
        for value in node.values():
            walk(value, entity, root, context, variables)

    for output in outputs:
        if output in aliases:
            visit_rule(resolve(output), query_entity, True)
    found: dict[str, tuple[int, str | None]] = {}
    for relation, current in directions.items():
        # A bare count supplies no child type evidence. Delay its declaration
        # fallback until the entire reachable closure has supplied explicit hints.
        hints = explicit_hints.get(relation) or fallback_hints.get(relation, set())
        if len(hints) > 1:
            raise ValueError(f"conflicting related entities for {relation}")
        found[relation] = (current, next(iter(hints)) if hints else None)
    return found
