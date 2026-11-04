"""Resolve companion relation direction from reachable executable expressions."""

from __future__ import annotations

from typing import Any


def executable_relation_directions(
    derived: dict[str, dict[str, Any]],
    outputs: list[str],
    period: dict[str, Any],
    query_entity: str,
    declared: dict[str, tuple[str, ...]],
    relations: list[dict[str, Any]] | None = None,
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
            source = derivation["source_relation"]
            slots = derivation.get("slot_entities") or []
            current_kind = slots[current] if len(slots) == 2 else None
            related_kind = slots[related] if len(slots) == 2 else None
            if root:
                record(source, (current, related_kind))
            relation_derivation(source, entity, root)
            # Only derived-relation predicates install RelationEvalContext.
            context = (current_kind, root, related_kind, False)
            walk(derivation["predicate"], related_kind or entity, False, context)
        finally:
            active_relations.remove(marker)

    def record(
        relation: str, direction: tuple[int, str | None], *, explicit: bool = True
    ) -> None:
        current, hint = direction
        if relation in directions and directions[relation] != current:
            raise ValueError(f"conflicting executable directions for {relation}")
        directions[relation] = current
        if hint is not None:
            hints = explicit_hints if explicit else fallback_hints
            hints.setdefault(relation, set()).add(hint)

    def walk(node: Any, entity: str, root: bool, context: tuple | None = None) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, entity, root, context)
            return
        if not isinstance(node, dict):
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
            walk(value, entity, root, context)

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
