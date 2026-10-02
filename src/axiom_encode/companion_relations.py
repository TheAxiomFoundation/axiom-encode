"""Resolve companion relation direction from reachable executable expressions."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


def executable_relation_directions(
    derived: dict[str, dict[str, Any]],
    outputs: list[str],
    period: dict[str, Any],
    query_entity: str,
    declared: dict[str, tuple[str, ...]],
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
    found: dict[str, tuple[int, str | None]] = {}
    visited: set[tuple[int, str | None]] = set()

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

    def independent_scalar(
        rule: dict[str, Any], seen: frozenset[int] = frozenset()
    ) -> bool:
        """Prove a compiled Scalar helper cannot depend on an entity or relation."""
        if rule.get("entity") != "Scalar" or id(rule) in seen:
            return False
        seen = seen | {id(rule)}

        def pure(node: Any) -> bool:
            if not isinstance(node, dict):
                return False
            kind = node.get("kind")
            if kind == "literal":
                value = node.get("value")
                if set(node) != {"kind", "value"} or not isinstance(value, dict):
                    return False
                if set(value) != {"kind", "value"}:
                    return False
                number = value["value"]
                if value["kind"] == "integer":
                    return type(number) is int and -(2**63) <= number < 2**63
                if value["kind"] == "decimal" and type(number) in (str, int):
                    try:
                        return Decimal(number).is_finite()
                    except InvalidOperation:
                        return False
                return False
            if kind == "derived":
                return set(node) <= {"kind", "name"} and independent_scalar(
                    resolve(str(node.get("name", ""))), seen
                )
            if kind == "add":
                items = node.get("items")
                return (
                    set(node) == {"kind", "items"}
                    and isinstance(items, list)
                    and bool(items)
                    and all(pure(item) for item in items)
                )
            if kind in {"sub", "mul", "div"}:
                return set(node) == {"kind", "left", "right"} and all(
                    pure(node[key]) for key in ("left", "right")
                )
            return False

        selected = expressions(rule)
        return bool(selected) and all(pure(expr) for expr in selected)

    def visit_rule(rule: dict[str, Any], entity: str) -> None:
        if independent_scalar(rule):
            return  # Scalar constants retain the surrounding entity context.
        if rule.get("entity") and rule["entity"] != entity:
            raise ValueError(
                "compiled derived reference changes entity outside aggregation"
            )
        marker = (id(rule), entity)
        if marker in visited:
            return
        visited.add(marker)
        for expr in expressions(rule):
            walk(expr, entity)

    def walk(node: Any, entity: str) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, entity)
            return
        if not isinstance(node, dict):
            return
        if node.get("kind") == "derived":
            visit_rule(resolve(str(node.get("name", ""))), entity)
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
                        if independent_scalar(child):
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
            if len(child_entities) > 1:
                raise ValueError(f"conflicting related entities for {relation}")
            if child_entities:
                other = next(iter(child_entities))
                if slots and sorted(slots) != sorted((entity, other)):
                    raise ValueError(
                        f"entity evidence conflicts with declaration for {relation}"
                    )
            elif slots and len(slots) == 2 and entity in slots:
                other = slots[1 - slots.index(entity)]
            else:
                other = None
            if entity == query_entity:
                direction = (current, other)
                if relation in found and found[relation] != direction:
                    raise ValueError(
                        f"conflicting executable directions for {relation}"
                    )
                found[relation] = direction
            for key, value in node.items():
                if key not in {"current_slot", "related_slot", "relation", "kind"}:
                    walk(value, other or "Entity")
            return
        for value in node.values():
            walk(value, entity)

    for output in outputs:
        if output in aliases:
            visit_rule(resolve(output), query_entity)
    return found
