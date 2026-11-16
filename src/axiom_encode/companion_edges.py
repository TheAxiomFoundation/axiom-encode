"""Explicit, typed companion edges over declared fixture identities.

Presence of ``relations`` opts a case into an explicit ``entity_id`` root.
Endpoints may reference that root or explicitly identified table rows only.
Legacy compact relation rows and generated IDs are not addressable here.
"""

from __future__ import annotations

import re
from typing import Any

from axiom_encode.companion_entities import require_distinct_entity_ids


def explicit_relation_root(case: dict[str, Any]) -> str:
    """Validate the opt-in carrier, including an empty relation list."""
    if not isinstance(case.get("relations"), list):
        raise ValueError("relations must be a list")
    root = case.get("entity_id")
    if not isinstance(root, str) or not root.strip():
        raise ValueError("relations requires a nonempty string case.entity_id")
    return root


def explicit_companion_relations(
    edges: Any,
    *,
    tables: Any,
    root_id: str,
    root_entity: str,
    compiled_relations: list[dict] | None,
    interval: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate canonical data-relation tuples without synthesizing identities."""
    explicit_relation_root({"relations": edges, "entity_id": root_id})
    identities = {root_id: {root_entity}}
    if tables is not None:
        if not isinstance(tables, dict):
            raise ValueError("tables must be a mapping")
        for entity, rows in tables.items():
            if not isinstance(entity, str) or not isinstance(rows, list):
                raise ValueError(
                    "relations requires named table entities and row lists"
                )
            snake = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", entity)
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake)
            snake = snake.replace("-", "_").lower()
            keys = ("id", "entity_id", f"{snake}_id")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("table rows must be mappings")
                values = [row[key] for key in keys if key in row]
                if not values:
                    continue
                if any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError("explicit table IDs must be nonempty strings")
                if len(set(values)) != 1:
                    raise ValueError("conflicting explicit table IDs")
                identities.setdefault(values[0], set()).add(entity)
    require_distinct_entity_ids(identities)
    catalog = {
        relation.get("name"): relation
        for relation in compiled_relations or []
        if isinstance(relation, dict)
    }
    result = []
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {"name", "tuple"}:
            raise ValueError("relations entries require exactly name and tuple")
        name = edge["name"]
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z][a-z0-9-]*:[^\s#]+#relation\.[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError("relations name must be a canonical relation reference")
        declaration = catalog.get(name)
        if declaration is None or "derivation" in declaration:
            raise ValueError(f"relation `{name}` is not a declared data_relation")
        slots = declaration.get("slot_entities")
        arity = declaration.get("arity")
        if (
            type(arity) is not int
            or arity < 1
            or not isinstance(slots, list)
            or len(slots) != arity
            or not all(isinstance(slot, str) and slot for slot in slots)
        ):
            raise ValueError(f"relation `{name}` requires explicit typed slots")
        endpoints = edge["tuple"]
        if not isinstance(endpoints, list) or len(endpoints) != arity:
            raise ValueError(f"relation `{name}` tuple must have arity {arity}")
        for endpoint, entity in zip(endpoints, slots):
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ValueError("relation endpoint IDs must be nonempty strings")
            if endpoint not in identities:
                raise ValueError(
                    f"relation endpoint `{endpoint}` has no explicit identity"
                )
            if identities[endpoint] != {entity}:
                raise ValueError(
                    f"relation endpoint `{endpoint}` must have entity `{entity}`"
                )
        result.append(
            {"name": name, "tuple": list(endpoints), "interval": dict(interval)}
        )
    return result
