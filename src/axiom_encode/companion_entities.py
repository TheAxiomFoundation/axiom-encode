"""Keep compact companion facts on their independently declared entities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


def declared_input_entities(
    module_file: Path,
    input_keys: Iterable[str],
    *,
    resolve_reference: Callable[[str], Path | None],
    resolve_import: Callable[[str, Path], Path | None],
) -> dict[str, str]:
    """Resolve facts from their source declarations, including imported facts.

    Compiled engine programs omit input declarations. Absolute keys may name
    a consuming module's input alias, so use that module's local declaration
    first, then an unambiguous declaration in its import closure. Legacy bare
    keys use the companion module as their scope. Resolvers are supplied by
    the caller so existing authorized-root checks remain authoritative.
    """
    payloads: dict[Path, dict[str, Any]] = {}

    def payload(path: Path) -> dict[str, Any]:
        if path not in payloads:
            data = yaml.safe_load(path.read_text()) if path.is_file() else {}
            payloads[path] = data if isinstance(data, dict) else {}
        return payloads[path]

    def declarations(path: Path) -> dict[str, str]:
        return {
            str(item["name"]): str(item["entity"])
            for item in payload(path).get("inputs", []) or []
            if isinstance(item, dict) and item.get("name") and item.get("entity")
        }

    entities: dict[str, str] = {}
    unresolved: dict[Path, dict[str, list[str]]] = {}
    for key in input_keys:
        key = str(key)
        if ":" in key and "#" in key:
            target = resolve_reference(key)
            if target is None:
                continue
            name = key.split("#", 1)[1].rsplit(".input.", 1)[-1]
            name = name.removeprefix("input.")
        else:
            target = module_file
            name = key
        entity = declarations(target).get(name)
        if entity is not None:
            entities[key] = entity
        else:
            unresolved.setdefault(target, {}).setdefault(name, []).append(key)

    for scope, keys_by_name in unresolved.items():
        candidates: dict[str, dict[Path, str]] = {name: {} for name in keys_by_name}
        visited = {scope}
        pending = [scope]
        while pending:
            current = pending.pop()
            for imported in payload(current).get("imports", []) or []:
                if not isinstance(imported, str):
                    continue
                target = resolve_import(imported, current)
                if target is None or target in visited:
                    continue
                visited.add(target)
                pending.append(target)
                for name, entity in declarations(target).items():
                    if name in candidates:
                        candidates[name][target] = entity
        for name, matches in candidates.items():
            if len(matches) > 1:
                raise ValueError(
                    f"input `{keys_by_name[name][0]}` has ambiguous imported "
                    "declarations; use its declaring module's absolute `#input` "
                    "reference"
                )
            if matches:
                entity = next(iter(matches.values()))
                entities.update({key: entity for key in keys_by_name[name]})
    return entities


def flat_input_entity_id(
    entity: str,
    query_entity: str,
    query_entity_id: str,
    *,
    reserved_entity_ids: Iterable[str] = (),
) -> str:
    """Retain owner IDs and avoid IDs occupied by other fixture entity kinds."""
    if entity == query_entity:
        return query_entity_id
    return _available_entity_id(
        f"{query_entity_id}:entity:{entity}", reserved_entity_ids
    )


def _available_entity_id(preferred: str, reserved_entity_ids: Iterable[str]) -> str:
    reserved = set(reserved_entity_ids)
    candidate = preferred
    suffix = 1
    while candidate in reserved:
        candidate = f"{preferred}:{suffix}"
        suffix += 1
    return candidate


def require_distinct_entity_ids(occupied_entity_ids: dict[str, set[str]]) -> None:
    """Reject explicit fixture IDs shared by different entity kinds."""
    for entity_id, entities in occupied_entity_ids.items():
        if len(entities) > 1:
            raise ValueError(
                f"fixture entity ID `{entity_id}` is shared by "
                f"{', '.join(f'`{entity}`' for entity in sorted(entities))}; "
                "use distinct IDs for different entity kinds"
            )


def assign_generated_fixture_ids(
    rows: Iterable[tuple[str, list[str], int, list[dict[str, Any]]]],
    occupied_entity_ids: dict[str, set[str]],
) -> None:
    """Allocate generated fixture rows after every explicit ID is reserved.

    Each row carries its entity, shared ID container, allocated slot and fact
    records. The container is a relation tuple or a table row's ID list, so
    updating it also updates relation aliases and row-ordered queries.
    Same-kind aliases keep their IDs; foreign-kind collisions get fresh IDs.
    """
    for entity, entity_ids, allocated_slot, inputs in rows:
        preferred = entity_ids[allocated_slot]
        reserved = {
            entity_id
            for entity_id, entities in occupied_entity_ids.items()
            if entities - {entity}
        }
        entity_id = (
            _available_entity_id(f"{preferred}:entity:{entity}", reserved)
            if preferred in reserved
            else preferred
        )
        entity_ids[allocated_slot] = entity_id
        for record in inputs:
            record["entity_id"] = entity_id
        occupied_entity_ids.setdefault(entity_id, set()).add(entity)


def cross_entity_input_findings(
    detail: str,
    case_input: dict[str, Any],
    input_entities: dict[str, str],
    output_entities: dict[str, str],
    declared_relation_slots: dict[str, tuple[str, ...]],
) -> list[str]:
    """Explain a missing fact without rejecting unused cross-entity facts.

    The engine remains the pass/fail authority. A relation declaration alone
    never justifies supplying a fact on the query entity or sharing its ID.
    """
    missing = re.search(
        r"\bmissing input\s+[`'\"]?([^`'\"\s,;]+)", detail, re.IGNORECASE
    )
    if missing is None:
        return []
    missing_key = missing.group(1)

    def input_identity(key: str) -> tuple[str, str]:
        module, separator, fragment = key.partition("#")
        if not separator:
            return "", key
        name = fragment.rsplit(".input.", 1)[-1].removeprefix("input.")
        return module, name

    findings = []
    for key, value in case_input.items():
        key = str(key)
        if isinstance(value, (dict, list)) or key not in input_entities:
            continue
        name = key.rsplit(".input.", 1)[-1].rsplit("#input.", 1)[-1]
        if missing_key != name and input_identity(missing_key) != input_identity(key):
            continue
        input_entity = input_entities[key]
        outputs_by_entity: dict[str, list[str]] = {}
        for output, output_entity in output_entities.items():
            outputs_by_entity.setdefault(output_entity, []).append(output)
        for output_entity, outputs in outputs_by_entity.items():
            if input_entity == output_entity:
                continue
            if any(
                input_entity in slots and output_entity in slots
                for slots in declared_relation_slots.values()
            ):
                continue
            findings.append(
                f"input `{key}` is declared on `{input_entity}` but is missing "
                f"when querying output(s) {', '.join(f'`{output}`' for output in outputs)} "
                f"on `{output_entity}` with no declared relation "
                "connecting these entities. Declare and traverse the appropriate "
                "relation, or correct the entity declaration."
            )
    return findings
