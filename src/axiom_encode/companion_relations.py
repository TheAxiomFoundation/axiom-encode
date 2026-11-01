"""Resolve companion relation direction from reachable executable expressions."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Any

_RELATION_ROLE_KINDS = frozenset(
    {"role_input", "role_derived", "relation_atom", "exists", "same_entity"}
)


def relation_role_query_entity_id(case: dict[str, Any], query_entity: str) -> str:
    """Use one root identity for relation-role companion requests."""
    entity_key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", query_entity)
    entity_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", entity_key)
    entity_key = f"{entity_key.replace('-', '_').lower()}_id"
    for key in ("entity_id", "id", entity_key):
        if key in case:
            return str(case[key])
    for key, value in case.items():
        if str(key).endswith("_id") and not isinstance(value, (dict, list)):
            return str(value)
    return "case"


def related_entity_from_relation(relation_name: str) -> str:
    """Infer a readable related-entity label for relation test inputs."""
    head = relation_name.split("_of_", 1)[0]
    return "".join(part.capitalize() for part in head.split("_") if part) or "Related"


def compact_relation_row(
    row: dict[str, Any], *, fallback_id: str
) -> tuple[str, bool, dict[str, Any]]:
    """Separate a compact member's identity from its factual inputs.

    The first truthy ID names the member, and a row without one takes
    `fallback_id`, as main's pipeline runner read compact rows.
    """
    entity_id = row.get("id") or row.get("entity_id")
    return (
        str(entity_id or fallback_id),
        bool(entity_id),
        {key: value for key, value in row.items() if key not in {"id", "entity_id"}},
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


def _slot_kinds(
    relation: dict[str, Any], relations_by_name: dict[Any, dict[str, Any]]
) -> list[Any]:
    """Read a relation's slot kinds, following a source that declares them."""
    seen: set[int] = set()
    current: dict[str, Any] | None = relation
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        derivation = current.get("derivation")
        if not isinstance(derivation, dict):
            return current.get("slot_entities") or []
        declared = derivation.get("slot_entities") or current.get("slot_entities")
        if declared or derivation.get("source_path"):
            return declared or []
        current = relations_by_name.get(derivation.get("source_relation"))
    return []


def filtered_entity_kinds(relations: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map each filtered entity to the innermost kind of its IDs, as the engine does.

    The engine queries a filtered entity with its source's current-slot IDs
    and canonicalizes every filtered entity before it binds a dataset. A chain
    of filters ends at its innermost kind. An entity whose relations disagree
    or whose chain cycles stays unmapped. Legacy programs map nothing, which
    keeps their existing request bytes.
    """
    if not uses_relation_roles(relations):
        return {}
    relations_by_name = {relation.get("name"): relation for relation in relations or []}
    current_kinds: dict[str, set[str]] = {}
    for relation in relations or []:
        derivation = relation.get("derivation")
        if not isinstance(derivation, dict) or not derivation.get("entity"):
            continue
        slots = _slot_kinds(relation, relations_by_name)
        current = derivation.get("current_slot")
        if type(current) is int and current < len(slots):
            current_kinds.setdefault(str(derivation["entity"]), set()).add(
                str(slots[current])
            )
    direct = {
        entity: next(iter(kinds))
        for entity, kinds in current_kinds.items()
        if len(kinds) == 1
    }
    innermost: dict[str, str] = {}
    for entity, kind in direct.items():
        visited = {entity}
        while direct.get(kind, kind) != kind and kind not in visited:
            visited.add(kind)
            kind = direct[kind]
        if kind not in visited:
            innermost[entity] = kind
    return innermost


def query_entity_kinds(
    query_entity: str, entity_kinds: Mapping[str, str]
) -> frozenset[str]:
    """Name each entity kind that a companion query's root ID carries."""
    root = entity_kinds.get(query_entity, query_entity)
    return frozenset(
        {query_entity, root}
        | {entity for entity, kind in entity_kinds.items() if kind == root}
    )


def _relation_local_name(name: str) -> str:
    return name.rsplit("#relation.", 1)[-1]


def _relations_read_by(
    outputs: dict[Any, Any],
    derived: Mapping[str, dict[str, Any]],
    relations: list[dict[str, Any]],
) -> set[str] | None:
    """Name each relation that a case's outputs can read.

    Every string in a reached rule or relation counts as a reference, so this
    can over-read a relation but never misses one. A full compiled relation
    name reads that relation alone. A filtered entity's rules reach its
    relation by the entity's name. Return None for a case with no outputs, or
    with an output that has no compiled rule.
    """
    pending: list[Any] = [derived.get(str(output)) for output in outputs]
    if not pending or any(rule is None for rule in pending):
        return None

    targets: dict[str, list[dict[str, Any]]] = {}
    qualified = {
        str(relation.get("name")): relation
        for relation in relations
        if "#relation." in str(relation.get("name"))
    }

    def add(alias: Any, target: dict[str, Any]) -> None:
        if alias:
            targets.setdefault(_relation_local_name(str(alias)), []).append(target)

    for rule in derived.values():
        add(rule.get("id"), rule)
        add(rule.get("name"), rule)
    for relation in relations:
        add(relation.get("name"), relation)
        derivation = relation.get("derivation")
        if isinstance(derivation, dict):
            add(derivation.get("entity"), relation)
            add(derivation.get("member_relation"), relation)

    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            if value in qualified:
                pending.append(qualified[value])
            else:
                pending.extend(targets.get(_relation_local_name(value), ()))
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict) and id(value) not in seen:
            seen.add(id(value))
            pending.extend(value.values())
    return {str(relation.get("name")) for relation in relations if id(relation) in seen}


def pair_relation_assignment_issues(
    relations: list[dict[str, Any]] | None,
    case: dict[str, Any],
    case_name: str,
    derived: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Require explicit pair facts, including empty sets, for role programs.

    The current engine treats an absent tuple as false and exposes no relation
    completeness declaration. Assigning every base pair relation that a case's
    outputs can read makes that closed-world choice explicit in each companion
    case. A relation no output reaches cannot change the case, so it needs no
    assignment. Without `derived`, or without a rule for every output, each
    base pair relation counts.
    """
    try:
        relation_role_program = uses_relation_roles(relations)
    except RecursionError:
        return [
            "[relation-input-analysis] Cannot inventory native relation roles: "
            "expression nesting exceeds Python analysis capacity."
        ]
    if not relation_role_program:
        return []
    inputs = case.get("input")
    assigned = set(map(str, inputs)) if isinstance(inputs, dict) else set()
    outputs = case.get("output")
    reachable = (
        _relations_read_by(outputs, derived, relations or [])
        if derived is not None and isinstance(outputs, dict)
        else None
    )
    declared = {str(relation.get("name")) for relation in relations or []}
    typed = {
        str(relation.get("name"))
        for relation in relations or []
        if relation.get("slot_entities")
    }
    issues = []
    for relation in relations or []:
        if relation.get("derivation") or relation.get("arity") != 2:
            continue
        name = str(relation.get("name") or "")
        if not name or (reachable is not None and name not in reachable):
            continue
        aliases = {name}
        if "#relation." not in name:
            aliases.update(
                key
                for key in assigned
                if key.endswith(f"#relation.{name}")
                and reaches_short_relation(key, name, declared=declared, typed=typed)
            )
        if not aliases & assigned:
            issues.append(
                f"Test case `{case_name}` must assign pair relation `{name}`, "
                "including `[]` when no tuples hold."
            )
    return issues


def reaches_short_relation(
    key: str,
    short_name: str,
    *,
    declared: Collection[str],
    typed: Collection[str],
) -> bool:
    """Tell whether a qualified relation key also feeds its short name.

    An untyped short name stands for the key's own relation. A typed one is a
    different relation whenever the key names a compiled relation itself.
    """
    return key not in declared or short_name not in typed


def reached_relation_names(
    names: list[str],
    *,
    declared: Collection[str],
    typed: Collection[str],
) -> list[str]:
    """Keep the compiled relations that one supplied key feeds."""
    key, *short_names = names
    return [
        key,
        *(
            short_name
            for short_name in short_names
            if reaches_short_relation(key, short_name, declared=declared, typed=typed)
        ),
    ]


def is_tuple_row(row: Any, *, relation_roles: bool) -> bool:
    """Tell whether a companion row takes the explicit tuple form.

    A legacy compact row may name a scalar fact `tuple`.
    """
    return (
        isinstance(row, dict)
        and "tuple" in row
        and (relation_roles or isinstance(row["tuple"], list))
    )


def explicit_relation_tuples(
    rows: list[Any],
    *,
    relation: str,
    slots: tuple[str, ...] | None,
    relation_roles: bool,
) -> list[list[str]] | None:
    """Read tuple-only companion rows in the relation's declared slot order.

    Return None for the existing compact relation-member mappings. Named table
    rows supply facts for the ids in explicit tuples; the engine checks kinds.
    """
    if not any(is_tuple_row(row, relation_roles=relation_roles) for row in rows):
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


def explicit_tuple_relation_names(case_input: Any, *, relation_roles: bool) -> set[str]:
    """Find supplied relations whose rows need no implicit anchor direction."""
    if not isinstance(case_input, dict):
        return set()
    return {
        str(name)
        for name, rows in case_input.items()
        if isinstance(rows, list)
        and (
            (relation_roles and not rows)
            or any(is_tuple_row(row, relation_roles=relation_roles) for row in rows)
        )
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
            and relation not in declared
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
