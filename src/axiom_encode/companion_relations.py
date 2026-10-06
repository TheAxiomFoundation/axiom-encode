"""Resolve companion row placement from the artifact that will execute it."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class RelationLayout:
    current_slot: int
    current_entity: str
    related_entity: str

    def tuple(self, current_id: str, related_id: str) -> list[str]:
        result = [related_id, related_id]
        result[self.current_slot] = current_id
        return result


LEGACY_LAYOUT = RelationLayout(1, "Entity", "Entity")


class CompanionRelations:
    """Executable usage is authoritative; declarations describe unused relations.

    Keep requests for artifacts without typed relations byte-for-byte compatible
    with the original companion producer, including its generic entity labels.
    """

    def __init__(self, program: dict):
        self.relations = {item["name"]: item for item in program.get("relations", [])}
        self.derived = {
            key: item
            for item in program.get("derived", [])
            for key in (item.get("id"), item.get("name"))
            if key
        }
        self.typed = any(item.get("slot_entities") for item in self.relations.values())
        self.usages: dict[str, list[tuple[int, tuple[str | None, ...]]]] = defaultdict(
            list
        )
        if not self.typed:
            return
        for rule in program.get("derived", []):
            for version in rule.get("versions") or [rule]:
                self._walk(version.get("expr"), rule.get("entity"))
        for schema in self.relations.values():
            if derivation := schema.get("derivation"):
                kinds = derivation.get("slot_entities", [])
                current_slot, related_slot = (
                    derivation["current_slot"],
                    derivation["related_slot"],
                )
                current = kinds[current_slot] if current_slot < len(kinds) else None
                related = kinds[related_slot] if related_slot < len(kinds) else None
                self._record(
                    derivation["source_relation"],
                    current_slot,
                    related_slot,
                    current,
                    related,
                )
                self._walk(derivation.get("predicate"), related, (current, related))

    def _referenced_entity(self, node: object) -> str | None:
        kinds: set[str] = set()

        def collect(value):
            if isinstance(value, dict):
                if value.get("kind") in {"count_related", "sum_related"}:
                    return  # These execute in a different entity context.
                if value.get("kind") == "derived":
                    rule = self.derived.get(value.get("name"), {})
                    if rule.get("entity") and rule["entity"] != "Scalar":
                        kinds.add(rule["entity"])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(node)
        return next(iter(kinds)) if len(kinds) == 1 else None

    def _record(self, name, current_slot, related_slot, current, related, stack=()):
        schema = self.relations.get(name, {})
        if {current_slot, related_slot} != {0, 1} or schema.get("arity", 2) != 2:
            self.usages[name].append((current_slot, (None, None)))
            return [None, None]
        kinds = [None, None]
        kinds[current_slot], kinds[related_slot] = current, related
        declared = schema.get("slot_entities", [])
        if len(declared) == 2 and kinds.count(None) == 1:
            # Fill the remaining kind, not its declared position: old executable
            # membership nodes can disagree with the serialized declaration.
            remaining = list(declared)
            known = next(kind for kind in kinds if kind is not None)
            if known in remaining:
                remaining.remove(known)
                kinds[kinds.index(None)] = remaining[0]
        self.usages[name].append((current_slot, tuple(kinds)))
        derivation = schema.get("derivation")
        if derivation and name not in stack:
            self._record(
                derivation["source_relation"],
                derivation["current_slot"],
                derivation["related_slot"],
                kinds[current_slot],
                kinds[related_slot],
                (*stack, name),
            )
        return kinds

    def _walk(self, node: object, entity: str | None, membership_context=None):
        if isinstance(node, list):
            for child in node:
                self._walk(child, entity, membership_context)
        elif isinstance(node, dict):
            if node.get("kind") in {"count_related", "sum_related", "relation_member"}:
                name = node["relation"]
                current_slot, related_slot = node["current_slot"], node["related_slot"]
                related = self._referenced_entity(
                    [node.get("value"), node.get("where")]
                )
                current = entity
                if node["kind"] == "relation_member" and membership_context:
                    current, related = membership_context
                elif node["kind"] != "relation_member" and (
                    derivation := self.relations.get(name, {}).get("derivation")
                ):
                    if entity and derivation.get("entity") == entity:
                        declared = derivation.get("slot_entities", [])
                        current = (
                            declared[current_slot]
                            if current_slot < len(declared)
                            else None
                        )
                kinds = self._record(name, current_slot, related_slot, current, related)
                self._walk(
                    node.get("where"),
                    kinds[related_slot] if related_slot < len(kinds) else None,
                )
            else:
                for child in node.values():
                    self._walk(child, entity, membership_context)

    def layout(self, names: list[str], owner: str | None) -> RelationLayout:
        # The CLI can emit both canonical and legacy short aliases. Resolve the
        # actual declared name, never use a suffix to choose between imports.
        name = next((name for name in names if name in self.relations), names[0])
        schema = self.relations.get(name, {})
        if not schema.get("slot_entities"):
            return LEGACY_LAYOUT
        if schema.get("arity", 2) != 2 or len(schema["slot_entities"]) != 2:
            raise ValueError(f"companion relation {name} must have two argument kinds")
        usages = self.usages.get(name)
        if usages:
            # Combine partial executable evidence; an unknown use must not
            # erase another use's known kind, or trigger declaration fallback.
            candidates = [
                {kinds[slot] for _, kinds in usages if kinds[slot] is not None}
                for slot in range(2)
            ]
            if any(len(kinds) > 1 for kinds in candidates):
                raise ValueError(f"conflicting executable relation slots for {name}")
            kinds = [next(iter(values)) if values else None for values in candidates]
            matches = [slot for slot, kind in enumerate(kinds) if kind == owner]
            if len(matches) == 1:
                slot = matches[0]
            elif len(matches) == 2 and owner is not None:
                current_slots = {current_slot for current_slot, _ in usages}
                if len(current_slots) != 1:
                    raise ValueError(
                        f"conflicting executable relation slots for {name}"
                    )
                slot = current_slots.pop()
            else:
                raise ValueError(f"cannot place companion owner {owner!r} in {name}")
            if kinds[1 - slot] is None:
                raise ValueError(f"unknown related entity kind for {name}")
            return RelationLayout(slot, kinds[slot], kinds[1 - slot])
        kinds = schema["slot_entities"]
        matches = [slot for slot, kind in enumerate(kinds) if kind == owner]
        if len(matches) == 2:
            return RelationLayout(1, owner, owner)
        if len(matches) != 1:
            raise ValueError(f"cannot place companion owner {owner!r} in unused {name}")
        slot = matches[0]
        return RelationLayout(slot, kinds[slot], kinds[1 - slot])
