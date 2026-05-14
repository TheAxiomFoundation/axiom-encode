"""Loader and data structures for the canonical-concept registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FORMAT = "axiom-encode/concepts/v1"


@dataclass(frozen=True)
class Concept:
    """One canonical legal concept with its approved variable name."""

    id: str
    canonical_name: str
    producer_anchor: str | None
    blocked_synonyms: tuple[str, ...] = ()
    producer_missing: bool = False
    description: str | None = None
    source_file: Path | None = None

    @property
    def has_producer(self) -> bool:
        return self.producer_anchor is not None and not self.producer_missing


@dataclass(frozen=True)
class ConceptRegistry:
    """Resolved canonical-concept registry. Look up by id, canonical, or synonym."""

    concepts_by_id: dict[str, Concept] = field(default_factory=dict)
    canonical_to_concept: dict[str, Concept] = field(default_factory=dict)
    synonym_to_concept: dict[str, Concept] = field(default_factory=dict)

    def lookup_canonical(self, name: str) -> Concept | None:
        return self.canonical_to_concept.get(name)

    def lookup_synonym(self, name: str) -> Concept | None:
        return self.synonym_to_concept.get(name)

    def concept_for_name(self, name: str) -> Concept | None:
        return self.canonical_to_concept.get(name) or self.synonym_to_concept.get(name)

    def validate(self) -> list[str]:
        issues: list[str] = []
        for name, concept in self.canonical_to_concept.items():
            if name in self.synonym_to_concept:
                other = self.synonym_to_concept[name]
                issues.append(
                    f"{name} is both canonical for {concept.id} and blocked synonym for {other.id}"
                )
        canonical_seen: dict[str, str] = {}
        for cid, concept in self.concepts_by_id.items():
            existing = canonical_seen.get(concept.canonical_name)
            if existing and existing != cid:
                issues.append(
                    f"Two concepts share canonical_name {concept.canonical_name!r}: "
                    f"{existing} and {cid}"
                )
            canonical_seen[concept.canonical_name] = cid
        return issues


def load_concept_registry(data_root: Path | None = None) -> ConceptRegistry:
    """Load packaged concept YAML files from concepts/data/."""
    root = data_root or Path(__file__).with_name("data")
    concepts_by_id: dict[str, Concept] = {}
    canonical_to_concept: dict[str, Concept] = {}
    synonym_to_concept: dict[str, Concept] = {}

    for path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(path.read_text()) or {}
        fmt = payload.get("format")
        if fmt != REGISTRY_FORMAT:
            raise ValueError(f"{path}: unsupported registry format {fmt!r}")
        raw_concepts = payload.get("concepts") or []
        if not isinstance(raw_concepts, list):
            raise ValueError(f"{path}: concepts must be a list")
        for raw in raw_concepts:
            concept = _concept_from_payload(raw, source_file=path)
            if concept.id in concepts_by_id:
                raise ValueError(
                    f"Duplicate concept id {concept.id!r} "
                    f"(in {concepts_by_id[concept.id].source_file} and {path})"
                )
            concepts_by_id[concept.id] = concept
            if concept.canonical_name in canonical_to_concept:
                other = canonical_to_concept[concept.canonical_name]
                raise ValueError(
                    f"Canonical name {concept.canonical_name!r} claimed by "
                    f"both {other.id} and {concept.id}"
                )
            canonical_to_concept[concept.canonical_name] = concept
            for syn in concept.blocked_synonyms:
                if syn in canonical_to_concept:
                    raise ValueError(
                        f"{syn!r} is canonical for "
                        f"{canonical_to_concept[syn].id} but blocked by {concept.id}"
                    )
                if syn in synonym_to_concept and synonym_to_concept[syn].id != concept.id:
                    raise ValueError(
                        f"{syn!r} is blocked synonym for both "
                        f"{synonym_to_concept[syn].id} and {concept.id}"
                    )
                synonym_to_concept[syn] = concept

    registry = ConceptRegistry(
        concepts_by_id=concepts_by_id,
        canonical_to_concept=canonical_to_concept,
        synonym_to_concept=synonym_to_concept,
    )
    issues = registry.validate()
    if issues:
        raise ValueError("Invalid concept registry: " + "; ".join(issues))
    return registry


def _concept_from_payload(payload: Any, *, source_file: Path) -> Concept:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_file}: each concept must be a mapping")
    required = ("id", "canonical_name")
    for key in required:
        if key not in payload:
            raise ValueError(f"{source_file}: concept missing required {key!r}: {payload!r}")
    blocked = payload.get("blocked_synonyms") or ()
    if isinstance(blocked, str):
        blocked = (blocked,)
    if not isinstance(blocked, (list, tuple)):
        raise ValueError(f"{source_file}: blocked_synonyms must be a list")
    return Concept(
        id=str(payload["id"]),
        canonical_name=str(payload["canonical_name"]),
        producer_anchor=(
            str(payload["producer_anchor"]) if payload.get("producer_anchor") else None
        ),
        blocked_synonyms=tuple(str(s) for s in blocked),
        producer_missing=bool(payload.get("producer_missing", False)),
        description=payload.get("description"),
        source_file=source_file,
    )
