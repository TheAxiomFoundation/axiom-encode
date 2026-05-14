"""Pre-write validator: reject generated RuleSpec that uses blocked synonyms
or conflicts with the canonical-concept registry.

Hook into `cli.py:_apply_generated_encoding_result` before `shutil.copy2` so
the encoder can't install drift into a live rules repo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .registry import Concept, ConceptRegistry

IDENT_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")
ANCHORED_REF_RE = re.compile(r"(us:[a-z0-9\-/\.]+)#(?:input\.)?([a-z][a-z0-9_]*)")


@dataclass(frozen=True)
class CanonicalNameViolation:
    kind: str  # "blocked_synonym" | "canonical_conflict" | "anchored_ref_miss"
    name: str
    where: str  # file:rule or file:anchored-ref
    concept_id: str | None
    detail: str

    def __str__(self) -> str:
        cid = f" ({self.concept_id})" if self.concept_id else ""
        return f"[{self.kind}] {self.name} at {self.where}{cid}: {self.detail}"


def validate_generated_against_registry(
    yaml_paths: Iterable[Path],
    registry: ConceptRegistry,
    *,
    apply_anchor: str | None = None,
) -> list[CanonicalNameViolation]:
    """Check every generated YAML file against the registry.

    `apply_anchor` (e.g. "us:regulations/7-cfr/273/10") is the anchor the
    generated content will live under once applied; used to decide whether a
    producer rule's name conflicts with the canonical for that concept.
    """
    violations: list[CanonicalNameViolation] = []
    for path in yaml_paths:
        if not path.exists():
            continue
        text = path.read_text()
        try:
            doc = yaml.safe_load(text)
        except Exception:
            doc = None

        # 1. Anchored-ref scan: catch any `us:file#name` whose name is a blocked synonym.
        for m in ANCHORED_REF_RE.finditer(text):
            anchor, name = m.group(1), m.group(2)
            blocked = registry.lookup_synonym(name)
            if blocked is not None:
                violations.append(
                    CanonicalNameViolation(
                        kind="blocked_synonym",
                        name=name,
                        where=f"{path}:{anchor}#{name}",
                        concept_id=blocked.id,
                        detail=(
                            f"use canonical {blocked.canonical_name!r} instead"
                        ),
                    )
                )

        if not isinstance(doc, dict):
            continue

        rules = doc.get("rules") or []
        for rule in rules if isinstance(rules, list) else []:
            if not isinstance(rule, dict):
                continue
            rname = rule.get("name") if isinstance(rule.get("name"), str) else None

            # 2. Producer rule name is a blocked synonym?
            if rname:
                blocked = registry.lookup_synonym(rname)
                if blocked is not None and not path.name.endswith(".test.yaml"):
                    violations.append(
                        CanonicalNameViolation(
                            kind="blocked_synonym",
                            name=rname,
                            where=f"{path}:rule {rname}",
                            concept_id=blocked.id,
                            detail=f"rename producer to canonical {blocked.canonical_name!r}",
                        )
                    )
                # 3. Producer rule name is a registered canonical but applied under wrong anchor?
                canonical = registry.lookup_canonical(rname)
                if (
                    canonical is not None
                    and apply_anchor is not None
                    and canonical.producer_anchor is not None
                    and canonical.producer_anchor != apply_anchor
                    and not path.name.endswith(".test.yaml")
                ):
                    violations.append(
                        CanonicalNameViolation(
                            kind="canonical_conflict",
                            name=rname,
                            where=f"{path}:rule {rname}",
                            concept_id=canonical.id,
                            detail=(
                                f"canonical anchor is {canonical.producer_anchor}, "
                                f"applying under {apply_anchor}"
                            ),
                        )
                    )

            # 4. Formula identifiers reference a blocked synonym?
            versions = rule.get("versions") or []
            for v in versions if isinstance(versions, list) else []:
                if not isinstance(v, dict):
                    continue
                formula = v.get("formula") or ""
                if not isinstance(formula, str):
                    continue
                for ident in set(IDENT_RE.findall(formula)):
                    blocked = registry.lookup_synonym(ident)
                    if blocked is not None:
                        violations.append(
                            CanonicalNameViolation(
                                kind="blocked_synonym",
                                name=ident,
                                where=f"{path}:rule {rname} formula",
                                concept_id=blocked.id,
                                detail=(
                                    f"use canonical {blocked.canonical_name!r} in formula"
                                ),
                            )
                        )

    # Dedup
    seen: set[tuple[str, str, str]] = set()
    deduped: list[CanonicalNameViolation] = []
    for v in violations:
        key = (v.kind, v.name, v.where)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped
