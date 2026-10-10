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
ANCHORED_REF_RE = re.compile(
    r"([a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+)#(input\.)?([a-z][a-z0-9_]*)"
)


@dataclass(frozen=True)
class CanonicalNameViolation:
    kind: str  # "blocked_synonym" | "canonical_conflict" | "anchored_ref_miss" | "parse_error"
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
    generated content will live under once applied. It identifies producer
    conflicts and distinguishes candidate-owned input slots from legacy slots
    exposed by imported modules. The latter are validated by overlay execution.

    A registered canonical may have several accepted producers (one per
    vintage, `Concept.producer_anchors`). A module may define the name, and a
    non-input reference may target it, iff its anchor is one of them.
    """
    violations: list[CanonicalNameViolation] = []
    for path in yaml_paths:
        if not path.exists():
            continue
        text = path.read_text()
        try:
            doc = yaml.safe_load(text)
        except Exception as exc:
            doc = None
            violations.append(
                CanonicalNameViolation(
                    kind="parse_error",
                    name="",
                    where=str(path),
                    concept_id=None,
                    detail=f"YAML did not parse, validator cannot scan rules: {exc}",
                )
            )

        # 1. Anchored-ref scan: catch any `us:file#name` whose name is a blocked synonym,
        #    or whose name is a registered canonical but referenced under the wrong anchor.
        for m in ANCHORED_REF_RE.finditer(text):
            anchor, input_prefix, name = m.group(1), m.group(2) or "", m.group(3)
            is_input_ref = bool(input_prefix)
            blocked = registry.lookup_synonym(name)
            if blocked is not None:
                if is_input_ref and (
                    blocked.accepts_producer_anchor(anchor)
                    or (apply_anchor is not None and anchor != apply_anchor)
                ):
                    continue
                violations.append(
                    CanonicalNameViolation(
                        kind="blocked_synonym",
                        name=name,
                        where=f"{path}:{anchor}#{name}",
                        concept_id=blocked.id,
                        detail=(f"use canonical {blocked.canonical_name!r} instead"),
                    )
                )
                continue
            canonical = registry.lookup_canonical(name)
            if (
                canonical is not None
                and canonical.has_producer
                and not canonical.accepts_producer_anchor(anchor)
                and not is_input_ref
            ):
                violations.append(
                    CanonicalNameViolation(
                        kind="anchored_ref_miss",
                        name=name,
                        where=f"{path}:{anchor}#{name}",
                        concept_id=canonical.id,
                        detail=_anchored_ref_miss_detail(canonical, anchor),
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
                    and canonical.has_producer
                    and not canonical.accepts_producer_anchor(apply_anchor)
                    and not path.name.endswith(".test.yaml")
                ):
                    violations.append(
                        CanonicalNameViolation(
                            kind="canonical_conflict",
                            name=rname,
                            where=f"{path}:rule {rname}",
                            concept_id=canonical.id,
                            detail=_canonical_conflict_detail(canonical, apply_anchor),
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


def _anchored_ref_miss_detail(concept: Concept, anchor: str) -> str:
    anchors = concept.producer_anchors
    if len(anchors) == 1:
        return (
            f"canonical {concept.canonical_name!r} is anchored at "
            f"{anchors[0]}, not {anchor}"
        )
    return (
        f"canonical {concept.canonical_name!r} has {len(anchors)} accepted "
        f"producers ({', '.join(anchors)}), and {anchor} is not one of them; "
        "reference the producer of the intended vintage explicitly "
        "(auto-repair never chooses a vintage)"
    )


def _canonical_conflict_detail(concept: Concept, apply_anchor: str) -> str:
    anchors = concept.producer_anchors
    if len(anchors) == 1:
        return f"canonical anchor is {anchors[0]}, applying under {apply_anchor}"
    return (
        f"accepted producer anchors are {', '.join(anchors)}; "
        f"applying under {apply_anchor}"
    )
