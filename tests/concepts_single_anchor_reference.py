"""Single-anchor concept registry semantics, frozen as a differential oracle.

These are the validator, test auto-repair and prompt guidance from
axiom-encode origin/main 436cf3044 (src/axiom_encode/concepts/validator.py
lines 38-188, auto_repair.py lines 71-109, and harness/evals.py lines
12186-12262), copied verbatim apart from the function names and these
changes, none of which alters behaviour for the inputs the tests pass:

- `concept.has_producer` is replaced by `_has_producer(concept)`, main's
  definition of that property (registry.py lines 26-28 at 436cf3044),
  because the live property now reads `producer_anchors`;
- the guidance copy requires `registry` (main's fallback that loads the
  packaged registry when none is passed is dropped) and inlines
  `_canonical_concept_token_index`.

For producers they read only `Concept.producer_anchor`. For a registry in
which every concept has at most one producer, the multi-producer
implementation must agree with them exactly
(tests/test_concepts_vintage_producers.py). Do not update this file to track
the live implementation.

The scanner regexes are frozen here too (validator.py lines 19-22 and
auto_repair.py lines 32-34 at 436cf3044), so a later change to the live
patterns shows up as a differential failure instead of moving the oracle.
Only the `CanonicalNameViolation` record type is shared with the live code:
the differential test compares violation lists with `==`, which needs the
same class.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import yaml

from axiom_encode.concepts.registry import Concept, ConceptRegistry
from axiom_encode.concepts.validator import CanonicalNameViolation

# validator.py IDENT_RE and ANCHORED_REF_RE at 436cf3044.
IDENT_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")
ANCHORED_REF_RE = re.compile(
    r"([a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+)#(input\.)?([a-z][a-z0-9_]*)"
)
# auto_repair.py ANCHORED_REF_RE at 436cf3044 (its own copy of the pattern).
AUTO_REPAIR_ANCHORED_REF_RE = re.compile(
    r"([a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+)#(input\.)?([a-z][a-z0-9_]*)"
)
# harness/evals.py _CANONICAL_CONCEPT_TOKEN_RE at 436cf3044.
CANONICAL_CONCEPT_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*")


def _has_producer(concept: Concept) -> bool:
    """`Concept.has_producer` as defined at 436cf3044."""
    return concept.producer_anchor is not None and not concept.producer_missing


def legacy_validate_generated_against_registry(
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
                    blocked.producer_anchor == anchor
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
                and _has_producer(canonical)
                and canonical.producer_anchor != anchor
                and not is_input_ref
            ):
                violations.append(
                    CanonicalNameViolation(
                        kind="anchored_ref_miss",
                        name=name,
                        where=f"{path}:{anchor}#{name}",
                        concept_id=canonical.id,
                        detail=(
                            f"canonical {name!r} is anchored at "
                            f"{canonical.producer_anchor}, not {anchor}"
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
                    and _has_producer(canonical)
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


def legacy_rewrite_anchored_refs(
    text: str,
    registry: ConceptRegistry,
    *,
    apply_anchor: str | None = None,
) -> str:
    def repl(match: re.Match[str]) -> str:
        anchor, input_prefix, name = (
            match.group(1),
            match.group(2) or "",
            match.group(3),
        )
        is_input_ref = bool(input_prefix)
        blocked = registry.lookup_synonym(name)
        if blocked is not None:
            # An imported module's input slots must match the names that module
            # actually exposes. Renaming a legacy external slot in the
            # companion test without migrating the imported module makes a
            # previously executable test silently lose its scenario-setting
            # value. The overlay validator proves that preserved external refs
            # resolve; canonical naming remains mandatory for the candidate's
            # own input slots.
            if is_input_ref and apply_anchor is not None and anchor != apply_anchor:
                return match.group(0)
            if is_input_ref and blocked.producer_anchor == anchor:
                return match.group(0)
            new_anchor = anchor if is_input_ref else (blocked.producer_anchor or anchor)
            return f"{new_anchor}#{input_prefix}{blocked.canonical_name}"
        canonical = registry.lookup_canonical(name)
        if (
            canonical is not None
            and _has_producer(canonical)
            and canonical.producer_anchor != anchor
            and not is_input_ref
        ):
            return f"{canonical.producer_anchor}#{input_prefix}{name}"
        return match.group(0)

    return AUTO_REPAIR_ANCHORED_REF_RE.sub(repl, text)


def legacy_format_canonical_concept_registry_guidance(
    source_text: str,
    workspace,
    context_files,
    *,
    registry: ConceptRegistry,
) -> str:
    """Inject canonical-concept registry directives scoped to mentioned concepts.

    Scans source text plus copied context files for any canonical name or
    blocked synonym in the registry; emits a terse "use these exact names"
    block for the matched concepts only. Concepts that never appear in any
    text are omitted so the prompt does not pay tokens for irrelevant rules.
    """
    if not registry.concepts_by_id:
        return ""

    haystack_parts: list[str] = [source_text]
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            haystack_parts.append(path.read_text())
        except OSError:
            continue
    haystack_tokens = set(CANONICAL_CONCEPT_TOKEN_RE.findall("\n".join(haystack_parts)))
    if not haystack_tokens:
        return ""

    token_index: dict[str, Concept] = {}
    for concept in registry.concepts_by_id.values():
        token_index[concept.canonical_name] = concept
        for synonym in concept.blocked_synonyms:
            token_index[synonym] = concept
    matched: list[Concept] = []
    seen_ids: set[str] = set()
    for token in haystack_tokens:
        concept = token_index.get(token)
        if concept is None or concept.id in seen_ids:
            continue
        seen_ids.add(concept.id)
        matched.append(concept)

    if not matched:
        return ""

    matched.sort(key=lambda c: c.id)
    lines: list[str] = []
    for concept in matched:
        parts: list[str] = [f"`{concept.canonical_name}`"]
        if _has_producer(concept):
            parts.append(f"producer `{concept.producer_anchor}`")
        if concept.blocked_synonyms:
            blocked = ", ".join(f"`{s}`" for s in concept.blocked_synonyms)
            parts.append(f"do not use: {blocked}")
        lines.append("- " + " — ".join(parts))

    return """
Canonical concept names:
Use these exact identifiers for the listed legal concepts; never introduce the blocked synonyms. The post-apply validator rejects drift, so picking the canonical name on the first pass avoids wasted re-encodes:
{lines}
""".format(lines="\n".join(lines))
