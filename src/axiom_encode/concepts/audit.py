"""Corpus-wide producer/consumer drift detection.

Walks one or more rules/rulespec roots, builds a producer→consumer graph, and
reports name drift relative to the canonical concept registry. Powers
`axiom-encode concepts audit` and seeds new entries into concepts/data/*.yaml.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .jurisdiction import jurisdiction_prefix
from .registry import ConceptRegistry

IDENT_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")
ANCHORED_REF_RE = re.compile(
    r"([a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+)#(?:input\.)?([a-z][a-z0-9_]*)"
)


@dataclass(frozen=True)
class DriftFinding:
    """One drift instance the audit surfaced."""

    kind: str  # "blocked_synonym" | "missing_producer" | "anchored_ref_miss" | "canonical_conflict"
    name: str
    anchor: str | None
    site_paths: tuple[Path, ...]
    detail: str
    nearby_producers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CorpusGraph:
    """Resolved producer/consumer graph across one corpus walk."""

    producers: dict[str, list[tuple[str, Path]]]
    consumers: dict[str, list[tuple[str, Path, str]]]
    anchored_refs: dict[tuple[str, str], list[Path]]
    file_to_producers: dict[str, set[str]]


def build_corpus_graph(
    roots: Iterable[Path], *, path_filter: re.Pattern[str] | None = None
) -> CorpusGraph:
    producers: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    consumers: dict[str, list[tuple[str, Path, str]]] = defaultdict(list)
    anchored_refs: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.yaml"):
            if path_filter is not None and not path_filter.search(str(path)):
                continue
            text = path.read_text()
            try:
                doc = yaml.safe_load(text)
            except Exception:
                doc = None
            anchor = _anchor_for(path, root)

            for m in ANCHORED_REF_RE.finditer(text):
                anchored_refs[(m.group(1), m.group(2))].append(path)

            if not isinstance(doc, dict):
                continue
            rules = doc.get("rules") or []
            for rule in rules if isinstance(rules, list) else []:
                if not isinstance(rule, dict):
                    continue
                name = rule.get("name") if isinstance(rule.get("name"), str) else None
                if name and not path.name.endswith(".test.yaml"):
                    producers[name].append((anchor, path))
                versions = rule.get("versions") or []
                for v in versions if isinstance(versions, list) else []:
                    if not isinstance(v, dict):
                        continue
                    formula = v.get("formula") or ""
                    if isinstance(formula, str):
                        for ident in set(IDENT_RE.findall(formula)):
                            consumers[ident].append((anchor, path, name or ""))

    file_to_producers: dict[str, set[str]] = defaultdict(set)
    for name, sites in producers.items():
        for anchor, _ in sites:
            file_to_producers[anchor].add(name)

    return CorpusGraph(
        producers=dict(producers),
        consumers=dict(consumers),
        anchored_refs=dict(anchored_refs),
        file_to_producers=dict(file_to_producers),
    )


def audit_corpus(
    roots: Iterable[Path],
    registry: ConceptRegistry,
    *,
    path_filter: re.Pattern[str] | None = None,
    name_prefixes: Iterable[str] | None = None,
) -> list[DriftFinding]:
    """Walk the corpus and return every drift finding."""
    graph = build_corpus_graph(roots, path_filter=path_filter)
    findings: list[DriftFinding] = []

    prefixes = tuple(name_prefixes or ())

    def _in_scope(name: str) -> bool:
        return not prefixes or name.startswith(prefixes)

    # Blocked synonyms in use anywhere
    for syn, concept in registry.synonym_to_concept.items():
        if not _in_scope(syn):
            continue
        sites_consumer = graph.consumers.get(syn, [])
        sites_producer = graph.producers.get(syn, [])
        ref_paths = [
            p
            for (anchor, name), files in graph.anchored_refs.items()
            if name == syn
            for p in files
        ]
        all_paths = tuple(
            dict.fromkeys(
                [p for _, p, _ in sites_consumer]
                + [p for _, p in sites_producer]
                + ref_paths
            )
        )
        if all_paths:
            findings.append(
                DriftFinding(
                    kind="blocked_synonym",
                    name=syn,
                    anchor=concept.producer_anchor,
                    site_paths=all_paths,
                    detail=(
                        f"{syn!r} is a blocked synonym for concept {concept.id} "
                        f"(canonical: {concept.canonical_name})"
                    ),
                    nearby_producers=(concept.canonical_name,),
                )
            )

    # Anchored-ref misses (file#name where the file doesn't produce that name)
    for (anchor, name), files in graph.anchored_refs.items():
        if not _in_scope(name):
            continue
        if registry.lookup_synonym(name) is not None:
            continue  # already covered by the blocked_synonym finding
        produced_here = graph.file_to_producers.get(anchor)
        if produced_here is None:
            continue  # anchor file outside the walked corpus
        if name in produced_here:
            continue
        registry_canonical = registry.lookup_canonical(name)
        if registry_canonical and registry_canonical.producer_anchor == anchor:
            continue  # registry says this is correct; producer just hasn't been re-encoded yet
        nearby = tuple(sorted(n for n in produced_here if _similar(name, n)))
        findings.append(
            DriftFinding(
                kind="anchored_ref_miss",
                name=name,
                anchor=anchor,
                site_paths=tuple(files),
                detail=f"{anchor}#{name} references a name not produced by {anchor}",
                nearby_producers=nearby,
            )
        )

    # Orphan consumers (referenced in some formula, no producer anywhere)
    for name, sites in graph.consumers.items():
        if not _in_scope(name):
            continue
        if name in graph.producers:
            continue
        if registry.lookup_synonym(name) is not None:
            continue  # already covered by the blocked_synonym finding
        # If registry says producer_missing or the name is a registered canonical
        # whose producer isn't encoded yet, classify as missing_producer (lower severity).
        registered = registry.concept_for_name(name)
        if registered and registered.producer_missing:
            continue
        nearby = tuple(sorted(n for n in graph.producers if _similar(name, n)))
        findings.append(
            DriftFinding(
                kind="missing_producer",
                name=name,
                anchor=None,
                site_paths=tuple({path for _, path, _ in sites}),
                detail=f"{name!r} is consumed but no rule produces it",
                nearby_producers=nearby,
            )
        )

    # Canonical-name conflicts: registered canonical produced under a different anchor
    for canonical, concept in registry.canonical_to_concept.items():
        producer_sites = graph.producers.get(canonical, [])
        if not producer_sites or not concept.has_producer:
            continue
        bad = [(a, p) for a, p in producer_sites if a != concept.producer_anchor]
        if not bad:
            continue
        bad_anchors = ", ".join(sorted({a for a, _ in bad}))
        findings.append(
            DriftFinding(
                kind="canonical_conflict",
                name=canonical,
                anchor=concept.producer_anchor,
                site_paths=tuple(p for _, p in bad),
                detail=(
                    f"{canonical} produced under {bad_anchors} but registry "
                    f"expects {concept.producer_anchor}"
                ),
            )
        )

    return findings


def _anchor_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    if rel.name.endswith(".test"):
        rel = rel.with_name(rel.name[:-5])
    return f"{jurisdiction_prefix(root)}:{rel.as_posix()}"


def _similar(a: str, b: str) -> bool:
    sa, sb = set(a.split("_")), set(b.split("_"))
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / max(len(sa), len(sb))
    return overlap >= 0.55
