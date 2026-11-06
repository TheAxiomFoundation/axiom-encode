"""Loader and data structures for the canonical-concept registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

REGISTRY_FORMAT = "axiom-encode/concepts/v1"

# A producer anchor must be a string the anchored-reference scanners can
# match (`<jurisdiction>:<path>` in validator/auto_repair/audit
# ANCHORED_REF_RE). An anchor those scanners cannot see could never be
# accepted, so it is rejected at load time instead.
PRODUCER_ANCHOR_RE = re.compile(r"[a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+")

PRODUCER_PERIOD_KEYS = frozenset({"anchor", "label", "effective_from", "effective_to"})


@dataclass(frozen=True)
class ProducerPeriod:
    """The dates one vintage's producer module covers.

    ``effective_from`` and ``effective_to`` are inclusive calendar dates, for
    example one federal fiscal year (``2026-10-01`` to ``2027-09-30``).
    ``label`` is the short name an author or a prompt uses for the vintage
    (``FY2027``). The registry uses periods only to describe vintages to the
    encoder model; acceptance is decided by ``Concept.producer_anchors``.
    """

    anchor: str
    label: str
    effective_from: date
    effective_to: date

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, str):
            raise ValueError(
                f"producer period anchor must be a string, got {self.anchor!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(
                f"producer period for {self.anchor!r}: label must be a non-empty string"
            )
        for key in ("effective_from", "effective_to"):
            value = getattr(self, key)
            if not isinstance(value, date) or isinstance(value, datetime):
                raise ValueError(
                    f"producer period for {self.anchor!r}: {key} must be a date"
                )
        if self.effective_from > self.effective_to:
            raise ValueError(
                f"producer period for {self.anchor!r}: effective_from "
                f"{self.effective_from.isoformat()} is after effective_to "
                f"{self.effective_to.isoformat()}"
            )

    def describe(self) -> str:
        """``FY2027, 2026-10-01 to 2027-09-30``."""
        return (
            f"{self.label}, {self.effective_from.isoformat()} to "
            f"{self.effective_to.isoformat()}"
        )


@dataclass(frozen=True)
class Concept:
    """One canonical legal concept with its approved variable name.

    ``producer_anchors`` is the complete set of modules allowed to define the
    canonical name. Most concepts have exactly one. A concept whose value is
    republished per period (for example the SNAP cost-of-living adjustment,
    one module per fiscal year) lists one anchor per vintage. Validation,
    test auto-repair and the corpus audit all decide against this set.

    ``producer_anchor`` is the legacy single-anchor field, kept so existing
    registry entries and callers keep working. When only it is given,
    ``producer_anchors`` is ``(producer_anchor,)``. When both are given,
    ``producer_anchor`` must be one of ``producer_anchors``; it is then only a
    label (the vintage current consumers import) and is never used to choose
    among vintages.

    ``producer_periods``, when given, names the dates each vintage covers:
    exactly one period per producer anchor, with no two periods overlapping.
    They are descriptive (the encoder prompt prints them so the model can
    pick the vintage whose dates it is encoding); they never widen or narrow
    which anchors are accepted.
    """

    id: str
    canonical_name: str
    producer_anchor: str | None
    blocked_synonyms: tuple[str, ...] = ()
    producer_missing: bool = False
    description: str | None = None
    source_file: Path | None = None
    producer_anchors: tuple[str, ...] = ()
    producer_periods: tuple[ProducerPeriod, ...] = ()

    def __post_init__(self) -> None:
        anchors = tuple(self.producer_anchors)
        if not anchors and self.producer_anchor is not None:
            anchors = (self.producer_anchor,)
        for anchor in anchors:
            if not isinstance(anchor, str) or not PRODUCER_ANCHOR_RE.fullmatch(anchor):
                raise ValueError(
                    f"concept {self.id}: producer anchor {anchor!r} is not a "
                    "<jurisdiction>:<path> RuleSpec anchor"
                )
        duplicates = sorted({a for a in anchors if anchors.count(a) > 1})
        if duplicates:
            raise ValueError(
                f"concept {self.id}: duplicate producer anchors {duplicates}"
            )
        if self.producer_anchor is not None and self.producer_anchor not in anchors:
            raise ValueError(
                f"concept {self.id}: producer_anchor {self.producer_anchor!r} "
                "must be one of producer_anchors"
            )
        object.__setattr__(self, "producer_anchors", anchors)
        object.__setattr__(
            self,
            "producer_periods",
            _validated_producer_periods(self.id, anchors, self.producer_periods),
        )

    @property
    def has_producer(self) -> bool:
        return bool(self.producer_anchors) and not self.producer_missing

    @property
    def unique_producer_anchor(self) -> str | None:
        """The one producer anchor, or ``None`` when there are zero or several.

        Auto-repair may redirect a reference only to this anchor. With several
        vintages there is no safe default, so callers must leave the
        reference alone and let validation flag it.
        """
        if len(self.producer_anchors) == 1:
            return self.producer_anchors[0]
        return None

    def accepts_producer_anchor(self, anchor: str | None) -> bool:
        """Whether a module at ``anchor`` may define this canonical name."""
        return anchor is not None and anchor in self.producer_anchors

    def producer_period(self, anchor: str | None) -> ProducerPeriod | None:
        """The registered period of the producer at ``anchor``, if any."""
        for period in self.producer_periods:
            if period.anchor == anchor:
                return period
        return None


def _validated_producer_periods(
    concept_id: str,
    anchors: tuple[str, ...],
    periods: tuple[ProducerPeriod, ...] | list[ProducerPeriod],
) -> tuple[ProducerPeriod, ...]:
    """Check a concept's periods and return them in ``anchors`` order.

    Empty is allowed (periods are optional). Otherwise there must be exactly
    one period per producer anchor, labels must be distinct, and no two
    periods may share a day: two vintages that both covered a date would
    leave the encoder no single producer to reference for it.
    """
    periods = tuple(periods)
    if not periods:
        return ()
    if not all(isinstance(period, ProducerPeriod) for period in periods):
        raise ValueError(
            f"concept {concept_id}: producer_periods must be ProducerPeriod values"
        )
    period_anchors = [period.anchor for period in periods]
    if sorted(period_anchors) != sorted(anchors):
        raise ValueError(
            f"concept {concept_id}: producer_periods must name each producer "
            f"anchor exactly once (anchors {list(anchors)}, periods "
            f"{period_anchors})"
        )
    labels = [period.label for period in periods]
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            f"concept {concept_id}: duplicate producer period labels {duplicate_labels}"
        )
    by_start = sorted(periods, key=lambda period: period.effective_from)
    for earlier, later in zip(by_start, by_start[1:]):
        if later.effective_from <= earlier.effective_to:
            raise ValueError(
                f"concept {concept_id}: producer periods {earlier.label} "
                f"({earlier.anchor}) and {later.label} ({later.anchor}) overlap"
            )
    by_anchor = {period.anchor: period for period in periods}
    return tuple(by_anchor[anchor] for anchor in anchors)


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
                if (
                    syn in synonym_to_concept
                    and synonym_to_concept[syn].id != concept.id
                ):
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
            raise ValueError(
                f"{source_file}: concept missing required {key!r}: {payload!r}"
            )
    blocked = payload.get("blocked_synonyms") or ()
    if isinstance(blocked, str):
        blocked = (blocked,)
    if not isinstance(blocked, (list, tuple)):
        raise ValueError(f"{source_file}: blocked_synonyms must be a list")
    raw_anchors = payload.get("producer_anchors")
    if raw_anchors is None:
        producer_anchors: tuple[str, ...] = ()
    else:
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise ValueError(
                f"{source_file}: concept {payload['id']!r} producer_anchors "
                "must be a non-empty list"
            )
        if not all(isinstance(anchor, str) for anchor in raw_anchors):
            raise ValueError(
                f"{source_file}: concept {payload['id']!r} producer_anchors "
                "must contain only strings"
            )
        producer_anchors = tuple(raw_anchors)
    producer_periods = _producer_periods_from_payload(
        payload.get("producer_periods"),
        concept_id=payload["id"],
        source_file=source_file,
    )
    try:
        return Concept(
            id=str(payload["id"]),
            canonical_name=str(payload["canonical_name"]),
            producer_anchor=(
                str(payload["producer_anchor"])
                if payload.get("producer_anchor")
                else None
            ),
            blocked_synonyms=tuple(str(s) for s in blocked),
            producer_missing=bool(payload.get("producer_missing", False)),
            description=payload.get("description"),
            source_file=source_file,
            producer_anchors=producer_anchors,
            producer_periods=producer_periods,
        )
    except ValueError as exc:
        raise ValueError(f"{source_file}: {exc}") from exc


def _producer_periods_from_payload(
    raw: Any, *, concept_id: Any, source_file: Path
) -> tuple[ProducerPeriod, ...]:
    """Parse ``producer_periods``: a list of {anchor, label, effective_from, effective_to}.

    A list (not a mapping keyed by anchor) so a repeated anchor is reported
    instead of being silently overwritten by the YAML parser.
    """
    if raw is None:
        return ()
    where = f"{source_file}: concept {concept_id!r} producer_periods"
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty list")
    periods: list[ProducerPeriod] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"{where}: each period must be a mapping")
        keys = set(item)
        if keys != PRODUCER_PERIOD_KEYS:
            missing = sorted(PRODUCER_PERIOD_KEYS - keys)
            unknown = sorted(str(key) for key in keys - PRODUCER_PERIOD_KEYS)
            raise ValueError(
                f"{where}: period keys must be {sorted(PRODUCER_PERIOD_KEYS)} "
                f"(missing {missing}, unknown {unknown})"
            )
        try:
            periods.append(
                ProducerPeriod(
                    anchor=item["anchor"],
                    label=item["label"],
                    effective_from=_period_date(item["effective_from"]),
                    effective_to=_period_date(item["effective_to"]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
    return tuple(periods)


def _period_date(value: Any) -> date:
    """A YAML date (``2026-10-01`` loads as ``date``) or an ISO date string."""
    if isinstance(value, datetime):
        raise ValueError(f"expected a calendar date, got datetime {value!r}")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError(f"expected an ISO calendar date (YYYY-MM-DD), got {value!r}")
