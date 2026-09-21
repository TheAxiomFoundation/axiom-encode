"""Pure contracts for the model-free legacy successor repoint transaction.

A *successor repoint* retires one legacy v1 RuleSpec primary whose exported
concepts already exist, byte-for-byte equivalent, in an existing signed-v5
module at an unrelated canonical path, and rewrites the legacy module's exact
dependents onto that successor.

This module deliberately cannot sign, invoke a model, read Git, or mutate a
checkout.  It owns the envelope grammar, the concept-equivalence proof, the
exact-token rewrite and its postimage proof, the reference inventory, the two
repoint-only metadata reconciliations, and the deterministic receipt identity.
The CLI owns provenance verification, validation, and the journaled install.

Deliberate scope limits (all fail closed, never silently widened):

* only ``kind: parameter`` concepts are provable.  A legacy module that exports
  a derived rule referenced by a dependent cannot be repointed here.
* ``indexed_by`` names may differ only when every dependent formula use of the
  concept is a literal integer subscript present in the successor's table.
* equivalence is proved over the **successor's** validity window, per the
  2026-09-20 ruling.  When a dependent uses the concept past that window the
  contract records ``post_window_behavior_change`` rather than extending the
  successor.
* a formula scalar written in a quoted YAML style cannot be rewritten in place;
  the transaction refuses rather than re-emitting the scalar.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Final, Mapping, Sequence

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS, RULESPEC_COMPOSITION_SPEC_ROOT
from .legacy_exact_dependent_concepts import (
    formula_segments,
    replace_formula_identifier,
)

ENVELOPE_SCHEMA: Final = "axiom-encode/legacy-successor-repoint/v1"
RECEIPT_SCHEMA: Final = "axiom-encode/legacy-successor-repoint-receipt/v1"
TOOL: Final = "axiom-encode repoint-legacy-successor"
RECEIPT_DIR: Final = PurePosixPath(".axiom/legacy-successor-repoints")
MANIFEST_TOOL: Final = "axiom-encode repoint-legacy-successor --dependent"
SUCCESSOR_MANIFEST_TOOL: Final = "axiom-encode repoint-legacy-successor --successor"

MAX_ENVELOPE_BYTES: Final = 64 * 1024
MAX_DEPENDENTS: Final = 32
MAX_CONCEPT_PAIRS: Final = 64
MAX_PROGRAM_SCOPE_UPDATES: Final = 32

UPSTREAM_SOURCE_CHECK_BASELINE: Final = PurePosixPath(
    ".axiom/upstream-source-check-baseline.txt"
)
MONEY_ATOM_RATCHET: Final = PurePosixPath("known-missing-money-atoms.yaml")

PROTECTED_CONTENT_ROOTS: Final = RULESPEC_ATOMIC_MODULE_ROOTS
PROGRAM_SPEC_ROOT: Final = RULESPEC_COMPOSITION_SPEC_ROOT

_CONCEPT_NAME = re.compile(r"[a-z][a-z0-9_]*")
_NUMERIC_LITERAL = re.compile(r"[+-]?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_INTEGER_LITERAL = re.compile(r"0|-?[1-9][0-9]*")
_IDENTIFIER_CHARACTER = "A-Za-z0-9_"
_DURABLE_REFERENCE_CHARACTER = r"A-Za-z0-9._:/-"
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
_SCOPE_KEY = re.compile(r"[a-z][a-z0-9_-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TEST_SUFFIX: Final = ".test.yaml"

_COMPARABLE_SURFACE_FIELDS: Final = ("kind", "dtype", "unit", "entity", "period")


class SuccessorRepointError(ValueError):
    """Raised when a repoint cannot be proved behavior-preserving."""


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConceptPair:
    """One old -> new exported concept rename declared by the operator."""

    old: str
    new: str


@dataclass(frozen=True, slots=True)
class ProgramScopeUpdate:
    """One ProgramSpec scope list that still names the legacy module."""

    program_spec: PurePosixPath
    scope: str


@dataclass(frozen=True, slots=True)
class RepointRequest:
    """One parsed, canonically serialized successor-repoint authority."""

    legacy_primary: PurePosixPath
    successor_primary: PurePosixPath
    dependents: tuple[PurePosixPath, ...]
    concept_map: tuple[ConceptPair, ...]
    program_scope_updates: tuple[ProgramScopeUpdate, ...]
    payload: Mapping[str, object]
    canonical_bytes: bytes
    sha256: str

    @property
    def jurisdiction(self) -> str:
        return self.legacy_primary.parts[0]

    @property
    def concept_renames(self) -> dict[str, str]:
        return {pair.old: pair.new for pair in self.concept_map}

    @property
    def legacy_companion(self) -> PurePosixPath:
        return companion_of(self.legacy_primary)

    @property
    def legacy_identity(self) -> str:
        return module_identity(self.legacy_primary)

    @property
    def successor_identity(self) -> str:
        return module_identity(self.successor_primary)

    @property
    def legacy_scope_path(self) -> str:
        return scope_module_path(self.legacy_primary)

    @property
    def successor_scope_path(self) -> str:
        return scope_module_path(self.successor_primary)


def companion_of(primary: PurePosixPath) -> PurePosixPath:
    """Return the mechanically coupled companion test path."""

    return primary.with_name(f"{primary.stem}{_TEST_SUFFIX}")


def module_identity(primary: PurePosixPath) -> str:
    """Convert a jurisdiction-prefixed primary path to its durable identity."""

    jurisdiction, content_root, *remainder = primary.with_suffix("").parts
    return f"{jurisdiction}:{content_root}/{'/'.join(remainder)}"


def scope_module_path(primary: PurePosixPath) -> str:
    """Convert a primary path to the jurisdiction-less ProgramSpec scope entry."""

    return PurePosixPath(*primary.with_suffix("").parts[1:]).as_posix()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _primary_path(raw: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise SuccessorRepointError(f"{field} must be a non-empty string")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) < 3
        or _JURISDICTION.fullmatch(path.parts[0]) is None
        or path.parts[1] not in PROTECTED_CONTENT_ROOTS
        or path.suffix != ".yaml"
        or path.name.endswith(_TEST_SUFFIX)
    ):
        raise SuccessorRepointError(
            f"{field} must be a jurisdiction-prefixed protected primary RuleSpec path"
        )
    return path


def _program_spec_path(raw: object, *, field: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise SuccessorRepointError(f"{field} must be a non-empty string")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) < 2
        or path.suffix != ".yaml"
        or path.name.endswith(_TEST_SUFFIX)
    ):
        raise SuccessorRepointError(
            f"{field} must be a repo-relative ProgramSpec .yaml path"
        )
    if path.parts[0] != PROGRAM_SPEC_ROOT and (
        len(path.parts) < 3 or path.parts[1] != PROGRAM_SPEC_ROOT
    ):
        raise SuccessorRepointError(
            f"{field} must live under programs/ or <jurisdiction>/programs/"
        )
    return path


def load_repoint_request_payload(payload: object) -> RepointRequest:
    """Parse one exact successor-repoint envelope object."""

    if not isinstance(payload, dict):
        raise SuccessorRepointError("successor repoint envelope must be a JSON object")
    expected_fields = {
        "schema",
        "legacy_primary",
        "successor_primary",
        "dependents",
        "concept_map",
        "program_scope_updates",
    }
    if set(payload) != expected_fields:
        raise SuccessorRepointError(
            "successor repoint envelope must contain exactly "
            + ", ".join(sorted(expected_fields))
        )
    if payload.get("schema") != ENVELOPE_SCHEMA:
        raise SuccessorRepointError(
            f"successor repoint envelope schema must be {ENVELOPE_SCHEMA}"
        )

    legacy = _primary_path(payload.get("legacy_primary"), field="legacy_primary")
    successor = _primary_path(
        payload.get("successor_primary"), field="successor_primary"
    )
    if legacy == successor:
        raise SuccessorRepointError(
            "successor repoint legacy and successor primaries must differ"
        )
    if legacy.parts[0] != successor.parts[0]:
        raise SuccessorRepointError(
            "successor repoint successor must share the legacy jurisdiction"
        )

    raw_dependents = payload.get("dependents")
    if (
        not isinstance(raw_dependents, list)
        or not raw_dependents
        or len(raw_dependents) > MAX_DEPENDENTS
    ):
        raise SuccessorRepointError(
            "successor repoint dependents must be a non-empty array of at most "
            f"{MAX_DEPENDENTS} paths"
        )
    dependents: list[PurePosixPath] = []
    for index, item in enumerate(raw_dependents):
        dependent = _primary_path(item, field=f"dependents[{index}]")
        if dependent in {legacy, successor}:
            raise SuccessorRepointError(
                "successor repoint dependents cannot name the legacy or successor "
                "primary"
            )
        if dependent.parts[0] != legacy.parts[0]:
            raise SuccessorRepointError(
                "successor repoint dependents must share the legacy jurisdiction"
            )
        if dependent in dependents:
            raise SuccessorRepointError("successor repoint dependents must be unique")
        dependents.append(dependent)

    raw_pairs = payload.get("concept_map")
    if (
        not isinstance(raw_pairs, list)
        or not raw_pairs
        or len(raw_pairs) > MAX_CONCEPT_PAIRS
    ):
        raise SuccessorRepointError(
            "successor repoint concept_map must be a non-empty array of at most "
            f"{MAX_CONCEPT_PAIRS} pairs"
        )
    pairs: list[ConceptPair] = []
    seen_old: set[str] = set()
    seen_new: set[str] = set()
    for index, item in enumerate(raw_pairs):
        field = f"concept_map[{index}]"
        if not isinstance(item, dict) or set(item) != {"from", "to"}:
            raise SuccessorRepointError(f"{field} must contain exactly from and to")
        old = item.get("from")
        new = item.get("to")
        for label, value in (("from", old), ("to", new)):
            if not isinstance(value, str) or _CONCEPT_NAME.fullmatch(value) is None:
                raise SuccessorRepointError(
                    f"{field}.{label} must be a lowercase concept identifier"
                )
        assert isinstance(old, str) and isinstance(new, str)
        if old == new:
            raise SuccessorRepointError(f"{field} is a no-op rename")
        if old in seen_old:
            raise SuccessorRepointError(f"{field}.from is duplicated")
        if new in seen_new:
            raise SuccessorRepointError(f"{field}.to collides with another rename")
        seen_old.add(old)
        seen_new.add(new)
        pairs.append(ConceptPair(old=old, new=new))
    collisions = sorted(seen_old & seen_new)
    if collisions:
        raise SuccessorRepointError(
            "successor repoint concept_map cannot chain or swap concept names: "
            + ", ".join(collisions)
        )

    raw_updates = payload.get("program_scope_updates")
    if not isinstance(raw_updates, list) or len(raw_updates) > (
        MAX_PROGRAM_SCOPE_UPDATES
    ):
        raise SuccessorRepointError(
            "successor repoint program_scope_updates must be an array of at most "
            f"{MAX_PROGRAM_SCOPE_UPDATES} entries"
        )
    updates: list[ProgramScopeUpdate] = []
    seen_updates: set[tuple[str, str]] = set()
    for index, item in enumerate(raw_updates):
        field = f"program_scope_updates[{index}]"
        if not isinstance(item, dict) or set(item) != {"program_spec", "scope"}:
            raise SuccessorRepointError(
                f"{field} must contain exactly program_spec and scope"
            )
        spec = _program_spec_path(
            item.get("program_spec"), field=f"{field}.program_spec"
        )
        scope = item.get("scope")
        if not isinstance(scope, str) or _SCOPE_KEY.fullmatch(scope) is None:
            raise SuccessorRepointError(f"{field}.scope must be a lowercase identifier")
        key = (spec.as_posix(), scope)
        if key in seen_updates:
            raise SuccessorRepointError(f"{field} is duplicated")
        seen_updates.add(key)
        updates.append(ProgramScopeUpdate(program_spec=spec, scope=scope))

    canonical_bytes = _canonical_json_bytes(payload)
    return RepointRequest(
        legacy_primary=legacy,
        successor_primary=successor,
        dependents=tuple(dependents),
        concept_map=tuple(pairs),
        program_scope_updates=tuple(updates),
        payload=dict(payload),
        canonical_bytes=canonical_bytes,
        sha256=hashlib.sha256(canonical_bytes).hexdigest(),
    )


def load_repoint_request_bytes(raw: bytes) -> RepointRequest:
    """Parse one exact successor-repoint envelope from bytes."""

    if len(raw) > MAX_ENVELOPE_BYTES:
        raise SuccessorRepointError("successor repoint envelope exceeds 64 KiB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise SuccessorRepointError(
            "successor repoint envelope is not valid UTF-8 JSON"
        ) from exc
    return load_repoint_request_payload(payload)


# ---------------------------------------------------------------------------
# Concept equivalence proof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParameterVersion:
    start: date
    end: date | None
    values: dict[int, Decimal]


@dataclass(frozen=True, slots=True)
class ConceptProof:
    """One proved old -> new parameter equivalence over the successor window."""

    old: str
    new: str
    indexed_by_from: str | None
    indexed_by_to: str | None
    keys: tuple[int, ...]
    window_start: str
    window_end: str | None
    probes: tuple[str, ...]
    formula_uses: int
    literal_subscripts: tuple[int, ...]
    reference_uses: int

    def as_receipt_entry(self) -> dict[str, object]:
        return {
            "from": self.old,
            "to": self.new,
            "indexed_by_from": self.indexed_by_from,
            "indexed_by_to": self.indexed_by_to,
            "keys": list(self.keys),
            "successor_window": {
                "effective_from": self.window_start,
                "effective_to": self.window_end,
            },
            "probed_dates": list(self.probes),
            "formula_uses": self.formula_uses,
            "literal_subscripts": list(self.literal_subscripts),
            "reference_uses": self.reference_uses,
        }


@dataclass(frozen=True, slots=True)
class ConceptProofSet:
    """Every proved rename plus the window semantics the receipt must record."""

    proofs: tuple[ConceptProof, ...]
    successor_window_start: str
    successor_window_end: str | None
    dependent_use_windows: tuple[dict[str, object], ...]
    post_window_behavior_change: bool

    @property
    def renames(self) -> dict[str, str]:
        return {proof.old: proof.new for proof in self.proofs}


def _load_module(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise SuccessorRepointError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(payload, dict):
        raise SuccessorRepointError(f"{label} is not a RuleSpec mapping")
    return payload


def _rules_by_name(
    payload: Mapping[str, object], *, label: str
) -> dict[str, dict[str, object]]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise SuccessorRepointError(f"{label} declares no rules list")
    result: dict[str, dict[str, object]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            raise SuccessorRepointError(f"{label} has a malformed rule entry")
        name = rule.get("name")
        if not isinstance(name, str) or _CONCEPT_NAME.fullmatch(name) is None:
            raise SuccessorRepointError(f"{label} has a rule without a valid name")
        if name in result:
            raise SuccessorRepointError(f"{label} exports duplicate concept {name!r}")
        result[name] = rule
    return result


def _iso_date(value: object, *, label: str) -> date:
    if not isinstance(value, str):
        raise SuccessorRepointError(f"{label} is not an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SuccessorRepointError(f"{label} is not an ISO date") from exc


def _decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise SuccessorRepointError(f"{label} is not a numeric value")
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip() if isinstance(value, (str, float)) else ""
    if _NUMERIC_LITERAL.fullmatch(text) is None:
        raise SuccessorRepointError(f"{label} is not a plain numeric literal")
    try:
        return Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - guarded by the regex
        raise SuccessorRepointError(f"{label} is not numeric") from exc


def _parameter_versions(
    rule: Mapping[str, object], *, concept: str, label: str
) -> tuple[_ParameterVersion, ...]:
    """Parse one scalar or indexed parameter into a contiguous version ladder."""

    if rule.get("kind") != "parameter":
        raise SuccessorRepointError(
            f"{label} concept {concept!r} is not kind: parameter"
        )
    indexed = rule.get("indexed_by") is not None
    versions = rule.get("versions")
    if not isinstance(versions, list) or not versions:
        raise SuccessorRepointError(f"{label} parameter {concept!r} has no versions")
    parsed: list[_ParameterVersion] = []
    for index, version in enumerate(versions):
        field = f"{label} parameter {concept!r} version {index}"
        if not isinstance(version, dict):
            raise SuccessorRepointError(f"{field} is malformed")
        start = _iso_date(
            version.get("effective_from"), label=f"{field}.effective_from"
        )
        end_raw = version.get("effective_to")
        end = (
            _iso_date(end_raw, label=f"{field}.effective_to")
            if end_raw is not None
            else None
        )
        if end is not None and end < start:
            raise SuccessorRepointError(f"{field} has an inverted period")
        if indexed:
            if "formula" in version:
                raise SuccessorRepointError(
                    f"{field} declares both an indexed table and a formula"
                )
            table = version.get("values")
            if not isinstance(table, dict) or not table:
                raise SuccessorRepointError(f"{field} has no indexed values table")
            values: dict[int, Decimal] = {}
            for key, cell in table.items():
                if isinstance(key, bool) or not isinstance(key, int):
                    raise SuccessorRepointError(
                        f"{field} has a non-integer table key {key!r}"
                    )
                if key in values:  # pragma: no cover - YAML rejects duplicate keys
                    raise SuccessorRepointError(f"{field} has a duplicate table key")
                values[key] = _decimal(cell, label=f"{field}.values[{key}]")
        else:
            if "values" in version:
                raise SuccessorRepointError(
                    f"{field} declares a values table without indexed_by"
                )
            values = {0: _decimal(version.get("formula"), label=f"{field}.formula")}
        parsed.append(_ParameterVersion(start=start, end=end, values=values))

    parsed.sort(key=lambda item: item.start)
    if [item.start for item in parsed] != sorted(
        {item.start for item in parsed}
    ):  # pragma: no cover - duplicate starts
        raise SuccessorRepointError(
            f"{label} parameter {concept!r} has duplicate effective_from dates"
        )
    for position, item in enumerate(parsed):
        last = position == len(parsed) - 1
        if not last:
            following = parsed[position + 1]
            if item.end is None:
                raise SuccessorRepointError(
                    f"{label} parameter {concept!r} has an open version before a later "
                    "version"
                )
            if following.start != item.end + timedelta(days=1):
                raise SuccessorRepointError(
                    f"{label} parameter {concept!r} has a gap or overlap between "
                    f"{item.end.isoformat()} and {following.start.isoformat()}"
                )
    return tuple(parsed)


def _window(versions: Sequence[_ParameterVersion]) -> tuple[date, date | None]:
    return versions[0].start, versions[-1].end


def _active(
    versions: Sequence[_ParameterVersion], at: date
) -> dict[int, Decimal] | None:
    for item in versions:
        if item.start <= at and (item.end is None or at <= item.end):
            return item.values
    return None


def _probe_dates(
    *,
    window: tuple[date, date | None],
    ladders: Sequence[Sequence[_ParameterVersion]],
) -> tuple[date, ...]:
    start, end = window
    probes: set[date] = {start}
    if end is not None:
        probes.add(end)

    def inside(value: date) -> bool:
        return start <= value and (end is None or value <= end)

    for ladder in ladders:
        for item in ladder:
            if inside(item.start):
                probes.add(item.start)
            if item.end is not None and inside(item.end + timedelta(days=1)):
                probes.add(item.end + timedelta(days=1))
    return tuple(sorted(probes))


def _formula_symbol_uses(formula: str, name: str) -> tuple[int | None, ...]:
    """Return one entry per unquoted, identifier-bounded use of ``name``.

    Each entry is the literal integer subscript immediately applied to the use,
    or ``None`` when the use is bare or subscripted by an expression.
    """

    pattern = re.compile(
        rf"(?<![{_IDENTIFIER_CHARACTER}]){re.escape(name)}(?![{_IDENTIFIER_CHARACTER}])"
    )
    uses: list[int | None] = []
    for unquoted, segment in formula_segments(formula):
        if not unquoted:
            continue
        for match in pattern.finditer(segment):
            tail = segment[match.end() :]
            if not tail.startswith("["):
                uses.append(None)
                continue
            closing = tail.find("]")
            if closing < 0:
                uses.append(None)
                continue
            subscript = tail[1:closing].strip()
            if _INTEGER_LITERAL.fullmatch(subscript) is None:
                uses.append(None)
                continue
            uses.append(int(subscript))
    return tuple(uses)


def _dependent_formula_versions(
    payload: Mapping[str, object], *, label: str
) -> tuple[tuple[str, date, date | None], ...]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return ()
    collected: list[tuple[str, date, date | None]] = []
    for rule_index, rule in enumerate(rules):
        versions = rule.get("versions") if isinstance(rule, dict) else None
        if not isinstance(versions, list):
            continue
        for version_index, version in enumerate(versions):
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            field = f"{label} rules[{rule_index}].versions[{version_index}]"
            start = _iso_date(
                version.get("effective_from"), label=f"{field}.effective_from"
            )
            end_raw = version.get("effective_to")
            end = (
                _iso_date(end_raw, label=f"{field}.effective_to")
                if end_raw is not None
                else None
            )
            if end is not None and end < start:
                raise SuccessorRepointError(f"{field} has an inverted period")
            collected.append((formula, start, end))
    return tuple(collected)


def _module_reference_uses(
    payload: Mapping[str, object],
    *,
    legacy_identity: str,
    label: str,
) -> dict[str, int]:
    """Count non-formula references to each legacy concept in one dependent."""

    counts: dict[str, int] = {}
    prefix = f"{legacy_identity}#"

    def note(value: object) -> None:
        if not isinstance(value, str) or not value.startswith(prefix):
            return
        fragment = value[len(prefix) :]
        if _CONCEPT_NAME.fullmatch(fragment) is None:
            raise SuccessorRepointError(
                f"{label} references the legacy module with a malformed fragment: "
                f"{value}"
            )
        counts[fragment] = counts.get(fragment, 0) + 1

    for item in payload.get("imports") or ():
        note(item)
    module = payload.get("module")
    deferred = module.get("deferred_outputs") if isinstance(module, dict) else None
    if isinstance(deferred, list):
        for entry in deferred:
            blocked = entry.get("blocked_by") if isinstance(entry, dict) else None
            if isinstance(blocked, list):
                for item in blocked:
                    note(item)
    rules = payload.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            for atom in _proof_import_atoms(rule):
                note(atom.get("target"))
    return counts


def _proof_import_atoms(rule: object) -> list[dict[str, object]]:
    metadata = rule.get("metadata") if isinstance(rule, dict) else None
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    if not isinstance(atoms, list):
        return []
    imports: list[dict[str, object]] = []
    for atom in atoms:
        imported = atom.get("import") if isinstance(atom, dict) else None
        if isinstance(imported, dict):
            imports.append(imported)
    return imports


def prove_concept_map(
    *,
    legacy_raw: bytes,
    successor_raw: bytes,
    request: RepointRequest,
    dependent_raws: Mapping[str, bytes],
) -> ConceptProofSet:
    """Prove every declared rename is value-identical over the successor window."""

    legacy_payload = _load_module(legacy_raw, label="legacy primary")
    successor_payload = _load_module(successor_raw, label="successor primary")
    legacy_rules = _rules_by_name(legacy_payload, label="legacy primary")
    successor_rules = _rules_by_name(successor_payload, label="successor primary")
    renames = request.concept_renames

    unknown_old = sorted(set(renames) - set(legacy_rules))
    if unknown_old:
        raise SuccessorRepointError(
            "concept_map names concepts the legacy module does not export: "
            + ", ".join(unknown_old)
        )
    unknown_new = sorted(set(renames.values()) - set(successor_rules))
    if unknown_new:
        raise SuccessorRepointError(
            "concept_map names concepts the successor module does not export: "
            + ", ".join(unknown_new)
        )

    formula_uses: dict[str, list[int | None]] = {old: [] for old in renames}
    use_windows: dict[str, set[tuple[date, date | None]]] = {
        old: set() for old in renames
    }
    reference_uses: dict[str, int] = {old: 0 for old in renames}
    for path, raw in sorted(dependent_raws.items()):
        payload = _load_module(raw, label=f"dependent {path}")
        local_rules = _rules_by_name(payload, label=f"dependent {path}")
        collisions = sorted(set(renames.values()) & set(local_rules))
        if collisions:
            raise SuccessorRepointError(
                f"dependent {path} already defines successor concepts: "
                + ", ".join(collisions)
            )
        imports = payload.get("imports")
        if imports is not None and (
            not isinstance(imports, list)
            or not all(isinstance(item, str) for item in imports)
        ):
            raise SuccessorRepointError(f"dependent {path} imports are malformed")
        for item in imports or ():
            assert isinstance(item, str)
            fragment = item.rsplit("#", 1)[-1] if "#" in item else None
            if fragment is not None and fragment in set(renames.values()):
                raise SuccessorRepointError(
                    f"dependent {path} already imports successor concept {fragment!r}"
                )
        if request.successor_identity in (imports or ()):
            raise SuccessorRepointError(
                f"dependent {path} already imports the successor module"
            )
        for fragment, count in _module_reference_uses(
            payload,
            legacy_identity=request.legacy_identity,
            label=f"dependent {path}",
        ).items():
            if fragment not in renames:
                raise SuccessorRepointError(
                    f"dependent {path} references unmapped legacy concept {fragment!r}"
                )
            reference_uses[fragment] += count
        for formula, start, end in _dependent_formula_versions(
            payload, label=f"dependent {path}"
        ):
            for old in renames:
                uses = _formula_symbol_uses(formula, old)
                if not uses:
                    continue
                formula_uses[old].extend(uses)
                use_windows[old].add((start, end))

    proofs: list[ConceptProof] = []
    window_starts: set[date] = set()
    window_ends: set[date | None] = set()
    for pair in request.concept_map:
        old_rule = legacy_rules[pair.old]
        new_rule = successor_rules[pair.new]
        mismatched = [
            field
            for field in _COMPARABLE_SURFACE_FIELDS
            if old_rule.get(field) != new_rule.get(field)
        ]
        if mismatched:
            raise SuccessorRepointError(
                f"concept {pair.old!r} -> {pair.new!r} differs on "
                + ", ".join(mismatched)
            )
        old_indexed_by = old_rule.get("indexed_by")
        new_indexed_by = new_rule.get("indexed_by")
        for label, value in (("legacy", old_indexed_by), ("successor", new_indexed_by)):
            if value is not None and not isinstance(value, str):
                raise SuccessorRepointError(
                    f"{label} concept {pair.old!r} has a malformed indexed_by"
                )
        if (old_indexed_by is None) != (new_indexed_by is None):
            raise SuccessorRepointError(
                f"concept {pair.old!r} -> {pair.new!r} changes between a scalar "
                "parameter and an indexed table"
            )

        old_versions = _parameter_versions(
            old_rule, concept=pair.old, label="legacy primary"
        )
        new_versions = _parameter_versions(
            new_rule, concept=pair.new, label="successor primary"
        )
        window = _window(new_versions)
        legacy_window = _window(old_versions)
        if window[1] is None and legacy_window[1] is not None:
            raise SuccessorRepointError(
                f"concept {pair.old!r} -> {pair.new!r} has an open successor window "
                "the legacy module does not cover"
            )
        probes = _probe_dates(window=window, ladders=(old_versions, new_versions))
        keys: tuple[int, ...] = ()
        for probe in probes:
            old_values = _active(old_versions, probe)
            new_values = _active(new_versions, probe)
            if old_values is None:
                raise SuccessorRepointError(
                    f"concept {pair.old!r} has no legacy value at "
                    f"{probe.isoformat()}, inside the successor window"
                )
            if new_values is None:  # pragma: no cover - contiguity guarantees this
                raise SuccessorRepointError(
                    f"concept {pair.new!r} has no successor value at "
                    f"{probe.isoformat()}"
                )
            if set(old_values) != set(new_values):
                raise SuccessorRepointError(
                    f"concept {pair.old!r} -> {pair.new!r} has different table keys at "
                    f"{probe.isoformat()}: "
                    f"{sorted(old_values)} vs {sorted(new_values)}"
                )
            differing = sorted(
                key for key in old_values if old_values[key] != new_values[key]
            )
            if differing:
                raise SuccessorRepointError(
                    f"concept {pair.old!r} -> {pair.new!r} differs at "
                    f"{probe.isoformat()} for keys {differing}"
                )
            keys = tuple(sorted(new_values))

        uses = formula_uses[pair.old]
        literal = tuple(sorted({use for use in uses if use is not None}))
        if old_indexed_by != new_indexed_by:
            if any(use is None for use in uses):
                raise SuccessorRepointError(
                    f"concept {pair.old!r} -> {pair.new!r} renames indexed_by "
                    f"{old_indexed_by!r} to {new_indexed_by!r} but a dependent uses it "
                    "without a literal integer subscript"
                )
            missing = sorted(key for key in literal if key not in keys)
            if missing:
                raise SuccessorRepointError(
                    f"concept {pair.old!r} -> {pair.new!r} is subscripted with keys "
                    f"{missing} that the successor table does not define"
                )
        window_starts.add(window[0])
        window_ends.add(window[1])
        proofs.append(
            ConceptProof(
                old=pair.old,
                new=pair.new,
                indexed_by_from=old_indexed_by,
                indexed_by_to=new_indexed_by,
                keys=keys,
                window_start=window[0].isoformat(),
                window_end=window[1].isoformat() if window[1] is not None else None,
                probes=tuple(probe.isoformat() for probe in probes),
                formula_uses=len(uses),
                literal_subscripts=literal,
                reference_uses=reference_uses[pair.old],
            )
        )

    if len(window_starts) != 1 or len(window_ends) != 1:
        raise SuccessorRepointError(
            "successor concepts do not share one validity window; repoint one "
            "window at a time"
        )
    successor_start = next(iter(window_starts))
    successor_end = next(iter(window_ends))

    recorded_windows: list[dict[str, object]] = []
    post_window_change = False
    for pair in request.concept_map:
        for start, end in sorted(
            use_windows[pair.old], key=lambda item: (item[0], item[1] or date.max)
        ):
            beyond = successor_end is not None and (end is None or end > successor_end)
            post_window_change = post_window_change or beyond
            recorded_windows.append(
                {
                    "concept": pair.old,
                    "effective_from": start.isoformat(),
                    "effective_to": end.isoformat() if end is not None else None,
                    "extends_past_successor_window": beyond,
                }
            )

    return ConceptProofSet(
        proofs=tuple(proofs),
        successor_window_start=successor_start.isoformat(),
        successor_window_end=(
            successor_end.isoformat() if successor_end is not None else None
        ),
        dependent_use_windows=tuple(recorded_windows),
        post_window_behavior_change=post_window_change,
    )


# ---------------------------------------------------------------------------
# Exact-token rewrite with a structural postimage proof
# ---------------------------------------------------------------------------


def _durable_pattern(token: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![{_DURABLE_REFERENCE_CHARACTER}])"
        rf"{re.escape(token)}"
        rf"(?![{_DURABLE_REFERENCE_CHARACTER}])"
    )


def _identifier_pattern(token: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![{_IDENTIFIER_CHARACTER}]){re.escape(token)}"
        rf"(?![{_IDENTIFIER_CHARACTER}])"
    )


def _scalar_nodes(
    root: Node,
) -> tuple[
    dict[tuple[object, ...], ScalarNode],
    dict[tuple[object, ...], ScalarNode],
]:
    """Index every composed scalar by its structural path (values and keys)."""

    values: dict[tuple[object, ...], ScalarNode] = {}
    keys: dict[tuple[object, ...], ScalarNode] = {}
    seen: set[int] = set()

    def walk(node: Node, path: tuple[object, ...]) -> None:
        if id(node) in seen:
            raise SuccessorRepointError(
                "RuleSpec document uses a YAML anchor or alias; the repoint rewrite "
                "requires one exact textual occurrence per reference"
            )
        seen.add(id(node))
        if isinstance(node, MappingNode):
            for key_node, value_node in node.value:
                if not isinstance(key_node, ScalarNode):
                    raise SuccessorRepointError(
                        "RuleSpec document has a non-scalar mapping key"
                    )
                child = path + (key_node.value,)
                keys[child] = key_node
                walk(value_node, child)
        elif isinstance(node, SequenceNode):
            for index, item in enumerate(node.value):
                walk(item, path + (index,))
        elif isinstance(node, ScalarNode):
            values[path] = node
        else:  # pragma: no cover - PyYAML has no other node kinds
            raise SuccessorRepointError("RuleSpec document has an unsupported node")

    walk(root, ())
    return values, keys


def _emit_token(node: ScalarNode, token: str) -> str:
    """Render ``token`` in the scalar's own YAML style."""

    if node.style in {"|", ">"}:
        raise SuccessorRepointError(
            "a durable reference is written as a block scalar; the repoint rewrite "
            "only replaces plain or quoted reference scalars"
        )
    if node.style in {"'", '"'}:
        return f"{node.style}{token}{node.style}"
    return token


def _map_reference(
    value: str,
    *,
    legacy_identity: str,
    successor_identity: str,
    renames: Mapping[str, str],
    label: str,
) -> str | None:
    """Rewrite one durable reference, or return None when it is unrelated."""

    pattern = _durable_pattern(legacy_identity)
    if pattern.search(value) is None:
        return None
    module, separator, fragment = value.partition("#")
    if pattern.fullmatch(module) is None:
        raise SuccessorRepointError(
            f"{label} embeds the legacy module identity in an unrecognized reference: "
            f"{value}"
        )
    if not separator:
        return successor_identity
    if fragment not in renames:
        raise SuccessorRepointError(
            f"{label} references unmapped legacy concept {fragment!r}"
        )
    return f"{successor_identity}#{renames[fragment]}"


@dataclass(frozen=True, slots=True)
class RepointRewrite:
    """One authenticated dependent postimage."""

    path: PurePosixPath
    before_sha256: str
    after_sha256: str
    raw: bytes
    replacements: tuple[dict[str, object], ...]

    def as_receipt_entry(self) -> dict[str, object]:
        return {
            "path": self.path.as_posix(),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "replacements": [dict(item) for item in self.replacements],
        }


def rewrite_repoint_file(
    raw: bytes,
    *,
    primary: bool,
    legacy_identity: str,
    successor_identity: str,
    successor_sha256: str,
    renames: Mapping[str, str],
    label: str,
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    """Rewrite one dependent file by exact tokens and prove the postimage.

    The rewrite touches exactly five surfaces on a primary: the module import
    list, ``module.deferred_outputs[].blocked_by``, proof import ``target`` /
    ``output`` / ``hash``, and unquoted identifier-bounded formula symbols.  On a
    companion test it touches only full-reference mapping keys.  Every other
    occurrence of the legacy identity or a mapped concept is a hard refusal.
    """

    if _SHA256.fullmatch(successor_sha256) is None:
        raise SuccessorRepointError("successor digest is not a SHA-256 hex string")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuccessorRepointError(f"{label} is not UTF-8") from exc
    payload = _load_module(raw, label=label) if primary else yaml.safe_load(text)
    try:
        root = yaml.compose(text)
    except (yaml.YAMLError, RecursionError) as exc:
        raise SuccessorRepointError(f"{label} is not composable YAML") from exc
    if root is None:
        raise SuccessorRepointError(f"{label} is empty")
    value_nodes, key_nodes = _scalar_nodes(root)

    edits: dict[int, tuple[int, int, str]] = {}
    authorized: set[int] = set()
    counts: dict[tuple[str, str], int] = {}
    hash_retargets = 0
    expected = yaml.safe_load(text)

    def record(old: str, new: str) -> None:
        counts[(old, new)] = counts.get((old, new), 0) + 1

    def replace_span(node: ScalarNode, replacement: str) -> None:
        start = node.start_mark.index
        end = node.end_mark.index
        if start in edits:  # pragma: no cover - paths are unique
            raise SuccessorRepointError(f"{label} has an ambiguous rewrite span")
        edits[start] = (start, end, replacement)

    def reference_node(
        path: tuple[object, ...], value: object, *, field: str
    ) -> str | None:
        node = value_nodes.get(path)
        if node is None:
            raise SuccessorRepointError(f"{label} {field} is not a scalar")
        authorized.add(id(node))
        if not isinstance(value, str):
            raise SuccessorRepointError(f"{label} {field} is not a string")
        mapped = _map_reference(
            value,
            legacy_identity=legacy_identity,
            successor_identity=successor_identity,
            renames=renames,
            label=f"{label} {field}",
        )
        if mapped is None or mapped == value:
            return None
        replace_span(node, _emit_token(node, mapped))
        record(value, mapped)
        return mapped

    if primary:
        assert isinstance(payload, dict)
        imports = payload.get("imports")
        if imports is not None and not isinstance(imports, list):
            raise SuccessorRepointError(f"{label} imports are malformed")
        for index, item in enumerate(imports or ()):
            mapped = reference_node(("imports", index), item, field=f"imports[{index}]")
            if mapped is not None:
                expected["imports"][index] = mapped
        if isinstance(imports, list) and len(set(expected.get("imports") or ())) != len(
            expected.get("imports") or ()
        ):
            raise SuccessorRepointError(
                f"{label} would import the successor module twice"
            )

        module = payload.get("module")
        deferred = module.get("deferred_outputs") if isinstance(module, dict) else None
        if isinstance(deferred, list):
            for entry_index, entry in enumerate(deferred):
                blocked = entry.get("blocked_by") if isinstance(entry, dict) else None
                if not isinstance(blocked, list):
                    continue
                for item_index, item in enumerate(blocked):
                    path = (
                        "module",
                        "deferred_outputs",
                        entry_index,
                        "blocked_by",
                        item_index,
                    )
                    field = (
                        f"module.deferred_outputs[{entry_index}]"
                        f".blocked_by[{item_index}]"
                    )
                    mapped = reference_node(path, item, field=field)
                    if mapped is not None:
                        expected["module"]["deferred_outputs"][entry_index][
                            "blocked_by"
                        ][item_index] = mapped

        rules = payload.get("rules")
        if not isinstance(rules, list):
            raise SuccessorRepointError(f"{label} declares no rules list")
        for rule_index, rule in enumerate(rules):
            versions = rule.get("versions") if isinstance(rule, dict) else None
            if isinstance(versions, list):
                for version_index, version in enumerate(versions):
                    formula = (
                        version.get("formula") if isinstance(version, dict) else None
                    )
                    if not isinstance(formula, str):
                        continue
                    path = ("rules", rule_index, "versions", version_index, "formula")
                    node = value_nodes.get(path)
                    if node is None:  # pragma: no cover - structure guarantees it
                        raise SuccessorRepointError(f"{label} has a non-scalar formula")
                    authorized.add(id(node))
                    rewritten_value = formula
                    for old, new in sorted(renames.items()):
                        rewritten_value = replace_formula_identifier(
                            rewritten_value, old, new
                        )
                    if rewritten_value == formula:
                        continue
                    if node.style in {"'", '"'}:
                        raise SuccessorRepointError(
                            f"{label} rules[{rule_index}].versions[{version_index}]"
                            ".formula is a quoted YAML scalar; the repoint rewrite "
                            "only rewrites plain or block formula scalars"
                        )
                    span = text[node.start_mark.index : node.end_mark.index]
                    rewritten_span = span
                    for old, new in sorted(renames.items()):
                        uses = len(_formula_symbol_uses(formula, old))
                        if not uses:
                            continue
                        rewritten_span = replace_formula_identifier(
                            rewritten_span, old, new
                        )
                        for _ in range(uses):
                            record(old, new)
                    replace_span(node, rewritten_span)
                    expected["rules"][rule_index]["versions"][version_index][
                        "formula"
                    ] = rewritten_value

            atoms = (
                rule.get("metadata", {}).get("proof", {}).get("atoms")
                if isinstance(rule, dict) and isinstance(rule.get("metadata"), dict)
                else None
            )
            if not isinstance(atoms, list):
                continue
            for atom_index, atom in enumerate(atoms):
                imported = atom.get("import") if isinstance(atom, dict) else None
                if not isinstance(imported, dict):
                    continue
                base = (
                    "rules",
                    rule_index,
                    "metadata",
                    "proof",
                    "atoms",
                    atom_index,
                    "import",
                )
                field = f"rules[{rule_index}].metadata.proof.atoms[{atom_index}].import"
                target = imported.get("target")
                mapped = reference_node(
                    base + ("target",), target, field=f"{field}.target"
                )
                if mapped is None:
                    continue
                expected_atom = expected["rules"][rule_index]["metadata"]["proof"][
                    "atoms"
                ][atom_index]["import"]
                expected_atom["target"] = mapped
                output = imported.get("output")
                if output is not None:
                    output_node = value_nodes.get(base + ("output",))
                    if output_node is None or not isinstance(output, str):
                        raise SuccessorRepointError(
                            f"{label} {field}.output is not a scalar string"
                        )
                    authorized.add(id(output_node))
                    if output in renames:
                        replace_span(
                            output_node, _emit_token(output_node, renames[output])
                        )
                        record(output, renames[output])
                        expected_atom["output"] = renames[output]
                    elif output in set(renames.values()):
                        raise SuccessorRepointError(
                            f"{label} {field}.output already names a successor concept"
                        )
                if "hash" not in imported:
                    raise SuccessorRepointError(
                        f"{label} {field} has no hash to retarget"
                    )
                hash_node = value_nodes.get(base + ("hash",))
                if hash_node is None:
                    raise SuccessorRepointError(f"{label} {field}.hash is not a scalar")
                authorized.add(id(hash_node))
                new_hash = f"sha256:{successor_sha256}"
                if imported.get("hash") != new_hash:
                    replace_span(hash_node, _emit_token(hash_node, new_hash))
                    hash_retargets += 1
                expected_atom["hash"] = new_hash
    else:
        for path, node in sorted(key_nodes.items(), key=lambda item: str(item[0])):
            value = node.value
            if not isinstance(value, str):
                continue
            mapped = _map_reference(
                value,
                legacy_identity=legacy_identity,
                successor_identity=successor_identity,
                renames=renames,
                label=f"{label} key {'.'.join(str(part) for part in path)}",
            )
            authorized.add(id(node))
            if mapped is None or mapped == value:
                continue
            replace_span(node, _emit_token(node, mapped))
            record(value, mapped)
        expected = _replace_mapping_keys(
            expected,
            legacy_identity=legacy_identity,
            successor_identity=successor_identity,
            renames=renames,
        )

    legacy_pattern = _durable_pattern(legacy_identity)
    concept_patterns = {old: _identifier_pattern(old) for old in renames}
    for path, node in list(value_nodes.items()) + list(key_nodes.items()):
        if id(node) in authorized:
            continue
        value = node.value
        if not isinstance(value, str):
            continue
        location = ".".join(str(part) for part in path) or "<root>"
        if legacy_pattern.search(value) is not None:
            raise SuccessorRepointError(
                f"{label} references the legacy module outside a rewritable surface: "
                f"{location}"
            )
        for old, pattern in concept_patterns.items():
            if pattern.search(value) is not None:
                raise SuccessorRepointError(
                    f"{label} names legacy concept {old!r} outside a rewritable "
                    f"surface: {location}"
                )

    if not edits:
        raise SuccessorRepointError(f"{label} has no legacy reference to rewrite")

    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(edits.values()):
        if start < cursor:  # pragma: no cover - spans are disjoint by construction
            raise SuccessorRepointError(f"{label} has overlapping rewrite spans")
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])
    rewritten = "".join(pieces).encode("utf-8")

    try:
        actual = yaml.safe_load(rewritten.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise SuccessorRepointError(
            f"{label} repoint rewrite produced invalid YAML"
        ) from exc
    if actual != expected:
        raise SuccessorRepointError(
            f"{label} repoint rewrite changed an unauthorized YAML surface"
        )
    if legacy_pattern.search(rewritten.decode("utf-8")) is not None:
        raise SuccessorRepointError(
            f"{label} still references the legacy module after the rewrite"
        )

    replacements = [
        {"from": old, "to": new, "count": count}
        for (old, new), count in sorted(counts.items())
    ]
    if hash_retargets:
        replacements.append(
            {"operation": "retarget_proof_import_hash", "count": hash_retargets}
        )
    return rewritten, tuple(replacements)


def _replace_mapping_keys(
    value: object,
    *,
    legacy_identity: str,
    successor_identity: str,
    renames: Mapping[str, str],
) -> object:
    if isinstance(value, dict):
        result: dict[object, object] = {}
        for key, item in value.items():
            mapped = (
                _map_reference(
                    key,
                    legacy_identity=legacy_identity,
                    successor_identity=successor_identity,
                    renames=renames,
                    label="companion test key",
                )
                if isinstance(key, str)
                else None
            )
            result[mapped if mapped is not None else key] = _replace_mapping_keys(
                item,
                legacy_identity=legacy_identity,
                successor_identity=successor_identity,
                renames=renames,
            )
        return result
    if isinstance(value, list):
        return [
            _replace_mapping_keys(
                item,
                legacy_identity=legacy_identity,
                successor_identity=successor_identity,
                renames=renames,
            )
            for item in value
        ]
    return value


# ---------------------------------------------------------------------------
# Reference inventory
# ---------------------------------------------------------------------------


def repoint_reference_inventory_issues(
    files: Mapping[str, bytes],
    *,
    request: RepointRequest,
) -> list[str]:
    """Fail closed on any legacy reference the transaction does not own.

    ``files`` maps repo-relative paths to their clean-HEAD bytes.  Protected
    RuleSpec references must belong to a declared dependent; ``programs/``
    references must belong to a declared ProgramSpec scope update.
    """

    identity = request.legacy_identity
    legacy_path = request.legacy_primary.as_posix()
    legacy_companion = request.legacy_companion.as_posix()
    scope_path = request.legacy_scope_path
    owned = {legacy_path, legacy_companion}
    declared_dependents = {item.as_posix() for item in request.dependents}
    declared_dependents |= {
        companion_of(item).as_posix() for item in request.dependents
    }
    declared_specs = {
        item.program_spec.as_posix() for item in request.program_scope_updates
    }

    identity_pattern = _durable_pattern(identity)
    path_pattern = _durable_pattern(legacy_path)
    scope_pattern = _durable_pattern(scope_path)

    issues: list[str] = []
    for path in sorted(files):
        if path in owned:
            continue
        relative = PurePosixPath(path)
        parts = relative.parts
        protected = (
            len(parts) >= 3
            and _JURISDICTION.fullmatch(parts[0]) is not None
            and parts[1] in PROTECTED_CONTENT_ROOTS
            and relative.suffix in {".yaml", ".yml"}
        )
        program = parts[0] == PROGRAM_SPEC_ROOT or (
            len(parts) >= 2 and parts[1] == PROGRAM_SPEC_ROOT
        )
        if not protected and not program:
            continue
        try:
            text = files[path].decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{path} is not UTF-8 and cannot be inventoried")
            continue
        if protected:
            if (
                identity_pattern.search(text) is None
                and path_pattern.search(text) is None
            ):
                continue
            if path not in declared_dependents:
                issues.append(
                    f"{path} references the legacy module but is not a declared "
                    "dependent"
                )
        else:
            if (
                scope_pattern.search(text) is None
                and identity_pattern.search(text) is None
                and path_pattern.search(text) is None
            ):
                continue
            if path not in declared_specs:
                issues.append(
                    f"{path} lists the legacy module but is not a declared "
                    "program_scope_updates entry"
                )
    for path in sorted(declared_dependents):
        if path.endswith(_TEST_SUFFIX):
            continue
        if path not in files:
            issues.append(f"declared dependent is not tracked at clean HEAD: {path}")
    for path in sorted(declared_specs):
        if path not in files:
            issues.append(f"declared ProgramSpec is not tracked at clean HEAD: {path}")
    return issues


# ---------------------------------------------------------------------------
# Repoint-only metadata reconciliations
# ---------------------------------------------------------------------------


def reconcile_upstream_source_check_baseline(
    raw: bytes, *, legacy_path: str
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    """Drop the retired module from the plain-text upstream-check allowlist."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuccessorRepointError(
            "upstream source-check baseline is not UTF-8"
        ) from exc
    lines = text.splitlines(keepends=True)
    removed = [line for line in lines if line.strip() == legacy_path]
    if len(removed) != 1:
        raise SuccessorRepointError(
            "upstream source-check baseline does not list the legacy module exactly "
            "once"
        )
    rewritten = "".join(line for line in lines if line not in removed).encode("utf-8")
    if _durable_pattern(legacy_path).search(rewritten.decode("utf-8")) is not None:
        raise SuccessorRepointError(
            "upstream source-check baseline still names the legacy module"
        )
    return rewritten, ({"operation": "remove_upstream_source_check_entry", "count": 1},)


def reconcile_money_atom_ratchet(
    raw: bytes, *, legacy_path: str
) -> tuple[bytes, tuple[dict[str, object], ...]]:
    """Drop the retired module's informational backlog comment line.

    ``total_allowed`` is an upper bound, so removing a backlog entry can only
    reduce the observed count.  The ratchet scalar is deliberately not lowered
    here: that is a separate, reviewable tightening.
    """

    try:
        text = raw.decode("utf-8")
        before = yaml.safe_load(text)
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise SuccessorRepointError("money-atom ratchet is not valid YAML") from exc
    lines = text.splitlines(keepends=True)
    pattern = re.compile(rf"^#\s+{re.escape(legacy_path)}:\s*[0-9]+\s*$")
    removed = [line for line in lines if pattern.fullmatch(line.rstrip("\r\n"))]
    if len(removed) != 1:
        raise SuccessorRepointError(
            "money-atom ratchet does not carry exactly one informational entry for "
            "the legacy module"
        )
    rewritten = "".join(line for line in lines if line not in removed).encode("utf-8")
    try:
        after = yaml.safe_load(rewritten.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise SuccessorRepointError(
            "money-atom ratchet removal produced invalid YAML"
        ) from exc
    if after != before:
        raise SuccessorRepointError(
            "money-atom ratchet removal changed a declared value"
        )
    if _durable_pattern(legacy_path).search(rewritten.decode("utf-8")) is not None:
        raise SuccessorRepointError("money-atom ratchet still names the legacy module")
    return rewritten, ({"operation": "remove_money_atom_backlog_entry", "count": 1},)


# ---------------------------------------------------------------------------
# Receipt identity
# ---------------------------------------------------------------------------


def receipt_identity_payload(
    *,
    request_sha256: str,
    base_commit: str,
    base_tree: str,
    legacy_manifest_sha256: str,
    successor_manifest_sha256: str,
    legacy_files: Sequence[Mapping[str, object]],
    successor_files: Sequence[Mapping[str, object]],
    dependents: Sequence[Mapping[str, object]],
    concept_proofs: Sequence[Mapping[str, object]],
    metadata_reconciliations: Sequence[Mapping[str, object]],
    program_scope_reconciliations: Sequence[Mapping[str, object]],
    semantics: Mapping[str, object],
) -> dict[str, object]:
    """Return the deterministic identity payload hashed into the receipt name."""

    return {
        "schema": RECEIPT_SCHEMA,
        "request_sha256": request_sha256,
        "base_commit": base_commit,
        "base_tree": base_tree,
        "legacy_manifest_sha256": legacy_manifest_sha256,
        "successor_manifest_sha256": successor_manifest_sha256,
        "legacy_files": [dict(item) for item in legacy_files],
        "successor_files": [dict(item) for item in successor_files],
        "dependents": [dict(item) for item in dependents],
        "concept_proofs": [dict(item) for item in concept_proofs],
        "metadata_reconciliations": [dict(item) for item in metadata_reconciliations],
        "program_scope_reconciliations": [
            dict(item) for item in program_scope_reconciliations
        ],
        "semantics": dict(semantics),
    }


def receipt_identity_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
