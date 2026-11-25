"""Dry-run check: can a signed manifest's source attestation carry to a new release?

The guard requires ``source_attestation.corpus_release`` (and its content and
selector digests) to equal the repository's pinned release exactly. A corpus
pin move therefore turns every signed manifest into a finding, even when the
new release serves the same source bytes. This module answers, per manifest,
whether a weaker statement still holds after the move:

    every corpus row that the encoder could read for the unit, and every
    corpus row that validation reads for the module, is byte-identical in the
    new release B and resolves there as it did in the attested release A.

It checks, from signed data:

1. A is one of the supplied verified release objects, and the attestation's
   release identity and provisions digests match it.
2. B holds the unit's provisions artifact at the attested sha256.
3. Closure: for every ``(jurisdiction, document_class)`` pair of the unit, of
   its component rows and of every corpus citation in the module files that
   the manifest binds by sha256 (proof atoms, interpretation sources,
   ``source_verification``), each provisions artifact that A held for the pair
   is in B at the same sha256.
4. Re-resolution in a materialized corpus of B: the unit resolves to the same
   attestation (except the three release identity fields), the same
   generation input, no amendment documents and no amendment request; and
   each cited path resolves without error to rows in artifacts that are
   byte-identical in A and B.

Rule 4 is mandatory when B adds a scope to any closed pair: an added scope can
shadow a citation, re-split a section, add an amendment, activate a dormant
row or add a descendant to a composed container, all without changing an
artifact of A. Without a corpus such an attestation is ``not-verifiable``.
Only when B adds nothing to the closed pairs may rules 1 to 3 alone give
``carried`` (with ``resolution_checked: false``).

Nothing calls this module from validation. ``axiom-encode
corpus-attestation-carry`` prints the verdicts; it changes no file and no
check result. A guard that adopts the rule must verify A with the
organization's trust root only, fetch A by the manifest's name and content
sha256, read any list of admissible releases from the protected base, and
apply the carry only in a pull request that changes nothing but the corpus
pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.corpus_release import (
    RELEASE_OBJECT_PUBLIC_KEY_ENV,
    CorpusReleaseObjectError,
    VerifiedCorpusReleaseObject,
    verify_release_object,
)

RELEASE_IDENTITY_FIELDS = (
    "corpus_release",
    "corpus_release_content_sha256",
    "corpus_release_selector_sha256",
)
DOCUMENT_CLASSES = frozenset(
    {
        "district-plan",
        "statute",
        "regulation",
        "guidance",
        "policy",
        "manual",
        "form",
        "rulemaking",
        "other",
    }
)
_PROVISIONS_PREFIX = "data/corpus/provisions/"
_PROVISIONS_SUFFIX = ".jsonl"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_RULESPEC_BYTES = 10 * 1024 * 1024
_MAX_RELEASE_OBJECT_BYTES = 64 * 1024 * 1024
_ATTESTATION_LOCATIONS = ("", "retired_manifest", "replacement_manifest")


class CarryStatus(str, Enum):
    """The verdict for one attestation against the new release."""

    CURRENT = "current"
    CARRIED = "carried"
    NOT_VERIFIABLE = "not-verifiable"
    UNIT_CHANGED = "unit-changed"
    UNIT_MISSING = "unit-missing"
    CONTEXT_CHANGED = "context-changed"
    NOT_IN_LINEAGE = "not-in-lineage"
    MALFORMED = "malformed"
    NO_ATTESTATION = "no-attestation"


@dataclass(frozen=True)
class ResolvedUnitEvidence:
    """What resolving the requested unit citation in the new release produced."""

    attestation: Mapping[str, Any]
    generation_input_sha256: str
    has_amendment_context: bool


@dataclass(frozen=True)
class ResolvedCitationEvidence:
    """The provisions files that one cited path resolved to in the new release."""

    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CarryVerdict:
    status: CarryStatus
    reasons: tuple[str, ...] = ()
    resolution_checked: bool = False
    pair_gained_scopes: tuple[str, ...] = ()


UnitResolver = Callable[[str], ResolvedUnitEvidence]
CitationResolver = Callable[[str], ResolvedCitationEvidence]


def _scope_pair(provision_file: object) -> tuple[str, str] | None:
    """Return ``(jurisdiction, document_class)`` of a canonical provisions path."""

    if not isinstance(provision_file, str):
        return None
    if not provision_file.startswith(_PROVISIONS_PREFIX):
        return None
    if not provision_file.endswith(_PROVISIONS_SUFFIX):
        return None
    parts = provision_file[len(_PROVISIONS_PREFIX) : -len(_PROVISIONS_SUFFIX)]
    components = parts.split("/")
    if len(components) != 3 or any(not item for item in components):
        return None
    if components[1] not in DOCUMENT_CLASSES:
        return None
    return components[0], components[1]


def _citation_pair(citation_path: object) -> tuple[str, str] | None:
    """Return ``(jurisdiction, document_class)`` of a canonical citation path."""

    if not isinstance(citation_path, str):
        return None
    components = citation_path.split("/")
    if len(components) < 2 or any(not item for item in components):
        return None
    if components[1] not in DOCUMENT_CLASSES:
        return None
    return components[0], components[1]


def _provisions_by_path(release: VerifiedCorpusReleaseObject) -> dict[str, str]:
    return {
        artifact.path: artifact.sha256
        for artifact in release.artifacts
        if artifact.artifact_class == "provisions"
    }


def _scopes_in_pairs(
    release: VerifiedCorpusReleaseObject, pairs: set[tuple[str, str]]
) -> set[str]:
    return {
        f"{scope.jurisdiction}/{scope.document_class}/{scope.version}"
        for scope in release.scopes
        if (scope.jurisdiction, scope.document_class) in pairs
    }


def _attested_files(attestation: Mapping[str, Any]) -> list[tuple[str, str]] | None:
    """Return every ``(provision_file, sha256)`` the attestation binds."""

    files: list[tuple[str, str]] = []
    path = attestation.get("provision_file")
    digest = attestation.get("provision_file_sha256")
    row = attestation.get("row")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    if not isinstance(row, Mapping):
        return None
    if row.get("provision_file") != path or row.get("provision_file_sha256") != digest:
        return None
    files.append((path, digest))
    components = attestation.get("component_rows", [])
    if not isinstance(components, list):
        return None
    for component in components:
        if not isinstance(component, Mapping):
            return None
        component_path = component.get("provision_file")
        component_digest = component.get("provision_file_sha256")
        if not isinstance(component_path, str) or not isinstance(component_digest, str):
            return None
        files.append((component_path, component_digest))
    return files


def classify_attestation_carry(
    attestation: Mapping[str, Any],
    *,
    target: VerifiedCorpusReleaseObject,
    attested_releases: Iterable[VerifiedCorpusReleaseObject],
    cited_paths: Sequence[str] = (),
    resolve: UnitResolver | None = None,
    resolve_citation: CitationResolver | None = None,
) -> CarryVerdict:
    """Decide whether one source attestation carries to ``target``.

    ``attested_releases`` are verified release objects the attestation may
    name. ``cited_paths`` are the corpus citations of the module files that
    the manifest binds. ``resolve`` and ``resolve_citation`` resolve in a
    materialized corpus of ``target``; rule 4 needs both.
    """

    if not isinstance(attestation, Mapping):
        return CarryVerdict(CarryStatus.MALFORMED, ("attestation is not an object",))
    name = attestation.get("corpus_release")
    content_sha256 = attestation.get("corpus_release_content_sha256")
    selector_sha256 = attestation.get("corpus_release_selector_sha256")
    if (
        name == target.name
        and content_sha256 == target.content_sha256
        and selector_sha256 == target.selector_sha256
    ):
        return CarryVerdict(CarryStatus.CURRENT)

    files = _attested_files(attestation)
    if not files:
        return CarryVerdict(
            CarryStatus.MALFORMED,
            ("attestation does not bind its provisions files consistently",),
        )
    pairs = {_scope_pair(path) for path, _digest in files}
    if None in pairs:
        return CarryVerdict(
            CarryStatus.MALFORMED,
            ("attestation names a non-canonical provisions path",),
        )
    cited_pairs = {_citation_pair(path) for path in cited_paths}
    if None in cited_pairs:
        return CarryVerdict(
            CarryStatus.MALFORMED,
            ("a bound module cites a non-canonical corpus citation path",),
        )
    closed_pairs: set[tuple[str, str]] = {
        pair for pair in pairs | cited_pairs if pair is not None
    }

    attested = next(
        (
            release
            for release in attested_releases
            if release.name == name and release.content_sha256 == content_sha256
        ),
        None,
    )
    if attested is None:
        return CarryVerdict(
            CarryStatus.NOT_IN_LINEAGE,
            (f"no verified release object supplied for attested release {name!r}",),
        )
    if selector_sha256 != attested.selector_sha256:
        return CarryVerdict(
            CarryStatus.NOT_IN_LINEAGE,
            ("attested selector sha256 does not match the attested release",),
        )
    old_files = _provisions_by_path(attested)
    for path, digest in files:
        if old_files.get(path) != digest:
            return CarryVerdict(
                CarryStatus.NOT_IN_LINEAGE,
                (f"attested release does not hold {path} at the attested sha256",),
            )

    new_files = _provisions_by_path(target)
    unit_path, unit_digest = files[0]
    if unit_path not in new_files:
        return CarryVerdict(
            CarryStatus.UNIT_MISSING, (f"{unit_path} is not in the new release",)
        )
    if new_files[unit_path] != unit_digest:
        return CarryVerdict(
            CarryStatus.UNIT_CHANGED,
            (f"{unit_path} has other bytes in the new release",),
        )
    reasons: list[str] = []
    for path, digest in files[1:]:
        if new_files.get(path) != digest:
            reasons.append(f"component file {path} changed or left")
    for path, digest in sorted(old_files.items()):
        if _scope_pair(path) in closed_pairs and new_files.get(path) != digest:
            reasons.append(f"closed-pair file {path} changed or left")
    if reasons:
        return CarryVerdict(CarryStatus.CONTEXT_CHANGED, tuple(reasons))
    gained = tuple(
        sorted(
            _scopes_in_pairs(target, closed_pairs)
            - _scopes_in_pairs(attested, closed_pairs)
        )
    )

    if resolve is None or (cited_paths and resolve_citation is None):
        if gained:
            return CarryVerdict(
                CarryStatus.NOT_VERIFIABLE,
                (
                    "the new release adds scopes to a closed pair; "
                    "re-resolution in a materialized corpus is required",
                ),
                pair_gained_scopes=gained,
            )
        return CarryVerdict(CarryStatus.CARRIED)

    def checked(status: CarryStatus, *why: str) -> CarryVerdict:
        return CarryVerdict(
            status, tuple(why), resolution_checked=True, pair_gained_scopes=gained
        )

    requested = attestation.get("requested_corpus_citation_path")
    if not isinstance(requested, str) or not requested:
        return CarryVerdict(
            CarryStatus.MALFORMED,
            ("attestation has no requested citation path",),
            pair_gained_scopes=gained,
        )
    try:
        evidence = resolve(requested)
    except ValueError as exc:
        return checked(
            CarryStatus.UNIT_MISSING,
            f"requested citation does not resolve in the new release: {exc}",
        )
    expected = dict(evidence.attestation)
    for identity_field in RELEASE_IDENTITY_FIELDS:
        expected.pop(identity_field, None)
    changed = sorted(
        key for key, value in expected.items() if attestation.get(key) != value
    )
    if attestation.get("generation_input_sha256") != evidence.generation_input_sha256:
        changed.append("generation_input_sha256")
    if changed:
        return checked(
            CarryStatus.UNIT_CHANGED,
            *(f"re-resolved {key} differs" for key in changed),
        )
    if evidence.has_amendment_context:
        return checked(
            CarryStatus.CONTEXT_CHANGED,
            "the new release gives this unit amendment context",
        )
    for cited in sorted(set(cited_paths)):
        assert resolve_citation is not None
        try:
            cited_evidence = resolve_citation(cited)
        except ValueError as exc:
            reasons.append(f"cited path {cited} does not resolve: {exc}")
            continue
        if not cited_evidence.files:
            reasons.append(f"cited path {cited} resolved to no provisions file")
        for path, digest in cited_evidence.files:
            if old_files.get(path) != digest or new_files.get(path) != digest:
                reasons.append(
                    f"cited path {cited} resolves to {path}, "
                    "which is not byte-identical in both releases"
                )
    if reasons:
        return checked(CarryStatus.CONTEXT_CHANGED, *reasons)
    return checked(CarryStatus.CARRIED)


@dataclass(frozen=True)
class ManifestAttestation:
    """One manifest location and what the dry run needs from it."""

    label: str
    attestation: Any
    cited_paths: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()


def _citations(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "corpus_citation_path" and isinstance(item, str):
                yield item
            elif key == "corpus_citation_paths" and isinstance(item, list):
                yield from (entry for entry in item if isinstance(entry, str))
            else:
                yield from _citations(item)
    elif isinstance(value, list):
        for item in value:
            yield from _citations(item)


def _bound_citations(
    repo_root: Path, holder: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the corpus citations of the RuleSpec files a manifest binds."""

    citations: set[str] = set()
    problems: list[str] = []
    applied = holder.get("applied_files")
    if not isinstance(applied, list):
        return (), ("manifest has no applied_files list",)
    for item in applied:
        if not isinstance(item, Mapping) or item.get("deleted") is True:
            continue
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str) or not path.endswith(".yaml"):
            continue
        parts = path.split("/")
        if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            problems.append(f"applied file path {path!r} is not repository-relative")
            continue
        file_path = repo_root / path
        if file_path.is_symlink() or not file_path.is_file():
            problems.append(f"applied file {path} is not a regular file")
            continue
        if file_path.stat().st_size > _MAX_RULESPEC_BYTES:
            problems.append(f"applied file {path} is too large")
            continue
        raw = file_path.read_bytes()
        try:
            document = yaml.safe_load(raw)
        except (yaml.YAMLError, ValueError, RecursionError):
            document = None
        if not isinstance(document, Mapping) or document.get("format") != "rulespec/v1":
            continue
        if hashlib.sha256(raw).hexdigest() != digest:
            problems.append(f"applied RuleSpec file {path} differs from the manifest")
            continue
        citations.update(_citations(document))
    return tuple(sorted(citations)), tuple(problems)


def iter_manifest_attestations(manifests_root: Path) -> Iterable[ManifestAttestation]:
    """Yield every manifest under a tree, with or without an attestation."""

    repo_root = manifests_root.parent.parent
    for path in sorted(manifests_root.rglob("*.json")):
        relative = path.relative_to(manifests_root).as_posix()
        if path.is_symlink():
            yield ManifestAttestation(
                relative, None, problems=("manifest is a symlink",)
            )
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            yield ManifestAttestation(
                relative, None, problems=("manifest is too large",)
            )
            continue
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, ValueError):
            yield ManifestAttestation(
                relative, None, problems=("manifest is not valid JSON",)
            )
            continue
        if not isinstance(payload, dict):
            yield ManifestAttestation(
                relative, None, problems=("manifest is not a JSON object",)
            )
            continue
        found = False
        for location in _ATTESTATION_LOCATIONS:
            holder = payload.get(location) if location else payload
            if isinstance(holder, dict) and "source_attestation" in holder:
                found = True
                cited, problems = _bound_citations(repo_root, holder)
                yield ManifestAttestation(
                    f"{relative}#{location}" if location else relative,
                    holder["source_attestation"],
                    cited,
                    problems,
                )
        if not found:
            yield ManifestAttestation(relative, None)


def _load_verified_release(path: Path, public_key: str) -> VerifiedCorpusReleaseObject:
    if path.stat().st_size > _MAX_RELEASE_OBJECT_BYTES:
        raise CorpusReleaseObjectError(f"release object is too large: {path}")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise CorpusReleaseObjectError(f"release object is not an object: {path}")
    return verify_release_object(payload, public_key=public_key)


def _corpus_resolvers(
    corpus_root: Path, target: VerifiedCorpusReleaseObject, public_key: str
) -> tuple[UnitResolver, CitationResolver]:
    """Resolve citations through the encoder's resolvers over a local corpus."""

    from axiom_encode.corpus_resolver import (
        LocalCorpusRelease,
        resolve_local_corpus_source,
    )
    from axiom_encode.harness.evals import resolve_corpus_source_unit

    release = LocalCorpusRelease(
        corpus_root, target.name, target.content_sha256, public_key
    )

    def resolve(requested: str) -> ResolvedUnitEvidence:
        unit = resolve_corpus_source_unit(requested, release)
        body = unit.body.replace("\r\n", "\n").replace("\r", "\n")
        return ResolvedUnitEvidence(
            attestation=unit.source_attestation,
            generation_input_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            has_amendment_context=bool(unit.amendment_documents)
            or unit.amendment_request is not None,
        )

    def resolve_citation(citation_path: str) -> ResolvedCitationEvidence:
        source = resolve_local_corpus_source(citation_path, release)
        files = {(source.provision_file, source.provision_file_sha256)}
        files.update(
            (row.provision_file, row.provision_file_sha256)
            for row in source.component_rows
        )
        return ResolvedCitationEvidence(tuple(sorted(files)))

    return resolve, resolve_citation


def run_corpus_attestation_carry(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="axiom-encode corpus-attestation-carry",
        description=(
            "Dry run: list, per signed manifest, whether its source attestation "
            "carries to a new signed corpus release. Changes nothing."
        ),
    )
    parser.add_argument("--rulespec-repo", type=Path, required=True)
    parser.add_argument(
        "--to-release",
        type=Path,
        required=True,
        help="signed release object JSON of the release the repository would pin",
    )
    parser.add_argument(
        "--from-release",
        type=Path,
        action="append",
        default=[],
        help="signed release object JSON a manifest may attest (repeatable)",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        help="materialized corpus of --to-release; re-resolves each citation",
    )
    parser.add_argument(
        "--public-key",
        help=(
            f"corpus release public key (default: ${RELEASE_OBJECT_PUBLIC_KEY_ENV}); "
            "a guard must use the organization trust root instead"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv))

    public_key = args.public_key or os.environ.get(RELEASE_OBJECT_PUBLIC_KEY_ENV)
    if not public_key:
        parser.error(
            f"--public-key or {RELEASE_OBJECT_PUBLIC_KEY_ENV} is required to verify "
            "release objects"
        )
    try:
        target = _load_verified_release(args.to_release, public_key)
        attested = [
            _load_verified_release(path, public_key) for path in args.from_release
        ]
    except (OSError, ValueError) as exc:
        print(f"corpus-attestation-carry: {exc}", file=sys.stderr)
        return 2
    resolve: UnitResolver | None = None
    resolve_citation: CitationResolver | None = None
    if args.corpus_root is not None:
        resolve, resolve_citation = _corpus_resolvers(
            args.corpus_root, target, public_key
        )
    manifests_root = args.rulespec_repo / ".axiom" / "encoding-manifests"
    rows: list[dict[str, Any]] = []
    for entry in iter_manifest_attestations(manifests_root):
        if entry.problems:
            verdict = CarryVerdict(CarryStatus.MALFORMED, entry.problems)
        elif entry.attestation is None:
            verdict = CarryVerdict(CarryStatus.NO_ATTESTATION)
        else:
            verdict = classify_attestation_carry(
                entry.attestation,
                target=target,
                attested_releases=attested,
                cited_paths=entry.cited_paths,
                resolve=resolve,
                resolve_citation=resolve_citation,
            )
        attestation = entry.attestation
        rows.append(
            {
                "manifest": entry.label,
                "attested_release": attestation.get("corpus_release")
                if isinstance(attestation, Mapping)
                else None,
                "status": verdict.status.value,
                "resolution_checked": verdict.resolution_checked,
                "pair_gained_scopes": list(verdict.pair_gained_scopes),
                "cited_paths": len(entry.cited_paths),
                "reasons": list(verdict.reasons),
            }
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    if args.json:
        print(
            json.dumps(
                {"to_release": target.name, "counts": counts, "manifests": rows},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for row in rows:
            if row["status"] == CarryStatus.NO_ATTESTATION.value:
                continue
            reason = f" ({'; '.join(row['reasons'])})" if row["reasons"] else ""
            checked = str(row["resolution_checked"]).lower()
            print(
                f"{row['status']}\t{row['manifest']}\t"
                f"resolution_checked: {checked}\t"
                f"pair_gained_scopes: {len(row['pair_gained_scopes'])}{reason}"
            )
        summary = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
        print(f"{target.name}: {len(rows)} manifest entries: {summary}")
    uncarried = {
        status
        for status in counts
        if status
        not in {
            CarryStatus.CURRENT.value,
            CarryStatus.CARRIED.value,
            CarryStatus.NO_ATTESTATION.value,
        }
    }
    return 1 if uncarried else 0
