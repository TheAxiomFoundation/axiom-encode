"""Dry-run check: can a signed manifest's source attestation carry to a new release?

The guard requires ``source_attestation.corpus_release`` (and its content and
selector digests) to equal the repository's pinned release exactly. A corpus
pin move therefore turns every signed manifest into a finding, even when the
new release serves the same source bytes. This module answers, per manifest,
whether a weaker statement still holds after the move:

    the source unit that the module was encoded from, and the corpus rows the
    encoder could read beside it, are byte-identical in the new release.

It checks this from signed data only:

1. The attested release is one of the supplied signed release objects, and
   the attestation's release identity and provisions digests match it.
2. In the new signed release object, every provisions artifact that the
   attested release held for the unit's ``(jurisdiction, document_class)``
   pairs is present with the same sha256. New scopes may be added to a pair;
   no artifact of the pair may change or leave. (The resolver reads amendment
   context and parent rows from every active scope of the pair.)
3. Optionally, with a materialized corpus of the new release, the requested
   citation resolves to an attestation that is equal in every field except
   the three release identity fields, with the same generation input, and
   with no amendment context or amendment routing.

Nothing calls this module from validation. ``axiom-encode
corpus-attestation-carry`` prints the verdicts; it changes no file and no
check result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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
_PROVISIONS_PREFIX = "data/corpus/provisions/"
_PROVISIONS_SUFFIX = ".jsonl"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_RELEASE_OBJECT_BYTES = 64 * 1024 * 1024
_ATTESTATION_LOCATIONS = ("", "retired_manifest", "replacement_manifest")


class CarryStatus(str, Enum):
    """The verdict for one attestation against the new release."""

    CURRENT = "current"
    CARRIED = "carried"
    UNIT_CHANGED = "unit-changed"
    UNIT_MISSING = "unit-missing"
    CONTEXT_CHANGED = "context-changed"
    NOT_IN_LINEAGE = "not-in-lineage"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ResolvedUnitEvidence:
    """What resolving the requested citation in the new release produced."""

    attestation: Mapping[str, Any]
    generation_input_sha256: str
    has_amendment_context: bool


@dataclass(frozen=True)
class CarryVerdict:
    status: CarryStatus
    reasons: tuple[str, ...] = ()
    resolution_checked: bool = False


Resolver = Callable[[str], ResolvedUnitEvidence]


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
    return components[0], components[1]


def _provisions_by_path(release: VerifiedCorpusReleaseObject) -> dict[str, str]:
    return {
        artifact.path: artifact.sha256
        for artifact in release.artifacts
        if artifact.artifact_class == "provisions"
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
    resolve: Resolver | None = None,
) -> CarryVerdict:
    """Decide whether one source attestation carries to ``target``.

    ``attested_releases`` are verified release objects the attestation may
    name. ``resolve`` resolves a requested citation in a materialized corpus
    of ``target``; without it the verdict rests on artifact digests alone and
    ``resolution_checked`` is false.
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
        if _scope_pair(path) in pairs and new_files.get(path) != digest:
            reasons.append(f"same-pair file {path} changed or left")
    if reasons:
        return CarryVerdict(CarryStatus.CONTEXT_CHANGED, tuple(reasons))

    if resolve is None:
        return CarryVerdict(CarryStatus.CARRIED)
    requested = attestation.get("requested_corpus_citation_path")
    if not isinstance(requested, str) or not requested:
        return CarryVerdict(
            CarryStatus.MALFORMED, ("attestation has no requested citation path",)
        )
    try:
        evidence = resolve(requested)
    except ValueError as exc:
        return CarryVerdict(
            CarryStatus.UNIT_MISSING,
            (f"requested citation does not resolve in the new release: {exc}",),
            resolution_checked=True,
        )
    expected = dict(evidence.attestation)
    for field in RELEASE_IDENTITY_FIELDS:
        expected.pop(field, None)
    changed = sorted(
        field for field, value in expected.items() if attestation.get(field) != value
    )
    if attestation.get("generation_input_sha256") != evidence.generation_input_sha256:
        changed.append("generation_input_sha256")
    if changed:
        return CarryVerdict(
            CarryStatus.UNIT_CHANGED,
            tuple(f"re-resolved {field} differs" for field in changed),
            resolution_checked=True,
        )
    if evidence.has_amendment_context:
        return CarryVerdict(
            CarryStatus.CONTEXT_CHANGED,
            ("the new release gives this unit amendment context",),
            resolution_checked=True,
        )
    return CarryVerdict(CarryStatus.CARRIED, resolution_checked=True)


def iter_manifest_attestations(
    manifests_root: Path,
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield ``(label, attestation)`` for each attestation under a manifest tree."""

    for path in sorted(manifests_root.rglob("*.json")):
        if not path.is_file() or path.is_symlink():
            continue
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            yield path.relative_to(manifests_root).as_posix(), {}
            continue
        try:
            payload = json.loads(path.read_bytes())
        except (OSError, ValueError):
            yield path.relative_to(manifests_root).as_posix(), {}
            continue
        if not isinstance(payload, dict):
            continue
        relative = path.relative_to(manifests_root).as_posix()
        for location in _ATTESTATION_LOCATIONS:
            holder = payload.get(location) if location else payload
            if isinstance(holder, dict) and "source_attestation" in holder:
                label = f"{relative}#{location}" if location else relative
                yield label, holder["source_attestation"]


def _load_verified_release(path: Path, public_key: str) -> VerifiedCorpusReleaseObject:
    if path.stat().st_size > _MAX_RELEASE_OBJECT_BYTES:
        raise CorpusReleaseObjectError(f"release object is too large: {path}")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise CorpusReleaseObjectError(f"release object is not an object: {path}")
    return verify_release_object(payload, public_key=public_key)


def _corpus_resolver(
    corpus_root: Path, target: VerifiedCorpusReleaseObject, public_key: str
) -> Resolver:
    """Resolve citations through the encoder's resolver over a local corpus."""

    import hashlib

    from axiom_encode.corpus_resolver import LocalCorpusRelease
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

    return resolve


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
        help=f"corpus release public key (default: ${RELEASE_OBJECT_PUBLIC_KEY_ENV})",
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
    resolve = (
        _corpus_resolver(args.corpus_root, target, public_key)
        if args.corpus_root is not None
        else None
    )
    manifests_root = args.rulespec_repo / ".axiom" / "encoding-manifests"
    rows: list[dict[str, Any]] = []
    for label, attestation in iter_manifest_attestations(manifests_root):
        verdict = classify_attestation_carry(
            attestation,
            target=target,
            attested_releases=[target, *attested],
            resolve=resolve,
        )
        rows.append(
            {
                "manifest": label,
                "attested_release": attestation.get("corpus_release")
                if isinstance(attestation, Mapping)
                else None,
                "status": verdict.status.value,
                "resolution_checked": verdict.resolution_checked,
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
            reason = f" ({'; '.join(row['reasons'])})" if row["reasons"] else ""
            print(f"{row['status']}\t{row['manifest']}{reason}")
        summary = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()))
        print(f"{target.name}: {len(rows)} attestations: {summary}")
    uncarried = {
        status
        for status in counts
        if status not in {CarryStatus.CURRENT.value, CarryStatus.CARRIED.value}
    }
    return 1 if uncarried else 0
