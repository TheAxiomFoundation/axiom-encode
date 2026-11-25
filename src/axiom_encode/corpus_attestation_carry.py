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
   its component rows, of every corpus citation in the module files that the
   manifest binds by sha256 (proof atoms, interpretation sources,
   ``source_verification``) and of every corpus candidate of the modules'
   ``imports``, each provisions artifact that A held for the pair is in B at
   the same sha256.
4. Resolution in materialized corpora of both A and B: the unit, every cited
   path and every import candidate resolve identically in A and B (same
   route, same text, same rows in artifacts byte-identical in both); the
   unit's A resolution equals the manifest's attestation; the unit has no
   amendment documents and no amendment request; and no row of a closed-pair
   artifact names a scope that B adds (a dormant row).

Rule 4 is mandatory when B adds a scope to any closed pair: an added scope can
shadow a citation, re-split a section, add an amendment, activate a dormant
row or add a descendant to a composed container, all without changing an
artifact of A. Without both corpora such an attestation is
``not-verifiable``. Only when B adds nothing to the closed pairs may rules 1
to 3 alone give ``carried`` (with ``resolution_checked: false``).

Nothing calls this module from validation. ``axiom-encode
corpus-attestation-carry`` prints the verdicts and writes no file of the
RuleSpec repository. In a lock-file corpus checkout, loading a release places
verified provisions files the checkout's locks pin, as ``corpus-fetch`` does;
a plain materialized corpus is only read. A guard that adopts the rule must
verify A with the organization's trust root only (the retired release key for
A only, never for B), fetch A by the manifest's name and content sha256, read
any list of admissible releases from the protected base, and apply the carry
only in a pull request that changes nothing but the corpus pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode

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
_MAX_PROVISIONS_BYTES = 512 * 1024 * 1024
_ATTESTATION_LOCATIONS = ("", "retired_manifest", "replacement_manifest")
_YAML_SUFFIXES = (".yaml", ".yml")
_TEST_SUFFIXES = (".test.yaml", ".test.yml")


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
    """What resolving the requested unit citation in one release produced."""

    attestation: Mapping[str, Any]
    generation_input_sha256: str
    has_amendment_context: bool


@dataclass(frozen=True)
class ResolvedCitationEvidence:
    """How one cited path resolved in one release, and from which files."""

    files: tuple[tuple[str, str], ...]
    route: tuple[str, ...] = ()
    text_sha256: str = ""


@dataclass(frozen=True)
class CorpusView:
    """Resolution of one verified release over one materialized corpus."""

    unit: Callable[[str], ResolvedUnitEvidence]
    citation: Callable[[str], ResolvedCitationEvidence]
    row_scopes: Callable[[str], frozenset[str]]


class CitationNotFound(ValueError):
    """A cited path that the release does not hold at all."""


@dataclass(frozen=True)
class CarryVerdict:
    status: CarryStatus
    reasons: tuple[str, ...] = ()
    resolution_checked: bool = False
    pair_gained_scopes: tuple[str, ...] = ()


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
    """Return the pair of an exact canonical corpus citation path, else None."""

    from axiom_encode.corpus_resolver import (
        InvalidCorpusCitationError,
        require_canonical_corpus_citation_path,
    )

    if not isinstance(citation_path, str):
        return None
    try:
        if require_canonical_corpus_citation_path(citation_path) != citation_path:
            return None
    except (InvalidCorpusCitationError, ValueError):
        return None
    components = citation_path.split("/")
    if len(components) < 2 or components[1] not in DOCUMENT_CLASSES:
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


def _without_identity(attestation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in attestation.items()
        if key not in RELEASE_IDENTITY_FIELDS
    }


def classify_attestation_carry(
    attestation: Mapping[str, Any],
    *,
    target: VerifiedCorpusReleaseObject,
    attested_releases: Iterable[VerifiedCorpusReleaseObject],
    cited_paths: Sequence[str] = (),
    dependency_paths: Sequence[str] = (),
    view_a: CorpusView | None = None,
    view_b: CorpusView | None = None,
) -> CarryVerdict:
    """Decide whether one source attestation carries to ``target``.

    ``attested_releases`` are verified release objects the attestation may
    name. ``cited_paths`` are the corpus citations of the module files that
    the manifest binds; ``dependency_paths`` the corpus candidates of their
    imports (absent in both releases is allowed). ``view_a`` and ``view_b``
    resolve in materialized corpora of the attested release and of
    ``target``; rule 4 needs both.
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
    cited_pairs = {_citation_pair(path) for path in (*cited_paths, *dependency_paths)}
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
    closed_files = sorted(
        (path, digest)
        for path, digest in old_files.items()
        if _scope_pair(path) in closed_pairs
    )
    for path, digest in closed_files:
        if new_files.get(path) != digest:
            reasons.append(f"closed-pair file {path} changed or left")
    if reasons:
        return CarryVerdict(CarryStatus.CONTEXT_CHANGED, tuple(reasons))
    gained = tuple(
        sorted(
            _scopes_in_pairs(target, closed_pairs)
            - _scopes_in_pairs(attested, closed_pairs)
        )
    )

    if view_a is None or view_b is None:
        if dependency_paths:
            return CarryVerdict(
                CarryStatus.NOT_VERIFIABLE,
                (
                    "import candidates must be resolved in materialized corpora "
                    "of both releases",
                ),
                pair_gained_scopes=gained,
            )
        if gained:
            return CarryVerdict(
                CarryStatus.NOT_VERIFIABLE,
                (
                    "the new release adds scopes to a closed pair; resolution in "
                    "materialized corpora of both releases is required",
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
        unit_a = view_a.unit(requested)
    except ValueError as exc:
        return checked(
            CarryStatus.NOT_IN_LINEAGE,
            f"requested citation does not resolve in the attested release: {exc}",
        )
    attested_view = _without_identity(unit_a.attestation)
    if {key: attestation.get(key) for key in attested_view} != attested_view or (
        attestation.get("generation_input_sha256") != unit_a.generation_input_sha256
    ):
        return checked(
            CarryStatus.NOT_IN_LINEAGE,
            "the attestation does not match its resolution in the attested release",
        )
    try:
        unit_b = view_b.unit(requested)
    except ValueError as exc:
        return checked(
            CarryStatus.UNIT_MISSING,
            f"requested citation does not resolve in the new release: {exc}",
        )
    expected = _without_identity(unit_a.attestation)
    actual = _without_identity(unit_b.attestation)
    changed = sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    if unit_a.generation_input_sha256 != unit_b.generation_input_sha256:
        changed.append("generation_input_sha256")
    if changed:
        return checked(
            CarryStatus.UNIT_CHANGED,
            *(f"re-resolved {key} differs" for key in changed),
        )
    if unit_a.has_amendment_context or unit_b.has_amendment_context:
        return checked(CarryStatus.CONTEXT_CHANGED, "the unit has amendment context")

    unverifiable: list[str] = []
    references = [(path, False) for path in sorted(set(cited_paths))]
    references += [(path, True) for path in sorted(set(dependency_paths))]
    for reference, may_be_absent in references:
        outcomes = []
        for view in (view_a, view_b):
            try:
                outcomes.append(view.citation(reference))
            except CitationNotFound:
                outcomes.append("absent")
            except ValueError as exc:
                outcomes.append(f"error: {exc}")
        result_a, result_b = outcomes
        if result_a == "absent" and result_b == "absent" and may_be_absent:
            unverifiable.append(f"import candidate {reference} is in neither release")
            continue
        if not isinstance(result_a, ResolvedCitationEvidence) or not isinstance(
            result_b, ResolvedCitationEvidence
        ):
            reasons.append(f"{reference} does not resolve in both releases")
            continue
        if result_a != result_b:
            reasons.append(f"{reference} resolves differently in the two releases")
            continue
        if not result_b.files:
            reasons.append(f"{reference} resolved to no provisions file")
        for path, digest in result_b.files:
            if old_files.get(path) != digest or new_files.get(path) != digest:
                reasons.append(
                    f"{reference} resolves to {path}, "
                    "which is not byte-identical in both releases"
                )
    gained_set = set(gained)
    for path, _digest in closed_files:
        dormant = sorted(view_b.row_scopes(path) & gained_set)
        if dormant:
            reasons.append(
                f"{path} holds rows of a scope the new release adds: "
                + ", ".join(dormant)
            )
    if reasons:
        return checked(CarryStatus.CONTEXT_CHANGED, *reasons)
    if unverifiable:
        return checked(CarryStatus.NOT_VERIFIABLE, *unverifiable)
    return checked(CarryStatus.CARRIED)


@dataclass(frozen=True)
class ManifestAttestation:
    """One manifest location and what the dry run needs from it."""

    label: str
    attestation: Any
    cited_paths: tuple[str, ...] = ()
    dependency_paths: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()


class _DuplicateKey(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node: MappingNode, deep: bool = False):
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise _DuplicateKey("YAML merge key")
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in result:
                raise _DuplicateKey(f"duplicate key {key!r}")
        except TypeError as exc:
            raise _DuplicateKey("nonscalar key") from exc
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor("tag:yaml.org,2002:map", _construct_unique_mapping)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate key {key!r}")
        result[key] = value
    return result


def _citations(value: object, problems: list[str]) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "corpus_citation_path":
                if isinstance(item, str):
                    yield item
                else:
                    problems.append("corpus_citation_path is not a string")
            elif key == "corpus_citation_paths":
                if isinstance(item, list) and all(isinstance(x, str) for x in item):
                    yield from item
                else:
                    problems.append("corpus_citation_paths is not a list of strings")
            else:
                yield from _citations(item, problems)
    elif isinstance(value, list):
        for item in value:
            yield from _citations(item, problems)


_RULESPEC_ROOT_RE = re.compile(r"(rulespec-[a-z]{2})/[a-z]{2}(?:-[a-z0-9_]+)*")
_WAIVER_LEDGER_PATH = "known-validation-gaps.yaml"


def _attested_rulespec_repo(holder: Mapping[str, Any]) -> str | None:
    """Return the canonical RuleSpec repository named by the attestation."""

    attestation = holder.get("source_attestation")
    root = (
        attestation.get("rulespec_root") if isinstance(attestation, Mapping) else None
    )
    if not isinstance(root, str):
        return None
    match = _RULESPEC_ROOT_RE.fullmatch(root)
    return match.group(1) if match else None


def _import_candidates(
    rulespec_repo: str | None, document: Mapping[str, Any], problems: list[str]
) -> set[str]:
    imports = document.get("imports")
    if imports is None:
        return set()
    if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
        problems.append("imports is not a list of strings")
        return set()
    if not imports:
        return set()
    if rulespec_repo is None:
        # The candidate of a bare import depends on the repository's
        # jurisdiction; a checkout directory name is not evidence of it.
        problems.append(
            "has imports, but the attestation names no canonical rulespec_root"
        )
        return set()
    from axiom_encode.harness.dependency_stubs import (
        _candidate_corpus_paths_for_import_target,
    )

    candidates: set[str] = set()
    for target in imports:
        candidates.update(
            _candidate_corpus_paths_for_import_target(target, Path(rulespec_repo))
        )
    return candidates


def _bound_references(
    repo_root: Path, holder: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return the citations, import candidates and problems of bound YAML files."""

    citations: set[str] = set()
    dependencies: set[str] = set()
    problems: list[str] = []
    applied = holder.get("applied_files")
    if not isinstance(applied, list):
        return (), (), ("manifest has no applied_files list",)
    resolved_root = repo_root.resolve()
    rulespec_repo = _attested_rulespec_repo(holder)
    for item in applied:
        if not isinstance(item, Mapping):
            problems.append("an applied_files entry is not an object")
            continue
        path = item.get("path")
        digest = item.get("sha256")
        if not isinstance(path, str):
            problems.append("an applied_files entry has no string path")
            continue
        if not path.endswith(_YAML_SUFFIXES):
            if path.lower().endswith(_YAML_SUFFIXES):
                problems.append(f"applied file {path} has an unexpected suffix case")
            continue
        if item.get("deleted") is True:
            continue
        parts = path.split("/")
        if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            problems.append(f"applied file path {path!r} is not repository-relative")
            continue
        file_path = repo_root / path
        if (
            file_path.is_symlink()
            or not file_path.is_file()
            or not file_path.resolve().is_relative_to(resolved_root)
            or file_path.resolve() != resolved_root.joinpath(*parts)
        ):
            problems.append(f"applied file {path} is not a regular file")
            continue
        if file_path.stat().st_size > _MAX_RULESPEC_BYTES:
            problems.append(f"applied file {path} is too large")
            continue
        raw = file_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            problems.append(f"applied file {path} differs from the manifest")
            continue
        try:
            document = yaml.load(raw.decode("utf-8"), Loader=_UniqueLoader)
        except (UnicodeError, yaml.YAMLError, RecursionError, _DuplicateKey) as exc:
            problems.append(f"applied file {path} is not strict YAML: {exc}")
            continue
        local: list[str] = []
        if path.endswith(_TEST_SUFFIXES):
            citations.update(_citations(document, local))
        elif path == _WAIVER_LEDGER_PATH:
            pass
        elif not isinstance(document, Mapping):
            local.append("is not a YAML mapping")
        elif document.get("format") != "rulespec/v1":
            local.append(f"has unsupported format {document.get('format')!r}")
        else:
            citations.update(_citations(document, local))
            dependencies.update(_import_candidates(rulespec_repo, document, local))
        problems.extend(f"applied file {path} {problem}" for problem in local)
    return tuple(sorted(citations)), tuple(sorted(dependencies)), tuple(problems)


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
            payload = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
        except _DuplicateKey as exc:
            yield ManifestAttestation(
                relative, None, problems=(f"manifest JSON has a {exc}",)
            )
            continue
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
                cited, dependencies, problems = _bound_references(repo_root, holder)
                if not isinstance(holder["source_attestation"], dict):
                    problems = ("source_attestation is not an object", *problems)
                yield ManifestAttestation(
                    f"{relative}#{location}" if location else relative,
                    holder["source_attestation"],
                    cited,
                    dependencies,
                    problems,
                )
        if not found:
            yield ManifestAttestation(relative, None)


def _load_verified_release(
    path: Path, public_keys: Sequence[str]
) -> VerifiedCorpusReleaseObject:
    if path.stat().st_size > _MAX_RELEASE_OBJECT_BYTES:
        raise CorpusReleaseObjectError(f"release object is too large: {path}")
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise CorpusReleaseObjectError(f"release object is not an object: {path}")
    return verify_release_object(payload, public_key=list(public_keys))


def corpus_view(
    corpus_root: Path,
    release: VerifiedCorpusReleaseObject,
    public_keys: Sequence[str],
) -> CorpusView:
    """Resolve through the encoder's resolvers over one local corpus."""

    from axiom_encode.corpus_resolver import (
        CorpusSourceNotFoundError,
        LocalCorpusRelease,
        resolve_local_corpus_source,
    )
    from axiom_encode.harness.evals import resolve_corpus_source_unit

    local = LocalCorpusRelease(
        corpus_root, release.name, release.content_sha256, list(public_keys)
    )

    def unit(requested: str) -> ResolvedUnitEvidence:
        resolved = resolve_corpus_source_unit(requested, local)
        body = resolved.body.replace("\r\n", "\n").replace("\r", "\n")
        return ResolvedUnitEvidence(
            attestation=resolved.source_attestation,
            generation_input_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            has_amendment_context=bool(resolved.amendment_documents)
            or resolved.amendment_request is not None,
        )

    def citation(citation_path: str) -> ResolvedCitationEvidence:
        try:
            source = resolve_local_corpus_source(citation_path, local)
        except CorpusSourceNotFoundError as exc:
            raise CitationNotFound(str(exc)) from exc
        files = {(source.provision_file, source.provision_file_sha256)}
        files.update(
            (row.provision_file, row.provision_file_sha256)
            for row in source.component_rows
        )
        return ResolvedCitationEvidence(
            files=tuple(sorted(files)),
            route=(
                source.citation_path,
                json.dumps(
                    [asdict(source.row), [asdict(r) for r in source.component_rows]],
                    sort_keys=True,
                ),
                str(source.slice_required),
            ),
            text_sha256=hashlib.sha256(source.body.encode("utf-8")).hexdigest(),
        )

    def row_scopes(provision_file: str) -> frozenset[str]:
        path = local.root / provision_file
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"provisions file is not readable: {provision_file}")
        if path.stat().st_size > _MAX_PROVISIONS_BYTES:
            raise ValueError(f"provisions file is too large: {provision_file}")
        scopes: set[str] = set()
        for line in path.read_bytes().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                scopes.add(
                    "{}/{}/{}".format(
                        row.get("jurisdiction"),
                        row.get("document_class"),
                        row.get("version"),
                    )
                )
        return frozenset(scopes)

    return CorpusView(unit=unit, citation=citation, row_scopes=row_scopes)


def run_corpus_attestation_carry(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="axiom-encode corpus-attestation-carry",
        description=(
            "Dry run: list, per signed manifest, whether its source attestation "
            "carries to a new signed corpus release. Writes nothing in the "
            "RuleSpec repository; a lock-file corpus checkout may receive the "
            "verified provisions files its locks pin, as corpus-fetch places them."
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
        help="materialized corpus of --to-release",
    )
    parser.add_argument(
        "--from-corpus-root",
        type=Path,
        help="materialized corpus of the attested releases (default: --corpus-root)",
    )
    parser.add_argument(
        "--public-key",
        action="append",
        default=[],
        help=(
            f"current corpus release public key, repeatable (default: "
            f"${RELEASE_OBJECT_PUBLIC_KEY_ENV}); a guard must use the "
            "organization trust root instead"
        ),
    )
    parser.add_argument(
        "--retired-public-key",
        action="append",
        default=[],
        help="retired release key, accepted for attested releases only (repeatable)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv))

    current_keys = list(args.public_key)
    if not current_keys and os.environ.get(RELEASE_OBJECT_PUBLIC_KEY_ENV):
        current_keys = [os.environ[RELEASE_OBJECT_PUBLIC_KEY_ENV]]
    if not current_keys:
        parser.error(
            f"--public-key or {RELEASE_OBJECT_PUBLIC_KEY_ENV} is required to verify "
            "release objects"
        )
    attested_keys = [*current_keys, *args.retired_public_key]
    try:
        target = _load_verified_release(args.to_release, current_keys)
        attested = [
            _load_verified_release(path, attested_keys) for path in args.from_release
        ]
    except (OSError, ValueError) as exc:
        print(f"corpus-attestation-carry: {exc}", file=sys.stderr)
        return 2
    manifests_root = args.rulespec_repo / ".axiom" / "encoding-manifests"
    if manifests_root.is_symlink() or not manifests_root.is_dir():
        print(
            f"corpus-attestation-carry: no manifest directory at {manifests_root}",
            file=sys.stderr,
        )
        return 2
    view_b: CorpusView | None = None
    views_a: dict[tuple[str, str], CorpusView] = {}
    if args.corpus_root is not None:
        from_root = args.from_corpus_root or args.corpus_root
        try:
            view_b = corpus_view(args.corpus_root, target, current_keys)
            for release in attested:
                views_a[(release.name, release.content_sha256)] = corpus_view(
                    from_root, release, attested_keys
                )
        except (OSError, ValueError) as exc:
            print(
                f"corpus-attestation-carry: cannot load a corpus: {exc}",
                file=sys.stderr,
            )
            return 2
    rows: list[dict[str, Any]] = []
    for entry in iter_manifest_attestations(manifests_root):
        attestation = entry.attestation
        if entry.problems:
            verdict = CarryVerdict(CarryStatus.MALFORMED, entry.problems)
        elif attestation is None:
            verdict = CarryVerdict(CarryStatus.NO_ATTESTATION)
        else:
            key = (
                (
                    attestation.get("corpus_release"),
                    attestation.get("corpus_release_content_sha256"),
                )
                if isinstance(attestation, Mapping)
                else None
            )
            verdict = classify_attestation_carry(
                attestation,
                target=target,
                attested_releases=attested,
                cited_paths=entry.cited_paths,
                dependency_paths=entry.dependency_paths,
                view_a=views_a.get(key) if key else None,
                view_b=view_b,
            )
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
                "dependency_paths": len(entry.dependency_paths),
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
