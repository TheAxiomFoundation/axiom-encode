"""Audit every committed encoding manifest against the file it attests.

The apply guard (``guard-generated``) asks *"is there a matching attestation
for this changed file?"*.  One good record answers that question, and the guard
says nothing about the other records committed alongside it.  So a rule can be
re-generated, gain a fresh manifest in a new location, and leave its previous
manifest behind asserting a hash the file no longer has — with CI green.

This module asks the auditor's question instead: **does every committed
attestation match its file?**  That is the property a public, tamper-evident
corpus advertises, and it is the check an outsider would run.  Two invariants
implement it:

``every attestation matches``
    Every ``applied_files[]`` entry in every committed manifest must resolve to
    a tracked file whose sha256 equals the attested value (or, for a deletion
    marker, to no file at all).

``at most one live manifest per rule path``
    Two manifests attesting the same file is the ambiguity that let the first
    invariant rot unnoticed: the guard found the good record and never looked
    at the bad one.  Superseded records are pruned, not accumulated.

The audit is deliberately **schema- and signature-agnostic**.  It hashes files
and compares bytes.  Signature and schema validity are enforced by the apply
guard; re-checking them here would make the audit inherit the guard's blind
spot — the current verifier accepts only the newest manifest schema, so a
schema-gated audit would silently skip every legacy record, which is precisely
where the stale claims live.

Where a claim genuinely cannot be pruned — the manifest is the sole surviving
attestation for a *sibling* file it also covers, so deleting it would destroy a
true record — the divergence is disclosed in a supersession ledger rather than
hidden.  A ledger entry is not an authorization and attests nothing; it is a
self-verifying disclosure whose every element is re-derived from the repository
and from git history.  An entry that does not hold fails the audit.

See TheAxiomFoundation/axiom-encode#1282.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import textwrap
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from axiom_encode.constants import RULESPEC_FILESYSTEM_ROOTS

MANIFEST_DIR = PurePosixPath(".axiom") / "encoding-manifests"
LEDGER_RELATIVE_PATH = PurePosixPath(".axiom") / "encoding-manifest-supersessions.yaml"
LEDGER_SCHEMA = "axiom-encode/manifest-supersessions/v1"

MAX_LEDGER_BYTES = 2_000_000
MAX_LEDGER_ENTRIES = 5_000
SHA256_LENGTH = 64
_HEX = frozenset("0123456789abcdef")

#: Finding kinds, ordered most to least severe for reporting.
MISMATCH = "mismatch"
MISSING = "missing"
RESURRECTED = "resurrected"
AMBIGUOUS = "ambiguous"
DUPLICATE = "duplicate"
UNREADABLE = "unreadable"
LEDGER_INVALID = "ledger-invalid"
LEDGER_UNUSED = "ledger-unused"

_FINDING_ORDER = (
    MISMATCH,
    MISSING,
    RESURRECTED,
    AMBIGUOUS,
    DUPLICATE,
    UNREADABLE,
    LEDGER_INVALID,
    LEDGER_UNUSED,
)


class ManifestAuditError(ValueError):
    """Raised when the audit cannot be performed at all."""


class LedgerSchemaError(ManifestAuditError):
    """Raised when the supersession ledger is malformed."""


@dataclass(frozen=True, slots=True)
class AuditFinding:
    kind: str
    detail: str
    manifest: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        where = self.manifest or ""
        if self.path:
            where = f"{where} -> {self.path}" if where else self.path
        return (
            f"[{self.kind}] {where}: {self.detail}"
            if where
            else f"[{self.kind}] {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class Claim:
    """One ``applied_files[]`` entry resolved against the working tree."""

    manifest: str
    entry_path: str
    resolved_path: str | None
    attested_sha256: str | None
    deleted: bool
    candidates: tuple[str, ...] = ()


@dataclass
class AuditResult:
    findings: list[AuditFinding] = field(default_factory=list)
    manifests: int = 0
    attestations: int = 0
    matched: int = 0
    disclosed: int = 0

    @property
    def passed(self) -> bool:
        return not self.findings

    def by_kind(self) -> dict[str, list[AuditFinding]]:
        grouped: dict[str, list[AuditFinding]] = defaultdict(list)
        for finding in self.findings:
            grouped[finding.kind].append(finding)
        return dict(grouped)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "manifests": self.manifests,
            "attestations": self.attestations,
            "matched": self.matched,
            "disclosed": self.disclosed,
            "findings": [
                {
                    "kind": finding.kind,
                    "manifest": finding.manifest,
                    "path": finding.path,
                    "detail": finding.detail,
                }
                for finding in self.findings
            ],
        }


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def manifest_trees(repo_path: Path) -> list[str]:
    """Return every ``.axiom/encoding-manifests`` tree prefix in the repo.

    A country monorepo carries one tree per jurisdiction subroot
    (``us/.axiom/...``, ``us-co/.axiom/...``) *and* a checkout-root tree
    (``.axiom/...``) that the #1078 relocation writes into.  All of them hold
    live attestations, so all of them are audited; enumerating only the
    checkout-root tree is how 248 of rulespec-us's manifests became invisible
    to the existing guard.
    """
    repo_path = Path(repo_path)
    prefixes: list[str] = []
    if (repo_path / MANIFEST_DIR).is_dir():
        prefixes.append("")
    for child in sorted(repo_path.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / MANIFEST_DIR).is_dir():
            prefixes.append(child.name)
    return prefixes


def jurisdiction_roots(repo_path: Path) -> list[str]:
    """Return top-level directories that hold RuleSpec content.

    Used only to resolve legacy country-relative manifest keys.  Derived from
    the tree rather than from ``repository-structure.yaml`` so the audit works
    on a checkout that predates (or postdates) any given layout gate.
    """
    repo_path = Path(repo_path)
    roots: list[str] = []
    for child in sorted(repo_path.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any((child / root).is_dir() for root in sorted(RULESPEC_FILESYSTEM_ROOTS)):
            roots.append(child.name)
    return roots


def manifest_paths(repo_path: Path) -> list[str]:
    """Return every committed manifest path, repo-relative, sorted."""
    repo_path = Path(repo_path)
    found: set[str] = set()
    for prefix in manifest_trees(repo_path):
        base = repo_path / prefix / MANIFEST_DIR if prefix else repo_path / MANIFEST_DIR
        for path in base.rglob("*.json"):
            if path.is_file():
                found.add(path.relative_to(repo_path).as_posix())
    return sorted(found)


def _tree_prefix(manifest_relpath: str) -> str:
    text = manifest_relpath
    marker = MANIFEST_DIR.as_posix()
    index = text.find(marker)
    if index <= 0:
        return ""
    return text[: index - 1]


def resolve_attested_path(
    repo_path: Path,
    manifest_relpath: str,
    entry_path: str,
    *,
    jurisdictions: Sequence[str],
) -> tuple[str | None, tuple[str, ...]]:
    """Resolve an ``applied_files[].path`` to a repo-relative file path.

    Manifest keys are not uniform: a jurisdiction-tree manifest keys its files
    country-relative (``statutes/26/32/c/2.yaml`` under ``us/.axiom/...``),
    the relocated checkout-root tree keys them repo-relative
    (``us/statutes/...``), and the pre-relocation checkout-root records key
    them country-relative with no jurisdiction at all.  Getting this wrong
    manufactures phantom findings — resolving rulespec-us with a repo-relative
    reading alone reports 828 files as missing that are simply keyed the older
    way.

    Returns ``(resolved, candidates)``.  ``resolved`` is ``None`` when nothing
    matches, or when more than one jurisdiction matches; ``candidates`` carries
    the ambiguous matches so the caller can report them rather than guess.
    """
    repo_path = Path(repo_path)
    prefix = _tree_prefix(manifest_relpath)

    if prefix:
        candidate = f"{prefix}/{entry_path}"
        if (repo_path / candidate).is_file():
            return candidate, ()
    if (repo_path / entry_path).is_file():
        return entry_path, ()

    matches = tuple(
        f"{juris}/{entry_path}"
        for juris in jurisdictions
        if (repo_path / juris / entry_path).is_file()
    )
    if len(matches) == 1:
        return matches[0], ()
    return None, matches


def _attested_path_candidates(
    manifest_relpath: str, entry_path: str, *, jurisdictions: Sequence[str]
) -> tuple[str, ...]:
    """Every repo-relative path a manifest key could denote, most specific first.

    Used to check a historical revision, where the working-tree existence test
    that ``resolve_attested_path`` relies on is unavailable.
    """
    prefix = _tree_prefix(manifest_relpath)
    candidates: list[str] = []
    if prefix:
        candidates.append(f"{prefix}/{entry_path}")
    candidates.append(entry_path)
    candidates.extend(f"{juris}/{entry_path}" for juris in jurisdictions)
    seen: set[str] = set()
    return tuple(c for c in candidates if not (c in seen or seen.add(c)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Claim collection
# --------------------------------------------------------------------------


def collect_claims(repo_path: Path) -> tuple[list[Claim], list[AuditFinding]]:
    """Parse every committed manifest into resolved claims."""
    repo_path = Path(repo_path)
    jurisdictions = jurisdiction_roots(repo_path)
    claims: list[Claim] = []
    findings: list[AuditFinding] = []

    for manifest_relpath in manifest_paths(repo_path):
        absolute = repo_path / manifest_relpath
        try:
            payload = json.loads(absolute.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            findings.append(
                AuditFinding(
                    kind=UNREADABLE,
                    manifest=manifest_relpath,
                    detail=f"cannot be read as JSON ({error})",
                )
            )
            continue
        if not isinstance(payload, dict):
            findings.append(
                AuditFinding(
                    kind=UNREADABLE,
                    manifest=manifest_relpath,
                    detail="does not contain a JSON object",
                )
            )
            continue
        entries = payload.get("applied_files")
        if entries is None:
            # Not every ``.json`` under the tree is an apply manifest.
            continue
        if not isinstance(entries, list):
            findings.append(
                AuditFinding(
                    kind=UNREADABLE,
                    manifest=manifest_relpath,
                    detail="has a non-list applied_files",
                )
            )
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                findings.append(
                    AuditFinding(
                        kind=UNREADABLE,
                        manifest=manifest_relpath,
                        detail=f"has a malformed applied_files[{index}]",
                    )
                )
                continue
            entry_path = entry["path"]
            deleted = entry.get("deleted") is True
            resolved, candidates = resolve_attested_path(
                repo_path,
                manifest_relpath,
                entry_path,
                jurisdictions=jurisdictions,
            )
            attested = entry.get("sha256")
            claims.append(
                Claim(
                    manifest=manifest_relpath,
                    entry_path=entry_path,
                    resolved_path=resolved,
                    attested_sha256=attested if isinstance(attested, str) else None,
                    deleted=deleted,
                    candidates=candidates,
                )
            )
    return claims, findings


# --------------------------------------------------------------------------
# Supersession ledger
# --------------------------------------------------------------------------


_ENTRY_REQUIRED = frozenset({"manifest", "attested_path", "attested_sha256", "reason"})
_ENTRY_OPTIONAL = frozenset(
    {"superseded_by", "retired_in", "current_path", "current_sha256", "issue"}
)
_ENTRY_ALLOWED = _ENTRY_REQUIRED | _ENTRY_OPTIONAL


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that refuses duplicate and merged mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "<<":
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not allowed",
                key_node.start_mark,
            )
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One disclaimed claim: manifest ``M`` no longer describes ``attested_path``.

    Exactly one resolution is recorded.  ``superseded_by`` names a newer
    manifest that attests the same file's current content.  ``retired_in``
    names the reviewed commits that ended the claim, together with where the
    content lives now (``current_path``/``current_sha256``) — or neither, when
    the content was removed outright.  A relocation is the case where
    ``current_path`` differs from the attested path while the hash is unchanged.
    """

    manifest: str
    attested_path: str
    attested_sha256: str
    reason: str
    superseded_by: str | None = None
    retired_in: tuple[str, ...] = ()
    current_path: str | None = None
    current_sha256: str | None = None
    issue: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.manifest, self.attested_path)


def _require_sha256(value: Any, *, field_name: str, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or not set(value) <= _HEX
    ):
        raise LedgerSchemaError(
            f"{where}: {field_name} must be a lowercase 64-character sha256 hex digest"
        )
    return value


def load_ledger(repo_path: Path) -> dict[tuple[str, str], LedgerEntry]:
    """Load and schema-check the supersession ledger.

    Absent ledger is not an error — it is the expected state for a clean repo.
    """
    ledger_path = Path(repo_path) / LEDGER_RELATIVE_PATH
    if not ledger_path.is_file():
        return {}
    size = ledger_path.stat().st_size
    if size > MAX_LEDGER_BYTES:
        raise LedgerSchemaError(
            f"{LEDGER_RELATIVE_PATH}: exceeds {MAX_LEDGER_BYTES} bytes"
        )
    try:
        document = yaml.load(  # noqa: S506 - hardened loader, not yaml.Loader
            ledger_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except (yaml.YAMLError, UnicodeError, OSError) as error:
        raise LedgerSchemaError(f"{LEDGER_RELATIVE_PATH}: {error}") from error
    if not isinstance(document, Mapping):
        raise LedgerSchemaError(f"{LEDGER_RELATIVE_PATH}: top level must be a mapping")
    unknown = set(document) - {"schema_version", "superseded"}
    if unknown:
        raise LedgerSchemaError(
            f"{LEDGER_RELATIVE_PATH}: unknown top-level keys {sorted(unknown)}"
        )
    if document.get("schema_version") != LEDGER_SCHEMA:
        raise LedgerSchemaError(
            f"{LEDGER_RELATIVE_PATH}: schema_version must be {LEDGER_SCHEMA!r}"
        )
    rows = document.get("superseded") or []
    if not isinstance(rows, list):
        raise LedgerSchemaError(f"{LEDGER_RELATIVE_PATH}: superseded must be a list")
    if len(rows) > MAX_LEDGER_ENTRIES:
        raise LedgerSchemaError(
            f"{LEDGER_RELATIVE_PATH}: exceeds {MAX_LEDGER_ENTRIES} entries"
        )

    entries: dict[tuple[str, str], LedgerEntry] = {}
    for index, row in enumerate(rows):
        where = f"{LEDGER_RELATIVE_PATH}[{index}]"
        if not isinstance(row, Mapping):
            raise LedgerSchemaError(f"{where}: entry must be a mapping")
        unknown = set(row) - _ENTRY_ALLOWED
        if unknown:
            raise LedgerSchemaError(f"{where}: unknown keys {sorted(unknown)}")
        missing = _ENTRY_REQUIRED - set(row)
        if missing:
            raise LedgerSchemaError(f"{where}: missing keys {sorted(missing)}")
        has_successor = "superseded_by" in row
        has_retirement = "retired_in" in row
        if has_successor == has_retirement:
            raise LedgerSchemaError(
                f"{where}: exactly one of superseded_by or retired_in is required"
            )
        for name in ("manifest", "attested_path", "reason"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise LedgerSchemaError(f"{where}: {name} must be a non-empty string")

        retired_in: tuple[str, ...] = ()
        current_path: str | None = None
        current_sha256: str | None = None
        if has_retirement:
            commits = row["retired_in"]
            if not isinstance(commits, list) or not commits:
                raise LedgerSchemaError(
                    f"{where}: retired_in must be a non-empty list of commit sha1s"
                )
            for commit in commits:
                if (
                    not isinstance(commit, str)
                    or len(commit) != 40
                    or not set(commit) <= _HEX
                ):
                    raise LedgerSchemaError(
                        f"{where}: retired_in entries must be full 40-character commit sha1s"
                    )
            retired_in = tuple(commits)
            # ``current_path`` and ``current_sha256`` travel together: either
            # the content still exists somewhere (both present) or it was
            # removed outright (both absent).  A half-filled pair would let an
            # entry assert a location without pinning its bytes.
            if ("current_path" in row) != ("current_sha256" in row):
                raise LedgerSchemaError(
                    f"{where}: current_path and current_sha256 must be given together"
                )
            if "current_path" in row:
                if (
                    not isinstance(row["current_path"], str)
                    or not row["current_path"].strip()
                ):
                    raise LedgerSchemaError(
                        f"{where}: current_path must be a non-empty string"
                    )
                current_path = row["current_path"]
                current_sha256 = _require_sha256(
                    row["current_sha256"], field_name="current_sha256", where=where
                )
        elif "current_path" in row or "current_sha256" in row:
            raise LedgerSchemaError(
                f"{where}: current_path/current_sha256 belong to retired_in entries; "
                "a superseded_by entry is pinned by its successor manifest"
            )

        successor = row.get("superseded_by")
        if has_successor and (not isinstance(successor, str) or not successor.strip()):
            raise LedgerSchemaError(
                f"{where}: superseded_by must be a non-empty string"
            )
        issue = row.get("issue")
        if issue is not None and not isinstance(issue, str):
            raise LedgerSchemaError(f"{where}: issue must be a string")
        entry = LedgerEntry(
            manifest=row["manifest"],
            attested_path=row["attested_path"],
            attested_sha256=_require_sha256(
                row["attested_sha256"], field_name="attested_sha256", where=where
            ),
            reason=row["reason"],
            superseded_by=successor if has_successor else None,
            retired_in=retired_in,
            current_path=current_path,
            current_sha256=current_sha256,
            issue=issue,
        )
        if entry.key in entries:
            raise LedgerSchemaError(
                f"{where}: duplicate entry for {entry.manifest} -> {entry.attested_path}"
            )
        entries[entry.key] = entry
    return entries


def _git(repo_path: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _blob_sha256_at(repo_path: Path, commit: str, path: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "cat-file", "blob", f"{commit}:{path}"],
            cwd=repo_path,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def verify_ledger_entry(
    repo_path: Path,
    entry: LedgerEntry,
    *,
    claim: Claim,
    claims_by_file: Mapping[str, list[Claim]],
    hashes: Mapping[str, str],
    jurisdictions: Sequence[str],
) -> list[str]:
    """Return the reasons ``entry`` fails to justify ``claim``.

    Every element is re-derived from the repository and from git; nothing in
    the ledger is taken on trust.  An entry that has gone stale — because the
    file moved again, or the successor was itself superseded — stops verifying
    and the audit fails, which is the intended fail-closed behaviour.
    """
    repo_path = Path(repo_path)
    problems: list[str] = []

    if claim.attested_sha256 != entry.attested_sha256:
        problems.append(
            "attested_sha256 does not match the manifest's actual claim "
            f"({claim.attested_sha256})"
        )
        return problems

    if entry.superseded_by is not None:
        if claim.resolved_path is None:
            problems.append(
                "names a successor manifest, but the attested path resolves to no "
                "file; use retired_in to record where the content went"
            )
            return problems
        current = hashes[claim.resolved_path]
        successor = entry.superseded_by
        if not (repo_path / successor).is_file():
            problems.append(f"superseded_by manifest {successor} does not exist")
            return problems
        successor_claims = [
            other
            for other in claims_by_file.get(claim.resolved_path, [])
            if other.manifest == successor and not other.deleted
        ]
        if not successor_claims:
            problems.append(
                f"superseded_by manifest {successor} does not attest {claim.resolved_path}"
            )
        elif not any(other.attested_sha256 == current for other in successor_claims):
            problems.append(
                f"superseded_by manifest {successor} does not attest the current "
                f"content of {claim.resolved_path}"
            )
        return problems

    # retired_in: the end of the claim must be provable from git history.
    if _git(repo_path, "rev-parse", "--git-dir") is None:
        problems.append(
            "retired_in entries require a git checkout to verify; none is available"
        )
        return problems
    for commit in entry.retired_in:
        if _git(repo_path, "cat-file", "-e", f"{commit}^{{commit}}") is None:
            problems.append(f"retired_in commit {commit} does not exist")
            return problems
        if _git(repo_path, "merge-base", "--is-ancestor", commit, "HEAD") is None:
            problems.append(f"retired_in commit {commit} is not an ancestor of HEAD")
            return problems

    first = entry.retired_in[0]
    parents = (_git(repo_path, "rev-parse", f"{first}^@") or "").split()
    if not parents:
        problems.append(f"retired_in commit {first} has no parent to compare against")
        return problems
    # The manifest's key may be country-relative, repo-relative, or
    # jurisdiction-prefixed; the claim must have held under one of them.
    candidates = _attested_path_candidates(
        claim.manifest, entry.attested_path, jurisdictions=jurisdictions
    )
    held = any(
        _blob_sha256_at(repo_path, parent, candidate) == entry.attested_sha256
        for parent in parents
        for candidate in candidates
    )
    if not held:
        problems.append(
            f"retired_in commit {first} does not begin at the attested content "
            f"(no parent holds {entry.attested_sha256[:12]}… at {entry.attested_path})"
        )
        return problems

    if entry.current_path is None:
        if claim.resolved_path is not None:
            problems.append(
                "records the content as removed, but "
                f"{claim.resolved_path} exists in the working tree"
            )
        return problems

    destination = repo_path / entry.current_path
    if not destination.is_file():
        problems.append(f"current_path {entry.current_path} does not exist")
        return problems
    actual = sha256_file(destination)
    if actual != entry.current_sha256:
        problems.append(
            f"current_sha256 does not match {entry.current_path} ({actual})"
        )
    return problems


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------


def audit_repository(repo_path: Path) -> AuditResult:
    """Audit every committed manifest in ``repo_path``."""
    repo_path = Path(repo_path).resolve()
    if not repo_path.is_dir():
        raise ManifestAuditError(f"{repo_path} is not a directory")

    result = AuditResult()
    claims, findings = collect_claims(repo_path)
    result.findings.extend(findings)
    result.manifests = len(manifest_paths(repo_path))
    result.attestations = len(claims)

    try:
        ledger = load_ledger(repo_path)
    except LedgerSchemaError as error:
        result.findings.append(AuditFinding(kind=LEDGER_INVALID, detail=str(error)))
        ledger = {}

    claims_by_file: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims:
        if claim.resolved_path is not None:
            claims_by_file[claim.resolved_path].append(claim)

    hashes: dict[str, str] = {}
    for path in claims_by_file:
        hashes[path] = sha256_file(repo_path / path)

    jurisdictions = jurisdiction_roots(repo_path)
    used_ledger_keys: set[tuple[str, str]] = set()

    def disclosed(claim: Claim) -> bool:
        """Consume a ledger entry for ``claim``; record any way it fails."""
        entry = ledger.get((claim.manifest, claim.entry_path))
        if entry is None:
            return False
        used_ledger_keys.add(entry.key)
        problems = verify_ledger_entry(
            repo_path,
            entry,
            claim=claim,
            claims_by_file=claims_by_file,
            hashes=hashes,
            jurisdictions=jurisdictions,
        )
        for problem in problems:
            result.findings.append(
                AuditFinding(
                    kind=LEDGER_INVALID,
                    manifest=claim.manifest,
                    path=claim.entry_path,
                    detail=f"supersession ledger entry {problem}",
                )
            )
        if not problems:
            result.disclosed += 1
        return True

    for claim in claims:
        if claim.candidates:
            result.findings.append(
                AuditFinding(
                    kind=AMBIGUOUS,
                    manifest=claim.manifest,
                    path=claim.entry_path,
                    detail=(
                        "resolves to more than one jurisdiction: "
                        + ", ".join(claim.candidates)
                    ),
                )
            )
            continue
        if claim.deleted:
            if claim.resolved_path is not None:
                result.findings.append(
                    AuditFinding(
                        kind=RESURRECTED,
                        manifest=claim.manifest,
                        path=claim.resolved_path,
                        detail="is recorded as deleted but exists in the working tree",
                    )
                )
            else:
                result.matched += 1
            continue
        if claim.resolved_path is None:
            # A relocated or removed file: the attested path holds nothing.
            # Disclosable, because the move is provable from git history.
            if not disclosed(claim):
                result.findings.append(
                    AuditFinding(
                        kind=MISSING,
                        manifest=claim.manifest,
                        path=claim.entry_path,
                        detail="is attested but no such file exists in the working tree",
                    )
                )
            continue
        if claim.attested_sha256 is None:
            result.findings.append(
                AuditFinding(
                    kind=UNREADABLE,
                    manifest=claim.manifest,
                    path=claim.entry_path,
                    detail="has no sha256",
                )
            )
            continue

        current = hashes[claim.resolved_path]
        if current == claim.attested_sha256:
            result.matched += 1
            continue

        if not disclosed(claim):
            result.findings.append(
                AuditFinding(
                    kind=MISMATCH,
                    manifest=claim.manifest,
                    path=claim.resolved_path,
                    detail=(
                        f"attests {claim.attested_sha256[:12]}… but the file is "
                        f"{current[:12]}…"
                    ),
                )
            )

    # Uniqueness: at most one live manifest may attest a given file.  The
    # invariant is over *manifests*, not entries — a manifest that happens to
    # list the same path twice is one record, not two.
    for path, path_claims in sorted(claims_by_file.items()):
        current = hashes[path]
        live: set[str] = set()
        for claim in path_claims:
            if claim.deleted:
                continue
            # A claim disclosed in the ledger is retired in place, not live.
            if (
                claim.manifest,
                claim.entry_path,
            ) in ledger and claim.attested_sha256 != current:
                continue
            live.add(claim.manifest)
        if len(live) <= 1:
            continue
        result.findings.append(
            AuditFinding(
                kind=DUPLICATE,
                path=path,
                detail=(
                    "is attested by more than one live manifest: "
                    + ", ".join(sorted(live))
                ),
            )
        )

    for key, entry in sorted(ledger.items()):
        if key in used_ledger_keys:
            continue
        result.findings.append(
            AuditFinding(
                kind=LEDGER_UNUSED,
                manifest=entry.manifest,
                path=entry.attested_path,
                detail=(
                    "supersession ledger entry no longer corresponds to a stale "
                    "attestation; remove it"
                ),
            )
        )

    result.findings.sort(
        key=lambda finding: (
            _FINDING_ORDER.index(finding.kind)
            if finding.kind in _FINDING_ORDER
            else len(_FINDING_ORDER),
            finding.manifest or "",
            finding.path or "",
        )
    )
    return result


# --------------------------------------------------------------------------
# Prune-on-supersede (write path)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrunePlan:
    """What writing ``manifest`` implies for previously committed records."""

    retire: tuple[str, ...]
    disclose: tuple[tuple[str, str], ...]

    @property
    def empty(self) -> bool:
        return not self.retire and not self.disclose


def plan_prune(
    repo_path: Path,
    *,
    new_manifest: str,
    attested_paths: Iterable[str],
    claims: Sequence[Claim] | None = None,
) -> PrunePlan:
    """Plan the retirement of records superseded by ``new_manifest``.

    A prior manifest whose every attested path is now covered by
    ``new_manifest`` is fully superseded and is deleted.  One that also covers
    *other* paths is only partly superseded: deleting it would destroy the
    surviving true claims, so its superseded claims are disclosed in the ledger
    instead.  This is the same distinction the rulespec-us cleanup had to make
    by hand — records that were wholly stale could be deleted, while 19 were
    the sole surviving attestation for a companion test file and could not be.

    ``claims`` lets a caller reuse a scan it already performed; a bulk encode
    writes one manifest per rule and the scan is the expensive part.
    """
    repo_path = Path(repo_path)
    targets = {path for path in attested_paths}
    if claims is None:
        claims, _ = collect_claims(repo_path)

    covered: dict[str, set[str]] = defaultdict(set)
    for claim in claims:
        if claim.manifest == new_manifest or claim.deleted:
            continue
        if claim.resolved_path is not None:
            covered[claim.manifest].add(claim.resolved_path)

    retire: list[str] = []
    disclose: list[tuple[str, str]] = []
    for manifest, paths in sorted(covered.items()):
        overlap = paths & targets
        if not overlap:
            continue
        if paths <= targets:
            retire.append(manifest)
        else:
            disclose.extend((manifest, path) for path in sorted(overlap))
    return PrunePlan(retire=tuple(retire), disclose=tuple(disclose))


_LEDGER_HEADER = """\
# Encoding-manifest supersession ledger
#
# A committed manifest whose claim about a file is no longer true is normally
# deleted outright: the record is superseded and git history retains it. That is
# not possible when the same manifest also carries the only attestation for
# another file it covers — deleting it would destroy a true record in order to
# remove a false one. Those claims are retired in place and disclosed here.
#
# An entry ATTESTS NOTHING. It discloses that a claim ended, and names either the
# manifest that replaced it or the reviewed commits that ended it.
# `axiom-encode manifest-audit` re-derives every field from the repository and
# from git history before accepting an entry; one that stops holding fails CI.
#
# Entries are removed, never edited, as their rules are re-encoded.
#
# See https://github.com/TheAxiomFoundation/axiom-encode/issues/1282
"""


def _render_ledger(entries: Sequence[LedgerEntry]) -> str:
    lines = [_LEDGER_HEADER, f"schema_version: {LEDGER_SCHEMA}", "superseded:"]
    for entry in sorted(entries, key=lambda e: (e.manifest, e.attested_path)):
        lines.append(f"  - manifest: {entry.manifest}")
        lines.append(f"    attested_path: {entry.attested_path}")
        lines.append(f"    attested_sha256: {entry.attested_sha256}")
        if entry.superseded_by:
            lines.append(f"    superseded_by: {entry.superseded_by}")
        if entry.retired_in:
            lines.append("    retired_in:")
            lines.extend(f"      - {commit}" for commit in entry.retired_in)
        if entry.current_path:
            lines.append(f"    current_path: {entry.current_path}")
            lines.append(f"    current_sha256: {entry.current_sha256}")
        lines.append("    reason: >-")
        wrapped = textwrap.wrap(" ".join(entry.reason.split()), width=82) or [""]
        lines.extend(f"      {chunk}" for chunk in wrapped)
        if entry.issue:
            lines.append(f"    issue: {entry.issue}")
    return "\n".join(lines) + "\n"


def write_ledger(repo_path: Path, entries: Sequence[LedgerEntry]) -> Path | None:
    """Write (or remove, when empty) the supersession ledger."""
    path = Path(repo_path) / LEDGER_RELATIVE_PATH
    if not entries:
        if path.is_file():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_ledger(entries))
    return path


def apply_prune(
    repo_path: Path,
    *,
    new_manifest: str,
    attested_paths: Iterable[str],
    reason: str,
    issue: str | None = None,
) -> PrunePlan:
    """Retire the records ``new_manifest`` supersedes, disclosing what it cannot.

    Called from the apply path so a re-generated rule never leaves a second,
    contradictory record behind — the accumulation that axiom-encode#1282
    found on rulespec-us, where 121 attestations had been wrong for weeks with
    CI green.  Deletion is used where the superseded record makes no other true
    claim; otherwise the superseded claim alone is disclosed in the ledger and
    the record survives to keep covering its siblings.
    """
    repo_path = Path(repo_path)
    targets = sorted(set(attested_paths))
    claims, _ = collect_claims(repo_path)
    plan = plan_prune(
        repo_path,
        new_manifest=new_manifest,
        attested_paths=targets,
        claims=claims,
    )
    if plan.empty:
        return plan

    claim_index = {
        (claim.manifest, claim.resolved_path): claim
        for claim in claims
        if claim.resolved_path is not None and not claim.deleted
    }

    for manifest in plan.retire:
        target = repo_path / manifest
        if target.is_file():
            target.unlink()
        parent = target.parent
        while parent != repo_path and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    if plan.disclose:
        entries = dict(load_ledger(repo_path))
        for manifest, path in plan.disclose:
            claim = claim_index.get((manifest, path))
            if claim is None or claim.attested_sha256 is None:
                continue
            entry = LedgerEntry(
                manifest=manifest,
                attested_path=claim.entry_path,
                attested_sha256=claim.attested_sha256,
                reason=reason,
                superseded_by=new_manifest,
                issue=issue,
            )
            entries[entry.key] = entry
        write_ledger(repo_path, list(entries.values()))
    return plan


def format_report(result: AuditResult, *, limit: int | None = None) -> str:
    """Render a human-readable audit report."""
    lines: list[str] = []
    if result.passed:
        lines.append(
            f"Every committed attestation matches its file "
            f"({result.matched} attestation(s) across {result.manifests} manifest(s)"
            + (
                f", {result.disclosed} disclosed as superseded"
                if result.disclosed
                else ""
            )
            + ")."
        )
        return "\n".join(lines)
    lines.append(
        f"{len(result.findings)} manifest audit finding(s) across "
        f"{result.manifests} manifest(s) / {result.attestations} attestation(s)."
    )
    grouped = result.by_kind()
    for kind in _FINDING_ORDER:
        entries = grouped.get(kind)
        if not entries:
            continue
        lines.append(f"\n{kind} ({len(entries)}):")
        shown = entries if limit is None else entries[:limit]
        for finding in shown:
            where = finding.manifest or finding.path or ""
            if finding.manifest and finding.path:
                where = f"{finding.manifest} -> {finding.path}"
            lines.append(f"  - {where}: {finding.detail}")
        if limit is not None and len(entries) > limit:
            lines.append(f"  … and {len(entries) - limit} more")
    return "\n".join(lines)
