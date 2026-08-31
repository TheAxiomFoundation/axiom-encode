"""Immutable Git-base proof for unmanifested legacy RuleSpec contractions."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .legacy_cleanup import (
    LEGACY_CLEANUP_RECEIPT_DIR,
    LegacyCleanupReceiptError,
    canonical_primary_paths,
    companion_path,
    decode_strict_json_object,
)

LEGACY_CLEANUP_TOOLCHAIN_PATH = Path(".axiom/toolchain.toml")
LEGACY_CLEANUP_WAIVER_PATH = Path("known-validation-gaps.yaml")
LEGACY_CLEANUP_PROVENANCE_DIRS = (
    Path(".axiom/encoding-manifests"),
    Path(".axiom/path-migrations"),
    Path(".axiom/legacy-replacements"),
    LEGACY_CLEANUP_RECEIPT_DIR,
)

_MAX_BASE_ENTRIES = 1_000_000
_MAX_BASE_BLOB_BYTES = 16 * 1024 * 1024
_MAX_PROVENANCE_BLOB_BYTES = 4 * 1024 * 1024
_MAX_PROVENANCE_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_REFERENCE_TOTAL_BYTES = 1024 * 1024 * 1024
_GIT_CONFIG_ARGUMENTS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.autocrlf=false",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "core.sparseCheckout=false",
)


class LegacyCleanupGitError(ValueError):
    """The immutable base cannot prove cleanup eligibility exactly."""


@dataclass(frozen=True, slots=True)
class LegacyCleanupProvenanceRecord:
    """One bounded provenance record read verbatim from the immutable base."""

    path: Path
    mode: str
    blob_oid: str
    sha256: str
    raw: bytes


@dataclass(frozen=True, slots=True)
class LegacyCleanupBasePlan:
    repository: str
    object_format: str
    base_commit: str
    base_tree: str
    projected_post_deletion_tree: str
    groups: tuple[dict[str, dict[str, str]], ...]
    base_files: dict[str, dict[str, str]]
    toolchain_values: dict[str, str]
    ownership_inventory_sha256: str
    provenance_records: tuple[LegacyCleanupProvenanceRecord, ...]
    surviving_reference_inventory_sha256: str
    surviving_blob_count: int

    @property
    def primary_paths(self) -> tuple[Path, ...]:
        return tuple(Path(group["primary"]["path"]) for group in self.groups)

    @property
    def deleted_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(group[key]["path"])
            for group in self.groups
            for key in ("primary", "companion")
        )


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    path: Path
    mode: str
    object_type: str
    oid: str
    size: int | None


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("GIT_"):
            environment.pop(name)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def _git_bytes(
    repo: Path,
    *arguments: str,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    completed = subprocess.run(
        ["git", *_GIT_CONFIG_ARGUMENTS, "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        env=dict(environment or _git_environment()),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LegacyCleanupGitError(
            f"cannot inspect immutable RuleSpec Git base ({' '.join(arguments)}): "
            f"{detail or 'git command failed'}"
        )
    return completed.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repo, *arguments).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LegacyCleanupGitError(
            "immutable RuleSpec Git output is not UTF-8"
        ) from exc


def _oid_length(object_format: str) -> int:
    if object_format == "sha1":
        return 40
    if object_format == "sha256":
        return 64
    raise LegacyCleanupGitError(f"unsupported Git object format: {object_format}")


def _require_oid(value: str, object_format: str, *, label: str) -> str:
    if (
        len(value) != _oid_length(object_format)
        or re.fullmatch(r"[0-9a-f]+", value) is None
    ):
        raise LegacyCleanupGitError(f"{label} is not a full lowercase Git OID")
    return value


def _canonical_github_repository(remote: str) -> str | None:
    value = remote.strip()
    for pattern in (
        r"https://github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
        r"git@github\.com:(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?$",
        r"ssh://git@github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?$",
    ):
        match = re.fullmatch(pattern, value)
        if match is not None:
            return f"github.com/{match.group('slug')}"
    return None


def canonical_rulespec_repository(repo: Path) -> str:
    checkout = Path(repo).resolve(strict=True)
    if re.fullmatch(r"rulespec-[a-z]{2}", checkout.name) is None:
        raise LegacyCleanupGitError(
            "legacy cleanup requires an exact canonical rulespec-<country> checkout"
        )
    top = Path(_git_text(checkout, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != checkout:
        raise LegacyCleanupGitError("legacy cleanup path is not the Git checkout root")
    expected = f"github.com/TheAxiomFoundation/{checkout.name}"
    actual = _canonical_github_repository(
        _git_text(checkout, "remote", "get-url", "origin")
    )
    if actual != expected:
        raise LegacyCleanupGitError(
            f"RuleSpec origin must be {expected}, got {actual or '<invalid>'}"
        )
    return expected


def clean_official_checkout_pin(
    repo: Path,
    *,
    expected_repository: str,
    version: str | None = None,
) -> dict[str, str]:
    """Return one exact clean official Git checkout pin.

    Cleanup validation executes code from the encoder and rules-engine checkouts,
    so a repository label in the receipt is not sufficient.  This proof binds the
    canonical origin, exact top level, full commit, and object format while using
    the same config- and replace-ref-neutral Git environment as base inspection.
    Ignored build products are not execution authority, but every tracked,
    staged, and non-ignored untracked path is rejected.
    """

    checkout = Path(repo).resolve(strict=True)
    top = Path(_git_text(checkout, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != checkout:
        raise LegacyCleanupGitError(
            f"{expected_repository} path is not the exact Git checkout root"
        )
    actual_repository = _canonical_github_repository(
        _git_text(checkout, "remote", "get-url", "origin")
    )
    if actual_repository != expected_repository:
        raise LegacyCleanupGitError(
            f"checkout origin must be {expected_repository}, got "
            f"{actual_repository or '<invalid>'}"
        )
    object_format = _git_text(
        checkout,
        "rev-parse",
        "--show-object-format",
    ).strip()
    commit = _require_oid(
        _git_text(checkout, "rev-parse", "HEAD^{commit}").strip(),
        object_format,
        label=f"{expected_repository} checkout commit",
    )
    status = _git_bytes(
        checkout,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    if status:
        raise LegacyCleanupGitError(
            f"{expected_repository} checkout must have no staged, tracked, or "
            "non-ignored untracked changes"
        )
    pin = {
        "repository": expected_repository,
        "object_format": object_format,
        "commit": commit,
    }
    if version is not None:
        pin["version"] = version
    return pin


def _full_commit(repo: Path, raw: str, object_format: str) -> str:
    _require_oid(raw, object_format, label="legacy cleanup protected base")
    resolved = _git_text(repo, "rev-parse", "--verify", f"{raw}^{{commit}}").strip()
    if resolved != raw:
        raise LegacyCleanupGitError("legacy cleanup base did not resolve exactly")
    return resolved


def _tree_inventory(
    repo: Path,
    commit: str,
    object_format: str,
    prefixes: Sequence[Path] = (),
) -> tuple[_TreeEntry, ...]:
    arguments = ["ls-tree", "-r", "-l", "-z", "--full-tree", commit]
    if prefixes:
        arguments.append("--")
        arguments.extend(path.as_posix() for path in prefixes)
    raw = _git_bytes(repo, *arguments)
    entries: list[_TreeEntry] = []
    seen: set[Path] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, object_type, oid, raw_size = metadata.decode("ascii").split()
            path_text = encoded_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise LegacyCleanupGitError(
                "immutable base tree entry is malformed"
            ) from exc
        path = Path(path_text)
        if (
            path.is_absolute()
            or path.as_posix() != path_text
            or any(part in {"", ".", ".."} for part in path.parts)
            or path in seen
        ):
            raise LegacyCleanupGitError("immutable base tree path is ambiguous")
        seen.add(path)
        _require_oid(oid, object_format, label="immutable base tree object")
        if object_type == "blob":
            try:
                size = int(raw_size)
            except ValueError as exc:
                raise LegacyCleanupGitError(
                    "immutable base blob size is malformed"
                ) from exc
            if size < 0:
                raise LegacyCleanupGitError("immutable base blob size is malformed")
        elif object_type == "commit":
            size = None
        else:
            raise LegacyCleanupGitError(
                "immutable base tree object type is unsupported"
            )
        entries.append(_TreeEntry(path, mode, object_type, oid, size))
        if len(entries) > _MAX_BASE_ENTRIES:
            raise LegacyCleanupGitError("immutable base tree contains too many entries")
    return tuple(entries)


def _entry_map(entries: Sequence[_TreeEntry]) -> dict[Path, _TreeEntry]:
    return {entry.path: entry for entry in entries}


def _blob(repo: Path, entry: _TreeEntry, *, max_bytes: int, label: str) -> bytes:
    if entry.object_type != "blob" or entry.size is None:
        raise LegacyCleanupGitError(f"{label} is not a blob")
    if entry.size > max_bytes:
        raise LegacyCleanupGitError(f"{label} exceeds the {max_bytes}-byte limit")
    raw = _git_bytes(repo, "cat-file", "blob", entry.oid)
    if len(raw) != entry.size:
        raise LegacyCleanupGitError(f"{label} size differs from its Git tree entry")
    return raw


def _file_record(repo: Path, entry: _TreeEntry, *, deletion: bool) -> dict[str, str]:
    if entry.mode != "100644" or entry.object_type != "blob":
        raise LegacyCleanupGitError(
            f"legacy cleanup base path must be a regular 100644 blob: {entry.path}"
        )
    raw = _blob(
        repo,
        entry,
        max_bytes=_MAX_BASE_BLOB_BYTES,
        label=f"legacy cleanup base path {entry.path.as_posix()}",
    )
    record = {
        "path": entry.path.as_posix(),
        "base_mode": entry.mode,
        "base_blob_oid": entry.oid,
        "base_sha256": hashlib.sha256(raw).hexdigest(),
    }
    if deletion:
        record["result"] = "absent"
    return record


def _safe_live_bytes(repo: Path, relative: Path) -> bytes:
    cursor = repo
    for index, part in enumerate(relative.parts):
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise LegacyCleanupGitError(
                f"legacy cleanup live preimage is unavailable: {relative}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LegacyCleanupGitError(
                f"legacy cleanup live preimage has a symlink component: {relative}"
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise LegacyCleanupGitError(
                f"legacy cleanup live preimage has a non-directory ancestor: {relative}"
            )
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o644:
        raise LegacyCleanupGitError(
            f"legacy cleanup live preimage must be a regular 0644 file: {relative}"
        )
    if metadata.st_size > _MAX_BASE_BLOB_BYTES:
        raise LegacyCleanupGitError(
            f"legacy cleanup live preimage exceeds its byte limit: {relative}"
        )
    try:
        raw = cursor.read_bytes()
    except OSError as exc:
        raise LegacyCleanupGitError(
            f"legacy cleanup live preimage is unreadable: {relative}"
        ) from exc
    after = cursor.lstat()
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise LegacyCleanupGitError(
            f"legacy cleanup live preimage changed while read: {relative}"
        )
    return raw


def _projected_tree(
    repo: Path,
    *,
    base: str,
    object_format: str,
    deleted: Sequence[Path],
) -> str:
    with tempfile.TemporaryDirectory(prefix="axiom-cleanup-index-") as directory:
        index_path = Path(directory) / "index"
        environment = _git_environment()
        environment["GIT_INDEX_FILE"] = str(index_path)
        _git_bytes(repo, "read-tree", base, environment=environment)
        _git_bytes(
            repo,
            "update-index",
            "--force-remove",
            "--",
            *(path.as_posix() for path in deleted),
            environment=environment,
        )
        tree = (
            _git_bytes(
                repo,
                "write-tree",
                environment=environment,
            )
            .decode("ascii")
            .strip()
        )
    return _require_oid(tree, object_format, label="projected post-deletion tree")


def _target_reference_variants(paths: Sequence[Path]) -> set[str]:
    variants: set[str] = set()
    for path in paths:
        without_suffix = path.with_suffix("")
        relative = Path(*path.parts[1:])
        relative_without_suffix = relative.with_suffix("")
        variants.update(
            {
                path.as_posix(),
                relative.as_posix(),
                without_suffix.as_posix(),
                relative_without_suffix.as_posix(),
                f"{path.parts[0]}:{relative_without_suffix.as_posix()}",
            }
        )
    return variants


def _json_strings(value: object) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, str):
        result.append(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            result.append(key)
            result.extend(_json_strings(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_json_strings(child))
    return tuple(result)


_REFERENCE_LEFT_TOKEN_BYTES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)
_REFERENCE_RIGHT_TOKEN_BYTES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/-"
)


def _contains_target_reference(raw: bytes, patterns: Sequence[bytes]) -> bool:
    """Match target path tokens while allowing absolute/path-prefixed forms."""

    for pattern in patterns:
        start = 0
        while True:
            index = raw.find(pattern, start)
            if index < 0:
                break
            end = index + len(pattern)
            before_is_token = (
                index > 0 and raw[index - 1] in _REFERENCE_LEFT_TOKEN_BYTES
            )
            after_is_token = (
                end < len(raw) and raw[end] in _REFERENCE_RIGHT_TOKEN_BYTES
            )
            if not before_is_token and not after_is_token:
                return True
            start = index + 1
    return False


def _provenance_inventory(
    repo: Path,
    *,
    base: str,
    object_format: str,
    deleted: Sequence[Path],
) -> tuple[str, tuple[LegacyCleanupProvenanceRecord, ...]]:
    entries = _tree_inventory(
        repo,
        base,
        object_format,
        LEGACY_CLEANUP_PROVENANCE_DIRS,
    )
    variants = _target_reference_variants(deleted)
    direct_manifests: set[Path] = set()
    for path in deleted:
        primary = (
            path.with_name(path.name.removesuffix(".test.yaml") + ".yaml")
            if path.name.endswith(".test.yaml")
            else path
        )
        for candidate in (primary, Path(*primary.parts[1:])):
            direct_manifests.add(
                Path(".axiom/encoding-manifests") / candidate.with_suffix(".json")
            )

    total_bytes = 0
    inventory: list[str] = []
    records: list[LegacyCleanupProvenanceRecord] = []
    for entry in entries:
        if entry.mode != "100644" or entry.object_type != "blob":
            raise LegacyCleanupGitError(
                "base provenance record is not a regular 100644 blob: "
                f"{entry.path.as_posix()}"
            )
        if entry.path.suffix != ".json":
            raise LegacyCleanupGitError(
                f"base provenance record is not JSON: {entry.path.as_posix()}"
            )
        raw = _blob(
            repo,
            entry,
            max_bytes=_MAX_PROVENANCE_BLOB_BYTES,
            label=f"base provenance {entry.path.as_posix()}",
        )
        total_bytes += len(raw)
        if total_bytes > _MAX_PROVENANCE_TOTAL_BYTES:
            raise LegacyCleanupGitError("base provenance inventory exceeds its limit")
        try:
            payload = decode_strict_json_object(
                raw,
                label=f"base provenance {entry.path.as_posix()}",
                max_bytes=_MAX_PROVENANCE_BLOB_BYTES,
            )
        except LegacyCleanupReceiptError as exc:
            raise LegacyCleanupGitError(str(exc)) from exc
        if entry.path in direct_manifests:
            raise LegacyCleanupGitError(
                f"base canonical manifest already owns {entry.path.as_posix()}"
            )
        for value in _json_strings(payload):
            if value in variants or any(
                value.startswith(f"{variant}#")
                for variant in variants
                if ":" in variant
            ):
                raise LegacyCleanupGitError(
                    "base provenance already covers a cleanup target: "
                    f"{entry.path.as_posix()}"
                )
        sha256 = hashlib.sha256(raw).hexdigest()
        inventory.append(
            "\0".join(
                (
                    entry.path.as_posix(),
                    entry.mode,
                    entry.oid,
                    sha256,
                )
            )
        )
        records.append(
            LegacyCleanupProvenanceRecord(
                path=entry.path,
                mode=entry.mode,
                blob_oid=entry.oid,
                sha256=sha256,
                raw=raw,
            )
        )
    digest = hashlib.sha256("\n".join(inventory).encode("utf-8")).hexdigest()
    return digest, tuple(records)


def _surviving_reference_inventory(
    repo: Path,
    *,
    base: str,
    object_format: str,
    deleted: Sequence[Path],
) -> tuple[str, int]:
    target_set = set(deleted)
    patterns = tuple(
        sorted(
            (
                variant.encode("utf-8")
                for variant in _target_reference_variants(deleted)
            ),
            key=lambda value: (-len(value), value),
        )
    )
    total_bytes = 0
    inventory: list[str] = []
    count = 0
    for entry in _tree_inventory(repo, base, object_format):
        if entry.path in target_set:
            continue
        if entry.object_type != "blob" or entry.size is None:
            raise LegacyCleanupGitError(
                "surviving base contains an unreadable non-blob entry: "
                f"{entry.path.as_posix()}"
            )
        raw = _blob(
            repo,
            entry,
            max_bytes=_MAX_BASE_BLOB_BYTES,
            label=f"surviving base blob {entry.path.as_posix()}",
        )
        total_bytes += len(raw)
        if total_bytes > _MAX_REFERENCE_TOTAL_BYTES:
            raise LegacyCleanupGitError(
                "surviving reference inventory exceeds its byte limit"
            )
        if _contains_target_reference(raw, patterns):
            raise LegacyCleanupGitError(
                "surviving base blob references a cleanup target: "
                f"{entry.path.as_posix()}"
            )
        inventory.append(
            "\0".join(
                (
                    entry.path.as_posix(),
                    entry.mode,
                    entry.oid,
                    hashlib.sha256(raw).hexdigest(),
                )
            )
        )
        count += 1
    return hashlib.sha256("\n".join(inventory).encode("utf-8")).hexdigest(), count


def _protected_base_files(
    repo: Path,
    entries: Mapping[Path, _TreeEntry],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    raw_files: dict[Path, bytes] = {}
    for key, path in (
        ("toolchain", LEGACY_CLEANUP_TOOLCHAIN_PATH),
        ("validation_waiver_set", LEGACY_CLEANUP_WAIVER_PATH),
    ):
        entry = entries.get(path)
        if entry is None:
            raise LegacyCleanupGitError(f"legacy cleanup base is missing {path}")
        records[key] = _file_record(repo, entry, deletion=False)
        raw_files[path] = _blob(
            repo,
            entry,
            max_bytes=_MAX_BASE_BLOB_BYTES,
            label=f"protected base file {path.as_posix()}",
        )

    try:
        parsed = tomllib.loads(raw_files[LEGACY_CLEANUP_TOOLCHAIN_PATH].decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LegacyCleanupGitError(
            "legacy cleanup base toolchain is not valid UTF-8 TOML"
        ) from exc
    values = parsed.get("toolchain") if isinstance(parsed, dict) else None
    expected = {
        "axiom_corpus_release",
        "axiom_corpus_release_content_sha256",
        "validation_waiver_set_sha256",
    }
    if (
        set(parsed) != {"toolchain"}
        or not isinstance(values, dict)
        or set(values) != expected
    ):
        raise LegacyCleanupGitError(
            "legacy cleanup base toolchain has a noncanonical field set"
        )
    result = {key: value for key, value in values.items() if isinstance(value, str)}
    if set(result) != expected or any(not value for value in result.values()):
        raise LegacyCleanupGitError("legacy cleanup base toolchain pins are malformed")
    for key in (
        "axiom_corpus_release_content_sha256",
        "validation_waiver_set_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", result[key]) is None:
            raise LegacyCleanupGitError(
                "legacy cleanup base toolchain digest is malformed"
            )
    if (
        result["validation_waiver_set_sha256"]
        != records["validation_waiver_set"]["base_sha256"]
    ):
        raise LegacyCleanupGitError(
            "legacy cleanup base waiver bytes disagree with the toolchain pin"
        )
    return records, result


def plan_legacy_cleanup_base(
    repo: Path,
    *,
    base_ref: str,
    primary_paths: Sequence[str | Path],
    require_clean_checkout: bool,
) -> LegacyCleanupBasePlan:
    """Prove exact eligibility from one immutable protected base ``B``."""

    checkout = Path(repo).resolve(strict=True)
    repository = canonical_rulespec_repository(checkout)
    object_format = _git_text(checkout, "rev-parse", "--show-object-format").strip()
    _oid_length(object_format)
    base = _full_commit(checkout, base_ref, object_format)
    base_tree = _require_oid(
        _git_text(checkout, "rev-parse", f"{base}^{{tree}}").strip(),
        object_format,
        label="legacy cleanup base tree",
    )
    primaries = canonical_primary_paths(primary_paths)
    deleted = tuple(
        path for primary in primaries for path in (primary, companion_path(primary))
    )

    entries = _entry_map(_tree_inventory(checkout, base, object_format))
    groups: list[dict[str, dict[str, str]]] = []
    for primary in primaries:
        companion = companion_path(primary)
        primary_entry = entries.get(primary)
        companion_entry = entries.get(companion)
        if primary_entry is None or companion_entry is None:
            missing = primary if primary_entry is None else companion
            raise LegacyCleanupGitError(
                f"legacy cleanup base is missing complete group member {missing}"
            )
        groups.append(
            {
                "primary": _file_record(checkout, primary_entry, deletion=True),
                "companion": _file_record(checkout, companion_entry, deletion=True),
            }
        )

    base_files, toolchain_values = _protected_base_files(checkout, entries)
    ownership_digest, provenance_records = _provenance_inventory(
        checkout,
        base=base,
        object_format=object_format,
        deleted=deleted,
    )
    reference_digest, surviving_count = _surviving_reference_inventory(
        checkout,
        base=base,
        object_format=object_format,
        deleted=deleted,
    )
    projected_tree = _projected_tree(
        checkout,
        base=base,
        object_format=object_format,
        deleted=deleted,
    )

    if require_clean_checkout:
        head = _git_text(checkout, "rev-parse", "HEAD").strip()
        if head != base:
            raise LegacyCleanupGitError(
                "legacy cleanup requires clean HEAD to equal the exact protected base"
            )
        status_raw = _git_bytes(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored",
            "-z",
        )
        if status_raw:
            raise LegacyCleanupGitError(
                "legacy cleanup requires no staged, unstaged, untracked, or ignored files"
            )
        for path in (
            *deleted,
            LEGACY_CLEANUP_TOOLCHAIN_PATH,
            LEGACY_CLEANUP_WAIVER_PATH,
        ):
            entry = entries[path]
            live = _safe_live_bytes(checkout, path)
            base_raw = _blob(
                checkout,
                entry,
                max_bytes=_MAX_BASE_BLOB_BYTES,
                label=f"immutable base blob {path.as_posix()}",
            )
            if live != base_raw:
                raise LegacyCleanupGitError(
                    f"legacy cleanup live preimage differs from base: {path.as_posix()}"
                )

    return LegacyCleanupBasePlan(
        repository=repository,
        object_format=object_format,
        base_commit=base,
        base_tree=base_tree,
        projected_post_deletion_tree=projected_tree,
        groups=tuple(groups),
        base_files=base_files,
        toolchain_values=toolchain_values,
        ownership_inventory_sha256=ownership_digest,
        provenance_records=provenance_records,
        surviving_reference_inventory_sha256=reference_digest,
        surviving_blob_count=surviving_count,
    )


def plan_payload_issues(
    payload: Mapping[str, object],
    plan: LegacyCleanupBasePlan,
) -> list[str]:
    """Compare all immutable-base fields with a freshly reconstructed proof."""

    issues: list[str] = []
    expected_repository = {
        "repository": plan.repository,
        "object_format": plan.object_format,
        "base_commit": plan.base_commit,
        "base_tree": plan.base_tree,
        "projected_post_deletion_tree": plan.projected_post_deletion_tree,
    }
    if payload.get("repository") != expected_repository:
        issues.append("legacy cleanup repository/base/projected-tree binding is stale")
    if payload.get("groups") != list(plan.groups):
        issues.append("legacy cleanup base blob/path/mode groups are stale")
    toolchain = payload.get("toolchain")
    if (
        not isinstance(toolchain, dict)
        or toolchain.get("base_files") != plan.base_files
    ):
        issues.append("legacy cleanup protected base-file pins are stale")
    base_proof = payload.get("base_proof")
    if not isinstance(base_proof, dict):
        issues.append("legacy cleanup immutable-base proof is missing")
    else:
        if base_proof.get("ownership_inventory_sha256") != (
            plan.ownership_inventory_sha256
        ):
            issues.append("legacy cleanup ownership inventory proof is stale")
        if base_proof.get("provenance_record_count") != len(plan.provenance_records):
            issues.append("legacy cleanup provenance-record count is stale")
        if base_proof.get("surviving_reference_inventory_sha256") != (
            plan.surviving_reference_inventory_sha256
        ):
            issues.append("legacy cleanup surviving-reference proof is stale")
        if base_proof.get("surviving_blob_count") != plan.surviving_blob_count:
            issues.append("legacy cleanup surviving-reference count is stale")
    return issues
