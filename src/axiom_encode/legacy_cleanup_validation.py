"""Executed validation for an atomic unmanifested-legacy contraction.

The validator materializes the exact protected base in a private temporary
checkout, removes only the base-plan paths, proves the projected Git tree, and
then executes the complete post-deletion validation matrix.  It deliberately
does not create a commit: the only durable transition remains the eventual
base-to-PR-head receipt addition plus its authorized deletions.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from base64 import b64encode
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import yaml

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS
from .legacy_cleanup import (
    LEGACY_CLEANUP_MAX_COMMAND_TOKEN_CHARS,
    LEGACY_CLEANUP_MAX_COMMAND_TOKENS,
    LEGACY_CLEANUP_MAX_PATH_CHARS,
    LEGACY_CLEANUP_VALIDATION_CHECKS,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    LegacyCleanupReceiptError,
    canonical_primary_path,
    companion_path,
    decode_strict_json_object,
)
from .legacy_cleanup_git import LegacyCleanupBasePlan
from .toolchain import RuleSpecToolchainError, local_corpus_release_verification

LEGACY_CLEANUP_MAX_VALIDATION_TARGETS = 100_000
LEGACY_CLEANUP_MAX_TARGET_LIST_BYTES = 8 * 1024 * 1024
LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES = 4 * 1024 * 1024
LEGACY_CLEANUP_MAX_ACTUAL_COMMANDS = 1_024
LEGACY_CLEANUP_MAX_COMMAND_BYTES = 512 * 1024
LEGACY_CLEANUP_MAX_TRACKED_FILE_BYTES = 16 * 1024 * 1024
LEGACY_CLEANUP_MAX_TRACKED_TOTAL_BYTES = 1024 * 1024 * 1024
LEGACY_CLEANUP_COMMAND_TIMEOUT_SECONDS = 2 * 60 * 60

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_JURISDICTION_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
_EVIDENCE_FIELDS = {
    "schema",
    "status",
    "engine_execution",
    "projected_post_deletion_tree",
    "checks",
}
_CHECK_FIELDS = {
    "name",
    "command",
    "target_count",
    "target_list_sha256",
    "exit_code",
    "output_sha256",
}


class LegacyCleanupValidationError(ValueError):
    """The projected state or one required validation command is invalid."""


@dataclass(frozen=True, slots=True)
class ValidationCommandResult:
    """Bounded combined stdout/stderr returned by a validation command."""

    exit_code: int
    output: bytes


class ValidationCommandRunner(Protocol):
    """Dependency-injection boundary used only to isolate unit tests."""

    def __call__(
        self,
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        output_limit: int,
    ) -> ValidationCommandResult: ...


@dataclass(frozen=True, slots=True)
class _TrackedEntry:
    mode: str
    oid: str
    path: Path


@dataclass(frozen=True, slots=True)
class _CheckoutState:
    head: str
    tree: str
    index_tree: str


@dataclass(slots=True)
class _ExecutionBudget:
    commands: int = 0


def _sanitized_git_environment() -> dict[str, str]:
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


def _git_bytes(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
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
            "-C",
            str(repo),
            *arguments,
        ],
        check=False,
        capture_output=True,
        env=_sanitized_git_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LegacyCleanupValidationError(
            "projected validation Git command failed "
            f"({' '.join(arguments)}): {detail or 'no diagnostic'}"
        )
    return completed.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repo, *arguments).decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise LegacyCleanupValidationError(
            "projected validation Git output is not UTF-8"
        ) from exc


def _canonical_directory(raw: Path, *, label: str) -> Path:
    path = Path(raw)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise LegacyCleanupValidationError(f"{label} is unavailable: {path}") from exc
    if not resolved.is_dir():
        raise LegacyCleanupValidationError(f"{label} is not a directory: {resolved}")
    return resolved


def _clean_checkout_state(repo: Path, *, label: str) -> _CheckoutState:
    top = Path(_git_text(repo, "rev-parse", "--show-toplevel")).resolve()
    if top != repo:
        raise LegacyCleanupValidationError(f"{label} is not an exact Git checkout")
    head = _git_text(repo, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_text(repo, "rev-parse", "--verify", "HEAD^{tree}")
    index_tree = _git_text(repo, "write-tree")
    status = _git_bytes(
        repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "-z",
    )
    if status:
        raise LegacyCleanupValidationError(f"{label} must be a clean Git checkout")
    if _OID_RE.fullmatch(head) is None or _OID_RE.fullmatch(tree) is None:
        raise LegacyCleanupValidationError(f"{label} has malformed Git identities")
    if index_tree != tree:
        raise LegacyCleanupValidationError(f"{label} index does not equal HEAD")
    _assert_index_matches_worktree(repo, label=label)
    return _CheckoutState(head=head, tree=tree, index_tree=index_tree)


def _assert_checkout_state(
    repo: Path,
    expected: _CheckoutState,
    *,
    label: str,
) -> None:
    actual = _clean_checkout_state(repo, label=label)
    if actual != expected:
        raise LegacyCleanupValidationError(f"{label} mutated during validation")


def _source_base_state(repo: Path, plan: LegacyCleanupBasePlan) -> _CheckoutState:
    state = _clean_checkout_state(repo, label="source RuleSpec checkout")
    if state.head != plan.base_commit or state.tree != plan.base_tree:
        raise LegacyCleanupValidationError(
            "source RuleSpec checkout no longer equals the exact protected base"
        )
    return state


def _clone_exact_base(
    source: Path,
    destination: Path,
    plan: LegacyCleanupBasePlan,
) -> None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--no-checkout",
            "--no-tags",
            "--",
            str(source),
            str(destination),
        ],
        check=False,
        capture_output=True,
        env=_sanitized_git_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LegacyCleanupValidationError(
            f"cannot clone exact protected base: {detail or 'git clone failed'}"
        )
    _git_bytes(destination, "checkout", "--detach", "--force", plan.base_commit)
    _git_bytes(
        destination,
        "remote",
        "set-url",
        "origin",
        f"https://{plan.repository}.git",
    )
    if _git_text(destination, "rev-parse", "--show-object-format") != (
        plan.object_format
    ):
        raise LegacyCleanupValidationError(
            "projected checkout Git object format differs from the base plan"
        )
    if _git_text(destination, "rev-parse", "HEAD") != plan.base_commit:
        raise LegacyCleanupValidationError(
            "projected checkout did not materialize the exact protected base"
        )
    if _git_text(destination, "rev-parse", "HEAD^{tree}") != plan.base_tree:
        raise LegacyCleanupValidationError(
            "projected checkout base tree differs from the base plan"
        )


def _tracked_entries(repo: Path) -> tuple[_TrackedEntry, ...]:
    raw = _git_bytes(repo, "ls-files", "--stage", "-z")
    entries: list[_TrackedEntry] = []
    seen: set[Path] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, encoded_path = record.split(b"\t", 1)
            mode, oid, stage = metadata.decode("ascii").split()
            text = encoded_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise LegacyCleanupValidationError(
                "projected checkout tracked-file index is malformed"
            ) from exc
        path = Path(text)
        if (
            stage != "0"
            or path.is_absolute()
            or path.as_posix() != text
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or len(text) > LEGACY_CLEANUP_MAX_PATH_CHARS
            or path in seen
        ):
            raise LegacyCleanupValidationError(
                "projected checkout tracked-file index is ambiguous"
            )
        seen.add(path)
        entries.append(_TrackedEntry(mode=mode, oid=oid, path=path))
        if len(entries) > LEGACY_CLEANUP_MAX_VALIDATION_TARGETS:
            raise LegacyCleanupValidationError(
                "projected checkout exceeds the tracked-target limit"
            )
    return tuple(sorted(entries, key=lambda item: item.path.as_posix()))


def _assert_default_index_flags(
    repo: Path,
    entries: Sequence[_TrackedEntry],
    *,
    label: str,
) -> None:
    """Reject index hints that can suppress ordinary worktree comparisons."""

    raw = _git_bytes(repo, "ls-files", "-v", "-z")
    tagged: dict[Path, str] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise LegacyCleanupValidationError(f"{label} index flags are malformed")
        try:
            tag = record[:1].decode("ascii")
            text = record[2:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LegacyCleanupValidationError(
                f"{label} index flags are malformed"
            ) from exc
        path = Path(text)
        if path in tagged:
            raise LegacyCleanupValidationError(f"{label} index flags are ambiguous")
        tagged[path] = tag
    expected = {entry.path for entry in entries}
    if set(tagged) != expected:
        raise LegacyCleanupValidationError(f"{label} index flags are ambiguous")
    nondefault = next(
        (
            (path, tag)
            for path, tag in sorted(tagged.items(), key=lambda item: item[0].as_posix())
            if tag != "H"
        ),
        None,
    )
    if nondefault is not None:
        path, tag = nondefault
        raise LegacyCleanupValidationError(
            f"{label} uses a prohibited Git index flag ({tag}): {path.as_posix()}"
        )


def _git_blob_oid(raw: bytes, *, object_format: str) -> str:
    if object_format == "sha1":
        digest = hashlib.sha1(usedforsecurity=False)
    elif object_format == "sha256":
        digest = hashlib.sha256()
    else:
        raise LegacyCleanupValidationError(
            "projected validation Git object format is unsupported"
        )
    digest.update(b"blob ")
    digest.update(str(len(raw)).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _safe_symlink_bytes(repo: Path, relative: Path, *, label: str) -> bytes:
    cursor = repo
    metadata = None
    for index, part in enumerate(relative.parts):
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise LegacyCleanupValidationError(f"{label} is unavailable") from exc
        if index < len(relative.parts) - 1:
            if stat.S_ISLNK(metadata.st_mode):
                raise LegacyCleanupValidationError(f"{label} has a symlink ancestor")
            if not stat.S_ISDIR(metadata.st_mode):
                raise LegacyCleanupValidationError(
                    f"{label} has a non-directory ancestor"
                )
    if metadata is None or not stat.S_ISLNK(metadata.st_mode):
        raise LegacyCleanupValidationError(f"{label} is not a symbolic link")
    try:
        target = os.readlink(cursor)
        after = cursor.lstat()
    except OSError as exc:
        raise LegacyCleanupValidationError(f"{label} cannot be read safely") from exc
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise LegacyCleanupValidationError(f"{label} changed while being read")
    return os.fsencode(target)


def _assert_index_matches_worktree(repo: Path, *, label: str) -> None:
    """Compare every stage-0 entry directly with worktree bytes and mode.

    Git's normal diff/status machinery intentionally honors assume-unchanged
    and skip-worktree hints.  Validation cannot: those hints would otherwise
    let a command hide a mutation from the projected-state proof.
    """

    entries = _tracked_entries(repo)
    _assert_default_index_flags(repo, entries, label=label)
    object_format = _git_text(repo, "rev-parse", "--show-object-format")
    total = 0
    for entry in entries:
        item_label = f"{label} tracked path {entry.path.as_posix()}"
        if entry.mode in {"100644", "100755"}:
            raw = _safe_regular_bytes(repo, entry.path, label=item_label)
            try:
                actual_mode = (repo / entry.path).lstat().st_mode
            except OSError as exc:
                raise LegacyCleanupValidationError(
                    f"{item_label} mode cannot be inspected"
                ) from exc
            executable = bool(actual_mode & 0o111)
            if executable != (entry.mode == "100755"):
                raise LegacyCleanupValidationError(
                    f"{item_label} mode differs from the Git index"
                )
        elif entry.mode == "120000":
            raw = _safe_symlink_bytes(repo, entry.path, label=item_label)
        else:
            raise LegacyCleanupValidationError(
                f"{item_label} has unsupported Git mode {entry.mode}"
            )
        total += len(raw)
        if total > LEGACY_CLEANUP_MAX_TRACKED_TOTAL_BYTES:
            raise LegacyCleanupValidationError(
                f"{label} tracked content exceeds its total byte limit"
            )
        if _git_blob_oid(raw, object_format=object_format) != entry.oid:
            raise LegacyCleanupValidationError(
                f"{item_label} bytes differ from the Git index"
            )


def _safe_regular_bytes(repo: Path, relative: Path, *, label: str) -> bytes:
    cursor = repo
    metadata = None
    for index, part in enumerate(relative.parts):
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise LegacyCleanupValidationError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LegacyCleanupValidationError(f"{label} has a symlink component")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise LegacyCleanupValidationError(
                f"{label} has a non-directory ancestor"
            )
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        raise LegacyCleanupValidationError(f"{label} is not a regular file")
    if metadata.st_size > LEGACY_CLEANUP_MAX_TRACKED_FILE_BYTES:
        raise LegacyCleanupValidationError(f"{label} exceeds its byte limit")
    try:
        raw = cursor.read_bytes()
        after = cursor.lstat()
    except OSError as exc:
        raise LegacyCleanupValidationError(f"{label} cannot be read safely") from exc
    before_identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != metadata.st_size:
        raise LegacyCleanupValidationError(f"{label} changed while being read")
    return raw


def _assert_absent(repo: Path, relative: Path) -> None:
    cursor = repo
    for index, part in enumerate(relative.parts):
        cursor /= part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise LegacyCleanupValidationError(
                f"cannot prove cleanup deletion remains absent: {relative}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise LegacyCleanupValidationError(
                f"cleanup deletion has a symlink component: {relative}"
            )
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise LegacyCleanupValidationError(
                f"cleanup deletion has a non-directory ancestor: {relative}"
            )
    raise LegacyCleanupValidationError(
        f"cleanup deletion reappeared during validation: {relative}"
    )


def _planned_records(plan: LegacyCleanupBasePlan) -> dict[Path, Mapping[str, str]]:
    records: dict[Path, Mapping[str, str]] = {}
    for group in plan.groups:
        if not isinstance(group, dict) or set(group) != {"primary", "companion"}:
            raise LegacyCleanupValidationError("base plan has malformed cleanup groups")
        for key in ("primary", "companion"):
            record = group[key]
            if not isinstance(record, dict):
                raise LegacyCleanupValidationError(
                    "base plan has malformed cleanup file evidence"
                )
            try:
                path = Path(record["path"])
            except (KeyError, TypeError) as exc:
                raise LegacyCleanupValidationError(
                    "base plan cleanup file path is malformed"
                ) from exc
            if path in records:
                raise LegacyCleanupValidationError(
                    "base plan cleanup deletion paths overlap"
                )
            records[path] = record
    if not records or len(records) > 128:
        raise LegacyCleanupValidationError(
            "base plan must contain 1..64 complete cleanup groups"
        )
    return records


def _delete_planned_paths(repo: Path, plan: LegacyCleanupBasePlan) -> tuple[Path, ...]:
    records = _planned_records(plan)
    entries = {entry.path: entry for entry in _tracked_entries(repo)}
    deleted = tuple(sorted(records, key=Path.as_posix))
    for path in deleted:
        record = records[path]
        entry = entries.get(path)
        if (
            entry is None
            or entry.mode != "100644"
            or record.get("base_mode") != "100644"
            or record.get("base_blob_oid") != entry.oid
            or record.get("result") != "absent"
        ):
            raise LegacyCleanupValidationError(
                f"projected deletion preimage differs from the base plan: {path}"
            )
        raw = _safe_regular_bytes(
            repo,
            path,
            label=f"projected deletion preimage {path.as_posix()}",
        )
        if hashlib.sha256(raw).hexdigest() != record.get("base_sha256"):
            raise LegacyCleanupValidationError(
                f"projected deletion bytes differ from the base plan: {path}"
            )
        try:
            (repo / path).unlink()
        except OSError as exc:
            raise LegacyCleanupValidationError(
                f"cannot remove projected cleanup path: {path}"
            ) from exc
    _git_bytes(
        repo,
        "update-index",
        "--force-remove",
        "--",
        *(path.as_posix() for path in deleted),
    )
    for path in deleted:
        _assert_absent(repo, path)
    return deleted


def _expected_cached_diff(deleted: Sequence[Path]) -> bytes:
    return b"".join(
        b"D\0" + path.as_posix().encode("utf-8") + b"\0"
        for path in sorted(deleted, key=Path.as_posix)
    )


def _assert_projected_state(
    repo: Path,
    plan: LegacyCleanupBasePlan,
    deleted: Sequence[Path],
) -> None:
    for path in deleted:
        _assert_absent(repo, path)
    tree = _git_text(repo, "write-tree")
    if tree != plan.projected_post_deletion_tree:
        raise LegacyCleanupValidationError(
            "projected post-deletion tree differs from the immutable base plan"
        )
    if _git_bytes(repo, "diff", "--"):
        raise LegacyCleanupValidationError(
            "projected validation command mutated a tracked worktree file"
        )
    _assert_index_matches_worktree(repo, label="projected RuleSpec checkout")
    cached = _git_bytes(
        repo,
        "diff",
        "--cached",
        "--name-status",
        "--no-renames",
        "-z",
        "HEAD",
        "--",
    )
    if cached != _expected_cached_diff(deleted):
        raise LegacyCleanupValidationError(
            "projected validation index differs from the exact deletion set"
        )
    untracked = _git_bytes(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    ignored = _git_bytes(
        repo,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "-z",
    )
    if untracked or ignored:
        raise LegacyCleanupValidationError(
            "projected validation command created an untracked or ignored path"
        )


def _is_primary(path: Path) -> bool:
    if path.name.endswith(".test.yaml") or path.suffix != ".yaml":
        return False
    try:
        return canonical_primary_path(path) == path
    except LegacyCleanupReceiptError:
        return False


def _is_companion(path: Path) -> bool:
    if not path.name.endswith(".test.yaml") or len(path.parts) < 3:
        return False
    primary = path.with_name(f"{path.name.removesuffix('.test.yaml')}.yaml")
    try:
        return companion_path(canonical_primary_path(primary)) == path
    except LegacyCleanupReceiptError:
        return False


def _discover_validation_targets(
    repo: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    entries = _tracked_entries(repo)
    entry_map = {entry.path: entry for entry in entries}
    primaries = tuple(entry.path for entry in entries if _is_primary(entry.path))
    companions = tuple(entry.path for entry in entries if _is_companion(entry.path))
    for path in (*primaries, *companions):
        if entry_map[path].mode != "100644":
            raise LegacyCleanupValidationError(
                f"surviving RuleSpec validation target is not 100644: {path}"
            )
    if not primaries:
        raise LegacyCleanupValidationError(
            "cleanup validation refuses a projected state with zero surviving primaries"
        )
    if not companions:
        raise LegacyCleanupValidationError(
            "cleanup validation refuses a projected state with zero surviving companions"
        )
    primary_set = set(primaries)
    orphaned = [
        path
        for path in companions
        if path.with_name(f"{path.name.removesuffix('.test.yaml')}.yaml")
        not in primary_set
    ]
    if orphaned:
        raise LegacyCleanupValidationError(
            "projected state contains an orphan companion: "
            + orphaned[0].as_posix()
        )
    repository_tests = tuple(
        entry.path
        for entry in entries
        if len(entry.path.parts) >= 2
        and entry.path.parts[0] == "tests"
        and entry.path.suffix == ".py"
        and (
            entry.path.name.startswith("test_")
            or entry.path.name.endswith("_test.py")
        )
    )
    tracked = tuple(entry.path for entry in entries)
    return primaries, companions, repository_tests, tracked


def _target_list_sha256(targets: Sequence[Path]) -> str:
    if len(targets) > LEGACY_CLEANUP_MAX_VALIDATION_TARGETS:
        raise LegacyCleanupValidationError("validation target list exceeds its limit")
    encoded: list[bytes] = []
    total = 0
    previous = ""
    for path in targets:
        text = path.as_posix()
        if (
            not text
            or path.is_absolute()
            or text != str(path)
            or len(text) > LEGACY_CLEANUP_MAX_PATH_CHARS
            or any(part in {"", ".", ".."} for part in path.parts)
            or text <= previous
        ):
            raise LegacyCleanupValidationError(
                "validation target list is not canonical, unique, and sorted"
            )
        previous = text
        raw = text.encode("utf-8")
        total += len(raw) + 1
        if total > LEGACY_CLEANUP_MAX_TARGET_LIST_BYTES:
            raise LegacyCleanupValidationError(
                "validation target list exceeds its byte limit"
            )
        encoded.append(raw)
    material = b"axiom-encode/legacy-cleanup-validation-target-list/v1\0"
    material += b"\0".join(encoded)
    return hashlib.sha256(material).hexdigest()


def _command_size(command: Sequence[str]) -> int:
    return sum(len(token.encode("utf-8")) + 1 for token in command)


def _validate_actual_command(command: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(command)
    if (
        not normalized
        or len(normalized) > LEGACY_CLEANUP_MAX_COMMAND_TOKENS
        or _command_size(normalized) > LEGACY_CLEANUP_MAX_COMMAND_BYTES
        or any(
            not isinstance(token, str)
            or not token
            or len(token) > LEGACY_CLEANUP_MAX_COMMAND_TOKEN_CHARS
            or "\0" in token
            or "\n" in token
            or "\r" in token
            for token in normalized
        )
    ):
        raise LegacyCleanupValidationError("validation command exceeds its bounds")
    return normalized


def _validate_portable_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if (
        not normalized
        or len(normalized) > 64
        or any(
            not isinstance(token, str)
            or not token
            or len(token) > LEGACY_CLEANUP_MAX_COMMAND_TOKEN_CHARS
            or "\0" in token
            or "\n" in token
            or "\r" in token
            or token.startswith("/")
            or re.match(r"^[A-Za-z]:[\\/]", token) is not None
            for token in normalized
        )
    ):
        raise LegacyCleanupValidationError(
            "portable validation command is malformed or contains a host path"
        )
    return normalized


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except (AttributeError, OSError):
        if process.poll() is None:
            process.kill()


def _bounded_subprocess_runner(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    output_limit: int,
) -> ValidationCommandResult:
    """Run one command without a shell and stop output growth at the limit."""

    with tempfile.TemporaryFile(prefix="axiom-cleanup-validation-output-") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise LegacyCleanupValidationError(
                f"cannot execute required validation command: {command[0]}"
            ) from exc
        deadline = time.monotonic() + LEGACY_CLEANUP_COMMAND_TIMEOUT_SECONDS
        while process.poll() is None:
            if output.tell() > output_limit:
                _kill_process_group(process)
                process.wait()
                raise LegacyCleanupValidationError(
                    "validation command output exceeds its byte limit"
                )
            if time.monotonic() > deadline:
                _kill_process_group(process)
                process.wait()
                raise LegacyCleanupValidationError("validation command timed out")
            time.sleep(0.01)
        # The session is private to this command.  Kill its process group even
        # after the leader exits successfully so a detached-in-the-background
        # descendant cannot mutate the projection after the state check.
        _kill_process_group(process)
        if output.tell() > output_limit:
            raise LegacyCleanupValidationError(
                "validation command output exceeds its byte limit"
            )
        output.seek(0)
        raw = output.read(output_limit + 1)
    if len(raw) > output_limit:
        raise LegacyCleanupValidationError(
            "validation command output exceeds its byte limit"
        )
    return ValidationCommandResult(exit_code=process.returncode, output=raw)


def _validation_environment(
    *,
    home: Path,
    repo: Path,
    dependency_roots: Sequence[Path],
) -> dict[str, str]:
    allowed = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    )
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    git_environment = _sanitized_git_environment()
    for name in (
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_LITERAL_PATHSPECS",
        "GIT_TERMINAL_PROMPT",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_OPTIONAL_LOCKS",
    ):
        environment[name] = git_environment[name]
    package_parent = Path(__file__).resolve().parent.parent
    environment.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(package_parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "AXIOM_RULESPEC_REPO_ROOTS": os.pathsep.join(
                str(path) for path in (repo, *dependency_roots)
            ),
        }
    )
    return environment


def _normalized_output(
    raw: bytes,
    *,
    replacements: Sequence[tuple[bytes, bytes]],
) -> bytes:
    normalized = raw.replace(b"\r\n", b"\n")
    normalized = re.sub(rb"\x1b\[[0-9;?]*[ -/]*[@-~]", b"", normalized)
    for source, destination in sorted(replacements, key=lambda pair: -len(pair[0])):
        if source:
            normalized = normalized.replace(source, destination)
    # Pytest prints elapsed wall time in an otherwise deterministic summary.
    # Normalize only that syntactic field; counts and substantive diagnostics
    # remain bound by the output digest.
    normalized = re.sub(
        rb"(?<=\bin )[0-9]+(?:\.[0-9]+)?s(?=(?:\s|$))",
        b"{elapsed}",
        normalized,
    )
    return normalized


def _output_sha256(outputs: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"axiom-encode/legacy-cleanup-validation-output/v1\0")
    for output in outputs:
        digest.update(len(output).to_bytes(8, "big"))
        digest.update(output)
    return digest.hexdigest()


def _portable_dependency_flag() -> tuple[str, ...]:
    return ("--rulespec-dependency-root", "{rulespec-dependency-root}")


def _actual_dependency_flags(roots: Sequence[Path]) -> tuple[str, ...]:
    return tuple(
        token
        for root in roots
        for token in ("--rulespec-dependency-root", str(root))
    )


def _internal_command(*arguments: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-P",
        "-B",
        "-m",
        "axiom_encode.legacy_cleanup_validation",
        *arguments,
    )


def verification_only_axiom_encode_command(
    public_keys_raw: Sequence[bytes],
) -> tuple[str, ...]:
    """Return a child prefix carrying only an authenticated public keyring.

    The returned command starts this module's verification wrapper. It never
    inherits or transports the parent signing-broker descriptor, and the
    canonical base64 arguments contain public verification material only.
    """

    candidates = tuple(public_keys_raw)
    if (
        not candidates
        or len(candidates) > 16
        or any(not isinstance(candidate, bytes) or len(candidate) != 32 for candidate in candidates)
    ):
        raise LegacyCleanupValidationError(
            "corpus verification keyring must contain 1..16 exact 32-byte keys"
        )
    encoded = tuple(sorted(b64encode(candidate).decode("ascii") for candidate in candidates))
    if len(set(encoded)) != len(encoded):
        raise LegacyCleanupValidationError(
            "corpus verification keyring must not contain duplicate keys"
        )
    return _internal_command(
        "axiom-encode-verification",
        *(
            token
            for key in encoded
            for token in ("--corpus-release-public-key", key)
        ),
        "--",
    )


def _verification_command_with_targets(
    command: Sequence[str],
    *,
    target_list: Path,
    target_prefix: Path | None = None,
    target_strip_components: int = 0,
) -> tuple[str, ...]:
    """Attach a bounded support-file target list to the verification wrapper."""

    selected = tuple(command)
    if (
        not selected
        or selected[-1] != "--"
        or "axiom-encode-verification" not in selected
        or target_strip_components < 0
        or target_strip_components > 8
    ):
        raise LegacyCleanupValidationError(
            "target-list execution requires the verification-only wrapper"
        )
    options: list[str] = ["--target-list", str(target_list)]
    if target_prefix is not None:
        options.extend(("--target-prefix", str(target_prefix)))
    if target_strip_components:
        options.extend(
            ("--target-strip-components", str(target_strip_components))
        )
    return _validate_actual_command((*selected[:-1], *options, "--"))


def _run_check(
    *,
    name: str,
    portable_command: Sequence[str],
    actual_commands: Sequence[Sequence[str]],
    targets: Sequence[Path],
    cwd: Path,
    environment: Mapping[str, str],
    runner: ValidationCommandRunner,
    output_limit: int,
    budget: _ExecutionBudget,
    state_check: Callable[[], None],
    output_replacements: Sequence[tuple[bytes, bytes]],
) -> dict[str, object]:
    if name not in LEGACY_CLEANUP_VALIDATION_CHECKS:
        raise LegacyCleanupValidationError(f"unknown validation check: {name}")
    portable = _validate_portable_command(portable_command)
    if not actual_commands:
        raise LegacyCleanupValidationError(
            f"required validation check did not schedule a command: {name}"
        )
    outputs: list[bytes] = []
    remaining = output_limit
    for raw_command in actual_commands:
        command = _validate_actual_command(raw_command)
        budget.commands += 1
        if budget.commands > LEGACY_CLEANUP_MAX_ACTUAL_COMMANDS:
            raise LegacyCleanupValidationError(
                "projected validation exceeds the command-count limit"
            )
        state_check()
        result = runner(
            command,
            cwd=cwd,
            environment=environment,
            output_limit=remaining,
        )
        if (
            not isinstance(result, ValidationCommandResult)
            or type(result.exit_code) is not int
            or not isinstance(result.output, bytes)
        ):
            raise LegacyCleanupValidationError(
                f"validation runner returned no executed result for {name}"
            )
        if len(result.output) > remaining:
            raise LegacyCleanupValidationError(
                f"validation command output exceeds its byte limit: {name}"
            )
        normalized = _normalized_output(
            result.output,
            replacements=output_replacements,
        )
        outputs.append(normalized)
        remaining -= len(result.output)
        state_check()
        if result.exit_code != 0:
            detail = normalized[-4096:].decode("utf-8", errors="replace").strip()
            raise LegacyCleanupValidationError(
                f"required validation check failed ({name}, exit "
                f"{result.exit_code}): {detail or 'no diagnostic'}"
            )
    return {
        "name": name,
        "command": portable,
        "target_count": len(targets),
        "target_list_sha256": _target_list_sha256(targets),
        "exit_code": 0,
        "output_sha256": _output_sha256(outputs),
    }


def _write_support_file(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
    except OSError as exc:
        raise LegacyCleanupValidationError(
            f"cannot create projected validation support input: {path.name}"
        ) from exc


def _target_list_bytes(targets: Sequence[Path]) -> bytes:
    _target_list_sha256(targets)
    if not targets:
        raise LegacyCleanupValidationError("validation target list must be nonempty")
    return "".join(f"{path.as_posix()}\n" for path in targets).encode("utf-8")


def _assert_support_inputs(
    support: Path,
    expected: Mapping[str, bytes],
) -> None:
    try:
        actual_names = {path.name for path in support.iterdir()}
    except OSError as exc:
        raise LegacyCleanupValidationError(
            "projected validation support inputs are unavailable"
        ) from exc
    if actual_names != set(expected):
        raise LegacyCleanupValidationError(
            "projected validation support input set mutated"
        )
    for name, bound in expected.items():
        raw = _safe_regular_bytes(
            support,
            Path(name),
            label=f"projected validation support input {name}",
        )
        if raw != bound:
            raise LegacyCleanupValidationError(
                f"projected validation support input mutated: {name}"
            )


def _install_dependency_links(
    temporary: Path,
    *,
    projected: Path,
    corpus: Path,
    engine: Path,
    dependency_roots: Sequence[Path],
) -> None:
    links = {
        "axiom-corpus": corpus,
        "axiom-rules-engine": engine,
    }
    for dependency in dependency_roots:
        if re.fullmatch(r"rulespec-[a-z]{2}", dependency.name) is None:
            raise LegacyCleanupValidationError(
                "RuleSpec dependency checkout name is not canonical"
            )
        if dependency.name in links:
            raise LegacyCleanupValidationError(
                "RuleSpec dependency checkout names overlap"
            )
        links[dependency.name] = dependency
    if projected.name in links:
        raise LegacyCleanupValidationError(
            "projected repository name overlaps a validation dependency"
        )
    for name, target in links.items():
        try:
            (temporary / name).symlink_to(target, target_is_directory=True)
        except OSError as exc:
            raise LegacyCleanupValidationError(
                f"cannot install isolated validation dependency link: {name}"
            ) from exc


def _base_blob(repo: Path, commit: str, path: Path) -> bytes:
    return _git_bytes(repo, "show", f"{commit}:{path.as_posix()}")


def _group_companions(companions: Sequence[Path]) -> tuple[tuple[Path, ...], ...]:
    grouped: dict[str, list[Path]] = {}
    for path in companions:
        grouped.setdefault(path.parts[0], []).append(path)
    return tuple(
        tuple(sorted(grouped[jurisdiction], key=Path.as_posix))
        for jurisdiction in sorted(grouped)
    )


def _canonical_command_prefix(command: Sequence[str] | None) -> tuple[str, ...]:
    if command is None:
        raise LegacyCleanupValidationError(
            "projected validation requires a verification-only axiom-encode "
            "command prefix"
        )
    selected = tuple(command)
    normalized = _validate_actual_command(selected)
    if "axiom-encode-verification" not in normalized or normalized[-1] != "--":
        raise LegacyCleanupValidationError(
            "projected validation command is not the verification-only wrapper"
        )
    return normalized


def execute_projected_validation(
    repo: Path,
    plan: LegacyCleanupBasePlan,
    *,
    corpus_checkout: Path,
    rules_engine_checkout: Path,
    rulespec_dependency_roots: Sequence[Path] = (),
    axiom_encode_command: Sequence[str] | None = None,
    command_runner: ValidationCommandRunner | None = None,
) -> dict[str, object]:
    """Execute and attest the complete validation matrix on ``B - deletions``.

    ``command_runner`` is an explicit unit-test seam.  When it is omitted,
    every named command is launched as a real bounded subprocess and any
    nonzero exit status aborts without returning passing evidence.
    """

    if not isinstance(plan, LegacyCleanupBasePlan):
        raise LegacyCleanupValidationError("projected validation requires a base plan")
    source = _canonical_directory(repo, label="source RuleSpec checkout")
    source_state = _source_base_state(source, plan)
    corpus = _canonical_directory(corpus_checkout, label="axiom-corpus checkout")
    engine = _canonical_directory(
        rules_engine_checkout,
        label="axiom-rules-engine checkout",
    )
    dependencies = tuple(
        _canonical_directory(path, label="RuleSpec dependency checkout")
        for path in rulespec_dependency_roots
    )
    all_checkouts = (source, corpus, engine, *dependencies)
    if len(set(all_checkouts)) != len(all_checkouts):
        raise LegacyCleanupValidationError(
            "validation checkouts must be unique and externally pinned"
        )
    external_states = {
        corpus: _clean_checkout_state(corpus, label="axiom-corpus checkout"),
        engine: _clean_checkout_state(engine, label="axiom-rules-engine checkout"),
        **{
            path: _clean_checkout_state(path, label="RuleSpec dependency checkout")
            for path in dependencies
        },
    }
    encoder = _canonical_command_prefix(axiom_encode_command)
    runner = command_runner or _bounded_subprocess_runner
    budget = _ExecutionBudget()

    with tempfile.TemporaryDirectory(prefix="axiom-cleanup-validation-") as raw_temp:
        temporary = Path(raw_temp)
        projected = temporary / source.name
        support = temporary / "support"
        home = temporary / "home"
        support.mkdir(mode=0o700)
        home.mkdir(mode=0o700)
        _clone_exact_base(source, projected, plan)
        _install_dependency_links(
            temporary,
            projected=projected,
            corpus=corpus,
            engine=engine,
            dependency_roots=dependencies,
        )
        deleted = _delete_planned_paths(projected, plan)
        _assert_projected_state(projected, plan, deleted)
        primaries, companions, repository_tests, tracked = (
            _discover_validation_targets(projected)
        )
        companion_groups = _group_companions(companions)

        protected_waiver = support / "protected-known-validation-gaps.yaml"
        changed_paths = support / "authorized-deletions.txt"
        deleted_targets = support / "deleted-targets.txt"
        primary_targets = support / "surviving-primary-targets.txt"
        waiver_path = Path("known-validation-gaps.yaml")
        deletion_lines = "".join(f"{path.as_posix()}\n" for path in deleted).encode()
        support_inputs: dict[str, bytes] = {
            protected_waiver.name: _base_blob(
                source,
                plan.base_commit,
                waiver_path,
            ),
            changed_paths.name: deletion_lines,
            deleted_targets.name: deletion_lines,
            primary_targets.name: _target_list_bytes(primaries),
        }
        companion_target_lists: list[tuple[tuple[Path, ...], Path]] = []
        for index, group in enumerate(companion_groups):
            target_list = support / f"surviving-companion-targets-{index:04d}.txt"
            support_inputs[target_list.name] = _target_list_bytes(group)
            companion_target_lists.append((group, target_list))
        for name, raw in support_inputs.items():
            _write_support_file(support / name, raw)

        environment = _validation_environment(
            home=home,
            repo=projected,
            dependency_roots=dependencies,
        )

        def state_check() -> None:
            _assert_projected_state(projected, plan, deleted)
            _assert_support_inputs(support, support_inputs)
            _assert_checkout_state(
                source,
                source_state,
                label="source RuleSpec checkout",
            )
            for checkout, expected in external_states.items():
                label = (
                    "axiom-corpus checkout"
                    if checkout == corpus
                    else "axiom-rules-engine checkout"
                    if checkout == engine
                    else "RuleSpec dependency checkout"
                )
                _assert_checkout_state(checkout, expected, label=label)

        replacements = [
            (str(projected).encode(), b"{projected-repository}"),
            (str(source).encode(), b"{source-repository}"),
            (str(corpus).encode(), b"{axiom-corpus}"),
            (str(engine).encode(), b"{axiom-rules-engine}"),
            (str(temporary).encode(), b"{validation-temporary-root}"),
            (str(Path(sys.executable).resolve()).encode(), b"{python}"),
            *[
                (str(path).encode(), b"{rulespec-dependency-root}")
                for path in dependencies
            ],
        ]
        dependency_flags = _actual_dependency_flags(dependencies)
        portable_dependency = (
            _portable_dependency_flag() if dependencies else ()
        )
        checks: list[dict[str, object]] = []

        repository_test_commands: tuple[tuple[str, ...], ...]
        if repository_tests:
            repository_test_commands = (
                (
                    sys.executable,
                    "-P",
                    "-B",
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "tests",
                ),
            )
            repository_test_portable = (
                "{python}",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "{repository-test-files}",
            )
        else:
            repository_test_commands = (
                _internal_command(
                    "repository-tests-presence",
                    "--repo",
                    str(projected),
                ),
            )
            repository_test_portable = (
                "{python}",
                "-m",
                "axiom_encode.legacy_cleanup_validation",
                "repository-tests-presence",
                "--repo",
                "{projected-repository}",
            )
        checks.append(
            _run_check(
                name="repository-tests",
                portable_command=repository_test_portable,
                actual_commands=repository_test_commands,
                targets=repository_tests,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        checks.append(
            _run_check(
                name="repository-layout",
                portable_command=(
                    "{python}",
                    "-m",
                    "axiom_encode.legacy_cleanup_validation",
                    "repository-layout",
                    "--repo",
                    "{projected-repository}",
                ),
                actual_commands=(
                    _internal_command(
                        "repository-layout",
                        "--repo",
                        str(projected),
                    ),
                ),
                targets=tracked,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        waiver_targets = tuple(sorted((waiver_path, *deleted), key=Path.as_posix))
        checks.append(
            _run_check(
                name="validation-waivers",
                portable_command=(
                    "axiom-encode",
                    "validation-waivers",
                    "audit",
                    "--root",
                    "{projected-repository}",
                    "--corpus-path",
                    "{axiom-corpus}",
                    "--protected-base",
                    "{protected-base-waivers}",
                    "--changed-paths",
                    "{authorized-deletions}",
                    "--axiom-rules-engine-path",
                    "{axiom-rules-engine}",
                    "--axiom-rules-engine-ref",
                    "{axiom-rules-engine-commit}",
                    *portable_dependency,
                ),
                actual_commands=(
                    (
                        *encoder,
                        "validation-waivers",
                        "audit",
                        "--root",
                        str(projected),
                        "--corpus-path",
                        str(corpus),
                        "--protected-base",
                        str(protected_waiver),
                        "--changed-paths",
                        str(changed_paths),
                        "--axiom-rules-engine-path",
                        str(engine),
                        "--axiom-rules-engine-ref",
                        external_states[engine].head,
                        *dependency_flags,
                    ),
                ),
                targets=waiver_targets,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        checks.append(
            _run_check(
                name="remaining-rulespec-validation",
                portable_command=(
                    "axiom-encode",
                    "validate",
                    "{all-surviving-primary-files}",
                    "--skip-reviewers",
                    "--corpus-path",
                    "{axiom-corpus}",
                    "--axiom-rules-engine-path",
                    "{axiom-rules-engine}",
                    "--axiom-rules-engine-ref",
                    "{axiom-rules-engine-commit}",
                    *portable_dependency,
                ),
                actual_commands=(
                    (
                        *_verification_command_with_targets(
                            encoder,
                            target_list=primary_targets,
                            target_prefix=projected,
                        ),
                        "validate",
                        "--skip-reviewers",
                        "--corpus-path",
                        str(corpus),
                        "--axiom-rules-engine-path",
                        str(engine),
                        "--axiom-rules-engine-ref",
                        external_states[engine].head,
                        *dependency_flags,
                    ),
                ),
                targets=primaries,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        companion_commands = tuple(
            (
                *_verification_command_with_targets(
                    encoder,
                    target_list=target_list,
                    target_strip_components=1,
                ),
                "test",
                "--root",
                str(projected / group[0].parts[0]),
                "--axiom-rules-engine-path",
                str(engine),
                "--axiom-rules-engine-ref",
                external_states[engine].head,
                *dependency_flags,
            )
            for group, target_list in companion_target_lists
        )
        checks.append(
            _run_check(
                name="remaining-companion-tests",
                portable_command=(
                    "axiom-encode",
                    "test",
                    "--root",
                    "{jurisdiction-root}",
                    "--axiom-rules-engine-path",
                    "{axiom-rules-engine}",
                    "--axiom-rules-engine-ref",
                    "{axiom-rules-engine-commit}",
                    *portable_dependency,
                    "{all-surviving-companion-files-by-jurisdiction}",
                ),
                actual_commands=companion_commands,
                targets=companions,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        checks.append(
            _run_check(
                name="remaining-proof-validation",
                portable_command=(
                    "axiom-encode",
                    "proof-validate",
                    "{all-surviving-primary-files}",
                    "--corpus-path",
                    "{axiom-corpus}",
                ),
                actual_commands=(
                    (
                        *_verification_command_with_targets(
                            encoder,
                            target_list=primary_targets,
                            target_prefix=projected,
                        ),
                        "proof-validate",
                        "--corpus-path",
                        str(corpus),
                    ),
                ),
                targets=primaries,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        money_ratchet = Path("known-missing-money-atoms.yaml")
        money_ratchet_arguments = (
            ("--ratchet-file", str(projected / money_ratchet))
            if money_ratchet in set(tracked)
            else ()
        )
        money_ratchet_portable = (
            ("--ratchet-file", "{money-atom-ratchet}")
            if money_ratchet_arguments
            else ()
        )
        checks.append(
            _run_check(
                name="money-atom-proof-validation",
                portable_command=(
                    "axiom-encode",
                    "proof-validate",
                    "{all-surviving-primary-files}",
                    "--money-atoms-only",
                    "--corpus-path",
                    "{axiom-corpus}",
                    *money_ratchet_portable,
                ),
                actual_commands=(
                    (
                        *_verification_command_with_targets(
                            encoder,
                            target_list=primary_targets,
                            target_prefix=projected,
                        ),
                        "proof-validate",
                        "--money-atoms-only",
                        "--corpus-path",
                        str(corpus),
                        *money_ratchet_arguments,
                    ),
                ),
                targets=primaries,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        checks.append(
            _run_check(
                name="oracle-coverage",
                portable_command=(
                    "axiom-encode",
                    "oracle-coverage",
                    "--root",
                    "{projected-repository}",
                    "--fail-on-unmapped",
                    "--fail-on-untested-comparable",
                    "--fail-on-incomplete-comparable",
                    "--fail-on-stale-pending",
                    "--fail-on-empty",
                    "--limit",
                    "50",
                ),
                actual_commands=(
                    (
                        *encoder,
                        "oracle-coverage",
                        "--root",
                        str(projected),
                        "--fail-on-unmapped",
                        "--fail-on-untested-comparable",
                        "--fail-on-incomplete-comparable",
                        "--fail-on-stale-pending",
                        "--fail-on-empty",
                        "--limit",
                        "50",
                    ),
                ),
                targets=primaries,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )

        checks.append(
            _run_check(
                name="metadata-reference-closure",
                portable_command=(
                    "{python}",
                    "-m",
                    "axiom_encode.legacy_cleanup_validation",
                    "metadata-reference-closure-index-scan",
                    "--repo",
                    "{projected-repository}",
                    "--deleted-targets",
                    "{authorized-deletions}",
                ),
                actual_commands=(
                    _internal_command(
                        "metadata-reference-closure-index-scan",
                        "--repo",
                        str(projected),
                        "--deleted-targets",
                        str(deleted_targets),
                    ),
                ),
                targets=tracked,
                cwd=projected,
                environment=environment,
                runner=runner,
                output_limit=LEGACY_CLEANUP_MAX_COMMAND_OUTPUT_BYTES,
                budget=budget,
                state_check=state_check,
                output_replacements=replacements,
            )
        )
        state_check()

    evidence: dict[str, object] = {
        "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
        "status": "passed",
        "engine_execution": True,
        "projected_post_deletion_tree": plan.projected_post_deletion_tree,
        "checks": checks,
    }
    issues = validation_execution_issues(
        evidence,
        expected_projected_tree=plan.projected_post_deletion_tree,
    )
    if issues:
        raise LegacyCleanupValidationError(issues[0])
    return evidence


def validation_execution_issues(
    value: object,
    *,
    expected_projected_tree: str | None = None,
) -> list[str]:
    """Return exact-schema issues for persisted execution evidence."""

    issues: list[str] = []
    if not isinstance(value, dict) or set(value) != _EVIDENCE_FIELDS:
        return ["cleanup validation execution has a noncanonical field set"]
    if value.get("schema") != LEGACY_CLEANUP_VALIDATION_SCHEMA:
        issues.append("cleanup validation execution schema is invalid")
    if value.get("status") != "passed":
        issues.append("cleanup validation execution status is not passed")
    if value.get("engine_execution") is not True:
        issues.append("cleanup validation execution did not run the engine")
    tree = value.get("projected_post_deletion_tree")
    if not isinstance(tree, str) or _OID_RE.fullmatch(tree) is None:
        issues.append("cleanup validation projected tree is invalid")
    if expected_projected_tree is not None and tree != expected_projected_tree:
        issues.append("cleanup validation projected tree is stale")
    checks = value.get("checks")
    if not isinstance(checks, list) or len(checks) != len(
        LEGACY_CLEANUP_VALIDATION_CHECKS
    ):
        issues.append("cleanup validation check matrix is incomplete")
        return issues
    names: list[object] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != _CHECK_FIELDS:
            issues.append(f"cleanup validation check {index} has a malformed shape")
            continue
        names.append(check.get("name"))
        try:
            _validate_portable_command(check.get("command", []))
        except LegacyCleanupValidationError:
            issues.append(
                f"cleanup validation check {index} command is not portable"
            )
        target_count = check.get("target_count")
        if (
            type(target_count) is not int
            or target_count < 0
            or target_count > LEGACY_CLEANUP_MAX_VALIDATION_TARGETS
        ):
            issues.append(f"cleanup validation check {index} target count is invalid")
        if _SHA256_RE.fullmatch(str(check.get("target_list_sha256", ""))) is None:
            issues.append(
                f"cleanup validation check {index} target digest is invalid"
            )
        if check.get("exit_code") != 0 or type(check.get("exit_code")) is not int:
            issues.append(f"cleanup validation check {index} did not pass")
        if _SHA256_RE.fullmatch(str(check.get("output_sha256", ""))) is None:
            issues.append(
                f"cleanup validation check {index} output digest is invalid"
            )
    if tuple(names) != LEGACY_CLEANUP_VALIDATION_CHECKS:
        issues.append("cleanup validation checks are not complete and ordered")
    if isinstance(checks, list):
        by_name = {
            check.get("name"): check
            for check in checks
            if isinstance(check, dict)
        }
        for name in (
            "remaining-rulespec-validation",
            "remaining-companion-tests",
            "remaining-proof-validation",
            "money-atom-proof-validation",
            "oracle-coverage",
        ):
            check = by_name.get(name)
            if isinstance(check, dict) and check.get("target_count") == 0:
                issues.append(f"cleanup validation check {name} passed vacuously")
    return issues


def _repository_test_paths(repo: Path) -> tuple[Path, ...]:
    return tuple(
        entry.path
        for entry in _tracked_entries(repo)
        if len(entry.path.parts) >= 2
        and entry.path.parts[0] == "tests"
        and entry.path.suffix == ".py"
        and (
            entry.path.name.startswith("test_")
            or entry.path.name.endswith("_test.py")
        )
    )


def _repository_tests_presence_check(repo: Path) -> None:
    tests = _repository_test_paths(repo)
    if tests:
        raise LegacyCleanupValidationError(
            "repository test presence check cannot replace executing present tests"
        )
    print("No tracked Python repository tests are present.")


def _string_set(payload: object, key: str) -> set[str]:
    value = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise LegacyCleanupValidationError(f"repository layout {key} is malformed")
    normalized = {item.strip("/") for item in value}
    if "" in normalized:
        raise LegacyCleanupValidationError(f"repository layout {key} is malformed")
    return normalized


def _configured_layout_check(repo: Path, tracked: Sequence[Path]) -> None:
    config_path = Path(".axiom/repository-structure.yaml")
    try:
        config = yaml.safe_load(
            _safe_regular_bytes(
                repo,
                config_path,
                label="repository structure configuration",
            )
        ) or {}
    except yaml.YAMLError as exc:
        raise LegacyCleanupValidationError(
            "repository structure configuration is invalid YAML"
        ) from exc
    if not isinstance(config, dict) or config.get("version") != 1:
        raise LegacyCleanupValidationError(
            "repository structure configuration must use version 1"
        )
    allowed_directories = _string_set(config, "allowed_root_directories")
    allowed_files = _string_set(config, "allowed_root_files")
    path_rules = config.get("path_rules")
    if not isinstance(path_rules, list) or not path_rules:
        raise LegacyCleanupValidationError("repository layout path_rules is malformed")
    normalized_rules: list[tuple[tuple[str, ...], set[str], set[str]]] = []
    for rule in path_rules:
        if not isinstance(rule, dict):
            raise LegacyCleanupValidationError(
                "repository layout path rule is malformed"
            )
        patterns = rule.get("patterns")
        extensions = rule.get("allow_extensions", [])
        filenames = rule.get("allow_filenames", [])
        if (
            not isinstance(patterns, list)
            or not patterns
            or not all(isinstance(item, str) and item for item in patterns)
            or not isinstance(extensions, list)
            or not all(isinstance(item, str) for item in extensions)
            or not isinstance(filenames, list)
            or not all(isinstance(item, str) for item in filenames)
        ):
            raise LegacyCleanupValidationError(
                "repository layout path rule is malformed"
            )
        normalized_rules.append(
            (
                tuple(pattern.strip("/") for pattern in patterns),
                set(extensions),
                set(filenames),
            )
        )
    for path in tracked:
        text = path.as_posix()
        if len(path.parts) == 1:
            if text not in allowed_files:
                raise LegacyCleanupValidationError(
                    f"repository layout rejects top-level file: {text}"
                )
            continue
        if path.parts[0] not in allowed_directories:
            raise LegacyCleanupValidationError(
                f"repository layout rejects top-level directory: {path.parts[0]}"
            )
        matched = next(
            (
                (extensions, filenames)
                for patterns, extensions, filenames in normalized_rules
                if any(fnmatch.fnmatchcase(text, pattern) for pattern in patterns)
            ),
            None,
        )
        if matched is None:
            raise LegacyCleanupValidationError(
                f"repository layout has no rule for: {text}"
            )
        extensions, filenames = matched
        pure = PurePosixPath(text)
        if pure.name not in filenames and pure.suffix not in extensions:
            raise LegacyCleanupValidationError(
                f"repository layout rejects file name or extension: {text}"
            )


def _repository_layout_check(repo: Path) -> None:
    entries = _tracked_entries(repo)
    tracked = tuple(entry.path for entry in entries)
    if Path(".axiom/repository-structure.yaml") in set(tracked):
        _configured_layout_check(repo, tracked)
    else:
        for path in tracked:
            if path.parts[0] in {"statute", "regulation", "policy"}:
                raise LegacyCleanupValidationError(
                    f"repository layout contains obsolete root: {path.parts[0]}"
                )
            if path.name in {"parameters.yaml", "tests.yaml"}:
                raise LegacyCleanupValidationError(
                    f"repository layout contains an obsolete file: {path}"
                )
            if (
                len(path.parts) >= 2
                and path.parts[0] == "tests"
                and path.suffix in {".yaml", ".yml"}
            ):
                raise LegacyCleanupValidationError(
                    f"repository layout contains YAML under tests/: {path}"
                )
            if (
                path.suffix in {".yaml", ".yml"}
                and len(path.parts) >= 2
                and _JURISDICTION_RE.fullmatch(path.parts[0]) is not None
                and path.parts[1] not in RULESPEC_ATOMIC_MODULE_ROOTS
                and path.parts[1] != "programs"
            ):
                raise LegacyCleanupValidationError(
                    f"repository layout contains RuleSpec YAML outside a known root: {path}"
                )
    for entry in entries:
        if (_is_primary(entry.path) or _is_companion(entry.path)) and entry.mode != (
            "100644"
        ):
            raise LegacyCleanupValidationError(
                f"repository layout RuleSpec target is not 100644: {entry.path}"
            )
    print(f"Repository layout passed for {len(tracked)} tracked path(s).")


def _read_deleted_targets(path: Path) -> tuple[Path, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LegacyCleanupValidationError("deleted-target input is unreadable") from exc
    if len(raw) > 128 * (LEGACY_CLEANUP_MAX_PATH_CHARS + 1):
        raise LegacyCleanupValidationError("deleted-target input exceeds its limit")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LegacyCleanupValidationError("deleted-target input is not UTF-8") from exc
    targets: list[Path] = []
    for text in lines:
        path_value = Path(text)
        if (
            not text
            or path_value.is_absolute()
            or path_value.as_posix() != text
            or any(part in {"", ".", ".."} for part in path_value.parts)
            or len(text) > LEGACY_CLEANUP_MAX_PATH_CHARS
        ):
            raise LegacyCleanupValidationError("deleted-target input is malformed")
        targets.append(path_value)
    if not targets or len(targets) > 128 or targets != sorted(set(targets)):
        raise LegacyCleanupValidationError(
            "deleted-target input must be nonempty, unique, and sorted"
        )
    return tuple(targets)


def _read_validation_target_list(path: Path) -> tuple[Path, ...]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LegacyCleanupValidationError(
            "validation target-list input is unreadable"
        ) from exc
    if len(raw) > LEGACY_CLEANUP_MAX_TARGET_LIST_BYTES:
        raise LegacyCleanupValidationError(
            "validation target-list input exceeds its byte limit"
        )
    if not raw or not raw.endswith(b"\n") or b"\0" in raw:
        raise LegacyCleanupValidationError("validation target-list input is malformed")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise LegacyCleanupValidationError(
            "validation target-list input is not UTF-8"
        ) from exc
    targets = tuple(Path(line) for line in lines)
    _target_list_sha256(targets)
    if not targets:
        raise LegacyCleanupValidationError("validation target-list input is empty")
    if raw != "".join(
        f"{target.as_posix()}\n" for target in targets
    ).encode("utf-8"):
        raise LegacyCleanupValidationError(
            "validation target-list input is not canonical LF-delimited UTF-8"
        )
    return targets


def _reference_variants(paths: Sequence[Path]) -> tuple[bytes, ...]:
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
    return tuple(
        sorted(
            (value.encode("utf-8") for value in variants),
            key=lambda value: (-len(value), value),
        )
    )


_REFERENCE_LEFT_TOKEN_BYTES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
)
_REFERENCE_RIGHT_TOKEN_BYTES = frozenset(
    b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:/-"
)


def _contains_deleted_reference(raw: bytes, patterns: Sequence[bytes]) -> bool:
    """Match complete path/reference tokens, not innocent longer identifiers."""

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


def _provision_index_check(
    payload: object,
    *,
    tracked: set[Path],
) -> None:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "axiom.rulespec.provisions_to_rules/v1"
        or not isinstance(payload.get("provisions"), dict)
    ):
        raise LegacyCleanupValidationError("provision index has an invalid schema")
    provisions = payload["provisions"]
    for citation, records in provisions.items():
        if not isinstance(citation, str) or not citation or not isinstance(records, list):
            raise LegacyCleanupValidationError("provision index records are malformed")
        for record in records:
            if (
                not isinstance(record, dict)
                or set(record) != {"module", "via"}
                or not isinstance(record.get("module"), str)
                or not isinstance(record.get("via"), list)
                or not record["via"]
                or not all(isinstance(value, str) and value for value in record["via"])
                or record["via"] != sorted(set(record["via"]))
            ):
                raise LegacyCleanupValidationError(
                    "provision index records are malformed"
                )
            module = Path(record["module"])
            if module not in tracked or not _is_primary(module):
                raise LegacyCleanupValidationError(
                    f"provision index references a missing module: {module}"
                )


def _metadata_reference_closure_check(repo: Path, deleted_file: Path) -> None:
    deleted = _read_deleted_targets(deleted_file)
    entries = _tracked_entries(repo)
    tracked = {entry.path for entry in entries}
    if tracked.intersection(deleted):
        raise LegacyCleanupValidationError("a deleted target remains tracked")
    for path in deleted:
        _assert_absent(repo, path)
    patterns = _reference_variants(deleted)
    total = 0
    for entry in entries:
        if entry.mode not in {"100644", "100755"}:
            raise LegacyCleanupValidationError(
                "metadata/reference scan cannot inspect a non-regular tracked path: "
                f"{entry.path}"
            )
        raw = _safe_regular_bytes(
            repo,
            entry.path,
            label=f"metadata/reference scan path {entry.path.as_posix()}",
        )
        total += len(raw)
        if total > LEGACY_CLEANUP_MAX_TRACKED_TOTAL_BYTES:
            raise LegacyCleanupValidationError(
                "metadata/reference scan exceeds its total byte limit"
            )
        if _contains_deleted_reference(raw, patterns):
            raise LegacyCleanupValidationError(
                "surviving metadata or source references a deleted target: "
                f"{entry.path.as_posix()}"
            )
        if entry.path.parts[:2] == (".axiom", "index"):
            if entry.path.suffix != ".json":
                raise LegacyCleanupValidationError(
                    f"repository index contains a non-JSON record: {entry.path}"
                )
            payload = decode_strict_json_object(
                raw,
                label=f"repository index {entry.path.as_posix()}",
                max_bytes=LEGACY_CLEANUP_MAX_TRACKED_FILE_BYTES,
            )
            if entry.path == Path(".axiom/index/provisions_to_rules.json"):
                _provision_index_check(payload, tracked=tracked)
    print(
        "Metadata/reference closure passed for "
        f"{len(entries)} tracked path(s) and {len(deleted)} deletion(s)."
    )


def _run_verification_only_axiom_encode(
    public_keys: Sequence[str],
    arguments: Sequence[str],
    *,
    target_lists: Sequence[Path] = (),
    target_prefix: Path | None = None,
    target_strip_components: int = 0,
) -> int:
    child_arguments = list(arguments)
    if child_arguments[:1] == ["--"]:
        child_arguments.pop(0)
    if not child_arguments:
        raise LegacyCleanupValidationError(
            "verification-only axiom-encode wrapper requires a subcommand"
        )
    if target_strip_components < 0 or target_strip_components > 8:
        raise LegacyCleanupValidationError(
            "verification-only target strip count is invalid"
        )
    expanded_targets: list[str] = []
    for target_list in target_lists:
        for target in _read_validation_target_list(target_list):
            if len(target.parts) <= target_strip_components:
                raise LegacyCleanupValidationError(
                    "verification-only target strip count removes the full path"
                )
            transformed = Path(*target.parts[target_strip_components:])
            if target_prefix is not None:
                transformed = target_prefix / transformed
            expanded_targets.append(str(transformed))
    if len(expanded_targets) > LEGACY_CLEANUP_MAX_VALIDATION_TARGETS:
        raise LegacyCleanupValidationError(
            "verification-only target expansion exceeds its limit"
        )
    child_arguments.extend(expanded_targets)
    previous_arguments = sys.argv[:]
    try:
        with local_corpus_release_verification(tuple(public_keys)):
            sys.argv[:] = ["axiom-encode", *child_arguments]
            from .entrypoint import main as axiom_encode_main

            try:
                result = axiom_encode_main()
            except SystemExit as exc:
                if exc.code is None:
                    return 0
                if type(exc.code) is int:
                    return exc.code
                print(str(exc.code), file=sys.stderr)
                return 1
            return int(result or 0)
    finally:
        sys.argv[:] = previous_arguments


def _internal_main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("repository-tests-presence", "repository-layout"):
        child = subparsers.add_parser(command)
        child.add_argument("--repo", type=Path, required=True)
    closure = subparsers.add_parser("metadata-reference-closure-index-scan")
    closure.add_argument("--repo", type=Path, required=True)
    closure.add_argument("--deleted-targets", type=Path, required=True)
    verification = subparsers.add_parser("axiom-encode-verification")
    verification.add_argument(
        "--corpus-release-public-key",
        action="append",
        required=True,
    )
    verification.add_argument(
        "--target-list",
        action="append",
        type=Path,
        default=[],
    )
    verification.add_argument("--target-prefix", type=Path)
    verification.add_argument(
        "--target-strip-components",
        type=int,
        default=0,
    )
    verification.add_argument("axiom_encode_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    try:
        if args.command == "axiom-encode-verification":
            return _run_verification_only_axiom_encode(
                args.corpus_release_public_key,
                args.axiom_encode_arguments,
                target_lists=args.target_list,
                target_prefix=args.target_prefix,
                target_strip_components=args.target_strip_components,
            )
        repo = _canonical_directory(args.repo, label="projected RuleSpec checkout")
        if args.command == "repository-tests-presence":
            _repository_tests_presence_check(repo)
        elif args.command == "repository-layout":
            _repository_layout_check(repo)
        else:
            _metadata_reference_closure_check(repo, args.deleted_targets)
    except (
        LegacyCleanupReceiptError,
        LegacyCleanupValidationError,
        RuleSpecToolchainError,
    ) as exc:
        print(f"legacy cleanup validation: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(_internal_main())
