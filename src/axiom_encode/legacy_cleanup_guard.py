"""Fail-closed guards for atomic legacy RuleSpec cleanup transitions.

The receipt is negative provenance, not an applied encoding manifest.  This
module therefore inspects its append-only receipt root before any caller can
consider ordinary RuleSpec ownership, and authorizes only the exact
``B -> H`` (or ``B -> worktree``) contraction recorded by one new receipt.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .legacy_cleanup import (
    LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
    LEGACY_CLEANUP_RECEIPT_DIR,
    LegacyCleanupReceiptError,
    canonical_receipt_bytes,
    deleted_paths,
    is_legacy_cleanup_receipt_path,
    parse_receipt_bytes,
)
from .legacy_cleanup_git import (
    LegacyCleanupGitError,
    plan_legacy_cleanup_base,
    plan_payload_issues,
)
from .legacy_cleanup_signing import legacy_cleanup_signature_issue
from .signing_broker import SigningBroker

_MAX_RECEIPTS = 100_000
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


@dataclass(frozen=True, slots=True)
class LegacyCleanupGuardResult:
    """Deterministic authorization result shared by guards and staging."""

    issues: tuple[str, ...]
    authorized_paths: tuple[Path, ...] = ()
    receipt_path: Path | None = None
    receipt: Mapping[str, object] | None = None

    @property
    def authorized(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class _ReceiptRecord:
    path: Path
    mode: str
    object_type: str
    oid: str | None
    raw: bytes | None
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _DiffChange:
    status: str
    paths: tuple[Path, ...]


class _InspectionError(ValueError):
    pass


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


def _git_bytes(repo: Path, *arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", *_GIT_CONFIG_ARGUMENTS, "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        env=_git_environment(),
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise _InspectionError(
            f"Git inspection failed ({' '.join(arguments)}): "
            f"{detail or 'git command failed'}"
        )
    return completed.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    try:
        return _git_bytes(repo, *arguments).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InspectionError("Git inspection output is not UTF-8") from exc


def _oid_length(object_format: str) -> int:
    if object_format == "sha1":
        return 40
    if object_format == "sha256":
        return 64
    raise _InspectionError(f"unsupported Git object format: {object_format}")


def _resolve_exact_commit(repo: Path, raw: str, object_format: str, label: str) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) != _oid_length(object_format)
        or re.fullmatch(r"[0-9a-f]+", raw) is None
    ):
        raise _InspectionError(f"{label} must be a full lowercase Git commit OID")
    resolved = _git_text(repo, "rev-parse", "--verify", f"{raw}^{{commit}}").strip()
    if resolved != raw:
        raise _InspectionError(f"{label} did not resolve exactly")
    return resolved


def _canonical_git_path(raw: bytes, *, label: str) -> Path:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InspectionError(f"{label} is not UTF-8") from exc
    path = Path(text)
    if (
        not text
        or len(text) > 4_096
        or "\\" in text
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise _InspectionError(f"{label} is not a canonical repository path")
    return path


def _append_issue(issues: list[str], issue: str) -> None:
    if issue not in issues:
        issues.append(issue)


def _receipt_tree_records(
    repo: Path,
    commit: str,
    *,
    verifier: SigningBroker | Ed25519PublicKey,
    label: str,
) -> tuple[dict[Path, _ReceiptRecord], list[str]]:
    issues: list[str] = []
    records: dict[Path, _ReceiptRecord] = {}
    raw = _git_bytes(
        repo,
        "ls-tree",
        "-r",
        "-l",
        "-z",
        "--full-tree",
        commit,
        "--",
        LEGACY_CLEANUP_RECEIPT_DIR.as_posix(),
    )
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        try:
            metadata, raw_path = encoded.split(b"\t", 1)
            mode, object_type, oid, raw_size = metadata.decode("ascii").split()
            path = _canonical_git_path(raw_path, label=f"{label} receipt-root path")
        except (ValueError, UnicodeDecodeError, _InspectionError) as exc:
            _append_issue(
                issues, f"{label} receipt-root tree entry is malformed: {exc}"
            )
            continue
        if path in records:
            _append_issue(
                issues,
                f"{label} receipt-root contains duplicate path {path.as_posix()}",
            )
            continue
        if len(records) >= _MAX_RECEIPTS:
            _append_issue(issues, f"{label} receipt-root exceeds its entry limit")
            break
        canonical_path = (
            path.parent == LEGACY_CLEANUP_RECEIPT_DIR
            and is_legacy_cleanup_receipt_path(path)
        )
        if not canonical_path:
            _append_issue(
                issues,
                f"{label} receipt-root entry is not a direct 64hex JSON receipt: "
                f"{path.as_posix()}",
            )
        if mode != "100644" or object_type != "blob":
            _append_issue(
                issues,
                f"{label} receipt-root entry is not a regular 100644 blob: "
                f"{path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, mode, object_type, oid, None, None)
            continue
        try:
            size = int(raw_size)
        except ValueError:
            size = -1
        if size < 0 or size > LEGACY_CLEANUP_MAX_RECEIPT_BYTES:
            _append_issue(
                issues,
                f"{label} receipt exceeds its byte limit: {path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, mode, object_type, oid, None, None)
            continue
        content = _git_bytes(repo, "cat-file", "blob", oid)
        if len(content) != size:
            _append_issue(
                issues,
                f"{label} receipt size differs from its tree entry: {path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, mode, object_type, oid, None, None)
            continue
        payload: dict[str, Any] | None = None
        if canonical_path:
            try:
                payload = parse_receipt_bytes(content, expected_path=path)
            except LegacyCleanupReceiptError as exc:
                _append_issue(
                    issues,
                    f"{label} receipt is invalid at {path.as_posix()}: {exc}",
                )
            if payload is not None:
                signature_issue = legacy_cleanup_signature_issue(payload, verifier)
                if signature_issue is not None:
                    _append_issue(
                        issues,
                        f"{label} receipt signature is invalid at "
                        f"{path.as_posix()}: {signature_issue}",
                    )
        records[path] = _ReceiptRecord(path, mode, object_type, oid, content, payload)
    return records, issues


def _live_receipt_records(
    repo: Path,
    *,
    verifier: SigningBroker | Ed25519PublicKey,
) -> tuple[dict[Path, _ReceiptRecord], list[str]]:
    issues: list[str] = []
    records: dict[Path, _ReceiptRecord] = {}
    root = repo / LEGACY_CLEANUP_RECEIPT_DIR
    try:
        root_metadata = root.lstat()
    except FileNotFoundError:
        return records, issues
    except OSError as exc:
        return records, [f"worktree receipt-root is unreadable: {exc}"]
    if not stat.S_ISDIR(root_metadata.st_mode) or stat.S_ISLNK(root_metadata.st_mode):
        return records, ["worktree receipt-root is not a real directory"]
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        return records, [f"worktree receipt-root is unreadable: {exc}"]
    if len(children) > _MAX_RECEIPTS:
        return records, ["worktree receipt-root exceeds its entry limit"]
    for target in children:
        path = LEGACY_CLEANUP_RECEIPT_DIR / target.name
        canonical_path = is_legacy_cleanup_receipt_path(path)
        try:
            before = target.lstat()
        except OSError as exc:
            _append_issue(
                issues,
                f"worktree receipt-root entry is unreadable: {path.as_posix()}: {exc}",
            )
            continue
        if not canonical_path:
            _append_issue(
                issues,
                "worktree receipt-root entry is not a direct 64hex JSON receipt: "
                f"{path.as_posix()}",
            )
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o644
        ):
            _append_issue(
                issues,
                "worktree receipt-root entry is not a regular 0644 file: "
                f"{path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, "", "", None, None, None)
            continue
        if before.st_size > LEGACY_CLEANUP_MAX_RECEIPT_BYTES:
            _append_issue(
                issues,
                f"worktree receipt exceeds its byte limit: {path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, "100644", "blob", None, None, None)
            continue
        try:
            content = target.read_bytes()
            after = target.lstat()
        except OSError as exc:
            _append_issue(
                issues,
                f"worktree receipt is unreadable: {path.as_posix()}: {exc}",
            )
            records[path] = _ReceiptRecord(path, "100644", "blob", None, None, None)
            continue
        before_fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_fingerprint != after_fingerprint or len(content) != before.st_size:
            _append_issue(
                issues,
                f"worktree receipt changed while read: {path.as_posix()}",
            )
            records[path] = _ReceiptRecord(path, "100644", "blob", None, None, None)
            continue
        payload: dict[str, Any] | None = None
        if canonical_path:
            try:
                payload = parse_receipt_bytes(content, expected_path=path)
            except LegacyCleanupReceiptError as exc:
                _append_issue(
                    issues,
                    f"worktree receipt is invalid at {path.as_posix()}: {exc}",
                )
            if payload is not None:
                signature_issue = legacy_cleanup_signature_issue(payload, verifier)
                if signature_issue is not None:
                    _append_issue(
                        issues,
                        "worktree receipt signature is invalid at "
                        f"{path.as_posix()}: {signature_issue}",
                    )
        records[path] = _ReceiptRecord(path, "100644", "blob", None, content, payload)
    return records, issues


def _committed_changes(repo: Path, base: str, head: str) -> list[_DiffChange]:
    raw = _git_bytes(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "--find-copies",
        "--no-ext-diff",
        base,
        head,
        "--",
    )
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    changes: list[_DiffChange] = []
    index = 0
    while index < len(parts):
        try:
            status = parts[index].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _InspectionError("Git diff status is malformed") from exc
        index += 1
        path_count = 2 if status.startswith(("R", "C")) else 1
        if index + path_count > len(parts):
            raise _InspectionError("Git diff name-status output is truncated")
        paths = tuple(
            _canonical_git_path(parts[index + offset], label="Git diff path")
            for offset in range(path_count)
        )
        index += path_count
        changes.append(_DiffChange(status, paths))
    return changes


def _worktree_status(repo: Path) -> list[_DiffChange]:
    raw = _git_bytes(
        repo,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignored",
    )
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()
    changes: list[_DiffChange] = []
    index = 0
    while index < len(parts):
        entry = parts[index]
        index += 1
        if len(entry) < 4 or entry[2:3] != b" ":
            raise _InspectionError("Git worktree status output is malformed")
        try:
            status = entry[:2].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _InspectionError("Git worktree status is malformed") from exc
        paths = [_canonical_git_path(entry[3:], label="Git worktree status path")]
        if "R" in status or "C" in status:
            if index >= len(parts):
                raise _InspectionError("Git worktree rename status is truncated")
            paths.append(
                _canonical_git_path(parts[index], label="Git worktree source path")
            )
            index += 1
        changes.append(_DiffChange(status, tuple(paths)))
    return changes


def _receipt_root_history_issues(
    base_records: Mapping[Path, _ReceiptRecord],
    next_records: Mapping[Path, _ReceiptRecord],
) -> tuple[list[Path], list[str]]:
    issues: list[str] = []
    for path in sorted(base_records, key=Path.as_posix):
        before = base_records[path]
        after = next_records.get(path)
        if after is None:
            _append_issue(
                issues,
                f"historical cleanup receipt was deleted or renamed: {path.as_posix()}",
            )
        elif (
            before.mode != after.mode
            or before.object_type != after.object_type
            or before.raw != after.raw
        ):
            _append_issue(
                issues,
                f"historical cleanup receipt was modified: {path.as_posix()}",
            )
    additions = sorted(set(next_records) - set(base_records), key=Path.as_posix)
    if len(additions) != 1:
        _append_issue(
            issues,
            "legacy cleanup transition must add exactly one new cleanup receipt",
        )
    return additions, issues


def _payload_deleted_paths(payload: Mapping[str, object]) -> tuple[Path, ...]:
    return tuple(Path(path) for path in deleted_paths(payload))


def _historical_overlap_issues(
    base_records: Mapping[Path, _ReceiptRecord],
    new_payload: Mapping[str, object] | None,
) -> list[str]:
    issues: list[str] = []
    owner: dict[Path, Path] = {}
    for receipt_path in sorted(base_records, key=Path.as_posix):
        payload = base_records[receipt_path].payload
        if payload is None:
            continue
        for target in _payload_deleted_paths(payload):
            prior = owner.get(target)
            if prior is not None:
                _append_issue(
                    issues,
                    "historical cleanup receipts overlap at "
                    f"{target.as_posix()}: {prior.as_posix()} and "
                    f"{receipt_path.as_posix()}",
                )
            else:
                owner[target] = receipt_path
    if new_payload is not None:
        for target in _payload_deleted_paths(new_payload):
            prior = owner.get(target)
            if prior is not None:
                _append_issue(
                    issues,
                    "new cleanup receipt overlaps historical receipt coverage at "
                    f"{target.as_posix()}: {prior.as_posix()}",
                )
    return issues


def _immutable_plan_issues(
    repo: Path,
    *,
    base: str,
    payload: Mapping[str, object],
    expected_toolchain: Mapping[str, object] | None,
) -> list[str]:
    issues: list[str] = []
    groups = payload.get("groups")
    primary_paths: list[str] = []
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            primary = group.get("primary")
            if isinstance(primary, dict) and isinstance(primary.get("path"), str):
                primary_paths.append(primary["path"])
    try:
        plan = plan_legacy_cleanup_base(
            repo,
            base_ref=base,
            primary_paths=primary_paths,
            require_clean_checkout=False,
        )
    except (LegacyCleanupGitError, LegacyCleanupReceiptError, OSError) as exc:
        return [f"cannot reconstruct cleanup proof from immutable base: {exc}"]
    issues.extend(plan_payload_issues(payload, plan))
    toolchain = payload.get("toolchain")
    if isinstance(toolchain, dict):
        corpus = toolchain.get("corpus_release")
        if not isinstance(corpus, dict) or (
            corpus.get("name") != plan.toolchain_values["axiom_corpus_release"]
            or corpus.get("content_sha256")
            != plan.toolchain_values["axiom_corpus_release_content_sha256"]
        ):
            issues.append("legacy cleanup corpus release disagrees with immutable base")
        if (
            toolchain.get("validation_waiver_set_sha256")
            != plan.toolchain_values["validation_waiver_set_sha256"]
        ):
            issues.append("legacy cleanup waiver pin disagrees with immutable base")
    if expected_toolchain is not None and toolchain != dict(expected_toolchain):
        issues.append(
            "legacy cleanup introduction toolchain does not match current pins"
        )
    return issues


def _exact_committed_change_issues(
    changes: Sequence[_DiffChange],
    receipt_path: Path,
    targets: Sequence[Path],
) -> list[str]:
    issues: list[str] = []
    expected = {receipt_path: "A", **{path: "D" for path in targets}}
    actual: dict[Path, str] = {}
    for change in changes:
        if change.status.startswith(("R", "C")):
            issues.append(
                "legacy cleanup transition contains a rename or copy: "
                + " -> ".join(path.as_posix() for path in change.paths)
            )
            continue
        if len(change.paths) != 1 or change.status not in {"A", "D"}:
            rendered = ", ".join(path.as_posix() for path in change.paths)
            issues.append(
                f"legacy cleanup transition has unauthorized status "
                f"{change.status} for {rendered}"
            )
            for path in change.paths:
                actual[path] = change.status
            continue
        path = change.paths[0]
        if path in actual:
            issues.append(
                f"legacy cleanup transition reports duplicate change {path.as_posix()}"
            )
        actual[path] = change.status
    if actual != expected:
        missing = sorted(
            set(expected.items()) - set(actual.items()),
            key=lambda item: item[0].as_posix(),
        )
        extra = sorted(
            set(actual.items()) - set(expected.items()),
            key=lambda item: item[0].as_posix(),
        )
        if missing:
            issues.append(
                "legacy cleanup transition is missing exact changes: "
                + ", ".join(f"{status} {path.as_posix()}" for path, status in missing)
            )
        if extra:
            issues.append(
                "legacy cleanup transition contains extra or mixed changes: "
                + ", ".join(f"{status} {path.as_posix()}" for path, status in extra)
            )
    return issues


def _exact_worktree_change_issues(
    changes: Sequence[_DiffChange],
    receipt_path: Path,
    targets: Sequence[Path],
) -> list[str]:
    issues: list[str] = []
    expected_paths = {receipt_path, *targets}
    actual_paths: set[Path] = set()
    for change in changes:
        if change.status == "!!":
            issues.append(
                "legacy cleanup worktree contains ignored path: "
                + ", ".join(path.as_posix() for path in change.paths)
            )
            continue
        if change.status in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            issues.append(
                "legacy cleanup worktree contains an unresolved index conflict: "
                + ", ".join(path.as_posix() for path in change.paths)
            )
        if "R" in change.status or "C" in change.status:
            issues.append(
                "legacy cleanup worktree contains a rename or copy: "
                + " -> ".join(path.as_posix() for path in change.paths)
            )
        actual_paths.update(change.paths)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths, key=Path.as_posix)
        extra = sorted(actual_paths - expected_paths, key=Path.as_posix)
        if missing:
            issues.append(
                "legacy cleanup worktree status is missing exact paths: "
                + ", ".join(path.as_posix() for path in missing)
            )
        if extra:
            issues.append(
                "legacy cleanup worktree contains extra or mixed paths: "
                + ", ".join(path.as_posix() for path in extra)
            )
    return issues


def _targets_absent_issues(
    repo: Path, targets: Sequence[Path], *, label: str
) -> list[str]:
    issues: list[str] = []
    for path in targets:
        try:
            (repo / path).lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            issues.append(
                f"{label} cleanup target is unreadable: {path.as_posix()}: {exc}"
            )
        else:
            issues.append(f"{label} cleanup target is not absent: {path.as_posix()}")
    return issues


def _commit_targets_absent_issues(
    repo: Path, head: str, targets: Sequence[Path]
) -> list[str]:
    issues: list[str] = []
    for path in targets:
        entry = _git_bytes(
            repo,
            "ls-tree",
            "-z",
            head,
            "--",
            path.as_posix(),
        )
        if entry:
            issues.append(f"committed cleanup target is not absent: {path.as_posix()}")
    return issues


def _final_result(
    issues: Sequence[str],
    *,
    receipt_path: Path | None,
    payload: Mapping[str, object] | None,
) -> LegacyCleanupGuardResult:
    unique: list[str] = []
    for issue in issues:
        _append_issue(unique, issue)
    if unique or receipt_path is None or payload is None:
        return LegacyCleanupGuardResult(tuple(unique))
    targets = _payload_deleted_paths(payload)
    return LegacyCleanupGuardResult(
        (),
        (receipt_path, *targets),
        receipt_path,
        payload,
    )


def verify_committed_legacy_cleanup_transition(
    repo: Path,
    *,
    base_ref: str,
    head_ref: str,
    verifier: SigningBroker | Ed25519PublicKey,
    expected_toolchain: Mapping[str, object] | None = None,
) -> LegacyCleanupGuardResult:
    """Verify exactly one signed atomic cleanup over the committed ``B..H`` diff."""

    issues: list[str] = []
    receipt_path: Path | None = None
    payload: Mapping[str, object] | None = None
    try:
        checkout = Path(repo).resolve(strict=True)
        object_format = _git_text(checkout, "rev-parse", "--show-object-format").strip()
        _oid_length(object_format)
        base = _resolve_exact_commit(checkout, base_ref, object_format, "cleanup base")
        head = _resolve_exact_commit(checkout, head_ref, object_format, "cleanup head")
        ancestor = subprocess.run(
            [
                "git",
                *_GIT_CONFIG_ARGUMENTS,
                "-C",
                str(checkout),
                "merge-base",
                "--is-ancestor",
                base,
                head,
            ],
            capture_output=True,
            check=False,
            env=_git_environment(),
        )
        if ancestor.returncode != 0:
            issues.append("cleanup base is not an ancestor of cleanup head")
        parents = _git_text(checkout, "rev-list", "--parents", "-n", "1", head).split()
        if parents != [head, base]:
            issues.append(
                "legacy cleanup must be one atomic commit whose only parent is the "
                "protected base"
            )

        base_records, base_issues = _receipt_tree_records(
            checkout, base, verifier=verifier, label="base"
        )
        head_records, head_issues = _receipt_tree_records(
            checkout, head, verifier=verifier, label="head"
        )
        issues.extend(base_issues)
        issues.extend(head_issues)
        additions, history_issues = _receipt_root_history_issues(
            base_records, head_records
        )
        issues.extend(history_issues)
        if len(additions) == 1:
            receipt_path = additions[0]
            payload = head_records[receipt_path].payload
            if payload is None:
                issues.append("new cleanup receipt is not valid and verifiable")
        issues.extend(_historical_overlap_issues(base_records, payload))

        changes = _committed_changes(checkout, base, head)
        if receipt_path is not None and payload is not None:
            targets = _payload_deleted_paths(payload)
            issues.extend(
                _exact_committed_change_issues(changes, receipt_path, targets)
            )
            issues.extend(_commit_targets_absent_issues(checkout, head, targets))
            issues.extend(
                _immutable_plan_issues(
                    checkout,
                    base=base,
                    payload=payload,
                    expected_toolchain=expected_toolchain,
                )
            )
        elif changes:
            issues.append(
                "committed changes have no single valid new cleanup receipt authorization"
            )
    except (
        _InspectionError,
        LegacyCleanupGitError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        issues.append(f"cannot inspect committed legacy cleanup transition: {exc}")
    return _final_result(issues, receipt_path=receipt_path, payload=payload)


def verify_worktree_legacy_cleanup_transition(
    repo: Path,
    *,
    base_ref: str,
    verifier: SigningBroker | Ed25519PublicKey,
    expected_toolchain: Mapping[str, object] | None = None,
) -> LegacyCleanupGuardResult:
    """Verify exact live cleanup bytes before staging them as one transaction."""

    issues: list[str] = []
    receipt_path: Path | None = None
    payload: Mapping[str, object] | None = None
    try:
        checkout = Path(repo).resolve(strict=True)
        object_format = _git_text(checkout, "rev-parse", "--show-object-format").strip()
        _oid_length(object_format)
        base = _resolve_exact_commit(checkout, base_ref, object_format, "cleanup base")
        head = _git_text(checkout, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if head != base:
            issues.append("cleanup worktree HEAD must equal the exact protected base")

        base_records, base_issues = _receipt_tree_records(
            checkout, base, verifier=verifier, label="base"
        )
        live_records, live_issues = _live_receipt_records(checkout, verifier=verifier)
        issues.extend(base_issues)
        issues.extend(live_issues)
        additions, history_issues = _receipt_root_history_issues(
            base_records, live_records
        )
        issues.extend(history_issues)
        if len(additions) == 1:
            receipt_path = additions[0]
            payload = live_records[receipt_path].payload
            if payload is None:
                issues.append(
                    "new worktree cleanup receipt is not valid and verifiable"
                )
        issues.extend(_historical_overlap_issues(base_records, payload))

        changes = _worktree_status(checkout)
        if receipt_path is not None and payload is not None:
            targets = _payload_deleted_paths(payload)
            issues.extend(_exact_worktree_change_issues(changes, receipt_path, targets))
            issues.extend(_targets_absent_issues(checkout, targets, label="worktree"))
            if live_records[receipt_path].raw != canonical_receipt_bytes(payload):
                issues.append(
                    "new worktree cleanup receipt bytes are not exact canonical JSON"
                )
            issues.extend(
                _immutable_plan_issues(
                    checkout,
                    base=base,
                    payload=payload,
                    expected_toolchain=expected_toolchain,
                )
            )
        elif changes:
            issues.append(
                "worktree changes have no single valid new cleanup receipt authorization"
            )
    except (
        _InspectionError,
        LegacyCleanupGitError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        issues.append(f"cannot inspect worktree legacy cleanup transition: {exc}")
    return _final_result(issues, receipt_path=receipt_path, payload=payload)
