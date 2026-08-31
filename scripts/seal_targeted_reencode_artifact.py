#!/usr/bin/env python3
"""Copy, seal, and verify one flat targeted re-encode audit artifact.

The workflow invokes this file from the root-owned, commit-pinned encoder clone.
It deliberately accepts only a small flat set of regular files so a mutable
runner-owned staging directory cannot smuggle links, devices, or an unbounded
payload across the privilege boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path
from typing import Any

MANIFEST_NAME = "ARTIFACT-MANIFEST.json"
MANIFEST_DIGEST_NAME = "ARTIFACT-MANIFEST.sha256"
MANIFEST_SCHEMA = "axiom-encode/sealed-targeted-reencode-artifact/v1"
MAX_FILE_COUNT = 256
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024
FIXED_CANDIDATE_NAMES = frozenset(
    {
        "apply-manifests.json",
        "canonical-refresh-bundle.json",
        "context-manifest.json",
        "dependent-2-context-manifest.json",
        "dependent-context-manifest.json",
        "existing-signed-imports.json",
        "guard-generated.json",
        "legacy-replacement-receipt.json",
        "metadata.json",
        "primary-required-test-cases.json",
        "rulespec-generated-head.txt",
        "rulespec-tree.txt",
        "signed-import-inventory.json",
        "source-bundle.json",
        "status.txt",
        "target-guard-generated.json",
        "target-operation.txt",
        "target-preflight-guard-generated.json",
        "worktree-copy.json",
    }
)
VARIABLE_CANDIDATE_NAME_PATTERNS = (
    re.compile(r"canonical-refresh-[0-9]{2}-context-manifest\.json"),
    re.compile(r"canonical-refresh-[0-9]{2}-guard-generated\.json"),
    re.compile(r"source-[0-9]{2}-context-manifest\.json"),
    re.compile(r"source-[0-9]{2}-guard-generated\.json"),
)


class ArtifactSealError(ValueError):
    """The candidate artifact cannot cross or satisfy the seal boundary."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactSealError(f"artifact manifest repeats JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ArtifactSealError(f"JSON contains a non-finite number: {value}")


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactSealError(f"artifact directory is unsafe: {path}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ArtifactSealError(f"artifact path is not a directory: {path}")
    return descriptor


def _validated_names(directory_fd: int) -> list[str]:
    names = sorted(os.listdir(directory_fd))
    if len(names) > MAX_FILE_COUNT + 2:
        raise ArtifactSealError("artifact contains too many files")
    for name in names:
        try:
            unicode_round_trip = os.fsencode(name).decode("utf-8") == name
        except UnicodeDecodeError:
            unicode_round_trip = False
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or os.sep in name
            or (os.altsep is not None and os.altsep in name)
            or not unicode_round_trip
            or unicodedata.normalize("NFC", name) != name
            or any(
                unicodedata.category(character).startswith("C") for character in name
            )
        ):
            raise ArtifactSealError(f"artifact filename is not canonical: {name!r}")
    identities = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if len(set(identities)) != len(identities):
        raise ArtifactSealError("artifact filenames collide by filesystem identity")
    return names


def _is_allowed_candidate_name(name: str) -> bool:
    return name in FIXED_CANDIDATE_NAMES or any(
        pattern.fullmatch(name) is not None
        for pattern in VARIABLE_CANDIDATE_NAME_PATTERNS
    )


def _is_guard_log(name: str) -> bool:
    return name == "guard-generated.json" or name.endswith("-guard-generated.json")


def _read_regular(directory_fd: int, name: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactSealError(f"artifact entry is unsafe: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactSealError(f"artifact entry is not regular: {name}")
        if before.st_size > MAX_FILE_BYTES:
            raise ArtifactSealError(f"artifact entry exceeds 16 MiB: {name}")
        chunks: list[bytes] = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(raw) != before.st_size:
            raise ArtifactSealError(f"artifact entry changed while read: {name}")
        if len(raw) > MAX_FILE_BYTES:
            raise ArtifactSealError(f"artifact entry exceeds 16 MiB: {name}")
        return raw, after
    finally:
        os.close(descriptor)


def _write_exclusive(
    directory_fd: int,
    name: str,
    raw: bytes,
    *,
    mode: int = 0o644,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, mode, dir_fd=directory_fd)
    except OSError as exc:
        raise ArtifactSealError(f"cannot create sealed artifact entry: {name}") from exc
    try:
        view = memoryview(raw)
        written = 0
        while written < len(view):
            written += os.write(descriptor, view[written:])
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_candidate(
    source: Path,
    destination: Path,
    *,
    exclude_guard_logs: bool = False,
) -> None:
    """Snapshot a mutable flat candidate into a new root-controlled directory."""

    source_fd = _open_directory(source)
    try:
        names = _validated_names(source_fd)
        forbidden = {MANIFEST_NAME, MANIFEST_DIGEST_NAME}.intersection(names)
        if forbidden:
            raise ArtifactSealError(
                "candidate artifact contains reserved seal entries: "
                + ", ".join(sorted(forbidden))
            )
        unexpected = [name for name in names if not _is_allowed_candidate_name(name)]
        if unexpected:
            raise ArtifactSealError(
                "candidate artifact contains unexpected entries: "
                + ", ".join(unexpected)
            )
        if exclude_guard_logs:
            names = [name for name in names if not _is_guard_log(name)]
        if not names:
            raise ArtifactSealError("candidate artifact is empty")
        try:
            os.mkdir(destination, 0o700)
        except OSError as exc:
            raise ArtifactSealError(
                f"sealed artifact destination already exists or is unsafe: {destination}"
            ) from exc
        destination_fd = _open_directory(destination)
        try:
            total_bytes = 0
            for name in names:
                raw, _ = _read_regular(source_fd, name)
                total_bytes += len(raw)
                if total_bytes > MAX_TOTAL_BYTES:
                    raise ArtifactSealError("artifact exceeds 64 MiB in total")
                _write_exclusive(destination_fd, name, raw)
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def _canonical_manifest(files: list[dict[str, Any]]) -> bytes:
    return (
        json.dumps(
            {"files": files, "schema": MANIFEST_SCHEMA},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def seal(destination: Path) -> str:
    """Inventory the copied files, create the digest witness, and remove writes."""

    directory_fd = _open_directory(destination)
    try:
        names = _validated_names(directory_fd)
        if MANIFEST_NAME in names or MANIFEST_DIGEST_NAME in names:
            raise ArtifactSealError("artifact is already sealed")
        if not names:
            raise ArtifactSealError("artifact is empty")
        files: list[dict[str, Any]] = []
        total_bytes = 0
        for name in names:
            raw, _ = _read_regular(directory_fd, name)
            total_bytes += len(raw)
            if total_bytes > MAX_TOTAL_BYTES:
                raise ArtifactSealError("artifact exceeds 64 MiB in total")
            files.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size": len(raw),
                }
            )
        manifest = _canonical_manifest(files)
        digest = hashlib.sha256(manifest).hexdigest()
        _write_exclusive(directory_fd, MANIFEST_NAME, manifest, mode=0o444)
        _write_exclusive(
            directory_fd,
            MANIFEST_DIGEST_NAME,
            f"{digest}\n".encode("ascii"),
            mode=0o444,
        )
        for name in names:
            os.chmod(name, 0o444, dir_fd=directory_fd, follow_symlinks=False)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.chmod(destination, 0o555, follow_symlinks=False)
    return digest


def verify(destination: Path, *, require_root_owned: bool) -> str:
    """Verify exact inventory, bytes, modes, ownership, and manifest digest."""

    directory_fd = _open_directory(destination)
    try:
        directory_metadata = os.fstat(directory_fd)
        names = _validated_names(directory_fd)
        if MANIFEST_NAME not in names or MANIFEST_DIGEST_NAME not in names:
            raise ArtifactSealError("artifact seal entries are missing")
        manifest_raw, manifest_metadata = _read_regular(directory_fd, MANIFEST_NAME)
        digest_raw, digest_metadata = _read_regular(directory_fd, MANIFEST_DIGEST_NAME)
        try:
            supplied_digest = digest_raw.decode("ascii").removesuffix("\n")
        except UnicodeDecodeError as exc:
            raise ArtifactSealError("artifact manifest digest is not ASCII") from exc
        expected_digest = hashlib.sha256(manifest_raw).hexdigest()
        if supplied_digest != expected_digest:
            raise ArtifactSealError("artifact manifest digest differs")
        try:
            payload = json.loads(
                manifest_raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ArtifactSealError("artifact manifest is invalid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"files", "schema"}:
            raise ArtifactSealError("artifact manifest root is malformed")
        if payload["schema"] != MANIFEST_SCHEMA or not isinstance(
            payload["files"], list
        ):
            raise ArtifactSealError("artifact manifest schema is unsupported")
        try:
            canonical_manifest = _canonical_manifest(payload["files"])
        except (TypeError, ValueError) as exc:
            raise ArtifactSealError("artifact manifest is not canonical") from exc
        if manifest_raw != canonical_manifest:
            raise ArtifactSealError("artifact manifest is not canonical")

        expected_names = {MANIFEST_NAME, MANIFEST_DIGEST_NAME}
        previous_name = ""
        total_bytes = 0
        for item in payload["files"]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
                raise ArtifactSealError("artifact manifest file entry is malformed")
            name = item["path"]
            if (
                not isinstance(name, str)
                or name <= previous_name
                or name in {MANIFEST_NAME, MANIFEST_DIGEST_NAME}
                or "/" in name
            ):
                raise ArtifactSealError("artifact manifest paths are not canonical")
            previous_name = name
            raw, metadata = _read_regular(directory_fd, name)
            expected_names.add(name)
            total_bytes += len(raw)
            if (
                item["size"] != len(raw)
                or item["sha256"] != hashlib.sha256(raw).hexdigest()
            ):
                raise ArtifactSealError(f"artifact manifest differs for: {name}")
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise ArtifactSealError(
                    f"sealed artifact file remains writable: {name}"
                )
            if require_root_owned and metadata.st_uid != 0:
                raise ArtifactSealError(
                    f"sealed artifact file is not root-owned: {name}"
                )
        if total_bytes > MAX_TOTAL_BYTES:
            raise ArtifactSealError("artifact exceeds 64 MiB in total")
        if set(names) != expected_names:
            raise ArtifactSealError("sealed artifact contains unmanifested entries")
        for name, metadata in (
            (MANIFEST_NAME, manifest_metadata),
            (MANIFEST_DIGEST_NAME, digest_metadata),
        ):
            if stat.S_IMODE(metadata.st_mode) & 0o222:
                raise ArtifactSealError(f"seal entry remains writable: {name}")
            if require_root_owned and metadata.st_uid != 0:
                raise ArtifactSealError(f"seal entry is not root-owned: {name}")
        if stat.S_IMODE(directory_metadata.st_mode) & 0o222:
            raise ArtifactSealError("sealed artifact directory remains writable")
        if require_root_owned and directory_metadata.st_uid != 0:
            raise ArtifactSealError("sealed artifact directory is not root-owned")
        return expected_digest
    finally:
        os.close(directory_fd)


def write_publication_witness(
    artifact: Path,
    destination: Path,
    *,
    base_commit: str,
    publication_commit: str,
    publication_tree: str,
) -> None:
    """Bind the protected Git object identity to the sealed audit manifest."""

    for label, value in (
        ("base commit", base_commit),
        ("publication commit", publication_commit),
        ("publication tree", publication_tree),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ArtifactSealError(f"{label} is malformed")
    manifest_digest = verify(artifact, require_root_owned=True)
    payload = (
        json.dumps(
            {
                "artifact_manifest_sha256": manifest_digest,
                "base_commit": base_commit,
                "publication_commit": publication_commit,
                "publication_tree": publication_tree,
                "schema": "axiom-encode/protected-publication-witness/v1",
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    parent_fd = _open_directory(destination.parent)
    try:
        _write_exclusive(parent_fd, destination.name, payload, mode=0o444)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def write_publication_receipt(
    destination: Path,
    raw: bytes,
    *,
    base_branch: str,
    base_commit: str,
    head_branch: str,
    head_commit: str,
    repository: str,
) -> None:
    """Validate and root-snapshot the GitHub PR response after publication."""

    for label, value in (
        ("base commit", base_commit),
        ("head commit", head_commit),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ArtifactSealError(f"{label} is malformed")
    for label, value in (
        ("base branch", base_branch),
        ("head branch", head_branch),
    ):
        if (
            not value
            or len(value.encode("utf-8")) > 255
            or any(
                unicodedata.category(character).startswith("C") for character in value
            )
        ):
            raise ArtifactSealError(f"{label} is malformed")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ArtifactSealError("publication repository is malformed")
    if len(raw) > MAX_FILE_BYTES:
        raise ArtifactSealError("publication receipt exceeds 16 MiB")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ArtifactSealError("publication receipt is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ArtifactSealError("publication receipt is not a JSON object")
    base = payload.get("base")
    head = payload.get("head")
    base_repo = base.get("repo") if isinstance(base, dict) else None
    head_repo = head.get("repo") if isinstance(head, dict) else None
    number = payload.get("number")
    if (
        not isinstance(base, dict)
        or not isinstance(head, dict)
        or not isinstance(base_repo, dict)
        or not isinstance(head_repo, dict)
        or isinstance(number, bool)
        or not isinstance(number, int)
        or number <= 0
        or payload.get("draft") is not True
        or payload.get("state") != "open"
        or base.get("ref") != base_branch
        or base.get("sha") != base_commit
        or base_repo.get("full_name") != repository
        or head.get("ref") != head_branch
        or head.get("sha") != head_commit
        or head_repo.get("full_name") != repository
    ):
        raise ArtifactSealError(
            "publication receipt does not bind the protected base and head"
        )
    try:
        canonical = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactSealError("publication receipt is not canonical JSON") from exc
    parent_fd = _open_directory(destination.parent)
    try:
        _write_exclusive(parent_fd, destination.name, canonical, mode=0o444)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    copy_parser = subparsers.add_parser("copy")
    copy_parser.add_argument("source", type=Path)
    copy_parser.add_argument("destination", type=Path)
    copy_parser.add_argument("--exclude-guard-logs", action="store_true")
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("destination", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("destination", type=Path)
    verify_parser.add_argument("--require-root-owned", action="store_true")
    witness_parser = subparsers.add_parser("write-publication-witness")
    witness_parser.add_argument("artifact", type=Path)
    witness_parser.add_argument("destination", type=Path)
    witness_parser.add_argument("--base-commit", required=True)
    witness_parser.add_argument("--publication-commit", required=True)
    witness_parser.add_argument("--publication-tree", required=True)
    receipt_parser = subparsers.add_parser("write-publication-receipt")
    receipt_parser.add_argument("destination", type=Path)
    receipt_parser.add_argument("--base-branch", required=True)
    receipt_parser.add_argument("--base-commit", required=True)
    receipt_parser.add_argument("--head-branch", required=True)
    receipt_parser.add_argument("--head-commit", required=True)
    receipt_parser.add_argument("--repository", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "copy":
            copy_candidate(
                args.source,
                args.destination,
                exclude_guard_logs=args.exclude_guard_logs,
            )
            return 0
        if args.command == "seal":
            print(seal(args.destination))
            return 0
        if args.command == "verify":
            print(
                verify(
                    args.destination,
                    require_root_owned=args.require_root_owned,
                )
            )
            return 0
        if args.command == "write-publication-witness":
            write_publication_witness(
                args.artifact,
                args.destination,
                base_commit=args.base_commit,
                publication_commit=args.publication_commit,
                publication_tree=args.publication_tree,
            )
            return 0
        if args.command == "write-publication-receipt":
            raw = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
            write_publication_receipt(
                args.destination,
                raw,
                base_branch=args.base_branch,
                base_commit=args.base_commit,
                head_branch=args.head_branch,
                head_commit=args.head_commit,
                repository=args.repository,
            )
            return 0
    except ArtifactSealError as exc:
        raise SystemExit(str(exc)) from exc
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
