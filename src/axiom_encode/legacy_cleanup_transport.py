"""Exact artifact transport for atomic legacy-cleanup publication.

The protected signer and the credentialed publisher never share a checkout or
credential.  They exchange only one canonical signed receipt and one canonical
inventory derived from that receipt.  This module packages an already-verified
``B -> worktree`` contraction and materializes it into a fresh, clean checkout
of the same immutable base after independently reconstructing the base proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from base64 import b64decode
from binascii import Error as BinasciiError
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .legacy_cleanup import (
    LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
    LEGACY_CLEANUP_RECEIPT_DIR,
    LegacyCleanupReceiptError,
    canonical_receipt_bytes,
    decode_strict_json_object,
    deleted_paths,
    is_legacy_cleanup_receipt_path,
    parse_receipt_bytes,
)
from .legacy_cleanup_git import (
    LegacyCleanupGitError,
    plan_legacy_cleanup_base,
    plan_payload_issues,
)
from .legacy_cleanup_guard import verify_worktree_legacy_cleanup_transition
from .legacy_cleanup_signing import verify_legacy_cleanup_receipt_signature

LEGACY_CLEANUP_TRANSPORT_SCHEMA = (
    "axiom-encode/legacy-rulespec-deletion-transport/v1"
)
LEGACY_CLEANUP_INVENTORY_NAME = "deletion-inventory.json"
LEGACY_CLEANUP_MAX_INVENTORY_BYTES = 512 * 1024

_INVENTORY_FIELDS = {
    "schema_version",
    "repository",
    "object_format",
    "base_commit",
    "base_tree",
    "receipt_path",
    "receipt_sha256",
    "deletions",
}
_DELETION_FIELDS = {
    "path",
    "base_mode",
    "base_blob_oid",
    "base_sha256",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OID_RE = re.compile(r"[0-9a-f]+")


class LegacyCleanupTransportError(ValueError):
    """The cleanup artifact cannot prove one exact atomic contraction."""


def canonical_inventory_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize the strict transport inventory."""

    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


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


def _git(repo: Path, *arguments: str) -> bytes:
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
            "core.sparseCheckout=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(repo),
            *arguments,
        ],
        capture_output=True,
        check=False,
        env=_git_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LegacyCleanupTransportError(
            f"cleanup transport Git inspection failed ({' '.join(arguments)}): "
            f"{detail or 'git command failed'}"
        )
    return completed.stdout


def _require_checkout_root(repo: Path) -> Path:
    try:
        checkout = Path(repo).resolve(strict=True)
        top = Path(_git(checkout, "rev-parse", "--show-toplevel").decode().strip())
        top = top.resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise LegacyCleanupTransportError(
            "cleanup transport repository is not a readable Git checkout"
        ) from exc
    if checkout != top:
        raise LegacyCleanupTransportError(
            "cleanup transport repository must be the exact Git checkout root"
        )
    return checkout


def _require_regular_0644(path: Path, *, label: str, max_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise LegacyCleanupTransportError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o644
        or before.st_size > max_bytes
    ):
        raise LegacyCleanupTransportError(
            f"{label} must be one bounded regular 0644 file"
        )
    try:
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise LegacyCleanupTransportError(f"{label} is unreadable") from exc
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise LegacyCleanupTransportError(f"{label} changed while it was read")
    if len(raw) != before.st_size:
        raise LegacyCleanupTransportError(f"{label} size changed while it was read")
    return raw


def _parse_public_key(material: str) -> Ed25519PublicKey:
    text = material.strip().replace("\\n", "\n")
    if not text or len(text) > 16_384:
        raise LegacyCleanupTransportError("apply trust root is empty or oversized")
    if text.startswith("-----BEGIN"):
        try:
            loaded = serialization.load_pem_public_key(text.encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise LegacyCleanupTransportError(
                "apply trust root PEM is malformed"
            ) from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise LegacyCleanupTransportError("apply trust root must be Ed25519")
        return loaded
    try:
        raw = b64decode(text.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError) as exc:
        raise LegacyCleanupTransportError(
            "apply trust root is not strict base64 or PEM"
        ) from exc
    if len(raw) != 32:
        raise LegacyCleanupTransportError(
            "apply trust root is not a raw Ed25519 public key"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


def load_apply_trust_root(path: Path) -> Ed25519PublicKey:
    """Load only the public apply root from a provisioned trust-root file."""

    raw = _require_regular_0644(
        Path(path),
        label="signing trust-root configuration",
        max_bytes=64 * 1024,
    )
    try:
        payload = decode_strict_json_object(
            raw,
            label="signing trust-root configuration",
            max_bytes=64 * 1024,
        )
    except LegacyCleanupReceiptError as exc:
        raise LegacyCleanupTransportError(str(exc)) from exc
    schema = payload.get("schema")
    if schema == "axiom-encode/signing-trust-roots/v2":
        expected_fields = {
            "schema",
            "apply_ed25519_public_key",
            "eval_ed25519_public_key",
            "corpus_release_ed25519_public_key",
        }
    elif schema == "axiom-encode/signing-trust-roots/v3":
        expected_fields = {
            "schema",
            "apply_ed25519_public_key",
            "eval_ed25519_public_key",
            "corpus_release_ed25519_public_keys",
        }
    else:
        raise LegacyCleanupTransportError(
            "signing trust-root configuration schema is unsupported"
        )
    if set(payload) != expected_fields:
        raise LegacyCleanupTransportError(
            "signing trust-root configuration has the wrong fields"
        )
    apply_root = payload.get("apply_ed25519_public_key")
    if not isinstance(apply_root, str):
        raise LegacyCleanupTransportError("apply trust root is malformed")
    return _parse_public_key(apply_root)


def _receipt_deletions(payload: Mapping[str, object]) -> list[dict[str, str]]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise LegacyCleanupTransportError("cleanup receipt groups are unavailable")
    result: list[dict[str, str]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise LegacyCleanupTransportError("cleanup receipt group is malformed")
        for kind in ("primary", "companion"):
            record = group.get(kind)
            if not isinstance(record, dict):
                raise LegacyCleanupTransportError(
                    "cleanup receipt deletion evidence is malformed"
                )
            try:
                result.append(
                    {
                        "path": str(record["path"]),
                        "base_mode": str(record["base_mode"]),
                        "base_blob_oid": str(record["base_blob_oid"]),
                        "base_sha256": str(record["base_sha256"]),
                    }
                )
            except KeyError as exc:
                raise LegacyCleanupTransportError(
                    "cleanup receipt deletion evidence is incomplete"
                ) from exc
    return sorted(result, key=lambda item: item["path"])


def inventory_for_receipt(
    payload: Mapping[str, object],
    *,
    receipt_path: Path,
    receipt_raw: bytes,
) -> dict[str, object]:
    repository = payload.get("repository")
    if not isinstance(repository, dict):
        raise LegacyCleanupTransportError("cleanup receipt repository proof is missing")
    return {
        "schema_version": LEGACY_CLEANUP_TRANSPORT_SCHEMA,
        "repository": repository["repository"],
        "object_format": repository["object_format"],
        "base_commit": repository["base_commit"],
        "base_tree": repository["base_tree"],
        "receipt_path": receipt_path.as_posix(),
        "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "deletions": _receipt_deletions(payload),
    }


def parse_inventory_bytes(raw: bytes) -> dict[str, Any]:
    """Parse and strictly validate one canonical deletion inventory."""

    try:
        payload = decode_strict_json_object(
            raw,
            label="legacy cleanup deletion inventory",
            max_bytes=LEGACY_CLEANUP_MAX_INVENTORY_BYTES,
        )
    except LegacyCleanupReceiptError as exc:
        raise LegacyCleanupTransportError(str(exc)) from exc
    if raw != canonical_inventory_bytes(payload):
        raise LegacyCleanupTransportError(
            "legacy cleanup deletion inventory is not canonical JSON"
        )
    if set(payload) != _INVENTORY_FIELDS:
        raise LegacyCleanupTransportError(
            "legacy cleanup deletion inventory has the wrong fields"
        )
    object_format = payload.get("object_format")
    oid_length = 40 if object_format == "sha1" else 64 if object_format == "sha256" else 0
    receipt_value = payload.get("receipt_path")
    receipt = Path(receipt_value) if isinstance(receipt_value, str) else Path()
    if (
        payload.get("schema_version") != LEGACY_CLEANUP_TRANSPORT_SCHEMA
        or not isinstance(payload.get("repository"), str)
        or not isinstance(payload.get("base_commit"), str)
        or len(str(payload.get("base_commit"))) != oid_length
        or _OID_RE.fullmatch(str(payload.get("base_commit"))) is None
        or not isinstance(payload.get("base_tree"), str)
        or len(str(payload.get("base_tree"))) != oid_length
        or _OID_RE.fullmatch(str(payload.get("base_tree"))) is None
        or not isinstance(receipt_value, str)
        or receipt.as_posix() != receipt_value
        or not is_legacy_cleanup_receipt_path(receipt)
        or _SHA256_RE.fullmatch(str(payload.get("receipt_sha256"))) is None
    ):
        raise LegacyCleanupTransportError(
            "legacy cleanup deletion inventory identity is malformed"
        )
    deletions = payload.get("deletions")
    if not isinstance(deletions, list) or not 2 <= len(deletions) <= 128:
        raise LegacyCleanupTransportError(
            "legacy cleanup deletion inventory must list 2..128 files"
        )
    paths: list[str] = []
    for index, record in enumerate(deletions):
        if not isinstance(record, dict) or set(record) != _DELETION_FIELDS:
            raise LegacyCleanupTransportError(
                f"legacy cleanup deletion inventory entry {index} is malformed"
            )
        path = record.get("path")
        relative = Path(path) if isinstance(path, str) else Path()
        if (
            not isinstance(path, str)
            or relative.is_absolute()
            or relative.as_posix() != path
            or any(part in {"", ".", ".."} for part in relative.parts)
            or record.get("base_mode") != "100644"
            or len(str(record.get("base_blob_oid"))) != oid_length
            or _OID_RE.fullmatch(str(record.get("base_blob_oid"))) is None
            or _SHA256_RE.fullmatch(str(record.get("base_sha256"))) is None
        ):
            raise LegacyCleanupTransportError(
                f"legacy cleanup deletion inventory entry {index} is invalid"
            )
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise LegacyCleanupTransportError(
            "legacy cleanup deletion inventory paths are not unique and sorted"
        )
    return payload


def _exclusive_regular_write(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LegacyCleanupTransportError(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise LegacyCleanupTransportError(f"{label} is not a real directory")


def _artifact_files(artifact_root: Path) -> tuple[Path, Path]:
    source_root = Path(artifact_root)
    _require_real_directory(source_root, label="cleanup artifact root")
    root = source_root.resolve(strict=True)
    files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            _require_real_directory(child, label="cleanup artifact directory")
            directories.append(child.relative_to(root))
        for name in file_names:
            child = current_path / name
            _require_regular_0644(
                child,
                label="cleanup artifact file",
                max_bytes=LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
            )
            files.append(child.relative_to(root))
    inventory_path = Path(LEGACY_CLEANUP_INVENTORY_NAME)
    receipts = [path for path in files if is_legacy_cleanup_receipt_path(path)]
    allowed_directories = {
        LEGACY_CLEANUP_RECEIPT_DIR.parent,
        LEGACY_CLEANUP_RECEIPT_DIR,
    }
    if (
        len(files) != 2
        or inventory_path not in files
        or len(receipts) != 1
        or set(directories) != allowed_directories
    ):
        raise LegacyCleanupTransportError(
            "cleanup artifact must contain exactly one canonical receipt and "
            "one deletion inventory"
        )
    return root / inventory_path, root / receipts[0]


def _validate_receipt_and_inventory(
    artifact_root: Path,
    *,
    verifier: Ed25519PublicKey,
) -> tuple[dict[str, Any], dict[str, Any], Path, bytes]:
    inventory_path, receipt_source = _artifact_files(artifact_root)
    inventory_raw = _require_regular_0644(
        inventory_path,
        label="cleanup deletion inventory",
        max_bytes=LEGACY_CLEANUP_MAX_INVENTORY_BYTES,
    )
    inventory = parse_inventory_bytes(inventory_raw)
    receipt_relative = receipt_source.relative_to(Path(artifact_root).resolve(strict=True))
    if receipt_relative.as_posix() != inventory["receipt_path"]:
        raise LegacyCleanupTransportError(
            "cleanup artifact receipt path differs from its deletion inventory"
        )
    receipt_raw = _require_regular_0644(
        receipt_source,
        label="cleanup signed receipt",
        max_bytes=LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
    )
    try:
        receipt = parse_receipt_bytes(
            receipt_raw,
            expected_path=receipt_relative,
        )
        verify_legacy_cleanup_receipt_signature(receipt, verifier)
    except LegacyCleanupReceiptError as exc:
        raise LegacyCleanupTransportError(str(exc)) from exc
    expected_inventory = inventory_for_receipt(
        receipt,
        receipt_path=receipt_relative,
        receipt_raw=receipt_raw,
    )
    if inventory != expected_inventory:
        raise LegacyCleanupTransportError(
            "cleanup deletion inventory differs from the signed receipt"
        )
    return inventory, receipt, receipt_relative, receipt_raw


def build_transport_artifact(
    repo: Path,
    *,
    base_ref: str,
    artifact_root: Path,
    verifier: Ed25519PublicKey,
) -> dict[str, object]:
    """Package one exact signer-produced worktree contraction."""

    checkout = _require_checkout_root(repo)
    result = verify_worktree_legacy_cleanup_transition(
        checkout,
        base_ref=base_ref,
        verifier=verifier,
    )
    if result.issues or result.receipt is None or result.receipt_path is None:
        raise LegacyCleanupTransportError(
            "cleanup signer worktree is not an exact authorized transition: "
            + "; ".join(result.issues or ("verified receipt is unavailable",))
        )
    receipt_path = result.receipt_path
    receipt_raw = _require_regular_0644(
        checkout / receipt_path,
        label="cleanup signer receipt",
        max_bytes=LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
    )
    if receipt_raw != canonical_receipt_bytes(result.receipt):
        raise LegacyCleanupTransportError("cleanup signer receipt bytes changed")
    try:
        plan = plan_legacy_cleanup_base(
            checkout,
            base_ref=base_ref,
            primary_paths=[
                group["primary"]["path"] for group in result.receipt["groups"]
            ],
            require_clean_checkout=False,
        )
    except (KeyError, LegacyCleanupGitError, TypeError) as exc:
        raise LegacyCleanupTransportError(
            "cleanup signer receipt cannot reconstruct its immutable base"
        ) from exc
    issues = plan_payload_issues(result.receipt, plan)
    if issues:
        raise LegacyCleanupTransportError(
            "cleanup signer receipt base proof is stale: " + "; ".join(issues)
        )
    output = Path(artifact_root)
    if output.exists() or output.is_symlink():
        raise LegacyCleanupTransportError(
            "cleanup artifact output must not already exist"
        )
    receipt_target = output / receipt_path
    receipt_target.parent.mkdir(parents=True, mode=0o755)
    inventory = inventory_for_receipt(
        result.receipt,
        receipt_path=receipt_path,
        receipt_raw=receipt_raw,
    )
    _exclusive_regular_write(receipt_target, receipt_raw)
    _exclusive_regular_write(
        output / LEGACY_CLEANUP_INVENTORY_NAME,
        canonical_inventory_bytes(inventory),
    )
    _artifact_files(output)
    return inventory


def _safe_target(repo: Path, relative: Path, *, label: str) -> Path:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise LegacyCleanupTransportError(f"{label} is not a canonical relative path")
    current = repo
    for part in relative.parts[:-1]:
        current /= part
        _require_real_directory(current, label=f"{label} ancestor")
    return current / relative.name


def _require_live_base_file(target: Path, record: Mapping[str, str]) -> bytes:
    raw = _require_regular_0644(
        target,
        label=f"cleanup target {record['path']}",
        max_bytes=16 * 1024 * 1024,
    )
    if hashlib.sha256(raw).hexdigest() != record["base_sha256"]:
        raise LegacyCleanupTransportError(
            f"cleanup target differs from base proof: {record['path']}"
        )
    return raw


def materialize_transport_artifact(
    repo: Path,
    *,
    base_ref: str,
    artifact_root: Path,
    verifier: Ed25519PublicKey,
) -> dict[str, object]:
    """Apply only a verified receipt and its exact deletions to clean base B."""

    checkout = _require_checkout_root(repo)
    inventory, receipt, receipt_relative, receipt_raw = (
        _validate_receipt_and_inventory(artifact_root, verifier=verifier)
    )
    if inventory["base_commit"] != base_ref:
        raise LegacyCleanupTransportError(
            "cleanup artifact does not bind the requested protected base"
        )
    receipt_groups = receipt.get("groups")
    if not isinstance(receipt_groups, list):
        raise LegacyCleanupTransportError("cleanup receipt groups are unavailable")
    try:
        plan = plan_legacy_cleanup_base(
            checkout,
            base_ref=base_ref,
            primary_paths=[group["primary"]["path"] for group in receipt_groups],
            require_clean_checkout=True,
        )
    except (KeyError, LegacyCleanupGitError, TypeError) as exc:
        raise LegacyCleanupTransportError(
            "cleanup publisher cannot reconstruct the immutable base proof"
        ) from exc
    issues = plan_payload_issues(receipt, plan)
    if issues:
        raise LegacyCleanupTransportError(
            "cleanup artifact base proof is stale: " + "; ".join(issues)
        )
    expected_paths = tuple(sorted(deleted_paths(receipt)))
    inventory_paths = tuple(record["path"] for record in inventory["deletions"])
    if expected_paths != inventory_paths:
        raise LegacyCleanupTransportError(
            "cleanup artifact deletion inventory is incomplete"
        )
    receipt_parent = checkout / LEGACY_CLEANUP_RECEIPT_DIR.parent
    _require_real_directory(receipt_parent, label="cleanup receipt parent")
    receipt_root = checkout / LEGACY_CLEANUP_RECEIPT_DIR
    if receipt_root.exists() or receipt_root.is_symlink():
        _require_real_directory(
            receipt_root,
            label="cleanup receipt destination directory",
        )
    receipt_target = receipt_root / receipt_relative.name
    if receipt_target.exists() or receipt_target.is_symlink():
        raise LegacyCleanupTransportError(
            "cleanup receipt destination already exists"
        )
    deletion_targets: list[tuple[Path, bytes]] = []
    for record in inventory["deletions"]:
        target = _safe_target(
            checkout,
            Path(record["path"]),
            label="cleanup deletion destination",
        )
        deletion_targets.append((target, _require_live_base_file(target, record)))

    created_receipt_root = False
    if not receipt_root.exists():
        receipt_root.mkdir(mode=0o755)
        created_receipt_root = True
    try:
        _exclusive_regular_write(receipt_target, receipt_raw)
        for target, _raw in deletion_targets:
            target.unlink()
        toolchain = receipt.get("toolchain")
        if not isinstance(toolchain, dict):
            raise LegacyCleanupTransportError("cleanup receipt toolchain is unavailable")
        result = verify_worktree_legacy_cleanup_transition(
            checkout,
            base_ref=base_ref,
            verifier=verifier,
            expected_toolchain=toolchain,
        )
        if result.issues or result.receipt_path != receipt_relative:
            raise LegacyCleanupTransportError(
                "materialized cleanup transition failed exact worktree verification: "
                + "; ".join(result.issues or ("receipt path mismatch",))
            )
    except BaseException:
        if receipt_target.exists() and not receipt_target.is_symlink():
            receipt_target.unlink()
        for target, raw in deletion_targets:
            if not target.exists() and not target.is_symlink():
                _exclusive_regular_write(target, raw)
        if created_receipt_root:
            try:
                receipt_target.parent.rmdir()
            except OSError:
                pass
        raise
    return inventory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package or materialize one exact legacy-cleanup artifact"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "materialize"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--repo", type=Path, required=True)
        subparser.add_argument("--base-ref", required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--trusted-signing-roots", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verifier = load_apply_trust_root(args.trusted_signing_roots)
        if args.command == "build":
            payload = build_transport_artifact(
                args.repo,
                base_ref=args.base_ref,
                artifact_root=args.artifact_root,
                verifier=verifier,
            )
        else:
            payload = materialize_transport_artifact(
                args.repo,
                base_ref=args.base_ref,
                artifact_root=args.artifact_root,
                verifier=verifier,
            )
    except (LegacyCleanupTransportError, OSError) as exc:
        parser = _parser()
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
