"""Pure contract for signed unmanifested-legacy RuleSpec deletion receipts.

This module has no Git, signing-key, staging, or filesystem mutation authority.
It defines the negative-provenance receipt schema, semantic identity, canonical
JSON parser, primary/companion grouping, and typed signing payload used by the
CLI, guard, and publisher.
"""

from __future__ import annotations

import hashlib
import json
import re
from base64 import b64decode
from binascii import Error as BinasciiError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS

LEGACY_CLEANUP_RECEIPT_SCHEMA = "axiom-encode/legacy-rulespec-deletion-receipt/v1"
LEGACY_CLEANUP_TOOL = "axiom-encode cleanup-unmanifested-legacy"
LEGACY_CLEANUP_PROVENANCE_CLASS = "unmanifested-legacy-contraction"
LEGACY_CLEANUP_RECEIPT_DIR = Path(".axiom/legacy-rulespec-deletion-receipts")
LEGACY_CLEANUP_SIGNATURE_DOMAIN = LEGACY_CLEANUP_RECEIPT_SCHEMA
LEGACY_CLEANUP_SIGNATURE_ALGORITHM = "ed25519-domain-v1"
LEGACY_CLEANUP_IDENTITY_DOMAIN = (
    b"axiom-encode/legacy-rulespec-deletion-receipt-identity/v1\x00"
)
LEGACY_CLEANUP_SIGNATURE_PREFIX = (
    LEGACY_CLEANUP_SIGNATURE_DOMAIN.encode("ascii") + b"\x00"
)
LEGACY_CLEANUP_BASE_PROOF_SCHEMA = "axiom-encode/legacy-rulespec-deletion-base-proof/v1"
LEGACY_CLEANUP_VALIDATION_SCHEMA = (
    "axiom-encode/legacy-rulespec-deletion-validation-execution/v1"
)
LEGACY_CLEANUP_MAX_GROUPS = 64
LEGACY_CLEANUP_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
LEGACY_CLEANUP_MAX_JSON_DEPTH = 24
LEGACY_CLEANUP_MAX_JSON_NODES = 32_768
LEGACY_CLEANUP_MAX_REASON_CHARS = 4_096
LEGACY_CLEANUP_MAX_PATH_CHARS = 1_024
LEGACY_CLEANUP_MAX_VERSION_CHARS = 128
LEGACY_CLEANUP_MAX_RELEASE_NAME_CHARS = 256
LEGACY_CLEANUP_MAX_COMMAND_TOKENS = 256
LEGACY_CLEANUP_MAX_COMMAND_TOKEN_CHARS = 1_024

LEGACY_CLEANUP_PROVENANCE_ASSERTIONS = {
    "model_generated": False,
    "source_attestation": False,
    "signed_import_inventory_eligible": False,
    "encoding_run_reconstruction_eligible": False,
}
LEGACY_CLEANUP_VALIDATION_CHECKS = (
    "repository-tests",
    "repository-layout",
    "validation-waivers",
    "remaining-rulespec-validation",
    "remaining-companion-tests",
    "remaining-proof-validation",
    "money-atom-proof-validation",
    "oracle-coverage",
    "metadata-reference-closure",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_JURISDICTION_RE = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
_GENERATED_AT_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_RECEIPT_FIELDS = {
    "schema_version",
    "tool",
    "provenance_class",
    "receipt_identity_sha256",
    "generated_at",
    "reason",
    "repository",
    "toolchain",
    "base_proof",
    "validation_execution",
    "provenance_assertions",
    "groups",
    "signature",
}
_REPOSITORY_FIELDS = {
    "repository",
    "object_format",
    "base_commit",
    "base_tree",
    "projected_post_deletion_tree",
}
_FILE_EVIDENCE_FIELDS = {
    "path",
    "base_mode",
    "base_blob_oid",
    "base_sha256",
}
_DELETION_EVIDENCE_FIELDS = _FILE_EVIDENCE_FIELDS | {"result"}
_TOOLCHAIN_FIELDS = {
    "axiom_encode",
    "axiom_rules_engine",
    "rulespec_dependencies",
    "corpus_release",
    "validation_waiver_set_sha256",
    "base_files",
}
_BASE_PROOF_FIELDS = {
    "schema",
    "ownership_inventory_sha256",
    "provenance_record_count",
    "surviving_reference_inventory_sha256",
    "surviving_blob_count",
}
_VALIDATION_FIELDS = {
    "schema",
    "status",
    "engine_execution",
    "projected_post_deletion_tree",
    "checks",
}
_VALIDATION_CHECK_FIELDS = {
    "name",
    "command",
    "target_count",
    "target_list_sha256",
    "exit_code",
    "output_sha256",
}


class LegacyCleanupReceiptError(ValueError):
    """A cleanup receipt or requested primary group is not canonical."""


def _canonical_path_text(raw: str | Path, *, label: str) -> str:
    if not isinstance(raw, (str, Path)):
        raise LegacyCleanupReceiptError(f"{label} must be a path string")
    source = str(raw)
    text = Path(source).as_posix()
    path = Path(text)
    if (
        not source
        or len(source) > LEGACY_CLEANUP_MAX_PATH_CHARS
        or "\\" in source
        or path.is_absolute()
        or text != source
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in source)
    ):
        raise LegacyCleanupReceiptError(f"{label} is not a canonical repository path")
    return text


def canonical_primary_path(raw: str | Path) -> Path:
    """Return one bounded canonical primary RuleSpec path."""

    try:
        text = _canonical_path_text(raw, label="legacy cleanup primary")
    except LegacyCleanupReceiptError as exc:
        raise LegacyCleanupReceiptError(
            f"legacy cleanup target is not a canonical primary RuleSpec path: {raw!s}"
        ) from exc
    path = Path(text)
    if (
        len(path.parts) < 3
        or _JURISDICTION_RE.fullmatch(path.parts[0]) is None
        or path.parts[1] not in RULESPEC_ATOMIC_MODULE_ROOTS
        or path.suffix != ".yaml"
        or path.name.endswith(".test.yaml")
    ):
        raise LegacyCleanupReceiptError(
            f"legacy cleanup target is not a canonical primary RuleSpec path: {text}"
        )
    return path


def canonical_primary_paths(raw_paths: Sequence[str | Path]) -> tuple[Path, ...]:
    if not raw_paths or len(raw_paths) > LEGACY_CLEANUP_MAX_GROUPS:
        raise LegacyCleanupReceiptError(
            f"legacy cleanup requires 1..{LEGACY_CLEANUP_MAX_GROUPS} primary paths"
        )
    paths = tuple(canonical_primary_path(path) for path in raw_paths)
    if len(set(paths)) != len(paths):
        raise LegacyCleanupReceiptError("legacy cleanup primary paths must be unique")
    return tuple(sorted(paths, key=Path.as_posix))


def companion_path(primary: Path) -> Path:
    canonical = canonical_primary_path(primary)
    return canonical.with_name(f"{canonical.stem}.test.yaml")


def utc_generated_at() -> str:
    """Return the one timestamp form accepted by the v1 schema."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def decode_strict_json_object(
    raw: bytes,
    *,
    label: str,
    max_bytes: int = LEGACY_CLEANUP_MAX_RECEIPT_BYTES,
) -> dict[str, Any]:
    """Decode bounded UTF-8 JSON with duplicate/non-finite rejection."""

    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise LegacyCleanupReceiptError(f"{label} exceeds its byte limit")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise LegacyCleanupReceiptError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise LegacyCleanupReceiptError(
            f"{label} contains invalid JSON constant {value}"
        )

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise LegacyCleanupReceiptError(
            f"{label} is not unambiguous UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise LegacyCleanupReceiptError(f"{label} must be a JSON object")

    node_count = 0

    def inspect(value: object, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if depth > LEGACY_CLEANUP_MAX_JSON_DEPTH:
            raise LegacyCleanupReceiptError(f"{label} exceeds its JSON depth limit")
        if node_count > LEGACY_CLEANUP_MAX_JSON_NODES:
            raise LegacyCleanupReceiptError(f"{label} exceeds its JSON node limit")
        if isinstance(value, float):
            raise LegacyCleanupReceiptError(f"{label} must not contain JSON floats")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise LegacyCleanupReceiptError(f"{label} has a non-string key")
                inspect(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                inspect(child, depth + 1)

    inspect(payload, 1)
    return payload


def canonical_receipt_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def receipt_identity_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return non-recursive semantic fields used for the receipt filename."""

    return {
        key: payload[key]
        for key in (
            "schema_version",
            "tool",
            "provenance_class",
            "reason",
            "repository",
            "toolchain",
            "base_proof",
            "validation_execution",
            "provenance_assertions",
            "groups",
        )
        if key in payload
    }


def receipt_identity_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        receipt_identity_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(LEGACY_CLEANUP_IDENTITY_DOMAIN + canonical).hexdigest()


def receipt_path(identity: str) -> Path:
    if _SHA256_RE.fullmatch(identity) is None:
        raise LegacyCleanupReceiptError("legacy cleanup receipt identity is invalid")
    return LEGACY_CLEANUP_RECEIPT_DIR / f"{identity}.json"


def is_legacy_cleanup_receipt_path(path: str | Path) -> bool:
    relative = Path(path)
    return (
        not relative.is_absolute()
        and relative.parent == LEGACY_CLEANUP_RECEIPT_DIR
        and relative.suffix == ".json"
        and _SHA256_RE.fullmatch(relative.stem) is not None
    )


def unsigned_receipt_bytes(payload: Mapping[str, object]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def cleanup_signature_payload(payload: Mapping[str, object]) -> bytes:
    return LEGACY_CLEANUP_SIGNATURE_PREFIX + unsigned_receipt_bytes(payload)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _valid_oid(value: object, object_format: str) -> bool:
    length = 40 if object_format == "sha1" else 64 if object_format == "sha256" else 0
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _valid_bounded_string(value: object, *, max_chars: int) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= max_chars
        and value == value.strip()
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _file_evidence_issue(
    value: object,
    *,
    object_format: str,
    expected_path: Path | None,
    deletion: bool,
) -> str | None:
    fields = _DELETION_EVIDENCE_FIELDS if deletion else _FILE_EVIDENCE_FIELDS
    if not isinstance(value, dict) or set(value) != fields:
        return "file evidence has a noncanonical shape"
    try:
        path_text = _canonical_path_text(value.get("path"), label="file evidence path")
    except LegacyCleanupReceiptError as exc:
        return str(exc)
    if expected_path is not None and path_text != expected_path.as_posix():
        return "file evidence path does not match its group"
    if value.get("base_mode") != "100644":
        return "file evidence is not a regular 100644 base blob"
    if not _valid_oid(value.get("base_blob_oid"), object_format):
        return "file evidence base blob OID is invalid"
    if not _valid_sha256(value.get("base_sha256")):
        return "file evidence base SHA-256 is invalid"
    if deletion and value.get("result") != "absent":
        return "deletion evidence result must be absent"
    return None


def _validate_generated_at(value: object) -> bool:
    if not isinstance(value, str) or _GENERATED_AT_RE.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo == timezone.utc and parsed.microsecond == 0


def receipt_structure_issues(payload: Mapping[str, object]) -> list[str]:
    """Return exact-schema issues without consulting mutable external state."""

    issues: list[str] = []
    if set(payload) != _RECEIPT_FIELDS:
        issues.append("receipt does not match the exact v1 field set")
    if payload.get("schema_version") != LEGACY_CLEANUP_RECEIPT_SCHEMA:
        issues.append("receipt schema_version is invalid")
    if payload.get("tool") != LEGACY_CLEANUP_TOOL:
        issues.append("receipt tool is invalid")
    if payload.get("provenance_class") != LEGACY_CLEANUP_PROVENANCE_CLASS:
        issues.append("receipt provenance class is invalid")
    reason = payload.get("reason")
    if (
        not isinstance(reason, str)
        or not reason
        or reason != reason.strip()
        or len(reason) > LEGACY_CLEANUP_MAX_REASON_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in reason)
    ):
        issues.append("receipt reason is not a bounded normalized audit string")
    if not _validate_generated_at(payload.get("generated_at")):
        issues.append("receipt generated_at is not canonical UTC RFC3339")

    repository = payload.get("repository")
    object_format = ""
    if not isinstance(repository, dict) or set(repository) != _REPOSITORY_FIELDS:
        issues.append("receipt repository binding is malformed")
    else:
        repository_name = repository.get("repository")
        if (
            not isinstance(repository_name, str)
            or re.fullmatch(
                r"github\.com/TheAxiomFoundation/rulespec-[a-z]{2}",
                repository_name,
            )
            is None
        ):
            issues.append("receipt repository identity is not canonical")
        object_format = str(repository.get("object_format", ""))
        if object_format not in {"sha1", "sha256"}:
            issues.append("receipt object format is unsupported")
        for field in (
            "base_commit",
            "base_tree",
            "projected_post_deletion_tree",
        ):
            if not _valid_oid(repository.get(field), object_format):
                issues.append(f"receipt repository.{field} is invalid")

    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != _TOOLCHAIN_FIELDS:
        issues.append("receipt toolchain binding is malformed")
    else:
        encoder = toolchain.get("axiom_encode")
        if (
            not isinstance(encoder, dict)
            or set(encoder) != {"repository", "object_format", "commit", "version"}
            or encoder.get("repository") != "github.com/TheAxiomFoundation/axiom-encode"
            or encoder.get("object_format") not in {"sha1", "sha256"}
            or not _valid_oid(
                encoder.get("commit"), str(encoder.get("object_format", ""))
            )
            or not _valid_bounded_string(
                encoder.get("version"), max_chars=LEGACY_CLEANUP_MAX_VERSION_CHARS
            )
        ):
            issues.append("receipt axiom-encode pin is malformed")
        engine = toolchain.get("axiom_rules_engine")
        if (
            not isinstance(engine, dict)
            or set(engine) != {"repository", "object_format", "commit"}
            or engine.get("repository")
            != "github.com/TheAxiomFoundation/axiom-rules-engine"
            or engine.get("object_format") not in {"sha1", "sha256"}
            or not _valid_oid(
                engine.get("commit"), str(engine.get("object_format", ""))
            )
        ):
            issues.append("receipt rules-engine pin is malformed")
        dependencies = toolchain.get("rulespec_dependencies")
        dependency_repositories: list[str] = []
        if (
            not isinstance(dependencies, list)
            or len(dependencies) > LEGACY_CLEANUP_MAX_GROUPS
        ):
            issues.append("receipt RuleSpec dependency pins are malformed")
        else:
            for dependency in dependencies:
                repository_name = (
                    dependency.get("repository")
                    if isinstance(dependency, dict)
                    else None
                )
                dependency_format = (
                    dependency.get("object_format")
                    if isinstance(dependency, dict)
                    else None
                )
                if (
                    not isinstance(dependency, dict)
                    or set(dependency) != {"repository", "object_format", "commit"}
                    or not isinstance(repository_name, str)
                    or re.fullmatch(
                        r"github\.com/TheAxiomFoundation/rulespec-[a-z]{2}",
                        repository_name,
                    )
                    is None
                    or dependency_format not in {"sha1", "sha256"}
                    or not _valid_oid(
                        dependency.get("commit"), str(dependency_format or "")
                    )
                ):
                    issues.append("receipt RuleSpec dependency pin is malformed")
                    continue
                dependency_repositories.append(repository_name)
            if dependency_repositories != sorted(dependency_repositories) or len(
                set(dependency_repositories)
            ) != len(dependency_repositories):
                issues.append(
                    "receipt RuleSpec dependency pins are not unique and sorted"
                )
        corpus = toolchain.get("corpus_release")
        if (
            not isinstance(corpus, dict)
            or set(corpus) != {"name", "content_sha256", "selector_sha256"}
            or not _valid_bounded_string(
                corpus.get("name"),
                max_chars=LEGACY_CLEANUP_MAX_RELEASE_NAME_CHARS,
            )
            or not _valid_sha256(corpus.get("content_sha256"))
            or not _valid_sha256(corpus.get("selector_sha256"))
        ):
            issues.append("receipt corpus release pin is malformed")
        if not _valid_sha256(toolchain.get("validation_waiver_set_sha256")):
            issues.append("receipt validation-waiver pin is malformed")
        base_files = toolchain.get("base_files")
        if not isinstance(base_files, dict) or set(base_files) != {
            "toolchain",
            "validation_waiver_set",
        }:
            issues.append("receipt protected base-file pins are malformed")
        else:
            for key, expected in (
                ("toolchain", Path(".axiom/toolchain.toml")),
                ("validation_waiver_set", Path("known-validation-gaps.yaml")),
            ):
                issue = _file_evidence_issue(
                    base_files.get(key),
                    object_format=object_format,
                    expected_path=expected,
                    deletion=False,
                )
                if issue:
                    issues.append(f"receipt {key} {issue}")

    base_proof = payload.get("base_proof")
    if not isinstance(base_proof, dict) or set(base_proof) != _BASE_PROOF_FIELDS:
        issues.append("receipt immutable-base proof is malformed")
    elif (
        base_proof.get("schema") != LEGACY_CLEANUP_BASE_PROOF_SCHEMA
        or not _valid_sha256(base_proof.get("ownership_inventory_sha256"))
        or type(base_proof.get("provenance_record_count")) is not int
        or not 0 <= base_proof.get("provenance_record_count", -1) <= 1_000_000
        or not _valid_sha256(base_proof.get("surviving_reference_inventory_sha256"))
        or type(base_proof.get("surviving_blob_count")) is not int
        or not 0 <= base_proof.get("surviving_blob_count", -1) <= 1_000_000
    ):
        issues.append("receipt immutable-base proof values are invalid")

    validation = payload.get("validation_execution")
    if not isinstance(validation, dict) or set(validation) != _VALIDATION_FIELDS:
        issues.append("receipt validation execution evidence is malformed")
    else:
        if (
            validation.get("schema") != LEGACY_CLEANUP_VALIDATION_SCHEMA
            or validation.get("status") != "passed"
            or validation.get("engine_execution") is not True
            or not _valid_oid(
                validation.get("projected_post_deletion_tree"), object_format
            )
            or (
                isinstance(repository, dict)
                and validation.get("projected_post_deletion_tree")
                != repository.get("projected_post_deletion_tree")
            )
        ):
            issues.append(
                "receipt validation execution did not pass the bound projection"
            )
        checks = validation.get("checks")
        if not isinstance(checks, list) or len(checks) != len(
            LEGACY_CLEANUP_VALIDATION_CHECKS
        ):
            issues.append("receipt validation checks are incomplete")
        else:
            names: list[str] = []
            for index, check in enumerate(checks):
                if (
                    not isinstance(check, dict)
                    or set(check) != _VALIDATION_CHECK_FIELDS
                ):
                    issues.append(f"receipt validation checks[{index}] is malformed")
                    continue
                name = check.get("name")
                if isinstance(name, str):
                    names.append(name)
                command = check.get("command")
                command_is_valid = (
                    isinstance(command, list)
                    and bool(command)
                    and len(command) <= LEGACY_CLEANUP_MAX_COMMAND_TOKENS
                    and all(
                        _valid_bounded_string(
                            token,
                            max_chars=LEGACY_CLEANUP_MAX_COMMAND_TOKEN_CHARS,
                        )
                        for token in command
                    )
                )
                target_count = check.get("target_count")
                if (
                    not command_is_valid
                    or type(target_count) is not int
                    or not 0 <= target_count <= 1_000_000
                    or not _valid_sha256(check.get("target_list_sha256"))
                    or check.get("exit_code") != 0
                    or not _valid_sha256(check.get("output_sha256"))
                ):
                    issues.append(
                        f"receipt validation checks[{index}] lacks executed evidence"
                    )
            if names != list(LEGACY_CLEANUP_VALIDATION_CHECKS):
                issues.append(
                    "receipt validation checks are not the exact ordered matrix"
                )

    if payload.get("provenance_assertions") != LEGACY_CLEANUP_PROVENANCE_ASSERTIONS:
        issues.append("receipt must disclaim generated/import/run provenance")

    groups = payload.get("groups")
    primary_paths: list[Path] = []
    if (
        not isinstance(groups, list)
        or not groups
        or len(groups) > LEGACY_CLEANUP_MAX_GROUPS
    ):
        issues.append("receipt groups must contain 1..64 entries")
    else:
        for index, group in enumerate(groups):
            if not isinstance(group, dict) or set(group) != {"primary", "companion"}:
                issues.append(f"receipt groups[{index}] is malformed")
                continue
            primary_value = group.get("primary")
            primary_text = (
                primary_value.get("path") if isinstance(primary_value, dict) else None
            )
            try:
                primary = canonical_primary_path(primary_text)
            except LegacyCleanupReceiptError as exc:
                issues.append(f"receipt groups[{index}] {exc}")
                continue
            primary_paths.append(primary)
            for key, expected in (
                ("primary", primary),
                ("companion", companion_path(primary)),
            ):
                issue = _file_evidence_issue(
                    group.get(key),
                    object_format=object_format,
                    expected_path=expected,
                    deletion=True,
                )
                if issue:
                    issues.append(f"receipt groups[{index}].{key} {issue}")
        if primary_paths != sorted(primary_paths, key=Path.as_posix) or len(
            set(primary_paths)
        ) != len(primary_paths):
            issues.append("receipt primary groups are not unique and sorted")

    identity = payload.get("receipt_identity_sha256")
    if not _valid_sha256(identity) or identity != receipt_identity_sha256(payload):
        issues.append("receipt semantic identity is stale")

    signature = payload.get("signature")
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "key_id",
        "domain",
        "value",
    }:
        issues.append("receipt signature envelope is malformed")
    else:
        if signature.get("algorithm") != LEGACY_CLEANUP_SIGNATURE_ALGORITHM:
            issues.append("receipt signature algorithm is invalid")
        if signature.get("domain") != LEGACY_CLEANUP_SIGNATURE_DOMAIN:
            issues.append("receipt signature domain is invalid")
        key_id = signature.get("key_id")
        if (
            not isinstance(key_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", key_id) is None
        ):
            issues.append("receipt signature key ID is invalid")
        value = signature.get("value")
        try:
            raw_signature = (
                b64decode(value.encode("ascii"), validate=True)
                if isinstance(value, str)
                else b""
            )
        except (BinasciiError, UnicodeEncodeError):
            raw_signature = b""
        if len(raw_signature) != 64:
            issues.append("receipt signature value is invalid")
    return issues


def parse_receipt_bytes(
    raw: bytes,
    *,
    expected_path: str | Path | None = None,
) -> dict[str, Any]:
    """Parse exact persisted bytes and enforce every pure v1 invariant."""

    payload = decode_strict_json_object(raw, label="legacy cleanup receipt")
    if raw != canonical_receipt_bytes(payload):
        raise LegacyCleanupReceiptError(
            "legacy cleanup receipt is not canonical persisted JSON"
        )
    issues = receipt_structure_issues(payload)
    if expected_path is not None:
        try:
            expected_text = _canonical_path_text(
                expected_path,
                label="legacy cleanup receipt path",
            )
        except LegacyCleanupReceiptError as exc:
            issues.append(str(exc))
        else:
            identity = payload.get("receipt_identity_sha256")
            if (
                not isinstance(identity, str)
                or expected_text != receipt_path(identity).as_posix()
            ):
                issues.append("legacy cleanup receipt path does not match its identity")
    if issues:
        raise LegacyCleanupReceiptError("; ".join(issues))
    return payload


def deleted_paths(payload: Mapping[str, object]) -> tuple[str, ...]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return ()
    result: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            return ()
        for key in ("primary", "companion"):
            record = group.get(key)
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                return ()
            result.append(record["path"])
    return tuple(result)
