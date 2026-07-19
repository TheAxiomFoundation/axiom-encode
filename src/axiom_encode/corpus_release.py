"""Strict consumer verification for signed axiom-corpus release objects."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from base64 import b64decode
from binascii import Error as BinasciiError
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

RELEASE_OBJECT_SCHEMA_V2 = "axiom-corpus/release-object/v2"
RELEASE_OBJECT_SCHEMA_VERSION = "axiom-corpus/release-object/v3"
RELEASE_OBJECT_SIGNATURE_ALGORITHM = "ed25519"
RELEASE_OBJECT_SIGNATURE_KEY_ID = "axiom-corpus-release-v2"
RELEASE_OBJECT_PUBLIC_KEY_ENV = "AXIOM_CORPUS_RELEASE_PUBLIC_KEY"
COMPLETE_EXPRESSION_DATES_PROFILE = "complete-expression-dates-v1"

_ARTIFACT_CLASSES = ("inventory", "provisions", "coverage", "sources")
_DOCUMENT_CLASSES = {
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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SCOPE_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,255}$")


class CorpusReleaseObjectError(ValueError):
    """A corpus release object is malformed, untrusted, or inconsistent."""


@dataclass(frozen=True, slots=True)
class VerifiedReleaseArtifact:
    """One local artifact authorized by a verified release object."""

    artifact_class: str
    path: str
    sha256: str
    byte_count: int
    row_count: int | None


@dataclass(frozen=True, slots=True)
class VerifiedReleaseScope:
    """One active corpus scope authorized by a verified release object."""

    jurisdiction: str
    document_class: str
    version: str
    provision_rows: int
    provision_projection_sha256: str
    navigation_projection_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedCorpusReleaseObject:
    """The immutable identities and local inventory from a release object."""

    name: str
    content_sha256: str
    selector_sha256: str
    scopes: tuple[VerifiedReleaseScope, ...]
    artifacts: tuple[VerifiedReleaseArtifact, ...]


def canonical_release_object_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return the signed canonical JSON representation of a release object."""

    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def verify_release_object(
    payload: Mapping[str, Any],
    *,
    public_key: str,
) -> VerifiedCorpusReleaseObject:
    """Verify a supported canonical axiom-corpus contract and signature."""

    materialized = copy.deepcopy(dict(payload))
    verified = _validate_unsigned_release_object(materialized)
    signature = materialized.get("signature")
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "key_id",
        "value",
    }:
        raise CorpusReleaseObjectError(
            "release object signature does not match the v2 schema"
        )
    if signature.get("algorithm") != RELEASE_OBJECT_SIGNATURE_ALGORITHM:
        raise CorpusReleaseObjectError(
            "release object uses an unsupported signature algorithm"
        )
    if signature.get("key_id") != RELEASE_OBJECT_SIGNATURE_KEY_ID:
        raise CorpusReleaseObjectError("release object uses an unknown signing key")
    encoded = signature.get("value")
    if not isinstance(encoded, str):
        raise CorpusReleaseObjectError("release object signature value is missing")
    try:
        raw_signature = b64decode(encoded.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError) as exc:
        raise CorpusReleaseObjectError(
            "release object signature encoding is invalid"
        ) from exc
    if len(raw_signature) != 64:
        raise CorpusReleaseObjectError("release object signature has invalid length")
    try:
        _load_ed25519_public_key(public_key).verify(
            raw_signature,
            canonical_release_object_bytes(materialized),
        )
    except InvalidSignature as exc:
        raise CorpusReleaseObjectError("release object signature is invalid") from exc
    return verified


def _validate_unsigned_release_object(
    payload: Mapping[str, Any],
) -> VerifiedCorpusReleaseObject:
    allowed = {"schema_version", "release", "content_sha256", "content", "signature"}
    if set(payload) - allowed:
        raise CorpusReleaseObjectError(
            "release object has unsupported top-level fields"
        )
    schema_version = payload.get("schema_version")
    if schema_version not in {
        RELEASE_OBJECT_SCHEMA_V2,
        RELEASE_OBJECT_SCHEMA_VERSION,
    }:
        raise CorpusReleaseObjectError(
            "release object uses an unsupported schema version"
        )
    release = _validate_release_name(payload.get("release"))
    content = payload.get("content")
    expected_content_fields = {
        "release",
        "created_at",
        "selector_sha256",
        "corpus_base",
        "git",
        "r2",
        "scopes",
        "artifacts",
        "validation",
    }
    if schema_version == RELEASE_OBJECT_SCHEMA_VERSION:
        expected_content_fields.add("quality_profile")
    if not isinstance(content, dict) or set(content) != expected_content_fields:
        raise CorpusReleaseObjectError(
            f"release object content does not match the {schema_version} schema"
        )
    quality_profile = content.get("quality_profile")
    if (
        schema_version == RELEASE_OBJECT_SCHEMA_VERSION
        and quality_profile != COMPLETE_EXPRESSION_DATES_PROFILE
    ):
        raise CorpusReleaseObjectError(
            "release object has an unsupported quality profile"
        )
    if content.get("release") != release:
        raise CorpusReleaseObjectError("release object name does not match its content")
    if not isinstance(content.get("created_at"), str) or not content["created_at"]:
        raise CorpusReleaseObjectError("release object has an invalid creation time")
    git = content.get("git")
    if not isinstance(git, dict) or set(git) != {"commit", "committed_at"}:
        raise CorpusReleaseObjectError("release object has invalid git provenance")
    if (
        not isinstance(git.get("commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", git["commit"]) is None
        or not isinstance(git.get("committed_at"), str)
        or not git["committed_at"]
    ):
        raise CorpusReleaseObjectError("release object has invalid git provenance")
    expected_content_sha256 = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    if payload.get("content_sha256") != expected_content_sha256:
        raise CorpusReleaseObjectError("release object content sha256 does not match")
    if content.get("corpus_base") != "data/corpus":
        raise CorpusReleaseObjectError(
            "release object uses a non-canonical corpus base"
        )
    r2 = content.get("r2")
    if (
        not isinstance(r2, dict)
        or set(r2) != {"bucket", "addressing"}
        or not isinstance(r2.get("bucket"), str)
        or not r2["bucket"]
        or r2.get("addressing") != "sha256"
    ):
        raise CorpusReleaseObjectError(
            "release object has an invalid R2 content boundary"
        )

    raw_scopes = content.get("scopes")
    raw_artifacts = content.get("artifacts")
    if not isinstance(raw_scopes, list) or not raw_scopes:
        raise CorpusReleaseObjectError("release object must contain at least one scope")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise CorpusReleaseObjectError("release object must contain artifact entries")
    scopes = _validate_scopes(raw_scopes)
    selector_payload = {
        "name": release,
        "scopes": [
            {
                "jurisdiction": scope.jurisdiction,
                "document_class": scope.document_class,
                "version": scope.version,
            }
            for scope in scopes
        ],
    }
    if schema_version == RELEASE_OBJECT_SCHEMA_VERSION:
        selector_payload["quality_profile"] = quality_profile
    selector_sha256 = content.get("selector_sha256")
    if (
        not isinstance(selector_sha256, str)
        or _SHA256_RE.fullmatch(selector_sha256) is None
        or selector_sha256
        != hashlib.sha256(_canonical_json_bytes(selector_payload)).hexdigest()
    ):
        raise CorpusReleaseObjectError(
            "release selector sha256 does not match its scopes"
        )
    artifacts = _validate_artifacts(raw_artifacts, bucket=r2["bucket"])
    _validate_scope_artifact_membership(scopes, artifacts)
    _validate_validation_attestation(
        content.get("validation"),
        scopes=scopes,
        artifacts=artifacts,
        bucket=r2["bucket"],
        quality_profile=(
            str(quality_profile)
            if schema_version == RELEASE_OBJECT_SCHEMA_VERSION
            else None
        ),
    )
    return VerifiedCorpusReleaseObject(
        name=release,
        content_sha256=expected_content_sha256,
        selector_sha256=selector_sha256,
        scopes=scopes,
        artifacts=artifacts,
    )


def _validate_scopes(raw_scopes: Sequence[Any]) -> tuple[VerifiedReleaseScope, ...]:
    scopes: list[VerifiedReleaseScope] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_scopes:
        if not isinstance(raw, dict) or set(raw) != {
            "jurisdiction",
            "document_class",
            "version",
            "provision_rows",
            "navigation_rows",
            "provision_projection_sha256",
            "navigation_projection_sha256",
        }:
            raise CorpusReleaseObjectError(
                "release object scope does not match the v2 schema"
            )
        identity = tuple(
            str(raw.get(field) or "")
            for field in ("jurisdiction", "document_class", "version")
        )
        if (
            any(_SCOPE_COMPONENT_RE.fullmatch(item) is None for item in identity)
            or identity[1] not in _DOCUMENT_CLASSES
            or identity in seen
        ):
            raise CorpusReleaseObjectError(
                "release object contains an invalid or duplicate scope"
            )
        seen.add(identity)
        rows = raw.get("provision_rows")
        navigation_rows = raw.get("navigation_rows")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows <= 0
            or not isinstance(navigation_rows, int)
            or isinstance(navigation_rows, bool)
            or navigation_rows != rows
        ):
            raise CorpusReleaseObjectError(
                "release object scope has inconsistent row counts"
            )
        provision_projection_sha256 = raw.get("provision_projection_sha256")
        navigation_projection_sha256 = raw.get("navigation_projection_sha256")
        if (
            not isinstance(provision_projection_sha256, str)
            or _SHA256_RE.fullmatch(provision_projection_sha256) is None
            or not isinstance(navigation_projection_sha256, str)
            or _SHA256_RE.fullmatch(navigation_projection_sha256) is None
        ):
            raise CorpusReleaseObjectError(
                "release object scope has invalid projection sha256 evidence"
            )
        scopes.append(
            VerifiedReleaseScope(
                identity[0],
                identity[1],
                identity[2],
                rows,
                provision_projection_sha256,
                navigation_projection_sha256,
            )
        )
    return tuple(scopes)


def _validate_artifacts(
    raw_artifacts: Sequence[Any],
    *,
    bucket: str,
) -> tuple[VerifiedReleaseArtifact, ...]:
    artifacts: list[VerifiedReleaseArtifact] = []
    seen: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise CorpusReleaseObjectError(
                "release object contains a non-object artifact"
            )
        required = {
            "artifact_class",
            "path",
            "sha256",
            "bytes",
            "r2_bucket",
            "r2_key",
        }
        artifact_class = raw.get("artifact_class")
        expected_fields = required | (
            {"rows"} if artifact_class == "provisions" else set()
        )
        if set(raw) != expected_fields:
            raise CorpusReleaseObjectError(
                "release artifact does not match the v2 schema"
            )
        path = raw.get("path")
        digest = raw.get("sha256")
        byte_count = raw.get("bytes")
        if artifact_class not in _ARTIFACT_CLASSES:
            raise CorpusReleaseObjectError("release artifact has an unsupported class")
        if (
            not isinstance(path, str)
            or not path.startswith(f"data/corpus/{artifact_class}/")
            or PathLikeParts(path).unsafe
            or path in seen
        ):
            raise CorpusReleaseObjectError(
                "release artifact path is non-canonical or duplicated"
            )
        seen.add(path)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise CorpusReleaseObjectError(
                f"release artifact has invalid sha256: {path}"
            )
        if raw.get("r2_key") != f"objects/sha256/{digest[:2]}/{digest}":
            raise CorpusReleaseObjectError(
                f"release artifact has a non-content-addressed R2 key: {path}"
            )
        if raw.get("r2_bucket") != bucket:
            raise CorpusReleaseObjectError(
                f"release artifact uses the wrong R2 bucket: {path}"
            )
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise CorpusReleaseObjectError(
                f"release artifact has invalid byte count: {path}"
            )
        row_count = raw.get("rows")
        if artifact_class == "provisions":
            if (
                not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count <= 0
            ):
                raise CorpusReleaseObjectError(
                    f"release provisions artifact has an invalid row count: {path}"
                )
        artifacts.append(
            VerifiedReleaseArtifact(
                artifact_class,
                path,
                digest,
                byte_count,
                row_count if isinstance(row_count, int) else None,
            )
        )
    if [item.path for item in artifacts] != sorted(item.path for item in artifacts):
        raise CorpusReleaseObjectError(
            "release artifacts are not in canonical path order"
        )
    return tuple(artifacts)


def _validate_scope_artifact_membership(
    scopes: Sequence[VerifiedReleaseScope],
    artifacts: Sequence[VerifiedReleaseArtifact],
) -> None:
    by_path = {artifact.path: artifact for artifact in artifacts}
    claimed: set[str] = set()
    for scope in scopes:
        prefix = f"{scope.jurisdiction}/{scope.document_class}/{scope.version}"
        required = {
            "inventory": f"data/corpus/inventory/{prefix}.json",
            "provisions": f"data/corpus/provisions/{prefix}.jsonl",
            "coverage": f"data/corpus/coverage/{prefix}.json",
        }
        for artifact_class, path in required.items():
            artifact = by_path.get(path)
            if artifact is None or artifact.artifact_class != artifact_class:
                raise CorpusReleaseObjectError(
                    f"release scope lacks its {artifact_class} artifact: {prefix}"
                )
            claimed.add(path)
        if by_path[required["provisions"]].row_count != scope.provision_rows:
            raise CorpusReleaseObjectError(
                f"release scope row count does not match its provisions artifact: {prefix}"
            )
        source_prefix = f"data/corpus/sources/{prefix}/"
        source_paths = [
            path
            for path, artifact in by_path.items()
            if path.startswith(source_prefix) and artifact.artifact_class == "sources"
        ]
        if not source_paths:
            raise CorpusReleaseObjectError(
                f"release scope lacks source artifacts: {prefix}"
            )
        claimed.update(source_paths)
    if set(by_path) != claimed:
        raise CorpusReleaseObjectError(
            "release object contains artifacts outside its declared scopes"
        )


def _validate_validation_attestation(
    raw: object,
    *,
    scopes: Sequence[VerifiedReleaseScope],
    artifacts: Sequence[VerifiedReleaseArtifact],
    bucket: str,
    quality_profile: str | None = None,
) -> None:
    expected_fields = {
        "passed",
        "deep_validation",
        "r2_readback",
        "supabase_projection_evidence",
    }
    if quality_profile is not None:
        expected_fields.add("quality_profile")
    if (
        not isinstance(raw, dict)
        or set(raw) != expected_fields
        or raw.get("passed") is not True
    ):
        raise CorpusReleaseObjectError(
            "release validation does not match its object schema"
        )
    if quality_profile is not None and raw.get("quality_profile") != quality_profile:
        raise CorpusReleaseObjectError(
            "release validation quality profile is inconsistent"
        )
    deep = raw.get("deep_validation")
    if not isinstance(deep, dict) or set(deep) != {
        "error_count",
        "warning_count",
        "scope_count",
    }:
        raise CorpusReleaseObjectError(
            "release object deep-validation evidence is inconsistent"
        )
    if (
        not isinstance(deep.get("error_count"), int)
        or isinstance(deep.get("error_count"), bool)
        or deep["error_count"] != 0
        or not isinstance(deep.get("scope_count"), int)
        or isinstance(deep.get("scope_count"), bool)
        or deep["scope_count"] != len(scopes)
        or not isinstance(deep.get("warning_count"), int)
        or isinstance(deep.get("warning_count"), bool)
        or deep["warning_count"] < 0
    ):
        raise CorpusReleaseObjectError(
            "release object deep-validation evidence is inconsistent"
        )
    readback = raw.get("r2_readback")
    expected_keys = [
        f"objects/sha256/{artifact.sha256[:2]}/{artifact.sha256}"
        for artifact in artifacts
    ]
    if (
        not isinstance(readback, dict)
        or set(readback)
        != {"bucket", "artifact_count", "artifact_bytes", "verified_keys"}
        or not isinstance(readback.get("artifact_count"), int)
        or isinstance(readback.get("artifact_count"), bool)
        or not isinstance(readback.get("artifact_bytes"), int)
        or isinstance(readback.get("artifact_bytes"), bool)
        or not isinstance(readback.get("verified_keys"), list)
        or not all(isinstance(key, str) for key in readback["verified_keys"])
        or readback.get("bucket") != bucket
        or readback.get("artifact_count") != len(artifacts)
        or readback.get("artifact_bytes")
        != sum(artifact.byte_count for artifact in artifacts)
        or readback.get("verified_keys") != expected_keys
    ):
        raise CorpusReleaseObjectError(
            "release object R2 readback evidence is inconsistent"
        )
    counts = raw.get("supabase_projection_evidence")
    if not isinstance(counts, list) or len(counts) != len(scopes):
        raise CorpusReleaseObjectError(
            "release object staged-count evidence is incomplete"
        )
    expected_scopes = {
        (scope.jurisdiction, scope.document_class, scope.version): scope
        for scope in scopes
    }
    seen: set[tuple[str, str, str]] = set()
    for item in counts:
        if not isinstance(item, dict) or set(item) != {
            "jurisdiction",
            "document_class",
            "version",
            "expected",
            "actual",
            "expected_navigation",
            "actual_navigation",
            "expected_provision_projection_sha256",
            "actual_provision_projection_sha256",
            "expected_navigation_projection_sha256",
            "actual_navigation_projection_sha256",
        }:
            raise CorpusReleaseObjectError(
                "release staged-count evidence does not match the v2 schema"
            )
        identity = tuple(
            str(item.get(field) or "")
            for field in ("jurisdiction", "document_class", "version")
        )
        if identity in seen or identity not in expected_scopes:
            raise CorpusReleaseObjectError(
                "release object has invalid staged-count identity"
            )
        seen.add(identity)
        scope = expected_scopes[identity]
        expected = scope.provision_rows
        if any(
            not isinstance(item.get(field), int)
            or isinstance(item.get(field), bool)
            or item[field] != expected
            for field in (
                "expected",
                "actual",
                "expected_navigation",
                "actual_navigation",
            )
        ):
            raise CorpusReleaseObjectError(
                "release object staged-count evidence does not match its scope"
            )
        expected_digests = {
            "expected_provision_projection_sha256": scope.provision_projection_sha256,
            "actual_provision_projection_sha256": scope.provision_projection_sha256,
            "expected_navigation_projection_sha256": scope.navigation_projection_sha256,
            "actual_navigation_projection_sha256": scope.navigation_projection_sha256,
        }
        if any(
            not isinstance(item.get(field), str)
            or _SHA256_RE.fullmatch(item[field]) is None
            or item[field] != expected_digest
            for field, expected_digest in expected_digests.items()
        ):
            raise CorpusReleaseObjectError(
                "release object staged projection evidence does not match its scope"
            )


@dataclass(frozen=True, slots=True)
class PathLikeParts:
    """Minimal lexical path check that is independent of host path semantics."""

    value: str

    @property
    def unsafe(self) -> bool:
        parts = self.value.split("/")
        return (
            self.value.startswith("/")
            or "\\" in self.value
            or any(part in {"", ".", ".."} for part in parts)
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_release_name(raw: object) -> str:
    if (
        not isinstance(raw, str)
        or len(raw) > 128
        or _RELEASE_NAME_RE.fullmatch(raw) is None
    ):
        raise CorpusReleaseObjectError("release object has an invalid release name")
    if raw == "current":
        raise CorpusReleaseObjectError(
            "legacy mutable release name 'current' is forbidden"
        )
    return raw


def _load_ed25519_public_key(public_key: str) -> Ed25519PublicKey:
    text = public_key.strip().replace("\\n", "\n")
    if text.startswith("-----BEGIN "):
        try:
            loaded = serialization.load_pem_public_key(text.encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise CorpusReleaseObjectError("release public key PEM is invalid") from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise CorpusReleaseObjectError("release public key must be Ed25519")
        return loaded
    try:
        raw = b64decode(text.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError, ValueError) as exc:
        raise CorpusReleaseObjectError(
            "release public key must be raw base64 or PEM"
        ) from exc
    if len(raw) != 32:
        raise CorpusReleaseObjectError("release public key must decode to 32 bytes")
    return Ed25519PublicKey.from_public_bytes(raw)
