"""Read immutable producer-owned review guidance, without granting source authority.

The protected caller must supply independently authenticated producer and source
identities. This module does not load keys, mint evidence, sign, or invoke models.
It is intentionally disconnected until generation and artifact verification both
bind those identities through their existing protected paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_GUIDANCE_BYTES = 512 * 1024
MAX_DESCRIPTOR_BYTES = 4096
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class ProducerReviewGuidance:
    producer_commit: str
    path: str
    blob_sha256: str
    size_bytes: int
    citation: str
    source_sha256: str
    content: bytes
    content_sha256: str


def _normalized_text(value: object, *, terminal_lf: bool = False) -> str:
    if type(value) is not str or not value or value.startswith("\ufeff"):
        raise ValueError("guidance text must be nonempty UTF-8 text without BOM")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("guidance text is not valid UTF-8") from exc
    if any((ord(c) < 32 and c not in "\n\t") or ord(c) == 127 for c in value):
        raise ValueError("guidance text contains a prohibited control character")
    if not terminal_lf and value.endswith("\n"):
        raise ValueError("finding must have no terminal newline")
    if not value.strip():
        raise ValueError("guidance text is empty")
    return value


def _strict_json(raw: bytes, limit: int) -> dict:
    if not 0 < len(raw) <= limit:
        raise ValueError("guidance JSON size is outside the allowed bounds")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate guidance JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite guidance JSON value")

    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("nonfinite guidance JSON value")
        return value

    try:
        text = raw.decode("utf-8", errors="strict")
        _normalized_text(text, terminal_lf=True)
        value = json.loads(
            text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid guidance JSON") from exc
    if type(value) is not dict:
        raise ValueError("guidance JSON must be an object")
    return value


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ALLOW_PROTOCOL": "",
        },
    )
    if result.returncode:
        raise ValueError("cannot read guidance from the immutable producer tree")
    return result.stdout


def _regular_blob(repo: Path, commit: str, path: str, size: int) -> bytes:
    # Use the protected Git broker's existing tree-size grammar. In particular,
    # cat-file -s and global options before -C are not admitted by that broker.
    parts = PurePosixPath(path).parts
    prefixes = {"/".join(parts[:index]).encode("ascii") for index in range(1, len(parts) + 1)}
    entries = {}
    listing = _git(repo, "ls-tree", "-r", "-t", "-l", "-z", "--full-tree", commit)
    if len(listing) > 8 * 1024 * 1024 or listing.count(b"\0") > 100_000:
        raise ValueError("guidance tree inventory exceeds metadata limit")
    if not listing.endswith(b"\0"):
        raise ValueError("guidance tree listing is malformed")
    for row in listing[:-1].split(b"\0"):
        metadata, actual_path = row.split(b"\t", 1)
        if actual_path not in prefixes:
            continue
        if actual_path in entries:
            raise ValueError("guidance path is ambiguous")
        fields = metadata.split()
        if len(fields) != 4:
            raise ValueError("guidance tree metadata is malformed")
        entries[actual_path] = fields
    for index in range(1, len(parts) + 1):
        prefix = "/".join(parts[:index]).encode("ascii")
        if prefix not in entries:
            raise ValueError("guidance path is missing")
        mode, kind, oid, object_size = entries[prefix]
        final = index == len(parts)
        if (mode, kind) != ((b"100644", b"blob") if final else (b"040000", b"tree")):
            raise ValueError("guidance must be a regular non-executable Git blob")
        if not _SHA40.fullmatch(oid.decode("ascii")):
            raise ValueError("guidance tree object identity is malformed")
        if not final and object_size != b"-":
            raise ValueError("guidance tree size is malformed")
        if final and (not object_size.isdigit() or int(object_size) != size):
            raise ValueError("guidance Git blob size mismatch")
    object_id = oid.decode("ascii")
    raw = _git(repo, "cat-file", "blob", object_id)
    if len(raw) != size:
        raise ValueError("guidance actual blob size mismatch")
    return raw


GUIDANCE_INPUT_NAMESPACE = "axiom-producer-review-guidance-ref:"
GUIDANCE_INPUT_PREFIX = GUIDANCE_INPUT_NAMESPACE + "v1:"


def parse_producer_guidance_descriptor(descriptor_text: str) -> dict:
    """Validate only the closed descriptor shape; perform no Git or source reads."""
    if type(descriptor_text) is not str:
        raise ValueError("guidance descriptor must be text")
    try:
        descriptor = _strict_json(descriptor_text.encode("utf-8"), MAX_DESCRIPTOR_BYTES)
    except UnicodeError as exc:
        raise ValueError("invalid descriptor encoding") from exc
    if set(descriptor) != {"schema", "producer_commit", "path", "sha256", "size_bytes"}:
        raise ValueError("unexpected guidance descriptor fields")
    if descriptor["schema"] != "axiom-encode/producer-review-finding-ref/v1":
        raise ValueError("unsupported guidance descriptor schema")
    if type(descriptor["producer_commit"]) is not str or not _SHA40.fullmatch(descriptor["producer_commit"]):
        raise ValueError("guidance producer must be an immutable commit")
    path = descriptor["path"]
    if (
        type(path) is not str
        or len(path) > 256
        or not path.startswith("review-guidance/")
        or not path.endswith(".json")
        or any(not _COMPONENT.fullmatch(part) for part in path.split("/"))
        or str(PurePosixPath(path)) != path
    ):
        raise ValueError("invalid producer guidance path")
    size = descriptor["size_bytes"]
    digest = descriptor["sha256"]
    if type(size) is not int or not 0 < size <= MAX_GUIDANCE_BYTES:
        raise ValueError("invalid guidance size")
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise ValueError("invalid guidance digest")
    return descriptor


def producer_guidance_descriptor_from_review_finding(value: str) -> str | None:
    """Return an explicit tagged descriptor, or leave ordinary inline text alone."""
    if type(value) is not str:
        raise ValueError("review finding input must be text")
    if not value.startswith(GUIDANCE_INPUT_NAMESPACE):
        return None
    if not value.startswith(GUIDANCE_INPUT_PREFIX):
        raise ValueError("unsupported producer guidance input version")
    descriptor_text = value[len(GUIDANCE_INPUT_PREFIX):]
    parse_producer_guidance_descriptor(descriptor_text)
    return descriptor_text


def read_producer_review_guidance(
    descriptor_text: str,
    *,
    encoder_repo: Path,
    expected_producer_sha: str,
    citation: str,
    authenticated_source_sha256: str,
) -> ProducerReviewGuidance:
    """Bind a producer blob to identities authenticated by the protected caller.

    An arbitrary caller passing strings is not a substitute for that caller's
    source-release/signature checks. The protected caller and artifact guard independently establish these identities.
    """
    descriptor = parse_producer_guidance_descriptor(descriptor_text)
    if type(expected_producer_sha) is not str or not _SHA40.fullmatch(
        expected_producer_sha
    ):
        raise ValueError("expected producer must be an immutable commit")
    if descriptor["producer_commit"] != expected_producer_sha:
        raise ValueError("guidance producer identity mismatch")
    if (
        _git(encoder_repo, "rev-parse", "HEAD").strip().decode("ascii")
        != expected_producer_sha
    ):
        raise ValueError("guidance checkout is not the expected producer")
    path = descriptor["path"]
    size = descriptor["size_bytes"]
    digest = descriptor["sha256"]
    raw = _regular_blob(encoder_repo, expected_producer_sha, path, size)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("guidance blob digest mismatch")
    envelope = _strict_json(raw, MAX_GUIDANCE_BYTES)
    if set(envelope) != {"schema", "citation", "source_sha256", "finding"}:
        raise ValueError("unexpected guidance envelope fields")
    if envelope["schema"] != "axiom-encode/producer-review-guidance/v1":
        raise ValueError("unsupported guidance envelope schema")
    if type(citation) is not str or not citation or envelope["citation"] != citation:
        raise ValueError("guidance citation mismatch")
    if (
        type(authenticated_source_sha256) is not str
        or not _SHA256.fullmatch(authenticated_source_sha256)
        or envelope["source_sha256"] != authenticated_source_sha256
    ):
        raise ValueError("guidance authenticated source mismatch")
    content = (_normalized_text(envelope["finding"]) + "\n").encode("utf-8")
    if len(content) > MAX_GUIDANCE_BYTES:
        raise ValueError("rendered guidance exceeds limit")
    return ProducerReviewGuidance(
        expected_producer_sha,
        path,
        digest,
        size,
        citation,
        authenticated_source_sha256,
        content,
        hashlib.sha256(content).hexdigest(),
    )


@dataclass(frozen=True)
class ProducerGuidanceVerificationRequest:
    descriptor_text: str
    encoder_checkout: Path
    producer_commit: str
    citation: str
    context_root: Path


def verify_guidance_for_verified_manifest(
    request: ProducerGuidanceVerificationRequest,
    *,
    manifest_path: str,
    manifest_sha256: str,
    verified_payload: dict,
) -> dict:
    """Bind guidance using a payload already authenticated by the existing guard.

    This function does not verify signatures. Only the guard's successful
    verified-loader return may supply this payload and its exact-byte digest.
    The caller must suppress this record if any other guard check fails.
    """
    from .corpus_resolver import read_bounded_regular_file

    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        raise ValueError("verified guidance manifest digest is missing")
    if (
        verified_payload.get("tool") != "axiom-encode encode --apply"
        or verified_payload.get("citation") != request.citation
        or verified_payload.get("backend") != "openai"
    ):
        raise ValueError("guidance requires the exact ordinary model-apply manifest")
    producer = verified_payload.get("axiom_encode_git")
    if not isinstance(producer, dict) or producer.get("commit") != request.producer_commit:
        raise ValueError("guidance signed producer does not match the request")
    source = verified_payload.get("source_attestation")
    if not isinstance(source, dict) or any(
        source.get(key) != request.citation
        for key in ("requested_corpus_citation_path", "resolved_corpus_citation_path")
    ):
        raise ValueError("guidance signed source identity is not exact")
    guidance = read_producer_review_guidance(
        request.descriptor_text,
        encoder_repo=request.encoder_checkout,
        expected_producer_sha=request.producer_commit,
        citation=request.citation,
        authenticated_source_sha256=source.get("generation_input_sha256"),
    )
    context_path = verified_payload.get("context_manifest_file")
    context_digest = verified_payload.get("context_manifest_sha256")
    if not isinstance(context_path, str) or not context_path or not isinstance(context_digest, str):
        raise ValueError("guidance signed context binding is missing")
    context_bytes = read_bounded_regular_file(
        request.context_root, Path(context_path), label="verified guidance context",
        max_bytes=16 * 1024 * 1024,
    )
    if hashlib.sha256(context_bytes).hexdigest() != context_digest:
        raise ValueError("guidance signed context digest mismatch")
    context = _strict_json(context_bytes, 16 * 1024 * 1024)
    expected_binding = {
        "producer_commit": guidance.producer_commit,
        "path": guidance.path,
        "blob_sha256": guidance.blob_sha256,
        "size_bytes": guidance.size_bytes,
        "citation": guidance.citation,
        "source_sha256": guidance.source_sha256,
        "content_sha256": guidance.content_sha256,
    }
    if context.get("citation") != request.citation or context.get("producer_review_guidance") != expected_binding:
        raise ValueError("guidance context identity does not match the immutable producer blob")
    findings = context.get("review_findings_files")
    if (
        not isinstance(findings, list) or len(findings) != 1
        or not isinstance(findings[0], dict)
        or findings[0].get("content") != guidance.content.decode("utf-8")
        or findings[0].get("sha256") != guidance.content_sha256
    ):
        raise ValueError("guidance context does not contain the exact mandatory finding")
    return {
        "schema": "axiom-encode/verified-producer-guidance/v1",
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "context_manifest_sha256": context_digest,
        **expected_binding,
    }


def validate_workflow_guidance_input(inputs: dict, *, producer_commit: str) -> str | None:
    """Parse the original dispatch inputs without changing ordinary inline mode."""
    for field in ("dependent_review_finding", "second_dependent_review_finding"):
        value = inputs.get(field, "")
        if isinstance(value, str) and value.startswith(GUIDANCE_INPUT_NAMESPACE):
            raise ValueError("producer guidance references are primary-only")
    descriptor_text = producer_guidance_descriptor_from_review_finding(inputs.get("review_finding", ""))
    if descriptor_text is None:
        return None
    from .corpus_resolver import require_canonical_corpus_citation_path
    from .prepare_signed_backfill import split_atomic_source_input

    descriptor = parse_producer_guidance_descriptor(descriptor_text)
    if descriptor["producer_commit"] != producer_commit:
        raise ValueError("guidance descriptor must name this workflow producer commit")
    require_canonical_corpus_citation_path(inputs.get("citation", ""))
    forbidden = (
        "queue_id", "queue_item_id", "queue_manifest_sha256", "queue_item_generation_sha256",
        "queue_dispatcher_run_id", "repair_run_id", "replace_legacy_rulespec_path",
        "legacy_exact_dependent_rulespec_path", "second_legacy_exact_dependent_rulespec_path",
        "dependent_citation", "second_dependent_citation", "dependent_review_finding",
        "second_dependent_review_finding",
    )
    if any(inputs.get(key) not in (None, "") for key in forbidden):
        raise ValueError("producer guidance cannot mix queue, repair, legacy or dependent lanes")
    for key in ("existing_signed_imports_json", "legacy_retained_successor_rulespec_paths_json"):
        if json.loads(inputs.get(key) or "[]") != []:
            raise ValueError("producer guidance cannot add import or successor lanes")
    atomic = split_atomic_source_input(inputs.get("source_bundle_json") or "[]")
    if (
        atomic["source_bundle"] != [] or atomic["canonical_refresh_bundle"] != []
        or atomic["require_complete_source_unit"] is not True
        or atomic.get("manifest_only_refresh", False)
        or atomic.get("reviewed_candidate_promotion", False)
        or atomic.get("repair_mode") is not None
    ):
        raise ValueError("producer guidance requires an ordinary single full-source target")
    return descriptor_text


def verify_packaged_producer_guidance(
    *,
    original_inputs: dict,
    producer_commit: str,
    encoder_checkout: Path,
    context_root: Path,
    manifest_path: str,
    manifest_bytes: bytes,
    guard_payload: dict,
) -> str:
    """Rebind original event guidance to the exact bytes verified by the guard.

    The caller supplies the run-local result of the protected guard, not an
    artifact's claimed status. This adds no signature or source authority.
    """
    descriptor = validate_workflow_guidance_input(original_inputs, producer_commit=producer_commit)
    if descriptor is None:
        raise ValueError("packaged guidance requires the original tagged event input")
    if guard_payload.get("passed") is not True or guard_payload.get("issues") != []:
        raise ValueError("packaged guidance requires the successful complete protected guard")
    record = guard_payload.get("producer_review_guidance")
    if not isinstance(record, dict) or (
        record.get("manifest_path") != manifest_path
        or record.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("packaged manifest differs from the verified guard bytes")
    payload = _strict_json(manifest_bytes, 1024 * 1024)
    request = ProducerGuidanceVerificationRequest(
        descriptor, encoder_checkout, producer_commit, original_inputs["citation"], context_root,
    )
    rebound = verify_guidance_for_verified_manifest(
        request, manifest_path=manifest_path,
        manifest_sha256=record["manifest_sha256"], verified_payload=payload,
    )
    if rebound != record:
        raise ValueError("packaged guidance binding differs from the protected guard record")
    guidance = read_producer_review_guidance(
        descriptor, encoder_repo=encoder_checkout, expected_producer_sha=producer_commit,
        citation=request.citation, authenticated_source_sha256=record["source_sha256"],
    )
    return guidance.content.decode("utf-8")
