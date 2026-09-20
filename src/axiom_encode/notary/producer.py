"""Host-side lineage construction from a completed, isolated encoder run.

Only the root-owned runtime controller calls these functions. Its generation
endpoint accepts a citation, never output files or an arbitrary signing body.
Correction requests are a separate, explicit actor-signed exception and still
need a distinct hardware correction-review signature before admission.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._schema import digest, relative_path
from .canonical import jcs_dumps, sha256_hex
from .identity import IdentityRefusal
from .lineage import (
    CORRECTION,
    GENERATION,
    POLICY_PATH,
    STORE_PREFIX,
    parse_path_policy,
    parse_record,
)
from .producers import Enrollment
from .refusal import Refusal
from .signer import _signed_sidecar
from .verification import Snapshot


def key_fingerprint(key):
    return sha256_hex(
        key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )


def observed_transitions(
    base: Snapshot, outputs: dict[str, bytes | None], *, lane: str
):
    policy = parse_path_policy(base.blobs.get(POLICY_PATH, b""), lane=lane)
    if isinstance(policy, Refusal) or not outputs or len(outputs) > 1000:
        raise IdentityRefusal("producer_output_policy")
    entries = {path: (mode, digest) for path, mode, digest in base.manifest}
    transitions, changed = [], {}
    for path, after in sorted(outputs.items(), key=lambda item: item[0].encode()):
        if (
            not relative_path(path)
            or path.startswith("/")
            or "\\" in path
            or not policy.protects(path)
            or not path.endswith((".yaml", ".yml"))
        ):
            raise IdentityRefusal("producer_output_path")
        before_mode, before_digest = entries.get(path, (None, None))
        if before_mode not in {None, "100644"} or (
            after is not None
            and (not isinstance(after, bytes) or len(after) > 32_000_000)
        ):
            raise IdentityRefusal("producer_output_mode_or_size")
        after_digest = sha256_hex(after) if after is not None else None
        if before_digest == after_digest:
            continue
        transitions.append(
            {
                "path": path,
                "before_blob_sha256": before_digest,
                "before_mode": before_mode,
                "after_blob_sha256": after_digest,
                "after_mode": "100644" if after is not None else None,
                "patch_note_sha256": None,
            }
        )
        changed[path] = after
    if not transitions:
        raise IdentityRefusal("producer_no_change")
    return transitions, changed


def _export(key, body, changed, *, role, base, run_id):
    if not isinstance(key, Ed25519PrivateKey):
        raise IdentityRefusal("producer_key_type")
    raw = jcs_dumps(body)
    if parse_record(raw) is None:
        raise IdentityRefusal("producer_record_metadata")
    address = sha256_hex(raw)
    evidence = {
        STORE_PREFIX + address + ".json": raw,
        STORE_PREFIX + address + f".json.{role}.sig": _signed_sidecar(key, raw, role),
    }
    return jcs_dumps(
        {
            "schema": "axiom/producer-export/v1",
            "lane": body["lane"],
            "epoch_sha256": body["epoch_sha256"],
            "base_commit_git_oid": base.commit,
            "run_id": run_id,
            "record_sha256": address,
            "files": [
                {
                    "path": path,
                    "mode": "100644" if value is not None else None,
                    "base64": base64.b64encode(value).decode()
                    if value is not None
                    else None,
                }
                for path, value in sorted((changed | evidence).items())
            ],
        }
    )


def generation_export(
    key,
    *,
    enrollment: Enrollment,
    base: Snapshot,
    lane: str,
    epoch: str,
    run_id: str,
    outputs: dict[str, bytes],
    model: str,
    prompts: list[str],
    source_id: str,
    source_bytes: bytes,
    draw_set_id: str,
    sampling: dict,
    independence: dict,
    references: dict,
):
    if key_fingerprint(key) != enrollment.body["producer_spki_sha256"]:
        raise IdentityRefusal("producer_enrolled_key")
    transitions, changed = observed_transitions(base, outputs, lane=lane)
    body = {
        "schema": GENERATION,
        "lane": lane,
        "epoch_sha256": epoch,
        "runtime_identity": enrollment.runtime_identity,
        "model": model,
        "cli_version": enrollment.body["codex_cli"]["version"],
        "cli_sha256": enrollment.body["codex_cli"]["sha256"],
        "prompt_sha256s": sorted(set(prompts)),
        "emitted_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "draw_set_id": draw_set_id,
        "sampling": sampling,
        "independence": independence,
        "source_capture": {
            "id": source_id,
            "content_sha256": sha256_hex(source_bytes),
            "oracles": references["oracles"],
            "reference_data": references["reference_data"],
        },
        "transitions": transitions,
    }
    return _export(key, body, changed, role="producer", base=base, run_id=run_id)


def correction_export(
    key,
    *,
    enrollment: Enrollment,
    base: Snapshot,
    lane: str,
    epoch: str,
    run_id: str,
    outputs: dict[str, bytes | None],
    github_user_id: str,
    reason: str,
    predecessor: str | None,
):
    if (
        key_fingerprint(key) != enrollment.body["actor_spki_sha256"]
        or github_user_id not in enrollment.body["github_user_ids"]
        or not isinstance(reason, str)
        or not reason.strip()
        or (predecessor is not None and not digest(predecessor))
    ):
        raise IdentityRefusal("actor_enrolled_identity_or_reason")
    if (
        predecessor is not None
        and STORE_PREFIX + predecessor + ".json" not in base.blobs
    ):
        raise IdentityRefusal("actor_predecessor_missing")
    transitions, changed = observed_transitions(base, outputs, lane=lane)
    body = {
        "schema": CORRECTION,
        "lane": lane,
        "epoch_sha256": epoch,
        "actor": "github:" + github_user_id,
        "reason": reason,
        "predecessor_record_sha256": predecessor,
        "transitions": transitions,
    }
    return _export(key, body, changed, role="actor", base=base, run_id=run_id)
