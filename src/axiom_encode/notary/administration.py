"""Typed administrative candidate construction from immutable lane snapshots."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .canonical import jcs_dumps, sha256_hex
from .chain import (
    BOOTSTRAP_PATHS,
    ChainState,
    InvalidChain,
    _registry,
    _require,
    _trust,
    legacy_spki,
)
from .consumer import epoch_template, parse_consumer
from .lineage import POLICY_PATH, parse_path_policy
from .manifest import manifest_diff, manifest_sha256
from .protocol import (
    PREFIX,
    TRANSITION_POLICY_PATH,
    parse_artifact,
    parse_transition_policy,
)
from .refusal import Refusal
from .registry import REGISTRY_PATH
from .verification import Snapshot


def legacy_inventory(
    snapshot: Snapshot,
    *,
    lane: str,
    apply_root: dict,
    expected_encoder_identity: dict,
    local_corpus_release,
) -> list[list[str]]:
    """Reuse the complete existing v5 contract with the frozen public root.

    All files come from verified raw Git blobs, without checkout filters or
    candidate execution. A v5 statement failing any current contract binding
    cannot grandfather even one of its files.
    """
    from cryptography.hazmat.primitives import serialization

    from axiom_encode import cli

    from ._schema import decode_base64

    legacy_spki(apply_root)
    public = serialization.load_der_public_key(
        decode_base64(apply_root["public_key_spki_der_base64"])
    )
    entries = {}
    with tempfile.TemporaryDirectory(prefix="axiom-genesis-v5-") as temporary:
        root = Path(temporary) / lane.split("/")[-1]
        root.mkdir()
        for relative, raw in snapshot.blobs.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        for name in sorted(snapshot.blobs, key=str.encode):
            if "/.axiom/encoding-manifests/" not in "/" + name or not name.endswith(
                ".json"
            ):
                continue
            payload, prefix, address, issues = (
                cli._load_verified_applied_encoding_manifest_payload(
                    root,
                    name,
                    signing_broker=public,
                    expected_encoder_identity=expected_encoder_identity,
                    local_corpus_release=local_corpus_release,
                )
            )
            if payload is None or issues:
                continue
            for item in payload["applied_files"]:
                if item.get("deleted"):
                    continue
                path = cli._prefix_applied_manifest_path(item["path"], prefix)
                if path in entries:
                    raise InvalidChain("duplicate_qualifying_v5_records")
                entries[path] = [path, item["sha256"], address]
    return [entries[path] for path in sorted(entries, key=str.encode)]


def build_genesis(
    snapshot: Snapshot,
    *,
    lane: str,
    notary_repository: str,
    prospective: dict[str, bytes],
    consumer_spec_path: str,
    consumer_template: bytes,
    legacy_apply_root: dict,
    legacy_eval_root: dict,
    frozen_apply_spki_sha256: str,
    frozen_eval_spki_sha256: str,
    ceremony_admin_spki_sha256: str,
    ceremony_notary_spki_sha256: str,
    expected_encoder_identity: dict,
    local_corpus_release,
) -> bytes:
    """Construct the exact candidate after the service verifies the live lane lock.

    Frozen fingerprints and ceremony identities are deployment-approved inputs,
    never supplied by the requesting runner. No signing happens here.
    """
    apply, evaluation = legacy_spki(legacy_apply_root), legacy_spki(legacy_eval_root)
    _require(notary_repository == lane + "-notary", "dedicated_notary_repository")
    _require(
        apply == frozen_apply_spki_sha256 and evaluation == frozen_eval_spki_sha256,
        "witnessed_legacy_root_mismatch",
    )
    registry = _trust(prospective, lane, apply, evaluation)
    _require(
        ceremony_admin_spki_sha256 in registry.keys["admin-approver"]
        and set(registry.keys["notary"]) == {ceremony_notary_spki_sha256},
        "ceremony_registry_mismatch",
    )
    consumer = parse_consumer(consumer_template)
    _require(
        consumer is not None
        and consumer["lane"] == lane
        and consumer["notary_repository"] == notary_repository
        and consumer["epoch_sha256"] == "0" * 64
        and consumer["notary_spki_sha256"] == ceremony_notary_spki_sha256
        and epoch_template(consumer_template, "0" * 64) == consumer_template,
        "consumer_template",
    )
    policy = parse_path_policy(prospective[POLICY_PATH], lane=lane)
    transition_policy = parse_transition_policy(
        prospective[TRANSITION_POLICY_PATH], lane
    )
    _require(
        consumer_spec_path not in BOOTSTRAP_PATHS.values()
        and transition_policy.protects(consumer_spec_path)
        and not policy.protects(consumer_spec_path),
        "consumer_bootstrap_path",
    )
    protected = {
        path: (mode, digest)
        for path, mode, digest in snapshot.manifest
        if policy.protects(path)
    }
    _require(
        all(mode == "100644" for mode, _ in protected.values()),
        "genesis_protected_mode",
    )
    attested = [
        row
        for row in legacy_inventory(
            snapshot,
            lane=lane,
            apply_root=legacy_apply_root,
            expected_encoder_identity=expected_encoder_identity,
            local_corpus_release=local_corpus_release,
        )
        if row[0] in protected
    ]
    _require(
        all(protected[row[0]][1] == row[1] for row in attested),
        "genesis_v5_blob_mismatch",
    )
    attested_paths = {row[0] for row in attested}
    baseline = [
        [path, protected[path][1]]
        for path in sorted(protected, key=str.encode)
        if path not in attested_paths
    ]
    body = {
        "schema": PREFIX + "genesis/v1",
        "lane": lane,
        "genesis_commit_git_oid": snapshot.commit,
        "genesis_tree_manifest_sha256": manifest_sha256(snapshot.manifest),
        "bootstrap_policies": {
            name: sha256_hex(prospective[path])
            for name, path in BOOTSTRAP_PATHS.items()
        },
        "activation_spec_template_sha256": sha256_hex(consumer_template),
        "consumer_spec_path": consumer_spec_path,
        "notary_repository": notary_repository,
        "legacy_apply_root": legacy_apply_root,
        "legacy_eval_root": legacy_eval_root,
        "v5_attested": attested,
        "baseline_unattested": baseline,
    }
    raw = jcs_dumps(body)
    _require(parse_artifact(raw, "genesis") is not None, "genesis_schema")
    return raw


def build_transition(
    base: Snapshot, subject: Snapshot, state: ChainState, *, reason: str
) -> bytes:
    """Recompute the complete administrative delta; reject any ordinary content."""
    _require(
        manifest_sha256(base.manifest) == state.tip.manifest, "transition_stale_base"
    )
    changes = manifest_diff(base.manifest, subject.manifest)
    _require(bool(changes), "state_identical")
    policy = parse_path_policy(state.trust_files[POLICY_PATH], lane=state.anchor.lane)
    transition_policy = parse_transition_policy(
        state.trust_files[TRANSITION_POLICY_PATH], state.anchor.lane
    )
    for change in changes:
        _require(
            transition_policy.protects(change.path)
            and not policy.protects(change.path),
            "transition_nontrust_delta",
        )
    _trust(
        subject.blobs,
        state.anchor.lane,
        state.legacy_apply_spki_sha256,
        state.legacy_eval_spki_sha256,
    )
    successor = parse_path_policy(subject.blobs[POLICY_PATH], lane=state.anchor.lane)
    _require(not isinstance(successor, Refusal), "successor_path_policy")
    if state.activated:
        _require(
            not any(
                successor.protects(path) and not policy.protects(path)
                for path, _, _ in subject.manifest
            ),
            "policy_expansion_requires_v34",
        )
    else:
        allowed = set(BOOTSTRAP_PATHS.values()) | {state.genesis["consumer_spec_path"]}
        _require(all(c.path in allowed for c in changes), "activation_extra_delta")
        for name, path in BOOTSTRAP_PATHS.items():
            _require(
                path in subject.blobs
                and sha256_hex(subject.blobs[path])
                == state.genesis["bootstrap_policies"][name],
                "activation_installed_policy",
            )
        template = epoch_template(
            subject.blobs.get(state.genesis["consumer_spec_path"], b""),
            state.anchor.epoch_sha256,
        )
        _require(
            template is not None
            and sha256_hex(template)
            == state.genesis["activation_spec_template_sha256"],
            "activation_installed_consumer",
        )
    _require(
        all(
            mode == "100644"
            for path, mode, _ in subject.manifest
            if successor.protects(path)
        ),
        "protected_mode",
    )
    # Every key update is validated under predecessor authority by the signer;
    # successor roots only define what takes effect after finalization.
    registry = _registry(
        subject.blobs[REGISTRY_PATH],
        state.anchor.lane,
        state.legacy_apply_spki_sha256,
        state.legacy_eval_spki_sha256,
    )
    consumer = parse_consumer(
        subject.blobs.get(state.genesis["consumer_spec_path"], b"")
    )
    _require(
        consumer is not None
        and consumer["lane"] == state.anchor.lane
        and consumer["epoch_sha256"] == state.anchor.epoch_sha256
        and consumer["notary_repository"] == state.anchor.notary_repository
        and set(registry.keys["notary"]) == {consumer["notary_spki_sha256"]},
        "successor_consumer_binding",
    )
    body = {
        "schema": PREFIX + "transition/v1",
        "lane": state.anchor.lane,
        "epoch_sha256": state.anchor.epoch_sha256,
        "chain_predecessor_sha256": state.tip.address,
        "chain_predecessor_kind": state.tip.kind,
        "base_tree_manifest_sha256": manifest_sha256(base.manifest),
        "subject_tree_manifest_sha256": manifest_sha256(subject.manifest),
        "subject_commit_git_oid": subject.commit,
        "delta": [c._asdict() for c in changes],
        "reason": reason,
    }
    raw = jcs_dumps(body)
    _require(parse_artifact(raw, "transition") is not None, "transition_schema")
    return raw
