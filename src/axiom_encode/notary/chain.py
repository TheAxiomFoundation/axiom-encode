"""Reconstruct the v33 append-only chain; pending artifacts never become a base.

Branch authenticity (the dedicated repository's App-only ruleset and remote
identity) is checked by the service adapter. This module validates its bytes
and ordered history, including every signed bundle and terminal marker.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ._schema import canonical_object, decode_base64, digest, fields
from .canonical import sha256_hex
from .consumer import epoch_template, parse_consumer
from .lineage import POLICY_PATH, parse_path_policy
from .protocol import (
    PREFIX,
    PROFILE_PATH,
    TRANSITION_POLICY_PATH,
    check_gates,
    parse_artifact,
    parse_transition_policy,
)
from .refusal import Refusal
from .registry import REGISTRY_PATH, KeyRegistry, parse_registry
from .signatures import verify_detached
from .verification import Predecessor, Snapshot

BOOTSTRAP_PATHS = {
    "path_policy_sha256": POLICY_PATH,
    "transition_path_policy_sha256": TRANSITION_POLICY_PATH,
    "profile_sha256": PROFILE_PATH,
    "key_registry_sha256": REGISTRY_PATH,
}
_BODY = re.compile(r"([0-9a-f]{64})\.json")
_RAW = re.compile(r"([0-9a-f]{64})\.raw")
_SIDECAR = re.compile(
    r"([0-9a-f]{64})\.json\.(genesis|transition|admin-approver|approver|notary)\.sig"
)


class InvalidChain(ValueError):
    pass


def _require(condition, detail):
    if not condition:
        raise InvalidChain(detail)


@dataclass(frozen=True)
class Anchor:
    lane: str
    epoch_sha256: str
    notary_repository: str
    notary_spki_sha256: str


@dataclass(frozen=True)
class Artifact:
    address: str
    kind: str
    body: dict
    candidate: dict | None
    files: frozenset[str]

    @property
    def claims(self):
        return self.candidate if self.kind == "receipt" else self.body

    @property
    def manifest(self):
        return (
            self.body["genesis_tree_manifest_sha256"]
            if self.kind == "genesis"
            else self.claims["subject_tree_manifest_sha256"]
        )

    @property
    def commit(self):
        return (
            self.body["genesis_commit_git_oid"]
            if self.kind == "genesis"
            else self.claims["subject_commit_git_oid"]
        )


@dataclass(frozen=True)
class ChainState:
    anchor: Anchor
    genesis: dict
    tip: Artifact
    sequence: int
    registry: KeyRegistry
    trust_files: Mapping[str, bytes]
    pending: Mapping[str, Artifact]
    terminal: Mapping[str, str]
    activated: bool
    legacy_apply_spki_sha256: str
    legacy_eval_spki_sha256: str

    def predecessor(self, *, lane_commit: str | None = None) -> Predecessor:
        return Predecessor(
            self.anchor.lane,
            self.anchor.epoch_sha256,
            self.tip.address,
            self.tip.kind,
            self.tip.commit if lane_commit is None else lane_commit,
            self.tip.manifest,
            next(iter(self.registry.keys["notary"])),
            self.legacy_apply_spki_sha256,
            self.legacy_eval_spki_sha256,
            self.genesis["consumer_spec_path"],
            self.activated,
        )


def legacy_spki(root: dict) -> str:
    raw = decode_base64(root["public_key_spki_der_base64"])
    _require(raw is not None, "legacy_base64")
    try:
        key = serialization.load_der_public_key(raw)
    except (ValueError, TypeError) as exc:
        raise InvalidChain("legacy_der") from exc
    _require(isinstance(key, Ed25519PublicKey), "legacy_key_type")
    _require(
        key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        == raw,
        "legacy_der_encoding",
    )
    actual = "sha256:" + sha256_hex(
        key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )
    _require(root["raw_key_id"] == actual, "legacy_raw_key_id")
    return sha256_hex(raw)


def _registry(raw: bytes, lane: str, apply: str, evaluation: str) -> KeyRegistry:
    # The genesis digest anchors the initial bytes. Each later registry is
    # authorized by the predecessor's administrative + notary signatures.
    body = canonical_object(raw)
    _require(
        body is not None
        and isinstance(body.get("notary"), list)
        and len(body["notary"]) == 1,
        "notary_registry",
    )
    root = body["notary"][0]
    _require(
        isinstance(root, dict) and digest(root.get("spki_sha256")), "notary_registry"
    )
    registry = parse_registry(
        raw,
        lane=lane,
        notary_spki_sha256=root["spki_sha256"],
        legacy_apply_root=apply,
        legacy_eval_root=evaluation,
    )
    _require(not isinstance(registry, Refusal), "invalid_registry")
    return registry


def _trust(
    files: Mapping[str, bytes], lane: str, apply: str, evaluation: str
) -> KeyRegistry:
    _require(
        not isinstance(
            parse_path_policy(files.get(POLICY_PATH, b""), lane=lane), Refusal
        ),
        "invalid_path_policy",
    )
    _require(
        not isinstance(
            parse_transition_policy(files.get(TRANSITION_POLICY_PATH, b""), lane),
            Refusal,
        ),
        "invalid_transition_policy",
    )
    profile = parse_artifact(files.get(PROFILE_PATH, b""), "profile")
    _require(profile is not None and profile["lane"] == lane, "invalid_profile")
    return _registry(files.get(REGISTRY_PATH, b""), lane, apply, evaluation)


def _preimage(blobs: Mapping[str, bytes], address: str) -> bytes:
    raw = blobs.get(address + ".raw")
    _require(raw is not None and sha256_hex(raw) == address, "missing_preimage")
    return raw


def _signature(blobs, address, role, registry):
    name = f"{address}.json.{role}.sig"
    _require(
        verify_detached(
            blobs.get(name, b""), body_sha256=address, role=role, registry=registry
        ),
        "invalid_" + role + "_signature",
    )
    return name


def _parse_body(blobs, address, kind):
    raw = blobs.get(address + ".json", b"")
    _require(sha256_hex(raw) == address, "body_address")
    body = parse_artifact(raw, kind)
    _require(body is not None, "body_schema")
    return body


def _genesis_bundle(blobs, address, anchor):
    body = _parse_body(blobs, address, "genesis")
    _require(
        address == anchor.epoch_sha256
        and body["lane"] == anchor.lane
        and body["notary_repository"] == anchor.notary_repository
        and anchor.notary_repository == anchor.lane + "-notary",
        "genesis_anchor",
    )
    apply, evaluation = (
        legacy_spki(body["legacy_apply_root"]),
        legacy_spki(body["legacy_eval_root"]),
    )
    trust = {
        path: _preimage(blobs, body["bootstrap_policies"][name])
        for name, path in BOOTSTRAP_PATHS.items()
    }
    registry = _trust(trust, anchor.lane, apply, evaluation)
    policy = parse_path_policy(trust[POLICY_PATH], lane=anchor.lane)
    paths = [row[0] for row in body["v5_attested"] + body["baseline_unattested"]]
    _require(
        len(paths) == len(set(paths)) and all(policy.protects(path) for path in paths),
        "genesis_inventory",
    )
    files = {
        address + ".json",
        _signature(blobs, address, "genesis", registry),
        _signature(blobs, address, "admin-approver", registry),
    } | {d + ".raw" for d in body["bootstrap_policies"].values()}
    return (
        Artifact(address, "genesis", body, None, frozenset(files)),
        trust,
        registry,
        apply,
        evaluation,
    )


def validate_pending(
    blobs: Mapping[str, bytes], address: str, kind: str, state: ChainState
) -> Artifact:
    """Verify a complete pending bundle under current predecessor roots."""
    _require(kind in ("receipt", "transition"), "pending_kind")
    body = _parse_body(blobs, address, kind)
    _require(
        body["lane"] == state.anchor.lane
        and body["epoch_sha256"] == state.anchor.epoch_sha256,
        "pending_binding",
    )
    files = {address + ".json"}
    candidate = None
    if kind == "receipt":
        _require(state.activated, "epoch_not_activated")
        candidate = _parse_body(blobs, body["candidate_sha256"], "receipt-candidate")
        report = _parse_body(blobs, candidate["report_sha256"], "report-pass")
        expected = report | {
            "schema": PREFIX + "receipt-candidate/v1",
            "report_sha256": candidate["report_sha256"],
            "job1": candidate["job1"],
        }
        _require(candidate == expected, "candidate_report_mismatch")
        _require(
            candidate["lane"] == body["lane"]
            and candidate["epoch_sha256"] == body["epoch_sha256"],
            "wrapper_binding",
        )
        approval_name = _signature(
            blobs, body["candidate_sha256"], "approver", state.registry
        )
        _require(
            body["authorization"]["approval_signature_sha256"]
            == sha256_hex(blobs[approval_name]),
            "approval_reference",
        )
        _require(
            body["authorization"]["environment"] == "notary-signing",
            "approval_environment",
        )
        _require(
            candidate["profile_sha256"] == sha256_hex(state.trust_files[PROFILE_PATH])
            and candidate["path_policy_sha256"]
            == sha256_hex(state.trust_files[POLICY_PATH]),
            "receipt_policy_binding",
        )
        _require(
            check_gates(
                parse_artifact(state.trust_files[PROFILE_PATH], "profile"),
                candidate["gates"],
            )
            is None,
            "receipt_gates",
        )
        files |= {
            body["candidate_sha256"] + ".json",
            candidate["report_sha256"] + ".json",
            approval_name,
            _signature(blobs, address, "notary", state.registry),
        }
    else:
        files |= {
            _signature(blobs, address, "transition", state.registry),
            _signature(blobs, address, "admin-approver", state.registry),
        }
        policy = parse_path_policy(state.trust_files[POLICY_PATH], lane=body["lane"])
        transitions = parse_transition_policy(
            state.trust_files[TRANSITION_POLICY_PATH], body["lane"]
        )
        _require(bool(body["delta"]), "empty_transition")
        for change in body["delta"]:
            path = change["path"]
            _require(
                transitions.protects(path) and not policy.protects(path),
                "transition_domain",
            )
            _require(
                (change["before_entry_sha256"], change["before_mode"])
                != (change["after_entry_sha256"], change["after_mode"]),
                "unchanged_delta_entry",
            )
            if change["after_entry_sha256"] is not None:
                _preimage(blobs, change["after_entry_sha256"])
                files.add(change["after_entry_sha256"] + ".raw")
        _transition_state(blobs, body, state)  # validate successor trust preimages now
    claims = candidate if candidate is not None else body
    _require(
        claims["chain_predecessor_sha256"] == state.tip.address
        and claims["chain_predecessor_kind"] == state.tip.kind
        and claims["base_tree_manifest_sha256"] == state.tip.manifest,
        "pending_predecessor",
    )
    _require(
        claims["base_tree_manifest_sha256"] != claims["subject_tree_manifest_sha256"],
        "state_identical",
    )
    return Artifact(address, kind, body, candidate, frozenset(files))


def _transition_state(blobs, body, state):
    trust = dict(state.trust_files)
    for change in body["delta"]:
        path, after = change["path"], change["after_entry_sha256"]
        if path in trust and state.activated:
            _require(
                sha256_hex(trust[path]) == change["before_entry_sha256"],
                "transition_before_preimage",
            )
        if after is None:
            trust.pop(path, None)
        else:
            trust[path] = _preimage(blobs, after)
    if not state.activated:
        allowed = set(BOOTSTRAP_PATHS.values()) | {state.genesis["consumer_spec_path"]}
        _require(
            all(d["path"] in allowed for d in body["delta"]), "activation_extra_path"
        )
        for name, path in BOOTSTRAP_PATHS.items():
            _require(
                path in trust
                and sha256_hex(trust[path])
                == state.genesis["bootstrap_policies"][name],
                "activation_policy",
            )
        consumer = trust.get(state.genesis["consumer_spec_path"], b"")
        template = epoch_template(consumer, state.anchor.epoch_sha256)
        _require(template is not None, "activation_epoch")
        _require(
            sha256_hex(template) == state.genesis["activation_spec_template_sha256"],
            "activation_spec_template",
        )
    registry = _trust(
        trust,
        state.anchor.lane,
        state.legacy_apply_spki_sha256,
        state.legacy_eval_spki_sha256,
    )
    consumer = parse_consumer(trust.get(state.genesis["consumer_spec_path"], b""))
    _require(
        consumer is not None
        and consumer["lane"] == state.anchor.lane
        and consumer["epoch_sha256"] == state.anchor.epoch_sha256
        and consumer["notary_repository"] == state.anchor.notary_repository
        and set(registry.keys["notary"]) == {consumer["notary_spki_sha256"]},
        "successor_consumer_binding",
    )
    return trust, registry


def reconstruct(history: Sequence[Snapshot], anchor: Anchor) -> ChainState | Refusal:
    """Validate snapshots in authenticated linear branch order, including HEAD."""
    try:
        return _reconstruct(history, anchor)
    except (InvalidChain, KeyError, TypeError, ValueError) as exc:
        return Refusal("chain-unresolvable", None, str(exc))


def _reconstruct(history, anchor):
    _require(len(history) >= 2, "genesis_not_finalized")
    previous: dict[str, bytes] = {}
    state = None
    pending: dict[str, Artifact] = {}
    terminal: dict[str, str] = {}
    for index, snapshot in enumerate(history):
        blobs = snapshot.blobs
        _require(
            all(mode == "100644" for _, mode, _ in snapshot.manifest), "chain_modes"
        )
        for name, raw in previous.items():
            _require(name == "HEAD.json" or blobs.get(name) == raw, "bundle_mutation")
        new = blobs.keys() - previous.keys()
        addressed = {}
        for name in new:
            if name == "HEAD.json":
                continue
            match = _BODY.fullmatch(name) or _RAW.fullmatch(name)
            if match:
                _require(sha256_hex(blobs[name]) == match[1], "chain_address")
                if name.endswith(".json"):
                    body = parse_artifact(blobs[name])
                    _require(body is not None, "chain_schema")
                    addressed[match[1]] = body
            else:
                _require(_SIDECAR.fullmatch(name) is not None, "chain_store_name")
        if index == 0:
            genesis, trust, registry, apply, evaluation = _genesis_bundle(
                blobs, anchor.epoch_sha256, anchor
            )
            _require(set(blobs) == genesis.files, "first_commit_bundle")
            pending[genesis.address] = genesis
            state = ChainState(
                anchor,
                genesis.body,
                genesis,
                0,
                registry,
                trust,
                dict(pending),
                {},
                False,
                apply,
                evaluation,
            )
            previous = dict(blobs)
            continue
        _require(state is not None, "missing_genesis")
        expected = set()
        pending_before = dict(pending)
        for address, body in sorted(addressed.items()):
            kind = body["schema"][len(PREFIX) : -3]
            _require(kind != "genesis", "second_genesis")
            if kind in ("receipt", "transition"):
                artifact = validate_pending(blobs, address, kind, state)
                pending[address] = artifact
                expected |= artifact.files
        markers = [
            (address, body)
            for address, body in addressed.items()
            if body["schema"] in (PREFIX + "finalization/v1", PREFIX + "void/v1")
        ]
        finals = [
            (a, b) for a, b in markers if b["schema"] == PREFIX + "finalization/v1"
        ]
        _require(len(finals) <= 1, "multiple_finalizations_in_commit")
        if index == 1:
            _require(
                len(markers) == 1
                and len(finals) == 1
                and finals[0][1]["target_sha256"] == anchor.epoch_sha256,
                "second_commit_genesis_finalization",
            )
            _require(
                new == {finals[0][0] + ".json", "HEAD.json"},
                "second_commit_extra_files",
            )
        marker_targets = [b["target_sha256"] for _, b in markers]
        _require(len(marker_targets) == len(set(marker_targets)), "conflicting_markers")
        for address, marker in finals + [
            (a, b) for a, b in markers if b["schema"] == PREFIX + "void/v1"
        ]:
            target = marker["target_sha256"]
            _require(
                marker["lane"] == anchor.lane
                and marker["epoch_sha256"] == anchor.epoch_sha256,
                "marker_binding",
            )
            _require(
                target in pending_before and target not in terminal,
                "marker_target_not_pending",
            )
            artifact = pending_before[target]
            _require(marker["target_kind"] == artifact.kind, "marker_kind")
            expected.add(address + ".json")
            if marker["schema"] == PREFIX + "finalization/v1":
                _require(
                    marker["merged_tip_manifest_sha256"] == artifact.manifest
                    and marker["sequence"] == str(state.sequence + 1),
                    "finalization_state",
                )
                if artifact.kind != "genesis":
                    _require(
                        artifact.claims["chain_predecessor_sha256"] == state.tip.address
                        and artifact.claims["chain_predecessor_kind"] == state.tip.kind
                        and artifact.claims["base_tree_manifest_sha256"]
                        == state.tip.manifest,
                        "finalization_predecessor",
                    )
                    siblings = {
                        a
                        for a, p in pending.items()
                        if a != target
                        and p.kind != "genesis"
                        and p.claims["chain_predecessor_sha256"] == state.tip.address
                    }
                    voids = {
                        b["target_sha256"]
                        for _, b in markers
                        if b["schema"] == PREFIX + "void/v1"
                    }
                    _require(siblings <= voids, "superseded_pending_not_voided")
                trust, registry = (state.trust_files, state.registry)
                if artifact.kind == "transition":
                    trust, registry = _transition_state(blobs, artifact.body, state)
                terminal[target] = "finalized"
                state = ChainState(
                    anchor,
                    state.genesis,
                    artifact,
                    state.sequence + 1,
                    registry,
                    trust,
                    {},
                    {},
                    state.activated or artifact.kind == "transition",
                    state.legacy_apply_spki_sha256,
                    state.legacy_eval_spki_sha256,
                )
            else:
                _require(artifact.kind != "genesis", "genesis_void")
                terminal[target] = "void"
            pending.pop(target)
        _require(new <= expected | {"HEAD.json"}, "unreachable_chain_file")
        _require(all(name in blobs for name in expected), "incomplete_bundle")
        head = canonical_object(blobs.get("HEAD.json", b""))
        _require(
            fields(head, {"schema", "tip_sha256", "tip_kind"})
            and head
            == {
                "schema": PREFIX + "head/v1",
                "tip_sha256": state.tip.address,
                "tip_kind": state.tip.kind,
            },
            "head_divergence",
        )
        previous = dict(blobs)
    _require(state is not None and state.sequence > 0, "unfinalized_chain")
    _require(
        set(state.registry.keys["notary"]) == {anchor.notary_spki_sha256},
        "consumer_notary_pin",
    )
    return ChainState(
        anchor,
        state.genesis,
        state.tip,
        state.sequence,
        state.registry,
        dict(state.trust_files),
        dict(pending),
        dict(terminal),
        state.activated,
        state.legacy_apply_spki_sha256,
        state.legacy_eval_spki_sha256,
    )
