"""Atomic publication/finalization plans, validated by full reconstruction.

Only the separate publisher-token broker executes finalization plans. A lane
workflow cannot supply a manifest, marker, sequence, or chain write token.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .canonical import jcs_dumps, sha256_hex
from .chain import Anchor, ChainState, InvalidChain, _transition_state, reconstruct
from .manifest import manifest_sha256
from .refusal import Refusal
from .verification import Snapshot


@dataclass(frozen=True)
class PublicationPlan:
    expected_chain_commit: str
    files: Mapping[str, bytes]
    state: ChainState
    admitted: bool


def _state(history: Sequence[Snapshot], anchor: Anchor) -> ChainState:
    result = reconstruct(history, anchor)
    if isinstance(result, Refusal):
        raise InvalidChain("publication_invalid_chain")
    return result


def _append(history: Sequence[Snapshot], files: Mapping[str, bytes]) -> Snapshot:
    blobs = dict(history[-1].blobs)
    for path, raw in files.items():
        if path != "HEAD.json" and path in blobs and blobs[path] != raw:
            raise InvalidChain("publication_rewrite")
        blobs[path] = raw
    return Snapshot(
        "f" * 40,
        sorted((p, "100644", sha256_hex(raw)) for p, raw in blobs.items()),
        blobs,
    )


def plan_publication(
    history: Sequence[Snapshot],
    anchor: Anchor,
    bundle: Mapping[str, bytes],
    address: str,
    kind: str,
    base: Snapshot,
    subject: Snapshot,
) -> PublicationPlan:
    state = _state(history, anchor)
    if kind not in ("receipt", "transition") or "HEAD.json" in bundle:
        raise InvalidChain("publication_kind_or_pointer")
    if address in state.terminal:
        raise InvalidChain("publication_terminal")
    if manifest_sha256(base.manifest) != state.tip.manifest:
        raise InvalidChain("publication_stale_base")
    if address in state.pending:
        artifact = state.pending[address]
        if set(bundle) != set(artifact.files) or any(
            history[-1].blobs.get(p) != b for p, b in bundle.items()
        ):
            raise InvalidChain("publication_retry_bundle")
        next_state = state
        files = {}
    else:
        updated = _append(history, bundle)
        next_state = _state([*history, updated], anchor)
        if set(next_state.pending) != set(state.pending) | {address}:
            raise InvalidChain("publication_extra_artifact")
        artifact = next_state.pending[address]
        if set(bundle) != set(artifact.files):
            raise InvalidChain("publication_open_bundle")
        files = {p: b for p, b in bundle.items() if p not in history[-1].blobs}
    if (
        artifact.kind != kind
        or artifact.manifest != manifest_sha256(subject.manifest)
        or artifact.commit != subject.commit
        or artifact.claims["chain_predecessor_sha256"] != state.tip.address
        or artifact.claims["chain_predecessor_kind"] != state.tip.kind
    ):
        raise InvalidChain("publication_subject_or_predecessor")
    return PublicationPlan(history[-1].commit, files, next_state, True)


def _marker(body: dict) -> tuple[str, bytes]:
    raw = jcs_dumps(body)
    return sha256_hex(raw) + ".json", raw


def plan_finalization(
    history: Sequence[Snapshot],
    anchor: Anchor,
    target: str,
    merged: Snapshot,
) -> PublicationPlan:
    """Re-derive the marker from broker-read merged bytes and pending authority.

    The caller authenticates a post-merge request, resolves the actual merge's
    target through its App-bound check, and performs an atomic non-force ref
    update from expected_chain_commit. A CAS loser reloads and recomputes.
    """
    state = _state(history, anchor)
    if target in state.terminal or target not in state.pending:
        raise InvalidChain("finalization_not_pending")
    artifact = state.pending[target]
    actual = manifest_sha256(merged.manifest)
    accepted = (
        actual == artifact.manifest
        and artifact.claims["chain_predecessor_sha256"] == state.tip.address
        and artifact.claims["chain_predecessor_kind"] == state.tip.kind
        and artifact.claims["base_tree_manifest_sha256"] == state.tip.manifest
    )
    common = {"lane": anchor.lane, "epoch_sha256": anchor.epoch_sha256}
    files = {}
    if accepted:
        name, raw = _marker(
            common
            | {
                "schema": "axiom/notary-finalization/v1",
                "target_sha256": target,
                "target_kind": artifact.kind,
                "merged_tip_manifest_sha256": actual,
                "sequence": str(state.sequence + 1),
            }
        )
        files[name] = raw
        files["HEAD.json"] = jcs_dumps(
            {
                "schema": "axiom/notary-head/v1",
                "tip_sha256": target,
                "tip_kind": artifact.kind,
            }
        )
        voids = [
            a
            for a in state.pending.values()
            if a.address != target
            and a.claims["chain_predecessor_sha256"] == state.tip.address
            and a.claims["chain_predecessor_kind"] == state.tip.kind
        ]
    else:
        voids = [artifact]
    for pending in voids:
        name, raw = _marker(
            common
            | {
                "schema": "axiom/notary-void/v1",
                "target_sha256": pending.address,
                "target_kind": pending.kind,
                "reason": "superseded"
                if accepted
                else "merged-state-or-predecessor-mismatch",
            }
        )
        files[name] = raw
    successor_anchor = anchor
    if accepted and artifact.kind == "transition":
        _, registry = _transition_state(history[-1].blobs, artifact.body, state)
        successor_anchor = replace(
            anchor, notary_spki_sha256=next(iter(registry.keys["notary"]))
        )
    next_state = _state([*history, _append(history, files)], successor_anchor)
    return PublicationPlan(history[-1].commit, files, next_state, accepted)
