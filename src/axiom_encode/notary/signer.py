"""Typed external signer core. No generation or repository-write operation.

The deployment adapter owns readers and key custody. Requesters supply only
an artifact digest and a reviewer signature; never roots, repositories, base
refs, policy choices, a signing scope, or arbitrary bytes to sign.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._schema import canonical_object, digest
from .administration import build_genesis, build_transition
from .canonical import jcs_dumps, sha256_hex
from .chain import BOOTSTRAP_PATHS, ChainState, _trust
from .identity import IdentityRefusal, JobIdentity, JobPolicy, ReadAPI, authenticate_job
from .producers import require_enrolled_submission
from .protocol import parse_artifact
from .provenance import report_provenance
from .refusal import Refusal
from .registry import KeyRegistry
from .signatures import ROLE_SCOPES, verify_detached
from .verification import Snapshot, reconcile, verify_snapshots


@dataclass(frozen=True)
class ReceiptInputs:
    base: Snapshot
    subject: Snapshot
    state: ChainState
    inventory: bytes
    report_archive: bytes
    candidate_raw: bytes
    pr_number: str


@dataclass(frozen=True)
class TransitionInputs:
    base: Snapshot
    subject: Snapshot
    state: ChainState
    candidate_raw: bytes


@dataclass(frozen=True)
class GenesisInputs:
    snapshot: Snapshot
    arguments: dict
    candidate_raw: bytes


class SignerInputs(Protocol):
    """Service-owned adapter: read exact artifacts and authenticated remote state.

    Implementations must authenticate the chain branch protection and history,
    derive the current lane tip from GitHub, and verify the administrative lane
    lock for genesis. They must never use a requester's checkout or config.
    """

    def receipt(
        self, identity: JobIdentity, candidate_sha256: str
    ) -> ReceiptInputs: ...
    def transition(
        self, identity: JobIdentity, candidate_sha256: str
    ) -> TransitionInputs: ...
    def genesis(
        self, identity: JobIdentity, candidate_sha256: str
    ) -> GenesisInputs: ...


def _signed_sidecar(key: Ed25519PrivateKey, body: bytes, role: str) -> bytes:
    scope, address = ROLE_SCOPES[role], sha256_hex(body)
    der = key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    signature = key.sign(
        b"axiom-encode/external-signer-sign/v2\0"
        + scope.encode()
        + b"\0"
        + address.encode()
    )
    return jcs_dumps(
        {
            "schema": "axiom/detached-signature/v1",
            "body_sha256": address,
            "scope": scope,
            "signer_spki_sha256": sha256_hex(der),
            "signature_base64": base64.b64encode(signature).decode(),
        }
    )


def make_candidate(
    api: ReadAPI,
    identity: JobIdentity,
    inputs: ReceiptInputs,
    *,
    require_recompute: bool,
) -> bytes:
    provenance, proposal = report_provenance(
        api, identity, inputs.report_archive, require_recompute=require_recompute
    )
    report = parse_artifact(proposal, "report-pass")
    if report is None:
        raise IdentityRefusal("refusal_or_malformed_report")
    if inputs.state.anchor.lane != identity.repository:
        raise IdentityRefusal("chain_lane")
    recomputed = verify_snapshots(
        inputs.base,
        inputs.subject,
        inputs.state.predecessor(lane_commit=inputs.base.commit),
        inputs.inventory,
        report["gates"],
    )
    if isinstance(reconcile(proposal, recomputed), Refusal):
        raise IdentityRefusal("report_reconciliation")
    pr = require_enrolled_submission(
        api,
        pr_number=inputs.pr_number,
        base=inputs.base,
        subject=inputs.subject,
        registry=inputs.state.registry,
        report_raw=recomputed,
    )
    if pr["base"].get("sha") != inputs.base.commit:
        raise IdentityRefusal("pr_base_moved")
    candidate = report | {
        "schema": "axiom/notary-receipt-candidate/v1",
        "report_sha256": sha256_hex(proposal),
        "job1": provenance,
    }
    raw = jcs_dumps(candidate)
    if parse_artifact(raw, "receipt-candidate") is None:
        raise IdentityRefusal("candidate_schema")
    return raw


class NotarySigner:
    """Separate service deployment with one notary key and read-only readers."""

    def __init__(
        self,
        key: Ed25519PrivateKey,
        policy: JobPolicy,
        api: ReadAPI,
        inputs: SignerInputs,
    ):
        if policy.job_name != "approve" or policy.environment != "notary-signing":
            raise ValueError("notary signer only serves the approve job")
        self._key, self._policy, self._api, self._inputs = key, policy, api, inputs

    def _authenticate(self, token: str, address: str) -> JobIdentity:
        if not digest(address):
            raise IdentityRefusal("candidate_address")
        return authenticate_job(token, self._policy, self._api)

    def _approve(
        self,
        candidate: bytes,
        requested: str,
        approval: bytes,
        registry: KeyRegistry,
        *,
        administrative: bool,
    ):
        der = self._key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        if set(registry.keys["notary"]) != {sha256_hex(der)}:
            raise IdentityRefusal("signer_not_current_root")
        if sha256_hex(candidate) != requested or not verify_detached(
            approval,
            body_sha256=requested,
            role="admin-approver" if administrative else "approver",
            registry=registry,
        ):
            raise IdentityRefusal("digest_bound_approval")

    def receipt(
        self, token: str, candidate_sha256: str, approval: bytes
    ) -> dict[str, bytes]:
        identity = self._authenticate(token, candidate_sha256)
        inputs = self._inputs.receipt(identity, candidate_sha256)
        candidate = make_candidate(self._api, identity, inputs, require_recompute=True)
        if candidate != inputs.candidate_raw:
            raise IdentityRefusal("candidate_reconciliation")
        self._approve(
            candidate,
            candidate_sha256,
            approval,
            inputs.state.registry,
            administrative=False,
        )
        parsed = parse_artifact(candidate, "receipt-candidate")
        wrapper = jcs_dumps(
            {
                "schema": "axiom/notary-receipt/v1",
                "lane": parsed["lane"],
                "epoch_sha256": parsed["epoch_sha256"],
                "candidate_sha256": candidate_sha256,
                "authorization": {
                    "environment": identity.environment,
                    "approve_check_run_id": identity.check_run_id,
                    "approval_signature_sha256": sha256_hex(approval),
                },
            }
        )
        from .provenance import read_report_archive

        report = read_report_archive(
            inputs.report_archive, parsed["job1"]["artifact_sha256"]
        )
        address = sha256_hex(wrapper)
        return {
            address + ".json": wrapper,
            address + ".json.notary.sig": _signed_sidecar(self._key, wrapper, "notary"),
            candidate_sha256 + ".json": candidate,
            candidate_sha256 + ".json.approver.sig": approval,
            parsed["report_sha256"] + ".json": report,
        }

    def transition(
        self, token: str, candidate_sha256: str, approval: bytes
    ) -> dict[str, bytes]:
        identity = self._authenticate(token, candidate_sha256)
        inputs = self._inputs.transition(identity, candidate_sha256)
        proposal = parse_artifact(inputs.candidate_raw, "transition")
        if proposal is None or inputs.state.anchor.lane != identity.repository:
            raise IdentityRefusal("administrative_candidate")
        candidate = build_transition(
            inputs.base, inputs.subject, inputs.state, reason=proposal["reason"]
        )
        if candidate != inputs.candidate_raw:
            raise IdentityRefusal("candidate_reconciliation")
        self._approve(
            candidate,
            candidate_sha256,
            approval,
            inputs.state.registry,
            administrative=True,
        )
        preimages = {
            row["after_entry_sha256"] + ".raw": inputs.subject.blobs[row["path"]]
            for row in proposal["delta"]
            if row["after_entry_sha256"] is not None
        }
        return preimages | {
            candidate_sha256 + ".json": candidate,
            candidate_sha256 + ".json.admin-approver.sig": approval,
            candidate_sha256 + ".json.transition.sig": _signed_sidecar(
                self._key, candidate, "transition"
            ),
        }

    def genesis(
        self, token: str, candidate_sha256: str, approval: bytes
    ) -> dict[str, bytes]:
        identity = self._authenticate(token, candidate_sha256)
        inputs = self._inputs.genesis(identity, candidate_sha256)
        if inputs.arguments["lane"] != identity.repository:
            raise IdentityRefusal("administrative_lane")
        candidate = build_genesis(inputs.snapshot, **inputs.arguments)
        if candidate != inputs.candidate_raw:
            raise IdentityRefusal("candidate_reconciliation")
        registry = _trust(
            inputs.arguments["prospective"],
            identity.repository,
            inputs.arguments["frozen_apply_spki_sha256"],
            inputs.arguments["frozen_eval_spki_sha256"],
        )
        self._approve(
            candidate, candidate_sha256, approval, registry, administrative=True
        )
        if (
            canonical_object(approval)["signer_spki_sha256"]
            != inputs.arguments["ceremony_admin_spki_sha256"]
        ):
            raise IdentityRefusal("bootstrap_ceremony_approver")
        preimages = {
            sha256_hex(inputs.arguments["prospective"][path])
            + ".raw": inputs.arguments["prospective"][path]
            for path in BOOTSTRAP_PATHS.values()
        }
        return preimages | {
            candidate_sha256 + ".json": candidate,
            candidate_sha256 + ".json.admin-approver.sig": approval,
            candidate_sha256 + ".json.genesis.sig": _signed_sidecar(
                self._key, candidate, "genesis"
            ),
        }
