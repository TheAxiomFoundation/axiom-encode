"""Detached signature verification against v33 registry roles; no signing API."""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature

from ._schema import canonical_object, decode_base64, digest, fields
from .registry import KeyRegistry

ROLE_SCOPES = {
    "producer": "axiom/lineage-generation/v1",
    "actor": "axiom/lineage-correction/v1",
    "review": "axiom/lineage-correction-review/v1",
    "genesis": "axiom/notary-genesis/v1",
    "transition": "axiom/notary-transition/v1",
    "approver": "axiom/notary-approval/v1",
    "admin-approver": "axiom/notary-admin-approval/v1",
    "notary": "axiom/notary-receipt/v1",
}
_FRAME = b"axiom-encode/external-signer-sign/v2\0"


def verify_detached(
    raw: bytes, *, body_sha256: str, role: str, registry: KeyRegistry
) -> bool:
    """Authenticate this body digest under the required role, never the sidecar's role."""
    if role not in ROLE_SCOPES or not digest(body_sha256):
        return False
    sidecar = canonical_object(raw)
    if not fields(
        sidecar,
        {"schema", "body_sha256", "scope", "signer_spki_sha256", "signature_base64"},
    ):
        return False
    scope = ROLE_SCOPES[role]
    if (
        sidecar["schema"] != "axiom/detached-signature/v1"
        or sidecar["body_sha256"] != body_sha256
        or sidecar["scope"] != scope
        or not digest(sidecar["signer_spki_sha256"])
    ):
        return False
    registry_role = "notary" if role in {"genesis", "transition"} else role
    public = registry.keys[registry_role].get(sidecar["signer_spki_sha256"])
    signature = decode_base64(sidecar["signature_base64"])
    if public is None or signature is None or len(signature) != 64:
        return False
    message = _FRAME + scope.encode("ascii") + b"\0" + body_sha256.encode("ascii")
    try:
        public.verify(signature, message)
    except InvalidSignature:
        return False
    return True
