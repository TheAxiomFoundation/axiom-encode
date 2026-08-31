"""Protected apply-root signing for unmanifested-legacy cleanup receipts.

The signing broker treats its input as opaque bytes.  These helpers add the
cleanup-specific inner domain and exact receipt-schema checks needed to keep a
cleanup receipt distinct from every other apply-root artifact.
"""

from __future__ import annotations

import hashlib
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .legacy_cleanup import (
    LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
    LEGACY_CLEANUP_SIGNATURE_DOMAIN,
    LegacyCleanupReceiptError,
    cleanup_signature_payload,
    receipt_structure_issues,
)
from .signing_broker import (
    SigningBroker,
    SigningBrokerError,
    canonical_signing_message,
)

_SIGNATURE_FIELDS = {"algorithm", "domain", "key_id", "value"}
_SIGNATURE_PLACEHOLDER = b64encode(bytes(64)).decode("ascii")


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def legacy_cleanup_key_id(public_key: Ed25519PublicKey) -> str:
    """Return the cleanup envelope ID for one raw Ed25519 apply key."""

    return f"sha256:{hashlib.sha256(_raw_public_key(public_key)).hexdigest()}"


def _coerce_apply_public_key(
    verifier: SigningBroker | Ed25519PublicKey,
) -> Ed25519PublicKey:
    if isinstance(verifier, Ed25519PublicKey):
        return verifier
    try:
        public_key_raw = verifier.apply_public_key_raw
    except (AttributeError, TypeError) as exc:
        raise SigningBrokerError(
            "Cleanup receipt verifier has no protected apply public key"
        ) from exc
    if not isinstance(public_key_raw, bytes) or len(public_key_raw) != 32:
        raise SigningBrokerError(
            "Trusted signing broker has no valid cleanup apply public key"
        )
    try:
        return Ed25519PublicKey.from_public_bytes(public_key_raw)
    except ValueError as exc:
        raise SigningBrokerError(
            "Trusted signing broker has no valid cleanup apply public key"
        ) from exc


def _placeholder_signature() -> dict[str, str]:
    return {
        "algorithm": LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
        "key_id": f"sha256:{'0' * 64}",
        "domain": LEGACY_CLEANUP_SIGNATURE_DOMAIN,
        "value": _SIGNATURE_PLACEHOLDER,
    }


def _unsigned_structure_issues(payload: Mapping[str, object]) -> list[str]:
    """Validate every receipt field independently of its real envelope."""

    candidate = dict(payload)
    candidate.pop("signature", None)
    candidate["signature"] = _placeholder_signature()
    try:
        return receipt_structure_issues(candidate)
    except (RecursionError, TypeError, UnicodeError, ValueError):
        return ["receipt contains a value outside the canonical v1 JSON schema"]


def legacy_cleanup_signature_issue(
    payload: Mapping[str, object],
    verifier: SigningBroker | Ed25519PublicKey,
) -> str | None:
    """Return a precise issue or verify one cleanup receipt signature.

    The inner cleanup domain is wrapped in the broker's existing
    ``apply_ed25519`` scope.  No applied-manifest serialization or verification
    helper participates in this contract.
    """

    if not isinstance(payload, Mapping):
        return "legacy cleanup receipt structure is invalid: expected an object"
    unsigned_issues = _unsigned_structure_issues(payload)
    if unsigned_issues:
        return (
            "legacy cleanup receipt structure is invalid: "
            + "; ".join(unsigned_issues)
        )

    signature = payload.get("signature")
    if not isinstance(signature, dict) or set(signature) != _SIGNATURE_FIELDS:
        return "legacy cleanup receipt signature envelope is malformed"
    if signature.get("algorithm") != LEGACY_CLEANUP_SIGNATURE_ALGORITHM:
        return "legacy cleanup receipt signature algorithm is invalid"
    if signature.get("domain") != LEGACY_CLEANUP_SIGNATURE_DOMAIN:
        return "legacy cleanup receipt signature domain is invalid"

    try:
        public_key = _coerce_apply_public_key(verifier)
    except SigningBrokerError as exc:
        return f"legacy cleanup receipt trust root is invalid: {exc}"
    if signature.get("key_id") != legacy_cleanup_key_id(public_key):
        return "legacy cleanup receipt uses an unknown signing key"

    encoded_signature = signature.get("value")
    if not isinstance(encoded_signature, str):
        return "legacy cleanup receipt signature value is malformed"
    try:
        raw_signature = b64decode(encoded_signature.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError):
        return "legacy cleanup receipt signature value is malformed"
    if len(raw_signature) != 64:
        return "legacy cleanup receipt signature value is malformed"

    try:
        public_key.verify(
            raw_signature,
            canonical_signing_message(
                "apply_ed25519",
                cleanup_signature_payload(payload),
            ),
        )
    except InvalidSignature:
        return "legacy cleanup receipt cryptographic signature is invalid"
    return None


def verify_legacy_cleanup_receipt_signature(
    payload: Mapping[str, object],
    verifier: SigningBroker | Ed25519PublicKey,
) -> None:
    """Raise when a receipt is malformed or not signed by the apply root."""

    issue = legacy_cleanup_signature_issue(payload, verifier)
    if issue is not None:
        raise LegacyCleanupReceiptError(issue)


def sign_legacy_cleanup_receipt(
    payload: dict[str, object],
    signing_broker: SigningBroker,
) -> None:
    """Validate and attach one cleanup-specific protected signature in place."""

    if not isinstance(payload, dict):
        raise LegacyCleanupReceiptError(
            "legacy cleanup unsigned receipt must be a mutable JSON object"
        )
    if "signature" in payload:
        raise LegacyCleanupReceiptError(
            "legacy cleanup unsigned receipt must not contain a signature"
        )
    unsigned_issues = _unsigned_structure_issues(payload)
    if unsigned_issues:
        raise LegacyCleanupReceiptError(
            "cannot sign malformed legacy cleanup unsigned receipt: "
            + "; ".join(unsigned_issues)
        )

    public_key = _coerce_apply_public_key(signing_broker)
    inner_payload = cleanup_signature_payload(payload)
    raw_signature = signing_broker.apply_ed25519_sign(inner_payload)
    if not isinstance(raw_signature, bytes) or len(raw_signature) != 64:
        raise SigningBrokerError(
            "Trusted signing broker returned an invalid cleanup receipt signature"
        )
    signature = {
        "algorithm": LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
        "key_id": legacy_cleanup_key_id(public_key),
        "domain": LEGACY_CLEANUP_SIGNATURE_DOMAIN,
        "value": b64encode(raw_signature).decode("ascii"),
    }
    signed_candidate = dict(payload)
    signed_candidate["signature"] = signature
    issue = legacy_cleanup_signature_issue(signed_candidate, public_key)
    if issue is not None:
        raise SigningBrokerError(
            "Trusted signing broker returned an unverifiable cleanup receipt "
            f"signature: {issue}"
        )
    if cleanup_signature_payload(payload) != inner_payload:
        raise SigningBrokerError(
            "Legacy cleanup unsigned receipt changed while it was being signed"
        )
    payload["signature"] = signature
