"""Asymmetric authentication for persisted eval-suite generation evidence."""

from __future__ import annotations

import hashlib
import json
from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Iterator

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from axiom_encode.signing_broker import (
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV as APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
)
from axiom_encode.signing_broker import (
    EVAL_EVIDENCE_PRIVATE_KEY_ENV as EVAL_EVIDENCE_PRIVATE_KEY_ENV,
)
from axiom_encode.signing_broker import (
    SigningBroker,
    SigningBrokerError,
    canonical_signing_message,
    get_signing_broker,
    reject_direct_private_signing_environment,
    scrub_private_signing_environment,
)

EVAL_EVIDENCE_SIGNATURE_ALGORITHM = "ed25519-domain-v1"


def canonical_evidence_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize unsigned evidence deterministically for signing."""

    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@contextmanager
def isolated_eval_evidence_signer() -> Iterator[SigningBroker]:
    """Yield the out-of-process eval signer attached before the entrypoint."""

    reject_direct_private_signing_environment()
    broker = get_signing_broker(capability="eval_ed25519")
    yield broker


def load_eval_evidence_public_key_from_broker() -> Ed25519PublicKey:
    """Load the broker-provisioned eval root; environment trust is forbidden."""

    try:
        broker = get_signing_broker()
    except SigningBrokerError as exc:
        raise ValueError(
            "A protected signing broker is required to verify authenticated "
            "eval result evidence"
        ) from exc
    public_key_raw = broker.eval_public_key_raw
    if public_key_raw is None or len(public_key_raw) != 32:
        raise ValueError("Protected signing broker has no eval evidence trust root")
    return Ed25519PublicKey.from_public_bytes(public_key_raw)


def sign_eval_evidence(
    payload: Mapping[str, object],
    signer: SigningBroker,
) -> dict[str, str]:
    """Return a detached Ed25519 signature for one evidence payload."""

    public_key_raw = signer.eval_public_key_raw
    if public_key_raw is None or len(public_key_raw) != 32:
        raise SigningBrokerError("Trusted signing broker has no eval public key")
    raw_signature = signer.eval_ed25519_sign(canonical_evidence_bytes(payload))
    public_key = Ed25519PublicKey.from_public_bytes(public_key_raw)
    return {
        "algorithm": EVAL_EVIDENCE_SIGNATURE_ALGORITHM,
        "key_id": eval_evidence_key_id(public_key),
        "value": b64encode(raw_signature).decode("ascii"),
    }


def verify_eval_evidence_signature(
    payload: Mapping[str, object],
    signature: object,
) -> None:
    """Verify one detached signature against the configured public trust root."""

    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "key_id",
        "value",
    }:
        raise ValueError("authenticated eval evidence signature is malformed")
    if signature.get("algorithm") != EVAL_EVIDENCE_SIGNATURE_ALGORITHM:
        raise ValueError("authenticated eval evidence signature algorithm is invalid")
    public_key = load_eval_evidence_public_key_from_broker()
    if signature.get("key_id") != eval_evidence_key_id(public_key):
        raise ValueError("authenticated eval evidence uses an unknown signing key")
    encoded = signature.get("value")
    if not isinstance(encoded, str):
        raise ValueError("authenticated eval evidence signature value is missing")
    try:
        raw_signature = b64decode(encoded.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError) as exc:
        raise ValueError(
            "authenticated eval evidence signature encoding is invalid"
        ) from exc
    if len(raw_signature) != 64:
        raise ValueError("authenticated eval evidence signature length is invalid")
    try:
        public_key.verify(
            raw_signature,
            canonical_signing_message(
                "eval_ed25519", canonical_evidence_bytes(payload)
            ),
        )
    except InvalidSignature as exc:
        raise ValueError("authenticated eval evidence signature is invalid") from exc


def eval_evidence_key_id(public_key: Ed25519PublicKey) -> str:
    """Return a stable identifier for one Ed25519 public key."""

    return f"sha256:{hashlib.sha256(_public_key_bytes(public_key)).hexdigest()}"


def scrub_attestation_signing_keys(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment without private keys or broker capabilities."""

    return scrub_private_signing_environment(environment)


def _public_key_bytes(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
