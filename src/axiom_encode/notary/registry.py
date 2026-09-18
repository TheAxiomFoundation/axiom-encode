"""V33 public-key registry validation; never establishes enrollment authority."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ._schema import canonical_object, decode_base64, digest, fields, lane_name
from .canonical import sha256_hex
from .refusal import Refusal

REGISTRY_PATH = ".axiom/notary/keys.json"
ROLES = (
    "actor",
    "admin-approver",
    "approver",
    "corpus-release",
    "notary",
    "producer",
    "review",
)


@dataclass(frozen=True, slots=True)
class KeyRegistry:
    """Validated public keys for a caller-authenticated base and lane.

    A validated registry alone says nothing about host custody, administrative
    approval, finalized-chain membership, or a claimed runtime identity.
    """

    lane: str
    body_sha256: str
    keys: Mapping[str, Mapping[str, Ed25519PublicKey]]


def parse_registry(
    raw: bytes,
    *,
    lane: str,
    notary_spki_sha256: str,
    legacy_apply_root: str,
    legacy_eval_root: str,
) -> KeyRegistry | Refusal:
    """Validate §5, using consumer/legacy pins supplied by the trusted caller."""
    refusal = Refusal("policy-invalid", REGISTRY_PATH, "invalid_key_registry")
    pins = (notary_spki_sha256, legacy_apply_root, legacy_eval_root)
    if not lane_name(lane) or not all(digest(pin) for pin in pins):
        return refusal
    if len(set(pins)) != len(pins):
        return refusal
    body = canonical_object(raw)
    if not fields(body, {"schema", "lane", *ROLES}):
        return refusal
    if body["schema"] != "axiom/notary-key-registry/v1" or body["lane"] != lane:
        return refusal

    seen = {legacy_apply_root, legacy_eval_root}
    roles = {}
    for role in ROLES:
        entries = body[role]
        if not isinstance(entries, list):
            return refusal
        keys = {}
        previous = ""
        for entry in entries:
            if not fields(entry, {"spki_sha256", "public_key_spki_der_base64"}):
                return refusal
            fingerprint = entry["spki_sha256"]
            if (
                not digest(fingerprint)
                or fingerprint <= previous
                or fingerprint in seen
            ):
                return refusal
            der = decode_base64(entry["public_key_spki_der_base64"])
            if der is None or sha256_hex(der) != fingerprint:
                return refusal
            try:
                public = serialization.load_der_public_key(der)
            except (ValueError, TypeError, UnsupportedAlgorithm):
                return refusal
            if not isinstance(public, Ed25519PublicKey):
                return refusal
            # Reject accepted-but-noncanonical DER encodings and trailing bytes.
            if (
                public.public_bytes(
                    serialization.Encoding.DER,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                != der
            ):
                return refusal
            seen.add(fingerprint)
            previous = fingerprint
            keys[fingerprint] = public
        roles[role] = MappingProxyType(keys)
    if set(roles["notary"]) != {notary_spki_sha256}:
        return refusal
    return KeyRegistry(lane, sha256_hex(raw), MappingProxyType(roles))
