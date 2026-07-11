"""Tests-only in-process Ed25519 signer fixtures.

Production code contains only the out-of-process broker client. Keeping this
implementation under ``tests`` prevents mutable Python application code from
offering a private-key signing fallback.
"""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as BinasciiError

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from axiom_encode.signing_broker import SigningBrokerError, canonical_signing_message


def _private_key(value: str, *, label: str) -> Ed25519PrivateKey:
    text = value.strip().replace("\\n", "\n")
    if text.startswith("-----BEGIN"):
        try:
            loaded = serialization.load_pem_private_key(
                text.encode("utf-8"),
                password=None,
            )
        except (TypeError, ValueError) as exc:
            raise SigningBrokerError(f"{label} private key PEM is invalid") from exc
        if not isinstance(loaded, Ed25519PrivateKey):
            raise SigningBrokerError(f"{label} private key must be Ed25519")
        return loaded
    try:
        raw = b64decode(text.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError) as exc:
        raise SigningBrokerError(
            f"{label} private key must be PEM or base64-encoded raw bytes"
        ) from exc
    if len(raw) != 32:
        raise SigningBrokerError(
            f"{label} Ed25519 private key must contain 32 raw bytes"
        )
    return Ed25519PrivateKey.from_private_bytes(raw)


def _public_key(value: str, *, label: str) -> Ed25519PublicKey:
    text = value.strip().replace("\\n", "\n")
    if text.startswith("-----BEGIN"):
        try:
            loaded = serialization.load_pem_public_key(text.encode("utf-8"))
        except ValueError as exc:
            raise SigningBrokerError(f"{label} public key PEM is invalid") from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise SigningBrokerError(f"{label} public key must be Ed25519")
        return loaded
    try:
        raw = b64decode(text.encode("ascii"), validate=True)
    except (BinasciiError, UnicodeEncodeError) as exc:
        raise SigningBrokerError(
            f"{label} public key must be PEM or base64-encoded raw bytes"
        ) from exc
    if len(raw) != 32:
        raise SigningBrokerError(
            f"{label} Ed25519 public key must contain 32 raw bytes"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


class SigningBrokerFixture:
    """Explicit in-process signer available only to the test suite."""

    def __init__(
        self,
        *,
        apply_private_key: str | None = None,
        apply_public_key: str | None = None,
        eval_private_key: str | None = None,
        eval_public_key: str | None = None,
        corpus_release_public_key: str | None = None,
    ):
        self._apply_private_key = (
            _private_key(apply_private_key, label="Apply manifest")
            if apply_private_key
            else None
        )
        self._apply_public = self._validate_pair(
            label="Apply manifest",
            private_key=self._apply_private_key,
            public_text=apply_public_key,
        )
        self._eval_private_key = (
            _private_key(eval_private_key, label="Eval evidence")
            if eval_private_key
            else None
        )
        self._eval_public = self._validate_pair(
            label="Eval evidence",
            private_key=self._eval_private_key,
            public_text=eval_public_key,
        )
        self._corpus_release_public = self._validate_pair(
            label="Corpus release",
            private_key=None,
            public_text=corpus_release_public_key,
        )

    @staticmethod
    def _validate_pair(
        *,
        label: str,
        private_key: Ed25519PrivateKey | None,
        public_text: str | None,
    ) -> bytes | None:
        if private_key is not None and not public_text:
            raise SigningBrokerError(
                f"{label} public key is required with the test signer"
            )
        if not public_text:
            return None
        public_key = _public_key(public_text, label=label)
        public_raw = _raw_public_key(public_key)
        if (
            private_key is not None
            and _raw_public_key(private_key.public_key()) != public_raw
        ):
            raise SigningBrokerError(f"{label} private/public keys do not match")
        return public_raw

    @property
    def capabilities(self) -> frozenset[str]:
        capabilities: set[str] = set()
        if self._apply_private_key is not None:
            capabilities.add("apply_ed25519")
        if self._eval_private_key is not None:
            capabilities.add("eval_ed25519")
        return frozenset(capabilities)

    @property
    def apply_public_key_raw(self) -> bytes | None:
        return self._apply_public

    @property
    def eval_public_key_raw(self) -> bytes | None:
        return self._eval_public

    @property
    def corpus_release_public_key_raw(self) -> bytes | None:
        return self._corpus_release_public

    def apply_ed25519_sign(self, payload: bytes) -> bytes:
        if self._apply_private_key is None:
            raise SigningBrokerError("Apply manifest signer is not provisioned")
        return self._apply_private_key.sign(
            canonical_signing_message("apply_ed25519", payload)
        )

    def eval_ed25519_sign(self, payload: bytes) -> bytes:
        if self._eval_private_key is None:
            raise SigningBrokerError("Eval evidence signer is not provisioned")
        return self._eval_private_key.sign(
            canonical_signing_message("eval_ed25519", payload)
        )
