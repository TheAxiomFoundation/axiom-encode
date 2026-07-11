"""Deterministic Ed25519 keys for eval-evidence tests."""

from base64 import b64encode

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.harness.eval_evidence import (
    EVAL_EVIDENCE_PRIVATE_KEY_ENV,
)
from axiom_encode.signing_broker import (
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

APPLY_MANIFEST_SIGNING_PUBLIC_KEY_ENV = "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY"
EVAL_EVIDENCE_PUBLIC_KEY_ENV = "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY"

TEST_EVAL_PRIVATE_KEY_BYTES = b"\x2d" * 32
TEST_EVAL_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(
    TEST_EVAL_PRIVATE_KEY_BYTES
)
TEST_EVAL_PRIVATE_KEY_B64 = b64encode(TEST_EVAL_PRIVATE_KEY_BYTES).decode("ascii")
TEST_EVAL_PUBLIC_KEY_B64 = b64encode(
    TEST_EVAL_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
).decode("ascii")
TEST_APPLY_PRIVATE_KEY_BYTES = b"\x11" * 32
TEST_APPLY_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(
    TEST_APPLY_PRIVATE_KEY_BYTES
)
TEST_APPLY_PRIVATE_KEY_B64 = b64encode(TEST_APPLY_PRIVATE_KEY_BYTES).decode("ascii")
TEST_APPLY_PUBLIC_KEY_B64 = b64encode(
    TEST_APPLY_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
).decode("ascii")


def make_test_eval_signing_broker(
    *,
    apply_private_key: str | None = None,
    apply_public_key: str | None = None,
) -> SigningBrokerFixture:
    """Build an explicit in-process broker for deterministic unit tests."""

    return SigningBrokerFixture(
        apply_private_key=apply_private_key,
        apply_public_key=apply_public_key,
        eval_private_key=TEST_EVAL_PRIVATE_KEY_B64,
        eval_public_key=TEST_EVAL_PUBLIC_KEY_B64,
    )


def install_test_eval_evidence_keys(
    monkeypatch,
    *,
    apply_private_key: str | None = None,
    apply_public_key: str | None = None,
) -> SigningBrokerFixture:
    """Install the deterministic public key and explicit test signing broker."""

    monkeypatch.delenv(EVAL_EVIDENCE_PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(EVAL_EVIDENCE_PUBLIC_KEY_ENV, raising=False)
    monkeypatch.delenv(APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, raising=False)
    monkeypatch.delenv(APPLY_MANIFEST_SIGNING_PUBLIC_KEY_ENV, raising=False)
    broker = make_test_eval_signing_broker(
        apply_private_key=apply_private_key,
        apply_public_key=apply_public_key,
    )
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker", broker)
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker_pid", None)
    return broker
