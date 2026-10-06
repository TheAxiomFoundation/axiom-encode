"""Mechanical-equivalence tests for receipt-backed apply verification."""

from __future__ import annotations

from base64 import b64decode, b64encode

import receipt.sign

from axiom_encode.cli import (
    APPLIED_ENCODING_SIGNATURE_ALGORITHM,
    _applied_encoding_manifest_signature_issue,
    _sign_applied_encoding_manifest,
    _unsigned_applied_encoding_manifest_bytes,
)
from axiom_encode.signing_broker import canonical_signing_message
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
    TEST_EVAL_PRIVATE_KEY_B64,
    TEST_EVAL_PUBLIC_KEY_B64,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

INVALID_SIGNATURE_ISSUE = "has an invalid encoder apply manifest signature"
TRUSTED_BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)


def _signed_payload() -> dict[str, object]:
    payload: dict[str, object] = {"schema_version": "test", "counter": 0}
    _sign_applied_encoding_manifest(payload, TRUSTED_BROKER)
    return payload


def _signature(payload: dict[str, object]) -> dict[str, object]:
    signature = payload["signature"]
    assert isinstance(signature, dict)
    return signature


def test_real_apply_sign_to_receipt_verify_round_trip() -> None:
    payload = _signed_payload()

    assert _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER) is None


def test_flipped_signature_byte_has_existing_invalid_issue() -> None:
    payload = _signed_payload()
    signature = _signature(payload)
    raw_signature = bytearray(b64decode(str(signature["value"]), validate=True))
    raw_signature[0] ^= 1
    signature["value"] = b64encode(raw_signature).decode("ascii")

    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == INVALID_SIGNATURE_ISSUE
    )


def test_flipped_payload_byte_has_existing_invalid_issue() -> None:
    payload = _signed_payload()
    payload["counter"] = 1

    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == INVALID_SIGNATURE_ISSUE
    )


def test_wrong_key_has_existing_invalid_issue_after_key_id_gate() -> None:
    payload = _signed_payload()
    trusted_key_id = _signature(payload)["key_id"]
    wrong_key_broker = SigningBrokerFixture(
        apply_private_key=TEST_EVAL_PRIVATE_KEY_B64,
        apply_public_key=TEST_EVAL_PUBLIC_KEY_B64,
    )
    _sign_applied_encoding_manifest(payload, wrong_key_broker)
    _signature(payload)["key_id"] = trusted_key_id

    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == INVALID_SIGNATURE_ISSUE
    )


def test_wrong_domain_scope_has_existing_invalid_issue() -> None:
    payload = _signed_payload()
    wrong_scope_broker = SigningBrokerFixture(
        eval_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        eval_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    _signature(payload)["value"] = b64encode(
        wrong_scope_broker.eval_ed25519_sign(
            _unsigned_applied_encoding_manifest_bytes(payload)
        )
    ).decode("ascii")

    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == INVALID_SIGNATURE_ISSUE
    )


def test_apply_verification_delegates_exact_inputs_to_receipt(monkeypatch) -> None:
    payload = _signed_payload()
    signature = _signature(payload)
    observed: dict[str, object] = {}
    real_verify_threshold = receipt.sign.verify_threshold

    def observing_verify_threshold(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return real_verify_threshold(*args, **kwargs)

    monkeypatch.setattr(receipt.sign, "verify_threshold", observing_verify_threshold)

    assert _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER) is None

    args = observed["args"]
    kwargs = observed["kwargs"]
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    unsigned_bytes, signatures, public_keys, keyring = args
    raw_public_key = TRUSTED_BROKER.apply_public_key_raw
    assert raw_public_key is not None
    assert unsigned_bytes == _unsigned_applied_encoding_manifest_bytes(payload)
    assert signatures == {"apply-root": b64decode(str(signature["value"]))}
    assert public_keys == {"apply-root": raw_public_key}
    assert keyring == receipt.sign.KeyringSpec(
        keys=(
            receipt.sign.KeySpec(
                key_id="apply-root",
                fingerprint=receipt.sign.raw_public_key_sha256(raw_public_key),
                scheme="raw-sha256",
            ),
        ),
        threshold=1,
    )
    assert kwargs == {
        "domain": canonical_signing_message("apply_ed25519", b""),
        "label": "encoder apply manifest",
        # The apply keyring declares no legacy generations; new-material
        # verification states that explicitly (receipt 0.3.0 requires it).
        "allow_legacy": False,
    }


def test_envelope_issue_strings_and_ordering_are_unchanged() -> None:
    payload: dict[str, object] = {"counter": 0}
    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == "is missing an encoder apply manifest signature"
    )

    payload["signature"] = None
    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == "is missing an encoder apply manifest signature"
    )

    payload["signature"] = {"algorithm": "unsupported"}
    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == "has a malformed encoder apply manifest signature"
    )

    payload["signature"] = {
        "algorithm": APPLIED_ENCODING_SIGNATURE_ALGORITHM,
        "key_id": "unused",
        "value": "unused",
        "extra": "unused",
    }
    assert (
        _applied_encoding_manifest_signature_issue(payload, TRUSTED_BROKER)
        == "has a malformed encoder apply manifest signature"
    )

    rootless_broker = SigningBrokerFixture()
    payload["signature"] = {
        "algorithm": "unsupported",
        "key_id": None,
        "value": None,
    }
    assert (
        _applied_encoding_manifest_signature_issue(payload, rootless_broker)
        == "uses an unsupported encoder apply manifest signature algorithm"
    )

    _signature(payload)["algorithm"] = APPLIED_ENCODING_SIGNATURE_ALGORITHM
    assert (
        _applied_encoding_manifest_signature_issue(payload, rootless_broker)
        == "cannot load the encoder apply manifest trust root: Trusted signing "
        "broker has no apply-manifest public key"
    )

    signed_payload = _signed_payload()
    signature = _signature(signed_payload)
    trusted_key_id = signature["key_id"]
    signature["key_id"] = "unknown"
    signature["value"] = None
    assert (
        _applied_encoding_manifest_signature_issue(signed_payload, TRUSTED_BROKER)
        == "uses an unknown encoder apply manifest signing key"
    )

    signature["key_id"] = trusted_key_id
    for invalid_value in (
        None,
        "not base64!",
        "snowman: \N{SNOWMAN}",
        b64encode(b"\x00" * 63).decode("ascii"),
    ):
        signature["value"] = invalid_value
        assert (
            _applied_encoding_manifest_signature_issue(
                signed_payload,
                TRUSTED_BROKER,
            )
            == INVALID_SIGNATURE_ISSUE
        )
