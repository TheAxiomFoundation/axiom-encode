"""Focused adversarial tests for legacy-cleanup receipt signing."""

from __future__ import annotations

import hashlib
from base64 import b64decode, b64encode
from copy import deepcopy

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from axiom_encode.legacy_cleanup import (
    LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
    LEGACY_CLEANUP_PROVENANCE_ASSERTIONS,
    LEGACY_CLEANUP_PROVENANCE_CLASS,
    LEGACY_CLEANUP_RECEIPT_SCHEMA,
    LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
    LEGACY_CLEANUP_SIGNATURE_DOMAIN,
    LEGACY_CLEANUP_TOOL,
    LEGACY_CLEANUP_VALIDATION_CHECKS,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    LegacyCleanupReceiptError,
    cleanup_signature_payload,
    receipt_identity_sha256,
    receipt_structure_issues,
    unsigned_receipt_bytes,
)
from axiom_encode.legacy_cleanup_signing import (
    legacy_cleanup_key_id,
    legacy_cleanup_signature_issue,
    sign_legacy_cleanup_receipt,
    verify_legacy_cleanup_receipt_signature,
)
from axiom_encode.signing_broker import (
    SigningBrokerError,
    canonical_signing_message,
)


def _raw_public_key(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


class _FakeApplyBroker:
    def __init__(
        self,
        private_key: Ed25519PrivateKey,
        *,
        advertised_public_key: bytes | None = None,
        raw_signature: bytes | None = None,
    ) -> None:
        self._private_key = private_key
        self.apply_public_key_raw = (
            _raw_public_key(private_key.public_key())
            if advertised_public_key is None
            else advertised_public_key
        )
        self.raw_signature = raw_signature
        self.payloads: list[bytes] = []

    def apply_ed25519_sign(self, payload: bytes) -> bytes:
        self.payloads.append(payload)
        if self.raw_signature is not None:
            return self.raw_signature
        return self._private_key.sign(
            canonical_signing_message("apply_ed25519", payload)
        )


def _file_evidence(path: str, marker: str, *, deletion: bool) -> dict[str, object]:
    evidence: dict[str, object] = {
        "path": path,
        "base_mode": "100644",
        "base_blob_oid": marker * 40,
        "base_sha256": marker * 64,
    }
    if deletion:
        evidence["result"] = "absent"
    return evidence


def _unsigned_payload() -> dict[str, object]:
    projected_tree = "3" * 40
    validation_checks = [
        {
            "name": name,
            "command": ["axiom-encode", "validate", name],
            "target_count": 1,
            "target_list_sha256": hashlib.sha256(
                f"targets:{name}".encode()
            ).hexdigest(),
            "exit_code": 0,
            "output_sha256": hashlib.sha256(
                f"output:{name}".encode()
            ).hexdigest(),
        }
        for name in LEGACY_CLEANUP_VALIDATION_CHECKS
    ]
    payload: dict[str, object] = {
        "schema_version": LEGACY_CLEANUP_RECEIPT_SCHEMA,
        "tool": LEGACY_CLEANUP_TOOL,
        "provenance_class": LEGACY_CLEANUP_PROVENANCE_CLASS,
        "generated_at": "2026-08-30T20:00:00Z",
        "reason": "Remove an exact unmanifested legacy aggregation",
        "repository": {
            "repository": "github.com/TheAxiomFoundation/rulespec-be",
            "object_format": "sha1",
            "base_commit": "1" * 40,
            "base_tree": "2" * 40,
            "projected_post_deletion_tree": projected_tree,
        },
        "toolchain": {
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "object_format": "sha1",
                "commit": "4" * 40,
                "version": "0.2.test",
            },
            "axiom_rules_engine": {
                "repository": (
                    "github.com/TheAxiomFoundation/axiom-rules-engine"
                ),
                "object_format": "sha1",
                "commit": "5" * 40,
            },
            "corpus_release": {
                "name": "legacy-cleanup-test",
                "content_sha256": "6" * 64,
                "selector_sha256": "7" * 64,
            },
            "validation_waiver_set_sha256": "8" * 64,
            "base_files": {
                "toolchain": _file_evidence(
                    ".axiom/toolchain.toml", "9", deletion=False
                ),
                "validation_waiver_set": _file_evidence(
                    "known-validation-gaps.yaml", "a", deletion=False
                ),
            },
        },
        "base_proof": {
            "schema": LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
            "ownership_inventory_sha256": "b" * 64,
            "provenance_record_count": 3,
            "surviving_reference_inventory_sha256": "c" * 64,
            "surviving_blob_count": 42,
        },
        "validation_execution": {
            "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
            "status": "passed",
            "engine_execution": True,
            "projected_post_deletion_tree": projected_tree,
            "checks": validation_checks,
        },
        "provenance_assertions": dict(LEGACY_CLEANUP_PROVENANCE_ASSERTIONS),
        "groups": [
            {
                "primary": _file_evidence(
                    "be/statutes/legacy.yaml", "d", deletion=True
                ),
                "companion": _file_evidence(
                    "be/statutes/legacy.test.yaml", "e", deletion=True
                ),
            }
        ],
    }
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    return payload


def _signed_payload() -> tuple[dict[str, object], _FakeApplyBroker]:
    payload = _unsigned_payload()
    broker = _FakeApplyBroker(Ed25519PrivateKey.from_private_bytes(b"\x11" * 32))
    sign_legacy_cleanup_receipt(payload, broker)
    return payload, broker


def _signature(payload: dict[str, object]) -> dict[str, object]:
    signature = payload["signature"]
    assert isinstance(signature, dict)
    return signature


def test_cleanup_signature_round_trip_uses_typed_inner_and_apply_outer() -> None:
    payload = _unsigned_payload()
    expected_inner = cleanup_signature_payload(payload)
    broker = _FakeApplyBroker(Ed25519PrivateKey.from_private_bytes(b"\x11" * 32))

    sign_legacy_cleanup_receipt(payload, broker)

    public_key = broker._private_key.public_key()
    signature = _signature(payload)
    assert broker.payloads == [expected_inner]
    assert signature == {
        "algorithm": LEGACY_CLEANUP_SIGNATURE_ALGORITHM,
        "key_id": (
            "sha256:"
            + hashlib.sha256(_raw_public_key(public_key)).hexdigest()
        ),
        "domain": LEGACY_CLEANUP_SIGNATURE_DOMAIN,
        "value": signature["value"],
    }
    assert signature["key_id"] == legacy_cleanup_key_id(public_key)
    assert receipt_structure_issues(payload) == []
    assert legacy_cleanup_signature_issue(payload, broker) is None
    verify_legacy_cleanup_receipt_signature(payload, public_key)
    public_key.verify(
        b64decode(str(signature["value"]), validate=True),
        canonical_signing_message("apply_ed25519", expected_inner),
    )


def test_signed_timestamp_tampering_is_a_cryptographic_failure() -> None:
    payload, broker = _signed_payload()
    payload["generated_at"] = "2026-08-30T20:00:01Z"

    assert (
        legacy_cleanup_signature_issue(payload, broker)
        == "legacy cleanup receipt cryptographic signature is invalid"
    )
    with pytest.raises(LegacyCleanupReceiptError, match="cryptographic"):
        verify_legacy_cleanup_receipt_signature(payload, broker)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("domain", "axiom-encode/applied-rulespec/v5", "domain is invalid"),
        ("algorithm", "ed25519", "algorithm is invalid"),
    ],
)
def test_wrong_envelope_contract_is_rejected(field, value, expected) -> None:
    payload, broker = _signed_payload()
    _signature(payload)[field] = value

    assert expected in str(legacy_cleanup_signature_issue(payload, broker))


def test_wrong_apply_key_is_rejected_before_crypto_verification() -> None:
    payload, _ = _signed_payload()
    wrong_public_key = Ed25519PrivateKey.from_private_bytes(b"\x22" * 32).public_key()

    assert (
        legacy_cleanup_signature_issue(payload, wrong_public_key)
        == "legacy cleanup receipt uses an unknown signing key"
    )


def test_apply_manifest_raw_payload_signature_cannot_replay() -> None:
    payload, broker = _signed_payload()
    replay = broker._private_key.sign(
        canonical_signing_message(
            "apply_ed25519",
            unsigned_receipt_bytes(payload),
        )
    )
    _signature(payload)["value"] = b64encode(replay).decode("ascii")

    assert "cryptographic signature is invalid" in str(
        legacy_cleanup_signature_issue(payload, broker)
    )


def test_eval_scope_signature_cannot_replay() -> None:
    payload, broker = _signed_payload()
    replay = broker._private_key.sign(
        canonical_signing_message(
            "eval_ed25519",
            cleanup_signature_payload(payload),
        )
    )
    _signature(payload)["value"] = b64encode(replay).decode("ascii")

    assert "cryptographic signature is invalid" in str(
        legacy_cleanup_signature_issue(payload, broker)
    )


@pytest.mark.parametrize(
    "encoded",
    [
        "not base64!",
        "snowman: \N{SNOWMAN}",
        b64encode(bytes(63)).decode("ascii"),
    ],
)
def test_malformed_signature_value_is_rejected(encoded: str) -> None:
    payload, broker = _signed_payload()
    _signature(payload)["value"] = encoded

    assert (
        legacy_cleanup_signature_issue(payload, broker)
        == "legacy cleanup receipt signature value is malformed"
    )


@pytest.mark.parametrize("public_key_raw", [None, b"short", "not-bytes"])
def test_malformed_broker_apply_key_is_rejected(public_key_raw) -> None:
    payload, _ = _signed_payload()

    class MalformedBroker:
        apply_public_key_raw = public_key_raw

    issue = legacy_cleanup_signature_issue(payload, MalformedBroker())

    assert issue is not None
    assert "trust root is invalid" in issue


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_signer_rejects_malformed_unsigned_schema_before_broker_call(
    mutation: str,
) -> None:
    payload = _unsigned_payload()
    if mutation == "missing":
        del payload["reason"]
    else:
        payload["unexpected"] = True
    broker = _FakeApplyBroker(Ed25519PrivateKey.from_private_bytes(b"\x11" * 32))

    with pytest.raises(LegacyCleanupReceiptError, match="malformed"):
        sign_legacy_cleanup_receipt(payload, broker)

    assert broker.payloads == []
    assert "signature" not in payload


def test_signer_rejects_a_preexisting_signature() -> None:
    payload = _unsigned_payload()
    payload["signature"] = {"value": "counterfeit"}
    broker = _FakeApplyBroker(Ed25519PrivateKey.from_private_bytes(b"\x11" * 32))

    with pytest.raises(LegacyCleanupReceiptError, match="must not contain"):
        sign_legacy_cleanup_receipt(payload, broker)

    assert broker.payloads == []


def test_signer_rejects_malformed_or_unverifiable_broker_output() -> None:
    payload = _unsigned_payload()
    malformed = _FakeApplyBroker(
        Ed25519PrivateKey.from_private_bytes(b"\x11" * 32),
        raw_signature=bytes(63),
    )

    with pytest.raises(SigningBrokerError, match="invalid cleanup"):
        sign_legacy_cleanup_receipt(payload, malformed)
    assert "signature" not in payload

    wrong_signer = _FakeApplyBroker(
        Ed25519PrivateKey.from_private_bytes(b"\x11" * 32),
        advertised_public_key=_raw_public_key(
            Ed25519PrivateKey.from_private_bytes(b"\x22" * 32).public_key()
        ),
    )
    with pytest.raises(SigningBrokerError, match="unverifiable"):
        sign_legacy_cleanup_receipt(payload, wrong_signer)
    assert "signature" not in payload


def test_issue_categories_separate_structure_envelope_and_crypto() -> None:
    payload, broker = _signed_payload()

    structurally_invalid = deepcopy(payload)
    del structurally_invalid["reason"]
    assert "structure is invalid" in str(
        legacy_cleanup_signature_issue(structurally_invalid, broker)
    )

    missing_envelope = deepcopy(payload)
    del missing_envelope["signature"]
    assert (
        legacy_cleanup_signature_issue(missing_envelope, broker)
        == "legacy cleanup receipt signature envelope is malformed"
    )

    cryptographically_invalid = deepcopy(payload)
    _signature(cryptographically_invalid)["value"] = b64encode(bytes(64)).decode(
        "ascii"
    )
    assert (
        legacy_cleanup_signature_issue(cryptographically_invalid, broker)
        == "legacy cleanup receipt cryptographic signature is invalid"
    )
