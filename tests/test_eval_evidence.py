"""Tests for broker-backed eval-suite evidence authentication."""

import json
import os
import sys
from base64 import b64encode

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.harness.eval_evidence import (
    APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV,
    EVAL_EVIDENCE_PRIVATE_KEY_ENV,
    isolated_eval_evidence_signer,
    load_eval_evidence_public_key_from_broker,
    scrub_attestation_signing_keys,
    sign_eval_evidence,
    verify_eval_evidence_signature,
)
from axiom_encode.harness.validator_pipeline import _run_subprocess_with_idle_timeout
from axiom_encode.signing_broker import SigningBrokerError
from tests.eval_evidence_fixtures import (
    TEST_EVAL_PRIVATE_KEY,
    TEST_EVAL_PRIVATE_KEY_B64,
    TEST_EVAL_PUBLIC_KEY_B64,
    install_test_eval_evidence_keys,
    make_test_eval_signing_broker,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

EVAL_EVIDENCE_PUBLIC_KEY_ENV = "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY"


def _install_public_key(monkeypatch) -> None:
    monkeypatch.setenv(EVAL_EVIDENCE_PUBLIC_KEY_ENV, TEST_EVAL_PUBLIC_KEY_B64)


def test_raw_keypair_signs_and_verifies_canonical_payload(monkeypatch):
    broker = make_test_eval_signing_broker()
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker", broker)
    payload = {"z": [3, 2, 1], "a": {"value": True}}
    signature = sign_eval_evidence(payload, broker)

    verify_eval_evidence_signature(
        {"a": {"value": True}, "z": [3, 2, 1]},
        signature,
    )


def test_pem_keypair_loads_in_explicit_test_broker(monkeypatch):
    private_pem = TEST_EVAL_PRIVATE_KEY.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        TEST_EVAL_PRIVATE_KEY.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    broker = SigningBrokerFixture(
        eval_private_key=private_pem,
        eval_public_key=public_pem,
    )
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker", broker)

    signature = sign_eval_evidence({"payload": "pem"}, broker)
    verify_eval_evidence_signature({"payload": "pem"}, signature)


def test_missing_protected_broker_fails_closed(monkeypatch):
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker", None)

    with pytest.raises(ValueError, match="protected signing broker"):
        load_eval_evidence_public_key_from_broker()


def test_mismatched_keypair_is_rejected():
    other_public = Ed25519PrivateKey.from_private_bytes(b"\x7a" * 32).public_key()
    other_public_b64 = b64encode(
        other_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode()

    with pytest.raises(SigningBrokerError, match="do not match"):
        SigningBrokerFixture(
            eval_private_key=TEST_EVAL_PRIVATE_KEY_B64,
            eval_public_key=other_public_b64,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("algorithm", "hmac-sha256", "algorithm"),
        ("algorithm", "ed25519", "algorithm"),
        ("key_id", "sha256:" + "0" * 64, "unknown signing key"),
        ("value", b64encode(b"\x00" * 64).decode(), "signature is invalid"),
        ("value", b64encode(b"\x00" * 63).decode(), "length"),
    ],
)
def test_invalid_signature_components_fail_closed(
    monkeypatch,
    field,
    value,
    expected_error,
):
    broker = make_test_eval_signing_broker()
    monkeypatch.setattr("axiom_encode.signing_broker._active_broker", broker)
    payload = {"payload": "authentic"}
    signature = sign_eval_evidence(payload, broker)
    signature[field] = value

    with pytest.raises(ValueError, match=expected_error):
        verify_eval_evidence_signature(payload, signature)


@pytest.mark.parametrize(
    ("key_kind", "value", "expected_error"),
    [
        ("private", b64encode(b"short").decode(), "32 raw bytes"),
        ("public", b64encode(b"short").decode(), "32 raw bytes"),
    ],
)
def test_invalid_broker_key_length_is_rejected(key_kind, value, expected_error):
    kwargs = {
        "eval_private_key": TEST_EVAL_PRIVATE_KEY_B64,
        "eval_public_key": TEST_EVAL_PUBLIC_KEY_B64,
    }
    kwargs[f"eval_{key_kind}_key"] = value
    with pytest.raises(SigningBrokerError, match=expected_error):
        SigningBrokerFixture(**kwargs)


def test_isolated_signer_uses_explicit_broker_without_private_environment(monkeypatch):
    broker = install_test_eval_evidence_keys(monkeypatch)
    monkeypatch.delenv(APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, raising=False)

    with isolated_eval_evidence_signer() as isolated:
        assert isolated is broker
        assert EVAL_EVIDENCE_PRIVATE_KEY_ENV not in os.environ
        assert APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV not in os.environ
        assert EVAL_EVIDENCE_PUBLIC_KEY_ENV not in os.environ
        signature = sign_eval_evidence({"inside": True}, isolated)
        verify_eval_evidence_signature({"inside": True}, signature)


def test_counterfeit_public_environment_cannot_override_broker(monkeypatch):
    broker = install_test_eval_evidence_keys(monkeypatch)
    other_public = Ed25519PrivateKey.from_private_bytes(b"\x7a" * 32).public_key()
    monkeypatch.setenv(
        EVAL_EVIDENCE_PUBLIC_KEY_ENV,
        b64encode(
            other_public.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).decode(),
    )

    with isolated_eval_evidence_signer() as isolated:
        assert isolated is broker
        signature = sign_eval_evidence({"protected": True}, isolated)
        verify_eval_evidence_signature({"protected": True}, signature)


def test_reviewer_oracle_subprocess_scrubs_keys_and_broker_markers(monkeypatch):
    _install_public_key(monkeypatch)
    monkeypatch.setenv(EVAL_EVIDENCE_PRIVATE_KEY_ENV, TEST_EVAL_PRIVATE_KEY_B64)
    monkeypatch.setenv(APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV, "private")
    sentinel_names = (
        "AWS_SECRET_ACCESS_KEY",
        "GH_TOKEN",
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "HTTPS_PROXY",
    )
    for name in sentinel_names:
        monkeypatch.setenv(name, f"{name}-sentinel")
    monkeypatch.setenv("PATH", "/hostile/bin")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/hostile/gitconfig")
    scrubbed = scrub_attestation_signing_keys()
    assert EVAL_EVIDENCE_PRIVATE_KEY_ENV not in scrubbed
    assert APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV not in scrubbed
    script = (
        "import json,os; print(json.dumps({"
        f"'eval_private': os.getenv('{EVAL_EVIDENCE_PRIVATE_KEY_ENV}'),"
        f"'apply_private': os.getenv('{APPLY_MANIFEST_SIGNING_PRIVATE_KEY_ENV}'),"
        "'sentinels': {name: os.getenv(name) for name in " + repr(sentinel_names) + "},"
        "'path': os.getenv('PATH'),"
        "'git_config': os.getenv('GIT_CONFIG_GLOBAL'),"
        "}, sort_keys=True))"
    )

    result = _run_subprocess_with_idle_timeout(
        [sys.executable, "-c", script],
        timeout=10,
        idle_timeout=10,
    )

    assert result.returncode == 0
    child = json.loads(result.output)
    assert child["eval_private"] is None
    assert child["apply_private"] is None
    assert child["sentinels"] == {name: None for name in sentinel_names}
    assert child["path"] != "/hostile/bin"
    assert child["git_config"] != "/hostile/gitconfig"
