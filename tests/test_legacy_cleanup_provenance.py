"""Fail-closed authentication tests for cleanup immutable-base provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.cli import (
    APPLIED_ENCODING_LEGACY_MANIFEST_SCHEMA,
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    _legacy_cleanup_base_provenance_issues,
    _sign_applied_encoding_manifest,
)
from axiom_encode.legacy_cleanup_git import LegacyCleanupProvenanceRecord
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)


def _record(
    repo: Path,
    relative: Path,
    payload: dict[str, object],
) -> LegacyCleanupProvenanceRecord:
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return LegacyCleanupProvenanceRecord(
        path=relative,
        mode="100644",
        blob_oid="b" * 40,
        sha256=hashlib.sha256(raw).hexdigest(),
        raw=raw,
    )


def _issues(repo: Path, *records: LegacyCleanupProvenanceRecord) -> list[str]:
    return _legacy_cleanup_base_provenance_issues(
        repo,
        plan=SimpleNamespace(provenance_records=records),
        verifier=BROKER,
        expected_waiver_set_sha256="c" * 64,
        local_corpus_release=None,
    )


def test_cleanup_base_rejects_cryptographically_invalid_v5_manifest(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    payload: dict[str, object] = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
    }
    _sign_applied_encoding_manifest(payload, BROKER)
    payload["tool"] = "tampered-after-signing"
    record = _record(
        repo,
        Path(".axiom/encoding-manifests/be/statutes/unrelated.json"),
        payload,
    )

    issues = _issues(repo, record)

    assert any("invalid encoder apply manifest signature" in issue for issue in issues)


def test_cleanup_base_verifies_v5_against_its_historical_encoder_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    historical_identity = {
        "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
        "commit": "d" * 40,
        "version": "0.2.1000",
        "identity_source": "git",
    }
    payload: dict[str, object] = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "validation_execution": {"axiom_encode": historical_identity},
    }
    _sign_applied_encoding_manifest(payload, BROKER)
    record = _record(
        repo,
        Path(".axiom/encoding-manifests/be/statutes/unrelated.json"),
        payload,
    )
    captured: list[dict[str, object]] = []

    def verify(*_args, **kwargs):
        captured.append(dict(kwargs["expected_encoder_identity"]))
        return payload, "", record.sha256, []

    monkeypatch.setattr(
        "axiom_encode.cli._load_verified_applied_encoding_manifest_payload",
        verify,
    )

    assert _issues(repo, record) == []
    assert captured == [historical_identity]


@pytest.mark.parametrize(
    "relative",
    [
        Path(".axiom/path-migrations") / f"{'1' * 64}.json",
        Path(".axiom/legacy-replacements") / f"{'2' * 64}.json",
    ],
)
def test_cleanup_base_rejects_orphan_linked_receipt_classes(
    tmp_path: Path,
    relative: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    record = _record(repo, relative, {})

    issues = _issues(repo, record)

    assert any("orphan" in issue and relative.as_posix() in issue for issue in issues)


def _legacy_manual_payload(digest: str) -> dict[str, object]:
    return {
        "schema_version": APPLIED_ENCODING_LEGACY_MANIFEST_SCHEMA,
        "tool": "axiom-encode sign-applied-files",
        "backend": "manual",
        "runner": "manual-attestation",
        "applied_files": [
            {"path": "be/statutes/unrelated.yaml", "sha256": digest},
        ],
        "signature": {
            "algorithm": "hmac-sha256",
            "key_id": "historical-v1",
            "value": "opaque-untrusted-evidence",
        },
    }


def test_cleanup_base_admits_exact_unrelated_v1_shape_without_granting_credit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    primary = repo / "be/statutes/unrelated.yaml"
    primary.parent.mkdir(parents=True)
    primary.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    digest = hashlib.sha256(primary.read_bytes()).hexdigest()
    record = _record(
        repo,
        Path(".axiom/encoding-manifests/be/statutes/unrelated.json"),
        _legacy_manual_payload(digest),
    )

    assert _issues(repo, record) == []


def test_cleanup_base_admits_v1_manifest_with_bounded_supplemental_primary(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    main = repo / "be/statutes/main.yaml"
    supplemental = repo / "be/policies/supplemental.yaml"
    main.parent.mkdir(parents=True)
    supplemental.parent.mkdir(parents=True)
    main.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    supplemental.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    payload = _legacy_manual_payload(hashlib.sha256(main.read_bytes()).hexdigest())
    assert isinstance(payload["applied_files"], list)
    assert isinstance(payload["applied_files"][0], dict)
    payload["applied_files"][0]["path"] = "be/statutes/main.yaml"
    payload["applied_files"].append(
        {
            "path": "be/policies/supplemental.yaml",
            "sha256": hashlib.sha256(supplemental.read_bytes()).hexdigest(),
        }
    )
    record = _record(
        repo,
        Path(".axiom/encoding-manifests/be/statutes/main.json"),
        payload,
    )

    assert _issues(repo, record) == []


def test_cleanup_base_rejects_malformed_v1_signature_shape(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-be"
    primary = repo / "be/statutes/unrelated.yaml"
    primary.parent.mkdir(parents=True)
    primary.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    digest = hashlib.sha256(primary.read_bytes()).hexdigest()
    payload = _legacy_manual_payload(digest)
    assert isinstance(payload["signature"], dict)
    payload["signature"]["algorithm"] = "ed25519"
    record = _record(
        repo,
        Path(".axiom/encoding-manifests/be/statutes/unrelated.json"),
        payload,
    )

    issues = _issues(repo, record)

    assert any("unknown signature provenance" in issue for issue in issues)


def test_cleanup_base_rejects_malformed_historical_cleanup_receipt(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    relative = (
        Path(".axiom/legacy-rulespec-deletion-receipts")
        / f"{'3' * 64}.json"
    )
    record = _record(repo, relative, {})

    issues = _issues(repo, record)

    assert any("base provenance" in issue and "invalid" in issue for issue in issues)
