"""Regression checks for reviewed successor-repoint guard boundaries."""

from __future__ import annotations

import json

import pytest

from axiom_encode.cli import guard_generated_change_issues
from tests.successor_repoint_fixtures import (
    DEPENDENT_MANIFEST,
    build_repoint_fixture,
    git,
    run_repoint,
)


@pytest.fixture
def landed_repoint(tmp_path, monkeypatch):
    fixture = build_repoint_fixture(tmp_path, monkeypatch)
    run_repoint(fixture)
    git(fixture.repo, "add", "-A")
    git(fixture.repo, "commit", "-qm", "repoint")
    fixture.base = git(fixture.repo, "rev-parse", "HEAD").strip()
    return fixture


def test_manifest_only_restoration_requires_its_repoint_receipt(landed_repoint):
    fixture = landed_repoint
    manifest = fixture.repo / DEPENDENT_MANIFEST
    original = manifest.read_bytes()
    manifest.unlink()
    git(fixture.repo, "commit", "-qam", "remove the old manifest")
    fixture.base = git(fixture.repo, "rev-parse", "HEAD").strip()
    manifest.write_bytes(original)

    issues = guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )

    assert any(
        "changed without introducing its successor repoint receipt" in i for i in issues
    ), issues


def test_manifest_only_signature_edit_is_verified(landed_repoint):
    fixture = landed_repoint
    manifest = fixture.repo / DEPENDENT_MANIFEST
    payload = json.loads(manifest.read_text())
    payload["signature"]["value"] = "A" * len(payload["signature"]["value"])
    manifest.write_text(json.dumps(payload, indent=2) + "\n")

    issues = guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )

    assert any("signature" in i for i in issues), issues


def test_receipt_only_change_is_refused(landed_repoint):
    fixture = landed_repoint
    receipt = next((fixture.repo / ".axiom/legacy-successor-repoints").iterdir())
    receipt.write_text(json.dumps(json.loads(receipt.read_text())) + "\n")

    issues = guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )

    assert any(
        "changed without the successor repoint it records" in i for i in issues
    ), issues


def _add_unrelated_signed_change(fixture):
    import hashlib

    from axiom_encode.cli import APPLIED_ENCODING_MANIFEST_SCHEMA
    from tests.successor_repoint_fixtures import SUCCESSOR, SUCCESSOR_MANIFEST
    from tests.test_cli import _signed_manifest_payload

    relative = "us/statutes/999.yaml"
    raw = (fixture.repo / SUCCESSOR).read_bytes()
    (fixture.repo / relative).write_bytes(raw)
    successor_manifest = json.loads((fixture.repo / SUCCESSOR_MANIFEST).read_text())
    payload = _signed_manifest_payload(
        {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "backend": "codex",
            "citation": successor_manifest["citation"],
            "validation_waiver_set_sha256": hashlib.sha256(
                (fixture.repo / "known-validation-gaps.yaml").read_bytes()
            ).hexdigest(),
            "source_attestation": successor_manifest["source_attestation"],
            "applied_files": [
                {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}
            ],
        }
    )
    manifest = fixture.repo / ".axiom/encoding-manifests/us/statutes/999.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )


@pytest.mark.parametrize(
    "promotion", [False, True], ids=["refresh", "reviewed-candidate"]
)
def test_manifest_only_valid_admission_is_preserved(landed_repoint, promotion):
    from axiom_encode.cli import (
        _REVIEWED_CANDIDATE_MANIFEST_FIELDS,
        APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL,
        _sign_applied_encoding_manifest,
    )
    from tests.successor_repoint_fixtures import BROKER

    fixture = landed_repoint
    assert _add_unrelated_signed_change(fixture) == []
    git(fixture.repo, "add", "-A")
    git(fixture.repo, "commit", "-qm", "unrelated signed module")
    fixture.base = git(fixture.repo, "rev-parse", "HEAD").strip()
    manifest = fixture.repo / ".axiom/encoding-manifests/us/statutes/999.json"
    payload = json.loads(manifest.read_text())
    payload["run_id"] = "manifest-only-refresh"
    if promotion:
        payload = {
            field: value
            for field, value in payload.items()
            if field in _REVIEWED_CANDIDATE_MANIFEST_FIELDS
        }
        payload["tool"] = APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
        payload["reviewed_rulespec_ref"] = fixture.base
    _sign_applied_encoding_manifest(payload, BROKER)
    manifest.write_text(json.dumps(payload, indent=2) + "\n")

    assert (
        guard_generated_change_issues(
            fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
        )
        == []
    )


def test_reformatting_landed_receipt_is_not_a_fresh_repoint(landed_repoint):
    fixture = landed_repoint
    assert _add_unrelated_signed_change(fixture) == []
    receipt = next((fixture.repo / ".axiom/legacy-successor-repoints").iterdir())
    receipt.write_text(json.dumps(json.loads(receipt.read_text())) + "\n")

    issues = guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )

    assert any(
        "existing successor repoint receipt and cannot be modified" in i for i in issues
    ), issues


def test_renaming_landed_receipt_is_refused(landed_repoint):
    fixture = landed_repoint
    assert _add_unrelated_signed_change(fixture) == []
    receipt = next((fixture.repo / ".axiom/legacy-successor-repoints").iterdir())
    receipt.rename(receipt.with_name("f" * 64 + ".json"))

    issues = guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.base
    )

    assert any(
        "successor repoint receipt and cannot be removed" in i for i in issues
    ), issues


def test_all_guard_does_not_reintroduce_landed_repoint_receipts(tmp_path, monkeypatch):
    import hashlib

    from axiom_encode.cli import (
        _line_preserving_yaml_mapping_removal,
        _sign_applied_encoding_manifest,
    )
    from tests.successor_repoint_fixtures import (
        BROKER,
        LEGACY,
        LEGACY_COMPANION,
        SUCCESSOR_MANIFEST,
    )

    def current_successor_without_waiver_transition(repo):
        # --all verifies unchanged model manifests against the current encoder
        # and waiver set too. Keep those independent bindings current so this
        # test isolates the landed-receipt change-set filter.
        waivers = repo / "known-validation-gaps.yaml"
        old_digest = hashlib.sha256(waivers.read_bytes()).hexdigest()
        raw, _count = _line_preserving_yaml_mapping_removal(
            waivers.read_bytes(), keys={LEGACY, LEGACY_COMPANION}
        )
        waivers.write_bytes(raw)
        new_digest = hashlib.sha256(raw).hexdigest()
        toolchain = repo / ".axiom/toolchain.toml"
        toolchain.write_text(toolchain.read_text().replace(old_digest, new_digest))
        manifest = repo / SUCCESSOR_MANIFEST
        payload = json.loads(manifest.read_text())
        payload["validation_waiver_set_sha256"] = new_digest
        payload["validation_execution"]["policy_pre_apply"][
            "validation_waiver_set_sha256"
        ] = new_digest
        _sign_applied_encoding_manifest(payload, BROKER)
        manifest.write_text(json.dumps(payload, indent=2) + "\n")

    fixture = build_repoint_fixture(
        tmp_path,
        monkeypatch,
        successor_encoder=None,
        before_commit=current_successor_without_waiver_transition,
    )
    run_repoint(fixture)
    git(fixture.repo, "add", "-A")
    git(fixture.repo, "commit", "-qm", "repoint with current model provenance")
    fixture.base = git(fixture.repo, "rev-parse", "HEAD").strip()
    (fixture.repo / "notes.txt").write_text("Unrelated later commit.\n")
    git(fixture.repo, "add", "notes.txt")
    git(fixture.repo, "commit", "-qm", "later unrelated change")

    issues = guard_generated_change_issues(
        fixture.repo,
        corpus_path=fixture.corpus,
        base_ref=fixture.base,
        all_files=True,
    )

    assert issues == [], issues


@pytest.mark.parametrize("version", [2, 3])
def test_recovery_refuses_jurisdictionless_manifest_writes(tmp_path, version):
    import os

    from axiom_encode.cli import (
        _APPLY_TRANSACTION_SCHEMA,
        _APPLY_TRANSACTION_SCHEMA_V3,
        _load_apply_transaction_journal,
    )

    repo = tmp_path / "rulespec-us"
    manifest = repo / ".axiom/encoding-manifests/policies/irs/legacy.json"
    manifest.parent.mkdir(parents=True)
    transaction = repo / ".axiom/.apply-transaction"
    transaction.mkdir(mode=0o700)
    (transaction / "backups").mkdir(mode=0o700)
    journal = {
        "schema": _APPLY_TRANSACTION_SCHEMA
        if version == 2
        else _APPLY_TRANSACTION_SCHEMA_V3,
        "state": "prepared",
        "entries": [
            {
                "path": manifest.relative_to(repo).as_posix(),
                "existed": False,
                "mode": 0o644,
                "old_sha256": None,
                "backup": None,
                "delete": False,
                "new_sha256": "0" * 64,
            }
        ],
        "created_directories": [],
    }
    if version == 3:
        journal["declared_program_specs"] = []
        journal["successor_repoint"] = True
    path = transaction / "journal.json"
    path.write_text(json.dumps(journal))
    os.chmod(path, 0o600)

    with pytest.raises(
        RuntimeError, match="not canonical|writes a jurisdiction-less manifest"
    ):
        _load_apply_transaction_journal(transaction, checkout_root=repo.resolve())


def test_legacy_replacement_refuses_to_rewrite_repoint_provenance(tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from axiom_encode.cli import _resolve_legacy_replacement_contract
    from tests.test_legacy_replacement import _legacy_checkout

    checkout, content_root, source = _legacy_checkout(tmp_path)
    receipt = checkout / ".axiom/legacy-successor-repoints" / ("a" * 64 + ".json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"legacy_primary": "us-la/statutes/47:32.yaml"}) + "\n"
    )
    git(checkout, "add", ".")
    git(checkout, "commit", "-qm", "persisted repoint provenance")

    with (
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        pytest.raises(
            ValueError,
            match="cannot rewrite persisted provenance.*legacy-successor-repoints",
        ),
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us-la/statutes/47:32.yaml"),
            destination_raw=Path("us-la/statutes/47/32.yaml"),
            policy_checkout_path=checkout,
            policy_repo_path=content_root,
            source_unit=source,
            corpus_release=SimpleNamespace(),
        )
