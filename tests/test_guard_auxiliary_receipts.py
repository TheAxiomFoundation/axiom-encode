"""Standalone changes to signed migration receipts retain trusted provenance."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from tests.test_cli import (
    TEST_APPLY_SIGNING_BROKER,
    TEST_PINNED_ENCODER_IDENTITY,
    _git,
    _test_eval_evidence_keys,
)
from tests.test_cli import (
    TestGuardGenerated as _GuardFixtures,
)


class TestGuardAuxiliaryReceipts:
    # Reuse the same authenticated corpus and pinned broker as the guard tests.
    tmp_path = _GuardFixtures.tmp_path
    _rulespec_toolchain = _GuardFixtures._rulespec_toolchain
    _source_backed_rule = _GuardFixtures._source_backed_rule
    _canonical_guard_repo = _GuardFixtures._canonical_guard_repo
    _attestation = _GuardFixtures._attestation
    _model_manifest = _GuardFixtures._model_manifest
    _path_migration_repo = _GuardFixtures._path_migration_repo
    _enable_path_migration = _GuardFixtures._enable_path_migration
    _test_eval_evidence_keys = staticmethod(_test_eval_evidence_keys)

    @pytest.fixture(autouse=True)
    def _guard_identity(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_read_only_guard_encoder_execution_identity",
            lambda *_: TEST_PINNED_ENCODER_IDENTITY,
        )

    @pytest.mark.parametrize("tree", ["path-migrations", "legacy-replacements"])
    @pytest.mark.parametrize("fault", ["unsigned", "tampered", "symlink", "dangling"])
    def test_standalone_auxiliary_receipt_is_verified(self, tmp_path, tree, fault):
        receipt_relative = f".axiom/{tree}/{'a' * 64}.json"
        receipt = tmp_path / receipt_relative
        receipt.parent.mkdir(parents=True)
        payload = {"tool": "untrusted receipt"}
        if fault == "unsigned":
            receipt.write_text(json.dumps(payload))
        elif fault == "tampered":
            cli._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
            payload["tool"] = "changed after signing"
            receipt.write_text(json.dumps(payload))
        else:
            target = tmp_path / "receipt-target.json"
            if fault == "symlink":
                target.write_text(json.dumps(payload))
            receipt.symlink_to(target)

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            changed_files=[receipt_relative],
        )

        expected = (
            "cannot be read safely" if fault in {"symlink", "dangling"} else "signature"
        )
        assert any(expected in issue for issue in issues), issues

    @pytest.mark.parametrize("tree", ["path-migrations", "legacy-replacements"])
    def test_git_discovered_auxiliary_receipt_is_verified(self, tmp_path, tree):
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "test@example.com")
        _git(tmp_path, "config", "user.name", "Test User")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initialize authenticated checkout")
        receipt = tmp_path / f".axiom/{tree}/{'a' * 64}.json"
        receipt.parent.mkdir()
        receipt.write_text("{}\n")
        _git(tmp_path, "add", ".")

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref="HEAD",
        )

        assert any("signature" in issue for issue in issues), issues

    @pytest.mark.parametrize("tree", ["path-migrations", "legacy-replacements"])
    def test_deleted_auxiliary_receipt_authenticates_base(self, tmp_path, tree):
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "test@example.com")
        _git(tmp_path, "config", "user.name", "Test User")
        receipt_relative = f".axiom/{tree}/{'a' * 64}.json"
        receipt = tmp_path / receipt_relative
        receipt.parent.mkdir()
        receipt.write_text("{}\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "record unauthenticated historical receipt")
        receipt.unlink()

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref="HEAD",
            changed_files=[receipt_relative],
        )

        assert any("signature" in issue for issue in issues), issues

    def _migrated_receipt(self, tmp_path, monkeypatch):
        repo, plan, _ = self._path_migration_repo(tmp_path)
        self._enable_path_migration(monkeypatch)
        cli.cmd_migrate_rulespec_paths(
            SimpleNamespace(
                policy_repo_path=repo,
                corpus_path=self.corpus_path,
                plan=plan,
            )
        )
        owners = cli._all_applied_encoding_manifest_paths(repo)
        assert len(owners) == 2
        owner = json.loads((repo / owners[0]).read_text())
        receipt_relative = owner["migration"]["receipt_path"]
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "record authenticated path migration")
        return repo, receipt_relative, owners

    @pytest.mark.parametrize("mutation", ["bytes", "delete", "rename"])
    def test_auxiliary_change_reverifies_every_surviving_manifest(
        self, tmp_path, monkeypatch, mutation
    ):
        repo, receipt_relative, owners = self._migrated_receipt(tmp_path, monkeypatch)
        receipt = repo / receipt_relative
        changed = [receipt_relative]
        if mutation == "bytes":
            # Payload/signature remain valid: each owner's exact-byte digest fails.
            receipt.write_bytes(receipt.read_bytes() + b"\n")
        elif mutation == "delete":
            receipt.unlink()
        else:
            destination = receipt.with_name("b" * 64 + ".json")
            receipt.rename(destination)
            changed.append(destination.relative_to(repo).as_posix())

        with patch.object(
            cli,
            "_load_verified_applied_encoding_manifest_payload",
            wraps=cli._load_verified_applied_encoding_manifest_payload,
        ) as loaded:
            issues = cli.guard_generated_change_issues(
                repo,
                corpus_path=self.corpus_path,
                base_ref="HEAD",
                changed_files=changed,
            )

        assert issues, "a changed auxiliary receipt invalidated surviving owners"
        assert set(owners) <= {call.args[1] for call in loaded.call_args_list}

    def test_valid_standalone_migration_receipt_is_accepted(
        self, tmp_path, monkeypatch
    ):
        repo, receipt_relative, _ = self._migrated_receipt(tmp_path, monkeypatch)
        assert (
            cli.guard_generated_change_issues(
                repo,
                corpus_path=self.corpus_path,
                base_ref="HEAD",
                changed_files=[receipt_relative],
            )
            == []
        )

    def _legacy_cascade_receipt(self, repo, *, normalized_binding=False):
        source = Path("us/statutes/26:1.yaml")
        destination = Path("us/statutes/26/1.yaml")
        dependent = Path("us/policies/dependent.yaml")
        retained_source = Path("us/statutes/26:2.yaml")
        retained_destination = Path("us/statutes/26/2.yaml")

        def write_manual_owner(relative):
            self._source_backed_rule(repo / relative)
            manifest = repo / cli._applied_encoding_manifest_path(relative)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "axiom-encode/applied-rulespec/v1",
                        "tool": "axiom-encode sign-applied-files",
                        "backend": "manual",
                        "runner": "manual-attestation",
                        "manual_exception": "historical fixture",
                        "applied_files": [
                            {
                                "path": relative.as_posix(),
                                "sha256": cli._sha256_file(repo / relative),
                            }
                        ],
                        "signature": {
                            "algorithm": "hmac-sha256",
                            "key_id": "historical-v1",
                            "value": "untrusted-historical-evidence",
                        },
                    }
                )
                + "\n"
            )

        legacy_files = (
            (source,) if normalized_binding else (source, dependent, retained_source)
        )
        for relative in legacy_files:
            write_manual_owner(relative)
        if not normalized_binding:
            with (repo / dependent).open("a") as stream:
                stream.write("imports:\n  - us:statutes/26:1#amount\n")
            dependent_manifest = repo / cli._applied_encoding_manifest_path(dependent)
            old = json.loads(dependent_manifest.read_text())
            old["applied_files"][0]["sha256"] = cli._sha256_file(repo / dependent)
            dependent_manifest.write_text(json.dumps(old) + "\n")
            self._source_backed_rule(repo / retained_destination)
            successor = self._model_manifest(
                [
                    {
                        "path": retained_destination.as_posix(),
                        "sha256": cli._sha256_file(repo / retained_destination),
                    }
                ]
            )
            successor["citation"] = "us/statute/26/1"
            cli._sign_applied_encoding_manifest(successor, TEST_APPLY_SIGNING_BROKER)
            successor_manifest = repo / cli._applied_encoding_manifest_path(
                retained_destination
            )
            successor_manifest.parent.mkdir(parents=True, exist_ok=True)
            successor_manifest.write_text(
                json.dumps(successor, indent=2, sort_keys=True) + "\n"
            )
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "record legacy and authenticated successor")
        contract = cli._resolve_legacy_replacement_contract(
            source_raw=source,
            destination_raw=destination,
            policy_checkout_path=repo,
            policy_repo_path=repo / "us",
            source_unit=cli.resolve_corpus_source_unit(
                "us/statute/26/1", self.corpus_release
            ),
            corpus_release=self.corpus_release,
            exact_dependent_paths=[] if normalized_binding else [dependent],
            retained_successor_paths=[] if normalized_binding else [retained_source],
        )
        raw = (repo / source).read_bytes()
        model = self._model_manifest(
            [
                {
                    "path": destination.as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ]
        )
        model["citation"] = "us/statute/26/1"
        model["validation_execution"]["schema"] = (
            cli._APPLY_VALIDATION_EXECUTION_SCHEMA_V2
        )
        model["validation_execution"]["rulespec_scope"] = (
            cli._APPLY_VALIDATION_SCOPE_REPLACEMENT
        )
        cli._sign_applied_encoding_manifest(model, TEST_APPLY_SIGNING_BROKER)
        model_path, model_raw, receipt_path, receipt_raw, linked = (
            cli._stage_signed_legacy_replacement_provenance(
                None,
                contract=contract,
                checkout_root=repo,
                content_root=repo / "us",
                planned={Path(*destination.parts[1:]): raw},
                model_manifest_relative=cli._applied_encoding_manifest_path(
                    destination
                ),
                model_manifest_bytes=(
                    json.dumps(model, indent=2, sort_keys=True) + "\n"
                ).encode(),
                signing_broker=TEST_APPLY_SIGNING_BROKER,
                axiom_encode_git=model["axiom_encode_git"],
                local_corpus_release=self.corpus_release,
            )
        )
        deleted = (source,) if normalized_binding else (source, retained_source)
        for relative in deleted:
            (repo / relative).unlink()
            (repo / cli._applied_encoding_manifest_path(relative)).unlink()
        (repo / destination).parent.mkdir(parents=True, exist_ok=True)
        (repo / destination).write_bytes(raw)
        for exact in contract.exact_dependents:
            for item in exact.live_files:
                (repo / item.path).write_bytes(item.raw)
        for relative, content in {
            model_path: model_raw,
            receipt_path: receipt_raw,
            **linked,
        }.items():
            (repo / relative).parent.mkdir(parents=True, exist_ok=True)
            (repo / relative).write_bytes(content)
        if normalized_binding:
            owner = json.loads((repo / model_path).read_text())
            owner["replacement"]["receipt_path"] = receipt_path.as_posix().replace(
                "/legacy-replacements/", "/legacy-replacements//"
            )
            cli._sign_applied_encoding_manifest(owner, TEST_APPLY_SIGNING_BROKER)
            (repo / model_path).write_text(json.dumps(owner) + "\n")
        owners = [model_path.as_posix(), *(path.as_posix() for path in linked)]
        assert len(owners) == (1 if normalized_binding else 3)
        for owner in owners:
            verified, _, _, issues = (
                cli._load_verified_applied_encoding_manifest_payload(
                    repo,
                    owner,
                    signing_broker=TEST_APPLY_SIGNING_BROKER,
                    expected_encoder_identity=TEST_PINNED_ENCODER_IDENTITY,
                    local_corpus_release=self.corpus_release,
                )
            )
            assert verified is not None, issues
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "record authenticated legacy cascade")
        return receipt_path.as_posix(), owners

    @pytest.mark.parametrize(
        "mutation", ["bytes", "delete", "rename", "normalized-delete"]
    )
    def test_legacy_auxiliary_change_reverifies_replacement_and_cascade_owners(
        self, tmp_path, mutation
    ):
        receipt_relative, owners = self._legacy_cascade_receipt(
            tmp_path, normalized_binding=mutation == "normalized-delete"
        )
        receipt = tmp_path / receipt_relative
        changed = [receipt_relative]
        if mutation == "bytes":
            receipt.write_bytes(receipt.read_bytes() + b"\n")
        elif mutation in {"delete", "normalized-delete"}:
            receipt.unlink()
        else:
            renamed = receipt.with_name("b" * 64 + ".json")
            receipt.rename(renamed)
            changed.append(renamed.relative_to(tmp_path).as_posix())
        with patch.object(
            cli,
            "_load_verified_applied_encoding_manifest_payload",
            wraps=cli._load_verified_applied_encoding_manifest_payload,
        ) as loaded:
            issues = cli.guard_generated_change_issues(
                tmp_path,
                corpus_path=self.corpus_path,
                base_ref="HEAD",
                changed_files=changed,
            )
        assert issues, "a changed legacy receipt invalidated its cascade owners"
        assert set(owners) <= {call.args[1] for call in loaded.call_args_list}
