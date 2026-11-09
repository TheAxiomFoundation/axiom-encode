"""Base receipt ownership must survive changes to provenance alone."""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _sha256_file,
    _sign_applied_encoding_manifest,
    guard_generated_change_issues,
)
from tests.test_cli import (
    TEST_APPLY_SIGNING_BROKER,
    _git,
    _test_eval_evidence_keys,  # noqa: F401 -- installs the signed fixture keys
)
from tests.test_cli import (
    TestCmdEncode as EncodeFixture,
)
from tests.test_cli import (
    TestGuardGenerated as GuardFixture,
)


class TestGuardReceiptDeletions:
    tmp_path = GuardFixture.tmp_path
    _rulespec_toolchain = GuardFixture._rulespec_toolchain
    _source_backed_rule = GuardFixture._source_backed_rule
    _attestation = GuardFixture._attestation
    _model_manifest = GuardFixture._model_manifest
    _receipt_only_manifest = GuardFixture._receipt_only_manifest
    _retirement_manifest = GuardFixture._retirement_manifest

    def _base_receipt(self, repo: Path) -> tuple[str, str, str]:
        relative, manifest_relative = self._receipt_only_manifest(repo)
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "initialize signed provenance")
        base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
        return relative, manifest_relative, base_ref

    @pytest.mark.parametrize("removal", ["delete", "rename_outside_tree"])
    @pytest.mark.parametrize("discovery", ["explicit", "git"])
    def test_receipt_removal_cannot_drop_live_module_coverage(
        self, tmp_path, removal, discovery
    ):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        manifest = tmp_path / manifest_relative
        changed = [manifest_relative]
        if removal == "delete":
            manifest.unlink()
        else:
            destination = tmp_path / "unprotected-provenance.json"
            manifest.rename(destination)
            changed.append(destination.name)

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=changed if discovery == "explicit" else None,
        )

        assert any(
            relative in issue and "loses verified provenance" in issue
            for issue in issues
        ), issues

    def test_modified_receipt_cannot_drop_an_unchanged_module(self, tmp_path):
        relative, manifest_relative, _base_ref = self._base_receipt(tmp_path)
        retained = "us/regulations/retained.yaml"
        self._source_backed_rule(tmp_path / retained)
        manifest = tmp_path / manifest_relative
        payload = self._model_manifest(
            [
                {"path": relative, "sha256": _sha256_file(tmp_path / relative)},
                {"path": retained, "sha256": _sha256_file(tmp_path / retained)},
            ]
        )
        manifest.write_text(json.dumps(payload) + "\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "add shared provenance")
        base_ref = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
        payload["applied_files"] = payload["applied_files"][1:]
        _sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
        manifest.write_text(json.dumps(payload) + "\n")

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any(
            relative in issue and "loses verified provenance" in issue
            for issue in issues
        ), issues

    @pytest.mark.parametrize("replacement_state", ["surviving", "new"])
    def test_removed_receipt_can_use_verified_head_coverage(
        self, tmp_path, replacement_state
    ):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        replacement_relative = ".axiom/encoding-manifests/replacement.json"
        replacement = tmp_path / replacement_relative
        replacement.write_bytes((tmp_path / manifest_relative).read_bytes())
        if replacement_state == "surviving":
            _git(tmp_path, "add", replacement_relative)
            _git(tmp_path, "commit", "-m", "retain replacement provenance")
            base_ref = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
        (tmp_path / manifest_relative).unlink()

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative]
            + ([replacement_relative] if replacement_state == "new" else []),
        )

        assert (tmp_path / relative).is_file()
        assert issues == []

    def test_removed_receipt_requires_authenticated_retirement(self, tmp_path):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        (tmp_path / relative).unlink()
        (tmp_path / manifest_relative).unlink()

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any("authenticated retirement" in issue for issue in issues), issues

    def test_removed_receipt_accepts_authenticated_retirement(self, tmp_path):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        (tmp_path / relative).unlink()
        (tmp_path / manifest_relative).unlink()
        retirement_relative = ".axiom/encoding-manifests/retirement.json"
        retirement = tmp_path / retirement_relative
        retirement.write_text(json.dumps(self._retirement_manifest([relative])) + "\n")

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative, retirement_relative],
        )

        assert issues == []

    @pytest.mark.parametrize("fault", ["unsigned", "stale", "tampered"])
    def test_removed_receipt_rejects_unverified_replacement(self, tmp_path, fault):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        replacement_relative = ".axiom/encoding-manifests/replacement.json"
        self._receipt_only_manifest(
            tmp_path, fault=fault, manifest_relative=replacement_relative
        )
        (tmp_path / manifest_relative).unlink()

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert issues, relative

    def test_removed_receipt_rejects_overlapping_head_owners(self, tmp_path):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        original = (tmp_path / manifest_relative).read_bytes()
        (tmp_path / manifest_relative).unlink()
        for stem in ("replacement-one", "replacement-two"):
            (tmp_path / f".axiom/encoding-manifests/{stem}.json").write_bytes(original)

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any(
            relative in issue and "overlapping apply-manifest ownership" in issue
            for issue in issues
        ), issues

    @pytest.mark.parametrize("fault", ["unsigned", "schema", "tampered"])
    def test_deleted_base_receipt_itself_is_authenticated(self, tmp_path, fault):
        _relative, manifest_relative = self._receipt_only_manifest(
            tmp_path, fault=fault
        )
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "test@example.com")
        _git(tmp_path, "config", "user.name", "Test User")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initialize predecessor receipt")
        base_ref = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
        self._receipt_only_manifest(
            tmp_path, manifest_relative=".axiom/encoding-manifests/replacement.json"
        )
        (tmp_path / manifest_relative).unlink()

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any("immutable base" in issue for issue in issues), issues

    def test_manual_signed_receipt_cannot_replace_generated_ownership(self, tmp_path):
        relative, manifest_relative, base_ref = self._base_receipt(tmp_path)
        replacement_relative = ".axiom/encoding-manifests/replacement.json"
        replacement = tmp_path / replacement_relative
        payload = json.loads((tmp_path / manifest_relative).read_text())
        payload["backend"] = "manual"
        _sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
        replacement.write_text(json.dumps(payload) + "\n")
        (tmp_path / manifest_relative).unlink()

        issues = guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any("backend is not canonical" in issue for issue in issues), issues
        assert any(
            relative in issue and "loses verified provenance" in issue
            for issue in issues
        ), issues

    def test_supervised_legacy_replacement_preserves_guard_behavior(self, tmp_path):
        fixture_root = tmp_path.parent / "legacy-apply-fixture"
        fixture_root.mkdir()
        fixture = EncodeFixture()
        preflights = contextmanager(
            EncodeFixture._neutralize_encode_preflights.__wrapped__
        )
        with preflights(fixture, fixture_root):
            fixture.test_apply_freshly_replaces_untrusted_v1_at_same_canonical_path(
                fixture_root,
                publication_mutation=None,
                has_target_waiver=False,
            )
        checkout = fixture_root / "rulespec-us"

        issues = guard_generated_change_issues(
            checkout,
            corpus_path=fixture_root / "axiom-corpus",
            base_ref="HEAD",
        )

        assert issues == []
