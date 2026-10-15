"""Real Git inventories cannot conceal provenance behind submodule pointers."""

import shutil
from pathlib import Path

import pytest

from axiom_encode import cli
from tests.test_cli import (
    TEST_PINNED_ENCODER_IDENTITY,
    _git,
    _test_eval_evidence_keys,  # noqa: F401 -- installs the signed fixture keys
)
from tests.test_cli import TestGuardGenerated as GuardFixture


class TestGuardInventoryGitlinks:
    tmp_path = GuardFixture.tmp_path
    _rulespec_toolchain = GuardFixture._rulespec_toolchain
    _source_backed_rule = GuardFixture._source_backed_rule
    _attestation = GuardFixture._attestation
    _model_manifest = GuardFixture._model_manifest
    _receipt_only_manifest = GuardFixture._receipt_only_manifest

    @pytest.fixture(autouse=True)
    def _guard_identity(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_read_only_guard_encoder_execution_identity",
            lambda *_: TEST_PINNED_ENCODER_IDENTITY,
        )

    @staticmethod
    def _commit(repo: Path, message: str) -> str:
        _git(repo, "add", "--all")
        _git(repo, "commit", "-m", message)
        return _git(repo, "rev-parse", "HEAD").stdout.strip()

    def _initialize(self, repo: Path) -> str:
        _git(repo, "init")
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test User")
        _git(repo, "config", "core.filemode", "true")
        return self._commit(repo, "initialize authenticated checkout")

    def _stage_gitlink(self, repo: Path, relative: str) -> Path:
        """Use an initialized nested repository, without expanding a submodule."""

        nested = repo / relative
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "placeholder").write_text("initialized provenance inventory\n")
        _git(repo, "rm", "--cached", "-r", "--ignore-unmatch", "--", relative)
        self._initialize(nested)
        _git(repo, "add", "--", relative)
        listing = _git(repo, "ls-files", "--stage", "--", relative).stdout
        assert listing.startswith("160000 "), listing
        assert listing.rstrip().endswith("\t" + relative), listing
        return nested

    @staticmethod
    def _assert_gitlink_refusal(issues: list[str], relative: str) -> None:
        assert any(relative in issue and "gitlink" in issue for issue in issues), issues

    def test_deleted_sole_owner_inside_gitlink_is_rejected_in_diff(self, tmp_path):
        """Git exposes only the pointer after deletion of its sole signed owner."""

        gitlink = ".axiom/encoding-manifests/owners"
        relative, manifest_relative = self._receipt_only_manifest(
            tmp_path, manifest_relative=f"{gitlink}/example.json"
        )
        self._initialize(tmp_path)
        nested = self._stage_gitlink(tmp_path, gitlink)
        base_ref = self._commit(tmp_path, "record sole owner inside gitlink")
        baseline_issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            all_files=True,
        )

        (tmp_path / manifest_relative).unlink()
        self._commit(nested, "delete the only signed owner")
        self._commit(tmp_path, "advance receipt inventory pointer")

        assert cli._git_changed_files(tmp_path, base_ref=base_ref, head_ref="HEAD") == [
            gitlink
        ]
        assert (tmp_path / relative).is_file()
        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
        )

        self._assert_gitlink_refusal(issues, gitlink)
        self._assert_gitlink_refusal(baseline_issues, gitlink)

    def test_unsigned_migration_inside_gitlink_is_rejected_in_all_with_base(
        self, tmp_path
    ):
        """Full-tree module coverage cannot reveal an unlisted auxiliary receipt."""

        self._receipt_only_manifest(tmp_path)
        self._initialize(tmp_path)
        gitlink = ".axiom/path-migrations"
        nested = self._stage_gitlink(tmp_path, gitlink)
        base_ref = self._commit(tmp_path, "record auxiliary receipt gitlink")
        receipt_relative = f"{gitlink}/{'a' * 64}.json"
        (tmp_path / receipt_relative).write_text("{}\n")
        self._commit(nested, "add unsigned canonical migration receipt")
        self._commit(tmp_path, "advance auxiliary inventory pointer")

        assert cli._git_changed_files(tmp_path, base_ref=base_ref, head_ref="HEAD") == [
            gitlink
        ]
        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            all_files=True,
        )

        self._assert_gitlink_refusal(issues, gitlink)

    @pytest.mark.parametrize(
        "gitlink",
        [
            ".axiom",
            ".axiom/encoding-manifests",
            ".axiom/encoding-manifests/owners",
            ".axiom/path-migrations",
            ".axiom/path-migrations/owners",
            ".axiom/legacy-replacements",
            ".axiom/legacy-replacements/owners",
        ],
    )
    @pytest.mark.parametrize("location", ["head_only", "base_only", "staged_only"])
    def test_gitlink_is_refused_at_above_or_inside_every_inventory(
        self, tmp_path, gitlink, location
    ):
        self._receipt_only_manifest(tmp_path)
        base_ref = self._initialize(tmp_path)
        nested = self._stage_gitlink(tmp_path, gitlink)
        if location != "staged_only":
            gitlink_ref = self._commit(tmp_path, "record provenance gitlink")
            if location == "base_only":
                base_ref = gitlink_ref
            _git(tmp_path, "rm", "--cached", "--", gitlink)
            shutil.rmtree(nested / ".git")
            _git(tmp_path, "add", "--all")
            if location == "base_only":
                self._commit(tmp_path, "replace inventory with regular files")
                regular_tree = _git(tmp_path, "ls-tree", "HEAD", "--", gitlink).stdout
                assert regular_tree.startswith("040000 "), regular_tree

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[],
            all_files=False,
        )

        self._assert_gitlink_refusal(issues, gitlink)
        with pytest.raises(RuntimeError) as discovery_refusal:
            cli._git_guard_changed_files(tmp_path, base_ref=base_ref, head_ref="HEAD")
        self._assert_gitlink_refusal([str(discovery_refusal.value)], gitlink)

    def test_gitlink_with_inventory_prefix_but_outside_inventory_is_accepted(
        self, tmp_path
    ):
        self._receipt_only_manifest(tmp_path)
        base_ref = self._initialize(tmp_path)
        gitlink = ".axiom/path-migrations-extra"
        self._stage_gitlink(tmp_path, gitlink)
        self._commit(tmp_path, "record unrelated nested checkout")

        assert cli._git_guard_changed_files(
            tmp_path, base_ref=base_ref, head_ref="HEAD"
        ) == [gitlink]
        for all_files in (False, True):
            assert (
                cli.guard_generated_change_issues(
                    tmp_path,
                    corpus_path=self.corpus_path,
                    base_ref=base_ref,
                    all_files=all_files,
                )
                == []
            )

    @pytest.mark.parametrize(
        "mode_source", ["index_only", "head_only", "filesystem_only"]
    )
    def test_executable_receipt_is_rejected_in_each_head_representation(
        self, tmp_path, mode_source
    ):
        _relative, manifest_relative = self._receipt_only_manifest(tmp_path)
        receipt = tmp_path / manifest_relative
        base_ref = self._initialize(tmp_path)
        receipt.chmod(0o755)
        if mode_source != "filesystem_only":
            _git(tmp_path, "add", "--", manifest_relative)
            if mode_source == "head_only":
                self._commit(tmp_path, "record executable signed receipt")
            receipt.chmod(0o644)
            if mode_source == "head_only":
                _git(tmp_path, "add", "--", manifest_relative)

        head_mode = _git(
            tmp_path, "ls-tree", "HEAD", "--", manifest_relative
        ).stdout.split()[0]
        index_mode = _git(
            tmp_path, "ls-files", "--stage", "--", manifest_relative
        ).stdout.split()[0]
        assert head_mode == ("100755" if mode_source == "head_only" else "100644")
        assert index_mode == ("100755" if mode_source == "index_only" else "100644")
        assert receipt.stat().st_mode & 0o777 == (
            0o755 if mode_source == "filesystem_only" else 0o644
        )

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[manifest_relative],
        )

        assert any(
            manifest_relative in issue and "100644" in issue for issue in issues
        ), issues

    @pytest.mark.parametrize("all_files", [False, True])
    def test_executable_receipt_is_rejected_before_the_next_legitimate_update(
        self, tmp_path, all_files
    ):
        _relative, manifest_relative = self._receipt_only_manifest(tmp_path)
        receipt = tmp_path / manifest_relative
        base_ref = self._initialize(tmp_path)
        assert (
            cli.guard_generated_change_issues(
                tmp_path,
                corpus_path=self.corpus_path,
                all_files=True,
            )
            == []
        )
        receipt.chmod(0o755)
        _git(tmp_path, "add", "--", manifest_relative)
        executable_entry = _git(
            tmp_path, "ls-files", "--stage", "--", manifest_relative
        ).stdout
        assert executable_entry.startswith("100755 "), executable_entry

        executable_issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            all_files=all_files,
        )

        # The rejected mode never becomes a legitimate immutable predecessor.
        # Reverting it permits this update and the following committed update.
        receipt.chmod(0o644)
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        _git(tmp_path, "add", "--", manifest_relative)
        assert (
            cli.guard_generated_change_issues(
                tmp_path,
                corpus_path=self.corpus_path,
                base_ref=base_ref,
                all_files=all_files,
            )
            == []
        )
        regular_base_ref = self._commit(tmp_path, "retain regular signed receipt")
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        assert (
            cli.guard_generated_change_issues(
                tmp_path,
                corpus_path=self.corpus_path,
                base_ref=regular_base_ref,
                all_files=all_files,
            )
            == []
        )
        assert any(
            manifest_relative in issue and "100644" in issue
            for issue in executable_issues
        ), executable_issues

    def test_unborn_head_still_checks_the_inventory_index(self, tmp_path):
        self._receipt_only_manifest(tmp_path)
        _git(tmp_path, "init")
        _git(tmp_path, "add", "--all")
        assert (
            cli.guard_generated_change_issues(
                tmp_path, corpus_path=self.corpus_path, changed_files=[]
            )
            == []
        )

        gitlink = ".axiom/path-migrations"
        self._stage_gitlink(tmp_path, gitlink)
        issues = cli.guard_generated_change_issues(
            tmp_path, corpus_path=self.corpus_path, changed_files=[]
        )

        self._assert_gitlink_refusal(issues, gitlink)

    @pytest.mark.parametrize("invalid_side", ["base", "head"])
    def test_explicit_invalid_ref_cannot_approve_an_empty_change_list(
        self, tmp_path, invalid_side
    ):
        self._receipt_only_manifest(tmp_path)
        self._initialize(tmp_path)
        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            changed_files=[],
            base_ref="nonexistent-inventory-ref" if invalid_side == "base" else None,
            head_ref="nonexistent-inventory-ref" if invalid_side == "head" else "HEAD",
        )

        location = "immutable base" if invalid_side == "base" else "head"
        assert any(
            f"Cannot inspect the {location} provenance inventory" in issue
            for issue in issues
        ), issues
