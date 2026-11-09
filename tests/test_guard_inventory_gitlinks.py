"""Real Git inventories cannot conceal provenance behind submodule pointers."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from axiom_encode import cli
from scripts.provision_verification_supervisor import _install_trusted_git_wrapper
from tests.environment_assertions import (
    environment_variable,
    hides_environment,
)
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

    @pytest.mark.parametrize("location", ["head", "immutable base", "index"])
    def test_gitlink_is_refused_through_production_trusted_git_allowlist(
        self, tmp_path, monkeypatch, location
    ):
        self._receipt_only_manifest(tmp_path)
        base_ref = self._initialize(tmp_path)
        gitlink = ".axiom/path-migrations"
        _git(
            tmp_path,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{base_ref},{gitlink}",
        )
        if location != "index":
            _git(tmp_path, "commit", "-m", "record provenance gitlink")
            if location == "immutable base":
                base_ref = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
            _git(tmp_path, "rm", "--cached", "--", gitlink)
            if location == "immutable base":
                _git(tmp_path, "commit", "-m", "remove provenance gitlink")

        trusted_git = shutil.which("git")
        assert trusted_git is not None
        tools = tmp_path / "trusted-tools"
        tools.mkdir()
        _install_trusted_git_wrapper(
            tools, Path(sys.executable).resolve(), Path(trusted_git).resolve()
        )
        monkeypatch.setenv(
            "PATH", str(tools) + os.pathsep + environment_variable("PATH")
        )

        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            changed_files=[],
        )
        assert issues == [
            f"{gitlink} is a gitlink in the {location}; provenance "
            "inventories must not be at, above, or inside submodules"
        ]
        with pytest.raises(RuntimeError) as discovery_refusal:
            cli._git_guard_changed_files(tmp_path, base_ref=base_ref, head_ref="HEAD")
        assert str(discovery_refusal.value) == issues[0]

    @pytest.mark.parametrize(
        "case",
        [
            "literal",
            "absent",
            "directory",
            "empty_directory",
            "executable",
            "symlink",
            "oversized",
        ],
    )
    def test_base_receipt_inspection_through_production_trusted_git_allowlist(
        self, tmp_path, monkeypatch, case
    ):
        relative = Path(".axiom/encoding-manifests/[owner]\t.json")
        receipt = tmp_path / relative
        receipt.parent.mkdir(parents=True)
        decoy = receipt.with_name("o\t.json")
        decoy.write_bytes(b'{"decoy": true}\n')
        payload = b'{"literal": true}\n'
        if case == "directory":
            receipt.mkdir()
            (receipt / "child").write_bytes(payload)
        elif case == "symlink":
            receipt.symlink_to(decoy.name)
        elif case not in {"absent", "empty_directory"}:
            receipt.write_bytes(
                b"x" * (4 * 1024 * 1024 + 1) if case == "oversized" else payload
            )
            if case == "executable":
                receipt.chmod(0o755)
        base_ref = self._initialize(tmp_path)
        if case == "empty_directory":
            # Porcelain does not stage empty directories, but submitted trees
            # can contain them. Construct one at the literal receipt pathname.
            def tree(entries):
                return cli.subprocess.run(
                    ["git", "-C", str(tmp_path), "mktree", "-z"],
                    input=entries,
                    capture_output=True,
                    check=True,
                ).stdout.strip()

            entry = (
                b"040000 tree " + tree(b"") + b"\t" + os.fsencode(relative.name) + b"\0"
            )
            for directory in reversed(relative.parts[:-1]):
                entry = (
                    b"040000 tree "
                    + tree(entry)
                    + b"\t"
                    + os.fsencode(directory)
                    + b"\0"
                )
            base_ref = _git(
                tmp_path,
                "commit-tree",
                tree(entry).decode("ascii"),
                "-p",
                base_ref,
                "-m",
                "record empty receipt directory",
            ).stdout.strip()

        trusted_git = shutil.which("git")
        assert trusted_git is not None
        tools = tmp_path / "trusted-tools"
        tools.mkdir()
        _install_trusted_git_wrapper(
            tools, Path(sys.executable).resolve(), Path(trusted_git).resolve()
        )
        monkeypatch.setenv(
            "PATH", str(tools) + os.pathsep + environment_variable("PATH")
        )

        real_run = cli.subprocess.run
        commands = []

        @hides_environment
        def record_run(command, *args, **kwargs):
            commands.append(command)
            return real_run(command, *args, **kwargs)

        monkeypatch.setattr(cli.subprocess, "run", record_run)
        raw, issues = cli._guard_base_receipt_bytes(tmp_path, base_ref, relative)
        if case == "literal":
            assert (raw, issues) == (payload, [])
            assert any("cat-file" in command for command in commands)
        else:
            assert raw is None
            assert issues == (
                []
                if case == "absent"
                else [
                    f"{relative.as_posix()} is not an exact bounded regular base receipt"
                ]
            )
            assert all("cat-file" not in command for command in commands)

    def test_trusted_git_wrapper_admits_only_exact_receipt_tree_query(self, tmp_path):
        (tmp_path / "owner.json").write_text("{}\n")
        base_ref = self._initialize(tmp_path)
        trusted_git = shutil.which("git")
        assert trusted_git is not None
        tools = tmp_path / "trusted-tools"
        tools.mkdir()
        wrapper = _install_trusted_git_wrapper(
            tools, Path(sys.executable).resolve(), Path(trusted_git).resolve()
        )
        arguments = ["ls-tree", "-r", "-t", "-l", "-z", "--full-tree", base_ref]

        def query(options):
            return cli.subprocess.run(
                [str(wrapper), "-C", str(tmp_path), *options], capture_output=True
            )

        admitted = query(arguments)
        assert admitted.returncode == 0, admitted.stderr
        assert b"\towner.json\0" in admitted.stdout
        for refused in (
            [*arguments, "--", "owner.json"],
            [*arguments, "--name-only"],
            [*arguments[:-1], "refs/heads/main"],
            [*arguments[:-1], "--help"],
            [*arguments[:2], "-l", "-t", *arguments[4:]],
            [arguments[0], "-t", *arguments[3:]],
            [*arguments[:4], "--full-tree", "-z", base_ref],
        ):
            result = query(refused)
            assert result.returncode == 126, refused
            assert result.stdout == b"", refused
            assert b"trusted git wrapper refused arguments for ls-tree" in result.stderr

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

        # Guard refs are resolved and frozen before any inventory query, so the
        # invalid ref is refused there, before it can approve anything.
        assert any(
            f"Cannot resolve guard {invalid_side}_ref 'nonexistent-inventory-ref'"
            in issue
            for issue in issues
        ), issues
