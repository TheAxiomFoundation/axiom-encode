"""Case aliases must never hide a changed provenance receipt from the guard."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from scripts.provision_verification_supervisor import _install_trusted_git_wrapper
from tests.test_cli import (
    TEST_PINNED_ENCODER_IDENTITY,
    _test_eval_evidence_keys,  # noqa: F401 -- installs authenticated fixture keys
)
from tests.test_cli import TestGuardGenerated as GuardFixture

_INVENTORIES = ("encoding-manifests", "path-migrations", "legacy-replacements")
_RECEIPT_NAME = "c" * 64


def _system_git(repo, *arguments, input_bytes=None):
    # The supervisor requires a root-owned executable. Homebrew Git need not
    # satisfy that admission requirement on macOS; system Git does.
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        input=input_bytes,
        capture_output=True,
        check=True,
        timeout=30,
    )


@st.composite
def _casefold_receipt_paths(draw):
    """Vary every protected component, including Unicode case-fold aliases."""

    inventory = draw(st.sampled_from(_INVENTORIES))

    def variant(component):
        alternatives = []
        for character in component:
            choices = {character, character.upper()}
            if character == "s":
                choices.add("ſ")
            alternatives.append(st.sampled_from(sorted(choices)))
        return "".join(draw(st.tuples(*alternatives)))

    return "/".join(
        (
            variant(".axiom"),
            variant(inventory),
            "owners",
            _RECEIPT_NAME + variant(".json"),
        )
    )


class TestGuardInventoryCaseAliases:
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

    def _guard(self, repo, monkeypatch, *, allow_uninspectable=False, **options):
        tools = repo.parent / "case-alias-trusted-tools"
        if not tools.exists():
            tools.mkdir()
            _install_trusted_git_wrapper(
                tools, Path(sys.executable).resolve(), Path("/usr/bin/git")
            )
        with monkeypatch.context() as wrapped:
            wrapped.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
            issues = cli.guard_generated_change_issues(
                repo, corpus_path=self.corpus_path, **options
            )
        if not allow_uninspectable:
            assert not any("cannot inspect" in issue.casefold() for issue in issues), (
                issues
            )
        return issues

    def _canonical_base(self, repo):
        self._receipt_only_manifest(repo)
        for inventory in _INVENTORIES[1:]:
            directory = repo / ".axiom" / inventory
            directory.mkdir(exist_ok=True)
            (directory / "placeholder").write_text("canonical baseline\n")
        _system_git(repo, "init")
        _system_git(repo, "config", "user.email", "test@example.invalid")
        _system_git(repo, "config", "user.name", "Case Alias Regression")
        _system_git(repo, "config", "core.ignorecase", "false")
        _system_git(repo, "add", "--all")
        _system_git(repo, "commit", "-m", "authenticated canonical baseline")
        base_ref = _system_git(repo, "rev-parse", "HEAD").stdout.decode().strip()
        return base_ref

    @staticmethod
    def _stage_alias(repo, relative, base_ref, *, gitlink=False):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if gitlink:
            path.mkdir(exist_ok=True)
            mode, object_id = "160000", base_ref
        else:
            path.write_bytes(b"{}\n")
            path.chmod(0o644)
            mode = "100644"
            object_id = (
                _system_git(repo, "hash-object", "-w", "--stdin", input_bytes=b"{}\n")
                .stdout.decode()
                .strip()
            )
        # Index plumbing preserves the attack's exact spelling even where the
        # filesystem resolves it to an already existing canonical directory.
        _system_git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"{mode},{object_id},{relative}",
        )

    @staticmethod
    def _commit_alias(repo, relative, base_ref):
        _system_git(repo, "commit", "-m", "introduce provenance case alias")
        raw = _system_git(
            repo,
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACDMRT",
            "-z",
            base_ref,
            "HEAD",
        ).stdout
        assert relative in [os.fsdecode(path) for path in raw.split(b"\0") if path]

    @pytest.mark.parametrize("all_files", [False, True])
    @pytest.mark.parametrize("attack", ["unsigned_receipt", "gitlink"])
    @pytest.mark.parametrize(
        "inventory",
        [
            ".AXIOM",
            ".axiom/ENCODING-MANIFESTS",
            ".axiom/PATH-MIGRATIONS",
            ".axiom/LEGACY-REPLACEMENTS",
        ],
    )
    def test_reviewer_canonical_base_inventory_alias_probes(
        self, tmp_path, monkeypatch, all_files, attack, inventory
    ):
        """Retain all sixteen inventory probes, including eight gitlink controls."""

        base_ref = self._canonical_base(tmp_path)
        if inventory == ".AXIOM":
            inventory += (
                "/encoding-manifests"
                if attack == "unsigned_receipt"
                else "/path-migrations"
            )
        relative = (
            inventory
            + "/"
            + (_RECEIPT_NAME + ".json" if attack == "unsigned_receipt" else "opaque")
        )
        self._stage_alias(tmp_path, relative, base_ref, gitlink=attack == "gitlink")
        self._commit_alias(tmp_path, relative, base_ref)

        issues = self._guard(
            tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files
        )

        # Two manifest --all probes also refused before this fix, giving ten
        # passing controls alongside the reviewer's twelve failing probes.
        assert issues, (relative, all_files, attack)

    @pytest.mark.parametrize("all_files", [False, True])
    @pytest.mark.parametrize("inventory", _INVENTORIES)
    def test_reviewer_canonical_base_json_suffix_alias_probes(
        self, tmp_path, monkeypatch, all_files, inventory
    ):
        """The reviewer's remaining six probes have an uppercase JSON suffix."""

        base_ref = self._canonical_base(tmp_path)
        relative = f".axiom/{inventory}/{_RECEIPT_NAME}.JSON"
        self._stage_alias(tmp_path, relative, base_ref)
        self._commit_alias(tmp_path, relative, base_ref)

        issues = self._guard(
            tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files
        )

        assert issues, (relative, all_files)
        assert any(relative in issue for issue in issues), issues

    @pytest.mark.parametrize("all_files", [False, True])
    @pytest.mark.parametrize(
        "relative",
        [
            f".Axiom/encoding-manifests/{_RECEIPT_NAME}.json",
            f".axiom/Encoding-Manifests/{_RECEIPT_NAME}.json",
            f".axiom/Path-Migrations/{_RECEIPT_NAME}.json",
            f".axiom/Legacy-Replacements/{_RECEIPT_NAME}.json",
            *[f".axiom/{inventory}/{_RECEIPT_NAME}.Json" for inventory in _INVENTORIES],
            f".axiom/encoding-manifeſts/{_RECEIPT_NAME}.json",
            f".axiom/path-migrationſ/{_RECEIPT_NAME}.json",
            f".axiom/legacy-replacementſ/{_RECEIPT_NAME}.json",
            *[f".axiom/{inventory}/{_RECEIPT_NAME}.jſon" for inventory in _INVENTORIES],
        ],
    )
    def test_mixed_case_and_unicode_aliases_are_refused(
        self, tmp_path, monkeypatch, all_files, relative
    ):
        base_ref = self._canonical_base(tmp_path)
        self._stage_alias(tmp_path, relative, base_ref)
        self._commit_alias(tmp_path, relative, base_ref)

        issues = self._guard(
            tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files
        )

        assert any(relative in issue and "noncanonical" in issue for issue in issues), (
            issues
        )

    @pytest.mark.parametrize(
        "mode", ["diff", "all_with_base", "explicit_empty_changes"]
    )
    @pytest.mark.parametrize("location", ["base", "head", "index"])
    @pytest.mark.parametrize(
        "relative",
        [
            f".Axiom/path-migrations/{_RECEIPT_NAME}.json",
            f".axiom/legacy-replacements/{_RECEIPT_NAME}.Json",
            f".axiom/encoding-manifeſts/{_RECEIPT_NAME}.json",
        ],
    )
    def test_each_git_inventory_is_inspected_even_without_changed_paths(
        self, tmp_path, monkeypatch, mode, location, relative
    ):
        base_ref = self._canonical_base(tmp_path)
        self._stage_alias(tmp_path, relative, base_ref)
        if location in {"base", "head"}:
            self._commit_alias(tmp_path, relative, base_ref)
            if location == "base":
                base_ref = (
                    _system_git(tmp_path, "rev-parse", "HEAD").stdout.decode().strip()
                )
            _system_git(tmp_path, "update-index", "--force-remove", "--", relative)
        (tmp_path / relative).unlink()
        if location == "base":
            _system_git(tmp_path, "commit", "-m", "remove alias from head and index")
        options = {"base_ref": base_ref, "all_files": mode == "all_with_base"}
        if mode == "explicit_empty_changes":
            options["changed_files"] = []

        issues = self._guard(tmp_path, monkeypatch, **options)

        assert any(relative in issue for issue in issues), (location, mode, issues)

    @pytest.mark.parametrize("mode", ["diff", "all", "all_with_base", "changed_files"])
    @pytest.mark.parametrize(
        "relative",
        [
            f".Axiom/path-migrations/{_RECEIPT_NAME}.json",
            f".axiom/legacy-replacements/{_RECEIPT_NAME}.Json",
            f".axiom/encoding-manifeſts/{_RECEIPT_NAME}.json",
        ],
    )
    def test_untracked_case_alias_cannot_escape_any_guard_mode(
        self, tmp_path, monkeypatch, mode, relative
    ):
        base_ref = self._canonical_base(tmp_path)
        if relative.startswith(".Axiom/"):
            # Materialize the alias spelling on case-insensitive filesystems;
            # writing through .Axiom alone would retain the stored .axiom name.
            intermediate = tmp_path / "case-inventory-intermediate"
            (tmp_path / ".axiom").rename(intermediate)
            intermediate.rename(tmp_path / ".Axiom")
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        options = {"all_files": mode in {"all", "all_with_base"}}
        if mode != "all":
            options["base_ref"] = base_ref
        if mode == "changed_files":
            options["changed_files"] = [relative]

        issues = self._guard(tmp_path, monkeypatch, **options)

        assert any(
            relative.casefold() == issue.split(" ", 1)[0].casefold()
            or relative.casefold().startswith(
                issue.split(" ", 1)[0].casefold().rstrip("/") + "/"
            )
            for issue in issues
        ), (mode, issues)

    def test_explicit_changed_files_refuses_alias_without_git_repository(
        self, tmp_path, monkeypatch
    ):
        self._receipt_only_manifest(tmp_path)
        relative = f".axiom/legacy-replacements/{_RECEIPT_NAME}.Json"
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")

        issues = self._guard(tmp_path, monkeypatch, changed_files=[relative])

        assert any(relative in issue for issue in issues), issues

    @pytest.mark.parametrize("all_files", [False, True])
    def test_absent_explicit_alias_is_refused_without_git_repository(
        self, tmp_path, monkeypatch, all_files
    ):
        self._receipt_only_manifest(tmp_path)
        relative = f".axiom/legacy-replacements/{_RECEIPT_NAME}.Json"
        assert not (tmp_path / relative).exists()
        # The shared fixture initialises Git; remove it to model a checkout
        # without a repository. Guard refs are resolved before any file check,
        # so such a checkout is refused outright, which also refuses the alias.
        shutil.rmtree(tmp_path / ".git", ignore_errors=True)
        assert not (tmp_path / ".git").exists()

        issues = self._guard(
            tmp_path,
            monkeypatch,
            allow_uninspectable=True,
            changed_files=[relative],
            all_files=all_files,
        )

        assert issues, "a checkout without Git must be refused"
        assert any("Cannot resolve guard head_ref" in issue for issue in issues), issues

    @pytest.mark.parametrize("all_files", [False, True])
    def test_signed_receipt_cannot_hide_noncanonical_ancestor_suffix(
        self, tmp_path, monkeypatch, all_files
    ):
        base_ref = self._canonical_base(tmp_path)
        canonical = ".axiom/encoding-manifests/us/regulations/example.json"
        relative = ".axiom/encoding-manifests/owners.JSON/receipt.json"
        (tmp_path / relative).parent.mkdir()
        _system_git(tmp_path, "mv", "--", canonical, relative)
        self._commit_alias(tmp_path, relative, base_ref)

        issues = self._guard(
            tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files
        )

        assert any(relative in issue and "noncanonical" in issue for issue in issues), (
            issues
        )

    @pytest.mark.parametrize("all_files", [False, True])
    def test_legitimate_signed_json_receipt_update_still_passes(
        self, tmp_path, monkeypatch, all_files
    ):
        base_ref = self._canonical_base(tmp_path)
        relative = ".axiom/encoding-manifests/us/regulations/example.json"
        manifest = tmp_path / relative
        # Whitespace preserves the genuine signature while changing receipt
        # bytes. This exercises successful authentication through the wrapper.
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        _system_git(tmp_path, "add", "--", relative)
        self._commit_alias(tmp_path, relative, base_ref)

        assert (
            self._guard(tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files)
            == []
        )

    @pytest.mark.parametrize("all_files", [False, True])
    @pytest.mark.parametrize(
        "relative",
        [
            ".AXIOM-extra/path-migrations/receipt.JSON",
            ".axiom/Encoding-Manifests-extra/receipt.JSON",
            ".axiom/Path-Migrations-extra/receipt.JSON",
            ".axiom/Legacy-Replacements-extra/receipt.JSON",
            "notes/.AXIOM/path-migrations/receipt.JSON",
        ],
    )
    def test_casefold_comparison_preserves_component_boundaries(
        self, tmp_path, monkeypatch, all_files, relative
    ):
        base_ref = self._canonical_base(tmp_path)
        self._stage_alias(tmp_path, relative, base_ref)
        self._commit_alias(tmp_path, relative, base_ref)

        assert (
            self._guard(tmp_path, monkeypatch, base_ref=base_ref, all_files=all_files)
            == []
        )

    @pytest.mark.parametrize("all_files", [False, True])
    @settings(
        max_examples=16,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(relative=f".axiom/PATH-MIGRATIONS/{_RECEIPT_NAME}.json")
    @example(relative=f".axiom/legacy-replacements/{_RECEIPT_NAME}.JSON")
    @example(relative=f".axiom/encoding-manifeſts/{_RECEIPT_NAME}.json")
    @example(relative=f".Axiom/encoding-manifests/{_RECEIPT_NAME}.Json")
    @example(relative=f".axiom/path-migrations/{_RECEIPT_NAME}.jſon")
    @given(relative=_casefold_receipt_paths())
    def test_casefold_receipt_is_verified_or_refused(
        self, tmp_path, monkeypatch, all_files, relative
    ):
        # Hypothesis reuses fixtures. Each example gets its own canonical signed
        # base, Git objects, index, filesystem, and production trusted wrapper.
        with tempfile.TemporaryDirectory(dir=tmp_path.parent) as temporary:
            repo = Path(temporary) / "rulespec-us"
            shutil.copytree(tmp_path / ".axiom", repo / ".axiom")
            shutil.copyfile(
                tmp_path / "known-validation-gaps.yaml",
                repo / "known-validation-gaps.yaml",
            )
            base_ref = self._canonical_base(repo)
            self._stage_alias(repo, relative, base_ref)
            self._commit_alias(repo, relative, base_ref)
            with (
                patch.object(
                    cli,
                    "_load_verified_applied_encoding_manifest_payload",
                    wraps=cli._load_verified_applied_encoding_manifest_payload,
                ) as manifests,
                patch.object(
                    cli,
                    "_load_verified_path_migration_receipt",
                    wraps=cli._load_verified_path_migration_receipt,
                ) as migrations,
                patch.object(
                    cli,
                    "_guard_auxiliary_receipt_authentication_issues",
                    wraps=cli._guard_auxiliary_receipt_authentication_issues,
                ) as auxiliary,
            ):
                issues = self._guard(
                    repo, monkeypatch, base_ref=base_ref, all_files=all_files
                )

            inspected = {
                str(call.args[1])
                for calls in (manifests.call_args_list, migrations.call_args_list)
                for call in calls
            } | {call.kwargs["receipt_path"] for call in auxiliary.call_args_list}
            assert issues or relative in inspected, (relative, all_files, inspected)
            # The fixture is deliberately unsigned: actual authentication must
            # reject it even if a future policy chooses verification over refusal.
            assert issues, (relative, all_files, inspected)

    @pytest.mark.parametrize("location", ["base", "head"])
    @pytest.mark.parametrize(
        "relative",
        [".axiom/PATH-MIGRATIONS", ".axiom/encoding-manifests/owners.JSON"],
    )
    def test_empty_aliased_tree_is_refused_without_index_or_worktree_entry(
        self, tmp_path, monkeypatch, location, relative
    ):
        canonical_ref = self._canonical_base(tmp_path)
        tree = _system_git(tmp_path, "mktree", "-z", input_bytes=b"").stdout.strip()
        parts = Path(relative).parts
        # Rebuild only the ancestors of the empty alias, preserving every
        # canonical entry in the authenticated base and the current checkout.
        for depth in reversed(range(len(parts))):
            parent = "/".join(parts[:depth])
            source = f"{canonical_ref}:{parent}" if parent else canonical_ref
            entries = _system_git(tmp_path, "ls-tree", "-z", source).stdout
            name = os.fsencode(parts[depth])
            retained = b"".join(
                record + b"\0"
                for record in entries.split(b"\0")
                if record and record.partition(b"\t")[2] != name
            )
            tree = _system_git(
                tmp_path,
                "mktree",
                "-z",
                input_bytes=retained + b"040000 tree " + tree + b"\t" + name + b"\0",
            ).stdout.strip()
        alias_ref = (
            _system_git(
                tmp_path,
                "commit-tree",
                tree.decode("ascii"),
                "-p",
                canonical_ref,
                "-m",
                "record empty provenance alias tree",
            )
            .stdout.decode()
            .strip()
        )

        issues = self._guard(
            tmp_path,
            monkeypatch,
            base_ref=alias_ref if location == "base" else canonical_ref,
            head_ref=alias_ref if location == "head" else canonical_ref,
            changed_files=[],
        )

        assert any(relative in issue and "noncanonical" in issue for issue in issues), (
            issues
        )
