"""Git's opaque directory entries must not hide changed provenance receipts."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from tests.test_cli import (
    TEST_PINNED_ENCODER_IDENTITY,
    _git,
    _test_eval_evidence_keys,
)
from tests.test_cli import TestGuardGenerated as _GuardFixtures

_INVENTORIES = ("encoding-manifests", "path-migrations", "legacy-replacements")


@st.composite
def _inventory_trees(draw):
    """Keep inventory roots disjoint, except for a single shared .axiom case."""

    above = draw(st.booleans())
    inventories = draw(
        st.lists(
            st.sampled_from(_INVENTORIES),
            min_size=1,
            max_size=1 if above else len(_INVENTORIES),
            unique=True,
        )
    )
    return [
        (
            inventory,
            draw(st.sampled_from(("regular", "gitlink", "symlink"))),
            "above"
            if above
            else draw(st.sampled_from(("at", "inside", "inside_json"))),
            draw(st.sampled_from(("add", "modify", "delete"))),
            draw(st.sampled_from(("both", "base", "head"))),
        )
        for inventory in inventories
    ]


def _init_git(repo):
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")


def _commit(repo, message):
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)


class TestGuardInventoryProperty:
    # These are the guard's authenticated corpus/model-manifest fixtures. No
    # signature, receipt verifier, corpus binding, or owner check is replaced.
    tmp_path = _GuardFixtures.tmp_path
    _rulespec_toolchain = _GuardFixtures._rulespec_toolchain
    _source_backed_rule = _GuardFixtures._source_backed_rule
    _attestation = _GuardFixtures._attestation
    _model_manifest = _GuardFixtures._model_manifest
    _receipt_only_manifest = _GuardFixtures._receipt_only_manifest
    _test_eval_evidence_keys = staticmethod(_test_eval_evidence_keys)

    @pytest.fixture(autouse=True)
    def _guard_identity(self, monkeypatch):
        monkeypatch.setattr(
            cli,
            "_read_only_guard_encoder_execution_identity",
            lambda *_: TEST_PINNED_ENCODER_IDENTITY,
        )

    @pytest.mark.parametrize("all_files", [False, True])
    @settings(
        max_examples=32,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @example(trees=[("encoding-manifests", "gitlink", "inside", "delete", "both")])
    @example(trees=[("path-migrations", "gitlink", "at", "add", "both")])
    @example(trees=[("legacy-replacements", "gitlink", "at", "add", "both")])
    @example(trees=[("encoding-manifests", "gitlink", "above", "modify", "both")])
    @example(trees=[("path-migrations", "gitlink", "inside", "modify", "base")])
    @example(trees=[("legacy-replacements", "gitlink", "inside", "modify", "head")])
    @example(trees=[("path-migrations", "symlink", "at", "add", "both")])
    @example(trees=[("legacy-replacements", "symlink", "inside", "modify", "both")])
    @example(trees=[("path-migrations", "symlink", "inside_json", "add", "both")])
    @example(
        trees=[("legacy-replacements", "symlink", "inside_json", "modify", "both")]
    )
    @example(trees=[("encoding-manifests", "symlink", "above", "modify", "head")])
    @example(trees=[("encoding-manifests", "regular", "at", "modify", "both")])
    @example(
        trees=[
            ("encoding-manifests", "regular", "inside", "modify", "both"),
            ("path-migrations", "gitlink", "at", "add", "both"),
            ("legacy-replacements", "symlink", "inside", "delete", "head"),
        ]
    )
    @given(trees=_inventory_trees())
    def test_changed_inventory_is_verified_or_refused(self, tmp_path, all_files, trees):
        """Actual Git discovery may never silently approve an opaque inventory."""

        # Hypothesis reuses function fixtures. Keep every base, index, nested
        # repository, and working tree isolated beneath pytest's basetemp.
        with tempfile.TemporaryDirectory(dir=tmp_path.parent) as temporary:
            repo = Path(temporary) / "rulespec-us"
            shutil.copytree(tmp_path / ".axiom", repo / ".axiom")
            shutil.copyfile(
                tmp_path / "known-validation-gaps.yaml",
                repo / "known-validation-gaps.yaml",
            )
            self._check_tree(repo, all_files, trees)

    def _check_tree(self, repo, all_files, trees):
        # An ordinary signed owner prevents --all's empty-corpus refusal from
        # concealing a receipt-only bypass in an auxiliary inventory.
        self._receipt_only_manifest(repo, stem="unchanged_baseline")
        cases = []
        expected_receipts = set()
        must_refuse = False
        for index, (inventory, kind, placement, mutation, side) in enumerate(trees):
            inventory_root = Path(".axiom") / inventory
            container_relative = (
                Path(".axiom")
                if placement == "above"
                else inventory_root
                / ("owners.json" if placement == "inside_json" else "owners")
                if placement.startswith("inside")
                else inventory_root
            )
            receipt_relative = (
                inventory_root if placement == "above" else container_relative
            ) / (
                f"owner_{index}.json"
                if inventory == "encoding-manifests"
                else "a" * 64 + ".json"
            )
            if inventory == "encoding-manifests":
                self._receipt_only_manifest(
                    repo,
                    stem=f"inventory_{index}",
                    manifest_relative=receipt_relative.as_posix(),
                )
            else:
                (repo / receipt_relative).parent.mkdir(parents=True, exist_ok=True)
                (repo / receipt_relative).write_text("{}\n")
            original = (repo / receipt_relative).read_bytes()
            if mutation == "add":
                (repo / receipt_relative).unlink()
            container = repo / container_relative
            # Nested repositories must have a commit even when their receipt
            # is introduced only at head.
            (container / ".inventory-placeholder").write_text("inventory fixture\n")
            target = repo / ".inventory-targets" / inventory
            if kind == "gitlink" and side in {"both", "base"}:
                _init_git(container)
                _commit(container, "initialize nested inventory")
            elif kind == "symlink" and side in {"both", "base"}:
                target.parent.mkdir(exist_ok=True)
                container.rename(target)
                container.symlink_to(target, target_is_directory=True)
            cases.append(
                (
                    container_relative,
                    receipt_relative,
                    original,
                    target,
                    kind,
                    mutation,
                    side,
                )
            )
            expected_receipts.add(receipt_relative.as_posix())
            must_refuse |= (
                kind != "regular"
                or inventory != "encoding-manifests"
                or mutation == "delete"
            )

        _init_git(repo)
        _commit(repo, "record provenance inventory base")
        base_ref = _git(repo, "rev-parse", "HEAD").stdout.strip()
        for (
            container_relative,
            receipt_relative,
            original,
            target,
            kind,
            mutation,
            side,
        ) in cases:
            container = repo / container_relative
            receipt = repo / receipt_relative
            if mutation == "delete":
                receipt.unlink()
            else:
                # Whitespace changes preserve the genuine model signature;
                # unsigned auxiliary evidence stays intentionally unsigned.
                receipt.write_bytes(original + b"\n")
            if kind == "gitlink":
                if side in {"both", "base"}:
                    _commit(container, "change nested provenance receipt")
                if side == "base":
                    _git(
                        repo,
                        "rm",
                        "--cached",
                        "-f",
                        "--",
                        container_relative.as_posix(),
                    )
                    shutil.rmtree(container / ".git")
                elif side == "head":
                    _git(
                        repo,
                        "rm",
                        "--cached",
                        "-r",
                        "--",
                        container_relative.as_posix(),
                    )
                    _init_git(container)
                    _commit(container, "introduce nested provenance inventory")
            elif kind == "symlink":
                if side == "base":
                    container.unlink()
                    target.rename(container)
                elif side == "head":
                    target.parent.mkdir(exist_ok=True)
                    container.rename(target)
                    container.symlink_to(target, target_is_directory=True)
        _commit(repo, "record provenance-only change")

        # Use the raw real-Git discovery here. Guarded discovery itself must
        # refuse opaque entries rather than pretending Git listed their files.
        assert cli._git_changed_files(repo, base_ref=base_ref, head_ref="HEAD")
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
            issues = cli.guard_generated_change_issues(
                repo,
                corpus_path=self.corpus_path,
                base_ref=base_ref,
                all_files=all_files,
            )

        inspected = {
            str(call.args[1])
            for calls in (manifests.call_args_list, migrations.call_args_list)
            for call in calls
        } | {call.kwargs["receipt_path"] for call in auxiliary.call_args_list}
        if must_refuse:
            assert issues, (trees, expected_receipts, inspected)
        else:
            assert issues == [], (trees, issues)
        assert issues or expected_receipts <= inspected, (
            trees,
            expected_receipts - inspected,
        )

    @pytest.mark.parametrize("all_files", [False, True])
    @pytest.mark.parametrize("inventory", ["path-migrations", "legacy-replacements"])
    def test_unstaged_symlink_inventory_is_refused(
        self, tmp_path, all_files, inventory
    ):
        """An untracked directory link must not disappear during classification."""

        self._receipt_only_manifest(tmp_path, stem="unchanged_baseline")
        target = tmp_path / "outside-provenance"
        target.mkdir()
        (target / ("a" * 64 + ".json")).write_text("{}\n")
        _init_git(tmp_path)
        _commit(tmp_path, "record tracked receipt outside inventory")
        base_ref = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
        relative = f".axiom/{inventory}"
        (tmp_path / relative).symlink_to(target, target_is_directory=True)

        # The target is already tracked and unchanged. Git sees only the new
        # non-.json link, so receipt suffix filtering cannot discover its bytes.
        assert cli._git_changed_files(tmp_path, base_ref=None, head_ref="HEAD") == [
            relative
        ]
        issues = cli.guard_generated_change_issues(
            tmp_path,
            corpus_path=self.corpus_path,
            base_ref=base_ref,
            all_files=all_files,
        )

        assert any(relative in issue and "symlink" in issue for issue in issues), issues
