from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.cli import cmd_migration_cascade_proof_hashes
from axiom_encode.proof_hash_migration import (
    REPORT_SCHEMA,
    apply_proof_hash_cascade,
    build_proof_hash_cascade_plan,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _migration_repo(tmp_path: Path) -> tuple[Path, str, Path, Path]:
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    target = root / "us-ak/policies/source.yaml"
    importer = root / "us-ak/policies/importer.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(
        """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-ak/policy/source/block-1
rules:
  - name: amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        value: 1
"""
    )
    old_hash = _sha256(target)
    importer.write_text(
        f"""format: rulespec/v1
imports:
  - us-ak:policies/source#amount
rules:
  - name: result
    kind: derived
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us-ak:policies/source#amount
              output: amount
              hash: 'sha256:{old_hash}' # preserve this comment
          - path: versions[0].formula
            kind: import
            import:
              target: us-ak:policies/source#amount
              output: amount
              hash: sha256:preexisting
    versions:
      - effective_from: '2026-01-01'
        formula: amount
"""
    )
    base = _commit(root, "base")
    target.write_text(
        target.read_text().replace("rules:\n", "  summary: migrated\nrules:\n")
    )
    _commit(root, "migrate source metadata")
    return root, base, target, importer


def test_cascade_rewrites_only_exact_base_hash_and_preserves_format(tmp_path):
    root, base, target, importer = _migration_repo(tmp_path)

    plan = build_proof_hash_cascade_plan(root, base)

    assert len(plan.eligible) == 1
    assert plan.eligible[0].importer_path == "us-ak/policies/importer.yaml"
    assert plan.eligible[0].target_path == "us-ak/policies/source.yaml"
    assert [item.status for item in plan.ignored] == ["preexisting_mismatch"]

    report_path = Path(".axiom/migrations/proof-hashes.json")
    payload = apply_proof_hash_cascade(plan, report_path)
    content = importer.read_text()
    expected = f"sha256:{_sha256(target)}"

    assert f"hash: '{expected}' # preserve this comment" in content
    assert "hash: sha256:preexisting" in content
    assert payload["schema"] == REPORT_SCHEMA
    assert payload["eligible_replacements"] == 1
    assert payload["ignored_mismatches"] == 1
    assert payload["applied_files"] == [
        {
            "path": "us-ak/policies/importer.yaml",
            "before_sha256": payload["applied_files"][0]["before_sha256"],
            "after_sha256": _sha256(importer),
            "replacements": 1,
        }
    ]
    report = json.loads((root / report_path).read_text())
    assert report == payload

    post_apply = build_proof_hash_cascade_plan(root, base)
    assert post_apply.eligible == ()
    assert [item.status for item in post_apply.ignored] == ["preexisting_mismatch"]


def test_cascade_reports_target_added_after_base_without_rewriting(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    marker = root / "README.md"
    marker.write_text("base\n")
    base = _commit(root, "base")
    target = root / "us-ny/policies/new-source.yaml"
    importer = root / "us-ny/policies/importer.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(
        """format: rulespec/v1
rules:
  - name: result
    metadata:
      proof:
        atoms:
          - kind: import
            import:
              target: us-ny:policies/new-source#amount
              hash: sha256:old
    versions: []
"""
    )
    _commit(root, "add source and importer")

    plan = build_proof_hash_cascade_plan(root, base)

    assert plan.eligible == ()
    assert [item.status for item in plan.ignored] == ["base_target_missing"]


def test_cascade_requires_ancestor_base(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    _git(root, "checkout", "--orphan", "unrelated")
    _git(root, "rm", "-rf", ".")
    (root / "README.md").write_text("unrelated\n")
    _commit(root, "unrelated")

    with pytest.raises(ValueError, match="is not an ancestor of HEAD"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_apply_rejects_dirty_checkout(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    plan = build_proof_hash_cascade_plan(root, base)
    importer.write_text(importer.read_text() + "# dirty\n")

    with pytest.raises(ValueError, match="requires a clean Git checkout"):
        apply_proof_hash_cascade(plan, Path(".axiom/migrations/proof-hashes.json"))

    assert not (root / ".axiom/migrations/proof-hashes.json").exists()


def test_cascade_apply_rejects_changed_head(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    plan = build_proof_hash_cascade_plan(root, base)
    (root / "README.md").write_text("new head\n")
    _commit(root, "advance head")

    with pytest.raises(ValueError, match="plan HEAD changed before apply"):
        apply_proof_hash_cascade(plan, Path(".axiom/migrations/proof-hashes.json"))


def test_cascade_apply_rejects_report_path_escape_and_symlink(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    plan = build_proof_hash_cascade_plan(root, base)

    with pytest.raises(ValueError, match="under .axiom/migrations"):
        apply_proof_hash_cascade(plan, Path("../escaped.json"))

    migrations = root / ".axiom/migrations"
    migrations.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    report = migrations / "proof-hashes.json"
    report.symlink_to(outside)
    _commit(root, "track report symlink")
    symlink_plan = build_proof_hash_cascade_plan(root, base)

    with pytest.raises(ValueError, match="must not be a symlink"):
        apply_proof_hash_cascade(symlink_plan, report)

    assert not outside.exists()


def test_cascade_requires_exact_worktree_root(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    nested = root / "rulespec-us"
    nested.mkdir()

    with pytest.raises(ValueError, match="must be the Git worktree root"):
        build_proof_hash_cascade_plan(nested, base)


def test_cascade_command_check_apply_and_post_apply_check(tmp_path, capsys):
    root, base, _target, _importer = _migration_repo(tmp_path)
    args = SimpleNamespace(
        root=root,
        base_ref=base,
        check=True,
        apply=False,
        report=None,
        json=True,
    )

    assert cmd_migration_cascade_proof_hashes(args) == 1
    assert json.loads(capsys.readouterr().out)["eligible_replacements"] == 1

    args.check = False
    args.apply = True
    args.report = Path(".axiom/migrations/proof-hashes.json")
    assert cmd_migration_cascade_proof_hashes(args) == 0
    assert json.loads(capsys.readouterr().out)["eligible_replacements"] == 1

    _commit(root, "apply proof hash cascade")
    args.check = True
    args.apply = False
    args.report = None
    assert cmd_migration_cascade_proof_hashes(args) == 0
    assert json.loads(capsys.readouterr().out)["eligible_replacements"] == 0


def test_cascade_command_requires_report_only_for_apply(tmp_path, capsys):
    root, base, _target, _importer = _migration_repo(tmp_path)
    args = SimpleNamespace(
        root=root,
        base_ref=base,
        check=False,
        apply=True,
        report=None,
        json=False,
    )

    assert cmd_migration_cascade_proof_hashes(args) == 2
    assert "--report is required with --apply" in capsys.readouterr().err

    args.check = True
    args.apply = False
    args.report = Path(".axiom/migrations/unexpected.json")
    assert cmd_migration_cascade_proof_hashes(args) == 2
    assert "--report is only valid with --apply" in capsys.readouterr().err
