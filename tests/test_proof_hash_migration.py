from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import axiom_encode.proof_hash_migration as proof_hash_migration
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


def _proof_import_module(target: str, target_hash: str) -> str:
    return f"""format: rulespec/v1
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
              target: {target}#amount
              hash: sha256:{target_hash}
    versions:
      - effective_from: '2026-01-01'
        formula: amount
"""


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

    _commit(root, "apply proof hash cascade")
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

    with pytest.raises(ValueError, match="directly under .axiom/migrations"):
        apply_proof_hash_cascade(
            plan, Path(".axiom/migrations/nested/proof-hashes.json")
        )

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

    with pytest.raises(ValueError, match="canonical rulespec-<country>"):
        build_proof_hash_cascade_plan(nested, base)


def test_cascade_selects_only_rule_proof_import_atoms(tmp_path):
    root, base, target, importer = _migration_repo(tmp_path)
    content = importer.read_text()
    base_hash = content.split("hash: 'sha256:", 1)[1].split("'", 1)[0]
    distracting = f"""module:
  import:
    target: us-ak:policies/source#amount
    hash: sha256:{base_hash}
"""
    importer.write_text(content.replace("imports:\n", distracting + "imports:\n"))
    _commit(root, "add non-proof import mapping")
    target.write_text(target.read_text() + "# another migration\n")
    _commit(root, "migrate target again")

    plan = build_proof_hash_cascade_plan(root, base)

    assert len(plan.eligible) == 1
    apply_proof_hash_cascade(plan, Path(".axiom/migrations/proof-hashes.json"))
    updated = importer.read_text()
    assert f"    hash: sha256:{base_hash}\n" in updated
    assert f"hash: 'sha256:{_sha256(target)}'" in updated


def test_cascade_resolves_transitive_importers_to_final_hashes(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    leaf = module_root / "leaf.yaml"
    middle = module_root / "middle.yaml"
    top = module_root / "top.yaml"
    leaf.write_text("format: rulespec/v1\nrules: []\n")
    middle.write_text(_proof_import_module("us-ak:policies/leaf", _sha256(leaf)))
    top.write_text(_proof_import_module("us-ak:policies/middle", _sha256(middle)))
    base = _commit(root, "base import chain")
    leaf.write_text(leaf.read_text() + "# migrated\n")
    _commit(root, "migrate leaf")

    plan = build_proof_hash_cascade_plan(root, base)

    assert len(plan.eligible) == 2
    assert {item.importer_path for item in plan.eligible} == {
        "us-ak/policies/middle.yaml",
        "us-ak/policies/top.yaml",
    }
    apply_proof_hash_cascade(plan, Path(".axiom/migrations/transitive.json"))
    assert f"hash: sha256:{_sha256(leaf)}" in middle.read_text()
    assert f"hash: sha256:{_sha256(middle)}" in top.read_text()
    _commit(root, "apply transitive proof hash cascade")
    assert build_proof_hash_cascade_plan(root, base).eligible == ()


def test_cascade_rejects_new_staleness_for_non_base_authorized_import(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    leaf = module_root / "leaf.yaml"
    middle = module_root / "middle.yaml"
    top = module_root / "top.yaml"
    leaf.write_text("format: rulespec/v1\nrules: []\n")
    middle.write_text(_proof_import_module("us-ak:policies/leaf", _sha256(leaf)))
    top.write_text(_proof_import_module("us-ak:policies/middle", _sha256(middle)))
    base = _commit(root, "base import chain")
    leaf.write_text(leaf.read_text() + "# migrated\n")
    middle.write_text(middle.read_text() + "# independently changed\n")
    top.write_text(_proof_import_module("us-ak:policies/middle", _sha256(middle)))
    _commit(root, "advance target and dependent")

    with pytest.raises(ValueError, match="non-base-authorized proof import"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_duplicate_proof_import_keys(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    importer.write_text(
        importer.read_text().replace(
            "              hash: 'sha256:",
            "              target: us-ak:policies/source#duplicate\n"
            "              hash: 'sha256:",
            1,
        )
    )
    _commit(root, "add duplicate proof target")

    with pytest.raises(ValueError, match="contains duplicate key 'target'"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_tracked_symlink_importer(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    external = tmp_path / "external-importer.yaml"
    external.write_text(importer.read_text())
    importer.unlink()
    importer.symlink_to(external)
    _commit(root, "replace importer with symlink")

    with pytest.raises(ValueError, match="importer must not contain a symlink"):
        build_proof_hash_cascade_plan(root, base)

    assert "sha256:preexisting" in external.read_text()


def test_cascade_rejects_tracked_symlink_target(tmp_path):
    root, base, target, _importer = _migration_repo(tmp_path)
    external = tmp_path / "external-target.yaml"
    external.write_text(target.read_text())
    target.unlink()
    target.symlink_to(external)
    _commit(root, "replace target with symlink")

    with pytest.raises(
        ValueError, match="proof import target must not contain a symlink"
    ):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_ignored_target_missing_from_head(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    target = module_root / "ignored-target.yaml"
    importer = module_root / "importer.yaml"
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(
        _proof_import_module("us-ak:policies/ignored-target", _sha256(target))
    )
    (root / ".gitignore").write_text("us-ak/policies/ignored-target.yaml\n")
    base = _commit(root, "track importer but ignore target")

    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="proof import target is missing from HEAD"):
        build_proof_hash_cascade_plan(root, base)


@pytest.mark.parametrize(
    ("hidden_file", "message"),
    [
        ("target", "proof import target worktree bytes differ from HEAD"),
        ("importer", "importer worktree bytes differ from HEAD"),
    ],
)
def test_cascade_rejects_assume_unchanged_worktree_bytes(
    tmp_path, hidden_file, message
):
    root, base, target, importer = _migration_repo(tmp_path)
    path = target if hidden_file == "target" else importer
    path.write_text(path.read_text() + "# hidden worktree change\n")
    relative = path.relative_to(root).as_posix()
    _git(root, "update-index", "--assume-unchanged", relative)

    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match=message):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_proof_hash_alias_outside_import_field(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    target = module_root / "source.yaml"
    importer = module_root / "importer.yaml"
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(
        f"""format: rulespec/v1
shared_hash: &shared_hash sha256:{_sha256(target)}
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
              hash: *shared_hash
    versions:
      - effective_from: '2026-01-01'
        formula: amount
"""
    )
    base = _commit(root, "base aliased proof hash")
    target.write_text(target.read_text() + "# migrated\n")
    _commit(root, "migrate source")

    with pytest.raises(ValueError, match="YAML alias outside its field"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_directly_anchored_proof_hash(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    importer.write_text(
        importer.read_text().replace("hash: 'sha256:", "hash: &proof_hash 'sha256:")
    )
    _commit(root, "anchor proof hash")

    with pytest.raises(ValueError, match="contains YAML anchors or aliases"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_nested_import_mapping_alias(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    target = module_root / "source.yaml"
    importer = module_root / "importer.yaml"
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(
        f"""format: rulespec/v1
rules:
  - name: result
    kind: derived
    dtype: Money
    metadata:
      proof:
        atoms:
          - kind: custom
            shared_import: &shared_import
              target: us-ak:policies/source#amount
              hash: sha256:{_sha256(target)}
          - path: versions[0].formula
            kind: import
            import: *shared_import
    versions:
      - effective_from: '2026-01-01'
        formula: amount
"""
    )
    base = _commit(root, "base nested import alias")
    target.write_text(target.read_text() + "# migrated\n")
    _commit(root, "migrate source")

    with pytest.raises(ValueError, match="YAML alias outside its field"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_ancestor_anchor_inherited_by_yaml_merge(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    content = importer.read_text().replace(
        "    metadata:\n      proof:",
        "    metadata: &shared_metadata\n      proof:",
        1,
    )
    importer.write_text(
        content
        + """  - name: inherited_result
    kind: derived
    dtype: Money
    metadata:
      <<: *shared_metadata
    versions:
      - effective_from: '2026-01-01'
        formula: amount
"""
    )
    _commit(root, "inherit anchored metadata with YAML merge")

    with pytest.raises(ValueError, match="contains a YAML merge key"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_alias_used_as_mapping_key(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    content = importer.read_text().replace(
        "format: rulespec/v1\n",
        "format: rulespec/v1\nshared_key: &metadata_key metadata\n",
        1,
    )
    importer.write_text(content.replace("    metadata:\n", "    *metadata_key:\n", 1))
    _commit(root, "use alias as metadata mapping key")

    with pytest.raises(ValueError, match="contains a YAML alias mapping key"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_alias_used_as_rule_sequence_item(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "us-ak/policies"
    module_root.mkdir(parents=True)
    target = module_root / "source.yaml"
    importer = module_root / "importer.yaml"
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(
        f"""format: rulespec/v1
rules:
  - name: anchor_holder
    shared_rule: &shared_rule
      name: result
      kind: derived
      dtype: Money
      metadata:
        proof:
          atoms:
            - path: versions[0].formula
              kind: import
              import:
                target: us-ak:policies/source#amount
                hash: sha256:{_sha256(target)}
      versions:
        - effective_from: '2026-01-01'
          formula: amount
  - *shared_rule
"""
    )
    base = _commit(root, "base rule sequence alias")
    target.write_text(target.read_text() + "# migrated\n")
    _commit(root, "migrate source")

    with pytest.raises(ValueError, match="YAML alias outside its sequence item"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_ignores_importers_outside_checkout_country(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    module_root = root / "ca/policies"
    module_root.mkdir(parents=True)
    target = module_root / "source.yaml"
    importer = module_root / "importer.yaml"
    target.write_text("format: rulespec/v1\nrules: []\n")
    importer.write_text(_proof_import_module("ca:policies/source", _sha256(target)))
    base = _commit(root, "base foreign-country modules")
    target.write_text(target.read_text() + "# migrated\n")
    _commit(root, "migrate foreign-country target")

    plan = build_proof_hash_cascade_plan(root, base)

    assert plan.occurrences == ()


def test_cascade_rejects_cross_country_target(tmp_path):
    root, base, _target, importer = _migration_repo(tmp_path)
    importer.write_text(importer.read_text().replace("us-ak:policies", "ca:policies"))
    _commit(root, "add cross-country target")

    with pytest.raises(ValueError, match="does not belong to rulespec-us"):
        build_proof_hash_cascade_plan(root, base)


def test_cascade_rejects_existing_report(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    report = root / ".axiom/migrations/proof-hashes.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}\n")
    _commit(root, "reserve migration report")
    plan = build_proof_hash_cascade_plan(root, base)

    with pytest.raises(ValueError, match="must not already exist"):
        apply_proof_hash_cascade(plan, report)

    assert report.read_text() == "{}\n"


def test_cascade_rejects_symlink_checkout_alias(tmp_path):
    root, base, _target, _importer = _migration_repo(tmp_path)
    alias_parent = tmp_path / "alias"
    alias_parent.mkdir()
    alias = alias_parent / "rulespec-us"
    alias.symlink_to(root, target_is_directory=True)

    with pytest.raises(ValueError, match="canonical rulespec-<country>"):
        build_proof_hash_cascade_plan(alias, base)


def test_cascade_journals_and_rolls_back_when_importer_write_fails(
    tmp_path, monkeypatch
):
    root, base, _target, importer = _migration_repo(tmp_path)
    original = importer.read_bytes()
    report = root / ".axiom/migrations/proof-hashes.json"
    plan = build_proof_hash_cascade_plan(root, base)
    real_atomic_replace = proof_hash_migration._atomic_replace_bytes

    def fail_importer(path, content, *, mode):
        if path == importer and content != original:
            assert json.loads(report.read_text())["status"] == "prepared"
            raise OSError("simulated importer write failure")
        real_atomic_replace(path, content, mode=mode)

    monkeypatch.setattr(proof_hash_migration, "_atomic_replace_bytes", fail_importer)

    with pytest.raises(OSError, match="simulated importer write failure"):
        apply_proof_hash_cascade(plan, report)

    assert importer.read_bytes() == original
    audit = json.loads(report.read_text())
    assert audit["status"] == "rolled_back"
    assert audit["failure"] == "OSError: simulated importer write failure"


def test_cascade_fsyncs_each_created_audit_directory_parent(tmp_path, monkeypatch):
    root, base, _target, _importer = _migration_repo(tmp_path)
    plan = build_proof_hash_cascade_plan(root, base)
    observed: list[Path] = []
    real_fsync_directory = proof_hash_migration._fsync_directory

    def observe_fsync(path):
        observed.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(proof_hash_migration, "_fsync_directory", observe_fsync)

    apply_proof_hash_cascade(plan, Path(".axiom/migrations/proof-hashes.json"))

    assert root in observed
    assert root / ".axiom" in observed
    assert root / ".axiom/migrations" in observed


def test_cascade_rolls_back_when_report_finalization_fails(tmp_path, monkeypatch):
    root, base, _target, importer = _migration_repo(tmp_path)
    original = importer.read_bytes()
    report = root / ".axiom/migrations/proof-hashes.json"
    plan = build_proof_hash_cascade_plan(root, base)
    real_atomic_replace = proof_hash_migration._atomic_replace_bytes

    def fail_complete_report(path, content, *, mode):
        if path == report and json.loads(content)["status"] == "complete":
            raise OSError("simulated report finalization failure")
        real_atomic_replace(path, content, mode=mode)

    monkeypatch.setattr(
        proof_hash_migration, "_atomic_replace_bytes", fail_complete_report
    )

    with pytest.raises(OSError, match="simulated report finalization failure"):
        apply_proof_hash_cascade(plan, report)

    assert importer.read_bytes() == original
    audit = json.loads(report.read_text())
    assert audit["status"] == "rolled_back"
    assert audit["failure"] == "OSError: simulated report finalization failure"


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
