"""Regression boundaries for negative-provenance legacy cleanup receipts."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_DIR,
    APPLIED_ENCODING_RETIRE_TOOL,
    _all_applied_encoding_manifest_paths,
    _is_applied_encoding_manifest_path,
    _is_protected_rulespec_yaml_path,
    _manifest_census,
    _manifest_coverage_by_file,
    _resolve_legacy_replacement_contract,
    cmd_migrate_rulespec_paths,
    cmd_retire,
    signed_import_inventory,
)
from axiom_encode.constants import RULESPEC_ATOMIC_MODULE_ROOTS
from axiom_encode.legacy_cleanup import (
    LEGACY_CLEANUP_RECEIPT_DIR,
    LEGACY_CLEANUP_RECEIPT_SCHEMA,
    LEGACY_CLEANUP_TOOL,
)
from axiom_encode.legacy_replacement import (
    RECEIPT_DIR as LEGACY_REPLACEMENT_RECEIPT_DIR,
)
from axiom_encode.rulespec_path_migration import (
    RECEIPT_DIR as PATH_MIGRATION_RECEIPT_DIR,
)
from axiom_encode.run_log_export import build_manifest_index
from axiom_encode.supabase_sync import (
    find_apply_manifests,
    sync_applied_manifest_runs,
)

ROOT = Path(__file__).resolve().parents[1]
ATOMIC_ROOTS = tuple(sorted(RULESPEC_ATOMIC_MODULE_ROOTS))
RECEIPT_RELATIVE = LEGACY_CLEANUP_RECEIPT_DIR / f"{'a' * 64}.json"
IMMUTABLE_RECEIPT_DIRS = (
    PATH_MIGRATION_RECEIPT_DIR,
    LEGACY_REPLACEMENT_RECEIPT_DIR,
    LEGACY_CLEANUP_RECEIPT_DIR,
)


def _write_group(repo: Path, stem: str) -> tuple[Path, Path]:
    primary = repo / f"us/statutes/{stem}.yaml"
    companion = primary.with_name(f"{stem}.test.yaml")
    primary.parent.mkdir(parents=True, exist_ok=True)
    primary.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    companion.write_text("cases: []\n", encoding="utf-8")
    return primary, companion


def _write_cleanup_receipt_placeholder(repo: Path) -> Path:
    """Write census/transport bait; receipt validity belongs to cleanup guards."""

    receipt = repo / RECEIPT_RELATIVE
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps({"schema_version": LEGACY_CLEANUP_RECEIPT_SCHEMA}) + "\n",
        encoding="utf-8",
    )
    return receipt


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_reference_receipt(repo: Path, directory: Path) -> Path:
    receipt = repo / directory / f"{'a' * 64}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps({"historical_path": "us/statutes/26:1.yaml"}) + "\n",
        encoding="utf-8",
    )
    return receipt


def test_cleanup_receipt_is_not_an_apply_manifest_or_retirement_target(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    receipt = _write_cleanup_receipt_placeholder(repo)

    assert LEGACY_CLEANUP_RECEIPT_DIR != APPLIED_ENCODING_MANIFEST_DIR
    assert LEGACY_CLEANUP_TOOL != APPLIED_ENCODING_RETIRE_TOOL
    assert not _is_applied_encoding_manifest_path(
        RECEIPT_RELATIVE,
        roots=ATOMIC_ROOTS,
    )
    assert not _is_protected_rulespec_yaml_path(
        RECEIPT_RELATIVE,
        roots=ATOMIC_ROOTS,
    )
    assert _all_applied_encoding_manifest_paths(repo, roots=ATOMIC_ROOTS) == []
    assert _manifest_coverage_by_file(
        repo,
        [],
        roots=ATOMIC_ROOTS,
        expected_encoder_identity={
            "repository": "github.com/TheAxiomFoundation/axiom-encode",
            "commit": "b" * 40,
            "version": "test",
            "identity_source": "git",
        },
    ) == {}
    assert receipt.is_file()


def test_cleanup_only_reduces_live_unmanifested_census_without_encoder_credit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    first = _write_group(repo, "legacy-first")
    _write_group(repo, "legacy-second")

    before = _manifest_census(repo, roots=ATOMIC_ROOTS)
    assert (before["total"], before["encoder"], before["unmanifested"]) == (
        4,
        0,
        4,
    )

    _write_cleanup_receipt_placeholder(repo)
    for path in first:
        path.unlink()

    after = _manifest_census(repo, roots=ATOMIC_ROOTS)
    assert (after["total"], after["encoder"], after["unmanifested"]) == (
        2,
        0,
        2,
    )
    assert after["encoder_pct"] == 0.0


def test_cleanup_receipt_cannot_authorize_retire(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    primary, companion = _write_group(repo, "legacy")
    receipt = _write_cleanup_receipt_placeholder(repo)
    originals = {
        path: path.read_bytes() for path in (primary, companion, receipt)
    }
    monkeypatch.setattr(
        "axiom_encode.cli.load_rulespec_local_corpus_release",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli._require_applied_encoding_manifest_signer",
        lambda: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli.verify_rulespec_validation_waiver_set",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        "axiom_encode.cli._current_guard_encoder_execution_identity",
        lambda: {"commit": "b" * 40},
    )

    with pytest.raises(
        SystemExit,
        match="current apply manifest group is missing or invalid",
    ):
        cmd_retire(
            SimpleNamespace(
                paths=["us/statutes/legacy.yaml"],
                policy_repo_path=repo,
                corpus_path=tmp_path / "axiom-corpus",
                reason="must remain generated-only",
            )
        )

    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_cleanup_receipt_is_not_signed_import_eligibility_evidence(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _write_cleanup_receipt_placeholder(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(
        ValueError,
        match="canonical checkout-relative primary RuleSpec module",
    ):
        signed_import_inventory(
            repo,
            corpus_path=tmp_path / "axiom-corpus",
            base_ref=base,
            rulespec_paths=(RECEIPT_RELATIVE.as_posix(),),
        )


def test_unmanifested_cleanup_group_is_not_signed_import_eligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    _write_group(repo, "legacy")
    _write_cleanup_receipt_placeholder(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(
        "axiom_encode.cli.verify_rulespec_validation_waiver_set",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        "axiom_encode.cli.load_rulespec_local_corpus_release",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli._applied_encoding_manifest_verifier",
        lambda: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli._read_only_guard_encoder_execution_identity",
        lambda *_args, **_kwargs: {"commit": "b" * 40},
    )

    with pytest.raises(
        ValueError,
        match="signed import manifest .* is not exactly present",
    ):
        signed_import_inventory(
            repo,
            corpus_path=tmp_path / "axiom-corpus",
            base_ref=base,
            rulespec_paths=("us/statutes/legacy.yaml",),
        )


def test_cleanup_receipt_creates_no_run_log_or_supabase_apply_credit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _write_cleanup_receipt_placeholder(repo)

    assert build_manifest_index([repo]) == {}
    assert find_apply_manifests(repo) == []
    assert sync_applied_manifest_runs([repo], dry_run=True) == {
        "total": 0,
        "synced": 0,
        "failed": 0,
        "skipped": 0,
        "preserved": 0,
    }


def test_targeted_signed_reencode_does_not_transport_or_sign_cleanup() -> None:
    workflow = (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text(
        encoding="utf-8"
    )

    assert LEGACY_CLEANUP_TOOL.removeprefix("axiom-encode ") not in workflow
    assert LEGACY_CLEANUP_RECEIPT_DIR.as_posix() not in workflow
    assert LEGACY_CLEANUP_RECEIPT_SCHEMA not in workflow


def test_every_path_rewrite_exclusion_preserves_all_receipt_classes() -> None:
    """Future migrations must fail, never rewrite any signed receipt bytes."""

    source = (ROOT / "src/axiom_encode/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    required = {
        "APPLIED_ENCODING_PATH_MIGRATION_RECEIPT_DIR",
        "APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_DIR",
        "LEGACY_CLEANUP_RECEIPT_DIR",
    }
    exclusions: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        if not target_names.intersection({"receipt_prefixes", "provenance_prefixes"}):
            continue
        exclusions.append(
            {child.id for child in ast.walk(node.value) if isinstance(child, ast.Name)}
        )

    assert len(exclusions) >= 3
    assert all(required <= names for names in exclusions)
    assert "Prior migration receipt still references a moved identity" in source
    assert "legacy replacement cannot rewrite persisted provenance" in source


@pytest.mark.parametrize("receipt_directory", IMMUTABLE_RECEIPT_DIRS)
def test_path_migration_refuses_to_rewrite_historical_receipt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_directory: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    source, companion = _write_group(repo, "26:1")
    receipt = _write_reference_receipt(repo, receipt_directory)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    plan = tmp_path / "migration-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": (
                    "axiom-encode/rulespec-path-migration-plan/v1"
                ),
                "base_commit": base,
                "moves": [
                    {
                        "from": "us/statutes/26:1.yaml",
                        "to": "us/statutes/26/1.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    original_receipt = receipt.read_bytes()
    monkeypatch.setattr(
        "axiom_encode.cli._require_applied_encoding_manifest_signer",
        lambda: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli.load_rulespec_local_corpus_release",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "axiom_encode.cli.verify_rulespec_validation_waiver_set",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        "axiom_encode.cli._current_guard_encoder_execution_identity",
        lambda: {"commit": "b" * 40},
    )
    monkeypatch.setattr(
        "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
        lambda: {"commit": "b" * 40},
    )

    with pytest.raises(
        SystemExit,
        match="Prior migration receipt still references a moved identity",
    ):
        cmd_migrate_rulespec_paths(
            SimpleNamespace(
                policy_repo_path=repo,
                corpus_path=tmp_path / "axiom-corpus",
                plan=plan,
            )
        )

    assert receipt.read_bytes() == original_receipt
    assert source.is_file()
    assert companion.is_file()
    assert not (repo / "us/statutes/26/1.yaml").exists()


@pytest.mark.parametrize("receipt_directory", IMMUTABLE_RECEIPT_DIRS)
def test_legacy_replacement_refuses_to_rewrite_historical_receipt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_directory: Path,
) -> None:
    repo = tmp_path / "rulespec-us"
    source, _companion = _write_group(repo, "26:1")
    source.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/statute/26/1\n"
        "rules: []\n",
        encoding="utf-8",
    )
    legacy_manifest = (
        repo / ".axiom/encoding-manifests/us/statutes/26:1.json"
    )
    legacy_manifest.parent.mkdir(parents=True, exist_ok=True)
    legacy_manifest.write_text("{}\n", encoding="utf-8")
    receipt = _write_reference_receipt(repo, receipt_directory)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    original_receipt = receipt.read_bytes()
    source_unit = SimpleNamespace(
        requested="us/statute/26/1",
        citation_path="us/statute/26/1",
        body="authoritative source",
        resolved_source="test-release",
    )
    monkeypatch.setattr(
        "axiom_encode.cli.resolve_corpus_source_unit",
        lambda *_args, **_kwargs: source_unit,
    )
    monkeypatch.setattr(
        "axiom_encode.cli._legacy_v1_manifest_issues",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(
        ValueError,
        match="legacy replacement cannot rewrite persisted provenance",
    ):
        _resolve_legacy_replacement_contract(
            source_raw=Path("us/statutes/26:1.yaml"),
            destination_raw=Path("us/statutes/26/1.yaml"),
            policy_checkout_path=repo,
            policy_repo_path=repo / "us",
            source_unit=source_unit,
            corpus_release=object(),
        )

    assert receipt.read_bytes() == original_receipt
    assert source.is_file()
    assert not (repo / "us/statutes/26/1.yaml").exists()
