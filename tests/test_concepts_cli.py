"""End-to-end tests for concepts-audit CLI and apply-time enforcement hook."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _enforce_canonical_concept_registry,
    _relative_output_to_anchor,
    _repo_jurisdiction_prefix,
)


def test_apply_hook_raises_on_blocked_synonym(tmp_path: Path):
    """The apply-time hook must refuse to install a file using a blocked synonym."""
    generated = tmp_path / "9.yaml"
    generated.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: snap_eligible
                kind: computed
                versions:
                  - effective_from: '2025-10-01'
                    formula: snap_gross_monthly_income <= 1000
            """
        )
    )
    test_file = tmp_path / "9.test.yaml"  # may or may not exist
    relative_output = Path("regulations/7-cfr/273/9.yaml")

    with pytest.raises(RuntimeError, match="blocked_synonym"):
        _enforce_canonical_concept_registry(
            candidate_files=[generated, test_file],
            relative_output=relative_output,
        )


def test_apply_hook_silent_on_canonical_only(tmp_path: Path):
    generated = tmp_path / "10.yaml"
    generated.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: snap_total_gross_income
                kind: parameter
                versions:
                  - effective_from: '2025-10-01'
                    formula: "0"
            """
        )
    )
    relative_output = Path("regulations/7-cfr/273/10.yaml")
    # Must not raise
    _enforce_canonical_concept_registry(
        candidate_files=[generated],
        relative_output=relative_output,
    )


def test_apply_hook_auto_repairs_test_yaml_blocked_synonym(tmp_path: Path):
    """The apply hook must auto-rewrite blocked synonyms in *.test.yaml files
    so encode --apply unblocks once the producer YAML itself is clean."""
    generated = tmp_path / "10.yaml"
    generated.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: snap_total_gross_income
                kind: parameter
                versions:
                  - effective_from: '2025-10-01'
                    formula: "0"
            """
        )
    )
    test_file = tmp_path / "10.test.yaml"
    test_file.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            cases:
              - inputs:
                  us:regulations/7-cfr/273/10#snap_gross_monthly_income: 1000
            """
        )
    )
    relative_output = Path("regulations/7-cfr/273/10.yaml")

    _enforce_canonical_concept_registry(
        candidate_files=[generated, test_file],
        relative_output=relative_output,
    )

    rewritten = test_file.read_text()
    assert "snap_gross_monthly_income" not in rewritten
    assert "snap_total_gross_income" in rewritten


def test_apply_hook_raises_on_canonical_under_wrong_anchor(tmp_path: Path):
    generated = tmp_path / "9.yaml"
    generated.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: snap_total_gross_income
                kind: parameter
                versions:
                  - effective_from: '2025-10-01'
                    formula: "0"
            """
        )
    )
    relative_output = Path("regulations/7-cfr/273/9.yaml")
    with pytest.raises(RuntimeError, match="canonical_conflict"):
        _enforce_canonical_concept_registry(
            candidate_files=[generated],
            relative_output=relative_output,
        )


def test_apply_hook_uses_state_repo_jurisdiction(tmp_path: Path):
    generated = tmp_path / "4.407.31.yaml"
    generated.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: snap_standard_utility_allowance
                kind: parameter
                versions:
                  - effective_from: '2025-10-01'
                    formula: "594"
            """
        )
    )
    repo = tmp_path / "rulespec-us" / "us-co"
    repo.mkdir(parents=True)
    _enforce_canonical_concept_registry(
        candidate_files=[generated],
        relative_output=Path("regulations/10-ccr-2506-1/4.407.31.yaml"),
        policy_repo_path=repo,
    )


def test_relative_output_to_anchor_uses_canonical_jurisdiction_root(tmp_path: Path):
    repo = tmp_path / "rulespec-us" / "us-co"
    repo.mkdir(parents=True)
    assert _repo_jurisdiction_prefix(repo) == "us-co"
    assert (
        _relative_output_to_anchor(
            Path("regulations/10-ccr-2506-1/4.407.31.yaml"),
            policy_repo_path=repo,
        )
        == "us-co:regulations/10-ccr-2506-1/4.407.31"
    )


def test_jurisdiction_prefix_rejects_legacy_flat_repo(tmp_path: Path):
    repo = tmp_path / "rulespec-us-co"
    repo.mkdir()

    with pytest.raises(ValueError, match="canonical"):
        _repo_jurisdiction_prefix(repo)


def test_concepts_audit_cli_runs_and_emits_json(tmp_path: Path):
    """Smoke test: CLI walks an empty root and emits JSON without crashing."""
    empty_root = tmp_path / "rulespec-us"
    empty_root.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "axiom_encode.cli",
            "concepts-audit",
            "--roots",
            str(empty_root),
            "--name-prefix",
            "snap_",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "findings" in payload
    assert payload["count"] == 0


def test_concepts_audit_cli_surfaces_blocked_synonym(tmp_path: Path):
    """Real-content smoke test: the CLI must surface a blocked-synonym finding
    with the expected structure when the corpus actually contains drift."""
    root = tmp_path / "rulespec-us"
    drift = root / "us" / "policies/example.yaml"
    drift.parent.mkdir(parents=True)
    drift.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: example_rule
                kind: derived
                versions:
                  - effective_from: '2025-10-01'
                    formula: snap_gross_monthly_income
            """
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "axiom_encode.cli",
            "concepts-audit",
            "--roots",
            str(root),
            "--name-prefix",
            "snap_",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["count"] >= 1
    blocked = [f for f in payload["findings"] if f["kind"] == "blocked_synonym"]
    assert any(f["name"] == "snap_gross_monthly_income" for f in blocked), payload
