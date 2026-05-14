"""Tests for the canonical-concept registry, audit, and validator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from axiom_encode.concepts.audit import audit_corpus
from axiom_encode.concepts.registry import (
    REGISTRY_FORMAT,
    load_concept_registry,
)
from axiom_encode.concepts.validator import validate_generated_against_registry


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_load_packaged_registry_succeeds():
    registry = load_concept_registry()
    assert "snap.household.gross_monthly_income" in registry.concepts_by_id
    gross = registry.lookup_canonical("snap_total_gross_income")
    assert gross is not None
    assert gross.producer_anchor == "us:regulations/7-cfr/273/10"
    assert registry.lookup_synonym("snap_gross_monthly_income") is gross


def test_load_rejects_duplicate_canonical(tmp_path: Path):
    _write(
        tmp_path,
        "dup.yaml",
        f"""
        format: {REGISTRY_FORMAT}
        concepts:
          - id: a.one
            canonical_name: same_name
            producer_anchor: us:x/1
          - id: a.two
            canonical_name: same_name
            producer_anchor: us:x/2
        """,
    )
    with pytest.raises(ValueError, match="claimed by"):
        load_concept_registry(tmp_path)


def test_load_rejects_canonical_overlapping_synonym(tmp_path: Path):
    _write(
        tmp_path,
        "overlap.yaml",
        f"""
        format: {REGISTRY_FORMAT}
        concepts:
          - id: a.one
            canonical_name: foo
            producer_anchor: us:x/1
          - id: a.two
            canonical_name: bar
            producer_anchor: us:x/2
            blocked_synonyms: [foo]
        """,
    )
    with pytest.raises(ValueError, match="canonical for"):
        load_concept_registry(tmp_path)


def test_load_rejects_unsupported_format(tmp_path: Path):
    _write(
        tmp_path,
        "bad.yaml",
        """
        format: axiom-encode/concepts/v0
        concepts: []
        """,
    )
    with pytest.raises(ValueError, match="unsupported registry format"):
        load_concept_registry(tmp_path)


def test_validator_flags_blocked_synonym_in_producer_name(tmp_path: Path):
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.yaml",
        """
        format: rulespec/v1
        rules:
          - name: snap_gross_monthly_income
            kind: parameter
            versions:
              - effective_from: '2025-10-01'
                formula: "0"
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    kinds = [v.kind for v in violations]
    assert "blocked_synonym" in kinds


def test_validator_flags_blocked_synonym_in_formula(tmp_path: Path):
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.yaml",
        """
        format: rulespec/v1
        rules:
          - name: snap_eligible
            kind: computed
            versions:
              - effective_from: '2025-10-01'
                formula: snap_gross_monthly_income <= 1000
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    assert any(
        v.kind == "blocked_synonym" and v.name == "snap_gross_monthly_income"
        for v in violations
    )


def test_validator_flags_anchored_ref_to_blocked_synonym(tmp_path: Path):
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.test.yaml",
        """
        format: rulespec/v1
        cases:
          - inputs:
              us:regulations/7-cfr/273/10#snap_gross_monthly_income: 1000
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    assert any(v.name == "snap_gross_monthly_income" for v in violations)


def test_validator_flags_state_anchored_ref_to_blocked_synonym(tmp_path: Path):
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.test.yaml",
        """
        format: rulespec/v1
        cases:
          - inputs:
              us-co:regulations/10-ccr-2506-1/4.401#snap_gross_monthly_income: 1000
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    assert any(v.name == "snap_gross_monthly_income" for v in violations)


def test_validator_flags_uppercase_path_anchored_ref_to_blocked_synonym(
    tmp_path: Path,
):
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.test.yaml",
        """
        format: rulespec/v1
        cases:
          - inputs:
              us:statutes/7/2014/e/6/A#input.snap_monthly_household_income: 1000
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    assert any(v.name == "snap_monthly_household_income" for v in violations)


def test_validator_passes_canonical_only_yaml(tmp_path: Path):
    registry = load_concept_registry()
    good = _write(
        tmp_path,
        "good.yaml",
        """
        format: rulespec/v1
        rules:
          - name: snap_total_gross_income
            kind: parameter
            versions:
              - effective_from: '2025-10-01'
                formula: "0"
        """,
    )
    violations = validate_generated_against_registry(
        [good], registry, apply_anchor="us:regulations/7-cfr/273/10"
    )
    assert violations == []


def test_validator_flags_consumer_anchored_ref_to_canonical_wrong_anchor(tmp_path: Path):
    """Consumer references a registered canonical name at the wrong producer anchor."""
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.test.yaml",
        """
        format: rulespec/v1
        cases:
          - inputs:
              us:regulations/7-cfr/273/9#snap_net_income: 500
        """,
    )
    violations = validate_generated_against_registry([drift], registry)
    assert any(
        v.kind == "anchored_ref_miss" and v.name == "snap_net_income"
        for v in violations
    )


def test_validator_flags_canonical_under_wrong_anchor(tmp_path: Path):
    registry = load_concept_registry()
    wrong = _write(
        tmp_path,
        "wrong.yaml",
        """
        format: rulespec/v1
        rules:
          - name: snap_total_gross_income
            kind: parameter
            versions:
              - effective_from: '2025-10-01'
                formula: "0"
        """,
    )
    violations = validate_generated_against_registry(
        [wrong], registry, apply_anchor="us:regulations/7-cfr/273/9"
    )
    assert any(v.kind == "canonical_conflict" for v in violations)


def test_validator_allows_producer_missing_canonical_under_temporary_anchor(
    tmp_path: Path,
):
    registry = load_concept_registry()
    state_producer = _write(
        tmp_path,
        "4.407.31.yaml",
        """
        format: rulespec/v1
        rules:
          - name: snap_standard_utility_allowance
            kind: parameter
            versions:
              - effective_from: '2025-10-01'
                formula: "594"
        """,
    )
    violations = validate_generated_against_registry(
        [state_producer],
        registry,
        apply_anchor="us-co:regulations/10-ccr-2506-1/4.407.31",
    )
    assert violations == []


def test_audit_uses_state_jurisdiction_prefix_for_rules_roots(tmp_path: Path):
    registry = load_concept_registry()
    root = tmp_path / "rules-us-co"
    path = root / "regulations/10-ccr-2506-1/4.401.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
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

    findings = audit_corpus([root], registry, name_prefixes=("snap_",))
    assert any(
        f.kind == "canonical_conflict"
        and f.name == "snap_total_gross_income"
        and f.detail.endswith("expects us:regulations/7-cfr/273/10")
        for f in findings
    )


def test_audit_name_prefix_filters_blocked_synonyms(tmp_path: Path):
    registry = load_concept_registry()
    root = tmp_path / "rulespec-us"
    path = root / "policies/example.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            rules:
              - name: example
                kind: derived
                versions:
                  - effective_from: '2025-10-01'
                    formula: snap_gross_monthly_income
            """
        )
    )

    findings = audit_corpus([root], registry, name_prefixes=("ny_snap_",))
    assert findings == []


def test_audit_scans_uppercase_path_anchored_refs(tmp_path: Path):
    registry = load_concept_registry()
    root = tmp_path / "rulespec-us"
    path = root / "policies/example.test.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        textwrap.dedent(
            """
            format: rulespec/v1
            cases:
              - inputs:
                  us:statutes/7/2014/e/6/A#input.snap_monthly_household_income: 1000
            """
        )
    )

    findings = audit_corpus([root], registry, name_prefixes=("snap_",))
    assert any(
        f.kind == "blocked_synonym" and f.name == "snap_monthly_household_income"
        for f in findings
    )
