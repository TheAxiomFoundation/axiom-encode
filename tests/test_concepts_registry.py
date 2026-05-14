"""Tests for the canonical-concept registry, audit, and validator."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from axiom_encode.concepts.registry import (
    REGISTRY_FORMAT,
    Concept,
    ConceptRegistry,
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
