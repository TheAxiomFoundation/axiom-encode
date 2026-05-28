"""Auto-repair for canonical-name violations in *.test.yaml.

Producer files must fail loudly so the encoder learns. Test files just carry
example inputs/cases — we rewrite them in place so a successful encode --apply
is not blocked by drift in the cases the model invented.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from axiom_encode.concepts.auto_repair import auto_repair_test_yaml_canonical_violations
from axiom_encode.concepts.registry import load_concept_registry
from axiom_encode.concepts.validator import validate_generated_against_registry


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"))
    return p


def test_repair_rewrites_blocked_synonym_in_anchored_ref(tmp_path: Path):
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
    changed = auto_repair_test_yaml_canonical_violations([drift], registry)
    assert drift in changed
    text = drift.read_text()
    assert "snap_gross_monthly_income" not in text
    assert "snap_total_gross_income" in text
    violations = validate_generated_against_registry([drift], registry)
    assert violations == []


def test_repair_rewrites_anchor_when_canonical_under_wrong_anchor(tmp_path: Path):
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
    auto_repair_test_yaml_canonical_violations([drift], registry)
    text = drift.read_text()
    canonical = registry.lookup_canonical("snap_net_income")
    assert canonical is not None
    assert canonical.producer_anchor in text
    assert "us:regulations/7-cfr/273/9#snap_net_income" not in text
    violations = validate_generated_against_registry([drift], registry)
    assert all(v.kind != "anchored_ref_miss" for v in violations)


def test_repair_handles_input_dot_prefixed_uppercase_paths(tmp_path: Path):
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
    auto_repair_test_yaml_canonical_violations([drift], registry)
    text = drift.read_text()
    assert "snap_monthly_household_income" not in text
    assert "snap_total_gross_income" in text
    # The `input.` prefix must be preserved so the runtime still treats it as
    # an input rather than a producer ref.
    assert "#input.snap_total_gross_income" in text


def test_repair_skips_non_test_files(tmp_path: Path):
    """Producer/source YAML must fail loudly — never silently rewritten."""
    registry = load_concept_registry()
    producer = _write(
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
    before = producer.read_text()
    changed = auto_repair_test_yaml_canonical_violations([producer], registry)
    assert changed == []
    assert producer.read_text() == before


def test_repair_returns_empty_when_nothing_to_fix(tmp_path: Path):
    registry = load_concept_registry()
    clean = _write(
        tmp_path,
        "clean.test.yaml",
        """
        format: rulespec/v1
        cases:
          - inputs:
              us:regulations/7-cfr/273/10#snap_total_gross_income: 1000
        """,
    )
    changed = auto_repair_test_yaml_canonical_violations([clean], registry)
    assert changed == []
