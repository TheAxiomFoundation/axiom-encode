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


def test_repair_preserves_consumer_anchor_on_input_refs(tmp_path: Path):
    """For `#input.X` refs the anchor names the consumer file (where X is read
    as an input slot), not the canonical producer. Rewriting the anchor to the
    producer would point the input slot at the wrong file. Surfaced live on
    7 USC 2014(c) test cases 2026-05-28: encoder wrote
    `us:statutes/7/2014/e/6/A#input.snap_monthly_household_income` (legit:
    `(e)(6)(A)` is the consumer, gross income is an input to net income).
    The repair swapped the anchor to `us:regulations/7-cfr/273/10` (the
    canonical producer of gross income), corrupting the test.
    """
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
    # Consumer anchor preserved — the input slot still belongs to (e)(6)(A).
    assert "us:statutes/7/2014/e/6/A#input.snap_total_gross_income" in text
    assert "us:regulations/7-cfr/273/10#input." not in text


def test_repair_preserves_producer_relation_child_input_refs(tmp_path: Path):
    """A producer may use a child/member input to derive a household-level
    canonical output. That input must not be rewritten to the output concept.
    """
    registry = load_concept_registry()
    drift = _write(
        tmp_path,
        "drift.test.yaml",
        """
        - name: household_has_elderly_or_disabled_member
          period: 2026-01
          input:
            us:statutes/7/2012/j#relation.member_of_household:
              - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
          output:
            us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
        """,
    )
    changed = auto_repair_test_yaml_canonical_violations([drift], registry)
    assert changed == []
    text = drift.read_text()
    assert "us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled" in text
    assert "#input.snap_household_has_elderly_or_disabled_member" not in text
    violations = validate_generated_against_registry([drift], registry)
    assert violations == []


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
