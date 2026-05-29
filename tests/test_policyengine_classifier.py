"""Tests for the PolicyEngine oracle classifier."""

from __future__ import annotations

from pathlib import Path

import yaml

from axiom_encode.oracles.policyengine.adapters import PolicyEngineUSVarAdapter
from axiom_encode.oracles.policyengine.classifier import (
    Classification,
    _build_rule_name_index,
    _classify_one,
    classify_rulespec_repo,
    classification_to_yaml_block,
    iter_rules_in_rulespec_file,
)


# A small adapter fixture used to exercise the dtype guard and direct
# variable / not_comparable branches without depending on the production
# adapter list.
_TEST_ADAPTERS = (
    PolicyEngineUSVarAdapter(
        rule_names=("monthly_allotment", "snap_normal_allotment"),
        pe_var="snap_normal_allotment",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("is_snap_eligible",),
        pe_var="is_snap_eligible",
        monthly=True,
        spm=True,
    ),
)


def _make_rule_index() -> dict[str, PolicyEngineUSVarAdapter]:
    return _build_rule_name_index(_TEST_ADAPTERS)


def test_money_pattern_matches_with_money_dtype() -> None:
    """A Money-typed rule whose name appears in the adapter catalog is promoted."""
    result = _classify_one(
        rule_name="monthly_allotment",
        legal_id="us-xx:regulations/abc#monthly_allotment",
        dtype="Money",
        source_text="The monthly allotment is the household allotment.",
        rule_index=_make_rule_index(),
    )
    assert result.mapping_type == "direct_variable"
    assert result.policyengine_variable == "snap_normal_allotment"
    assert result.entity == "spm_unit"
    assert result.period == "month"
    assert result.unit == "USD"
    assert result.comparison == "money"
    assert result.matched_adapter == "monthly_allotment"


def test_money_pattern_rejects_judgment_dtype() -> None:
    """A Judgment-typed rule named like a money variable falls through to not_comparable.

    Without the dtype guard, a boolean predicate like
    `monthly_allotment_required` could be misclassified as the dollar amount.
    The guard rejects the promotion based on the rule's dtype.
    """
    result = _classify_one(
        rule_name="monthly_allotment",
        legal_id="us-xx:regulations/abc#monthly_allotment",
        dtype="Judgment",
        source_text="placeholder",
        rule_index=_make_rule_index(),
    )
    assert result.mapping_type == "not_comparable"
    assert result.policyengine_variable is None


def test_decision_pattern_matches_with_judgment_dtype() -> None:
    result = _classify_one(
        rule_name="is_snap_eligible",
        legal_id="us-xx:regulations/abc#is_snap_eligible",
        dtype="Judgment",
        source_text=None,
        rule_index=_make_rule_index(),
    )
    assert result.mapping_type == "direct_variable"
    assert result.comparison == "decision"
    assert result.unit is None


def test_unmatched_rule_name_falls_through_to_not_comparable() -> None:
    result = _classify_one(
        rule_name="some_state_specific_predicate",
        legal_id="us-xx:regulations/abc#some_state_specific_predicate",
        dtype="Judgment",
        source_text="Some predicate.",
        rule_index=_make_rule_index(),
    )
    assert result.mapping_type == "not_comparable"
    assert "some_state_specific_predicate" in result.rationale


def test_classify_rulespec_repo_walks_yaml_files(tmp_path: Path) -> None:
    """Walking a repo yields one Classification per executable rule."""
    repo = tmp_path / "rulespec-us-xx"
    rules_dir = repo / "regulations" / "abc" / "100"
    rules_dir.mkdir(parents=True)
    (rules_dir / "block-1.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "monthly_allotment",
                        "kind": "derived",
                        "dtype": "Money",
                        "source": "Source text for monthly allotment.",
                        "versions": [{"effective_from": "2025-01-01", "formula": "1"}],
                    },
                    {
                        "name": "some_predicate",
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "Source text for some predicate.",
                        "versions": [{"effective_from": "2025-01-01", "formula": "true"}],
                    },
                ],
            }
        )
    )
    # Companion test file must be skipped.
    (rules_dir / "block-1.test.yaml").write_text("scenarios: []")

    classifications = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-xx",
        program="snap",
        adapters=_TEST_ADAPTERS,
    )

    assert len(classifications) == 2
    by_name = {c.legal_id.split("#")[1]: c for c in classifications}
    assert by_name["monthly_allotment"].mapping_type == "direct_variable"
    assert by_name["monthly_allotment"].policyengine_variable == "snap_normal_allotment"
    assert by_name["some_predicate"].mapping_type == "not_comparable"


def test_iter_rules_skips_non_executable_kinds(tmp_path: Path) -> None:
    """`source_relation` and missing-kind rules are not executable outputs."""
    path = tmp_path / "block-1.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {"name": "real", "kind": "derived", "dtype": "Money"},
                    {"name": "ignored_rel", "kind": "source_relation"},
                    {"name": "no_kind"},
                ]
            }
        )
    )
    rules = list(iter_rules_in_rulespec_file(path))
    assert [r[0] for r in rules] == ["real"]


def test_classification_to_yaml_block_quotes_special_characters() -> None:
    """A rationale containing colons or quotes still produces valid YAML."""
    c = Classification(
        legal_id="us-xx:regulations/abc/1#rule",
        mapping_type="not_comparable",
        policyengine_variable=None,
        entity=None,
        period=None,
        unit=None,
        comparison=None,
        rationale='Includes "quotes": and colons that would otherwise break YAML.',
        matched_adapter=None,
    )
    rendered = classification_to_yaml_block(c, program="snap")
    # The block is shaped for splicing into us.yaml under `mappings:`. Parse it
    # by reconstructing the surrounding list context.
    parsed = yaml.safe_load("mappings:\n" + rendered)
    entries = parsed["mappings"]
    assert len(entries) == 1
    assert entries[0]["legal_id"] == "us-xx:regulations/abc/1#rule"
    assert entries[0]["mapping_type"] == "not_comparable"
    assert "Includes" in entries[0]["rationale"]
