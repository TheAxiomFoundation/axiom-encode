"""Tests for the PolicyEngine oracle classifier."""

from __future__ import annotations

from pathlib import Path

import yaml

from axiom_encode.oracles.policyengine.adapters import PolicyEngineUSVarAdapter
from axiom_encode.oracles.policyengine.classifier import (
    Classification,
    _build_rule_name_index,
    _classify_one,
    classification_to_yaml_block,
    classify_rulespec_repo,
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


def test_health_rate_pattern_matches_with_rate_dtype() -> None:
    adapters = (
        PolicyEngineUSVarAdapter(
            rule_names=("medicaid_income_level",),
            pe_var="medicaid_income_level",
            entity="person",
            period="year",
            unit="/1",
            comparison="rate",
        ),
    )
    result = _classify_one(
        rule_name="medicaid_income_level",
        legal_id="us-co:regulations/hcpf#medicaid_income_level",
        dtype="Rate",
        source_text=None,
        rule_index=_build_rule_name_index(adapters),
    )
    assert result.mapping_type == "direct_variable"
    assert result.policyengine_variable == "medicaid_income_level"
    assert result.entity == "person"
    assert result.period == "year"
    assert result.unit == "/1"
    assert result.comparison == "rate"


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
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
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


def test_classify_rulespec_repo_supports_health_program_catalog(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-us-co"
    rules_dir = repo / "regulations" / "hcpf" / "health-coverage"
    rules_dir.mkdir(parents=True)
    (rules_dir / "eligibility.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "is_medicaid_eligible",
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "A person is eligible for Health First Colorado.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
                    },
                    {
                        "name": "is_chip_eligible",
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "A child is eligible for CHP+.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
                    },
                    {
                        "name": "aca_ptc",
                        "kind": "derived",
                        "dtype": "Money",
                        "source": "The premium tax credit is allowed.",
                        "versions": [{"effective_from": "2025-01-01", "formula": "1"}],
                    },
                ],
            }
        )
    )
    snap_dir = repo / "policies" / "cdhs" / "snap"
    snap_dir.mkdir(parents=True)
    (snap_dir / "benefit.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "snap_sick_vacation_bonus_earned_income",
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "A SNAP-only output must not appear in health classify.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
                    },
                ],
            }
        )
    )
    tax_dir = repo / "statutes" / "39" / "39-22-104" / "4" / "n"
    tax_dir.mkdir(parents=True)
    (tax_dir / "5.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": (
                            "activity_secondarily_treated_woody_fuels_by_lopping_"
                            "scattering_piling_chipping_removing_from_site"
                        ),
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "A tax forestry output must not be treated as CHIP.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
                    },
                ],
            }
        )
    )

    classifications = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="health",
    )

    by_name = {c.legal_id.split("#")[1]: c for c in classifications}
    assert by_name["is_medicaid_eligible"].mapping_type == "direct_variable"
    assert (
        by_name["is_medicaid_eligible"].policyengine_variable == "is_medicaid_eligible"
    )
    assert by_name["is_medicaid_eligible"].entity == "person"
    assert by_name["is_chip_eligible"].policyengine_variable == "is_chip_eligible"
    assert by_name["aca_ptc"].policyengine_variable == "aca_ptc"
    assert by_name["aca_ptc"].entity == "tax_unit"
    assert by_name["aca_ptc"].unit == "USD"
    assert "snap_sick_vacation_bonus_earned_income" not in by_name
    assert (
        "activity_secondarily_treated_woody_fuels_by_lopping_scattering_piling_"
        "chipping_removing_from_site" not in by_name
    )

    medicaid_only = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="medicaid",
    )
    assert [c.legal_id.split("#")[1] for c in medicaid_only] == ["is_medicaid_eligible"]

    chip_only = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="chip",
    )
    assert [c.legal_id.split("#")[1] for c in chip_only] == ["is_chip_eligible"]

    aca_only = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="aca_ptc",
    )
    assert [c.legal_id.split("#")[1] for c in aca_only] == ["aca_ptc"]


def test_classify_splits_combined_medicaid_chip_sources_by_rule_name(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-us-co"
    rules_dir = repo / "policies" / "cms"
    rules_dir.mkdir(parents=True)
    (rules_dir / "medicaid-chip-bhp-eligibility-levels.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "children_separate_chip_income_standard",
                        "kind": "parameter",
                        "dtype": "Rate",
                        "source": "Colorado CHIP income standard.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "2.60"}
                        ],
                    },
                    {
                        "name": "adult_expansion_medicaid_income_standard",
                        "kind": "parameter",
                        "dtype": "Rate",
                        "source": "Colorado Medicaid income standard.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "1.33"}
                        ],
                    },
                    {
                        "name": "magi_fpl_disregard_equivalent",
                        "kind": "parameter",
                        "dtype": "Rate",
                        "source": "MAGI disregard.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "0.05"}
                        ],
                    },
                ],
            }
        )
    )

    chip = classify_rulespec_repo(repo_root=repo, jurisdiction="us-co", program="chip")
    medicaid = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="medicaid",
    )
    health = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us-co",
        program="health",
    )

    assert [c.legal_id.split("#")[1] for c in chip] == [
        "children_separate_chip_income_standard"
    ]
    assert [c.legal_id.split("#")[1] for c in medicaid] == [
        "adult_expansion_medicaid_income_standard"
    ]
    assert {c.legal_id.split("#")[1] for c in health} == {
        "children_separate_chip_income_standard",
        "adult_expansion_medicaid_income_standard",
        "magi_fpl_disregard_equivalent",
    }


def test_classify_rulespec_repo_filters_federal_ssi_sources(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-us"
    ssi_dir = repo / "statutes" / "42" / "1382a" / "b"
    ssi_dir.mkdir(parents=True)
    (ssi_dir / "2.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "ssi_general_income_exclusion_annual_amount",
                        "kind": "parameter",
                        "dtype": "Money",
                        "source": "The first $240 per year is excluded.",
                        "versions": [
                            {"effective_from": "1974-01-01", "formula": "240"}
                        ],
                    }
                ],
            }
        )
    )
    snap_dir = repo / "statutes" / "7" / "2014" / "e"
    snap_dir.mkdir(parents=True)
    (snap_dir / "2.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "snap_earned_income_deduction_rate",
                        "kind": "parameter",
                        "dtype": "Rate",
                        "source": "SNAP earned income deduction.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "0.2"}
                        ],
                    }
                ],
            }
        )
    )

    classifications = classify_rulespec_repo(
        repo_root=repo,
        jurisdiction="us",
        program="ssi",
    )

    assert [c.legal_id.split("#")[1] for c in classifications] == [
        "ssi_general_income_exclusion_annual_amount"
    ]
    assert classifications[0].mapping_type == "not_comparable"


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


def test_pe_us_var_adapters_is_consumable_by_classifier() -> None:
    """Smoke test: the production adapter catalog feeds through the classifier.

    A typo or breaking schema change in `PE_US_VAR_ADAPTERS` (the catalog that
    ECPS comparison already consults) would otherwise silently corrupt every
    `axiom-encode classify` invocation. This guards against that.
    """
    from axiom_encode.oracles.policyengine.adapters import PE_US_VAR_ADAPTERS

    # The index must build without raising.
    index = _build_rule_name_index(PE_US_VAR_ADAPTERS)
    # The catalog has at least the well-known SNAP outputs.
    assert "snap_normal_allotment" in index
    assert "snap_standard_deduction" in index
    assert "snap_standard_utility_allowance" in index
    # Each entry resolves to a PolicyEngineUSVarAdapter with a non-empty
    # `pe_var` and non-empty `rule_names`.
    for rule_name, adapter in index.items():
        assert adapter.pe_var, f"adapter for {rule_name!r} has empty pe_var"
        assert rule_name in adapter.rule_names

    # Driving a sample rule through `_classify_one` against the production
    # catalog should promote it to `direct_variable` (not raise, not fall
    # through to not_comparable).
    result = _classify_one(
        rule_name="snap_standard_deduction",
        legal_id="us-xx:regulations/abc#snap_standard_deduction",
        dtype="Money",
        source_text=None,
        rule_index=index,
    )
    assert result.mapping_type == "direct_variable"
    assert result.policyengine_variable == "snap_standard_deduction"


def test_pe_us_health_var_adapters_is_consumable_by_classifier() -> None:
    from axiom_encode.oracles.policyengine.adapters import (
        PE_US_HEALTH_VAR_ADAPTERS,
        PE_US_PROGRAM_VAR_ADAPTERS,
    )

    index = _build_rule_name_index(PE_US_HEALTH_VAR_ADAPTERS)
    assert "is_medicaid_eligible" in index
    assert "is_chip_eligible" in index
    assert "aca_ptc" in index

    result = _classify_one(
        rule_name="aca_ptc",
        legal_id="us:statutes/26/36B#aca_ptc",
        dtype="Money",
        source_text=None,
        rule_index=index,
    )
    assert result.mapping_type == "direct_variable"
    assert result.policyengine_variable == "aca_ptc"
    assert result.entity == "tax_unit"
    assert result.period == "year"

    medicaid_index = _build_rule_name_index(PE_US_PROGRAM_VAR_ADAPTERS["medicaid"])
    chip_index = _build_rule_name_index(PE_US_PROGRAM_VAR_ADAPTERS["chip"])
    aca_index = _build_rule_name_index(PE_US_PROGRAM_VAR_ADAPTERS["aca_ptc"])
    assert "is_medicaid_eligible" in medicaid_index
    assert "is_chip_eligible" not in medicaid_index
    assert "is_chip_eligible" in chip_index
    assert "aca_ptc" not in chip_index
    assert "aca_ptc" in aca_index


def test_cmd_classify_write_us_yaml_round_trip(tmp_path: Path) -> None:
    """End-to-end: classify writes per-rule entries and strips bulk prefixes.

    Builds a fixture us.yaml that contains:
    - one mapping unrelated to our state (must survive)
    - one bulk `legal_id_prefix: us-xx:...` entry that should be removed
    - a `prefixes:` section with one unrelated prefix (must survive)

    Builds a fixture rulespec-us-xx repo with one Money-typed rule whose name
    is in the test adapter catalog (must be promoted) and one Judgment-typed
    rule whose name is not (must become not_comparable).

    Asserts the final us.yaml has both new per-rule entries, has dropped the
    bulk prefix entry, and still parses as valid YAML.
    """
    from unittest.mock import patch

    from axiom_encode.cli import cmd_classify

    # Build the fixture rulespec-us-xx repo.
    repo = tmp_path / "rulespec-us-xx"
    section = repo / "regulations" / "abc" / "100"
    section.mkdir(parents=True)
    (section / "block-1.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "monthly_allotment",
                        "kind": "derived",
                        "dtype": "Money",
                        "source": "Monthly SNAP allotment for the household.",
                        "versions": [{"effective_from": "2025-01-01", "formula": "1"}],
                    },
                    {
                        "name": "household_qualifies_for_review",
                        "kind": "derived",
                        "dtype": "Judgment",
                        "source": "Predicate gating second-party review.",
                        "versions": [
                            {"effective_from": "2025-01-01", "formula": "true"}
                        ],
                    },
                ],
            }
        )
    )

    # Build the fixture us.yaml.
    us_yaml = tmp_path / "us.yaml"
    us_yaml.write_text(
        "mappings:\n"
        "  - legal_id: us:statutes/7/2014/e/2#snap_earned_income_deduction\n"
        "    country: us\n"
        "    program: snap\n"
        "    mapping_type: direct_variable\n"
        "    policyengine_variable: snap_earned_income_deduction\n"
        "    rationale: Existing federal mapping that must survive the rewrite.\n"
        "\n"
        "prefixes:\n"
        "  - legal_id_prefix: us-xx:regulations/abc/100/block-1#\n"
        "    country: us\n"
        "    program: snap\n"
        "    mapping_type: not_comparable\n"
        "    rationale: Bulk fig leaf that the rewrite should remove.\n"
        "\n"
        "  - legal_id_prefix: us-yy:regulations/zzz#\n"
        "    country: us\n"
        "    program: snap\n"
        "    mapping_type: not_comparable\n"
        "    rationale: Unrelated state's prefix that must survive.\n"
    )

    # Patch the adapter catalog used by the classifier to the small fixture
    # list, so the round trip doesn't depend on the production catalog.
    args = type(
        "Args",
        (),
        {
            "state": "xx",
            "program": "snap",
            "repo": repo,
            "write_us_yaml": us_yaml,
        },
    )()

    with patch(
        "axiom_encode.oracles.policyengine.classifier.PE_US_VAR_ADAPTERS",
        _TEST_ADAPTERS,
    ):
        with pytest.raises(SystemExit) as exc_info:
            cmd_classify(args)
        assert exc_info.value.code == 0

    final = yaml.safe_load(us_yaml.read_text())

    # The existing federal mapping must survive.
    legal_ids = {m["legal_id"] for m in final["mappings"]}
    assert "us:statutes/7/2014/e/2#snap_earned_income_deduction" in legal_ids

    # The two new per-rule entries must be present.
    assert "us-xx:regulations/abc/100/block-1#monthly_allotment" in legal_ids
    assert (
        "us-xx:regulations/abc/100/block-1#household_qualifies_for_review" in legal_ids
    )

    by_id = {m["legal_id"]: m for m in final["mappings"]}
    promoted = by_id["us-xx:regulations/abc/100/block-1#monthly_allotment"]
    assert promoted["mapping_type"] == "direct_variable"
    assert promoted["policyengine_variable"] == "snap_normal_allotment"

    fallback = by_id["us-xx:regulations/abc/100/block-1#household_qualifies_for_review"]
    assert fallback["mapping_type"] == "not_comparable"
    assert "Predicate gating" in fallback["rationale"]

    # The fixture's bulk legal_id_prefix for this state must be gone.
    prefix_ids = {p["legal_id_prefix"] for p in final.get("prefixes", [])}
    assert "us-xx:regulations/abc/100/block-1#" not in prefix_ids
    # The unrelated state's prefix must still be present.
    assert "us-yy:regulations/zzz#" in prefix_ids


# Lazy pytest import so the rest of the module is importable without pytest
# (the test runner already loads it).
import pytest  # noqa: E402  (intentional late import for the round-trip test)
