"""Tests for RuleSpec repository identity helpers."""

import os
import subprocess
from pathlib import Path

from axiom_encode.cli import _rulespec_test_relation_request_name
from axiom_encode.concepts.jurisdiction import jurisdiction_prefix
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _candidate_rulespec_repo_roots,
    _canonical_rulespec_compile_path,
)
from axiom_encode.repo_routing import canonical_rulespec_repo_name


def _init_checkout(path: Path, origin: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_canonical_rulespec_repo_name_uses_origin_for_temp_checkout_name(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    assert canonical_rulespec_repo_name(checkout) == "rulespec-us"
    assert canonical_rulespec_repo_name(checkout / "statutes" / "26") == "rulespec-us"
    assert jurisdiction_prefix(checkout) == "us"


def test_candidate_roots_include_temp_checkout_when_origin_matches(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    roots = _candidate_rulespec_repo_roots("rulespec-us", checkout)

    assert checkout.resolve() in [root.resolve() for root in roots]


def test_compile_env_exposes_temp_checkout_under_canonical_name(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    pipeline = ValidatorPipeline(
        policy_repo_path=checkout,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    env = pipeline._rulespec_compile_env()
    roots = [Path(root) for root in env["AXIOM_RULESPEC_REPO_ROOTS"].split(os.pathsep)]

    alias_parent = next(root for root in roots if (root / "rulespec-us").is_symlink())
    assert (alias_parent / "rulespec-us").resolve() == checkout.resolve()


def test_canonical_compile_path_exposes_temp_checkout_file_under_origin_name(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    rules_file = checkout / "statutes" / "26" / "63" / "f.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    compile_path = _canonical_rulespec_compile_path(rules_file, checkout)

    assert "rulespec-us" in compile_path.parts
    assert "rulespec-us-clean.abcd" not in str(compile_path)
    assert compile_path.resolve() == rules_file.resolve()


def test_compiled_program_maps_alias_temp_prefix_to_canonical_legal_id(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    pipeline = ValidatorPipeline(
        policy_repo_path=checkout,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    payload = {
        "program": {
            "derived": [
                {
                    "name": "blind_under_subsection_f",
                    "id": ("us-clean.abcd:statutes/26/63/f#blind_under_subsection_f"),
                }
            ],
            "parameters": [],
        }
    }

    derived, _parameters = pipeline._rulespec_program_maps(payload)
    legal_ids = pipeline._rulespec_legal_ids_by_friendly_output_name(payload)

    assert "us:statutes/26/63/f#blind_under_subsection_f" in derived
    assert "us-clean.abcd:statutes/26/63/f#blind_under_subsection_f" in derived
    assert "blind_under_subsection_f" not in derived
    assert legal_ids == {
        "blind_under_subsection_f": ["us:statutes/26/63/f#blind_under_subsection_f"]
    }


def test_compiled_program_maps_alias_country_subdir_temp_prefix_to_canonical_legal_id(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    rules_file = policy_root / "regulations" / "7-cfr" / "275" / "23" / "d" / "2.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: payment_liability_amount
    kind: derived
    entity: StateAgency
    dtype: Money
    period: Year
    versions:
      - effective_from: '2003-01-01'
        formula: payment_error_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    payload = {
        "program": {
            "derived": [
                {
                    "name": "payment_liability_amount",
                    "id": (
                        "us-clean.abcd:us/regulations/7-cfr/275/23/d/2"
                        "#payment_liability_amount"
                    ),
                }
            ],
            "parameters": [],
        }
    }

    derived, _parameters = pipeline._rulespec_program_maps(payload)
    legal_ids = pipeline._rulespec_legal_ids_by_friendly_output_name(payload)

    assert "us:regulations/7-cfr/275/23/d/2#payment_liability_amount" in derived
    assert (
        "us-clean.abcd:us/regulations/7-cfr/275/23/d/2#payment_liability_amount"
        in derived
    )
    assert legal_ids == {
        "payment_liability_amount": [
            "us:regulations/7-cfr/275/23/d/2#payment_liability_amount"
        ]
    }


def test_dataset_preserves_canonical_same_repo_refs_for_alias_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    rules_file = checkout / "statutes" / "26" / "63" / "f.yaml"
    imported_file = checkout / "statutes" / "26" / "151.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: spouse_of_taxpayer_for_subsection_f
    kind: data_relation
    data_relation:
      predicate: us:concepts/tax#spouse_of_taxpayer_for_subsection_f
      arity: 2
      arguments:
        - name: spouse
          type: Person
        - name: tax_unit
          type: TaxUnit
  - name: taxpayer_blind_additional_amount_entitlement
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: taxpayer_is_blind_at_close_of_taxable_year
"""
    )
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: tin_included_on_return_claiming_exemption
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=checkout,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    dataset = pipeline._build_rulespec_dataset(
        {
            "us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f": [
                {
                    "id": "spouse",
                    "us:statutes/26/151#input.tin_included_on_return_claiming_exemption": True,
                }
            ],
            "us:statutes/26/63/f#input.taxpayer_is_blind_at_close_of_taxable_year": True,
        },
        period={"start": "2026-01-01", "end": "2026-12-31"},
        query_entity="TaxUnit",
        query_entity_id="case-1",
        require_legal_input_keys=True,
    )

    assert dataset["relations"][0]["name"] == (
        "us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f"
    )
    assert dataset["inputs"][0]["name"] == (
        "us:statutes/26/151#input.tin_included_on_return_claiming_exemption"
    )
    assert dataset["inputs"][1]["name"] == (
        "us:statutes/26/63/f#input.taxpayer_is_blind_at_close_of_taxable_year"
    )


def test_dataset_preserves_country_subdir_canonical_refs_for_alias_checkout(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()
    rules_file = policy_root / "regulations" / "7-cfr" / "275" / "23" / "d" / "2.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: payment_liability_amount
    kind: derived
    entity: StateAgency
    dtype: Money
    period: Year
    versions:
      - effective_from: '2003-01-01'
        formula: payment_error_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    dataset = pipeline._build_rulespec_dataset(
        {
            "us:regulations/7-cfr/275/23/d/2#input.payment_error_rate": 0.08,
        },
        period={"start": "2003-01-01", "end": "2003-12-31"},
        query_entity="StateAgency",
        query_entity_id="state-agency",
        require_legal_input_keys=True,
    )

    assert dataset["inputs"][0]["name"] == (
        "us:regulations/7-cfr/275/23/d/2#input.payment_error_rate"
    )


def test_cli_relation_request_preserves_country_subdir_canonical_refs(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    relation_name = _rulespec_test_relation_request_name(
        "us:regulations/7-cfr/275/23/d/1#relation.state_agencies_for_quality_control_fiscal_year",
        policy_repo_path=policy_root,
    )

    assert relation_name == (
        "us:regulations/7-cfr/275/23/d/1"
        "#relation.state_agencies_for_quality_control_fiscal_year"
    )
