"""Tests for RuleSpec repository identity helpers."""

import subprocess
from pathlib import Path

import pytest

from axiom_encode.cli import _rulespec_test_relation_request_name
from axiom_encode.concepts.jurisdiction import jurisdiction_prefix
from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILESYSTEM_ROOTS,
)
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline as _ValidatorPipeline,
)
from axiom_encode.harness.validator_pipeline import (
    _candidate_rulespec_repo_roots,
    _canonical_rulespec_compile_path,
)
from axiom_encode.repo_routing import (
    candidate_jurisdiction_content_dirs,
    canonical_rulespec_repo_name,
    canonical_rulespec_root_identity,
    find_policy_repo_root,
    is_composition_policy_repo_root,
    is_policy_repo_root,
    jurisdiction_subdir_names,
)


class ValidatorPipeline(_ValidatorPipeline):
    """Test convenience wrapper for corpus-free routing helpers."""

    def __init__(self, *args, local_corpus_release=None, **kwargs):
        super().__init__(
            *args,
            local_corpus_release=local_corpus_release,
            **kwargs,
        )


def _init_checkout(path: Path, origin: str) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", origin],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_rulespec_filesystem_and_atomic_root_contracts_are_distinct():
    assert RULESPEC_ATOMIC_MODULE_ROOTS == frozenset(
        {"legislation", "policies", "regulations", "statutes"}
    )
    assert RULESPEC_COMPOSITION_SPEC_ROOT == "programs"
    assert RULESPEC_FILESYSTEM_ROOTS == (
        RULESPEC_ATOMIC_MODULE_ROOTS | {RULESPEC_COMPOSITION_SPEC_ROOT}
    )


def test_composition_checkout_identity_admits_only_top_level_programs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "programs/us/snap").mkdir(parents=True)

    assert is_policy_repo_root(checkout) is False
    assert is_composition_policy_repo_root(checkout) is True
    assert jurisdiction_subdir_names(checkout, allow_composition_specs=True) == set()

    (checkout / "statutes").mkdir()
    assert is_composition_policy_repo_root(checkout) is False


def test_canonical_rulespec_root_identity_is_stable_for_direct_content_root(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    policy_root.mkdir()

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us-co"
    assert canonical_rulespec_repo_name(policy_root) == "rulespec-us-co"
    assert jurisdiction_prefix(policy_root) == "us-co"


def test_canonical_rulespec_root_identity_rejects_flat_and_aliased_roots(tmp_path):
    flat = tmp_path / "rulespec-us-co"
    flat.mkdir()
    alias = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(alias, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (alias / "us").mkdir()

    assert canonical_rulespec_root_identity(flat) is None
    assert canonical_rulespec_root_identity(alias / "us") is None
    assert canonical_rulespec_repo_name(alias) is None


def test_country_checkout_identity_requires_existing_directory(tmp_path):
    nonexistent = tmp_path / "rulespec-us"
    regular_file = tmp_path / "files" / "rulespec-us"
    regular_file.parent.mkdir()
    regular_file.write_text("not a checkout\n")

    assert is_policy_repo_root(nonexistent) is False
    assert canonical_rulespec_repo_name(nonexistent) is None
    assert is_policy_repo_root(regular_file) is False
    assert canonical_rulespec_repo_name(regular_file) is None


@pytest.mark.parametrize(
    "root_name",
    ["legislation", "policies", "programs", "regulations", "statutes"],
)
def test_country_checkout_rejects_flat_root_level_content(tmp_path, root_name):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "us").mkdir()
    (checkout / root_name).mkdir()

    assert is_policy_repo_root(checkout) is False
    assert canonical_rulespec_root_identity(checkout / "us") is None


@pytest.mark.parametrize(
    "jurisdiction",
    ["us-", "us--co", "us-co.backup", "us-_scratch"],
)
def test_canonical_identity_rejects_malformed_jurisdiction_names(
    tmp_path,
    jurisdiction,
):
    checkout = tmp_path / "rulespec-us"
    (checkout / "us").mkdir(parents=True)
    malformed = checkout / jurisdiction
    malformed.mkdir()

    assert canonical_rulespec_root_identity(malformed) is None
    assert jurisdiction not in jurisdiction_subdir_names(checkout)


def test_canonical_identity_rejects_symlinked_checkout_and_content_roots(tmp_path):
    real_checkout = tmp_path / "real" / "rulespec-us"
    real_content = real_checkout / "us-co"
    real_content.mkdir(parents=True)
    checkout_alias = tmp_path / "alias" / "rulespec-us"
    checkout_alias.parent.mkdir()
    checkout_alias.symlink_to(real_checkout, target_is_directory=True)
    content_alias = real_checkout / "us-ny"
    external_content = tmp_path / "external-us-ny"
    external_content.mkdir()
    content_alias.symlink_to(external_content, target_is_directory=True)

    assert canonical_rulespec_root_identity(checkout_alias / "us-co") is None
    assert candidate_jurisdiction_content_dirs(checkout_alias, "us-co") == []
    assert canonical_rulespec_root_identity(content_alias) is None
    assert find_policy_repo_root(content_alias / "statutes" / "1.yaml") is None


def test_axiom_workspace_inside_monorepo_is_not_canonical_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    (checkout / "us").mkdir()
    (checkout / "us-dc").mkdir()
    axiom_workspace = checkout / "_axiom"
    (axiom_workspace / "rulespec-us" / "us").mkdir(parents=True)

    assert canonical_rulespec_repo_name(axiom_workspace) is None
    assert candidate_jurisdiction_content_dirs(axiom_workspace, "us") == []
    assert (
        canonical_rulespec_root_identity(axiom_workspace / "rulespec-us" / "us") is None
    )


def test_canonical_identity_observes_new_parent_git_boundary(tmp_path):
    checkout = tmp_path / "rulespec-us"
    policy_root = checkout / "us-co"
    policy_root.mkdir(parents=True)

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us-co"

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)

    assert canonical_rulespec_root_identity(policy_root) is None


def test_canonical_identity_observes_changed_git_origin(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    policy_root.mkdir()

    assert canonical_rulespec_root_identity(policy_root) == "rulespec-us/us"

    subprocess.run(
        ["git", "remote", "set-url", "origin", "https://example.com/not-us.git"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    assert canonical_rulespec_root_identity(policy_root) is None


def test_canonical_identity_rejects_observed_git_boundary_when_git_unavailable(
    monkeypatch,
    tmp_path,
):
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init"], cwd=outer, check=True, capture_output=True)
    nested_root = outer / "nested" / "rulespec-us" / "us-co"
    nested_root.mkdir(parents=True)
    plain_root = tmp_path / "plain" / "rulespec-us" / "us-co"
    plain_root.mkdir(parents=True)

    def unavailable_git(*_args, **_kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr("axiom_encode.repo_routing.subprocess.run", unavailable_git)

    assert canonical_rulespec_root_identity(nested_root) is None
    assert canonical_rulespec_root_identity(plain_root) == "rulespec-us/us-co"


def test_candidate_roots_reject_noncanonical_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        _candidate_rulespec_repo_roots("rulespec-us", checkout)


def test_compile_roots_reject_temp_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")

    pipeline = ValidatorPipeline(
        policy_repo_path=checkout,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        pipeline._rulespec_compile_roots()


def test_canonical_compile_path_rejects_temp_checkout_alias(tmp_path):
    checkout = tmp_path / "rulespec-us-clean.abcd"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    rules_file = checkout / "statutes" / "26" / "63" / "f.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    with pytest.raises(
        UnsafeRulespecContextPath,
        match="exact direct jurisdiction child",
    ):
        _canonical_rulespec_compile_path(rules_file, checkout)


def test_canonical_compile_path_uses_direct_monorepo_jurisdiction_path(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    rules_file = policy_root / "regulations" / "10-ccr-2506-1" / "4.407.31.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    compile_path = _canonical_rulespec_compile_path(rules_file, policy_root)

    assert compile_path == rules_file.resolve()
    assert compile_path.parts[-5:] == (
        "rulespec-us",
        "us-co",
        "regulations",
        "10-ccr-2506-1",
        "4.407.31.yaml",
    )
    assert compile_path.resolve() == rules_file.resolve()


def test_canonical_compile_path_keeps_canonical_country_content_root_segment(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    rules_file = policy_root / "statutes" / "26" / "3121" / "i.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")

    compile_path = _canonical_rulespec_compile_path(rules_file, policy_root)

    assert compile_path == rules_file
    assert compile_path.parts[-5:] == (
        "us",
        "statutes",
        "26",
        "3121",
        "i.yaml",
    )


def test_compile_roots_expose_only_explicit_monorepo_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us-co"
    policy_root.mkdir()

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    assert pipeline._rulespec_compile_roots() == (checkout.resolve(),)
    assert pipeline._rulespec_root_args() == [
        "--rulespec-root",
        str(checkout.resolve()),
    ]


def test_dataset_preserves_canonical_same_repo_refs(tmp_path):
    checkout = tmp_path / "rulespec-us"
    _init_checkout(checkout, "https://github.com/TheAxiomFoundation/rulespec-us.git")
    policy_root = checkout / "us"
    rules_file = policy_root / "statutes" / "26" / "63" / "f.yaml"
    imported_file = policy_root / "statutes" / "26" / "151.yaml"
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
        policy_repo_path=policy_root,
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


def test_dataset_preserves_country_subdir_canonical_refs(
    tmp_path,
):
    checkout = tmp_path / "rulespec-us"
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
    checkout = tmp_path / "rulespec-us"
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
