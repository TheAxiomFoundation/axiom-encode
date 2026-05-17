"""Tests for RuleSpec repository identity helpers."""

import os
import subprocess
from pathlib import Path

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
