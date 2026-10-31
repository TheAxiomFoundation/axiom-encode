"""Execute synthetic guarded relation-role inputs through both companion runners."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from axiom_encode.companion_relations import pair_relation_assignment_issues
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from axiom_encode.harness.validator_pipeline import (
    find_missing_derived_companion_output_issues,
)
from tests.test_companion_declared_entities import _companions

FIXTURES = Path(__file__).parent / "fixtures" / "relation_roles"
TARGET = "us:policies/example/guarded_household"
SOURCE = "us/guidance/axiom/relation-roles"


def reference_fixture(tmp_path):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/guarded_household.yaml"
    module.parent.mkdir(parents=True)
    for suffix in (".yaml", ".test.yaml"):
        module.with_suffix(suffix).write_text(
            (FIXTURES / f"guarded_household{suffix}").read_text()
        )
    return root, module


def test_guarded_reference_has_proof_atoms_for_every_policy_rule():
    content = (FIXTURES / "guarded_household.yaml").read_text()
    result = validate_rulespec_proofs(
        content,
        require_policy_proofs=True,
        source_texts={SOURCE: (FIXTURES / "source.txt").read_text()},
    )
    assert result.issues == []
    assert result.atoms_checked == 11


@pytest.mark.parametrize(
    "rule_name",
    [
        "lives_with",
        "snap_tie",
        "snap_within_two_ties",
        "snap_household_member",
        "snap_participating_member",
    ],
)
def test_relation_predicate_forms_keep_missing_proof_failures(rule_name):
    payload = yaml.safe_load((FIXTURES / "guarded_household.yaml").read_text())
    rule = next(rule for rule in payload["rules"] if rule["name"] == rule_name)
    del rule["metadata"]["proof"]
    result = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={SOURCE: (FIXTURES / "source.txt").read_text()},
    )
    assert result.issues == [
        f"Proof missing: rule `{rule_name}` is policy-bearing and must declare "
        "`metadata.proof.atoms`."
    ]


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_guarded_reference_compiles_and_runs_companion_memberships(
    tmp_path, real_engine, executor
):
    root, module = reference_fixture(tmp_path)
    assert (
        _companions(
            executor,
            binary=real_engine.binary,
            engine_root=real_engine.root,
            root=root,
            module=module,
            artifact=tmp_path / "compiled.json",
        )
        == []
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_guarded_long_chain_without_supplied_ids_fails_in_real_engine(
    tmp_path, real_engine, executor
):
    root, module = reference_fixture(tmp_path)
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    case = next(
        case for case in cases if case["name"] == "three_tie_chain_reads_supplied_ids"
    )
    for row in case["tables"]["Person"]:
        del row[f"{TARGET}#input.snap_household_unit"]
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump([case]))
    issues = _companions(
        executor,
        binary=real_engine.binary,
        engine_root=real_engine.root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert issues
    assert "snap_household_unit" in "\n".join(issues)
    assert "missing" in "\n".join(issues).lower()


@pytest.mark.parametrize(
    "relation",
    [
        "member_of_household",
        "spouse_of",
        "parent_of",
        "under_parental_control_of",
    ],
)
def test_every_base_pair_fact_requires_assignment_even_when_false(relation):
    payload = yaml.safe_load((FIXTURES / "guarded_household.yaml").read_text())
    schemas = []
    for rule in payload["rules"]:
        if rule["kind"] == "data_relation":
            schemas.append({"name": rule["name"], **rule["data_relation"]})
        elif rule["kind"] == "derived_relation":
            schemas.append(
                {"name": rule["name"], "derivation": rule["derived_relation"]}
            )
    case = copy.deepcopy(
        yaml.safe_load((FIXTURES / "guarded_household.test.yaml").read_text())[0]
    )
    assert pair_relation_assignment_issues(schemas, case, case["name"]) == []
    del case["input"][f"{TARGET}#relation.{relation}"]
    assert pair_relation_assignment_issues(schemas, case, case["name"]) == [
        f"Test case `{case['name']}` must assign pair relation `{relation}`, "
        "including `[]` when no tuples hold."
    ]


def test_reference_companions_cover_views_through_scalar_outputs(tmp_path):
    root, module = reference_fixture(tmp_path)
    assert (
        find_missing_derived_companion_output_issues(
            module.read_text(),
            yaml.safe_load(module.with_suffix(".test.yaml").read_text()),
            rules_file=module,
            policy_repo_path=root / "us",
        )
        == []
    )


def test_unused_relation_view_reports_exact_missing_coverage(tmp_path):
    root, module = reference_fixture(tmp_path)
    payload = yaml.safe_load(module.read_text())
    unused = copy.deepcopy(
        next(rule for rule in payload["rules"] if rule["name"] == "snap_tie")
    )
    unused["name"] = "unused_members"
    payload["rules"].append(unused)
    assert find_missing_derived_companion_output_issues(
        yaml.safe_dump(payload),
        yaml.safe_load(module.with_suffix(".test.yaml").read_text()),
        rules_file=module,
        policy_repo_path=root / "us",
    ) == [
        "Derived relation missing companion output coverage: "
        f"`{TARGET}#unused_members` is not reached by an asserted scalar output "
        "in the companion `.test.yaml` file."
    ]


@pytest.mark.parametrize(
    "formula,covered", [("min(age, 22)", False), ("len(min)", True)]
)
def test_relation_coverage_distinguishes_builtin_call_and_bare_relation(
    tmp_path, formula, covered
):
    root, module = reference_fixture(tmp_path)
    payload = yaml.safe_load(module.read_text())
    unused = copy.deepcopy(
        next(rule for rule in payload["rules"] if rule["name"] == "snap_tie")
    )
    unused["name"] = "min"
    payload["rules"].append(unused)
    scalar = next(
        rule for rule in payload["rules"] if rule["name"] == "snap_household_size"
    )
    scalar["versions"][0]["formula"] = formula
    issues = find_missing_derived_companion_output_issues(
        yaml.safe_dump(payload),
        yaml.safe_load(module.with_suffix(".test.yaml").read_text()),
        rules_file=module,
        policy_repo_path=root / "us",
    )
    assert issues == (
        []
        if covered
        else [
            "Derived relation missing companion output coverage: "
            f"`{TARGET}#min` is not reached by an asserted scalar output "
            "in the companion `.test.yaml` file."
        ]
    )
