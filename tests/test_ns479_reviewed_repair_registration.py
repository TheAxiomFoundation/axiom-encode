"""Preserve the authenticated NS479 seed contract and all reviewed controls."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.prepare_signed_backfill import split_atomic_source_input
from axiom_encode.yaml_serialization import safe_dump_yaml

ROOT = Path(__file__).resolve().parents[1]
PATCH_ID = "issue233-ca-ns479-2025-43506739931"
SEED = ROOT / "tests/fixtures/reviewed_repair_43506739931"


@pytest.fixture
def package():
    registered = patch.REVIEWED_REPAIR_PATCHES[PATCH_ID]
    folder = ROOT / "reviewed-repair-patches" / PATCH_ID
    members = tuple((folder / name).read_bytes() for name in patch.MEMBERS)
    assert (
        tuple((patch.git_blob_id(raw), patch.sha256(raw), len(raw)) for raw in members)
        == registered.members
    )
    patch.validate_manifest(members[0], PATCH_ID)
    return patch.ReviewedRepairPackage(PATCH_ID, registered.commit, *members)


def reconstruct(package):
    return patch.reconstruct_postimage(
        package,
        rulespec=(SEED / "seed.yaml").read_bytes(),
        tests=(SEED / "seed.test.yaml").read_bytes(),
    )


def test_all_native_cases_survive_with_only_reviewed_scalar_types_restored(package):
    rules, tests = reconstruct(package)
    cases = patch.strict_yaml(tests, tests=True)
    original = patch.strict_yaml((SEED / "seed.test.yaml").read_bytes(), tests=True)
    assert len(cases) == 23
    assert sum(len(case["output"]) for case in cases) == 81
    assert [c["name"] for c in cases] == [c["name"] for c in original]
    changed_types = []
    for before, after in zip(original, cases):
        assert before == after  # Numeric values, inputs and periods are unchanged.
        for name, value in before["output"].items():
            actual = after["output"][name]
            if type(value) is not type(actual):
                assert type(value) is int and type(actual) is float
                changed_types.append(before["name"])
    assert len(changed_types) == 10
    assert set(changed_types) == {
        "per_child_900_100",
        "new_explicit_refinement_of_underdetermined_historical_case",
        "volunteer_not_applicable_same_entered_rows",
        "claimed_by_others_equals_column_d",
    }
    names = {r["name"] for r in patch.strict_yaml(rules, tests=False)["rules"]}
    assert len(names) == 21
    assert not names & {"form_tax_year", "printed_child_table_entry_spaces"}
    assert (
        patch.sha256((SEED / "source.txt").read_bytes())
        == package.manifest["source"]["body_sha256"]
    )


def test_original_eight_required_cases_remain_typed_exact(package):
    contract = json.loads((SEED / "transaction.json").read_bytes())
    assert (
        patch.sha256(patch.canonical_json(contract))
        == package.manifest["contract_sha256"]
    )
    normalized = split_atomic_source_input(json.dumps(contract))
    assert normalized["repair_mode"] == "full_artifact"
    assert normalized["require_complete_source_unit"] is True
    required = normalized["primary_required_test_cases"]
    assert len(required) == 8
    assert sum(len(c["required_output"]) for c in required) == 17
    _, tests = reconstruct(package)
    by_name = {c["name"]: c for c in patch.strict_yaml(tests, tests=True)}
    for anchor in required:
        actual = by_name[anchor["name"]]
        for key in ("input", "period"):
            assert patch.canonical_json(actual[key]) == patch.canonical_json(
                anchor[key]
            )
        for name, expected in anchor["required_output"].items():
            assert patch.canonical_json(actual["output"][name]) == patch.canonical_json(
                expected
            )


def test_undeclared_rule_removal_is_refused(package):
    manifest = copy.deepcopy(package.manifest)
    manifest["changes"]["rules"]["removed"] = []
    with pytest.raises(ValueError, match="semantic inventory"):
        reconstruct(replace(package, manifest_bytes=patch.canonical_json(manifest)))


def test_numerically_equal_but_unreviewed_test_type_is_refused(package):
    cases = patch.strict_yaml(package.tests_bytes, tests=True)
    case = next(c for c in cases if c["name"] == "per_child_900_100")
    output = next(k for k in case["output"] if k.endswith("column_d_capped_expenses"))
    assert type(case["output"][output]) is float
    case["output"][output] = int(case["output"][output])
    with pytest.raises(ValueError, match="postimage"):
        reconstruct(replace(package, tests_bytes=safe_dump_yaml(cases).encode()))


@pytest.mark.parametrize("member", ["rulespec", "tests"])
def test_another_failed_seed_cannot_use_this_registration(package, member):
    seed = (SEED / "seed.yaml").read_bytes()
    tests = (SEED / "seed.test.yaml").read_bytes()
    if member == "rulespec":
        seed += b"\n# different native candidate\n"
    else:
        tests += b"\n# different native companion\n"
    with pytest.raises(ValueError, match="seed bytes"):
        patch.reconstruct_postimage(package, rulespec=seed, tests=tests)
