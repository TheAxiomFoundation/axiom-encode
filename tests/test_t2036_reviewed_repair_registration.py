"""Authenticated T2036 repair: preserve the approved cases and closed inventory."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.prepare_signed_backfill import split_atomic_source_input
from axiom_encode.yaml_serialization import safe_dump_yaml

ROOT = Path(__file__).resolve().parents[1]
PATCH_ID = "issue233-ca-t2036-2025-39337066830"
SEED = ROOT / "tests/fixtures/reviewed_repair_39337066830"
EXPECTED = ROOT / "tests/fixtures/t2036_country_case_selection"


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


def test_exact_reconstruction_keeps_the_full_reviewed_companion(package):
    rules, tests = reconstruct(package)
    assert rules == (EXPECTED / "candidate.yaml").read_bytes()
    assert tests == (EXPECTED / "candidate.test.yaml").read_bytes()
    assert (
        patch.sha256((EXPECTED / "source.txt").read_bytes())
        == (package.manifest["source"]["body_sha256"])
    )
    cases = patch.strict_yaml(tests, tests=True)
    assert len(cases) == 45
    assert sum(len(case["output"]) for case in cases) == 401
    assert (
        sum(
            len(value) if isinstance(value, list) else 1
            for case in cases
            for value in case["output"].values()
        )
        == 402
    )
    added_projection = cases[-1]
    assert added_projection["name"] == "transfer_equal_existing_child_credit_outputs"
    assert list(added_projection["output"].values()) == [[150, 150]]


def test_original_required_inputs_periods_and_outputs_are_not_weakened(package):
    contract = json.loads((SEED / "transaction.json").read_bytes())
    assert (
        patch.sha256(patch.canonical_json(contract))
        == (package.manifest["contract_sha256"])
    )
    normalized = split_atomic_source_input(json.dumps(contract))
    assert normalized["repair_mode"] == "full_artifact"
    assert normalized["require_complete_source_unit"] is True
    required = normalized["primary_required_test_cases"]
    assert len(required) == 13
    assert sum(len(case["required_output"]) for case in required) == 197
    _, tests = reconstruct(package)
    cases = patch.strict_yaml(tests, tests=True)
    by_name = {case["name"]: case for case in cases}
    assert len(by_name) == len(cases)
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
    removed = package.manifest["changes"]["cases"]["removed"]
    assert removed == [
        "auto_zero_t2036_amount_to_form_428",
        "auto_zero_t2036_amount_to_form_428mj",
    ]
    assert not set(removed) & {case["name"] for case in required}


@pytest.mark.parametrize("member", ["rulespec", "tests"])
def test_different_failed_seed_cannot_use_the_registered_patch(package, member):
    seed = (SEED / "seed.yaml").read_bytes()
    tests = (SEED / "seed.test.yaml").read_bytes()
    if member == "rulespec":
        seed += b"\n# different failed candidate\n"
    else:
        tests += b"\n# different failed companion\n"
    with pytest.raises(ValueError):
        patch.reconstruct_postimage(package, rulespec=seed, tests=tests)


@pytest.mark.parametrize("field", ["added", "removed", "replaced"])
def test_undeclared_case_changes_are_refused(package, field):
    manifest = copy.deepcopy(package.manifest)
    assert manifest["changes"]["cases"][field]
    manifest["changes"]["cases"][field] = []
    altered = replace(package, manifest_bytes=patch.canonical_json(manifest))
    with pytest.raises(ValueError):
        reconstruct(altered)


def test_relational_projection_cannot_be_replaced_with_a_scalar(package):
    cases = patch.strict_yaml(package.tests_bytes, tests=True)
    output = cases[-1]["output"]
    assert list(output.values()) == [[150, 150]]
    output[next(iter(output))] = 300
    altered = replace(package, tests_bytes=safe_dump_yaml(cases).encode())
    with pytest.raises(ValueError):
        reconstruct(altered)
