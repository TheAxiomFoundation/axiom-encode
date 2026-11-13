"""Real registered Schedule 2 reconstruction and complete seed preservation."""

import base64
import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.prepare_signed_backfill import split_atomic_source_input

ROOT = Path(__file__).resolve().parents[1]
PATCH_ID = "ca-schedule2-2025-full60-seed43376780727"
FIXTURE = ROOT / "tests/fixtures/reviewed_repair_schedule2_43376780727"
FOLDER = ROOT / "reviewed-repair-patches" / PATCH_ID


@pytest.fixture
def package():
    registration = patch.REVIEWED_REPAIR_PATCHES[PATCH_ID]
    raws = tuple((FOLDER / name).read_bytes() for name in patch.MEMBERS)
    assert (
        tuple((patch.git_blob_id(raw), patch.sha256(raw), len(raw)) for raw in raws)
        == registration.members
    )
    return patch.ReviewedRepairPackage(PATCH_ID, registration.commit, *raws)


def binding(package):
    return patch.build_repair_binding(
        package,
        rulespec=(FIXTURE / "seed.yaml").read_bytes(),
        tests=(FIXTURE / "seed.test.yaml").read_bytes(),
    )


def test_real_seed_reconstructs_full60_and_preserves_all_original66(package):
    restored, (rules, tests) = patch.verify_repair_binding(binding(package))
    assert restored == package
    assert rules == package.rules_bytes
    assert tests == package.tests_bytes
    cases = yaml.safe_load(tests)
    assert len(cases) == 60
    assert sum(len(case["output"]) for case in cases) == 263
    original = yaml.safe_load((FIXTURE / "seed.test.yaml").read_bytes())
    assert len(original) == 10
    assert sum(len(case["output"]) for case in original) == 66
    by_name = {case["name"]: case for case in cases}
    renamed = {"line12_floor", "threshold_above_stipulated_line76"}
    explicit_disability = {
        "paperwork_and_nonfiling_instructions",
        "auto_positive_schedule_2_send_new_t2201",
    }
    for seed in original:
        expected = copy.deepcopy(seed)
        if seed["name"] in renamed:
            expected["name"] = "seed_original_" + seed["name"]
        if seed["name"] in explicit_disability:
            expected["input"][
                "ca:policies/cra/t1-2025/federal-spouse-transfer-amounts"
                "#input.spouse_disability_amount_line_31600"
            ] = 10138
        assert patch.canonical_json(by_name[expected["name"]]) == patch.canonical_json(
            expected
        )
    assert package.manifest["changes"]["cases"]["removed"] == []


def test_original_contract_is_bound_without_enlargement(package):
    transaction = json.loads((FIXTURE / "transaction.json").read_bytes())
    assert (
        patch.sha256(patch.canonical_json(transaction))
        == package.manifest["contract_sha256"]
    )
    parsed = split_atomic_source_input(json.dumps(transaction))
    assert parsed["repair_mode"] == "full_artifact"
    assert parsed["require_complete_source_unit"] is True
    anchors = parsed["primary_required_test_cases"]
    assert len(anchors) == 1
    assert len(anchors[0]["required_output"]) == 2
    case = next(
        c
        for c in yaml.safe_load(package.tests_bytes)
        if c["name"] == anchors[0]["name"]
    )
    assert patch.canonical_json(case["input"]) == patch.canonical_json(
        anchors[0]["input"]
    )
    for name, value in anchors[0]["required_output"].items():
        assert patch.canonical_json(case["output"][name]) == patch.canonical_json(value)


@pytest.mark.parametrize("member", patch.MEMBERS)
def test_changed_package_member_is_refused(package, member):
    receipt = binding(package)
    raw = base64.b64decode(receipt["package"][member])
    receipt["package"][member] = base64.b64encode(raw + b"\n").decode()
    with pytest.raises(ValueError, match="differs from registry"):
        patch.verify_repair_binding(receipt)


@pytest.mark.parametrize("kind", ["rulespec", "tests"])
def test_changed_seed_is_refused(package, kind):
    receipt = binding(package)
    raw = base64.b64decode(receipt["seed"][kind])
    receipt["seed"][kind] = base64.b64encode(raw + b"\n").decode()
    with pytest.raises(ValueError, match="seed bytes differ"):
        patch.verify_repair_binding(receipt)


@pytest.mark.parametrize("field", ["package_commit", "patch_id"])
def test_changed_registration_identity_is_refused(package, field):
    receipt = binding(package)
    receipt[field] = "0" * 40 if field == "package_commit" else "unregistered"
    with pytest.raises(ValueError, match="registered"):
        patch.verify_repair_binding(receipt)
