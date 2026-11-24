"""Exact public package reconstruction; no protected signer or runtime is mocked."""

import base64
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.harness.evals import _preserves_companion_test_cases
from axiom_encode.harness.student_loan_schedule_binding import (
    bind_student_loan_schedule,
)
from axiom_encode.prepare_signed_backfill import split_atomic_source_input
from axiom_encode.student_loan_schedule_source import recognize_student_loan_schedule

ROOT = Path(__file__).resolve().parents[1]
PATCH_ID = "issue233-ca-schedule-m-2025-43445779735"
FIXTURE = ROOT / "tests/fixtures/reviewed_repair_schedule_m_43445779735"
FOLDER = ROOT / "reviewed-repair-patches" / PATCH_ID
SOURCE = ROOT / "tests/fixtures/student_loan_schedule/source.txt"


@pytest.fixture
def package():
    registration = patch.REVIEWED_REPAIR_PATCHES[PATCH_ID]
    raws = tuple((FOLDER / name).read_bytes() for name in patch.MEMBERS)
    # Uses the real entry, without fixture registration or assuming a full Git clone.
    assert (
        tuple((patch.git_blob_id(raw), patch.sha256(raw), len(raw)) for raw in raws)
        == registration.members
    )
    return patch.ReviewedRepairPackage(PATCH_ID, registration.commit, *raws)


def test_exact_public_seed_reconstructs_closed_graph_and_preserves_all_30_cases(
    package,
):
    seed = (FIXTURE / "seed.yaml").read_bytes()
    original_tests = package.tests_bytes
    rules, tests = patch.reconstruct_postimage(
        package, rulespec=seed, tests=original_tests
    )
    assert tests == original_tests  # All seed cases, not only the required subset.
    cases = yaml.safe_load(tests)
    assert len(cases) == 30
    assert sum(len(case["output"]) for case in cases) == 60
    manifest = package.manifest
    source = recognize_student_loan_schedule(SOURCE.read_text(), manifest["citation"])
    assert source is not None
    assert patch.sha256(SOURCE.read_bytes()) == manifest["source"]["body_sha256"]
    assert bind_student_loan_schedule(source, seed.decode()) is None
    bound = bind_student_loan_schedule(source, rules.decode())
    assert bound is not None
    assert len(bound.rule_roles) == 7
    assert len(bound.input_roles) == 9
    assert manifest["changes"]["cases"] == {"added": [], "replaced": [], "removed": []}
    assert len(manifest["changes"]["rules"]["removed"]) == 4
    assert manifest["seed"]["run_id"] == "43445779735"
    assert manifest["seed"]["encoder_commit"] == (
        "5da8cd49dadb880247d5c275be442e02fa250edd"
    )


def test_original_18_typed_contracts_remain_bound_without_enlargement(package):
    transaction = json.loads((FIXTURE / "transaction.json").read_bytes())
    assert (
        patch.sha256(patch.canonical_json(transaction))
        == (package.manifest["contract_sha256"])
    )
    normalized = split_atomic_source_input(json.dumps(transaction))
    assert normalized["repair_mode"] == "full_artifact"
    assert normalized["require_complete_source_unit"] is True
    anchors = normalized["primary_required_test_cases"]
    assert len(anchors) == 18
    assert sum(len(case["required_output"]) for case in anchors) == 48
    by_name = {case["name"]: case for case in yaml.safe_load(package.tests_bytes)}
    for anchor in anchors:
        authored = {
            "name": anchor["name"],
            "period": anchor["period"],
            "input": anchor["input"],
            "output": anchor["required_output"],
        }
        # Canonical JSON distinguishes Boolean true from integer 1.
        assert patch.canonical_json(by_name[anchor["name"]]) == patch.canonical_json(
            authored
        )
    assert _preserves_companion_test_cases(
        package.tests_bytes.decode(), package.tests_bytes.decode(), anchors
    )


def test_offline_consumer_reconstructs_real_registered_bytes(package):
    seed = (FIXTURE / "seed.yaml").read_bytes()
    binding = patch.build_repair_binding(
        package, rulespec=seed, tests=package.tests_bytes
    )
    restored, postimage = patch.verify_repair_binding(binding)
    assert restored == package
    assert (
        patch.sha256(postimage[0])
        == package.manifest["expected_postimage"]["rulespec_sha256"]
    )
    assert postimage[1] == package.tests_bytes
    # This is an unsigned reconstruction receipt, not live protected admission.
    assert "signature" not in binding


@pytest.mark.parametrize("member", patch.MEMBERS)
def test_consumer_refuses_any_unregistered_member_bytes(package, member):
    binding = patch.build_repair_binding(
        package,
        rulespec=(FIXTURE / "seed.yaml").read_bytes(),
        tests=package.tests_bytes,
    )
    raw = base64.b64decode(binding["package"][member])
    binding["package"][member] = base64.b64encode(raw + b"\n").decode()
    with pytest.raises(ValueError, match="differs from registry"):
        patch.verify_repair_binding(binding)


@pytest.mark.parametrize("field", ["package_commit", "patch_id"])
def test_consumer_refuses_rebinding_to_another_commit_or_patch(package, field):
    binding = patch.build_repair_binding(
        package,
        rulespec=(FIXTURE / "seed.yaml").read_bytes(),
        tests=package.tests_bytes,
    )
    binding[field] = "0" * 40 if field == "package_commit" else "unregistered"
    with pytest.raises(ValueError, match="registered"):
        patch.verify_repair_binding(binding)


@pytest.mark.parametrize("seed_kind", ["rulespec", "tests"])
def test_consumer_refuses_seed_byte_changes(package, seed_kind):
    binding = patch.build_repair_binding(
        package,
        rulespec=(FIXTURE / "seed.yaml").read_bytes(),
        tests=package.tests_bytes,
    )
    original = base64.b64decode(binding["seed"][seed_kind])
    binding["seed"][seed_kind] = base64.b64encode(original + b"\n").decode()
    with pytest.raises(ValueError, match="seed bytes differ"):
        patch.verify_repair_binding(binding)


@pytest.mark.parametrize(
    "field", ["rulespec_base_commit", "seed", "source", "contract_sha256", "engine_ref"]
)
def test_measured_source_seed_base_engine_and_contract_cannot_drift(package, field):
    actual = {
        key: copy.deepcopy(package.manifest[key])
        for key in (
            "citation",
            "rulespec_path",
            "rulespec_base_commit",
            "predecessor",
            "seed",
            "source",
            "contract_sha256",
            "engine_ref",
        )
    }
    if field == "seed":
        actual[field]["run_id"] = "43445779736"
    elif field == "source":
        actual[field]["body_sha256"] = "0" * 64
    else:
        actual[field] = "0" * len(actual[field])
    with pytest.raises(ValueError, match="authenticated execution bindings"):
        patch.verify_bindings(package, actual)


def test_undeclared_case_removal_and_postimage_drift_refuse(package):
    seed = (FIXTURE / "seed.yaml").read_bytes()
    dropped_case = yaml.safe_load(package.tests_bytes)[0]["name"]
    altered_tests = yaml.safe_dump(
        [
            {"name": dropped_case, "repair_remove": True},
            *yaml.safe_load(package.tests_bytes)[1:],
        ],
        sort_keys=False,
    ).encode()
    with pytest.raises(ValueError, match="postimage differs"):
        patch.reconstruct_postimage(
            replace(package, tests_bytes=altered_tests),
            rulespec=seed,
            tests=package.tests_bytes,
        )
    altered_rules = package.rules_bytes.replace(b"max(0,", b"max(1,", 1)
    assert altered_rules != package.rules_bytes
    with pytest.raises(ValueError, match="postimage differs"):
        patch.reconstruct_postimage(
            replace(package, rules_bytes=altered_rules),
            rulespec=seed,
            tests=package.tests_bytes,
        )
