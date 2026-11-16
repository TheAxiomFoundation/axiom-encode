"""Bind the T936 full-unit repair to its reviewed cases and real failed seed."""

import json
import re
from pathlib import Path

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.prepare_signed_backfill import split_atomic_source_input

ROOT = Path(__file__).resolve().parents[1]
PATCH_ID = "issue233-ca-t936-2025-43812124952"
SEED = ROOT / "tests/fixtures/reviewed_repair_43812124952"


def test_registered_t936_reconstruction_preserves_source_and_native_controls():
    record = patch.REVIEWED_REPAIR_PATCHES[PATCH_ID]
    folder = ROOT / "reviewed-repair-patches" / PATCH_ID
    members = tuple((folder / name).read_bytes() for name in patch.MEMBERS)
    assert (
        tuple((patch.git_blob_id(raw), patch.sha256(raw), len(raw)) for raw in members)
        == record.members
    )
    manifest = patch.validate_manifest(members[0], PATCH_ID)
    package = patch.ReviewedRepairPackage(PATCH_ID, record.commit, *members)
    original_tests = (SEED / "seed.test.yaml").read_bytes()
    _, tests = patch.reconstruct_postimage(
        package, rulespec=(SEED / "seed.yaml").read_bytes(), tests=original_tests
    )
    cases = patch.strict_yaml(tests, tests=True)
    by_name = {c["name"]: c for c in cases}
    assert len(by_name) == len(cases) == 91
    assert sum(len(c["output"]) for c in cases) == 193

    guidance = (ROOT / "review-guidance/ca/t936-2025-full-unit.json").read_bytes()
    assert patch.sha256(guidance) == (
        "f83d2b94793d46d9c5d106f7393dca79e5b19c1cf1aa63de766340ee26e4c070"
    )
    _, companion = re.findall(
        r"```yaml\n(.*?)\n```", json.loads(guidance)["finding"], re.S
    )
    reviewed = patch.strict_yaml(companion.encode(), tests=True)
    assert len(reviewed) == 85
    assert sum(len(c["output"]) for c in reviewed) == 177
    for case in reviewed:
        assert patch.canonical_json(by_name[case["name"]]) == patch.canonical_json(case)

    for case in patch.strict_yaml(original_tests, tests=True):
        actual = by_name[case["name"]]
        for field in ("period", "output"):
            assert patch.canonical_json(actual[field]) == patch.canonical_json(
                case[field]
            )

    contract = (SEED / "transaction.json").read_bytes()
    assert patch.sha256(contract) == manifest["contract_sha256"]
    normalized = split_atomic_source_input(contract.decode())
    assert normalized["require_complete_source_unit"] is True
    for anchor in normalized["primary_required_test_cases"]:
        actual = by_name[anchor["name"]]
        for field in ("period", "input"):
            assert patch.canonical_json(actual[field]) == patch.canonical_json(
                anchor[field]
            )
        for name, value in anchor["required_output"].items():
            assert patch.canonical_json(actual["output"][name]) == patch.canonical_json(
                value
            )
