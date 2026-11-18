"""Public failed seeds: exact reconstruction and retained full case contracts."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode import reviewed_repair_patch as patch
from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.bound_schedule_structure import recognize_bound_schedule_structure
from axiom_encode.harness.bound_form_operands import bind_ordered_form
from axiom_encode.harness.bound_schedule_operands import bind_schedule
from axiom_encode.prepare_signed_backfill import split_atomic_source_input

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ("issue233-ca-nl479-2025-43477826390", 43477826390, 17, 79, 13, 59),
    ("issue233-ca-nb-s12-2025-43477831512", 43477831512, 25, 45, 21, 41),
]


@pytest.fixture(params=TARGETS, ids=[x[0] for x in TARGETS])
def registered_target(request):
    patch_id, run, *_ = request.param
    reg = patch.REVIEWED_REPAIR_PATCHES[patch_id]
    folder = ROOT / "reviewed-repair-patches" / patch_id
    raws = tuple((folder / name).read_bytes() for name in patch.MEMBERS)
    assert (
        tuple((patch.git_blob_id(raw), patch.sha256(raw), len(raw)) for raw in raws)
        == reg.members
    )
    package = patch.ReviewedRepairPackage(patch_id, reg.commit, *raws)
    fixture = ROOT / "tests/fixtures" / f"reviewed_repair_{run}"
    patch.validate_manifest(package.manifest_bytes, patch_id)
    return request.param, package, fixture


def bound(package, text, source):
    citation = package.manifest["citation"]
    if "nl479" in citation:
        form = recognize_bound_form_structure(source, citation)
        binder = bind_ordered_form
    else:
        form = recognize_bound_schedule_structure(source, citation)
        binder = bind_schedule
    assert form is not None
    return binder(
        form, text, {citation: source}, root_relative=package.manifest["rulespec_path"]
    )


def test_exact_seed_restores_binding_and_preserves_every_case(registered_target):
    (
        (_, run, cases_count, assertions, required_count, required_assertions),
        package,
        fixture,
    ) = registered_target
    seed = (fixture / "seed.yaml").read_bytes()
    source = (fixture / "source.txt").read_bytes()
    manifest = package.manifest
    assert manifest["seed"]["run_id"] == str(run)
    assert patch.sha256(seed) == manifest["seed"]["rulespec_sha256"]
    assert patch.sha256(source) == manifest["source"]["body_sha256"]
    assert patch.sha256(package.tests_bytes) == manifest["seed"]["tests_sha256"]
    after, tests = patch.reconstruct_postimage(
        package, rulespec=seed, tests=package.tests_bytes
    )
    assert patch.sha256(after) == manifest["expected_postimage"]["rulespec_sha256"]
    assert tests == package.tests_bytes
    cases = yaml.safe_load(tests)
    assert len(cases) == cases_count
    assert sum(len(case["output"]) for case in cases) == assertions
    assert manifest["changes"]["cases"] == {"added": [], "removed": [], "replaced": []}
    contract = json.loads((fixture / "transaction.json").read_bytes())
    assert patch.sha256(patch.canonical_json(contract)) == manifest["contract_sha256"]
    normalized = split_atomic_source_input(json.dumps(contract))
    assert normalized["require_complete_source_unit"] is True
    assert normalized["repair_mode"] == "full_artifact"
    required = normalized["primary_required_test_cases"]
    assert len(required) == required_count
    assert sum(len(case["required_output"]) for case in required) == required_assertions
    by_name = {case["name"]: case for case in cases}
    assert len(by_name) == len(cases)
    for anchor in required:
        actual = by_name[anchor["name"]]
        for key in ("input", "period"):
            assert patch.canonical_json(actual[key]) == patch.canonical_json(
                anchor[key]
            )
        assert patch.canonical_json(actual["output"]) == patch.canonical_json(
            anchor["required_output"]
        )
    assert bound(package, seed.decode(), source.decode()) is None
    assert bound(package, after.decode(), source.decode()) is not None
    before_rules = {r["name"]: r for r in yaml.safe_load(seed)["rules"]}
    after_rules = {r["name"]: r for r in yaml.safe_load(after)["rules"]}
    assert set(before_rules) - set(after_rules) == set(
        manifest["changes"]["rules"]["removed"]
    )
    for name, rule in after_rules.items():
        old = before_rules[name]
        assert [v["formula"] for v in rule["versions"]] == [
            v["formula"] for v in old["versions"]
        ]
        for key in ("kind", "type", "entity", "unit", "period"):
            assert rule.get(key) == old.get(key)
    if "nb-s12" in manifest["citation"]:
        assert manifest["changes"]["rules"]["removed"] == ["nb_s12_schedule_tax_year"]
        restored = yaml.safe_load(after)
        restored["rules"].append(before_rules["nb_s12_schedule_tax_year"])
        assert bound(package, yaml.safe_dump(restored), source.decode()) is None
    else:
        assert manifest["changes"]["rules"]["removed"] == []


@pytest.mark.parametrize(
    "mutation", ["drop_case", "change_output", "change_description", "change_seed"]
)
def test_changed_seed_or_companion_cannot_be_relabelled(registered_target, mutation):
    _, package, fixture = registered_target
    seed = (fixture / "seed.yaml").read_bytes()
    cases = yaml.safe_load(package.tests_bytes)
    if mutation == "drop_case":
        cases.pop()
    elif mutation == "change_output":
        name = next(iter(cases[0]["output"]))
        cases[0]["output"][name] = "unreviewed"
    elif mutation == "change_description":
        cases[0]["description"] = "changed claimant facts"
    else:
        seed += b"\n# different authenticated seed\n"
    altered = (
        package
        if mutation == "change_seed"
        else replace(package, tests_bytes=yaml.safe_dump(cases).encode())
    )
    with pytest.raises(ValueError):
        patch.reconstruct_postimage(altered, rulespec=seed, tests=package.tests_bytes)


def test_removed_inventory_or_source_binding_cannot_be_changed(registered_target):
    _, package, fixture = registered_target
    manifest = copy.deepcopy(package.manifest)
    manifest["changes"]["rules"]["replaced"] = []
    altered = replace(package, manifest_bytes=patch.canonical_json(manifest))
    with pytest.raises(ValueError):
        patch.reconstruct_postimage(
            altered,
            rulespec=(fixture / "seed.yaml").read_bytes(),
            tests=package.tests_bytes,
        )
