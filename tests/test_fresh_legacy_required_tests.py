"""Exercise required-test routing without granting refresh/replay privileges."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.prepare_signed_backfill import split_atomic_source_input

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "Validate atomic source inputs",
    "Verify existing signed imports",
    "Encode, review, validate, and apply",
]
CASE = {
    "name": "direct table cell",
    "period": {"period_kind": "month", "start": "2023-10-01", "end": "2023-10-31"},
    "input": {"household_size": 1},
    "required_output": {"income_table": 1215},
}


@pytest.mark.parametrize("version", [2, 3, 4, 5])
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "conflict",
    [
        None,
        "REPAIR_RUN_ID",
        "REPAIR_TESTS_ONLY",
        "QUEUE_ID",
        "DEPENDENT_CITATION",
        "SECOND_DEPENDENT_CITATION",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
        "different_path",
        "source_bundle",
        "canonical_refresh_bundle",
        "require_complete_source_unit",
        "manifest_only_refresh",
        "reviewed_candidate_promotion",
    ],
)
def test_fresh_legacy_required_tests_route_all_stages(stage, conflict, version):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        s["run"] for s in workflow["jobs"]["encode"]["steps"] if s.get("name") == stage
    )
    # Execute the actual dispatch routing block from each protected stage.
    command = command[
        command.index("canonical_refresh_primary_required_test_cases_json=") :
    ]
    command = command.split("source_bundle_args=(", 1)[0]
    command += '\nprintf "%s\\n%s\\n" "$canonical_refresh_primary_required_test_cases_json" "$primary_required_test_cases_json"\n'
    transaction = {
        "schema": f"axiom-encode/atomic-source-transaction/v{version}",
        "source_bundle": [],
        "canonical_refresh_bundle": [],
        "primary_required_test_cases": [CASE],
    }
    if version >= 3:
        transaction["require_complete_source_unit"] = True
    if version >= 4:
        transaction["manifest_only_refresh"] = False
    if version >= 5:
        transaction["reviewed_candidate_promotion"] = False
    payload = split_atomic_source_input(json.dumps(transaction))
    env = {
        **os.environ,
        **{
            key: ""
            for key in [
                "REPAIR_RUN_ID",
                "QUEUE_ID",
                "DEPENDENT_CITATION",
                "SECOND_DEPENDENT_CITATION",
                "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
                "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
            ]
        },
        "REPAIR_TESTS_ONLY": "false",
        "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
        "REPLACE_RULESPEC_PATH": "us/policies/example.yaml",
        "REPLACE_LEGACY_RULESPEC_PATH": "us/policies/example.yaml",
        "primary_required_test_cases_json": json.dumps([CASE]),
    }
    if conflict in {
        "source_bundle",
        "canonical_refresh_bundle",
        "require_complete_source_unit",
        "manifest_only_refresh",
        "reviewed_candidate_promotion",
    }:
        payload[conflict] = (
            ["unexpected"]
            if conflict.endswith("bundle")
            else conflict != "require_complete_source_unit"
        )
    elif conflict == "different_path":
        env["REPLACE_RULESPEC_PATH"] = "us/policies/other.yaml"
    elif conflict:
        env[conflict] = '["unexpected"]' if conflict.endswith("JSON") else "true"
    env["atomic_source_payload"] = json.dumps(payload)
    result = subprocess.run(
        ["bash", "-eu", "-c", command], env=env, capture_output=True, text=True
    )
    if conflict:
        assert result.returncode != 0
        assert "cannot mix transaction modes" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        refresh_cases, signing_cases = result.stdout.splitlines()
        assert json.loads(refresh_cases) == []
        assert json.loads(signing_cases) == [CASE]


@pytest.mark.parametrize("legacy", [True, False])
def test_fresh_replacement_encode_receives_exact_required_contract(tmp_path, legacy):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        s for s in workflow["jobs"]["encode"]["steps"] if s.get("name") == STAGES[2]
    )
    command = step["run"]
    # Run the actual native argument builder, stopping before external repair/signing.
    builder = command.split("run_signed_encode() {", 1)[1].split(
        'if [ -n "${REPAIR_CANDIDATE_ROOT:-}" ]', 1
    )[0]
    invocation = command.rsplit("run_signed_encode \\\n", 1)[1].split("\n  fi", 1)[0]
    invocation = invocation.split("\nfi", 1)[0]
    script = "retained_successor_paths=()\nrun_signed_encode() {" + builder
    script += '\nprintf "%s\\0" "${args[@]}"\n}\nrun_signed_encode \\\n' + invocation
    cases = [
        CASE,
        {
            **CASE,
            "name": "direct table size eight",
            "input": {"household_size": 8},
            "required_output": {"income_table": 4214},
        },
    ]
    env = {
        **os.environ,
        **{key: "" for key in step["env"]},
        "workflow_python": os.sys.executable,
        "backfill_helper": str(ROOT / "scripts/prepare_signed_backfill.py"),
        "PYTHONPATH": str(ROOT / "src"),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_WORKSPACE": str(tmp_path),
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "CITATION": "us/guidance/example",
        "REVIEW_FINDING": "",
        "REPLACE_RULESPEC_PATH": "us/policies/example.yaml",
        "REPLACE_LEGACY_RULESPEC_PATH": "us/policies/example.yaml" if legacy else "",
        "required_imports_enabled": "false",
        "target_require_complete_source_unit": "true",
        "primary_required_test_cases_json": json.dumps(cases),
        "dependent_rulespec_path": "",
        "second_dependent_rulespec_path": "",
    }
    result = subprocess.run(
        ["bash", "-eu", "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.rstrip("\0").split("\0")
    assert "--require-complete-source-unit" in args
    assert ("--replace-legacy-rulespec-path" in args) == legacy
    assert json.loads(args[args.index("--review-contract-json") + 1]) == {
        "schema": "axiom-encode/review-contract/v2",
        "citation": env["CITATION"],
        "rulespec_path": env["REPLACE_RULESPEC_PATH"],
        "required_deferred_outputs": [],
        "required_test_cases": cases,
    }
