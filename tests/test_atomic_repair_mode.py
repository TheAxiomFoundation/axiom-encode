"""Explicit repair execution is separate from the immutable source contract."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode.prepare_signed_backfill import (
    immutable_atomic_source_contract,
    repair_execution_metadata,
    resolve_atomic_repair_mode,
    split_atomic_source_input,
)
from scripts.extract_repair_candidate import _repair_lane_for_atomic_source

ROOT = Path(__file__).resolve().parents[1]
CASE = {
    "name": "required",
    "period": {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    "input": {"cost": 100},
    "required_output": {"amount": 39},
}
DISPATCH = {
    "REPAIR_RUN_ID": "12345",
    "GITHUB_RUN_ID": "67890",
    "CITATION": "ca/policy/example/benefit",
    "REPLACE_RULESPEC_PATH": "ca/policies/example/benefit.yaml",
}


def transaction(version=3):
    result = {
        "schema": f"axiom-encode/atomic-source-transaction/v{version}",
        "source_bundle": [],
        "canonical_refresh_bundle": [],
        "primary_required_test_cases": [copy.deepcopy(CASE)],
    }
    if version >= 3:
        result["require_complete_source_unit"] = True
    if version >= 4:
        result["manifest_only_refresh"] = False
    if version >= 5:
        result["reviewed_candidate_promotion"] = False
    return result


def envelope(mode="full_artifact", version=3):
    return {
        "schema": "axiom-encode/atomic-source-transaction/v6",
        "transaction": transaction(version),
        "repair_mode": mode,
    }


@pytest.mark.parametrize("version", [2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["full_artifact", "tests_only"])
def test_explicit_mode_preserves_exact_inner_contract(version, mode):
    raw = json.dumps(envelope(mode, version))
    old = json.dumps(transaction(version))
    assert immutable_atomic_source_contract(raw) == split_atomic_source_input(old)
    assert resolve_atomic_repair_mode(raw, DISPATCH) == {
        "mode": mode,
        "tests_only": mode == "tests_only",
    }
    assert resolve_atomic_repair_mode(old, DISPATCH)["tests_only"] is True
    assert _repair_lane_for_atomic_source({"atomic_source_input": old}, raw) == (
        "target",
        ["target"],
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("REPAIR_RUN_ID", ""),
        ("REPAIR_RUN_ID", "67890"),
        ("REPAIR_RUN_ID", "35160240952"),
        ("REPAIR_RUN_ID", "not-a-run"),
        ("QUEUE_ID", "queue"),
        ("DEPENDENT_CITATION", "ca/policy/dependent"),
        ("SECOND_DEPENDENT_CITATION", "ca/policy/dependent"),
        ("REPLACE_LEGACY_RULESPEC_PATH", "legacy.yaml"),
        ("LEGACY_EXACT_DEPENDENT_RULESPEC_PATH", "legacy.yaml"),
        ("SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH", "legacy.yaml"),
        ("EXISTING_SIGNED_IMPORTS_JSON", '["import"]'),
        ("LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON", '["successor"]'),
        ("REPLACE_RULESPEC_PATH", "ca/policies/wrong.yaml"),
    ],
)
def test_explicit_mode_rejects_mixed_or_unauthenticated_dispatch(field, value):
    with pytest.raises(ValueError):
        resolve_atomic_repair_mode(json.dumps(envelope()), {**DISPATCH, field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_bundle", ["ca/policy/extra"]),
        ("canonical_refresh_bundle", [{"citation": "ca/policy/extra"}]),
        ("primary_required_test_cases", []),
        ("require_complete_source_unit", False),
        ("manifest_only_refresh", True),
        ("reviewed_candidate_promotion", True),
    ],
)
def test_explicit_mode_rejects_other_atomic_modes(field, value):
    payload = envelope(version=5)
    payload["transaction"][field] = value
    with pytest.raises(ValueError):
        split_atomic_source_input(json.dumps(payload))


@pytest.mark.parametrize("mutation", ["mode", "unknown", "nested", "list", "duplicate"])
def test_wrapper_is_closed_and_unambiguous(mutation):
    payload = envelope()
    if mutation == "mode":
        payload["repair_mode"] = "auto"
    elif mutation == "unknown":
        payload["ignored"] = True
    elif mutation == "nested":
        payload["transaction"] = envelope()
    elif mutation == "list":
        payload["transaction"] = []
    raw = json.dumps(payload)
    if mutation == "duplicate":
        raw = raw.replace('"repair_mode":', '"repair_mode":"tests_only","repair_mode":')
    with pytest.raises(ValueError):
        split_atomic_source_input(raw)


def test_immutable_comparison_keeps_case_and_schema_differences():
    raw = json.dumps(envelope())
    changed = transaction()
    changed["primary_required_test_cases"][0]["required_output"]["amount"] = 40
    with pytest.raises(ValueError, match="metadata mismatch"):
        _repair_lane_for_atomic_source(
            {"atomic_source_input": json.dumps(changed)}, raw
        )
    # Historical normalization intentionally distinguishes v4's explicit flag.
    with pytest.raises(ValueError, match="metadata mismatch"):
        _repair_lane_for_atomic_source(
            {"atomic_source_input": json.dumps(transaction(4))}, raw
        )


def test_prior_v6_metadata_binds_actual_selection():
    raw = json.dumps(envelope())
    metadata = {"atomic_source_input": raw}
    with pytest.raises(ValueError, match="execution mode"):
        _repair_lane_for_atomic_source(metadata, raw)
    metadata["repair_execution"] = repair_execution_metadata(
        raw, "123", "full_artifact", "false"
    )
    assert _repair_lane_for_atomic_source(metadata, raw) == ("target", ["target"])
    with pytest.raises(ValueError, match="differs"):
        repair_execution_metadata(raw, "123", "tests_only", "true")


def workflow_steps():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    return {
        step["name"]: step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "name" in step
    }


@pytest.mark.parametrize("mode", [None, "tests_only", "full_artifact"])
def test_actual_resolve_shell_selects_mode_before_artifact_access(tmp_path, mode):
    # Execute the actual Resolve prefix, using the real helper and interpreter.
    # The only substitution is the local checkout/interpreter path, not the mode.
    steps = workflow_steps()
    script = steps["Resolve trusted prior-run repair candidate"]["run"].split(
        'if [ "$REPAIR_RUN_ID" = "35160240952" ]; then', 1
    )[0]
    script = script.replace("axiom-encode/.venv/bin/python", sys.executable)
    script = script.replace(
        "axiom-encode/scripts/prepare_signed_backfill.py",
        str(ROOT / "scripts/prepare_signed_backfill.py"),
    )
    raw = json.dumps(transaction() if mode is None else envelope(mode))
    output = tmp_path / "outputs"
    env = {
        **os.environ,
        **DISPATCH,
        "ATOMIC_SOURCE_JSON": raw,
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        ["bash", "-e", "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    selected = dict(line.split("=", 1) for line in output.read_text().splitlines())
    expected = mode or "tests_only"
    selection_metadata = json.loads(selected.pop("execution"))
    assert selection_metadata["mode"] == expected
    assert selected == {
        "mode": expected,
        "tests_only": str(expected == "tests_only").lower(),
    }
    from axiom_encode.harness.evals import (
        EvalWorkspace,
        ValidationRetryCandidate,
        _build_rulespec_eval_prompt,
    )

    source = tmp_path / "source.txt"
    source.write_text("The credit equals income multiplied by 0.1.")
    prompt = _build_rulespec_eval_prompt(
        DISPATCH["CITATION"],
        "repo-augmented",
        EvalWorkspace(tmp_path, source, tmp_path / "manifest.json"),
        [],
        "benefit.yaml",
        "ca:policies/example/benefit",
        True,
        "openai",
        None,
        require_complete_source_unit=True,
        validation_retry_candidate=ValidationRetryCandidate("rules: []", "cases: []"),
        repair_candidate_tests_only=selected["tests_only"] == "true",
    )
    assert ("Complete-source coverage index" in prompt) is (expected == "full_artifact")
    assert ("immutable in this repair" in prompt) is (expected == "tests_only")
    # All downstream checks consume the actual output, never a forced false.
    env.update(REPAIR_MODE=selected["mode"], REPAIR_TESTS_ONLY=selected["tests_only"])
    checked = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_signed_backfill.py"),
            "resolve-atomic-repair-mode",
            raw,
            "--check-selected",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    for name in (
        "Validate atomic source inputs",
        "Verify existing signed imports",
        "Encode, review, validate, and apply",
    ):
        assert "--check-selected" in steps[name]["run"]
        assert (
            steps[name]["env"]["REPAIR_MODE"]
            == "${{ steps.repair_candidate.outputs.mode }}"
        )
    assert (
        repair_execution_metadata(
            raw, DISPATCH["REPAIR_RUN_ID"], selected["mode"], selected["tests_only"]
        )["mode"]
        == expected
    )


@pytest.mark.parametrize(
    "raw,selection",
    [
        ('{"schema":"axiom-encode/atomic-source-transaction/v6"}', ""),
        ("not json", "not json"),
        (
            json.dumps(envelope()),
            json.dumps(
                {
                    "requested_mode": "full_artifact",
                    "mode": "full_artifact",
                    "tests_only": False,
                }
            ),
        ),
    ],
)
def test_actual_early_failure_package_needs_no_installed_encoder(
    tmp_path, raw, selection
):
    step = workflow_steps()["Package failed re-encode diagnostics"]
    env = {
        **os.environ,
        "COPYFILE_DISABLE": "1",
        "RUNNER_TEMP": str(tmp_path),
        "CITATION": DISPATCH["CITATION"],
        "COUNTRY": "ca",
        "CORPUS_REF": "corpus",
        "RULESPEC_REF": "base",
        "RULES_ENGINE_REF": "engine",
        "GITHUB_SHA": "encoder",
        "GITHUB_RUN_ID": "67890",
        "GITHUB_RUN_ATTEMPT": "1",
        "PROVISION_SIGNING_SUPERVISOR_CONCLUSION": "failure",
        "ATOMIC_SOURCE_JSON": raw,
        "REPAIR_EXECUTION_JSON": selection,
        "REPAIR_RUN_ID": "12345",
    }
    # Actual /usr/bin/python3 -I route; no package path or protected installation.
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(
        (tmp_path / "targeted-reencode-failure/metadata.json").read_text()
    )
    assert metadata["atomic_source_input"] == raw
    assert metadata["repair_execution"] == (
        json.loads(selection) if selection.startswith("{") else None
    )
    assert (tmp_path / "targeted-reencode-failure.tar").is_file()


@pytest.mark.parametrize("run_id", ["", "35160240952"])
def test_actual_always_run_or_resolve_rejects_v6_before_artifact_access(
    tmp_path, run_id
):
    name = (
        "Resolve trusted prior-run repair candidate"
        if run_id
        else "Validate atomic source inputs"
    )
    script = workflow_steps()[name]["run"]
    if run_id:
        script = script.split('if [ "$REPAIR_RUN_ID" = "35160240952" ]; then', 1)[0]
    else:
        script = script.split("source_bundle_json=", 1)[0]
    script = script.replace("axiom-encode/.venv/bin/python", sys.executable).replace(
        "axiom-encode/scripts/prepare_signed_backfill.py",
        str(ROOT / "scripts/prepare_signed_backfill.py"),
    )
    env = {
        **os.environ,
        **DISPATCH,
        "REPAIR_RUN_ID": run_id,
        "ATOMIC_SOURCE_JSON": json.dumps(envelope()),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
    }
    result = subprocess.run(
        ["bash", "-e", "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (
        "distinct authenticated" in result.stderr or "signed-success" in result.stderr
    )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_lane", "dependent"),
        ("replace_rulespec_path", None),
        ("existing_signed_imports_json", '["ca:policies/import"]'),
        ("workflow_run_id", "35160240952"),
    ],
)
def test_direct_extractor_rejects_v6_side_modes_before_archive(tmp_path, field, value):
    from argparse import Namespace

    from scripts.extract_repair_candidate import extract_candidate

    arguments = {
        "destination": tmp_path / "output",
        "repair_lane": "target",
        "atomic_source_json": json.dumps(envelope()),
        "workflow_run_id": "12345",
        "citation": DISPATCH["CITATION"],
        "replace_rulespec_path": DISPATCH["REPLACE_RULESPEC_PATH"],
        "existing_signed_imports_json": "[]",
    }
    arguments[field] = value
    # There is deliberately no archive field: rejection precedes archive access.
    with pytest.raises(ValueError):
        extract_candidate(Namespace(**arguments))
