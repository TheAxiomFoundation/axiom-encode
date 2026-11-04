"""Fresh ordinary required cases do not imply canonical predecessor ownership."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.prepare_signed_backfill import (
    parse_canonical_refresh_bundle,
    split_atomic_source_input,
    validate_fresh_primary_test_target,
)

ROOT = Path(__file__).resolve().parents[1]
CITATION = "ca/policy/example/benefit"
TARGET = "ca/policies/example/benefit.yaml"
CASE = {
    "name": "annual",
    "period": {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    "input": {"cost": 100},
    "required_output": {"benefit": 39},
}
STAGES = [
    "Validate atomic source inputs",
    "Verify existing signed imports",
    "Encode, review, validate, and apply",
]


def git(repo, *args):
    return (
        subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
        )
        .decode()
        .strip()
    )


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "rulespec-ca"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.test")
    primary = repo / TARGET
    primary.parent.mkdir(parents=True)
    primary.write_text("rules: []\n")
    primary.with_suffix(".test.yaml").write_text("cases: []\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "baseline")
    return repo, git(repo, "rev-parse", "HEAD")


def validate(repo, head, **kwargs):
    return validate_fresh_primary_test_target(
        repo,
        kwargs.get("citation", CITATION),
        kwargs.get("target", TARGET),
        head,
        kwargs.get("cases", json.dumps([CASE])),
    )


def test_fresh_target_needs_no_predecessor_manifest_but_refresh_still_does(repository):
    repo, head = repository
    result = validate(repo, head)
    assert result["base"] == head
    assert result["required_test_cases"] == [CASE]
    assert len(result["files"]) == 2
    with pytest.raises(ValueError, match="manifest.*tracked"):
        parse_canonical_refresh_bundle(
            repo,
            "[]",
            primary_citation=CITATION,
            primary_rulespec_path=TARGET,
            primary_required_test_cases_json=json.dumps([CASE]),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "dirty",
        "staged",
        "missing",
        "symlink",
        "mode",
        "companion_dirty",
        "companion_symlink",
        "companion_missing",
        "index_only",
        "head_mode",
    ],
)
def test_reject_modified_or_unsafe_target_and_companion(repository, mutation):
    repo, head = repository
    path = repo / TARGET
    if mutation.startswith("companion_"):
        path = path.with_suffix(".test.yaml")
        mutation = mutation.removeprefix("companion_")
    if mutation in {"dirty", "staged", "index_only"}:
        old = path.read_bytes()
        path.write_text("changed\n")
        if mutation != "dirty":
            git(repo, "add", str(path))
        if mutation == "index_only":
            path.write_bytes(old)
    elif mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to("/dev/null")
    elif mutation in {"mode", "head_mode"}:
        path.chmod(0o755)
        if mutation == "head_mode":
            git(repo, "add", str(path))
            git(repo, "commit", "-qm", "executable")
            head = git(repo, "rev-parse", "HEAD")
            path.chmod(0o644)
            git(repo, "add", str(path))
    with pytest.raises(ValueError):
        validate(repo, head)


def test_optional_companion_absence_is_bound_and_untracked_file_rejected(repository):
    repo, head = repository
    path = (repo / TARGET).with_suffix(".test.yaml")
    git(repo, "rm", str(path))
    git(repo, "commit", "-qm", "no companion")
    head = git(repo, "rev-parse", "HEAD")
    assert validate(repo, head)["files"][path.relative_to(repo).as_posix()] is None
    path.write_text("untracked")
    with pytest.raises(ValueError, match="untracked"):
        validate(repo, head)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "ca/policies/other.yaml"},
        {"target": "../escape.yaml"},
        {"citation": "us/policy/example/benefit"},
        {"cases": "[]"},
        {"cases": "[{}]"},
        {"cases": "x" * 65537},
        {"cases": '[{"name":"a","name":"b"}]'},
    ],
)
def test_reject_invalid_path_country_or_contract(repository, kwargs):
    repo, head = repository
    with pytest.raises(ValueError):
        validate(repo, head, **kwargs)


def test_reject_wrong_base(repository):
    repo, _ = repository
    with pytest.raises(ValueError, match="immutable"):
        validate(repo, "a" * 40)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "conflict",
    [
        None,
        "QUEUE_ID",
        "DEPENDENT_CITATION",
        "SECOND_DEPENDENT_CITATION",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "REPAIR_TESTS_ONLY",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
        "source_bundle",
        "canonical_refresh_bundle",
        "require_complete_source_unit",
        "manifest_only_refresh",
        "reviewed_candidate_promotion",
        "REPAIR_RUN_ID",
    ],
)
def test_actual_three_stage_routing_preserves_contract_and_modes(
    repository, tmp_path, stage, conflict
):
    repo, head = repository
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        s for s in workflow["jobs"]["encode"]["steps"] if s.get("name") == stage
    )
    command = step["run"].split(
        "canonical_refresh_primary_required_test_cases_json=", 1
    )[1]
    command = (
        "canonical_refresh_primary_required_test_cases_json="
        + command.split("source_bundle_args=(", 1)[0]
    )
    # Actual helper runs; wrapper records argv and imports this checkout explicitly.
    runner = tmp_path / "helper-python"
    runner.write_text(
        '#!/bin/sh\n[ "$1" != -I ] || shift\nshift\nexec "'
        + os.sys.executable
        + '" "'
        + str(ROOT / "scripts/prepare_signed_backfill.py")
        + '" "$@"\n'
    )
    runner.chmod(0o755)
    command = command.replace("axiom-encode/.venv/bin/python", str(runner))
    payload = split_atomic_source_input(
        json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v5",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [CASE],
                "require_complete_source_unit": True,
                "manifest_only_refresh": False,
                "reviewed_candidate_promotion": False,
            }
        )
    )
    env = {
        **os.environ,
        **{key: "" for key in step["env"]},
        "PYTHONPATH": str(ROOT / "src"),
        "REPLACE_RULESPEC_PATH": TARGET,
        "REPLACE_LEGACY_RULESPEC_PATH": "",
        "CITATION": CITATION,
        "RULESPEC_REF": head,
        "RULESPEC_CHECKOUT": str(repo),
        "primary_required_test_cases_json": json.dumps([CASE]),
        "REPAIR_TESTS_ONLY": "false",
        "workflow_python": str(runner),
        "backfill_helper": str(ROOT / "scripts/prepare_signed_backfill.py"),
        "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
    }
    if conflict in payload:
        payload[conflict] = (
            ["unexpected"]
            if conflict.endswith("bundle")
            else conflict != "require_complete_source_unit"
        )
    elif conflict:
        env[conflict] = '["unexpected"]' if conflict.endswith("JSON") else "true"
    env["atomic_source_payload"] = json.dumps(payload)
    command += '\nprintf "%s\\n%s\\n" "$canonical_refresh_primary_required_test_cases_json" "$primary_required_test_cases_json"\n'
    result = subprocess.run(
        ["bash", "-eu", "-c", command], env=env, text=True, capture_output=True
    )
    prior_modes = {
        "canonical_refresh_bundle",
        "require_complete_source_unit",
        "manifest_only_refresh",
        "reviewed_candidate_promotion",
        "REPAIR_RUN_ID",
    }
    if conflict and conflict not in prior_modes:
        assert result.returncode != 0
        assert "cannot mix transaction modes" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        refresh_cases, signing_cases = map(json.loads, result.stdout.splitlines())
        assert signing_cases == [CASE]
        if conflict in prior_modes - {"REPAIR_RUN_ID"}:
            assert refresh_cases == [
                CASE
            ]  # Existing canonical admission remains mandatory.
        elif conflict is None or conflict == "REPAIR_RUN_ID":
            assert refresh_cases == []


def test_actual_protected_git_wrapper_accepts_immutable_file_checks(
    repository, tmp_path, monkeypatch
):
    import shutil

    from scripts.provision_verification_supervisor import _install_trusted_git_wrapper

    repo, head = repository
    real_git = Path(shutil.which("git")).resolve()
    bindir = tmp_path / "protected-bin"
    bindir.mkdir()
    _install_trusted_git_wrapper(bindir, Path(os.sys.executable).resolve(), real_git)
    monkeypatch.setenv("PATH", str(bindir))
    assert validate(repo, head)["required_test_cases"] == [CASE]
    (repo / TARGET).write_text("dirty\n")
    with pytest.raises(ValueError, match="differs from HEAD"):
        validate(repo, head)
