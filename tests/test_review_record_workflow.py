"""The signing workflow and the review record switch.

The switch is the optional wrapper key ``"require_review_record": true`` of a
promotion dispatch. These tests run the workflow's own embedded fragments and
prove that without the key the supervised promotion command, the lane pull
request body and the run artifact are what they were, and that with the key
the command gets ``--require-review-record`` and the record is published.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import prepare_signed_backfill as helper
from axiom_encode import reviewed_pr_admission as admission

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"
READ_STEP = "Read candidate review records"

needs_shell = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="jq and bash are required to execute the workflow fragments",
)


def steps() -> list[dict]:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["encode"]["steps"]


def step(name: str) -> dict:
    return next(item for item in steps() if item.get("name") == name)


def fragment(run: str, start: str, end: str) -> str:
    lines = run.splitlines()
    first = next(index for index, line in enumerate(lines) if start in line)
    last = next(
        index for index, line in enumerate(lines) if index > first and end in line
    )
    return "\n".join(lines[first : last + 1])


def wrapper(**extra) -> str:
    return json.dumps(
        {
            "schema": "axiom-encode/reviewed-pr-promotion/v1",
            "rulespec_pr": 42,
            "transaction": {
                "schema": "axiom-encode/atomic-source-transaction/v5",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
                "require_complete_source_unit": True,
                "manifest_only_refresh": False,
                "reviewed_candidate_promotion": True,
            },
            **extra,
        }
    )


# --- The wrapper key --------------------------------------------------------------


def test_without_the_key_the_wrapper_is_read_as_before():
    plain = helper.split_atomic_source_input(wrapper())
    assert "promotion_review_record_required" not in plain
    assert plain["rulespec_pr"] == 42
    assert plain["reviewed_candidate_promotion"] is True
    required = helper.split_atomic_source_input(wrapper(require_review_record=True))
    assert required == {**plain, "promotion_review_record_required": True}


@pytest.mark.parametrize("value", [False, "true", 1, None, [], {}])
def test_only_exactly_true_is_accepted_for_the_key(value):
    with pytest.raises(ValueError, match="must be exactly true"):
        helper.split_atomic_source_input(wrapper(require_review_record=value))


def test_the_key_combines_with_a_unit_list(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)
    units = ["us/statute/7/2015/a", "us/statute/7/2015/b"]
    payload = helper.split_atomic_source_input(
        wrapper(units=units, require_review_record=True)
    )
    assert payload["promotion_units"] == units
    assert payload["promotion_review_record_required"] is True


def test_another_unknown_key_is_still_refused():
    with pytest.raises(ValueError, match="exactly transaction and rulespec_pr"):
        helper.split_atomic_source_input(wrapper(require_review=True))


def test_the_key_adds_no_dispatch_input():
    inputs = yaml.safe_load(WORKFLOW.read_text())[True]["workflow_dispatch"]["inputs"]
    # GitHub refuses a workflow with more than 25 dispatch inputs, and this
    # workflow has 25. The switch is therefore a key of the source wrapper.
    assert len(inputs) == 25
    assert not any("review_record" in name for name in inputs)


# --- The supervised promotion command ----------------------------------------------

BASE_COMMAND = [
    "command",
    "/opt/axiom-verification/axiom-encode",
    "promote-reviewed-candidate",
    "--repo",
    "/work/rulespec-us",
    "--rulespec-path",
    "us/statutes/7/2015/a.yaml",
    "--citation",
    "us/statute/7/2015/a",
    "--reviewed-rulespec-ref",
    "c" * 40,
    "--axiom-rules-engine-path",
    "/work/axiom-rules-engine",
    "--corpus-path",
    "/work/axiom-corpus",
    "--run-id",
    "777",
]


def promotion_command(payload: dict) -> list[str]:
    script = fragment(
        step("Encode, review, validate, and apply")["run"],
        "promotion_args=(",
        'queue_operation command "${promotion_args[@]}"',
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nqueue_operation() { printf "%s\\n" "$@"; }\n' + script,
        ],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "RULESPEC_CHECKOUT": "/work/rulespec-us",
            "REPLACE_RULESPEC_PATH": "us/statutes/7/2015/a.yaml",
            "CITATION": "us/statute/7/2015/a",
            "RULESPEC_REF": "c" * 40,
            "GITHUB_WORKSPACE": "/work",
            "GITHUB_RUN_ID": "777",
            "RUNNER_TEMP": "/runner/temp",
            "atomic_source_payload": json.dumps(payload),
        },
    )
    return completed.stdout.splitlines()


@needs_shell
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"promotion_units": ["us/statute/7/2015/a"]},
        {"promotion_review_record_required": False},
        {"promotion_review_record_required": "true"},
        {"promotion_review_record_required": None},
    ],
)
def test_switch_off_the_supervised_command_is_what_it_was(payload):
    assert promotion_command(payload) == BASE_COMMAND


@needs_shell
def test_switch_off_the_real_wrapper_gives_the_command_it_gave_before():
    assert (
        promotion_command(helper.split_atomic_source_input(wrapper())) == BASE_COMMAND
    )


@needs_shell
def test_switch_on_the_command_requires_the_record_file():
    payload = helper.split_atomic_source_input(wrapper(require_review_record=True))
    assert promotion_command(payload) == [
        *BASE_COMMAND,
        "--require-review-record",
        "/runner/temp/promotion-review-records/candidate-review-records.json",
    ]


def test_the_workflow_passes_the_flag_in_one_place_only():
    text = WORKFLOW.read_text()
    assert text.count("--require-review-record") == 1
    # The model reviewer of the promotion stays off in the hosted workflow.
    assert "--require-review " not in text and "--require-review\n" not in text


# --- The reader step ----------------------------------------------------------------


def test_the_reader_step_runs_only_for_a_promotion_that_names_the_key():
    names = [item.get("name") for item in steps()]
    reader = step(READ_STEP)
    assert (
        names.index("Preflight live reviewed PR promotion")
        < names.index(READ_STEP)
        < names.index("Verify immutable checkout identities")
        < names.index("Provision protected signing supervisor")
    )
    assert reader["if"] == (
        "${{ steps.reviewed_pr_admission.outputs.enabled == 'true' && "
        "contains(inputs.source_bundle_json, '\"require_review_record\"') }}"
    )
    # No token, no key, no secret: the read is anonymous.
    assert set(reader["env"]) == {"DISPATCH_JSON", "RULESPEC_CHECKOUT"}
    assert "secrets." not in json.dumps(reader)
    assert "scripts/read_promotion_review_records.py" in reader["run"]
    assert "continue-on-error" not in reader


def test_no_other_step_changed_its_condition_for_the_switch():
    conditional = [
        item.get("name")
        for item in steps()
        if "review_record" in str(item.get("if", ""))
    ]
    assert conditional == [READ_STEP]


# --- The lane pull request body -------------------------------------------------------


def lane_body(tmp_path: Path, summary: str | None) -> str:
    script = fragment(
        step("Push lane branch and open draft pull request")["run"],
        'review_records="$RUNNER_TEMP/promotion-review-records/',
        'body="$(printf \'%s\\n\' "$body" \'\' "$(cat "$review_records")")"',
    )
    if summary is not None:
        directory = tmp_path / "promotion-review-records"
        directory.mkdir()
        (directory / "candidate-review-records.md").write_text(summary)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nbody="existing body"\n'
            + script
            + '\nfi\nprintf "%s" "$body"\n',
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
    )
    return completed.stdout


@needs_shell
def test_switch_off_the_lane_pull_request_body_is_unchanged(tmp_path):
    assert lane_body(tmp_path, None) == "existing body"


@needs_shell
def test_switch_on_the_lane_pull_request_body_carries_the_record(tmp_path):
    summary = "## Review record\n\nstatus `passed`, attempt 2, $(not run) `x`\n"
    assert lane_body(tmp_path, summary) == "existing body\n\n" + summary.rstrip("\n")


# --- The run artifact -----------------------------------------------------------------


@needs_shell
@pytest.mark.parametrize("required", [False, True])
def test_the_artifact_holds_the_read_document_only_when_required(tmp_path, required):
    script = fragment(
        step("Finalize signed re-encode artifact checksums")["run"],
        'if [ -d "$RUNNER_TEMP/promotion-review-records" ]; then',
        'cp "$RUNNER_TEMP/promotion-review-records/"* "$artifact/"',
    )
    artifact = tmp_path / "targeted-reencode"
    artifact.mkdir()
    (artifact / "metadata.json").write_text("{}")
    if required:
        directory = tmp_path / "promotion-review-records"
        directory.mkdir()
        for name in (
            "candidate-review-records.json",
            "candidate-review-records.md",
            "candidate-pull-request-body.md",
        ):
            (directory / name).write_text(name)
    subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + script + "\nfi\n"],
        check=True,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "artifact": str(artifact)},
    )
    names = sorted(path.name for path in artifact.iterdir())
    assert names == (
        [
            "candidate-pull-request-body.md",
            "candidate-review-records.json",
            "candidate-review-records.md",
            "metadata.json",
        ]
        if required
        else ["metadata.json"]
    )
