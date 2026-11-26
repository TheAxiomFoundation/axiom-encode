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
import sys
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


def run_promotion_fragment(payload: dict, switch: str | None):
    script = fragment(
        step("Encode, review, validate, and apply")["run"],
        "promotion_args=(",
        'queue_operation command "${promotion_args[@]}"',
    )
    environment = {
        **os.environ,
        "RULESPEC_CHECKOUT": "/work/rulespec-us",
        "REPLACE_RULESPEC_PATH": "us/statutes/7/2015/a.yaml",
        "CITATION": "us/statute/7/2015/a",
        "RULESPEC_REF": "c" * 40,
        "GITHUB_WORKSPACE": "/work",
        "GITHUB_RUN_ID": "777",
        "RUNNER_TEMP": "/runner/temp",
        "atomic_source_payload": json.dumps(payload),
    }
    environment.pop("REVIEW_RECORD_REQUIRED", None)
    if switch is not None:
        environment["REVIEW_RECORD_REQUIRED"] = switch
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nqueue_operation() { printf "%s\\n" "$@"; }\n' + script,
        ],
        capture_output=True,
        text=True,
        env=environment,
    )


def promotion_command(payload: dict, switch: str | None = "false") -> list[str]:
    completed = run_promotion_fragment(payload, switch)
    assert completed.returncode == 0, completed.stderr
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
    assert promotion_command(payload, "true") == [
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


SWITCH_STEP = "Decide the review record switch"


def test_one_parsed_value_gates_the_reader_step_and_the_promotion():
    names = [item.get("name") for item in steps()]
    switch = step(SWITCH_STEP)
    reader = step(READ_STEP)
    assert (
        names.index("Preflight live reviewed PR promotion")
        < names.index(SWITCH_STEP)
        < names.index(READ_STEP)
        < names.index("Verify immutable checkout identities")
        < names.index("Provision protected signing supervisor")
    )
    # The switch step runs in every run and parses the wrapper once.
    assert "if" not in switch and switch["id"] == "review_record_switch"
    assert set(switch["env"]) == {"DISPATCH_JSON"}
    assert "read_promotion_review_records.py switch" in switch["run"]
    # The reader step is gated by that value alone, not by a text search.
    assert (
        reader["if"] == "${{ steps.review_record_switch.outputs.required == 'true' }}"
    )
    assert "contains(" not in WORKFLOW.read_text().split(READ_STEP)[1].split("run:")[0]
    # No token, no key, no secret: the read is anonymous.
    assert set(reader["env"]) == {"DISPATCH_JSON", "RULESPEC_CHECKOUT"}
    assert "secrets." not in json.dumps(reader) + json.dumps(switch)
    assert "read_promotion_review_records.py read" in reader["run"]
    assert "continue-on-error" not in reader and "continue-on-error" not in switch
    # The promotion and the publication read the same output.
    output = "${{ steps.review_record_switch.outputs.required }}"
    for name in (
        "Encode, review, validate, and apply",
        "Prepare reviewed lane pull request branch",
    ):
        assert step(name)["env"]["REVIEW_RECORD_REQUIRED"] == output


def test_no_other_step_changed_its_condition_for_the_switch():
    conditional = [
        item.get("name")
        for item in steps()
        if "review_record" in str(item.get("if", ""))
    ]
    assert conditional == [READ_STEP]


@needs_shell
@pytest.mark.parametrize(
    ("payload", "switch"),
    [
        # The wrapper parses ON and the switch step said no, or did not run.
        ({"promotion_review_record_required": True}, "false"),
        ({"promotion_review_record_required": True}, None),
        ({"promotion_review_record_required": True}, ""),
        # The switch step said yes and the wrapper parses OFF.
        ({}, "true"),
    ],
)
def test_a_switch_mismatch_refuses_and_queues_no_command(payload, switch):
    completed = run_promotion_fragment(payload, switch)
    # The step stops at the guard; no supervised command is queued.
    assert completed.returncode == 1
    assert completed.stdout == ""


SPACED_WRAPPERS = [
    wrapper(require_review_record=True),
    wrapper(require_review_record=True).replace(
        '"require_review_record": true', '"require_review_record"  :\n\ttrue'
    ),
    wrapper(require_review_record=True).replace(
        '"require_review_record"', '"require_review\\u005frecord"'
    ),
]


@needs_shell
@pytest.mark.parametrize("source", SPACED_WRAPPERS)
def test_the_switch_does_not_depend_on_the_spelling_of_the_key(source, tmp_path):
    """Spacing or an escape in the key gives the same value in both places."""

    assert json.loads(source)["require_review_record"] is True
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/read_promotion_review_records.py"),
            "switch",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DISPATCH_JSON": json.dumps({"source_bundle_json": source}),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "required=true\n"
    assert summary.read_text() == "Review record required: yes\n"
    payload = helper.split_atomic_source_input(source)
    assert "--require-review-record" in promotion_command(payload, "true")


@pytest.mark.parametrize(
    "source",
    ["[]", "", "not json", wrapper(), json.dumps({"schema": "other"}), "{}"],
)
def test_every_other_run_states_that_the_switch_is_off(source, tmp_path):
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/read_promotion_review_records.py"),
            "switch",
        ],
        check=True,
        env={
            **os.environ,
            "DISPATCH_JSON": json.dumps({"source_bundle_json": source}),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )
    assert output.read_text() == "required=false\n"
    assert summary.read_text() == "Review record required: no\n"


# --- The lane pull request body -------------------------------------------------------


def lane_body(tmp_path: Path, summary: str | None, switch: str | None = None):
    """Run the two fragments as the workflow does: prepare, then publish.

    The first fragment is in the token-free step that prepares the branch;
    the second is in the step that holds the repository token.
    """

    prepare = fragment(
        step("Prepare reviewed lane pull request branch")["run"],
        'lane_review_record="$RUNNER_TEMP/lane-review-record.md"',
        "printf '%s\\n' 'Review record required: no' > \"$lane_review_record\"",
    )
    publish = fragment(
        step("Push lane branch and open draft pull request")["run"],
        'review_record_section="$(jq -rRs',
        'body="$(printf \'%s\\n\' "$body" \'\' "$review_record_section")"',
    )
    if summary is not None:
        directory = tmp_path / "promotion-review-records"
        directory.mkdir()
        (directory / "candidate-review-records.md").write_text(summary)
    environment = {
        **{k: v for k, v in os.environ.items() if k != "REVIEW_RECORD_REQUIRED"},
        "RUNNER_TEMP": str(tmp_path),
    }
    prepared = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n" + prepare + "\nfi\n"],
        capture_output=True,
        text=True,
        env={
            **environment,
            **({} if switch is None else {"REVIEW_RECORD_REQUIRED": switch}),
        },
    )
    if prepared.returncode != 0:
        return None
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nbody="existing body"\n'
            + publish
            + '\nprintf "%s" "$body"\n',
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout if completed.returncode == 0 else None


@needs_shell
def test_the_token_step_stops_when_the_prepared_text_is_missing(tmp_path):
    publish = fragment(
        step("Push lane branch and open draft pull request")["run"],
        'review_record_section="$(jq -rRs',
        'body="$(printf \'%s\\n\' "$body" \'\' "$review_record_section")"',
    )
    for content in (None, ""):
        if content is not None:
            (tmp_path / "lane-review-record.md").write_text(content)
        completed = subprocess.run(
            ["bash", "-c", 'set -euo pipefail\nbody="x"\n' + publish + "\necho done\n"],
            capture_output=True,
            text=True,
            env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        )
        assert completed.returncode != 0 and "done" not in completed.stdout


def test_the_token_step_takes_the_review_record_text_as_a_file_only():
    token_step = step("Push lane branch and open draft pull request")
    assert "REVIEW_RECORD_REQUIRED" not in token_step["env"]
    assert "promotion-review-records" not in token_step["run"]
    assert "cat " not in token_step["run"]
    prepare = step("Prepare reviewed lane pull request branch")
    assert "secrets." not in json.dumps(prepare.get("env", {}))
    assert prepare["env"]["REVIEW_RECORD_REQUIRED"] == (
        "${{ steps.review_record_switch.outputs.required }}"
    )


def test_the_new_steps_have_step_checks():
    from axiom_encode import infra_checks

    for name in (SWITCH_STEP, READ_STEP):
        check_id = infra_checks.step_check_id(name)
        assert any(
            definition["id"] == check_id
            for definition in infra_checks.STEP_CHECK_DEFINITIONS
        )


@needs_shell
@pytest.mark.parametrize("switch", [None, "", "false"])
def test_switch_off_the_lane_pull_request_body_says_no(tmp_path, switch):
    assert lane_body(tmp_path, None, switch) == (
        "existing body\n\nReview record required: no"
    )


@needs_shell
def test_switch_on_the_lane_pull_request_body_carries_the_record(tmp_path):
    summary = "## Review record\n\nReview record required: yes\n$(not run) `x`\n"
    assert lane_body(tmp_path, summary, "true") == (
        "existing body\n\n" + summary.rstrip("\n")
    )


@needs_shell
def test_switch_on_without_the_section_stops_the_publication(tmp_path):
    assert lane_body(tmp_path, None, "true") is None


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
