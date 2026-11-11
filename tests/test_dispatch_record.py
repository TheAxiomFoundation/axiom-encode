"""The dispatch record of the targeted signed re-encode workflow.

One record per dispatch says how the run was triggered and which publication
it asked for. It is telemetry: the job that writes it has no secret, cannot
fail or delay a run, and copies only allowlisted values.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "dispatch_record.py"
WORKFLOW = REPO / ".github" / "workflows" / "targeted-signed-reencode.yml"
FINALIZER = REPO / ".github" / "workflows" / "finalize-signed-snap-queue.yml"

_spec = importlib.util.spec_from_file_location("_dispatch_record", SCRIPT)
dispatch_record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dispatch_record)

SHA = "a" * 40
SHA256 = "b" * 64
ENVIRON = {
    "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
    "GITHUB_RUN_ID": "36174590343",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_SHA": SHA,
    "GITHUB_ACTOR": "github-actions[bot]",
    "GITHUB_TRIGGERING_ACTOR": "PavelMakarchuk",
}


def _inputs(**overrides):
    inputs = {
        "citation": "us/statute/42/1396a/xx",
        "country": "us",
        "rulespec_ref": SHA,
        "pr_base_branch": "main",
        "corpus_ref": SHA,
        "rules_engine_ref": SHA,
        "review_finding": "",
        "repair_run_id": "",
        "source_bundle_json": "[]",
        "existing_signed_imports_json": "[]",
        "legacy_retained_successor_rulespec_paths_json": "[]",
        "open_pr": False,
        "queue_id": "",
        "queue_item_id": "",
        "queue_manifest_sha256": "",
        "queue_item_generation_sha256": "",
        "queue_dispatcher_run_id": "",
    }
    inputs.update(overrides)
    return inputs


def _record(**overrides):
    return dispatch_record.build_record(
        _inputs(**overrides), ENVIRON, recorded_at="2026-11-11T00:00:00Z"
    )


def test_adhoc_dispatch_record():
    record = _record()
    assert record["schema"] == "axiom-encode/dispatch-record/v1"
    assert record["trigger_kind"] == "adhoc"
    assert (record["is_queue"], record["is_repair"], record["is_promotion"]) == (
        False,
        False,
        False,
    )
    assert record["open_pr"] is False
    assert record["pr_base_branch"] == "main"
    assert record["repair_run_id"] is None
    assert record["queue_id"] is None
    assert record["citation"] == "us/statute/42/1396a/xx"
    assert record["run_id"] == "36174590343"
    assert record["head_sha"] == SHA
    assert record["actor"] == "github-actions[bot]"
    assert record["triggering_actor"] == "PavelMakarchuk"
    assert record["invalid_inputs"] == []
    assert not any(record["inputs_present"].values())
    assert dispatch_record.record_tag(record) == (
        "kind=adhoc open_pr=false pr_base_branch=main repair_run_id=none"
    )


def test_queue_dispatch_record():
    record = _record(
        queue_id="snap-or-ut-pilot",
        queue_item_id="ut-0001",
        queue_manifest_sha256=SHA256,
        queue_item_generation_sha256=SHA256,
        queue_dispatcher_run_id="123",
        open_pr=True,
    )
    assert record["trigger_kind"] == "queue"
    assert record["is_queue"] is True
    assert record["queue_id"] == "snap-or-ut-pilot"
    assert record["queue_item_id"] == "ut-0001"
    assert record["queue_item_generation_sha256"] == SHA256
    assert record["queue_dispatcher_run_id"] == "123"
    assert record["open_pr"] is True


def test_repair_outranks_queue_and_keeps_both_flags():
    record = _record(queue_id="q", queue_item_id="i", repair_run_id="35160240952")
    assert record["trigger_kind"] == "repair"
    assert record["is_queue"] is True and record["is_repair"] is True
    assert record["repair_run_id"] == "35160240952"
    assert "repair_run_id=35160240952" in dispatch_record.record_tag(record)


@pytest.mark.parametrize(
    ("bundle", "live_reviewed_pr", "promoted"),
    [
        (
            {
                "schema": "axiom-encode/reviewed-pr-promotion/v1",
                "transaction": {"reviewed_candidate_promotion": True},
            },
            True,
            True,
        ),
        ({"reviewed_candidate_promotion": True, "sources": []}, False, True),
    ],
)
def test_promotion_dispatch_record(bundle, live_reviewed_pr, promoted):
    record = _record(source_bundle_json=json.dumps(bundle), repair_run_id="7")
    assert record["trigger_kind"] == "promotion"
    assert record["is_promotion"] is True
    assert record["live_reviewed_pr"] is live_reviewed_pr
    assert record["reviewed_candidate_promotion"] is promoted
    assert record["is_repair"] is True
    assert record["inputs_present"]["source_bundle_json"] is True


@pytest.mark.parametrize(
    "bundle",
    [
        ["us/statute/7/2014/e"],
        {"reviewed_candidate_promotion": False},
        {"reviewed_candidate_promotion": "true"},
        {"schema": "axiom-encode/other/v1", "transaction": {}},
    ],
)
def test_other_source_bundles_are_not_promotions(bundle):
    record = _record(source_bundle_json=json.dumps(bundle))
    assert record["is_promotion"] is False
    assert record["trigger_kind"] == "adhoc"


@pytest.mark.parametrize(
    "bundle", ["{not json", "[" * 100_000, "x" * (dispatch_record.MAX_BUNDLE_BYTES + 1)]
)
def test_unreadable_source_bundle_is_unknown_not_an_error(bundle):
    record = _record(source_bundle_json=bundle, repair_run_id="9")
    assert record["is_promotion"] is None
    assert record["trigger_kind"] == "repair"
    assert "source_bundle_json" in record["invalid_inputs"]


def test_free_text_inputs_are_recorded_as_presence_only():
    secret = "ghp_THIS-MUST-NOT-LEAK finding text"
    free_text = {name: secret for name in dispatch_record.PRESENCE_INPUTS}
    record = _record(**free_text)
    text = json.dumps(record)
    assert "MUST-NOT-LEAK" not in text
    assert all(record["inputs_present"].values())
    assert "MUST-NOT-LEAK" not in dispatch_record.record_tag(record)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("pr_base_branch", "main; rm -rf /"),
        ("pr_base_branch", "feature\nkind=queue"),
        ("repair_run_id", "12 OR 1=1"),
        ("repair_run_id", "0"),
        ("queue_id", "$(id)"),
        ("citation", "us/statute/7 `x`"),
        ("rulespec_ref", "main"),
        ("queue_manifest_sha256", SHA),
        ("country", "US"),
    ],
)
def test_values_outside_the_allowlist_become_null(name, value):
    record = _record(**{name: value})
    assert record[name] is None
    assert name in record["invalid_inputs"]
    if not value.isalnum():
        assert value not in json.dumps(record)
    tag = dispatch_record.record_tag(record)
    assert re.fullmatch(r"[A-Za-z0-9._/= -]+", tag)
    assert len(tag.split(" ")) == 4


def test_malformed_repair_run_id_still_marks_the_repair_path():
    record = _record(repair_run_id="not-a-run")
    assert record["is_repair"] is True
    assert record["trigger_kind"] == "repair"
    assert record["repair_run_id"] is None
    assert "repair_run_id=none" in dispatch_record.record_tag(record)


def test_every_dispatch_input_is_recorded_by_value_or_by_presence():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    inputs = set(workflow[True]["workflow_dispatch"]["inputs"])
    by_value = set(dispatch_record.VALUE_INPUTS) | {"open_pr"}
    by_presence = set(dispatch_record.PRESENCE_INPUTS)
    assert by_value & by_presence == set()
    assert by_value | by_presence == inputs


@pytest.mark.parametrize("inputs", [None, [], "text", 7])
def test_malformed_inputs_never_raise(inputs):
    record = dispatch_record.build_record(inputs, {}, recorded_at="t")
    assert record["trigger_kind"] == "adhoc"
    assert record["invalid_inputs"] == ["inputs"]
    assert record["run_id"] is None


@pytest.mark.parametrize("dispatch_json", ["{not json", "", json.dumps(_inputs())])
def test_script_always_exits_zero_and_writes_the_record(tmp_path, dispatch_json):
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")
    }
    env.update(
        ENVIRON,
        DISPATCH_JSON=dispatch_json,
        GITHUB_OUTPUT=str(output),
        GITHUB_STEP_SUMMARY=str(summary),
    )
    completed = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(tmp_path / "record")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    record = json.loads((tmp_path / "record" / "dispatch-record.json").read_text())
    assert record["schema"] == "axiom-encode/dispatch-record/v1"
    assert record["run_id"] == "36174590343"
    assert completed.stdout.startswith("dispatch_record {")
    assert json.loads(completed.stdout.removeprefix("dispatch_record ")) == record
    tag = output.read_text().removeprefix("tag=").strip()
    assert tag == dispatch_record.record_tag(record)
    assert tag in summary.read_text()


def test_script_exits_zero_when_it_cannot_write(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    completed = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), str(blocker / "record")],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "DISPATCH_JSON": "{}"},
    )
    assert completed.returncode == 0


# --- workflow contract -----------------------------------------------------


def _job():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow, workflow["jobs"]["dispatch_record"]


def test_dispatch_record_job_is_unprotected_and_fail_open():
    workflow, job = _job()
    assert "environment" not in job
    assert "needs" not in job
    assert job["continue-on-error"] is True
    assert job["permissions"] == {"contents": "read"}
    assert job["if"] == "${{ github.event_name == 'workflow_dispatch' }}"
    assert all(step.get("continue-on-error") is True for step in job["steps"])
    text = json.dumps(job)
    assert "secrets." not in text
    assert "github.token" not in text
    # Nothing waits for the record: no job needs it or reads its result.
    for name, other in workflow["jobs"].items():
        if name == "dispatch_record":
            continue
        needs = other.get("needs", [])
        assert "dispatch_record" not in ([needs] if isinstance(needs, str) else needs)
        assert "dispatch_record" not in json.dumps(other)


def test_dispatch_record_job_passes_inputs_as_data_only():
    _, job = _job()
    write = next(step for step in job["steps"] if step.get("id") == "dispatch_record")
    assert write["env"] == {"DISPATCH_JSON": "${{ toJSON(inputs) }}"}
    # No expression is interpolated into the shell; the script reads the env.
    assert "${{" not in write["run"]
    assert write["run"].strip() == (
        'python3 -I "$GITHUB_WORKSPACE/axiom-encode/scripts/dispatch_record.py" '
        '"$RUNNER_TEMP/dispatch-record" || true'
    )
    checkout = job["steps"][0]
    assert checkout["with"]["persist-credentials"] is False


def test_dispatch_record_job_names_stay_clear_of_existing_consumers():
    workflow, job = _job()
    # The dashboard collector finds the encode job by /re-encode/ and the
    # budget job by /budget/ in the job name.
    assert not re.search(r"re-encode|budget", job["name"], re.I)
    # The queue verifier selects the encode job by its exact name.
    assert workflow["jobs"]["encode"]["name"] == (
        "Queue protected signed RuleSpec re-encode"
    )
    upload = job["steps"][-1]
    assert upload["name"] == (
        "Upload dispatch record: ${{ steps.dispatch_record.outputs.tag }}"
    )
    artifact = upload["with"]["name"]
    assert artifact == "dispatch-record-${{ github.run_id }}-${{ github.run_attempt }}"
    # The queue finalizer requires exactly one artifact with this prefix.
    assert "targeted-reencode-" in FINALIZER.read_text(encoding="utf-8")
    assert not artifact.startswith("targeted-reencode")
    assert upload["with"]["if-no-files-found"] == "warn"
    encode_uploads = [
        step["uses"]
        for step in workflow["jobs"]["encode"]["steps"]
        if "upload-artifact" in step.get("uses", "")
    ]
    assert upload["uses"] in encode_uploads
    # Step names of the protected jobs are check sites; keep this job's apart.
    protected = {
        step["name"]
        for name in ("attempt_budget", "encode")
        for step in workflow["jobs"][name]["steps"]
    }
    assert protected.isdisjoint(step["name"] for step in job["steps"])
