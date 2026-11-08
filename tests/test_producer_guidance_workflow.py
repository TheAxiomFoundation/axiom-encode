"""Execute the workflow's actual shell argument branches without invoking tools."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/targeted-signed-reencode.yml"


def scripts():
    document = yaml.safe_load(WORKFLOW.read_text())
    return document, [step["run"] for job in document["jobs"].values() for step in job.get("steps", []) if isinstance(step.get("run"), str)]


def generation_branch():
    _, runs = scripts()
    run = next(run for run in runs if "run_signed_encode()" in run)
    start = run.index('case "$review_finding" in')
    return run[start:run.index("esac", start) + 4]


def execute_branch(tmp_path, finding, *, lane="target"):
    env = {**os.environ, "TEST_FINDING": finding, "RUNNER_TEMP": str(tmp_path),
           "GITHUB_WORKSPACE": str(tmp_path), "CITATION": "ca/policy/example", "TEST_LANE": lane}
    command = 'set -euo pipefail\nargs=()\nreview_finding="$TEST_FINDING"\ncitation="$CITATION"\noutput_lane="$TEST_LANE"\n'
    command += generation_branch() + '\nprintf "%s\\0" "${args[@]}"\n'
    return subprocess.run(["bash", "-c", command], env=env, cwd=tmp_path, capture_output=True)


def test_no_extra_dispatch_input():
    document, _ = scripts()
    assert len(document.get("on", document.get(True))["workflow_dispatch"]["inputs"]) == 25


@pytest.mark.parametrize("finding", ["ordinary", "line1\nline2\n", '{"json":"inline"}', ' $(touch SHOULD_NOT_EXIST) '])
def test_actual_inline_branch_preserves_bytes(tmp_path, finding):
    result = execute_branch(tmp_path, finding)
    assert result.returncode == 0, result.stderr
    args = result.stdout.split(b"\0")[:-1]
    assert args[0] == b"--review-findings" and len(args) == 2
    assert Path(os.fsdecode(args[1])).read_bytes() == (finding + "\n").encode()
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_actual_pointer_branch_preserves_suffix_without_inline_file(tmp_path):
    descriptor = '{\n "schema": "closed-descriptor"\n}\n'
    result = execute_branch(tmp_path, "axiom-producer-review-guidance-ref:v1:" + descriptor)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split(b"\0")[:-1] == [
        b"--producer-review-guidance-ref", descriptor.encode(),
        b"--producer-review-guidance-checkout", os.fsencode(tmp_path / "axiom-encode"),
    ]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("finding,lane", [
    ("axiom-producer-review-guidance-ref:v2:{}", "target"),
    ("axiom-producer-review-guidance-ref:v1:{}", "dependent"),
])
def test_actual_branch_refuses_wrong_version_or_lane(tmp_path, finding, lane):
    result = execute_branch(tmp_path, finding, lane=lane)
    assert result.returncode != 0
    assert result.stdout == b""


def test_actual_packaging_context_read_rejects_self_consistent_stale_bytes():
    import hashlib
    import textwrap

    _, runs = scripts()
    run = next(run for run in runs if "initial_manifest_digests = {}" in run)
    start = run.index("    context_digest = hashlib.sha256(context_bytes).hexdigest()")
    end = run.index("    context_payload = json.loads(context_bytes)", start)
    snippet = textwrap.dedent(run[start:end])
    old = b'{"old":"self-consistent"}'
    old_digest = hashlib.sha256(old).hexdigest()
    scope = {"hashlib": hashlib, "context_bytes": old, "primary_guidance_descriptor": "reference",
             "guidance_guard_payload": {"producer_review_guidance": {"context_manifest_sha256": "0" * 64}},
             "evidence_manifest": {"context_manifest_sha256": old_digest}}
    with pytest.raises(SystemExit, match="packaged context bytes"):
        exec(compile(snippet, "workflow-context-recheck", "exec"), scope)


def test_actual_packaging_initial_snapshot_is_guard_bound():
    import textwrap

    _, runs = scripts()
    run = next(run for run in runs if "initial_manifest_digests = {}" in run)
    end = run.index('    manifest_path = Path(os.environ["RULESPEC_CHECKOUT"]) / relative_path')
    start = run.rindex("    if primary_guidance_descriptor is not None:", 0, end)
    snippet = textwrap.dedent(run[start:end])
    scope = {"primary_guidance_descriptor": "reference", "relative_path": "manifest.json",
             "initial_manifest_digests": {"manifest.json": "a" * 64},
             "guidance_guard_payload": {"producer_review_guidance": {"manifest_sha256": "b" * 64}}}
    with pytest.raises(SystemExit, match="initial apply manifest snapshot"):
        exec(compile(snippet, "workflow-manifest-recheck", "exec"), scope)
