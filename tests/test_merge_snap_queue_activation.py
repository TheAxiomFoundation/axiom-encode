"""Run the SNAP queue activation merge's shell against a fake ``gh``.

``test_signing_supervisor.py`` pins the workflow's text. These tests execute
the merge-queue wait and the branch cleanup, so a branch name with URL
characters or a queue entry whose CI never starts behaves as intended.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/merge-snap-queue-activation.yml"
HEAD_SHA = "a" * 40
QUEUE_SHA = "b" * 40

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="needs bash and jq",
)

# The fake gh answers from a JSON scenario file and logs every call. Each key
# of "answers" is a substring of the joined arguments; the first match wins.
FAKE_GH = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import hashlib, json, os, subprocess, sys
    args = sys.argv[1:]
    joined = " ".join(args)
    with open(os.environ["FAKE_GH_LOG"], "a") as log:
        log.write(json.dumps(args) + "\\n")
    scenario = json.load(open(os.environ["FAKE_GH_SCENARIO"]))
    for needle, answer in scenario["answers"]:
        if needle in joined:
            if isinstance(answer, list):
                counter = os.environ["FAKE_GH_LOG"] + "." + hashlib.sha256(needle.encode()).hexdigest()[:12]
                n = int(open(counter).read()) if os.path.exists(counter) else 0
                open(counter, "w").write(str(n + 1))
                answer = answer[min(n, len(answer) - 1)]
            if answer is None:
                sys.exit(1)
            text = answer if isinstance(answer, str) else json.dumps(answer)
            if "--jq" in args:
                expr = args[args.index("--jq") + 1]
                text = subprocess.run(
                    ["jq", "-r", expr], input=text, text=True,
                    capture_output=True, check=True,
                ).stdout.strip()
            sys.stdout.write(text)
            sys.exit(0)
    sys.exit(1)
    """
)


def _steps() -> dict[str, dict]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return {step["name"]: step for step in workflow["jobs"]["merge"]["steps"]}


def _wait_loop() -> str:
    command = _steps()["Revalidate evidence and merge against live RuleSpec tip"]["run"]
    start = command.index("# Retry a read three times")
    merge_start = command.index("# With a merge queue on main")
    merge_end = command.index("merged=false\n")
    # Leave out the enqueue itself; the wait starts as if it just ran.
    command = command[:merge_start] + command[merge_end:]
    end = command.index("did not merge within 90 minutes")
    end = command.index("fi\n", end) + len("fi\n")
    # Each 30-second wait counts as 30 seconds without sleeping.
    loop = command[start:end].replace("sleep 30", "SECONDS=$((SECONDS + 30))")
    loop = loop.replace("sleep $((attempt * 5))", ":")
    return "set -euo pipefail\n" + loop.replace("sleep 10", ":")


def _run(tmp_path: Path, script: str, scenario: dict, env: dict[str, str]):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    log = tmp_path / "gh.log"
    log.touch()
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    result = subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_LOG": str(log),
            "FAKE_GH_SCENARIO": str(tmp_path / "scenario.json"),
            "RUNNER_TEMP": str(runner_temp),
            "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
            "PULL_REQUEST_NUMBER": "17",
            "HEAD_SHA": HEAD_SHA,
            "api_version": "2026-03-10",
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return result, calls


def _cleanup_scenario(name: str, oid: str) -> dict:
    head_ref = {"id": "REF_ID", "name": name, "target": {"oid": oid}}
    return {
        "answers": [
            ["deleteRef", {"data": {"deleteRef": {"clientMutationId": None}}}],
            [
                "headRef",
                {"data": {"repository": {"pullRequest": {"headRef": head_ref}}}},
            ],
        ]
    }


def test_cleanup_deletes_the_activation_ref_by_id_for_a_hash_branch(tmp_path):
    branch = "other#activation"
    result, calls = _run(
        tmp_path,
        _steps()["Delete merged activation branch"]["run"],
        _cleanup_scenario(branch, HEAD_SHA),
        {"HEAD_REF": branch},
    )
    assert result.returncode == 0, result.stderr
    deletes = [call for call in calls if "deleteRef" in " ".join(call)]
    assert len(deletes) == 1
    assert "id=REF_ID" in deletes[0]
    # No call names the branch in a URL, where "#" would cut it to "other".
    assert not any("refs/heads" in " ".join(call) for call in calls)
    assert not any(arg == "-X" for call in calls for arg in call)


@pytest.mark.parametrize(
    ("name", "oid"),
    [("other", HEAD_SHA), ("other#activation", "c" * 40)],
    ids=["other-branch-name", "branch-moved"],
)
def test_cleanup_skips_a_ref_that_is_not_the_reviewed_branch(tmp_path, name, oid):
    result, calls = _run(
        tmp_path,
        _steps()["Delete merged activation branch"]["run"],
        _cleanup_scenario(name, oid),
        {"HEAD_REF": "other#activation"},
    )
    assert result.returncode == 0, result.stderr
    assert not any("deleteRef" in " ".join(call) for call in calls)
    assert "not deleting it" in result.stdout


def _open_pr() -> dict:
    return {"state": "open", "merged_at": None}


def _entry(state: str) -> dict:
    entry = {"state": state}
    return {"data": {"repository": {"pullRequest": {"mergeQueueEntry": entry}}}}


QUEUE_RUNS = "actions/runs?event=merge_group"
NOW = "2999-01-01T00:00:00Z"
OLD = "2000-01-01T00:00:00Z"


def _runs(*runs: tuple[str, str]) -> dict:
    return {
        "workflow_runs": [
            {"head_branch": f"gh-readonly-queue/main/{name}", "created_at": created}
            for name, created in runs
        ]
    }


NO_RUNS = _runs(("pr-170-" + "c" * 40, NOW), ("pr-17-" + "d" * 40, OLD))


def test_wait_fails_when_queue_ci_never_starts(tmp_path):
    scenario = {
        "answers": [
            ["mergeQueueEntry", _entry("AWAITING_CHECKS")],
            [QUEUE_RUNS, NO_RUNS],
            ["pulls/17", _open_pr()],
        ]
    }
    result, calls = _run(tmp_path, _wait_loop(), scenario, {})
    assert result.returncode == 1
    assert "started no merge_group CI run" in result.stderr
    run_checks = [call for call in calls if QUEUE_RUNS in " ".join(call)]
    # Runs are matched on the queue branch of this pull request, not a SHA:
    # pr-170's run and pr-17's run from an earlier enqueue do not count.
    assert "gh-readonly-queue/main/pr-17-" in " ".join(run_checks[0])
    # 300 seconds at 30 per poll: it gave the CI about five minutes.
    assert 10 <= len(run_checks) <= 12


def test_wait_keeps_waiting_while_queue_ci_runs_then_sees_the_merge(tmp_path):
    merged = {"state": "closed", "merged_at": "2026-11-06T00:00:00Z"}
    scenario = {
        "answers": [
            ["mergeQueueEntry", _entry("AWAITING_CHECKS")],
            [QUEUE_RUNS, _runs(("pr-17-" + "e" * 40, NOW))],
            ["pulls/17", [_open_pr()] * 40 + [merged]],
        ]
    }
    result, calls = _run(
        tmp_path, _wait_loop() + 'test "$merged" = true\n', scenario, {}
    )
    assert result.returncode == 0, result.stderr
    # Once the queue's CI started, the wait stops looking for it.
    assert sum(QUEUE_RUNS in " ".join(call) for call in calls) == 1


def test_wait_restarts_the_five_minutes_after_a_group_rebuild(tmp_path):
    merged = {"state": "closed", "merged_at": "2026-11-06T00:00:00Z"}
    entries = (
        [_entry("AWAITING_CHECKS")] * 8
        + [_entry("QUEUED")] * 4
        + [_entry("AWAITING_CHECKS")] * 8
    )
    scenario = {
        "answers": [
            ["mergeQueueEntry", entries],
            [QUEUE_RUNS, NO_RUNS],
            ["pulls/17", [_open_pr()] * 20 + [merged]],
        ]
    }
    result, _ = _run(tmp_path, _wait_loop() + 'test "$merged" = true\n', scenario, {})
    assert result.returncode == 0, result.stderr


def test_wait_needs_a_new_ci_run_after_a_group_rebuild(tmp_path):
    # The first build's CI started; after the rebuild no run starts again.
    entries = [_entry("AWAITING_CHECKS")] * 2 + [_entry("QUEUED")] * 2
    entries += [_entry("AWAITING_CHECKS")] * 20
    started = _runs(("pr-17-" + "e" * 40, NOW))
    scenario = {
        "answers": [
            ["mergeQueueEntry", entries],
            [QUEUE_RUNS, [started, NO_RUNS]],
            ["pulls/17", _open_pr()],
        ]
    }
    result, _ = _run(tmp_path, _wait_loop(), scenario, {})
    assert result.returncode == 1
    assert "started no merge_group CI run" in result.stderr


def test_wait_does_not_count_time_spent_queued_behind_other_entries(tmp_path):
    merged = {"state": "closed", "merged_at": "2026-11-06T00:00:00Z"}
    scenario = {
        "answers": [
            [
                "mergeQueueEntry",
                [_entry("QUEUED")] * 20 + [_entry("AWAITING_CHECKS")],
            ],
            [QUEUE_RUNS, NO_RUNS],
            ["pulls/17", [_open_pr()] * 24 + [merged]],
        ]
    }
    result, _ = _run(tmp_path, _wait_loop() + 'test "$merged" = true\n', scenario, {})
    assert result.returncode == 0, result.stderr


def test_wait_retries_a_failed_read(tmp_path):
    merged = {"state": "closed", "merged_at": "2026-11-06T00:00:00Z"}
    # A None answer makes the fake gh exit 1, as for an API error.
    scenario = {"answers": [["pulls/17", [None, None, merged]]]}
    result, calls = _run(
        tmp_path, _wait_loop() + 'test "$merged" = true\n', scenario, {}
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 3


def _cleanup_state(merged: bool, queued: bool, auto_merge: bool) -> dict:
    pull = {
        "id": "PR_ID",
        "merged": merged,
        "mergeQueueEntry": {"state": "QUEUED"} if queued else None,
        "autoMergeRequest": {"enabledAt": "x"} if auto_merge else None,
    }
    return {
        "answers": [
            ["dequeuePullRequest", {"data": {}}],
            ["disablePullRequestAutoMerge", {"data": {}}],
            ["autoMergeRequest", {"data": {"repository": {"pullRequest": pull}}}],
        ]
    }


@pytest.mark.parametrize(
    ("merged", "queued", "auto_merge", "expected"),
    [
        (False, True, False, ["dequeuePullRequest"]),
        (False, False, True, ["disablePullRequestAutoMerge"]),
        (True, True, True, []),
        (False, False, False, []),
    ],
    ids=["queued", "auto-merge", "already-merged", "nothing-to-do"],
)
def test_failure_step_takes_an_unmerged_activation_out_of_the_queue(
    tmp_path, merged, queued, auto_merge, expected
):
    result, calls = _run(
        tmp_path,
        _steps()["Take an unmerged activation out of the merge queue"]["run"],
        _cleanup_state(merged, queued, auto_merge),
        {},
    )
    assert result.returncode == 0, result.stderr
    mutations = [call for call in calls if "mutation" in " ".join(call)]
    assert [
        name
        for call in mutations
        for name in ("dequeuePullRequest", "disablePullRequestAutoMerge")
        if name in " ".join(call)
    ] == expected
    # Both mutations take the pull request's node ID, not a queue entry ID.
    assert all("id=PR_ID" in call for call in mutations)


def test_failure_step_runs_only_after_a_failure_or_cancel():
    step = _steps()["Take an unmerged activation out of the merge queue"]
    assert "failure() || cancelled()" in step["if"]


def test_wait_fails_fast_on_a_closed_unmerged_pull_request(tmp_path):
    scenario = {"answers": [["pulls/17", {"state": "closed", "merged_at": None}]]}
    result, calls = _run(tmp_path, _wait_loop(), scenario, {})
    assert result.returncode == 1
    assert "closed without a merge" in result.stderr
    assert len(calls) == 1


def test_wait_rereads_once_when_the_entry_leaves_the_queue(tmp_path):
    no_entry = {"data": {"repository": {"pullRequest": {"mergeQueueEntry": None}}}}
    merged = {"state": "closed", "merged_at": "2026-11-06T00:00:00Z"}
    scenario = {
        "answers": [
            ["mergeQueueEntry", no_entry],
            ["pulls/17", [_open_pr(), merged]],
        ]
    }
    result, _ = _run(tmp_path, _wait_loop() + 'test "$merged" = true\n', scenario, {})
    assert result.returncode == 0, result.stderr

    scenario["answers"][1][1] = _open_pr()
    (tmp_path / "again").mkdir()
    result, _ = _run(tmp_path / "again", _wait_loop(), scenario, {})
    assert result.returncode == 1
    assert "the merge queue removed the activation pull request" in result.stderr
