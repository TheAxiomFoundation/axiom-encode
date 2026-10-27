from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github/workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_snap_finalizer_pr_token_is_environment_scoped_and_main_only() -> None:
    job = _workflow("finalize-signed-snap-queue.yml")["jobs"]["finalize"]

    assert job.get("environment") == "snap-queue-finalization"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.run_attempt == 1" in job["if"]


def test_snap_finalizer_uses_repo_token_only_to_create_pr() -> None:
    workflow = _workflow("finalize-signed-snap-queue.yml")
    job = workflow["jobs"]["finalize"]
    steps = job["steps"]
    publish = next(
        step
        for step in steps
        if step.get("name") == "Open reviewed queue activation pull request"
    )
    precedent = next(
        step
        for step in _workflow("targeted-signed-reencode.yml")["jobs"]["encode"]["steps"]
        if step.get("id") == "publish_lane_pull_request"
    )

    assert publish["env"]["GH_TOKEN"] == precedent["env"]["GH_TOKEN"]
    assert publish["env"]["GH_TOKEN"] == "${{ secrets.AXIOM_REPO_TOKEN }}"
    assert 'test -n "$GH_TOKEN"' in publish["run"]
    assert 'gh api --method POST "repos/$GITHUB_REPOSITORY/pulls"' in publish["run"]
    assert "GH_TOKEN" not in workflow.get("env", {})
    assert "GH_TOKEN" not in job.get("env", {})
    token_steps = [
        step
        for step in steps
        if step is not publish and "GH_TOKEN" in step.get("env", {})
    ]
    assert {step["name"] for step in token_steps} == {
        "Fetch live finalization evidence",
        "Create exact queue activation commit",
    }
    assert all(step["env"]["GH_TOKEN"] == "${{ github.token }}" for step in token_steps)
    assert all(
        "secrets.AXIOM_REPO_TOKEN" not in str(step)
        for step in steps
        if step is not publish
    )


@pytest.mark.parametrize(
    "workflow_name,job_name",
    [
        ("validate-snap-queue-activation.yml", "validate"),
        ("merge-snap-queue-activation.yml", "merge"),
    ],
)
def test_snap_activation_consumers_keep_github_token_without_creating_prs(
    workflow_name: str, job_name: str
) -> None:
    workflow = _workflow(workflow_name)
    job = workflow["jobs"][job_name]
    token_steps = [step for step in job["steps"] if "GH_TOKEN" in step.get("env", {})]

    assert token_steps
    assert all(step["env"]["GH_TOKEN"] == "${{ github.token }}" for step in token_steps)
    for step in job["steps"]:
        command = step.get("run", "")
        assert "gh pr create" not in command
        assert 'gh api --method POST "repos/$GITHUB_REPOSITORY/pulls"' not in command
        assert "secrets.AXIOM_REPO_TOKEN" not in str(step)
