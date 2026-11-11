"""The workflow steps that keep a signed encode off an unversioned encoder head.

axiom-encode#2048 assigns the encoder version after merge, so main can briefly
carry encoder changes that no bump covers. These steps refuse such a head before
any production-signing approval; nothing else reads them, so pin them here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github/workflows"
CHECK = "version_provenance.py"
REFUSE = "Refuse an encoder head that no version bump covers"


def _job(workflow: str, job: str) -> dict:
    return yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"][job]


def _index(steps: list[dict], name: str) -> int:
    return next(i for i, step in enumerate(steps) if step.get("name") == name)


def test_the_re_encode_budget_job_refuses_an_unversioned_head_first():
    workflow = yaml.safe_load((WORKFLOWS / "targeted-signed-reencode.yml").read_text())
    assert workflow["jobs"]["encode"]["needs"] == "attempt_budget"
    steps = workflow["jobs"]["attempt_budget"]["steps"]
    checkout = steps[_index(steps, "Checkout axiom-encode")]
    assert checkout["with"]["fetch-depth"] == 0
    refuse = _index(steps, REFUSE)
    assert CHECK in steps[refuse]["run"]
    assert "check --repo" in steps[refuse]["run"]
    assert "if" not in steps[refuse]
    assert _index(steps, "Checkout axiom-encode") < refuse
    assert refuse < _index(steps, "Enforce failed-attempt budget")


def test_the_protected_queue_refuses_to_dispatch_an_unversioned_head():
    workflow = yaml.safe_load(
        (WORKFLOWS / "dispatch-signed-snap-queue.yml").read_text()
    )
    job = next(
        job
        for job in workflow["jobs"].values()
        if any(step.get("name") == REFUSE for step in job["steps"])
    )
    steps = job["steps"]
    refuse = steps[_index(steps, REFUSE)]
    assert refuse["if"] == "inputs.dispatch"
    assert f"{CHECK} check --repo . --ref HEAD" in refuse["run"]
    assert _index(steps, REFUSE) < _index(
        steps, "Plan or dispatch idempotent protected runs"
    )
    checkout = next(
        step for step in steps if "actions/checkout" in step.get("uses", "")
    )
    assert checkout["with"]["fetch-depth"] == 0


def test_the_queue_tick_checks_the_encoder_out_with_full_history():
    workflow = yaml.safe_load((WORKFLOWS / "snap-dispatch-queue.yml").read_text())
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if step.get("name") == "Checkout axiom-encode":
                assert step["with"]["fetch-depth"] == 0
                return
    raise AssertionError("snap-dispatch-queue.yml checks no encoder out")


def _commit_all(repo: Path, message: str) -> None:
    for command in (
        ["add", "-A"],
        [
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            message,
        ],
    ):
        subprocess.run(
            ["git", "-C", str(repo), *command], check=True, capture_output=True
        )


def _versioned_checkout(workspace: Path) -> Path:
    """An ``axiom-encode`` checkout whose only commit declares its version."""
    repo = workspace / "axiom-encode"
    package = repo / "src/axiom_encode"
    package.mkdir(parents=True)
    script = ROOT / "src/axiom_encode/version_provenance.py"
    (package / "version_provenance.py").write_text(script.read_text())
    (package / "cli.py").write_text("MARKER = 0\n")
    (package / "__init__.py").write_text('__version__ = "0.2.10"\n')
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "axiom-encode"\nversion = "0.2.10"\n'
    )
    (repo / "uv.lock").write_text(
        '[[package]]\nname = "axiom-encode"\nversion = "0.2.10"\n'
        'source = { editable = "." }\n'
    )
    subprocess.run(
        ["git", "init", "-q", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    _commit_all(repo, "init")
    return repo


@pytest.mark.parametrize(
    "workflow", ["targeted-signed-reencode.yml", "dispatch-signed-snap-queue.yml"]
)
def test_the_refusal_step_stops_an_unversioned_head(workflow: str, tmp_path: Path):
    document = yaml.safe_load((WORKFLOWS / workflow).read_text())
    job, step = next(
        (job, step)
        for job in document["jobs"].values()
        for step in job["steps"]
        if step.get("name") == REFUSE
    )
    # With no `shell` or `defaults`, GitHub runs the step as `bash -e {0}`;
    # with `continue-on-error`, a refusal would not stop the job.
    assert "shell" not in step and "defaults" not in job
    assert "defaults" not in document
    assert "continue-on-error" not in step and "continue-on-error" not in job
    repo = _versioned_checkout(tmp_path)

    def run_step() -> subprocess.CompletedProcess[str]:
        # A runner's default shell for a step on Linux.
        return subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            cwd=repo,
            env={"PATH": os.environ["PATH"], "GITHUB_WORKSPACE": str(tmp_path)},
            capture_output=True,
            text=True,
        )

    versioned = run_step()
    assert versioned.returncode == 0, versioned.stderr

    (repo / "src/axiom_encode/cli.py").write_text("MARKER = 1\n")
    _commit_all(repo, "encoder change")

    refused = run_step()
    assert refused.returncode != 0
    assert "after the latest encoder version bump" in refused.stderr
