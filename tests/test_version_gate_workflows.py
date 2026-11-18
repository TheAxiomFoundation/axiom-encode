"""The workflow steps that keep a signed encode off an unversioned encoder head.

axiom-encode#2048 assigns the encoder version after merge, so main can briefly
carry encoder changes that no bump covers. These steps refuse such a head before
any production-signing approval; nothing else reads them, so pin them here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).parents[1] / ".github/workflows"
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
