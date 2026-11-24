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
    # The same choice as the encode job: the newest versioned commit of the
    # dispatched head, or the head alone when a maintainer asks for that.
    assert (
        'signing-commit --repo "$GITHUB_WORKSPACE/axiom-encode" --ref "$GITHUB_SHA"'
        in (steps[refuse]["run"])
    )
    assert steps[refuse]["env"]["HEAD_ONLY"] == (
        "${{ vars.AXIOM_SIGNING_HEAD_ONLY == 'true' }}"
    )
    assert "--head-only" in steps[refuse]["run"]
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


SELECT = "Select the versioned encoder commit"


def _encode_steps() -> list[dict]:
    return _job("targeted-signed-reencode.yml", "encode")["steps"]


def test_the_encode_job_selects_its_encoder_before_any_encoder_code_runs():
    steps = _encode_steps()
    select = _index(steps, SELECT)
    assert select == _index(steps, "Checkout axiom-encode") + 1
    assert steps[select - 1]["with"]["fetch-depth"] == 0
    run = steps[select]["run"]
    # Selected from the dispatched head, then checked out detached and clean.
    assert 'test "$(git -C axiom-encode rev-parse HEAD)" = "$GITHUB_SHA"' in run
    assert 'signing-commit --repo axiom-encode --ref "$GITHUB_SHA"' in run
    assert (
        'git -C axiom-encode merge-base --is-ancestor "$encoder_commit" "$GITHUB_SHA"'
        in run
    )
    assert 'checkout --quiet --detach "$encoder_commit"' in run
    assert 'test "$(git -C axiom-encode rev-parse HEAD)" = "$encoder_commit"' in run
    assert 'echo "AXIOM_ENCODER_COMMIT=$encoder_commit" >> "$GITHUB_ENV"' in run
    # A queue run keeps the old head-only rule: its finalizer binds the signed
    # artifact's encoder commit to the run head. A maintainer can force it.
    assert steps[select]["env"]["HEAD_ONLY"] == (
        "${{ inputs.queue_id != '' || vars.AXIOM_SIGNING_HEAD_ONLY == 'true' }}"
    )
    assert "--head-only" in run
    # No step before it runs repository code other than the selector itself.
    for step in steps[:select]:
        assert "run" not in step


def test_encoder_identity_steps_bind_the_selected_commit():
    steps = _encode_steps()
    by_name = {step.get("name"): step for step in steps}
    assert (
        'test "$(git rev-parse HEAD)" = "$AXIOM_ENCODER_COMMIT"'
        in by_name["Install locked preflight runtime"]["run"]
    )
    identities = by_name["Verify immutable checkout identities"]["run"]
    assert (
        'test "$(git -C axiom-encode rev-parse HEAD)" = "$AXIOM_ENCODER_COMMIT"'
        in identities
    )
    assert (
        'git -C axiom-encode merge-base --is-ancestor "$AXIOM_ENCODER_COMMIT" '
        '"$GITHUB_SHA"' in identities
    )
    provision = by_name["Provision protected signing supervisor"]["run"]
    assert '--encoder-commit "$AXIOM_ENCODER_COMMIT"' in provision
    assert '--workflow-commit "$GITHUB_SHA"' in provision
    assert '--encoder-commit "$GITHUB_SHA"' not in provision
    assert (
        'producer_commit=os.environ["AXIOM_ENCODER_COMMIT"]'
        in by_name["Validate producer guidance transport mode"]["run"]
    )
    text = (WORKFLOWS / "targeted-signed-reencode.yml").read_text()
    assert 'producer_commit=os.environ["GITHUB_SHA"]' not in text


def test_the_signing_workflow_runs_no_workflow_file_but_its_own():
    # The selection compares this one file between the head and the selected
    # commit. A reusable workflow or a local action would run code that the
    # comparison does not cover.
    workflow = yaml.safe_load((WORKFLOWS / "targeted-signed-reencode.yml").read_text())
    for name, job in workflow["jobs"].items():
        assert "uses" not in job, name
        for step in job["steps"]:
            assert not step.get("uses", "").startswith("./"), (name, step)
