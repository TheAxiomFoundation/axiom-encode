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

from axiom_encode import version_provenance

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
    # The one place that reads the variable; the encode job uses its outputs.
    assert steps[refuse]["env"] == {
        "HEAD_ONLY_SETTING": "${{ vars.AXIOM_SIGNING_HEAD_ONLY }}"
    }
    assert steps[refuse]["id"] == "encoder_head"
    assert workflow["jobs"]["attempt_budget"]["outputs"] == {
        "head_only": "${{ steps.encoder_head.outputs.head_only }}",
        "encoder_commit": "${{ steps.encoder_head.outputs.encoder_commit }}",
    }
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
    # The selector that decides is the candidate commit's own stored copy,
    # never a file of the checkout (the head's, possibly unversioned, code).
    assert (
        "git -C axiom-encode cat-file blob \\\n"
        '  "$candidate:src/axiom_encode/version_provenance.py" > "$selector"' in run
    )
    assert 'python3 -I "$selector"' in run
    assert "python3 -I axiom-encode/" not in run
    assert 'selector="$RUNNER_TEMP/' in run
    assert (
        f"rev-list --first-parent --max-count={version_provenance.SIGNING_MAX_BEHIND + 1} "
        '"$GITHUB_SHA"' in run
    )
    assert (
        f"one of the {version_provenance.SIGNING_MAX_BEHIND + 1} newest first-parent"
        in run
    )
    assert (
        'git -C axiom-encode merge-base --is-ancestor "$encoder_commit" "$GITHUB_SHA"'
        in run
    )
    assert 'checkout --quiet --detach "$encoder_commit"' in run
    assert 'test "$(git -C axiom-encode rev-parse HEAD)" = "$encoder_commit"' in run
    assert 'echo "AXIOM_ENCODER_COMMIT=$encoder_commit" >> "$GITHUB_ENV"' in run
    # A queue run keeps the old head-only rule: its finalizer binds the signed
    # artifact's encoder commit to the run head. A maintainer can force it.
    assert steps[select]["env"] == {
        "HEAD_ONLY": (
            "${{ inputs.queue_id != '' "
            "|| needs.attempt_budget.outputs.head_only != 'false' }}"
        ),
        # Read here too, so a re-run of this job sees a switch set after the
        # first attempt; it can only add head-only.
        "HEAD_ONLY_SETTING": "${{ vars.AXIOM_SIGNING_HEAD_ONLY }}",
        "BUDGET_ENCODER_COMMIT": "${{ needs.attempt_budget.outputs.encoder_commit }}",
    }
    assert (
        'test -z "$(git -C axiom-encode status --porcelain --ignored '
        '--untracked-files=all)"' in run
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


def _step_run(job: str, name: str) -> str:
    steps = _job("targeted-signed-reencode.yml", job)["steps"]
    return steps[_index(steps, name)]["run"]


def _fake_tools(bin_dir: Path, encoder_commit: str) -> Path:
    """python3 records its arguments and the content of the selector file it
    is given (``-I <file> ...``), and prints the selected commit, or, with
    --head-only, refuses as the selector does at an unversioned head; python
    (the infra issue recorder) does nothing."""
    bin_dir.mkdir()
    log = bin_dir / "python3.log"
    (bin_dir / "python3").write_text(
        f'#!/bin/sh\necho "$@" >> "{log}"\n'
        f'if [ -f "$2" ]; then cat "$2" >> "{log}"; fi\n'
        'case "$*" in *--head-only*) echo "signs only from the dispatched head" >&2; '
        "exit 1;; esac\n"
        f"echo {encoder_commit}\n"
    )
    (bin_dir / "python").write_text("#!/bin/sh\nexit 0\n")
    for tool in ("python3", "python"):
        (bin_dir / tool).chmod(0o755)
    return log


@pytest.mark.parametrize(
    "setting,head_only",
    [
        ("", "false"),
        ("false", "false"),
        ("true", "true"),
        ("True", "true"),
        ("TRUE", "true"),
        ("1", "true"),
        ("yes", "true"),
        ("False", "true"),
        (" false", "true"),
        ("0", "true"),
    ],
)
def test_the_head_only_variable_fails_closed(tmp_path: Path, setting, head_only):
    log = _fake_tools(tmp_path / "bin", "c" * 40)
    output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash", "-c", _step_run("attempt_budget", REFUSE)],
        env={
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "HEAD_ONLY_SETTING": setting,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_SHA": "a" * 40,
            "GITHUB_WORKSPACE": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert completed.returncode == (1 if head_only == "true" else 0), completed.stderr
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if line
    )
    expected = {"head_only": head_only}
    if head_only == "false":
        expected["encoder_commit"] = "c" * 40
    assert outputs == expected
    assert ("--head-only" in log.read_text()) == (head_only == "true")


def _encoder_checkout(root: Path) -> tuple[str, str]:
    """axiom-encode with two commits: (older C, head H)."""
    repo = root / "axiom-encode"
    repo.mkdir(parents=True)
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    commits = []
    selector = repo / "src/axiom_encode/version_provenance.py"
    selector.parent.mkdir(parents=True)
    for marker in ("versioned", "head"):
        (repo / "file.txt").write_text(marker + "\n")
        selector.write_text(f"# selector of the {marker} commit\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", marker], check=True)
        commits.append(
            subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    return commits[0], commits[1]


def _run_select(
    tmp_path: Path,
    *,
    head_only: str,
    budget: str,
    ignored: bool = False,
    setting: str = "",
    attempt: str = "1",
    selected: str = "OLDER",
    extra_commits: int = 0,
):
    older, head = _encoder_checkout(tmp_path)
    for number in range(extra_commits):
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path / "axiom-encode"),
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                f"later {number}",
            ],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(tmp_path / "axiom-encode"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if ignored:
        (tmp_path / "axiom-encode/.gitignore").write_text("leftover\n")
        subprocess.run(
            ["git", "-C", str(tmp_path / "axiom-encode"), "add", ".gitignore"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_path / "axiom-encode"), "commit", "-qm", "ignore"],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(tmp_path / "axiom-encode"), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (tmp_path / "axiom-encode/leftover").write_text("x\n")
        older = head
    log = _fake_tools(
        tmp_path / "bin", selected.replace("OLDER", older).replace("HEAD", head)
    )
    env_file = tmp_path / "github-env"
    (tmp_path / "runner-temp").mkdir()
    completed = subprocess.run(
        ["bash", "-c", _step_run("encode", SELECT)],
        cwd=tmp_path,
        env={
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "HEAD_ONLY": head_only,
            "HEAD_ONLY_SETTING": setting,
            "BUDGET_ENCODER_COMMIT": budget.replace("OLDER", older).replace(
                "HEAD", head
            ),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
            "GITHUB_ENV": str(env_file),
            "GITHUB_RUN_ATTEMPT": attempt,
            "GITHUB_SHA": head,
        },
        capture_output=True,
        text=True,
    )
    return completed, older, head, log, env_file


def test_the_selection_step_checks_out_the_older_commit(tmp_path: Path):
    completed, older, head, log, env_file = _run_select(
        tmp_path, head_only="false", budget="OLDER"
    )
    assert completed.returncode == 0, completed.stderr
    assert env_file.read_text() == f"AXIOM_ENCODER_COMMIT={older}\n"
    assert "--head-only" not in log.read_text()
    checked_out = subprocess.run(
        ["git", "-C", str(tmp_path / "axiom-encode"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert checked_out == older != head
    assert "::notice::Signing from versioned encoder commit" in completed.stdout
    # The selector that ran is the older commit's stored copy, and it is gone.
    assert "# selector of the versioned commit" in log.read_text()
    assert "# selector of the head commit" not in log.read_text()
    assert list((tmp_path / "runner-temp").iterdir()) == []


def test_a_head_only_run_uses_the_heads_own_selector(tmp_path: Path):
    # Head-only ignores the budget job's candidate: only the head may sign,
    # and the head's selector decides whether it is versioned.
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only="true", budget="OLDER"
    )
    assert completed.returncode != 0 and not env_file.exists()
    assert "# selector of the head commit" in log.read_text()
    assert "# selector of the versioned commit" not in log.read_text()


def test_the_selection_step_refuses_a_candidate_whose_selector_disagrees(
    tmp_path: Path,
):
    # The budget job (the head's code) names the older commit, but that
    # commit's own selector selects the head: the run refuses.
    completed, _, _, _, env_file = _run_select(
        tmp_path, head_only="false", budget="OLDER", selected="HEAD"
    )
    assert completed.returncode != 0 and not env_file.exists()
    assert "the attempt-budget job selected another encoder commit" in completed.stderr


def test_the_selection_step_refuses_an_older_commit_without_a_budget_candidate(
    tmp_path: Path,
):
    # No candidate from the budget job: the head's selector may select only
    # the head itself.
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only="false", budget="", selected="OLDER"
    )
    assert completed.returncode != 0 and not env_file.exists()
    assert "own selector selects another commit" in completed.stderr
    assert "# selector of the head commit" in log.read_text()


def test_the_selection_step_refuses_a_candidate_beyond_the_lag_bound(tmp_path: Path):
    # The candidate is an ancestor on the first-parent chain, but 6 commits
    # below the head: its code is never run.
    completed, _, _, log, env_file = _run_select(
        tmp_path,
        head_only="false",
        budget="OLDER",
        extra_commits=version_provenance.SIGNING_MAX_BEHIND,
    )
    assert completed.returncode != 0 and not env_file.exists()
    assert "newest first-parent commits" in completed.stderr
    assert not log.exists()


def test_the_selection_step_accepts_a_candidate_at_the_lag_bound(tmp_path: Path):
    completed, older, _, _, env_file = _run_select(
        tmp_path,
        head_only="false",
        budget="OLDER",
        extra_commits=version_provenance.SIGNING_MAX_BEHIND - 1,
    )
    assert completed.returncode == 0, completed.stderr
    assert env_file.read_text() == f"AXIOM_ENCODER_COMMIT={older}\n"


@pytest.mark.parametrize("budget", ["HEAD~1", "main", "--help", "D" * 40])
def test_the_selection_step_refuses_a_malformed_candidate(tmp_path: Path, budget):
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only="false", budget=budget.replace("HEAD~1", "@~1")
    )
    assert completed.returncode != 0 and not env_file.exists()
    assert "not a full lowercase commit SHA" in completed.stderr
    assert not log.exists()


@pytest.mark.parametrize("head_only", ["true", "", "False", "anything"])
def test_the_selection_step_is_head_only_unless_exactly_false(
    tmp_path: Path, head_only
):
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only=head_only, budget=""
    )
    assert "--head-only" in log.read_text()
    assert completed.returncode != 0 and not env_file.exists()


@pytest.mark.parametrize("setting", ["true", "1", "True", "yes", " false"])
def test_a_switch_set_after_the_budget_job_still_forces_the_head(
    tmp_path: Path, setting
):
    # The budget job's outputs said "new rule"; the variable was set later.
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only="false", budget="OLDER", setting=setting
    )
    assert "--head-only" in log.read_text()
    assert completed.returncode != 0 and not env_file.exists()
    assert "AXIOM_SIGNING_HEAD_ONLY is set" in completed.stderr


@pytest.mark.parametrize("setting", ["", "false"])
def test_an_unset_or_false_switch_keeps_the_budget_decision(tmp_path: Path, setting):
    completed, older, _, log, env_file = _run_select(
        tmp_path, head_only="false", budget="OLDER", setting=setting
    )
    assert completed.returncode == 0, completed.stderr
    assert "--head-only" not in log.read_text()
    assert env_file.read_text() == f"AXIOM_ENCODER_COMMIT={older}\n"


@pytest.mark.parametrize("attempt", ["2", "3", ""])
def test_a_rerun_signs_only_from_the_head(tmp_path: Path, attempt):
    completed, _, _, log, env_file = _run_select(
        tmp_path, head_only="false", budget="OLDER", attempt=attempt
    )
    assert "--head-only" in log.read_text()
    assert completed.returncode != 0 and not env_file.exists()
    assert "a re-run (attempt" in completed.stderr


def test_the_selection_step_refuses_a_commit_the_budget_job_did_not_pick(
    tmp_path: Path,
):
    completed, _, _, _, env_file = _run_select(
        tmp_path, head_only="false", budget="d" * 40
    )
    assert completed.returncode != 0
    assert "newest first-parent commits" in completed.stderr
    assert not env_file.exists()


def test_the_selection_step_refuses_an_ignored_leftover(tmp_path: Path):
    completed, _, _, _, env_file = _run_select(
        tmp_path, head_only="false", budget="", ignored=True
    )
    assert completed.returncode != 0
    assert not env_file.exists()
