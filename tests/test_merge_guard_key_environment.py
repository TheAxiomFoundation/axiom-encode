"""Pin which workflow jobs can select the guard App key environment.

``docs/merge-guard-key-protection.md`` proposes a mechanism for checklist
step 1 of ``docs/content-bound-merge-guard.md`` section 8. GitHub makes an
environment's secrets available to every job that references the environment,
subject to its protections, and a deployment branch rule limits the ref, not
the workflow file or job. So the jobs that can select the environment that
will hold the guard App's private key must stay a fixed, reviewed list,
whichever mechanism the maintainers choose.

This file is a tripwire, not the control. A pull request can edit it along
with a workflow. The control the proposal recommends is code-owner review of
every change under ``.github/``; this test makes an unintended selector fail
CI and gives that review a fixed list to compare against. It changes nothing
at run time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# The environment name in the design and #2237, and the new name the proposal
# gives the key so that no earlier workflow revision names it (a re-run uses
# the original commit). Compared case-insensitively.
KEY_ENVIRONMENTS = frozenset({"merge-guard", "merge-guard-key"})
GUARD_WORKFLOW = "merge-guard-runner.yml"
# (workflow file name, job id) pairs that may select a key environment.
ALLOWED_SELECTORS = frozenset({(GUARD_WORKFLOW, "run")})
LOCAL_REUSABLE_PREFIX = "./.github/workflows/"


def _workflows() -> dict[str, object]:
    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths
    }


def _triggers(workflow: dict) -> object:
    # PyYAML reads the bare key ``on`` as the boolean True.
    return workflow.get("on", workflow.get(True))


def _declares_workflow_call(triggers: object) -> bool:
    if isinstance(triggers, (dict, list)):
        return "workflow_call" in triggers
    return triggers == "workflow_call"


def selection_problems(workflows: dict[str, object]) -> list[str]:
    """Return every way a job could reach a key environment outside the list."""

    problems: list[str] = []
    for name, workflow in workflows.items():
        if not isinstance(workflow, dict):
            problems.append(f"{name}: not a workflow mapping")
            continue
        if name == GUARD_WORKFLOW and _declares_workflow_call(_triggers(workflow)):
            problems.append(
                f"{name}: declares workflow_call, so another workflow could run "
                "its key-bearing job"
            )
        jobs = workflow.get("jobs")
        if not isinstance(jobs, dict):
            problems.append(f"{name}: jobs is not a mapping")
            continue
        for job_id, job in jobs.items():
            where = f"{name}:{job_id}"
            if not isinstance(job, dict):
                problems.append(f"{where}: job is not a mapping")
                continue
            if "environment" in job:
                environment = job["environment"]
                if isinstance(environment, dict):
                    environment = environment.get("name")
                if not isinstance(environment, str) or not environment.strip():
                    problems.append(f"{where}: environment has no static name")
                elif "${{" in environment:
                    problems.append(
                        f"{where}: environment is chosen by an expression, "
                        "which can evaluate to the key environment"
                    )
                elif (
                    environment.strip().casefold() in KEY_ENVIRONMENTS
                    and (name, job_id) not in ALLOWED_SELECTORS
                ):
                    problems.append(
                        f"{where}: selects the guard key environment {environment!r}"
                    )
            if "uses" in job:
                uses = job["uses"]
                if not isinstance(uses, str) or not uses.startswith(
                    LOCAL_REUSABLE_PREFIX
                ):
                    problems.append(
                        f"{where}: calls a reusable workflow from outside this "
                        f"repository ({uses!r}); a job there can set environment"
                    )
                elif Path(uses.split("@", 1)[0]).name == GUARD_WORKFLOW:
                    problems.append(f"{where}: calls the guard workflow")
    return problems


def test_every_workflow_parses_as_a_mapping_of_jobs():
    workflows = _workflows()
    assert workflows, "no workflow files found"
    for name, workflow in workflows.items():
        assert isinstance(workflow, dict), name
        assert isinstance(workflow.get("jobs"), dict), name


def test_only_the_listed_job_can_select_the_guard_key_environment():
    assert selection_problems(_workflows()) == []


def _workflow(jobs: dict, *, on: object = "workflow_dispatch") -> dict:
    return {"on": on, "jobs": jobs}


def _job(**fields: object) -> dict:
    return {"runs-on": "ubuntu-latest", "steps": [{"run": "true"}], **fields}


@pytest.mark.parametrize(
    "workflows",
    [
        pytest.param(
            {"other.yml": _workflow({"steal": _job(environment="merge-guard")})},
            id="another-workflow-names-the-environment",
        ),
        pytest.param(
            {"other.yml": _workflow({"steal": _job(environment="merge-guard-key")})},
            id="another-workflow-names-the-new-environment",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {"steal": _job(environment={"name": " Merge-Guard "})}
                )
            },
            id="mapping-form-other-case-and-spaces",
        ),
        pytest.param(
            {GUARD_WORKFLOW: _workflow({"gate": _job(environment="merge-guard")})},
            id="guard-workflow-other-job",
        ),
        pytest.param(
            {
                "merge-guard-runner-copy.yml": _workflow(
                    {"run": _job(environment="merge-guard")}
                )
            },
            id="same-job-id-in-a-copied-workflow",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {"pick": _job(environment="${{ inputs.target }}")}
                )
            },
            id="dynamic-selector",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {"pick": _job(environment={"name": "${{ vars.TARGET }}"})}
                )
            },
            id="dynamic-selector-in-mapping",
        ),
        pytest.param(
            {"other.yml": _workflow({"pick": _job(environment=None)})},
            id="environment-without-a-name",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {
                        "call": {
                            "uses": "TheAxiomFoundation/.github/.github/workflows/x.yml@main",
                            "secrets": "inherit",
                        }
                    }
                )
            },
            id="remote-reusable-workflow",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {"call": {"uses": f"{LOCAL_REUSABLE_PREFIX}{GUARD_WORKFLOW}"}}
                )
            },
            id="local-call-to-the-guard-workflow",
        ),
        pytest.param(
            {
                GUARD_WORKFLOW: _workflow(
                    {"run": _job(environment="merge-guard")},
                    on={"workflow_dispatch": None, "workflow_call": None},
                )
            },
            id="guard-workflow-callable-mapping",
        ),
        pytest.param(
            {
                GUARD_WORKFLOW: _workflow(
                    {"run": _job(environment="merge-guard")},
                    on=["workflow_dispatch", "workflow_call"],
                )
            },
            id="guard-workflow-callable-list",
        ),
        pytest.param(
            {
                GUARD_WORKFLOW: {
                    True: "workflow_call",
                    "jobs": {"run": _job(environment="merge-guard")},
                }
            },
            id="guard-workflow-callable-bare-on-key",
        ),
        pytest.param({"broken.yml": None}, id="empty-workflow-file"),
        pytest.param({"broken.yml": {"on": "push"}}, id="workflow-without-jobs"),
    ],
)
def test_the_tripwire_reports_each_way_to_reach_the_key(workflows):
    assert selection_problems(workflows) != []


@pytest.mark.parametrize(
    "workflows",
    [
        pytest.param(
            {GUARD_WORKFLOW: _workflow({"run": _job(environment="merge-guard")})},
            id="the-listed-job",
        ),
        pytest.param(
            {GUARD_WORKFLOW: _workflow({"run": _job(environment="merge-guard-key")})},
            id="the-listed-job-with-the-new-name",
        ),
        pytest.param(
            {"other.yml": _workflow({"sign": _job(environment="production-signing")})},
            id="another-static-environment",
        ),
        pytest.param(
            {
                "other.yml": _workflow(
                    {"call": {"uses": f"{LOCAL_REUSABLE_PREFIX}helper.yml"}}
                )
            },
            id="local-call-to-another-workflow",
        ),
    ],
)
def test_the_tripwire_accepts_the_listed_and_unrelated_selectors(workflows):
    assert selection_problems(workflows) == []
