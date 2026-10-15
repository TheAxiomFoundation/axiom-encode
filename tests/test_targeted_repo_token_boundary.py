import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _steps() -> list[dict]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    return workflow["jobs"]["encode"]["steps"]


def test_repo_token_steps_never_invoke_python_helpers() -> None:
    for step in _steps():
        if not any(
            "AXIOM_REPO_TOKEN" in str(value) for value in step.get("env", {}).values()
        ):
            continue
        command = step.get("run", "")
        assert "python" not in command, step["name"]
        assert "scripts/" not in command, step["name"]
        assert "uv " not in command, step["name"]


@pytest.mark.parametrize(
    ("helper", "consumer", "output"),
    [
        (
            "Parse RuleSpec-pinned axiom-compose reference",
            "Checkout RuleSpec-pinned axiom-compose",
            "${{ steps.axiom_compose_request.outputs.ref }}",
        ),
        (
            "Prepare reviewed lane pull request branch",
            "Push lane branch and open draft pull request",
            "${{ steps.reviewed_lane_pull_request_branch.outputs.branch }}",
        ),
        (
            "Prepare retired source metadata pull request branch",
            "Push retired source metadata migration branch and open draft pull request",
            "${{ steps.retired_source_metadata_pull_request_branch.outputs.branch }}",
        ),
    ],
)
def test_repo_token_helpers_run_earlier_without_secrets(
    helper: str, consumer: str, output: str
) -> None:
    steps = _steps()
    by_name = {step.get("name"): step for step in steps}
    preparation = by_name[helper]
    publication = by_name[consumer]
    assert steps.index(preparation) < steps.index(publication)
    assert "secrets." not in str(preparation.get("env", {}))
    assert "python" in preparation["run"]
    assert output in publication["env"].values()


def test_publication_preserves_verification_fetch_and_branch_order() -> None:
    steps = _steps()
    names = [step.get("name") for step in steps]
    for ordered_names in [
        [
            "Commit reviewed lane changes locally",
            "Verify reviewed lane pull request base",
            "Prepare reviewed lane pull request branch",
            "Push lane branch and open draft pull request",
        ],
        [
            "Commit retired source metadata migration locally",
            "Check retired source metadata publication token",
            "Verify retired source metadata migration before publication",
            "Verify retired source metadata pull request base",
            "Prepare retired source metadata pull request branch",
            "Push retired source metadata migration branch and open draft pull request",
        ],
    ]:
        indices = [names.index(name) for name in ordered_names]
        assert indices == sorted(indices)
        conditions = [steps[index]["if"] for index in indices[:-1]]
        assert len(set(conditions)) == 1
        assert conditions[0][4:-3] in steps[indices[-1]]["if"]
    verify = steps[
        names.index("Verify retired source metadata migration before publication")
    ]
    assert "verify-committed" in verify["run"]
    assert "secrets." not in str(verify["env"])


@pytest.mark.parametrize(
    ("publication_id", "preflight_id"),
    [
        ("publish_lane_pull_request", "verify_reviewed_lane_pull_request_base"),
        (
            "publish_retired_source_metadata_pull_request",
            "check_retired_source_metadata_publication_token",
        ),
    ],
)
def test_publication_reports_preparation_failure_without_publishing(
    publication_id: str, preflight_id: str
) -> None:
    by_id = {step.get("id"): step for step in _steps()}
    publication = by_id[publication_id]
    assert "!cancelled()" in publication["if"]
    assert f"steps.{preflight_id}.outcome != 'skipped'" in publication["if"]
    assert publication["env"]["PRIOR_STATUS"] == "${{ job.status }}"
    assert publication["run"].splitlines()[:2] == [
        "set -euo pipefail",
        'test "$PRIOR_STATUS" = success',
    ]
    # An attempted publication reports failure before it reads credentials or
    # reaches network commands when any separated preparation step failed.
    completed = subprocess.run(
        ["bash", "-c", publication["run"]],
        env={"PRIOR_STATUS": "failure"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr == ""
