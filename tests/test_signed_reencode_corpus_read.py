"""The signed re-encode reads R2-stored corpus scopes with read-only credentials.

Corpus scopes ingested after the storage switch exist only in R2, so the
provision-fetch step needs credentials. Those credentials are the organization's
read-only pair (R2_CORPUS_READ_*), they reach exactly one step, and their absence
stops the run before the production-signing gate rather than after it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/targeted-signed-reencode.yml"
FETCH_STEP = "Fetch pinned corpus provision objects"
PREFLIGHT_STEP = "Require corpus read credentials before the signing gate"
READ_ID = "${{ secrets.R2_CORPUS_READ_ACCESS_KEY_ID }}"
READ_SECRET = "${{ secrets.R2_CORPUS_READ_SECRET_ACCESS_KEY }}"
R2_SECRET = re.compile(r"secrets\s*\.\s*R2_", re.IGNORECASE)


def _workflow() -> dict:
    return yaml.safe_load((ROOT / WORKFLOW).read_text(encoding="utf-8"))


def _steps(workflow: dict):
    for job_id, job in workflow["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            yield job_id, index, step


def _step(workflow: dict, name: str) -> tuple[str, int, dict]:
    matches = [(j, i, s) for j, i, s in _steps(workflow) if s.get("name") == name]
    assert len(matches) == 1, f"expected exactly one step named {name!r}"
    return matches[0]


def test_fetch_step_uses_the_read_only_corpus_secrets():
    _, _, step = _step(_workflow(), FETCH_STEP)
    env = step["env"]
    assert env["R2_ACCESS_KEY_ID"] == READ_ID
    assert env["R2_SECRET_ACCESS_KEY"] == READ_SECRET


def test_only_the_fetch_step_holds_r2_credentials():
    """The preflight may test that the secrets are set; only the fetch step gets values."""
    holders = []
    for job_id, _, step in _steps(_workflow()):
        for key, value in (step.get("env") or {}).items():
            if isinstance(value, str) and R2_SECRET.search(value):
                holders.append((job_id, step.get("name"), key, value))
    value_holders = {
        (job, name) for job, name, _, value in holders if "!=" not in value
    }
    presence_checks = {(job, name) for job, name, _, value in holders if "!=" in value}
    assert value_holders == {("encode", FETCH_STEP)}
    assert presence_checks == {("attempt_budget", PREFLIGHT_STEP)}


def test_the_preflight_runs_before_the_gate_and_before_the_budget():
    workflow = _workflow()
    job_id, index, step = _step(workflow, PREFLIGHT_STEP)
    assert job_id == "attempt_budget"
    assert "environment" not in workflow["jobs"][job_id]
    names = [s.get("name") for s in workflow["jobs"][job_id]["steps"]]
    assert index < names.index("Enforce failed-attempt budget")
    assert workflow["jobs"]["encode"]["needs"] == "attempt_budget"
    expression = step["env"]["HAS_CORPUS_READ"]
    assert "secrets.R2_CORPUS_READ_ACCESS_KEY_ID != ''" in expression
    assert "secrets.R2_CORPUS_READ_SECRET_ACCESS_KEY != ''" in expression


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
    )


def test_the_preflight_fails_unless_both_secrets_are_set():
    _, _, step = _step(_workflow(), PREFLIGHT_STEP)
    script = step["run"]
    assert _run(script, {"HAS_CORPUS_READ": "true"}).returncode == 0
    for value in ("false", "", "True"):
        result = _run(script, {"HAS_CORPUS_READ": value})
        assert result.returncode == 1
        assert "R2_CORPUS_READ_ACCESS_KEY_ID" in result.stderr


def test_the_fetch_step_stops_before_fetching_without_credentials(tmp_path):
    _, _, step = _step(_workflow(), FETCH_STEP)
    # Run only the guard: everything before the first corpus-lock branch.
    guard = step["run"].split("if [ -d axiom-corpus/.axiom/corpus-locks ]", 1)[0]
    for env in ({}, {"R2_ACCESS_KEY_ID": "id"}, {"R2_SECRET_ACCESS_KEY": "s"}):
        result = _run(guard, env)
        assert result.returncode == 1
        assert "R2 corpus read credentials" in result.stderr
    assert (
        _run(guard, {"R2_ACCESS_KEY_ID": "id", "R2_SECRET_ACCESS_KEY": "s"}).returncode
        == 0
    )
