"""The signed re-encode reads R2-stored corpus scopes with read-only credentials.

Corpus scopes ingested after the storage switch exist only in R2, so the
provision-fetch step needs credentials. Those credentials are the organization's
read-only pair (R2_CORPUS_READ_*), they reach exactly one step, and their absence
stops the run before the production-signing gate rather than after it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
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


PRESENCE = (
    "${{ secrets.R2_CORPUS_READ_ACCESS_KEY_ID != '' "
    "&& secrets.R2_CORPUS_READ_SECRET_ACCESS_KEY != '' }}"
)
# Any spelling of a reference to an R2 secret: dot or bracket access, any case.
R2_REFERENCE = re.compile(r"\bsecrets\s*(?:\.\s*R2_|\[\s*['\"]R2_)", re.IGNORECASE)


def _strings(value: object, path: tuple = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _strings(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _strings(child, (*path, index))
    elif isinstance(value, str):
        yield path, value


def test_r2_secrets_appear_only_at_the_two_sanctioned_places():
    """Walk the whole workflow: R2 secret values reach only the fetch step's two
    env entries, and the preflight's single env entry tests presence only."""
    workflow = _workflow()
    _, fetch_index, _ = _step(workflow, FETCH_STEP)
    _, preflight_index, _ = _step(workflow, PREFLIGHT_STEP)
    expected = {
        ("jobs", "encode", "steps", fetch_index, "env", "R2_ACCESS_KEY_ID"): READ_ID,
        (
            "jobs",
            "encode",
            "steps",
            fetch_index,
            "env",
            "R2_SECRET_ACCESS_KEY",
        ): READ_SECRET,
        (
            "jobs",
            "attempt_budget",
            "steps",
            preflight_index,
            "env",
            "HAS_CORPUS_READ",
        ): PRESENCE,
    }
    found = {
        path: value for path, value in _strings(workflow) if R2_REFERENCE.search(value)
    }
    assert found == expected


def test_the_preflight_env_is_exactly_the_presence_check_and_cannot_be_skipped():
    workflow = _workflow()
    _, _, step = _step(workflow, PREFLIGHT_STEP)
    assert step["env"] == {"HAS_CORPUS_READ": PRESENCE}
    assert "continue-on-error" not in step
    assert "if" not in step
    assert "continue-on-error" not in workflow["jobs"]["attempt_budget"]


def test_encode_is_admitted_only_after_a_successful_or_skipped_preflight():
    condition = _workflow()["jobs"]["encode"]["if"]
    results = re.findall(
        r"needs\.attempt_budget\.result\s*(==|!=)\s*'([a-z]+)'", condition
    )
    assert sorted(results) == [("==", "skipped"), ("==", "success")]
    assert "always()" not in condition


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


def _fetch_sandbox(tmp_path, *, locks: bool, provisions: bool):
    """A working directory for the fetch step, with a stub `uv` that records any fetch."""
    corpus = tmp_path / "axiom-corpus"
    if locks:
        (corpus / ".axiom" / "corpus-locks").mkdir(parents=True)
    if provisions:
        (corpus / "data" / "corpus" / "provisions").mkdir(parents=True)
    release = tmp_path / "release.json"
    release.write_text(
        '{"content": {"artifacts": [{"artifact_class": "provisions",'
        ' "path": "data/corpus/provisions/us/statute/x.jsonl"}]}}'
    )
    stub = tmp_path / "bin"
    stub.mkdir()
    marker = tmp_path / "fetched"
    (stub / "uv").write_text(f'#!/bin/sh\necho "$@" > "{marker}"\n')
    (stub / "uv").chmod(0o755)
    jq = shutil.which("jq")
    assert jq, "jq is required to run the fetch step"
    path = f"{stub}:{Path(jq).parent}:/usr/bin:/bin"
    return release, marker, path


def _bash4() -> str | None:
    """A bash with mapfile (4+). CI's Ubuntu runners have one; macOS /bin/bash is 3.2."""
    for candidate in (
        shutil.which("bash"),
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
    ):
        if candidate and Path(candidate).exists():
            out = subprocess.run(
                [candidate, "-c", "echo ${BASH_VERSINFO[0]}"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out.isdigit() and int(out) >= 4:
                return candidate
    return None


def _run_fetch(step, tmp_path, release, path, env, bash="bash"):
    return subprocess.run(
        [bash, "-c", step["run"]],
        env={"PATH": path, "RELEASE_OBJECT": str(release), **env},
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_a_locked_pin_stops_before_fetching_without_credentials(tmp_path):
    _, _, step = _step(_workflow(), FETCH_STEP)
    release, marker, path = _fetch_sandbox(tmp_path, locks=True, provisions=False)
    for env in ({}, {"R2_ACCESS_KEY_ID": "id"}, {"R2_SECRET_ACCESS_KEY": "s"}):
        result = _run_fetch(step, tmp_path, release, path, env)
        assert result.returncode == 1
        assert "R2 corpus read credentials" in result.stderr
        assert not marker.exists()


def test_a_locked_pin_with_credentials_fetches_its_provisions(tmp_path):
    bash = _bash4()
    if bash is None:
        pytest.skip("the fetch step needs bash 4+ (mapfile); CI runs it on Ubuntu")
    _, _, step = _step(_workflow(), FETCH_STEP)
    release, marker, path = _fetch_sandbox(tmp_path, locks=True, provisions=False)
    result = _run_fetch(
        step,
        tmp_path,
        release,
        path,
        {"R2_ACCESS_KEY_ID": "id", "R2_SECRET_ACCESS_KEY": "s"},
        bash=bash,
    )
    assert result.returncode == 0, result.stderr
    assert "--path data/corpus/provisions/us/statute/x.jsonl" in marker.read_text()


def test_a_lockless_pin_needs_no_credentials(tmp_path):
    # A git-history pin carries its provisions in the checkout and never reads R2.
    _, _, step = _step(_workflow(), FETCH_STEP)
    release, marker, path = _fetch_sandbox(tmp_path, locks=False, provisions=True)
    result = _run_fetch(step, tmp_path, release, path, {})
    assert result.returncode == 0, result.stderr
    assert "R2 corpus read credentials" not in result.stderr
    assert not marker.exists()


def test_a_pin_with_neither_locks_nor_provisions_fails(tmp_path):
    _, _, step = _step(_workflow(), FETCH_STEP)
    release, marker, path = _fetch_sandbox(tmp_path, locks=False, provisions=False)
    result = _run_fetch(step, tmp_path, release, path, {})
    assert result.returncode == 1
    assert "neither locks nor provisions" in result.stderr
    assert not marker.exists()
