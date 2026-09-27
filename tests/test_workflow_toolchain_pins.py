"""Every workflow's toolchain comes from a pin, and the test suite runs on the lock.

``ci.yml`` and ``signing-supervisor.yml`` used to build the test environment
with ``uv venv`` plus ``uv pip install -e ".[dev]"``, which ignores ``uv.lock``
and floats every ``>=`` dependency to the index's latest release. CI run
36166075448 (2026-09-25) ran the suite on sqlite-utils 4.2.1, cryptography
50.0.1 and supabase 2.31.0 while the lock, and so every production
``uv sync --locked``, pinned 3.39, 46.0.3 and 2.27.3. The lint job installed
the latest ruff rather than the locked one. ``ci.yml`` also tag-pinned its
actions, and five ``setup-uv`` steps in four workflows had no ``version`` or
pinned another release. These tests fail if any of that comes back.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# workflow -> jobs whose environment runs the repository's tests or linters.
TEST_ENVIRONMENTS = {
    "ci.yml": {"test", "lint"},
    "signing-supervisor.yml": {"build-and-test"},
}
LOCKED_DEV_SYNC = re.compile(
    r"uv sync --locked --python (?:3\.13|\$\{\{ matrix\.python-version \}\})"
    r" --extra dev"
)
# Commands that build a Python environment without reading uv.lock.
UNLOCKED_INSTALLERS = re.compile(r"\b(?:uv (?:pip|venv|tool|add)|uvx|pip3? install)\b")


def _logical_lines(script: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", script)
    lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(re.sub(r"\s+", " ", stripped))
    return lines


def _workflows() -> list[tuple[str, dict]]:
    return [
        (path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    ]


def _steps():
    for name, workflow in _workflows():
        for job_name, job in workflow["jobs"].items():
            for index, step in enumerate(job.get("steps", [])):
                yield name, job_name, index, step


def _action_steps():
    for name, job_name, index, step in _steps():
        if "uses" in step and not step["uses"].startswith("./"):
            yield name, job_name, index, step


def test_discovery_is_not_vacuous():
    names = {name for name, _ in _workflows()}
    assert set(TEST_ENVIRONMENTS) <= names
    assert len(names) >= 10
    assert sum(1 for _ in _action_steps()) >= 50
    setup_uv = [
        step
        for _, _, _, step in _action_steps()
        if step["uses"].startswith("astral-sh/setup-uv@")
    ]
    assert len(setup_uv) >= 8


def test_every_action_is_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job}[{index}] {step['uses']}"
        for name, job, index, step in _action_steps()
        if not re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", step["uses"])
    ]
    assert unpinned == []


def test_reusable_workflow_calls_are_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job_name} {job['uses']}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        if "uses" in job
        and not job["uses"].startswith("./")
        and not re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", job["uses"])
    ]
    assert unpinned == []


def test_each_action_resolves_to_one_commit_across_workflows():
    commits: dict[str, set[str]] = defaultdict(set)
    for _, _, _, step in _action_steps():
        action, _, ref = step["uses"].partition("@")
        commits[action].add(ref)
    assert {action: refs for action, refs in commits.items() if len(refs) > 1} == {}


def test_every_setup_uv_step_pins_the_same_exact_uv_release():
    versions: dict[str, str | None] = {}
    for name, job, index, step in _action_steps():
        if step["uses"].startswith("astral-sh/setup-uv@"):
            version = (step.get("with") or {}).get("version")
            versions[f"{name}:{job}[{index}]"] = version
    assert {
        where: version
        for where, version in versions.items()
        if not (isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version))
    } == {}
    assert len(set(versions.values())) == 1, versions


def test_every_uv_sync_and_run_reads_the_lock():
    unlocked = []
    for name, job, index, step in _steps():
        for line in _logical_lines(step.get("run", "")):
            for command in re.split(r"&&|\|\||;|\|", line):
                if re.search(r"\buv (?:sync|run)\b", command) and not re.search(
                    r"--(?:locked|frozen)\b", command
                ):
                    unlocked.append(f"{name}:{job}[{index}] {command.strip()}")
    assert unlocked == []


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    sorted(
        (workflow_name, job_name)
        for workflow_name, jobs in TEST_ENVIRONMENTS.items()
        for job_name in jobs
    ),
)
def test_test_environment_is_the_locked_dev_set(workflow_name, job_name):
    workflow = yaml.safe_load((WORKFLOWS / workflow_name).read_text(encoding="utf-8"))
    steps = workflow["jobs"][job_name]["steps"]
    lines = [
        (index, line)
        for index, step in enumerate(steps)
        for line in _logical_lines(step.get("run", ""))
    ]

    syncs = [(index, line) for index, line in lines if re.search(r"\buv sync\b", line)]
    assert len(syncs) == 1, syncs
    sync_index, sync_line = syncs[0]
    assert LOCKED_DEV_SYNC.fullmatch(sync_line), sync_line

    assert [line for _, line in lines if UNLOCKED_INSTALLERS.search(line)] == []

    # Every tool the job runs comes from the synced .venv, after the sync.
    venv_uses = [(index, line) for index, line in lines if ".venv/bin/" in line]
    assert venv_uses
    assert all(index > sync_index for index, _ in venv_uses), venv_uses
    bare_tools = [
        line
        for _, line in lines
        if re.match(r"(?:sudo )?(?:pytest|ruff|towncrier|python3?)\b", line)
    ]
    assert bare_tools == []


def test_no_workflow_builds_an_environment_outside_the_lock_with_uv():
    offenders = [
        f"{name}:{job}[{index}] {line}"
        for name, job, index, step in _steps()
        for line in _logical_lines(step.get("run", ""))
        if re.search(r"\buv (?:venv|tool install|add)\b|\buvx\b", line)
    ]
    assert offenders == []
