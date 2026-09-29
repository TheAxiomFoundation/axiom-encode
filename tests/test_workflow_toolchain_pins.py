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

Commands are found with ``tests/workflow_shell.py``, which reads each ``run:``
script as bash does and parses uv calls against the pinned uv's option table.
A script it cannot follow fails the suite rather than being skipped.

Out of scope: what ``uv pip install --target`` stages for the verification and
axiom-compose runtimes (the verification tree is locked by
``test_verification_site_packages_lock.py`` once #1712 lands), and
``uv version --bump``, which re-locks without ``--frozen``.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from tests import workflow_shell
from tests.workflow_shell import Command, UnanalyzableScript

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
UV_OPTIONS = workflow_shell.load_uv_options()

# workflow -> jobs whose environment runs the repository's tests or linters.
TEST_ENVIRONMENTS = {
    "ci.yml": {"test", "lint"},
    "signing-supervisor.yml": {"build-and-test"},
}
LOCKED_DEV_SYNCS = {
    ("uv", "sync", "--locked", "--python", python, "--extra", "dev")
    for python in ("3.13", "${{matrix.python-version}}")
}
ACTION_PIN = re.compile(r"[^@\s]+@[0-9a-f]{40}|docker://[^@\s]+@sha256:[0-9a-f]{64}")
# Options that layer unlocked or upgraded packages over the lock even when the
# command also passes --locked or --frozen.
LOCK_OVERRIDES = frozenset(
    {
        "-P",
        "-U",
        "-w",
        "--upgrade",
        "--upgrade-package",
        "--with",
        "--with-editable",
        "--with-requirements",
    }
)
TOOLS = re.compile(
    r"pytest|py\.test|ruff|towncrier|pre-commit|python[0-9.]*|pip[0-9.]*"
)
VENV_TOOL = re.compile(r"(?:\./)?\.venv/bin/\S+")


@functools.cache
def _workflows() -> tuple[tuple[str, dict], ...]:
    return tuple(
        (path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    )


@functools.cache
def _script_commands(script: str) -> tuple[Command, ...]:
    return tuple(workflow_shell.commands(script))


def _steps():
    for name, workflow in _workflows():
        for job_name, job in workflow["jobs"].items():
            for index, step in enumerate(job.get("steps", [])):
                yield name, job_name, index, step


def _action_steps():
    for name, job_name, index, step in _steps():
        if "uses" in step and not step["uses"].startswith("./"):
            yield name, job_name, index, step


def _commands(steps) -> list[tuple[str, Command]]:
    return [
        (f"{name}:{job}[{index}]", command)
        for name, job, index, step in steps
        for command in _script_commands(step.get("run", ""))
    ]


def _uv_invocations(steps=None):
    for where, command in _commands(_steps() if steps is None else steps):
        invocation = workflow_shell.uv_invocation(command, UV_OPTIONS)
        if invocation is not None:
            yield where, invocation


def _job_steps(workflow_name: str, job_name: str):
    workflow = dict(_workflows())[workflow_name]
    for index, step in enumerate(workflow["jobs"][job_name]["steps"]):
        yield workflow_name, job_name, index, step


def _uv(script: str) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (invocation.subcommand, invocation.options)
        for command in workflow_shell.commands(script)
        if (invocation := workflow_shell.uv_invocation(command, UV_OPTIONS))
    ]


def _programs(script: str) -> list[str]:
    return [command.program for command in workflow_shell.commands(script)]


def test_every_run_script_is_analyzable():
    unanalyzable = []
    for name, job, index, step in _steps():
        try:
            for command in _script_commands(step.get("run", "")):
                workflow_shell.uv_invocation(command, UV_OPTIONS)
        except UnanalyzableScript as error:
            unanalyzable.append(f"{name}:{job}[{index}] {error}")
    assert unanalyzable == []


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
    assert len(_commands(_steps())) >= 1000
    found = {
        (where.split("[")[0], invocation.subcommand)
        for where, invocation in _uv_invocations()
    }
    assert {
        ("ci.yml:test", "sync"),
        ("ci.yml:lint", "sync"),
        ("signing-supervisor.yml:build-and-test", "sync"),
        ("prepare-all-state-snap-queue.yml:prepare", "run"),
        ("finalize-signed-snap-queue.yml:finalize", "lock"),
        ("targeted-signed-reencode.yml:encode", "export"),
        ("bulk-encode.yml:preclassify", "pip install"),
    } <= found


# script -> the uv calls it makes, as (subcommand, canonical options).
READER_CASES = {
    "uv sync --locked --python 3.13 --extra dev": [
        ("sync", ("--locked", "--python", "--extra"))
    ],
    # A global option before the subcommand.
    "uv --no-cache sync --python 3.13": [("sync", ("--no-cache", "--python"))],
    # Options after `uv run`'s command belong to that command.
    "uv run --frozen --python 3.13 python s.py -P --upgrade": [
        ("run", ("--frozen", "--python"))
    ],
    # A valued option is not mistaken for uv run's command.
    "uv run --frozen --link-mode copy --with requests python s.py": [
        ("run", ("--frozen", "--link-mode", "--with"))
    ],
    'uv run --frozen --env-file "my settings.env" --with requests python s.py': [
        ("run", ("--frozen", "--env-file", "--with"))
    ],
    # Attached values and clustered short flags.
    "uv run --frozen -wrequests python s.py": [("run", ("--frozen", "-w"))],
    "uv sync --frozen -Pcryptography": [("sync", ("--frozen", "-P"))],
    "uv sync --frozen -qU": [("sync", ("--frozen", "-q", "-U"))],
    "uv sync --frozen --upgrade-package=ruff": [
        ("sync", ("--frozen", "--upgrade-package"))
    ],
    # A hidden alias resolves to its option.
    "uv pip install --target t --requirement r.txt": [
        ("pip install", ("--target", "--requirements"))
    ],
    # Comments, including a `#` inside quotes.
    "uv sync --python 3.13  # --frozen": [("sync", ("--python",))],
    'echo "step #1"; uv sync --python 3.13': [("sync", ("--python",))],
    'echo "documentation # text"; uv tool run ruff': [("tool run", ())],
    # Substitutions, wrappers and nested shells run their commands.
    'x="$(uv export --no-dev | grep y)"': [("export", ("--no-dev",))],
    "cat <(uv export --locked)": [("export", ("--locked",))],
    "`uv lock`": [("lock", ())],
    "timeout -k 5 600 uv sync": [("sync", ())],
    "sudo -E -u runner env -i PATH=/bin A=1 uv lock --offline": [
        ("lock", ("--offline",))
    ],
    "nice -n 5 nohup uv sync": [("sync", ())],
    "bash -c 'uv pip sync requirements.txt'": [("pip sync", ())],
    'sh -ec "uv sync --frozen"': [("sync", ("--frozen",))],
    "printf '%s\\n' r.txt | xargs -n 1 uv pip sync": [("pip sync", ())],
    "{ uv sync; }": [("sync", ())],
    "f() { uv sync; }": [("sync", ())],
    "if ! uv export --locked --no-dev; then exit 1; fi": [
        ("export", ("--locked", "--no-dev"))
    ],
    "uvx ruff": [("uvx", ())],
    # Text that mentions uv runs nothing.
    "echo uv sync": [],
    "printf '%s\\n' 'uv sync'": [],
    'echo "example; uv tool run ruff"': [],
    "command -v uv": [],
    "echo $(( 1 + 2 ))": [],
    "cat <<'DOC'\nuv sync\nDOC\necho done": [],
    # An unquoted heredoc body is data, but its substitutions run.
    "cat <<DOC\nuv sync\n$(uv lock)\nDOC": [("lock", ())],
    "a=1 \\\n  uv sync --frozen": [("sync", ("--frozen",))],
}


@pytest.mark.parametrize(("script", "expected"), READER_CASES.items())
def test_reader_finds_the_uv_calls_a_script_makes(script, expected):
    assert _uv(script) == expected


@pytest.mark.parametrize(
    "script",
    [
        "uv sync --frozen --no-such-option",
        "uv run --frozen --no-such-option value python s.py",
        "uv sync -Z",
        "uv cache clean",
        "echo 'unterminated",
        "x=$(uv sync",
        "cat <<EOF\nno terminator",
        "bash -c",
    ],
)
def test_reader_fails_closed_on_what_it_cannot_follow(script):
    with pytest.raises(UnanalyzableScript):
        _uv(script)


def test_reader_resolves_the_program_behind_quotes_and_wrappers():
    assert _programs("'pytest' tests/") == ["pytest"]
    assert _programs("bash -c 'pytest tests/'") == ["pytest"]
    assert _programs("cd tests && sudo -E /usr/bin/python3 -m pytest") == [
        "cd",
        "python3",
    ]
    assert _programs("cmd 2>&1 | tee log >/dev/null") == ["cmd", "tee"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_uv_option_table_matches_the_pinned_uv():
    installed = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[1]
    if installed != UV_OPTIONS["uv_version"]:
        pytest.skip(f"uv {installed} is installed; the table is for the pinned uv")
    assert workflow_shell.uv_option_table() == UV_OPTIONS


def test_uv_option_table_is_for_the_uv_every_workflow_pins():
    versions = {
        (step.get("with") or {}).get("version")
        for _, _, _, step in _action_steps()
        if step["uses"].startswith("astral-sh/setup-uv@")
    }
    assert versions == {UV_OPTIONS["uv_version"]}


def test_every_action_is_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job}[{index}] {step['uses']}"
        for name, job, index, step in _action_steps()
        if not ACTION_PIN.fullmatch(step["uses"])
    ]
    assert unpinned == []


def test_reusable_workflow_calls_are_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job_name} {job['uses']}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        if "uses" in job
        and not job["uses"].startswith("./")
        and not ACTION_PIN.fullmatch(job["uses"])
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


def test_every_uv_sync_run_export_and_lock_stays_on_the_lock():
    offenders = []
    for where, invocation in _uv_invocations():
        options = set(invocation.options)
        text = f"{where} {invocation.command.line}"
        if options & LOCK_OVERRIDES:
            offenders.append(text)
        elif invocation.subcommand in {"sync", "run", "export"}:
            if not options & {"--locked", "--frozen"}:
                offenders.append(text)
        elif invocation.subcommand == "lock":
            # `uv lock` without these may rewrite uv.lock from the network.
            if not options & {"--check", "--dry-run", "--offline"}:
                offenders.append(text)
    assert offenders == []


def test_no_workflow_runs_an_unpinned_uv_tool_or_edits_the_dependencies():
    offenders = [
        f"{where} {invocation.command.line}"
        for where, invocation in _uv_invocations()
        if invocation.subcommand in {"add", "remove", "uvx", "tool install", "tool run"}
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    sorted(
        (workflow_name, job_name)
        for workflow_name, jobs in TEST_ENVIRONMENTS.items()
        for job_name in jobs
    ),
)
def test_test_environment_is_the_locked_dev_set(workflow_name, job_name):
    commands = [
        command for _, command in _commands(_job_steps(workflow_name, job_name))
    ]
    invocations = [
        (position, invocation)
        for position, command in enumerate(commands)
        if (invocation := workflow_shell.uv_invocation(command, UV_OPTIONS))
    ]

    syncs = [
        (position, invocation.command.words)
        for position, invocation in invocations
        if invocation.subcommand == "sync"
    ]
    assert len(syncs) == 1, syncs
    sync_position, sync_words = syncs[0]
    assert sync_words in LOCKED_DEV_SYNCS, sync_words

    # Nothing else builds or changes a Python environment in these jobs.
    assert [
        invocation.command.line
        for _, invocation in invocations
        if invocation.subcommand.split()[0]
        in {"add", "pip", "remove", "tool", "uvx", "venv"}
    ] == []
    assert [
        command.line
        for command in commands
        if re.fullmatch(r"python[0-9.]*", command.program)
        and re.search(r"(?:^| )-m ?pip (?:install|download)\b", " ".join(command.words))
    ] == []

    # Every tool the job runs comes from the synced .venv, after the sync.
    venv_uses = [
        position
        for position, command in enumerate(commands)
        if VENV_TOOL.fullmatch(command.words[0])
    ]
    assert venv_uses
    assert all(position > sync_position for position in venv_uses), venv_uses
    assert [
        command.line
        for command in commands
        if TOOLS.fullmatch(command.program)
        and not VENV_TOOL.fullmatch(command.words[0])
    ] == []
