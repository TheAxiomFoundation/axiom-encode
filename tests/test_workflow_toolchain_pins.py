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

Staging a tree with ``uv pip install --target`` (the verification and
axiom-compose runtimes) is out of scope here; those installs are checked
against the lock where they are staged.
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
ACTION_PIN = re.compile(r"[^@\s]+@[0-9a-f]{40}|docker://[^@\s]+@sha256:[0-9a-f]{64}")

# uv options that take a separate value, so the value is not mistaken for the
# subcommand (global options) or for the start of `uv run`'s command.
UV_GLOBAL_VALUED = frozenset(
    {
        "--allow-insecure-host",
        "--cache-dir",
        "--color",
        "--config-file",
        "--directory",
        "--project",
        "--python-preference",
    }
)
UV_VALUED = UV_GLOBAL_VALUED | {
    "-f",
    "-o",
    "-p",
    "-P",
    "-w",
    "--default-index",
    "--env-file",
    "--extra",
    "--extra-index-url",
    "--find-links",
    "--format",
    "--group",
    "--index",
    "--index-url",
    "--no-emit-package",
    "--no-group",
    "--only-group",
    "--output-file",
    "--package",
    "--python",
    "--python-platform",
    "--refresh-package",
    "--reinstall-package",
    "--upgrade-package",
    "--with",
    "--with-editable",
    "--with-requirements",
}
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
# Words that may precede a command without being the command.
COMMAND_PREFIXES = frozenset(
    {"!", "command", "do", "elif", "else", "env", "exec", "if", "sudo", "then"}
    | {"time", "until", "while"}
)
RAW_LOCK_READER = re.compile(
    r"(?:^|[\s(\"'`])uv(?:\s+--?[\w-]+(?:=\S+)?)*\s+(?:sync|run|export|lock)\b"
)
BARE_TOOL = re.compile(r"(?:pytest|ruff|towncrier|python3?|pip3?)(?:\s|$)")


def _logical_lines(script: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", script)
    lines = []
    for line in joined.splitlines():
        # Drop comments: a `#` at the start of a line or after whitespace.
        stripped = re.sub(r"(?:^|\s)#.*$", "", line).strip()
        if stripped:
            lines.append(re.sub(r"\s+", " ", stripped))
    return lines


def _simple_commands(script: str) -> list[str]:
    """Split each line at shell operators and command substitutions."""
    commands = []
    for line in _logical_lines(script):
        for command in re.split(r"&&|\|\||[;|`()]|\$\(", line):
            if command.strip(" \"'"):
                commands.append(command.strip())
    return commands


def _uv_invocation(command: str) -> tuple[str, list[str]] | None:
    """Return (subcommand, option names) when `command` runs uv, else None.

    `uv tool run`-style subcommands come back as "tool run". For `uv run`,
    only the options before the command it runs are returned.
    """
    tokens = [token.strip("\"'") for token in command.split()]
    start = 0
    while start < len(tokens) and (
        tokens[start] in COMMAND_PREFIXES or re.fullmatch(r"\w+=\S*", tokens[start])
    ):
        start += 1
    if start == len(tokens):
        return None
    head, rest = tokens[start], tokens[start + 1 :]
    if head == "uvx" or head.endswith("/uvx"):
        return "uvx", []
    if head != "uv" and not head.endswith("/uv"):
        return None

    options: list[str] = []
    index = 0
    while index < len(rest) and rest[index].startswith("-"):
        name = rest[index].split("=", 1)[0]
        options.append(name)
        takes_value = name in UV_GLOBAL_VALUED and "=" not in rest[index]
        index += 2 if takes_value else 1
    if index == len(rest):
        return None
    subcommand = rest[index]
    index += 1
    if subcommand in {"pip", "python", "tool"} and index < len(rest):
        subcommand = f"{subcommand} {rest[index]}"
        index += 1
    while index < len(rest):
        token = rest[index]
        if not token.startswith("-"):
            if subcommand == "run":
                break
            index += 1
            continue
        name = token.split("=", 1)[0]
        options.append(name)
        index += 2 if name in UV_VALUED and "=" not in token else 1
    return subcommand, options


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


def _uv_invocations():
    for name, job, index, step in _steps():
        for command in _simple_commands(step.get("run", "")):
            invocation = _uv_invocation(command)
            if invocation is not None:
                yield f"{name}:{job}[{index}] {command}", *invocation


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
    invocations = list(_uv_invocations())
    subcommands = {subcommand for _, subcommand, _ in invocations}
    assert {"sync", "run", "export", "lock", "pip install"} <= subcommands
    # The parser sees at least every lock-reading uv command a plain regex does.
    raw = sum(
        len(RAW_LOCK_READER.findall(line))
        for _, _, _, step in _steps()
        for line in _logical_lines(step.get("run", ""))
    )
    parsed = sum(
        1
        for _, subcommand, _ in invocations
        if subcommand in {"sync", "run", "export", "lock"}
    )
    assert raw >= 10
    assert parsed >= raw


def test_uv_invocation_parser():
    cases = {
        "uv sync --locked --python 3.13 --extra dev": (
            "sync",
            ["--locked", "--python", "--extra"],
        ),
        "uv --no-cache sync --python 3.13": ("sync", ["--no-cache", "--python"]),
        "uv --directory x sync --frozen": ("sync", ["--directory", "--frozen"]),
        # Options after `uv run`'s command belong to that command.
        "uv run --frozen --python 3.13 python s.py -P --upgrade": (
            "run",
            ["--frozen", "--python"],
        ),
        "uv run --frozen -w requests python s.py": ("run", ["--frozen", "-w"]),
        "uv sync --frozen -P cryptography": ("sync", ["--frozen", "-P"]),
        "uv tool run ruff": ("tool run", []),
        "uvx ruff": ("uvx", []),
        "sudo env A=1 uv lock --offline": ("lock", ["--offline"]),
        "if ! uv export --locked --no-dev": ("export", ["--locked", "--no-dev"]),
        "uv pip install --target t -r r.txt": ("pip install", ["--target", "-r"]),
        "echo uv sync": None,
        "command -v uv": None,
    }
    for command, expected in cases.items():
        assert _uv_invocation(command) == expected, command
    assert _simple_commands("uv sync --python 3.13 # --frozen") == [
        "uv sync --python 3.13"
    ]
    assert _simple_commands('x="$(uv export --locked | grep y)"') == [
        'x="',
        "uv export --locked",
        "grep y",
    ]


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
    for where, subcommand, options in _uv_invocations():
        if set(options) & LOCK_OVERRIDES:
            offenders.append(where)
        elif subcommand in {"sync", "run", "export"}:
            if not {"--locked", "--frozen"} & set(options):
                offenders.append(where)
        elif subcommand == "lock":
            # `uv lock --offline` (finalize-signed-snap-queue.yml, after a
            # version bump) cannot fetch newer releases; a networked one can.
            if not {"--check", "--locked", "--offline"} & set(options):
                offenders.append(where)
    assert offenders == []


def test_no_workflow_runs_an_unpinned_uv_tool_or_edits_the_lock_inputs():
    offenders = [
        where
        for where, subcommand, _ in _uv_invocations()
        if subcommand in {"add", "remove", "uvx", "tool install", "tool run"}
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
    workflow = yaml.safe_load((WORKFLOWS / workflow_name).read_text(encoding="utf-8"))
    commands = [
        ((step_index, command_index), command)
        for step_index, step in enumerate(workflow["jobs"][job_name]["steps"])
        for command_index, command in enumerate(_simple_commands(step.get("run", "")))
    ]
    uv_commands = [
        (position, command, *invocation)
        for position, command in commands
        if (invocation := _uv_invocation(command)) is not None
    ]

    syncs = [
        (position, command)
        for position, command, subcommand, _ in uv_commands
        if subcommand == "sync"
    ]
    assert len(syncs) == 1, syncs
    sync_position, sync_command = syncs[0]
    assert LOCKED_DEV_SYNC.fullmatch(sync_command), sync_command

    # Nothing else builds or changes a Python environment in these jobs.
    assert [
        command
        for _, command, subcommand, _ in uv_commands
        if subcommand.split()[0] in {"add", "pip", "remove", "tool", "uvx", "venv"}
    ] == []
    assert [
        command for _, command in commands if re.search(r"\bpip3? install\b", command)
    ] == []

    # Every tool the job runs comes from the synced .venv, after the sync.
    venv_uses = [position for position, command in commands if ".venv/bin/" in command]
    assert venv_uses
    assert all(position > sync_position for position in venv_uses), venv_uses
    bare_tools = []
    for _, command in commands:
        words = command.split()
        while words and (words[0] in COMMAND_PREFIXES or "=" in words[0]):
            words.pop(0)
        if words and BARE_TOOL.match(" ".join(words)):
            bare_tools.append(command)
    assert bare_tools == []
