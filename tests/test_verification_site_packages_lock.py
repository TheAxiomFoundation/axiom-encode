"""Every trusted runtime's site-packages must be exactly a uv lock.

``scripts/provision_verification_supervisor.py`` copies a caller-staged
``--site-packages`` tree into the signer's trusted Python import root. A
resolving ``uv pip install --target ... ".[api]"`` ignores ``uv.lock`` and
floats every ``>=`` dependency to the index's latest at run time: targeted run
36174590343 logged ``claude-agent-sdk==0.1.47`` in the locked ``.venv`` and
``claude-agent-sdk==0.2.159`` in the verification tree of the same job.

Commands are found with ``tests/workflow_shell.py``, so ``pip3``,
``python -m pip``, ``sudo``/``env``/``bash -c``/``eval`` wrappers, attached
``-tDIR`` and ``--target=DIR`` are all seen. Every ``--target`` install in a
workflow (except the provisioner's selftest) must be a ``uv pip install
--no-deps`` of either a hash-pinned ``uv export --locked`` output written in the
same step (with ``--require-hashes``), the commit-pinned ``axiom-oracles`` git
requirement, or the project itself. That covers the verification tree and the
axiom-compose runtime ``targeted-signed-reencode.yml`` stages from
axiom-compose's lock. Every production provision must consume either that
staged tree or the purelib of a ``uv sync --locked``/``--frozen`` environment
nothing else installs into. The staging steps are then executed against a fake
``uv`` to prove their fail-closed paths.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from tests import workflow_shell
from tests.workflow_shell import Command

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
UV_OPTIONS = workflow_shell.load_uv_options()
PIP = re.compile(r"pip[0-9.]*")
PYTHON = re.compile(r"python[0-9.]*t?")
PROVISIONER = "provision_verification_supervisor.py"

# The provisioner's own selftest stages a throwaway tree (pyyaml plus hostile
# startup-hook fixtures) under a stand-in supervisor with test-only trust roots;
# nothing it provisions verifies or signs anything.
EXEMPT_WORKFLOWS = frozenset({"provision-selftest.yml"})

SITE_PACKAGES = "$RUNNER_TEMP/axiom-verification-site-packages"
ORACLES_PATTERN = (
    r"axiom-oracles @ git\+https://github\.com/TheAxiomFoundation/"
    r"axiom-oracles@[0-9a-f]{40}"
)
LOCKED_GIT_LINE = (
    "axiom-oracles @ git+https://github.com/TheAxiomFoundation/"
    "axiom-oracles@e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
)

# workflow -> (interpreter the staged tree is installed for, extra flag, sync)
STAGING = {
    "bulk-encode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "golden-regeneration.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "targeted-signed-reencode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "signed-apply-reusable.yml": (
        '"$PYBIN"',
        "",
        'uv sync --locked --no-dev --python "$PYBIN"',
    ),
}
# Non-vacuity: every --target install the discovery must see. Each staging
# workflow has three (locked set, git requirement, project); targeted also
# stages the axiom-compose runtime.
EXPECTED_TARGET_INSTALLS = {
    "bulk-encode.yml": 3,
    "golden-regeneration.yml": 3,
    "signed-apply-reusable.yml": 3,
    "targeted-signed-reencode.yml": 4,
}
# Provisions whose --site-packages is the purelib of a locked project
# environment, keyed by workflow -> that environment's interpreter.
LOCKED_VENV_PROVISIONS = {
    "issue233-canada-source-diagnostic.yml": (
        "$GITHUB_WORKSPACE/axiom-encode/.venv/bin/python"
    ),
}
PURELIB = (
    'site="$($python_path -c '
    '\'import sysconfig; print(sysconfig.get_paths()["purelib"])\')"'
)


def _logical_lines(script: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", script)
    lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(re.sub(r"\s+", " ", stripped))
    return lines


def _load_workflows() -> dict[str, dict]:
    return {
        path.name: yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for path in sorted(WORKFLOWS.glob("*.yml"))
    }


def _steps(workflows: dict[str, dict]):
    for name, workflow in workflows.items():
        for job_name, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps", [])):
                if "run" in step:
                    yield name, job_name, index, step


def _commands(script: str) -> list[Command]:
    return workflow_shell.expand(workflow_shell.commands(script), UV_OPTIONS)


def _runs_pip(command: Command) -> bool:
    """Whether ``command`` runs pip: directly, as ``python -m pip``, or as
    ``"$PYBIN" -m pip`` (an interpreter named by a variable, the form the
    signed-apply workflow used to stage its tree unlocked)."""
    if not command.words:
        return False
    if PIP.fullmatch(command.program):
        return True
    return bool(
        (PYTHON.fullmatch(command.program) or command.words[0].startswith("$"))
        and PIP.fullmatch(workflow_shell.python_module(command.words) or "")
    )


def _pip_names_a_target(words: tuple[str, ...]) -> bool:
    """``-t DIR``, ``-tDIR``, a short cluster such as ``-qt DIR``, or ``--target``.

    A short cluster that merely contains ``t`` in an attached value
    (``-rrequirements.txt``) also counts; for pip that only fails closed.
    """
    return any(
        word == "--target"
        or word.startswith("--target=")
        or (word.startswith("-") and not word.startswith("--") and "t" in word[1:])
        for word in words
    )


def _uv_arguments(
    command: Command,
) -> tuple[str, dict[str, list[str]], list[str]] | None:
    """A uv call as (subcommand, {canonical option: values}, positionals)."""
    invocation = workflow_shell.uv_invocation(command, UV_OPTIONS)
    if invocation is None or not invocation.subcommand:
        return None
    table = UV_OPTIONS["commands"][invocation.subcommand]
    words = list(command.words[1:])
    sub = invocation.subcommand.split()
    index = next(i for i in range(len(words)) if words[i : i + len(sub)] == sub)
    index += len(sub)
    values: dict[str, list[str]] = defaultdict(list)
    positionals: list[str] = []
    while index < len(words):
        arg = words[index]
        if arg == "--":
            positionals.extend(words[index + 1 :])
            break
        if arg.startswith("--"):
            name, has_value, value = arg.partition("=")
            name = table.get("aliases", {}).get(name, name)
            if name in table["valued"] and not has_value:
                index += 1
                value = words[index]
            values[name].append(value)
        elif arg.startswith("-") and arg != "-":
            for offset, letter in enumerate(arg[1:], start=1):
                name = "-" + letter
                canonical = table.get("short", {}).get(name, name)
                if name in table["valued"]:
                    value = arg[offset + 1 :]
                    if not value:
                        index += 1
                        value = words[index]
                    values[canonical].append(value)
                    break
                values[canonical].append("")
        else:
            positionals.append(arg)
        index += 1
    return invocation.subcommand, dict(values), positionals


def _locked_exports(commands: list[Command]) -> set[str]:
    """Output files of hash-pinned ``uv export --locked``/``--frozen`` calls."""
    exported = set()
    for command in commands:
        parsed = _uv_arguments(command)
        if parsed is None or parsed[0] != "export":
            continue
        _, values, _ = parsed
        if (
            ("--locked" in values or "--frozen" in values)
            and "--no-hashes" not in values
            and "--output-file" in values
        ):
            exported.add(values["--output-file"][-1])
    return exported


def _target_install_violations(
    workflows: dict[str, dict], exempt: frozenset[str] = EXEMPT_WORKFLOWS
) -> tuple[list[str], dict[str, int]]:
    violations: list[str] = []
    found: dict[str, int] = defaultdict(int)
    for name, job, index, step in _steps(workflows):
        if name in exempt:
            continue
        where = f"{name}:{job}[{index}]"
        commands = _commands(step["run"])
        exported = _locked_exports(commands)
        for command in commands:
            if _runs_pip(command) and "install" in command.words:
                install = command.words.index("install")
                if _pip_names_a_target(command.words[install + 1 :]):
                    found[name] += 1
                    violations.append(f"{where}: pip ignores uv.lock: {command.line}")
                continue
            parsed = _uv_arguments(command)
            if (
                parsed is None
                or parsed[0] not in {"pip install", "pip sync"}
                or "--target" not in parsed[1]
            ):
                continue
            found[name] += 1
            subcommand, values, positionals = parsed
            problem = None
            requirements = values.get("--requirements", [])
            if subcommand != "pip install":
                problem = f"`uv {subcommand} --target` is not a locked install"
            elif "--no-deps" not in values:
                problem = "resolves dependencies (no --no-deps)"
            elif "--editable" in values or values.keys() & {
                "--constraints",
                "--overrides",
            }:
                problem = "installs outside the exported requirement set"
            elif requirements:
                if positionals:
                    problem = f"mixes {positionals} with requirement files"
                elif "--require-hashes" not in values:
                    problem = "installs a requirement file without --require-hashes"
                elif not set(requirements) <= exported:
                    problem = (
                        f"{requirements} is not a hash-pinned `uv export --locked` "
                        "output written in this step"
                    )
            elif positionals == ["$oracles"]:
                if f"grep -xE '{ORACLES_PATTERN}'" not in step["run"]:
                    problem = "$oracles is not the commit-pinned axiom-oracles line"
            elif positionals != ["."]:
                problem = f"installs unpinned requirements {positionals}"
            if problem:
                violations.append(f"{where}: {problem}: {command.line}")
    return violations, dict(found)


def _option_value(words: tuple[str, ...], option: str) -> str | None:
    for index, word in enumerate(words):
        if word == option and index + 1 < len(words):
            return words[index + 1]
        if word.startswith(option + "="):
            return word.split("=", 1)[1]
    return None


def _installs_anything(commands: list[Command]) -> bool:
    for command in commands:
        if _runs_pip(command) and "install" in command.words:
            return True
        parsed = _uv_arguments(command)
        if parsed is not None and parsed[0] in {"pip install", "pip sync", "add"}:
            return True
    return False


def _provision_violations(workflows: dict[str, dict]) -> tuple[list[str], int]:
    violations: list[str] = []
    provisions = 0
    steps = list(_steps(workflows))
    for name, job, index, step in steps:
        if name in EXEMPT_WORKFLOWS:
            continue
        run = step["run"]
        for command in _commands(run):
            if not any(word.endswith(PROVISIONER) for word in command.words):
                continue
            provisions += 1
            where = f"{name}:{job}[{index}]"
            site = _option_value(command.words, "--site-packages")
            job_steps = [s for n, j, i, s in steps if n == name and j == job]
            earlier = [
                s for n, j, i, s in steps if n == name and j == job and i < index
            ]
            if site in {SITE_PACKAGES, "$staged"}:
                if site == "$staged" and f'staged="{SITE_PACKAGES}"' not in run:
                    violations.append(f"{where}: $staged is not {SITE_PACKAGES}")
                if not any(
                    f'site_packages="{SITE_PACKAGES}"' in s["run"] for s in earlier
                ):
                    violations.append(f"{where}: no earlier locked staging step")
            elif site == "$site" and name in LOCKED_VENV_PROVISIONS:
                python = LOCKED_VENV_PROVISIONS[name]
                synced = [
                    parsed
                    for s in earlier
                    for c in _commands(s["run"])
                    if (parsed := _uv_arguments(c)) is not None
                    and parsed[0] == "sync"
                    and parsed[1].keys() & {"--locked", "--frozen"}
                ]
                if (
                    f'python_path="{python}"' not in run
                    or PURELIB not in _logical_lines(run)
                    or command.words[0] != "$python_path"
                ):
                    violations.append(f"{where}: $site is not {python}'s purelib")
                if not synced:
                    violations.append(f"{where}: no `uv sync --locked|--frozen`")
                if any(_installs_anything(_commands(s["run"])) for s in job_steps):
                    violations.append(f"{where}: the job installs into the tree")
            else:
                violations.append(f"{where}: --site-packages {site} is not locked")
    return violations, provisions


def _staging_step(workflow: str, workflows: dict[str, dict] | None = None) -> dict:
    steps = [
        step
        for name, _job, _index, step in _steps(workflows or _load_workflows())
        if name == workflow and f'site_packages="{SITE_PACKAGES}"' in step["run"]
    ]
    assert len(steps) == 1, f"{workflow}: expected one staging step, got {steps}"
    return steps[0]


def _provision_step(workflow: str, workflows: dict[str, dict]) -> dict:
    steps = [
        step
        for name, _job, _index, step in _steps(workflows)
        if name == workflow and PROVISIONER in step["run"]
    ]
    assert len(steps) == 1, f"{workflow}: expected one provision step, got {steps}"
    return steps[0]


def _canonical_staging(python: str, extra: str, sync: str) -> list[str]:
    extra = f" {extra}" if extra else ""
    install = f'uv pip install --python {python} --target "$site_packages"'
    return [
        "set -euo pipefail",
        sync,
        f'site_packages="{SITE_PACKAGES}"',
        'locked="$RUNNER_TEMP/axiom-verification-locked-requirements.txt"',
        'rm -rf "$site_packages"',
        f"uv export --locked --no-dev{extra} --no-emit-project"
        " --no-emit-package axiom-oracles --format requirements-txt"
        ' --output-file "$locked"',
        f'oracles="$(uv export --locked --no-dev{extra} --no-emit-project'
        " --no-hashes --no-annotate --no-header --format requirements-txt"
        f" | grep -xE '{ORACLES_PATTERN}')\"",
        f'{install} --require-hashes --no-deps -r "$locked"',
        f'{install} --no-deps "$oracles"',
        f"{install} --no-deps .",
        "uv pip freeze --python .venv/bin/python --exclude-editable"
        ' > "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        f'uv pip freeze --python {python} --target "$site_packages"'
        ' --exclude axiom-encode > "$RUNNER_TEMP/axiom-verification-staged.freeze"',
        'test -s "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        'diff -u "$RUNNER_TEMP/axiom-verification-locked.freeze"'
        ' "$RUNNER_TEMP/axiom-verification-staged.freeze"',
    ]


def test_every_workflow_target_install_is_locked():
    violations, found = _target_install_violations(_load_workflows())

    assert violations == []
    assert found == EXPECTED_TARGET_INSTALLS


def test_the_selftest_exemption_is_still_needed():
    workflows = _load_workflows()
    for workflow in EXEMPT_WORKFLOWS:
        violations, _ = _target_install_violations(
            {workflow: workflows[workflow]}, exempt=frozenset()
        )
        assert violations, f"{workflow} no longer needs its exemption"


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_is_exactly_the_locked_install(workflow):
    lines = _logical_lines(_staging_step(workflow)["run"])

    if workflow == "signed-apply-reusable.yml":
        # The interpreter the tree is installed for must be the toolcache
        # python the provisioner copies.
        assert lines[0] == "set -euo pipefail"
        assert lines[1] == 'case "$PYBIN" in'
        del lines[1 : lines.index("esac") + 1]
    assert lines == _canonical_staging(*STAGING[workflow])


def test_every_production_provision_consumes_a_locked_tree():
    violations, provisions = _provision_violations(_load_workflows())

    assert violations == []
    assert provisions == len(STAGING) + len(LOCKED_VENV_PROVISIONS)


# Unlocked installs a later edit could add after staging. Each must be caught.
UNLOCKED_INSTALLS = {
    "pip3": f'pip3 install --upgrade --target "{SITE_PACKAGES}" ".[api]"',
    "python-m-pip-attached-t": f'python -m pip install -t"{SITE_PACKAGES}" .',
    "versioned-python-target-eq": "python3.13 -m pip install --target=/opt/x requests",
    "pip-short-cluster": f'pip install -qt "{SITE_PACKAGES}" requests',
    "uv-resolving": f'uv pip install --target "{SITE_PACKAGES}" ".[api]"',
    "uv-attached-t-unpinned": f'uv pip install -t"{SITE_PACKAGES}" --no-deps requests',
    "uv-requirements-without-hashes": (
        f'sudo uv pip install --target "{SITE_PACKAGES}" --no-deps -r "$locked"'
    ),
    "uv-unexported-requirements": (
        "bash -c 'uv pip install --target=/x --no-deps --require-hashes "
        "-r /tmp/handwritten.txt'"
    ),
    "uv-editable": f'uv pip install --target "{SITE_PACKAGES}" --no-deps -e .',
    "uv-pip-sync": f'uv pip sync --target "{SITE_PACKAGES}" requirements.txt',
    "eval-pip": f"eval 'pip install --target \"{SITE_PACKAGES}\" requests'",
    "variable-interpreter": f'"$PYBIN" -m pip install --quiet --target "{SITE_PACKAGES}" .',
}


@pytest.mark.parametrize("workflow", sorted(STAGING))
@pytest.mark.parametrize("mutation", sorted(UNLOCKED_INSTALLS))
@pytest.mark.parametrize("placement", ["end-of-staging", "start-of-provision"])
def test_an_unlocked_install_after_staging_is_rejected(workflow, mutation, placement):
    workflows = copy.deepcopy(_load_workflows())
    line = UNLOCKED_INSTALLS[mutation]
    if placement == "end-of-staging":
        step = _staging_step(workflow, workflows)
        step["run"] = f"{step['run']}{line}\n"
    else:
        step = _provision_step(workflow, workflows)
        step["run"] = f"{line}\n{step['run']}"

    baseline, _ = _target_install_violations(_load_workflows())
    violations, _ = _target_install_violations(workflows)

    assert set(violations) - set(baseline), f"{line!r} was not caught"


def test_the_compose_runtime_install_must_stay_hash_required():
    workflows = copy.deepcopy(_load_workflows())
    targeted = workflows["targeted-signed-reencode.yml"]
    steps = [
        step
        for job in targeted["jobs"].values()
        for step in job["steps"]
        if "axiom-compose-site-packages" in step.get("run", "")
        and "uv export" in step["run"]
    ]
    assert len(steps) == 1
    assert "--require-hashes --no-deps" in steps[0]["run"]
    steps[0]["run"] = steps[0]["run"].replace("--require-hashes --no-deps ", "")

    violations, _ = _target_install_violations(workflows)

    assert any("axiom-compose-site-packages" in v for v in violations), violations


@pytest.mark.parametrize(
    "mutation",
    [
        'uv pip install --python "$python_path" requests',
        'pip install --prefix "$GITHUB_WORKSPACE/axiom-encode/.venv" requests',
        "(cd axiom-encode && uv add requests)",
    ],
)
def test_a_locked_venv_provision_rejects_any_install_into_the_job(mutation):
    workflows = copy.deepcopy(_load_workflows())
    name = next(iter(LOCKED_VENV_PROVISIONS))
    step = _provision_step(name, workflows)
    step["run"] = f"{mutation}\n{step['run']}"

    violations, _ = _provision_violations(workflows)

    assert any("installs into the tree" in v for v in violations), violations


def test_a_provision_of_an_unlocked_tree_is_rejected():
    workflows = copy.deepcopy(_load_workflows())
    step = _provision_step("bulk-encode.yml", workflows)
    step["run"] = step["run"].replace(
        f'--site-packages "{SITE_PACKAGES}"', '--site-packages "$RUNNER_TEMP/other"'
    )

    violations, _ = _provision_violations(workflows)

    assert any("$RUNNER_TEMP/other is not locked" in v for v in violations)


FAKE_UV = """\
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(argv) + "\\n")
if argv[:1] == ["export"]:
    if os.environ.get("FAKE_EXPORT_FAIL"):
        sys.exit(2)
    if "--no-hashes" in argv:
        sys.stdout.write(os.environ["FAKE_UNHASHED_EXPORT"])
    else:
        target = argv[argv.index("--output-file") + 1]
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("requests==2.32.5 \\\\\\n    --hash=sha256:" + "0" * 64 + "\\n")
elif argv[:2] == ["pip", "freeze"]:
    key = "FAKE_STAGED_FREEZE" if "--target" in argv else "FAKE_LOCKED_FREEZE"
    sys.stdout.write(os.environ[key])
"""

LOCKED_FREEZE = f"claude-agent-sdk==0.1.47\n{LOCKED_GIT_LINE}\nrequests==2.32.5\n"


def _run_staging_step(
    tmp_path: Path,
    workflow: str,
    *,
    unhashed_export: str | None = None,
    staged_freeze: str = LOCKED_FREEZE,
    export_fails: bool = False,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(f"#!{sys.executable}\n{FAKE_UV}", encoding="utf-8")
    fake_uv.chmod(0o755)
    toolcache = tmp_path / "toolcache"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    log = tmp_path / "uv-argv.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "RUNNER_TOOL_CACHE": str(toolcache),
        "PYBIN": f"{toolcache}/Python/3.14.4/x64/bin/python",
        "FAKE_UV_LOG": str(log),
        "FAKE_UNHASHED_EXPORT": (
            unhashed_export
            if unhashed_export is not None
            else f"claude-agent-sdk==0.1.47\n{LOCKED_GIT_LINE}\nrequests==2.32.5\n"
        ),
        "FAKE_LOCKED_FREEZE": LOCKED_FREEZE,
        "FAKE_STAGED_FREEZE": staged_freeze,
    }
    if export_fails:
        env["FAKE_EXPORT_FAIL"] = "1"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", _staging_step(workflow)["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls, runner_temp


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_runs_the_locked_sequence(tmp_path, workflow):
    result, calls, runner_temp = _run_staging_step(tmp_path, workflow)

    assert result.returncode == 0, result.stderr
    python, extra, sync = STAGING[workflow]
    python = python.replace(
        '"$PYBIN"', f"{tmp_path}/toolcache/Python/3.14.4/x64/bin/python"
    )
    extra_args = extra.split()
    site_packages = str(runner_temp / "axiom-verification-site-packages")
    locked = str(runner_temp / "axiom-verification-locked-requirements.txt")
    install = ["pip", "install", "--python", python, "--target", site_packages]
    export = ["export", "--locked", "--no-dev", *extra_args, "--no-emit-project"]
    sync_argv = sync.split()[1:]
    if "$PYBIN" in sync:
        sync_argv[sync_argv.index('"$PYBIN"')] = python
    assert calls == [
        sync_argv,
        [
            *export,
            "--no-emit-package",
            "axiom-oracles",
            "--format",
            "requirements-txt",
            "--output-file",
            locked,
        ],
        [
            *export,
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements-txt",
        ],
        [*install, "--require-hashes", "--no-deps", "-r", locked],
        [*install, "--no-deps", LOCKED_GIT_LINE],
        [*install, "--no-deps", "."],
        ["pip", "freeze", "--python", ".venv/bin/python", "--exclude-editable"],
        [
            "pip",
            "freeze",
            "--python",
            python,
            "--target",
            site_packages,
            "--exclude",
            "axiom-encode",
        ],
    ]


@pytest.mark.parametrize("workflow", sorted(STAGING))
@pytest.mark.parametrize(
    "unhashed_export",
    [
        pytest.param(
            "axiom-oracles @ git+https://github.com/TheAxiomFoundation/axiom-oracles@main\n",
            id="branch-ref",
        ),
        pytest.param(
            "axiom-oracles @ git+https://github.com/attacker/axiom-oracles@"
            + "e" * 40
            + "\n",
            id="foreign-repository",
        ),
        pytest.param("requests==2.32.5\n", id="missing"),
    ],
)
def test_staging_step_refuses_an_unpinned_git_dependency(
    tmp_path, workflow, unhashed_export
):
    result, calls, _ = _run_staging_step(
        tmp_path, workflow, unhashed_export=unhashed_export
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_staged_tree_drifts_from_the_lock(
    tmp_path, workflow
):
    drifted = LOCKED_FREEZE.replace("0.1.47", "0.2.159")
    result, calls, _ = _run_staging_step(tmp_path, workflow, staged_freeze=drifted)

    assert result.returncode != 0
    assert "+claude-agent-sdk==0.2.159" in result.stdout
    assert calls[-1][:2] == ["pip", "freeze"]


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_lock_cannot_be_exported(tmp_path, workflow):
    result, calls, _ = _run_staging_step(tmp_path, workflow, export_fails=True)

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


def test_signed_apply_staging_refuses_a_non_toolcache_interpreter(tmp_path):
    step = _staging_step("signed-apply-reusable.yml")
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "RUNNER_TOOL_CACHE": str(tmp_path / "toolcache"),
            "PYBIN": "/usr/bin/python3",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "resolved interpreter is not the toolcache python" in result.stdout
