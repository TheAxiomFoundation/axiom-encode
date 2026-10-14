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

Nothing in a ``--target`` install may build with backends ``uv.lock`` does not
pin. Each one either passes ``--no-build`` or builds uncached (``--no-cache``)
against ``--build-constraints``, which must name the hash-pinned export of the
lock's ``build`` group written earlier in the same step. The one exception is
signed-apply-reusable.yml, whose pinned encoder predates that group, and
``test_signed_apply_builds_against_its_encoder_build_group`` ends it when the
pin moves. Every one also passes ``--no-config``, so no user or parent-directory
uv.toml can change how it installs or builds.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import tomllib
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

# The provisioner's own selftest provisions a throwaway tree (pyyaml plus hostile
# startup-hook fixtures) under a stand-in supervisor with test-only trust roots;
# nothing it provisions verifies or signs anything. It is exempt from the
# provision rule only: its one --target install is locked like the rest.
EXEMPT_WORKFLOWS = frozenset({"provision-selftest.yml"})

SITE_PACKAGES = "$RUNNER_TEMP/axiom-verification-site-packages"
BUILD_CONSTRAINTS = "$RUNNER_TEMP/axiom-verification-build-constraints.txt"
ORACLES_PATTERN = (
    r"axiom-oracles @ git\+https://github\.com/TheAxiomFoundation/"
    r"axiom-oracles@[0-9a-f]{40}"
)
LOCKED_GIT_LINE = (
    "axiom-oracles @ git+https://github.com/TheAxiomFoundation/"
    "axiom-oracles@e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
)

# workflow -> (interpreter the staged tree is installed for, extra flag, sync,
# whether its builds take the lock's `build` group). The first three stage the
# axiom-encode commit the workflow runs from. signed-apply-reusable.yml stages
# the encoder its AXIOM_ENCODE_REF pins, whose lock predates the group, so its
# export would fail; test_signed_apply_builds_against_its_encoder_build_group
# demands the constraints there as soon as that pin reaches a ref with the group.
STAGING = {
    "bulk-encode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --no-config --locked --python 3.13 --extra api --no-dev",
        True,
    ),
    "golden-regeneration.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --no-config --locked --python 3.13 --extra api --no-dev",
        True,
    ),
    "targeted-signed-reencode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --no-config --locked --python 3.13 --extra api --no-dev",
        True,
    ),
    "signed-apply-reusable.yml": (
        '"$PYBIN"',
        "",
        'uv sync --no-config --locked --no-dev --python "$PYBIN"',
        False,
    ),
}
CONSTRAINED = sorted(workflow for workflow, row in STAGING.items() if row[3])
# Staging workflows whose staging step may still build with unpinned backends.
# Only that step: any other install in the workflow is held to the build rule,
# and the step itself to its exact text (test_staging_step_is_exactly_...).
UNPINNED_BUILDS = frozenset(w for w, row in STAGING.items() if not row[3])
# Non-vacuity: every --target install the discovery must see. Each staging
# workflow has three (locked set, git requirement, project); targeted also
# stages the axiom-compose runtime, and the provisioner selftest its pyyaml.
EXPECTED_TARGET_INSTALLS = {
    "bulk-encode.yml": 3,
    "golden-regeneration.yml": 3,
    "provision-selftest.yml": 1,
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


def _locked_build_exports(commands: list[Command]) -> set[str]:
    """Output files of hash-pinned exports of the lock's `build` group."""
    exported = set()
    for command in commands:
        parsed = _uv_arguments(command)
        if parsed is None or parsed[0] != "export":
            continue
        _, values, _ = parsed
        if (
            "--locked" in values
            and values.get("--only-group") == ["build"]
            and values.keys().isdisjoint({"--group", "--all-groups", "--extra"})
            and "--no-hashes" not in values
            and "--output-file" in values
        ):
            exported.add(values["--output-file"][-1])
    return exported


def _build_constraints_tampered(
    commands: list[Command], build_exported: set[str]
) -> bool:
    """Whether anything but the locked export writes or swaps the constraints.

    The file may be named by its variable or by its literal path. Any command
    that mentions either must be its single assignment, the export, or an
    install that reads it through ``--build-constraints``; and none of them
    may redirect, since ``> "$RUNNER_TEMP/...-build-constraints.txt"`` on any
    command, even the install that reads it, empties it first.
    """
    names = {out.lstrip("$").strip("{}") for out in build_exported}
    literals = set()
    assignments = 0
    for command in commands:
        for assignment in command.assignments:
            name, _, value = assignment.partition("=")
            if name in names:
                assignments += 1
                literals.add(value.rpartition("/")[2])
    if assignments > 1:
        return True
    for command in commands:
        if not any(token in command.line for token in names | literals):
            continue
        if command.assignments and not command.words:
            continue
        if re.search(r"[<>]", command.line):
            return True
        parsed = _uv_arguments(command)
        if parsed is None:
            return True
        subcommand, values, _ = parsed
        if subcommand == "export" and set(values.get("--output-file", [])) <= (
            build_exported
        ):
            continue
        if (
            subcommand == "pip install"
            and values.get("--build-constraints")
            and not any(literal in command.line for literal in literals)
        ):
            continue
        return True
    return False


def _target_install_violations(
    workflows: dict[str, dict], exempt: frozenset[str] = frozenset()
) -> tuple[list[str], dict[str, int]]:
    violations: list[str] = []
    found: dict[str, int] = defaultdict(int)
    for name, job, index, step in _steps(workflows):
        if name in exempt:
            continue
        where = f"{name}:{job}[{index}]"
        commands = _commands(step["run"])
        exported = _locked_exports(commands)
        build_exported = _locked_build_exports(commands)
        tampered = _build_constraints_tampered(commands, build_exported)
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
                elif not set(requirements) <= exported - build_exported:
                    problem = (
                        f"{requirements} is not a hash-pinned `uv export --locked` "
                        "output written in this step"
                    )
            elif positionals == ["$oracles"]:
                if f"grep -xE '{ORACLES_PATTERN}'" not in step["run"]:
                    problem = "$oracles is not the commit-pinned axiom-oracles line"
            elif positionals != ["."]:
                problem = f"installs unpinned requirements {positionals}"
            if problem is None and "--no-config" not in values:
                problem = "reads user or parent-directory uv config (no --no-config)"
            if problem is None:
                staging = f'site_packages="{SITE_PACKAGES}"' in step["run"]
                problem = _build_problem(
                    name if staging else "", values, build_exported, tampered
                )
            if problem:
                violations.append(f"{where}: {problem}: {command.line}")
    return violations, dict(found)


def _build_problem(
    staging_workflow: str,
    values: dict[str, list[str]],
    build_exported: set[str],
    tampered: bool = False,
) -> str | None:
    """Why this install could build with backends uv.lock does not pin, if so.

    ``staging_workflow`` is the workflow when the install is in its staging
    step, else "".
    """
    constraints = values.get("--build-constraints", [])
    if constraints:
        if not set(constraints) <= build_exported:
            return (
                f"--build-constraints {constraints} is not the locked `build` "
                "group export written in this step"
            )
        if "--no-cache" not in values:
            # uv reuses a cached wheel whatever the build constraints say.
            return "--build-constraints without --no-cache"
        if tampered:
            return "the build constraints are written outside their locked export"
        return None
    if "--no-build" in values or staging_workflow in UNPINNED_BUILDS:
        return None
    return "may build with unpinned backends (no --build-constraints or --no-build)"


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


def _provision_violations(
    workflows: dict[str, dict], exempt: frozenset[str] = EXEMPT_WORKFLOWS
) -> tuple[list[str], int]:
    violations: list[str] = []
    provisions = 0
    steps = list(_steps(workflows))
    for name, job, index, step in steps:
        if name in exempt:
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


def _canonical_staging(
    python: str, extra: str, sync: str, constrained: bool
) -> list[str]:
    extra = f" {extra}" if extra else ""
    install = f'uv pip install --no-config --python {python} --target "$site_packages"'
    if constrained:
        install += ' --no-cache --build-constraints "$build_constraints"'
    return [
        "set -euo pipefail",
        sync,
        f'site_packages="{SITE_PACKAGES}"',
        'locked="$RUNNER_TEMP/axiom-verification-locked-requirements.txt"',
        *([f'build_constraints="{BUILD_CONSTRAINTS}"'] if constrained else []),
        'rm -rf "$site_packages"',
        f"uv export --no-config --locked --no-dev{extra} --no-emit-project"
        " --no-emit-package axiom-oracles --format requirements-txt"
        ' --output-file "$locked"',
        *(
            [
                "uv export --no-config --locked --only-group build --no-emit-project"
                ' --format requirements-txt --output-file "$build_constraints"'
            ]
            if constrained
            else []
        ),
        f'oracles="$(uv export --no-config --locked --no-dev{extra} --no-emit-project'
        " --no-hashes --no-annotate --no-header --format requirements-txt"
        f" | grep -xE '{ORACLES_PATTERN}')\"",
        f'{install} --require-hashes --no-deps -r "$locked"',
        f'{install} --no-deps "$oracles"',
        f"{install} --no-deps .",
        "uv pip freeze --no-config --python .venv/bin/python --exclude-editable"
        ' > "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        f'uv pip freeze --no-config --python {python} --target "$site_packages"'
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
        violations, _ = _provision_violations(
            {workflow: workflows[workflow]}, exempt=frozenset()
        )
        assert violations, f"{workflow} no longer needs its exemption"


def test_the_selftest_stages_the_locked_pyyaml():
    # Its one install is the hash-pinned export of uv.lock's
    # `provision-selftest` group, from a wheel (`--no-build`).
    [step] = [
        step
        for name, _job, _index, step in _steps(_load_workflows())
        if name == "provision-selftest.yml" and "selftest_locked" in step["run"]
    ]
    lines = _logical_lines(step["run"])
    assignment = 'selftest_locked="$RUNNER_TEMP/selftest-locked-requirements.txt"'
    export = (
        "uv export --no-config --locked --only-group provision-selftest"
        " --no-emit-project --format requirements-txt"
        ' --output-file "$selftest_locked"'
    )
    install = (
        'uv pip install --no-config --python "$PYBIN" --target "$RUNNER_TEMP/staged"'
        ' --require-hashes --no-deps --no-build -r "$selftest_locked"'
    )

    assert lines.count(assignment) == lines.count(export) == lines.count(install) == 1
    assert lines.index(assignment) < lines.index(export) < lines.index(install)
    assert sum("selftest_locked" in line for line in lines) == 3


def test_signed_apply_builds_against_its_encoder_build_group():
    """signed-apply's builds take the `build` group once its pinned encoder has one.

    The step stages the encoder that AXIOM_ENCODE_REF pins, from that ref's
    uv.lock, which has no `build` group yet: exporting one would fail closed for
    every caller. This test flips when the pin moves to a ref whose lock has the
    group, so the change that moves it must also add the constraints (and set
    the STAGING flag, which also ends its UNPINNED_BUILDS exception).
    """
    workflow = yaml.safe_load(
        (WORKFLOWS / "signed-apply-reusable.yml").read_text(encoding="utf-8")
    )
    ref = workflow["env"]["AXIOM_ENCODE_REF"]
    assert re.fullmatch(r"[0-9a-f]{40}", ref), ref
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{ref}:pyproject.toml"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shown.returncode != 0:
        # CI checks out full history, so the pinned commit must be there.
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(f"AXIOM_ENCODE_REF {ref} is not in this clone: {shown.stderr}")
        pytest.skip(f"AXIOM_ENCODE_REF {ref} is not in this clone")
    has_group = "build" in tomllib.loads(shown.stdout).get("dependency-groups", {})

    assert STAGING["signed-apply-reusable.yml"][3] == has_group
    assert ("signed-apply-reusable.yml" in UNPINNED_BUILDS) == (not has_group)


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
    "uv-reads-config": (
        f'uv pip install --no-build --target "{SITE_PACKAGES}" --no-deps .'
    ),
    "uv-builds-unpinned": (
        f'uv pip install --no-config --target "{SITE_PACKAGES}" --no-deps .'
    ),
    "uv-handwritten-build-constraints": (
        f'uv pip install --no-config --no-cache --target "{SITE_PACKAGES}"'
        " --no-deps --build-constraints /tmp/hatchling.txt ."
    ),
    "uv-build-constraints-on-a-warm-cache": (
        f'uv pip install --no-config --target "{SITE_PACKAGES}" --no-deps'
        ' -b "$build_constraints" .'
    ),
    "uv-installs-the-build-backends": (
        f'uv pip install --no-config --target "{SITE_PACKAGES}" --no-deps'
        ' --no-build --require-hashes -r "$build_constraints"'
    ),
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

    if (
        workflow in UNPINNED_BUILDS
        and placement == "end-of-staging"
        and mutation == "uv-builds-unpinned"
    ):
        # Inside the one step whose builds may float, the step's exact text
        # is the guard, not the build rule.
        python, extra, sync, constrained = STAGING[workflow]
        lines = _logical_lines(step["run"])
        del lines[1 : lines.index("esac") + 1]
        assert lines != _canonical_staging(python, extra, sync, constrained)
        return
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


TAMPERING = {
    "truncate-literal": f': > "{BUILD_CONSTRAINTS}"',
    "copy-over-literal": f'cp /tmp/hatchling.txt "{BUILD_CONSTRAINTS}"',
    "install-redirects-literal": (
        f'uv pip install --no-config --python .venv/bin/python --target "{SITE_PACKAGES}"'
        ' --no-cache --build-constraints "$build_constraints" --no-deps .'
        f' > "{BUILD_CONSTRAINTS}"'
    ),
    "append-variable": 'echo "hatchling==0" >> "$build_constraints"',
    "reassign": 'build_constraints="$RUNNER_TEMP/other.txt"',
}


@pytest.mark.parametrize("workflow", CONSTRAINED)
@pytest.mark.parametrize("form", sorted(TAMPERING))
def test_writes_to_the_build_constraints_are_rejected(workflow, form):
    workflows = copy.deepcopy(_load_workflows())
    step = _staging_step(workflow, workflows)
    step["run"] = f"{step['run']}{TAMPERING[form]}\n"

    violations, _ = _target_install_violations(workflows)

    assert any("written outside their locked export" in v for v in violations), (
        violations
    )


def test_the_compose_runtime_install_builds_nothing():
    # axiom-compose's lock pins no build backend; its no-extras export is wheels
    # only, and --no-build makes a future sdist fail instead of building.
    workflows = copy.deepcopy(_load_workflows())
    [step] = [
        step
        for job in workflows["targeted-signed-reencode.yml"]["jobs"].values()
        for step in job["steps"]
        if "axiom-compose-site-packages" in step.get("run", "")
        and "uv export" in step["run"]
    ]
    assert "--require-hashes --no-deps --no-build" in step["run"]
    step["run"] = step["run"].replace("--no-deps --no-build", "--no-deps")

    violations, _ = _target_install_violations(workflows)

    assert any(
        "axiom-compose-site-packages" in v and "unpinned backends" in v
        for v in violations
    ), violations


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
    building = "--only-group" in argv
    if os.environ.get("FAKE_BUILD_EXPORT_FAIL" if building else "FAKE_EXPORT_FAIL"):
        sys.exit(2)
    if "--no-hashes" in argv:
        sys.stdout.write(os.environ["FAKE_UNHASHED_EXPORT"])
    else:
        target = argv[argv.index("--output-file") + 1]
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(os.environ["FAKE_BUILD_EXPORT" if building else "FAKE_EXPORT"])
elif argv[:2] == ["pip", "install"] and "--build-constraints" in argv:
    # What each install was given as its build constraints, read at call time.
    path = argv[argv.index("--build-constraints") + 1]
    seen = open(path, encoding="utf-8").read() if os.path.exists(path) else None
    with open(os.environ["FAKE_UV_BUILD_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(seen) + "\\n")
elif argv[:2] == ["pip", "freeze"]:
    key = "FAKE_STAGED_FREEZE" if "--target" in argv else "FAKE_LOCKED_FREEZE"
    sys.stdout.write(os.environ[key])
"""

LOCKED_FREEZE = f"claude-agent-sdk==0.1.47\n{LOCKED_GIT_LINE}\nrequests==2.32.5\n"
LOCKED_EXPORT = "requests==2.32.5 \\\n    --hash=sha256:" + "0" * 64 + "\n"
BUILD_EXPORT = "hatchling==1.32.4 \\\n    --hash=sha256:" + "1" * 64 + "\n"


def _run_staging_step(
    tmp_path: Path,
    workflow: str,
    *,
    unhashed_export: str | None = None,
    staged_freeze: str = LOCKED_FREEZE,
    export_fails: bool = False,
    build_export_fails: bool = False,
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
    build_log = tmp_path / "uv-build-constraints.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "RUNNER_TOOL_CACHE": str(toolcache),
        "PYBIN": f"{toolcache}/Python/3.14.4/x64/bin/python",
        "FAKE_UV_LOG": str(log),
        "FAKE_UV_BUILD_LOG": str(build_log),
        "FAKE_EXPORT": LOCKED_EXPORT,
        "FAKE_BUILD_EXPORT": BUILD_EXPORT,
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
    if build_export_fails:
        env["FAKE_BUILD_EXPORT_FAIL"] = "1"
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
    seen = (
        [json.loads(line) for line in build_log.read_text().splitlines()]
        if build_log.exists()
        else []
    )
    return result, calls, runner_temp, seen


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_runs_the_locked_sequence(tmp_path, workflow):
    result, calls, runner_temp, seen = _run_staging_step(tmp_path, workflow)

    assert result.returncode == 0, result.stderr
    python, extra, sync, constrained = STAGING[workflow]
    python = python.replace(
        '"$PYBIN"', f"{tmp_path}/toolcache/Python/3.14.4/x64/bin/python"
    )
    extra_args = extra.split()
    site_packages = str(runner_temp / "axiom-verification-site-packages")
    locked = str(runner_temp / "axiom-verification-locked-requirements.txt")
    build_constraints = str(runner_temp / "axiom-verification-build-constraints.txt")
    install = [
        *("pip", "install", "--no-config", "--python", python),
        *("--target", site_packages),
    ]
    if constrained:
        install += ["--no-cache", "--build-constraints", build_constraints]
    export = [
        *("export", "--no-config", "--locked", "--no-dev"),
        *(*extra_args, "--no-emit-project"),
    ]
    build_export = [
        *("export", "--no-config", "--locked", "--only-group", "build"),
        *("--no-emit-project", "--format", "requirements-txt"),
        *("--output-file", build_constraints),
    ]
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
        *([build_export] if constrained else []),
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
        [
            *("pip", "freeze", "--no-config", "--python", ".venv/bin/python"),
            "--exclude-editable",
        ],
        [
            "pip",
            "freeze",
            "--no-config",
            "--python",
            python,
            "--target",
            site_packages,
            "--exclude",
            "axiom-encode",
        ],
    ]
    # Each of the three installs read the build backends the lock exported.
    assert seen == ([BUILD_EXPORT] * 3 if constrained else [])


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
    result, calls, _, _ = _run_staging_step(
        tmp_path, workflow, unhashed_export=unhashed_export
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_staged_tree_drifts_from_the_lock(
    tmp_path, workflow
):
    drifted = LOCKED_FREEZE.replace("0.1.47", "0.2.159")
    result, calls, _, _ = _run_staging_step(tmp_path, workflow, staged_freeze=drifted)

    assert result.returncode != 0
    assert "+claude-agent-sdk==0.2.159" in result.stdout
    assert calls[-1][:2] == ["pip", "freeze"]


@pytest.mark.parametrize("workflow", CONSTRAINED)
def test_staging_step_fails_when_the_build_backends_cannot_be_exported(
    tmp_path, workflow
):
    # An old lock without the `build` group, or any export failure, must stop
    # the step before anything is built with unpinned backends.
    result, calls, _, seen = _run_staging_step(
        tmp_path, workflow, build_export_fails=True
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)
    assert seen == []


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_lock_cannot_be_exported(tmp_path, workflow):
    result, calls, _, _ = _run_staging_step(tmp_path, workflow, export_fails=True)

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
