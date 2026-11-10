"""Every trusted runtime's site-packages must be exactly a uv lock.

``scripts/provision_verification_supervisor.py`` copies a caller-staged
``--site-packages`` tree into the signer's trusted Python import root, and
``scripts/provision_axiom_compose_runtime.py`` does the same for the
axiom-compose runtime. A resolving ``uv pip install --target ... ".[api]"``
ignores ``uv.lock`` and floats every ``>=`` dependency to the index's latest at
run time: targeted run 36174590343 logged ``claude-agent-sdk==0.1.47`` in the
locked ``.venv`` and ``claude-agent-sdk==0.2.159`` in the verification tree of
the same job.

Commands are found with ``tests/workflow_shell.py`` through the toolchain
test's reader, which also follows the reviewed signed-encode ``advance.sh``
script. So ``pip3``, ``python -m pip``, ``"$VAR" -m pip``, the
``sudo``/``env``/``bash -c``/``eval`` wrappers, attached ``-tDIR``,
``--target=DIR`` and redirections are all seen. In every workflow except the
provisioner's selftest:

- pip installs and uninstalls nothing. It ignores ``uv.lock``, and
  ``PIP_TARGET`` or a pip config file can aim it at a tree the command never
  names.
- ``uv pip uninstall`` never runs.
- Every ``--target`` install is a ``uv pip install --no-deps`` of a hash-pinned
  ``uv export --locked`` output written in the same step (with
  ``--require-hashes``), the commit-pinned ``axiom-oracles`` requirement, or
  the project.
- Outside the verification staging steps, which must equal the canonical block
  line for line, two commands may name a protected tree: its locked install
  and its provisioner. Any other command, assignment or redirection that names
  one fails, and so does any other text in the workflow: a loop header, an
  array, a heredoc, a wrapper's option or a YAML key such as ``env:``. Comment
  lines and step names may name a tree.
- Every provision of either provisioner resolves its ``--site-packages`` with
  that provisioner's own argument parser. The compose runtime's must be its
  locked tree. The verification supervisor's must be the staged tree or, in the
  issue233 diagnostic, the purelib of a ``uv sync --locked``/``--frozen``
  environment. That job runs no other uv call and installs nothing with pip,
  and nothing else in it names the purelib, assigns ``site`` or ``python_path``
  again or names the environment outside its ``bin/`` directory.
- A provision's ``if:`` holds no status function (``success()``, ``always()``,
  ``failure()`` or ``cancelled()``), so the runner's implicit ``success()``
  keeps it from running after a failed step. A staging step carries no key that
  would let its ``diff`` fail without stopping the job, and no ``${{ }}``
  expression, which the runner substitutes before bash reads even a comment.

The staging steps are then executed against a fake ``uv`` to prove their
fail-closed paths. As in ``test_workflow_toolchain_pins.py``, the threat model
is an accidental regression written in ordinary shell. A command that reaches a
tree without naming it (a glob over ``$RUNNER_TEMP``, a path assembled in a
variable) is out of reach.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import workflow_shell
from tests.test_workflow_toolchain_pins import _script_commands, _strings
from tests.workflow_shell import Command

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
UV_OPTIONS = workflow_shell.load_uv_options()
PIP = re.compile(r"pip[0-9.]*")
PYTHON = re.compile(r"python[0-9.]*t?")

# The provisioner's own selftest stages a throwaway tree (pyyaml plus hostile
# startup-hook fixtures) under a stand-in supervisor with test-only trust roots;
# nothing it provisions verifies or signs anything.
EXEMPT_WORKFLOWS = frozenset({"provision-selftest.yml"})

SITE_PACKAGES = "$RUNNER_TEMP/axiom-verification-site-packages"
COMPOSE_SITE_PACKAGES = "$RUNNER_TEMP/axiom-compose-site-packages"
VERIFICATION_PROVISIONER = "provision_verification_supervisor.py"
COMPOSE_PROVISIONER = "provision_axiom_compose_runtime.py"
# Each protected tree, by the directory name every spelling of its path
# contains, and the provisioner that consumes it.
PROTECTED_TREES = {
    "axiom-verification-site-packages": VERIFICATION_PROVISIONER,
    "axiom-compose-site-packages": COMPOSE_PROVISIONER,
}
ORACLES_PATTERN = (
    r"axiom-oracles @ git\+https://github\.com/TheAxiomFoundation/"
    r"axiom-oracles@[0-9a-f]{40}"
)
LOCKED_GIT_LINE = (
    "axiom-oracles @ git+https://github.com/TheAxiomFoundation/"
    "axiom-oracles@e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
)

# The workflows that stage the verification tree, all with the same block.
STAGING = ("bulk-encode.yml", "golden-regeneration.yml", "targeted-signed-reencode.yml")
# Non-vacuity: every --target install the discovery must see. Each staging
# workflow has three (locked set, git requirement, project); targeted also
# stages the axiom-compose runtime.
EXPECTED_TARGET_INSTALLS = {
    "bulk-encode.yml": 3,
    "golden-regeneration.yml": 3,
    "targeted-signed-reencode.yml": 4,
}
# Non-vacuity: the commands outside staging that name each protected tree. The
# verification tree's are the provisions of the three staging workflows; the
# compose runtime's are its install and its provision.
EXPECTED_TREE_CONSUMERS = {
    "axiom-verification-site-packages": 3,
    "axiom-compose-site-packages": 2,
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
# What names a locked environment's purelib, or the environment outside bin/.
PURELIB_NAME = re.compile(r"\$\{?site\b|\.venv(?!/bin\b)")
PURELIB_ASSIGNMENT = re.compile(r"(?:^|[\s;&|(])site\+?=")
INTERPRETER_ASSIGNMENT = re.compile(r"(?<![-\w])python_path\+?=")
# A step with one of these in its `if:` runs after an earlier step failed.
RUNS_AFTER_FAILURE = re.compile(
    r"\b(?:always|failure|cancelled|success)\s*\(", re.IGNORECASE
)


def _logical_lines(script: str) -> list[str]:
    """``script``'s lines, joined as bash joins them outside quotes: a
    backslash-newline disappears and leaves nothing between its neighbours, a
    full-line comment starts only where a word can start and ends at its
    newline, even after a trailing backslash, and a blank line ends a
    continuation."""
    lines: list[str] = []

    def finish(text: str) -> None:
        if text.strip(" \t"):
            lines.append(re.sub(r"[ \t]+", " ", text.strip(" \t")))

    pending = ""
    physical = script.split("\n")
    for number, raw in enumerate(physical, start=1):
        body = raw.lstrip(" \t")
        if body.startswith("#") and (
            raw != body or not pending or pending[-1] in " \t"
        ):
            finish(pending)
            pending = ""
            continue
        text = pending + raw
        backslashes = len(text) - len(text.rstrip("\\"))
        if backslashes % 2 and number < len(physical):
            pending = text[:-1]
            continue
        finish(text)
        pending = ""
    return lines


def _code_lines(strings: Iterable[str]) -> Iterator[str]:
    """Each line of ``strings`` that holds more than a comment."""
    for text in strings:
        for line in text.splitlines():
            if not line.lstrip().startswith("#"):
                yield line


@functools.cache
def _parsed_workflows(directory: Path) -> tuple[tuple[str, dict], ...]:
    # GitHub runs both extensions from .github/workflows.
    paths = sorted(p for p in directory.iterdir() if p.suffix in {".yml", ".yaml"})
    return tuple((p.name, yaml.safe_load(p.read_text(encoding="utf-8"))) for p in paths)


def _load_workflows(directory: Path = WORKFLOWS) -> dict[str, dict]:
    """Every workflow in ``directory``, as a copy a test may edit."""
    return copy.deepcopy(dict(_parsed_workflows(directory)))


def _every_step(workflows: dict[str, dict]):
    for name, workflow in workflows.items():
        for job_name, job in (workflow.get("jobs") or {}).items():
            for index, step in enumerate(job.get("steps") or []):
                yield name, job_name, index, step


def _steps(workflows: dict[str, dict]):
    for name, job_name, index, step in _every_step(workflows):
        if "run" in step:
            yield name, job_name, index, step


def _commands(name: str, job: str, script: str) -> list[Command]:
    commands = list(_script_commands(script, name, job))
    # This one reviewed array has two bounded values, not arbitrary shell
    # arguments. Recognize its complete definition and every textual use;
    # unknown/reassigned arrays stay literal and fail the provisioner's parser.
    definition = [
        "reviewed_pr_args=()",
        'if [ "$LIVE_PR_ADMISSION" = "true" ]; then',
        'reviewed_pr_args+=(--reviewed-pr-selector "$RUNNER_TEMP/reviewed-pr-selector.json")',
        "fi",
    ]
    lines = _logical_lines(script)
    definitions = [
        index
        for index in range(len(lines))
        if lines[index : index + len(definition)] == definition
    ]
    if len(definitions) != 1 or script.count("reviewed_pr_args") != 3:
        return commands
    expansion = '"${reviewed_pr_args[@]}"'
    if script.count(expansion) != 1:
        return commands
    definition_end = definitions[0] + len(definition)
    if not any(expansion in line for line in lines[definition_end:]):
        return commands
    for index, command in enumerate(commands):
        if command.words[-1:] != ("${reviewed_pr_args[@]}",):
            continue
        disabled = replace(command, words=command.words[:-1])
        enabled = replace(
            command,
            words=(
                *command.words[:-1],
                "--reviewed-pr-selector",
                "$RUNNER_TEMP/reviewed-pr-selector.json",
            ),
        )
        off, on = _provision(disabled), _provision(enabled)
        if off is None or on is None or off[1] is None or on[1] is None:
            continue
        if off[0] != VERIFICATION_PROVISIONER or on[0] != off[0]:
            continue
        off_values, on_values = vars(off[1]).copy(), vars(on[1]).copy()
        off_selector = off_values.pop("reviewed_pr_selector", None)
        on_selector = on_values.pop("reviewed_pr_selector", None)
        if (
            off_selector is None
            and on_selector == Path("$RUNNER_TEMP/reviewed-pr-selector.json")
            and off_values == on_values
        ):
            commands[index] = enabled
    return commands


def _runs_pip(command: Command) -> bool:
    """Whether ``command`` runs pip: directly, as ``python -m pip``, or as
    ``"$PYBIN" -m pip`` (an interpreter named by a variable)."""
    if not command.words:
        return False
    if PIP.fullmatch(command.program):
        return True
    return bool(
        (PYTHON.fullmatch(command.program) or command.words[0].startswith("$"))
        and PIP.fullmatch(workflow_shell.python_module(command.words) or "")
    )


def _uv_arguments(
    command: Command,
) -> tuple[str, dict[str, list[str]], tuple[str, ...]] | None:
    """A uv call as (subcommand, {long option: values}, positionals).

    The values and positionals are the ones the shared reader parsed, so what
    ``uv run`` launches stays out of them and a global option's value is never
    taken for the subcommand.
    """
    invocation = workflow_shell.uv_invocation(command, UV_OPTIONS)
    if invocation is None or not invocation.subcommand:
        return None
    values: dict[str, list[str]] = defaultdict(list)
    for name, value in invocation.values:
        values[name].append(value)
    return invocation.subcommand, dict(values), invocation.positionals


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
        commands = _commands(name, job, step["run"])
        exported = _locked_exports(commands)
        for command in commands:
            if _runs_pip(command):
                if {"install", "uninstall"} & set(command.words):
                    violations.append(
                        f"{where}: pip changes a tree outside uv.lock: {command.line}"
                    )
                continue
            parsed = _uv_arguments(command)
            if parsed is None:
                continue
            subcommand, values, positionals = parsed
            if subcommand == "pip uninstall":
                violations.append(f"{where}: removes packages: {command.line}")
                continue
            if (
                subcommand not in {"pip install", "pip sync"}
                or "--target" not in values
            ):
                continue
            found[name] += 1
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
                    problem = f"mixes {list(positionals)} with requirement files"
                elif "--require-hashes" not in values:
                    problem = "installs a requirement file without --require-hashes"
                elif not set(requirements) <= exported:
                    problem = (
                        f"{requirements} is not a hash-pinned `uv export --locked` "
                        "output written in this step"
                    )
            elif positionals == ("$oracles",):
                if f"grep -xE '{ORACLES_PATTERN}'" not in step["run"]:
                    problem = "$oracles is not the commit-pinned axiom-oracles line"
            elif positionals != (".",):
                problem = f"installs unpinned requirements {list(positionals)}"
            if problem:
                violations.append(f"{where}: {problem}: {command.line}")
    return violations, dict(found)


def _script_module(name: str) -> ModuleType:
    """``scripts/<name>.py``, loaded the way its sibling imports expect."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@functools.cache
def _parser(script: str) -> argparse.ArgumentParser:
    # The compose provisioner imports the verification provisioner by name.
    _script_module("provision_verification_supervisor")
    return _script_module(script.removesuffix(".py"))._argument_parser()


def _provisioner(name: str) -> str | None:
    """The provisioner script that ``name``, a path or a module, is, if any."""
    for script in PROTECTED_TREES.values():
        stem = script.removesuffix(".py")
        if name.rsplit("/", 1)[-1] == script or name in {stem, f"scripts.{stem}"}:
            return script
    return None


def _provision(command: Command) -> tuple[str, argparse.Namespace | None] | None:
    """The provisioner ``command`` runs and the arguments its own parser
    resolves (None if the parser refuses them), or None for any other command,
    including one that only reads the provisioner's file."""
    if not command.words:
        return None
    script, arguments = _provisioner(command.words[0]), 1
    if script is None and (
        PYTHON.fullmatch(command.program) or command.words[0].startswith("$")
    ):
        program = workflow_shell.python_program(command.words)
        if program is not None:
            script, arguments = _provisioner(program[0]), program[1]
    if script is None:
        # Behind a wrapper the reader doesn't know ("$SUDO", chronic, coverage
        # run), the script is the word its --site-packages follows.
        for index, word in enumerate(command.words):
            script, arguments = _provisioner(word), index + 1
            if script and any(
                later.startswith("--site") for later in command.words[arguments:]
            ):
                break
        else:
            return None
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return script, _parser(script).parse_args(list(command.words[arguments:]))
    except SystemExit:
        return script, None


def _changes_an_environment(commands: list[Command]) -> bool:
    """Whether ``commands`` do more to a locked environment than sync it: a pip
    install or uninstall, or a second uv call of any kind."""
    return sum(_uv_arguments(command) is not None for command in commands) > 1 or any(
        _runs_pip(command) and {"install", "uninstall"} & set(command.words)
        for command in commands
    )


def _names_more_than_its_purelib(job: dict) -> bool:
    """Whether ``job``'s text, comment lines and step names aside, names the
    purelib beyond its one assignment and its one provision, names the
    environment outside ``bin/``, or assigns its interpreter more than once."""
    steps = [
        {key: value for key, value in step.items() if key != "name"}
        for step in job.get("steps", [])
    ]
    lines = list(_code_lines(_strings({**job, "steps": steps})))
    return any(
        sum(len(pattern.findall(line)) for line in lines) != 1
        for pattern in (PURELIB_NAME, PURELIB_ASSIGNMENT, INTERPRETER_ASSIGNMENT)
    )


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
        for command in _commands(name, job, run):
            provision = _provision(command)
            if provision is None:
                continue
            where = f"{name}:{job}[{index}]"
            script, args = provision
            if RUNS_AFTER_FAILURE.search(str(step.get("if", ""))):
                violations.append(f"{where}: provisions after a failed step")
            if script == COMPOSE_PROVISIONER:
                if args is None or str(args.site_packages) != COMPOSE_SITE_PACKAGES:
                    violations.append(
                        f"{where}: the compose provision does not read its locked tree"
                    )
                continue
            provisions += 1
            if args is None:
                violations.append(f"{where}: the provisioner refuses its arguments")
                continue
            # argparse keeps the last --site-packages and accepts abbreviations
            # such as --site-pack; this is the value the provisioner acts on.
            site = str(args.site_packages)
            job_steps = [(i, s) for n, j, i, s in steps if n == name and j == job]
            earlier = [s for i, s in job_steps if i < index]
            if site == SITE_PACKAGES:
                if not any(
                    f'site_packages="{SITE_PACKAGES}"' in s["run"] for s in earlier
                ):
                    violations.append(f"{where}: no earlier locked staging step")
            elif site == "$site" and name in LOCKED_VENV_PROVISIONS:
                python = LOCKED_VENV_PROVISIONS[name]
                synced = [
                    parsed
                    for s in earlier
                    for c in _commands(name, job, s["run"])
                    if (parsed := _uv_arguments(c)) is not None
                    and parsed[0] == "sync"
                    and parsed[1].keys() & {"--locked", "--frozen"}
                ]
                logical = _logical_lines(run)
                if (
                    f'python_path="{python}"' not in logical
                    or PURELIB not in logical
                    or command.words[0] != "$python_path"
                ):
                    violations.append(f"{where}: $site is not {python}'s purelib")
                if not synced:
                    violations.append(f"{where}: no `uv sync --locked|--frozen`")
                job_commands = [
                    c for _, s in job_steps for c in _commands(name, job, s["run"])
                ]
                if _changes_an_environment(job_commands) or (
                    _names_more_than_its_purelib(workflows[name]["jobs"][job])
                ):
                    violations.append(
                        f"{where}: the job changes the locked environment"
                    )
            else:
                violations.append(f"{where}: --site-packages {site} is not locked")
    return violations, provisions


def _is_verification_staging(name: str, step: dict) -> bool:
    return name in STAGING and f'site_packages="{SITE_PACKAGES}"' in step.get("run", "")


def _tree_mentions(command: Command, tree: str) -> list[tuple[str, str]]:
    """Where ``command`` names ``tree``, as (word|assignment|redirect, text)."""
    return [
        *(("word", word) for word in command.words if tree in word),
        *(("assignment", text) for text in command.assignments if tree in text),
        *(("redirect", target) for _, target in command.redirects if tree in target),
    ]


def _consumes(command: Command, tree: str) -> bool:
    """Whether ``command`` is ``tree``'s locked install or its provisioner, and
    names the tree there and nowhere else."""
    mentions = _tree_mentions(command, tree)
    if len(mentions) != 1 or mentions[0][0] != "word":
        return False
    path = f"$RUNNER_TEMP/{tree}"
    parsed = _uv_arguments(command)
    if parsed is not None:
        subcommand, values, _ = parsed
        return subcommand == "pip install" and values.get("--target", [""])[-1] == path
    provision = _provision(command)
    return (
        provision is not None
        and provision[0] == PROTECTED_TREES[tree]
        and provision[1] is not None
        and str(provision[1].site_packages) == path
    )


def _tree_violations(
    workflows: dict[str, dict], exempt: frozenset[str] = EXEMPT_WORKFLOWS
) -> tuple[list[str], dict[str, int]]:
    violations: list[str] = []
    consumers: Counter[str] = Counter()
    for name, workflow in workflows.items():
        if name in exempt:
            continue
        # `env:` at these levels reaches every step below it.
        outer = {
            "workflow": {k: v for k, v in workflow.items() if k != "jobs"},
            **{
                f"job {job}": {k: v for k, v in body.items() if k != "steps"}
                for job, body in (workflow.get("jobs") or {}).items()
            },
        }
        for scope, keys in outer.items():
            lines = list(_code_lines(_strings(keys)))
            for tree in PROTECTED_TREES:
                if any(tree in line for line in lines):
                    violations.append(f"{name}: the {scope} keys name {tree}")
    for name, job, index, step in _every_step(workflows):
        if name in exempt or _is_verification_staging(name, step):
            continue
        where = f"{name}:{job}[{index}]"
        consumed: Counter[str] = Counter()
        for command in _commands(name, job, step.get("run", "")):
            for tree in PROTECTED_TREES:
                if not _tree_mentions(command, tree):
                    continue
                if _consumes(command, tree):
                    consumed[tree] += 1
                    # The words join quoted pieces; the line keeps the split.
                    if tree not in command.line:
                        violations.append(
                            f"{where}: splits {tree} across quotes: {command.line}"
                        )
                else:
                    violations.append(
                        f"{where}: names {tree} outside its staging, locked "
                        f"install and provisioner: {command.line}"
                    )
        consumers.update(consumed)
        # The reader keeps no `for` header, array word, heredoc body or wrapper
        # option, and it never reads YAML keys, so count the step's text too.
        lines = list(
            _code_lines(_strings({k: v for k, v in step.items() if k != "name"}))
        )
        for tree in PROTECTED_TREES:
            mentions = sum(line.count(tree) for line in lines)
            if mentions != consumed[tree]:
                violations.append(
                    f"{where}: its text names {tree} {mentions} time(s) for "
                    f"{consumed[tree]} locked install(s) and provision(s)"
                )
    return violations, dict(consumers)


@functools.cache
def _baseline() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    workflows = _load_workflows()
    return (
        frozenset(_target_install_violations(workflows)[0]),
        frozenset(_tree_violations(workflows)[0]),
        frozenset(_provision_violations(workflows)[0]),
    )


def _new_violations(workflows: dict[str, dict]) -> set[str]:
    targets, trees, provisions = _baseline()
    return (
        (set(_target_install_violations(workflows)[0]) - targets)
        | (set(_tree_violations(workflows)[0]) - trees)
        | (set(_provision_violations(workflows)[0]) - provisions)
    )


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
        if name == workflow and VERIFICATION_PROVISIONER in step["run"]
    ]
    assert len(steps) == 1, f"{workflow}: expected one provision step, got {steps}"
    return steps[0]


def _compose_step(workflows: dict[str, dict]) -> dict:
    steps = [
        step
        for name, _job, _index, step in _steps(workflows)
        if name == "targeted-signed-reencode.yml"
        and "axiom-compose-site-packages" in step["run"]
        and "uv export" in step["run"]
    ]
    assert len(steps) == 1, steps
    return steps[0]


def _canonical_staging() -> list[str]:
    install = 'uv pip install --python .venv/bin/python --target "$site_packages"'
    return [
        "set -euo pipefail",
        "uv sync --locked --python 3.13 --extra api --no-dev",
        f'site_packages="{SITE_PACKAGES}"',
        'locked="$RUNNER_TEMP/axiom-verification-locked-requirements.txt"',
        'rm -rf "$site_packages"',
        "uv export --locked --no-dev --extra api --no-emit-project"
        " --no-emit-package axiom-oracles --format requirements-txt"
        ' --output-file "$locked"',
        'oracles="$(uv export --locked --no-dev --extra api --no-emit-project'
        " --no-hashes --no-annotate --no-header --format requirements-txt"
        f" | grep -xE '{ORACLES_PATTERN}')\"",
        f'{install} --require-hashes --no-deps -r "$locked"',
        f'{install} --no-deps "$oracles"',
        f"{install} --no-deps .",
        "uv pip freeze --python .venv/bin/python --exclude-editable"
        ' > "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        'uv pip freeze --python .venv/bin/python --target "$site_packages"'
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


def _staging_problems(step: dict) -> list[str]:
    """What keeps ``step`` from running the canonical block and nothing else."""
    problems = []
    if _logical_lines(step["run"]) != _canonical_staging():
        problems.append("its lines differ from the canonical block")
    # The runner substitutes an expression before bash reads the script, even in
    # a comment, and a newline in the value ends the comment.
    if "${{" in step["run"]:
        problems.append("it holds an expression")
    # `continue-on-error` or `if:` would let a job provision a tree that failed
    # the diff.
    extra = set(step) - {"name", "run", "working-directory"}
    if extra:
        problems.append(f"it carries {sorted(extra)}")
    return problems


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_is_exactly_the_locked_install(workflow):
    assert _staging_problems(_staging_step(workflow)) == []


@pytest.mark.parametrize("workflow", STAGING)
def test_a_staging_comment_may_hold_no_expression(workflow):
    step = _staging_step(workflow)
    step["run"] += "# note: ${{ inputs.note }}\n"

    assert _staging_problems(step) == ["it holds an expression"]


@pytest.mark.parametrize("key", ["continue-on-error", "if"])
@pytest.mark.parametrize("workflow", STAGING)
def test_a_staging_step_carries_no_key_that_could_excuse_its_diff(workflow, key):
    step = _staging_step(workflow)
    step[key] = True

    assert _staging_problems(step) == [f"it carries ['{key}']"]


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            '# Drop one package. \\\nrm -rf "$x"\n',
            ['rm -rf "$x"'],
            id="a-comment-ends-at-its-newline",
        ),
        pytest.param("a \\\n  b\n", ["a b"], id="a-continuation-joins"),
        pytest.param("a \\\n# b\nc\n", ["a", "c"], id="a-comment-ends-a-continuation"),
        pytest.param(
            "a \\\n# b \\\nc\n",
            ["a", "c"],
            id="a-continued-comment-still-ends-at-its-newline",
        ),
        pytest.param("a \\\n\nc\n", ["a", "c"], id="a-blank-line-ends-a-continuation"),
        pytest.param("a\\\nb\n", ["ab"], id="a-glued-continuation-adds-no-space"),
        pytest.param("a\\\n  b\n", ["a b"], id="the-next-line-keeps-its-indent"),
        pytest.param(
            "a\\\n#b\nc\n", ["a#b", "c"], id="a-hash-inside-a-word-opens-no-comment"
        ),
        pytest.param(
            "a\\\n  #b\nc\n", ["a", "c"], id="a-hash-after-blanks-opens-a-comment"
        ),
        pytest.param(
            "a \\\\\nb\n", ["a \\\\", "b"], id="an-escaped-backslash-ends-the-line"
        ),
        pytest.param(
            "a \\ \nb\n", ["a \\", "b"], id="a-backslash-before-a-blank-ends-the-line"
        ),
        pytest.param("a \\", ["a \\"], id="a-backslash-at-the-end-of-the-script-stays"),
    ],
)
def test_logical_lines_read_continuations_as_bash_does(script, expected):
    assert _logical_lines(script) == expected


@settings(max_examples=120, deadline=None)
@given(
    st.lists(
        st.sampled_from(
            [
                "echo a",
                "echo b \\",
                "echo e\\",
                "echo f \\\\",
                "#;echo g",
                "# note",
                "# note \\",
                "   # indented \\",
                "",
                "  echo c",
                "echo d # trailing",
            ]
        ),
        max_size=8,
    )
)
def test_logical_lines_run_what_bash_runs(lines):
    def run(text: str) -> str:
        return subprocess.run(
            ["/bin/bash", "-c", text], capture_output=True, text=True, check=True
        ).stdout

    script = "\n".join(lines) + "\n"

    assert run("\n".join(_logical_lines(script)) + "\n") == run(script)


def test_bash_ends_a_comment_at_its_newline():
    result = subprocess.run(
        ["/bin/bash", "-c", "# a comment \\\necho ran\n"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == "ran\n"


@pytest.mark.parametrize("workflow", STAGING)
def test_a_command_glued_to_a_continuation_breaks_the_canonical_block(workflow):
    step = _staging_step(workflow)
    assignment = f'site_packages="{SITE_PACKAGES}"'
    assert step["run"].count(assignment) == 1
    step["run"] = step["run"].replace(assignment, f"{assignment}\\\n#;echo hidden")

    assert _staging_problems(step) == ["its lines differ from the canonical block"]


@pytest.mark.parametrize("workflow", STAGING)
def test_a_command_behind_a_comment_breaks_the_canonical_block(workflow):
    step = _staging_step(workflow)
    step["run"] += '# Drop one package. \\\nrm -rf "$site_packages/requests"\n'

    assert _staging_problems(step) == ["its lines differ from the canonical block"]


def test_every_production_provision_consumes_a_locked_tree():
    violations, provisions = _provision_violations(_load_workflows())

    assert violations == []
    assert provisions == len(STAGING) + len(LOCKED_VENV_PROVISIONS)


def test_only_their_installs_and_provisioners_name_the_protected_trees():
    violations, consumers = _tree_violations(_load_workflows())

    assert violations == []
    assert consumers == EXPECTED_TREE_CONSUMERS


def test_workflows_with_either_extension_are_audited(tmp_path):
    for name in ("added.yml", "added.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump(
                {
                    "jobs": {
                        "stage": {
                            "steps": [
                                {"run": f'uv pip install --target "{SITE_PACKAGES}" .'}
                            ]
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    violations, found = _target_install_violations(_load_workflows(tmp_path))

    assert found == {"added.yaml": 1, "added.yml": 1}
    assert len(violations) == 2


# Installs and removals a later edit could add after staging. Each must be caught.
UNLOCKED_INSTALLS = {
    "pip3": f'pip3 install --upgrade --target "{SITE_PACKAGES}" ".[api]"',
    "python-m-pip-attached-t": f'python -m pip install -t"{SITE_PACKAGES}" .',
    "versioned-python-target-eq": "python3.13 -m pip install --target=/opt/x requests",
    "pip-short-cluster": f'pip install -qt "{SITE_PACKAGES}" requests',
    "pip-target-from-the-environment": (
        f'PIP_TARGET="{SITE_PACKAGES}" "$PYBIN" -m pip install --upgrade claude-agent-sdk'
    ),
    "pip-uninstall": "python3 -m pip uninstall -y requests",
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
    "uv-uninstall": f'uv pip uninstall --target "{SITE_PACKAGES}" requests',
    "eval-pip": f"eval 'pip install --target \"{SITE_PACKAGES}\" requests'",
    "variable-interpreter": f'"$PYBIN" -m pip install --quiet --target "{SITE_PACKAGES}" .',
}


@pytest.mark.parametrize("workflow", STAGING)
@pytest.mark.parametrize("mutation", sorted(UNLOCKED_INSTALLS))
@pytest.mark.parametrize("placement", ["end-of-staging", "start-of-provision"])
def test_an_unlocked_install_after_staging_is_rejected(workflow, mutation, placement):
    workflows = _load_workflows()
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


# What the verification provisioner requires, beyond the trees a case varies.
PROVISION_ARGUMENTS = (
    "--supervisor s --apply-root a --eval-root e --corpus-release-root c --git g"
)

# Writes into a protected tree that name it without installing anything.
TREE_WRITES = {
    "remove-a-package": f'rm -rf "{SITE_PACKAGES}/requests"',
    "braced-runner-temp": (
        'rm -rf "${RUNNER_TEMP}/axiom-verification-site-packages/requests"'
    ),
    "copy-in": f'cp evil.py "{SITE_PACKAGES}/"',
    "redirect-into": f'echo "import os" > "{SITE_PACKAGES}/evil.pth"',
    "redirect-only": f'> "{SITE_PACKAGES}/requests/__init__.py"',
    "exec-redirect": f'exec 3> "{SITE_PACKAGES}/evil.pth"',
    "alias": f'tree="{SITE_PACKAGES}"',
    "export-to-later-steps": f'echo "TREE={SITE_PACKAGES}" >> "$GITHUB_ENV"',
    "second-mention-in-the-provision": (
        f"sudo .venv/bin/python scripts/{VERIFICATION_PROVISIONER} --site-packages "
        f'"{SITE_PACKAGES}" --destination "{SITE_PACKAGES}/copy" {PROVISION_ARGUMENTS}'
    ),
    "loop-header": (
        f'for cache in "{SITE_PACKAGES}"/*/__pycache__; do rm -rf "$cache"; done'
    ),
    "array": f'trees=("{SITE_PACKAGES}")',
    "array-append": f'trees+=("{SITE_PACKAGES}")',
    "python-heredoc": (
        "python3 - <<'PY'\n"
        "import os, shutil\n"
        'shutil.rmtree(os.path.join(os.environ["RUNNER_TEMP"],'
        ' "axiom-verification-site-packages", "requests"))\n'
        "PY"
    ),
    "wrapper-option": f'env -C "{SITE_PACKAGES}" rm -rf requests',
}


@pytest.mark.parametrize("workflow", STAGING)
@pytest.mark.parametrize("mutation", sorted(TREE_WRITES))
def test_a_write_into_the_staged_tree_is_rejected(workflow, mutation):
    workflows = _load_workflows()
    step = _provision_step(workflow, workflows)
    step["run"] = f"{TREE_WRITES[mutation]}\n{step['run']}"

    assert _new_violations(workflows), f"{TREE_WRITES[mutation]!r} was not caught"


def test_a_consumer_must_name_its_tree_in_its_text():
    # The reader joins the quoted pieces, so only the step's text shows the split.
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    split = SITE_PACKAGES.replace("-site-", '-site""-')
    assert step["run"].count(SITE_PACKAGES) == 1
    step["run"] = step["run"].replace(SITE_PACKAGES, split)

    violations, _ = _tree_violations(workflows)

    assert any("names axiom-verification-site-packages 0 time" in v for v in violations)


def test_a_split_consumer_cannot_lend_its_count_to_other_text():
    # The loop header's mention balances the text count the split consumer lost.
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    split = SITE_PACKAGES.replace("-site-", '-site""-')
    loop = f'for t in "{SITE_PACKAGES}"/*; do rm -rf "$t"; done'
    assert step["run"].count(SITE_PACKAGES) == 1
    step["run"] = f"{loop}\n{step['run'].replace(SITE_PACKAGES, split)}"

    violations, _ = _tree_violations(workflows)

    assert any(
        "splits axiom-verification-site-packages across quotes" in v for v in violations
    ), violations


def test_a_provision_that_names_its_tree_twice_does_not_consume_it():
    (command, *_) = workflow_shell.commands(
        TREE_WRITES["second-mention-in-the-provision"]
    )

    assert _provision(command)[1] is not None
    assert not _consumes(command, "axiom-verification-site-packages")


GITHUB_TREE = "${{ runner.temp }}/axiom-verification-site-packages"
YAML_TREE_KEYS = {
    "step-env": lambda _workflow, _job, step: step.setdefault("env", {}).update(
        TREE=GITHUB_TREE
    ),
    "working-directory": lambda _workflow, _job, step: step.update(
        {"working-directory": GITHUB_TREE}
    ),
    "cache-path": lambda _workflow, job, _step: job["steps"].insert(
        0, {"uses": "actions/cache@0", "with": {"path": GITHUB_TREE}}
    ),
    "job-env": lambda _workflow, job, _step: job.setdefault("env", {}).update(
        TREE=GITHUB_TREE
    ),
    "workflow-env": lambda workflow, _job, _step: workflow.setdefault("env", {}).update(
        TREE=GITHUB_TREE
    ),
}


@pytest.mark.parametrize("workflow", STAGING)
@pytest.mark.parametrize("key", sorted(YAML_TREE_KEYS))
def test_a_yaml_key_naming_a_protected_tree_is_rejected(workflow, key):
    workflows = _load_workflows()
    step = _provision_step(workflow, workflows)
    job = next(
        job
        for job in workflows[workflow]["jobs"].values()
        if any(candidate is step for candidate in job.get("steps", []))
    )
    YAML_TREE_KEYS[key](workflows[workflow], job, step)

    assert _new_violations(workflows), f"{key} was not caught"


@pytest.mark.parametrize("key", ["comment", "step-name"])
def test_a_comment_or_step_name_may_name_a_protected_tree(key):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    if key == "comment":
        step["run"] = f"# Reads {SITE_PACKAGES}.\n{step['run']}"
    else:
        step["name"] = f"Provision from {SITE_PACKAGES}"

    assert not _new_violations(workflows)


@pytest.mark.parametrize(
    "line",
    [
        'rm -rf "$RUNNER_TEMP/axiom-compose-site-packages/yaml"',
        'cp evil.pth "$RUNNER_TEMP/axiom-compose-site-packages/"',
        'uv pip uninstall --target "$RUNNER_TEMP/axiom-compose-site-packages" pyyaml',
    ],
)
def test_a_write_into_the_compose_runtime_is_rejected(line):
    workflows = _load_workflows()
    step = _compose_step(workflows)
    # Between the locked install and the provision.
    marker = "sudo rm -rf /opt/axiom-compose-verification\n"
    assert step["run"].count(marker) == 1
    step["run"] = step["run"].replace(marker, f"{line}\n{marker}")

    assert _new_violations(workflows), f"{line!r} was not caught"


def test_a_compose_provision_of_another_tree_is_rejected():
    workflows = _load_workflows()
    step = _compose_step(workflows)
    locked = '--site-packages "$RUNNER_TEMP/axiom-compose-site-packages"'
    assert step["run"].count(locked) == 1
    step["run"] = step["run"].replace(locked, '--site-packages "$RUNNER_TEMP/other"')

    violations, _ = _provision_violations(workflows)

    assert any("does not read its locked tree" in v for v in violations), violations


def test_the_compose_runtime_install_must_stay_hash_required():
    workflows = _load_workflows()
    step = _compose_step(workflows)
    assert "--require-hashes --no-deps" in step["run"]
    step["run"] = step["run"].replace("--require-hashes --no-deps ", "")

    violations, _ = _target_install_violations(workflows)

    assert any("axiom-compose-site-packages" in v for v in violations), violations


@pytest.mark.parametrize(
    "mutation",
    [
        'uv pip install --python "$python_path" requests',
        'pip install --prefix "$GITHUB_WORKSPACE/axiom-encode/.venv" requests',
        "(cd axiom-encode && uv add requests)",
        'uv pip uninstall --python "$python_path" pyyaml',
        '"$python_path" -m pip uninstall -y pyyaml',
        "(cd axiom-encode && uv remove pyyaml)",
        "(cd axiom-encode && uv venv --clear)",
        "(cd axiom-encode && uv sync --frozen --extra dev)",
        'rm -rf "$site/yaml"',
        'rm -rf "${site}/yaml"',
        'echo "import os" > "$site/evil.pth"',
        'cp evil.py "$site/"',
        'for hook in "$site"/*.pth; do rm "$hook"; done',
        "rm -rf axiom-encode/.venv/lib/python3.14/site-packages/yaml",
        "rm -rf axiom-encode/.venv",
        'site="$RUNNER_TEMP/unlocked"',
        'export site="$RUNNER_TEMP/unlocked"',
        'declare site="$RUNNER_TEMP/unlocked"',
        'true; site="$RUNNER_TEMP/unlocked"',
        "site+=/unlocked",
    ],
)
def test_a_locked_venv_provision_rejects_any_change_to_the_environment(mutation):
    workflows = _load_workflows()
    name = next(iter(LOCKED_VENV_PROVISIONS))
    step = _provision_step(name, workflows)
    step["run"] = f"{mutation}\n{step['run']}"

    violations, _ = _provision_violations(workflows)

    assert any("changes the locked environment" in v for v in violations), violations


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(
            '# python_path="{python}"\npython_path="$RUNNER_TEMP/unlocked/bin/python"',
            id="a-comment-holds-the-locked-interpreter",
        ),
        pytest.param(
            'python_path="{python}"\npython_path="$RUNNER_TEMP/unlocked/bin/python"',
            id="a-second-assignment",
        ),
        pytest.param(
            'python_path="{python}"\n'
            'if [ ! -x "$python_path" ]; then python_path="$(command -v python3)"; fi',
            id="a-fallback-assignment",
        ),
        pytest.param(
            'python_path="{python}"\nexport python_path=/usr/bin/python3',
            id="an-exported-assignment",
        ),
        pytest.param(
            'python_path="{python}"\npython_path+="x"',
            id="an-appended-assignment",
        ),
    ],
)
def test_a_locked_venv_provision_computes_the_purelib_with_its_interpreter(replacement):
    workflows = _load_workflows()
    name, python = next(iter(LOCKED_VENV_PROVISIONS.items()))
    step = _provision_step(name, workflows)
    assignment = f'python_path="{python}"'
    assert step["run"].count(assignment) == 1
    step["run"] = step["run"].replace(assignment, replacement.format(python=python))

    violations, _ = _provision_violations(workflows)

    assert any(
        "changes the locked environment" in v or "purelib" in v for v in violations
    ), violations


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        pytest.param(
            'sudo "$python_path"', "sudo python3", id="another-interpreter-runs-it"
        ),
        pytest.param(
            PURELIB,
            f'# {PURELIB}\nsite="$RUNNER_TEMP/unlocked"',
            id="a-comment-holds-the-purelib-line",
        ),
    ],
)
def test_a_locked_venv_provision_reads_its_purelib_with_its_python(
    original, replacement
):
    workflows = _load_workflows()
    name = next(iter(LOCKED_VENV_PROVISIONS))
    step = _provision_step(name, workflows)
    assert step["run"].count(original) == 1
    step["run"] = step["run"].replace(original, replacement)

    violations, _ = _provision_violations(workflows)

    assert any("purelib" in v for v in violations), violations


@pytest.mark.parametrize(
    "harmless",
    [
        pytest.param({"name": "Report from the locked .venv"}, id="a-step-name"),
        pytest.param(
            {"run": 'curl "https://example.invalid/search?site=laws"'}, id="a-url-query"
        ),
        pytest.param({"run": "tool --site=laws"}, id="another-tools-option"),
        pytest.param(
            {"run": 'ls -l "$GITHUB_WORKSPACE/axiom-encode/.venv/bin"'},
            id="the-environments-bin-directory",
        ),
    ],
)
def test_a_locked_venv_job_may_do_what_cannot_touch_the_environment(harmless):
    workflows = _load_workflows()
    name = next(iter(LOCKED_VENV_PROVISIONS))
    step = _provision_step(name, workflows)
    job = next(
        job
        for job in workflows[name]["jobs"].values()
        if any(candidate is step for candidate in job["steps"])
    )
    job["steps"].insert(0, {"run": "true", **harmless})

    violations, _ = _provision_violations(workflows)

    assert violations == []


@pytest.mark.parametrize(
    ("arguments", "caught"),
    [
        pytest.param(
            f'--site-packages "{SITE_PACKAGES}"',
            '--site-packages "$RUNNER_TEMP/other"',
            id="replaced",
        ),
        pytest.param(
            f'--site-packages "{SITE_PACKAGES}"',
            f'--site-packages "{SITE_PACKAGES}" --site-packages "$RUNNER_TEMP/other"',
            id="a-later-duplicate-wins",
        ),
        pytest.param(
            f'--site-packages "{SITE_PACKAGES}"',
            f'--site-packages "{SITE_PACKAGES}" --site-pack "$RUNNER_TEMP/other"',
            id="an-abbreviation-wins",
        ),
        pytest.param(
            f'--site-packages "{SITE_PACKAGES}"',
            f'--site-packages "{SITE_PACKAGES}" --site-packages=$RUNNER_TEMP/other',
            id="an-attached-duplicate-wins",
        ),
    ],
)
def test_a_provision_of_an_unlocked_tree_is_rejected(arguments, caught):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    assert step["run"].count(arguments) == 1
    step["run"] = step["run"].replace(arguments, caught)

    violations, _ = _provision_violations(workflows)

    assert any("$RUNNER_TEMP/other is not locked" in v for v in violations), violations


def test_a_provision_uses_the_value_its_parser_keeps():
    # An earlier, discarded --site-packages is harmless: argparse keeps the last.
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    locked = f'--site-packages "{SITE_PACKAGES}"'
    step["run"] = step["run"].replace(
        locked, f'--site-packages "$RUNNER_TEMP/other" {locked}'
    )

    violations, provisions = _provision_violations(workflows)

    assert violations == []
    assert provisions == len(STAGING) + len(LOCKED_VENV_PROVISIONS)


def test_reviewed_selector_array_preserves_both_provision_argument_sets():
    workflows = _load_workflows()
    step = _provision_step("targeted-signed-reencode.yml", workflows)
    commands = _commands("targeted-signed-reencode.yml", "encode", step["run"])
    parsed = [result for command in commands if (result := _provision(command))]
    assert len(parsed) == 1
    assert parsed[0][1].reviewed_pr_selector == Path(
        "$RUNNER_TEMP/reviewed-pr-selector.json"
    )
    assert str(parsed[0][1].site_packages) == SITE_PACKAGES
    disabled_script = step["run"].replace('"${reviewed_pr_args[@]}"', "")
    disabled = [
        result
        for command in _commands(
            "targeted-signed-reencode.yml", "encode", disabled_script
        )
        if (result := _provision(command))
    ]
    assert len(disabled) == 1
    assert disabled[0][1].reviewed_pr_selector is None
    assert str(disabled[0][1].site_packages) == SITE_PACKAGES


@pytest.mark.parametrize(
    "old,new",
    [
        ("reviewed_pr_args=()", "reviewed_pr_args=(--site-packages /tmp/unlocked)"),
        (
            "reviewed_pr_args=()",
            "reviewed_pr_args=()\nreviewed_pr_args+=(--unknown value)",
        ),
        ("reviewed_pr_args=()", ""),
        ('"${reviewed_pr_args[@]}"', '"${unknown_args[@]}"'),
        ('"${reviewed_pr_args[@]}"', '"${reviewed_pr_args[*]}"'),
        ('"${reviewed_pr_args[@]}"', '"${reviewed_pr_args[@]}" --unknown value'),
        (
            "reviewed_pr_args+=(--reviewed-pr-selector",
            "reviewed_pr_args+=(--site-packages",
        ),
        ("reviewed_pr_args+=(--reviewed-pr-selector", "reviewed_pr_args+=(--unknown"),
        ('if [ "$LIVE_PR_ADMISSION" = "true" ]; then', "if false; then"),
    ],
)
def test_unrecognized_reviewed_selector_array_still_fails_closed(old, new):
    workflows = _load_workflows()
    step = _provision_step("targeted-signed-reencode.yml", workflows)
    assert old in step["run"]
    step["run"] = step["run"].replace(old, new)
    violations, _ = _provision_violations(workflows)
    assert any("provisioner refuses its arguments" in item for item in violations)


@pytest.mark.parametrize(
    "interpreter",
    [
        ".venv/bin/python scripts/",
        "python3.13 -I -S scripts/",
        '"$PYBIN" -X utf8 scripts/',
        "python3 -Wignore scripts/",
        "python3 -- scripts/",
        "./scripts/",
        "$HOME/bin/",
        '"$SUDO" .venv/bin/python scripts/',
        "$SUDO .venv/bin/python scripts/",
        "chronic .venv/bin/python scripts/",
        "retry .venv/bin/python scripts/",
        "coverage run scripts/",
        "python3 -m coverage run scripts/",
        "python3 -m pdb scripts/",
        "pypy3 scripts/",
        "python3-dbg scripts/",
        ".venv/bin/$py scripts/",
    ],
)
def test_every_way_of_running_the_provisioner_is_a_provision(interpreter):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    line = (
        f"{interpreter}{VERIFICATION_PROVISIONER} --destination d "
        f'--site-packages "$RUNNER_TEMP/other" {PROVISION_ARGUMENTS}'
    )
    step["run"] = f"{line}\n{step['run']}"

    violations, _ = _provision_violations(workflows)

    assert any("$RUNNER_TEMP/other is not locked" in v for v in violations), violations


@pytest.mark.parametrize(
    "option",
    [
        pytest.param('--site-packages="$RUNNER_TEMP/other"', id="attached"),
        pytest.param('--site-pack "$RUNNER_TEMP/other"', id="abbreviated"),
    ],
)
def test_a_wrapped_provisioner_counts_under_any_spelling_of_its_option(option):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    line = (
        f"chronic .venv/bin/python scripts/{VERIFICATION_PROVISIONER} "
        f"--destination d {option} {PROVISION_ARGUMENTS}"
    )
    step["run"] = f"{line}\n{step['run']}"

    violations, _ = _provision_violations(workflows)

    assert any("$RUNNER_TEMP/other is not locked" in v for v in violations), violations


@pytest.mark.parametrize(
    "module",
    [
        "python3 -m scripts.provision_verification_supervisor",
        "python3 -Imscripts.provision_verification_supervisor",
    ],
)
def test_running_the_provisioner_as_a_module_is_a_provision(module):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    line = (
        f'{module} --destination d --site-packages "$RUNNER_TEMP/other" '
        f"{PROVISION_ARGUMENTS}"
    )
    step["run"] = f"{line}\n{step['run']}"

    violations, _ = _provision_violations(workflows)

    assert any("$RUNNER_TEMP/other is not locked" in v for v in violations), violations


@pytest.mark.parametrize(
    "line",
    [
        f"sha256sum scripts/{VERIFICATION_PROVISIONER}",
        f"git diff --quiet -- scripts/{VERIFICATION_PROVISIONER}",
        f"python -m ruff check scripts/{VERIFICATION_PROVISIONER}",
        f"python3 tools/lint.py scripts/{VERIFICATION_PROVISIONER}",
    ],
)
def test_reading_the_provisioner_is_not_a_provision(line):
    workflows = _load_workflows()
    step = _provision_step("bulk-encode.yml", workflows)
    step["run"] = f"{line}\n{step['run']}"

    violations, provisions = _provision_violations(workflows)

    assert violations == []
    assert provisions == len(STAGING) + len(LOCKED_VENV_PROVISIONS)


@pytest.mark.parametrize(
    "condition",
    [
        "${{ always() }}",
        "${{ !cancelled() }}",
        "failure()",
        "${{ Always() }}",
        "${{ always () }}",
        "${{ always( ) }}",
        "${{ success() || inputs.force_provision }}",
        "${{ !success() }}",
        "success() || steps.diff.outcome == 'failure'",
    ],
)
def test_a_provision_never_runs_after_a_failed_staging_step(condition):
    workflows = _load_workflows()
    _provision_step("bulk-encode.yml", workflows)["if"] = condition

    violations, _ = _provision_violations(workflows)

    assert any("after a failed step" in v for v in violations), violations


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        pytest.param(
            "uv run --frozen --no-sync python -c 'print(1)' --python",
            ("run", {"--frozen": [""], "--no-sync": [""]}, ()),
            id="launched-arguments-stay-with-the-launched-command",
        ),
        pytest.param(
            'uv --directory export export --locked --output-file "$f"',
            (
                "export",
                {"--directory": ["export"], "--locked": [""], "--output-file": ["$f"]},
                (),
            ),
            id="a-global-value-is-not-the-subcommand",
        ),
        pytest.param(
            'uv pip install -qt"$t" --no-deps -r a -rb -- c',
            (
                "pip install",
                {
                    "--quiet": [""],
                    "--target": ["$t"],
                    "--no-deps": [""],
                    "--requirements": ["a", "b"],
                },
                ("c",),
            ),
            id="clusters-attached-values-and-double-dash",
        ),
    ],
)
def test_uv_arguments_come_from_the_shared_parser(script, expected):
    (command, *_) = workflow_shell.commands(script)

    assert _uv_arguments(command) == expected


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
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    log = tmp_path / "uv-argv.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
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


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_runs_the_locked_sequence(tmp_path, workflow):
    result, calls, runner_temp = _run_staging_step(tmp_path, workflow)

    assert result.returncode == 0, result.stderr
    site_packages = str(runner_temp / "axiom-verification-site-packages")
    locked = str(runner_temp / "axiom-verification-locked-requirements.txt")
    python = ".venv/bin/python"
    install = ["pip", "install", "--python", python, "--target", site_packages]
    export = ["export", "--locked", "--no-dev", "--extra", "api", "--no-emit-project"]
    assert calls == [
        ["sync", "--locked", "--python", "3.13", "--extra", "api", "--no-dev"],
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
        ["pip", "freeze", "--python", python, "--exclude-editable"],
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


@pytest.mark.parametrize("workflow", STAGING)
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


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_fails_when_the_staged_tree_drifts_from_the_lock(
    tmp_path, workflow
):
    drifted = LOCKED_FREEZE.replace("0.1.47", "0.2.159")
    result, calls, _ = _run_staging_step(tmp_path, workflow, staged_freeze=drifted)

    assert result.returncode != 0
    assert "+claude-agent-sdk==0.2.159" in result.stdout
    assert calls[-1][:2] == ["pip", "freeze"]


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_fails_when_the_lock_cannot_be_exported(tmp_path, workflow):
    result, calls, _ = _run_staging_step(tmp_path, workflow, export_fails=True)

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)
