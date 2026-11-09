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
``--target=DIR`` and redirections are all seen. In every workflow:

- pip installs and uninstalls nothing. It ignores ``uv.lock``, and
  ``PIP_TARGET`` or a pip config file can aim it at a tree the command never
  names.
- ``uv pip uninstall`` never runs.
- Every ``--target`` install is a ``uv pip install --no-deps`` of a hash-pinned
  ``uv export --locked`` output written in the same step (with
  ``--require-hashes``), the commit-pinned ``axiom-oracles`` requirement, or
  the project. It passes ``--no-config``, so no user or parent-directory
  ``uv.toml`` changes how it installs or builds. It builds nothing with
  backends ``uv.lock`` does not pin: it passes ``--no-build``, or builds
  uncached (``--no-cache``) and isolated against ``--build-constraints``
  naming the hash-pinned export of the lock's ``build`` group written in the
  same step. Nothing in that step names that file but its one assignment, the
  export and the installs that read it; no redirection, copy, alias or second
  assignment touches it.
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
  and nothing else in it names the purelib, assigns ``site`` again or names the
  environment outside its ``bin/`` directory.
- No provision runs after a failed step (``always()``, ``failure()`` or
  ``cancelled()`` in its ``if:``), and a staging step carries no key that
  would let its ``diff`` fail without stopping the job.

Only the provisions in the provisioner's selftest escape these last two rules;
its throwaway tree verifies and signs nothing.

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

# The provisioner's own selftest provisions a throwaway tree (pyyaml plus hostile
# startup-hook fixtures) under a stand-in supervisor with test-only trust roots;
# nothing it provisions verifies or signs anything. It is exempt from the
# provision rule only: its one --target install is locked like the rest.
EXEMPT_WORKFLOWS = frozenset({"provision-selftest.yml"})

SITE_PACKAGES = "$RUNNER_TEMP/axiom-verification-site-packages"
BUILD_CONSTRAINTS = "$RUNNER_TEMP/axiom-verification-build-constraints.txt"
BUILD_EXPORT_OPTIONS = {
    "--no-config",
    "--locked",
    "--only-group",
    "--no-emit-project",
    "--format",
    "--output-file",
}
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
# stages the axiom-compose runtime, and the provisioner selftest its pyyaml.
EXPECTED_TARGET_INSTALLS = {
    "bulk-encode.yml": 3,
    "golden-regeneration.yml": 3,
    "provision-selftest.yml": 1,
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
PURELIB_NAME = re.compile(r"\$\{?site\b|\.venv(?!/bin/)")
PURELIB_ASSIGNMENT = re.compile(r"\bsite\+?=")
# A step with one of these in its `if:` runs after an earlier step failed.
RUNS_AFTER_FAILURE = re.compile(r"always\(\)|failure\(\)|cancelled\(\)", re.IGNORECASE)


def _logical_lines(script: str) -> list[str]:
    """``script``'s lines, with continuations joined as bash joins them: a
    comment ends at its newline, even after a trailing backslash, and a blank
    line ends a continuation."""
    lines: list[str] = []
    pending = ""
    for raw in script.splitlines():
        text = raw.strip()
        comment = text.startswith("#")
        if text.endswith("\\") and not comment:
            pending += f"{text[:-1]} "
            continue
        logical = pending if comment else f"{pending}{text}"
        pending = ""
        if logical.strip():
            lines.append(re.sub(r"\s+", " ", logical.strip()))
    if pending.strip():
        lines.append(re.sub(r"\s+", " ", pending.strip()))
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
    return list(_script_commands(script, name, job))


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


def _locked_build_exports(commands: list[Command]) -> set[str]:
    """Output files of hash-pinned exports of the lock's ``build`` group.

    The options must be exactly the export's own: uv accepts ``--no-emit-package``,
    ``--prune`` and ``--no-group`` here and writes a file that omits backends, and
    uv resolves whatever a constraints file omits freely.
    """
    exported = set()
    for command in commands:
        parsed = _uv_arguments(command)
        if parsed is None or parsed[0] != "export":
            continue
        _, values, _ = parsed
        if (
            values.keys() == BUILD_EXPORT_OPTIONS
            and values["--only-group"] == ["build"]
            and values["--format"] == ["requirements-txt"]
        ):
            exported.add(values["--output-file"][-1])
    return exported


def _build_constraints_tampered(
    commands: list[Command], build_exported: set[str], script: str
) -> bool:
    """Whether anything but the locked export writes, swaps or aliases the
    build constraints.

    The file may be named by its variable, in any expansion, or by its literal
    path. A command may name it only as its one bare assignment, as the
    ``--output-file`` of a locked ``build`` group export, or as the
    ``--build-constraints`` value of an install, spelled with the variable. No
    redirection may name it: ``> "$RUNNER_TEMP/...-build-constraints.txt"`` on
    any command, even the install that reads it, or a line that only redirects,
    such as ``exec 3> "$build_constraints"``, empties it before uv reads it.

    The step's text outside full-line comments holds the variable's name only
    in those places and the literal path only in that assignment. So a command
    the reader drops (a loop header) or reads as plain words (``export``,
    ``read``, ``printf -v``) cannot rebind it.
    """
    names = {out.lstrip("$").strip("{}") for out in build_exported}
    if not names:
        return False
    literals = {
        assignment.partition("=")[2].rpartition("/")[2]
        for command in commands
        for assignment in command.assignments
        if assignment.partition("=")[0] in names
    } - {""}
    alternatives = "|".join(re.escape(name) for name in sorted(names))
    expansion = re.compile(rf"\$\{{?(?:{alternatives})\b")
    name_in_text = re.compile(rf"\b(?:{alternatives})\b")

    def names_it(text: str) -> bool:
        return bool(expansion.search(text)) or any(lit in text for lit in literals)

    definitions = consumers = 0
    for command in commands:
        if any(names_it(target) for _, target in command.redirects):
            return True
        for assignment in command.assignments:
            name, _, value = assignment.partition("=")
            if name in names:
                definitions += 1
                if command.words or len(command.assignments) != 1:
                    return True  # not a bare `build_constraints=...` line
            elif names_it(value):
                return True  # an alias
        words = [word for word in command.words if names_it(word)]
        if not words:
            continue
        # A line that names the file and redirects anything is refused
        # outright, in case a redirection reads as something else.
        if re.search(r"[<>]", command.line):
            return True
        parsed = _uv_arguments(command)
        if parsed is None:
            return True
        subcommand, values, _ = parsed
        if subcommand == "export":
            if _locked_build_exports([command]) and len(words) == len(
                values["--output-file"]
            ):
                consumers += len(words)
                continue
            return True
        constraints = values.get("--build-constraints", [])
        if (
            subcommand == "pip install"
            and constraints
            and set(constraints) <= build_exported
            and len(words) == len(constraints)
            and not any(literal in word for word in words for literal in literals)
        ):
            consumers += len(words)
            continue
        return True
    lines = list(_code_lines([script]))
    return (
        definitions != 1
        or sum(len(name_in_text.findall(line)) for line in lines)
        != definitions + consumers
        or sum(line.count(lit) for line in lines for lit in literals) != definitions
    )


def _build_problem(
    values: dict[str, list[str]], build_exported: set[str], tampered: bool
) -> str | None:
    """Why a ``--target`` install could build with backends uv.lock does not
    pin, if it could."""
    if values.keys() & {"--no-build-isolation", "--no-build-isolation-package"}:
        # A build without isolation takes whatever backends the environment
        # has, not the constrained ones.
        return "builds without isolation"
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
    if "--no-build" in values:
        return None
    return "may build with unpinned backends (no --build-constraints or --no-build)"


def _target_install_violations(
    workflows: dict[str, dict], exempt: frozenset[str] = frozenset()
) -> tuple[list[str], dict[str, int]]:
    violations: list[str] = []
    found: dict[str, int] = defaultdict(int)
    for name, job, index, step in _steps(workflows):
        if name in exempt:
            continue
        where = f"{name}:{job}[{index}]"
        commands = _commands(name, job, step["run"])
        exported = _locked_exports(commands)
        build_exported = _locked_build_exports(commands)
        tampered = _build_constraints_tampered(commands, build_exported, step["run"])
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
                elif not set(requirements) <= exported - build_exported:
                    problem = (
                        f"{requirements} is not a hash-pinned `uv export --locked` "
                        "output written in this step"
                    )
            elif positionals == ("$oracles",):
                if f"grep -xE '{ORACLES_PATTERN}'" not in step["run"]:
                    problem = "$oracles is not the commit-pinned axiom-oracles line"
            elif positionals != (".",):
                problem = f"installs unpinned requirements {list(positionals)}"
            if problem is None and "--no-config" not in values:
                problem = "reads user or parent-directory uv config (no --no-config)"
            if problem is None:
                problem = _build_problem(values, build_exported, tampered)
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
        if program is None:
            return None
        script, arguments = _provisioner(program[0]), program[1]
    if script is None:
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
    """Whether ``job``'s text, comments aside, names the purelib beyond its one
    assignment and its one provision, or the environment outside ``bin/``."""
    lines = list(_code_lines(_strings(job)))
    return (
        sum(len(PURELIB_NAME.findall(line)) for line in lines) != 1
        or sum(len(PURELIB_ASSIGNMENT.findall(line)) for line in lines) != 1
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
                if (
                    f'python_path="{python}"' not in run
                    or PURELIB not in _logical_lines(run)
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
    workflows: dict[str, dict], exempt: frozenset[str] = frozenset()
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
            if sum(line.count(tree) for line in lines) > consumed[tree]:
                violations.append(
                    f"{where}: its text names {tree} outside its locked install "
                    "and provisioner"
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
    install = (
        'uv pip install --no-config --python .venv/bin/python --target "$site_packages"'
        ' --no-cache --build-constraints "$build_constraints"'
    )
    return [
        "set -euo pipefail",
        "uv sync --no-config --locked --python 3.13 --extra api --no-dev",
        f'site_packages="{SITE_PACKAGES}"',
        'locked="$RUNNER_TEMP/axiom-verification-locked-requirements.txt"',
        f'build_constraints="{BUILD_CONSTRAINTS}"',
        'rm -rf "$site_packages"',
        "uv export --no-config --locked --no-dev --extra api --no-emit-project"
        " --no-emit-package axiom-oracles --format requirements-txt"
        ' --output-file "$locked"',
        "uv export --no-config --locked --only-group build --no-emit-project"
        ' --format requirements-txt --output-file "$build_constraints"',
        'oracles="$(uv export --no-config --locked --no-dev --extra api'
        " --no-emit-project --no-hashes --no-annotate --no-header"
        " --format requirements-txt"
        f" | grep -xE '{ORACLES_PATTERN}')\"",
        f'{install} --require-hashes --no-deps -r "$locked"',
        f'{install} --no-deps "$oracles"',
        f"{install} --no-deps .",
        "uv pip freeze --no-config --python .venv/bin/python --exclude-editable"
        ' > "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        "uv pip freeze --no-config --python .venv/bin/python"
        ' --target "$site_packages" --exclude axiom-encode'
        ' > "$RUNNER_TEMP/axiom-verification-staged.freeze"',
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


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_is_exactly_the_locked_install(workflow):
    step = _staging_step(workflow)

    assert _logical_lines(step["run"]) == _canonical_staging()
    # `continue-on-error` or `if:` would let a job provision a tree that failed
    # the diff.
    assert set(step) <= {"name", "run", "working-directory"}


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
    ],
)
def test_logical_lines_read_continuations_as_bash_does(script, expected):
    assert _logical_lines(script) == expected


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.sampled_from(
            [
                "echo a",
                "echo b \\",
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
def test_a_command_behind_a_comment_breaks_the_canonical_block(workflow):
    run = _staging_step(workflow)["run"]
    hidden = f'{run}# Drop one package. \\\nrm -rf "$site_packages/requests"\n'

    assert _logical_lines(hidden) != _canonical_staging()


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
    "uv-builds-without-isolation": (
        f'uv pip install --no-config --no-cache --target "{SITE_PACKAGES}"'
        ' --no-deps --build-constraints "$build_constraints"'
        " --no-build-isolation ."
    ),
    "uv-installs-the-build-backends": (
        f'uv pip install --no-config --target "{SITE_PACKAGES}" --no-deps'
        ' --no-build --require-hashes -r "$build_constraints"'
    ),
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


# Writes, swaps and aliases of the build constraints, inserted between their
# export and the first install that reads them.
TAMPERING = {
    "truncate-literal": f': > "{BUILD_CONSTRAINTS}"',
    "copy-over-literal": f'cp /tmp/hatchling.txt "{BUILD_CONSTRAINTS}"',
    "install-redirects-literal": (
        f'uv pip install --no-config --python .venv/bin/python --target "{SITE_PACKAGES}"'
        ' --no-cache --build-constraints "$build_constraints" --no-deps .'
        f' > "{BUILD_CONSTRAINTS}"'
    ),
    "append-variable": 'echo "hatchling==0" >> "$build_constraints"',
    "braced-variable": 'cp /tmp/hatchling.txt "${build_constraints}"',
    "reassign": 'build_constraints="$RUNNER_TEMP/other.txt"',
    "alias": 'other="$build_constraints"',
    "another-group-into-the-file": (
        "uv export --no-config --locked --only-group dev --no-emit-project"
        ' --format requirements-txt --output-file "$build_constraints"'
    ),
    # A line that only redirects still truncates the file.
    "exec-redirect-variable": 'exec 3> "$build_constraints"',
    "exec-redirect-literal": f'exec 3> "{BUILD_CONSTRAINTS}"',
    "bare-redirect-variable": '> "$build_constraints"',
    "bare-redirect-literal": f'> "{BUILD_CONSTRAINTS}"',
    # Other expansions of the variable, and rebinding it without a plain
    # assignment.
    "truncate-guarded-expansion": ': > "${build_constraints:?}"',
    "copy-guarded-expansion": 'cp -f /etc/hosts "${build_constraints:?}"',
    "alias-guarded-expansion": 'other="${build_constraints:?}"',
    "export": "export build_constraints=/etc/hosts",
    "declare": "declare build_constraints=/etc/hosts",
    "readonly": "readonly build_constraints=/etc/hosts",
    "append-assignment": "build_constraints+=.bak",
    "read": "read -r build_constraints < /etc/hosts",
    "printf-v": "printf -v build_constraints %s /etc/hosts",
    "loop-header": "for build_constraints in /etc/hosts; do :; done",
    "loop-header-literal": f'for f in "{BUILD_CONSTRAINTS}"; do :; done',
}
BUILD_EXPORT_LINE = '  --format requirements-txt --output-file "$build_constraints"\n'


@pytest.mark.parametrize("workflow", STAGING)
@pytest.mark.parametrize("form", sorted(TAMPERING))
def test_writes_to_the_build_constraints_are_rejected(workflow, form):
    workflows = _load_workflows()
    step = _staging_step(workflow, workflows)
    assert step["run"].count(BUILD_EXPORT_LINE) == 1
    step["run"] = step["run"].replace(
        BUILD_EXPORT_LINE, f"{BUILD_EXPORT_LINE}{TAMPERING[form]}\n"
    )

    violations, _ = _target_install_violations(workflows)

    assert any("written outside their locked export" in v for v in violations), (
        violations
    )


@pytest.mark.parametrize("workflow", STAGING)
@pytest.mark.parametrize(
    "option",
    [
        "--no-emit-package hatchling",
        "--prune hatchling",
        "--no-group build",
        "--no-hashes",
        "--only-group dev",
        "--extra dev",
    ],
)
def test_a_build_export_that_changes_its_contents_is_not_the_locked_export(
    workflow, option
):
    workflows = _load_workflows()
    step = _staging_step(workflow, workflows)
    assert step["run"].count(BUILD_EXPORT_LINE) == 1
    step["run"] = step["run"].replace(
        BUILD_EXPORT_LINE, f"{BUILD_EXPORT_LINE.rstrip()} {option}\n"
    )

    violations, _ = _target_install_violations(workflows)

    assert any("is not the locked `build`" in v for v in violations), violations


def test_the_compose_runtime_install_builds_nothing():
    # axiom-compose's lock pins no build backend; its no-extras export is wheels
    # only, and --no-build makes a future sdist fail instead of building.
    workflows = _load_workflows()
    step = _compose_step(workflows)
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
    ["${{ always() }}", "${{ !cancelled() }}", "failure()", "${{ Always() }}"],
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
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    log = tmp_path / "uv-argv.jsonl"
    build_log = tmp_path / "uv-build-constraints.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
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


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_runs_the_locked_sequence(tmp_path, workflow):
    result, calls, runner_temp, seen = _run_staging_step(tmp_path, workflow)

    assert result.returncode == 0, result.stderr
    site_packages = str(runner_temp / "axiom-verification-site-packages")
    locked = str(runner_temp / "axiom-verification-locked-requirements.txt")
    build_constraints = str(runner_temp / "axiom-verification-build-constraints.txt")
    python = ".venv/bin/python"
    install = [
        *("pip", "install", "--no-config", "--python", python),
        *("--target", site_packages),
        *("--no-cache", "--build-constraints", build_constraints),
    ]
    export = [
        *("export", "--no-config", "--locked", "--no-dev"),
        *("--extra", "api", "--no-emit-project"),
    ]
    assert calls == [
        [
            *("sync", "--no-config", "--locked", "--python", "3.13"),
            *("--extra", "api", "--no-dev"),
        ],
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
            *("export", "--no-config", "--locked", "--only-group", "build"),
            *("--no-emit-project", "--format", "requirements-txt"),
            *("--output-file", build_constraints),
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
        ["pip", "freeze", "--no-config", "--python", python, "--exclude-editable"],
        [
            *("pip", "freeze", "--no-config", "--python", python),
            *("--target", site_packages, "--exclude", "axiom-encode"),
        ],
    ]
    # Each of the three installs read the build backends the lock exported.
    assert seen == [BUILD_EXPORT] * 3


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
    result, calls, _, _ = _run_staging_step(
        tmp_path, workflow, unhashed_export=unhashed_export
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_fails_when_the_staged_tree_drifts_from_the_lock(
    tmp_path, workflow
):
    drifted = LOCKED_FREEZE.replace("0.1.47", "0.2.159")
    result, calls, _, _ = _run_staging_step(tmp_path, workflow, staged_freeze=drifted)

    assert result.returncode != 0
    assert "+claude-agent-sdk==0.2.159" in result.stdout
    assert calls[-1][:2] == ["pip", "freeze"]


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_fails_when_the_lock_cannot_be_exported(tmp_path, workflow):
    result, calls, _, _ = _run_staging_step(tmp_path, workflow, export_fails=True)

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


@pytest.mark.parametrize("workflow", STAGING)
def test_staging_step_fails_when_the_build_backends_cannot_be_exported(
    tmp_path, workflow
):
    # A lock without the `build` group, or any export failure, must stop the
    # step before anything builds with unpinned backends.
    result, calls, _, seen = _run_staging_step(
        tmp_path, workflow, build_export_fails=True
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)
    assert seen == []
