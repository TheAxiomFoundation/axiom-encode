"""Content-keyed packaging of validate-rulespec pins for ``axiom-encode ci``.

``axiom-encode ci`` runs the embedded Python programs of the pinned reusable
workflow unchanged and reproduces the bash glue around them. So a new pin
needs code review only where something ci reproduces changed. This module
keys that review by content instead of by pin.

Glue units
    The workflow's top level, each job without its steps, and each step. A
    step's glue is the whole step, except that the program of an
    *exit-status* step (one whose program ci runs only for its exit status) is
    replaced by a marker and represented by its interface instead: the
    environment names it reads, its imports, the executables it starts, the
    absolute paths it names and the effects it has. Programs whose output ci
    parses or re-derives stay in the glue. Each unit also carries the
    ``workflow_call`` input declarations it references. A unit is identified
    by the sha256 of its canonical JSON (its *digest*).

Reviewed glue
    Every glue unit of a template pin (``TEMPLATE_PINS``: pins reviewed whole,
    with the executor code that supports them) is reviewed.
    ``ci_parity_workflows/reviewed-glue.json`` lists their digests; the tests check it against the templates, so a change
    to the extraction itself is caught. A reviewed variant requires the value
    of each in-scope feature on which every template containing it agrees.

Admission
    A pin is admitted when every glue unit is reviewed, its executed programs
    are well formed, its step order equals a template's with only whole
    feature groups removed, its variants' feature requirements agree, every
    producer/consumer pair it relies on occurs together in some template, and
    it declares every input ci reads. The merged features become the pin's
    gate parameters.

``axiom-encode ci-pin add <sha> --github-checkout <path>`` packages a pin
byte-exact from TheAxiomFoundation/.github main when it is admitted; otherwise
it prints each unreviewed unit diffed against its nearest reviewed variant and
exits non-zero without writing anything.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import functools
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, TextIO

import yaml

WORKFLOW_DIRECTORY = Path(__file__).with_name("ci_parity_workflows")
MANIFEST_NAME = "pins.json"
GITHUB_REPOSITORY = "TheAxiomFoundation/.github"
GITHUB_MAIN_URL = "https://github.com/TheAxiomFoundation/.github.git"
GITHUB_WORKFLOW_PATH = ".github/workflows/validate-rulespec.yml"
FETCHED_MAIN_REF = "refs/axiom-encode/ci-pin/main"
PROGRAM_MARKER = "<exit-status program ci executes unchanged>"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ABBREVIATED_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_QUOTED_HEREDOC_RE = re.compile(r"<<'(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'")
# Any heredoc operator bash accepts: quoted or not, with or without <<-.
_ANY_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*")
_INLINE_PYTHON = "python3 -c '"
_EXPRESSION_RE = re.compile(r"\$\{\{(?P<body>.*?)\}\}", re.DOTALL)
_GITHUB_REMOTE_RE = re.compile(
    r"(?i:github\.com)[:/](?P<slug>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
_INPUT_REFERENCE_RE = re.compile(r"\binputs\.(?P<name>[A-Za-z_][A-Za-z0-9_-]*)")
_INPUTS_TOKEN_RE = re.compile(r"\binputs\b(?!\.[A-Za-z_])")
_STEP_REFERENCE_RE = re.compile(r"\bsteps\.(?P<id>[A-Za-z_][A-Za-z0-9_-]*)\.")
_NEEDS_REFERENCE_RE = re.compile(r"\bneeds\.(?P<job>[A-Za-z_][A-Za-z0-9_-]*)\.")
_ENV_NAME_RE = re.compile(r"\b(?P<name>[A-Z][A-Z0-9_]*)=")
_ABSOLUTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._@+-]+(?:/[A-Za-z0-9._@+-]*)*$")


@dataclass(frozen=True, slots=True)
class ExecutedProgram:
    """How ci runs one embedded program of a pinned workflow.

    ``embedding`` is "heredoc" (the step's one quoted heredoc, read by
    PinnedWorkflow.python) or "inline" (its one ``python3 -c '...'``, read by
    PinnedWorkflow.inline_python). ``consumes`` is "exit-status" when ci uses
    only the program's exit status (its output is shown, never parsed), or
    "output" when ci parses its output or re-derives what it computes.
    """

    embedding: str
    consumes: str


_EXIT = "exit-status"
_OUTPUT = "output"
EXECUTED_PROGRAMS: Mapping[tuple[str, str], ExecutedProgram] = {
    # stdin is ci's reproduction of the step's git listing; exit status only.
    ("shards", "Reject unsupported tracked paths"): ExecutedProgram("inline", _EXIT),
    ("shards", "Reject unmanifested RuleSpec content"): ExecutedProgram(
        "heredoc", _EXIT
    ),
    # Its stdout lines become the matrix and its last stderr line the scope.
    ("shards", "Compute validation shards"): ExecutedProgram("heredoc", _OUTPUT),
    # Its key=value stdout feeds the retired-schema freeze.
    ("validate", "Authorize exact reviewed migration"): ExecutedProgram(
        "heredoc", _OUTPUT
    ),
    # ci re-derives its GITHUB_OUTPUT values with load_rulespec_toolchain.
    ("validate", "Resolve RuleSpec toolchain"): ExecutedProgram("heredoc", _OUTPUT),
    ("validate", "Reserve isolated dependency directories"): ExecutedProgram(
        "heredoc", _EXIT
    ),
    ("validate", "Validate composer pin"): ExecutedProgram("heredoc", _EXIT),
    # Its argv[1] file lines become the retired-schema skip list.
    ("validate", "Verify immutable retired-schema freeze"): ExecutedProgram(
        "heredoc", _OUTPUT
    ),
    # ci relies on the verified object it renames into place.
    ("validate", "Fetch pinned signed corpus release object"): ExecutedProgram(
        "heredoc", _OUTPUT
    ),
    ("validate", "Reject disallowed repository layout"): ExecutedProgram(
        "heredoc", _EXIT
    ),
    ("validate", "Enforce validation waiver ratchet"): ExecutedProgram(
        "heredoc", _EXIT
    ),
    # Its stdout "corpus-only" selects the corpus-only mode.
    ("validate", "Select RuleSpec validation targets"): ExecutedProgram(
        "heredoc", _OUTPUT
    ),
    # The SKIP programs' stdout lines become skip lists.
    ("validate", "Validate RuleSpec YAML"): ExecutedProgram("heredoc", _OUTPUT),
    ("validate", "Validate RuleSpec proofs and claims"): ExecutedProgram(
        "heredoc", _OUTPUT
    ),
    (
        "validate",
        "Validate changed PolicyEngine oracle coverage classification",
    ): ExecutedProgram("heredoc", _EXIT),
}


class GlueError(ValueError):
    """A workflow whose glue cannot be extracted (and so cannot be admitted)."""


# ---------------------------------------------------------------------------
# Embedded programs
# ---------------------------------------------------------------------------


def _heredoc_spans(run: str) -> list[tuple[int, int]]:
    """Line spans ``(first, end)`` of every quoted heredoc body in ``run``."""

    lines = run.split("\n")
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        match = _QUOTED_HEREDOC_RE.search(lines[index])
        index += 1
        if match is None:
            continue
        tag = match.group("tag")
        try:
            end = lines.index(tag, index)
        except ValueError as exc:
            raise GlueError(f"Unterminated heredoc {tag} in workflow step") from exc
        spans.append((index, end))
        index = end + 1
    return spans


def heredoc_bodies(run: str) -> list[str]:
    """Return every quoted heredoc body in a bash ``run`` script, in order."""

    lines = run.split("\n")
    return [
        "".join(f"{line}\n" for line in lines[first:end])
        for first, end in _heredoc_spans(run)
    ]


def _inline_span(run: str) -> tuple[int, int]:
    start = run.index(_INLINE_PYTHON) + len(_INLINE_PYTHON)
    end = run.index("\n'", start)
    return start, end + 1


def single_quoted_python(run: str) -> str:
    """Return the program of the one ``python3 -c '...'`` in a ``run`` script."""

    start, end = _inline_span(run)
    return run[start:end]


def executed_program(run: str, program: ExecutedProgram) -> tuple[str, str]:
    """Split ``run`` into (run with the program replaced by a marker, program).

    Refuses a step whose program ci and bash could read differently: an
    inline program containing a quote (bash ends the word there), a heredoc
    step with any other heredoc, an expression GitHub would substitute before
    bash runs, or a program that does not parse.
    """

    if program.embedding == "heredoc":
        if len(_ANY_HEREDOC_RE.findall(run)) != 1:
            raise GlueError("an executed heredoc step must have exactly one heredoc")
        spans = _heredoc_spans(run)
        if len(spans) != 1:
            raise GlueError("an executed heredoc step must quote its heredoc tag")
        first, end = spans[0]
        lines = run.split("\n")
        source = "".join(f"{line}\n" for line in lines[first:end])
        glue = "\n".join([*lines[:first], PROGRAM_MARKER, *lines[end:]])
    elif program.embedding == "inline":
        if run.count(_INLINE_PYTHON) != 1:
            raise GlueError("an executed inline step must have one python3 -c '")
        try:
            start, end = _inline_span(run)
        except ValueError as exc:
            raise GlueError("the inline program is not a multi-line word") from exc
        source = run[start:end]
        if "'" in source:
            raise GlueError("an inline program must contain no single quote")
        glue = run[:start] + PROGRAM_MARKER + run[end:]
    else:
        raise GlueError(f"unknown program embedding {program.embedding!r}")
    if "${{" in source:
        raise GlueError(
            "an executed program contains a ${{ }} expression, which GitHub "
            "substitutes before bash runs and ci does not"
        )
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise GlueError(f"an executed program does not parse: {exc}") from exc
    return glue, source


_SUBPROCESS_CALLS = frozenset(
    {
        "run",
        "call",
        "check_call",
        "check_output",
        "Popen",
        "getoutput",
        "getstatusoutput",
        "system",
        "popen",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "posix_spawn",
        "posix_spawnp",
        "startfile",
    }
)
_EFFECT_CALLS = frozenset(
    {
        "write_text",
        "write_bytes",
        "mkdir",
        "makedirs",
        "touch",
        "unlink",
        "remove",
        "rmdir",
        "removedirs",
        "rename",
        "renames",
        "replace",
        "symlink",
        "symlink_to",
        "hardlink_to",
        "link",
        "chmod",
        "chown",
        "lchmod",
        "copy",
        "copy2",
        "copyfile",
        "copytree",
        "move",
        "rmtree",
        "mkstemp",
        "mkdtemp",
        "NamedTemporaryFile",
        "TemporaryDirectory",
        "TemporaryFile",
        "SpooledTemporaryFile",
        "urlopen",
        "urlretrieve",
        "socket",
        "create_connection",
    }
)
_ENVIRON_ACCESSORS = frozenset({"get", "setdefault", "pop", "__getitem__"})


@dataclass(frozen=True, slots=True)
class ProgramInterface:
    """What an exit-status program reads or changes outside its own text.

    ci runs the program unchanged, but supplies its environment and runs it in
    its own interpreter and file system. A program that starts reading another
    variable, importing another module, starting another executable, naming
    another absolute path or changing files in another way needs review even
    when its step's glue is unchanged.
    """

    environment: tuple[str, ...]
    imports: tuple[str, ...]
    executables: tuple[str, ...]
    paths: tuple[str, ...]
    effects: tuple[str, ...]
    flags: tuple[str, ...]

    def as_json(self) -> dict[str, list[str]]:
        return {
            "environment": list(self.environment),
            "imports": list(self.imports),
            "executables": list(self.executables),
            "paths": list(self.paths),
            "effects": list(self.effects),
            "flags": list(self.flags),
        }


def _is_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute) and node.attr in {"environ", "environb"}
    ) or (isinstance(node, ast.Name) and node.id in {"environ", "environb"})


def _constant_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str | bytes):
        value = node.value
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
    return "<dynamic>"


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return ""


def _open_mode(node: ast.Call) -> str | None:
    """The mode of an ``open(...)`` / ``Path.open(...)`` call, if writable."""

    name = _call_name(node)
    if name not in {"open", "fdopen"}:
        return None
    positional = 0 if isinstance(node.func, ast.Attribute) else 1
    mode: ast.AST | None = None
    if len(node.args) > positional:
        mode = node.args[positional]
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
    if mode is None:
        return None
    text = _constant_name(mode)
    if text == "<dynamic>":
        return "open:<dynamic>"
    return f"open:{text}" if set(text) & set("wax+") else None


def program_interface(source: str) -> ProgramInterface:
    """Return the outside surface of an exit-status program's ``source``."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise GlueError(f"embedded program does not parse: {exc}") from exc
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    environment: set[str] = set()
    imports: set[str] = set()
    executables: set[str] = set()
    paths: set[str] = set()
    effects: set[str] = set()
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                flags.add("relative-import")
            imports.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _ABSOLUTE_PATH_RE.fullmatch(node.value):
                paths.add(node.value)
        elif _is_environ(node):
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript) and parent.value is node:
                environment.add(_constant_name(parent.slice))
            elif (
                isinstance(parent, ast.Attribute)
                and parent.value is node
                and parent.attr in _ENVIRON_ACCESSORS
                and isinstance(parents.get(parent), ast.Call)
                and parents[parent].func is parent
            ):
                call = parents[parent]
                environment.add(_constant_name(call.args[0] if call.args else None))
            elif (
                isinstance(parent, ast.Compare)
                and node in parent.comparators
                and all(isinstance(op, ast.In | ast.NotIn) for op in parent.ops)
            ):
                environment.add(_constant_name(parent.left))
            elif not isinstance(parent, ast.alias):
                # The whole environment escapes (passed on, copied, iterated):
                # what reaches the program is no longer a named set.
                flags.add("environment-escapes")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"getenv", "getenvb"}:
                environment.add(_constant_name(node.args[0] if node.args else None))
            elif name in {
                "__import__",
                "import_module",
                "find_spec",
                "spec_from_file_location",
            }:
                flags.add("dynamic-import")
            elif name in {"exec", "eval", "compile"}:
                flags.add("dynamic-code")
            elif name in {"expanduser", "home"}:
                flags.add("home-directory")
            if name in _SUBPROCESS_CALLS:
                command = node.args[0] if node.args else None
                if isinstance(command, ast.List | ast.Tuple) and command.elts:
                    executables.add(_constant_name(command.elts[0]))
                elif isinstance(command, ast.Constant) and isinstance(
                    command.value, str
                ):
                    executables.add("shell:" + (command.value.split() or [""])[0])
                else:
                    executables.add("<dynamic>")
            if name in _EFFECT_CALLS:
                effects.add(name)
            mode = _open_mode(node)
            if mode is not None:
                effects.add(mode)
    return ProgramInterface(
        environment=tuple(sorted(environment)),
        imports=tuple(sorted(imports)),
        executables=tuple(sorted(executables)),
        paths=tuple(sorted(paths)),
        effects=tuple(sorted(effects)),
        flags=tuple(sorted(flags)),
    )


# ---------------------------------------------------------------------------
# Glue units and digests
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            ("on" if key is True else str(key)): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)


def _expressions(value: Any) -> list[str]:
    return [
        match.group("body")
        for text in _strings(value)
        for match in _EXPRESSION_RE.finditer(text)
    ]


def workflow_call(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    trigger = payload.get("on", payload.get(True, {}))
    call = trigger.get("workflow_call") if isinstance(trigger, Mapping) else None
    return call if isinstance(call, Mapping) else {}


def declared_inputs(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Every ``workflow_call`` input, in declaration order, without descriptions."""

    raw = workflow_call(payload).get("inputs") or {}
    if not isinstance(raw, Mapping):
        raise GlueError("workflow_call inputs is not a mapping")
    inputs: dict[str, dict[str, Any]] = {}
    for name, declaration in raw.items():
        if not isinstance(name, str) or not isinstance(declaration, Mapping):
            raise GlueError(f"workflow_call input {name!r} is malformed")
        inputs[name] = {
            "type": declaration.get("type"),
            "required": bool(declaration.get("required", False)),
            "default": declaration.get("default"),
        }
    return inputs


def _referenced_inputs(
    value: Any, inputs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    names: set[str] = set()
    for expression in _expressions(value):
        if _INPUTS_TOKEN_RE.search(expression):
            raise GlueError(
                f"expression ${{{{{expression}}}}} reads inputs other than by "
                "inputs.<name>"
            )
        names.update(
            match.group("name") for match in _INPUT_REFERENCE_RE.finditer(expression)
        )
    return {name: inputs.get(name) for name in sorted(names)}


@dataclass(frozen=True, slots=True)
class GlueUnit:
    """One reviewable piece of a workflow: the workflow, a job or a step."""

    kind: str
    job: str | None
    step: str | None
    view: Mapping[str, Any]
    digest: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def label(self) -> str:
        if self.kind == "workflow":
            return "workflow"
        if self.kind == "job":
            return f"job {self.job}"
        return f"{self.job} / {self.step}"

    def text(self) -> str:
        return glue_text(self.view)


def glue_digest(view: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(view).encode("utf-8")).hexdigest()


def step_glue_view(
    job: str, step: Mapping[str, Any], inputs: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return a step's glue view (see the module docstring)."""

    glue = dict(step)
    program = EXECUTED_PROGRAMS.get((job, str(step.get("name"))))
    described: dict[str, Any] | None = None
    run = step.get("run")
    if program is not None:
        if not isinstance(run, str):
            raise GlueError(f"{job} / {step.get('name')} has no run script")
        stripped, source = executed_program(run, program)
        described = {"embedding": program.embedding, "consumes": program.consumes}
        if program.consumes == _EXIT:
            glue["run"] = stripped
            described["interface"] = program_interface(source).as_json()
    elif isinstance(run, str):
        # Unexecuted heredocs stay glue, but must still be well formed.
        _heredoc_spans(run)
    return {
        "kind": "step",
        "job": job,
        "step": glue,
        "program": described,
        "inputs": _referenced_inputs(step, inputs),
    }


def glue_units(payload: Mapping[str, Any]) -> list[GlueUnit]:
    """Every glue unit of a pinned workflow, in file order."""

    if not isinstance(payload, Mapping):
        raise GlueError("the workflow is not a mapping")
    inputs = declared_inputs(payload)
    jobs = payload.get("jobs")
    if not isinstance(jobs, Mapping) or not jobs:
        raise GlueError("the workflow has no jobs")
    settings = {key: value for key, value in payload.items() if key != "jobs"}
    trigger_key = "on" if "on" in settings else True
    trigger = settings.get(trigger_key)
    if isinstance(trigger, Mapping):
        trigger = dict(trigger)
        call = dict(trigger.get("workflow_call") or {})
        call.pop("inputs", None)
        trigger["workflow_call"] = call
        settings[trigger_key] = trigger
    workflow_view = {
        "kind": "workflow",
        "settings": settings,
        "inputs": _referenced_inputs(settings, inputs),
    }
    units = [
        GlueUnit(
            "workflow", None, None, workflow_view, glue_digest(workflow_view), payload
        )
    ]
    for job_name, job in jobs.items():
        if not isinstance(job_name, str) or not isinstance(job, Mapping):
            raise GlueError(f"job {job_name!r} is malformed")
        job_settings = {key: value for key, value in job.items() if key != "steps"}
        job_view = {
            "kind": "job",
            "job": job_name,
            "settings": job_settings,
            "inputs": _referenced_inputs(job_settings, inputs),
        }
        units.append(
            GlueUnit("job", job_name, None, job_view, glue_digest(job_view), job)
        )
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            raise GlueError(f"job {job_name} steps are not a list")
        for step in steps:
            if not isinstance(step, Mapping) or not isinstance(step.get("name"), str):
                raise GlueError(f"job {job_name} has a step without a name")
            view = step_glue_view(job_name, step, inputs)
            units.append(
                GlueUnit("step", job_name, step["name"], view, glue_digest(view), step)
            )
    return units


def glue_text(view: Mapping[str, Any]) -> str:
    """A line-oriented rendering of a glue view, for review diffs."""

    lines: list[str] = []
    if view["kind"] == "step":
        step = dict(view["step"])
        run = step.pop("run", None)
        lines.append(f"job: {view['job']}")
        lines.extend(
            yaml.safe_dump(_jsonable(step), sort_keys=True, width=1000).splitlines()
        )
        if isinstance(run, str):
            lines.append("run: |")
            lines.extend(f"  {line}" if line else "" for line in run.split("\n"))
        program = view.get("program")
        if program:
            lines.append(
                f"executed program: {program['embedding']}, ci consumes its "
                f"{program['consumes']}"
            )
            for key, values in (program.get("interface") or {}).items():
                lines.append(f"  {key}: {', '.join(values) or '-'}")
    else:
        if view["kind"] == "job":
            lines.append(f"job: {view['job']}")
        lines.extend(
            yaml.safe_dump(
                _jsonable(view["settings"]), sort_keys=True, width=1000
            ).splitlines()
        )
    referenced = view.get("inputs") or {}
    if referenced:
        lines.append("referenced inputs:")
        for name, declaration in referenced.items():
            lines.append(f"  {name}: {canonical_json(declaration)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Line spans
# ---------------------------------------------------------------------------


def step_spans(text: str) -> dict[tuple[str, str], tuple[int, int]]:
    """1-based inclusive line span of every named step of a workflow file.

    A step runs from its first line up to the line before the next step or
    job (or the end of the file), less trailing blank lines and comments that
    sit at the step list's indentation (they introduce the next step).
    """

    lines = text.split("\n")
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        raise GlueError("the workflow is not a mapping")
    jobs = next((value for key, value in root.value if key.value == "jobs"), None)
    if not isinstance(jobs, yaml.MappingNode):
        raise GlueError("the workflow has no jobs")
    starts: list[tuple[int, int, str, str]] = []
    boundaries = {len(lines) + 1}
    for job_key, job in jobs.value:
        boundaries.add(job_key.start_mark.line + 1)
        if not isinstance(job, yaml.MappingNode):
            continue
        steps = next((value for key, value in job.value if key.value == "steps"), None)
        if not isinstance(steps, yaml.SequenceNode):
            continue
        for step in steps.value:
            if not isinstance(step, yaml.MappingNode):
                continue
            line = step.start_mark.line + 1
            boundaries.add(line)
            name = next(
                (value.value for key, value in step.value if key.value == "name"),
                None,
            )
            if isinstance(name, str):
                starts.append((line, step.start_mark.column, job_key.value, name))
    ordered = sorted(boundaries)
    spans: dict[tuple[str, str], tuple[int, int]] = {}
    for line, column, job, name in starts:
        end = next(boundary for boundary in ordered if boundary > line) - 1
        while end > line:
            content = lines[end - 1]
            stripped = content.lstrip()
            indent = len(content) - len(stripped)
            if stripped and not (stripped.startswith("#") and indent < column):
                break
            end -= 1
        spans[(job, name)] = (line, end)
    return spans


# ---------------------------------------------------------------------------
# Reviewed glue
# ---------------------------------------------------------------------------

# Pin features: the WorkflowGateParameters fields that differ between pins
# that execute their own scripts.
FEATURES: tuple[str, ...] = (
    "unmanifested_precheck",
    "exact_guard_base_ref",
    "retired_corpus_release_key",
    "isolated_dependencies",
    "composer_runtime",
    "program_spec_precheck",
)
# Features whose steps a pin has all or none of. A pin's step order must equal
# a template's with zero or more of these groups removed whole, and the
# feature is on exactly when its group is present.
FEATURE_STEPS: Mapping[str, frozenset[tuple[str, str]]] = {
    "unmanifested_precheck": frozenset(
        {("shards", "Reject unmanifested RuleSpec content")}
    ),
    "program_spec_precheck": frozenset(
        {
            ("shards", "Set up Python for the RuleSpec pre-check"),
            ("shards", "Install the ProgramSpec shape-check parser"),
        }
    ),
    "isolated_dependencies": frozenset(
        {
            ("validate", "Reserve isolated dependency directories"),
            ("validate", "Relocate dependency axiom-encode"),
            ("validate", "Relocate dependency axiom-rules-engine"),
            ("validate", "Relocate dependency axiom-corpus"),
            ("validate", "Relocate dependency rulespec-us"),
            ("validate", "Relocate dependency axiom-compose"),
            ("validate", "Relocate dependency axiom-encode-oracle-coverage"),
        }
    ),
    "composer_runtime": frozenset(
        {
            ("validate", "Validate composer pin"),
            ("validate", "Checkout axiom-compose"),
            ("validate", "Authenticate composer commit"),
            ("validate", "Install uv for composer runtime"),
            ("validate", "Build protected axiom-compose runtime"),
        }
    ),
}


@dataclass(frozen=True, slots=True)
class FeatureScope:
    """The glue units whose content can depend on a feature.

    A reviewed variant requires a feature's value only when the unit is in the
    feature's scope. A scope is a review claim ("nothing else reads this
    feature"); tests check each claim's witness text stays inside its scope.
    """

    jobs: frozenset[str] = frozenset()
    steps: frozenset[tuple[str, str]] = frozenset()

    def contains(self, unit: GlueUnit) -> bool:
        if unit.job in self.jobs:
            return True
        return unit.kind == "step" and (unit.job, unit.step) in self.steps


FEATURE_SCOPES: Mapping[str, FeatureScope] = {
    # The pre-check and its ProgramSpec parser are shards-job steps; the
    # validate job neither reads GUARD_PROGRAMS_ROOT's shape check nor PyYAML.
    "unmanifested_precheck": FeatureScope(
        steps=frozenset(FEATURE_STEPS["unmanifested_precheck"])
    ),
    "program_spec_precheck": FeatureScope(
        steps=frozenset(
            FEATURE_STEPS["program_spec_precheck"]
            | FEATURE_STEPS["unmanifested_precheck"]
        )
    ),
    # The exact-SHA check is the guard step's own bash (generated-guard-base-ref).
    "exact_guard_base_ref": FeatureScope(
        steps=frozenset({("validate", "Reject manual RuleSpec changes")})
    ),
    "retired_corpus_release_key": FeatureScope(
        steps=frozenset({("validate", "Provision protected verification supervisor")})
    ),
    # Dependency checkouts, their layout and the composer exist only in the
    # validate job; the shards job checks out the rules repository alone.
    "isolated_dependencies": FeatureScope(jobs=frozenset({"validate"})),
    "composer_runtime": FeatureScope(jobs=frozenset({"validate"})),
}
# Pins reviewed whole, with the executor code that supports them, and the
# features that review established. Every glue unit they contain is reviewed;
# their step order is the order ci's executor reproduces.
TEMPLATE_FEATURES: Mapping[str, Mapping[str, bool]] = {
    "0effa6a5b05e7fac53902df7d523e909bd7fc48a": {
        "unmanifested_precheck": False,
        "exact_guard_base_ref": False,
        "retired_corpus_release_key": True,
        "isolated_dependencies": False,
        "composer_runtime": False,
        "program_spec_precheck": False,
    },
    "6f11be2655f79dd0a3b582db46525f58332ca120": {
        "unmanifested_precheck": True,
        "exact_guard_base_ref": True,
        "retired_corpus_release_key": True,
        "isolated_dependencies": False,
        "composer_runtime": False,
        "program_spec_precheck": False,
    },
    "92ccb0f2266bf06aba6111d2bfd0e5f03af55fc3": {
        "unmanifested_precheck": True,
        "exact_guard_base_ref": True,
        "retired_corpus_release_key": True,
        "isolated_dependencies": True,
        "composer_runtime": True,
        "program_spec_precheck": True,
    },
}
TEMPLATE_PINS: tuple[str, ...] = tuple(TEMPLATE_FEATURES)
# Inputs ci reads from the caller for a pin that executes its own scripts
# (ci_parity.py ``caller.inputs[...]``), with the type ci assumes.
CI_READ_INPUTS: Mapping[str, str] = {
    "python-version": "string",
    "axiom-encode-ref": "string",
    "axiom-rules-engine-ref": "string",
    "axiom-corpus-ref": "string",
    "rulespec-us-ref": "string",
    "corpus-release-base-url": "string",
    "corpus-release-registry-url": "string",
    "corpus-release-registry-anon-key": "string",
    "validate-roots": "string",
    "validation-workers": "number",
    "run-pytest": "boolean",
    "run-generated-guard": "boolean",
    "migration-authorization-path": "string",
    "retired-schema-bootstrap-sha256": "string",
    "allow-retired-schema-prefreeze": "boolean",
    "validation-waiver-bootstrap-sha256": "string",
    "guard-programs-root": "boolean",
    "run-money-atom-check": "boolean",
}
COMPOSER_INPUT = "axiom-compose-ref"


@dataclass(frozen=True, slots=True)
class ReviewedGlue:
    """One reviewed glue digest, as it occurs in the template pins.

    ``reviewed_in`` is the 8-character prefix of the first template pin that
    contains it. ``gate_flags`` names a gate step's literal-flag profile when
    it differs from the default (ci_parity._WORKFLOW_GATE_FLAG_PROFILES).
    """

    kind: str
    job: str | None
    step: str | None
    digest: str
    reviewed_in: str
    gate_flags: str | None = None


REVIEWED_GLUE_NAME = "reviewed-glue.json"


def load_reviewed_glue(
    directory: Path = WORKFLOW_DIRECTORY,
) -> tuple[ReviewedGlue, ...]:
    """The reviewed glue digests, from ``ci_parity_workflows/reviewed-glue.json``."""

    raw = json.loads((directory / REVIEWED_GLUE_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{REVIEWED_GLUE_NAME} is not a list")
    entries = []
    for item in raw:
        if not isinstance(item, dict) or not set(item) <= {
            "kind",
            "job",
            "step",
            "digest",
            "reviewed_in",
            "gate_flags",
        }:
            raise ValueError(f"malformed {REVIEWED_GLUE_NAME} entry {item!r}")
        entries.append(
            ReviewedGlue(
                item["kind"],
                item.get("job"),
                item.get("step"),
                item["digest"],
                item["reviewed_in"],
                item.get("gate_flags"),
            )
        )
    return tuple(entries)


def write_reviewed_glue(entries: Sequence[ReviewedGlue], directory: Path) -> None:
    """Write the reviewed glue one entry per line, ordered by unit, for review diffs."""

    def order(entry: ReviewedGlue) -> tuple[Any, ...]:
        kinds = {"workflow": 0, "job": 1, "step": 2}
        return (entry.job or "", kinds[entry.kind], entry.step or "", entry.reviewed_in)

    rows = []
    for entry in sorted(entries, key=order):
        item: dict[str, Any] = {"kind": entry.kind}
        if entry.job is not None:
            item["job"] = entry.job
        if entry.step is not None:
            item["step"] = entry.step
        item["reviewed_in"] = entry.reviewed_in
        item["digest"] = entry.digest
        if entry.gate_flags is not None:
            item["gate_flags"] = entry.gate_flags
        rows.append(json.dumps(item, ensure_ascii=False))
    (directory / REVIEWED_GLUE_NAME).write_text(
        "[\n" + ",\n".join(f"  {row}" for row in rows) + "\n]\n", encoding="utf-8"
    )


def template_glue(
    templates: Mapping[str, Mapping[str, Any]],
    existing: Sequence[ReviewedGlue] = (),
) -> list[ReviewedGlue]:
    """Reviewed glue entries for every unit of ``templates`` (in TEMPLATE_PINS order).

    Existing entries keep their gate-flag profile; new ones have none.
    """

    flags = {entry.digest: entry.gate_flags for entry in existing}
    entries: dict[str, ReviewedGlue] = {}
    for sha in TEMPLATE_PINS:
        if sha not in templates:
            continue
        for unit in glue_units(templates[sha]):
            if unit.digest not in entries:
                entries[unit.digest] = ReviewedGlue(
                    unit.kind,
                    unit.job,
                    unit.step,
                    unit.digest,
                    sha[:8],
                    flags.get(unit.digest),
                )
    return list(entries.values())


@dataclass(frozen=True, slots=True)
class ReviewedVariant:
    entry: ReviewedGlue
    unit: GlueUnit
    templates: tuple[str, ...]
    requires: Mapping[str, bool]


@dataclass(frozen=True, slots=True)
class ReviewContext:
    variants: Mapping[str, ReviewedVariant]
    shapes: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]]
    digests: Mapping[str, frozenset[str]]


def build_review_context(
    templates: Mapping[str, Mapping[str, Any]],
    entries: Sequence[ReviewedGlue] | None = None,
    template_features: Mapping[str, Mapping[str, bool]] | None = None,
) -> ReviewContext:
    """Index the reviewed glue by digest, checked against the template pins."""

    template_features = (
        TEMPLATE_FEATURES if template_features is None else template_features
    )

    table: dict[str, ReviewedGlue] = {}
    for entry in load_reviewed_glue() if entries is None else entries:
        if entry.digest in table:
            raise ValueError(f"{REVIEWED_GLUE_NAME} lists {entry.digest} twice")
        table[entry.digest] = entry
    found: dict[str, tuple[GlueUnit, list[str]]] = {}
    shapes = {}
    digests = {}
    for sha, payload in templates.items():
        units = glue_units(payload)
        shapes[sha] = job_shape(payload)
        digests[sha] = frozenset(unit.digest for unit in units)
        for unit in units:
            entry = table.get(unit.digest)
            if entry is None or (entry.kind, entry.job, entry.step) != (
                unit.kind,
                unit.job,
                unit.step,
            ):
                raise ValueError(
                    f"template pin {sha[:8]} {unit.label} has glue {unit.digest} "
                    "that reviewed-glue.json does not list"
                )
            found.setdefault(unit.digest, (unit, []))[1].append(sha)
    missing = set(table) - set(found)
    if missing:
        raise ValueError(
            f"{REVIEWED_GLUE_NAME} lists digests no template has: {sorted(missing)}"
        )
    variants: dict[str, ReviewedVariant] = {}
    for digest, (unit, shas) in found.items():
        entry = table[digest]
        if entry.reviewed_in != shas[0][:8]:
            raise ValueError(
                f"{REVIEWED_GLUE_NAME} says {unit.label} {digest[:12]} was reviewed in "
                f"{entry.reviewed_in}, but it first occurs in {shas[0][:8]}"
            )
        requires = {}
        for feature in FEATURES:
            if not FEATURE_SCOPES[feature].contains(unit):
                continue
            values = {template_features[sha][feature] for sha in shas}
            if len(values) == 1:
                requires[feature] = values.pop()
        variants[digest] = ReviewedVariant(entry, unit, tuple(shas), requires)
    return ReviewContext(variants, shapes, digests)


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

Shape = tuple[tuple[str, tuple[str, ...]], ...]


def job_shape(payload: Mapping[str, Any]) -> Shape:
    """Each job with its step names, in file order."""

    return tuple(
        (
            str(job_name),
            tuple(
                str(step.get("name"))
                for step in (job.get("steps") or [])
                if isinstance(step, Mapping)
            ),
        )
        for job_name, job in payload["jobs"].items()
        if isinstance(job, Mapping)
    )


def _without_groups(shape: Shape, groups: Iterable[str]) -> Shape:
    removed: set[tuple[str, str]] = set()
    for group in groups:
        removed |= FEATURE_STEPS[group]
    return tuple(
        (job, tuple(name for name in names if (job, name) not in removed))
        for job, names in shape
    )


def _present_groups(shape: Shape) -> list[str]:
    present = {(job, name) for job, names in shape for name in names}
    return [group for group, steps in FEATURE_STEPS.items() if steps <= present]


@dataclass(frozen=True, slots=True)
class Problem:
    kind: str
    message: str
    unit: GlueUnit | None = None


@dataclass(slots=True)
class Admission:
    """The outcome of checking one packaged or candidate pin."""

    problems: list[Problem]
    features: dict[str, bool]
    template: str | None = None
    removed_groups: tuple[str, ...] = ()
    units: list[GlueUnit] = field(default_factory=list)
    gate_flags: dict[str, str] = field(default_factory=dict)

    @property
    def admitted(self) -> bool:
        return not self.problems


def _env_exports(unit: GlueUnit) -> set[str]:
    run = unit.view["step"].get("run") if unit.kind == "step" else None
    if not isinstance(run, str) or "GITHUB_ENV" not in run:
        return set()
    return {match.group("name") for match in _ENV_NAME_RE.finditer(run)}


def _dependencies(
    units: Sequence[GlueUnit],
) -> tuple[list[tuple[GlueUnit, GlueUnit, str]], list[Problem]]:
    """(consumer, producer, reference) pairs a pin's glue relies on."""

    pairs: list[tuple[GlueUnit, GlueUnit, str]] = []
    problems: list[Problem] = []
    jobs = {unit.job: unit for unit in units if unit.kind == "job"}
    for index, unit in enumerate(units):
        if unit.kind == "workflow":
            continue
        same_job = [
            other
            for other in units[:index]
            if other.kind == "step" and other.job == unit.job
        ]
        text = canonical_json(unit.view)
        for match in _STEP_REFERENCE_RE.finditer(text):
            step_id = match.group("id")
            producers = [
                other
                for other in units
                if other.kind == "step"
                and other.job == unit.job
                and other.raw.get("id") == step_id
            ]
            if not producers:
                problems.append(
                    Problem(
                        "dependency",
                        f"{unit.label} reads steps.{step_id}, which no step of its job defines",
                        unit,
                    )
                )
            pairs.extend((unit, producer, f"steps.{step_id}") for producer in producers)
        for match in _NEEDS_REFERENCE_RE.finditer(text):
            needed = jobs.get(match.group("job"))
            if needed is None:
                problems.append(
                    Problem(
                        "dependency",
                        f"{unit.label} reads needs.{match.group('job')}, which is no job",
                        unit,
                    )
                )
                continue
            pairs.append((unit, needed, f"needs.{match.group('job')}"))
        for producer in same_job:
            for name in _env_exports(producer):
                if re.search(rf"(?:\$\{{?|\benv\.){re.escape(name)}\b", text):
                    pairs.append((unit, producer, f"GITHUB_ENV {name}"))
    unique = {(c.digest, p.digest, ref): (c, p, ref) for c, p, ref in pairs}
    return list(unique.values()), problems


def admit(payload: Mapping[str, Any], context: ReviewContext) -> Admission:
    """Check a pinned workflow against the reviewed glue and template pins."""

    problems: list[Problem] = []
    try:
        units = glue_units(payload)
        inputs = declared_inputs(payload)
    except GlueError as exc:
        return Admission([Problem("malformed", str(exc))], {})
    seen: set[tuple[str | None, str | None]] = set()
    for unit in units:
        if unit.kind == "step":
            if (unit.job, unit.step) in seen:
                problems.append(
                    Problem(
                        "duplicate",
                        f"{unit.label} appears twice; ci finds steps by name",
                        unit,
                    )
                )
            seen.add((unit.job, unit.step))
        variant = context.variants.get(unit.digest)
        if variant is None or (
            variant.unit.kind,
            variant.unit.job,
            variant.unit.step,
        ) != (
            unit.kind,
            unit.job,
            unit.step,
        ):
            problems.append(
                Problem("unreviewed", f"{unit.label}: glue is not reviewed", unit)
            )

    shape = job_shape(payload)
    template: str | None = None
    removed: tuple[str, ...] = ()
    for template_sha, template_shape in context.shapes.items():
        groups = _present_groups(template_shape)
        for size in range(len(groups) + 1):
            for subset in combinations(groups, size):
                if _without_groups(template_shape, subset) == shape:
                    template, removed = template_sha, subset
                    break
            if template is not None:
                break
        if template is not None:
            break
    if template is None:
        problems.append(Problem("shape", _shape_message(shape, context)))

    sources: dict[str, dict[bool, list[str]]] = {
        feature: {True: [], False: []} for feature in FEATURES
    }
    gate_flags: dict[str, str] = {}
    for unit in units:
        variant = context.variants.get(unit.digest)
        if variant is None:
            continue
        for feature, value in variant.requires.items():
            sources[feature][value].append(unit.label)
        if variant.entry.gate_flags is not None and unit.step is not None:
            gate_flags[unit.step] = variant.entry.gate_flags
    present = {(job, name) for job, names in shape for name in names}
    unreviewed = [unit for unit in units if unit.digest not in context.variants]
    features: dict[str, bool] = {}
    for feature in FEATURES:
        wanted = sources[feature]
        group = FEATURE_STEPS.get(feature)
        if group is not None and group & present and not group <= present:
            # A partial group is a shape problem, reported above.
            features[feature] = False
            continue
        if wanted[True] and wanted[False]:
            problems.append(
                Problem(
                    "features",
                    f"{feature}: on for {', '.join(wanted[True])}; off for "
                    f"{', '.join(wanted[False])}",
                )
            )
        if group is not None:
            features[feature] = group <= present
            contrary = wanted[not features[feature]]
            if contrary:
                problems.append(
                    Problem(
                        "features",
                        f"{feature} is {'on' if features[feature] else 'off'} "
                        f"(its steps are {'present' if features[feature] else 'absent'}), "
                        f"but {', '.join(contrary)} require it "
                        f"{'off' if features[feature] else 'on'}",
                    )
                )
        elif wanted[True] or wanted[False]:
            features[feature] = bool(wanted[True])
        else:
            features[feature] = False
            # An unreviewed unit in the feature's scope is reported already.
            if not any(FEATURE_SCOPES[feature].contains(unit) for unit in unreviewed):
                problems.append(
                    Problem(
                        "features",
                        f"no reviewed variant in this pin decides {feature}",
                    )
                )

    pairs, dependency_problems = _dependencies(units)
    problems.extend(dependency_problems)
    for consumer, producer, reference in pairs:
        if (
            consumer.digest not in context.variants
            or producer.digest not in context.variants
        ):
            continue
        if not any(
            consumer.digest in digests and producer.digest in digests
            for digests in context.digests.values()
        ):
            problems.append(
                Problem(
                    "dependency",
                    f"{consumer.label} reads {reference} from {producer.label}, but "
                    "no reviewed pin has these two variants together",
                    consumer,
                )
            )

    expected_inputs = dict(CI_READ_INPUTS)
    if features.get("composer_runtime"):
        expected_inputs[COMPOSER_INPUT] = "string"
    for name, kind in expected_inputs.items():
        declared = inputs.get(name)
        if declared is None or declared["type"] != kind:
            problems.append(
                Problem(
                    "inputs",
                    f"ci reads input {name} as a {kind}, but the workflow declares "
                    + (
                        "no such input" if declared is None else f"a {declared['type']}"
                    ),
                )
            )
    return Admission(problems, features, template, tuple(removed), units, gate_flags)


def _shape_message(shape: Shape, context: ReviewContext) -> str:
    def render(value: Shape) -> list[str]:
        return [f"job {job}" for job, _ in value] + [
            f"{job} / {name}" for job, names in value for name in names
        ]

    best = max(
        context.shapes,
        key=lambda sha: difflib.SequenceMatcher(
            None, render(context.shapes[sha]), render(shape)
        ).ratio(),
    )
    diff = "\n".join(
        f"    {line}"
        for line in difflib.unified_diff(
            render(context.shapes[best]),
            render(shape),
            f"template {best[:8]}",
            "candidate",
            lineterm="",
        )
    )
    return (
        "step order matches no template pin with whole feature groups removed; "
        f"nearest template {best[:8]}:\n{diff}"
    )


def nearest_reviewed(unit: GlueUnit, context: ReviewContext) -> ReviewedVariant | None:
    """The reviewed variant most like ``unit``: same step first, then same job."""

    pool = [
        variant
        for variant in context.variants.values()
        if variant.unit.kind == unit.kind and variant.unit.job == unit.job
    ]
    same = [variant for variant in pool if variant.unit.step == unit.step]
    text = unit.text()
    ranked = sorted(
        same or pool,
        key=lambda variant: difflib.SequenceMatcher(
            None, variant.unit.text(), text, autojunk=False
        ).ratio(),
        reverse=True,
    )
    return ranked[0] if ranked else None


def render_report(admission: Admission, context: ReviewContext, *, label: str) -> str:
    """Explain an admission: each unreviewed unit diffed against its nearest."""

    out: list[str] = []
    if admission.admitted:
        out.append(f"{label}: admitted; every glue unit is reviewed")
    else:
        out.append(f"{label}: NOT admitted ({len(admission.problems)} problem(s))")
    for problem in admission.problems:
        out.append("")
        out.append(f"- {problem.message}")
        unit = problem.unit
        if problem.kind != "unreviewed" or unit is None:
            continue
        nearest = nearest_reviewed(unit, context)
        if nearest is None:
            out.append("  no reviewed variant in this job; its glue:")
            out.extend(f"  | {line}" for line in unit.text().splitlines())
            continue
        relation = (
            "this step"
            if nearest.unit.step == unit.step
            else f"another step, {nearest.unit.label}"
        )
        out.append(
            f"  nearest reviewed variant: {relation}, reviewed in "
            f"{nearest.entry.reviewed_in} ({nearest.unit.digest[:12]}); this pin "
            f"has {unit.digest[:12]}"
        )
        out.extend(
            f"  {line}"
            for line in difflib.unified_diff(
                nearest.unit.text().splitlines(),
                unit.text().splitlines(),
                f"reviewed ({nearest.entry.reviewed_in})",
                "this pin",
                lineterm="",
                n=2,
            )
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# The packaged-pin manifest
# ---------------------------------------------------------------------------

# Pins before 0effa6a5 run ci's own fixed gate sequence (execute_gates), not
# the workflow's scripts; content-keyed admission does not apply to them.
LEGACY_PINS: frozenset[str] = frozenset(
    {
        "615c1df9b9ace7deea84da65efd137f46f8bad2b",
        "34bcfab235c585c47292c95f51be1a4f4f91d29e",
    }
)


@dataclass(frozen=True, slots=True)
class PackagedPin:
    sha: str
    file: str
    blob: str


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def workflow_file_name(sha: str) -> str:
    return f"validate-rulespec-{sha[:8]}.yml"


def load_manifest(directory: Path = WORKFLOW_DIRECTORY) -> dict[str, PackagedPin]:
    """Every packaged pin, from ``ci_parity_workflows/pins.json``."""

    raw = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("the packaged-pin manifest is not an object")
    pins: dict[str, PackagedPin] = {}
    for sha, entry in raw.items():
        if (
            not _SHA_RE.fullmatch(sha)
            or not isinstance(entry, dict)
            or set(entry) != {"file", "blob"}
            or not _SHA_RE.fullmatch(str(entry["blob"]))
        ):
            raise ValueError(f"malformed packaged-pin manifest entry {sha!r}")
        if entry["file"] != workflow_file_name(sha):
            raise ValueError(f"packaged pin {sha} names file {entry['file']!r}")
        pins[sha] = PackagedPin(sha, entry["file"], entry["blob"])
    return pins


def write_manifest(pins: Mapping[str, PackagedPin], directory: Path) -> None:
    payload = {
        sha: {"blob": pin.blob, "file": pin.file} for sha, pin in sorted(pins.items())
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def read_packaged(pin: PackagedPin, directory: Path = WORKFLOW_DIRECTORY) -> bytes:
    path = directory / pin.file
    data = path.read_bytes()
    actual = git_blob_id(data)
    if actual != pin.blob:
        raise ValueError(
            f"Packaged workflow {path} is blob {actual}, not the "
            f"validate-rulespec@{pin.sha} blob {pin.blob}"
        )
    return data


def load_workflow(data: bytes) -> dict[str, Any]:
    payload = yaml.load(data, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    if not isinstance(payload, dict):
        raise GlueError("the workflow is not a mapping")
    return payload


@functools.cache
def packaged_manifest() -> Mapping[str, PackagedPin]:
    return load_manifest()


@functools.cache
def packaged_bytes(sha: str) -> bytes:
    return read_packaged(packaged_manifest()[sha])


@functools.cache
def packaged_payload(sha: str) -> dict[str, Any]:
    """The verified, parsed packaged workflow for ``sha`` (cached; do not mutate)."""

    return load_workflow(packaged_bytes(sha))


@functools.cache
def review_context() -> ReviewContext:
    return build_review_context({sha: packaged_payload(sha) for sha in TEMPLATE_PINS})


@functools.cache
def packaged_admission(sha: str) -> Admission:
    """Admission of a packaged pin against the reviewed glue (cached)."""

    return admit(packaged_payload(sha), review_context())


# ---------------------------------------------------------------------------
# axiom-encode ci-pin
# ---------------------------------------------------------------------------


def register_ci_pin_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "ci-pin",
        help="Package a validate-rulespec workflow pin for axiom-encode ci",
    )
    actions = parser.add_subparsers(dest="ci_pin_action", required=True)
    add = actions.add_parser(
        "add",
        help=(
            "Copy validate-rulespec.yml at a TheAxiomFoundation/.github main "
            "commit into this checkout and register it, if every glue unit is "
            "already reviewed"
        ),
    )
    add.add_argument("sha", help="the .github commit (full or abbreviated)")
    add.add_argument(
        "--github-checkout",
        type=Path,
        required=True,
        help="a clone of TheAxiomFoundation/.github",
    )
    add.add_argument(
        "--no-fetch",
        action="store_true",
        help=(
            "check ancestry against the checkout's origin/main instead of "
            f"fetching main from {GITHUB_MAIN_URL}"
        ),
    )
    add.add_argument(
        "--dry-run",
        action="store_true",
        help="report whether the pin would be admitted and write nothing",
    )
    add.add_argument(
        "--template",
        action="store_true",
        help=(
            "package a pin whose glue was reviewed with an executor change and "
            "whose features are declared in TEMPLATE_FEATURES; its glue becomes "
            "reviewed"
        ),
    )
    actions.add_parser("list", help="list packaged pins and their derived features")


def run_ci_pin(args: argparse.Namespace, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    if args.ci_pin_action == "list":
        return list_pins(out)
    try:
        return add_pin(
            args.sha,
            args.github_checkout,
            fetch=not getattr(args, "no_fetch", False),
            dry_run=bool(getattr(args, "dry_run", False)),
            template=bool(getattr(args, "template", False)),
            out=out,
        )
    except PinSourceError as exc:
        print(f"ci-pin add: {exc}", file=out)
        return 2


class PinSourceError(ValueError):
    """The .github checkout or commit cannot be trusted as a pin source."""


def _hermetic_git_environment() -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(checkout: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_hermetic_git_environment(),
        check=False,
    )


def _git_text(checkout: Path, *arguments: str) -> str:
    result = _git(checkout, *arguments)
    if result.returncode:
        raise PinSourceError(
            f"git {' '.join(arguments)} failed in {checkout}: "
            + result.stderr.decode("utf-8", "replace").strip()
        )
    return result.stdout.decode("utf-8").strip()


@dataclass(frozen=True, slots=True)
class PinSource:
    sha: str
    main: str
    main_source: str
    data: bytes
    blob: str


def read_pin_source(
    sha: str,
    checkout: Path,
    *,
    fetch: bool = True,
    main_url: str = GITHUB_MAIN_URL,
) -> PinSource:
    """Read validate-rulespec.yml at ``sha`` from a verified .github checkout.

    The checkout's origin must be github.com/TheAxiomFoundation/.github, and
    ``sha`` must be an ancestor of .github main: fetched from ``main_url`` into
    a private ref, or, with ``fetch=False``, the checkout's origin/main. Replace
    refs, grafts, shallow history and URL rewriting are refused or ignored,
    and the file must be a regular blob that hashes to its id.
    """

    checkout = checkout.expanduser().resolve()
    requested = sha.strip().lower()
    if not _ABBREVIATED_SHA_RE.fullmatch(requested):
        raise PinSourceError(f"{sha!r} is not a hexadecimal commit id")
    if _git_text(checkout, "rev-parse", "--is-inside-work-tree") != "true":
        raise PinSourceError(f"{checkout} is not a git work tree")
    git_dir = Path(_git_text(checkout, "rev-parse", "--absolute-git-dir"))
    if (git_dir / "info" / "grafts").exists():
        raise PinSourceError(f"{git_dir}/info/grafts rewrites history")
    if _git_text(checkout, "rev-parse", "--is-shallow-repository") != "false":
        raise PinSourceError(f"{checkout} is a shallow clone; ancestry is incomplete")
    rewrites = _git(checkout, "config", "--get-regexp", r"^url\.")
    if rewrites.returncode == 0 and rewrites.stdout.strip():
        raise PinSourceError(f"{checkout} rewrites remote URLs (url.*.insteadOf)")
    origin = _git(checkout, "config", "--get", "remote.origin.url")
    origin_url = origin.stdout.decode("utf-8").strip()
    match = _GITHUB_REMOTE_RE.search(origin_url)
    if (
        origin.returncode
        or match is None
        or match.group("slug").lower() != GITHUB_REPOSITORY.lower()
    ):
        raise PinSourceError(
            f"{checkout}'s origin is {origin_url or 'unset'}, not "
            f"github.com/{GITHUB_REPOSITORY}"
        )
    if fetch:
        _git_text(
            checkout,
            "fetch",
            "--quiet",
            "--no-tags",
            "--no-recurse-submodules",
            main_url,
            f"+refs/heads/main:{FETCHED_MAIN_REF}",
        )
        main_ref, main_source = FETCHED_MAIN_REF, f"main fetched from {main_url}"
    else:
        main_ref, main_source = "refs/remotes/origin/main", "the checkout's origin/main"
    main = _git_text(
        checkout, "rev-parse", "--verify", "--quiet", f"{main_ref}^{{commit}}"
    )
    resolved = _git(
        checkout, "rev-parse", "--verify", "--quiet", f"{requested}^{{commit}}"
    )
    if resolved.returncode:
        raise PinSourceError(f"{sha} names no commit in {checkout}")
    commit = resolved.stdout.decode("utf-8").strip()
    if not commit.startswith(requested):
        raise PinSourceError(
            f"{sha} resolves to {commit}, which it does not abbreviate"
        )
    ancestry = _git(checkout, "merge-base", "--is-ancestor", commit, main)
    if ancestry.returncode == 1:
        raise PinSourceError(
            f"{commit} is not an ancestor of {main_source} ({main}); only "
            f"commits merged into {GITHUB_REPOSITORY} main can be packaged"
        )
    if ancestry.returncode:
        raise PinSourceError(
            "git merge-base failed: "
            + ancestry.stderr.decode("utf-8", "replace").strip()
        )
    entry = _git_text(
        checkout, "ls-tree", "--full-tree", commit, "--", GITHUB_WORKFLOW_PATH
    )
    fields = entry.split(None, 3)
    if len(fields) != 4 or fields[3] != GITHUB_WORKFLOW_PATH:
        raise PinSourceError(f"{commit} has no {GITHUB_WORKFLOW_PATH}")
    mode, kind, blob = fields[:3]
    if (mode, kind) != ("100644", "blob"):
        raise PinSourceError(
            f"{GITHUB_WORKFLOW_PATH} at {commit} is a {kind} with mode {mode}, "
            "not a regular file"
        )
    data = _git(checkout, "cat-file", "blob", blob)
    if data.returncode or git_blob_id(data.stdout) != blob:
        raise PinSourceError(f"blob {blob} cannot be read or does not hash to its id")
    return PinSource(commit, main, main_source, data.stdout, blob)


def source_workflow_directory() -> Path:
    """This checkout's packaged-workflow directory, which ci-pin add writes."""

    result = _git(WORKFLOW_DIRECTORY, "rev-parse", "--show-toplevel")
    if result.returncode:
        raise PinSourceError(
            f"{WORKFLOW_DIRECTORY} is not in a git checkout; run ci-pin add from "
            "an axiom-encode checkout (uv run axiom-encode ci-pin add ...)"
        )
    top = Path(result.stdout.decode("utf-8").strip())
    expected = top / "src" / "axiom_encode" / "ci_parity_workflows"
    if expected.resolve() != WORKFLOW_DIRECTORY.resolve():
        raise PinSourceError(
            f"{WORKFLOW_DIRECTORY} is not {top}'s packaged-workflow directory"
        )
    return WORKFLOW_DIRECTORY


def _directory_templates(
    directory: Path, extra: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    """The template pins' workflows as packaged in ``directory`` (or ``extra``)."""

    manifest = load_manifest(directory)
    templates: dict[str, Mapping[str, Any]] = {}
    for sha in TEMPLATE_PINS:
        if sha in extra:
            templates[sha] = extra[sha]
        elif sha in manifest:
            templates[sha] = load_workflow(read_packaged(manifest[sha], directory))
        else:
            raise PinSourceError(
                f"template pin {sha} is declared in TEMPLATE_FEATURES but not "
                f"packaged; package it with ci-pin add {sha} --template"
            )
    return templates


def add_pin(
    sha: str,
    github_checkout: Path,
    *,
    fetch: bool = True,
    dry_run: bool = False,
    template: bool = False,
    destination: Path | None = None,
    main_url: str = GITHUB_MAIN_URL,
    out: TextIO | None = None,
) -> int:
    """Package ``sha``'s validate-rulespec.yml if all of its glue is reviewed.

    With ``template``, ``sha`` must already be declared in TEMPLATE_FEATURES
    (the reviewed executor change that supports it): its glue becomes
    reviewed, and reviewed-glue.json is regenerated from every template.
    Returns 0 when the pin is (or already was) packaged, 1 when it is not
    admitted. Nothing is written unless it is admitted.
    """

    out = out or sys.stdout
    source = read_pin_source(sha, github_checkout, fetch=fetch, main_url=main_url)
    directory = destination or (
        WORKFLOW_DIRECTORY if dry_run else source_workflow_directory()
    )
    print(
        f"{GITHUB_REPOSITORY}@{source.sha}\n"
        f"  {GITHUB_WORKFLOW_PATH}: blob {source.blob}\n"
        f"  an ancestor of {source.main_source} ({source.main})",
        file=out,
    )
    manifest = load_manifest(directory)
    if source.sha in manifest:
        packaged = manifest[source.sha]
        if packaged.blob != source.blob:
            raise PinSourceError(
                f"{source.sha} is packaged as blob {packaged.blob}, but .github "
                f"has blob {source.blob}"
            )
        read_packaged(packaged, directory)
        print(f"already packaged as {packaged.file}", file=out)
        return 0
    if template and source.sha not in TEMPLATE_FEATURES:
        raise PinSourceError(
            f"{source.sha} is not declared in TEMPLATE_FEATURES; review the "
            "executor change it needs and declare its features there first"
        )
    try:
        payload = load_workflow(source.data)
    except (GlueError, yaml.YAMLError) as exc:
        print(f"NOT admitted: the workflow does not parse: {exc}", file=out)
        return 1
    existing = load_reviewed_glue(directory)
    entries = list(existing)
    try:
        if template:
            templates = _directory_templates(directory, {source.sha: payload})
            entries = template_glue(templates, existing)
        else:
            templates = _directory_templates(directory, {})
        context = build_review_context(templates, entries)
    except GlueError as exc:
        print(f"NOT admitted: {exc}", file=out)
        return 1
    admission = admit(payload, context)
    if (
        template
        and admission.admitted
        and admission.features != dict(TEMPLATE_FEATURES[source.sha])
    ):
        admission.problems.append(
            Problem(
                "features",
                f"its glue implies features {admission.features}, not the "
                f"declared {dict(TEMPLATE_FEATURES[source.sha])}",
            )
        )
    print(render_report(admission, context, label=source.sha[:8]), file=out, end="")
    if not admission.admitted:
        print(
            "\nNothing was written. Review each difference above, extend ci's "
            "executor where it must change, then package the pin as a template "
            "(docs/ci-parity.md, 'Adding a validate-rulespec pin').",
            file=out,
        )
        return 1
    on = [feature for feature, value in admission.features.items() if value]
    removed = ", ".join(admission.removed_groups) or "none"
    print(
        f"template {admission.template[:8] if admission.template else '-'}, "
        f"removed feature groups: {removed}\n"
        f"features: {', '.join(on) or 'none'}",
        file=out,
    )
    added = [entry for entry in entries if entry not in set(existing)]
    if template:
        print(f"{len(added)} newly reviewed glue unit(s):", file=out)
        for entry in added:
            label = entry.step or entry.job or "workflow"
            print(
                f"  {entry.kind} {entry.job or ''} {label} {entry.digest[:12]}",
                file=out,
            )
    if dry_run:
        print("dry run: nothing written", file=out)
        return 0
    file_name = workflow_file_name(source.sha)
    path = directory / file_name
    if path.exists() and path.read_bytes() != source.data:
        raise PinSourceError(f"{path} exists with other content")
    path.write_bytes(source.data)
    manifest[source.sha] = PackagedPin(source.sha, file_name, source.blob)
    write_manifest(manifest, directory)
    if template:
        write_reviewed_glue(entries, directory)
    print(f"packaged {file_name} and registered it in {MANIFEST_NAME}", file=out)
    return 0


def list_pins(out: TextIO) -> int:
    for sha, pin in sorted(packaged_manifest().items()):
        if sha in LEGACY_PINS:
            print(f"{sha}  {pin.file}  legacy: ci's fixed gate sequence", file=out)
            continue
        admission = packaged_admission(sha)
        if not admission.admitted:
            print(f"{sha}  {pin.file}  NOT ADMITTED", file=out)
            continue
        on = [feature for feature, value in admission.features.items() if value]
        template = admission.template[:8] if admission.template else "-"
        if sha in TEMPLATE_FEATURES:
            origin = "template"
        else:
            removed = ", ".join(admission.removed_groups)
            origin = f"like {template}" + (f" without {removed}" if removed else "")
        print(
            f"{sha}  {pin.file}  {origin}; features: {', '.join(on) or 'none'}",
            file=out,
        )
    return 0
