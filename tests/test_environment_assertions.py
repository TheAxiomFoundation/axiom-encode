"""Failing asserts in this suite must not print a whole environment.

See ``tests/environment_assertions.py`` for why. The check below models what
pytest's assertion rewriting (``_pytest/assertion/rewrite.py``) prints when an
assert fails:

- each operand of a comparison or an arithmetic operator, and each operand of
  ``and`` / ``or`` that was evaluated, true ones included;
- each call, with its function and its arguments;
- each attribute, with the object it is read from;
- local names, and module globals that are neither callable nor named (a dict
  is printed; a function or a module is not);
- the message's value.

Anything else (a subscript, a comprehension, a literal, an f-string) is printed
as its result alone.

An assert is flagged when something it prints may be, or may hold, a whole
environment. The check tracks, for every value, the keys and positions that
lead from it to an environment inside it. Environments start as ``os.environ``
(under any module path), the result of a call whose name says it builds one
(``*_env``, ``*_environment``, ``scrub_*``), a parameter named ``env`` or
``environment``, and the ``env`` keyword argument a fake records (``**kwargs``,
``.kwargs``, ``.call_args``). They are followed through copies and filtered
copies, assignments, containers a test or a module-level helper stores them
in or returns, and reads back out by key or position.
``test_the_check_agrees_with_pytest`` runs pytest on examples of each shape to
keep the model honest.

The model flags more than pytest prints in one case. When ``==`` fails between
two strings, sequences, sets or dicts, or a set comparison fails, pytest prints
its own diff of the two results in place of what their operands would print.
The operands' types are not known here, so such asserts are flagged as well.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.environment_assertions import changed_names

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

# A path is the keys and positions leading from a value to an environment
# inside it: () is the value itself, ("env",) its "env" item, (ANY, 1, "env")
# the "env" item of the second element of any of its items.
Path_ = tuple[object, ...]
ANY = "*"
# Deeper paths are dropped, which keeps the analysis finite.
MAX_DEPTH = 6
# Keyword arguments, and parameters, that carry a subprocess environment.
ENVIRONMENT_KEYS = frozenset({"env", "environment"})
# Where the environment sits in the keyword arguments a fake records.
KWARGS: frozenset[Path_] = frozenset((key,) for key in ENVIRONMENT_KEYS)
# Calls returning their first argument's contents.
COPIES = frozenset({"dict", "copy", "deepcopy", "MappingProxyType"})
# Calls returning their first argument's items, by position.
SEQUENCES = frozenset({"list", "tuple", "sorted", "reversed"})
# Calls returning text that holds their arguments' values.
TEXT = frozenset({"str", "repr", "format", "dumps", "pformat", "saferepr"})
# Calls whose name says they build a process environment, and names that say
# they build something else (a formula's or a rule's "environment", or an
# object read from one).
BUILDER = re.compile(
    r"(?:^|_)(?:env|environ|environment)$|^scrub_\w*(?:env|environment|signing)"
)
NOT_BUILDER = re.compile(r"(?:^|_)(?:formula|rule|dependency|from)(?:_|$)")
# Code that launches processes records an environment with each call it fakes
# (``**kwargs``, ``.kwargs``, ``.call_args``); elsewhere those hold other
# keyword arguments. A function launches when it names a launcher, patches the
# ``subprocess`` module, passes ``env=``, or reads an ``"env"`` key.
LAUNCHERS = frozenset({"run", "Popen", "call", "check_call", "check_output"})
LAUNCHES = re.compile(r"subprocess\.|run_process|Popen|check_output|check_call")
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _descend(paths: Iterable[Path_], key: object) -> set[Path_]:
    """The paths left after reading ``key`` from a value with ``paths``."""
    return {
        path[1:]
        for path in paths
        if path and (key == ANY or path[0] == ANY or path[0] == key)
    }


def _under(key: object, paths: Iterable[Path_]) -> set[Path_]:
    """The paths of a container holding a value with ``paths`` at ``key``."""
    return {(key, *path) for path in paths if len(path) < MAX_DEPTH}


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """``node``'s descendants, not entering nested functions or classes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (*FUNCTIONS, ast.ClassDef)):
            yield from _own_nodes(child)


def _parameters(function: ast.AST) -> list[str]:
    arguments = function.args
    return [arg.arg for arg in arguments.posonlyargs + arguments.args]


def _launches(node: ast.AST) -> bool:
    """Whether code under ``node`` launches, or fakes launching, a process."""
    for child in ast.walk(node):
        if isinstance(child, ast.keyword) and child.arg in ENVIRONMENT_KEYS:
            return True
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            if child.value in ENVIRONMENT_KEYS or LAUNCHES.search(child.value):
                return True
        if isinstance(child, ast.Name) and LAUNCHES.search(child.id):
            return True
        if isinstance(child, ast.Attribute):
            owner = child.value
            owner_name = getattr(owner, "id", getattr(owner, "attr", ""))
            if child.attr == "subprocess" or LAUNCHES.search(child.attr):
                return True
            if child.attr in LAUNCHERS and owner_name == "subprocess":
                return True
    return False


class _Scope:
    """The paths to environments in each name one function binds."""

    def __init__(
        self,
        bindings: dict[str, set[Path_]] | None = None,
        *,
        launches: bool = False,
    ) -> None:
        self.bindings = {name: set(paths) for name, paths in (bindings or {}).items()}
        self.launches = launches

    def get(self, name: str) -> set[Path_]:
        return self.bindings.get(name, set())

    def bind(self, name: str, paths: Iterable[Path_]) -> bool:
        known = self.bindings.setdefault(name, set())
        new = {path for path in paths if len(path) <= MAX_DEPTH} - known
        known |= new
        return bool(new)


class _Check:
    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "os"
            for alias in node.names
            if alias.name in {"environ", "environb"}
        }
        # Imported names are never stored into: storing into ``os.environ``
        # does not make the ``os`` module something pytest prints.
        self.imports = {
            alias.asname or alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.functions = [
            node
            for owner in [tree, *[n for n in tree.body if isinstance(n, ast.ClassDef)]]
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.helpers = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # What each module-level helper returns, and what it stores into
        # each of its parameters (such as a fake ``main`` installer filling
        # the caller's record).
        self.returns: dict[str, set[Path_]] = {}
        self.filled: dict[str, dict[str, set[Path_]]] = {}
        self.globals = _Scope()
        self.scopes: dict[ast.AST, _Scope] = {}
        changed = True
        while changed:
            changed = self._fill(self.globals, _own_nodes(tree))
            for function in self.functions:
                scope = self._bind(function)
                self.scopes[function] = scope
                changed |= self._export(function, scope)

    # -- where environments are --------------------------------------------

    @staticmethod
    def _key(node: ast.AST) -> object:
        if isinstance(node, ast.Constant) and type(node.value) in (str, int):
            return node.value
        return ANY

    def paths(self, node: ast.AST | None, scope: _Scope) -> set[Path_]:
        """The paths from ``node``'s value to the environments it may hold."""
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return {()} if node.id in self.aliases else scope.get(node.id)
        if isinstance(node, ast.Attribute):
            if node.attr in {"environ", "environb", *ENVIRONMENT_KEYS}:
                return {()}
            if scope.launches and node.attr == "kwargs":
                return set(KWARGS)
            if scope.launches and node.attr in {"call_args", "await_args"}:
                return _under(1, KWARGS)
            if scope.launches and node.attr in {
                "call_args_list",
                "await_args_list",
                "mock_calls",
            }:
                return _under(ANY, _under(1, KWARGS))
            return _descend(self.paths(node.value, scope), f".{node.attr}")
        if isinstance(node, (ast.NamedExpr, ast.Starred, ast.Await)):
            return self.paths(node.value, scope)
        if isinstance(node, ast.Subscript):
            inner = self.paths(node.value, scope)
            if isinstance(node.slice, ast.Slice):
                return inner
            return _descend(inner, self._key(node.slice))
        if isinstance(node, ast.Dict):
            found: set[Path_] = set()
            for key, value in zip(node.keys, node.values):
                inner = self.paths(value, scope)
                found |= inner if key is None else _under(self._key(key), inner)
            return found
        if isinstance(node, (ast.Tuple, ast.List)):
            return {
                path
                for position, element in enumerate(node.elts)
                for path in _under(
                    ANY if isinstance(element, ast.Starred) else position,
                    self.paths(element, scope),
                )
            }
        if isinstance(node, ast.Set):
            return {
                path
                for element in node.elts
                for path in _under(ANY, self.paths(element, scope))
            }
        if isinstance(node, ast.DictComp):
            # ``{name: value for name, value in environment.items() if ...}``
            if any(() in self.paths(each.iter, scope) for each in node.generators):
                return {()}
            return _under(ANY, self.paths(node.value, scope))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return _under(ANY, self.paths(node.elt, scope))
        if isinstance(node, ast.JoinedStr):
            formatted = [
                part.value
                for part in node.values
                if isinstance(part, ast.FormattedValue)
            ]
            return {()} if any(self.paths(part, scope) for part in formatted) else set()
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mod):
                return {()} if self.paths(node.right, scope) else set()
            return self.paths(node.left, scope) | self.paths(node.right, scope)
        if isinstance(node, ast.BoolOp):
            return {path for value in node.values for path in self.paths(value, scope)}
        if isinstance(node, ast.IfExp):
            return self.paths(node.body, scope) | self.paths(node.orelse, scope)
        if isinstance(node, ast.Call):
            return self._call_paths(node, scope)
        return set()

    def _call_paths(self, node: ast.Call, scope: _Scope) -> set[Path_]:
        function = node.func
        name = getattr(function, "id", getattr(function, "attr", ""))
        arguments = [self.paths(argument, scope) for argument in node.args]
        first = arguments[0] if arguments else set()
        if isinstance(function, ast.Attribute):
            receiver = self.paths(function.value, scope)
            if name == "copy" and not node.args:
                return receiver
            if name in {"get", "pop", "setdefault"} and node.args:
                default = arguments[1] if len(arguments) > 1 else set()
                return _descend(receiver, self._key(node.args[0])) | default
            # An environment's items and values hold its values.
            whole = {()} if () in receiver else set()
            if name == "items":
                return whole | {(ANY, 1, *path[1:]) for path in receiver if path}
            if name == "values":
                return whole | {(ANY, *path[1:]) for path in receiver if path}
        if name in COPIES and (node.args or node.keywords):
            found = set(first)
            for keyword in node.keywords:
                inner = self.paths(keyword.value, scope)
                found |= inner if keyword.arg is None else _under(keyword.arg, inner)
            return found
        if name in SEQUENCES and node.args:
            return _under(ANY, _descend(first, ANY))
        if name == "next" and node.args:
            return _descend(first, ANY)
        if name in TEXT and (
            any(arguments)
            or any(self.paths(keyword.value, scope) for keyword in node.keywords)
        ):
            return {()}
        if BUILDER.search(name) and not NOT_BUILDER.search(name):
            return {()}
        if isinstance(function, ast.Name):
            return set(self.returns.get(name, set()))
        return set()

    # -- binding ------------------------------------------------------------

    def _bind(self, function: ast.AST) -> _Scope:
        scope = _Scope(self.globals.bindings, launches=_launches(function))
        for node in ast.walk(function):
            if isinstance(node, FUNCTIONS):
                arguments = node.args
                for argument in (
                    arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                ):
                    if argument.arg in ENVIRONMENT_KEYS:
                        scope.bind(argument.arg, {()})
                if arguments.kwarg is not None and scope.launches:
                    scope.bind(arguments.kwarg.arg, KWARGS)
        self._fill(scope, list(ast.walk(function)))
        return scope

    def _fill(self, scope: _Scope, nodes: Iterable[ast.AST]) -> bool:
        nodes = list(nodes)
        grew = False
        changed = True
        while changed:
            changed = False
            for node in nodes:
                changed |= self._bind_node(node, scope)
            grew |= changed
        return grew

    def _export(self, function: ast.AST, scope: _Scope) -> bool:
        if self.helpers.get(getattr(function, "name", None)) is not function:
            return False
        returned = {
            path
            for node in _own_nodes(function)
            if isinstance(node, ast.Return)
            for path in self.paths(node.value, scope)
        }
        arguments = function.args
        filled = {
            argument.arg: scope.get(argument.arg)
            - ({()} if argument.arg in ENVIRONMENT_KEYS else set())
            for argument in arguments.posonlyargs
            + arguments.args
            + arguments.kwonlyargs
        }
        filled = {name: paths for name, paths in filled.items() if paths}
        changed = returned != self.returns.get(function.name, set())
        changed |= filled != self.filled.get(function.name, {})
        self.returns[function.name] = returned
        self.filled[function.name] = filled
        return changed

    def _place(self, node: ast.AST) -> tuple[str, Path_] | None:
        """The name a target or receiver lives in, and the path to it there."""
        keys: list[object] = []
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    return None
                keys.append(self._key(node.slice))
            else:
                keys.append(f".{node.attr}")
            node = node.value
        if isinstance(node, ast.Name) and node.id not in self.imports:
            return node.id, tuple(reversed(keys))
        return None

    def _store(self, target: ast.AST, paths: set[Path_], scope: _Scope) -> bool:
        if not paths:
            return False
        if isinstance(target, (ast.Tuple, ast.List)):
            changed = False
            for position, element in enumerate(target.elts):
                if isinstance(element, ast.Starred):
                    changed |= self._store(
                        element.value, _under(ANY, _descend(paths, ANY)), scope
                    )
                else:
                    changed |= self._store(element, _descend(paths, position), scope)
            return changed
        place = self._place(target)
        if place is None:
            return False
        name, keys = place
        found = paths
        for key in reversed(keys):
            found = _under(key, found)
        return scope.bind(name, found)

    def _bind_node(self, node: ast.AST, scope: _Scope) -> bool:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found = self.paths(node.value, scope)
            return any([self._store(target, found, scope) for target in targets])
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            return self._store(
                node.target, _descend(self.paths(node.iter, scope), ANY), scope
            )
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            return self._store(
                node.optional_vars, self.paths(node.context_expr, scope), scope
            )
        if not isinstance(node, ast.Call):
            return False
        changed = False
        function = node.func
        if isinstance(function, ast.Attribute):
            stored = self._stored(node, scope)
            if stored:
                changed |= self._store(function.value, stored, scope)
        if isinstance(function, ast.Name) and self.filled.get(function.id):
            helper = self.helpers[function.id]
            filled = self.filled[function.id]
            positions = _parameters(helper)
            for position, argument in enumerate(node.args):
                if position < len(positions) and positions[position] in filled:
                    changed |= self._store(argument, filled[positions[position]], scope)
            for keyword in node.keywords:
                if keyword.arg in filled:
                    changed |= self._store(keyword.value, filled[keyword.arg], scope)
        return changed

    def _stored(self, node: ast.Call, scope: _Scope) -> set[Path_]:
        """What a method call stores into its receiver."""
        method = node.func.attr
        arguments = [self.paths(argument, scope) for argument in node.args]
        if method == "update":
            found = {path for paths in arguments for path in paths}
            for keyword in node.keywords:
                inner = self.paths(keyword.value, scope)
                found |= inner if keyword.arg is None else _under(keyword.arg, inner)
            return found
        if method in {"append", "add", "insert", "appendleft"} and arguments:
            return _under(ANY, arguments[-1])
        if method == "extend" and arguments:
            return _under(ANY, _descend(arguments[0], ANY))
        if method == "setdefault" and len(arguments) > 1:
            return _under(self._key(node.args[0]), arguments[1])
        return set()

    # -- what pytest prints -------------------------------------------------

    def printed(self, node: ast.AST, shown: bool = True) -> Iterator[ast.AST]:
        """The expressions pytest prints when an assert on ``node`` fails.

        With ``shown`` false, ``node``'s own value is left out but its parts
        are not.
        """
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                yield from self.printed(value)
        elif isinstance(node, ast.UnaryOp):
            yield from self.printed(node.operand)
        elif isinstance(node, ast.BinOp):
            yield from self.printed(node.left)
            yield from self.printed(node.right)
        elif isinstance(node, ast.Compare):
            yield from self.printed(node.left)
            for comparator in node.comparators:
                yield from self.printed(comparator)
        elif isinstance(node, ast.Call):
            if shown:
                yield node
            yield from self.printed(node.func)
            for argument in node.args:
                yield from self.printed(argument)
            for keyword in node.keywords:
                yield from self.printed(keyword.value)
        elif isinstance(node, ast.Starred):
            yield from self.printed(node.value)
        elif isinstance(node, ast.Attribute):
            if shown:
                yield node
            yield from self.printed(node.value)
        elif shown:
            yield node

    def leaks(self) -> list[int]:
        """Lines of asserts that may print a whole environment."""
        lines = []
        for function in self.functions:
            scope = self.scopes[function]
            for node in ast.walk(function):
                if not isinstance(node, ast.Assert):
                    continue
                test = node.test
                if isinstance(test, (ast.Name, ast.Subscript)):
                    # ``assert x`` fails only when x is empty.
                    shown = []
                elif (
                    isinstance(test, ast.Compare)
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.IsNot)
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value is None
                ):
                    # ``assert x is not None`` fails only when x is None, but
                    # what x is computed from is printed.
                    shown = list(self.printed(test.left, shown=False))
                else:
                    shown = list(self.printed(test))
                if node.msg is not None:
                    shown.append(node.msg)
                if any(self.paths(part, scope) for part in shown):
                    lines.append(node.lineno)
        return sorted(set(lines))


def environment_leaks(source: str) -> list[int]:
    return _Check(ast.parse(source)).leaks()


def test_no_failing_assert_prints_a_whole_environment() -> None:
    leaks = [
        f"{path.relative_to(REPO)}:{line}"
        for path in sorted(TESTS.rglob("*.py"))
        if "fixtures" not in path.relative_to(TESTS).parts
        for line in environment_leaks(path.read_text(encoding="utf-8"))
    ]
    # Reduce the environment to names first; see tests/environment_assertions.py.
    assert leaks == []


SENTINEL = "aaa-sentinel-value-not-secret"

# Each test fails. Those named ``test_leaks_*`` print the sentinel's value.
# Those named ``test_flagged_*`` don't, but are flagged (see the docstring).
EXAMPLES = """
import os
from types import SimpleNamespace
from unittest.mock import Mock

from tests.environment_assertions import changed_names

GLOBAL_ENVIRONMENT = dict(os.environ)
helpers = SimpleNamespace(child_environment=lambda: dict(os.environ))


def record_into(record):
    record["environ"] = dict(os.environ)


def recorder():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    fake_run(["uv"], env=dict(os.environ))
    return calls


def test_leaks_restored_environment():
    before = dict(os.environ)
    os.environ["AAA_CHANGED"] = "1"
    assert dict(os.environ) == before


def test_leaks_membership():
    assert "AAA_MISSING" in os.environ


def test_leaks_lookup():
    assert os.environ.get("AAA_MISSING") == "x"


def test_leaks_lookup_is_not_none():
    assert os.environ.get("AAA_MISSING") is not None


def test_leaks_module_path():
    module = SimpleNamespace(os=os)
    assert "AAA_MISSING" in module.os.environ


def test_leaks_copy_by_name():
    environment = os.environ.copy()
    assert "AAA_SENTINEL" not in environment


def test_leaks_filtered_copy():
    assert {
        name: value for name, value in os.environ.items() if name != "AAA_OTHER"
    } == {}


def test_leaks_global():
    assert "AAA_SENTINEL" not in GLOBAL_ENVIRONMENT


def test_leaks_builder():
    environment = helpers.child_environment()
    assert "AAA_SENTINEL" not in environment


def test_leaks_record():
    record = {}
    record["environ"] = dict(os.environ)
    assert record == {}


def test_leaks_recorded_keyword():
    kwargs = {"env": {**os.environ, "EXTRA": "1"}}
    assert "AAA_SENTINEL" not in kwargs["env"]


def test_leaks_named_parameter():
    seen = {}

    def fake(command, *, env):
        seen.update(env=env)

    fake(["git"], env=dict(os.environ))
    assert "AAA_SENTINEL" not in seen["env"]


def test_leaks_merged_kwargs():
    seen = {}

    def fake(command, **kwargs):
        seen.update(kwargs)

    fake(["git"], env=dict(os.environ))
    assert "AAA_SENTINEL" not in seen["env"]


def test_leaks_nested_record():
    calls = []

    def fake(command, **kwargs):
        calls.append((command, kwargs))

    fake(["git"], env=dict(os.environ))
    assert "AAA_SENTINEL" not in calls[0][1]["env"]


def test_leaks_call_args():
    run = Mock()
    run(["git"], env=dict(os.environ))
    assert "AAA_SENTINEL" not in run.call_args[1]["env"]


def test_leaks_count():
    environments = []
    environments.append(dict(os.environ))
    assert len(environments) == 2


def test_leaks_returned_recorder():
    calls = recorder()
    assert len(calls) == 2


def test_leaks_call_argument():
    def probe(env):
        return 1

    assert probe(env=dict(os.environ)) == 0


def test_leaks_helper_record():
    record = {}
    record_into(record)
    assert "AAA_SENTINEL" not in record["environ"]


def test_leaks_is_none():
    environment = dict(os.environ)
    assert environment is None


def test_leaks_and_operand():
    environment = dict(os.environ)
    names = set(environment)
    assert environment is not None and "AAA_MISSING" in names


def test_leaks_formatted_message():
    environment = dict(os.environ)
    names = set(environment)
    assert "AAA_MISSING" in names, f"missing from {environment}"


def test_leaks_recorded_position():
    calls = []
    calls.append((["git"], dict(os.environ)))
    for command, environment in calls:
        assert "AAA_SENTINEL" not in environment


def test_restored_names():
    before = dict(os.environ)
    os.environ["AAA_CHANGED_TOO"] = "1"
    changed = changed_names(before, os.environ)
    assert changed == []


def test_membership_names():
    names = set(os.environ)
    assert "AAA_MISSING" in names


def test_lookup_value():
    value = os.environ.get("AAA_MISSING")
    assert value == "x"


def test_one_value():
    environment = dict(os.environ)
    assert environment["AAA_OTHER"] == "y"


def test_recorded_names():
    kwargs = {"env": dict(os.environ)}
    names = set(kwargs["env"])
    assert "AAA_SENTINEL" not in names


def test_named_parameter_names():
    seen = {}

    def fake(command, *, env):
        seen.update(env=env)

    fake(["git"], env=dict(os.environ))
    names = set(seen["env"])
    assert "AAA_SENTINEL" not in names


def test_comprehension():
    environment = dict(os.environ)
    assert not any(name.startswith("AAA_") for name in environment)


def test_count_first():
    environments = [dict(os.environ)]
    count = len(environments)
    assert count == 2


def test_count_commands():
    calls = recorder()
    commands = [command for command, _kwargs in calls]
    assert len(commands) == 2


def test_call_args_positional():
    run = Mock()
    run(["git"], env=dict(os.environ))
    assert run.call_args[0] == (["cargo"],)


def test_is_not_none():
    kwargs = {}
    environment = kwargs.get("env")
    assert environment is not None


def test_other_position():
    calls = []
    calls.append((["git"], dict(os.environ)))
    for command, environment in calls:
        assert command == ["cargo"], command


def test_truthiness():
    environments = []
    if not os.environ:
        environments.append(dict(os.environ))
    assert environments


def test_flagged_diffed_lists():
    assert sorted(os.environ) == []


def test_flagged_diffed_strings():
    assert str(len(os.environ)) == "0"
"""


def _failure_output(output: str) -> dict[str, str]:
    """Each failed test's report and short-summary line, by test name."""
    reports: dict[str, str] = {}
    name = None
    for line in output.splitlines():
        if header := re.fullmatch(r"_+ (test_\w+) _+", line):
            name = header.group(1)
        elif line.startswith("="):
            name = None
        elif summary := re.match(r"FAILED \S+::(test_\w+)", line):
            reports[summary.group(1)] = reports.get(summary.group(1), "") + line
        elif name is not None:
            reports[name] = reports.get(name, "") + line + "\n"
    return reports


def test_the_check_agrees_with_pytest(tmp_path: Path) -> None:
    examples = tmp_path / "test_examples.py"
    examples.write_text(EXAMPLES)
    tree = ast.parse(EXAMPLES)
    tests = {
        node.name: range(node.lineno, node.end_lineno + 1)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    flagged = {
        name
        for line in environment_leaks(EXAMPLES)
        for name, lines in tests.items()
        if line in lines
    }

    # The most verbose assertion output, in an environment holding nothing else.
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-vv", "-p", "no:cacheprovider"]
        + ["-c", os.devnull, "--rootdir", str(tmp_path), str(examples)],
        cwd=tmp_path,
        env={
            "AAA_SENTINEL": SENTINEL,
            "AAA_OTHER": "x",
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "PYTHONPATH": str(REPO),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    reports = _failure_output(completed.stdout)
    printed = {name for name, text in reports.items() if SENTINEL in text}
    diffed = {name for name in tests if "_flagged_" in name}

    assert sorted(reports) == sorted(tests)
    assert sorted(printed) == sorted(name for name in tests if "_leaks_" in name)
    assert sorted(flagged) == sorted(printed | diffed)


def test_changed_names_example() -> None:
    before = {"KEPT": "1", "CHANGED": "a", "REMOVED": "x", "UNSET": None}
    after = {"KEPT": "1", "CHANGED": "b", "ADDED": "y"}
    assert changed_names(before, after) == ["ADDED", "CHANGED", "REMOVED", "UNSET"]


environments = st.dictionaries(
    st.sampled_from(["A", "B", "C", "D"]), st.sampled_from([None, "", "0", "1"])
)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(before=environments, after=environments)
def test_changed_names_is_empty_exactly_when_equal(
    before: dict[str, str | None], after: dict[str, str | None]
) -> None:
    changed = changed_names(before, after)
    assert (changed == []) == (before == after)
    assert changed == changed_names(after, before) == sorted(set(changed))
    assert set(changed) <= before.keys() | after.keys()
    for name in changed:
        assert (name in before, before.get(name)) != (name in after, after.get(name))
