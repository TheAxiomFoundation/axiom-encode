"""Failing asserts in this suite must not print a whole environment.

See ``tests/environment_assertions.py`` for why. The check below models what
pytest prints when an assert fails. Its assertion rewriting
(``_pytest/assertion/rewrite.py``) prints:

- each operand of a comparison or an arithmetic operator, and each operand of
  ``and`` / ``or`` that was evaluated, true ones included;
- each call, with its function and its arguments;
- each attribute, with the object it is read from;
- local names, and module globals that are neither callable nor named (a dict
  is printed; a function or a module is not, apart from a fixture function,
  printed as ``<pytest_fixture(<function ...>)>``);
- the message's value.

Anything else (a subscript, a comprehension, a literal, an f-string) is printed
as its result alone. Its traceback, in the default ``auto`` style, also prints
the arguments of the test and of the function the assert failed in, whatever
the assert says; frames between them are shortened to their source line.
Below ``-vvv`` each argument is cut to 240 characters, which still holds an
environment's first variables.

An assert is flagged when something it prints may be, or may hold, a whole
environment, and a test is flagged (at its ``def``) when its arguments may.
The check tracks, for every value, the keys and positions that lead from it
to an environment inside it. Environments start as:

- ``os.environ``, under any module path;
- the result of a call whose name says it builds one (``*_env``,
  ``*_environment``, ``scrub_*``);
- a parameter named ``env`` or ``environment``;
- the ``env`` keyword argument a fake records (``**kwargs``, ``.kwargs``,
  ``.call_args``). This counts in a function that launches a process: one that
  names a launcher, patches ``subprocess``, passes ``env=`` or reads an
  ``"env"`` key. A fake defined in a fixture, a class or a helper counts too,
  where a function that launches reads back what it recorded.

They are followed through copies and filtered copies, assignments, and
containers a test or a module-level helper stores them in or returns. They
are followed into helpers' parameters, into what fixtures return or yield
(this module's and its conftests'), and into what a module-level class's
methods store on its instances. A plain class's instance prints as its name
and address, so only reading its attributes counts. Reads back out by key or
position follow them too. One value of an environment (``os.environ["X"]``,
a ``.values()`` item) may be printed; a container or a joined text of them
holds them all. A fake whose arguments hold an environment checks its call
with ``expect`` rather than ``assert``, so the failing frame holds no
environment; ``expect``'s own arguments are printed, and checked here.
``test_the_check_agrees_with_pytest`` runs pytest on examples of each shape to
keep the model honest.

The model flags more than pytest prints in one case. When ``==`` fails between
two strings, sequences, sets or dicts, or a set comparison fails, pytest prints
its own diff of the two results in place of what their operands would print.
The operands' types are not known here, so such asserts are flagged as well.

It misses what it cannot see. A test that does not itself launch anything may
read a mock the code under test launched through; ``mock.call_args.kwargs``
then holds an environment, and pytest prints it as the receiver of
``.get(...)``. Read such values by subscript. ``--tb=long`` and
``--showlocals`` print every frame's arguments or locals, the code under
test's included, and an exception raised inside the code under test or the
standard library (``subprocess.run(..., check=True)``) prints the arguments of
the frame that raised it. Those are outside what an assert can control.
"""

from __future__ import annotations

import ast
import functools
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.environment_assertions import changed_names

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

# A path is the keys and positions leading from a value to an environment
# inside it: () is the value itself, ("env",) its "env" item, (ANY, 1, "env")
# the "env" item of the second element of any of its items. A path ending in
# FRAGMENT leads to one of an environment's values instead: printing one value
# is allowed, but a container of them (``list(os.environ.values())``) holds
# them all.
# A path through HIDDEN passes through an object that prints as its class
# and address, not its attributes: ``recorder`` prints nothing it records,
# but ``recorder.calls`` does.
Path_ = tuple[object, ...]
ANY = "*"
FRAGMENT = "<value>"
HIDDEN = "<object>"
# Deeper paths are dropped, which keeps the analysis finite.
MAX_DEPTH = 6
# Keyword arguments, and parameters, that carry a subprocess environment.
ENVIRONMENT_KEYS = frozenset({"env", "environment"})
# Where the environment sits in the keyword arguments a fake records.
KWARGS: frozenset[Path_] = frozenset((key,) for key in ENVIRONMENT_KEYS)
# Calls returning their first argument's contents.
COPIES = frozenset({"dict", "copy", "deepcopy", "MappingProxyType"})
# Calls returning their first argument's items, by position.
SEQUENCES = frozenset({"list", "tuple", "sorted", "reversed", "set", "frozenset"})
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
# This package's own launchers take an environment too (``_run_cli``,
# ``_run_embedded_python``, ``run_rulespec_compile`` and others).
LAUNCHES = re.compile(
    r"subprocess\.|run_process|Popen|check_output|check_call|run_cli|run_embedded"
    r"|run_subprocess|run_rulespec_compile|execute_rulespec_test|python_build_of"
)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _descend(paths: Iterable[Path_], key: object) -> set[Path_]:
    """The paths left after reading ``key`` from a value with ``paths``."""
    return {
        path[1:]
        for path in paths
        if path
        and path[0] not in (FRAGMENT, HIDDEN)
        and (key == ANY or path[0] == ANY or path[0] == key)
    }


def _holds(paths: Iterable[Path_]) -> bool:
    """Whether printing a value with ``paths`` may print a whole environment."""
    return any(path != (FRAGMENT,) and HIDDEN not in path for path in paths)


def _text(found: Iterable[set[Path_]]) -> set[Path_]:
    """The paths of text formatted from values with each of ``found``."""
    found = list(found)
    if any(_holds(paths) for paths in found):
        return {()}
    return {(FRAGMENT,)} if any(found) else set()


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


def _name(node: ast.AST) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    return getattr(target, "id", getattr(target, "attr", ""))


def _shows_attributes(owner: ast.ClassDef) -> bool:
    """Whether printing an instance of ``owner`` may print its attributes.

    A plain class (or an exception) prints as its name and address; a
    dataclass, a class with its own ``__repr__``, or one built on another
    class may print what it holds.
    """
    if any(getattr(node, "name", "") == "__repr__" for node in owner.body):
        return True
    if any(_name(decorator) != "final" for decorator in owner.decorator_list):
        return True
    return any(
        _name(base) != "object" and not _name(base).endswith(("Error", "Exception"))
        for base in owner.bases
    )


def _is_fixture(function: ast.AST) -> bool:
    return any(
        _name(decorator) == "fixture"
        for decorator in getattr(function, "decorator_list", [])
    )


class _Check:
    def __init__(
        self,
        tree: ast.Module,
        fixtures: Mapping[str, set[Path_]] | None = None,
        *,
        recorded: bool = False,
    ) -> None:
        self.tree = tree
        # Whether every fake's recorded keyword arguments may hold an
        # environment, not only a launching function's own. A function the
        # module calls by name is not a fake: its callers are in view.
        self.recorded = recorded
        self.called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
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
        self.classes = {
            node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
        }
        # What each module-level helper returns, and what it stores into
        # each of its parameters (such as a fake ``main`` installer filling
        # the caller's record).
        self.returns: dict[str, set[Path_]] = {}
        self.filled: dict[str, dict[str, set[Path_]]] = {}
        # What each fixture provides to the tests naming it (a conftest's
        # first, then this module's own), and what each class's methods
        # store into its instances (such as a fake recording its calls).
        self.fixtures: dict[str, set[Path_]] = dict(fixtures or {})
        self.instances: dict[str, set[Path_]] = {}
        # What the module's calls pass each helper's parameters.
        self.passed: dict[str, dict[str, set[Path_]]] = {}
        self.passing = False
        self.globals = _Scope()
        self.scopes: dict[ast.AST, _Scope] = {}
        changed = True
        while changed:
            self.passing = False
            changed = self._fill(self.globals, _own_nodes(tree))
            for function in self.functions:
                scope = self._bind(function)
                self.scopes[function] = scope
                changed |= self._export(function, scope)
            for name, owner in self.classes.items():
                stored = {
                    path
                    for method in owner.body
                    if method in self.scopes
                    and _parameters(method)
                    and not any(
                        getattr(decorator, "id", "") == "staticmethod"
                        for decorator in method.decorator_list
                    )
                    for path in self.scopes[method].get(_parameters(method)[0])
                }
                if not _shows_attributes(owner):
                    stored = _under(HIDDEN, stored)
                changed |= stored != self.instances.get(name, set())
                self.instances[name] = stored
            changed |= self.passing

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
            owner = {
                path[1:] if path[:1] == (HIDDEN,) else path
                for path in self.paths(node.value, scope)
            }
            return _descend(owner, f".{node.attr}")
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
            return _text(
                self.paths(part.value, scope)
                for part in node.values
                if isinstance(part, ast.FormattedValue)
            )
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Mod):
                return _text([self.paths(node.right, scope)])
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
            # An environment's items and values hold its values: each one is
            # a fragment, and all of them together are the environment.
            whole = () in receiver
            if name == "items":
                return ({(), (ANY, 1, FRAGMENT)} if whole else set()) | {
                    (ANY, 1, *path[1:]) for path in receiver if path
                }
            if name == "values":
                return ({(), (ANY, FRAGMENT)} if whole else set()) | {
                    (ANY, *path[1:]) for path in receiver if path
                }
            # Joined text holds every element's value.
            if name == "join" and node.args:
                return {()} if _descend(first, ANY) else set()
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
        if name in TEXT:
            return _text(
                [
                    *arguments,
                    *(self.paths(keyword.value, scope) for keyword in node.keywords),
                ]
            )
        if BUILDER.search(name) and not NOT_BUILDER.search(name):
            return {()}
        if isinstance(function, ast.Name):
            return self.returns.get(name, set()) | self.instances.get(name, set())
        return set()

    # -- binding ------------------------------------------------------------

    def _seeds(self, function: ast.AST) -> dict[str, set[Path_]]:
        """What each of ``function``'s parameters may be passed.

        pytest passes a parameter named after a fixture its value; a helper's
        parameter may be passed what any call in the module passes it; a
        parameter named ``env`` or ``environment`` is taken to be one.
        """
        passed = self.passed.get(getattr(function, "name", None), {})
        arguments = function.args
        return {
            argument.arg: self.fixtures.get(argument.arg, set())
            | passed.get(argument.arg, set())
            | ({()} if argument.arg in ENVIRONMENT_KEYS else set())
            for argument in arguments.posonlyargs
            + arguments.args
            + arguments.kwonlyargs
        }

    def _bind(self, function: ast.AST) -> _Scope:
        launches = _launches(function)
        scope = _Scope(self.globals.bindings, launches=self.recorded or launches)
        for name, paths in self._seeds(function).items():
            scope.bind(name, paths)
        for node in ast.walk(function):
            if isinstance(node, FUNCTIONS):
                arguments = node.args
                for argument in (
                    arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                ):
                    if argument.arg in ENVIRONMENT_KEYS:
                        scope.bind(argument.arg, {()})
                called = node is function and function.name in self.called
                if arguments.kwarg is not None and (
                    launches or (self.recorded and not called)
                ):
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
        changed = False
        if _is_fixture(function):
            provided = {
                path
                for node in _own_nodes(function)
                if isinstance(node, (ast.Return, ast.Yield))
                for path in self.paths(node.value, scope)
            }
            changed = provided != self.fixtures.get(function.name, set())
            self.fixtures[function.name] = provided
        if self.helpers.get(getattr(function, "name", None)) is not function:
            return changed
        returned = {
            path
            for node in _own_nodes(function)
            if isinstance(node, ast.Return)
            for path in self.paths(node.value, scope)
        }
        # What a call passes back in includes what an earlier pass filled, so
        # filling only grows.
        known = self.filled.get(function.name, {})
        filled = {
            name: known.get(name, set()) | (scope.get(name) - seeded)
            for name, seeded in self._seeds(function).items()
        }
        filled = {name: paths for name, paths in filled.items() if paths}
        changed |= returned != self.returns.get(function.name, set())
        changed |= filled != known
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
        if isinstance(function, ast.Name) and function.id in self.helpers:
            self._pass(node, scope)
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

    def _pass(self, node: ast.Call, scope: _Scope) -> None:
        """Record what a call to a module helper passes its parameters."""
        helper = self.helpers[node.func.id]
        passed = self.passed.setdefault(helper.name, {})
        names = _parameters(helper)
        arguments = [
            *(
                (names[position], argument)
                for position, argument in enumerate(node.args)
                if position < len(names) and not isinstance(argument, ast.Starred)
            ),
            *((keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg),
        ]
        for name, argument in arguments:
            paths = self.paths(argument, scope) - passed.get(name, set())
            if paths:
                passed.setdefault(name, set()).update(paths)
                self.passing = True

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

    @staticmethod
    def _arguments_hold(function: ast.AST, scope: _Scope) -> bool:
        """Whether printing ``function``'s arguments may print an environment."""
        arguments = function.args
        names = [
            argument.arg
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                arguments.vararg,
                arguments.kwarg,
            ]
            if argument is not None
        ]
        return any(_holds(scope.get(name)) for name in names)

    def leaks(self) -> list[int]:
        """Lines of asserts, and tests, that may print a whole environment."""
        lines = []
        for function in self.functions:
            if self.recorded and not _launches(function):
                continue
            scope = self.scopes[function]
            # pytest's traceback shows the arguments of the failing test, and
            # of the function whose assert failed, whatever the assert says.
            if function.name.startswith("test") and self._arguments_hold(
                function, scope
            ):
                lines.append(function.lineno)
            for frame in ast.walk(function):
                if isinstance(
                    frame, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and self._arguments_hold(frame, scope):
                    lines.extend(
                        node.lineno
                        for node in _own_nodes(frame)
                        if isinstance(node, ast.Assert)
                    )
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "expect"
                ):
                    # ``expect`` shows its arguments' values, as its own.
                    shown = [*node.args, *(k.value for k in node.keywords)]
                    if any(_holds(self.paths(part, scope)) for part in shown):
                        lines.append(node.lineno)
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
                if any(_holds(self.paths(part, scope)) for part in shown):
                    lines.append(node.lineno)
        return sorted(set(lines))


def environment_leaks(
    source: str, fixtures: Mapping[str, set[Path_]] | None = None
) -> list[int]:
    """Lines of asserts in ``source`` that may print a whole environment.

    ``fixtures`` are what a conftest's fixtures provide (see
    ``fixture_environments``).
    """
    tree = ast.parse(source)
    lines = set(_Check(tree, fixtures).leaks())
    # A fake may record calls in a fixture, a helper or a class that does not
    # itself launch anything; a test that does launch reads them back.
    if _launches(tree):
        lines |= set(_Check(tree, fixtures, recorded=True).leaks())
    return sorted(lines)


def fixture_environments(
    source: str, fixtures: Mapping[str, set[Path_]] | None = None
) -> dict[str, set[Path_]]:
    """What the fixtures ``source`` defines, or inherits, provide."""
    return _Check(ast.parse(source), fixtures).fixtures


@functools.cache
def _conftest_fixtures(directory: Path) -> dict[str, set[Path_]]:
    """What the fixtures of the conftest files governing ``directory`` provide."""
    fixtures: dict[str, set[Path_]] = {}
    for folder in reversed([directory, *directory.parents]):
        conftest = folder / "conftest.py"
        if (folder == REPO or REPO in folder.parents) and conftest.is_file():
            source = conftest.read_text(encoding="utf-8")
            fixtures = fixture_environments(source, fixtures)
    return fixtures


def test_no_failing_assert_prints_a_whole_environment() -> None:
    leaks = [
        f"{path.relative_to(REPO)}:{line}"
        for path in sorted(TESTS.rglob("*.py"))
        if "fixtures" not in path.relative_to(TESTS).parts
        for line in environment_leaks(
            path.read_text(encoding="utf-8"), _conftest_fixtures(path.parent)
        )
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

import pytest

from tests.environment_assertions import changed_names, expect

GLOBAL_ENVIRONMENT = dict(os.environ)
helpers = SimpleNamespace(child_environment=lambda: dict(os.environ))
runner = SimpleNamespace(run_embedded=lambda script, **kwargs: 0)


def record_into(record):
    record["environ"] = dict(os.environ)


def recorder():
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    fake_run(["uv"], env=dict(os.environ))
    return calls


def copy_of(mapping):
    return dict(mapping)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *, environment):
        self.calls.append({"environment": environment})


def make_recorder():
    return Recorder()


@pytest.fixture
def saved():
    return dict(os.environ)


@pytest.fixture
def kept():
    yield dict(os.environ)


@pytest.fixture
def embedded(monkeypatch):
    calls = []
    real = runner.run_embedded

    def spy(script, **kwargs):
        calls.append({"script": script, **kwargs})
        return real(script, **kwargs)

    monkeypatch.setattr(runner, "run_embedded", spy)
    return calls


@pytest.fixture
def saved_names():
    return set(os.environ)


@pytest.fixture
def embedded_names(monkeypatch):
    calls = []
    real = runner.run_embedded

    def spy(script, *, environment=None):
        calls.append({"script": script, "environment": sorted(environment or {})})
        return real(script, environment=environment)

    monkeypatch.setattr(runner, "run_embedded", spy)
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


def test_leaks_sorted_items():
    pairs = sorted(os.environ.items())
    assert ("AAA_MISSING", "x") in pairs


def test_leaks_values_message():
    names = set(os.environ)
    assert "AAA_MISSING" in names, list(os.environ.values())


def test_leaks_joined():
    lines = "\\n".join(f"{name}={value}" for name, value in os.environ.items())
    assert "AAA_MISSING=x" in lines


def test_leaks_helper_parameter():
    copied = copy_of(os.environ)
    assert "AAA_SENTINEL" not in copied


def test_leaks_fixture_return(saved):
    assert "AAA_SENTINEL" not in saved


def test_leaks_fixture_yield(kept):
    assert "AAA_SENTINEL" not in kept


def test_leaks_fixture_spy(embedded):
    runner.run_embedded("pass", environment=dict(os.environ))
    assert len(embedded) == 0


def test_leaks_class_recorder():
    fake = Recorder()
    fake(environment=dict(os.environ))
    [call] = fake.calls
    assert call["environment"] is None


def test_leaks_helper_instance():
    fake = make_recorder()
    fake(environment=dict(os.environ))
    assert len(fake.calls) == 2


def test_leaks_fixture_argument(saved):
    names = set(saved)
    assert "AAA_SENTINEL" not in names


def test_leaks_fixture_spy_argument(embedded):
    runner.run_embedded("pass", environment=dict(os.environ))
    scripts = [call["script"] for call in embedded]
    assert scripts == []


def test_leaks_helper_argument():
    def check(env):
        names = set(env)
        assert "AAA_MISSING" in names

    check(dict(os.environ))


def test_leaks_fake_assert():
    def fake_run(command, **kwargs):
        assert command == ["cargo"]

    fake_run(["git"], env=dict(os.environ))


def test_leaks_expect_detail():
    environment = dict(os.environ)
    expect("AAA_MISSING" in environment, environment)


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


def test_sorted_names():
    names = sorted(os.environ)
    assert "AAA_MISSING" in names


def test_names_from_items():
    names = [name for name, value in os.environ.items()]
    assert "AAA_MISSING" in names


def test_fixture_names(saved_names):
    assert "AAA_SENTINEL" not in saved_names


def test_fixture_spy_names(embedded_names):
    runner.run_embedded("pass", environment=dict(os.environ))
    assert embedded_names == []


def test_helper_names():
    def check(names):
        assert "AAA_MISSING" in names

    check(sorted(os.environ))


def test_fake_expect():
    def fake_run(command, **kwargs):
        expect(command == ["cargo"], command)

    fake_run(["git"], env=dict(os.environ))


def test_plain_object():
    fake = Recorder()
    fake(environment=dict(os.environ))
    assert fake is None


def test_has_environment():
    fake = make_recorder()
    fake(environment=dict(os.environ))
    [call] = fake.calls
    has_environment = call["environment"] is not None
    assert not has_environment


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
