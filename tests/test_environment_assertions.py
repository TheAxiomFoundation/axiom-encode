"""Failing tests in this suite must not print a whole environment.

See ``tests/environment_assertions.py`` for why. The check below models what
pytest prints when a test fails. Its assertion rewriting
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
as its result alone. Its traceback, in the default ``auto`` style, prints the
arguments of the first and last frames it shows, whatever the assert says, and
shortens those between to their source line. It shows an exception's frames
from the test on, less those that set ``__tracebackhide__``
(``Function._traceback_filter`` in ``_pytest/python.py``): the test's arguments
are printed, and those of the frame the exception was raised in. For each
exception it was chained from (``raise ... from``, or one raised while handling
another), it shows that exception's frames from the first in the test's file,
which, where the code under test wrapped what a fake raised, is the fake's.
``Code.getargs`` (``_pytest/_code/code.py``) adds the flags for ``*args`` and
``**kwargs`` (4 and 8) to the count of arguments, so a function taking either
has its first few locals printed as well, and keyword-only arguments are
printed only then. A plain dict is printed to its first four keys in sorted
order at any verbosity, and an object printed through its own repr (a
dataclass record, ``os.environ``) in full at ``-vvv``; below ``-vvv`` each
argument is cut to 240 characters. Either holds an environment's first
variables. A mock raises the exception it is given as its side effect from
``unittest.mock``'s own frame, whose arguments are its call's. The standard
library's launchers raise from frames of their own whose arguments hold the
environment they are passed: ``subprocess.run`` and ``check_call`` when they
check and the program fails, and ``Popen``'s ``_execute_child`` (or
``_posix_spawn``) when the program can't be started. ``os.environ``'s own
methods are Python code too: a read or ``del`` of a variable that is not set
raises ``KeyError`` from a frame whose ``self`` is the whole environment, and
``monkeypatch.delenv`` raises it from a frame it passes ``os.environ``.
Captured output and logs are printed with a failure, and warnings in the
session's summary.

So any function an exception may pass through may have its arguments printed:
a test, a fake, a helper, a fixture. One that is passed an environment takes
it through ``hides_environment``, which passes it ``Names`` in each
environment's place; a mock standing in for a launcher raises through
``raising``. The check flags:

- an assert, when something it prints may be, or may hold, a whole
  environment;
- a function (a method or a lambda too), at its ``def``, when its printed
  arguments may. They are read from the compiled function, and every
  parameter counts too, should pytest come to print keyword-only ones. A
  method's ``self`` counts unless its class prints as its name and address.
  ``hides_environment``, decorating a function or wrapping a lambda, is
  modelled as it behaves (``_mask``): it hides what is held under the name
  ``env`` or ``environment``, so an environment passed under another name is
  still flagged. It does so only as the decorator nearest the ``def``: above
  another, it wraps that decorator's wrapper, whose parameters need not be
  the function's;
- a ``raise``, when the exception it raises or its cause may hold one, or what
  the call building it is passed, whatever its name; so too ``raising``'s
  exception, a mock's ``side_effect``, a generator's ``throw``, a ``match=``
  pattern, ``pytest.fail``'s message, and what ``print``, ``pprint``,
  ``sys.stdout.write``, ``warnings.warn`` or a logger is passed. An exception
  holds what it is built with and what its class's methods store on it;
- in a function that launches, a mock side effect that is an exception rather
  than ``raising(...)``;
- a call to one of the standard library's launchers (``subprocess.run``,
  ``Popen``, ``call``, ``check_call``, ``check_output``, asyncio's
  ``create_subprocess_exec`` and ``create_subprocess_shell``), by its
  module's name, whatever the module is imported as, or a name the module
  binds it to (``_REAL_RUN = subprocess.run``), whose ``env`` may hold one. A
  launcher reads its ``env`` as a mapping, so it is passed a ``Names``;
- a call handing what may hold one to code out of view: a function the module
  imports, or a method of anything but a container or what is known to be an
  instance of a class the module defines (``self``, ``cls``, ``super()``, a
  call to the class, a name every binding of which is one), a mock's
  ``assert_called_with``, ``functools.partial`` and ``os.execvpe`` among
  them. Should that code raise, pytest prints the arguments of the frame it
  raised in, so a test hands it a ``Names`` (or a minimal environment of its
  own);
- a read or ``del`` of a variable of ``os.environ`` (or a name bound to it) by
  subscript, a one-argument ``pop`` or a ``popitem`` of it, and
  ``monkeypatch.delenv`` unless it is passed ``raising=False``. A test reads a
  variable through ``environment_variable`` and unsets one through
  ``unset_variable``, which raise the same ``KeyError`` holding the name
  alone.

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
  where a function that launches reads back what it recorded, and so does a
  fake any function hands to code out of view (``_Check._handed``), in a
  module that launches something itself;
- the ``env`` or ``environment`` field of a record that prints its fields (a
  dataclass, this module's or one imported from ``axiom_encode``), wherever
  one is built or a parameter is annotated with its class.

They are followed through copies and filtered copies (``dataclasses.asdict``,
``vars`` and ``getattr`` too), assignments, and containers a test or a
module-level helper stores them in or returns, or a function stores in a
module global. They are followed into helpers' parameters, nested helpers'
included, into what fixtures return or yield (this module's and its
conftests'), into a function a fixture provides as the tests call it, and into
what a class's methods store on its instances and its properties return, its
module bases' included, a class defined within a function too. A plain
class's instance, or one built on this module's plain classes alone, prints
as its name and address, so only reading its attributes counts. Reads back out
by key or position follow them too. One value of an environment
(``os.environ["X"]``, a ``.values()`` item, a value read by one of its own
names) may be printed; a container or a joined text of them holds them all.
Its names alone hold no values, and neither do values read by names the test
chose. A ``Names`` prints names alone, and a ``dataclasses.replace`` that swaps
one in holds no environment, but what is read back out of it (a copy, its
items, a value, what it keeps) is followed as before.
``test_the_check_agrees_with_pytest`` runs pytest at ``-vvv`` on examples of
each shape to keep the model honest.

The model flags more than pytest prints in seven cases. When ``==`` fails
between two strings, sequences, sets or dicts, or a set comparison fails,
pytest prints its own diff of the two results in place of what their operands
would print. The operands' types are not known here, so such asserts are
flagged as well. A keyword-only parameter of a function taking neither
``*args`` nor ``**kwargs`` is not printed, but counts here. A function defined
within another shares its names: a parameter named like one of the outer
function's locals holds what that local does, unmasked by
``hides_environment``. A launcher, or code out of view, that a test or a
fixture has replaced with a fake counts as what it replaced. A method of
something the check cannot tell from code out of view counts as it, though it
run no Python frame of its own (``dict.fromkeys``) or be one of this module's
(on an instance a fixture returns). An object that prints its attributes, one
of them an environment, counts where it is passed to a function
``hides_environment`` wraps, which refuses it. And a function is
flagged whether or not anything in it can raise, unless all it does is return
constants, and so is a read of ``os.environ`` by subscript, whether or not the
variable is set, and of a name the module binds to ``os.environ`` anywhere,
whatever it is bound to where it is read.

It misses what it cannot see. A test that does not itself launch anything may
read a mock the code under test launched through; ``mock.call_args.kwargs``
then holds an environment, and pytest prints it as the receiver of
``.get(...)``. Read such values by subscript. A method that code under test
looks up on a fake it is handed (``runner.run``) is not taken to be handed,
and a function passed to ``hides_environment`` other than as a decorator or
a lambda is not taken to be hidden. In a module that launches nothing itself,
a fake handed to code out of view is not taken to be passed an environment.
``hides_environment`` reads through mappings, lists, tuples, named tuples and
dataclass records, and refuses, raising ``TypeError``, what it cannot copy
with the environment hidden: a list or tuple subclass (a mock's ``call``), a
record whose ``__post_init__`` replaces what it is given, and an object that
prints its attributes. The check takes every container to be one it reads
through; an environment inside a generator or a view reaches the function as
it is. ``--tb=long``, ``--showlocals`` and ``--fulltrace`` print every
frame's arguments or locals, the code under test's included. An environment
the code under test builds for itself is printed when an exception passes a
frame of its own that holds it: the frame that raised, or, for an exception it
wraps that was raised with no frame in the test's file to start from (as
``raising``'s is), the frame that wrapped it. A fake defined in the test's
file and hidden would give pytest that frame to start from instead. A
launcher reached through ``getattr(subprocess, name)``, or bound to a name
other than by plain assignment, is not taken to be one, and ``os.environ``
bound to anything but a name (an attribute, a parameter's default) is not
taken to be it. Setting variables to what the platform rejects raises from
``os.environ``'s frames too: an empty name or one with ``=`` in it, a name or
value holding a null byte, or a pair given to ``update`` that is not two
items, through ``os.environ`` itself, its ``update`` or ``setdefault``, or
``monkeypatch.setenv``. Setting is not flagged, so a test sets variables to
names and strings it has written itself. What a launched program prints is
not taken to hold an environment: the signing supervisor's trusted runtime
prints its own and its child's, so its tests print a run without its output
and read the report through ``Names``.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import json
import os
import re
import subprocess
import sys
import types
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.environment_assertions import Names, changed_names, hides_environment

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent

# A path is the keys and positions leading from a value to an environment
# inside it: () is the value itself, ("env",) its "env" item, (ANY, 1, "env")
# the "env" item of the second element of any of its items. A path ending in
# FRAGMENT leads to one of an environment's values instead: printing one value
# is allowed, but a container of them (``list(os.environ.values())``) holds
# them all.
# A path ending in KEY leads to one of an environment's names, which may be
# printed; a value read by one is a FRAGMENT.
# A path through HIDDEN passes through an object that prints as its class
# and address, not its attributes: ``recorder`` prints nothing it records,
# but ``recorder.calls`` does.
# A path through NAMES passes through a ``Names``, which prints its names
# alone; what is read out of it (a copy, its items, a value) is not hidden.
Path_ = tuple[object, ...]
ANY = "*"
FRAGMENT = "<value>"
KEY = "<name>"
HIDDEN = "<object>"
NAMES = "<names>"
# Deeper paths are dropped, which keeps the analysis finite.
MAX_DEPTH = 6
# Keyword arguments, and parameters, that carry a subprocess environment.
ENVIRONMENT_KEYS = frozenset({"env", "environment"})
# Where the environment sits in the keyword arguments a fake records.
KWARGS: frozenset[Path_] = frozenset((key,) for key in ENVIRONMENT_KEYS)
# Calls returning their first argument's contents.
COPIES = frozenset({"dict", "copy", "deepcopy", "MappingProxyType", "OrderedDict"})
# Calls returning their first argument's items, by position.
SEQUENCES = frozenset({"list", "tuple", "sorted", "reversed", "set", "frozenset"})
# Calls returning text that holds their arguments' values.
TEXT = frozenset(
    {"str", "repr", "format", "dumps", "pformat", "saferepr", "escape", "quote"}
)
# Exceptions print the arguments they are built with.
EXCEPTION = re.compile(
    r"(?:Error|Exception|ExceptionGroup|Exit|Expired|Warning|Interrupt|Iteration)$"
)
# Calls whose arguments pytest prints even when no assert fails in them:
# captured output, captured logs and the warnings summary.
OUTPUT = frozenset(
    {
        "print",
        "warn",
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "critical",
        "log",
        "pprint",
        "pp",
    }
)
# ``tests/environment_assertions.py``'s decorator, which passes a function
# ``Names`` in place of each environment, and its side effect for mocks.
HIDES = "hides_environment"
RAISING = "raising"
# Calls that install what they are passed in place of something their other
# arguments name (``monkeypatch.setattr(subprocess, "run", fake)``,
# ``patch("...subprocess.run", side_effect=fake)``), and mocks, which call
# their side effect.
INSTALLERS = frozenset({"setattr", "setitem", "patch", "object"})
MOCKS = frozenset(
    {"Mock", "MagicMock", "AsyncMock", "NonCallableMock", "create_autospec"}
)
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
# The standard library's launchers raise in frames of their own, whose
# arguments pytest prints, the environment among them: ``run`` and
# ``check_call`` when the program fails and they check, and every one when
# the program can't be started.
SPAWNERS = frozenset(
    {("subprocess", name) for name in LAUNCHERS}
    | {
        (module, name)
        for module in ("asyncio", "subprocess")
        for name in ("create_subprocess_exec", "create_subprocess_shell")
    }
)
# Calls the check follows as it does data: what they return holds what they
# are passed, and their own frames are hidden or print nothing.
FOLLOWED = frozenset(
    {
        "Names",
        "changed_names",
        HIDES,
        RAISING,
        "replace",
        "asdict",
        "astuple",
        "getattr",
        "vars",
        *COPIES,
        *SEQUENCES,
        *TEXT,
    }
)
# Methods of the standard containers, which run no Python frame of their own.
CONTAINER_METHODS = frozenset(
    {
        "append",
        "appendleft",
        "extend",
        "insert",
        "add",
        "update",
        "setdefault",
        "get",
        "pop",
        "popitem",
        "remove",
        "discard",
        "clear",
        "items",
        "keys",
        "values",
        "count",
        "index",
        "join",
    }
)
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _unwrapped(paths: Iterable[Path_]) -> set[Path_]:
    """The paths of what is read out of a value with ``paths``, ``Names`` too."""
    return {path[1:] if path[:1] == (NAMES,) else path for path in paths}


def _descend(paths: Iterable[Path_], key: object) -> set[Path_]:
    """The paths left after reading ``key`` from a value with ``paths``."""
    return {
        path[1:]
        for path in _unwrapped(paths)
        if path
        and path[0] not in (FRAGMENT, KEY, HIDDEN)
        and (key == ANY or path[0] == ANY or path[0] == key)
    }


def _holds(paths: Iterable[Path_]) -> bool:
    """Whether printing a value with ``paths`` may print a whole environment."""
    return any(
        path != (FRAGMENT,)
        and KEY not in path
        and HIDDEN not in path
        and NAMES not in path
        for path in paths
    )


def _elements(paths: Iterable[Path_]) -> set[Path_]:
    """The paths of the elements iterating a value with ``paths`` yields."""
    paths = _unwrapped(paths)
    return _descend(paths, ANY) | ({(KEY,)} if () in paths else set())


def _text(found: Iterable[set[Path_]]) -> set[Path_]:
    """The paths of text formatted from values with each of ``found``."""
    found = [{path for path in paths if KEY not in path} for paths in found]
    if any(_holds(paths) for paths in found):
        return {()}
    return {(FRAGMENT,)} if any(found) else set()


def _under(key: object, paths: Iterable[Path_]) -> set[Path_]:
    """The paths of a container holding a value with ``paths`` at ``key``."""
    return {(key, *path) for path in paths if len(path) < MAX_DEPTH}


def _mask(path: Path_, name: object = None) -> Path_:
    """``path`` once ``hides_environment`` has passed its value on.

    It replaces with ``Names`` each environment held under the name ``name``
    (an argument's, a key's or a record field's), reading through mappings,
    lists, tuples and records; the check takes every container it reads
    through to be one of those.
    """
    if not path:
        return (NAMES,) if name in ENVIRONMENT_KEYS else path
    head, rest = path[0], path[1:]
    if head in (NAMES, FRAGMENT, KEY, HIDDEN):
        return path
    if head == ANY or isinstance(head, int):
        return (head, *_mask(rest))
    if isinstance(head, str) and head.startswith("."):
        # A record is read through its environment fields alone.
        return (head, *_mask(rest, head[1:])) if head[1:] in ENVIRONMENT_KEYS else path
    return (head, *_mask(rest, head))


def _attributes(paths: Iterable[Path_]) -> set[Path_]:
    """The paths of ``vars()`` of a value with ``paths``: its attributes."""
    found = set()
    for path in paths:
        if path[:1] == (HIDDEN,):
            path = path[1:]
        if path[:1] == (NAMES,):
            found.add(("_environment", *path[1:]))
        elif path and isinstance(path[0], str) and path[0].startswith("."):
            found.add((path[0][1:], *path[1:]))
    return found


def _record_items(path: Path_, positions: bool = False) -> Path_:
    """``path`` once ``dataclasses.asdict`` (or ``astuple``) has copied it.

    Each record's fields become keys (or positions); a ``Names`` and an object
    printing as its address are copied as they are.
    """
    for index, head in enumerate(path):
        if head in (NAMES, HIDDEN):
            return path
        if isinstance(head, str) and head.startswith("."):
            field = ANY if positions else head[1:]
            return (*path[:index], field, *_record_items(path[index + 1 :], positions))
    return path


def _launcher(text: object) -> bool:
    """Whether ``text`` names a launcher (``"subprocess.run"``, ``"_run_cli"``)."""
    return isinstance(text, str) and (
        bool(LAUNCHES.search(text)) or text.rpartition(".")[2] in LAUNCHERS
    )


def _inert(function: ast.AST) -> bool:
    """Whether no exception can pass through ``function``'s frame: it calls
    nothing and reads nothing, and only returns constants, or tuples or lists
    of them."""

    def literal(node: ast.AST | None) -> bool:
        if node is None or isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return all(literal(element) for element in node.elts)
        return False

    if isinstance(function, ast.Lambda):
        return literal(function.body)
    return all(
        isinstance(statement, ast.Pass)
        or (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
        )
        or (isinstance(statement, ast.Return) and literal(statement.value))
        for statement in function.body
    )


def _is_fail(node: ast.AST, fails: Iterable[str] = ()) -> bool:
    """Whether ``node`` calls ``pytest.fail``, or one of the names ``fails``
    a module imports it as."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in fails
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "fail"
        and getattr(node.func.value, "id", "") == "pytest"
    )


def _locals(function: ast.AST) -> set[str]:
    """The names ``function``, or a function within it, binds as its own."""
    shared = {
        name
        for node in ast.walk(function)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }
    stored = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    parameters = {
        argument.arg
        for node in ast.walk(function)
        if isinstance(node, FUNCTIONS)
        for argument in _all_parameters(node)
    }
    return (stored | parameters) - shared


def _all_parameters(function: ast.AST) -> list[ast.arg]:
    arguments = function.args
    return [
        argument
        for argument in [
            *arguments.posonlyargs,
            *arguments.args,
            arguments.vararg,
            *arguments.kwonlyargs,
            arguments.kwarg,
        ]
        if argument is not None
    ]


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """``node``'s descendants, not entering nested functions or classes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (*FUNCTIONS, ast.ClassDef)):
            yield from _own_nodes(child)


def _dotted(node: ast.AST) -> str:
    """``node``'s dotted name (``ci_parity.CliInvocation``), or ``""``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _dotted(node.value)
        return f"{owner}.{node.attr}" if owner else ""
    return ""


def _class_fields(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """The annotated fields, in order, of each class ``tree`` defines whose
    instances print them."""
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    return {
        name: tuple(
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        )
        for name, node in classes.items()
        if _shows_attributes(node, classes)
    }


@functools.cache
def _src_fields(module: str) -> dict[str, tuple[str, ...]]:
    """``_class_fields`` of this package's module ``module``, if it is one."""
    base = REPO / "src" / Path(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return _class_fields(ast.parse(path.read_text(encoding="utf-8")))
    return {}


def _code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    """``code`` and the code objects of every function defined within it."""
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _code_objects(constant)


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
        # The functions and classes defined within the one this scope binds,
        # by name, and its own parameters.
        self.nested: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.parameters: set[str] = set()

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


def _shows_attributes(
    owner: ast.ClassDef, classes: Mapping[str, ast.ClassDef] | None = None
) -> bool:
    """Whether printing an instance of ``owner`` may print its attributes.

    A plain class (or an exception) prints as its name and address; a
    dataclass, a class with its own ``__repr__``, or one built on another
    class may print what it holds, unless that class is one of ``classes``
    that does not.
    """
    if any(getattr(node, "name", "") == "__repr__" for node in owner.body):
        return True
    if any(_name(decorator) != "final" for decorator in owner.decorator_list):
        return True
    classes = classes or {}
    return any(
        _shows_attributes(classes[_dotted(base)], classes)
        if _dotted(base) in classes
        else _name(base) != "object"
        and not _name(base).endswith(("Error", "Exception"))
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
        # Each function's code object, by name and first line (a decorated
        # function's starts at its first decorator).
        self.codes = {
            (code.co_name, code.co_firstlineno): code
            for code in _code_objects(compile(tree, "<test>", "exec"))
        }
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
        # Names bound to ``os.environ`` itself (``values = os.environ``),
        # anywhere in the module.
        self.live: set[str] = set()
        while True:
            bound = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                and node.value is not None
                and self._is_environ(node.value)
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            if bound <= self.live:
                break
            self.live |= bound
        # Imported names are never stored into: storing into ``os.environ``
        # does not make the ``os`` module something pytest prints.
        self.imports = {
            alias.asname or alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        # Modules imported under another name (``import subprocess as sp``).
        self.modules = {
            alias.asname: alias.name.rpartition(".")[2]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.asname
        }
        # The standard library's launchers, imported by name, and the names
        # the module binds them to (``_REAL_RUN = subprocess.run``).
        self.spawners = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if (node.module, alias.name) in SPAWNERS
        }
        while True:
            bound = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, (ast.Name, ast.Attribute))
                and self._launcher(_dotted(node.value))
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if bound <= self.spawners:
                break
            self.spawners |= bound
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
        self.owners = {
            method: name
            for name, owner in self.classes.items()
            for method in owner.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # Every class, and the class each method is defined in.
        every_class = [
            node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        self.method_classes = {
            method: owner
            for owner in every_class
            for method in owner.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        # The methods of each class, its bases' in this module included, and
        # the names bound to an instance of one (``fake = Recorder()``).
        own = {
            owner.name: {
                method.name
                for method in owner.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for owner in every_class
        }
        self.bases = {
            owner.name: {_dotted(base) for base in owner.bases} for owner in every_class
        }

        def methods(name: str, seen: frozenset[str] = frozenset()) -> set[str]:
            found = set(own.get(name, set()))
            for base in self.bases.get(name, set()) - seen:
                found |= methods(base, seen | {name})
            return found

        self.class_methods = {name: methods(name) for name in own}
        # A name is an instance of one only if every binding of it, anywhere
        # in the module, is a call to that class.
        built = {
            id(target): node.value.func.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in own
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        bindings: dict[str, set[str | None]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                bindings.setdefault(node.id, set()).add(built.get(id(node)))
            elif isinstance(node, ast.arg):
                bindings.setdefault(node.arg, set()).add(None)
            elif isinstance(node, ast.alias):
                bindings.setdefault(node.asname or node.name, set()).add(None)
            elif isinstance(node, (FUNCTIONS, ast.ClassDef)) and hasattr(node, "name"):
                bindings.setdefault(node.name, set()).add(None)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                if node.name:
                    bindings.setdefault(node.name, set()).add(None)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bindings.setdefault(node.rest, set()).add(None)
        self.typed = {
            name: owner
            for name, owners in bindings.items()
            if len(owners) == 1
            for owner in owners
            if owner is not None
        }
        # This module's exception classes, those defined within a function
        # too, which print what they are built with.
        self.exceptions: set[str] = set()
        while True:
            found = {
                owner.name
                for owner in every_class
                if any(
                    EXCEPTION.search(_name(base)) or _name(base) in self.exceptions
                    for base in owner.bases
                )
            }
            if found <= self.exceptions:
                break
            self.exceptions |= found
        # The names this module calls ``pytest.fail`` by.
        self.fails = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "pytest"
            for alias in node.names
            if alias.name == "fail"
        }
        # Lambdas passed to ``hides_environment``, which hides their
        # environments as it does a decorated function's.
        self.hidden_lambdas = {
            id(argument)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _name(node) == HIDES
            for argument in node.args
            if isinstance(argument, ast.Lambda)
        }
        # Where each node sits, and each name's reads other than to call it: a
        # function read so may be passed, returned or installed, and then
        # called by code out of view (see ``_handed``).
        self.parents = {
            id(child): node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        callees = {
            id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        self.reads: dict[str, list[ast.Name]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in callees
            ):
                self.reads.setdefault(node.id, []).append(node)
        self.handed_attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and id(node) not in callees
        }
        self.handing: dict[int, bool] = {}
        # Classes whose instances print a field named ``env`` or
        # ``environment`` (such as ``CliInvocation``), by the dotted name this
        # module uses: their fields, in order.
        self.carriers = self._carriers(tree)
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
        # The functions each of this module's fixtures provides (``return
        # check``), and what the tests calling one pass its parameters.
        self.provided: dict[str, list[ast.AST]] = {}
        for function in self.functions:
            if _is_fixture(function):
                nested = {
                    node.name: node
                    for node in ast.walk(function)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node is not function
                }
                self.provided[function.name] = [
                    nested[node.value.id]
                    for node in _own_nodes(function)
                    if isinstance(node, (ast.Return, ast.Yield))
                    and isinstance(node.value, ast.Name)
                    and node.value.id in nested
                ]
        self.into: dict[ast.AST, dict[str, set[Path_]]] = {}
        # The module's own globals, which its functions may store into.
        module_names = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
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
                # What it stores into a global (``CALLS.append(...)``), the
                # module's other functions read.
                for name in module_names - _locals(function):
                    changed |= self.globals.bind(name, scope.get(name))
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
                # A property reads as what it returns.
                stored |= {
                    path
                    for method in owner.body
                    if method in self.scopes
                    and any(
                        _name(decorator) in {"property", "cached_property"}
                        for decorator in method.decorator_list
                    )
                    for node in _own_nodes(method)
                    if isinstance(node, ast.Return)
                    for path in _under(
                        f".{method.name}", self.paths(node.value, self.scopes[method])
                    )
                }
                # And it holds what its module bases' methods store.
                stored |= {
                    path[1:] if path[:1] == (HIDDEN,) else path
                    for base in owner.bases
                    for path in self.instances.get(_dotted(base), set())
                }
                if not _shows_attributes(owner, self.classes):
                    stored = _under(HIDDEN, stored)
                changed |= stored != self.instances.get(name, set())
                self.instances[name] = stored
            changed |= self.passing

    # -- where environments are --------------------------------------------

    @staticmethod
    def _carriers(tree: ast.Module) -> dict[str, tuple[str, ...]]:
        carriers: dict[str, tuple[str, ...]] = {}

        def add(prefix: str, classes: Mapping[str, tuple[str, ...]]) -> None:
            for name, fields in classes.items():
                if ENVIRONMENT_KEYS & set(fields):
                    carriers[prefix + name] = fields

        add("", _class_fields(tree))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.partition(".")[0] != "axiom_encode":
                    continue
                classes = _src_fields(node.module)
                for alias in node.names:
                    local = alias.asname or alias.name
                    add("", {local: classes.get(alias.name, ())})
                    add(f"{local}.", _src_fields(f"{node.module}.{alias.name}"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.partition(".")[0] == "axiom_encode":
                        add(f"{alias.asname or alias.name}.", _src_fields(alias.name))
        return carriers

    def _annotated(self, node: ast.AST | None) -> set[Path_]:
        """The paths to environments in a value of the type ``node`` names."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                node = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                return set()
        if node is None:
            return set()
        fields = self.carriers.get(_dotted(node), ())
        if fields:
            return {(f".{field}",) for field in fields if field in ENVIRONMENT_KEYS}
        if isinstance(node, ast.BinOp):
            return self._annotated(node.left) | self._annotated(node.right)
        if isinstance(node, ast.Subscript):
            parts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            inner = {path for part in parts for path in self._annotated(part)}
            if _name(node.value) in {"Optional", "Union", "Annotated", "Final"}:
                return inner
            return _under(ANY, inner)
        return set()

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
            found = self.paths(node.value, scope)
            if node.attr == "__dict__":
                return _attributes(found)
            owner = {path[1:] if path[:1] == (HIDDEN,) else path for path in found}
            if node.attr == "_environment":
                # What a ``Names`` keeps.
                owner |= {
                    ("._environment", *path[1:])
                    for path in owner
                    if path[:1] == (NAMES,)
                }
            return _descend(owner, f".{node.attr}")
        if isinstance(node, (ast.NamedExpr, ast.Starred, ast.Await)):
            return self.paths(node.value, scope)
        if isinstance(node, ast.Subscript):
            inner = _unwrapped(self.paths(node.value, scope))
            if isinstance(node.slice, ast.Slice):
                return inner
            key = self._key(node.slice)
            # One value of an environment, read by one of its names.
            named = (KEY,) in self.paths(node.slice, scope)
            fragment = {(FRAGMENT,)} if named and () in inner else set()
            return _descend(inner, key) | fragment
        if isinstance(node, ast.Dict):
            found: set[Path_] = set()
            for key, value in zip(node.keys, node.values):
                inner = self.paths(value, scope)
                found |= (
                    _unwrapped(inner) if key is None else _under(self._key(key), inner)
                )
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
            receiver = _unwrapped(self.paths(function.value, scope))
            if name == "copy" and not node.args:
                return receiver
            if name == "with_traceback":
                return receiver
            if name in {"get", "pop", "setdefault"} and node.args:
                default = arguments[1] if len(arguments) > 1 else set()
                key = self._key(node.args[0])
                named = (KEY,) in arguments[0]
                fragment = {(FRAGMENT,)} if named and () in receiver else set()
                return _descend(receiver, key) | default | fragment
            # An environment's items and values hold its values: each one is
            # a fragment, and all of them together are the environment.
            whole = () in receiver
            if name == "keys" and whole:
                return {(ANY, KEY)}
            if name == "items":
                return ({(), (ANY, 0, KEY), (ANY, 1, FRAGMENT)} if whole else set()) | {
                    (ANY, 1, *path[1:]) for path in receiver if path
                }
            if name == "values":
                return ({(), (ANY, FRAGMENT)} if whole else set()) | {
                    (ANY, *path[1:]) for path in receiver if path
                }
            # Joined text holds every element's value.
            if name == "join" and node.args:
                values = {path for path in _descend(first, ANY) if KEY not in path}
                return {()} if values else set()
        fields = self.carriers.get(_dotted(function), ())
        if fields:
            # A carrier holds what its environment field is given.
            found = set()
            for position, argument in enumerate(node.args):
                if isinstance(argument, ast.Starred):
                    break
                if position < len(fields) and fields[position] in ENVIRONMENT_KEYS:
                    found |= _under(f".{fields[position]}", arguments[position])
            for keyword in node.keywords:
                if keyword.arg in ENVIRONMENT_KEYS:
                    found |= _under(f".{keyword.arg}", self.paths(keyword.value, scope))
            return found | self.instances.get(_dotted(function), set())
        if (
            name == "replace"
            and node.args
            and _dotted(function)
            in {
                "replace",
                "dataclasses.replace",
            }
        ):
            # ``dataclasses.replace``: the fields given are replaced.
            given = {f".{keyword.arg}" for keyword in node.keywords}
            found = {path for path in first if not path or path[0] not in given}
            for keyword in node.keywords:
                if keyword.arg:
                    found |= _under(f".{keyword.arg}", self.paths(keyword.value, scope))
            return found
        if name in {"asdict", "astuple"} and node.args:
            return {_record_items(path, name == "astuple") for path in first}
        if name == "vars" and node.args:
            return _attributes(first)
        if name == "getattr" and len(node.args) > 1:
            # An attribute read by name, or any attribute.
            attribute = node.args[1]
            default = arguments[2] if len(arguments) > 2 else set()
            if isinstance(attribute, ast.Constant) and isinstance(attribute.value, str):
                read = ast.Attribute(value=node.args[0], attr=attribute.value)
                return self.paths(read, scope) | default
            owner = {path[1:] if path[:1] == (HIDDEN,) else path for path in first}
            return _descend(owner, ANY) | default
        if name == "Names":
            # It prints an environment's names alone.
            return _under(NAMES, first)
        if name in COPIES and (node.args or node.keywords):
            found = _unwrapped(first)
            for keyword in node.keywords:
                inner = self.paths(keyword.value, scope)
                found |= (
                    _unwrapped(inner)
                    if keyword.arg is None
                    else _under(keyword.arg, inner)
                )
            return found
        if name in SEQUENCES and node.args:
            return _under(ANY, _elements(first))
        if name == "next" and node.args:
            return _elements(first)
        if name in TEXT:
            return _text(
                [
                    *arguments,
                    *(self.paths(keyword.value, scope) for keyword in node.keywords),
                ]
            )
        if name in {HIDES, RAISING}:
            # A function, which prints as its name and address.
            return set()
        if BUILDER.search(name) and not NOT_BUILDER.search(name):
            return {()}
        # What the methods of a class store on its instances.
        held = set()
        if isinstance(function, ast.Name):
            held = self.instances.get(name, set())
            if name in scope.classes:
                held = held | self._instance(scope.classes[name], scope)
        if EXCEPTION.search(name) or name in self.exceptions:
            # An exception prints what it is built with.
            given = [
                *arguments,
                *(self.paths(keyword.value, scope) for keyword in node.keywords),
            ]
            return {path for argument in given for path in _under(ANY, argument)} | held
        if isinstance(function, ast.Name):
            return self.returns.get(name, set()) | held
        return set()

    def _instance(self, owner: ast.ClassDef, scope: _Scope) -> set[Path_]:
        """What an instance of a class defined within a function holds."""
        stored = {
            path
            for method in owner.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _parameters(method)
            and "staticmethod"
            not in {_name(decorator) for decorator in method.decorator_list}
            for path in scope.get(_parameters(method)[0])
        }
        if _shows_attributes(owner, self.classes):
            return stored
        return _under(HIDDEN, stored)

    # -- binding ------------------------------------------------------------

    def _seeds(self, function: ast.AST) -> dict[str, set[Path_]]:
        """What each of ``function``'s parameters may be passed.

        pytest passes a parameter named after a fixture its value; a helper's
        parameter may be passed what any call in the module passes it; a
        parameter named ``env`` or ``environment`` is taken to be one.
        """
        passed = self.passed.get(getattr(function, "name", None), {})
        arguments = function.args
        holders = {arguments.vararg, arguments.kwarg} - {None}
        seeds = {
            argument.arg: passed.get(argument.arg, set())
            | (
                set()
                if argument in holders
                else self.fixtures.get(argument.arg, set())
                | ({()} if argument.arg in ENVIRONMENT_KEYS else set())
            )
            for argument in _all_parameters(function)
        }
        # A method reads what its class's methods store on the instance.
        owner = self.owners.get(function)
        parameters = _parameters(function)
        if (
            owner is not None
            and parameters
            and "staticmethod"
            not in {_name(decorator) for decorator in function.decorator_list}
        ):
            seeds[parameters[0]] |= {
                path[1:] if path[:1] == (HIDDEN,) else path
                for path in self.instances.get(owner, set())
            }
        return {
            name: self._passed(function, name, paths) for name, paths in seeds.items()
        }

    def _hides(self, function: ast.AST) -> bool:
        """Whether ``function`` takes its arguments through ``hides_environment``.

        Only as the decorator nearest the ``def``: above another, it wraps
        that decorator's wrapper, whose parameters need not be the function's.
        """
        if isinstance(function, ast.Lambda):
            return id(function) in self.hidden_lambdas
        decorators = function.decorator_list
        return bool(decorators) and _name(decorators[-1]) == HIDES

    def _passed(
        self, function: ast.AST, name: str, paths: Iterable[Path_]
    ) -> set[Path_]:
        """What ``function``'s parameter ``name`` holds when it is passed a
        value with ``paths``.

        ``hides_environment`` passes it ``Names`` in each environment's place
        (``_mask``), apart from a method's ``self`` or ``cls``; the elements of
        ``*args`` and the items of ``**kwargs`` are named by their positions
        and keys.
        """
        if not self._hides(function):
            return set(paths)
        parameters = _parameters(function)
        if parameters[:1] in (["self"], ["cls"]) and name == parameters[0]:
            return set(paths)
        arguments = function.args
        holders = {
            argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument
        }
        return {_mask(path, None if name in holders else name) for path in paths}

    def _bind(self, function: ast.AST) -> _Scope:
        launches = _launches(function)
        scope = _Scope(self.globals.bindings, launches=self.recorded or launches)
        for name, paths in self._seeds(function).items():
            scope.bind(name, paths)
        scope.parameters = {argument.arg for argument in _all_parameters(function)}
        scope.nested = {
            node.name: node
            for node in ast.walk(function)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node is not function
        }
        scope.classes = {
            node.name: node
            for node in ast.walk(function)
            if isinstance(node, ast.ClassDef) and node is not function
        }
        for node in ast.walk(function):
            if isinstance(node, FUNCTIONS):
                arguments = node.args
                if getattr(node, "name", None) not in self.called:
                    # Code out of view passes a parameter declared to hold an
                    # environment carrier one.
                    for argument in (
                        arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                    ):
                        scope.bind(
                            argument.arg,
                            self._passed(
                                node, argument.arg, self._annotated(argument.annotation)
                            ),
                        )
                    for argument in (arguments.vararg, arguments.kwarg):
                        if argument is not None:
                            annotated = _under(
                                ANY, self._annotated(argument.annotation)
                            )
                            scope.bind(
                                argument.arg,
                                self._passed(node, argument.arg, annotated),
                            )
                for argument in (
                    arguments.posonlyargs + arguments.args + arguments.kwonlyargs
                ):
                    if argument.arg in ENVIRONMENT_KEYS:
                        scope.bind(argument.arg, self._passed(node, argument.arg, {()}))
                if node is function:
                    # A function the module calls by name is not a fake: its
                    # callers are in view.
                    receives = launches or (
                        self.recorded
                        and function.name not in self.called
                        and self._handed(function)
                    )
                else:
                    # A fake defined within it: one called by name is passed
                    # what its callers pass it.
                    receives = (launches or self.recorded) and (
                        _launches(node) or self._handed(node)
                    )
                if arguments.kwarg is not None and receives:
                    scope.bind(
                        arguments.kwarg.arg,
                        self._passed(node, arguments.kwarg.arg, KWARGS),
                    )
                # What tests pass a function a fixture provides.
                for name, paths in self.into.get(node, {}).items():
                    scope.bind(name, self._passed(node, name, paths))
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
                node.target, _elements(self.paths(node.iter, scope)), scope
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
        if not isinstance(function, ast.Name):
            return changed
        if function.id in scope.nested:
            changed |= self._enter(scope.nested[function.id], node, scope)
        elif function.id in scope.parameters:
            # A test calling a function its fixture provides; a parameter
            # hides the module function of its name.
            for provided in self.provided.get(function.id, []):
                into = self.into.setdefault(provided, {})
                for name, paths in self._arguments(provided, node, scope).items():
                    if paths - into.get(name, set()):
                        into.setdefault(name, set()).update(paths)
                        self.passing = True
            return changed
        elif function.id in self.helpers:
            self._pass(node, scope)
        if self.filled.get(function.id):
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

    def _arguments(
        self, function: ast.AST, node: ast.Call, scope: _Scope
    ) -> dict[str, set[Path_]]:
        """What the call ``node`` passes each of ``function``'s parameters."""
        arguments = function.args
        positional = [*arguments.posonlyargs, *arguments.args]
        named = [*arguments.args, *arguments.kwonlyargs]
        found: dict[str, set[Path_]] = {}

        def add(name: str, paths: set[Path_]) -> None:
            if paths:
                found.setdefault(name, set()).update(paths)

        unknown = False
        for position, argument in enumerate(node.args):
            if isinstance(argument, ast.Starred):
                # It and what follows it land at positions not known here.
                unknown = True
                paths = _descend(self.paths(argument.value, scope), ANY)
            else:
                paths = self.paths(argument, scope)
            if unknown:
                for parameter in positional[position:]:
                    add(parameter.arg, paths)
                if arguments.vararg is not None:
                    add(arguments.vararg.arg, _under(ANY, paths))
            elif position < len(positional):
                add(positional[position].arg, paths)
            elif arguments.vararg is not None:
                add(arguments.vararg.arg, _under(position - len(positional), paths))
        for keyword in node.keywords:
            paths = self.paths(keyword.value, scope)
            if keyword.arg is None:
                # ``**mapping``: each item goes to the parameter it names.
                items = _unwrapped(paths)
                for parameter in named:
                    add(parameter.arg, _descend(items, parameter.arg))
                if arguments.kwarg is not None:
                    add(arguments.kwarg.arg, items)
            elif keyword.arg in {parameter.arg for parameter in named}:
                add(keyword.arg, paths)
            elif arguments.kwarg is not None:
                add(arguments.kwarg.arg, _under(keyword.arg, paths))
        return found

    def _enter(self, function: ast.AST, node: ast.Call, scope: _Scope) -> bool:
        """Bind a nested function's parameters to what ``node`` passes them."""
        changed = False
        for name, paths in self._arguments(function, node, scope).items():
            changed |= scope.bind(name, self._passed(function, name, paths))
        return changed

    def _pass(self, node: ast.Call, scope: _Scope) -> None:
        """Record what a call to a module helper passes its parameters."""
        helper = self.helpers[node.func.id]
        passed = self.passed.setdefault(helper.name, {})
        for name, paths in self._arguments(helper, node, scope).items():
            if paths - passed.get(name, set()):
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
        # An exception's note is printed with it.
        if (
            method in {"append", "add", "insert", "appendleft", "add_note"}
            and arguments
        ):
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

    def _arguments_hold(self, function: ast.AST, scope: _Scope) -> bool:
        """Whether printing ``function``'s arguments may print an environment.

        pytest prints the first names of its code object's ``co_varnames``,
        as many as ``Code.getargs`` counts: assertion rewriting only adds
        names, so the unrewritten code object's are as many or more.
        """
        names = {argument.arg for argument in _all_parameters(function)}
        decorators = getattr(function, "decorator_list", [])
        first = min(node.lineno for node in [function, *decorators])
        code = self.codes.get((getattr(function, "name", "<lambda>"), first))
        if code is not None:
            count = (
                code.co_argcount
                + (code.co_flags & inspect.CO_VARARGS)
                + (code.co_flags & inspect.CO_VARKEYWORDS)
            )
            names.update(code.co_varnames[:count])
        # A plain class's instance, and a class, print as their name.
        owner = self.method_classes.get(function)
        parameters = _parameters(function)
        if (
            owner is not None
            and parameters
            and not _shows_attributes(owner, self.classes)
            and "staticmethod" not in {_name(decorator) for decorator in decorators}
        ):
            names.discard(parameters[0])
        return any(_holds(scope.get(name)) for name in names)

    def _handed(self, frame: ast.AST) -> bool:
        """Whether code out of view may call ``frame`` with what it launches.

        A function or lambda may be, when the module hands it on (``_toward``)
        other than in place of something that does not launch; and so may a
        class's ``__call__``, a method read other than to call it, and the
        ``__init__`` of a class handed on.
        """
        if isinstance(frame, ast.Lambda):
            return self._toward(frame)
        owner = self.method_classes.get(frame)
        if owner is None:
            return any(self._toward(read) for read in self.reads.get(frame.name, []))
        return (
            frame.name == "__call__"
            or frame.name in self.handed_attributes
            or (
                frame.name in {"__init__", "__new__"}
                and any(self._toward(read) for read in self.reads.get(owner.name, []))
            )
        )

    def _toward(self, node: ast.AST, seen: frozenset[int] = frozenset()) -> bool:
        """Whether the function ``node`` reads may reach code out of view
        that launches.

        It may when passed to such code (anything but a call this module can
        read), returned, or installed in place of a launcher: by
        ``setattr``, ``patch`` or a mock's ``side_effect``, by assigning it to
        an attribute or key, or under a name the check cannot read. A function
        passed to a module helper may when that helper launches or hands a
        function on in turn.
        """
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        parent = self.parents.get(id(node))
        if parent is None or isinstance(
            parent,
            (
                # Reading the function's attributes, comparing or formatting
                # it, or testing it, does not hand it on.
                ast.Attribute,
                ast.Compare,
                ast.Expr,
                ast.FormattedValue,
                ast.UnaryOp,
                ast.Assert,
                ast.If,
                ast.While,
            ),
        ):
            return False
        if isinstance(parent, ast.Subscript):
            return parent.value is node and self._toward(parent, seen)
        if isinstance(
            parent,
            (
                ast.List,
                ast.Tuple,
                ast.Set,
                ast.Starred,
                ast.IfExp,
                ast.BoolOp,
                ast.NamedExpr,
                ast.Await,
            ),
        ):
            return self._toward(parent, seen)
        if isinstance(parent, ast.keyword):
            return self._toward_call(self.parents[id(parent)], seen)
        if isinstance(parent, ast.Call):
            return parent.func is not node and self._toward_call(parent, seen)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = (
                parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            )
            return any(self._stored_toward(target, seen) for target in targets)
        if isinstance(parent, ast.Dict):
            keys = [k for k, v in zip(parent.keys, parent.values) if v is node]
            return any(
                not isinstance(key, ast.Constant) or _launcher(key.value)
                for key in keys
            )
        # Returned, a default argument, iterated over, or anything else.
        return True

    def _stored_toward(self, target: ast.AST, seen: frozenset[int]) -> bool:
        if isinstance(target, ast.Name):
            # Followed to where the name is read.
            return any(
                self._toward(read, seen) for read in self.reads.get(target.id, [])
            )
        if isinstance(target, ast.Attribute):
            return _launcher(target.attr)
        if isinstance(target, ast.Subscript):
            key = target.slice
            return not isinstance(key, ast.Constant) or _launcher(key.value)
        return True

    def _toward_call(self, call: ast.Call, seen: frozenset[int]) -> bool:
        name = _name(call)
        if isinstance(call.func, ast.Name) and name in self.helpers:
            helper = self.helpers[name]
            return _launches(helper) or self._hands_on(helper)
        if name in INSTALLERS:
            # What it installs in place of is named by its other arguments.
            named = [
                argument.value
                for argument in [*call.args, *(k.value for k in call.keywords)]
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            return not named or any(_launcher(text) for text in named)
        if name in MOCKS or name in {HIDES, "partial"}:
            # A mock calling it, or a wrapper, is handed on in its place.
            return self._toward(call, seen)
        return True

    def _hands_on(self, helper: ast.AST) -> bool:
        """Whether a module helper hands on a function: one of its parameters
        or one it defines."""
        if id(helper) not in self.handing:
            self.handing[id(helper)] = False
            names = {argument.arg for argument in _all_parameters(helper)} | {
                node.name
                for node in ast.walk(helper)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node is not helper
            }
            within = {id(node) for node in ast.walk(helper)}
            self.handing[id(helper)] = any(
                self._toward(read)
                for name in names
                for read in self.reads.get(name, [])
                if id(read) in within
            )
        return self.handing[id(helper)]

    def _raises(self, node: ast.AST | None, built: set[str]) -> bool:
        """Whether a mock's side effect ``node`` may be an exception, which
        the mock raises from its own frame.

        ``built`` are the names a function binds to exceptions it builds.
        """
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self._raises(element, built) for element in node.elts)
        if isinstance(node, ast.Name) and node.id in built:
            return True
        target = node.func if isinstance(node, ast.Call) else node
        if not isinstance(target, (ast.Name, ast.Attribute)):
            return False
        return bool(EXCEPTION.search(_name(target))) or _name(target) in self.exceptions

    def _hands_out(self, node: ast.Call, function: ast.AST) -> bool:
        """Whether ``node``, in ``function``, hands what it is passed to code
        out of view.

        That is a function the module imports, or a method of anything but a
        container or what is known to be an instance of a class the module
        defines, unless the check follows the call as data. Should that code
        raise, pytest prints the arguments of the frame it raised in, a mock's
        ``assert_called_with`` among them.
        """
        name = _name(node)
        if name in FOLLOWED or name in INSTALLERS or name in MOCKS:
            return False
        if isinstance(node.func, ast.Name):
            return name in self.imports
        return (
            isinstance(node.func, ast.Attribute)
            and name not in CONTAINER_METHODS
            and not self._own_method(node.func, function)
        )

    def _own_method(self, method: ast.Attribute, function: ast.AST) -> bool:
        """Whether ``method``, read in ``function``, is a method of a class
        this module defines (its bases' in the module included), read from
        what is known to be an instance of it: ``self`` or ``cls`` in a method
        of the class or of one defined within ``function``, ``super()`` there
        for the class's bases, a call to the class, or a name every binding of
        which is one."""
        receiver = method.value
        enclosing = {
            node.name for node in ast.walk(function) if isinstance(node, ast.ClassDef)
        }
        if function in self.owners:
            enclosing.add(self.owners[function])
        if isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}:
            owners = enclosing
        elif isinstance(receiver, ast.Call) and _name(receiver) == "super":
            owners = {base for owner in enclosing for base in self.bases[owner]}
        elif isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name):
            owners = {receiver.func.id}
        elif isinstance(receiver, ast.Name):
            owners = {self.typed.get(receiver.id, "")}
        else:
            owners = set()
        return any(
            method.attr in self.class_methods.get(owner, set()) for owner in owners
        )

    def _launcher(self, dotted: str) -> bool:
        """Whether ``dotted`` names one of the standard library's launchers,
        its module imported under any name, or a name bound to one."""
        parts = dotted.split(".")
        if len(parts) >= 2:
            parts[-2] = self.modules.get(parts[-2], parts[-2])
        return tuple(parts[-2:]) in SPAWNERS or dotted in self.spawners

    def _spawns(self, node: ast.Call) -> bool:
        """Whether ``node`` calls one of the standard library's launchers."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.spawners
        return self._launcher(_dotted(node.func))

    def _is_environ(self, node: ast.AST) -> bool:
        """Whether ``node`` is ``os.environ`` (or ``os.environb``) itself, or a
        name bound to it."""
        if isinstance(node, ast.Name):
            return node.id in self.aliases or node.id in self.live
        return (
            isinstance(node, ast.Attribute)
            and node.attr in {"environ", "environb"}
            and _dotted(node.value).rpartition(".")[2] == "os"
        )

    def _environ_raises(self, node: ast.AST) -> bool:
        """Whether ``node`` may raise from a frame whose arguments hold the
        whole of ``os.environ``: its own, for a read or ``del`` of a variable
        that is not set, a one-argument ``pop`` or a ``popitem``, or
        ``monkeypatch``'s, for a ``delenv`` that raises."""
        if isinstance(node, ast.Subscript):
            return not isinstance(node.ctx, ast.Store) and self._is_environ(node.value)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        if node.func.attr == "delenv":
            raising = [
                *node.args[1:2],
                *(k.value for k in node.keywords if k.arg == "raising"),
            ]
            return not any(
                isinstance(value, ast.Constant) and not value.value for value in raising
            )
        return self._is_environ(node.func.value) and (
            (node.func.attr == "pop" and len(node.args) == 1 and not node.keywords)
            or node.func.attr == "popitem"
        )

    def leaks(self) -> list[int]:
        """Lines of asserts, functions and calls that may print a whole
        environment."""
        lines = []
        for function in self.functions:
            scope = self.scopes[function]
            launches = _launches(function)
            # pytest's traceback prints the arguments of the failing test, of
            # the function an exception was raised in, and of the first frame
            # in this module of each exception it was chained from, whatever
            # the assert says. ``pytest.fail`` hides its own frame, so it is
            # its caller's. Any function an exception may pass through counts.
            frames = [
                node
                for node in ast.walk(function)
                if isinstance(node, FUNCTIONS) and not _inert(node)
            ]
            if self.recorded and not launches:
                # Of a function that does not launch anything, only the fakes
                # it hands to code out of view may be passed what that
                # launches.
                frames = [frame for frame in frames if self._handed(frame)]
            lines.extend(
                frame.lineno for frame in frames if self._arguments_hold(frame, scope)
            )
            if self.recorded and not launches:
                continue
            built = {
                target.id
                for node in ast.walk(function)
                if isinstance(node, ast.Assign) and self._raises(node.value, set())
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(function):
                if self._environ_raises(node):
                    lines.append(node.lineno)
                raised: list[ast.AST | None] = []
                shown: list[ast.AST | None] = []
                if isinstance(node, ast.Raise):
                    # The exception raised, and its cause, are printed.
                    raised = [node.exc, node.cause]
                elif isinstance(node, ast.Call):
                    if _name(node) == RAISING:
                        raised = list(node.args[:1])
                    # A mock raises, and prints, the exception it is given;
                    # an expected pattern is printed when it does not match;
                    # a generator's ``throw`` raises what it is passed.
                    raised += [
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in {"side_effect", "match"}
                    ]
                    if getattr(node.func, "attr", "") in {"match", "throw"}:
                        raised += node.args[:1]
                    if self._spawns(node):
                        # Its own frames print the environment it is passed.
                        shown = [
                            keyword.value
                            for keyword in node.keywords
                            if keyword.arg in {"env", None}
                        ]
                    elif self._hands_out(node, function):
                        shown = [*node.args, *(k.value for k in node.keywords)]
                    if (
                        _is_fail(node, self.fails)
                        or _name(node) in OUTPUT
                        or _dotted(node.func)
                        in {"sys.stdout.write", "sys.stderr.write"}
                    ):
                        # ``pytest.fail`` prints its message; captured output,
                        # logs and warnings are printed too.
                        shown = [*node.args, *(k.value for k in node.keywords)]
                    if launches and any(
                        keyword.arg == "side_effect"
                        and self._raises(keyword.value, built)
                        for keyword in node.keywords
                    ):
                        # A mock standing in for a launcher raises from a
                        # frame whose arguments are its call's.
                        lines.append(node.lineno)
                elif isinstance(node, ast.Assign) and any(
                    getattr(target, "attr", "") == "side_effect"
                    for target in node.targets
                ):
                    raised = [node.value]
                    if launches and self._raises(node.value, built):
                        lines.append(node.lineno)
                shown += raised
                # Whatever its name, the call building what is raised prints
                # what it is passed.
                for part in raised:
                    if isinstance(part, ast.Call):
                        shown += [*part.args, *(k.value for k in part.keywords)]
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
        # The helpers' frames that are passed an environment are hidden from
        # pytest's traceback, ``Names``'s own apart, which hold a ``Names``.
        and path != TESTS / "environment_assertions.py"
        for line in environment_leaks(
            path.read_text(encoding="utf-8"), _conftest_fixtures(path.parent)
        )
    ]
    # Reduce the environment to names first; see tests/environment_assertions.py.
    assert leaks == []


# ``re.escape`` leaves it as it is, so an escaped pattern holds it too.
SENTINEL = "aaa_sentinel_value_not_secret"

# Each test fails. Those named ``test_leaks_*`` print the sentinel's value.
# Those named ``test_flagged_*`` don't, but are flagged (see the docstring).
EXAMPLES = """
import asyncio
import dataclasses
import functools
import logging
import os
import pprint
import re
import subprocess
import subprocess as sp
import sys
import warnings
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
from pytest import fail

import under_test
from tests.environment_assertions import (
    Names,
    changed_names,
    environment_variable,
    hides_environment,
    raising,
    unset_variable,
)
from under_test import probe

GLOBAL_ENVIRONMENT = dict(os.environ)
CALLS = []
REAL_RUN = subprocess.run
helpers = SimpleNamespace(child_environment=lambda: dict(os.environ))
runner = SimpleNamespace(run_embedded=lambda script, **kwargs: 0)


def logged(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def fake_global(command, **kwargs):
    CALLS.append(kwargs)


def forwarding(fake):
    def child(command, **kwargs):
        return fake(command, **kwargs)

    return child


def hidden_forwarding(fake):
    @hides_environment
    def child(command, **kwargs):
        return fake(command, **kwargs)

    return child


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
def checker():
    def check(mapping):
        names = set(mapping)
        assert "AAA_MISSING" in names

    return check


@pytest.fixture
def embedded_names(monkeypatch):
    calls = []
    real = runner.run_embedded

    @hides_environment
    def spy(script, *, environment=None):
        calls.append({"script": script, "environment": sorted(environment or {})})
        return real(script, environment=environment)

    monkeypatch.setattr(runner, "run_embedded", spy)
    return calls


@dataclasses.dataclass(frozen=True)
class Invocation:
    command: str
    environment: dict | None = None


@dataclasses.dataclass
class Unset:
    command: str
    environment: dict = dataclasses.field(init=False)


class BatchRecorder:
    def __init__(self):
        self.batches = []

    def __call__(self, invocations: list[Invocation]):
        self.batches.append(list(invocations))

    @property
    def invocations(self):
        return [invocation for batch in self.batches for invocation in batch]


class AuditRecorder(BatchRecorder):
    def __call__(self, invocations: list[Invocation]):
        super().__call__(invocations)


class Captured(Exception):
    pass


class Carrying(Exception):
    def __init__(self):
        super().__init__("carrying")
        self.saved = dict(os.environ)


class Keyworded(Exception):
    def __init__(self, *, detail):
        super().__init__(detail)


class NamesRecorder:
    def __init__(self):
        self.invocations = []

    @hides_environment
    def __call__(self, invocations: list[Invocation]):
        self.invocations.extend(invocations)


class Launch(NamedTuple):
    command: str
    env: dict


@dataclasses.dataclass
class Rebuilt:
    command: str
    env: dict

    def __post_init__(self):
        self.env = dict(self.env)


class Shown:
    def __init__(self, command):
        self.command = command

    def __repr__(self):
        return f"Shown({vars(self)!r})"


class Slotted:
    __slots__ = ("command", "env")

    def __init__(self, command):
        self.command = command

    def __repr__(self):
        return f"Slotted({self.command!r}, {getattr(self, 'env', None)!r})"


class Probe:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self, command):
        self.probe(command, dict(os.environ))


class QuietProbe(Probe):
    def check(self, command):
        super().probe(command, dict(os.environ))


@pytest.fixture
def prober():
    return Probe()


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


def test_leaks_fake_by_accident():
    def fake_run(command, **kwargs):
        return kwargs["missing"]

    fake_run(["git"], env=dict(os.environ))


def test_leaks_chained():
    def fake_run(command, **kwargs):
        raise OSError(command[0])

    under_test.launch(fake_run)


def test_leaks_handed_fake():
    def cargo(command, **kwargs):
        raise FileNotFoundError("cargo missing")

    under_test.launch(forwarding(cargo))


def test_leaks_mock_side_effect():
    run = Mock(side_effect=OSError("no git"))
    run(["git"], env=dict(os.environ))


def test_leaks_mock_side_effect_assigned():
    run = Mock()
    run.side_effect = FileNotFoundError
    run(["git"], env=dict(os.environ))


def test_leaks_nested_class_recorder():
    class Fake:
        def __init__(self):
            self.calls = []

        @hides_environment
        def __call__(self, command, **kwargs):
            self.calls.append(dict(kwargs["env"]))

    fake = Fake()
    fake(["git"], env=dict(os.environ))
    assert len(fake.calls) == 2


def test_leaks_fixture_checker(checker):
    checker(dict(os.environ))


def test_leaks_global_recorder():
    fake_global(["git"], env=dict(os.environ))
    assert len(CALLS) == 2


def test_leaks_asdict():
    record = dataclasses.asdict(Invocation("check", dict(os.environ)))
    assert record is None


def test_leaks_getattr():
    invocation = Invocation("check", dict(os.environ))
    environment = getattr(invocation, "environment")
    assert environment is None


def test_leaks_getattr_fields():
    invocation = Invocation("check", dict(os.environ))
    values = [getattr(invocation, field) for field in ("command", "environment")]
    assert values == []


def test_leaks_names_internals():
    wrapped = Names(os.environ)
    assert wrapped._environment is None


def test_leaks_names_vars():
    wrapped = Names(os.environ)
    assert vars(wrapped) is None


def test_leaks_exception_instance():
    error = Carrying()
    assert error.saved is None


def test_leaks_exception_keyword():
    raise Keyworded(detail=dict(os.environ))


def test_leaks_exception_group():
    raise ExceptionGroup("several", [ValueError(dict(os.environ))])


def test_leaks_nested_exception_class():
    class Boom(Exception):
        pass

    raise Boom(dict(os.environ))


def test_leaks_unsuffixed_exception():
    raise under_test.Refusal(dict(os.environ))


def test_leaks_note():
    error = ValueError("noted")
    error.add_note(str(dict(os.environ)))
    raise error


def test_leaks_with_traceback():
    raise ValueError(dict(os.environ)).with_traceback(None)


def test_leaks_mock_payload():
    Mock(side_effect=RuntimeError(dict(os.environ)))()


def test_leaks_expected_pattern():
    with pytest.raises(ValueError, match=re.escape(str(dict(os.environ)))):
        raise ValueError("other")


def test_leaks_bare_fail():
    fail(str(dict(os.environ)))


def test_leaks_printed():
    print(dict(os.environ))
    assert False


def test_leaks_logged():
    logging.getLogger("examples").warning("%s", dict(os.environ))
    assert False


def test_leaks_warned():
    warnings.warn(str(dict(os.environ)))
    assert False


def test_leaks_fake_local():
    def fake_run(*args, **kwargs):
        environment = dict(os.environ)
        assert args == (["cargo"],)

    fake_run(["git"])


def test_leaks_fake_raise():
    def fake_run(command, **kwargs):
        raise TimeoutError(command)

    fake_run(["git"], env=dict(os.environ))


def test_leaks_fake_fail():
    def fake_run(command, **kwargs):
        pytest.fail(f"unexpected {command}")

    fake_run(["git"], env=dict(os.environ))


def test_leaks_recorded_invocations():
    recorder = BatchRecorder()
    recorder([Invocation("check", dict(os.environ))])
    assert recorder.invocations == []


def test_leaks_inherited_recorder():
    recorder = AuditRecorder()
    recorder([Invocation("check", dict(os.environ))])
    assert recorder.invocations == []


def test_leaks_invocation_attribute():
    invocation = Invocation("check", dict(os.environ))
    assert not invocation.command


def test_leaks_values_by_name():
    values = [os.environ[name] for name in os.environ]
    assert "AAA_MISSING" in values


def test_leaks_nested_helper_mapping():
    def check(mapping):
        names = set(mapping)
        assert "AAA_MISSING" in names

    check(dict(os.environ))


def test_leaks_names_copy():
    wrapped = Names(os.environ)
    captured = dict(wrapped)
    assert captured is None


def test_leaks_raise_payload():
    raise ValueError(dict(os.environ))


def test_leaks_module_exception():
    raise Captured(dict(os.environ))


def test_leaks_fail_message():
    pytest.fail(f"environment: {dict(os.environ)}")


def test_leaks_checked_launch():
    subprocess.run(
        [sys.executable, "-c", "raise SystemExit(1)"], env=dict(os.environ), check=True
    )


def test_leaks_check_call():
    subprocess.check_call([sys.executable, "-c", "raise SystemExit(1)"], env=os.environ)


def test_leaks_missing_program():
    subprocess.run(["/nonexistent/program"], env=dict(os.environ))


def test_leaks_spawned_missing_program():
    subprocess.Popen(["/nonexistent/program"], env=dict(os.environ), close_fds=False)


def test_leaks_aliased_launch():
    REAL_RUN(
        [sys.executable, "-c", "raise SystemExit(1)"], env=dict(os.environ), check=True
    )


def test_leaks_handed_by_position():
    under_test.probe(["git"], dict(os.environ))


def test_leaks_handed_by_keyword():
    under_test.probe(["git"], environment=dict(os.environ))


def test_leaks_handed_to_an_import():
    probe(["git"], os.environ)


def test_leaks_handed_through_a_rebound_name():
    target = Probe()
    target = under_test
    target.probe(["git"], dict(os.environ))


def test_leaks_missing_through_a_bound_name():
    live = os.environ
    assert live["AAA_MISSING"] == "x"


def test_leaks_built_then_raised():
    error = under_test.Refusal(dict(os.environ))
    raise error


def test_leaks_thrown():
    (_ for _ in ()).throw(ValueError(dict(os.environ)))


def test_leaks_pretty_printed():
    pprint.pprint(dict(os.environ))
    assert False


def test_leaks_written():
    sys.stdout.write(str(dict(os.environ)))
    assert False


def test_leaks_mock_assert():
    run = Mock()
    run(["git"])
    run.assert_called_once_with(["git"], env=dict(os.environ))


def test_leaks_module_alias_launch():
    sp.run(
        [sys.executable, "-c", "raise SystemExit(1)"], env=dict(os.environ), check=True
    )


def test_leaks_local_alias_launch():
    launch = subprocess.check_call
    launch([sys.executable, "-c", "raise SystemExit(1)"], env=dict(os.environ))


def test_leaks_partial_launch():
    launch = functools.partial(subprocess.run, env=dict(os.environ), check=True)
    launch([sys.executable, "-c", "raise SystemExit(1)"])


def test_leaks_exec_missing_program():
    os.execvpe("/nonexistent/program", ["program"], dict(os.environ))


def test_leaks_asyncio_launch():
    asyncio.run(
        asyncio.subprocess.create_subprocess_exec(
            "/nonexistent/program", env=dict(os.environ)
        )
    )


def test_leaks_missing_variable():
    assert len(os.environ["AAA_MISSING"]) == 1


def test_leaks_deleted_missing_variable():
    del os.environ["AAA_MISSING"]


def test_leaks_popped_missing_variable():
    os.environ.pop("AAA_MISSING")


def test_leaks_delenv(monkeypatch):
    monkeypatch.delenv("AAA_MISSING")


def test_leaks_hidden_above_another():
    @hides_environment
    @logged
    def check(command, env):
        raise RuntimeError(command)

    check(["git"], dict(os.environ))


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

    @hides_environment
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


def test_flagged_fixture_spy_names(embedded_names):
    runner.run_embedded("pass", environment=dict(os.environ))
    assert embedded_names == []


def test_helper_names():
    def check(names):
        assert "AAA_MISSING" in names

    check(sorted(os.environ))


def test_hidden_fake_assert():
    @hides_environment
    def fake_run(command, **kwargs):
        assert command == ["cargo"], command

    fake_run(["git"], env=dict(os.environ))


def test_hidden_fake_raise():
    @hides_environment
    def fake_run(command, **kwargs):
        raise TimeoutError(command)

    fake_run(["git"], env=dict(os.environ))


def test_hidden_fake_by_accident():
    @hides_environment
    def fake_run(command, *, env):
        return env["AAA_MISSING"]

    fake_run(["git"], env=dict(os.environ))


def test_hidden_lambda():
    fake_run = hides_environment(lambda command, **kwargs: kwargs["missing"])
    fake_run(["git"], env=dict(os.environ))


def test_hidden_default():
    @hides_environment
    def check(command="git", env=dict(os.environ)):
        raise RuntimeError(command)

    check()


def test_hidden_unset_field():
    @hides_environment
    def check(record):
        raise RuntimeError(record.command)

    unset = Unset("git")
    unset.environment = dict(os.environ)
    check(unset)


def test_hidden_checked_launch():
    subprocess.run(
        [sys.executable, "-c", "raise SystemExit(1)"],
        env=Names(dict(os.environ)),
        check=True,
    )


def test_hidden_missing_program():
    subprocess.run(["/nonexistent/program"], env=Names(os.environ))


def test_hidden_spawned_missing_program():
    subprocess.Popen(
        ["/nonexistent/program"], env=Names(dict(os.environ)), close_fds=False
    )


def test_hidden_chained():
    @hides_environment
    def fake_run(command, **kwargs):
        raise OSError(command[0])

    under_test.launch(fake_run)


def test_hidden_handed_fake():
    @hides_environment
    def cargo(command, **kwargs):
        raise FileNotFoundError("cargo missing")

    under_test.launch(hidden_forwarding(cargo))


def test_mock_raising():
    run = Mock(side_effect=raising(OSError("no git")))
    run(["git"], env=dict(os.environ))


def test_mock_raising_chained():
    under_test.launch(Mock(side_effect=raising(FileNotFoundError)))


def test_hidden_nested_class_recorder():
    class Fake:
        def __init__(self):
            self.calls = []

        @hides_environment
        def __call__(self, command, **kwargs):
            self.calls.append(kwargs)

    fake = Fake()
    fake(["git"], env=dict(os.environ))
    assert len(fake.calls) == 2


def test_record_names():
    record = dataclasses.asdict(Invocation("check", Names(os.environ)))
    assert record is None


def test_local_past_the_printed_arguments():
    def run(*options, ask=None):
        command = ["pytest", *options]
        report = "report"
        environment = dict(os.environ)
        assert ask is None, ask

    run(ask="x")


def test_wrapped_invocations():
    recorder = NamesRecorder()
    recorder([Invocation("check", dict(os.environ))])
    assert recorder.invocations == []


def test_names_alone():
    wrapped = Names(os.environ)
    assert wrapped is None


def test_raise_names():
    raise ValueError(sorted(os.environ))


def test_values_by_constant_names():
    environment = dict(os.environ)
    values = [environment[name] for name in ("AAA_OTHER", "HOME")]
    assert values == []


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


def test_hidden_aliased_launch():
    REAL_RUN(
        [sys.executable, "-c", "raise SystemExit(1)"],
        env=Names(os.environ),
        check=True,
    )


def test_handed_names():
    under_test.probe(["git"], Names(os.environ))


def test_handed_names_to_an_import():
    probe(["git"], environment=Names(dict(os.environ)))


def test_handed_to_a_method_of_a_built_instance():
    Probe().probe(["git"], dict(os.environ))


def test_handed_to_a_method_of_a_bound_instance():
    checked = Probe()
    checked.probe(["git"], environment=dict(os.environ))


def test_handed_to_a_method_through_self():
    Probe().check(["git"])


def test_handed_to_a_method_through_super():
    QuietProbe().check(["git"])


def test_built_names_then_raised():
    error = under_test.Refusal(sorted(os.environ))
    raise error


def test_thrown_names():
    (_ for _ in ()).throw(ValueError(sorted(os.environ)))


def test_mock_assert_names():
    run = Mock()
    run(["git"])
    run.assert_called_once_with(["git"], env=Names(os.environ))


def test_hidden_below_another():
    @logged
    @hides_environment
    def check(command, env):
        raise RuntimeError(command)

    check(["git"], dict(os.environ))


def test_hidden_module_alias_launch():
    sp.run(
        [sys.executable, "-c", "raise SystemExit(1)"],
        env=Names(os.environ),
        check=True,
    )


def test_missing_variable_by_name():
    assert len(environment_variable("AAA_MISSING")) == 1


def test_unset_missing_variable(monkeypatch):
    unset_variable(monkeypatch, "AAA_MISSING")


def test_popped_with_a_default():
    os.environ.pop("AAA_MISSING", None)
    assert "AAA_MISSING" == "x"


def test_delenv_not_raising(monkeypatch):
    monkeypatch.delenv("AAA_MISSING", raising=False)
    assert "AAA_MISSING" == "x"


def test_installed_fake_compared(monkeypatch):
    @hides_environment
    def fake_run(command, **kwargs):
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert subprocess.run is not fake_run


def test_hidden_named_tuple():
    @hides_environment
    def check(launch):
        raise RuntimeError(launch.command)

    check(Launch("git", dict(os.environ)))


def test_hidden_refuses_a_mock_call():
    @hides_environment
    def check(call):
        raise RuntimeError("unreachable")

    run = Mock()
    run(["git"], env=dict(os.environ))
    check(run.call_args)


def test_hidden_refuses_a_rebuilding_record():
    @hides_environment
    def check(record):
        raise RuntimeError(record.command)

    check(Rebuilt("git", dict(os.environ)))


def test_flagged_object_printing_its_attributes():
    @hides_environment
    def check(shown):
        raise RuntimeError(shown.command)

    shown = Shown("git")
    shown.env = dict(os.environ)
    check(shown)


def test_flagged_slotted_object_printing_its_attributes():
    @hides_environment
    def check(shown):
        raise RuntimeError(shown.command)

    shown = Slotted("git")
    shown.env = dict(os.environ)
    check(shown)


def test_flagged_method_of_a_fixture_instance(prober):
    prober.probe(["git"], dict(os.environ))


def test_flagged_diffed_lists():
    assert sorted(os.environ) == []


def test_flagged_diffed_strings():
    assert str(len(os.environ)) == "0"


def test_flagged_keyword_only():
    def fake_run(command, *, env):
        assert command == "cargo"

    fake_run("git", env=dict(os.environ))


def test_flagged_shared_name():
    @hides_environment
    def check(env):
        raise RuntimeError("git")

    env = dict(os.environ)
    check(env)


def test_flagged_present_variable():
    path = os.environ["PATH"]
    assert path == ""


def test_flagged_builtin_method():
    names = dict.fromkeys(os.environ)
    assert "AAA_MISSING" in names


def test_flagged_replaced_launcher(monkeypatch):
    @hides_environment
    def fake_run(command, **kwargs):
        raise RuntimeError("cargo")

    monkeypatch.setattr(subprocess, "run", fake_run)
    subprocess.run(["cargo"], env=dict(os.environ))
"""

# The code under test the examples hand fakes to. It wraps what they raise, as
# ``ci_parity`` and ``repo_routing`` do, so pytest prints the chain.
UNDER_TEST = """
import os


class Refusal(Exception):
    pass


def probe(command, environment):
    raise ValueError(command)


def launch(run):
    try:
        run(["git"], env=dict(os.environ))
    except OSError as error:
        raise ValueError("launch failed") from error
"""


def _failure_output(output: str) -> dict[str, str]:
    """Each failed test's report, short-summary line and warnings, by name."""
    reports: dict[str, str] = {}
    name = None
    warned = False
    for line in output.splitlines():
        if header := re.fullmatch(r"_+ (test_\w+) _+", line):
            name = header.group(1)
        elif line.startswith("="):
            name = None
            warned = "warnings summary" in line
        elif summary := re.match(r"FAILED \S+::(test_\w+)", line):
            reports[summary.group(1)] = reports.get(summary.group(1), "") + line
        elif warned and (source := re.match(r"\S+::(test_\w+)", line)):
            name = source.group(1)
        elif name is not None:
            reports[name] = reports.get(name, "") + line + "\n"
    return reports


def test_the_check_agrees_with_pytest(tmp_path: Path) -> None:
    examples = tmp_path / "test_examples.py"
    examples.write_text(EXAMPLES)
    (tmp_path / "under_test.py").write_text(UNDER_TEST)
    tree = ast.parse(EXAMPLES)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    tests = {name: node for name, node in functions.items() if name.startswith("test_")}
    # A test's report holds its own frames and those of the fixtures it names.
    fixtures = {name: node for name, node in functions.items() if _is_fixture(node)}
    lines = {
        name: {
            line
            for node in [
                test,
                *(fixtures[a] for a in _parameters(test) if a in fixtures),
            ]
            for line in range(node.lineno, node.end_lineno + 1)
        }
        for name, test in tests.items()
    }
    flagged = {
        name
        for line in environment_leaks(EXAMPLES)
        for name in tests
        if line in lines[name]
    }

    # The most verbose assertion output, in an environment holding nothing else.
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-vvv", "-p", "no:cacheprovider"]
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
        timeout=600,
    )
    reports = _failure_output(completed.stdout)
    printed = {name for name, text in reports.items() if SENTINEL in text}
    overflagged = {name for name in tests if "_flagged_" in name}

    assert sorted(reports) == sorted(tests)
    assert sorted(printed) == sorted(name for name in tests if "_leaks_" in name)
    assert sorted(flagged) == sorted(printed | overflagged)


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


values = st.dictionaries(
    st.sampled_from(["A", "B", "C", "D"]), st.sampled_from(["", "0", "secret"])
)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(kept=values, other=values)
def test_names_compares_by_value_and_prints_names(
    kept: dict[str, str], other: dict[str, str]
) -> None:
    names = Names(kept)
    assert (names == other) == (kept == other) == (other == names)
    assert (names == Names(other)) == (kept == other)
    assert dict(names) == kept
    assert repr(names) == f"Names({sorted(kept)!r})"


passed = st.recursive(
    st.one_of(st.none(), st.integers(), st.text(max_size=3), values),
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(
            st.sampled_from(["env", "environment", "other"]), children, max_size=3
        ),
    ),
    max_leaves=8,
)


def _hidden(original: object, received: object, name: object = None) -> bool:
    """Whether ``received`` is ``original`` with each environment in Names."""
    if isinstance(original, Mapping) and name in ("env", "environment"):
        return isinstance(received, Names) and dict(received) == dict(original)
    if isinstance(original, Mapping):
        return isinstance(received, Mapping) and all(
            _hidden(value, received[key], key) for key, value in original.items()
        )
    if type(original) in (list, tuple):
        return type(received) is type(original) and all(
            _hidden(old, new) for old, new in zip(original, received, strict=True)
        )
    return received is original


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    args=st.lists(passed, max_size=3),
    kwargs=st.dictionaries(
        st.sampled_from(["env", "environment", "other"]), passed, max_size=3
    ),
)
def test_hides_environment_passes_equal_values_with_names(
    args: list[object], kwargs: dict[str, object]
) -> None:
    received: dict[str, object] = {}

    @hides_environment
    def fake(*args: object, **kwargs: object) -> str:
        received.update(args=list(args), kwargs=kwargs)
        return "result"

    assert fake(*args, **kwargs) == "result"
    assert received["args"] == args
    assert received["kwargs"] == kwargs
    assert _hidden(args, received["args"])
    assert _hidden(kwargs, received["kwargs"])


@dataclasses.dataclass
class _Record:
    command: str
    environment: object = None
    env: object = dataclasses.field(default=None, init=False)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(held=passed, unset=passed, default=passed, defaulted=st.booleans())
def test_hides_environment_masks_defaults_and_record_fields(
    held: object, unset: object, default: object, defaulted: bool
) -> None:
    received: dict[str, object] = {}
    built = _Record("git", held)
    built.env = unset

    @hides_environment
    def fake(record: _Record, env: object = default) -> None:
        received.update(record=record, env=env)

    if defaulted:
        fake(built)
    else:
        fake(built, env=default)
    assert received["env"] == default
    assert _hidden(default, received["env"], "env")
    copy = received["record"]
    assert isinstance(copy, _Record) and copy.command == "git"
    # Read by name: the check takes a read of ``.environment`` or ``.env`` to
    # be an environment, and these hold what Hypothesis draws.
    for field, drawn in (("environment", held), ("env", unset)):
        kept = getattr(copy, field)
        assert kept == drawn
        assert _hidden(drawn, kept, field)


class _Slotted:
    __slots__ = ("command", "env")

    def __init__(self, command: str) -> None:
        self.command = command

    def __repr__(self) -> str:
        return f"_Slotted({self.command!r}, {getattr(self, 'env', None)!r})"


class _SlottedChild(_Slotted):
    __slots__ = ()


@pytest.mark.parametrize("kind", [_Slotted, _SlottedChild])
def test_hides_environment_refuses_slotted_attributes_it_would_print(
    kind: type[_Slotted],
) -> None:
    seen: list[_Slotted] = []

    @hides_environment
    def fake(record: _Slotted) -> None:
        seen.append(record)

    # Without an environment, the record is passed as it is.
    kept = kind("git")
    fake(kept)
    assert seen[0] is kept
    kept.env = {"A": "1"}
    with pytest.raises(TypeError, match="could not hide"):
        fake(kept)


def _launched(variables: Mapping[str, str], spawned: bool) -> dict[str, str]:
    """The environment a child starts with when launched with ``variables``,
    through ``posix_spawn`` where the platform has it, or fork-exec, which a
    ``cwd`` forces (``posix_spawn`` takes none)."""
    completed = subprocess.run(
        [sys.executable, "-c", "import json, os; print(json.dumps(dict(os.environ)))"],
        env=variables,
        close_fds=not spawned,
        cwd=None if spawned else os.getcwd(),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


@settings(deadline=None, max_examples=12, suppress_health_check=[HealthCheck.too_slow])
@given(
    drawn=st.dictionaries(
        st.sampled_from(["A", "B", "C"]), st.text(alphabet="az09 =", max_size=3)
    ),
    spawned=st.booleans(),
)
def test_a_launcher_starts_a_child_with_what_names_holds(
    drawn: dict[str, str], spawned: bool
) -> None:
    held = {"PATH": os.defpath, **drawn}
    seen = _launched(Names(held), spawned)
    assert seen == _launched(held, spawned)
    assert held.items() <= seen.items()
