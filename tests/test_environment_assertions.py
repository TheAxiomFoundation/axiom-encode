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
  method's ``self`` counts unless its class, and each of the module's classes
  built on it, prints as its name and address, and the method does not bind
  it again.
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
  imports, a class whose ``__new__`` or ``__init__`` may be out of view (as
  may any class with a metaclass, a decorator but ``dataclass``, or a base
  out of view but a built-in class or ``typing.NamedTuple``, which may run
  its own or replace the class's as it builds it), or a method of anything
  but a container that is not in view (a mock's ``assert_called_with``,
  ``functools.partial`` and ``os.execvpe`` among them). Those two bases are
  known by what binds their names: a built-in's name the module does not bind,
  and ``NamedTuple`` imported from ``typing`` (or read off ``typing``), and
  neither where the module may replace that name as an attribute, by the name
  it is read by or the one it is imported as
  (``monkeypatch.setattr(typing, "NamedTuple", ...)``); ``object`` is left
  out of an order only where the module neither binds its name nor may
  replace it so. A method is in view only where Python's lookup, along the
  C3 order of each class the receiver may be an instance of, finds a ``def``
  of the module's before any class out of view, under no decorator but
  ``hides_environment``, ``staticmethod`` and ``classmethod``, and none of
  those classes defines ``__getattribute__``.
  The receiver is known as a call to a class with no base out of view that
  the module binds once, by its ``class`` statement (or a name it assigns
  that class alone), whose ``__new__`` is ``object``'s; as a name every
  binding of which is such a call; and as the first parameter of the method
  it is read in, or ``super()`` there, where it may be an instance of the
  method's class or of any of the module's classes built on it, none of which
  may have a base out of view but those two. That parameter is not taken to
  be what the lookup bound in a static method, under a decorator that may
  call the method otherwise, in a method the module reads off a class (by its
  name, a name any binding of which may give one, ``type(...)``, a
  ``__class__``, a class method's first parameter, or a choice of these), or
  where the method rebinds it by any form (an assignment, an import, a
  ``def``, an ``except ... as``, a pattern's capture). A method name the
  module may replace other than by ``def`` (as an attribute, through
  ``setattr``, ``delattr`` or ``patch``, as a key of a namespace, or in a
  class's body) is never in view, and no method is where such a name is not
  a constant. Should code out of view raise, pytest prints the arguments of
  the frame it raised in, so a test hands it a ``Names`` (or a minimal
  environment of its own);
- a call passing what may hold one to a parameter of a function in view that
  it runs (a method, or a class's ``__new__`` or ``__init__``, and where that
  ``__new__`` is the module's, which may return an instance of any class
  built on it, the ``__init__`` of each), where ``hides_environment`` does
  not mask it. Arguments are bound as Python binds them: ``__new__`` is
  passed the class and a static ``__init__`` no instance; a literal sequence
  starred in place passes its items in turn, and one starred otherwise may
  pass any number, even none, so it and every argument after it may reach
  any parameter from the first that no argument ahead of it fills;
- a read or ``del`` of a variable of ``os.environ`` (or a name bound to it) by
  subscript, a one-argument ``pop`` or a ``popitem`` of it, and
  ``monkeypatch.delenv`` unless it is passed ``raising=False``. A test reads a
  variable through ``environment_variable`` and unsets one through
  ``unset_variable``, which raise the same ``KeyError`` holding the name
  alone.

The check tracks, for every value, the keys and positions that lead from it
to an environment inside it. Environments start as:

- ``os.environ``: ``environ`` read off a name the module binds to ``os``, or
  off another module's ``os`` (``subprocess.os``), whatever this module
  binds that name to;
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
  dataclass, one this module defines at its top level, with those of the
  module's classes it is built on, or one imported from ``axiom_encode``),
  where a parameter is annotated with its class, or one of its own methods
  (``__post_init__`` among them) is passed it as ``self``. A class method is
  passed the class, or one of the module's built on it, which prints as its
  name where its metaclass is ``type``: where the class has no metaclass, no
  decorator but ``dataclass`` and no base out of view but those two; where
  any may have another, what the records' fields and the classes' methods
  and bodies store count.

Wherever the module builds a record, each field holds what the call gives
it: a keyword, the field it names (a ``**`` mapping's keys too); an argument
by position, any of them, since the check does not read which one it fills
from the order of the record's fields (its bases', a plain base's
annotations a dataclass leaves out, fields it takes by keyword alone or not
at all, and an ``__init__`` of its own all change it). A call builds a
record through its name, a name the module assigns it (``Alias = Record``,
or a choice of such), ``type(...)`` or a ``__class__`` (any of the module's
classes), or a class method's first parameter (its class, or any built on
it); a record built on an alias of one has its fields, and a field holds its
default too (``dataclasses.field``'s ``default``, or what its
``default_factory`` returns). The ``self`` of its methods, and of its module
bases', holds what the module's calls build it with, and where the module
has a class whose bases it can't read, what they build each class the module
may build on it with.

The check follows them through copies and filtered copies
(``dataclasses.asdict``, ``vars`` and ``getattr`` too), assignments, what a
``match`` pattern captures, containers a test or a module-level helper
stores them in or returns, or a function stores in a module global, and a
name a function declares ``global`` and binds. It
follows them into helpers' parameters, nested helpers' included, into what
fixtures return or yield (this module's and its conftests'), into a function
a fixture provides as the tests call it, and into what a function returns, or
a generator yields, where the module calls it by its name or by a name it
assigns it (a lambda too). It follows what a method returns into each call
of a method of its name, and what a class's ``__call__``, ``__getitem__``,
``__getattr__``, ``__enter__``, ``__iter__`` and ``__next__`` return into
each call, subscript, attribute read, ``with`` and iteration: a top-level
class's wherever the module makes them, and those of a class a function
defines within that function. It follows them into what a class's methods
store on its instances and its properties return, its module bases'
included, a class defined within a function too; into what the body of a
class at the module's top level binds, in a scope of its own, through its
compound statements and every form of binding; and into what the module
stores on the class itself (``Holder.saved = ...``). The class's instances
read both, and so does a read off the class. A plain class's instance prints
as its name and address, so only reading its attributes counts, whatever the
module stores on it. A plain class binds no ``__repr__`` anywhere in its
body, and has no metaclass, no decorator but ``typing.final``, and no base
but ``object``, a built-in exception class, or the module's plain classes.
The check knows each by what binds its name, and trusts none where the
module may replace it as an attribute, nor any class where the module may
replace a class's ``__repr__`` or an instance's ``__class__`` other than by
``def`` (``Holder.__repr__ = ...``). A class whose metaclass is ``type``
prints as its name likewise. Reads back out by key or position follow them
too. One value of an environment
(``os.environ["X"]``, a ``.values()`` item, a value read by one of its own
names) may be printed; a container or a joined text of them holds them all.
Its names alone hold no values, and neither do values read by names the test
chose. A ``Names`` prints names alone, and a ``dataclasses.replace`` that swaps
one in holds no environment, but what is read back out of it (a copy, its
items, a value, what it keeps) is followed as before.
``test_the_check_agrees_with_pytest`` runs pytest at ``-vvv`` on examples of
each shape to keep the model honest.

The model flags more than pytest prints in these cases. When ``==`` fails
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
may run no Python frame of its own (``dict.fromkeys``) or be one of this
module's: on an instance a fixture returns; on what a call to a class returns
where the module binds the class's name more than once or the class has a
base out of view; past a base out of view that does not define it; on a first
parameter, where the class it is read in, or one built on it, has a base out
of view but a built-in class or ``typing.NamedTuple`` (which may leave its
methods as they are), where the module reads the method off a name only some
binding of which gives a class, or in a module with a class whose bases it
can't read (one a call builds, say); or under a name the module may replace.
So does a call to a class with such a base, or whose ``__new__`` or
``__init__`` the module may replace or is out of view, but a built-in
class's (or a named tuple's), which run no Python frame that can raise; and
where its ``__new__`` is the module's, the ``__init__`` of each class built
on it counts, though that ``__new__`` may return none of them. An argument
after one starred other than as a literal counts as passed to each parameter
from the first that no argument ahead of it fills. An argument a call gives
a record by position counts for each of its fields, though it fills one
alone: reading another back out counts, and so does a field that does not
print (``repr=False``), or an ``__init__`` of the record's own that stores
something else. What a record's methods, its body or the module store on it
beyond its fields counts as what it prints, though its repr may print its
fields alone. A name holds what any binding of it gives, wherever the module
reads it, so rebinding it (``record = replace(record, env=...)``) leaves what
it held; and a ``**`` mapping replaces the fields it names in a
``dataclasses.replace`` only where it is a literal that writes out all its
keys. What a top-level class's method returns counts for each call of a
method of its name on anything, and what a class's ``__call__``,
``__getitem__``, ``__getattr__``, ``__enter__``, ``__iter__`` or ``__next__``
returns for each call (of anything but a function or class the module
defines), subscript, attribute read, ``with`` and iteration where it counts at
all. A generator counts as holding what it yields, though it prints as its
name and address; a class pattern's positional pattern captures any attribute
of the subject, and the subject itself; and where the module has a class whose
bases it can't read, a method's first parameter counts as an instance of each
class the module may have built on its class. ``environ`` read off an ``os``
attribute of anything but a module (``fake.os.environ``) counts as
``os.environ``. A class method
of a class whose metaclass may be other than ``type`` counts as passed a
class printing what the record's fields and its methods store, though that
metaclass may print the name alone (``abc.ABCMeta``). An object holding an
environment under another name counts where it is passed to a function
``hides_environment`` wraps, though that refuses it when it also answers to
``env`` or ``environment`` (through ``__getattr__``, say). And a function is
flagged whether or not anything in it can raise, unless all it does is return
constants, and so is a read of ``os.environ`` by subscript, whether or not the
variable is set, and of a name the module binds to ``os.environ`` anywhere,
or of ``environ`` read off a name it binds to ``os`` anywhere, whatever the
name is bound to where it is read.

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
record whose ``__post_init__`` replaces what it is given, and an object with
a repr of its own (not a class or a module) that holds an environment under
``env`` or ``environment``, however it keeps it. The check takes every
container to be one it reads through; an environment inside a generator or a
view reaches the function as it is. ``--tb=long``, ``--showlocals`` and
``--fulltrace`` print every frame's arguments or locals, the code under
test's included. An environment
the code under test builds for itself is printed when an exception passes a
frame of its own that holds it: the frame that raised, or, for an exception it
wraps that was raised with no frame in the test's file to start from (as
``raising``'s is), the frame that wrapped it. A fake defined in the test's
file and hidden would give pytest that frame to start from instead. A
launcher reached through ``getattr(subprocess, name)``, or bound to a name
other than by plain assignment, is not taken to be one, and ``os.environ``
is not taken to be what holds it other than a name it is assigned to (with
``=``, an annotation, a walrus, or unpacked from a literal pair): an
attribute, a parameter's default, a ``for`` target, a container's item or a
function's result. Nor is ``os`` taken to be bound otherwise than by
``import`` or the assignment of a name it is bound to (``sys.modules``,
``importlib``). A class built on one of this module's other than by a
``class`` statement of its own (``type(...)``, a class out of view) is not
taken to be what a first parameter may hold, and a method's function reached
off its class other than by attribute, ``getattr`` or a key of its namespace
(``vars(cls).get``, ``inspect.getmembers``), or off a class reached other
than through a name (an item of a container, a function's result), is not
taken to be called unbound. The check does not take a record a function
defines to hold what its calls give it, nor the ``self`` of its methods (its
``__post_init__`` among them), which pytest prints with its fields; nor an
attribute of a class a function defines, or one the module stores on a class
other than through its name or a method's first parameter (``setattr``, an
alias), to hold what the module binds it to. Nor does it take a record a
call builds through anything else (a parameter, ``functools.partial``, an
item of a container) to hold what the call gives it; nor a call to give what
a function returns where it reaches the function otherwise (a parameter,
``partial``, a container), or what a method of a class a function defines
returns outside that function; nor an operator, an ``await`` or any other
special method of the module's to give what it returns. Setting variables to
what the platform rejects raises from ``os.environ``'s frames too: an empty
name or one with ``=`` in it, a name or value holding a null byte, or a pair given
to ``update`` that is not two items, through ``os.environ`` itself, its
``update`` or ``setdefault``, or ``monkeypatch.setenv``. Setting is not
flagged, so a test sets variables to names and strings it has written itself.
What a launched program prints is not taken to hold an environment: the
signing supervisor's trusted runtime prints its own and its child's, so its
tests print a run without its output and read the report through ``Names``.
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import functools
import inspect
import json
import os
import re
import subprocess
import sys
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
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
# Calls that replace or delete an attribute one of their arguments names
# (``setattr``, ``monkeypatch.setattr``, ``object.__setattr__``, ``patch``,
# ``patch.object``, ``patch.multiple``), and the methods that write into a
# mapping (an object's ``__dict__``).
RENAMERS = frozenset(
    {"setattr", "__setattr__", "delattr", "__delattr__", "patch", "object", "multiple"}
)
NAMESPACE_WRITES = frozenset(
    {"update", "setdefault", "__setitem__", "__delitem__", "pop", "popitem", "clear"}
)
# The decorators under which a method's first parameter is still what its
# lookup bound (pytest's included), and those under which a method found in
# view runs as it is defined.
RECEIVING = frozenset(
    {
        HIDES,
        "classmethod",
        "property",
        "cached_property",
        "fixture",
        "parametrize",
        "usefixtures",
        "skip",
        "skipif",
        "xfail",
        "filterwarnings",
    }
)
PLAIN = frozenset({HIDES, "staticmethod", "classmethod"})
# Methods whose first parameter is a class.
CLASS_METHODS = frozenset({"__new__", "__init_subclass__", "__class_getitem__"})


def _unwrapped(paths: Iterable[Path_]) -> set[Path_]:
    """The paths of what is read out of a value with ``paths``, ``Names`` too."""
    return {path[1:] if path[:1] == (NAMES,) else path for path in paths}


def _unhidden(paths: Iterable[Path_]) -> set[Path_]:
    """The paths of what is read off a value with ``paths``, an object
    printing as its class and address too."""
    return {path[1:] if path[:1] == (HIDDEN,) else path for path in paths}


def _returns_of(name: str, method: bool) -> str:
    """The name a scope binds what a function (a method, with ``method``)
    named ``name`` returns or yields under: one no Python name can take."""
    return f"<returns {'.' if method else ''}{name}>"


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


def _paired(target: ast.AST, value: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Each name an assignment of ``value`` to ``target`` binds, with what it
    is bound to, where an unpacked literal pairs them (``a, b = x, y``)."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(element, ast.Starred) for element in target.elts)
    ):
        for element, item in zip(target.elts, value.elts):
            yield from _paired(element, item)
    else:
        yield target, value


def _binds(node: ast.AST) -> str | None:
    """The name ``node`` binds, if any: a name stored or deleted, a
    parameter, an import, a ``def`` or ``class``, an ``except ... as``, or a
    pattern's capture."""
    if isinstance(node, ast.Name):
        return None if isinstance(node.ctx, ast.Load) else node.id
    if isinstance(node, ast.arg):
        return node.arg
    if isinstance(node, ast.alias):
        return node.asname or node.name.partition(".")[0]
    if isinstance(node, (*FUNCTIONS, ast.ClassDef)):
        return getattr(node, "name", None)
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name
    if isinstance(node, ast.MatchMapping):
        return node.rest
    return None


def _rebinds(function: ast.AST, name: str) -> bool:
    """Whether ``function`` binds its parameter ``name`` again, by any form
    (``_binds``), in any scope within it."""
    return any(
        node is not function and not isinstance(node, ast.arg) and _binds(node) == name
        for node in ast.walk(function)
    )


def _placed(arguments: list[ast.expr]) -> Iterator[tuple[ast.expr, int, bool]]:
    """Each argument a call passes by position, with the first position it
    may take and whether it may take any after that. A literal sequence
    starred in place passes its items in turn; one starred otherwise may
    pass any number, even none, so it and every argument after it may take
    any position from the first that no argument ahead of it fills."""
    given: list[ast.expr] = []
    for argument in arguments:
        if isinstance(argument, ast.Starred) and isinstance(
            argument.value, (ast.Tuple, ast.List)
        ):
            given.extend(argument.value.elts)
        else:
            given.append(argument)
    fixed = 0
    unknown = False
    for argument in given:
        unknown |= isinstance(argument, ast.Starred)
        yield argument, fixed, unknown
        fixed += not isinstance(argument, ast.Starred)


def _merged(sequences: list[list[object]]) -> list[object] | None:
    """C3's merge of ``sequences``, or None where they have no consistent
    order."""
    sequences = [list(sequence) for sequence in sequences if sequence]
    merged: list[object] = []
    while sequences:
        for sequence in sequences:
            head = sequence[0]
            if not any(head in other[1:] for other in sequences):
                break
        else:
            return None
        merged.append(head)
        sequences = [
            sequence[1:] if sequence[0] == head else sequence for sequence in sequences
        ]
        sequences = [sequence for sequence in sequences if sequence]
    return merged


def _bound_in(body: list[ast.stmt]) -> set[str]:
    """The names statements ``body`` bind in their own scope other than by
    ``def``: what a class's body may replace its methods with."""
    names: set[str] = set()
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Its decorators and defaults are evaluated here.
            stack.extend(node.decorator_list)
            stack.extend(node.args.defaults)
            stack.extend(default for default in node.args.kw_defaults if default)
            continue
        if isinstance(node, (ast.Lambda, ast.ClassDef)):
            names.add(getattr(node, "name", ""))
            continue
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.partition(".")[0])
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            names.add(node.name or "")
        elif isinstance(node, ast.MatchMapping):
            names.add(node.rest or "")
        stack.extend(ast.iter_child_nodes(node))
    return names - {""}


def _namespace(node: ast.AST) -> bool:
    """Whether ``node`` is an object's namespace: ``vars(...)`` or a
    ``__dict__``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "vars"
    ) or (isinstance(node, ast.Attribute) and node.attr == "__dict__")


def _renamed(node: ast.Call) -> ast.AST | None:
    """The argument of a ``RENAMERS`` call that names the attribute it
    replaces or deletes, or None where it names none."""
    name = _name(node)
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    arguments = node.args
    if name == "patch":
        return arguments[0] if arguments else keywords.get("target")
    if name == "object":
        return arguments[1] if len(arguments) > 1 else keywords.get("attribute")
    # ``monkeypatch.setattr("module.Class.name", value)`` names it by a path.
    if (
        arguments
        and isinstance(arguments[0], ast.Constant)
        and isinstance(arguments[0].value, str)
    ):
        return arguments[0]
    return arguments[1] if len(arguments) > 1 else keywords.get("name")


def _class_fields(
    tree: ast.Module,
    plain: Callable[[ast.expr, bool], bool] | None = None,
    known: Mapping[str, ast.ClassDef] | None = None,
) -> dict[str, frozenset[str]]:
    """The annotated fields of each class ``tree`` defines at its top level
    whose instances print them: those of the classes of the module's it is
    built on, and its own. Where a call builds one, an argument given by
    position may be given to any of them, so their order is not kept.

    A base is read as the class ``known`` names (``Alias = Record`` too), or
    as the top-level class of its name; ``plain`` is ``_shows_attributes``'s.
    """
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    bases = {**classes, **(known or {})}
    fields: dict[int, frozenset[str]] = {}

    def of(node: ast.ClassDef, seen: frozenset[int]) -> frozenset[str]:
        if id(node) not in fields:
            inherited = {
                field
                for base in node.bases
                if (parent := bases.get(_dotted(base))) is not None
                and id(parent) not in seen
                for field in of(parent, seen | {id(node)})
            }
            own = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            fields[id(node)] = frozenset(inherited | own)
        return fields[id(node)]

    return {
        name: of(node, frozenset())
        for name, node in classes.items()
        if _shows_attributes(node, bases, plain)
    }


@functools.cache
def _src_fields(module: str) -> dict[str, frozenset[str]]:
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
        # The function it binds (None at the module's top level), and the
        # names in it that hold an object printing as its name and address
        # whatever it stores on it as attributes (``_Check._hidden``).
        self.function: ast.AST | None = None
        self.hidden: set[str] = set()
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


def _plain_by_name(node: ast.expr, decorator: bool) -> bool:
    """Whether a class decorator, or a base out of view, leaves the class's
    instances printing as their class and address, read by its name alone:
    ``final``, and ``object`` or an exception class."""
    if decorator:
        return _name(node) == "final"
    return _name(node) == "object" or _name(node).endswith(("Error", "Exception"))


def _shows_attributes(
    owner: ast.ClassDef,
    classes: Mapping[str, ast.ClassDef] | None = None,
    plain: Callable[[ast.expr, bool], bool] | None = None,
    seen: frozenset[int] = frozenset(),
) -> bool:
    """Whether printing an instance of ``owner`` may print its attributes.

    A plain class (or an exception) prints as its name and address; a
    dataclass, a class with a metaclass, one whose body binds ``__repr__``
    anywhere (under an ``if`` too), or one built on another class may print
    what it holds, unless that class is one of ``classes`` that does not, or
    ``plain`` (``_plain_by_name`` by default) says that decorator or base
    leaves it printing so.
    """
    plain = plain or _plain_by_name
    if id(owner) in seen or owner.keywords:
        return True
    if any(_binds(node) == "__repr__" for node in _own_nodes(owner)):
        return True
    if not all(plain(decorator, True) for decorator in owner.decorator_list):
        return True
    classes = classes or {}
    return any(
        _shows_attributes(classes[_dotted(base)], classes, plain, seen | {id(owner)})
        if _dotted(base) in classes
        else not plain(base, False)
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
        # Imported names are never stored into: storing into ``os.environ``
        # does not make the ``os`` module something pytest prints.
        self.imports = {
            alias.asname or alias.name.partition(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        # Modules under another name: imported as it (``import subprocess as
        # sp``), or a name the module binds one to (``system = os``). A name
        # may be bound to different modules in different scopes, or in turn
        # (``os, sys = sys, os``), so each holds every module it may be, and
        # the sets only grow.
        self.modules: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.modules.setdefault(
                        alias.asname or alias.name.partition(".")[0], set()
                    ).add(
                        alias.name.rpartition(".")[2]
                        if alias.asname
                        else alias.name.partition(".")[0]
                    )
        # Each name an assignment binds, with what it is bound to.
        paired = [
            (target.id, value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
            and node.value is not None
            for whole in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target, value in _paired(whole, node.value)
            if isinstance(target, ast.Name)
        ]
        renames = [
            (target, value.id)
            for target, value in paired
            if isinstance(value, ast.Name)
        ]
        changed = True
        while changed:
            changed = False
            for target, value in renames:
                if value in self.modules and not self.modules[value] <= (
                    held := self.modules.setdefault(target, set())
                ):
                    held |= self.modules[value]
                    changed = True
        # Names bound to ``os.environ`` itself (``values = os.environ``, an
        # unpacked pair, a walrus), anywhere in the module.
        self.live: set[str] = set()
        while True:
            bound = {
                target.id
                for node in ast.walk(tree)
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                and node.value is not None
                for whole in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target, value in _paired(whole, node.value)
                if isinstance(target, ast.Name) and self._is_environ(value)
            }
            if bound <= self.live:
                break
            self.live |= bound
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
        # The function or lambda each node is in. A function's decorators,
        # defaults and annotations are evaluated where it is defined.
        self.enclosing: dict[ast.AST, ast.AST | None] = {}
        for node in ast.walk(tree):
            body = getattr(node, "body", []) if isinstance(node, FUNCTIONS) else []
            inner = {id(part) for part in (body if isinstance(body, list) else [body])}
            for child in ast.iter_child_nodes(node):
                self.enclosing[child] = (
                    node if id(child) in inner else self.enclosing.get(node)
                )
        # What a method call is looked up on. A class is known by its name
        # only where the module binds that name once, by its ``class``
        # statement, and none where it imports ``*``; a name holds an instance
        # of one only where every binding of it is a call to that class. The
        # first parameter of a method is read from the method.
        bindings: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if name := _binds(node):
                bindings.setdefault(name, []).append(node)
        self.bound = set(bindings)
        self.bindings = bindings
        star = any(
            alias.name == "*"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        )
        self.known_classes = {
            name: found[0]
            for name, found in bindings.items()
            if len(found) == 1 and isinstance(found[0], ast.ClassDef) and not star
        }
        # A name every binding of which assigns it one known class (``Alias =
        # Probe``) is known as that class.
        assigned = {
            id(target): node.value.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Name)
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        }
        while True:
            aliases = {}
            for name, found in bindings.items():
                values = {assigned.get(id(binding)) for binding in found}
                if name not in self.known_classes and len(values) == 1:
                    (value,) = values
                    if value in self.known_classes:
                        aliases[name] = self.known_classes[value]
            if not aliases:
                break
            self.known_classes.update(aliases)
        # Names a class statement binds, known or not.
        self.class_names = {node.name for node in every_class}
        self.every_class = every_class
        self.orders: dict[int, list[object] | None] = {}
        built = {
            id(target): node.value.func.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in self.known_classes
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        self.typed: dict[str, str] = {}
        for name, found in bindings.items():
            classes = {built.get(id(binding)) for binding in found}
            if len(classes) == 1 and None not in classes:
                self.typed[name] = classes.pop()
        # Method names the module may replace other than by ``def``: an
        # attribute it stores or deletes (``fake.run = ...``), a name it gives
        # ``setattr`` or ``delattr`` (``monkeypatch``'s and ``object``'s too)
        # or ``patch`` (``.object``, ``.multiple``), a key it stores into
        # ``vars(...)`` or a ``__dict__``, and a name a class's body binds
        # other than by ``def``. A lookup of one may find what replaced it.
        # Where such a name is not a constant, it may be any.
        self.overridden = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load)
        }
        self.overrides_all = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.overridden |= _bound_in(node.body)
            elif isinstance(node, ast.Subscript) and _namespace(node.value):
                if not isinstance(node.ctx, ast.Load):
                    self._override(node.slice)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _namespace(node.func.value)
                and node.func.attr in NAMESPACE_WRITES
            ):
                self.overrides_all = True
            elif (
                isinstance(node, ast.Call)
                and _name(node) == "setitem"
                and node.args
                and _namespace(node.args[0])
            ):
                self._override(node.args[1] if len(node.args) > 1 else None)
            elif (
                isinstance(node, ast.Call)
                and _name(node) in RENAMERS
                # ``object()`` builds a sentinel.
                and not (isinstance(node.func, ast.Name) and node.func.id == "object")
            ):
                if any(keyword.arg is None for keyword in node.keywords):
                    self.overrides_all = True
                if _name(node) == "multiple":
                    self.overridden |= {k.arg for k in node.keywords if k.arg}
                else:
                    self._override(_renamed(node))
        # The methods the module reads off a class, whose functions may then
        # be called with anything as their first argument: each ``def`` such a
        # read finds, by its ``id``, and by name where the class isn't known.
        self.unbound: set[str] = set()
        self.unbound_defs: set[int] = set()
        self.unbound_all = False
        # A name any binding of which may give a class of the module's
        # (``Alias = Probe``, ``kind = type(probe)``) is taken for one.
        self.class_valued: set[str] = set()
        while found := {
            name
            for name, value in paired
            if name not in self.class_valued and self._of_class(value)
        }:
            self.class_valued |= found
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and self._of_class(node.value):
                self._unbind(node.value, node.attr)
            elif isinstance(node, ast.Subscript) and _namespace(node.value):
                namespace = node.value
                owner = (
                    namespace.args[0]
                    if isinstance(namespace, ast.Call) and namespace.args
                    else getattr(namespace, "value", None)
                )
                if owner is not None and self._of_class(owner):
                    self._unbind(owner, node.slice)
            elif (
                isinstance(node, ast.Call)
                and _name(node) == "getattr"
                and len(node.args) > 1
                and self._of_class(node.args[0])
            ):
                self._unbind(node.args[0], node.args[1])
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
        # Records, whose instances print their fields (such as
        # ``CliInvocation``), by the dotted name this module uses: their
        # fields.
        self.carriers = self._carriers(tree)
        # The records and classes of the module's (by the names ``carriers``
        # and ``classes`` know them) a name may be when called: those any
        # binding of it may give it (``Alias = Record``, ``kind =
        # type(record)``, see ``_kinds``).
        self.kinds: dict[str, set[str]] = {}
        while True:
            grew = False
            for name, value in paired:
                found = self._kinds(value) - self.kinds.get(name, set())
                if found:
                    self.kinds.setdefault(name, set()).update(found)
                    grew = True
            if not grew:
                break
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
        # What the module's calls build each of its records with.
        self.built: dict[str, set[Path_]] = {}
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
        # The module's own globals, which its functions may store into or
        # bind (``global SHARED``).
        module_names = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        } | {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        self.passing = False
        self.globals = _Scope()
        self.globals.hidden = self._hidden(None)
        self.scopes: dict[ast.AST, _Scope] = {}
        changed = True
        while changed:
            self.passing = False
            changed = self._fill(self.globals, _own_nodes(tree))
            self._build(self.globals, _own_nodes(tree))
            for function in self.functions:
                scope = self._bind(function)
                self.scopes[function] = scope
                changed |= self._export(function, scope)
                # What a method of a class at its top level returns, the
                # module's calls of its name give.
                if function in self.owners:
                    key = _returns_of(function.name, True)
                    changed |= self.globals.bind(key, scope.get(key))
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
                # What its body binds, and what the module stores on the class
                # itself (``Holder.saved = ...``): its instances read them, and
                # so does a read off the class, which prints them where its
                # metaclass may.
                attributes = self._class_body(owner) | {
                    path
                    for scope in [self.globals, *self.scopes.values()]
                    for path in _unhidden(scope.get(name))
                    if path[:1] and str(path[0]).startswith(".")
                }
                changed |= self.globals.bind(
                    name,
                    _under(HIDDEN, attributes) if self._typed(owner) else attributes,
                )
                # And it holds what its module bases' methods store.
                stored |= attributes | {
                    path
                    for base in owner.bases
                    for kind in self._kinds(base)
                    for path in self.instances.get(kind, set())
                }
                stored = _unhidden(stored)
                if not self._prints_attributes(owner):
                    stored = _under(HIDDEN, stored)
                changed |= stored != self.instances.get(name, set())
                self.instances[name] = stored
            changed |= self.passing

    # -- where environments are --------------------------------------------

    def _carriers(self, tree: ast.Module) -> dict[str, frozenset[str]]:
        carriers: dict[str, frozenset[str]] = {}

        def add(prefix: str, classes: Mapping[str, frozenset[str]]) -> None:
            for name, fields in classes.items():
                if fields:
                    carriers[prefix + name] = fields

        add("", _class_fields(tree, self._plain, self.known_classes))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.partition(".")[0] != "axiom_encode":
                    continue
                classes = _src_fields(node.module)
                for alias in node.names:
                    local = alias.asname or alias.name
                    add("", {local: classes.get(alias.name, frozenset())})
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
        fields = {
            field for kind in self._kinds(node) for field in self.carriers.get(kind, ())
        }
        if fields:
            return {(f".{field}",) for field in fields & ENVIRONMENT_KEYS}
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
            return self._attribute(self.paths(node.value, scope), node.attr, scope)
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
            given = scope.get(_returns_of("__getitem__", True))
            return _descend(inner, key) | fragment | given
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

    def _attribute(
        self, found: set[Path_], attribute: str, scope: _Scope
    ) -> set[Path_]:
        """The paths of ``attribute`` read off a value with ``found``."""
        if attribute in {"environ", "environb", *ENVIRONMENT_KEYS}:
            return {()}
        if scope.launches and attribute == "kwargs":
            return set(KWARGS)
        if scope.launches and attribute in {"call_args", "await_args"}:
            return _under(1, KWARGS)
        if scope.launches and attribute in {
            "call_args_list",
            "await_args_list",
            "mock_calls",
        }:
            return _under(ANY, _under(1, KWARGS))
        if attribute == "__dict__":
            return _attributes(found)
        owner = _unhidden(found)
        if attribute == "_environment":
            # What a ``Names`` keeps.
            owner |= {
                ("._environment", *path[1:]) for path in owner if path[:1] == (NAMES,)
            }
        # A class's ``__getattr__`` may give any attribute.
        given = scope.get(_returns_of("__getattr__", True)) | scope.get(
            _returns_of("__getattribute__", True)
        )
        return _descend(owner, f".{attribute}") | given

    def _call_paths(self, node: ast.Call, scope: _Scope) -> set[Path_]:
        """What a call gives: what ``_called`` reads off it, and what the
        functions of the module's it may run return (``_returns_key``): a
        function or a lambda it calls by a name bound to it, a method of the
        name it calls, and, unless it calls a function or class of the
        module's by name, a class's ``__call__``."""
        found = self._called(node, scope)
        function = node.func
        if isinstance(function, ast.Attribute):
            found |= scope.get(_returns_of(function.attr, True))
        elif isinstance(function, ast.Name):
            found |= scope.get(_returns_of(function.id, False))
        if not (
            isinstance(function, ast.Name)
            and (
                function.id in self.helpers
                or function.id in scope.nested
                or function.id in scope.classes
                or self._kinds(function)
            )
        ):
            found |= scope.get(_returns_of("__call__", True))
        return found

    def _called(self, node: ast.Call, scope: _Scope) -> set[Path_]:
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
        # What a call to a class of the module's (or a record) builds holds
        # what its methods store, and a record's fields what the call gives.
        kinds = self._kinds(function)
        held = set()
        for kind in kinds:
            fields = self.carriers.get(kind, frozenset())
            if fields:
                held |= self._record(kind, fields, node, scope)
            held |= self.instances.get(kind, set())
        if any(self.carriers.get(kind) for kind in kinds):
            return held
        if (
            name == "replace"
            and node.args
            and _dotted(function)
            in {
                "replace",
                "dataclasses.replace",
            }
        ):
            # ``dataclasses.replace`` replaces the fields its keywords name. A
            # ``**`` mapping's items fill the fields their keys name, and
            # replace them where it is a literal that writes out all its keys.
            replaced = set()
            given = set()
            for keyword in node.keywords:
                inner = self.paths(keyword.value, scope)
                if keyword.arg is not None:
                    replaced.add(f".{keyword.arg}")
                    given |= _under(f".{keyword.arg}", inner)
                    continue
                value = keyword.value
                if isinstance(value, ast.Dict) and all(
                    isinstance(key, ast.Constant) and isinstance(key.value, str)
                    for key in value.keys
                ):
                    replaced |= {f".{key.value}" for key in value.keys}
                markers = {FRAGMENT, KEY, HIDDEN, NAMES}
                for path in _unwrapped(inner):
                    if path[:1] == (ANY,):
                        given.add(path)
                    elif (
                        path[:1] and isinstance(path[0], str) and path[0] not in markers
                    ):
                        given.add((f".{path[0]}", *path[1:]))
            found = {path for path in first if not path or path[0] not in replaced}
            return found | given
        if name == "field" and (
            _dotted(function) == "dataclasses.field"
            or (
                isinstance(function, ast.Name)
                and self._from(function.id, "dataclasses", "field")
            )
        ):
            # A record field's default, or what its factory returns.
            found = set()
            for keyword in node.keywords:
                if keyword.arg == "default":
                    found |= self.paths(keyword.value, scope)
                elif keyword.arg == "default_factory":
                    factory = keyword.value
                    found |= (
                        self.paths(factory.body, scope)
                        if isinstance(factory, ast.Lambda)
                        else self.paths(
                            ast.Call(func=factory, args=[], keywords=[]), scope
                        )
                    )
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
            return _descend(_unhidden(first), ANY) | default
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
            return _under(ANY, self._items(first, scope))
        if name == "next" and node.args:
            return self._items(first, scope)
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
        # What the methods of a class defined within the function store on
        # its instances.
        if isinstance(function, ast.Name) and name in scope.classes:
            held |= self._instance(scope.classes[name], scope)
        if EXCEPTION.search(name) or name in self.exceptions:
            # An exception prints what it is built with.
            given = [
                *arguments,
                *(self.paths(keyword.value, scope) for keyword in node.keywords),
            ]
            return {path for argument in given for path in _under(ANY, argument)} | held
        if isinstance(function, ast.Name):
            return self.returns.get(name, set()) | held
        return held

    def _record(
        self, record: str, fields: frozenset[str], node: ast.Call, scope: _Scope
    ) -> set[Path_]:
        """What the call ``node`` builds the record ``record``, a carrier
        with ``fields``, with."""
        # A carrier holds what a call gives its fields. The order of its
        # fields does not tell which an argument given by position fills:
        # its bases' fields, a plain base's annotations (which a dataclass
        # leaves out), fields it takes by keyword alone or not at all, and an
        # ``__init__`` of its own all change it. So it may be any of them, and
        # replacing or masking one leaves the others.
        given = fields | ENVIRONMENT_KEYS
        found = set()
        for argument, _, _ in _placed(node.args):
            held = self.paths(argument, scope)
            if isinstance(argument, ast.Starred):
                held = _descend(held, ANY)
            for field in fields:
                found |= _under(f".{field}", held)
        for keyword in node.keywords:
            held = self.paths(keyword.value, scope)
            if keyword.arg is None:
                # ``**{"env": ...}``: each key fills the field it names.
                for field in given:
                    found |= _under(f".{field}", _descend(held, field))
            elif keyword.arg in given:
                found |= _under(f".{keyword.arg}", held)
        # Its methods' ``self`` holds what any call builds it with.
        if record in self.classes and found - self.built.get(record, set()):
            self.built.setdefault(record, set()).update(found)
            self.passing = True
        return found

    def _instance(self, owner: ast.ClassDef, scope: _Scope) -> set[Path_]:
        """What an instance of a class defined within a function holds."""
        stored = _unhidden(
            path
            for method in owner.body
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _parameters(method)
            and "staticmethod"
            not in {_name(decorator) for decorator in method.decorator_list}
            for path in scope.get(_parameters(method)[0])
        )
        if self._prints_attributes(owner):
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
        # A method reads what the methods of its class, or of a class the
        # module may build on it (``_possible_receivers``), store on the
        # instance, and a record's, what the module's calls build it with
        # and what its annotated environment fields may hold; an instance
        # of a plain class keeps it hidden. A class method gets the class,
        # or one built on it, which prints as its name where its metaclass is
        # ``type`` (``_named``): what it reads off it is kept, hidden. A
        # metaclass of another kind may print what the class holds.
        owner = self.owners.get(function)
        parameters = _parameters(function)
        decorators = {_name(decorator) for decorator in function.decorator_list}
        if owner is not None and parameters and "staticmethod" not in decorators:
            receivers = [
                each.name
                for each in self._possible_receivers(self.classes[owner])
                if self.classes.get(each.name) is each
            ]
            held = {
                path
                for each in receivers
                for path in self.instances.get(each, set())
                | self.built.get(each, set())
            }
            of_class = "classmethod" in decorators or function.name in CLASS_METHODS
            if of_class:
                held = _unhidden(held)
            if of_class and self._named(self.classes[owner]):
                held = _under(HIDDEN, held)
            else:
                for each in receivers:
                    held |= self._annotated(ast.Name(id=each))
            seeds[parameters[0]] |= held
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
        scope.function = function
        scope.hidden = self._hidden(function)
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
        self._build(scope, ast.walk(function))
        return scope

    def _build(self, scope: _Scope, nodes: Iterable[ast.AST]) -> None:
        """Note what each call among ``nodes`` to a record of the module's
        builds it with (``built``), which its methods' ``self`` holds."""
        for node in nodes:
            if (
                isinstance(node, ast.Call)
                and self._kinds(node.func) & self.classes.keys()
            ):
                self.paths(node, scope)

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
        returned = scope.get(_returns_of(function.name, False))
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
        if keys and str(keys[0]).startswith(".") and name in scope.hidden:
            # An attribute stored on an object printing as its name and
            # address.
            found = _under(HIDDEN, found)
        return scope.bind(name, found)

    def _bind_node(self, node: ast.AST, scope: _Scope) -> bool:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found = self.paths(node.value, scope)
            return any([self._store(target, found, scope) for target in targets])
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            return self._store(
                node.target, self._items(self.paths(node.iter, scope), scope), scope
            )
        if isinstance(node, ast.withitem) and node.optional_vars is not None:
            entered = (
                self.paths(node.context_expr, scope)
                | scope.get(_returns_of("__enter__", True))
                | scope.get(_returns_of("__aenter__", True))
            )
            return self._store(node.optional_vars, entered, scope)
        if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom)):
            # What a function returns, or a generator yields, its calls give.
            key = self._returns_key(self.enclosing.get(node))
            if key is None or node.value is None:
                return False
            value = self.paths(node.value, scope)
            if isinstance(node, ast.Yield):
                value = _under(ANY, value)
            elif isinstance(node, ast.YieldFrom):
                value = _under(ANY, self._items(value, scope))
            return scope.bind(key, value)
        if isinstance(node, ast.Lambda):
            # A lambda returns its body to the calls of what it is bound to.
            parent = self.parents.get(id(node))
            keys = []
            if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = (
                    parent.targets
                    if isinstance(parent, ast.Assign)
                    else [parent.target]
                )
                keys = [
                    _returns_of(target.id, False)
                    if isinstance(target, ast.Name)
                    else _returns_of(target.attr, True)
                    for target in targets
                    if isinstance(target, (ast.Name, ast.Attribute))
                ]
            elif isinstance(parent, ast.keyword) and parent.arg:
                keys = [_returns_of(parent.arg, True)]
            value = self.paths(node.body, scope)
            return any([scope.bind(key, value) for key in keys])
        if isinstance(node, ast.Match):
            subject = self.paths(node.subject, scope)
            return any(
                [self._capture(case.pattern, subject, scope) for case in node.cases]
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
            for argument, first, unknown in _placed(node.args):
                taken = positions[first:] if unknown else positions[first : first + 1]
                paths = {
                    path for name in taken if name in filled for path in filled[name]
                }
                if isinstance(argument, ast.Starred):
                    argument, paths = argument.value, _under(ANY, paths)
                changed |= self._store(argument, paths, scope)
            for keyword in node.keywords:
                if keyword.arg in filled:
                    changed |= self._store(keyword.value, filled[keyword.arg], scope)
        return changed

    def _returns_key(self, frame: ast.AST | None) -> str | None:
        """The name ``frame``'s returns are bound to (``_returns_of``)."""
        if not isinstance(frame, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        return _returns_of(frame.name, frame in self.method_classes)

    def _items(self, paths: set[Path_], scope: _Scope) -> set[Path_]:
        """``_elements``, and what a class's ``__iter__`` yields or its
        ``__next__`` returns."""
        return (
            _elements(paths)
            | _descend(scope.get(_returns_of("__iter__", True)), ANY)
            | _descend(scope.get(_returns_of("__aiter__", True)), ANY)
            | scope.get(_returns_of("__next__", True))
            | scope.get(_returns_of("__anext__", True))
        )

    def _capture(self, pattern: ast.pattern, paths: set[Path_], scope: _Scope) -> bool:
        """Bind the names ``pattern`` captures from a subject with ``paths``.

        A sequence's items are read by position up to a starred one, and as
        any item after it; a mapping's by key, its ``**rest`` holding what the
        subject does; a class pattern's keywords as attributes, and its
        positional patterns as any attribute or, for a built-in class, the
        subject itself.
        """
        changed = False
        if isinstance(pattern, ast.MatchAs):
            if pattern.name:
                changed |= scope.bind(pattern.name, paths)
            if pattern.pattern is not None:
                changed |= self._capture(pattern.pattern, paths, scope)
        elif isinstance(pattern, ast.MatchOr):
            for each in pattern.patterns:
                changed |= self._capture(each, paths, scope)
        elif isinstance(pattern, ast.MatchSequence):
            starred = False
            for position, each in enumerate(pattern.patterns):
                if isinstance(each, ast.MatchStar):
                    starred = True
                    if each.name:
                        changed |= scope.bind(
                            each.name, _under(ANY, _descend(paths, ANY))
                        )
                else:
                    item = _descend(paths, ANY if starred else position)
                    changed |= self._capture(each, item, scope)
        elif isinstance(pattern, ast.MatchMapping):
            for key, each in zip(pattern.keys, pattern.patterns):
                changed |= self._capture(each, _descend(paths, self._key(key)), scope)
            if pattern.rest:
                changed |= scope.bind(pattern.rest, _unwrapped(paths))
        elif isinstance(pattern, ast.MatchClass):
            for each in pattern.patterns:
                item = paths | _descend(_unhidden(paths), ANY)
                changed |= self._capture(each, item, scope)
            for attribute, each in zip(pattern.kwd_attrs, pattern.kwd_patterns):
                item = self._attribute(paths, attribute, scope)
                changed |= self._capture(each, item, scope)
        return changed

    def _arguments(
        self, function: ast.AST, node: ast.Call, scope: _Scope, bound: int = 0
    ) -> dict[str, set[Path_]]:
        """What the call ``node`` passes each of ``function``'s parameters,
        less the first ``bound``, which a method's lookup binds."""
        arguments = function.args
        positional = [*arguments.posonlyargs, *arguments.args]
        skipped = positional[:bound]
        positional = positional[bound:]
        named = [
            parameter
            for parameter in [*arguments.args, *arguments.kwonlyargs]
            if not any(parameter is each for each in skipped)
        ]
        found: dict[str, set[Path_]] = {}

        def add(name: str, paths: set[Path_]) -> None:
            if paths:
                found.setdefault(name, set()).update(paths)

        for argument, first, unknown in _placed(node.args):
            paths = self.paths(argument, scope)
            if isinstance(argument, ast.Starred):
                paths = _descend(paths, ANY)
            if unknown:
                for parameter in positional[first:]:
                    add(parameter.arg, paths)
                if arguments.vararg is not None:
                    add(arguments.vararg.arg, _under(ANY, paths))
            elif first < len(positional):
                add(positional[first].arg, paths)
            elif arguments.vararg is not None:
                add(arguments.vararg.arg, _under(first - len(positional), paths))
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
        if self._prints_as_name(function):
            names.discard(_parameters(function)[0])
        return any(_holds(scope.get(name)) for name in names)

    def _prints_as_name(self, function: ast.AST) -> bool:
        """Whether ``function``'s first parameter prints as its name (and
        address): a plain class's instance, and a class whose metaclass is
        ``type``, do, where the method may get no other (a class built
        on its own, ``_receivers``) and does not rebind it."""
        owner = self.method_classes.get(function)
        parameters = _parameters(function) if isinstance(function, FUNCTIONS) else []
        kinds = {
            _name(decorator) for decorator in getattr(function, "decorator_list", [])
        }
        if (
            owner is None
            or not parameters
            or "staticmethod" in kinds
            or _rebinds(function, parameters[0])
        ):
            return False
        if "classmethod" in kinds or function.name in CLASS_METHODS:
            return self._named(owner)
        receivers = self._receivers(owner)
        return receivers is not None and not any(
            self._prints_attributes(each) for each in receivers
        )

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

    def _hands_out(self, node: ast.Call) -> bool:
        """Whether ``node`` hands what it is passed to code out of view.

        That is a function the module imports, a class whose ``__new__`` or
        ``__init__`` may be out of view (``_constructed``), or a method of
        anything but a container that is not found in view (``_in_view``),
        unless the check follows the call as data. Should that code raise,
        pytest prints the arguments of the frame it raised in, a mock's
        ``assert_called_with`` among them.
        """
        name = _name(node)
        if name in FOLLOWED or name in INSTALLERS or name in MOCKS:
            return False
        if isinstance(node.func, ast.Name):
            return name in self.imports or self._constructed(node) is None
        return (
            isinstance(node.func, ast.Attribute)
            and name not in CONTAINER_METHODS
            and self._in_view(node.func) is None
        )

    def _passes_in_view(self, node: ast.Call, scope: _Scope) -> bool:
        """Whether ``node`` passes what may hold an environment to a parameter
        of a function of this module's that it runs and that prints it: a
        method found in view, or a class's ``__new__`` or ``__init__``."""
        if isinstance(node.func, ast.Name):
            runs = self._constructed(node)
        elif isinstance(node.func, ast.Attribute) and _name(node) not in (
            CONTAINER_METHODS
        ):
            runs = self._in_view(node.func)
        else:
            runs = None
        return any(
            _holds(self._passed(definition, name, paths))
            for definition, bound in runs or []
            for name, paths in self._arguments(definition, node, scope, bound).items()
        )

    def _in_view(self, method: ast.Attribute) -> list[tuple[ast.AST, int]] | None:
        """The ``def`` of this module's each class the receiver may be an
        instance of runs for ``method``, with how many of its parameters the
        lookup binds; None unless every one is found in view.

        Python looks a method up along the method resolution order of the
        receiver's class (C3), and this module's classes are read in it. The
        receiver is known as a call to a class with no base out of view that
        the module binds once, by its ``class`` statement, whose ``__new__``
        is ``object``'s (none of its classes defines another); as a name
        every binding of which is such a call; and as the first parameter of
        the method it is read in (``_receiver``), or ``super()`` there, where
        it may be an instance of the method's class or of any class of the
        module's built on it (``_subclasses``). Each must find a ``def`` of
        the module's, under
        no decorator but ``hides_environment``, ``staticmethod`` and
        ``classmethod``, before any class out of view, and none of the
        classes may define ``__getattribute__``. A method name the module may
        replace (``overridden``) is never found.
        """
        if (
            self.overrides_all
            or method.attr in self.overridden
            or {"__getattribute__", "__class__"} & self.overridden
        ):
            return None
        receiver = method.value
        frame = self.enclosing.get(method)
        lookups: list[tuple[list[object], bool]] = []
        if (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
            and "super" not in self.bound
        ):
            found = self._receiver(frame)
            if receiver.args or receiver.keywords or found is None:
                return None
            owner, of_class = found
            orders = self._subclasses(owner)
            if orders is None:
                return None
            lookups = [(order[order.index(owner) + 1 :], of_class) for order in orders]
        elif isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Name):
            owner = self.known_classes.get(receiver.func.id)
            if owner is None or not self._builds_itself(owner):
                return None
            lookups = [(self._order(owner), False)]
        elif isinstance(receiver, ast.Name) and receiver.id in self.typed:
            owner = self.known_classes[self.typed[receiver.id]]
            if not self._builds_itself(owner):
                return None
            lookups = [(self._order(owner), False)]
        elif isinstance(receiver, ast.Name):
            found = self._receiver(frame, receiver.id)
            if found is None:
                return None
            owner, of_class = found
            orders = self._subclasses(owner)
            if orders is None:
                return None
            lookups = [(order, of_class) for order in orders]
        else:
            return None
        runs = []
        for order, of_class in lookups:
            if any(
                isinstance(owner, ast.ClassDef)
                and any(
                    getattr(node, "name", "") == "__getattribute__"
                    for node in owner.body
                )
                for owner in order
            ):
                return None
            definition = self._looked_up(order, method.attr)
            if (
                not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
                or {_name(decorator) for decorator in definition.decorator_list} - PLAIN
            ):
                return None
            runs.append((definition, self._bound(definition, of_class)))
        return runs

    def _receiver(
        self, frame: ast.AST | None, name: str | None = None
    ) -> tuple[ast.ClassDef, bool] | None:
        """The class ``frame`` is a method of, and whether its first parameter
        holds a class rather than an instance, where that parameter (named
        ``name``, if given) holds what Python's lookup bound it to.

        The parameter is not taken to be bound so in a static method, under a
        decorator that may call the method otherwise (any but
        ``hides_environment``, ``classmethod``, a property or one of
        pytest's), in a method the module reads off a class (``unbound``), or
        where the method rebinds it by any form (``_binds``: an assignment, an
        import, a ``def``, an ``except ... as``, a pattern's capture).
        """
        owner = self.method_classes.get(frame)
        if owner is None:
            return None
        positional = [*frame.args.posonlyargs, *frame.args.args]
        if not positional or (name is not None and positional[0].arg != name):
            return None
        first = positional[0].arg
        decorators = {_name(decorator) for decorator in frame.decorator_list}
        if decorators - RECEIVING:
            return None
        # Read off a class, a class method is still bound to it.
        if "classmethod" not in decorators and (
            self.unbound_all
            or frame.name in self.unbound
            or id(frame) in self.unbound_defs
        ):
            return None
        if _rebinds(frame, first):
            return None
        return owner, "classmethod" in decorators or frame.name in CLASS_METHODS

    def _of_class(self, node: ast.AST) -> bool:
        """Whether ``node`` may be a class of this module's, whose functions
        are read off it unbound: a name a ``class`` statement binds or one any
        binding of which may give a class (``class_valued``: an alias of one
        too), ``type(...)``, a ``__class__``, a class method's first
        parameter, a walrus binding one, or a choice of one (``a or b``,
        ``a if c else b``)."""
        if isinstance(node, ast.NamedExpr):
            return self._of_class(node.value)
        if isinstance(node, ast.IfExp):
            return self._of_class(node.body) or self._of_class(node.orelse)
        if isinstance(node, ast.BoolOp):
            return any(self._of_class(value) for value in node.values)
        if isinstance(node, ast.Name):
            if node.id in self.class_names or node.id in self.class_valued:
                return True
            frame = self.enclosing.get(node)
            return (
                frame in self.method_classes
                and _parameters(frame)[:1] == [node.id]
                and (
                    frame.name in CLASS_METHODS
                    or "classmethod"
                    in {_name(decorator) for decorator in frame.decorator_list}
                )
            )
        if isinstance(node, ast.Call):
            return isinstance(node.func, ast.Name) and node.func.id == "type"
        return isinstance(node, ast.Attribute) and node.attr == "__class__"

    def _override(self, name: ast.AST | None) -> None:
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            self.overridden.add(name.value.rpartition(".")[2])
        else:
            self.overrides_all = True

    def _unbind(self, owner: ast.AST, name: ast.AST | str) -> None:
        """Note that the module reads the method ``name`` off ``owner``, a
        class: what that finds may be called with anything first."""
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            name = name.value
        if not isinstance(name, str):
            self.unbound_all = True
            return
        known = (
            self.known_classes.get(owner.id) if isinstance(owner, ast.Name) else None
        )
        order = None if known is None else self._order(known)
        if order is None:
            self.unbound.add(name)
            return
        found = self._looked_up(order, name)
        if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.unbound_defs.add(id(found))

    def _bound(self, definition: ast.AST, of_class: bool) -> int:
        """How many of ``definition``'s parameters a lookup on an instance
        (or, with ``of_class``, on a class) binds."""
        decorators = {_name(decorator) for decorator in definition.decorator_list}
        if "staticmethod" in decorators or definition.name == "__new__":
            return 0
        if "classmethod" in decorators:
            return 1
        return 0 if of_class else 1

    def _order(self, owner: ast.ClassDef) -> list[object] | None:
        """``owner``'s method resolution order, by C3.

        This module's classes appear as their ``class`` statements, and each
        class out of view as its source text, standing for it and all it
        inherits (``*bases`` too); ``object`` is left out, unless the module
        binds its name or may replace it as an attribute (``builtins.object =
        ...``, ``overridden``). None where it can't be known: under a class
        decorator other than ``dataclass``, with a metaclass or other keyword,
        or with bases in no consistent order.
        """
        key = id(owner)
        if key not in self.orders:
            # A class built on itself has none.
            self.orders[key] = None
            if owner.keywords or any(
                _name(d) != "dataclass" for d in owner.decorator_list
            ):
                return None
            parents: list[list[object]] = []
            for base in owner.bases:
                if (
                    _dotted(base) == "object"
                    and "object" not in self.bound
                    and not self.overrides_all
                    and "object" not in self.overridden
                ):
                    continue
                known = self.known_classes.get(_dotted(base))
                parent = [ast.unparse(base)] if known is None else self._order(known)
                if parent is None:
                    return None
                parents.append(parent)
            self.orders[key] = _merged(
                [[owner], *parents, [parent[0] for parent in parents]]
            )
        return self.orders[key]

    def _built_on(
        self, other: ast.ClassDef, owner: ast.ClassDef, seen: frozenset[int]
    ) -> bool | None:
        """Whether ``other`` is ``owner`` or built on it, by its bases' names;
        None where a base may be anything: a name the module binds other than
        by importing it or by one ``class`` statement, or a call."""
        if other is owner:
            return True
        if id(other) in seen:
            return False
        result: bool | None = False
        for base in other.bases:
            while isinstance(base, ast.Subscript):
                base = base.value
            known = self.known_classes.get(_dotted(base))
            if known is not None:
                found = self._built_on(known, owner, seen | {id(other)})
            elif isinstance(base, ast.Name):
                found = (
                    None
                    if base.id in self.bound and base.id not in self.imports
                    else False
                )
            else:
                found = False if _dotted(base) else None
            if found:
                return True
            if found is None:
                result = None
        return result

    def _receivers(self, owner: ast.ClassDef) -> list[ast.ClassDef] | None:
        """``owner`` and each class of this module's built on it: what a
        method of ``owner`` may get first (or an instance of). None where
        a base that may be anything may build a class on it
        (``_built_on``)."""
        found = []
        for other in self.every_class:
            built = self._built_on(other, owner, frozenset())
            if built is None:
                return None
            if built:
                found.append(other)
        return found

    def _subclasses(self, owner: ast.ClassDef) -> list[list[object]] | None:
        """The method resolution orders of ``owner`` and of each class of
        this module's built on it (``_receivers``), or None where one may be
        built on it whose order is not known, or with a base out of view that
        may replace its methods as it builds it (``__init_subclass__``, a
        metaclass): any but a built-in class or a named tuple
        (``_inert_base``)."""
        receivers = self._receivers(owner)
        if receivers is None:
            return None
        orders = []
        for other in receivers:
            order = self._order(other)
            if order is None or not all(map(self._inert_base, order)):
                return None
            orders.append(order)
        return orders

    def _inert_base(self, base: object) -> bool:
        """Whether ``base``, in a method resolution order, is this module's
        class, or a built-in class or ``typing.NamedTuple`` out of view: one
        that leaves what a class built on it defines as it is, and whose
        ``__new__`` and ``__init__`` run no Python frame that can raise with
        what a call passes (a named tuple's ``__new__`` is a lambda passing
        them to ``tuple.__new__``). The check knows each by what binds its
        name, not by its spelling: a built-in's name the module does not bind, and
        ``NamedTuple`` imported from ``typing`` (``_from``). Either may be
        anything where the module may replace that name as an attribute
        (``monkeypatch.setattr(typing, "NamedTuple", ...)``, ``overridden``),
        by the name the module reads it by or the one it imports it as."""
        if not isinstance(base, str):
            return True
        text = re.split(r"[\[(]", base)[0]
        root, _, rest = text.partition(".")
        if rest:
            trusted = rest == "NamedTuple" and self._from(root, "", "typing")
        elif self._from(root, "typing", "NamedTuple"):
            trusted, rest = True, "NamedTuple"
        else:
            trusted = root not in self.bound and isinstance(
                getattr(builtins, root, None), type
            )
        return (
            trusted
            and not self.overrides_all
            and not {text.rpartition(".")[2], rest} & self.overridden
        )

    def _named(self, owner: ast.ClassDef) -> bool:
        """Whether a class method of ``owner`` is passed a class printing as
        its name: ``owner`` and each class built on it (``_receivers``) has
        ``type`` for its metaclass (``_typed``)."""
        receivers = self._receivers(owner)
        return receivers is not None and all(map(self._typed, receivers))

    def _typed(self, owner: ast.ClassDef) -> bool:
        """Whether ``owner`` has ``type`` for its metaclass, and so prints as
        its name, as it does where the check knows its order (no metaclass, no class
        decorator but ``dataclass``) and holds no class out of view but a
        built-in class or ``typing.NamedTuple`` (``_inert_base``)."""
        order = self._order(owner)
        return order is not None and all(map(self._inert_base, order))

    def _possible_receivers(self, owner: ast.ClassDef) -> list[ast.ClassDef]:
        """What a method of ``owner`` may get first (or an instance of):
        ``_receivers``, or, where a base that may be anything may build a
        class on ``owner``, each class whose bases the check can't rule out
        reaching ``owner`` (``_built_on``)."""
        receivers = self._receivers(owner)
        if receivers is not None:
            return receivers
        return [
            other
            for other in self.every_class
            if self._built_on(other, owner, frozenset()) is not False
        ]

    def _plain(self, node: ast.expr, decorator: bool) -> bool:
        """``_plain_by_name``, read by what binds the name: ``final``
        imported from ``typing`` (or read off it), and ``object`` or a
        built-in exception class the module does not bind; none where the
        module may replace that name as an attribute (``overridden``), by
        the name the module reads it by or the one it imports it as."""
        text = _dotted(node)
        root, _, rest = text.partition(".")
        if decorator:
            trusted = (
                rest == "final" and self._from(root, "", "typing")
                if rest
                else self._from(root, "typing", "final")
            )
            names = {root, "final"}
        else:
            value = None if rest else getattr(builtins, root, None)
            trusted = (
                isinstance(value, type)
                and (value is object or issubclass(value, BaseException))
                and root not in self.bound
            )
            names = {root}
        return trusted and not self.overrides_all and not names & self.overridden

    def _prints_attributes(self, owner: ast.ClassDef) -> bool:
        """``_shows_attributes``, its module bases known by
        ``known_classes`` and the rest by what binds their names
        (``_plain``); and so for every class where the module may replace a
        class's ``__repr__``, or an instance's ``__class__``, other than by
        ``def`` (``Holder.__repr__ = ...``, ``overridden``)."""
        return (
            self.overrides_all
            or bool({"__repr__", "__class__"} & self.overridden)
            or _shows_attributes(owner, self.known_classes, self._plain)
        )

    def _hidden(self, function: ast.AST | None) -> set[str]:
        """The names in ``function`` (at the module's top level, for None)
        holding an object that prints as its name and address whatever is
        stored on it as an attribute: a method's first parameter where it
        does (``_prints_as_name``); a top-level class the module binds once,
        in any scope, whose metaclass is ``type`` (``_typed``); and a name
        every binding of which calls a class whose ``__new__`` is
        ``object``'s (``typed``, ``_builds_itself``) and whose instances
        print so (``_prints_attributes``). A store in a function within
        ``function`` reaches an object of the same kind: no scope binds any
        of these names to anything else."""
        names = {
            name
            for name, owner in self.classes.items()
            if self.known_classes.get(name) is owner and self._typed(owner)
        }
        if function is not None and self._prints_as_name(function):
            names.add(_parameters(function)[0])
        return names | {
            name
            for name, kind in self.typed.items()
            if self._builds_itself(owner := self.known_classes[kind])
            and not self._prints_attributes(owner)
        }

    def _class_body(self, owner: ast.ClassDef) -> set[Path_]:
        """What the body of ``owner``, a class at the module's top level,
        binds, as its attributes. It runs in a scope of its own, which reads
        the module's globals for a name it does not bind, through its
        compound statements, unpacking and every other form of binding."""
        scope = _Scope(self.globals.bindings)
        nodes = list(_own_nodes(owner))
        self._fill(scope, nodes)
        self._build(scope, nodes)
        return {
            path
            for name in _bound_in(owner.body)
            for path in _under(f".{name}", scope.get(name))
        }

    def _kinds(self, node: ast.AST) -> set[str]:
        """The records and classes of the module's, by the names
        ``carriers`` and ``classes`` know them, that ``node`` may be when
        called: one it names, one a name any binding of which may give it
        (``kinds``: ``Alias = Record``), or a choice of these (``a if c else
        b``); each class a class method's first parameter may hold; and
        any of them for ``type(...)`` or a ``__class__``."""
        if isinstance(node, ast.NamedExpr):
            return self._kinds(node.value)
        if isinstance(node, ast.IfExp):
            return self._kinds(node.body) | self._kinds(node.orelse)
        if isinstance(node, ast.BoolOp):
            return {kind for value in node.values for kind in self._kinds(value)}
        dotted = _dotted(node)
        found = {dotted} if dotted in self.carriers or dotted in self.classes else set()
        if isinstance(node, ast.Name):
            found |= self.kinds.get(node.id, set())
            frame = self.enclosing.get(node)
            owner = self.method_classes.get(frame)
            if (
                owner is not None
                and _parameters(frame)[:1] == [node.id]
                and (
                    frame.name in CLASS_METHODS
                    or "classmethod"
                    in {_name(decorator) for decorator in frame.decorator_list}
                )
            ):
                found |= {
                    each.name
                    for each in self._possible_receivers(owner)
                    if self.classes.get(each.name) is each
                }
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
        ) or (isinstance(node, ast.Attribute) and node.attr == "__class__"):
            found |= set(self.classes)
        return found

    def _from(self, name: str, module: str, imported: str) -> bool:
        """Whether every binding of ``name`` imports ``imported`` from
        ``module`` (``from typing import NamedTuple``), or, where ``module``
        is empty, imports the module ``imported`` (``import typing``)."""
        found = self.bindings.get(name, [])
        return bool(found) and all(
            isinstance(binding, ast.alias)
            and binding.name == imported
            and (
                isinstance(statement := self.parents.get(id(binding)), ast.ImportFrom)
                and statement.module == module
                and not statement.level
                if module
                else isinstance(self.parents.get(id(binding)), ast.Import)
            )
            for binding in found
        )

    def _builds_itself(self, owner: ast.ClassDef) -> bool:
        """Whether a call to ``owner`` returns an instance of it: its
        ``__new__`` is ``object``'s, from a known order with no class out of
        view."""
        order = self._order(owner)
        return (
            order is not None
            and "__new__" not in self.overridden
            and all(
                isinstance(each, ast.ClassDef)
                and not any(
                    getattr(node, "name", "") == "__new__" for node in each.body
                )
                for each in order
            )
        )

    @staticmethod
    def _looked_up(order: list[object], method: str) -> ast.AST | str | None:
        """What looking ``method`` up along ``order`` finds: the last ``def``
        of it in a class of this module's, a class out of view (its source
        text), or nothing (``None``)."""
        for owner in order:
            if isinstance(owner, str):
                return owner
            found = [
                node
                for node in owner.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method
            ]
            if found:
                return found[-1]
        return None

    def _constructed(self, node: ast.Call) -> list[tuple[ast.AST, int]] | None:
        """The ``__new__`` and ``__init__`` of this module's that a call to a
        class may run, with how many of their parameters the call binds:
        ``__new__`` is passed the class, a static ``__init__`` nothing.

        ``[]`` for a call to anything but a class of this module's, or where
        they are a built-in class's, which run no Python frame; None where the
        class is not known (``known_classes``, ``_order``) or has a base out
        of view but those (``_inert_base``), which may run its own or replace
        the class's. A ``__new__`` of the module's may return an instance of
        any class built on the one called, whose ``__init__`` then runs, so
        each of theirs is taken (``_subclasses``).
        """
        name = node.func.id
        if name not in self.class_names and name not in self.known_classes:
            return []
        owner = self.known_classes.get(name)
        order = None if owner is None else self._order(owner)
        if (
            order is None
            or not all(map(self._inert_base, order))
            or self.overrides_all
            or {"__new__", "__init__"} & self.overridden
        ):
            return None
        new = self._looked_up(order, "__new__")
        orders: list[list[object]] | None = [order]
        if isinstance(new, (ast.FunctionDef, ast.AsyncFunctionDef)):
            orders = self._subclasses(owner)
            if orders is None:
                return None
        runs = []
        for definition in [
            new,
            *(self._looked_up(each, "__init__") for each in orders),
        ]:
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = {
                    _name(decorator) for decorator in definition.decorator_list
                }
                if decorators - {HIDES, "staticmethod"}:
                    return None
                static = "staticmethod" in decorators and definition.name != "__new__"
                runs.append((definition, 0 if static else 1))
        return runs

    def _launcher(self, dotted: str) -> bool:
        """Whether ``dotted`` names one of the standard library's launchers,
        its module imported under any name, or a name bound to one."""
        parts = dotted.split(".")
        return dotted in self.spawners or (
            len(parts) >= 2
            and any((module, parts[-1]) in SPAWNERS for module in self.origins(parts))
        )

    def origins(self, parts: list[str]) -> set[str]:
        """The modules the name before the last in ``parts`` (a dotted path,
        ``["os", "environ"]``) may be. Where it is the first, each imported
        as it, and each a name bound to it may be; itself, where the module
        binds it to none. Where it is another module's attribute
        (``subprocess.os``), what its name says, whatever this module binds
        that name to."""
        if len(parts) > 2:
            return {parts[-2]}
        return self.modules.get(parts[-2], {parts[-2]})

    def _spawns(self, node: ast.Call) -> bool:
        """Whether ``node`` calls one of the standard library's launchers."""
        if isinstance(node.func, ast.Name):
            return node.func.id in self.spawners
        return self._launcher(_dotted(node.func))

    def _is_environ(self, node: ast.AST) -> bool:
        """Whether ``node`` is, or may be, ``os.environ`` (or ``os.environb``)
        itself: under any module path or name ``os`` is bound to, a name bound
        to it, a walrus binding it, or a conditional expression, ``or`` or
        ``and`` one side of which is."""
        if isinstance(node, ast.NamedExpr):
            return self._is_environ(node.value)
        if isinstance(node, ast.IfExp):
            return self._is_environ(node.body) or self._is_environ(node.orelse)
        if isinstance(node, ast.BoolOp):
            return any(self._is_environ(value) for value in node.values)
        if isinstance(node, ast.Name):
            return node.id in self.aliases or node.id in self.live
        if not isinstance(node, ast.Attribute):
            return False
        parts = _dotted(node).split(".")
        return (
            node.attr in {"environ", "environb"}
            and len(parts) >= 2
            and "os" in self.origins(parts)
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
                    elif self._hands_out(node):
                        shown = [*node.args, *(k.value for k in node.keywords)]
                    elif self._passes_in_view(node, scope):
                        # The frame of a function in view it runs prints its
                        # arguments, unless ``hides_environment`` masks them.
                        lines.append(node.lineno)
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
import contextlib
import dataclasses
import functools
import logging
import os
import pprint
import re
import subprocess
import os as system
import subprocess as sp
import sys
import typing
import warnings
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock, patch

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
rebound_os = os
REAL_RUN = subprocess.run
helpers = SimpleNamespace(child_environment=lambda: dict(os.environ))
runner = SimpleNamespace(run_embedded=lambda script, **kwargs: 0)


def logged(function):
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    return wrapper


def platform_version():
    # A local name for another module leaves the global ``system`` and ``sp``
    # what they are imported as.
    system = sys
    sp = sys
    return system.version, sp.version


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


def record_second(command, into):
    into.append(dict(os.environ))


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


@dataclasses.dataclass
class FirstBased:
    env: dict


@dataclasses.dataclass
class SecondBased:
    name: str = "git"


# Its fields are ``FirstBased``'s, then ``SecondBased``'s: a dataclass takes
# its bases' fields in reverse method resolution order.
@dataclasses.dataclass
class TwoBased(SecondBased, FirstBased):
    pass


@dataclasses.dataclass
class Paired:
    data: object
    env: object


# So are its, though it names ``SecondBased`` last.
@dataclasses.dataclass
class OverTwo(TwoBased, SecondBased):
    pass


class PlainCommand:
    command: str


# A dataclass takes no field from a plain class: ``env`` is its first.
@dataclasses.dataclass
class OnPlain(PlainCommand):
    env: dict


@dataclasses.dataclass
class KeywordFirst:
    name: str = dataclasses.field(default="git", kw_only=True)
    env: dict | None = None


@dataclasses.dataclass
class LateLog:
    log: list = dataclasses.field(init=False, default_factory=list)
    env: dict | None = None


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


class Forwarding:
    def __init__(self, command):
        self.command = command
        self.held = {}

    def __getattr__(self, name):
        if name == "env":
            return self.held
        raise AttributeError(name)

    def __repr__(self):
        return f"Forwarding({self.command!r}, {self.held!r})"


class Probe:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)


class Shadowed:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)


class Swapping:
    @hides_environment
    def swap(self, command, environment):
        raise ValueError(command)


class DottedSwap:
    @hides_environment
    def dotted_swap(self, command, environment):
        raise ValueError(command)


class PatchedSwap:
    @hides_environment
    def patched_swap(self, command, environment):
        raise ValueError(command)


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


def test_leaks_missing_through_an_aliased_module():
    assert system.environ["AAA_MISSING"] == "x"


def test_leaks_missing_through_a_walrus():
    assert (walrus_environ := os.environ)["AAA_MISSING"] == "x"


def test_leaks_missing_through_an_unpacked_name():
    unpacked_environ, unused = os.environ, None
    assert unpacked_environ["AAA_MISSING"] == "x"


def test_leaks_missing_through_a_conditional():
    chosen_environ = os.environ if sys.argv else {}
    assert chosen_environ["AAA_MISSING"] == "x"


def test_leaks_through_a_nested_self():
    class NestedSelf:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            def inner(self):
                self.probe(["git"], dict(os.environ))

            inner(under_test.Base())

    NestedSelf().check()


def test_leaks_through_a_method_parameter():
    class Relaying:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, other):
            other.probe(["git"], dict(os.environ))

    Relaying().check(under_test.Base())


def test_leaks_through_a_rebound_self():
    class Rebinding:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self = under_test.Base()
            self.probe(["git"], dict(os.environ))

    Rebinding().check()


def test_leaks_through_a_static_method():
    class StaticProbe:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        @staticmethod
        def check(cls):
            cls.probe(["git"], dict(os.environ))

    StaticProbe.check(under_test)


def test_leaks_through_an_inherited_method():
    class QuietSibling:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    class LoudSibling(under_test.Base):
        def check(self):
            self.probe(["git"], dict(os.environ))

    LoudSibling().check()


def test_leaks_through_super_out_of_view():
    class Below(under_test.Base):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            super().probe(["git"], dict(os.environ))

    Below().check()


def test_leaks_through_a_rebound_class_name():
    Shadowed = under_test.Base
    Shadowed().probe(["git"], dict(os.environ))


def test_leaks_through_a_class_named_twice():
    class Twice(under_test.Base):
        pass

    Twice().probe(["git"], dict(os.environ))


def test_leaks_through_a_base_out_of_view_first():
    class OutOfViewFirst(under_test.Base, Probe):
        pass

    OutOfViewFirst().probe(["git"], dict(os.environ))


def test_leaks_through_a_class_attribute():
    class Reassigned:
        @hides_environment
        def relay(self, command, environment):
            raise ValueError(command)

        relay = staticmethod(under_test.probe)

    Reassigned().relay(["git"], dict(os.environ))


def test_leaks_through_an_overridden_method(monkeypatch):
    swapped = Swapping()
    monkeypatch.setattr(swapped, "swap", under_test.probe)
    swapped.swap(["git"], dict(os.environ))


def test_leaks_missing_through_a_conditional_else():
    else_environ = {} if not sys.argv else os.environ
    assert else_environ["AAA_MISSING"] == "x"


def test_leaks_missing_through_or():
    or_environ = os.environ or {}
    assert or_environ["AAA_MISSING"] == "x"


def test_leaks_missing_through_a_module_bound_to_a_name():
    assert rebound_os.environ["AAA_MISSING"] == "x"


def test_leaks_handed_to_a_method_in_view():
    class Unhidden:
        def probe(self, command, variables):
            raise ValueError(command)

    Unhidden().probe(["git"], dict(os.environ))


def test_leaks_handed_to_a_hidden_method_under_another_name():
    class HiddenOtherName:
        @hides_environment
        def probe(self, command, variables):
            raise ValueError(command)

    HiddenOtherName().probe(["git"], dict(os.environ))


def test_leaks_handed_to_a_static_method_in_view():
    class StaticInView:
        @staticmethod
        def probe(command, variables):
            raise ValueError(command)

    StaticInView().probe(["git"], dict(os.environ))


def test_leaks_handed_to_a_method_through_self():
    class SelfUnhidden:
        def probe(self, command, variables):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))

    SelfUnhidden().check()


def test_leaks_handed_to_a_constructor():
    class Constructed:
        def __init__(self, command, variables):
            raise ValueError(command)

    Constructed(["git"], dict(os.environ))


def test_leaks_handed_to_a_static_constructor():
    class StaticBuilt:
        @staticmethod
        @hides_environment
        def __init__(environment, command):
            raise ValueError(command)

    StaticBuilt(["git"], dict(os.environ))


def test_leaks_handed_to_the_constructor_of_a_class_new_returns():
    class QuietNew:
        @hides_environment
        def __new__(cls, command, environment):
            return object.__new__(LoudNew)

        @hides_environment
        def __init__(self, command, environment):
            pass

    class LoudNew(QuietNew):
        def __init__(self, command, values):
            raise ValueError(command)

    QuietNew(["git"], dict(os.environ))


def test_leaks_handed_past_an_empty_star():
    class EmptyStar:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    EmptyStar().probe(*(), dict(os.environ), ["git"])


def test_leaks_handed_past_a_star_of_no_items():
    class NoItemsStar:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    leading = []
    NoItemsStar().probe(*leading, dict(os.environ), ["git"])


def test_leaks_handed_past_a_star_of_items():
    class ItemsStar:
        @hides_environment
        def probe(self, command, environment, values=None):
            raise ValueError(command)

    leading = [["first"]]
    ItemsStar().probe(*leading, ["git"], dict(os.environ))


def test_leaks_to_a_record_field_past_a_star():
    leading = [["git"]]
    record = Launch(*leading, dict(os.environ))
    assert record is None


def test_leaks_to_a_record_field_past_an_empty_star():
    record = Launch(*(), ["git"], dict(os.environ))
    assert record is None


def test_leaks_through_a_class_alias():
    class AliasRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def alias_read(self, command):
            self.probe(command, dict(os.environ))

    ReadAlias = AliasRead
    ReadAlias.alias_read(under_test.Base(), ["git"])


def test_leaks_through_a_class_chosen():
    class ChosenRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def chosen_read(self, command):
            self.probe(command, dict(os.environ))

    chosen = ChosenRead if sys.argv else type(under_test.Base())
    chosen.chosen_read(under_test.Base(), ["git"])


def test_leaks_through_a_class_picked():
    class PickedRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def picked_read(self, command):
            self.probe(command, dict(os.environ))

    picked = None or PickedRead
    picked.picked_read(under_test.Base(), ["git"])


def test_leaks_through_a_class_bound_in_place():
    class WalrusRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def walrus_read(self, command):
            self.probe(command, dict(os.environ))

    (walrused := WalrusRead).walrus_read(under_test.Base(), ["git"])


def test_leaks_through_a_named_tuple_of_another_module():
    class OtherNamed(under_test.NamedTuple):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))

    OtherNamed().check()


def test_leaks_filled_past_a_star():
    holder = []
    leading = [["git"]]
    record_second(*leading, holder)
    assert holder == []


def test_leaks_filled_through_a_star():
    pair = (["git"], [])
    record_second(*pair)
    assert pair == ()


def test_leaks_through_a_receiver_imported_over():
    class ImportedOver:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(importer):
            import under_test as importer

            importer.probe(["git"], dict(os.environ))

    ImportedOver().check()


def test_leaks_through_a_receiver_captured_over():
    class CapturedOver:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(captured):
            match under_test.Base():
                case captured:
                    captured.probe(["git"], dict(os.environ))

    CapturedOver().check()


def test_leaks_handed_to_a_constructor_out_of_view():
    class Inheriting(under_test.Built):
        pass

    Inheriting(["git"], dict(os.environ))


def test_leaks_through_self_in_a_subclass():
    class QuietMixin:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))

    class LoudMixed(under_test.Base, QuietMixin):
        pass

    LoudMixed().check()


def test_leaks_through_a_first_parameter_otherwise_named():
    class QuietThis:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(this):
            this.probe(["git"], dict(os.environ))

    class LoudThis(under_test.Base, QuietThis):
        pass

    LoudThis().check()


def test_leaks_through_super_in_a_diamond():
    class QuietTop:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    class Middle(QuietTop):
        def check(self):
            super().probe(["git"], dict(os.environ))

    class Other(under_test.Base, QuietTop):
        pass

    class Mixed(Middle, Other):
        pass

    Mixed().check()


def test_leaks_through_a_diamond_out_of_view():
    class QuietRoot:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    class Left(QuietRoot):
        pass

    class Right(under_test.Base, QuietRoot):
        pass

    class Both(Left, Right):
        def check(self):
            self.probe(["git"], dict(os.environ))

    Both().check()


def test_leaks_through_a_subclass_of_an_alias():
    class AliasedQuiet:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))

    QuietAlias = AliasedQuiet

    class ViaAlias(under_test.Base, QuietAlias):
        pass

    ViaAlias().check()


def test_leaks_through_a_method_read_off_its_class():
    class Unbound:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, command):
            self.probe(command, dict(os.environ))

    Unbound.check(under_test.Base(), ["git"])


def test_leaks_through_self_under_another_decorator():
    def called_with_base(function):
        def wrapper(self):
            return function(under_test.Base())

        return wrapper

    class Redirected:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        @called_with_base
        def check(self):
            self.probe(["git"], dict(os.environ))

    Redirected().check()


def test_leaks_through_a_default_evaluated_outside():
    self = under_test.Base()

    class Defaulted:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, value=self.probe(["git"], dict(os.environ))):
            pass


def test_leaks_through_a_class_building_another():
    class Building:
        def __new__(cls):
            return under_test.Base()

        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    Building().probe(["git"], dict(os.environ))


def test_leaks_through_an_attribute_lookup_of_its_own():
    class LookingUp:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def __getattribute__(self, name):
            return under_test.probe

    LookingUp().probe(["git"], dict(os.environ))


def test_leaks_through_a_method_under_another_decorator():
    def replaced(function):
        return under_test.Base.probe

    class Decorated:
        @replaced
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    Decorated().probe(["git"], dict(os.environ))


def test_leaks_through_a_name_also_a_parameter():
    class ParameterTyped:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    def relay(typed_holder):
        typed_holder.probe(["git"], dict(os.environ))

    typed_holder = ParameterTyped()
    relay(under_test.Base())


def test_leaks_through_a_stored_attribute():
    class Stored:
        @hides_environment
        def stored_probe(self, command, environment):
            raise ValueError(command)

    stored = Stored()
    stored.stored_probe = under_test.probe
    stored.stored_probe(["git"], dict(os.environ))


def test_leaks_through_a_method_set_by_its_path(monkeypatch):
    monkeypatch.setattr("test_examples.DottedSwap.dotted_swap", under_test.Base.probe)
    DottedSwap().dotted_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_patched_by_its_path():
    with patch("test_examples.PatchedSwap.patched_swap", under_test.Base.probe):
        PatchedSwap().patched_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_set_through_object():
    class ObjectSwap:
        @hides_environment
        def object_swap(self, command, environment):
            raise ValueError(command)

    object_swapped = ObjectSwap()
    object.__setattr__(object_swapped, "object_swap", under_test.probe)
    object_swapped.object_swap(["git"], dict(os.environ))


def test_leaks_through_methods_patched_together():
    class MultipleSwap:
        @hides_environment
        def multiple_swap(self, command, environment):
            raise ValueError(command)

    with patch.multiple(MultipleSwap, multiple_swap=under_test.Base.probe):
        MultipleSwap().multiple_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_patched_by_keyword():
    class KeywordSwap:
        @hides_environment
        def keyword_swap(self, command, environment):
            raise ValueError(command)

    with patch.object(KeywordSwap, attribute="keyword_swap", new=under_test.Base.probe):
        KeywordSwap().keyword_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_stored_in_a_namespace():
    class NamespaceSwap:
        @hides_environment
        def namespace_swap(self, command, environment):
            raise ValueError(command)

    namespace_swapped = NamespaceSwap()
    vars(namespace_swapped)["namespace_swap"] = under_test.probe
    namespace_swapped.namespace_swap(["git"], dict(os.environ))


def test_leaks_through_a_name_a_class_body_binds():
    class BodySwap:
        @hides_environment
        def body_swap(self, command, environment):
            raise ValueError(command)

        for body_swap in (staticmethod(under_test.probe),):
            pass

    BodySwap().body_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_set_by_keyword(monkeypatch):
    class NamedSwap:
        @hides_environment
        def named_swap(self, command, environment):
            raise ValueError(command)

    named_swapped = NamedSwap()
    monkeypatch.setattr(named_swapped, name="named_swap", value=under_test.probe)
    named_swapped.named_swap(["git"], dict(os.environ))


def test_leaks_through_a_method_set_into_a_namespace(monkeypatch):
    class SetItemSwap:
        @hides_environment
        def setitem_swap(self, command, environment):
            raise ValueError(command)

    setitem_swapped = SetItemSwap()
    monkeypatch.setitem(vars(setitem_swapped), "setitem_swap", under_test.probe)
    setitem_swapped.setitem_swap(["git"], dict(os.environ))


def test_leaks_through_a_name_bound_to_two_classes():
    class FirstBound:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    TwoBound = FirstBound
    TwoBound = under_test.Base
    TwoBound().probe(["git"], dict(os.environ))


def test_leaks_through_a_base_building_another():
    class FromFactory(under_test.Factory):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    FromFactory().probe(["git"], dict(os.environ))


def test_leaks_through_self_under_a_base_rewriting_it():
    class SelfRewritten(under_test.Rewriting):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))

    SelfRewritten().check()


def test_leaks_through_a_method_read_off_a_type():
    class TypeRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def type_read(self, command):
            self.probe(command, dict(os.environ))

    type(TypeRead()).type_read(under_test.Base(), ["git"])


def test_leaks_through_a_method_read_off_cls():
    class ClsRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def cls_read(self, command):
            self.probe(command, dict(os.environ))

        @classmethod
        def run_cls_read(cls):
            cls.cls_read(under_test.Base(), ["git"])

    ClsRead.run_cls_read()


def test_leaks_through_a_method_read_off_dunder_class():
    class DunderRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def dunder_read(self, command):
            self.probe(command, dict(os.environ))

    DunderRead().__class__.dunder_read(under_test.Base(), ["git"])


def test_leaks_through_a_method_read_from_a_class_namespace():
    class NamespaceRead:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def namespace_read(self, command):
            self.probe(command, dict(os.environ))

    vars(NamespaceRead)["namespace_read"](under_test.Base(), ["git"])


def test_leaks_handed_to_a_method_through_super():
    class LoudParent:
        def probe(self, command, variables):
            raise ValueError(command)

    class QuietChild(LoudParent):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            super().probe(["git"], dict(os.environ))

    QuietChild().check()


def test_leaks_handed_to_a_plain_method_through_cls():
    class ClassHanded:
        @hides_environment
        def class_handed(self, environment, command=None):
            raise ValueError(command)

        @classmethod
        def run_class_handed(cls):
            cls.class_handed(dict(os.environ), ["git"])

    ClassHanded.run_class_handed()


def test_leaks_handed_to_a_plain_method_through_cls_in_new():
    class NewHanded:
        def __new__(cls):
            cls.new_handed(dict(os.environ), ["git"])

        @hides_environment
        def new_handed(self, environment, command=None):
            raise ValueError(command)

    NewHanded()


def test_leaks_handed_to_a_constructor_under_another_decorator():
    def built_elsewhere(function):
        return under_test.Built.__init__

    class DecoratedInit:
        @built_elsewhere
        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    DecoratedInit(["git"], dict(os.environ))


def test_leaks_handed_to_a_constructor_a_base_rewrites():
    class InitRewritten(under_test.Rewriting):
        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    InitRewritten(["git"], dict(os.environ))


def test_leaks_handed_to_a_class_under_a_decorator():
    def replaced_class(owner):
        return under_test.Built

    @replaced_class
    class ReplacedClass:
        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    ReplacedClass(["git"], dict(os.environ))


def test_leaks_handed_to_a_class_with_a_metaclass():
    class Calling(type):
        def __call__(cls, command, variables):
            raise ValueError(command)

    class Metaclassed(metaclass=Calling):
        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    Metaclassed(["git"], dict(os.environ))


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


def test_leaks_local_alias_of_a_module_alias_launch():
    aliased_launch = sp.check_call
    aliased_launch(
        [sys.executable, "-c", "raise SystemExit(1)"], env=dict(os.environ)
    )


def test_leaks_a_record_built_on_two():
    record = TwoBased(dict(os.environ))
    assert record is None


def test_leaks_a_record_built_on_two_in_another_order():
    record = OverTwo(dict(os.environ), "git")
    assert record is None


def test_leaks_a_record_built_on_a_plain_class():
    record = OnPlain(dict(os.environ))
    assert record is None


def test_leaks_to_a_record_past_a_field_taken_by_keyword():
    record = KeywordFirst(dict(os.environ))
    assert record is None


def test_leaks_to_a_record_past_a_field_it_does_not_take():
    record = LateLog(dict(os.environ))
    assert record is None


def test_leaks_a_record_s_other_field_past_its_replaced_environment_field():
    record = Paired(dict(os.environ), {})
    replaced = dataclasses.replace(record, env={})
    assert replaced is None


def test_leaks_a_record_s_other_field_past_its_masked_environment_field():
    @hides_environment
    def check(record):
        raise RuntimeError("masked")

    check(Paired(dict(os.environ), {}))


def test_leaks_a_record_s_other_field_by_keyword():
    record = Paired(data=dict(os.environ), env={})
    assert record is None


def test_leaks_a_record_with_no_environment_field():
    record = SecondBased(dict(os.environ))
    assert record is None


def test_leaks_a_record_given_unpacked_keywords():
    record = Paired(**{"data": [], "env": dict(os.environ)})
    assert record is None


def test_leaks_a_record_given_keywords_by_any_name():
    given = {name: dict(os.environ) for name in ["data", "env"]}
    record = Paired(**given)
    assert record.data is None


def test_leaks_a_record_s_other_field_by_unpacked_keywords():
    record = Paired(**{"data": dict(os.environ), "env": {}})
    assert record is None


def test_hidden_record_given_unpacked_keywords():
    @hides_environment
    def check(record):
        raise RuntimeError(record.data)

    check(Paired(**{"data": "git", "env": dict(os.environ)}))


def test_leaks_a_rebound_first_parameter():
    class Plain:
        def check(self):
            self = dict(os.environ)
            raise ValueError("rebound")

    Plain().check()


def test_leaks_a_first_parameter_rebound_in_a_nested_function():
    class Plain:
        def check(self):
            def rebind():
                nonlocal self
                self = dict(os.environ)

            rebind()
            raise ValueError("rebound")

    Plain().check()


def test_leaks_a_first_parameter_rebound_by_a_match_capture():
    class Plain:
        def check(self):
            match dict(os.environ):
                case self:
                    raise ValueError("captured")

    Plain().check()


def test_leaks_an_item_a_sequence_pattern_captures():
    match [dict(os.environ)]:
        case [found]:
            assert found is None


def test_leaks_what_a_starred_pattern_captures():
    match ["git", dict(os.environ)]:
        case ["git", *rest]:
            assert rest is None


def test_leaks_a_value_a_mapping_pattern_captures():
    match {"held": dict(os.environ)}:
        case {"held": found}:
            assert found is None


def test_leaks_a_mapping_pattern_s_rest():
    match dict(os.environ):
        case {**rest}:
            assert rest is None


def test_value_a_mapping_pattern_reads_by_name():
    match dict(os.environ):
        case {"AAA_OTHER": found}:
            assert found is None


def test_leaks_a_class_pattern_s_keyword():
    match SimpleNamespace(held=dict(os.environ)):
        case SimpleNamespace(held=found):
            assert found is None


def test_leaks_a_built_in_class_pattern_s_subject():
    match dict(os.environ):
        case dict(found):
            assert found is None


def test_leaks_an_alternative_pattern_s_capture():
    match dict(os.environ):
        case [found] | found:
            assert found is None


def test_leaks_what_a_nested_function_returns():
    def made():
        return dict(os.environ)

    assert made() is None


def test_leaks_what_a_lambda_returns():
    made = lambda: dict(os.environ)
    assert made() is None


def test_leaks_what_a_generator_yields():
    def made():
        yield dict(os.environ)

    for each in made():
        assert each is None


def test_leaks_what_a_context_manager_yields():
    @contextlib.contextmanager
    def made():
        yield dict(os.environ)

    with made() as each:
        assert each is None


def test_leaks_what_a_method_returns():
    class Maker:
        def made(self):
            return dict(os.environ)

    assert Maker().made() is None


def test_leaks_what_an_instance_s_call_returns():
    class Calling:
        def __call__(self):
            return dict(os.environ)

    calling = Calling()
    assert calling() is None


def test_leaks_what_a_class_enters_with():
    class Entering:
        def __enter__(self):
            return dict(os.environ)

        def __exit__(self, *details):
            return False

    with Entering() as made:
        assert made is None


def test_leaks_an_item_a_class_gives():
    class Indexed:
        def __getitem__(self, key):
            return dict(os.environ)

    assert Indexed()["held"] is None


def test_leaks_what_a_class_iterates_over():
    class Iterated:
        def __iter__(self):
            yield dict(os.environ)

    for each in Iterated():
        assert each is None


def test_leaks_an_attribute_a_class_gives():
    class Forwarded:
        def __getattr__(self, name):
            return dict(os.environ)

    assert Forwarded().held is None


def test_record_masked_by_unpacked_keywords():
    record = Paired(data="git", env=dict(os.environ))
    masked = dataclasses.replace(record, **{"env": Names(record.env)})
    assert masked is None


def test_leaks_a_record_s_field_replaced_by_unpacked_keywords():
    record = Paired(data="git", env={})
    replaced = dataclasses.replace(record, **{"data": dict(os.environ)})
    assert replaced is None


def test_flagged_record_masked_by_a_mapping_s_keywords():
    record = Paired(data="git", env=dict(os.environ))
    changes = {"env": Names(record.env)}
    masked = dataclasses.replace(record, **changes)
    assert masked is None


def test_flagged_record_masked_under_its_own_name():
    record = Paired(data="git", env=dict(os.environ))
    record = dataclasses.replace(record, env=Names(record.env))
    assert record is None


def test_leaks_a_capture_inside_an_as_pattern():
    match [dict(os.environ)]:
        case [found] as whole:
            assert found is None


def test_item_a_sequence_pattern_reads_by_position():
    match [dict(os.environ), "git"]:
        case [_, found]:
            assert found is None


def test_leaks_an_item_after_a_starred_pattern():
    match ["git", "status", dict(os.environ)]:
        case [*_, last]:
            assert last is None


@dataclasses.dataclass
class Matched:
    held: dict


def test_leaks_a_record_s_attribute_a_class_pattern_captures():
    match Matched(dict(os.environ)):
        case Matched(held=found):
            assert found is None


def yielded_mapping():
    yield dict(os.environ)


def test_leaks_what_a_module_generator_yields():
    for each in yielded_mapping():
        assert each is None


def fill_shared():
    global SHARED_BY_FILL
    SHARED_BY_FILL = dict(os.environ)


def test_leaks_a_global_a_function_binds():
    fill_shared()
    assert SHARED_BY_FILL is None


def test_class_a_test_defines_built_not_called():
    class Calls:
        def __call__(self):
            return dict(os.environ)

    made = Calls()
    assert made is None


class StoredOn:
    pass


def test_class_printed_after_the_test_stores_on_it():
    StoredOn.saved = dict(os.environ)
    assert StoredOn is None


class NestedStore:
    pass


nested_store_kept = NestedStore()


def test_class_printed_after_a_nested_function_stores_on_it():
    def keep():
        NestedStore.saved = dict(os.environ)

    keep()
    assert NestedStore is None


def test_instance_printed_after_a_nested_function_stores_on_it():
    def keep():
        nested_store_kept.saved = dict(os.environ)

    keep()
    assert nested_store_kept is None


class ShadowsAName:
    shadowed_by_a_class = dict(os.environ)


shadowed_by_a_class = {}


def test_global_beside_a_class_attribute_of_its_name():
    assert shadowed_by_a_class is None


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
    class SelfProbe:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, command):
            self.probe(command, dict(os.environ))

    SelfProbe().check(["git"])


def test_handed_to_a_method_through_cls():
    class ClassProbe:
        @classmethod
        @hides_environment
        def probe(cls, command, environment):
            raise ValueError(command)

        @classmethod
        def check(cls, command):
            cls.probe(command, dict(os.environ))

    ClassProbe.check(["git"])


def test_handed_to_a_method_through_super():
    class BaseProbe:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    class QuietProbe(BaseProbe):
        def check(self, command):
            super().probe(command, dict(os.environ))

    QuietProbe().check(["git"])


def test_handed_to_a_method_through_self_beside_a_built_in_base():
    class BuiltInBased(dict):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, command):
            self.probe(command, dict(os.environ))

    BuiltInBased().check(["git"])


def test_handed_to_a_method_through_self_of_a_named_tuple():
    class Pair(NamedTuple):
        first: str

        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, command):
            self.probe(command, dict(os.environ))

    Pair("a").check(["git"])


def test_handed_to_a_method_beside_a_sentinel():
    class Sentinelled:
        @hides_environment
        def probe(self, command, environment, marker=None):
            raise ValueError(command)

    Sentinelled().probe(["git"], dict(os.environ), object())


def test_handed_to_a_static_method_in_view():
    class StaticHidden:
        @staticmethod
        @hides_environment
        def probe(environment, command):
            raise ValueError(command)

    StaticHidden().probe(dict(os.environ), ["git"])


def test_handed_to_a_built_in_constructor():
    class Kept(dict):
        pass

    Kept(dict(os.environ))
    raise RuntimeError("kept")


def test_handed_to_a_hidden_constructor():
    class HiddenConstructed:
        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    HiddenConstructed(["git"], dict(os.environ))


def test_handed_to_a_hidden_static_constructor():
    class StaticHiddenBuilt:
        @staticmethod
        @hides_environment
        def __init__(command, environment):
            raise ValueError(command)

    StaticHiddenBuilt(["git"], dict(os.environ))


def test_handed_to_a_hidden_static_new():
    class StaticHiddenNew:
        @staticmethod
        @hides_environment
        def __new__(cls, command, environment):
            raise ValueError(command)

    StaticHiddenNew(["git"], dict(os.environ))


def test_handed_to_a_hidden_new_and_init():
    class HiddenNewInit:
        @hides_environment
        def __new__(cls, command, environment):
            return object.__new__(cls)

        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    HiddenNewInit(["git"], dict(os.environ))


def test_handed_through_a_receiver_named_as_its_method():
    class SelfNamed:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def own(own):
            own.probe(["git"], dict(os.environ))

    SelfNamed().own()


def test_handed_past_a_literal_star():
    class LiteralStar:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    LiteralStar().probe(*(["git"],), *[dict(os.environ)])


def test_handed_to_a_method_through_self_of_a_dotted_named_tuple():
    class DottedPair(typing.NamedTuple):
        first: str

        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self, command):
            self.probe(command, dict(os.environ))

    DottedPair("a").check(["git"])


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

    check(Launch("git", env=dict(os.environ)))


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

    check(Rebuilt("git", env=dict(os.environ)))


def test_hidden_refuses_an_object_printing_its_attributes():
    @hides_environment
    def check(record):
        raise RuntimeError(record.command)

    shown = Shown("git")
    shown.env = dict(os.environ)
    check(shown)


def test_hidden_refuses_a_slotted_object_printing_its_attributes():
    @hides_environment
    def check(record):
        raise RuntimeError(record.command)

    slotted = Slotted("git")
    slotted.env = dict(os.environ)
    check(slotted)


def test_flagged_object_forwarding_its_attributes():
    @hides_environment
    def check(record):
        raise RuntimeError(record.command)

    forwarding = Forwarding("git")
    forwarding.held.update(os.environ)
    check(forwarding)


def test_flagged_method_of_a_fixture_instance(prober):
    prober.probe(["git"], dict(os.environ))


def test_flagged_class_named_twice():
    class Twice:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

    Twice().probe(["git"], dict(os.environ))


def test_flagged_module_base_before_one_out_of_view():
    class ModuleFirst(Probe, under_test.Base):
        def check(self, command):
            self.probe(command, dict(os.environ))

    ModuleFirst().check(["git"])


def test_flagged_new_of_a_class_built_on_by_one_out_of_view():
    class QuietRenewed:
        @hides_environment
        def __new__(cls, command, environment):
            return object.__new__(cls)

        @hides_environment
        def __init__(self, command, environment):
            raise ValueError(command)

    class RenewedOutside(QuietRenewed, under_test.Base):
        pass

    QuietRenewed(["git"], dict(os.environ))


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


def test_flagged_a_record_s_field_by_position():
    record = Paired(dict(os.environ), {})
    assert dataclasses.asdict(record)["env"] is None


def test_flagged_hidden_named_tuple_by_position():
    @hides_environment
    def check(launch):
        raise RuntimeError(launch.command)

    check(Launch("git", dict(os.environ)))


def test_flagged_os_of_an_object():
    fake = SimpleNamespace(os=SimpleNamespace(environ={"X": "ok"}))
    assert fake.os.environ["AAA_MISSING"] == "x"
"""

# The code under test the examples hand fakes to. It wraps what they raise, as
# ``ci_parity`` and ``repo_routing`` do, so pytest prints the chain.
UNDER_TEST = """
import os


class Refusal(Exception):
    pass


def probe(command, environment):
    raise ValueError(command)


def make(record):
    return record(dict(os.environ))


def hand(function, record):
    return function(make(record))


class ShownError(Exception):
    def __repr__(self):
        return repr(vars(self))


def final(cls):
    cls.__repr__ = lambda self: repr(vars(self))
    return cls


class Base:
    def probe(self, command, environment):
        raise ValueError(command)


class Built:
    def __init__(self, command, environment):
        raise ValueError(command)


class Factory:
    def __new__(cls, *arguments):
        return Base()


class Rewriting:
    def __init_subclass__(cls, **keywords):
        super().__init_subclass__(**keywords)
        if "probe" in vars(cls):
            cls.probe = Base.probe
        if "__init__" in vars(cls):
            cls.__init__ = Built.__init__


NamedTuple = Rewriting


class Printing(type):
    def __repr__(cls):
        return repr(getattr(cls, "env", None))


class PrintingBase(metaclass=Printing):
    pass


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


def _pytest_reports(directory: Path, *paths: Path) -> dict[str, str]:
    """Each failed test's report from pytest at its most verbose over
    ``paths``, run in ``directory`` in an environment holding nothing else."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-vvv", "-p", "no:cacheprovider"]
        + ["-c", os.devnull, "--rootdir", str(directory), *map(str, paths)],
        cwd=directory,
        env={
            "AAA_SENTINEL": SENTINEL,
            "AAA_OTHER": "x",
            "PATH": os.defpath,
            "HOME": str(directory),
            "PYTHONPATH": str(REPO),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    return _failure_output(completed.stdout)


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

    reports = _pytest_reports(tmp_path, examples)
    printed = {name for name, text in reports.items() if SENTINEL in text}
    overflagged = {name for name in tests if "_flagged_" in name}

    assert sorted(reports) == sorted(tests)
    assert sorted(printed) == sorted(name for name in tests if "_leaks_" in name)
    assert sorted(flagged) == sorted(printed | overflagged)


# Shapes that change what the check knows of a whole module, which
# ``EXAMPLES`` can't hold: a star import, ``setattr`` given a name that is not
# a constant, a method writing into a namespace, ``getattr`` reading a class's
# function by such a name, a base a call builds or one a name holds that the
# module binds otherwise, ``patch.multiple`` given names by ``**``, a class's
# ``__new__`` replaced, a built-in class's name bound, ``NamedTuple`` or
# ``object`` imported as another class, an instance's class reassigned, a
# dotted import binding its first name, two modules' names swapped,
# another module imported as ``os`` (the control), another module's ``os``
# read where this one binds that name otherwise, and ``NamedTuple`` (read
# by its own name or imported as another) or a built-in base (``object``
# too) replaced as an attribute; and a record's method (one built on another
# record's fields too), flagged at its ``def``, outside any test, where the
# code under test builds the record, beside a class method, which prints
# only the class; a record's method where a call builds it with another
# field, and a plain base's where a record built on it is; a class method where a
# metaclass prints what the class holds: the class's own, one it inherits
# from a class of the module's or from one out of view, one of a plain
# class, one of a class built on its own, and one its body binds; a class
# method that binds its class's parameter again, and one of a plain base
# given a class on a base a call builds; a class attribute an instance reads,
# bound in the class's body (through an alias, under an ``if``, unpacked) or
# stored on the class, and one read off the class; a record built through an
# alias, on an alias, through ``type(...)``, through a class method's class,
# through a choice a function binds, and unassigned through an alias; a
# field's default from its factory, its value and a helper; a parameter
# annotated with a record's alias; a class given its repr under an ``if``, by
# metaclass or by assignment (a class a test defines too); the first
# parameter of a class built on an imported or replaced ``object``, on a
# class built on an imported ``object``, on an imported exception class, or
# given another module's ``final``; an implicit class method storing on a
# class that prints its instances' attributes; and what a method, a static
# method and an instance's ``__call__`` return. With its
# trigger in place of ``TRIGGER``, each source prints the sentinel and the
# check flags the lines marked ``# flagged``; with the control there, the
# check flags nothing.
MODULE_WIDE: dict[str, tuple[str, str, str]] = {
    "star_import": (
        """import os

from tests.environment_assertions import hides_environment
TRIGGER


def test_unrelated():
    class Base:
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)


def test_star_import():
    Base().probe(["git"], dict(os.environ))  # flagged
""",
        "from under_test import *",
        "",
    ),
    "name_set_by_a_variable": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Looped:
    @hides_environment
    def looped(self, command, environment):
        raise ValueError(command)


def test_name_set_by_a_variable(monkeypatch):
    looped_probe = Looped()
    for name in ("looped",):
        TRIGGER
    looped_probe.looped(["git"], dict(os.environ))  # flagged
""",
        "monkeypatch.setattr(looped_probe, name, under_test.probe)",
        "pass",
    ),
    "namespace_updated": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Updated:
    @hides_environment
    def updated(self, command, environment):
        raise ValueError(command)


def test_namespace_updated():
    updated_probe = Updated()
    TRIGGER
    updated_probe.updated(["git"], dict(os.environ))  # flagged
""",
        "vars(updated_probe).update(updated=under_test.probe)",
        "pass",
    ),
    "method_read_by_a_variable": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Unbound:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self, command):
        self.probe(command, dict(os.environ))  # flagged


def test_method_read_by_a_variable():
    name = "check"
    TRIGGER
""",
        'getattr(Unbound, name)(under_test.Base(), ["git"])',
        'Unbound().check(["git"])',
    ),
    "base_built_by_a_call": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Quiet:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self):
        self.probe(["git"], dict(os.environ))  # flagged


def quiet():
    return Quiet


class Loud(TRIGGER):
    pass


def test_base_built_by_a_call():
    Loud().check()
""",
        "under_test.Base, quiet()",
        "Quiet",
    ),
    "base_bound_by_a_call": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Picked:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self):
        self.probe(["git"], dict(os.environ))  # flagged


def picked():
    return Picked


TRIGGER


class Chooser(under_test.Base, Chosen):
    pass


def test_base_bound_by_a_call():
    Chooser().check()
""",
        "Chosen = picked()",
        "",
    ),
    "methods_patched_by_a_mapping": (
        """import os
from contextlib import nullcontext
from unittest.mock import patch

import under_test
from tests.environment_assertions import hides_environment


class Mapped:
    @hides_environment
    def mapped(self, command, environment):
        raise ValueError(command)


def test_methods_patched_by_a_mapping():
    mapped_probe = Mapped()
    with TRIGGER:
        mapped_probe.mapped(["git"], dict(os.environ))  # flagged
""",
        'patch.multiple(Mapped, **{"mapped": under_test.Base.probe})',
        "nullcontext()",
    ),
    "new_replaced": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Renewed:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)


def test_new_replaced(monkeypatch):
    TRIGGER
    Renewed().probe(["git"], dict(os.environ))  # flagged
""",
        'monkeypatch.setattr(Renewed, "__new__", lambda cls: under_test.Base())',
        "pass",
    ),
    "built_in_shadowed": (
        """import os

import under_test


TRIGGER


class OnShadowed(bytearray):
    pass


def test_built_in_shadowed():
    OnShadowed(["git"], dict(os.environ))  # flagged
""",
        "bytearray = under_test.Built",
        "",
    ),
    "record_method": (
        """import dataclasses

import under_test


@dataclasses.dataclass
class HeldLaunch:
    TRIGGER: dict

    def check(self):  # flagged
        raise ValueError("held")

    @classmethod
    def build(cls):
        raise ValueError(cls)

    def __init_subclass__(cls):
        raise ValueError(cls)


def test_record_method():
    under_test.make(HeldLaunch).check()


def test_record_class_method():
    HeldLaunch.build()


def test_record_subclassed():
    class HeldChild(HeldLaunch):
        pass
""",
        "env",
        "values",
    ),
    "inherited_record_method": (
        """import dataclasses

import under_test


@dataclasses.dataclass
class BaseLaunch:
    TRIGGER: dict


@dataclasses.dataclass
class NamedLaunch:
    name: str = "git"


@dataclasses.dataclass
class ChildLaunch(NamedLaunch, BaseLaunch):
    command: str = "run"

    def check(self):  # flagged
        raise ValueError(self.command)


def test_inherited_record_method():
    under_test.make(ChildLaunch).check()
""",
        "env",
        "values",
    ),
    "record_method_given_another_field": (
        """import dataclasses
import os


@dataclasses.dataclass
class Given:
    values: dict

    def check(self):  # flagged
        raise ValueError("given")


def test_record_method_given_another_field():
    Given(TRIGGER).check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_method_beside_a_base_a_call_builds": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    values: dict

    def check(self):  # flagged
        raise ValueError("record")


def base():
    return Record


class Other(base()):
    pass


def test_record_method_beside_a_base_a_call_builds():
    Record(TRIGGER).check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_by_another_class_s_method": (
        """import dataclasses
import os


@dataclasses.dataclass
class Given:
    values: dict

    def check(self):  # flagged
        raise ValueError("given")


class Maker:
    def make(self):
        return Given(TRIGGER)


def test_record_built_by_another_class_s_method():
    Maker().make().check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_at_import_and_kept": (
        """import dataclasses
import os

KEPT = []


@dataclasses.dataclass
class Kept:
    values: dict

    def __post_init__(self):  # flagged
        KEPT.append(self)

    def check(self):  # flagged
        raise ValueError("kept")


Kept(TRIGGER)


def test_record_built_at_import_and_kept():
    KEPT[0].check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_name_shadowed_by_a_mapping": (
        """import dataclasses
import os


@dataclasses.dataclass
class Shadowed:
    values: dict

    def check(self):  # flagged
        raise ValueError("shadowed")


def shadow():
    Shadowed = {"held": dict(os.environ)}
    return len(Shadowed)


def test_record_name_shadowed_by_a_mapping():
    Shadowed(TRIGGER).check()  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "plain_base_s_method_on_a_record": (
        """import dataclasses
import os


class Plain:
    def check(self):  # flagged
        raise ValueError("plain")


TRIGGER
class Shown(Plain):
    values: dict = dataclasses.field(default_factory=dict)


def test_plain_base_s_method_on_a_record():
    Shown(dict(os.environ)).check()
""",
        "@dataclasses.dataclass",
        "",
    ),
    "named_tuple_imported_otherwise": (
        """import os

from tests.environment_assertions import hides_environment
TRIGGER


class QuietNamed(NamedTuple):
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self):
        self.probe(["git"], dict(os.environ))  # flagged


def test_named_tuple_imported_otherwise():
    QuietNamed().check()
""",
        "from under_test import NamedTuple",
        "from typing import NamedTuple",
    ),
    "object_imported_otherwise": (
        """import os

from tests.environment_assertions import hides_environment
TRIGGER


class QuietObject(object):
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self):
        self.probe(["git"], dict(os.environ))  # flagged


def test_object_imported_otherwise():
    QuietObject().check()
""",
        "from under_test import Rewriting as object",
        "",
    ),
    "class_reassigned": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


class Reclassed:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)


def test_class_reassigned():
    reclassed = Reclassed()
    TRIGGER
    reclassed.probe(["git"], dict(os.environ))  # flagged
""",
        "reclassed.__class__ = under_test.Base",
        "pass",
    ),
    "dotted_import": (
        """import os

from tests.environment_assertions import hides_environment


class Holder:
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)


def test_dotted_import():
    holder = Holder()
    TRIGGER
    holder.probe(["git"], dict(os.environ))  # flagged
""",
        "import holder.sub",
        "pass",
    ),
    "modules_swapped": (
        """import os
import sys

TRIGGER


def test_modules_swapped():
    assert sys.environ["AAA_MISSING"] == "x"  # flagged
""",
        "os, sys = sys, os",
        "",
    ),
    "module_imported_as_another": (
        """import sys

TRIGGER


def test_module_imported_as_another():
    assert os.environ["AAA_MISSING"] == "x"  # flagged
""",
        "import os",
        "import sys as os",
    ),
    "another_module_s_os": (
        """import subprocess
import sys as os


def test_another_module_s_os():
    assert TRIGGER["AAA_MISSING"] == "x"  # flagged
""",
        "subprocess.os.environ",
        "os.environ",
    ),
    "named_tuple_replaced": (
        """import os
import typing

import under_test
from tests.environment_assertions import hides_environment


def test_named_tuple_replaced(monkeypatch):
    TRIGGER

    class Replaced(typing.NamedTuple):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))  # flagged

    Replaced().check()
""",
        'monkeypatch.setattr(typing, "NamedTuple", under_test.Rewriting)',
        "pass",
    ),
    "built_in_base_replaced": (
        """import os

import under_test
from tests.environment_assertions import hides_environment


def test_built_in_base_replaced(monkeypatch):
    TRIGGER

    class Replaced(complex):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))  # flagged

    Replaced().check()
""",
        'monkeypatch.setattr("builtins.complex", under_test.Rewriting)',
        "pass",
    ),
    "named_tuple_replaced_and_imported_as_another": (
        """import os
import typing

import under_test
from tests.environment_assertions import hides_environment


def test_named_tuple_replaced_and_imported_as_another(monkeypatch):
    TRIGGER
    from typing import NamedTuple as Tuple

    class Replaced(Tuple):
        @hides_environment
        def probe(self, command, environment):
            raise ValueError(command)

        def check(self):
            self.probe(["git"], dict(os.environ))  # flagged

    Replaced().check()
""",
        'monkeypatch.setattr(typing, "NamedTuple", under_test.Rewriting)',
        "pass",
    ),
    "object_replaced": (
        """import builtins
import os

import under_test
from tests.environment_assertions import hides_environment


def test_object_replaced(monkeypatch):
    with monkeypatch.context() as patch:
        TRIGGER

        class Replaced(object):
            @hides_environment
            def probe(self, command, environment):
                raise ValueError(command)

            def check(self):
                self.probe(["git"], dict(os.environ))  # flagged

    Replaced().check()
""",
        'patch.setattr(builtins, "object", under_test.Rewriting)',
        "pass",
    ),
    "class_method_of_a_printing_class": (
        """import dataclasses
import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.env)


@dataclasses.dataclass
class Shown(TRIGGER):
    env: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("shown")


def test_class_method_of_a_printing_class():
    Shown.env = dict(os.environ)
    Shown.check()
""",
        "metaclass=Printing",
        "",
    ),
    "class_method_of_a_class_inheriting_a_metaclass": (
        """import dataclasses
import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.env)


class PrintingBase(metaclass=Printing):
    pass


class PlainBase:
    pass


@dataclasses.dataclass
class Inheriting(TRIGGER):
    env: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("inheriting")


def test_class_method_of_a_class_inheriting_a_metaclass():
    Inheriting.env = dict(os.environ)
    Inheriting.check()
""",
        "PrintingBase",
        "PlainBase",
    ),
    "class_method_of_a_class_on_a_base_out_of_view": (
        """import dataclasses
import os

import under_test


@dataclasses.dataclass
class Based(TRIGGER):
    env: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("based")


def test_class_method_of_a_class_on_a_base_out_of_view():
    Based.env = dict(os.environ)
    Based.check()
""",
        "under_test.PrintingBase",
        "object",
    ),
    "class_method_of_a_plain_printing_class": (
        """import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.env)


class Kept(TRIGGER):
    @classmethod
    def keep(cls):  # flagged
        cls.env = dict(os.environ)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("kept")

    def __init_subclass__(cls):  # flagged
        raise ValueError("subclassed")


def test_class_method_of_a_plain_printing_class():
    Kept.keep()
    Kept.check()
""",
        "metaclass=Printing",
        "",
    ),
    "class_method_rebinding_its_class": (
        """import dataclasses
import os


@dataclasses.dataclass
class Rebinding:
    env: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def check(cls):  # flagged
        TRIGGER
        raise ValueError("rebound")


def test_class_method_rebinding_its_class():
    Rebinding.check()
""",
        "cls = dict(os.environ)",
        "held = dict(os.environ)",
    ),
    "class_method_inherited_by_a_printing_class": (
        """import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.env)


class Base:
    @classmethod
    def keep(cls):  # flagged
        cls.env = dict(os.environ)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("inherited")


class Child(Base, TRIGGER):
    pass


def test_class_method_inherited_by_a_printing_class():
    Child.keep()
    Child.check()
""",
        "metaclass=Printing",
        "",
    ),
    "class_method_inherited_through_a_base_chosen_at_run_time": (
        """import os


class Printing(type):
    def __repr__(cls):  # flagged
        return repr(cls.env)


class Base:
    @classmethod
    def keep(cls):  # flagged
        cls.env = dict(os.environ)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("chosen")


Chosen = Base if os.name else object


class Child(TRIGGER):
    pass


def test_class_method_inherited_through_a_base_chosen_at_run_time():
    Child.keep()
    Child.check()
""",
        "Chosen, metaclass=Printing",
        "Base",
    ),
    "class_attribute_read_off_an_instance": (
        """import os

import under_test


class Holder:
    saved = TRIGGER

    def check(self):
        under_test.probe(["git"], self.saved)  # flagged


def test_class_attribute_read_off_an_instance():
    Holder().check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "annotated_class_attribute_read_off_an_instance": (
        """import os

import under_test


class Holder:
    saved: dict = TRIGGER

    def check(self):
        under_test.probe(["git"], self.saved)  # flagged


def test_annotated_class_attribute_read_off_an_instance():
    Holder().check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_attribute_stored_on_the_class": (
        """import os

import under_test


class Holder:
    def check(self):
        under_test.probe(["git"], self.saved)  # flagged


def test_class_attribute_stored_on_the_class():
    Holder.saved = TRIGGER
    Holder().check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_attribute_printed_by_a_metaclass": (
        """import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.saved)


class Holder(TRIGGER):
    saved = dict(os.environ)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("held")


def test_class_attribute_printed_by_a_metaclass():
    Holder.check()
""",
        "metaclass=Printing",
        "",
    ),
    "record_child_of_a_plain_base_beside_a_base_a_call_builds": (
        """import dataclasses
import os


class Base:
    def check(self):  # flagged
        raise ValueError("inherited")


@dataclasses.dataclass
class Child(Base):
    data: dict


class Other((lambda: object)()):
    pass


def test_record_child_of_a_plain_base_beside_a_base_a_call_builds():
    Child(TRIGGER).check()
""",
        "dict(os.environ)",
        "{}",
    ),
    "first_parameter_of_a_class_built_on_an_imported_object": (
        """import os
TRIGGER


class Plain(object):
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("imported")


def test_first_parameter_of_a_class_built_on_an_imported_object():
    Plain().check()
""",
        "from types import SimpleNamespace as object",
        "",
    ),
    "first_parameter_of_a_class_built_on_a_replaced_object": (
        """import builtins
import os
from types import SimpleNamespace


def test_first_parameter_of_a_class_built_on_a_replaced_object(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "TRIGGER", SimpleNamespace)

        class Plain(object):
            def check(self):  # flagged
                self.saved = dict(os.environ)
                raise ValueError("replaced")

    Plain().check()
""",
        "object",
        "ascii",
    ),
    "record_built_through_an_alias": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict


Alias = Record


def test_record_built_through_an_alias():
    record = Alias(data=TRIGGER)
    assert record is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_on_an_alias": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict


Alias = Record


@dataclasses.dataclass
class Child(Alias):
    pass


def test_record_built_on_an_alias():
    record = Child(data=TRIGGER)
    assert record is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_through_its_type": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict


def test_record_built_through_its_type():
    built = type(Record({}))(TRIGGER)
    assert built is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_by_a_class_method_s_class": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict

    @classmethod
    def made(cls):
        return cls(TRIGGER)


def test_record_built_by_a_class_method_s_class():
    assert Record.made() is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_given_a_default_by_its_field_s_factory": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict = dataclasses.field(default_factory=lambda: TRIGGER)


def test_record_given_a_default_by_its_field_s_factory():
    record = Record()
    assert record is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_given_a_default_by_a_helper": (
        """import os
from dataclasses import dataclass, field


def defaults():
    return TRIGGER


@dataclass
class Record:
    data: dict = field(default_factory=defaults)


def test_record_given_a_default_by_a_helper():
    record = Record()
    assert record is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_given_a_default_value_by_its_field": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: tuple = dataclasses.field(default=TRIGGER)


def test_record_given_a_default_value_by_its_field():
    record = Record()
    assert record is None  # flagged
""",
        "tuple(os.environ.items())",
        "()",
    ),
    "class_attribute_bound_through_an_alias": (
        """import os


class Holder:
    saved = TRIGGER
    alias = saved


def test_class_attribute_bound_through_an_alias():
    held = Holder().alias
    assert held is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_attribute_bound_under_an_if": (
        """import os


class Holder:
    if os.name:
        saved = TRIGGER


def test_class_attribute_bound_under_an_if():
    held = Holder().saved
    assert held is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_attribute_unpacked": (
        """import os


class Holder:
    (saved,) = [TRIGGER]


def test_class_attribute_unpacked():
    held = Holder().saved
    assert held is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_attribute_read_off_the_class": (
        """import os


class Holder:
    saved = TRIGGER

    def check(self):
        assert self is None

    def keep(self):
        self.kept = dict(os.environ)
        assert self is None


def test_class_attribute_read_off_the_class():
    assert Holder.saved is None  # flagged


def test_class_printed():
    assert Holder is None


def test_instance_printed_by_its_method():
    Holder().check()


def test_instance_printed_after_its_method_stores():
    Holder().keep()


def test_instance_printed_after_the_test_stores():
    holder = Holder()
    holder.held = dict(os.environ)
    assert holder is None
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_binding_its_repr_under_an_if": (
        """import os


class Holder:
    if os.name:

        def TRIGGER(self):
            return repr(vars(self))

    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("bound")


def test_class_binding_its_repr_under_an_if():
    Holder().check()
""",
        "__repr__",
        "describe",
    ),
    "class_given_its_repr_by_a_metaclass": (
        """import os


class Giving(type):
    def __new__(mcs, name, bases, namespace):
        namespace["__repr__"] = lambda self: repr(vars(self))
        return super().__new__(mcs, name, bases, namespace)


class Holder(TRIGGER):
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("given")


def test_class_given_its_repr_by_a_metaclass():
    Holder().check()
""",
        "metaclass=Giving",
        "",
    ),
    "class_given_its_repr_by_assignment": (
        """import os


class Holder:
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("assigned")


Holder.TRIGGER = lambda self: repr(vars(self))


def test_class_given_its_repr_by_assignment():
    Holder().check()
""",
        "__repr__",
        "describe",
    ),
    "what_a_method_returns": (
        """import os


class Maker:
    def made(self):
        return TRIGGER

    @staticmethod
    def given():
        return {}


def test_what_a_method_returns():
    assert Maker().made() is None  # flagged


def test_what_another_method_returns():
    assert Maker.given() is None
""",
        "dict(os.environ)",
        "{}",
    ),
    "what_a_static_method_returns": (
        """import os


class Maker:
    @staticmethod
    def made():
        return TRIGGER


def test_what_a_static_method_returns():
    made = Maker.made()
    assert made is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "what_an_instance_s_call_returns": (
        """import os


class Calling:
    def __call__(self):
        return TRIGGER


def test_what_an_instance_s_call_returns():
    calling = Calling()
    assert calling() is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "class_method_given_a_class_on_a_base_a_call_builds": (
        """import os


class Printing(type):
    def __repr__(cls):
        return repr(cls.saved)


class Base:
    saved = dict(os.environ)

    @classmethod
    def check(cls):  # flagged
        raise ValueError("chosen")


class Child(TRIGGER):
    pass


def test_class_method_given_a_class_on_a_base_a_call_builds():
    Child.check()
""",
        "(lambda: Base)(), metaclass=Printing",
        "Base",
    ),
    "record_built_through_a_choice_a_function_binds": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict


def choose():
    global Chosen
    Chosen = Record if os.name else Record


choose()
Alias = Chosen


def test_record_built_through_a_choice_a_function_binds():
    record = Alias(data=TRIGGER)
    assert record is None  # flagged
""",
        "dict(os.environ)",
        "{}",
    ),
    "record_built_unassigned_through_an_alias": (
        """import dataclasses
import os


@dataclasses.dataclass
class Record:
    data: dict

    def __post_init__(self):  # flagged
        raise ValueError("built")


Alias = Record


def test_record_built_unassigned_through_an_alias():
    Alias(TRIGGER)
""",
        "dict(os.environ)",
        "{}",
    ),
    "parameter_annotated_with_a_record_s_alias": (
        """import dataclasses

import under_test


@dataclasses.dataclass
class Record:
    env: dict


Alias = Record


def examine(record: TRIGGER):  # flagged
    raise ValueError("examine")


def test_parameter_annotated_with_a_record_s_alias():
    under_test.hand(examine, Record)
""",
        "Alias",
        "object",
    ),
    "instance_of_a_class_a_test_defines_given_its_repr_by_assignment": (
        """import os


def test_instance_of_a_class_a_test_defines_given_its_repr_by_assignment():
    class Local:
        def keep(self):  # flagged
            self.saved = dict(os.environ)

    Local.TRIGGER  # flagged
    local = Local()
    local.keep()
    assert local is None  # flagged
""",
        "__repr__ = lambda self: repr(vars(self))",
        "__doc__ = None",
    ),
    "first_parameter_of_a_class_built_on_an_imported_exception": (
        """import os

from under_test import ShownError


class Plain(TRIGGER):
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("plain")


def test_first_parameter_of_a_class_built_on_an_imported_exception():
    Plain().check()
""",
        "ShownError",
        "Exception",
    ),
    "first_parameter_of_a_class_given_another_module_s_final": (
        """import os

from TRIGGER import final


@final
class Plain:
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("plain")


def test_first_parameter_of_a_class_given_another_module_s_final():
    Plain().check()
""",
        "under_test",
        "typing",
    ),
    "first_parameter_of_a_class_built_on_one_built_on_an_imported_object": (
        """import os
TRIGGER


class Base(object):
    pass


class Plain(Base):
    def check(self):  # flagged
        self.saved = dict(os.environ)
        raise ValueError("plain")


def test_first_parameter_of_a_class_built_on_one_built_on_an_imported_object():
    Plain().check()
""",
        "from types import SimpleNamespace as object",
        "",
    ),
    "class_attribute_beside_an_implicit_class_method_storing_on_its_class": (
        """import os


class Shows:
    saved = TRIGGER

    def __repr__(self):
        return "shows"

    def __class_getitem__(cls, item):
        cls.kept = dict(os.environ)
        raise ValueError("item")


def test_class_attribute_beside_an_implicit_class_method_storing_on_its_class():
    assert Shows.saved is None  # flagged


def test_class_printed_by_its_implicit_class_method():
    Shows[0]
""",
        "dict(os.environ)",
        "{}",
    ),
}


def test_the_check_reads_the_whole_module(tmp_path: Path) -> None:
    (tmp_path / "under_test.py").write_text(UNDER_TEST)
    package = tmp_path / "holder"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def probe(command, environment):\n    raise ValueError(command)\n"
    )
    (package / "sub.py").write_text("")
    paths = []
    for name, (source, trigger, control) in MODULE_WIDE.items():
        triggered = source.replace("TRIGGER", trigger)
        marked = [
            number
            for number, line in enumerate(triggered.splitlines(), 1)
            if line.endswith("# flagged")
        ]
        assert environment_leaks(triggered) == marked, name
        assert environment_leaks(source.replace("TRIGGER", control)) == [], name
        paths.append(tmp_path / f"test_{name}.py")
        paths[-1].write_text(triggered)
    reports = _pytest_reports(tmp_path, *paths)
    printed = sorted(name for name, text in reports.items() if SENTINEL in text)
    assert printed == sorted(f"test_{name}" for name in MODULE_WIDE)


@pytest.mark.parametrize(
    ("bound", "base", "trusted"),
    [
        ("from typing import NamedTuple", "NamedTuple", True),
        ("import typing", "typing.NamedTuple", True),
        ("from .typing import NamedTuple", "NamedTuple", False),
        ("from under_test import typing", "typing.NamedTuple", False),
    ],
)
def test_only_typing_s_named_tuple_is_trusted(
    bound: str, base: str, trusted: bool
) -> None:
    """A base named ``NamedTuple`` leaves a class's methods in view only where
    it is ``typing``'s; a module of another name may replace them. These
    sources import what this file's examples can't, so only the check runs."""
    source = f"""import os
{bound}
from tests.environment_assertions import hides_environment


class Quiet({base}):
    @hides_environment
    def probe(self, command, environment):
        raise ValueError(command)

    def check(self):
        self.probe(["git"], dict(os.environ))


def test_quiet():
    Quiet().check()
"""
    lines = source.splitlines()
    marked = next(n for n, line in enumerate(lines, 1) if "self.probe(" in line)
    assert environment_leaks(source) == ([] if trusted else [marked])


def test_a_name_is_read_as_this_module_binds_it() -> None:
    """A name is the module this one binds it to; an attribute of another
    module, what its own name says, whatever this one binds that name to."""
    check = _Check(ast.parse("import os as subprocess\nimport sys as os\n"))
    assert check.origins(["os", "environ"]) == {"sys"}
    assert check.origins(["subprocess", "os", "environ"]) == {"os"}
    assert not check._launcher("subprocess.run")
    assert check._launcher("asyncio.subprocess.create_subprocess_exec")
    assert check._launcher("under_test.subprocess.run")
    # ``environ`` read off what no name leads to is no module's.
    assert environment_leaks("def test_read():\n    assert make().environ['X']\n") == []


@pytest.mark.parametrize(
    ("replacing", "named_tuple", "built_in", "plain"),
    [
        ("pass", True, True, True),
        ("typing.NamedTuple = None", False, True, True),
        ('monkeypatch.setattr("builtins.complex", None)', True, False, True),
        ('monkeypatch.setattr(builtins, "object", None)', True, True, False),
        ("setattr(builtins, name, None)", False, False, False),
    ],
)
def test_a_base_the_module_may_replace_is_not_trusted(
    replacing: str, named_tuple: bool, built_in: bool, plain: bool
) -> None:
    """``typing.NamedTuple`` and a built-in class are trusted as bases only
    where the module replaces neither as an attribute, by its name, the name
    it is imported as or one that is not a constant; and ``object`` is left
    out of an order only so."""
    check = _Check(
        ast.parse(
            "import builtins\nimport typing\n"
            "from typing import NamedTuple as Tuple\n\n\n"
            "class Plain(object):\n    pass\n\n\n"
            f"def test_it():\n    {replacing}\n"
        )
    )
    assert check._inert_base("typing.NamedTuple") == named_tuple
    assert check._inert_base("Tuple") == named_tuple
    assert check._inert_base("complex") == built_in
    assert check._named(check.known_classes["Plain"]) == plain


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.lists(st.integers(0, 5), max_size=3), min_size=1, max_size=6))
def test_the_lookup_order_is_pythons(bases: list[list[int]]) -> None:
    """``_Check._order`` is each class's ``__mro__`` (less ``object``), and
    None exactly where Python refuses to build the class."""
    chosen = [
        [f"C{base}" for base in each if base < index]
        for index, each in enumerate(bases)
    ]
    source = "\n".join(
        f"class C{index}({', '.join(names)}):\n    pass\n"
        for index, names in enumerate(chosen)
    )
    check = _Check(ast.parse(source))
    built: dict[str, type] = {}
    for index, names in enumerate(chosen):
        name = f"C{index}"
        if all(base in built for base in names):
            try:
                built[name] = type(name, tuple(built[base] for base in names), {})
            except TypeError:
                pass
        order = check._order(check.known_classes[name])
        if name in built:
            assert order is not None
            assert [owner.name for owner in order] == [
                owner.__name__ for owner in built[name].__mro__[:-1]
            ]
        else:
            assert order is None


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


class _Printing:
    """An object whose repr prints what it holds under the field it is
    given."""

    def __init__(self, field: str, held: object) -> None:
        self.field = field
        setattr(self, field, held)

    def __repr__(self) -> str:
        return f"_Printing({getattr(self, self.field)!r})"


@pytest.mark.parametrize("field", ["env", "environment"])
def test_hides_environment_refuses_an_object_printing_an_environment(
    field: str,
) -> None:
    seen: list[object] = []

    @hides_environment
    def fake(record: object) -> None:
        seen.append(record)

    # Holding ``Names``, it prints names alone, and is passed as it is.
    kept = _Printing(field, Names({"A": "1"}))
    fake(kept)
    assert len(seen) == 1 and seen[0] is kept
    with pytest.raises(TypeError, match="could not hide"):
        fake(_Printing(field, {"A": "1"}))


def test_hides_environment_passes_classes_and_modules_as_they_are() -> None:
    seen: list[object] = []

    @hides_environment
    def fake(owner: object) -> None:
        seen.append(owner)

    class Holder:
        env = {"A": "1"}

    module = types.ModuleType("holder")
    module.environment = {"A": "1"}
    fake(Holder)
    fake(module)
    assert seen[0] is Holder and seen[1] is module


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
