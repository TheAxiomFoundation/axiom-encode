"""Failing asserts in this suite must not print a whole environment.

See ``tests/environment_assertions.py`` for why. The check below models what
pytest's assertion rewriting (``_pytest/assertion/rewrite.py``) prints for a
failing assert: each operand of a comparison or a boolean or arithmetic
operator; each call with its function and arguments; each attribute with the
object it is read from; local names; and the assert's message. Anything else (a
subscript, a comprehension, a literal) is printed as its result alone.

An assert is flagged when something it prints may be a whole environment:
``os.environ``, a copy of it, an ``env`` keyword argument a fake recorded from
the code under test, a name bound to one of those, or a container one was
stored in. ``test_the_check_agrees_with_pytest`` runs pytest on examples of
each shape to keep the model honest.

The model flags more than pytest prints in one case. When ``==`` fails between
two strings, lists, sets or dicts, pytest prints its own diff of the two results
in place of what their operands would print. The operands' types are not known
here, so such asserts are flagged as well.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from tests.environment_assertions import changed_names

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
# Keyword arguments that carry a subprocess environment.
ENVIRONMENT_KEYS = frozenset({"env", "environment"})
# Names a fake gives the keyword arguments it records.
KWARGS_NAMES = frozenset({"kwargs", "kw", "kwds"})
COPIES = frozenset({"dict", "copy", "MappingProxyType"})


class _Scope:
    """What one test (or helper) binds to whole environments."""

    def __init__(self) -> None:
        # Names bound to a whole environment.
        self.environments: set[str] = set()
        # Containers holding one, with the keys it is stored under: "*" for
        # every item, ("item", i) for position i of every item.
        self.holders: dict[str, set[object]] = {}

    def hold(self, name: str, key: object) -> bool:
        keys = self.holders.setdefault(name, set())
        added = key not in keys
        keys.add(key)
        return added


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
        self.functions = [
            node
            for owner in [tree, *[n for n in tree.body if isinstance(n, ast.ClassDef)]]
            for node in owner.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        # Helpers that store a whole environment into a parameter, such as a
        # fake ``main`` installer filling the caller's record:
        # helper name -> {argument position: keys}.
        self.filled_parameters: dict[str, dict[int, set[object]]] = {}
        for _ in range(2):
            for function in self.functions:
                scope = self._bind(function)
                parameters = function.args.posonlyargs + function.args.args
                for position, parameter in enumerate(parameters):
                    if parameter.arg in scope.holders:
                        self.filled_parameters.setdefault(function.name, {})[
                            position
                        ] = set(scope.holders[parameter.arg])

    # -- what is a whole environment --------------------------------------

    def _environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "environb"}:
            return isinstance(node.value, ast.Name) and node.value.id == "os"
        return isinstance(node, ast.Name) and node.id in self.aliases

    @staticmethod
    def _kwargs(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in KWARGS_NAMES
        return isinstance(node, ast.Attribute) and node.attr == "kwargs"

    def environment(self, node: ast.AST, scope: _Scope) -> bool:
        """Whether ``node``'s value may be a whole environment."""
        if self._environ(node):
            return True
        if isinstance(node, ast.Name):
            return node.id in scope.environments
        if isinstance(node, ast.NamedExpr):
            return self.environment(node.value, scope)
        if isinstance(node, ast.Dict):
            return any(
                key is None and self.environment(value, scope)
                for key, value in zip(node.keys, node.values)
            )
        if isinstance(node, ast.Call):
            function = node.func
            name = getattr(function, "id", getattr(function, "attr", ""))
            if name in COPIES:
                if node.args and self.environment(node.args[0], scope):
                    return True
                if isinstance(function, ast.Attribute) and name == "copy":
                    return self.environment(function.value, scope)
            if (
                name == "get"
                and isinstance(function, ast.Attribute)
                and self._kwargs(function.value)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ENVIRONMENT_KEYS
            ):
                return True
        if isinstance(node, ast.Subscript):
            key = node.slice.value if isinstance(node.slice, ast.Constant) else "*"
            if key in ENVIRONMENT_KEYS and self._kwargs(node.value):
                return True
            if isinstance(node.value, ast.Name):
                keys = scope.holders.get(node.value.id, set())
                return key in keys or "*" in keys
        return False

    def holder(self, node: ast.AST, scope: _Scope) -> bool:
        """Whether ``node``'s value may contain a whole environment."""
        if isinstance(node, ast.Name):
            return node.id in scope.holders
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return any(self._either(element, scope) for element in node.elts)
        if isinstance(node, ast.Dict):
            return any(self._either(value, scope) for value in node.values)
        return False

    def _either(self, node: ast.AST, scope: _Scope) -> bool:
        return self.environment(node, scope) or self.holder(node, scope)

    # -- binding ------------------------------------------------------------

    def _bind(self, function: ast.AST) -> _Scope:
        scope = _Scope()
        changed = True
        while changed:
            changed = False
            for node in ast.walk(function):
                changed |= self._bind_node(node, scope)
        return scope

    def _bind_node(self, node: ast.AST, scope: _Scope) -> bool:
        changed = False
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                return False
            for target in targets:
                changed |= self._bind_target(target, value, scope)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            if isinstance(node.iter, ast.Name):
                keys = scope.holders.get(node.iter.id, set())
                target = node.target
                if isinstance(target, ast.Name):
                    if "*" in keys and target.id not in scope.environments:
                        scope.environments.add(target.id)
                        changed = True
                    for key in keys:
                        if isinstance(key, tuple):
                            changed |= scope.hold(target.id, key[1])
                elif isinstance(target, ast.Tuple):
                    for position, element in enumerate(target.elts):
                        if (
                            isinstance(element, ast.Name)
                            and ("*" in keys or ("item", position) in keys)
                            and element.id not in scope.environments
                        ):
                            scope.environments.add(element.id)
                            changed = True
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            if isinstance(receiver, ast.Name):
                method = node.func.attr
                if method == "update":
                    if any(self.environment(arg, scope) for arg in node.args):
                        if receiver.id not in scope.environments:
                            scope.environments.add(receiver.id)
                            changed = True
                    for keyword in node.keywords:
                        if keyword.arg and self._either(keyword.value, scope):
                            changed |= scope.hold(receiver.id, keyword.arg)
                elif method in {"append", "add", "insert"}:
                    item = node.args[-1] if node.args else None
                    if isinstance(item, ast.Tuple):
                        for position, element in enumerate(item.elts):
                            if self._either(element, scope):
                                changed |= scope.hold(receiver.id, ("item", position))
                    elif item is not None and self._either(item, scope):
                        changed |= scope.hold(receiver.id, "*")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for position, keys in self.filled_parameters.get(node.func.id, {}).items():
                if position < len(node.args) and isinstance(
                    node.args[position], ast.Name
                ):
                    for key in keys:
                        changed |= scope.hold(node.args[position].id, key)
        return changed

    def _bind_target(self, target: ast.AST, value: ast.AST, scope: _Scope) -> bool:
        if isinstance(target, ast.Name):
            if self.environment(value, scope):
                if target.id not in scope.environments:
                    scope.environments.add(target.id)
                    return True
            elif self.holder(value, scope):
                return scope.hold(target.id, "*")
            return False
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and self._either(value, scope)
        ):
            key = target.slice.value if isinstance(target.slice, ast.Constant) else "*"
            return scope.hold(target.value.id, key)
        return False

    # -- what pytest prints -------------------------------------------------

    def printed(self, node: ast.AST) -> Iterator[ast.AST]:
        """The expressions pytest prints when an assert on ``node`` fails."""
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                yield from self.printed(value)
        elif isinstance(node, ast.UnaryOp):
            yield from self.printed(node.operand)
        elif isinstance(node, ast.BinOp):
            yield from self.printed(node.left)
            yield from self.printed(node.right)
        elif isinstance(node, ast.Compare):
            # ``x is not None`` fails only when x is None, so prints only None.
            if not (
                len(node.ops) == 1
                and isinstance(node.ops[0], ast.IsNot)
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value is None
            ):
                yield from self.printed(node.left)
            for comparator in node.comparators:
                yield from self.printed(comparator)
        elif isinstance(node, ast.Call):
            yield node
            yield from self.printed(node.func)
            for argument in node.args:
                yield from self.printed(argument)
            for keyword in node.keywords:
                yield from self.printed(keyword.value)
        elif isinstance(node, ast.Starred):
            yield from self.printed(node.value)
        elif isinstance(node, ast.Attribute):
            yield node
            yield from self.printed(node.value)
        else:
            yield node

    def leaks(self) -> list[int]:
        """Lines of asserts that may print a whole environment."""
        lines = []
        for function in self.functions:
            scope = self._bind(function)
            for node in ast.walk(function):
                if not isinstance(node, ast.Assert):
                    continue
                parts = [] if node.msg is None else [node.msg]
                # ``assert x`` fails only when x is empty, so prints nothing in it.
                if not isinstance(node.test, (ast.Name, ast.Subscript)):
                    parts.append(node.test)
                if any(
                    self._either(shown, scope)
                    for part in parts
                    for shown in self.printed(part)
                ):
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

from tests.environment_assertions import changed_names


def record_into(record):
    record["environ"] = dict(os.environ)


def test_leaks_restored_environment():
    before = dict(os.environ)
    os.environ["AAA_CHANGED"] = "1"
    assert dict(os.environ) == before


def test_leaks_membership():
    assert "AAA_MISSING" in os.environ


def test_leaks_lookup():
    assert os.environ.get("AAA_MISSING") == "x"


def test_leaks_copy_by_name():
    environment = os.environ.copy()
    assert "AAA_SENTINEL" not in environment


def test_leaks_record():
    record = {}
    record["environ"] = dict(os.environ)
    assert record == {}


def test_leaks_recorded_keyword():
    kwargs = {"env": {**os.environ, "EXTRA": "1"}}
    assert "AAA_SENTINEL" not in kwargs["env"]


def test_leaks_count():
    environments = []
    environments.append(dict(os.environ))
    assert len(environments) == 2


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


def test_comprehension():
    environment = dict(os.environ)
    assert not any(name.startswith("AAA_") for name in environment)


def test_count_first():
    environments = [dict(os.environ)]
    count = len(environments)
    assert count == 2


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
        if header := re.fullmatch(r"_{3,} (test_\w+) _{3,}", line):
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
    before = {"KEPT": "1", "CHANGED": "a", "REMOVED": "x"}
    after = {"KEPT": "1", "CHANGED": "b", "ADDED": "y"}
    assert changed_names(before, after) == ["ADDED", "CHANGED", "REMOVED"]


environments = st.dictionaries(
    st.sampled_from(["A", "B", "C", "D"]), st.sampled_from(["", "0", "1"])
)


@given(before=environments, after=environments)
def test_changed_names_is_empty_exactly_when_equal(
    before: dict[str, str], after: dict[str, str]
) -> None:
    changed = changed_names(before, after)
    assert (changed == []) == (before == after)
    assert changed == changed_names(after, before) == sorted(set(changed))
    assert set(changed) <= before.keys() | after.keys()
    for name in changed:
        assert before.get(name) != after.get(name)
