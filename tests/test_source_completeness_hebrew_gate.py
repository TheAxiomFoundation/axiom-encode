"""The wall's one gate: every Hebrew read in source_completeness sits behind it.

Review r10 broke the wall through a rule's ``source:`` annotation and review
r11 through a proof excerpt (#1882): each time a Hebrew helper ran on a rule
input that the per-site reasoning had not considered. Round 23 makes one gate
the only door. `analyze_complete_source_unit` opens it per candidate, from the
authoritative source and the deferral reasons (`_hebrew_reading`), and this
module checks over the AST of `source_completeness.py` that

- the Hebrew modules are imported once, as a module, and read only by
  attribute;
- every read of them, and of a name built from them or listed below, inside a
  function sits behind the gate: under ``if`` or an ``and`` or a conditional
  expression on `_hebrew_script()` (or a local bound only from it), or after
  an early return when it is closed;
- the gate's context variable is set only in `_hebrew_reading`, and only the
  listed doors open it;
- a cached function whose result depends on the gate keys its cache on it.
- the named dynamic namespace routes, direct imported-reader escapes and
  delayed calls below are rejected; imported Hebrew module dunder access and
  module-level bindings of Hebrew-call results are rejected too.

This is a syntactic regression guard, not a general Python data-flow proof.
Other reflection, package indirection, alias, door and cache routes identified
by r13 N2 remain outside it; Round 25's static audit records their absence.

A new unconditioned read fails here; so does each mutant below, which removes
one gate from the real module.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness

SOURCE = Path(source_completeness.__file__)
SRC_ROOT = SOURCE.parents[1]
HEBREW_MODULES = frozenset({"hebrew_source_text", "hebrew_negators", "hebrew_prefixes"})
GATE = "_hebrew_script"
GATE_VARIABLE = "_HEBREW_SCRIPT"
GATE_OPENER = "_hebrew_reading"

# Pure predicates that compute the gate itself; reading them opens nothing.
ALLOWLIST = {
    ("_candidate_holds_hebrew_script", "holds_hebrew_script"): (
        "computes the gate from the authoritative source and the deferral reasons"
    ),
}
# These four module-level declarations assemble the patterns selected behind
# the gate. Only their named constants/constructor may be read at import time;
# assigning a module, helper, container or callback is not pattern construction.
PATTERN_CONSTRUCTIONS = {
    "_HEBREW_EDITORIAL_OMISSION_ONLY": {
        "HEBREW_OUTLINE_LABEL",
        "EDITORIAL_OMISSION_WORDS",
    },
    "_HEBREW_MISSING_DEPENDENCY_LANGUAGE": {"MISSING_DEPENDENCY_LANGUAGE"},
    "_HEBREW_ADVERSATIVE_LANGUAGE": {"AdversativePattern", "ADVERSATIVE_LANGUAGE"},
    "_HEBREW_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE": {"RUNTIME_GAP_LANGUAGE"},
}
# This generator's sole Hebrew call computes the outer iterable immediately,
# and tuple consumes it synchronously. The checker also requires that exact
# shape; moving the call into a delayed element does not inherit the exception.
DELAYED_ALLOWLIST = {
    ("recognize_source_structure", "GeneratorExp", "hebrew_outline_markers"): (
        "the outer iterable is computed eagerly and tuple immediately consumes it"
    ),
}
# Module-level names whose values do not reference the Hebrew modules but
# exist only for the Hebrew readers; they are Hebrew names too.
HEBREW_ONLY_NAMES = {
    "_NEGATION_LANGUAGE": "the English and German negators the Hebrew reader takes",
    "_SOURCE_LINK_NEGATION_LANGUAGE": "the link negations of the cross-language copy",
    "_clause_masked_dependencies": "the citations the Hebrew reason view masks",
}
# Where the gate opens, each computing it with `_candidate_holds_hebrew_script`
# from that candidate's source and reasons.
DOORS = {
    ("harness/source_completeness.py", "analyze_complete_source_unit"),
    ("harness/coverage_index.py", "format_coverage_index"),
    ("cli.py", "_try_repair_generated_mixed_missing_deferred_outputs_for_apply"),
    ("harness/validator_pipeline.py", "_stated_conversion_ungrounded_literal_hint"),
    ("harness/same_case_execution.py", "_has_consumer"),
}
CACHE_DECORATORS = frozenset({"lru_cache", "cache"})


def _is_hebrew_module(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id in HEBREW_MODULES


def _hebrew_names(tree: ast.Module) -> set[str]:
    """Module-level names built from the Hebrew modules, transitively."""

    names = set(HEBREW_ONLY_NAMES)
    assignments = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            value = node.value
            if value is None:
                continue
            if not any(
                _is_hebrew_module(sub)
                or (isinstance(sub, ast.Name) and sub.id in names)
                for sub in ast.walk(value)
            ):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


class _Function:
    def __init__(self, node: ast.AST, parent: _Function | None) -> None:
        self.node = node
        self.parent = parent
        self.name = getattr(node, "name", "<lambda>")
        self.bindings: dict[str, list[ast.AST | None]] = defaultdict(list)


def _own_nodes(function: ast.AST):
    """The nodes of a function body, not of the functions nested in it."""

    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


class GateChecker:
    """Find every Hebrew read in a module that does not sit behind the gate."""

    def __init__(self, source: str) -> None:
        self.tree = ast.parse(source)
        self.hebrew_names = _hebrew_names(self.tree)
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.functions: dict[ast.AST, _Function] = {}
        self._collect(self.tree, None)
        self.imported_hebrew_modules = set(HEBREW_MODULES)
        self.imported_hebrew_helpers: set[str] = set()
        self.importlib_names = {"importlib"}
        self.sys_names = {"sys"}
        self.sys_module_names: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    if module.rsplit(".", 1)[-1] in HEBREW_MODULES:
                        self.imported_hebrew_helpers.add(name)
                    elif alias.name in HEBREW_MODULES:
                        self.imported_hebrew_modules.add(name)
                    if module.split(".", 1)[0] == "importlib":
                        self.importlib_names.add(name)
                    if module == "sys" and alias.name == "modules":
                        self.sys_module_names.add(name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".", 1)[0]
                    if alias.name.rsplit(".", 1)[-1] in HEBREW_MODULES:
                        self.imported_hebrew_modules.add(name)
                    if alias.name.split(".", 1)[0] == "importlib":
                        self.importlib_names.add(name)
                    if alias.name == "sys":
                        self.sys_names.add(name)

    def _collect(self, node: ast.AST, parent: _Function | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                function = _Function(child, parent)
                self.functions[child] = function
                arguments = child.args
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                    *(a for a in (arguments.vararg, arguments.kwarg) if a),
                ):
                    function.bindings[argument.arg].append(None)
                for sub in _own_nodes(child):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            for name in ast.walk(target):
                                if isinstance(name, ast.Name):
                                    value = sub.value if target is name else None
                                    function.bindings[name.id].append(value)
                    elif isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
                        target = sub.target
                        if isinstance(target, ast.Name):
                            value = (
                                sub.value
                                if isinstance(sub, (ast.AnnAssign, ast.NamedExpr))
                                else None
                            )
                            function.bindings[target.id].append(value)
                    elif isinstance(sub, (ast.For, ast.AsyncFor, ast.comprehension)):
                        for name in ast.walk(sub.target):
                            if isinstance(name, ast.Name):
                                function.bindings[name.id].append(None)
                    elif isinstance(sub, (ast.With, ast.AsyncWith)):
                        for item in sub.items:
                            if item.optional_vars is not None:
                                for name in ast.walk(item.optional_vars):
                                    if isinstance(name, ast.Name):
                                        function.bindings[name.id].append(None)
                    elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function.bindings[sub.name].append(None)
                self._collect(child, function)
            else:
                self._collect(child, parent)

    # -- the gate ---------------------------------------------------------

    def _gate_local(self, name: str, function: _Function | None, seen=()) -> bool:
        while function is not None:
            if name in function.bindings:
                values = function.bindings[name]
                return bool(values) and all(
                    value is not None
                    and self.asserts_gate(value, function, (*seen, name))
                    for value in values
                )
            function = function.parent
        return False

    def asserts_gate(self, node: ast.AST, function: _Function | None, seen=()) -> bool:
        """Whether ``node`` is true only when the gate is open."""

        if isinstance(node, ast.Call):
            return (
                isinstance(node.func, ast.Name)
                and node.func.id == GATE
                and not node.args
                and not node.keywords
            )
        if isinstance(node, ast.Name):
            return node.id not in seen and self._gate_local(node.id, function, seen)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return any(
                self.asserts_gate(value, function, seen) for value in node.values
            )
        return False

    def asserts_closed(self, node: ast.AST, function: _Function | None) -> bool:
        """Whether ``node`` is true only when the gate is closed."""

        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.Not)
            and self.asserts_gate(node.operand, function)
        )

    def _exits(self, statements: list[ast.stmt]) -> bool:
        return bool(statements) and isinstance(
            statements[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break)
        )

    def _enclosing_function(self, node: ast.AST) -> _Function | None:
        while node in self.parents:
            node = self.parents[node]
            if node in self.functions:
                return self.functions[node]
        return None

    def gated(self, node: ast.AST) -> bool:
        """Whether ``node`` runs only when the gate is open."""

        function = self._enclosing_function(node)
        if function is None:
            return False
        child = node
        while child is not function.node:
            parent = self.parents[child]
            if isinstance(parent, (ast.If, ast.While)):
                if child in parent.body and self.asserts_gate(parent.test, function):
                    return True
                if child in parent.orelse and self.asserts_closed(
                    parent.test, function
                ):
                    return True
            elif isinstance(parent, ast.IfExp):
                if child is parent.body and self.asserts_gate(parent.test, function):
                    return True
                if child is parent.orelse and self.asserts_closed(
                    parent.test, function
                ):
                    return True
            elif isinstance(parent, ast.BoolOp):
                index = parent.values.index(child)
                earlier = parent.values[:index]
                if isinstance(parent.op, ast.And) and any(
                    self.asserts_gate(value, function) for value in earlier
                ):
                    return True
                if isinstance(parent.op, ast.Or) and any(
                    self.asserts_closed(value, function) for value in earlier
                ):
                    return True
            for field in ("body", "orelse", "finalbody"):
                statements = getattr(parent, field, None)
                if isinstance(statements, list) and child in statements:
                    for statement in statements[: statements.index(child)]:
                        if (
                            isinstance(statement, ast.If)
                            and not statement.orelse
                            and self.asserts_closed(statement.test, function)
                            and self._exits(statement.body)
                        ):
                            return True
            child = parent
        return False

    # -- the reads ----------------------------------------------------------

    def hebrew_reads(self):
        """Every read of a Hebrew module or Hebrew name inside a function."""

        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in self.imported_hebrew_modules
            ):
                yield node, node.attr
            elif (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in self.hebrew_names | self.imported_hebrew_helpers
            ):
                yield node, node.id

    def _violation(self, node: ast.AST, message: str) -> str:
        function = self._enclosing_function(node)
        owner = function.name if function else "module level"
        return f"{owner} line {node.lineno}: {message}"

    def _attribute_chain(self, node: ast.AST) -> ast.AST:
        """The complete attribute expression rooted at a loaded name."""

        while (
            isinstance(self.parents.get(node), ast.Attribute)
            and self.parents[node].value is node
        ):
            node = self.parents[node]
        return node

    def _called_directly(self, node: ast.AST) -> bool:
        parent = self.parents.get(node)
        return isinstance(parent, ast.Call) and parent.func is node

    def _pattern_construction(self, node: ast.Name) -> bool:
        attribute = self.parents.get(node)
        if not isinstance(attribute, ast.Attribute) or attribute.value is not node:
            return False
        statement: ast.AST = node
        while statement in self.parents and self.parents[statement] is not self.tree:
            statement = self.parents[statement]
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            return False
        targets = (
            statement.targets
            if isinstance(statement, ast.Assign)
            else [statement.target]
        )
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return False
        allowed = PATTERN_CONSTRUCTIONS.get(targets[0].id, set())
        if attribute.attr not in allowed:
            return False
        # Constants are consumed as string expressions; the one constructor is
        # invoked directly. Neither shape may export a reader for later use.
        if attribute.attr == "AdversativePattern":
            return (
                self._called_directly(attribute)
                and statement.value is self.parents[attribute]
            )
        return isinstance(self.parents.get(attribute), (ast.BinOp, ast.FormattedValue))

    def restricted_access_violations(self) -> list[str]:
        """Reject the listed namespace routes and direct imported-reader escapes."""

        found = []
        forbidden = {"globals", "vars", "locals", "__import__"}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                if (module or "").split(".", 1)[0] == "importlib" or any(
                    alias.name.split(".", 1)[0] == "importlib" for alias in node.names
                ):
                    found.append(self._violation(node, "importlib is forbidden"))
                if module == "builtins" and any(
                    alias.name in forbidden
                    or (
                        alias.name == "getattr"
                        and alias.asname not in {None, "getattr"}
                    )
                    for alias in node.names
                ):
                    found.append(
                        self._violation(node, "imported dynamic namespace alias")
                    )
                if any(alias.name == GATE_VARIABLE for alias in node.names):
                    found.append(
                        self._violation(node, "imported gate ContextVar alias")
                    )
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in forbidden | self.importlib_names | self.sys_module_names:
                    found.append(
                        self._violation(node, f"dynamic namespace access: {node.id}")
                    )
                if node.id in self.sys_names and not (
                    isinstance(self.parents.get(node), ast.Attribute)
                    and self.parents[node].value is node
                ):
                    found.append(self._violation(node, "escaped sys module"))
                if node.id == "getattr":
                    call = self.parents.get(node)
                    if not (
                        isinstance(call, ast.Call)
                        and call.func is node
                        and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant)
                        and isinstance(call.args[1].value, str)
                        and call.args[1].value
                        not in forbidden | {"importlib", "getattr"}
                    ):
                        found.append(
                            self._violation(
                                node, "getattr requires a permitted literal name"
                            )
                        )
                if (
                    node.id
                    in self.imported_hebrew_modules | self.imported_hebrew_helpers
                ):
                    expression = self._attribute_chain(node)
                    is_module = node.id in self.imported_hebrew_modules
                    attribute = expression
                    while isinstance(attribute, ast.Attribute):
                        if (
                            is_module
                            and attribute.attr.startswith("__")
                            and attribute.attr.endswith("__")
                        ):
                            found.append(
                                self._violation(
                                    attribute,
                                    "Hebrew module dunder access is forbidden",
                                )
                            )
                        attribute = attribute.value
                    if not (
                        (expression is not node or not is_module)
                        and self._called_directly(expression)
                    ) and not self._pattern_construction(node):
                        found.append(
                            self._violation(node, f"escaped Hebrew reader: {node.id}")
                        )
                if node.id == GATE_VARIABLE:
                    expression = self.parents.get(node)
                    function = self._enclosing_function(node)
                    if not (
                        isinstance(expression, ast.Attribute)
                        and expression.value is node
                        and self._called_directly(expression)
                        and function is not None
                        and function.parent is None
                        and (
                            (
                                function.name == GATE_OPENER
                                and expression.attr in {"set", "reset"}
                            )
                            or (function.name == GATE and expression.attr == "get")
                        )
                    ):
                        found.append(
                            self._violation(node, "escaped gate ContextVar or setter")
                        )
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "modules"
                and isinstance(node.value, ast.Name)
                and node.value.id in self.sys_names
            ):
                found.append(self._violation(node, "sys.modules is forbidden"))
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden | {"importlib"}:
                    found.append(
                        self._violation(node, f"dynamic namespace access: {node.attr}")
                    )
                if node.attr == "getattr":
                    call = self.parents.get(node)
                    if not (
                        isinstance(call, ast.Call)
                        and call.func is node
                        and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant)
                        and isinstance(call.args[1].value, str)
                        and call.args[1].value
                        not in forbidden | {"importlib", "getattr"}
                    ):
                        found.append(
                            self._violation(
                                node, "getattr requires a permitted literal name"
                            )
                        )
        return found

    def _hebrew_call_name(self, call: ast.Call) -> str | None:
        expression = call.func
        while isinstance(expression, ast.Attribute):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id in self.imported_hebrew_modules
            ):
                return expression.attr
            expression = expression.value
        if isinstance(expression, ast.Name) and expression.id in (
            self.imported_hebrew_helpers | self.hebrew_names
        ):
            return expression.id
        return None

    def module_bound_call_violations(self) -> list[str]:
        """A module-level call result must not retain an imported Hebrew reader."""

        found = []
        for call in ast.walk(self.tree):
            if not isinstance(call, ast.Call):
                continue
            name = self._hebrew_call_name(call)
            if name is None or self._enclosing_function(call) is not None:
                continue
            expression = call.func
            while isinstance(expression, ast.Attribute):
                expression = expression.value
            if isinstance(expression, ast.Name) and self._pattern_construction(
                expression
            ):
                continue
            statement: ast.AST = call
            while statement in self.parents:
                statement = self.parents[statement]
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    found.append(
                        self._violation(call, f"module-level bound Hebrew call: {name}")
                    )
                    break
        return found

    def _allowed_delayed_call(self, node: ast.AST, call: ast.Call, name: str) -> bool:
        function = self._enclosing_function(node)
        key = (function.name if function else "", type(node).__name__, name)
        if key not in DELAYED_ALLOWLIST or not isinstance(node, ast.GeneratorExp):
            return False
        parent = self.parents.get(node)
        return (
            len(node.generators) == 1
            and node.generators[0].iter is call
            and isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "tuple"
            and parent.args == [node]
            and not parent.keywords
        )

    def delayed_read_violations(self) -> list[str]:
        """Lexical gates do not authorize a reader executed after they close."""

        found = []
        for call in ast.walk(self.tree):
            if not isinstance(call, ast.Call):
                continue
            name = self._hebrew_call_name(call)
            if name is None:
                continue
            node: ast.AST = call
            while node in self.parents:
                node = self.parents[node]
                delayed = isinstance(
                    node, (ast.Yield, ast.YieldFrom, ast.GeneratorExp, ast.Lambda)
                )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function = self.functions[node]
                    delayed = function.parent is not None or any(
                        isinstance(sub, (ast.Yield, ast.YieldFrom))
                        for sub in _own_nodes(node)
                    )
                if delayed and not self._allowed_delayed_call(node, call, name):
                    found.append(
                        self._violation(
                            call,
                            f"delayed Hebrew call in {type(node).__name__}: {name}",
                        )
                    )
        return found

    def violations(self) -> list[str]:
        found = (
            self.restricted_access_violations()
            + self.module_bound_call_violations()
            + self.delayed_read_violations()
        )
        for node, name in self.hebrew_reads():
            function = self._enclosing_function(node)
            if function is None:
                # Module level: only pattern construction may read them.
                statement = node
                while self.parents.get(statement) is not self.tree:
                    statement = self.parents[statement]
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    found.append(f"module level line {node.lineno}: {name}")
                continue
            outer = function
            while outer.parent is not None:
                outer = outer.parent
            if (outer.name, name) in ALLOWLIST or (function.name, name) in ALLOWLIST:
                continue
            if not self.gated(node):
                found.append(f"{function.name} line {node.lineno}: {name}")
        return found

    # -- the closure and caches -----------------------------------------------

    def gate_dependent_functions(self) -> set[str]:
        """Top-level functions whose result can depend on the gate."""

        top = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        calls: dict[str, set[str]] = {}
        for name, node in top.items():
            calls[name] = {
                sub.id
                for sub in ast.walk(node)
                if isinstance(sub, ast.Name) and sub.id in top
            }
        dependent = {GATE}
        changed = True
        while changed:
            changed = False
            for name, callees in calls.items():
                if name not in dependent and callees & dependent:
                    dependent.add(name)
                    changed = True
        return dependent


def _decorator_name(decorator: ast.AST) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    return getattr(decorator, "id", "")


@pytest.fixture(scope="module")
def checker() -> GateChecker:
    return GateChecker(SOURCE.read_text())


def test_every_hebrew_read_sits_behind_the_gate(checker):
    assert checker.violations() == []


def test_the_hebrew_modules_are_imported_once_as_a_module(checker):
    imports = []
    for node in ast.walk(checker.tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if module.rsplit(".", 1)[-1] in HEBREW_MODULES or (
                    alias.name in HEBREW_MODULES
                ):
                    imports.append((module, alias.name, alias.asname))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.rsplit(".", 1)[-1] in HEBREW_MODULES:
                    imports.append(("", alias.name, alias.asname))
    assert imports == [("axiom_encode.harness", "hebrew_source_text", None)]
    # No alias of the module object hides a read from the checker.
    for node in ast.walk(checker.tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            assert not (value is not None and _is_hebrew_module(value)), node.lineno
        if isinstance(node, ast.Call):
            for argument in (*node.args, *(k.value for k in node.keywords)):
                assert not _is_hebrew_module(argument), node.lineno


def test_no_other_shared_module_imports_the_hebrew_modules():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path.stem in HEBREW_MODULES or path == SOURCE:
            continue
        text = path.read_text()
        for module in HEBREW_MODULES:
            assert not re.search(rf"\bimport\b[^\n]*\b{module}\b", text), path
            assert not re.search(rf"\bfrom\b[^\n]*\b{module}\b", text), path


def test_the_allowlist_names_only_what_it_allows(checker):
    reads = {
        (checker._enclosing_function(node).name, name)
        for node, name in checker.hebrew_reads()
        if checker._enclosing_function(node) is not None
    }
    assert set(ALLOWLIST) <= reads
    for name in HEBREW_ONLY_NAMES:
        assert name in checker.hebrew_names


def test_the_gate_is_set_only_by_its_opener(checker):
    openers = [
        node
        for node in checker.tree.body
        if isinstance(node, ast.FunctionDef) and node.name == GATE_OPENER
    ]
    assert len(openers) == 1
    assert [ast.unparse(node) for node in openers[0].decorator_list] == [
        "contextlib.contextmanager"
    ]
    setters = []
    for node in ast.walk(checker.tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == GATE_VARIABLE
            and node.attr in {"set", "reset"}
        ):
            setters.append(checker._enclosing_function(node).name)
    assert setters == [GATE_OPENER, GATE_OPENER]
    # The opener computes the gate from the candidate, and nothing else does.
    computes = [
        checker._enclosing_function(node).name
        for node in ast.walk(checker.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_candidate_holds_hebrew_script"
    ]
    assert computes == [GATE_OPENER]
    # (hebrew_source_text has a regex of the same name; only a ``set`` of the
    # gate's variable would open it.)
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if path != SOURCE:
            text = path.read_text()
            assert not re.search(rf"\b{GATE_VARIABLE}\.(?:set|reset)\b", text), path
            assert "_candidate_holds_hebrew_script" not in text, path


def test_only_the_listed_doors_open_the_gate():
    doors = set()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == GATE_OPENER)
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == GATE_OPENER
                    )
                )
            ):
                continue
            # Every door passes a candidate payload and its source text.
            assert len(node.args) == 2 and not node.keywords, (path, node.lineno)
            owner = node
            while owner in parents and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents[owner]
            doors.add((str(path.relative_to(SRC_ROOT)), owner.name))
    assert doors == DOORS


def test_a_cached_function_keys_its_cache_on_the_gate(checker):
    dependent = checker.gate_dependent_functions()
    cached = []
    for node in ast.walk(checker.tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_decorator_name(d) in CACHE_DECORATORS for d in node.decorator_list):
            continue
        callees = {
            sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)
        } & dependent
        if not callees:
            continue
        cached.append(node.name)
        keyword_only = {argument.arg for argument in node.args.kwonlyargs}
        assert "hebrew_script" in keyword_only, node.name
        index = [a.arg for a in node.args.kwonlyargs].index("hebrew_script")
        assert node.args.kw_defaults[index] is None, f"{node.name} gives it a default"
    assert cached == ["_cached_source_imperative_arithmetic_operations"]
    # Each call passes the gate itself.
    for node in ast.walk(checker.tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in cached
        ):
            (keyword,) = [k for k in node.keywords if k.arg == "hebrew_script"]
            assert checker.asserts_gate(
                keyword.value, checker._enclosing_function(node)
            ), node.lineno


def test_shared_patterns_keep_mains_literals_behind_selectors():
    for name in (
        "_EDITORIAL_OMISSION_ONLY",
        "_MISSING_DEPENDENCY_LANGUAGE",
        "_ADVERSATIVE_LANGUAGE",
        "_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE",
    ):
        main = getattr(source_completeness, name)
        hebrew = getattr(source_completeness, f"_HEBREW_{name.lstrip('_')}")
        assert isinstance(main, re.Pattern)
        assert hebrew.pattern.startswith(
            main.pattern.removesuffix(r"\s*[.;]?\s*$")[:20]
        )
        assert not re.search("[֐-׿]", main.pattern), name


# -- Mutants: each removes one gate from the real module ------------------------

MUTANTS = {
    "_strip_source_clause_marker": (
        "        hebrew_source_text.strip_leading_outline_chain(text)\n"
        "        if _hebrew_script()\n"
        "        else None\n",
        "        hebrew_source_text.strip_leading_outline_chain(text)\n",
    ),
    "_reason_names_source_bound_runtime_gap": (
        "    # past (#1882). `_EXPLICIT_COMPUTATION` stays as it is.\n"
        "    if _hebrew_script() and hebrew_source_text.holds_hebrew_script(\n",
        "    # past (#1882). `_EXPLICIT_COMPUTATION` stays as it is.\n"
        "    if hebrew_source_text.holds_hebrew_script(\n",
    ),
    "_editorial_omission_only": (
        "        _HEBREW_EDITORIAL_OMISSION_ONLY\n"
        "        if _hebrew_script()\n"
        "        else _EDITORIAL_OMISSION_ONLY\n",
        "        _HEBREW_EDITORIAL_OMISSION_ONLY\n",
    ),
    "_source_link_scope_has_unreset_negation": (
        "    if not _hebrew_script():\n        return False\n"
        "    cross_language_scope = ",
        "    cross_language_scope = ",
    ),
    "_paths_from_source_reference": (
        "    if _hebrew_script() and hebrew_script:\n        paths.update(",
        "    if hebrew_script:\n        paths.update(",
    ),
    "_branch_citation": (
        "        and _hebrew_script()\n"
        "        and hebrew_source_text.HEBREW_LETTER.search(branch.text)\n",
        "        and hebrew_source_text.HEBREW_LETTER.search(branch.text)\n",
    ),
    # Sol r12's existing-reader mutation retains the gated expression, so all
    # six original mutation checks still run on the bypassed source.
    "_strip_source_clause_marker[globals]": (
        "    hebrew_body = (\n"
        "        hebrew_source_text.strip_leading_outline_chain(text)\n"
        "        if _hebrew_script()\n"
        "        else None\n"
        "    )",
        "    hebrew_body = (\n"
        '        globals()["hebrew_source_text"].strip_leading_outline_chain(text) or (\n'
        "        hebrew_source_text.strip_leading_outline_chain(text)\n"
        "        if _hebrew_script()\n"
        "        else None\n"
        "        )\n"
        "    )",
    ),
    "_review_gate_escape": (
        "def _strip_source_clause_marker(text: str) -> str:",
        "def _review_gate_escape(text):\n"
        "    if _hebrew_script():\n"
        "        return (hebrew_source_text.fold_marks(piece) for piece in [text])\n"
        "    return ()\n\n\n"
        "def _strip_source_clause_marker(text: str) -> str:",
    ),
    "module level[setter alias]": (
        "def _hebrew_script() -> bool:",
        "_review_gate_alias = _HEBREW_SCRIPT\n"
        "_review_gate_alias.set(True)\n\n\n"
        "def _hebrew_script() -> bool:",
    ),
}


@pytest.mark.parametrize("function", sorted(MUTANTS))
def test_removing_a_gate_fails_the_check(function):
    original, mutated = MUTANTS[function]
    source = SOURCE.read_text()
    assert source.count(original) == 1, function
    violations = GateChecker(source.replace(original, mutated)).violations()
    owner = function.split("[", 1)[0]
    assert any(v.startswith(f"{owner} ") for v in violations), violations


def test_r20_module_level_dunder_tuple_binding_fails_the_check():
    source = SOURCE.read_text()
    imported = "from axiom_encode.harness import hebrew_source_text\n"
    assert source.count(imported) == 1
    source = source.replace(
        imported,
        imported
        + '_REVIEW_STRIP, _REVIEW_UNUSED = hebrew_source_text.__getattribute__("strip_leading_outline_chain"), None\n',
        1,
    )
    anchor = (
        '    """\n\n    hebrew_body = (\n'
        "        hebrew_source_text.strip_leading_outline_chain(text)\n"
    )
    assert source.count(anchor) == 1
    source = source.replace(
        anchor,
        '    """\n\n    _REVIEW_STRIP(text)\n    hebrew_body = (\n'
        "        hebrew_source_text.strip_leading_outline_chain(text)\n",
        1,
    )
    violations = GateChecker(source).violations()
    assert any("Hebrew module dunder access" in issue for issue in violations)
    assert any("module-level bound Hebrew call" in issue for issue in violations)


@pytest.mark.parametrize(
    "body",
    [
        'globals()["hebrew_source_text"].fold_marks(text)',
        'vars(hebrew_source_text)["fold_marks"](text)',
        'locals()["hebrew_source_text"].fold_marks(text)',
        '__import__("axiom_encode.harness.hebrew_source_text")',
        'importlib.import_module("axiom_encode.harness.hebrew_source_text")',
        'sys.modules["axiom_encode.harness.hebrew_source_text"]',
        "builtins.globals()",
        "builtins.vars(text)",
        "builtins.locals()",
        'builtins.__import__("axiom_encode.harness.hebrew_source_text")',
        "getattr(text, name)",
        "builtins.getattr(text, name)",
        'getattr(builtins, "globals")()["hebrew_source_text"].fold_marks(text)',
        'getattr(builtins, "__import__")("axiom_encode.harness.hebrew_source_text", fromlist=["fold_marks"]).fold_marks(text)',
        'builtins.getattr(builtins, "locals")()',
        'getattr(builtins, "vars")()',
        'getattr(builtins, "getattr")(text, name)',
        "getter = getattr",
        "reader = hebrew_source_text.fold_marks",
        "use(hebrew_source_text.fold_marks)",
        "readers = [hebrew_source_text.fold_marks]",
        "reader = hebrew_source_text.HEBREW_LETTER.search",
        "variable = _HEBREW_SCRIPT",
        "setter = _HEBREW_SCRIPT.set",
        "_HEBREW_SCRIPT.set(True)",
        "_HEBREW_SCRIPT.reset(token)",
        "return lambda: hebrew_source_text.fold_marks(text)",
        "return (hebrew_source_text.fold_marks(piece) for piece in [text])",
        "yield hebrew_source_text.fold_marks(text)",
        "yield from hebrew_source_text.fold_marks(text)",
        "def later():\n            return hebrew_source_text.fold_marks(text)",
    ],
)
def test_a_lexical_gate_does_not_authorize_escaped_or_delayed_reads(body):
    source = (
        "def f(text, name, token):\n    if _hebrew_script():\n        " + body + "\n"
    )
    assert GateChecker(source).violations(), body


@pytest.mark.parametrize(
    "source",
    [
        "import importlib as loader\n",
        "from importlib import import_module as load\n",
        "from builtins import globals as reflect\ndef f():\n    return reflect()\n",
        "from builtins import __import__ as load\n",
        "from builtins import getattr as read\ndef f(text, name):\n    return read(text, name)\n",
        "from axiom_encode.harness.source_completeness import _HEBREW_SCRIPT as gate\n"
        "gate.set(True)\n",
        "import sys as system\ndef f():\n    return system.modules\n",
        "import sys\nsystem = sys\nsystem.modules\n",
        "from sys import modules as loaded\ndef f():\n    return loaded\n",
        "from axiom_encode.harness.hebrew_source_text import fold_marks as fold\n"
        "def f(text):\n    if _hebrew_script():\n        return fold\n",
        "MODULES = [hebrew_source_text]\n",
        "READER = hebrew_source_text.fold_marks\n",
        "class Reader:\n    module = hebrew_source_text\n",
        "def f(text, reader=hebrew_source_text.fold_marks):\n    return reader(text)\n",
        'VALUE = hebrew_source_text.fold_marks("text")\n',
        'VALUE: str = hebrew_source_text.fold_marks("text")\n',
        'VALUE, UNUSED = hebrew_source_text.fold_marks("text"), None\n',
        'VALUE = [hebrew_source_text.fold_marks("text")]\n',
        'VALUE = (BOUND := hebrew_source_text.fold_marks("text"))\n',
        "from axiom_encode.harness import hebrew_source_text as reader\n"
        'VALUE = reader.fold_marks("text")\n',
        "from axiom_encode.harness.hebrew_source_text import fold_marks as fold\n"
        'VALUE = fold("text")\n',
    ],
)
def test_module_construction_cannot_hide_a_reader_or_dynamic_namespace(source):
    assert GateChecker(source).violations()


@pytest.mark.parametrize("attribute", ["__getattribute__", "__dict__", "__class__"])
def test_a_lexical_gate_does_not_authorize_hebrew_module_dunder_access(attribute):
    source = (
        "def f(text):\n    if _hebrew_script():\n"
        f"        return hebrew_source_text.{attribute}(text)\n"
    )
    assert any(
        "Hebrew module dunder access" in issue
        for issue in GateChecker(source).violations()
    )


def test_a_named_eager_generator_exception_cannot_cover_a_delayed_element():
    source = (
        "def recognize_source_structure(text):\n"
        "    if _hebrew_script():\n"
        "        return tuple(hebrew_source_text.hebrew_outline_markers(piece) "
        "for piece in [text])\n"
    )
    assert GateChecker(source).delayed_read_violations()


def test_a_local_bound_from_anything_else_is_no_gate():
    checker = GateChecker(
        "def f(text, gate):\n"
        "    hebrew_script = holds(text)\n"
        "    if hebrew_script:\n"
        "        return hebrew_source_text.fold_marks(text)\n"
        "    if gate:\n"
        "        return hebrew_source_text.fold_marks(text)\n"
        "    ok = _hebrew_script()\n"
        "    if ok and text:\n"
        "        return hebrew_source_text.fold_marks(text)\n"
    )
    assert checker.violations() == ["f line 4: fold_marks", "f line 6: fold_marks"]
