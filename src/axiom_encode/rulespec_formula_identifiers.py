"""Lexical references for fixture generation and copied RuleSpec context.

The engine distinguishes a named call from a bare variable with the same name.
This collector makes that distinction without evaluating or repairing formulas.
Compilation remains responsible for rejecting unknown functions and syntax.
"""

import ast
import re
import textwrap

# These are Expr::Call names accepted by the Rust RuleSpec formula lowerer.
_CALLABLES = frozenset(
    {
        "calendar_years_to_months",
        "ceil",
        "count_over_periods",
        "count_where",
        "date_add_days",
        "date_add_months",
        "date_add_years",
        "days_between",
        "exactly_one",
        "floor",
        "len",
        "max",
        "max_over_periods",
        "min",
        "sum",
        "sum_over_periods",
        "sum_top_n_over_periods",
        "sum_where",
    }
)
# Lexer keywords. In particular,
# count/all/any/in are not reserved variables in the Rust formula language.
_RESERVED = frozenset(
    {
        "False",
        "True",
        "amend",
        "and",
        "elif",
        "else",
        "entity",
        "false",
        "from",
        "if",
        "match",
        "not",
        "or",
        "to",
        "true",
    }
)
_NON_CODE = re.compile(r""""(?:\\.|[^"\\])*"|'(?:''|\\.|[^'\\])*'|\#[^\n]*""")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PERIOD_SYMBOLS = frozenset({"period_start", "period_end"})


def _period_references(formula: str, *, judgment: bool | None) -> set[str]:
    """Distinguish scalar date constants from judgment/relation input names.

    Python's expression/if syntax covers these local reference contexts without
    evaluating them. RuleSpec-only syntax (such as match arrows) falls back to
    retaining possible references; only the Rust compiler admits semantics.
    """
    try:
        tree = ast.parse(textwrap.dedent(formula).strip())
    except (SyntaxError, ValueError, RecursionError):
        return set(_PERIOD_SYMBOLS)
    references = set()

    def walk(node: ast.AST, is_judgment: bool) -> None:
        if isinstance(node, ast.Name):
            if is_judgment and node.id in _PERIOD_SYMBOLS:
                references.add(node.id)
        elif isinstance(node, ast.Module):
            for statement in node.body:
                walk(statement, is_judgment)
        elif isinstance(node, ast.Expr):
            walk(node.value, is_judgment)
        elif isinstance(node, (ast.If, ast.IfExp)):
            walk(node.test, True)
            for branch in (node.body, node.orelse):
                for item in branch if isinstance(branch, list) else [branch]:
                    walk(item, is_judgment)
        elif isinstance(node, ast.BoolOp):
            for item in node.values:
                walk(item, True)
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand, isinstance(node.op, ast.Not))
        elif isinstance(node, ast.Compare):
            for item in [node.left, *node.comparators]:
                walk(item, False)
        elif isinstance(node, ast.BinOp):
            walk(node.left, False)
            walk(node.right, False)
        elif isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if name not in _CALLABLES:
                walk(node.func, True)
            # Relation names, predicate arguments and selected relation fields
            # are references, not scalar expressions in the Rust lowerer.
            named_arguments = {"len", "sum", "count_where", "sum_where"}
            argument_judgment = (
                name == "exactly_one"
                or name in named_arguments
                or name not in _CALLABLES
            )
            for item in node.args:
                walk(item, argument_judgment)
            for item in node.keywords:
                walk(item.value, True)
        elif isinstance(node, ast.Subscript):
            walk(node.value, True)  # named parameter-table reference
            walk(node.slice, False)
        elif isinstance(node, ast.Attribute):
            if node.attr in _PERIOD_SYMBOLS:
                references.add(node.attr)
            walk(node.value, True)
        elif not isinstance(node, ast.Constant):
            # Unsupported syntax must not erase a possible input dependency.
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and item.id in _PERIOD_SYMBOLS:
                    references.add(item.id)
                elif isinstance(item, ast.Attribute) and item.attr in _PERIOD_SYMBOLS:
                    references.add(item.attr)

    try:
        walk(tree, judgment is not False)
    except RecursionError:
        return set(_PERIOD_SYMBOLS)
    return references


def formula_reference_identifiers(
    formula: str, *, judgment: bool | None = None
) -> set[str]:
    """Keep references and unknown calls, omitting known call occurrences.

    Full-rule callers provide the rule's judgment/scalar position. With no root
    context, a bare period symbol remains a possible input; nested scalar/date
    arguments still resolve their own context.
    """
    scrubbed = _NON_CODE.sub(lambda match: " " * len(match.group()), formula)
    references = set()
    period_references = None
    for match in _IDENTIFIER.finditer(scrubbed):
        name = match.group()
        if name in _RESERVED:
            continue
        if name in _CALLABLES and scrubbed[match.end() :].lstrip().startswith("("):
            continue
        if name in _PERIOD_SYMBOLS:
            if period_references is None:
                period_references = _period_references(formula, judgment=judgment)
            if name not in period_references:
                continue
        references.add(name)
    return references
