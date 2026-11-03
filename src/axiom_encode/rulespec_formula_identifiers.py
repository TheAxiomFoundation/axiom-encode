"""Lexical references for fixture generation and copied RuleSpec context.

The engine distinguishes a named call from a bare variable with the same name.
This collector makes that distinction without evaluating or repairing formulas.
Compilation remains responsible for rejecting unknown functions and syntax.
"""

import re

# These are Expr::Call names accepted by the Rust RuleSpec formula lowerer,
# plus abs and round, which the validator's builtin list has always treated
# as calls rather than inputs.
_CALLABLES = frozenset(
    {
        "abs",
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
        "only_related",
        "round",
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


def formula_reference_identifiers(
    formula: str, *, judgment: bool | None = None
) -> set[str]:
    """Keep references and unknown calls, omitting known call occurrences.

    A bare ``period_start`` or ``period_end`` is the engine's date for the
    period being computed, in every position: the engine refuses to test one
    as a judgment, so it is never a factual input. Only a field access such as
    ``members.period_start`` names a reference. ``judgment`` is accepted for
    callers that know the rule's dtype; no current rule depends on it.
    """
    del judgment
    scrubbed = _NON_CODE.sub(lambda match: " " * len(match.group()), formula)
    references = set()
    for match in _IDENTIFIER.finditer(scrubbed):
        name = match.group()
        if name in _RESERVED:
            continue
        if name in _CALLABLES and scrubbed[match.end() :].lstrip().startswith("("):
            continue
        if name in _PERIOD_SYMBOLS and not scrubbed[: match.start()].rstrip().endswith(
            "."
        ):
            continue
        references.add(name)
    return references


def formula_calls_lifetime_reduction(formula: str) -> bool:
    """Detect actual lifetime calls for conservative fixture routing.

    This is not formula admission: the Rust compiler still validates syntax and
    semantics. Bare facts, quoted text, and comments cannot make a scalar output
    look like a lifetime calculation.
    """
    scrubbed = _NON_CODE.sub(lambda match: " " * len(match.group()), formula)
    return any(
        match.group()
        in {
            "sum_over_periods",
            "max_over_periods",
            "count_over_periods",
            "sum_top_n_over_periods",
        }
        and scrubbed[match.end() :].lstrip().startswith("(")
        for match in _IDENTIFIER.finditer(scrubbed)
    )
