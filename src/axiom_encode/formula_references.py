"""Read formula references without treating relation terms as input names.

The grammar port mirrors the engine; this module follows its lowering positions
while retaining declaration names for the existing source validators.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from axiom_encode.formula_grammar import (
    Expr,
    FormulaSyntaxError,
    parse_formula,
    tokenise,
)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_QUOTED = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
_TERMS = frozenset({"anchor", "candidate"})
_KEYWORDS = frozenset(
    {
        "and",
        "or",
        "not",
        "if",
        "elif",
        "else",
        "match",
        "entity",
        "amend",
        "from",
        "to",
        "true",
        "True",
        "false",
        "False",
        "period_start",
        "period_end",
    }
)
# Keep calls aligned with the engine's formula.rs BUILTIN_FUNCTIONS contract.
_FUNCTIONS = frozenset(
    {
        "max",
        "min",
        "ceil",
        "floor",
        "days_between",
        "date_add_days",
        "date_add_months",
        "date_add_years",
        "sum_over_periods",
        "max_over_periods",
        "count_over_periods",
        "sum_top_n_over_periods",
        "len",
        "sum",
        "only",
        "input_for_year",
        "only_related",
        "count_where",
        "sum_where",
        "min_where",
        "max_where",
        "exactly_one",
        "exists",
    }
)
FORMULA_BUILTINS = _KEYWORDS | _FUNCTIONS


def rule_uses_relation_predicate(rule: object) -> bool:
    """Keep the established filtered-entity scanners on their scalar route.

    Only a complete legacy destination (entity plus member relation) qualifies.
    Role syntax still opts into predicate analysis even on that legacy shape;
    malformed/new declarations never gain a parse-failure fallback here.
    """

    if not isinstance(rule, Mapping) or rule.get("kind") != "derived_relation":
        return False
    config = rule.get("derived_relation")
    if (
        not isinstance(config, Mapping)
        or "source_path" in config
        or not all(
            isinstance(config.get(key), str) and config[key].strip()
            for key in ("entity", "member_relation")
        )
    ):
        return True
    versions = rule.get("versions")
    for version in versions if isinstance(versions, list) else ():
        if not isinstance(version, Mapping) or not isinstance(
            version.get("formula"), str
        ):
            continue
        formula = version["formula"]
        try:
            tree = parse_formula(formula)
        except FormulaSyntaxError:
            # Classification needs no guessed read inventory. Lexing alone can
            # prove the absence of reserved role spellings in a legacy formula
            # such as an unsupported bare date; its existing validator refuses
            # the date while the historical name scan still sees its input.
            try:
                tokens = tokenise(formula)
            except FormulaSyntaxError:
                return True
            if any(
                token.kind == "ident" and token.value in _TERMS | {"exists"}
                for token in tokens
            ):
                return True
            if any(
                token.kind in {"ident", "path"}
                and token.value not in _FUNCTIONS
                and following.kind == "("
                for token, following in zip(tokens, tokens[1:])
            ):
                return True
            continue
        pending = [tree]
        while pending:
            expr = pending.pop()
            if (
                expr.kind == "var"
                and expr.value in _TERMS
                or expr.kind == "call"
                and (expr.value == "exists" or expr.value not in _FUNCTIONS)
            ):
                return True
            pending.extend(expr.children)
            for pattern, value in expr.cases:
                pending.extend((pattern, value))
            if expr.default is not None:
                pending.append(expr.default)
    return False


@dataclass(frozen=True)
class RelationPredicateReferences:
    """Separate declaration reads, relation reads and local entity terms."""

    identifiers: frozenset[str]
    atom_relations: frozenset[str]
    bound_variables: frozenset[str]
    out_of_scope_variables: frozenset[str]
    unbound_atom_terms: frozenset[str]
    qualified_reads: frozenset[str]
    relation_arguments: frozenset[str]
    parse_error: str | None = None
    parse_error_position: tuple[int, int] | None = None
    fallback_used: bool = False
    analysis_errors: tuple[str, ...] = ()
    input_reads: frozenset[str] = frozenset()
    rule_reads: frozenset[str] = frozenset()
    parameter_reads: frozenset[str] = frozenset()


def analyze_relation_predicate(
    formula: str,
    *,
    predicate_terms: bool = True,
    lowering_entry: Literal["derived_relation", "rule"] = "derived_relation",
    relation_names: frozenset[str] | None = None,
    rule_names: frozenset[str] = frozenset(),
    parameter_names: frozenset[str] = frozenset(),
) -> RelationPredicateReferences:
    """Walk parsed expressions in the positions that engine lowering reads.

    ``lowering_entry`` selects the engine's relation namespace independently of
    the legacy scanner's ``predicate_terms`` dispatch. Ordinary rule lowering
    never resolves a bare judgment name as a relation.
    Optional declaration inventories resolve the engine's three namespaces.
    Without them, unknown value names remain input candidates and unknown calls
    remain visible as atoms, so source validators can diagnose declarations.
    A predicate parse failure never invents references. Scalar callers retain
    the historical name inventory on failure and can inspect ``fallback_used``.
    """

    try:
        tree = parse_formula(formula)
    except FormulaSyntaxError as error:
        empty = frozenset()
        return RelationPredicateReferences(
            identifiers=empty
            if predicate_terms
            else frozenset(formula_identifiers(formula)),
            atom_relations=empty,
            bound_variables=empty,
            out_of_scope_variables=empty,
            unbound_atom_terms=empty,
            qualified_reads=empty,
            relation_arguments=empty,
            parse_error=str(error),
            parse_error_position=(error.line, error.col),
            fallback_used=not predicate_terms,
        )

    def descendants(expr: Expr):
        pending = [expr]
        while pending:
            current = pending.pop()
            yield current
            pending.extend(current.children)
            for pattern, value in current.cases:
                pending.extend((pattern, value))
            if current.default is not None:
                pending.append(current.default)

    # The engine reserves every non-role spelling in an exists generator before
    # lowering either branch. A name cannot become an input outside its body.
    bound = {
        arg.value
        for expr in descendants(tree)
        if predicate_terms
        and expr.kind == "call"
        and expr.value == "exists"
        and expr.children
        and expr.children[0].kind == "call"
        for arg in expr.children[0].children
        if arg.kind == "var" and arg.value not in _TERMS
    }
    names: set[str] = set()
    atoms: set[str] = set()
    out_of_scope: set[str] = set()
    unbound_terms: set[str] = set()
    qualified_reads: set[str] = set()
    relation_arguments: set[str] = set()
    inputs: set[str] = set()
    rules: set[str] = set()
    parameters: set[str] = set()
    errors: list[str] = []
    initial_scope = _TERMS if predicate_terms else frozenset()

    def error(expr: Expr, message: str) -> None:
        errors.append(f"line {expr.line}, col {expr.col}: {message}")

    def check_name(name: str, expr: Expr, scope: frozenset[str]) -> bool:
        if name in bound and name not in scope:
            out_of_scope.add(name)
            error(expr, f"bound variable `{name}` is outside its exists body")
            return False
        if name in scope:
            error(expr, f"term `{name}` cannot be read as a value")
            return False
        return True

    def read(
        name: str, expr: Expr, scope: frozenset[str], *, related: bool = False
    ) -> None:
        if not check_name(name, expr, scope):
            return
        if not related and name in {"period_start", "period_end"}:
            return
        names.add(name)
        if not related and name in parameter_names:
            parameters.add(name)
        elif name in rule_names:
            rules.add(name)
        else:
            inputs.add(name)

    def relation(
        name: str, expr: Expr, scope: frozenset[str], *, atom: bool = False
    ) -> None:
        if not check_name(name, expr, scope):
            return
        names.add(name)
        (atoms if atom else relation_arguments).add(name)
        # Scalar reductions infer their relation declarations. Atoms and
        # generators, in contrast, require the declared-relation namespace.
        if (
            (atom or predicate_terms)
            and relation_names is not None
            and name not in relation_names
        ):
            error(expr, f"undeclared relation `{name}`")

    def field_access(expr: Expr) -> tuple[str, str] | None:
        if expr.kind == "field" and expr.children[0].kind == "var":
            return expr.children[0].value, expr.value
        return None

    def atom(expr: Expr, scope: frozenset[str], mode: str) -> None:
        relation(expr.value, expr, scope, atom=True)
        if not predicate_terms:
            error(expr, "relation atoms require a derived relation predicate")
        if mode != "judgment":
            error(expr, "a relation atom cannot be read as a scalar value")
        if expr.value in _FUNCTIONS:
            error(expr, f"relation `{expr.value}` shadows a built-in function")
        for arg in expr.children:
            if arg.kind != "var":
                error(arg, "a relation atom argument must be a term")
            elif arg.value not in scope:
                if arg.value in bound:
                    out_of_scope.add(arg.value)
                else:
                    unbound_terms.add(arg.value)
                error(arg, f"`{arg.value}` is not a term in scope")

    def walk(expr: Expr, scope: frozenset[str], mode: str):
        kind, value, children = expr.kind, expr.value, expr.children
        if kind in {"int", "float", "string", "true", "false"}:
            if mode == "judgment" and kind not in {"true", "false"}:
                error(expr, "a scalar literal is not a judgment")
            return
        if kind == "var":
            if mode == "judgment" and value in {"period_start", "period_end"}:
                error(expr, f"`{value}` is a date and cannot be tested as a judgment")
            if (
                mode == "judgment"
                and lowering_entry == "derived_relation"
                and relation_names is not None
                and value in relation_names
                and value not in parameter_names | rule_names
            ):
                relation(value, expr, scope)
            else:
                read(value, expr, scope)
            return
        if kind == "field":
            field = field_access(expr)
            if field and field[0] in scope:
                qualified_reads.add(field[1])
                read(field[1], expr, scope)
                if relation_names is not None and field[1] in relation_names:
                    error(expr, "a term-qualified read cannot read a relation")
                if field[1] in parameter_names | {"period_start", "period_end"}:
                    error(expr, "a term-qualified read requires an entity")
            else:
                if field and field[0] in bound:
                    out_of_scope.add(field[0])
                error(expr, "bare field access requires a term in scope")
            return
        if kind == "binary":
            if value in {"and", "or"}:
                if mode != "judgment":
                    error(expr, "a Boolean operator cannot be read as a scalar")
                for child in children:
                    yield walk(child, scope, "judgment")
            elif value in {"<", ">", "<=", ">=", "==", "!="}:
                if mode != "judgment":
                    error(expr, "a comparison cannot be read as a scalar")
                terms = [
                    child.kind == "var" and child.value in scope for child in children
                ]
                if any(terms):
                    if not all(terms) or value not in {"==", "!="}:
                        error(
                            expr,
                            "entity terms support only == and != with another term",
                        )
                    for child, is_term in zip(children, terms, strict=True):
                        if not is_term:
                            yield walk(child, scope, "scalar")
                else:
                    for child in children:
                        yield walk(child, scope, "scalar")
            else:
                if mode == "judgment":
                    error(expr, "an arithmetic expression is not a judgment")
                for child in children:
                    yield walk(child, scope, "scalar")
            return
        if kind == "unary":
            expected = "judgment" if value == "not" else "scalar"
            if mode != expected:
                error(expr, f"`{value}` is not valid in {mode} position")
            yield walk(children[0], scope, expected)
            return
        if kind == "index":
            target, index = children
            if target.kind != "var":
                error(target, "parameter lookup target must be a named parameter")
            else:
                read(target.value, target, scope)
                if parameter_names and target.value not in parameter_names:
                    error(target, "parameter lookup target is not a parameter")
            yield walk(index, scope, "scalar")
            if mode == "judgment":
                error(expr, "parameter lookup is not a judgment")
            return
        if kind == "cond":
            if mode == "judgment":
                error(expr, "a conditional must be read as a scalar")
            yield walk(children[0], scope, "judgment")
            for child in children[1:]:
                yield walk(child, scope, "scalar")
            return
        if kind == "match":
            if mode == "judgment":
                error(expr, "a match must be read as a scalar")
            if not expr.cases and expr.default is None:
                error(expr, "empty match")
            # Lowering discards the subject of a wildcard-only match. It never
            # reaches the compiled expression and therefore reads no input.
            if expr.cases or expr.default is None:
                yield walk(children[0], scope, "scalar")
            else:
                # Lowering validates the discarded subject, so an atom or
                # unknown call cannot hide behind a wildcard-only match.
                read_sets = (
                    names,
                    atoms,
                    qualified_reads,
                    relation_arguments,
                    inputs,
                    rules,
                    parameters,
                )
                saved = tuple(set(read_set) for read_set in read_sets)
                yield walk(children[0], scope, "scalar")
                for read_set, before in zip(read_sets, saved, strict=True):
                    read_set.clear()
                    read_set.update(before)
            for pattern, result in expr.cases:
                yield walk(pattern, scope, "scalar")
                yield walk(result, scope, "scalar")
            if expr.default is not None:
                yield walk(expr.default, scope, "scalar")
            return
        assert kind == "call"
        if value == "exists":
            if not predicate_terms or mode != "judgment":
                error(
                    expr,
                    "exists requires judgment position in a derived relation predicate",
                )
            if len(children) != 2 or children[0].kind != "call":
                error(expr, "exists requires a generator atom and a body")
                return
            generator, body = children
            relation(generator.value, generator, scope, atom=True)
            if generator.value in _FUNCTIONS:
                error(
                    generator, "exists generator relation shadows a built-in function"
                )
            fresh = [
                arg.value
                for arg in generator.children
                if arg.kind == "var" and arg.value not in scope
            ]
            if any(arg.kind != "var" for arg in generator.children):
                error(
                    generator, "generator arguments must be terms or its new variable"
                )
            if len(fresh) != 1:
                error(
                    generator, "an exists generator must bind exactly one new variable"
                )
                yield walk(body, scope, "judgment")
                return
            variable = fresh[0]
            if not re.fullmatch(r"[a-z][a-z0-9_]*", variable):
                error(generator, "an exists variable must be a lowercase identifier")
            if variable in parameter_names | rule_names | (
                relation_names or frozenset()
            ):
                error(generator, "an exists variable conflicts with a declaration")
            yield walk(body, scope | {variable}, "judgment")
            return
        is_atom = value not in _FUNCTIONS or (
            relation_names is not None
            and value in relation_names
            and any(arg.kind == "var" and arg.value in scope for arg in children)
        )
        if is_atom:
            atom(expr, scope, mode)
            return
        if value == "exactly_one":
            if mode != "judgment" or len(children) < 2:
                error(expr, "exactly_one requires at least two judgments")
            for child in children:
                yield walk(child, scope, "judgment")
            return
        if mode == "judgment":
            error(expr, f"unknown function `{value}` in judgment position")
        first = children[0] if children else None
        field = field_access(first) if first else None
        if value == "len":
            # The engine ignores both a projected field and trailing args.
            name = (
                first.value
                if first and first.kind == "var"
                else field[0]
                if field
                else None
            )
            if name is None:
                error(expr, "len requires a relation argument")
            else:
                relation(name, first, scope)
            return
        if value in {"sum", "only"} or (
            value in {"min", "max"}
            and len(children) == 1
            and field
            and field[0] not in scope
        ):
            if field is None or (value == "only" and len(children) != 1):
                error(expr, f"{value} requires a relation field access")
            else:
                relation(field[0], first, scope)
                read(field[1], first, scope, related=True)
            return
        if value in {
            "only_related",
            "count_where",
            "sum_where",
            "min_where",
            "max_where",
        }:
            arity = 2 if value in {"only_related", "count_where"} else 3
            if len(children) != arity or any(child.kind != "var" for child in children):
                error(expr, f"{value} requires {arity} unqualified names")
                return
            relation(children[0].value, children[0], scope)
            for child in children[1:]:
                read(child.value, child, scope, related=True)
            if value == "only_related" and children[1].value in rule_names:
                error(children[1], "only_related requires an input, not a derived rule")
            return
        if value == "input_for_year":
            if (
                len(children) != 2
                or children[0].kind != "var"
                or children[1].kind != "int"
            ):
                error(expr, "input_for_year requires an input name and integer year")
            else:
                read(children[0].value, children[0], scope, related=True)
                if children[0].value in rule_names | parameter_names:
                    error(
                        children[0],
                        "input_for_year requires a declared primitive input",
                    )
                year = children[1].value.replace("_", "").lstrip("0") or "0"
                if not 1 <= int(year) <= 9999:
                    error(children[1], "input_for_year year must be 1..9999")
            return
        arities = {
            "ceil": 1,
            "floor": 1,
            "days_between": 2,
            "date_add_days": 2,
            "date_add_months": 2,
            "date_add_years": 2,
            "sum_over_periods": 1,
            "max_over_periods": 1,
            "count_over_periods": 1,
            "sum_top_n_over_periods": 2,
        }
        if value in arities and len(children) != arities[value]:
            error(expr, f"{value} takes {arities[value]} arguments")
        if value in {"min", "max"} and not children:
            error(expr, f"{value} requires at least one operand")
        for child in children:
            yield walk(child, scope, "scalar")

    # Suspend each child walk to an explicit stack. A valid long Boolean or
    # arithmetic chain must not acquire Python's recursion limit as grammar.
    pending = [walk(tree, initial_scope, "judgment" if predicate_terms else "scalar")]
    while pending:
        try:
            pending.append(next(pending[-1]))
        except StopIteration:
            pending.pop()

    return RelationPredicateReferences(
        identifiers=frozenset(names),
        atom_relations=frozenset(atoms),
        bound_variables=frozenset(bound),
        out_of_scope_variables=frozenset(out_of_scope),
        unbound_atom_terms=frozenset(unbound_terms),
        qualified_reads=frozenset(qualified_reads),
        relation_arguments=frozenset(relation_arguments),
        analysis_errors=tuple(dict.fromkeys(errors)),
        input_reads=frozenset(inputs),
        rule_reads=frozenset(rules),
        parameter_reads=frozenset(parameters),
    )


def formula_identifiers(formula: str, *, relation_predicate: bool = False) -> set[str]:
    """Return names; predicates distinguish builtin calls from bare reads."""

    if relation_predicate:
        return set(analyze_relation_predicate(formula).identifiers)
    return set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _QUOTED.sub(" ", formula)))


def relation_source_names(config: object) -> set[str]:
    """Read both candidate-source forms without inventing a single-source alias."""

    if not isinstance(config, Mapping):
        return set()
    names: set[str] = set()
    source = config.get("source_relation")
    if isinstance(source, str) and source.strip():
        names.add(source.strip())
    path = config.get("source_path")
    if isinstance(path, list):
        names.update(
            step["relation"].strip()
            for step in path
            if isinstance(step, Mapping)
            and isinstance(step.get("relation"), str)
            and step["relation"].strip()
        )
    return names


def local_reference_name(reference: str) -> str:
    """Resolve the local symbol of an executable or relation import reference."""
    if "#" not in reference:
        return reference.strip()
    return reference.rsplit("#", 1)[1].removeprefix("relation.").strip()
