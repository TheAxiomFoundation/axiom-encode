"""Read formula references without treating relation terms as input names.

The engine owns parsing and type checking. This walk only follows balanced
formula groups and the documented ``exists(atom, body)`` binding grammar, so it
also works inside RuleSpec's ``if`` and ``match`` expression syntax.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TOKENS = re.compile(
    r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\#[^\n]*"
    r"|[A-Za-z_][A-Za-z0-9_]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|==|!=|<=|>=|[^\s]"
)
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


@dataclass(frozen=True)
class RelationPredicateReferences:
    """Separate declaration references from predicate-local entity terms."""

    identifiers: frozenset[str]
    atom_relations: frozenset[str]
    bound_variables: frozenset[str]
    out_of_scope_variables: frozenset[str]
    unbound_atom_terms: frozenset[str]
    qualified_reads: frozenset[str]


def analyze_relation_predicate(formula: str) -> RelationPredicateReferences:
    """Follow the scope of every well-shaped bounded quantifier.

    Invalid syntax remains visible to the engine. In particular, this walk
    never drops an unknown atom argument or an ordinary input inside a body.
    A bound variable used outside its body stays a scope error, rather than
    becoming a fictitious input declaration.
    """

    tokens = [
        token for token in _TOKENS.findall(formula) if token[0] not in {"'", '"', "#"}
    ]
    closes: dict[int, int] = {}
    stack: list[int] = []
    for index, token in enumerate(tokens):
        if token in {"(", "[", "{"}:
            stack.append(index)
        elif token in {")", "]", "}"} and stack:
            opening = stack.pop()
            if (tokens[opening], token) in {("(", ")"), ("[", "]"), ("{", "}")}:
                closes[opening] = index

    def arguments(start: int, end: int) -> list[tuple[int, int]]:
        parts: list[tuple[int, int]] = []
        first = index = start
        while index < end:
            if index in closes:
                index = closes[index] + 1
                continue
            if tokens[index] == ",":
                parts.append((first, index))
                first = index + 1
            index += 1
        parts.append((first, end))
        return parts

    def atom_arguments(start: int, end: int) -> list[tuple[int, int]] | None:
        if (
            end - start >= 3
            and _IDENTIFIER.fullmatch(tokens[start])
            and tokens[start + 1] == "("
            and closes.get(start + 1) == end - 1
        ):
            return arguments(start + 2, end - 1)
        return None

    bindings: dict[int, str] = {}

    def discover(start: int, end: int, scope: frozenset[str]) -> None:
        index = start
        while index < end:
            if (
                tokens[index] == "exists"
                and index + 1 in closes
                and tokens[index + 1] == "("
            ):
                closing = closes[index + 1]
                parts = arguments(index + 2, closing)
                if len(parts) == 2:
                    generator, body = parts
                    atom_parts = atom_arguments(*generator)
                    fresh = [
                        tokens[left]
                        for left, right in atom_parts or []
                        if right == left + 1
                        and _IDENTIFIER.fullmatch(tokens[left])
                        and tokens[left] not in scope
                    ]
                    if len(fresh) == 1:
                        bindings[index] = fresh[0]
                        local_scope = scope | {fresh[0]}
                        discover(*generator, local_scope)
                        discover(*body, local_scope)
                        index = closing + 1
                        continue
            index += 1

    discover(0, len(tokens), _TERMS)
    bound = frozenset(bindings.values())
    names: set[str] = set()
    atoms: set[str] = set()
    out_of_scope: set[str] = set()
    unbound_terms: set[str] = set()
    qualified_reads: set[str] = set()

    def walk(start: int, end: int, scope: frozenset[str]) -> None:
        index = start
        while index < end:
            token = tokens[index]
            if index in bindings:
                closing = closes[index + 1]
                for part in arguments(index + 2, closing):
                    walk(*part, scope | {bindings[index]})
                index = closing + 1
                continue
            if _IDENTIFIER.fullmatch(token):
                is_call = index + 1 < end and tokens[index + 1] == "("
                is_field = index > start and tokens[index - 1] == "."
                if is_field and index >= 2 and tokens[index - 2] in scope:
                    qualified_reads.add(token)
                if token in bound and token not in scope:
                    out_of_scope.add(token)
                if (
                    token not in _TERMS | bound
                    and (is_field or token not in _KEYWORDS)
                    and not (is_call and token in _FUNCTIONS)
                ):
                    names.add(token)
                if (
                    index + 1 < end
                    and tokens[index + 1] == "("
                    and index + 1 in closes
                    and token not in _FUNCTIONS | _KEYWORDS
                ):
                    atoms.add(token)
                    for left, right in arguments(index + 2, closes[index + 1]):
                        if (
                            right == left + 1
                            and _IDENTIFIER.fullmatch(tokens[left])
                            and tokens[left] not in scope
                        ):
                            unbound_terms.add(tokens[left])
            index += 1

    walk(0, len(tokens), _TERMS)
    return RelationPredicateReferences(
        identifiers=frozenset(names),
        atom_relations=frozenset(atoms),
        bound_variables=bound,
        out_of_scope_variables=frozenset(out_of_scope),
        unbound_atom_terms=frozenset(unbound_terms),
        qualified_reads=frozenset(qualified_reads),
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
