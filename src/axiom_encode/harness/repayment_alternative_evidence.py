"""Closed source/AST correspondence for explicit repayment alternatives.

This evidence addresses selector opacity only. It does not execute imported
rules, certify a worksheet calculation, or supply any companion-case evidence.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Mapping, Sequence

from axiom_encode.harness.imported_export_evidence import (
    _BULLET,
    _ROW,
    BoundEnteredObservation,
    ResolvedImportedExport,
    _atoms,
    _frame,
    _named,
    _payload,
    _sha,
    _source,
    bind_entered_observations,
)
from axiom_encode.printed_source_rows import recognize_printed_rows

_GROUP = re.compile(
    r"[A-Z][A-Za-z -]{3,80}:\r?\n\r?\n"
    r"Complete the chart for line (?P<result>[1-9]\d{4}) using your Federal Worksheet "
    r"if one or more of the following apply:\r?\n"
    r"(?P<bullets>(?:\r?\n- [^\r\n]+\r?\n){2,6})\r?\n"
    r"Line [1-9]\d{0,2} and line (?P=result): "
    r'If not, enter "0" on line (?P=result)\. \^'
)
_RULE_KEYS = {
    "name",
    "kind",
    "dtype",
    "unit",
    "entity",
    "period",
    "versions",
    "metadata",
    "source",
    "description",
}
_VERSION_KEYS = {"effective_from", "effective_to", "formula"}


@dataclass(frozen=True)
class RepaymentAlternativeRecord:
    consumer_sha256: str
    citation: str
    raw_source_sha256: str
    source_sha256: str
    raw_span: tuple[int, int]
    span: tuple[int, int]
    text: str
    owner: str
    version: int
    start: str
    end: str
    formula_sha256: str
    tree: tuple[Any, ...]
    observations: tuple[BoundEnteredObservation, ...]
    local_proofs: tuple[tuple[str, int, str, str, tuple[int, int]], ...]


@dataclass(frozen=True)
class RepaymentAlternativeCoverage:
    records: tuple[RepaymentAlternativeRecord, ...] = ()

    def owns_condition(
        self,
        *,
        owner: str,
        consumer_content: str,
        version: int,
        formula: str,
        start: str | None,
        end: str | None,
        clause_start: int,
        clause_end: int,
        clause_text: str,
        source_text: str,
        citation: str,
    ) -> bool:
        return any(
            r.owner == owner
            and r.consumer_sha256 == _sha(consumer_content)
            and r.version == version
            and r.formula_sha256 == _sha(formula)
            and (r.start, r.end) == (start, end)
            and r.span == (clause_start, clause_end)
            and r.text == clause_text
            and r.citation == citation
            and r.source_sha256 == _sha(source_text)
            for r in self.records
        )


def _logical(kind: str, children: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Normalize commutative order/parentheses, preserving multiplicity."""
    flattened = []
    for child in children:
        flattened.extend(child[1] if child[0] == kind else (child,))
    return kind, tuple(sorted(flattened, key=repr))


def _money(rule: Mapping[str, Any], *, kind: str) -> bool:
    return (
        not set(rule) - _RULE_KEYS
        and rule.get("kind") == kind
        and rule.get("dtype") == "Money"
        and rule.get("unit") == "CAD"
        and (
            kind == "parameter"
            or (rule.get("entity"), rule.get("period")) == ("Person", "Year")
        )
        and (kind != "parameter" or ("entity" not in rule and "period" not in rule))
    )


def _versions(
    rule: Mapping[str, Any], context: Mapping[str, str | None]
) -> list[tuple[dict[str, Any], tuple[int, str, str]]] | None:
    values = rule.get("versions")
    if not isinstance(values, list) or not values:
        return None
    result = []
    for i, version in enumerate(values):
        if not isinstance(version, dict) or set(version) - _VERSION_KEYS:
            return None
        frame = _frame(rule, i, version, context)
        if frame is None or any(frame[0] == f[0] for _, f in result):
            return None
        result.append((version, frame))
    return result


def _proof_covers(
    rule: Mapping[str, Any],
    index: int,
    citation: str,
    body: str,
    span: tuple[int, int],
    context: Mapping[str, str | None],
) -> bool:
    atoms = _atoms(rule, f"versions[{index}].formula", "formula")
    if len(atoms) != 1:
        return False
    proof = _source(atoms[0], context)
    return bool(
        proof
        and proof[0] == citation
        and proof[1] == body
        and proof[3] <= span[0]
        and span[1] <= proof[3] + len(proof[2])
    )


def _literal(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?", str(value)) is None:
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def certify_repayment_alternatives(
    content: str,
    *,
    source_text: str,
    raw_source_text: str,
    citation: str,
    source_context: Mapping[str, str | None] | None,
    imported_exports: Sequence[ResolvedImportedExport],
    parse_branch: Callable[[str], Any],
    parse_expression: Callable[[str], ast.expr | None],
) -> RepaymentAlternativeCoverage:
    """Require full source/formula correspondence; unknowns grant no coverage."""
    empty = RepaymentAlternativeCoverage()
    if (
        not source_context
        or source_context.get(citation) != raw_source_text
        or raw_source_text.strip() != source_text
    ):
        return empty
    payload = _payload(content)
    structure = recognize_printed_rows(source_text)
    if payload is None or structure is None:
        return empty
    rules, inputs = _named(payload, "rules"), _named(payload, "inputs")
    if rules is None or inputs is None or set(rules) & set(inputs):
        return empty
    units = payload.get("units")
    if not isinstance(units, list) or [
        u for u in units if isinstance(u, dict) and u.get("name") == "CAD"
    ] != [{"name": "CAD", "kind": "currency", "minor_units": 2}]:
        return empty
    delta = len(raw_source_text) - len(raw_source_text.lstrip())
    groups = [
        (a, b, _GROUP.fullmatch(source_text[a:b])) for a, b in structure.atomic_spans
    ]
    groups = [(a, b, m) for a, b, m in groups if m is not None]
    records = []
    for owner, rule in rules.items():
        if not _money(rule, kind="derived"):
            continue
        versions = _versions(rule, source_context)
        if versions is None:
            continue
        observations = bind_entered_observations(
            content,
            owner=owner,
            consumer_citation=citation,
            source_context=source_context,
            imported_exports=imported_exports,
        )
        if not observations:
            continue
        by_alias = {o.export_uri.rsplit("#", 1)[1]: o for o in observations}
        owned = []
        for index, (version, frame) in enumerate(versions):
            formula = version.get("formula")
            if not isinstance(formula, str):
                break
            parsed_formula = formula.strip()
            node = parse_branch(parsed_formula)
            if (
                node is None
                or node.kind != "if"
                or node.start != 0
                or node.end != len(parsed_formula)
                or len(node.selectors) != 1
                or len(node.choices) != 2
                or node.patterns
            ):
                break
            positive, negative = (parse_expression(x) for x in node.choices)
            if (
                not isinstance(positive, ast.Name)
                or positive.id not in inputs
                or not isinstance(negative, ast.Constant)
                or isinstance(negative.value, bool)
                or _literal(negative.value) != 0
            ):
                break
            fact = inputs[positive.id]
            if set(fact) - {
                "name",
                "dtype",
                "unit",
                "entity",
                "period",
                "description",
            } or (
                fact.get("dtype"),
                fact.get("unit"),
                fact.get("entity"),
                fact.get("period"),
            ) != ("Money", "CAD", "Person", "Year"):
                break
            matches = []
            for start, end, group in groups:
                raw_span = (start + delta, end + delta)
                if not _proof_covers(
                    rule, index, citation, raw_source_text, raw_span, source_context
                ):
                    continue
                bullets = [
                    line[2:]
                    for line in group["bullets"].splitlines()
                    if line.startswith("- ")
                ]
                parsed = [_BULLET.fullmatch(line) for line in bullets]
                if any(p is None for p in parsed):
                    continue
                expected = []
                for p in parsed:
                    assert p is not None
                    entered = [
                        ("entered", int(p[k])) for k in ("first", "second") if p[k]
                    ]
                    observation = (
                        entered[0] if len(entered) == 1 else _logical("or", entered)
                    )
                    expected.append(
                        _logical(
                            "and",
                            (
                                observation,
                                (
                                    "gt",
                                    int(p["income"]),
                                    Decimal(p["amount"].replace(",", "")),
                                ),
                            ),
                        )
                    )
                source_tree = _logical("or", expected)
                local_proofs = set()
                used_observations = set()

                # Bind names by source-owned row/parameter proofs, not spelling.
                def income(name: str) -> int | None:
                    candidate = rules.get(name)
                    if candidate is None or not _money(candidate, kind="derived"):
                        return None
                    vv = _versions(candidate, source_context)
                    if vv is None or {f for _, f in vv} != {f for _, f in versions}:
                        return None
                    coordinates = set()
                    for j, (v, _) in enumerate(vv):
                        if (
                            not isinstance(v.get("formula"), str)
                            or not v["formula"].strip()
                        ):
                            return None
                        aa = _atoms(candidate, f"versions[{j}].formula", "formula")
                        proof = _source(aa[0], source_context) if len(aa) == 1 else None
                        row = (
                            _ROW.fullmatch(proof[2])
                            if proof
                            and proof[0] == citation
                            and proof[1] == raw_source_text
                            else None
                        )
                        if row is None or (
                            proof[3] and raw_source_text[proof[3] - 1] != "\n"
                        ):
                            return None
                        if (
                            proof[3] + len(proof[2]) < len(raw_source_text)
                            and raw_source_text[proof[3] + len(proof[2])] not in "\r\n"
                        ):
                            return None
                        local_proofs.add(
                            (
                                name,
                                j,
                                citation,
                                _sha(raw_source_text),
                                (proof[3], proof[3] + len(proof[2])),
                            )
                        )
                        coordinates.add(int(row[1]))
                    return next(iter(coordinates)) if len(coordinates) == 1 else None

                def threshold(name: str) -> Decimal | None:
                    candidate = rules.get(name)
                    if candidate is None or not _money(candidate, kind="parameter"):
                        return None
                    vv = _versions(candidate, source_context)
                    if vv is None or {f for _, f in vv} != {f for _, f in versions}:
                        return None
                    active = None
                    for j, (v, f) in enumerate(vv):
                        value = _literal(v.get("formula"))
                        if value is None:
                            return None
                        # The active value must be tied to this exact group's
                        # threshold occurrence, not an unrelated equal number.
                        if f == frame:
                            matching = [
                                line
                                for line, p in zip(bullets, parsed)
                                if p and Decimal(p["amount"].replace(",", "")) == value
                            ]
                            if len(matching) != 1:
                                return None
                            position = raw_source_text.find(
                                matching[0], raw_span[0], raw_span[1]
                            )
                            if position < 0 or not _proof_covers(
                                candidate,
                                j,
                                citation,
                                raw_source_text,
                                (position, position + len(matching[0])),
                                source_context,
                            ):
                                return None
                            local_proofs.add(
                                (
                                    name,
                                    j,
                                    citation,
                                    _sha(raw_source_text),
                                    (position, position + len(matching[0])),
                                )
                            )
                            active = value
                    return active

                def tree(expression: ast.expr | None) -> tuple[Any, ...] | None:
                    if isinstance(expression, ast.Name):
                        observed = by_alias.get(expression.id)
                        if observed is None or not any(
                            v.year == frame[0]
                            and v.predicate_citation == citation
                            and raw_span[0]
                            <= v.predicate_span[0]
                            < v.predicate_span[1]
                            <= raw_span[1]
                            for v in observed.versions
                        ):
                            return None
                        used_observations.add(expression.id)
                        return "entered", observed.coordinate
                    if isinstance(expression, ast.BoolOp) and isinstance(
                        expression.op, (ast.And, ast.Or)
                    ):
                        values = [tree(v) for v in expression.values]
                        if any(v is None for v in values):
                            return None
                        return _logical(
                            "and" if isinstance(expression.op, ast.And) else "or",
                            values,
                        )
                    if (
                        isinstance(expression, ast.Compare)
                        and len(expression.ops) == 1
                        and isinstance(expression.ops[0], ast.Gt)
                        and isinstance(expression.left, ast.Name)
                        and len(expression.comparators) == 1
                        and isinstance(expression.comparators[0], ast.Name)
                    ):
                        coordinate, value = (
                            income(expression.left.id),
                            threshold(expression.comparators[0].id),
                        )
                        if coordinate is not None and value is not None:
                            return "gt", coordinate, value
                    return None

                if tree(parse_expression(node.selectors[0])) != source_tree:
                    continue
                matches.append(
                    RepaymentAlternativeRecord(
                        _sha(content),
                        citation,
                        _sha(raw_source_text),
                        _sha(source_text),
                        raw_span,
                        (start, end),
                        source_text[start:end],
                        owner,
                        index,
                        version["effective_from"],
                        version["effective_to"],
                        _sha(formula),
                        source_tree,
                        tuple(by_alias[name] for name in sorted(used_observations)),
                        tuple(sorted(local_proofs)),
                    )
                )
            if len(matches) != 1:
                break
            owned.extend(matches)
        if len(owned) == len(versions):
            records.extend(owned)
    return RepaymentAlternativeCoverage(tuple(records))
