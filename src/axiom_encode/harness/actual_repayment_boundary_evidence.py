"""Own-case equality/above evidence for a source-certified repayment threshold.

An added worksheet observation is admitted only on the branch that reads it.
No case is perturbed, and no asserted output supplies a missing runtime value.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from .imported_export_evidence import (
    _BULLET,
    _ROW,
    ResolvedImportedExport,
    _named,
    _payload,
)
from .repayment_alternative_evidence import (
    RepaymentAlternativeRecord,
    certify_repayment_alternatives,
)


@dataclass(frozen=True)
class ThresholdOccurrence:
    record: RepaymentAlternativeRecord
    span: tuple[int, int]
    income: str
    parameter: str
    worksheet: str
    coordinate: int
    threshold: Decimal
    entered: tuple[int, ...]


def _boundary_span(branch: Any, boundary: Any, source: str, branches: Sequence[Any]):
    from . import source_completeness as s

    raw = s._source_branch_direct_text(branch, branches=branches)
    if (
        not isinstance(raw, str)
        or not 0 <= branch.start < branch.end <= len(source)
        or source[branch.start : branch.end] != branch.text
        or source[branch.start : branch.start + len(raw)] != raw
    ):
        return None
    mapping = s._NumericRecallSourceMap(list(range(len(raw))))
    recall = s.authoritative_numeric_recall_text(raw, _source_map=mapping)
    span = mapping.source_span(boundary.start, boundary.end)
    if span is None:
        return None
    start, end = span
    if raw[start:end] != recall[boundary.start : boundary.end]:
        return None
    return branch.start + start, branch.start + end


def bind_occurrences(
    content: str,
    branch: Any,
    boundary: Any,
    *,
    source_text: str,
    citation: str,
    context: Mapping[str, str | None] | None,
    records: Sequence[ResolvedImportedExport],
    principal: Mapping[str, dict],
    boundary_branches: Sequence[Any],
    source_branches: Sequence[Any] = (),
) -> tuple[ThresholdOccurrence, ...]:
    """Re-certify the entire selector, then bind one physical amount occurrence."""
    from . import source_completeness as s

    raw = context.get(citation) if context else None
    payload = _payload(content)
    if not isinstance(raw, str) or payload is None or raw.strip() != source_text:
        return ()
    rules = _named(payload, "rules")
    if rules is None or any(rules.get(n) != r for n, r in principal.items()):
        return ()
    structure = s._unconflicted_printed_rows(source_text, source_branches)
    span = _boundary_span(branch, boundary, source_text, boundary_branches)
    if structure is None or span is None:
        return ()
    coverage = certify_repayment_alternatives(
        content,
        source_text=source_text,
        raw_source_text=raw,
        citation=citation,
        source_context=context,
        imported_exports=records,
        parse_branch=s._parse_first_formula_branch_node,
        parse_expression=s._parse_formula_expression,
    )
    delta = len(raw) - len(raw.lstrip())
    found = []
    for record in coverage.records:
        if record.span not in structure.atomic_spans:
            continue
        cursor = record.span[0]
        bullets = []
        for line in record.text.splitlines(keepends=True):
            quote = line.rstrip("\r\n")
            match = _BULLET.fullmatch(quote[2:]) if quote.startswith("- ") else None
            if match is not None:
                a, b = match.span("amount")
                if (cursor + 2 + a, cursor + 2 + b) == span:
                    bullets.append((match, cursor + 2, cursor + len(quote)))
            cursor += len(line)
        if len(bullets) != 1:
            continue
        bullet, start, end = bullets[0]
        threshold = Decimal(bullet["amount"].replace(",", ""))
        if Decimal(str(boundary.value)) != threshold:
            continue
        coordinate = int(bullet["income"])
        income, parameters = set(), set()
        for name, version, proof_citation, _, proof_span in record.local_proofs:
            if version != record.version or proof_citation != citation:
                continue
            proof = raw[slice(*proof_span)]
            row = _ROW.fullmatch(proof)
            if row is not None and int(row[1]) == coordinate:
                income.add(name)
            if proof_span == (start + delta, end + delta):
                parameters.add(name)
        if len(income) != 1 or len(parameters) != 1:
            continue
        owner = rules[record.owner]
        node = s._parse_first_formula_branch_node(
            owner["versions"][record.version]["formula"]
        )
        if node is None:
            continue
        selector = s._parse_formula_expression(node.selectors[0])
        income_name, parameter = next(iter(income)), next(iter(parameters))
        comparisons = (
            [
                n
                for n in ast.walk(selector)
                if isinstance(n, ast.Compare)
                and isinstance(n.left, ast.Name)
                and n.left.id == income_name
                and len(n.comparators) == 1
                and isinstance(n.comparators[0], ast.Name)
                and n.comparators[0].id == parameter
                and len(n.ops) == 1
                and isinstance(n.ops[0], ast.Gt)
            ]
            if selector is not None
            else []
        )
        if len(comparisons) != 1:
            continue
        found.append(
            ThresholdOccurrence(
                record,
                span,
                income_name,
                parameter,
                node.choices[0].strip(),
                coordinate,
                threshold,
                tuple(int(bullet[k]) for k in ("first", "second") if bullet[k]),
            )
        )
    # Different owners cannot jointly claim one physical source obligation.
    if len({f.record.owner for f in found}) != 1:
        return ()
    return tuple(found)


def _truth(tree: tuple, income: Decimal, entered: Mapping[int, bool], coordinate: int):
    kind = tree[0]
    if kind == "entered":
        return entered.get(tree[1])
    if kind == "gt":
        return income > tree[2] if tree[1] == coordinate else None
    if kind not in {"and", "or"}:
        return None
    children = [_truth(t, income, entered, coordinate) for t in tree[1]]
    if any(v is None for v in children):
        return None
    return all(children) if kind == "and" else any(children)


def _single_money_change(
    evidence, low: Mapping, high: Mapping, worksheet: str, unit: str
):
    """Inspect actual fact maps; never run an evaluator on a modified case."""
    if set(high) != set(low) | {worksheet} or worksheet in low:
        return False
    changes = [k for k in low if type(low[k]) is not type(high[k]) or low[k] != high[k]]
    if len(changes) != 1:
        return False
    key = changes[0]
    if not isinstance(key, str) or key.count("#input.") != 1:
        return False
    if any(
        type(v) not in (int, float) or not Decimal(str(v)).is_finite()
        for v in (low[key], high[key])
    ):
        return False
    module, name = key.split("#input.")
    files = [
        f for f in evidence.files if f.relative == module.replace(":", "/", 1) + ".yaml"
    ]
    payload = _payload(files[0].raw.decode()) if len(files) == 1 else None
    inputs = _named(payload, "inputs") if payload is not None else None
    fact = inputs.get(name) if inputs is not None else None
    return bool(
        isinstance(fact, dict)
        and not set(fact) - {"name", "dtype", "unit", "entity", "period", "description"}
        and (
            fact.get("dtype"),
            fact.get("unit"),
            fact.get("entity"),
            fact.get("period"),
        )
        == ("Money", unit, "Person", "Year")
    )


def _witnesses(
    evidence,
    branch: Any,
    boundary: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    imported_exports: Sequence[ResolvedImportedExport],
    source_context: Mapping[str, str | None] | None,
    principal_rules: dict[str, dict],
    asserted_by_rule: dict[str, list],
    formula_environment: dict[str, Any],
    boundary_branches: Sequence[Any],
    source_branches: Sequence[Any] = (),
) -> set[tuple]:
    import json
    from itertools import combinations

    from . import source_completeness as s
    from .actual_floor_evidence import _explicit_identity, _query
    from .actual_sparse_worksheet_evidence import _exclusive_worksheet
    from .same_case_execution import case_key

    if not evidence.bound(content, imported_exports):
        return set()
    bindings = bind_occurrences(
        content,
        branch,
        boundary,
        source_text=source_text,
        citation=corpus_citation_path,
        context=source_context,
        records=imported_exports,
        principal=principal_rules,
        boundary_branches=boundary_branches,
        source_branches=source_branches,
    )
    if not bindings:
        return set()
    module = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    if corpus_citation_path != evidence.root_relative.removesuffix(".yaml").replace(
        "/policies/", "/policy/"
    ):
        return set()
    payload = _payload(content)
    rules = _named(payload, "rules") if payload else None
    if rules is None:
        return set()
    try:
        derived = json.loads(evidence.artifact_bytes)["program"]["derived"]
    except (ValueError, KeyError, TypeError):
        return set()
    if not isinstance(derived, list) or any(not isinstance(d, dict) for d in derived):
        return set()
    result = set()
    for binding in bindings:
        record = binding.record
        aliases = [o.export_uri.rsplit("#", 1)[1] for o in record.observations]
        coordinates = [o.coordinate for o in record.observations]
        inputs = [o.input_uri for o in record.observations]
        if (
            len(set(aliases)) != len(aliases)
            or len(set(coordinates)) != len(coordinates)
            or len(set(inputs)) != len(inputs)
            or set(aliases) & (set(formula_environment) | set(rules))
        ):
            continue
        owner = rules[record.owner]
        unit = owner["unit"]
        worksheet_uri = module + "#input." + binding.worksheet
        if not _exclusive_worksheet(evidence, record.owner, binding.worksheet):
            continue
        required = [
            (module + "#" + n, "decimal", "scalar", unit)
            for n in (record.owner, binding.income)
        ]
        required += [
            (o.export_uri, "judgment", "judgment", None) for o in record.observations
        ]
        if any(
            len(ds := [d for d in derived if d.get("id") == ref]) != 1
            or any(
                ds[0].get(k) != v
                for k, v in {
                    "entity": "Person",
                    "period": "Year",
                    "dtype": dtype,
                    "semantics": semantics,
                    "unit": currency,
                }.items()
            )
            for ref, dtype, semantics, currency in required
        ):
            continue
        executed = []
        for case in asserted_by_rule.get(record.owner, ()):
            query = _query(evidence, case)
            facts = case.get("input")
            if (
                query is None
                or not isinstance(facts, dict)
                or not isinstance(case.get("output"), dict)
            ):
                continue
            period = s._normalized_case_period(case)
            if (
                not isinstance(period, str)
                or not record.start <= period <= record.end
                or query.get("period")
                != {
                    "period_kind": "tax_year",
                    "start": record.start,
                    "end": record.end,
                }
            ):
                continue
            values = evidence.values(case, content)
            if values is None:
                continue
            imported, observed = values
            if imported.keys() & formula_environment.keys():
                continue
            entered, direct = {}, {}
            for observation in record.observations:
                input_module, input_name = observation.input_uri.rsplit("#", 1)
                value = facts.get(input_module + "#input." + input_name)
                alias = observation.export_uri.rsplit("#", 1)[1]
                if type(value) is not bool or (
                    alias in imported and imported[alias] is not value
                ):
                    break
                # A certified direct wrapper can establish an explicit supplied
                # Bool even when short-circuiting leaves no trace node. This is
                # factual correspondence, never a claim that that node ran.
                entered[observation.coordinate] = value
                direct[alias] = value
            else:
                environment = {**formula_environment, **imported, **direct}
                replay = s._case_dependency_environment(
                    principal_rules,
                    case,
                    formula_environment=environment,
                    require_asserted_value=False,
                )
                if (
                    record.owner not in replay
                    or binding.income not in replay
                    or any(
                        n not in observed
                        or not s._formula_runtime_values_equal(value, observed[n])
                        or not s._asserted_formula_runtime_values_equal(
                            principal_rules[n],
                            value,
                            s._test_case_asserted_output_value(case, n),
                        )
                        for n, value in replay.items()
                    )
                ):
                    continue
                income = replay[binding.income]
                if isinstance(income, bool) or not isinstance(
                    income, (int, float, Decimal)
                ):
                    continue
                income = Decimal(str(income))
                if not income.is_finite() or income < binding.threshold:
                    continue
                above = income > binding.threshold
                if (worksheet_uri in facts) != above or not any(
                    entered.get(k) is True for k in binding.entered
                ):
                    continue
                # Every other alternative must be false in this same world.
                # Equal numeric thresholds elsewhere cannot claim this span.
                target = []
                for subtree in record.tree[1]:
                    comparisons = (
                        [c for c in subtree[1] if c[0] == "gt"]
                        if subtree[0] == "and"
                        else []
                    )
                    if comparisons == [("gt", binding.coordinate, binding.threshold)]:
                        target.append(subtree)
                    elif (
                        _truth(subtree, income, entered, binding.coordinate)
                        is not False
                    ):
                        break
                else:
                    if (
                        len(target) != 1
                        or _truth(target[0], income, entered, binding.coordinate)
                        is not above
                    ):
                        continue
                    execution = s._case_formula_execution(
                        owner,
                        case,
                        formula_environment=environment,
                        dependency_environment=replay,
                    )
                    expected_leaf = binding.worksheet if above else "0"
                    if (
                        execution is None
                        or len(execution.trace) != 1
                        or execution.leaf.strip() != expected_leaf
                    ):
                        continue
                    actual = observed[record.owner]
                    leaf = facts.get(worksheet_uri) if above else 0
                    if isinstance(
                        leaf, bool
                    ) or not s._asserted_formula_runtime_values_equal(
                        owner, actual, leaf
                    ):
                        continue
                    if above and s._formula_runtime_values_equal(actual, 0):
                        continue
                    executed.append((case, query, above, actual))
        for left, right in combinations(executed, 2):
            a, aq, ab, av = left
            b, bq, bb, bv = right
            if (
                ab == bb
                or av == bv
                or _explicit_identity(a) != _explicit_identity(b)
                or any(aq.get(k) != bq.get(k) for k in ("period", "assessment_date"))
                or a.get("output", {}).keys() != b.get("output", {}).keys()
            ):
                continue
            low, high = (a, b) if not ab else (b, a)
            if not _single_money_change(
                evidence, low["input"], high["input"], worksheet_uri, unit
            ):
                continue
            result.add(
                (
                    record.owner,
                    record.version,
                    record.source_sha256,
                    binding.span,
                    *sorted((case_key(a), case_key(b))),
                )
            )
    return result


def witnesses(*args: Any, **kwargs: Any) -> set[tuple]:
    """Malformed optional receipts provide no new boundary coverage.

    This wrapper encloses only this certificate, never ordinary validation or
    execution. Unsupported shapes leave the existing obligation unresolved.
    """
    try:
        return _witnesses(*args, **kwargs)
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        InvalidOperation,
    ):
        return set()
