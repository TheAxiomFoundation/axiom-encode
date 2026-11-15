"""Actual, separate-case evidence for an explicitly sourced subtraction floor.

No receipt is evaluated under changed facts. Legacy counterfactual and boundary
witnesses are deliberately not involved in this narrowly shaped certificate.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from decimal import Decimal
from itertools import combinations
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.printed_source_rows import recognize_printed_rows

from .imported_export_evidence import (
    ResolvedImportedExport,
    _atoms,
    _canonical_record,
    _frame,
    _source,
)
from .same_case_execution import (
    SameCaseEvidence,
    _payload,
    case_key,
    formula_witnesses,
    strict_explain_metadata,
)

_FLOOR = re.compile(
    r"Line (?P<target>\d+): Line (?P<left>\d+) minus line (?P<right>\d+) "
    r'\(if negative, enter ["“]0["”]\) [A-Za-z][A-Za-z ]*\s*=\s*\^',
    re.IGNORECASE,
)


def _subtraction(formula: str) -> ast.BinOp | None:
    try:
        expression = ast.parse(formula.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "max"
        and not expression.keywords
        and len(expression.args) == 2
        and isinstance(expression.args[0], ast.Constant)
        and type(expression.args[0].value) is int
        and expression.args[0].value == 0
        and isinstance(expression.args[1], ast.BinOp)
        and isinstance(expression.args[1].op, ast.Sub)
        and isinstance(expression.args[1].left, ast.Name)
        and isinstance(expression.args[1].right, ast.Name)
        and expression.args[1].left.id != expression.args[1].right.id
    ):
        return None
    return expression.args[1]


def _query(evidence: SameCaseEvidence, case: dict[str, Any]) -> dict[str, Any] | None:
    from .validator_pipeline import ValidatorPipeline

    records = [r for r in evidence.records if r.case_sha256 == case_key(case)]
    if len(records) != 1 or type(records[0].case_index) is not int:
        return None
    record = records[0]
    if record.case_index < 1 or case.get("tables") or case.get("relations"):
        return None
    request, response = (
        json.loads(record.request_json),
        json.loads(record.response_json),
    )
    if not isinstance(request, dict) or not isinstance(response, dict):
        return None
    queries, rows = request.get("queries"), response.get("results")
    if (
        not isinstance(queries, list)
        or len(queries) != 1
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        return None
    query, row = queries[0], rows[0]
    if not isinstance(query, dict) or not isinstance(row, dict):
        return None
    trace = row.get("trace")
    if not isinstance(trace, dict) or any(
        not isinstance(node, dict) for node in trace.values()
    ):
        return None
    pipeline = object.__new__(ValidatorPipeline)
    try:
        if query.get("period") != pipeline._coerce_rulespec_period(case.get("period")):
            return None
    except (ValueError, TypeError):
        return None
    expected_id = pipeline._rulespec_case_query_entity_id(
        case, "Person", record.case_index
    )
    if query.get("entity_id") != expected_id or any(
        row.get(key) != query.get(key)
        for key in ("entity_id", "period", "assessment_date")
    ):
        return None
    if not strict_explain_metadata(request, response):
        return None
    return query


def _explicit_identity(case: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key, str(value))
            for key, value in case.items()
            if key in {"entity_id", "id", "person_id"} or key.endswith("_id")
        )
    )


def _changed_primitive(
    evidence: SameCaseEvidence, left: dict, right: dict
) -> str | None:
    li, ri = left.get("input"), right.get("input")
    if not isinstance(li, dict) or not isinstance(ri, dict) or li.keys() != ri.keys():
        return None
    changes = [
        key for key in li if type(li[key]) is not type(ri[key]) or li[key] != ri[key]
    ]
    if len(changes) != 1:
        return None
    key = changes[0]
    if not isinstance(key, str) or "#input." not in key:
        return None
    if any(
        type(value) not in (int, float) or not Decimal(str(value)).is_finite()
        for value in (li[key], ri[key])
    ):
        return None
    module, name = key.split("#input.", 1)
    relative = module.replace(":", "/", 1) + ".yaml"
    files = [f for f in evidence.files if f.relative == relative]
    if len(files) != 1:
        return None
    try:
        inputs = _payload(files[0].raw).get("inputs", [])
    except (ValueError, TypeError, yaml.YAMLError):
        return None
    if not isinstance(inputs, list) or not all(isinstance(d, dict) for d in inputs):
        return None
    declarations = [d for d in inputs if d.get("name") == name]
    if len(declarations) != 1:
        return None
    declaration = declarations[0]
    if any(k in declaration for k in ("default", "formula", "versions")) or (
        declaration.get("dtype"),
        declaration.get("entity"),
        declaration.get("period"),
    ) != ("Money", "Person", "Year"):
        return None
    return declaration.get("unit")


_ROW = re.compile(
    r"(?m)^Line (?P<row>[1-9]\d*)(?: and (?:line )?(?P<field>[1-9]\d*))?: [^\r\n]+$"
)


def _ordered_operands(
    evidence: SameCaseEvidence,
    payload: dict[str, Any],
    content: str,
    owner: dict[str, Any],
    subtraction: ast.BinOp,
    branch: Any,
    citation: str,
    source_text: str,
    context: Mapping[str, str | None] | None,
    records: Sequence[ResolvedImportedExport],
) -> bool:
    """Bind operand positions to actual printed rows, including page carry-in."""
    raw_body = context.get(citation) if context else None
    if (
        not isinstance(raw_body, str)
        or raw_body.strip() != source_text
        or not evidence.bound(content, records)
    ):
        return False
    module = payload.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    root_citation = evidence.root_relative.removesuffix(".yaml").replace(
        "/policies/", "/policy/"
    )
    if (
        not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != citation
        or citation != root_citation
    ):
        return False
    imports = payload.get("imports", [])
    if len(records) != len(imports) or {r.import_uri for r in records} != set(imports):
        return False
    snapshots, module_paths = {}, {}
    for record in records:
        if (
            not _canonical_record(record)
            or record.consumer_imports != tuple(imports)
            or record.content_sha256
            != hashlib.sha256(record.content.encode()).hexdigest()
        ):
            return False
        snapshot = (record.content_sha256, record.content)
        module_id = record.import_uri.rsplit("#", 1)[0]
        if (
            snapshots.setdefault(record.resolved_path, snapshot) != snapshot
            or module_paths.setdefault(module_id, record.resolved_path)
            != record.resolved_path
        ):
            return False
    match = _FLOOR.fullmatch(branch.text.strip())
    if match is None:
        return False
    target, left, right = (int(match[k]) for k in ("target", "left", "right"))
    family = citation.rsplit("/", 1)[0]
    grids = {}
    for key, body in context.items():
        if (
            not isinstance(key, str)
            or not isinstance(body, str)
            or key.rsplit("/", 1)[0] != family
        ):
            continue
        heading = re.match(r"Step ([1-9]\d*) - ", body.lstrip())
        if recognize_printed_rows(body.strip()) is None:
            if heading and any(
                int(m["row"]) in (target, left, right, left - 1)
                for m in _ROW.finditer(body)
            ):
                return False
            continue
        matches = [m for m in _ROW.finditer(body) if int(m["row"]) < 1000]
        coordinates = [int(m["row"]) for m in matches]
        if not coordinates or coordinates != list(
            range(coordinates[0], coordinates[-1] + 1)
        ):
            return False
        rows = dict(zip(coordinates, matches))
        heading = re.match(r"Step ([1-9]\d*) - ", body.lstrip())
        if rows and heading:
            grids[key] = (body, rows, int(heading[1]))
    if citation not in grids:
        return False
    body, rows, step = grids[citation]
    if (
        target not in rows
        or rows[target].group() != branch.text
        or left != min(rows)
        or right not in rows
    ):
        return False
    transfer = re.fullmatch(
        r"Line ([1-9]\d*): Enter the amount from line ([1-9]\d*) of the previous page\. \^",
        rows[left].group(),
    )
    if transfer is None or int(transfer[1]) != left:
        return False
    previous = int(transfer[2])
    if previous + 1 != left:
        return False
    providers = [(key, data) for key, data in grids.items() if previous in data[1]]
    if len(providers) != 1:
        return False
    provider_citation, (provider_body, provider_rows, provider_step) = providers[0]
    if provider_step + 1 != step or max(provider_rows) != previous:
        return False
    # All physical grids in the authenticated family must agree uniquely on
    # each coordinate involved in the source operation, not just the excerpt.
    if any(
        sum(coordinate in g[1] for g in grids.values()) != 1
        for coordinate in (target, left, right, previous)
    ):
        return False
    local = {r["name"]: r for r in payload["rules"]}
    right_rule = local.get(subtraction.right.id)
    if right_rule is None or subtraction.left.id in local:
        return False
    candidates = [r for r in records if r.export_name == subtraction.left.id]
    if len(candidates) != 1:
        return False
    record = candidates[0]
    if (
        record.import_uri.rsplit("#", 1)[0].replace(":policies/", "/policy/")
        != provider_citation
    ):
        return False
    try:
        provider = _payload(record.content.encode())
    except (ValueError, TypeError, yaml.YAMLError):
        return False
    provider_module = provider.get("module")
    pv = (
        provider_module.get("source_verification")
        if isinstance(provider_module, dict)
        else None
    )
    rr = provider.get("rules")
    if (
        not isinstance(pv, dict)
        or pv.get("corpus_citation_path") != provider_citation
        or not isinstance(rr, list)
        or not all(isinstance(r, dict) for r in rr)
    ):
        return False
    exported = [r for r in rr if r.get("name") == record.export_name]
    if len(exported) != 1:
        return False
    left_rule = exported[0]
    owner_versions = owner.get("versions", [])
    frames = [
        _frame(owner, i, version, context) for i, version in enumerate(owner_versions)
    ]
    if not frames or None in frames or len(set(frames)) != len(frames):
        return False
    for rule, expected_citation, row, expected_body in (
        (left_rule, provider_citation, provider_rows[previous], provider_body),
        (right_rule, citation, rows[right], body),
    ):
        versions = rule.get("versions")
        if (
            not isinstance(versions, list)
            or len(versions) != len(frames)
            or not all(isinstance(v, dict) for v in versions)
        ):
            return False
        if (
            rule.get("dtype"),
            rule.get("entity"),
            rule.get("period"),
            rule.get("unit"),
        ) != ("Money", "Person", "Year", owner.get("unit")):
            return False
        for i, version in enumerate(versions):
            if _frame(rule, i, version, context) != frames[i]:
                return False
            atoms = _atoms(rule, f"versions[{i}].formula", "formula")
            source = _source(atoms[0], context) if len(atoms) == 1 else None
            if source is None:
                return False
            row_citation, row_body, excerpt, start = source
            if rule is right_rule:
                if excerpt != row.group():
                    return False
            else:
                operative = re.fullmatch(
                    r"Line [1-9]\d* and (?:line )?[1-9]\d*: Line (?P<a>[1-9]\d*) minus line (?P<b>[1-9]\d*) "
                    r'\(if negative, enter "0"\)',
                    excerpt,
                )
                if operative is None or any(
                    int(operative[k]) not in provider_rows for k in ("a", "b")
                ):
                    return False
            if (
                row_citation != expected_citation
                or row_body != expected_body
                or start != row.start()
                or not row.group().startswith(excerpt)
                or len(excerpt) <= row.group().index(":") + 1
            ):
                return False
    return True


def witnesses(
    evidence: SameCaseEvidence,
    branch: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
    imported_exports: Sequence[ResolvedImportedExport] = (),
    source_context: Mapping[str, str | None] | None = None,
    **ordinary_context: Any,
) -> set[tuple[str, str, str]]:
    from . import source_completeness as s

    if (
        not content
        or not evidence.bound(content, imported_exports)
        or source_text[branch.start : branch.end] != branch.text
    ):
        return set()
    if not _FLOOR.fullmatch(branch.text.strip()) or source_text.count(branch.text) != 1:
        return set()
    try:
        payload = _payload(content.encode())
        artifact = json.loads(evidence.artifact_bytes)
    except (ValueError, TypeError, yaml.YAMLError):
        return set()
    source_rules, inputs = payload.get("rules"), payload.get("inputs", [])
    if (
        not isinstance(source_rules, list)
        or not isinstance(inputs, list)
        or not all(
            isinstance(r, dict) and isinstance(r.get("name"), str)
            for r in [*source_rules, *inputs]
        )
    ):
        return set()
    source_by_name = {r["name"]: r for r in source_rules}
    if len(source_by_name) != len(source_rules) or any(
        source_by_name.get(n) != r for n, r in principal_rules.items()
    ):
        return set()
    imports = payload.get("imports", [])
    if any(ref.count("#") != 1 or not all(ref.split("#")) for ref in imports):
        return set()
    aliases = [ref.rsplit("#", 1)[1] for ref in imports]
    local_names = {r["name"] for r in [*source_rules, *inputs]}
    if len(set(aliases)) != len(aliases) or set(aliases) & local_names:
        return set()
    if not isinstance(artifact, dict) or not isinstance(artifact.get("program"), dict):
        return set()
    declarations = artifact["program"].get("derived")
    if not isinstance(declarations, list) or not all(
        isinstance(d, dict) for d in declarations
    ):
        return set()
    owners = []
    for name, rule in principal_rules.items():
        if branch.required_principal_output not in (None, name):
            continue
        if (rule.get("dtype"), rule.get("entity"), rule.get("period")) != (
            "Money",
            "Person",
            "Year",
        ) or not rule.get("unit"):
            continue
        versions = rule.get("versions")
        metadata = rule.get("metadata")
        proof = metadata.get("proof") if isinstance(metadata, dict) else None
        atoms = proof.get("atoms") if isinstance(proof, dict) else None
        if (
            not isinstance(versions, list)
            or not versions
            or not all(isinstance(v, dict) for v in versions)
        ):
            continue
        if not isinstance(atoms, list) or not all(
            isinstance(a, dict) and isinstance(a.get("source"), dict) for a in atoms
        ):
            continue
        shapes = [_subtraction(str(v.get("formula", ""))) for v in versions]
        if (
            any(shape is None for shape in shapes)
            or len({ast.dump(shape) for shape in shapes}) != 1
        ):
            continue
        if not all(
            any(
                atom.get("path") == f"versions[{i}].formula"
                and atom.get("kind") == "formula"
                and atom.get("source", {}).get("corpus_citation_path")
                == corpus_citation_path
                and atom.get("source", {}).get("excerpt") == branch.text
                for atom in atoms
                if isinstance(atom, dict)
            )
            for i in range(len(versions))
        ):
            continue
        owners.append((name, rule, shapes[0]))
    if len(owners) != 1:
        return set()
    name, rule, subtraction = owners[0]
    if not _ordered_operands(
        evidence,
        payload,
        content,
        rule,
        subtraction,
        branch,
        corpus_citation_path,
        source_text,
        source_context,
        imported_exports,
    ):
        return set()
    module = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    refs = {ref.rsplit("#", 1)[1]: ref for ref in imports}
    refs.update({n: module + "#" + n for n in principal_rules})
    for operand in (name, subtraction.left.id, subtraction.right.id):
        matches = [d for d in declarations if d.get("id") == refs.get(operand)]
        if len(matches) != 1 or any(
            matches[0].get(k) != expected
            for k, expected in {
                "entity": "Person",
                "period": "Year",
                "dtype": "decimal",
                "unit": rule["unit"],
                "semantics": "scalar",
            }.items()
        ):
            return set()
    executions = []
    for case in asserted_by_rule.get(name, ()):
        query = _query(evidence, case)
        if query is None:
            continue
        values = evidence.values(case, content)
        if values is None:
            continue
        imported, observed = values
        if imported.keys() & formula_environment.keys() or name not in observed:
            continue
        ordinary_witness = formula_witnesses(
            evidence,
            branch,
            content=content,
            source_text=source_text,
            corpus_citation_path=corpus_citation_path,
            principal_rules=principal_rules,
            rule_names={name},
            asserted_by_rule={name: [case]},
            formula_environment=formula_environment,
            **ordinary_context,
        )
        environment = {**formula_environment, **imported}
        dependencies = s._case_dependency_environment(
            principal_rules,
            case,
            formula_environment=environment,
            require_asserted_value=False,
        )
        if name not in dependencies or any(
            key not in observed
            or not s._formula_runtime_values_equal(value, observed[key])
            or not s._asserted_formula_runtime_values_equal(
                principal_rules[key],
                value,
                s._test_case_asserted_output_value(case, key),
            )
            for key, value in dependencies.items()
        ):
            continue
        actual = {**environment, **dependencies}
        operands = [actual.get(subtraction.left.id), actual.get(subtraction.right.id)]
        if any(
            isinstance(v, bool) or not isinstance(v, (int, Decimal)) for v in operands
        ):
            continue
        difference = Decimal(operands[0]) - Decimal(operands[1])
        if difference > 0 and not ordinary_witness:
            continue
        result = observed[name]
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, Decimal))
            or result != max(Decimal(0), difference)
        ):
            continue
        executions.append((case, query, difference))
    result = set()
    for (left, lq, ld), (right, rq, rd) in combinations(executions, 2):
        if not ((ld < 0 < rd) or (rd < 0 < ld)):
            continue
        if _explicit_identity(left) != _explicit_identity(right) or any(
            lq.get(k) != rq.get(k) for k in ("period", "assessment_date")
        ):
            continue
        if (
            left.get("output", {}).keys() != right.get("output", {}).keys()
            or _changed_primitive(evidence, left, right) != rule["unit"]
        ):
            continue
        result.add((name, *sorted((case_key(left), case_key(right)))))
    return result


def _same_page_floor_shape(formula: str) -> tuple[str | None, str, str] | None:
    """Extract a local floor; a conditional must have exactly one named gate.

    This is syntax only. The gate still needs source binding and actual selected
    leaf verification; a false-arm zero is never floor execution evidence.
    """
    direct = _subtraction(formula)
    if direct is not None:
        return None, direct.left.id, direct.right.id
    try:
        statements = ast.parse(formula.strip(), mode="exec").body
    except (SyntaxError, ValueError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], ast.If):
        return None
    conditional = statements[0]
    if (
        not isinstance(conditional.test, ast.Name)
        or len(conditional.body) != 1
        or len(conditional.orelse) != 1
        or not isinstance(conditional.body[0], ast.Expr)
        or not isinstance(conditional.orelse[0], ast.Expr)
    ):
        return None
    otherwise = conditional.orelse[0].value
    if (
        not isinstance(otherwise, ast.Constant)
        or type(otherwise.value) is not int
        or otherwise.value != 0
    ):
        return None
    subtraction = _subtraction(ast.unparse(conditional.body[0].value))
    if subtraction is None:
        return None
    return conditional.test.id, subtraction.left.id, subtraction.right.id


def _local_sum_names(formula: str, count: int) -> tuple[str, ...] | None:
    try:
        expression = ast.parse(formula.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    pending, names = [expression], []
    while pending:
        node = pending.pop()
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            pending.extend((node.right, node.left))
        elif isinstance(node, ast.Name):
            names.append(node.id)
        else:
            return None
    return tuple(names) if len(names) == count == len(set(names)) else None


def bind_same_page_floors(
    content: str, source_context: Mapping[str, str | None]
) -> dict[int, dict[str, Any]]:
    """Identify immediate floor operands, not correctness of upstream returns.

    No witness is issued here. In particular, the broad lines 8..11 excerpt
    does not authenticate the individual upstream line projections. Their
    ordinary obligations remain; a consumer may allocate only floors 12/13.
    """
    try:
        payload = _payload(content.encode())
    except (ValueError, TypeError, yaml.YAMLError):
        return {}
    module = payload.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if not isinstance(verification, dict) or payload.get("imports", []) != []:
        return {}
    citation = verification.get("corpus_citation_path")
    if citation != "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts":
        return {}
    body = source_context.get(citation)
    if (
        not isinstance(body, str)
        or verification.get("source_sha256")
        != hashlib.sha256(body.encode()).hexdigest()
    ):
        return {}
    structure = recognize_printed_rows(body, allow_detached=True)
    if structure is None or len(structure.same_page_floors) != 2:
        return {}
    rules = payload.get("rules")
    if not isinstance(rules, list) or not all(
        isinstance(r, dict) and isinstance(r.get("name"), str) for r in rules
    ):
        return {}
    by_name = {r["name"]: r for r in rules}
    if len(by_name) != len(rules):
        return {}
    header = body.split("\n", 1)[0]
    floors = {f.target: f for f in structure.same_page_floors}
    spans = {
        6: floors[13].operand_spans[0][-1],
        7: floors[12].operand_spans[0][0],
        11: floors[12].operand_spans[1][0],
        12: floors[12].operation_span,
        13: floors[13].operation_span,
    }
    bound: dict[int, dict[str, Any]] = {}
    for row, (start, end) in spans.items():
        found = []
        for rule in rules:
            if tuple(
                rule.get(k) for k in ("kind", "dtype", "entity", "period", "unit")
            ) != ("derived", "Money", "Person", "Year", "CAD"):
                continue
            versions = rule.get("versions")
            if (
                not isinstance(versions, list)
                or len(versions) != 1
                or not isinstance(versions[0], dict)
            ):
                continue
            version = versions[0]
            if any(
                version.get(k) != v
                for k, v in (
                    ("effective_from", "2025-01-01"),
                    ("effective_to", "2025-12-31"),
                )
            ):
                continue
            if any(
                len(atoms := _atoms(rule, f"versions[0].{field}", "effective_period"))
                != 1
                or _source(atoms[0], source_context) != (citation, body, header, 0)
                for field in ("effective_from", "effective_to")
            ):
                continue
            owned = []
            for atom in _atoms(rule, "versions[0].formula", "formula"):
                source = _source(atom, source_context)
                if source is None or source[:2] != (citation, body):
                    continue
                _, _, excerpt, offset = source
                # The addition may live inside its complete detached chart.
                if row == 6:
                    matches = excerpt == body[start:end]
                elif row == 11:
                    chart_start = next(
                        a
                        for a, b in structure.ownership_spans
                        if body[a:b].startswith("Amount from line 30000")
                    )
                    chart_end = next(
                        b
                        for a, b in structure.ownership_spans
                        if body[a:b].startswith("+\n=\n\n10")
                    )
                    chart = body[chart_start:chart_end]
                    # The proof ends at the printed 11 cell; the trailing
                    # transfer arrow is layout, not an extra arithmetic row.
                    cell_end = chart.index("\n\n11\n") + len("\n\n11\n")
                    chart = chart[:cell_end]
                    matches = offset == chart_start and excerpt == chart
                else:
                    expected = body[start:end].split("\n\n", 1)[0]
                    if row in (7, 12):
                        expected = expected.split("\n", 1)[1]
                    matches = excerpt == expected and start <= offset < end
                if matches:
                    owned.append(source)
            if len(owned) != 1:
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            if row in (6, 11):
                children = _local_sum_names(formula, 5 if row == 6 else 3)
                if children is None or any(
                    n not in by_name or by_name[n].get("kind") != "derived"
                    for n in children
                ):
                    continue
                invalid_children = False
                for name in children:
                    child = by_name[name]
                    child_versions = child.get("versions")
                    if (
                        not isinstance(child_versions, list)
                        or len(child_versions) != 1
                        or not isinstance(child_versions[0], dict)
                        or tuple(
                            child.get(k) for k in ("dtype", "entity", "period", "unit")
                        )
                        != ("Money", "Person", "Year", "CAD")
                        or any(
                            child_versions[0].get(k) != version[k]
                            for k in ("effective_from", "effective_to")
                        )
                    ):
                        invalid_children = True
                        break
                    if row == 11 and (
                        not isinstance(child_versions[0].get("formula"), str)
                        or not re.fullmatch(
                            r"[A-Za-z_][A-Za-z0-9_]*", child_versions[0]["formula"]
                        )
                    ):
                        invalid_children = True
                        break
                if invalid_children:
                    continue
            elif row in (12, 13):
                shape = _same_page_floor_shape(formula)
                if shape is None or (shape[0] is None) != (row == 12):
                    continue
            found.append(rule)
        if len(found) != 1:
            return {}
        bound[row] = found[0]
    if len({r["name"] for r in bound.values()}) != len(bound):
        return {}
    for target, floor in floors.items():
        shape = _same_page_floor_shape(bound[target]["versions"][0]["formula"])
        if shape[1:] != (bound[floor.left]["name"], bound[floor.right]["name"]):
            return {}
    selector = _same_page_floor_shape(bound[13]["versions"][0]["formula"])[0]
    gate = by_name.get(selector)
    if not isinstance(gate, dict) or tuple(
        gate.get(k) for k in ("kind", "dtype", "entity", "period")
    ) != ("derived", "Judgment", "Person", "Year"):
        return {}
    if gate.get("unit") is not None:
        return {}
    if any(
        len(atoms := _atoms(gate, f"versions[0].{field}", "effective_period")) != 1
        or _source(atoms[0], source_context) != (citation, body, header, 0)
        for field in ("effective_from", "effective_to")
    ):
        return {}
    gate_versions = gate.get("versions")
    if (
        not isinstance(gate_versions, list)
        or len(gate_versions) != 1
        or not isinstance(gate_versions[0], dict)
    ):
        return {}
    if any(
        gate_versions[0].get(k) != v
        for k, v in (("effective_from", "2025-01-01"), ("effective_to", "2025-12-31"))
    ):
        return {}
    scope_start = body.index("Complete this schedule")
    separation_start = body.index("If you were separated")
    separation_end = body.index("If your spouse or common-law partner is filing")
    required_gate_excerpts = {
        body[scope_start:separation_start],
        body[separation_start:separation_end],
    }
    actual_gate_excerpts = {
        source[2]
        for atom in _atoms(gate, "versions[0].formula", "formula")
        if (source := _source(atom, source_context)) is not None
        and source[:2] == (citation, body)
    }
    if not required_gate_excerpts <= actual_gate_excerpts:
        return {}
    # Only ownership is bound here. The witness must replay this exact gate,
    # corroborate its typed result and prove the selected floor leaf reached.
    return bound


def _same_page_case(
    evidence: SameCaseEvidence, case: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Rebuild ordinary transport from the exact saved companion ordinal."""
    from .validator_pipeline import ValidatorPipeline, _safe_load_unique_keys

    query = _query(evidence, case)
    if query is None or not _explicit_identity(case):
        return None
    matches = [r for r in evidence.records if r.case_sha256 == case_key(case)]
    if len(matches) != 1:
        return None
    record = matches[0]
    companion_path = evidence.root_relative.removesuffix(".yaml") + ".test.yaml"
    companions = [f for f in evidence.files if f.relative == companion_path]
    if len(companions) != 1:
        return None
    cases = _safe_load_unique_keys(companions[0].raw.decode())
    if (
        not isinstance(cases, list)
        or not 0 < record.case_index <= len(cases)
        or case_key(cases[record.case_index - 1]) != case_key(case)
        or sum(case_key(c) == case_key(case) for c in cases) != 1
    ):
        return None
    inputs, outputs = case.get("input"), case.get("output")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return None
    module = evidence.snapshot_module(evidence.root_relative.removesuffix(".yaml"))
    declarations = payload.get("inputs", [])
    if not isinstance(declarations, list) or not all(
        isinstance(d, dict) for d in declarations
    ):
        return None
    declared = {
        module + "#input." + d["name"]: d
        for d in declarations
        if isinstance(d.get("name"), str)
    }
    if len(declared) != len(declarations) or not inputs.keys() <= declared.keys():
        return None
    if any(declared[k].get("entity") != "Person" for k in inputs):
        return None
    if (
        not isinstance(query.get("outputs"), list)
        or len(query["outputs"]) != len(outputs)
        or set(query["outputs"]) != outputs.keys()
        or query.get("assessment_date") is not None
    ):
        return None
    pipe = object.__new__(ValidatorPipeline)
    pipe.policy_repo_path = (
        evidence.snapshot_root / evidence.root_relative.split("/", 1)[0]
    )
    dataset = pipe._build_rulespec_dataset(
        inputs,
        period=query["period"],
        query_entity="Person",
        query_entity_id=query["entity_id"],
        require_legal_input_keys=True,
        module_target=module,
        input_entities={k: "Person" for k in declared},
    )
    request = json.loads(record.request_json)
    if json.dumps(dataset, sort_keys=True) != json.dumps(
        request.get("dataset"), sort_keys=True
    ):
        return None
    return query, json.loads(record.response_json)["results"][0]


def same_page_witnesses(
    evidence: SameCaseEvidence,
    branch: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
    source_context: Mapping[str, str | None] | None = None,
) -> set[tuple[str, str, str]]:
    """Credit only one exact local floor from two fresh ordinary executions."""
    from . import source_completeness as s

    if corpus_citation_path != "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts":
        return set()
    if source_context is None or not evidence.bound(content, ()):
        return set()
    if source_context.get(corpus_citation_path) != source_text:
        return set()
    bound = bind_same_page_floors(content, source_context)
    if not bound:
        return set()
    structure = recognize_printed_rows(source_text, allow_detached=True)
    matching = [
        f
        for f in structure.same_page_floors
        if (
            (branch.start, branch.end) == f.operation_span
            and branch.text == source_text[slice(*f.operation_span)]
            and branch.kind == "exception-clause"
        )
    ]
    if len(matching) != 1:
        return set()
    floor = matching[0]
    owner = bound[floor.target]
    name = owner["name"]
    if branch.required_principal_output not in (None, name):
        return set()
    payload = _payload(content.encode())
    local = {r["name"]: r for r in payload["rules"] if r.get("kind") == "derived"}
    if any(local.get(n) != r for n, r in principal_rules.items()) or any(
        n not in principal_rules
        for n in (name, bound[floor.left]["name"], bound[floor.right]["name"])
    ):
        return set()
    try:
        artifact = json.loads(evidence.artifact_bytes)
    except (ValueError, TypeError):
        return set()
    if not isinstance(artifact, dict) or not isinstance(artifact.get("program"), dict):
        return set()
    declarations = artifact["program"].get("derived")
    if not isinstance(declarations, list) or not all(
        isinstance(d, dict) for d in declarations
    ):
        return set()
    module = evidence.snapshot_module(evidence.root_relative.removesuffix(".yaml"))
    for r in (owner, bound[floor.left], bound[floor.right]):
        declarations_for_rule = [
            d for d in declarations if d.get("id") == module + "#" + r["name"]
        ]
        if len(declarations_for_rule) != 1 or any(
            declarations_for_rule[0].get(k) != v
            for k, v in {
                "entity": "Person",
                "period": "Year",
                "dtype": "decimal",
                "unit": "CAD",
                "semantics": "scalar",
            }.items()
        ):
            return set()
    shape = _same_page_floor_shape(owner["versions"][0]["formula"])
    valid = []
    for case in asserted_by_rule.get(name, ()):
        try:
            transport = _same_page_case(evidence, case, payload)
            values = evidence.values(case, content)
            if transport is None or values is None or values[0]:
                continue
            query, response = transport
            observed = values[1]
            replay = s._case_dependency_environment(
                principal_rules,
                case,
                formula_environment=formula_environment,
                require_asserted_value=False,
            )
            execution = s._case_formula_execution(
                owner,
                case,
                formula_environment=formula_environment,
                dependency_environment=replay,
            )
            if execution is None:
                continue
            reached_floor = _subtraction(execution.leaf)
            if (
                reached_floor is None
                or (reached_floor.left.id, reached_floor.right.id) != shape[1:]
            ):
                continue
            if shape[0] is not None and (
                replay.get(shape[0]) is not True or observed.get(shape[0]) is not True
            ):
                continue
            environment = s._case_formula_identifier_environment(
                case,
                formula_environment=formula_environment,
                dependency_environment=replay,
            )
            if environment is None:
                continue
            pending, visited, coherent = [name], set(), True
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                if (
                    current not in replay
                    or current not in observed
                    or not s._formula_runtime_values_equal(
                        replay[current], observed[current]
                    )
                ):
                    coherent = False
                    break
                child = s._case_formula_execution(
                    principal_rules[current],
                    case,
                    formula_environment=formula_environment,
                    dependency_environment=replay,
                )
                if child is None:
                    coherent = False
                    break
                reached = s._reached_formula_expression_identifier_names(
                    child.leaf, environment=environment
                )
                for step in child.trace:
                    for selector in step.selectors:
                        reached.update(
                            s._reached_formula_expression_identifier_names(
                                selector, environment=environment
                            )
                        )
                pending.extend(reached & principal_rules.keys() - visited)
            if not coherent:
                continue
            ref = module + "#" + name
            result, trace = (
                response.get("outputs", {}).get(ref),
                response.get("trace", {}).get(ref),
            )
            if (
                not isinstance(result, dict)
                or not isinstance(trace, dict)
                or any(
                    result.get(k) != trace.get(k)
                    for k in ("id", "kind", "dtype", "unit", "value")
                )
            ):
                continue
            if not s._asserted_formula_runtime_values_equal(
                owner, observed[name], s._test_case_asserted_output_value(case, name)
            ):
                continue
            operands = [replay.get(n) for n in shape[1:]]
            if any(
                isinstance(v, bool) or not isinstance(v, (int, Decimal))
                for v in operands
            ):
                continue
            difference = Decimal(operands[0]) - Decimal(operands[1])
            if observed[name] != max(Decimal(0), difference):
                continue
            valid.append((case, query, difference))
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            yaml.YAMLError,
        ):
            continue
    witnesses = set()
    for (left, lq, ld), (right, rq, rd) in combinations(valid, 2):
        if not (ld < 0 < rd or rd < 0 < ld):
            continue
        if _explicit_identity(left) != _explicit_identity(right) or any(
            lq.get(k) != rq.get(k) for k in ("entity_id", "period", "assessment_date")
        ):
            continue
        if (
            set(lq["outputs"]) != set(rq["outputs"])
            or _changed_primitive(evidence, left, right) != "CAD"
        ):
            continue
        witnesses.add(
            (
                f"same-page-floor:{floor.target}:{branch.start}:{name}",
                *sorted((case_key(left), case_key(right))),
            )
        )
    # A printed floor owns one obligation even when several pairs prove it.
    return {min(witnesses)} if witnesses else set()


def bind_same_page_choices(
    content: str, source_context: Mapping[str, str | None]
) -> dict[int, dict[str, Any]]:
    """Bind closed local choices; this alone issues no execution witness."""
    floors = bind_same_page_floors(content, source_context)
    if not floors:
        return {}
    payload = _payload(content.encode())
    citation = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"
    body = source_context[citation]
    header = body.split("\n", 1)[0]
    rules = {r["name"]: r for r in payload["rules"]}
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or not all(
        isinstance(i, dict) and isinstance(i.get("name"), str) for i in inputs
    ):
        return {}
    facts = {i["name"]: i for i in inputs}
    if len(facts) != len(inputs) or facts.keys() & rules.keys():
        return {}
    age = (
        "Age amount (if your spouse or common-law partner was born in 1960 or earlier):\n"
        "Enter the amount from line 30100 of their return.\n\n35200\n\n1\n"
    )
    threshold = (
        "Enter the amount from line 26000 of their return on line 7 if it is $57,375 or less.\n"
        "If it is more than $57,375, enter the result of the following calculation:\n"
        "amount from line 76 of their return\n÷ 14.5% ="
    )
    parameter_excerpt = (
        "Spouse's or common-law partner's taxable income:\n" + threshold + "\n"
    )

    def proved(rule: Any, kind: str, dtype: str, excerpt: str, atom_kind: str) -> bool:
        if not isinstance(rule, dict) or tuple(
            rule.get(k) for k in ("kind", "dtype", "entity", "period", "unit")
        ) != (
            kind,
            dtype,
            "Person" if kind == "derived" else None,
            "Year",
            "CAD" if dtype == "Money" else None,
        ):
            return False
        if rule.get("rounding") is not None or any(
            k in rule for k in ("resolved_currency_rounding", "default", "formula")
        ):
            return False
        versions = rule.get("versions")
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            return False
        if not isinstance(versions[0].get("formula"), str):
            return False
        if tuple(
            str(versions[0].get(k)) for k in ("effective_from", "effective_to")
        ) != ("2025-01-01", "2025-12-31"):
            return False
        metadata = rule.get("metadata")
        proof = metadata.get("proof") if isinstance(metadata, dict) else None
        atoms = proof.get("atoms") if isinstance(proof, dict) else None
        if not isinstance(atoms, list) or not all(isinstance(a, dict) for a in atoms):
            return False
        for suffix, expected_kind, text in (
            ("formula", atom_kind, excerpt),
            ("effective_from", "effective_period", header),
            ("effective_to", "effective_period", header),
        ):
            found = [a for a in atoms if a.get("path") == "versions[0]." + suffix]
            if len(found) != 1 or found[0].get("kind") != expected_kind:
                return False
            source = found[0].get("source")
            if (
                not isinstance(source, dict)
                or source.get("corpus_citation_path") != citation
                or source.get("excerpt") != text
            ):
                return False
        return True

    def factual(name: str, dtype: str) -> bool:
        fact = facts.get(name)
        return (
            isinstance(fact, dict)
            and tuple(fact.get(k) for k in ("dtype", "entity", "period", "unit"))
            == (dtype, "Person", "Year", "CAD" if dtype == "Money" else None)
            and not any(
                k in fact for k in ("default", "formula", "versions", "indexed_by")
            )
        )

    bound = {}
    for row, excerpt, selector, amount, cutoff_value, cutoff_dtype, atom_kind in (
        (
            1,
            age,
            "spouse_birth_year",
            "spouse_age_amount_line_30100",
            1960,
            "Integer",
            "formula",
        ),
        (
            7,
            threshold,
            "spouse_taxable_income_line_26000",
            "spouse_return_line_76_amount",
            57375,
            "Money",
            "parameter",
        ),
    ):
        found = []
        for rule in rules.values():
            if not proved(rule, "derived", "Money", excerpt, "formula"):
                continue
            try:
                statements = ast.parse(rule["versions"][0]["formula"]).body
            except (SyntaxError, TypeError, ValueError):
                continue
            if len(statements) != 1 or not isinstance(statements[0], ast.If):
                continue
            choice = statements[0]
            test = choice.test
            if not (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == selector
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.LtE)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Name)
            ):
                continue
            cutoff = rules.get(test.comparators[0].id)
            if not proved(
                cutoff,
                "parameter",
                cutoff_dtype,
                age if row == 1 else parameter_excerpt,
                atom_kind,
            ):
                continue
            try:
                value = ast.literal_eval(cutoff["versions"][0]["formula"])
            except (SyntaxError, TypeError, ValueError):
                continue
            if type(value) is not int or value != cutoff_value:
                continue
            if not factual(selector, "Integer" if row == 1 else "Money") or not factual(
                amount, "Money"
            ):
                continue
            if (
                len(choice.body) != 1
                or len(choice.orelse) != 1
                or not isinstance(choice.body[0], ast.Expr)
                or not isinstance(choice.orelse[0], ast.Expr)
            ):
                continue
            yes, no = choice.body[0].value, choice.orelse[0].value
            if not isinstance(yes, ast.Name) or yes.id != (
                amount if row == 1 else selector
            ):
                continue
            if row == 1:
                if (
                    not isinstance(no, ast.Constant)
                    or type(no.value) is not int
                    or no.value != 0
                ):
                    continue
            else:
                if not (
                    isinstance(no, ast.BinOp)
                    and isinstance(no.op, ast.Div)
                    and isinstance(no.left, ast.Name)
                    and no.left.id == amount
                    and isinstance(no.right, ast.Name)
                ):
                    continue
                rate = rules.get(no.right.id)
                if not proved(
                    rate, "parameter", "Rate", parameter_excerpt, "parameter"
                ):
                    continue
                if rate["versions"][0].get("formula") != "0.145":
                    continue
                if rule != floors[7]:
                    continue
            found.append(rule)
        if len(found) != 1:
            return {}
        bound[row] = found[0]
    total_terms = _local_sum_names(floors[6]["versions"][0]["formula"], 5)
    if total_terms is None or total_terms.count(bound[1]["name"]) != 1:
        return {}
    return bound


def _same_page_choice_observation(
    evidence: SameCaseEvidence,
    case: dict[str, Any],
    *,
    content: str,
    source_context: Mapping[str, str | None],
    required_outputs: set[str],
    principal_rules: dict[str, dict[str, Any]],
    formula_environment: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Corroborate one ordinary captured case; never change its input facts."""
    from . import source_completeness as s

    if not evidence.bound(content, ()) or not bind_same_page_choices(
        content, source_context
    ):
        return None
    try:
        payload = _payload(content.encode())
        local = {r["name"]: r for r in payload["rules"] if r.get("kind") == "derived"}
        if (
            not required_outputs
            or local != principal_rules
            or not required_outputs <= local.keys()
        ):
            return None
        authentic_environment = s._constant_rule_environment(payload)
        if formula_environment != authentic_environment:
            return None
        formula_environment = authentic_environment
        transport = _same_page_case(evidence, case, payload)
        values = evidence.values(case, content)
        if transport is None or values is None or values[0]:
            return None
        query, response = transport
        observed = values[1]
        module = evidence.snapshot_module(evidence.root_relative.removesuffix(".yaml"))
        if not {module + "#" + n for n in required_outputs} <= set(query["outputs"]):
            return None
        artifact = json.loads(evidence.artifact_bytes)
        declarations = artifact["program"]["derived"]
        if not isinstance(declarations, list) or not all(
            isinstance(d, dict) for d in declarations
        ):
            return None
        replay = s._case_dependency_environment(
            principal_rules,
            case,
            formula_environment=formula_environment,
            require_asserted_value=False,
        )
        environment = s._case_formula_identifier_environment(
            case,
            formula_environment=formula_environment,
            dependency_environment=replay,
        )
        if environment is None:
            return None
        pending, visited, executions = list(required_outputs), set(), {}
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            rule = principal_rules[name]
            if (
                name not in observed
                or name not in replay
                or not s._formula_runtime_values_equal(replay[name], observed[name])
            ):
                return None
            ref = module + "#" + name
            declared = [d for d in declarations if d.get("id") == ref]
            dtype = (
                "decimal"
                if rule.get("dtype") == "Money"
                else "judgment"
                if rule.get("dtype") == "Judgment"
                else None
            )
            if (
                dtype is None
                or len(declared) != 1
                or any(
                    declared[0].get(k) != v
                    for k, v in {
                        "entity": "Person",
                        "period": "Year",
                        "dtype": dtype,
                        "unit": rule.get("unit"),
                        "semantics": "judgment" if dtype == "judgment" else "scalar",
                    }.items()
                )
            ):
                return None
            execution = s._case_formula_execution(
                rule,
                case,
                formula_environment=formula_environment,
                dependency_environment=replay,
            )
            if execution is None:
                return None
            executions[name] = execution
            reached = s._reached_formula_expression_identifier_names(
                execution.leaf, environment=environment
            )
            for step in execution.trace:
                for selector in step.selectors:
                    reached.update(
                        s._reached_formula_expression_identifier_names(
                            selector, environment=environment
                        )
                    )
            pending.extend(reached & principal_rules.keys() - visited)
            if name in required_outputs:
                output = response.get("outputs", {}).get(ref)
                trace = response.get("trace", {}).get(ref)
                if (
                    not isinstance(output, dict)
                    or not isinstance(trace, dict)
                    or any(
                        output.get(k) != trace.get(k)
                        for k in ("id", "kind", "dtype", "unit", "value", "outcome")
                    )
                ):
                    return None
                if not s._asserted_formula_runtime_values_equal(
                    rule, observed[name], s._test_case_asserted_output_value(case, name)
                ):
                    return None
        return query, observed, executions
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def same_page_choice_witnesses(
    evidence: SameCaseEvidence,
    branch: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    principal_rules: dict[str, dict[str, Any]],
    asserted_by_rule: dict[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any],
    source_context: Mapping[str, str | None] | None = None,
) -> set[tuple[str, str, str]]:
    """Prove closed source choices/composition from ordinary captured execution.

    Choice arm coverage is not a claim that the age/income cases differ in
    one fact. Only the unused-amount composition uses a one-fact causal pair.
    """
    from . import source_completeness as s

    citation = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"
    if (
        corpus_citation_path != citation
        or source_context is None
        or source_context.get(citation) != source_text
        or not evidence.bound(content, ())
        or branch.kind != "exception-clause"
    ):
        return set()
    bound = bind_same_page_choices(content, source_context)
    floors = bind_same_page_floors(content, source_context)
    if not bound or not floors:
        return set()
    excerpts = {
        1: "Age amount (if your spouse or common-law partner was born in 1960 or earlier):\n"
        "Enter the amount from line 30100 of their return.\n\n35200\n\n1",
        7: "Spouse's or common-law partner's taxable income:\n"
        "Enter the amount from line 26000 of their return on line 7 if it is $57,375 or less.\n"
        "If it is more than $57,375, enter the result of the following calculation:\n"
        "amount from line 76 of their return\n÷ 14.5% =",
        0: "You may be able to claim all or part of certain amounts that your spouse or "
        "common-law partner qualifies for if they do not need\n"
        "to use them to reduce their federal tax to zero.",
    }
    matches = [
        row
        for row, excerpt in excerpts.items()
        if source_text.count(excerpt) == 1
        and (branch.start, branch.end)
        == (source_text.index(excerpt), source_text.index(excerpt) + len(excerpt))
        and branch.text == excerpt
    ]
    if len(matches) != 1:
        return set()
    row = matches[0]
    final = floors[13]["name"]
    owner = bound[row]["name"] if row else final
    if branch.required_principal_output not in (None, owner):
        return set()
    shape = _same_page_floor_shape(floors[13]["versions"][0]["formula"])
    if shape is None or shape[0] is None:
        return set()
    gate = shape[0]
    valid = []
    for case in asserted_by_rule.get(final, ()):
        observation = _same_page_choice_observation(
            evidence,
            case,
            content=content,
            source_context=source_context,
            required_outputs={owner, final},
            principal_rules=principal_rules,
            formula_environment=formula_environment,
        )
        if observation is None:
            continue
        query, observed, executions = observation
        if (
            observed.get(gate) is not True
            or _subtraction(executions[final].leaf) is None
        ):
            continue
        environment = s._case_formula_identifier_environment(
            case,
            formula_environment=formula_environment,
            dependency_environment=observed,
        )
        if environment is None:
            continue
        if row:
            choice = ast.parse(bound[row]["versions"][0]["formula"]).body[0]
            selector = environment.get(choice.test.left.id)
            if isinstance(selector, bool) or not isinstance(selector, (int, Decimal)):
                continue
            try:
                leaf = ast.parse(executions[owner].leaf, mode="eval").body
            except (SyntaxError, TypeError, ValueError):
                continue
            if row == 1:
                label = (
                    "at" if selector == 1960 else "after" if selector == 1961 else None
                )
                if not (
                    (
                        label == "at"
                        and isinstance(leaf, ast.Name)
                        and observed[owner] > 0
                        and observed[final] > 0
                    )
                    or (
                        label == "after"
                        and isinstance(leaf, ast.Constant)
                        and leaf.value == 0
                        and observed[owner] == 0
                        and environment.get(choice.body[0].value.id) == 0
                    )
                ):
                    continue
            else:
                label = (
                    "below"
                    if selector < 57375
                    else "at"
                    if selector == 57375
                    else "above"
                )
                if not (
                    (label in {"below", "at"} and isinstance(leaf, ast.Name))
                    or (
                        label == "above"
                        and isinstance(leaf, ast.BinOp)
                        and isinstance(leaf.op, ast.Div)
                    )
                ):
                    continue
            valid.append((case, query, observed, label))
        else:
            valid.append((case, query, observed, None))
    if row:
        labels = {entry[3] for entry in valid}
        required = {"at", "after"} if row == 1 else {"below", "at", "above"}
        if not required <= labels or (
            row == 7
            and not any(
                label == "above" and observed[final] > 0
                for _, _, observed, label in valid
            )
        ):
            return set()
        keys = sorted({case_key(case) for case, _, _, _ in valid})
        # One token for this source obligation regardless of how many cases pass.
        return {
            (
                f"same-page-choice:{row}:{branch.start}:{owner}",
                keys[0],
                "|".join(keys[1:]),
            )
        }
    joint = {floors[n]["name"] for n in (6, 7, 11, 12, 13)} | {gate}
    if not any(
        _same_page_choice_observation(
            evidence,
            case,
            content=content,
            source_context=source_context,
            required_outputs=joint,
            principal_rules=principal_rules,
            formula_environment=formula_environment,
        )
        is not None
        for case, _, _, _ in valid
    ):
        return set()
    threshold_choice = ast.parse(bound[7]["versions"][0]["formula"]).body[0]
    selector = threshold_choice.test.left.id
    witnesses = set()
    for left, right in combinations(valid, 2):
        lc, lq, lv, _ = left
        rc, rq, rv, _ = right
        if (
            _explicit_identity(lc) != _explicit_identity(rc)
            or any(
                lq.get(k) != rq.get(k)
                for k in ("entity_id", "period", "assessment_date")
            )
            or set(lq["outputs"]) != set(rq["outputs"])
            or _changed_primitive(evidence, lc, rc) != "CAD"
        ):
            continue
        changed = [k for k in lc["input"] if lc["input"][k] != rc["input"][k]]
        if len(changed) != 1 or not changed[0].endswith("#input." + selector):
            continue
        six, twelve = floors[6]["name"], floors[12]["name"]
        if any(n not in values for values in (lv, rv) for n in (six, twelve, final)):
            continue
        positive, blocked = (lv, rv) if lv[final] > 0 else (rv, lv)
        if not (
            positive[final] > 0
            and positive[twelve] == 0
            and positive[final] == positive[six] == blocked[six]
            and blocked[final] == 0
            and blocked[twelve] > blocked[six]
        ):
            continue
        witnesses.add(
            (
                f"same-page-unused:{branch.start}:{final}",
                *sorted((case_key(lc), case_key(rc))),
            )
        )
    return {min(witnesses)} if witnesses else set()
