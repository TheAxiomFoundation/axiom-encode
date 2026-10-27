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
from .same_case_execution import SameCaseEvidence, _payload, case_key, formula_witnesses

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
    if request.get("mode") != "explain" or response.get("metadata") != {
        "requested_mode": "explain",
        "actual_mode": "explain",
        "fallback_reason": None,
    }:
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
