"""Exact printed bracket notation represented by signed numeric results.

This certificate does not claim a rendered form or a changed entitlement. It
requires separate actual negative and nonnegative signed arithmetic observations.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.printed_source_rows import recognize_printed_rows

from .actual_floor_evidence import _ROW, _explicit_identity, _query
from .imported_export_evidence import (
    _URI,
    ResolvedImportedExport,
    _atoms,
    _canonical_record,
    _frame,
    _source,
)
from .same_case_execution import SameCaseEvidence, _payload, case_key, formula_witnesses

_FORMAT = re.compile(
    r"Line (?P<target>[1-9]\d*) and (?:line )?[1-9]\d*: Line (?P<left>[1-9]\d*) minus line (?P<right>[1-9]\d*) "
    r"\(if negative, show in brackets\) (?P<caption>[A-Za-z][A-Za-z ]*)\s*=\s*\^"
)


@dataclass(frozen=True)
class SignedFormatObservation:
    owner: str
    case_sha256: str
    case_index: int
    value: Decimal
    source_citation: str
    canonical_operands: tuple[str, str]
    period_json: str
    source_sha256: str
    source_span: tuple[int, int]
    consumer_sha256: str
    artifact_sha256: str


def _subtraction(formula: Any) -> ast.BinOp | None:
    if not isinstance(formula, str):
        return None
    try:
        expression = ast.parse(formula.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.Sub)
        and isinstance(expression.left, ast.Name)
        and isinstance(expression.right, ast.Name)
        and expression.left.id != expression.right.id
    ):
        return None
    return expression


def _exact_source(atom: Mapping[str, Any], context: Mapping[str, str | None]):
    # The ordinary atom resolver does not authenticate optional span/hash fields.
    # This bounded certificate accepts only its fully supported two-field shape.
    source = atom.get("source")
    if not isinstance(source, dict) or set(source) != {
        "corpus_citation_path",
        "excerpt",
    }:
        return None
    return _source(atom, context)


def _closed_frame(rule, index, version, context):
    for field in ("effective_from", "effective_to"):
        atoms = _atoms(rule, f"versions[{index}].{field}", "effective_period")
        if len(atoms) != 1 or _exact_source(atoms[0], context) is None:
            return None
    return _frame(rule, index, version, context)


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
    match = _FORMAT.fullmatch(branch.text.strip())
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
        _closed_frame(owner, i, version, context)
        for i, version in enumerate(owner_versions)
    ]
    if (
        not frames
        or None in frames
        or len(set(frames)) != len(frames)
        or any(frame[1].rsplit("/", 1)[0] != family for frame in frames)
    ):
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
        if rule.get("kind") != "derived":
            return False
        if (
            rule.get("dtype"),
            rule.get("entity"),
            rule.get("period"),
            rule.get("unit"),
        ) != ("Money", "Person", "Year", owner.get("unit")):
            return False
        for i, version in enumerate(versions):
            if _closed_frame(rule, i, version, context) != frames[i]:
                return False
            atoms = _atoms(rule, f"versions[{i}].formula", "formula")
            source = _exact_source(atoms[0], context) if len(atoms) == 1 else None
            if source is None:
                return False
            row_citation, row_body, excerpt, start = source
            # Both operands own complete printed rows. The imported source row
            # is an ordinary computation, not the floor-specific prefix route.
            if excerpt != row.group():
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
    source_branches: Sequence[Any] = (),
    **ordinary_context: Any,
) -> set[tuple[str, str, str]]:
    from . import source_completeness as s

    if (
        not content
        or not evidence.bound(content, imported_exports)
        or source_text[branch.start : branch.end] != branch.text
    ):
        return set()
    match = _FORMAT.fullmatch(branch.text.strip())
    if match is None or source_text.count(branch.text) != 1:
        return set()
    if re.search(
        r"\b(?:if|unless|when|only|provided|except|otherwise|eligible|eligibility|must|may|shall|required|subject|where|while|until)\b",
        match["caption"],
        re.IGNORECASE,
    ):
        return set()
    structure = s._unconflicted_printed_rows(source_text, source_branches)
    if structure is None or (branch.start, branch.end) not in structure.ownership_spans:
        return set()
    labels = s.bare_outline(source_text) or ()
    boundaries = {
        *(point for b in source_branches for point in (b.start, b.end)),
        *(point for label in labels for point in (label.start, label.marker_end)),
        *(start for start, _ in s.bare_definition_headers(source_text, labels)),
    }
    if any(branch.start < point < branch.end for point in boundaries):
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
    names = [r["name"] for r in [*source_rules, *inputs]]
    if len(names) != len(set(names)):
        return set()
    if len(source_by_name) != len(source_rules) or any(
        source_by_name.get(n) != r for n, r in principal_rules.items()
    ):
        return set()
    imports = payload.get("imports", [])
    if not isinstance(imports, list) or any(
        not isinstance(ref, str) or _URI.fullmatch(ref) is None for ref in imports
    ):
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
        if rule.get("kind") != "derived":
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
        shapes = [_subtraction(v.get("formula")) for v in versions]
        if (
            any(shape is None for shape in shapes)
            or len({ast.dump(shape) for shape in shapes}) != 1
        ):
            continue
        source_proofs = []
        for i in range(len(versions)):
            own_atoms = _atoms(rule, f"versions[{i}].formula", "formula")
            source = (
                _exact_source(own_atoms[0], source_context or {})
                if len(own_atoms) == 1
                else None
            )
            if source is None:
                break
            citation, body, excerpt, start = source
            if (
                citation != corpus_citation_path
                or body.strip() != source_text
                or excerpt != branch.text
                or start != len(body) - len(body.lstrip()) + branch.start
            ):
                break
            source_proofs.append(source)
        if len(source_proofs) != len(versions):
            continue
        owners.append((name, rule, shapes[0]))
    if len(owners) != 1:
        return set()
    name, rule, subtraction = owners[0]
    units = payload.get("units")
    if (
        not isinstance(units, list)
        or not all(
            isinstance(u, dict) and isinstance(u.get("name"), str) for u in units
        )
        or len({u["name"] for u in units}) != len(units)
        or len(
            [
                u
                for u in units
                if isinstance(u, dict)
                and u.get("name") == rule["unit"]
                and u.get("kind") == "currency"
            ]
        )
        != 1
    ):
        return set()
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
        if not ordinary_witness:
            continue
        result = observed[name]
        if (
            isinstance(result, bool)
            or not isinstance(result, (int, Decimal))
            or result != difference
        ):
            continue
        # A reached leaf must still be the exact selected signed subtraction.
        execution = s._case_formula_execution(rule, case, formula_environment=actual)
        if (
            execution is None
            or execution.trace
            or _subtraction(execution.leaf) is None
            or ast.dump(_subtraction(execution.leaf)) != ast.dump(subtraction)
        ):
            continue
        record = next(r for r in evidence.records if r.case_sha256 == case_key(case))
        observation = SignedFormatObservation(
            name,
            case_key(case),
            record.case_index,
            difference,
            corpus_citation_path,
            (refs[subtraction.left.id], refs[subtraction.right.id]),
            json.dumps(query.get("period"), sort_keys=True),
            hashlib.sha256(source_text.encode()).hexdigest(),
            (branch.start, branch.end),
            hashlib.sha256(content.encode()).hexdigest(),
            hashlib.sha256(evidence.artifact_bytes).hexdigest(),
        )
        executions.append((case, query, observation))
    result = set()
    for (left, lq, ld), (right, rq, rd) in combinations(executions, 2):
        if not ((ld.value < 0 <= rd.value) or (rd.value < 0 <= ld.value)):
            continue
        if _explicit_identity(left) != _explicit_identity(right) or any(
            lq.get(k) != rq.get(k) for k in ("period", "assessment_date")
        ):
            continue
        result.add((name, *sorted((case_key(left), case_key(right)))))
    return result
