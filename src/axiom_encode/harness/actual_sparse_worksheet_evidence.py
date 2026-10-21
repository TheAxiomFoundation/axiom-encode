"""Actual sparse worksheet pairs for one closed, source-owned choice.

This is neither a general sparse-case matcher nor evidence of worksheet math.
The existing consumer-scoped entered-observation binder is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from axiom_encode.printed_source_rows import recognize_printed_rows

from .actual_floor_evidence import _explicit_identity, _query
from .imported_export_evidence import (
    _URI,
    ResolvedImportedExport,
    _atoms,
    _canonical_record,
    _entered_roles,
    _frame,
    _named,
    _payload,
    _source,
)
from .same_case_execution import SameCaseEvidence, case_key

_CLAUSE = re.compile(
    r"Line [1-9]\d* and line (?P<result>[1-9]\d*): [A-Z][A-Za-z ]+ "
    r"\(enter the amount from line (?P<amount>[1-9]\d*) "
    r"if you did not enter an amount on line (?P<entered>[1-9]\d*); "
    r"otherwise, use Federal Worksheet\) \^"
)
_ROW = re.compile(r"Line [1-9]\d* and (?:line )?(?P<field>[1-9]\d*): [^\r\n]+")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class DirectEnteredRole:
    export_uri: str
    input_uri: str
    coordinate: int
    frames: tuple[tuple[int, str, str], ...]
    # Preserve exact semantic source ownership, independently of the consumer.
    proofs: tuple[tuple[str, str, int, int, str, str, int, int], ...]
    provider_sha256: str


@dataclass(frozen=True)
class WorksheetChoice:
    owner: str
    selector: str
    worksheet: str
    amount: str
    role: DirectEnteredRole
    unit: str
    consumer_sha256: str
    source_sha256: str
    span: tuple[int, int]


def _versions(
    rule: Mapping[str, Any], context: Mapping[str, str | None]
) -> tuple | None:
    versions = rule.get("versions")
    if (
        not isinstance(versions, list)
        or not versions
        or any(
            not isinstance(v, dict)
            or set(v) - {"formula", "effective_from", "effective_to"}
            for v in versions
        )
    ):
        return None
    frames = tuple(_frame(rule, i, v, context) for i, v in enumerate(versions))
    if None in frames or len(set(frames)) != len(frames):
        return None
    return frames


def _physical_source(
    atom: Mapping[str, Any], context: Mapping[str, str | None]
) -> tuple | None:
    source = _source(atom, context)
    if source is None:
        return None
    citation, body, excerpt, start = source
    if (
        (start and body[start - 1] != "\n")
        or (
            start + len(excerpt) < len(body)
            and body[start + len(excerpt)] not in "\r\n"
        )
        or recognize_printed_rows(body.strip()) is None
    ):
        return None
    return source


def _provider(record: ResolvedImportedExport) -> tuple | None:
    if (
        not _canonical_record(record)
        or hashlib.sha256(record.content.encode()).hexdigest() != record.content_sha256
    ):
        return None
    payload = _payload(record.content)
    if payload is None:
        return None
    rules, inputs = _named(payload, "rules"), _named(payload, "inputs")
    imports = payload.get("imports", [])
    if (
        rules is None
        or inputs is None
        or set(rules) & set(inputs)
        or not isinstance(imports, list)
        or any(not isinstance(i, str) or _URI.fullmatch(i) is None for i in imports)
    ):
        return None
    aliases = [i.rsplit("#", 1)[1] for i in imports]
    if len(set(aliases)) != len(aliases) or set(aliases) & (set(rules) | set(inputs)):
        return None
    rule = rules.get(record.export_name)
    module = payload.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    citation = record.import_uri.rsplit("#", 1)[0].replace(":policies/", "/policy/")
    if (
        rule is None
        or not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != citation
    ):
        return None
    return payload, rule, inputs, citation


def bind_direct_entered_role(
    record: ResolvedImportedExport, context: Mapping[str, str | None], frames: tuple
) -> DirectEnteredRole | None:
    """Read a direct observation's own authenticated predicate, not a new scope."""
    provider = _provider(record)
    if provider is None:
        return None
    payload, rule, inputs, citation = provider
    if (
        set(rule)
        - {
            "name",
            "kind",
            "dtype",
            "entity",
            "period",
            "versions",
            "metadata",
            "description",
        }
        or (rule.get("kind"), rule.get("dtype"), rule.get("entity"), rule.get("period"))
        != ("derived", "Judgment", "Person", "Year")
        or _versions(rule, context) != frames
    ):
        return None
    names, coordinates, proofs = set(), set(), []
    for i, version in enumerate(rule["versions"]):
        formula = version.get("formula")
        if not isinstance(formula, str) or _NAME.fullmatch(formula.strip()) is None:
            return None
        name = formula.strip()
        fact = inputs.get(name)
        if (
            fact is None
            or set(fact) - {"name", "dtype", "entity", "period", "description"}
            or (fact.get("dtype"), fact.get("entity"), fact.get("period"))
            != ("Bool", "Person", "Year")
        ):
            return None
        atoms, predicates = (
            _atoms(rule, f"versions[{i}].formula", "formula"),
            _atoms(rule, f"versions[{i}].formula", "predicate"),
        )
        row = _physical_source(atoms[0], context) if len(atoms) == 1 else None
        predicate = _source(predicates[0], context) if len(predicates) == 1 else None
        if row is None or predicate is None:
            return None
        rc, body, excerpt, start = row
        pc, pb, quote, ps = predicate
        match = _ROW.fullmatch(excerpt)
        if (
            rc != citation
            or match is None
            or pc.rsplit("/", 1)[0] != citation.rsplit("/", 1)[0]
            or frames[i][1].rsplit("/", 1)[0] != citation.rsplit("/", 1)[0]
        ):
            return None
        coordinate = int(match["field"])
        if coordinate not in _entered_roles(pb).get(quote, ()):
            return None
        # A shared primitive cannot acquire conflicting roles through siblings.
        for sibling in payload["rules"]:
            if sibling is rule or sibling.get("dtype") != "Judgment":
                continue
            versions = sibling.get("versions", [])
            if not isinstance(versions, list):
                return None
            for j, v in enumerate(versions):
                if (
                    isinstance(v, dict)
                    and isinstance(v.get("formula"), str)
                    and v["formula"].strip() == name
                ):
                    aa = _atoms(sibling, f"versions[{j}].formula", "formula")
                    sr = _physical_source(aa[0], context) if len(aa) == 1 else None
                    sm = _ROW.fullmatch(sr[2]) if sr else None
                    if sm is None or int(sm["field"]) != coordinate:
                        return None
        names.add(name)
        coordinates.add(coordinate)
        proofs.append(
            (
                rc,
                hashlib.sha256(body.encode()).hexdigest(),
                start,
                start + len(excerpt),
                pc,
                hashlib.sha256(pb.encode()).hexdigest(),
                ps,
                ps + len(quote),
            )
        )
    if len(names) != 1 or len(coordinates) != 1:
        return None
    return DirectEnteredRole(
        record.import_uri,
        record.import_uri.rsplit("#", 1)[0] + "#input." + next(iter(names)),
        next(iter(coordinates)),
        frames,
        tuple(proofs),
        record.content_sha256,
    )


def _bind(
    evidence: SameCaseEvidence,
    content: str,
    branch: Any,
    source_text: str,
    citation: str,
    context: Mapping[str, str | None] | None,
    records: Sequence[ResolvedImportedExport],
    principal: Mapping[str, dict],
    source_branches: Sequence[Any],
) -> WorksheetChoice | None:
    from . import source_completeness as s

    if (
        not context
        or not isinstance(context.get(citation), str)
        or context[citation].strip() != source_text
        or not evidence.bound(content, records)
    ):
        return None
    match = _CLAUSE.fullmatch(branch.text)
    structure = s._unconflicted_printed_rows(source_text, source_branches)
    if (
        match is None
        or structure is None
        or (branch.start, branch.end) not in structure.atomic_spans
        or source_text[branch.start : branch.end] != branch.text
    ):
        return None
    payload = _payload(content)
    if payload is None:
        return None
    rules, inputs = _named(payload, "rules"), _named(payload, "inputs")
    imports = payload.get("imports", [])
    module = payload.get("module", {})
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if (
        rules is None
        or inputs is None
        or set(rules) & set(inputs)
        or any(rules.get(n) != r for n, r in principal.items())
        or not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != citation
        or citation
        != evidence.root_relative.removesuffix(".yaml").replace(
            "/policies/", "/policy/"
        )
    ):
        return None
    if (
        not isinstance(imports, list)
        or any(not isinstance(i, str) or _URI.fullmatch(i) is None for i in imports)
        or len(records) != len(imports)
        or {r.import_uri for r in records} != set(imports)
    ):
        return None
    aliases = [i.rsplit("#", 1)[1] for i in imports]
    if len(set(aliases)) != len(aliases) or set(aliases) & (set(rules) | set(inputs)):
        return None
    snapshots, modules = {}, {}
    for record in records:
        if (
            record.consumer_imports != tuple(imports)
            or _provider(record) is None
            or snapshots.setdefault(
                record.resolved_path, (record.content_sha256, record.content)
            )
            != (record.content_sha256, record.content)
            or modules.setdefault(
                record.import_uri.rsplit("#", 1)[0], record.resolved_path
            )
            != record.resolved_path
        ):
            return None
    by_alias = {r.export_name: r for r in records}
    candidates = []
    for name, owner in rules.items():
        if (
            branch.required_principal_output not in (None, name)
            or (
                owner.get("kind"),
                owner.get("dtype"),
                owner.get("entity"),
                owner.get("period"),
            )
            != ("derived", "Money", "Person", "Year")
            or not isinstance(owner.get("unit"), str)
        ):
            continue
        frames = _versions(owner, context)
        if frames is None or any(
            f[1].rsplit("/", 1)[0] != citation.rsplit("/", 1)[0] for f in frames
        ):
            continue
        shapes = []
        for i, v in enumerate(owner["versions"]):
            formula = v.get("formula")
            node = (
                s._parse_first_formula_branch_node(formula)
                if isinstance(formula, str)
                else None
            )
            aa = _atoms(owner, f"versions[{i}].formula", "formula")
            proof = _source(aa[0], context) if len(aa) == 1 else None
            if (
                node is None
                or node.kind != "if"
                or formula[: node.start].strip()
                or formula[node.end :].strip()
                or len(node.selectors) != 1
                or len(node.choices) != 2
                or any(
                    _NAME.fullmatch(x.strip()) is None
                    for x in (*node.selectors, *node.choices)
                )
                or proof is None
                or proof[0] != citation
                or proof[2] != branch.text
            ):
                break
            shapes.append(tuple(x.strip() for x in (*node.selectors, *node.choices)))
        if len(shapes) != len(frames) or len(set(shapes)) != 1:
            continue
        selector, worksheet, amount = shapes[0]
        if (
            selector not in by_alias
            or amount not in by_alias
            or worksheet not in inputs
        ):
            continue
        fact = inputs[worksheet]
        if set(fact) - {"name", "dtype", "unit", "entity", "period", "description"} or (
            fact.get("dtype"),
            fact.get("unit"),
            fact.get("entity"),
            fact.get("period"),
        ) != ("Money", owner["unit"], "Person", "Year"):
            continue
        role = bind_direct_entered_role(by_alias[selector], context, frames)
        if (
            role is None
            or role.coordinate != int(match["entered"])
            or any(
                p[0].rsplit("/", 1)[0] != citation.rsplit("/", 1)[0]
                or p[4].rsplit("/", 1)[0] != citation.rsplit("/", 1)[0]
                for p in role.proofs
            )
        ):
            continue
        amount_provider = _provider(by_alias[amount])
        assert amount_provider is not None
        _, amount_rule, _, amount_citation = amount_provider
        if (
            amount_citation.rsplit("/", 1)[0] != citation.rsplit("/", 1)[0]
            or (
                amount_rule.get("kind"),
                amount_rule.get("dtype"),
                amount_rule.get("entity"),
                amount_rule.get("period"),
                amount_rule.get("unit"),
            )
            != ("derived", "Money", "Person", "Year", owner["unit"])
            or _versions(amount_rule, context) != frames
        ):
            continue
        for i in range(len(frames)):
            aa = _atoms(amount_rule, f"versions[{i}].formula", "formula")
            row = _physical_source(aa[0], context) if len(aa) == 1 else None
            rm = _ROW.fullmatch(row[2]) if row else None
            if (
                row is None
                or row[0] != amount_citation
                or rm is None
                or int(rm["field"]) != int(match["amount"])
            ):
                break
        else:
            if not _exclusive_worksheet(evidence, name, worksheet):
                continue
            candidates.append(
                WorksheetChoice(
                    name,
                    selector,
                    worksheet,
                    amount,
                    role,
                    owner["unit"],
                    hashlib.sha256(content.encode()).hexdigest(),
                    hashlib.sha256(source_text.encode()).hexdigest(),
                    (branch.start, branch.end),
                )
            )
    return candidates[0] if len(candidates) == 1 else None


def _exclusive_worksheet(
    evidence: SameCaseEvidence, owner: str, worksheet: str
) -> bool:
    """Resolve primitive identity by module, never by foreign bare-name matches."""
    module = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    direct_refs = {module + "#" + worksheet, module + "#input." + worksheet}
    for item in evidence.files:
        if item.relative.endswith(".test.yaml"):
            continue
        payload = _payload(item.raw.decode())
        if payload is None:
            return False
        rules = _named(payload, "rules")
        imports = payload.get("imports", [])
        if (
            rules is None
            or not isinstance(imports, list)
            or any(
                not isinstance(ref, str)
                or _URI.fullmatch(ref) is None
                or ref in direct_refs
                for ref in imports
            )
        ):
            return False
        for name, rule in rules.items():
            versions = rule.get("versions", [])
            if not isinstance(versions, list) or not all(
                isinstance(v, dict) and "formula" in v for v in versions
            ):
                return False
            for version in versions:
                formula = version["formula"]
                if not isinstance(formula, str):
                    # Typed literal parameters have no identifiers to traverse.
                    if (
                        rule.get("kind") != "parameter"
                        or type(formula) not in (int, float)
                        or (isinstance(formula, float) and not math.isfinite(formula))
                    ):
                        return False
                    continue
                if any(ref in formula for ref in direct_refs):
                    return False
                if (
                    item.relative == evidence.root_relative
                    and name != owner
                    and worksheet in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
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
    imported_exports: Sequence[ResolvedImportedExport],
    source_context: Mapping[str, str | None] | None,
    principal_rules: dict[str, dict],
    asserted_by_rule: dict[str, list],
    formula_environment: dict[str, Any],
    source_branches: Sequence[Any] = (),
) -> set[tuple]:
    from . import source_completeness as s

    choice = _bind(
        evidence,
        content,
        branch,
        source_text,
        corpus_citation_path,
        source_context,
        imported_exports,
        principal_rules,
        source_branches,
    )
    if choice is None:
        return set()
    module = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    worksheet_uri = module + "#input." + choice.worksheet
    try:
        derived = json.loads(evidence.artifact_bytes)["program"]["derived"]
    except (ValueError, KeyError, TypeError):
        return set()
    refs = {r.export_name: r.import_uri for r in imported_exports}
    refs[choice.owner] = module + "#" + choice.owner
    if not isinstance(derived, list) or not all(isinstance(d, dict) for d in derived):
        return set()
    for name in (choice.owner, choice.amount, choice.selector):
        ds = [d for d in derived if d.get("id") == refs[name]]
        expected = {
            "entity": "Person",
            "period": "Year",
            "dtype": "judgment" if name == choice.selector else "decimal",
            "semantics": "judgment" if name == choice.selector else "scalar",
            "unit": None if name == choice.selector else choice.unit,
        }
        if len(ds) != 1 or any(ds[0].get(k) != v for k, v in expected.items()):
            return set()
    executed = []
    for case in asserted_by_rule.get(choice.owner, ()):
        query = _query(evidence, case)
        facts = case.get("input")
        if (
            query is None
            or not isinstance(facts, dict)
            or type(facts.get(choice.role.input_uri)) is not bool
        ):
            continue
        entered = facts[choice.role.input_uri]
        if (worksheet_uri in facts) != entered:
            continue
        values = evidence.values(case, content)
        if values is None:
            continue
        imported, observed = values
        if (
            imported.keys() & formula_environment.keys()
            or imported.get(choice.selector) is not entered
        ):
            continue
        environment = {**formula_environment, **imported}
        replay = s._case_dependency_environment(
            principal_rules,
            case,
            formula_environment=environment,
            require_asserted_value=False,
        )
        if choice.owner not in replay or any(
            n not in observed
            or not s._formula_runtime_values_equal(value, observed[n])
            or not s._asserted_formula_runtime_values_equal(
                principal_rules[n], value, s._test_case_asserted_output_value(case, n)
            )
            for n, value in replay.items()
        ):
            continue
        execution = s._case_formula_execution(
            principal_rules[choice.owner],
            case,
            formula_environment=environment,
            dependency_environment=replay,
        )
        expected = choice.worksheet if entered else choice.amount
        if (
            execution is None
            or execution.leaf.strip() != expected
            or len(execution.trace) != 1
        ):
            continue
        leaf_value = (
            facts.get(worksheet_uri) if entered else imported.get(choice.amount)
        )
        if isinstance(leaf_value, bool) or not s._asserted_formula_runtime_values_equal(
            principal_rules[choice.owner], observed[choice.owner], leaf_value
        ):
            continue
        executed.append((case, query, entered, observed[choice.owner]))
    result = set()
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
        false, true = (a, b) if not ab else (b, a)
        fi, ti = false["input"], true["input"]
        if set(ti) != set(fi) | {worksheet_uri} or any(
            type(fi[k]) is not type(ti[k]) or fi[k] != ti[k]
            for k in fi
            if k != choice.role.input_uri
        ):
            continue
        result.add(
            (
                choice.owner,
                choice.source_sha256,
                choice.span,
                *sorted((case_key(a), case_key(b))),
            )
        )
    return result
