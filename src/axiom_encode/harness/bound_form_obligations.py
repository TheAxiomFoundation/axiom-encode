"""Closed source ownership for document requirements, not condition clearance.

A premise is an explicit caller fact. Clause proof authenticates the obligation,
not that fact's truth. Conservative name/description correlation cannot detect a
coherent redefinition of a reviewed fact; the caller mapping still needs review.
This helper neither admits snapshots nor allocates paired-condition witnesses.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping

from axiom_encode.bound_form_structure import BoundFormStructure

_SHEET = re.compile(
    r"(?m)^If you need more space, attach an additional sheet of paper\.\r?$"
)
_PREMISE = re.compile(
    r"The (?:individual|filer) completing this form needs "
    r"(?:more|additional) space for its entries\."
)
_PREMISE_NAME = re.compile(r"(?:(?:filer|individual)_)?needs_(?:more|additional)_space")
_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


@dataclass(frozen=True)
class DocumentRequirement:
    rule_name: str
    input_name: str
    source_span: tuple[int, int]
    source_excerpt: str
    entity: str
    frame: tuple[str, str]
    formula_ast: str


def _proofs(rule: Mapping[str, Any], form: BoundFormStructure, excerpt: str) -> bool:
    metadata = rule.get("metadata")
    proof = metadata.get("proof") if isinstance(metadata, dict) else None
    atoms = proof.get("atoms") if isinstance(proof, dict) else None
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"proof"}
        or not isinstance(proof, dict)
        or set(proof) != {"atoms"}
        or not isinstance(atoms, list)
        or len(atoms) != 3
    ):
        return False
    expected = {
        "versions[0].formula": ("formula", excerpt),
        "versions[0].effective_from": ("effective_period", str(form.header.year)),
        "versions[0].effective_to": ("effective_period", str(form.header.year)),
    }
    seen = set()
    for atom in atoms:
        if not isinstance(atom, dict) or set(atom) != {"path", "kind", "source"}:
            return False
        path = atom.get("path")
        if not isinstance(path, str) or path not in expected or path in seen:
            return False
        seen.add(path)
        kind, text = expected[path]
        source = atom.get("source")
        if (
            atom.get("kind") != kind
            or not isinstance(source, dict)
            or set(source) != {"corpus_citation_path", "excerpt"}
            or source.get("corpus_citation_path") != form.citation
            or source.get("excerpt") != text
        ):
            return False
    return seen == set(expected)


def bind_document_requirements(
    form: BoundFormStructure,
    rules: Mapping[str, Mapping[str, Any]],
    inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[DocumentRequirement, ...] | None:
    """Account for every supplied non-row rule/input or refuse the partition.

    The caller must independently establish that these maps and the complete row
    maps partition the *original* module. No projection gains execution identity.
    Arbitrary extra Judgment rules are not allowed. Generic source correlation,
    ordinary proof validation and actual typed witness allocation remain required.
    """
    if not isinstance(form, BoundFormStructure) or not form.bound(
        form.source, form.citation
    ):
        return None
    if not isinstance(rules, Mapping) or not isinstance(inputs, Mapping):
        return None
    if set(rules) & set(inputs):
        return None
    for items in (rules, inputs):
        if any(
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or not isinstance(item, Mapping)
            or item.get("name") != name
            for name, item in items.items()
        ):
            return None
    if not rules:
        return () if not inputs else None
    clauses = tuple(_SHEET.finditer(form.source))
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    span = clause.span()
    if not any(a <= span[0] and span[1] <= b for a, b in form.unowned_spans):
        return None
    frame = (f"{form.header.year}-01-01", f"{form.header.year}-12-31")
    result = []
    used_inputs: set[str] = set()
    used_spans: set[tuple[int, int]] = set()
    for name, rule in rules.items():
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
                "source",
                "description",
            }
            or rule.get("kind") != "derived"
            or rule.get("dtype") != "Judgment"
            or rule.get("entity") != "Person"
            or rule.get("period") != "Year"
            or "unit" in rule
        ):
            return None
        versions = rule.get("versions")
        if not isinstance(versions, list) or len(versions) != 1:
            return None
        version = versions[0]
        if (
            not isinstance(version, dict)
            or set(version) != {"formula", "effective_from", "effective_to"}
            or tuple(version[k] for k in ("effective_from", "effective_to")) != frame
            or not isinstance(version["formula"], str)
        ):
            return None
        try:
            expression = ast.parse(version["formula"], mode="eval").body
        except (SyntaxError, ValueError):
            return None
        if not isinstance(expression, ast.Name) or expression.id not in inputs:
            return None
        fact_name = expression.id
        fact = inputs[fact_name]
        if (
            not _PREMISE_NAME.fullmatch(fact_name)
            or fact.get("dtype") != "Boolean"
            or fact.get("entity") != "Person"
            or fact.get("period") != "Year"
            or set(fact) != {"name", "dtype", "entity", "period", "description"}
            or not isinstance(fact["description"], str)
            or not _PREMISE.fullmatch(fact["description"])
            or fact_name in used_inputs
            or span in used_spans
            or not _proofs(rule, form, clause.group())
        ):
            return None
        used_inputs.add(fact_name)
        used_spans.add(span)
        result.append(
            DocumentRequirement(
                name,
                fact_name,
                span,
                clause.group(),
                "Person",
                frame,
                ast.dump(expression),
            )
        )
    return tuple(result) if used_inputs == set(inputs) else None


def observed_document_pairs(
    requirements: tuple[DocumentRequirement, ...],
    evidence: Any,
    cases: Any,
    prefix: str,
    period: Mapping[str, Any],
) -> bool:
    """Require separate, strictly typed Person witnesses for every owned clause.

    The caller authenticates full content, compiled inventory and capture freshness.
    In particular, engine895 alone does not reject a same-ID wrong-entity fact.
    """
    import json

    from .same_case_execution import case_key, strict_explain_metadata

    try:
        for requirement in requirements:
            witnessed: dict[bool, str] = {}
            ref = prefix + requirement.rule_name
            fact_ref = prefix + "input." + requirement.input_name
            for case in cases or ():
                if not isinstance(case, dict) or case.get("period") != period:
                    continue
                entity_id = case.get("entity_id")
                if not isinstance(entity_id, str) or not entity_id:
                    continue
                facts = case.get("input")
                if not isinstance(facts, dict) or set(facts) != {fact_ref}:
                    continue
                value = facts[fact_ref]
                if type(value) is not bool:
                    continue
                outcome = "holds" if value else "not_holds"
                if case.get("output") != {ref: outcome}:
                    continue
                records = [
                    r for r in evidence.records if r.case_sha256 == case_key(case)
                ]
                if len(records) != 1:
                    continue
                request = json.loads(records[0].request_json)
                response = json.loads(records[0].response_json)
                query = {
                    "entity_id": entity_id,
                    "period": dict(period),
                    "outputs": [ref],
                }
                fact = {
                    "name": fact_ref,
                    "entity": requirement.entity,
                    "entity_id": entity_id,
                    "interval": {"start": period["start"], "end": period["end"]},
                    "value": {"kind": "bool", "value": value},
                }
                actual_facts = request.get("dataset", {}).get("inputs")
                if (
                    not isinstance(actual_facts, list)
                    or len(actual_facts) != 1
                    or not isinstance(actual_facts[0], dict)
                    or not isinstance(actual_facts[0].get("value"), dict)
                    or type(actual_facts[0]["value"].get("value")) is not bool
                ):
                    continue
                if (
                    not strict_explain_metadata(request, response)
                    or request.get("queries") != [query]
                    or request.get("dataset") != {"inputs": [fact], "relations": []}
                    or len(response.get("results", [])) != 1
                ):
                    continue
                result = response["results"][0]
                output = {
                    "kind": "judgment",
                    "name": requirement.rule_name,
                    "id": ref,
                    "unit": None,
                    "outcome": outcome,
                }
                if (
                    result.get("entity_id") != entity_id
                    or result.get("period") != period
                    or result.get("assessment_date") is not None
                    or result.get("outputs") != {ref: output}
                    or set(result.get("trace", {})) != {ref}
                ):
                    continue
                trace = result["trace"][ref]
                expected_trace = {
                    **output,
                    "entity_id": entity_id,
                    "source": requirement.source_excerpt,
                    "dependencies": [],
                }
                if set(trace) != set(expected_trace) | {"executed_expression"} or any(
                    trace[k] != v for k, v in expected_trace.items()
                ):
                    continue
                expression = trace["executed_expression"]
                if not isinstance(expression, str) or ast.dump(
                    ast.parse(expression, mode="eval")
                ) != ast.dump(
                    ast.parse(
                        'input("' + requirement.input_name + '") == true', mode="eval"
                    )
                ):
                    continue
                witnessed[value] = entity_id
            if set(witnessed) != {False, True} or witnessed[False] != witnessed[True]:
                return False
        return True
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        SyntaxError,
        RecursionError,
    ):
        return False
