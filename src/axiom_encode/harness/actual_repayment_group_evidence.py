"""Whole alternative-group coverage from every source-bound actual threshold pair.

This is not leaf-level MC/DC. Partial and overlapping source obligations do not
receive evidence, and existing boundary execution certificates remain unchanged.
"""

from __future__ import annotations

from decimal import InvalidOperation
from typing import Any

from . import actual_repayment_boundary_evidence as boundary_evidence
from .imported_export_evidence import _BULLET
from .repayment_alternative_evidence import certify_repayment_alternatives


def _witnesses(
    evidence: Any,
    branch: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    imported_exports: Any,
    source_context: Any,
    principal_rules: Any,
    asserted_by_rule: Any,
    formula_environment: Any,
    boundary_obligations: Any,
    boundary_branches: Any,
    source_branches: Any = (),
) -> set[tuple]:
    from . import source_completeness as s

    if not evidence.bound(content, imported_exports):
        return set()
    raw = source_context.get(corpus_citation_path) if source_context else None
    if (
        not isinstance(raw, str)
        or raw.strip() != source_text
        or not 0 <= branch.start < branch.end <= len(source_text)
        or source_text[branch.start : branch.end] != branch.text
    ):
        return set()
    coverage = certify_repayment_alternatives(
        content,
        source_text=source_text,
        raw_source_text=raw,
        citation=corpus_citation_path,
        source_context=source_context,
        imported_exports=imported_exports,
        parse_branch=s._parse_first_formula_branch_node,
        parse_expression=s._parse_formula_expression,
    )
    records = tuple(
        record
        for record in coverage.records
        if record.span == (branch.start, branch.end) and record.text == branch.text
    )
    if not records or len({record.owner for record in records}) != 1:
        return set()
    # Never accept a subset of the certified owner's versions or another group.
    owner = records[0].owner
    if branch.required_principal_output not in (None, owner):
        return set()
    if tuple(r for r in coverage.records if r.owner == owner) != records:
        return set()
    kwargs = dict(
        content=content,
        source_text=source_text,
        corpus_citation_path=corpus_citation_path,
        imported_exports=imported_exports,
        source_context=source_context,
        principal_rules=principal_rules,
        asserted_by_rule=asserted_by_rule,
        formula_environment=formula_environment,
        boundary_branches=boundary_branches,
        source_branches=source_branches,
    )
    all_pairs = set()
    for record in records:
        expected = []
        cursor = record.span[0]
        for line in record.text.splitlines(keepends=True):
            quote = line.rstrip("\r\n")
            match = _BULLET.fullmatch(quote[2:]) if quote.startswith("- ") else None
            if match is not None:
                a, b = match.span("amount")
                expected.append((cursor + 2 + a, cursor + 2 + b))
            cursor += len(line)
        if (
            record.tree[0] != "or"
            or len(expected) != len(record.tree[1])
            or len(set(expected)) != len(expected)
        ):
            return set()
        for span in expected:
            candidates = []
            for source_branch, occurrence in boundary_obligations:
                bindings = boundary_evidence.bind_occurrences(
                    content,
                    source_branch,
                    occurrence,
                    source_text=source_text,
                    citation=corpus_citation_path,
                    context=source_context,
                    records=imported_exports,
                    principal=principal_rules,
                    boundary_branches=boundary_branches,
                    source_branches=source_branches,
                )
                if any(b.record == record and b.span == span for b in bindings):
                    candidates.append((source_branch, occurrence))
            if len(candidates) != 1:
                return set()
            source_branch, occurrence = candidates[0]
            pairs = {
                pair
                for pair in boundary_evidence.witnesses(
                    evidence, source_branch, occurrence, **kwargs
                )
                if pair[:4] == (owner, record.version, record.source_sha256, span)
            }
            if not pairs:
                return set()
            all_pairs.update(pairs)
    return {
        (
            "actual-repayment-whole-group",
            owner,
            records[0].consumer_sha256,
            records[0].source_sha256,
            (branch.start, branch.end),
            tuple(sorted(all_pairs)),
        )
    }


def witnesses(*args: Any, **kwargs: Any) -> set[tuple]:
    """Malformed optional evidence leaves the ordinary obligation unresolved."""
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
