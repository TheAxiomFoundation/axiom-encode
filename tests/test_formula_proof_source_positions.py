"""Repeated worksheet instructions require their own authoritative context."""

import json
from dataclasses import replace

import pytest
import yaml

from axiom_encode.corpus_resolver import PROOF_EVIDENCE_SEGMENT_SEPARATOR
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.coverage_index import format_coverage_index
from axiom_encode.harness.proof_validator import (
    _source_contains_proof_evidence,
    _source_proof_evidence_spans,
)

CITATION = "ca/policy/worksheet"
INSTRUCTION = "Subtract line 4 from line 1."
ALPHA = f"Alpha worksheet. {INSTRUCTION}"
BETA = f"Beta worksheet. {INSTRUCTION}"
SOURCE = f"{ALPHA} {BETA}"


def rule(name, excerpt, citation=CITATION):
    return {
        "name": name,
        "kind": "derived",
        "source": CITATION,
        "versions": [{"effective_from": "2025-01-01", "formula": "amount - offset"}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": citation,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
    }


def clauses(source=SOURCE):
    starts = [source.index(INSTRUCTION), source.rindex(INSTRUCTION)]
    return tuple(
        sc.SourceStructureBranch(
            path=("worksheet",),
            kind="formula",
            label=f"formula {index}",
            text=INSTRUCTION,
            start=start,
            end=start + len(INSTRUCTION),
        )
        for index, start in enumerate(starts)
    )


def bindings(source, branches, rules):
    named = {r["name"]: r for r in rules}
    return sc._principal_formula_clause_rules(
        branches,
        source_text=source,
        principal_rules=named,
        principal_rule_paths={name: {("worksheet",)} for name in named},
        named_rules=named,
        corpus_citation_path=CITATION,
    )


def test_identical_instructions_bind_only_their_own_full_context():
    branches = clauses()
    result = bindings(SOURCE, branches, [rule("alpha", ALPHA), rule("beta", BETA)])
    assert result[branches[0]] == {"alpha"}
    assert result[branches[1]] == {"beta"}


@pytest.mark.parametrize("excerpt", [INSTRUCTION, ALPHA])
def test_repeated_complete_excerpt_is_not_an_ownership_declaration(excerpt):
    source = f"{ALPHA} {ALPHA}"
    result = bindings(source, clauses(source), [rule("ambiguous", excerpt)])
    assert all(not owners for owners in result.values())


def test_overlapping_full_contexts_are_not_mistaken_for_unique_evidence():
    source = f"{INSTRUCTION} {INSTRUCTION} {INSTRUCTION}"
    excerpt = f"{INSTRUCTION} {INSTRUCTION}"
    width = len(INSTRUCTION) + 1
    assert _source_proof_evidence_spans(source_text=source, evidence_text=excerpt) == (
        (0, len(excerpt)),
        (width, width + len(excerpt)),
    )
    result = bindings(source, clauses(source), [rule("ambiguous", excerpt)])
    assert all(not owners for owners in result.values())


def test_whitespace_matching_preserves_raw_coordinates():
    source = "Alpha\nworksheet. Subtract\tline 4 from line 1. " + BETA
    start = source.index("Subtract")
    end = source.index(". ", start) + 1
    branches = (
        replace(clauses()[0], start=start, end=end, text=source[start:end]),
        clauses(source)[1],
    )
    result = bindings(source, branches, [rule("alpha", ALPHA)])
    assert result[branches[0]] == {"alpha"}
    assert not result[branches[1]]
    assert _source_proof_evidence_spans(source_text=source, evidence_text=ALPHA) == (
        (0, end),
    )


@pytest.mark.parametrize("change", ["start", "end", "text", "source"])
def test_stale_or_forged_coordinates_do_not_bind(change):
    branches = list(clauses())
    source = SOURCE
    if change == "source":
        source = "Prefix. " + source
    else:
        value = "different" if change == "text" else getattr(branches[0], change) + 1
        branches[0] = replace(branches[0], **{change: value})
    result = bindings(source, branches, [rule("alpha", ALPHA)])
    assert not result[branches[0]]


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("value", [True, 16.0, "16", None])
def test_non_integer_coordinates_are_unresolved(field, value):
    branches = list(clauses())
    branches[0] = replace(branches[0], **{field: value})
    assert not bindings(SOURCE, branches, [rule("alpha", ALPHA)])[branches[0]]


def test_wrong_citation_and_changed_context_cannot_claim_same_instruction():
    branches = clauses()
    for candidate in (
        rule("wrong", ALPHA, "ca/policy/another"),
        rule("wrong", ALPHA.lower()),
        rule("wrong", ALPHA.replace("worksheet.", "worksheet:")),
    ):
        assert all(not x for x in bindings(SOURCE, branches, [candidate]).values())


@pytest.mark.parametrize(
    ("source", "quote", "expected"),
    [
        ("A\nB", "A B", ((0, 3),)),
        ("A\n\nB", "A B", ()),
        ("A\n\nB", "A\n\nB", ((0, 4),)),
        ("A\u2029B", "A B", ()),
        ("A" + PROOF_EVIDENCE_SEGMENT_SEPARATOR + "B", "A B", ()),
        ("foobar", "bar", ()),
        ("minus -50 dollars", "50", ()),
        ("3 142 WORK CHART", "142 WORK CHART", ()),
        ("ל־ 1⁄2", "ל־1⁄2", ((0, 6),)),
        ("מאה־ ו־ כ־ שלושה", "מאה־ו־כ־שלושה", ((0, 16),)),
    ],
)
def test_locations_keep_normal_proof_boundaries(source, quote, expected):
    assert (
        _source_proof_evidence_spans(source_text=source, evidence_text=quote)
        == expected
    )
    assert _source_contains_proof_evidence(
        source_text=source, evidence_text=quote
    ) == bool(expected)


def test_segment_offsets_and_duplicate_positions_are_not_conflated():
    source = ALPHA + PROOF_EVIDENCE_SEGMENT_SEPARATOR + ALPHA
    second = len(ALPHA) + len(PROOF_EVIDENCE_SEGMENT_SEPARATOR)
    assert _source_proof_evidence_spans(source_text=source, evidence_text=ALPHA) == (
        (0, len(ALPHA)),
        (second, second + len(ALPHA)),
    )


def test_direct_single_clause_structural_route_is_unchanged():
    branch = clauses()[0]
    assert bindings(SOURCE, [branch], [rule("structural", "not excerpt evidence")])[
        branch
    ] == {"structural"}


def test_omitted_leading_marker_keeps_the_local_body_coordinates():
    source = f"(a) {INSTRUCTION} Alpha context. {BETA}"
    first = replace(clauses(source)[0], start=0, text=f"(a) {INSTRUCTION}")
    branches = (first, clauses(source)[1])
    result = bindings(
        source, branches, [rule("alpha", f"{INSTRUCTION} Alpha context.")]
    )
    assert result[first] == {"alpha"}
    assert not result[branches[1]]


def test_normal_analyzer_and_generation_index_agree_on_unowned_copy():
    first = f"Alpha worksheet: base income. {INSTRUCTION}"
    source = f"{first} Beta worksheet: pension income. {INSTRUCTION}"
    content = yaml.safe_dump({"format": "rulespec/v1", "rules": [rule("alpha", first)]})
    rows = [
        json.loads(line)
        for line in format_coverage_index(
            source, CITATION, candidate=content
        ).splitlines()
        if line.startswith('{"span"')
    ]
    assert [r["bound_rules"] for r in rows] == [["alpha"], []]
    analysis = sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=vp.extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    issues = [
        x
        for x in analysis.issues
        if x.startswith("[complete-source-unit:formula-output]")
    ]
    assert len(issues) == 1
    start, end = rows[1]["span"]
    assert f"characters {start}:{end}:" in issues[0]
