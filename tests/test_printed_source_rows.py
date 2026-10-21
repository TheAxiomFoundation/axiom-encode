"""Printed operation rows retain raw ownership and inherited condition text."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as s

P = Path(__file__).parent / "fixtures/form-numeric-locators"


def clauses(text):
    return list(s._source_clause_spans(text, branches=()))


def formulas(text):
    return s._source_formula_branches(
        text,
        corpus_citation_path="ca/policy/test",
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )


def test_heading_does_not_make_residency_instruction_arithmetic():
    text = (P / "step-2.txt").read_text()
    found = formulas(text)
    assert not any("As a resident" in x.text for x in found)
    assert any("Add lines 1 to 20" in x.text for x in found)
    assert any("Line 28 plus line 32" in x.text for x in found)


def test_three_independent_step3_operations_do_not_share_whole_form():
    text = (P / "step-3.txt").read_text()
    found = formulas(text)
    assert any(
        x.text.startswith("Line 53 and line 23300:") and "Line 54" not in x.text
        for x in found
    )
    assert any(
        x.text.startswith("Line 54 and line 23400:") and "Line 55" not in x.text
        for x in found
    )
    assert all(text[x.start : x.end] == x.text for x in found)


def test_inherited_repayment_guards_keep_explicit_else_row():
    text = (P / "step-3.txt").read_text()
    found = [x for _, _, x in clauses(text) if "Line 55 and line 23500:" in x]
    assert len(found) == 1
    clause = found[0]
    assert clause.startswith("Social benefits repayment:")
    for part in [
        "Complete the chart",
        "$82,125",
        "$93,454",
        "line 11900",
        "line 11300",
        "line 14600",
        'If not, enter "0"',
    ]:
        assert part in clause
    assert "Line 56" not in clause


def test_conditional_other_payments_remains_one_raw_row():
    text = (P / "step-4.txt").read_text()
    found = [x for _, _, x in clauses(text) if "Line 60 and line 25000:" in x]
    assert len(found) == 1 and "; otherwise, use Federal Worksheet)" in found[0]
    assert "Line 61" not in found[0]


def test_unresolved_loss_advisory_is_retained():
    text = (P / "step-3.txt").read_text()
    found = [
        x
        for _, _, x in clauses(text)
        if "If negative, you may have a non-capital loss" in x
    ]
    assert len(found) == 1
    assert "negative amount is to be used for certain calculations" in found[0]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_row",
        "missing_row",
        "duplicate_field",
        "quoted_header",
        "unknown_inherited",
        "missing_inherited_bullet",
        "mismatched_inherited_field",
        "quoted_guard",
    ],
)
def test_ambiguous_structure_retains_legacy_boundaries(mutation):
    from axiom_encode.printed_source_rows import (
        mask_printed_heading_dash,
        recognize_printed_rows,
    )

    text = (P / "step-3.txt").read_text()
    if mutation == "duplicate_row":
        text = text.replace("Line 37 and 20805:", "Line 36 and 20805:")
    elif mutation == "missing_row":
        text = text.replace("Line 37 and 20805:", "Unknown row:")
    elif mutation == "duplicate_field":
        text = text.replace("Line 37 and 20805:", "Line 37 and 20800:")
    elif mutation == "quoted_header":
        text = '"' + text + '"'
    elif mutation == "unknown_inherited":
        text = text.replace('If not, enter "0"', 'If otherwise qualified, enter "0"')
    elif mutation == "missing_inherited_bullet":
        text = text.replace(
            "- You entered an amount for EI and other benefits on line 11900 and the amount on line 23400 is more than $82,125",
            "",
        )
    elif mutation == "mismatched_inherited_field":
        text = text.replace(
            "Complete the chart for line 23500", "Complete the chart for line 99999"
        )
    else:
        text = text.replace(
            "Social benefits repayment:", '"Social benefits repayment:'
        ).replace("Line 56 and line 23600:", '"\nLine 56 and line 23600:')
    assert recognize_printed_rows(text) is None
    assert mask_printed_heading_dash(text) == text


@pytest.mark.parametrize("step", [2, 3, 4])
def test_all_operative_characters_stay_at_exact_source_positions(step):
    from axiom_encode.printed_source_rows import (
        mask_printed_heading_dash,
        recognize_printed_rows,
    )

    text = (P / f"step-{step}.txt").read_text()
    evidence = recognize_printed_rows(text)
    assert evidence is not None
    masked = mask_printed_heading_dash(text)
    a, b = evidence.heading_dash
    assert (
        len(masked) == len(text) and masked[:a] == text[:a] and masked[b:] == text[b:]
    )
    for start, end, clause in clauses(text):
        assert text[start:end] == clause
    for start, end in evidence.atomic_spans:
        assert any(a <= start and end <= b for a, b, _ in clauses(text))


def test_unknown_within_row_semicolon_is_not_silently_merged():
    from axiom_encode.printed_source_rows import recognize_printed_rows

    text = (
        (P / "step-4.txt")
        .read_text()
        .replace("; otherwise, use Federal Worksheet", "; subtract 200 dollars")
    )
    evidence = recognize_printed_rows(text)
    assert evidence is not None
    assert not any(
        "subtract 200 dollars" in text[a:b] for a, b in evidence.atomic_spans
    )
    assert any("subtract 200 dollars" in c for _, _, c in clauses(text))


@pytest.mark.parametrize(
    "step,phrase",
    [(2, "Add lines 1 to 20"), (3, "Add lines 35 to 52"), (4, "Add lines 58 to 67")],
)
def test_closed_add_rows_remain_explicit_computations(step, phrase):
    text = (P / f"step-{step}.txt").read_text()
    assert any(phrase in x.text for x in formulas(text))


@pytest.mark.parametrize(
    "replacement",
    [
        "Add lines 52 to 35.",
        "Add lines 35 to 99.",
        "Add lines 34 to 52 only if eligible.",
        "Amounts from lines 35 to 52.",
    ],
)
def test_unknown_add_row_is_not_force_classified(replacement):
    from axiom_encode.printed_source_rows import recognize_printed_rows

    text = (P / "step-3.txt").read_text().replace("Add lines 35 to 52.", replacement)
    evidence = recognize_printed_rows(text)
    assert evidence is None or not evidence.additions


@pytest.mark.parametrize(
    "step,needle", [(3, "you entered an amount for ei"), (4, "otherwise, use")]
)
def test_explicit_owner_inside_atomic_row_refuses_printed_interpretation(step, needle):
    text = (P / f"step-{step}.txt").read_text()
    start = text.lower().index(needle)
    end = start + len(needle)
    branch = s.SourceStructureBranch(
        ("explicit",), "paragraph", "explicit", text[start:end], start, end
    )
    assert s._unconflicted_printed_rows(text, (branch,)) is None
    spans = list(s._source_clause_spans(text, branches=(branch,)))
    assert any(a == start and b == end for a, b, _ in spans)
    # Both consumers use the same refused evidence; no isolated Add inference.
    found = s._source_formula_branches(
        text, branches=(branch,), active_branches=(branch,), deferred_paths=set()
    )
    assert not any(x.text.startswith("Line 53 and line 23300:") for x in found)


@pytest.mark.parametrize("step,row", [(2, 21), (2, 27), (2, 32), (3, 53), (4, 68)])
def test_add_operation_context_binds_actual_selected_version(step, row):
    text = (P / f"step-{step}.txt").read_text()
    context = s._printed_operation_context(text, (), "ca/policy/test")
    branch = next(
        b
        for b in formulas(text)
        if b.text.startswith(f"Line {row}:") or b.text.startswith(f"Line {row} and")
    )
    a, b = next(
        (a, b)
        for a, b in context.structure.additions
        if branch.start <= a < b <= branch.end
    )
    excerpt = text[a:b]
    rule = {
        "name": "total",
        "kind": "derived",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": "a + b",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {
                            "corpus_citation_path": "ca/policy/test",
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
    }
    case = {"period": "2025"}
    assert s._printed_operation_proof(context, branch, excerpt, "ca/policy/test")
    peers = formulas(text)
    assert len([b for b in peers if b.path == branch.path]) > 1
    bound = s._principal_formula_clause_rules(
        peers,
        source_text=text,
        principal_rules={"total": rule},
        principal_rule_paths={"total": {branch.path}},
        named_rules={"total": rule},
        corpus_citation_path="ca/policy/test",
        printed_operations=context,
    )
    assert bound[branch] == {"total"}
    legacy = s._principal_formula_clause_rules(
        peers,
        source_text=text,
        principal_rules={"total": rule},
        principal_rule_paths={"total": {branch.path}},
        named_rules={"total": rule},
        corpus_citation_path="ca/policy/test",
    )
    assert legacy[branch] == set()
    wrong_clause = (
        next(b for b in peers if b != branch and "Add lines" in b.text)
        if step == 2
        else s.SourceStructureBranch((), "paragraph", "wrong", text[:4], 0, 4)
    )
    assert not s._printed_operation_proof(
        context, wrong_clause, excerpt, "ca/policy/test"
    )
    assert not s._selected_principal_formula_has_exact_source_proof(
        rule,
        case=case,
        branch=wrong_clause,
        corpus_citation_path="ca/policy/test",
        printed_operations=context,
    )
    if step in (3, 4):
        atomic_start, atomic_end = context.structure.atomic_spans[0]
        conflict = s.SourceStructureBranch(
            ("owner",),
            "paragraph",
            "owner",
            text[atomic_start + 1 : atomic_end - 1],
            atomic_start + 1,
            atomic_end - 1,
        )
        refused = s._printed_operation_context(text, (conflict,), "ca/policy/test")
        assert refused is None
        assert not s._selected_principal_formula_has_exact_source_proof(
            rule,
            case=case,
            branch=branch,
            corpus_citation_path="ca/policy/test",
            printed_operations=refused,
        )
        assert (
            s._principal_formula_clause_rules(
                peers,
                source_text=text,
                principal_rules={"total": rule},
                principal_rule_paths={"total": {branch.path}},
                named_rules={"total": rule},
                corpus_citation_path="ca/policy/test",
                printed_operations=refused,
            )[branch]
            == set()
        )
    assert s._selected_principal_formula_has_exact_source_proof(
        rule,
        case=case,
        branch=branch,
        corpus_citation_path="ca/policy/test",
        printed_operations=context,
    )
    assert not s._printed_operation_proof(
        context, branch, excerpt[:-1], "ca/policy/test"
    )
    assert not s._printed_operation_proof(context, branch, excerpt, "ca/policy/foreign")
    assert not s._selected_principal_formula_has_exact_source_proof(
        rule,
        case={"period": "2024"},
        branch=branch,
        corpus_citation_path="ca/policy/test",
        printed_operations=context,
    )
    assert s._printed_operation_context(excerpt, (), "ca/policy/test") is None
    duplicated = s._printed_operation_context(
        text + "\n" + excerpt, (), "ca/policy/test"
    )
    assert duplicated is not None
    assert not s._printed_operation_proof(duplicated, branch, excerpt, "ca/policy/test")


@pytest.mark.parametrize(
    "formula,expected,accepted",
    [
        ("a + b", 13, True),
        ("a - b", 7, False),
        ("a * b", 30, False),
        ("a / b", 10 / 3, False),
        ("7", 7, False),
    ],
)
def test_source_bound_add_rejects_wrong_operations_with_correct_assertions(
    formula, expected, accepted
):
    from axiom_encode.harness.validator_pipeline import (
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    text = (P / "step-4.txt").read_text()
    branch = next(b for b in formulas(text) if b.text.startswith("Line 68"))
    context = s._printed_operation_context(text, (), "ca/policy/test")
    rule = {
        "name": "total",
        "kind": "derived",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": formula,
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {
                            "corpus_citation_path": "ca/policy/test",
                            "excerpt": branch.text,
                        },
                    }
                ]
            }
        },
    }
    case = {
        "name": "actual",
        "period": "2025",
        "input": {"a": 10, "b": 3},
        "output": {"total": expected},
    }
    kwargs = dict(
        corpus_citation_path="ca/policy/test",
        principal_rules={"total": rule},
        rule_names={"total"},
        asserted_by_rule={"total": [case]},
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=text,
        printed_operations=context,
    )
    assert bool(s._formula_branch_test_witnesses(branch, **kwargs)) is accepted
    duplicated_text = text + "\nQuoted prose: " + branch.text
    duplicate_context = s._printed_operation_context(
        duplicated_text, (), "ca/policy/test"
    )
    assert duplicate_context.operation_kinds(branch) == frozenset({"add"})
    assert not duplicate_context.covers(branch, branch.text)
    assert (
        bool(
            s._formula_branch_test_witnesses(
                branch,
                **{
                    **kwargs,
                    "source_text": duplicated_text,
                    "printed_operations": duplicate_context,
                },
            )
        )
        is accepted
    )
    if accepted:
        foreign = s._printed_operation_context(text, (), "ca/policy/foreign")
        assert not s._formula_branch_test_witnesses(
            branch, **{**kwargs, "printed_operations": foreign}
        )
        conflict = type(
            "ConflictingReplacement",
            (),
            {"operation_kinds": lambda self, **kw: frozenset({"subtract"})},
        )()
        assert not s._formula_branch_test_witnesses(
            branch, **{**kwargs, "certified_replacement": conflict}
        )
