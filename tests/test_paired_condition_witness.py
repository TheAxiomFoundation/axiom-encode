"""Per-part witnesses for paired source conditions (#1920).

The properties here are the module's invariants: a faithful formula with a full
companion grid passes; a formula that flips a bound's direction, swaps an
inclusive bound for an exclusive one (or back) or shifts it by one is refused;
each part needs its own case pair; and the parser's work grows linearly with the
source text.
"""

from __future__ import annotations

import functools
import itertools
from decimal import Decimal

import pytest
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import paired_condition_witness as pcw
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.paired_condition_differential import (
    PAIRED,
    analyze_with_part_witness_differential,
)

CITATION = "de/statute/estg/32a"


def _analyze(source: str, formula: str, cases: list[dict], *, citation=CITATION):
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
  - name: result
    kind: derived
    source: {citation}(1)
    versions:
      - formula: '{formula}'
"""
    return analyze_with_part_witness_differential(
        content,
        f"(1) {source}",
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _paired(result) -> bool:
    return any(PAIRED in issue for issue in result.issues)


def _case(inputs: dict, output: bool) -> dict:
    return {
        "name": str(tuple(inputs.values())),
        "period": "2026",
        "input": inputs,
        "output": {"result": output},
    }


# --- parsing -----------------------------------------------------------------


def test_compound_inclusive_condition_parses_into_its_parts():
    condition = pcw.parse_paired_condition(
        "(1) If income is greater than or equal to the income threshold and the "
        "applicant is a resident, the applicant is ineligible."
    )
    assert condition is not None
    comparison, flag = condition.parts
    assert (comparison.left.tokens, comparison.relation, comparison.right.tokens) == (
        ("income",),
        ">=",
        ("income", "threshold"),
    )
    assert (flag.kind, flag.subject, flag.attribute) == (
        "flag",
        ("applicant",),
        "resident",
    )
    assert condition.tree.kind == "and"
    assert condition.requirement == "exclude"
    assert pcw.is_active(condition)


@pytest.mark.parametrize(
    "phrase, relation",
    (
        ("is equal to or exceeds", ">="),
        ("equals or exceeds", ">="),
        ("exceeds or equals", ">="),
        ("is at least", ">="),
        ("exceeds", ">"),
        ("is more than", ">"),
        ("does not exceed", "<="),
        ("is at most", "<="),
        ("is less than", "<"),
    ),
)
def test_relation_phrases_read_as_one_relation(phrase, relation):
    condition = pcw.parse_paired_condition(
        f"If income {phrase} the income threshold, the applicant is ineligible."
    )
    assert condition is not None
    (part,) = condition.parts
    assert part.relation == relation


def test_shared_heads_and_compound_operands_keep_their_full_names():
    shared = pcw.parse_paired_condition(
        "If income is greater than or equal to both the earned and unearned income "
        "threshold, the applicant is ineligible."
    )
    compound = pcw.parse_paired_condition(
        "If total earned and unearned income is greater than or equal to the income "
        "threshold, the applicant is ineligible."
    )
    assert shared is not None and compound is not None
    assert [part.right.tokens for part in shared.parts] == [
        ("earned", "income", "threshold"),
        ("unearned", "income", "threshold"),
    ]
    (part,) = compound.parts
    assert part.left.tokens == ("total", "earned", "and", "unearned", "income")


def test_distributed_parts_name_their_own_operands():
    condition = pcw.parse_paired_condition(
        "If both income and assets exceed both the gross and net income limit, the "
        "applicant is ineligible."
    )
    assert condition is not None
    # The relation keeps the source's own words.
    assert [part.text for part in condition.parts] == [
        "income exceed gross income limit",
        "income exceed net income limit",
        "assets exceed gross income limit",
        "assets exceed net income limit",
    ]


def test_digit_group_commas_stay_inside_their_number():
    condition = pcw.parse_paired_condition(
        "If income exceeds $20,000.50 and the applicant is a resident, the applicant "
        "is ineligible."
    )
    assert condition is not None
    assert condition.parts[0].right.literal == Decimal("20000.50")


@pytest.mark.parametrize(
    "clause, requirement",
    (
        (
            "If the applicant qualifies, the credit is not allowed, provided income "
            "does not exceed the income threshold.",
            "exclude",
        ),
        (
            "Unless the applicant is disabled, the credit is allowed, provided income "
            "does not exceed the income threshold.",
            "enable",
        ),
        (
            "Provided that the applicant is disabled, the credit is not allowed, "
            "provided income does not exceed the income threshold.",
            "exclude",
        ),
        (
            "The individual (and spouse or children if required) is ineligible, "
            "provided income does not exceed the income threshold.",
            "exclude",
        ),
        (
            "If the applicant qualifies, the worker reviews the record, provided "
            "income does not exceed the income threshold.",
            "change",
        ),
    ),
)
def test_proviso_frames_join_their_conditions(clause, requirement):
    condition = pcw.parse_paired_condition(clause)
    assert condition is not None
    assert condition.requirement == requirement
    assert condition.parts[-1].relation == "<="


@pytest.mark.parametrize(
    "clause",
    (
        # Mixed connectors without parentheses have no single reading.
        "If income exceeds the income threshold and the applicant is a resident or "
        "the applicant is disabled, the applicant is ineligible.",
        # Wording outside the grammar.
        "If the household has a disabled member and income exceeds the limit, the "
        "applicant is ineligible.",
        "If income exceeds 165% of the limit and the applicant is a resident, the "
        "applicant is ineligible.",
        "When the eligibility specialist has determined income, apply the deduction.",
        "If income, as defined above, exceeds the limit, the applicant is ineligible.",
        # A cue inside the effect is a qualification the grammar does not read.
        "If income exceeds the limit, the applicant is ineligible if required.",
    ),
)
def test_unknown_or_ambiguous_clauses_are_left_to_the_clause_witness(clause):
    assert pcw.parse_paired_condition(clause) is None


def test_single_literal_comparisons_and_single_flags_stay_with_the_clause_witness():
    literal = pcw.parse_paired_condition(
        "If income is greater than or equal to 100, the applicant is ineligible."
    )
    flag = pcw.parse_paired_condition(
        "If the applicant is disabled, the applicant is ineligible."
    )
    assert literal is not None and flag is not None
    assert not pcw.is_active(literal)
    assert not pcw.is_active(flag)


# --- witnesses ---------------------------------------------------------------

_SOURCE = (
    "If income is greater than or equal to the income threshold and the applicant "
    "is a resident, the applicant is ineligible."
)
_FORMULA = "if income >= income_threshold and is_resident: false else: true"


def _grid(incomes=(99, 100, 101), residents=(False, True), *, threshold=100):
    return [
        _case(
            {"income": income, "income_threshold": threshold, "is_resident": resident},
            not (income >= threshold and resident),
        )
        for income, resident in itertools.product(incomes, residents)
    ]


def test_full_grid_passes_and_changes_nothing():
    result = _analyze(_SOURCE, _FORMULA, _grid())
    assert not result.issues
    assert not result.part_gaps
    assert not result.changed


@pytest.mark.parametrize(
    "cases, missing",
    (
        # Residence is toggled only below the threshold, where it cannot decide.
        (
            [
                _case({"income": 99, "income_threshold": 100, "is_resident": r}, True)
                for r in (False, True)
            ]
            + [
                _case(
                    {"income": i, "income_threshold": 100, "is_resident": True}, i < 100
                )
                for i in (99, 100)
            ],
            "the applicant is a resident",
        ),
        # Income is toggled only for non-residents, where it cannot decide.
        (
            [
                _case(
                    {"income": i, "income_threshold": 100, "is_resident": False}, True
                )
                for i in (99, 100)
            ]
            + [
                _case({"income": 100, "income_threshold": 100, "is_resident": r}, not r)
                for r in (False, True)
            ],
            "income is greater than or equal to the income threshold",
        ),
    ),
)
def test_each_part_needs_its_own_pair(cases, missing):
    result = _analyze(_SOURCE, _FORMULA, cases)
    assert _paired(result)
    assert result.changed
    (gap,) = result.part_gaps
    assert [part.text for part in gap.parts] == [missing]
    assert any(
        f"`{missing}` needs its own same-period case pair" in issue
        for issue in result.issues
    )


def test_a_pair_skipping_equality_cannot_witness_an_inclusive_named_bound():
    result = _analyze(_SOURCE, _FORMULA, _grid(incomes=(99, 101)))
    assert _paired(result)
    assert result.changed
    assert "including a case at equality" in " ".join(result.issues)


def test_a_proviso_clause_without_another_cue_becomes_a_demand():
    source = (
        "Provided that the applicant is disabled, the credit is not allowed, provided "
        "income does not exceed the income threshold."
    )
    faithful = [
        _case(
            {"income": 100, "income_threshold": threshold, "is_disabled": disabled},
            not (disabled and 100 <= threshold),
        )
        for threshold, disabled in itertools.product((99, 100, 101), (False, True))
    ]
    formula = "if is_disabled and income_threshold >= income: false else: true"
    assert not _analyze(source, formula, faithful).issues
    sparse = [case for case in faithful if case["input"]["is_disabled"]]
    result = _analyze(source, formula, sparse)
    assert _paired(result)
    assert result.changed


def test_the_us_manual_numeric_limit_filter_still_applies_to_proviso_clauses():
    source = (
        "Provided that the applicant is disabled, the credit is not allowed, provided "
        "income does not exceed the income threshold."
    )
    result = _analyze(
        source,
        "if is_disabled: false else: true",
        [_case({"income": 100, "income_threshold": 100, "is_disabled": True}, False)],
        citation="us-fl/manual/test/section-1",
    )
    assert not result.changed


# --- generated boundary defects -------------------------------------------------

_PHRASES = {
    ">=": ("is greater than or equal to", "is equal to or greater than"),
    ">": ("is greater than", "exceeds"),
    "<=": ("is less than or equal to", "does not exceed"),
    "<": ("is less than",),
}
_SWAPPED = {">=": ">", ">": ">=", "<=": "<", "<": "<="}
_FLIPPED = {">=": "<=", ">": "<", "<=": ">=", "<": ">"}


def _holds(value, relation, bound):
    return {
        ">=": value >= bound,
        ">": value > bound,
        "<=": value <= bound,
        "<": value < bound,
    }[relation]


@st.composite
def _paired_conditions(draw):
    relation = draw(st.sampled_from(sorted(_PHRASES)))
    named = draw(st.booleans())
    threshold = draw(st.integers(min_value=2, max_value=10_000))
    connector = draw(st.sampled_from(("and", "or")))
    comparison_first = draw(st.booleans())
    phrase = draw(st.sampled_from(_PHRASES[relation]))
    bound_text = "the income threshold" if named else str(threshold)
    comparison = f"income {phrase} {bound_text}"
    flag = "the applicant is a resident"
    parts = (comparison, flag) if comparison_first else (flag, comparison)
    return {
        "relation": relation,
        "named": named,
        "threshold": threshold,
        "connector": connector,
        "comparison_first": comparison_first,
        "source": f"If {f' {connector} '.join(parts)}, the applicant is ineligible.",
    }


def _formula_and_cases(scenario, relation, *, shift=0):
    threshold = scenario["threshold"]
    named = scenario["named"]
    bound = "income_threshold" if named else str(threshold)
    if shift:
        bound = f"{bound} + {shift}" if named else str(threshold + shift)
    comparison = f"income {relation} {bound}"
    connector = scenario["connector"]
    parts = (
        (comparison, "is_resident")
        if scenario["comparison_first"]
        else ("is_resident", comparison)
    )
    formula = f"if {f' {connector} '.join(parts)}: false else: true"
    cases = []
    for income, resident in itertools.product(
        (threshold - 1, threshold, threshold + 1), (False, True)
    ):
        comparison_holds = _holds(income, relation, threshold + shift)
        blocked = (
            comparison_holds and resident
            if connector == "and"
            else comparison_holds or resident
        )
        inputs = {"income": income, "is_resident": resident}
        if named:
            inputs["income_threshold"] = threshold
        # Outputs are what the formula computes, so only the source can object.
        cases.append(_case(inputs, not blocked))
    return formula, cases


@settings(max_examples=40, deadline=None, phases=(Phase.generate,), database=None)
@given(scenario=_paired_conditions())
def test_generated_boundary_defects_are_refused_and_faithful_formulas_pass(scenario):
    relation = scenario["relation"]
    source = scenario["source"]
    faithful = _analyze(source, *_formula_and_cases(scenario, relation))
    assert not _paired(faithful), (source, faithful.issues)
    assert not faithful.part_gaps
    if scenario["named"]:
        # A literal bound also meets the separate numeric-recall gate.
        assert not faithful.issues, (source, faithful.issues)
    for defect, shift in (
        (_SWAPPED[relation], 0),
        (_FLIPPED[relation], 0),
        (relation, 1),
        (relation, -1),
    ):
        result = _analyze(source, *_formula_and_cases(scenario, defect, shift=shift))
        assert _paired(result), (source, defect, shift)


# --- work bound -------------------------------------------------------------------


_CLAUSE = (
    "If income is greater than or equal to the income threshold and the applicant "
    "is a resident, the applicant is ineligible, provided assets do not exceed the "
    "asset threshold."
)


def _parse_operations(source: str) -> int:
    budget = pcw._Budget()
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    pcw.proviso_condition_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        deferred_clauses=(),
        formula_branches=(),
        numeric_limits_only=False,
        exception_branches=(),
        budget=budget,
    )
    for start, end, text in sc._source_clause_spans(source, branches=branches):
        pcw.parse_paired_condition(text, budget=budget)
    return budget.operations


def test_clause_reading_work_grows_linearly_with_the_source():
    sizes = (4, 8, 16, 32, 64)
    sources = [" ".join([_CLAUSE] * size) for size in sizes]
    operations = [_parse_operations(source) for source in sources]
    per_character = [count / len(source) for count, source in zip(operations, sources)]
    # Constant work per source character, so doubling the source at most
    # doubles the work (with a little slack for the fixed per-call cost).
    assert max(per_character) <= 16
    for smaller, larger in zip(operations, operations[1:]):
        assert larger <= 2.1 * smaller


@pytest.mark.parametrize("repeats", (100, 1_000))
def test_an_oversized_clause_is_left_unread_in_linear_work(repeats):
    clause = (
        "If "
        + " and ".join(f"income{index} exceeds the limit" for index in range(repeats))
        + ", the applicant is ineligible."
    )
    budget = pcw._Budget()
    assert pcw.parse_paired_condition(clause, budget=budget) is None
    assert budget.operations <= 2 * len(clause)


def test_connector_heavy_clauses_stay_within_the_per_clause_cap():
    # Twelve connectors is the most one group may split at; each DP cell is
    # charged, so the count bounds the quadratic step by a constant.
    clause = (
        "If "
        + " and ".join(
            f"{name} exceeds the limit"
            for name in ("income", "assets", "savings", "earnings")
        )
        + ", the applicant is ineligible."
    )
    budget = pcw._Budget()
    assert pcw.parse_paired_condition(clause, budget=budget) is not None
    assert budget.operations <= 40 * len(clause)
