"""Fail any test whose analyzer verdict the per-part condition witness changes.

The per-part witness only adds paired demands. When one of its demands reaches
an issue, the gate calls ``paired_condition_witness.note_refused_parts``. This
plugin records those calls per test, so the whole suite works as a differential
between the clause witness alone and the clause witness plus part witnesses: a
test whose verdict changes must be listed in ``REVIEWED_CHANGES`` with the
reason its formula or companion grid is wrong. Analyzer runs in subprocesses are
not observed.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness import paired_condition_witness as pcw

_RELATION_GUARDS = "tests/test_source_completeness_manual_relation_guards.py"
_CONDITION_SEMANTICS = "tests/test_source_completeness_manual_condition_semantics.py"
_PROCEDURE_REGRESSIONS = (
    "tests/test_source_completeness_manual_procedure_regressions.py"
)
_PART_WITNESS = "tests/test_paired_condition_witness.py"

# Test functions (all parametrizations) whose verdict the part witness changes,
# each with the defect the part witness shows.
REVIEWED_CHANGES = {
    f"{_RELATION_GUARDS}::test_inclusive_dynamic_bound_rejects_boolean_shortcut_without_equality": (
        "a Boolean input named after the comparison stands for an inclusive "
        "comparison of two named amounts; the formula never compares them"
    ),
    f"{_RELATION_GUARDS}::test_unreadable_proviso_effect_rejects_boolean_shortcut": (
        "the qualification is never read and the proviso's comparison is a "
        "Boolean shortcut"
    ),
    f"{_RELATION_GUARDS}::test_inclusive_disjuncts_each_require_their_own_equality_case": (
        "one inclusive disjunct has no case at equality"
    ),
    f"{_RELATION_GUARDS}::test_compound_inclusive_comparison_rejects_incorrect_boundary": (
        "a resident toggle stood for the income comparison: equality missing, "
        "bound shifted, inclusivity flipped or the threshold swapped"
    ),
    f"{_RELATION_GUARDS}::test_compound_inclusive_equality_is_satisfiable_and_cannot_shift": (
        "the inclusive bound is shifted by a constant"
    ),
    f"{_CONDITION_SEMANTICS}::test_round4_compound_operand_projection_keeps_the_obligation": (
        "the formula compares a projected operand, not the source's compound amount"
    ),
    f"{_CONDITION_SEMANTICS}::test_shared_quantity_head_cannot_be_projected_from_enumerated_operands": (
        "the formula compares `earned`, not the earned income threshold"
    ),
    f"{_CONDITION_SEMANTICS}::test_distinct_source_subjects_keep_independent_flag_obligations": (
        "the household's flag is a separate condition the formula never reads"
    ),
    f"{_CONDITION_SEMANTICS}::test_generated_provided_example_keeps_the_inclusive_obligation": (
        "the proviso includes equality; the formula's comparison is strict"
    ),
    f"{_CONDITION_SEMANTICS}::test_generated_source_conditions_require_semantic_equivalence": (
        "generated defects: flipped inclusivity, swapped operands, dropped parts "
        "and sparse grids that never toggle one part"
    ),
    f"{_PROCEDURE_REGRESSIONS}::test_verbatim_diagnostic_case_grids_keep_their_paired_issue": (
        "review grids that drop, project or shift one part of the condition"
    ),
    f"{_PROCEDURE_REGRESSIONS}::test_verbatim_diagnostic_probes_record_the_clause_witness_verdict": (
        "the same review grids, checked against the clause witness's verdict"
    ),
    "tests/test_source_credit_difference_conditions.py"
    "::test_credit_difference_preserves_all_antecedent_fact_demands": (
        "the incomplete variants drop one antecedent flag from the formula; the "
        "faithful formula still passes"
    ),
    f"{_PART_WITNESS}::test_generated_boundary_defects_are_refused_and_faithful_formulas_pass": (
        "generated boundary flips and inclusive/exclusive swaps"
    ),
    f"{_PART_WITNESS}::test_each_part_needs_its_own_pair": (
        "a pair that toggles one part stands for the other"
    ),
    f"{_PART_WITNESS}::test_a_pair_skipping_equality_cannot_witness_an_inclusive_named_bound": (
        "no case at equality for an inclusive comparison of two named amounts"
    ),
    f"{_PART_WITNESS}::test_a_proviso_clause_without_another_cue_becomes_a_demand": (
        "a sparse grid never toggles the qualification"
    ),
}


class _Audit:
    records: list[tuple[pcw.ConditionPartGap, ...]] | None = None


def _record(gaps) -> None:
    if gaps and _Audit.records is not None:
        _Audit.records.append(tuple(gaps))


def _function_id(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


def pytest_configure(config):
    patch = pytest.MonkeyPatch()
    patch.setattr(pcw, "note_refused_parts", _record)
    config._paired_condition_audit_patch = patch


def pytest_unconfigure(config):
    patch = getattr(config, "_paired_condition_audit_patch", None)
    if patch is not None:
        patch.undo()


@pytest.fixture(autouse=True)
def _paired_condition_verdict_audit(request):
    records: list[tuple[pcw.ConditionPartGap, ...]] = []
    _Audit.records = records
    yield
    _Audit.records = None
    if records and _function_id(request.node.nodeid) not in REVIEWED_CHANGES:
        refused = sorted(
            {
                f"`{gap.branch.text[:160]}`: "
                + "; ".join(part.text for part in gap.parts)
                for gaps in records
                for gap in gaps
            }
        )
        pytest.fail(
            "The per-part condition witness changed this test's verdict. Show the "
            "formula or grid is wrong and list the test in "
            "tests/paired_condition_audit.py REVIEWED_CHANGES, or fix the witness:\n"
            + "\n".join(refused),
            pytrace=False,
        )
