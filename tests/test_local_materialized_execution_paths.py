"""Exact materializer path controls; saved replay seams are not fresh evidence."""

from copy import deepcopy

import pytest

from axiom_encode.harness.local_relation_source_evidence import (
    _ordered_conditional_path,
)
from axiom_encode.harness.source_completeness import _FormulaTraceStep as Step
from axiom_encode.harness.validator_pipeline import _repair_else_if_formula
from tests import test_local_expense_group_execution as expense
from tests import test_local_remaining_chart_execution as chart


@pytest.mark.parametrize("count", [1, 2, 3])
def test_all_ordered_arms_and_corruptions(count):
    selectors = tuple(f"selector_{i}" for i in range(count))
    for choice in range(count + 1):
        flat = (Step("if", selectors[: choice + 1], choice),)
        nested = tuple(
            Step("if", (selectors[i],), 0 if i == choice else 1)
            for i in range(min(choice + 1, count))
        )
        for trace in (flat, nested):
            assert _ordered_conditional_path(trace, selectors, choice)
            assert not _ordered_conditional_path(
                trace + (Step("if", ("extra",), 0),), selectors, choice
            )
            assert not _ordered_conditional_path(trace[:-1], selectors, choice)
            for i, step in enumerate(trace):
                for wrong in (
                    Step("match", step.selectors, step.choice),
                    Step("if", ("other",), step.choice),
                    Step("if", step.selectors, True),
                    Step("if", step.selectors, step.choice + 1),
                ):
                    changed = trace[:i] + (wrong,) + trace[i + 1 :]
                    assert not _ordered_conditional_path(changed, selectors, choice)
    for choice in (-1, count + 1, True, None):
        assert not _ordered_conditional_path((), selectors, choice)


def materialized(data):
    result = deepcopy(data)
    for rule in result["rules"]:
        for version in rule.get("versions", []):
            if isinstance(version.get("formula"), str):
                version["formula"] = _repair_else_if_formula(version["formula"])
    return result


@pytest.mark.parametrize(
    "span", [(7150, 7346), (8017, 8067), (8068, 8159), (8160, 8308)]
)
def test_materialized_remaining_chart_seams(monkeypatch, span):
    monkeypatch.setattr(chart, "DATA", materialized(chart.DATA))
    chart.test_complete_roles(monkeypatch, span, "none")


def test_materialized_expense_seam(monkeypatch):
    monkeypatch.setattr(expense, "DATA", materialized(expense.DATA))
    expense.test_whole_group(monkeypatch, "none")
