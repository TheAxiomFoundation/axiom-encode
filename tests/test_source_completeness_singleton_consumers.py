"""Audit singleton assertion consumers and their queried-row boundary."""

from __future__ import annotations

import ast
import copy
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import worksheet_composition_evidence as composition
from axiom_encode.harness import worksheet_exception_evidence as exception
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness import worksheet_replacement_evidence as replacement
from tests import test_worksheet_composition_evidence as fixture


def _reader_callsites():
    calls = []

    class Readers(ast.NodeVisitor):
        def __init__(self, module):
            self.module = module
            self.functions = []

        def visit_FunctionDef(self, node):
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            if name == "_test_case_asserted_output_value":
                calls.append(
                    (
                        self.module,
                        self.functions[-1],
                        node.lineno,
                        any(k.arg == "principal_rules" for k in node.keywords),
                    )
                )
            self.generic_visit(node)

    for path in sorted(Path(sc.__file__).parent.glob("*.py")):
        Readers(path.stem).visit(ast.parse(path.read_text()))
    return tuple(calls)


CALLSITES = _reader_callsites()
EXPECTED_CONSUMERS = {
    ("printed_signed_format_evidence", "witnesses"): 1,
    ("actual_floor_evidence", "witnesses"): 1,
    ("actual_sparse_worksheet_evidence", "witnesses"): 1,
    ("same_case_execution", "formula_witnesses"): 1,
    ("reporting_route_evidence", "certify_reporting_routes"): 1,
    ("partnership_reporting_evidence", "certify_partnership_reporting"): 1,
    ("measurement_role_evidence", "evidence"): 1,
    ("measurement_role_evidence", "_current_evidence"): 1,
    ("source_completeness", "_annual_source_evidence"): 1,
    ("source_completeness", "_formula_execution_is_source_branch_witness"): 2,
    ("source_completeness", "_unasserted_reached_formula_dependency_names"): 1,
    ("source_completeness", "_case_dependency_environment"): 2,
    ("source_completeness", "_unconditional_nonapplicability_witnesses"): 1,
    ("source_completeness", "_source_owned_residence_text_intervention"): 2,
    ("source_completeness", "_source_qualified_cardinality_evidence"): 1,
    ("source_completeness", "_source_owned_equality_witness"): 1,
    ("source_completeness", "_toggled_formula_calendar_age_selectors"): 2,
    ("source_completeness", "_toggled_formula_numeric_selectors"): 2,
    ("source_completeness", "_case_pair_stable_asserted_formula_dependencies"): 2,
    ("source_completeness", "_exception_witness_for_case_pair"): 2,
    ("source_completeness", "_closed_fractional_rounding_operand_witness"): 2,
    ("source_completeness", "_fractional_rounding_case_witnesses"): 1,
    ("source_path_shadow", "corroborated_case_links"): 1,
    ("worksheet_composition_evidence", "certify"): 2,
    ("worksheet_country_evidence", "_certify"): 1,
    ("worksheet_exception_evidence", "existing_witnesses"): 1,
    ("worksheet_operation_evidence", "_zero_and_computation_witnesses"): 3,
    ("worksheet_operation_evidence", "certify_worksheet_operations"): 1,
    ("worksheet_replacement_evidence", "certify_worksheet_replacement"): 2,
    ("worksheet_transfer_evidence", "certify_worksheet_transfers"): 1,
}


def test_singleton_reader_consumer_inventory_is_complete():
    # A new caller must explicitly join the audit and its evidence regressions.
    assert Counter((module, name) for module, name, _, _ in CALLSITES) == Counter(
        EXPECTED_CONSUMERS
    )


@pytest.mark.parametrize(
    "callsite",
    CALLSITES,
    ids=[f"{m}.{f}:{line}" for m, f, line, _ in CALLSITES],
)
@settings(max_examples=15, deadline=None)
@given(
    flat=st.integers(min_value=-1_000_000, max_value=1_000_000),
    difference=st.integers(min_value=1, max_value=1_000_000),
    entity=st.sampled_from(["Payment", "Person", "CalculationUnit"]),
)
def test_every_reader_callsite_refuses_wrong_row_scalar_credit(
    callsite, flat, difference, entity
):
    """Exercise each actual reader mode, including opt-in dependency readers.

    Raw consumers receive the original list. Bound consumers receive no value.
    In either mode the singleton cannot corroborate the flat replay's scalar.
    Presence-only diagnostic/projection callers do not consume a scalar value.
    """

    rule = {
        "name": "result",
        "kind": "derived",
        "entity": entity,
        "dtype": "Integer",
        "versions": [{"formula": "income"}],
    }
    rules = {"result": rule}
    case = {
        "input": {"income": flat},
        "tables": {entity: [{"income": flat + difference}]},
        "output": {"result": [flat]},
    }
    kwargs = {"principal_rules": rules} if callsite[3] else {}
    asserted = sc._test_case_asserted_output_value(case, "result", **kwargs)
    assert not sc._asserted_formula_runtime_values_equal(rule, flat, asserted)


@settings(max_examples=40, deadline=None)
@given(
    flat=st.integers(min_value=-1_000_000, max_value=1_000_000),
    difference=st.integers(min_value=1, max_value=1_000_000),
)
def test_bound_scalar_sibling_and_dependencies_refuse_wrong_row(flat, difference):
    rules = {
        name: {
            "name": name,
            "kind": "derived",
            "entity": "Payment",
            "dtype": "Integer",
            "versions": [{"formula": "income"}],
        }
        for name in ("scalar_result", "listed_result")
    }
    case = {
        "input": {"income": flat},
        "tables": {"Payment": [{"income": flat + difference}]},
        "output": {"scalar_result": flat, "listed_result": [flat]},
    }
    untouched = copy.deepcopy(case)
    for name in rules:
        assert (
            sc._test_case_asserted_output_value(case, name, principal_rules=rules)
            is sc._UNRESOLVED_CONDITION_VALUE
        )
    assert not sc._case_asserted_dependency_environment(
        rules, case, formula_environment={}, allow_bound_row_assertions=True
    )
    assert case == untouched


@pytest.fixture(scope="module")
def wrong_row_worksheet_bound():
    payload, cases = copy.deepcopy((fixture.PAYLOAD, fixture.CASES))
    first = cases[0]
    key = next(k for k in first["output"] if k.endswith("#" + op._LINE2))
    first["output"][key] = [first["output"][key]]
    row = copy.deepcopy(first["input"])
    credit = next(k for k in row if k.endswith("#input." + op._CREDIT))
    row[credit] += 1
    first["tables"] = {"Person": [row]}
    # Rebuild real producer certificates after the mutation. Stale identities
    # cannot be the reason the downstream consumers refuse this evidence.
    return fixture.build(payload, cases)


def test_wrong_row_cannot_supply_worksheet_replacement(wrong_row_worksheet_bound):
    bound = wrong_row_worksheet_bound
    result = replacement.certify_worksheet_replacement(
        bound.payload, bound.cases, bound.source, bound.context, bound.bundle
    )
    assert result.unresolved
    assert not result.records


def test_wrong_row_cannot_supply_worksheet_composition(wrong_row_worksheet_bound):
    result = composition.certify(wrong_row_worksheet_bound)
    assert result.unresolved
    assert not result.execution_records


def test_wrong_row_cannot_supply_existing_worksheet_exception(
    wrong_row_worksheet_bound,
):
    bound = wrong_row_worksheet_bound
    rules = op.paths._index(bound.payload["rules"])
    asserted = {
        name: [case for case in bound.cases if name in sc._test_case_output_names(case)]
        for name in rules
    }
    witnesses = sc._toggled_formula_numeric_selectors(
        rules,
        asserted_by_rule=asserted,
        formula_environment=sc._constant_rule_environment(bound.payload),
    )
    branches = sc.recognize_source_structure(bound.source)
    branch = next(
        b
        for b in sc._source_exception_branches(
            bound.source,
            branches=branches,
            active_branches=branches,
            deferred_paths=set(),
        )
        if (b.start, b.end) == exception.SPAN
    )
    assert not exception.existing_witnesses(
        bound,
        branch,
        witnesses,
        principal_rules=rules,
        asserted_by_rule=asserted,
    )
