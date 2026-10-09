"""A maximum witness needs the source comparison's own Boolean contribution."""

from __future__ import annotations

import pytest
import yaml
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from tests.test_source_completeness_max_witnesses import (
    _SHARED_CREDIT_CITATION,
    _SHARED_CREDIT_REFERENCE,
    _paired_issues,
    _shared_credit_candidate,
)


def _case(values, result, index):
    return {
        "name": f"causal_case_{index}",
        "period": {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        "input": {
            _SHARED_CREDIT_REFERENCE + "input." + name: value
            for name, value in values.items()
        },
        "output": {
            _SHARED_CREDIT_REFERENCE + "credit_allowed": (
                "holds" if result else "not_holds"
            )
        },
    }


def _proof(payload, source, count):
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={_SHARED_CREDIT_CITATION: source},
    )
    assert proof.passed
    assert proof.atoms_checked == count
    assert not proof.issues


def test_cancelled_source_comparison_cannot_borrow_a_sibling_maximum_effect():
    payload, _, _, sentences, _ = _shared_credit_candidate()
    sentences[1] = sentences[1].replace("investment income", "earned income")
    rule = payload["rules"][0]
    rule["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = sentences[1]
    source = "(1) " + " ".join(sentences)
    formula = (
        "(max(adjusted_gross_income, earned_income) > first_threshold and false) "
        "or max(adjusted_gross_income, earned_income) > second_threshold"
    )
    rule["versions"][0]["formula"] = formula
    cases = []
    for earned in (8, 13, 14):
        values = {
            "adjusted_gross_income": 9,
            "earned_income": earned,
            "investment_income": 8,
            "first_threshold": 10,
            "second_threshold": 12,
        }
        required = any(max(9, earned) > threshold for threshold in (10, 12))
        assert sc._evaluate_rulespec_formula(formula, environment=values) is required
        cases.append(_case(values, required, earned))
    # Two distinct crossing pairs defeated the previous allocation check.
    assert len(cases) == 3
    _proof(payload, source, 2)
    values = {
        "adjusted_gross_income": 9,
        "earned_income": 11,
        "first_threshold": 10,
        "second_threshold": 12,
    }
    assert max(values["adjusted_gross_income"], values["earned_income"]) > 10
    assert sc._evaluate_rulespec_formula(formula, environment=values) is False
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    assert any(sentences[0] in issue for issue in issues)


def test_impossible_maximum_replay_cannot_credit_a_cancelled_ordinary_arm():
    payload, _, source, sentences, _ = _shared_credit_candidate(1)
    formula = (
        "(max(adjusted_gross_income, earned_income) < adjusted_gross_income and true) "
        "or (adjusted_gross_income > first_threshold and false) "
        "or earned_income > first_threshold"
    )
    payload["rules"][0]["versions"][0]["formula"] = formula
    cases = []
    for index, earned in enumerate((8, 11)):
        values = {
            "adjusted_gross_income": 9,
            "earned_income": earned,
            "first_threshold": 10,
        }
        required = max(9, earned) > 10
        assert max(9, earned) >= values["adjusted_gross_income"]
        assert max(9, earned) >= values["earned_income"]
        assert sc._evaluate_rulespec_formula(formula, environment=values) is required
        cases.append(_case(values, required, index))
    _proof(payload, source, 1)
    values = {
        "adjusted_gross_income": 11,
        "earned_income": 8,
        "first_threshold": 10,
    }
    assert max(values["adjusted_gross_income"], values["earned_income"]) > 10
    assert sc._evaluate_rulespec_formula(formula, environment=values) is False
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    assert any(sentences[0] in issue for issue in issues)


@pytest.mark.parametrize(
    "sibling",
    (
        "earned_income / 10 + 0.1 == 1.2",
        "earned_income / 10 + 0.1 > 1.19999999999999999999",
    ),
)
def test_source_comparison_reachability_uses_executed_decimal_arithmetic(sibling):
    payload, _, source, sentences, _ = _shared_credit_candidate(1)
    formula = (
        f"({sibling}) or max(adjusted_gross_income, earned_income) > first_threshold"
    )
    payload["rules"][0]["versions"][0]["formula"] = formula
    cases = []
    for index, earned in enumerate((8, 11)):
        values = {
            "adjusted_gross_income": 9,
            "earned_income": earned,
            "first_threshold": 10,
        }
        required = max(9, earned) > 10
        assert sc._evaluate_rulespec_formula(formula, environment=values) is required
        assert sc._evaluate_rulespec_formula(sibling, environment=values) is bool(index)
        cases.append(_case(values, required, index))
    # Binary float arithmetic or reparsing an AST-rounded literal can invent a
    # reached, effective source comparison after the executed sibling is true.
    assert 11 / 10 + 0.1 != 1.2
    _proof(payload, source, 1)
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    assert any(sentences[0] in issue for issue in issues)


# These tuple trees and their evaluator are intentionally independent of the
# production parser, replay implementation, and comparison-reachability helpers.
# There is exactly one source-owned comparison, named "source"; siblings compare
# a different threshold or the maximum with one of its operands.
_LEAVES = ("true", "false", "sibling", "greater_guard", "impossible_guard")
_SIBLINGS = st.recursive(
    st.sampled_from(_LEAVES),
    lambda children: st.tuples(st.sampled_from(("and", "or")), children, children),
    max_leaves=4,
)
_WRAPPER = st.tuples(st.sampled_from(("and", "or")), st.booleans(), _SIBLINGS)


def _composition(wrappers):
    tree = "source"
    for operator, source_first, sibling in wrappers:
        tree = (operator, tree, sibling) if source_first else (operator, sibling, tree)
    return tree


def _render(tree, reverse_arguments):
    if isinstance(tree, tuple):
        operator, left, right = tree
        return (
            f"({_render(left, reverse_arguments)} {operator} "
            f"{_render(right, reverse_arguments)})"
        )
    maximum = (
        "max(earned_income, adjusted_gross_income)"
        if reverse_arguments
        else "max(adjusted_gross_income, earned_income)"
    )
    return {
        "source": f"{maximum} > first_threshold",
        "sibling": f"{maximum} > second_threshold",
        "greater_guard": f"{maximum} > adjusted_gross_income",
        "impossible_guard": f"{maximum} < adjusted_gross_income",
        "true": "true",
        "false": "false",
    }[tree]


def _oracle(tree, values, *, flip_source=False):
    """Execute actual inputs and optionally intervene on one Boolean leaf.

    Every numeric expression always uses the executed inputs. The intervention
    changes only the designated comparison's Boolean value; it cannot invent a
    numeric maximum, switch another comparison, or alter a guard.
    """
    reached = False

    def evaluate(node):
        nonlocal reached
        if isinstance(node, tuple):
            operator, left, right = node
            left_result = evaluate(left)
            if operator == "and":
                return left_result and evaluate(right)
            return left_result or evaluate(right)
        maximum = max(values["adjusted_gross_income"], values["earned_income"])
        assert maximum >= values["adjusted_gross_income"]
        assert maximum >= values["earned_income"]
        if node == "source":
            reached = True
            result = maximum > values["first_threshold"]
            return not result if flip_source else result
        return {
            "true": True,
            "false": False,
            "sibling": maximum > values["second_threshold"],
            "greater_guard": maximum > values["adjusted_gross_income"],
            "impossible_guard": maximum < values["adjusted_gross_income"],
        }[node]

    return evaluate(tree), reached


@settings(
    max_examples=80,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@example(
    wrappers=(("and", True, "false"), ("or", True, "sibling")),
    second_threshold=10,
    changes_ordinary=False,
    reverse_arguments=False,
)
@example(
    wrappers=(("or", True, "sibling"),),
    second_threshold=10,
    changes_ordinary=False,
    reverse_arguments=False,
)
@example(
    wrappers=(("and", False, "greater_guard"),),
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
)
@example(
    wrappers=(("or", False, "impossible_guard"),),
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=True,
)
@example(
    wrappers=(),
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
)
@example(
    wrappers=(("and", True, "true"), ("or", False, "false")),
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=True,
)
@given(
    wrappers=st.lists(_WRAPPER, min_size=0, max_size=3).map(tuple),
    second_threshold=st.sampled_from((7, 10, 12)),
    changes_ordinary=st.booleans(),
    reverse_arguments=st.booleans(),
)
def test_boolean_composition_credit_matches_independent_causal_oracle(
    wrappers, second_threshold, changes_ordinary, reverse_arguments
):
    tree = _composition(wrappers)
    formula = _render(tree, reverse_arguments)
    payload, _, source, sentences, _ = _shared_credit_candidate(1)
    payload["inputs"].append(
        {
            "name": "second_threshold",
            "entity": "TaxUnit",
            "dtype": "Money",
            "period": "Year",
        }
    )
    payload["rules"][0]["versions"][0]["formula"] = formula
    executions = []
    cases = []
    for index, changing in enumerate((8, 11)):
        values = {
            "adjusted_gross_income": changing if changes_ordinary else 9,
            "earned_income": 9 if changes_ordinary else changing,
            "first_threshold": 10,
            "second_threshold": second_threshold,
        }
        actual, reached = _oracle(tree, values)
        intervened, _ = _oracle(tree, values, flip_source=True)
        required = max(values["adjusted_gross_income"], values["earned_income"]) > 10
        executions.append((actual, required, reached, actual != intervened))
        assert sc._evaluate_rulespec_formula(formula, environment=values) is actual
        cases.append(_case(values, actual, index))
    assert [execution[1] for execution in executions] == [False, True]
    # The two assignments switch the maximum's binding operand in either
    # intervention direction, with no tie or unrelated changed input.
    expected_credit = all(
        actual == required and reached and sensitive
        for actual, required, reached, sensitive in executions
    )
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    credited = not any(sentences[0] in issue for issue in issues)
    assert credited is expected_credit, (formula, executions, issues)
