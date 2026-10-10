"""A maximum witness needs exact source structure and its own Boolean effect."""

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
# Structural eligibility is decided before evaluating any supplied assignment;
# a coincident numerical value never authenticates a source maximum.
_LEAVES = ("true", "false", "sibling", "greater_guard", "impossible_guard")
_SIBLINGS = st.recursive(
    st.sampled_from(_LEAVES),
    lambda children: st.tuples(st.sampled_from(("and", "or")), children, children),
    max_leaves=4,
)
_WRAPPER = st.tuples(st.sampled_from(("and", "or")), st.booleans(), _SIBLINGS)
_COMPARISON_SHAPES = (
    "direct",
    "reverse_comparison",
    "ordinary_name",
    "alternative_name",
    "coincident_maximum",
    "unrelated_maximum",
    "cancelled_maximum",
    "impossible_maximum",
    "arbitrary_selector",
    "equal_guard",
    "ge_guard",
    "greater_guard",
    "unless_equal_guard",
    "unless_ge_guard",
    "wrong_guard_polarity",
    "wrong_guard_threshold",
    "wrong_guard_operand",
    "noncomplementary_guards",
    "overlapping_guards",
    "complementary_guards",
    "if_equal_guard",
    "if_unless_ge_guard",
)


def _comparison_tree(shape):
    if shape in {"direct", "reverse_comparison"}:
        return "source" if shape == "direct" else "reverse_source"
    if shape in {"ordinary_name", "alternative_name"}:
        return "ordinary" if shape == "ordinary_name" else "alternative"
    if shape in {"coincident_maximum", "unrelated_maximum"}:
        return shape
    if shape == "cancelled_maximum":
        return ("or", ("and", "source", "false"), "alternative")
    if shape == "impossible_maximum":
        return ("or", ("and", "impossible_guard", "true"), "alternative")
    if shape in {
        "noncomplementary_guards",
        "overlapping_guards",
        "complementary_guards",
    }:
        first, second = {
            "noncomplementary_guards": ("gt_guard", "reverse_gt_guard"),
            "overlapping_guards": ("ge_guard", "reverse_ge_guard"),
            "complementary_guards": ("ge_guard", "reverse_gt_guard"),
        }[shape]
        return ("or", ("and", first, "ordinary"), ("and", second, "alternative"))
    guard = {
        "arbitrary_selector": "arbitrary_guard",
        "equal_guard": "equal_guard",
        "ge_guard": "ge_guard",
        "greater_guard": "greater_guard",
        "unless_equal_guard": ("not", "equal_guard"),
        "unless_ge_guard": ("not", "ge_guard"),
        "wrong_guard_polarity": "equal_guard",
        "wrong_guard_threshold": "equal_guard",
        "wrong_guard_operand": "unrelated_guard",
        "if_equal_guard": "equal_guard",
        "if_unless_ge_guard": ("not", "ge_guard"),
    }[shape]
    true_arm, false_arm = "ordinary", "alternative"
    if shape in {
        "greater_guard",
        "unless_equal_guard",
        "unless_ge_guard",
        "wrong_guard_polarity",
        "if_unless_ge_guard",
    }:
        true_arm, false_arm = false_arm, true_arm
    if shape == "wrong_guard_threshold":
        false_arm = "wrong_threshold_alternative"
    if shape.startswith("if_"):
        return ("if", guard, true_arm, false_arm)
    selected = (
        "or",
        ("and", guard, true_arm),
        ("and", ("not", guard), false_arm),
    )
    if shape == "arbitrary_selector":
        # The exact strict-winner r13 attack: a reached, cancelled maximum
        # supplies observations while an arbitrary boundary chooses the arms.
        return ("or", ("and", "impossible_guard", "false"), selected)
    return selected


def _composition(wrappers, shape):
    tree = _comparison_tree(shape)
    # Inline RuleSpec selections live at expression entry. Their selected arms
    # are exact comparisons; Boolean composition is exercised by the OR forms.
    if shape.startswith("if_"):
        return tree
    for operator, source_first, sibling in wrappers:
        tree = (operator, tree, sibling) if source_first else (operator, sibling, tree)
    return tree


def _render(tree, reverse_arguments):
    if isinstance(tree, tuple):
        operator, *children = tree
        if operator == "not":
            return f"not ({_render(children[0], reverse_arguments)})"
        if operator == "if":
            guard, true_arm, false_arm = children
            return (
                f"if {_render(guard, reverse_arguments)}: "
                f"{_render(true_arm, reverse_arguments)} else: "
                f"{_render(false_arm, reverse_arguments)}"
            )
        left, right = children
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
        "reverse_source": f"first_threshold < {maximum}",
        "ordinary": "adjusted_gross_income > first_threshold",
        "alternative": "earned_income > first_threshold",
        "coincident_maximum": "max(earned_income, earned_income) > first_threshold",
        "unrelated_maximum": "max(earned_income, investment_income) > first_threshold",
        "wrong_threshold_alternative": "earned_income > second_threshold",
        "sibling": f"{maximum} > second_threshold",
        "equal_guard": f"{maximum} == adjusted_gross_income",
        "ge_guard": "adjusted_gross_income >= earned_income",
        "gt_guard": "adjusted_gross_income > earned_income",
        "reverse_ge_guard": "earned_income >= adjusted_gross_income",
        "reverse_gt_guard": "earned_income > adjusted_gross_income",
        "greater_guard": f"{maximum} > adjusted_gross_income",
        "impossible_guard": f"{maximum} < adjusted_gross_income",
        "arbitrary_guard": "earned_income == 8",
        "unrelated_guard": "adjusted_gross_income >= investment_income",
        "true": "true",
        "false": "false",
    }[tree]


def _winner_guard_arms(guard):
    """Authenticate winners and tie polarity without inspecting case values."""
    if isinstance(guard, tuple) and guard[0] == "not":
        arms = _winner_guard_arms(guard[1])
        return None if arms is None else (arms[1], arms[0], not arms[2])
    return {
        "equal_guard": ("ordinary", "alternative", True),
        "ge_guard": ("ordinary", "alternative", True),
        "gt_guard": ("ordinary", "alternative", False),
        "greater_guard": ("alternative", "ordinary", False),
        "reverse_gt_guard": ("alternative", "ordinary", False),
        "reverse_ge_guard": ("alternative", "ordinary", True),
    }.get(guard)


def _supported_source_comparisons(tree):
    if not isinstance(tree, tuple):
        return {tree} if tree in {"source", "reverse_source"} else set()
    operator, *children = tree
    supported = set().union(*(_supported_source_comparisons(c) for c in children))
    if operator == "if":
        guard, true_arm, false_arm = children
        arms = _winner_guard_arms(guard)
        if arms is not None and arms[:2] == (true_arm, false_arm):
            supported.update((true_arm, false_arm))
    if operator == "or":
        left, right = children
        if (
            isinstance(left, tuple)
            and isinstance(right, tuple)
            and left[0] == right[0] == "and"
            and (arms := _winner_guard_arms(left[1])) is not None
            and arms[:2] == (left[2], right[2])
            and _winner_guard_arms(right[1]) == (arms[1], arms[0], not arms[2])
        ):
            supported.update((left[2], right[2]))
    return supported


def _oracle(tree, values, *, flip_comparison=None):
    """Execute actual inputs and optionally intervene on one Boolean leaf.

    Every numeric expression always uses the executed inputs. The intervention
    changes only the designated comparison's Boolean value; it cannot invent a
    numeric maximum, switch another comparison, or alter a guard.
    """
    reached = set()

    def evaluate(node):
        if isinstance(node, tuple):
            operator, *children = node
            if operator == "not":
                return not evaluate(children[0])
            if operator == "if":
                guard, true_arm, false_arm = children
                return evaluate(true_arm if evaluate(guard) else false_arm)
            left, right = children
            left_result = evaluate(left)
            if operator == "and":
                return left_result and evaluate(right)
            return left_result or evaluate(right)
        ordinary = values["adjusted_gross_income"]
        alternative = values["earned_income"]
        maximum = max(ordinary, alternative)
        assert maximum >= ordinary
        assert maximum >= alternative
        result = {
            "source": maximum > values["first_threshold"],
            "reverse_source": maximum > values["first_threshold"],
            "ordinary": ordinary > values["first_threshold"],
            "alternative": alternative > values["first_threshold"],
            "coincident_maximum": alternative > values["first_threshold"],
            "unrelated_maximum": max(alternative, values["investment_income"])
            > values["first_threshold"],
            "wrong_threshold_alternative": alternative > values["second_threshold"],
            "true": True,
            "false": False,
            "sibling": maximum > values["second_threshold"],
            "equal_guard": maximum == ordinary,
            "ge_guard": ordinary >= alternative,
            "gt_guard": ordinary > alternative,
            "reverse_gt_guard": alternative > ordinary,
            "reverse_ge_guard": alternative >= ordinary,
            "greater_guard": maximum > ordinary,
            "impossible_guard": maximum < ordinary,
            "arbitrary_guard": alternative == 8,
            "unrelated_guard": ordinary >= values["investment_income"],
        }[node]
        reached.add(node)
        return not result if node == flip_comparison else result

    return evaluate(tree), reached


@settings(
    max_examples=80,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@example(
    wrappers=(),
    comparison_shape="arbitrary_selector",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="impossible_maximum",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=True,
)
@example(
    wrappers=(("and", True, "false"), ("or", True, "sibling")),
    comparison_shape="direct",
    second_threshold=10,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(("or", True, "sibling"),),
    comparison_shape="direct",
    second_threshold=10,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(("and", False, "greater_guard"),),
    comparison_shape="direct",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(("or", False, "impossible_guard"),),
    comparison_shape="direct",
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=True,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="direct",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(("and", True, "true"), ("or", False, "false")),
    comparison_shape="direct",
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=True,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="equal_guard",
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=True,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="ge_guard",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="unless_equal_guard",
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="if_unless_ge_guard",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=True,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="noncomplementary_guards",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="overlapping_guards",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="complementary_guards",
    second_threshold=12,
    changes_ordinary=False,
    reverse_arguments=False,
    tie=False,
)
@example(
    wrappers=(),
    comparison_shape="direct",
    second_threshold=12,
    changes_ordinary=True,
    reverse_arguments=False,
    tie=True,
)
@given(
    wrappers=st.lists(_WRAPPER, min_size=0, max_size=3).map(tuple),
    comparison_shape=st.sampled_from(_COMPARISON_SHAPES),
    second_threshold=st.sampled_from((7, 10, 12)),
    changes_ordinary=st.booleans(),
    reverse_arguments=st.booleans(),
    tie=st.booleans(),
)
def test_boolean_composition_credit_matches_independent_causal_oracle(
    wrappers,
    comparison_shape,
    second_threshold,
    changes_ordinary,
    reverse_arguments,
    tie,
):
    tree = _composition(wrappers, comparison_shape)
    formula = _render(tree, reverse_arguments)
    supported = _supported_source_comparisons(tree)
    payload, _, source, sentences, _ = _shared_credit_candidate(1)
    payload["inputs"].extend(
        {
            "name": name,
            "entity": "TaxUnit",
            "dtype": "Money",
            "period": "Year",
        }
        for name in ("second_threshold", "investment_income")
    )
    payload["rules"][0]["versions"][0]["formula"] = formula
    executions = []
    source_choices = []
    cases = []
    for index, changing in enumerate((9 if tie else 8, 11)):
        values = {
            "adjusted_gross_income": changing if changes_ordinary else 9,
            "earned_income": 9 if changes_ordinary else changing,
            "investment_income": 9,
            "first_threshold": 10,
            "second_threshold": second_threshold,
        }
        actual, reached = _oracle(tree, values)
        sensitive = any(
            actual != _oracle(tree, values, flip_comparison=comparison)[0]
            for comparison in supported & reached
        )
        required = max(values["adjusted_gross_income"], values["earned_income"]) > 10
        executions.append((actual, required, bool(supported & reached), sensitive))
        source_choices.append(values["earned_income"] > values["adjusted_gross_income"])
        execution = sc._execute_formula_text(
            formula, environment=values, constant_environment={}
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is actual
        cases.append(_case(values, actual, index))
    assert [execution[1] for execution in executions] == [False, True]
    # A guarded winner's selected comparison must contribute on each execution.
    # The guard itself need not be decisive: both incomes are below threshold
    # in the low execution, so switching its choice can leave the output false.
    # The source uses ordinary income at a tie. Raising ordinary income after
    # a tie leaves its "or, if greater" selection unchanged and cannot witness
    # that exception, even if the output and an exact max comparison change.
    expected_credit = set(source_choices) == {False, True} and all(
        actual == required and reached and sensitive
        for actual, required, reached, sensitive in executions
    )
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    credited = not any(sentences[0] in issue for issue in issues)
    assert credited is expected_credit, (
        formula,
        executions,
        source_choices,
        supported,
        issues,
    )
