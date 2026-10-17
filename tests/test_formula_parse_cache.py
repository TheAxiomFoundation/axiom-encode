"""Formula syntax reuse must not reuse case values or retain unbounded text."""

import pytest

from axiom_encode.harness import source_completeness as sc


@pytest.fixture(autouse=True)
def empty_formula_parse_cache():
    sc._cached_first_formula_branch_node.cache_clear()
    yield
    sc._cached_first_formula_branch_node.cache_clear()


def test_reused_formula_syntax_still_executes_each_environment():
    formula = (
        'if label == "if match":\n'
        "  0 # else: 999\n"
        "else:\n"
        "  income # match income: 0 => 999\n"
    )
    executions = [
        sc._execute_formula_text(
            formula,
            environment=environment,
            constant_environment={},
        )
        for environment in (
            {"label": "if match", "income": 12},
            {"label": "other", "income": 12},
            {"label": "other", "income": 0},
        )
    ]

    assert all(execution is not None for execution in executions)
    assert [execution.trace[0].choice for execution in executions] == [0, 1, 1]
    assert [execution.evaluates_to_zero for execution in executions] == [
        True,
        False,
        True,
    ]
    assert executions[1].evaluated_value != executions[2].evaluated_value


def test_formula_parse_cache_stays_bounded_and_keeps_invalid_syntax_unresolved():
    cache = sc._cached_first_formula_branch_node
    malformed = "if eligible: 1"
    assert sc._first_formula_branch_node(malformed) is None
    assert sc._first_formula_branch_node(malformed) is None
    assert cache.cache_info().hits == 1
    assert sc._first_formula_branch_node(malformed + " else: 2") is not None

    for number in range(257):
        assert sc._first_formula_branch_node(f"input_{number}") is None

    assert cache.cache_info().currsize == 256


def test_oversized_formula_is_parsed_without_being_retained():
    formula = "if eligible: 1 else: " + " " * 16_384 + "2"
    node = sc._first_formula_branch_node(formula)

    assert node is not None
    assert node.start == 0
    assert node.kind == "if"
    assert node.selectors == ("eligible",)
    assert len(node.choices) == 2
    assert sc._cached_first_formula_branch_node.cache_info().currsize == 0
