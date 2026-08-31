"""Tests for bounded nested review-contract JSON values."""

from __future__ import annotations

from decimal import Decimal

import pytest

from axiom_encode.review_contract_values import (
    MAX_REVIEW_CONTRACT_CONTAINER_ITEMS,
    MAX_REVIEW_CONTRACT_VALUE_DEPTH,
    MAX_REVIEW_CONTRACT_VALUE_NODES,
    ReviewContractValueBudget,
    ReviewContractValueError,
    normalize_review_contract_value,
    review_contract_values_exactly_equal,
)


def test_normalizes_deep_copy_with_deterministically_sorted_mapping_keys() -> None:
    source = {
        "z": [{"nested": [True, 7, 1.25, "line one\nline two\tend"]}],
        "a": {"beta": 2, "alpha": ""},
    }

    normalized = normalize_review_contract_value(source)

    assert list(normalized) == ["a", "z"]
    assert list(normalized["a"]) == ["alpha", "beta"]
    source["z"][0]["nested"].append(9)
    source["a"]["beta"] = 99
    assert normalized == {
        "a": {"alpha": "", "beta": 2},
        "z": [{"nested": [True, 7, 1.25, "line one\nline two\tend"]}],
    }


@pytest.mark.parametrize(
    "value", [True, 0, -7, 1.25, "", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"]
)
def test_accepts_exact_builtin_scalar_leaves(value: object) -> None:
    normalized = normalize_review_contract_value(value)

    assert type(normalized) is type(value)
    assert normalized == value


@pytest.mark.parametrize(
    "value",
    [None, Decimal("1"), b"bytes", (1,), {1}, complex(1, 2)],
)
def test_rejects_non_json_or_null_leaves(value: object) -> None:
    with pytest.raises(ReviewContractValueError, match="exact builtin"):
        normalize_review_contract_value(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ReviewContractValueError, match="finite float"):
        normalize_review_contract_value(value)


class _IntSubclass(int):
    pass


class _StringSubclass(str):
    pass


class _ListSubclass(list[object]):
    pass


class _DictSubclass(dict[str, object]):
    pass


@pytest.mark.parametrize(
    "value",
    [_IntSubclass(1), _StringSubclass("x"), _ListSubclass(), _DictSubclass()],
)
def test_rejects_scalar_and_container_subclasses(value: object) -> None:
    with pytest.raises(ReviewContractValueError, match="exact builtin"):
        normalize_review_contract_value(value)


@pytest.mark.parametrize(
    "value",
    [
        " leading",
        "trailing ",
        "e\N{COMBINING ACUTE ACCENT}",
        "embedded\rreturn",
        "embedded\x00null",
        "embedded\x0bvertical-tab",
        "embedded\x7fdelete",
    ],
)
def test_rejects_noncanonical_string_leaves(value: str) -> None:
    with pytest.raises(ReviewContractValueError, match="string must be NFC"):
        normalize_review_contract_value(value)


@pytest.mark.parametrize(
    "key",
    [
        "",
        " leading",
        "trailing ",
        "e\N{COMBINING ACUTE ACCENT}",
        "embedded\nnewline",
        "embedded\ttab",
        "embedded\x00null",
        "embedded\x7fdelete",
    ],
)
def test_rejects_empty_or_noncanonical_mapping_keys(key: str) -> None:
    with pytest.raises(ReviewContractValueError, match="mapping keys"):
        normalize_review_contract_value({key: 1})


def test_rejects_nonbuiltin_string_mapping_key() -> None:
    with pytest.raises(ReviewContractValueError, match="mapping keys"):
        normalize_review_contract_value({_StringSubclass("key"): 1})


@pytest.mark.parametrize(
    "value",
    [
        list(range(MAX_REVIEW_CONTRACT_CONTAINER_ITEMS + 1)),
        {
            f"key-{index:02d}": index
            for index in range(MAX_REVIEW_CONTRACT_CONTAINER_ITEMS + 1)
        },
    ],
)
def test_rejects_oversized_list_or_mapping(value: object) -> None:
    with pytest.raises(ReviewContractValueError, match="at most 64 items"):
        normalize_review_contract_value(value)


def test_depth_boundary_allows_eight_edges_and_rejects_nine() -> None:
    at_limit: object = 0
    for _ in range(MAX_REVIEW_CONTRACT_VALUE_DEPTH):
        at_limit = [at_limit]

    assert normalize_review_contract_value(at_limit) == at_limit

    over_limit = [at_limit]
    with pytest.raises(ReviewContractValueError, match="maximum.*depth 8"):
        normalize_review_contract_value(over_limit)


def test_node_boundary_allows_2048_and_rejects_2049_nodes() -> None:
    at_limit = [[0] * 64 for _ in range(31)] + [[0] * 31]
    assert normalize_review_contract_value(at_limit) == at_limit

    over_limit = [[0] * 64 for _ in range(31)] + [[0] * 32]
    with pytest.raises(ReviewContractValueError, match="2048-node"):
        normalize_review_contract_value(over_limit)


@pytest.mark.parametrize("container_kind", ["list", "mapping"])
def test_rejects_cyclic_container_graphs(container_kind: str) -> None:
    if container_kind == "list":
        value: object = []
        value.append(value)  # type: ignore[attr-defined]
    else:
        value = {}
        value["self"] = value  # type: ignore[index]

    with pytest.raises(ReviewContractValueError, match="cyclic"):
        normalize_review_contract_value(value)
    assert not review_contract_values_exactly_equal(value, value)


def test_repeated_noncyclic_container_is_copied_at_each_location() -> None:
    shared = {"value": 1}

    normalized = normalize_review_contract_value([shared, shared])

    assert normalized == [{"value": 1}, {"value": 1}]
    assert normalized[0] is not normalized[1]


def test_budget_is_reusable_across_normalization_calls() -> None:
    budget = ReviewContractValueBudget(node_limit=3)

    assert normalize_review_contract_value([1], budget=budget) == [1]
    assert budget.used_nodes == 2
    assert budget.remaining_nodes == 1
    assert normalize_review_contract_value(True, budget=budget) is True
    assert budget.used_nodes == budget.node_limit == 3
    with pytest.raises(ReviewContractValueError, match="3-node"):
        normalize_review_contract_value("extra", budget=budget)


@pytest.mark.parametrize(
    "node_limit",
    [True, 0, -1, MAX_REVIEW_CONTRACT_VALUE_NODES + 1, 1.0, "2048"],
)
def test_budget_rejects_invalid_or_weakened_limits(node_limit: object) -> None:
    with pytest.raises(ValueError, match="node limit"):
        ReviewContractValueBudget(node_limit=node_limit)  # type: ignore[arg-type]


def test_exact_equality_ignores_mapping_order_but_preserves_list_order() -> None:
    left = {"b": [1, 2], "a": {"y": False, "x": "value"}}
    right = {"a": {"x": "value", "y": False}, "b": [1, 2]}

    assert review_contract_values_exactly_equal(left, right)
    assert not review_contract_values_exactly_equal(left, {**right, "b": [2, 1]})


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (True, 1),
        (True, 1.0),
        (1, 1.0),
        ({"value": False}, {"value": 0}),
        ([1], [1.0]),
        (0.0, -0.0),
    ],
)
def test_exact_equality_distinguishes_bool_int_and_float_values(
    left: object,
    right: object,
) -> None:
    assert not review_contract_values_exactly_equal(left, right)


@pytest.mark.parametrize("value", [None, float("nan"), {"bad key\n": 1}])
def test_invalid_values_compare_unequal_even_to_themselves(value: object) -> None:
    assert not review_contract_values_exactly_equal(value, value)
