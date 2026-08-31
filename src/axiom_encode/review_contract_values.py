"""Bounded canonical values for structured review-contract test cases.

The signing path consumes review contracts from more than one entry point.  This
module keeps their nested JSON-value rules pure and shared: callers can reuse one
budget while normalizing a case's input and required-output mappings, then use
the exact comparator without Python's ``True == 1 == 1.0`` coercions.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Final, TypeAlias

MAX_REVIEW_CONTRACT_VALUE_DEPTH: Final = 8
MAX_REVIEW_CONTRACT_CONTAINER_ITEMS: Final = 64
MAX_REVIEW_CONTRACT_VALUE_NODES: Final = 2048

ReviewContractValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | list["ReviewContractValue"]
    | dict[str, "ReviewContractValue"]
)


class ReviewContractValueError(ValueError):
    """Raised when a value is outside the bounded canonical JSON subset."""


class ReviewContractValueBudget:
    """A cumulative node budget reusable across values from one test case."""

    __slots__ = ("_node_limit", "_used_nodes")

    def __init__(self, node_limit: int = MAX_REVIEW_CONTRACT_VALUE_NODES) -> None:
        if (
            type(node_limit) is not int
            or node_limit < 1
            or node_limit > MAX_REVIEW_CONTRACT_VALUE_NODES
        ):
            raise ValueError(
                "review-contract value node limit must be an integer from 1 "
                f"through {MAX_REVIEW_CONTRACT_VALUE_NODES}"
            )
        self._node_limit = node_limit
        self._used_nodes = 0

    @property
    def node_limit(self) -> int:
        """Return the fixed maximum number of nodes for this budget."""

        return self._node_limit

    @property
    def used_nodes(self) -> int:
        """Return the cumulative number of nodes consumed so far."""

        return self._used_nodes

    @property
    def remaining_nodes(self) -> int:
        """Return the number of nodes still available."""

        return self._node_limit - self._used_nodes

    def _consume(self, *, label: str) -> None:
        if self._used_nodes >= self._node_limit:
            raise ReviewContractValueError(
                f"{label} exceeds the {self._node_limit}-node review-contract "
                "value budget"
            )
        self._used_nodes += 1


def normalize_review_contract_value(
    value: object,
    *,
    label: str = "review-contract value",
    budget: ReviewContractValueBudget | None = None,
) -> ReviewContractValue:
    """Validate and recursively copy one bounded canonical JSON value.

    The root is depth zero, so a leaf reached through eight container edges is
    admitted.  Mapping keys are sorted in the returned copy.  Pass the same
    ``budget`` to multiple calls to enforce one cumulative per-case limit.
    """

    active_budget = budget if budget is not None else ReviewContractValueBudget()
    if not isinstance(active_budget, ReviewContractValueBudget):
        raise TypeError("budget must be a ReviewContractValueBudget")
    return _normalize_review_contract_value(
        value,
        label=label,
        budget=active_budget,
        depth=0,
        active_container_ids=set(),
    )


def review_contract_values_exactly_equal(left: object, right: object) -> bool:
    """Return exact equality for two valid bounded review-contract values.

    Invalid values compare unequal.  Dict insertion order is immaterial, list
    order is significant, scalar builtin types must match, and signed float zero
    remains distinct because it has a distinct canonical JSON representation.
    """

    try:
        normalized_left = normalize_review_contract_value(left, label="left value")
        normalized_right = normalize_review_contract_value(right, label="right value")
    except (ReviewContractValueError, TypeError):
        return False
    return _normalized_values_exactly_equal(normalized_left, normalized_right)


def _normalize_review_contract_value(
    value: object,
    *,
    label: str,
    budget: ReviewContractValueBudget,
    depth: int,
    active_container_ids: set[int],
) -> ReviewContractValue:
    if depth > MAX_REVIEW_CONTRACT_VALUE_DEPTH:
        raise ReviewContractValueError(
            f"{label} exceeds maximum review-contract value depth "
            f"{MAX_REVIEW_CONTRACT_VALUE_DEPTH}"
        )
    budget._consume(label=label)

    value_type = type(value)
    if value_type is bool:
        return value  # type: ignore[return-value]
    if value_type is int:
        return value  # type: ignore[return-value]
    if value_type is float:
        if not math.isfinite(value):  # type: ignore[arg-type]
            raise ReviewContractValueError(f"{label} must be a finite float")
        return value  # type: ignore[return-value]
    if value_type is str:
        if not _is_normalized_string_value(value):  # type: ignore[arg-type]
            raise ReviewContractValueError(
                f"{label} string must be NFC, trimmed, and free of disallowed "
                "control characters"
            )
        return value  # type: ignore[return-value]
    if value_type is list:
        if len(value) > MAX_REVIEW_CONTRACT_CONTAINER_ITEMS:  # type: ignore[arg-type]
            raise ReviewContractValueError(
                f"{label} list must contain at most "
                f"{MAX_REVIEW_CONTRACT_CONTAINER_ITEMS} items"
            )
        container_id = id(value)
        if container_id in active_container_ids:
            raise ReviewContractValueError(
                f"{label} contains a cyclic review-contract value"
            )
        active_container_ids.add(container_id)
        try:
            return [
                _normalize_review_contract_value(
                    item,
                    label=f"{label}[{index}]",
                    budget=budget,
                    depth=depth + 1,
                    active_container_ids=active_container_ids,
                )
                for index, item in enumerate(value)  # type: ignore[arg-type]
            ]
        finally:
            active_container_ids.remove(container_id)
    if value_type is dict:
        if len(value) > MAX_REVIEW_CONTRACT_CONTAINER_ITEMS:  # type: ignore[arg-type]
            raise ReviewContractValueError(
                f"{label} mapping must contain at most "
                f"{MAX_REVIEW_CONTRACT_CONTAINER_ITEMS} items"
            )
        for key in value:  # type: ignore[union-attr]
            if type(key) is not str or not _is_normalized_mapping_key(key):
                raise ReviewContractValueError(
                    f"{label} mapping keys must be nonempty, NFC, trimmed, and "
                    "free of control characters"
                )
        container_id = id(value)
        if container_id in active_container_ids:
            raise ReviewContractValueError(
                f"{label} contains a cyclic review-contract value"
            )
        active_container_ids.add(container_id)
        try:
            return {
                key: _normalize_review_contract_value(
                    value[key],  # type: ignore[index]
                    label=f"{label}[{key!r}]",
                    budget=budget,
                    depth=depth + 1,
                    active_container_ids=active_container_ids,
                )
                for key in sorted(value)  # type: ignore[arg-type]
            }
        finally:
            active_container_ids.remove(container_id)
    raise ReviewContractValueError(
        f"{label} must contain only exact builtin bool, int, finite float, "
        "string, list, or mapping values; null is not allowed"
    )


def _is_normalized_mapping_key(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value == unicodedata.normalize("NFC", value)
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def _is_normalized_string_value(value: str) -> bool:
    return (
        value == value.strip()
        and value == unicodedata.normalize("NFC", value)
        and "\r" not in value
        and not any(
            (ord(character) < 32 and character not in {"\n", "\t"})
            or ord(character) == 127
            for character in value
        )
    )


def _normalized_values_exactly_equal(
    left: ReviewContractValue,
    right: ReviewContractValue,
) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return left.hex() == right.hex()  # type: ignore[union-attr]
    if type(left) in {bool, int, str}:
        return left == right
    if type(left) is list:
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _normalized_values_exactly_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return tuple(left) == tuple(right) and all(  # type: ignore[arg-type]
        _normalized_values_exactly_equal(left[key], right[key])  # type: ignore[index]
        for key in left  # type: ignore[union-attr]
    )
