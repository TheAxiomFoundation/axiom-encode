"""Fact values and the household grid.

A fact with explicit ``values`` keeps them. Any other fact gets its values
from the modules: zero, and each threshold that the modules compare the fact
with, with the value just below and just above it. A bool fact gets both
values.

The grid is the product of the facts' values, in the order the link lists the
facts, with the last fact varying fastest. When the grid holds more
combinations than the link's ``sampling.max_combinations``, the step runs a
sample: the first and the last combination, then combination
``sha256("<seed>:<k>") mod size`` for ``k = 0, 1, 2, ...`` until the cap is
reached. The sample depends only on the seed, the cap and the grid size.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from .link import Fact, canonical_decimal

SAMPLING_METHOD = "sha256-counter/v1"


def generate_values(
    fact: Fact, thresholds: Iterable[Decimal]
) -> list[bool | int | str]:
    """The values of one fact that has no explicit values."""

    if fact.kind == "bool":
        return [False, True]
    candidates = {Decimal(0)}
    for threshold in thresholds:
        if fact.kind == "integer" and threshold != threshold.to_integral_value():
            candidates.add(threshold.to_integral_value(rounding=ROUND_FLOOR))
            candidates.add(threshold.to_integral_value(rounding=ROUND_CEILING))
            continue
        candidates.update((threshold - fact.step, threshold, threshold + fact.step))
    kept = sorted(
        value
        for value in candidates
        if (fact.minimum is None or value >= fact.minimum)
        and (fact.maximum is None or value <= fact.maximum)
    )
    if fact.kind == "integer":
        return [int(value) for value in kept]
    return list(dict.fromkeys(canonical_decimal(value) for value in kept))


def grid_size(value_lists: Sequence[Sequence[object]]) -> int:
    size = 1
    for values in value_lists:
        size *= len(values)
    return size


def combination(index: int, value_lists: Sequence[Sequence[object]]) -> tuple:
    """Combination ``index`` of the grid, in ``itertools.product`` order."""

    picked = []
    for values in reversed(value_lists):
        index, position = divmod(index, len(values))
        picked.append(values[position])
    return tuple(reversed(picked))


def select_indices(size: int, cap: int, seed: int) -> tuple[list[int], bool]:
    """The grid indices to run, in grid order, and whether they are a sample."""

    if size <= cap:
        return list(range(size)), False
    chosen = {0, size - 1} if cap >= 2 else {0}
    counter = 0
    while len(chosen) < cap:
        digest = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        chosen.add(int.from_bytes(digest, "big") % size)
        counter += 1
    return sorted(chosen), True
