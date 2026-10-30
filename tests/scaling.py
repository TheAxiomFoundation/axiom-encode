"""Complexity guards that don't depend on how fast or busy a runner is.

A limit in seconds fails on a slow runner, and wall-clock time also counts every
moment the scheduler keeps the process waiting, so even a ratio of two wall-clock
times can fail correct code or pass a quadratic scan on a busy host. These helpers
compare CPU time, which leaves the waiting out, at two input sizes, and keep each
size's fastest run. Where a call's work can be counted exactly, count it as well.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Four times the input may cost at most eight times as much. Linear work costs
# about four times as much and quadratic work sixteen.
MAX_GROWTH = 8
# A shorter CPU time is too small to compare.
FLOOR_SECONDS = 0.02


def fastest_cpu_seconds(
    call: Callable[[], T], check: Callable[[T], object], runs: int = 3
) -> float:
    """The lowest CPU time of ``runs`` calls; ``check`` tests each result, untimed.

    A call that remembers its input has to forget it first, or the later runs
    time a cache hit.
    """
    fastest = float("inf")
    for _ in range(runs):
        started = time.process_time()
        result = call()
        fastest = min(fastest, time.process_time() - started)
        check(result)
    return fastest


def assert_grows_linearly(small: float, large: float) -> None:
    """``large``, measured on four times ``small``'s input, stayed within the growth bound."""
    assert large < MAX_GROWTH * max(small, FLOOR_SECONDS), (small, large)
