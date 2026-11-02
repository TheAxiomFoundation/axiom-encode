"""Compare work at different input sizes without counting scheduler waits.

On macOS, use this process's retired instructions when the kernel exposes them;
elsewhere (or if that counter is unavailable), use process CPU seconds. Neither
counts waiting for a core. Instructions also reduce sensitivity to CPU frequency,
but coverage, caches and changes in the executed code can still move a ratio;
CPU time additionally moves with contention and frequency. Prefer exact counts
when they account for the expensive work, including scans inside C functions.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class _RusageInfoV6(ctypes.Structure):
    # Full struct rusage_info_v6 from macOS <sys/resource.h>. The kernel writes
    # the entire struct, even though we only read ri_instructions (children are
    # excluded). Keep trailing fields and reserved space in the allocation.
    _fields_ = (
        [("ri_uuid", ctypes.c_uint8 * 16)]
        + [
            ("ri_" + name, ctypes.c_uint64)
            for name in (
                "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
                "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
                "child_user_time child_system_time child_pkg_idle_wkups "
                "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread "
                "diskio_byteswritten cpu_time_qos_default cpu_time_qos_maintenance "
                "cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy "
                "cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
                "serviced_system_time logical_writes lifetime_max_phys_footprint "
                "instructions cycles billed_energy serviced_energy interval_max_phys_footprint "
                "runnable_time flags user_ptime system_ptime pinstructions pcycles energy_nj "
                "penergy_nj secure_time_in_system secure_ptime_in_system neural_footprint "
                "lifetime_max_neural_footprint interval_max_neural_footprint"
            ).split()
        ]
        + [("ri_reserved", ctypes.c_uint64 * 9)]
    )


def _instruction_reader() -> Callable[[], float]:
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    read_usage = library.proc_pid_rusage
    read_usage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    read_usage.restype = ctypes.c_int

    def read() -> float:
        info = _RusageInfoV6()
        if read_usage(os.getpid(), 6, ctypes.byref(info)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage")
        return float(info.ri_instructions)

    return read


@dataclass(frozen=True)
class _WorkClock:
    read: Callable[[], float]
    floor: float
    unit: str


def _select_work_clock() -> _WorkClock:
    if sys.platform == "darwin":
        try:
            read = _instruction_reader()
            before = read()
            # A supported counter must advance, not just return success/zero.
            sum(range(1_000))
            if read() > before > 0:
                return _WorkClock(read, 100_000, "instructions")
        except (OSError, AttributeError):
            pass
    # Keep the floor below measurable short batches: a 2ms denominator
    # diluted the rejection margin for their quadratic regressions.
    return _WorkClock(time.process_time, 0.0001, "CPU seconds")


_CLOCK = _select_work_clock()


class WorkSample(float):
    """The lowest work of one operation, which can measure that operation again."""

    def __new__(cls, work: float, measure: Callable[[], WorkSample]) -> WorkSample:
        sample = super().__new__(cls, work)
        sample.measure = measure
        return sample


def lowest_work(
    call: Callable[[], T],
    check: Callable[[T], object],
    *,
    setup: Callable[[], object] | None = None,
    runs: int = 3,
) -> WorkSample:
    """Lowest work of several calls, with setup and every output check untimed.

    Clear caches in ``setup`` on every sample, so later samples do real work.
    A counter failing after selection raises rather than mixing units mid-run.
    The returned sample's ``measure`` repeats all of this.
    """
    if runs < 1:
        raise ValueError("runs must be positive")

    def measure() -> WorkSample:
        lowest = float("inf")
        for _ in range(runs):
            if setup is not None:
                setup()
            started = _CLOCK.read()
            result = call()
            work = _CLOCK.read() - started
            assert work > 0, (_CLOCK.unit, work)
            lowest = min(lowest, work)
            check(result)
        return WorkSample(lowest, measure)

    return measure()


def _within_headroom(small: float, large: float, input_ratio: float) -> bool:
    return large < 3 * input_ratio * max(small, _CLOCK.floor)


def assert_linear_growth(small: float, large: float, *, input_ratio: float) -> None:
    """Allow 3x headroom over linear growth, with a floor in the active unit.

    Use ratios above three for quadratic guards to leave room below r squared.
    Smaller ratios can expose exponential regressions with finite fixtures.
    Size fixtures so a known regression reaches that bound despite the floor.

    CPU seconds for one fixed loop moved 3.6x on a loaded Mac at utility
    priority, past that headroom. So a pair from ``lowest_work`` over the bound
    measures again and fails only if the fresh pair exceeds it too. The fresh
    pair replaces both samples, because the old small one may be the lucky one.
    Plain numbers can't measure again, so they fail at once.
    """
    if input_ratio <= 1:
        raise ValueError("input ratio must exceed one")
    assert small > 0 and large > 0, (small, large)
    first = small, large
    if (
        not _within_headroom(small, large, input_ratio)
        and isinstance(small, WorkSample)
        and isinstance(large, WorkSample)
    ):
        small, large = small.measure(), large.measure()
    assert _within_headroom(small, large, input_ratio), (
        _CLOCK.unit,
        first,
        small,
        large,
        input_ratio,
    )
