"""Contracts for work measurements, including their unavailable-counter fallback."""

from __future__ import annotations

import ctypes
import os
from types import SimpleNamespace

import pytest

from tests import scaling


def test_lowest_work_excludes_setup_and_checks_every_result(monkeypatch):
    events = []
    readings = iter((100, 130, 200, 210, 300, 320))

    def read():
        events.append("read")
        return next(readings)

    def call():
        events.append("call")
        return 42

    def check(value):
        assert value == 42
        events.append("check")

    monkeypatch.setattr(scaling, "_CLOCK", scaling._WorkClock(read, 1, "test work"))
    assert scaling.lowest_work(call, check, setup=lambda: events.append("setup")) == 10
    # Setup precedes the first read, and checks follow the second, every sample.
    assert events == ["setup", "read", "call", "read", "check"] * 3


def test_lowest_work_checks_the_slower_results_too(monkeypatch):
    readings = iter((0, 1, 1, 100))
    results = iter((True, False))
    monkeypatch.setattr(
        scaling, "_CLOCK", scaling._WorkClock(lambda: next(readings), 1, "test work")
    )

    def check(value):
        assert value

    with pytest.raises(AssertionError):
        scaling.lowest_work(lambda: next(results), check, runs=2)


@pytest.mark.parametrize("runs", (0, -1))
def test_lowest_work_requires_a_sample(runs):
    with pytest.raises(ValueError, match="positive"):
        scaling.lowest_work(lambda: None, lambda _: None, runs=runs)


@pytest.mark.parametrize("delta", (0, -1))
def test_lowest_work_rejects_a_stalled_or_reversed_counter(monkeypatch, delta):
    readings = iter((10, 10 + delta))
    monkeypatch.setattr(
        scaling, "_CLOCK", scaling._WorkClock(lambda: next(readings), 1, "test work")
    )
    with pytest.raises(AssertionError):
        scaling.lowest_work(lambda: None, lambda _: None, runs=1)


@pytest.mark.parametrize("floor", (0.0001, 100_000))
def test_growth_has_threefold_headroom_and_a_unit_specific_floor(monkeypatch, floor):
    monkeypatch.setattr(scaling, "_CLOCK", scaling._WorkClock(lambda: 0, floor, "test"))
    # 16x input permits <48x work. The floor only applies to the small sample.
    scaling.assert_linear_growth(floor * 2, floor * 95, input_ratio=16)
    with pytest.raises(AssertionError):
        scaling.assert_linear_growth(floor * 2, floor * 96, input_ratio=16)
    scaling.assert_linear_growth(floor / 100, floor * 47, input_ratio=16)
    with pytest.raises(AssertionError):
        scaling.assert_linear_growth(floor / 100, floor * 48, input_ratio=16)


@pytest.mark.parametrize("ratio", (0, 1, -1))
def test_growth_requires_increasing_inputs(ratio):
    with pytest.raises(ValueError, match="exceed one"):
        scaling.assert_linear_growth(1, 1, input_ratio=ratio)


def test_small_input_ratios_can_guard_exponential_regressions():
    unit = scaling._CLOCK.floor
    scaling.assert_linear_growth(unit, 5 * unit, input_ratio=2)
    with pytest.raises(AssertionError):
        scaling.assert_linear_growth(unit, 256 * unit, input_ratio=2)


@pytest.mark.parametrize("small,large", ((0, 1), (1, 0), (-1, 1)))
def test_growth_rejects_nonpositive_samples(small, large):
    with pytest.raises(AssertionError):
        scaling.assert_linear_growth(small, large, input_ratio=16)


def test_macos_selects_an_advancing_instruction_counter(monkeypatch):
    readings = iter((100, 200))

    def read():
        return next(readings)

    monkeypatch.setattr(scaling, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(scaling, "_instruction_reader", lambda: read)
    clock = scaling._select_work_clock()
    assert clock.read is read
    assert clock.unit == "instructions"
    assert clock.floor == 100_000


@pytest.mark.parametrize("platform", ("linux", "win32"))
def test_other_platforms_use_cpu_without_loading_libproc(monkeypatch, platform):
    monkeypatch.setattr(scaling, "sys", SimpleNamespace(platform=platform))

    def unavailable():
        pytest.fail("libproc must only be loaded on macOS")

    monkeypatch.setattr(scaling, "_instruction_reader", unavailable)
    clock = scaling._select_work_clock()
    assert clock.read is scaling.time.process_time
    assert clock.unit == "CPU seconds"
    assert clock.floor == 0.0001


@pytest.mark.parametrize("failure", (OSError, AttributeError))
def test_macos_falls_back_when_libproc_is_unavailable(monkeypatch, failure):
    monkeypatch.setattr(scaling, "sys", SimpleNamespace(platform="darwin"))

    def unavailable():
        raise failure("unavailable")

    monkeypatch.setattr(scaling, "_instruction_reader", unavailable)
    assert scaling._select_work_clock().unit == "CPU seconds"


@pytest.mark.parametrize("values", ((0, 0), (100, 100), (100, 99)))
def test_macos_falls_back_when_instructions_do_not_advance(monkeypatch, values):
    readings = iter(values)
    monkeypatch.setattr(scaling, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(scaling, "_instruction_reader", lambda: lambda: next(readings))
    assert scaling._select_work_clock().unit == "CPU seconds"


def test_instruction_reader_uses_the_full_v6_buffer(monkeypatch):
    def usage(pid, flavor, pointer):
        assert pid == os.getpid()
        assert flavor == 6
        info = ctypes.cast(pointer, ctypes.POINTER(scaling._RusageInfoV6)).contents
        info.ri_instructions = 123456
        info.ri_reserved[8] = 7
        return 0

    monkeypatch.setattr(
        scaling.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(proc_pid_rusage=usage)
    )
    assert scaling._instruction_reader()() == 123456
    # These offsets/size follow <sys/resource.h>, not the host's Python types.
    assert scaling._RusageInfoV6.ri_instructions.offset == 248
    assert ctypes.sizeof(scaling._RusageInfoV6) == 464


def test_instruction_reader_reports_a_kernel_refusal(monkeypatch):
    def usage(*args):
        return -1

    monkeypatch.setattr(
        scaling.ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(proc_pid_rusage=usage)
    )
    with pytest.raises(OSError, match="proc_pid_rusage"):
        scaling._instruction_reader()()


def test_a_selected_counter_failure_does_not_mix_units(monkeypatch):
    def refused():
        raise OSError("counter stopped working")

    monkeypatch.setattr(
        scaling, "_CLOCK", scaling._WorkClock(refused, 100_000, "instructions")
    )
    with pytest.raises(OSError, match="stopped working"):
        scaling.lowest_work(lambda: None, lambda _: None)


@pytest.mark.parametrize("use_cpu", (False, True), ids=("selected", "cpu-fallback"))
def test_the_selected_measure_moves_with_work(monkeypatch, use_cpu):
    if use_cpu:
        # Exercise the real fallback reader on macOS too, rather than only
        # checking its identity when platform selection is mocked.
        monkeypatch.setattr(
            scaling,
            "_CLOCK",
            scaling._WorkClock(scaling.time.process_time, 0.0001, "CPU seconds"),
        )

    def work(count):
        return sum(index * index for index in range(count))

    def measure(count):
        def check(value):
            assert value == (count - 1) * count * (2 * count - 1) // 6

        return scaling.lowest_work(lambda: work(count), check)

    # Both sizes exceed the clock overhead. This tests the real selected reader
    # under the sandbox; a constant or fabricated counter cannot pass it.
    small, large = measure(2_000), measure(32_000)
    assert large > small * 4, (scaling._CLOCK.unit, small, large)
    scaling.assert_linear_growth(small, large, input_ratio=16)
