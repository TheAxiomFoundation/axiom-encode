"""Per-try wall time and phase timelines for the encode loop."""

from __future__ import annotations

import io
import threading
from datetime import datetime, timedelta, timezone

import pytest

from axiom_encode import encode_timing
from axiom_encode.encode_timing import (
    ENCODE_LOOP_TIMING_SCHEMA,
    PHASE_CANDIDATE_VALIDATION,
    PHASE_MODEL_CALL,
    PHASE_OTHER,
    PHASE_PREPARE,
    PHASE_RECORD_RESULT,
    PHASE_REVIEW_MODEL_CALL,
    PHASE_STAGE_CANDIDATE,
    TOOL_CI_STATIC_CHECKS,
    TOOL_CI_TEST_CASES,
    TOOL_POLICYENGINE_ORACLE,
    TOOL_RULES_ENGINE_COMPILE,
    TOOL_SOURCE_COMPLETENESS_CHECKS,
    EncodeLoopTimer,
    activate_encode_loop_timer,
    active_encode_loop_timer,
    encode_phase,
    encode_tool,
    iteration_timing_payload,
    line_buffer_stdout,
    mark_encode_phase,
    timed_phase,
    timed_tool,
)

ORIGIN_WALL = datetime(2026, 10, 7, 16, 0, 0, tzinfo=timezone.utc)


class FakeClock:
    """A monotonic clock that only moves when the test says so."""

    def __init__(self, start: float = 500.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def _timer(clock: FakeClock, lines: list[str] | None = None) -> EncodeLoopTimer:
    return EncodeLoopTimer(
        monotonic=clock,
        origin=(clock.now, ORIGIN_WALL),
        emit=(lines.append if lines is not None else None),
    )


def _iso(offset_ms: int) -> str:
    return encode_timing.format_utc(ORIGIN_WALL + timedelta(milliseconds=offset_ms))


def _assert_contiguous(timing) -> None:
    """Phases tile the try exactly: no gap, no overlap, exact sum."""
    assert timing.phases
    assert timing.phases[0].started_at == timing.started_at
    assert timing.phases[-1].finished_at == timing.finished_at
    for earlier, later in zip(timing.phases, timing.phases[1:]):
        assert earlier.finished_at == later.started_at
        assert earlier.name != later.name or earlier.breakdown_ms != later.breakdown_ms
    assert sum(phase.duration_ms for phase in timing.phases) == (
        timing.wall_duration_ms
    )
    for phase in timing.phases:
        assert phase.duration_ms > 0
        if phase.breakdown_ms is not None:
            assert sum(phase.breakdown_ms.values()) == phase.duration_ms


def test_try_timeline_sums_exactly_to_wall_time_with_a_fake_clock():
    clock = FakeClock()
    timer = _timer(clock)
    with activate_encode_loop_timer(timer):
        clock.tick(2.0)  # loop setup before the first try
        timer.start_try(1)
        mark_encode_phase(PHASE_PREPARE)
        clock.tick(1.5)
        mark_encode_phase(PHASE_MODEL_CALL)
        clock.tick(40.0)
        mark_encode_phase(PHASE_STAGE_CANDIDATE)
        clock.tick(0.25)
        mark_encode_phase(PHASE_CANDIDATE_VALIDATION)
        clock.tick(0.1)
        with encode_tool(TOOL_RULES_ENGINE_COMPILE):
            clock.tick(3.0)
        with encode_tool(TOOL_CI_STATIC_CHECKS):
            clock.tick(1.0)
            with encode_tool(TOOL_CI_TEST_CASES):
                clock.tick(10.0)
            clock.tick(0.5)
        with encode_phase(PHASE_REVIEW_MODEL_CALL):
            clock.tick(20.0)
        clock.tick(0.4)
        mark_encode_phase(PHASE_RECORD_RESULT)
        clock.tick(0.3)
        timing = timer.finish_try()

    assert timing is not None
    assert timing.attempt == 1
    assert timing.started_at == _iso(2000)
    assert timing.finished_at == _iso(79_050)
    assert timing.wall_duration_ms == 77_050
    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_PREPARE, 1_500),
        (PHASE_MODEL_CALL, 40_000),
        (PHASE_STAGE_CANDIDATE, 250),
        (PHASE_CANDIDATE_VALIDATION, 14_600),
        (PHASE_REVIEW_MODEL_CALL, 20_000),
        (PHASE_CANDIDATE_VALIDATION, 400),
        (PHASE_RECORD_RESULT, 300),
    ]
    # Tools never reach the timeline; the innermost tool owns each instant
    # and the unclaimed rest of the phase is "other".
    assert timing.phases[3].breakdown_ms == {
        TOOL_CI_STATIC_CHECKS: 1_500,
        TOOL_CI_TEST_CASES: 10_000,
        PHASE_OTHER: 100,
        TOOL_RULES_ENGINE_COMPILE: 3_000,
    }
    assert timing.phases[5].breakdown_ms is None
    _assert_contiguous(timing)


def test_untimed_time_is_an_explicit_other_phase():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    clock.tick(0.7)  # nothing instrumented yet
    timer.mark(PHASE_PREPARE)
    clock.tick(1.0)
    timer.mark(None)  # back to uninstrumented work
    clock.tick(0.2)
    timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_OTHER, 700),
        (PHASE_PREPARE, 1_000),
        (PHASE_OTHER, 200),
    ]
    _assert_contiguous(timing)


def test_nested_phase_splits_and_resumes_the_enclosing_phase():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(3)
    timer.mark("apply_repair")
    clock.tick(1.0)
    with timer.phase("overlay_validation"):
        clock.tick(5.0)
        with timer.tool(TOOL_RULES_ENGINE_COMPILE):
            clock.tick(2.0)
    clock.tick(1.0)
    with timer.phase("overlay_validation"):
        clock.tick(4.0)
    timer.mark("apply_write")
    clock.tick(0.5)
    timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        ("apply_repair", 1_000),
        ("overlay_validation", 7_000),
        ("apply_repair", 1_000),
        ("overlay_validation", 4_000),
        ("apply_write", 500),
    ]
    assert timing.phases[1].breakdown_ms == {
        PHASE_OTHER: 5_000,
        TOOL_RULES_ENGINE_COMPILE: 2_000,
    }
    _assert_contiguous(timing)


def test_zero_length_phases_are_dropped_and_neighbours_merge():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    timer.mark(PHASE_PREPARE)
    clock.tick(1.0)
    with timer.phase(PHASE_STAGE_CANDIDATE):
        pass  # same instant: never becomes a phase
    with timer.phase(PHASE_PREPARE):  # nested same name: one phase
        clock.tick(1.0)
    clock.tick(1.0)
    timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_PREPARE, 3_000),
    ]
    _assert_contiguous(timing)


def test_nested_same_name_phases_unwind_by_identity():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    with timer.phase("a"):
        clock.tick(1.0)
        with timer.phase("b"):
            clock.tick(1.0)
            with timer.phase("a"):
                clock.tick(1.0)
            clock.tick(1.0)  # still inside "b"
        clock.tick(1.0)  # back inside the outer "a"
    timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        ("a", 1_000),
        ("b", 1_000),
        ("a", 1_000),
        ("b", 1_000),
        ("a", 1_000),
    ]


def test_tools_entered_outside_a_nested_phase_do_not_leak_into_it():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    timer.mark(PHASE_CANDIDATE_VALIDATION)
    with timer.tool(TOOL_CI_STATIC_CHECKS):
        clock.tick(1.0)
        with timer.phase(PHASE_REVIEW_MODEL_CALL):
            clock.tick(2.0)
            with timer.tool(TOOL_POLICYENGINE_ORACLE):
                clock.tick(3.0)
        clock.tick(1.0)
    timing = timer.finish_try()

    validation, review, validation_again = timing.phases
    assert validation.breakdown_ms == {TOOL_CI_STATIC_CHECKS: 1_000}
    assert review.breakdown_ms == {PHASE_OTHER: 2_000, TOOL_POLICYENGINE_ORACLE: 3_000}
    assert validation_again.breakdown_ms == {TOOL_CI_STATIC_CHECKS: 1_000}
    _assert_contiguous(timing)


def test_tries_are_contiguous_and_loop_totals_add_up():
    clock = FakeClock()
    timer = _timer(clock)
    clock.tick(4.0)
    timer.start_try(1)
    timer.mark(PHASE_MODEL_CALL)
    clock.tick(10.0)
    timer.start_try(2)  # ends try 1 at the same instant
    clock.tick(20.0)
    timer.finish_try()
    clock.tick(0.75)

    first, second = timer.tries
    assert first.finished_at == second.started_at
    assert (first.wall_duration_ms, second.wall_duration_ms) == (10_000, 20_000)
    # Untimed time in try 2 is still accounted for.
    assert [(phase.name, phase.duration_ms) for phase in second.phases] == [
        (PHASE_OTHER, 20_000)
    ]
    assert timer.loop_timing() == {
        "schema": ENCODE_LOOP_TIMING_SCHEMA,
        "origin": "explicit",
        "started_at": _iso(0),
        "finished_at": _iso(34_750),
        "wall_duration_ms": 34_750,
        "setup_ms": 4_000,
        "tries_ms": 30_000,
        "between_tries_ms": 0,
        "finalize_ms": 750,
        "try_count": 2,
    }


def test_loop_totals_count_gaps_between_tries():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    clock.tick(1.0)
    timer.finish_try()
    clock.tick(2.0)
    timer.start_try(2)
    clock.tick(3.0)
    timer.finish_try()

    totals = timer.loop_timing()
    assert totals["between_tries_ms"] == 2_000
    assert (
        totals["setup_ms"]
        + totals["tries_ms"]
        + totals["between_tries_ms"]
        + totals["finalize_ms"]
    ) == totals["wall_duration_ms"]


def test_progress_lines_mark_each_phase_change_with_the_try_number():
    clock = FakeClock()
    lines: list[str] = []
    timer = _timer(clock, lines)
    clock.tick(1.0)
    timer.start_try(2)
    timer.mark(PHASE_PREPARE)
    clock.tick(1.5)
    with timer.phase(PHASE_MODEL_CALL):
        clock.tick(40.0)
    timer.mark(PHASE_STAGE_CANDIDATE)
    with timer.tool(TOOL_RULES_ENGINE_COMPILE):  # tools print nothing
        clock.tick(0.2)
    timer.finish_try()

    assert lines == [
        f"  try=2 start at={_iso(1_000)}",
        f"  try=2 phase=prepare at={_iso(1_000)}",
        f"  try=2 phase=model_call at={_iso(2_500)} prev=prepare:1500ms",
        # The scoped call ended back in "prepare" for zero time.
        f"  try=2 phase=prepare at={_iso(42_500)} prev=model_call:40000ms",
        f"  try=2 phase=stage_candidate at={_iso(42_500)}",
        f"  try=2 end at={_iso(42_700)} wall_ms=41700 prev=stage_candidate:200ms",
    ]


def test_module_helpers_are_noops_without_an_active_timer():
    assert active_encode_loop_timer() is None
    mark_encode_phase(PHASE_PREPARE)
    with encode_phase(PHASE_MODEL_CALL), encode_tool(TOOL_RULES_ENGINE_COMPILE):
        pass


def test_calls_from_other_threads_are_ignored():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    with activate_encode_loop_timer(timer):
        mark_encode_phase(PHASE_PREPARE)

        def worker() -> None:
            # A worker that inherits the context must not touch the timeline.
            assert active_encode_loop_timer() is None
            with encode_phase(PHASE_REVIEW_MODEL_CALL):
                clock.tick(5.0)

        import contextvars

        context = contextvars.copy_context()
        thread = threading.Thread(target=lambda: context.run(worker))
        thread.start()
        thread.join()
        timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_PREPARE, 5_000)
    ]


def test_decorators_time_each_call():
    clock = FakeClock()
    timer = _timer(clock)

    @timed_phase(PHASE_REVIEW_MODEL_CALL)
    def review(seconds: float) -> str:
        clock.tick(seconds)
        return "reviewed"

    @timed_tool(TOOL_CI_TEST_CASES)
    def run_tests(seconds: float) -> str:
        clock.tick(seconds)
        return "tested"

    assert review.__name__ == "review"
    with activate_encode_loop_timer(timer):
        timer.start_try(1)
        timer.mark(PHASE_CANDIDATE_VALIDATION)
        assert run_tests(2.0) == "tested"
        assert review(3.0) == "reviewed"
        timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_CANDIDATE_VALIDATION, 2_000),
        (PHASE_REVIEW_MODEL_CALL, 3_000),
    ]
    assert timing.phases[0].breakdown_ms == {TOOL_CI_TEST_CASES: 2_000}


def test_a_phase_that_raises_still_closes():
    clock = FakeClock()
    timer = _timer(clock)
    timer.start_try(1)
    with pytest.raises(RuntimeError):
        with timer.phase(PHASE_MODEL_CALL):
            clock.tick(1.0)
            raise RuntimeError("model failed")
    clock.tick(1.0)
    timing = timer.finish_try()

    assert [(phase.name, phase.duration_ms) for phase in timing.phases] == [
        (PHASE_MODEL_CALL, 1_000),
        (PHASE_OTHER, 1_000),
    ]


def test_first_real_clock_timer_claims_the_import_origin(monkeypatch):
    origin = (encode_timing.time.monotonic() - 3.0, datetime.now(timezone.utc))
    monkeypatch.setattr(encode_timing, "_PROCESS_ORIGIN", origin)

    first = EncodeLoopTimer(emit=None)
    second = EncodeLoopTimer(emit=None)

    assert first.origin_kind == "process"
    assert first.loop_timing()["setup_ms"] >= 3_000
    assert second.origin_kind == "encode_loop"
    # A fake clock never borrows the real-clock origin.
    monkeypatch.setattr(encode_timing, "_PROCESS_ORIGIN", origin)
    assert EncodeLoopTimer(monotonic=FakeClock(), emit=None).origin_kind == (
        "encode_loop"
    )


def test_iteration_timing_payload_keeps_only_well_formed_fields():
    class Untimed:
        started_at = None
        finished_at = None
        wall_duration_ms = None
        phases = None

    assert iteration_timing_payload(Untimed()) == {}
    stored = {
        "attempt": 1,
        "started_at": "2026-10-07T16:00:00.000Z",
        "finished_at": "2026-10-07T16:00:01.000Z",
        "wall_duration_ms": 1000,
        "phases": [
            {
                "name": "model_call",
                "started_at": "2026-10-07T16:00:00.000Z",
                "finished_at": "2026-10-07T16:00:01.000Z",
                "duration_ms": 1000,
                "breakdown_ms": {"rules_engine_compile": 10, "bad": "x"},
            },
            {"name": "", "duration_ms": 5},
            {"name": "other", "duration_ms": True},
            "not a phase",
        ],
    }
    assert iteration_timing_payload(stored) == {
        "started_at": "2026-10-07T16:00:00.000Z",
        "finished_at": "2026-10-07T16:00:01.000Z",
        "wall_duration_ms": 1000,
        "phases": [
            {
                "name": "model_call",
                "started_at": "2026-10-07T16:00:00.000Z",
                "finished_at": "2026-10-07T16:00:01.000Z",
                "duration_ms": 1000,
                "breakdown_ms": {"rules_engine_compile": 10},
            }
        ],
    }


def test_line_buffer_stdout_flushes_each_line(monkeypatch):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    monkeypatch.setattr(encode_timing.sys, "stdout", stream)

    line_buffer_stdout()
    stream.write("try=1 phase=model_call\n")

    assert stream.line_buffering is True
    assert raw.getvalue() == b"try=1 phase=model_call\n"


def test_line_buffer_stdout_tolerates_streams_without_reconfigure(monkeypatch):
    monkeypatch.setattr(encode_timing.sys, "stdout", object())
    line_buffer_stdout()
    monkeypatch.setattr(encode_timing.sys, "stdout", None)
    line_buffer_stdout()


@pytest.mark.parametrize(
    ("owner", "attribute", "expected"),
    [
        (
            "validator_pipeline.ValidatorPipeline",
            "_compile_rulespec_to_artifact",
            ("tool", TOOL_RULES_ENGINE_COMPILE),
        ),
        (
            "validator_pipeline.ValidatorPipeline",
            "_run_ci",
            ("tool", TOOL_CI_STATIC_CHECKS),
        ),
        (
            "validator_pipeline.ValidatorPipeline",
            "_run_rulespec_test_cases",
            ("tool", TOOL_CI_TEST_CASES),
        ),
        (
            "validator_pipeline.ValidatorPipeline",
            "_complete_source_unit_issues",
            ("tool", TOOL_SOURCE_COMPLETENESS_CHECKS),
        ),
        (
            "validator_pipeline.ValidatorPipeline",
            "_run_policyengine",
            ("tool", TOOL_POLICYENGINE_ORACLE),
        ),
        (
            "validator_pipeline.ValidatorPipeline",
            "_run_reviewer",
            ("phase", PHASE_REVIEW_MODEL_CALL),
        ),
        ("evals", "_apply_generated_eval_repairs", ("phase", "artifact_repair")),
        (
            "evals",
            "_overlay_validation_retry_candidate",
            ("phase", "repair_overlay"),
        ),
    ],
)
def test_validator_and_eval_steps_report_stable_names(owner, attribute, expected):
    import importlib

    module_name, _, class_name = owner.partition(".")
    target = importlib.import_module(f"axiom_encode.harness.{module_name}")
    if class_name:
        target = getattr(target, class_name)
    assert getattr(target, attribute).encode_timing == expected


class TickingClock:
    """A monotonic clock that advances one second on every read."""

    def __init__(self, start: float = 900.0) -> None:
        self.now = start

    def __call__(self) -> float:
        self.now += 1.0
        return self.now


def test_a_decorated_validator_check_lands_in_its_tool(tmp_path):
    """A real decorated validator method attributes its time on the timeline."""
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    clock = TickingClock()
    timer = EncodeLoopTimer(monotonic=clock, origin=(clock.now, ORIGIN_WALL), emit=None)
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        enable_oracles=False,
    )

    with activate_encode_loop_timer(timer):
        timer.start_try(1)
        mark_encode_phase(PHASE_CANDIDATE_VALIDATION)
        issues = pipeline._complete_source_unit_issues(
            "format: rulespec/v1\n", validation_source_texts={}, test_cases=[]
        )
        timing = timer.finish_try()

    assert issues == []
    other, validation = timing.phases
    assert (other.name, other.duration_ms) == (PHASE_OTHER, 1_000)
    assert validation.name == PHASE_CANDIDATE_VALIDATION
    # Clock reads: mark, tool enter, tool exit, finish -> 1 s each.
    assert validation.breakdown_ms == {
        PHASE_OTHER: 2_000,
        TOOL_SOURCE_COMPLETENESS_CHECKS: 1_000,
    }
