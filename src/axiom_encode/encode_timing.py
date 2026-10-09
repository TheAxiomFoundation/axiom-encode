"""Wall-clock timing for each try of the ``encode`` validator-retry loop.

``encode`` runs one *try* per generation attempt
(``cli._run_encode_attempts_with_retries``): prepare the workspace and prompt,
call the model, then judge the candidate (standalone compile/CI validation,
review, deterministic repairs and, under ``--apply``, apply-time repairs, the
policy-overlay validation and the signed apply).  ``Iteration.duration_ms``
records only the model call.  This module records where the rest of each try's
wall time goes, so a 51-minute encode step is no longer 2 minutes of model
time and 49 minutes nobody can attribute.

Model
-----
* A try starts when the loop starts it (:meth:`EncodeLoopTimer.start_try`) and
  ends when the next try starts or when the loop has finished judging the last
  candidate (:meth:`EncodeLoopTimer.finish_try`).  Tries are contiguous.
* Inside a try every millisecond belongs to exactly one *phase*: the innermost
  scoped phase (``with encode_phase(name)``) if one is active, else the try's
  *base* phase (set by :func:`mark_encode_phase`), else ``other``.  The phases
  of a try form an ordered, contiguous timeline whose ``duration_ms`` values
  sum to the try's wall time exactly: every boundary is an integer number of
  milliseconds on one monotonic clock.
* *Tools* (``with encode_tool(name)``) are nested measurements that never
  appear on the timeline.  Their time is added to the enclosing phase
  segment's ``breakdown_ms`` (the innermost tool wins) and the rest of that
  segment is recorded under ``other``, so a breakdown also sums to its
  segment's duration.  A phase that compiles and tests dozens of dependent
  files therefore stays one timeline entry instead of hundreds.
* Recording is bound to the thread that created the timer.  Calls from worker
  threads, and calls while no timer is active, do nothing.
* Each change of phase prints one short progress line (try number, phase,
  UTC time and the previous phase's duration) and flushes it, so the job log
  timestamps mark real phase boundaries.

Phase names are stable identifiers; see ``PHASE_*`` and ``TOOL_*`` below.
"""

from __future__ import annotations

import functools
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

ENCODE_LOOP_TIMING_SCHEMA = "axiom-encode/encode-loop-timing/v1"

#: Time no instrumented phase (or tool) claims.
PHASE_OTHER = "other"
#: Resolve paths, load the corpus release and source unit, resolve the
#: replacement target, prepare the eval workspace and build the prompt.
PHASE_PREPARE = "prepare"
#: Validate a retained repair candidate before any model call.
PHASE_RETAINED_CANDIDATE_PREFLIGHT = "retained_candidate_preflight"
#: The generation model call(s), including the empty-artifact retry call and
#: writing the model's files into the eval output.
PHASE_MODEL_CALL = "model_call"
#: Overlay the previous try's candidate onto a partial repair output.
PHASE_REPAIR_OVERLAY = "repair_overlay"
#: Hydrate the eval root around the candidate and write the model trace.
PHASE_STAGE_CANDIDATE = "stage_candidate"
#: Standalone validation of the candidate (compile, CI, grounding, proof and
#: completeness checks, numeric metrics), repeated after each artifact repair.
PHASE_CANDIDATE_VALIDATION = "candidate_validation"
#: The reviewer model call.
PHASE_REVIEW_MODEL_CALL = "review_model_call"
#: Deterministic repairs of the candidate between validation rounds.
PHASE_ARTIFACT_REPAIR = "artifact_repair"
#: Hash the artifacts, build and emit the eval result, print the try summary
#: and log the run.
PHASE_RECORD_RESULT = "record_result"
#: ``--apply`` only: apply-time deterministic repairs and checks between
#: policy-overlay validations.
PHASE_APPLY_REPAIR = "apply_repair"
#: ``--apply`` only: validate the candidate and its dependents in a staged
#: copy of the RuleSpec checkout.
PHASE_OVERLAY_VALIDATION = "overlay_validation"
#: ``--apply`` only: install the files and write the signed manifest.
PHASE_APPLY_WRITE = "apply_write"
#: Decide whether to retry, capture the rejected candidate and clear the
#: next try's output.
PHASE_RETRY_HANDOFF = "retry_handoff"

#: Compile RuleSpec through the Axiom rules engine (and axiom-compose).
TOOL_RULES_ENGINE_COMPILE = "rules_engine_compile"
#: CI checks other than compile, test execution and completeness: numeric
#: grounding, proof atoms and the structural lints.
TOOL_CI_STATIC_CHECKS = "ci_static_checks"
#: Execute the companion test cases in the rules engine.
TOOL_CI_TEST_CASES = "ci_test_cases"
#: Complete-source-unit coverage checks.
TOOL_SOURCE_COMPLETENESS_CHECKS = "source_completeness_checks"
#: PolicyEngine oracle comparison.
TOOL_POLICYENGINE_ORACLE = "policyengine_oracle"

_F = TypeVar("_F", bound=Callable[..., Any])

# The first timer in a process starts its clock at this module's import, which
# ``axiom_encode/__init__.py`` triggers first, so a loop's setup includes the
# package import and command dispatch. Later timers start at their creation.
_PROCESS_ORIGIN: tuple[float, datetime] | None = (
    time.monotonic(),
    datetime.now(timezone.utc),
)
_PROCESS_ORIGIN_LOCK = threading.Lock()


def _claim_process_origin() -> tuple[float, datetime] | None:
    global _PROCESS_ORIGIN
    with _PROCESS_ORIGIN_LOCK:
        origin, _PROCESS_ORIGIN = _PROCESS_ORIGIN, None
    return origin


def format_utc(moment: datetime) -> str:
    """ISO-8601 UTC with millisecond precision and a ``Z`` suffix."""
    return (
        moment.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _print_line(line: str) -> None:
    try:
        print(line, flush=True)
    except (OSError, ValueError):
        pass


@dataclass(frozen=True)
class PhaseTiming:
    """One contiguous segment of a try's timeline."""

    name: str
    started_at: str
    finished_at: str
    duration_ms: int
    breakdown_ms: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
        }
        if self.breakdown_ms:
            payload["breakdown_ms"] = dict(self.breakdown_ms)
        return payload


@dataclass(frozen=True)
class TryTiming:
    """Wall time of one try and its phase timeline."""

    attempt: int
    started_at: str
    finished_at: str
    wall_duration_ms: int
    phases: tuple[PhaseTiming, ...]

    def phase_dicts(self) -> list[dict[str, Any]]:
        return [phase.to_dict() for phase in self.phases]


@dataclass
class _Segment:
    name: str
    start_ms: int
    end_ms: int
    breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(eq=False)
class _Frame:
    # Identity, not value, equality: nested phases may share a name.
    name: str
    tool_depth: int


class _OpenTry:
    def __init__(self, attempt: int, start_ms: int) -> None:
        self.attempt = attempt
        self.start_ms = start_ms
        self.base = PHASE_OTHER
        self.frames: list[_Frame] = []
        self.tools: list[str] = []
        self.cursor_ms = start_ms
        self.segments: list[_Segment] = [_Segment(PHASE_OTHER, start_ms, start_ms)]
        # When the effective phase last changed: a progress line's ``prev``
        # counts from here, also after a sub-ms phase merged back.
        self.last_transition_ms = start_ms

    def effective_phase(self) -> str:
        return self.frames[-1].name if self.frames else self.base

    def effective_tool(self) -> str:
        floor = self.frames[-1].tool_depth if self.frames else 0
        return self.tools[-1] if len(self.tools) > floor else PHASE_OTHER

    def advance(self, now_ms: int) -> None:
        elapsed = now_ms - self.cursor_ms
        if elapsed <= 0:
            return
        segment = self.segments[-1]
        tool = self.effective_tool()
        segment.breakdown[tool] = segment.breakdown.get(tool, 0) + elapsed
        segment.end_ms = now_ms
        self.cursor_ms = now_ms

    def open_segment(self, name: str, now_ms: int) -> None:
        """Start a timeline segment for ``name`` at ``now_ms``."""
        current = self.segments[-1]
        if current.end_ms == current.start_ms:
            # Drop an empty segment; resume its predecessor when it has the
            # same name and ends here, so A, (empty B), A stays one A.
            self.segments.pop()
            if (
                self.segments
                and self.segments[-1].name == name
                and self.segments[-1].end_ms == now_ms
            ):
                return
        self.segments.append(_Segment(name, now_ms, now_ms))


class EncodeLoopTimer:
    """Partition one encode loop's wall time into tries and phases."""

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        origin: tuple[float, datetime] | None = None,
        emit: Callable[[str], None] | None = _print_line,
    ) -> None:
        self._monotonic = monotonic
        #: Where ``started_at`` comes from: ``process`` (this module's import,
        #: claimed by the first timer), ``encode_loop`` (this timer's
        #: creation) or ``explicit`` (the caller's ``origin``).
        self.origin_kind = "explicit"
        if origin is None and monotonic is time.monotonic:
            # The import-time origin is on the real monotonic clock only.
            origin = _claim_process_origin()
            self.origin_kind = "process"
        if origin is None:
            origin = (monotonic(), datetime.now(timezone.utc))
            self.origin_kind = "encode_loop"
        self._origin_monotonic, origin_wall = origin
        self._origin_wall = origin_wall.astimezone(timezone.utc)
        self._emit = emit
        self._thread_id = threading.get_ident()
        self._last_ms = 0
        self._current: _OpenTry | None = None
        self._tries: list[TryTiming] = []
        self._try_bounds: list[tuple[int, int]] = []

    # -- clock -------------------------------------------------------------

    def _now_ms(self) -> int:
        now = int(round((self._monotonic() - self._origin_monotonic) * 1000))
        # Never step backwards, so every segment has a non-negative length.
        self._last_ms = max(self._last_ms, now)
        return self._last_ms

    def _at(self, offset_ms: int) -> str:
        return format_utc(self._origin_wall + timedelta(milliseconds=offset_ms))

    def owns_current_thread(self) -> bool:
        return threading.get_ident() == self._thread_id

    def _line(self, text: str) -> None:
        if self._emit is not None:
            self._emit(text)

    # -- tries -------------------------------------------------------------

    @property
    def tries(self) -> tuple[TryTiming, ...]:
        """Closed tries, in order."""
        return tuple(self._tries)

    def start_try(self, attempt: int) -> None:
        """Start a try; an open try ends at the same instant."""
        now_ms = self._now_ms()
        if self._current is not None:
            self._close_try(now_ms)
        self._current = _OpenTry(attempt, now_ms)
        self._line(f"  try={attempt} start at={self._at(now_ms)}")

    def finish_try(self) -> TryTiming | None:
        """End the open try (if any) now and return its timing."""
        if self._current is None:
            return None
        return self._close_try(self._now_ms())

    def _close_try(self, now_ms: int) -> TryTiming:
        open_try = self._current
        assert open_try is not None
        open_try.advance(now_ms)
        previous = open_try.effective_phase()
        previous_ms = self._since_transition(open_try, now_ms)
        segments = [
            segment
            for segment in open_try.segments
            if segment.end_ms > segment.start_ms
        ]
        timing = TryTiming(
            attempt=open_try.attempt,
            started_at=self._at(open_try.start_ms),
            finished_at=self._at(now_ms),
            wall_duration_ms=now_ms - open_try.start_ms,
            phases=tuple(self._phase_timing(segment) for segment in segments),
        )
        self._tries.append(timing)
        self._try_bounds.append((open_try.start_ms, now_ms))
        self._current = None
        self._line(
            f"  try={open_try.attempt} end at={self._at(now_ms)} "
            f"wall_ms={timing.wall_duration_ms}"
            + self._previous_suffix(previous, previous_ms)
        )
        return timing

    def _phase_timing(self, segment: _Segment) -> PhaseTiming:
        breakdown = {
            name: milliseconds
            for name, milliseconds in sorted(segment.breakdown.items())
            if milliseconds > 0
        }
        return PhaseTiming(
            name=segment.name,
            started_at=self._at(segment.start_ms),
            finished_at=self._at(segment.end_ms),
            duration_ms=segment.end_ms - segment.start_ms,
            breakdown_ms=(breakdown if set(breakdown) - {PHASE_OTHER} else None),
        )

    @staticmethod
    def _since_transition(open_try: _OpenTry, now_ms: int) -> int:
        return now_ms - open_try.last_transition_ms

    @staticmethod
    def _previous_suffix(previous: str, previous_ms: int) -> str:
        return f" prev={previous}:{previous_ms}ms" if previous_ms > 0 else ""

    # -- phases ------------------------------------------------------------

    def _transition(self, open_try: _OpenTry, now_ms: int, before: str) -> None:
        after = open_try.effective_phase()
        if after == before:
            return
        previous_ms = self._since_transition(open_try, now_ms)
        open_try.open_segment(after, now_ms)
        open_try.last_transition_ms = now_ms
        self._line(
            f"  try={open_try.attempt} phase={after} at={self._at(now_ms)}"
            + self._previous_suffix(before, previous_ms)
        )

    def mark(self, name: str | None) -> None:
        """Set the open try's base phase; ``None`` resets it to ``other``."""
        open_try = self._current
        if open_try is None:
            return
        now_ms = self._now_ms()
        open_try.advance(now_ms)
        before = open_try.effective_phase()
        open_try.base = name or PHASE_OTHER
        self._transition(open_try, now_ms, before)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Attribute the enclosed time to ``name`` (nested phases win)."""
        open_try = self._current
        if open_try is None:
            yield
            return
        now_ms = self._now_ms()
        open_try.advance(now_ms)
        before = open_try.effective_phase()
        frame = _Frame(name, len(open_try.tools))
        open_try.frames.append(frame)
        self._transition(open_try, now_ms, before)
        try:
            yield
        finally:
            if self._current is open_try and frame in open_try.frames:
                now_ms = self._now_ms()
                open_try.advance(now_ms)
                before = open_try.effective_phase()
                open_try.frames.remove(frame)
                self._transition(open_try, now_ms, before)

    @contextmanager
    def tool(self, name: str) -> Iterator[None]:
        """Add the enclosed time to the current phase's ``breakdown_ms``."""
        open_try = self._current
        if open_try is None:
            yield
            return
        open_try.advance(self._now_ms())
        open_try.tools.append(name)
        depth = len(open_try.tools)
        try:
            yield
        finally:
            if self._current is open_try and len(open_try.tools) >= depth:
                open_try.advance(self._now_ms())
                del open_try.tools[depth - 1 :]

    # -- loop totals -------------------------------------------------------

    def loop_timing(self) -> dict[str, Any]:
        """Loop totals at this instant; ``setup + tries + between + finalize``
        equals ``wall_duration_ms``.  An open try counts up to now."""
        now_ms = self._now_ms()
        bounds = list(self._try_bounds)
        if self._current is not None:
            bounds.append((self._current.start_ms, now_ms))
        tries_ms = sum(end - start for start, end in bounds)
        setup_ms = bounds[0][0] if bounds else now_ms
        between_ms = sum(
            later[0] - earlier[1] for earlier, later in zip(bounds, bounds[1:])
        )
        finalize_ms = now_ms - bounds[-1][1] if bounds else 0
        return {
            "schema": ENCODE_LOOP_TIMING_SCHEMA,
            "origin": self.origin_kind,
            "started_at": self._at(0),
            "finished_at": self._at(now_ms),
            "wall_duration_ms": now_ms,
            "setup_ms": setup_ms,
            "tries_ms": tries_ms,
            "between_tries_ms": between_ms,
            "finalize_ms": finalize_ms,
            "try_count": len(bounds),
        }


def _timing_phase_payload(phase: Any) -> dict[str, Any] | None:
    """One well-formed phase record, or ``None`` for anything malformed."""
    if not isinstance(phase, dict):
        return None
    name = phase.get("name")
    duration_ms = phase.get("duration_ms")
    if (
        not isinstance(name, str)
        or not name
        or isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
    ):
        return None
    payload: dict[str, Any] = {
        "name": name,
        "started_at": phase.get("started_at"),
        "finished_at": phase.get("finished_at"),
        "duration_ms": duration_ms,
    }
    breakdown = phase.get("breakdown_ms")
    if isinstance(breakdown, dict):
        cleaned = {
            str(key): value
            for key, value in breakdown.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        if cleaned:
            payload["breakdown_ms"] = cleaned
    return payload


def iteration_timing_payload(iteration: Any) -> dict[str, Any]:
    """The timing fields an iteration (object or stored dict) recorded.

    Shared by the local DB, the run log and the Supabase sync so every copy
    of an iteration carries the same keys. Unset or malformed fields are
    omitted.
    """

    def value_of(name: str) -> Any:
        if isinstance(iteration, dict):
            return iteration.get(name)
        return getattr(iteration, name, None)

    payload: dict[str, Any] = {}
    for name in ("started_at", "finished_at"):
        value = value_of(name)
        if isinstance(value, str) and value:
            payload[name] = value
    wall = value_of("wall_duration_ms")
    if isinstance(wall, int) and not isinstance(wall, bool):
        payload["wall_duration_ms"] = wall
    phases = value_of("phases")
    if isinstance(phases, list):
        payload["phases"] = [
            cleaned
            for cleaned in (_timing_phase_payload(phase) for phase in phases)
            if cleaned is not None
        ]
    return payload


_ACTIVE_TIMER: ContextVar[EncodeLoopTimer | None] = ContextVar(
    "axiom_encode_loop_timer", default=None
)


@contextmanager
def activate_encode_loop_timer(timer: EncodeLoopTimer) -> Iterator[EncodeLoopTimer]:
    """Make ``timer`` the one :func:`encode_phase` and friends report to."""
    token = _ACTIVE_TIMER.set(timer)
    try:
        yield timer
    finally:
        _ACTIVE_TIMER.reset(token)


def active_encode_loop_timer() -> EncodeLoopTimer | None:
    """The active timer, when the caller runs on its owning thread."""
    timer = _ACTIVE_TIMER.get()
    if timer is None or not timer.owns_current_thread():
        return None
    return timer


@contextmanager
def encode_phase(name: str) -> Iterator[None]:
    """Scoped phase on the active timer; a no-op without one."""
    timer = active_encode_loop_timer()
    if timer is None:
        yield
        return
    with timer.phase(name):
        yield


@contextmanager
def encode_tool(name: str) -> Iterator[None]:
    """Scoped tool measurement on the active timer; a no-op without one."""
    timer = active_encode_loop_timer()
    if timer is None:
        yield
        return
    with timer.tool(name):
        yield


def mark_encode_phase(name: str | None) -> None:
    """Set the active try's base phase; a no-op without an active timer."""
    timer = active_encode_loop_timer()
    if timer is not None:
        timer.mark(name)


def timed_phase(name: str) -> Callable[[_F], _F]:
    """Decorate a function so each call runs inside ``encode_phase(name)``."""

    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with encode_phase(name):
                return function(*args, **kwargs)

        wrapper.encode_timing = ("phase", name)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def timed_tool(name: str) -> Callable[[_F], _F]:
    """Decorate a function so each call runs inside ``encode_tool(name)``."""

    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with encode_tool(name):
                return function(*args, **kwargs)

        wrapper.encode_timing = ("tool", name)  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorate


def line_buffer_stdout() -> None:
    """Flush stdout at every newline.

    The supervised encoder runs as ``python -I``, and isolated mode ignores
    ``PYTHONUNBUFFERED``.  With stdout on a pipe (as in GitHub Actions)
    Python then block-buffers it, so progress lines reach the job log in 8 KiB
    bursts and their log timestamps say nothing about when they happened.
    """
    stream = sys.stdout
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(line_buffering=True)
    except (AttributeError, OSError, ValueError, TypeError):
        pass
