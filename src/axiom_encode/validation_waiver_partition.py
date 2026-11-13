"""Deterministic ownership, ordering and progress for the waiver audit.

The shared RuleSpec validator runs ``validation-waivers audit`` once per matrix
shard. Every shard passes the same ``--partition-keys-json``, loads the same
complete waiver ledger and reads the same optional cost table from the same
head checkout. Each shard therefore derives the same owner for every waiver
path, and each path has exactly one auditing shard.

Without a cost table every path weighs the same, and the assignment equals the
round-robin it replaced: the ``i``-th path of the sorted ledger belongs to the
``i mod n``-th sorted key. A repository may commit a table at
``COST_TABLE_PATH`` that records each module's measured audit time and,
optionally, how much other work each shard's job does. Paths then go longest
first to the key with the least work so far (LPT list scheduling), each key
starting from its baseline, and a shard starts its own modules longest first.

The table only places and orders work. Nothing in it reaches a fingerprint, a
verdict or a result row.

Invariants (``tests/test_validation_waiver_partition_balance.py``):

* exactly once: every path has exactly one owner among the keys;
* determinism: ownership is a pure function of the path set, the key set and
  the table; the order of either input does not change it;
* legacy equivalence: without a table, or with equal costs and no baselines,
  ownership equals the legacy round-robin;
* bounded imbalance: with ``n`` keys, ``n`` times the largest final load is at
  most ``max(n * largest baseline, total cost + total baseline + (n - 1) *
  largest cost)``;
* with no baselines, the largest load is at most ``4/3 - 1/(3n)`` times the
  smallest achievable largest load (Graham 1969).
"""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import json
import os
import re
import statistics
import sys
import threading
import time
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from axiom_encode.corpus_resolver import (
    CorpusResolutionError,
    read_bounded_regular_file,
)

COST_TABLE_PATH = ".axiom/validation-waiver-audit-costs.json"
COST_TABLE_SCHEMA = "axiom-encode/validation-waiver-audit-costs/v1"
COST_TABLE_MAX_BYTES = 4 * 1024 * 1024
# About 115 days. A larger value is a unit error, not a measurement.
MAX_MILLISECONDS = 10**10
DEFAULT_MODULE_MILLISECONDS = 1_000
_MAX_SOURCE_LENGTH = 512
_MAX_KEY_LENGTH = 256
_TOP_LEVEL_FIELDS = frozenset(
    {"schema", "source", "module_milliseconds", "partition_baseline_milliseconds"}
)

# The audit writes one of these per module when that module finishes:
# ``validation-waivers audit: <path> <seconds with three decimals>s``. GitHub
# job logs prefix each line with a timestamp, and ``gh run view --log`` adds
# the job and step names separated by tabs, so allow any prefix that ends in a
# space or a tab.
_TIMING_LINE = re.compile(
    r"(?:^|[\t ])validation-waivers audit: (?P<path>\S(?:.*\S)?) "
    r"(?P<seconds>[0-9]+)\.(?P<millis>[0-9]{3})s$"
)
# The audit prints this once per shard (``_cmd_validation_waivers_audit``), saying
# how many modules it will execute and so how many timing lines should follow.
_PLAN_LINE = re.compile(
    r"(?:^|[\t ])validation-waivers audit plan: (?P<owner>.*); [0-9]+ selected, "
    r"(?P<execute>[0-9]+) to execute$"
)
# A partitioned audit's owner text. A key may hold inner spaces, so the match is
# anchored on the last "(i of n, assignment) owns" the writer appends.
_PLAN_PARTITION = re.compile(
    r"^partition (?P<key>.+) \((?P<index>[0-9]+) of (?P<count>[0-9]+), "
    r"(?:round-robin|cost-balanced)\) owns [0-9]+ of [0-9]+ waiver path\(s\)$"
)
_JOB_KEY = re.compile(r"\((?P<key>[^()]+)\)$")
DEFAULT_AUDIT_STEP_NAME = "Enforce validation waiver ratchet"

AUDIT_LINE_PREFIX = "validation-waivers audit"
RECHECK_LINE_PREFIX = "validation-waivers audit recheck"
HEARTBEAT_SECONDS = 60.0


class WaiverAuditCostTableError(ValueError):
    """The committed cost table cannot be read exactly."""


def _contains_control_or_format(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in value
    )


def _canonical_path(raw: object, *, label: str) -> str:
    """Return ``raw`` if it is a canonical repository-relative POSIX path."""
    if not isinstance(raw, str) or not raw:
        raise WaiverAuditCostTableError(f"{label} must be a non-empty string")
    if _contains_control_or_format(raw):
        raise WaiverAuditCostTableError(
            f"{label} must not contain control or format characters: {raw!r}"
        )
    if (
        raw != raw.strip()
        or "\\" in raw
        or raw.startswith("/")
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        raise WaiverAuditCostTableError(
            f"{label} must be a canonical repository-relative path: {raw!r}"
        )
    return raw


def _partition_key(raw: object, *, label: str) -> str:
    if not isinstance(raw, str) or not raw or len(raw) > _MAX_KEY_LENGTH:
        raise WaiverAuditCostTableError(
            f"{label} must be a non-empty string of at most {_MAX_KEY_LENGTH} "
            "characters"
        )
    if raw != raw.strip() or _contains_control_or_format(raw):
        raise WaiverAuditCostTableError(
            f"{label} must not contain surrounding whitespace, control or "
            f"format characters: {raw!r}"
        )
    return raw


def _milliseconds(raw: object, *, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise WaiverAuditCostTableError(f"{label} must be integer milliseconds")
    if not 0 <= raw <= MAX_MILLISECONDS:
        raise WaiverAuditCostTableError(
            f"{label} must be between 0 and {MAX_MILLISECONDS} milliseconds"
        )
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WaiverAuditCostTableError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_float(raw: str) -> float:
    raise WaiverAuditCostTableError(
        f"cost table numbers must be integer milliseconds, not {raw!r}"
    )


def _reject_constant(raw: str) -> float:
    raise WaiverAuditCostTableError(f"cost table must not contain {raw}")


@dataclass(frozen=True, eq=False)
class WaiverAuditCosts:
    """A parsed cost table: per-module and per-key work in milliseconds."""

    module_milliseconds: Mapping[str, int]
    baseline_milliseconds: Mapping[str, int]
    default_milliseconds: int
    sha256: str
    source: str | None = None

    def cost(self, path: str) -> int:
        """Planning weight of one module; at least 1 so no path weighs nothing."""
        return max(1, self.module_milliseconds.get(path, self.default_milliseconds))

    def baseline(self, key: str) -> int:
        """Work a key's job does besides its audit share, in module time."""
        return self.baseline_milliseconds.get(key, 0)


def parse_waiver_audit_costs(raw: bytes) -> WaiverAuditCosts:
    """Parse the strict JSON cost table; reject anything ambiguous."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WaiverAuditCostTableError("cost table must be UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except WaiverAuditCostTableError:
        raise
    except (ValueError, RecursionError) as error:
        # JSONDecodeError, an integer past Python's digit limit, and nesting
        # past the recursion limit all mean the same thing here.
        raise WaiverAuditCostTableError(
            f"cost table is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise WaiverAuditCostTableError("cost table must be a JSON object")
    unknown = sorted(set(payload) - _TOP_LEVEL_FIELDS)
    if unknown:
        raise WaiverAuditCostTableError(
            f"cost table has unknown fields: {', '.join(unknown)}"
        )
    if payload.get("schema") != COST_TABLE_SCHEMA:
        raise WaiverAuditCostTableError(
            f"cost table schema must be {COST_TABLE_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    source = payload.get("source")
    if source is not None and (
        not isinstance(source, str)
        or not source
        or len(source) > _MAX_SOURCE_LENGTH
        or _contains_control_or_format(source)
    ):
        raise WaiverAuditCostTableError(
            f"cost table source must be a non-empty single-line string of at "
            f"most {_MAX_SOURCE_LENGTH} characters"
        )
    modules = payload.get("module_milliseconds")
    if not isinstance(modules, dict):
        raise WaiverAuditCostTableError(
            "cost table module_milliseconds must be a JSON object"
        )
    module_milliseconds = {
        _canonical_path(path, label="cost table module path"): _milliseconds(
            value, label=f"cost of {path}"
        )
        for path, value in modules.items()
    }
    baselines = payload.get("partition_baseline_milliseconds", {})
    if not isinstance(baselines, dict):
        raise WaiverAuditCostTableError(
            "cost table partition_baseline_milliseconds must be a JSON object"
        )
    baseline_milliseconds = {
        _partition_key(key, label="cost table partition key"): _milliseconds(
            value, label=f"baseline of {key}"
        )
        for key, value in baselines.items()
    }
    default = (
        statistics.median_low(module_milliseconds.values())
        if module_milliseconds
        else DEFAULT_MODULE_MILLISECONDS
    )
    return WaiverAuditCosts(
        module_milliseconds=MappingProxyType(dict(sorted(module_milliseconds.items()))),
        baseline_milliseconds=MappingProxyType(
            dict(sorted(baseline_milliseconds.items()))
        ),
        default_milliseconds=max(1, default),
        sha256=hashlib.sha256(raw).hexdigest(),
        source=source,
    )


def load_waiver_audit_costs(root: Path) -> WaiverAuditCosts | None:
    """Read ``root``'s committed cost table, or ``None`` when it has none.

    A table that exists but cannot be read exactly fails the audit: every shard
    reads the same file, so a silent fallback would only hide the defect.
    """
    root = Path(root)
    candidate = root / COST_TABLE_PATH
    if not os.path.lexists(candidate):
        return None
    try:
        raw = read_bounded_regular_file(
            root,
            candidate,
            label="validation waiver audit cost table",
            max_bytes=COST_TABLE_MAX_BYTES,
        )
    except CorpusResolutionError as error:
        raise WaiverAuditCostTableError(str(error)) from error
    try:
        return parse_waiver_audit_costs(raw)
    except WaiverAuditCostTableError as error:
        raise WaiverAuditCostTableError(f"{COST_TABLE_PATH}: {error}") from error


def assign_waiver_owners(
    paths: Iterable[str],
    keys: Iterable[str],
    costs: WaiverAuditCosts | None = None,
) -> dict[str, str]:
    """Map every waiver path to the one partition key that audits it.

    Paths go in decreasing cost (ties in path order) to the key with the
    smallest load (ties in key order). Loads start at each key's baseline.
    Integer milliseconds keep every shard's arithmetic exact and identical.
    """
    ordered_paths = sorted(paths)
    if len(set(ordered_paths)) != len(ordered_paths):
        raise ValueError("waiver audit paths must be unique")
    ordered_keys = sorted(keys)
    if not ordered_keys:
        raise ValueError("waiver audit partition needs at least one key")
    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("waiver audit partition keys must be unique")
    weights = [1 if costs is None else costs.cost(path) for path in ordered_paths]
    loads = [
        (0 if costs is None else costs.baseline(key), index)
        for index, key in enumerate(ordered_keys)
    ]
    heapq.heapify(loads)
    owner_index: dict[str, int] = {}
    for position in sorted(
        range(len(ordered_paths)), key=lambda item: (-weights[item], item)
    ):
        load, index = heapq.heappop(loads)
        owner_index[ordered_paths[position]] = index
        heapq.heappush(loads, (load + weights[position], index))
    return {path: ordered_keys[owner_index[path]] for path in ordered_paths}


def baseline_key_mismatch(
    keys: Iterable[str], costs: WaiverAuditCosts | None
) -> str | None:
    """Describe matrix keys and table baselines that do not match, if any.

    Planning is unchanged: a key without a baseline starts at 0 and a
    baseline without a key is unused. The note only makes a stale table
    visible after the audit matrix changes.
    """
    if costs is None or not costs.baseline_milliseconds:
        return None
    matrix = set(keys)
    table = set(costs.baseline_milliseconds)
    missing = sorted(matrix - table)
    unused = sorted(table - matrix)
    if not missing and not unused:
        return None
    return (
        "waiver audit cost table: partition baselines do not match the matrix "
        f"keys (no baseline, planned from 0 ms: {', '.join(missing) or 'none'}; "
        f"baseline for a key not in the matrix: {', '.join(unused) or 'none'})"
    )


def partition_loads(
    owners: Mapping[str, str],
    keys: Iterable[str],
    costs: WaiverAuditCosts | None = None,
) -> dict[str, int]:
    """Each key's planned load: its baseline plus its paths' weights."""
    loads = {key: 0 if costs is None else costs.baseline(key) for key in sorted(keys)}
    for path, key in owners.items():
        loads[key] += 1 if costs is None else costs.cost(path)
    return loads


def audit_order(
    paths: Iterable[str], costs: WaiverAuditCosts | None = None
) -> list[str]:
    """The order a shard starts its modules: longest first when costs exist."""
    ordered = sorted(paths)
    if costs is None:
        return ordered
    return sorted(ordered, key=lambda path: (-costs.cost(path), path))


def timing_observations(lines: Iterable[str]) -> dict[str, list[int]]:
    """Collect each module's audit time from the audit's timing lines."""
    observed: dict[str, list[int]] = {}
    for raw in lines:
        match = _TIMING_LINE.search(raw.rstrip("\r\n"))
        if match is None:
            continue
        path = _canonical_path(match["path"], label="timing line module path")
        milliseconds = int(match["seconds"]) * 1000 + int(match["millis"])
        observed.setdefault(path, []).append(
            _milliseconds(milliseconds, label=f"time of {path}")
        )
    return observed


@dataclass(frozen=True)
class AuditPlan:
    """One shard's audit plan line: how many modules it will execute and, for
    a partitioned audit, which shard of how many it is (1-based)."""

    execute: int
    key: str | None = None
    index: int | None = None
    count: int | None = None


def audit_plans(lines: Iterable[str]) -> list[AuditPlan]:
    """Every audit plan line in a log, in order.

    A log of one shard has one; ``gh run view --log`` for a whole run has one
    per shard. A failed download, or a log from an encoder that predates the
    plan line, has none.
    """
    plans: list[AuditPlan] = []
    for raw in lines:
        match = _PLAN_LINE.search(raw.rstrip("\r\n"))
        if match is None:
            continue
        execute = int(match["execute"])
        owner = _PLAN_PARTITION.match(match["owner"])
        if owner is None:
            plans.append(AuditPlan(execute))
        else:
            plans.append(
                AuditPlan(
                    execute,
                    key=owner["key"],
                    index=int(owner["index"]),
                    count=int(owner["count"]),
                )
            )
    return plans


def planned_executions(lines: Iterable[str]) -> int | None:
    """How many modules the log's audit plan lines say will execute, summed
    over its shards; ``None`` when the log has no plan line."""
    plans = audit_plans(lines)
    return sum(plan.execute for plan in plans) if plans else None


def plan_coverage_gaps(
    plans: Sequence[AuditPlan], expected_keys: Iterable[str] = ()
) -> list[str]:
    """Shards that a set of logs should account for but does not.

    Partitioned plan lines say how many shards the run had, so a missing log
    shows as a missing index. ``expected_keys`` (the jobs listing's shards)
    names the missing ones. Logs without plan lines can't be checked.
    """
    partitioned = [plan for plan in plans if plan.count is not None]
    if not partitioned:
        return []
    gaps: list[str] = []
    counts = sorted({plan.count for plan in partitioned if plan.count is not None})
    if len(counts) > 1:
        gaps.append(
            "plan lines disagree on the shard count: "
            + ", ".join(str(count) for count in counts)
        )
    seen = {plan.index for plan in partitioned}
    if len(seen) < counts[-1]:
        gaps.append(f"plan lines cover {len(seen)} of {counts[-1]} shard(s)")
    absent = sorted(set(expected_keys) - {plan.key for plan in partitioned})
    if absent:
        shown = ", ".join(absent[:10]) + (", ..." if len(absent) > 10 else "")
        gaps.append(
            f"the jobs listing names {len(absent)} shard(s) that no plan line "
            f"names: {shown}"
        )
    return gaps


def load_jobs_listing(text: str) -> Any:
    """Decode a jobs listing: one JSON value, or several printed back to back.

    ``gh api --paginate`` without ``--slurp`` prints one JSON object per page
    with nothing between them, which ``json.loads`` rejects as extra data.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            break
        value, index = decoder.raw_decode(text, index)
        values.append(value)
    if not values:
        raise ValueError("jobs JSON is empty")
    return values[0] if len(values) == 1 else values


def _iter_jobs(payload: Any) -> Iterator[Mapping[str, Any]]:
    """Accept one jobs page, a ``--slurp`` list of pages, or a bare job list."""
    if isinstance(payload, Mapping):
        if "jobs" not in payload:
            raise ValueError("jobs JSON object must contain a 'jobs' array")
        payload = payload["jobs"]
        if not isinstance(payload, list):
            raise ValueError("jobs JSON 'jobs' must be an array")
        for job in payload:
            if not isinstance(job, Mapping):
                raise ValueError("every job must be a JSON object")
            yield job
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping) and "jobs" in item:
                yield from _iter_jobs(item)
            elif isinstance(item, Mapping):
                yield item
            else:
                raise ValueError("jobs JSON array items must be objects")
        return
    raise ValueError("jobs JSON must be an object or an array")


def _timestamp(raw: object, *, label: str) -> datetime:
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be an ISO 8601 timestamp")
    try:
        return datetime.fromisoformat(raw)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO 8601 timestamp: {raw!r}") from error


def partition_baselines_from_jobs(
    payload: Any,
    *,
    workers: int,
    audit_step: str = DEFAULT_AUDIT_STEP_NAME,
) -> dict[str, int]:
    """Each matrix job's non-audit time, as module time for ``workers`` workers.

    ``payload`` is the GitHub Actions jobs listing of one run. A job whose name
    ends in ``(<key>)`` and that ran ``audit_step`` contributes
    ``(job time - audit step time) * workers``: an audit shard with ``workers``
    workers finishes ``workers`` seconds of module time per second, so this
    puts a job's other work on the same scale as module costs.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    baselines: dict[str, int] = {}
    for job in _iter_jobs(payload):
        name = job.get("name")
        match = _JOB_KEY.search(name) if isinstance(name, str) else None
        steps = job.get("steps")
        if match is None or not isinstance(steps, list):
            continue
        step = next(
            (
                item
                for item in steps
                if isinstance(item, Mapping) and item.get("name") == audit_step
            ),
            None,
        )
        if step is None or step.get("conclusion") not in {"success", "failure"}:
            continue
        key = _partition_key(match["key"], label=f"partition key of job {name!r}")
        job_seconds = (
            _timestamp(job.get("completed_at"), label=f"{name} completed_at")
            - _timestamp(job.get("started_at"), label=f"{name} started_at")
        ).total_seconds()
        step_seconds = (
            _timestamp(step.get("completed_at"), label=f"{name} step completed_at")
            - _timestamp(step.get("started_at"), label=f"{name} step started_at")
        ).total_seconds()
        if key in baselines:
            raise ValueError(f"jobs JSON lists partition key {key!r} more than once")
        baselines[key] = _milliseconds(
            round(max(0.0, job_seconds - step_seconds) * workers * 1000),
            label=f"baseline of {key}",
        )
    return dict(sorted(baselines.items()))


def render_waiver_audit_costs(
    observations: Mapping[str, Sequence[int]],
    *,
    baselines: Mapping[str, int] | None = None,
    source: str | None = None,
) -> bytes:
    """Render a cost table that ``parse_waiver_audit_costs`` reads back exactly.

    A module observed more than once (several logs) takes its median time.
    """
    if not observations:
        raise ValueError("no 'validation-waivers audit: <path> <seconds>s' lines")
    payload: dict[str, Any] = {"schema": COST_TABLE_SCHEMA}
    if source is not None:
        payload["source"] = source
    payload["module_milliseconds"] = {
        path: statistics.median_low(sorted(values))
        for path, values in sorted(observations.items())
    }
    if baselines:
        payload["partition_baseline_milliseconds"] = dict(sorted(baselines.items()))
    rendered = (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    # The writer never emits a table the reader would reject.
    parse_waiver_audit_costs(rendered)
    return rendered


def write_audit_line(line: str) -> None:
    """Write one whole diagnostic line to stderr; stdout carries only reports."""
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


class WaiverAuditProgress:
    """Which modules an audit phase has started and finished, and since when.

    The parent process owns it: in serial mode its main thread reports each
    module, and in parallel mode it reports what it dispatches and collects.
    ``heartbeat_line`` names every running module and how long it has run, so
    a long module reads as running, not as a hung job.
    """

    def __init__(
        self,
        total: int,
        *,
        prefix: str = AUDIT_LINE_PREFIX,
        clock: Any = time.monotonic,
    ) -> None:
        self.total = total
        self.prefix = prefix
        self._clock = clock
        self._began = clock()
        self._finished = 0
        self._running: dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self, path: str) -> None:
        with self._lock:
            self._running[path] = self._clock()
        write_audit_line(f"{self.prefix} start: {path}")

    def finish(self, path: str) -> None:
        with self._lock:
            if self._running.pop(path, None) is not None:
                self._finished += 1

    def heartbeat_line(self) -> str:
        with self._lock:
            now = self._clock()
            running = sorted(self._running.items(), key=lambda item: (item[1], item[0]))
            finished = self._finished
        detail = "; ".join(f"{path} {now - began:.0f}s" for path, began in running)
        return (
            f"{self.prefix} heartbeat: {finished} of {self.total} module(s) "
            f"finished after {now - self._began:.0f}s; {len(running)} running"
            + (f": {detail}" if detail else "")
        )

    def heartbeat(self) -> None:
        write_audit_line(self.heartbeat_line())


@contextlib.contextmanager
def heartbeat_thread(
    progress: WaiverAuditProgress, *, interval: float | None = None
) -> Iterator[None]:
    """Print ``progress``'s heartbeat every ``interval`` seconds meanwhile.

    For the serial executor, whose main thread is busy inside one module. The
    parallel executor's parent waits on its futures with a timeout instead,
    so no thread runs while its pool starts worker processes.
    """
    seconds = HEARTBEAT_SECONDS if interval is None else interval
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(seconds):
            progress.heartbeat()

    thread = threading.Thread(
        target=beat, name="validation-waiver-audit-heartbeat", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
