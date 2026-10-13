"""Split the test suite across CI shards without dropping a test.

CI runs the suite as several parallel jobs. Each one sets
``AXIOM_ENCODE_TEST_SHARD=k/N`` on its pytest run, and ``tests/conftest.py``
keeps only the tests this module assigns to shard ``k``. Unset, nothing here
runs and pytest behaves as it always has.

The unit of assignment is the test function (a parametrized function's cases
stay together), because a few functions take most of the suite's time and
their files must be split. ``assign`` is a pure function of the collected
functions, how many cases each has, and ``ci_shard_seconds.json``; every shard
collects the same suite, so every shard computes the same assignment and each
function runs in exactly one of them.

That argument needs every shard to see the same collection and the same
table, and each shard to run what it kept. So a shard publishes a report only
after pytest exits 0 having run every test it kept. The report carries a
digest of the collection and of the assignment computed from it. The
``test (3.13)`` job in ``ci.yml`` runs ``check_reports``: it fails unless every
shard published a report, the digests and totals agree, and the kept counts add
up to the collected ones.

``ci_shard_seconds.json`` holds seconds per test file, and per test function
for the slow ones. It only balances the shards; a stale or missing entry never
drops a test. To refresh it, save the "Run tests" logs of a green run's shards
and run::

    python -m tests.ci_shards seconds shard-1.log shard-2.log ... > tests/ci_shard_seconds.json

No shard can be shorter than the slowest single test function.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path

SHARD_ENV = "AXIOM_ENCODE_TEST_SHARD"
REPORT_ENV = "AXIOM_ENCODE_TEST_SHARD_REPORT"
SECONDS_PATH = Path(__file__).with_name("ci_shard_seconds.json")
# Weight of a file the seconds table does not name yet. Most files take about a second.
DEFAULT_FILE_MILLISECONDS = 1_000
# A function gets its own entry in a refreshed table once its listed tests take this long.
FUNCTION_ENTRY_SECONDS = 20

_SHARD = re.compile(r"([1-9][0-9]{0,2})/([1-9][0-9]{0,2})")
_REPORT = re.compile(
    r"([0-9]+)/([0-9]+):([0-9]+):([0-9]+):([0-9]+):([0-9]+):([0-9a-f]{64})"
)
_LOG_FILE_LINE = re.compile(r"(tests/\S+\.py) ([.sxXFE]+)\s*(?:\[\s*\d+%\])?")
_LOG_MORE_LINE = re.compile(r"([.sxXFE]+)\s*(?:\[\s*\d+%\])?")
_LOG_DURATION = re.compile(r"(\d+\.\d+)s (?:call|setup|teardown)\s+(tests/\S+\.py::.+)")
_LOG_STAMP = re.compile(
    r"\ufeff?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?)\d*Z (.*)"
)


def parse_shard(value: str) -> tuple[int, int]:
    """``"k/N"`` as ``(k, N)`` with ``1 <= k <= N``; anything else is an error."""
    match = _SHARD.fullmatch(value)
    if match is None:
        raise ValueError(f"{SHARD_ENV} must look like 3/8, not {value!r}")
    shard, shards = int(match.group(1)), int(match.group(2))
    if shard > shards:
        raise ValueError(f"{SHARD_ENV}={value!r} names a shard past the last one")
    return shard, shards


def unit_of(nodeid: str, name: str, originalname: str | None) -> str:
    """The test function a collected item belongs to.

    A parametrized item's ``name`` is ``originalname`` plus ``[case id]``; its
    unit is the node id without that suffix. An item pytest gives no function
    name for is its own unit.
    """
    if (
        originalname
        and name.startswith(originalname)
        and nodeid.endswith(name)
        and (len(name) == len(originalname) or name[len(originalname)] == "[")
    ):
        return nodeid[: len(nodeid) - len(name)] + originalname
    return nodeid


def _file_of(unit: str) -> str:
    return unit.split("::", 1)[0]


def _milliseconds(seconds: float) -> int:
    return max(0, round(seconds * 1000))


def weights(
    unit_cases: Mapping[str, int], seconds: Mapping[str, float]
) -> dict[str, int]:
    """Each unit's expected time in whole microseconds (at least 1).

    ``seconds`` maps test files, and some test functions, to seconds. A function
    with its own entry weighs that. The others share what is left of their
    file's time in proportion to their case counts.
    """
    shared_cases: Counter[str] = Counter()
    own_milliseconds: Counter[str] = Counter()
    for unit, cases in unit_cases.items():
        if cases < 1:
            raise ValueError(f"{unit} has {cases} cases")
        if "::" in unit and unit in seconds:
            own_milliseconds[_file_of(unit)] += _milliseconds(seconds[unit])
        else:
            shared_cases[_file_of(unit)] += cases
    result = {}
    for unit, cases in unit_cases.items():
        file = _file_of(unit)
        if "::" in unit and unit in seconds:
            microseconds = _milliseconds(seconds[unit]) * 1000
        else:
            shared = (
                DEFAULT_FILE_MILLISECONDS
                if file not in seconds
                else max(0, _milliseconds(seconds[file]) - own_milliseconds[file])
            )
            microseconds = shared * 1000 * cases // shared_cases[file]
        result[unit] = max(1, microseconds)
    return result


def assign(
    unit_cases: Mapping[str, int], seconds: Mapping[str, float], shards: int
) -> dict[str, int]:
    """Shard (1-based) of every unit: heaviest first, each to the lightest shard.

    Depends only on the mappings' contents, never on their order. Ties go to
    the lower shard and, between units, to the lexically smaller one.
    """
    if shards < 1:
        raise ValueError(f"cannot split into {shards} shards")
    unit_weights = weights(unit_cases, seconds)
    loads = [0] * shards
    assignment = {}
    for unit in sorted(unit_weights, key=lambda u: (-unit_weights[u], u)):
        lightest = min(range(shards), key=lambda index: (loads[index], index))
        loads[lightest] += unit_weights[unit]
        assignment[unit] = lightest + 1
    return assignment


def digest(unit_cases: Mapping[str, int], assignment: Mapping[str, int]) -> str:
    """Identifies a collection and its split: each unit, its cases and its shard.

    Two shards with equal digests collected the same functions with the same
    case counts and assigned them the same way, so they hold the same table.
    """
    lines = "".join(
        json.dumps([unit, cases, assignment[unit]]) + "\n"
        for unit, cases in sorted(unit_cases.items())
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def load_seconds(path: Path = SECONDS_PATH) -> dict[str, float]:
    table = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or not all(
        isinstance(key, str)
        and isinstance(value, int | float)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in table.items()
    ):
        raise ValueError(f"{path.name} must map test files and functions to seconds")
    return table


def select(
    units: Sequence[str],
    shard: int,
    shards: int,
    seconds: Mapping[str, float],
) -> tuple[list[bool], str]:
    """Which collected items (given as their units, in order) shard ``k`` keeps.

    Returns the keep flags and the report line the shard publishes.
    """
    if not 1 <= shard <= shards:
        raise ValueError(f"shard {shard} is not one of 1..{shards}")
    unit_cases = Counter(units)
    assignment = assign(unit_cases, seconds, shards)
    keep = [assignment[unit] == shard for unit in units]
    kept_units = sum(1 for unit in unit_cases if assignment[unit] == shard)
    return keep, report(
        shard=shard,
        shards=shards,
        units=len(unit_cases),
        kept_units=kept_units,
        cases=len(units),
        kept_cases=sum(keep),
        collection=digest(unit_cases, assignment),
    )


def report(
    *,
    shard: int,
    shards: int,
    units: int,
    kept_units: int,
    cases: int,
    kept_cases: int,
    collection: str,
) -> str:
    """``k/N:functions:kept functions:tests:kept tests:collection digest``."""
    return f"{shard}/{shards}:{units}:{kept_units}:{cases}:{kept_cases}:{collection}"


def check_reports(reports: Sequence[str]) -> list[str]:
    """Why these shard reports do not prove the whole suite ran; empty if they do.

    ``reports[k - 1]`` must be the report of shard ``k`` of ``len(reports)``;
    pass ``""`` for a shard that published none.
    """
    if not reports:
        return ["no shards"]
    problems = []
    parsed = []
    for index, text in enumerate(reports, start=1):
        match = _REPORT.fullmatch(text)
        if match is None:
            problems.append(f"shard {index} published no valid report: {text!r}")
            continue
        shard, shards, units, kept_units, cases, kept_cases = (
            int(match.group(group)) for group in range(1, 7)
        )
        if (shard, shards) != (index, len(reports)):
            problems.append(
                f"report {index} of {len(reports)} is from shard {shard} of {shards}"
            )
        if kept_units > units or kept_cases > cases:
            problems.append(f"shard {index} kept more than it collected: {text!r}")
        parsed.append((units, kept_units, cases, kept_cases, match.group(7)))
    if problems:
        return problems
    collections = {
        (units, cases, collection) for units, _, cases, _, collection in parsed
    }
    if len(collections) != 1:
        return [f"shards collected different suites: {sorted(collections)}"]
    units, cases, _ = next(iter(collections))
    kept_units = sum(row[1] for row in parsed)
    kept_cases = sum(row[3] for row in parsed)
    if kept_units != units:
        problems.append(f"shards kept {kept_units} test functions of {units}")
    if kept_cases != cases:
        problems.append(f"shards kept {kept_cases} tests of {cases}")
    if cases == 0:
        problems.append("the suite collected no tests")
    return problems


def seconds_from_logs(logs: Iterable[str]) -> dict[str, float]:
    """The seconds table, from the given "Run tests" step logs added together.

    Files: pytest's default progress lines (``tests/x.py ....  [ 5%]``) and the
    timestamp GitHub puts before each one. A file's time in one log runs from
    the previous progress line (or the session start) to its last line.

    Functions: the ``--durations`` lines (``12.34s call tests/x.py::test_y[z]``).
    A function whose listed tests add up to ``FUNCTION_ENTRY_SECONDS`` or more
    gets its own entry. The list is pytest's slowest tests, not all of them, so
    an entry can undercount; the rest of the file's time is shared out anyway.
    """
    totals: Counter[str] = Counter()
    functions: Counter[str] = Counter()
    for log in logs:
        previous = None
        current = None
        for raw in log.splitlines():
            stamped = _LOG_STAMP.fullmatch(raw.split("\t")[-1])
            if stamped is None:
                continue
            at = datetime.fromisoformat(stamped.group(1))
            text = stamped.group(2)
            if "test session starts" in text:
                previous, current = at, None
                continue
            duration = _LOG_DURATION.fullmatch(text)
            if duration is not None:
                functions[duration.group(2).split("[", 1)[0].rstrip()] += float(
                    duration.group(1)
                )
                continue
            if previous is None:
                continue
            file_line = _LOG_FILE_LINE.fullmatch(text)
            if file_line is not None:
                current = file_line.group(1)
            elif _LOG_MORE_LINE.fullmatch(text) is None or current is None:
                continue
            totals[current] += (at - previous).total_seconds()
            previous = at
    table = {file: round(seconds, 1) for file, seconds in totals.items()}
    for function, seconds in functions.items():
        if seconds >= FUNCTION_ENTRY_SECONDS and _file_of(function) in table:
            table[function] = round(seconds, 1)
    return dict(sorted(table.items()))


def main(argv: Sequence[str]) -> int:
    if len(argv) >= 2 and argv[0] == "seconds":
        logs = [
            Path(name).read_text(encoding="utf-8", errors="replace")
            for name in argv[1:]
        ]
        json.dump(seconds_from_logs(logs), sys.stdout, indent=0, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    if len(argv) >= 2 and argv[0] == "check":
        problems = check_reports(argv[1:])
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if not problems:
            print(f"all {len(argv) - 1} shards agree and together ran the whole suite")
        return 1 if problems else 0
    print(
        "usage: python -m tests.ci_shards (seconds LOG... | check REPORT...)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
