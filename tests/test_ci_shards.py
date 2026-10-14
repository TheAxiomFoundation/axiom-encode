"""The CI shards together run every test exactly once, or the gate job fails."""

from __future__ import annotations

import itertools
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from tests import ci_shards

ROOT = Path(__file__).resolve().parents[1]
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))

_NAMES = st.text(alphabet="abcxyz_", min_size=1, max_size=4)
_FILES = _NAMES.map(lambda name: f"tests/test_{name}.py")
_UNITS = st.tuples(_FILES, _NAMES).map(lambda pair: f"{pair[0]}::test_{pair[1]}")
_UNIT_CASES = st.dictionaries(
    _UNITS, st.integers(min_value=1, max_value=40), max_size=30
)
# Seconds for files and, as the refreshed table has for slow tests, for functions.
_SECONDS = st.dictionaries(
    st.one_of(_FILES, _UNITS),
    st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
    max_size=14,
)
_SHARDS = st.integers(min_value=1, max_value=12)


def _items(unit_cases: dict[str, int], seed: int) -> list[str]:
    """One entry per collected test, as its unit, in a shuffled collection order."""
    items = [unit for unit, cases in unit_cases.items() for _ in range(cases)]
    random.Random(seed).shuffle(items)
    return items


def _reports(items, shards, seconds) -> list[str]:
    return [
        ci_shards.select(items, shard, shards, seconds)[1]
        for shard in range(1, shards + 1)
    ]


@given(_UNIT_CASES, _SECONDS, _SHARDS)
def test_every_unit_gets_exactly_one_shard(unit_cases, seconds, shards):
    assignment = ci_shards.assign(unit_cases, seconds, shards)

    assert assignment.keys() == unit_cases.keys()
    assert all(1 <= shard <= shards for shard in assignment.values())


@given(_UNIT_CASES, _SECONDS, _SHARDS, st.integers())
def test_shards_keep_every_test_once_and_their_reports_pass_the_gate(
    unit_cases, seconds, shards, seed
):
    items = _items(unit_cases, seed)
    selections = [
        ci_shards.select(items, shard, shards, seconds)
        for shard in range(1, shards + 1)
    ]

    kept_by = [
        sum(keep[index] for keep, _ in selections) for index in range(len(items))
    ]
    assert kept_by == [1] * len(items)
    problems = ci_shards.check_reports([report for _, report in selections])
    assert problems == ([] if items else ["the suite collected no tests"])


@given(_UNIT_CASES, _SECONDS, _SHARDS, st.integers(), st.integers())
def test_assignment_ignores_collection_and_table_order(
    unit_cases, seconds, shards, seed, other_seed
):
    def reordered(mapping, by):
        keys = list(mapping)
        random.Random(by).shuffle(keys)
        return {key: mapping[key] for key in keys}

    first = ci_shards.assign(unit_cases, seconds, shards)
    second = ci_shards.assign(
        reordered(unit_cases, seed), reordered(seconds, other_seed), shards
    )

    assert first == second
    assert ci_shards.digest(unit_cases, first) == ci_shards.digest(
        reordered(unit_cases, seed), reordered(second, other_seed)
    )
    items = _items(unit_cases, seed)
    for shard in range(1, shards + 1):
        keep, _ = ci_shards.select(items, shard, shards, seconds)
        assert keep == [first[unit] == shard for unit in items]


@given(_UNIT_CASES, _SECONDS, _SHARDS)
def test_no_shard_is_heavier_than_another_by_more_than_one_unit(
    unit_cases, seconds, shards
):
    unit_weights = ci_shards.weights(unit_cases, seconds)
    assignment = ci_shards.assign(unit_cases, seconds, shards)
    loads = Counter({shard: 0 for shard in range(1, shards + 1)})
    for unit, shard in assignment.items():
        loads[shard] += unit_weights[unit]

    assert all(weight >= 1 for weight in unit_weights.values())
    assert max(loads.values()) - min(loads.values()) <= max(
        unit_weights.values(), default=0
    )


@given(
    _UNIT_CASES.filter(bool),
    _SECONDS,
    st.integers(min_value=2, max_value=12),
    st.data(),
)
def test_gate_rejects_a_missing_or_disagreeing_shard(unit_cases, seconds, shards, data):
    reports = _reports(_items(unit_cases, 0), shards, seconds)
    victim = data.draw(st.integers(min_value=0, max_value=shards - 1))
    name, units, kept_units, cases, kept_cases, collection = reports[victim].split(":")
    neighbour = reports[(victim + 1) % shards]

    def swapped(report: str) -> list[str]:
        return reports[:victim] + [report] + reports[victim + 1 :]

    other_collection = "0" * 64 if collection != "0" * 64 else "1" * 64
    broken = [
        swapped(""),
        swapped(f"{name}:0:0:0:0:"),
        # A dropped shard, and another shard's report in this one's place.
        reports[:-1],
        swapped(neighbour),
        swapped(f"{name}:{units}:{kept_units}:{cases}:{kept_cases}:{other_collection}"),
        swapped(
            f"{name}:{int(units) + 1}:{kept_units}:{int(cases) + 1}:{kept_cases}:{collection}"
        ),
    ]
    if int(kept_cases):
        broken.append(
            swapped(
                f"{name}:{units}:{kept_units}:{cases}:{int(kept_cases) - 1}:{collection}"
            )
        )
    if int(kept_units):
        broken.append(
            swapped(
                f"{name}:{units}:{int(kept_units) - 1}:{cases}:{kept_cases}:{collection}"
            )
        )

    assert ci_shards.check_reports(reports) == []
    for candidate in broken:
        assert ci_shards.check_reports(candidate) != []


def test_shards_holding_different_tables_cannot_pass_the_gate():
    # Each shard, with its own table, keeps test_a; test_b would run nowhere.
    items = ["tests/test_a.py::test_a", "tests/test_a.py::test_b"]
    table_one = {"tests/test_a.py::test_a": 9, "tests/test_a.py::test_b": 1}
    table_two = {"tests/test_a.py::test_a": 1, "tests/test_a.py::test_b": 9}
    keep_one, report_one = ci_shards.select(items, 1, 2, table_one)
    keep_two, report_two = ci_shards.select(items, 2, 2, table_two)

    assert keep_one == keep_two == [True, False]
    assert ci_shards.check_reports([report_one, report_two]) != []


def test_a_measured_function_weighs_its_own_seconds_and_the_rest_share_the_file():
    unit_cases = {
        "tests/test_a.py::test_slow": 2,
        "tests/test_a.py::test_one": 1,
        "tests/test_a.py::test_three": 3,
        "tests/test_b.py::test_unlisted_file": 5,
    }
    seconds = {"tests/test_a.py": 100, "tests/test_a.py::test_slow": 60}

    assert ci_shards.weights(unit_cases, seconds) == {
        "tests/test_a.py::test_slow": 60_000_000,
        "tests/test_a.py::test_one": 10_000_000,
        "tests/test_a.py::test_three": 30_000_000,
        "tests/test_b.py::test_unlisted_file": ci_shards.DEFAULT_FILE_MILLISECONDS
        * 1000,
    }
    # A function entry larger than its file's leaves the others the minimum, not less.
    stale = {"tests/test_a.py": 10, "tests/test_a.py::test_slow": 60}
    assert ci_shards.weights(unit_cases, stale)["tests/test_a.py::test_one"] == 1
    with pytest.raises(ValueError):
        ci_shards.weights({"tests/test_a.py::test_x": 0}, seconds)


def test_gate_rejects_no_shards_and_an_empty_suite():
    empty = ci_shards.digest({}, {})

    assert ci_shards.check_reports([]) == ["no shards"]
    assert ci_shards.check_reports([f"1/1:0:0:0:0:{empty}"]) == [
        "the suite collected no tests"
    ]
    assert ci_shards.check_reports([f"1/1:1:2:1:1:{empty}"]) != []


@pytest.mark.parametrize(
    "value,expected",
    [("1/1", (1, 1)), ("3/8", (3, 8)), ("8/8", (8, 8)), ("12/100", (12, 100))],
)
def test_shard_spec_parses(value, expected):
    assert ci_shards.parse_shard(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0/8",
        "9/8",
        "1/0",
        "3",
        "3/",
        "/8",
        " 3/8",
        "3/8 ",
        "3/8\n",
        "-1/8",
        "3/08",
        "03/8",
        "1/1000",
        "a/b",
    ],
)
def test_shard_spec_rejects_anything_else(value):
    with pytest.raises(ValueError):
        ci_shards.parse_shard(value)


@pytest.mark.parametrize(
    "nodeid,name,originalname,expected",
    [
        ("tests/test_a.py::test_x", "test_x", "test_x", "tests/test_a.py::test_x"),
        (
            "tests/test_a.py::test_x[1]",
            "test_x[1]",
            "test_x",
            "tests/test_a.py::test_x",
        ),
        (
            "tests/test_a.py::TestK::test_x[a::b[c]]",
            "test_x[a::b[c]]",
            "test_x",
            "tests/test_a.py::TestK::test_x",
        ),
        # A longer function name that only starts with another's is not its case.
        (
            "tests/test_a.py::test_x_more",
            "test_x_more",
            "test_x",
            "tests/test_a.py::test_x_more",
        ),
        ("tests/test_a.py::test_x", "test_x", None, "tests/test_a.py::test_x"),
        ("tests/readme.md::line 3", "line 3", None, "tests/readme.md::line 3"),
    ],
)
def test_unit_is_the_test_function(nodeid, name, originalname, expected):
    assert ci_shards.unit_of(nodeid, name, originalname) == expected


def test_this_sessions_real_tests_split_without_loss(request):
    items = request.session.items
    units = [
        ci_shards.unit_of(item.nodeid, item.name, getattr(item, "originalname", None))
        for item in items
    ]
    seconds = ci_shards.load_seconds()

    assert all(
        item.nodeid.startswith(unit) for item, unit in zip(items, units, strict=True)
    )
    for shards in (1, 2, 8):
        selections = [
            ci_shards.select(units, shard, shards, seconds)
            for shard in range(1, shards + 1)
        ]
        kept_by = [sum(keep[i] for keep, _ in selections) for i in range(len(units))]
        assert kept_by == [1] * len(units)
        assert ci_shards.check_reports([report for _, report in selections]) == []


# --- real pytest runs under this repository's conftest ------------------------------

_GENERATED_TESTS = """\
import os

import pytest


@pytest.mark.parametrize("n", range(4))
def test_many(n):
    pass


@pytest.mark.parametrize("n", "ab")
def test_pair(n):
    pass


def test_one():
    pass


def test_skipped():
    pytest.skip("not here")


class TestK:
    def test_method(self):
        pass

    @pytest.mark.parametrize("n", "xyz")
    def test_cases(self, n):
        pass


def test_environment_has_no_shard_variables():
    assert not [name for name in os.environ if name.startswith("AXIOM_ENCODE_TEST_SHARD")]


def test_fails_when_asked():
    assert not os.environ.get("GENERATED_TEST_FAILS")


def test_nests_a_pytest_run_when_asked(tmp_path):
    if os.environ.get("GENERATED_TEST_NESTS"):
        inner = tmp_path / "test_inner.py"
        inner.write_text("def test_inner():\\n    pass\\n")
        arguments = ["-q", "-p", "no:cacheprovider", "-p", "tests.conftest", str(inner)]
        assert pytest.main(arguments) == 0


def test_stops_the_session_when_asked():
    if os.environ.get("GENERATED_TEST_EXITS"):
        pytest.exit("stopping the session", returncode=0)
"""
# The generated file is outside the repository, so pytest prints its tests as "::name".
_RESULT_LINE = re.compile(r"^\S*?::(\S+) (PASSED|FAILED|SKIPPED|ERROR)\b", re.M)


_ASKS = ("GENERATED_TEST_FAILS", "GENERATED_TEST_NESTS", "GENERATED_TEST_EXITS")


def _run(tmp_path: Path, shard: str | None, *options: str, ask: str | None = None):
    """Run the generated tests under ``shard``: exit code, results by node, report, stderr."""
    tests = tmp_path / "test_generated.py"
    tests.write_text(_GENERATED_TESTS, encoding="utf-8")
    report = tmp_path / "report"
    report.write_text("", encoding="utf-8")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {ci_shards.SHARD_ENV, ci_shards.REPORT_ENV, *_ASKS}
    }
    environment[ci_shards.REPORT_ENV] = str(report)
    if shard is not None:
        environment[ci_shards.SHARD_ENV] = shard
    if ask is not None:
        assert ask in _ASKS
        environment[ask] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            # The generated file is outside tests/, so load the conftest by name.
            "-p",
            "tests.conftest",
            "--rootdir",
            str(ROOT),
            "-c",
            str(ROOT / "pyproject.toml"),
            *options,
            str(tests),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    results = dict(_RESULT_LINE.findall(result.stdout))
    return result.returncode, results, report.read_text(encoding="utf-8"), result.stderr


def _report_of(line: str, shard: int) -> str:
    name, _, report = line.strip().partition("=")
    assert name == f"shard_{shard}", line
    return report


def test_each_test_runs_in_exactly_one_shard_and_unsharded_runs_all(tmp_path):
    code, whole, no_report, _ = _run(tmp_path, None)
    shards = [_run(tmp_path, f"{shard}/3") for shard in (1, 2, 3)]

    assert code == 0 and no_report == ""
    assert len(whole) == 16 and whole["test_skipped"] == "SKIPPED"
    assert [code for code, _, _, _ in shards] == [0, 0, 0]
    assert sum(len(results) for _, results, _, _ in shards) == len(whole)
    merged = {}
    for _, results, _, _ in shards:
        merged.update(results)
    assert merged == whole
    reports = [
        _report_of(line, shard) for shard, (_, _, line, _) in enumerate(shards, 1)
    ]
    assert ci_shards.check_reports(reports) == []
    # A parametrized function's cases stay in one shard.
    assert sum(1 for _, results, _, _ in shards if "test_many[0]" in results) == 1
    owner = next(results for _, results, _, _ in shards if "test_many[0]" in results)
    assert {f"test_many[{n}]" for n in range(4)} <= owner.keys()


def test_a_shard_whose_tests_fail_publishes_no_report(tmp_path):
    shards = [
        _run(tmp_path, f"{shard}/2", ask="GENERATED_TEST_FAILS") for shard in (1, 2)
    ]
    failing = [
        index
        for index, (_, results, _, _) in enumerate(shards)
        if "FAILED" in results.values()
    ]

    assert len(failing) == 1
    code, _, line, _ = shards[failing[0]]
    other_code, _, other_line, _ = shards[1 - failing[0]]
    assert code == 1 and line == ""
    assert other_code == 0 and other_line != ""
    reports = ["", ""]
    reports[1 - failing[0]] = _report_of(other_line, 2 - failing[0])
    assert ci_shards.check_reports(reports) != []


def test_a_test_that_stops_the_session_leaves_no_report(tmp_path):
    # The shard's last kept test stops the session with exit status 0.
    code, results, line, stderr = _run(tmp_path, "1/1", ask="GENERATED_TEST_EXITS")

    assert code == 0
    assert "test_stops_the_session_when_asked" not in results
    assert line == ""
    assert "not publishing its report" in stderr


def test_a_nested_pytest_run_leaves_the_shard_count_alone(tmp_path):
    code, results, line, _ = _run(tmp_path, "1/1", ask="GENERATED_TEST_NESTS")

    assert code == 0
    assert results["test_nests_a_pytest_run_when_asked"] == "PASSED"
    assert line.count("\n") == 1
    assert ci_shards.check_reports([_report_of(line, 1)]) == []


def test_a_shard_that_only_collects_publishes_no_report(tmp_path):
    code, results, line, stderr = _run(tmp_path, "1/2", "--collect-only")

    assert code == 0 and results == {}
    assert line == ""
    assert "not publishing its report" in stderr


def test_pytest_filters_apply_before_the_split(tmp_path):
    _, whole, _, _ = _run(tmp_path, None, "-k", "many or pair or method")
    shards = [
        _run(tmp_path, f"{shard}/2", "-k", "many or pair or method") for shard in (1, 2)
    ]

    assert len(whole) == 7
    merged = {}
    for code, results, _, _ in shards:
        assert code == 0
        merged.update(results)
    assert merged == whole
    assert sum(len(results) for _, results, _, _ in shards) == len(whole)
    reports = [
        _report_of(line, shard) for shard, (_, _, line, _) in enumerate(shards, 1)
    ]
    assert ci_shards.check_reports(reports) == []
    # The reports count the filtered tests, which are the ones that ran.
    assert reports[0].split(":")[3] == "7"


def test_tests_do_not_see_the_shard_variables(tmp_path):
    code, results, line, _ = _run(tmp_path, "1/1")

    assert code == 0
    assert results["test_environment_has_no_shard_variables"] == "PASSED"
    assert ci_shards.check_reports([_report_of(line, 1)]) == []


def test_conftest_refuses_a_malformed_shard(tmp_path):
    code, results, line, stderr = _run(tmp_path, "4/3")

    assert code == 4 and results == {} and line == ""
    assert "past the last one" in stderr


def test_seconds_table_names_test_files_and_functions():
    table = ci_shards.load_seconds()

    assert table
    assert all(re.fullmatch(r"tests/[\w/]+\.py(::\S+)?", entry) for entry in table)
    # Every function entry sits beside its file's entry, which bounds it.
    assert all(entry.split("::", 1)[0] in table for entry in table)


def test_seconds_come_from_a_run_tests_log():
    log = "\n".join(
        [
            "test\tRun tests\t﻿2026-10-13T12:41:10.0000000Z ##[group]Run .venv/bin/pytest",
            "test\tRun tests\t2026-10-13T12:41:18.5000000Z ===== test session starts =====",
            "test\tRun tests\t2026-10-13T12:41:18.6000000Z collected 9 items / 4 deselected",
            "test\tRun tests\t2026-10-13T12:41:20.5000000Z tests/test_a.py ..s        [ 33%]",
            "test\tRun tests\t2026-10-13T12:41:50.5000000Z tests/test_b.py ....       [ 77%]",
            "test\tRun tests\t2026-10-13T12:42:00.5000000Z ..                         [100%]",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z ===== slowest 200 durations =====",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z 15.25s call     tests/test_b.py::test_slow[a b]",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z 14.50s call     tests/test_b.py::test_slow[c]",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z 0.25s setup    tests/test_b.py::test_slow[c]",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z 9.00s call     tests/test_b.py::TestK::test_quick",
            "test\tRun tests\t2026-10-13T12:42:00.6000000Z 25.00s call     tests/test_gone.py::test_not_run_here",
            "test\tRun tests\t2026-10-13T12:42:01.0000000Z ===== 8 passed, 1 skipped in 42s =====",
        ]
    )

    assert ci_shards.seconds_from_logs([log]) == {
        "tests/test_a.py": 2.0,
        "tests/test_b.py": 40.0,
        "tests/test_b.py::test_slow": 30.0,
    }
    twice = ci_shards.seconds_from_logs([log, log])
    assert twice["tests/test_b.py"] == 80.0
    assert twice["tests/test_b.py::test_slow"] == 60.0
    assert "tests/test_b.py::TestK::test_quick" not in twice


def test_command_line_checks_reports(capsys):
    unit_cases = {"tests/test_a.py::test_x": 2}
    collection = ci_shards.digest(unit_cases, ci_shards.assign(unit_cases, {}, 2))
    good = [f"1/2:1:1:2:2:{collection}", f"2/2:1:0:2:0:{collection}"]

    assert ci_shards.main(["check", *good]) == 0
    assert ci_shards.main(["check", good[0], ""]) == 1
    assert "shard 2 published no valid report" in capsys.readouterr().err
    assert ci_shards.main([]) == 2


# --- the workflow is wired to the functions above ---------------------------------

_GATE_NAME = "test (3.13)"


def _rendered_names(job_id: str, job: dict) -> set[str]:
    """Every check name a job can produce, over all its matrix combinations."""
    matrix = job.get("strategy", {}).get("matrix", {})
    names = set()
    for values in itertools.product(*matrix.values()):
        if "name" in job:
            name = job["name"]
            for key, value in zip(matrix, values, strict=True):
                name = name.replace(f"${{{{ matrix.{key} }}}}", str(value))
        elif values:
            # GitHub's default for a matrix job without a name.
            name = f"{job_id} ({', '.join(str(value) for value in values)})"
        else:
            name = job_id
        names.add(name)
    return names


def test_rendered_names_follow_github():
    matrix = {"strategy": {"matrix": {"python-version": ["3.13"], "os": ["a", "b"]}}}

    assert _rendered_names("lint", {}) == {"lint"}
    assert _rendered_names("test", matrix) == {"test (3.13, a)", "test (3.13, b)"}
    assert _rendered_names(
        "test", {**matrix, "name": "t ${{ matrix.os }} (${{ matrix.python-version }})"}
    ) == {"t a (3.13)", "t b (3.13)"}
    # What this workflow's serial test job was called, and the gate is called now.
    assert _rendered_names(
        "test", {"strategy": {"matrix": {"python-version": ["3.13"]}}}
    ) == {_GATE_NAME}


def test_workflow_runs_every_shard_and_reports_each():
    job = CI["jobs"]["test"]
    shards = job["strategy"]["matrix"]["shard"]
    (run_tests,) = [step for step in job["steps"] if step.get("id") == "tests"]

    assert shards == list(range(1, len(shards) + 1)) and len(shards) >= 2
    # No include or exclude that could drop a shard, and no way to pass on failure.
    assert job["strategy"] == {
        "fail-fast": False,
        "matrix": {"python-version": ["3.13"], "shard": shards},
    }
    assert job.keys() == {"name", "runs-on", "strategy", "outputs", "steps"}
    assert run_tests == {
        "name": "Run tests",
        "id": "tests",
        "env": {ci_shards.SHARD_ENV: f"${{{{ matrix.shard }}}}/{len(shards)}"},
        "run": (
            "set -e\n"
            f'export {ci_shards.REPORT_ENV}="$RUNNER_TEMP/shard-report"\n'
            '.venv/bin/pytest --durations=200 --durations-min=5 -m "not real_engine" tests/\n'
            "# Reached only when pytest exits 0, so a failure anywhere in the run,\n"
            "# including after the report was written, publishes nothing.\n"
            'cat "$RUNNER_TEMP/shard-report" >> "$GITHUB_OUTPUT"\n'
        ),
    }
    assert job["outputs"] == {
        f"shard_{shard}": f"${{{{ steps.tests.outputs.shard_{shard} }}}}"
        for shard in shards
    }
    # No other step may see the variable: a pytest run there would be cut down too.
    assert all("env" not in step for step in job["steps"] if step is not run_tests)
    assert "env" not in CI


def test_required_check_is_the_gate_over_all_shards():
    shards = CI["jobs"]["test"]["strategy"]["matrix"]["shard"]
    gate = CI["jobs"]["test-gate"]

    # Branch protection requires this exact name, and only the gate may carry it.
    assert gate["name"] == _GATE_NAME
    for key, job in CI["jobs"].items():
        if key != "test-gate":
            assert _GATE_NAME not in _rendered_names(key, job), key
    assert _rendered_names("test", CI["jobs"]["test"]) == {
        f"test shard {shard} of {len(shards)} (3.13)" for shard in shards
    }
    # A job whose needs failed is skipped, and a skipped required check passes.
    # Nothing here may let the gate pass after a failure either.
    assert gate.keys() == {"name", "if", "needs", "runs-on", "steps"}
    assert gate["if"] == "always()"
    assert gate["needs"] == "test"
    checkout, check = gate["steps"]
    assert checkout.keys() == {"uses"}
    assert check.keys() == {"name", "env", "run"}
    assert check["env"] == {
        "SHARDS_RESULT": "${{ needs.test.result }}",
        **{
            f"SHARD_{shard}": f"${{{{ needs.test.outputs.shard_{shard} }}}}"
            for shard in shards
        },
    }
    arguments = " ".join(f'"$SHARD_{shard}"' for shard in shards)
    assert " ".join(check["run"].replace("\\\n", " ").split()) == (
        'if [ "$SHARDS_RESULT" != "success" ]; then '
        "echo \"error: the test shards finished as '$SHARDS_RESULT'\" >&2 "
        "exit 1 fi "
        f"python3 -I tests/ci_shards.py check {arguments}"
    )
