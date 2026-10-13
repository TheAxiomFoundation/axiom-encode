"""The CI shards together run every test exactly once, or the gate job fails."""

from __future__ import annotations

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
_FILE_SECONDS = st.dictionaries(
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


@given(_UNIT_CASES, _FILE_SECONDS, _SHARDS)
def test_every_unit_gets_exactly_one_shard(unit_cases, file_seconds, shards):
    assignment = ci_shards.assign(unit_cases, file_seconds, shards)

    assert assignment.keys() == unit_cases.keys()
    assert all(1 <= shard <= shards for shard in assignment.values())


@given(_UNIT_CASES, _FILE_SECONDS, _SHARDS, st.integers())
def test_shards_keep_every_test_once_and_their_reports_pass_the_gate(
    unit_cases, file_seconds, shards, seed
):
    items = _items(unit_cases, seed)
    selections = [
        ci_shards.select(items, shard, shards, file_seconds)
        for shard in range(1, shards + 1)
    ]

    kept_by = [
        sum(keep[index] for keep, _ in selections) for index in range(len(items))
    ]
    assert kept_by == [1] * len(items)
    problems = ci_shards.check_reports([report for _, report in selections])
    assert problems == ([] if items else ["the suite collected no tests"])


@given(_UNIT_CASES, _FILE_SECONDS, _SHARDS, st.integers(), st.integers())
def test_assignment_ignores_collection_and_table_order(
    unit_cases, file_seconds, shards, seed, other_seed
):
    def reordered(mapping, by):
        keys = list(mapping)
        random.Random(by).shuffle(keys)
        return {key: mapping[key] for key in keys}

    first = ci_shards.assign(unit_cases, file_seconds, shards)
    second = ci_shards.assign(
        reordered(unit_cases, seed), reordered(file_seconds, other_seed), shards
    )

    assert first == second
    assert ci_shards.digest(unit_cases) == ci_shards.digest(reordered(unit_cases, seed))
    items = _items(unit_cases, seed)
    for shard in range(1, shards + 1):
        keep, _ = ci_shards.select(items, shard, shards, file_seconds)
        assert keep == [first[unit] == shard for unit in items]


@given(_UNIT_CASES, _FILE_SECONDS, _SHARDS)
def test_no_shard_is_heavier_than_another_by_more_than_one_unit(
    unit_cases, file_seconds, shards
):
    unit_weights = ci_shards.weights(unit_cases, file_seconds)
    assignment = ci_shards.assign(unit_cases, file_seconds, shards)
    loads = Counter({shard: 0 for shard in range(1, shards + 1)})
    for unit, shard in assignment.items():
        loads[shard] += unit_weights[unit]

    assert all(weight >= 1 for weight in unit_weights.values())
    assert max(loads.values()) - min(loads.values()) <= max(
        unit_weights.values(), default=0
    )


@given(
    _UNIT_CASES.filter(bool),
    _FILE_SECONDS,
    st.integers(min_value=2, max_value=12),
    st.data(),
)
def test_gate_rejects_a_missing_or_disagreeing_shard(
    unit_cases, file_seconds, shards, data
):
    items = _items(unit_cases, 0)
    reports = [
        ci_shards.select(items, shard, shards, file_seconds)[1]
        for shard in range(1, shards + 1)
    ]
    victim = data.draw(st.integers(min_value=0, max_value=shards - 1))
    name, units, kept_units, cases, kept_cases, collection = reports[victim].split(":")
    neighbour = reports[(victim + 1) % shards]

    def swapped(report: str) -> list[str]:
        return reports[:victim] + [report] + reports[victim + 1 :]

    other_collection = ci_shards.digest(
        {**unit_cases, "tests/test_new.py::test_new": 1}
    )
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
    empty = ci_shards.digest({})

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
    file_seconds = ci_shards.load_seconds()

    assert all(
        item.nodeid.startswith(unit) for item, unit in zip(items, units, strict=True)
    )
    for shards in (1, 2, 8):
        selections = [
            ci_shards.select(units, shard, shards, file_seconds)
            for shard in range(1, shards + 1)
        ]
        kept_by = [sum(keep[i] for keep, _ in selections) for i in range(len(units))]
        assert kept_by == [1] * len(units)
        assert ci_shards.check_reports([report for _, report in selections]) == []


def _collect(tmp_path: Path, shard: str | None) -> tuple[set[str], str]:
    """Node ids pytest selects for this file under ``shard``, and the report line."""
    report = tmp_path / f"report-{(shard or 'whole').replace('/', '-')}"
    report.write_text("", encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {ci_shards.SHARD_ENV, ci_shards.REPORT_ENV}
    }
    environment[ci_shards.REPORT_ENV] = str(report)
    if shard is not None:
        environment[ci_shards.SHARD_ENV] = shard
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            __file__,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in {0, 5}, result.stdout + result.stderr
    selected = {line for line in result.stdout.splitlines() if "::" in line}
    return selected, report.read_text(encoding="utf-8")


def test_conftest_runs_each_test_in_one_shard_and_all_of_them_unsharded(tmp_path):
    whole, no_report = _collect(tmp_path, None)
    shards = [_collect(tmp_path, f"{shard}/3") for shard in (1, 2, 3)]

    assert len(whole) > 20
    assert no_report == ""
    assert sum(len(selected) for selected, _ in shards) == len(whole)
    assert set().union(*(selected for selected, _ in shards)) == whole
    lines = [line.strip() for _, line in shards]
    assert [line.split("=")[0] for line in lines] == ["shard_1", "shard_2", "shard_3"]
    assert ci_shards.check_reports([line.split("=", 1)[1] for line in lines]) == []


def test_conftest_refuses_a_malformed_shard(tmp_path):
    environment = {**os.environ, ci_shards.SHARD_ENV: "4/3"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            __file__,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 4, result.stdout + result.stderr
    assert "past the last one" in result.stderr


def test_seconds_table_covers_real_test_files_only():
    table = ci_shards.load_seconds()

    assert table
    assert all((ROOT / entry.split("::", 1)[0]).is_file() for entry in table), (
        "drop entries for deleted files"
    )


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
    collection = ci_shards.digest({"tests/test_a.py::test_x": 2})
    good = [f"1/2:1:1:2:2:{collection}", f"2/2:1:0:2:0:{collection}"]

    assert ci_shards.main(["check", *good]) == 0
    assert ci_shards.main(["check", good[0], ""]) == 1
    assert "shard 2 published no valid report" in capsys.readouterr().err
    assert ci_shards.main([]) == 2


# --- the workflow is wired to the functions above ---------------------------------


def test_workflow_runs_every_shard_and_reports_each():
    job = CI["jobs"]["test"]
    shards = job["strategy"]["matrix"]["shard"]
    run_tests = [step for step in job["steps"] if step.get("id") == "tests"]

    assert shards == list(range(1, len(shards) + 1)) and len(shards) >= 2
    assert job["strategy"]["fail-fast"] is False
    assert len(run_tests) == 1
    assert run_tests[0]["env"] == {
        ci_shards.SHARD_ENV: f"${{{{ matrix.shard }}}}/{len(shards)}"
    }
    assert f'export {ci_shards.REPORT_ENV}="$GITHUB_OUTPUT"' in run_tests[0]["run"]
    assert re.search(r"^\.venv/bin/pytest .*\btests/", run_tests[0]["run"], re.M)
    assert job["outputs"] == {
        f"shard_{shard}": f"${{{{ steps.tests.outputs.shard_{shard} }}}}"
        for shard in shards
    }
    # No other step may see the variable: a pytest run there would be cut down too.
    assert "env" not in job
    assert all("env" not in step for step in job["steps"] if step.get("id") != "tests")
    assert ci_shards.SHARD_ENV not in (CI.get("env") or {})


def test_required_check_is_the_gate_over_all_shards():
    shards = CI["jobs"]["test"]["strategy"]["matrix"]["shard"]
    gate = CI["jobs"]["test-gate"]
    names = [job.get("name", key) for key, job in CI["jobs"].items()]
    (check,) = [step for step in gate["steps"] if "run" in step]

    # Branch protection requires this exact name, and only the gate may carry it.
    assert gate["name"] == "test (3.13)"
    assert names.count("test (3.13)") == 1
    assert "test (3.13)" not in CI["jobs"]["test"]["name"].replace(
        "${{ matrix.shard }}", "1"
    )
    # A job whose needs failed is skipped, and a skipped required check passes.
    assert gate["if"] == "always()"
    assert gate["needs"] == "test"
    assert check["env"] == {
        "SHARDS_RESULT": "${{ needs.test.result }}",
        **{
            f"SHARD_{shard}": f"${{{{ needs.test.outputs.shard_{shard} }}}}"
            for shard in shards
        },
    }
    script = " ".join(check["run"].replace("\\\n", " ").split())
    assert 'if [ "$SHARDS_RESULT" != "success" ]; then' in script
    assert "exit 1 fi" in script
    arguments = " ".join(f'"$SHARD_{shard}"' for shard in shards)
    assert script.endswith(f"python3 -I tests/ci_shards.py check {arguments}")
