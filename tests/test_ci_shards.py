"""The CI shards together run every test they select exactly once, or the gate job fails."""

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
from tests import conftest as tests_conftest

Collected = ci_shards.Collected
ROOT = Path(__file__).resolve().parents[1]
CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))

_NAMES = st.text(alphabet="abcxyz_", min_size=1, max_size=4)
_FILES = _NAMES.map(lambda name: f"tests/test_{name}.py")
_FUNCTIONS = st.tuples(_FILES, _NAMES).map(lambda pair: f"{pair[0]}::test_{pair[1]}")
# Each function's case count, and whether its tests share a module- or
# class-scoped fixture.
_FUNCTION_CASES = st.dictionaries(
    _FUNCTIONS,
    st.tuples(
        st.integers(min_value=1, max_value=40),
        st.sampled_from(["", "module", "class"]),
    ),
    max_size=30,
)
# Seconds for files and, as the refreshed table has for slow tests, for
# functions; often past SPLIT_FUNCTION_SECONDS, so functions split.
_SECONDS = st.dictionaries(
    st.one_of(_FILES, _FUNCTIONS),
    st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
    max_size=14,
)
_SHARDS = st.integers(min_value=1, max_value=12)


def _tests(function_cases, seed: int = 0) -> list[ci_shards.Collected]:
    """The collected tests, in a shuffled collection order."""
    tests = []
    for function, (cases, scope) in function_cases.items():
        if scope == "class":
            # A method: class-scoped fixtures are per class only inside one.
            function = function.replace("::", "::TestShared::", 1)
        for case in range(cases):
            nodeid = function if cases == 1 else f"{function}[{case}]"
            tests.append(Collected(nodeid, function, scope or None))
    random.Random(seed).shuffle(tests)
    return tests


def _reports(tests, shards, seconds) -> list[str]:
    return [
        ci_shards.select(tests, shard, shards, seconds)[1]
        for shard in range(1, shards + 1)
    ]


@given(_FUNCTION_CASES, _SECONDS, _SHARDS)
def test_every_unit_gets_exactly_one_shard(function_cases, seconds, shards):
    tests = _tests(function_cases)
    assignment = ci_shards.assign(tests, seconds, shards)

    assert assignment.keys() == set(ci_shards.units_for(tests, seconds))
    assert all(1 <= shard <= shards for shard in assignment.values())


@given(_FUNCTION_CASES, _SECONDS, _SHARDS, st.integers())
def test_shards_keep_every_test_once_and_their_reports_pass_the_gate(
    function_cases, seconds, shards, seed
):
    items = _tests(function_cases, seed)
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


@given(_FUNCTION_CASES, _SECONDS, _SHARDS, st.integers(), st.integers())
def test_assignment_ignores_collection_and_table_order(
    function_cases, seconds, shards, seed, other_seed
):
    def reordered(mapping, by):
        keys = list(mapping)
        random.Random(by).shuffle(keys)
        return {key: mapping[key] for key in keys}

    items = _tests(function_cases, seed)
    first = ci_shards.assign(_tests(function_cases, other_seed), seconds, shards)
    second = ci_shards.assign(items, reordered(seconds, other_seed), shards)
    units = ci_shards.units_for(items, seconds)
    memberships = Counter(zip(units, (test.function for test in items)))

    assert first == second
    assert ci_shards.digest(memberships, first) == ci_shards.digest(
        reordered(memberships, seed), reordered(second, other_seed)
    )
    for shard in range(1, shards + 1):
        keep, _ = ci_shards.select(items, shard, shards, seconds)
        assert keep == [first[unit] == shard for unit in units]


@given(_FUNCTION_CASES, _SECONDS, _SHARDS)
def test_no_shard_is_heavier_than_another_by_more_than_one_unit(
    function_cases, seconds, shards
):
    tests = _tests(function_cases)
    unit_weights = ci_shards.weights(tests, seconds)
    assignment = ci_shards.assign(tests, seconds, shards)
    loads = Counter({shard: 0 for shard in range(1, shards + 1)})
    for unit, shard in assignment.items():
        loads[shard] += unit_weights[unit]

    assert all(weight >= 1 for weight in unit_weights.values())
    assert max(loads.values()) - min(loads.values()) <= max(
        unit_weights.values(), default=0
    )


@given(
    _FUNCTION_CASES.filter(bool),
    _SECONDS,
    st.integers(min_value=2, max_value=12),
    st.data(),
)
def test_gate_rejects_a_missing_or_disagreeing_shard(
    function_cases, seconds, shards, data
):
    reports = _reports(_tests(function_cases), shards, seconds)
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


_METHODS = st.tuples(_FILES, _NAMES).map(
    lambda pair: f"{pair[0]}::TestK::test_{pair[1]}"
)
_METHOD_SECONDS = st.dictionaries(
    st.one_of(_FILES, _METHODS),
    st.floats(min_value=0, max_value=5000, allow_nan=False, allow_infinity=False),
    max_size=14,
)
_SCOPES = st.sampled_from([None, "module", "class"])


@given(
    st.dictionaries(
        _METHODS,
        st.tuples(st.integers(min_value=1, max_value=6), _SCOPES, _SCOPES),
        max_size=12,
    ),
    _METHOD_SECONDS,
    _METHOD_SECONDS,
    st.lists(st.integers(min_value=0, max_value=1), min_size=1, max_size=6),
)
def test_shards_that_see_different_scopes_or_tables_pass_only_if_each_test_runs_once(
    methods, seconds_one, seconds_two, picks
):
    # Two views of one collection: the same tests, but each function's fixture
    # scope and the table may differ between them. Each shard holds one view.
    # Whenever the gate passes, every test must have run exactly once.
    nodeids = [
        (method if cases == 1 else f"{method}[{case}]", method)
        for method, (cases, _, _) in sorted(methods.items())
        for case in range(cases)
    ]
    views = [
        (
            [
                Collected(nodeid, method, methods[method][1 + view])
                for nodeid, method in nodeids
            ],
            (seconds_one, seconds_two)[view],
        )
        for view in (0, 1)
    ]
    kept = Counter()
    reports = []
    for shard, pick in enumerate(picks, start=1):
        tests, seconds = views[pick]
        keep, line = ci_shards.select(tests, shard, len(picks), seconds)
        reports.append(line)
        kept.update(
            test.nodeid
            for test, kept_here in zip(tests, keep, strict=True)
            if kept_here
        )

    if ci_shards.check_reports(reports) == []:
        assert kept == Counter(nodeid for nodeid, _ in nodeids)


def test_shards_holding_different_tables_cannot_pass_the_gate():
    # Each shard, with its own table, keeps test_a; test_b would run nowhere.
    items = _tests(
        {"tests/test_a.py::test_a": (1, ""), "tests/test_a.py::test_b": (1, "")}
    )
    items.sort()
    table_one = {"tests/test_a.py::test_a": 9, "tests/test_a.py::test_b": 1}
    table_two = {"tests/test_a.py::test_a": 1, "tests/test_a.py::test_b": 9}
    keep_one, report_one = ci_shards.select(items, 1, 2, table_one)
    keep_two, report_two = ci_shards.select(items, 2, 2, table_two)

    assert keep_one == keep_two == [True, False]
    assert ci_shards.check_reports([report_one, report_two]) != []


def test_shards_holding_tables_that_split_differently_cannot_pass_the_gate():
    # One table splits test_slow into its cases, the other keeps it whole.
    items = sorted(_tests({"tests/test_a.py::test_slow": (2, "")}))
    split = {"tests/test_a.py::test_slow": ci_shards.SPLIT_FUNCTION_SECONDS}
    whole = {"tests/test_a.py::test_slow": ci_shards.SPLIT_FUNCTION_SECONDS - 1}
    reports = [
        ci_shards.select(items, 1, 2, split)[1],
        ci_shards.select(items, 2, 2, whole)[1],
    ]

    assert ci_shards.check_reports(reports) != []


def test_a_measured_function_weighs_its_own_seconds_and_the_rest_share_the_file():
    tests = _tests(
        {
            "tests/test_a.py::test_slow": (2, ""),
            "tests/test_a.py::test_one": (1, ""),
            "tests/test_a.py::test_three": (3, ""),
            "tests/test_b.py::test_unlisted_file": (5, ""),
        }
    )
    seconds = {"tests/test_a.py": 100, "tests/test_a.py::test_slow": 60}

    assert ci_shards.weights(tests, seconds) == {
        "tests/test_a.py::test_slow": 60_000_000,
        "tests/test_a.py::test_one": 10_000_000,
        "tests/test_a.py::test_three": 30_000_000,
        "tests/test_b.py::test_unlisted_file": ci_shards.DEFAULT_FILE_MILLISECONDS
        * 1000,
    }
    # A function entry larger than its file's leaves the others the minimum, not less.
    stale = {"tests/test_a.py": 10, "tests/test_a.py::test_slow": 60}
    assert ci_shards.weights(tests, stale)["tests/test_a.py::test_one"] == 1


def test_a_slow_parametrized_function_splits_into_equal_cases():
    slow = "tests/test_a.py::test_slow"
    tests = _tests(
        {
            slow: (3, ""),
            "tests/test_a.py::test_one": (1, ""),
            # As slow, but one case: there is nothing to split.
            "tests/test_a.py::test_single": (1, ""),
        }
    )
    seconds = {
        "tests/test_a.py": 1000,
        slow: 900,
        "tests/test_a.py::test_single": 90,
    }

    assert ci_shards.weights(tests, seconds) == {
        f"{slow}[0]": 300_000_000,
        f"{slow}[1]": 300_000_000,
        f"{slow}[2]": 300_000_000,
        "tests/test_a.py::test_one": 10_000_000,
        "tests/test_a.py::test_single": 90_000_000,
    }
    # Three shards: each case of the slow function gets its own.
    assignment = ci_shards.assign(tests, seconds, 3)
    assert sorted(assignment[f"{slow}[{case}]"] for case in range(3)) == [1, 2, 3]
    # Below the threshold the function stays one unit.
    below = {**seconds, slow: ci_shards.SPLIT_FUNCTION_SECONDS - 0.1}
    assert {
        ci_shards.units_for(tests, below)[index]
        for index, test in enumerate(tests)
        if test.function == slow
    } == {slow}


def test_tests_sharing_a_scoped_fixture_stay_one_unit():
    module = "tests/test_a.py"
    tests = [
        # Both use a module-scoped fixture, so they form the module's unit,
        # even the slow parametrized one that would otherwise split.
        Collected(f"{module}::test_slow[0]", f"{module}::test_slow", "module"),
        Collected(f"{module}::test_slow[1]", f"{module}::test_slow", "module"),
        Collected(f"{module}::test_other", f"{module}::test_other", "module"),
        Collected(f"{module}::test_alone", f"{module}::test_alone"),
        Collected(f"{module}::TestK::test_x", f"{module}::TestK::test_x", "class"),
        Collected(f"{module}::TestK::test_y", f"{module}::TestK::test_y", "class"),
        Collected(
            "tests/test_b.py::TestK::test_x", "tests/test_b.py::TestK::test_x", "class"
        ),
        Collected(
            "tests/test_b.py::TestK::test_y", "tests/test_b.py::TestK::test_y", "class"
        ),
    ]
    seconds = {
        module: 1000,
        f"{module}::test_slow": 600,
        f"{module}::test_other": 100,
        "tests/test_b.py": 10,
    }

    # In a module with a module-scoped fixture, class-fixture tests join the
    # module's unit too: TestK's methods might also use the module's fixture.
    # Elsewhere a class's fixture users form the class's unit.
    assert ci_shards.units_for(tests, seconds) == [
        module,
        module,
        module,
        f"{module}::test_alone",
        module,
        module,
        "tests/test_b.py::TestK",
        "tests/test_b.py::TestK",
    ]
    # The module's unit weighs its functions' entries and its share of the
    # rest of the file's time; test_alone has the other share.
    assert ci_shards.weights(tests, seconds) == {
        module: 900_000_000,
        f"{module}::test_alone": 100_000_000,
        "tests/test_b.py::TestK": 10_000_000,
    }


def test_a_method_using_both_scopes_stays_with_one_using_only_its_class():
    # test_a uses the module's fixture and its class's; test_b only the class's.
    module = "tests/test_a.py"
    tests = [
        Collected(f"{module}::TestK::test_a", f"{module}::TestK::test_a", "module"),
        Collected(f"{module}::TestK::test_b", f"{module}::TestK::test_b", "class"),
    ]

    assert ci_shards.units_for(tests, {}) == [module, module]
    for shard in (1, 2):
        keep, _ = ci_shards.select(tests, shard, 2, {})
        assert keep in ([True, True], [False, False])


@given(_FUNCTION_CASES, _SECONDS)
def test_units_follow_the_fixture_scope_then_the_split_rule(function_cases, seconds):
    tests = _tests(function_cases)
    units = ci_shards.units_for(tests, seconds)
    cases = Counter(test.function for test in tests)

    fixture_modules = {
        test.nodeid.split("::")[0] for test in tests if test.scope == "module"
    }
    for test, unit in zip(tests, units, strict=True):
        module = test.nodeid.split("::")[0]
        if test.scope == "module" or (
            test.scope == "class" and module in fixture_modules
        ):
            assert unit == module
        elif test.scope == "class":
            assert unit == test.function.rsplit("::", 1)[0]
        elif (
            cases[test.function] > 1
            and seconds.get(test.function, 0) >= ci_shards.SPLIT_FUNCTION_SECONDS
        ):
            assert unit == test.nodeid
        else:
            assert unit == test.function
    # A unit weighs its tests' shares of their functions' time, so splitting and
    # grouping move time between units but, up to rounding, keep the total.
    whole = ci_shards.weights(
        [Collected(test.function, test.function) for test in tests], seconds
    )
    total = sum(ci_shards.weights(tests, seconds).values())
    assert abs(total - sum(whole.values())) <= 2 * len(tests)


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
    # Every test this session collected, also when it runs as one shard.
    tests = request.config.stash[tests_conftest.COLLECTED_TESTS]
    seconds = ci_shards.load_seconds()

    assert any(test.nodeid == request.node.nodeid for test in tests)
    assert all(test.nodeid.startswith(test.function) for test in tests)
    assert {test.scope for test in tests} <= {None, "module", "class"}
    # pytest builds one fixture closure per function: its cases share a scope.
    scopes = {(test.function, test.scope) for test in tests}
    assert len(scopes) == len({test.function for test in tests})
    for shards in (1, 2, 8):
        selections = [
            ci_shards.select(tests, shard, shards, seconds)
            for shard in range(1, shards + 1)
        ]
        kept_by = [sum(keep[i] for keep, _ in selections) for i in range(len(tests))]
        assert kept_by == [1] * len(tests)
        assert ci_shards.check_reports([report for _, report in selections]) == []


# Functions the committed table splits into their cases. A case is known by its
# node id, so each one here was checked to take literal parametrize ids (the same
# in every shard) and to build its own inputs, independent of the other cases.
# A refreshed table that splits another function fails the test below until the
# function is checked and listed.
REVIEWED_SPLITS = {
    "tests/test_source_path_evidence.py"
    "::test_actual_analyzer_bad_evidence_retains_monetary_findings",
    "tests/test_worksheet_operation_evidence.py"
    "::test_actual_analyzer_failed_certificate_never_hides_annual_or_other_owner",
    "tests/test_worksheet_transfer_evidence.py"
    "::test_actual_analyzer_failed_bundle_cannot_grant_transfer_or_annual",
}


def test_the_table_splits_only_reviewed_functions(request):
    tests = request.config.stash[tests_conftest.COLLECTED_TESTS]
    units = ci_shards.units_for(tests, ci_shards.load_seconds())
    split = {
        test.function
        for test, unit in zip(tests, units, strict=True)
        if unit != test.function and unit == test.nodeid
    }

    assert split <= REVIEWED_SPLITS


class _Scope:
    def __init__(self, scope: str) -> None:
        self.scope = scope


class _Item:
    """What ``collected`` reads from a pytest item."""

    def __init__(self, nodeid, name, originalname, scopes, cls=None) -> None:
        self.nodeid = nodeid
        self.name = name
        self.originalname = originalname
        self.cls = cls

        class _Info:
            name2fixturedefs = {
                f"fixture_{index}": tuple(_Scope(scope) for scope in chain)
                for index, chain in enumerate(scopes)
            }

        self._fixtureinfo = _Info()


@pytest.mark.parametrize(
    "scopes,cls,scope",
    [
        ([], None, None),
        ([("function",), ("session",), ("package",)], None, None),
        ([("module",)], None, "module"),
        # Every definition of a name counts: an override may use the one
        # it overrides, by requesting its own name.
        ([("function", "module")], None, "module"),
        ([("module", "function")], None, "module"),
        ([("class", "function")], object, "class"),
        ([("class",)], object, "class"),
        ([("class",), ("module",)], object, "module"),
        # Outside a class, pytest sets a class-scoped fixture up per test.
        ([("class",)], None, None),
    ],
)
def test_collected_reads_the_fixture_scopes(scopes, cls, scope):
    nodeid = (
        "tests/test_a.py::TestK::test_x[1]" if cls else "tests/test_a.py::test_x[1]"
    )
    item = _Item(nodeid, "test_x[1]", "test_x", scopes, cls)

    assert ci_shards.collected(item) == Collected(
        nodeid, nodeid.removesuffix("[1]"), scope
    )


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


@pytest.fixture(scope="module")
def module_setup():
    return object()


def test_uses_the_module_fixture(module_setup):
    pass


def test_also_uses_the_module_fixture(module_setup):
    pass


@pytest.fixture
def cached_value(request):
    return request.param


# The parametrize call caches the function-scoped fixture per module.
@pytest.mark.parametrize("cached_value", [1, 2], indirect=True, scope="module")
def test_parametrized_at_module_scope(cached_value):
    pass


@pytest.fixture(scope="module")
def wrapped():
    return []


class TestWrap:
    # Overrides the module's fixture and uses it.
    @pytest.fixture
    def wrapped(self, wrapped):
        return wrapped

    def test_wrapped(self, wrapped):
        pass


class TestShared:
    @pytest.fixture(scope="class")
    def class_setup(self):
        return object()

    def test_first(self, class_setup):
        pass

    def test_second(self, class_setup):
        pass


def test_environment_has_no_shard_variables():
    assert not [name for name in os.environ if name.startswith("AXIOM_ENCODE_TEST_SHARD")]


def test_fails_when_asked():
    assert not os.environ.get("GENERATED_TEST_FAILS")


def test_nests_a_pytest_run_when_asked(tmp_path):
    if os.environ.get("GENERATED_TEST_NESTS"):
        inner = tmp_path / "test_inner.py"
        inner.write_text("def test_inner():\\n    pass\\n")
        arguments = [
            "-q", "-o", "addopts=", "-p", "no:cacheprovider", "-p", "tests.conftest", str(inner)
        ]
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
    assert len(whole) == 23 and whole["test_skipped"] == "SKIPPED"
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
    # A parametrized function's cases stay in one shard, and so do the tests
    # that share a module- or class-scoped fixture. The module has a
    # module-scoped fixture, so TestShared's users join the module's unit, and
    # so do the function parametrized at module scope and TestWrap's test,
    # whose fixture wraps a module-scoped one: 16 functions, 11 units.
    for together in (
        {f"test_many[{n}]" for n in range(4)},
        {
            "test_uses_the_module_fixture",
            "test_also_uses_the_module_fixture",
            "test_parametrized_at_module_scope[1]",
            "test_parametrized_at_module_scope[2]",
            "TestWrap::test_wrapped",
        },
        {"TestShared::test_first", "TestShared::test_second"},
    ):
        owners = [results for _, results, _, _ in shards if together & results.keys()]
        assert len(owners) == 1 and together <= owners[0].keys()
    assert {report.split(":")[1] for report in reports} == {"11"}


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
    # The shard's last kept test stops the session with exit status 0. If it
    # were not last, the tests after it would be missing from the count anyway
    # and the test could not tell a count taken at start from one taken at end.
    _, whole, _, _ = _run(tmp_path, "1/1")
    assert list(whole)[-1] == "test_stops_the_session_when_asked"

    code, results, line, stderr = _run(tmp_path, "1/1", ask="GENERATED_TEST_EXITS")

    assert code == 0
    assert list(results) == list(whole)[:-1]
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
    tests = _tests({"tests/test_a.py::test_x": (2, "")})
    memberships = {("tests/test_a.py::test_x", "tests/test_a.py::test_x"): 2}
    collection = ci_shards.digest(memberships, ci_shards.assign(tests, {}, 2))
    good = [f"1/2:1:1:2:2:{collection}", f"2/2:1:0:2:0:{collection}"]

    assert ci_shards.main(["check", *good]) == 0
    assert ci_shards.main(["check", good[0], ""]) == 1
    assert "shard 2 published no valid report" in capsys.readouterr().err
    assert ci_shards.main([]) == 2


# --- the workflow is wired to the functions above ---------------------------------

_GATE_NAME = "test (3.13)"
# The conditions on ci.yml's jobs (tests/test_workflow_toolchain_pins.py pins them too).
_FULL_RUN = "github.event_name != 'pull_request' || !github.event.pull_request.draft"
_DRAFT_RUN = "github.event_name == 'pull_request' && github.event.pull_request.draft"
_DRAFT_MESSAGE = (
    "draft PR: the full suite runs when the PR is marked ready "
    "(or dispatch CI on the branch: gh workflow run CI --ref <branch>)"
)


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
    assert job.keys() == {
        "name",
        "if",
        "runs-on",
        "concurrency",
        "strategy",
        "outputs",
        "steps",
    }
    assert job["if"] == _FULL_RUN
    assert run_tests == {
        "name": "Run tests",
        "id": "tests",
        "env": {
            ci_shards.SHARD_ENV: f"${{{{ matrix.shard }}}}/{len(shards)}",
            # Only a bump pull request runs the version guard test (#2048).
            "AXIOM_ENCODE_REQUIRE_VERSION_BUMP": (
                "${{ startsWith(github.head_ref, 'automation/encoder-version-') "
                "&& '1' || '' }}"
            ),
        },
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
    # real-engine and locked-builds too, so a passing gate means both passed in
    # its run. locked-builds (scripts/check_locked_builds.py) is one result for
    # both of its Python legs.
    assert gate["needs"] == ["test", "real-engine", "locked-builds"]
    locked_builds = CI["jobs"]["locked-builds"]
    assert _rendered_names("locked-builds", locked_builds) == {
        "locked builds (3.13)",
        "locked builds (3.14)",
    }
    # Its result means only what the gate reads if a failure cannot pass and the
    # job's last act is the check.
    assert "continue-on-error" not in locked_builds
    assert all("continue-on-error" not in step for step in locked_builds["steps"])
    assert locked_builds["steps"][-1] == {
        "name": "Build every locked source against the locked build backends",
        "run": ".venv/bin/python scripts/check_locked_builds.py",
    }
    checkout, check = gate["steps"]
    assert checkout.keys() == {"uses"}
    assert check.keys() == {"name", "env", "run"}
    assert check["env"] == {
        "SHARDS_RESULT": "${{ needs.test.result }}",
        "REAL_ENGINE_RESULT": "${{ needs.real-engine.result }}",
        "LOCKED_BUILDS_RESULT": "${{ needs.locked-builds.result }}",
        "DRAFT_PULL_REQUEST": f"${{{{ {_DRAFT_RUN} }}}}",
        **{
            f"SHARD_{shard}": f"${{{{ needs.test.outputs.shard_{shard} }}}}"
            for shard in shards
        },
    }
    arguments = " ".join(f'"$SHARD_{shard}"' for shard in shards)
    # DRAFT_PULL_REQUEST only adds a message: the gate fails either way.
    assert " ".join(check["run"].replace("\\\n", " ").split()) == (
        'if [ "$SHARDS_RESULT" != "success" ] || '
        '[ "$REAL_ENGINE_RESULT" != "success" ] || '
        '[ "$LOCKED_BUILDS_RESULT" != "success" ]; then '
        'if [ "$DRAFT_PULL_REQUEST" = "true" ]; then '
        f'echo "::error::{_DRAFT_MESSAGE}" '
        "fi "
        "echo \"error: the test shards finished as '$SHARDS_RESULT', "
        "real-engine as '$REAL_ENGINE_RESULT' and the locked builds as "
        "'$LOCKED_BUILDS_RESULT'\" >&2 "
        "exit 1 fi "
        f"python3 -I tests/ci_shards.py check {arguments}"
    )


def test_concurrency_groups_sit_on_the_jobs_that_do_the_work():
    # The gate runs even in a cancelled run, so a workflow-level group would keep
    # a pull request's new run waiting until the old gate got a runner. Each job
    # that does the work keys its own group instead, and each test shard and
    # locked-builds leg keys one per matrix leg, so the legs of one run never
    # cancel each other. On any event but a pull request, the key ends in the
    # run's own id: nothing cancels it.
    suffix = (
        "${{ github.event_name == 'pull_request' && format('pr-{0}', "
        "github.event.pull_request.number) || format('run-{0}', github.run_id) }}"
    )
    legs = "${{ matrix.python-version }}-${{ matrix.shard }}"
    assert "concurrency" not in CI
    assert "concurrency" not in CI["jobs"]["test-gate"]
    assert all(
        "concurrency" in job for key, job in CI["jobs"].items() if key != "test-gate"
    )
    assert {
        key: job["concurrency"]
        for key, job in CI["jobs"].items()
        if "concurrency" in job
    } == {
        key: {
            "group": f"${{{{ github.workflow }}}}-{group}-{suffix}",
            "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
        }
        for key, group in {
            "test": f"test-{legs}",
            "real-engine": "real-engine",
            "locked-builds": "locked-builds-${{ matrix.python-version }}",
            "quick-tests": "quick-tests",
            "lint": "lint",
        }.items()
    }


# --- draft pull requests run a light CI and cannot pass the gate --------------------

# Events a run of ci.yml can come from, as the expressions below see them.
_EVENTS = {
    "push to main": {
        "github.event_name": "push",
        "github.event.pull_request.draft": None,
    },
    "workflow_dispatch": {
        "github.event_name": "workflow_dispatch",
        "github.event.pull_request.draft": None,
    },
    "merge queue": {
        "github.event_name": "merge_group",
        "github.event.pull_request.draft": None,
    },
    # Opened, pushed to, reopened or turned back into a draft.
    "draft pull request": {
        "github.event_name": "pull_request",
        "github.event.pull_request.draft": True,
    },
    "pull request ready for review": {
        "github.event_name": "pull_request",
        "github.event.pull_request.draft": False,
    },
}


def _evaluate(expression: str, context: dict) -> object:
    """A GitHub expression made of ``==``, ``!=``, ``!``, ``&&``, ``||``, context
    names and string literals, evaluated as GitHub does for these operators."""
    tokens = re.findall(r"'[^']*'|[A-Za-z_][\w.]*|==|!=|&&|\|\||!|\S", expression)
    python = []
    for token in tokens:
        if token in context:
            python.append(f"context[{token!r}]")
        elif token.startswith("'"):
            python.append(repr(token[1:-1]))
        else:
            python.append(
                {"&&": "and", "||": "or", "!": "not", "==": "==", "!=": "!="}[token]
            )
    return eval(" ".join(python), {"context": context})  # noqa: S307 - pinned expressions


def test_pull_requests_run_on_ready_for_review_and_any_branch_can_dispatch():
    triggers = CI[True]  # YAML 1.1 reads the key `on` as True.

    assert triggers == {
        "push": {"branches": ["main"]},
        # No `labeled`: labelling a pull request would start a run whose
        # cancels and skipped gate replace the real one's checks.
        "pull_request": {
            "branches": ["main"],
            "types": [
                "opened",
                "synchronize",
                "reopened",
                "ready_for_review",
                # A pull request turned back into a draft fails its gate again.
                "converted_to_draft",
            ],
        },
        "workflow_dispatch": None,
        # The required checks must report on the merge queue's branch.
        "merge_group": {"types": ["checks_requested"]},
    }


def test_drafts_run_lint_and_quick_tests_and_everything_else_runs_in_full():
    jobs = CI["jobs"]
    assert (
        jobs["test"]["if"]
        == jobs["real-engine"]["if"]
        == jobs["locked-builds"]["if"]
        == _FULL_RUN
    )
    assert jobs["quick-tests"]["if"] == _DRAFT_RUN
    assert "if" not in jobs["lint"]
    # The gate runs on every event; nothing about drafts may skip it.
    assert jobs["test-gate"]["if"] == "always()"
    assert jobs.keys() == {
        "test",
        "test-gate",
        "real-engine",
        "locked-builds",
        "quick-tests",
        "lint",
    }

    runs = {
        event: {
            "shards, real-engine and locked builds": bool(
                _evaluate(_FULL_RUN, context)
            ),
            "quick tests": bool(_evaluate(_DRAFT_RUN, context)),
            "gate says draft": _evaluate(_DRAFT_RUN, context) is True,
        }
        for event, context in _EVENTS.items()
    }
    full = {
        "shards, real-engine and locked builds": True,
        "quick tests": False,
        "gate says draft": False,
    }
    assert runs == {
        "push to main": full,
        "workflow_dispatch": full,
        "merge queue": full,
        "draft pull request": {
            "shards, real-engine and locked builds": False,
            "quick tests": True,
            "gate says draft": True,
        },
        "pull request ready for review": full,
    }


def _gate_script() -> str:
    (_, check) = CI["jobs"]["test-gate"]["steps"]
    return check["run"]


def _run_gate(
    shards_result,
    real_engine_result,
    draft,
    reports=("",) * 8,
    locked_builds_result="success",
):
    environment = {
        "PATH": os.environ["PATH"],
        "SHARDS_RESULT": shards_result,
        "REAL_ENGINE_RESULT": real_engine_result,
        "LOCKED_BUILDS_RESULT": locked_builds_result,
        "DRAFT_PULL_REQUEST": draft,
        **{f"SHARD_{shard}": report for shard, report in enumerate(reports, 1)},
    }
    return subprocess.run(
        ["bash", "-e", "-c", _gate_script()],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _agreeing_reports() -> list[str]:
    tests = _tests(
        {"tests/test_a.py::test_x": (3, ""), "tests/test_b.py::test_y": (9, "")}
    )
    return [ci_shards.select(tests, shard, 8, {})[1] for shard in range(1, 9)]


@pytest.mark.parametrize(
    "shards_result,real_engine_result,locked_builds_result,draft,says_draft",
    [
        # A draft's skipped jobs fail the gate, with the draft message.
        ("skipped", "skipped", "skipped", "true", True),
        ("skipped", "skipped", "skipped", "false", False),
        ("failure", "success", "success", "false", False),
        ("cancelled", "cancelled", "cancelled", "false", False),
        # Shards that passed do not carry a failed or skipped real-engine.
        ("success", "failure", "success", "false", False),
        ("success", "skipped", "success", "false", False),
        ("success", "cancelled", "success", "false", False),
        # Nor a failed, cancelled, skipped or missing locked-builds result.
        ("success", "success", "failure", "false", False),
        ("success", "success", "cancelled", "false", False),
        ("success", "success", "skipped", "false", False),
        ("success", "success", "", "false", False),
        # Even successful jobs without shard reports fail it.
        ("success", "success", "success", "true", False),
    ],
)
def test_gate_fails_unless_every_job_passed_and_reported(
    shards_result, real_engine_result, locked_builds_result, draft, says_draft
):
    # Agreeing reports, so only the job results can fail the gate, except in
    # the last case, which has none.
    reports = (
        ("",) * 8
        if (shards_result, real_engine_result, locked_builds_result) == ("success",) * 3
        else _agreeing_reports()
    )

    result = _run_gate(
        shards_result,
        real_engine_result,
        draft,
        reports,
        locked_builds_result=locked_builds_result,
    )

    assert result.returncode == 1
    assert (f"::error::{_DRAFT_MESSAGE}" in result.stdout) is says_draft
    if (shards_result, real_engine_result, locked_builds_result) != ("success",) * 3:
        assert f"the locked builds as '{locked_builds_result}'" in result.stderr, (
            result.stderr
        )


def test_gate_passes_when_every_job_passed_and_the_reports_agree():
    result = _run_gate("success", "success", "false", _agreeing_reports())

    assert result.returncode == 0, result.stderr


def _quick_tests_script() -> str:
    (step,) = [
        step
        for step in CI["jobs"]["quick-tests"]["steps"]
        if step.get("name") == "Run the test files the pull request changes"
    ]
    return step["run"]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _pull_request_merge(tmp_path: Path, changes: dict[str, str | None]) -> Path:
    """A checkout like actions/checkout's of a pull request: its merge commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    for name in ("tests/test_kept.py", "tests/test_gone.py", "src/x.py"):
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text("base\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    for name, text in changes.items():
        path = repo / name
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "feature")
    _git(repo, "checkout", "-q", "main")
    (repo / "README").write_text("main moved on\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge", "feature")
    return repo


def _run_quick_tests(repo: Path, pytest_status: int = 0):
    """Run the quick-tests step with a stand-in pytest: exit code, output, its argv."""
    calls = repo.parent / "pytest-calls"
    pytest_path = repo / ".venv" / "bin" / "pytest"
    pytest_path.parent.mkdir(parents=True)
    pytest_path.write_text(
        f'#!/bin/sh\nprintf "%s\\0" "$@" > "{calls}"\nexit {pytest_status}\n',
        encoding="utf-8",
    )
    pytest_path.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", _quick_tests_script()],
        cwd=repo,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )
    argv = (
        calls.read_text(encoding="utf-8").split("\0")[:-1] if calls.exists() else None
    )
    return result.returncode, result.stdout + result.stderr, argv


def test_quick_tests_run_only_the_test_files_the_pull_request_changes(tmp_path):
    repo = _pull_request_merge(
        tmp_path,
        {
            "tests/test_kept.py": "changed\n",
            "tests/test_new.py": "new\n",
            # git quotes this name in --name-only output unless given -z.
            "tests/test_\u00e9t\u00e9.py": "unicode\n",
            # Without -z, or read by line, git splits this name in two.
            "tests/test_one\ntwo.py": "newline\n",
            # test_*.py matches an empty suffix too.
            "tests/test_.py": "empty\n",
            "tests/test_gone.py": None,
            "tests/fixtures/test_data.py": "fixture\n",
            "tests/helper.py": "helper\n",
            "src/x.py": "changed\n",
        },
    )

    code, _, argv = _run_quick_tests(repo)

    assert code == 0
    assert argv == [
        "-o",
        "addopts=",
        "-m",
        "not real_engine",
        "tests/test_.py",
        "tests/test_kept.py",
        "tests/test_new.py",
        "tests/test_one\ntwo.py",
        "tests/test_\u00e9t\u00e9.py",
    ]


@pytest.mark.parametrize("status,expected", [(0, 0), (1, 1), (2, 2), (5, 0)])
def test_quick_tests_pass_on_pytest_success_or_no_selected_test(
    tmp_path, status, expected
):
    repo = _pull_request_merge(tmp_path, {"tests/test_kept.py": "changed\n"})

    code, _, argv = _run_quick_tests(repo, pytest_status=status)

    assert code == expected and argv is not None


def test_quick_tests_pass_with_a_message_when_no_test_file_changed(tmp_path):
    repo = _pull_request_merge(tmp_path, {"src/x.py": "changed\n"})

    code, output, argv = _run_quick_tests(repo)

    assert code == 0 and argv is None
    assert "changes no tests/test_*.py file" in output


def test_quick_tests_fail_outside_a_pull_request_merge_commit(tmp_path):
    repo = _pull_request_merge(tmp_path, {"tests/test_kept.py": "changed\n"})
    _git(repo, "checkout", "-q", "feature")

    code, _, argv = _run_quick_tests(repo)

    assert code != 0 and argv is None
