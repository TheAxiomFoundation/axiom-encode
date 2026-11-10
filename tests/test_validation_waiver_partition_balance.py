"""Invariants of the cost-balanced waiver audit partition and its progress output.

Every matrix shard computes the same owners, so each waiver path is audited by
exactly one shard. The properties here run for arbitrary ledgers, key sets and
cost tables:

* exactly once: the shards' selections are disjoint and cover the ledger;
* determinism: owners depend only on the path set, key set and table;
* legacy equivalence: no table (or equal costs, no baselines) gives the
  round-robin the partition used before cost tables (a differential test);
* the list-scheduling bound and Graham's 4/3 LPT bound on the largest load;
* the cost table round-trips exactly, and the audit's own timing line parses
  back to the milliseconds it printed.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode import validation_waiver_partition as partition
from tests.test_validation_waiver_audit_cli import (
    _audit_args,
    _entry,
    _metadata,
    _waiver_set,
)
from tests.test_validation_waiver_audit_parallel import _FakePool, _release_stub

_SEGMENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_.", min_size=1)
_PATH = st.lists(_SEGMENT, min_size=1, max_size=4).map(
    lambda parts: (
        "/".join(part if part not in {".", ".."} else f"{part}x" for part in parts)
        + ".yaml"
    )
)
_PATHS = st.lists(_PATH, unique=True, max_size=40)
_KEYS = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=8),
    unique=True,
    min_size=1,
    max_size=9,
)
_MILLISECONDS = st.integers(min_value=0, max_value=10**7)


def _legacy_round_robin(paths, keys):
    """The partition before cost tables: sorted path i -> sorted key i mod n."""
    ordered_keys = sorted(keys)
    return {
        path: ordered_keys[index % len(ordered_keys)]
        for index, path in enumerate(sorted(paths))
    }


def _costs(module_milliseconds, baseline_milliseconds=None, source=None):
    rendered = partition.render_waiver_audit_costs(
        {path: [value] for path, value in module_milliseconds.items()},
        baselines=baseline_milliseconds,
        source=source,
    )
    return partition.parse_waiver_audit_costs(rendered)


@st.composite
def _cases(draw, *, with_table=True):
    paths = draw(_PATHS)
    keys = draw(_KEYS)
    if not with_table or not draw(st.booleans()):
        return paths, keys, None
    listed = draw(st.lists(st.sampled_from(paths), unique=True)) if paths else []
    extra = draw(st.lists(_PATH, unique=True, max_size=3))
    modules = {path: draw(_MILLISECONDS) for path in [*listed, *extra]}
    if not modules:
        modules = {"unused/module.yaml": draw(_MILLISECONDS)}
    baselines = {
        key: draw(_MILLISECONDS)
        for key in draw(st.lists(st.sampled_from(keys), unique=True))
    }
    return paths, keys, _costs(modules, baselines)


def _weight(costs, path):
    return 1 if costs is None else costs.cost(path)


# ---------------------------------------------------------------- partition


@settings(max_examples=300, deadline=None, database=None)
@given(paths=_PATHS, keys=_KEYS)
@example(paths=[f"us/{index:03d}.yaml" for index in range(17)], keys=["us", "us-ak"])
def test_without_a_table_owners_equal_the_legacy_round_robin(paths, keys):
    assert partition.assign_waiver_owners(paths, keys) == _legacy_round_robin(
        paths, keys
    )


@settings(max_examples=200, deadline=None, database=None)
@given(paths=_PATHS, keys=_KEYS, value=_MILLISECONDS, data=st.data())
def test_equal_costs_without_baselines_equal_the_legacy_round_robin(
    paths, keys, value, data
):
    listed = data.draw(st.lists(st.sampled_from(paths), unique=True)) if paths else []
    costs = _costs({path: value for path in listed} or {"x.yaml": value})
    assert partition.assign_waiver_owners(paths, keys, costs) == _legacy_round_robin(
        paths, keys
    )


@settings(max_examples=300, deadline=None, database=None)
@given(case=_cases(), data=st.data())
def test_every_path_has_exactly_one_owner_and_input_order_is_irrelevant(case, data):
    paths, keys, costs = case
    owners = partition.assign_waiver_owners(paths, keys, costs)
    assert list(owners) == sorted(paths)
    assert set(owners.values()) <= set(keys)
    shuffled_paths = data.draw(st.permutations(paths))
    shuffled_keys = data.draw(st.permutations(keys))
    assert partition.assign_waiver_owners(shuffled_paths, shuffled_keys, costs) == (
        owners
    )
    # A second parse of the same bytes is the same table.
    if costs is not None:
        again = partition.parse_waiver_audit_costs(
            partition.render_waiver_audit_costs(
                {path: [value] for path, value in costs.module_milliseconds.items()},
                baselines=dict(costs.baseline_milliseconds),
            )
        )
        assert partition.assign_waiver_owners(paths, keys, again) == owners


@settings(max_examples=300, deadline=None, database=None)
@given(case=_cases())
def test_cli_partitions_are_disjoint_and_cover_the_ledger(case):
    paths, keys, costs = case
    classified = [(path, "active") for path in sorted(paths)]
    keys_json = json.dumps(list(reversed(keys)))
    selections = {
        key: cli._validation_waiver_audit_partition(
            classified, partition_key=key, partition_keys_json=keys_json, costs=costs
        )
        for key in keys
    }
    flattened = [item for selected, _, _ in selections.values() for item in selected]
    assert sorted(flattened) == classified
    assert len(flattened) == len(set(flattened))
    for key, (selected, index, count) in selections.items():
        assert (index, count) == (sorted(keys).index(key), len(keys))
        # Selections keep the ledger's order.
        assert selected == sorted(selected)


@settings(max_examples=300, deadline=None, database=None)
@given(case=_cases())
def test_largest_load_respects_the_list_scheduling_bound(case):
    """n * max load <= max(n * largest baseline, W + B + (n - 1) * largest cost).

    The last module placed on the most loaded key went to the least loaded
    key, whose load was then at most the average of everything placed so far.
    """
    paths, keys, costs = case
    owners = partition.assign_waiver_owners(paths, keys, costs)
    loads = partition.partition_loads(owners, keys, costs)
    n = len(keys)
    weights = [_weight(costs, path) for path in paths]
    baselines = [0 if costs is None else costs.baseline(key) for key in keys]
    bound = max(
        n * max(baselines),
        sum(weights) + sum(baselines) + (n - 1) * max(weights, default=0),
    )
    assert n * max(loads.values()) <= bound
    assert sum(loads.values()) == sum(weights) + sum(baselines)


def _optimal_largest_load(weights, n):
    best = None
    for assignment in itertools.product(range(n), repeat=len(weights)):
        loads = [0] * n
        for weight, key in zip(weights, assignment, strict=True):
            loads[key] += weight
        best = max(loads) if best is None else min(best, max(loads))
    return best or 0


@settings(max_examples=200, deadline=None, database=None)
@given(
    weights=st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=7),
    n=st.integers(min_value=1, max_value=3),
)
@example(weights=[3, 3, 2, 2, 2], n=2)  # LPT 7 vs optimal 6: the bound is tight
def test_lpt_is_within_grahams_bound_of_the_optimum(weights, n):
    paths = [f"us/m{index}.yaml" for index in range(len(weights))]
    keys = [f"k{index}" for index in range(n)]
    costs = _costs(dict(zip(paths, weights, strict=True)))
    loads = partition.partition_loads(
        partition.assign_waiver_owners(paths, keys, costs), keys, costs
    )
    optimal = _optimal_largest_load([costs.cost(path) for path in paths], n)
    # LPT <= (4/3 - 1/(3n)) * OPT  <=>  3n * LPT <= (4n - 1) * OPT
    assert 3 * n * max(loads.values()) <= (4 * n - 1) * optimal


def test_baselines_keep_long_modules_off_the_busiest_shard():
    """The observed hazard: plain LPT gives the longest module to key 0 ('us'),
    the shard with the most validation work of its own."""
    paths = ["us-vt/long.yaml", "us/a.yaml", "us/b.yaml", "us/c.yaml"]
    keys = ["us", "us-ak", "us-al"]
    plain = _costs({"us-vt/long.yaml": 50_000, "us/a.yaml": 10, "us/b.yaml": 10})
    assert partition.assign_waiver_owners(paths, keys, plain)["us-vt/long.yaml"] == (
        "us"
    )
    weighted = _costs(
        {"us-vt/long.yaml": 50_000, "us/a.yaml": 10, "us/b.yaml": 10},
        {"us": 200_000},
    )
    owners = partition.assign_waiver_owners(paths, keys, weighted)
    assert "us" not in owners.values()
    assert owners["us-vt/long.yaml"] == "us-ak"


@pytest.mark.parametrize(
    ("paths", "keys", "message"),
    [
        (["a.yaml", "a.yaml"], ["k"], "unique"),
        (["a.yaml"], [], "at least one key"),
        (["a.yaml"], ["k", "k"], "keys must be unique"),
    ],
)
def test_assignment_rejects_ambiguous_inputs(paths, keys, message):
    with pytest.raises(ValueError, match=message):
        partition.assign_waiver_owners(paths, keys)


@settings(max_examples=200, deadline=None, database=None)
@given(case=_cases())
def test_audit_order_is_a_longest_first_permutation(case):
    paths, _keys, costs = case
    order = partition.audit_order(reversed(paths), costs)
    assert sorted(order) == sorted(paths)
    if costs is None:
        assert order == sorted(paths)
    else:
        assert order == sorted(paths, key=lambda path: (-costs.cost(path), path))


# ---------------------------------------------------------------- cost table


@settings(max_examples=200, deadline=None, database=None)
@given(
    observations=st.dictionaries(
        _PATH, st.lists(_MILLISECONDS, min_size=1, max_size=4), max_size=20
    ).filter(bool),
    baselines=st.dictionaries(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=8),
        _MILLISECONDS,
        max_size=6,
    ),
    source=st.one_of(st.none(), st.text(alphabet="abc /:0123456789", min_size=1)),
)
def test_cost_table_round_trips_exactly(observations, baselines, source):
    rendered = partition.render_waiver_audit_costs(
        observations, baselines=baselines, source=source
    )
    costs = partition.parse_waiver_audit_costs(rendered)
    assert dict(costs.module_milliseconds) == {
        path: sorted(values)[(len(values) - 1) // 2]
        for path, values in observations.items()
    }
    assert dict(costs.baseline_milliseconds) == baselines
    assert costs.source == source
    # Rendering is canonical: the parsed table renders to the same bytes.
    assert (
        partition.render_waiver_audit_costs(
            {path: [value] for path, value in costs.module_milliseconds.items()},
            baselines=dict(costs.baseline_milliseconds),
            source=costs.source,
        )
        == rendered
    )


def _table(**fields):
    payload = {
        "schema": partition.COST_TABLE_SCHEMA,
        "module_milliseconds": {"us/a.yaml": 5},
    }
    payload.update(fields)
    return json.dumps(payload).encode()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\xff", "UTF-8"),
        (b"[]", "JSON object"),
        (b"{", "not valid JSON"),
        (_table(schema="v0"), "schema must be"),
        (_table(extra=1), "unknown fields: extra"),
        (_table(module_milliseconds=[]), "must be a JSON object"),
        (_table(module_milliseconds={"us/a.yaml": 1.5}), "integer milliseconds"),
        (_table(module_milliseconds={"us/a.yaml": True}), "integer milliseconds"),
        (_table(module_milliseconds={"us/a.yaml": -1}), "between 0 and"),
        (_table(module_milliseconds={"us/a.yaml": 10**10 + 1}), "between 0 and"),
        (_table(module_milliseconds={"/us/a.yaml": 1}), "canonical"),
        (_table(module_milliseconds={"us/../a.yaml": 1}), "canonical"),
        (_table(module_milliseconds={"us\\a.yaml": 1}), "canonical"),
        (_table(module_milliseconds={"us/a\u200b.yaml": 1}), "control or format"),
        (_table(partition_baseline_milliseconds={" us": 1}), "surrounding"),
        (_table(partition_baseline_milliseconds={"us": 2.0}), "integer"),
        (_table(source=""), "source must be"),
        (_table(source="a\nb"), "source must be"),
        (b'{"schema": 1, "schema": 2}', "duplicate JSON key"),
        (
            b'{"schema": "' + partition.COST_TABLE_SCHEMA.encode() + b'", '
            b'"module_milliseconds": {"us/a.yaml": NaN}}',
            "must not contain NaN",
        ),
    ],
)
def test_cost_table_parser_rejects_ambiguous_tables(raw, message):
    with pytest.raises(partition.WaiverAuditCostTableError, match=re.escape(message)):
        partition.parse_waiver_audit_costs(raw)


def test_defaults_and_digest():
    raw = _table(
        module_milliseconds={"us/a.yaml": 0, "us/b.yaml": 30, "us/c.yaml": 40},
        partition_baseline_milliseconds={"us": 7},
    )
    costs = partition.parse_waiver_audit_costs(raw)
    assert costs.sha256 == __import__("hashlib").sha256(raw).hexdigest()
    assert costs.default_milliseconds == 30  # median_low
    assert costs.cost("us/missing.yaml") == 30
    assert costs.cost("us/a.yaml") == 1  # no module weighs nothing
    assert (costs.baseline("us"), costs.baseline("us-ak")) == (7, 0)
    empty = partition.parse_waiver_audit_costs(_table(module_milliseconds={}))
    assert empty.cost("us/x.yaml") == partition.DEFAULT_MODULE_MILLISECONDS


def test_load_reads_only_a_contained_regular_file(tmp_path):
    root = tmp_path / "rulespec-us"
    (root / ".axiom").mkdir(parents=True)
    assert partition.load_waiver_audit_costs(root) is None
    table = root / partition.COST_TABLE_PATH
    table.write_bytes(_table())
    loaded = partition.load_waiver_audit_costs(root)
    assert loaded is not None and dict(loaded.module_milliseconds) == {"us/a.yaml": 5}

    table.write_bytes(b"{}")
    with pytest.raises(partition.WaiverAuditCostTableError, match=r"\.axiom/"):
        partition.load_waiver_audit_costs(root)

    table.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(_table())
    table.symlink_to(outside)
    with pytest.raises(partition.WaiverAuditCostTableError):
        partition.load_waiver_audit_costs(root)

    table.unlink()
    table.mkdir()
    with pytest.raises(partition.WaiverAuditCostTableError):
        partition.load_waiver_audit_costs(root)

    table.rmdir()
    with open(table, "wb") as handle:
        handle.truncate(partition.COST_TABLE_MAX_BYTES + 1)
    with pytest.raises(partition.WaiverAuditCostTableError, match="safety limit"):
        partition.load_waiver_audit_costs(root)


def test_load_refuses_a_symlinked_axiom_directory(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / Path(partition.COST_TABLE_PATH).name).write_bytes(_table())
    (root / ".axiom").symlink_to(elsewhere)
    with pytest.raises(partition.WaiverAuditCostTableError):
        partition.load_waiver_audit_costs(root)


# ---------------------------------------------------------------- timing lines


@settings(max_examples=200, deadline=None, database=None)
@given(path=_PATH, milliseconds=st.integers(min_value=0, max_value=10**9))
def test_the_audits_own_timing_line_parses_back_exactly(path, milliseconds):
    """Differential: the writer in cli.py and the reader here agree."""
    ticks = iter([1000.0, 1000.0 + milliseconds / 1000])
    stderr = io.StringIO()
    with (
        patch.object(cli.time, "perf_counter", lambda: next(ticks)),
        redirect_stderr(stderr),
        cli._validation_waiver_module_timing(path, enabled=True),
    ):
        pass
    line = stderr.getvalue().rstrip("\n")
    for prefix in (
        "",
        "2026-11-04T10:02:03.1234567Z ",
        "\ufeff2026-11-04T10:02:03.1234567Z ",
        "validate / validate (us-ms)\tEnforce validation waiver ratchet\t"
        "2026-11-04T10:02:03.1234567Z ",
    ):
        assert partition.timing_observations([prefix + line + "\n"]) == {
            path: [milliseconds]
        }


def test_timing_observations_skip_progress_lines_and_take_every_run():
    lines = [
        "validation-waivers audit plan: 3 module(s) on 4 worker(s), in path order",
        "validation-waivers audit start: us/a.yaml",
        "validation-waivers audit heartbeat: 0 of 3 module(s) finished after 60s; "
        "1 running: us/a.yaml 60s",
        "validation-waivers audit recheck start: us/a.yaml",
        "validation-waivers audit: us/a.yaml 12.345s",
        "validation-waivers audit: us/with space.yaml 0.001s\r",
        "validation-waivers audit: us/a.yaml 2.000s",
        "validation-waivers audit: us/a.yaml 2.0s",  # not the audit's format
        "noise validation-waivers audit: us/b.yaml 1.000s trailing",
    ]
    assert partition.timing_observations(lines) == {
        "us/a.yaml": [12345, 2000],
        "us/with space.yaml": [1],
    }
    with pytest.raises(partition.WaiverAuditCostTableError, match="canonical"):
        partition.timing_observations(["validation-waivers audit: /abs.yaml 1.000s"])


def _job(key, *, job=(0, 600), step=(100, 400), conclusion="success", name=None):
    def stamp(seconds):
        return f"2026-11-04T10:{seconds // 60:02d}:{seconds % 60:02d}Z"

    steps = [
        {"name": "Set up job", "conclusion": "success"},
        {
            "name": partition.DEFAULT_AUDIT_STEP_NAME,
            "conclusion": conclusion,
            "started_at": stamp(step[0]),
            "completed_at": stamp(step[1]),
        },
    ]
    return {
        "name": name or f"validate / validate ({key})",
        "started_at": stamp(job[0]),
        "completed_at": stamp(job[1]),
        "steps": steps,
    }


def test_partition_baselines_from_jobs():
    page = {
        "jobs": [
            _job("us", job=(0, 3000), step=(1000, 1500)),
            _job("us-ak", job=(0, 600), step=(100, 400)),
            _job("us-al", conclusion="skipped"),
            {"name": "shards", "steps": []},
            _job("x", name="lint"),
        ]
    }
    expected = {"us": (3000 - 500) * 4 * 1000, "us-ak": (600 - 300) * 4 * 1000}
    assert partition.partition_baselines_from_jobs(page, workers=4) == expected
    # `gh api --paginate --slurp` wraps pages in an array.
    assert (
        partition.partition_baselines_from_jobs(
            [{"jobs": page["jobs"][:1]}, {"jobs": page["jobs"][1:]}], workers=4
        )
        == expected
    )
    with pytest.raises(ValueError, match="more than once"):
        partition.partition_baselines_from_jobs(
            {"jobs": [_job("us"), _job("us")]}, workers=4
        )
    with pytest.raises(ValueError, match="positive integer"):
        partition.partition_baselines_from_jobs(page, workers=0)


def test_audit_costs_command_writes_a_loadable_table(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text(
        "\ufeff2026-11-04T10:00:00.0000000Z validation-waivers audit start: us/a.yaml\n"
        "2026-11-04T10:00:09.0000000Z validation-waivers audit: us/a.yaml 9.000s\n"
        "validation-waivers audit: us-ak/b.yaml 0.250s\n"
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [_job("us", job=(0, 900), step=(300, 600))]}))
    root = tmp_path / "rulespec-us"
    (root / ".axiom").mkdir(parents=True)
    output = root / partition.COST_TABLE_PATH
    args = SimpleNamespace(
        logs=[str(log)],
        jobs_json=jobs,
        workers=2,
        audit_step=partition.DEFAULT_AUDIT_STEP_NAME,
        source="rulespec-us run 1",
        output=output,
        json=True,
    )
    assert cli._cmd_validation_waivers_audit_costs(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["modules"] == 2 and summary["partition_baselines"] == 1
    costs = partition.load_waiver_audit_costs(root)
    assert costs is not None and costs.sha256 == summary["sha256"]
    assert dict(costs.module_milliseconds) == {"us-ak/b.yaml": 250, "us/a.yaml": 9000}
    assert dict(costs.baseline_milliseconds) == {"us": 600 * 2 * 1000}
    assert costs.source == "rulespec-us run 1"
    assert not list(output.parent.glob(".*.tmp"))

    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n")
    args.logs, args.jobs_json = [str(empty)], None
    with pytest.raises(ValueError, match="no 'validation-waivers audit"):
        cli._cmd_validation_waivers_audit_costs(args)


def test_main_registers_the_audit_costs_subcommand(tmp_path):
    with (
        patch(
            "sys.argv",
            [
                "axiom-encode",
                "validation-waivers",
                "audit-costs",
                "--log",
                "a.log",
                "--log",
                "-",
                "--output",
                str(tmp_path / "costs.json"),
            ],
        ),
        patch.object(cli, "cmd_validation_waivers") as command,
    ):
        cli.main()
    args = command.call_args.args[0]
    assert args.validation_waivers_command == "audit-costs"
    assert args.logs == ["a.log", "-"]
    assert (args.workers, args.jobs_json, args.source) == (4, None, None)
    assert args.audit_step == partition.DEFAULT_AUDIT_STEP_NAME


# ---------------------------------------------------------------- progress


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_progress_heartbeat_names_running_modules_and_their_ages(capsys):
    clock = _Clock()
    progress = partition.WaiverAuditProgress(3, clock=clock)
    progress.start("us/b.yaml")
    clock.now += 30
    progress.start("us/a.yaml")
    clock.now += 45
    assert progress.heartbeat_line() == (
        "validation-waivers audit heartbeat: 0 of 3 module(s) finished after 75s; "
        "2 running: us/b.yaml 75s; us/a.yaml 45s"
    )
    progress.finish("us/b.yaml")
    progress.finish("us/b.yaml")  # a second finish of one start counts once
    progress.heartbeat()
    recheck = partition.WaiverAuditProgress(
        1, prefix=partition.RECHECK_LINE_PREFIX, clock=clock
    )
    recheck.heartbeat()
    assert capsys.readouterr().err.splitlines() == [
        "validation-waivers audit start: us/b.yaml",
        "validation-waivers audit start: us/a.yaml",
        "validation-waivers audit heartbeat: 1 of 3 module(s) finished after 75s; "
        "1 running: us/a.yaml 45s",
        "validation-waivers audit recheck heartbeat: 0 of 1 module(s) finished "
        "after 0s; 0 running",
    ]


def test_heartbeat_thread_beats_until_the_block_exits():
    beats = []
    progress = SimpleNamespace(heartbeat=lambda: beats.append(1))
    with partition.heartbeat_thread(progress, interval=0.001):
        deadline = __import__("time").monotonic() + 10
        while len(beats) < 3 and __import__("time").monotonic() < deadline:
            pass
    stopped = len(beats)
    assert stopped >= 3
    __import__("time").sleep(0.05)
    assert len(beats) == stopped


def _run(paths, *, costs=None, workers=2, wait=None):
    _FakePool.submitted = []
    _FakePool.path_of = {}
    for path in paths:
        _FakePool.chunk_results[path] = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
        ]
    stderr = io.StringIO()
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(os.environ, {cli._WAIVER_AUDIT_WORKERS_ENV: str(workers)})
        )
        stack.enter_context(
            patch.object(
                cli, "load_rulespec_local_corpus_release", return_value=_release_stub()
            )
        )
        stack.enter_context(
            patch.object(cli.concurrent.futures, "ProcessPoolExecutor", _FakePool)
        )
        if wait is not None:
            stack.enter_context(patch.object(cli, "_wait_for_waiver_tasks", wait))
        stack.enter_context(redirect_stderr(stderr))
        rows = cli._fingerprint_validation_waiver_modules_parallel(
            [Path(path) for path in paths],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
            costs=costs,
        )
    submitted = [chunk[0] for _fn, chunk, _args in _FakePool.submitted]
    return rows, submitted, stderr.getvalue().splitlines()


def test_parallel_audit_starts_longest_first_and_reports_its_plan():
    paths = [f"us/{name}.yaml" for name in "abcde"]
    costs = _costs({"us/c.yaml": 9_000, "us/e.yaml": 5_000, "us/a.yaml": 10})
    rows, submitted, lines = _run(paths, costs=costs)
    assert [row["path"] for row in rows] == paths
    # b and d take the table's median (5000) and keep path order among ties.
    assert submitted == [
        "us/c.yaml",
        "us/b.yaml",
        "us/d.yaml",
        "us/e.yaml",
        "us/a.yaml",
    ]
    assert lines[0] == (
        "validation-waivers audit plan: 5 module(s) on 2 worker(s), longest "
        f"estimated cost first (24s of module time, cost table "
        f"sha256:{costs.sha256[:12]})"
    )
    assert [line for line in lines if " start: " in line] == [
        f"validation-waivers audit start: {path}" for path in submitted
    ]
    _rows, plain, plain_lines = _run(paths)
    assert plain == paths
    assert plain_lines[0] == (
        "validation-waivers audit plan: 5 module(s) on 2 worker(s), in path order"
    )


def test_parallel_audit_beats_while_a_module_runs():
    paths = ["us/a.yaml", "us/b.yaml", "us/c.yaml"]
    calls = []

    def wait(futures, timeout):
        calls.append(sorted(_FakePool.path_of[future] for future in futures))
        if len(calls) <= 2:
            return set()  # time out twice before anything finishes
        return {min(futures, key=lambda future: paths.index(_FakePool.path_of[future]))}

    _rows, _submitted, lines = _run(paths, workers=2, wait=wait)
    heartbeats = [line for line in lines if " heartbeat: " in line]
    assert len(heartbeats) == 2
    assert all(
        re.fullmatch(
            r"validation-waivers audit heartbeat: 0 of 3 module\(s\) finished "
            r"after \d+s; 2 running: us/a\.yaml \d+s; us/b\.yaml \d+s",
            line,
        )
        for line in heartbeats
    ), heartbeats
    # Never more tasks in flight than workers.
    assert all(len(in_flight) <= 2 for in_flight in calls)


def test_serial_audit_runs_inside_a_heartbeat(monkeypatch):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    entered = []

    @contextlib.contextmanager
    def recording(progress, *, interval=None):
        entered.append((progress, interval))
        yield

    def serial(modules, **kwargs):
        progress = kwargs["progress"]
        assert entered and entered[-1][0] is progress
        for module in modules:
            progress.start(str(module))
            progress.finish(str(module))
        return [{"path": str(module)} for module in modules]

    stderr = io.StringIO()
    with (
        patch.object(partition, "heartbeat_thread", recording),
        patch.object(cli, "_fingerprint_validation_waiver_modules", serial),
        redirect_stderr(stderr),
    ):
        rows = cli._fingerprint_validation_waiver_modules_parallel(
            [Path("us/b.yaml"), Path("us/a.yaml")],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )
    assert rows == [{"path": "us/a.yaml"}, {"path": "us/b.yaml"}]
    assert entered[0][1] == partition.HEARTBEAT_SECONDS
    assert stderr.getvalue().splitlines() == [
        "validation-waivers audit plan: 2 module(s) on 1 worker, in path order",
        "validation-waivers audit start: us/a.yaml",
        "validation-waivers audit start: us/b.yaml",
    ]


# ---------------------------------------------------------------- whole audit


def _audit_every_shard(root, paths, keys):
    waivers = _waiver_set(
        {path: _entry(path, active=_metadata("sha256:base")) for path in paths}
    )
    for path in paths:
        module = root / path
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("format: rulespec/v1\n")
    base = root.parent / "base.yaml"
    base.write_text("validate_failures: {}\n")
    changed = root.parent / "changed.txt"
    changed.write_text("")

    def fingerprint(modules, **kwargs):
        return [
            {
                "path": module.as_posix(),
                "passed": False,
                "fingerprint": "sha256:base",
                "outcome": {},
            }
            for module in modules
        ]

    reports = {}
    for key in keys:
        args = _audit_args(root, base, changed, root.parent / "corpus")
        args.partition_key = key
        args.partition_keys_json = json.dumps(list(reversed(keys)))
        output, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(
                cli._validation_waivers, "load_validation_waivers", return_value=waivers
            ),
            patch.object(
                cli._validation_waivers,
                "protected_base_transition_issues",
                return_value=(),
            ),
            patch.object(
                cli,
                "_fingerprint_validation_waiver_modules_parallel",
                side_effect=fingerprint,
            ),
            redirect_stdout(output),
            redirect_stderr(stderr),
        ):
            code = cli._cmd_validation_waivers_audit(args)
        assert code == 0
        reports[key] = (json.loads(output.getvalue()), stderr.getvalue())
    return reports


def test_whole_audit_with_a_table_audits_every_waiver_once(tmp_path):
    root = tmp_path / "rulespec-us"
    paths = [
        f"us-{state}/statutes/{index}.yaml" for state in "ab" for index in range(5)
    ]
    paths.append("us/statutes/heavy.yaml")
    keys = ["us", "us-ak", "us-al"]
    table = partition.render_waiver_audit_costs(
        {"us/statutes/heavy.yaml": [600_000], "us-a/statutes/0.yaml": [1_000]},
        baselines={"us": 2_000_000},
    )
    (root / ".axiom").mkdir(parents=True)
    (root / partition.COST_TABLE_PATH).write_bytes(table)
    reports = _audit_every_shard(root, paths, keys)

    audited = [
        item["path"] for report, _ in reports.values() for item in report["results"]
    ]
    assert sorted(audited) == sorted(paths)
    assert len(audited) == len(set(audited))
    digest = __import__("hashlib").sha256(table).hexdigest()
    for key, (report, stderr) in reports.items():
        assert report["partition"]["cost_table_sha256"] == digest
        plan = [line for line in stderr.splitlines() if " plan: " in line]
        assert len(plan) == 1 and "cost-balanced" in plan[0] and key in plan[0]
    # The busiest shard's own work keeps every waiver off it.
    assert reports["us"][0]["results"] == []


def test_whole_audit_without_a_table_keeps_legacy_ownership_and_bytes(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir()
    paths = [
        f"us-{state}/statutes/{index}.yaml" for state in "abc" for index in range(4)
    ]
    keys = ["us", "us-ak", "us-al"]
    reports = _audit_every_shard(root, paths, keys)
    legacy = _legacy_round_robin(paths, keys)
    for key, (report, stderr) in reports.items():
        assert [item["path"] for item in report["results"]] == [
            path for path in sorted(paths) if legacy[path] == key
        ]
        assert set(report["partition"]) == {"key", "index", "count", "total_paths"}
        assert "round-robin" in stderr
