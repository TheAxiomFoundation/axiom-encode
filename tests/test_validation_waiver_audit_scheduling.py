"""Work-conserving scheduling with deterministic, simulated module costs."""

import pickle
from collections import Counter
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from tests.test_validation_waiver_audit_parallel import _release_stub


def _simulate(modules, workers, *, fail=None):
    costs = {f"us/statutes/{key:03d}.yaml": cost for key, cost in modules.items()}
    calls = []
    intervals = []

    def fingerprint(paths, **kwargs):
        rows = []
        for path in sorted(map(str, paths)):
            calls.append(path)
            if path == fail:
                raise RuntimeError("worker exploded")
            rows.append(
                {"path": path, "passed": False, "fingerprint": path, "outcome": {}}
            )
        return rows

    class CostPool:
        """Model the executor's shared task queue without clocks or sleeps.

        A task goes to the next available worker. Modules inside a task run
        sequentially on that worker, just as in the real chunk entrypoint.
        Thus fixed slices leave measurable holes in the module timeline.
        """

        def __init__(self, max_workers):
            self.ready = [0] * max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, paths, *args):
            worker = min(range(len(self.ready)), key=self.ready.__getitem__)
            for path in paths:
                start = self.ready[worker]
                self.ready[worker] += costs[path]
                intervals.append((worker, path, start, self.ready[worker]))
            future = Future()
            try:
                # Exercise the same pickle boundary as a process pool,
                # including private timing receipts and worker exceptions.
                value = fn(paths, *args)
                future.set_result(pickle.loads(pickle.dumps(value)))
            except Exception as error:
                future.set_exception(error)
            return future

    with (
        patch.dict(cli.os.environ, {cli._WAIVER_AUDIT_WORKERS_ENV: str(workers)}),
        patch.object(
            cli, "load_rulespec_local_corpus_release", return_value=_release_stub()
        ),
        patch.object(cli, "LocalCorpusRelease", return_value=_release_stub()),
        patch.object(cli, "_fingerprint_validation_waiver_modules", fingerprint),
        patch.object(cli.concurrent.futures, "ProcessPoolExecutor", CostPool),
    ):
        result = cli._fingerprint_validation_waiver_modules_parallel(
            [Path(path) for path in reversed(costs)],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )
    return result, calls, intervals


_MODULE_COSTS = st.dictionaries(st.integers(0, 999), st.integers(1, 30), max_size=12)


@settings(max_examples=100, deadline=None)
@given(modules=_MODULE_COSTS, workers=st.integers(1, 8))
@example(modules={0: 9, 1: 1, 2: 1, 3: 1}, workers=2)
def test_dynamic_schedule_is_work_conserving_and_fingerprints_once(modules, workers):
    result, calls, intervals = _simulate(modules, workers)
    expected = sorted(f"us/statutes/{key:03d}.yaml" for key in modules)
    assert result == [
        {"path": path, "passed": False, "fingerprint": path, "outcome": {}}
        for path in expected
    ]
    assert Counter(calls) == Counter(expected)
    if not intervals:  # workers <= 1 delegates directly to the serial executor.
        assert workers == 1 or len(modules) <= 1
        return

    last_start = max(start for _worker, _path, start, _end in intervals)
    for worker in range(min(workers, len(modules))):
        busy = sum(
            max(0, min(end, last_start) - start)
            for owner, _path, start, end in intervals
            if owner == worker
        )
        assert busy == last_start, (
            "worker idle while an unstarted module remains",
            worker,
            intervals,
        )


@settings(max_examples=50, deadline=None)
@given(modules=_MODULE_COSTS.filter(bool), workers=st.integers(1, 8), data=st.data())
def test_random_worker_exception_fails_audit(modules, workers, data):
    failed = data.draw(st.sampled_from(list(modules)))
    with pytest.raises(RuntimeError, match="worker exploded"):
        _simulate(modules, workers, fail=f"us/statutes/{failed:03d}.yaml")
