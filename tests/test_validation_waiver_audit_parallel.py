"""Dispatch, ordering, and integrity tests for the parallel waiver audit."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli

_RELEASE_IDENTITY = ("/corpus", "test-release", "0" * 64, "cHVibGljLWtleQ==")


def _release_stub():
    return SimpleNamespace(
        root=Path("/corpus"),
        name="test-release",
        content_sha256="0" * 64,
        public_key="cHVibGljLWtleQ==",
    )


class _FakeFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


class _FakePool:
    """Deterministic in-process stand-in for ProcessPoolExecutor."""

    submitted: list[tuple] = []
    chunk_results: dict[str, list] = {}

    def __init__(self, max_workers):
        type(self).max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, chunk, *args):
        type(self).submitted.append((fn, tuple(chunk), args))
        return _FakeFuture(type(self).chunk_results[chunk[0]])


@pytest.fixture(autouse=True)
def _reset_fake_pool():
    _FakePool.submitted = []
    _FakePool.chunk_results = {}


def test_worker_count_defaults_to_cpu_bounded(monkeypatch):
    monkeypatch.delenv(cli._WAIVER_AUDIT_WORKERS_ENV, raising=False)
    with patch.object(cli.os, "cpu_count", return_value=4):
        assert cli._waiver_audit_worker_count(100) == 4
        assert cli._waiver_audit_worker_count(2) == 2
    with patch.object(cli.os, "cpu_count", return_value=32):
        assert cli._waiver_audit_worker_count(100) == 8


def test_worker_count_env_override(monkeypatch):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "3")
    assert cli._waiver_audit_worker_count(100) == 3
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "0")
    assert cli._waiver_audit_worker_count(100) == 1
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    assert cli._waiver_audit_worker_count(100) == 1
    # Overrides are hard-capped so a hostile value cannot fork-bomb the runner.
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "3109")
    assert cli._waiver_audit_worker_count(3109) == cli._WAIVER_AUDIT_WORKERS_MAX


@pytest.mark.parametrize("raw", ["-1", "many"])
def test_worker_count_rejects_invalid_env(monkeypatch, raw):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, raw)
    with pytest.raises(ValueError):
        cli._waiver_audit_worker_count(100)


def test_single_worker_delegates_to_serial(monkeypatch):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    sentinel = [{"path": "us/a.yaml", "passed": False, "fingerprint": "f"}]
    with patch.object(
        cli, "_fingerprint_validation_waiver_modules", return_value=sentinel
    ) as serial:
        result = cli._fingerprint_validation_waiver_modules_parallel(
            [Path("us/b.yaml"), Path("us/a.yaml")],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )
    assert result == sentinel
    assert serial.call_count == 1
    # The serial path receives the same sorted order the parallel path uses.
    assert [str(p) for p in serial.call_args.args[0]] == ["us/a.yaml", "us/b.yaml"]


def _run_parallel(paths, monkeypatch, workers="2"):
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, workers)
    with (
        patch.object(
            cli, "load_rulespec_local_corpus_release", return_value=_release_stub()
        ),
        patch.object(cli.concurrent.futures, "ProcessPoolExecutor", _FakePool),
    ):
        return cli._fingerprint_validation_waiver_modules_parallel(
            [Path(p) for p in paths],
            root=Path("/repo"),
            corpus_path=Path("/corpus"),
            axiom_rules_path=Path("/engine"),
        )


def test_parallel_results_are_merged_and_sorted(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    # Contiguous chunks of the sorted list; return each chunk's rows reversed
    # to prove the merge re-establishes deterministic order.
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        _FakePool.chunk_results[piece[0]] = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in reversed(piece)
        ]
    results = _run_parallel(list(reversed(paths)), monkeypatch)
    assert [row["path"] for row in results] == sorted(paths)
    submitted_paths = [
        path for _fn, chunk_paths, _args in _FakePool.submitted for path in chunk_paths
    ]
    assert submitted_paths == sorted(paths)
    for _fn, _chunk, args in _FakePool.submitted:
        assert args == (
            "/repo",
            "/corpus",
            "/engine",
            (),
            _RELEASE_IDENTITY,
        )


def test_parallel_detects_lost_results(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        rows = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in piece
        ]
        _FakePool.chunk_results[piece[0]] = rows[:-1] if start == 0 else rows
    with pytest.raises(RuntimeError, match="corrupted module set"):
        _run_parallel(paths, monkeypatch)


def test_parallel_detects_duplicate_masking_missing(monkeypatch):
    # Same cardinality, wrong multiset: one path duplicated, one omitted.
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        rows = [
            {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
            for path in piece
        ]
        if start == 0:
            rows[-1] = dict(rows[0])
        _FakePool.chunk_results[piece[0]] = rows
    with pytest.raises(RuntimeError, match="duplicated"):
        _run_parallel(paths, monkeypatch)


def test_parallel_propagates_worker_failure(monkeypatch):
    paths = [f"us/{index:03d}.yaml" for index in range(20)]
    chunk = max(8, -(-len(paths) // (2 * 8)))
    for start in range(0, len(paths), chunk):
        piece = paths[start : start + chunk]
        _FakePool.chunk_results[piece[0]] = (
            ValueError("worker exploded")
            if start == 0
            else [
                {"path": path, "passed": False, "fingerprint": f"fp-{path}"}
                for path in piece
            ]
        )
    with pytest.raises(ValueError, match="worker exploded"):
        _run_parallel(paths, monkeypatch)


def test_chunk_worker_builds_release_without_broker():
    sentinel = [{"path": "us/a.yaml", "passed": True, "fingerprint": "fp"}]
    built = {}

    def _fake_release(root, name, sha, key):
        built["identity"] = (str(root), name, sha, key)
        return "release-object"

    with (
        patch.object(cli, "LocalCorpusRelease", side_effect=_fake_release),
        patch.object(
            cli, "_fingerprint_validation_waiver_modules", return_value=sentinel
        ) as serial,
    ):
        result = cli._fingerprint_waiver_chunk(
            ["us/a.yaml"],
            "/repo",
            "/corpus",
            "/engine",
            (),
            _RELEASE_IDENTITY,
        )
    assert result == sentinel
    assert built["identity"] == _RELEASE_IDENTITY
    assert serial.call_args.kwargs["corpus_release"] == "release-object"
