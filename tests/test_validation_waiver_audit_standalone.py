"""Standalone fingerprints and CLI bytes survive audit task scheduling."""

from __future__ import annotations

import concurrent.futures
import contextlib
import io
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.harness import validator_pipeline
from axiom_encode.validation_waiver_scope import WaiverAuditScope
from tests.test_validation_waiver_audit_cli import (
    _audit_args,
    _bound_corpus,
    _engine,
    _entry,
    _metadata,
    _pipeline_result,
    _waiver_set,
)
from tests.test_validation_waiver_scope import _module


class _ScopeCheckingPipeline(validator_pipeline.ValidatorPipeline):
    """Keep real setup and validate scopes; replace engine validation only."""

    scopes: list[object] = []

    def _axiom_rules_binary(self):
        return self.axiom_rules_path / "axiom-rules-engine"

    def _validate_with_authoritative_roots(self, rulespec_file, skip_reviewers=False):
        assert skip_reviewers is True
        cache = validator_pipeline._RULESPEC_RESOLUTION_CACHE.get()
        assert cache is not None
        # A batch-level resolution scope would leak this marker into the
        # next module, reproducing the historical partition-dependent bug.
        assert ("standalone-test-module",) not in cache.target_files
        cache.target_files[("standalone-test-module",)] = rulespec_file
        type(self).scopes.append(cache)
        passed = "fail" not in rulespec_file.stem
        message = (
            f"validation failed for {rulespec_file}; "
            f"engine={self.axiom_rules_path}; corpus={self.local_corpus_release.root}"
        )
        return _pipeline_result(
            passed=passed,
            issues=() if passed else (message,),
            error=None if passed else message,
        )


class _ScheduledFuture(concurrent.futures.Future):
    def __init__(self, fn, args):
        super().__init__()
        self._call = (fn, args)

    def result(self, timeout=None):
        if not self.done():
            fn, args = self._call
            try:
                # A fork worker inherits its own copy of a redirected
                # StringIO (as used by ci_parity._run_cli). Child writes
                # cannot reach the parent's captured stderr directly.
                with contextlib.redirect_stderr(io.StringIO()):
                    self.set_result(fn(*args))
            except Exception as error:
                self.set_exception(error)
        return super().result(timeout)


class _SimulatedExecutor:
    """Run real worker entrypoints in-process in controlled completion order."""

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        return _ScheduledFuture(fn, args)


def _completion_order(order):
    def completed(futures):
        ordered = list(futures)
        if order == "reverse":
            ordered.reverse()
        elif order == "stride":
            ordered = ordered[1::2] + ordered[::2]
        for future in ordered:
            future.result()
            yield future

    return completed


@pytest.fixture
def standalone_repo(tmp_path, monkeypatch):
    root = tmp_path / "rulespec-us"
    paths = [
        "us/statutes/a-pass.yaml",
        "us/statutes/b-fail.yaml",
        "us-ca/policies/c-composition.yaml",
        "us-ca/statutes/d-parse.yaml",
        "us-ny/statutes/e-compile.yaml",
        "us-ny/statutes/f-pass.yaml",
    ]
    for relative in paths:
        module = _module(root, relative)
        if "composition" in relative:
            module.write_text(
                "format: rulespec/v1\nmodule:\n  kind: composition\nrules: []\n"
            )
        if not relative.endswith("a-pass.yaml"):
            module.with_suffix(".test.yaml").write_text(
                "not: valid cases\n" if "parse" in relative else "cases: []\n"
            )
    engine = _engine(tmp_path)
    corpus = _bound_corpus(tmp_path, root)
    _ScopeCheckingPipeline.scopes = []
    monkeypatch.setattr(cli, "ValidatorPipeline", _ScopeCheckingPipeline)

    def compile_stub(**kwargs):
        if "e-compile" in kwargs["program"].name:
            return subprocess.CompletedProcess(
                [],
                1,
                "",
                f"compile failed: {kwargs['program']}; output={kwargs['output']}",
            )
        kwargs["output"].write_text(
            json.dumps({"program": {"parameters": [], "derived": [], "relations": []}})
        )
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(cli, "run_rulespec_compile", compile_stub)
    monkeypatch.setattr(validator_pipeline, "run_rulespec_compile", compile_stub)
    kwargs = {"root": root, "corpus_path": corpus, "axiom_rules_path": engine}
    standalone = sorted(
        [
            cli._fingerprint_validation_waiver_modules([Path(path)], **kwargs)[0]
            for path in paths
        ],
        key=lambda result: result["path"],
    )
    return SimpleNamespace(
        root=root,
        paths=[Path(path) for path in paths],
        kwargs=kwargs,
        standalone=standalone,
        base=tmp_path / "base.yaml",
        changed=tmp_path / "changed.txt",
    )


@pytest.mark.parametrize("workers", [1, 2, 3, 4])
@pytest.mark.parametrize("completion", ["forward", "reverse", "stride"])
def test_all_schedules_equal_standalone_fingerprints(
    standalone_repo, monkeypatch, workers, completion
):
    repo = standalone_repo
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, str(workers))
    monkeypatch.setattr(
        cli.concurrent.futures, "ProcessPoolExecutor", _SimulatedExecutor
    )
    monkeypatch.setattr(
        cli.concurrent.futures, "as_completed", _completion_order(completion)
    )
    orderings = [
        repo.paths,
        list(reversed(repo.paths)),
        repo.paths[2:] + repo.paths[:2],
    ]
    for ordering in orderings:
        assert (
            cli._fingerprint_validation_waiver_modules(ordering, **repo.kwargs)
            == repo.standalone
        )
        assert (
            cli._fingerprint_validation_waiver_modules_parallel(ordering, **repo.kwargs)
            == repo.standalone
        )
    assert len({id(scope) for scope in _ScopeCheckingPipeline.scopes}) == len(
        _ScopeCheckingPipeline.scopes
    )
    assert any(result["passed"] for result in repo.standalone)
    assert any(not result["passed"] for result in repo.standalone)
    serialized = json.dumps(repo.standalone)
    assert str(repo.root) not in serialized
    assert str(repo.kwargs["axiom_rules_path"]) not in serialized
    assert "requires an explicit axiom-compose executable" in serialized
    assert "<tmp>" in serialized


def _assert_timings(stderr, paths):
    lines = stderr.splitlines()
    assert len(lines) == len(paths)
    for path in paths:
        matching = [line for line in lines if path in line]
        assert len(matching) == 1
        assert re.search(r"\b\d+(?:\.\d+)?(?:s| seconds)\b", matching[0])


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("mode", ["full", "incremental", "partition"])
@pytest.mark.parametrize("recheck", [False, True])
def test_audit_json_bytes_and_one_timing_per_initial_module(
    standalone_repo, monkeypatch, capsys, workers, mode, recheck
):
    repo = standalone_repo
    rows = [result for result in repo.standalone if not result["passed"]]
    paths = [result["path"] for result in rows]
    selected = paths[:3] if mode == "incremental" else paths
    if mode == "partition":
        selected = paths[::2]
    bad_path = selected[0] if recheck else None
    wrong_fingerprint = "sha256:" + "0" * 64
    waivers = _waiver_set(
        {
            result["path"]: _entry(
                result["path"],
                active=_metadata(
                    wrong_fingerprint
                    if result["path"] == bad_path
                    else result["fingerprint"]
                ),
            )
            for result in rows
        }
    )
    repo.base.write_text("validate_failures: {}\n")
    repo.changed.write_text("")
    args = _audit_args(repo.root, repo.base, repo.changed, repo.kwargs["corpus_path"])
    args.axiom_rules_path = repo.kwargs["axiom_rules_path"]
    if mode == "incremental":
        args.scope = "incremental"
    if mode == "partition":
        args.partition_key = "shard-a"
        args.partition_keys_json = '["shard-b", "shard-a"]'
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, str(workers))
    monkeypatch.setattr(
        cli.concurrent.futures, "ProcessPoolExecutor", _SimulatedExecutor
    )
    monkeypatch.setattr(
        cli.concurrent.futures, "as_completed", _completion_order("reverse")
    )
    reasons = {path: ["fixture module changed"] for path in selected}
    with (
        patch.object(
            cli._validation_waivers, "load_validation_waivers", return_value=waivers
        ),
        patch.object(
            cli._validation_waivers, "protected_base_transition_issues", return_value=[]
        ),
        patch.object(
            cli,
            "select_validation_waiver_scope",
            return_value=WaiverAuditScope("incremental", tuple(selected), reasons),
        ),
    ):
        code = cli._cmd_validation_waivers_audit(args)
    captured = capsys.readouterr()
    expected_rows = []
    errors = []
    for actual in rows:
        path = actual["path"]
        if path not in selected:
            continue
        record = {
            "path": path,
            "kind": "active",
            **{key: value for key, value in actual.items() if key != "path"},
        }
        if path == bad_path:
            record["isolated_recheck"] = True
        record["expected_fingerprint"] = waivers.entries[path].active.fingerprint
        record["success"] = path != bad_path
        if path == bad_path:
            error = (
                f"{path}: validation failure fingerprint drifted; expected "
                f"{wrong_fingerprint}, got {actual['fingerprint']}"
            )
            record["error"] = error
            errors.append(error)
        expected_rows.append(record)
    report = {
        "success": not errors,
        "root": str(repo.root),
        "waivers": "known-validation-gaps.yaml",
        "checked": len(selected),
        "results": expected_rows,
        "errors": errors,
        "scope": "incremental" if mode == "incremental" else "full",
    }
    if mode == "incremental":
        report.update(
            selected=len(selected),
            total_waivers=len(paths),
            selection_reasons=reasons,
        )
    if mode == "partition":
        report["partition"] = {
            "key": "shard-a",
            "index": 0,
            "count": 2,
            "total_paths": len(paths),
        }
    assert code == int(recheck)
    assert captured.out == json.dumps(report, indent=2) + "\n"
    _assert_timings(captured.err, selected)


def test_public_fingerprint_json_bytes_stay_unchanged(standalone_repo, capsys):
    repo = standalone_repo
    args = SimpleNamespace(modules=list(reversed(repo.paths)), json=True, **repo.kwargs)
    assert cli._cmd_validation_waivers_fingerprint(args) == 0
    captured = capsys.readouterr()
    expected = {"success": True, "root": str(repo.root), "modules": repo.standalone}
    assert captured.out == json.dumps(expected, indent=2) + "\n"
    assert captured.err == ""


def test_parallel_worker_failure_preserves_timing_in_parent(
    standalone_repo, monkeypatch, capsys
):
    repo = standalone_repo
    paths = [Path("us/statutes/a-pass.yaml"), Path("us/statutes/b-fail.yaml")]
    original = _ScopeCheckingPipeline._validate_with_authoritative_roots

    def validate(self, module, skip_reviewers=False):
        if module.name == "b-fail.yaml":
            raise RuntimeError("worker validation broke")
        return original(self, module, skip_reviewers=skip_reviewers)

    monkeypatch.setattr(
        _ScopeCheckingPipeline, "_validate_with_authoritative_roots", validate
    )
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "2")
    monkeypatch.setattr(
        cli.concurrent.futures, "ProcessPoolExecutor", _SimulatedExecutor
    )
    monkeypatch.setattr(
        cli.concurrent.futures, "as_completed", _completion_order("reverse")
    )
    with pytest.raises(RuntimeError, match="worker validation broke"):
        cli._fingerprint_validation_waiver_modules_parallel(paths, **repo.kwargs)
    output = capsys.readouterr()
    assert output.out == ""
    # The failing task completes first. The pool still runs the successful
    # task, so its receipt and timing must be drained before raising.
    _assert_timings(output.err, [str(path) for path in paths])
