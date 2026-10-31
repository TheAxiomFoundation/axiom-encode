"""SHARDS-lane regressions against the two pinned workflow jobs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity
from axiom_encode.ci_parity import GateResult, ShardSelection, execute_workflow_gates
from axiom_encode.validation_waivers import load_validation_waivers
from tests import test_ci_parity_workflow_gates as workflow_gates
from tests.test_ci_parity_runtime import _CiHarness
from tests.test_ci_parity_workflow_gates import (
    CHANGED,
    EMBEDDED_PINS,
    FREEZE_PATH,
    PIN_6F11BE26,
    PLAIN,
    RETIRED,
    RULESPEC_US_REMOTE,
    _commit,
    _freeze,
    _module_run,
    _patch_batch,
    _plan,
    _repo,
    _run,
    _spec,
)


@EMBEDDED_PINS
@pytest.mark.parametrize(
    "phase",
    [
        "resolve_workflow_toolchain",
        "load_rulespec_toolchain",
        "verify_dependency_checkout",
        "acquire_workflow_release_object",
        "authenticate_release_provenance",
        "load_rulespec_local_corpus_release",
    ],
)
def test_resolution_failure_preserves_the_completed_shards_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pin: str,
    phase: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=pin)

    def fail(*_args, **_kwargs):
        raise ValueError(f"{phase} failed")

    monkeypatch.setattr(ci_parity, phase, fail)

    assert ci_parity.run_ci(harness.args()) == 1
    report = json.loads(capsys.readouterr().out)
    expected = ["unsupported_paths"]
    if pin == PIN_6F11BE26:
        expected.append("unmanifested_rulespec")
    expected.append("compute_shards")
    assert [gate["gate"] for gate in report["gates"]] == expected
    assert all(gate["status"] == "PASS" for gate in report["gates"])
    assert report["resolution_error"] == f"{phase} failed"
    assert report["verdict"] == "FAIL"
    assert "execute_workflow_gates" not in harness.calls


@pytest.mark.parametrize("failure", ["unsupported_paths", "unmanifested_rulespec"])
def test_a_failed_shards_step_skips_the_remaining_job_and_validate_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=PIN_6F11BE26)
    monkeypatch.setattr(
        ci_parity,
        f"_gate_{failure}",
        lambda run, spec: GateResult(spec.key, spec.name, "FAIL", failures=["bad"]),
    )

    assert ci_parity.run_ci(harness.args()) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["gates"][-1]["gate"] == failure
    assert report["gates"][-1]["status"] == "FAIL"
    assert report["verdict"] == "FAIL"
    assert "compute_shard_plan" not in harness.calls
    assert "resolve_workflow_toolchain" not in harness.calls
    assert "execute_workflow_gates" not in harness.calls


def test_shard_computation_failure_reports_earlier_gates_and_its_own_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=PIN_6F11BE26)

    def fail(*_args):
        raise ValueError("no jurisdiction directories")

    monkeypatch.setattr(ci_parity, "compute_shard_plan", fail)

    assert ci_parity.run_ci(harness.args()) == 1
    report = json.loads(capsys.readouterr().out)
    assert [gate["status"] for gate in report["gates"]] == ["PASS", "PASS", "FAIL"]
    assert report["gates"][-1]["gate"] == "compute_shards"
    assert report["resolution_error"] == "no jurisdiction directories"
    assert "resolve_workflow_toolchain" not in harness.calls


@EMBEDDED_PINS
def test_shards_checks_execute_once_before_the_validate_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pin: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=pin)
    for key in ci_parity.SUPPORTED_WORKFLOW_PINS[pin].gates:
        if key not in {"unsupported_paths", "unmanifested_rulespec"}:
            monkeypatch.setattr(
                ci_parity,
                f"_gate_{key}",
                lambda run, spec: GateResult(spec.key, spec.name, "PASS"),
            )
    monkeypatch.setattr(ci_parity, "execute_workflow_gates", execute_workflow_gates)

    assert ci_parity.run_ci(harness.args()) == 0
    report = json.loads(capsys.readouterr().out)
    keys = [gate["gate"] for gate in report["gates"]]
    assert keys.count("unsupported_paths") == 1
    assert keys.count("compute_shards") == 1
    assert len(harness.calls["_gate_unsupported_paths"]) == 1
    if pin == PIN_6F11BE26:
        assert keys.count("unmanifested_rulespec") == 1
        assert len(harness.calls["_gate_unmanifested_rulespec"]) == 1


@EMBEDDED_PINS
def test_selection_prints_ci_headings_for_selected_and_empty_shards(
    tmp_path: Path, pin: str
) -> None:
    repo = _repo(
        tmp_path,
        {
            "us-ak/statutes/a.yaml": PLAIN,
            "us-ak/statutes/a.test.yaml": "cases: []\n",
            "us/statutes/b.yaml": PLAIN,
        },
    )
    _commit(repo, {"us-ak/statutes/a.yaml": CHANGED})
    run = _run(repo, tmp_path, sha=pin, plan=_plan("us-ak", "us"))

    result = ci_parity._gate_select_targets(run, _spec("select_targets", pin))

    assert result.output.startswith(
        "[us-ak] RuleSpec validation mode: changed\n"
        "[us-ak] RuleSpec files selected:\n"
        "[us-ak] - us-ak/statutes/a.yaml\n"
        "[us-ak] RuleSpec companion tests selected:\n"
        "[us-ak] - us-ak/statutes/a.test.yaml\n"
        "[us] RuleSpec validation mode: changed\n"
        "[us] RuleSpec files selected:\n"
        "[us] RuleSpec companion tests selected:\n"
    )


@settings(deadline=None, max_examples=20)
@given(
    shards=st.lists(st.sampled_from(["us", "us-ak", "us-ca"]), min_size=1, unique=True)
)
def test_every_shard_prints_both_selection_headings(shards: list[str]) -> None:
    # For all matrices, including empty selections, each leg has both lists.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ci_parity, "_changed_path_list", lambda *_: [])
        run = ci_parity.WorkflowRun(
            caller=None,
            workflow=None,
            paths={},
            repo=Path("unused"),
            simulation=ci_parity.PullRequestSimulation(
                "a" * 40, "b" * 40, "main", "o/r", None, True
            ),
            plan=_plan(*shards),
            validate_roots_input="auto",
            keyring=(),
        )
        result = ci_parity._gate_select_targets(run, _spec("select_targets"))
    assert result.output.count("RuleSpec files selected:\n") == len(shards)
    assert result.output.count("RuleSpec companion tests selected:\n") == len(shards)


@EMBEDDED_PINS
@pytest.mark.parametrize(
    "rule_files,test_files", [((), ()), (("a.yaml",), ("a.test.yaml",))]
)
def test_selection_headed_blocks_match_the_workflow_bash(
    tmp_path: Path, pin: str, rule_files: tuple[str, ...], test_files: tuple[str, ...]
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path, sha=pin, plan=_plan("us"))
    files = tuple(f"us/statutes/{name}" for name in rule_files)
    tests = tuple(f"us/statutes/{name}" for name in test_files)
    rules_path, tests_path = tmp_path / "rules.txt", tmp_path / "tests.txt"
    rules_path.write_text("".join(f"{name}\n" for name in files))
    tests_path.write_text("".join(f"{name}\n" for name in tests))
    body = run.workflow.step("validate", "Select RuleSpec validation targets")["run"]
    block = body[body.index('echo "RuleSpec validation mode: $mode"') :]
    script = "mode=changed\nrulespec_file_list=$1\ntest_file_list=$2\n" + block
    expected = subprocess.run(
        [
            bash,
            "--noprofile",
            "--norc",
            "-eo",
            "pipefail",
            "-c",
            script,
            "bash",
            str(rules_path),
            str(tests_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={
            "PATH": str(Path(shutil.which("sed") or "/usr/bin/sed").parent),
            "LC_ALL": "C",
        },
    ).stdout
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ci_parity, "_changed_path_list", lambda *_: [*files, *tests])
        patch.setattr(
            ci_parity,
            "_add_rulespec_target",
            lambda repo, path, roots, auto, rules, found_tests, listings: (
                rules.update(files),
                found_tests.update(tests),
            ),
        )
        result = ci_parity._gate_select_targets(run, _spec("select_targets", pin))
    assert result.output.startswith(expected)


@settings(deadline=None, max_examples=12)
@given(unsupported=st.booleans(), manifested=st.booleans())
def test_shards_job_runs_exactly_through_its_first_failure(
    unsupported: bool, manifested: bool
) -> None:
    run = ci_parity._ShardsRun(
        caller=SimpleNamespace(guard_programs_root=False),
        workflow=ci_parity.PinnedWorkflow(PIN_6F11BE26),
        repo=Path("unused"),
        simulation=None,
    )
    with (
        mock.patch.object(
            ci_parity,
            "_gate_unsupported_paths",
            side_effect=lambda run, spec: GateResult(
                spec.key, spec.name, "PASS" if unsupported else "FAIL"
            ),
        ),
        mock.patch.object(
            ci_parity,
            "_gate_unmanifested_rulespec",
            side_effect=lambda run, spec: GateResult(
                spec.key, spec.name, "PASS" if manifested else "FAIL"
            ),
        ),
        mock.patch.object(
            ci_parity, "compute_shard_plan", return_value=_plan("us")
        ) as compute,
    ):
        plan = ci_parity._execute_shards_job(run, "auto")
    expected = ["unsupported_paths"]
    if unsupported:
        expected.append("unmanifested_rulespec")
        if manifested:
            expected.append("compute_shards")
    assert [result.gate for result in run.results] == expected
    assert (plan is not None) == (unsupported and manifested)
    assert compute.call_count == int(unsupported and manifested)


def test_validate_diagnostic_continuation_is_noted_on_every_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path, {"us/statutes/a.yaml": PLAIN})
    run = _run(repo, tmp_path)
    run.shards_done = True
    for key in run.workflow.pin.gates:
        monkeypatch.setattr(
            ci_parity,
            f"_gate_{key}",
            lambda run, spec: GateResult(
                spec.key,
                spec.name,
                "FAIL" if spec.key == "retired_schema_freeze" else "PASS",
                note="existing note",
            ),
        )

    results = execute_workflow_gates(run)
    failed = next(
        index for index, result in enumerate(results) if result.status == "FAIL"
    )
    assert all(result.note == "existing note" for result in results[: failed + 1])
    assert all(
        "CI skips later steps on each leg with an earlier failure" in result.note
        and result.note.startswith("existing note; ")
        for result in results[failed + 1 :]
    )


@EMBEDDED_PINS
@pytest.mark.parametrize("failure", ["prefreeze", "added", "changed"])
def test_retired_schema_nonfirst_leg_keeps_its_own_skip_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str, failure: str
) -> None:
    module = "us/statutes/x.yaml"
    base = {module: RETIRED, "us-ak/statutes/a.yaml": PLAIN}
    if failure == "changed":
        base[FREEZE_PATH] = _freeze({module: RETIRED})
    repo = _repo(tmp_path, base, remote=RULESPEC_US_REMOTE)
    content = RETIRED + "# edited\n" if failure == "changed" else RETIRED
    _commit(
        repo,
        {
            module: content,
            FREEZE_PATH: _freeze({module: content}),
            "us-ak/statutes/a.yaml": CHANGED,
            "us/statutes/x.test.yaml": "cases: []\n",
        },
    )
    run = _run(
        repo,
        tmp_path,
        sha=pin,
        inputs={"allow-retired-schema-prefreeze": failure == "prefreeze"},
        plan=_plan("us-ak", "us"),
    )
    freeze = ci_parity._gate_retired_schema_freeze(
        run, _spec("retired_schema_freeze", pin)
    )

    assert freeze.status == "FAIL"
    assert run.retired_skip == frozenset()
    assert run.retired_skip_by_shard == {
        "us-ak": frozenset(),
        "us": frozenset({module}),
    }
    run.selections = {
        "us-ak": ShardSelection("us-ak", ("us-ak",), ("us-ak/statutes/a.yaml",), ()),
        "us": ShardSelection("us", ("us",), (module,), ("us/statutes/x.test.yaml",)),
    }
    recorder = _patch_batch(monkeypatch)
    validate = ci_parity._gate_validate(run, _spec("validate", pin))
    companions = ci_parity._gate_companion_tests(run, _spec("companion_tests", pin))
    proof = ci_parity._gate_proof_validate(run, _spec("proof_validate", pin))

    assert all(result.status == "PASS" for result in (validate, companions, proof))
    assert run.validate_skip == run.retired_skip_by_shard
    assert (
        "[us] SKIPPED (known-validation-gaps validate_failures): " + module
        in validate.output
    )
    assert "[us] SKIPPED companion" in companions.output
    assert (
        "[us] SKIPPED (known-validation-gaps validate_failures): " + module
        in proof.output
    )
    assert all(
        str(repo / module) not in call.arguments for call in recorder.invocations
    )
    assert not any(call.arguments[0] == "test" for call in recorder.invocations)


@EMBEDDED_PINS
def test_retired_schema_head_failure_leaves_every_leg_skip_empty(
    tmp_path: Path, pin: str
) -> None:
    module = "us/statutes/x.yaml"
    repo = _repo(
        tmp_path,
        {module: RETIRED, FREEZE_PATH: _freeze({module: RETIRED})},
        remote=RULESPEC_US_REMOTE,
    )
    _commit(repo, {module: RETIRED + "# digest mismatch\n"})
    run = _run(repo, tmp_path, sha=pin, plan=_plan("us-ak", "us"))

    result = ci_parity._gate_retired_schema_freeze(
        run, _spec("retired_schema_freeze", pin)
    )

    assert result.status == "FAIL"
    assert run.retired_skip_by_shard == {
        "us-ak": frozenset(),
        "us": frozenset(),
    }


_skip_paths = st.sets(
    st.sampled_from(["us/a.yaml", "us/b.yaml", "us-ca/c.yml"]), max_size=3
)


@settings(max_examples=40, deadline=None)
@given(active=_skip_paths, first=_skip_paths, other=_skip_paths, failed=st.booleans())
def test_each_shard_skip_is_exactly_its_waivers_union_its_retired_list(
    active: set[str], first: set[str], other: set[str], failed: bool
) -> None:
    # For all active/first/nonfirst sets, successful SKIP is their union;
    # failure gives every leg an empty list. Cached waiver output is shared.
    with tempfile.TemporaryDirectory(prefix="axiom-shards-skip-") as name:
        repo = Path(name)
        (repo / "known-validation-gaps.yaml").write_text("ledger\n", encoding="utf-8")
        run = SimpleNamespace(
            repo=repo,
            workflow=SimpleNamespace(python=lambda *_args: "script"),
            retired_skip=frozenset(first),
            retired_skip_by_shard={
                "first": frozenset(first),
                "other": frozenset(other),
            },
        )
        with mock.patch.object(
            ci_parity,
            "_run_embedded_python",
            return_value=(
                1 if failed else 0,
                "\n".join(sorted(active)),
                "failure" if failed else "",
            ),
        ) as execute:
            skips = ci_parity._SkipList(run, "validate")
            first_skip, first_error = skips.build("first")
            other_skip, other_error = skips.build("other")
            fallback_skip, _ = skips.build("unmapped")
        assert execute.call_count == 1
        assert first_skip == (frozenset() if failed else active | first)
        assert other_skip == (frozenset() if failed else active | other)
        assert fallback_skip == first_skip
        assert bool(first_error) == bool(other_error) == failed


@EMBEDDED_PINS
def test_companion_groups_stop_after_first_failure_without_stopping_other_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str
) -> None:
    run = _module_run(
        tmp_path,
        {
            "__all__": (
                (),
                (
                    "statutes/a.test.yaml",
                    "policies/b.test.yaml",
                    "regulations/c.test.yaml",
                ),
            ),
            "us": ((), ("us/statutes/d.test.yaml",)),
        },
    )
    run.workflow = ci_parity.PinnedWorkflow(pin)
    recorder = _patch_batch(
        monkeypatch,
        lambda call: (
            (1, "first group failed\n")
            if Path(call.arguments[2]).name == "statutes"
            else (0, "ok\n")
        ),
    )

    result = ci_parity._gate_companion_tests(run, _spec("companion_tests", pin))

    assert result.status == "FAIL"
    assert [Path(call.arguments[2]).name for call in recorder.invocations] == [
        "statutes",
        "us",
    ]
    assert result.failures == ["[__all__] first group failed"]
    assert "bash associative-array order" in result.note
    assert "Each shard stops at its first failure." in result.note


@settings(max_examples=40, deadline=None)
@given(codes=st.lists(st.integers(min_value=0, max_value=3), min_size=1, max_size=8))
def test_companion_groups_execute_exactly_through_the_first_failure(
    codes: list[int],
) -> None:
    # For all group outcomes, run exactly the prefix through the first nonzero.
    groups = tuple(f"jurisdiction-{index}/a.test.yaml" for index in range(len(codes)))
    run = SimpleNamespace(
        plan=SimpleNamespace(matrix=("__all__",)),
        validate_skip={},
        selections={"__all__": SimpleNamespace(test_files=groups)},
        repo=Path("/synthetic/rulespec-us"),
        paths={
            "engine": Path("/synthetic/engine"),
            "rulespec_us": Path("/synthetic/us"),
        },
        jobs=1,
        keyring=(),
        signing_roots={},
    )
    executed: list[int] = []

    def batch(invocations, **_kwargs):
        results = []
        for call in invocations:
            index = int(Path(call.arguments[2]).name.removeprefix("jurisdiction-"))
            executed.append(index)
            results.append((codes[index], f"group {index}\n"))
        return results

    with mock.patch.object(ci_parity, "_run_cli_batch", batch):
        result = ci_parity._gate_companion_tests(run, _spec("companion_tests"))

    end = next((index + 1 for index, code in enumerate(codes) if code), len(codes))
    assert executed == list(range(end))
    assert (result.status == "FAIL") == any(codes)


@EMBEDDED_PINS
@pytest.mark.parametrize("codes", [(1, 1, 0), (0, 1, 1), (0, 0, 0)])
def test_companion_group_errexit_matches_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str, codes: tuple[int, ...]
) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")
    run = _module_run(
        tmp_path,
        {
            "__all__": (
                (),
                tuple(f"j{index}/a.test.yaml" for index in range(len(codes))),
            )
        },
    )
    observed: list[str] = []
    recorder = _patch_batch(
        monkeypatch,
        lambda call: (codes[int(Path(call.arguments[2]).name[1:])], "group\n"),
    )
    script = """set -euo pipefail
axiom_encode() {
  echo "$1"
  return "$2"
}
while [ "$#" -gt 0 ]; do
  axiom_encode "$1" "$2"
  shift 2
done
"""
    arguments = [
        item for index, code in enumerate(codes) for item in (f"j{index}", str(code))
    ]
    shell = subprocess.run(
        [bash, "--noprofile", "--norc", "-c", script, "groups", *arguments],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
        check=False,
    )
    result = ci_parity._gate_companion_tests(run, _spec("companion_tests", pin))
    observed.extend(Path(call.arguments[2]).name for call in recorder.invocations)

    assert observed == shell.stdout.splitlines()
    assert (result.status == "FAIL") == (shell.returncode != 0)


def test_run_ci_keeps_the_head_and_base_resolved_before_the_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 6f11be26 L157-158/L562-563, 0effa6a5 L157-158/L209-210: all
    # jobs receive the event's immutable commits, even if local refs move.
    harness = _CiHarness(tmp_path, monkeypatch)
    head, base = "c" * 40, "d" * 40

    def resolve_commit(repo: Path, ref: str) -> str:
        harness._record("resolve_commit", repo, ref)
        return {"HEAD": head, "origin/main": base}[ref]

    monkeypatch.setattr(ci_parity, "resolve_commit", resolve_commit)

    assert ci_parity.run_ci(harness.args()) == 0

    assert harness.simulation_kwargs["head"] == head
    assert harness.simulation_kwargs["base"] == base
    assert harness.calls["resolve_commit"] == [
        (harness.repo.resolve(), "HEAD"),
        (harness.repo.resolve(), "origin/main"),
    ]
    capsys.readouterr()


def test_run_ci_without_a_github_repository_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 6f11be26 L1141/0effa6a5 L788 require github.repository, the base
    # repository's slug; a host alias or mirror must supply an override.
    harness = _CiHarness(tmp_path, monkeypatch)
    monkeypatch.setattr(ci_parity, "resolve_github_repository", lambda *_: "")

    assert ci_parity.run_ci(harness.args()) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL"
    assert "--github-repository OWNER/NAME" in report["resolution_error"]
    assert report["gates"] == []
    assert "execute_workflow_gates" not in harness.calls


@pytest.mark.parametrize("pin_prefix", ["0effa6a5", "6f11be26"])
def test_shard_scope_and_selection_use_the_test_merge_not_the_pr_head(
    tmp_path: Path, pin_prefix: str
) -> None:
    import dataclasses

    from tests.test_ci_parity_workflow_gates import (
        PLAIN,
        _commit,
        _git,
        _plan,
        _repo,
        _run,
        _spec,
    )

    # 6f11be26 L563/L2144, 0effa6a5 L210/L1688: github.sha is
    # the test merge, whereas PR_HEAD_SHA remains the proposed head.
    pin = next(
        sha for sha in ci_parity.SUPPORTED_WORKFLOW_PINS if sha.startswith(pin_prefix)
    )
    repo = _repo(
        tmp_path,
        {"us/statutes/main.yaml": PLAIN, "us-ca/statutes/topic.yaml": PLAIN},
    )
    fork = _git(repo, "rev-parse", "HEAD")
    base = _commit(repo, {"us/statutes/main.yaml": PLAIN + "\n# base update\n"})
    _git(repo, "checkout", "-q", "-b", "topic", fork)
    head = _commit(repo, {"us-ca/statutes/topic.yaml": PLAIN + "\n# topic update\n"})

    with ci_parity.committed_checkout(
        repo, head, base, repository="TheAxiomFoundation/rulespec-us", merge=True
    ) as checkout:
        merge = _git(checkout, "rev-parse", "HEAD")
        run = _run(
            checkout, tmp_path, sha=pin, plan=_plan("us", "us-ca"), base_ref=base
        )
        run.simulation = dataclasses.replace(run.simulation, head_sha=head, sha=merge)
        run.plan = ci_parity.compute_shard_plan(
            run.workflow, checkout, run.simulation, "auto", False
        )

        assert run.plan.matrix == ("us-ca",)
        result = ci_parity._gate_select_targets(run, _spec("select_targets", pin))
        assert result.status == "PASS", result.output
        assert run.changed == ("us-ca/statutes/topic.yaml",)
        assert run.selections["us-ca"].rulespec_files == ("us-ca/statutes/topic.yaml",)


def test_retired_schema_unexpected_nonfirst_execution_failure_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _module_run(tmp_path, {"us-ak": ((), ()), "us": ((), ())})

    def execute(_script, *, cwd, argv, environment):
        if environment["IS_FIRST_SHARD"] == "true":
            Path(argv[0]).write_text("", encoding="utf-8")
            return 0, "Verified 0 immutable retired-schema module(s).\n", ""
        return 1, "", "python worker failed"

    monkeypatch.setattr(ci_parity, "_run_embedded_python", execute)
    result = ci_parity._gate_retired_schema_freeze(run, _spec("retired_schema_freeze"))
    assert result.status == "FAIL"
    assert result.failures == ["[us] python worker failed"]


def _path_waiver_record() -> dict[str, str]:
    return {
        "fingerprint": f"sha256:{'a' * 64}",
        "owner": "@MaxGhenis",
        "issue": "https://github.com/TheAxiomFoundation/rulespec-us/issues/1",
        "expires": "2026-01-31",
    }


@pytest.mark.parametrize(
    "pin", [workflow_gates.PIN_0EFFA6A5, workflow_gates.PIN_6F11BE26]
)
@pytest.mark.parametrize("path", ["us/statutes/A.yaml", "us/statutes/Sub/a.yaml"])
def test_waiver_audit_rejects_case_aliases_before_supervised_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pin: str, path: str
) -> None:
    entries = {path: {"active": _path_waiver_record()}}
    repo, _ = workflow_gates._waiver_repo(tmp_path, entries)
    workflow_gates._commit(repo, {"us/statutes/sub/a.yaml": workflow_gates.PLAIN})
    load_validation_waivers(
        repo / workflow_gates.LEDGER,
        today=date(2026, 1, 1),
        require_paths=False,
    )
    run = workflow_gates._run(repo, tmp_path, sha=pin, plan=workflow_gates._plan("us"))

    result, recorder = workflow_gates._waiver_gate(run, monkeypatch, pin)

    assert result.status == "FAIL", result.output
    assert f"validate_failures path does not exist: {path}" in result.output
    assert recorder.invocations == []


@pytest.mark.parametrize(
    "path", ["us/statutes/a.yaml", "us/statutes/A.yaml", "us/statutes/Sub/a.yaml"]
)
def test_waiver_path_spelling_matches_git_tree_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo, _ = workflow_gates._waiver_repo(
        tmp_path, {path: {"active": _path_waiver_record()}}
    )
    workflow_gates._commit(repo, {"us/statutes/sub/a.yaml": workflow_gates.PLAIN})
    load_validation_waivers(
        repo / workflow_gates.LEDGER,
        today=date(2026, 1, 1),
        require_paths=False,
    )
    run = workflow_gates._run(repo, tmp_path, plan=workflow_gates._plan("us"))
    # Git's exact tree lookup supplies the Linux checkout's path spelling,
    # regardless of whether the host filesystem merges case variants.
    git_lookup = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{path}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=ci_parity._git_environment(),
    )

    result, recorder = workflow_gates._waiver_gate(run, monkeypatch)

    assert (result.status == "PASS") == (git_lookup.returncode == 0)
    assert bool(recorder.invocations) == (git_lookup.returncode == 0)


def test_waiver_loader_can_accept_an_alias_on_a_case_insensitive_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Set case-insensitive lstat behavior explicitly, independent of the host.
    # This is the kernel behavior _gate_validation_waivers must compensate for.
    module = tmp_path / "us" / "statutes" / "foo.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    alias = module.with_name("Foo.yaml")
    ledger = tmp_path / "known-validation-gaps.yaml"
    expiry = (date(2026, 1, 1) + timedelta(days=30)).isoformat()
    ledger.write_text(
        "validate_failures:\n"
        "  us/statutes/Foo.yaml:\n"
        "    active:\n"
        f'      fingerprint: "sha256:{"a" * 64}"\n'
        '      owner: "@MaxGhenis"\n'
        '      issue: "https://github.com/TheAxiomFoundation/rulespec-us/issues/1"\n'
        f'      expires: "{expiry}"\n'
    )
    original_lstat = Path.lstat

    def case_insensitive_lstat(path: Path):
        return original_lstat(module if path == alias else path)

    monkeypatch.setattr(Path, "lstat", case_insensitive_lstat)

    loaded = load_validation_waivers(ledger, repo_root=tmp_path, today=date(2026, 1, 1))

    assert loaded.active_paths == {"us/statutes/Foo.yaml"}
    assert not ci_parity._exists_with_exact_case(tmp_path, "us/statutes/Foo.yaml", {})


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    name=st.text("abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12),
    uppercase=st.booleans(),
)
def test_waiver_audit_runs_only_for_exactly_spelled_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    uppercase: bool,
) -> None:
    # For every basename and either spelling, the audit runs iff every key
    # resolves with exact directory entry spelling. Audit work is mocked;
    # path spellings remain real directory entries on any host filesystem.
    root = tmp_path / name
    module = root / "us" / "statutes" / f"{name}.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("format: rulespec/v1\n")
    requested = f"us/statutes/{name.upper() if uppercase else name}.yaml"
    ledger = root / "known-validation-gaps.yaml"
    ledger.write_bytes(
        workflow_gates._ledger({requested: {"active": _path_waiver_record()}})
    )
    run = SimpleNamespace(
        repo=root,
        simulation=SimpleNamespace(base_sha="a" * 40, github_sha="b" * 40),
        caller=SimpleNamespace(inputs={"validation-waiver-bootstrap-sha256": ""}),
        temp=root,
        workflow=ci_parity.PinnedWorkflow(workflow_gates.PIN_0EFFA6A5),
        paths={"corpus": root / "corpus", "engine": root / "engine"},
        plan=workflow_gates._plan("us"),
        jobs=1,
        keyring=workflow_gates.KEYRING,
        signing_roots=None,
    )
    recorder = workflow_gates._AuditRecorder()
    with monkeypatch.context() as patch:
        patch.setattr(
            ci_parity.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=b"", stderr=b""
            ),
        )
        patch.setattr(ci_parity, "_run_embedded_python", lambda *a, **kw: (0, "", ""))
        patch.setattr(ci_parity, "_run_cli_batch", recorder)
        (root / "waiver-audit-changed-paths.txt").write_text("")

        result = ci_parity._gate_validation_waivers(
            run, workflow_gates._spec("validation_waivers")
        )

    assert (result.status == "FAIL") == uppercase
    assert bool(recorder.invocations) == (not uppercase)


def test_find_helpers_match_real_find_as_sets(tmp_path: Path) -> None:
    executable = shutil.which("find")
    if executable is None:
        pytest.skip("find is not installed")
    supported = subprocess.run(
        [executable, str(tmp_path), "-maxdepth", "0", "-print"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if supported.returncode:
        pytest.skip("find does not support the required scoped -maxdepth invocation")
    for name in ("us/z.yaml", "us/42/b.rac", "us/26/a.rac", "ignored/x.rac"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("file\n")
    (tmp_path / "us/link.rac").symlink_to("z.yaml")
    (tmp_path / "us/empty.rac").mkdir()
    regular = subprocess.run(
        [executable, "us", "-maxdepth", "8", "-type", "f"],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    named = subprocess.run(
        [
            executable,
            ".",
            "-maxdepth",
            "8",
            "(",
            "-path",
            "./ignored",
            ")",
            "-prune",
            "-o",
            "-name",
            "*.rac",
            "-print",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()

    assert set(ci_parity._find_regular_files(tmp_path, "us")) == set(regular)
    assert set(
        ci_parity._find_named(tmp_path, ("*.rac",), frozenset({"./ignored"}))
    ) == set(named)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    files=st.sets(
        st.tuples(
            st.sampled_from(["alpha", "beta/sub", "ignored"]),
            st.text("abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        ),
        max_size=12,
    )
)
def test_find_file_set_is_independent_of_directory_order_and_pruned_subtrees(
    tmp_path: Path, files: set[tuple[str, str]]
) -> None:
    # For every tree and pruned subtree, regular-file enumeration equals
    # exactly the unpruned files, regardless of directory listing order.
    identity = hashlib.sha256(repr(sorted(files)).encode()).hexdigest()
    root = tmp_path / identity
    root.mkdir(exist_ok=True)
    for directory, name in files:
        target = root / directory / f"{name}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("file\n")
    actual = set(
        ci_parity._find_regular_files(root, ".", prune=frozenset({"./ignored"}))
    )

    assert actual == {
        f"./{directory}/{name}.yaml"
        for directory, name in files
        if directory != "ignored"
    }
