"""Use real checkout closure planning; fixture sources are not release admission."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness.same_case_execution import (
    _materialize_snapshot,
    _snapshot_plan,
)
from tests.test_same_page_floor_binding import BODY, CITATION, CONTENT


@pytest.fixture
def local_plan(tmp_path):
    root = tmp_path / "rulespec-ca"
    policy = root / "ca"
    rules = policy / "policies/cra/t1-2025/federal-spouse-transfer-amounts.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(CONTENT)
    tests = rules.with_suffix(".test.yaml")
    tests.write_text("tests: []\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    binary = tmp_path / "fixture-binary"
    binary.write_bytes(b"not executable; snapshot planning test only")
    pipe = SimpleNamespace(
        require_complete_source_unit=True,
        _declared_engine_pin=lambda: SimpleNamespace(sha="a" * 40),
        _rulespec_compile_roots=lambda: [root],
        _rulespec_test_path=lambda _: tests,
        rulespec_dependency_roots=(),
        policy_repo_path=policy,
        _source_texts_for_rulespec_content=lambda _: {CITATION: BODY},
        _proof_source_texts_for_rulespec_content=lambda _, source_texts: source_texts,
        _axiom_rules_binary=lambda: binary,
    )
    return pipe, rules


def test_no_import_closed_floor_uses_existing_materialized_snapshot(
    local_plan, tmp_path
):
    pipe, rules = local_plan
    plan = _snapshot_plan(pipe, rules)
    assert plan is not None
    assert len(plan[1]) == 2  # exact root and companion; no invented imports
    capture = _materialize_snapshot(pipe, rules, plan, str(tmp_path / "capture"))
    assert capture is not None
    assert capture.evidence() is None  # no compilation, tests or actual engine records


@pytest.mark.parametrize(
    "change", ["unknown_source", "dependency_root", "no_companion", "unknown_module"]
)
def test_other_no_import_modules_still_do_not_get_capture(local_plan, change):
    pipe, rules = local_plan
    if change == "unknown_source":
        pipe._source_texts_for_rulespec_content = lambda _: {
            CITATION: BODY.replace("90 days", "89 days")
        }
    elif change == "dependency_root":
        pipe.rulespec_dependency_roots = (Path("/unrelated"),)
    elif change == "no_companion":
        pipe._rulespec_test_path(rules).unlink()
    else:
        rules.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    assert _snapshot_plan(pipe, rules) is None
