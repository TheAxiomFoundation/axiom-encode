"""Fresh diagnostic captures must credit the captured companion's exact cases.

The corpus fixture supplies TEST source authority only. The real engine's local
binding checks drift; this test makes no production or historical-authentication
claim.
"""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.engine_binding import resolve_pinned_engine_binary
from axiom_encode.harness.same_case_execution import case_key, compilation_snapshot
from axiom_encode.harness.student_loan_schedule_binding import (
    bind_student_loan_schedule,
)
from axiom_encode.harness.student_loan_schedule_execution import (
    _PROFILE_PIN,
    observe_student_loan_schedule,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _authoritative_corpus_scope,
    _authoritative_rulespec_dependency_scope,
    _safe_load_unique_keys,
)
from axiom_encode.student_loan_schedule_source import recognize_student_loan_schedule
from tests.process_limits import HANG_TIMEOUT_SECONDS
from tests.test_cli import _bind_test_corpus_release

FIXTURE = Path(__file__).parent / "fixtures/student_loan_schedule/observed-graph.json"
ROOT = "ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.yaml"
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest"


class _NoBuildPipeline(ValidatorPipeline):
    def _axiom_rules_binary(self):
        return resolve_pinned_engine_binary(
            self.axiom_rules_path, self._declared_engine_pin(), allow_build=False
        )


@pytest.mark.parametrize("wrapped", [False, True], ids=["list", "cases-wrapper"])
def test_companion_edit_and_revert_cannot_change_captured_cases(
    tmp_path, monkeypatch, real_engine, wrapped
):
    fixture = json.loads(FIXTURE.read_text())
    content, source, cases = (
        fixture["content"],
        fixture["source"],
        fixture["cases"],
    )
    transient_cases = copy.deepcopy(cases)
    transient_cases[0]["name"] += "-transient-untracked-case"

    def companion_bytes(values):
        return yaml.safe_dump(
            {"cases": values} if wrapped else values, sort_keys=False
        ).encode()

    original_bytes = companion_bytes(cases)
    transient_bytes = companion_bytes(transient_cases)
    checkout = tmp_path.resolve() / "rulespec-ca"
    policy = checkout / ROOT
    policy.parent.mkdir(parents=True)
    policy.write_text(content)
    companion = policy.with_suffix(".test.yaml")
    companion.write_bytes(original_bytes)
    release = _bind_test_corpus_release(
        checkout,
        tmp_path.resolve() / "axiom-corpus",
        citation_path=CITATION,
        body=source,
    )
    runtime_pin = subprocess.check_output(
        ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"],
        text=True,
        timeout=HANG_TIMEOUT_SECONDS,
    ).strip()
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Test fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Diagnostic student-loan fixture",
        ],
    ):
        subprocess.run(
            ["git", *arguments],
            cwd=checkout,
            capture_output=True,
            check=True,
            timeout=HANG_TIMEOUT_SECONDS,
        )

    pipeline = _NoBuildPipeline(
        checkout / "ca",
        real_engine.root,
        local_corpus_release=release,
        enable_oracles=False,
        max_workers=1,
        require_complete_source_unit=True,
        source_text=source,
        source_citation_path=CITATION,
        axiom_rules_engine_ref=runtime_pin,
    )
    assert pipeline._declared_engine_pin().sha == runtime_pin
    assert pipeline._axiom_rules_binary() == real_engine.binary

    supplied_cases = []
    run_test_cases = ValidatorPipeline._run_rulespec_test_cases

    def restore_companion_and_run(self, **kwargs):
        # Revert the original file after parsing, before ordinary execution.
        # Do not replace the cases supplied by the caller.
        assert companion.read_bytes() == transient_bytes
        supplied_cases.append(copy.deepcopy(kwargs["cases"]))
        companion.write_bytes(original_bytes)
        return run_test_cases(self, **kwargs)

    with (
        _authoritative_corpus_scope(release),
        _authoritative_rulespec_dependency_scope(()),
        compilation_snapshot(pipeline, policy) as capture,
    ):
        assert capture is not None
        assert len(capture.files) == 2
        captured_bytes = next(
            item.raw for item in capture.files if item.relative.endswith(".test.yaml")
        )
        assert captured_bytes == original_bytes
        captured_cases = _safe_load_unique_keys(captured_bytes.decode())
        if wrapped:
            captured_cases = captured_cases["cases"]
        assert captured_cases == cases
        expected_keys = [case_key(case) for case in captured_cases]
        saved = {}
        original_evidence = capture.evidence

        def retain_live_evidence():
            value = original_evidence()
            if value is not None:
                saved["evidence"] = value
                saved["fresh"] = capture.fresh()
            return value

        # CI deletes its temporary compiled artifact before returning. Retain
        # the unchanged evidence while the normal pipeline consumes it.
        monkeypatch.setattr(capture, "evidence", retain_live_evidence)
        baseline = pipeline._run_rulespec_ci_impl(policy, execution_capture=capture)
        assert saved["fresh"] is True
        assert [
            record.case_sha256 for record in saved["evidence"].records
        ] == expected_keys
        assert len(expected_keys) == 18
        saved.clear()
        monkeypatch.setattr(
            ValidatorPipeline, "_run_rulespec_test_cases", restore_companion_and_run
        )
        companion.write_bytes(transient_bytes)
        result = pipeline._run_rulespec_ci_impl(policy, execution_capture=capture)

        assert list(map(str, result.issues)) == list(map(str, baseline.issues))
        assert supplied_cases == [cases]
        assert companion.read_bytes() == original_bytes
        assert (
            capture.snapshot_root / companion.relative_to(checkout)
        ).read_bytes() == original_bytes
        assert saved["fresh"] is True
        evidence = saved["evidence"]
        assert [record.case_sha256 for record in evidence.records] == expected_keys
        binding = bind_student_loan_schedule(
            recognize_student_loan_schedule(source, CITATION), content
        )
        assert binding is not None
        observation = observe_student_loan_schedule(
            binding, evidence, content, captured_cases
        )
        if runtime_pin == _PROFILE_PIN:
            assert baseline.passed and result.passed, result.issues
            assert result.issues == []
            assert observation is not None
        else:
            # The snapshot-read regression runs at CI's engine pin too. Only
            # the observer's audited compiler profile can grant source credit.
            assert not baseline.passed and not result.passed
            assert observation is None
        assert (
            observe_student_loan_schedule(binding, evidence, content, transient_cases)
            is None
        )
