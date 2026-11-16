"""Fresh local captures execute the captured form companion's exact cases.

The diagnostic fixture uses TEST corpus authority. The normal binary receipt
checks local drift; this test makes no historical or production claim.
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
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _authoritative_corpus_scope,
    _authoritative_rulespec_dependency_scope,
    _safe_load_unique_keys,
)
from tests.process_limits import HANG_TIMEOUT_SECONDS
from tests.test_cli import _bind_test_corpus_release

FIXTURE = Path(__file__).parent / "fixtures/bound_credit_excess/observed-graph.json"
ROOT = "ca/policies/cra/t1-2025/nl479-credits.yaml"
CITATION = "ca/policy/cra/t1-2025/nl479-credits"


class _NoBuildPipeline(ValidatorPipeline):
    def _axiom_rules_binary(self):
        return resolve_pinned_engine_binary(
            self.axiom_rules_path, self._declared_engine_pin(), allow_build=False
        )


@pytest.mark.parametrize("wrapped", [False, True], ids=["list", "cases-wrapper"])
def test_companion_edit_and_revert_executes_only_captured_form_cases(
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
            "Diagnostic form fixture",
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
        raw = next(
            item.raw for item in capture.files if item.relative.endswith(".test.yaml")
        )
        assert raw == original_bytes
        captured_cases = _safe_load_unique_keys(raw.decode())
        if wrapped:
            captured_cases = captured_cases["cases"]
        assert captured_cases == cases
        expected_keys = [case_key(case) for case in captured_cases]
        assert len(expected_keys) == 13
        saved = {}
        original_evidence = capture.evidence

        def retain_live_evidence():
            value = original_evidence()
            if value is not None:
                saved["evidence"] = value
                saved["fresh"] = capture.fresh()
            return value

        # Read evidence before normal CI removes its temporary compiled file.
        monkeypatch.setattr(capture, "evidence", retain_live_evidence)
        baseline = pipeline._run_rulespec_ci_impl(policy, execution_capture=capture)
        assert saved["fresh"] is True
        assert [r.case_sha256 for r in saved["evidence"].records] == expected_keys
        saved.clear()

        monkeypatch.setattr(
            ValidatorPipeline, "_run_rulespec_test_cases", restore_companion_and_run
        )
        companion.write_bytes(transient_bytes)
        result = pipeline._run_rulespec_ci_impl(policy, execution_capture=capture)

        assert result.passed == baseline.passed
        assert list(map(str, result.issues)) == list(map(str, baseline.issues))
        if runtime_pin == "89571cc2a938707fd60a5489b951135345697296":
            assert baseline.passed and result.passed
            assert not baseline.issues
            assert not result.issues
        assert supplied_cases == [cases]
        assert companion.read_bytes() == original_bytes
        assert (
            capture.snapshot_root / companion.relative_to(checkout)
        ).read_bytes() == original_bytes
        assert saved["fresh"] is True
        assert [r.case_sha256 for r in saved["evidence"].records] == expected_keys
        assert case_key(transient_cases[0]) not in expected_keys
