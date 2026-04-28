"""Tests for model comparison eval helpers."""

import json
import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

from axiom_encode.harness.evals import (
    EvalArtifactMetrics,
    EvalReadinessGates,
    EvalResult,
    EvalSuiteCase,
    EvalSuiteManifest,
    GroundingMetric,
    _build_eval_prompt,
    _clean_generated_file_content,
    _command_looks_out_of_bounds,
    _eval_result_from_payload,
    _hydrate_eval_root,
    _is_single_amount_table_slice,
    _materialize_eval_artifact,
    _normalize_nonannual_test_period_value,
    _normalize_test_periods_to_effective_dates,
    _post_openai_eval_request,
    _run_codex_prompt_eval,
    _wait_for_codex_process,
    evaluate_artifact,
    load_eval_suite_manifest,
    parse_runner_spec,
    prepare_eval_workspace,
    run_eval_suite,
    run_source_eval,
    select_context_files,
    summarize_readiness,
)
from axiom_encode.harness.validator_pipeline import ValidationResult, ValidatorPipeline


@pytest.fixture(autouse=True)
def _mock_generalist_reviewer():
    """Keep eval tests deterministic unless they explicitly inspect reviewer behavior."""
    with patch.object(
        ValidatorPipeline,
        "_run_reviewer",
        return_value=ValidationResult(
            "generalist-reviewer", True, score=8.0, issues=[]
        ),
    ):
        yield


class TestParseRunnerSpec:
    def test_parses_named_runner(self):
        runner = parse_runner_spec("gpt=codex:gpt-5.4")
        assert runner.name == "gpt"
        assert runner.backend == "codex"
        assert runner.model == "gpt-5.4"

    def test_parses_default_name(self):
        runner = parse_runner_spec("claude:opus")
        assert runner.name == "claude-opus"
        assert runner.backend == "claude"
        assert runner.model == "opus"

    def test_parses_openai_runner(self):
        runner = parse_runner_spec("openai:gpt-5.4")
        assert runner.name == "openai-gpt-5.4"
        assert runner.backend == "openai"
        assert runner.model == "gpt-5.4"


class TestCodexPromptEval:
    def test_wait_for_codex_process_terminates_after_stable_last_message(
        self, tmp_path
    ):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True


def test_build_eval_prompt_targets_rulespec_yaml(tmp_path):
    runner = parse_runner_spec("codex:gpt-5.4")
    workspace = prepare_eval_workspace(
        citation="tn_snap_standard_utility_allowance",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="The standard utility allowance is $451.",
        axiom_rules_path=tmp_path / "us-tn",
        mode="cold",
    )

    prompt = _build_eval_prompt(
        "tn_snap_standard_utility_allowance",
        "cold",
        workspace,
        [],
        target_file_name="tn-snap-standard-utility-allowance.yaml",
        include_tests=True,
        policyengine_rule_hint="snap_standard_utility_allowance",
    )

    assert "format: rulespec/v1" in prompt
    assert "RuleSpec YAML" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.yaml ===" in prompt
    assert "=== FILE: tn-snap-standard-utility-allowance.test.yaml ===" in prompt
    assert "snap_standard_utility_allowance" in prompt
    assert "Do not use bare year periods like `2024`" in prompt
    assert "period_kind: custom" in prompt
    assert (
        "Do not create named `parameter` rules for structural table row labels"
        in prompt
    )


def test_materialize_eval_artifact_writes_rulespec_bundle(tmp_path):
    output_file = tmp_path / "runner" / "source" / "tn-snap.yaml"
    llm_response = """=== FILE: tn-snap.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
rules:
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
=== FILE: tn-snap.test.yaml ===
- name: base
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

    wrote = _materialize_eval_artifact(
        llm_response,
        output_file,
        source_text="The standard utility allowance is $451.",
    )

    assert wrote is True
    assert output_file.exists()
    assert output_file.with_name("tn-snap.test.yaml").exists()
    assert output_file.read_text().startswith("format: rulespec/v1")


def test_eval_result_payload_round_trips_prompt_digests():
    result = EvalResult(
        citation="snap_test",
        runner="codex-gpt-5.4",
        backend="codex",
        model="gpt-5.4",
        mode="repo-augmented",
        output_file="/tmp/snap_test.yaml",
        trace_file="/tmp/snap_test.trace.json",
        context_manifest_file="/tmp/snap_test.context.json",
        duration_ms=1234,
        success=True,
        error=None,
        input_tokens=11,
        output_tokens=22,
        cache_read_tokens=33,
        cache_creation_tokens=44,
        reasoning_output_tokens=55,
        estimated_cost_usd=0.12,
        actual_cost_usd=None,
        retrieved_files=["/tmp/context.yaml"],
        unexpected_accesses=[],
        metrics=EvalArtifactMetrics(
            compile_pass=True,
            compile_issues=[],
            ci_pass=True,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1,
            ungrounded_numeric_count=0,
            grounding=[],
            generalist_review_pass=True,
            generalist_review_score=9.0,
            generalist_review_issues=[],
            generalist_review_prompt_sha256="review-digest",
            policyengine_pass=True,
            policyengine_score=1.0,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        ),
        generation_prompt_sha256="generation-digest",
    )

    restored = _eval_result_from_payload(result.to_dict())

    assert restored.generation_prompt_sha256 == "generation-digest"
    assert restored.metrics is not None
    assert restored.metrics.generalist_review_prompt_sha256 == "review-digest"

    def test_wait_for_codex_process_terminates_after_persistent_output(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        last_message.write_text("ready\n")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                if self.returncode is None:
                    last_message.touch()
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        terminated = _wait_for_codex_process(
            process,
            last_message,
            timeout=1,
            settle_seconds=1,
            max_output_wait_seconds=0,
            poll_interval=0,
        )

        assert terminated is True
        assert process.terminated is True

    def test_wait_for_codex_process_times_out_when_heartbeat_stalls(self, tmp_path):
        last_message = tmp_path / ".codex-last-message.txt"
        stdout_path = tmp_path / "stdout.log"
        stdout_path.write_text("")

        class FakeProcess:
            def __init__(self):
                self.args = ["codex", "exec"]
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        with pytest.raises(subprocess.TimeoutExpired):
            _wait_for_codex_process(
                process,
                last_message,
                timeout=1,
                heartbeat_paths=[stdout_path],
                max_idle_seconds=0,
                poll_interval=0,
            )

        assert process.terminated is True

    def test_run_codex_prompt_eval_accepts_stable_last_message_on_termination(
        self, tmp_path
    ):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/3/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="nil amount",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"
        event_lines = "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "fallback"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "cached_input_tokens": 0,
                        },
                    }
                ),
            ]
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)
                stdout.write(event_lines + "\n")
                stdout.flush()

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            process.terminate()
            process.wait()
            return True

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()
        assert response.tokens is not None
        assert response.tokens.input_tokens == 10
        assert response.tokens.output_tokens == 4

    def test_run_codex_prompt_eval_salvages_last_message_on_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4A/1",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="maximum disregard",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = "=== FILE: example.yaml ===\nformat: rulespec/v1\nrules: []\n"

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = None
                Path(cwd, ".codex-last-message.txt").write_text(bundle)

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["codex", "exec"], timeout=600
                ),
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()

    def test_run_codex_prompt_eval_uses_longer_idle_timeout(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="10-ccr-2506-1/4.403.11/b/c/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="self-employment expenses",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        class FakePopen:
            def __init__(self, cmd, stdout, stderr, text, cwd):
                self.args = cmd
                self.returncode = 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        observed: dict[str, float] = {}

        def fake_wait(
            process,
            last_message_file,
            timeout,
            heartbeat_paths=None,
            settle_seconds=5.0,
            max_output_wait_seconds=30.0,
            max_idle_seconds=120.0,
            poll_interval=0.5,
        ):
            observed["max_idle_seconds"] = max_idle_seconds
            process.returncode = 0
            return False

        with (
            patch("axiom_encode.harness.evals.subprocess.Popen", FakePopen),
            patch(
                "axiom_encode.harness.evals._wait_for_codex_process",
                side_effect=fake_wait,
            ),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert observed["max_idle_seconds"] == 300
        assert response.text == ""


class TestEvaluateArtifact:
    def test_uses_fallback_source_text_for_grounding(self, tmp_path):
        rulespec_file = tmp_path / "24" / "a.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: (a) Allowance of credit There shall be allowed a credit of $1,000.\n"
            "rules:\n"
            "  - name: ctc_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2025-01-01'\n"
            "        formula: 2200\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)

        with (
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text=(
                    "(a) Allowance of credit There shall be allowed a credit of $1,000."
                ),
            )

        assert metrics.compile_pass
        assert not metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 1
        assert [item.raw for item in metrics.grounding if not item.grounded] == ["2200"]
        assert any(
            "Ungrounded generated numeric literal" in issue and "2200" in issue
            for issue in metrics.ci_issues
        )

    def test_fails_ci_when_repeated_source_scalar_has_only_one_named_definition(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    2A. Where earnings are less than £20 in any week and would not exceed £20.
rules:
  - name: pc_special_employment_maximum_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2025-03-31'
        formula: 20
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text=(
                    "2A. Where earnings are less than £20 in any week and "
                    "would not exceed £20."
                ),
            )

        assert metrics.compile_pass
        assert not metrics.ci_pass
        assert any(
            "appears 2 time(s), but only 1 named scalar definition(s)" in issue
            for issue in metrics.ci_issues
        )

    def test_accepts_pence_threshold_grounded_as_decimal_gbp(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    13. Small amounts of state pension credit

    Where the amount of state pension credit payable is less than 10 pence per week,
    the credit shall not be payable unless the claimant is in receipt of another benefit
    payable with the credit.
rules:
  - name: small_amount_threshold
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2025-03-21'
        formula: 0.10
  - name: amount_payable
    kind: input
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
  - name: is_payable
    kind: derived
    entity: Person
    dtype: Boolean
    period: Week
    versions:
      - effective_from: '2025-03-21'
        formula: amount_payable >= small_amount_threshold
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text=(
                    "Where the amount of state pension credit payable is less than "
                    "10 pence per week, the credit shall not be payable."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_runs_generalist_reviewer_and_records_result(self, tmp_path):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: Provision text with £10.
rules:
  - name: example_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: 10
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            False,
            score=4.5,
            issues=["Merged distinct statutory branches."],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            metrics = evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text="Provision text with £10.",
            )

        assert metrics.compile_pass is True
        assert metrics.ci_pass is True
        assert metrics.generalist_review_pass is False
        assert metrics.generalist_review_score == 4.5
        assert metrics.generalist_review_issues == [
            "Merged distinct statutory branches."
        ]
        mock_reviewer.assert_called_once()
        assert mock_reviewer.call_args.args[0] == "generalist-reviewer"
        assert "atomic source slice" in mock_reviewer.call_args.kwargs["review_context"]
        assert (
            "stale, generic, or misleading"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_timing_clause_review_context_mentions_boolean_day_predicate(
        self, tmp_path
    ):
        rulespec_file = tmp_path / "example.yaml"
        rulespec_file.write_text(
            """format: rulespec/v1
module:
  summary: On the first day of the next benefit week.
rules:
  - name: example_timing_rule
    kind: parameter
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-01-01'
        formula: true
"""
        )

        compile_result = ValidationResult("compile", True, issues=[])
        ci_result = ValidationResult("ci", True, issues=[])
        reviewer_result = ValidationResult(
            "generalist-reviewer",
            True,
            score=8.0,
            issues=[],
        )

        with (
            patch.object(
                ValidatorPipeline, "_run_compile_check", return_value=compile_result
            ),
            patch.object(ValidatorPipeline, "_run_ci", return_value=ci_result),
            patch.object(
                ValidatorPipeline, "_run_reviewer", return_value=reviewer_result
            ) as mock_reviewer,
        ):
            evaluate_artifact(
                rulespec_file=rulespec_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text="On the first day of the next benefit week.",
            )

        assert (
            "boolean day-predicate helper"
            in mock_reviewer.call_args.kwargs["review_context"]
        )

    def test_build_eval_prompt_for_uk_timing_leaf_discourages_invented_day_offsets(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the assessed amount comprises income from capital, it shall be "
                "deemed to increase or decrease on the first day of the next benefit "
                "week to commence on or after the day on which the income increases "
                "or decreases."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not convert relative temporal phrases" in prompt
        assert "`*_offset = 1`" in prompt

    def test_build_eval_prompt_for_atomic_conjunctive_branch_discourages_normative_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the Secretary of State is informed that the arrangements under "
                "which the assessed amount is paid contains provision—\n\n"
                "(b)\n\n"
                "for the date on which the increase is to be paid; and"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "atomic conjunctive branch slices" in prompt
        assert "do not pretend to encode the whole parent consequence" in prompt
        assert "avoid standalone normative names like `..._must_...`" in prompt
        assert "do not make the principal output a bare input stub" in prompt
        assert "feed the asserted output back into `input:`" in prompt
        assert "treat the carve-out as displacing this slice" in prompt

    def test_build_eval_prompt_for_comparative_month_apart_phrase_discourages_numeric_thresholds(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the last four payments if the last two payments are less than one month apart; or"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "less than one month apart" in prompt
        assert "one_month_threshold = 1" in prompt
        assert "the `one month` comparator is not a standalone numeric scalar" in prompt
        assert "do not invent `1`-valued threshold/count helpers" in prompt
        assert (
            "branch-specific output is a `Count` or other non-Boolean basis selector"
            in prompt
        )
        assert "do not write an inline conditional without `else`" in prompt
        assert (
            "negative tests should usually assert only the `_applies` boolean" in prompt
        )
        assert (
            "expect the principal basis-count output to remain the active legal basis"
            in prompt
        )
        assert "trigger decomposed-date CI failures" in prompt

    def test_build_eval_prompt_for_single_payment_period_discourages_parallel_units(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "where the period in respect of which a payment is made exceeds a week, "
                "and in a case where that period is three months, the amount is calculated ..."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "keep one canonical fact or classification for that single period" in prompt
        )
        assert "parallel free inputs like `*_in_weeks` and `*_in_months`" in prompt
        assert "do not require a second independent duration input" in prompt
        assert (
            "do not feed the same legal period through contradictory units or categories"
            in prompt
        )

    def test_build_eval_prompt_for_amount_included_determination_requires_applicability_bound_money_output(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(ii)\n\n"
                "in a case where that period is three months, by multiplying the amount of the payment by 4 and dividing the product by 52;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not leave that money output unconditional" in prompt
        assert "typically with an explicit `else: 0`" in prompt
        assert (
            "paragraph-level exceptions or a different payment period displace the limb"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_includes_leaf_discourages_blanket_negation(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                'Subject to paragraphs (3), (4) and (4A), "earnings" in the case '
                "of employment as an employed earner, means any remuneration or "
                "profit derived from that employment and includes—\n\n"
                "(a)\n\n"
                "any bonus or commission;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Subject to paragraphs (3), (4) and (4A), ... includes—" in prompt
        assert "blanket negating gate" in prompt
        assert "Do not make a composite `subject_to_*_satisfied`" in prompt
        assert "branch-specific fact gate" in prompt
        assert "permits this branch to count" in prompt
        assert (
            "do not collapse all cited qualifications into one opaque helper" in prompt
        )
        assert (
            "one paragraph-specific qualification input or import per cited paragraph"
            in prompt
        )

    def test_build_eval_prompt_for_payment_level_slice_discourages_blind_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, where the period in respect "
                "of which a payment is made does not exceed a week, the whole of that "
                "payment shall be included in the claimant's weekly income."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "individual payment or `that payment`" in prompt
        assert "preserve that payment-scoped subject" in prompt
        assert "prefer `entity: Payment`" in prompt
        assert "prefer `entity: Asset`" in prompt
        assert "provide per-payment rows under `tables:`" in prompt
        assert "provide per-item rows under `tables:`" in prompt
        assert "exact entity name `Payment:`" in prompt
        assert "Use `status: entity_not_supported`" in prompt
        assert "only as a last resort" in prompt
        assert "Do not prefer that fallback" in prompt

    def test_build_eval_prompt_for_except_where_and_citations_discourages_joint_exception(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Except where paragraph (2) and (4) apply, the amount to be included "
                "shall be determined—"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Except where paragraph (2) and (4) apply" in prompt
        assert (
            "do not assume the exception is displaced only when both cited paragraphs apply simultaneously"
            in prompt
        )
        assert (
            "treat the slice as inoperative when any cited paragraph applies" in prompt
        )

    def test_build_eval_prompt_for_payable_phrase_preserves_payability_fact(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "model payability as the legal fact" in prompt
        assert "Do not replace `payable` with `receives` or `received`" in prompt

    def test_build_eval_prompt_for_regular_pattern_clause_preserves_full_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the claimant's regular pattern of work is such that he does not "
                "work the same hours every week;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "regular pattern of work is such that" in prompt
        assert (
            "Do not shorten the branch to only `does not work the same hours every week`"
            in prompt
        )

    def test_build_eval_prompt_for_enumerated_payments_discourages_or_collapse(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "statutory sick pay and statutory maternity pay payable by the "
                "employer under the 1992 Act;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not collapse them into a single `x_or_y` principal output" in prompt
        assert "statutory_sick_pay_or_statutory_maternity_pay_*" in prompt

    def test_build_eval_prompt_for_branch_slice_preserves_binding_lead_in_conjuncts(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/10",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "This paragraph applies where the period which—\n\n"
                "is a period of the same length as the period in respect of which "
                "the last payment of the pre-increase assessed amount was made.\n\n"
                "(b)\n\n"
                "ends on the first increased payment date,"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "distinguish mere placement context from binding lead-in conjuncts"
            in prompt
        )
        assert "preserve both conjuncts" in prompt
        assert "do not drop the same-length requirement" in prompt

    def test_build_eval_prompt_for_where_on_branch_discourages_material_implication(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13B",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "All benefits except those mentioned in paragraph (1) shall be treated as paid—\n\n"
                "(b)\n\n"
                "where the benefit is paid in arrears, on the last day of the benefit week "
                "in which the benefit is payable."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13B",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13B.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat `X` and `Y` as a positive conjunction for this branch" in prompt
        assert "Do not rewrite that as material implication like `not X or Y`" in prompt
        assert (
            "if the branch-triggering condition itself is false, the branch-specific output should usually be `false`"
            in prompt
        )

    def test_build_eval_prompt_for_disjunctive_payment_description_preserves_qualifier_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(1)(j) (income to include income of prescribed descriptions), "
                "income of the following descriptions is prescribed—\n\n"
                "(ac)\n\n"
                "any retired pay, pension or allowance granted in respect of disablement or any pension or "
                "allowance granted to a widow, widower or surviving civil partner in respect of a death due to "
                "service or war injury under an instrument specified in section 639(2) of the Income Tax "
                "(Earnings and Pensions) Act 2003, where such payment does not fall within paragraph (a) of the "
                "definition of “war disablement pension” in section 17(1) of the State Pension Credit Act 2002 or, "
                "in respect of any retired pay or pension granted in respect of disablement, where such payment "
                "does not fall within paragraph (b) of that definition;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies"
            in prompt
        )
        assert (
            "do not narrow the first `where ...` clause to only the later-mentioned category"
            in prompt
        )
        assert "preserve the paragraph-(a) path for retired pay and pension" in prompt

    def test_build_eval_prompt_for_royalties_slice_preserves_consideration_scope(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "PART III Income\n\n"
                "17. Calculation of weekly income\n\n"
                "(5)\n\n"
                "This paragraph applies to—\n\n"
                "(a)\n\n"
                "royalties or other sums received as a consideration for the use of, or the "
                "right to use, any copyright, design, patent or trade mark;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the consideration-for-use/right-to-use qualifier across both `royalties` and `other sums`"
            in prompt
        )
        assert "do not model `royalty` as a free-standing qualifying limb" in prompt
        assert "a bare `payment_is_royalty` fact is too broad" in prompt

    def test_build_eval_prompt_for_employed_earner_definition_preserves_shared_qualifier(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/5",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "17A. Earnings of an employed earner\n\n"
                "(5)\n\n"
                "In this regulation “employed earner” means a person who is gainfully "
                "employed in Great Britain either under a contract of service, or in an "
                "office (including elective office) with emoluments chargeable to income "
                "tax under Schedule E."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/5",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-5.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "preserve the shared qualifying employment/office across each alternative limb"
            in prompt
        )
        assert (
            "do not decompose the rule into one free-standing `person_is_X` fact plus separate `under_A` and `in_B` facts"
            in prompt
        )
        assert (
            "distribute the shared qualifier across the alternatives with branch-specific combined facts"
            in prompt
        )

    def test_build_eval_prompt_for_complete_capital_bands_discourages_fractional_division(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/15",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of section 15(2) (deemed income from capital) and subject to "
                "regulation 17(8) (capital to be disregarded), a claimant’s capital shall be "
                "deemed to yield a weekly income of—\n\n"
                "(a)\n\n"
                "£1 for each £500 in excess of £10,000; and"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat `for each £500` as counting complete bands, not proportional fractions"
            in prompt
        )
        assert "derive the band count with `floor(excess / band_size)`" in prompt
        assert (
            "include a non-exact-multiple excess case like `£750` above threshold"
            in prompt
        )


class TestGeneratedBundleCleaning:
    def test_clean_generated_file_content_strips_fence_and_trailing_prose(self):
        content = (
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "The encoding captures the enhanced rate."
        )

        cleaned = _clean_generated_file_content(content)

        assert cleaned == (
            "- name: base\n  output:\n    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_clean_generated_file_content_strips_inline_currency_suffixes(self):
        content = (
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is 26.05 GBP.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05 GBP\n"
        )

        cleaned = _clean_generated_file_content(content)

        assert "26.05 GBP" not in cleaned.split("formula:", 1)[1]
        assert "formula: 26.05" in cleaned

    def test_materialize_eval_artifact_rejects_non_rulespec_bundle(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = (
            "=== FILE: example.yaml ===\nrules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_eval_artifact_cleans_bundled_rulespec_fences(self, tmp_path):
        output_file = tmp_path / "source" / "uksi-2006-965-regulation-2.yaml"
        llm_response = (
            "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
            "```yaml\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
            "```\n"
            "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "Trailing prose.\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "child_benefit_enhanced_rate: 26.05" in test_text
        assert "period: '2025-04-07'" in test_text

    def test_materialize_eval_artifact_salvages_rulespec_workspace_files_when_response_is_summary(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-2025-03-31.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The weekly amount is £19.30.\n"
            "rules:\n"
            "  - name: pc_housing_non_dependant_deduction_other_weekly_amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-03-21'\n"
            "        formula: 19.30\n"
        )
        (workspace_root / output_file.with_suffix(".test.yaml").name).write_text(
            "- name: base_case\n"
            "  period: 2025-04-01\n"
            "  input: {}\n"
            "  output:\n"
            "    pc_housing_non_dependant_deduction_other_weekly_amount: 19.30\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            source_text="£19.30",
            workspace_root=workspace_root,
        )

        assert wrote is True
        assert output_file.read_text().startswith("format: rulespec/v1\n")
        assert output_file.with_suffix(".test.yaml").exists()

    def test_materialize_eval_artifact_rejects_non_rulespec_workspace_main(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "rules:\n  - name: missing_format_header\n"
        )

        wrote = _materialize_eval_artifact(
            "Both files written.",
            output_file,
            workspace_root=workspace_root,
        )

        assert wrote is False
        assert not output_file.exists()

    def test_materialize_eval_artifact_normalizes_rulespec_test_periods(self, tmp_path):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451, effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
=== FILE: example.test.yaml ===
- name: pre_effective_zero
  period: 2025-09
  input: {}
  output:
    snap_standard_utility_allowance: 0
- name: applies
  period: 2026-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert "pre_effective_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_materialize_eval_artifact_normalizes_mapping_style_tests_to_list(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-10-5-b-ii.yaml"
        response = """=== FILE: uksi-2002-1792-regulation-10-5-b-ii.yaml ===
format: rulespec/v1
module:
  summary: The day referred to in branch ii is true when the condition applies.
rules:
  - name: day_referred_to_10_5_b_ii
    kind: derived
    entity: Person
    dtype: Boolean
    period: Day
    versions:
      - effective_from: '2025-03-21'
        formula: some_fact
=== FILE: uksi-2002-1792-regulation-10-5-b-ii.test.yaml ===
case_branch_ii_applies:
  period: 2025-03-21
  input:
    some_fact: true
  output:
    day_referred_to_10_5_b_ii: true
"""

        wrote = _materialize_eval_artifact(response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".test.yaml").read_text()
        assert test_text.lstrip().startswith("- ")
        assert "name: case_branch_ii_applies" in test_text
        assert "case_branch_ii_applies:" not in test_text

    def test_normalize_test_periods_drops_speculative_pre_effective_zero_case_for_monthly_update(
        self,
    ):
        rulespec_text = """format: rulespec/v1
module:
  summary: The SUA is $451 effective October 1, 2025.
rules:
  - name: snap_standard_utility_allowance
    kind: parameter
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 451
"""
        source_text = (
            "Current-effective Tennessee utility allowance slice.\n\n"
            "The Standard Utility Allowance (SUA) is used when the household is\n"
            "responsible for heating or cooling costs.\n"
            "The SUA is $451, effective October 1, 2025.\n"
        )
        test_text = _normalize_test_periods_to_effective_dates(
            "- name: applies\n"
            "  period: 2026-01\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 451\n"
            "- name: pre_effective_month_zero\n"
            "  period: 2025-09\n"
            "  output:\n"
            "    snap_standard_utility_allowance: 0\n",
            rulespec_content=rulespec_text,
            source_text=source_text,
        )

        assert "pre_effective_month_zero" not in test_text
        assert "period: 2026-01" in test_text

    def test_materialize_eval_artifact_adds_missing_oracle_hint_output_from_rulespec(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "example.yaml"
        response = """=== FILE: example.yaml ===
format: rulespec/v1
module:
  summary: Homeless Shelter Deduction - $198.99.
rules:
  - name: snap_homeless_shelter_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: 198.99
  - name: snap_homeless_shelter_deduction_available
    kind: parameter
    dtype: Boolean
    versions:
      - effective_from: '2025-10-01'
        formula: true
=== FILE: example.test.yaml ===
- name: base
  period: 2025-10
  output:
    snap_homeless_shelter_deduction_amount: 198.99
"""

        wrote = _materialize_eval_artifact(
            response,
            output_file,
            policyengine_rule_hint="snap_homeless_shelter_deduction_available",
        )

        assert wrote is True
        test_payload = yaml.safe_load(output_file.with_suffix(".test.yaml").read_text())
        assert (
            test_payload[0]["output"]["snap_homeless_shelter_deduction_available"]
            is True
        )

    def test_can_include_policyengine_metrics_for_uk_artifact(self, tmp_path):
        rules_file = tmp_path / "source" / "uksi-2006-965-regulation-2.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states the enhanced rate is £26.05.\n"
            "rules:\n"
            "  - name: child_benefit_enhanced_rate\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    versions:\n"
            "      - effective_from: '2025-04-07'\n"
            "        formula: 26.05\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)
        pe_result = ValidationResult(
            "policyengine",
            passed=True,
            score=1.0,
            issues=[],
        )

        with (
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
                return_value=compile_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_ci",
                return_value=ci_result,
            ),
            patch(
                "axiom_encode.harness.validator_pipeline.ValidatorPipeline._run_policyengine",
                return_value=pe_result,
            ) as mock_policyengine,
        ):
            metrics = evaluate_artifact(
                rulespec_file=rules_file,
                policy_repo_root=tmp_path,
                axiom_rules_path=Path("/tmp/axiom-rules"),
                source_text="The enhanced rate is £26.05 from 2025-04-07.",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.policyengine_pass is True
        assert metrics.policyengine_score == 1.0
        assert metrics.policyengine_issues == []
        mock_policyengine.assert_called_once()


class TestEvalPrompt:
    def test_build_eval_prompt_includes_rulespec_schema_guardrails(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Grant standard is 165 for one child.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1.yaml",
            include_tests=True,
        )

        assert (
            "Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`."
            in prompt
        )
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "dtype:" in prompt
        assert "RuleSpec requirements:" in prompt
        assert "The programme file must begin with `format: rulespec/v1`" in prompt
        assert (
            "Use chained `if condition: value else: other_value` expressions" in prompt
        )
        assert "Do not use bare year periods like `2024`" in prompt

    def test_build_eval_prompt_includes_supported_schema_enums(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 25.60 ... 16.95",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not invent new entities, periods, or dtypes." in prompt
        assert (
            "Allowed `entity:` values are `Payment`, `Person`, `TaxUnit`, `Household`, "
            "`Family`, `TanfUnit`, `SnapUnit`, `SPMUnit`, `Corporation`, `Business`, "
            "`Asset`."
        ) in prompt
        assert "Allowed `period:` values are `Year`, `Month`, `Week`, `Day`." in prompt
        assert (
            "Allowed `dtype:` values are `Money`, `Rate`, `Boolean`, `Integer`, "
            "`Count`, `String`, `Decimal`, `Float`, or `Enum[Name]`."
        ) in prompt

    def test_build_eval_prompt_includes_unsupported_entity_fallback(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="ukpga/2010/1/section/1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) cease to be in force",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "ukpga/2010/1/section/1",
            "cold",
            workspace,
            [],
            target_file_name="ukpga-2010-1-section-1.yaml",
            include_tests=True,
        )

        assert (
            "If the source cannot be represented faithfully with the supported schema"
            in prompt
        )
        assert "`module.status: entity_not_supported`" in prompt
        assert "`module.status: deferred`" in prompt
        assert "leave the companion `.test.yaml` empty" in prompt
        assert "assertions against deferred symbols" in prompt

    def test_build_eval_prompt_for_editorially_omitted_slice_allows_deferred_docstring(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/10/c",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "(c)\n\n"
                ". . . . . . . . . . . . . . . . . . . . . . . ."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/10/c",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-10-c.yaml",
            include_tests=True,
        )

        assert (
            "editorially omitted or repealed text shown by ellipses or dotted placeholders"
            in prompt
        )
        assert "leave `.test.yaml` empty" in prompt

    def test_build_eval_prompt_forbids_python_inline_ternaries(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "Do not use Python inline ternaries" in prompt
        assert "`x if cond else y`" in prompt

    def test_build_eval_prompt_requires_rulespec_conditional_expression_syntax(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "`if condition: value else: other_value`" in prompt
        assert "do not use YAML-style `if:` / `then:` / `else:` blocks" in prompt
        assert (
            "Do not append a multiline conditional directly onto another expression"
            in prompt
        )

    def test_build_eval_prompt_requires_decimal_ratios_for_rate_dtype(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The percentage prescribed is 60 per cent.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-7.yaml",
            include_tests=True,
        )

        assert (
            "For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals."
            in prompt
        )
        assert (
            "Do not respond with summaries, markdown prose, or file-write confirmations"
            in prompt
        )
        assert (
            "do not use inline assignment syntax like `:=` inside formula blocks"
            in prompt
        )

    def test_build_eval_prompt_for_uk_leaf_prefers_person_over_family_constant(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "The weekly rate of child benefit payable in respect of a child "
                "or qualifying young person shall be 26.05."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert (
            'Prefer `Person` when the source states an amount or condition "in respect of"'
            in prompt
        )
        assert (
            "do not collapse it into an unconditional family-level constant" in prompt
        )

    def test_build_eval_prompt_for_uk_pence_threshold_requires_gbp_decimal_and_weekly_cadence(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "Where the amount of state pension credit payable is less than 10 pence per week, "
                "the credit shall not be payable unless the claimant is in receipt of another "
                "benefit payable with the credit."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "include `unit: GBP`" in prompt
        assert "`10 pence` should become `0.10`, not `10`" in prompt
        assert "do not disguise it as arithmetic like `1 / 10`" in prompt
        assert "prefer a money variable with matching `period:` cadence" in prompt

    def test_build_eval_prompt_for_positive_conditional_uk_leaf_requires_zero_or_false_else_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "£20 is disregarded if the claimant or, if he has a partner, his partner "
                "is in receipt of Scottish adult disability living allowance."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-schedule-vi-paragraph-4-1-a-iva.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "positive conditional leaves" in prompt
        assert (
            "the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`"
            in prompt
        )
        assert "do not use an unconditional amount or `else: true`" in prompt
        assert (
            "fixed supplement, allowance, or addition is payable only while an eligibility condition holds"
            in prompt
        )
        assert "do not leave that money output unconditional" in prompt
        assert (
            "do not collapse source-stated component facts into an opaque local input like `*_eligible_for_*`"
            in prompt
        )
        assert "prefer direct facts like `client_is_pregnant_parent`" in prompt

    def test_build_eval_prompt_for_determination_limb_discourages_invented_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the weekly amount of that claimant's income shall be determined—\n\n"
                "(i)\n\n"
                "if there is a recognised cycle of work, by reference to his average "
                "weekly income over the period of the complete cycle; or"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not invent sibling outcomes for non-applicable cases with `else: 0`"
            in prompt
        )
        assert "leave other cases to sibling limbs" in prompt
        assert "keep a branch-specific money or rate output for that basis" in prompt
        assert (
            "do not invent sibling outcomes for inapplicable cases with `else: 0`"
            in prompt
        )
        assert (
            "pair the branch-specific money or rate output with a separate applicability boolean"
            in prompt
        )
        assert (
            "omit assertions about the branch-specific money or rate output" in prompt
        )
        assert (
            "qualifies its averaging basis with operative parenthetical text" in prompt
        )
        assert (
            "includes periods in which the claimant does no work but disregards other absences"
            in prompt
        )
        assert "generic `average_weekly_income_*` input" in prompt
        assert (
            "`such other payments as may ... enable the claimant's average weekly income to be determined more accurately`"
            in prompt
        )
        assert (
            "do not leave the branch money output unconditionally equal to the input average"
            in prompt
        )
        assert (
            "do not reuse the parent provision's generic final-amount phrase" in prompt
        )
        assert (
            "name the principal money or rate output after this limb's own basis or method"
            in prompt
        )

    def test_build_eval_prompt_for_purpose_limited_deeming_discourages_unsupported_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17/9A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (9)(b), and for that purpose only, "
                "the amounts specified in paragraph (5) shall be treated as though "
                "they were earnings."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/9A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-9A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "purpose-limited deeming clauses" in prompt
        assert "do not use `status: entity_not_supported`" in prompt
        assert (
            "paragraph-(5) amounts treated as earnings for paragraph-(9)(b) only"
            in prompt
        )

    def test_build_eval_prompt_for_uk_residual_determination_limb_requires_other_case_condition(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "the amount to be included in the claimant's weekly income shall be determined—\n\n"
                "(iv)\n\n"
                "in any other case, by multiplying the amount of the payment by 7 and dividing "
                "the product by the number of days in the period in respect of which it is made."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For residual sibling limbs phrased like `in any other case`" in prompt
        assert "Do not treat the shared parent triggers alone as sufficient" in prompt
        assert "model a local residual-case fact or applicability helper" in prompt
        assert "no more specific sibling case applies" in prompt
        assert (
            "include a case where the parent conditions hold but the residual `other case` condition is false"
            in prompt
        )

    def test_build_eval_prompt_for_shall_be_treated_discourages_fact_input_and_vacuous_true(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "If a claimant is entitled to receive a payment to which paragraph (5) "
                "applies, the amount of that payment shall be treated as if made in "
                "respect of a period of a year."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not introduce a `*_fact` input" in prompt
        assert "do not use vacuous `else: true`" in prompt
        assert (
            "do not replace the amount-level legal effect with a `Person`/`Day` boolean stand-in"
            in prompt
        )
        assert (
            "prefer `status: entity_not_supported` over a pseudo-boolean approximation"
            in prompt
        )
        assert (
            "If the current ontology cannot faithfully tie the deeming effect to the same payment amount"
            in prompt
        )

    def test_build_eval_prompt_for_claimant_incurred_expenses_preserves_claimant_predicate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/f/i",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "travelling expenses incurred by the claimant between his home and place "
                "of employment;"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/f/i",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-f-i.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "If an expenses limb says the expenses are `incurred by the claimant`"
            in prompt
        )
        assert "preserve that claimant-incurred predicate explicitly" in prompt
        assert "Do not collapse it into only an employer-made-payment fact" in prompt

    def test_build_eval_prompt_for_claim_date_reference_day_uses_single_operative_date(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "For the purposes of paragraph (2)(b) the last payments are the last payments "
                "before the date the claim was made or treated as made or, if there is a "
                "subsequent supersession under section 10 of the Social Security Act 1998, "
                "the last payments before the date of the supersession."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve a single legally operative reference day" in prompt
        assert "model one canonical operative claim-date fact" in prompt
        assert (
            "do not encode separate `day_is_date_claim_was_made` and `day_is_date_claim_was_treated_as_made` facts and then combine them with `or`"
            in prompt
        )
        assert (
            "include one no-supersession case for the operative claim date and one supersession case for the supersession date"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_override_discourages_permission_gate(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to regulation 17B(6), in the case of any income taken into "
                "account for the purpose of calculating a person's income, there shall "
                "be disregarded any amount payable by way of tax."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "treat the cited provision as a possible override or displacement" in prompt
        )
        assert "model a local override/displacement boolean" in prompt
        assert (
            "Do not encode those `Subject to ...` qualifiers as helper names like `*_permits_*`"
            in prompt
        )

    def test_build_eval_prompt_for_subject_to_unavailable_imports_allows_paragraph_specific_inputs(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A/2/e",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to paragraphs (3), (4) and (4A), “earnings” in the case of "
                "employment as an employed earner, means any remuneration or profit "
                "derived from that employment and includes any payment by way of a retainer."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/e",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-e.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "When canonical imports for those cited paragraphs are available in the workspace, import them."
            in prompt
        )
        assert (
            "paragraph-specific local inputs are acceptable for an isolated slice artifact"
            in prompt
        )
        assert (
            "preserve the cited paragraph numbers and the branch-specific legal effect"
            in prompt
        )

    def test_build_eval_prompt_for_pure_cross_reference_computation_preserves_distinct_cited_alternatives(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "In the case of the earnings of self-employed earners, the amounts "
                "specified in paragraph (10) shall be taken into account in accordance "
                "with paragraph (4) or, as the case may be, paragraph (10) of regulation "
                "13 of the Computation of Earnings Regulations, as having effect in the "
                "case of state pension credit."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "do not replace the cited computation with local boolean `*_route_is_satisfied` or `*_fact` placeholders"
            in prompt
        )
        assert "do not emit a top-level `status: deferred` stub" in prompt
        assert (
            "do not collapse those cited alternatives into one generic treatment gate"
            in prompt
        )
        assert (
            "preserve the distinct cited alternatives with paragraph-specific imports or local facts/amounts"
            in prompt
        )
        assert "do not invent an extra `no treatment applies` branch" in prompt
        assert (
            "do not make the cited route-selection flags part of whether the paragraph itself applies"
            in prompt
        )
        assert (
            "do not encode the consequence as an unqualified `if paragraph_4_route: paragraph_4_amount else: paragraph_10_amount`"
            in prompt
        )
        assert (
            "Paragraph (10) must be selected by a paragraph-(10) route fact/import or by a derived paragraph-(10) route helper"
            in prompt
        )
        assert "prefer a single mutually exclusive route selector" in prompt
        assert (
            "Do not expose two independent route booleans that allow both routes or neither route to be selected"
            in prompt
        )
        assert "a safe local-placeholder shape is" in prompt
        assert (
            "paragraph-(10) route is derived as the applicable paragraph with not paragraph-(4) route"
            in prompt
        )
        assert "Do not create an invalid-route output branch that returns `0`" in prompt
        assert "self-employed earnings trigger the paragraph" in prompt
        assert (
            "regulation 13 paragraph (4) or paragraph (10) chooses the accounting route"
            in prompt
        )
        assert (
            "avoid a false case that makes a self-employed-earner branch fail merely because neither local route flag was selected"
            in prompt
        )
        assert "include separate cases for the distinct cited alternatives" in prompt

    def test_build_eval_prompt_requires_calendar_date_test_periods(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/13A/3/b",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "The amount of the guarantee credit payable in respect of the part-week "
                "shall be determined by multiplying the resulting figure by the number "
                "of days in the part-week."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13A/3/b",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13A-3-b.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use concrete ISO calendar dates like `2025-03-21`" in prompt
        assert "do not use ISO week strings like `2025-W13`" in prompt

    def test_build_eval_prompt_for_uk_leaf_forbids_speculative_future_period_tests(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
        )

        assert "The test file must contain YAML only" in prompt
        assert "must be a YAML list beginning with `- name:` entries" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert (
            "Use factual predicates or quantities in `input:`, not the output variable being asserted"
            in prompt
        )
        assert "Use concrete scalar values, not formula strings" in prompt
        assert "Use `period`, `input`, and `output` keys" in prompt

    def test_build_eval_prompt_for_uk_branch_leaves_requires_branch_specific_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) 332.95 per week in the case of a claimant who has a partner.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "encode the branch identity in the output variable name" in prompt
        assert "principal output variable must encode that deepest token" in prompt
        assert "`standard_minimum_guarantee`" in prompt
        assert "`child_benefit_weekly_rate`" in prompt

    def test_build_eval_prompt_for_where_must_clauses_requires_inapplicable_case(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/4A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where the young person is aged 19, he or she must have started the education "
                "or training before reaching that age."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/4A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-4A.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Where X, Y must ..." in prompt
        assert "Include a `.test.yaml` case where `X` is false" in prompt

    def test_build_eval_prompt_for_uk_leaf_discourages_opaque_condition_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) ... only person or elder or eldest person ... £26.05.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "avoid opaque placeholders like `*_condition`" in prompt
        assert "`child_benefit_is_only_person`" in prompt
        assert "`claimant_has_partner`" in prompt
        assert "`is_joint_claimant`" in prompt

    def test_build_eval_prompt_for_single_row_fixed_amount_discourages_placeholder_names_and_applies_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "Structured table:\n"
                "Element | Amount for each assessment period\n"
                "single claimant aged under 25 | £316.98"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2013-376-regulation-36.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use a descriptive legal variable name" in prompt
        assert "not a path- or source-id-derived placeholder" in prompt
        assert "do not invent a fresh `*_applies` helper" in prompt
        assert "do not invent alternate zero-amount tests" in prompt
        assert "Do not emit `otherwise:`" in prompt
        assert "Do not emit `before YYYY-MM-DD: 0`" in prompt
        assert "Do not emit malformed date blocks like `from 0:`" in prompt
        assert "use boolean or fact-shaped helper inputs" in prompt
        assert "Do not invent sample ages like `2`, `3`, `24`, or `25`" in prompt
        assert "keep `.test.yaml` outputs scalar" in prompt
        assert "keep the row-defining conditions satisfied" in prompt
        assert "principal amount rule should usually be a grounded constant" in prompt
        assert "Do not include `alternate_branch_*` tests" in prompt
        assert "write `2500`, not `2,500`" in prompt

    def test_build_eval_prompt_requires_named_scalars_for_repeated_and_threshold_numbers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/schedule/VI/2A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Where a person is engaged in employments specified in paragraph 2 but "
                "his earnings are less than £20 and he is also engaged in other employment, "
                "so much of his other earnings as would not exceed £20. "
                "A non-dependant aged 18 or over is treated differently. "
                "See section 3(4)."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/2A",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec"
            in prompt
        )
        assert (
            "If the same numeric value appears twice in materially different legal roles"
            in prompt
        )
        assert (
            "reuse that named scalar everywhere the rule compares against or computes with that number"
            in prompt
        )
        assert (
            'If `./source.txt` says someone is "aged 18 or over", "under 25"' in prompt
        )
        assert "Do not create scalar variables for citation numbers" in prompt
        assert (
            "Do not invent `dtype: String` variables just to restate the effective date"
            in prompt
        )
        assert (
            "Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables"
            in prompt
        )
        assert "`==` for equality" in prompt

    def test_prepare_eval_workspace_injects_resolved_defined_term_stub(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        definition_files = [
            item for item in workspace.context_files if item.kind == "definition_stub"
        ]
        assert len(definition_files) == 1
        assert (
            definition_files[0].workspace_path
            == "context/legislation/ukpga/2002/16/section/3ZA/3.yaml"
        )
        assert (
            definition_files[0].import_path == "legislation/ukpga/2002/16/section/3ZA/3"
        )
        stub_path = workspace.root / definition_files[0].workspace_path
        assert stub_path.exists()
        assert "is_member_of_mixed_age_couple" in stub_path.read_text()

    def test_prepare_eval_workspace_copies_resolved_canonical_concept_file(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rules-us-co"
        concept_file = policy_repo_root / "statute" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        concept_files = [
            item for item in workspace.context_files if item.kind == "canonical_concept"
        ]
        assert len(concept_files) == 1
        assert concept_files[0].workspace_path == "context/statute/crs/26-2-703/12.yaml"
        assert concept_files[0].import_path == "statute/crs/26-2-703/12"
        copied_path = workspace.root / concept_files[0].workspace_path
        assert copied_path.exists()
        assert "is_individual_responsibility_contract" in copied_path.read_text()

    def test_hydrate_eval_root_places_resolved_definition_stub_under_runner_root(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        runner_root = tmp_path / "case" / "openai-gpt-5.4"
        source_dir = runner_root / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "example.yaml").write_text("format: rulespec/v1\nrules: []\n")

        _hydrate_eval_root(runner_root, workspace)

        hydrated = (
            runner_root
            / "legislation"
            / "ukpga"
            / "2002"
            / "16"
            / "section"
            / "3ZA"
            / "3.yaml"
        )
        assert hydrated.exists()
        assert "is_member_of_mixed_age_couple" in hydrated.read_text()

    def test_build_eval_prompt_includes_resolved_defined_term_guidance(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Resolved definition files are available below." in prompt
        assert "mixed-age couple" in prompt
        assert (
            "legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple"
            in prompt
        )
        assert (
            "import that canonical definition instead of inventing a leaf-local helper"
            in prompt
        )
        assert "Do not replace that import with a local deferred stub" in prompt
        assert (
            "Do not encode such local factual predicates as placeholder constants like `true` or `false`."
            in prompt
        )
        assert (
            "Do not encode such local factual predicates as `status: deferred`"
            in prompt
        )

    def test_build_eval_prompt_includes_resolved_canonical_concept_guidance(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "rules-us-co"
        concept_file = policy_repo_root / "statute" / "crs" / "26-2-703" / "12.yaml"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    C.R.S. § 26-2-703(12)
    Definitions

    "Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
rules:
  - name: is_individual_responsibility_contract
    kind: input
    entity: Person
    dtype: Boolean
    period: Month
"""
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            axiom_rules_path=policy_repo_root,
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "co/regulation/3.609.1/A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "Resolved canonical concept files from this corpus are available below."
            in prompt
        )
        assert "individual responsibility contract" in prompt
        assert "statute/crs/26-2-703/12#is_individual_responsibility_contract" in prompt
        assert (
            "import or re-export that exact canonical concept instead of duplicating it locally"
            in prompt
        )

    def test_build_eval_prompt_includes_import_vs_local_helper_protocol(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="26 USC 24(c)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text='The term "qualifying child" means a qualifying child as defined in section 152(c).',
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 24(c)",
            "cold",
            workspace,
            [],
            target_file_name="24-c.yaml",
            include_tests=True,
        )

        assert (
            "emit the upstream import instead of restating the concept locally"
            in prompt
        )
        assert "says a value is determined `in accordance with section X`" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt

    def test_build_eval_prompt_discourages_fabricated_same_instrument_imports(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) except where paragraph (b) applies, £81.50 per week if paragraph 1(1)(a), (b) or (c) of Part I of Schedule I is satisfied.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6-5-a.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Do not fabricate sibling-file imports" in prompt
        assert "do not guess" in prompt
        assert "do not invent `import` statements or `imports:` blocks" in prompt

    def test_build_eval_prompt_for_openai_inlines_source_text(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert "You do not have filesystem tool access in this eval" in prompt
        assert "=== BEGIN SOURCE.TXT ===" in prompt
        assert "Editorial note: current text valid from 2025-04-07." in prompt
        assert "26.05" in prompt

    def test_build_eval_prompt_for_date_silent_source_includes_neutral_scaffold_fallback(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(E)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Applications received will be certified for six (6) consecutive months "
                "beginning the first month the assistance unit is found eligible for basic cash assistance."
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(E)",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1-E.yaml",
            include_tests=True,
            runner_backend="codex",
        )

        assert "effective_from: '0001-01-01'" in prompt
        assert "harness-only fallback" in prompt


class TestOpenAIEvalRequest:
    def test_post_openai_eval_request_retries_transient_status(self):
        error_response = Mock()
        error_response.status_code = 502
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [error_response, ok_response]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2

    def test_post_openai_eval_request_retries_request_exception(self):
        ok_response = Mock()
        ok_response.status_code = 200

        with (
            patch("axiom_encode.harness.evals.requests.post") as mock_post,
            patch("axiom_encode.harness.evals.time.sleep"),
        ):
            mock_post.side_effect = [
                requests.exceptions.ReadTimeout("timed out"),
                ok_response,
            ]

            response = _post_openai_eval_request(
                headers={"Authorization": "Bearer test"},
                body={"model": "gpt-5.4", "input": "hi"},
            )

        assert response is ok_response
        assert mock_post.call_count == 2


class TestEvalSuiteManifest:
    def test_load_eval_suite_manifest_supports_policyengine_rule_hint(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    source_id: uc-std-allowance-single
    source_file: ./source.txt
    oracle: policyengine
    policyengine_country: uk
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert (
            manifest.cases[0].policyengine_rule_hint
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_load_eval_suite_manifest_supports_generalist_review_gate(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
gates:
  min_generalist_review_pass_rate: 0.95
cases:
  - kind: source
    name: sample
    source_id: sample-source
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.gates.min_generalist_review_pass_rate == 0.95

    def test_run_eval_suite_passes_policyengine_rule_hint_to_source_runner(
        self, tmp_path
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK source suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: uc-standard-allowance-single-young
    source_id: uc-std-allowance-single
    source_file: ./source.txt
    oracle: policyengine
    policyengine_country: uk
    policyengine_rule_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "uc-std-allowance-single")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            return_value=[source_result],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [source_result]
        assert (
            mock_source.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_run_eval_suite_records_active_case_before_dispatch(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Active case suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        snapshots: list[dict] = []

        def fake_run_source_eval(**_kwargs):
            snapshots.append(json.loads((output_root / "suite-run.json").read_text()))
            return [source_result]

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=fake_run_source_eval,
            ),
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert len(snapshots) == 1
        active_state = snapshots[0]
        assert active_state["status"] == "running"
        assert active_state["completed_cases"] == 0
        assert active_state["result_count"] == 0
        assert active_state["active_case"]["index"] == 1
        assert active_state["active_case"]["name"] == "tanf-slice"
        assert active_state["active_case"]["output_root"] == str(
            output_root / "01-tanf-slice"
        )
        final_state = json.loads((output_root / "suite-run.json").read_text())
        assert final_state["status"] == "completed"
        assert "active_case" not in final_state

    def test_run_eval_suite_routes_source_case_to_enclosing_policy_repo(self, tmp_path):
        policy_repo = tmp_path / "rules-us-tn"
        source_file = (
            policy_repo
            / "sources"
            / "slices"
            / "tenncare"
            / "post-eligibility"
            / "current-effective"
            / "snap_standard_utility_allowance_tn.txt"
        )
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("Tennessee source text")
        runtime_axiom_rules = tmp_path / "axiom-rules"
        runtime_axiom_rules.mkdir()
        output_root = tmp_path / "out"

        manifest = EvalSuiteManifest(
            name="TN suite",
            path=tmp_path / "suite.yaml",
            runners=["openai:gpt-5.4"],
            mode="repo-augmented",
            allow_context=[],
            gates=EvalReadinessGates(),
            cases=[
                EvalSuiteCase(
                    kind="source",
                    name="snap-tn-sua",
                    source_id="snap_standard_utility_allowance_tn",
                    source_file=source_file,
                    mode="repo-augmented",
                )
            ],
        )
        source_result = _fake_eval_result("openai-gpt-5.4", "snap-tn-sua")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                return_value=[source_result],
            ) as mock_run_source_eval,
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=runtime_axiom_rules,
                axiom_path=None,
            )

        assert mock_run_source_eval.call_args.kwargs["policy_path"] == policy_repo
        assert mock_run_source_eval.call_args.kwargs["source_path"] == source_file
        assert (
            mock_run_source_eval.call_args.kwargs["runtime_axiom_rules_path"]
            == runtime_axiom_rules
        )

    def test_run_eval_suite_retries_transient_exception(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=[RuntimeError("stream disconnected"), [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_run_eval_suite_retries_error_results(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")
        failed.success = False
        failed.error = "Reconnecting..."
        source_result = _fake_eval_result("openai-gpt-5.4", "co-tanf-f")

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            side_effect=[[failed], [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [source_result]
        assert mock_source.call_count == 2

    def test_run_eval_suite_does_not_retry_compile_failures(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        failed = _fake_eval_result(
            "openai-gpt-5.4",
            "co-tanf-f",
            compile_pass=False,
        )

        with patch(
            "axiom_encode.harness.evals.run_source_eval",
            return_value=[failed],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [failed]
        assert mock_source.call_count == 1

    def test_run_eval_suite_stops_after_usage_limit_error(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Usage limit suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    source_file: ./source.txt
  - kind: source
    name: case-two
    source_id: case-two
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"

        usage_limited = _fake_eval_result("openai-gpt-5.4", "case-one")
        usage_limited.metrics.generalist_review_pass = False
        usage_limited.metrics.generalist_review_score = None
        usage_limited.metrics.generalist_review_issues = [
            "Reviewer CLI exited 1: You've hit your usage limit."
        ]
        second = _fake_eval_result("openai-gpt-5.4", "case-two")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=[[usage_limited], [second]],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [usage_limited]
        assert mock_source.call_count == 1
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "failed"
        assert "usage limit" in state["error"].lower()
        assert state["completed_cases"] == 1
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_run_eval_suite_retries_reviewer_timeout(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Timeout retry suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)

        timed_out = _fake_eval_result("openai-gpt-5.4", "case-one")
        timed_out.metrics.generalist_review_pass = False
        timed_out.metrics.generalist_review_score = None
        timed_out.metrics.generalist_review_issues = [
            "Reviewer error: Reviewer CLI exited 1: Timeout after 300s"
        ]
        recovered = _fake_eval_result("openai-gpt-5.4", "case-one")

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                side_effect=[[timed_out], [recovered]],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
            )

        assert results == [recovered]
        assert mock_source.call_count == 2

    def test_run_eval_suite_resume_skips_completed_cases(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Resume suite
runners:
  - openai:gpt-5.4
cases:
  - kind: source
    name: case-one
    source_id: case-one
    source_file: ./source.txt
  - kind: source
    name: case-two
    source_id: case-two
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)
        output_root = tmp_path / "out"
        output_root.mkdir()

        first = _fake_eval_result("openai-gpt-5.4", "case-one")
        second = _fake_eval_result("openai-gpt-5.4", "case-two")
        (output_root / "suite-run.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": manifest.name,
                        "path": str(manifest.path),
                        "runners": manifest.runners,
                        "effective_runners": manifest.runners,
                    },
                    "status": "running",
                    "started_at": "2026-04-10T16:17:28+00:00",
                    "updated_at": "2026-04-10T16:30:00+00:00",
                    "total_cases": 2,
                    "completed_cases": 1,
                    "result_count": 1,
                    "last_case_name": "case-one",
                }
            )
            + "\n"
        )
        (output_root / "suite-results.jsonl").write_text(
            json.dumps(
                {
                    "case_index": 1,
                    "case_name": "case-one",
                    "case_kind": "source",
                    "result": first.to_dict(),
                },
                sort_keys=True,
            )
            + "\n"
        )

        with (
            patch(
                "axiom_encode.harness.evals.run_source_eval",
                return_value=[second],
            ) as mock_source,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                axiom_rules_path=tmp_path / "axiom-rules",
                axiom_path=None,
                resume_existing=True,
            )

        assert [result.citation for result in results] == ["case-one", "case-two"]
        mock_source.assert_called_once()
        assert mock_source.call_args.kwargs["source_id"] == "case-two"
        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "completed"
        assert state["started_at"] == "2026-04-10T16:17:28+00:00"
        assert state["completed_cases"] == 2
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_repo_us_co_colorado_works_seed_manifest_loads_expected_cases(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_seed.yaml"
        )

        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 5
        assert manifest.gates.min_cases == 5
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert all(case.kind == "source" for case in manifest.cases)
        assert [case.name for case in manifest.cases] == [
            "co-3-606-1-f",
            "co-3-606-1-g",
            "co-3-606-1-h",
            "co-3-606-1-i",
            "co-3-606-1-k",
        ]
        assert manifest.cases[0].allow_context == []
        assert manifest.cases[2].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve()
        ]
        assert manifest.cases[4].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.yaml"
            ).resolve(),
        ]

    def test_repo_us_co_colorado_works_leaf_seed_manifest_loads_expected_cases(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_leaf_seed.yaml"
        )

        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 6
        assert manifest.gates.min_cases == 6
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert all(case.kind == "source" for case in manifest.cases)
        assert [case.name for case in manifest.cases] == [
            "co-3-606-1-e",
            "co-3-606-1-g",
            "co-3-606-1-h",
            "co-3-606-1-i",
            "co-3-606-1-j",
            "co-3-606-1-k",
        ]
        assert manifest.cases[0].allow_context == []
        assert manifest.cases[2].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve()
        ]
        assert manifest.cases[5].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.yaml"
            ).resolve(),
        ]

    def test_repo_us_co_colorado_works_leaf_repair_manifest_loads_expected_cases(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_leaf_repair.yaml"
        )

        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 5
        assert manifest.gates.min_cases == 5
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert all(case.kind == "source" for case in manifest.cases)
        assert [case.name for case in manifest.cases] == [
            "co-3-606-1-g",
            "co-3-606-1-h",
            "co-3-606-1-i",
            "co-3-606-1-j",
            "co-3-606-1-k",
        ]
        assert manifest.cases[0].allow_context == []
        assert manifest.cases[1].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve()
        ]
        assert manifest.cases[4].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.yaml"
            ).resolve(),
        ]

    def test_repo_us_co_colorado_works_leaf_k_repair_manifest_loads_expected_case(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_leaf_k_repair.yaml"
        )

        assert manifest.name == "Colorado Works leaf K repair"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.5
        assert manifest.cases[0].name == "co-3-606-1-k"
        assert manifest.cases[0].source_id == "9 CCR 2503-6 3.606.1(K)"
        assert manifest.cases[0].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.yaml"
            ).resolve(),
        ]

    def test_repo_us_co_colorado_works_leaf_h_repair_manifest_loads_expected_case(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_leaf_h_repair.yaml"
        )

        assert manifest.name == "Colorado Works leaf H repair"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.5
        assert manifest.cases[0].name == "co-3-606-1-h"
        assert manifest.cases[0].source_id == "9 CCR 2503-6 3.606.1(H)"
        assert manifest.cases[0].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve()
        ]

    def test_repo_us_co_colorado_works_leaf_closeout_manifest_loads_expected_cases(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_co_colorado_works_leaf_closeout.yaml"
        )

        assert manifest.name == "Colorado Works leaf closeout"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 2
        assert manifest.gates.min_cases == 2
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.5
        assert [case.name for case in manifest.cases] == [
            "co-3-606-1-h",
            "co-3-606-1-k",
        ]
        assert manifest.cases[0].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve()
        ]
        assert manifest.cases[1].allow_context == [
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.yaml"
            ).resolve(),
        ]

    def test_repo_us_snap_federal_reconstruction_seed_manifest_loads_expected_cases(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_federal_reconstruction_seed.yaml"
        )

        assert manifest.name == "SNAP federal reconstruction seed"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 4
        assert manifest.gates.min_cases == 4
        assert manifest.gates.min_success_rate == 0.75
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.75
        assert all(case.kind == "source" for case in manifest.cases)
        assert [case.name for case in manifest.cases] == [
            "snap-2017-a",
            "snap-2017-c-1",
            "snap-2017-c-3",
            "snap-fy2026-cola-allotments",
        ]
        assert manifest.cases[0].allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2014" / "e.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.yaml"
            ).resolve(),
        ]
        assert manifest.cases[1].allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2017" / "a.yaml"
            ).resolve()
        ]
        assert manifest.cases[2].allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2017" / "a.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us"
                / "statute"
                / "7"
                / "2017"
                / "c"
                / "1.yaml"
            ).resolve(),
        ]
        assert manifest.cases[3].allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2017" / "a.yaml"
            ).resolve()
        ]

    def test_repo_us_snap_federal_c3_repair_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_federal_c3_repair.yaml"
        )

        assert manifest.name == "SNAP federal 2017(c)(3) repair"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.2
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap-2017-c-3"
        assert case.source_id == "7 USC 2017(c)(3)"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "7-USC"
                / "2017"
                / "c"
                / "3.txt"
            ).resolve()
        )
        assert case.allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2017" / "a.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us"
                / "statute"
                / "7"
                / "2017"
                / "c"
                / "1.yaml"
            ).resolve(),
        ]

    def test_repo_us_snap_fy2026_cola_table_repair_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_fy2026_cola_table_repair.yaml"
        )

        assert manifest.name == "SNAP FY2026 COLA table repair"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.max_mean_estimated_cost_usd == 0.3
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap-fy2026-cola-allotments"
        assert case.source_id == "USDA SNAP FY 2026 COLA allotment table"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "usda"
                / "snap"
                / "fy-2026-cola"
                / "allotment-table.txt"
            ).resolve()
        )
        assert case.allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2017" / "a.yaml"
            ).resolve()
        ]

    def test_repo_us_snap_asset_test_refresh_manifest_loads_expected_case(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_asset_test_refresh.yaml"
        )

        assert manifest.name == "SNAP asset test refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "meets_snap_asset_test"
        assert case.source_id == "7 USC 2014(g)(1)"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "7-USC"
                / "2014"
                / "g"
                / "1.txt"
            ).resolve()
        )
        assert case.allow_context == [
            (
                repo_root.parent
                / "rules-us"
                / "usda"
                / "snap"
                / "fy-2026-cola"
                / "2.yaml"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "meets_snap_asset_test"

    def test_repo_us_snap_asset_test_current_effective_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_asset_test_current_effective_refresh.yaml"
        )

        assert manifest.name == "SNAP asset test current-effective refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "meets_snap_asset_test_current_effective"
        assert case.source_id == "USDA SNAP FY2026 maximum asset limits"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "usda"
                / "snap"
                / "fy-2026-cola"
                / "asset-limits-current-effective.txt"
            ).resolve()
        )
        assert case.allow_context == [
            (
                repo_root.parent
                / "rules-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.yaml"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "meets_snap_asset_test"

    def test_repo_us_snap_eligibility_refresh_manifest_loads_expected_case(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_eligibility_refresh.yaml"
        )

        assert manifest.name == "SNAP eligibility refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_cases == 1
        assert manifest.gates.min_success_rate == 1.0
        assert manifest.gates.min_compile_pass_rate == 1.0
        assert manifest.gates.min_ci_pass_rate == 1.0
        assert manifest.gates.min_zero_ungrounded_rate == 1.0
        assert manifest.gates.min_generalist_review_pass_rate == 1.0
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "is_snap_eligible"
        assert case.source_id == "Federal SNAP current-effective household eligibility"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "7-USC"
                / "snap"
                / "current-effective"
                / "is_snap_eligible.txt"
            ).resolve()
        )
        assert case.allow_context == [
            (
                repo_root.parent / "rules-us" / "statute" / "7" / "2014" / "c.yaml"
            ).resolve(),
            (
                repo_root.parent
                / "rules-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.yaml"
            ).resolve(),
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "is_snap_eligible"

    def test_repo_us_snap_earned_income_deduction_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_earned_income_deduction_refresh.yaml"
        )

        assert manifest.name == "SNAP earned income deduction refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_earned_income_deduction"
        assert (
            case.source_id == "SNAP earned income deduction under 7 USC 2014(e)(2)(B)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "7-USC"
                / "snap"
                / "current-effective"
                / "snap_earned_income_deduction.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_earned_income_deduction"

    def test_repo_us_snap_net_income_pre_shelter_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_net_income_pre_shelter_refresh.yaml"
        )

        assert manifest.name == "SNAP pre-shelter net income refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_net_income_pre_shelter"
        assert case.source_id == "SNAP pre-shelter net income under 7 USC 2014(e)(6)(A)"
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us"
                / "sources"
                / "slices"
                / "7-USC"
                / "snap"
                / "current-effective"
                / "snap_net_income_pre_shelter.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_net_income_pre_shelter"

    def test_repo_us_snap_nc_standard_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_nc_standard_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "North Carolina SNAP standard utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_standard_utility_allowance_nc"
        assert (
            case.source_id
            == "North Carolina SNAP standard utility allowance under FNS 360.01(A)(1)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-nc"
                / "sources"
                / "slices"
                / "ncdhhs"
                / "fns"
                / "360"
                / "current-effective"
                / "snap_standard_utility_allowance_nc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_standard_utility_allowance"

    def test_repo_us_snap_nc_limited_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_nc_limited_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "North Carolina SNAP limited utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_limited_utility_allowance_nc"
        assert (
            case.source_id
            == "North Carolina SNAP basic utility allowance under FNS 360.01(A)(2)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-nc"
                / "sources"
                / "slices"
                / "ncdhhs"
                / "fns"
                / "360"
                / "current-effective"
                / "snap_limited_utility_allowance_nc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_limited_utility_allowance"

    def test_repo_us_snap_nc_individual_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_nc_individual_utility_allowance_refresh.yaml"
        )

        assert (
            manifest.name == "North Carolina SNAP individual utility allowance refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_individual_utility_allowance_nc"
        assert (
            case.source_id
            == "North Carolina SNAP telephone utility allowance under FNS 360.01(A)(3)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-nc"
                / "sources"
                / "slices"
                / "ncdhhs"
                / "fns"
                / "360"
                / "current-effective"
                / "snap_individual_utility_allowance_nc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_individual_utility_allowance"

    def test_repo_us_snap_tn_standard_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_tn_standard_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "Tennessee SNAP standard utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_standard_utility_allowance_tn"
        assert (
            case.source_id
            == "Tennessee SNAP standard utility allowance under TennCare ABD Manual 125.020 section 3.d.ii.1.c.i"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-tn"
                / "sources"
                / "slices"
                / "tenncare"
                / "post-eligibility"
                / "current-effective"
                / "snap_standard_utility_allowance_tn.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_standard_utility_allowance"

    def test_repo_us_snap_tn_limited_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_tn_limited_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "Tennessee SNAP limited utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_limited_utility_allowance_tn"
        assert (
            case.source_id
            == "Tennessee SNAP limited utility allowance under TennCare ABD Manual 125.020 section 3.d.ii.1.c.ii"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-tn"
                / "sources"
                / "slices"
                / "tenncare"
                / "post-eligibility"
                / "current-effective"
                / "snap_limited_utility_allowance_tn.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_limited_utility_allowance"

    def test_repo_us_snap_tn_individual_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_tn_individual_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "Tennessee SNAP individual utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_individual_utility_allowance_tn"
        assert (
            case.source_id
            == "Tennessee SNAP telephone utility allowance under TennCare ABD Manual 125.020 section 3.d.ii.1.c.iii"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-tn"
                / "sources"
                / "slices"
                / "tenncare"
                / "post-eligibility"
                / "current-effective"
                / "snap_individual_utility_allowance_tn.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_individual_utility_allowance"

    def test_repo_us_snap_tn_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_tn_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Tennessee SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_tn"
        assert (
            case.source_id
            == "Tennessee SNAP child support deduction election under SNAP Policy Manual Chapter 23.L"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-tn"
                / "sources"
                / "slices"
                / "tdhs"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_tn.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_tn_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_tn_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Tennessee SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_tn"
        assert (
            case.source_id
            == "Tennessee SNAP self-employment expense option under TDHS SNAP Income Policy 24.14"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-tn"
                / "sources"
                / "slices"
                / "tdhs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_tn.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_ca_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ca_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "California SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_ca"
        assert (
            case.source_id
            == "California CalFresh self-employment expense choice under MPP 63-503.413"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ca"
                / "sources"
                / "slices"
                / "cdss"
                / "calfresh"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_ca.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_ca_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ca_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "California SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_ca"
        assert (
            case.source_id
            == "California CalFresh child support exclusion under MPP 63-502.2(p)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ca"
                / "sources"
                / "slices"
                / "cdss"
                / "calfresh"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_ca.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_co_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_co_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Colorado SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_co"
        assert (
            case.source_id
            == "Colorado SNAP self-employment expense deduction rule under 10 CCR 2506-1 section 4.403.11(B)-(C)(3)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-co"
                / "sources"
                / "slices"
                / "cdhs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_co.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_co_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_co_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Colorado SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_co"
        assert (
            case.source_id
            == "Colorado SNAP child support exclusion under 10 CCR 2506-1 section 4.407.5"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-co"
                / "sources"
                / "slices"
                / "cdhs"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_co.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_ny_individual_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ny_individual_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "New York SNAP individual utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_individual_utility_allowance_ny"
        assert (
            case.source_id
            == "New York SNAP telephone utility allowance under OTDA LDSS-5006 effective October 1, 2025"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ny"
                / "sources"
                / "slices"
                / "otda"
                / "snap"
                / "current-effective"
                / "snap_individual_utility_allowance_ny.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_individual_utility_allowance"

    def test_repo_us_snap_ny_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ny_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "New York SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_ny"
        assert (
            case.source_id
            == "New York SNAP child support deduction election under OTDA SNAP Source Book section 13.G.2"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ny"
                / "sources"
                / "slices"
                / "otda"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_ny.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_nc_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_nc_child_support_deduction_option_refresh.yaml"
        )

        assert (
            manifest.name
            == "North Carolina SNAP child support deduction option refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_nc"
        assert (
            case.source_id
            == "North Carolina SNAP child support deduction election under FNS 340 section 340.19"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-nc"
                / "sources"
                / "slices"
                / "ncdhhs"
                / "fns"
                / "340"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_nc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_nc_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_nc_self_employment_expense_option_refresh.yaml"
        )

        assert (
            manifest.name
            == "North Carolina SNAP self-employment expense option refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_nc"
        assert (
            case.source_id
            == "North Carolina SNAP self-employment expense option under FNS 315 section 315.28"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-nc"
                / "sources"
                / "slices"
                / "ncdhhs"
                / "fns"
                / "315"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_nc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_ny_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ny_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "New York SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_ny"
        assert (
            case.source_id
            == "New York SNAP self-employment expense treatment under OTDA SNAP Source Book section 13.H"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ny"
                / "sources"
                / "slices"
                / "otda"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_ny.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_ny_standard_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ny_standard_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "New York SNAP standard utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_standard_utility_allowance_ny"
        assert (
            case.source_id
            == "New York SNAP standard utility allowance under OTDA LDSS-5006 effective October 1, 2025"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ny"
                / "sources"
                / "slices"
                / "otda"
                / "snap"
                / "current-effective"
                / "snap_standard_utility_allowance_ny.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_standard_utility_allowance"

    def test_repo_us_snap_ny_limited_utility_allowance_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ny_limited_utility_allowance_refresh.yaml"
        )

        assert manifest.name == "New York SNAP limited utility allowance refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_limited_utility_allowance_ny"
        assert (
            case.source_id
            == "New York SNAP limited utility allowance under OTDA LDSS-5006 effective October 1, 2025"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ny"
                / "sources"
                / "slices"
                / "otda"
                / "snap"
                / "current-effective"
                / "snap_limited_utility_allowance_ny.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_limited_utility_allowance"

    def test_repo_us_snap_fl_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_fl_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Florida SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_fl"
        assert (
            case.source_id
            == "Florida SNAP child support deduction election under ESS Program Policy Manual 2610.0410"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-fl"
                / "sources"
                / "slices"
                / "myflfamilies"
                / "ess"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_fl.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_fl_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_fl_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Florida SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_fl"
        assert (
            case.source_id
            == "Florida SNAP self-employment expense option under ESS Program Policy Manual 1810.0302"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-fl"
                / "sources"
                / "slices"
                / "myflfamilies"
                / "ess"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_fl.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_md_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_md_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Maryland SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_md"
        assert (
            case.source_id
            == "Maryland SNAP child support deduction election under SNAP Manual Section 408 Verification"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-md"
                / "sources"
                / "slices"
                / "dhs"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_md.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_md_self_employment_simplified_deduction_rate_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_md_self_employment_simplified_deduction_rate_refresh.yaml"
        )

        assert (
            manifest.name
            == "Maryland SNAP self-employment simplified deduction rate refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_simplified_deduction_rate_md"
        assert (
            case.source_id
            == "Maryland SNAP self-employment simplified deduction rate under SNAP Manual Section 104 Self-employed Households"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-md"
                / "sources"
                / "slices"
                / "dhs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_simplified_deduction_rate_md.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_simplified_deduction_rate"
        )

    def test_repo_us_snap_ga_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ga_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Georgia SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_ga"
        assert (
            case.source_id
            == "Georgia SNAP child support deduction election under DFCS SNAP Manual 3035 Verification"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ga"
                / "sources"
                / "slices"
                / "dfcs"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_ga.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_ga_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ga_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Georgia SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_ga"
        assert (
            case.source_id
            == "Georgia SNAP self-employment expense option under DFCS SNAP Manual 3425 Self-Employment Income"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ga"
                / "sources"
                / "slices"
                / "dfcs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_ga.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_ga_self_employment_simplified_deduction_rate_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ga_self_employment_simplified_deduction_rate_refresh.yaml"
        )

        assert (
            manifest.name
            == "Georgia SNAP self-employment simplified deduction rate refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_simplified_deduction_rate_ga"
        assert (
            case.source_id
            == "Georgia SNAP self-employment simplified deduction rate under DFCS SNAP Manual 3425 Self-Employment Income"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ga"
                / "sources"
                / "slices"
                / "dfcs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_simplified_deduction_rate_ga.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_simplified_deduction_rate"
        )

    def test_repo_us_snap_sc_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_sc_child_support_deduction_option_refresh.yaml"
        )

        assert (
            manifest.name
            == "South Carolina SNAP child support deduction option refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_sc"
        assert (
            case.source_id
            == "South Carolina SNAP child support deduction under SNAP Manual Vol 65 section 12.7"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-sc"
                / "sources"
                / "slices"
                / "scdss"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_sc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_sc_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_sc_self_employment_expense_option_refresh.yaml"
        )

        assert (
            manifest.name
            == "South Carolina SNAP self-employment expense option refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_sc"
        assert (
            case.source_id
            == "South Carolina SNAP self-employment expense option under SNAP Manual Vol 65 section 11.4(3)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-sc"
                / "sources"
                / "slices"
                / "scdss"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_sc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_sc_self_employment_simplified_deduction_rate_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_sc_self_employment_simplified_deduction_rate_refresh.yaml"
        )

        assert (
            manifest.name
            == "South Carolina SNAP self-employment simplified deduction rate refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_simplified_deduction_rate_sc"
        assert (
            case.source_id
            == "South Carolina SNAP self-employment simplified deduction rate under SNAP Manual Vol 65 section 11.4(3)"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-sc"
                / "sources"
                / "slices"
                / "scdss"
                / "snap"
                / "current-effective"
                / "snap_self_employment_simplified_deduction_rate_sc.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_simplified_deduction_rate"
        )

    def test_repo_us_snap_al_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_al_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Alabama SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_al"
        assert (
            case.source_id
            == "Alabama SNAP child support deduction under DHR Food Assistance Program Points of Eligibility Manual Appendix I section E"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-al"
                / "sources"
                / "slices"
                / "aldhr"
                / "poe"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_al.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_al_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_al_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Alabama SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_al"
        assert (
            case.source_id
            == "Alabama SNAP self-employment expense option under DHR Food Assistance Program Points of Eligibility Manual Chapter 11 section D"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-al"
                / "sources"
                / "slices"
                / "aldhr"
                / "poe"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_al.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )

    def test_repo_us_snap_al_self_employment_simplified_deduction_rate_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_al_self_employment_simplified_deduction_rate_refresh.yaml"
        )

        assert (
            manifest.name
            == "Alabama SNAP self-employment simplified deduction rate refresh"
        )
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_simplified_deduction_rate_al"
        assert (
            case.source_id
            == "Alabama SNAP self-employment simplified deduction rate under DHR Food Assistance Program Points of Eligibility Manual Chapter 11 section 1100"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-al"
                / "sources"
                / "slices"
                / "aldhr"
                / "poe"
                / "current-effective"
                / "snap_self_employment_simplified_deduction_rate_al.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_simplified_deduction_rate"
        )

    def test_repo_us_snap_ar_child_support_deduction_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ar_child_support_deduction_option_refresh.yaml"
        )

        assert manifest.name == "Arkansas SNAP child support deduction option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_state_uses_child_support_deduction_ar"
        assert (
            case.source_id
            == "Arkansas SNAP child support deduction under DHS SNAP Certification Manual section 6550 Child Support Deductions"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ar"
                / "sources"
                / "slices"
                / "ardhs"
                / "snap"
                / "current-effective"
                / "snap_state_uses_child_support_deduction_ar.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rule_hint == "snap_state_uses_child_support_deduction"

    def test_repo_us_snap_ar_self_employment_expense_option_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root
            / "benchmarks"
            / "us_snap_ar_self_employment_expense_option_refresh.yaml"
        )

        assert manifest.name == "Arkansas SNAP self-employment expense option refresh"
        assert manifest.mode == "repo-augmented"
        assert len(manifest.cases) == 1
        assert manifest.gates.min_policyengine_pass_rate == 1.0
        case = manifest.cases[0]
        assert case.kind == "source"
        assert case.name == "snap_self_employment_expense_based_deduction_applies_ar"
        assert (
            case.source_id
            == "Arkansas SNAP self-employment expense option under DHS SNAP Certification Manual section 5663 Costs of Producing Self-Employment Income"
        )
        assert (
            case.source_file
            == (
                repo_root.parent
                / "rules-us-ar"
                / "sources"
                / "slices"
                / "ardhs"
                / "snap"
                / "current-effective"
                / "snap_self_employment_expense_based_deduction_applies_ar.txt"
            ).resolve()
        )
        assert case.allow_context == []
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert (
            case.policyengine_rule_hint
            == "snap_self_employment_expense_based_deduction_applies"
        )


class TestReadinessSummary:
    def test_summarize_readiness_applies_suite_gates(self):
        gates = EvalReadinessGates(
            min_cases=3,
            min_success_rate=1.0,
            min_compile_pass_rate=1.0,
            min_ci_pass_rate=1.0,
            min_zero_ungrounded_rate=1.0,
            min_generalist_review_pass_rate=1.0,
            min_policyengine_pass_rate=0.8,
            max_mean_estimated_cost_usd=0.5,
        )
        results = [
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-a",
                compile_pass=True,
                ci_pass=True,
                policyengine_pass=True,
                policyengine_score=1.0,
                estimated_cost_usd=0.20,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-b",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=False,
                generalist_review_score=4.0,
                policyengine_pass=False,
                policyengine_score=0.5,
                estimated_cost_usd=0.40,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-c",
                compile_pass=True,
                ci_pass=True,
                generalist_review_pass=True,
                generalist_review_score=7.5,
                policyengine_pass=None,
                policyengine_score=None,
                estimated_cost_usd=0.30,
            ),
        ]

        summary = summarize_readiness(results, gates)

        assert summary.total_cases == 3
        assert summary.compile_pass_rate == 1.0
        assert summary.ci_pass_rate == 1.0
        assert summary.zero_ungrounded_rate == 1.0
        assert summary.generalist_review_pass_rate == pytest.approx(
            2 / 3, rel=0, abs=1e-6
        )
        assert summary.mean_generalist_review_score == pytest.approx(6.5)
        assert summary.policyengine_case_count == 2
        assert summary.policyengine_pass_rate == 0.5
        assert summary.mean_estimated_cost_usd == 0.3
        assert summary.ready is False
        gate_results = {gate.name: gate for gate in summary.gate_results}
        assert gate_results["min_cases"].passed is True
        assert gate_results["min_generalist_review_pass_rate"].passed is False
        assert gate_results["min_policyengine_pass_rate"].passed is False
        assert gate_results["max_mean_estimated_cost_usd"].passed is True


class TestRepoAugmentedContext:
    def test_prepare_eval_workspace_allows_arbitrary_identifier_with_explicit_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        context_file = (
            repo_root / "rules-us" / "statute" / "26" / "32" / "b" / "2" / "A.yaml"
        )
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(F)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="F. Determining Eligibility ... 165 345 518",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[context_file],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "26/32/b/2/A"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_select_context_files_excludes_target(self, tmp_path):
        statute_root = tmp_path / "rules-us" / "statute"
        section_dir = statute_root / "26" / "24"
        section_dir.mkdir(parents=True)
        (section_dir / "a.yaml").write_text("target")
        (section_dir / "b.yaml").write_text("sibling b")
        (section_dir / "c.yaml").write_text("sibling c")

        selected = select_context_files("26 USC 24(a)", statute_root)

        assert section_dir / "a.yaml" not in selected
        assert section_dir / "b.yaml" in selected
        assert section_dir / "c.yaml" in selected

    def test_prepare_eval_workspace_writes_manifest_and_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rules-us" / "statute" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["source_file"] == "source.txt"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        assert manifest["context_files"][0]["import_path"] == "26/24/b"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_prepare_eval_workspace_materializes_source_metadata_sidecar(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo = repo_root / "rules-us-tn"
        policy_repo.mkdir(parents=True)
        source_path = policy_repo / "sources" / "slices" / "tn" / "snap_sua.txt"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("Tennessee source text")
        source_metadata_path = source_path.with_name("snap_sua.meta.yaml")
        source_metadata_path.write_text(
            "version: 1\n"
            "relations:\n"
            "  - relation: sets\n"
            "    target: cfr/7/273.9/d/6/iii#snap_standard_utility_allowance\n"
            "    jurisdiction: TN\n"
        )

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="snap_sua_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="Tennessee source text",
            axiom_rules_path=repo_root / "axiom-rules",
            mode="cold",
            source_path=source_path,
            extra_context_paths=[],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["source_metadata_file"] == "source-metadata.json"
        assert (
            manifest["source_metadata"]["relations"][0]["target"]
            == "cfr/7/273.9/d/6/iii#snap_standard_utility_allowance"
        )
        assert workspace.source_metadata_file is not None
        assert workspace.source_metadata_file.exists()

    def test_build_eval_prompt_lists_canonical_context_import_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        external_file = (
            repo_root
            / "rules-us-co"
            / "regulation"
            / "9-CCR-2503-6"
            / "3.606.1"
            / "F.yaml"
        )
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text(
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: grant_standard_for_assistance_unit\n"
            "    kind: input\n"
            "    entity: TanfUnit\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(I)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Deduct the total from step 2, above, from the grant amount.",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[external_file],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(I)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="9-CCR-2503-6-3.606.1-I.yaml",
        )

        assert (
            "inspect `context/regulation/9-CCR-2503-6/3.606.1/F.yaml`; "
            "import target `regulation/9-CCR-2503-6/3.606.1/F`"
        ) in prompt
        assert "do not wrap import targets in quotes" in prompt
        assert (
            "Use the listed import target rather than the `./context/...` inspection path"
            in prompt
        )
        assert (
            "do not guess contradictory expectations for those imported values"
            in prompt
        )
        assert (
            "keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file"
            in prompt
        )
        assert (
            "Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0`"
            in prompt
        )
        assert (
            "Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file"
            in prompt
        )

    def test_hydrate_eval_root_copies_context_into_import_tree(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rules-us" / "statute" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "c.yaml"
        context_file.write_text(
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)

        assert (eval_root / "26" / "24" / "c.yaml").read_text() == (
            "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
        )

    def test_prepare_eval_workspace_expands_transitive_context_imports(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)

        section_root = repo_root / "rules-us" / "statute" / "26" / "24"
        section_root.mkdir(parents=True)
        aggregator = section_root / "24.yaml"
        aggregator.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - 26/24/a#ctc_allowance\n"
            "  - 26/24/c#qualifying_child_count\n"
            "rules:\n"
            "  - name: section_24_credit\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Money\n"
            "    period: Year\n"
        )
        selected = section_root / "c.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - 26/24/c/2#ctc_meets_citizenship_requirement\n"
            "  - 26/152/c#qualifying_child_of_taxpayer\n"
            "rules:\n"
            "  - name: qualifying_child_count\n"
            "    kind: derived\n"
            "    entity: TaxUnit\n"
            "    dtype: Integer\n"
            "    period: Year\n"
        )

        dep_local = section_root / "c" / "2.yaml"
        dep_local.parent.mkdir(parents=True)
        dep_local.write_text("format: rulespec/v1\nrules: []\n")

        dep_cross_section = repo_root / "rules-us" / "statute" / "26" / "152" / "c.yaml"
        dep_cross_section.parent.mkdir(parents=True)
        dep_cross_section.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[aggregator, selected],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(dep_local)] == "implementation_dependency"
        assert copied_sources[str(dep_cross_section)] == "implementation_dependency"
        assert str(section_root / "a.yaml") not in copied_sources

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert (
            eval_root / "26" / "24" / "c" / "2.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"
        assert (
            eval_root / "26" / "152" / "c.yaml"
        ).read_text() == "format: rulespec/v1\nrules: []\n"

    def test_repo_augmented_context_resolves_statute_prefixed_dependencies(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rules-us" / "statute" / "7" / "2014"
        statute_root.mkdir(parents=True)

        selected = statute_root / "e.yaml"
        selected.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - statute/7/2014/2014#snap_household_has_elderly_or_disabled_member\n"
            "  - statute/7/2014/d#snap_gross_income\n"
            "rules:\n"
            "  - name: snap_net_income\n"
            "    kind: derived\n"
            "    entity: Household\n"
            "    dtype: Money\n"
            "    period: Month\n"
        )

        section_file = statute_root / "2014.yaml"
        section_file.write_text("format: rulespec/v1\nrules: []\n")
        cross_file = statute_root / "d.yaml"
        cross_file.write_text("format: rulespec/v1\nrules: []\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="7 USC 2017(a)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="2017(a) ...",
            axiom_rules_path=policy_repo_root,
            mode="repo-augmented",
            extra_context_paths=[selected],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(section_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }
        assert copied_sources[str(cross_file)] in {
            "implementation_precedent",
            "implementation_dependency",
        }

    def test_prompt_includes_scaffold_dates_from_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        policy_repo_root = repo_root / "axiom-rules"
        policy_repo_root.mkdir(parents=True)
        statute_root = repo_root / "rules-us" / "statute" / "26" / "24"
        statute_root.mkdir(parents=True)
        context_file = statute_root / "b.yaml"
        context_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: The threshold is $1,000 and later $2,000.\n"
            "rules:\n"
            "  - name: threshold\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '1998-01-01'\n"
            "        formula: 1000\n"
            "      - effective_from: '2018-01-01'\n"
            "        formula: 2000\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "axiom_encode.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                axiom_rules_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        prompt = _build_eval_prompt(
            "26 USC 24(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            "a.yaml",
        )

        assert "`1998-01-01`" in prompt
        assert "`2018-01-01`" in prompt
        assert "Prefer the earliest scaffold date" in prompt


class TestUnexpectedAccessDetection:
    def test_flags_parent_directory_traversal(self, tmp_path):
        assert _command_looks_out_of_bounds("bash -lc 'find .. -name *.yaml'", tmp_path)

    def test_allows_workspace_paths(self, tmp_path):
        local = tmp_path / "context" / "b.yaml"
        local.parent.mkdir(parents=True)
        local.write_text("format: rulespec/v1\nrules: []\n")


class TestSourceEval:
    def test_run_source_eval_uses_explicit_context_without_statute_lookup(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules"
        policy_repo_root.mkdir()
        context_file = tmp_path / "examples" / "piecewise.yaml"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("format: rulespec/v1\nrules: []\n")

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: 9-CCR-2503-6-3.606.1-F.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: F. Determining Eligibility ...\n"
                "rules:\n"
                "  - name: grant_standard\n"
                "    kind: parameter\n"
                "    entity: TaxUnit\n"
                "    dtype: Money\n"
                "    period: Month\n"
                "    versions:\n"
                "      - effective_from: '2024-07-01'\n"
                "        formula: 165\n"
                "=== FILE: 9-CCR-2503-6-3.606.1-F.test.yaml ===\n"
                "- name: base case\n"
                "  period: 2024-07\n"
                "  input: {}\n"
                "  output:\n"
                "    grant_standard: 165\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None

            mock_evaluate_artifact.return_value = None

            results = run_source_eval(
                source_id="9 CCR 2503-6 3.606.1(F)",
                source_text="F. Determining Eligibility ... 165",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="repo-augmented",
                extra_context_paths=[context_file],
            )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert Path(result.output_file).exists()
        assert Path(result.output_file).with_suffix(".test.yaml").exists()
        assert result.retrieved_files == [str(context_file)]

        prompt = mock_prompt_eval.call_args.args[2]
        assert ".test.yaml" in prompt
        assert "=== FILE:" in prompt

    def test_run_source_eval_passes_oracle_settings_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules"
        policy_repo_root.mkdir()

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2006-965-regulation-2.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: https://www.legislation.gov.uk/uksi/2006/965/regulation/2 states 26.05.\n"
                "rules:\n"
                "  - name: child_benefit_enhanced_rate\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 26.05\n"
                "=== FILE: uksi-2006-965-regulation-2.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    child_benefit_enhanced_rate: 26.05\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_id="uksi/2006/965/regulation/2",
                source_text="26.05",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert mock_evaluate_artifact.call_args.kwargs["oracle"] == "policyengine"
        assert mock_evaluate_artifact.call_args.kwargs["policyengine_country"] == "uk"

    def test_run_source_eval_passes_policyengine_rule_hint_to_evaluate_artifact(
        self, tmp_path
    ):
        policy_repo_root = tmp_path / "axiom-rules"
        policy_repo_root.mkdir()

        with (
            patch(
                "axiom_encode.harness.evals._run_prompt_eval",
            ) as mock_prompt_eval,
            patch(
                "axiom_encode.harness.evals.evaluate_artifact",
            ) as mock_evaluate_artifact,
        ):
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.yaml ===\n"
                "format: rulespec/v1\n"
                "module:\n"
                "  summary: The amount is 317.82.\n"
                "rules:\n"
                "  - name: source_row_amount\n"
                "    kind: parameter\n"
                "    dtype: Money\n"
                "    unit: GBP\n"
                "    versions:\n"
                "      - effective_from: '2025-04-07'\n"
                "        formula: 317.82\n"
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.test.yaml ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    source_row_amount: 317.82\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_id="uksi/2013/376/regulation/36/3",
                source_text="317.82",
                runner_specs=["openai:gpt-5.4"],
                output_root=tmp_path / "out",
                policy_path=policy_repo_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
                policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
            )

        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_rule_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_build_eval_prompt_includes_policyengine_rule_hint(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="317.82",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36/3",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="uc_standard_allowance_single_claimant_aged_under_25",
        )

        assert "uc_standard_allowance_single_claimant_aged_under_25" in prompt
        assert "Keep `.test.yaml` inputs oracle-comparable" in prompt
        assert (
            "Prefer a contemporary monthly `.test.yaml` period like `2022-01` or `2024-01`"
            in prompt
        )
        assert (
            "assert `uc_standard_allowance_single_claimant_aged_under_25` directly in every non-empty `output:` mapping"
            in prompt
        )
        assert (
            "prefer the oracle's direct component facts over inverted household proxy inputs"
            in prompt
        )
        assert "assert that copied downstream output named by the oracle hint" in prompt
        assert (
            "avoid pre-2015 historical periods that PolicyEngine US cannot evaluate"
            in prompt
        )

    def test_build_eval_prompt_includes_sets_source_metadata_guidance(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        source_path = tmp_path / "snap_standard_utility_allowance_tn.txt"
        source_path.write_text("The SUA is $451, effective October 1, 2025.")
        source_path.with_name(
            "snap_standard_utility_allowance_tn.meta.yaml"
        ).write_text(
            "version: 1\n"
            "relations:\n"
            "  - relation: sets\n"
            "    target: cfr/7/273.9/d/6/iii#snap_standard_utility_allowance\n"
            "    jurisdiction: TN\n"
        )
        workspace = prepare_eval_workspace(
            citation="snap_standard_utility_allowance_tn",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="The SUA is $451, effective October 1, 2025.",
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            source_path=source_path,
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "snap_standard_utility_allowance_tn",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
            policyengine_rule_hint="snap_standard_utility_allowance",
        )

        assert "./source-metadata.json" in prompt
        assert "relation: sets" not in prompt
        assert '"relation": "sets"' in prompt
        assert (
            "setting the effective jurisdiction-specific value for that delegated slot"
            in prompt
        )
        assert "cfr/7/273.9/d/6/iii#snap_standard_utility_allowance" in prompt
        assert "...#*_applies` or `...#*_uses_*" in prompt
        assert (
            "do not add a top-level `imports:` entry to the bare canonical `cfr/...#...` or `usc/...#...` path"
            in prompt
        )
        assert "`*_is_in_state` or `*_is_in_jurisdiction`" in prompt
        assert (
            "use only positive/continuity cases rather than a fabricated out-of-jurisdiction false case"
            in prompt
        )
        assert (
            "encode the canonical boolean slot as a direct dated constant `true` or `false`"
            in prompt
        )
        assert (
            "omit an inapplicable false test unless `./source.txt` itself states a narrower in-jurisdiction condition"
            in prompt
        )

    def test_build_eval_prompt_single_amount_slice_disallows_speculative_future_tests(
        self, tmp_path
    ):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/2005/schedule/2",
            runner=runner,
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-06.\n\n"
                "Structured table:\n"
                "Relevant element | Maximum annual rate\n"
                "Severe disability element | £1734\n"
            ),
            axiom_rules_path=tmp_path / "axiom-rules",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/2005/schedule/2",
            "cold",
            workspace,
            [],
            target_file_name="example.yaml",
            include_tests=True,
            runner_backend="openai",
        )

        assert (
            "For a single fixed-amount source slice, a base case is sufficient."
            in prompt
        )
        assert (
            "For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test."
            in prompt
        )
        assert (
            "Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date."
            in prompt
        )
        assert "Do not add speculative future-period tests" in prompt

    def test_single_amount_slice_detection_excludes_conditional_money_leaf(self):
        assert (
            _is_single_amount_table_slice(
                "£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance."
            )
            is False
        )

    def test_normalize_nonannual_test_period_value_converts_iso_week_to_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2025-W13", date(2025, 3, 21))
            == "2025-03-21"
        )

    def test_normalize_nonannual_test_period_value_bumps_explicit_day_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value("2026-04-01", date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_bumps_yaml_date_before_effective_date(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(date(2026, 4, 1), date(2026, 4, 2))
            == "2026-04-02"
        )

    def test_normalize_nonannual_test_period_value_uses_month_period_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-10-01",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-10"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_normalize_nonannual_test_period_value_preserves_prior_day_month_for_monthly_rules(
        self,
    ):
        assert (
            _normalize_nonannual_test_period_value(
                "2025-09-30",
                date(2025, 10, 1),
                granularity="Month",
            )
            == "2025-09"
        )

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/26/24/b.yaml'"
        assert not _command_looks_out_of_bounds(command, tmp_path)


def _fake_eval_result(
    runner: str,
    citation: str,
    *,
    compile_pass: bool = True,
    ci_pass: bool = True,
    generalist_review_pass: bool | None = True,
    generalist_review_score: float | None = 8.0,
    policyengine_pass: bool | None = None,
    policyengine_score: float | None = None,
    estimated_cost_usd: float | None = 0.25,
    ungrounded_numeric_count: int = 0,
) -> EvalResult:
    return EvalResult(
        citation=citation,
        runner=runner,
        backend="codex",
        model="gpt-5.4",
        mode="cold",
        output_file=f"/tmp/{citation}.yaml",
        trace_file=f"/tmp/{citation}.json",
        context_manifest_file=f"/tmp/{citation}.manifest.json",
        duration_ms=1000,
        success=True,
        error=None,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=EvalArtifactMetrics(
            compile_pass=compile_pass,
            compile_issues=[],
            ci_pass=ci_pass,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1 if ungrounded_numeric_count == 0 else 0,
            ungrounded_numeric_count=ungrounded_numeric_count,
            grounding=[
                GroundingMetric(
                    line=1,
                    raw="26.05",
                    value=26.05,
                    grounded=ungrounded_numeric_count == 0,
                )
            ],
            generalist_review_pass=generalist_review_pass,
            generalist_review_score=generalist_review_score,
            generalist_review_issues=[],
            policyengine_pass=policyengine_pass,
            policyengine_score=policyengine_score,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        ),
    )
