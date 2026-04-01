"""Tests for model comparison eval helpers."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from autorac.harness.evals import (
    EvalArtifactMetrics,
    EvalReadinessGates,
    EvalResult,
    FetchedLegislationGovUkDocument,
    GroundingMetric,
    _build_eval_prompt,
    _clean_generated_file_content,
    _command_looks_out_of_bounds,
    _fetch_legislation_gov_uk_document,
    _hydrate_eval_root,
    _materialize_eval_artifact,
    _normalize_legislation_gov_uk_source_ref,
    _post_openai_eval_request,
    _resolve_akn_section_eid,
    _run_codex_prompt_eval,
    _wait_for_codex_process,
    evaluate_artifact,
    extract_akn_section_text,
    load_eval_suite_manifest,
    parse_runner_spec,
    prepare_eval_workspace,
    run_akn_section_eval,
    run_eval_suite,
    run_legislation_gov_uk_section_eval,
    run_source_eval,
    select_context_files,
    summarize_readiness,
)
from autorac.harness.validator_pipeline import ValidationResult, ValidatorPipeline


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
    def test_wait_for_codex_process_terminates_after_stable_last_message(self, tmp_path):
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

    def test_run_codex_prompt_eval_accepts_stable_last_message_on_termination(self, tmp_path):
        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/3/a",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="nil amount",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = (
            "=== FILE: example.rac ===\n"
            "status: encoded\n"
        )
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

        def fake_wait(process, last_message_file, timeout, settle_seconds=5.0, poll_interval=0.5):
            process.terminate()
            process.wait()
            return True

        with patch("autorac.harness.evals.subprocess.Popen", FakePopen), patch(
            "autorac.harness.evals._wait_for_codex_process",
            side_effect=fake_wait,
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        bundle = (
            "=== FILE: example.rac ===\n"
            "status: encoded\n"
        )

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

        with patch("autorac.harness.evals.subprocess.Popen", FakePopen), patch(
            "autorac.harness.evals._wait_for_codex_process",
            side_effect=subprocess.TimeoutExpired(cmd=["codex", "exec"], timeout=600),
        ):
            response = _run_codex_prompt_eval(runner, workspace, "prompt")

        assert response.error is None
        assert response.text == bundle.strip()


class TestEvaluateArtifact:
    def test_uses_fallback_source_text_for_grounding(self, tmp_path):
        rac_file = tmp_path / "24" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '"""\n(a) Allowance of credit There shall be allowed a credit of $1,000.\n"""\n'
            "status: encoded\n\n"
            "ctc_amount:\n"
            "    description: \"Child tax credit amount\"\n"
            "    unit: USD\n"
            "    from 2018-01-01: 1000\n"
            "    from 2025-01-01: 2200\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)

        with patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
            return_value=compile_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_ci",
            return_value=ci_result,
        ):
            metrics = evaluate_artifact(
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
                source_text="(a) Allowance of credit There shall be allowed a credit of $1,000.",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 1
        assert [item.raw for item in metrics.grounding if not item.grounded] == ["2200"]

    def test_fails_ci_when_repeated_source_scalar_has_only_one_named_definition(
        self, tmp_path
    ):
        rac_file = tmp_path / "example.rac"
        rac_file.write_text(
            '''"""
2A. Where earnings are less than £20 in any week and would not exceed £20.
"""

pc_special_employment_maximum_weekly_amount:
    entity: Person
    period: Week
    dtype: Money
    from 2025-03-31:
        20
'''
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
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
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
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_clean_generated_file_content_strips_inline_currency_suffixes(self):
        content = (
            "child_benefit_enhanced_rate_amount:\n"
            "  from 2025-04-07: 26.05 GBP\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate_amount: 26.05 GBP\n"
        )

        cleaned = _clean_generated_file_content(content)

        assert "26.05 GBP" not in cleaned
        assert "26.05" in cleaned

    def test_materialize_eval_artifact_normalizes_arithmetic_single_amount_rows(
        self, tmp_path
    ):
        response = (
            "=== FILE: example.rac ===\n"
            '"""\n£22,020 for joint claimants not resident in Greater London.\n"""\n\n'
            "benefit_cap_amount:\n"
            "    entity: Family\n"
            "    period: Day\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21:\n"
            "        if is_joint_claimant and not resident_in_greater_london: 2025 * (7 + 3 + 1) - 21 * (7 + 3 + 2) - 3 else: 0\n"
            "=== FILE: example.rac.test ===\n"
            "- name: base\n"
            "  period: 2025-03-21\n"
            "  input:\n"
            "    is_joint_claimant: true\n"
            "    resident_in_greater_london: false\n"
            "  output:\n"
            "    benefit_cap_amount: 22020\n"
        )
        expected = tmp_path / "source" / "example.rac"

        wrote = _materialize_eval_artifact(
            response,
            expected,
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "£22,020 for joint claimants not resident in Greater London.\n"
            ),
        )

        assert wrote is True
        content = expected.read_text()
        assert "2025 * (7 + 3 + 1)" not in content
        assert "from 2025-03-21: 22020" in content

    def test_materialize_eval_artifact_cleans_bundled_fences(self, tmp_path):
        output_file = tmp_path / "source" / "uksi-2006-965-regulation-2.rac"
        llm_response = (
            "=== FILE: uksi-2006-965-regulation-2.rac ===\n"
            "```\n"
            "child_benefit_enhanced_rate:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "```\n"
            "=== FILE: uksi-2006-965-regulation-2.rac.test ===\n"
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "Trailing prose.\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        assert output_file.read_text() == (
            "child_benefit_enhanced_rate:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
        )
        assert output_file.with_suffix(".rac.test").read_text() == (
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_materialize_eval_artifact_salvages_workspace_files_when_response_is_summary(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-2025-03-31.rac"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            "pc_housing_non_dependant_deduction_other_weekly_amount:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21:\n"
            "        19.30\n"
        )
        (workspace_root / output_file.with_suffix(".rac.test").name).write_text(
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
        assert output_file.read_text().startswith(
            "pc_housing_non_dependant_deduction_other_weekly_amount:\n"
        )
        assert output_file.with_suffix(".rac.test").exists()

    def test_materialize_eval_artifact_prefers_workspace_files_over_prose_bundle(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2013-376-2025-04-07.rac"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True)
        (workspace_root / output_file.name).write_text(
            '"""\nCapital limit.\n"""\n\n'
            "uc_capital_limit_single_claimant_amount:\n"
            "    entity: Person\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-04-07:\n"
            "        16000\n"
        )
        (workspace_root / output_file.with_suffix(".rac.test").name).write_text(
            "- name: base_case\n"
            "  period: 2025\n"
            "  input: {}\n"
            "  output:\n"
            "    uc_capital_limit_single_claimant_amount: 16000\n"
        )
        llm_response = (
            "=== FILE: uksi-2013-376-2025-04-07.rac ===\n"
            "Encodes regulation 18(1)(a) of UKSI 2013/376.\n\n"
            "=== FILE: uksi-2013-376-2025-04-07.rac.test ===\n"
            "Single base case confirming the amount is 16000.\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text="£16,000",
            workspace_root=workspace_root,
        )

        assert wrote is True
        assert output_file.read_text().startswith('"""')
        assert "Encodes regulation" not in output_file.read_text()

    def test_materialize_eval_artifact_normalizes_single_row_block_conditional_and_tests(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-2005-schedule-2-second-adult-element.rac"
        source_text = (
            "Editorial note: current text valid from 2024-04-06.\n\n"
            "Structured table:\n"
            "Relevant element of working tax credit | Maximum annual rate\n"
            "4. Second adult element | £2,500"
        )
        llm_response = (
            "=== FILE: uksi-2002-2005-schedule-2-second-adult-element.rac ===\n"
            "wtc_second_adult_element_eligible:\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Boolean\n\n"
            "wtc_second_adult_element_amount:\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2024-04-06:\n"
            "        if wtc_second_adult_element_eligible: 2,500 else: 0\n"
            "=== FILE: uksi-2002-2005-schedule-2-second-adult-element.rac.test ===\n"
            "base_case:\n"
            "  period: 2024-04-06\n"
            "  input:\n"
            "    wtc_second_adult_element_eligible: true\n"
            "  output:\n"
            "    wtc_second_adult_element_amount: 2,500\n\n"
            "alternate_branch:\n"
            "  period: 2024-04-06\n"
            "  input:\n"
            "    wtc_second_adult_element_eligible: false\n"
            "  output:\n"
            "    wtc_second_adult_element_amount: 0\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        assert "2,500" not in output_file.read_text()
        assert "if wtc_second_adult_element_eligible" not in output_file.read_text()
        assert "from 2024-04-06: 2500" in output_file.read_text()
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "alternate_branch" not in test_text
        assert "2500" in test_text

    def test_materialize_eval_artifact_drops_annual_effective_date_boundary_for_single_row(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-2005-schedule-2-worker-element.rac"
        source_text = (
            "Editorial note: text valid from 2021-04-06.\n\n"
            "Structured table:\n"
            "Relevant element of working tax credit | Maximum annual rate\n"
            "30 hour element | £830"
        )
        llm_response = (
            "=== FILE: uksi-2002-2005-schedule-2-worker-element.rac ===\n"
            "wtc_worker_element_amount:\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2021-04-06: 830\n"
            "=== FILE: uksi-2002-2005-schedule-2-worker-element.rac.test ===\n"
            "- name: base_case\n"
            "  period: 2021\n"
            "  output:\n"
            "    wtc_worker_element_amount: 830\n"
            "- name: effective_date_boundary\n"
            "  period: 2022\n"
            "  output:\n"
            "    wtc_worker_element_amount: 830\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "base_case" in test_text
        assert "effective_date_boundary" not in test_text

    def test_materialize_eval_artifact_infers_annual_base_period_when_missing(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-2005-schedule-2-worker-element.rac"
        source_text = (
            "Editorial note: text valid from 2021-04-06.\n\n"
            "Structured table:\n"
            "Relevant element of working tax credit | Maximum annual rate\n"
            "30 hour element | £830"
        )
        llm_response = (
            "=== FILE: uksi-2002-2005-schedule-2-worker-element.rac ===\n"
            "wtc_worker_element_amount:\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2021-04-06: 830\n"
            "=== FILE: uksi-2002-2005-schedule-2-worker-element.rac.test ===\n"
            "- name: base_case\n"
            "  output:\n"
            "    wtc_worker_element_amount: 830\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "period: 2021" in test_text

    def test_materialize_eval_artifact_normalizes_single_row_inline_conditional_without_else(
        self, tmp_path
    ):
        output_file = (
            tmp_path / "source" / "uksi-2013-376-regulation-36-3-single-25-or-over.rac"
        )
        source_text = (
            "Editorial note: current text valid from 2025-04-07.\n\n"
            "Structured table:\n"
            "Element | Amount for each assessment period\n"
            "single claimant aged 25 or over | £400.14"
        )
        llm_response = (
            "=== FILE: uksi-2013-376-regulation-36-3-single-25-or-over.rac ===\n"
            "uc_claimant_is_single:\n"
            "    entity: TaxUnit\n"
            "    period: Month\n"
            "    dtype: Boolean\n\n"
            "uc_claimant_aged_25_or_over:\n"
            "    entity: TaxUnit\n"
            "    period: Month\n"
            "    dtype: Boolean\n\n"
            "uc_standard_allowance_single_claimant_aged_25_or_over:\n"
            "    entity: TaxUnit\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-04-07: if uc_claimant_is_single and uc_claimant_aged_25_or_over: 400.14\n"
            "=== FILE: uksi-2013-376-regulation-36-3-single-25-or-over.rac.test ===\n"
            "- name: base_case\n"
            "  period: 2025-04\n"
            "  input:\n"
            "    uc_claimant_is_single: true\n"
            "    uc_claimant_aged_25_or_over: true\n"
            "  output:\n"
            "    joint_claimants_where_either_is_aged_25_or_over: true\n"
            "    uc_standard_allowance_single_claimant_aged_25_or_over: 400.14\n"
            "- name: alternate_real_condition_single_status_varied\n"
            "  period: 2025-04\n"
            "  input:\n"
            "    uc_claimant_is_single: false\n"
            "    uc_claimant_aged_25_or_over: true\n"
            "  output:\n"
            "    uc_standard_allowance_single_claimant_aged_25_or_over: 400.14\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        rac_text = output_file.read_text()
        assert "from 2025-04-07: 400.14" in rac_text
        assert "if uc_claimant_is_single" not in rac_text
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "alternate_real_condition_single_status_varied" not in test_text
        assert "joint_claimants_where_either_is_aged_25_or_over" not in test_text

    def test_materialize_eval_artifact_normalizes_non_slice_code_numeric_literals(
        self, tmp_path
    ):
        output_file = (
            tmp_path / "source" / "uksi-2013-376-regulation-80A-2025-04-01.rac"
        )
        llm_response = (
            '"""\n'
            "The applicable annual limit is £25,323 for joint claimants.\n"
            '"""\n\n'
            "regulation_80A_2_b_i_applicable_annual_limit:\n"
            "    entity: Family\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    from 2025-03-21:\n"
            "        if is_joint_claimant and either_joint_claimant_resident_in_greater_london: 25,323\n"
            "        else: 0\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        artifact_text = output_file.read_text()
        assert "£25,323" in artifact_text
        code_text = artifact_text.split('"""', 2)[-1]
        assert "25,323" not in code_text
        assert (
            "if is_joint_claimant and either_joint_claimant_resident_in_greater_london: 25323"
            in artifact_text
        )

    def test_can_include_policyengine_metrics_for_uk_artifact(self, tmp_path):
        rac_file = tmp_path / "source" / "uksi-2006-965-regulation-2.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '"""\nhttps://www.legislation.gov.uk/uksi/2006/965/regulation/2\n"""\n'
            "status: encoded\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)
        pe_result = ValidationResult(
            "policyengine",
            passed=True,
            score=1.0,
            issues=[],
        )

        with patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
            return_value=compile_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_ci",
            return_value=ci_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_policyengine",
            return_value=pe_result,
        ) as mock_policyengine:
            metrics = evaluate_artifact(
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
                source_text="26.05",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.policyengine_pass is True
        assert metrics.policyengine_score == 1.0
        assert metrics.policyengine_issues == []
        mock_policyengine.assert_called_once()


class TestAknSectionEval:
    def test_extract_akn_section_text_supports_standard_akn_subsection_tags(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content>
            <p>Example subsection text.</p>
          </content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1-1")

        assert "1 Example section" in text
        assert "(1)" in text
        assert "Example subsection text." in text

    def test_extract_akn_section_text_includes_parent_intro_for_leaf_levels(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <intro>
            <p>Lead-in text for child levels.</p>
          </intro>
          <level eId="section-1-1-a">
            <num>(a)</num>
            <content>
              <p>Leaf paragraph text.</p>
            </content>
          </level>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1-1-a")

        assert "1 Example section" in text
        assert "(1)" in text
        assert "Lead-in text for child levels." in text
        assert "(a)" in text
        assert "Leaf paragraph text." in text

    def test_extract_akn_section_text_includes_heading_and_table_rows(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <content>
          <p>A. Payment of Basic Cash Assistance Grants</p>
          <table>
            <tr><th>Children</th><th>Grant</th></tr>
            <tr><td>0</td><td>165</td></tr>
            <tr><td>1</td><td>345</td></tr>
          </table>
          <p>G. Pregnancy allowance text.</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "sec_3_606_1")

        assert "3.606.1 Basic Cash Assistance" in text
        assert "A. Payment of Basic Cash Assistance Grants" in text
        assert "Structured table:" in text
        assert "Children | Grant" in text
        assert "0 | 165" in text
        assert "G. Pregnancy allowance text." in text

    def test_extract_akn_section_text_can_target_specific_table_row(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <hcontainer eId="regulation-36">
        <num>36.</num>
        <heading>Table showing amounts of elements</heading>
        <paragraph eId="regulation-36-3">
          <num>(3)</num>
          <content>
            <p>In the case of an award where the claimant is a member of a couple, but claims as a single person, the amounts are those shown in the table for a single claimant.</p>
            <table>
              <tr><td>Element</td><td>Amount for each assessment period</td></tr>
              <tr><td>Standard allowance</td><td></td></tr>
              <tr><td>single claimant aged under 25</td><td>£316.98</td></tr>
              <tr><td>single claimant aged 25 or over</td><td>£400.14</td></tr>
            </table>
          </content>
        </paragraph>
      </hcontainer>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(
            akn_file,
            "regulation-36-3",
            table_row_query="single claimant aged under 25",
        )

        assert "36. Table showing amounts of elements" in text
        assert "(3)" in text
        assert "Standard allowance" in text
        assert "single claimant aged under 25 | £316.98" in text
        assert "single claimant aged 25 or over" not in text
        assert "the amounts are those shown in the table for a single claimant" not in text

    def test_extract_akn_section_text_matches_table_row_query_despite_inline_spacing_edits(
        self, tmp_path
    ):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <paragraph eId="regulation-36-3">
        <num>(3)</num>
        <content>
          <table>
            <tr><td>Element</td><td>Amount</td></tr>
            <tr><td>Child element—</td><td></td></tr>
            <tr><td>second and each subsequenteach child or qualifying young person</td><td>£292.81</td></tr>
          </table>
        </content>
      </paragraph>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(
            akn_file,
            "regulation-36-3",
            table_row_query="second and each subsequent each child or qualifying young person",
        )

        assert "second and each subsequenteach child or qualifying young person | £292.81" in text
        assert "Child element—" in text
        assert "Element | Amount" in text
        assert "single claimant aged under 25" not in text

    def test_extract_akn_section_text_includes_editorial_effective_date(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1_f">
        <num>F</num>
        <heading>Grant Standard</heading>
        <remark status="editorial">Effective date: 2024-07-01</remark>
        <content>
          <p>Grant table text.</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "sec_3_606_1_f")

        assert "F Grant Standard" in text
        assert "Effective date: 2024-07-01" in text
        assert "Grant table text." in text

    def test_extract_akn_section_text_includes_expression_valid_from_date(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <meta>
      <identification source="#">
        <FRBRExpression>
          <FRBRdate date="2025-04-07" name="validFrom"/>
        </FRBRExpression>
      </identification>
    </meta>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example</heading>
        <content>
          <p>Current amount is 26.05.</p>
        </content>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1")

        assert "Editorial note: current text valid from 2025-04-07." in text
        assert "Current amount is 26.05." in text

    def test_run_akn_section_eval_uses_extracted_section_text(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <content>
          <p>A. Payment of Basic Cash Assistance Grants</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        with patch(
            "autorac.harness.evals.run_source_eval",
            return_value=["ok"],
        ) as mock_run_source_eval:
            results = run_akn_section_eval(
                source_id="9 CCR 2503-6 3.606.1",
                akn_file=akn_file,
                section_eid="sec_3_606_1",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                mode="cold",
                extra_context_paths=[],
            )

        assert results == ["ok"]
        mock_run_source_eval.assert_called_once()
        assert (
            mock_run_source_eval.call_args.kwargs["source_text"]
            == "3.606.1 Basic Cash Assistance\n\nA. Payment of Basic Cash Assistance Grants"
        )
        assert mock_run_source_eval.call_args.kwargs["oracle"] == "none"
        assert mock_run_source_eval.call_args.kwargs["policyengine_country"] == "auto"

    def test_run_akn_section_eval_rejects_parent_section_by_default(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <hcontainer name="section" eId="sec_3_606_1_f">
          <num>F</num>
          <content><p>Grant standard table text.</p></content>
        </hcontainer>
        <hcontainer name="section" eId="sec_3_606_1_g">
          <num>G</num>
          <content><p>Pregnancy allowance text.</p></content>
        </hcontainer>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        with patch("autorac.harness.evals.run_source_eval") as mock_run_source_eval:
            try:
                run_akn_section_eval(
                    source_id="9 CCR 2503-6 3.606.1",
                    akn_file=akn_file,
                    section_eid="sec_3_606_1",
                    runner_specs=["codex:gpt-5.4"],
                    output_root=tmp_path / "out",
                    rac_path=tmp_path / "rac",
                    mode="cold",
                    extra_context_paths=[],
                )
            except ValueError as exc:
                message = str(exc)
            else:
                raise AssertionError("Expected parent-section guardrail to raise")

        assert "Choose an atomic child section instead" in message
        assert "sec_3_606_1_f" in message
        assert "sec_3_606_1_g" in message
        mock_run_source_eval.assert_not_called()

    def test_resolve_akn_section_eid_allows_parent_when_opted_in(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <hcontainer name="section" eId="sec_3_606_1_f">
          <num>F</num>
          <content><p>Grant standard table text.</p></content>
        </hcontainer>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        assert (
            _resolve_akn_section_eid(
                akn_file,
                "sec_3_606_1",
                allow_parent=True,
            )
            == "sec_3_606_1"
        )

    def test_resolve_akn_section_eid_rejects_standard_akn_parent_sections(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content><p>Example subsection text.</p></content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        try:
            _resolve_akn_section_eid(akn_file, "section-1")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected parent-section guardrail to raise")

        assert "Choose an atomic child section instead" in message
        assert "section-1-1" in message


class TestUkLegislationFetch:
    def test_normalize_legislation_gov_uk_source_ref_strips_data_extension(self):
        source_id, content_url = _normalize_legislation_gov_uk_source_ref(
            "https://www.legislation.gov.uk/ukpga/2010/1/section/1/data.akn"
        )

        assert source_id == "ukpga/2010/1/section/1"
        assert content_url == "https://www.legislation.gov.uk/ukpga/2010/1/section/1"

    def test_fetch_legislation_gov_uk_document_writes_akn_and_clml(self, tmp_path):
        class FakeResponse:
            def __init__(self, text: str):
                self.text = text

            def raise_for_status(self):
                return None

        with patch(
            "requests.get",
            side_effect=[
                FakeResponse("<akomaNtoso/>"),
                FakeResponse("<Legislation/>"),
            ],
        ) as mock_get:
            fetched = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                tmp_path,
            )

        assert fetched.source_id == "ukpga/2010/1/section/1"
        assert fetched.akn_file.read_text() == "<akomaNtoso/>"
        assert fetched.clml_file.read_text() == "<Legislation/>"
        assert mock_get.call_args_list[0].args[0].endswith("/data.akn")
        assert mock_get.call_args_list[1].args[0].endswith("/data.xml")

    def test_fetch_legislation_gov_uk_document_uses_cache_when_files_exist(
        self, tmp_path
    ):
        source_dir = tmp_path / "_legislation_gov_uk" / "ukpga-2010-1-section-1"
        source_dir.mkdir(parents=True)
        (source_dir / "source.akn").write_text("<cached-akn/>")
        (source_dir / "source.xml").write_text("<cached-clml/>")

        with patch("requests.get") as mock_get:
            fetched = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                tmp_path,
            )

        assert fetched.akn_file.read_text() == "<cached-akn/>"
        assert fetched.clml_file.read_text() == "<cached-clml/>"
        mock_get.assert_not_called()

    def test_fetch_legislation_gov_uk_document_uses_shared_fetch_cache_root(
        self, tmp_path
    ):
        class FakeResponse:
            def __init__(self, text: str):
                self.text = text
                self.status_code = 200

            def raise_for_status(self):
                return None

        shared_cache_root = tmp_path / "suite-cache"
        first_output = tmp_path / "case-a"
        second_output = tmp_path / "case-b"

        with patch(
            "requests.get",
            side_effect=[
                FakeResponse("<akomaNtoso/>"),
                FakeResponse("<Legislation/>"),
            ],
        ) as mock_get:
            first = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                first_output,
                fetch_cache_root=shared_cache_root,
            )
            second = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                second_output,
                fetch_cache_root=shared_cache_root,
            )

        assert first.akn_file == second.akn_file
        assert first.clml_file == second.clml_file
        assert str(first.akn_file).startswith(str(shared_cache_root))
        assert mock_get.call_count == 2

    def test_fetch_legislation_gov_uk_document_retries_transient_http_errors(
        self, tmp_path
    ):
        class FakeResponse:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(response=self)
                return None

        with patch(
            "requests.get",
            side_effect=[
                FakeResponse("bad gateway", 502),
                FakeResponse("<akomaNtoso/>"),
                FakeResponse("<Legislation/>"),
            ],
        ) as mock_get, patch("time.sleep"):
            fetched = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                tmp_path,
            )

        assert fetched.akn_file.read_text() == "<akomaNtoso/>"
        assert fetched.clml_file.read_text() == "<Legislation/>"
        assert mock_get.call_count == 3

    def test_fetch_legislation_gov_uk_document_allows_missing_clml_when_akn_succeeds(
        self, tmp_path
    ):
        class FakeResponse:
            def __init__(self, text: str, status_code: int = 200):
                self.text = text
                self.status_code = status_code

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(response=self)
                return None

        with patch(
            "requests.get",
            side_effect=[
                FakeResponse("<akomaNtoso/>"),
                requests.exceptions.ReadTimeout("timed out"),
                requests.exceptions.ReadTimeout("timed out"),
                requests.exceptions.ReadTimeout("timed out"),
                requests.exceptions.ReadTimeout("timed out"),
                requests.exceptions.ReadTimeout("timed out"),
                requests.exceptions.ReadTimeout("timed out"),
            ],
        ) as mock_get, patch("time.sleep"):
            fetched = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                tmp_path,
            )

        assert fetched.akn_file.read_text() == "<akomaNtoso/>"
        assert "unavailable" in fetched.clml_file.read_text().lower()
        assert mock_get.call_count == 7

    def test_run_legislation_gov_uk_section_eval_uses_primary_akn_node_when_unspecified(
        self, tmp_path
    ):
        fetched_dir = tmp_path / "fetched"
        fetched_dir.mkdir()
        akn_file = fetched_dir / "source.akn"
        clml_file = fetched_dir / "source.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content><p>Example subsection text.</p></content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )
        clml_file.write_text("<Legislation/>")

        with patch(
            "autorac.harness.evals._fetch_legislation_gov_uk_document"
        ) as mock_fetch, patch(
            "autorac.harness.evals.run_akn_section_eval",
            return_value=["ok"],
        ) as mock_run:
            mock_fetch.return_value.source_id = "ukpga/2010/1/section/1"
            mock_fetch.return_value.content_url = (
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1"
            )
            mock_fetch.return_value.akn_file = akn_file
            mock_fetch.return_value.clml_file = clml_file

            results = run_legislation_gov_uk_section_eval(
                source_ref="https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                mode="cold",
                extra_context_paths=[],
            )

        assert results == ["ok"]
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["section_eid"] == "section-1"
        assert mock_run.call_args.kwargs["source_id"] == "ukpga/2010/1/section/1"
        assert mock_run.call_args.kwargs["oracle"] == "policyengine"
        assert mock_run.call_args.kwargs["policyengine_country"] == "uk"


class TestEvalPrompt:
    def test_build_eval_prompt_includes_rac_syntax_guardrails(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Grant standard is 165 for one child.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1.rac",
            include_tests=True,
        )

        assert "Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`." in prompt
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "dtype:" in prompt

    def test_build_eval_prompt_includes_supported_schema_enums(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 25.60 ... 16.95",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "Do not invent new entities, periods, or dtypes." in prompt
        assert "Allowed `entity:` values are `Person`, `TaxUnit`, `Household`, `Family`" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "ukpga/2010/1/section/1",
            "cold",
            workspace,
            [],
            target_file_name="ukpga-2010-1-section-1.rac",
            include_tests=True,
        )

        assert "If the source cannot be represented faithfully with the supported schema" in prompt
        assert "`status: entity_not_supported`" in prompt
        assert "`status: deferred`" in prompt

    def test_build_eval_prompt_forbids_python_inline_ternaries(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "Do not use Python inline ternaries" in prompt
        assert "`x if cond else y`" in prompt

    def test_build_eval_prompt_requires_rac_conditional_expression_syntax(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "`if condition: value else: other_value`" in prompt
        assert "Do not use YAML-style `if:` / `then:` / `else:` blocks." in prompt
        assert "Do not append a multiline conditional directly onto another expression" in prompt

    def test_build_eval_prompt_requires_decimal_ratios_for_rate_dtype(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The percentage prescribed is 60 per cent.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-7.rac",
            include_tests=True,
        )

        assert "For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals." in prompt
        assert "Do not respond with summaries like `Both files written`" in prompt
        assert "Do not use inline assignment syntax like `:=` inside `from` blocks" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert 'Prefer `Person` when the source states an amount or condition "in respect of"' in prompt
        assert "do not collapse it into an unconditional family-level constant" in prompt

    def test_build_eval_prompt_for_uk_leaf_forbids_speculative_future_period_tests(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "The `.rac.test` file must contain YAML only" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert "must vary a real legal condition" in prompt
        assert "must contain factual predicates or quantities, not the output variable" in prompt
        assert "Use `output:` mappings in `.rac.test` cases" in prompt

    def test_build_eval_prompt_for_uk_branch_leaves_requires_branch_specific_names(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) 332.95 per week in the case of a claimant who has a partner.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "encode the branch identity in the output variable name" in prompt
        assert "`standard_minimum_guarantee`" in prompt
        assert "`child_benefit_weekly_rate`" in prompt

    def test_build_eval_prompt_for_uk_leaf_discourages_opaque_condition_helpers(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) ... only person or elder or eldest person ... £26.05.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2013-376-regulation-36.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Use a descriptive legal variable name" in prompt
        assert "not a path- or source-id-derived placeholder" in prompt
        assert "do not invent a fresh `*_applies` helper" in prompt
        assert "do not invent alternate zero-amount tests" in prompt
        assert "Do not emit `otherwise:`" in prompt
        assert "Do not emit `before YYYY-MM-DD: 0`" in prompt
        assert "Do not emit stray blocks like `from 0:`" in prompt
        assert "use boolean or fact-shaped helper inputs" in prompt
        assert "Do not invent sample ages like `2`, `3`, `24`, or `25`" in prompt
        assert "keep `.rac.test` outputs scalar" in prompt
        assert "keep the row-defining conditions satisfied" in prompt
        assert "principal amount variable should usually be a grounded constant" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/2A",
            "cold",
            workspace,
            [],
            target_file_name="example.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RAC" in prompt
        assert "If the same numeric value appears twice in materially different legal roles" in prompt
        assert 'If `./source.txt` says someone is "aged 18 or over", "under 25"' in prompt
        assert "Do not create scalar variables for citation numbers" in prompt
        assert "Do not invent `dtype: String` variables just to restate the effective date" in prompt

    def test_build_eval_prompt_includes_import_vs_local_helper_protocol(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="26 USC 24(c)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text='The term "qualifying child" means a qualifying child as defined in section 152(c).',
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "26 USC 24(c)",
            "cold",
            workspace,
            [],
            target_file_name="24-c.rac",
        )

        assert "emit the upstream import instead of restating the concept locally" in prompt
        assert "still emit the unresolved import path" in prompt
        assert "otherwise keep the helper local to this leaf" in prompt

    def test_build_eval_prompt_discourages_fabricated_same_instrument_imports(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/6/5/a",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) except where paragraph (b) applies, £81.50 per week if paragraph 1(1)(a), (b) or (c) of Part I of Schedule I is satisfied.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/6/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-6-5-a.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not fabricate sibling-file imports" in prompt
        assert "do not guess" in prompt
        assert "instead of inventing `import` statements or `imports:` blocks" in prompt

    def test_build_eval_prompt_for_openai_inlines_source_text(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "You do not have filesystem tool access in this eval" in prompt
        assert "=== BEGIN SOURCE.TXT ===" in prompt
        assert "Editorial note: current text valid from 2025-04-07." in prompt
        assert "26.05" in prompt


class TestOpenAIEvalRequest:
    def test_post_openai_eval_request_retries_transient_status(self):
        error_response = Mock()
        error_response.status_code = 502
        ok_response = Mock()
        ok_response.status_code = 200

        with patch("autorac.harness.evals.requests.post") as mock_post, patch(
            "autorac.harness.evals.time.sleep"
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

        with patch("autorac.harness.evals.requests.post") as mock_post, patch(
            "autorac.harness.evals.time.sleep"
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
    def test_run_eval_suite_rejects_incomplete_repeated_scalar_sibling_set(
        self, tmp_path
    ):
        akn_file = tmp_path / "source.akn"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="regulation-22-1-b">
        <num>(b)</num>
        <intro><p>the following amount of earned income—</p></intro>
        <level eId="regulation-22-1-b-i">
          <num>(i)</num>
          <content><p>in a case where no work allowance is specified, 55% of that earned income;</p></content>
        </level>
        <level eId="regulation-22-1-b-ii">
          <num>(ii)</num>
          <content><p>in any other case, 55% of the amount above the work allowance.</p></content>
        </level>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )
        clml_file = tmp_path / "source.xml"
        clml_file.write_text("<Legislation/>")
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK taper
runners:
  - claude:opus
cases:
  - kind: uk_legislation
    name: taper-with-work-allowance
    source_ref: /uksi/2013/376/2025-04-07
    section_eid: regulation-22-1-b-ii
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)

        with patch(
            "autorac.harness.evals._fetch_legislation_gov_uk_document",
            return_value=FetchedLegislationGovUkDocument(
                source_id="uksi/2013/376/2025-04-07",
                content_url="https://www.legislation.gov.uk/uksi/2013/376/2025-04-07",
                akn_file=akn_file,
                clml_file=clml_file,
            ),
        ):
            with pytest.raises(ValueError, match="repeated-scalar sibling set"):
                run_eval_suite(
                    manifest=manifest,
                    output_root=tmp_path / "out",
                    rac_path=tmp_path / "rac",
                    atlas_path=None,
                )

    def test_run_eval_suite_allows_complete_repeated_scalar_sibling_set(
        self, tmp_path
    ):
        akn_file = tmp_path / "source.akn"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="regulation-22-1-b">
        <num>(b)</num>
        <intro><p>the following amount of earned income—</p></intro>
        <level eId="regulation-22-1-b-i">
          <num>(i)</num>
          <content><p>in a case where no work allowance is specified, 55% of that earned income;</p></content>
        </level>
        <level eId="regulation-22-1-b-ii">
          <num>(ii)</num>
          <content><p>in any other case, 55% of the amount above the work allowance.</p></content>
        </level>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )
        clml_file = tmp_path / "source.xml"
        clml_file.write_text("<Legislation/>")
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK taper
runners:
  - claude:opus
cases:
  - kind: uk_legislation
    name: taper-no-work-allowance
    source_ref: /uksi/2013/376/2025-04-07
    section_eid: regulation-22-1-b-i
  - kind: uk_legislation
    name: taper-with-work-allowance
    source_ref: /uksi/2013/376/2025-04-07
    section_eid: regulation-22-1-b-ii
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        first = _fake_eval_result("claude-opus", "taper-no-work-allowance")
        second = _fake_eval_result("claude-opus", "taper-with-work-allowance")

        with patch(
            "autorac.harness.evals._fetch_legislation_gov_uk_document",
            return_value=FetchedLegislationGovUkDocument(
                source_id="uksi/2013/376/2025-04-07",
                content_url="https://www.legislation.gov.uk/uksi/2013/376/2025-04-07",
                akn_file=akn_file,
                clml_file=clml_file,
            ),
        ), patch(
            "autorac.harness.evals.run_legislation_gov_uk_section_eval",
            side_effect=[[first], [second]],
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert results == [first, second]

    def test_load_eval_suite_manifest_supports_uk_legislation_cases(self, tmp_path):
        manifest_file = tmp_path / "uk-readiness.yaml"
        manifest_file.write_text(
            """
name: UK readiness
runners:
  - codex:gpt-5.4
mode: cold
gates:
  min_cases: 20
  min_compile_pass_rate: 0.9
cases:
  - kind: uk_legislation
    name: child-benefit-enhanced
    source_ref: /uksi/2006/965/regulation/2/2025-04-07
    section_eid: regulation-2-1-a
            """.strip()
        )

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.name == "UK readiness"
        assert manifest.runners == ["codex:gpt-5.4"]
        assert manifest.mode == "cold"
        assert manifest.gates.min_cases == 20
        assert manifest.gates.min_compile_pass_rate == 0.9
        assert len(manifest.cases) == 1
        assert manifest.cases[0].kind == "uk_legislation"
        assert manifest.cases[0].name == "child-benefit-enhanced"
        assert manifest.cases[0].source_ref == "/uksi/2006/965/regulation/2/2025-04-07"
        assert manifest.cases[0].section_eid == "regulation-2-1-a"

    def test_load_eval_suite_manifest_supports_table_row_query(self, tmp_path):
        manifest_file = tmp_path / "uk-expanded.yaml"
        manifest_file.write_text(
            """
name: UK expanded
runners:
  - openai:gpt-5.4
cases:
  - kind: uk_legislation
    name: uc-standard-allowance-single-young
    source_ref: /uksi/2013/376/regulation/36/2025-04-01
    section_eid: regulation-36-3
    table_row_query: single claimant aged under 25
            """.strip()
        )

        manifest = load_eval_suite_manifest(manifest_file)

        assert manifest.cases[0].table_row_query == "single claimant aged under 25"

    def test_load_eval_suite_manifest_supports_policyengine_rac_var_hint(
        self, tmp_path
    ):
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
    policyengine_rac_var_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")

        manifest = load_eval_suite_manifest(manifest_file)

        assert (
            manifest.cases[0].policyengine_rac_var_hint
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_run_eval_suite_dispatches_to_matching_case_runner(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Mixed suite
runners:
  - codex:gpt-5.4
cases:
  - kind: uk_legislation
    name: child-benefit-enhanced
    source_ref: /uksi/2006/965/regulation/2
    section_eid: regulation-2-1-a
  - kind: source
    name: tanf-slice
    source_id: co-tanf-f
    source_file: ./source.txt
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative source text")
        manifest = load_eval_suite_manifest(manifest_file)

        uk_result = _fake_eval_result("codex-gpt-5.4", "child-benefit-enhanced")
        source_result = _fake_eval_result("codex-gpt-5.4", "co-tanf-f")

        with patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ), patch(
            "autorac.harness.evals.run_legislation_gov_uk_section_eval",
            return_value=[uk_result],
        ) as mock_uk, patch(
            "autorac.harness.evals.run_source_eval",
            return_value=[source_result],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert results == [uk_result, source_result]
        mock_uk.assert_called_once()
        mock_source.assert_called_once()

    def test_run_eval_suite_passes_table_row_query_to_uk_runner(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK row suite
runners:
  - openai:gpt-5.4
cases:
  - kind: uk_legislation
    name: uc-standard-allowance-single-young
    source_ref: /uksi/2013/376/regulation/36/2025-04-01
    section_eid: regulation-36-3
    table_row_query: single claimant aged under 25
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        uk_result = _fake_eval_result("openai-gpt-5.4", "uc-standard-allowance-single-young")

        with patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ), patch(
            "autorac.harness.evals.run_legislation_gov_uk_section_eval",
            return_value=[uk_result],
        ) as mock_uk:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert results == [uk_result]
        assert mock_uk.call_args.kwargs["table_row_query"] == "single claimant aged under 25"

    def test_run_eval_suite_passes_shared_fetch_cache_root_to_uk_runner(
        self, tmp_path
    ):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK row suite
runners:
  - openai:gpt-5.4
cases:
  - kind: uk_legislation
    name: child-benefit-enhanced
    source_ref: /uksi/2006/965/regulation/2/2025-04-07
    section_eid: regulation-2-1-a
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        uk_result = _fake_eval_result("openai-gpt-5.4", "child-benefit-enhanced")
        output_root = tmp_path / "out"

        with patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ), patch(
            "autorac.harness.evals.run_legislation_gov_uk_section_eval",
            return_value=[uk_result],
        ) as mock_uk:
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert mock_uk.call_args.kwargs["fetch_cache_root"] == output_root

    def test_run_eval_suite_passes_policyengine_rac_var_hint_to_source_runner(
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
    policyengine_rac_var_hint: uc_standard_allowance_single_claimant_aged_under_25
            """.strip()
        )
        (tmp_path / "source.txt").write_text("authoritative row text")
        manifest = load_eval_suite_manifest(manifest_file)
        source_result = _fake_eval_result("openai-gpt-5.4", "uc-std-allowance-single")

        with patch(
            "autorac.harness.evals.run_source_eval",
            return_value=[source_result],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert results == [source_result]
        assert (
            mock_source.call_args.kwargs["policyengine_rac_var_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_run_eval_suite_records_case_failure_and_continues(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: Mixed suite
runners:
  - openai:gpt-5.4
cases:
  - kind: uk_legislation
    name: child-benefit-enhanced
    source_ref: /uksi/2006/965/regulation/2
    section_eid: regulation-2-1-a
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
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ), patch(
            "autorac.harness.evals.run_legislation_gov_uk_section_eval",
            side_effect=RuntimeError("502 Server Error"),
        ), patch(
            "autorac.harness.evals.run_source_eval",
            return_value=[source_result],
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert len(results) == 2
        failed = results[0]
        assert failed.runner == "openai-gpt-5.4"
        assert failed.success is False
        assert failed.error == "502 Server Error"
        assert failed.metrics is None
        assert results[1] == source_result

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
            "autorac.harness.evals.run_source_eval",
            side_effect=[RuntimeError("stream disconnected"), [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
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
            "autorac.harness.evals.run_source_eval",
            side_effect=[[failed], [source_result]],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
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
            "autorac.harness.evals.run_source_eval",
            return_value=[failed],
        ) as mock_source:
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        assert results == [failed]
        assert mock_source.call_count == 1


class TestReadinessSummary:
    def test_summarize_readiness_applies_suite_gates(self):
        gates = EvalReadinessGates(
            min_cases=3,
            min_success_rate=1.0,
            min_compile_pass_rate=1.0,
            min_ci_pass_rate=1.0,
            min_zero_ungrounded_rate=1.0,
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
                policyengine_pass=False,
                policyengine_score=0.5,
                estimated_cost_usd=0.40,
            ),
            _fake_eval_result(
                "codex-gpt-5.4",
                "case-c",
                compile_pass=True,
                ci_pass=True,
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
        assert summary.policyengine_case_count == 2
        assert summary.policyengine_pass_rate == 0.5
        assert summary.mean_estimated_cost_usd == 0.3
        assert summary.ready is False
        gate_results = {gate.name: gate for gate in summary.gate_results}
        assert gate_results["min_cases"].passed is True
        assert gate_results["min_policyengine_pass_rate"].passed is False
        assert gate_results["max_mean_estimated_cost_usd"].passed is True


class TestRepoAugmentedContext:
    def test_prepare_eval_workspace_allows_arbitrary_identifier_with_explicit_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        context_file = repo_root / "rac-us" / "statute" / "26" / "32" / "b" / "2" / "A.rac"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(F)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="F. Determining Eligibility ... 165 345 518",
            rac_path=rac_root,
            mode="repo-augmented",
            extra_context_paths=[context_file],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_select_context_files_excludes_target(self, tmp_path):
        rac_us_root = tmp_path / "rac-us" / "statute"
        section_dir = rac_us_root / "26" / "24"
        section_dir.mkdir(parents=True)
        (section_dir / "a.rac").write_text("target")
        (section_dir / "b.rac").write_text("sibling b")
        (section_dir / "c.rac").write_text("sibling c")

        selected = select_context_files("26 USC 24(a)", rac_us_root)

        assert section_dir / "a.rac" not in selected
        assert section_dir / "b.rac" in selected
        assert section_dir / "c.rac" in selected

    def test_prepare_eval_workspace_writes_manifest_and_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "b.rac"
        context_file.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["source_file"] == "source.txt"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_hydrate_eval_root_copies_context_into_import_tree(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "c.rac"
        context_file.write_text("status: stub\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)

        assert (eval_root / "26" / "24" / "c.rac").read_text() == "status: stub\n"

    def test_prepare_eval_workspace_expands_transitive_context_imports(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)

        section_root = repo_root / "rac-us" / "statute" / "26" / "24"
        section_root.mkdir(parents=True)
        aggregator = section_root / "24.rac"
        aggregator.write_text(
            "section_24_credit:\n"
            "    imports:\n"
            "        - 26/24/a#ctc_allowance\n"
            "        - 26/24/c#qualifying_child_count\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Money\n"
        )
        selected = section_root / "c.rac"
        selected.write_text(
            "qualifying_child_count:\n"
            "    imports:\n"
            "        - 26/24/c/2#ctc_meets_citizenship_requirement\n"
            "        - 26/152/c#qualifying_child_of_taxpayer\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Integer\n"
        )

        dep_local = section_root / "c" / "2.rac"
        dep_local.parent.mkdir(parents=True)
        dep_local.write_text("status: encoded\n")

        dep_cross_section = repo_root / "rac-us" / "statute" / "26" / "152" / "c.rac"
        dep_cross_section.parent.mkdir(parents=True)
        dep_cross_section.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[aggregator, selected],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
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
        assert str(section_root / "a.rac") not in copied_sources

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert (eval_root / "26" / "24" / "c" / "2.rac").read_text() == "status: encoded\n"
        assert (eval_root / "26" / "152" / "c.rac").read_text() == "status: encoded\n"

    def test_prompt_includes_scaffold_dates_from_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "b.rac"
        context_file.write_text(
            "status: encoded\n\n"
            "threshold:\n"
            "    from 1998-01-01: 1000\n"
            "    from 2018-01-01: 2000\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        prompt = _build_eval_prompt(
            "26 USC 24(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            "a.rac",
        )

        assert "`1998-01-01`" in prompt
        assert "`2018-01-01`" in prompt
        assert "Prefer the earliest scaffold date" in prompt


class TestUnexpectedAccessDetection:
    def test_flags_parent_directory_traversal(self, tmp_path):
        assert _command_looks_out_of_bounds("bash -lc 'find .. -name *.rac'", tmp_path)

    def test_allows_workspace_paths(self, tmp_path):
        local = tmp_path / "context" / "b.rac"
        local.parent.mkdir(parents=True)
        local.write_text("status: encoded\n")


class TestSourceEval:
    def test_run_source_eval_uses_explicit_context_without_statute_lookup(self, tmp_path):
        rac_root = tmp_path / "rac"
        rac_root.mkdir()
        context_file = tmp_path / "examples" / "piecewise.rac"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("status: encoded\n")

        with patch(
            "autorac.harness.evals._run_prompt_eval",
        ) as mock_prompt_eval, patch(
            "autorac.harness.evals.evaluate_artifact",
        ) as mock_evaluate_artifact:
            mock_prompt_eval.return_value.text = (
                "=== FILE: 9-CCR-2503-6-3.606.1-F.rac ===\n"
                '"""\nF. Determining Eligibility ...\n"""\n'
                "status: encoded\n"
                "grant_standard:\n"
                "    entity: TaxUnit\n"
                "    period: Month\n"
                "    dtype: Money\n"
                "    from 2024-07-01: 165\n"
                "=== FILE: 9-CCR-2503-6-3.606.1-F.rac.test ===\n"
                "grant_standard:\n"
                '  - name: "base case"\n'
                "    period: 2024-07\n"
                "    inputs: {}\n"
                "    expect: 165\n"
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
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[context_file],
            )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert Path(result.output_file).exists()
        assert Path(result.output_file).with_suffix(".rac.test").exists()
        assert result.retrieved_files == [str(context_file)]

        prompt = mock_prompt_eval.call_args.args[2]
        assert ".rac.test" in prompt
        assert "=== FILE:" in prompt

    def test_run_source_eval_passes_oracle_settings_to_evaluate_artifact(self, tmp_path):
        rac_root = tmp_path / "rac"
        rac_root.mkdir()

        with patch(
            "autorac.harness.evals._run_prompt_eval",
        ) as mock_prompt_eval, patch(
            "autorac.harness.evals.evaluate_artifact",
        ) as mock_evaluate_artifact:
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2006-965-regulation-2.rac ===\n"
                '"""\nhttps://www.legislation.gov.uk/uksi/2006/965/regulation/2\n"""\n'
                "status: encoded\n"
                "=== FILE: uksi-2006-965-regulation-2.rac.test ===\n"
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
                rac_path=rac_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert mock_evaluate_artifact.call_args.kwargs["oracle"] == "policyengine"
        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_country"] == "uk"
        )

    def test_run_source_eval_passes_policyengine_rac_var_hint_to_evaluate_artifact(
        self, tmp_path
    ):
        rac_root = tmp_path / "rac"
        rac_root.mkdir()

        with patch(
            "autorac.harness.evals._run_prompt_eval",
        ) as mock_prompt_eval, patch(
            "autorac.harness.evals.evaluate_artifact",
        ) as mock_evaluate_artifact:
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.rac ===\n"
                '"""\n317.82\n"""\n'
                "status: encoded\n"
                "=== FILE: uksi-2013-376-regulation-36-3-single-under-25.rac.test ===\n"
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
                rac_path=rac_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
                policyengine_rac_var_hint="uc_standard_allowance_single_claimant_aged_under_25",
            )

        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_rac_var_hint"]
            == "uc_standard_allowance_single_claimant_aged_under_25"
        )

    def test_build_eval_prompt_includes_policyengine_rac_var_hint(self, tmp_path):
        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="uksi/2013/376/regulation/36/3",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="317.82",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2013/376/regulation/36/3",
            "cold",
            workspace,
            [],
            target_file_name="example.rac",
            include_tests=True,
            runner_backend="openai",
            policyengine_rac_var_hint="uc_standard_allowance_single_claimant_aged_under_25",
        )

        assert "uc_standard_allowance_single_claimant_aged_under_25" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/2005/schedule/2",
            "cold",
            workspace,
            [],
            target_file_name="example.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For a single fixed-amount source slice, a base case is sufficient." in prompt
        assert (
            "For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test."
            in prompt
        )
        assert (
            "Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date."
            in prompt
        )
        assert "Do not add speculative future-period tests" in prompt

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/26/24/b.rac'"
        assert not _command_looks_out_of_bounds(command, tmp_path)


def _fake_eval_result(
    runner: str,
    citation: str,
    *,
    compile_pass: bool = True,
    ci_pass: bool = True,
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
        output_file=f"/tmp/{citation}.rac",
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
            policyengine_pass=policyengine_pass,
            policyengine_score=policyengine_score,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        ),
    )
