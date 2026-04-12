"""Tests for model comparison eval helpers."""

import json
import subprocess
from datetime import date
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
    _is_single_amount_table_slice,
    _materialize_eval_artifact,
    _normalize_legislation_gov_uk_source_ref,
    _normalize_nonannual_test_period_value,
    _post_openai_eval_request,
    _resolve_akn_section_eid,
    _resolve_legislation_gov_uk_fetch_cache_root,
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

    def test_accepts_pence_threshold_grounded_as_decimal_gbp(self, tmp_path):
        rac_file = tmp_path / "example.rac"
        rac_file.write_text(
            '''"""
13. Small amounts of state pension credit

Where the amount of state pension credit payable is less than 10 pence per week,
the credit shall not be payable unless the claimant is in receipt of another benefit
payable with the credit.
"""

small_amount_threshold:
    entity: Person
    period: Week
    dtype: Money
    unit: GBP
    from 2025-03-21:
        0.10

amount_payable:
    entity: Person
    period: Week
    dtype: Money
    unit: GBP

is_payable:
    entity: Person
    period: Week
    dtype: Boolean
    from 2025-03-21:
        amount_payable >= small_amount_threshold
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
                    "Where the amount of state pension credit payable is less than "
                    "10 pence per week, the credit shall not be payable."
                ),
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.ungrounded_numeric_count == 0
        assert metrics.missing_source_numeric_occurrence_count == 0

    def test_runs_generalist_reviewer_and_records_result(self, tmp_path):
        rac_file = tmp_path / "example.rac"
        rac_file.write_text(
            '''"""
Provision text with £10.
"""
status: encoded

example_amount:
    entity: Person
    period: Year
    dtype: Money
    from 2025-01-01:
        10
'''
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
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
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
        assert "stale, generic, or misleading" in mock_reviewer.call_args.kwargs["review_context"]

    def test_timing_clause_review_context_mentions_boolean_day_predicate(self, tmp_path):
        rac_file = tmp_path / "example.rac"
        rac_file.write_text(
            '''"""
On the first day of the next benefit week.
"""
status: encoded

example_timing_rule:
    entity: Person
    period: Day
    dtype: Boolean
    from 2025-01-01:
        true
'''
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
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "less than one month apart" in prompt
        assert "one_month_threshold = 1" in prompt
        assert "the `one month` comparator is not a standalone numeric scalar" in prompt
        assert "do not invent `1`-valued threshold/count helpers" in prompt
        assert "branch-specific output is a `Count` or other non-Boolean basis selector" in prompt
        assert "do not write an inline conditional without `else`" in prompt
        assert "negative tests should usually assert only the `_applies` boolean" in prompt
        assert "expect the principal basis-count output to remain the active legal basis" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "keep one canonical fact or classification for that single period" in prompt
        assert "parallel free inputs like `*_in_weeks` and `*_in_months`" in prompt
        assert "do not require a second independent duration input" in prompt
        assert "do not feed the same legal period through contradictory units or categories" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not leave that money output unconditional" in prompt
        assert "typically with an explicit `else: 0`" in prompt
        assert "paragraph-level exceptions or a different payment period displace the limb" in prompt

    def test_build_eval_prompt_for_subject_to_includes_leaf_discourages_blanket_negation(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/17A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text=(
                "Subject to paragraphs (3), (4) and (4A), \"earnings\" in the case "
                "of employment as an employed earner, means any remuneration or "
                "profit derived from that employment and includes—\n\n"
                "(a)\n\n"
                "any bonus or commission;"
            ),
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Subject to paragraphs (3), (4) and (4A), ... includes—" in prompt
        assert "blanket negating gate" in prompt
        assert "Do not make a composite `subject_to_*_satisfied`" in prompt
        assert "branch-specific fact gate" in prompt
        assert "permits this branch to count" in prompt
        assert "do not collapse all cited qualifications into one opaque helper" in prompt
        assert "one paragraph-specific qualification input or import per cited paragraph" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Except where paragraph (2) and (4) apply" in prompt
        assert "do not assume the exception is displaced only when both cited paragraphs apply simultaneously" in prompt
        assert "treat the slice as inoperative when any cited paragraph applies" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/10",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-10.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "distinguish mere placement context from binding lead-in conjuncts" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13B",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13B.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat `X` and `Y` as a positive conjunction for this branch" in prompt
        assert "Do not rewrite that as material implication like `not X or Y`" in prompt
        assert "if the branch-triggering condition itself is false, the branch-specific output should usually be `false`" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies" in prompt
        assert "do not narrow the first `where ...` clause to only the later-mentioned category" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/5/a",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-5-a.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve the consideration-for-use/right-to-use qualifier across both `royalties` and `other sums`" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/5",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-5.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve the shared qualifying employment/office across each alternative limb" in prompt
        assert "do not decompose the rule into one free-standing `person_is_X` fact plus separate `under_A` and `in_B` facts" in prompt
        assert "distribute the shared qualifier across the alternatives with branch-specific combined facts" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/15",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-15.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat `for each £500` as counting complete bands, not proportional fractions" in prompt
        assert "derive the band count with `floor(excess / band_size)`" in prompt
        assert "include a non-exact-multiple excess case like `£750` above threshold" in prompt


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

    def test_materialize_eval_artifact_normalizes_direct_scalar_expression(
        self, tmp_path
    ):
        response = (
            "=== FILE: example.rac ===\n"
            '"""\nWhere the amount payable is less than 10 pence per week.\n"""\n\n'
            "small_amount_threshold:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21:\n"
            "        1 / 10\n"
            "=== FILE: example.rac.test ===\n"
            "- name: base\n"
            "  period: 2025-03-21\n"
            "  output:\n"
            "    small_amount_threshold: 0.1\n"
        )
        expected = tmp_path / "source" / "example.rac"

        wrote = _materialize_eval_artifact(
            response,
            expected,
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "Where the amount payable is less than 10 pence per week.\n"
            ),
        )

        assert wrote is True
        assert "1 / 10" not in expected.read_text()
        assert "from 2025-03-21: 0.1" in expected.read_text()

    def test_materialize_eval_artifact_normalizes_test_arithmetic_literals(
        self, tmp_path
    ):
        response = (
            "=== FILE: example.rac ===\n"
            '"""\nDeduction equals allowable expenses over £35.\n"""\n\n'
            "allowable_expenses:\n"
            "    entity: Person\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "threshold:\n"
            "    entity: Person\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21: 35\n"
            "deduction:\n"
            "    entity: Person\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21: max(0, allowable_expenses - threshold)\n"
            "=== FILE: example.rac.test ===\n"
            "- name: positive\n"
            "  period: 2025-03-21\n"
            "  input:\n"
            "    allowable_expenses: 35 + 3\n"
            "  output:\n"
            "    deduction: 1 + 2\n"
        )
        expected = tmp_path / "source" / "example.rac"

        wrote = _materialize_eval_artifact(
            response,
            expected,
            source_text="Deduction equals allowable expenses over £35.\n",
        )

        assert wrote is True
        test_text = expected.with_suffix(".rac.test").read_text()
        assert "35 + 3" not in test_text
        assert "1 + 2" not in test_text
        assert "allowable_expenses: 38" in test_text
        assert "deduction: 3" in test_text

    def test_materialize_eval_artifact_does_not_crash_for_conditional_money_leaf(
        self, tmp_path
    ):
        response = (
            "=== FILE: example.rac ===\n"
            '"""\n£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance.\n"""\n\n'
            "claimant_receives_benefit:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Boolean\n\n"
            "earnings_disregard_amount:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21: 20\n\n"
            "sum_disregarded:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "    unit: GBP\n"
            "    from 2025-03-21:\n"
            "        if claimant_receives_benefit: earnings_disregard_amount else: 0\n"
            "=== FILE: example.rac.test ===\n"
            "- name: base\n"
            "  period: 2025-03-21\n"
            "  input:\n"
            "    claimant_receives_benefit: true\n"
            "  output:\n"
            "    sum_disregarded: 20\n"
        )
        expected = tmp_path / "source" / "example.rac"

        wrote = _materialize_eval_artifact(
            response,
            expected,
            source_text=(
                "Editorial note: current text valid from 2025-03-21.\n\n"
                "£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance.\n"
            ),
        )

        assert wrote is True
        assert "if claimant_receives_benefit" in expected.read_text()

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

    def test_materialize_eval_artifact_normalizes_bundled_day_periods_with_dotted_paths(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "9-CCR-2503-6-3.606.1-K.rac"
        llm_response = (
            "=== FILE: ./9-CCR-2503-6-3.606.1-K.rac ===\n"
            "authorized_basic_cash_assistance_grant_amount:\n"
            "    entity: TanfUnit\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    from 2026-04-02: 0\n"
            "=== FILE: ./9-CCR-2503-6-3.606.1-K.rac.test ===\n"
            "- name: base_case\n"
            "  period: 2026-04-01\n"
            "  output:\n"
            "    authorized_basic_cash_assistance_grant_amount: 0\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "period: '2026-04'" in test_text or "period: 2026-04" in test_text

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
        assert "period: '2021-04-06'" in test_text or "period: 2021-04-06" in test_text

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
        assert "period: '2021-04-06'" in test_text or "period: 2021-04-06" in test_text

    def test_materialize_eval_artifact_normalizes_year_only_annual_test_periods(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-9-a.rac"
        source_text = (
            "Editorial note: current text valid from 2025-03-21.\n\n"
            "working tax credit;"
        )
        llm_response = (
            "=== FILE: uksi-2002-1792-regulation-9-a.rac ===\n"
            "working_tax_credit:\n"
            "    entity: Person\n"
            "    period: Year\n"
            "    dtype: Money\n\n"
            "qualifying_income_excluded_9_a:\n"
            "    entity: Person\n"
            "    period: Year\n"
            "    dtype: Money\n"
            "    from 2025-03-21:\n"
            "        working_tax_credit\n"
            "=== FILE: uksi-2002-1792-regulation-9-a.rac.test ===\n"
            "- name: base_case\n"
            "  period: 2025\n"
            "  input:\n"
            "    working_tax_credit: 1\n"
            "  output:\n"
            "    qualifying_income_excluded_9_a: 1\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "period: '2025-03-21'" in test_text or "period: 2025-03-21" in test_text

    def test_materialize_eval_artifact_normalizes_mapping_style_tests_to_list(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-10-5-b-ii.rac"
        source_text = (
            "Editorial note: current text valid from 2025-03-21.\n\n"
            "where head (i) does not apply, the first day of the next benefit week "
            "following that increased payment date."
        )
        llm_response = (
            "=== FILE: uksi-2002-1792-regulation-10-5-b-ii.rac ===\n"
            "day_referred_to_10_5_b_ii:\n"
            "    entity: Person\n"
            "    period: Day\n"
            "    dtype: Boolean\n"
            "    from 2025-03-21:\n"
            "        some_fact\n"
            "=== FILE: uksi-2002-1792-regulation-10-5-b-ii.rac.test ===\n"
            "case_branch_ii_applies:\n"
            "  period: 2025-03-21\n"
            "  input:\n"
            "    some_fact: true\n"
            "  output:\n"
            "    day_referred_to_10_5_b_ii: true\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert test_text.lstrip().startswith("- ")
        assert "name: case_branch_ii_applies" in test_text
        assert "case_branch_ii_applies:" not in test_text

    def test_materialize_eval_artifact_rewrites_source_text_wrapper_to_docstring(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-10-6-a.rac"
        llm_response = (
            "=== FILE: uksi-2002-1792-regulation-10-6-a.rac ===\n"
            "source_text:\n"
            "    entity: Person\n"
            "    period: Day\n"
            "    dtype: String\n"
            "    from 2025-03-21:\n"
            "        \"\"\"\n"
            "        Example source line.\n"
            "        \"\"\"\n\n"
            "example_fact:\n"
            "    entity: Person\n"
            "    period: Day\n"
            "    dtype: Boolean\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file, source_text=None)

        assert wrote is True
        rac_text = output_file.read_text()
        assert rac_text.startswith('"""\nExample source line.\n"""')
        assert "source_text:" not in rac_text

    def test_materialize_eval_artifact_fills_missing_period_and_flattens_wrappers(
        self, tmp_path
    ):
        output_file = tmp_path / "source" / "uksi-2002-1792-regulation-9-e.rac"
        source_text = (
            "Editorial note: current text valid from 2025-03-21.\n\n"
            "maternity allowance;"
        )
        llm_response = (
            "=== FILE: uksi-2002-1792-regulation-9-e.rac ===\n"
            "maternity_allowance_amount:\n"
            "    entity: Person\n"
            "    period: Month\n"
            "    dtype: Money\n\n"
            "qualifying_income_exclusion_9_e_maternity_allowance:\n"
            "    entity: Person\n"
            "    period: Month\n"
            "    dtype: Money\n"
            "    from 2025-03-21:\n"
            "        maternity_allowance_amount\n"
            "=== FILE: uksi-2002-1792-regulation-9-e.rac.test ===\n"
            "- name: base_case\n"
            "  input:\n"
            "    maternity_allowance_amount:\n"
            "      entity: Person\n"
            "      period: Month\n"
            "      dtype: Money\n"
            "      values:\n"
            "        2025-03: 3\n"
            "  output:\n"
            "    qualifying_income_exclusion_9_e_maternity_allowance:\n"
            "      2025-03: 3\n"
        )

        wrote = _materialize_eval_artifact(
            llm_response,
            output_file,
            source_text=source_text,
        )

        assert wrote is True
        test_text = output_file.with_suffix(".rac.test").read_text()
        assert "period: '2025-03-21'" in test_text or "period: 2025-03-21" in test_text
        assert "entity: Person" not in test_text
        assert "values:" not in test_text
        assert "qualifying_income_exclusion_9_e_maternity_allowance: 3" in test_text

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

    def test_resolve_legislation_gov_uk_fetch_cache_root_prefers_override(self, tmp_path):
        override = tmp_path / "shared-cache"
        with patch.dict(
            "os.environ",
            {"AUTORAC_SHARED_LEGISLATION_CACHE": str(override)},
            clear=False,
        ):
            resolved = _resolve_legislation_gov_uk_fetch_cache_root(tmp_path / "run-root")

        assert resolved == override.resolve()

    def test_run_legislation_gov_uk_section_eval_uses_shared_cache_root_by_default(
        self, tmp_path
    ):
        shared_root = tmp_path / "shared-cache"
        fetched = FetchedLegislationGovUkDocument(
            source_id="uksi/2002/1792",
            content_url="https://www.legislation.gov.uk/uksi/2002/1792",
            akn_file=tmp_path / "source.akn",
            clml_file=tmp_path / "source.xml",
        )
        fetched.akn_file.write_text("<akomaNtoso/>")
        fetched.clml_file.write_text("<Legislation/>")

        with patch(
            "autorac.harness.evals._resolve_legislation_gov_uk_fetch_cache_root",
            return_value=shared_root,
        ), patch(
            "autorac.harness.evals._fetch_legislation_gov_uk_document",
            return_value=fetched,
        ) as mock_fetch, patch(
            "autorac.harness.evals.run_akn_section_eval",
            return_value=[],
        ) as mock_run:
            results = run_legislation_gov_uk_section_eval(
                source_ref="https://www.legislation.gov.uk/uksi/2002/1792",
                section_eid="regulation-1",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
            )

        assert results == []
        assert mock_fetch.call_args.kwargs["fetch_cache_root"] == shared_root
        mock_run.assert_called_once()

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
        assert mock_run.call_args.kwargs["oracle"] == "none"
        assert mock_run.call_args.kwargs["policyengine_country"] == "auto"


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
        assert "leave `.rac.test` empty" in prompt
        assert "assertions against deferred symbols" in prompt
        assert "emit only a top-level `status: deferred`" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/10/c",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-10-c.rac",
            include_tests=True,
        )

        assert "omitted/repealed text shown only by ellipses" in prompt
        assert "keep the embedded source/docstring showing that omission" in prompt
        assert "editorially omitted or repealed text shown by ellipses or dotted placeholders" in prompt
        assert "leave `.rac.test` empty" in prompt

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
        assert "model truncation toward zero rather than toward negative infinity" in prompt
        assert "rounded to the nearest whole dollar" in prompt
        assert "Model explicit half-up rounding instead" in prompt
        assert "keep that rounding on the downstream output" in prompt
        assert "Do not push it upstream into an intermediate component" in prompt
        assert "Wrong for a clause like `allotment equals the thrifty food plan reduced by 30 per centum of income" in prompt
        assert "keep `snap_expected_contribution = snap_net_income * 0.3`" in prompt
        assert "Do not fabricate an `imports:` target from a cited section unless that exact `path#symbol` import target is listed" in prompt
        assert "instead of guessing a broken import like `statute/...#symbol`" in prompt
        assert "compute the pre-rounding amount exactly" in prompt
        assert "`251 * 0.08 = 20.08`, which still rounds to `20`, not `21`" in prompt
        assert "floor(amount + 0.5)" in prompt
        assert "if amount >= 0: floor(amount) else: ceil(amount)" in prompt
        assert "Reserve bare `floor(...)` for instructions that explicitly say `round down`" in prompt
        assert "unsupported operators such as `%`" in prompt
        assert "include a `.rac.test` case with a negative fractional amount" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13.rac",
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/schedule/VI/paragraph/4/1/a/iva",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-schedule-vi-paragraph-4-1-a-iva.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "positive conditional leaves" in prompt
        assert "the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`" in prompt
        assert "do not use an unconditional amount or `else: true`" in prompt
        assert "fixed supplement, allowance, or addition is payable only while an eligibility condition holds" in prompt
        assert "do not leave that money output unconditional" in prompt
        assert "do not collapse those facts into an opaque local input like `*_eligible_for_*`" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not invent sibling outcomes for non-applicable cases with `else: 0`" in prompt
        assert "leave other cases to sibling limbs" in prompt
        assert "keep a branch-specific money or rate output for that basis" in prompt
        assert "do not invent sibling outcomes for inapplicable cases with `else: 0`" in prompt
        assert "pair the branch-specific money or rate output with a separate applicability boolean" in prompt
        assert "omit assertions about the branch-specific money or rate output" in prompt
        assert "qualifies its averaging basis with operative parenthetical text" in prompt
        assert "includes periods in which the claimant does no work but disregards other absences" in prompt
        assert "generic `average_weekly_income_*` input" in prompt
        assert "`such other payments as may ... enable the claimant's average weekly income to be determined more accurately`" in prompt
        assert "do not leave the branch money output unconditionally equal to the input average" in prompt
        assert "do not reuse the parent provision's generic final-amount phrase" in prompt
        assert "name the principal money or rate output after this limb's own basis or method" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17/9A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17-9A.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "purpose-limited deeming clauses" in prompt
        assert "do not use `status: entity_not_supported`" in prompt
        assert "paragraph-(5) amounts treated as earnings for paragraph-(9)(b) only" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "For residual sibling limbs phrased like `in any other case`" in prompt
        assert "Do not treat the shared parent triggers alone as sufficient" in prompt
        assert "model a local residual-case fact or applicability helper" in prompt
        assert "no more specific sibling case applies" in prompt
        assert "include a case where the parent conditions hold but the residual `other case` condition is false" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not introduce a `*_fact` input" in prompt
        assert "do not use vacuous `else: true`" in prompt
        assert "do not replace the amount-level legal effect with a `Person`/`Day` boolean stand-in" in prompt
        assert "prefer `status: entity_not_supported` over a pseudo-boolean approximation" in prompt
        assert "If the current ontology cannot faithfully tie the deeming effect to the same payment amount" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/f/i",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-f-i.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "If an expenses limb says the expenses are `incurred by the claimant`" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "preserve a single legally operative reference day" in prompt
        assert "model one canonical operative claim-date fact" in prompt
        assert "do not encode separate `day_is_date_claim_was_made` and `day_is_date_claim_was_treated_as_made` facts and then combine them with `or`" in prompt
        assert "include one no-supersession case for the operative claim date and one supersession case for the supersession date" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "treat the cited provision as a possible override or displacement" in prompt
        assert "model a local override/displacement boolean" in prompt
        assert "Do not encode those `Subject to ...` qualifiers as helper names like `*_permits_*`" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17A/2/e",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17A-2-e.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "When canonical imports for those cited paragraphs are available in the workspace, import them." in prompt
        assert "paragraph-specific local inputs are acceptable for an isolated slice artifact" in prompt
        assert "preserve the cited paragraph numbers and the branch-specific legal effect" in prompt

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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/17",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-17.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "do not replace the cited computation with local boolean `*_route_is_satisfied` or `*_fact` placeholders" in prompt
        assert "do not emit a top-level `status: deferred` stub" in prompt
        assert "do not collapse those cited alternatives into one generic treatment gate" in prompt
        assert "preserve the distinct cited alternatives with paragraph-specific imports or local facts/amounts" in prompt
        assert "do not invent an extra `no treatment applies` branch" in prompt
        assert "do not make the cited route-selection flags part of whether the paragraph itself applies" in prompt
        assert "do not encode the consequence as an unqualified `if paragraph_4_route: paragraph_4_amount else: paragraph_10_amount`" in prompt
        assert "Paragraph (10) must be selected by a paragraph-(10) route fact/import or by a derived paragraph-(10) route helper" in prompt
        assert "prefer a single mutually exclusive route selector" in prompt
        assert "Do not expose two independent route booleans that allow both routes or neither route to be selected" in prompt
        assert "a safe local-placeholder shape is" in prompt
        assert "paragraph-(10) route is derived as the applicable paragraph with not paragraph-(4) route" in prompt
        assert "Do not create an invalid-route output branch that returns `0`" in prompt
        assert "self-employed earnings trigger the paragraph" in prompt
        assert "regulation 13 paragraph (4) or paragraph (10) chooses the accounting route" in prompt
        assert "avoid a false case that makes a self-employed-earner branch fail merely because neither local route flag was selected" in prompt
        assert "include separate cases for the distinct cited alternatives" in prompt

    def test_build_eval_prompt_requires_calendar_date_test_periods(
        self, tmp_path
    ):
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/13A/3/b",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-13A-3-b.rac",
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
        assert "must be a YAML list of cases beginning with `- name:`" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert "must contain factual predicates or quantities, not the output variable" in prompt
        assert "use concrete numeric literals in inputs and outputs" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/4A",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2002-1792-regulation-4A.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Where X, Y must ..." in prompt
        assert "Include a `.rac.test` case where `X` is false" in prompt

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
        assert "reuse that named scalar everywhere the rule compares against or computes with that number" in prompt
        assert 'If `./source.txt` says someone is "aged 18 or over", "under 25"' in prompt
        assert "Do not create scalar variables for citation numbers" in prompt
        assert "Do not invent `dtype: String` variables just to restate the effective date" in prompt
        assert "Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables" in prompt
        assert "1st September following the person's 19th birthday" in prompt
        assert "Use `==` for equality comparisons inside RAC expressions" in prompt

    def test_prepare_eval_workspace_injects_resolved_defined_term_stub(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        definition_files = [
            item for item in workspace.context_files if item.kind == "definition_stub"
        ]
        assert len(definition_files) == 1
        assert (
            definition_files[0].workspace_path
            == "context/legislation/ukpga/2002/16/section/3ZA/3.rac"
        )
        assert (
            definition_files[0].import_path
            == "legislation/ukpga/2002/16/section/3ZA/3"
        )
        stub_path = workspace.root / definition_files[0].workspace_path
        assert stub_path.exists()
        assert "is_member_of_mixed_age_couple" in stub_path.read_text()

    def test_prepare_eval_workspace_copies_resolved_canonical_concept_file(
        self, tmp_path
    ):
        rac_root = tmp_path / "rac-us-co"
        concept_file = rac_root / "statute" / "crs" / "26-2-703" / "12.rac"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            '''
"""
C.R.S. § 26-2-703(12)
Definitions

"Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
"""

is_individual_responsibility_contract:
    entity: Person
    period: Month
    dtype: Boolean
'''
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            rac_path=rac_root,
            mode="cold",
            extra_context_paths=[],
        )

        concept_files = [
            item for item in workspace.context_files if item.kind == "canonical_concept"
        ]
        assert len(concept_files) == 1
        assert concept_files[0].workspace_path == "context/statute/crs/26-2-703/12.rac"
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        runner_root = tmp_path / "case" / "openai-gpt-5.4"
        source_dir = runner_root / "source"
        source_dir.mkdir(parents=True)
        (source_dir / "example.rac").write_text("status: encoded\n")

        _hydrate_eval_root(runner_root, workspace)

        hydrated = runner_root / "legislation" / "ukpga" / "2002" / "16" / "section" / "3ZA" / "3.rac"
        assert hydrated.exists()
        assert "is_member_of_mixed_age_couple" in hydrated.read_text()

    def test_build_eval_prompt_includes_resolved_defined_term_guidance(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2002/1792/regulation/7A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="A person who is a member of a mixed-age couple is not entitled.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2002/1792/regulation/7A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Resolved definition files are available below." in prompt
        assert "mixed-age couple" in prompt
        assert "legislation/ukpga/2002/16/section/3ZA/3#is_member_of_mixed_age_couple" in prompt
        assert "import that canonical definition instead of inventing a leaf-local helper" in prompt
        assert "Exact RAC import syntax for a resolved definition:" in prompt
        assert "imports:" in prompt
        assert "Do not replace that import with a local deferred stub" in prompt
        assert "Do not encode such local factual predicates as placeholder constants like `true` or `false`." in prompt
        assert "Do not encode such local factual predicates as `status: deferred`" in prompt
        assert "preserve that post-adjustment quantity directly as an input/helper" in prompt
        assert "prefer an input like `countable_gross_earned_income_after_disregards` over raw `gross_earned_income`" in prompt
        assert "after all other applicable deductions have been allowed" in prompt
        assert "do not truncate the logic to only the deduction categories exercised by the example tests" in prompt
        assert "import those exact symbols instead of inventing renamed locals that overlap with the copied file" in prompt
        assert "import that helper and alias the requested target to it instead of rebuilding the same arithmetic locally" in prompt
        assert "alias `snap_net_income_pre_shelter` to `snap_income_after_non_shelter_deductions`" in prompt
        assert "preserve that helper's nearest input surface in tests" in prompt
        assert "prefer test inputs like `snap_gross_income` plus the applicable deduction symbols" in prompt
        assert "creating near-duplicate locals such as `snap_excess_medical_deduction`" in prompt
        assert "make it a real rate-valued helper" in prompt
        assert "encode `50 percent` as `0.5` and `130 percent` as `1.3`" in prompt
        assert "do not collapse the principal output to an unconditional `true` or `false`" in prompt

    def test_build_eval_prompt_includes_resolved_canonical_concept_guidance(
        self, tmp_path
    ):
        rac_root = tmp_path / "rac-us-co"
        concept_file = rac_root / "statute" / "crs" / "26-2-703" / "12.rac"
        concept_file.parent.mkdir(parents=True, exist_ok=True)
        concept_file.write_text(
            '''
"""
C.R.S. § 26-2-703(12)
Definitions

"Individual responsibility contract" or "IRC" means the contract entered into by the participant and the county department pursuant to section 26-2-708.
"""

is_individual_responsibility_contract:
    entity: Person
    period: Month
    dtype: Boolean
'''
        )

        workspace = prepare_eval_workspace(
            citation="co/regulation/3.609.1/A",
            runner=parse_runner_spec("openai:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="The participant must comply with the individual responsibility contract.",
            rac_path=rac_root,
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "co/regulation/3.609.1/A",
            "cold",
            workspace,
            workspace.context_files,
            target_file_name="example.rac",
            include_tests=True,
            runner_backend="openai",
        )

        assert "Resolved canonical concept files from this corpus are available below." in prompt
        assert "individual responsibility contract" in prompt
        assert "statute/crs/26-2-703/12#is_individual_responsibility_contract" in prompt
        assert "import or re-export that exact canonical concept instead of duplicating it locally" in prompt

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
            include_tests=True,
        )

        assert "emit the upstream import instead of restating the concept locally" in prompt
        assert "says a value is determined `in accordance with section X`" in prompt
        assert "statute/7/2014/e#snap_net_income" in prompt
        assert "snap_household_income_under_2014_d_and_e" in prompt
        assert "author it as an amendment layer targeting those canonical symbols" in prompt
        assert "emit dated `amend` blocks for those canonical symbols" in prompt
        assert "Do not import a canonical output like `snap_one_person_thrifty_food_plan_cost`" in prompt
        assert "Wrong for annual parameter tables" in prompt
        assert "do not create documentary scalar constants like `snap_household_size_four: 4`" in prompt
        assert "Right pattern for the USDA SNAP FY2026 table" in prompt
        assert "amend snap_one_person_thrifty_food_plan_cost:" in prompt
        assert "amend snap_minimum_allotment:" in prompt
        assert "amend snap_maximum_allotment:" in prompt
        assert "after the 15th day of a month" in prompt
        assert "do not decompose it into separate numeric `*_day`" in prompt
        assert "Do not add a documentary scalar like `*_cutoff_day: 15`" in prompt
        assert "include at least one `.rac.test` case that exercises the positive non-zero path" in prompt
        assert "Do not write only zero-output tests for a thresholded deduction" in prompt
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
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(E)",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1-E.rac",
            include_tests=True,
            runner_backend="codex",
        )

        assert "from 0001-01-01:" in prompt
        assert "harness-only fallback" in prompt


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

    def test_run_eval_suite_passes_oracle_settings_to_uk_runner(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            """
name: UK oracle suite
runners:
  - openai:gpt-5.4
cases:
  - kind: uk_legislation
    name: regulation-2-1-a
    source_ref: /uksi/2006/965/regulation/2/2025-04-07
    section_eid: regulation-2-1-a
    oracle: none
    policyengine_country: auto
            """.strip()
        )
        manifest = load_eval_suite_manifest(manifest_file)
        uk_result = _fake_eval_result("openai-gpt-5.4", "regulation-2-1-a")

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
        assert mock_uk.call_args.kwargs["oracle"] == "none"
        assert mock_uk.call_args.kwargs["policyengine_country"] == "auto"

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

    def test_run_eval_suite_persists_run_state_and_case_result_ledger(self, tmp_path):
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
        output_root = tmp_path / "out"

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
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                rac_path=tmp_path / "rac",
                atlas_path=None,
            )

        state = json.loads((output_root / "suite-run.json").read_text())
        assert state["status"] == "completed"
        assert state["completed_cases"] == 2
        assert state["result_count"] == 2
        lines = (output_root / "suite-results.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["case_index"] == 1
        assert first["case_name"] == "child-benefit-enhanced"
        assert first["result"]["success"] is False
        assert second["case_index"] == 2
        assert second["case_name"] == "tanf-slice"
        assert second["result"]["success"] is True
        assert "active_case" not in state

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

        with patch(
            "autorac.harness.evals.run_source_eval",
            side_effect=fake_run_source_eval,
        ), patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ):
            run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                rac_path=tmp_path / "rac",
                atlas_path=None,
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

        with patch(
            "autorac.harness.evals.run_source_eval",
            side_effect=[[usage_limited], [second]],
        ) as mock_source, patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                rac_path=tmp_path / "rac",
                atlas_path=None,
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

        with patch(
            "autorac.harness.evals.run_source_eval",
            side_effect=[[timed_out], [recovered]],
        ) as mock_source, patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                atlas_path=None,
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

        with patch(
            "autorac.harness.evals.run_source_eval",
            return_value=[second],
        ) as mock_source, patch(
            "autorac.harness.evals._validate_uk_shared_scalar_sibling_sets",
            return_value=None,
        ):
            results = run_eval_suite(
                manifest=manifest,
                output_root=output_root,
                rac_path=tmp_path / "rac",
                atlas_path=None,
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

    def test_repo_uk_starter_and_readiness_enable_policyengine_uk(self):
        repo_root = Path(__file__).resolve().parents[1]

        for filename in ("uk_starter.yaml", "uk_readiness.yaml"):
            manifest = load_eval_suite_manifest(repo_root / "benchmarks" / filename)

            assert manifest.gates.min_policyengine_pass_rate is not None
            assert len(manifest.cases) == 11
            assert all(case.oracle == "policyengine" for case in manifest.cases)
            assert all(case.policyengine_country == "uk" for case in manifest.cases)
            assert all(case.policyengine_rac_var_hint for case in manifest.cases)

    def test_repo_uk_policyengine_readiness_is_oracle_only(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "uk_policyengine_readiness.yaml"
        )

        assert len(manifest.cases) == 33
        assert manifest.gates.min_cases == 33
        assert manifest.gates.min_policyengine_pass_rate == 0.85
        assert all(case.oracle == "policyengine" for case in manifest.cases)
        assert all(case.policyengine_country == "uk" for case in manifest.cases)
        assert all(case.policyengine_rac_var_hint for case in manifest.cases)

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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve()
        ]
        assert manifest.cases[4].allow_context == [
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.rac"
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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve()
        ]
        assert manifest.cases[5].allow_context == [
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.rac"
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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve()
        ]
        assert manifest.cases[4].allow_context == [
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.rac"
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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.rac"
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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve()
        ]

    def test_repo_us_co_colorado_works_leaf_closeout_manifest_loads_expected_cases(self):
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
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve()
        ]
        assert manifest.cases[1].allow_context == [
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "F.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us-co"
                / "regulation"
                / "9-CCR-2503-6"
                / "3.606.1"
                / "I.rac"
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
            (repo_root.parent / "rac-us" / "statute" / "7" / "2014" / "e.rac").resolve(),
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.rac"
            ).resolve(),
        ]
        assert manifest.cases[1].allow_context == [
            (repo_root.parent / "rac-us" / "statute" / "7" / "2017" / "a.rac").resolve()
        ]
        assert manifest.cases[2].allow_context == [
            (repo_root.parent / "rac-us" / "statute" / "7" / "2017" / "a.rac").resolve(),
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2017"
                / "c"
                / "1.rac"
            ).resolve(),
        ]
        assert manifest.cases[3].allow_context == [
            (repo_root.parent / "rac-us" / "statute" / "7" / "2017" / "a.rac").resolve()
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
        assert case.source_file == (
            repo_root.parent / "rac-us" / "sources" / "slices" / "7-USC" / "2017" / "c" / "3.txt"
        ).resolve()
        assert case.allow_context == [
            (repo_root.parent / "rac-us" / "statute" / "7" / "2017" / "a.rac").resolve(),
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2017"
                / "c"
                / "1.rac"
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
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "usda"
            / "snap"
            / "fy-2026-cola"
            / "allotment-table.txt"
        ).resolve()
        assert case.allow_context == [
            (repo_root.parent / "rac-us" / "statute" / "7" / "2017" / "a.rac").resolve()
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
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "7-USC"
            / "2014"
            / "g"
            / "1.txt"
        ).resolve()
        assert case.allow_context == [
            (
                repo_root.parent
                / "rac-us"
                / "usda"
                / "snap"
                / "fy-2026-cola"
                / "2.rac"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rac_var_hint == "meets_snap_asset_test"

    def test_repo_us_snap_asset_test_current_effective_refresh_manifest_loads_expected_case(
        self,
    ):
        repo_root = Path(__file__).resolve().parents[1]
        manifest = load_eval_suite_manifest(
            repo_root / "benchmarks" / "us_snap_asset_test_current_effective_refresh.yaml"
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
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "usda"
            / "snap"
            / "fy-2026-cola"
            / "asset-limits-current-effective.txt"
        ).resolve()
        assert case.allow_context == [
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.rac"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rac_var_hint == "meets_snap_asset_test"

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
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "7-USC"
            / "snap"
            / "current-effective"
            / "is_snap_eligible.txt"
        ).resolve()
        assert case.allow_context == [
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "c.rac"
            ).resolve(),
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "g"
                / "1.rac"
            ).resolve(),
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rac_var_hint == "is_snap_eligible"

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
        assert case.source_id == "SNAP earned income deduction under 7 USC 2014(e)(2)(B)"
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "7-USC"
            / "snap"
            / "current-effective"
            / "snap_earned_income_deduction.txt"
        ).resolve()
        assert case.allow_context == [
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "e.rac"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rac_var_hint == "snap_earned_income_deduction"

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
        assert case.source_file == (
            repo_root.parent
            / "rac-us"
            / "sources"
            / "slices"
            / "7-USC"
            / "snap"
            / "current-effective"
            / "snap_net_income_pre_shelter.txt"
        ).resolve()
        assert case.allow_context == [
            (
                repo_root.parent
                / "rac-us"
                / "statute"
                / "7"
                / "2014"
                / "e.rac"
            ).resolve()
        ]
        assert case.oracle == "policyengine"
        assert case.policyengine_country == "auto"
        assert case.policyengine_rac_var_hint == "snap_net_income_pre_shelter"


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
        assert manifest["context_files"][0]["import_path"] == "26/32/b/2/A"
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
        assert manifest["context_files"][0]["import_path"] == "26/24/b"
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_build_eval_prompt_lists_canonical_context_import_target(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        external_file = (
            repo_root / "rac-us-co" / "regulation" / "9-CCR-2503-6" / "3.606.1" / "F.rac"
        )
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text(
            "grant_standard_for_assistance_unit:\n"
            "    entity: TanfUnit\n"
            "    period: Month\n"
            "    dtype: Money\n"
        )

        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(I)",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Deduct the total from step 2, above, from the grant amount.",
            rac_path=rac_root,
            mode="repo-augmented",
            extra_context_paths=[external_file],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1(I)",
            "repo-augmented",
            workspace,
            workspace.context_files,
            target_file_name="9-CCR-2503-6-3.606.1-I.rac",
        )

        assert (
            "inspect `context/regulation/9-CCR-2503-6/3.606.1/F.rac`; "
            "import target `regulation/9-CCR-2503-6/3.606.1/F`"
        ) in prompt
        assert "do not wrap import targets in quotes" in prompt
        assert "use the listed import target rather than the `./context/...` inspection path" in prompt
        assert "do not guess contradictory `.rac.test` expectations for those imported values" in prompt
        assert "keep `.rac.test` inputs and expected outputs consistent with the rows visible in that imported file" in prompt
        assert "Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0`" in prompt
        assert "Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file" in prompt
        assert "Do not use a `0 children / 0 caretakers` household as the primary threshold test" in prompt
        assert "Wrong (`.rac.test` guesses a degenerate chart row):" in prompt
        assert "Right (`.rac.test` uses a visible chart row like one child / no caretaker):" in prompt

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

        assert (eval_root / "statute" / "26" / "24" / "c.rac").read_text() == "status: stub\n"

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

    def test_repo_augmented_context_resolves_statute_prefixed_dependencies(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "7" / "2014"
        rac_us_root.mkdir(parents=True)

        selected = rac_us_root / "e.rac"
        selected.write_text(
            "imports:\n"
            "    - statute/7/2014/2014#snap_household_has_elderly_or_disabled_member\n"
            "    - statute/7/2014/d#snap_gross_income\n"
            "snap_net_income:\n"
            "    entity: Household\n"
            "    period: Month\n"
            "    dtype: Money\n"
        )

        section_file = rac_us_root / "2014.rac"
        section_file.write_text("status: encoded\n")
        cross_file = rac_us_root / "d.rac"
        cross_file.write_text("status: encoded\n")

        runner = parse_runner_spec("openai:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="7 USC 2017(a)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="2017(a) ...",
            rac_path=rac_root,
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
        assert "keep `.rac.test` inputs oracle-comparable" in prompt
        assert (
            "prefer a contemporary monthly `.rac.test` period like `2022-01` or `2024-01`"
            in prompt
        )
        assert (
            "prefer importing those component tests over collapsing them into a single aggregate helper"
            in prompt
        )
        assert "prefer the oracle's direct component facts over inverted household proxy inputs" in prompt
        assert (
            "preserve that as a person-level fact instead of turning it into a whole-household bar"
            in prompt
        )
        assert (
            "prefer the direct component surface `meets_snap_gross_income_test`, `meets_snap_net_income_test`, `meets_snap_asset_test`"
            in prompt
        )
        assert (
            "do not introduce household proxy inputs like `snap_household_has_eligible_participating_member`, renamed variants like `snap_household_has_member_individually_eligible_to_participate`, or count proxies like `snap_number_of_members_eligible_to_participate`"
            in prompt
        )
        assert (
            "import that copied current-effective symbol rather than jumping past it to an older base-statute symbol"
            in prompt
        )
        assert (
            "every import path must point to a file that is actually copied into the workspace"
            in prompt
        )
        assert "assert a copied downstream output named by the oracle hint" in prompt

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

    def test_single_amount_slice_detection_excludes_conditional_money_leaf(self):
        assert (
            _is_single_amount_table_slice(
                "£20 is disregarded if the claimant is in receipt of Scottish adult disability living allowance."
            )
            is False
        )

    def test_normalize_nonannual_test_period_value_converts_iso_week_to_effective_date(self):
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
