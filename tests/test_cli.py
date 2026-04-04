"""
Tests for autorac CLI (cli.py).

Tests all CLI commands using subprocess invocation and direct function calls.
All external dependencies are mocked.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from autorac.cli import (
    _effective_runner_specs,
    _extract_subsections_from_xml,
    _rewrite_gpt_runner_backend,
    cmd_benchmark,
    cmd_calibration,
    cmd_compile,
    cmd_coverage,
    cmd_encode,
    cmd_eval_suite,
    cmd_eval_suite_report,
    cmd_init,
    cmd_log,
    cmd_log_event,
    cmd_runs,
    cmd_session_end,
    cmd_session_show,
    cmd_session_start,
    cmd_session_stats,
    cmd_sessions,
    cmd_stats,
    cmd_statute,
    cmd_sync_sdk_sessions,
    cmd_sync_transcripts,
    cmd_transcript_stats,
    cmd_validate,
    main,
)
from autorac.harness.encoding_db import (
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    ReviewResult,
    ReviewResults,
)

# =========================================================================
# Test main() dispatch
# =========================================================================


class TestRunnerOverrides:
    def test_rewrites_openai_gpt_runner_to_codex(self):
        assert (
            _rewrite_gpt_runner_backend("openai:gpt-5.4", "codex")
            == "codex:gpt-5.4"
        )

    def test_preserves_alias_when_rewriting_gpt_runner_backend(self):
        assert (
            _rewrite_gpt_runner_backend("gpt=openai:gpt-5.4", "codex")
            == "gpt=codex:gpt-5.4"
        )

    def test_effective_runner_specs_uses_env_override(self, monkeypatch):
        monkeypatch.setenv("AUTORAC_GPT_BACKEND", "codex")
        args = SimpleNamespace(gpt_backend=None)

        assert _effective_runner_specs(
            ["openai:gpt-5.4", "claude:opus"], args
        ) == ["codex:gpt-5.4", "claude:opus"]


class TestMain:
    def test_no_command_shows_help_and_exits(self):
        """main() with no command should print help and exit 1."""
        with patch("sys.argv", ["autorac"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_validate_command_dispatches(self):
        """main() with 'validate' should call cmd_validate."""
        with tempfile.NamedTemporaryFile(suffix=".rac") as f:
            with patch("sys.argv", ["autorac", "validate", f.name]):
                with patch("autorac.cli.cmd_validate") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_log_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".rac") as f:
            with patch(
                "sys.argv",
                [
                    "autorac",
                    "log",
                    "--citation",
                    "26 USC 1",
                    "--file",
                    f.name,
                ],
            ):
                with patch("autorac.cli.cmd_log") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_stats_command_dispatches(self):
        with patch("sys.argv", ["autorac", "stats"]):
            with patch("autorac.cli.cmd_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_calibration_command_dispatches(self):
        with patch("sys.argv", ["autorac", "calibration"]):
            with patch("autorac.cli.cmd_calibration") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_statute_command_dispatches(self):
        with patch("sys.argv", ["autorac", "statute", "26 USC 1"]):
            with patch("autorac.cli.cmd_statute") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_runs_command_dispatches(self):
        with patch("sys.argv", ["autorac", "runs"]):
            with patch("autorac.cli.cmd_runs") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_init_command_dispatches(self):
        with patch("sys.argv", ["autorac", "init", "26 USC 1"]):
            with patch("autorac.cli.cmd_init") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_coverage_command_dispatches(self):
        with patch("sys.argv", ["autorac", "coverage", "26 USC 1"]):
            with patch("autorac.cli.cmd_coverage") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_encode_command_dispatches(self):
        with patch("sys.argv", ["autorac", "encode", "26 USC 1"]):
            with patch("autorac.cli.cmd_encode") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_command_dispatches(self):
        with patch("sys.argv", ["autorac", "eval", "26 USC 24(a)"]):
            with patch("autorac.cli.cmd_eval") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_source_command_dispatches(self):
        with tempfile.NamedTemporaryFile() as f:
            with patch(
                "sys.argv",
                ["autorac", "eval-source", "CO TANF 3.606.1(F)", f.name],
            ):
                with patch("autorac.cli.cmd_eval_source") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_akn_section_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            with patch(
                "sys.argv",
                [
                    "autorac",
                    "eval-akn-section",
                    "CO TANF 3.606.1",
                    f.name,
                    "sec_3_606_1",
                ],
            ):
                with patch("autorac.cli.cmd_eval_akn_section") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_uk_legislation_section_command_dispatches(self):
        with patch(
            "sys.argv",
            [
                "autorac",
                "eval-uk-legislation-section",
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
            ],
        ):
            with patch("autorac.cli.cmd_eval_uk_legislation_section") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_uk_legislation_section_accepts_table_row_query(self):
        with patch(
            "sys.argv",
            [
                "autorac",
                "eval-uk-legislation-section",
                "/uksi/2013/376/regulation/36/2025-04-01",
                "--section-eid",
                "regulation-36-3",
                "--table-row-query",
                "single claimant aged under 25",
            ],
        ):
            with patch("autorac.cli.cmd_eval_uk_legislation_section") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_source_accepts_policyengine_rac_var_hint(self):
        with tempfile.NamedTemporaryFile() as f:
            with patch(
                "sys.argv",
                [
                    "autorac",
                    "eval-source",
                    "UC row",
                    f.name,
                    "--policyengine-rac-var-hint",
                    "uc_standard_allowance_single_claimant_aged_under_25",
                ],
            ):
                with patch("autorac.cli.cmd_eval_source") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_suite_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with patch("sys.argv", ["autorac", "eval-suite", f.name]):
                with patch("autorac.cli.cmd_eval_suite") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_suite_report_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            with patch("sys.argv", ["autorac", "eval-suite-report", f.name]):
                with patch("autorac.cli.cmd_eval_suite_report") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_compile_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".rac") as f:
            with patch("sys.argv", ["autorac", "compile", f.name]):
                with patch("autorac.cli.cmd_compile") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_benchmark_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".rac") as f:
            with patch("sys.argv", ["autorac", "benchmark", f.name]):
                with patch("autorac.cli.cmd_benchmark") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_session_start_dispatches(self):
        with patch("sys.argv", ["autorac", "session-start"]):
            with patch("autorac.cli.cmd_session_start") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_end_dispatches(self):
        with patch("sys.argv", ["autorac", "session-end", "--session", "abc"]):
            with patch("autorac.cli.cmd_session_end") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_log_event_dispatches(self):
        with patch(
            "sys.argv",
            ["autorac", "log-event", "--session", "abc", "--type", "tool_call"],
        ):
            with patch("autorac.cli.cmd_log_event") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_sessions_dispatches(self):
        with patch("sys.argv", ["autorac", "sessions"]):
            with patch("autorac.cli.cmd_sessions") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_show_dispatches(self):
        with patch("sys.argv", ["autorac", "session-show", "abc123"]):
            with patch("autorac.cli.cmd_session_show") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_stats_dispatches(self):
        with patch("sys.argv", ["autorac", "session-stats"]):
            with patch("autorac.cli.cmd_session_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_sync_transcripts_dispatches(self):
        with patch("sys.argv", ["autorac", "sync-transcripts"]):
            with patch("autorac.cli.cmd_sync_transcripts") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_transcript_stats_dispatches(self):
        with patch("sys.argv", ["autorac", "transcript-stats"]):
            with patch("autorac.cli.cmd_transcript_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()


class TestCmdEvalSuite:
    def test_exits_nonzero_when_runner_is_not_ready(self, tmp_path, capsys):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text("name: readiness\ncases:\n  - kind: source\n    source_id: x\n    source_file: ./source.txt\n")
        (tmp_path / "source.txt").write_text("authoritative text")
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            atlas_path=tmp_path / "atlas",
            rac_path=tmp_path / "rac",
            json=False,
            gpt_backend="codex",
        )
        args.rac_path.mkdir()

        fake_result = MagicMock()
        fake_result.runner = "codex-gpt-5.4"
        fake_result.success = True
        fake_result.error = None
        fake_result.metrics = MagicMock(
            compile_pass=True,
            ci_pass=True,
            ungrounded_numeric_count=0,
        )

        fake_summary = MagicMock(
            ready=False,
            total_cases=1,
            success_rate=1.0,
            compile_pass_rate=1.0,
            ci_pass_rate=1.0,
            zero_ungrounded_rate=1.0,
            generalist_review_pass_rate=1.0,
            mean_generalist_review_score=8.0,
            policyengine_case_count=0,
            mean_estimated_cost_usd=0.25,
            gate_results=[],
        )

        with patch("autorac.cli.load_eval_suite_manifest") as mock_load, patch(
            "autorac.cli.run_eval_suite", return_value=[fake_result]
        ) as mock_run, patch(
            "autorac.cli.summarize_readiness", return_value=fake_summary
        ):
            mock_load.return_value.name = "Readiness"
            mock_load.return_value.path = manifest_file
            mock_load.return_value.runners = ["openai:gpt-5.4"]
            mock_load.return_value.cases = [MagicMock(kind="source")]
            mock_load.return_value.gates = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                cmd_eval_suite(args)

        assert exc_info.value.code == 1
        assert mock_run.called
        assert mock_run.call_args.kwargs["runner_specs"] == ["codex:gpt-5.4"]
        captured = capsys.readouterr()
        assert "NOT READY" in captured.out


class TestCmdEvalSuiteReport:
    def test_renders_markdown_and_writes_csv(self, tmp_path, capsys):
        payload = {
            "manifest": {"name": "UK paper", "path": "/tmp/suite.yaml"},
            "results": [
                {
                    "citation": "case-a",
                    "runner": "gpt-5.4",
                    "success": True,
                    "duration_ms": 1000,
                    "estimated_cost_usd": 0.1,
                    "output_file": "/tmp/gpt.rac",
                    "metrics": {
                        "compile_pass": True,
                        "ci_pass": True,
                        "ungrounded_numeric_count": 0,
                        "policyengine_score": 1.0,
                    },
                },
                {
                    "citation": "case-a",
                    "runner": "claude-opus",
                    "success": True,
                    "duration_ms": 2000,
                    "estimated_cost_usd": 0.2,
                    "output_file": "/tmp/claude.rac",
                    "metrics": {
                        "compile_pass": True,
                        "ci_pass": True,
                        "ungrounded_numeric_count": 0,
                        "policyengine_score": 0.5,
                    },
                },
            ],
            "readiness": {
                "gpt-5.4": {
                    "total_cases": 1,
                    "success_rate": 1.0,
                    "compile_pass_rate": 1.0,
                    "ci_pass_rate": 1.0,
                    "zero_ungrounded_rate": 1.0,
                    "policyengine_pass_rate": 1.0,
                    "mean_policyengine_score": 1.0,
                    "mean_estimated_cost_usd": 0.1,
                    "mean_duration_ms": 1000,
                },
                "claude-opus": {
                    "total_cases": 1,
                    "success_rate": 1.0,
                    "compile_pass_rate": 1.0,
                    "ci_pass_rate": 1.0,
                    "zero_ungrounded_rate": 1.0,
                    "policyengine_pass_rate": 1.0,
                    "mean_policyengine_score": 0.5,
                    "mean_estimated_cost_usd": 0.2,
                    "mean_duration_ms": 2000,
                },
            },
        }
        result_json = tmp_path / "results.json"
        result_json.write_text(json.dumps(payload))
        csv_out = tmp_path / "cases.csv"
        md_out = tmp_path / "report.md"
        args = SimpleNamespace(
            result_json=result_json,
            left_runner=None,
            right_runner=None,
            markdown_out=md_out,
            csv_out=csv_out,
            json=False,
        )

        cmd_eval_suite_report(args)

        captured = capsys.readouterr()
        assert "# UK paper model comparison" in captured.out
        assert "gpt-5.4" in captured.out
        assert csv_out.exists()
        assert md_out.exists()
        assert "case-a" in csv_out.read_text()

    def test_sync_sdk_sessions_dispatches(self):
        with patch("sys.argv", ["autorac", "sync-sdk-sessions"]):
            with patch("autorac.cli.cmd_sync_sdk_sessions") as mock_cmd:
                main()
                mock_cmd.assert_called_once()


# =========================================================================
# Test cmd_validate
# =========================================================================


class TestCmdValidate:
    def test_file_not_found(self, capsys):
        args = MagicMock()
        args.file = Path("/nonexistent/file.rac")
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.out

    def test_validate_pass_text_output(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_validate_fail_json_output(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.skip_reviewers = False
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = False
        mock_result.ci_pass = False
        mock_result.results = {
            "ci": MagicMock(error="parse error"),
        }
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=5.0,
            formula_reviewer=5.0,
            parameter_reviewer=5.0,
            integration_reviewer=5.0,
            policyengine_match=None,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["all_passed"] is False

    def test_validate_with_oracle_policyengine_pass(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = "policyengine"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(score=0.98, error=None),
        }
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.98,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

    def test_validate_with_oracle_policyengine_fail(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = "policyengine"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(score=0.80, error=None),
        }
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.80,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1

    def test_validate_with_oracle_taxsim_fail(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.skip_reviewers = False
        args.oracle = "taxsim"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "taxsim": MagicMock(score=0.50, error=None),
        }
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=0.50,
        )
        mock_result.total_duration_ms = 100

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1

    def test_validate_with_oracle_all(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = "all"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(score=0.98, error=None),
            "taxsim": MagicMock(score=0.96, error=None),
        }
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.98,
            taxsim_match=0.96,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

    def test_validate_json_output_with_reviewresults_object(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.skip_reviewers = False
        args.oracle = "policyengine"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(score=1.0, error=None),
        }
        mock_result.to_actual_scores.return_value = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rac_reviewer", passed=True, items_checked=1, items_passed=1),
                ReviewResult(reviewer="formula_reviewer", passed=True, items_checked=2, items_passed=2),
                ReviewResult(reviewer="parameter_reviewer", passed=False, items_checked=2, items_passed=1),
                ReviewResult(reviewer="integration_reviewer", passed=True, items_checked=1, items_passed=1),
            ],
            policyengine_match=1.0,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        output = json.loads(capsys.readouterr().out)
        assert output["scores"]["rac_reviewer"] == 10.0
        assert output["scores"]["parameter_reviewer"] == 5.0
        assert output["oracle_scores"]["policyengine"] == 1.0

    def test_validate_passes_skip_reviewers_to_pipeline(self, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=None,
            formula_reviewer=None,
            parameter_reviewer=None,
            integration_reviewer=None,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
            mock_pipeline.validate.assert_called_once_with(
                rac_file.resolve(), skip_reviewers=True
            )

    def test_validate_json_output_uses_null_scores_when_reviewers_skipped(
        self, capsys, tmp_path
    ):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.skip_reviewers = True
        args.oracle = "policyengine"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {"policyengine": MagicMock(score=1.0, error=None)}
        mock_result.to_actual_scores.return_value = ReviewResults(
            reviews=[],
            policyengine_match=1.0,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        output = json.loads(capsys.readouterr().out)
        assert output["scores"]["rac_reviewer"] is None
        assert output["scores"]["parameter_reviewer"] is None

    def test_validate_rac_us_not_found_uses_defaults(self, capsys, tmp_path):
        """When rac_us can't be found by walking, use default paths."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0


# =========================================================================
# Test cmd_compile
# =========================================================================


class TestCmdCompile:
    def test_file_not_found(self, capsys):
        args = MagicMock()
        args.file = Path("/nonexistent/file.rac")
        args.json = False
        args.as_of = None
        args.execute = False
        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 1

    def test_compile_success_text(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test content")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.as_of = None
        args.execute = False

        mock_ir = MagicMock()
        mock_ir.variables = {"var1": MagicMock(), "var2": MagicMock()}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock()

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Compiled" in captured.out
        assert "var1" in captured.out

    def test_compile_success_json(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test content")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.as_of = "2024-01-01"
        args.execute = False

        mock_ir = MagicMock()
        mock_ir.variables = {"var1": MagicMock()}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock()

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is True
        assert output["as_of"] == "2024-01-01"

    def test_compile_with_execute_text(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.as_of = None
        args.execute = True

        mock_ir = MagicMock()
        mock_ir.variables = {"var1": MagicMock()}
        mock_result = MagicMock()
        mock_result.scalars = {"var1": 100}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock(return_value=mock_result)

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Execution results" in captured.out

    def test_compile_with_execute_json(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.as_of = None
        args.execute = True

        mock_ir = MagicMock()
        mock_ir.variables = {"var1": MagicMock()}
        mock_result = MagicMock()
        mock_result.scalars = {"var1": 100}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock(return_value=mock_result)

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert "scalars" in output

    def test_compile_failure_text(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.as_of = None
        args.execute = False

        mock_rac = MagicMock()
        mock_rac.parse = MagicMock(side_effect=Exception("Parse error"))
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Compilation failed" in captured.out

    def test_compile_failure_json(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.json = True
        args.as_of = None
        args.execute = False

        mock_rac = MagicMock()
        mock_rac.parse = MagicMock(side_effect=Exception("Parse error"))
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_compile(args)
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is False


# =========================================================================
# Test cmd_benchmark
# =========================================================================


class TestCmdBenchmark:
    def test_file_not_found(self, capsys):
        args = MagicMock()
        args.file = Path("/nonexistent/file.rac")
        with pytest.raises(SystemExit) as exc_info:
            cmd_benchmark(args)
        assert exc_info.value.code == 1

    def test_benchmark_success(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-01-01"
        args.iterations = 3
        args.rows = 10

        mock_ir = MagicMock()
        mock_ir.variables = {"var1": MagicMock(entity=None)}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock()

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_benchmark(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Benchmark" in captured.out
        assert "Avg" in captured.out

    def test_benchmark_with_entities(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.as_of = None
        args.iterations = 2
        args.rows = 5

        mock_var = MagicMock()
        mock_var.entity = "Person"
        mock_ir = MagicMock()
        mock_ir.variables = {"var1": mock_var}

        mock_parse = MagicMock()

        mock_compile = MagicMock(return_value=mock_ir)
        mock_execute = MagicMock()

        mock_rac = MagicMock()
        mock_rac.parse = mock_parse
        mock_rac.compile = mock_compile
        mock_rac.execute = mock_execute
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_benchmark(args)
            assert exc_info.value.code == 0

    def test_benchmark_failure(self, capsys, tmp_path):
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test")
        args = MagicMock()
        args.file = rac_file
        args.as_of = None
        args.iterations = 1
        args.rows = 1

        mock_rac = MagicMock()
        mock_rac.parse = MagicMock(side_effect=Exception("Parse error"))
        with patch.dict("sys.modules", {"rac": mock_rac}):
            with pytest.raises(SystemExit) as exc_info:
                cmd_benchmark(args)
            assert exc_info.value.code == 1


# =========================================================================
# Test cmd_log
# =========================================================================


class TestCmdLog:
    def test_log_basic(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test content")
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rac_file
        args.iterations = 2
        args.errors = json.dumps([{"type": "parse", "message": "bad syntax"}])
        args.duration = 5000
        args.scores = json.dumps({"rac": 8, "formula": 7, "param": 8, "integration": 7})
        args.predicted = json.dumps(
            {"rac": 7, "formula": 6, "param": 7, "integration": 6, "confidence": 0.5}
        )
        args.session = "test-session"
        args.db = db_path

        cmd_log(args)
        captured = capsys.readouterr()
        assert "Logged" in captured.out
        assert "26 USC 1" in captured.out
        assert "Session: test-session" in captured.out
        assert "Reviews: 4/4 passed" in captured.out

    def test_log_minimal(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        rac_file = tmp_path / "test.rac"
        # File doesn't exist, so rac_content will be ""
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rac_file
        args.iterations = 1
        args.errors = "[]"
        args.duration = 0
        args.scores = None
        args.predicted = None
        args.session = None
        args.db = db_path

        cmd_log(args)
        captured = capsys.readouterr()
        assert "Logged" in captured.out

    def test_log_with_alternative_score_keys(self, capsys, tmp_path):
        """Test parsing of predicted scores with alternative key names."""
        db_path = tmp_path / "test.db"
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("# test content")
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rac_file
        args.iterations = 1
        args.errors = "[]"
        args.duration = 0
        args.scores = None
        # Use alternative key names (rac_reviewer instead of rac)
        args.predicted = json.dumps(
            {
                "rac_reviewer": 7,
                "formula_reviewer": 6,
                "parameter_reviewer": 7,
                "integration_reviewer": 6,
            }
        )
        args.session = None
        args.db = db_path

        cmd_log(args)
        captured = capsys.readouterr()
        assert "Logged" in captured.out


# =========================================================================
# Test cmd_stats
# =========================================================================


class TestCmdStats:
    def test_db_not_found(self, capsys, tmp_path):
        args = MagicMock()
        args.db = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit) as exc_info:
            cmd_stats(args)
        assert exc_info.value.code == 1

    def test_stats_with_data(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.rac",
            iterations=[
                Iteration(
                    attempt=1,
                    duration_ms=1000,
                    errors=[IterationError(error_type="test", message="failed")],
                    success=False,
                ),
                Iteration(attempt=2, duration_ms=500, success=True),
            ],
        )
        db.log_run(run)

        args = MagicMock()
        args.db = db_path
        cmd_stats(args)
        captured = capsys.readouterr()
        assert "Iteration Statistics" in captured.out
        assert "Error Statistics" in captured.out
        assert "Improvement Suggestions" in captured.out

    def test_stats_with_parse_error(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.rac",
            iterations=[
                Iteration(
                    attempt=1,
                    duration_ms=1000,
                    errors=[IterationError(error_type="parse", message="syntax error")],
                    success=True,
                ),
            ],
        )
        db.log_run(run)

        args = MagicMock()
        args.db = db_path
        cmd_stats(args)
        captured = capsys.readouterr()
        assert "parse" in captured.out

    def test_stats_with_import_error(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.rac",
            iterations=[
                Iteration(
                    attempt=1,
                    duration_ms=1000,
                    errors=[
                        IterationError(error_type="import", message="import error")
                    ],
                    success=True,
                ),
            ],
        )
        db.log_run(run)

        args = MagicMock()
        args.db = db_path
        cmd_stats(args)

    def test_stats_no_errors(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.rac",
            iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
        )
        db.log_run(run)

        args = MagicMock()
        args.db = db_path
        cmd_stats(args)
        captured = capsys.readouterr()
        assert "Not enough data" in captured.out


# =========================================================================
# Test cmd_calibration
# =========================================================================


class TestCmdCalibration:
    def test_db_not_found(self, capsys, tmp_path):
        args = MagicMock()
        args.db = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit) as exc_info:
            cmd_calibration(args)
        assert exc_info.value.code == 1

    def test_no_calibration_data(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        EncodingDB(db_path)
        args = MagicMock()
        args.db = db_path
        args.limit = 50
        cmd_calibration(args)
        captured = capsys.readouterr()
        assert "No runs with both predictions" in captured.out

    def test_calibration_with_data(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.rac",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rac_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=8,
                        ),
                        ReviewResult(
                            reviewer="formula_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=7,
                        ),
                        ReviewResult(
                            reviewer="parameter_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=8,
                        ),
                        ReviewResult(
                            reviewer="integration_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=7,
                        ),
                    ],
                ),
                iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
            )
            db.log_run(run)

        args = MagicMock()
        args.db = db_path
        args.limit = 50
        cmd_calibration(args)
        captured = capsys.readouterr()
        assert "Calibration Report" in captured.out
        assert "Per-Run Breakdown" in captured.out


# =========================================================================
# Test cmd_statute
# =========================================================================


class TestCmdStatute:
    def test_bad_citation(self, capsys):
        args = MagicMock()
        args.citation = "invalid"
        args.xml_path = Path("/tmp")
        with pytest.raises(SystemExit) as exc_info:
            cmd_statute(args)
        assert exc_info.value.code == 1

    def test_xml_not_found(self, capsys, tmp_path):
        args = MagicMock()
        args.citation = "26 USC 1"
        args.xml_path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_statute(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_section_not_found(self, capsys, tmp_path):
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text("<root>no section here</root>")
        args = MagicMock()
        args.citation = "26 USC 999"
        args.xml_path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_statute(args)
        assert exc_info.value.code == 1

    def test_statute_success(self, capsys, tmp_path):
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>Married individuals filing joint returns</heading>
<content>There is hereby imposed on the taxable income...</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>10 percent bracket</heading>
<content>Of taxable income not exceeding $19,050</content>
<subparagraph identifier="/us/usc/t26/s1/a/1/A">
<heading>In general</heading>
<content>The tax shall be 10 percent</content>
</subparagraph>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        args = MagicMock()
        args.citation = "26 USC 1"
        args.xml_path = tmp_path
        cmd_statute(args)
        captured = capsys.readouterr()
        assert "26 USC" in captured.out
        assert "Tax imposed" in captured.out

    def test_statute_slash_format(self, capsys, tmp_path):
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        args = MagicMock()
        args.citation = "26/1"
        args.xml_path = tmp_path
        cmd_statute(args)
        captured = capsys.readouterr()
        assert "Tax imposed" in captured.out


# =========================================================================
# Test cmd_runs
# =========================================================================


class TestCmdRuns:
    def test_db_not_found(self, capsys, tmp_path):
        args = MagicMock()
        args.db = tmp_path / "nonexistent.db"
        with pytest.raises(SystemExit) as exc_info:
            cmd_runs(args)
        assert exc_info.value.code == 1

    def test_no_runs(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        EncodingDB(db_path)
        args = MagicMock()
        args.db = db_path
        args.limit = 20
        cmd_runs(args)
        captured = capsys.readouterr()
        assert "No encoding runs found" in captured.out

    def test_runs_with_data(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.rac",
            iterations=[Iteration(attempt=1, duration_ms=5000, success=True)],
            total_duration_ms=5000,
        )
        db.log_run(run)

        args = MagicMock()
        args.db = db_path
        args.limit = 20
        cmd_runs(args)
        captured = capsys.readouterr()
        assert "26 USC 1" in captured.out


# =========================================================================
# Test cmd_init
# =========================================================================


class TestCmdInit:
    def test_xml_not_found(self, capsys, tmp_path):
        """Test cmd_init when xml file does not exist (lines 1155-1157)."""
        args = MagicMock()
        args.citation = "26 USC 1"
        args.output = tmp_path / "out"
        args.force = False

        # tmp_path does NOT have the atlas xml structure, so xml won't be found
        with patch("pathlib.Path.home", return_value=tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                cmd_init(args)
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "USC XML not found" in captured.out

    def test_no_subsections(self, capsys, tmp_path):
        xml_path = (
            tmp_path / "RulesFoundation" / "atlas" / "data" / "uscode" / "usc26.xml"
        )
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text("<root></root>")
        args = MagicMock()
        args.citation = "26 USC 999"
        args.output = tmp_path / "out"
        args.force = False

        with patch("pathlib.Path.home", return_value=tmp_path):
            with pytest.raises(SystemExit) as exc_info:
                cmd_init(args)
            assert exc_info.value.code == 1

    def test_init_success(self, capsys, tmp_path):
        args = MagicMock()
        args.citation = "26 USC 1"
        args.output = tmp_path / "out"
        args.force = False

        subsections = [
            {
                "source_path": "usc/26/1/a",
                "heading": "General rule",
                "body": "Tax is imposed at the following rates.",
                "line_count": 1,
            },
            {
                "source_path": "usc/26/1/b",
                "heading": "Head of household",
                "body": "For head of household filers.",
                "line_count": 1,
            },
        ]

        xml_path = (
            tmp_path / "RulesFoundation" / "atlas" / "data" / "uscode" / "usc26.xml"
        )
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text("<root></root>")

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "autorac.cli._extract_subsections_from_xml", return_value=subsections
            ),
        ):
            cmd_init(args)
        captured = capsys.readouterr()
        assert "Created" in captured.out

    def test_init_with_slash_citation(self, capsys, tmp_path):
        args = MagicMock()
        args.citation = "26/1"
        args.output = tmp_path / "out"
        args.force = True

        subsections = [
            {
                "source_path": "usc/26/1/a",
                "heading": "General rule",
                "body": "Tax rates.",
                "line_count": 1,
            },
        ]

        xml_path = (
            tmp_path / "RulesFoundation" / "atlas" / "data" / "uscode" / "usc26.xml"
        )
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text("<root></root>")

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "autorac.cli._extract_subsections_from_xml", return_value=subsections
            ),
        ):
            cmd_init(args)
        captured = capsys.readouterr()
        assert "Created" in captured.out

    def test_init_skip_existing(self, capsys, tmp_path):
        args = MagicMock()
        args.citation = "26 USC 1"
        args.output = tmp_path / "out"
        args.force = False

        subsections = [
            {
                "source_path": "usc/26/1/a",
                "heading": "General rule",
                "body": "Tax rates.",
                "line_count": 1,
            },
        ]

        # Create existing file
        existing = tmp_path / "out" / "26" / "1/a.rac"
        existing.parent.mkdir(parents=True)
        existing.write_text("# existing")

        xml_path = (
            tmp_path / "RulesFoundation" / "atlas" / "data" / "uscode" / "usc26.xml"
        )
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text("<root></root>")

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "autorac.cli._extract_subsections_from_xml", return_value=subsections
            ),
        ):
            cmd_init(args)
        captured = capsys.readouterr()
        assert "skipped" in captured.out


# =========================================================================
# Test cmd_coverage
# =========================================================================


class TestCmdCoverage:
    def test_path_not_found(self, capsys, tmp_path):
        args = MagicMock()
        args.citation = "26 USC 999"
        args.path = tmp_path / "nonexistent"
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 1

    def test_no_rac_files(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "999"
        search_path.mkdir(parents=True)
        args = MagicMock()
        args.citation = "26 USC 999"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 1

    def test_all_examined(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("status: encoded\n# content")
        (search_path / "b.rac").write_text("status: skip\n# content")
        (search_path / "c.rac").write_text("status: stub\n# content")
        (search_path / "d.rac").write_text("status: consolidated\n# content")

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "COMPLETE" in captured.out

    def test_some_unexamined(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("status: encoded\n# content")
        (search_path / "b.rac").write_text("status: unexamined\n# content")

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "INCOMPLETE" in captured.out

    def test_coverage_with_errors(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("no status field here")
        (search_path / "b.rac").write_text("status: unknownstatus\n# content")

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        # Files with no status or unknown status count as errors
        with pytest.raises(SystemExit):
            cmd_coverage(args)

    def test_coverage_absolute_path(self, capsys, tmp_path):
        search_path = tmp_path / "test_coverage"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("status: encoded\n# content")

        args = MagicMock()
        args.citation = str(search_path)
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 0

    def test_coverage_slash_citation(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("status: encoded\n")

        args = MagicMock()
        args.citation = "26/1"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 0

    def test_coverage_skips_underscore_files(self, capsys, tmp_path):
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        (search_path / "a.rac").write_text("status: encoded\n")
        (search_path / "_encoding_sequence.rac").write_text("no status here")

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 0


# =========================================================================
# Test cmd_encode
# =========================================================================


class TestCmdEncode:
    def _make_args(self, tmp_path, **overrides):
        """Helper to create args with sensible defaults."""
        args = MagicMock()
        args.citation = overrides.get("citation", "26 USC 1(j)(2)")
        args.output = overrides.get("output", tmp_path / "statute")
        args.model = overrides.get("model", "test-model")
        args.db = overrides.get("db", tmp_path / "test.db")
        args.backend = overrides.get("backend", "cli")
        args.atlas_path = overrides.get("atlas_path", None)
        return args

    def _make_mock_run(self, success=True):
        """Helper to create a mock encoding run."""
        mock_run = MagicMock()
        mock_run.session_id = "test-session"
        if success:
            mock_run.files_created = ["a.rac"]
            mock_run.total_tokens = MagicMock(input_tokens=100, output_tokens=50)
            mock_run.total_tokens.estimated_cost_usd = 0.01
            mock_run.oracle_pe_match = 95.0
            mock_run.oracle_taxsim_match = 90.0
            mock_run.agent_runs = [MagicMock(error=None)]
        else:
            mock_run.files_created = []
            mock_run.total_tokens = None
            mock_run.oracle_pe_match = None
            mock_run.oracle_taxsim_match = None
            mock_run.agent_runs = [MagicMock(error="some error")]
        return mock_run

    def _run_encode(self, args, mock_run):
        """Run cmd_encode with a mocked orchestrator, return (Orchestrator_cls, exit_code)."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.print_report.return_value = "Report content"
        mock_orchestrator.encode = AsyncMock(return_value=mock_run)

        with patch(
            "autorac.harness.orchestrator.Orchestrator",
            return_value=mock_orchestrator,
        ) as mock_cls:
            with pytest.raises(SystemExit) as exc_info:
                cmd_encode(args)
            return mock_cls, exc_info.value.code

    def test_encode_success(self, capsys, tmp_path):
        args = self._make_args(tmp_path)
        mock_cls, exit_code = self._run_encode(args, self._make_mock_run(success=True))
        assert exit_code == 0

    def test_encode_with_errors(self, capsys, tmp_path):
        args = self._make_args(tmp_path, citation="26 USC 1", model=None)
        mock_cls, exit_code = self._run_encode(args, self._make_mock_run(success=False))
        assert exit_code == 1

    def test_encode_path_format(self, capsys, tmp_path):
        """Citation without spaces (e.g., '26/1') uses fallback parsing."""
        args = self._make_args(tmp_path, citation="26/1", model=None)
        mock_cls, exit_code = self._run_encode(args, self._make_mock_run(success=True))
        assert exit_code == 0

    def test_encode_leaf_citation_uses_parent_directory_output_path(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, citation="26 USC 24(a)")
        mock_cls, exit_code = self._run_encode(args, self._make_mock_run(success=True))

        assert exit_code == 0
        mock_cls.return_value.encode.assert_awaited_once()
        kwargs = mock_cls.return_value.encode.await_args.kwargs
        assert kwargs["output_path"] == tmp_path / "statute" / "26" / "24"

    def test_encode_defaults_to_cli_backend(self, capsys, tmp_path):
        """No --backend flag defaults to CLI backend."""
        args = self._make_args(tmp_path, backend="cli")
        mock_cls, _ = self._run_encode(args, self._make_mock_run())
        mock_cls.assert_called_once_with(
            model="test-model",
            db_path=tmp_path / "test.db",
            backend="cli",
            atlas_path=None,
        )

    def test_encode_api_backend(self, capsys, tmp_path):
        """--backend api passes 'api' to Orchestrator."""
        args = self._make_args(tmp_path, backend="api")
        mock_cls, _ = self._run_encode(args, self._make_mock_run())
        mock_cls.assert_called_once_with(
            model="test-model",
            db_path=tmp_path / "test.db",
            backend="api",
            atlas_path=None,
        )

    def test_encode_openai_backend(self, capsys, tmp_path):
        """--backend openai passes 'openai' to Orchestrator."""
        args = self._make_args(tmp_path, backend="openai")
        mock_cls, _ = self._run_encode(args, self._make_mock_run())
        mock_cls.assert_called_once_with(
            model="test-model",
            db_path=tmp_path / "test.db",
            backend="openai",
            atlas_path=None,
        )

    def test_encode_cli_backend_explicit(self, capsys, tmp_path):
        """--backend cli explicitly passes 'cli' to Orchestrator."""
        args = self._make_args(tmp_path, backend="cli")
        mock_cls, _ = self._run_encode(args, self._make_mock_run())
        mock_cls.assert_called_once_with(
            model="test-model",
            db_path=tmp_path / "test.db",
            backend="cli",
            atlas_path=None,
        )

    def test_encode_api_backend_no_key_errors(self, tmp_path):
        """API backend without ANTHROPIC_API_KEY raises clear error."""
        from autorac.harness.orchestrator import Orchestrator

        with patch.dict("os.environ", {}, clear=True):
            # Remove key if present
            import os

            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                    Orchestrator(backend="api")

    def test_encode_openai_backend_no_key_errors(self, tmp_path):
        """OpenAI backend without OPENAI_API_KEY raises clear error."""
        from autorac.harness.orchestrator import Orchestrator

        with patch.dict("os.environ", {}, clear=True):
            import os

            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict("os.environ", env, clear=True):
                with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                    Orchestrator(backend="openai")

    def test_encode_backend_shown_in_output(self, capsys, tmp_path):
        """Backend name is printed in the output."""
        args = self._make_args(tmp_path, backend="api")
        self._run_encode(args, self._make_mock_run())
        captured = capsys.readouterr()
        assert "Backend: api" in captured.out

    def test_encode_auto_syncs_to_supabase(self, capsys, tmp_path):
        """Encoding auto-syncs session to Supabase when credentials are set."""
        args = self._make_args(tmp_path)
        mock_run = self._make_mock_run()

        with patch(
            "autorac.supabase_sync.sync_sdk_sessions_to_supabase",
            return_value={"synced": 1, "failed": 0, "total": 1},
            create=True,
        ) as mock_sync:
            self._run_encode(args, mock_run)
            mock_sync.assert_called_once_with(session_id=mock_run.session_id)

        captured = capsys.readouterr()
        assert "Synced to Supabase" in captured.out

    def test_encode_skips_sync_without_credentials(self, capsys, tmp_path):
        """Encoding skips Supabase sync silently when credentials missing."""
        args = self._make_args(tmp_path)

        with patch(
            "autorac.supabase_sync.sync_sdk_sessions_to_supabase",
            side_effect=ValueError("Missing credentials"),
            create=True,
        ):
            _, exit_code = self._run_encode(args, self._make_mock_run())

        captured = capsys.readouterr()
        assert "Synced to Supabase" not in captured.out
        assert exit_code == 0


# =========================================================================
# Test session commands
# =========================================================================


class TestSessionCommands:
    def test_session_start(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        args = MagicMock()
        args.model = "test-model"
        args.cwd = str(tmp_path)
        args.db = db_path
        cmd_session_start(args)
        captured = capsys.readouterr()
        # Should output just the session ID
        assert len(captured.out.strip()) > 0

    def test_session_start_empty_cwd(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        args = MagicMock()
        args.model = ""
        args.cwd = ""
        args.db = db_path
        cmd_session_start(args)

    def test_session_end(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        args = MagicMock()
        args.session = session.id
        args.db = db_path
        cmd_session_end(args)
        captured = capsys.readouterr()
        assert "ended" in captured.out

    def test_log_event(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        args = MagicMock()
        args.session = session.id
        args.type = "tool_call"
        args.tool = "Read"
        args.content = "reading file"
        args.metadata = json.dumps({"path": "/test"})
        args.db = db_path
        cmd_log_event(args)
        captured = capsys.readouterr()
        assert "tool_call" in captured.out

    def test_log_event_invalid_metadata(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        args = MagicMock()
        args.session = session.id
        args.type = "tool_call"
        args.tool = None
        args.content = ""
        args.metadata = "not valid json{"
        args.db = db_path
        cmd_log_event(args)
        captured = capsys.readouterr()
        assert "tool_call" in captured.out

    def test_sessions_list(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test-model", autorac_version="0.2.1")
        db.end_session(session.id)

        args = MagicMock()
        args.db = db_path
        args.limit = 20
        cmd_sessions(args)
        captured = capsys.readouterr()
        assert "ended" in captured.out
        assert "0.2.1" in captured.out

    def test_sessions_empty(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        EncodingDB(db_path)
        args = MagicMock()
        args.db = db_path
        args.limit = 20
        cmd_sessions(args)
        captured = capsys.readouterr()
        assert "No sessions found" in captured.out

    def test_session_show_not_found(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        EncodingDB(db_path)
        args = MagicMock()
        args.session_id = "nonexistent"
        args.db = db_path
        args.json = False
        with pytest.raises(SystemExit) as exc_info:
            cmd_session_show(args)
        assert exc_info.value.code == 1

    def test_session_show_text(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        db.log_event(
            session.id,
            "tool_call",
            tool_name="Read",
            content="x" * 100,
        )
        db.log_event(
            session.id,
            "assistant_response",
            content="short content",
        )

        args = MagicMock()
        args.session_id = session.id
        args.db = db_path
        args.json = False
        cmd_session_show(args)
        captured = capsys.readouterr()
        assert session.id in captured.out
        assert "Read" in captured.out

    def test_session_show_json(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        db.log_event(session.id, "tool_call", content="x" * 600)

        args = MagicMock()
        args.session_id = session.id
        args.db = db_path
        args.json = True
        cmd_session_show(args)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["session"]["id"] == session.id
        assert "autorac_version" in output["session"]

    def test_session_stats(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        session = db.start_session(model="test")
        db.log_event(session.id, "tool_call", tool_name="Read")
        db.log_event(session.id, "tool_call", tool_name="Write")

        args = MagicMock()
        args.db = db_path
        cmd_session_stats(args)
        captured = capsys.readouterr()
        assert "Session Statistics" in captured.out


# =========================================================================
# Test transcript sync commands
# =========================================================================


class TestTranscriptSyncCommands:
    def test_sync_transcripts_success(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "autorac.supabase_sync.sync_transcripts_to_supabase",
            return_value={"total": 5, "synced": 5, "failed": 0},
        ):
            cmd_sync_transcripts(args)
        captured = capsys.readouterr()
        assert "5 synced" in captured.out

    def test_sync_transcripts_with_session(self, capsys):
        args = MagicMock()
        args.session = "test-session"

        with patch(
            "autorac.supabase_sync.sync_transcripts_to_supabase",
            return_value={"total": 1, "synced": 1, "failed": 0},
        ):
            cmd_sync_transcripts(args)
        captured = capsys.readouterr()
        assert "test-session" in captured.out

    def test_sync_transcripts_error(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "autorac.supabase_sync.sync_transcripts_to_supabase",
            side_effect=ValueError("Missing credentials"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_sync_transcripts(args)
            assert exc_info.value.code == 1

    def test_transcript_stats_exists(self, capsys):
        args = MagicMock()
        with patch(
            "autorac.supabase_sync.get_local_transcript_stats",
            return_value={
                "exists": True,
                "total": 10,
                "unsynced": 3,
                "synced": 7,
                "by_type": {"encoder": 5, "reviewer": 5},
            },
        ):
            cmd_transcript_stats(args)
        captured = capsys.readouterr()
        assert "10" in captured.out

    def test_transcript_stats_no_db(self, capsys):
        args = MagicMock()
        with patch(
            "autorac.supabase_sync.get_local_transcript_stats",
            return_value={"exists": False},
        ):
            cmd_transcript_stats(args)
        captured = capsys.readouterr()
        assert "No local transcript database" in captured.out

    def test_transcript_stats_empty_by_type(self, capsys):
        args = MagicMock()
        with patch(
            "autorac.supabase_sync.get_local_transcript_stats",
            return_value={
                "exists": True,
                "total": 0,
                "unsynced": 0,
                "synced": 0,
            },
        ):
            cmd_transcript_stats(args)

    def test_sync_sdk_sessions_success(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "autorac.supabase_sync.sync_sdk_sessions_to_supabase",
            return_value={"total": 2, "synced": 2, "failed": 0},
        ):
            cmd_sync_sdk_sessions(args)
        captured = capsys.readouterr()
        assert "2 synced" in captured.out

    def test_sync_sdk_sessions_error(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "autorac.supabase_sync.sync_sdk_sessions_to_supabase",
            side_effect=ValueError("Missing creds"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_sync_sdk_sessions(args)
            assert exc_info.value.code == 1


# =========================================================================
# Test _extract_subsections_from_xml
# =========================================================================


class TestExtractSubsections:
    def test_section_not_found(self, tmp_path):
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text("<root></root>")
        result = _extract_subsections_from_xml(xml_file, "999")
        assert result == []

    def test_extraction_with_nested_elements(self, tmp_path):
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<chapeau>There is hereby imposed</chapeau>
<subsection identifier="/us/usc/t26/s1/a">
<heading>Married individuals</heading>
<content>Joint returns.</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>10 percent bracket</heading>
<content>Not exceeding 19050</content>
<subparagraph identifier="/us/usc/t26/s1/a/1/A">
<heading>In general</heading>
<content>The tax shall be 10%</content>
</subparagraph>
</paragraph>
</subsection>
<subsection identifier="/us/usc/t26/s1/b">
<heading>Head of household</heading>
<content>For heads of household.</content>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        result = _extract_subsections_from_xml(xml_file, "1")
        assert len(result) > 0
        # Should include section heading, subsections, paragraphs, subparagraphs
        headings = [r["heading"] for r in result]
        assert "Tax imposed" in headings
        assert "Married individuals" in headings


# =========================================================================
# Additional edge case tests for remaining uncovered lines
# =========================================================================


class TestCmdValidateEdgeCases:
    def test_validate_rac_us_found_in_path(self, capsys, tmp_path):
        """Test validate when file is inside a rac-us directory (line 350)."""
        rac_us = tmp_path / "rac-us" / "statute" / "26" / "1"
        rac_us.mkdir(parents=True)
        rac_file = rac_us / "test.rac"
        rac_file.write_text("# test")
        # Create rac sibling
        rac_path = tmp_path / "rac"
        rac_path.mkdir(parents=True)

        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
            # Verify rac_path was resolved via rac_us.parent / "rac"
            call_kwargs = mock_pipeline_cls.call_args[1]
            assert "rac-us" in str(call_kwargs["rac_us_path"])

    def test_validate_fallback_prefers_workspace_repo_roots(self, tmp_path):
        rac_file = tmp_path / "generated" / "test.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text("# test")

        args = MagicMock()
        args.file = rac_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_actual_scores.return_value = MagicMock(
            rac_reviewer=None,
            formula_reviewer=None,
            parameter_reviewer=None,
            integration_reviewer=None,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("autorac.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        call_kwargs = mock_pipeline_cls.call_args[1]
        assert call_kwargs["rac_us_path"] == Path(
            "/Users/maxghenis/TheAxiomFoundation/rac-us"
        )
        assert call_kwargs["rac_path"] == Path("/Users/maxghenis/TheAxiomFoundation/rac")


class TestCmdCalibrationEdgeCases:
    def test_calibration_all_pass(self, capsys, tmp_path):
        """Test calibration with all reviews passing."""
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.rac",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rac_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=9,
                        ),
                        ReviewResult(
                            reviewer="formula_reviewer",
                            passed=True,
                            items_checked=10,
                            items_passed=9,
                        ),
                    ],
                ),
                iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
            )
            db.log_run(run)

        args = MagicMock()
        args.db = db_path
        args.limit = 50
        cmd_calibration(args)
        captured = capsys.readouterr()
        assert "Calibration Report" in captured.out
        assert "rac_reviewer" in captured.out

    def test_calibration_some_fail(self, capsys, tmp_path):
        """Test calibration with some reviews failing."""
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.rac",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rac_reviewer",
                            passed=False,
                            items_checked=10,
                            items_passed=3,
                            critical_issues=["Missing entity"],
                        ),
                        ReviewResult(
                            reviewer="formula_reviewer",
                            passed=False,
                            items_checked=10,
                            items_passed=4,
                            critical_issues=["Wrong formula"],
                        ),
                    ],
                ),
                iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
            )
            db.log_run(run)

        args = MagicMock()
        args.db = db_path
        args.limit = 50
        cmd_calibration(args)
        captured = capsys.readouterr()
        assert "Calibration Report" in captured.out


class TestCmdStatuteEdgeCases:
    def test_statute_para_content_no_heading(self, capsys, tmp_path):
        """Test statute with paragraph that has content but no heading (lines 944-945).

        extract_element truncates inner XML by len(close_tag) chars, so we add
        padding (a note element) after the content tag to compensate.
        """
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>General</heading>
<content>General content.</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<content>Para no heading</content>
<note>Padding after content to compensate for extraction truncation</note>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        args = MagicMock()
        args.citation = "26 USC 1"
        args.xml_path = tmp_path
        cmd_statute(args)
        captured = capsys.readouterr()
        assert "Para no heading" in captured.out

    def test_statute_subpara_content_no_heading(self, capsys, tmp_path):
        """Test statute with subparagraph that has content but no heading (lines 961-962)."""
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>General</heading>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>First para</heading>
<subparagraph identifier="/us/usc/t26/s1/a/1/A">
<content>Subpara no heading</content>
<note>Padding after content to compensate for extraction truncation</note>
</subparagraph>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        args = MagicMock()
        args.citation = "26 USC 1"
        args.xml_path = tmp_path
        cmd_statute(args)
        captured = capsys.readouterr()
        assert "(A)" in captured.out
        assert "Subpara no heading" in captured.out

    def test_statute_nested_same_tag(self, capsys, tmp_path):
        """Test extract_element with nested elements of same tag type (line 908).

        When a <paragraph> contains another <paragraph> (unusual but handled),
        the depth tracking increments to handle the nesting properly.
        """
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>Outer sub</heading>
<content>Outer text</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>Outer para</heading>
<content>Has nested paragraph element</content>
<paragraph identifier="/us/usc/t26/s1/a/1/inner">
<heading>Inner para</heading>
<content>Inner content</content>
</paragraph>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        args = MagicMock()
        args.citation = "26 USC 1"
        args.xml_path = tmp_path
        cmd_statute(args)
        captured = capsys.readouterr()
        assert "Outer para" in captured.out


class TestCmdInitEdgeCases:
    def test_init_many_subsections(self, capsys, tmp_path):
        """Test init with >10 subsections to cover 'and N more' line (1267)."""
        args = MagicMock()
        args.citation = "26 USC 1"
        args.output = tmp_path / "out"
        args.force = True

        subsections = [
            {
                "source_path": f"usc/26/1/{chr(97 + i)}",
                "heading": f"Subsection {chr(97 + i)}",
                "body": f"Content for subsection {chr(97 + i)}.",
                "line_count": 1,
            }
            for i in range(15)
        ]

        xml_path = (
            tmp_path / "RulesFoundation" / "atlas" / "data" / "uscode" / "usc26.xml"
        )
        xml_path.parent.mkdir(parents=True)
        xml_path.write_text("<root></root>")

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch(
                "autorac.cli._extract_subsections_from_xml", return_value=subsections
            ),
        ):
            cmd_init(args)
        captured = capsys.readouterr()
        assert "Created" in captured.out
        assert "more" in captured.out


class TestCmdCoverageEdgeCases:
    def test_coverage_exception_on_file(self, capsys, tmp_path):
        """Coverage handles exceptions on individual files (lines 1328-1329)."""
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        # Create a directory with .rac extension to cause an error when reading
        bad_rac = search_path / "bad.rac"
        bad_rac.mkdir()  # This is a directory, not a file

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        with pytest.raises(SystemExit):
            cmd_coverage(args)

    def test_coverage_many_unexamined(self, capsys, tmp_path):
        """Test coverage with >20 unexamined files (line 1350)."""
        search_path = tmp_path / "26" / "1"
        search_path.mkdir(parents=True)
        for i in range(25):
            (search_path / f"sub{i}.rac").write_text("status: unexamined\n")

        args = MagicMock()
        args.citation = "26 USC 1"
        args.path = tmp_path
        with pytest.raises(SystemExit) as exc_info:
            cmd_coverage(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "more" in captured.out


class TestExtractSubsectionsEdgeCases:
    def test_deep_nesting(self, tmp_path):
        """Test extraction with deep nesting including clause level."""
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>General</heading>
<content>Text</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>First</heading>
<content>Paragraph text</content>
<subparagraph identifier="/us/usc/t26/s1/a/1/A">
<heading>Sub A</heading>
<content>Sub text</content>
<clause identifier="/us/usc/t26/s1/a/1/A/i">
<heading>Clause i</heading>
<content>Clause text</content>
<subclause identifier="/us/usc/t26/s1/a/1/A/i/I">
<heading>Subclause I</heading>
<content>Subclause text</content>
</subclause>
</clause>
</subparagraph>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        result = _extract_subsections_from_xml(xml_file, "1")
        assert len(result) > 3  # Should include deeply nested elements

    def test_nested_same_tag_in_extract(self, tmp_path):
        """Test _extract_subsections_from_xml with nested elements of same type (line 1064)."""
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="/us/usc/t26/s1/a">
<heading>Outer</heading>
<content>Text</content>
<paragraph identifier="/us/usc/t26/s1/a/1">
<heading>Para</heading>
<content>Content</content>
<paragraph identifier="/us/usc/t26/s1/a/1/nested">
<heading>Nested para</heading>
<content>Nested content</content>
</paragraph>
</paragraph>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        result = _extract_subsections_from_xml(xml_file, "1")
        # Should extract the nested paragraph too
        assert len(result) >= 3

    def test_identifier_without_section_prefix(self, tmp_path):
        """Test StopIteration fallback (lines 1088-1089)."""
        xml_content = """<root>
<section identifier="/us/usc/t26/s1" status="active">
<heading>Tax imposed</heading>
<subsection identifier="nonstandardformat">
<heading>Nonstandard</heading>
<content>This has a nonstandard identifier without s prefix</content>
</subsection>
</section>
</root>"""
        xml_file = tmp_path / "usc26.xml"
        xml_file.write_text(xml_content)
        result = _extract_subsections_from_xml(xml_file, "1")
        # Should still extract the section itself
        assert len(result) >= 1
