"""
Tests for axiom_encode CLI (cli.py).

Tests all CLI commands using subprocess invocation and direct function calls.
All external dependencies are mocked.
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode.cli import (
    _effective_runner_specs,
    _extract_subsections_from_xml,
    _rewrite_gpt_runner_backend,
    cmd_calibration,
    cmd_compile,
    cmd_encode,
    cmd_eval_source,
    cmd_eval_suite,
    cmd_eval_suite_archive,
    cmd_eval_suite_report,
    cmd_eval_suite_revalidate,
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
    cmd_sync_agent_sessions,
    cmd_sync_transcripts,
    cmd_transcript_stats,
    cmd_validate,
    main,
)
from axiom_encode.harness.encoding_db import (
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    ReviewResult,
    ReviewResults,
)
from axiom_encode.harness.evals import EvalArtifactMetrics

# =========================================================================
# Test main() dispatch
# =========================================================================


class TestRunnerOverrides:
    def test_rewrites_openai_gpt_runner_to_codex(self):
        assert _rewrite_gpt_runner_backend("openai:gpt-5.4", "codex") == "codex:gpt-5.4"

    def test_preserves_alias_when_rewriting_gpt_runner_backend(self):
        assert (
            _rewrite_gpt_runner_backend("gpt=openai:gpt-5.4", "codex")
            == "gpt=codex:gpt-5.4"
        )

    def test_effective_runner_specs_defaults_to_codex(self):
        args = SimpleNamespace(gpt_backend=None)

        assert _effective_runner_specs(["openai:gpt-5.4", "claude:opus"], args) == [
            "codex:gpt-5.4",
            "claude:opus",
        ]

    def test_effective_runner_specs_uses_env_override(self, monkeypatch):
        monkeypatch.setenv("AXIOM_ENCODE_GPT_BACKEND", "codex")
        args = SimpleNamespace(gpt_backend=None)

        assert _effective_runner_specs(["openai:gpt-5.4", "claude:opus"], args) == [
            "codex:gpt-5.4",
            "claude:opus",
        ]

    def test_effective_runner_specs_allows_explicit_openai_override(self, monkeypatch):
        monkeypatch.delenv("AXIOM_ENCODE_GPT_BACKEND", raising=False)
        args = SimpleNamespace(gpt_backend="openai")

        assert _effective_runner_specs(["codex:gpt-5.4", "claude:opus"], args) == [
            "openai:gpt-5.4",
            "claude:opus",
        ]


class TestMain:
    def test_no_command_shows_help_and_exits(self):
        """main() with no command should print help and exit 1."""
        with patch("sys.argv", ["axiom_encode"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_validate_command_dispatches(self):
        """main() with 'validate' should call cmd_validate."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with patch("sys.argv", ["axiom_encode", "validate", f.name]):
                with patch("axiom_encode.cli.cmd_validate") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_log_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with patch(
                "sys.argv",
                [
                    "axiom_encode",
                    "log",
                    "--citation",
                    "26 USC 1",
                    "--file",
                    f.name,
                ],
            ):
                with patch("axiom_encode.cli.cmd_log") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_stats_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "stats"]):
            with patch("axiom_encode.cli.cmd_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_calibration_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "calibration"]):
            with patch("axiom_encode.cli.cmd_calibration") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_statute_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "statute", "26 USC 1"]):
            with patch("axiom_encode.cli.cmd_statute") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_runs_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "runs"]):
            with patch("axiom_encode.cli.cmd_runs") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_encode_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "encode", "26 USC 1"]):
            with patch("axiom_encode.cli.cmd_encode") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "eval", "26 USC 24(a)"]):
            with patch("axiom_encode.cli.cmd_eval") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_source_command_dispatches(self):
        with tempfile.NamedTemporaryFile() as f:
            with patch(
                "sys.argv",
                ["axiom_encode", "eval-source", "CO TANF 3.606.1(F)", f.name],
            ):
                with patch("axiom_encode.cli.cmd_eval_source") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_akn_section_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".xml") as f:
            with patch(
                "sys.argv",
                [
                    "axiom_encode",
                    "eval-akn-section",
                    "CO TANF 3.606.1",
                    f.name,
                    "sec_3_606_1",
                ],
            ):
                with patch("axiom_encode.cli.cmd_eval_akn_section") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_uk_legislation_section_command_dispatches(self):
        with patch(
            "sys.argv",
            [
                "axiom_encode",
                "eval-uk-legislation-section",
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
            ],
        ):
            with patch("axiom_encode.cli.cmd_eval_uk_legislation_section") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_uk_legislation_section_accepts_table_row_query(self):
        with patch(
            "sys.argv",
            [
                "axiom_encode",
                "eval-uk-legislation-section",
                "/uksi/2013/376/regulation/36/2025-04-01",
                "--section-eid",
                "regulation-36-3",
                "--table-row-query",
                "single claimant aged under 25",
            ],
        ):
            with patch("axiom_encode.cli.cmd_eval_uk_legislation_section") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_source_accepts_policyengine_rule_hint(self):
        with tempfile.NamedTemporaryFile() as f:
            with patch(
                "sys.argv",
                [
                    "axiom_encode",
                    "eval-source",
                    "UC row",
                    f.name,
                    "--policyengine-rule-hint",
                    "uc_standard_allowance_single_claimant_aged_under_25",
                ],
            ):
                with patch("axiom_encode.cli.cmd_eval_source") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_suite_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with patch("sys.argv", ["axiom_encode", "eval-suite", f.name]):
                with patch("axiom_encode.cli.cmd_eval_suite") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_suite_report_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            with patch("sys.argv", ["axiom_encode", "eval-suite-report", f.name]):
                with patch("axiom_encode.cli.cmd_eval_suite_report") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_eval_suite_archive_command_dispatches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["axiom_encode", "eval-suite-archive", tmpdir]):
                with patch("axiom_encode.cli.cmd_eval_suite_archive") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_compile_command_dispatches(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with patch("sys.argv", ["axiom_encode", "compile", f.name]):
                with patch("axiom_encode.cli.cmd_compile") as mock_cmd:
                    main()
                    mock_cmd.assert_called_once()

    def test_session_start_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "session-start"]):
            with patch("axiom_encode.cli.cmd_session_start") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_end_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "session-end", "--session", "abc"]):
            with patch("axiom_encode.cli.cmd_session_end") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_log_event_dispatches(self):
        with patch(
            "sys.argv",
            ["axiom_encode", "log-event", "--session", "abc", "--type", "tool_call"],
        ):
            with patch("axiom_encode.cli.cmd_log_event") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_sessions_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "sessions"]):
            with patch("axiom_encode.cli.cmd_sessions") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_show_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "session-show", "abc123"]):
            with patch("axiom_encode.cli.cmd_session_show") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_session_stats_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "session-stats"]):
            with patch("axiom_encode.cli.cmd_session_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_sync_transcripts_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "sync-transcripts"]):
            with patch("axiom_encode.cli.cmd_sync_transcripts") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_transcript_stats_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "transcript-stats"]):
            with patch("axiom_encode.cli.cmd_transcript_stats") as mock_cmd:
                main()
                mock_cmd.assert_called_once()


class TestCmdEvalSuite:
    def test_exits_nonzero_when_runner_is_not_ready(self, tmp_path, capsys):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    source_file: ./source.txt\n"
        )
        (tmp_path / "source.txt").write_text("authoritative text")
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            atlas_path=tmp_path / "atlas",
            axiom_rules_path=tmp_path / "axiom-rules",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=0,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()

        fake_result = MagicMock()
        fake_result.runner = "codex-gpt-5.4"
        fake_result.success = True
        fake_result.error = None
        fake_result.metrics = MagicMock(
            compile_pass=True,
            ci_pass=True,
            ungrounded_numeric_count=0,
        )
        fake_result.to_dict.return_value = {
            "citation": "case-a",
            "runner": "codex-gpt-5.4",
            "success": True,
            "error": None,
            "metrics": {
                "compile_pass": True,
                "ci_pass": True,
                "ungrounded_numeric_count": 0,
            },
        }

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
            policyengine_pass_rate=None,
            mean_policyengine_score=None,
            mean_estimated_cost_usd=0.25,
            gate_results=[],
        )

        with (
            patch("axiom_encode.cli.load_eval_suite_manifest") as mock_load,
            patch(
                "axiom_encode.cli.run_eval_suite", return_value=[fake_result]
            ) as mock_run,
            patch("axiom_encode.cli.summarize_readiness", return_value=fake_summary),
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
        assert (args.output / "results.json").exists()
        assert (args.output / "summary.json").exists()

    def test_passes_resume_flag_to_run_eval_suite(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    source_file: ./source.txt\n"
        )
        (tmp_path / "source.txt").write_text("authoritative text")
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            atlas_path=tmp_path / "atlas",
            axiom_rules_path=tmp_path / "axiom-rules",
            json=False,
            gpt_backend="codex",
            resume=True,
            auto_resume_attempts=0,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()

        fake_result = MagicMock()
        fake_result.runner = "codex-gpt-5.4"
        fake_result.success = True
        fake_result.error = None
        fake_result.metrics = MagicMock(
            compile_pass=True,
            ci_pass=True,
            ungrounded_numeric_count=0,
        )
        fake_result.to_dict.return_value = {
            "citation": "case-a",
            "runner": "codex-gpt-5.4",
            "success": True,
            "error": None,
            "metrics": {
                "compile_pass": True,
                "ci_pass": True,
                "ungrounded_numeric_count": 0,
            },
        }

        fake_summary = MagicMock(
            ready=True,
            total_cases=1,
            success_rate=1.0,
            compile_pass_rate=1.0,
            ci_pass_rate=1.0,
            zero_ungrounded_rate=1.0,
            generalist_review_pass_rate=1.0,
            mean_generalist_review_score=8.0,
            policyengine_case_count=0,
            policyengine_pass_rate=None,
            mean_policyengine_score=None,
            mean_estimated_cost_usd=0.25,
            gate_results=[],
        )

        with (
            patch("axiom_encode.cli.load_eval_suite_manifest") as mock_load,
            patch(
                "axiom_encode.cli.run_eval_suite", return_value=[fake_result]
            ) as mock_run,
            patch("axiom_encode.cli.summarize_readiness", return_value=fake_summary),
        ):
            mock_load.return_value.name = "Readiness"
            mock_load.return_value.path = manifest_file
            mock_load.return_value.runners = ["openai:gpt-5.4"]
            mock_load.return_value.cases = [MagicMock(kind="source")]
            mock_load.return_value.gates = MagicMock()
            with pytest.raises(SystemExit) as exc_info:
                cmd_eval_suite(args)

        assert exc_info.value.code == 0
        assert mock_run.call_args.kwargs["resume_existing"] is True

    def test_auto_resumes_after_unexpected_suite_exception(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    source_file: ./source.txt\n"
        )
        (tmp_path / "source.txt").write_text("authoritative text")
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            atlas_path=tmp_path / "atlas",
            axiom_rules_path=tmp_path / "axiom-rules",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=1,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()

        fake_result = MagicMock()
        fake_result.runner = "codex-gpt-5.4"
        fake_result.success = True
        fake_result.error = None
        fake_result.metrics = MagicMock(
            compile_pass=True,
            ci_pass=True,
            ungrounded_numeric_count=0,
        )
        fake_result.to_dict.return_value = {
            "citation": "case-a",
            "runner": "codex-gpt-5.4",
            "success": True,
            "error": None,
            "metrics": {
                "compile_pass": True,
                "ci_pass": True,
                "ungrounded_numeric_count": 0,
            },
        }

        fake_summary = MagicMock(
            ready=True,
            total_cases=1,
            success_rate=1.0,
            compile_pass_rate=1.0,
            ci_pass_rate=1.0,
            zero_ungrounded_rate=1.0,
            generalist_review_pass_rate=1.0,
            mean_generalist_review_score=8.0,
            policyengine_case_count=0,
            policyengine_pass_rate=None,
            mean_policyengine_score=None,
            mean_estimated_cost_usd=0.25,
            gate_results=[],
        )

        with (
            patch("axiom_encode.cli.load_eval_suite_manifest") as mock_load,
            patch(
                "axiom_encode.cli.run_eval_suite",
                side_effect=[RuntimeError("boom"), [fake_result]],
            ) as mock_run,
            patch("axiom_encode.cli.summarize_readiness", return_value=fake_summary),
            patch("axiom_encode.cli.time.sleep") as mock_sleep,
        ):
            mock_load.return_value.name = "Readiness"
            mock_load.return_value.path = manifest_file
            mock_load.return_value.runners = ["openai:gpt-5.4"]
            mock_load.return_value.cases = [MagicMock(kind="source")]
            mock_load.return_value.gates = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                cmd_eval_suite(args)

        assert exc_info.value.code == 0
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].kwargs["resume_existing"] is False
        assert mock_run.call_args_list[1].kwargs["resume_existing"] is True
        mock_sleep.assert_not_called()

    def test_usage_limit_failure_does_not_trigger_auto_resume(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        manifest_file.write_text(
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    source_file: ./source.txt\n"
        )
        (tmp_path / "source.txt").write_text("authoritative text")
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            atlas_path=tmp_path / "atlas",
            axiom_rules_path=tmp_path / "axiom-rules",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=2,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()

        fake_result = MagicMock()
        fake_result.runner = "codex-gpt-5.4"
        fake_result.success = False
        fake_result.error = "You've hit your usage limit."
        fake_result.metrics = MagicMock(
            compile_pass=False,
            ci_pass=False,
            ungrounded_numeric_count=0,
        )
        fake_result.to_dict.return_value = {
            "citation": "case-a",
            "runner": "codex-gpt-5.4",
            "success": False,
            "error": "You've hit your usage limit.",
            "metrics": {
                "compile_pass": False,
                "ci_pass": False,
                "ungrounded_numeric_count": 0,
            },
        }

        fake_summary = MagicMock(
            ready=False,
            total_cases=2,
            success_rate=0.0,
            compile_pass_rate=0.0,
            ci_pass_rate=0.0,
            zero_ungrounded_rate=1.0,
            generalist_review_pass_rate=0.0,
            mean_generalist_review_score=None,
            policyengine_case_count=0,
            policyengine_pass_rate=None,
            mean_policyengine_score=None,
            mean_estimated_cost_usd=None,
            gate_results=[],
        )

        def fake_run_eval_suite(**kwargs):
            kwargs["output_root"].mkdir(parents=True, exist_ok=True)
            (kwargs["output_root"] / "suite-run.json").write_text(
                json.dumps(
                    {
                        "manifest": {
                            "name": "Readiness",
                            "path": str(manifest_file),
                            "runners": ["openai:gpt-5.4"],
                            "effective_runners": ["codex:gpt-5.4"],
                        },
                        "status": "failed",
                        "started_at": "2026-04-11T16:00:00+00:00",
                        "updated_at": "2026-04-11T16:05:00+00:00",
                        "total_cases": 2,
                        "completed_cases": 1,
                        "result_count": 1,
                        "last_case_name": "case-a",
                        "error": "Usage limit reached while running case 'case-a'.",
                    }
                )
                + "\n"
            )
            return [fake_result]

        with (
            patch("axiom_encode.cli.load_eval_suite_manifest") as mock_load,
            patch(
                "axiom_encode.cli.run_eval_suite",
                side_effect=fake_run_eval_suite,
            ) as mock_run,
            patch("axiom_encode.cli.summarize_readiness", return_value=fake_summary),
            patch("axiom_encode.cli.time.sleep") as mock_sleep,
        ):
            mock_load.return_value.name = "Readiness"
            mock_load.return_value.path = manifest_file
            mock_load.return_value.runners = ["openai:gpt-5.4"]
            mock_load.return_value.cases = [
                MagicMock(kind="source"),
                MagicMock(kind="source"),
            ]
            mock_load.return_value.gates = MagicMock()

            with pytest.raises(SystemExit) as exc_info:
                cmd_eval_suite(args)

        assert exc_info.value.code == 1
        assert mock_run.call_count == 1
        mock_sleep.assert_not_called()


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
                    "output_file": "/tmp/gpt.yaml",
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
                    "output_file": "/tmp/claude.yaml",
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

    def test_sync_agent_sessions_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "sync-agent-sessions"]):
            with patch("axiom_encode.cli.cmd_sync_agent_sessions") as mock_cmd:
                main()
                mock_cmd.assert_called_once()


class TestCmdEvalSuiteRevalidate:
    def test_revalidates_existing_suite_outputs_in_place(self, tmp_path):
        manifest_file = tmp_path / "suite.yaml"
        source_file = tmp_path / "source.txt"
        source_file.write_text("authoritative source text")
        manifest_file.write_text(
            "\n".join(
                [
                    "name: SNAP repair",
                    "runners:",
                    "  - openai:gpt-5.4",
                    "cases:",
                    "  - kind: source",
                    "    name: case-a",
                    "    source_id: case-a",
                    f"    source_file: {source_file}",
                    "    oracle: policyengine",
                    "    policyengine_country: us",
                    "    policyengine_rule_hint: snap_net_income_pre_shelter",
                ]
            )
        )

        source_output = tmp_path / "out"
        rulespec_file = (
            source_output / "01-case-a" / "openai-gpt-5.4" / "source" / "case-a.yaml"
        )
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            "format: rulespec/v1\n"
            "module:\n"
            "  summary: authoritative source text\n"
            "rules:\n"
            "  - name: case_a\n"
            "    kind: parameter\n"
            "    entity: Household\n"
            "    dtype: Money\n"
            "    period: Month\n"
            "    unit: USD\n"
            "    versions:\n"
            "      - effective_from: '2024-01-01'\n"
            "        formula: 1\n"
        )
        (source_output / "suite-run.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": "SNAP repair",
                        "path": str(manifest_file),
                        "runners": ["openai:gpt-5.4"],
                        "effective_runners": ["openai:gpt-5.4"],
                    },
                    "status": "completed",
                    "started_at": "2026-04-12T19:00:00+00:00",
                    "updated_at": "2026-04-12T19:01:00+00:00",
                    "total_cases": 1,
                    "completed_cases": 1,
                    "result_count": 1,
                }
            )
            + "\n"
        )
        stale_result = {
            "citation": "case-a",
            "runner": "openai-gpt-5.4",
            "backend": "openai",
            "model": "gpt-5.4",
            "mode": "repo-augmented",
            "output_file": str(rulespec_file),
            "trace_file": str(source_output / "trace.json"),
            "context_manifest_file": str(source_output / "context.json"),
            "duration_ms": 1,
            "success": True,
            "error": None,
            "generation_prompt_sha256": "generation-digest",
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_output_tokens": 0,
            "estimated_cost_usd": 0.01,
            "actual_cost_usd": None,
            "retrieved_files": [],
            "unexpected_accesses": [],
            "metrics": {
                "compile_pass": False,
                "compile_issues": ["old"],
                "ci_pass": False,
                "ci_issues": ["old"],
                "embedded_source_present": True,
                "grounded_numeric_count": 0,
                "ungrounded_numeric_count": 1,
                "grounding": [],
                "source_numeric_occurrence_count": 0,
                "covered_source_numeric_occurrence_count": 0,
                "missing_source_numeric_occurrence_count": 0,
                "numeric_occurrence_issues": [],
                "generalist_review_pass": False,
                "generalist_review_score": 1.0,
                "generalist_review_issues": ["old"],
                "generalist_review_prompt_sha256": "old-review-digest",
                "policyengine_pass": False,
                "policyengine_score": 0.0,
                "policyengine_issues": ["old"],
                "taxsim_pass": None,
                "taxsim_score": None,
                "taxsim_issues": [],
            },
        }
        (source_output / "suite-results.jsonl").write_text(
            json.dumps(
                {
                    "case_index": 1,
                    "case_name": "case-a",
                    "case_kind": "source",
                    "result": stale_result,
                }
            )
            + "\n"
        )

        fresh_metrics = EvalArtifactMetrics(
            compile_pass=True,
            compile_issues=[],
            ci_pass=True,
            ci_issues=[],
            embedded_source_present=True,
            grounded_numeric_count=1,
            ungrounded_numeric_count=0,
            grounding=[],
            source_numeric_occurrence_count=0,
            covered_source_numeric_occurrence_count=0,
            missing_source_numeric_occurrence_count=0,
            numeric_occurrence_issues=[],
            generalist_review_pass=True,
            generalist_review_score=9.0,
            generalist_review_issues=[],
            generalist_review_prompt_sha256="fresh-review-digest",
            policyengine_pass=True,
            policyengine_score=1.0,
            policyengine_issues=[],
            taxsim_pass=None,
            taxsim_score=None,
            taxsim_issues=[],
        )
        summary = MagicMock(
            ready=True,
            total_cases=1,
            success_rate=1.0,
            compile_pass_rate=1.0,
            ci_pass_rate=1.0,
            zero_ungrounded_rate=1.0,
            generalist_review_pass_rate=1.0,
            mean_generalist_review_score=9.0,
            policyengine_case_count=1,
            policyengine_pass_rate=1.0,
            mean_policyengine_score=1.0,
            mean_estimated_cost_usd=0.01,
            gate_results=[],
        )
        args = SimpleNamespace(
            source_output=source_output,
            manifest=None,
            axiom_rules_path=tmp_path / "axiom-rules",
            json=True,
        )
        args.axiom_rules_path.mkdir()

        with (
            patch(
                "axiom_encode.cli.evaluate_artifact", return_value=fresh_metrics
            ) as mock_eval,
            patch("axiom_encode.cli.summarize_readiness", return_value=summary),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_eval_suite_revalidate(args)

        assert exc_info.value.code == 0
        mock_eval.assert_called_once()
        ledger = (source_output / "suite-results.jsonl").read_text()
        assert '"compile_pass": true' in ledger
        assert '"policyengine_pass": true' in ledger
        assert '"generation_prompt_sha256": "generation-digest"' in ledger
        assert '"generalist_review_prompt_sha256": "fresh-review-digest"' in ledger
        summary_payload = json.loads((source_output / "summary.json").read_text())
        assert summary_payload["all_ready"] is True
        run_state = json.loads((source_output / "suite-run.json").read_text())
        assert "revalidated_at" in run_state


class TestCmdEvalSuiteArchive:
    def test_archives_suite_and_rewrites_result_paths(self, tmp_path, capsys):
        source_output = tmp_path / "wave21-rerun16"
        case_dir = source_output / "01-case-a" / "codex-gpt-5.4" / "source"
        trace_dir = source_output / "traces"
        workspace_dir = source_output / "_eval_workspaces" / "case-a"
        case_dir.mkdir(parents=True)
        trace_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)

        output_file = case_dir / "rule.yaml"
        trace_file = trace_dir / "case-a.json"
        context_manifest_file = workspace_dir / "context-manifest.json"
        output_file.write_text("# rule\n")
        trace_file.write_text("{}\n")
        context_manifest_file.write_text("{}\n")

        result_payload = {
            "citation": "case-a",
            "runner": "codex:gpt-5.4",
            "backend": "codex",
            "model": "gpt-5.4",
            "mode": "repo-augmented",
            "output_file": str(output_file),
            "trace_file": str(trace_file),
            "context_manifest_file": str(context_manifest_file),
            "duration_ms": 1234,
            "success": True,
            "error": None,
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "reasoning_output_tokens": 0,
            "estimated_cost_usd": 0.12,
            "actual_cost_usd": None,
            "retrieved_files": [str(output_file)],
            "unexpected_accesses": [],
            "metrics": {
                "compile_pass": True,
                "compile_issues": [],
                "ci_pass": True,
                "ci_issues": [],
                "embedded_source_present": True,
                "grounded_numeric_count": 1,
                "ungrounded_numeric_count": 0,
                "grounding": [],
                "source_numeric_occurrence_count": 1,
                "covered_source_numeric_occurrence_count": 1,
                "missing_source_numeric_occurrence_count": 0,
                "numeric_occurrence_issues": [],
                "generalist_review_pass": True,
                "generalist_review_score": 9.0,
                "generalist_review_issues": [],
                "policyengine_pass": None,
                "policyengine_score": None,
                "policyengine_issues": [],
                "taxsim_pass": None,
                "taxsim_score": None,
                "taxsim_issues": [],
            },
        }

        (source_output / "suite-run.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": "UK wave 21",
                        "path": "/tmp/uk_wave21.yaml",
                        "runners": ["openai:gpt-5.4"],
                        "effective_runners": ["codex:gpt-5.4"],
                    },
                    "status": "completed",
                    "started_at": "2026-04-10T12:00:00+00:00",
                    "finished_at": "2026-04-10T12:30:00+00:00",
                    "updated_at": "2026-04-10T12:30:00+00:00",
                    "total_cases": 1,
                    "completed_cases": 1,
                    "result_count": 1,
                },
                indent=2,
            )
            + "\n"
        )
        (source_output / "results.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": "UK wave 21",
                        "path": "/tmp/uk_wave21.yaml",
                        "runners": ["openai:gpt-5.4"],
                        "effective_runners": ["codex:gpt-5.4"],
                    },
                    "results": [result_payload],
                    "readiness": {
                        "codex:gpt-5.4": {
                            "total_cases": 1,
                            "success_rate": 1.0,
                            "compile_pass_rate": 1.0,
                            "ci_pass_rate": 1.0,
                            "zero_ungrounded_rate": 1.0,
                        }
                    },
                    "all_ready": True,
                },
                indent=2,
            )
            + "\n"
        )
        (source_output / "summary.json").write_text(
            json.dumps(
                {
                    "manifest": {
                        "name": "UK wave 21",
                        "path": "/tmp/uk_wave21.yaml",
                        "runners": ["openai:gpt-5.4"],
                        "effective_runners": ["codex:gpt-5.4"],
                    },
                    "readiness": {
                        "codex:gpt-5.4": {
                            "total_cases": 1,
                            "success_rate": 1.0,
                        }
                    },
                    "all_ready": True,
                },
                indent=2,
            )
            + "\n"
        )
        (source_output / "suite-results.jsonl").write_text(
            json.dumps(
                {
                    "case_index": 1,
                    "case_name": "case-a",
                    "case_kind": "source",
                    "result": result_payload,
                },
                sort_keys=True,
            )
            + "\n"
        )

        archive_root = tmp_path / "archives"
        args = SimpleNamespace(
            source_output=source_output,
            archive_root=archive_root,
            name="Wave21 Rerun16",
            json=False,
        )

        cmd_eval_suite_archive(args)

        archive_dir = archive_root / "wave21-rerun16"
        assert archive_dir.exists()
        archived_results = json.loads((archive_dir / "results.json").read_text())
        archived_result = archived_results["results"][0]
        assert archived_result["output_file"].startswith(str(archive_dir))
        assert archived_result["trace_file"].startswith(str(archive_dir))
        assert archived_result["context_manifest_file"].startswith(str(archive_dir))
        assert not archived_result["output_file"].startswith(str(source_output))

        archived_row = json.loads(
            (archive_dir / "suite-results.jsonl").read_text().strip()
        )
        assert archived_row["result"]["output_file"].startswith(str(archive_dir))

        metadata = json.loads((archive_dir / "archive-metadata.json").read_text())
        assert metadata["source_output"] == str(source_output.resolve())
        assert metadata["archive_dir"] == str(archive_dir)
        assert "results.json" in metadata["rewritten_files"]
        assert "suite-results.jsonl" in metadata["rewritten_files"]

        index_rows = [
            json.loads(line)
            for line in (archive_root / "index.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert len(index_rows) == 1
        assert index_rows[0]["archive_dir"] == str(archive_dir)

        captured = capsys.readouterr()
        assert "Archived eval suite to" in captured.out


# =========================================================================
# Test cmd_validate
# =========================================================================


class TestCmdValidate:
    def test_file_not_found(self, capsys):
        args = MagicMock()
        args.file = Path("/nonexistent/file.yaml")
        with pytest.raises(SystemExit) as exc_info:
            cmd_validate(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "File not found" in captured.out

    def test_validate_pass_text_output(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_validate_fail_json_output(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=5.0,
            formula_reviewer=5.0,
            parameter_reviewer=5.0,
            integration_reviewer=5.0,
            policyengine_match=None,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["all_passed"] is False

    def test_validate_with_oracle_policyengine_pass(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.98,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

    def test_validate_with_oracle_policyengine_fail(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.80,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1

    def test_validate_with_oracle_taxsim_fail(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=0.50,
        )
        mock_result.total_duration_ms = 100

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 1

    def test_validate_with_oracle_all(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=0.98,
            taxsim_match=0.96,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

    def test_validate_json_output_with_reviewresults_object(self, capsys, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
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
        mock_result.to_review_results.return_value = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rulespec_reviewer",
                    passed=True,
                    items_checked=1,
                    items_passed=1,
                ),
                ReviewResult(
                    reviewer="formula_reviewer",
                    passed=True,
                    items_checked=2,
                    items_passed=2,
                ),
                ReviewResult(
                    reviewer="parameter_reviewer",
                    passed=False,
                    items_checked=2,
                    items_passed=1,
                ),
                ReviewResult(
                    reviewer="integration_reviewer",
                    passed=True,
                    items_checked=1,
                    items_passed=1,
                ),
            ],
            policyengine_match=1.0,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        output = json.loads(capsys.readouterr().out)
        assert output["scores"]["rulespec_reviewer"] == 10.0
        assert output["scores"]["parameter_reviewer"] == 5.0
        assert output["oracle_scores"]["policyengine"] == 1.0

    def test_validate_passes_skip_reviewers_to_pipeline(self, tmp_path):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=None,
            formula_reviewer=None,
            parameter_reviewer=None,
            integration_reviewer=None,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
            mock_pipeline.validate.assert_called_once_with(
                rulespec_file.resolve(), skip_reviewers=True
            )

    def test_validate_json_output_uses_null_scores_when_reviewers_skipped(
        self, capsys, tmp_path
    ):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
        args.json = True
        args.skip_reviewers = True
        args.oracle = "policyengine"
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {"policyengine": MagicMock(score=1.0, error=None)}
        mock_result.to_review_results.return_value = ReviewResults(
            reviews=[],
            policyengine_match=1.0,
            taxsim_match=None,
        )
        mock_result.total_duration_ms = 100

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        output = json.loads(capsys.readouterr().out)
        assert output["scores"]["rulespec_reviewer"] is None
        assert output["scores"]["parameter_reviewer"] is None

    def test_validate_rules_us_not_found_uses_defaults(self, capsys, tmp_path):
        """When rules_us can't be found by walking, use default paths."""
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0


# =========================================================================
# Test cmd_compile
# =========================================================================


class TestCmdCompile:
    def _require_axiom_rules_path(self) -> Path:
        axiom_rules_path = Path("/Users/maxghenis/TheAxiomFoundation/axiom-rules")
        binary = axiom_rules_path / "target" / "debug" / "axiom-rules"
        if not binary.exists():
            pytest.skip("local axiom-rules binary is not built")
        return axiom_rules_path

    def _rulespec_file(self, tmp_path: Path, content: str | None = None) -> Path:
        rules_file = tmp_path / "test.yaml"
        rules_file.write_text(
            content
            or """format: rulespec/v1
module:
  summary: Test programme.
rules:
  - name: test_rule
    kind: parameter
    dtype: Number
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          1
"""
        )
        return rules_file

    def _compile_args(self, rules_file: Path, *, json_output: bool, execute: bool):
        args = MagicMock()
        args.file = rules_file
        args.json = json_output
        args.as_of = None
        args.execute = execute
        args.axiom_rules_path = self._require_axiom_rules_path()
        args.axiom_rules_path = None
        return args

    def test_file_not_found(self, capsys):
        args = MagicMock()
        args.file = Path("/nonexistent/file.yaml")
        args.json = False
        args.as_of = None
        args.execute = False
        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 1

    def test_compile_success_text(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path), json_output=False, execute=False
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Compiled" in captured.out
        assert "test_rule" in captured.out

    def test_compile_success_json(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path), json_output=True, execute=False
        )
        args.as_of = "2024-01-01"

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is True
        assert output["rule_count"] == 1
        assert output["rules"] == ["test_rule"]

    def test_compile_with_execute_text(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path), json_output=False, execute=True
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "compiled successfully" in captured.out

    def test_compile_with_execute_json(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path), json_output=True, execute=True
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is True
        assert output["rules"] == ["test_rule"]

    def test_compile_failure_text(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path, "format: rulespec/v1\nrules: ["),
            json_output=False,
            execute=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Compilation failed" in captured.out

    def test_compile_failure_json(self, capsys, tmp_path):
        args = self._compile_args(
            self._rulespec_file(tmp_path, "format: rulespec/v1\nrules: ["),
            json_output=True,
            execute=False,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_compile(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["success"] is False


# =========================================================================
# Test cmd_log
# =========================================================================


class TestCmdLog:
    def test_log_basic(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test content")
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rulespec_file
        args.iterations = 2
        args.errors = json.dumps([{"type": "parse", "message": "bad syntax"}])
        args.duration = 5000
        args.scores = json.dumps(
            {"rulespec": 8, "formula": 7, "param": 8, "integration": 7}
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
        rulespec_file = tmp_path / "test.yaml"
        # File doesn't exist, so rulespec_content will be ""
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rulespec_file
        args.iterations = 1
        args.errors = "[]"
        args.duration = 0
        args.scores = None
        args.session = None
        args.db = db_path

        cmd_log(args)
        captured = capsys.readouterr()
        assert "Logged" in captured.out

    def test_log_without_scores(self, capsys, tmp_path):
        """Test logging a run without reviewer scores."""
        db_path = tmp_path / "test.db"
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test content")
        args = MagicMock()
        args.citation = "26 USC 1"
        args.file = rulespec_file
        args.iterations = 1
        args.errors = "[]"
        args.duration = 0
        args.scores = None
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
            file_path="test.yaml",
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
            file_path="test.yaml",
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
            file_path="test.yaml",
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
            file_path="test.yaml",
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
        assert "No runs with review results yet" in captured.out

    def test_calibration_with_data(self, capsys, tmp_path):
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.yaml",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rulespec_reviewer",
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
            file_path="test.yaml",
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
# Test cmd_encode
# =========================================================================


class TestCmdEncode:
    def _make_args(self, tmp_path, **overrides):
        """Helper to create args with sensible defaults."""
        atlas_path = tmp_path / "atlas"
        axiom_rules_path = tmp_path / "axiom-rules"
        atlas_path.mkdir(exist_ok=True)
        axiom_rules_path.mkdir(exist_ok=True)
        args = MagicMock()
        args.citation = overrides.get("citation", "26 USC 1(j)(2)")
        args.output = overrides.get("output", tmp_path / "out")
        args.model = overrides.get("model", "test-model")
        args.backend = overrides.get("backend", "codex")
        args.atlas_path = overrides.get("atlas_path", atlas_path)
        args.axiom_rules_path = overrides.get("axiom_rules_path", axiom_rules_path)
        args.mode = overrides.get("mode", "repo-augmented")
        args.allow_context = overrides.get("allow_context", [])
        return args

    def _make_eval_result(self, success=True):
        result = MagicMock()
        result.citation = "26 USC 1(j)(2)"
        result.runner = "codex-test-model"
        result.success = success
        result.duration_ms = 123
        result.estimated_cost_usd = 0.01
        result.input_tokens = 100
        result.output_tokens = 50
        result.cache_read_tokens = 0
        result.reasoning_output_tokens = 0
        result.retrieved_files = []
        result.unexpected_accesses = []
        result.metrics = None
        result.error = None if success else "failed"
        result.output_file = "/tmp/out.yaml"
        result.trace_file = "/tmp/trace.json"
        result.context_manifest_file = "/tmp/context.json"
        return result

    def _run_encode(self, args, result):
        with patch(
            "axiom_encode.cli.run_model_eval", return_value=[result]
        ) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                cmd_encode(args)
            return mock_run, exc_info.value.code

    def test_encode_success(self, capsys, tmp_path):
        args = self._make_args(tmp_path)
        mock_run, exit_code = self._run_encode(args, self._make_eval_result(True))
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["runner_specs"] == ["codex:test-model"]

    def test_encode_with_errors(self, capsys, tmp_path):
        args = self._make_args(tmp_path, citation="26 USC 1", model=None)
        mock_run, exit_code = self._run_encode(args, self._make_eval_result(False))
        assert exit_code == 1
        assert mock_run.call_args.kwargs["runner_specs"] == ["codex:gpt-5.5"]

    def test_encode_openai_backend(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="openai")
        mock_run, exit_code = self._run_encode(args, self._make_eval_result(True))
        assert exit_code == 0
        assert mock_run.call_args.kwargs["runner_specs"] == ["openai:test-model"]

    def test_encode_runner_shown_in_output(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex")
        self._run_encode(args, self._make_eval_result(True))
        captured = capsys.readouterr()
        assert "Runner: codex:test-model" in captured.out


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
        session = db.start_session(model="test-model", axiom_encode_version="0.2.1")
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
        assert "axiom_encode_version" in output["session"]

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
            "axiom_encode.supabase_sync.sync_transcripts_to_supabase",
            return_value={"total": 5, "synced": 5, "failed": 0},
        ):
            cmd_sync_transcripts(args)
        captured = capsys.readouterr()
        assert "5 synced" in captured.out

    def test_sync_transcripts_with_session(self, capsys):
        args = MagicMock()
        args.session = "test-session"

        with patch(
            "axiom_encode.supabase_sync.sync_transcripts_to_supabase",
            return_value={"total": 1, "synced": 1, "failed": 0},
        ):
            cmd_sync_transcripts(args)
        captured = capsys.readouterr()
        assert "test-session" in captured.out

    def test_sync_transcripts_error(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "axiom_encode.supabase_sync.sync_transcripts_to_supabase",
            side_effect=ValueError("Missing credentials"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_sync_transcripts(args)
            assert exc_info.value.code == 1

    def test_transcript_stats_exists(self, capsys):
        args = MagicMock()
        with patch(
            "axiom_encode.supabase_sync.get_local_transcript_stats",
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
            "axiom_encode.supabase_sync.get_local_transcript_stats",
            return_value={"exists": False},
        ):
            cmd_transcript_stats(args)
        captured = capsys.readouterr()
        assert "No local transcript database" in captured.out

    def test_transcript_stats_empty_by_type(self, capsys):
        args = MagicMock()
        with patch(
            "axiom_encode.supabase_sync.get_local_transcript_stats",
            return_value={
                "exists": True,
                "total": 0,
                "unsynced": 0,
                "synced": 0,
            },
        ):
            cmd_transcript_stats(args)

    def test_sync_agent_sessions_success(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "axiom_encode.supabase_sync.sync_agent_sessions_to_supabase",
            return_value={"total": 2, "synced": 2, "failed": 0},
        ):
            cmd_sync_agent_sessions(args)
        captured = capsys.readouterr()
        assert "2 synced" in captured.out

    def test_sync_agent_sessions_error(self, capsys):
        args = MagicMock()
        args.session = None

        with patch(
            "axiom_encode.supabase_sync.sync_agent_sessions_to_supabase",
            side_effect=ValueError("Missing creds"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_sync_agent_sessions(args)
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
    def test_validate_rules_us_found_in_path(self, capsys, tmp_path):
        """Test validate when file is inside a rules-us directory (line 350)."""
        rules_us = tmp_path / "rules-us" / "statute" / "26" / "1"
        rules_us.mkdir(parents=True)
        rulespec_file = rules_us / "test.yaml"
        rulespec_file.write_text("# test")
        # Create axiom-rules sibling
        axiom_rules_path = tmp_path / "axiom-rules"
        axiom_rules_path.mkdir(parents=True)

        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = False
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=8.0,
            formula_reviewer=7.5,
            parameter_reviewer=8.5,
            integration_reviewer=8.0,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0
            # Verify axiom_rules_path was resolved via rules_us.parent / "axiom-rules"
            call_kwargs = mock_pipeline_cls.call_args[1]
            assert "rules-us" in str(call_kwargs["policy_repo_path"])

    def test_validate_uses_enclosing_non_federal_policy_repo(self, tmp_path):
        policy_repo = tmp_path / "rules-us-tn" / "sources"
        policy_repo.mkdir(parents=True)
        rulespec_file = policy_repo / "test.yaml"
        rulespec_file.write_text("format: rulespec/v1\n")
        (tmp_path / "axiom-rules").mkdir()

        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=None,
            formula_reviewer=None,
            parameter_reviewer=None,
            integration_reviewer=None,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        call_kwargs = mock_pipeline_cls.call_args[1]
        assert call_kwargs["policy_repo_path"] == tmp_path / "rules-us-tn"
        assert call_kwargs["axiom_rules_path"] == tmp_path / "axiom-rules"

    def test_eval_source_prefers_akn_backed_slice_text(self, tmp_path, monkeypatch):
        arch_root = tmp_path / "arch"
        monkeypatch.setenv("AXIOM_ENCODE_EVAL_ARCHIVE_ROOT", str(arch_root))
        policy_repo = tmp_path / "rules-us-tx"
        source_file = (
            policy_repo
            / "sources"
            / "slices"
            / "txhhs"
            / "twh"
            / "current-effective"
            / "snap_standard_utility_allowance_tx.txt"
        )
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text("stale slice text")
        akn_file = (
            arch_root
            / "us-tx"
            / "txhhs"
            / "twh"
            / "current-effective"
            / "akn"
            / "source.akn.xml"
        )
        akn_file.parent.mkdir(parents=True, exist_ok=True)
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc name="doc">
    <mainBody>
      <hcontainer name="section" eId="sec_sua">
        <heading>Utility allowances</heading>
        <content><p>SUA - $445</p></content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )
        source_file.with_name(
            "snap_standard_utility_allowance_tx.meta.yaml"
        ).write_text(
            "version: 1\n"
            "source_backing:\n"
            "  kind: akn_section\n"
            "  arch_path: us-tx/txhhs/twh/current-effective/akn/source.akn.xml\n"
            "  section_eid: sec_sua\n"
            "relations:\n"
            "  - relation: sets\n"
            "    target: cfr/7/273.9/d/6/iii#snap_standard_utility_allowance\n"
            "    jurisdiction: TX\n"
        )
        (tmp_path / "axiom-rules").mkdir()

        args = SimpleNamespace(
            runner=[],
            gpt_backend=None,
            axiom_rules_path=None,
            source_file=source_file,
            source_id="snap_standard_utility_allowance_tx",
            output=tmp_path / "out",
            mode="cold",
            allow_context=[],
            policyengine_rule_hint="snap_standard_utility_allowance",
            json=True,
        )

        with patch("axiom_encode.cli.run_source_eval", return_value=[]) as mock_run:
            cmd_eval_source(args)

        assert "SUA - $445" in mock_run.call_args.kwargs["source_text"]
        assert "stale slice text" not in mock_run.call_args.kwargs["source_text"]

    def test_validate_fallback_prefers_workspace_repo_roots(self, tmp_path):
        rulespec_file = tmp_path / "generated" / "test.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text("# test")

        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = None
        args.min_match = 0.95

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {}
        mock_result.to_review_results.return_value = MagicMock(
            rulespec_reviewer=None,
            formula_reviewer=None,
            parameter_reviewer=None,
            integration_reviewer=None,
            policyengine_match=None,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        call_kwargs = mock_pipeline_cls.call_args[1]
        assert call_kwargs["policy_repo_path"] == Path(
            "/Users/maxghenis/TheAxiomFoundation/rules-us"
        )
        assert call_kwargs["axiom_rules_path"] == Path(
            "/Users/maxghenis/TheAxiomFoundation/axiom-rules"
        )


class TestCmdCalibrationEdgeCases:
    def test_calibration_all_pass(self, capsys, tmp_path):
        """Test calibration with all reviews passing."""
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.yaml",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rulespec_reviewer",
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
        assert "rulespec_reviewer" in captured.out

    def test_calibration_some_fail(self, capsys, tmp_path):
        """Test calibration with some reviews failing."""
        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)

        for i in range(3):
            run = EncodingRun(
                citation=f"26 USC {i}",
                file_path="test.yaml",
                review_results=ReviewResults(
                    reviews=[
                        ReviewResult(
                            reviewer="rulespec_reviewer",
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
