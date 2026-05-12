"""
Tests for axiom_encode CLI (cli.py).

Tests all CLI commands using subprocess invocation and direct function calls.
All external dependencies are mocked.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_SIGNING_KEY_ENV,
    _apply_generated_encoding_result,
    _discover_rulespec_test_files,
    _effective_runner_specs,
    _find_rulespec_dependents,
    _insert_false_input_default,
    _rewrite_gpt_runner_backend,
    _sha256_file,
    _sign_applied_encoding_manifest,
    _source_relation_preservation_issues,
    _validate_generated_encoding_in_policy_overlay,
    cmd_calibration,
    cmd_compile,
    cmd_encode,
    cmd_eval_suite,
    cmd_eval_suite_archive,
    cmd_eval_suite_report,
    cmd_eval_suite_revalidate,
    cmd_guard_generated,
    cmd_inventory,
    cmd_log,
    cmd_log_event,
    cmd_oracle_candidates,
    cmd_oracle_coverage,
    cmd_repair_nonnegative_floors,
    cmd_repair_zero_branch_tests,
    cmd_runs,
    cmd_session_end,
    cmd_session_show,
    cmd_session_start,
    cmd_session_stats,
    cmd_sessions,
    cmd_stats,
    cmd_sync_agent_sessions,
    cmd_sync_transcripts,
    cmd_test,
    cmd_transcript_stats,
    cmd_validate,
    guard_generated_change_issues,
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
from axiom_encode.statute import citation_to_citation_path, parse_usc_citation

TEST_APPLY_SIGNING_KEY = "test-apply-signing-key"


def _signed_manifest_payload(payload: dict) -> dict:
    _sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_KEY)
    return payload


# =========================================================================
# Test main() dispatch
# =========================================================================


class TestStatuteCitationPaths:
    def test_builds_clean_citation_path_from_us_code_citation(self):
        assert citation_to_citation_path("26 USC 3101(a)") == "us/statute/26/3101/a"

    def test_parses_absolute_rulespec_statute_path(self):
        parts = parse_usc_citation("us:statutes/26/3101/a")

        assert (parts.title, parts.section, parts.fragments) == (
            "26",
            "3101",
            ("a",),
        )


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

    def test_inventory_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "inventory"]):
            with patch("axiom_encode.cli.cmd_inventory") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_oracle_coverage_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "oracle-coverage"]):
            with patch("axiom_encode.cli.cmd_oracle_coverage") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_oracle_candidates_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "oracle-candidates"]):
            with patch("axiom_encode.cli.cmd_oracle_candidates") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_calibration_command_dispatches(self):
        with patch("sys.argv", ["axiom_encode", "calibration"]):
            with patch("axiom_encode.cli.cmd_calibration") as mock_cmd:
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
        with patch(
            "sys.argv",
            ["axiom_encode", "eval-source", "us-co/regulation/9-ccr-2503-6/3.606.1/F"],
        ):
            with patch("axiom_encode.cli.cmd_eval_source") as mock_cmd:
                main()
                mock_cmd.assert_called_once()

    def test_eval_source_accepts_policyengine_rule_hint(self):
        with patch(
            "sys.argv",
            [
                "axiom_encode",
                "eval-source",
                "us/statute/7/2014/e/2/B",
                "--policyengine-rule-hint",
                "snap_earned_income_deduction",
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
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    corpus_citation_path: us/statute/7/2017\n"
        )
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            corpus_path=tmp_path / "axiom-corpus",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=0,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()
        args.corpus_path.mkdir()

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
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    corpus_citation_path: us/statute/7/2017\n"
        )
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            corpus_path=tmp_path / "axiom-corpus",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            json=False,
            gpt_backend="codex",
            resume=True,
            auto_resume_attempts=0,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()
        args.corpus_path.mkdir()

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
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    corpus_citation_path: us/statute/7/2017\n"
        )
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            corpus_path=tmp_path / "axiom-corpus",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=1,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()
        args.corpus_path.mkdir()

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
            "name: readiness\ncases:\n  - kind: source\n    source_id: x\n    corpus_citation_path: us/statute/7/2017\n"
        )
        args = SimpleNamespace(
            manifest=manifest_file,
            runner=None,
            output=tmp_path / "out",
            corpus_path=tmp_path / "axiom-corpus",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            json=False,
            gpt_backend="codex",
            resume=False,
            auto_resume_attempts=2,
            auto_resume_delay_seconds=0,
        )
        args.axiom_rules_path.mkdir()
        args.corpus_path.mkdir()

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
                    "    corpus_citation_path: us/statute/7/2014/e/6/A",
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
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            corpus_path=tmp_path / "axiom-corpus",
            json=True,
        )
        args.axiom_rules_path.mkdir()
        args.corpus_path.mkdir()

        with (
            patch(
                "axiom_encode.cli.resolve_corpus_source_unit",
                return_value=SimpleNamespace(body="authoritative source text"),
            ),
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
            assert mock_pipeline_cls.call_args.kwargs["oracle_validators"] is None
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
            assert mock_pipeline_cls.call_args.kwargs["oracle_validators"] is None
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
            assert mock_pipeline_cls.call_args.kwargs["oracle_validators"] == (
                "policyengine",
            )

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
            assert mock_pipeline_cls.call_args.kwargs["oracle_validators"] == (
                "policyengine",
            )

    def test_validate_with_oracle_classification_required_fails_unmapped(
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
        args.require_oracle_classification = True

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(
                score=1.0,
                error=None,
                details={
                    "coverage": {
                        "comparable": 1,
                        "passed": 1,
                        "failed": 0,
                        "unmapped": 2,
                        "unsupported": 0,
                    }
                },
            ),
        }
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
            assert exc_info.value.code == 1

        output = json.loads(capsys.readouterr().out)
        assert output["oracle_passed"] is False
        assert "policyengine: 2 unclassified oracle output(s)" in output["errors"]

    def test_validate_with_oracle_classification_required_allows_classified(
        self, capsys, tmp_path
    ):
        rulespec_file = tmp_path / "test.yaml"
        rulespec_file.write_text("# test")
        args = MagicMock()
        args.file = rulespec_file
        args.json = False
        args.skip_reviewers = True
        args.oracle = "policyengine"
        args.min_match = 0.95
        args.require_oracle_classification = True

        mock_result = MagicMock()
        mock_result.all_passed = True
        mock_result.ci_pass = True
        mock_result.results = {
            "policyengine": MagicMock(
                score=1.0,
                error=None,
                details={
                    "coverage": {
                        "comparable": 1,
                        "passed": 1,
                        "failed": 0,
                        "unmapped": 0,
                        "unsupported": 2,
                    }
                },
            ),
        }
        mock_result.to_review_results.return_value = ReviewResults(
            reviews=[],
            policyengine_match=1.0,
            taxsim_match=None,
        )

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls:
            mock_pipeline_cls.return_value.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        assert "unmapped=0 unsupported=2" in capsys.readouterr().out

    def test_oracle_coverage_fail_on_unmapped_exits_nonzero(self, capsys, tmp_path):
        args = MagicMock()
        args.root = tmp_path
        args.oracle = "policyengine"
        args.program = None
        args.limit = 25
        args.fail_on_unmapped = True
        args.fail_on_untested_comparable = False
        args.json = True

        with patch(
            "axiom_encode.cli.build_policyengine_coverage_report",
            return_value={
                "oracle": "policyengine",
                "root": str(tmp_path),
                "total_outputs": 1,
                "status_counts": {"unmapped": 1},
                "untested_comparable": 0,
                "program_counts": {"snap": 1},
                "repos": [],
                "items": [
                    {
                        "legal_id": "us:statutes/7/9999#snap_new_output",
                        "status": "unmapped",
                    }
                ],
            },
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_oracle_coverage(args)

        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert output["status_counts"]["unmapped"] == 1

    def test_oracle_coverage_fail_on_untested_comparable_exits_nonzero(
        self, capsys, tmp_path
    ):
        args = MagicMock()
        args.root = tmp_path
        args.oracle = "policyengine"
        args.program = None
        args.limit = 25
        args.fail_on_unmapped = False
        args.fail_on_untested_comparable = True
        args.json = False

        with patch(
            "axiom_encode.cli.build_policyengine_coverage_report",
            return_value={
                "oracle": "policyengine",
                "root": str(tmp_path),
                "total_outputs": 1,
                "status_counts": {"comparable": 1},
                "untested_comparable": 1,
                "program_counts": {"tax": 1},
                "repos": [],
                "items": [
                    {
                        "legal_id": "us:statutes/26/3101/a#oasdi_wage_tax_rate",
                        "status": "comparable",
                        "tested": False,
                    }
                ],
            },
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_oracle_coverage(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "Untested comparable outputs: 1" in output
        assert "us:statutes/26/3101/a#oasdi_wage_tax_rate" in output

    def test_oracle_candidates_prints_priority_queue(self, capsys, tmp_path):
        args = MagicMock()
        args.root = tmp_path
        args.oracle = "policyengine"
        args.program = "snap"
        args.limit = 1
        args.json = False

        with patch(
            "axiom_encode.cli.build_policyengine_candidate_report",
            return_value={
                "oracle": "policyengine",
                "root": str(tmp_path),
                "program": "snap",
                "policyengine_variables_available": True,
                "total_candidates": 1,
                "category_counts": {"exact_variable_unmapped": 1},
                "priority_counts": {"P1": 1},
                "coverage_status_counts": {"unmapped": 1},
                "items": [
                    {
                        "legal_id": "us:statutes/7/9999#snap_new_exact_variable",
                        "category": "exact_variable_unmapped",
                        "priority": "P1",
                        "recommendation": "Review mapping.",
                        "policyengine_variable": "snap_new_exact_variable",
                        "policyengine_parameter": None,
                        "tested": True,
                        "rationale": None,
                    }
                ],
            },
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_oracle_candidates(args)

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "PolicyEngine oracle candidates" in output
        assert "[P1] exact_variable_unmapped" in output
        assert "snap_new_exact_variable" in output

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
            assert mock_pipeline_cls.call_args.kwargs["oracle_validators"] is None

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
        axiom_rules_path = Path(
            "/Users/maxghenis/TheAxiomFoundation/axiom-rules-engine"
        )
        binary = axiom_rules_path / "target" / "debug" / "axiom-rules-engine"
        if not binary.exists():
            pytest.skip("local axiom-rules-engine binary is not built")
        return axiom_rules_path

    def _rulespec_file(self, tmp_path: Path, content: str | None = None) -> Path:
        rules_file = tmp_path / "test.yaml"
        rules_file.write_text(
            content
            or """format: rulespec/v1
module:
  summary: Test RuleSpec file.
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
# Test cmd_test
# =========================================================================


class TestCmdTest:
    def _require_axiom_rules_path(self) -> Path:
        axiom_rules_path = Path(
            "/Users/maxghenis/TheAxiomFoundation/axiom-rules-engine"
        )
        binary = axiom_rules_path / "target" / "debug" / "axiom-rules-engine"
        if not binary.exists():
            pytest.skip("local axiom-rules-engine binary is not built")
        return axiom_rules_path

    def _write_rulespec_with_test(
        self,
        tmp_path: Path,
        *,
        expected_benefit: int,
    ) -> Path:
        repo = tmp_path / "rulespec-us"
        target = repo / "statutes/1/1.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  summary: Test companion execution.
rules:
  - name: threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: '10'
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: income + threshold
"""
        )
        target.with_name("1.test.yaml").write_text(
            f"""- name: computes_derived_and_parameter_output
  period: 2026-01
  input:
    us:statutes/1/1#input.income: 5
  output:
    us:statutes/1/1#threshold: 10
    us:statutes/1/1#benefit: {expected_benefit}
"""
        )
        return repo

    def _args(self, repo: Path, *, json_output: bool):
        args = MagicMock()
        args.root = repo
        args.paths = []
        args.json = json_output
        args.axiom_rules_path = self._require_axiom_rules_path()
        return args

    def test_executes_companion_tests_success_json(self, capsys, tmp_path):
        repo = self._write_rulespec_with_test(tmp_path, expected_benefit=15)

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(self._args(repo, json_output=True))

        assert exc_info.value.code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True
        assert output["test_files"] == 1
        assert output["cases"] == 1
        assert output["failures"] == []

    def test_executes_companion_tests_failure_json(self, capsys, tmp_path):
        repo = self._write_rulespec_with_test(tmp_path, expected_benefit=16)

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(self._args(repo, json_output=True))

        assert exc_info.value.code == 1
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is False
        assert "expected 16" in output["failures"][0]["message"]

    def test_executes_companion_tests_prefers_current_repo_over_env_root(
        self, capsys, monkeypatch, tmp_path
    ):
        stale_repo = tmp_path / "canonical" / "rulespec-us"
        stale_child = stale_repo / "statutes/1/child.yaml"
        stale_child.parent.mkdir(parents=True)
        stale_child.write_text(
            """format: rulespec/v1
rules:
  - name: child_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: stale_income + 1
"""
        )

        repo = tmp_path / "workspace" / "rulespec-us"
        child = repo / "statutes/1/child.yaml"
        child.parent.mkdir(parents=True)
        child.write_text(
            """format: rulespec/v1
rules:
  - name: child_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: current_income + 1
"""
        )
        parent = repo / "statutes/1/parent.yaml"
        parent.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/1/child
rules:
  - name: total_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: child_benefit + 1
"""
        )
        parent.with_name("parent.test.yaml").write_text(
            """- name: imported_current_child_input
  period: 2026-01
  input:
    us:statutes/1/child#input.current_income: 5
  output:
    us:statutes/1/parent#total_benefit: 7
"""
        )
        monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_repo))

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(self._args(repo, json_output=True))

        assert exc_info.value.code == 0
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True

    def test_executes_companion_tests_passes_rulespec_env_to_engine(
        self, capsys, monkeypatch, tmp_path
    ):
        repo = tmp_path / "workspace" / "rulespec-us"
        rules_file = repo / "statutes/1/1.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
rules:
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: income + 1
"""
        )
        rules_file.with_name("1.test.yaml").write_text(
            """- name: computes_benefit
  period: 2026-01
  input:
    us:statutes/1/1#input.income: 5
  output:
    us:statutes/1/1#benefit: 6
"""
        )
        stale_repo = tmp_path / "stale-rulespec-us"
        stale_repo.mkdir()
        monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_repo))

        engine_root = tmp_path / "axiom-rules-engine"
        binary = engine_root / "target/debug/axiom-rules-engine"
        captured_envs: list[dict[str, str] | None] = []

        def fake_binary(self):
            return binary

        def fake_run(cmd, **kwargs):
            captured_envs.append(kwargs.get("env"))
            if "compile" in cmd:
                output_path = Path(cmd[cmd.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "program": {
                                "parameters": [],
                                "derived": [
                                    {
                                        "id": "us:statutes/1/1#benefit",
                                        "name": "benefit",
                                    }
                                ],
                            }
                        }
                    )
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "run-compiled" in cmd:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=json.dumps(
                        {
                            "results": [
                                {
                                    "outputs": {
                                        "us:statutes/1/1#benefit": {
                                            "kind": "scalar",
                                            "value": {"kind": "integer", "value": 6},
                                        }
                                    }
                                }
                            ]
                        }
                    ),
                    stderr="",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        args = MagicMock()
        args.root = repo
        args.paths = []
        args.json = True
        args.axiom_rules_path = engine_root

        monkeypatch.setattr(
            "axiom_encode.harness.validator_pipeline.ValidatorPipeline._axiom_rules_binary",
            fake_binary,
        )
        monkeypatch.setattr("axiom_encode.cli.subprocess.run", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(args)

        assert exc_info.value.code == 0
        assert json.loads(capsys.readouterr().out)["success"] is True
        assert len(captured_envs) == 2
        for env in captured_envs:
            assert env is not None
            roots = env["AXIOM_RULESPEC_REPO_ROOTS"].split(os.pathsep)
            assert roots[:2] == [str(repo.resolve()), str(repo.resolve().parent)]
            assert str(stale_repo) in roots

    def test_discovery_skips_axiom_dependency_tree(self, tmp_path):
        root = tmp_path / "workspace"
        valid_test = root / "statutes/1/1.test.yaml"
        vendored_test = root / "_axiom/axiom-rules-engine/tests/fixtures/bad.test.yaml"
        sibling_fixture_test = root / "axiom-rules-engine/tests/fixtures/bad.test.yaml"
        valid_test.parent.mkdir(parents=True)
        vendored_test.parent.mkdir(parents=True)
        sibling_fixture_test.parent.mkdir(parents=True)
        valid_test.write_text("[]\n")
        vendored_test.write_text("[]\n")
        sibling_fixture_test.write_text("[]\n")

        assert _discover_rulespec_test_files([], root=root) == [valid_test]


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
# Test cmd_inventory
# =========================================================================


class TestCmdInventory:
    def test_inventory_counts_rulespec_files_and_kinds(self, capsys, tmp_path):
        statute_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
        statute_file.parent.mkdir(parents=True)
        statute_file.write_text(
            """
format: rulespec/v1
rules:
  - name: regular_allotment
    kind: derived
  - name: maximum_allotment
    kind: parameter
"""
        )
        test_file = statute_file.with_name("a.test.yaml")
        test_file.write_text(
            """
cases:
  - name: ignored
"""
        )

        composition_file = (
            tmp_path
            / "rulespec-us-co"
            / "policies"
            / "cdhs"
            / "snap"
            / "fy-2026-benefit-calculation.yaml"
        )
        composition_file.parent.mkdir(parents=True)
        composition_file.write_text(
            """
format: rulespec/v1
module:
  summary: Colorado SNAP FY 2026 benefit calculation composition.
rules:
  - name: final_allotment
    kind: derived
"""
        )

        relation_file = (
            tmp_path
            / "rulespec-us-co"
            / "regulations"
            / "10-ccr-2506-1"
            / "4.407.1.yaml"
        )
        relation_file.parent.mkdir(parents=True)
        relation_file.write_text(
            """
format: rulespec/v1
rules:
  - name: co_standard_deduction_restates_usda
    kind: source_relation
"""
        )

        args = SimpleNamespace(root=tmp_path, json=True)

        cmd_inventory(args)

        output = json.loads(capsys.readouterr().out)
        assert output["total_files"] == 3
        assert output["source_provision_files"] == 2
        assert output["composition_files"] == 1
        assert output["total_rules"] == 4
        assert output["kind_counts"] == {
            "derived": 2,
            "parameter": 1,
            "source_relation": 1,
        }
        assert output["repos"] == [
            {
                "repo": "rulespec-us",
                "files": 1,
                "source_provision_files": 1,
                "composition_files": 0,
                "rules": 2,
                "roots": {"statutes": 1},
                "kinds": {"derived": 1, "parameter": 1},
            },
            {
                "repo": "rulespec-us-co",
                "files": 2,
                "source_provision_files": 1,
                "composition_files": 1,
                "rules": 2,
                "roots": {"policies": 1, "regulations": 1},
                "kinds": {"derived": 1, "source_relation": 1},
            },
        ]

    def test_inventory_prints_human_summary(self, capsys, tmp_path):
        rulespec_file = tmp_path / "rulespec-us" / "policies" / "usda" / "cola.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text(
            """
format: rulespec/v1
rules:
  - name: snap_maximum_allotment
    kind: parameter
"""
        )
        args = SimpleNamespace(root=tmp_path, json=False)

        cmd_inventory(args)

        output = capsys.readouterr().out
        assert "RuleSpec inventory" in output
        assert "Files: 1 total; 1 source/provision; 0 composition" in output
        assert "Kinds: parameter=1" in output
        assert "rulespec-us: files=1" in output


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
        corpus_path = tmp_path / "axiom-corpus"
        axiom_rules_path = tmp_path / "axiom-rules-engine"
        policy_repo_path = tmp_path / "rulespec-us"
        corpus_path.mkdir(exist_ok=True)
        axiom_rules_path.mkdir(exist_ok=True)
        policy_repo_path.mkdir(exist_ok=True)
        args = MagicMock()
        args.citation = overrides.get("citation", "26 USC 1(j)(2)")
        args.source_id = overrides.get("source_id", None)
        args.output = overrides.get("output", tmp_path / "out")
        args.model = overrides.get("model", "test-model")
        args.backend = overrides.get("backend", "codex")
        args.corpus_path = overrides.get("corpus_path", corpus_path)
        args.axiom_rules_path = overrides.get("axiom_rules_path", axiom_rules_path)
        args.policy_repo_path = overrides.get("policy_repo_path", policy_repo_path)
        args.mode = overrides.get("mode", "repo-augmented")
        args.allow_context = overrides.get("allow_context", [])
        args.db = overrides.get("db", tmp_path / "encodings.db")
        args.sync = overrides.get("sync", True)
        args.apply = overrides.get("apply", False)
        args.apply_target_only = overrides.get("apply_target_only", False)
        return args

    def _make_eval_result(self, success=True):
        result = MagicMock()
        result.citation = "26 USC 1(j)(2)"
        result.runner = "codex-test-model"
        result.backend = "codex"
        result.model = "test-model"
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
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(SystemExit) as exc_info:
                    cmd_encode(args)
            return mock_run, exc_info.value.code

    def test_encode_success(self, capsys, tmp_path):
        args = self._make_args(tmp_path)
        mock_run, exit_code = self._run_encode(args, self._make_eval_result(True))
        assert exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["runner_specs"] == ["codex:test-model"]
        assert mock_run.call_args.kwargs["include_tests"] is True
        assert mock_run.call_args.kwargs["policy_path"] == args.policy_repo_path
        assert (
            mock_run.call_args.kwargs["runtime_axiom_rules_path"]
            == args.axiom_rules_path
        )

    def test_encode_with_source_id_uses_corpus_source_unit(self, capsys, tmp_path):
        args = self._make_args(
            tmp_path,
            citation="us/guidance/irs/rev-proc-2025-32/page-18",
            source_id="us/policies/irs/rev-proc-2025-32/standard-deduction",
            sync=False,
        )
        result = self._make_eval_result(True)
        result.citation = args.source_id

        with (
            patch(
                "axiom_encode.cli.resolve_corpus_source_unit",
                return_value=SimpleNamespace(
                    body="standard deduction source text",
                    citation_path="us/guidance/irs/rev-proc-2025-32/page-18",
                    source="local",
                    requested="us/guidance/irs/rev-proc-2025-32/page-18",
                ),
            ) as mock_resolve,
            patch(
                "axiom_encode.cli.run_source_eval", return_value=[result]
            ) as mock_run_source,
            patch("axiom_encode.cli.run_model_eval") as mock_run_model,
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 0
        mock_resolve.assert_called_once_with(args.citation, args.corpus_path)
        mock_run_model.assert_not_called()
        assert mock_run_source.call_args.kwargs["source_id"] == args.source_id
        assert (
            mock_run_source.call_args.kwargs["source_text"]
            == "standard deduction source text"
        )
        assert mock_run_source.call_args.kwargs["runner_specs"] == ["codex:test-model"]
        assert mock_run_source.call_args.kwargs["policy_path"] == args.policy_repo_path
        assert (
            mock_run_source.call_args.kwargs["runtime_axiom_rules_path"]
            == args.axiom_rules_path
        )
        assert mock_run_source.call_args.kwargs["source_metadata_payload"] == {
            "corpus_citation_path": "us/guidance/irs/rev-proc-2025-32/page-18",
            "corpus_source": "local",
            "requested_source": "us/guidance/irs/rev-proc-2025-32/page-18",
        }
        output = capsys.readouterr().out
        assert (
            "RuleSpec source id: us/policies/irs/rev-proc-2025-32/standard-deduction"
            in output
        )

    def test_encode_with_errors(self, capsys, tmp_path):
        args = self._make_args(tmp_path, citation="26 USC 1", model=None)
        result = self._make_eval_result(False)
        result.citation = "26 USC 1"
        result.output_file = str(tmp_path / "out.yaml")
        result.trace_file = str(tmp_path / "trace.json")
        result.context_manifest_file = str(tmp_path / "context.json")
        mock_run, exit_code = self._run_encode(args, result)
        assert exit_code == 1
        assert mock_run.call_args.kwargs["runner_specs"] == ["codex:gpt-5.5"]
        repair_manifest = tmp_path / "out.repair.json"
        assert repair_manifest.exists()
        assert json.loads(repair_manifest.read_text())["citation"] == "26 USC 1"

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

    def test_encode_logs_completed_run(self, tmp_path):
        args = self._make_args(tmp_path, backend="codex")
        result = self._make_eval_result(True)
        output_file = tmp_path / "out.yaml"
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        result.output_file = str(output_file)

        _, exit_code = self._run_encode(args, result)

        assert exit_code == 0
        runs = EncodingDB(args.db).get_recent_runs(limit=1)
        assert len(runs) == 1
        assert runs[0].citation == "26 USC 1(j)(2)"
        assert runs[0].rulespec_content.startswith("format: rulespec/v1")
        assert runs[0].session_id is not None
        session = EncodingDB(args.db).get_session(runs[0].session_id)
        assert session is not None
        assert session.run_id == runs[0].id
        assert session.total_tokens == 150
        assert session.event_count == 3
        assert runs[0].outcome["status"] == "standalone_validated"

    def test_encode_syncs_when_credentials_are_configured(self, tmp_path):
        args = self._make_args(tmp_path, backend="codex")
        result = self._make_eval_result(True)

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch.dict(
                os.environ,
                {
                    "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                    "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "secret",
                },
                clear=True,
            ),
            patch(
                "axiom_encode.supabase_sync.sync_run_to_supabase",
                return_value=True,
            ) as mock_sync,
            patch(
                "axiom_encode.supabase_sync.sync_agent_sessions_to_supabase",
                return_value={"total": 1, "synced": 1, "failed": 0},
            ) as mock_session_sync,
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_encode(args)

        assert exc_info.value.code == 0
        mock_sync.assert_called_once()
        synced_run = mock_sync.call_args.args[0]
        assert synced_run.citation == "26 USC 1(j)(2)"
        assert synced_run.session_id is not None
        assert mock_sync.call_args.args[1] == "reviewer_agent"
        mock_session_sync.assert_called_once_with(
            session_id=synced_run.session_id,
            db_path=args.db,
        )

    def test_apply_generated_encoding_writes_manifest(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us-ny"
        generated = (
            output_root
            / "codex-test-model"
            / "regulations"
            / "18-nycrr"
            / "387"
            / "12"
            / "f"
            / "3"
            / "v"
            / "c.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated.with_name("c.test.yaml").write_text("[]\n")
        policy_repo.mkdir()
        result = self._make_eval_result(True)
        result.output_file = str(generated)
        result.context_manifest_file = str(tmp_path / "context.json")
        result.trace_file = str(tmp_path / "trace.json")
        result.generation_prompt_sha256 = "prompt-sha"
        Path(result.context_manifest_file).write_text("{}\n")
        Path(result.trace_file).write_text("{}\n")

        with (
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
            patch(
                "axiom_encode.cli._git_repo_provenance",
                return_value={
                    "root": "/repo/axiom-encode",
                    "commit": "abc123",
                    "dirty_tracked": False,
                },
            ),
        ):
            applied = _apply_generated_encoding_result(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                run_id="run-123",
            )

        target = policy_repo / "regulations/18-nycrr/387/12/f/3/v/c.yaml"
        target_test = policy_repo / "regulations/18-nycrr/387/12/f/3/v/c.test.yaml"
        manifest = (
            policy_repo
            / ".axiom/encoding-manifests/regulations/18-nycrr/387/12/f/3/v/c.json"
        )
        assert applied == [target, target_test, manifest]
        payload = json.loads(manifest.read_text())
        assert payload["schema_version"] == APPLIED_ENCODING_MANIFEST_SCHEMA
        assert payload["tool"] == "axiom-encode encode --apply"
        assert payload["run_id"] == "run-123"
        assert payload["generation_prompt_sha256"] == "prompt-sha"
        assert payload["axiom_encode_git"] == {
            "root": "/repo/axiom-encode",
            "commit": "abc123",
            "dirty_tracked": False,
        }
        assert payload["signature"]["algorithm"] == "hmac-sha256"
        assert payload["applied_files"] == [
            {
                "path": "regulations/18-nycrr/387/12/f/3/v/c.yaml",
                "sha256": _sha256_file(target),
            },
            {
                "path": "regulations/18-nycrr/387/12/f/3/v/c.test.yaml",
                "sha256": _sha256_file(target_test),
            },
        ]

    def test_apply_generated_encoding_rejects_dirty_encoder_build(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes" / "26" / "25B.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        policy_repo.mkdir()
        result = self._make_eval_result(True)
        result.output_file = str(generated)

        with (
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
            patch(
                "axiom_encode.cli._git_repo_provenance",
                return_value={
                    "root": "/repo/axiom-encode",
                    "commit": "abc123",
                    "dirty_tracked": True,
                },
            ),
            pytest.raises(RuntimeError, match="dirty axiom-encode checkout"),
        ):
            _apply_generated_encoding_result(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
            )

        assert not (policy_repo / "statutes/26/25B.yaml").exists()

    def test_apply_generated_encoding_rejects_unknown_encoder_build(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes" / "26" / "25B.yaml"
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        policy_repo.mkdir()
        result = self._make_eval_result(True)
        result.output_file = str(generated)

        with (
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
            patch("axiom_encode.cli._git_repo_provenance", return_value=None),
            pytest.raises(RuntimeError, match="git provenance is unavailable"),
        ):
            _apply_generated_encoding_result(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
            )

        assert not (policy_repo / "statutes/26/25B.yaml").exists()

    def test_apply_generated_encoding_requires_signing_key(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us-ny"
        generated = (
            output_root
            / "codex-test-model"
            / "regulations"
            / "18-nycrr"
            / "387"
            / "12"
            / "f"
            / "3"
            / "v"
            / "c.yaml"
        )
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        policy_repo.mkdir()
        result = self._make_eval_result(True)
        result.output_file = str(generated)

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match=APPLIED_ENCODING_SIGNING_KEY_ENV),
        ):
            _apply_generated_encoding_result(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
            )

    def test_encode_apply_exits_nonzero_when_overlay_validation_blocks(self, tmp_path):
        args = self._make_args(tmp_path, backend="codex")
        args.apply = True
        result = self._make_eval_result(True)

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                return_value=(False, ["dependent failed"], {}),
            ),
            patch("axiom_encode.cli._apply_generated_encoding_result") as mock_apply,
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 1
        mock_apply.assert_not_called()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["status"] == "apply_blocked_validation"
        assert run.outcome["final_success"] is False
        assert run.outcome["apply_error"] == "dependent failed"
        events = EncodingDB(args.db).get_session_events(run.session_id)
        assert [event.event_type for event in events] == [
            "encode_request",
            "encode_result",
            "encode_outcome",
            "encode_issue",
        ]

    def test_encode_apply_allows_overlay_to_rescue_failed_standalone_validation(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = tmp_path / "out" / "codex-test-model" / "regulations/example.yaml"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "regulations/example.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                return_value=(True, [], {}),
            ) as mock_overlay,
            patch(
                "axiom_encode.cli._apply_generated_encoding_result",
                return_value=[applied_file],
            ) as mock_apply,
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "standalone_success=False" in output
        assert "outcome=apply_applied final_success=True" in output
        mock_overlay.assert_called_once()
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.iterations[0].success is False
        assert run.outcome["standalone_validation_success"] is False
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["apply_success"] is True
        assert run.outcome["final_success"] is True
        assert run.outcome["status"] == "apply_applied"
        assert run.success is True
        assert not output_file.with_suffix(".repair.json").exists()
        events = EncodingDB(args.db).get_session_events(run.session_id)
        assert [event.event_type for event in events] == [
            "encode_request",
            "encode_result",
            "encode_outcome",
        ]

    def test_encode_apply_blocks_non_validation_generation_failure(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "backend timed out"
        output_file = tmp_path / "out" / "codex-test-model" / "regulations/example.yaml"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        result.output_file = str(output_file)

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                return_value=(True, [], {}),
            ) as mock_overlay,
            patch("axiom_encode.cli._apply_generated_encoding_result") as mock_apply,
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "apply=blocked_generation:backend timed out" in output
        mock_overlay.assert_not_called()
        mock_apply.assert_not_called()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["status"] == "apply_blocked_generation"
        assert run.outcome["overlay_validation_success"] is None
        assert run.outcome["final_success"] is False
        assert run.success is False
        assert output_file.with_suffix(".repair.json").exists()

    def test_encode_apply_blocks_failed_overlay_after_standalone_validation_failure(
        self, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed compile validation"
        output_file = tmp_path / "out" / "codex-test-model" / "regulations/example.yaml"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        result.output_file = str(output_file)

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                return_value=(False, ["overlay compile failed"], {}),
            ) as mock_overlay,
            patch("axiom_encode.cli._apply_generated_encoding_result") as mock_apply,
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 1
        mock_overlay.assert_called_once()
        mock_apply.assert_not_called()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["status"] == "apply_blocked_validation"
        assert run.outcome["overlay_validation_success"] is False
        assert run.outcome["apply_error"] == "overlay compile failed"
        assert run.outcome["final_success"] is False

    def test_encode_apply_records_final_success_when_overlay_apply_passes(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(True)
        output_file = tmp_path / "out" / "codex-test-model" / "regulations/example.yaml"
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "regulations/example.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                return_value=(True, [], {}),
            ),
            patch(
                "axiom_encode.cli._apply_generated_encoding_result",
                return_value=[applied_file],
            ),
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_encode(args)

        assert exc_info.value.code == 0
        output = capsys.readouterr().out
        assert "standalone_success=True" in output
        assert "outcome=apply_applied final_success=True" in output
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.iterations[0].success is True
        assert run.outcome["standalone_validation_success"] is True
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["apply_success"] is True
        assert run.outcome["final_success"] is True
        assert run.outcome["status"] == "apply_applied"
        assert run.success is True
        assert not output_file.with_suffix(".repair.json").exists()
        events = EncodingDB(args.db).get_session_events(run.session_id)
        assert [event.event_type for event in events] == [
            "encode_request",
            "encode_result",
            "encode_outcome",
        ]


class TestGuardGenerated:
    def test_rejects_rulespec_change_without_encoder_manifest(self, tmp_path):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")

        issues = guard_generated_change_issues(
            tmp_path,
            changed_files=["regulations/example.yaml"],
        )

        assert issues == [
            "regulations/example.yaml changed without a matching .axiom/encoding-manifests manifest"
        ]

    def test_accepts_rulespec_change_with_matching_encoder_manifest(self, tmp_path):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")
        manifest = tmp_path / ".axiom/encoding-manifests/regulations/example.json"
        manifest.parent.mkdir(parents=True)
        manifest_payload = _signed_manifest_payload(
            {
                "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                "applied_files": [
                    {
                        "path": "regulations/example.yaml",
                        "sha256": _sha256_file(rule),
                    }
                ],
            }
        )
        manifest.write_text(json.dumps(manifest_payload) + "\n")

        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            issues = guard_generated_change_issues(
                tmp_path,
                changed_files=[
                    "regulations/example.yaml",
                    ".axiom/encoding-manifests/regulations/example.json",
                ],
            )

        assert issues == []

    def test_rejects_rulespec_change_with_stale_encoder_manifest(self, tmp_path):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")
        manifest = tmp_path / ".axiom/encoding-manifests/regulations/example.json"
        manifest.parent.mkdir(parents=True)
        manifest_payload = _signed_manifest_payload(
            {
                "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                "applied_files": [
                    {
                        "path": "regulations/example.yaml",
                        "sha256": "not-current",
                    }
                ],
            }
        )
        manifest.write_text(json.dumps(manifest_payload) + "\n")

        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            issues = guard_generated_change_issues(
                tmp_path,
                changed_files=[
                    "regulations/example.yaml",
                    ".axiom/encoding-manifests/regulations/example.json",
                ],
            )

        assert issues == [
            "regulations/example.yaml content does not match the encoder apply manifest sha256"
        ]

    def test_rejects_rulespec_change_with_unsigned_encoder_manifest(self, tmp_path):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")
        manifest = tmp_path / ".axiom/encoding-manifests/regulations/example.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                    "applied_files": [
                        {
                            "path": "regulations/example.yaml",
                            "sha256": _sha256_file(rule),
                        }
                    ],
                }
            )
            + "\n"
        )

        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            issues = guard_generated_change_issues(
                tmp_path,
                changed_files=[
                    "regulations/example.yaml",
                    ".axiom/encoding-manifests/regulations/example.json",
                ],
            )

        assert issues == [
            ".axiom/encoding-manifests/regulations/example.json is missing an encoder apply manifest signature"
        ]

    def test_guard_generated_command_exits_nonzero_for_manual_change(
        self, capsys, tmp_path
    ):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")
        args = SimpleNamespace(
            repo=tmp_path,
            base_ref=None,
            head_ref="HEAD",
            roots="regulations",
            json=False,
        )

        with (
            patch(
                "axiom_encode.cli._git_changed_files",
                return_value=["regulations/example.yaml"],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cmd_guard_generated(args)

        assert exc_info.value.code == 1
        assert "Manual RuleSpec changes are not allowed." in capsys.readouterr().out


class TestApplyDependencyValidation:
    def test_source_relation_preservation_rejects_dropped_existing_edge(self):
        existing = """format: rulespec/v1
rules:
  - name: sets_snap_telephone_utility_allowance
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulations/7-cfr/273/9#snap_individual_utility_allowance_state_option
      authority: state
      value: us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance
      basis:
        delegation: us:regulations/7-cfr/273/9#snap_state_standard_utility_allowance_delegation
"""
        generated = """format: rulespec/v1
rules:
  - name: snap_individual_utility_allowance
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: telephone_standard_allowance_amount
"""

        issues = _source_relation_preservation_issues(existing, generated)

        assert len(issues) == 1
        assert "dropped existing source_relation" in issues[0]
        assert "sets_snap_telephone_utility_allowance" in issues[0]
        assert (
            "us:regulations/7-cfr/273/9#snap_individual_utility_allowance_state_option"
            in issues[0]
        )

    def test_source_relation_preservation_allows_preserved_edge(self):
        existing = """format: rulespec/v1
rules:
  - name: sets_snap_telephone_utility_allowance
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulations/7-cfr/273/9#snap_individual_utility_allowance_state_option
      authority: state
      value: us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance
      basis:
        delegation: us:regulations/7-cfr/273/9#snap_state_standard_utility_allowance_delegation
"""
        generated = """format: rulespec/v1
rules:
  - name: renamed_relation_record
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulations/7-cfr/273/9#snap_individual_utility_allowance_state_option
      authority: state
      value: us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance
      basis:
        delegation: us:regulations/7-cfr/273/9#snap_state_standard_utility_allowance_delegation
"""

        assert _source_relation_preservation_issues(existing, generated) == []

    def test_apply_overlay_validation_rejects_dropped_source_relation(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us-ny"
        target = policy_repo / "regulations/18-nycrr/387/12/f/3/v/c.yaml"
        generated = (
            output_root
            / "codex-test-model"
            / "regulations/18-nycrr/387/12/f/3/v/c.yaml"
        )
        target.parent.mkdir(parents=True)
        generated.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: sets_snap_telephone_utility_allowance
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulations/7-cfr/273/9#snap_individual_utility_allowance_state_option
      authority: state
      value: us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance
      basis:
        delegation: us:regulations/7-cfr/273/9#snap_state_standard_utility_allowance_delegation
"""
        )
        generated.write_text("format: rulespec/v1\nrules: []\n")
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
            result,
            output_root=output_root,
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        assert ok is False
        assert supplemental == {}
        assert len(issues) == 1
        assert issues[0].startswith("regulations/18-nycrr/387/12/f/3/v/c.yaml: ")
        assert "dropped existing source_relation" in issues[0]

    def test_apply_overlay_validation_rejects_deferred_replacement_of_executable_file(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "regulations/7-cfr/273/10.yaml"
        generated = output_root / "codex-test-model" / "regulations/7-cfr/273/10.yaml"
        target.parent.mkdir(parents=True)
        generated.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_calculated_monthly_allotment
"""
        )
        generated.write_text(
            """format: rulespec/v1
module:
  status: deferred
rules: []
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
            result,
            output_root=output_root,
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        assert ok is False
        assert supplemental == {}
        assert len(issues) == 1
        assert issues[0].startswith("regulations/7-cfr/273/10.yaml: ")
        assert "cannot replace existing executable rules" in issues[0]

    def test_apply_overlay_validation_requires_policy_proofs(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/7/2014/a.yaml"
        generated.parent.mkdir(parents=True)
        policy_repo.mkdir()
        generated.write_text("format: rulespec/v1\nrules: []\n")
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")
        seen_require_policy_proofs: list[bool] = []

        class FakePipeline:
            def __init__(self, **kwargs):
                seen_require_policy_proofs.append(
                    bool(kwargs.get("require_policy_proofs"))
                )

            def validate(self, _path, *, skip_reviewers):
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with patch("axiom_encode.cli.ValidatorPipeline", FakePipeline):
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
            )

        assert ok is True
        assert issues == []
        assert supplemental == {}
        assert seen_require_policy_proofs == [True]

    def test_repair_nonnegative_floors_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "regulations/7-cfr/273/10.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if state_agency_rounds_thirty_percent_net_income_up: snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate) else: floor(snap_maximum_allotment_for_household_size - (snap_net_monthly_income * snap_allotment_net_income_reduction_rate))

  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if household_initial_month and snap_calculated_monthly_allotment_before_minimums < snap_initial_month_minimum_issuance: 0 else: snap_calculated_monthly_allotment_before_minimums
"""
        )
        test_file = policy_repo / "regulations/7-cfr/273/10.test.yaml"
        test_file.write_text(
            """- name: existing_positive_case
  period: 2026-01
  input:
    us:regulations/7-cfr/273/10#input.household_size: 1
  output:
    us:regulations/7-cfr/273/10#snap_monthly_allotment: 24
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("regulations/7-cfr/273/10.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_nonnegative_floors(args)

        content = target.read_text()
        assert "max(0, snap_maximum_allotment_for_household_size - ceil(" in content
        assert "else: max(0, floor(snap_maximum_allotment_for_household_size" in content
        test_content = test_file.read_text()
        assert "initial_month_high_income_zero_allotment" in test_content
        assert "us:regulations/7-cfr/273/10#snap_monthly_allotment: 0" in test_content
        manifest = (
            policy_repo / ".axiom/encoding-manifests/regulations/7-cfr/273/10.json"
        )
        payload = json.loads(manifest.read_text())
        assert payload["schema_version"] == APPLIED_ENCODING_MANIFEST_SCHEMA
        assert payload["backend"] == "deterministic"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "regulations/7-cfr/273/10.yaml",
            "regulations/7-cfr/273/10.test.yaml",
        ]

    def test_repair_zero_branch_tests_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/21.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: cdcc_expense_limit_before_exclusion
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if cdcc_qualifying_individual_count == 1:
              cdcc_one_qualifying_individual_expense_limit
          else:
              if cdcc_qualifying_individual_count >= 2:
                  cdcc_two_or_more_qualifying_individuals_expense_limit
              else:
                  0
"""
        )
        test_file = policy_repo / "statutes/26/21.test.yaml"
        test_file.write_text(
            """- name: existing_positive_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/21#input.taxable_year_begins_in_2021: false
  output:
    us:statutes/26/21#cdcc_expense_limit_before_exclusion: 3000
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/21.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_zero_branch_tests(args)

        test_content = test_file.read_text()
        assert "no_qualifying_individual_zero_cdcc_expense_limit" in test_content
        assert (
            "us:statutes/26/21#input.is_dependent_under_section_152_a_1: false"
            in test_content
        )
        assert "us:statutes/26/21#input.age: 0" in test_content
        assert (
            "us:statutes/26/21#input.employment_related_expense_requirements_satisfied: false"
            in test_content
        )
        assert (
            "us:statutes/26/21#cdcc_expense_limit_before_exclusion: 0" in test_content
        )
        assert "us:statutes/26/21#cdcc_creditable_expenses: 0" in test_content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/21.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "zero-branch-test-v1"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/21.yaml",
            "statutes/26/21.test.yaml",
        ]

    def test_apply_overlay_validation_checks_direct_dependents_by_default(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/7/2015/d/2/C.yaml"
        dependent = policy_repo / "statutes/7/2015/d/2.yaml"
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/7/2015/d/2/C\nrules: []\n"
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")
        validated_paths: list[Path] = []

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                validated_paths.append(Path(path))
                return SimpleNamespace(all_passed=True, results={})

        with patch("axiom_encode.cli.ValidatorPipeline", FakePipeline):
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
            )

        assert ok is True
        assert issues == []
        assert supplemental == {}
        assert [path.name for path in validated_paths] == ["C.yaml", "2.yaml"]

    def test_apply_overlay_validation_can_skip_dependents_for_cascading_migration(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/24/h.yaml"
        dependent = policy_repo / "statutes/26/24/d.yaml"
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/26/24/h\nrules: []\n"
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")
        validated_paths: list[Path] = []

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                validated_paths.append(Path(path))
                return SimpleNamespace(all_passed=True, results={})

        with patch("axiom_encode.cli.ValidatorPipeline", FakePipeline):
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
                validate_dependents=False,
            )

        assert ok is True
        assert issues == []
        assert supplemental == {}
        assert [path.name for path in validated_paths] == ["h.yaml"]

    def test_apply_overlay_validation_fills_dependent_inputs_from_baseline(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/63/c.yaml"
        generated_test = generated.with_name("c.test.yaml")
        dependent = policy_repo / "statutes/26/63.yaml"
        dependent_test = dependent.with_name("63.test.yaml")
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated_test.write_text(
            """- name: baseline
  input:
    us:statutes/26/63/c#input.married_filing_separately_and_either_spouse_itemizes: false
    us:statutes/26/63/c#input.additional_standard_deduction_entitlement_count_under_subsection_f: 0
    us:statutes/26/63/c/5#input.earned_income: 0
  output:
    us:statutes/26/63/c#standard_deduction: 16100
"""
        )
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/26/63/c\nrules: []\n"
        )
        dependent_test.write_text(
            """- name: case_one
  input:
    us:statutes/26/63#input.adjusted_gross_income: 80000
  output:
    us:statutes/26/63#taxable_income: 63900
- name: case_two
  input:
    us:statutes/26/63#input.adjusted_gross_income: 90000
  output:
    us:statutes/26/63#taxable_income: 73900
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")
        required = [
            "married_filing_separately_and_either_spouse_itemizes",
            "additional_standard_deduction_entitlement_count_under_subsection_f",
            "earned_income",
        ]

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "c.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                test_content = Path(path).with_name("63.test.yaml").read_text()
                for input_name in required:
                    if test_content.count(f"#input.{input_name}:") < 2:
                        return SimpleNamespace(
                            all_passed=False,
                            results={
                                "ci": SimpleNamespace(
                                    error=(
                                        "Test case `case_one` execution failed: "
                                        f"missing input `{input_name}`"
                                    )
                                )
                            },
                        )
                return SimpleNamespace(all_passed=True, results={})

        with patch("axiom_encode.cli.ValidatorPipeline", FakePipeline):
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
            )

        assert ok is True
        assert issues == []
        updated = supplemental[Path("statutes/26/63.test.yaml")]
        assert (
            "us:statutes/26/63/c#input.married_filing_separately_and_either_spouse_itemizes: false"
            in updated
        )
        assert (
            "us:statutes/26/63/c#input.additional_standard_deduction_entitlement_count_under_subsection_f: 0"
            in updated
        )
        assert "us:statutes/26/63/c/5#input.earned_income: 0" in updated
        assert (
            updated.count(
                "us:statutes/26/63/c#input.married_filing_separately_and_either_spouse_itemizes"
            )
            == 2
        )

    def test_find_rulespec_dependents_finds_canonical_imports(self, tmp_path):
        repo = tmp_path / "rulespec-us-ny"
        target = repo / "regulations/18-nycrr/387/12/f/3/v/c.yaml"
        dependent = repo / "policies/otda/snap/fy-2026-benefit-calculation.yaml"
        unrelated = repo / "policies/otda/snap/other.yaml"
        for path in (target, dependent, unrelated):
            path.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("format: rulespec/v1\nrules: []\n")
        dependent.write_text(
            """format: rulespec/v1
imports:
  - us-ny:regulations/18-nycrr/387/12/f/3/v/c
rules: []
"""
        )
        unrelated.write_text("format: rulespec/v1\nimports: []\nrules: []\n")

        dependents = _find_rulespec_dependents(
            repo, Path("regulations/18-nycrr/387/12/f/3/v/c.yaml")
        )

        assert dependents == [dependent]

    def test_insert_false_input_default_uses_base_anchor(self):
        content = """- name: first_case
  period: 2026-01
  input: &base_case
    existing: true
  output:
    result: 1

- name: second_case
  period: 2026-01
  input:
    <<: *base_case
  output:
    result: 2
"""

        updated = _insert_false_input_default(
            content,
            "us-ny:regulations/18-nycrr/387/12/f/3/v/c#input.household_has_basic_telephone_service_cost",
        )

        assert (
            "  input: &base_case\n"
            "    us-ny:regulations/18-nycrr/387/12/f/3/v/c#input.household_has_basic_telephone_service_cost: false\n"
            "    existing: true\n"
        ) in updated


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
        args.all = False

        with patch(
            "axiom_encode.supabase_sync.sync_agent_sessions_to_supabase",
            return_value={"total": 2, "synced": 2, "failed": 0},
        ) as mock_sync:
            cmd_sync_agent_sessions(args)
        mock_sync.assert_called_once_with(session_id=None, include_all=False)
        captured = capsys.readouterr()
        assert "2 synced" in captured.out

    def test_sync_agent_sessions_include_all(self, capsys):
        args = SimpleNamespace(session=None, all=True)

        with patch(
            "axiom_encode.supabase_sync.sync_agent_sessions_to_supabase",
            return_value={"total": 3, "synced": 3, "failed": 0},
        ) as mock_sync:
            cmd_sync_agent_sessions(args)
        mock_sync.assert_called_once_with(session_id=None, include_all=True)
        captured = capsys.readouterr()
        assert "3 synced" in captured.out

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
# Additional edge case tests for remaining uncovered lines
# =========================================================================


class TestCmdValidateEdgeCases:
    def test_validate_rules_us_found_in_path(self, capsys, tmp_path):
        """Test validate when file is inside a rulespec-us directory (line 350)."""
        rules_us = tmp_path / "rulespec-us" / "statutes" / "26" / "1"
        rules_us.mkdir(parents=True)
        rulespec_file = rules_us / "test.yaml"
        rulespec_file.write_text("# test")
        # Create axiom-rules-engine sibling
        axiom_rules_path = tmp_path / "axiom-rules-engine"
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
            # Verify axiom_rules_path was resolved via rules_us.parent / "axiom-rules-engine"
            call_kwargs = mock_pipeline_cls.call_args[1]
            assert "rulespec-us" in str(call_kwargs["policy_repo_path"])

    def test_validate_uses_enclosing_non_federal_policy_repo(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us-tn" / "sources"
        policy_repo.mkdir(parents=True)
        rulespec_file = policy_repo / "test.yaml"
        rulespec_file.write_text("format: rulespec/v1\n")
        (tmp_path / "axiom-rules-engine").mkdir()

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
        assert call_kwargs["policy_repo_path"] == tmp_path / "rulespec-us-tn"
        assert call_kwargs["axiom_rules_path"] == tmp_path / "axiom-rules-engine"

    def test_validate_fallback_prefers_workspace_repo_roots(self, tmp_path):
        rulespec_file = tmp_path / "generated" / "test.yaml"
        rulespec_file.parent.mkdir(parents=True)
        rulespec_file.write_text("# test")
        workspace_root = tmp_path / "workspace"
        rules_us_path = workspace_root / "rulespec-us"
        axiom_rules_path = workspace_root / "axiom-rules-engine"
        rules_us_path.mkdir(parents=True)
        axiom_rules_path.mkdir()

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

        with (
            patch(
                "axiom_encode.cli._resolve_repo_checkout",
                side_effect=lambda name: workspace_root / name,
            ),
            patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline = mock_pipeline_cls.return_value
            mock_pipeline.validate.return_value = mock_result
            with pytest.raises(SystemExit) as exc_info:
                cmd_validate(args)
            assert exc_info.value.code == 0

        call_kwargs = mock_pipeline_cls.call_args[1]
        assert call_kwargs["policy_repo_path"] == rules_us_path
        assert call_kwargs["axiom_rules_path"] == axiom_rules_path


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
