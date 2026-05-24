"""
Tests for axiom_encode CLI (cli.py).

Tests all CLI commands using subprocess invocation and direct function calls.
All external dependencies are mocked.
"""

import json
import os
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from axiom_encode import __version__ as AXIOM_ENCODE_TEST_VERSION
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_SIGNING_KEY_ENV,
    _append_generated_derived_output_tests_if_missing,
    _apply_generated_encoding_result,
    _complete_missing_imported_test_inputs,
    _default_generated_test_input_value,
    _discover_rulespec_test_files,
    _effective_runner_specs,
    _ensure_generated_dependency_files_safe,
    _find_rulespec_dependents,
    _has_zero_output_test,
    _hoist_nested_test_tables,
    _insert_false_input_default,
    _local_factual_input_names_from_rules_content,
    _repair_employer_scoped_entities,
    _repair_generated_import_symbol_near_misses,
    _repair_input_field_accesses_in_formulas,
    _repair_missing_source_proof_atoms,
    _repair_mixed_scalar_output_tests,
    _repair_section_151_imports,
    _repair_section_151_temporal_fact_names,
    _repair_shared_statutory_rate_names,
    _require_axiom_encode_version_provenance,
    _rewrite_gpt_runner_backend,
    _sha256_file,
    _sha256_text,
    _sign_applied_encoding_manifest,
    _source_relation_preservation_issues,
    _suppress_rulespec_ancestor_targets_for_subsection_overlay,
    _try_repair_generated_policyengine_oracle_inputs_for_apply,
    _try_repair_generated_section_1401_b_1_self_employment_income_for_apply,
    _try_repair_generated_source_table_band_scalars_for_apply,
    _validate_generated_encoding_in_policy_overlay,
    _write_applied_encoding_manifest,
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
    cmd_repair_current_year_final_amounts,
    cmd_repair_imported_test_inputs,
    cmd_repair_missing_source_proofs,
    cmd_repair_nonnegative_floors,
    cmd_repair_oracle_parameter_tests,
    cmd_repair_proof_import_hashes,
    cmd_repair_section_63_f_stale_test_inputs,
    cmd_repair_section_172_c_capacity,
    cmd_repair_section_911_a_1_exclusion,
    cmd_repair_tax_filing_status_branches,
    cmd_repair_tax_status_components,
    cmd_repair_unreferenced_proof_imports,
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


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_test_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")


def _write_test_encoder_version(repo: Path, version: str) -> None:
    package_init = repo / "src/axiom_encode/__init__.py"
    package_init.parent.mkdir(parents=True, exist_ok=True)
    package_init.write_text(f'__version__ = "{version}"\n')
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "axiom-encode"\nversion = "{version}"\n'
    )
    (repo / "uv.lock").write_text(
        f'[[package]]\nname = "axiom-encode"\nversion = "{version}"\n'
    )


def _write_test_encoder_source(repo: Path, content: str) -> None:
    prompt = repo / "src/axiom_encode/prompts/encoder.py"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(f'PROMPT = "{content}"\n')


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

    def test_executes_companion_tests_with_custom_period_name(self, capsys, tmp_path):
        repo = tmp_path / "rulespec-us"
        target = repo / "policies/ssa/base.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: contribution_base
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: 100
"""
        )
        target.with_name("base.test.yaml").write_text(
            """- name: custom_calendar_year_period
  period:
    period_kind: custom
    name: calendar_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    us:policies/ssa/base#contribution_base: 100
"""
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(self._args(repo, json_output=True))

        assert exc_info.value.code == 0
        assert json.loads(capsys.readouterr().out)["success"] is True

    def test_executes_companion_tests_with_table_rows(self, capsys, tmp_path):
        repo = tmp_path / "rulespec-us"
        target = repo / "statutes/1/payment.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: net_payment
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: payment_amount - excluded_amount
"""
        )
        target.with_name("payment.test.yaml").write_text(
            """- name: two_payment_rows
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  tables:
    Payment:
      - us:statutes/1/payment#input.payment_amount: 100
        us:statutes/1/payment#input.excluded_amount: 40
      - us:statutes/1/payment#input.payment_amount: 20
        us:statutes/1/payment#input.excluded_amount: 50
  output:
    us:statutes/1/payment#net_payment:
      - 60
      - -30
"""
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_test(self._args(repo, json_output=True))

        assert exc_info.value.code == 0
        assert json.loads(capsys.readouterr().out)["success"] is True

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
            if (
                len(cmd) >= 6
                and cmd[:3] == ["git", "-C", str(repo)]
                and cmd[3:] == ["remote", "get-url", "origin"]
            ):
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="https://github.com/TheAxiomFoundation/rulespec-us.git\n",
                    stderr="",
                )
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

    def test_executes_companion_tests_from_canonical_alias_checkout(
        self, capsys, monkeypatch, tmp_path
    ):
        repo = tmp_path / "rulespec-us-clean.abcd"
        rules_file = repo / "statutes/1/alias.yaml"
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
        rules_file.with_name("alias.test.yaml").write_text(
            """- name: canonical_temp_checkout_output
  period: 2026-01
  input:
    us:statutes/1/alias#input.income: 5
  output:
    us:statutes/1/alias#benefit: 6
"""
        )

        engine_root = tmp_path / "axiom-rules-engine"
        binary = engine_root / "target/debug/axiom-rules-engine"
        compiled_ids: list[str] = []
        compile_programs: list[Path] = []

        def fake_binary(self):
            return binary

        def fake_run(cmd, **kwargs):
            if len(cmd) >= 6 and cmd[:3] == ["git", "-C", str(repo)]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout="https://github.com/TheAxiomFoundation/rulespec-us.git\n",
                    stderr="",
                )
            if "compile" in cmd:
                program_path = Path(cmd[cmd.index("--program") + 1])
                compile_programs.append(program_path)
                item_id = "us-clean.abcd:statutes/1/alias#benefit"
                compiled_ids.append(item_id)
                output_path = Path(cmd[cmd.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "program": {
                                "parameters": [],
                                "derived": [
                                    {
                                        "id": item_id,
                                        "name": "benefit",
                                        "entity": "Household",
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
                                        "us:statutes/1/alias#benefit": {
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
        assert compiled_ids == ["us-clean.abcd:statutes/1/alias#benefit"]
        assert len(compile_programs) == 1
        assert "rulespec-us" in compile_programs[0].parts
        assert "rulespec-us-clean.abcd" not in str(compile_programs[0])

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
            patch(
                "axiom_encode.cli._require_axiom_encode_version_provenance",
                return_value={
                    "version": AXIOM_ENCODE_TEST_VERSION,
                    "version_commit": "version123",
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
            "version": AXIOM_ENCODE_TEST_VERSION,
            "version_commit": "version123",
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

    def test_write_applied_manifest_deduplicates_applied_files(self, tmp_path):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/1401.yaml"
        target_test = policy_repo / "statutes/26/1401.test.yaml"
        generated = output_root / "deterministic-repair" / "statutes/26/1401.yaml"
        target.parent.mkdir(parents=True)
        generated.parent.mkdir(parents=True)
        target.write_text("format: rulespec/v1\nrules: []\n")
        target_test.write_text("- name: existing_case\n  period: 2026\n")
        generated.write_text(target.read_text())
        result = SimpleNamespace(
            output_file=str(generated),
            context_manifest_file=None,
            trace_file=None,
            generation_prompt_sha256=None,
            tool="axiom-encode test",
            citation="us:statutes/26/1401",
            runner="deterministic",
            backend="deterministic",
            model="manifest-test",
        )

        manifest = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=policy_repo,
            relative_output=Path("statutes/26/1401.yaml"),
            applied_files=[target, target_test, target_test],
            axiom_encode_git={
                "root": "/repo/axiom-encode",
                "commit": "abc123",
                "dirty_tracked": False,
            },
            signing_key=TEST_APPLY_SIGNING_KEY,
            run_id="run-123",
        )

        payload = json.loads(manifest.read_text())
        assert [item["path"] for item in payload["applied_files"]] == [
            "statutes/26/1401.yaml",
            "statutes/26/1401.test.yaml",
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

    def test_apply_version_provenance_allows_changes_in_version_bump_commit(
        self, tmp_path
    ):
        repo = tmp_path / "axiom-encode"
        _init_test_git_repo(repo)
        _write_test_encoder_version(repo, "0.2.68")
        _write_test_encoder_source(repo, "old guard")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "Initial encoder version")

        _write_test_encoder_version(repo, AXIOM_ENCODE_TEST_VERSION)
        _write_test_encoder_source(repo, "new guard")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "Bump encoder version")

        provenance = _require_axiom_encode_version_provenance(repo)

        assert provenance["version"] == AXIOM_ENCODE_TEST_VERSION
        assert len(provenance["version_commit"]) == 40

    def test_apply_version_provenance_rejects_code_after_version_bump(self, tmp_path):
        repo = tmp_path / "axiom-encode"
        _init_test_git_repo(repo)
        _write_test_encoder_version(repo, AXIOM_ENCODE_TEST_VERSION)
        _write_test_encoder_source(repo, "versioned guard")
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "Bump encoder version")
        _write_test_encoder_source(repo, "unversioned guard")
        _git(repo, "add", "src/axiom_encode/prompts/encoder.py")
        _git(repo, "commit", "-m", "Change encoder without version")

        with pytest.raises(RuntimeError, match="encoder-affecting files changed"):
            _require_axiom_encode_version_provenance(repo)

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

    def test_encode_apply_auto_repairs_nonnegative_taxable_income_floor(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "63.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: taxable_income_for_individual_who_does_not_itemize else: taxable_income_general_rule
"""
        )
        test_file = output_file.with_name("63.test.yaml")
        test_file.write_text(
            """- name: existing_positive_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100000
  output:
    us:statutes/26/63#taxable_income: 100000
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/63.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/63.yaml: ci: "
                            "Nonnegative taxable income missing floor: "
                            "`taxable_income` can return a negative amount."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert "apply=auto_repaired_nonnegative_floors:taxable_income" in output
        assert "outcome=apply_applied final_success=True" in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        assert (
            "if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: "
            "max(0, taxable_income_for_individual_who_does_not_itemize) "
            "else: max(0, taxable_income_general_rule)" in output_file.read_text()
        )
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_nonnegative_floors"] == ["taxable_income"]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_removes_empty_deferred_source_values(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "3201.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3201
  deferred_outputs:
    - output: us:statutes/26/3201#tier_1_applicable_percentage
      reason: Requires upstream section 3101 rates.
      source_values: []
rules: []
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3201.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/3201.yaml: ci: "
                            "module.deferred_outputs[0].source_values must list "
                            "absolute RuleSpec targets for source-stated values "
                            "retained for the deferred output."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert "apply=auto_repaired_empty_deferred_source_values:" in output
        assert "source_values:" not in output_file.read_text()
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_empty_deferred_source_values"] == [
            "source_values[0]"
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_promotes_module_imports(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "3201.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
  imports:
    - us:statutes/26/3241/b#section_3201_applicable_percentage_for_tax_unit
  rules:
    - name: tier_2_employee_tax
      kind: derived
      entity: TaxUnit
      dtype: Money
      period: Year
      versions:
        - effective_from: '2026-01-01'
          formula: compensation * section_3201_applicable_percentage_for_tax_unit
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3201.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/3201.yaml: ci: Test input assignment "
                            "missing: imported symbol was not available."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_module_imports:"
            "us:statutes/26/3241/b#section_3201_applicable_percentage_for_tax_unit"
            in output
        )
        assert "apply=auto_repaired_module_rules:tier_2_employee_tax" in output
        repaired = yaml.safe_load(output_file.read_text())
        assert repaired["imports"] == [
            "us:statutes/26/3241/b#section_3201_applicable_percentage_for_tax_unit"
        ]
        assert [rule["name"] for rule in repaired["rules"]] == ["tier_2_employee_tax"]
        assert "imports" not in repaired["module"]
        assert "rules" not in repaired["module"]
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_module_imports"] == [
            "us:statutes/26/3241/b#section_3201_applicable_percentage_for_tax_unit"
        ]
        assert run.outcome["auto_repaired_module_rules"] == ["tier_2_employee_tax"]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_repairs_generated_proof_import_hashes(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        imported_file = args.policy_repo_path / "statutes/26/3241/b.yaml"
        imported_file.parent.mkdir(parents=True)
        imported_file.write_text("format: rulespec/v1\nrules: []\n")
        expected_hash = _sha256_file(imported_file)
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "3211.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
- us:statutes/26/3241/b#section_3211_and_3221_applicable_percentage_for_tax_unit
rules:
- name: employee_representative_tier_2_tax
  kind: derived
  entity: TaxUnit
  dtype: Money
  period: Year
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: import
        import:
          target: us:statutes/26/3241/b#section_3211_and_3221_applicable_percentage_for_tax_unit
          output: section_3211_and_3221_applicable_percentage_for_tax_unit
          hash: sha256:old
  versions:
  - effective_from: '2026-01-01'
    formula: compensation * section_3211_and_3221_applicable_percentage_for_tax_unit
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3211.yaml"
        hash_issue = (
            "statutes/26/3211.yaml: ci: Proof import hash mismatch: "
            "`employee_representative_tier_2_tax` proof atom 0 imports "
            "`us:statutes/26/3241/b#section_3211_and_3221_applicable_percentage_for_tax_unit` "
            "with `hash: sha256:old`."
        )

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (False, [hash_issue], {}),
                    (True, [], {}),
                ],
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
        assert "apply=auto_repaired_proof_import_hashes:hash[0]" in output
        assert "hash: sha256:old" not in output_file.read_text()
        assert f"hash: sha256:{expected_hash}" in output_file.read_text()
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_proof_import_hashes"] == ["hash[0]"]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_repairs_person_scoped_rate_base(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "3201.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/3241/b#section_3201_applicable_percentage_for_tax_unit
module:
  source_verification:
    corpus_citation_path: us/statute/26/3201
  deferred_outputs:
    - output: us:statutes/26/3201#a#tier_1_income_tax
      reason: 26 USC 3201(a) depends on section 3101 rates.
      source_values: []
rules:
  - name: tier_2_income_tax_rate
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: section_3201_applicable_percentage_for_tax_unit
  - name: tier_2_income_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: compensation_received_for_services_rendered_during_calendar_year * tier_2_income_tax_rate
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3201.yaml"
        person_scope_issue = (
            "statutes/26/3201.yaml: ci: Person-scoped rate base at unit scope: "
            "`tier_2_income_tax` is declared on `TaxUnit` and multiplies a "
            "base by a rate, but the source text states the amount or tax for "
            "each/every individual, person, member, employee, or similar "
            "lower-entity subject."
        )

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (False, [person_scope_issue], {}),
                    (False, [person_scope_issue], {}),
                    (False, [person_scope_issue], {}),
                    (True, [], {}),
                ],
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
        assert "apply=auto_repaired_empty_deferred_source_values:" in output
        assert (
            "apply=auto_repaired_deferred_output_paths:"
            "us:statutes/26/3201/a#tier_1_income_tax"
        ) in output
        assert (
            "apply=auto_repaired_person_scoped_rate_base:"
            "tier_2_income_tax,tier_2_income_tax_rate"
        ) in output
        repaired = yaml.safe_load(output_file.read_text())
        assert "source_values" not in repaired["module"]["deferred_outputs"][0]
        assert (
            repaired["module"]["deferred_outputs"][0]["output"]
            == "us:statutes/26/3201/a#tier_1_income_tax"
        )
        assert [rule["entity"] for rule in repaired["rules"]] == ["Person", "Person"]
        assert mock_overlay.call_count == 4
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_empty_deferred_source_values"] == [
            "source_values[0]"
        ]
        assert run.outcome["auto_repaired_deferred_output_paths"] == [
            "us:statutes/26/3201/a#tier_1_income_tax"
        ]
        assert run.outcome["auto_repaired_person_scoped_rate_base"] == [
            "tier_2_income_tax",
            "tier_2_income_tax_rate",
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_auto_repairs_section_1401_policyengine_oracle_inputs(
        self, capsys, tmp_path
    ):
        args = self._make_args(
            tmp_path, backend="codex", citation="26 USC 1401(a)", sync=False
        )
        args.apply = True
        result = self._make_eval_result(True)
        output_file = (
            tmp_path
            / "out"
            / "codex-test-model"
            / "statutes"
            / "26"
            / "1401"
            / "a.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-01-01'
        formula: taxable_self_employment_income_for_section_1401_a * 0.124
"""
        )
        test_file = output_file.with_name("a.test.yaml")
        test_file.write_text(
            """- name: section_1401_a_uncapped
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1200
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
    us:statutes/26/1402/b#input.contribution_and_benefit_base_under_section_230_of_social_security_act: 5000
    us:statutes/26/1402/b#input.wages_paid_to_individual_for_section_1401_a: 100
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 114.514
- name: section_1401_a_cap
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 2500
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
    us:statutes/26/1402/b#input.contribution_and_benefit_base_under_section_230_of_social_security_act: 1200
    us:statutes/26/1402/b#input.wages_paid_to_individual_for_section_1401_a: 500
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 86.8
- name: section_1401_a_2026_cap_uses_policyengine_base
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 2500
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
    us:statutes/26/1402/b#input.contribution_and_benefit_base_under_section_230_of_social_security_act: 1200
    us:statutes/26/1402/b#input.wages_paid_to_individual_for_section_1401_a: 500
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 86.8
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/1401/a.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._policyengine_us_oasdi_base_for_year",
                side_effect=lambda year: {
                    2024: Decimal("168600"),
                    2026: Decimal("186000"),
                }.get(year),
            ),
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
        assert (
            "apply=auto_repaired_policyengine_oracle_inputs:"
            "section_1401_a_uncapped,section_1401_a_cap,"
            "section_1401_a_2026_cap_uses_policyengine_base"
        ) in output
        assert mock_overlay.call_count == 1
        mock_apply.assert_called_once()
        test_payload = yaml.safe_load(test_file.read_text())
        assert test_payload[0]["oracle_inputs"]["policyengine"] == {
            "self_employment_income": 1000,
            "employment_income": 100,
        }
        assert test_payload[1]["oracle_inputs"]["policyengine"] == {
            "self_employment_income": 2300,
            "employment_income": 167900,
        }
        assert test_payload[2]["oracle_inputs"]["policyengine"] == {
            "self_employment_income": 2300,
            "employment_income": 185300,
        }
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_policyengine_oracle_inputs"] == [
            "section_1401_a_uncapped",
            "section_1401_a_cap",
            "section_1401_a_2026_cap_uses_policyengine_base",
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_apply_repair_adds_section_1401_b_policyengine_oracle_inputs(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        output_file = (
            output_root
            / "codex-test-model"
            / "statutes"
            / "26"
            / "1401"
            / "b"
            / "1.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        test_file = output_file.with_name("1.test.yaml")
        test_file.write_text(
            """- name: section_1401_b_1_medicare
  period: 2024
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1200
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
    us:statutes/26/1402/b#input.wages_paid_to_individual_for_section_1401_a: 100
  output:
    us:statutes/26/1401/b/1#self_employment_income_tax: 26.7815
"""
        )
        result = SimpleNamespace(
            output_file=str(output_file),
            runner="codex-test-model",
        )

        repaired = _try_repair_generated_policyengine_oracle_inputs_for_apply(
            result,
            output_root=output_root,
        )

        assert repaired == ["section_1401_b_1_medicare"]
        test_payload = yaml.safe_load(test_file.read_text())
        assert test_payload[0]["oracle_inputs"]["policyengine"] == {
            "self_employment_income": 1000,
        }

    def test_apply_repair_uses_uncapped_1402b_income_for_section_1401_b_1(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        imported_file = policy_repo / "statutes" / "26" / "1402" / "b.yaml"
        imported_file.parent.mkdir(parents=True)
        imported_file.write_text("format: rulespec/v1\nrules: []\n")
        imported_hash = _sha256_file(imported_file)
        output_file = (
            output_root
            / "codex-test-model"
            / "statutes"
            / "26"
            / "1401"
            / "b"
            / "1.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate
  - us:statutes/26/1402/b#self_employment_income_for_section_1401_a
rules:
  - name: self_employment_income_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/1402/b#self_employment_income_for_section_1401_a
              output: self_employment_income_for_section_1401_a
              hash: sha256:old
    versions:
      - effective_from: '1990-01-01'
        formula: self_employment_income_tax_rate * self_employment_income_for_section_1401_a
"""
        )
        test_file = output_file.with_name("1.test.yaml")
        test_file.write_text(
            """- name: self_employment_income_tax_applies_2_9_percent_to_self_employment_income
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1200
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
  output:
    us:statutes/26/1401/b/1#self_employment_income_tax: 26.7815
"""
        )
        result = SimpleNamespace(
            output_file=str(output_file),
            runner="codex-test-model",
        )

        repaired = (
            _try_repair_generated_section_1401_b_1_self_employment_income_for_apply(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
            )
        )

        assert repaired == [
            "import:self_employment_income",
            "formula:self_employment_income_tax",
            "proof:self_employment_income",
            "test:self_employment_income_tax_ignores_section_1401_a_cap",
        ]
        rules_payload = yaml.safe_load(output_file.read_text())
        assert rules_payload["imports"] == [
            "us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate",
            "us:statutes/26/1402/b#self_employment_income",
        ]
        rule = rules_payload["rules"][0]
        assert (
            rule["versions"][0]["formula"]
            == "self_employment_income_tax_rate * self_employment_income"
        )
        import_atom = rule["metadata"]["proof"]["atoms"][0]["import"]
        assert import_atom == {
            "target": "us:statutes/26/1402/b#self_employment_income",
            "output": "self_employment_income",
            "hash": f"sha256:{imported_hash}",
        }
        test_payload = yaml.safe_load(test_file.read_text())
        assert test_payload[-1]["name"] == (
            "self_employment_income_tax_ignores_section_1401_a_cap"
        )
        assert test_payload[-1]["output"] == {
            "us:statutes/26/1401/b/1#self_employment_income_tax": 61.59745
        }

    def test_repair_employer_scoped_entities_sets_rate_helpers(self, tmp_path):
        rules_file = tmp_path / "3221.yaml"
        rules_file.write_text(
            """format: rulespec/v1
rules:
  - name: tier_2_employer_tax_rate
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: section_3211_and_3221_applicable_percentage_for_tax_unit
  - name: tier_2_employer_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: compensation_paid * tier_2_employer_tax_rate
"""
        )

        repaired = _repair_employer_scoped_entities(
            rules_file=rules_file,
            target_names=["tier_2_employer_tax"],
        )

        assert repaired == ["tier_2_employer_tax", "tier_2_employer_tax_rate"]
        payload = yaml.safe_load(rules_file.read_text())
        assert [rule["entity"] for rule in payload["rules"]] == [
            "Employer",
            "Employer",
        ]

    def test_repair_shared_statutory_rate_names_updates_tests(self, tmp_path):
        rules_file = tmp_path / "statutes" / "26" / "3241" / "b.yaml"
        test_file = tmp_path / "statutes" / "26" / "3241" / "b.test.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage
    for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: section_3211_3221_applicable_percentage_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          2: 0.181
  - name: section_3211_and_3221_applicable_percentage_for_tax_unit
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: section_3211_3221_applicable_percentage_by_ratio_band[average_account_benefits_ratio_band]
"""
        )
        test_file.write_text(
            """- name: rate
  period: '2026'
  input: {}
  output:
    us:statutes/26/3241/b#section_3211_and_3221_applicable_percentage_for_tax_unit: 0.181
"""
        )

        repaired = _repair_shared_statutory_rate_names(
            rules_file=rules_file,
            test_file=test_file,
            relative_output=Path("statutes/26/3241/b.yaml"),
            target_names=[
                "section_3211_and_3221_applicable_percentage_for_tax_unit",
            ],
        )

        assert repaired == [
            "section_3211_3221_applicable_percentage_by_ratio_band"
            "->applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band",
            "section_3211_and_3221_applicable_percentage_for_tax_unit"
            "->applicable_percentage_for_sections_3211_b_and_3221_b",
        ]
        payload = yaml.safe_load(rules_file.read_text())
        assert payload["rules"][1]["name"] == (
            "applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band"
        )
        assert payload["rules"][2]["name"] == (
            "applicable_percentage_for_sections_3211_b_and_3221_b"
        )
        assert (
            "applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band"
            in payload["rules"][2]["versions"][0]["formula"]
        )
        test_payload = yaml.safe_load(test_file.read_text())
        assert test_payload[0]["output"] == {
            "us:statutes/26/3241/b#applicable_percentage_for_sections_3211_b_and_3221_b": 0.181
        }

    def test_apply_repair_source_table_band_scalars_inlines_bounds(self, tmp_path):
        output_root = tmp_path / "out"
        rules_file = output_root / "runner" / "statutes" / "26" / "3241" / "b.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band_1_upper
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_2_lower
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: average_account_benefits_ratio_band_1_upper
  - name: average_account_benefits_ratio_band_2_upper
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_1_upper: 1
          elif average_account_benefits_ratio >= average_account_benefits_ratio_band_2_lower and average_account_benefits_ratio < average_account_benefits_ratio_band_2_upper: 2
          else: 3
"""
        )
        result = SimpleNamespace(output_file=rules_file, runner="runner")

        repaired = _try_repair_generated_source_table_band_scalars_for_apply(
            result,
            output_root=output_root,
            issues=[
                "Source table row/band scalar parameters: "
                "`average_account_benefits_ratio_band_1_upper`"
            ],
        )

        content = rules_file.read_text()
        assert "average_account_benefits_ratio_band_1_upper" not in content
        assert "average_account_benefits_ratio_band_2_upper" not in content
        assert "average_account_benefits_ratio < 2.5" in content
        assert "average_account_benefits_ratio < 3.0" in content
        assert "average_account_benefits_ratio_band" in repaired

    def test_repair_input_field_accesses_in_formulas(self, tmp_path):
        rules_file = tmp_path / "statutes" / "26" / "3221.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
rules:
  - name: tier_2_employer_tax
    kind: derived
    entity: Employer
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: applicable_percentage * max(0, input.compensation_paid)
"""
        )

        repaired = _repair_input_field_accesses_in_formulas(rules_file=rules_file)

        content = rules_file.read_text()
        assert repaired == ["tier_2_employer_tax:versions[0].formula"]
        assert "input.compensation_paid" not in content
        assert "max(0, compensation_paid)" in content

    def test_encode_apply_auto_repairs_generic_zero_branch_test(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "213.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
rules:
  - name: lodging_treated_as_medical_care
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          lodging_not_lavish_or_extravagant
          and lodging_away_from_home_primarily_for_and_essential_to_medical_care
  - name: lodging_medical_care_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if lodging_treated_as_medical_care:
              min(lodging_amount_paid, lodging_medical_care_nightly_cap * lodging_nights * lodging_individuals)
          else:
              0
"""
        )
        test_file = output_file.with_name("213.test.yaml")
        test_file.write_text(
            """- name: qualifying_lodging_positive
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/213#input.lodging_not_lavish_or_extravagant: true
    us:statutes/26/213#input.lodging_away_from_home_primarily_for_and_essential_to_medical_care: true
    us:statutes/26/213#input.lodging_amount_paid: 300
    us:statutes/26/213#input.lodging_medical_care_nightly_cap: 50
    us:statutes/26/213#input.lodging_nights: 2
    us:statutes/26/213#input.lodging_individuals: 2
  output:
    us:statutes/26/213#lodging_medical_care_amount: 200
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/213.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/213.yaml: ci: "
                            "Zero branch test coverage missing: "
                            "`lodging_medical_care_amount` has a formula branch "
                            "that returns 0."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_zero_branch_tests:"
            "auto_zero_lodging_medical_care_amount"
        ) in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert "auto_zero_lodging_medical_care_amount" in test_content
        assert (
            "us:statutes/26/213#input.lodging_not_lavish_or_extravagant: false"
            in test_content
        )
        assert "us:statutes/26/213#lodging_medical_care_amount: 0" in test_content

    def test_encode_apply_auto_repairs_exception_positive_companion(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "24.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text("format: rulespec/v1\nrules: []\n")
        test_file = output_file.with_name("24.test.yaml")
        test_file.write_text(
            """- name: noncitizen_exception_denies_qualifying_child
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/24#input.qualifying_child_under_section_152_c: true
    us:statutes/26/24#input.age: 10
    us:statutes/26/24#input.noncitizen_exception_to_qualifying_child_under_section_24: true
  output:
    us:statutes/26/24#ctc_qualifying_child: not_holds
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/24.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/24.yaml: ci: "
                            "Exception test coverage missing: "
                            "`ctc_qualifying_child` negates "
                            "`noncitizen_exception_to_qualifying_child_under_section_24`, "
                            "but no companion test sets "
                            "`#input.noncitizen_exception_to_qualifying_child_under_section_24` "
                            "true and expects `ctc_qualifying_child` to be not_holds "
                            "while an otherwise identical positive companion sets that "
                            "exception false and expects holds."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        case_name = (
            "auto_positive_ctc_qualifying_child_"
            "noncitizen_exception_to_qualifying_child_under_section_24"
        )
        assert f"apply=auto_repaired_exception_tests:{case_name}" in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert case_name in test_content
        assert (
            "us:statutes/26/24#input.noncitizen_exception_to_qualifying_child_under_section_24: false"
            in test_content
        )
        assert "us:statutes/26/24#ctc_qualifying_child: holds" in test_content
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_exception_tests"] == [case_name]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_auto_repairs_missing_test_input_assignments(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "151.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: section_151_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if dependent_under_section_152 and tin_included_on_return:
              adjusted_gross_income
          else:
              0
"""
        )
        test_file = output_file.with_name("151.test.yaml")
        test_file.write_text(
            """- name: single_senior_after_2017_exemption_zero_and_senior_deduction_allowed
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/151#input.filing_status: 0
  output:
    us:statutes/26/151#section_151_deduction: 0
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/151.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/151.yaml: ci: "
                            "Test input assignment missing: "
                            "`single_senior_after_2017_exemption_zero_and_senior_deduction_allowed` "
                            "does not assign #input.adjusted_gross_income, "
                            "#input.dependent_under_section_152, "
                            "#input.tin_included_on_return. Every test for a "
                            "proof-required RuleSpec module must set all local "
                            "factual inputs, including false facts, so tests "
                            "cannot pass through implicit defaults."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_test_input_assignments:"
            "single_senior_after_2017_exemption_zero_and_senior_deduction_allowed"
        ) in output
        assert "outcome=apply_applied final_success=True" in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert "us:statutes/26/151#input.adjusted_gross_income: 0" in test_content
        assert (
            "us:statutes/26/151#input.dependent_under_section_152: false"
            in test_content
        )
        assert "us:statutes/26/151#input.tin_included_on_return: false" in test_content
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_test_input_assignments"] == [
            "single_senior_after_2017_exemption_zero_and_senior_deduction_allowed"
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_hoist_nested_test_tables_normalizes_entity_rows(self, tmp_path):
        test_file = tmp_path / "3121" / "a" / "2.test.yaml"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """- name: sickness_disability_payment_received_under_workmans_compensation_law_excluded
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    tables:
      Payment:
      - us:statutes/26/3121/a/2#input.payment_amount: 1000
        us:statutes/26/3121/a/2#input.payment_received_under_workmans_compensation_law: true
  output:
    us:statutes/26/3121/a/2#employer_plan_sickness_medical_death_payment_excluded_from_wages:
    - 1000
"""
        )

        repaired = _hoist_nested_test_tables(test_file)

        assert repaired == [
            "sickness_disability_payment_received_under_workmans_compensation_law_excluded"
        ]
        payload = yaml.safe_load(test_file.read_text())
        assert payload[0]["input"] == {}
        assert payload[0]["tables"] == {
            "Payment": [
                {
                    "us:statutes/26/3121/a/2#input.payment_amount": 1000,
                    "us:statutes/26/3121/a/2#input.payment_received_under_workmans_compensation_law": True,
                }
            ]
        }

    def test_hoist_nested_test_tables_replaces_empty_existing_tables(self, tmp_path):
        test_file = tmp_path / "3121" / "a" / "2.test.yaml"
        test_file.parent.mkdir(parents=True)
        test_file.write_text(
            """- name: existing_empty_tables
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  tables: {}
  input:
    tables:
      Payment:
      - us:statutes/26/3121/a/2#input.payment_amount: 1000
  output:
    us:statutes/26/3121/a/2#employer_plan_sickness_medical_death_payment_excluded_from_wages:
    - 1000
"""
        )

        repaired = _hoist_nested_test_tables(test_file)

        assert repaired == ["existing_empty_tables"]
        payload = yaml.safe_load(test_file.read_text())
        assert payload[0]["input"] == {}
        assert payload[0]["tables"] == {
            "Payment": [{"us:statutes/26/3121/a/2#input.payment_amount": 1000}]
        }

    def test_encode_apply_auto_repairs_nested_test_tables(self, capsys, tmp_path):
        args = self._make_args(
            tmp_path, backend="codex", citation="26 USC 3121(a)(2)", sync=False
        )
        args.apply = True
        result = self._make_eval_result(False)
        result.citation = "26 USC 3121(a)(2)"
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path
            / "out"
            / "codex-test-model"
            / "statutes"
            / "26"
            / "3121"
            / "a"
            / "2.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_sickness_medical_death_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: payment_amount
"""
        )
        test_file = output_file.with_name("2.test.yaml")
        test_file.write_text(
            """- name: sickness_disability_payment_received_under_workmans_compensation_law_excluded
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    tables:
      Payment:
      - us:statutes/26/3121/a/2#input.payment_amount: 1000
  output:
    us:statutes/26/3121/a/2#employer_plan_sickness_medical_death_payment_excluded_from_wages:
    - 1000
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3121/a/2.yaml"

        def validate_after_repair(*args_, **kwargs):
            payload = yaml.safe_load(test_file.read_text())
            assert payload[0]["input"] == {}
            assert "Payment" in payload[0]["tables"]
            return True, [], {}

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=validate_after_repair,
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
        assert (
            "apply=auto_repaired_nested_test_tables:"
            "sickness_disability_payment_received_under_workmans_compensation_law_excluded"
        ) in output
        assert mock_overlay.call_count == 1
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_nested_test_tables"] == [
            "sickness_disability_payment_received_under_workmans_compensation_law_excluded"
        ]
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_auto_repairs_imported_output_test_mismatches(
        self, capsys, tmp_path
    ):
        args = self._make_args(
            tmp_path, backend="codex", citation="26 USC 1402(b)", sync=False
        )
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path
            / "out"
            / "codex-test-model"
            / "statutes"
            / "26"
            / "1402"
            / "b.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
imports:
  - from: us:statutes/26/1402/a#net_earnings_from_self_employment
rules:
  - name: self_employment_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: net_earnings_from_self_employment
"""
        )
        test_file = output_file.with_name("b.test.yaml")
        test_file.write_text(
            """- name: nonresident_alien_exception_for_us_territory_resident
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 15000
    us:statutes/26/1402/b#input.individual_is_resident_of_us_territory: true
  output:
    us:statutes/26/1402/b#self_employment_income: 15000
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/1402/b.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/1402/b.yaml: ci: "
                            "Test case "
                            "nonresident_alien_exception_for_us_territory_resident "
                            "output "
                            "us:statutes/26/1402/b#self_employment_income "
                            "expected integer 15000, got decimal 13852.5."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_imported_output_tests:"
            "nonresident_alien_exception_for_us_territory_resident"
        ) in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_payload = yaml.safe_load(test_file.read_text())
        assert (
            test_payload[0]["output"]["us:statutes/26/1402/b#self_employment_income"]
            == 13852.5
        )
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_imported_output_tests"] == [
            "nonresident_alien_exception_for_us_territory_resident"
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_auto_repairs_missing_derived_output_coverage(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "151.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: tin_included_on_return_claiming_exemption and is_taxpayer
  - name: section_151_exemption_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxpayer_is_individual:
              exemption_amount
          else:
              0
  - name: exemption_phaseout_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 0
"""
        )
        test_file = output_file.with_name("151.test.yaml")
        test_file.write_text(
            """- name: taxpayer_exemption_eligible
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
    us:statutes/26/151#input.is_taxpayer: true
  output:
    us:statutes/26/151#exemption_individual_eligible: holds
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/151.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/151.yaml: ci: "
                            "Derived rule missing companion output coverage: "
                            "`us:statutes/26/151#exemption_phaseout_applicable_percentage` "
                            "is not asserted by the companion `.test.yaml` file."
                        ],
                        {},
                    ),
                    (
                        False,
                        [
                            "statutes/26/151.yaml: ci: "
                            "Derived rule missing companion output coverage: "
                            "`us:statutes/26/151#section_151_exemption_deduction` "
                            "is not asserted by the companion `.test.yaml` file."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_derived_output_tests:"
            "auto_output_exemption_phaseout_applicable_percentage"
        ) in output
        assert (
            "apply=auto_repaired_derived_output_tests:"
            "auto_output_section_151_exemption_deduction"
        ) in output
        assert mock_overlay.call_count == 3
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert "auto_output_exemption_phaseout_applicable_percentage" in test_content
        assert "auto_output_section_151_exemption_deduction" in test_content
        assert (
            "us:statutes/26/151#exemption_phaseout_applicable_percentage: 0"
            in test_content
        )
        assert "us:statutes/26/151#section_151_exemption_deduction: 0" in test_content
        assert "us:statutes/26/151#input.taxpayer_is_individual: false" in test_content
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_derived_output_tests"] == [
            "auto_output_exemption_phaseout_applicable_percentage",
            "auto_output_section_151_exemption_deduction",
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_derived_output_test_repair_uses_indexed_selector_key(self, tmp_path):
        repo_path = tmp_path / "rulespec-us"
        rules_file = repo_path / "statutes" / "26" / "3241" / "b.yaml"
        test_file = repo_path / "statutes" / "26" / "3241" / "b.test.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
rules:
  - name: average_account_benefits_ratio_bracket
    kind: derived
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: 'if average_account_benefits_ratio < 2.5: 1 else: 2'
  - name: applicable_percentage_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.221
          2: 0.181
"""
        )
        test_file.write_text("[]\n")

        repaired = _append_generated_derived_output_tests_if_missing(
            rules_file=rules_file,
            test_file=test_file,
            repo_path=repo_path,
            relative_output=Path("statutes/26/3241/b.yaml"),
            issues=[
                "Derived rule missing companion output coverage: "
                "`us:statutes/26/3241/b#average_account_benefits_ratio_bracket` "
                "is not asserted by the companion `.test.yaml` file."
            ],
        )

        assert repaired == ["auto_output_average_account_benefits_ratio_bracket"]
        test_payload = yaml.safe_load(test_file.read_text())
        assert test_payload[0]["output"] == {
            "us:statutes/26/3241/b#average_account_benefits_ratio_bracket": 1
        }

    def test_encode_apply_removes_local_import_output_input_placeholders(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "151.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules:
  - name: senior_deduction_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income + amount_excluded_from_gross_income_under_section_931
"""
        )
        test_file = output_file.with_name("151.test.yaml")
        test_file.write_text(
            """- name: modified_agi_without_possession_income
  input:
    us:statutes/26/151#input.adjusted_gross_income: 100
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/931#input.bona_fide_resident_of_specified_possession_during_entire_taxable_year: false
  output:
    us:statutes/26/151#senior_deduction_modified_adjusted_gross_income: 100
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/151.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/151.yaml: ci: Test case "
                            "`modified_agi_without_possession_income` execution "
                            "failed: dataset input "
                            "`us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931` "
                            "must use an absolute legal RuleSpec reference that "
                            "resolves to an input slot, derived rule, or parameter "
                            "in the compiled program"
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_import_output_inputs:"
            "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931"
        ) in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert (
            "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931"
            not in test_content
        )
        assert (
            "us:statutes/26/931#input.bona_fide_resident_of_specified_possession_during_entire_taxable_year"
            in test_content
        )
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_import_output_inputs"] == [
            "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931"
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_removes_invalid_imported_test_inputs(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "63" / "f.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151#exemption_individual_eligible
rules: []
"""
        )
        test_file = output_file.with_name("f.test.yaml")
        invalid_ref = (
            "us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer"
        )
        test_file.write_text(
            f"""- name: spouse_entitlement
  input:
    us:statutes/26/63/f#relation.spouse_for_additional_amount_of_tax_unit:
    - us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
      {invalid_ref}: false
      us:statutes/26/151#input.is_spouse_of_taxpayer: true
  output:
    us:statutes/26/63/f#spouse_aged_additional_amount_entitlement: holds
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/63/f.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/63/f.yaml: ci: Test case "
                            "`spouse_entitlement` input invalid: input "
                            f"`{invalid_ref}` does not resolve to an input slot "
                            "in statutes/26/151.yaml."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert f"apply=auto_repaired_invalid_test_inputs:{invalid_ref}" in output
        assert mock_overlay.call_count == 2
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert invalid_ref not in test_content
        assert (
            "us:statutes/26/151#input.tin_included_on_return_claiming_exemption"
            in test_content
        )
        assert "us:statutes/26/151#input.is_spouse_of_taxpayer" in test_content
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_invalid_test_inputs"] == [invalid_ref]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_removes_unreferenced_proof_imports(self, capsys, tmp_path):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "63" / "f.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151#exemption_individual_eligible
rules:
  - name: spouse_aged_additional_amount_entitlement
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              corpus_citation_path: us/statute/26/63
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/151#exemption_individual_eligible
              output: exemption_individual_eligible
              hash: sha256:abc
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          count_where(spouse_of_taxpayer_for_subsection_f, spouse_aged_person_entitlement) > 0
  - name: spouse_aged_additional_amount_person_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          spouse_has_attained_age_65_before_close_of_taxable_year
"""
        )
        test_file = output_file.with_name("f.test.yaml")
        test_file.write_text("[]\n")
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/63/f.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/63/f.yaml: ci: Proof import not "
                            "referenced: `spouse_aged_additional_amount_entitlement` "
                            "proof imports `exemption_individual_eligible`, but "
                            "the rule formula does not reference that imported "
                            "symbol."
                        ],
                        {},
                    ),
                    (
                        False,
                        [
                            "statutes/26/63/f.yaml: ci: Derived rule missing "
                            "companion output coverage: "
                            "`us:statutes/26/63/f#spouse_aged_additional_amount_person_entitlement` "
                            "is not asserted by the companion `.test.yaml` file."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        repaired = (
            "spouse_aged_additional_amount_entitlement:exemption_individual_eligible"
        )
        pruned_import = "unused_import:us:statutes/26/151#exemption_individual_eligible"
        assert (
            f"apply=auto_repaired_unreferenced_proof_imports:{repaired},{pruned_import}"
        ) in output
        assert (
            "apply=auto_repaired_derived_output_tests:"
            "auto_output_spouse_aged_additional_amount_person_entitlement"
        ) in output
        assert mock_overlay.call_count == 3
        mock_apply.assert_called_once()
        content = yaml.safe_load(output_file.read_text())
        assert not content.get("imports")
        atoms = content["rules"][0]["metadata"]["proof"]["atoms"]
        assert all(atom.get("kind") != "import" for atom in atoms)
        assert (
            "auto_output_spouse_aged_additional_amount_person_entitlement"
            in test_file.read_text()
        )
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_unreferenced_proof_imports"] == [
            repaired,
            pruned_import,
        ]
        assert run.outcome["auto_repaired_derived_output_tests"] == [
            "auto_output_spouse_aged_additional_amount_person_entitlement"
        ]
        assert run.outcome["overlay_validation_success"] is True
        assert run.outcome["status"] == "apply_applied"

    def test_encode_apply_repairs_zero_branch_after_input_assignments(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path
            / "out"
            / "codex-test-model"
            / "statutes"
            / "26"
            / "3121"
            / "a"
            / "1.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: predecessor_remuneration_considered_paid_by_successor
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if successor_employer_wage_base_continuity_applies:
              predecessor_remuneration_before_acquisition
          else:
              0
"""
        )
        test_file = output_file.with_name("1.test.yaml")
        test_file.write_text(
            """- name: no_successor_continuity
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    us:statutes/26/3121/a/1#predecessor_remuneration_considered_paid_by_successor: 100
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/3121/a/1.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/3121/a/1.yaml: ci: "
                            "Test input assignment missing: "
                            "`no_successor_continuity` does not assign "
                            "#input.predecessor_remuneration_before_acquisition, "
                            "#input.successor_employer_wage_base_continuity_applies. "
                            "Every test for a proof-required RuleSpec module must "
                            "set all local factual inputs, including false facts, "
                            "so tests cannot pass through implicit defaults."
                        ],
                        {},
                    ),
                    (
                        False,
                        [
                            "statutes/26/3121/a/1.yaml: ci: "
                            "Zero branch test coverage missing: "
                            "`predecessor_remuneration_considered_paid_by_successor` "
                            "has a formula branch that returns 0."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert (
            "apply=auto_repaired_test_input_assignments:no_successor_continuity"
            in output
        )
        assert (
            "apply=auto_repaired_zero_branch_tests:"
            "auto_zero_predecessor_remuneration_considered_paid_by_successor"
        ) in output
        assert mock_overlay.call_count == 3
        mock_apply.assert_called_once()
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_test_input_assignments"] == [
            "no_successor_continuity"
        ]
        assert run.outcome["auto_repaired_zero_branch_tests"] == [
            "auto_zero_predecessor_remuneration_considered_paid_by_successor"
        ]
        assert run.outcome["status"] == "apply_applied"

    def test_generated_test_input_defaults_do_not_match_rate_inside_separate(self):
        rules_payload = {
            "rules": [
                {
                    "name": "successor_employer_wage_base_continuity_applies",
                    "kind": "derived",
                    "dtype": "Judgment",
                    "versions": [
                        {
                            "formula": (
                                "successor_employer_acquired_substantially_all_property_"
                                "used_in_predecessor_trade_or_business_or_separate_unit_"
                                "during_calendar_year and "
                                "successor_employer_immediately_after_acquisition_"
                                "employs_individual"
                            )
                        }
                    ],
                }
            ]
        }

        value = _default_generated_test_input_value(
            (
                "successor_employer_acquired_substantially_all_property_used_in_"
                "predecessor_trade_or_business_or_separate_unit_during_calendar_year"
            ),
            rules_payload=rules_payload,
        )

        assert value is False

    def test_generated_test_input_defaults_treat_any_wages_fact_as_boolean(self):
        rules_payload = {
            "rules": [
                {
                    "name": "qualified_wages",
                    "kind": "derived",
                    "dtype": "Money",
                    "versions": [
                        {
                            "formula": (
                                "if any_wages_taken_into_account_under_section_51:\n"
                                "    0\n"
                                "else:\n"
                                "    wages_paid"
                            )
                        }
                    ],
                }
            ]
        }

        value = _default_generated_test_input_value(
            "any_wages_taken_into_account_under_section_51",
            rules_payload=rules_payload,
        )

        assert value is False

    def test_generated_test_input_defaults_treat_embedded_has_fact_as_boolean(self):
        rules_payload = {
            "rules": [
                {
                    "name": "taxpayer_aged_additional_amount_entitlement",
                    "kind": "derived",
                    "dtype": "Judgment",
                    "versions": [
                        {
                            "formula": (
                                "taxpayer_has_attained_age_65_before_close_of_"
                                "taxable_year"
                            )
                        }
                    ],
                }
            ]
        }

        value = _default_generated_test_input_value(
            "taxpayer_has_attained_age_65_before_close_of_taxable_year",
            rules_payload=rules_payload,
        )

        assert value is False

    def test_generated_test_input_defaults_treat_condition_wages_fact_as_boolean(
        self,
    ):
        rules_payload = {
            "rules": [
                {
                    "name": "qualified_wages",
                    "kind": "derived",
                    "dtype": "Money",
                    "versions": [
                        {
                            "formula": (
                                "if wages_taken_into_account_under_section_51:\n"
                                "    0\n"
                                "else:\n"
                                "    wages_paid"
                            )
                        }
                    ],
                }
            ]
        }

        value = _default_generated_test_input_value(
            "wages_taken_into_account_under_section_51",
            rules_payload=rules_payload,
        )

        assert value is False

    def test_generated_test_input_defaults_treat_remuneration_as_money(self):
        rules_payload = {
            "rules": [
                {
                    "name": "predecessor_remuneration_considered_paid_by_successor",
                    "kind": "derived",
                    "dtype": "Money",
                    "versions": [
                        {
                            "formula": (
                                "if successor_employer_wage_base_continuity_applies:\n"
                                "    predecessor_remuneration_before_acquisition\n"
                                "else:\n"
                                "    0"
                            )
                        }
                    ],
                }
            ]
        }

        value = _default_generated_test_input_value(
            "predecessor_remuneration_before_acquisition",
            rules_payload=rules_payload,
        )

        assert value == 0

    def test_encode_apply_repeats_missing_test_input_assignment_repairs(
        self, capsys, tmp_path
    ):
        args = self._make_args(tmp_path, backend="codex", sync=False)
        args.apply = True
        result = self._make_eval_result(False)
        result.error = "Generated RuleSpec failed CI validation"
        output_file = (
            tmp_path / "out" / "codex-test-model" / "statutes" / "26" / "6012.yaml"
        )
        output_file.parent.mkdir(parents=True)
        output_file.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: filing_requirement
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          gross_income > taxable_income
"""
        )
        test_file = output_file.with_name("6012.test.yaml")
        test_file.write_text(
            """- name: first_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/6012#input.gross_income: 1
  output:
    us:statutes/26/6012#filing_requirement: holds
- name: second_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/6012#input.gross_income: 2
  output:
    us:statutes/26/6012#filing_requirement: holds
"""
        )
        result.output_file = str(output_file)
        applied_file = args.policy_repo_path / "statutes/26/6012.yaml"

        with (
            patch("axiom_encode.cli.run_model_eval", return_value=[result]),
            patch(
                "axiom_encode.cli._validate_generated_encoding_in_policy_overlay",
                side_effect=[
                    (
                        False,
                        [
                            "statutes/26/6012.yaml: ci: "
                            "Test input assignment missing: `first_case` does not "
                            "assign #input.taxable_income. Every test for a "
                            "proof-required RuleSpec module must set all local "
                            "factual inputs, including false facts, so tests "
                            "cannot pass through implicit defaults."
                        ],
                        {},
                    ),
                    (
                        False,
                        [
                            "statutes/26/6012.yaml: ci: "
                            "Test input assignment missing: `second_case` does not "
                            "assign #input.taxable_income. Every test for a "
                            "proof-required RuleSpec module must set all local "
                            "factual inputs, including false facts, so tests "
                            "cannot pass through implicit defaults."
                        ],
                        {},
                    ),
                    (True, [], {}),
                ],
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
        assert "apply=auto_repaired_test_input_assignments:first_case" in output
        assert "apply=auto_repaired_test_input_assignments:second_case" in output
        assert mock_overlay.call_count == 3
        mock_apply.assert_called_once()
        test_content = test_file.read_text()
        assert test_content.count("us:statutes/26/6012#input.taxable_income: 0") == 2
        run = EncodingDB(args.db).get_recent_runs(limit=1)[0]
        assert run.outcome["auto_repaired_test_input_assignments"] == [
            "first_case",
            "second_case",
        ]
        assert run.outcome["status"] == "apply_applied"

    def test_overlay_validation_suppresses_ancestor_targets_for_subsection_migration(
        self, tmp_path
    ):
        repo = tmp_path / "rulespec-us"
        ancestor = repo / "statutes/26/151.yaml"
        ancestor_test = repo / "statutes/26/151.test.yaml"
        target = repo / "statutes/26/151/d.yaml"
        sibling = repo / "statutes/26/152.yaml"
        for path in (ancestor, ancestor_test, target, sibling):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("format: rulespec/v1\nrules: []\n")

        suppressed = _suppress_rulespec_ancestor_targets_for_subsection_overlay(
            repo,
            Path("statutes/26/151/d.yaml"),
        )

        assert Path("statutes/26/151.yaml") in suppressed
        assert Path("statutes/26/151.test.yaml") in suppressed
        assert not ancestor.exists()
        assert not ancestor_test.exists()
        assert target.exists()
        assert sibling.exists()

    def test_overlay_validation_preserves_imported_ancestor_target(self, tmp_path):
        repo = tmp_path / "rulespec-us"
        ancestor = repo / "statutes/26/163/h/4/C.yaml"
        ancestor_test = repo / "statutes/26/163/h/4/C.test.yaml"
        target = repo / "statutes/26/163/h/4/C/ii/I.yaml"
        for path in (ancestor, ancestor_test, target):
            path.parent.mkdir(parents=True, exist_ok=True)
        ancestor.write_text("format: rulespec/v1\nrules: []\n")
        ancestor_test.write_text("[]\n")
        target.write_text(
            "format: rulespec/v1\n"
            "imports:\n"
            "  - us:statutes/26/163/h/4/C#interest_taken_into_account_after_dollar_limit\n"
            "rules: []\n"
        )

        suppressed = _suppress_rulespec_ancestor_targets_for_subsection_overlay(
            repo,
            Path("statutes/26/163/h/4/C/ii/I.yaml"),
        )

        assert Path("statutes/26/163/h/4/C.yaml") not in suppressed
        assert Path("statutes/26/163/h/4/C.test.yaml") not in suppressed
        assert ancestor.exists()
        assert ancestor_test.exists()

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

    def test_mixed_scalar_output_test_repair_splits_parameter_outputs(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        rules_file = policy_repo / "statutes/26/164/f.yaml"
        test_file = policy_repo / "statutes/26/164/f.test.yaml"
        rules_file.parent.mkdir(parents=True)
        rules_file.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_tax_deduction_fraction
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: 1 / 2
  - name: self_employment_tax_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: self_employment_tax * self_employment_tax_deduction_fraction
"""
        )
        test_file.write_text(
            """- name: zero_liability_individual
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/164/f#input.self_employment_tax: 0
  output:
    us:statutes/26/164/f#self_employment_tax_deduction_fraction: 0.5
    us:statutes/26/164/f#self_employment_tax_deduction: 0
"""
        )

        repaired = _repair_mixed_scalar_output_tests(
            rules_file=rules_file,
            test_file=test_file,
            repo_path=policy_repo,
            relative_output=Path("statutes/26/164/f.yaml"),
        )

        repaired_content = test_file.read_text()
        assert "&id" not in repaired_content
        assert "*id" not in repaired_content
        cases = yaml.safe_load(test_file.read_text())
        assert repaired == ["zero_liability_individual"]
        assert cases == [
            {
                "name": "zero_liability_individual",
                "period": {
                    "period_kind": "tax_year",
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                "input": {"us:statutes/26/164/f#input.self_employment_tax": 0},
                "output": {"us:statutes/26/164/f#self_employment_tax_deduction": 0},
            },
            {
                "name": "zero_liability_individual_scalar_outputs",
                "period": {
                    "period_kind": "tax_year",
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                "input": {"us:statutes/26/164/f#input.self_employment_tax": 0},
                "output": {
                    "us:statutes/26/164/f#self_employment_tax_deduction_fraction": 0.5
                },
            },
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

    def test_rejects_existing_rulespec_without_encoder_manifest_in_all_mode(
        self, tmp_path
    ):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")

        issues = guard_generated_change_issues(
            tmp_path,
            roots=("regulations",),
            all_files=True,
        )

        assert issues == [
            "regulations/example.yaml is missing a matching .axiom/encoding-manifests manifest"
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

    def test_accepts_rulespec_change_when_any_changed_manifest_matches(self, tmp_path):
        rule = tmp_path / "regulations/example.yaml"
        rule.parent.mkdir(parents=True)
        rule.write_text("format: rulespec/v1\nrules: []\n")
        current_hash = _sha256_file(rule)

        repair_manifest = tmp_path / ".axiom/encoding-manifests/regulations/repair.json"
        owner_manifest = tmp_path / ".axiom/encoding-manifests/regulations/example.json"
        repair_manifest.parent.mkdir(parents=True)
        repair_manifest.write_text(
            json.dumps(
                _signed_manifest_payload(
                    {
                        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                        "applied_files": [
                            {
                                "path": "regulations/example.yaml",
                                "sha256": current_hash,
                            }
                        ],
                    }
                )
            )
            + "\n"
        )
        owner_manifest.write_text(
            json.dumps(
                _signed_manifest_payload(
                    {
                        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                        "applied_files": [
                            {
                                "path": "regulations/example.yaml",
                                "sha256": "stale-owner-hash",
                            }
                        ],
                    }
                )
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
                    ".axiom/encoding-manifests/regulations/repair.json",
                    ".axiom/encoding-manifests/regulations/example.json",
                ],
            )

        assert issues == []

    def test_accepts_existing_rulespec_with_matching_encoder_manifest_in_all_mode(
        self, tmp_path
    ):
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
                roots=("regulations",),
                all_files=True,
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
    def test_repair_generated_import_symbol_near_miss_updates_imports_and_formulas(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/1401/a.yaml"
        generated = policy_repo / "statutes/26/164/f.yaml"
        imported.parent.mkdir(parents=True)
        generated.parent.mkdir(parents=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: 0
"""
        )
        generated.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1401/a#old_age_survivors_and_disability_tax
rules:
  - name: self_employment_tax_deduction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: 'if taxpayer_is_individual: self_employment_tax_deduction_fraction *
          (old_age_survivors_and_disability_tax + self_employment_income_tax) else: 0'
    metadata:
      proof:
        atoms:
          - kind: import
            import:
              target: us:statutes/26/1401/a
              output: old_age_survivors_and_disability_tax
              hash: sha256:abc123
"""
        )

        repaired = _repair_generated_import_symbol_near_misses(
            rules_file=generated,
            repo_path=policy_repo,
        )

        content = generated.read_text()
        assert repaired == [
            "old_age_survivors_and_disability_tax"
            "->old_age_survivors_and_disability_insurance_tax"
        ]
        assert "old_age_survivors_and_disability_tax" not in content
        assert (
            "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax"
            in content
        )
        assert (
            "old_age_survivors_and_disability_insurance_tax + "
            "self_employment_income_tax" in content
        )
        assert "output: old_age_survivors_and_disability_insurance_tax" in content

    def test_repair_generated_import_symbol_near_miss_requires_clear_match(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/9999.yaml"
        generated = policy_repo / "statutes/26/8888.yaml"
        imported.parent.mkdir(parents=True)
        generated.parent.mkdir(parents=True, exist_ok=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: foo_a_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-01-01'
        formula: 0
  - name: foo_b_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-01-01'
        formula: 0
"""
        )
        generated.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/9999#foo_tax
rules: []
"""
        )
        original = generated.read_text()

        repaired = _repair_generated_import_symbol_near_misses(
            rules_file=generated,
            repo_path=policy_repo,
        )

        assert repaired == []
        assert generated.read_text() == original

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

    def test_local_factual_input_names_ignore_relation_aggregator_functions(self):
        content = """format: rulespec/v1
rules:
  - name: child_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: child_of_tax_unit
      arity: 2
      arguments:
        - TaxUnit
        - Person
  - name: eligible_child_count
    kind: derived
    entity: TaxUnit
    dtype: Number
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          len(child_of_tax_unit)
          + count_where(child_of_tax_unit, child_is_eligible)
"""

        inputs = _local_factual_input_names_from_rules_content(content)

        assert "len" not in inputs
        assert "count_where" not in inputs
        assert inputs == {"child_is_eligible"}

    def test_local_factual_input_names_ignore_imported_output_fragments(self):
        content = """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules:
  - name: modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          adjusted_gross_income
          + amount_excluded_from_gross_income_under_section_931
"""

        inputs = _local_factual_input_names_from_rules_content(content)

        assert inputs == {"adjusted_gross_income"}

    def test_complete_missing_imported_test_inputs_adds_import_defaults(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/1/h.yaml"
        dependent = policy_repo / "statutes/26/199A.yaml"
        dependent_test = policy_repo / "statutes/26/199A.test.yaml"
        imported.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True, exist_ok=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: net_capital_gain
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, long_term_capital_gains - investment_income_amount)
"""
        )
        dependent.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1/h#net_capital_gain
rules: []
"""
        )
        dependent_test.write_text(
            """- name: case_one
  input:
    us:statutes/26/1/h#input.long_term_capital_gains: 0
  output:
    us:statutes/26/199A#qbi_taxable_income_limit: 20000
- name: case_two
  input:
    us:statutes/26/1/h#input.long_term_capital_gains: 1000
  output:
    us:statutes/26/199A#qbi_taxable_income_limit: 20000
"""
        )
        validation = SimpleNamespace(
            results={
                "ci": SimpleNamespace(
                    error=(
                        "Test case `case_one` execution failed: "
                        "missing input `investment_income_amount`"
                    )
                )
            }
        )

        changed = _complete_missing_imported_test_inputs(
            rules_file=dependent,
            test_file=dependent_test,
            repo_path=policy_repo,
            validation=validation,
        )

        assert changed is True
        updated = dependent_test.read_text()
        assert (
            updated.count("us:statutes/26/1/h#input.investment_income_amount: 0") == 2
        )

    def test_complete_missing_imported_test_inputs_adds_transitive_import_defaults(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        leaf = policy_repo / "statutes/26/1402/b.yaml"
        middle = policy_repo / "statutes/26/1401/a.yaml"
        dependent = policy_repo / "statutes/26/164/f.yaml"
        dependent_test = policy_repo / "statutes/26/164/f.test.yaml"
        leaf.parent.mkdir(parents=True)
        middle.parent.mkdir(parents=True, exist_ok=True)
        dependent.parent.mkdir(parents=True, exist_ok=True)
        leaf.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_is_nonresident_alien: 0
          else: net_earnings_from_self_employment
"""
        )
        middle.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1402/b#self_employment_income_for_section_1401_a
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: self_employment_income_for_section_1401_a * 0.124
"""
        )
        dependent.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax
rules: []
"""
        )
        dependent_test.write_text(
            """- name: case_one
  input:
    us:statutes/26/164/f#input.taxpayer_is_individual: true
  output:
    us:statutes/26/164/f#self_employment_tax_deduction: 0
"""
        )
        validation = SimpleNamespace(
            results={
                "ci": SimpleNamespace(
                    error=(
                        "Test case `case_one` execution failed: "
                        "missing input `individual_is_nonresident_alien`"
                    )
                )
            }
        )

        changed = _complete_missing_imported_test_inputs(
            rules_file=dependent,
            test_file=dependent_test,
            repo_path=policy_repo,
            validation=validation,
        )

        assert changed is True
        updated = dependent_test.read_text()
        assert (
            "us:statutes/26/1402/b#input.individual_is_nonresident_alien: false"
            in updated
        )

    def test_repair_imported_test_inputs_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/1/h.yaml"
        target = policy_repo / "statutes/26/1/j.yaml"
        test_file = policy_repo / "statutes/26/1/j.test.yaml"
        imported.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: capital_gains_excluded_from_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          long_term_capital_gains - net_capital_gain_taken_into_account_as_investment_income_under_section_163_d_4_B_iii
"""
        )
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1/h
rules: []
"""
        )
        test_file.write_text(
            """- name: ordinary_income_case
  input:
    us:statutes/26/1/h#input.long_term_capital_gains: 0
  output: {}
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/1/j.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            calls = 0

            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                FakePipeline.calls += 1
                if FakePipeline.calls == 1:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `ordinary_income_case` execution failed: "
                                    "missing input `net_capital_gain_taken_into_account_as_investment_income_under_section_163_d_4_B_iii`"
                                )
                            )
                        },
                    )
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
            cmd_repair_imported_test_inputs(args)

        updated = test_file.read_text()
        assert (
            "us:statutes/26/1/h#input."
            "net_capital_gain_taken_into_account_as_investment_income_under_section_163_d_4_B_iii: 0"
            in updated
        )
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/1/j.json"
        payload = json.loads(manifest.read_text())
        assert payload["model"] == "imported-test-inputs-v1"
        assert payload["tool"] == "axiom-encode repair-imported-test-inputs"

    def test_repair_imported_test_inputs_removes_import_output_placeholders(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/224.yaml"
        target = policy_repo / "statutes/26/63.yaml"
        test_file = policy_repo / "statutes/26/63.test.yaml"
        imported.parent.mkdir(parents=True)
        imported.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules: []
"""
        )
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/224
rules: []
"""
        )
        test_file.write_text(
            """- name: transitive_case
  input:
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/63#input.adjusted_gross_income: 100000
  output: {}
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/63.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            calls = 0

            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                FakePipeline.calls += 1
                if FakePipeline.calls == 1:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `transitive_case` execution failed: "
                                    "dataset input "
                                    "`us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931` "
                                    "must use an absolute legal RuleSpec reference "
                                    "that resolves to an input slot, derived rule, "
                                    "or parameter in the compiled program"
                                )
                            )
                        },
                    )
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
            cmd_repair_imported_test_inputs(args)

        updated = test_file.read_text()
        assert (
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
            not in updated
        )
        assert "us:statutes/26/63#input.adjusted_gross_income: 100000" in updated
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/63.json"
        payload = json.loads(manifest.read_text())
        assert payload["model"] == "imported-test-inputs-v1"

    def test_repair_imported_test_inputs_removes_stale_invalid_refs_then_fills_missing_imports(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/26/1401/b/1.yaml"
        target = policy_repo / "statutes/26/32/c/2.yaml"
        test_file = policy_repo / "statutes/26/32/c/2.test.yaml"
        imported.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: self_employment_income_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: self_employment_income * 0.029
"""
        )
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1401/b/1#self_employment_income_tax
rules: []
"""
        )
        test_file.write_text(
            """- name: transitive_case
  input:
    us:statutes/26/1401#input.self_employment_income: 1000
    us:statutes/26/32/c/2#input.taxpayer_is_individual: true
  output: {}
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/32/c/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            calls = 0

            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                FakePipeline.calls += 1
                if FakePipeline.calls == 1:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `transitive_case` execution failed: "
                                    "dataset input "
                                    "`us:statutes/26/1401#input.self_employment_income` "
                                    "must use an absolute legal RuleSpec reference "
                                    "that resolves to an input slot, derived rule, "
                                    "or parameter in the compiled program"
                                )
                            )
                        },
                    )
                if FakePipeline.calls == 2:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `transitive_case` execution failed: "
                                    "missing input `self_employment_income`"
                                )
                            )
                        },
                    )
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
            cmd_repair_imported_test_inputs(args)

        updated = test_file.read_text()
        assert "us:statutes/26/1401#input.self_employment_income" not in updated
        assert "us:statutes/26/32/c/2#input.taxpayer_is_individual: true" in updated
        assert "us:statutes/26/1401/b/1#input.self_employment_income: 0" in updated
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/32/c/2.json"
        payload = json.loads(manifest.read_text())
        assert payload["model"] == "imported-test-inputs-v1"

    def test_repair_imported_test_inputs_rewrites_stale_exclusion_inputs(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/63.yaml"
        test_file = policy_repo / "statutes/26/63.test.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/224
  - us:statutes/26/225
rules: []
"""
        )
        test_file.write_text(
            """- name: transitive_case
  input:
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_911: 0
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/225#input.amount_excluded_from_gross_income_under_section_933: 0
    us:statutes/26/63#input.adjusted_gross_income: 100000
  output: {}
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/63.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            calls = 0

            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                FakePipeline.calls += 1
                if FakePipeline.calls == 1:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `transitive_case` input invalid: "
                                    "input "
                                    "`us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_911` "
                                    "does not resolve to an input slot; "
                                    "input "
                                    "`us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931` "
                                    "does not resolve to an input slot; "
                                    "input "
                                    "`us:statutes/26/225#input.amount_excluded_from_gross_income_under_section_933` "
                                    "does not resolve to an input slot"
                                )
                            )
                        },
                    )
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
            cmd_repair_imported_test_inputs(args)

        updated = test_file.read_text()
        assert (
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_911"
            not in updated
        )
        assert (
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
            not in updated
        )
        assert (
            "us:statutes/26/225#input.amount_excluded_from_gross_income_under_section_933"
            not in updated
        )
        assert (
            "us:statutes/26/911/a/1#input.elected_foreign_earned_income_exclusion_amount: 0"
            in updated
        )
        assert (
            "us:statutes/26/931#input.gross_income_excluded_under_section_931: 0"
            in updated
        )
        assert (
            "us:statutes/26/933#input.gross_income_excluded_under_section_933: 0"
            in updated
        )
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/63.json"
        payload = json.loads(manifest.read_text())
        assert payload["model"] == "imported-test-inputs-v1"

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

    def test_apply_overlay_validation_allows_deferred_replacement_of_executable_file(
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

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

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

    def test_apply_overlay_validation_skips_sibling_with_active_canonical_name(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us-worktree"
        sibling_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/3201.yaml"
        policy_repo.mkdir()
        sibling_repo.mkdir()
        generated.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **kwargs):
                assert Path(kwargs["policy_repo_path"]).name == "rulespec-us"

            def validate(self, _path, *, skip_reviewers):
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch(
                "axiom_encode.cli.canonical_rulespec_repo_name",
                return_value="rulespec-us",
            ),
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
        ):
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
            )

        assert ok is True
        assert issues == []
        assert supplemental == {}

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
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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
        manifest = (
            policy_repo / ".axiom/encoding-manifests/regulations/7-cfr/273/10.json"
        )
        payload = json.loads(manifest.read_text())
        assert payload["schema_version"] == APPLIED_ENCODING_MANIFEST_SCHEMA
        assert payload["backend"] == "deterministic"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "regulations/7-cfr/273/10.yaml",
        ]
        assert payload["tool"] == "axiom-encode repair-nonnegative-floors"

    def test_repair_current_year_final_amounts_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        imported = (
            policy_repo / "policies/irs/rev-proc-2025-32/earned-income-credit.yaml"
        )
        imported.parent.mkdir(parents=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: eitc_maximum_credit_amounts
    kind: parameter
    dtype: Money
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 664
          1: 4427
"""
        )
        target = policy_repo / "statutes/26/32.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
imports:
  - us:policies/irs/rev-proc-2025-32/earned-income-credit
rules:
  - name: eitc_capped_child_count
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(eitc_child_count, 3)
  - name: eitc_maximum
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/32
              text: "credit percentage of the earned income amount"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * eitc_earned_income_amount
"""
        )
        test_file = policy_repo / "statutes/26/32.test.yaml"
        test_file.write_text(
            """- name: one_child_phase_in
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  output:
    us:statutes/26/32#eitc_capped_child_count: 1
    us:statutes/26/32#eitc_maximum: 4426.8
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/32.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(
                    all_passed=False,
                    results={
                        "ci": SimpleNamespace(
                            error=(
                                "Nonnegative amount income base missing floor: "
                                "`eitc_phased_in`"
                            )
                        )
                    },
                )

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
            cmd_repair_current_year_final_amounts(args)

        content = target.read_text()
        assert isinstance(yaml.safe_load(content), dict)
        assert "match eitc_capped_child_count:" in content
        assert "1 => eitc_maximum_credit_amounts[1]" in content
        assert "output: eitc_maximum_credit_amounts" in content
        assert "us:statutes/26/32#eitc_maximum: 4427" in test_file.read_text()
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/32.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/32.yaml",
            "statutes/26/32.test.yaml",
        ]
        assert payload["tool"] == "axiom-encode repair-current-year-final-amounts"

    def test_repair_nonnegative_floors_taxable_income_test_checks_final_output_only(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/63.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: taxable_income_for_individual_who_does_not_itemize else: taxable_income_general_rule
"""
        )
        test_file = policy_repo / "statutes/26/63.test.yaml"
        test_file.write_text(
            """- name: existing_positive_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/63#input.gross_income: 100000
  output:
    us:statutes/26/63#taxable_income: 100000
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/63.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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

    def test_repair_nonnegative_floors_manifest_keeps_unchanged_companion_test(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/999.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: example_credit_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          maximum_credit_amount - adjusted_gross_income
"""
        )
        test_file = policy_repo / "statutes/26/999.test.yaml"
        test_file.write_text(
            """- name: existing_case
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/999#input.maximum_credit_amount: 1000
    us:statutes/26/999#input.adjusted_gross_income: 200
  output:
    us:statutes/26/999#example_credit_amount: 800
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/999.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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

        assert (
            "max(0, maximum_credit_amount - adjusted_gross_income)"
            in target.read_text()
        )
        assert test_file.read_text().startswith("- name: existing_case")
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/999.json"
        payload = json.loads(manifest.read_text())
        assert [item["path"] for item in payload["applied_files"]] == [
            "statutes/26/999.yaml",
        ]

    def test_has_zero_output_test_requires_exact_legal_id(self):
        cases = [
            {
                "output": {
                    "us:statutes/26/1#taxable_income": 0,
                    "taxable_income": 0,
                }
            }
        ]

        assert _has_zero_output_test(cases, "us:statutes/26/63#taxable_income") is False
        assert _has_zero_output_test(cases, "us:statutes/26/1#taxable_income") is True

    def test_repair_tax_filing_status_branches_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/3101/b/2.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  summary: joint / surviving spouse and any other case
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          match filing_status:
              1 => additional_medicare_wage_tax_joint_threshold
              2 => additional_medicare_wage_tax_joint_threshold / 2
              0 => additional_medicare_wage_tax_other_threshold
"""
        )
        test_file = policy_repo / "statutes/26/3101/b/2.test.yaml"
        test_file.write_text(
            """- name: joint_above_threshold
  period:
    period_kind: tax_year
    start: 2026-01-01
    end: 2026-12-31
  input:
    us:statutes/26/3101/b/2#input.filing_status: 1
    us:statutes/26/3101/b/2#input.wages: 300000
  output:
    us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold: 250000
    us:statutes/26/3101/b/2#additional_medicare_excess_wages: 50000
    us:statutes/26/3101/b/2#additional_medicare_tax: 450
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/3101/b/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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
            cmd_repair_tax_filing_status_branches(args)

        content = target.read_text()
        assert "4 => additional_medicare_wage_tax_joint_threshold" in content
        assert "3 => additional_medicare_wage_tax_other_threshold" in content
        test_content = test_file.read_text()
        assert "surviving_spouse_uses_joint_threshold" not in test_content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/3101/b/2.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "tax-filing-status-branch-v1"
        assert payload["tool"] == "axiom-encode repair-tax-filing-status-branches"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/3101/b/2.yaml"
        ]

    def test_repair_tax_filing_status_branches_does_not_mutate_without_signing_key(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/3101/b/2.yaml"
        target.parent.mkdir(parents=True)
        original_content = """format: rulespec/v1
module:
  summary: joint / surviving spouse and any other case
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          match filing_status:
              1 => additional_medicare_wage_tax_joint_threshold
              2 => additional_medicare_wage_tax_joint_threshold / 2
              0 => additional_medicare_wage_tax_other_threshold
"""
        target.write_text(original_content)
        test_file = policy_repo / "statutes/26/3101/b/2.test.yaml"
        original_test_content = """- name: joint_above_threshold
  output:
    us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold: 250000
"""
        test_file.write_text(original_test_content)
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/3101/b/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match=APPLIED_ENCODING_SIGNING_KEY_ENV),
        ):
            cmd_repair_tax_filing_status_branches(args)

        assert target.read_text() == original_content
        assert test_file.read_text() == original_test_content

    def test_repair_tax_status_components_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "policies/irs/rev-proc-2025-32/standard-deduction.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: additional_standard_deduction_for_aged_or_blind
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxable_year_begins_after_2017:
              if individual_is_unmarried_and_not_surviving_spouse: higher_amount else: regular_amount
          else:
              regular_amount
"""
        )
        test_file = target.with_name("standard-deduction.test.yaml")
        test_file.write_text(
            """- name: head_of_household
  input:
    us:policies/irs/rev-proc-2025-32/standard-deduction#input.filing_status: 3
    us:policies/irs/rev-proc-2025-32/standard-deduction#input.individual_is_unmarried_and_not_surviving_spouse: true
    us:policies/irs/rev-proc-2025-32/standard-deduction#input.taxable_year_begins_after_2017: true
  output:
    us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_for_aged_or_blind: 2050
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("policies/irs/rev-proc-2025-32/standard-deduction.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                assert "individual_is_unmarried" not in target.read_text()
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch("axiom_encode.cli._companion_test_issues", return_value=[]),
            patch(
                "axiom_encode.cli._ensure_no_unmanifested_preexisting_rulespec_changes"
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_tax_status_components(args)

        content = target.read_text()
        assert "individual_is_unmarried_and_not_surviving_spouse" not in content
        assert "filing_status == 0 or filing_status == 3" in content
        assert "taxable_year_begins_after_2017" not in content
        assert (
            "taxable_year_begins_after_tcja_exemption_amount_zero_effective_date"
            in content
        )
        test_content = test_file.read_text()
        assert (
            "input.individual_is_unmarried_and_not_surviving_spouse" not in test_content
        )
        assert "input.taxable_year_begins_after_2017" not in test_content
        assert (
            "input.taxable_year_begins_after_tcja_exemption_amount_zero_effective_date"
            in test_content
        )
        manifest = (
            policy_repo
            / ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/standard-deduction.json"
        )
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "tax-status-component-v1"
        assert payload["tool"] == "axiom-encode repair-tax-status-components"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "policies/irs/rev-proc-2025-32/standard-deduction.yaml",
            "policies/irs/rev-proc-2025-32/standard-deduction.test.yaml",
        ]

    def test_repair_tax_status_components_allows_preexisting_test_failures(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "policies/irs/rev-proc-2025-32/standard-deduction.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: additional_standard_deduction_for_aged_or_blind
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_is_unmarried_and_not_surviving_spouse: higher_amount else: regular_amount
"""
        )
        test_file = target.with_name("standard-deduction.test.yaml")
        test_file.write_text(
            """- name: legacy_companion_shape
  input:
    us:policies/irs/rev-proc-2025-32/standard-deduction#input.filing_status: 3
    us:policies/irs/rev-proc-2025-32/standard-deduction#input.individual_is_unmarried_and_not_surviving_spouse: true
  output:
    us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_for_aged_or_blind: 2050
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("policies/irs/rev-proc-2025-32/standard-deduction.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )
        preexisting_issue = (
            "policies/irs/rev-proc-2025-32/standard-deduction.test.yaml :: "
            "legacy_companion_shape: unknown executable output "
            "us:policies/irs/rev-proc-2025-32/standard-deduction"
            "#additional_standard_deduction_for_aged_or_blind"
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                assert "individual_is_unmarried" not in target.read_text()
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._companion_test_issues",
                side_effect=[[preexisting_issue], [preexisting_issue]],
            ) as companion_tests,
            patch(
                "axiom_encode.cli._ensure_no_unmanifested_preexisting_rulespec_changes"
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_tax_status_components(args)

        assert companion_tests.call_count == 2
        assert (
            "individual_is_unmarried_and_not_surviving_spouse" not in target.read_text()
        )
        assert (
            "input.individual_is_unmarried_and_not_surviving_spouse"
            not in test_file.read_text()
        )

    def test_repair_tax_status_temporal_rewrite_only_updates_formulas(self):
        content = """format: rulespec/v1
module:
  summary: taxable_year_begins_after_2017 remains historical prose.
rules:
  - name: exemption_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              text: taxable_year_begins_after_2017 remains quoted source text
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxable_year_begins_after_2017: 0 else: 2000
"""

        repaired, repairs = _repair_section_151_temporal_fact_names(content)

        assert repairs == [
            "taxable_year_begins_after_2017->taxable_year_begins_after_tcja_exemption_amount_zero_effective_date"
        ]
        assert (
            "summary: taxable_year_begins_after_2017 remains historical prose."
            in repaired
        )
        assert (
            "text: taxable_year_begins_after_2017 remains quoted source text"
            in repaired
        )
        assert (
            "if taxable_year_begins_after_tcja_exemption_amount_zero_effective_date: 0 else: 2000"
            in repaired
        )

    def test_repair_tax_status_components_keeps_unchanged_test_in_manifest(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/151.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: exemption_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: if taxable_year_begins_after_2017: 0 else: 5000
"""
        )
        test_file = target.with_name("151.test.yaml")
        test_file.write_text(
            """- name: unchanged_test
  input:
    us:statutes/26/151#input.adjusted_gross_income: 0
  output:
    us:statutes/26/151#exemption_amount: 0
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/151.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch("axiom_encode.cli._companion_test_issues", return_value=[]),
            patch(
                "axiom_encode.cli._ensure_no_unmanifested_preexisting_rulespec_changes"
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_tax_status_components(args)

        assert test_file.read_text().startswith("- name: unchanged_test")
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/151.json"
        payload = json.loads(manifest.read_text())
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/151.yaml",
            "statutes/26/151.test.yaml",
        ]

    def test_repair_tax_status_components_rejects_dirty_related_tests(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/151.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: exemption_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: if taxable_year_begins_after_2017: 0 else: 5000
"""
        )
        target.with_name("151.test.yaml").write_text(
            """- name: exemption_amount
  input:
    us:statutes/26/151#input.taxable_year_begins_after_2017: true
  output:
    us:statutes/26/151#exemption_amount: 0
"""
        )
        related_test = policy_repo / "statutes/26/63.test.yaml"
        related_test.write_text(
            """- name: unrelated_baseline
  input:
    us:example#input.value: 1
  output:
    us:example#output.value: 1
"""
        )
        subprocess.run(
            ["git", "init"], cwd=policy_repo, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(["git", "add", "statutes"], cwd=policy_repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Axiom Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=policy_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        dirty_related_content = """- name: dirty_related_151_case
  input:
    us:statutes/26/151#input.taxable_year_begins_after_2017: true
  output:
    us:example#output.value: 1
"""
        related_test.write_text(dirty_related_content)
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/151.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        with (
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
            pytest.raises(
                RuntimeError,
                match="Refusing to sign over pre-existing RuleSpec target changes",
            ) as excinfo,
        ):
            cmd_repair_tax_status_components(args)

        assert ".axiom/encoding-manifests/statutes/26/151.json" in str(excinfo.value)
        assert related_test.read_text() == dirty_related_content

    def test_generated_dependency_guard_rejects_unmanifested_existing_file(
        self, tmp_path
    ):
        repo = tmp_path / "rulespec-us"
        target = repo / "statutes/26/931.yaml"
        target.parent.mkdir(parents=True)
        target.write_text("format: rulespec/v1\nrules: []\n")

        with pytest.raises(
            RuntimeError,
            match="Refusing to overwrite existing RuleSpec dependency files",
        ):
            _ensure_generated_dependency_files_safe(
                repo,
                generated_files={
                    target: "format: rulespec/v1\nrules:\n  - name: generated\n"
                },
            )

    def test_repair_tax_status_components_defers_151_68b_phaseout(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/151.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/151
  summary: |-
    Section 151(d)(3) refers to the applicable amount in effect under section 68(b).
rules:
  - name: exemption_phaseout_rate_per_increment
    kind: parameter
    dtype: Rate
    source: 26 USC 151(d)(3)(B)
    versions:
      - effective_from: '2026-01-01'
        formula: 0.02
  - name: exemption_phaseout_increment
    kind: parameter
    dtype: Money
    unit: USD
    source: 26 USC 151(d)(3)(B)
    versions:
      - effective_from: '2026-01-01'
        formula: 2500
  - name: exemption_phaseout_increment_separate
    kind: parameter
    dtype: Money
    unit: USD
    source: 26 USC 151(d)(3)(B)
    versions:
      - effective_from: '2026-01-01'
        formula: 1250
  - name: exemption_phaseout_maximum_percentage
    kind: parameter
    dtype: Rate
    source: 26 USC 151(d)(3)(B)
    versions:
      - effective_from: '2026-01-01'
        formula: 1
  - name: exemption_phaseout_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    source: 26 USC 151(d)(3)(A)-(B)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, adjusted_gross_income - applicable_amount_in_effect_under_section_68_b)
  - name: senior_deduction_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 151(d)(5)(C)(iii)(II)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/151
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          adjusted_gross_income
          + amount_excluded_from_gross_income_under_section_911
          + amount_excluded_from_gross_income_under_section_931
          + amount_excluded_from_gross_income_under_section_933
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    source: 26 USC 151(c)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/151
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          tin_included_on_return_claiming_exemption
          and is_dependent_under_section_152_of_taxpayer
  - name: senior_deduction_allowed
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    source: 26 USC 151(d)(5)(C)(i)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          not taxpayer_is_married_individual_within_section_7703
"""
        )
        test_file = target.with_name("151.test.yaml")
        test_file.write_text(
            """- name: senior_magi
  input:
    us:statutes/26/151#input.adjusted_gross_income: 100000
    us:statutes/26/151#input.applicable_amount_in_effect_under_section_68_b: 0
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_911: 5000
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_933: 0
    us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
    us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer: true
  output:
    us:statutes/26/151#exemption_phaseout_applicable_percentage: 1
    us:statutes/26/151#senior_deduction_modified_adjusted_gross_income: 105000
    us:statutes/26/151#exemption_individual_eligible: holds
"""
        )
        section_63_test = policy_repo / "statutes/26/63.test.yaml"
        section_63_test.write_text(
            """- name: dependent_151_case
  input:
    us:statutes/26/151#input.taxable_year_begins_after_2017: true
    us:statutes/26/151#input.taxable_year_begins_before_2029: true
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_911: 0
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_933: 0
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931: 0
    us:statutes/26/151#input.taxpayer_is_married_individual_within_section_7703: false
    us:statutes/26/63/f#input.additional_exemption_allowable_for_spouse_under_section_151_b: false
  output:
    us:statutes/26/63#taxable_income: 0
"""
        )
        section_63_f_test = policy_repo / "statutes/26/63/f.test.yaml"
        section_63_f_test.parent.mkdir(parents=True, exist_ok=True)
        section_63_f_test.write_text(
            """- name: spouse_entitlements_require_section_151_exemption
  input:
    us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f:
    - us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
      us:statutes/26/151#input.is_spouse_of_taxpayer: true
  output:
    us:statutes/26/63/f#spouse_aged_additional_amount_entitlement: holds
"""
        )
        section_152_c = policy_repo / "statutes/26/152/c.yaml"
        section_152_c.parent.mkdir(parents=True, exist_ok=True)
        section_152_c.write_text(
            """format: rulespec/v1
rules:
  - name: qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: qualifying_child_before_tiebreaker
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/151.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if path != target.resolve():
                    assert path in {
                        policy_repo / "statutes/26/911/a/1.yaml",
                        policy_repo / "statutes/26/911.yaml",
                        policy_repo / "statutes/26/152.yaml",
                        policy_repo / "statutes/26/152/d.yaml",
                        policy_repo / "statutes/26/931.yaml",
                        policy_repo / "statutes/26/933.yaml",
                    }
                    return SimpleNamespace(all_passed=True, results={})
                content = target.read_text()
                assert "applicable_amount_in_effect_under_section_68_b" not in content
                assert (
                    "  - name: exemption_phaseout_applicable_percentage\n"
                    not in content
                )
                assert "deferred_outputs:" in content
                assert "us:statutes/26/68/b#applicable_amount" in content
                assert (
                    "us:statutes/26/911#income_excluded_from_gross_income_under_section_911"
                    in content
                )
                assert "is_dependent_under_section_152_of_taxpayer" not in content
                assert "us:statutes/26/152#dependent_of_taxpayer" in content
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch("axiom_encode.cli._companion_test_issues", return_value=[]),
            patch(
                "axiom_encode.cli._ensure_no_unmanifested_preexisting_rulespec_changes"
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_tax_status_components(args)

        content = target.read_text()
        assert "amount_excluded_from_gross_income_under_section_911" not in content
        assert "income_excluded_from_gross_income_under_section_911" in content
        assert "dependent_of_taxpayer" in content
        assert "us:statutes/26/151#exemption_phaseout_increment_separate" in content
        test_content = test_file.read_text()
        assert "applicable_amount_in_effect_under_section_68_b" not in test_content
        assert "exemption_phaseout_applicable_percentage" not in test_content
        assert "is_dependent_under_section_152_of_taxpayer" not in test_content
        assert (
            "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_911"
            not in test_content
        )
        assert (
            "us:statutes/26/911/a/1#input.elected_foreign_earned_income_exclusion_amount: 5000"
            in test_content
        )
        assert (
            "us:statutes/26/931#input.gross_income_excluded_under_section_931: 0"
            in test_content
        )
        assert (
            "us:statutes/26/933#input.gross_income_excluded_under_section_933: 0"
            in test_content
        )
        assert (
            "us:statutes/26/152/c#input.individual_is_child_of_taxpayer" in test_content
        )
        related_test_content = section_63_test.read_text()
        assert "us:statutes/26/151#input.taxable_year_begins_after_2017" not in (
            related_test_content
        )
        assert (
            "us:statutes/26/151#input.taxable_year_begins_after_tcja_exemption_amount_zero_effective_date: true"
            in related_test_content
        )
        assert "us:statutes/26/151#input.taxable_year_begins_before_2029" not in (
            related_test_content
        )
        assert (
            "us:statutes/26/151#input.taxable_year_begins_before_senior_deduction_expiration_date: true"
            in related_test_content
        )
        assert (
            "us:statutes/26/151#input.taxpayer_is_married_individual_within_section_7703"
            not in related_test_content
        )
        assert "additional_exemption_allowable_for_spouse_under_section_151_b" not in (
            related_test_content
        )
        assert (
            "us:statutes/26/63/c#input.additional_standard_deduction_amount_under_subsection_f: 0"
            in related_test_content
        )
        assert (
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
            not in related_test_content
        )
        assert (
            "us:statutes/26/931#input.gross_income_excluded_under_section_931: 0"
            in related_test_content
        )
        related_63_f_content = section_63_f_test.read_text()
        assert "us:statutes/26/151#input.filing_status: 0" in related_63_f_content
        assert (
            "us:statutes/26/63/c#input.additional_standard_deduction_amount_under_subsection_f"
            not in related_63_f_content
        )
        generated_911_a_1 = policy_repo / "statutes/26/911/a/1.yaml"
        generated_911 = policy_repo / "statutes/26/911.yaml"
        assert generated_911_a_1.exists()
        assert generated_911.exists()
        assert f"hash: sha256:{_sha256_file(generated_911)}" in content
        assert f"hash: sha256:{_sha256_file(generated_911_a_1)}" in (
            generated_911.read_text()
        )
        assert (policy_repo / "statutes/26/152.yaml").exists()
        assert (policy_repo / "statutes/26/152/d.yaml").exists()
        assert (policy_repo / "statutes/26/931.yaml").exists()
        assert (policy_repo / "statutes/26/933.yaml").exists()
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/151.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "tax-status-component-v1"
        assert payload["tool"] == "axiom-encode repair-tax-status-components"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/151.yaml",
            "statutes/26/151.test.yaml",
            "statutes/26/63.test.yaml",
            "statutes/26/63/f.test.yaml",
            "statutes/26/911/a/1.yaml",
            "statutes/26/911/a/1.test.yaml",
            "statutes/26/911.yaml",
            "statutes/26/911.test.yaml",
            "statutes/26/152/d.yaml",
            "statutes/26/152/d.test.yaml",
            "statutes/26/152.yaml",
            "statutes/26/152.test.yaml",
            "statutes/26/931.yaml",
            "statutes/26/931.test.yaml",
            "statutes/26/933.yaml",
            "statutes/26/933.test.yaml",
        ]

    def test_repair_section_151_911_import_hashes_generated_and_existing_files(
        self, tmp_path
    ):
        content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: senior_deduction_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income + amount_excluded_from_gross_income_under_section_911
"""
        repo_without_911 = tmp_path / "without-911"
        repaired, repairs, generated = _repair_section_151_imports(
            content, repo_path=repo_without_911
        )
        generated_911 = generated[repo_without_911 / "statutes/26/911.yaml"]
        assert repairs == [
            "amount_excluded_from_gross_income_under_section_911->income_excluded_from_gross_income_under_section_911"
        ]
        assert f"hash: sha256:{_sha256_text(generated_911)}" in repaired

        repo_with_911 = tmp_path / "with-911"
        section_911 = repo_with_911 / "statutes/26/911.yaml"
        section_911.parent.mkdir(parents=True)
        section_911.write_text(
            """format: rulespec/v1
rules:
  - name: income_excluded_from_gross_income_under_section_911
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: existing_911_passthrough
"""
        )
        repaired, repairs, generated = _repair_section_151_imports(
            content, repo_path=repo_with_911
        )
        assert generated == {}
        assert repairs == [
            "amount_excluded_from_gross_income_under_section_911->income_excluded_from_gross_income_under_section_911"
        ]
        assert f"hash: sha256:{_sha256_file(section_911)}" in repaired

    def test_repair_tax_status_components_imports_24_child_dependencies(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/24.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/24
  summary: |-
    Section 24 uses section 152(c), section 151, and sections 911, 931, and 933.
rules:
  - name: ctc_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/24
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          adjusted_gross_income
          + amount_excluded_from_gross_income_under_section_911
          + amount_excluded_from_gross_income_under_section_931
          + amount_excluded_from_gross_income_under_section_933
  - name: ctc_qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/24
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          qualifying_child_under_section_152_c
          and allowed_deduction_under_section_151_for_child
          and not certain_noncitizen_exception_applies
  - name: ctc_after_advance_payments
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/24
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          max(0, ctc_before_advance_payments - aggregate_advance_payments_under_section_7527A)
"""
        )
        test_file = target.with_name("24.test.yaml")
        test_file.write_text(
            """- name: child
  input:
    us:statutes/26/24#input.qualifying_child_under_section_152_c: true
    us:statutes/26/24#input.allowed_deduction_under_section_151_for_child: true
    us:statutes/26/24#input.certain_noncitizen_exception_applies: false
  output:
    us:statutes/26/24#ctc_qualifying_child: holds
- name: child_relation_row
  input:
    us:statutes/26/24#relation.ctc_qualifying_child_of_tax_unit:
    - us:statutes/26/24#input.qualifying_child_under_section_152_c: true
      us:statutes/26/24#input.allowed_deduction_under_section_151_for_child: true
      us:statutes/26/24#input.certain_noncitizen_exception_applies: false
  output:
    us:statutes/26/24#ctc_qualifying_child: holds
- name: magi
  input:
    us:statutes/26/24#input.adjusted_gross_income: 100
    us:statutes/26/24#input.amount_excluded_from_gross_income_under_section_911: 1
    us:statutes/26/24#input.amount_excluded_from_gross_income_under_section_931: 2
    us:statutes/26/24#input.amount_excluded_from_gross_income_under_section_933: 3
  output:
    us:statutes/26/24#ctc_modified_adjusted_gross_income: 106
- name: advance_payment_reduction
  input:
    us:statutes/26/24#input.ctc_before_advance_payments: 500
    us:statutes/26/24#input.aggregate_advance_payments_under_section_7527A: 100
  output:
    us:statutes/26/24#ctc_after_advance_payments: 400
"""
        )
        section_151 = policy_repo / "statutes/26/151.yaml"
        section_151.write_text(
            """format: rulespec/v1
rules:
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: tin_included_on_return_claiming_exemption and is_taxpayer
"""
        )
        section_152_c = policy_repo / "statutes/26/152/c.yaml"
        section_152_c.parent.mkdir(parents=True, exist_ok=True)
        section_152_c.write_text(
            """format: rulespec/v1
rules:
  - name: qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: individual_is_child_of_taxpayer_or_descendant_of_such_child
"""
        )
        section_911 = policy_repo / "statutes/26/911.yaml"
        section_911.write_text(
            """format: rulespec/v1
rules:
  - name: income_excluded_from_gross_income_under_section_911
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: foreign_earned_income_excluded_from_gross_income
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/24.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if path == target.resolve():
                    content = target.read_text()
                    assert "qualifying_child_under_section_152_c" not in content
                    assert (
                        "allowed_deduction_under_section_151_for_child" not in content
                    )
                    assert "us:statutes/26/152/c#qualifying_child" in content
                    assert "us:statutes/26/151#exemption_individual_eligible" in content
                    assert (
                        "us:statutes/26/7527A#aggregate_advance_payments_under_section_7527A"
                        in content
                    )
                    assert "qualifying_child" in content
                    assert "exemption_individual_eligible" in content
                    return SimpleNamespace(all_passed=True, results={})
                assert path in {
                    policy_repo / "statutes/26/931.yaml",
                    policy_repo / "statutes/26/933.yaml",
                    policy_repo / "statutes/26/7527A.yaml",
                }
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch("axiom_encode.cli._companion_test_issues", return_value=[]),
            patch(
                "axiom_encode.cli._ensure_no_unmanifested_preexisting_rulespec_changes"
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_tax_status_components(args)

        test_content = test_file.read_text()
        assert "us:statutes/26/24#input.qualifying_child_under_section_152_c" not in (
            test_content
        )
        assert (
            "us:statutes/26/24#input.allowed_deduction_under_section_151_for_child"
            not in test_content
        )
        assert (
            "us:statutes/26/152/c#input.individual_is_child_of_taxpayer_or_descendant_of_such_child: true"
            in test_content
        )
        assert (
            "us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true"
            in test_content
        )
        assert (
            "    - us:statutes/26/152/c#input.individual_is_child_of_taxpayer_or_descendant_of_such_child: true"
            in test_content
        )
        assert (
            "us:statutes/26/911/a/1#input.elected_foreign_earned_income_exclusion_amount: 1"
            in test_content
        )
        assert (
            "us:statutes/26/931#input.gross_income_excluded_under_section_931: 2"
            in test_content
        )
        assert (
            "us:statutes/26/933#input.gross_income_excluded_under_section_933: 3"
            in test_content
        )
        assert (
            "us:statutes/26/7527A#input.advance_payments_made_under_section_7527A: 100"
            in test_content
        )
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/24.json"
        payload = json.loads(manifest.read_text())
        assert payload["tool"] == "axiom-encode repair-tax-status-components"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/24.yaml",
            "statutes/26/24.test.yaml",
            "statutes/26/931.yaml",
            "statutes/26/931.test.yaml",
            "statutes/26/933.yaml",
            "statutes/26/933.test.yaml",
            "statutes/26/7527A.yaml",
            "statutes/26/7527A.test.yaml",
        ]

    def test_repair_unreferenced_proof_imports_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/1402/a.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/1401#self_employment_oasdi_tax_rate
module:
  proof_validation:
    required: true
rules:
  - name: paragraph_12_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/1402
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/1401#self_employment_oasdi_tax_rate
              hash: sha256:abc123
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          net_earnings_before_paragraph_12_adjustment * paragraph_12_deduction_rate
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/1402/a.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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
            cmd_repair_unreferenced_proof_imports(args)

        content = target.read_text()
        assert "us:statutes/26/1401#self_employment_oasdi_tax_rate" not in content
        assert "formula: |-" in content
        assert (
            "          net_earnings_before_paragraph_12_adjustment * "
            "paragraph_12_deduction_rate"
        ) in content
        assert "  - name: paragraph_12_deduction\n" in content
        payload = yaml.safe_load(content)
        atoms = payload["rules"][0]["metadata"]["proof"]["atoms"]
        assert all(atom.get("kind") != "import" for atom in atoms)
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/1402/a.json"
        manifest_payload = json.loads(manifest.read_text())
        assert manifest_payload["backend"] == "deterministic"
        assert manifest_payload["model"] == "proof-import-reference-v1"
        assert (
            manifest_payload["tool"] == "axiom-encode repair-unreferenced-proof-imports"
        )
        assert [
            applied_file["path"] for applied_file in manifest_payload["applied_files"]
        ] == ["statutes/26/1402/a.yaml"]

    def test_repair_section_172_c_capacity_writes_signed_manifests(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target_1212 = policy_repo / "statutes/26/1212/a/1.yaml"
        test_1212 = policy_repo / "statutes/26/1212/a/1.test.yaml"
        target_1222 = policy_repo / "statutes/26/1222.yaml"
        test_1222 = policy_repo / "statutes/26/1222.test.yaml"
        target_1212.parent.mkdir(parents=True)
        target_1222.parent.mkdir(parents=True, exist_ok=True)
        target_1212.write_text(
            """format: rulespec/v1
rules:
  - name: corporation_capital_loss_carryback_to_taxable_year
    kind: derived
    entity: Corporation
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          min(
              net_capital_loss_not_attributable_to_foreign_expropriation_capital_loss,
              capital_loss_carryback_amount_that_does_not_increase_or_produce_net_operating_loss_under_section_172_c
          )
"""
        )
        test_1212.write_text(
            """- name: carryback_case
  input:
    ? us:statutes/26/1212/a/1#input.capital_loss_carryback_amount_that_does_not_increase_or_produce_net_operating_loss_under_section_172_c
    : 8000
  output:
    us:statutes/26/1212/a/1#corporation_capital_loss_carryback_to_taxable_year: 8000
"""
        )
        target_1222.write_text(
            """format: rulespec/v1
rules:
  - name: short_term_capital_loss
    kind: derived
    entity: Corporation
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/1212/a/1#corporation_capital_loss_carryback_to_taxable_year
              output: corporation_capital_loss_carryback_to_taxable_year
              hash: sha256:stale
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          corporation_capital_loss_carryback_to_taxable_year
"""
        )
        test_1222.write_text(
            """- name: dependent_case
  input:
    ? us:statutes/26/1212/a/1#input.capital_loss_carryback_amount_that_does_not_increase_or_produce_net_operating_loss_under_section_172_c
    : 3000
  output:
    us:statutes/26/1222#short_term_capital_loss: 3000
"""
        )
        subprocess.run(
            ["git", "init"], cwd=policy_repo, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(["git", "add", "statutes"], cwd=policy_repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Axiom Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=policy_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        args = SimpleNamespace(
            repo=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )
        companion_failures = MagicMock(return_value=[])

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                assert path in {
                    policy_repo / "statutes/26/172/c.yaml",
                    target_1212,
                    target_1222,
                }
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                companion_failures,
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_section_172_c_capacity(args)

        assert [call.args[0] for call in companion_failures.call_args_list] == [
            policy_repo / "statutes/26/172/c.test.yaml",
            test_1212,
            test_1222,
        ]
        assert (policy_repo / "statutes/26/172/c.yaml").exists()
        assert (
            "us:statutes/26/172/c#deduction_capacity_before_net_operating_loss"
            in target_1212.read_text()
        )
        assert (
            "capital_loss_carryback_amount_that_does_not_increase_or_produce"
            not in target_1212.read_text()
        )
        assert "us:statutes/26/172/c#input.gross_income: 8000" in test_1212.read_text()
        assert "us:statutes/26/172/c#input.gross_income: 3000" in test_1222.read_text()
        assert f"hash: sha256:{_sha256_file(target_1212)}" in target_1222.read_text()
        for manifest in (
            policy_repo / ".axiom/encoding-manifests/statutes/26/172/c.json",
            policy_repo / ".axiom/encoding-manifests/statutes/26/1212/a/1.json",
            policy_repo / ".axiom/encoding-manifests/statutes/26/1222.json",
        ):
            payload = json.loads(manifest.read_text())
            assert payload["model"] == "section-172-c-capacity-v1"
            assert payload["tool"] == "axiom-encode repair-section-172-c-capacity"
            applied_paths = [
                applied_file["path"] for applied_file in payload["applied_files"]
            ]
            assert (
                applied_paths
                == {
                    policy_repo / ".axiom/encoding-manifests/statutes/26/172/c.json": [
                        "statutes/26/172/c.yaml",
                        "statutes/26/172/c.test.yaml",
                    ],
                    policy_repo
                    / ".axiom/encoding-manifests/statutes/26/1212/a/1.json": [
                        "statutes/26/1212/a/1.yaml",
                        "statutes/26/1212/a/1.test.yaml",
                    ],
                    policy_repo / ".axiom/encoding-manifests/statutes/26/1222.json": [
                        "statutes/26/1222.yaml",
                        "statutes/26/1222.test.yaml",
                    ],
                }[manifest]
            )

        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            assert (
                guard_generated_change_issues(
                    policy_repo,
                    changed_files=[
                        "statutes/26/172/c.yaml",
                        "statutes/26/172/c.test.yaml",
                        "statutes/26/1212/a/1.yaml",
                        "statutes/26/1212/a/1.test.yaml",
                        "statutes/26/1222.yaml",
                        "statutes/26/1222.test.yaml",
                        ".axiom/encoding-manifests/statutes/26/172/c.json",
                        ".axiom/encoding-manifests/statutes/26/1212/a/1.json",
                        ".axiom/encoding-manifests/statutes/26/1222.json",
                    ],
                )
                == []
            )

    def test_repair_section_172_c_capacity_refuses_dirty_targets(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target_1212 = policy_repo / "statutes/26/1212/a/1.yaml"
        target_1212.parent.mkdir(parents=True)
        target_1212.write_text("format: rulespec/v1\nrules: []\n")
        subprocess.run(
            ["git", "init"], cwd=policy_repo, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(["git", "add", "statutes"], cwd=policy_repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Axiom Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=policy_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        target_1212.write_text("format: rulespec/v1\nrules:\n  - manual\n")
        args = SimpleNamespace(
            repo=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        with (
            patch("axiom_encode.cli.ValidatorPipeline", side_effect=AssertionError),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
            pytest.raises(RuntimeError, match="Refusing to sign over pre-existing"),
        ):
            cmd_repair_section_172_c_capacity(args)

        assert not (policy_repo / "statutes/26/172").exists()

    def test_repair_section_911_a_1_exclusion_writes_signed_manifests(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target_1411 = policy_repo / "statutes/26/1411.yaml"
        test_1411 = policy_repo / "statutes/26/1411.test.yaml"
        target_1411.parent.mkdir(parents=True)
        target_1411.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/911/d/6#section_911_disallowed_deductions_and_exclusions
rules:
  - name: niit_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: definition
            source:
              corpus_citation_path: us/statute/26/1411
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/911/d/6#section_911_disallowed_deductions_and_exclusions
              output: section_911_disallowed_deductions_and_exclusions
              hash: sha256:abc
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          adjusted_gross_income
          + amount_excluded_from_gross_income_under_section_911_a_1
          - section_911_disallowed_deductions_and_exclusions
"""
        )
        test_1411.write_text(
            """- name: section_911_case
  input:
    us:statutes/26/1411#input.amount_excluded_from_gross_income_under_section_911_a_1: 10000
  output:
    us:statutes/26/1411#niit_modified_adjusted_gross_income: 10000
"""
        )
        subprocess.run(
            ["git", "init"], cwd=policy_repo, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(["git", "add", "statutes"], cwd=policy_repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Axiom Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=policy_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        args = SimpleNamespace(
            repo=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )
        companion_failures = MagicMock(return_value=[])

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                assert path in {
                    policy_repo / "statutes/26/911/a/1.yaml",
                    policy_repo / "statutes/26/67/e.yaml",
                    target_1411,
                }
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                companion_failures,
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_section_911_a_1_exclusion(args)

        target_911 = policy_repo / "statutes/26/911/a/1.yaml"
        target_67 = policy_repo / "statutes/26/67/e.yaml"
        assert target_911.exists()
        assert target_67.exists()
        assert (
            "us:statutes/26/911/a/1#foreign_earned_income_excluded_from_gross_income"
            in target_1411.read_text()
        )
        assert f"hash: sha256:{_sha256_file(target_911)}" in target_1411.read_text()
        assert (
            "us:statutes/26/911/a/1#input.elected_foreign_earned_income_exclusion_amount: 10000"
            in test_1411.read_text()
        )
        assert [call.args[0] for call in companion_failures.call_args_list] == [
            policy_repo / "statutes/26/911/a/1.test.yaml",
            policy_repo / "statutes/26/67/e.test.yaml",
            test_1411,
        ]
        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            assert (
                guard_generated_change_issues(
                    policy_repo,
                    changed_files=[
                        "statutes/26/911/a/1.yaml",
                        "statutes/26/911/a/1.test.yaml",
                        "statutes/26/67/e.yaml",
                        "statutes/26/67/e.test.yaml",
                        "statutes/26/1411.yaml",
                        "statutes/26/1411.test.yaml",
                        ".axiom/encoding-manifests/statutes/26/911/a/1.json",
                        ".axiom/encoding-manifests/statutes/26/67/e.json",
                        ".axiom/encoding-manifests/statutes/26/1411.json",
                    ],
                )
                == []
            )

    def test_repair_section_63_f_stale_test_inputs_writes_signed_manifests(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target_63_c = policy_repo / "statutes/26/63/c.yaml"
        target_63 = policy_repo / "statutes/26/63.yaml"
        target_6012 = policy_repo / "statutes/26/6012.yaml"
        target_1_f_3 = policy_repo / "statutes/26/1/f/3.yaml"
        target_121 = policy_repo / "statutes/26/121.yaml"
        target_911_a_1 = policy_repo / "statutes/26/911/a/1.yaml"
        target_911 = policy_repo / "statutes/26/911.yaml"
        target_443_a_1 = policy_repo / "statutes/26/443/a/1.yaml"
        target_7703 = policy_repo / "statutes/26/7703.yaml"
        target_6013_a = policy_repo / "statutes/26/6013/a.yaml"
        for target in (
            target_1_f_3,
            target_121,
            target_911_a_1,
            target_911,
            target_443_a_1,
            target_63_c,
            target_63,
            target_6012,
            target_7703,
            target_6013_a,
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
        target_63_c.write_text(
            """format: rulespec/v1
rules:
  - name: head_of_household_basic_standard_deduction_amount
    kind: derived
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          if taxable_year_begins_after_2025:
            23625 + floor(23625 * cost_of_living_adjustment_under_section_1_f_3 / 50) * 50
          else:
            23625
  - name: standard_deduction_ineligible
    kind: derived
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          return_under_section_443_a_1_for_less_than_12_months_due_to_accounting_period_change
"""
        )
        target_63.write_text("format: rulespec/v1\nrules: []\n")
        target_6012.write_text(
            """format: rulespec/v1
rules:
  - name: gross_income_for_section_6012
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/6012
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          gross_income + gain_excluded_from_gross_income_under_section_121
          + income_excluded_from_gross_income_under_section_911
  - name: unmarried_individual_exception_to_return_requirement_under_2018_2025_rule
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/6012
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          taxable_year_begins_after_2017_and_before_2026
          and individual_not_married_under_section_7703
          and gross_income_for_section_6012 <= standard_deduction
  - name: joint_return_exception_to_return_requirement_under_2018_2025_rule
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/6012
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          individual_entitled_to_make_joint_return
          and combined_gross_income_of_individual_and_spouse_for_section_6012 <= standard_deduction
"""
        )
        target_443_a_1.write_text("format: rulespec/v1\nrules: []\n")
        target_7703.write_text("format: rulespec/v1\nrules: []\n")
        target_6013_a.write_text("format: rulespec/v1\nrules: []\n")
        for target in (target_63_c, target_63, target_6012):
            test_content = """- name: stale_input_case
  input:
    us:statutes/26/63/c#input.taxable_year_begins_after_2025: true
    us:statutes/26/63/c#input.cost_of_living_adjustment_under_section_1_f_3: 0.1
    us:statutes/26/63/c#input.return_under_section_443_a_1_for_less_than_12_months_due_to_accounting_period_change: false
    us:statutes/26/63/f#input.filing_status: 0
    us:statutes/26/63/f#input.additional_exemption_allowable_for_spouse_under_section_151_b: false
    us:example#input.other_value: 1
  output:
    us:example#output.value: 1
"""
            if target == target_6012:
                test_content = test_content.replace(
                    "    us:example#input.other_value: 1\n",
                    "    us:statutes/26/6012#input.individual_not_married_under_section_7703: true\n"
                    "    us:statutes/26/6012#input.individual_entitled_to_make_joint_return: true\n"
                    "    us:statutes/26/6012#input.taxable_year_begins_after_2017_and_before_2026: true\n"
                    "    us:statutes/26/6012#input.gain_excluded_from_gross_income_under_section_121: 0\n"
                    "    us:statutes/26/6012#input.income_excluded_from_gross_income_under_section_911: 0\n"
                    "    us:example#input.other_value: 1\n",
                )
            target.with_name(f"{target.stem}.test.yaml").write_text(test_content)
        subprocess.run(
            ["git", "init"], cwd=policy_repo, check=True, stdout=subprocess.PIPE
        )
        subprocess.run(["git", "add", "statutes"], cwd=policy_repo, check=True)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Axiom Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "baseline",
            ],
            cwd=policy_repo,
            check=True,
            stdout=subprocess.PIPE,
        )
        args = SimpleNamespace(
            repo=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )
        companion_failures = MagicMock(return_value=[])

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures",
                companion_failures,
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_section_63_f_stale_test_inputs(args)

        for relative in (
            "statutes/26/63/c.test.yaml",
            "statutes/26/63.test.yaml",
            "statutes/26/6012.test.yaml",
        ):
            content = (policy_repo / relative).read_text()
            assert "additional_exemption_allowable_for_spouse" not in content
            assert "us:statutes/26/63/f#input.filing_status" not in content
            assert "cost_of_living_adjustment_under_section_1_f_3" not in content
            assert "return_under_section_443_a_1" not in content
            assert (
                "us:statutes/26/1/f/3#input.consumer_price_index_for_preceding_calendar_year: 110"
                in content
            )
            assert (
                "us:statutes/26/443/a/1#input.taxpayer_changes_annual_accounting_period: false"
                in content
            )
            assert (
                "us:statutes/26/63/c#input.additional_standard_deduction_amount_under_subsection_f: 0"
                in content
            )
            assert "us:example#input.other_value: 1" in content
        assert target_1_f_3.exists()
        assert "cost_of_living_adjustment" in target_1_f_3.read_text()
        assert target_121.exists()
        assert "principal_residence_sale_gain_exclusion" in target_121.read_text()
        assert target_911_a_1.exists()
        assert (
            "foreign_earned_income_excluded_from_gross_income"
            in target_911_a_1.read_text()
        )
        assert target_911.exists()
        assert (
            "income_excluded_from_gross_income_under_section_911"
            in target_911.read_text()
        )
        assert (
            "cost_of_living_adjustment_under_section_1_f_3"
            not in target_63_c.read_text()
        )
        assert (
            "us:statutes/26/1/f/3#cost_of_living_adjustment" in target_63_c.read_text()
        )
        assert "taxable_year_begins_after_2025" not in target_63_c.read_text()
        assert (
            "taxable_year_begins_after_tcja_standard_deduction_transition_period"
            in target_63_c.read_text()
        )
        assert "return_under_section_443_a_1" not in target_63_c.read_text()
        assert (
            "annual_accounting_period_change_with_secretary_approval"
            in target_63_c.read_text()
        )
        assert (
            "gain_excluded_from_gross_income_under_section_121"
            not in target_6012.read_text()
        )
        assert "principal_residence_sale_gain_exclusion" in target_6012.read_text()
        assert (
            "us:statutes/26/911#income_excluded_from_gross_income_under_section_911"
            in target_6012.read_text()
        )
        assert (
            "taxable_year_begins_after_2017_and_before_2026"
            not in target_6012.read_text()
        )
        assert (
            "taxable_year_is_in_temporary_effective_window" in target_6012.read_text()
        )
        assert (
            "individual_not_married_under_section_7703" not in target_6012.read_text()
        )
        assert (
            "not taxpayer_considered_married_after_living_apart_rule"
            in target_6012.read_text()
        )
        assert "individual_entitled_to_make_joint_return" not in target_6012.read_text()
        assert "joint_return_may_be_made" in target_6012.read_text()
        assert (
            "us:statutes/26/7703#input.taxpayer_married_at_close_of_taxable_year: false"
            in (policy_repo / "statutes/26/6012.test.yaml").read_text()
        )
        assert (
            "us:statutes/26/6013/a#input.taxpayers_are_husband_and_wife: true"
            in (policy_repo / "statutes/26/6012.test.yaml").read_text()
        )
        assert (
            "us:statutes/26/121#input.principal_residence_sale_gain_excluded_from_gross_income: 0"
            in (policy_repo / "statutes/26/6012.test.yaml").read_text()
        )
        assert (
            "us:statutes/26/911/a/1#input.elected_foreign_earned_income_exclusion_amount: 0"
            in (policy_repo / "statutes/26/6012.test.yaml").read_text()
        )
        assert {call.args[0] for call in companion_failures.call_args_list} == {
            policy_repo / "statutes/26/1/f/3.test.yaml",
            policy_repo / "statutes/26/121.test.yaml",
            policy_repo / "statutes/26/911/a/1.test.yaml",
            policy_repo / "statutes/26/911.test.yaml",
            policy_repo / "statutes/26/63/c.test.yaml",
            policy_repo / "statutes/26/63.test.yaml",
            policy_repo / "statutes/26/6012.test.yaml",
        }
        with patch.dict(
            os.environ,
            {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
        ):
            assert (
                guard_generated_change_issues(
                    policy_repo,
                    changed_files=[
                        "statutes/26/1/f/3.yaml",
                        "statutes/26/1/f/3.test.yaml",
                        "statutes/26/121.yaml",
                        "statutes/26/121.test.yaml",
                        "statutes/26/911/a/1.yaml",
                        "statutes/26/911/a/1.test.yaml",
                        "statutes/26/911.yaml",
                        "statutes/26/911.test.yaml",
                        "statutes/26/63/c.yaml",
                        "statutes/26/63/c.test.yaml",
                        "statutes/26/63.test.yaml",
                        "statutes/26/6012.yaml",
                        "statutes/26/6012.test.yaml",
                        ".axiom/encoding-manifests/statutes/26/1/f/3.json",
                        ".axiom/encoding-manifests/statutes/26/121.json",
                        ".axiom/encoding-manifests/statutes/26/911/a/1.json",
                        ".axiom/encoding-manifests/statutes/26/911.json",
                        ".axiom/encoding-manifests/statutes/26/63/c.json",
                        ".axiom/encoding-manifests/statutes/26/63.json",
                        ".axiom/encoding-manifests/statutes/26/6012.json",
                    ],
                )
                == []
            )

    def test_repair_missing_source_proofs_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/3101/b/2.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3101
  summary: Medicare tax threshold source text.
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    source: 26 USC 3101(b)(2)(A)
    versions:
      - effective_from: '2013-01-01'
        formula: '0.009'
  - name: additional_medicare_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3101(b)(2)
    versions:
      - effective_from: '2013-01-01'
        formula: wages * additional_medicare_tax_rate
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/3101/b/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

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
            cmd_repair_missing_source_proofs(args)

        content = target.read_text()
        assert "proof_validation:\n    required: true" in content
        assert content.count("metadata:\n      proof:\n        atoms:") == 2
        assert "kind: parameter" in content
        assert "kind: formula" in content
        assert "corpus_citation_path: us/statute/26/3101" in content
        assert "span: '26 USC 3101(b)(2)(A)'" in content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/3101/b/2.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "source-proof-atom-v1"
        assert payload["tool"] == "axiom-encode repair-missing-source-proofs"
        assert [applied_file["path"] for applied_file in payload["applied_files"]] == [
            "statutes/26/3101/b/2.yaml",
        ]

    def test_repair_missing_source_proofs_treats_derived_relation_as_executable(
        self,
    ):
        content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/7/273/1
rules:
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
"""

        repaired, repaired_rules = _repair_missing_source_proof_atoms(content)

        assert repaired_rules == ["snap_unit"]
        assert "metadata:\n      proof:\n        atoms:" in repaired
        assert "kind: formula" in repaired
        assert "corpus_citation_path: us/regulation/7/273/1" in repaired
        assert "span: '7 CFR 273.1(a)'" in repaired

    def test_repair_missing_source_proofs_allows_pending_tax_branch_repair(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/3101/b/2.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3101
  summary: joint return or surviving spouse.
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3101(b)(2)(A)-(C)
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          match filing_status:
              1 => additional_medicare_wage_tax_joint_threshold
              2 => additional_medicare_wage_tax_joint_threshold / 2
              0 => additional_medicare_wage_tax_other_threshold
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/3101/b/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(
                    all_passed=False,
                    results={
                        "filing_status": SimpleNamespace(
                            error=(
                                "Filing status branch missing surviving spouse: "
                                "`additional_medicare_wage_tax_threshold` handles "
                                "joint-return status code 1 while this source "
                                "mentions surviving spouse."
                            )
                        )
                    },
                )

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
            cmd_repair_missing_source_proofs(args)

        content = target.read_text()
        assert "metadata:\n      proof:\n        atoms:" in content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/3101/b/2.json"
        assert manifest.exists()

    def test_repair_missing_source_proofs_allows_pending_nonnegative_floor_repair(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/1/h.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
  summary: capital gains taxable income exclusion.
rules:
  - name: capital_gains_excluded_from_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 1(h)(1)(A)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          taxable_income - capital_gains_exclusion
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/1/h.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(
                    all_passed=False,
                    results={
                        "nonnegative": SimpleNamespace(
                            error=(
                                "Nonnegative taxable income missing floor: "
                                "`capital_gains_excluded_from_taxable_income` "
                                "can return a negative amount."
                            )
                        )
                    },
                )

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
            cmd_repair_missing_source_proofs(args)

        content = target.read_text()
        assert "metadata:\n      proof:\n        atoms:" in content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/1/h.json"
        assert manifest.exists()

    def test_repair_proof_import_hashes_writes_signed_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/22.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: section_22_age_threshold
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          65
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/22#section_22_age_threshold
              output: section_22_age_threshold
              hash: sha256:20182e27b153a3a48aad21a7321215bf9a581fcd69e4a87d815d8ade8a2cccff
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          age >= section_22_age_threshold
"""
        )
        test_file = target.with_name("22.test.yaml")
        test_file.write_text("- name: existing_case\n  period: 2026\n")
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/22.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(
                    all_passed=False,
                    results={
                        "zero_branch": SimpleNamespace(
                            error=(
                                "Zero branch test coverage missing: "
                                "`section_22_initial_amount_before_disability_cap` "
                                "has a formula branch that returns 0."
                            )
                        )
                    },
                )

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
            cmd_repair_proof_import_hashes(args)

        content = target.read_text()
        assert "hash: sha256:local" in content
        assert "20182e27" not in content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/22.json"
        payload = json.loads(manifest.read_text())
        assert payload["backend"] == "deterministic"
        assert payload["model"] == "proof-import-hash-v1"
        assert payload["tool"] == "axiom-encode repair-proof-import-hashes"
        assert [item["path"] for item in payload["applied_files"]] == [
            "statutes/26/22.yaml",
            "statutes/26/22.test.yaml",
        ]

    def test_repair_imported_proof_hashes_writes_target_file_hash(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        imported = policy_repo / "statutes/7/2015/d/2/C.yaml"
        target = policy_repo / "statutes/7/2015/d/2.yaml"
        imported.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: bona_fide_student_half_time_enrollment_exemption_applies
    kind: derived
    entity: Person
    dtype: Boolean
    period: Month
    formula: |-
      true
"""
        )
        target.write_text(
            """format: rulespec/v1
rules:
  - name: paragraph_1_work_requirements_exemption_applies
    kind: derived
    entity: Person
    dtype: Boolean
    period: Month
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/7/2015/d/2/C#bona_fide_student_half_time_enrollment_exemption_applies
              output: bona_fide_student_half_time_enrollment_exemption_applies
              hash: sha256:old
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          bona_fide_student_half_time_enrollment_exemption_applies
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/7/2015/d/2.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

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
            cmd_repair_proof_import_hashes(args)

        expected_hash = f"sha256:{_sha256_file(imported)}"
        content = target.read_text()
        assert f"hash: {expected_hash}" in content
        assert "sha256:old" not in content
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/7/2015/d/2.json"
        payload = json.loads(manifest.read_text())
        assert payload["model"] == "proof-import-hash-v1"
        assert payload["tool"] == "axiom-encode repair-proof-import-hashes"
        assert payload["applied_files"][0]["path"] == "statutes/7/2015/d/2.yaml"

    def test_repair_proof_import_hashes_refreshes_existing_manifest(self, tmp_path):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/63.yaml"
        test_file = policy_repo / "statutes/26/63.test.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income
"""
        )
        test_file.write_text(
            """- name: taxable_income_case
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100
  output:
    us:statutes/26/63#taxable_income: 100
"""
        )
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/63.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"model": "proof-import-hash-v1"}')
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/63.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

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
            cmd_repair_proof_import_hashes(args)

        payload = json.loads(manifest.read_text())
        assert payload["model"] == "proof-import-hash-v1"
        assert payload["axiom_encode_git"]["commit"] == "abc123"
        assert [item["path"] for item in payload["applied_files"]] == [
            "statutes/26/63.yaml",
            "statutes/26/63.test.yaml",
        ]

    def test_repair_proof_import_hashes_removes_import_output_placeholders(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        section_151 = policy_repo / "statutes/26/151.yaml"
        section_224 = policy_repo / "statutes/26/224.yaml"
        target = policy_repo / "statutes/26/63.yaml"
        test_file = policy_repo / "statutes/26/63.test.yaml"
        target.parent.mkdir(parents=True)
        section_151.write_text(
            """format: rulespec/v1
rules:
  - name: section_151_exemption_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 0
"""
        )
        section_224.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules: []
"""
        )
        target.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151#section_151_exemption_deduction
  - us:statutes/26/224
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/151#section_151_exemption_deduction
              output: section_151_exemption_deduction
              hash: sha256:old
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income - section_151_exemption_deduction
"""
        )
        test_file.write_text(
            """- name: invalid_import_output_input
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100000
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931: 0
  output:
    us:statutes/26/63#taxable_income: 100000
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/63.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakePipeline:
            calls = 0

            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is False

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                FakePipeline.calls += 1
                if FakePipeline.calls == 1:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `invalid_import_output_input` execution failed: "
                                    "dataset input "
                                    "`us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931` "
                                    "must use an absolute legal RuleSpec reference "
                                    "that resolves to an input slot, derived rule, "
                                    "or parameter in the compiled program"
                                )
                            )
                        },
                    )
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
            cmd_repair_proof_import_hashes(args)

        assert "sha256:old" not in target.read_text()
        updated = test_file.read_text()
        assert (
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
            not in updated
        )
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/63.json"
        payload = json.loads(manifest.read_text())
        assert [item["path"] for item in payload["applied_files"]] == [
            "statutes/26/63.yaml",
            "statutes/26/63.test.yaml",
        ]

    def test_repair_oracle_parameter_tests_adds_missing_parameter_output(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/32.yaml"
        test_file = policy_repo / "statutes/26/32.test.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: eitc_phase_in_rates
    kind: parameter
    dtype: Rate
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0765
          1: 0.34
"""
        )
        test_file.write_text(
            """- name: existing
  output:
    us:statutes/26/32#some_other_output: 1
"""
        )
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/32.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        class FakeRegistry:
            def mapping_for_legal_id(self, legal_id, *, country=None):
                assert country == "us"
                if legal_id == "us:statutes/26/32#eitc_phase_in_rates":
                    return SimpleNamespace(mapping_type="parameter_value")
                return None

        class FakePipeline:
            def __init__(self, **kwargs):
                assert kwargs["require_policy_proofs"] is True

            def validate(self, path, *, skip_reviewers):
                assert path == target.resolve()
                assert skip_reviewers is True
                return SimpleNamespace(all_passed=True, results={})

        with (
            patch("axiom_encode.cli.ValidatorPipeline", FakePipeline),
            patch(
                "axiom_encode.cli.load_policyengine_registry",
                return_value=FakeRegistry(),
            ),
            patch(
                "axiom_encode.cli._rulespec_companion_test_failures", return_value=[]
            ),
            patch(
                "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
                return_value={"commit": "abc123", "dirty_tracked": False},
            ),
            patch.dict(
                os.environ,
                {APPLIED_ENCODING_SIGNING_KEY_ENV: TEST_APPLY_SIGNING_KEY},
            ),
        ):
            cmd_repair_oracle_parameter_tests(args)

        payload = yaml.safe_load(test_file.read_text())
        added = payload[-1]
        assert added["name"] == "oracle_parameter_eitc_phase_in_rates_0"
        assert added["input"] == {"us:statutes/26/32#input.qualifying_child_count": 0}
        assert added["output"] == {"us:statutes/26/32#eitc_phase_in_rates": 0.0765}
        manifest = policy_repo / ".axiom/encoding-manifests/statutes/26/32.json"
        manifest_payload = json.loads(manifest.read_text())
        assert manifest_payload["model"] == "oracle-parameter-test-v1"
        assert manifest_payload["tool"] == "axiom-encode repair-oracle-parameter-tests"
        assert [item["path"] for item in manifest_payload["applied_files"]] == [
            "statutes/26/32.yaml",
            "statutes/26/32.test.yaml",
        ]

    def test_repair_oracle_parameter_tests_does_not_mutate_without_signing_key(
        self, tmp_path
    ):
        policy_repo = tmp_path / "rulespec-us"
        target = policy_repo / "statutes/26/32.yaml"
        test_file = policy_repo / "statutes/26/32.test.yaml"
        target.parent.mkdir(parents=True)
        target.write_text(
            """format: rulespec/v1
rules:
  - name: eitc_phase_in_rates
    kind: parameter
    dtype: Rate
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0765
"""
        )
        original_test_content = """- name: existing
  output:
    us:statutes/26/32#some_other_output: 1
"""
        test_file.write_text(original_test_content)
        args = SimpleNamespace(
            repo=policy_repo,
            file=Path("statutes/26/32.yaml"),
            axiom_rules_path=tmp_path / "axiom-rules-engine",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(RuntimeError, match=APPLIED_ENCODING_SIGNING_KEY_ENV),
        ):
            cmd_repair_oracle_parameter_tests(args)

        assert test_file.read_text() == original_test_content

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

    def test_apply_overlay_validation_fills_relation_row_inputs_from_baseline(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/151.yaml"
        generated_test = generated.with_name("151.test.yaml")
        dependent = policy_repo / "statutes/26/7703.yaml"
        dependent_test = dependent.with_name("7703.test.yaml")
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        generated_test.write_text(
            """- name: baseline
  input:
    us:statutes/26/151#input.individual_is_dependent_of_taxpayer: false
  output:
    us:statutes/26/151#exemption_individual_eligible: holds
"""
        )
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/26/151\nrules: []\n"
        )
        dependent_test.write_text(
            """- name: relation_case
  input:
    us:statutes/26/7703#relation.living_apart_child_of_tax_unit:
    - us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
      us:statutes/26/151#input.is_taxpayer: false
      us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer: true
  output:
    us:statutes/26/7703#taxpayer_not_considered_married_when_living_apart: holds
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "151.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                test_content = Path(path).with_name("7703.test.yaml").read_text()
                relation_block = test_content.split(
                    "us:statutes/26/7703#relation.living_apart_child_of_tax_unit:",
                    1,
                )[1]
                if (
                    "us:statutes/26/151#input.individual_is_dependent_of_taxpayer"
                    not in relation_block
                ):
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Test case `relation_case` execution failed: "
                                    "missing input `individual_is_dependent_of_taxpayer`"
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
        updated = supplemental[Path("statutes/26/7703.test.yaml")]
        assert (
            "      us:statutes/26/151#input.individual_is_dependent_of_taxpayer: true"
            in updated
        )
        assert (
            "us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer"
            not in updated
        )

    def test_apply_overlay_validation_fills_target_imported_relation_row_inputs(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = (
            output_root / "codex-test-model" / "statutes" / "26" / "63" / "f.yaml"
        )
        generated_test = generated.with_name("f.test.yaml")
        imported = policy_repo / "statutes/26/151.yaml"
        generated.parent.mkdir(parents=True)
        imported.parent.mkdir(parents=True)
        imported.write_text(
            """format: rulespec/v1
rules:
  - name: exemption_individual_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          tin_included_on_return_claiming_exemption
          and is_dependent_under_section_152_of_taxpayer
"""
        )
        generated.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151#exemption_individual_eligible
rules: []
"""
        )
        generated_test.write_text(
            """- name: spouse_entitlements_require_section_151_exemption
  input:
    us:statutes/26/63/f#relation.spouse_of_taxpayer_for_subsection_f:
    - us:statutes/26/151#input.tin_included_on_return_claiming_exemption: true
      us:statutes/26/151#input.is_spouse_of_taxpayer: true
  output:
    us:statutes/26/63/f#spouse_aged_additional_amount_entitlement: holds
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                test_content = Path(path).with_name("f.test.yaml").read_text()
                if (
                    "us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer"
                    not in test_content
                ):
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                validator_name="ci",
                                error=(
                                    "Test case "
                                    "`spouse_entitlements_require_section_151_exemption` "
                                    "execution failed: missing input "
                                    "`is_dependent_under_section_152_of_taxpayer`"
                                ),
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
                validate_dependents=False,
            )

        assert ok is True
        assert issues == []
        updated = supplemental[Path("statutes/26/63/f.test.yaml")]
        assert (
            "      us:statutes/26/151#input.is_dependent_under_section_152_of_taxpayer: false"
            in updated
        )

    def test_apply_overlay_validation_repairs_dependent_proof_import_hashes(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/151.yaml"
        dependent = policy_repo / "statutes/26/63.yaml"
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text(
            """format: rulespec/v1
rules:
  - name: section_151_exemption_deduction
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          0
"""
        )
        dependent.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151
rules:
  - name: deductions_referred_to_in_subsection_b
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/151#section_151_exemption_deduction
              output: section_151_exemption_deduction
              hash: sha256:old
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          section_151_exemption_deduction
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "151.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                content = Path(path).read_text()
                if "hash: sha256:old" in content:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                error=(
                                    "Proof import hash mismatch: "
                                    "`deductions_referred_to_in_subsection_b` "
                                    "proof atom 0 imports "
                                    "`us:statutes/26/151#section_151_exemption_deduction` "
                                    "with `hash: sha256:old`."
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
        updated = supplemental[Path("statutes/26/63.yaml")]
        assert "hash: sha256:old" not in updated
        assert f"hash: sha256:{_sha256_file(generated)}" in updated

    def test_apply_overlay_validation_removes_obsolete_dependent_test_inputs(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/151.yaml"
        dependent = policy_repo / "statutes/26/63.yaml"
        dependent_test = dependent.with_name("63.test.yaml")
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/26/151\nrules: []\n"
        )
        dependent_test.write_text(
            """- name: old_151_input
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100000
    us:statutes/26/151#input.taxable_year_begins_after_2017: true
  output:
    us:statutes/26/63#taxable_income: 80000
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "151.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                content = Path(path).with_name("63.test.yaml").read_text()
                if "us:statutes/26/151#input.taxable_year_begins_after_2017" in content:
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                validator_name="ci",
                                error=(
                                    "Test case `old_151_input` input invalid: "
                                    "input "
                                    "`us:statutes/26/151#input.taxable_year_begins_after_2017` "
                                    "does not resolve to an input slot in "
                                    "statutes/26/151.yaml."
                                ),
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
        assert "us:statutes/26/151#input.taxable_year_begins_after_2017" not in updated
        assert "us:statutes/26/63#input.adjusted_gross_income" in updated

    def test_apply_overlay_validation_removes_dependent_import_output_placeholders(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/151.yaml"
        dependent = policy_repo / "statutes/26/63.yaml"
        dependent_test = dependent.with_name("63.test.yaml")
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules: []
"""
        )
        dependent.write_text(
            "format: rulespec/v1\nimports:\n  - us:statutes/26/151\nrules: []\n"
        )
        dependent_test.write_text(
            """- name: nonitemizer
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100000
    us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931: 0
  output:
    us:statutes/26/63#taxable_income: 80000
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "151.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                content = Path(path).with_name("63.test.yaml").read_text()
                if (
                    "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931"
                    in content
                ):
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                validator_name="ci",
                                error=(
                                    "Test case `nonitemizer` execution failed: "
                                    "dataset input "
                                    "`us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931` "
                                    "must use an absolute legal RuleSpec reference "
                                    "that resolves to an input slot, derived rule, "
                                    "or parameter in the compiled program"
                                ),
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
            "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931"
            not in updated
        )
        assert "us:statutes/26/63#input.adjusted_gross_income" in updated

    def test_apply_overlay_validation_removes_transitive_import_output_placeholders(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/151.yaml"
        dependent = policy_repo / "statutes/26/63.yaml"
        dependent_test = dependent.with_name("63.test.yaml")
        transitive = policy_repo / "statutes/26/224.yaml"
        generated.parent.mkdir(parents=True)
        dependent.parent.mkdir(parents=True)
        generated.write_text("format: rulespec/v1\nrules: []\n")
        transitive.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules: []
"""
        )
        dependent.write_text(
            """format: rulespec/v1
imports:
  - us:statutes/26/151
  - us:statutes/26/224
rules: []
"""
        )
        dependent_test.write_text(
            """- name: nonitemizer
  input:
    us:statutes/26/63#input.adjusted_gross_income: 100000
    us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931: 0
  output:
    us:statutes/26/63#taxable_income: 80000
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        class FakePipeline:
            def __init__(self, **_kwargs):
                pass

            def validate(self, path, *, skip_reviewers):
                assert skip_reviewers is True
                if Path(path).name == "151.yaml":
                    return SimpleNamespace(all_passed=True, results={})
                content = Path(path).with_name("63.test.yaml").read_text()
                if (
                    "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
                    in content
                ):
                    return SimpleNamespace(
                        all_passed=False,
                        results={
                            "ci": SimpleNamespace(
                                validator_name="ci",
                                error=(
                                    "Test case `nonitemizer` execution failed: "
                                    "dataset input "
                                    "`us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931` "
                                    "must use an absolute legal RuleSpec reference "
                                    "that resolves to an input slot, derived rule, "
                                    "or parameter in the compiled program"
                                ),
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
            "us:statutes/26/224#input.amount_excluded_from_gross_income_under_section_931"
            not in updated
        )
        assert "us:statutes/26/63#input.adjusted_gross_income" in updated

    def test_apply_overlay_validation_rejects_generated_filing_status_input(
        self, tmp_path
    ):
        output_root = tmp_path / "out"
        policy_repo = tmp_path / "rulespec-us"
        generated = output_root / "codex-test-model" / "statutes/26/63/c.yaml"
        generated_test = generated.with_name("c.test.yaml")
        generated.parent.mkdir(parents=True)
        policy_repo.mkdir()
        generated.write_text(
            """format: rulespec/v1
rules:
  - name: basic_standard_deduction_by_filing_status
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 1: 32200 else: 16100
"""
        )
        generated_test.write_text(
            """- name: joint_status_code
  input:
    us:statutes/26/63/c#input.filing_status: 1
  output:
    us:statutes/26/63/c#basic_standard_deduction_by_filing_status: 32200
"""
        )
        result = SimpleNamespace(output_file=str(generated), runner="codex-test-model")

        with patch("axiom_encode.cli.ValidatorPipeline") as mock_pipeline:
            ok, issues, supplemental = _validate_generated_encoding_in_policy_overlay(
                result,
                output_root=output_root,
                policy_repo_path=policy_repo,
                axiom_rules_path=tmp_path / "axiom-rules-engine",
            )

        assert ok is False
        assert supplemental == {}
        assert mock_pipeline.call_count == 0
        assert any(
            "Filing status is a derived legal classification" in issue
            for issue in issues
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
