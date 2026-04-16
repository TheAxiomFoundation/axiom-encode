"""
Tests for encoder harness with Claude Code CLI integration.

These tests verify:
1. _get_predictions() correctly calls Claude CLI and parses JSON response
2. _encode() generates valid RAC content via Claude CLI
3. _get_suggestions() analyzes validation failures and returns improvements
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from autorac.harness.encoder_harness import (
    EncoderConfig,
    EncoderHarness,
    run_claude_code,
)
from autorac.harness.validator_pipeline import PipelineResult, ValidationResult


@pytest.fixture
def temp_config():
    """Create a temporary encoder configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "rac-us").mkdir()
        (tmpdir_path / "rac").mkdir()
        yield EncoderConfig(
            rac_us_path=tmpdir_path / "rac-us",
            rac_path=tmpdir_path / "rac",
            db_path=tmpdir_path / "experiments.db",
            enable_oracles=False,  # Disable oracles for faster tests
        )


class TestRunClaudeCode:
    """Tests for run_claude_code function."""

    def test_returns_output_and_returncode(self):
        """Test that function returns tuple of (output, returncode)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="test output", stderr="", returncode=0)

            output, code = run_claude_code("test prompt")

            assert "test output" in output
            assert code == 0

    def test_handles_timeout(self):
        """Test timeout handling."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)

            output, code = run_claude_code("test", timeout=60)

            assert "Timeout" in output
            assert code == 1

    def test_handles_missing_cli(self):
        """Test handling when claude CLI is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output, code = run_claude_code("test")

            assert "not found" in output
            assert code == 1


class TestRunClaudeCodeAdditional:
    """Additional tests for run_claude_code covering missing branches."""

    def test_no_plugin_dir_flag(self):
        """Test run_claude_code does NOT include --plugin-dir (self-contained)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="output", stderr="", returncode=0)
            output, code = run_claude_code("test")

            cmd = mock_run.call_args[0][0]
            assert "--plugin-dir" not in cmd
            assert "--agent" not in cmd

    def test_generic_exception(self):
        """Test run_claude_code handles generic exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")
            output, code = run_claude_code("test")

            assert "Error" in output
            assert code == 1


class TestEncoderConfigSelfContained:
    """Tests for self-contained EncoderConfig (no plugin dependency)."""

    def test_config_has_no_plugin_path(self, tmp_path):
        """Test EncoderConfig has no rac_plugin_path (self-contained)."""
        rac_us = tmp_path / "rac-us"
        rac_us.mkdir()
        rac = tmp_path / "rac"
        rac.mkdir()

        config = EncoderConfig(
            rac_us_path=rac_us,
            rac_path=rac,
        )
        assert not hasattr(config, "rac_plugin_path")


class TestIterateUntilPass:
    """Tests for iterate_until_pass method."""

    def test_iterate_passes_first_time(self, temp_config):
        """Test iterate_until_pass when first iteration passes."""
        harness = EncoderHarness(temp_config)

        prediction_json = json.dumps(
            {
                "rac_reviewer": 8.0,
                "confidence": 0.7,
            }
        )
        rac_content = "test_var:\n  entity: TaxUnit\n"

        # Mock the full pipeline validate to return all_passed=True
        passing_pipeline_result = PipelineResult(
            results={
                "ci": ValidationResult("ci", True, None, [], 100),
                "rac_reviewer": ValidationResult("rac_reviewer", True, 8.0, [], 500),
                "formula_reviewer": ValidationResult(
                    "formula_reviewer", True, 8.0, [], 500
                ),
                "parameter_reviewer": ValidationResult(
                    "parameter_reviewer", True, 8.0, [], 500
                ),
                "integration_reviewer": ValidationResult(
                    "integration_reviewer", True, 8.0, [], 500
                ),
            },
            total_duration_ms=600,
            all_passed=True,
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch.object(
                harness.pipeline, "validate", return_value=passing_pipeline_result
            ):
                mock_encoder.return_value = (prediction_json, 0)
                mock_encoder.side_effect = [
                    (prediction_json, 0),  # predict
                    (rac_content, 0),  # encode
                ]

                output_path = temp_config.rac_us_path / "test.rac"
                iterations = harness.iterate_until_pass(
                    citation="26 USC 1",
                    statute_text="Test statute",
                    output_path=output_path,
                )

                assert len(iterations) == 1
                run, result = iterations[0]
                assert run.citation == "26 USC 1"
                assert result.all_passed is True

    def test_iterate_multiple_times(self, temp_config):
        """Test iterate_until_pass with multiple iterations."""
        harness = EncoderHarness(temp_config)

        prediction_json = json.dumps({"rac_reviewer": 8.0, "confidence": 0.7})
        rac_content = "test_var:\n  entity: TaxUnit\n"
        suggestions_json = json.dumps(
            [
                {
                    "category": "validator",
                    "description": "fix",
                    "predicted_impact": "high",
                }
            ]
        )

        call_count = 0

        def mock_encoder_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (prediction_json, 0)
            elif call_count <= 4:
                return (rac_content, 0)
            else:
                return (suggestions_json, 0)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch(
                "autorac.harness.validator_pipeline.run_claude_code"
            ) as mock_validator:
                # First iteration fails, second passes
                fail_result = '{"score": 4.0, "passed": false, "issues": ["Bad formula"], "reasoning": "Wrong"}'
                pass_result = (
                    '{"score": 8.0, "passed": true, "issues": [], "reasoning": "Good"}'
                )

                mock_encoder.side_effect = [
                    (prediction_json, 0),  # 1st predict
                    (rac_content, 0),  # 1st encode
                    (prediction_json, 0),  # 2nd predict
                    (rac_content, 0),  # 2nd encode
                ]
                mock_validator.side_effect = [
                    (fail_result, 0),  # 1st validator fail (rac)
                    (fail_result, 0),  # 1st validator fail (formula)
                    (fail_result, 0),  # 1st validator fail (param)
                    (fail_result, 0),  # 1st validator fail (integration)
                    (pass_result, 0),  # 2nd validator pass
                    (pass_result, 0),
                    (pass_result, 0),
                    (pass_result, 0),
                ]

                output_path = temp_config.rac_us_path / "test.rac"
                iterations = harness.iterate_until_pass(
                    citation="26 USC 1",
                    statute_text="Test statute",
                    output_path=output_path,
                )

                assert len(iterations) >= 1


class TestRunEncodingExperiment:
    """Tests for run_encoding_experiment convenience function."""

    def test_run_encoding_experiment(self, tmp_path):
        """Test the convenience function."""
        from autorac.harness.encoder_harness import run_encoding_experiment

        rac_us = tmp_path / "rac-us"
        rac_us.mkdir()
        rac = tmp_path / "rac"
        rac.mkdir()
        output_dir = rac_us / "statute" / "26"
        output_dir.mkdir(parents=True)

        prediction_json = json.dumps({"rac_reviewer": 8.0, "confidence": 0.7})
        rac_content = "test_var:\n  entity: TaxUnit\n"

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch(
                "autorac.harness.validator_pipeline.run_claude_code"
            ) as mock_validator:
                mock_encoder.side_effect = [
                    (prediction_json, 0),
                    (rac_content, 0),
                ]
                mock_validator.return_value = (
                    '{"score": 8.0, "passed": true, "issues": [], "reasoning": "Good"}',
                    0,
                )

                config = EncoderConfig(
                    rac_us_path=rac_us,
                    rac_path=rac,
                    db_path=tmp_path / "experiments.db",
                    enable_oracles=False,
                )

                iterations = run_encoding_experiment(
                    citation="26 USC 1",
                    statute_text="Test statute",
                    output_dir=output_dir,
                    config=config,
                )

                assert len(iterations) >= 1

    def test_run_encoding_experiment_auto_config(self, tmp_path):
        """Test auto-config derivation in run_encoding_experiment."""
        from autorac.harness.encoder_harness import run_encoding_experiment

        # Create rac-us directory structure
        rac_us = tmp_path / "rac-us"
        rac_us.mkdir()
        output_dir = rac_us / "statute" / "26"
        output_dir.mkdir(parents=True)
        # rac sibling
        rac = tmp_path / "rac"
        rac.mkdir()

        prediction_json = json.dumps({"rac_reviewer": 8.0, "confidence": 0.7})
        rac_content = "test_var:\n  entity: TaxUnit\n"

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch(
                "autorac.harness.validator_pipeline.run_claude_code"
            ) as mock_validator:
                mock_encoder.side_effect = [
                    (prediction_json, 0),
                    (rac_content, 0),
                ]
                mock_validator.return_value = (
                    '{"score": 8.0, "passed": true, "issues": [], "reasoning": "Good"}',
                    0,
                )

                iterations = run_encoding_experiment(
                    citation="26 USC 1",
                    statute_text="Test statute",
                    output_dir=output_dir,
                )

                assert len(iterations) >= 1


class TestEncodeReadsFile:
    """Test _encode reads from file when output_path exists."""

    def test_encode_reads_existing_file(self, temp_config):
        """Test _encode reads RAC content from file when it exists."""
        harness = EncoderHarness(temp_config)
        output_path = temp_config.rac_us_path / "output.rac"

        # Mock run_claude_code but also create the file as a side effect
        def mock_run_and_create_file(*args, **kwargs):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("file_var:\n  entity: TaxUnit\n  dtype: Money\n")
            return ("CLI output ignored", 0)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.side_effect = mock_run_and_create_file

            result = harness._encode("26 USC 1", "Test statute", output_path)
            assert "file_var" in result


class TestGetLessonsNoFailures:
    """Test _get_lessons with no failures (all passed)."""

    def test_lessons_empty_when_all_passed(self, temp_config):
        """Test _get_lessons returns empty string when all passed."""
        harness = EncoderHarness(temp_config)

        passing_result = PipelineResult(
            results={
                "ci": ValidationResult("ci", True, None, [], 100),
            },
            total_duration_ms=100,
            all_passed=True,
        )

        result = harness._get_lessons("26 USC 32", "content", passing_result)
        assert result == ""


class TestGetLessonsWithFailures:
    """Tests for _get_lessons method."""

    def test_returns_lessons_text(self, temp_config):
        """Test that lessons text is returned from CLI."""
        harness = EncoderHarness(temp_config)

        failing_result = PipelineResult(
            results={
                "ci": ValidationResult(
                    "ci", False, None, ["Parse error"], 100, error="Parse error"
                ),
            },
            total_duration_ms=100,
            all_passed=False,
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (
                "The encoding failed because bracket syntax was incorrect.",
                0,
            )

            result = harness._get_lessons("26 USC 32", "content", failing_result)
            assert "bracket syntax" in result

    def test_handles_cli_failure_gracefully(self, temp_config):
        """Test that CLI failures return basic failure summary."""
        harness = EncoderHarness(temp_config)

        failing_result = PipelineResult(
            results={
                "ci": ValidationResult(
                    "ci", False, None, ["Parse error"], 100, error="Parse error"
                ),
            },
            total_duration_ms=100,
            all_passed=False,
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.side_effect = Exception("CLI error")

            result = harness._get_lessons("26 USC 32", "content", failing_result)
            assert "ci failed" in result


class TestEncode:
    """Tests for _encode method."""

    def test_generates_valid_rac_content(self, temp_config):
        """Test that valid RAC content is generated and saved."""
        harness = EncoderHarness(temp_config)

        expected_rac = '''"""
Sample statute text
"""

sample_var:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Sample Variable"
  formula: |
    return 0
  default: 0
'''

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (expected_rac, 0)

            output_path = temp_config.rac_us_path / "test.rac"
            result = harness._encode("26 USC 1", "Sample statute text", output_path)

            assert '"""' in result
            assert "sample_var:" in result
            assert output_path.exists()

    def test_strips_markdown_code_blocks(self, temp_config):
        """Test that markdown code blocks are stripped from response."""
        harness = EncoderHarness(temp_config)

        response_with_markdown = '''```yaml
"""
Statute text
"""

test:
  dtype: Money
```'''

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (response_with_markdown, 0)

            output_path = temp_config.rac_us_path / "test.rac"
            result = harness._encode("26 USC 1", "Statute text", output_path)

            assert not result.startswith("```")
            assert not result.endswith("```")
            assert '"""' in result

    def test_creates_output_directory(self, temp_config):
        """Test that output directory is created if it doesn't exist."""
        harness = EncoderHarness(temp_config)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = ("text: test", 0)

            output_path = temp_config.rac_us_path / "nested" / "dir" / "test.rac"
            harness._encode("26 USC 1", "Statute text", output_path)

            assert output_path.parent.exists()

    def test_returns_fallback_on_cli_failure(self, temp_config):
        """Test that a valid fallback is returned on CLI failure."""
        harness = EncoderHarness(temp_config)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.side_effect = Exception("CLI error")

            output_path = temp_config.rac_us_path / "fallback.rac"
            result = harness._encode("26 USC 32", "EITC statute", output_path)

            assert '"""' in result
            assert "TODO(#issue-needed): Implement formula" in result
            assert output_path.exists()


class TestEncodeWithFeedback:
    """Integration tests for encode_with_feedback method."""

    def test_full_encode_cycle(self, temp_config):
        """Test the full encode-validate-log cycle."""
        harness = EncoderHarness(temp_config)

        prediction_json = json.dumps(
            {
                "rac_reviewer": 8.0,
                "formula_reviewer": 7.5,
                "parameter_reviewer": 8.0,
                "integration_reviewer": 7.5,
                "ci_pass": True,
                "confidence": 0.7,
            }
        )

        rac_content = '''"""
Test statute
"""

test:
  entity: TaxUnit
  dtype: Money
  period: Year
  formula: |
    return 0
'''

        # Mock run_claude_code for both encoder and validator
        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch(
                "autorac.harness.validator_pipeline.run_claude_code"
            ) as mock_validator:
                # Encoder returns prediction then RAC content
                mock_encoder.side_effect = [
                    (prediction_json, 0),  # _get_predictions
                    (rac_content, 0),  # _encode
                ]

                # Validators return passing scores
                mock_validator.return_value = (
                    '{"score": 8.0, "passed": true, "issues": [], "reasoning": "Good"}',
                    0,
                )

                output_path = temp_config.rac_us_path / "statute" / "26" / "1.rac"

                run, result = harness.encode_with_feedback(
                    citation="26 USC 1",
                    statute_text="Test statute text",
                    output_path=output_path,
                )

                assert run.citation == "26 USC 1"
                assert run.predicted is not None
                assert run.actual is not None
                assert output_path.exists()
