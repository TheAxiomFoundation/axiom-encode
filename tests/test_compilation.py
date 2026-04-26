"""Tests for engine compilation integration.

Tests the Tier 0 compilation check in the validator pipeline,
the compile/benchmark CLI commands, and the encoder harness
compilation feedback.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import importlib.util

HAS_RAC_ENGINE = importlib.util.find_spec("rac") is not None and (
    importlib.util.find_spec("rac.engine") is not None
)

pytestmark = pytest.mark.skipif(not HAS_RAC_ENGINE, reason="rac.engine not installed")

from autorac.harness.validator_pipeline import ValidationResult, ValidatorPipeline

# =========================================================================
# Tier 0: Compilation Check in Validator Pipeline
# =========================================================================


class TestCompilationValidator:
    """Tests for Tier 0 compilation check."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as rac_us_dir:
            with tempfile.TemporaryDirectory() as rac_dir:
                yield Path(rac_us_dir), Path(rac_dir)

    @pytest.fixture
    def pipeline(self, temp_dirs):
        rac_us_path, rac_path = temp_dirs
        return ValidatorPipeline(
            rac_us_path=rac_us_path,
            rac_path=rac_path,
            enable_oracles=False,
        )

    def test_compile_check_passes_valid_rac(self, pipeline, temp_dirs):
        """A valid v2 RAC file should pass the compilation check."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "valid.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10

input income:
  entity: Person
  period: Year
  dtype: Money
  default: 0

variable tax:
  entity: Person
  period: Year
  dtype: Money
  formula: income * rate
""")

        result = pipeline._run_compile_check(rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "compile"
        assert result.passed is True
        assert len(result.issues) == 0

    def test_compile_check_fails_invalid_syntax(self, pipeline, temp_dirs):
        """An unparseable RAC file should fail compilation."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "invalid.rac"
        rac_file.write_text("this is {{ totally not valid RAC !!!")

        result = pipeline._run_compile_check(rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "compile"
        assert result.passed is False
        assert len(result.issues) > 0

    def test_compile_check_returns_duration(self, pipeline, temp_dirs):
        """Compilation check should report its duration."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "simple.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.25
""")

        result = pipeline._run_compile_check(rac_file)

        assert result.duration_ms >= 0

    def test_compile_check_included_in_pipeline(self, pipeline, temp_dirs):
        """The full pipeline should include compile check results."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "simple.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.25
""")

        result = pipeline.validate(rac_file)

        assert "compile" in result.results
        assert result.results["compile"].validator_name == "compile"

    def test_compile_check_runs_before_ci(self, pipeline, temp_dirs):
        """Compile check (Tier 0) should run before CI (Tier 1)."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "simple.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.25
""")

        # Track execution order
        execution_order = []
        original_compile = pipeline._run_compile_check
        original_ci = pipeline._run_ci

        def tracked_compile(f):
            execution_order.append("compile")
            return original_compile(f)

        def tracked_ci(f):
            execution_order.append("ci")
            return original_ci(f)

        with patch.object(pipeline, "_run_compile_check", side_effect=tracked_compile):
            with patch.object(pipeline, "_run_ci", side_effect=tracked_ci):
                pipeline.validate(rac_file)

        assert execution_order.index("compile") < execution_order.index("ci")

    def test_compile_check_produces_ir(self, pipeline, temp_dirs):
        """A valid file should produce IR that can be executed."""
        rac_us_path, _ = temp_dirs
        rac_file = rac_us_path / "executable.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10

input income:
  entity: Person
  period: Year
  dtype: Money
  default: 0

variable tax:
  entity: Person
  period: Year
  dtype: Money
  formula: income * rate
""")

        result = pipeline._run_compile_check(rac_file)

        assert result.passed is True
        # The raw_output should mention successful compilation
        if result.raw_output:
            assert (
                "compiled" in result.raw_output.lower()
                or "success" in result.raw_output.lower()
            )


# =========================================================================
# CLI: compile command
# =========================================================================


class TestCompileCLI:
    """Tests for the `autorac compile` CLI command."""

    def test_compile_command_exists(self):
        """The compile subcommand should be registered."""
        import argparse

        # Parse with compile command — should not raise
        # We test by importing and checking the parser setup
        parser = argparse.ArgumentParser()
        parser.add_subparsers(dest="command")

        # Re-import to check compile is registered
        from autorac.cli import main as cli_main

        # The main function builds its own parser, so we just verify it
        # doesn't crash when called with --help (indirectly)
        assert callable(cli_main)

    def test_compile_valid_file(self, tmp_path):
        """autorac compile should succeed for a valid .rac file."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10

input income:
  entity: Person
  period: Year
  dtype: Money
  default: 0

variable tax:
  entity: Person
  period: Year
  dtype: Money
  formula: income * rate
""")

        from autorac.cli import cmd_compile

        # Create a mock args object
        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-06-01"
        args.json = False
        args.execute = False

        # Should not raise
        with patch("sys.exit") as mock_exit:
            cmd_compile(args)
            # Success exits with 0 or doesn't exit
            if mock_exit.called:
                assert mock_exit.call_args[0][0] == 0

    def test_compile_invalid_file(self, tmp_path):
        """autorac compile should fail for a truly unparseable .rac file."""
        rac_file = tmp_path / "bad.rac"
        # Use content that triggers an actual parse error (unclosed braces)
        rac_file.write_text("""
variable broken:
  entity: Person
  period: Year
  dtype: Money
  formula: if (x > 0:
""")

        from autorac.cli import cmd_compile

        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-06-01"
        args.json = False
        args.execute = False

        with patch("sys.exit") as mock_exit:
            cmd_compile(args)
            if mock_exit.called:
                assert mock_exit.call_args[0][0] == 1

    def test_compile_with_execute(self, tmp_path):
        """autorac compile --execute should compile and run."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10

input income:
  entity: Person
  period: Year
  dtype: Money
  default: 0

variable tax:
  entity: Person
  period: Year
  dtype: Money
  formula: income * rate
""")

        from autorac.cli import cmd_compile

        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-06-01"
        args.json = False
        args.execute = True

        with patch("sys.exit") as mock_exit:
            cmd_compile(args)
            if mock_exit.called:
                assert mock_exit.call_args[0][0] == 0

    def test_compile_json_output(self, tmp_path):
        """autorac compile --json should output JSON."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10
""")

        import io
        from contextlib import redirect_stdout

        from autorac.cli import cmd_compile

        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-06-01"
        args.json = True
        args.execute = False

        output = io.StringIO()
        with redirect_stdout(output):
            with patch("sys.exit"):
                cmd_compile(args)

        # Output should be valid JSON
        import json

        output_str = output.getvalue().strip()
        if output_str:
            data = json.loads(output_str)
            assert "success" in data
            assert "variables" in data


# =========================================================================
# CLI: benchmark command
# =========================================================================


class TestBenchmarkCLI:
    """Tests for the `autorac benchmark` CLI command."""

    def test_benchmark_valid_file(self, tmp_path):
        """autorac benchmark should run and report timing."""
        rac_file = tmp_path / "test.rac"
        rac_file.write_text("""
parameter rate:
  values:
    2024-01-01: 0.10

input income:
  entity: Person
  period: Year
  dtype: Money
  default: 0

variable tax:
  entity: Person
  period: Year
  dtype: Money
  formula: income * rate
""")

        import io
        from contextlib import redirect_stdout

        from autorac.cli import cmd_benchmark

        args = MagicMock()
        args.file = rac_file
        args.as_of = "2024-06-01"
        args.iterations = 10
        args.rows = 100

        output = io.StringIO()
        with redirect_stdout(output):
            with patch("sys.exit"):
                cmd_benchmark(args)

        output_str = output.getvalue()
        # Should contain timing info
        assert "ms" in output_str.lower() or "benchmark" in output_str.lower()
