"""
Validator Pipeline - 3-tier validation architecture.

Tiers (run in order):
1. CI checks (rac pytest) - instant, catches syntax/format errors
2. External oracles (PolicyEngine, TAXSIM) - fast (~10s), generates comparison data
3. LLM reviewers (rac, formula, parameter, integration) - uses oracle context

Oracles run BEFORE LLM reviewers because:
- They're fast and free (no API costs)
- They generate rich comparison context for LLM analysis
- LLMs can diagnose WHY discrepancies exist, not just that they exist

Uses Claude Code CLI (subprocess) for reviewer agents - cheaper than direct API.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from autorac.constants import REVIEWER_CLI_MODEL

from .encoding_db import EncodingDB, ReviewResult, ReviewResults


def run_claude_code(
    prompt: str,
    model: str = REVIEWER_CLI_MODEL,
    timeout: int = 120,
    cwd: Optional[Path] = None,
) -> tuple[str, int]:
    """
    Run Claude Code CLI as subprocess.

    Returns:
        Tuple of (output text, return code)
    """
    cmd = ["claude", "--print", "--model", model, "-p", prompt]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s", 1
    except FileNotFoundError:
        return "Claude CLI not found", 1
    except Exception as e:
        return f"Error: {e}", 1


_REVIEW_JSON_FORMAT = """
Output your review as JSON:
{
  "score": <float 1-10>,
  "passed": <boolean>,
  "issues": ["issue1", "issue2"],
  "reasoning": "<brief explanation>"
}
"""

RAC_REVIEWER_PROMPT = (
    """You are an expert RAC (Rules as Code) reviewer specializing in structure and legal citations.

Review the RAC file for:
1. **Structure**: Proper definition with `name:` (no `variable`/`parameter` keywords), all required fields (entity, period, dtype, formula)
2. **Legal Citations**: Accurate citation format (e.g., "26 USC 32(a)(1)")
3. **Imports**: Correct import paths using path#name syntax
4. **Entity Hierarchy**: Proper entity usage (Person < TaxUnit < Household)
5. **DSL Compliance**: Unified syntax — `name:`, `from yyyy-mm-dd:` temporal entries, `\"\"\"...\"\"\"` text blocks, tests in `.rac.test` files
"""
    + _REVIEW_JSON_FORMAT
)

FORMULA_REVIEWER_PROMPT = (
    """You are an expert formula reviewer for RAC (Rules as Code) encodings.

Review the RAC file formulas for:
1. **Logic Correctness**: Does the formula correctly implement the statute logic?
2. **Edge Cases**: Are edge cases handled (zero values, negative numbers, thresholds)?
3. **Circular Dependencies**: No circular references between definitions
4. **Return Statements**: Every code path returns a value
5. **Type Consistency**: Return type matches declared dtype
6. **Temporal Values**: Uses `from yyyy-mm-dd:` syntax for date-based entries
"""
    + _REVIEW_JSON_FORMAT
)

PARAMETER_REVIEWER_PROMPT = (
    """You are an expert reviewer for RAC (Rules as Code) encodings, focused on policy values and parameters.

Review the RAC file for policy value usage:
1. **No Magic Numbers**: Only -1, 0, 1, 2, 3 allowed as literals. All other values must be defined as named entries.
2. **Sourcing**: Policy values should reference authoritative sources
3. **Time-Varying Values**: Rate thresholds and amounts should use `from yyyy-mm-dd:` temporal entries
4. **Reference Format**: Correct reference syntax (unified `name:` format, no `parameter` keyword)
5. **Default Values**: Appropriate defaults for optional inputs
"""
    + _REVIEW_JSON_FORMAT
)

INTEGRATION_REVIEWER_PROMPT = (
    """You are an expert integration reviewer for RAC (Rules as Code) encodings.

Review the RAC file for integration quality:
1. **Test Coverage**: At least 3-5 test cases in the companion `.rac.test` file covering normal and edge cases
2. **Dependency Resolution**: All imports can be resolved
3. **Cross-Definition Consistency**: Named definitions work together correctly
4. **Documentation**: Clear labels and descriptions
5. **Completeness**: Full statute implementation, no TODO placeholders
6. **Syntax**: Unified syntax — `name:`, `from yyyy-mm-dd:` temporal entries, tests in `.rac.test` files
"""
    + _REVIEW_JSON_FORMAT
)


@dataclass
class ValidationResult:
    """Result from a single validator."""

    validator_name: str
    passed: bool
    score: Optional[float] = None  # 0-10 for reviewers, 0-1 for oracles
    issues: list[str] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregated results from all validators."""

    results: dict[str, ValidationResult]
    total_duration_ms: int
    all_passed: bool
    oracle_context: dict = field(
        default_factory=dict
    )  # Context passed to LLM reviewers

    def to_review_results(self) -> ReviewResults:
        """Convert pipeline results to ReviewResults for encoding DB."""
        reviews = []
        for name in [
            "rac_reviewer",
            "formula_reviewer",
            "parameter_reviewer",
            "integration_reviewer",
        ]:
            vr = self.results.get(name, ValidationResult("", False))
            reviews.append(
                ReviewResult(
                    reviewer=name,
                    passed=vr.passed,
                    items_checked=len(vr.issues) + (1 if vr.passed else 0),
                    items_passed=1 if vr.passed else 0,
                    critical_issues=[issue for issue in (vr.issues or [])]
                    if not vr.passed
                    else [],
                    important_issues=vr.issues or [] if vr.passed else [],
                )
            )

        return ReviewResults(
            reviews=reviews,
            policyengine_match=self.results.get(
                "policyengine", ValidationResult("", False)
            ).score,
            taxsim_match=self.results.get("taxsim", ValidationResult("", False)).score,
            oracle_context=self.oracle_context,
        )

    def to_actual_scores(self) -> ReviewResults:
        """Backward compat alias for to_review_results."""
        return self.to_review_results()

    @property
    def ci_pass(self) -> bool:
        """Check if CI passed."""
        return self.results.get("ci", ValidationResult("", False)).passed


class ValidatorPipeline:
    """Runs validators in 3 tiers with session event logging."""

    def __init__(
        self,
        rac_us_path: Path,
        rac_path: Path,
        enable_oracles: bool = True,
        max_workers: int = 4,
        encoding_db: Optional[EncodingDB] = None,
        session_id: Optional[str] = None,
    ):
        self.rac_us_path = Path(rac_us_path)
        self.rac_path = Path(rac_path)
        self.enable_oracles = enable_oracles
        self.max_workers = max_workers
        self.encoding_db = encoding_db
        self.session_id = session_id

    def _log_event(
        self, event_type: str, content: str = "", metadata: Optional[dict] = None
    ):
        """Log a validation event if session tracking is enabled."""
        if self.encoding_db and self.session_id:
            self.encoding_db.log_event(
                session_id=self.session_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            )

    def validate(self, rac_file: Path) -> PipelineResult:
        """Run 4-tier validation on a RAC file.

        Tiers run in order:
        0. Compile check - can the .rac file compile to engine IR?
        1. CI checks (instant) - parse, lint, inline tests, rac pytest validation
        2. Oracles (fast, ~10s) - PolicyEngine + TAXSIM comparison data
        3. LLM reviewers (uses oracle context) - diagnose issues

        Oracle results are passed to LLM reviewers as context.
        Each tier is logged as session events with timestamps.
        """
        start = time.time()
        results = {}

        # Tier 0: Compile check (fast, catches structural errors early)
        self._log_event(
            "validation_compile_start",
            f"Starting compilation check for {rac_file.name}",
        )
        compile_start = time.time()
        results["compile"] = self._run_compile_check(rac_file)
        self._log_event(
            "validation_compile_end",
            "Compilation check complete",
            {
                "passed": results["compile"].passed,
                "issues": results["compile"].issues,
                "duration_ms": int((time.time() - compile_start) * 1000),
            },
        )

        # Tier 1: CI checks (instant, blocks further validation if fails)
        self._log_event(
            "validation_ci_start", f"Starting CI validation for {rac_file.name}"
        )
        ci_start = time.time()
        try:
            results["ci"] = self._run_ci(rac_file)
        except Exception as e:
            results["ci"] = ValidationResult(
                validator_name="ci",
                passed=False,
                error=str(e),
                issues=[str(e)],
            )
        self._log_event(
            "validation_ci_end",
            "CI validation complete",
            {
                "passed": results["ci"].passed,
                "issues": results["ci"].issues,
                "duration_ms": int((time.time() - ci_start) * 1000),
            },
        )

        # Tier 2: Oracles (parallel, fast, generates comparison context)
        oracle_context = {}
        if self.enable_oracles:
            self._log_event(
                "validation_oracle_start", "Starting oracle validation (PE + TAXSIM)"
            )
            oracle_start = time.time()

            oracle_validators = {
                "policyengine": lambda: self._run_policyengine(rac_file),
                "taxsim": lambda: self._run_taxsim(rac_file),
            }

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(fn): name for name, fn in oracle_validators.items()
                }

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                        # Build context for LLM reviewers (with full details)
                        oracle_context[name] = {
                            "score": results[name].score,
                            "passed": results[name].passed,
                            "issues": results[name].issues,
                            "duration_ms": results[name].duration_ms,
                        }
                    except Exception as e:
                        results[name] = ValidationResult(
                            validator_name=name,
                            passed=False,
                            error=str(e),
                        )
                        oracle_context[name] = {
                            "score": None,
                            "passed": False,
                            "issues": [str(e)],
                            "error": str(e),
                        }

            self._log_event(
                "validation_oracle_end",
                "Oracle validation complete",
                {
                    "oracle_context": oracle_context,
                    "duration_ms": int((time.time() - oracle_start) * 1000),
                },
            )

        # Tier 3: LLM reviewers (parallel, use oracle context)
        self._log_event(
            "validation_llm_start",
            "Starting LLM reviewers with oracle context",
            {
                "oracle_context_summary": {
                    k: v.get("score") for k, v in oracle_context.items()
                },
            },
        )
        llm_start = time.time()

        llm_validators = {
            "rac_reviewer": lambda: self._run_reviewer(
                "rac-reviewer", rac_file, oracle_context
            ),
            "formula_reviewer": lambda: self._run_reviewer(
                "Formula Reviewer", rac_file, oracle_context
            ),
            "parameter_reviewer": lambda: self._run_reviewer(
                "Parameter Reviewer", rac_file, oracle_context
            ),
            "integration_reviewer": lambda: self._run_reviewer(
                "Integration Reviewer", rac_file, oracle_context
            ),
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fn): name for name, fn in llm_validators.items()}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = ValidationResult(
                        validator_name=name,
                        passed=False,
                        error=str(e),
                    )

        self._log_event(
            "validation_llm_end",
            "LLM reviewers complete",
            {
                "scores": {
                    k: results[k].score for k in llm_validators.keys() if k in results
                },
                "duration_ms": int((time.time() - llm_start) * 1000),
            },
        )

        total_duration = int((time.time() - start) * 1000)
        all_passed = all(r.passed for r in results.values())

        return PipelineResult(
            results=results,
            total_duration_ms=total_duration,
            all_passed=all_passed,
            oracle_context=oracle_context,
        )

    def _run_compile_check(self, rac_file: Path) -> ValidationResult:
        """Tier 0: Compile check — can the .rac file compile to engine IR?

        Parses the v2 .rac file, converts it to engine format, and compiles
        to IR. Catches type errors, missing dependencies, and circular
        references earlier than CI.
        """
        start = time.time()
        issues = []

        try:
            from datetime import date

            from rac.dsl_parser import parse_dsl
            from rac.engine import compile as engine_compile
            from rac.engine.converter import convert_v2_to_engine_module

            # Step 1: Parse v2 format
            rac_content = rac_file.read_text()
            v2_module = parse_dsl(rac_content)

            # Step 2: Convert to engine module
            # Derive module_path from file path (e.g., statute/26/32 -> 26/32)
            module_path = rac_file.stem
            engine_module = convert_v2_to_engine_module(
                v2_module, module_path=module_path
            )

            # Step 3: Compile to IR
            ir = engine_compile([engine_module], as_of=date.today())

            duration = int((time.time() - start) * 1000)
            var_count = len(ir.variables)

            return ValidationResult(
                validator_name="compile",
                passed=True,
                issues=[],
                duration_ms=duration,
                raw_output=f"Successfully compiled {var_count} variables to engine IR",
            )

        except Exception as e:
            duration = int((time.time() - start) * 1000)
            issues.append(f"Compilation failed: {e}")

            return ValidationResult(
                validator_name="compile",
                passed=False,
                issues=issues,
                duration_ms=duration,
                error=str(e),
            )

    def _run_ci(self, rac_file: Path) -> ValidationResult:
        """Run CI checks: parse, lint, inline tests."""
        start = time.time()
        issues = []

        # 1. Parse check
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"""
import sys
sys.path.insert(0, '{self.rac_path}/src')
from rac import parse_file
parse_file('{rac_file}')
print('PARSE_OK')
""",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if "PARSE_OK" not in result.stdout:
                issues.append(f"Parse error: {result.stderr}")
        except subprocess.TimeoutExpired:
            issues.append("Parse timeout")
        except Exception as e:
            issues.append(f"Parse exception: {e}")

        # 2. Run inline tests
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"""
import sys
sys.path.insert(0, '{self.rac_path}/src')
from rac.test_runner import run_tests_for_file
report = run_tests_for_file('{rac_file}')
print(f'TESTS:{{report.passed}}/{{report.total}}')
""",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if "TESTS:" in result.stdout:
                test_line = [
                    line for line in result.stdout.split("\n") if "TESTS:" in line
                ][0]
                passed, total = test_line.split(":")[1].split("/")
                if int(passed) < int(total):
                    issues.append(f"Tests failed: {passed}/{total}")
            else:
                issues.append(f"Test error: {result.stderr}")
        except subprocess.TimeoutExpired:
            issues.append("Test timeout")
        except Exception as e:
            issues.append(f"Test exception: {e}")

        # 3. Run rac validation tests (param values in text, hardcoded values, etc.)
        try:
            # Set STATUTE_DIR to a temp dir containing just this file
            # so pytest parametrization picks up only this file
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy the file to temp dir
                tmp_file = Path(tmpdir) / rac_file.name
                shutil.copy(rac_file, tmp_file)

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        f"{self.rac_path}/tests/rac_validation/",
                        "-v",
                        "--tb=short",
                        f"-k={rac_file.stem}",  # Filter to just this file
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "STATUTE_DIR": tmpdir},
                    cwd=str(self.rac_path),
                )

                # Parse pytest output for failures
                if result.returncode != 0:
                    # Extract FAILED lines with test names (dedupe)
                    seen = set()
                    for line in result.stdout.split("\n"):
                        if "FAILED" in line and "::" in line:
                            # Format: "test_file.py::TestClass::test_name[param] FAILED"
                            parts = line.split("::")
                            if len(parts) >= 2:
                                test_part = parts[-1].split(" FAILED")[0].strip()
                                if test_part not in seen:
                                    seen.add(test_part)
                                    issues.append(f"Validation failed: {test_part}")
        except subprocess.TimeoutExpired:
            issues.append("Validation timeout")
        except Exception as e:
            issues.append(f"Validation exception: {e}")

        duration = int((time.time() - start) * 1000)

        return ValidationResult(
            validator_name="ci",
            passed=len(issues) == 0,
            issues=issues,
            duration_ms=duration,
            error=issues[0] if issues else None,
        )

    def _run_reviewer(
        self,
        reviewer_type: str,
        rac_file: Path,
        oracle_context: Optional[dict] = None,
    ) -> ValidationResult:
        """Run a reviewer agent via Claude Code CLI with oracle context.

        Args:
            reviewer_type: Type of reviewer (rac-reviewer, formula-reviewer, etc.)
            rac_file: Path to the RAC file to review
            oracle_context: Results from oracle validators (PE, TAXSIM) for context

        Returns:
            ValidationResult with score, issues, and raw output
        """
        start = time.time()

        # Read RAC file content
        try:
            rac_content = Path(rac_file).read_text()
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name=reviewer_type,
                passed=False,
                score=0.0,
                issues=[f"Failed to read RAC file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Build review prompt based on reviewer type
        review_focus = {
            "rac-reviewer": "structure, legal citations, imports, entity hierarchy, DSL compliance",
            "formula-reviewer": "logic correctness, edge cases, circular dependencies, return statements, type consistency",
            "parameter-reviewer": "no magic numbers (only -1,0,1,2,3 allowed), parameter sourcing, time-varying values",
            "integration-reviewer": "test coverage, dependency resolution, documentation, completeness",
            "Formula Reviewer": "logic correctness, edge cases, circular dependencies, return statements",
            "Parameter Reviewer": "no magic numbers (only -1,0,1,2,3 allowed), parameter sourcing",
            "Integration Reviewer": "test coverage, dependency resolution, documentation",
        }.get(reviewer_type, "overall quality")

        # Build oracle context section if available
        oracle_section = ""
        if oracle_context:
            oracle_section = "\n## Oracle Validation Results (use to diagnose issues)\n"
            for oracle_name, ctx in oracle_context.items():
                oracle_section += f"\n### {oracle_name.upper()}\n"
                oracle_section += f"- Score: {ctx.get('score', 'N/A')}\n"
                oracle_section += f"- Passed: {ctx.get('passed', 'N/A')}\n"
                if ctx.get("issues"):
                    oracle_section += f"- Issues: {', '.join(ctx['issues'][:3])}\n"

        prompt = f"""Review this RAC file for: {review_focus}

File: {rac_file}

Content:
{rac_content[:3000]}{"..." if len(rac_content) > 3000 else ""}
{oracle_section}
If oracle validators show discrepancies, investigate WHY the encoding differs from consensus.

Output ONLY valid JSON:
{{
  "score": <float 1-10>,
  "passed": <boolean>,
  "issues": ["issue1", "issue2"],
  "reasoning": "<brief explanation>"
}}
"""

        try:
            output, returncode = run_claude_code(
                prompt,
                model=REVIEWER_CLI_MODEL,
                timeout=120,
                cwd=self.rac_us_path,
            )

            # Parse JSON from output
            json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in output")

            score = float(data.get("score", 5.0))
            passed = bool(data.get("passed", score >= 7.0))
            issues = data.get("issues", [])

            duration = int((time.time() - start) * 1000)

            return ValidationResult(
                validator_name=reviewer_type,
                passed=passed,
                score=score,
                issues=issues if isinstance(issues, list) else [str(issues)],
                duration_ms=duration,
                raw_output=output,
            )

        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name=reviewer_type,
                passed=False,
                score=None,
                issues=[f"Reviewer error: {e}"],
                duration_ms=duration,
                error=str(e),
            )

    def _find_pe_python(self) -> Optional[str]:
        """Find a Python interpreter with policyengine-us installed.

        Checks: 1) current interpreter, 2) known PE venv paths.
        Returns the path to a working Python, or None.
        """
        # Try current interpreter first
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from policyengine_us import Simulation; print('ok')",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and "ok" in result.stdout:
                return sys.executable
        except Exception:
            pass

        # Try known PE venv locations
        pe_venv_paths = [
            Path.home() / "policyengine-us" / ".venv" / "bin" / "python",
            Path.home()
            / "RulesFoundation"
            / "policyengine-us"
            / ".venv"
            / "bin"
            / "python",
            Path.home()
            / "PolicyEngine"
            / "policyengine-us"
            / ".venv"
            / "bin"
            / "python",
        ]
        for pe_python in pe_venv_paths:
            if pe_python.exists():
                try:
                    result = subprocess.run(
                        [
                            str(pe_python),
                            "-c",
                            "from policyengine_us import Simulation; print('ok')",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0 and "ok" in result.stdout:
                        return str(pe_python)
                except Exception:  # pragma: no cover
                    continue  # pragma: no cover

        # Try auto-installing into current venv as last resort
        try:
            print("  PolicyEngine not found, attempting install...")
            install_result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "policyengine-us"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if install_result.returncode == 0:
                return sys.executable
            else:
                print(f"  Install failed: {install_result.stderr[:200]}")
        except Exception as e:
            print(f"  Auto-install failed: {e}")

        return None

    def _run_pe_subprocess(self, script: str, pe_python: str) -> Optional[str]:
        """Run a Python script using the PE-capable interpreter.

        Returns stdout on success, None on failure.
        """
        try:
            result = subprocess.run(
                [pe_python, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def _run_policyengine(self, rac_file: Path) -> ValidationResult:
        """Validate against PolicyEngine oracle.

        Uses scenario-based comparison: builds standard PE households from
        RAC test case inputs and compares the PE-calculated output variable
        against the RAC test's expected value.

        For programs like SNAP where RAC tests use intermediate inputs
        (snap_net_income, thrifty_food_plan_cost), we run PE with equivalent
        raw household scenarios and compare at the output variable level.
        """
        start = time.time()
        issues = []

        # Read and parse RAC file
        try:
            rac_content = Path(rac_file).read_text()
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="policyengine",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RAC file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Extract per-variable tests from RAC v2 format
        tests = self._extract_tests_from_rac_v2(rac_content)

        if not tests:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=None,
                issues=["No test cases with expected values found"],
                duration_ms=duration,
            )

        # Find a PE-capable Python interpreter
        pe_python = self._find_pe_python()
        if not pe_python:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="policyengine",
                passed=False,
                score=None,
                issues=[
                    "No PolicyEngine-capable Python found (tried local, known venvs, auto-install)"
                ],
                duration_ms=duration,
                error="policyengine-us not available",
            )

        # Map RAC variables to PE variables
        pe_var_map = self._get_pe_variable_map()

        # Run comparison for each test
        matches = 0
        total = 0
        for test in tests:
            rac_var = test.get("variable", "")
            pe_var = pe_var_map.get(rac_var)
            expected = test.get("expect")
            inputs = test.get("inputs", {})
            period = test.get("period", "2024-01")
            year = period.split("-")[0] if "-" in str(period) else str(period)

            if expected is None:
                continue

            if not pe_var:
                issues.append(f"No PE mapping for RAC variable '{rac_var}'")
                total += 1
                continue

            # Build and run PE scenario — include period in inputs for monthly detection
            inputs_with_period = {**inputs, "period": str(period)}
            scenario_script = self._build_pe_scenario_script(
                pe_var, inputs_with_period, year, expected
            )
            output = self._run_pe_subprocess(scenario_script, pe_python)

            if output is None:
                issues.append(
                    f"PE calculation failed for '{test.get('name', rac_var)}'"
                )
                total += 1
                continue

            # Parse result
            try:
                lines = output.strip().split("\n")
                result_line = [line for line in lines if line.startswith("RESULT:")]
                if result_line:
                    parts = result_line[0].split(":")
                    pe_value = float(parts[1])
                    expected_float = float(expected)
                    match = self._values_match(pe_value, expected_float, tolerance=0.02)
                    if match:
                        matches += 1
                    else:
                        issues.append(
                            f"'{test.get('name', rac_var)}': PE={pe_value:.2f}, RAC expects={expected_float:.2f}"
                        )
                    total += 1
                else:
                    issues.append(
                        f"No RESULT in PE output for '{test.get('name', rac_var)}'"
                    )
                    total += 1
            except Exception as parse_err:
                issues.append(
                    f"Parse error for '{test.get('name', rac_var)}': {parse_err}"
                )
                total += 1

        score = matches / total if total > 0 else None
        passed = score is not None and score >= 0.8

        duration = int((time.time() - start) * 1000)
        return ValidationResult(
            validator_name="policyengine",
            passed=passed,
            score=score,
            issues=issues,
            duration_ms=duration,
        )

    def _run_taxsim(self, rac_file: Path) -> ValidationResult:
        """Validate against TAXSIM oracle.

        Converts test cases to TAXSIM format, runs through TAXSIM API,
        and compares relevant outputs. Returns match rate as score (0-1).
        """
        start = time.time()
        issues = []

        # Read RAC file
        try:
            rac_content = Path(rac_file).read_text()
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RAC file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Extract tests
        tests = self._extract_tests_from_rac(rac_content)

        if not tests:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=True,
                score=1.0,
                issues=["No test cases found to validate"],
                duration_ms=duration,
            )

        # Try to run through TAXSIM
        try:
            import requests

            # TAXSIM API endpoint
            taxsim_url = "https://taxsim.nber.org/taxsim35/taxsim.cgi"

            matches = 0
            total = 0

            for test in tests:
                try:
                    # Convert test to TAXSIM input format
                    taxsim_input = self._build_taxsim_input(test.get("inputs", {}))

                    if not taxsim_input:
                        # Skip tests that can't be converted to TAXSIM format
                        continue

                    # Submit to TAXSIM
                    response = requests.post(
                        taxsim_url,
                        data=taxsim_input,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        # Parse TAXSIM output and compare
                        taxsim_result = self._parse_taxsim_output(response.text)
                        expected = test.get("expect")

                        if expected is not None and self._values_match(
                            taxsim_result, expected
                        ):
                            matches += 1

                    total += 1

                except requests.RequestException as req_error:
                    issues.append(f"TAXSIM request failed: {req_error}")
                    total += 1
                except Exception as test_error:
                    issues.append(
                        f"Test '{test.get('name', 'unknown')}' failed: {test_error}"
                    )
                    total += 1

            score = matches / total if total > 0 else 0.0
            passed = score >= 0.8

            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=passed,
                score=score,
                issues=issues,
                duration_ms=duration,
            )

        except ImportError:
            # requests not installed
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=False,
                score=None,
                issues=["requests package not installed for TAXSIM API"],
                duration_ms=duration,
                error="requests not available",
            )
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=False,
                score=None,
                issues=[f"TAXSIM validation error: {e}"],
                duration_ms=duration,
                error=str(e),
            )

    def _extract_tests_from_rac(self, rac_content: str) -> list[dict]:
        """Extract test cases from RAC file content.

        Returns list of test dictionaries with name, inputs, expect keys.
        """
        tests = []

        # Try to parse as YAML-like structure
        # RAC tests are typically in the format:
        # tests:
        #   - name: "test name"
        #     period: 2024-01
        #     inputs:
        #       var: value
        #     expect: expected_value

        try:
            # Find tests section in RAC content
            tests_match = re.search(
                r"tests:\s*\n((?:\s+-.*\n?)+)", rac_content, re.MULTILINE
            )

            if tests_match:
                tests_yaml = tests_match.group(1)
                # Parse the YAML tests section
                parsed = yaml.safe_load(f"tests:\n{tests_yaml}")
                if parsed and "tests" in parsed:
                    tests = parsed["tests"]
        except Exception:
            # If YAML parsing fails, try to extract simple test patterns
            test_blocks = re.findall(
                r'-\s*name:\s*["\']([^"\']+)["\'].*?expect:\s*(\S+)',
                rac_content,
                re.DOTALL,
            )
            for name, expect in test_blocks:
                tests.append({"name": name, "expect": expect, "inputs": {}})

        return tests

    def _build_pe_situation(self, inputs: dict) -> dict:
        """Build PolicyEngine situation dictionary from test inputs."""
        # Default situation structure
        situation = {
            "people": {"person": {}},
            "tax_units": {"tax_unit": {"members": ["person"]}},
            "households": {"household": {"members": ["person"]}},
        }

        # Map inputs to PE variables
        for key, value in inputs.items():
            # Simple mapping - real impl would be more sophisticated
            if "person." in key:
                var_name = key.replace("person.", "")
                situation["people"]["person"][var_name] = value
            elif "tax_unit." in key:
                var_name = key.replace("tax_unit.", "")
                situation["tax_units"]["tax_unit"][var_name] = value
            else:
                # Default to person-level
                situation["people"]["person"][key] = value

        return situation

    def _build_taxsim_input(self, inputs: dict) -> Optional[str]:
        """Build TAXSIM input string from test inputs.

        Returns None if inputs cannot be mapped to TAXSIM format.
        """
        # TAXSIM input mapping
        # See: https://taxsim.nber.org/taxsim35/

        # Build input line
        values = ["0"] * 27  # TAXSIM expects 27 fields

        # Set defaults
        values[0] = "1"  # taxsimid
        values[1] = "2024"  # year
        values[2] = "0"  # state
        values[3] = "1"  # marital status (single)

        # Map inputs
        mapped = False
        for key, value in inputs.items():
            key_lower = key.lower()
            if "wage" in key_lower:
                values[7] = str(value)
                mapped = True
            elif "self_employment" in key_lower or "semp" in key_lower:
                values[9] = str(value)
                mapped = True
            elif "year" in key_lower:
                values[1] = str(value)
                mapped = True

        if not mapped:
            return None

        return ",".join(values)

    def _parse_taxsim_output(self, output: str) -> Optional[float]:
        """Parse TAXSIM output and extract federal tax liability."""
        try:
            # TAXSIM returns comma-separated values
            # Field 7 is typically federal tax liability
            lines = output.strip().split("\n")
            if len(lines) >= 2:
                # Skip header line
                data_line = lines[-1]
                values = data_line.split(",")
                if len(values) > 7:
                    return float(values[7])
        except Exception:
            pass
        return None

    def _values_match(
        self, actual: Any, expected: Any, tolerance: float = 0.01
    ) -> bool:
        """Check if two values match within tolerance."""
        try:
            actual_float = float(actual) if actual is not None else 0.0
            expected_float = float(expected) if expected is not None else 0.0

            if expected_float == 0:
                return actual_float == 0

            relative_diff = abs(actual_float - expected_float) / abs(expected_float)
            return relative_diff <= tolerance
        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(actual) == str(expected)

    def _extract_tests_from_rac_v2(self, rac_content: str) -> list[dict]:
        """Extract test cases from per-definition test blocks.

        Supports both unified `name:` and legacy `variable name:` syntax.
        Returns list of dicts with keys: variable, name, period, inputs, expect.
        """
        tests = []

        # Match both unified `name:` and legacy `variable name:` definition blocks
        var_pattern = re.compile(
            r"^(?:variable\s+)?(\w+):\s*\n(.*?)(?=^(?:variable\s+)?\w+:|\Z)",
            re.MULTILINE | re.DOTALL,
        )

        for var_match in var_pattern.finditer(rac_content):
            var_name = var_match.group(1)
            var_block = var_match.group(2)

            # Find tests section within this variable block
            tests_pattern = re.compile(
                r"^\s+tests:\s*\n((?:\s+-.*\n?|\s+\w.*\n?)*)",
                re.MULTILINE,
            )
            tests_match = tests_pattern.search(var_block)
            if not tests_match:
                continue

            tests_yaml_str = tests_match.group(1)
            try:
                parsed = yaml.safe_load(f"items:\n{tests_yaml_str}")
                if parsed and "items" in parsed and isinstance(parsed["items"], list):
                    for test_case in parsed["items"]:
                        if isinstance(test_case, dict) and "expect" in test_case:
                            test_case["variable"] = var_name
                            tests.append(test_case)
            except Exception:
                # Try individual test extraction as fallback
                pass

        # If no v2-style tests found, fall back to legacy extraction
        if not tests:
            tests = self._extract_tests_from_rac(rac_content)

        return tests

    def _get_pe_variable_map(self) -> dict[str, str]:
        """Map RAC variable names to PolicyEngine variable names.

        Returns dict of rac_var_name -> pe_var_name.
        """
        return {
            # EITC
            "eitc": "eitc",
            "earned_income_credit": "eitc",
            "eitc_amount": "eitc",
            # CTC
            "ctc": "ctc",
            "child_tax_credit": "ctc",
            # Income tax
            "income_tax": "income_tax",
            "federal_income_tax": "income_tax",
            # Standard deduction
            "standard_deduction": "standard_deduction",
            "basic_standard_deduction": "basic_standard_deduction",
            # AGI
            "agi": "adjusted_gross_income",
            "adjusted_gross_income": "adjusted_gross_income",
            # SNAP
            "snap_allotment": "snap_normal_allotment",
            "snap_benefits": "snap",
            "snap": "snap",
            "snap_maximum_allotment": "snap_max_allotment",
            "minimum_allotment": "snap_min_allotment",
            "snap_net_income_calculation": "snap_net_income",
        }

    # PE variables that are defined as monthly (not annual)
    _PE_MONTHLY_VARS = {
        "snap",
        "snap_normal_allotment",
        "snap_max_allotment",
        "snap_net_income",
        "snap_expected_contribution",
        "snap_min_allotment",
        "snap_gross_income",
        "snap_emergency_allotment",
        "ssi",
        "ssi_amount_if_eligible",
        "tanf",
    }

    # PE variables at spm_unit level (need spm_units in situation)
    _PE_SPM_VARS = {
        "snap",
        "snap_normal_allotment",
        "snap_max_allotment",
        "snap_net_income",
        "snap_expected_contribution",
        "snap_min_allotment",
    }

    def _build_pe_scenario_script(
        self, pe_var: str, inputs: dict, year: str, expected: Any
    ) -> str:
        """Build a Python script to run a PE scenario via subprocess.

        Handles period detection (monthly vs annual PE variables),
        builds appropriate household structures, and overrides PE
        intermediate variables to match RAC test inputs for apples-to-apples
        comparison.
        """
        # Determine household composition from inputs
        household_size = inputs.get("household_size", 1)
        filing_status = inputs.get("filing_status", "SINGLE")

        # Determine period for calculation
        is_monthly = pe_var in self._PE_MONTHLY_VARS
        if is_monthly:
            period = inputs.get("period", f"{year}-01")
            if "-" not in str(period):
                period = f"{year}-01"
            calc_period = f"'{period}'"
        else:
            calc_period = f"int('{year}')"

        # Build people
        people_parts = [f"'adult': {{'age': {{'{year}': 30}}}}"]
        members = ["'adult'"]

        # Check for employment income / earned income
        earned = inputs.get(
            "employment_income", inputs.get("earned_income", inputs.get("wages", 0))
        )
        if earned:
            people_parts[0] = (
                f"'adult': {{'age': {{'{year}': 30}}, 'employment_income': {{'{year}': {earned}}}}}"
            )

        # Add spouse if joint
        if filing_status.upper() in ("JOINT", "MARRIED_FILING_JOINTLY"):
            people_parts.append(f"'spouse': {{'age': {{'{year}': 30}}}}")
            members.append("'spouse'")

        # Add children based on household_size (subtract adults)
        num_adults = (
            2 if filing_status.upper() in ("JOINT", "MARRIED_FILING_JOINTLY") else 1
        )
        num_children = max(0, int(household_size) - num_adults)
        for i in range(num_children):
            people_parts.append(
                f"'child{i}': {{'age': {{'{year}': 8}}, 'is_tax_unit_dependent': {{'{year}': True}}}}"
            )
            members.append(f"'child{i}'")

        members_str = "[" + ", ".join(members) + "]"
        people_str = "{" + ", ".join(people_parts) + "}"

        # Build SPM unit overrides for SNAP intermediate variables
        # This allows apples-to-apples comparison when RAC tests pass
        # pre-computed intermediate values (snap_net_income, etc.)
        snap_overridable = {
            "snap_net_income": "snap_net_income",
            "snap_gross_income": "snap_gross_income",
        }
        override_parts = []
        for rac_key, pe_key in snap_overridable.items():
            if rac_key in inputs:
                val = inputs[rac_key]
                if is_monthly:
                    override_parts.append(f"'{pe_key}': {{'{period}': {val}}}")
                else:
                    override_parts.append(f"'{pe_key}': {{'{year}': {val}}}")

        spm_extra = ""
        if override_parts:
            spm_extra = ", " + ", ".join(override_parts)

        script = f"""
from policyengine_us import Simulation

situation = {{
    'people': {people_str},
    'tax_units': {{'tu': {{'members': {members_str}}}}},
    'spm_units': {{'spm': {{'members': {members_str}{spm_extra}}}}},
    'households': {{'hh': {{'members': {members_str}, 'state_name': {{'{year}': 'CA'}}}}}},
    'families': {{'fam': {{'members': {members_str}}}}},
    'marital_units': {{'mu': {{'members': ['adult']}}}},
}}

sim = Simulation(situation=situation)
result = sim.calculate('{pe_var}', {calc_period})
val = float(result[0]) if hasattr(result, '__len__') and len(result) > 0 else float(result)
print(f'RESULT:{{val}}')
"""
        return script


def validate_file(rac_file: str | Path) -> PipelineResult:
    """Convenience function to validate a single file."""
    # Auto-detect paths based on file location
    file_path = Path(rac_file)

    # Find repo roots
    rac_us = file_path
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent

    rac = rac_us.parent / "rac"

    pipeline = ValidatorPipeline(
        rac_us_path=rac_us,
        rac_path=rac,
    )

    return pipeline.validate(file_path)
