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

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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
6. **Cross-Statute Definitions**: If the source text says a term is defined in another section, import that upstream definition instead of restating it locally
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
7. **Cross-Statute Imports**: References like "as defined in section 152(c)" are satisfied by imports from the cited section
"""
    + _REVIEW_JSON_FORMAT
)

GROUNDING_VALUE_PATTERN = re.compile(
    r"^\s*(?:from\s+\d{4}-\d{2}-\d{2}:\s*|value:\s*)"
    r"(-?[\d,]+(?:\.\d+)?)"
)
GROUNDING_SCALAR_PATTERN = re.compile(r"^(\w[\w_]*):\s*(-?[\d,]+(?:\.\d+)?)\s*$")
GROUNDING_ALLOWED_VALUES = {-1, 0, 1, 2, 3}
GROUNDING_INLINE_TEMPORAL_PATTERN = re.compile(
    r"^\s*from\s+\d{4}-\d{2}-\d{2}:\s*(.*?)\s*$"
)
GROUNDING_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
GROUNDING_FORMULA_NUMBER_PATTERN = re.compile(
    r"(?<![\w./])(-?[\d,]+(?:\.\d+)?)(?![\w./])"
)
GROUNDING_METADATA_KEYS = {
    "entity",
    "period",
    "dtype",
    "unit",
    "label",
    "description",
    "status",
    "indexed_by",
    "formula",
    "tests",
    "imports",
    "variable",
}

IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")
_PE_UNSUPPORTED_ERROR_PATTERNS = (
    re.compile(r"ParameterNotFoundError"),
    re.compile(r"VariableNotFoundError"),
    re.compile(r"was not found in the .*tax and benefit system", re.IGNORECASE),
)
_DEFINITION_CROSS_REFERENCE_PATTERN = re.compile(
    r"(?:as defined in|defined in|meaning given in|within the meaning of|described in)\s+"
    r"section\s+([0-9A-Za-z.-]+(?:\([^)]+\))*)",
    re.IGNORECASE,
)


def extract_grounding_values(content: str) -> list[tuple[int, str, float]]:
    """Extract grounded numeric values from RAC definitions, excluding formulas/tests."""
    values = []
    in_formula = False
    in_tests = False
    in_docstring = False
    formula_indent = 0

    for line_number, line in enumerate(content.split("\n"), 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if '"""' in stripped:
            in_docstring = not in_docstring
            continue
        if in_docstring or stripped.startswith("#"):
            continue

        if in_formula and stripped and indent <= formula_indent:
            in_formula = False

        if re.match(r"\s*formula:\s*\|", line):
            in_formula = True
            formula_indent = indent
            continue
        if re.match(r"\s*tests:", line):
            in_tests = True
            in_formula = False
            continue
        if in_tests and stripped and not line.startswith(" "):
            in_tests = False
        if in_tests:
            continue

        if in_formula:
            values.extend(_extract_formula_grounding_values(line_number, line))
            continue

        if re.match(r'\s*description:\s*"', line) or re.match(
            r"\s*description:\s*'", line
        ):
            continue
        if re.match(r'\s*label:\s*"', line) or re.match(r"\s*label:\s*'", line):
            continue

        match = GROUNDING_VALUE_PATTERN.match(line)
        if match:
            raw = match.group(1).replace(",", "")
            with contextlib.suppress(ValueError):
                value = float(raw)
                if value not in GROUNDING_ALLOWED_VALUES:
                    values.append((line_number, raw, value))
            continue

        inline_temporal_match = GROUNDING_INLINE_TEMPORAL_PATTERN.match(line)
        if inline_temporal_match:
            tail = inline_temporal_match.group(1).strip()
            if tail:
                values.extend(_extract_formula_grounding_values(line_number, tail))
            else:
                in_formula = True
                formula_indent = indent
            continue

        match = GROUNDING_SCALAR_PATTERN.match(stripped)
        if match:
            key = match.group(1)
            if key.lower() in GROUNDING_METADATA_KEYS:
                continue
            raw = match.group(2).replace(",", "")
            with contextlib.suppress(ValueError):
                value = float(raw)
                if value not in GROUNDING_ALLOWED_VALUES:
                    values.append((line_number, raw, value))

    return values


def _extract_formula_grounding_values(
    line_number: int, formula_text: str
) -> list[tuple[int, str, float]]:
    """Extract numeric literals from a formula expression or formula line."""
    cleaned = formula_text.split("#", 1)[0]
    cleaned = GROUNDING_DATE_PATTERN.sub(" ", cleaned)

    values: list[tuple[int, str, float]] = []
    for match in GROUNDING_FORMULA_NUMBER_PATTERN.finditer(cleaned):
        raw = match.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            value = float(raw)
            if value not in GROUNDING_ALLOWED_VALUES:
                values.append((line_number, raw, value))
    return values


def extract_numbers_from_text(text: str) -> set[float]:
    """Extract numeric values from embedded statute text."""
    numbers = set()

    for match in re.finditer(r"(?:^|(?<=[\s$(\[,]))(-?[\d,]+(?:\.\d+)?)\b", text):
        raw = match.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            numbers.add(float(raw))

    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s+(?:percent|per\s*centum)", text, re.IGNORECASE
    ):
        with contextlib.suppress(ValueError):
            numbers.add(float(match.group(1)) / 100)

    fraction_words = {
        "one-half": 0.5,
        "one half": 0.5,
        "one-third": 1 / 3,
        "one third": 1 / 3,
        "two-thirds": 2 / 3,
        "two thirds": 2 / 3,
        "one-quarter": 0.25,
        "one quarter": 0.25,
        "three-quarters": 0.75,
        "three quarters": 0.75,
    }
    text_lower = text.lower()
    for phrase, value in fraction_words.items():
        if phrase in text_lower:
            numbers.add(value)

    return numbers


def extract_embedded_source_text(content: str) -> str:
    """Extract the leading source-text docstring from a RAC file."""
    status_index = content.find("\nstatus:")
    header = content[:status_index] if status_index != -1 else content
    blocks = re.findall(r'"""(.*?)"""', header, re.DOTALL)
    if blocks:
        return "\n".join(block.strip() for block in blocks if block.strip())

    fallback_blocks = re.findall(r'"""(.*?)"""', content, re.DOTALL)
    return "\n".join(block.strip() for block in fallback_blocks if block.strip())


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
class OracleSubprocessResult:
    """Structured result from a local oracle subprocess."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


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

    def _pythonpath_env(self) -> dict[str, str]:
        """Build an env that prefers the configured local rac checkout."""
        env = dict(os.environ)
        rac_src = self.rac_path / "src"
        if rac_src.exists():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{rac_src}{os.pathsep}{existing}" if existing else str(rac_src)
            )
        return env

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
        """Tier 0: Compile check against the current public rac engine APIs."""
        start = time.time()
        issues = []
        rac_src = self.rac_path / "src"
        inserted_path = False

        try:
            from datetime import date

            if rac_src.exists() and str(rac_src) not in sys.path:
                sys.path.insert(0, str(rac_src))
                inserted_path = True

            from rac import compile as rac_compile
            from rac import parse_file

            module = parse_file(rac_file)
            ir = rac_compile([module], as_of=date.today())

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
        finally:
            if inserted_path:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(rac_src))

    def _run_ci(self, rac_file: Path) -> ValidationResult:
        """Run CI checks with the current rac CLI entry points."""
        start = time.time()
        issues = []
        env = self._pythonpath_env()

        # 1. Run companion .rac.test cases through the current test-runner CLI.
        try:
            result = subprocess.run(
                [sys.executable, "-m", "rac.test_runner", str(rac_file)],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if result.returncode != 0:
                summary = next(
                    (
                        line.strip()
                        for line in result.stdout.splitlines()
                        if line.strip().startswith("Tests:")
                    ),
                    "",
                )
                error_text = summary or result.stderr.strip() or result.stdout.strip()
                issues.append(f"Test runner failed: {error_text}")
        except subprocess.TimeoutExpired:
            issues.append("Test timeout")
        except Exception as e:
            issues.append(f"Test exception: {e}")

        # 2. Run the current structural validator on a temp directory containing
        # this RAC file plus its import closure so repo-augmented evals validate
        # against the same allowed dependency set as the original workspace.
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self._copy_validation_import_closure(
                    rac_file=rac_file,
                    destination_root=Path(tmpdir),
                )

                result = subprocess.run(
                    [sys.executable, "-m", "rac.validate", "all", tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env=env,
                    cwd=str(self.rac_path),
                )

                if result.returncode != 0:
                    detail_lines = [
                        line.strip()
                        for line in result.stdout.splitlines()
                        if line.strip()
                        and not line.startswith("Checked ")
                        and not line.startswith("Found ")
                    ]
                    if result.stderr.strip():
                        detail_lines.append(result.stderr.strip())
                    if detail_lines:
                        issues.extend(
                            f"Validation failed: {line}" for line in detail_lines[:10]
                        )
                    else:
                        issues.append("Validation failed")
        except subprocess.TimeoutExpired:
            issues.append("Validation timeout")
        except Exception as e:
            issues.append(f"Validation exception: {e}")

        try:
            issues.extend(self._check_cross_statute_definition_imports(rac_file))
        except Exception as e:
            issues.append(f"Cross-reference import check exception: {e}")

        duration = int((time.time() - start) * 1000)

        return ValidationResult(
            validator_name="ci",
            passed=len(issues) == 0,
            issues=issues,
            duration_ms=duration,
            error=issues[0] if issues else None,
        )

    def _copy_validation_import_closure(
        self,
        rac_file: Path,
        destination_root: Path,
    ) -> None:
        """Copy a RAC file and its imported dependencies into a temp validation tree."""
        source_root = self._validation_source_root(rac_file)
        pending = [rac_file.resolve()]
        copied: set[Path] = set()

        while pending:
            current = pending.pop()
            resolved = current.resolve()
            if resolved in copied:
                continue
            copied.add(resolved)

            relative = current.relative_to(source_root)
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, target)

            for dependency in self._resolve_import_dependencies(current, source_root):
                if dependency.resolve() not in copied:
                    pending.append(dependency)

    def _validation_source_root(self, rac_file: Path) -> Path:
        """Resolve the root directory used for import lookup during CI validation."""
        resolved_file = rac_file.resolve()
        resolved_root = self.rac_us_path.resolve()
        with contextlib.suppress(ValueError):
            resolved_file.relative_to(resolved_root)
            return resolved_root
        return resolved_file.parent

    def _resolve_import_dependencies(
        self,
        rac_file: Path,
        source_root: Path,
    ) -> list[Path]:
        """Resolve imported RAC files for a single file."""
        dependencies: list[Path] = []
        for import_path in self._extract_import_paths(rac_file.read_text()):
            target = source_root / self._import_to_relative_rac_path(import_path)
            if target.exists():
                dependencies.append(target)
        return dependencies

    def _extract_import_paths(self, content: str) -> list[str]:
        """Extract import file references from an imports block."""
        paths: list[str] = []
        in_imports = False
        imports_indent = 0

        for line in content.splitlines():
            imports_match = re.match(r"^(\s*)imports:\s*$", line)
            if imports_match:
                in_imports = True
                imports_indent = len(imports_match.group(1))
                continue

            if not in_imports:
                continue

            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())
            if indent <= imports_indent:
                in_imports = False
                continue

            item_match = IMPORT_ITEM_PATTERN.match(line)
            if not item_match:
                continue

            item = item_match.group(2).strip()
            import_target = item.split("#", 1)[0].strip()
            if import_target:
                paths.append(import_target)

        return paths

    def _import_to_relative_rac_path(self, import_target: str) -> Path:
        """Convert an import target like 26/24/c#name into 26/24/c.rac."""
        normalized = import_target.strip().strip('"').strip("'")
        if normalized.endswith(".rac"):
            return Path(normalized)
        return Path(f"{normalized}.rac")

    def _check_cross_statute_definition_imports(self, rac_file: Path) -> list[str]:
        """Flag missing imports for explicit cross-statute definition references."""
        if rac_file.stem == rac_file.parent.name:
            return []

        content = rac_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        title = self._infer_title_from_rac_path(rac_file)
        if not title:
            return []

        imports = self._extract_import_paths(content)
        issues: list[str] = []
        for citation, import_path in self._extract_definition_cross_references(
            source_text, title
        ):
            if any(
                existing == import_path or existing.startswith(import_path + "/")
                for existing in imports
            ):
                continue
            issues.append(
                "Cross-statute definition import missing: "
                f"source text references section {citation} but file does not import "
                f"from {import_path}"
            )
        return issues

    def _extract_definition_cross_references(
        self, source_text: str, title: str
    ) -> list[tuple[str, str]]:
        """Extract cited sections that the source text explicitly uses as definitions."""
        refs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for match in _DEFINITION_CROSS_REFERENCE_PATTERN.finditer(source_text):
            citation = match.group(1)
            import_path = self._section_reference_to_import_path(title, citation)
            if not import_path or import_path in seen:
                continue
            seen.add(import_path)
            refs.append((citation, import_path))
        return refs

    def _section_reference_to_import_path(
        self, title: str, section_reference: str
    ) -> str | None:
        """Convert `152(c)(1)(A)` into `26/152/c/1/A`."""
        match = re.match(
            r"^(?P<section>[0-9A-Za-z.-]+)(?P<tail>(?:\([^)]+\))*)$",
            section_reference.strip(),
        )
        if not match:
            return None
        fragments = re.findall(r"\(([^)]+)\)", match.group("tail"))
        return "/".join([title, match.group("section"), *fragments])

    def _infer_title_from_rac_path(self, rac_file: Path) -> str | None:
        """Infer the USC title from the RAC file path."""
        resolved_root = self.rac_us_path.resolve()
        resolved_file = rac_file.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_file.relative_to(resolved_root)
            if relative.parts:
                return relative.parts[0]

        parts = list(resolved_file.parts)
        with contextlib.suppress(ValueError):
            statute_idx = parts.index("statute")
            if statute_idx + 1 < len(parts):
                return parts[statute_idx + 1]

        numericish_parts = [
            part
            for part in resolved_file.parts
            if re.fullmatch(r"[0-9A-Za-z.-]+", part) and any(ch.isdigit() for ch in part)
        ]
        return numericish_parts[0] if numericish_parts else None

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
        result = self._run_pe_subprocess_detailed(script, pe_python)
        if result.returncode == 0:
            return result.stdout
        return None

    def _run_pe_subprocess_detailed(
        self, script: str, pe_python: str
    ) -> OracleSubprocessResult:
        """Run a Python script using the PE-capable interpreter with stderr."""
        try:
            result = subprocess.run(
                [pe_python, "-c", script],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return OracleSubprocessResult(
                returncode=result.returncode,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        except subprocess.TimeoutExpired as exc:
            return OracleSubprocessResult(
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "").strip() or "Timeout after 120s",
            )
        except Exception as exc:
            return OracleSubprocessResult(returncode=1, stderr=str(exc))

    def _is_pe_unsupported_error(self, error_text: str) -> bool:
        """Return True when PE cannot evaluate the cited period or variable."""
        if not error_text:
            return False
        return any(pattern.search(error_text) for pattern in _PE_UNSUPPORTED_ERROR_PATTERNS)

    def _summarize_oracle_error(self, error_text: str) -> str:
        """Collapse multi-line stderr into a short human-readable issue."""
        if not error_text:
            return "unknown error"
        for line in reversed(error_text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return "unknown error"

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

        # Read test content (from .rac.test companion or inline)
        try:
            rac_content = self._read_test_content(rac_file)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="policyengine",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RAC/test file: {e}"],
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
        unsupported_count = 0
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
            output = self._run_pe_subprocess_detailed(scenario_script, pe_python)

            if output.returncode != 0:
                summary = self._summarize_oracle_error(output.stderr or output.stdout)
                if self._is_pe_unsupported_error(output.stderr or output.stdout):
                    issues.append(
                        f"PolicyEngine unavailable for '{test.get('name', rac_var)}': {summary}"
                    )
                    unsupported_count += 1
                    continue
                issues.append(
                    f"PE calculation failed for '{test.get('name', rac_var)}': {summary}"
                )
                total += 1
                continue

            # Parse result
            try:
                lines = output.stdout.strip().split("\n")
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

        if total == 0:
            duration = int((time.time() - start) * 1000)
            if unsupported_count:
                issues.append("PolicyEngine could not evaluate any oracle-comparable tests")
            return ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=None,
                issues=issues or ["No PolicyEngine-comparable tests found"],
                duration_ms=duration,
            )

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

    def _read_test_content(self, rac_file: Path) -> str:
        """Read test content from .rac.test companion file or inline tests.

        Checks for companion .rac.test file first, falls back to inline.
        """
        # Check for companion .rac.test file
        test_file = Path(str(rac_file) + ".test")
        if test_file.exists():
            return test_file.read_text()
        # Fall back to inline tests in the .rac file itself
        return rac_file.read_text()

    def _run_taxsim(self, rac_file: Path) -> ValidationResult:
        """Validate against TAXSIM oracle.

        Converts test cases to TAXSIM format, runs through TAXSIM API,
        and compares relevant outputs. Returns match rate as score (0-1).
        """
        start = time.time()
        issues = []

        # Read test content (from .rac.test companion or inline)
        try:
            rac_content = self._read_test_content(rac_file)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RAC/test file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Extract tests (v2 parser handles .rac.test format)
        tests = self._extract_tests_from_rac_v2(rac_content)

        if not tests:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=True,
                score=None,
                issues=["No test cases found — cannot validate"],
                duration_ms=duration,
            )

        # Try to run through TAXSIM
        try:
            import requests

            # TAXSIM API endpoint
            taxsim_url = "https://taxsim.nber.org/taxsim35/taxsim.cgi"

            matches = 0
            total = 0
            unmappable = 0

            for test in tests:
                try:
                    # Convert test to TAXSIM input format
                    taxsim_input = self._build_taxsim_input(test.get("inputs", {}))

                    if not taxsim_input:
                        issues.append(
                            f"TAXSIM could not map inputs for '{test.get('name', 'unknown')}'"
                        )
                        unmappable += 1
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

            if total == 0:
                duration = int((time.time() - start) * 1000)
                if unmappable:
                    issues.append("TAXSIM could not evaluate any oracle-comparable tests")
                return ValidationResult(
                    validator_name="taxsim",
                    passed=True,
                    score=None,
                    issues=issues or ["No TAXSIM-comparable tests found"],
                    duration_ms=duration,
                )

            score = matches / total
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

    def _run_microdata_benchmark(
        self,
        output_path: Path,
        pe_variable: str = "eitc",
        year: int = 2024,
        sample_size: int | None = None,
    ) -> ValidationResult:
        """Benchmark RAC encoding against PolicyEngine using CPS microdata.

        Runs PE Microsimulation on the enhanced CPS, extracts the target
        variable for all tax units, and reports the benchmark. Since RAC
        doesn't have a runtime yet, this establishes the PE baseline that
        RAC must match as inputs get wired up.

        Args:
            output_path: Directory containing .rac files for the section.
            pe_variable: PE variable to benchmark against (e.g., "eitc").
            year: Tax year for the simulation.
            sample_size: If set, only use this many tax units (for speed).

        Returns:
            ValidationResult with benchmark statistics.
        """
        start = time.time()
        issues = []

        # Find PE-capable Python
        pe_python = self._find_pe_python()
        if not pe_python:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="microdata_benchmark",
                passed=False,
                score=0.0,
                issues=["No PolicyEngine-capable Python found"],
                duration_ms=duration,
                error="policyengine-us not available",
            )

        # Run PE microsimulation and collect statistics
        script = f"""
import json
import numpy as np
from policyengine_us import Microsimulation

m = Microsimulation()
values = m.calculate('{pe_variable}', {year})
weights = m.calculate('tax_unit_weight', {year})

# Core stats
total = len(values)
nonzero = int(np.sum(np.array(values) > 0))
weights_arr = np.array(weights)
values_arr = np.array(values)
weighted_nonzero = float(np.sum(weights_arr * (values_arr > 0)))
weighted_total = float(np.sum(weights_arr))
weighted_sum = float(np.sum(weights_arr * values_arr))
mean_val = float(values.mean())
median_val = float(np.median(values))
max_val = float(values.max())
p25 = float(np.percentile(values[values > 0], 25)) if nonzero > 0 else 0
p75 = float(np.percentile(values[values > 0], 75)) if nonzero > 0 else 0

result = {{
    "variable": "{pe_variable}",
    "year": {year},
    "total_tax_units": total,
    "nonzero_count": nonzero,
    "weighted_nonzero": weighted_nonzero,
    "weighted_total": weighted_total,
    "weighted_sum_billions": weighted_sum / 1e9,
    "mean": mean_val,
    "median": median_val,
    "max": max_val,
    "p25_nonzero": p25,
    "p75_nonzero": p75,
}}
print("BENCHMARK:" + json.dumps(result))
"""

        output = self._run_pe_subprocess(script, pe_python)

        if output is None:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="microdata_benchmark",
                passed=False,
                score=0.0,
                issues=["PE microsimulation failed to run"],
                duration_ms=duration,
                error="Microsimulation execution failed",
            )

        # Parse benchmark results
        try:
            benchmark_line = [
                line
                for line in output.strip().split("\n")
                if line.startswith("BENCHMARK:")
            ]
            if not benchmark_line:
                duration = int((time.time() - start) * 1000)
                return ValidationResult(
                    validator_name="microdata_benchmark",
                    passed=False,
                    score=0.0,
                    issues=[f"No BENCHMARK output from PE. Output: {output[:200]}"],
                    duration_ms=duration,
                )

            stats = json.loads(benchmark_line[0].split("BENCHMARK:")[1])

            # RAC match rate starts at 0% — no runtime yet
            rac_match_rate = 0.0

            issues = [
                f"PE benchmark for {pe_variable} ({year}):",
                f"  Tax units: {stats['total_tax_units']:,} "
                f"({stats['nonzero_count']:,} with {pe_variable} > 0)",
                f"  Weighted recipients: {stats['weighted_nonzero']:,.0f}",
                f"  Weighted total: ${stats['weighted_sum_billions']:.1f}B",
                f"  Mean: ${stats['mean']:,.0f}, Median: ${stats['median']:,.0f}, "
                f"Max: ${stats['max']:,.0f}",
                f"  P25-P75 (nonzero): ${stats['p25_nonzero']:,.0f}-${stats['p75_nonzero']:,.0f}",
                f"  RAC match rate: {rac_match_rate:.1%} "
                f"(no RAC runtime — benchmark only)",
            ]

            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="microdata_benchmark",
                passed=False,  # 0% match until RAC runtime exists
                score=rac_match_rate,
                issues=issues,
                duration_ms=duration,
                raw_output=json.dumps(stats, indent=2),
            )

        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="microdata_benchmark",
                passed=False,
                score=0.0,
                issues=[f"Failed to parse benchmark output: {e}"],
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

        Supports three formats:
        1. .rac.test format: `variable_name:` with direct list of test cases
        2. Unified `name:` blocks with nested `tests:` section
        3. Legacy `tests:` top-level section

        Returns list of dicts with keys: variable, name, period, inputs, expect.
        """
        tests = []

        # Try full YAML parse first — handles .rac.test format cleanly
        try:
            # Strip comments and docstrings before parsing
            content_lines = []
            in_docstring = False
            for line in rac_content.split("\n"):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        in_docstring = False
                        continue
                    elif stripped.count('"""') == 1 or stripped.count("'''") == 1:
                        in_docstring = True
                        continue
                if in_docstring:
                    continue
                if stripped.startswith("#"):
                    continue
                content_lines.append(line)

            clean_content = "\n".join(content_lines)
            parsed = yaml.safe_load(clean_content)

            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    # Skip non-test keys (status, imports, entity, etc.)
                    if key in (
                        "status",
                        "imports",
                        "entity",
                        "period",
                        "dtype",
                        "unit",
                        "label",
                        "description",
                        "default",
                    ):
                        continue
                    # .rac.test format: variable_name: [list of test dicts]
                    if isinstance(value, list):
                        for test_case in value:
                            if isinstance(test_case, dict) and "expect" in test_case:
                                test_case["variable"] = key
                                tests.append(test_case)
                    # Nested tests: section within a variable block
                    elif isinstance(value, dict) and "tests" in value:
                        for test_case in value.get("tests", []):
                            if isinstance(test_case, dict) and "expect" in test_case:
                                test_case["variable"] = key
                                tests.append(test_case)
        except Exception:
            pass

        # Fallback: regex-based extraction for inline tests
        if not tests:
            var_pattern = re.compile(
                r"^(?:variable\s+)?(\w+):\s*\n(.*?)(?=^(?:variable\s+)?\w+:|\Z)",
                re.MULTILINE | re.DOTALL,
            )

            for var_match in var_pattern.finditer(rac_content):
                var_name = var_match.group(1)
                var_block = var_match.group(2)

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
                    if (
                        parsed
                        and "items" in parsed
                        and isinstance(parsed["items"], list)
                    ):
                        for test_case in parsed["items"]:
                            if isinstance(test_case, dict) and "expect" in test_case:
                                test_case["variable"] = var_name
                                tests.append(test_case)
                except Exception:
                    pass

        # Last resort: legacy extraction
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
        filing_status = inputs.get("filing_status", "SINGLE")
        joint_filing = filing_status.upper() in ("JOINT", "MARRIED_FILING_JOINTLY")
        num_adults = 2 if joint_filing else 1

        household_size = inputs.get("household_size")
        explicit_child_count = None
        for key, value in inputs.items():
            key_lower = str(key).lower()
            if key_lower in {
                "qualifying_children_allowed_section_151_deduction_count",
                "qualifying_children_with_section_151_deduction_count",
                "qualifying_child_count",
                "ctc_qualifying_children",
                "dependent_child_count",
                "child_count",
            } or (
                key_lower.endswith("_count")
                and "qualifying" in key_lower
                and "child" in key_lower
            ):
                with contextlib.suppress(TypeError, ValueError):
                    explicit_child_count = max(0, int(value))
                    break

        household_children = 0
        if household_size is not None:
            with contextlib.suppress(TypeError, ValueError):
                household_children = max(0, int(household_size) - num_adults)

        num_children = (
            explicit_child_count if explicit_child_count is not None else household_children
        )

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
        if joint_filing:
            people_parts.append(f"'spouse': {{'age': {{'{year}': 30}}}}")
            members.append("'spouse'")

        # Add children based on explicit qualifying-child counts or household size.
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
    'marital_units': {{'mu': {{'members': {['adult', 'spouse'] if joint_filing else ['adult']}}}}},
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
