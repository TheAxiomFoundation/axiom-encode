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
from collections import Counter
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
2. **No Embedded Scalars**: Legal scalar amounts, thresholds, and limits should be declared as named variables, not embedded inside formulas or conditional branches.
3. **Sourcing**: Policy values should reference authoritative sources
4. **Time-Varying Values**: Rate thresholds and amounts should use `from yyyy-mm-dd:` temporal entries
5. **Reference Format**: Correct reference syntax (unified `name:` format, no `parameter` keyword)
6. **Default Values**: Appropriate defaults for optional inputs
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
SOURCE_TEXT_NUMBER_PATTERN = re.compile(
    r"(?:^|(?<=[\s$£€(\[,]))(-?[\d,]+(?:\.\d+)?)\b"
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
_EMBEDDED_SCALAR_BLOCK_HEADER = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
_EMBEDDED_SCALAR_TEMPORAL_LINE = re.compile(
    r"^(\s*)from\s+\d{4}-\d{2}-\d{2}:\s*(.*?)\s*$"
)
_EMBEDDED_SCALAR_FORMULA_HEADER = re.compile(r"^(\s*)formula:\s*\|\s*$")
_EMBEDDED_SCALAR_DIRECT_VALUE = re.compile(r"-?[\d,]+(?:\.\d+)?")
_EMBEDDED_SCALAR_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_EMBEDDED_SCALAR_ALLOWED_VALUES = {"-1", "0", "1", "2", "3"}
_QUOTED_STRING_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
_STRUCTURAL_SOURCE_LINE_PATTERN = re.compile(
    r"^[\(\[]?(?:\d+[A-Za-z]?|[ivxlcdm]+|[a-z])[\)\].]?$", re.IGNORECASE
)
_STRUCTURAL_SOURCE_HEADING_PATTERN = re.compile(
    r"^(PART|CHAPTER|SCHEDULE|REGULATION|ARTICLE)\b", re.IGNORECASE
)
_STRUCTURAL_SOURCE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:\d+[A-Za-z]?\.\s+|\([0-9A-Za-zivxlcdm]+\)\s+)", re.IGNORECASE
)
_SOURCE_REFERENCE_TARGET_PATTERN = (
    r"(?:\([^)]+\)|\d+[A-Za-z./-]*(?:\([^)]+\))*(?=$|[\s,.;:])|[ivxlcdm]+\b|[A-Z]{1,4}\b|[a-z]\b)"
)
_SOURCE_REFERENCE_SEQUENCE_PATTERN = (
    rf"{_SOURCE_REFERENCE_TARGET_PATTERN}"
    rf"(?:\s*(?:,|or|and)\s*{_SOURCE_REFERENCE_TARGET_PATTERN})*"
)
_SOURCE_REFERENCE_PATTERNS = (
    re.compile(
        r"\b(?:section|sections|paragraph|paragraphs|regulation|regulations|part|parts|chapter|chapters|schedule|schedules|article|articles|subparagraph|subparagraphs|sub-paragraph|sub-paragraphs|subsection|subsections)\s+"
        rf"{_SOURCE_REFERENCE_SEQUENCE_PATTERN}(?:\s+to\s+{_SOURCE_REFERENCE_SEQUENCE_PATTERN})?",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:Act|Order|Regulations?)\s+\d{4}\b"),
)
_DIRECT_SCALAR_VALUE_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")


@dataclass(frozen=True)
class NamedScalarOccurrence:
    """One direct named scalar definition found in a RAC file."""

    line: int
    name: str
    value: float
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
_DEFINED_SYMBOL_METADATA_KEYS = {
    "imports",
    "status",
    "description",
    "label",
    "entity",
    "period",
    "dtype",
    "unit",
    "indexed_by",
    "tests",
    "default",
    "stub_for",
    "skip_reason",
}


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

    for match in re.finditer(
        r"(?:^|(?<=[\s$£€(\[,]))(-?[\d,]+(?:\.\d+)?)\b", text
    ):
        raw = match.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            numbers.add(float(raw))

    for match in re.finditer(
        r"(\d+(?:\.\d+)?)\s+(?:percent|per\s*cent(?:um)?)", text, re.IGNORECASE
    ):
        with contextlib.suppress(ValueError):
            numbers.add(float(match.group(1)) / 100)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
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


def extract_numeric_occurrences_from_text(text: str) -> list[float]:
    """Extract substantive numeric occurrences from source text, preserving repeats."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _STRUCTURAL_SOURCE_LINE_PATTERN.match(stripped):
            continue
        if _STRUCTURAL_SOURCE_HEADING_PATTERN.match(stripped):
            continue
        cleaned_lines.append(_STRUCTURAL_SOURCE_PREFIX_PATTERN.sub("", line))

    cleaned = "\n".join(cleaned_lines)
    cleaned = GROUNDING_DATE_PATTERN.sub(" ", cleaned)
    for pattern in _SOURCE_REFERENCE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    occurrences: list[float] = []
    spans: list[tuple[int, int]] = []

    for pattern in (
        re.compile(r"(\d+(?:\.\d+)?)\s+(?:percent|per\s*cent(?:um)?)", re.IGNORECASE),
        re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    ):
        for match in pattern.finditer(cleaned):
            with contextlib.suppress(ValueError):
                occurrences.append(float(match.group(1).replace(",", "")) / 100)
                spans.append(match.span())

    def overlaps_percent(span: tuple[int, int]) -> bool:
        return any(not (span[1] <= start or span[0] >= end) for start, end in spans)

    for match in SOURCE_TEXT_NUMBER_PATTERN.finditer(cleaned):
        span = match.span(1)
        if overlaps_percent(span):
            continue
        with contextlib.suppress(ValueError):
            value = float(match.group(1).replace(",", ""))
            if value.is_integer() and 1900 <= value <= 2100:
                continue
            occurrences.append(value)

    occurrence_counts = Counter(occurrences)
    normalized: list[float] = []
    for value in occurrences:
        scaled = round(value * 100, 9)
        if value <= 1 and scaled in occurrence_counts:
            continue
        normalized.append(value)
    return normalized


def extract_named_scalar_occurrences(content: str) -> list[NamedScalarOccurrence]:
    """Extract direct named scalar definitions from a RAC file, preserving repeats."""
    occurrences: list[NamedScalarOccurrence] = []
    current_variable: str | None = None
    temporal_block = False
    temporal_indent = 0

    for line_number, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        header_match = _EMBEDDED_SCALAR_BLOCK_HEADER.match(line)
        if header_match and indent == 0:
            current_variable = header_match.group(1)

        if temporal_block and stripped and indent <= temporal_indent:
            temporal_block = False

        scalar_match = GROUNDING_SCALAR_PATTERN.match(stripped)
        if scalar_match:
            name = scalar_match.group(1)
            if name.lower() not in GROUNDING_METADATA_KEYS:
                raw = scalar_match.group(2).replace(",", "")
                with contextlib.suppress(ValueError):
                    occurrences.append(
                        NamedScalarOccurrence(
                            line=line_number,
                            name=name,
                            value=float(raw),
                        )
                    )
            continue

        temporal_match = GROUNDING_INLINE_TEMPORAL_PATTERN.match(line)
        if temporal_match:
            tail = temporal_match.group(1).strip().replace(",", "")
            temporal_indent = indent
            if tail and _DIRECT_SCALAR_VALUE_PATTERN.fullmatch(tail):
                with contextlib.suppress(ValueError):
                    occurrences.append(
                        NamedScalarOccurrence(
                            line=line_number,
                            name=current_variable or "<unknown>",
                            value=float(tail),
                        )
                    )
                temporal_block = False
            else:
                temporal_block = True
            continue

        if temporal_block and stripped:
            normalized = stripped.replace(",", "")
            if _DIRECT_SCALAR_VALUE_PATTERN.fullmatch(normalized):
                with contextlib.suppress(ValueError):
                    occurrences.append(
                        NamedScalarOccurrence(
                            line=line_number,
                            name=current_variable or "<unknown>",
                            value=float(normalized),
                        )
                    )

    return occurrences


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
        policyengine_country: str = "auto",
        policyengine_rac_var_hint: str | None = None,
    ):
        self.rac_us_path = Path(rac_us_path)
        self.rac_path = Path(rac_path)
        self.enable_oracles = enable_oracles
        self.max_workers = max_workers
        self.encoding_db = encoding_db
        self.session_id = session_id
        self.policyengine_country = policyengine_country
        self.policyengine_rac_var_hint = policyengine_rac_var_hint

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

    def validate(self, rac_file: Path, skip_reviewers: bool = False) -> PipelineResult:
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
        if skip_reviewers:
            self._log_event(
                "validation_llm_skipped",
                "Skipping LLM reviewers",
                {
                    "oracle_context_summary": {
                        k: v.get("score") for k, v in oracle_context.items()
                    },
                },
            )
        else:
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
                futures = {
                    executor.submit(fn): name for name, fn in llm_validators.items()
                }

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

        with contextlib.suppress(Exception):
            issues.extend(self._check_embedded_scalar_literals(rac_file))

        advisories: list[str] = []
        with contextlib.suppress(Exception):
            advisories = self._build_import_advisories(rac_file)

        duration = int((time.time() - start) * 1000)

        return ValidationResult(
            validator_name="ci",
            passed=len(issues) == 0,
            issues=issues,
            duration_ms=duration,
            error=issues[0] if issues else None,
            raw_output="\n".join(advisories) if advisories else None,
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

    def _extract_defined_symbols(self, content: str) -> list[str]:
        """Extract top-level RAC definition names."""
        definitions: list[str] = []
        for line in content.splitlines():
            match = re.match(r"^([A-Za-z_]\w*):\s*$", line)
            if not match:
                continue
            name = match.group(1)
            if name in _DEFINED_SYMBOL_METADATA_KEYS:
                continue
            definitions.append(name)
        return definitions

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

    def _check_embedded_scalar_literals(self, rac_file: Path) -> list[str]:
        """Flag substantive scalar literals embedded inside formulas."""
        issues: list[str] = []
        for line_number, name, literal, expression in self._collect_embedded_scalar_literals(
            rac_file.read_text()
        ):
            issues.append(
                "Embedded scalar literal: "
                f"{name} line {line_number} embeds {literal} in `{expression}`; "
                "extract the scalar to its own named variable"
            )
        return issues

    def _collect_embedded_scalar_literals(
        self,
        content: str,
    ) -> list[tuple[int, str, str, str]]:
        """Return embedded substantive scalar literals found in RAC expressions."""
        issues: list[tuple[int, str, str, str]] = []
        current_name: str | None = None
        temporal_block = False
        temporal_indent = 0
        formula_block = False
        formula_indent = 0

        for line_number, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            header_match = _EMBEDDED_SCALAR_BLOCK_HEADER.match(line)
            if header_match and indent == 0:
                current_name = header_match.group(1)

            if temporal_block and stripped and indent <= temporal_indent:
                temporal_block = False
            if formula_block and stripped and indent <= formula_indent:
                formula_block = False

            temporal_match = _EMBEDDED_SCALAR_TEMPORAL_LINE.match(line)
            if temporal_match:
                temporal_indent = len(temporal_match.group(1))
                tail = temporal_match.group(2).strip()
                if tail:
                    if not self._is_direct_scalar_expression(tail):
                        issues.extend(
                            (
                                line_number,
                                current_name or "<unknown>",
                                literal,
                                tail,
                            )
                            for literal in self._extract_embedded_scalar_literals(tail)
                        )
                    temporal_block = False
                else:
                    temporal_block = True
                continue

            formula_match = _EMBEDDED_SCALAR_FORMULA_HEADER.match(line)
            if formula_match:
                formula_block = True
                formula_indent = len(formula_match.group(1))
                continue

            if (temporal_block or formula_block) and stripped and not self._is_direct_scalar_expression(
                stripped
            ):
                issues.extend(
                    (
                        line_number,
                        current_name or "<unknown>",
                        literal,
                        stripped,
                    )
                    for literal in self._extract_embedded_scalar_literals(stripped)
                )

        return issues

    def _is_direct_scalar_expression(self, expression: str) -> bool:
        normalized = expression.replace(",", "")
        return bool(_EMBEDDED_SCALAR_DIRECT_VALUE.fullmatch(normalized))

    def _extract_embedded_scalar_literals(self, expression: str) -> list[str]:
        literals: list[str] = []
        scrubbed_expression = _QUOTED_STRING_PATTERN.sub(" ", expression)
        for match in _EMBEDDED_SCALAR_NUMBER.finditer(scrubbed_expression):
            start, end = match.span()
            prev = scrubbed_expression[start - 1] if start > 0 else ""
            nxt = scrubbed_expression[end] if end < len(scrubbed_expression) else ""
            if (prev.isalnum() or prev in {"_", ".", "/"}) or (
                nxt.isalnum() or nxt in {"_", ".", "/"}
            ):
                continue
            literal = match.group(0)
            if literal in _EMBEDDED_SCALAR_ALLOWED_VALUES:
                continue
            literals.append(literal)
        return sorted(set(literals))

    def _build_import_advisories(self, rac_file: Path) -> list[str]:
        """Return non-blocking advice about likely shared concepts."""
        content = rac_file.read_text()
        definitions = self._extract_defined_symbols(content)
        if not definitions:
            return []

        source_root = self._validation_source_root(rac_file)
        search_root = self._candidate_concept_search_root(rac_file, source_root)
        if not search_root.exists():
            return []

        imports = set(self._extract_import_paths(content))
        advisories: list[str] = []
        seen: set[tuple[str, str]] = set()

        for candidate_file in search_root.rglob("*.rac"):
            if candidate_file.resolve() == rac_file.resolve():
                continue
            candidate_defs = set(
                self._extract_defined_symbols(candidate_file.read_text())
            )
            overlap = sorted(set(definitions) & candidate_defs)
            if not overlap:
                continue
            import_base = self._relative_import_base(candidate_file, source_root)
            if not import_base or import_base in imports:
                continue
            for name in overlap:
                key = (name, import_base)
                if key in seen:
                    continue
                seen.add(key)
                advisories.append(
                    "Shared concept advisory: "
                    f"`{name}` is also defined in `{import_base}#{name}`. "
                    "If the semantics match, prefer importing or re-exporting that "
                    "canonical concept instead of duplicating it locally."
                )
        return advisories

    def _candidate_concept_search_root(self, rac_file: Path, source_root: Path) -> Path:
        """Choose a nearby subtree for conservative shared-concept advisories."""
        with contextlib.suppress(ValueError):
            relative = rac_file.resolve().relative_to(source_root.resolve())
            if len(relative.parts) >= 2:
                return source_root / relative.parts[0] / relative.parts[1]
        return rac_file.parent

    def _relative_import_base(
        self, candidate_file: Path, source_root: Path
    ) -> str | None:
        """Convert a RAC file path to an import base without the symbol suffix."""
        with contextlib.suppress(ValueError):
            relative = candidate_file.resolve().relative_to(source_root.resolve())
            return str(relative.with_suffix("")).replace(os.sep, "/")
        return None

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
            if relative.parts and re.fullmatch(
                r"[0-9A-Za-z.-]+", relative.parts[0]
            ) and any(ch.isdigit() for ch in relative.parts[0]):
                return relative.parts[0]
            return None

        parts = list(resolved_file.parts)
        with contextlib.suppress(ValueError):
            statute_idx = parts.index("statute")
            if statute_idx + 1 < len(parts):
                return parts[statute_idx + 1]
        return None

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

    def _detect_policyengine_country(self, rac_file: Path, rac_content: str) -> str:
        """Infer which PolicyEngine country package to use."""
        if self.policyengine_country in {"us", "uk"}:
            return self.policyengine_country

        haystack = f"{rac_file}\n{rac_content}".lower()
        if "legislation.gov.uk" in haystack or re.search(
            r"\b(?:ukpga|uksi|asp|ssi|wsi|nisi|anaw|asc)(?:/|-)", haystack
        ):
            return "uk"
        return "us"

    def _find_pe_python(self, country: str = "us") -> Optional[str]:
        """Find a Python interpreter with the requested PolicyEngine package installed.

        Checks: 1) current interpreter, 2) known PE venv paths.
        Returns the path to a working Python, or None.
        """
        module_name = f"policyengine_{country}"
        package_name = f"policyengine-{country}"
        repo_name = f"policyengine-{country}"

        # Try current interpreter first
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"from {module_name} import Simulation; print('ok')",
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
            Path.home() / repo_name / ".venv" / "bin" / "python",
            Path.home()
            / "RulesFoundation"
            / repo_name
            / ".venv"
            / "bin"
            / "python",
            Path.home()
            / "PolicyEngine"
            / repo_name
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
                            f"from {module_name} import Simulation; print('ok')",
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
                [sys.executable, "-m", "pip", "install", package_name],
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

        try:
            rac_source_content = rac_file.read_text()
        except Exception:
            rac_source_content = ""

        country = self._detect_policyengine_country(rac_file, rac_source_content)

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
        pe_python = self._find_pe_python(country)
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
                error=f"policyengine-{country} not available",
            )

        # Map RAC variables to PE variables
        # Run comparison for each test
        matches = 0
        total = 0
        unsupported_count = 0
        for test in tests:
            test_rac_var = test.get("variable", "")
            oracle_rac_var = self.policyengine_rac_var_hint or test_rac_var
            pe_var = self._resolve_pe_variable(country, oracle_rac_var)
            expected = test.get("expect")
            raw_inputs = test.get("inputs", {})
            inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else raw_inputs
            period = (
                test.get("period")
                or test.get("date")
                or (
                    inputs.get("period")
                    if isinstance(inputs, dict)
                    else None
                )
                or (
                    inputs.get("date")
                    if isinstance(inputs, dict)
                    else None
                )
                or "2024-01"
            )
            period_str = str(period)
            year = period_str.split("-")[0] if "-" in period_str else period_str
            if isinstance(inputs, dict):
                inputs.pop("period", None)
                inputs.pop("date", None)

            if expected is None:
                continue

            mappable, reason = self._is_pe_test_mappable(
                country, oracle_rac_var, inputs, expected
            )
            if not mappable:
                issues.append(
                    f"PolicyEngine unavailable for '{test.get('name', test_rac_var)}': {reason}"
                )
                unsupported_count += 1
                continue

            if not pe_var:
                if self.policyengine_rac_var_hint:
                    issues.append(
                        "No PE mapping for RAC variable "
                        f"'{test_rac_var}' with oracle hint "
                        f"'{self.policyengine_rac_var_hint}'"
                    )
                else:
                    issues.append(f"No PE mapping for RAC variable '{test_rac_var}'")
                total += 1
                continue

            # Build and run PE scenario — include period in inputs for monthly detection
            inputs_with_period = {**inputs, "period": str(period)}
            scenario_script = self._build_pe_scenario_script(
                pe_var,
                inputs_with_period,
                year,
                expected,
                country=country,
                rac_var=oracle_rac_var,
            )
            output = self._run_pe_subprocess_detailed(scenario_script, pe_python)

            if output.returncode != 0:
                summary = self._summarize_oracle_error(output.stderr or output.stdout)
                if self._is_pe_unsupported_error(output.stderr or output.stdout):
                    issues.append(
                        f"PolicyEngine unavailable for '{test.get('name', test_rac_var)}': {summary}"
                    )
                    unsupported_count += 1
                    continue
                issues.append(
                    f"PE calculation failed for '{test.get('name', test_rac_var)}': {summary}"
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
                            f"'{test.get('name', test_rac_var)}': PE={pe_value:.2f}, RAC expects={expected_float:.2f}"
                        )
                    total += 1
                else:
                    issues.append(
                        f"No RESULT in PE output for '{test.get('name', test_rac_var)}'"
                    )
                    total += 1
            except Exception as parse_err:
                issues.append(
                    f"Parse error for '{test.get('name', test_rac_var)}': {parse_err}"
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

        def normalize_test_value(value: Any) -> Any:
            if isinstance(value, dict):
                lowered_keys = {str(key).lower() for key in value.keys()}
                if "value" in lowered_keys and lowered_keys.issubset(
                    {"entity", "value"}
                ):
                    for key, inner in value.items():
                        if str(key).lower() == "value":
                            return normalize_test_value(inner)
                singleton_entity_value_keys = {
                    "person",
                    "people",
                    "family",
                    "families",
                    "household",
                    "households",
                    "tax_unit",
                    "taxunit",
                    "tax_units",
                    "benunit",
                    "benunits",
                }
                if "value" in lowered_keys:
                    other_keys = lowered_keys - {"value"}
                    if len(other_keys) == 1 and next(iter(other_keys)) in singleton_entity_value_keys:
                        for key, inner in value.items():
                            if str(key).lower() == "value":
                                return normalize_test_value(inner)
                normalized = {
                    key: normalize_test_value(inner) for key, inner in value.items()
                }
                if len(normalized) == 1:
                    only_value = next(iter(normalized.values()))
                    if not isinstance(only_value, (dict, list)):
                        return only_value
                return normalized
            if isinstance(value, list):
                normalized_items = [normalize_test_value(item) for item in value]
                if len(normalized_items) == 1 and not isinstance(
                    normalized_items[0], (dict, list)
                ):
                    return normalized_items[0]
                return normalized_items
            if isinstance(value, str):
                compact = value.replace(",", "").strip()
                if re.fullmatch(r"-?\d+", compact):
                    return int(compact)
                if re.fullmatch(r"-?\d+\.\d+", compact):
                    return float(compact)
            return value

        def unwrap_entity_wrapper(value: Any) -> Any:
            if not isinstance(value, dict) or len(value) != 1:
                return value
            wrapper, inner = next(iter(value.items()))
            if not isinstance(inner, dict):
                return value
            wrapper_key = str(wrapper).lower().replace(" ", "_")
            entity_wrappers = {
                "person",
                "people",
                "family",
                "families",
                "household",
                "households",
                "tax_unit",
                "taxunit",
                "tax_units",
                "benunit",
                "benunits",
            }
            if wrapper_key not in entity_wrappers:
                return value
            if len(inner) == 1:
                _, nested = next(iter(inner.items()))
                if isinstance(nested, dict):
                    return nested
            return inner

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

            def append_top_level_io_tests(test_cases: Any) -> None:
                if not isinstance(test_cases, list):
                    return
                for test_case in test_cases:
                    if not isinstance(test_case, dict):
                        continue
                    outputs = test_case.get("output", test_case.get("expect"))
                    if not isinstance(outputs, dict):
                        continue
                    inputs = test_case.get("input", test_case.get("inputs", {}))
                    inputs = unwrap_entity_wrapper(inputs)
                    outputs = unwrap_entity_wrapper(outputs)
                    normalized_inputs = (
                        {
                            key: normalize_test_value(value)
                            for key, value in inputs.items()
                        }
                        if isinstance(inputs, dict)
                        else inputs or {}
                    )
                    for variable, expected in outputs.items():
                        tests.append(
                            {
                                "variable": variable,
                                "name": test_case.get("name"),
                                "period": test_case.get("period"),
                                "inputs": normalized_inputs,
                                "expect": normalize_test_value(expected),
                            }
                        )

            if isinstance(parsed, dict):
                if "tests" in parsed:
                    append_top_level_io_tests(parsed.get("tests"))
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
                                normalized_case = dict(test_case)
                                normalized_case["expect"] = normalize_test_value(
                                    normalized_case.get("expect")
                                )
                                if isinstance(normalized_case.get("inputs"), dict):
                                    normalized_case["inputs"] = {
                                        input_key: normalize_test_value(input_value)
                                        for input_key, input_value in normalized_case[
                                            "inputs"
                                        ].items()
                                    }
                                normalized_case["variable"] = key
                                tests.append(normalized_case)
                    # Top-level named test block:
                    # case_name:
                    #   period: ...
                    #   input: ...
                    #   output/expect: ...
                    elif isinstance(value, dict) and any(
                        io_key in value for io_key in ("input", "inputs")
                    ) and any(io_key in value for io_key in ("output", "expect")):
                        outputs = value.get("output", value.get("expect"))
                        if isinstance(outputs, dict):
                            inputs = value.get("input", value.get("inputs", {}))
                            inputs = unwrap_entity_wrapper(inputs)
                            outputs = unwrap_entity_wrapper(outputs)
                            normalized_inputs = (
                                {
                                    input_key: normalize_test_value(input_value)
                                    for input_key, input_value in inputs.items()
                                }
                                if isinstance(inputs, dict)
                                else inputs or {}
                            )
                            for variable, expected in outputs.items():
                                tests.append(
                                    {
                                        "variable": variable,
                                        "name": value.get("name", key),
                                        "period": value.get("period"),
                                        "inputs": normalized_inputs,
                                        "expect": normalize_test_value(expected),
                                    }
                                )
                    # Nested tests: section within a variable block
                    elif isinstance(value, dict) and "tests" in value:
                        for test_case in value.get("tests", []):
                            if isinstance(test_case, dict) and "expect" in test_case:
                                normalized_case = dict(test_case)
                                normalized_case["expect"] = normalize_test_value(
                                    normalized_case.get("expect")
                                )
                                if isinstance(normalized_case.get("inputs"), dict):
                                    normalized_case["inputs"] = {
                                        input_key: normalize_test_value(input_value)
                                        for input_key, input_value in normalized_case[
                                            "inputs"
                                        ].items()
                                    }
                                normalized_case["variable"] = key
                                tests.append(normalized_case)
            elif isinstance(parsed, list):
                append_top_level_io_tests(parsed)
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

    def _get_pe_variable_map(self, country: str = "us") -> dict[str, str]:
        """Map RAC variable names to PolicyEngine variable names.

        Returns dict of rac_var_name -> pe_var_name.
        """
        if country == "uk":
            return {
                "child_benefit_enhanced_rate": "child_benefit_respective_amount",
                "child_benefit_enhanced_rate_amount": "child_benefit_respective_amount",
                "child_benefit_enhanced_weekly_rate": "child_benefit_respective_amount",
                "child_benefit_rate_a_enhanced_rate": "child_benefit_respective_amount",
                "child_benefit_regulation_2_1_a_amount": "child_benefit_respective_amount",
                "child_benefit_reg2_1_a": "child_benefit_respective_amount",
                "child_benefit_weekly_rate": "child_benefit_respective_amount",
                "uk_child_benefit_other_child_weekly_rate": "child_benefit_respective_amount",
                "child_benefit_other_child_weekly_rate": "child_benefit_respective_amount",
                "child_benefit_weekly_rate_other_case": "child_benefit_respective_amount",
                "child_benefit_regulation_2_1_b_amount": "child_benefit_respective_amount",
                "child_benefit_reg2_1_b": "child_benefit_respective_amount",
                "standard_minimum_guarantee_couple_weekly_rate": "standard_minimum_guarantee",
                "standard_minimum_guarantee_single_weekly_rate": "standard_minimum_guarantee",
                "pc_severe_disability_addition_one_eligible_adult_weekly_rate": "severe_disability_minimum_guarantee_addition",
                "pc_severe_disability_addition_two_eligible_adults_weekly_rate": "severe_disability_minimum_guarantee_addition",
                "pc_carer_addition_weekly_rate": "carer_minimum_guarantee_addition",
                "pc_child_addition_weekly_rate": "child_minimum_guarantee_addition",
                "pc_disabled_child_addition_weekly_rate": "child_minimum_guarantee_addition",
                "pc_severely_disabled_child_addition_weekly_rate": "child_minimum_guarantee_addition",
                "scottish_child_payment_weekly_rate": "scottish_child_payment",
                "scottish_child_payment_weekly_amount": "scottish_child_payment",
                "scottish_child_payment_regulation_20_1_amount": "scottish_child_payment",
                "benefit_cap_single_claimant_greater_london_annual_limit": "benefit_cap",
                "benefit_cap_family_outside_london_annual_limit": "benefit_cap",
                "uc_standard_allowance_single_claimant_aged_under_25": "uc_standard_allowance",
                "uc_standard_allowance_single_claimant_aged_25_or_over": "uc_standard_allowance",
                "uc_standard_allowance_joint_claimants_both_aged_under_25": "uc_standard_allowance",
                "uc_standard_allowance_joint_claimants_one_or_both_aged_25_or_over": "uc_standard_allowance",
                "uc_carer_element_amount": "uc_carer_element",
                "uc_child_element_first_child_higher_amount": "uc_individual_child_element",
                "uc_child_element_second_and_subsequent_child_amount": "uc_individual_child_element",
                "uc_disabled_child_element_amount": "uc_individual_disabled_child_element",
                "uc_severely_disabled_child_element_amount": "uc_individual_severely_disabled_child_element",
                "uc_lcwra_element_amount": "uc_LCWRA_element",
                "uc_work_allowance_with_housing_amount": "uc_work_allowance",
                "uc_work_allowance_without_housing_amount": "uc_work_allowance",
                "uc_maximum_childcare_element_one_child_amount": "uc_maximum_childcare_element_amount",
                "uc_maximum_childcare_element_two_or_more_children_amount": "uc_maximum_childcare_element_amount",
                "uc_individual_non_dep_deduction_amount": "uc_individual_non_dep_deduction",
                "wtc_basic_element_amount": "WTC_basic_element",
                "wtc_lone_parent_element_amount": "WTC_lone_parent_element",
                "wtc_couple_element_amount": "WTC_couple_element",
                "wtc_second_adult_element_amount": "WTC_couple_element",
                "wtc_worker_element_amount": "WTC_worker_element",
                "working_tax_credit_worker_element_amount": "WTC_worker_element",
                "working_tax_credit_30_hours_element_amount": "WTC_worker_element",
                "wtc_disabled_element_amount": "WTC_disabled_element",
                "working_tax_credit_disabled_element_amount": "WTC_disabled_element",
                "wtc_severely_disabled_element_amount": "WTC_severely_disabled_element",
                "working_tax_credit_severely_disabled_element_amount": "WTC_severely_disabled_element",
            }

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

    @staticmethod
    def _is_uk_child_benefit_rate_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "child_benefit_enhanced_rate",
                "child_benefit_enhanced_weekly_rate",
                "child_benefit_rate_a",
                "child_benefit_weekly_rate",
                "regulation_2_1_a",
                "reg2_1_a",
                "child_benefit_other_child",
                "other_case",
                "regulation_2_1_b",
                "reg2_1_b",
            )
        )

    @staticmethod
    def _is_uk_child_benefit_other_child_rate_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "child_benefit_other_child",
                "other_case",
                "regulation_2_1_b",
                "reg2_1_b",
                "child_benefit_rate_b",
                "child_benefit_weekly_rate_b",
            )
        )

    @staticmethod
    def _is_uk_pension_credit_standard_minimum_guarantee_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        if "minimum_guarantee" in rac_var_lower and any(
            marker in rac_var_lower
            for marker in (
                "standard_minimum_guarantee",
                "pension_credit",
                "partner",
                "couple",
                "single",
                "guarantee_credit_standard_minimum_a",
                "guarantee_credit_standard_minimum_b",
            )
        ):
            return True
        return "guarantee_credit" in rac_var_lower and any(
            marker in rac_var_lower
            for marker in (
                "6_1_a",
                "6_1_b",
                "regulation_6_1",
                "standard_minimum_a",
                "standard_minimum_b",
            )
        )

    @staticmethod
    def _is_uk_pension_credit_couple_rate_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        if any(
            marker in rac_var_lower
            for marker in ("claimant_has_partner", "exception_applies", "_applies")
        ):
            return False
        return "no_partner" not in rac_var_lower and any(
            marker in rac_var_lower
            for marker in (
                "couple",
                "partner_rate",
                "with_partner",
                "partner",
                "6_1_a",
                "regulation_6_1_a",
                "minimum_guarantee_a",
                "guarantee_credit_standard_minimum_guarantee_a",
                "guarantee_credit_standard_minimum_a",
            )
        )

    @staticmethod
    def _is_uk_pension_credit_single_rate_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "single",
                "no_partner",
                "without_partner",
                "6_1_b",
                "regulation_6_1_b",
                "minimum_guarantee_b",
                "guarantee_credit_standard_minimum_guarantee_b",
                "guarantee_credit_standard_minimum_b",
            )
        )

    @staticmethod
    def _is_uk_pc_severe_disability_addition_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "pc_severe_disability_addition" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_pc_carer_addition_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "pc_carer_addition" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_pc_child_addition_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "pc_child_addition" in rac_var_lower
            and "disabled" not in rac_var_lower
            and not rac_var_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_pc_disabled_child_addition_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "pc_disabled_child_addition" in rac_var_lower
            and "severe" not in rac_var_lower
            and not rac_var_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_pc_severely_disabled_child_addition_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "pc_severely_disabled_child_addition" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_scottish_child_payment_rate_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        if "scottish_child_payment" not in rac_var_lower:
            return False
        if rac_var_lower == "scottish_child_payment":
            return True
        if any(
            marker in rac_var_lower
            for marker in ("_applies", "eligible", "would_claim", "qualifying")
        ):
            return False
        return any(
            marker in rac_var_lower
            for marker in (
                "amount",
                "rate",
                "weekly",
                "value",
                "regulation_20_1",
                "reg20_1",
            )
        )

    @staticmethod
    def _is_uk_benefit_cap_amount_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        if "benefit_cap" not in rac_var_lower and not (
            "80a_2_" in rac_var_lower
            and any(
                marker in rac_var_lower
                for marker in ("annual_limit", "relevant_amount", "_amount")
            )
        ):
            return False
        if rac_var_lower == "benefit_cap":
            return True
        if any(
            marker in rac_var_lower
            for marker in ("_applies", "exempt", "reduction", "relevant_amount_applies")
        ):
            return False
        return any(
            marker in rac_var_lower
            for marker in (
                "annual_limit",
                "_amount",
                "relevant_amount",
                "80a_2_",
                "single_claimant",
                "joint_claimant",
                "greater_london",
                "outside_london",
                "family",
            )
        )

    @staticmethod
    def _is_uk_uc_standard_allowance_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "uc_standard_allowance" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_uc_carer_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "uc_carer_element" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_uc_child_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "uc_child_element" in rac_var_lower and not rac_var_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_uc_lcwra_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in ("uc_lcwra_element", "uc_limited_capability_for_work_related_activity")
        ) and not rac_var_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_disabled_child_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            any(
                marker in rac_var_lower
                for marker in (
                    "uc_disabled_child_element",
                    "uc_child_element_disabled",
                    "universal_credit_disabled_child_element",
                )
            )
            and "severe" not in rac_var_lower
            and not rac_var_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_uc_severely_disabled_child_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "uc_severely_disabled_child_element",
                "uc_child_element_severely_disabled",
                "universal_credit_severely_disabled_child_element",
            )
        ) and not rac_var_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_work_allowance_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            (
                "uc_work_allowance" in rac_var_lower
                or "universal_credit_work_allowance" in rac_var_lower
            )
            and "eligible" not in rac_var_lower
            and not rac_var_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_uc_maximum_childcare_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "uc_maximum_childcare_element",
                "uc_childcare_cap",
                "universal_credit_childcare_cap",
            )
        ) and not rac_var_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_non_dep_deduction_amount_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            marker in rac_var_lower
            for marker in (
                "uc_individual_non_dep_deduction",
                "uc_housing_non_dep_deduction",
                "universal_credit_non_dep_deduction",
            )
        ) and not any(
            marker in rac_var_lower
            for marker in ("_eligible", "_exempt", "_applies", "non_dep_deductions")
        )

    @staticmethod
    def _is_uk_wtc_basic_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "wtc_basic" in rac_var_lower or "working_tax_credit_basic_element" in rac_var_lower

    @staticmethod
    def _is_uk_wtc_lone_parent_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return "wtc_lone_parent" in rac_var_lower or "working_tax_credit_lone_parent_element" in rac_var_lower

    @staticmethod
    def _is_uk_wtc_couple_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "wtc_couple" in rac_var_lower
            or "working_tax_credit_couple_element" in rac_var_lower
            or "wtc_second_adult" in rac_var_lower
            or "working_tax_credit_second_adult_element" in rac_var_lower
        )

    @staticmethod
    def _is_uk_wtc_worker_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "wtc_worker" in rac_var_lower
            or "working_tax_credit_worker_element" in rac_var_lower
            or "30_hour" in rac_var_lower
            or "30_hours" in rac_var_lower
            or "thirty_hour" in rac_var_lower
        )

    @staticmethod
    def _is_uk_wtc_disabled_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "wtc_disabled" in rac_var_lower
            or "working_tax_credit_disabled_element" in rac_var_lower
        ) and "severe" not in rac_var_lower

    @staticmethod
    def _is_uk_wtc_severely_disabled_element_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return (
            "wtc_severely_disabled" in rac_var_lower
            or "working_tax_credit_severely_disabled_element" in rac_var_lower
            or "working_tax_credit_severe_disability_element" in rac_var_lower
            or "wtc_severe_disability" in rac_var_lower
        )

    @staticmethod
    def _is_uk_table_row_amount_var(rac_var: str) -> bool:
        rac_var_lower = rac_var.lower()
        return any(
            checker(rac_var_lower)
            for checker in (
                ValidatorPipeline._is_uk_uc_standard_allowance_var,
                ValidatorPipeline._is_uk_uc_carer_element_var,
                ValidatorPipeline._is_uk_uc_child_element_var,
                ValidatorPipeline._is_uk_uc_lcwra_element_var,
                ValidatorPipeline._is_uk_uc_disabled_child_element_var,
                ValidatorPipeline._is_uk_uc_severely_disabled_child_element_var,
                ValidatorPipeline._is_uk_uc_work_allowance_var,
                ValidatorPipeline._is_uk_uc_maximum_childcare_element_var,
                ValidatorPipeline._is_uk_uc_non_dep_deduction_amount_var,
                ValidatorPipeline._is_uk_wtc_basic_element_var,
                ValidatorPipeline._is_uk_wtc_lone_parent_element_var,
                ValidatorPipeline._is_uk_wtc_couple_element_var,
                ValidatorPipeline._is_uk_wtc_worker_element_var,
                ValidatorPipeline._is_uk_wtc_disabled_element_var,
                ValidatorPipeline._is_uk_wtc_severely_disabled_element_var,
            )
        )

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

    def _is_pe_test_mappable(
        self, country: str, rac_var: str, inputs: dict, expected: Any = None
    ) -> tuple[bool, str | None]:
        """Return whether the test case can be represented in PolicyEngine."""
        rac_var_lower = rac_var.lower()
        if country == "uk" and isinstance(expected, dict):
            return (
                False,
                "RAC test expects multi-entity outputs that the current PolicyEngine UK harness cannot compare directly",
            )
        if (
            country == "uk"
            and self._is_uk_table_row_amount_var(rac_var_lower)
            and expected in {0, 0.0, "0", "0.0"}
        ):
            return (
                False,
                "RAC test is a row-specific zero case for a table amount slice that PolicyEngine UK does not represent as a separate zero-valued branch",
            )
        if country == "uk" and self._is_uk_child_benefit_rate_var(rac_var_lower):
            for key, value in inputs.items():
                key_lower = str(key).lower()
                if (
                    "subject_to_paragraphs" in key_lower
                    or "paragraphs_two_to_five_apply" in key_lower
                    or "paragraphs_2_to_5_apply" in key_lower
                ) and bool(value):
                    return (
                        False,
                        "RAC test uses placeholder paragraph-exception conditions that PolicyEngine UK does not represent directly",
                    )
                if "payable" in key_lower and not bool(value):
                    return (
                        False,
                        "RAC test encodes take-up/payability conditions that PolicyEngine UK's statutory rate variable does not represent directly",
                    )
            explicit_false_keys = {
                str(key).lower()
                for key, value in inputs.items()
                if value is not None and not bool(value)
            }
            if (
                "is_child_or_qualifying_young_person" in explicit_false_keys
                or (
                    any("is_child" in key for key in explicit_false_keys)
                    and any("qualifying_young_person" in key for key in explicit_false_keys)
                )
            ):
                return (
                    False,
                    "RAC test negates child-or-qualifying-young-person subject status that PolicyEngine UK's statutory child benefit rate does not expose as a separate comparable branch",
                )
        if country == "uk" and self._is_uk_child_benefit_rate_var(
            rac_var_lower
        ) and rac_var_lower.endswith("_applies"):
            return (
                False,
                "RAC helper boolean does not have a direct PolicyEngine UK analogue",
            )
        if country == "uk" and self._is_uk_pension_credit_standard_minimum_guarantee_var(
            rac_var_lower
        ):
            if (
                rac_var_lower.endswith("_applies")
                or "claimant_has_partner" in rac_var_lower
                or "exception_applies" in rac_var_lower
            ):
                return (
                    False,
                    "RAC helper boolean does not have a direct PolicyEngine UK analogue",
                )
            for key, value in inputs.items():
                key_lower = str(key).lower()
                if "exception_applies" in key_lower and bool(value):
                    return (
                        False,
                        "RAC test uses downstream regulation exceptions that PolicyEngine UK does not represent directly",
                    )
            if self._is_uk_pension_credit_single_rate_var(rac_var_lower) and any(
                (
                    (
                        "has_partner" in str(key).lower()
                        and "no_partner" not in str(key).lower()
                        and bool(value)
                    )
                    or (
                        "no_partner" in str(key).lower()
                        and value is not None
                        and not bool(value)
                    )
                )
                for key, value in inputs.items()
            ):
                return (
                    False,
                    "RAC test negates the pension-credit single-rate branch using partner facts that PolicyEngine UK only exposes through the parent standard minimum guarantee",
                )
            if self._is_uk_pension_credit_couple_rate_var(rac_var_lower) and (
                any(
                    "no_partner" in str(key).lower() and bool(value)
                    for key, value in inputs.items()
                )
                or any(
                    "has_partner" in str(key).lower()
                    and "no_partner" not in str(key).lower()
                    and value is not None
                    and not bool(value)
                    for key, value in inputs.items()
                )
            ):
                return (
                    False,
                    "RAC test negates the pension-credit couple-rate branch using partner facts that PolicyEngine UK only exposes through the parent standard minimum guarantee",
                )
        if (
            country == "uk"
            and "scottish_child_payment" in rac_var_lower
            and rac_var_lower.endswith("_applies")
        ):
            return (
                False,
                "RAC helper boolean does not have a direct PolicyEngine UK analogue",
            )
        if (
            country == "uk"
            and "benefit_cap" in rac_var_lower
            and rac_var_lower.endswith("_applies")
        ):
            return (
                False,
                "RAC helper boolean does not have a direct PolicyEngine UK analogue",
            )
        return True, None

    def _resolve_pe_variable(self, country: str, rac_var: str) -> str | None:
        """Resolve a RAC variable to a PolicyEngine variable, including heuristics."""
        pe_var = self._get_pe_variable_map(country).get(rac_var)
        if pe_var:
            return pe_var

        rac_var_lower = rac_var.lower()
        if country == "uk" and self._is_uk_child_benefit_rate_var(rac_var_lower):
            return "child_benefit_respective_amount"
        if country == "uk" and self._is_uk_pension_credit_standard_minimum_guarantee_var(
            rac_var_lower
        ):
            return "standard_minimum_guarantee"
        if country == "uk" and self._is_uk_pc_severe_disability_addition_var(
            rac_var_lower
        ):
            return "severe_disability_minimum_guarantee_addition"
        if country == "uk" and self._is_uk_pc_carer_addition_var(rac_var_lower):
            return "carer_minimum_guarantee_addition"
        if country == "uk" and (
            self._is_uk_pc_child_addition_var(rac_var_lower)
            or self._is_uk_pc_disabled_child_addition_var(rac_var_lower)
            or self._is_uk_pc_severely_disabled_child_addition_var(rac_var_lower)
        ):
            return "child_minimum_guarantee_addition"
        if country == "uk" and self._is_uk_uc_standard_allowance_var(rac_var_lower):
            return "uc_standard_allowance"
        if country == "uk" and self._is_uk_uc_carer_element_var(rac_var_lower):
            return "uc_carer_element"
        if country == "uk" and self._is_uk_uc_child_element_var(rac_var_lower):
            return "uc_individual_child_element"
        if country == "uk" and self._is_uk_uc_lcwra_element_var(rac_var_lower):
            return "uc_LCWRA_element"
        if country == "uk" and self._is_uk_uc_disabled_child_element_var(rac_var_lower):
            return "uc_individual_disabled_child_element"
        if country == "uk" and self._is_uk_uc_severely_disabled_child_element_var(
            rac_var_lower
        ):
            return "uc_individual_severely_disabled_child_element"
        if country == "uk" and self._is_uk_uc_work_allowance_var(rac_var_lower):
            return "uc_work_allowance"
        if country == "uk" and self._is_uk_uc_maximum_childcare_element_var(
            rac_var_lower
        ):
            return "uc_maximum_childcare_element_amount"
        if country == "uk" and self._is_uk_uc_non_dep_deduction_amount_var(
            rac_var_lower
        ):
            return "uc_individual_non_dep_deduction"
        if country == "uk" and self._is_uk_wtc_basic_element_var(rac_var_lower):
            return "WTC_basic_element"
        if country == "uk" and self._is_uk_wtc_lone_parent_element_var(rac_var_lower):
            return "WTC_lone_parent_element"
        if country == "uk" and self._is_uk_wtc_couple_element_var(rac_var_lower):
            return "WTC_couple_element"
        if country == "uk" and self._is_uk_wtc_worker_element_var(rac_var_lower):
            return "WTC_worker_element"
        if country == "uk" and self._is_uk_wtc_disabled_element_var(rac_var_lower):
            return "WTC_disabled_element"
        if country == "uk" and self._is_uk_wtc_severely_disabled_element_var(
            rac_var_lower
        ):
            return "WTC_severely_disabled_element"
        if country == "uk" and self._is_uk_scottish_child_payment_rate_var(
            rac_var_lower
        ):
            return "scottish_child_payment"
        if country == "uk" and self._is_uk_benefit_cap_amount_var(rac_var_lower):
            return "benefit_cap"

        return None

    def _build_pe_scenario_script(
        self,
        pe_var: str,
        inputs: dict,
        year: str,
        expected: Any,
        country: str = "us",
        rac_var: str | None = None,
    ) -> str:
        """Build a Python script to run a PE scenario via subprocess.

        Handles period detection (monthly vs annual PE variables),
        builds appropriate household structures, and overrides PE
        intermediate variables to match RAC test inputs for apples-to-apples
        comparison.
        """
        if country == "uk":
            return self._build_pe_uk_scenario_script(pe_var, inputs, year, rac_var)

        return self._build_pe_us_scenario_script(pe_var, inputs, year)

    def _build_pe_us_scenario_script(self, pe_var: str, inputs: dict, year: str) -> str:
        """Build a Python script to run a US PolicyEngine scenario."""
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

    def _build_pe_uk_scenario_script(
        self, pe_var: str, inputs: dict, year: str, rac_var: str | None = None
    ) -> str:
        """Build a Python script to run a UK PolicyEngine scenario."""
        period_value = str(inputs.get("period", f"{year}-04"))
        month_period = period_value[:7] if len(period_value) >= 7 else f"{year}-04"
        year_key = repr(str(year))
        rac_var_lower = (rac_var or "").lower()
        lowered = {str(key).lower(): value for key, value in inputs.items()}

        if pe_var == "uc_standard_allowance" and self._is_uk_uc_standard_allowance_var(
            rac_var_lower
        ):
            is_single = "couple" not in rac_var_lower and not any(
                marker in rac_var_lower for marker in ("joint", "partner")
            )
            if any(
                ("couple" in key or "joint" in key) and value is not None
                for key, value in lowered.items()
            ):
                is_single = not any(
                    bool(value)
                    for key, value in lowered.items()
                    if ("couple" in key or "joint" in key) and value is not None
                )
            under_25 = any(
                marker in rac_var_lower
                for marker in ("under_25", "aged_under_25", "young")
            ) and "over_25" not in rac_var_lower and "25_or_over" not in rac_var_lower
            if any(
                ("25_or_over" in key or "over_25" in key) and value is not None
                for key, value in lowered.items()
            ):
                under_25 = not any(
                    bool(value)
                    for key, value in lowered.items()
                    if ("25_or_over" in key or "over_25" in key) and value is not None
                )

            adult_ages = [24] if under_25 else [30]
            if not is_single:
                adult_ages = [24, 24] if under_25 else [30, 24]

            people_parts = [
                f"'adult': {{'age': {{{year_key}: {adult_ages[0]}}}}}",
            ]
            members = ["adult"]
            if not is_single:
                people_parts.append(
                    f"'spouse': {{'age': {{{year_key}: {adult_ages[1]}}}}}"
                )
                members.append("spouse")
            people = "{" + ", ".join(people_parts) + "}"
            members_str = "[" + ", ".join(f"'{member}'" for member in members) + "]"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {members_str}}}}},
    'households': {{'household': {{'members': {members_str}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_standard_allowance', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_carer_element" and self._is_uk_uc_carer_element_var(
            rac_var_lower
        ):
            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'adult': {{'age': {{{year_key}: 30}}, 'receives_carers_allowance': {{{year_key}: True}}}}}},
    'benunits': {{'benunit': {{'members': ['adult']}}}},
    'households': {{'household': {{'members': ['adult']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_carer_element', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_LCWRA_element" and self._is_uk_uc_lcwra_element_var(
            rac_var_lower
        ):
            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'adult': {{'age': {{{year_key}: 30}}, 'is_disabled_for_benefits': {{{year_key}: True}}}}}},
    'benunits': {{'benunit': {{'members': ['adult']}}}},
    'households': {{'household': {{'members': ['adult']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_LCWRA_element', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_individual_child_element" and self._is_uk_uc_child_element_var(
            rac_var_lower
        ):
            target_is_first_higher = any(
                marker in rac_var_lower for marker in ("first", "higher")
            )
            target_is_later_child = any(
                marker in rac_var_lower for marker in ("second", "subsequent")
            )

            if target_is_first_higher:
                people = (
                    f"{{'child': {{'age': {{{year_key}: 10}}, 'birth_year': {{{year_key}: 2015}}}}}}"
                )
                benunit_members = "['child']"
                household_members = "['child']"
                target_index = 0
            elif target_is_later_child:
                people = (
                    f"{{'older': {{'age': {{{year_key}: 10}}, 'birth_year': {{{year_key}: 2015}}}}, "
                    f"'child': {{'age': {{{year_key}: 7}}, 'birth_year': {{{year_key}: 2018}}}}}}"
                )
                benunit_members = "['older', 'child']"
                household_members = "['older', 'child']"
                target_index = 1
            else:
                people = (
                    f"{{'child': {{'age': {{{year_key}: 7}}, 'birth_year': {{{year_key}: 2018}}}}}}"
                )
                benunit_members = "['child']"
                household_members = "['child']"
                target_index = 0

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {benunit_members}}}}},
    'households': {{'household': {{'members': {household_members}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_individual_child_element', int('{year}'))
target_index = {target_index}
val = float(annual[target_index]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_individual_disabled_child_element" and self._is_uk_uc_disabled_child_element_var(
            rac_var_lower
        ):
            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'child': {{'age': {{{year_key}: 6}}, 'is_disabled_for_benefits': {{{year_key}: True}}}}}},
    'benunits': {{'benunit': {{'members': ['child']}}}},
    'households': {{'household': {{'members': ['child']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_individual_disabled_child_element', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_individual_severely_disabled_child_element" and self._is_uk_uc_severely_disabled_child_element_var(
            rac_var_lower
        ):
            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'child': {{'age': {{{year_key}: 6}}, 'is_severely_disabled_for_benefits': {{{year_key}: True}}, 'is_disabled_for_benefits': {{{year_key}: True}}}}}},
    'benunits': {{'benunit': {{'members': ['child']}}}},
    'households': {{'household': {{'members': ['child']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_individual_severely_disabled_child_element', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_work_allowance" and self._is_uk_uc_work_allowance_var(
            rac_var_lower
        ):
            explicit_with_housing = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if "with_housing" in str(key).lower() and value is not None
                ),
                None,
            )
            explicit_without_housing = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if "without_housing" in str(key).lower() and value is not None
                ),
                None,
            )
            if explicit_with_housing is not None:
                with_housing = explicit_with_housing
            elif explicit_without_housing is not None:
                with_housing = not explicit_without_housing
            elif "without_housing" in rac_var_lower:
                with_housing = False
            else:
                with_housing = True
            housing_costs_element = 1 if with_housing else 0

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{
        'adult': {{'age': {{{year_key}: 30}}}},
        'child': {{'age': {{{year_key}: 10}}}},
    }},
    'benunits': {{'benunit': {{'members': ['adult', 'child'], 'uc_housing_costs_element': {{{year_key}: {housing_costs_element}}}}}}},
    'households': {{'household': {{'members': ['adult', 'child']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_work_allowance', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_maximum_childcare_element_amount" and self._is_uk_uc_maximum_childcare_element_var(
            rac_var_lower
        ):
            explicit_children = next(
                (
                    int(value)
                    for key, value in lowered.items()
                    if (
                        "eligible_children" in str(key).lower()
                        or "childcare_children" in str(key).lower()
                    )
                    and value is not None
                ),
                None,
            )
            if explicit_children is not None:
                eligible_children = explicit_children
            elif "two_or_more" in rac_var_lower or "two_or_more_children" in rac_var_lower:
                eligible_children = 2
            else:
                eligible_children = 1

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'parent': {{'age': {{{year_key}: 30}}}}}},
    'benunits': {{'benunit': {{'members': ['parent'], 'uc_childcare_element_eligible_children': {{{year_key}: {eligible_children}}}}}}},
    'households': {{'household': {{'members': ['parent']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_maximum_childcare_element_amount', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var == "uc_individual_non_dep_deduction" and self._is_uk_uc_non_dep_deduction_amount_var(
            rac_var_lower
        ):
            explicit_exempt = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if "non_dep_deduction_exempt" in str(key).lower()
                    and value is not None
                ),
                False,
            )
            explicit_age = next(
                (
                    int(value)
                    for key, value in lowered.items()
                    if str(key).lower().endswith("age") and value is not None
                ),
                30,
            )

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'person': {{'age': {{{year_key}: {explicit_age}}}, 'uc_non_dep_deduction_exempt': {{{year_key}: {explicit_exempt}}}}}}},
    'benunits': {{'benunit': {{'members': ['person'], 'benunit_rent': {{{year_key}: 0}}}}}},
    'households': {{'household': {{'members': ['person']}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('uc_individual_non_dep_deduction', int('{year}'))
val = float(annual[0]) / 12
print(f'RESULT:{{val}}')
"""

        if pe_var in {
            "WTC_basic_element",
            "WTC_lone_parent_element",
            "WTC_couple_element",
            "WTC_worker_element",
            "WTC_disabled_element",
            "WTC_severely_disabled_element",
        } and (
            self._is_uk_wtc_basic_element_var(rac_var_lower)
            or self._is_uk_wtc_lone_parent_element_var(rac_var_lower)
            or self._is_uk_wtc_couple_element_var(rac_var_lower)
            or self._is_uk_wtc_worker_element_var(rac_var_lower)
            or self._is_uk_wtc_disabled_element_var(rac_var_lower)
            or self._is_uk_wtc_severely_disabled_element_var(rac_var_lower)
        ):
            if pe_var == "WTC_lone_parent_element":
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 16}}, 'working_tax_credit_reported': {{{year_key}: 1}}}}, "
                    f"'child': {{'age': {{{year_key}: 10}}}}}}"
                )
                benunit_members = "['adult', 'child']"
                household_members = "['adult', 'child']"
            elif pe_var == "WTC_couple_element":
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}}}, "
                    f"'spouse': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 0}}}}}}"
                )
                benunit_members = "['adult', 'spouse']"
                household_members = "['adult', 'spouse']"
            elif pe_var == "WTC_disabled_element":
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}, 'is_disabled_for_benefits': {{{year_key}: True}}}}}}"
                )
                benunit_members = "['adult']"
                household_members = "['adult']"
            elif pe_var == "WTC_severely_disabled_element":
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}, 'is_disabled_for_benefits': {{{year_key}: True}}, 'is_severely_disabled_for_benefits': {{{year_key}: True}}}}}}"
                )
                benunit_members = "['adult']"
                household_members = "['adult']"
            else:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}}}}}"
                )
                benunit_members = "['adult']"
                household_members = "['adult']"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {benunit_members}}}}},
    'households': {{'household': {{'members': {household_members}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('{pe_var}', int('{year}'))
val = float(annual[0])
print(f'RESULT:{{val}}')
"""

        if pe_var == "standard_minimum_guarantee" and self._is_uk_pension_credit_standard_minimum_guarantee_var(
            rac_var_lower
        ):
            explicit_has_partner = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        ("has_partner" in str(key).lower())
                        and "no_partner" not in str(key).lower()
                        and value is not None
                    )
                ),
                None,
            )
            relation_type = next(
                (
                    str(value).lower()
                    for key, value in lowered.items()
                    if "relation_type" in key and value is not None
                ),
                None,
            )
            if explicit_has_partner is not None:
                scenario_is_couple = explicit_has_partner
            elif relation_type is not None:
                scenario_is_couple = "couple" in relation_type
            elif any("no_partner" in key and bool(value) for key, value in lowered.items()):
                scenario_is_couple = False
            elif any(
                (
                    ("has_partner" in key and "no_partner" not in key)
                    or "is_couple" in key
                    or key.endswith("_couple")
                )
                and bool(value)
                for key, value in lowered.items()
            ):
                scenario_is_couple = True
            elif self._is_uk_pension_credit_couple_rate_var(rac_var_lower):
                scenario_is_couple = True
            else:
                scenario_is_couple = False

            people = f"{{'adult': {{'age': {{{year_key}: 70}}}}}}"
            benunit_members = "['adult']"
            household_members = "['adult']"
            if scenario_is_couple:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'spouse': {{'age': {{{year_key}: 70}}}}}}"
                )
                benunit_members = "['adult', 'spouse']"
                household_members = "['adult', 'spouse']"

            if self._is_uk_pension_credit_couple_rate_var(rac_var_lower):
                result_logic = """
if scenario_is_couple:
    val = weekly
else:
    val = 0.0
"""
            elif self._is_uk_pension_credit_single_rate_var(rac_var_lower):
                result_logic = """
if scenario_is_couple:
    val = 0.0
else:
    val = weekly
"""
            else:
                result_logic = "val = weekly"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {benunit_members}}}}},
    'households': {{'household': {{'members': {household_members}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('{pe_var}', int('{year}'))
weekly = float(annual[0]) / 52
scenario_is_couple = {scenario_is_couple}
{result_logic.rstrip()}
print(f'RESULT:{{val}}')
"""

        if pe_var == "severe_disability_minimum_guarantee_addition" and self._is_uk_pc_severe_disability_addition_var(
            rac_var_lower
        ):
            explicit_eligible_adults = next(
                (
                    int(value)
                    for key, value in lowered.items()
                    if "eligible_adult" in str(key).lower() and value is not None
                ),
                None,
            )
            ineligible = any(
                (
                    any(
                        marker in str(key).lower()
                        for marker in (
                            "severe_disability",
                            "paragraph_1",
                            "schedule_i",
                            "additional_amount",
                            "qualifies",
                            "eligible",
                            "applies",
                        )
                    )
                    and value is not None
                    and not bool(value)
                )
                for key, value in lowered.items()
            ) or any(
                "carer" in str(key).lower()
                and "no_carer" not in str(key).lower()
                and value is not None
                and bool(value)
                for key, value in lowered.items()
            )
            if explicit_eligible_adults is not None:
                eligible_adults = explicit_eligible_adults
            elif ineligible:
                eligible_adults = 0
            elif "two_eligible_adults" in rac_var_lower or "double" in rac_var_lower:
                eligible_adults = 2
            else:
                eligible_adults = 1

            if eligible_adults >= 2:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}, 'attendance_allowance': {{{year_key}: 1}}}}, "
                    f"'spouse': {{'age': {{{year_key}: 70}}, 'attendance_allowance': {{{year_key}: 1}}}}}}"
                )
                members = "['adult', 'spouse']"
            elif eligible_adults == 1:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}, 'attendance_allowance': {{{year_key}: 1}}}}}}"
                )
                members = "['adult']"
            else:
                people = f"{{'adult': {{'age': {{{year_key}: 70}}}}}}"
                members = "['adult']"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {members}}}}},
    'households': {{'household': {{'members': {members}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('severe_disability_minimum_guarantee_addition', int('{year}'))
val = float(annual[0]) / 52
print(f'RESULT:{{val}}')
"""

        if pe_var == "carer_minimum_guarantee_addition" and self._is_uk_pc_carer_addition_var(
            rac_var_lower
        ):
            explicit_eligible_carers = next(
                (
                    int(value)
                    for key, value in lowered.items()
                    if "eligible_carer" in str(key).lower() and value is not None
                ),
                None,
            )
            if explicit_eligible_carers is not None:
                eligible_carers = explicit_eligible_carers
            elif any(
                (
                    any(
                        marker in str(key).lower()
                        for marker in ("carer", "paragraph_4", "schedule_i", "applies")
                    )
                    and value is not None
                    and not bool(value)
                )
                for key, value in lowered.items()
            ):
                eligible_carers = 0
            else:
                eligible_carers = 1

            if eligible_carers >= 2:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}, 'carers_allowance': {{{year_key}: 1}}}}, "
                    f"'spouse': {{'age': {{{year_key}: 70}}, 'carers_allowance': {{{year_key}: 1}}}}}}"
                )
                members = "['adult', 'spouse']"
            elif eligible_carers == 1:
                people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}, 'carers_allowance': {{{year_key}: 1}}}}}}"
                )
                members = "['adult']"
            else:
                people = f"{{'adult': {{'age': {{{year_key}: 70}}}}}}"
                members = "['adult']"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {members}}}}},
    'households': {{'household': {{'members': {members}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('carer_minimum_guarantee_addition', int('{year}'))
val = float(annual[0]) / 52
print(f'RESULT:{{val}}')
"""

        if pe_var == "child_minimum_guarantee_addition" and (
            self._is_uk_pc_child_addition_var(rac_var_lower)
            or self._is_uk_pc_disabled_child_addition_var(rac_var_lower)
            or self._is_uk_pc_severely_disabled_child_addition_var(rac_var_lower)
        ):
            has_child = not any(
                (
                    any(
                        marker in str(key).lower()
                        for marker in (
                            "child_addition_applies",
                            "disabled_child_addition_applies",
                            "severely_disabled_child_addition_applies",
                            "qualifying_young_person",
                            "is_child",
                            "has_child",
                        )
                    )
                    and value is not None
                    and not bool(value)
                )
                for key, value in lowered.items()
            )
            target_mode = "base"
            if self._is_uk_pc_severely_disabled_child_addition_var(rac_var_lower):
                target_mode = "severe"
            elif self._is_uk_pc_disabled_child_addition_var(rac_var_lower):
                target_mode = "disabled"

            if has_child:
                base_people = (
                    f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}}}}}"
                )
                if target_mode == "disabled":
                    target_people = (
                        f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}, 'dla': {{{year_key}: 1}}}}}}"
                    )
                elif target_mode == "severe":
                    target_people = (
                        f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}, 'dla': {{{year_key}: 1}}, 'receives_highest_dla_sc': {{{year_key}: True}}}}}}"
                    )
                else:
                    target_people = base_people
                members = "['adult', 'child']"
            else:
                base_people = f"{{'adult': {{'age': {{{year_key}: 70}}}}}}"
                target_people = base_people
                members = "['adult']"

            if target_mode == "base":
                result_logic = "val = float(target_annual[0]) / 52"
            else:
                result_logic = "val = (float(target_annual[0]) - float(base_annual[0])) / 52"

            return f"""
from policyengine_uk import Simulation

base_situation = {{
    'people': {base_people},
    'benunits': {{'benunit': {{'members': {members}}}}},
    'households': {{'household': {{'members': {members}}}}},
}}
target_situation = {{
    'people': {target_people},
    'benunits': {{'benunit': {{'members': {members}}}}},
    'households': {{'household': {{'members': {members}}}}},
}}

base_sim = Simulation(situation=base_situation)
target_sim = Simulation(situation=target_situation)
base_annual = base_sim.calculate('child_minimum_guarantee_addition', int('{year}'))
target_annual = target_sim.calculate('child_minimum_guarantee_addition', int('{year}'))
{result_logic}
print(f'RESULT:{{val}}')
"""

        if pe_var == "scottish_child_payment" and self._is_uk_scottish_child_payment_rate_var(
            rac_var_lower
        ):
            in_scotland = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if "scotland" in str(key).lower() and value is not None
                ),
                True,
            )
            would_claim = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "would_claim_scp" in str(key).lower()
                        or "claim_scp" in str(key).lower()
                        or "payable" in str(key).lower()
                    )
                    and value is not None
                ),
                True,
            )
            eligible_child = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "eligible_child" in str(key).lower()
                        or "is_child" in str(key).lower()
                        or "qualifying_child" in str(key).lower()
                    )
                    and value is not None
                ),
                True,
            )
            child_age = 10 if eligible_child else 17
            qualifying_benefit_amount = next(
                (
                    float(value)
                    for key, value in lowered.items()
                    if "universal_credit" in str(key).lower() and value is not None
                ),
                1.0,
            )
            if any(
                ("qualifying_benefit" in str(key).lower()) and not bool(value)
                for key, value in lowered.items()
            ):
                qualifying_benefit_amount = 0.0

            country_value = "SCOTLAND" if in_scotland else "ENGLAND"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {{'child': {{'age': {{{year_key}: {child_age}}}, 'would_claim_scp': {{{year_key}: {would_claim}}}}}}},
    'benunits': {{'benunit': {{'members': ['child'], 'universal_credit': {{{year_key}: {qualifying_benefit_amount}}}}}}},
    'households': {{'household': {{'members': ['child'], 'country': {{{year_key}: '{country_value}'}}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('scottish_child_payment', int('{year}'))
val = float(annual[0]) / 52
print(f'RESULT:{{val}}')
"""

        if pe_var == "benefit_cap" and self._is_uk_benefit_cap_amount_var(
            rac_var_lower
        ):
            lowered_keys = [str(key).lower() for key in lowered.keys()]
            branch_category = None
            if "80a_2_a" in rac_var_lower:
                branch_category = ("single", "london", "no_child")
            elif "80a_2_b_ii" in rac_var_lower:
                branch_category = ("single", "london", "child")
            elif "80a_2_b_i" in rac_var_lower:
                branch_category = ("joint", "london", "any")
            elif "80a_2_b" in rac_var_lower:
                branch_category = ("other", "london", "mixed")
            elif "80a_2_c" in rac_var_lower:
                branch_category = ("single", "outside_london", "no_child")
            elif "80a_2_d_ii" in rac_var_lower:
                branch_category = ("single", "outside_london", "child")
            elif "80a_2_d_i" in rac_var_lower:
                branch_category = ("joint", "outside_london", "any")
            elif "80a_2_d" in rac_var_lower:
                branch_category = ("other", "outside_london", "mixed")

            leaf_in_london = any(
                marker in rac_var_lower
                for marker in ("greater_london", "in_london", "london")
            ) and "outside_london" not in rac_var_lower
            leaf_is_single = (
                any(marker in rac_var_lower for marker in ("single_claimant", "single"))
                and not any(
                    marker in rac_var_lower
                    for marker in ("joint_claimant", "couple", "family")
                )
            )
            leaf_has_child = any(
                marker in rac_var_lower
                for marker in ("child", "young_person", "family")
            ) and "no_child" not in rac_var_lower
            has_leaf_location_hint = any(
                marker in rac_var_lower
                for marker in (
                    "greater_london",
                    "in_london",
                    "london",
                    "outside_london",
                    "not_resident_in_greater_london",
                )
            )
            has_leaf_single_hint = any(
                marker in rac_var_lower
                for marker in (
                    "single_claimant",
                    "single",
                    "joint_claimant",
                    "joint_claimants",
                    "couple",
                    "family",
                )
            )
            has_leaf_child_hint = any(
                marker in rac_var_lower
                for marker in (
                    "no_child",
                    "without_child",
                    "not_responsible_for_child_or_qualifying_young_person",
                    "responsible_for_child_or_qualifying_young_person",
                    "child",
                    "young_person",
                    "family",
                )
            )

            if branch_category is not None:
                leaf_is_single = branch_category[0] == "single"
                leaf_in_london = branch_category[1] == "london"
                leaf_has_child = branch_category[2] == "child"

            if branch_category is None:
                if not has_leaf_location_hint:
                    if any("outside_london" in key for key in lowered_keys):
                        leaf_in_london = False
                    elif any(
                        "not_resident_in_greater_london" in key
                        for key in lowered_keys
                    ):
                        leaf_in_london = False
                    elif any("greater_london" in key for key in lowered_keys):
                        leaf_in_london = True

                if not has_leaf_single_hint:
                    if any(
                        "joint_claimant" in key or "couple" in key or "family" in key
                        for key in lowered_keys
                    ):
                        leaf_is_single = False
                    elif any(
                        "single_claimant" in key or key.endswith("single")
                        for key in lowered_keys
                    ):
                        leaf_is_single = True

                if not has_leaf_child_hint:
                    if any(
                        "not_responsible_for_child_or_qualifying_young_person" in key
                        or "no_child" in key
                        or "without_child" in key
                        for key in lowered_keys
                    ):
                        leaf_has_child = False
                    elif any(
                        (
                            "responsible_for_child_or_qualifying_young_person" in key
                            or "child" in key
                            or "young_person" in key
                            or "family" in key
                        )
                        and "not_responsible_for_child_or_qualifying_young_person"
                        not in key
                        for key in lowered_keys
                    ):
                        leaf_has_child = True

            in_london = leaf_in_london
            explicit_greater_london_keys = [
                bool(value)
                for key, value in lowered.items()
                if (
                    "greater_london" in str(key).lower()
                    and "not_resident_in_greater_london" not in str(key).lower()
                    and value is not None
                )
            ]
            if explicit_greater_london_keys:
                in_london = any(explicit_greater_london_keys)
            elif any(
                "not_resident_in_greater_london" in str(key).lower() and value is not None
                for key, value in lowered.items()
            ):
                in_london = not any(
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "not_resident_in_greater_london" in str(key).lower()
                        and value is not None
                    )
                )
            elif any(
                "outside_london" in str(key).lower() and value is not None
                for key, value in lowered.items()
            ):
                in_london = False

            is_single = leaf_is_single
            if any(
                (
                    str(key).lower() in {"joint_claimant", "joint_claimants"}
                    or str(key).lower().endswith("_joint_claimant")
                    or str(key).lower().endswith("_joint_claimants")
                    or "couple" in str(key).lower()
                )
                and value is not None
                for key, value in lowered.items()
            ):
                is_single = not any(
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        str(key).lower() in {"joint_claimant", "joint_claimants"}
                        or str(key).lower().endswith("_joint_claimant")
                        or str(key).lower().endswith("_joint_claimants")
                        or "couple" in str(key).lower()
                    )
                    and value is not None
                )
            elif any(
                "single" in str(key).lower() and value is not None
                for key, value in lowered.items()
            ):
                is_single = any(
                    bool(value)
                    for key, value in lowered.items()
                    if "single" in str(key).lower() and value is not None
                )

            has_child = leaf_has_child
            explicit_not_responsible = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "not_responsible_for_child_or_qualifying_young_person"
                        in str(key).lower()
                    )
                    and value is not None
                ),
                None,
            )
            if explicit_not_responsible is not None:
                has_child = not explicit_not_responsible
            explicit_responsible = next(
                (
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "responsible_for_child_or_qualifying_young_person"
                        in str(key).lower()
                        and "not_responsible_for_child_or_qualifying_young_person"
                        not in str(key).lower()
                        and value is not None
                    )
                ),
                None,
            )
            if explicit_responsible is not None:
                has_child = explicit_responsible
            if any(
                (
                    "no_child" in str(key).lower()
                    or "without_child" in str(key).lower()
                )
                and bool(value)
                for key, value in lowered.items()
            ):
                has_child = False
            elif (
                explicit_not_responsible is None
                and explicit_responsible is None
                and any(
                (
                    "child" in str(key).lower()
                    or "young_person" in str(key).lower()
                )
                and value is not None
                for key, value in lowered.items()
                )
            ):
                has_child = any(
                    bool(value)
                    for key, value in lowered.items()
                    if (
                        "child" in str(key).lower()
                        or "young_person" in str(key).lower()
                    )
                    and value is not None
                )

            members = ["adult"] if is_single else ["adult", "spouse"]
            people_parts = [f"'adult': {{'age': {{{year_key}: 30}}}}"]
            if not is_single:
                people_parts.append(f"'spouse': {{'age': {{{year_key}: 30}}}}")
            if has_child:
                members.append("child")
                people_parts.append(f"'child': {{'age': {{{year_key}: 10}}}}")

            region_value = "LONDON" if in_london else "NORTH_EAST"
            people = "{" + ", ".join(people_parts) + "}"
            members_str = "[" + ", ".join(f"'{member}'" for member in members) + "]"
            if branch_category == ("joint", "london", "any"):
                match_condition = "if not is_single and in_london:"
            elif branch_category == ("single", "london", "child"):
                match_condition = "if is_single and in_london and has_child:"
            elif branch_category == ("joint", "outside_london", "any"):
                match_condition = "if not is_single and not in_london:"
            elif branch_category == ("single", "outside_london", "child"):
                match_condition = "if is_single and not in_london and has_child:"
            elif leaf_is_single and leaf_in_london and not leaf_has_child:
                match_condition = "if is_single and in_london and not has_child:"
            elif leaf_is_single and not leaf_in_london and not leaf_has_child:
                match_condition = "if is_single and not in_london and not has_child:"
            elif leaf_in_london:
                match_condition = "if in_london and (not is_single or has_child):"
            else:
                match_condition = "if not in_london and (not is_single or has_child):"

            return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {members_str}, 'is_benefit_cap_exempt': {{{year_key}: False}}}}}},
    'households': {{'household': {{'members': {members_str}, 'region': {{{year_key}: '{region_value}'}}}}}},
}}

sim = Simulation(situation=situation)
annual = sim.calculate('benefit_cap', int('{year}'))
is_single = {is_single}
in_london = {in_london}
has_child = {has_child}
{match_condition}
    val = float(annual[0])
else:
    val = 0.0
print(f'RESULT:{{val}}')
"""

        only_person = any(
            "only_person" in key and bool(value) for key, value in lowered.items()
        )
        elder_or_eldest = any(
            (
                "elder_or_eldest" in key
                or "eldest_person" in key
                or "eldest_child" in key
                or "only_or_eldest" in key
                or "eldest_or_only" in key
            )
            and bool(value)
            for key, value in lowered.items()
        )
        payable = next(
            (
                bool(value)
                for key, value in lowered.items()
                if "payable" in key or "would_claim_child_benefit" in key
            ),
            True,
        )
        other_case = next(
            (
                bool(value)
                for key, value in lowered.items()
                if "other_case" in key and value is not None
            ),
            None,
        )
        enhanced_rate_condition = next(
            (
                bool(value)
                for key, value in lowered.items()
                if "enhanced_rate_condition" in key and value is not None
            ),
            None,
        )
        if enhanced_rate_condition is not None and not (only_person or elder_or_eldest):
            elder_or_eldest = enhanced_rate_condition
        child_or_qyp = next(
            (
                bool(value)
                for key, value in lowered.items()
                if (
                    "child_or_qualifying_young_person" in key
                    or "child_or_qyp" in key
                )
                and value is not None
            ),
            True,
        )
        explicit_is_child = next(
            (
                bool(value)
                for key, value in lowered.items()
                if str(key).lower() == "is_child" and value is not None
            ),
            None,
        )
        explicit_is_qyp = next(
            (
                bool(value)
                for key, value in lowered.items()
                if str(key).lower() == "is_qualifying_young_person" and value is not None
            ),
            None,
        )
        if explicit_is_child is not None or explicit_is_qyp is not None:
            child_or_qyp = bool(explicit_is_child) or bool(explicit_is_qyp)
        age_order = next(
            (
                int(value)
                for key, value in lowered.items()
                if "age_order" in key and value is not None
            ),
            None,
        )

        if not child_or_qyp:
            people = f"{{'target': {{'age': {{{year_key}: 20}}}}}}"
            benunit_members = "['target']"
            household_members = "['target']"
            target_index = 0
        elif age_order is not None:
            if age_order <= 1:
                people = f"{{'target': {{'age': {{{year_key}: 10}}}}}}"
                benunit_members = "['target']"
                household_members = "['target']"
                target_index = 0
            else:
                people = f"""{{'older': {{'age': {{{year_key}: 12}}}}, 'target': {{'age': {{{year_key}: 11}}}}}}"""
                benunit_members = "['older', 'target']"
                household_members = "['older', 'target']"
                target_index = 1
        elif only_person:
            people = f"{{'target': {{'age': {{{year_key}: 10}}}}}}"
            benunit_members = "['target']"
            household_members = "['target']"
            target_index = 0
        elif elder_or_eldest or other_case is False:
            people = f"""{{'target': {{'age': {{{year_key}: 12}}}}, 'younger': {{'age': {{{year_key}: 11}}}}}}"""
            benunit_members = "['target', 'younger']"
            household_members = "['target', 'younger']"
            target_index = 0
        else:
            people = f"""{{'older': {{'age': {{{year_key}: 12}}}}, 'target': {{'age': {{{year_key}: 11}}}}}}"""
            benunit_members = "['older', 'target']"
            household_members = "['older', 'target']"
            target_index = 1

        value_expr = "float(monthly[target_index]) * 12 / 52"
        use_other_child_branch = self._is_uk_child_benefit_other_child_rate_var(
            rac_var_lower
        ) or (
            rac_var_lower == "child_benefit_weekly_rate" and other_case is not None
        )
        if use_other_child_branch:
            result_logic = f"""
if bool(eldest[target_index]):
    val = 0.0
else:
    val = {value_expr}
"""
        else:
            result_logic = f"""
if bool(eldest[target_index]):
    val = {value_expr}
else:
    val = 0.0
"""

        return f"""
from policyengine_uk import Simulation

situation = {{
    'people': {people},
    'benunits': {{'benunit': {{'members': {benunit_members}, 'would_claim_child_benefit': {{{year_key}: {payable}}}}}}},
    'households': {{'household': {{'members': {household_members}}}}},
}}

sim = Simulation(situation=situation)
monthly = sim.calculate('{pe_var}', '{month_period}')
eldest = sim.calculate('is_eldest_child', '{month_period}')
target_index = {target_index}
{result_logic.rstrip()}
print(f'RESULT:{{val}}')
"""


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
