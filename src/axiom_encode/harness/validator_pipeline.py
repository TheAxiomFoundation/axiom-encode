"""
Validator Pipeline - 3-tier validation architecture.

Tiers (run in order):
1. RuleSpec compile checks - instant, catches syntax/format errors
2. External oracles (PolicyEngine, TAXSIM) - fast (~10s), generates comparison data
3. LLM reviewers (RuleSpec, formula, parameter, integration) - uses oracle context

Oracles run BEFORE LLM reviewers because:
- They're fast and free (no API costs)
- They generate rich comparison context for LLM analysis
- LLMs can diagnose WHY discrepancies exist, not just that they exist

Uses Claude Code CLI (subprocess) for reviewer agents - cheaper than direct API.
"""

import contextlib
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from calendar import monthrange
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from axiom_encode.codex_cli import resolve_codex_cli
from axiom_encode.constants import DEFAULT_OPENAI_MODEL, REVIEWER_CLI_MODEL
from axiom_encode.oracles.policyengine.adapters import (
    PE_US_MONTHLY_VAR_NAMES,
    PE_US_SPM_VAR_NAMES,
    PolicyEngineUSVarAdapter,
    get_pe_us_var_adapter,
    normalize_state_code_from_utility_region,
)
from axiom_encode.oracles.policyengine.registry import (
    PolicyEngineMapping,
    PolicyEngineOracleCoverage,
    load_policyengine_registry,
)
from axiom_encode.repo_routing import find_policy_repo_root

from .dependency_stubs import (
    has_corpus_provision_for_import_target,
    resolve_canonical_concepts_from_text,
    resolve_defined_terms_from_text,
    rulespec_content_has_stub_status,
    rulespec_file_has_stub_status,
)
from .encoding_db import EncodingDB, ReviewResult, ReviewResults
from .proof_validator import find_rulespec_proof_issues, validate_rulespec_proofs

logger = logging.getLogger(__name__)

DEFAULT_AXIOM_SUPABASE_URL = "https://swocpijqqahhuwtuahwc.supabase.co"
DEFAULT_AXIOM_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN3b2NwaWpxcWFoaHV3dHVhaHdjI"
    "iwicm9sZSI6ImFub24iLCJpYXQiOjE3NzczMzU3NzcsImV4cCI6MjA5Mjkx"
    "MTc3N30."
    "spiF6Z6LLJmETL8eI0z_QbwgXce7J5CIqHTiXZ6K9Zk"
)


def run_claude_code(
    prompt: str,
    model: str = REVIEWER_CLI_MODEL,
    timeout: int = 120,
    cwd: Optional[Path] = None,
) -> tuple[str, int]:
    """
    Run reviewer CLI as subprocess.

    Prefer Claude Code CLI when available, but fall back to Codex CLI on
    machines where Claude is not installed. This keeps reviewer-based evals
    working in local Codex-only environments.

    Returns:
        Tuple of (output text, return code)
    """
    reviewer_cli_preference = os.getenv("AXIOM_ENCODE_REVIEWER_CLI", "").strip().lower()
    if reviewer_cli_preference == "codex":
        return _run_codex_reviewer_cli(prompt, timeout=timeout, cwd=cwd)

    cmd = ["claude", "--print", "--model", model, "-p", prompt]

    try:
        idle_timeout_env = os.getenv(
            "AXIOM_ENCODE_REVIEWER_CLAUDE_IDLE_TIMEOUT_SECONDS"
        )
        idle_timeout = timeout
        if idle_timeout_env is not None:
            idle_timeout = min(timeout, max(0, int(idle_timeout_env)))
        result = _run_subprocess_with_idle_timeout(
            cmd,
            timeout=timeout,
            idle_timeout=idle_timeout,
            cwd=cwd,
        )
        return result.output, result.returncode
    except subprocess.TimeoutExpired as exc:
        return f"Timeout after {exc.timeout}s", 1
    except FileNotFoundError:
        return _run_codex_reviewer_cli(prompt, timeout=timeout, cwd=cwd)
    except Exception as e:
        return f"Error: {e}", 1


def _run_codex_reviewer_cli(
    prompt: str,
    timeout: int = 120,
    cwd: Optional[Path] = None,
) -> tuple[str, int]:
    """Run reviewer prompts through Codex CLI and return assistant text."""
    cmd = [
        resolve_codex_cli(),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        os.environ.get("AXIOM_ENCODE_REVIEWER_CODEX_MODEL", DEFAULT_OPENAI_MODEL),
    ]
    if cwd is not None:
        cmd.extend(["-C", str(cwd)])
    cmd.append(prompt)

    try:
        idle_timeout = min(
            timeout,
            max(
                0,
                int(
                    os.getenv(
                        "AXIOM_ENCODE_REVIEWER_CODEX_IDLE_TIMEOUT_SECONDS",
                        "45",
                    )
                ),
            ),
        )
        result = _run_subprocess_with_idle_timeout(
            cmd,
            timeout=timeout,
            idle_timeout=idle_timeout,
            cwd=cwd,
        )
        return _extract_codex_text_output(result.output), result.returncode
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s", 1
    except FileNotFoundError:
        return "Reviewer CLIs not found (missing claude and codex)", 1
    except Exception as e:
        return f"Error: {e}", 1


@dataclass
class _SubprocessRunResult:
    """Captured subprocess output plus exit status."""

    output: str
    returncode: int


def _sha256_text(text: str | None) -> str | None:
    """Return a stable digest for prompt text."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_subprocess_with_idle_timeout(
    cmd: list[str],
    *,
    timeout: int,
    idle_timeout: int,
    cwd: Optional[Path] = None,
    poll_interval: float = 0.5,
) -> _SubprocessRunResult:
    """Run a subprocess, aborting if it stops emitting output for too long."""
    with (
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as stdout_file,
        tempfile.NamedTemporaryFile(mode="w+", delete=False) as stderr_file,
    ):
        stdout_path = Path(stdout_file.name)
        stderr_path = Path(stderr_file.name)
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            cwd=cwd,
        )

    start = time.time()
    last_activity = start
    last_snapshot: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None

    def _snapshot() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        values: list[tuple[int, int, int]] = []
        for path in (stdout_path, stderr_path):
            try:
                stat = path.stat()
            except OSError:
                values.append((0, 0, 0))
                continue
            values.append((1, stat.st_size, stat.st_mtime_ns))
        return values[0], values[1]

    try:
        while True:
            if process.poll() is not None:
                break

            now = time.time()
            if now - start > timeout:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, timeout)

            snapshot = _snapshot()
            if snapshot != last_snapshot:
                last_snapshot = snapshot
                last_activity = now
            elif idle_timeout >= 0 and now - last_activity >= idle_timeout:
                process.kill()
                process.wait()
                raise subprocess.TimeoutExpired(cmd, idle_timeout)

            time.sleep(poll_interval)

        output = stdout_path.read_text() + stderr_path.read_text()
        return _SubprocessRunResult(output=output, returncode=process.returncode or 0)
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def _extract_codex_text_output(output: str) -> str:
    """Return the concatenated assistant text from a Codex JSONL stream."""
    assistant_messages: list[str] = []
    last_error: str | None = None

    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        if payload_type == "item.completed":
            item = payload.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                assistant_messages.append(item["text"])
        elif payload_type == "error":
            last_error = payload.get("message") or "codex exec error"

    return "\n".join(assistant_messages).strip() or last_error or output


_REVIEW_JSON_KEYS = {
    "score",
    "passed",
    "issues",
    "blocking_issues",
    "non_blocking_issues",
    "reasoning",
}


def _looks_like_review_json(data: dict[str, Any]) -> bool:
    """Return true for the reviewer payload, not surrounding CLI metadata."""
    return bool(_REVIEW_JSON_KEYS.intersection(data))


def _strip_trailing_json_commas(text: str) -> str:
    """Remove common model-emitted trailing commas before JSON closers."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _decode_json_object_candidate(text: str) -> dict[str, Any] | None:
    """Decode a JSON object candidate with strict and reviewer-friendly modes."""
    cleaned = text.strip().replace("\u3000", " ")
    if not cleaned:
        return None

    variants = [cleaned]
    without_trailing_commas = _strip_trailing_json_commas(cleaned)
    if without_trailing_commas != cleaned:
        variants.append(without_trailing_commas)

    for variant in variants:
        for strict in (True, False):
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(variant, strict=strict)
                if isinstance(data, dict):
                    return data

        for strict in (True, False):
            decoder = json.JSONDecoder(strict=strict)
            with contextlib.suppress(json.JSONDecodeError):
                data, _ = decoder.raw_decode(variant)
                if isinstance(data, dict):
                    return data

    return None


def _iter_balanced_json_object_snippets(output: str) -> list[str]:
    """Return brace-balanced object snippets for repair-oriented parsing."""
    snippets: list[str] = []
    for start, char in enumerate(output):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(output)):
            current = output[index]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    snippets.append(output[start : index + 1])
                    break
    return snippets


def _iter_terminal_object_brace_repairs(output: str) -> list[str]:
    """Repair reviewer output that is missing only the final top-level brace."""
    snippets: list[str] = []
    for start, char in enumerate(output):
        if char != "{":
            continue

        stack: list[str] = []
        in_string = False
        escaped = False
        balanced_before_end = False
        invalid = False
        for index in range(start, len(output)):
            current = output[index]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if current == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if current == "{":
                stack.append("}")
            elif current == "[":
                stack.append("]")
            elif current in "}]":
                if not stack or stack[-1] != current:
                    invalid = True
                    break
                stack.pop()
                if not stack:
                    balanced_before_end = True
                    break

        if invalid or balanced_before_end or in_string or stack != ["}"]:
            continue

        candidate = output[start:].strip()
        candidate = re.sub(r"\s*```\s*$", "", candidate).strip()
        snippets.append(candidate + "}")

    return snippets


def _extract_json_object(output: str) -> dict[str, Any]:
    """Extract the reviewer JSON object from model output."""
    candidates: list[dict[str, Any]] = []
    fenced_blocks = re.findall(
        r"```(?:json)?\s*(.*?)```",
        output,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in fenced_blocks:
        data = _decode_json_object_candidate(block)
        if data is not None:
            candidates.append(data)

    for snippet in _iter_balanced_json_object_snippets(output):
        data = _decode_json_object_candidate(snippet)
        if data is not None:
            candidates.append(data)

    for snippet in _iter_terminal_object_brace_repairs(output):
        data = _decode_json_object_candidate(snippet)
        if data is not None:
            candidates.append(data)

    for data in candidates:
        if _looks_like_review_json(data):
            return data
    if candidates:
        return candidates[0]

    raise ValueError("No JSON found in output")


_REVIEW_JSON_FORMAT = """
Output your review as JSON:
{
  "score": <float 1-10>,
  "passed": <boolean>,
  "issues": ["issue1", "issue2"],
  "reasoning": "<brief explanation>"
}
"""

_GENERALIST_REVIEW_JSON_FORMAT = """
Output your review as JSON:
{
  "score": <float 1-10>,
  "passed": <boolean>,
  "blocking_issues": ["issue1", "issue2"],
  "non_blocking_issues": ["issue3", "issue4"],
  "reasoning": "<brief explanation>"
}

Use `passed: true` when the encoding is safe to promote even if there are minor cleanup notes.
Only place substantive statutory-fidelity defects in `blocking_issues`.
Place minor naming cleanups, dead code, test naming nits, or possible-but-uncertain import suggestions in `non_blocking_issues`.
"""

RULESPEC_REVIEWER_PROMPT = (
    """You are an expert Axiom RuleSpec reviewer specializing in structure and legal citations.

Review the RuleSpec file for:
1. **Structure**: Proper definition with `name:` (no `variable`/`parameter` keywords), all required fields (entity, period, dtype, formula)
2. **Legal Citations**: Accurate citation format (e.g., "26 USC 32(a)(1)")
3. **Imports**: Correct import paths using path#name syntax
4. **Entity Hierarchy**: Proper entity usage (Person < TaxUnit < Household)
5. **RuleSpec Compliance**: The file must be valid RuleSpec YAML
6. **Cross-Statute Definitions**: If the source text says a term is defined in another section, import that upstream definition instead of restating it locally
"""
    + _REVIEW_JSON_FORMAT
)

FORMULA_REVIEWER_PROMPT = (
    """You are an expert formula reviewer for Axiom RuleSpec encodings.

Review the RuleSpec formulas for:
1. **Logic Correctness**: Does the formula correctly implement the statute logic?
2. **Edge Cases**: Are edge cases handled (zero values, negative numbers, thresholds)?
3. **Circular Dependencies**: No circular references between definitions
4. **Return Statements**: Every code path returns a value
5. **Type Consistency**: Return type matches declared dtype
6. **Temporal Values**: Uses `versions` with `effective_from: 'yyyy-mm-dd'` for date-based entries
"""
    + _REVIEW_JSON_FORMAT
)

PARAMETER_REVIEWER_PROMPT = (
    """You are an expert reviewer for Axiom RuleSpec encodings, focused on policy values and parameters.

Review the RuleSpec file for policy value usage:
1. **No Magic Numbers**: Only -1, 0, 1, 2, 3 allowed as literals. All other values must be defined as named entries.
2. **No Embedded Scalars**: Legal scalar amounts, thresholds, and limits should be declared as named variables, not embedded inside formulas or conditional branches.
3. **Structured Scales**: Source-stated numeric tables/scales keyed by household size, family size, age band, income band, or similar row keys must use `kind: parameter`, `indexed_by`, and versioned `values`; formulas should reference them with `table_name[index_expr]`, not `match` arms with embedded policy cells.
4. **Sourcing**: Policy values should reference authoritative sources
5. **Time-Varying Values**: Rate thresholds and amounts should use `versions` with `effective_from`
6. **Reference Format**: Correct RuleSpec syntax (`kind: parameter`, no standalone `parameter` keyword block)
7. **Default Values**: Appropriate defaults for optional inputs
"""
    + _REVIEW_JSON_FORMAT
)

INTEGRATION_REVIEWER_PROMPT = (
    """You are an expert integration reviewer for Axiom RuleSpec encodings.

Review the RuleSpec file for integration quality:
1. **Test Coverage**: At least 3-5 test cases in the companion `.test.yaml` file covering normal and edge cases
2. **Dependency Resolution**: All imports can be resolved
3. **Cross-Definition Consistency**: Named definitions work together correctly
4. **Documentation**: Clear labels and descriptions
5. **Completeness**: Full statute implementation, no TODO placeholders
6. **Syntax**: RuleSpec YAML with `format: rulespec/v1`, `rules:`, versioned formulas, and tests in `.test.yaml` files
7. **Cross-Statute Imports**: References like "as defined in section 152(c)" are satisfied by imports from the cited section
"""
    + _REVIEW_JSON_FORMAT
)

GENERALIST_REVIEWER_PROMPT = (
    """You are a senior statutory-fidelity reviewer for Axiom RuleSpec encodings.

Review the file holistically for:
1. **Citation fidelity**: When the file path encodes a legal citation, the file must match it exactly; when review context says the file path is generic benchmark output, use the embedded source text and review context as the citation anchor instead.
2. **Slice fidelity**: When the target is an atomic source slice or branch leaf, judge fidelity to that slice itself. Do not fail solely because sibling limbs, parent consequences, or downstream cross-referenced effects are omitted unless the file claims to encode them.
3. **Whole-rule fidelity**: All operative branches, exceptions, and conditions from the cited text are present for the slice being encoded.
3. **No semantic compression**: Distinct statutory branches or repeated scalar occurrences are not collapsed into a single over-generic helper.
4. **Defined terms and imports**: Explicitly or implicitly legally defined terms are imported from their canonical source when one exists.
5. **Fact modeling**: Factual predicates are modeled as inputs or canonical imports, not hard-coded booleans or deferred placeholders.
6. **Structured parameters**: Source-stated numeric schedules are parameter tables with `indexed_by` and versioned `values`, not derived `match` formulas with embedded policy cells.
7. **Entity / period / dtype plausibility**: Core variables use a coherent ontology for the rule being encoded.
8. **Tests reflect applicability**: Tests cover both applicable and inapplicable branches when the source text makes them meaningful.
9. **Blocking threshold**: Fail only for substantive fidelity defects that would make promotion unsafe. Minor cleanup notes, naming issues, dead code, or arguably missing-but-uncertain imports are non-blocking.
10. **Unsupported ontology fallback**: A file that honestly declares `status: entity_not_supported` is not automatically a blocking failure. If the source slice genuinely depends on an unsupported ontology or granularity, treat that explicit fallback as acceptable so long as the file does not pretend to compute the rule and the unsupported reason is plausible from the source text.
11. **Editorial omission fallback**: If the embedded source text is only an editorial omission or dotted ellipsis with no operative rule content for the target slice, a top-level `status: deferred` fallback is acceptable and should not be failed merely for lacking a computable rule body.
12. **Subject-to qualification placeholders**: When a slice says `Subject to paragraphs ...` and the cited provisions are not available in the workspace, paragraph-specific local inputs can be acceptable for an isolated slice artifact so long as they preserve the cited paragraph numbers and the branch-specific legal effect. Prefer imports when available, but do not fail solely because the file cannot import unavailable cited paragraphs.

Scoring rubric:
- 9-10: strong, promotion-ready
- 7-8: promotion-ready with only non-blocking issues
- 5-6: material concerns remain
- 1-4: unsafe / seriously incorrect
"""
    + _GENERALIST_REVIEW_JSON_FORMAT
)

GROUNDING_ALLOWED_VALUES = {-1, 0, 1, 2, 3}
GROUNDING_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
GROUNDING_MONTH_PERIOD_PATTERN = re.compile(r"\b\d{4}-\d{2}\b")
GROUNDING_FORMULA_NUMBER_PATTERN = re.compile(
    r"(?<![\w./])(-?[\d,]+(?:\.\d+)?)(?![\w./])"
)
SOURCE_TEXT_NUMBER_PATTERN = re.compile(
    r"(?:^|(?<=[\s$£€(\[,]))(-?(?:[\d,]+(?:\.\d+)?|\.\d+))\b"
)
IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")
IMPORT_MAPPING_PATTERN = re.compile(r"^\s*[A-Za-z_]\w*:\s*(['\"]?)([^'\"]+?)\1\s*$")
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
_STRUCTURAL_SOURCE_CITATION_PATTERN = re.compile(
    r"^\d+\s+[A-Z]{2,}(?:\s+\d+[A-Za-z0-9./-]*)+\s*$"
)
_STRUCTURAL_SOURCE_CITATION_PREFIX_PATTERN = re.compile(
    r"^\s*\d+\s+(?:U\.?\s*S\.?\s*C\.?|USC|C\.?\s*F\.?\s*R\.?|CFR|CCR)\s+"
    r"\d+[A-Za-z0-9./-]*(?:\([^)]+\))*\s*",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:\d+(?:\.\d+){2,}\s+|\d+[A-Za-z]?\.\s+|\d+\s+(?=[A-Z][A-Za-z].*:)|\([0-9A-Za-zivxlcdm]+\)\s+)",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_MANUAL_NUMBER_PATTERN = re.compile(
    r"\b(?:(?:Policy|Procedure|Operations)\s+)?"
    r"(?:[A-Z][A-Za-z&/-]*\s+){0,4}?"
    r"Manual(?:\s+Number)?(?:\s*,)?\s+"
    r"(?:\d+(?:\.\d+)+(?:-\d+)?|\d{3,5})\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_MANUAL_VOLUME_PATTERN = re.compile(
    r"\b(?:Vol\.?|Volume)\s+\d+\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_POLICY_LABEL_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z&/-]*\s+){0,4}?"
    r"(?:Policy|Procedure|Chapter)\s+"
    r"\d+(?:\.\d+)+(?:-\d+)?\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_BULLETIN_NUMBER_PATTERN = re.compile(
    r"\bBulletin(?:\s+(?:No\.?|Number))?\s+"
    r"\d+(?:[.-]\d+)+(?:-\d+)?\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_REVISION_PATTERN = re.compile(
    r"\b(?:Rev\.?|Revision)\s+\d{1,2}/\d{4}\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_REVISION_CODE_PATTERN = re.compile(
    r"\b(?:Rev\.?|Revision)\s+\d{1,2}-\d+\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_HANDBOOK_SECTION_PATTERN = re.compile(
    r"\b[A-Z]-\d+(?:\.\d+)+(?:\([A-Za-z0-9]+\))*\b"
)
_STRUCTURAL_SOURCE_FORM_NUMBER_PATTERN = re.compile(
    r"\bForm\s+[A-Z]?\d+[A-Za-z0-9-]*\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_CODE_CITATION_PATTERN = re.compile(
    r"\b\d+\s+"
    r"(?:U\.?\s*S\.?\s*C\.?|USC|C\.?\s*F\.?\s*R\.?|CFR|C\.?\s*C\.?\s*R\.?|CCR)\s+"
    r"\d+(?:[.-]\d+)*(?:\([A-Za-z0-9]+\))*"
    r"(?=$|[\s,.;:])",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_SECTION_PATTERN = re.compile(
    r"\b(?:section|sec\.?)\s+\d+(?:[.-]\d+)*"
    r"(?:"
    r"(?:\([A-Za-z0-9]+\))+"
    r"(?:-(?:\([A-Za-z0-9]+\))+)*)?"
    r"(?=$|[\s,.;:])",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_QUOTE_CHARS = "\"'`“”‘’"
_SYNTHETIC_MODELING_INSTRUCTION_PATTERN = re.compile(
    r"^\s*model\s+`[^`]+`\s+as\b",
    re.IGNORECASE,
)
_SYNTHETIC_STATEWIDE_ALLOWANCE_RESTATEMENT_PATTERN = re.compile(
    r"^\s*For\s+[A-Za-z][A-Za-z .'-]*,\s+the\s+allowance\s+is\s+statewide\s+at\s+\$?\d+(?:\.\d+)?\.?\s*$",
    re.IGNORECASE,
)
_SOURCE_REFERENCE_TARGET_PATTERN = (
    r"(?:(?:\([^)]+\))+|"
    r"\d+[A-Za-z./-]*(?:\([^)]+\))*(?=$|[\s,.;:])"
    r"(?!\s*(?:percent|per\s*cent(?:um)?))|"
    r"[ivxlcdm]+\b|[A-Z]{1,4}\b|[a-z]\b)"
)
_SOURCE_REFERENCE_SEPARATOR_PATTERN = r"(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)"
_SOURCE_REFERENCE_SEQUENCE_PATTERN = (
    rf"{_SOURCE_REFERENCE_TARGET_PATTERN}"
    rf"(?:\s*{_SOURCE_REFERENCE_SEPARATOR_PATTERN}{_SOURCE_REFERENCE_TARGET_PATTERN})*"
)
_SOURCE_REFERENCE_PATTERNS = (
    re.compile(
        r"\b(?:section|sections|paragraph|paragraphs|regulation|regulations|part|parts|chapter|chapters|schedule|schedules|article|articles|subparagraph|subparagraphs|sub-paragraph|sub-paragraphs|subsection|subsections)\s+"
        rf"{_SOURCE_REFERENCE_SEQUENCE_PATTERN}(?:\s+to\s+{_SOURCE_REFERENCE_SEQUENCE_PATTERN})?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:column|columns)\s+{_SOURCE_REFERENCE_SEQUENCE_PATTERN}(?:\s+to\s+{_SOURCE_REFERENCE_SEQUENCE_PATTERN})?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:step|steps)\s+{_SOURCE_REFERENCE_SEQUENCE_PATTERN}(?:\s*,?\s*(?:above|below))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:\d+\s+(?:U\.?\s*S\.?\s*C\.?|USC|C\.?\s*F\.?\s*R\.?|CFR|CCR)\s+)?"
        r"\d+[A-Za-z0-9./-]*(?:\([^)]+\))+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:of\s+)?title\s+\d+[A-Za-z0-9./-]*(?:\([^)]+\))*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[A-Z]{2,6}[ \t]+\d+(?:\.\d+)*(?:\([^)]+\))*",
    ),
    re.compile(r"\b(?:Act|Order|Regulations?)\s+\d{4}\b"),
)
_DIRECT_SCALAR_VALUE_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")
_MONTH_NAME_BODY = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_MONTH_NAME_PATTERN = re.compile(rf"\b{_MONTH_NAME_BODY}(?=$|[\s,.;:])", re.IGNORECASE)
_MONTH_NAME_DATE_PATTERN = re.compile(
    rf"\b{_MONTH_NAME_BODY}\s+\d{{1,2}},\s+\d{{4}}\b",
    re.IGNORECASE,
)
_SLASH_DATE_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_MONTH_NAME_DAY_PATTERN = re.compile(
    rf"\b{_MONTH_NAME_BODY}\s+\d{{1,2}}\b",
    re.IGNORECASE,
)
_MONTH_DAY_OF_MONTH_PATTERN = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)\s+day\s+of\s+(?:a|the)\s+month\b",
    re.IGNORECASE,
)
_STRUCTURAL_SOURCE_SUBDIVISION_MARKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])\((?:[A-Za-z]|[ivxlcdmIVXLCDM]+|\d{1,2})\)"
)
_TABLE_HEADING_PATTERN = re.compile(
    r"^\s*table\s+\d+[A-Za-z]?(?:\s*:.*)?$", re.IGNORECASE
)
_ORDINAL_NUMBER_PATTERN = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_SCHEDULE_BLOCK_HEADING_PATTERN = re.compile(r"^[A-Z][A-Z0-9_ ]+:\s*$")
_SCHEDULE_SIZE_ROW_PATTERN = re.compile(
    r"^\s*[-*]?\s*(?:size|household size|unit size)(?:\s+\d+(?:\s+or\s+more)?)?\s*:\s*"
    r"(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_SCHEDULE_PIPE_ROW_PATTERN = re.compile(
    r"^\s*[-*]?\s*(?:\d+(?:\s+or\s+more)?|size\s+\d+(?:\s+or\s+more)?|"
    r"household size\s+\d+(?:\s+or\s+more)?|unit size\s+\d+(?:\s+or\s+more)?)\s*\|\s*"
    r"(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_SCHEDULE_ARROW_ROW_PATTERN = re.compile(
    r"^\s*[-*]?\s*(?:\d+(?:\s+or\s+more)?|size\s+\d+(?:\s+or\s+more)?|"
    r"household size\s+\d+(?:\s+or\s+more)?|unit size\s+\d+(?:\s+or\s+more)?)\s*"
    r"(?:=>|->|=)\s*(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_SCHEDULE_BARE_ARROW_ROW_PATTERN = re.compile(
    r"^\s*[-*]\s*(?:=>|->|=)\s*(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_VALUE_BEARING_TABLE_ROW_PATTERN = re.compile(
    r"^\s*[-*]?\s*[^:]+:\s*(?:[$£€]\s*)?(-?[\d,]+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)
_SUBPOUND_MONEY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:pence|penny)\b", re.IGNORECASE
)
_TABLE_KEY_ASSIGNMENT_PATTERN = re.compile(r"\b\d+(?=\s*=)")
_TABLE_ROW_LABEL_PATTERN = re.compile(
    r"\b(?:size|household size|unit size)\s+\d+(?:\s+or\s+more)?(?=\s*:)",
    re.IGNORECASE,
)
_SCHEDULE_SIZE_CAP_RESTATEMENT_PATTERN = re.compile(
    r"\babove\s+(\d+)\s+use(?:s)?\s+the\s+rate\s+for\s+(?:a|an)\s+\1(?:\s+member)?\s+household\b",
    re.IGNORECASE,
)
_SCHEDULE_INDEX_NAME_PATTERN = r"[A-Za-z_]\w*_size(?:_[A-Za-z_]\w*)*"
_STRUCTURAL_ENUM_INDEX_NAME_PATTERN = r"(?:filing_status|tax_filing_status)"
_CARDINAL_WORD_VALUES = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
}
_CARDINAL_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _CARDINAL_WORD_VALUES) + r")\b",
    re.IGNORECASE,
)
_CARDINAL_VALUE_WORDS = {
    int(value): word
    for word, value in _CARDINAL_WORD_VALUES.items()
    if float(value).is_integer()
}
_DATE_DECOMPOSITION_CUE_TOKENS = {
    "date",
    "birthday",
    "anniversary",
    "cutoff",
    "effective",
    "calendar",
    "commencement",
    "start",
    "end",
}


@dataclass(frozen=True)
class NamedScalarOccurrence:
    """One direct named scalar definition found in a RuleSpec file."""

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
    r"section\s+(?P<section>[0-9A-Za-z.-]+(?:\([^)]+\))*)"
    r"(?:\s+of\s+title\s+(?P<title>[0-9A-Za-z.-]+))?",
    re.IGNORECASE,
)


def _load_nearby_eval_source_metadata(rulespec_file: Path) -> dict[str, object] | None:
    """Load source-metadata from a nearby eval workspace when present."""
    for ancestor in rulespec_file.parents:
        eval_root = ancestor / "_eval_workspaces"
        if not eval_root.exists():
            continue
        for manifest_path in sorted(eval_root.glob("**/context-manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text())
            except Exception:
                continue
            metadata = payload.get("source_metadata")
            if isinstance(metadata, dict):
                return metadata
    return None


def _source_metadata_sets_target_symbol(
    source_metadata: dict[str, object] | None, symbol_name: str
) -> bool:
    """Return whether source metadata declares a `sets` relation for the symbol."""
    if not isinstance(source_metadata, dict):
        return False
    relations = source_metadata.get("relations")
    if not isinstance(relations, list):
        return False

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if str(relation.get("relation", "")).lower() != "sets":
            continue
        target = str(relation.get("target", ""))
        _, _, target_symbol = target.partition("#")
        if target_symbol == symbol_name:
            return True
    return False


def _source_metadata_jurisdiction(
    source_metadata: dict[str, object] | None,
) -> str | None:
    """Return a jurisdiction code from source metadata when present."""
    if not isinstance(source_metadata, dict):
        return None
    relations = source_metadata.get("relations")
    if not isinstance(relations, list):
        return None

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        jurisdiction = relation.get("jurisdiction")
        if jurisdiction is None:
            continue
        jurisdiction_str = str(jurisdiction).strip()
        if jurisdiction_str:
            return jurisdiction_str
    return None


def _infer_us_state_code_from_rulespec_path(
    rulespec_file: Path,
    rulespec_source_content: str = "",
) -> str | None:
    """Infer a US state code from canonical RuleSpec repo paths or legal ids."""
    path_text = rulespec_file.as_posix().lower()
    match = re.search(r"(?:^|/)rulespec-us-([a-z]{2})(?:/|$)", path_text)
    if match:
        return match.group(1).upper()

    source_text = rulespec_source_content.lower()
    match = re.search(r"\bus-([a-z]{2}):", source_text)
    if match:
        return match.group(1).upper()
    return None


def _default_snap_utility_type_for_rule(rule_name: str | None) -> str | None:
    return {
        "snap_standard_utility_allowance": "SUA",
        "snap_limited_utility_allowance": "BUA",
        "snap_individual_utility_allowance": "TUA",
    }.get(str(rule_name or ""))


def _default_snap_utility_region_for_jurisdiction(
    jurisdiction: str | None,
) -> str | None:
    if not jurisdiction:
        return None
    normalized = jurisdiction.strip().upper()
    if normalized == "NY":
        return "NY_NYC"
    return normalized


def _policyengine_us_snap_input_aliases(inputs: dict[str, Any]) -> dict[str, Any]:
    """Derive standard SNAP PE inputs from source-document input names."""

    def input_value(key: str) -> Any | None:
        if key in inputs:
            return inputs[key]
        for input_key, value in inputs.items():
            key_text = str(input_key)
            if (
                key_text.endswith(f"#input.{key}")
                or key_text.endswith(f"#{key}")
                or key_text.endswith(f".{key}")
            ):
                return value
        return None

    def nested_input_values(key: str) -> list[Any]:
        matches: list[Any] = []
        stack = list(inputs.values())
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    nested_key_text = str(nested_key)
                    if (
                        nested_key_text == key
                        or nested_key_text.endswith(f"#input.{key}")
                        or nested_key_text.endswith(f"#{key}")
                        or nested_key_text.endswith(f".{key}")
                    ):
                        matches.append(nested_value)
                    elif isinstance(nested_value, (dict, list)):
                        stack.append(nested_value)
            elif isinstance(value, list):
                stack.extend(value)
        return matches

    def numeric_value(key: str) -> float | None:
        value = input_value(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    aliases: dict[str, Any] = {}
    earned = numeric_value("employee_wages_received")
    if earned is not None:
        aliases.setdefault("snap_earned_income", earned)
        aliases.setdefault("snap_gross_income", earned)

    countable_earned = numeric_value("snap_countable_earned_income")
    countable_unearned = numeric_value("snap_countable_unearned_income")
    work_supplementation = numeric_value("work_supplementation_earned_income")
    if countable_earned is not None:
        aliases.setdefault(
            "snap_earned_income",
            max(0.0, countable_earned - (work_supplementation or 0.0)),
        )
    if countable_unearned is not None:
        aliases.setdefault("snap_unearned_income", countable_unearned)
    if countable_earned is not None or countable_unearned is not None:
        aliases.setdefault(
            "snap_gross_income",
            (countable_earned or 0.0) + (countable_unearned or 0.0),
        )

    monthly_household_income = numeric_value("snap_monthly_household_income")
    if monthly_household_income is not None:
        aliases.setdefault("snap_gross_income", monthly_household_income)

    standard_deduction = numeric_value("snap_standard_deduction")
    if standard_deduction is not None:
        aliases.setdefault("snap_standard_deduction", standard_deduction)

    elderly_or_disabled_member = input_value("snap_member_is_elderly_or_disabled")
    if elderly_or_disabled_member is None:
        nested_elderly_or_disabled = nested_input_values(
            "snap_member_is_elderly_or_disabled"
        )
        if nested_elderly_or_disabled:
            elderly_or_disabled_member = any(
                bool(item) for item in nested_elderly_or_disabled
            )
    if elderly_or_disabled_member is not None:
        aliases.setdefault(
            "snap_household_has_elderly_or_disabled_member",
            bool(elderly_or_disabled_member),
        )
        aliases.setdefault(
            "has_usda_elderly_disabled", bool(elderly_or_disabled_member)
        )

    shelter_cost = numeric_value("household_shelter_costs_incurred")
    if shelter_cost is None:
        shelter_cost = numeric_value("snap_allowable_shelter_costs")
    if shelter_cost is not None:
        aliases.setdefault("housing_cost", shelter_cost)
        aliases.setdefault("snap_utility_allowance_type", "NONE")

    deduction_aliases = {
        "dependent_care_deduction": "snap_dependent_care_deduction",
        "child_support_deduction": "snap_child_support_deduction",
        "medical_deduction": "snap_excess_medical_expense_deduction",
        "excess_shelter_deduction": "snap_excess_shelter_expense_deduction",
    }
    for source_key, alias_key in deduction_aliases.items():
        deduction_value = numeric_value(source_key)
        if deduction_value is not None:
            aliases.setdefault(alias_key, deduction_value)

    heating_or_cooling = bool(
        input_value(
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage"
        )
    )
    if heating_or_cooling:
        aliases["snap_utility_allowance_type"] = "SUA"
    elif any(
        str(key).startswith("household_pays_") or "#input.household_pays_" in str(key)
        for key in inputs
    ):
        non_heating_cooling_utility_keys = (
            "household_pays_electricity_utility_cost",
            "household_pays_water_utility_cost",
            "household_pays_sewer_utility_cost",
            "household_pays_trash_utility_cost",
            "household_pays_cooking_fuel_utility_cost",
        )
        non_heating_cooling_count = sum(
            1 for key in non_heating_cooling_utility_keys if input_value(key)
        )
        aliases.setdefault(
            "snap_utility_allowance_type",
            "LUA" if non_heating_cooling_count >= 2 else "NONE",
        )

    return {key: value for key, value in aliases.items() if key not in inputs}


def _policyengine_expected_float(expected: Any) -> float:
    """Normalize RuleSpec expected values for numeric PE result comparison."""
    if isinstance(expected, str):
        normalized = expected.strip().lower()
        if normalized == "holds":
            return 1.0
        if normalized == "not_holds":
            return 0.0
    return float(expected)


def _normalize_us_tax_filing_status(value: Any) -> str:
    """Normalize Axiom/RuleSpec filing-status test inputs to PE-US enum keys."""
    numeric_statuses = {
        0: "SINGLE",
        1: "JOINT",
        2: "SEPARATE",
        3: "HEAD_OF_HOUSEHOLD",
        4: "SURVIVING_SPOUSE",
    }
    if isinstance(value, int) and not isinstance(value, bool):
        return numeric_statuses.get(value, "SINGLE")
    value_text = str(value or "SINGLE").strip().upper()
    if value_text.isdigit():
        return numeric_statuses.get(int(value_text), "SINGLE")
    aliases = {
        "MARRIED_FILING_JOINTLY": "JOINT",
        "MARRIED JOINT": "JOINT",
        "MARRIED_FILING_SEPARATELY": "SEPARATE",
        "MARRIED SEPARATE": "SEPARATE",
        "HEAD OF HOUSEHOLD": "HEAD_OF_HOUSEHOLD",
        "HOH": "HEAD_OF_HOUSEHOLD",
        "QUALIFYING_SURVIVING_SPOUSE": "SURVIVING_SPOUSE",
    }
    return aliases.get(value_text, value_text if value_text else "SINGLE")


def _tax_unit_member_aged_flags(inputs: dict[str, Any]) -> list[bool]:
    """Return explicit aged flags from a RuleSpec member_of_tax_unit input."""
    members = inputs.get("member_of_tax_unit")
    if isinstance(members, bool):
        return [members]
    if not isinstance(members, list):
        members = []
        for key, value in inputs.items():
            key_text = str(key).lower()
            if (
                "#relation." in key_text
                and key_text.rsplit("#relation.", 1)[1].endswith(
                    "member_of_tax_unit"
                )
                and isinstance(value, list)
            ):
                members = value
                break
    if not isinstance(members, list):
        return []
    aged_flags: list[bool] = []
    for member in members:
        if isinstance(member, bool):
            aged_flags.append(member)
            continue
        if not isinstance(member, dict):
            continue
        explicit_aged = ValidatorPipeline._rulespec_test_input_value(
            member, "is_aged_65_or_over"
        )
        if explicit_aged is not None:
            aged_flags.append(bool(explicit_aged))
            continue
        age = ValidatorPipeline._rulespec_test_input_value(member, "age")
        with contextlib.suppress(TypeError, ValueError):
            aged_flags.append(int(age) >= 65)
            continue
        aged_flags.append(False)
    return aged_flags


def _policyengine_period_string(value: Any, fallback: str = "2024-01") -> str:
    """Normalize RuleSpec test period values to a PE scenario period string."""
    if isinstance(value, dict):
        start = value.get("start") or value.get("date")
        if start:
            start_text = str(start)
            if value.get("period_kind") == "month" and len(start_text) >= 7:
                return start_text[:7]
            if len(start_text) >= 4:
                return start_text[:4]
        return fallback
    if value is None:
        return fallback
    value_text = str(value)
    if not value_text:
        return fallback
    return value_text


def extract_grounding_values(content: str) -> list[tuple[int, str, float]]:
    """Extract grounded numeric values from RuleSpec definitions."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("rules"), list)
        ):
            values: list[tuple[int, str, float]] = []
            for rule in payload["rules"]:
                if not isinstance(rule, dict):
                    continue
                versions = rule.get("versions")
                if not isinstance(versions, list):
                    continue
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    formula = version.get("formula")
                    if isinstance(formula, (int, float)) and not isinstance(
                        formula, bool
                    ):
                        value = float(formula)
                        if value not in GROUNDING_ALLOWED_VALUES:
                            values.append((1, str(formula), value))
                    elif isinstance(formula, str):
                        values.extend(_extract_formula_grounding_values(1, formula))
                    table_values = version.get("values")
                    if isinstance(table_values, dict):
                        for table_value in table_values.values():
                            extracted = _numeric_rule_value(table_value)
                            if extracted is None:
                                continue
                            raw, value = extracted
                            if value not in GROUNDING_ALLOWED_VALUES:
                                values.append((1, raw, value))
            return values

    return []


def _numeric_rule_value(value: Any) -> tuple[str, float] | None:
    """Return a display string and numeric value for a YAML scalar."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value), float(value)
    if isinstance(value, str) and _DIRECT_SCALAR_VALUE_PATTERN.fullmatch(value.strip()):
        raw = value.strip()
        return raw, float(raw.replace(",", ""))
    return None


def _extract_formula_grounding_values(
    line_number: int, formula_text: str
) -> list[tuple[int, str, float]]:
    """Extract numeric literals from a formula expression or formula line."""
    cleaned = formula_text.split("#", 1)[0]
    cleaned = GROUNDING_DATE_PATTERN.sub(" ", cleaned)

    values: list[tuple[int, str, float]] = []
    for match in GROUNDING_FORMULA_NUMBER_PATTERN.finditer(cleaned):
        raw = match.group(1).replace(",", "")
        if raw == "0.5" and _is_half_up_rounding_expression(cleaned):
            continue
        if _is_structural_table_key_literal(cleaned, raw):
            continue
        if _is_structural_schedule_index_literal(cleaned, raw):
            continue
        if _is_structural_enum_index_literal(cleaned, raw):
            continue
        with contextlib.suppress(ValueError):
            value = float(raw)
            if value not in GROUNDING_ALLOWED_VALUES:
                values.append((line_number, raw, value))
    return values


def _is_half_up_rounding_expression(expression: str) -> bool:
    """Return True when an expression uses the standard half-up rounding offset."""
    compact = re.sub(r"\s+", "", expression)
    return _call_body_contains_any(compact, "floor", ("+0.5", "0.5+")) or (
        _call_body_contains_any(compact, "ceil", ("-0.5", "0.5-"))
    )


def _is_half_up_rounding_helper_scalar(symbol_name: str, value: float) -> bool:
    """Return True when a named scalar only defines the standard half-up offset."""
    if value != 0.5:
        return False
    normalized = symbol_name.lower()
    if "half_increment" in normalized:
        return True
    return "half_up" in normalized and (
        "rounding" in normalized or "offset" in normalized
    )


def _is_structural_table_key_literal(expression: str, literal: str) -> bool:
    """Return true when a small integer only selects a parameter-table row."""
    if literal in _EMBEDDED_SCALAR_ALLOWED_VALUES:
        return False
    if not re.fullmatch(r"\d+(?:\.0+)?", literal):
        return False
    with contextlib.suppress(ValueError):
        numeric_value = float(literal)
        if not numeric_value.is_integer() or not (0 <= int(numeric_value) <= 20):
            return False
    return bool(
        re.search(
            rf"\[[ \t]*{re.escape(literal)}[ \t]*\]",
            expression,
        )
    )


def _is_structural_schedule_index_literal(expression: str, literal: str) -> bool:
    """Return True when a small integer only serves as a schedule index."""
    if literal in _EMBEDDED_SCALAR_ALLOWED_VALUES:
        return False
    if not re.fullmatch(r"\d+(?:\.0+)?", literal):
        return False
    with contextlib.suppress(ValueError):
        numeric_value = float(literal)
        if not numeric_value.is_integer() or not (4 <= int(numeric_value) <= 8):
            return False
    if not re.search(rf"\b{_SCHEDULE_INDEX_NAME_PATTERN}\b", expression):
        return False

    normalized = re.sub(r"\s+", " ", expression)
    comparison_pattern = re.compile(
        rf"\b{_SCHEDULE_INDEX_NAME_PATTERN}\s*(?:==|>=|>|<=|<)\s*{re.escape(literal)}\b"
    )
    delta_pattern = re.compile(
        rf"(?:\(\s*)?{_SCHEDULE_INDEX_NAME_PATTERN}\s*-\s*{re.escape(literal)}(?:\s*\))?"
    )
    match_arm_pattern = re.compile(rf"\b{re.escape(literal)}\s*=>")
    return bool(
        comparison_pattern.search(normalized)
        or delta_pattern.search(normalized)
        or (
            re.search(
                rf"\bmatch\s+{_SCHEDULE_INDEX_NAME_PATTERN}\s*:",
                normalized,
            )
            and match_arm_pattern.search(normalized)
        )
    )


def _is_structural_enum_index_literal(expression: str, literal: str) -> bool:
    """Return True when a small integer only serves as an internal enum code."""
    if literal in _EMBEDDED_SCALAR_ALLOWED_VALUES:
        return False
    if not re.fullmatch(r"\d+(?:\.0+)?", literal):
        return False
    with contextlib.suppress(ValueError):
        numeric_value = float(literal)
        if not numeric_value.is_integer() or not (0 <= int(numeric_value) <= 20):
            return False
    if not re.search(rf"\b{_STRUCTURAL_ENUM_INDEX_NAME_PATTERN}\b", expression):
        return False

    normalized = re.sub(r"\s+", " ", expression)
    comparison_pattern = re.compile(
        rf"\b{_STRUCTURAL_ENUM_INDEX_NAME_PATTERN}\s*(?:==|!=|>=|>|<=|<)\s*{re.escape(literal)}\b"
    )
    match_arm_pattern = re.compile(rf"\b{re.escape(literal)}\s*=>")
    return bool(
        comparison_pattern.search(normalized)
        or (
            re.search(
                rf"\bmatch\s+{_STRUCTURAL_ENUM_INDEX_NAME_PATTERN}\s*:",
                normalized,
            )
            and match_arm_pattern.search(normalized)
        )
    )


def _is_structural_schedule_index_helper(name: str, value: float) -> bool:
    """Return True when a scalar helper only labels a schedule row index."""
    if not value.is_integer() or not (4 <= int(value) <= 8):
        return False
    normalized_name = name.lower()
    if "or_more" in normalized_name:
        return False
    if "threshold" in normalized_name and int(value) >= 5:
        return False
    index = int(value)
    word = {
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
    }.get(index)
    return bool(
        re.search(rf"(?:^|_)size_{index}(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_)household_size_{index}(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_)unit_size_{index}(?:_|$)", normalized_name)
        or re.search(
            rf"(?:^|_){word}_person_(?:household_)?size(?:_|$)", normalized_name
        )
        or re.search(rf"(?:^|_){word}_person_unit_size(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_){word}_person_spm_unit_size(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_)size_row_{index}(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_)household_size_row_{index}(?:_|$)", normalized_name)
        or re.search(rf"(?:^|_)unit_size_row_{index}(?:_|$)", normalized_name)
        or (
            word is not None
            and re.search(rf"(?:^|_)size_{word}(?:_|$)", normalized_name)
        )
        or (
            word is not None
            and re.search(rf"(?:^|_)household_size_{word}(?:_|$)", normalized_name)
        )
        or (
            word is not None
            and re.search(rf"(?:^|_)unit_size_{word}(?:_|$)", normalized_name)
        )
    )


def _call_body_contains_any(
    compact_expression: str,
    function_name: str,
    needles: tuple[str, ...],
) -> bool:
    token = f"{function_name}("
    search_start = 0
    while True:
        call_start = compact_expression.find(token, search_start)
        if call_start == -1:
            return False

        index = call_start + len(token)
        depth = 1
        body_chars: list[str] = []
        while index < len(compact_expression) and depth > 0:
            char = compact_expression[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    break
            body_chars.append(char)
            index += 1

        if any(needle in "".join(body_chars) for needle in needles):
            return True

        search_start = call_start + 1


def extract_numbers_from_text(text: str) -> set[float]:
    """Extract numeric values from embedded statute text."""
    original_text = text
    text = _clean_source_text_for_numeric_extraction(text)
    schedule_occurrences, text = _extract_collapsed_schedule_row_occurrences(text)
    numbers = set()
    occupied_spans: list[tuple[int, int]] = []
    numbers.update(schedule_occurrences)

    for match in re.finditer(
        r"\b(?:age|aged)\s+(\d{1,3})(?=\b)", original_text, re.IGNORECASE
    ):
        with contextlib.suppress(ValueError):
            numbers.add(float(match.group(1)))

    for span, value in _iter_normalized_special_numeric_matches(text):
        numbers.add(value)
        occupied_spans.append(span)
    numbers.update(_extract_percentage_context_values(text))

    for match in SOURCE_TEXT_NUMBER_PATTERN.finditer(text):
        if _span_overlaps(match.span(1), occupied_spans):
            continue
        raw = match.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            numbers.add(float(raw))

    for match in _ORDINAL_NUMBER_PATTERN.finditer(text):
        with contextlib.suppress(ValueError):
            numbers.add(float(match.group(1)))

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

    for match in _CARDINAL_WORD_PATTERN.finditer(text_lower):
        numbers.add(_CARDINAL_WORD_VALUES[match.group(1)])

    return numbers


def _ordinal_is_calendar_day_reference(text: str, end_index: int, value: float) -> bool:
    """Return True when an ordinal is functioning as a calendar day before a month name."""
    if not value.is_integer() or not (1 <= value <= 31):
        return False
    trailing = text[end_index:]
    return bool(re.match(rf"\s+{_MONTH_NAME_PATTERN.pattern}", trailing, re.IGNORECASE))


def _iter_normalized_special_numeric_matches(
    text: str,
) -> list[tuple[tuple[int, int], float]]:
    """Return normalized special-case numeric matches like percentages, pence, and table values."""
    matches: list[tuple[tuple[int, int], float]] = []

    for pattern in (
        re.compile(
            r"(\d+(?:\.\d+)?)(?:\s+|-)(?:percent|per\s*cent(?:um)?)",
            re.IGNORECASE,
        ),
        re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    ):
        for match in pattern.finditer(text):
            with contextlib.suppress(ValueError):
                matches.append(
                    (match.span(), float(match.group(1).replace(",", "")) / 100)
                )

    for match in _SUBPOUND_MONEY_PATTERN.finditer(text):
        with contextlib.suppress(ValueError):
            matches.append((match.span(), float(match.group(1).replace(",", "")) / 100))

    for match in re.finditer(r"(?<=[=+])\s*(-?[\d,]+(?:\.\d+)?)\b", text):
        with contextlib.suppress(ValueError):
            matches.append((match.span(1), float(match.group(1).replace(",", ""))))

    return matches


def _extract_percentage_context_values(text: str) -> set[float]:
    """Return decimal rate equivalents for numbers in percentage table contexts."""
    values: set[float] = set()
    for match in SOURCE_TEXT_NUMBER_PATTERN.finditer(text):
        raw = match.group(1).replace(",", "")
        with contextlib.suppress(ValueError):
            value = float(raw)
            if not (1 < value <= 100):
                continue
            context = text[max(0, match.start() - 250) : match.end() + 80].lower()
            if re.search(r"\bpercent(?:age|ages)?\b", context):
                values.add(value / 100)
    return values


def _span_overlaps(
    span: tuple[int, int], occupied_spans: list[tuple[int, int]]
) -> bool:
    return any(
        not (span[1] <= start or span[0] >= end) for start, end in occupied_spans
    )


def _clean_source_text_for_numeric_extraction(text: str) -> str:
    """Strip structural source scaffolding before numeric extraction."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        structural_stripped = stripped.strip(_STRUCTURAL_SOURCE_QUOTE_CHARS)
        if _STRUCTURAL_SOURCE_LINE_PATTERN.match(structural_stripped):
            continue
        if _STRUCTURAL_SOURCE_HEADING_PATTERN.match(structural_stripped):
            continue
        if _STRUCTURAL_SOURCE_CITATION_PATTERN.match(structural_stripped):
            continue
        if _TABLE_HEADING_PATTERN.match(structural_stripped):
            continue
        if _SYNTHETIC_MODELING_INSTRUCTION_PATTERN.match(structural_stripped):
            continue
        if _SYNTHETIC_STATEWIDE_ALLOWANCE_RESTATEMENT_PATTERN.match(
            structural_stripped
        ):
            continue

        normalized_line = line.lstrip(_STRUCTURAL_SOURCE_QUOTE_CHARS)
        normalized_line = _STRUCTURAL_SOURCE_CITATION_PREFIX_PATTERN.sub(
            "", normalized_line, count=1
        )
        value_row_match = _VALUE_BEARING_TABLE_ROW_PATTERN.match(normalized_line)
        schedule_row_match = (
            _SCHEDULE_SIZE_ROW_PATTERN.fullmatch(normalized_line)
            or _SCHEDULE_PIPE_ROW_PATTERN.fullmatch(normalized_line)
            or _SCHEDULE_ARROW_ROW_PATTERN.fullmatch(normalized_line)
            or _SCHEDULE_BARE_ARROW_ROW_PATTERN.fullmatch(normalized_line)
        )
        if value_row_match and not schedule_row_match:
            normalized_line = value_row_match.group(1)
        normalized_line = _TABLE_ROW_LABEL_PATTERN.sub("size", normalized_line)
        cleaned_lines.append(_STRUCTURAL_SOURCE_PREFIX_PATTERN.sub("", normalized_line))

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\[[^\]]*\d[^\]]*\]", " ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_MANUAL_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_MANUAL_VOLUME_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_POLICY_LABEL_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_BULLETIN_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_REVISION_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_REVISION_CODE_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_HANDBOOK_SECTION_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_FORM_NUMBER_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_CODE_CITATION_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_SECTION_PATTERN.sub(" ", cleaned)
    cleaned = GROUNDING_DATE_PATTERN.sub(" ", cleaned)
    cleaned = GROUNDING_MONTH_PERIOD_PATTERN.sub(" ", cleaned)
    cleaned = _MONTH_NAME_DATE_PATTERN.sub(" ", cleaned)
    cleaned = _SLASH_DATE_PATTERN.sub(" ", cleaned)
    cleaned = _MONTH_NAME_DAY_PATTERN.sub(" ", cleaned)
    cleaned = _MONTH_DAY_OF_MONTH_PATTERN.sub(" ", cleaned)
    cleaned = _STRUCTURAL_SOURCE_SUBDIVISION_MARKER_PATTERN.sub(" ", cleaned)
    cleaned = _SCHEDULE_SIZE_CAP_RESTATEMENT_PATTERN.sub(
        lambda match: f"above {match.group(1)} use the capped household rate",
        cleaned,
    )
    cleaned = _TABLE_KEY_ASSIGNMENT_PATTERN.sub(" ", cleaned)
    for pattern in _SOURCE_REFERENCE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _extract_collapsed_schedule_row_occurrences(
    text: str,
) -> tuple[list[float], str]:
    """Extract schedule row values once per contiguous value block and remove row lines."""
    occurrences: list[float] = []
    retained_lines: list[str] = []
    current_heading: str | None = None
    last_value_by_block: dict[str, float] = {}
    seen_values: set[float] = set()
    ungrouped_block = 0
    current_ungrouped_block: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if _SCHEDULE_BLOCK_HEADING_PATTERN.fullmatch(stripped):
            current_heading = stripped
            current_ungrouped_block = None
            retained_lines.append(line)
            continue

        row_match = (
            _SCHEDULE_SIZE_ROW_PATTERN.fullmatch(stripped)
            or _SCHEDULE_PIPE_ROW_PATTERN.fullmatch(stripped)
            or _SCHEDULE_ARROW_ROW_PATTERN.fullmatch(stripped)
            or _SCHEDULE_BARE_ARROW_ROW_PATTERN.fullmatch(stripped)
        )
        if row_match:
            with contextlib.suppress(ValueError):
                value = float(row_match.group(1).replace(",", ""))
                if current_heading is not None:
                    block_key = current_heading
                else:
                    if current_ungrouped_block is None:
                        ungrouped_block += 1
                        current_ungrouped_block = f"__ungrouped_{ungrouped_block}"
                    block_key = current_ungrouped_block
                if (
                    last_value_by_block.get(block_key) != value
                    and value not in seen_values
                ):
                    occurrences.append(value)
                    last_value_by_block[block_key] = value
                    seen_values.add(value)
            continue

        if stripped:
            current_heading = None
            current_ungrouped_block = None
        retained_lines.append(line)

    return occurrences, "\n".join(retained_lines)


def extract_numeric_occurrences_from_text(text: str) -> list[float]:
    """Extract substantive numeric occurrences from source text, preserving repeats."""
    cleaned = _clean_source_text_for_numeric_extraction(text)
    collapsed_schedule_occurrences, cleaned = (
        _extract_collapsed_schedule_row_occurrences(cleaned)
    )

    occurrences: list[float] = list(collapsed_schedule_occurrences)
    spans: list[tuple[int, int]] = []

    for span, value in _iter_normalized_special_numeric_matches(cleaned):
        occurrences.append(value)
        spans.append(span)

    for match in SOURCE_TEXT_NUMBER_PATTERN.finditer(cleaned):
        span = match.span(1)
        if _span_overlaps(span, spans):
            continue
        with contextlib.suppress(ValueError):
            value = float(match.group(1).replace(",", ""))
            if value.is_integer() and 1900 <= value <= 2100:
                continue
            occurrences.append(value)

    for match in _ORDINAL_NUMBER_PATTERN.finditer(cleaned):
        with contextlib.suppress(ValueError):
            value = float(match.group(1))
            if value.is_integer() and 1900 <= value <= 2100:
                continue
            if _ordinal_is_calendar_day_reference(cleaned, match.end(), value):
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
    """Extract direct named scalar definitions from a RuleSpec file."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if (
            isinstance(payload, dict)
            and payload.get("format") == "rulespec/v1"
            and isinstance(payload.get("rules"), list)
        ):
            occurrences: list[NamedScalarOccurrence] = []
            for rule in payload["rules"]:
                if not isinstance(rule, dict):
                    continue
                name = str(rule.get("name") or "").strip()
                if not name:
                    continue
                versions = rule.get("versions")
                if not isinstance(versions, list):
                    continue
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    formula = version.get("formula")
                    raw: str | None = None
                    if isinstance(formula, (int, float)) and not isinstance(
                        formula, bool
                    ):
                        raw = str(formula)
                    elif isinstance(formula, str):
                        stripped = formula.strip()
                        if _DIRECT_SCALAR_VALUE_PATTERN.fullmatch(stripped):
                            raw = stripped
                    if raw is not None:
                        with contextlib.suppress(ValueError):
                            occurrences.append(
                                NamedScalarOccurrence(
                                    line=1,
                                    name=name,
                                    value=float(raw.replace(",", "")),
                                )
                            )
                    table_values = version.get("values")
                    if isinstance(table_values, dict):
                        for key, table_value in table_values.items():
                            extracted = _numeric_rule_value(table_value)
                            if extracted is None:
                                continue
                            _, value = extracted
                            occurrences.append(
                                NamedScalarOccurrence(
                                    line=1,
                                    name=f"{name}[{key}]",
                                    value=value,
                                )
                            )
            return occurrences

    return []


def extract_embedded_source_text(content: str) -> str:
    """Extract embedded source text from RuleSpec YAML."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if isinstance(payload, dict) and payload.get("format") == "rulespec/v1":
            module = payload.get("module")
            if isinstance(module, dict):
                summary = module.get("summary")
                if isinstance(summary, str) and summary.strip():
                    return summary.strip()

    return ""


def find_ungrounded_numeric_issues(
    content: str,
    source_text: str | None = None,
) -> list[str]:
    """Return issues for generated numeric literals absent from source text."""
    grounding_values = extract_grounding_values(content)
    if not grounding_values:
        return []

    if source_text is not None:
        source = source_text.strip()
    else:
        source = (extract_numeric_grounding_source_text(content) or "").strip()
    if not source:
        return [
            "Numeric source required: RuleSpec defines policy numeric literals "
            "but does not provide `source_verification.corpus_citation_path` "
            "or `source_verification.corpus_citation_paths` text. "
            "`module.summary` is not accepted as source text for numeric grounding."
        ]

    source_numbers = extract_numbers_from_text(source)
    issues: list[str] = []
    for _, raw, value in grounding_values:
        if numeric_value_is_grounded(value, source_numbers):
            continue
        display = raw if raw == f"{value:g}" else f"{raw} ({value:g})"
        issues.append(
            "Ungrounded generated numeric literal: "
            f"{display} does not appear as a substantive numeric value in the source text."
        )
    return issues


def extract_numeric_grounding_source_text(content: str) -> str | None:
    """Return authoritative source text usable for numeric grounding.

    Numeric grounding must use ingested corpus source text. The human-readable
    module summary is intentionally excluded.
    """
    return _extract_source_verification_text(content)


def find_deprecated_source_url_issues(content: str) -> list[str]:
    """Reject raw source URL references in RuleSpec encodings."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []

    locations: list[str] = []
    module = payload.get("module")
    if isinstance(module, dict):
        if "source_url" in module:
            locations.append("module.source_url")
        source_verification = module.get("source_verification")
        if (
            isinstance(source_verification, dict)
            and "source_url" in source_verification
        ):
            locations.append("module.source_verification.source_url")

    source_verification = payload.get("source_verification")
    if isinstance(source_verification, dict) and "source_url" in source_verification:
        locations.append("source_verification.source_url")

    rules = payload.get("rules")
    if isinstance(rules, list):
        for index, rule in enumerate(rules):
            if isinstance(rule, dict) and "source_url" in rule:
                name = rule.get("name") or f"rules[{index}]"
                locations.append(f"rules.{name}.source_url")

    if not locations:
        return []
    return [
        "Legacy source URL metadata not allowed: "
        + ", ".join(locations[:5])
        + ("; ..." if len(locations) > 5 else "")
        + ". Use `module.source_verification.corpus_citation_path` or "
        "`module.source_verification.corpus_citation_paths`."
    ]


_SOURCE_CLAIM_ALLOWED_KINDS = frozenset(
    {
        "defines",
        "sets",
        "implements",
        "amends",
        "supersedes",
        "restates",
        "delegates",
        "applies_to",
        "requires",
        "creates_exception",
    }
)
_SOURCE_CLAIM_EXECUTABLE_KEYS = frozenset(
    {
        "formula",
        "formulas",
        "input",
        "inputs",
        "output",
        "outputs",
        "case",
        "cases",
        "test",
        "tests",
        "test_cases",
        "runtime",
        "trace",
        "traces",
        "result",
        "results",
        "eligibility",
        "benefit_amount",
        "decision",
    }
)
_SOURCE_CLAIM_ABSOLUTE_TARGET_ID = re.compile(r"^[a-z][a-z0-9_.-]*:[^\s]+$")
_SOURCE_CLAIM_FRIENDLY_CONCEPT_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


def find_source_claim_reference_issues(content: str) -> list[str]:
    """Validate optional RuleSpec refs to accepted corpus-backed source claims."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []

    module = payload.get("module")
    if not isinstance(module, dict) or "source_claims" not in module:
        return []

    raw_refs = module.get("source_claims")
    if not isinstance(raw_refs, list) or not raw_refs:
        return [
            "Source claims malformed: `module.source_claims` must be a non-empty "
            "list of accepted claim IDs."
        ]

    source_verification = _source_verification_block(payload)
    citation_paths: tuple[str, ...] = ()
    if source_verification is not None:
        citation_paths, _ = _source_verification_source_fields(source_verification)

    issues: list[str] = []
    if not citation_paths:
        issues.append(
            "Source claims require direct source verification: "
            "`module.source_claims` may only supplement, not replace, "
            "`module.source_verification.corpus_citation_path` or "
            "`module.source_verification.corpus_citation_paths`."
        )

    claim_ids = _extract_source_claim_ids(raw_refs)
    if not claim_ids:
        issues.append(
            "Source claims malformed: `module.source_claims` must contain claim "
            "IDs as strings or `{id: ...}` mappings."
        )
        return issues

    for claim_id in claim_ids:
        claim = _fetch_local_source_claim_record(claim_id)
        if claim is None:
            issues.append(
                "Source claim missing: "
                f"`{claim_id}` was not found in local corpus claim artifacts."
            )
            continue
        issues.extend(
            _validate_source_claim_record(
                claim_id=claim_id,
                claim=claim,
                rulespec_citation_paths=citation_paths,
            )
        )

    return issues


def _extract_source_claim_ids(raw_refs: list[Any]) -> list[str]:
    claim_ids: list[str] = []
    for raw_ref in raw_refs:
        claim_id = ""
        if isinstance(raw_ref, str):
            claim_id = raw_ref.strip()
        elif isinstance(raw_ref, dict):
            claim_id = str(raw_ref.get("id") or "").strip()
        if claim_id:
            claim_ids.append(claim_id)
    return claim_ids


def _validate_source_claim_record(
    *,
    claim_id: str,
    claim: dict[str, Any],
    rulespec_citation_paths: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []

    actual_id = str(claim.get("id") or "").strip()
    if actual_id != claim_id:
        issues.append(
            "Source claim ID mismatch: "
            f"`{claim_id}` resolved to a claim with id `{actual_id or '<missing>'}`."
        )

    status = str(claim.get("status") or "").strip()
    if status != "accepted":
        issues.append(
            "Source claim not accepted: "
            f"`{claim_id}` has status `{status or '<missing>'}`; RuleSpec may only "
            "reference accepted source claims."
        )

    kind = str(claim.get("kind") or "").strip()
    if kind not in _SOURCE_CLAIM_ALLOWED_KINDS:
        allowed = ", ".join(sorted(_SOURCE_CLAIM_ALLOWED_KINDS))
        issues.append(
            "Source claim kind invalid: "
            f"`{claim_id}` has kind `{kind or '<missing>'}`; allowed kinds are "
            f"{allowed}."
        )

    executable_paths = _source_claim_executable_field_paths(claim)
    if executable_paths:
        issues.append(
            "Source claim is executable: "
            f"`{claim_id}` contains execution fields "
            + ", ".join(f"`{path}`" for path in executable_paths[:5])
            + ("; ..." if len(executable_paths) > 5 else "")
            + ". Claims may assert source meaning but must not contain formulas, "
            "case inputs, outputs, tests, runtime traces, decisions, or benefit amounts."
        )

    issues.extend(_validate_source_claim_subject(claim_id=claim_id, claim=claim))

    evidence = claim.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        issues.append(
            "Source claim evidence missing: "
            f"`{claim_id}` must cite at least one corpus evidence span."
        )
        return issues

    for index, evidence_item in enumerate(evidence):
        if not isinstance(evidence_item, dict):
            issues.append(
                "Source claim evidence malformed: "
                f"`{claim_id}.evidence[{index}]` must be a mapping."
            )
            continue
        evidence_path = str(evidence_item.get("corpus_citation_path") or "").strip()
        if not evidence_path:
            issues.append(
                "Source claim evidence missing corpus path: "
                f"`{claim_id}.evidence[{index}]` must declare "
                "`corpus_citation_path`."
            )
            continue
        if rulespec_citation_paths and evidence_path not in rulespec_citation_paths:
            issues.append(
                "Source claim evidence outside RuleSpec source: "
                f"`{claim_id}` cites `{evidence_path}`, but the RuleSpec verifies "
                "against "
                + _format_source_verification_paths(rulespec_citation_paths)
                + ". Add the corpus path to `module.source_verification` or split "
                "the claim reference."
            )
        quote = str(evidence_item.get("quote") or "").strip()
        if quote:
            source_text = _fetch_corpus_source_text(evidence_path)
            if source_text is not None and quote not in source_text:
                issues.append(
                    "Source claim quote not found: "
                    f"`{claim_id}.evidence[{index}].quote` does not appear in "
                    f"`{evidence_path}`."
                )

    return issues


def _validate_source_claim_subject(
    *,
    claim_id: str,
    claim: dict[str, Any],
) -> list[str]:
    subject = claim.get("subject")
    if not isinstance(subject, dict):
        return [
            "Source claim subject missing: "
            f"`{claim_id}` must declare `subject` with an absolute legal or "
            "RuleSpec target."
        ]

    subject_id = str(subject.get("id") or "").strip()
    subject_type = str(subject.get("type") or "").strip()
    issues: list[str] = []
    if not _SOURCE_CLAIM_ABSOLUTE_TARGET_ID.match(subject_id):
        issues.append(
            "Source claim subject target invalid: "
            f"`{claim_id}.subject.id` is `{subject_id or '<missing>'}`; use an "
            "absolute legal, corpus, or RuleSpec target such as "
            "`us:statutes/7/2014/e`."
        )
    if subject_type == "concept" or _SOURCE_CLAIM_FRIENDLY_CONCEPT_ID.match(subject_id):
        issues.append(
            "Source claim subject placeholder not allowed: "
            f"`{claim_id}` uses `{subject_id or '<missing>'}`; friendly concept "
            "IDs are not valid claim subjects."
        )
    return issues


def _source_claim_executable_field_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text in _SOURCE_CLAIM_EXECUTABLE_KEYS:
                paths.append(path)
            paths.extend(_source_claim_executable_field_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(
                _source_claim_executable_field_paths(child, f"{prefix}[{index}]")
            )
    return paths


@functools.lru_cache(maxsize=512)
def _fetch_local_source_claim_record(claim_id: str) -> dict[str, Any] | None:
    normalized_id = claim_id.strip()
    if not normalized_id:
        return None

    for claims_root in _local_corpus_claims_roots():
        for claim_file in sorted(claims_root.rglob("*.jsonl")):
            claim = _read_local_source_claim_file(claim_file, normalized_id)
            if claim is not None:
                return claim
    return None


def _read_local_source_claim_file(
    claim_file: Path,
    claim_id: str,
) -> dict[str, Any] | None:
    try:
        lines = claim_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("id") != claim_id:
            continue
        return record
    return None


def _local_corpus_claims_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for env_name in (
        "AXIOM_CORPUS_CLAIMS_ROOT",
        "AXIOM_CORPUS_ARTIFACT_ROOT",
        "AXIOM_CORPUS_REPO",
    ):
        raw_root = os.environ.get(env_name)
        if raw_root:
            roots.append(Path(raw_root).expanduser())

    with contextlib.suppress(OSError):
        cwd = Path.cwd().resolve()
        for base in (cwd, *cwd.parents):
            roots.extend(
                (
                    base,
                    base / "axiom-corpus",
                    base / "TheAxiomFoundation" / "axiom-corpus",
                    base.parent / "axiom-corpus",
                    base / "_axiom" / "axiom-corpus",
                )
            )

    with contextlib.suppress(RuntimeError, OSError):
        roots.append(Path.home() / "TheAxiomFoundation" / "axiom-corpus")

    claims_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for candidate in (
            root,
            root / "claims",
            root / "data" / "corpus",
            root / "data" / "corpus" / "claims",
        ):
            claims_root = (
                candidate if candidate.name == "claims" else candidate / "claims"
            )
            with contextlib.suppress(OSError):
                resolved = claims_root.resolve()
                if resolved.is_dir() and resolved not in seen:
                    seen.add(resolved)
                    claims_roots.append(resolved)
    return tuple(claims_roots)


def find_structured_scale_parameter_issues(content: str) -> list[str]:
    """Flag source-stated numeric scales encoded as branch formulas."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").lower()
        if kind == "parameter":
            if rule.get("indexed_by") is None and any(
                isinstance(version, dict)
                and isinstance(version.get("values"), dict)
                and version.get("values")
                for version in rule.get("versions") or []
            ):
                name = str(rule.get("name") or "<unknown>")
                issues.append(
                    "Structured parameter table malformed: "
                    f"{name} uses versioned `values` but does not declare `indexed_by`."
                )
            continue
        if kind != "derived":
            continue
        name = str(rule.get("name") or "<unknown>")
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            selector = _embedded_integer_scale_selector(formula)
            if selector is not None:
                issues.append(
                    "Structured parameter table required: "
                    f"{name} encodes a numeric schedule keyed by {selector} "
                    "inside a derived formula; move source-stated cells to a "
                    "`kind: parameter` rule with `indexed_by` and versioned `values`, "
                    "then reference it with table lookup syntax."
                )
                break
    return issues


def find_versioned_derived_formula_issues(content: str) -> list[str]:
    """Flag derived rules that rely on unsupported period-selected formulas."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "derived":
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        formula_versions = [
            version
            for version in versions
            if isinstance(version, dict)
            and isinstance(version.get("formula"), str)
            and version["formula"].strip()
        ]
        if len(formula_versions) <= 1:
            continue
        name = str(rule.get("name") or "<unknown>")
        issues.append(
            "Versioned derived formula unsupported: "
            f"{name} has {len(formula_versions)} formula versions. "
            "Use a single source-faithful conditional formula or resolve the "
            "currently applicable source context before encoding."
        )
    return issues


def find_upstream_placement_issues(
    content: str,
    *,
    rules_file: Path | None = None,
    source_metadata: dict[str, object] | None = None,
) -> list[str]:
    """Flag rules encoded downstream of their canonical legal authority."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    if source_metadata is None and rules_file is not None:
        source_metadata = _load_nearby_eval_source_metadata(rules_file)

    issues: list[str] = []
    issues.extend(_find_rule_metadata_schema_issues(rules))
    issues.extend(
        _find_source_metadata_upstream_issues(
            rules=rules,
            source_metadata=source_metadata,
        )
    )
    issues.extend(_find_restatement_executable_copy_issues(rules, source_metadata))
    issues.extend(
        _find_duplicate_upstream_executable_issues(
            rules=rules,
            rules_file=rules_file,
        )
    )
    return issues


_SOURCE_METADATA_RESTATEMENT_RELATIONS = {
    "restate",
    "restates",
    "restated",
    "copy",
    "copies",
    "copied",
}
_SOURCE_METADATA_DECLARATIVE_RELATIONS = {
    "sets": "sets",
    "set": "sets",
    "amends": "amends",
    "amend": "amends",
    "amended": "amends",
    "implements": "implements",
    "implement": "implements",
    "implemented": "implements",
}
_RULE_METADATA_TARGET_KEYS = ("defines", "delegates", "implements", "sets", "amends")


def _find_rule_metadata_schema_issues(rules: list[Any]) -> list[str]:
    """Reject source-relation metadata on executable RuleSpec rules."""
    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue

        metadata = rule.get("metadata")
        if metadata is None:
            continue

        rule_name = str(rule.get("name") or "<unnamed>").strip() or "<unnamed>"
        if not isinstance(metadata, dict):
            issues.append(
                f"RuleSpec relation metadata malformed: rule `{rule_name}` metadata "
                "must be a mapping."
            )
            continue

        relation_keys = [
            key
            for key in ("source_relation", *_RULE_METADATA_TARGET_KEYS)
            if key in metadata
        ]
        if relation_keys:
            keys = ", ".join(f"`metadata.{key}`" for key in relation_keys)
            issues.append(
                f"RuleSpec source-relation metadata is not allowed on executable "
                f"rule `{rule_name}`: {keys}. Encode legal/provenance edges as "
                "separate `kind: source_relation` records."
            )
            continue

        if str(metadata.get("concept_id") or "").strip():
            issues.append(
                f"RuleSpec relation metadata has placeholder target: rule "
                f"`{rule_name}` declares `metadata.concept_id`; use an absolute "
                "RuleSpec or corpus target in a separate `kind: source_relation` "
                "record instead."
            )

    return issues


def _find_source_metadata_upstream_issues(
    *,
    rules: list[Any],
    source_metadata: dict[str, object] | None,
) -> list[str]:
    """Enforce generic upstream/source relations from structured source metadata."""
    issues: list[str] = []
    for relation, target in _iter_source_metadata_target_relations(source_metadata):
        if relation in _SOURCE_METADATA_RESTATEMENT_RELATIONS:
            if _rules_include_source_relation_target(
                rules,
                target,
                relation_type="restates",
            ):
                continue
            issues.append(
                "Source metadata upstream relation requires source_relation: "
                f"source metadata says this source `{relation}` `{target}`, so "
                "encode it as `kind: source_relation` with "
                "`source_relation.type: restates` and `source_relation.target` "
                "instead of redefining executable policy locally."
            )
            continue

        metadata_key = _SOURCE_METADATA_DECLARATIVE_RELATIONS.get(relation)
        if metadata_key is None:
            continue
        if _rules_include_source_relation_target(
            rules,
            target,
            relation_type=metadata_key,
        ):
            continue
        issues.append(
            "Source metadata upstream relation not recorded: "
            f"source metadata says this source `{relation}` `{target}`, so "
            "the corresponding RuleSpec file must include a "
            f"`kind: source_relation` record with `source_relation.type: {metadata_key}` "
            f"and `source_relation.target: {target}`."
        )
    return issues


def _iter_source_metadata_target_relations(
    source_metadata: dict[str, object] | None,
) -> Iterable[tuple[str, str]]:
    if not isinstance(source_metadata, dict):
        return

    relations = source_metadata.get("relations")
    if not isinstance(relations, list):
        return

    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_name = str(relation.get("relation") or "").strip().lower()
        target = _normalize_relation_target(relation.get("target"))
        if relation_name and target:
            yield relation_name, target


def _rules_include_source_relation_target(
    rules: list[Any],
    target: str,
    *,
    relation_type: str,
) -> bool:
    return any(
        isinstance(rule, dict)
        and str(rule.get("kind") or "").lower() == "source_relation"
        and isinstance(rule.get("source_relation"), dict)
        and str(rule["source_relation"].get("type") or "").strip().lower()
        == relation_type
        and _target_matches(
            _normalize_relation_target(rule["source_relation"].get("target")),
            target,
        )
        for rule in rules
    )


def _normalize_relation_target(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().strip('"').strip("'")
    if not normalized:
        return None
    base, separator, symbol = normalized.partition("#")
    if base.endswith((".yaml", ".yml")):
        base = str(Path(base).with_suffix(""))
        base = str(Path(base).with_suffix(""))
    return f"{base}{separator}{symbol}" if separator else base


def _target_matches(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    return _normalize_relation_target(left) == _normalize_relation_target(right)


def _find_restatement_executable_copy_issues(
    rules: list[Any],
    source_metadata: dict[str, object] | None,
) -> list[str]:
    """Reject local executable rules that copy explicitly restated targets."""
    restated_symbols: dict[str, str] = {}
    for relation, target in _iter_source_metadata_target_relations(source_metadata):
        if relation not in _SOURCE_METADATA_RESTATEMENT_RELATIONS:
            continue
        symbol = _target_symbol(target)
        if symbol is not None:
            restated_symbols.setdefault(symbol, target)

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "source_relation":
            continue
        source_relation = rule.get("source_relation")
        if not isinstance(source_relation, dict):
            continue
        if str(source_relation.get("type") or "").strip().lower() != "restates":
            continue
        target = _normalize_relation_target(source_relation.get("target"))
        if target is None:
            continue
        symbol = _target_symbol(target)
        if symbol is not None:
            restated_symbols.setdefault(symbol, target)
        verification = rule.get("verification")
        if isinstance(verification, dict) and isinstance(
            verification.get("values"), dict
        ):
            for value_name in verification["values"]:
                restated_symbols.setdefault(str(value_name), target)

    issues: list[str] = []
    for rule in rules:
        if not _is_executable_rulespec_rule(rule):
            continue
        name = _rulespec_rule_name(rule)
        target = restated_symbols.get(name)
        if target is None:
            continue
        issues.append(
            "Restated upstream target copied as executable RuleSpec: "
            f"`{name}` duplicates `{target}`. Remove the local executable rule "
            "and keep only an import or a non-executable `kind: source_relation` "
            "record with verification if the downstream source restates it."
        )
    return issues


@dataclass(frozen=True)
class _IndexedExecutableRule:
    """Executable RuleSpec rule discovered in a local rule repository."""

    target: str
    symbol: str
    signature: str
    source_file: str


def _find_duplicate_upstream_executable_issues(
    *,
    rules: list[Any],
    rules_file: Path | None,
) -> list[str]:
    """Reject copied executable rules when an upstream RuleSpec target exists."""
    if rules_file is None:
        return []

    repo_root = _rulespec_repo_root(rules_file)
    if repo_root is None:
        return []

    prefix = _rulespec_repo_prefix(repo_root)
    current_file = Path(rules_file).resolve()
    candidate_roots = _candidate_upstream_rulespec_roots(repo_root)
    index = _rulespec_executable_index_for_roots(
        tuple(str(root.resolve()) for root in candidate_roots)
    )
    if not index:
        return []

    issues: list[str] = []
    for rule in rules:
        if not _is_executable_rulespec_rule(rule):
            continue
        signature = _rulespec_executable_signature(rule)
        if signature is None:
            continue
        name = _rulespec_rule_name(rule)
        current_target = _canonical_rulespec_target(
            prefix=prefix,
            repo_root=repo_root,
            rules_file=current_file,
            symbol=name,
        )
        for candidate in index:
            if candidate.symbol != name or candidate.signature != signature:
                continue
            if candidate.target == current_target:
                continue
            if Path(candidate.source_file).resolve() == current_file:
                continue
            issues.append(
                "Upstream placement violation: "
                f"executable rule `{name}` duplicates existing RuleSpec target "
                f"`{candidate.target}`. Remove the local executable copy and "
                "use an import, or encode only a non-executable "
                "`kind: source_relation` record if this source restates the "
                "upstream rule."
            )
            break
    return issues


def _rulespec_repo_root(rules_file: Path) -> Path | None:
    path = Path(rules_file).resolve()
    search = path if path.is_dir() else path.parent
    for candidate in (search, *search.parents):
        if candidate.name.startswith("rulespec-"):
            return candidate
    return None


def _rulespec_repo_prefix(repo_root: Path) -> str:
    return repo_root.name.removeprefix("rulespec-")


def _candidate_upstream_rulespec_roots(repo_root: Path) -> tuple[Path, ...]:
    """Return repos that can contain canonical targets for this repo."""
    roots: list[Path] = []

    def add(candidate: Path) -> None:
        if candidate.exists() and candidate.is_dir():
            roots.append(candidate)

    add(repo_root)
    prefix_parts = _rulespec_repo_prefix(repo_root).split("-")
    for length in range(len(prefix_parts) - 1, 0, -1):
        ancestor_prefix = "-".join(prefix_parts[:length])
        add(repo_root.parent / f"rulespec-{ancestor_prefix}")
        add(repo_root / "_axiom" / f"rulespec-{ancestor_prefix}")
        add(repo_root.parent / "_axiom" / f"rulespec-{ancestor_prefix}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(root)
    return tuple(unique)


@functools.lru_cache(maxsize=64)
def _rulespec_executable_index_for_roots(
    root_paths: tuple[str, ...],
) -> tuple[_IndexedExecutableRule, ...]:
    records: list[_IndexedExecutableRule] = []
    for root_path in root_paths:
        root = Path(root_path)
        if not root.exists():
            continue
        prefix = _rulespec_repo_prefix(root)
        for rules_file in sorted(root.rglob("*.yaml")):
            if rules_file.name.endswith(".test.yaml"):
                continue
            if "_axiom" in rules_file.relative_to(root).parts:
                continue
            try:
                payload = yaml.safe_load(rules_file.read_text())
            except (OSError, yaml.YAMLError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
                continue
            rules = payload.get("rules")
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if not _is_executable_rulespec_rule(rule):
                    continue
                signature = _rulespec_executable_signature(rule)
                if signature is None:
                    continue
                symbol = _rulespec_rule_name(rule)
                records.append(
                    _IndexedExecutableRule(
                        target=_canonical_rulespec_target(
                            prefix=prefix,
                            repo_root=root,
                            rules_file=rules_file,
                            symbol=symbol,
                        ),
                        symbol=symbol,
                        signature=signature,
                        source_file=str(rules_file.resolve()),
                    )
                )
    return tuple(records)


def _is_executable_rulespec_rule(rule: Any) -> bool:
    if not isinstance(rule, dict):
        return False
    return str(rule.get("kind") or "").strip().lower() in {"parameter", "derived"}


def _has_module_source_locator(payload: dict[str, Any]) -> bool:
    module = payload.get("module")
    if not isinstance(module, dict):
        return False
    source_verification = module.get("source_verification")
    if not isinstance(source_verification, dict):
        return False
    if source_verification.get("corpus_citation_path"):
        return True
    citation_paths = source_verification.get("corpus_citation_paths")
    return isinstance(citation_paths, list) and any(citation_paths)


def find_rule_source_metadata_issues(content: str) -> list[str]:
    """Require source metadata for executable RuleSpec records."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    executable_rules = [rule for rule in rules if _is_executable_rulespec_rule(rule)]
    if not executable_rules:
        return []

    issues: list[str] = []
    if not _has_module_source_locator(payload):
        issues.append(
            "Rule source locator required: module.source_verification must include "
            "`corpus_citation_path` or `corpus_citation_paths` when executable "
            "rules are present."
        )

    for rule in executable_rules:
        name = _rulespec_rule_name(rule)
        if not rule.get("source"):
            issues.append(
                "Rule source metadata required: "
                f"`{name}` is an executable rule and must include `source:` "
                "with the legal citation/span supporting it."
            )

    return issues


def _rulespec_rule_name(rule: dict[str, Any]) -> str:
    return str(rule.get("name") or "<unknown>").strip() or "<unknown>"


def _canonical_rulespec_target(
    *,
    prefix: str,
    repo_root: Path,
    rules_file: Path,
    symbol: str,
) -> str:
    relative = rules_file.resolve().relative_to(repo_root.resolve())
    if relative.suffix in {".yaml", ".yml"}:
        relative = relative.with_suffix("")
    return f"{prefix}:{relative.as_posix()}#{symbol}"


def _canonical_rulespec_file_target(
    *,
    policy_repo_path: Path | None,
    rules_file: Path,
    symbol: str,
) -> str | None:
    repo_root = (
        Path(policy_repo_path)
        if policy_repo_path is not None
        and Path(policy_repo_path).name.startswith("rulespec-")
        else _rulespec_repo_root(rules_file)
    )
    if repo_root is None:
        return None
    try:
        return _canonical_rulespec_target(
            prefix=_rulespec_repo_prefix(repo_root),
            repo_root=repo_root,
            rules_file=rules_file,
            symbol=symbol,
        )
    except ValueError:
        return None


def _strict_rules_repo_layout_checks_enabled(
    *,
    policy_repo_path: Path | None,
    rules_file: Path,
) -> bool:
    if policy_repo_path is None:
        return False
    repo_root = Path(policy_repo_path)
    if not repo_root.name.startswith("rulespec-"):
        return False
    try:
        rules_file.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return True


def find_missing_derived_companion_output_issues(
    content: str,
    cases: list[Any],
    *,
    rules_file: Path,
    policy_repo_path: Path | None = None,
) -> list[str]:
    """Require companion tests to assert every local derived output."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    covered_outputs: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            continue
        outputs = case.get("output")
        if isinstance(outputs, dict):
            covered_outputs.update(str(name) for name in outputs)

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("kind") != "derived":
            continue
        name = _rulespec_rule_name(rule)
        target = _canonical_rulespec_file_target(
            policy_repo_path=policy_repo_path,
            rules_file=rules_file,
            symbol=name,
        )
        if target is None or target in covered_outputs:
            continue
        issues.append(
            "Derived rule missing companion output coverage: "
            f"`{target}` is not asserted by the companion `.test.yaml` file."
        )

    return issues


def _rulespec_executable_signature(rule: dict[str, Any]) -> str | None:
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return None

    normalized_versions: list[dict[str, Any]] = []
    substantive = False
    for version in versions:
        if not isinstance(version, dict):
            continue
        normalized_version: dict[str, Any] = {}
        for key in ("effective_from", "from", "effective_to", "to"):
            if key in version:
                normalized_version[key] = _canonical_yaml_value(version.get(key))
        values = version.get("values")
        if isinstance(values, dict) and values:
            normalized_version["values"] = _canonical_yaml_value(values)
            substantive = True
        else:
            formula = version.get("formula")
            if isinstance(formula, str) and formula.strip():
                normalized_formula = _normalize_formula_signature(formula)
                normalized_version["formula"] = normalized_formula
                if _formula_signature_is_substantive(normalized_formula):
                    substantive = True
        if "values" in normalized_version or "formula" in normalized_version:
            normalized_versions.append(normalized_version)

    if not normalized_versions or not substantive:
        return None

    payload = {
        "kind": str(rule.get("kind") or "").strip().lower(),
        "dtype": str(rule.get("dtype") or "").strip(),
        "period": str(rule.get("period") or "").strip(),
        "unit": str(rule.get("unit") or "").strip(),
        "indexed_by": str(rule.get("indexed_by") or "").strip(),
        "versions": normalized_versions,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _normalize_formula_signature(formula: str) -> str:
    return re.sub(r"\s+", " ", formula.strip())


def _formula_signature_is_substantive(formula: str) -> bool:
    numeric = _numeric_rule_value(formula)
    if numeric is not None:
        value = numeric[1]
        return not (float(value).is_integer() and int(value) in {-1, 0, 1, 2, 3})
    if re.fullmatch(r"[A-Za-z_][\w.]*", formula):
        return False
    return True


def _canonical_yaml_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonical_yaml_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, list):
        return [_canonical_yaml_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _target_symbol(target: str | None) -> str | None:
    if not target:
        return None
    _, separator, symbol = target.partition("#")
    if not separator:
        return None
    symbol = symbol.strip()
    return symbol or None


def find_source_verification_issues(
    content: str,
    *,
    source_texts: dict[str, str] | None = None,
) -> list[str]:
    """Validate declared RuleSpec values against an ingested corpus source page."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []

    source_verification = _source_verification_block(payload)
    if source_verification is None:
        return []

    citation_paths, source_label = _source_verification_source_fields(
        source_verification
    )
    expected_values = source_verification.get("values")
    if not citation_paths:
        return [
            "Source verification source required: missing `corpus_citation_path`, "
            "or `corpus_citation_paths`."
        ]
    if not isinstance(expected_values, dict) or not expected_values:
        if expected_values is not None:
            return [
                "Source verification values invalid: "
                "`source_verification.values` must be a non-empty mapping when present."
            ]
        expected_values = {}

    rulespec_values, _, load_issue = _extract_rulespec_parameter_values(payload)
    if load_issue is not None:
        return [f"Source verification RuleSpec invalid: {load_issue}"]

    issues: list[str] = []
    for value_name, expected_value in expected_values.items():
        value_key = str(value_name)
        if value_key not in rulespec_values:
            issues.append(
                "Source verification RuleSpec value missing: "
                f"`{value_key}` is declared for source verification but is not "
                "defined as a scalar/table parameter."
            )
            continue
        issues.extend(
            _compare_source_verification_expected_value(
                value_name=value_key,
                expected_value=expected_value,
                rulespec_value=rulespec_values[value_key],
            )
        )

    source_text = _source_verification_text(
        citation_paths=citation_paths,
        source_label=source_label,
        source_texts=source_texts,
    )
    if source_text is None:
        issues.append(
            "Source verification source missing: "
            + f"{_format_source_verification_paths(citation_paths)} "
            "was not found in corpus.provisions."
        )
        return issues
    if not expected_values:
        return issues

    for value_name, expected_value in expected_values.items():
        value_key = str(value_name)
        issues.extend(
            _find_source_text_value_issues(
                source_label=source_label,
                source_text=source_text,
                value_name=value_key,
                expected_value=expected_value,
            )
        )

    return issues


_COST_AVAILABILITY_SOURCE_PATTERN = re.compile(
    r"\b(available|allowed|eligible|entitled)\b.{0,180}"
    r"\b(billed|cost|costs|expense|expenses|incur|incurs|incurred)\b"
    r"|"
    r"\b(billed|cost|costs|expense|expenses|incur|incurs|incurred)\b"
    r".{0,180}\b(available|allowed|eligible|entitled)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def find_source_condition_coverage_issues(
    content: str,
    *,
    source_texts: dict[str, str] | None = None,
) -> list[str]:
    """Flag eligibility formulas that collapse cost availability to only exclusions."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    source_verification = _source_verification_block(payload)
    if source_verification is None:
        return []
    citation_paths, source_label = _source_verification_source_fields(
        source_verification
    )
    source_text = _source_verification_text(
        citation_paths=citation_paths,
        source_label=source_label,
        source_texts=source_texts,
    )
    if not source_text or not _COST_AVAILABILITY_SOURCE_PATTERN.search(source_text):
        return []

    issues: list[str] = []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return issues
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        kind = str(rule.get("kind") or "").strip().lower()
        if kind != "derived":
            continue
        name = str(rule.get("name") or "").strip()
        normalized_name = name.lower()
        if not any(
            token in normalized_name
            for token in ("eligible", "available", "allowance", "applies")
        ):
            continue
        dtype = str(rule.get("dtype") or "").strip().lower()
        if dtype not in {"judgment", "boolean", "bool"}:
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            identifiers = set(_RULESPEC_IDENTIFIER.findall(formula))
            identifiers -= _RULESPEC_FORMULA_BUILTINS
            if not identifiers:
                continue
            negated = {
                match.group(1)
                for match in re.finditer(r"\bnot\s+([A-Za-z_][A-Za-z0-9_]*)\b", formula)
            }
            if identifiers and identifiers <= negated:
                issues.append(
                    "Source condition coverage missing: "
                    f"`{name}` is grounded in source text that makes cost/expense availability conditional, "
                    "but its formula only negates other predicates. Add a positive fact predicate "
                    "for the source-stated cost, billing, payment, or incurrence condition."
                )
                continue
    return issues


_BROAD_APPLICATION_FURNISHING_SOURCE_PATTERN = re.compile(
    r"\b(?:shall|must|may)\s+(?:be\s+)?"
    r"(?:furnished|provided|paid|made\s+available|granted|issued)\b"
    r"(?:(?!\n\n).){0,240}\b(?:eligible|eligibility)\b"
    r"(?:(?!\n\n).){0,160}\b(?:application|applicant|apply|applies)\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def find_broad_application_passthrough_issues(content: str) -> list[str]:
    """Flag administrative furnishing/application clauses encoded as outputs."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    source_text = extract_embedded_source_text(
        content
    ) or _extract_source_verification_text(content)
    if not source_text or not _BROAD_APPLICATION_FURNISHING_SOURCE_PATTERN.search(
        source_text
    ):
        return []

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "derived":
            continue
        name = str(rule.get("name") or "").strip()
        if not name:
            continue
        if not re.search(
            r"\b(?:furnish|provid|paid|grant|issue|assistance|entitle|participat)",
            name.replace("_", " "),
            flags=re.IGNORECASE,
        ):
            continue
        dtype = str(rule.get("dtype") or "").strip().lower()
        if dtype not in {"judgment", "boolean", "bool"}:
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            identifiers = _formula_local_identifiers(formula)
            if not identifiers or len(identifiers) > 3:
                continue
            if not any("eligib" in identifier for identifier in identifiers):
                continue
            if not any(
                re.search(r"applic|applicant|apply|applies", identifier)
                for identifier in identifiers
            ):
                continue
            issues.append(
                "Broad application pass-through: "
                f"`{name}` encodes an administrative furnishing/application clause as "
                "a generic executable output. Keep that clause in `module.summary`; "
                "encode only source-specific eligibility conditions, exceptions, "
                "parameters, or source relations."
            )
            break
    return issues


def find_copied_cross_reference_source_issues(
    content: str,
    *,
    rules_file: Path,
    policy_repo_path: Path,
) -> list[str]:
    """Reject copying cited same-section subsection bodies into this module."""
    source_text = extract_embedded_source_text(content)
    if not source_text:
        return []

    statute_path = _statute_path_parts_for_file(rules_file, policy_repo_path)
    if statute_path is None:
        return []
    title, section, current_fragments = statute_path

    issues: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\bsubsection\s+\((?P<subsection>[A-Za-z0-9]+)\)\s+of\s+this\s+section\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        subsection = match.group("subsection")
        if subsection in current_fragments:
            continue
        if subsection in seen:
            continue
        copied_body_pattern = re.compile(
            rf"(?:^|\s)\({re.escape(subsection)}\)\s+[A-Z][A-Za-z]"
        )
        if not copied_body_pattern.search(source_text):
            continue
        seen.add(subsection)
        import_base = "/".join(["statutes", title, section, subsection])
        issues.append(
            "Copied cross-reference source: "
            f"`{rules_file.name}` embeds the body of cited subsection "
            f"`{import_base}` in its own source summary. Encode/import that "
            "subsection separately instead of re-encoding it locally."
        )
    return issues


def find_missing_same_section_subsection_import_issues(
    content: str,
    *,
    rules_file: Path,
    policy_repo_path: Path,
) -> list[str]:
    """Require imports for cited same-section subsections used as carve-outs."""
    source_text = extract_embedded_source_text(content)
    if not source_text or not re.search(
        r"\b(?:except|unless|notwithstanding)\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        return []

    statute_path = _statute_path_parts_for_file(rules_file, policy_repo_path)
    if statute_path is None:
        return []
    title, section, current_fragments = statute_path
    imports = {
        _normalize_rulespec_import_path_static(import_path)
        for import_path in _extract_import_paths_from_content(content)
    }

    issues: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\bsubsection\s+\((?P<subsection>[A-Za-z0-9]+)\)\s+of\s+this\s+section\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        subsection = match.group("subsection")
        if subsection in current_fragments or subsection in seen:
            continue
        import_base = "/".join(["statutes", title, section, subsection])
        if _imports_cover_path_static(
            imports,
            import_base,
        ) or _transitive_imports_cover_path_static(
            imports,
            import_base,
            rules_file=rules_file,
            policy_repo_path=policy_repo_path,
        ):
            continue
        seen.add(subsection)
        issues.append(
            "Same-section subsection import missing: "
            f"source text cites subsection `{import_base}` in an exception/cross-reference "
            "clause, but the file does not import it. Encode/import the cited "
            "subsection instead of modeling its requirements as a local fact."
        )
    return issues


def find_rule_name_path_suffix_issues(
    content: str,
    *,
    rules_file: Path,
    policy_repo_path: Path,
) -> list[str]:
    """Reject rule names that append the file's legal path fragments."""
    statute_path = _statute_path_parts_for_file(rules_file, policy_repo_path)
    if statute_path is None:
        return []
    _title, section, fragments = statute_path
    if not fragments:
        return []

    suffixes = _path_suffix_tokens_for_rule_name(section, fragments)
    if not suffixes:
        return []

    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return []
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return []

        issues: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if not name:
                continue
            normalized = name.lower()
            for suffix in suffixes:
                if normalized.endswith(f"_{suffix}"):
                    issues.append(
                        "Rule name includes citation suffix: "
                        f"rule `{name}` ends with `_{suffix}`. The file path is "
                        "the legal ID; use a semantic rule name without path "
                        "fragments."
                    )
                    break
        return issues
    return []


def _path_suffix_tokens_for_rule_name(
    section: str,
    fragments: tuple[str, ...],
) -> set[str]:
    """Return path-fragment suffixes specific enough to avoid common words."""
    normalized_fragments = tuple(
        _normalize_rule_name_suffix_token(item) for item in fragments
    )
    normalized_fragments = tuple(item for item in normalized_fragments if item)
    if not normalized_fragments:
        return set()

    suffixes: set[str] = set()
    section_token = _normalize_rule_name_suffix_token(section)
    if section_token:
        suffixes.add("_".join((section_token, *normalized_fragments)))
    if len(normalized_fragments) >= 2:
        suffixes.add("_".join(normalized_fragments))
        suffixes.add("_".join(normalized_fragments[-2:]))
    return {suffix for suffix in suffixes if suffix}


def _normalize_rule_name_suffix_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_sibling_rule_name_collision_issues(
    content: str, rules_file: Path
) -> list[str]:
    """Reject exported rule names that collide with sibling child fragments."""
    current_names = _rulespec_rule_names_from_content(content)
    if not current_names:
        return []

    issues: list[str] = []
    current_path = rules_file.resolve()
    sibling_names: dict[str, Path] = {}
    for sibling in sorted(rules_file.parent.glob("*.yaml")):
        if sibling.name.endswith(".test.yaml") or sibling.resolve() == current_path:
            continue
        for name in _rulespec_rule_names_from_file(sibling):
            sibling_names.setdefault(name, sibling)

    for name in sorted(current_names & sibling_names.keys()):
        sibling = sibling_names[name]
        issues.append(
            "Sibling rule name collision: "
            f"rule `{name}` is also exported by sibling `{sibling.name}`. "
            "Use semantic branch-specific names so aggregate parent provisions "
            "can import sibling outputs without ambiguity."
        )
    return issues


_FORMULA_ABSOLUTE_REFERENCE_PATTERN = re.compile(
    r"\b[a-z][a-z0-9-]*:[A-Za-z0-9_./-]+#[A-Za-z_][A-Za-z0-9_]*\b"
)


def find_formula_absolute_reference_issues(content: str) -> list[str]:
    """Reject absolute RuleSpec import references inside formula text."""
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return []
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return []

        issues: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if not name:
                continue
            versions = rule.get("versions")
            if not isinstance(versions, list):
                continue
            seen: set[str] = set()
            for version in versions:
                if not isinstance(version, dict):
                    continue
                formula = version.get("formula")
                if not isinstance(formula, str):
                    continue
                for match in _FORMULA_ABSOLUTE_REFERENCE_PATTERN.finditer(formula):
                    target = match.group(0)
                    if target in seen:
                        continue
                    seen.add(target)
                    issues.append(
                        "Formula absolute import reference: "
                        f"`{name}` contains `{target}` inside a formula. Add "
                        "that target to `imports:` and reference the imported "
                        "rule by bare local name in formula text."
                    )
        return issues
    return []


def _rulespec_rule_names_from_file(path: Path) -> set[str]:
    with contextlib.suppress(OSError):
        return _rulespec_rule_names_from_content(path.read_text())
    return set()


def _rulespec_rule_names_from_content(content: str) -> set[str]:
    with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
        payload = yaml.safe_load(content)
        if not isinstance(payload, dict):
            return set()
        rules = payload.get("rules")
        if not isinstance(rules, list):
            return set()
        return {
            name
            for rule in rules
            if isinstance(rule, dict)
            for name in [str(rule.get("name") or "").strip()]
            if name
        }
    return set()


def _statute_path_parts_for_file(
    rules_file: Path,
    policy_repo_path: Path,
) -> tuple[str, str, tuple[str, ...]] | None:
    """Return title, section, and subsection fragments for a statute RuleSpec file."""
    resolved_file = rules_file.resolve()
    resolved_root = policy_repo_path.resolve()
    relative_parts: tuple[str, ...] | None = None
    with contextlib.suppress(ValueError):
        relative_parts = resolved_file.relative_to(resolved_root).parts
    if relative_parts is None:
        parts = resolved_file.parts
        with contextlib.suppress(ValueError):
            statutes_index = parts.index("statutes")
            relative_parts = tuple(parts[statutes_index:])
    if not relative_parts or len(relative_parts) < 4 or relative_parts[0] != "statutes":
        return None

    title = relative_parts[1]
    section = relative_parts[2]
    fragments = list(relative_parts[3:-1])
    stem = Path(relative_parts[-1]).stem
    if stem not in {"index", "__init__"}:
        fragments.append(stem)
    return title, section, tuple(fragments)


def _extract_import_paths_from_content(content: str) -> list[str]:
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
        if item_match:
            item = item_match.group(2).strip()
        else:
            mapping_match = IMPORT_MAPPING_PATTERN.match(line)
            if not mapping_match:
                continue
            item = mapping_match.group(2).strip()
        import_target = item.split("#", 1)[0].strip()
        if import_target:
            paths.append(import_target)

    return paths


def _normalize_rulespec_import_path_static(import_path: str) -> str:
    normalized = import_path.split("#", 1)[0].strip().strip("\"'")
    if ":" in normalized:
        _, tail = normalized.split(":", 1)
        normalized = tail
    return normalized.strip("/")


def _imports_cover_path_static(imports: set[str], expected_path: str) -> bool:
    expected = expected_path.strip("/")
    for import_path in imports:
        if import_path == expected:
            return True
        if import_path.startswith(expected + "/"):
            return True
        if expected.startswith(import_path + "/"):
            return True
    return False


def _transitive_imports_cover_path_static(
    imports: set[str],
    expected_path: str,
    *,
    rules_file: Path,
    policy_repo_path: Path,
) -> bool:
    """Return whether imported RuleSpecs import an expected path."""
    for import_path in imports:
        import_file = _resolve_rulespec_import_file_static(
            import_path,
            rules_file=rules_file,
            policy_repo_path=policy_repo_path,
        )
        if import_file is None:
            continue
        with contextlib.suppress(OSError):
            nested_imports = {
                _normalize_rulespec_import_path_static(path)
                for path in _extract_import_paths_from_content(import_file.read_text())
            }
            if _imports_cover_path_static(nested_imports, expected_path):
                return True
    return False


def _resolve_rulespec_import_file_static(
    import_path: str,
    *,
    rules_file: Path,
    policy_repo_path: Path,
) -> Path | None:
    """Resolve a normalized import path to a local RuleSpec file."""
    normalized = _normalize_rulespec_import_path_static(import_path)
    if not normalized:
        return None
    candidate = policy_repo_path / f"{normalized}.yaml"
    if candidate.exists():
        return candidate

    repo_prefix = _rulespec_import_prefix_static(import_path)
    if repo_prefix:
        candidate = (
            policy_repo_path.parent / f"rulespec-{repo_prefix}" / f"{normalized}.yaml"
        )
        if candidate.exists():
            return candidate

    with contextlib.suppress(ValueError):
        relative_parent = rules_file.parent.resolve().relative_to(
            policy_repo_path.resolve()
        )
        candidate = policy_repo_path / relative_parent / f"{normalized}.yaml"
        if candidate.exists():
            return candidate
    return None


def _rulespec_import_prefix_static(import_path: str) -> str | None:
    normalized = import_path.split("#", 1)[0].strip().strip("\"'")
    if ":" not in normalized:
        return None
    prefix, tail = normalized.split(":", 1)
    if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and tail:
        return prefix
    return None


def _formula_local_identifiers(formula: str) -> set[str]:
    """Return non-builtin identifiers referenced by a RuleSpec formula."""
    return set(_RULESPEC_IDENTIFIER.findall(formula)) - _RULESPEC_FORMULA_BUILTINS


def find_test_input_assignment_issues(
    content: str,
    test_cases: Any,
) -> list[str]:
    """Require tests to assign every local factual input used by formulas."""
    if not isinstance(test_cases, list):
        return []
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    module = payload.get("module")
    proof_validation = (
        module.get("proof_validation") if isinstance(module, dict) else {}
    )
    if not (
        isinstance(proof_validation, dict) and proof_validation.get("required") is True
    ):
        return []

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    defined_symbols = _defined_rulespec_symbols(rules)
    symbol_inputs: dict[str, set[str]] = {}
    symbol_dependencies: dict[str, set[str]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name") or "").strip()
        if not rule_name:
            continue
        if str(rule.get("kind") or "").strip().lower() == "parameter":
            symbol_inputs[rule_name] = _indexed_by_input_names(rule.get("indexed_by"))
            symbol_dependencies[rule_name] = set()
            continue
        if str(rule.get("kind") or "").strip().lower() != "derived":
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        formula_identifiers: set[str] = set()
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            formula_identifiers.update(_formula_local_identifiers(formula))
        symbol_inputs[rule_name] = formula_identifiers - defined_symbols
        symbol_dependencies[rule_name] = formula_identifiers & defined_symbols

    if not symbol_inputs:
        return []

    imports_present = bool(payload.get("imports"))
    assigned_input_names = _all_test_input_names(test_cases)
    globally_local_inputs = (
        set().union(*symbol_inputs.values()) if symbol_inputs else set()
    )
    if imports_present:
        globally_local_inputs &= assigned_input_names
    if not globally_local_inputs:
        return []

    issues: list[str] = []
    for index, test_case in enumerate(test_cases, start=1):
        if not isinstance(test_case, dict):
            continue
        test_name = str(test_case.get("name") or f"case {index}").strip()
        required_inputs = _required_inputs_for_test_outputs(
            test_case.get("output"),
            symbol_inputs=symbol_inputs,
            symbol_dependencies=symbol_dependencies,
        )
        local_inputs = required_inputs & globally_local_inputs
        if not local_inputs:
            continue
        inputs = test_case.get("input")
        if not isinstance(inputs, dict):
            issues.append(
                "Test input assignment missing: "
                f"`{test_name}` must provide an `input` mapping assigning every "
                "local factual `#input.<fact>` referenced by this module's formulas."
            )
            continue
        assigned = _input_names_from_mapping(inputs)
        missing = sorted(local_inputs - assigned)
        if not missing:
            continue
        missing_display = ", ".join(f"#input.{name}" for name in missing[:8])
        if len(missing) > 8:
            missing_display += f", and {len(missing) - 8} more"
        issues.append(
            "Test input assignment missing: "
            f"`{test_name}` does not assign {missing_display}. Every test for a "
            "proof-required RuleSpec module must set all local factual inputs, "
            "including false facts, so tests cannot pass through implicit defaults."
        )
    return issues


def _defined_rulespec_symbols(rules: list[Any]) -> set[str]:
    return {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }


def _indexed_by_input_names(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value.strip()} if value.strip() else set()
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


def _required_inputs_for_test_outputs(
    outputs: Any,
    *,
    symbol_inputs: dict[str, set[str]],
    symbol_dependencies: dict[str, set[str]],
) -> set[str]:
    if not isinstance(outputs, dict):
        return set().union(*symbol_inputs.values()) if symbol_inputs else set()

    required: set[str] = set()
    for output_name in outputs:
        fragment = _test_reference_fragment(output_name)
        if fragment in symbol_inputs:
            required.update(
                _required_inputs_for_symbol(
                    fragment,
                    symbol_inputs=symbol_inputs,
                    symbol_dependencies=symbol_dependencies,
                    seen=set(),
                )
            )
    return required


def _required_inputs_for_symbol(
    symbol: str,
    *,
    symbol_inputs: dict[str, set[str]],
    symbol_dependencies: dict[str, set[str]],
    seen: set[str],
) -> set[str]:
    if symbol in seen:
        return set()
    seen.add(symbol)
    required = set(symbol_inputs.get(symbol, set()))
    for dependency in symbol_dependencies.get(symbol, set()):
        required.update(
            _required_inputs_for_symbol(
                dependency,
                symbol_inputs=symbol_inputs,
                symbol_dependencies=symbol_dependencies,
                seen=seen,
            )
        )
    return required


def _all_test_input_names(test_cases: list[Any]) -> set[str]:
    names: set[str] = set()
    for test_case in test_cases:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        if isinstance(inputs, dict):
            names.update(_input_names_from_mapping(inputs))
    return names


def _input_names_from_mapping(inputs: dict[Any, Any]) -> set[str]:
    names: set[str] = set()
    for key in inputs:
        fragment = _test_reference_fragment(key)
        if fragment.startswith("input."):
            names.add(fragment.removeprefix("input."))
    return names


def find_exception_test_coverage_issues(
    content: str,
    test_cases: Any,
) -> list[str]:
    """Require each encoded exception predicate to have a blocking test."""
    source_text = extract_embedded_source_text(
        content
    ) or _extract_source_verification_text(content)
    if not source_text or not re.search(
        r"\b(?:except|unless|notwithstanding)\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        return []

    if not isinstance(test_cases, list):
        return []
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    defined_symbols = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "derived":
            continue
        dtype = str(rule.get("dtype") or "").strip().lower()
        if dtype not in {"judgment", "boolean", "bool"}:
            continue
        rule_name = str(rule.get("name") or "").strip()
        if not rule_name:
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        exception_inputs: set[str] = set()
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            exception_inputs.update(
                _exception_formula_inputs(formula) - defined_symbols
            )
        for exception_input in sorted(exception_inputs):
            if _has_exception_blocking_test(
                test_cases,
                rule_name=rule_name,
                exception_input=exception_input,
            ):
                continue
            issues.append(
                "Exception test coverage missing: "
                f"`{rule_name}` negates `{exception_input}`, but no companion test "
                f"sets `#input.{exception_input}` true and expects `{rule_name}` "
                "to be not_holds while an otherwise identical positive companion "
                "sets that exception false and expects holds."
            )
    return issues


def find_aggregate_exception_predicate_issues(content: str) -> list[str]:
    """Flag source exception lists collapsed into one factual predicate."""
    source_text = extract_embedded_source_text(
        content
    ) or _extract_source_verification_text(content)
    if not source_text or not re.search(
        r"\b(?:except|unless|notwithstanding)\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        return []

    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    defined_symbols = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name") or "").strip() or "<unnamed>"
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if not isinstance(formula, str):
                continue
            for identifier in sorted(_formula_local_identifiers(formula)):
                if identifier in defined_symbols:
                    continue
                if not _is_aggregate_exception_identifier(identifier):
                    continue
                issues.append(
                    "Aggregate exception predicate: "
                    f"`{rule_name}` references `{identifier}`, which collapses "
                    "multiple cited exception/cross-reference carve-outs into one "
                    "factual input. Encode or import each cited exception separately "
                    "so each one can be tested independently."
                )
    return issues


def _is_aggregate_exception_identifier(identifier: str) -> bool:
    normalized = identifier.lower()
    if "_and_" not in normalized:
        return False
    if "section" not in normalized and "subsection" not in normalized:
        return False
    return any(
        token in normalized
        for token in ("exception", "except", "preclude", "displace", "carve")
    )


def _exception_formula_inputs(formula: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"\bnot\s+([A-Za-z_][A-Za-z0-9_]*)\b", formula)
        if _is_exception_identifier(match.group(1))
    }


def _is_exception_identifier(identifier: str) -> bool:
    normalized = identifier.lower()
    return any(
        token in normalized
        for token in ("exception", "except", "exclusion", "carve_out", "displace")
    )


def _has_exception_blocking_test(
    test_cases: list[Any],
    *,
    rule_name: str,
    exception_input: str,
) -> bool:
    for test_case in test_cases:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        outputs = test_case.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if not any(
            _test_reference_fragment(key) == f"input.{exception_input}"
            and _is_truthy_fact_value(value)
            for key, value in inputs.items()
        ):
            continue
        if not any(
            _test_reference_fragment(key) == rule_name
            and _is_negative_judgment_value(value)
            for key, value in outputs.items()
        ):
            continue
        if _has_exception_positive_companion_test(
            test_cases,
            rule_name=rule_name,
            exception_input=exception_input,
            negative_inputs=inputs,
        ):
            return True
    return False


def _has_exception_positive_companion_test(
    test_cases: list[Any],
    *,
    rule_name: str,
    exception_input: str,
    negative_inputs: dict[Any, Any],
) -> bool:
    """Return true when a positive case proves the exception flips the outcome."""
    expected_inputs = _normalized_test_inputs(
        negative_inputs,
        exclude_input=exception_input,
    )
    for test_case in test_cases:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        outputs = test_case.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if not any(
            _test_reference_fragment(key) == f"input.{exception_input}"
            and not _is_truthy_fact_value(value)
            for key, value in inputs.items()
        ):
            continue
        if not any(
            _test_reference_fragment(key) == rule_name
            and str(value).strip().lower().replace("-", "_") == "holds"
            for key, value in outputs.items()
        ):
            continue
        if (
            _normalized_test_inputs(inputs, exclude_input=exception_input)
            == expected_inputs
        ):
            return True
    return False


def _normalized_test_inputs(
    inputs: dict[Any, Any],
    *,
    exclude_input: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    excluded_fragment = f"input.{exclude_input}"
    for key, value in inputs.items():
        fragment = _test_reference_fragment(key)
        if fragment == excluded_fragment:
            continue
        normalized[fragment] = _normalized_test_value(value)
    return normalized


def _normalized_test_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)) or value is None:
        return str(value).strip().lower()
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value).strip()


def _test_reference_fragment(key: Any) -> str:
    key_text = str(key).strip()
    return key_text.split("#", 1)[1].strip() if "#" in key_text else key_text


def _is_truthy_fact_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "holds", "yes", "1"}


def _is_negative_judgment_value(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"not_holds", "false", "no", "0"}


def _source_verification_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    module = payload.get("module")
    if isinstance(module, dict) and isinstance(module.get("source_verification"), dict):
        return module["source_verification"]
    source_verification = payload.get("source_verification")
    if isinstance(source_verification, dict):
        return source_verification
    return None


def _source_verification_source_fields(
    source_verification: dict[str, Any],
) -> tuple[tuple[str, ...], str]:
    citation_paths: list[str] = []
    raw_citation_path = str(
        source_verification.get("corpus_citation_path") or ""
    ).strip()
    if raw_citation_path:
        citation_paths.append(raw_citation_path)
    raw_citation_paths = source_verification.get("corpus_citation_paths")
    if isinstance(raw_citation_paths, list):
        for raw_path in raw_citation_paths:
            if not isinstance(raw_path, str):
                continue
            citation_path = raw_path.strip()
            if citation_path:
                citation_paths.append(citation_path)
    citation_path_tuple = tuple(dict.fromkeys(citation_paths))
    return citation_path_tuple, ", ".join(citation_path_tuple)


def _source_verification_text(
    *,
    citation_paths: tuple[str, ...],
    source_label: str,
    source_texts: dict[str, str] | None = None,
) -> str | None:
    if source_texts is not None:
        source_text = source_texts.get(source_label)
        if source_text is not None:
            return source_text
    if citation_paths:
        source_text_values: list[str] = []
        for citation_path in citation_paths:
            source_text = (
                source_texts.get(citation_path) if source_texts is not None else None
            )
            if source_text is None:
                source_text = _fetch_corpus_source_text(citation_path)
            if source_text is None:
                return None
            source_text_values.append(source_text)
        return "\n\n".join(source_text_values)
    return None


def _extract_source_verification_text(content: str) -> str | None:
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return None
    source_verification = _source_verification_block(payload)
    if source_verification is None:
        return None
    citation_paths, source_label = _source_verification_source_fields(
        source_verification
    )
    if not source_label:
        return None
    return _source_verification_text(
        citation_paths=citation_paths,
        source_label=source_label,
    )


def _format_source_verification_paths(citation_paths: tuple[str, ...]) -> str:
    if len(citation_paths) == 1:
        return f"`{citation_paths[0]}`"
    return (
        "`"
        + "`, `".join(citation_paths[:3])
        + ("`, ..." if len(citation_paths) > 3 else "`")
    )


def _compare_source_verification_expected_value(
    *,
    value_name: str,
    expected_value: Any,
    rulespec_value: Any,
) -> list[str]:
    """Check the verification block agrees with the RuleSpec parameter values."""
    if isinstance(expected_value, dict):
        if not isinstance(rulespec_value, dict):
            return [
                "Source verification RuleSpec mismatch: "
                f"`{value_name}` is declared as a table but the RuleSpec value is scalar."
            ]
        issues: list[str] = []
        for raw_key, expected_cell in expected_value.items():
            cell_key = str(raw_key)
            if cell_key not in rulespec_value:
                issues.append(
                    "Source verification RuleSpec value missing: "
                    f"`{value_name}[{cell_key}]` is declared but not defined."
                )
                continue
            actual_cell = rulespec_value[cell_key]
            if not _verification_values_equal(expected_cell, actual_cell):
                issues.append(
                    "Source verification RuleSpec mismatch: "
                    f"`{value_name}[{cell_key}]` declares "
                    f"{_format_verification_value(expected_cell)}, but RuleSpec has "
                    f"{_format_verification_value(actual_cell)}."
                )
        return issues

    if isinstance(rulespec_value, dict):
        return [
            "Source verification RuleSpec mismatch: "
            f"`{value_name}` is declared as a scalar but the RuleSpec value is a table."
        ]
    if _verification_values_equal(expected_value, rulespec_value):
        return []
    return [
        "Source verification RuleSpec mismatch: "
        f"`{value_name}` declares {_format_verification_value(expected_value)}, "
        f"but RuleSpec has {_format_verification_value(rulespec_value)}."
    ]


def _find_source_text_value_issues(
    *,
    source_label: str,
    source_text: str,
    value_name: str,
    expected_value: Any,
) -> list[str]:
    """Check expected values are present in the ingested source page text."""
    normalized_text = _normalize_source_verification_text(source_text)
    if isinstance(expected_value, dict):
        issues: list[str] = []
        for raw_key, expected_cell in expected_value.items():
            cell_key = str(raw_key)
            if not _source_text_contains_indexed_value(
                normalized_text,
                index=cell_key,
                value=expected_cell,
            ):
                issues.append(
                    "Source verification value missing: "
                    f"`{source_label}` does not contain `{value_name}[{cell_key}]` = "
                    f"{_format_verification_value(expected_cell)}."
                )
        if issues and _source_text_contains_table_value_multiset(
            normalized_text,
            expected_value.values(),
        ):
            return []
        return issues

    if _source_text_contains_scalar_value(normalized_text, expected_value):
        return []
    return [
        "Source verification value missing: "
        f"`{source_label}` does not contain `{value_name}` = "
        f"{_format_verification_value(expected_value)}."
    ]


@functools.lru_cache(maxsize=512)
def _fetch_corpus_source_text(citation_path: str) -> str | None:
    """Fetch a corpus.provisions body by exact citation path.

    Local corpus artifacts are preferred so encoder and CI runs verify against
    normalized source text without re-reading original PDFs or HTML pages.
    Supabase is the network fallback for environments without local artifacts.
    """
    local_text = _fetch_local_corpus_source_text(citation_path)
    if local_text is not None:
        return local_text
    return _fetch_supabase_corpus_source_text(citation_path)


@functools.lru_cache(maxsize=512)
def _fetch_local_corpus_source_text(citation_path: str) -> str | None:
    normalized_path = citation_path.strip().strip("/")
    if not normalized_path:
        return None

    for provisions_root in _local_corpus_provisions_roots():
        for provision_file in _candidate_local_corpus_provision_files(
            provisions_root,
            normalized_path,
        ):
            source_text = _read_local_corpus_provision_file(
                provision_file,
                normalized_path,
            )
            if source_text is not None:
                return source_text
    return None


def _local_corpus_provisions_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for env_name in ("AXIOM_CORPUS_ARTIFACT_ROOT", "AXIOM_CORPUS_REPO"):
        raw_root = os.environ.get(env_name)
        if raw_root:
            roots.append(Path(raw_root).expanduser())

    with contextlib.suppress(OSError):
        cwd = Path.cwd().resolve()
        for base in (cwd, *cwd.parents):
            roots.extend(
                (
                    base,
                    base / "axiom-corpus",
                    base / "TheAxiomFoundation" / "axiom-corpus",
                    base.parent / "axiom-corpus",
                )
            )

    with contextlib.suppress(RuntimeError, OSError):
        roots.append(Path.home() / "TheAxiomFoundation" / "axiom-corpus")

    provisions_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for candidate in (
            root,
            root / "provisions",
            root / "data" / "corpus",
            root / "data" / "corpus" / "provisions",
        ):
            provisions_root = (
                candidate
                if candidate.name == "provisions"
                else candidate / "provisions"
            )
            with contextlib.suppress(OSError):
                resolved = provisions_root.resolve()
                if resolved.is_dir() and resolved not in seen:
                    seen.add(resolved)
                    provisions_roots.append(resolved)
    return tuple(provisions_roots)


def _candidate_local_corpus_provision_files(
    provisions_root: Path,
    citation_path: str,
) -> tuple[Path, ...]:
    parts = citation_path.split("/")
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_files(base: Path) -> None:
        for path in sorted(base.glob("*.jsonl")):
            with contextlib.suppress(OSError):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)

    if len(parts) >= 2:
        add_files(provisions_root / parts[0] / parts[1])
    if not candidates:
        for path in sorted(provisions_root.rglob("*.jsonl")):
            with contextlib.suppress(OSError):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
    return tuple(candidates)


def _read_local_corpus_provision_file(
    provision_file: Path,
    citation_path: str,
) -> str | None:
    try:
        lines = provision_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("citation_path") != citation_path:
            continue
        body = record.get("body")
        return str(body) if body is not None else None
    return None


@functools.lru_cache(maxsize=512)
def _fetch_supabase_corpus_source_text(citation_path: str) -> str | None:
    """Fetch current corpus source text by exact citation path from Supabase."""
    supabase_url = os.environ.get(
        "AXIOM_SUPABASE_URL", DEFAULT_AXIOM_SUPABASE_URL
    ).rstrip("/")
    anon_key = (
        os.environ.get("SUPABASE_ANON_KEY")
        or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
        or DEFAULT_AXIOM_SUPABASE_ANON_KEY
    )
    params = urllib.parse.urlencode(
        {
            "select": "body",
            "citation_path": f"eq.{citation_path}",
            "limit": "1",
        }
    )
    request = urllib.request.Request(
        f"{supabase_url}/rest/v1/current_provisions?{params}",
        headers={
            "apikey": anon_key,
            "Authorization": f"Bearer {anon_key}",
            "Accept-Profile": "corpus",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read())
    except (
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ):
        return None
    if not isinstance(data, list) or not data:
        return None
    body = data[0].get("body") if isinstance(data[0], dict) else None
    return str(body) if body is not None else None


def _normalize_source_verification_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(",", "")).strip()


def _source_text_contains_indexed_value(text: str, *, index: str, value: Any) -> bool:
    value_texts = _source_verification_numeric_texts(value)
    if not value_texts:
        return False
    index_text = re.escape(str(index).replace(",", ""))
    return any(
        re.search(
            rf"(?<!\d){index_text}\s+\$?{re.escape(value_text)}(?!\d)",
            text,
            re.IGNORECASE,
        )
        for value_text in value_texts
    )


def _source_text_contains_scalar_value(text: str, value: Any) -> bool:
    value_texts = _source_verification_numeric_texts(value)
    if not value_texts:
        return str(value).strip() in text
    return any(
        re.search(
            rf"(?:\$|\+)?{re.escape(value_text)}(?!\d)",
            text,
            re.IGNORECASE,
        )
        for value_text in value_texts
    )


def _source_text_contains_table_value_multiset(
    text: str,
    values: Iterable[Any],
) -> bool:
    expected_counts: Counter[str] = Counter()
    for value in values:
        value_texts = _source_verification_numeric_texts(value)
        if not value_texts:
            return False
        expected_counts["\0".join(value_texts)] += 1

    return all(
        max(
            _source_text_value_occurrence_count(text, value_text)
            for value_text in value_texts_key.split("\0")
        )
        >= expected_count
        for value_texts_key, expected_count in expected_counts.items()
    )


def _source_text_value_occurrence_count(text: str, value_text: str) -> int:
    value_pattern = re.escape(value_text)
    return len(re.findall(rf"(?:\$|\+)?{value_pattern}(?!\d)", text, re.IGNORECASE))


def _source_verification_numeric_text(value: Any) -> str | None:
    texts = _source_verification_numeric_texts(value)
    return texts[0] if texts else None


def _source_verification_numeric_texts(value: Any) -> tuple[str, ...]:
    numeric = _numeric_rule_value(value)
    if numeric is None:
        return ()
    raw, number = numeric
    texts: list[str] = []
    if float(number).is_integer():
        texts.append(str(int(number)))
    else:
        texts.append(raw.replace(",", ""))
    with contextlib.suppress(TypeError, ValueError):
        numeric_float = float(number)
        if 0 < abs(numeric_float) <= 1:
            percent = numeric_float * 100
            percent_text = (
                str(int(percent))
                if float(percent).is_integer()
                else f"{percent:g}"
            )
            texts.append(percent_text)
            texts.append(f"{percent_text}%")
            texts.append(f"{percent_text} percent")
            texts.append(f"{percent_text}-percent")
            texts.append(f"{percent_text} per cent")
            if float(percent).is_integer():
                percent_word = _CARDINAL_VALUE_WORDS.get(int(percent))
                if percent_word:
                    texts.append(f"{percent_word} percent")
                    texts.append(f"{percent_word} per cent")
    return tuple(dict.fromkeys(texts))


@dataclass(frozen=True)
class _RuleSpecTargetRef:
    """Parsed canonical RuleSpec target reference."""

    prefix: str
    repo_name: str
    relative_path: Path
    symbol: str | None


def find_source_relation_issues(
    content: str,
    *,
    policy_repo_path: Path | None = None,
) -> list[str]:
    """Validate non-executable source-relation records."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").lower() != "source_relation":
            continue
        name = str(rule.get("name") or "<unknown>")
        source_relation = rule.get("source_relation")
        target = ""
        target_ref: _RuleSpecTargetRef | None = None
        relation_type = ""
        if isinstance(source_relation, dict):
            relation_type = str(source_relation.get("type") or "").strip().lower()
        if relation_type != "restates":
            continue
        if (
            not isinstance(source_relation, dict)
            or not str(source_relation.get("target") or "").strip()
        ):
            issues.append(
                "Source relation target required: "
                f"{name} must declare `source_relation.target` pointing to the canonical RuleSpec rule."
            )
        else:
            target = str(source_relation.get("target") or "").strip()
            target_ref = _parse_rulespec_target(target)
            if target_ref is None:
                issues.append(
                    "Source relation target invalid: "
                    f"{name} uses `{target}`, expected `<jurisdiction>:<path>#<rule>`."
                )
            elif target_ref.symbol is None:
                issues.append(
                    "Source relation target rule required: "
                    f"{name} must point to a specific RuleSpec rule with `#rule_name`."
                )
        if rule.get("versions"):
            issues.append(
                "Source relation must be non-executable: "
                f"{name} should not declare `versions`; use the canonical target for formulas and values."
            )
        verification = rule.get("verification")
        if (
            target
            and target_ref is not None
            and target_ref.symbol is not None
            and isinstance(verification, dict)
            and isinstance(verification.get("values"), dict)
        ):
            issues.extend(
                _find_source_relation_value_verification_issues(
                    name=name,
                    target=target,
                    target_ref=target_ref,
                    expected_values=verification["values"],
                    policy_repo_path=policy_repo_path,
                )
            )
    return issues


def _find_source_relation_value_verification_issues(
    *,
    name: str,
    target: str,
    target_ref: _RuleSpecTargetRef,
    expected_values: dict[Any, Any],
    policy_repo_path: Path | None,
) -> list[str]:
    """Compare a source relation's expected values against its canonical target file."""
    target_file = _resolve_rulespec_target_file(target_ref, policy_repo_path)
    if target_file is None:
        return [
            "Source relation verification target unavailable: "
            f"{name} points to `{target}`, but repository `{target_ref.repo_name}` "
            "was not found."
        ]

    target_values, target_symbols, load_issue = _extract_source_relation_target_values(
        target_file
    )
    if load_issue is not None:
        return [
            "Source relation verification target invalid: "
            f"{name} points to `{target}`, but {load_issue}"
        ]

    issues: list[str] = []
    if target_ref.symbol and target_ref.symbol not in target_symbols:
        issues.append(
            "Source relation target rule missing: "
            f"{name} points to `{target}`, but `{target_ref.symbol}` is not defined "
            f"in {target_ref.relative_path}."
        )

    for value_name, expected_value in expected_values.items():
        value_key = str(value_name)
        if value_key not in target_values:
            issues.append(
                "Source relation verification target missing value: "
                f"{name} expects `{value_key}` but `{target}` does not define it."
            )
            continue
        actual_value = target_values[value_key]
        issues.extend(
            _compare_source_relation_verification_value(
                name=name,
                target=target,
                value_name=value_key,
                expected_value=expected_value,
                actual_value=actual_value,
            )
        )

    return issues


def _parse_rulespec_target(target: str) -> _RuleSpecTargetRef | None:
    """Parse `us:policies/foo#rule` into a target repo and relative file path."""
    normalized = target.strip().strip("'\"")
    match = re.match(
        r"^(?P<prefix>[a-z][a-z0-9_-]*):(?P<path>[^#]+)(?:#(?P<symbol>[^#]+))?$",
        normalized,
    )
    if match is None:
        return None

    path_text = match.group("path").strip().strip("/")
    if not path_text:
        return None
    relative_path = Path(path_text)
    if relative_path.is_absolute() or any(
        part in {"", ".", ".."} for part in relative_path.parts
    ):
        return None
    if not path_text.endswith((".yaml", ".yml")):
        relative_path = Path(f"{path_text}.yaml")

    prefix = match.group("prefix")
    symbol = match.group("symbol")
    return _RuleSpecTargetRef(
        prefix=prefix,
        repo_name=f"rulespec-{prefix}",
        relative_path=relative_path,
        symbol=symbol.strip() if symbol and symbol.strip() else None,
    )


def _resolve_rulespec_target_file(
    target_ref: _RuleSpecTargetRef,
    policy_repo_path: Path | None,
) -> Path | None:
    """Resolve a canonical RuleSpec target file across sibling/CI checkouts."""
    for root in _candidate_rulespec_repo_roots(
        target_ref.repo_name,
        policy_repo_path,
    ):
        target_file = root / target_ref.relative_path
        if target_file.exists():
            return target_file
    return None


def _candidate_rulespec_repo_roots(
    repo_name: str,
    policy_repo_path: Path | None,
) -> list[Path]:
    """Return possible local roots for a canonical rules repository."""
    candidates: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        expanded = candidate.expanduser()
        if expanded.name == repo_name:
            candidates.append(expanded)
        else:
            candidates.append(expanded / repo_name)

    env_roots = os.environ.get("AXIOM_RULESPEC_REPO_ROOTS", "")
    for raw_root in env_roots.split(os.pathsep):
        if raw_root.strip():
            add(Path(raw_root.strip()))

    if policy_repo_path is not None:
        policy_root = Path(policy_repo_path).resolve()
        add(policy_root)
        add(policy_root.parent / repo_name)
        add(policy_root / "_axiom" / repo_name)
        add(policy_root.parent / "_axiom" / repo_name)

    cwd = Path.cwd()
    add(cwd / repo_name)
    add(cwd / "_axiom" / repo_name)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def _extract_source_relation_target_values(
    target_file: Path,
) -> tuple[dict[str, Any], set[str], str | None]:
    """Extract scalar/table parameter values from a canonical RuleSpec file."""
    try:
        payload = yaml.safe_load(target_file.read_text())
    except (OSError, yaml.YAMLError, ValueError) as exc:
        return {}, set(), f"{target_file} could not be read as RuleSpec YAML: {exc}"
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return {}, set(), f"{target_file} is not a RuleSpec v1 file"
    return _extract_rulespec_parameter_values(payload, source_label=str(target_file))


def _extract_rulespec_parameter_values(
    payload: dict[str, Any],
    *,
    source_label: str = "RuleSpec payload",
) -> tuple[dict[str, Any], set[str], str | None]:
    """Extract scalar/table parameter values from a RuleSpec payload."""
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return {}, set(), f"{source_label} has no `rules` list"

    values: dict[str, Any] = {}
    symbols: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        raw_name = rule.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        name = raw_name.strip()
        symbols.add(name)
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            table_values = version.get("values")
            if isinstance(table_values, dict):
                values[name] = {str(key): value for key, value in table_values.items()}
                continue
            scalar = _numeric_rule_value(version.get("formula"))
            if scalar is not None:
                values[name] = scalar[0]
    return values, symbols, None


def _compare_source_relation_verification_value(
    *,
    name: str,
    target: str,
    value_name: str,
    expected_value: Any,
    actual_value: Any,
) -> list[str]:
    """Return mismatch issues for one verified scalar or table value."""
    if isinstance(expected_value, dict):
        if not isinstance(actual_value, dict):
            return [
                "Source relation verification mismatch: "
                f"{name} expects `{value_name}` to be a table, but `{target}` "
                "defines a scalar."
            ]
        issues: list[str] = []
        for raw_key, expected_cell in expected_value.items():
            cell_key = str(raw_key)
            if cell_key not in actual_value:
                issues.append(
                    "Source relation verification target missing value: "
                    f"{name} expects `{value_name}[{cell_key}]` but `{target}` "
                    "does not define it."
                )
                continue
            actual_cell = actual_value[cell_key]
            if not _verification_values_equal(expected_cell, actual_cell):
                issues.append(
                    "Source relation verification mismatch: "
                    f"{name} expects `{value_name}[{cell_key}]` = "
                    f"{_format_verification_value(expected_cell)}, but `{target}` "
                    f"has {_format_verification_value(actual_cell)}."
                )
        return issues

    if isinstance(actual_value, dict):
        return [
            "Source relation verification mismatch: "
            f"{name} expects `{value_name}` to be a scalar, but `{target}` "
            "defines a table."
        ]
    if _verification_values_equal(expected_value, actual_value):
        return []
    return [
        "Source relation verification mismatch: "
        f"{name} expects `{value_name}` = "
        f"{_format_verification_value(expected_value)}, but `{target}` has "
        f"{_format_verification_value(actual_value)}."
    ]


def _verification_values_equal(expected_value: Any, actual_value: Any) -> bool:
    """Compare verification scalar values, allowing numeric text/int equivalence."""
    expected_numeric = _numeric_rule_value(expected_value)
    actual_numeric = _numeric_rule_value(actual_value)
    if expected_numeric is not None and actual_numeric is not None:
        return math.isclose(expected_numeric[1], actual_numeric[1], abs_tol=1e-12)
    return str(expected_value).strip() == str(actual_value).strip()


def _format_verification_value(value: Any) -> str:
    """Format a verification value for validation messages."""
    return repr(value)


def _embedded_integer_scale_selector(formula: str) -> str | None:
    normalized = re.sub(r"\s+", " ", formula)
    for match in re.finditer(r"\bmatch\s+([A-Za-z_][\w.]*)\s*:", normalized):
        numeric_arms = re.findall(
            r"=>\s*-?(?:\d{2,}|\d+\.\d+)",
            normalized[match.end() :],
        )
        if len(numeric_arms) >= 2:
            return match.group(1)
    return None


def numeric_value_is_grounded(value: float, source_numbers: set[float]) -> bool:
    """Return true when a generated number is present in extracted source numbers."""
    for source_value in source_numbers:
        if math.isclose(value, source_value, rel_tol=0, abs_tol=1e-12):
            return True
        if 0 < abs(value) <= 1 and math.isclose(
            value * 100,
            source_value,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            return True
    return False


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
    prompt_sha256: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)


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
            "rulespec_reviewer",
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

    @property
    def ci_pass(self) -> bool:
        """Check if CI passed."""
        return self.results.get("ci", ValidationResult("", False)).passed


def _rulespec_public_item_key(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    item_id = str(item.get("id") or "").strip()
    if item_id:
        return item_id
    return str(item.get("name") or "").strip()


def _rulespec_item_friendly_name_and_legal_id(item: Any) -> tuple[str, str] | None:
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "").strip()
    item_id = str(item.get("id") or "").strip()
    if not name or not item_id or name == item_id:
        return None
    return name, item_id


_RULESPEC_ABSOLUTE_REFERENCE = re.compile(r"^[a-z][a-z0-9_-]*:[^\s]+$")
_RULESPEC_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_RULESPEC_FORMULA_BUILTINS = {
    "False",
    "None",
    "True",
    "abs",
    "all",
    "and",
    "any",
    "ceil",
    "count",
    "count_where",
    "else",
    "floor",
    "if",
    "len",
    "max",
    "min",
    "not",
    "or",
    "round",
    "sum",
    "sum_where",
    "true",
    "false",
}


@dataclass(frozen=True)
class _RuleSpecReferenceSummary:
    """Symbols a RuleSpec file can legitimately expose to companion tests."""

    derived: frozenset[str]
    parameters: frozenset[str]
    relations: frozenset[str]
    input_slots: frozenset[str]


def _rulespec_program_has_legal_ids(compiled_payload: dict[str, Any]) -> bool:
    program = (
        compiled_payload.get("program") if isinstance(compiled_payload, dict) else {}
    )
    if not isinstance(program, dict):
        return False
    for collection in ("derived", "parameters"):
        for item in program.get(collection, []):
            if _rulespec_item_friendly_name_and_legal_id(item) is not None:
                return True
    return False


def _rulespec_module_target(compiled_payload: dict[str, Any]) -> str | None:
    program = (
        compiled_payload.get("program") if isinstance(compiled_payload, dict) else {}
    )
    if not isinstance(program, dict):
        return None
    for collection in ("derived", "parameters"):
        for item in program.get(collection, []):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            if "#" in item_id:
                return item_id.split("#", 1)[0]
    return None


def _rulespec_runtime_name_from_absolute_test_reference(reference: str) -> str:
    if not _RULESPEC_ABSOLUTE_REFERENCE.match(reference) or "#" not in reference:
        return ""
    fragment = reference.split("#", 1)[1].strip()
    if ".input." in fragment:
        return fragment.rsplit(".input.", 1)[1].strip()
    for prefix in ("input.", "relation."):
        if fragment.startswith(prefix):
            return fragment.removeprefix(prefix).strip()
    return fragment


def _rulespec_formula_identifiers(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()

    identifiers: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if isinstance(formula, str):
                identifiers.update(_RULESPEC_IDENTIFIER.findall(formula))
    return identifiers - _RULESPEC_FORMULA_BUILTINS


def _rulespec_reference_summary(target_file: Path) -> _RuleSpecReferenceSummary:
    try:
        payload = yaml.safe_load(target_file.read_text()) or {}
    except (OSError, yaml.YAMLError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    derived: set[str] = set()
    parameters: set[str] = set()
    relations: set[str] = set()
    rules = payload.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if not name:
                continue
            kind = str(rule.get("kind") or "").strip().lower()
            if kind == "parameter":
                parameters.add(name)
            elif kind == "derived":
                derived.add(name)
            elif kind == "data_relation":
                relations.add(name)

    formula_identifiers = _rulespec_formula_identifiers(payload)
    if isinstance(rules, list):
        for rule in rules:
            if (
                isinstance(rule, dict)
                and str(rule.get("kind") or "").lower() == "parameter"
            ):
                formula_identifiers.update(
                    _indexed_by_input_names(rule.get("indexed_by"))
                )
    input_slots = formula_identifiers - derived - parameters - relations
    return _RuleSpecReferenceSummary(
        derived=frozenset(derived),
        parameters=frozenset(parameters),
        relations=frozenset(relations),
        input_slots=frozenset(input_slots),
    )


def _rulespec_absolute_test_reference_issue(
    reference: str,
    *,
    label: str,
    policy_repo_path: Path | None,
    allow_input_slots: bool,
    allow_relations: bool,
    allow_outputs: bool,
) -> str | None:
    target_ref = _parse_rulespec_target(reference)
    if target_ref is None or not target_ref.symbol:
        return f"{label} `{reference}` must be an absolute legal RuleSpec reference."

    target_file = _resolve_rulespec_target_file(target_ref, policy_repo_path)
    if target_file is None:
        return (
            f"{label} `{reference}` points to a RuleSpec file that could not be "
            f"resolved: {target_ref.repo_name}/{target_ref.relative_path.as_posix()}."
        )

    summary = _rulespec_reference_summary(target_file)
    fragment = target_ref.symbol.strip()

    if ".input." in fragment:
        owner, input_slot = fragment.rsplit(".input.", 1)
        input_slot = input_slot.strip()
        owner = owner.strip()
        if owner and owner not in summary.derived and owner not in summary.parameters:
            return (
                f"{label} `{reference}` does not resolve to a derived rule or "
                f"parameter owner in {target_ref.relative_path.as_posix()}."
            )
        if input_slot in summary.input_slots:
            if allow_input_slots:
                return None
            return f"{label} `{reference}` resolves to an input slot, which is not allowed here."
        return (
            f"{label} `{reference}` does not resolve to an input slot in "
            f"{target_ref.relative_path.as_posix()}."
        )

    if fragment.startswith("input."):
        input_slot = fragment.removeprefix("input.").strip()
        if input_slot in summary.input_slots:
            if allow_input_slots:
                return None
            return f"{label} `{reference}` resolves to an input slot, which is not allowed here."
        return (
            f"{label} `{reference}` does not resolve to an input slot in "
            f"{target_ref.relative_path.as_posix()}."
        )

    if fragment.startswith("relation."):
        relation = fragment.removeprefix("relation.").strip()
        if relation in summary.relations:
            if allow_relations:
                return None
            return f"{label} `{reference}` resolves to a relation, which is not allowed here."
        return (
            f"{label} `{reference}` does not resolve to a declared relation in "
            f"{target_ref.relative_path.as_posix()}."
        )

    if fragment in summary.derived or fragment in summary.parameters:
        if allow_outputs:
            return None
        return f"{label} `{reference}` resolves to a derived rule or parameter, which is not allowed here."
    allowed_kinds = []
    if allow_input_slots:
        allowed_kinds.append("input slot")
    if allow_relations:
        allowed_kinds.append("relation")
    if allow_outputs:
        allowed_kinds.extend(["derived rule", "parameter"])
    allowed = ", ".join(allowed_kinds[:-1])
    if len(allowed_kinds) > 1:
        allowed = f"{allowed}, or {allowed_kinds[-1]}"
    elif allowed_kinds:
        allowed = allowed_kinds[0]
    else:
        allowed = "allowed RuleSpec target"
    article = "an" if allowed.startswith(("input", "allowed")) else "a"
    return (
        f"{label} `{reference}` does not resolve to {article} {allowed} in "
        f"{target_ref.relative_path.as_posix()}."
    )


def _rulespec_test_input_key_suggestion(
    friendly_name: str,
    *,
    legal_ids_by_friendly_name: dict[str, list[str]],
    module_target: str | None,
) -> str:
    legal_ids = legal_ids_by_friendly_name.get(friendly_name)
    if legal_ids:
        if len(legal_ids) == 1:
            return f"; use `{legal_ids[0]}`"
        legal_id_list = ", ".join(f"`{item}`" for item in legal_ids)
        return f"; use one of {legal_id_list}"
    if module_target:
        return (
            f"; use `{module_target}#input.{friendly_name}` for a local fact slot "
            "or the upstream absolute RuleSpec target for an imported legal value"
        )
    return ""


def _rulespec_runtime_name_for_test_input_key(
    input_key: str,
    *,
    label: str,
    require_legal_input_keys: bool,
    legal_ids_by_friendly_name: dict[str, list[str]],
    module_target: str | None,
    policy_repo_path: Path | None,
    allow_input_slots: bool,
    allow_relations: bool,
    allow_outputs: bool,
) -> str:
    if _RULESPEC_ABSOLUTE_REFERENCE.match(input_key):
        resolution_issue = _rulespec_absolute_test_reference_issue(
            input_key,
            label=label,
            policy_repo_path=policy_repo_path,
            allow_input_slots=allow_input_slots,
            allow_relations=allow_relations,
            allow_outputs=allow_outputs,
        )
        if resolution_issue:
            raise ValueError(resolution_issue)
        runtime_name = _rulespec_runtime_name_from_absolute_test_reference(input_key)
        if runtime_name:
            return runtime_name
        raise ValueError(
            f"{label} `{input_key}` must include a fragment naming the runtime slot."
        )

    if require_legal_input_keys:
        suggestion = _rulespec_test_input_key_suggestion(
            input_key,
            legal_ids_by_friendly_name=legal_ids_by_friendly_name,
            module_target=module_target,
        )
        raise ValueError(
            f"{label} `{input_key}` must use an absolute legal RuleSpec id "
            f"instead of the friendly name{suggestion}."
        )

    return input_key


class ValidatorPipeline:
    """Runs validators in 3 tiers with session event logging."""

    def __init__(
        self,
        policy_repo_path: Path,
        axiom_rules_path: Path,
        enable_oracles: bool = True,
        oracle_validators: tuple[str, ...] | None = None,
        max_workers: int = 4,
        encoding_db: Optional[EncodingDB] = None,
        session_id: Optional[str] = None,
        policyengine_country: str = "auto",
        policyengine_rule_hint: str | None = None,
        require_policy_proofs: bool = False,
        enforce_repository_layout: bool = True,
    ):
        self.policy_repo_path = Path(policy_repo_path)
        self.axiom_rules_path = Path(axiom_rules_path)
        self.enable_oracles = enable_oracles
        self.oracle_validators = oracle_validators or ("policyengine", "taxsim")
        self.max_workers = max_workers
        self.encoding_db = encoding_db
        self.session_id = session_id
        self.policyengine_country = policyengine_country
        self.policyengine_rule_hint = policyengine_rule_hint
        self.require_policy_proofs = require_policy_proofs
        self.enforce_repository_layout = enforce_repository_layout
        self.policyengine_registry = load_policyengine_registry()

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
        """Build an env that prefers the configured Axiom rules engine checkout."""
        env = dict(os.environ)
        rules_src = self.axiom_rules_path / "src"
        if rules_src.exists():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{rules_src}{os.pathsep}{existing}" if existing else str(rules_src)
            )
        return env

    def _rulespec_compile_env(self) -> dict[str, str]:
        """Build an env that can resolve canonical RuleSpec repo imports."""
        env = self._pythonpath_env()
        roots = [self.policy_repo_path, self.policy_repo_path.parent]
        existing_roots = env.get("AXIOM_RULESPEC_REPO_ROOTS", "")
        if existing_roots:
            roots.extend(
                Path(root) for root in existing_roots.split(os.pathsep) if root
            )
        deduped_roots: list[str] = []
        seen: set[str] = set()
        for root in roots:
            raw = str(root)
            if raw and raw not in seen:
                seen.add(raw)
                deduped_roots.append(raw)
        env["AXIOM_RULESPEC_REPO_ROOTS"] = os.pathsep.join(deduped_roots)
        return env

    def validate(
        self, rulespec_file: Path, skip_reviewers: bool = False
    ) -> PipelineResult:
        """Run 4-tier validation on a RuleSpec file.

        Tiers run in order:
        0. Compile check - can the file compile to engine IR?
        1. CI checks (instant) - parse, lint, companion tests, structural validation
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
            f"Starting compilation check for {rulespec_file.name}",
        )
        compile_start = time.time()
        results["compile"] = self._run_compile_check(rulespec_file)
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
            "validation_ci_start", f"Starting CI validation for {rulespec_file.name}"
        )
        ci_start = time.time()
        try:
            results["ci"] = self._run_ci(rulespec_file)
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

            available_oracle_validators = {
                "policyengine": lambda: self._run_policyengine(rulespec_file),
                "taxsim": lambda: self._run_taxsim(rulespec_file),
            }
            oracle_validators = {
                name: available_oracle_validators[name]
                for name in self.oracle_validators
                if name in available_oracle_validators
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
                            "details": results[name].details,
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
                "rulespec_reviewer": lambda: self._run_reviewer(
                    "rulespec-reviewer", rulespec_file, oracle_context
                ),
                "formula_reviewer": lambda: self._run_reviewer(
                    "Formula Reviewer", rulespec_file, oracle_context
                ),
                "parameter_reviewer": lambda: self._run_reviewer(
                    "Parameter Reviewer", rulespec_file, oracle_context
                ),
                "integration_reviewer": lambda: self._run_reviewer(
                    "Integration Reviewer", rulespec_file, oracle_context
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
                        k: results[k].score
                        for k in llm_validators.keys()
                        if k in results
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

    def _is_rulespec_file(self, rules_file: Path) -> bool:
        """Return true for current RuleSpec files."""
        if rules_file.suffix not in {".yaml", ".yml"} or rules_file.name.endswith(
            ".test.yaml"
        ):
            return False
        try:
            payload = yaml.safe_load(rules_file.read_text())
        except (OSError, yaml.YAMLError, ValueError):
            return False
        return isinstance(payload, dict) and payload.get("format") == "rulespec/v1"

    def _rulespec_test_path(self, rules_file: Path) -> Path:
        """Return the companion RuleSpec test file path."""
        return rules_file.with_name(f"{rules_file.stem}.test.yaml")

    def _axiom_rules_binary(self) -> Path:
        """Resolve the local Axiom rules engine CLI binary."""
        candidates = [
            self.axiom_rules_path / "target" / "debug" / "axiom-rules-engine",
            self.axiom_rules_path / "target" / "release" / "axiom-rules-engine",
            self.axiom_rules_path / "axiom-rules-engine",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if resolved := shutil.which("axiom-rules-engine"):
            return Path(resolved)
        raise FileNotFoundError(
            f"axiom-rules-engine binary not found under {self.axiom_rules_path} or on PATH"
        )

    def _compile_rulespec_to_artifact(
        self,
        rules_file: Path,
        output_path: Path,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
        """Compile RuleSpec YAML to an Axiom rules engine artifact JSON file."""
        binary = self._axiom_rules_binary()
        result = subprocess.run(
            [
                str(binary),
                "compile",
                "--program",
                str(rules_file),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(self.axiom_rules_path) if self.axiom_rules_path.exists() else None,
            env=self._rulespec_compile_env(),
        )
        if result.returncode != 0:
            return result, None
        return result, json.loads(output_path.read_text())

    def _rulespec_compile_success_output(self, payload: Any) -> str:
        """Return a concise successful compile summary for validator output."""
        program = payload.get("program") if isinstance(payload, dict) else {}
        if not isinstance(program, dict):
            program = {}
        rule_count = sum(
            len(program.get(key) or ())
            for key in ("parameters", "derived", "relations")
        )
        return f"Successfully compiled {rule_count} RuleSpec rule(s) with the Axiom rules engine"

    def _run_rulespec_compile_check(self, rules_file: Path) -> ValidationResult:
        """Compile RuleSpec YAML through the Axiom rules engine."""
        start = time.time()
        issues: list[str] = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "compiled.json"
                result, payload = self._compile_rulespec_to_artifact(
                    rules_file, output_path
                )
                if result.returncode != 0:
                    detail = result.stderr.strip() or result.stdout.strip()
                    issues.append(f"Axiom rules engine compile failed: {detail}")
                    return ValidationResult(
                        validator_name="compile",
                        passed=False,
                        issues=issues,
                        duration_ms=int((time.time() - start) * 1000),
                        error=issues[0],
                        raw_output=result.stdout + result.stderr,
                    )
            return ValidationResult(
                validator_name="compile",
                passed=True,
                issues=[],
                duration_ms=int((time.time() - start) * 1000),
                raw_output=self._rulespec_compile_success_output(payload),
            )
        except Exception as exc:
            issues.append(f"Axiom rules engine compile failed: {exc}")
            return ValidationResult(
                validator_name="compile",
                passed=False,
                issues=issues,
                duration_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )

    def _run_compile_check(self, rulespec_file: Path) -> ValidationResult:
        """Tier 0: Compile check against the Axiom rules engine RuleSpec."""
        if not self._is_rulespec_file(rulespec_file):
            return ValidationResult(
                validator_name="compile",
                passed=False,
                issues=["RuleSpec YAML artifacts are required."],
                error="RuleSpec YAML artifacts are required",
            )

        return self._run_rulespec_compile_check(rulespec_file)

    def _coerce_rulespec_period(self, value: Any) -> dict[str, Any]:
        """Coerce compact `.test.yaml` period shorthands to engine JSON."""
        if isinstance(value, dict):
            period = {
                key: (item.isoformat() if isinstance(item, date) else item)
                for key, item in value.items()
            }
            if period.get("period_kind") == "year":
                period["period_kind"] = "tax_year"
            required = {"period_kind", "start", "end"}
            missing = sorted(required - set(period))
            if missing:
                raise ValueError(
                    "period mapping missing required field(s): " + ", ".join(missing)
                )
            if period["period_kind"] not in {
                "month",
                "benefit_week",
                "tax_year",
                "custom",
            }:
                raise ValueError(f"unsupported period_kind: {period['period_kind']!r}")
            for key in ("start", "end"):
                try:
                    date.fromisoformat(str(period[key]))
                except ValueError as exc:
                    raise ValueError(
                        f"period {key} must be an ISO date, got {period[key]!r}"
                    ) from exc
            if period["period_kind"] == "custom" and not period.get("name"):
                raise ValueError("custom period mappings must include name")
            return period
        if isinstance(value, int):
            raise ValueError(
                "bare year periods are ambiguous; use an explicit period mapping "
                "with period_kind/start/end"
            )
        if isinstance(value, date):
            day = value.isoformat()
            return {
                "period_kind": "custom",
                "name": "day",
                "start": day,
                "end": day,
            }
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"\d{4}", stripped):
                raise ValueError(
                    "bare year periods are ambiguous; use an explicit period "
                    "mapping with period_kind/start/end"
                )
            if re.fullmatch(r"\d{4}-\d{2}", stripped):
                year = int(stripped[:4])
                month = int(stripped[5:])
                return {
                    "period_kind": "month",
                    "start": date(year, month, 1).isoformat(),
                    "end": date(year, month, monthrange(year, month)[1]).isoformat(),
                }
        raise ValueError(f"unsupported period shorthand: {value!r}")

    def _rulespec_case_query_entity_id(
        self,
        case: dict[str, Any],
        query_entity: str,
        index: int,
    ) -> str:
        """Pick a stable entity id for a compact RuleSpec test case."""
        entity_key = f"{self._snake_case(query_entity)}_id"
        for key in ("entity_id", "id", entity_key):
            if key in case:
                return str(case[key])
        for key, value in case.items():
            if key.endswith("_id") and not isinstance(value, (dict, list)):
                return str(value)
        return f"case-{index}"

    def _snake_case(self, value: str) -> str:
        """Convert a PascalCase/CamelCase label to snake_case."""
        value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
        return value.replace("-", "_").lower()

    def _related_entity_from_relation(self, relation_name: str) -> str:
        """Infer a readable related-entity label for relation test inputs."""
        head = relation_name.split("_of_", 1)[0]
        return (
            "".join(part.capitalize() for part in head.split("_") if part) or "Related"
        )

    def _rulespec_scalar_value(self, value: Any) -> dict[str, Any]:
        """Coerce Python/YAML scalar values to Axiom rules engine ScalarValueSpec JSON."""
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int):
            return {"kind": "integer", "value": value}
        if isinstance(value, float):
            return {"kind": "decimal", "value": str(value)}
        if isinstance(value, Decimal):
            return {"kind": "decimal", "value": str(value)}
        if isinstance(value, date):
            return {"kind": "date", "value": value.isoformat()}
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
                return {"kind": "date", "value": stripped}
            if re.fullmatch(r"-?\d+", stripped):
                return {"kind": "integer", "value": int(stripped)}
            if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
                return {"kind": "decimal", "value": stripped}
            return {"kind": "text", "value": value}
        raise ValueError(f"unsupported scalar test value {value!r}")

    def _rulespec_expected_scalar_value(self, value: Any) -> dict[str, Any]:
        """Coerce expected output YAML values without interpreting strings."""
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int):
            return {"kind": "integer", "value": value}
        if isinstance(value, float):
            return {"kind": "decimal", "value": str(value)}
        if isinstance(value, Decimal):
            return {"kind": "decimal", "value": str(value)}
        if isinstance(value, date):
            return {"kind": "date", "value": value.isoformat()}
        if isinstance(value, str):
            return {"kind": "text", "value": value}
        raise ValueError(f"unsupported expected scalar value {value!r}")

    def _build_rulespec_dataset(
        self,
        case_input: Any,
        *,
        period: dict[str, Any],
        query_entity: str,
        query_entity_id: str,
        require_legal_input_keys: bool = False,
        legal_ids_by_friendly_name: dict[str, list[str]] | None = None,
        module_target: str | None = None,
    ) -> dict[str, Any]:
        """Build an Axiom rules engine dataset from compact RuleSpec test inputs."""
        if case_input in (None, ""):
            case_input = {}
        if not isinstance(case_input, dict):
            raise ValueError("input must be a mapping")

        interval = {"start": period["start"], "end": period["end"]}
        inputs: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        legal_ids_by_friendly_name = legal_ids_by_friendly_name or {}

        for name, value in case_input.items():
            input_key = str(name)
            if isinstance(value, list):
                relation_name = _rulespec_runtime_name_for_test_input_key(
                    input_key,
                    label="relation input",
                    require_legal_input_keys=require_legal_input_keys,
                    legal_ids_by_friendly_name=legal_ids_by_friendly_name,
                    module_target=module_target,
                    policy_repo_path=self.policy_repo_path,
                    allow_input_slots=False,
                    allow_relations=True,
                    allow_outputs=False,
                )
                relation_request_name = (
                    input_key
                    if _RULESPEC_ABSOLUTE_REFERENCE.match(input_key)
                    else relation_name
                )
                related_entity = self._related_entity_from_relation(relation_name)
                for item_index, item in enumerate(value, 1):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"relation `{name}` item #{item_index} must be a mapping"
                        )
                    related_id = str(
                        item.get("id")
                        or item.get("entity_id")
                        or f"{query_entity_id}-{name}-{item_index}"
                    )
                    relations.append(
                        {
                            "name": relation_request_name,
                            "tuple": [related_id, query_entity_id],
                            "interval": interval,
                        }
                    )
                    for child_name, child_value in item.items():
                        if child_name in {"id", "entity_id"}:
                            continue
                        if isinstance(child_value, (dict, list)):
                            raise ValueError(
                                f"relation `{name}` input `{child_name}` must be scalar"
                            )
                        child_input_name = _rulespec_runtime_name_for_test_input_key(
                            str(child_name),
                            label="input",
                            require_legal_input_keys=require_legal_input_keys,
                            legal_ids_by_friendly_name=legal_ids_by_friendly_name,
                            module_target=module_target,
                            policy_repo_path=self.policy_repo_path,
                            allow_input_slots=True,
                            allow_relations=False,
                            allow_outputs=True,
                        )
                        child_input_key = str(child_name)
                        child_request_name = (
                            child_input_key
                            if _RULESPEC_ABSOLUTE_REFERENCE.match(child_input_key)
                            else child_input_name
                        )
                        inputs.append(
                            {
                                "name": child_request_name,
                                "entity": related_entity,
                                "entity_id": related_id,
                                "interval": interval,
                                "value": self._rulespec_scalar_value(child_value),
                            }
                        )
                continue

            if isinstance(value, dict):
                raise ValueError(f"input `{name}` must be scalar or relation list")

            input_name = _rulespec_runtime_name_for_test_input_key(
                input_key,
                label="input",
                require_legal_input_keys=require_legal_input_keys,
                legal_ids_by_friendly_name=legal_ids_by_friendly_name,
                module_target=module_target,
                policy_repo_path=self.policy_repo_path,
                allow_input_slots=True,
                allow_relations=False,
                allow_outputs=True,
            )
            input_request_name = (
                input_key
                if _RULESPEC_ABSOLUTE_REFERENCE.match(input_key)
                else input_name
            )
            inputs.append(
                {
                    "name": input_request_name,
                    "entity": query_entity,
                    "entity_id": query_entity_id,
                    "interval": interval,
                    "value": self._rulespec_scalar_value(value),
                }
            )

        return {"inputs": inputs, "relations": relations}

    def _rulespec_program_maps(
        self, compiled_payload: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return compiled derived-output and scalar-parameter maps by public key."""
        program = (
            compiled_payload.get("program")
            if isinstance(compiled_payload, dict)
            else {}
        )
        if not isinstance(program, dict):
            program = {}
        derived = {
            _rulespec_public_item_key(item): item
            for item in program.get("derived", [])
            if _rulespec_public_item_key(item)
        }
        parameters = {
            _rulespec_public_item_key(item): item
            for item in program.get("parameters", [])
            if _rulespec_public_item_key(item)
        }
        return derived, parameters

    def _rulespec_legal_ids_by_friendly_output_name(
        self, compiled_payload: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Return legal output ids keyed by local friendly name for repo-backed rules."""
        program = (
            compiled_payload.get("program")
            if isinstance(compiled_payload, dict)
            else {}
        )
        if not isinstance(program, dict):
            program = {}

        legal_ids_by_name: dict[str, set[str]] = {}
        for collection in ("derived", "parameters"):
            for item in program.get(collection, []):
                pair = _rulespec_item_friendly_name_and_legal_id(item)
                if pair is None:
                    continue
                name, item_id = pair
                legal_ids_by_name.setdefault(name, set()).add(item_id)
        return {name: sorted(item_ids) for name, item_ids in legal_ids_by_name.items()}

    def _rulespec_compiled_parameter_value(
        self,
        parameter: dict[str, Any],
        period: dict[str, Any],
        inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the live scalar value for a compiled scalar or table parameter."""
        period_start = date.fromisoformat(str(period["start"]))
        versions = parameter.get("versions") or []
        live_versions = [
            version
            for version in versions
            if isinstance(version, dict)
            and date.fromisoformat(str(version["effective_from"])) <= period_start
        ]
        if not live_versions:
            raise ValueError(
                f"parameter `{parameter.get('name')}` has no version at {period['start']}"
            )
        version = max(
            live_versions,
            key=lambda item: date.fromisoformat(str(item["effective_from"])),
        )
        values = version.get("values") or {}
        value = values.get("0", values.get(0))
        indexed_by = parameter.get("indexed_by")
        if indexed_by:
            index_names = (
                [str(indexed_by)]
                if isinstance(indexed_by, str)
                else [str(item) for item in indexed_by]
            )
            if len(index_names) != 1:
                raise ValueError(
                    f"parameter `{parameter.get('name')}` has unsupported "
                    "multi-index test output"
                )
            input_value = self._rulespec_test_input_value(inputs or {}, index_names[0])
            if input_value is None:
                raise ValueError(
                    f"parameter `{parameter.get('name')}` needs input "
                    f"`{index_names[0]}` for indexed output"
                )
            value = values.get(str(input_value), values.get(input_value))
        if not isinstance(value, dict):
            raise ValueError(
                f"parameter `{parameter.get('name')}` has no scalar value "
                "for the requested test output"
            )
        return value

    def _rulespec_decimal(self, value: Any) -> Decimal:
        """Coerce a scalar value to Decimal for numeric equality checks."""
        try:
            return Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"{value!r} is not numeric") from exc

    def _rulespec_scalar_values_equal(
        self,
        actual: dict[str, Any],
        expected: dict[str, Any],
    ) -> bool:
        """Compare Axiom rules engine scalar value specs, allowing int/decimal equality."""
        actual_kind = actual.get("kind")
        expected_kind = expected.get("kind")
        numeric = {"integer", "decimal"}
        if actual_kind in numeric and expected_kind in numeric:
            return self._rulespec_decimal(
                actual.get("value")
            ) == self._rulespec_decimal(expected.get("value"))
        if actual_kind == "bool" and expected_kind == "bool":
            return bool(actual.get("value")) == bool(expected.get("value"))
        if actual_kind != expected_kind:
            return False
        return str(actual.get("value")) == str(expected.get("value"))

    def _format_rulespec_actual_value(self, output: dict[str, Any]) -> str:
        """Format a response/parameter output value for failure messages."""
        if output.get("kind") == "judgment":
            return str(output.get("outcome"))
        value = output.get("value") if output.get("kind") == "scalar" else output
        if isinstance(value, dict):
            return self._format_rulespec_scalar_value(value)
        return str(value)

    def _format_rulespec_scalar_value(self, value: dict[str, Any]) -> str:
        """Format a scalar value spec with its kind for failure messages."""
        return f"{value.get('kind')} {value.get('value')}"

    def _compare_rulespec_output(
        self,
        *,
        case_name: str,
        output_name: str,
        expected_value: Any,
        actual_output: dict[str, Any],
    ) -> str | None:
        """Compare a single expected output; return an issue string on mismatch."""
        if actual_output.get("kind") == "judgment":
            expected = str(expected_value).strip().lower().replace("-", "_")
            if expected not in {"holds", "not_holds", "undetermined"}:
                return (
                    f"Test case `{case_name}` output `{output_name}` expected "
                    f"{expected_value!r}, but actual output is a judgment."
                )
            actual = str(actual_output.get("outcome"))
            if actual != expected:
                return (
                    f"Test case `{case_name}` output `{output_name}` expected "
                    f"{expected}, got {actual}."
                )
            return None

        actual_scalar = actual_output.get("value")
        if actual_output.get("kind") != "scalar":
            actual_scalar = actual_output
        if not isinstance(actual_scalar, dict):
            return (
                f"Test case `{case_name}` output `{output_name}` returned "
                "an unrecognised value shape."
            )
        expected_scalar = self._rulespec_expected_scalar_value(expected_value)
        if not self._rulespec_scalar_values_equal(actual_scalar, expected_scalar):
            return (
                f"Test case `{case_name}` output `{output_name}` expected "
                f"{self._format_rulespec_scalar_value(expected_scalar)}, got "
                f"{self._format_rulespec_actual_value(actual_output)}."
            )
        return None

    def _run_rulespec_derived_test_case(
        self,
        *,
        binary: Path,
        compiled_path: Path,
        case: dict[str, Any],
        case_name: str,
        case_index: int,
        period: dict[str, Any],
        output_names: list[str],
        derived_by_key: dict[str, Any],
        require_legal_input_keys: bool,
        legal_ids_by_friendly_name: dict[str, list[str]],
        module_target: str | None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Execute one compact RuleSpec test case through `run-compiled`."""
        query_entity = str(derived_by_key[output_names[0]].get("entity") or "Case")
        query_entity_id = self._rulespec_case_query_entity_id(
            case, query_entity, case_index
        )
        try:
            dataset = self._build_rulespec_dataset(
                case.get("input", {}),
                period=period,
                query_entity=query_entity,
                query_entity_id=query_entity_id,
                require_legal_input_keys=require_legal_input_keys,
                legal_ids_by_friendly_name=legal_ids_by_friendly_name,
                module_target=module_target,
            )
        except ValueError as exc:
            return None, [f"Test case `{case_name}` input invalid: {exc}"]

        request = {
            "mode": "explain",
            "dataset": dataset,
            "queries": [
                {
                    "entity_id": query_entity_id,
                    "period": period,
                    "outputs": output_names,
                }
            ],
        }
        result = subprocess.run(
            [str(binary), "run-compiled", "--artifact", str(compiled_path)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(self.axiom_rules_path) if self.axiom_rules_path.exists() else None,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            return None, [f"Test case `{case_name}` execution failed: {detail}"]
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return None, [f"Test case `{case_name}` response JSON parse failed: {exc}"]
        results = response.get("results") if isinstance(response, dict) else None
        if not isinstance(results, list) or not results:
            return None, [f"Test case `{case_name}` returned no results."]
        outputs = results[0].get("outputs")
        if not isinstance(outputs, dict):
            return None, [f"Test case `{case_name}` returned no output map."]
        return self._rulespec_outputs_by_reference(outputs), []

    def _rulespec_outputs_by_reference(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Index runtime outputs by response key and durable id only."""
        outputs_by_reference: dict[str, Any] = {}
        for output_key, output in outputs.items():
            outputs_by_reference[str(output_key)] = output
            if not isinstance(output, dict):
                continue
            reference = str(output.get("id") or "").strip()
            if reference:
                outputs_by_reference[reference] = output
        return outputs_by_reference

    def _rulespec_output_satisfies_policyengine_hint(
        self,
        output_name: str,
        *,
        derived_by_key: dict[str, Any],
        parameter_by_key: dict[str, Any],
    ) -> bool:
        if output_name == self.policyengine_rule_hint:
            return True
        item = derived_by_key.get(output_name) or parameter_by_key.get(output_name)
        return (
            isinstance(item, dict)
            and str(item.get("name") or "") == self.policyengine_rule_hint
        )

    def _run_rulespec_test_cases(
        self,
        *,
        rules_file: Path,
        compiled_path: Path,
        compiled_payload: dict[str, Any],
        cases: list[Any],
    ) -> list[str]:
        """Run compact RuleSpec `.test.yaml` cases against the compiled artifact."""
        issues: list[str] = []
        binary = self._axiom_rules_binary()
        derived_by_key, parameter_by_key = self._rulespec_program_maps(compiled_payload)
        legal_ids_by_friendly_name = self._rulespec_legal_ids_by_friendly_output_name(
            compiled_payload
        )
        require_legal_input_keys = _rulespec_program_has_legal_ids(compiled_payload)
        module_target = _rulespec_module_target(compiled_payload)

        for index, case in enumerate(cases, 1):
            if not isinstance(case, dict):
                issues.append(f"Test case #{index} must be a mapping.")
                continue
            case_name = str(case.get("name") or f"#{index}")
            if "name" not in case:
                issues.append(f"Test case #{index} is missing name.")
            if "period" not in case:
                issues.append(f"Test case `{case_name}` is missing period.")
                continue
            try:
                period = self._coerce_rulespec_period(case["period"])
            except ValueError as exc:
                issues.append(f"Test case `{case_name}` period invalid: {exc}")
                continue

            output_map = case.get("output")
            if "output" not in case:
                issues.append(f"Test case #{index} is missing output.")
                continue
            if not isinstance(output_map, dict) or not output_map:
                issues.append(f"Test case `{case_name}` output must be a mapping.")
                continue
            if self.policyengine_rule_hint and not any(
                self._rulespec_output_satisfies_policyengine_hint(
                    str(output_name),
                    derived_by_key=derived_by_key,
                    parameter_by_key=parameter_by_key,
                )
                for output_name in output_map
            ):
                issues.append(
                    f"Test case #{index} output must assert "
                    f"{self.policyengine_rule_hint}."
                )

            derived_outputs: list[str] = []
            parameter_outputs: list[str] = []
            for output_name in output_map:
                output_key = str(output_name)
                if output_key in derived_by_key:
                    derived_outputs.append(output_key)
                elif output_key in parameter_by_key:
                    parameter_outputs.append(output_key)
                else:
                    if _RULESPEC_ABSOLUTE_REFERENCE.match(output_key):
                        resolution_issue = _rulespec_absolute_test_reference_issue(
                            output_key,
                            label="output",
                            policy_repo_path=self.policy_repo_path,
                            allow_input_slots=False,
                            allow_relations=False,
                            allow_outputs=True,
                        )
                        if resolution_issue:
                            issues.append(f"Test case `{case_name}` {resolution_issue}")
                            continue
                    legal_ids = legal_ids_by_friendly_name.get(output_key)
                    if legal_ids:
                        if len(legal_ids) == 1:
                            issue = (
                                f"Test case `{case_name}` output `{output_key}` "
                                f"must use legal RuleSpec id `{legal_ids[0]}` "
                                "instead of the friendly name."
                            )
                        else:
                            legal_id_list = ", ".join(f"`{item}`" for item in legal_ids)
                            issue = (
                                f"Test case `{case_name}` output `{output_key}` "
                                "must use a legal RuleSpec id instead of the "
                                f"ambiguous friendly name; use one of {legal_id_list}."
                            )
                        issues.append(issue)
                        continue
                    issues.append(
                        f"Test case `{case_name}` output `{output_key}` is not "
                        f"a compiled derived output or scalar parameter in {rules_file.name}."
                    )

            actual_outputs: dict[str, Any] = {}
            if derived_outputs:
                response_outputs, execution_issues = (
                    self._run_rulespec_derived_test_case(
                        binary=binary,
                        compiled_path=compiled_path,
                        case=case,
                        case_name=case_name,
                        case_index=index,
                        period=period,
                        output_names=derived_outputs,
                        derived_by_key=derived_by_key,
                        require_legal_input_keys=require_legal_input_keys,
                        legal_ids_by_friendly_name=legal_ids_by_friendly_name,
                        module_target=module_target,
                    )
                )
                issues.extend(execution_issues)
                if response_outputs is not None:
                    actual_outputs.update(response_outputs)

            for output_name in parameter_outputs:
                try:
                    parameter_value = self._rulespec_compiled_parameter_value(
                        parameter_by_key[output_name],
                        period,
                        case.get("input", {}),
                    )
                except ValueError as exc:
                    issues.append(f"Test case `{case_name}` parameter failed: {exc}")
                    continue
                actual_outputs[output_name] = {
                    "kind": "scalar",
                    "value": parameter_value,
                }

            for output_name, expected_value in output_map.items():
                output_key = str(output_name)
                actual_output = actual_outputs.get(output_key)
                if actual_output is None:
                    if output_key in derived_outputs or output_key in parameter_outputs:
                        issues.append(
                            f"Test case `{case_name}` output `{output_key}` missing "
                            "from execution response."
                        )
                    continue
                mismatch = self._compare_rulespec_output(
                    case_name=case_name,
                    output_name=output_key,
                    expected_value=expected_value,
                    actual_output=actual_output,
                )
                if mismatch:
                    issues.append(mismatch)

        return issues

    def _run_rulespec_ci(self, rules_file: Path) -> ValidationResult:
        """Run RuleSpec compile, executable tests, and source-grounding checks."""
        start = time.time()
        issues: list[str] = []
        content = rules_file.read_text()
        raw_output: str | None = None
        compiled_payload: dict[str, Any] | None = None
        compiled_path: Path | None = None

        tmpdir_cm = tempfile.TemporaryDirectory()
        tmpdir = Path(tmpdir_cm.name)
        try:
            compiled_path = tmpdir / "compiled.json"
            compile_result, payload = self._compile_rulespec_to_artifact(
                rules_file, compiled_path
            )
            raw_output = compile_result.stdout + compile_result.stderr
            if compile_result.returncode != 0:
                detail = compile_result.stderr.strip() or compile_result.stdout.strip()
                issues.append(f"Axiom rules engine compile failed: {detail}")
            elif isinstance(payload, dict):
                compiled_payload = payload
                raw_output = self._rulespec_compile_success_output(payload)
            else:
                issues.append(
                    "Axiom rules engine compile did not return an artifact payload."
                )
        except Exception as exc:
            issues.append(f"Axiom rules engine compile failed: {exc}")

        issues.extend(find_ungrounded_numeric_issues(content))
        issues.extend(find_deprecated_source_url_issues(content))
        issues.extend(find_source_claim_reference_issues(content))
        proof_issues = (
            validate_rulespec_proofs(
                content,
                require_policy_proofs=self.require_policy_proofs,
            ).issues
            if self.require_policy_proofs
            else find_rulespec_proof_issues(content)
        )
        issues.extend(proof_issues)
        issues.extend(find_structured_scale_parameter_issues(content))
        issues.extend(find_versioned_derived_formula_issues(content))
        issues.extend(find_upstream_placement_issues(content, rules_file=rules_file))
        issues.extend(find_source_verification_issues(content))
        issues.extend(find_source_condition_coverage_issues(content))
        issues.extend(find_broad_application_passthrough_issues(content))
        issues.extend(find_formula_absolute_reference_issues(content))
        issues.extend(
            find_copied_cross_reference_source_issues(
                content,
                rules_file=rules_file,
                policy_repo_path=self.policy_repo_path,
            )
        )
        issues.extend(
            find_missing_same_section_subsection_import_issues(
                content,
                rules_file=rules_file,
                policy_repo_path=self.policy_repo_path,
            )
        )
        issues.extend(find_aggregate_exception_predicate_issues(content))
        issues.extend(self._check_cross_reference_exception_placeholders(rules_file))
        issues.extend(
            find_source_relation_issues(content, policy_repo_path=self.policy_repo_path)
        )
        issues.extend(
            find_rule_name_path_suffix_issues(
                content,
                rules_file=rules_file,
                policy_repo_path=self.policy_repo_path,
            )
        )
        issues.extend(find_sibling_rule_name_collision_issues(content, rules_file))
        strict_layout_checks = (
            _strict_rules_repo_layout_checks_enabled(
                policy_repo_path=self.policy_repo_path,
                rules_file=rules_file,
            )
            and self.enforce_repository_layout
        )
        if strict_layout_checks:
            issues.extend(find_rule_source_metadata_issues(content))

        test_path = self._rulespec_test_path(rules_file)
        if test_path.exists():
            try:
                payload = yaml.safe_load(test_path.read_text())
            except (yaml.YAMLError, ValueError) as exc:
                issues.append(f"Test YAML parse failed: {exc}")
            else:
                if payload in (None, ""):
                    if not self._is_nonassertable_rulespec_artifact(rules_file):
                        issues.append("No tests found.")
                elif not isinstance(payload, list):
                    if isinstance(payload, dict) and isinstance(
                        payload.get("cases"), list
                    ):
                        payload = payload["cases"]
                    else:
                        issues.append("RuleSpec tests must be a YAML list of cases.")
                        payload = None
                if isinstance(payload, list) and compiled_payload and compiled_path:
                    pre_test_issue_count = len(issues)
                    if strict_layout_checks:
                        issues.extend(
                            find_missing_derived_companion_output_issues(
                                content,
                                payload,
                                rules_file=rules_file,
                                policy_repo_path=self.policy_repo_path,
                            )
                        )
                    issues.extend(find_test_input_assignment_issues(content, payload))
                    issues.extend(find_exception_test_coverage_issues(content, payload))
                    if len(issues) == pre_test_issue_count:
                        issues.extend(
                            self._run_rulespec_test_cases(
                                rules_file=rules_file,
                                compiled_path=compiled_path,
                                compiled_payload=compiled_payload,
                                cases=payload,
                            )
                        )
        elif not issues and not self._is_nonassertable_rulespec_artifact(rules_file):
            issues.append("No tests found.")

        duration = int((time.time() - start) * 1000)
        try:
            return ValidationResult(
                validator_name="ci",
                passed=len(issues) == 0,
                issues=issues,
                duration_ms=duration,
                error=issues[0] if issues else None,
                raw_output=raw_output,
            )
        finally:
            tmpdir_cm.cleanup()

    def _is_nonassertable_rulespec_artifact(self, rules_file: Path) -> bool:
        """Return true when a RuleSpec artifact intentionally has no assertions."""
        try:
            payload = yaml.safe_load(rules_file.read_text())
        except (yaml.YAMLError, ValueError):
            return False
        if not isinstance(payload, dict):
            return False
        module = payload.get("module")
        status = (
            str(module.get("status", "")).strip()
            if isinstance(module, dict)
            else str(payload.get("status", "")).strip()
        )
        rules = payload.get("rules")
        if (
            isinstance(rules, list)
            and rules
            and all(
                isinstance(rule, dict)
                and str(rule.get("kind") or "").lower() == "source_relation"
                for rule in rules
            )
        ):
            return True
        return status in {"deferred", "entity_not_supported"} and not payload.get(
            "rules"
        )

    def _run_ci(self, rulespec_file: Path) -> ValidationResult:
        """Run CI checks for RuleSpec artifacts."""
        if not self._is_rulespec_file(rulespec_file):
            return ValidationResult(
                validator_name="ci",
                passed=False,
                issues=["RuleSpec YAML artifacts are required."],
                error="RuleSpec YAML artifacts are required",
            )
        return self._run_rulespec_ci(rulespec_file)

    def _copy_validation_import_closure(
        self,
        rulespec_file: Path,
        destination_root: Path,
        root_destination_relative: Path | None = None,
        include_root_companion_test: bool = False,
    ) -> None:
        """Copy a RuleSpec file and dependencies into a temp tree."""
        source_root = self._validation_source_root(rulespec_file)
        root_resolved = rulespec_file.resolve()
        pending = [root_resolved]
        copied: set[Path] = set()

        while pending:
            current = pending.pop()
            resolved = current.resolve()
            if resolved in copied:
                continue
            copied.add(resolved)

            if resolved == root_resolved and root_destination_relative is not None:
                relative = root_destination_relative
            else:
                relative = current.relative_to(source_root)
            target = destination_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, target)

            if include_root_companion_test and resolved == root_resolved:
                companion_test = self._rulespec_test_path(current)
                if companion_test.exists():
                    companion_target = self._rulespec_test_path(target)
                    companion_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(companion_test, companion_target)

            for dependency in self._resolve_import_dependencies(current, source_root):
                if dependency.resolve() not in copied:
                    pending.append(dependency)

    def _validation_source_root(self, rulespec_file: Path) -> Path:
        """Resolve the root directory used for import lookup during CI validation."""
        resolved_file = rulespec_file.resolve()
        resolved_root = self.policy_repo_path.resolve()
        with contextlib.suppress(ValueError):
            resolved_file.relative_to(resolved_root)
            return resolved_root
        if resolved_file.parent.name == "source":
            runner_root = resolved_file.parent.parent
            if any(
                (runner_root / sibling).exists()
                for sibling in ("external", "legislation", "regulation", "statutes")
            ):
                return runner_root
        return resolved_file.parent

    def _resolve_import_dependencies(
        self,
        rulespec_file: Path,
        source_root: Path,
    ) -> list[Path]:
        """Resolve imported RuleSpec files for a single file."""
        dependencies: list[Path] = []
        for import_path in self._extract_import_paths(rulespec_file.read_text()):
            target = source_root / self._import_to_relative_rulespec_path(import_path)
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
            if item_match:
                item = item_match.group(2).strip()
            else:
                mapping_match = IMPORT_MAPPING_PATTERN.match(line)
                if not mapping_match:
                    continue
                item = mapping_match.group(2).strip()
            import_target = item.split("#", 1)[0].strip()
            if import_target:
                paths.append(import_target)

        return paths

    def _import_to_relative_rulespec_path(self, import_target: str) -> Path:
        """Convert an import target like 26/24/c#name into 26/24/c.yaml."""
        normalized = import_target.strip().strip('"').strip("'")
        if normalized.endswith((".yaml", ".yml")):
            return Path(normalized)
        return Path(f"{normalized}.yaml")

    def _extract_defined_symbols(self, content: str) -> list[str]:
        """Extract RuleSpec rule names."""
        with contextlib.suppress(yaml.YAMLError, TypeError, ValueError):
            payload = yaml.safe_load(content)
            if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
                return sorted(
                    {
                        str(rule.get("name")).strip()
                        for rule in payload["rules"]
                        if isinstance(rule, dict) and str(rule.get("name", "")).strip()
                    }
                )
        return []

    def _check_cross_statute_definition_imports(self, rulespec_file: Path) -> list[str]:
        """Flag missing imports for explicit cross-statute definition references."""
        if rulespec_file.stem == rulespec_file.parent.name:
            return []

        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        title = self._infer_title_from_rulespec_path(rulespec_file)
        if not title:
            return []

        imports = self._extract_import_paths(content)
        issues: list[str] = []
        for citation, import_path in self._extract_definition_cross_references(
            source_text, title
        ):
            if not self._rulespec_import_target_exists(import_path):
                continue
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

    def _rulespec_import_target_exists(self, import_path: str) -> bool:
        """Return whether a canonical statute import target exists locally."""
        target = (
            self.policy_repo_path
            / "statutes"
            / self._import_to_relative_rulespec_path(import_path)
        )
        if target.exists():
            return True
        return target.with_suffix("").is_dir()

    def _check_cross_reference_exception_placeholders(
        self, rulespec_file: Path
    ) -> list[str]:
        """Reject local placeholder facts for cited exception sections."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text or not re.search(
            r"\b(?:except|unless|notwithstanding)\b",
            source_text,
            flags=re.IGNORECASE,
        ):
            return []

        title = self._infer_title_from_rulespec_path(rulespec_file)
        if not title:
            return []
        current_section = self._infer_section_from_rulespec_path(rulespec_file)

        imports = {
            self._normalize_rulespec_import_path(import_path)
            for import_path in self._extract_import_paths(content)
        }
        issues: list[str] = []
        seen: set[tuple[str, str]] = set()
        for block in self._extract_definition_blocks(content):
            formula = "\n".join(str(line) for line in block["body_lines"])
            identifiers = _formula_local_identifiers(formula)
            identifiers.update(_exception_formula_inputs(formula))
            for identifier in sorted(identifiers):
                import_base = self._cross_reference_placeholder_import_base(
                    identifier,
                    title=title,
                    current_section=current_section,
                    source_text=source_text,
                )
                if not import_base:
                    continue
                if self._imports_cover_path(imports, import_base):
                    continue
                key = (str(block["name"]), identifier)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(
                    "Cross-reference placeholder: "
                    f"`{block['name']}` uses local fact `{identifier}` "
                    f"for a cited legal section. Encode the cited source and import "
                    f"`{import_base}` instead of creating a local cross-reference input."
                )
        return issues

    def _normalize_rulespec_import_path(self, import_path: str) -> str:
        """Normalize a RuleSpec import path for repository-local comparison."""
        normalized = import_path.split("#", 1)[0].strip().strip("\"'")
        if ":" in normalized:
            _, tail = normalized.split(":", 1)
            normalized = tail
        return normalized.strip("/")

    def _imports_cover_path(self, imports: set[str], expected_path: str) -> bool:
        """Return whether any import covers an expected repo-relative path."""
        expected = expected_path.strip("/")
        for import_path in imports:
            if import_path == expected:
                return True
            if import_path.startswith(expected + "/"):
                return True
            if expected.startswith(import_path + "/"):
                return True
        return False

    def _cross_reference_placeholder_import_base(
        self,
        identifier: str,
        *,
        title: str,
        current_section: str | None,
        source_text: str,
    ) -> str | None:
        """Infer the canonical imported path from a section/subsection placeholder."""
        subsection_match = re.match(
            r"^subsection_(?P<subsection>[A-Za-z0-9]+)(?:_(?P<tail>[A-Za-z0-9_]+))?$",
            identifier,
        )
        if subsection_match and current_section:
            subsection = subsection_match.group("subsection")
            if re.search(
                rf"\bsubsection\s+\({re.escape(subsection)}\)",
                source_text,
                flags=re.IGNORECASE,
            ):
                return "/".join(["statutes", title, current_section, subsection])

        match = re.match(
            r"^section_(?P<section>[0-9][A-Za-z0-9.-]*)(?:_(?P<tail>[A-Za-z0-9_]+))?$",
            identifier,
        )
        if not match:
            return None
        if not (
            _is_exception_identifier(identifier)
            or re.search(
                rf"\bsection\s+{re.escape(match.group('section'))}\b",
                source_text,
                flags=re.IGNORECASE,
            )
        ):
            return None
        fragments: list[str] = []
        for fragment in (match.group("tail") or "").split("_"):
            if fragment in {
                "exception",
                "exceptions",
                "except",
                "exclusion",
                "exclusions",
                "carve",
                "carveout",
                "applies",
                "apply",
                "preclude",
                "precludes",
                "displace",
                "displaces",
                "requirements",
                "met",
            }:
                break
            if fragment:
                fragments.append(fragment)
        return "/".join(["statutes", title, match.group("section"), *fragments])

    def _check_resolved_defined_term_imports(self, rulespec_file: Path) -> list[str]:
        """Flag missing imports for known legally-defined terms mentioned in source text."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        imports = self._extract_import_paths(content)
        issues: list[str] = []
        for term in resolve_defined_terms_from_text(source_text):
            import_base = term.import_target.split("#", 1)[0]
            if any(
                existing == import_base or existing.startswith(import_base + "/")
                for existing in imports
            ):
                continue
            issues.append(
                "Defined term import missing: "
                f"`{term.term}` resolves to {term.citation} but file does not import "
                f"from {import_base}"
            )
        return issues

    def _check_resolved_canonical_concept_imports(
        self, rulespec_file: Path
    ) -> list[str]:
        """Flag missing imports for uniquely resolved nearby canonical concepts."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        imports = self._extract_import_paths(content)
        source_root = self._validation_source_root(rulespec_file)
        issues: list[str] = []
        for concept in resolve_canonical_concepts_from_text(
            source_text,
            source_root,
            current_file=rulespec_file,
        ):
            import_base = concept.import_target.split("#", 1)[0]
            if any(
                existing == import_base or existing.startswith(import_base + "/")
                for existing in imports
            ):
                continue
            issues.append(
                "Canonical concept import missing: "
                f"`{concept.term}` resolves to {concept.citation} via "
                f"{concept.import_target} but file does not import from {import_base}"
            )
        return issues

    def _check_promoted_stub_file(self, rulespec_file: Path) -> list[str]:
        """Flag committed RuleSpec stubs when their corpus source exists."""
        source_root = self._validation_source_root(rulespec_file)
        try:
            relative = rulespec_file.resolve().relative_to(source_root.resolve())
        except ValueError:
            return []

        if not rulespec_content_has_stub_status(rulespec_file.read_text()):
            return []
        if not has_corpus_provision_for_import_target(
            relative.with_suffix("").as_posix(), source_root
        ):
            return []

        return [
            "Promoted RuleSpec stub with corpus source: "
            f"{relative.as_posix()} still declares `status: stub` even though corpus.provisions has source text; "
            "replace the stub with a real encoding before promotion"
        ]

    def _check_imported_stub_dependencies(self, rulespec_file: Path) -> list[str]:
        """Flag imports that point at stubs even though corpus source exists."""
        source_root = self._validation_source_root(rulespec_file)
        issues: list[str] = []

        for import_path in self._extract_import_paths(rulespec_file.read_text()):
            target = source_root / self._import_to_relative_rulespec_path(import_path)
            if not rulespec_file_has_stub_status(target):
                continue
            if not has_corpus_provision_for_import_target(import_path, source_root):
                continue
            issues.append(
                "Imported stub dependency with corpus source: "
                f"{rulespec_file.name} imports `{import_path}` but `{target.relative_to(source_root).as_posix()}` "
                "is still a stub while corpus.provisions has source text; encode the upstream file instead"
            )

        return issues

    def _check_placeholder_fact_variables(self, rulespec_file: Path) -> list[str]:
        """Flag local factual predicates encoded as constant/deferred placeholders."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []
        source_metadata = _load_nearby_eval_source_metadata(rulespec_file)

        issues: list[str] = []
        for block in self._extract_definition_blocks(content):
            if block["dtype"] != "Boolean":
                continue
            if block["imports"]:
                continue

            status = str(block["status"] or "").lower()
            constant_boolean = bool(block["constant_boolean"])
            if not (constant_boolean or status == "deferred"):
                continue
            if _source_metadata_sets_target_symbol(source_metadata, block["name"]):
                continue

            issues.append(
                "Placeholder fact variable: "
                f"{block['name']} line {block['line']} is a source-stated factual predicate "
                f"but is encoded as {'a constant boolean' if constant_boolean else '`status: deferred`'}; "
                "expose it as a plain fact-shaped input or import a canonical definition instead"
            )
        return issues

    def _check_except_where_carve_out_logic(self, rulespec_file: Path) -> list[str]:
        """Flag carve-out branches that incorrectly treat the exception as satisfaction."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []
        if not re.search(
            r"\bexcept where\b.+\bapplies\b",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return []

        issues: list[str] = []
        for block in self._extract_definition_blocks(content):
            if block["dtype"] != "Boolean":
                continue
            for line_number, expression in self._iter_true_on_applies_patterns(
                block["body_lines"]
            ):
                issues.append(
                    "Carve-out logic inverted: "
                    f"{block['name']} line {line_number} treats `{expression}` as automatically satisfied "
                    "when an `except where ... applies` carve-out should displace this slice"
                )
        return issues

    def _check_embedded_scalar_literals(self, rulespec_file: Path) -> list[str]:
        """Flag substantive scalar literals embedded inside formulas."""
        issues: list[str] = []
        for (
            line_number,
            name,
            literal,
            expression,
        ) in self._collect_embedded_scalar_literals(rulespec_file.read_text()):
            issues.append(
                "Embedded scalar literal: "
                f"{name} line {line_number} embeds {literal} in `{expression}`; "
                "extract the scalar to its own named variable"
            )
        return issues

    def _check_decomposed_date_scalars(self, rulespec_file: Path) -> list[str]:
        """Flag numeric year/month/day scalars derived from non-substantive date references."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []
        if not (
            GROUNDING_DATE_PATTERN.search(source_text)
            or _MONTH_NAME_PATTERN.search(source_text)
        ):
            return []

        issues: list[str] = []
        for occurrence in extract_named_scalar_occurrences(content):
            tokens = set(occurrence.name.lower().split("_"))
            if not (_DATE_DECOMPOSITION_CUE_TOKENS & tokens):
                continue
            if (
                "year" in tokens
                and occurrence.value.is_integer()
                and 1900 <= occurrence.value <= 2100
            ):
                issues.append(
                    "Decomposed date scalar: "
                    f"{occurrence.name} line {occurrence.line} encodes calendar year "
                    f"{int(occurrence.value)} as a numeric scalar; keep legal date references "
                    "semantic instead of splitting them into year/month/day variables"
                )
            elif (
                "month" in tokens
                and occurrence.value.is_integer()
                and 1 <= occurrence.value <= 12
            ):
                issues.append(
                    "Decomposed date scalar: "
                    f"{occurrence.name} line {occurrence.line} encodes calendar month "
                    f"{int(occurrence.value)} as a numeric scalar; keep legal date references "
                    "semantic instead of splitting them into year/month/day variables"
                )
            elif (
                "day" in tokens
                and occurrence.value.is_integer()
                and 1 <= occurrence.value <= 31
            ):
                issues.append(
                    "Decomposed date scalar: "
                    f"{occurrence.name} line {occurrence.line} encodes calendar day "
                    f"{int(occurrence.value)} as a numeric scalar; keep legal date references "
                    "semantic instead of splitting them into year/month/day variables"
                )
        return issues

    def _check_branch_specific_output_names(self, rulespec_file: Path) -> list[str]:
        """Flag branch leaves whose principal output name drops the deepest branch token."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        expected_branch = self._extract_expected_branch_token(source_text)
        if expected_branch is None:
            return []

        blocks = self._extract_definition_blocks(content)
        if not blocks:
            return []

        principal_name = str(blocks[-1]["name"]).lower()
        if self._name_contains_branch_token(principal_name, expected_branch):
            return []

        return [
            "Branch-specific output name missing: "
            f"source text targets branch ({expected_branch}), but the principal output "
            f"`{principal_name}` does not encode that deepest branch token"
        ]

    def _check_function_style_variable_calls(self, rulespec_file: Path) -> list[str]:
        """Flag variable references that incorrectly use function-call syntax."""
        content = rulespec_file.read_text()
        defined_symbols = set(self._extract_defined_symbols(content))
        if not defined_symbols:
            return []

        issues: list[str] = []
        for block in self._extract_definition_blocks(content):
            for offset, line in enumerate(block["body_lines"], start=1):
                stripped = line.strip()
                if not stripped or stripped.endswith(":"):
                    continue
                for symbol in defined_symbols:
                    if symbol == block["name"]:
                        continue
                    if re.search(rf"\b{re.escape(symbol)}\s*\(", stripped):
                        issues.append(
                            "Function-style variable reference: "
                            f"{block['name']} line {block['line'] + offset} calls `{symbol}(...)`; "
                            "reference RuleSpec variables by bare name instead of function-call syntax"
                        )
        return issues

    def _check_exclusion_list_principal_outputs(self, rulespec_file: Path) -> list[str]:
        """Flag exclusion-list leaves whose principal output collapses to a constant."""
        content = rulespec_file.read_text()
        source_text = extract_embedded_source_text(content)
        if not source_text:
            return []

        normalized_source = " ".join(source_text.lower().split())
        if (
            "except the following which is not to be treated as qualifying income"
            not in normalized_source
        ):
            return []

        blocks = self._extract_definition_blocks(content)
        if not blocks:
            return []

        principal = blocks[-1]
        status = str(principal["status"] or "").lower()
        if not principal["constant_boolean"] and status != "deferred":
            return []

        principal_name = str(principal["name"])
        detail = (
            "a constant boolean"
            if principal["constant_boolean"]
            else "`status: deferred`"
        )
        return [
            "Exclusion-list leaf collapsed to placeholder output: "
            f"`{principal_name}` encodes a qualifying-income exclusion branch as {detail}; "
            "encode either the excluded amount itself or a fact-sensitive classification that changes with the source-stated subject/input"
        ]

    def _extract_definition_blocks(self, content: str) -> list[dict[str, object]]:
        """Extract simple summaries of RuleSpec rules."""
        try:
            payload = yaml.safe_load(content)
        except (yaml.YAMLError, ValueError):
            return []
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            return []

        blocks: list[dict[str, object]] = []
        for index, rule in enumerate(payload["rules"], start=1):
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "").strip()
            if not name:
                continue
            formula_lines: list[str] = []
            constant_boolean = False
            versions = rule.get("versions")
            if isinstance(versions, list):
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    formula = version.get("formula")
                    if isinstance(formula, bool):
                        constant_boolean = True
                        formula_lines.append(str(formula).lower())
                    elif isinstance(formula, (int, float)):
                        formula_lines.append(str(formula))
                    elif isinstance(formula, str):
                        stripped_formula = formula.strip()
                        if stripped_formula.lower() in {"true", "false"}:
                            constant_boolean = True
                        formula_lines.extend(stripped_formula.splitlines())

            imports_payload = rule.get("imports")
            imports = (
                [str(item) for item in imports_payload]
                if isinstance(imports_payload, list)
                else []
            )
            blocks.append(
                {
                    "name": name,
                    "line": index,
                    "body_lines": formula_lines,
                    "imports": imports,
                    "dtype": rule.get("dtype"),
                    "status": rule.get("status"),
                    "constant_boolean": constant_boolean,
                }
            )
        return blocks

    def _iter_true_on_applies_patterns(
        self, body_lines: list[str]
    ) -> list[tuple[int, str]]:
        """Return `(line_number_offset, condition)` pairs for `if <applies>: true` branches."""
        findings: list[tuple[int, str]] = []
        for index, line in enumerate(body_lines):
            inline_match = re.search(
                r"\bif\s+([A-Za-z_]\w*applies[A-Za-z_0-9]*)\b[^:\n]*:\s*true\b",
                line,
                flags=re.IGNORECASE,
            )
            if inline_match:
                findings.append((index + 1, inline_match.group(1)))
                continue

            branch_match = re.match(
                r"^\s*if\s+([A-Za-z_]\w*applies[A-Za-z_0-9]*)\b[^:\n]*:\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if not branch_match:
                continue
            if index + 1 >= len(body_lines):
                continue
            next_line = body_lines[index + 1].strip().lower()
            if next_line == "true":
                findings.append((index + 1, branch_match.group(1)))
        return findings

    def _extract_expected_branch_token(self, source_text: str) -> str | None:
        """Return the deepest non-numeric structural branch token from source text."""
        tokens: list[str] = []
        for line in source_text.splitlines():
            stripped = line.strip()
            if not _STRUCTURAL_SOURCE_LINE_PATTERN.match(stripped):
                continue
            token = stripped.strip("()[] .").lower()
            if token.isdigit():
                continue
            tokens.append(token)
        return tokens[-1] if tokens else None

    def _name_contains_branch_token(self, name: str, token: str) -> bool:
        """Return True when a definition name encodes a structural branch token."""
        return bool(re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", name))

    def _collect_embedded_scalar_literals(
        self,
        content: str,
    ) -> list[tuple[int, str, str, str]]:
        """Return embedded substantive scalar literals found in RuleSpec formulas."""
        issues: list[tuple[int, str, str, str]] = []
        try:
            payload = yaml.safe_load(content)
        except (yaml.YAMLError, ValueError):
            return []
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            return []

        for rule_index, rule in enumerate(payload["rules"], start=1):
            if not isinstance(rule, dict):
                continue
            name = str(rule.get("name") or "<unknown>")
            versions = rule.get("versions")
            if not isinstance(versions, list):
                continue
            for version in versions:
                if not isinstance(version, dict):
                    continue
                formula = version.get("formula")
                if isinstance(formula, (bool, int, float)):
                    continue
                if not isinstance(formula, str):
                    continue
                for line in formula.splitlines() or [formula]:
                    stripped = line.strip()
                    if not stripped or self._is_direct_scalar_expression(stripped):
                        continue
                    issues.extend(
                        (rule_index, name, literal, stripped)
                        for literal in self._extract_embedded_scalar_literals(stripped)
                    )
        return issues

    def _is_direct_scalar_expression(self, expression: str) -> bool:
        normalized = expression.replace(",", "")
        return bool(_EMBEDDED_SCALAR_DIRECT_VALUE.fullmatch(normalized))

    def _extract_embedded_scalar_literals(self, expression: str) -> list[str]:
        literals: list[str] = []
        scrubbed_expression = _QUOTED_STRING_PATTERN.sub(" ", expression)
        half_up_rounding_expression = _is_half_up_rounding_expression(
            scrubbed_expression
        )
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
            if literal == "0.5" and half_up_rounding_expression:
                continue
            if _is_structural_schedule_index_literal(scrubbed_expression, literal):
                continue
            if _is_structural_enum_index_literal(scrubbed_expression, literal):
                continue
            literals.append(literal)
        return sorted(set(literals))

    def _build_import_advisories(self, rulespec_file: Path) -> list[str]:
        """Return non-blocking advice about likely shared concepts."""
        content = rulespec_file.read_text()
        definitions = self._extract_defined_symbols(content)
        if not definitions:
            return []

        source_root = self._validation_source_root(rulespec_file)
        search_root = self._candidate_concept_search_root(rulespec_file, source_root)
        if not search_root.exists():
            return []

        imports = set(self._extract_import_paths(content))
        advisories: list[str] = []
        seen: set[tuple[str, str]] = set()

        for candidate_file in search_root.rglob("*.yaml"):
            if candidate_file.name.endswith(".test.yaml"):
                continue
            if candidate_file.resolve() == rulespec_file.resolve():
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

    def _candidate_concept_search_root(
        self, rulespec_file: Path, source_root: Path
    ) -> Path:
        """Choose a nearby subtree for conservative shared-concept advisories."""
        with contextlib.suppress(ValueError):
            relative = rulespec_file.resolve().relative_to(source_root.resolve())
            if len(relative.parts) >= 2:
                return source_root / relative.parts[0] / relative.parts[1]
        return rulespec_file.parent

    def _relative_import_base(
        self, candidate_file: Path, source_root: Path
    ) -> str | None:
        """Convert a RuleSpec path to an import base."""
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
            citation = match.group("section")
            target_title = match.group("title") or title
            import_path = self._section_reference_to_import_path(target_title, citation)
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

    def _infer_title_from_rulespec_path(self, rulespec_file: Path) -> str | None:
        """Infer the USC title from the RuleSpec file path."""
        resolved_root = self.policy_repo_path.resolve()
        resolved_file = rulespec_file.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_file.relative_to(resolved_root)
            if (
                len(relative.parts) >= 2
                and relative.parts[0] == "statutes"
                and re.fullmatch(r"[0-9A-Za-z.-]+", relative.parts[1])
                and any(ch.isdigit() for ch in relative.parts[1])
            ):
                return relative.parts[1]
            if (
                relative.parts
                and re.fullmatch(r"[0-9A-Za-z.-]+", relative.parts[0])
                and any(ch.isdigit() for ch in relative.parts[0])
            ):
                return relative.parts[0]
            return None

        parts = list(resolved_file.parts)
        with contextlib.suppress(ValueError):
            statutes_idx = parts.index("statutes")
            if statutes_idx + 1 < len(parts):
                return parts[statutes_idx + 1]
        return None

    def _infer_section_from_rulespec_path(self, rulespec_file: Path) -> str | None:
        """Infer the USC section from the RuleSpec file path."""
        resolved_root = self.policy_repo_path.resolve()
        resolved_file = rulespec_file.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_file.relative_to(resolved_root)
            if (
                len(relative.parts) >= 3
                and relative.parts[0] == "statutes"
                and re.fullmatch(r"[0-9A-Za-z.-]+", relative.parts[2])
                and any(ch.isdigit() for ch in relative.parts[2])
            ):
                return relative.parts[2]
            if (
                len(relative.parts) >= 2
                and re.fullmatch(r"[0-9A-Za-z.-]+", relative.parts[1])
                and any(ch.isdigit() for ch in relative.parts[1])
            ):
                return relative.parts[1]
            return None

        parts = list(resolved_file.parts)
        with contextlib.suppress(ValueError):
            statutes_idx = parts.index("statutes")
            if statutes_idx + 2 < len(parts):
                return parts[statutes_idx + 2]
        return None

    def _run_reviewer(
        self,
        reviewer_type: str,
        rulespec_file: Path,
        oracle_context: Optional[dict] = None,
        review_context: str | None = None,
    ) -> ValidationResult:
        """Run a reviewer agent via Claude Code CLI with oracle context.

        Args:
            reviewer_type: Type of reviewer (rulespec-reviewer, formula-reviewer, etc.)
            rulespec_file: Path to the RuleSpec file to review
            oracle_context: Results from oracle validators (PE, TAXSIM) for context

        Returns:
            ValidationResult with score, issues, and raw output
        """
        start = time.time()

        # Read RuleSpec file content
        try:
            rulespec_content = Path(rulespec_file).read_text()
            test_content = None
            companion_test = self._rulespec_test_path(rulespec_file)
            if companion_test.exists():
                test_content = companion_test.read_text()
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name=reviewer_type,
                passed=False,
                score=0.0,
                issues=[f"Failed to read RuleSpec file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Build review prompt based on reviewer type
        review_focus = {
            "rulespec-reviewer": "structure, legal citations, imports, entity hierarchy, RuleSpec compliance",
            "formula-reviewer": "logic correctness, edge cases, circular dependencies, return statements, type consistency",
            "parameter-reviewer": "no magic numbers (only -1,0,1,2,3 allowed), parameter sourcing, time-varying values",
            "integration-reviewer": "test coverage, dependency resolution, documentation, completeness",
            "generalist-reviewer": "overall statutory fidelity, missing or merged branches, defined terms, factual predicates, and suspicious semantic compression",
            "Formula Reviewer": "logic correctness, edge cases, circular dependencies, return statements",
            "Parameter Reviewer": "no magic numbers (only -1,0,1,2,3 allowed), parameter sourcing",
            "Integration Reviewer": "test coverage, dependency resolution, documentation",
        }.get(reviewer_type, "overall quality")
        prompt_template = {
            "rulespec-reviewer": RULESPEC_REVIEWER_PROMPT,
            "generalist-reviewer": GENERALIST_REVIEWER_PROMPT,
        }.get(reviewer_type)

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
        review_context_section = ""
        if review_context:
            review_context_section = f"\n## Review Context\n{review_context}\n"
        test_section = ""
        if test_content:
            test_section = (
                "\n## Companion Test File\n"
                f"{test_content[:3000]}{'...' if len(test_content) > 3000 else ''}\n"
            )

        if prompt_template is not None:
            prompt = f"""{prompt_template}

---

# TASK

Review this encoding holistically.

File: benchmark artifact (RuleSpec YAML)

Content:
{rulespec_content[:6000]}{"..." if len(rulespec_content) > 6000 else ""}
{test_section}{review_context_section}{oracle_section}
If oracle validators show discrepancies, investigate WHY the encoding differs from consensus.

Output ONLY valid JSON matching the schema above.
"""
        else:
            prompt = f"""Review this RuleSpec file for: {review_focus}

File: {rulespec_file}

Content:
{rulespec_content[:3000]}{"..." if len(rulespec_content) > 3000 else ""}
{test_section}{review_context_section}{oracle_section}
If oracle validators show discrepancies, investigate WHY the encoding differs from consensus.

Output ONLY valid JSON:
{{
  "score": <float 1-10>,
  "passed": <boolean>,
  "issues": ["issue1", "issue2"],
  "reasoning": "<brief explanation>"
}}
"""
        prompt_sha256 = _sha256_text(prompt)

        try:
            reviewer_timeout = int(
                os.getenv("AXIOM_ENCODE_REVIEWER_TIMEOUT_SECONDS", "300")
            )
            output, returncode = run_claude_code(
                prompt,
                model=REVIEWER_CLI_MODEL,
                timeout=reviewer_timeout,
                cwd=self.policy_repo_path,
            )
            if returncode != 0:
                output_excerpt = output.strip()[:500] or "no output"
                raise RuntimeError(
                    f"Reviewer CLI exited {returncode}: {output_excerpt}"
                )

            # Parse JSON from output. If parsing fails, return a
            # structured "reviewer_parse_failed" result instead of
            # letting the generic except below swallow the detail.
            try:
                data = _extract_json_object(output)
            except (ValueError, json.JSONDecodeError) as parse_err:
                duration = int((time.time() - start) * 1000)
                raw_snippet = (output or "").strip()
                truncated = raw_snippet[:500]
                if len(raw_snippet) > 500:
                    truncated += "... [truncated]"
                logger.warning(
                    "Reviewer %s output failed to parse as JSON: %s. "
                    "Raw output (first 500 chars): %s",
                    reviewer_type,
                    parse_err,
                    truncated,
                )
                return ValidationResult(
                    validator_name=reviewer_type,
                    passed=False,
                    score=None,
                    issues=[
                        "reviewer_parse_failed",
                        f"Could not parse reviewer JSON output: {parse_err}",
                    ],
                    duration_ms=duration,
                    raw_output=output,
                    error=f"reviewer_parse_failed: {parse_err}",
                    prompt_sha256=prompt_sha256,
                )

            score = float(data.get("score", 5.0))
            if reviewer_type == "generalist-reviewer":
                blocking_issues = data.get("blocking_issues", [])
                non_blocking_issues = data.get("non_blocking_issues", [])
                if not isinstance(blocking_issues, list):
                    blocking_issues = [str(blocking_issues)]
                if not isinstance(non_blocking_issues, list):
                    non_blocking_issues = [str(non_blocking_issues)]

                if "passed" in data:
                    passed = bool(data["passed"])
                elif "blocking_issues" in data:
                    passed = len(blocking_issues) == 0
                else:
                    passed = score >= 7.0

                reported_issues = data.get("issues", [])
                if not isinstance(reported_issues, list):
                    reported_issues = [str(reported_issues)]
                issues = list(blocking_issues)
                issues.extend(
                    f"[non-blocking] {issue}" for issue in non_blocking_issues
                )
                for issue in reported_issues:
                    if issue not in issues:
                        issues.append(issue)
            else:
                passed = bool(data.get("passed", score >= 7.0))
                issues = data.get("issues", [])
            if not isinstance(issues, list):
                issues = [str(issues)]
            if score < 7.0:
                if passed:
                    issues.append(
                        "reviewer_score_below_pass_threshold: "
                        f"score {score:.1f}/10 is below 7.0"
                    )
                passed = False

            duration = int((time.time() - start) * 1000)

            return ValidationResult(
                validator_name=reviewer_type,
                passed=passed,
                score=score,
                issues=issues,
                duration_ms=duration,
                raw_output=output,
                prompt_sha256=prompt_sha256,
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
                prompt_sha256=prompt_sha256,
            )

    def _detect_policyengine_country(
        self, rulespec_file: Path, rulespec_content: str
    ) -> str:
        """Infer which PolicyEngine country package to use."""
        if self.policyengine_country in {"us", "uk"}:
            return self.policyengine_country

        haystack = f"{rulespec_file}\n{rulespec_content}".lower()
        if "legislation.gov.uk" in haystack or re.search(
            r"\b(?:ukpga|uksi|asp|ssi|wsi|nisi|anaw|asc)(?:/|-)", haystack
        ):
            return "uk"
        return "us"

    def _find_pe_python(self, country: str = "us") -> Optional[str]:
        """Find a Python interpreter with the requested PolicyEngine package installed.

        Checks: 1) explicit env override, 2) known PE checkout/worktree venv paths,
        3) current interpreter, 4) auto-install.
        Returns the path to a working Python, or None.
        """
        module_name = f"policyengine_{country}"
        package_name = f"policyengine-{country}"
        repo_name = f"policyengine-{country}"
        env_var_name = f"AXIOM_ENCODE_POLICYENGINE_{country.upper()}_PYTHON"

        def _python_imports_policyengine(python_path: str) -> bool:
            try:
                result = subprocess.run(
                    [
                        python_path,
                        "-c",
                        f"from {module_name} import Simulation; print('ok')",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                return result.returncode == 0 and "ok" in result.stdout
            except Exception:
                return False

        env_python = os.getenv(env_var_name)
        if env_python and Path(env_python).exists():
            if _python_imports_policyengine(env_python):
                return env_python

        pe_venv_paths = [
            Path.home()
            / "worktrees"
            / f"{repo_name}-main-view"
            / ".venv"
            / "bin"
            / "python",
            Path.home() / "worktrees" / repo_name / ".venv" / "bin" / "python",
            Path.home() / repo_name / ".venv" / "bin" / "python",
            Path.home() / "TheAxiomFoundation" / repo_name / ".venv" / "bin" / "python",
            Path.home() / "PolicyEngine" / repo_name / ".venv" / "bin" / "python",
        ]
        for pe_python in pe_venv_paths:
            if pe_python.exists() and _python_imports_policyengine(str(pe_python)):
                return str(pe_python)

        # Try current interpreter after explicit checkout/worktree environments so
        # local source trees win over stale globally-installed packages.
        if _python_imports_policyengine(sys.executable):
            return sys.executable

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
        timeout = int(os.getenv("AXIOM_ENCODE_POLICYENGINE_TIMEOUT_SECONDS", "300"))
        try:
            idle_timeout = min(
                timeout,
                max(
                    0,
                    int(
                        os.getenv(
                            "AXIOM_ENCODE_POLICYENGINE_IDLE_TIMEOUT_SECONDS",
                            "45",
                        )
                    ),
                ),
            )
            result = _run_subprocess_with_idle_timeout(
                [pe_python, "-c", script],
                timeout=timeout,
                idle_timeout=idle_timeout,
            )
            return OracleSubprocessResult(
                returncode=result.returncode,
                stdout=result.output or "",
                stderr="",
            )
        except subprocess.TimeoutExpired as exc:
            return OracleSubprocessResult(
                returncode=124,
                stdout=getattr(exc, "stdout", "") or "",
                stderr=(getattr(exc, "stderr", "") or "").strip()
                or f"Timeout after {timeout}s",
            )
        except Exception as exc:
            return OracleSubprocessResult(returncode=1, stderr=str(exc))

    def _is_pe_unsupported_error(self, error_text: str) -> bool:
        """Return True when PE cannot evaluate the cited period or variable."""
        if not error_text:
            return False
        return any(
            pattern.search(error_text) for pattern in _PE_UNSUPPORTED_ERROR_PATTERNS
        )

    def _summarize_oracle_error(self, error_text: str) -> str:
        """Collapse multi-line stderr into a short human-readable issue."""
        if not error_text:
            return "unknown error"
        for line in reversed(error_text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return "unknown error"

    def _run_policyengine(self, rulespec_file: Path) -> ValidationResult:
        """Validate against PolicyEngine oracle.

        Uses scenario-based comparison: builds standard PE households from
        RuleSpec test case inputs and compares the PE-calculated output variable
        against the RuleSpec test's expected value.

        For programs like SNAP where RuleSpec tests use intermediate inputs
        (snap_net_income, thrifty_food_plan_cost), we run PE with equivalent
        raw household scenarios and compare at the output variable level.
        """
        start = time.time()
        issues = []

        # Read companion RuleSpec test content.
        try:
            rulespec_content = self._read_test_content(rulespec_file)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="policyengine",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RuleSpec/test file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        try:
            rulespec_source_content = rulespec_file.read_text()
        except Exception:
            rulespec_source_content = ""
        source_metadata = _load_nearby_eval_source_metadata(rulespec_file)

        country = self._detect_policyengine_country(
            rulespec_file, rulespec_source_content
        )

        # Extract RuleSpec test cases.
        tests = self._extract_rulespec_tests(rulespec_content)

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
                details={
                    "coverage": PolicyEngineOracleCoverage(setup_errors=1).as_dict()
                },
            )

        # Run comparison for each legal-ID keyed test output.
        matches = 0
        total = 0
        coverage = PolicyEngineOracleCoverage()
        for test in tests:
            test_rule_name = str(test.get("variable", ""))
            raw_test_rule_name = str(test.get("raw_variable") or test_rule_name)
            oracle_rule_name = self.policyengine_rule_hint or raw_test_rule_name
            if not self._should_compare_pe_test_output(
                country, raw_test_rule_name, oracle_rule_name
            ):
                coverage.skipped += 1
                continue
            expected = test.get("expect")
            raw_inputs = test.get("inputs", {})
            inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else raw_inputs
            oracle_inputs = test.get("oracle_inputs", {})
            if isinstance(inputs, dict) and isinstance(oracle_inputs, dict):
                policyengine_inputs = oracle_inputs.get("policyengine", {})
                if isinstance(policyengine_inputs, dict):
                    inputs.update(policyengine_inputs)
            period = (
                test.get("period")
                or test.get("date")
                or (inputs.get("period") if isinstance(inputs, dict) else None)
                or (inputs.get("date") if isinstance(inputs, dict) else None)
                or "2024-01"
            )
            period_str = _policyengine_period_string(period)
            year = period_str.split("-")[0] if "-" in period_str else period_str
            if isinstance(inputs, dict):
                inputs.pop("period", None)
                inputs.pop("date", None)

            if expected is None:
                continue
            coverage.total_outputs += 1

            mapping = self._resolve_pe_mapping(country, raw_test_rule_name)
            pe_var = mapping.policyengine_variable if mapping else None
            pe_parameter = mapping.policyengine_parameter if mapping else None
            pe_target = pe_var or pe_parameter
            if mapping is not None and not mapping.comparable:
                coverage.unsupported += 1
                continue

            mappable, reason = self._is_pe_test_mappable(
                country, raw_test_rule_name, inputs, expected, pe_var=pe_var
            )
            if not mappable:
                issues.append(
                    f"PolicyEngine unavailable for '{test.get('name', test_rule_name)}': {reason}"
                )
                coverage.unsupported += 1
                continue

            if not pe_target:
                coverage.unmapped += 1
                continue

            # Build and run PE scenario — include period in inputs for monthly detection
            inputs_with_period = {**inputs, "period": str(period)}
            inputs_with_period = {
                **_policyengine_us_snap_input_aliases(inputs_with_period),
                **inputs_with_period,
            }
            source_jurisdiction = None
            if country == "us":
                source_jurisdiction = _source_metadata_jurisdiction(
                    source_metadata
                ) or _infer_us_state_code_from_rulespec_path(
                    rulespec_file,
                    rulespec_source_content,
                )
                if source_jurisdiction and "state_code_str" not in inputs_with_period:
                    inputs_with_period["state_code_str"] = source_jurisdiction
                if (
                    source_jurisdiction
                    and inputs_with_period.get("snap_utility_allowance_type")
                    and "snap_utility_region_str" not in inputs_with_period
                ):
                    inputs_with_period["snap_utility_region_str"] = (
                        _default_snap_utility_region_for_jurisdiction(
                            source_jurisdiction
                        )
                    )
                if pe_var in {
                    "snap_standard_utility_allowance",
                    "snap_limited_utility_allowance",
                    "snap_individual_utility_allowance",
                }:
                    default_utility_type = _default_snap_utility_type_for_rule(pe_var)
                    if (
                        inputs_with_period.get("snap_utility_allowance_type")
                        in (None, "NONE")
                        and default_utility_type is not None
                    ):
                        inputs_with_period["snap_utility_allowance_type"] = (
                            default_utility_type
                        )
                    if source_jurisdiction:
                        inputs_with_period.setdefault(
                            "snap_utility_region_str",
                            _default_snap_utility_region_for_jurisdiction(
                                source_jurisdiction
                            ),
                        )
            if pe_parameter:
                try:
                    scenario_script = self._build_pe_parameter_script(
                        mapping,
                        inputs_with_period,
                        year,
                    )
                except ValueError as exc:
                    issues.append(
                        f"PolicyEngine unavailable for '{test.get('name', test_rule_name)}': {exc}"
                    )
                    coverage.unsupported += 1
                    continue
            else:
                scenario_script = self._build_pe_scenario_script(
                    pe_var,
                    inputs_with_period,
                    year,
                    expected,
                    country=country,
                    rule_name=pe_var,
                )
            output = self._run_pe_subprocess_detailed(scenario_script, pe_python)

            if output.returncode != 0:
                summary = self._summarize_oracle_error(output.stderr or output.stdout)
                if self._is_pe_unsupported_error(output.stderr or output.stdout):
                    issues.append(
                        f"PolicyEngine unavailable for '{test.get('name', test_rule_name)}': {summary}"
                    )
                    coverage.unsupported += 1
                    continue
                issues.append(
                    "PE calculation failed for "
                    f"'{test.get('name', test_rule_name)}' "
                    f"({raw_test_rule_name} -> {pe_target}): {summary}"
                )
                coverage.adapter_errors += 1
                total += 1
                continue

            # Parse result
            try:
                lines = output.stdout.strip().split("\n")
                result_line = [line for line in lines if line.startswith("RESULT:")]
                if result_line:
                    parts = result_line[0].split(":")
                    pe_value = float(parts[1])
                    if mapping and mapping.result_multiplier is not None:
                        pe_value *= mapping.result_multiplier
                    expected_float = _policyengine_expected_float(expected)
                    match = self._values_match(pe_value, expected_float, tolerance=0.02)
                    if match:
                        matches += 1
                        coverage.passed += 1
                    else:
                        issues.append(
                            f"'{test.get('name', test_rule_name)}' "
                            f"({raw_test_rule_name} -> {pe_target}): "
                            f"PE={pe_value:.2f}, RuleSpec expects={expected_float:.2f}"
                        )
                        coverage.failed += 1
                    coverage.comparable += 1
                    total += 1
                else:
                    issues.append(
                        "No RESULT in PE output for "
                        f"'{test.get('name', test_rule_name)}' "
                        f"({raw_test_rule_name} -> {pe_target})"
                    )
                    coverage.adapter_errors += 1
                    total += 1
            except Exception as parse_err:
                issues.append(
                    f"Parse error for '{test.get('name', test_rule_name)}' "
                    f"({raw_test_rule_name} -> {pe_target}): {parse_err}"
                )
                coverage.adapter_errors += 1
                total += 1

        if total == 0:
            duration = int((time.time() - start) * 1000)
            if coverage.unmapped:
                issues.append(
                    "PolicyEngine could not evaluate unclassified legal outputs"
                )
            if not issues and not coverage.unsupported:
                issues.append("No PolicyEngine-comparable tests found")
            return ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=None,
                issues=issues,
                duration_ms=duration,
                details={"coverage": coverage.as_dict()},
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
            details={"coverage": coverage.as_dict()},
        )

    def _read_test_content(self, rulespec_file: Path) -> str:
        """Read test content from the companion RuleSpec `.test.yaml` file."""
        test_file = self._rulespec_test_path(rulespec_file)
        if test_file.exists():
            return test_file.read_text()
        return ""

    def _run_taxsim(self, rulespec_file: Path) -> ValidationResult:
        """Validate against TAXSIM oracle.

        Converts test cases to TAXSIM format, runs through TAXSIM API,
        and compares relevant outputs. Returns match rate as score (0-1).
        """
        start = time.time()
        issues = []

        # Read companion RuleSpec test content.
        try:
            rulespec_content = self._read_test_content(rulespec_file)
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="taxsim",
                passed=False,
                score=0.0,
                issues=[f"Failed to read RuleSpec/test file: {e}"],
                duration_ms=duration,
                error=str(e),
            )

        # Extract RuleSpec test cases.
        tests = self._extract_rulespec_tests(rulespec_content)

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
                    issues.append(
                        "TAXSIM could not evaluate any oracle-comparable tests"
                    )
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
        """Benchmark RuleSpec encoding against PolicyEngine using CPS microdata.

        Runs PE Microsimulation on ECPS, extracts the target
        variable for all tax units, and reports the benchmark. This establishes
        the PE baseline that RuleSpec must match as inputs get wired up.

        Args:
            output_path: Directory containing RuleSpec files for the section.
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

            # RuleSpec match rate starts at 0% until runtime replay is wired here.
            rulespec_match_rate = 0.0

            issues = [
                f"PE benchmark for {pe_variable} ({year}):",
                f"  Tax units: {stats['total_tax_units']:,} "
                f"({stats['nonzero_count']:,} with {pe_variable} > 0)",
                f"  Weighted recipients: {stats['weighted_nonzero']:,.0f}",
                f"  Weighted total: ${stats['weighted_sum_billions']:.1f}B",
                f"  Mean: ${stats['mean']:,.0f}, Median: ${stats['median']:,.0f}, "
                f"Max: ${stats['max']:,.0f}",
                f"  P25-P75 (nonzero): ${stats['p25_nonzero']:,.0f}-${stats['p75_nonzero']:,.0f}",
                f"  RuleSpec match rate: {rulespec_match_rate:.1%} "
                f"(runtime replay not wired here — benchmark only)",
            ]

            duration = int((time.time() - start) * 1000)
            return ValidationResult(
                validator_name="microdata_benchmark",
                passed=False,  # 0% match until runtime replay is wired here.
                score=rulespec_match_rate,
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

    def _extract_rulespec_tests(self, test_content: str) -> list[dict]:
        """Extract oracle-comparable cases from a RuleSpec `.test.yaml` file."""
        tests = []

        def normalize_variable_name(value: Any) -> str:
            text = str(value)
            if "#" in text:
                return text.rsplit("#", 1)[1]
            return text

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
                    if (
                        len(other_keys) == 1
                        and next(iter(other_keys)) in singleton_entity_value_keys
                    ):
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

        def normalize_input_value(key: Any, value: Any) -> Any:
            key_text = str(key)
            if "#relation." in key_text or ".relation." in key_text:
                return value
            return normalize_test_value(value)

        def normalized_input_alias(key: Any) -> str:
            key_text = str(key)
            if "#" not in key_text:
                return ""
            fragment = key_text.split("#", 1)[1].strip()
            if ".input." in fragment:
                return fragment.rsplit(".input.", 1)[1].strip()
            if fragment.startswith("input."):
                return fragment.removeprefix("input.").strip()
            return ""

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

        try:
            content_lines = []
            for line in test_content.split("\n"):
                stripped = line.strip()
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
                            key: normalize_input_value(key, value)
                            for key, value in inputs.items()
                        }
                        if isinstance(inputs, dict)
                        else inputs or {}
                    )
                    if isinstance(normalized_inputs, dict):
                        for key, value in list(normalized_inputs.items()):
                            alias = normalized_input_alias(key)
                            if alias:
                                normalized_inputs.setdefault(alias, value)
                    oracle_inputs = test_case.get("oracle_inputs", {})
                    normalized_oracle_inputs: dict[str, dict[str, Any]] = {}
                    if isinstance(oracle_inputs, dict):
                        for oracle_name, oracle_raw_inputs in oracle_inputs.items():
                            if not isinstance(oracle_raw_inputs, dict):
                                continue
                            normalized_oracle_inputs[str(oracle_name)] = {
                                str(key): normalize_input_value(key, value)
                                for key, value in oracle_raw_inputs.items()
                            }
                    for variable, expected in outputs.items():
                        tests.append(
                            {
                                "variable": normalize_variable_name(variable),
                                "raw_variable": str(variable),
                                "name": test_case.get("name"),
                                "period": test_case.get("period"),
                                "inputs": normalized_inputs,
                                "expect": normalize_test_value(expected),
                                **(
                                    {"oracle_inputs": normalized_oracle_inputs}
                                    if normalized_oracle_inputs
                                    else {}
                                ),
                            }
                        )

            if isinstance(parsed, dict) and isinstance(parsed.get("cases"), list):
                append_top_level_io_tests(parsed["cases"])
            elif isinstance(parsed, dict) and isinstance(parsed.get("tests"), list):
                append_top_level_io_tests(parsed["tests"])
            elif isinstance(parsed, list):
                append_top_level_io_tests(parsed)
        except Exception:
            pass

        return tests

    def _get_pe_variable_map(self, country: str = "us") -> dict[str, str]:
        """Map encoded variable names to PolicyEngine variable names.

        Returns dict of encoded_var_name -> pe_var_name.
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

        mapping = {
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
        }
        return mapping

    def _resolve_pe_mapping(
        self, country: str, legal_id: str | None
    ) -> PolicyEngineMapping | None:
        """Resolve a canonical Axiom legal output ID to a PE registry mapping."""
        if not legal_id or "#" not in str(legal_id) or ":" not in str(legal_id):
            return None
        return self.policyengine_registry.mapping_for_legal_id(
            str(legal_id), country=country
        )

    @staticmethod
    def _get_pe_us_var_adapter(name: str) -> PolicyEngineUSVarAdapter | None:
        """Return the PE-US adapter row for a mapped PE var or encoded alias."""
        return get_pe_us_var_adapter(name)

    @staticmethod
    def _is_uk_child_benefit_rate_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
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
    def _is_uk_child_benefit_other_child_rate_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
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
    def _is_uk_pension_credit_standard_minimum_guarantee_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        if "minimum_guarantee" in rule_name_lower and any(
            marker in rule_name_lower
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
        return "guarantee_credit" in rule_name_lower and any(
            marker in rule_name_lower
            for marker in (
                "6_1_a",
                "6_1_b",
                "regulation_6_1",
                "standard_minimum_a",
                "standard_minimum_b",
            )
        )

    @staticmethod
    def _is_uk_pension_credit_couple_rate_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        if any(
            marker in rule_name_lower
            for marker in ("claimant_has_partner", "exception_applies", "_applies")
        ):
            return False
        return "no_partner" not in rule_name_lower and any(
            marker in rule_name_lower
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
    def _is_uk_pension_credit_single_rate_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
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
    def _is_uk_pc_severe_disability_addition_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "pc_severe_disability_addition" in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_pc_carer_addition_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return "pc_carer_addition" in rule_name_lower and not rule_name_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_pc_child_addition_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "pc_child_addition" in rule_name_lower
            and "disabled" not in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_pc_disabled_child_addition_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "pc_disabled_child_addition" in rule_name_lower
            and "severe" not in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_pc_severely_disabled_child_addition_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "pc_severely_disabled_child_addition" in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_scottish_child_payment_rate_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        if "scottish_child_payment" not in rule_name_lower:
            return False
        if rule_name_lower == "scottish_child_payment":
            return True
        if any(
            marker in rule_name_lower
            for marker in ("_applies", "eligible", "would_claim", "qualifying")
        ):
            return False
        return any(
            marker in rule_name_lower
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
    def _is_uk_benefit_cap_amount_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        if "benefit_cap" not in rule_name_lower and not (
            "80a_2_" in rule_name_lower
            and any(
                marker in rule_name_lower
                for marker in ("annual_limit", "relevant_amount", "_amount")
            )
        ):
            return False
        if rule_name_lower == "benefit_cap":
            return True
        if any(
            marker in rule_name_lower
            for marker in ("_applies", "exempt", "reduction", "relevant_amount_applies")
        ):
            return False
        return any(
            marker in rule_name_lower
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
    def _is_uk_uc_standard_allowance_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "uc_standard_allowance" in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_uc_carer_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return "uc_carer_element" in rule_name_lower and not rule_name_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_uc_child_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return "uc_child_element" in rule_name_lower and not rule_name_lower.endswith(
            "_applies"
        )

    @staticmethod
    def _is_uk_uc_lcwra_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
            for marker in (
                "uc_lcwra_element",
                "uc_limited_capability_for_work_related_activity",
            )
        ) and not rule_name_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_disabled_child_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            any(
                marker in rule_name_lower
                for marker in (
                    "uc_disabled_child_element",
                    "uc_child_element_disabled",
                    "universal_credit_disabled_child_element",
                )
            )
            and "severe" not in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_uc_severely_disabled_child_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
            for marker in (
                "uc_severely_disabled_child_element",
                "uc_child_element_severely_disabled",
                "universal_credit_severely_disabled_child_element",
            )
        ) and not rule_name_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_work_allowance_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            (
                "uc_work_allowance" in rule_name_lower
                or "universal_credit_work_allowance" in rule_name_lower
            )
            and "eligible" not in rule_name_lower
            and not rule_name_lower.endswith("_applies")
        )

    @staticmethod
    def _is_uk_uc_maximum_childcare_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
            for marker in (
                "uc_maximum_childcare_element",
                "uc_childcare_cap",
                "universal_credit_childcare_cap",
            )
        ) and not rule_name_lower.endswith("_applies")

    @staticmethod
    def _is_uk_uc_non_dep_deduction_amount_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            marker in rule_name_lower
            for marker in (
                "uc_individual_non_dep_deduction",
                "uc_housing_non_dep_deduction",
                "universal_credit_non_dep_deduction",
            )
        ) and not any(
            marker in rule_name_lower
            for marker in ("_eligible", "_exempt", "_applies", "non_dep_deductions")
        )

    @staticmethod
    def _is_uk_wtc_basic_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_basic" in rule_name_lower
            or "working_tax_credit_basic_element" in rule_name_lower
        )

    @staticmethod
    def _is_uk_wtc_lone_parent_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_lone_parent" in rule_name_lower
            or "working_tax_credit_lone_parent_element" in rule_name_lower
        )

    @staticmethod
    def _is_uk_wtc_couple_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_couple" in rule_name_lower
            or "working_tax_credit_couple_element" in rule_name_lower
            or "wtc_second_adult" in rule_name_lower
            or "working_tax_credit_second_adult_element" in rule_name_lower
        )

    @staticmethod
    def _is_uk_wtc_worker_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_worker" in rule_name_lower
            or "working_tax_credit_worker_element" in rule_name_lower
            or "30_hour" in rule_name_lower
            or "30_hours" in rule_name_lower
            or "thirty_hour" in rule_name_lower
        )

    @staticmethod
    def _is_uk_wtc_disabled_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_disabled" in rule_name_lower
            or "working_tax_credit_disabled_element" in rule_name_lower
        ) and "severe" not in rule_name_lower

    @staticmethod
    def _is_uk_wtc_severely_disabled_element_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return (
            "wtc_severely_disabled" in rule_name_lower
            or "working_tax_credit_severely_disabled_element" in rule_name_lower
            or "working_tax_credit_severe_disability_element" in rule_name_lower
            or "wtc_severe_disability" in rule_name_lower
        )

    @staticmethod
    def _is_uk_table_row_amount_var(rule_name: str) -> bool:
        rule_name_lower = rule_name.lower()
        return any(
            checker(rule_name_lower)
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
        "snap_gross_income",
        "snap_emergency_allotment",
        "ssi",
        "ssi_amount_if_eligible",
        "tanf",
    } | PE_US_MONTHLY_VAR_NAMES

    _PE_US_TAX_UNIT_OVERRIDE_INPUTS = {
        "adjusted_gross_income": "adjusted_gross_income",
        "modified_adjusted_gross_income": "adjusted_gross_income",
        "taxable_income": "taxable_income",
        "taxable_income_deductions": "taxable_income_deductions",
        "taxable_income_deductions_if_itemizing": (
            "taxable_income_deductions_if_itemizing"
        ),
        "taxable_income_deductions_if_not_itemizing": (
            "taxable_income_deductions_if_not_itemizing"
        ),
        "tax_unit_itemizes": "tax_unit_itemizes",
        "itemized_taxable_income_deductions": "itemized_taxable_income_deductions",
        "standard_deduction": "standard_deduction",
        "basic_standard_deduction": "basic_standard_deduction",
        "additional_standard_deduction": "additional_standard_deduction",
        "qualified_business_income_deduction": "qualified_business_income_deduction",
        "wagering_losses_deduction": "wagering_losses_deduction",
        "charitable_deduction_for_non_itemizers": (
            "charitable_deduction_for_non_itemizers"
        ),
        "tip_income_deduction": "tip_income_deduction",
        "overtime_income_deduction": "overtime_income_deduction",
        "additional_senior_deduction": "additional_senior_deduction",
        "auto_loan_interest_deduction": "auto_loan_interest_deduction",
        "exemptions": "exemptions",
        "salt_deduction": "salt_deduction",
        "misc_deduction": "misc_deduction",
        "amt_income": "amt_income",
        "amt_excluded_deductions": "amt_excluded_deductions",
        "amt_separate_addition": "amt_separate_addition",
        "amt_exemption": "amt_exemption",
        "amt_income_less_exemptions": "amt_income_less_exemptions",
        "amt_lower_base_tax": "amt_lower_base_tax",
        "amt_higher_base_tax": "amt_higher_base_tax",
        "amt_base_tax": "amt_base_tax",
        "amt_part_iii_required": "amt_part_iii_required",
        "amt_tax_including_capital_gains": "amt_tax_including_cg",
        "amt_kiddie_tax_applies": "amt_kiddie_tax_applies",
        "alternative_minimum_tax_foreign_tax_credit": "foreign_tax_credit_potential",
        "foreign_tax_credit_potential": "foreign_tax_credit_potential",
        "form_4972_lumpsum_distributions": "form_4972_lumpsum_distributions",
        "income_tax_main_rates": "income_tax_main_rates",
        "regular_tax_before_credits": "regular_tax_before_credits",
        "capital_gains_tax": "capital_gains_tax",
        "income_tax_before_credits": "income_tax_before_credits",
        "income_tax_before_refundable_credits": (
            "income_tax_before_refundable_credits"
        ),
        "net_investment_income_tax": "net_investment_income_tax",
        "recapture_of_investment_credit": "recapture_of_investment_credit",
        "qualified_retirement_penalty": "qualified_retirement_penalty",
        "foreign_tax_credit": "foreign_tax_credit",
        "cdcc": "cdcc",
        "non_refundable_american_opportunity_credit": (
            "non_refundable_american_opportunity_credit"
        ),
        "lifetime_learning_credit": "lifetime_learning_credit",
        "savers_credit": "savers_credit",
        "residential_clean_energy_credit": "residential_clean_energy_credit",
        "energy_efficient_home_improvement_credit": (
            "energy_efficient_home_improvement_credit"
        ),
        "elderly_disabled_credit": "elderly_disabled_credit",
        "new_clean_vehicle_credit": "new_clean_vehicle_credit",
        "used_clean_vehicle_credit": "used_clean_vehicle_credit",
        "non_refundable_ctc": "non_refundable_ctc",
        "tax_unit_childcare_expenses": "tax_unit_childcare_expenses",
        "min_head_spouse_earned": "min_head_spouse_earned",
        "social_security_benefits_received": "tax_unit_social_security",
        "taxable_social_security_benefits_included": (
            "tax_unit_taxable_social_security"
        ),
        "filer_adjusted_earnings": "filer_adjusted_earnings",
        "eitc_relevant_investment_income": "eitc_relevant_investment_income",
        "ctc": "ctc",
        "ctc_limiting_tax_liability": "ctc_limiting_tax_liability",
        "ctc_social_security_tax": "ctc_social_security_tax",
        "unreported_payroll_tax": "unreported_payroll_tax",
        "self_employment_tax_ald": "self_employment_tax_ald",
        "additional_medicare_tax": "additional_medicare_tax",
        "unrecaptured_section_1250_gain": "unrecaptured_section_1250_gain",
        "capital_gains_28_percent_rate_gain": "capital_gains_28_percent_rate_gain",
        "taxable_net_gain_from_dispositions": "loss_limited_net_capital_gains",
        "loss_limited_net_capital_gains": "loss_limited_net_capital_gains",
        "excess_payroll_tax_withheld": "excess_payroll_tax_withheld",
        "refundable_ctc": "refundable_ctc",
        "refundable_american_opportunity_credit": (
            "refundable_american_opportunity_credit"
        ),
        "recovery_rebate_credit": "recovery_rebate_credit",
        "refundable_payroll_tax_credit": "refundable_payroll_tax_credit",
        "eitc": "eitc",
    }
    _PE_US_PERSON_OVERRIDE_INPUTS = {
        "taxable_interest_income": "taxable_interest_income",
        "dividend_income": "dividend_income",
        "long_term_capital_gains": "long_term_capital_gains",
        "short_term_capital_gains": "short_term_capital_gains",
        "qualified_dividend_income": "qualified_dividend_income",
        "rental_income": "rental_income",
        "employee_social_security_tax": "employee_social_security_tax",
        "employee_medicare_tax": "employee_medicare_tax",
        "pension_annuity_disability_benefits_received": "pension_income",
        "taxable_pension_annuity_disability_benefits_included": (
            "taxable_pension_income"
        ),
        "section_22_disability_income": "total_disability_payments",
    }

    # PE variables at spm_unit level (need spm_units in situation)
    _PE_SPM_VARS = set(PE_US_SPM_VAR_NAMES)

    def _is_pe_test_mappable(
        self,
        country: str,
        rule_name: str,
        inputs: dict,
        expected: Any = None,
        pe_var: str | None = None,
    ) -> tuple[bool, str | None]:
        """Return whether the test case can be represented in PolicyEngine."""
        rule_name_lower = rule_name.lower()
        if country == "us":
            pe_var_name = pe_var or rule_name
            education_credit_vars = {
                "american_opportunity_credit",
                "refundable_american_opportunity_credit",
                "non_refundable_american_opportunity_credit_potential",
                "non_refundable_american_opportunity_credit_credit_limit",
                "non_refundable_american_opportunity_credit",
                "lifetime_learning_credit_potential",
                "lifetime_learning_credit_credit_limit",
                "lifetime_learning_credit",
                "education_tax_credits",
            }
            if pe_var_name in education_credit_vars:
                filing_status = _normalize_us_tax_filing_status(
                    self._rulespec_test_input_value(inputs, "filing_status")
                )
                if filing_status == "SEPARATE":
                    return (
                        False,
                        "PolicyEngine does not model the section 25A(g)(6) married-filing-separately disallowance",
                    )
                if self._rulespec_test_input_value(inputs, "is_nonresident_alien") and not bool(
                    self._rulespec_test_input_value(
                        inputs,
                        "section_6013_resident_alien_election",
                    )
                ):
                    return (
                        False,
                        "PolicyEngine does not model the section 25A(g)(7) nonresident-alien disallowance",
                    )
                if self._rulespec_test_input_value(inputs, "taxpayer_is_section_1_g_child"):
                    return (
                        False,
                        "PolicyEngine does not model the section 25A(i) kiddie-tax refundability exception",
                    )
            if pe_var_name in {"net_investment_income", "net_investment_income_tax"}:
                unsupported_niit_inputs = {
                    "annuity_income",
                    "royalty_income",
                    "passive_activity_business_income",
                    "financial_trading_business_income",
                    "allocable_investment_deductions",
                    "qualified_plan_distributions",
                    "self_employment_income_subject_to_1401_b",
                    "section_911_excluded_gross_income",
                    "section_911_disallowed_deductions_and_exclusions",
                }

                def has_nonzero_input(name: str) -> bool:
                    value = self._rulespec_test_input_value(inputs, name)
                    if value is None:
                        return False
                    with contextlib.suppress(TypeError, ValueError):
                        return float(value) != 0
                    return bool(value)

                nonzero_unsupported = sorted(
                    name for name in unsupported_niit_inputs if has_nonzero_input(name)
                )
                if nonzero_unsupported:
                    return (
                        False,
                        "PolicyEngine does not expose all section 1411(c)/(d) net-investment-income and modified-AGI components",
                    )
                if pe_var_name == "net_investment_income_tax" and self._rulespec_test_input_value(
                    inputs, "is_nonresident_alien"
                ):
                    return (
                        False,
                        "PolicyEngine does not model the section 1411(e)(1) nonresident-alien exclusion",
                    )
            if pe_var_name == "elderly_disabled_credit":
                if self._rulespec_test_input_value(inputs, "is_nonresident_alien"):
                    return (
                        False,
                        "PolicyEngine does not model the section 22(f) nonresident-alien disallowance",
                    )
                filing_status = _normalize_us_tax_filing_status(
                    self._rulespec_test_input_value(inputs, "filing_status")
                )
                if filing_status == "SEPARATE" and not bool(
                    self._rulespec_test_input_value(
                        inputs, "spouses_lived_apart_all_year"
                    )
                ):
                    return (
                        False,
                        "PolicyEngine does not model the section 22(e)(1) married-filing-separately living-apart gate",
                    )
            adapter = self._get_pe_us_var_adapter(pe_var or rule_name)
            if adapter is not None and (
                adapter.unsupported_input_keys or adapter.unsupported_input_patterns
            ):
                lowered_input_keys = {str(key).lower() for key in inputs}
                unsupported_keys = {
                    key
                    for key in adapter.unsupported_input_keys
                    if key.lower() in lowered_input_keys
                }
                for input_key in lowered_input_keys:
                    if any(
                        pattern.lower() in input_key
                        for pattern in adapter.unsupported_input_patterns
                    ):
                        unsupported_keys.add(input_key)
                if unsupported_keys:
                    reason = adapter.unsupported_input_reason or (
                        "RuleSpec test supplies unsupported PolicyEngine US scenario inputs"
                    )
                    return False, f"{reason}: {', '.join(sorted(unsupported_keys))}"
        if country == "uk" and isinstance(expected, dict):
            return (
                False,
                "RuleSpec test expects multi-entity outputs that the current PolicyEngine UK harness cannot compare directly",
            )
        if (
            country == "uk"
            and self._is_uk_table_row_amount_var(rule_name_lower)
            and expected in {0, 0.0, "0", "0.0"}
        ):
            return (
                False,
                "RuleSpec test is a row-specific zero case for a table amount slice that PolicyEngine UK does not represent as a separate zero-valued branch",
            )
        if country == "uk" and self._is_uk_child_benefit_rate_var(rule_name_lower):
            for key, value in inputs.items():
                key_lower = str(key).lower()
                if (
                    "subject_to_paragraphs" in key_lower
                    or "paragraphs_two_to_five_apply" in key_lower
                    or "paragraphs_2_to_5_apply" in key_lower
                ) and bool(value):
                    return (
                        False,
                        "RuleSpec test uses placeholder paragraph-exception conditions that PolicyEngine UK does not represent directly",
                    )
                if "payable" in key_lower and not bool(value):
                    return (
                        False,
                        "RuleSpec test encodes take-up/payability conditions that PolicyEngine UK's statutory rate variable does not represent directly",
                    )
            explicit_false_keys = {
                str(key).lower()
                for key, value in inputs.items()
                if value is not None and not bool(value)
            }
            if "is_child_or_qualifying_young_person" in explicit_false_keys or (
                any("is_child" in key for key in explicit_false_keys)
                and any("qualifying_young_person" in key for key in explicit_false_keys)
            ):
                return (
                    False,
                    "RuleSpec test negates child-or-qualifying-young-person subject status that PolicyEngine UK's statutory child benefit rate does not expose as a separate comparable branch",
                )
        if (
            country == "uk"
            and self._is_uk_child_benefit_rate_var(rule_name_lower)
            and rule_name_lower.endswith("_applies")
        ):
            return (
                False,
                "RuleSpec helper boolean does not have a direct PolicyEngine UK analogue",
            )
        if (
            country == "uk"
            and self._is_uk_pension_credit_standard_minimum_guarantee_var(
                rule_name_lower
            )
        ):
            if (
                rule_name_lower.endswith("_applies")
                or "claimant_has_partner" in rule_name_lower
                or "exception_applies" in rule_name_lower
            ):
                return (
                    False,
                    "RuleSpec helper boolean does not have a direct PolicyEngine UK analogue",
                )
            for key, value in inputs.items():
                key_lower = str(key).lower()
                if "exception_applies" in key_lower and bool(value):
                    return (
                        False,
                        "RuleSpec test uses downstream regulation exceptions that PolicyEngine UK does not represent directly",
                    )
            if self._is_uk_pension_credit_single_rate_var(rule_name_lower) and any(
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
                    "RuleSpec test negates the pension-credit single-rate branch using partner facts that PolicyEngine UK only exposes through the parent standard minimum guarantee",
                )
            if self._is_uk_pension_credit_couple_rate_var(rule_name_lower) and (
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
                    "RuleSpec test negates the pension-credit couple-rate branch using partner facts that PolicyEngine UK only exposes through the parent standard minimum guarantee",
                )
        if (
            country == "uk"
            and "scottish_child_payment" in rule_name_lower
            and rule_name_lower.endswith("_applies")
        ):
            return (
                False,
                "RuleSpec helper boolean does not have a direct PolicyEngine UK analogue",
            )
        if (
            country == "uk"
            and "benefit_cap" in rule_name_lower
            and rule_name_lower.endswith("_applies")
        ):
            return (
                False,
                "RuleSpec helper boolean does not have a direct PolicyEngine UK analogue",
            )
        return True, None

    def _resolve_pe_variable(self, country: str, rule_name: str) -> str | None:
        """Resolve an encoded variable to a PolicyEngine variable, including heuristics."""
        if country == "us":
            mapping = self._resolve_pe_mapping(country, rule_name)
            return (
                mapping.policyengine_variable
                if mapping and mapping.comparable
                else None
            )

        pe_var = self._get_pe_variable_map(country).get(rule_name)
        if pe_var:
            return pe_var

        rule_name_lower = rule_name.lower()
        if country == "uk" and self._is_uk_child_benefit_rate_var(rule_name_lower):
            return "child_benefit_respective_amount"
        if (
            country == "uk"
            and self._is_uk_pension_credit_standard_minimum_guarantee_var(
                rule_name_lower
            )
        ):
            return "standard_minimum_guarantee"
        if country == "uk" and self._is_uk_pc_severe_disability_addition_var(
            rule_name_lower
        ):
            return "severe_disability_minimum_guarantee_addition"
        if country == "uk" and self._is_uk_pc_carer_addition_var(rule_name_lower):
            return "carer_minimum_guarantee_addition"
        if country == "uk" and (
            self._is_uk_pc_child_addition_var(rule_name_lower)
            or self._is_uk_pc_disabled_child_addition_var(rule_name_lower)
            or self._is_uk_pc_severely_disabled_child_addition_var(rule_name_lower)
        ):
            return "child_minimum_guarantee_addition"
        if country == "uk" and self._is_uk_uc_standard_allowance_var(rule_name_lower):
            return "uc_standard_allowance"
        if country == "uk" and self._is_uk_uc_carer_element_var(rule_name_lower):
            return "uc_carer_element"
        if country == "uk" and self._is_uk_uc_child_element_var(rule_name_lower):
            return "uc_individual_child_element"
        if country == "uk" and self._is_uk_uc_lcwra_element_var(rule_name_lower):
            return "uc_LCWRA_element"
        if country == "uk" and self._is_uk_uc_disabled_child_element_var(
            rule_name_lower
        ):
            return "uc_individual_disabled_child_element"
        if country == "uk" and self._is_uk_uc_severely_disabled_child_element_var(
            rule_name_lower
        ):
            return "uc_individual_severely_disabled_child_element"
        if country == "uk" and self._is_uk_uc_work_allowance_var(rule_name_lower):
            return "uc_work_allowance"
        if country == "uk" and self._is_uk_uc_maximum_childcare_element_var(
            rule_name_lower
        ):
            return "uc_maximum_childcare_element_amount"
        if country == "uk" and self._is_uk_uc_non_dep_deduction_amount_var(
            rule_name_lower
        ):
            return "uc_individual_non_dep_deduction"
        if country == "uk" and self._is_uk_wtc_basic_element_var(rule_name_lower):
            return "WTC_basic_element"
        if country == "uk" and self._is_uk_wtc_lone_parent_element_var(rule_name_lower):
            return "WTC_lone_parent_element"
        if country == "uk" and self._is_uk_wtc_couple_element_var(rule_name_lower):
            return "WTC_couple_element"
        if country == "uk" and self._is_uk_wtc_worker_element_var(rule_name_lower):
            return "WTC_worker_element"
        if country == "uk" and self._is_uk_wtc_disabled_element_var(rule_name_lower):
            return "WTC_disabled_element"
        if country == "uk" and self._is_uk_wtc_severely_disabled_element_var(
            rule_name_lower
        ):
            return "WTC_severely_disabled_element"
        if country == "uk" and self._is_uk_scottish_child_payment_rate_var(
            rule_name_lower
        ):
            return "scottish_child_payment"
        if country == "uk" and self._is_uk_benefit_cap_amount_var(rule_name_lower):
            return "benefit_cap"

        return None

    def _should_compare_pe_test_output(
        self, country: str, test_rule_name: str, oracle_rule_name: str
    ) -> bool:
        """Return whether a RuleSpec test output should be compared against PolicyEngine."""
        if not self.policyengine_rule_hint:
            return True
        if test_rule_name == oracle_rule_name:
            return True
        if country == "us":
            test_mapping = self._resolve_pe_mapping(country, test_rule_name)
            if not test_mapping or not test_mapping.comparable:
                return False
            hinted_mapping = self._resolve_pe_mapping(country, oracle_rule_name)
            hinted_pe_target = (
                self._policyengine_mapping_target(hinted_mapping)
                if hinted_mapping and hinted_mapping.comparable
                else oracle_rule_name
            )
            return self._policyengine_mapping_target(test_mapping) == hinted_pe_target
        hinted_pe_var = self._resolve_pe_variable(country, oracle_rule_name)
        if not hinted_pe_var:
            return False
        test_pe_var = self._resolve_pe_variable(country, test_rule_name)
        return test_pe_var is not None and test_pe_var == hinted_pe_var

    @staticmethod
    def _policyengine_mapping_target(mapping: PolicyEngineMapping | None) -> str | None:
        """Return a stable comparison target for a PolicyEngine registry mapping."""
        if mapping is None:
            return None
        if mapping.policyengine_variable:
            multiplier = mapping.result_multiplier or 1.0
            return f"variable:{mapping.policyengine_variable}:{multiplier}"
        if mapping.policyengine_parameter:
            key = (
                mapping.parameter_key
                or ",".join(mapping.parameter_keys)
                or mapping.parameter_key_input
                or ValidatorPipeline._format_pe_parameter_key_path(
                    mapping.parameter_key_path
                )
                or mapping.parameter_calc_input
                or (
                    str(mapping.parameter_calc_value)
                    if mapping.parameter_calc_value is not None
                    else ""
                )
                or ""
            )
            multiplier = mapping.result_multiplier or 1.0
            return f"parameter:{mapping.policyengine_parameter}:{key}:{multiplier}"
        return mapping.expression

    @staticmethod
    def _format_pe_parameter_key_path(parameter_key_path: tuple[Any, ...]) -> str:
        """Return a stable display key for a nested PolicyEngine parameter path."""
        parts: list[str] = []
        for part in parameter_key_path:
            if isinstance(part, dict):
                parts.append(str(part.get("input", "")))
            else:
                parts.append(str(part))
        return "/".join(part for part in parts if part)

    def _build_pe_scenario_script(
        self,
        pe_var: str,
        inputs: dict,
        year: str,
        expected: Any,
        country: str = "us",
        rule_name: str | None = None,
    ) -> str:
        """Build a Python script to run a PE scenario via subprocess.

        Handles period detection (monthly vs annual PE variables),
        builds appropriate household structures, and overrides PE
        intermediate variables to match RuleSpec test inputs for apples-to-apples
        comparison.
        """
        if country == "uk":
            return self._build_pe_uk_scenario_script(pe_var, inputs, year, rule_name)

        return self._build_pe_us_scenario_script(pe_var, inputs, year)

    def _build_pe_parameter_script(
        self,
        mapping: PolicyEngineMapping,
        inputs: dict,
        year: str,
    ) -> str:
        """Build a Python script that reads a PolicyEngine parameter value."""
        if not mapping.policyengine_parameter:
            raise ValueError("PolicyEngine parameter mapping is missing a path")

        parameter_keys = self._resolve_pe_parameter_keys(mapping, inputs)
        parameter_key_paths = self._resolve_pe_parameter_key_paths(mapping, inputs)
        parameter_calc_values = self._resolve_pe_parameter_calc_values(
            mapping,
            inputs,
        )
        period = self._normalize_monthly_pe_period(
            inputs.get("period"),
            year,
            "01",
        )
        if str(mapping.period or "").lower() != "month":
            period = year

        parameter_path = json.dumps(mapping.policyengine_parameter)
        parameter_keys_literal = json.dumps(parameter_keys)
        parameter_key_paths_literal = json.dumps(parameter_key_paths)
        parameter_calc_values_literal = json.dumps(parameter_calc_values)
        period_literal = json.dumps(period)
        return f"""
from policyengine_us import CountryTaxBenefitSystem

def get_parameter(root, path):
    value = root
    for part in path.split('.'):
        value = getattr(value, part)
    return value

system = CountryTaxBenefitSystem()
params = system.parameters({period_literal})
value = get_parameter(params, {parameter_path})
keys = {parameter_keys_literal}
key_paths = {parameter_key_paths_literal} or [[key] for key in keys]
calc_values = {parameter_calc_values_literal}
if calc_values:
    values = [float(value.calc(calc_value)) for calc_value in calc_values]
    if any(item != values[0] for item in values):
        raise ValueError(f'Parameter calc values disagree: {{dict(zip(calc_values, values))}}')
    value = values[0]
elif key_paths:
    values = []
    for key_path in key_paths:
        selected = value
        for key in key_path:
            try:
                selected = selected[key]
            except (KeyError, IndexError, TypeError):
                if isinstance(key, str) and hasattr(selected, key):
                    selected = getattr(selected, key)
                else:
                    raise
        values.append(float(selected))
    if any(item != values[0] for item in values):
        raise ValueError(f'Parameter keys disagree: {{dict(zip(key_paths, values))}}')
    value = values[0]
print(f'RESULT:{{float(value)}}')
"""

    def _resolve_pe_parameter_keys(
        self,
        mapping: PolicyEngineMapping,
        inputs: dict,
    ) -> list[str]:
        if mapping.parameter_key is not None:
            return [mapping.parameter_key]
        if mapping.parameter_keys:
            return list(mapping.parameter_keys)
        if mapping.parameter_key_path:
            return []
        if mapping.parameter_calc_input or mapping.parameter_calc_value is not None:
            return []
        if not mapping.parameter_key_input:
            return []
        input_value = self._rulespec_test_input_value(
            inputs,
            mapping.parameter_key_input,
        )
        if input_value is None:
            raise ValueError(
                "PolicyEngine parameter mapping needs input "
                f"`{mapping.parameter_key_input}`"
            )
        raw_key = str(input_value)
        if raw_key in mapping.parameter_key_map:
            return [mapping.parameter_key_map[raw_key]]
        if mapping.parameter_key_map:
            raise ValueError(
                "PolicyEngine parameter mapping has no key for "
                f"`{mapping.parameter_key_input}={raw_key}`"
            )
        return [raw_key]

    def _resolve_pe_parameter_key_paths(
        self,
        mapping: PolicyEngineMapping,
        inputs: dict,
    ) -> list[list[Any]]:
        if not mapping.parameter_key_path:
            return []
        resolved_path: list[Any] = []
        for part in mapping.parameter_key_path:
            if not isinstance(part, dict):
                resolved_path.append(part)
                continue
            input_name = part.get("input")
            if not input_name:
                raise ValueError(
                    "PolicyEngine parameter_key_path entries with mappings need "
                    "an input name"
                )
            input_value = self._rulespec_test_input_value(inputs, str(input_name))
            if input_value is None:
                raise ValueError(
                    "PolicyEngine parameter mapping needs input "
                    f"`{input_name}`"
                )
            key_map = part.get("parameter_key_map") or part.get("key_map") or {}
            key_map = {str(key): str(value) for key, value in key_map.items()}
            raw_key = str(input_value)
            if raw_key in key_map:
                resolved_path.append(key_map[raw_key])
            elif key_map:
                raise ValueError(
                    "PolicyEngine parameter mapping has no key for "
                    f"`{input_name}={raw_key}`"
                )
            else:
                resolved_path.append(raw_key)
        return [resolved_path]

    def _resolve_pe_parameter_calc_values(
        self,
        mapping: PolicyEngineMapping,
        inputs: dict,
    ) -> list[Any]:
        if mapping.parameter_calc_value is not None:
            return [mapping.parameter_calc_value]
        if not mapping.parameter_calc_input:
            return []
        input_value = self._rulespec_test_input_value(
            inputs,
            mapping.parameter_calc_input,
        )
        if input_value is None:
            raise ValueError(
                "PolicyEngine parameter scale mapping needs input "
                f"`{mapping.parameter_calc_input}`"
            )
        return [input_value]

    @staticmethod
    def _rulespec_test_input_value(inputs: dict, name: str) -> Any:
        if name in inputs:
            return inputs[name]
        for key, value in inputs.items():
            key_text = str(key)
            if (
                key_text.endswith(f"#input.{name}")
                or key_text.endswith(f"#{name}")
                or key_text.endswith(f".{name}")
            ):
                return value
        return None

    def _normalize_monthly_pe_period(
        self,
        period: Any,
        year: str,
        fallback_month: str,
    ) -> str:
        """Normalize oracle monthly periods to YYYY-MM."""
        period_str = str(period).strip() if period is not None else ""
        if not period_str:
            return f"{year}-{fallback_month}"
        if re.fullmatch(r"\d{4}-\d{2}", period_str):
            return period_str
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period_str):
            return period_str[:7]
        if re.fullmatch(r"\d{4}", period_str):
            return f"{period_str}-{fallback_month}"
        if len(period_str) >= 7 and re.fullmatch(r"\d{4}-\d{2}.*", period_str):
            return period_str[:7]
        return f"{year}-{fallback_month}"

    def _build_pe_us_scenario_script(self, pe_var: str, inputs: dict, year: str) -> str:
        """Build a Python script to run a US PolicyEngine scenario."""
        inputs = {**_policyengine_us_snap_input_aliases(inputs), **inputs}

        def derive_override_value(
            operation: str, source_values: list[float]
        ) -> float | None:
            derived_value: float | None = None
            if operation == "difference" and len(source_values) >= 2:
                derived_value = source_values[0] - sum(source_values[1:])
            elif operation == "difference_floor_zero" and len(source_values) >= 2:
                derived_value = max(0.0, source_values[0] - sum(source_values[1:]))
            elif (
                operation == "difference_floor_zero_annualized"
                and len(source_values) >= 2
            ):
                derived_value = (
                    max(0.0, source_values[0] - sum(source_values[1:])) * 12.0
                )
            elif operation == "monthly_to_annual" and len(source_values) == 1:
                derived_value = source_values[0] * 12.0
            return derived_value

        def pe_literal(value: Any) -> str:
            if isinstance(value, str):
                return repr(value)
            if isinstance(value, bool):
                return "True" if value else "False"
            return str(value)

        def normalize_pe_override_value(pe_key: str, value: Any) -> Any:
            if pe_key == "snap_utility_allowance_type" and isinstance(value, str):
                normalized = value.strip().upper()
                return {
                    "SUA": "SUA",
                    "LUA": "LUA",
                    "IUA": "IUA",
                    "NONE": "NONE",
                    "BUA": "LUA",
                    "TUA": "IUA",
                }.get(normalized, normalized)
            return value

        # Determine household composition from inputs
        filing_status = _normalize_us_tax_filing_status(
            inputs.get("filing_status", "SINGLE")
        )
        tax_unit_parts = [
            f"'filing_status': {{'{year}': {repr(filing_status)}}}",
        ]
        for rule_key, pe_key in self._PE_US_TAX_UNIT_OVERRIDE_INPUTS.items():
            if pe_key == pe_var:
                continue
            value = self._rulespec_test_input_value(inputs, rule_key)
            if value is None:
                continue
            tax_unit_parts.append(f"'{pe_key}': {{'{year}': {pe_literal(value)}}}")
        tax_unit_extra = ", ".join(tax_unit_parts)
        joint_filing = filing_status == "JOINT"
        num_adults = 2 if joint_filing else 1
        aged_flags = _tax_unit_member_aged_flags(inputs)

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
        relation_child_rows, relation_adult_dependent_rows = (
            self._us_tax_dependent_member_rows_from_relation_inputs(inputs)
        )
        relation_filer_rows = self._us_tax_filer_member_rows_from_relation_inputs(
            inputs
        )
        relation_child_count = (
            len(relation_child_rows) if relation_child_rows is not None else None
        )

        household_children = 0
        if household_size is not None:
            with contextlib.suppress(TypeError, ValueError):
                household_children = max(0, int(household_size) - num_adults)

        num_children = (
            explicit_child_count
            if explicit_child_count is not None
            else relation_child_count
            if relation_child_count is not None
            else household_children
        )
        num_adult_dependents = (
            len(relation_adult_dependent_rows)
            if relation_child_rows is not None
            else 0
        )
        dependent_care_keys = (
            "snap_dependent_care_actual_costs",
            "snap_dependent_care_deduction",
        )
        if num_children == 0:
            for key in dependent_care_keys:
                value = inputs.get(key)
                with contextlib.suppress(TypeError, ValueError):
                    if value is not None and float(value) > 0:
                        num_children = 1
                        break

        # Determine period for calculation
        is_monthly = pe_var in self._PE_MONTHLY_VARS
        if is_monthly:
            period = self._normalize_monthly_pe_period(inputs.get("period"), year, "01")
            calc_period = f"'{period}'"
        else:
            calc_period = f"int('{year}')"

        adapter = self._get_pe_us_var_adapter(pe_var)

        adult_age = 65 if aged_flags[:1] == [True] else 30
        if relation_filer_rows:
            adult_age = self._us_tax_relation_member_age(
                relation_filer_rows[0],
                adult_age,
            )
        explicit_adult_age = self._rulespec_test_input_value(inputs, "age")
        with contextlib.suppress(TypeError, ValueError):
            if explicit_adult_age is not None:
                adult_age = int(explicit_adult_age)
        adult_attrs = [f"'age': {{'{year}': {adult_age}}}"]
        members = ["'adult'"]

        # Check for employment income / earned income
        earned = inputs.get(
            "employment_income", inputs.get("earned_income", inputs.get("wages", 0))
        )
        if earned:
            adult_attrs.append(f"'employment_income': {{'{year}': {earned}}}")
        for rule_key, pe_key in self._PE_US_PERSON_OVERRIDE_INPUTS.items():
            value = self._rulespec_test_input_value(inputs, rule_key)
            if value is None:
                continue
            adult_attrs.append(f"'{pe_key}': {{'{year}': {pe_literal(value)}}}")
        if relation_filer_rows:
            adult_attrs.extend(
                self._us_tax_person_attrs_from_relation_row(
                    relation_filer_rows[0],
                    year,
                )
            )

        if adapter is not None:
            for rule_key, pe_attr in adapter.annualized_person_inputs:
                value = inputs.get(rule_key)
                if value is None:
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    annual_value = float(value) * 12
                    adult_attrs.append(f"'{pe_attr}': {{'{year}': {annual_value}}}")
            for rule_key, pe_attr in adapter.boolean_person_inputs:
                if rule_key in inputs:
                    adult_attrs.append(
                        f"'{pe_attr}': {{'{year}': {bool(inputs[rule_key])}}}"
                    )
            for rule_key, pe_attr in adapter.monthly_boolean_person_inputs:
                if rule_key in inputs:
                    adult_attrs.append(
                        f"'{pe_attr}': {{'{period}': {bool(inputs[rule_key])}}}"
                    )

        snap_eligible_member_proxy = None
        if "snap_household_has_eligible_participating_member" in inputs:
            snap_eligible_member_proxy = bool(
                inputs["snap_household_has_eligible_participating_member"]
            )
        elif "snap_household_has_member_individually_eligible_to_participate" in inputs:
            snap_eligible_member_proxy = bool(
                inputs["snap_household_has_member_individually_eligible_to_participate"]
            )

        if (
            adapter is not None
            and adapter.pe_var == "is_snap_eligible"
            and snap_eligible_member_proxy is not None
            and "is_snap_ineligible_student" not in inputs
            and "is_snap_immigration_status_eligible" not in inputs
        ):
            has_eligible_member = snap_eligible_member_proxy
            adult_attrs.append(
                f"'is_snap_ineligible_student': {{'{year}': {not has_eligible_member}}}"
            )
            adult_attrs.append(
                f"'is_snap_immigration_status_eligible': "
                f"{{'{period}': {has_eligible_member}}}"
            )
        elif (
            adapter is not None
            and adapter.pe_var == "is_snap_eligible"
            and "snap_number_of_members_eligible_to_participate" in inputs
            and "is_snap_ineligible_student" not in inputs
            and "is_snap_immigration_status_eligible" not in inputs
        ):
            has_eligible_member = (
                float(inputs["snap_number_of_members_eligible_to_participate"]) > 0
            )
            adult_attrs.append(
                f"'is_snap_ineligible_student': {{'{year}': {not has_eligible_member}}}"
            )
            adult_attrs.append(
                f"'is_snap_immigration_status_eligible': "
                f"{{'{period}': {has_eligible_member}}}"
            )

        people_parts = [f"'adult': {{{', '.join(adult_attrs)}}}"]

        # Add spouse if joint
        if joint_filing:
            spouse_age = 65 if len(aged_flags) > 1 and aged_flags[1] else 30
            spouse_attrs = [f"'age': {{'{year}': {spouse_age}}}"]
            if len(relation_filer_rows) > 1:
                spouse_attrs[0] = (
                    f"'age': "
                    f"{{'{year}': {self._us_tax_relation_member_age(relation_filer_rows[1], spouse_age)}}}"
                )
                spouse_attrs.extend(
                    self._us_tax_person_attrs_from_relation_row(
                        relation_filer_rows[1],
                        year,
                    )
                )
            people_parts.append(f"'spouse': {{{', '.join(spouse_attrs)}}}")
            members.append("'spouse'")

        if explicit_child_count is None and relation_child_rows is not None:
            child_rows = relation_child_rows
            adult_dependent_rows = relation_adult_dependent_rows
        else:
            child_rows = [{} for _ in range(num_children)]
            adult_dependent_rows = [{} for _ in range(num_adult_dependents)]

        # Add children based on explicit qualifying-child counts, relation rows, or household size.
        for i, row in enumerate(child_rows):
            age = self._us_tax_relation_member_age(row, 8)
            child_attrs = [
                f"'age': {{'{year}': {age}}}",
                f"'is_tax_unit_dependent': {{'{year}': True}}",
            ]
            incapable = self._rulespec_test_input_value(row, "is_incapable_of_self_care")
            if incapable is not None:
                child_attrs.append(
                    f"'is_incapable_of_self_care': "
                    f"{{'{year}': {pe_literal(bool(incapable))}}}"
                )
            child_attrs.extend(
                self._us_tax_person_attrs_from_relation_row(
                    row,
                    year,
                )
            )
            people_parts.append(
                f"'child{i}': {{{', '.join(child_attrs)}}}"
            )
            members.append(f"'child{i}'")
        for i, row in enumerate(adult_dependent_rows):
            age = self._us_tax_relation_member_age(row, 30)
            adult_dep_attrs = [
                f"'age': {{'{year}': {age}}}",
                f"'is_tax_unit_dependent': {{'{year}': True}}",
            ]
            incapable = self._rulespec_test_input_value(row, "is_incapable_of_self_care")
            if incapable is not None:
                adult_dep_attrs.append(
                    f"'is_incapable_of_self_care': "
                    f"{{'{year}': {pe_literal(bool(incapable))}}}"
                )
            adult_dep_attrs.extend(
                self._us_tax_person_attrs_from_relation_row(
                    row,
                    year,
                )
            )
            people_parts.append(
                f"'adult_dep{i}': {{{', '.join(adult_dep_attrs)}}}"
            )
            members.append(f"'adult_dep{i}'")

        members_str = "[" + ", ".join(members) + "]"
        people_str = "{" + ", ".join(people_parts) + "}"

        # Build SPM unit overrides for SNAP intermediate variables
        # This allows apples-to-apples comparison when RuleSpec tests pass
        # pre-computed intermediate values (snap_net_income, etc.)
        snap_overridable = {
            "snap_net_income": "snap_net_income",
            "snap_gross_income": "snap_gross_income",
            "snap_unit_size": "snap_unit_size",
            "spm_unit_size": "snap_unit_size",
        }
        override_values: dict[str, Any] = {}
        for rule_key, pe_key in snap_overridable.items():
            if rule_key in inputs:
                override_values[pe_key] = normalize_pe_override_value(
                    pe_key, inputs[rule_key]
                )

        if adapter is not None:
            for rule_key, pe_key in adapter.direct_spm_overrides:
                if rule_key in inputs:
                    override_values[pe_key] = normalize_pe_override_value(
                        pe_key, inputs[rule_key]
                    )
            for target_key, operation, source_keys in adapter.derived_spm_overrides:
                if target_key in override_values:
                    continue
                if not all(source_key in inputs for source_key in source_keys):
                    continue
                try:
                    source_values = [
                        float(inputs[source_key]) for source_key in source_keys
                    ]
                except (TypeError, ValueError):
                    continue
                derived_value = derive_override_value(operation, source_values)
                if derived_value is None:
                    continue
                override_values[target_key] = (
                    int(derived_value) if derived_value.is_integer() else derived_value
                )
        annual_override_values: dict[str, Any] = {}
        if adapter is not None:
            for rule_key, pe_key in adapter.annual_direct_spm_overrides:
                if rule_key in inputs:
                    annual_override_values[pe_key] = normalize_pe_override_value(
                        pe_key, inputs[rule_key]
                    )
            for (
                target_key,
                operation,
                source_keys,
            ) in adapter.annual_derived_spm_overrides:
                if target_key in annual_override_values:
                    continue
                missing_as_zero = (
                    target_key == "snap_assets"
                    and source_keys
                    and source_keys[0] == "snap_total_resources_before_exclusions"
                    and source_keys[0] in inputs
                )
                try:
                    source_values = [
                        float(inputs[source_key]) if source_key in inputs else 0.0
                        for source_key in source_keys
                        if source_key in inputs or missing_as_zero
                    ]
                except (TypeError, ValueError):
                    continue
                if len(source_values) != len(source_keys):
                    continue
                derived_value = derive_override_value(operation, source_values)
                if derived_value is None:
                    continue
                annual_override_values[target_key] = (
                    int(derived_value) if derived_value.is_integer() else derived_value
                )

        override_parts = []
        for pe_key, val in override_values.items():
            if is_monthly:
                override_parts.append(f"'{pe_key}': {{'{period}': {pe_literal(val)}}}")
            else:
                override_parts.append(f"'{pe_key}': {{'{year}': {pe_literal(val)}}}")
        for pe_key, val in annual_override_values.items():
            override_parts.append(f"'{pe_key}': {{'{year}': {pe_literal(val)}}}")

        spm_extra = ""
        if override_parts:
            spm_extra = ", " + ", ".join(override_parts)

        household_state = "CA"
        if adapter is not None and adapter.default_state_code is not None:
            household_state = adapter.default_state_code
        if adapter is not None and adapter.state_code_from_boolean_input is not None:
            input_key, true_state, false_state = adapter.state_code_from_boolean_input
            if input_key in inputs:
                household_state = true_state if bool(inputs[input_key]) else false_state
        utility_region = None
        if "snap_utility_region" in inputs:
            utility_region = str(inputs["snap_utility_region"])
        elif "snap_utility_region_str" in inputs:
            utility_region = str(inputs["snap_utility_region_str"])
        if (
            pe_var
            in {
                "snap_standard_utility_allowance",
                "snap_limited_utility_allowance",
                "snap_individual_utility_allowance",
            }
            and utility_region is not None
            and utility_region.strip().upper() == "NY"
        ):
            utility_region = "NY_NYC"

        if "state_code_str" in inputs:
            household_state = str(inputs["state_code_str"])
        elif "state_name" in inputs:
            household_state = str(inputs["state_name"])
        elif utility_region is not None:
            household_state = normalize_state_code_from_utility_region(utility_region)

        household_extra_parts = [
            f"'state_name': {{'{year}': {repr(household_state)}}}",
            f"'state_code_str': {{'{year}': {repr(household_state)}}}",
        ]
        if utility_region is not None:
            household_extra_parts.append(
                f"'snap_utility_region_str': {{'{year}': {repr(utility_region)}}}"
            )
        if "state_group_str" in inputs:
            household_extra_parts.append(
                f"'state_group_str': {{'{year}': {repr(inputs['state_group_str'])}}}"
            )
        elif "state_group" in inputs:
            household_extra_parts.append(
                f"'state_group_str': {{'{year}': {repr(inputs['state_group'])}}}"
            )
        household_extra = ", ".join(household_extra_parts)

        if adapter is not None and adapter.parameter_path is not None:
            parameter_period = self._normalize_monthly_pe_period(
                inputs.get("period"), year, "01"
            )
            value_expr = f"params.{adapter.parameter_path}[{repr(household_state)}]"
            if adapter.parameter_value_mode == "float":
                return f"""
from policyengine_us import CountryTaxBenefitSystem

system = CountryTaxBenefitSystem()
params = system.parameters('{parameter_period}')
val = float({value_expr})
print(f'RESULT:{{val}}')
"""
            return f"""
from policyengine_us import CountryTaxBenefitSystem

system = CountryTaxBenefitSystem()
params = system.parameters('{parameter_period}')
val = 1.0 if bool({value_expr}) else 0.0
print(f'RESULT:{{val}}')
"""

        script = f"""
from policyengine_us import Simulation

situation = {{
    'people': {people_str},
    'tax_units': {{'tu': {{'members': {members_str}, {tax_unit_extra}}}}},
    'spm_units': {{'spm': {{'members': {members_str}{spm_extra}}}}},
    'households': {{'hh': {{'members': {members_str}, {household_extra}}}}},
    'families': {{'fam': {{'members': {members_str}}}}},
    'marital_units': {{'mu': {{'members': {["adult", "spouse"] if joint_filing else ["adult"]}}}}},
}}

sim = Simulation(situation=situation)
result = sim.calculate('{pe_var}', {calc_period})
val = float(result[0]) if hasattr(result, '__len__') and len(result) > 0 else float(result)
print(f'RESULT:{{val}}')
"""
        return script

    @staticmethod
    def _us_tax_relation_member_age(row: dict, default: int) -> int:
        age_value = ValidatorPipeline._rulespec_test_input_value(row, "age")
        with contextlib.suppress(TypeError, ValueError):
            return int(age_value)
        return default

    def _us_tax_dependent_member_rows_from_relation_inputs(
        self,
        inputs: dict,
    ) -> tuple[list[dict] | None, list[dict]]:
        """Infer PE dependent composition from RuleSpec member relation rows."""
        child_rows: list[dict] = []
        adult_dependent_rows: list[dict] = []
        saw_relation = False
        for key, rows in inputs.items():
            key_text = str(key).lower()
            if "#relation." not in key_text:
                continue
            relation_name = key_text.rsplit("#relation.", 1)[1]
            if not relation_name.endswith("member_of_tax_unit"):
                continue
            if not isinstance(rows, list):
                continue
            saw_relation = True
            for row in rows:
                if not isinstance(row, dict):
                    continue
                dependent = self._rulespec_test_input_value(
                    row,
                    "is_tax_unit_dependent",
                )
                is_eitc_qualifying_child = (
                    self._rulespec_test_input_value(
                        row,
                        "is_qualifying_child_dependent",
                    )
                    is True
                )
                if dependent is not True and not is_eitc_qualifying_child:
                    continue
                if is_eitc_qualifying_child:
                    child_rows.append(row)
                    continue
                age_value = self._rulespec_test_input_value(row, "age")
                if age_value is None:
                    child_rows.append(row)
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    age = int(age_value)
                    if age < 17:
                        child_rows.append(row)
                    else:
                        adult_dependent_rows.append(row)
        if not saw_relation:
            return None, []
        return child_rows, adult_dependent_rows

    def _us_tax_filer_member_rows_from_relation_inputs(
        self,
        inputs: dict,
    ) -> list[dict]:
        """Return non-dependent tax-unit member rows for PE head/spouse facts."""
        filer_rows: list[dict] = []
        for key, rows in inputs.items():
            key_text = str(key).lower()
            if "#relation." not in key_text:
                continue
            relation_name = key_text.rsplit("#relation.", 1)[1]
            if not relation_name.endswith("member_of_tax_unit"):
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                dependent = self._rulespec_test_input_value(
                    row,
                    "is_tax_unit_dependent",
                )
                is_eitc_qualifying_child = (
                    self._rulespec_test_input_value(
                        row,
                        "is_qualifying_child_dependent",
                    )
                    is True
                )
                if dependent is True or is_eitc_qualifying_child:
                    continue
                filer_rows.append(row)
        return filer_rows

    def _us_tax_person_attrs_from_relation_row(
        self,
        row: dict,
        year: str,
    ) -> list[str]:
        """Convert RuleSpec relation-row person facts into PE person inputs."""

        def pe_literal(value: Any) -> str:
            if isinstance(value, str):
                return repr(value)
            if isinstance(value, bool):
                return "True" if value else "False"
            return str(value)

        attrs: list[str] = []
        for rule_key, pe_key in self._PE_US_PERSON_OVERRIDE_INPUTS.items():
            value = self._rulespec_test_input_value(row, rule_key)
            if value is None:
                continue
            attrs.append(f"'{pe_key}': {{'{year}': {pe_literal(value)}}}")

        education_expenses = self._rulespec_test_input_value(
            row,
            "qualified_tuition_and_related_expenses",
        )
        if education_expenses is None:
            education_expenses = self._rulespec_test_input_value(
                row,
                "qualified_tuition_expenses",
            )
        if education_expenses is not None:
            excluded_assistance = self._rulespec_test_input_value(
                row,
                "excludable_educational_assistance",
            )
            try:
                adjusted_expenses = max(
                    0.0,
                    float(education_expenses) - float(excluded_assistance or 0),
                )
            except (TypeError, ValueError):
                adjusted_expenses = education_expenses
            attrs.append(
                f"'qualified_tuition_expenses': "
                f"{{'{year}': {pe_literal(adjusted_expenses)}}}"
            )

        aotc_eligible = self._rulespec_test_input_value(
            row,
            "is_eligible_for_american_opportunity_credit",
        )
        if aotc_eligible is None:
            aotc_eligible = self._pe_aotc_eligible_from_relation_row(row)
        if aotc_eligible is not None:
            attrs.append(
                f"'is_eligible_for_american_opportunity_credit': "
                f"{{'{year}': {pe_literal(bool(aotc_eligible))}}}"
            )

        retired_on_total_disability = self._rulespec_test_input_value(
            row,
            "retired_on_total_disability",
        )
        if retired_on_total_disability is None:
            retired = self._rulespec_test_input_value(
                row,
                "retired_on_disability_before_year_end",
            )
            unable = self._rulespec_test_input_value(
                row,
                "unable_to_engage_substantial_gainful_activity",
            )
            impairment = self._rulespec_test_input_value(
                row,
                "medically_determinable_impairment",
            )
            death = self._rulespec_test_input_value(
                row,
                "impairment_expected_to_result_in_death",
            )
            proof = self._rulespec_test_input_value(row, "disability_proof_furnished")
            duration = self._rulespec_test_input_value(row, "impairment_duration_months")
            long_duration = False
            with contextlib.suppress(TypeError, ValueError):
                long_duration = float(duration) >= 12
            if all(value is not None for value in (retired, unable, impairment, proof)):
                retired_on_total_disability = (
                    bool(retired)
                    and bool(unable)
                    and bool(impairment)
                    and (bool(death) or long_duration)
                    and bool(proof)
                )
        if retired_on_total_disability is not None:
            attrs.append(
                f"'retired_on_total_disability': "
                f"{{'{year}': {pe_literal(bool(retired_on_total_disability))}}}"
            )
        return attrs

    def _pe_aotc_eligible_from_relation_row(self, row: dict) -> bool | None:
        """Derive PE's AOTC eligibility input from section 25A relation facts."""

        def fact(name: str) -> Any:
            return self._rulespec_test_input_value(row, name)

        direct_claim_allowed = fact("aotc_claim_allowed")
        if direct_claim_allowed is not None:
            return bool(direct_claim_allowed)

        relationship_facts = (
            fact("is_taxpayer"),
            fact("is_spouse"),
            fact("is_tax_unit_dependent"),
        )
        if all(value is None for value in relationship_facts):
            relationship_allowed = None
        else:
            relationship_allowed = any(bool(value) for value in relationship_facts)

        prior_year_count = fact("aotc_prior_year_election_count")
        prior_year_ok = None
        if prior_year_count is not None:
            with contextlib.suppress(TypeError, ValueError):
                prior_year_ok = float(prior_year_count) < 4

        required_facts = {
            "aotc_election_in_effect": fact("aotc_election_in_effect"),
            "relationship_allowed": relationship_allowed,
            "meets_higher_education_act_student_requirements": fact(
                "meets_higher_education_act_student_requirements"
            ),
            "at_least_half_time_student": fact("at_least_half_time_student"),
            "prior_year_ok": prior_year_ok,
            "completed_first_four_years_postsecondary_before_year": fact(
                "completed_first_four_years_postsecondary_before_year"
            ),
            "has_felony_drug_conviction": fact("has_felony_drug_conviction"),
            "education_credit_identification_requirements_met": fact(
                "education_credit_identification_requirements_met"
            ),
            "institution_employer_identification_number_included": fact(
                "institution_employer_identification_number_included"
            ),
            "payee_statement_received": fact("payee_statement_received"),
            "aotc_disallowance_period_applies": fact(
                "aotc_disallowance_period_applies"
            ),
        }
        if any(value is None for value in required_facts.values()):
            return None
        return (
            bool(required_facts["aotc_election_in_effect"])
            and bool(required_facts["relationship_allowed"])
            and bool(
                required_facts["meets_higher_education_act_student_requirements"]
            )
            and bool(required_facts["at_least_half_time_student"])
            and bool(required_facts["prior_year_ok"])
            and not bool(
                required_facts["completed_first_four_years_postsecondary_before_year"]
            )
            and not bool(required_facts["has_felony_drug_conviction"])
            and bool(required_facts["education_credit_identification_requirements_met"])
            and bool(
                required_facts["institution_employer_identification_number_included"]
            )
            and bool(required_facts["payee_statement_received"])
            and not bool(required_facts["aotc_disallowance_period_applies"])
        )

    def _build_pe_uk_scenario_script(
        self, pe_var: str, inputs: dict, year: str, rule_name: str | None = None
    ) -> str:
        """Build a Python script to run a UK PolicyEngine scenario."""
        month_period = self._normalize_monthly_pe_period(
            inputs.get("period"), year, "04"
        )
        year_key = repr(str(year))
        rule_name_lower = (rule_name or "").lower()
        lowered = {str(key).lower(): value for key, value in inputs.items()}

        if pe_var == "uc_standard_allowance" and self._is_uk_uc_standard_allowance_var(
            rule_name_lower
        ):
            is_single = "couple" not in rule_name_lower and not any(
                marker in rule_name_lower for marker in ("joint", "partner")
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
            under_25 = (
                any(
                    marker in rule_name_lower
                    for marker in ("under_25", "aged_under_25", "young")
                )
                and "over_25" not in rule_name_lower
                and "25_or_over" not in rule_name_lower
            )
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
            rule_name_lower
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
            rule_name_lower
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
            rule_name_lower
        ):
            target_is_first_higher = any(
                marker in rule_name_lower for marker in ("first", "higher")
            )
            target_is_later_child = any(
                marker in rule_name_lower for marker in ("second", "subsequent")
            )

            if target_is_first_higher:
                people = f"{{'child': {{'age': {{{year_key}: 10}}, 'birth_year': {{{year_key}: 2015}}}}}}"
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
                people = f"{{'child': {{'age': {{{year_key}: 7}}, 'birth_year': {{{year_key}: 2018}}}}}}"
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

        if (
            pe_var == "uc_individual_disabled_child_element"
            and self._is_uk_uc_disabled_child_element_var(rule_name_lower)
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

        if (
            pe_var == "uc_individual_severely_disabled_child_element"
            and self._is_uk_uc_severely_disabled_child_element_var(rule_name_lower)
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
            rule_name_lower
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
            elif "without_housing" in rule_name_lower:
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

        if (
            pe_var == "uc_maximum_childcare_element_amount"
            and self._is_uk_uc_maximum_childcare_element_var(rule_name_lower)
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
            elif (
                "two_or_more" in rule_name_lower
                or "two_or_more_children" in rule_name_lower
            ):
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

        if (
            pe_var == "uc_individual_non_dep_deduction"
            and self._is_uk_uc_non_dep_deduction_amount_var(rule_name_lower)
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
            self._is_uk_wtc_basic_element_var(rule_name_lower)
            or self._is_uk_wtc_lone_parent_element_var(rule_name_lower)
            or self._is_uk_wtc_couple_element_var(rule_name_lower)
            or self._is_uk_wtc_worker_element_var(rule_name_lower)
            or self._is_uk_wtc_disabled_element_var(rule_name_lower)
            or self._is_uk_wtc_severely_disabled_element_var(rule_name_lower)
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
                people = f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}, 'is_disabled_for_benefits': {{{year_key}: True}}}}}}"
                benunit_members = "['adult']"
                household_members = "['adult']"
            elif pe_var == "WTC_severely_disabled_element":
                people = f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}, 'is_disabled_for_benefits': {{{year_key}: True}}, 'is_severely_disabled_for_benefits': {{{year_key}: True}}}}}}"
                benunit_members = "['adult']"
                household_members = "['adult']"
            else:
                people = f"{{'adult': {{'age': {{{year_key}: 30}}, 'weekly_hours': {{{year_key}: 30}}, 'working_tax_credit_reported': {{{year_key}: 1}}}}}}"
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

        if (
            pe_var == "standard_minimum_guarantee"
            and self._is_uk_pension_credit_standard_minimum_guarantee_var(
                rule_name_lower
            )
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
            elif any(
                "no_partner" in key and bool(value) for key, value in lowered.items()
            ):
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
            elif self._is_uk_pension_credit_couple_rate_var(rule_name_lower):
                scenario_is_couple = True
            else:
                scenario_is_couple = False

            people = f"{{'adult': {{'age': {{{year_key}: 70}}}}}}"
            benunit_members = "['adult']"
            household_members = "['adult']"
            if scenario_is_couple:
                people = f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'spouse': {{'age': {{{year_key}: 70}}}}}}"
                benunit_members = "['adult', 'spouse']"
                household_members = "['adult', 'spouse']"

            if self._is_uk_pension_credit_couple_rate_var(rule_name_lower):
                result_logic = """
if scenario_is_couple:
    val = weekly
else:
    val = 0.0
"""
            elif self._is_uk_pension_credit_single_rate_var(rule_name_lower):
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

        if (
            pe_var == "severe_disability_minimum_guarantee_addition"
            and self._is_uk_pc_severe_disability_addition_var(rule_name_lower)
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
            elif (
                "two_eligible_adults" in rule_name_lower or "double" in rule_name_lower
            ):
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
                people = f"{{'adult': {{'age': {{{year_key}: 70}}, 'attendance_allowance': {{{year_key}: 1}}}}}}"
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

        if (
            pe_var == "carer_minimum_guarantee_addition"
            and self._is_uk_pc_carer_addition_var(rule_name_lower)
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
                people = f"{{'adult': {{'age': {{{year_key}: 70}}, 'carers_allowance': {{{year_key}: 1}}}}}}"
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
            self._is_uk_pc_child_addition_var(rule_name_lower)
            or self._is_uk_pc_disabled_child_addition_var(rule_name_lower)
            or self._is_uk_pc_severely_disabled_child_addition_var(rule_name_lower)
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
            if self._is_uk_pc_severely_disabled_child_addition_var(rule_name_lower):
                target_mode = "severe"
            elif self._is_uk_pc_disabled_child_addition_var(rule_name_lower):
                target_mode = "disabled"

            if has_child:
                base_people = f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}}}}}"
                if target_mode == "disabled":
                    target_people = f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}, 'dla': {{{year_key}: 1}}}}}}"
                elif target_mode == "severe":
                    target_people = f"{{'adult': {{'age': {{{year_key}: 70}}}}, 'child': {{'age': {{{year_key}: 10}}, 'dla': {{{year_key}: 1}}, 'receives_highest_dla_sc': {{{year_key}: True}}}}}}"
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
                result_logic = (
                    "val = (float(target_annual[0]) - float(base_annual[0])) / 52"
                )

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

        if (
            pe_var == "scottish_child_payment"
            and self._is_uk_scottish_child_payment_rate_var(rule_name_lower)
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
            rule_name_lower
        ):
            lowered_keys = [str(key).lower() for key in lowered.keys()]
            branch_category = None
            if "80a_2_a" in rule_name_lower:
                branch_category = ("single", "london", "no_child")
            elif "80a_2_b_ii" in rule_name_lower:
                branch_category = ("single", "london", "child")
            elif "80a_2_b_i" in rule_name_lower:
                branch_category = ("joint", "london", "any")
            elif "80a_2_b" in rule_name_lower:
                branch_category = ("other", "london", "mixed")
            elif "80a_2_c" in rule_name_lower:
                branch_category = ("single", "outside_london", "no_child")
            elif "80a_2_d_ii" in rule_name_lower:
                branch_category = ("single", "outside_london", "child")
            elif "80a_2_d_i" in rule_name_lower:
                branch_category = ("joint", "outside_london", "any")
            elif "80a_2_d" in rule_name_lower:
                branch_category = ("other", "outside_london", "mixed")

            leaf_in_london = (
                any(
                    marker in rule_name_lower
                    for marker in ("greater_london", "in_london", "london")
                )
                and "outside_london" not in rule_name_lower
            )
            leaf_is_single = any(
                marker in rule_name_lower for marker in ("single_claimant", "single")
            ) and not any(
                marker in rule_name_lower
                for marker in ("joint_claimant", "couple", "family")
            )
            leaf_has_child = (
                any(
                    marker in rule_name_lower
                    for marker in ("child", "young_person", "family")
                )
                and "no_child" not in rule_name_lower
            )
            has_leaf_location_hint = any(
                marker in rule_name_lower
                for marker in (
                    "greater_london",
                    "in_london",
                    "london",
                    "outside_london",
                    "not_resident_in_greater_london",
                )
            )
            has_leaf_single_hint = any(
                marker in rule_name_lower
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
                marker in rule_name_lower
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
                        "not_resident_in_greater_london" in key for key in lowered_keys
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
                "not_resident_in_greater_london" in str(key).lower()
                and value is not None
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
                ("no_child" in str(key).lower() or "without_child" in str(key).lower())
                and bool(value)
                for key, value in lowered.items()
            ):
                has_child = False
            elif (
                explicit_not_responsible is None
                and explicit_responsible is None
                and any(
                    ("child" in str(key).lower() or "young_person" in str(key).lower())
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
                if ("child_or_qualifying_young_person" in key or "child_or_qyp" in key)
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
                if str(key).lower() == "is_qualifying_young_person"
                and value is not None
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
            rule_name_lower
        ) or (rule_name_lower == "child_benefit_weekly_rate" and other_case is not None)
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


def validate_file(rulespec_file: str | Path) -> PipelineResult:
    """Convenience function to validate a single file."""
    file_path = Path(rulespec_file)
    policy_repo_root = find_policy_repo_root(file_path)
    if policy_repo_root is None:
        policy_repo_root = file_path.parent
    axiom_rules_path = policy_repo_root.parent / "axiom-rules-engine"
    if not axiom_rules_path.exists():
        axiom_rules_path = Path(__file__).resolve().parents[4] / "axiom-rules-engine"

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
    )

    return pipeline.validate(file_path)
