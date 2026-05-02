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
from axiom_encode.repo_routing import find_policy_repo_root

from .dependency_stubs import (
    has_ingested_source_for_import_target,
    resolve_canonical_concepts_from_text,
    resolve_defined_terms_from_text,
    rulespec_content_has_stub_status,
    rulespec_file_has_stub_status,
)
from .encoding_db import EncodingDB, ReviewResult, ReviewResults

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


@dataclass(frozen=True)
class _PolicyEngineUSVarAdapter:
    """Declarative mapping/config for PE-US replay of an encoded variable."""

    rule_names: tuple[str, ...]
    pe_var: str
    monthly: bool = False
    spm: bool = False
    annualized_person_inputs: tuple[tuple[str, str], ...] = ()
    boolean_person_inputs: tuple[tuple[str, str], ...] = ()
    monthly_boolean_person_inputs: tuple[tuple[str, str], ...] = ()
    direct_spm_overrides: tuple[tuple[str, str], ...] = ()
    derived_spm_overrides: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    annual_direct_spm_overrides: tuple[tuple[str, str], ...] = ()
    annual_derived_spm_overrides: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    unsupported_input_keys: tuple[str, ...] = ()
    unsupported_input_patterns: tuple[str, ...] = ()
    unsupported_input_reason: str | None = None
    default_state_code: str | None = None
    state_code_from_boolean_input: tuple[str, str, str] | None = None
    parameter_path: str | None = None
    parameter_value_mode: str = "bool"


def _normalize_state_code_from_utility_region(region: str) -> str:
    """Map sub-state SNAP utility region codes back to their parent state code."""
    match = re.match(r"^([A-Z]{2})_", region)
    if match:
        return match.group(1)
    return region


def _sha256_text(text: str | None) -> str | None:
    """Return a stable digest for prompt text."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PE_US_VAR_ADAPTERS = (
    _PolicyEngineUSVarAdapter(
        rule_names=("snap", "snap_benefits"),
        pe_var="snap",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_allotment", "snap_normal_allotment"),
        pe_var="snap_normal_allotment",
        monthly=True,
        spm=True,
        unsupported_input_keys=(
            "snap_max_allotment",
            "snap_expected_contribution",
            "snap_min_allotment",
            "is_snap_eligible",
        ),
        unsupported_input_reason=(
            "RuleSpec test supplies intermediate SNAP allotment inputs that "
            "PolicyEngine US does not expose as scenario inputs"
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_expected_contribution",),
        pe_var="snap_expected_contribution",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_earned_income_deduction",),
        pe_var="snap_earned_income_deduction",
        monthly=True,
        spm=True,
        direct_spm_overrides=(("snap_earned_income", "snap_earned_income"),),
        derived_spm_overrides=(
            (
                "snap_earned_income",
                "difference_floor_zero",
                (
                    "snap_earned_income_before_exclusions",
                    "snap_child_earned_income_exclusion",
                    "snap_other_earned_income_exclusions",
                    "snap_work_support_public_assistance_income",
                ),
            ),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_min_allotment", "minimum_allotment"),
        pe_var="snap_min_allotment",
        monthly=True,
        spm=True,
        unsupported_input_keys=("snap_one_person_thrifty_food_plan_cost",),
        unsupported_input_patterns=("thrifty_food_plan_cost",),
        unsupported_input_reason=(
            "RuleSpec test supplies a thrifty-food-plan cost input that "
            "PolicyEngine US treats as an internal parameter, not a scenario input"
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_net_income", "snap_net_income_calculation"),
        pe_var="snap_net_income",
        monthly=True,
        spm=True,
        derived_spm_overrides=(
            (
                "snap_net_income",
                "difference",
                ("snap_household_income", "snap_deductions"),
            ),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_net_income_pre_shelter",),
        pe_var="snap_net_income_pre_shelter",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            (
                "snap_monthly_household_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_monthly_household_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_household_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_household_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            ("snap_gross_income", "snap_gross_income"),
            ("snap_standard_deduction", "snap_standard_deduction"),
            ("snap_earned_income_deduction", "snap_earned_income_deduction"),
            ("snap_child_support_deduction", "snap_child_support_deduction"),
            (
                "snap_excess_medical_expense_deduction",
                "snap_excess_medical_expense_deduction",
            ),
        ),
        derived_spm_overrides=(
            (
                "snap_earned_income",
                "difference_floor_zero",
                (
                    "snap_earned_income_before_exclusions",
                    "snap_child_earned_income_exclusion",
                    "snap_other_earned_income_exclusions",
                    "snap_work_support_public_assistance_income",
                ),
            ),
        ),
        annual_derived_spm_overrides=(
            (
                "spm_unit_pre_subsidy_childcare_expenses",
                "difference_floor_zero_annualized",
                (
                    "snap_dependent_care_actual_costs",
                    "snap_dependent_care_excluded_expenses",
                ),
            ),
            (
                "spm_unit_pre_subsidy_childcare_expenses",
                "monthly_to_annual",
                ("snap_dependent_care_deduction",),
            ),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_utility_allowance",),
        pe_var="snap_standard_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_limited_utility_allowance",),
        pe_var="snap_limited_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_individual_utility_allowance",),
        pe_var="snap_individual_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_state_using_standard_utility_allowance",),
        pe_var="snap_state_using_standard_utility_allowance",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_state_uses_child_support_deduction",),
        pe_var="snap_state_uses_child_support_deduction",
        default_state_code="TN",
        parameter_path="gov.usda.snap.income.deductions.child_support",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_self_employment_expense_based_deduction_applies",),
        pe_var="snap_self_employment_expense_based_deduction_applies",
        default_state_code="CA",
        parameter_path=(
            "gov.usda.snap.income.deductions.self_employment."
            "expense_based_deduction_applies"
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_self_employment_simplified_deduction_rate",),
        pe_var="snap_self_employment_simplified_deduction_rate",
        default_state_code="MD",
        parameter_path="gov.usda.snap.income.deductions.self_employment.rate",
        parameter_value_mode="float",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_medical_expense_deduction",),
        pe_var="snap_standard_medical_expense_deduction",
        parameter_path="gov.usda.snap.income.deductions.excess_medical_expense.standard",
        parameter_value_mode="float",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_homeless_shelter_deduction_available",),
        pe_var="snap_homeless_shelter_deduction_available",
        parameter_path=(
            "gov.usda.snap.income.deductions.excess_shelter_expense.homeless.available"
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_tanf_non_cash_gross_income_limit_fpg_ratio",),
        pe_var="snap_tanf_non_cash_gross_income_limit_fpg_ratio",
        default_state_code="TX",
        parameter_path="gov.hhs.tanf.non_cash.income_limit.gross",
        parameter_value_mode="float",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_tanf_non_cash_asset_limit",),
        pe_var="snap_tanf_non_cash_asset_limit",
        default_state_code="TX",
        parameter_path="gov.hhs.tanf.non_cash.asset_limit",
        parameter_value_mode="float",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_asset_test",),
        pe_var="meets_snap_asset_test",
        monthly=True,
        spm=True,
        annual_direct_spm_overrides=(
            ("snap_assets", "snap_assets"),
            ("snap_countable_resources", "snap_assets"),
            ("snap_countable_financial_resources", "snap_assets"),
            ("snap_financial_resources", "snap_assets"),
            (
                "snap_household_has_elderly_or_disabled_member",
                "has_usda_elderly_disabled",
            ),
        ),
        annual_derived_spm_overrides=(
            (
                "snap_assets",
                "difference",
                (
                    "snap_total_resources_before_exclusions",
                    "snap_mandatory_retirement_account_resource_exclusion",
                    "snap_discretionary_retirement_account_resource_exclusion",
                    "snap_mandatory_education_account_resource_exclusion",
                    "snap_discretionary_education_account_resource_exclusion",
                    "snap_other_resource_exclusions_under_g",
                ),
            ),
        ),
        unsupported_input_keys=(
            "snap_statutory_asset_limit",
            "snap_applicable_asset_limit",
            "snap_asset_limit",
            "snap_asset_limit_with_elderly_or_disabled_member",
        ),
        unsupported_input_reason=(
            "RuleSpec test restates the SNAP asset-test threshold with local "
            "limit/resource abstractions that PolicyEngine US does not expose "
            "as scenario inputs"
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_gross_income_test",),
        pe_var="meets_snap_gross_income_test",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_net_income_test",),
        pe_var="meets_snap_net_income_test",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("is_snap_eligible",),
        pe_var="is_snap_eligible",
        monthly=True,
        spm=True,
        boolean_person_inputs=(
            ("is_snap_ineligible_student", "is_snap_ineligible_student"),
        ),
        monthly_boolean_person_inputs=(
            (
                "is_snap_immigration_status_eligible",
                "is_snap_immigration_status_eligible",
            ),
        ),
        direct_spm_overrides=(
            ("meets_snap_gross_income_test", "meets_snap_gross_income_test"),
            ("meets_snap_net_income_test", "meets_snap_net_income_test"),
            ("meets_snap_asset_test", "meets_snap_asset_test"),
            (
                "meets_snap_categorical_eligibility",
                "meets_snap_categorical_eligibility",
            ),
            ("meets_snap_work_requirements", "meets_snap_work_requirements"),
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_deduction",),
        pe_var="snap_standard_deduction",
        monthly=True,
        spm=True,
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_child_support_deduction",),
        pe_var="snap_child_support_gross_income_deduction",
        monthly=True,
        spm=True,
        annualized_person_inputs=(
            ("snap_child_support_payments_made", "child_support_expense"),
        ),
        state_code_from_boolean_input=(
            "snap_state_uses_child_support_deduction",
            "TX",
            "CA",
        ),
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_excess_medical_expense_deduction",),
        pe_var="snap_excess_medical_expense_deduction",
        monthly=True,
        spm=True,
        annualized_person_inputs=(
            (
                "snap_allowable_medical_expenses_before_threshold",
                "medical_out_of_pocket_expenses",
            ),
        ),
        boolean_person_inputs=(
            (
                "snap_household_has_elderly_or_disabled_member",
                "is_usda_disabled",
            ),
        ),
        default_state_code="NY",
    ),
    _PolicyEngineUSVarAdapter(
        rule_names=("snap_maximum_allotment",),
        pe_var="snap_max_allotment",
        monthly=True,
        spm=True,
    ),
)

_PE_US_VARIABLE_MAP = {
    rule_name: adapter.pe_var
    for adapter in _PE_US_VAR_ADAPTERS
    for rule_name in adapter.rule_names
}

_PE_US_VAR_ADAPTERS_BY_NAME = {
    name: adapter
    for adapter in _PE_US_VAR_ADAPTERS
    for name in (adapter.pe_var, *adapter.rule_names)
}

_PE_US_MONTHLY_VAR_NAMES = {
    name
    for adapter in _PE_US_VAR_ADAPTERS
    if adapter.monthly
    for name in (adapter.pe_var, *adapter.rule_names)
}

_PE_US_SPM_VAR_NAMES = {
    name
    for adapter in _PE_US_VAR_ADAPTERS
    if adapter.spm
    for name in (adapter.pe_var, *adapter.rule_names)
}


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
    cleaned = text.strip()
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
SOURCE_TEXT_NUMBER_PATTERN = re.compile(r"(?:^|(?<=[\s$£€(\[,]))(-?[\d,]+(?:\.\d+)?)\b")
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
_SOURCE_REFERENCE_TARGET_PATTERN = r"(?:\([^)]+\)|\d+[A-Za-z./-]*(?:\([^)]+\))*(?=$|[\s,.;:])|[ivxlcdm]+\b|[A-Z]{1,4}\b|[a-z]\b)"
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
}
_CARDINAL_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _CARDINAL_WORD_VALUES) + r")\b",
    re.IGNORECASE,
)
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
    r"section\s+([0-9A-Za-z.-]+(?:\([^)]+\))*)",
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
        if _is_structural_schedule_index_literal(cleaned, raw):
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

    for match in re.finditer(r"(?:^|(?<=[\s$£€(\[,]))(-?[\d,]+(?:\.\d+)?)\b", text):
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
        re.compile(r"(\d+(?:\.\d+)?)\s+(?:percent|per\s*cent(?:um)?)", re.IGNORECASE),
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
    source = (
        source_text
        if source_text is not None
        else extract_embedded_source_text(content)
    ).strip()
    if not source:
        return []

    source_numbers = extract_numbers_from_text(source)
    issues: list[str] = []
    for _, raw, value in extract_grounding_values(content):
        if numeric_value_is_grounded(value, source_numbers):
            continue
        display = raw if raw == f"{value:g}" else f"{raw} ({value:g})"
        issues.append(
            "Ungrounded generated numeric literal: "
            f"{display} does not appear as a substantive numeric value in the source text."
        )
    return issues


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

    path = _normalized_rulespec_path(rules_file)
    imports = {
        str(item).strip() for item in payload.get("imports") or [] if str(item).strip()
    }
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
    for contract in _UPSTREAM_PLACEMENT_CONTRACTS:
        issues.extend(
            _find_upstream_placement_contract_issues(
                contract=contract,
                rules=rules,
                path=path,
                imports=imports,
            )
        )
    return issues


def _normalized_rulespec_path(rules_file: Path | None) -> str:
    if rules_file is None:
        return ""
    return rules_file.as_posix().lstrip("./").lower()


_SOURCE_METADATA_REITERATION_RELATIONS = {
    "reiterate",
    "reiterates",
    "reiterated",
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
_RULE_METADATA_SOURCE_RELATIONS = (*_RULE_METADATA_TARGET_KEYS, "reiterates")
_RULE_METADATA_SOURCE_RELATION_SET = frozenset(_RULE_METADATA_SOURCE_RELATIONS)


def _find_rule_metadata_schema_issues(rules: list[Any]) -> list[str]:
    """Validate generic source relation metadata on executable RuleSpec rules."""
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

        source_relation = _rule_metadata_source_relation(rule)
        target_keys = [
            key
            for key in _RULE_METADATA_TARGET_KEYS
            if list(_iter_relation_target_values(metadata.get(key)))
        ]

        if (
            source_relation is not None
            and source_relation not in _RULE_METADATA_SOURCE_RELATION_SET
        ):
            allowed = ", ".join(
                f"`{relation}`" for relation in _RULE_METADATA_SOURCE_RELATIONS
            )
            issues.append(
                f"RuleSpec relation metadata has unknown source relation: rule "
                f"`{rule_name}` declares `metadata.source_relation: "
                f"{source_relation}`, but allowed values are {allowed}."
            )
            continue

        if target_keys and source_relation is None:
            targets = ", ".join(f"`metadata.{key}`" for key in target_keys)
            issues.append(
                f"RuleSpec relation metadata is missing source relation: rule "
                f"`{rule_name}` declares {targets} and must also declare "
                "`metadata.source_relation`."
            )
            continue

        if source_relation == "reiterates":
            if str(rule.get("kind") or "").strip().lower() != "reiteration":
                issues.append(
                    f"RuleSpec relation metadata has executable reiteration: rule "
                    f"`{rule_name}` declares `metadata.source_relation: reiterates`; "
                    "use `kind: reiteration` with `reiterates.target` instead."
                )
            continue

        if source_relation == "defines":
            has_target = any(_iter_relation_target_values(metadata.get("defines")))
            has_concept_id = bool(str(metadata.get("concept_id") or "").strip())
            if not has_target and not has_concept_id:
                issues.append(
                    f"RuleSpec relation metadata is incomplete: rule `{rule_name}` "
                    "declares `metadata.source_relation: defines` and must also "
                    "declare `metadata.defines` or `metadata.concept_id`."
                )
            continue

        if source_relation in _RULE_METADATA_TARGET_KEYS and not any(
            _iter_relation_target_values(metadata.get(source_relation))
        ):
            issues.append(
                f"RuleSpec relation metadata is incomplete: rule `{rule_name}` "
                f"declares `metadata.source_relation: {source_relation}` and "
                f"must also declare `metadata.{source_relation}`."
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
        if relation in _SOURCE_METADATA_REITERATION_RELATIONS:
            if _rules_include_reiteration_target(rules, target):
                continue
            issues.append(
                "Source metadata upstream relation requires reiteration: "
                f"source metadata says this source `{relation}` `{target}`, so "
                "encode it as `kind: reiteration` with `reiterates.target` "
                "instead of redefining executable policy locally."
            )
            continue

        metadata_key = _SOURCE_METADATA_DECLARATIVE_RELATIONS.get(relation)
        if metadata_key is None:
            continue
        if _rules_include_metadata_relation_target(rules, metadata_key, target):
            continue
        issues.append(
            "Source metadata upstream relation not recorded: "
            f"source metadata says this source `{relation}` `{target}`, so "
            "the corresponding RuleSpec rule must declare "
            f"`metadata.source_relation: {metadata_key}` and "
            f"`metadata.{metadata_key}: {target}`."
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


def _rules_include_reiteration_target(rules: list[Any], target: str) -> bool:
    return any(
        isinstance(rule, dict)
        and str(rule.get("kind") or "").lower() == "reiteration"
        and _target_matches(
            _normalize_relation_target(
                (rule.get("reiterates") or {}).get("target")
                if isinstance(rule.get("reiterates"), dict)
                else None
            ),
            target,
        )
        for rule in rules
    )


def _rules_include_metadata_relation_target(
    rules: list[Any],
    metadata_key: str,
    target: str,
) -> bool:
    return any(
        isinstance(rule, dict)
        and _rule_metadata_source_relation(rule) == metadata_key
        and any(
            _target_matches(candidate, target)
            for candidate in _iter_rule_metadata_targets(rule, metadata_key)
        )
        for rule in rules
    )


def _rule_metadata_source_relation(rule: dict[str, Any]) -> str | None:
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict):
        return None
    relation = str(metadata.get("source_relation") or "").strip().lower()
    return relation or None


def _iter_rule_metadata_targets(
    rule: dict[str, Any],
    metadata_key: str,
) -> Iterable[str]:
    metadata = rule.get("metadata")
    if isinstance(metadata, dict):
        yield from _iter_relation_target_values(metadata.get(metadata_key))


def _iter_relation_target_values(value: Any) -> Iterable[str]:
    target = _normalize_relation_target(value)
    if target:
        yield target
        return

    if isinstance(value, dict):
        target = _normalize_relation_target(value.get("target"))
        if target:
            yield target
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_relation_target_values(item)


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


@dataclass(frozen=True)
class UpstreamPlacementTargetPattern:
    """A name pattern and the canonical import target it should resolve to."""

    pattern: str
    target: str


@dataclass(frozen=True)
class UpstreamPlacementContract:
    """Declarative rule for keeping encodings at their upstream authority."""

    authority_label: str
    placement_noun: str
    allowed_path_patterns: tuple[str, ...]
    canonical_import: str | None = None
    exact_targets: dict[str, str] = field(default_factory=dict)
    target_patterns: tuple[UpstreamPlacementTargetPattern, ...] = ()
    reference_symbols: tuple[str, ...] = ()
    definition_name_patterns: tuple[str, ...] = ()
    definition_formula_patterns: tuple[str, ...] = ()
    definition_target: str | None = None
    context_path_patterns: tuple[str, ...] = ()
    context_rule_name_patterns: tuple[str, ...] = ()
    context_source_terms: tuple[str, ...] = ()


_SNAP_CONTEXT_PATH_PATTERNS = (
    r"(?:^|/)policies/[^/]+/snap(?:/|$)",
    r"(?:^|/)regulations/10-ccr-2506-1(?:/|$)",
)
_SNAP_CONTEXT_RULE_NAME_PATTERNS = (r"(?:^|_)snap(?:_|$)",)
_SNAP_CONTEXT_SOURCE_TERMS = (
    "snap",
    "supplemental nutrition assistance",
)

_UPSTREAM_PLACEMENT_CONTRACTS: tuple[UpstreamPlacementContract, ...] = (
    UpstreamPlacementContract(
        authority_label="7 USC 2012(j)",
        placement_noun="the SNAP elderly-or-disabled member category",
        allowed_path_patterns=(r"(?:^|/)statutes/7/2012/j\.yaml$",),
        canonical_import="us:statutes/7/2012/j",
        reference_symbols=("snap_household_has_elderly_or_disabled_member",),
        definition_name_patterns=(
            r"^elderly_or_disabled_household$",
            r"^snap_household_has_.*elderly.*disabled.*member$",
        ),
        definition_formula_patterns=(
            r"count_where\s*\(\s*member_of_household\s*,\s*"
            r"snap_member_is_[a-z0-9_]*elderly[a-z0-9_]*disabled[a-z0-9_]*\s*\)",
        ),
        definition_target=(
            "us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member"
        ),
        context_path_patterns=_SNAP_CONTEXT_PATH_PATTERNS,
        context_rule_name_patterns=_SNAP_CONTEXT_RULE_NAME_PATTERNS,
        context_source_terms=_SNAP_CONTEXT_SOURCE_TERMS,
    ),
    UpstreamPlacementContract(
        authority_label="7 USC 2017(a)",
        placement_noun="the federal SNAP regular allotment formula",
        allowed_path_patterns=(r"(?:^|/)statutes/7/2017/a\.yaml$",),
        canonical_import="us:statutes/7/2017/a",
        exact_targets={
            "snap_household_food_contribution_rate": (
                "us:statutes/7/2017/a#snap_household_food_contribution_rate"
            ),
            "snap_net_income_for_allotment": (
                "us:statutes/7/2017/a#snap_net_income_for_allotment"
            ),
            "net_income_for_benefit_formula": (
                "us:statutes/7/2017/a#snap_net_income_for_allotment"
            ),
            "snap_household_food_contribution": (
                "us:statutes/7/2017/a#snap_household_food_contribution"
            ),
            "household_food_contribution": (
                "us:statutes/7/2017/a#snap_household_food_contribution"
            ),
            "snap_allotment_before_minimum": (
                "us:statutes/7/2017/a#snap_allotment_before_minimum"
            ),
            "snap_minimum_monthly_allotment": (
                "us:statutes/7/2017/a#snap_minimum_monthly_allotment"
            ),
            "snap_regular_month_allotment": (
                "us:statutes/7/2017/a#snap_regular_month_allotment"
            ),
        },
        reference_symbols=(
            "snap_household_food_contribution_rate",
            "snap_net_income_for_allotment",
            "net_income_for_benefit_formula",
            "snap_household_food_contribution",
            "household_food_contribution",
            "snap_allotment_before_minimum",
            "snap_minimum_monthly_allotment",
            "snap_regular_month_allotment",
        ),
        context_path_patterns=_SNAP_CONTEXT_PATH_PATTERNS,
        context_rule_name_patterns=_SNAP_CONTEXT_RULE_NAME_PATTERNS,
        context_source_terms=_SNAP_CONTEXT_SOURCE_TERMS,
    ),
    UpstreamPlacementContract(
        authority_label="7 USC 2014(e)(2)",
        placement_noun="the federal SNAP earned-income deduction",
        allowed_path_patterns=(r"(?:^|/)statutes/7/2014/e/2\.yaml$",),
        canonical_import="us:statutes/7/2014/e/2",
        exact_targets={
            "snap_earned_income_deduction_rate": (
                "us:statutes/7/2014/e/2#snap_earned_income_deduction_rate"
            ),
            "snap_earned_income_subject_to_deduction": (
                "us:statutes/7/2014/e/2#snap_earned_income_subject_to_deduction"
            ),
            "snap_earned_income_deduction": (
                "us:statutes/7/2014/e/2#snap_earned_income_deduction"
            ),
            "earned_income_deduction": (
                "us:statutes/7/2014/e/2#snap_earned_income_deduction"
            ),
        },
        reference_symbols=(
            "snap_earned_income_deduction_rate",
            "snap_earned_income_subject_to_deduction",
            "snap_earned_income_deduction",
            "earned_income_deduction",
        ),
        context_path_patterns=_SNAP_CONTEXT_PATH_PATTERNS,
        context_rule_name_patterns=_SNAP_CONTEXT_RULE_NAME_PATTERNS,
        context_source_terms=_SNAP_CONTEXT_SOURCE_TERMS,
    ),
    UpstreamPlacementContract(
        authority_label="USDA income eligibility standards",
        placement_noun="the federal SNAP USDA income eligibility standards",
        allowed_path_patterns=(
            r"(?:^|/)policies/usda/snap/fy-\d{4}-cola/"
            r"income-eligibility-standards\.yaml$",
        ),
        canonical_import=(
            "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards"
        ),
        exact_targets={
            "gross_income_limit_table": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_gross_income_limit_130_percent_fpl_48_states_dc_table"
            ),
            "gross_income_limit_additional_member": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_gross_income_limit_130_percent_fpl_48_states_dc_additional_member"
            ),
            "gross_income_limit": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_gross_income_limit_130_percent_fpl_48_states_dc"
            ),
            "net_income_limit_table": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_net_income_limit_100_percent_fpl_48_states_dc_table"
            ),
            "net_income_limit_additional_member": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_net_income_limit_100_percent_fpl_48_states_dc_additional_member"
            ),
            "net_income_limit": (
                "us:policies/usda/snap/fy-<year>-cola/"
                "income-eligibility-standards"
                "#snap_net_income_limit_100_percent_fpl_48_states_dc"
            ),
        },
        target_patterns=(
            UpstreamPlacementTargetPattern(
                pattern=r"^snap_net_income_limit_100_percent_fpl(?:$|_48_states_dc)",
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/"
                    "income-eligibility-standards"
                    "#snap_net_income_limit_100_percent_fpl_48_states_dc"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=r"^snap_gross_income_limit_130_percent_fpl(?:$|_48_states_dc)",
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/"
                    "income-eligibility-standards"
                    "#snap_gross_income_limit_130_percent_fpl_48_states_dc"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=r"^snap_gross_income_limit_165_percent_fpl(?:$|_48_states_dc)",
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/"
                    "income-eligibility-standards"
                    "#snap_gross_income_limit_165_percent_fpl_48_states_dc"
                ),
            ),
        ),
        reference_symbols=(
            "snap_net_income_limit_100_percent_fpl_48_states_dc",
            "snap_gross_income_limit_130_percent_fpl_48_states_dc",
            "snap_gross_income_limit_165_percent_fpl_48_states_dc",
            "gross_income_limit",
            "net_income_limit",
        ),
        context_path_patterns=_SNAP_CONTEXT_PATH_PATTERNS,
        context_rule_name_patterns=_SNAP_CONTEXT_RULE_NAME_PATTERNS,
        context_source_terms=_SNAP_CONTEXT_SOURCE_TERMS,
    ),
    UpstreamPlacementContract(
        authority_label="USDA COLA policy file",
        placement_noun="a federal SNAP annual COLA value",
        allowed_path_patterns=(r"(?:^|/)policies/usda/snap/fy-\d{4}-cola/",),
        exact_targets={
            "standard_deduction_table": (
                "us:policies/usda/snap/fy-<year>-cola/deductions"
                "#snap_standard_deduction_48_states_dc_table"
            ),
            "standard_deduction": (
                "us:policies/usda/snap/fy-<year>-cola/deductions"
                "#snap_standard_deduction"
            ),
            "excess_shelter_deduction_cap": (
                "us:policies/usda/snap/fy-<year>-cola/deductions"
                "#snap_maximum_excess_shelter_deduction_48_states_dc"
            ),
            "snap_homeless_shelter_deduction_amount": (
                "us:policies/usda/snap/fy-<year>-cola/deductions"
                "#snap_homeless_shelter_deduction"
            ),
            "snap_resource_limit": (
                "us:policies/usda/snap/fy-<year>-cola/deductions#snap_asset_limit"
            ),
            "snap_asset_limit": (
                "us:policies/usda/snap/fy-<year>-cola/deductions#snap_asset_limit"
            ),
        },
        target_patterns=(
            UpstreamPlacementTargetPattern(
                pattern=(
                    r"^snap_maximum_allotment"
                    r"(?:$|_table$|_additional_member$|_alaska|_guam|_hawaii|_virgin_islands)"
                ),
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/maximum-allotments"
                    "#snap_maximum_allotment"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=(
                    r"^snap_standard_deduction"
                    r"(?:$|_48_states_dc|_alaska|_guam|_hawaii|_virgin_islands)"
                ),
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/deductions"
                    "#snap_standard_deduction_48_states_dc"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=(
                    r"^snap_asset_limit_"
                    r"(?:elderly_or_disabled_member|other_households)$"
                ),
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/deductions#snap_asset_limit"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=(
                    r"^snap_maximum_excess_shelter_deduction"
                    r"(?:$|_48_states_dc|_alaska|_guam|_hawaii|_virgin_islands)"
                ),
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/deductions"
                    "#snap_maximum_excess_shelter_deduction_48_states_dc"
                ),
            ),
            UpstreamPlacementTargetPattern(
                pattern=r"^snap_homeless_shelter_deduction$",
                target=(
                    "us:policies/usda/snap/fy-<year>-cola/deductions"
                    "#snap_homeless_shelter_deduction"
                ),
            ),
        ),
        context_path_patterns=_SNAP_CONTEXT_PATH_PATTERNS,
        context_rule_name_patterns=_SNAP_CONTEXT_RULE_NAME_PATTERNS,
        context_source_terms=_SNAP_CONTEXT_SOURCE_TERMS,
    ),
)


def _find_upstream_placement_contract_issues(
    *,
    contract: UpstreamPlacementContract,
    rules: list[Any],
    path: str,
    imports: set[str],
) -> list[str]:
    if _path_matches_any(path, contract.allowed_path_patterns):
        return []

    issues: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").lower() == "reiteration":
            continue
        if not _rule_matches_upstream_contract_context(
            contract=contract,
            path=path,
            rule=rule,
        ):
            continue

        name = str(rule.get("name") or "<unknown>")
        target = _upstream_placement_target(contract, rule)
        if target is not None:
            issues.append(_upstream_placement_issue(contract, name, target))
            continue

        referenced_symbol = _referenced_upstream_symbol(
            rule,
            contract.reference_symbols,
        )
        if (
            referenced_symbol is not None
            and contract.canonical_import is not None
            and contract.canonical_import not in imports
        ):
            issues.append(
                "Upstream import required: "
                f"`{name}` references {contract.placement_noun} symbol "
                f"`{referenced_symbol}` but does not import "
                f"`{contract.canonical_import}`."
            )

    return issues


def _upstream_placement_issue(
    contract: UpstreamPlacementContract,
    name: str,
    target: str,
) -> str:
    return (
        "Upstream placement violation: "
        f"`{name}` appears to encode {contract.placement_noun} outside "
        f"{contract.authority_label}. Move the rule to `{target}` and use an "
        "import or a non-executable `reiteration` marker in downstream "
        "policy/manual files."
    )


def _upstream_placement_target(
    contract: UpstreamPlacementContract,
    rule: dict[str, Any],
) -> str | None:
    if _rule_matches_definition_patterns(contract, rule):
        return contract.definition_target or contract.canonical_import

    name = str(rule.get("name") or "").lower()
    target = contract.exact_targets.get(name)
    if target is not None:
        return target

    for target_pattern in contract.target_patterns:
        if re.search(target_pattern.pattern, name, flags=re.IGNORECASE):
            return target_pattern.target
    return None


def _rule_matches_definition_patterns(
    contract: UpstreamPlacementContract,
    rule: dict[str, Any],
) -> bool:
    if not (contract.definition_name_patterns or contract.definition_formula_patterns):
        return False

    name = str(rule.get("name") or "").lower()
    if any(
        re.search(pattern, name, flags=re.IGNORECASE)
        for pattern in contract.definition_name_patterns
    ):
        return True

    return any(
        re.search(pattern, formula, flags=re.IGNORECASE)
        for formula in _iter_rulespec_formula_strings(rule)
        for pattern in contract.definition_formula_patterns
    )


def _rule_matches_upstream_contract_context(
    *,
    contract: UpstreamPlacementContract,
    path: str,
    rule: dict[str, Any],
) -> bool:
    if not (
        contract.context_path_patterns
        or contract.context_rule_name_patterns
        or contract.context_source_terms
    ):
        return True

    if _path_matches_any(path, contract.context_path_patterns):
        return True

    name = str(rule.get("name") or "").lower()
    if any(
        re.search(pattern, name, flags=re.IGNORECASE)
        for pattern in contract.context_rule_name_patterns
    ):
        return True

    source = str(rule.get("source") or "").lower()
    return any(term.lower() in source for term in contract.context_source_terms)


def _path_matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, path, flags=re.IGNORECASE) for pattern in patterns)


def _referenced_upstream_symbol(
    rule: dict[str, Any],
    symbols: Iterable[str],
) -> str | None:
    for formula in _iter_rulespec_formula_strings(rule):
        for symbol in symbols:
            if re.search(rf"\b{re.escape(symbol)}\b", formula):
                return symbol
    return None


def _iter_rulespec_formula_strings(rule: dict[str, Any]) -> Iterable[str]:
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return
    for version in versions:
        if not isinstance(version, dict):
            continue
        formula = version.get("formula")
        if isinstance(formula, str):
            yield formula


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

    citation_path = str(source_verification.get("corpus_citation_path") or "").strip()
    expected_values = source_verification.get("values")
    if not citation_path:
        return [
            "Source verification corpus path required: missing `corpus_citation_path`."
        ]
    if not isinstance(expected_values, dict) or not expected_values:
        return [
            "Source verification values required: "
            "`source_verification.values` must list RuleSpec values to verify."
        ]

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

    source_text = (
        source_texts.get(citation_path)
        if source_texts is not None
        else _fetch_corpus_source_text(citation_path)
    )
    if source_text is None:
        issues.append(
            "Source verification corpus source missing: "
            f"`{citation_path}` was not found in corpus.provisions."
        )
        return issues

    for value_name, expected_value in expected_values.items():
        value_key = str(value_name)
        issues.extend(
            _find_source_text_value_issues(
                citation_path=citation_path,
                source_text=source_text,
                value_name=value_key,
                expected_value=expected_value,
            )
        )

    return issues


def _source_verification_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    module = payload.get("module")
    if isinstance(module, dict) and isinstance(module.get("source_verification"), dict):
        return module["source_verification"]
    source_verification = payload.get("source_verification")
    if isinstance(source_verification, dict):
        return source_verification
    return None


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
            if not _reiteration_values_equal(expected_cell, actual_cell):
                issues.append(
                    "Source verification RuleSpec mismatch: "
                    f"`{value_name}[{cell_key}]` declares "
                    f"{_format_reiteration_value(expected_cell)}, but RuleSpec has "
                    f"{_format_reiteration_value(actual_cell)}."
                )
        return issues

    if isinstance(rulespec_value, dict):
        return [
            "Source verification RuleSpec mismatch: "
            f"`{value_name}` is declared as a scalar but the RuleSpec value is a table."
        ]
    if _reiteration_values_equal(expected_value, rulespec_value):
        return []
    return [
        "Source verification RuleSpec mismatch: "
        f"`{value_name}` declares {_format_reiteration_value(expected_value)}, "
        f"but RuleSpec has {_format_reiteration_value(rulespec_value)}."
    ]


def _find_source_text_value_issues(
    *,
    citation_path: str,
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
                    f"`{citation_path}` does not contain `{value_name}[{cell_key}]` = "
                    f"{_format_reiteration_value(expected_cell)}."
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
        f"`{citation_path}` does not contain `{value_name}` = "
        f"{_format_reiteration_value(expected_value)}."
    ]


def _fetch_corpus_source_text(citation_path: str) -> str | None:
    """Fetch a corpus.provisions body by exact citation path from Supabase."""
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
        f"{supabase_url}/rest/v1/provisions?{params}",
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
    value_text = _source_verification_numeric_text(value)
    if value_text is None:
        return False
    index_text = re.escape(str(index).replace(",", ""))
    value_pattern = re.escape(value_text)
    return bool(re.search(rf"(?<!\d){index_text}\s+\${value_pattern}(?!\d)", text))


def _source_text_contains_scalar_value(text: str, value: Any) -> bool:
    value_text = _source_verification_numeric_text(value)
    if value_text is None:
        return str(value).strip() in text
    value_pattern = re.escape(value_text)
    return bool(re.search(rf"(?:\$|\+)?{value_pattern}(?!\d)", text))


def _source_text_contains_table_value_multiset(
    text: str,
    values: Iterable[Any],
) -> bool:
    expected_counts: Counter[str] = Counter()
    for value in values:
        value_text = _source_verification_numeric_text(value)
        if value_text is None:
            return False
        expected_counts[value_text] += 1

    return all(
        _source_text_value_occurrence_count(text, value_text) >= expected_count
        for value_text, expected_count in expected_counts.items()
    )


def _source_text_value_occurrence_count(text: str, value_text: str) -> int:
    value_pattern = re.escape(value_text)
    return len(re.findall(rf"(?:\$|\+)?{value_pattern}(?!\d)", text))


def _source_verification_numeric_text(value: Any) -> str | None:
    numeric = _numeric_rule_value(value)
    if numeric is None:
        return None
    raw, number = numeric
    if float(number).is_integer():
        return str(int(number))
    return raw.replace(",", "")


@dataclass(frozen=True)
class _ReiterationTargetRef:
    """Parsed canonical RuleSpec target for a reiteration marker."""

    prefix: str
    repo_name: str
    relative_path: Path
    symbol: str | None


def find_reiteration_issues(
    content: str,
    *,
    policy_repo_path: Path | None = None,
) -> list[str]:
    """Validate non-executable reiteration markers."""
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
        if str(rule.get("kind") or "").lower() != "reiteration":
            continue
        name = str(rule.get("name") or "<unknown>")
        reiterates = rule.get("reiterates")
        target = ""
        target_ref: _ReiterationTargetRef | None = None
        if (
            not isinstance(reiterates, dict)
            or not str(reiterates.get("target") or "").strip()
        ):
            issues.append(
                "Reiteration target required: "
                f"{name} must declare `reiterates.target` pointing to the canonical RuleSpec rule."
            )
        else:
            target = str(reiterates.get("target") or "").strip()
            target_ref = _parse_reiteration_target(target)
            if target_ref is None:
                issues.append(
                    "Reiteration target invalid: "
                    f"{name} uses `{target}`, expected `<jurisdiction>:<path>#<rule>`."
                )
            elif target_ref.symbol is None:
                issues.append(
                    "Reiteration target rule required: "
                    f"{name} must point to a specific RuleSpec rule with `#rule_name`."
                )
        if rule.get("versions"):
            issues.append(
                "Reiteration must be non-executable: "
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
                _find_reiteration_value_verification_issues(
                    name=name,
                    target=target,
                    target_ref=target_ref,
                    expected_values=verification["values"],
                    policy_repo_path=policy_repo_path,
                )
            )
    return issues


def _find_reiteration_value_verification_issues(
    *,
    name: str,
    target: str,
    target_ref: _ReiterationTargetRef,
    expected_values: dict[Any, Any],
    policy_repo_path: Path | None,
) -> list[str]:
    """Compare a reiteration's expected values against its canonical target file."""
    target_file = _resolve_reiteration_target_file(target_ref, policy_repo_path)
    if target_file is None:
        return [
            "Reiteration verification target unavailable: "
            f"{name} points to `{target}`, but repository `{target_ref.repo_name}` "
            "was not found."
        ]

    target_values, target_symbols, load_issue = _extract_reiteration_target_values(
        target_file
    )
    if load_issue is not None:
        return [
            "Reiteration verification target invalid: "
            f"{name} points to `{target}`, but {load_issue}"
        ]

    issues: list[str] = []
    if target_ref.symbol and target_ref.symbol not in target_symbols:
        issues.append(
            "Reiteration target rule missing: "
            f"{name} points to `{target}`, but `{target_ref.symbol}` is not defined "
            f"in {target_ref.relative_path}."
        )

    for value_name, expected_value in expected_values.items():
        value_key = str(value_name)
        if value_key not in target_values:
            issues.append(
                "Reiteration verification target missing value: "
                f"{name} expects `{value_key}` but `{target}` does not define it."
            )
            continue
        actual_value = target_values[value_key]
        issues.extend(
            _compare_reiteration_verification_value(
                name=name,
                target=target,
                value_name=value_key,
                expected_value=expected_value,
                actual_value=actual_value,
            )
        )

    return issues


def _parse_reiteration_target(target: str) -> _ReiterationTargetRef | None:
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
    return _ReiterationTargetRef(
        prefix=prefix,
        repo_name=f"rules-{prefix}",
        relative_path=relative_path,
        symbol=symbol.strip() if symbol and symbol.strip() else None,
    )


def _resolve_reiteration_target_file(
    target_ref: _ReiterationTargetRef,
    policy_repo_path: Path | None,
) -> Path | None:
    """Resolve a canonical RuleSpec target file across sibling/CI checkouts."""
    for root in _candidate_reiteration_repo_roots(
        target_ref.repo_name,
        policy_repo_path,
    ):
        target_file = root / target_ref.relative_path
        if target_file.exists():
            return target_file
    return None


def _candidate_reiteration_repo_roots(
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

    env_roots = os.environ.get("AXIOM_RULE_REPO_ROOTS", "")
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


def _extract_reiteration_target_values(
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


def _compare_reiteration_verification_value(
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
                "Reiteration verification mismatch: "
                f"{name} expects `{value_name}` to be a table, but `{target}` "
                "defines a scalar."
            ]
        issues: list[str] = []
        for raw_key, expected_cell in expected_value.items():
            cell_key = str(raw_key)
            if cell_key not in actual_value:
                issues.append(
                    "Reiteration verification target missing value: "
                    f"{name} expects `{value_name}[{cell_key}]` but `{target}` "
                    "does not define it."
                )
                continue
            actual_cell = actual_value[cell_key]
            if not _reiteration_values_equal(expected_cell, actual_cell):
                issues.append(
                    "Reiteration verification mismatch: "
                    f"{name} expects `{value_name}[{cell_key}]` = "
                    f"{_format_reiteration_value(expected_cell)}, but `{target}` "
                    f"has {_format_reiteration_value(actual_cell)}."
                )
        return issues

    if isinstance(actual_value, dict):
        return [
            "Reiteration verification mismatch: "
            f"{name} expects `{value_name}` to be a scalar, but `{target}` "
            "defines a table."
        ]
    if _reiteration_values_equal(expected_value, actual_value):
        return []
    return [
        "Reiteration verification mismatch: "
        f"{name} expects `{value_name}` = "
        f"{_format_reiteration_value(expected_value)}, but `{target}` has "
        f"{_format_reiteration_value(actual_value)}."
    ]


def _reiteration_values_equal(expected_value: Any, actual_value: Any) -> bool:
    """Compare verification scalar values, allowing numeric text/int equivalence."""
    expected_numeric = _numeric_rule_value(expected_value)
    actual_numeric = _numeric_rule_value(actual_value)
    if expected_numeric is not None and actual_numeric is not None:
        return math.isclose(expected_numeric[1], actual_numeric[1], abs_tol=1e-12)
    return str(expected_value).strip() == str(actual_value).strip()


def _format_reiteration_value(value: Any) -> str:
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
    return any(
        math.isclose(value, source_value, rel_tol=0, abs_tol=1e-12)
        for source_value in source_numbers
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
    prompt_sha256: Optional[str] = None


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
        """Build an env that prefers the configured Axiom Rules checkout."""
        env = dict(os.environ)
        rules_src = self.axiom_rules_path / "src"
        if rules_src.exists():
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{rules_src}{os.pathsep}{existing}" if existing else str(rules_src)
            )
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
        """Resolve the local Axiom Rules CLI binary."""
        candidates = [
            self.axiom_rules_path / "target" / "debug" / "axiom-rules",
            self.axiom_rules_path / "target" / "release" / "axiom-rules",
            self.axiom_rules_path / "axiom-rules",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        if resolved := shutil.which("axiom-rules"):
            return Path(resolved)
        raise FileNotFoundError(
            f"axiom-rules binary not found under {self.axiom_rules_path} or on PATH"
        )

    def _compile_rulespec_to_artifact(
        self,
        rules_file: Path,
        output_path: Path,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, Any] | None]:
        """Compile RuleSpec YAML to an Axiom Rules artifact JSON file."""
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
        return f"Successfully compiled {rule_count} RuleSpec rule(s) with Axiom Rules"

    def _run_rulespec_compile_check(self, rules_file: Path) -> ValidationResult:
        """Compile RuleSpec YAML through the Axiom Rules engine."""
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
                    issues.append(f"Axiom Rules compile failed: {detail}")
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
            issues.append(f"Axiom Rules compile failed: {exc}")
            return ValidationResult(
                validator_name="compile",
                passed=False,
                issues=issues,
                duration_ms=int((time.time() - start) * 1000),
                error=str(exc),
            )

    def _run_compile_check(self, rulespec_file: Path) -> ValidationResult:
        """Tier 0: Compile check against Axiom Rules RuleSpec."""
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
        """Coerce Python/YAML scalar values to Axiom Rules ScalarValueSpec JSON."""
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
    ) -> dict[str, Any]:
        """Build an Axiom Rules dataset from compact RuleSpec test inputs."""
        if case_input in (None, ""):
            case_input = {}
        if not isinstance(case_input, dict):
            raise ValueError("input must be a mapping")

        interval = {"start": period["start"], "end": period["end"]}
        inputs: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        for name, value in case_input.items():
            if isinstance(value, list):
                related_entity = self._related_entity_from_relation(str(name))
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
                            "name": str(name),
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
                        inputs.append(
                            {
                                "name": str(child_name),
                                "entity": related_entity,
                                "entity_id": related_id,
                                "interval": interval,
                                "value": self._rulespec_scalar_value(child_value),
                            }
                        )
                continue

            if isinstance(value, dict):
                raise ValueError(f"input `{name}` must be scalar or relation list")

            inputs.append(
                {
                    "name": str(name),
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
    ) -> dict[str, Any]:
        """Return the live key-0 scalar value for a compiled scalar parameter."""
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
        if not isinstance(value, dict):
            raise ValueError(f"parameter `{parameter.get('name')}` has no key 0 value")
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
        """Compare Axiom Rules scalar value specs, allowing int/decimal equality."""
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
                    )
                )
                issues.extend(execution_issues)
                if response_outputs is not None:
                    actual_outputs.update(response_outputs)

            for output_name in parameter_outputs:
                try:
                    parameter_value = self._rulespec_compiled_parameter_value(
                        parameter_by_key[output_name], period
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
                issues.append(f"Axiom Rules compile failed: {detail}")
            elif isinstance(payload, dict):
                compiled_payload = payload
                raw_output = self._rulespec_compile_success_output(payload)
            else:
                issues.append("Axiom Rules compile did not return an artifact payload.")
        except Exception as exc:
            issues.append(f"Axiom Rules compile failed: {exc}")

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
                    issues.extend(
                        self._run_rulespec_test_cases(
                            rules_file=rules_file,
                            compiled_path=compiled_path,
                            compiled_payload=compiled_payload,
                            cases=payload,
                        )
                    )
        elif not self._is_nonassertable_rulespec_artifact(rules_file):
            issues.append("No tests found.")

        issues.extend(find_ungrounded_numeric_issues(content))
        issues.extend(find_structured_scale_parameter_issues(content))
        issues.extend(find_upstream_placement_issues(content, rules_file=rules_file))
        issues.extend(find_source_verification_issues(content))
        issues.extend(
            find_reiteration_issues(content, policy_repo_path=self.policy_repo_path)
        )

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
                and str(rule.get("kind") or "").lower() == "reiteration"
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
        """Flag committed RuleSpec stubs when their source is already ingested."""
        source_root = self._validation_source_root(rulespec_file)
        try:
            relative = rulespec_file.resolve().relative_to(source_root.resolve())
        except ValueError:
            return []

        if not rulespec_content_has_stub_status(rulespec_file.read_text()):
            return []
        if not has_ingested_source_for_import_target(
            relative.with_suffix("").as_posix(), source_root
        ):
            return []

        return [
            "Promoted RuleSpec stub with ingested source: "
            f"{relative.as_posix()} still declares `status: stub` even though the official source is present locally; "
            "replace the stub with a real encoding before promotion"
        ]

    def _check_imported_stub_dependencies(self, rulespec_file: Path) -> list[str]:
        """Flag imports that still point at stubs even though source is already ingested."""
        source_root = self._validation_source_root(rulespec_file)
        issues: list[str] = []

        for import_path in self._extract_import_paths(rulespec_file.read_text()):
            target = source_root / self._import_to_relative_rulespec_path(import_path)
            if not rulespec_file_has_stub_status(target):
                continue
            if not has_ingested_source_for_import_target(import_path, source_root):
                continue
            issues.append(
                "Imported stub dependency with ingested source: "
                f"{rulespec_file.name} imports `{import_path}` but `{target.relative_to(source_root).as_posix()}` "
                "is still a stub while the official source is already present locally; encode the upstream file instead"
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
        timeout = int(os.getenv("AXIOM_ENCODE_POLICYENGINE_TIMEOUT_SECONDS", "120"))
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
            )

        # Map encoded variables to PE variables.
        # Run comparison for each test
        matches = 0
        total = 0
        unsupported_count = 0
        for test in tests:
            test_rule_name = test.get("variable", "")
            oracle_rule_name = self.policyengine_rule_hint or test_rule_name
            if not self._should_compare_pe_test_output(
                country, test_rule_name, oracle_rule_name
            ):
                continue
            pe_var = self._resolve_pe_variable(country, oracle_rule_name)
            expected = test.get("expect")
            raw_inputs = test.get("inputs", {})
            inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else raw_inputs
            period = (
                test.get("period")
                or test.get("date")
                or (inputs.get("period") if isinstance(inputs, dict) else None)
                or (inputs.get("date") if isinstance(inputs, dict) else None)
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
                country, oracle_rule_name, inputs, expected
            )
            if not mappable:
                issues.append(
                    f"PolicyEngine unavailable for '{test.get('name', test_rule_name)}': {reason}"
                )
                unsupported_count += 1
                continue

            if not pe_var:
                if self.policyengine_rule_hint:
                    issues.append(
                        "PolicyEngine unavailable for "
                        f"'{test.get('name', test_rule_name)}': no PE mapping for "
                        "encoded variable "
                        f"'{test_rule_name}' with oracle hint "
                        f"'{self.policyengine_rule_hint}'"
                    )
                else:
                    issues.append(
                        "PolicyEngine unavailable for "
                        f"'{test.get('name', test_rule_name)}': no PE mapping for "
                        f"encoded variable '{test_rule_name}'"
                    )
                unsupported_count += 1
                continue

            # Build and run PE scenario — include period in inputs for monthly detection
            inputs_with_period = {**inputs, "period": str(period)}
            source_jurisdiction = None
            if country == "us":
                source_jurisdiction = _source_metadata_jurisdiction(source_metadata)
                if source_jurisdiction and "state_code_str" not in inputs_with_period:
                    inputs_with_period["state_code_str"] = source_jurisdiction
                if source_jurisdiction and pe_var in {
                    "snap_standard_utility_allowance",
                    "snap_limited_utility_allowance",
                    "snap_individual_utility_allowance",
                }:
                    inputs_with_period.setdefault(
                        "snap_utility_allowance_type",
                        _default_snap_utility_type_for_rule(oracle_rule_name),
                    )
                    inputs_with_period.setdefault(
                        "snap_utility_region_str",
                        _default_snap_utility_region_for_jurisdiction(
                            source_jurisdiction
                        ),
                    )
            scenario_script = self._build_pe_scenario_script(
                pe_var,
                inputs_with_period,
                year,
                expected,
                country=country,
                rule_name=oracle_rule_name,
            )
            output = self._run_pe_subprocess_detailed(scenario_script, pe_python)

            if output.returncode != 0:
                summary = self._summarize_oracle_error(output.stderr or output.stdout)
                if self._is_pe_unsupported_error(output.stderr or output.stdout):
                    issues.append(
                        f"PolicyEngine unavailable for '{test.get('name', test_rule_name)}': {summary}"
                    )
                    unsupported_count += 1
                    continue
                issues.append(
                    f"PE calculation failed for '{test.get('name', test_rule_name)}': {summary}"
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
                            f"'{test.get('name', test_rule_name)}': "
                            f"PE={pe_value:.2f}, RuleSpec expects={expected_float:.2f}"
                        )
                    total += 1
                else:
                    issues.append(
                        f"No RESULT in PE output for '{test.get('name', test_rule_name)}'"
                    )
                    total += 1
            except Exception as parse_err:
                issues.append(
                    f"Parse error for '{test.get('name', test_rule_name)}': {parse_err}"
                )
                total += 1

        if total == 0:
            duration = int((time.time() - start) * 1000)
            if unsupported_count:
                issues.append(
                    "PolicyEngine could not evaluate any oracle-comparable tests"
                )
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

        Runs PE Microsimulation on the enhanced CPS, extracts the target
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
                            key: normalize_test_value(value)
                            for key, value in inputs.items()
                        }
                        if isinstance(inputs, dict)
                        else inputs or {}
                    )
                    for variable, expected in outputs.items():
                        tests.append(
                            {
                                "variable": normalize_variable_name(variable),
                                "raw_variable": str(variable),
                                "name": test_case.get("name"),
                                "period": test_case.get("period"),
                                "inputs": normalized_inputs,
                                "expect": normalize_test_value(expected),
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
        mapping.update(_PE_US_VARIABLE_MAP)
        return mapping

    @staticmethod
    def _get_pe_us_var_adapter(name: str) -> _PolicyEngineUSVarAdapter | None:
        """Return the PE-US adapter row for a mapped PE var or encoded alias."""
        return _PE_US_VAR_ADAPTERS_BY_NAME.get(name)

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
    } | _PE_US_MONTHLY_VAR_NAMES

    # PE variables at spm_unit level (need spm_units in situation)
    _PE_SPM_VARS = set(_PE_US_SPM_VAR_NAMES)

    def _is_pe_test_mappable(
        self, country: str, rule_name: str, inputs: dict, expected: Any = None
    ) -> tuple[bool, str | None]:
        """Return whether the test case can be represented in PolicyEngine."""
        rule_name_lower = rule_name.lower()
        if country == "us":
            adapter = self._get_pe_us_var_adapter(rule_name)
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
        hinted_pe_var = self._resolve_pe_variable(country, oracle_rule_name)
        if not hinted_pe_var:
            return False
        test_pe_var = self._resolve_pe_variable(country, test_rule_name)
        return test_pe_var is not None and test_pe_var == hinted_pe_var

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
            explicit_child_count
            if explicit_child_count is not None
            else household_children
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

        adult_attrs = [f"'age': {{'{year}': 30}}"]
        members = ["'adult'"]

        # Check for employment income / earned income
        earned = inputs.get(
            "employment_income", inputs.get("earned_income", inputs.get("wages", 0))
        )
        if earned:
            adult_attrs.append(f"'employment_income': {{'{year}': {earned}}}")

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
            household_state = _normalize_state_code_from_utility_region(utility_region)

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
    'tax_units': {{'tu': {{'members': {members_str}}}}},
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
    axiom_rules_path = policy_repo_root.parent / "axiom-rules"
    if not axiom_rules_path.exists():
        axiom_rules_path = Path(__file__).resolve().parents[4] / "axiom-rules"

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
    )

    return pipeline.validate(file_path)
