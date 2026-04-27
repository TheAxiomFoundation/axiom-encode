"""Model comparison evals for statute and source-backed policy encoding."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Literal, Sequence

import requests
import yaml

from axiom_encode.codex_cli import resolve_codex_cli
from axiom_encode.constants import DEFAULT_OPENAI_MODEL
from axiom_encode.repo_routing import find_policy_repo_root
from axiom_encode.statute import (
    CitationParts,
    citation_to_relative_rulespec_path,
    find_citation_text,
    parse_usc_citation,
)

from .dependency_stubs import (
    ResolvedCanonicalConcept,
    ResolvedDefinedTerm,
    import_target_to_relative_rulespec_path,
    materialize_registered_stub,
    resolve_canonical_concepts_from_text,
    resolve_defined_terms_from_text,
)
from .encoding_db import TokenUsage
from .eval_prompt_surface import (
    render_date_silent_scaffold_guidance,
    render_single_amount_row_guidance,
    render_uk_legislation_guidance,
)
from .observability import emit_eval_result, extract_reasoning_output_tokens
from .pricing import estimate_usage_cost_usd
from .validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
)

EvalMode = Literal["cold", "repo-augmented"]
EvalOracleMode = Literal["none", "policyengine", "all"]
IMPORT_ITEM_PATTERN = re.compile(r"^\s*-\s*(['\"]?)([^'\"]+?)\1\s*$")
SUPPORTED_EVAL_ENTITIES = (
    "Payment",
    "Person",
    "TaxUnit",
    "Household",
    "Family",
    "TanfUnit",
    "SnapUnit",
    "SPMUnit",
    "Corporation",
    "Business",
    "Asset",
)
SUPPORTED_EVAL_PERIODS = ("Year", "Month", "Week", "Day")
SUPPORTED_EVAL_DTYPES = (
    "Money",
    "Rate",
    "Boolean",
    "Integer",
    "Count",
    "String",
    "Decimal",
    "Float",
)
_PURE_NUMERIC_EXPRESSION_PATTERN = re.compile(r"^[\d\s()+\-*/.,]+$")
_ISO_WEEK_PERIOD_PATTERN = re.compile(r"^\d{4}-W\d{2}(?:-\d)?$")
_CONDITIONAL_AMOUNT_SLICE_PATTERN = re.compile(
    r"\b(?:if|where|unless|except|subject to|treated as paid)\b",
    re.IGNORECASE,
)
_LOCAL_IMPORT_ROOT_TOKENS = {"legislation", "statute", "regulation"}


@dataclass(frozen=True)
class EvalRunnerSpec:
    """How to invoke a model in an eval."""

    name: str
    backend: str
    model: str


@dataclass
class GroundingMetric:
    """A numeric grounding decision."""

    line: int
    raw: str
    value: float
    grounded: bool


@dataclass
class EvalArtifactMetrics:
    """Deterministic checks over a produced RuleSpec artifact."""

    compile_pass: bool
    compile_issues: list[str]
    ci_pass: bool
    ci_issues: list[str]
    embedded_source_present: bool
    grounded_numeric_count: int
    ungrounded_numeric_count: int
    grounding: list[GroundingMetric]
    source_numeric_occurrence_count: int = 0
    covered_source_numeric_occurrence_count: int = 0
    missing_source_numeric_occurrence_count: int = 0
    numeric_occurrence_issues: list[str] = field(default_factory=list)
    generalist_review_pass: bool | None = None
    generalist_review_score: float | None = None
    generalist_review_issues: list[str] = field(default_factory=list)
    generalist_review_prompt_sha256: str | None = None
    policyengine_pass: bool | None = None
    policyengine_score: float | None = None
    policyengine_issues: list[str] = field(default_factory=list)
    taxsim_pass: bool | None = None
    taxsim_score: float | None = None
    taxsim_issues: list[str] = field(default_factory=list)


@dataclass
class EvalContextFile:
    """A context file copied into the eval workspace."""

    source_path: str
    workspace_path: str
    import_path: str
    kind: str
    label: str | None = None


@dataclass
class EvalWorkspace:
    """Prepared workspace bundle for an eval run."""

    root: Path
    source_file: Path
    manifest_file: Path
    source_metadata_file: Path | None = None
    source_metadata: dict[str, object] | None = None
    context_files: list[EvalContextFile] = field(default_factory=list)


@dataclass
class EvalPromptResponse:
    """Raw model output and trace from a prompt-only eval run."""

    text: str
    duration_ms: int
    tokens: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    trace: dict | None = None
    unexpected_accesses: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalResult:
    """One citation x runner result."""

    citation: str
    runner: str
    backend: str
    model: str
    mode: EvalMode
    output_file: str
    trace_file: str
    context_manifest_file: str
    duration_ms: int
    success: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    reasoning_output_tokens: int
    estimated_cost_usd: float | None
    actual_cost_usd: float | None
    retrieved_files: list[str]
    unexpected_accesses: list[str]
    metrics: EvalArtifactMetrics | None
    generation_prompt_sha256: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.metrics is not None:
            data["metrics"] = asdict(self.metrics)
        return data


@dataclass(frozen=True)
class EvalReadinessGates:
    """Thresholds that determine whether a benchmark suite is bulk-ready."""

    min_cases: int = 1
    min_success_rate: float | None = None
    min_compile_pass_rate: float | None = None
    min_ci_pass_rate: float | None = None
    min_zero_ungrounded_rate: float | None = None
    min_generalist_review_pass_rate: float | None = 1.0
    min_policyengine_pass_rate: float | None = None
    max_mean_estimated_cost_usd: float | None = None


@dataclass
class EvalSuiteCase:
    """One manifest entry in an eval suite."""

    kind: Literal["citation", "source"]
    name: str
    mode: EvalMode
    allow_context: list[Path] = field(default_factory=list)
    citation: str | None = None
    source_id: str | None = None
    source_file: Path | None = None
    metadata_file: Path | None = None
    policyengine_rule_hint: str | None = None
    oracle: EvalOracleMode = "none"
    policyengine_country: str = "auto"


@dataclass
class EvalSuiteManifest:
    """Manifest describing a benchmark suite and its readiness gates."""

    name: str
    path: Path
    runners: list[str]
    mode: EvalMode
    allow_context: list[Path]
    gates: EvalReadinessGates
    cases: list[EvalSuiteCase]


@dataclass(frozen=True)
class EvalReadinessGateResult:
    """Outcome of one readiness threshold."""

    name: str
    comparator: Literal["min", "max"]
    threshold: float | int
    actual: float | int | None
    passed: bool


@dataclass
class EvalReadinessSummary:
    """Aggregated readiness summary for one runner across a suite."""

    total_cases: int
    success_rate: float
    compile_pass_rate: float
    ci_pass_rate: float
    zero_ungrounded_rate: float
    generalist_review_pass_rate: float
    mean_generalist_review_score: float | None
    policyengine_case_count: int
    policyengine_pass_rate: float | None
    mean_policyengine_score: float | None
    mean_estimated_cost_usd: float | None
    gate_results: list[EvalReadinessGateResult]
    ready: bool


def parse_runner_spec(spec: str) -> EvalRunnerSpec:
    """Parse `[name=]backend:model` into a structured runner spec."""
    alias = ""
    target = spec
    if "=" in spec:
        alias, target = spec.split("=", 1)

    if ":" not in target:
        raise ValueError(
            f"Invalid runner spec '{spec}'. Expected [name=]backend:model."
        )

    backend, model = target.split(":", 1)
    backend = backend.strip()
    model = model.strip()
    name = alias.strip() or re.sub(r"[^a-zA-Z0-9._-]+", "-", f"{backend}-{model}")

    if backend not in {"claude", "codex", "openai"}:
        raise ValueError(f"Unsupported backend '{backend}' in runner spec '{spec}'")

    return EvalRunnerSpec(name=name, backend=backend, model=model)


def run_model_eval(
    citations: list[str],
    runner_specs: list[str],
    output_root: Path,
    axiom_rules_path: Path,
    atlas_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one or more citations."""
    xml_root = atlas_path / "data" / "uscode"
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        for citation in citations:
            results.append(
                _run_single_eval(
                    citation=citation,
                    runner=runner,
                    output_root=output_root,
                    axiom_rules_path=axiom_rules_path,
                    xml_root=xml_root,
                    mode=mode,
                    extra_context_paths=extra_context_paths or [],
                )
            )

    return results


def run_source_eval(
    source_id: str,
    source_text: str,
    runner_specs: list[str],
    output_root: Path,
    policy_path: Path,
    source_path: Path | None = None,
    source_metadata_path: Path | None = None,
    source_metadata_payload: dict[str, object] | None = None,
    runtime_axiom_rules_path: Path | None = None,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one arbitrary source-backed text unit."""
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        results.append(
            _run_single_source_eval(
                source_id=source_id,
                source_text=source_text,
                runner=runner,
                output_root=output_root,
                policy_path=policy_path,
                source_path=source_path,
                source_metadata_path=source_metadata_path,
                source_metadata_payload=source_metadata_payload,
                runtime_axiom_rules_path=runtime_axiom_rules_path or policy_path,
                mode=mode,
                extra_context_paths=extra_context_paths or [],
                oracle=oracle,
                policyengine_country=policyengine_country,
                policyengine_rule_hint=policyengine_rule_hint,
            )
        )

    return results


def _sha256_text(text: str | None) -> str | None:
    """Return a stable digest for a prompt or prompt-derived text blob."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_eval_suite_manifest(path: Path) -> EvalSuiteManifest:
    """Load a manifest describing a benchmark suite and readiness gates."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Eval suite manifest must be a mapping: {path}")

    base_dir = Path(path).resolve().parent
    default_mode = _coerce_eval_mode(raw.get("mode", "repo-augmented"))
    default_context = [
        _resolve_manifest_path(base_dir, entry)
        for entry in raw.get("allow_context", []) or []
    ]
    runners = [
        str(item) for item in (raw.get("runners") or [f"codex:{DEFAULT_OPENAI_MODEL}"])
    ]

    gates_raw = raw.get("gates") or {}
    gates = EvalReadinessGates(
        min_cases=int(gates_raw.get("min_cases", 1)),
        min_success_rate=_optional_float(gates_raw.get("min_success_rate")),
        min_compile_pass_rate=_optional_float(gates_raw.get("min_compile_pass_rate")),
        min_ci_pass_rate=_optional_float(gates_raw.get("min_ci_pass_rate")),
        min_zero_ungrounded_rate=_optional_float(
            gates_raw.get("min_zero_ungrounded_rate")
        ),
        min_generalist_review_pass_rate=_optional_float(
            gates_raw.get("min_generalist_review_pass_rate", 1.0)
        ),
        min_policyengine_pass_rate=_optional_float(
            gates_raw.get("min_policyengine_pass_rate")
        ),
        max_mean_estimated_cost_usd=_optional_float(
            gates_raw.get("max_mean_estimated_cost_usd")
        ),
    )

    cases_raw = raw.get("cases") or []
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError(f"Eval suite manifest has no cases: {path}")

    cases: list[EvalSuiteCase] = []
    for index, item in enumerate(cases_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Eval suite case #{index} must be a mapping")
        kind = str(item.get("kind", "")).strip()
        if kind not in {"citation", "source"}:
            raise ValueError(f"Unsupported eval suite case kind '{kind}'")

        case_mode = _coerce_eval_mode(item.get("mode", default_mode))
        name = str(item.get("name", "")).strip() or str(
            item.get("citation") or item.get("source_id") or f"case-{index}"
        )

        case = EvalSuiteCase(
            kind=kind,
            name=name,
            mode=case_mode,
            allow_context=[
                _resolve_manifest_path(base_dir, entry)
                for entry in item.get("allow_context", []) or []
            ],
            citation=item.get("citation"),
            source_id=item.get("source_id"),
            source_file=(
                _resolve_manifest_path(base_dir, item["source_file"])
                if item.get("source_file")
                else None
            ),
            metadata_file=(
                _resolve_manifest_path(base_dir, item["metadata_file"])
                if item.get("metadata_file")
                else None
            ),
            policyengine_rule_hint=(
                str(item.get("policyengine_rule_hint")).strip()
                if item.get("policyengine_rule_hint") is not None
                else None
            ),
            oracle=str(item.get("oracle", "none")),
            policyengine_country=str(item.get("policyengine_country", "auto")),
        )
        _validate_eval_suite_case(case, index)
        cases.append(case)

    return EvalSuiteManifest(
        name=str(raw.get("name") or Path(path).stem),
        path=Path(path).resolve(),
        runners=runners,
        mode=default_mode,
        allow_context=default_context,
        gates=gates,
        cases=cases,
    )


def run_eval_suite(
    manifest: EvalSuiteManifest,
    output_root: Path,
    axiom_rules_path: Path,
    atlas_path: Path | None = None,
    runner_specs: list[str] | None = None,
    suite_retry_attempts: int = 2,
    resume_existing: bool = False,
) -> list[EvalResult]:
    """Run every case in a benchmark suite manifest."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_runners = runner_specs or manifest.runners
    parsed_runners = [parse_runner_spec(spec) for spec in resolved_runners]
    results: list[EvalResult] = []
    started_at = _utc_now_iso()
    completed_case_indexes: set[int] = set()
    completed_cases = 0
    last_case_name: str | None = None
    active_case_index: int | None = None
    active_case_name: str | None = None
    active_case_started_at: str | None = None
    active_case_output_root: Path | None = None
    if resume_existing:
        (
            started_at,
            results,
            completed_case_indexes,
        ) = _load_eval_suite_resume_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            runner_count=len(parsed_runners),
        )
        completed_cases = _contiguous_completed_case_count(
            completed_case_indexes, len(manifest.cases)
        )
        if completed_cases > 0:
            last_case_name = manifest.cases[completed_cases - 1].name
    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="running",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    try:
        for index, case in enumerate(manifest.cases, start=1):
            if index in completed_case_indexes:
                continue
            case_output_root = output_root / f"{index:02d}-{_slugify(case.name)}"
            extra_context = [*manifest.allow_context, *case.allow_context]
            attempts = max(suite_retry_attempts, 0) + 1
            active_case_index = index
            active_case_name = case.name
            active_case_started_at = _utc_now_iso()
            active_case_output_root = case_output_root
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
                active_case_index=active_case_index,
                active_case_name=active_case_name,
                active_case_started_at=active_case_started_at,
                active_case_output_root=active_case_output_root,
            )
            for attempt_index in range(attempts):
                try:
                    if case.kind == "citation":
                        if atlas_path is None:
                            raise ValueError(
                                "atlas_path is required for citation eval suite cases"
                            )
                        case_results = run_model_eval(
                            citations=[case.citation or ""],
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            axiom_rules_path=axiom_rules_path,
                            atlas_path=atlas_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                        )
                    elif case.kind == "source":
                        policy_repo_root = (
                            find_policy_repo_root(case.source_file)
                            if case.source_file is not None
                            else None
                        ) or axiom_rules_path
                        case_results = run_source_eval(
                            source_id=case.source_id or case.name,
                            source_text=load_source_text_for_eval(
                                case.source_file or Path()
                            ),
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            policy_path=policy_repo_root,
                            source_path=case.source_file,
                            runtime_axiom_rules_path=axiom_rules_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                            oracle=case.oracle,
                            policyengine_country=case.policyengine_country,
                            policyengine_rule_hint=case.policyengine_rule_hint,
                        )
                    else:
                        raise ValueError(
                            f"Unsupported eval suite case kind '{case.kind}'"
                        )
                except Exception as exc:
                    case_results = _suite_case_failure_results(
                        case, parsed_runners, exc
                    )

                if (
                    attempt_index >= attempts - 1
                    or not _suite_case_results_should_retry(case_results)
                ):
                    break

            for result in case_results:
                if case.name and case.name != result.citation:
                    result.citation = f"{case.name} ({result.citation})"
            results.extend(case_results)
            completed_case_indexes.add(index)
            completed_cases = index
            last_case_name = case.name
            active_case_index = None
            active_case_name = None
            active_case_started_at = None
            active_case_output_root = None
            _append_eval_suite_case_results(output_root, index, case, case_results)
            if _suite_case_results_hit_usage_limit(case_results):
                _write_eval_suite_run_state(
                    output_root=output_root,
                    manifest=manifest,
                    resolved_runners=resolved_runners,
                    status="failed",
                    started_at=started_at,
                    completed_cases=completed_cases,
                    result_count=len(results),
                    last_case_name=last_case_name,
                    error=(
                        "Usage limit reached while running "
                        f"case '{case.name}'. Stop the suite and retry after quota resets."
                    ),
                )
                return results
            _write_eval_suite_run_state(
                output_root=output_root,
                manifest=manifest,
                resolved_runners=resolved_runners,
                status="running",
                started_at=started_at,
                completed_cases=completed_cases,
                result_count=len(results),
                last_case_name=last_case_name,
            )
    except BaseException as exc:
        _write_eval_suite_run_state(
            output_root=output_root,
            manifest=manifest,
            resolved_runners=resolved_runners,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            started_at=started_at,
            completed_cases=completed_cases,
            result_count=len(results),
            last_case_name=last_case_name,
            error=_format_suite_exception(exc),
            active_case_index=active_case_index,
            active_case_name=active_case_name,
            active_case_started_at=active_case_started_at,
            active_case_output_root=active_case_output_root,
        )
        raise

    _write_eval_suite_run_state(
        output_root=output_root,
        manifest=manifest,
        resolved_runners=resolved_runners,
        status="completed",
        started_at=started_at,
        completed_cases=completed_cases,
        result_count=len(results),
        last_case_name=last_case_name,
    )
    return results


def _contiguous_completed_case_count(
    completed_case_indexes: set[int],
    total_cases: int,
) -> int:
    """Return the largest completed case prefix represented in the ledger."""
    completed = 0
    for index in range(1, total_cases + 1):
        if index not in completed_case_indexes:
            break
        completed = index
    return completed


def _utc_now_iso() -> str:
    """Return the current UTC time in a stable JSON-friendly format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_suite_exception(exc: BaseException) -> str:
    """Return a non-empty error string for suite state logs."""
    message = str(exc).strip()
    return message or exc.__class__.__name__


def _load_eval_suite_resume_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    runner_count: int,
) -> tuple[str, list[EvalResult], set[int]]:
    """Load prior suite state and completed case results for resumption."""
    state_path = output_root / "suite-run.json"
    ledger_path = output_root / "suite-results.jsonl"
    started_at = _utc_now_iso()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        manifest_payload = state.get("manifest") or {}
        existing_path = manifest_payload.get("path")
        if existing_path and existing_path != str(manifest.path):
            raise ValueError(
                "Cannot resume eval suite with a different manifest path: "
                f"{existing_path}"
            )
        existing_runners = manifest_payload.get("effective_runners")
        if existing_runners and list(existing_runners) != list(resolved_runners):
            raise ValueError(
                "Cannot resume eval suite with different effective runners: "
                f"{existing_runners}"
            )
        started_at = state.get("started_at") or started_at

    if not ledger_path.exists():
        return started_at, [], set()

    rows_by_case: dict[int, list[dict]] = defaultdict(list)
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        case_index = int(payload.get("case_index", 0) or 0)
        if case_index <= 0:
            continue
        rows_by_case[case_index].append(payload)

    completed_case_indexes: set[int] = set()
    results: list[EvalResult] = []
    for case_index in sorted(rows_by_case):
        rows = rows_by_case[case_index]
        if len(rows) < runner_count:
            continue
        completed_case_indexes.add(case_index)
        for payload in rows[:runner_count]:
            results.append(_eval_result_from_payload(payload.get("result") or {}))

    return started_at, results, completed_case_indexes


def _write_eval_suite_run_state(
    output_root: Path,
    manifest: EvalSuiteManifest,
    resolved_runners: list[str],
    status: str,
    started_at: str,
    completed_cases: int,
    result_count: int,
    last_case_name: str | None = None,
    error: str | None = None,
    active_case_index: int | None = None,
    active_case_name: str | None = None,
    active_case_started_at: str | None = None,
    active_case_output_root: Path | None = None,
) -> None:
    """Persist suite lifecycle state so interrupted runs remain inspectable."""
    payload = {
        "manifest": {
            "name": manifest.name,
            "path": str(manifest.path),
            "runners": manifest.runners,
            "effective_runners": resolved_runners,
        },
        "status": status,
        "started_at": started_at,
        "updated_at": _utc_now_iso(),
        "total_cases": len(manifest.cases),
        "completed_cases": completed_cases,
        "result_count": result_count,
    }
    if last_case_name:
        payload["last_case_name"] = last_case_name
    if error:
        payload["error"] = error
    if active_case_index is not None and active_case_name:
        payload["active_case"] = {
            "index": active_case_index,
            "name": active_case_name,
            "started_at": active_case_started_at,
            "output_root": str(active_case_output_root)
            if active_case_output_root is not None
            else None,
        }
    if status != "running":
        payload["finished_at"] = payload["updated_at"]
    (output_root / "suite-run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _append_eval_suite_case_results(
    output_root: Path,
    case_index: int,
    case: EvalSuiteCase,
    case_results: list[EvalResult],
) -> None:
    """Append finalized case results to a durable JSONL ledger."""
    ledger_path = output_root / "suite-results.jsonl"
    with ledger_path.open("a", encoding="utf-8") as handle:
        for result in case_results:
            payload = {
                "case_index": case_index,
                "case_name": case.name,
                "case_kind": case.kind,
                "result": result.to_dict(),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _eval_result_from_payload(payload: dict) -> EvalResult:
    """Rehydrate an EvalResult from a persisted JSON payload."""
    metrics_payload = payload.get("metrics")
    metrics = None
    if isinstance(metrics_payload, dict):
        grounding = [
            GroundingMetric(
                line=int(item.get("line", 0) or 0),
                raw=str(item.get("raw", "")),
                value=float(item.get("value", 0.0) or 0.0),
                grounded=bool(item.get("grounded", False)),
            )
            for item in metrics_payload.get("grounding") or []
        ]
        metrics = EvalArtifactMetrics(
            compile_pass=bool(metrics_payload.get("compile_pass", False)),
            compile_issues=list(metrics_payload.get("compile_issues") or []),
            ci_pass=bool(metrics_payload.get("ci_pass", False)),
            ci_issues=list(metrics_payload.get("ci_issues") or []),
            embedded_source_present=bool(
                metrics_payload.get("embedded_source_present", False)
            ),
            grounded_numeric_count=int(
                metrics_payload.get("grounded_numeric_count", 0) or 0
            ),
            ungrounded_numeric_count=int(
                metrics_payload.get("ungrounded_numeric_count", 0) or 0
            ),
            grounding=grounding,
            source_numeric_occurrence_count=int(
                metrics_payload.get("source_numeric_occurrence_count", 0) or 0
            ),
            covered_source_numeric_occurrence_count=int(
                metrics_payload.get("covered_source_numeric_occurrence_count", 0) or 0
            ),
            missing_source_numeric_occurrence_count=int(
                metrics_payload.get("missing_source_numeric_occurrence_count", 0) or 0
            ),
            numeric_occurrence_issues=list(
                metrics_payload.get("numeric_occurrence_issues") or []
            ),
            generalist_review_pass=metrics_payload.get("generalist_review_pass"),
            generalist_review_score=metrics_payload.get("generalist_review_score"),
            generalist_review_issues=list(
                metrics_payload.get("generalist_review_issues") or []
            ),
            generalist_review_prompt_sha256=metrics_payload.get(
                "generalist_review_prompt_sha256"
            ),
            policyengine_pass=metrics_payload.get("policyengine_pass"),
            policyengine_score=metrics_payload.get("policyengine_score"),
            policyengine_issues=list(metrics_payload.get("policyengine_issues") or []),
            taxsim_pass=metrics_payload.get("taxsim_pass"),
            taxsim_score=metrics_payload.get("taxsim_score"),
            taxsim_issues=list(metrics_payload.get("taxsim_issues") or []),
        )

    return EvalResult(
        citation=str(payload.get("citation", "")),
        runner=str(payload.get("runner", "")),
        backend=str(payload.get("backend", "")),
        model=str(payload.get("model", "")),
        mode=payload.get("mode", "cold"),
        output_file=str(payload.get("output_file", "")),
        trace_file=str(payload.get("trace_file", "")),
        context_manifest_file=str(payload.get("context_manifest_file", "")),
        duration_ms=int(payload.get("duration_ms", 0) or 0),
        success=bool(payload.get("success", False)),
        error=payload.get("error"),
        generation_prompt_sha256=payload.get("generation_prompt_sha256"),
        input_tokens=int(payload.get("input_tokens", 0) or 0),
        output_tokens=int(payload.get("output_tokens", 0) or 0),
        cache_read_tokens=int(payload.get("cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(payload.get("cache_creation_tokens", 0) or 0),
        reasoning_output_tokens=int(payload.get("reasoning_output_tokens", 0) or 0),
        estimated_cost_usd=payload.get("estimated_cost_usd"),
        actual_cost_usd=payload.get("actual_cost_usd"),
        retrieved_files=list(payload.get("retrieved_files") or []),
        unexpected_accesses=list(payload.get("unexpected_accesses") or []),
        metrics=metrics,
    )


def _suite_case_failure_results(
    case: EvalSuiteCase,
    runners: list[EvalRunnerSpec],
    exc: Exception,
) -> list[EvalResult]:
    """Convert an exception into explicit failed results for each runner."""
    return [
        EvalResult(
            citation=case.name,
            runner=runner.name,
            backend=runner.backend,
            model=runner.model,
            mode=case.mode,
            output_file="",
            trace_file="",
            context_manifest_file="",
            duration_ms=0,
            success=False,
            error=str(exc),
            generation_prompt_sha256=None,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            reasoning_output_tokens=0,
            estimated_cost_usd=None,
            actual_cost_usd=None,
            retrieved_files=[],
            unexpected_accesses=[],
            metrics=None,
        )
        for runner in runners
    ]


def _suite_case_results_should_retry(case_results: list[EvalResult]) -> bool:
    """Return True when a suite case likely failed for a transient reason."""
    if _suite_case_results_hit_usage_limit(case_results):
        return False
    return any(
        result.error is not None
        or result.metrics is None
        or _eval_result_indicates_retryable_timeout(result)
        for result in case_results
    )


def _suite_case_results_hit_usage_limit(case_results: list[EvalResult]) -> bool:
    """Return True when a case result indicates hard quota exhaustion."""
    return any(_eval_result_indicates_usage_limit(result) for result in case_results)


def _eval_result_indicates_usage_limit(result: EvalResult) -> bool:
    """Return True when one result contains a non-retryable usage-limit error."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    return any("usage limit" in text.lower() for text in texts)


def _eval_result_indicates_retryable_timeout(result: EvalResult) -> bool:
    """Return True when one result failed due to a transient timeout."""
    texts: list[str] = []
    if result.error:
        texts.append(result.error)
    if result.metrics is not None:
        texts.extend(result.metrics.compile_issues)
        texts.extend(result.metrics.ci_issues)
        texts.extend(result.metrics.generalist_review_issues)
        texts.extend(result.metrics.policyengine_issues)
        texts.extend(result.metrics.taxsim_issues)

    lowered = [text.lower() for text in texts]
    return any("timeout after" in text or "timed out" in text for text in lowered)


def summarize_readiness(
    results: list[EvalResult],
    gates: EvalReadinessGates,
) -> EvalReadinessSummary:
    """Summarize suite readiness for one runner."""
    total_cases = len(results)
    success_rate = _fraction(
        sum(1 for result in results if result.success), total_cases
    )
    compile_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.compile_pass
        ),
        total_cases,
    )
    ci_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.ci_pass
        ),
        total_cases,
    )
    zero_ungrounded_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None
            and result.metrics.ungrounded_numeric_count == 0
        ),
        total_cases,
    )
    generalist_review_pass_rate = _fraction(
        sum(
            1
            for result in results
            if result.metrics is not None and result.metrics.generalist_review_pass
        ),
        total_cases,
    )
    generalist_scores = [
        result.metrics.generalist_review_score
        for result in results
        if result.metrics is not None
        and result.metrics.generalist_review_score is not None
    ]
    mean_generalist_review_score = (
        round(mean(generalist_scores), 6) if generalist_scores else None
    )

    policyengine_results = [
        result
        for result in results
        if result.metrics is not None and result.metrics.policyengine_score is not None
    ]
    policyengine_case_count = len(policyengine_results)
    policyengine_pass_rate = (
        _fraction(
            sum(
                1
                for result in policyengine_results
                if result.metrics is not None and result.metrics.policyengine_pass
            ),
            policyengine_case_count,
        )
        if policyengine_case_count
        else None
    )
    mean_policyengine_score = (
        round(
            mean(
                result.metrics.policyengine_score
                for result in policyengine_results
                if result.metrics is not None
                and result.metrics.policyengine_score is not None
            ),
            6,
        )
        if policyengine_case_count
        else None
    )

    costs = [
        result.estimated_cost_usd
        for result in results
        if result.estimated_cost_usd is not None
    ]
    mean_estimated_cost_usd = round(mean(costs), 6) if costs else None

    gate_results: list[EvalReadinessGateResult] = [
        _min_gate("min_cases", total_cases, gates.min_cases),
    ]
    if gates.min_success_rate is not None:
        gate_results.append(
            _min_gate("min_success_rate", success_rate, gates.min_success_rate)
        )
    if gates.min_compile_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_compile_pass_rate",
                compile_pass_rate,
                gates.min_compile_pass_rate,
            )
        )
    if gates.min_ci_pass_rate is not None:
        gate_results.append(
            _min_gate("min_ci_pass_rate", ci_pass_rate, gates.min_ci_pass_rate)
        )
    if gates.min_zero_ungrounded_rate is not None:
        gate_results.append(
            _min_gate(
                "min_zero_ungrounded_rate",
                zero_ungrounded_rate,
                gates.min_zero_ungrounded_rate,
            )
        )
    if gates.min_generalist_review_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_generalist_review_pass_rate",
                generalist_review_pass_rate,
                gates.min_generalist_review_pass_rate,
            )
        )
    if gates.min_policyengine_pass_rate is not None:
        gate_results.append(
            _min_gate(
                "min_policyengine_pass_rate",
                policyengine_pass_rate,
                gates.min_policyengine_pass_rate,
            )
        )
    if gates.max_mean_estimated_cost_usd is not None:
        gate_results.append(
            _max_gate(
                "max_mean_estimated_cost_usd",
                mean_estimated_cost_usd,
                gates.max_mean_estimated_cost_usd,
            )
        )

    return EvalReadinessSummary(
        total_cases=total_cases,
        success_rate=success_rate,
        compile_pass_rate=compile_pass_rate,
        ci_pass_rate=ci_pass_rate,
        zero_ungrounded_rate=zero_ungrounded_rate,
        generalist_review_pass_rate=generalist_review_pass_rate,
        mean_generalist_review_score=mean_generalist_review_score,
        policyengine_case_count=policyengine_case_count,
        policyengine_pass_rate=policyengine_pass_rate,
        mean_policyengine_score=mean_policyengine_score,
        mean_estimated_cost_usd=mean_estimated_cost_usd,
        gate_results=gate_results,
        ready=all(result.passed for result in gate_results),
    )


def _coerce_eval_mode(value: str) -> EvalMode:
    """Validate a manifest eval mode."""
    normalized = str(value).strip()
    if normalized not in {"cold", "repo-augmented"}:
        raise ValueError(f"Unsupported eval mode '{value}'")
    return normalized  # type: ignore[return-value]


def _optional_float(value: object) -> float | None:
    """Convert optional numeric manifest values to float."""
    if value is None:
        return None
    return float(value)


def _resolve_manifest_path(base_dir: Path, value: object) -> Path:
    """Resolve a manifest path entry relative to the manifest file."""
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _validate_eval_suite_case(case: EvalSuiteCase, index: int) -> None:
    """Validate one suite case after parsing."""
    if case.kind == "citation" and not case.citation:
        raise ValueError(f"Eval suite case #{index} is missing 'citation'")
    if case.kind == "source":
        if not case.source_id:
            raise ValueError(f"Eval suite case #{index} is missing 'source_id'")
        if case.source_file is None:
            raise ValueError(f"Eval suite case #{index} is missing 'source_file'")


def _fraction(numerator: int, denominator: int) -> float:
    """Return a rounded fraction or 0 when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _min_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate a lower-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="min",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual >= threshold,
    )


def _max_gate(
    name: str,
    actual: float | int | None,
    threshold: float | int,
) -> EvalReadinessGateResult:
    """Evaluate an upper-bound readiness gate."""
    return EvalReadinessGateResult(
        name=name,
        comparator="max",
        threshold=threshold,
        actual=actual,
        passed=actual is not None and actual <= threshold,
    )


def select_context_files(
    citation: str | CitationParts,
    statute_root: Path,
    max_files: int = 6,
) -> list[Path]:
    """Select canonical implementation precedent files for repo-augmented evals."""
    parts = (
        citation
        if isinstance(citation, CitationParts)
        else parse_usc_citation(citation)
    )
    section_root = Path(statute_root) / parts.title / parts.section
    target_rel = citation_to_relative_rulespec_path(parts)
    target_path = Path(statute_root) / target_rel

    candidates: list[Path] = []
    if section_root.exists():
        candidates.extend(
            sorted(
                path
                for path in section_root.rglob("*.yaml")
                if path.resolve() != target_path.resolve()
                and not path.name.endswith(".test.yaml")
            )
        )

    if not candidates:
        title_root = Path(statute_root) / parts.title
        candidates.extend(
            sorted(
                path
                for path in title_root.rglob("*.yaml")
                if path != target_path and not path.name.endswith(".test.yaml")
            )
        )

    # Bias toward nearby files first, then shallower paths for readability.
    candidates.sort(
        key=lambda path: (
            0 if path.parent == section_root else 1,
            len(path.relative_to(statute_root).parts),
            str(path),
        )
    )

    selected: list[Path] = []
    for candidate in candidates:
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= max_files:
            break
    return selected


def prepare_eval_workspace(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    source_text: str,
    axiom_rules_path: Path,
    mode: EvalMode,
    source_path: Path | None = None,
    source_metadata_path: Path | None = None,
    source_metadata_payload: dict[str, object] | None = None,
    extra_context_paths: list[Path] | None = None,
) -> EvalWorkspace:
    """Create an isolated workspace bundle for a single eval."""
    slug = _slugify(citation)
    workspace_root = (
        Path(output_root) / "_eval_workspaces" / runner.name / slug / "workspace"
    )
    if workspace_root.parent.exists():
        shutil.rmtree(workspace_root.parent)
    workspace_root.mkdir(parents=True, exist_ok=True)

    source_file = workspace_root / "source.txt"
    source_file.write_text(source_text.strip() + "\n")
    source_metadata: dict[str, object] | None = None
    if source_metadata_payload is not None:
        source_metadata = dict(source_metadata_payload)
    elif source_metadata_path is not None:
        payload = yaml.safe_load(source_metadata_path.read_text())
        if payload is not None and not isinstance(payload, dict):
            raise ValueError(
                "Explicit source metadata sidecar must decode to a mapping: "
                f"{source_metadata_path}"
            )
        source_metadata = payload
    else:
        source_metadata_path, source_metadata = _load_source_metadata_for_path(
            source_path
        )
    source_metadata_file: Path | None = None
    if source_metadata is not None:
        source_metadata_file = workspace_root / "source-metadata.json"
        source_metadata_file.write_text(
            json.dumps(source_metadata, indent=2, sort_keys=True) + "\n"
        )

    context_files: list[EvalContextFile] = []
    context_root = workspace_root / "context"
    target_rel = _target_rel_for_eval_identifier(citation)
    current_file = axiom_rules_path / target_rel if target_rel is not None else None
    for resolved_term in resolve_defined_terms_from_text(source_text):
        context_files.append(
            _materialize_resolved_definition_stub(
                context_root=context_root,
                resolved_term=resolved_term,
                workspace_root=workspace_root,
            )
        )
    for resolved_concept in resolve_canonical_concepts_from_text(
        source_text,
        axiom_rules_path,
        current_file=current_file,
    ):
        context_files.append(
            _materialize_resolved_canonical_concept(
                context_root=context_root,
                resolved_concept=resolved_concept,
                workspace_root=workspace_root,
            )
        )

    context_corpus_root = _repo_augmented_context_root(axiom_rules_path)

    if mode == "repo-augmented":
        selected = _auto_select_context_files(citation, context_corpus_root)
        for extra_path in extra_context_paths or []:
            path = Path(extra_path)
            if path.exists():
                selected.append(path)

        expanded_context = _expand_context_files(
            selected, context_corpus_root, target_rel
        )

        for source_path, kind in expanded_context:
            relative_target = _context_import_relative_target(
                source_path, axiom_rules_path
            )

            workspace_relative_path = Path("context") / relative_target
            workspace_path = workspace_root / workspace_relative_path
            workspace_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, workspace_path)
            context_files.append(
                EvalContextFile(
                    source_path=str(source_path),
                    workspace_path=str(workspace_relative_path),
                    import_path=_relative_rulespec_path_to_import_target(
                        relative_target
                    ),
                    kind=kind,
                )
            )

    manifest_file = workspace_root / "context-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "citation": citation,
                "mode": mode,
                "source_file": str(source_file.relative_to(workspace_root)),
                "source_metadata_file": (
                    str(source_metadata_file.relative_to(workspace_root))
                    if source_metadata_file is not None
                    else None
                ),
                "source_metadata": source_metadata,
                "context_files": [asdict(item) for item in context_files],
            },
            indent=2,
            sort_keys=True,
        )
    )

    return EvalWorkspace(
        root=workspace_root,
        source_file=source_file,
        manifest_file=manifest_file,
        source_metadata_file=source_metadata_file,
        source_metadata=source_metadata,
        context_files=context_files,
    )


def _load_source_metadata_for_path(
    source_path: Path | None,
) -> tuple[Path | None, dict[str, object] | None]:
    """Return companion source metadata for a source file when present."""
    if source_path is None:
        return None, None

    candidates = [
        source_path.with_name(f"{source_path.stem}.meta.yaml"),
        source_path.with_name(f"{source_path.stem}.meta.yml"),
        source_path.with_suffix(source_path.suffix + ".meta.yaml"),
        source_path.with_suffix(source_path.suffix + ".meta.yml"),
    ]
    metadata_path = next((path for path in candidates if path.exists()), None)
    if metadata_path is None:
        return None, None

    payload = yaml.safe_load(metadata_path.read_text())
    if payload is None:
        return metadata_path, None
    if not isinstance(payload, dict):
        raise ValueError(
            f"Source metadata sidecar must decode to a mapping: {metadata_path}"
        )
    return metadata_path, payload


def load_source_text_for_eval(source_path: Path) -> str:
    """Load authoritative eval text directly from a source file."""
    return Path(source_path).read_text()


def _materialize_resolved_definition_stub(
    *,
    context_root: Path,
    resolved_term: ResolvedDefinedTerm,
    workspace_root: Path,
) -> EvalContextFile:
    """Write one resolved definition stub into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rulespec_path(
        resolved_term.import_target
    )
    materialize_registered_stub(
        workspace_root,
        [resolved_term],
        prefix=Path("context"),
    )
    return EvalContextFile(
        source_path=resolved_term.citation,
        workspace_path=str(relative_target),
        import_path=_relative_rulespec_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="definition_stub",
        label=resolved_term.label,
    )


def _materialize_resolved_canonical_concept(
    *,
    context_root: Path,
    resolved_concept: ResolvedCanonicalConcept,
    workspace_root: Path,
) -> EvalContextFile:
    """Copy one resolved canonical concept file into the eval workspace context."""
    relative_target = Path("context") / import_target_to_relative_rulespec_path(
        resolved_concept.import_target
    )
    target = workspace_root / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved_concept.source_file, target)
    return EvalContextFile(
        source_path=str(resolved_concept.source_file),
        workspace_path=str(relative_target),
        import_path=_relative_rulespec_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="canonical_concept",
        label=resolved_concept.label,
    )


def _auto_select_context_files(citation: str, statute_root: Path) -> list[Path]:
    """Best-effort auto-context selection for statute citations only."""
    try:
        return select_context_files(citation, statute_root)
    except Exception:
        return []


def _repo_augmented_context_root(policy_path: Path) -> Path:
    """Resolve the corpus root used for automatic repo-augmented context selection."""
    resolved = Path(policy_path).resolve()
    if resolved.name == "axiom-rules":
        fallback = resolved.parent / "rules-us" / "statute"
        if fallback.exists():
            return fallback
        return resolved

    statute_root = resolved / "statute"
    if statute_root.exists():
        return statute_root
    return resolved


def _context_import_relative_target(source_path: Path, axiom_rules_path: Path) -> Path:
    """Prefer canonical repo-relative import targets for copied precedent files."""
    repo_parent = axiom_rules_path.parent.resolve()
    resolved_source = source_path.resolve()

    for candidate in sorted(repo_parent.glob("rules-*")):
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_source.relative_to(resolved_candidate)
            if relative.parts and relative.parts[0] == "statute":
                return Path(*relative.parts[1:])
            return relative

    return Path("external") / resolved_source.name


def _relative_rulespec_path_to_import_target(path: Path) -> str:
    """Convert a relative RuleSpec file path into a bare import target."""
    normalized = path.with_suffix("") if path.suffix in {".yaml", ".yml"} else path
    return normalized.as_posix()


def _target_rel_for_eval_identifier(citation: str) -> Path | None:
    """Return the canonical RuleSpec target path for USC citations when parseable."""
    try:
        return citation_to_relative_rulespec_path(citation)
    except Exception:
        return None


def evaluate_artifact(
    rulespec_file: Path,
    policy_repo_root: Path,
    axiom_rules_path: Path,
    source_text: str,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
) -> EvalArtifactMetrics:
    """Evaluate one RuleSpec artifact with deterministic checks plus optional oracles."""
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=oracle != "none",
        policyengine_country=policyengine_country,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    compile_result = pipeline._run_compile_check(rulespec_file)
    ci_result = pipeline._run_ci(rulespec_file)

    policyengine_result = None
    taxsim_result = None
    if oracle in ("policyengine", "all"):
        try:
            policyengine_result = pipeline._run_policyengine(rulespec_file)
        except Exception as exc:
            policyengine_result = ValidationResult(
                validator_name="policyengine",
                passed=False,
                error=str(exc),
                issues=[str(exc)],
            )
    if oracle == "all":
        try:
            taxsim_result = pipeline._run_taxsim(rulespec_file)
        except Exception as exc:
            taxsim_result = ValidationResult(
                validator_name="taxsim",
                passed=False,
                error=str(exc),
                issues=[str(exc)],
            )

    oracle_context: dict[str, dict[str, object]] = {}
    if policyengine_result is not None:
        oracle_context["policyengine"] = {
            "score": policyengine_result.score,
            "passed": policyengine_result.passed,
            "issues": policyengine_result.issues,
            "duration_ms": policyengine_result.duration_ms,
        }
    if taxsim_result is not None:
        oracle_context["taxsim"] = {
            "score": taxsim_result.score,
            "passed": taxsim_result.passed,
            "issues": taxsim_result.issues,
            "duration_ms": taxsim_result.duration_ms,
        }
    review_context = (
        "This review is running inside an eval-suite benchmark workspace. "
        "The artifact file path is generic benchmark output and is not itself the legal citation. "
        "Benchmark directory labels may be stale, generic, or misleading and must be ignored as legal cues. "
        "The benchmark target is an atomic source slice/unit, so judge fidelity to exactly this source text rather than demanding omitted sibling limbs or parent consequences unless the RuleSpec claims to encode them. "
        "Judge citation fidelity against the embedded source-text docstring and this authoritative source excerpt:\n\n"
        f"{source_text.strip()[:4000]}"
    )
    if re.search(
        r"\bon the first day\b|\bnext benefit week\b|\bon or after the day\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        review_context += (
            "\n\nThis is a temporal timing clause. The RuleSpec eval path does not expose a native date-valued output here, "
            "so a boolean day-predicate helper on `period: Day`, plus explicit trigger preconditions from the source text, "
            "is an acceptable representation."
        )
    try:
        generalist_review_result = pipeline._run_reviewer(
            "generalist-reviewer",
            rulespec_file,
            oracle_context or None,
            review_context=review_context,
        )
    except Exception as exc:
        generalist_review_result = ValidationResult(
            validator_name="generalist-reviewer",
            passed=False,
            error=str(exc),
            issues=[f"Reviewer error: {exc}"],
        )

    content = rulespec_file.read_text()
    embedded_source = extract_embedded_source_text(content)
    source_numbers = extract_numbers_from_text(embedded_source or source_text)
    source_numeric_occurrences = Counter(
        extract_numeric_occurrences_from_text(embedded_source or source_text)
    )
    named_scalar_occurrences = Counter(
        item.value for item in extract_named_scalar_occurrences(content)
    )

    grounding_metrics: list[GroundingMetric] = []
    for line, raw, value in extract_grounding_values(content):
        grounding_metrics.append(
            GroundingMetric(
                line=line,
                raw=raw,
                value=value,
                grounded=value in source_numbers,
            )
        )

    numeric_occurrence_issues: list[str] = []
    covered_source_numeric_occurrence_count = 0
    missing_source_numeric_occurrence_count = 0
    for value, expected_count in sorted(source_numeric_occurrences.items()):
        covered_count = min(expected_count, named_scalar_occurrences.get(value, 0))
        covered_source_numeric_occurrence_count += covered_count
        if covered_count < expected_count:
            missing_count = expected_count - covered_count
            missing_source_numeric_occurrence_count += missing_count
            numeric_occurrence_issues.append(
                f"Source numeric value {value:g} appears {expected_count} time(s), "
                f"but only {covered_count} named scalar definition(s) with that value were found."
            )

    ungrounded_numeric_issues = find_ungrounded_numeric_issues(
        content,
        embedded_source or source_text,
    )
    ci_issues = []
    seen_ci_issues: set[str] = set()
    for issue in (
        list(ci_result.issues) + ungrounded_numeric_issues + numeric_occurrence_issues
    ):
        if issue in seen_ci_issues:
            continue
        ci_issues.append(issue)
        seen_ci_issues.add(issue)
    ci_pass = (
        ci_result.passed
        and not ungrounded_numeric_issues
        and not numeric_occurrence_issues
    )

    return EvalArtifactMetrics(
        compile_pass=compile_result.passed,
        compile_issues=compile_result.issues,
        ci_pass=ci_pass,
        ci_issues=ci_issues,
        embedded_source_present=bool(embedded_source),
        grounded_numeric_count=sum(1 for item in grounding_metrics if item.grounded),
        ungrounded_numeric_count=sum(
            1 for item in grounding_metrics if not item.grounded
        ),
        grounding=grounding_metrics,
        source_numeric_occurrence_count=sum(source_numeric_occurrences.values()),
        covered_source_numeric_occurrence_count=covered_source_numeric_occurrence_count,
        missing_source_numeric_occurrence_count=missing_source_numeric_occurrence_count,
        numeric_occurrence_issues=numeric_occurrence_issues,
        generalist_review_pass=generalist_review_result.passed,
        generalist_review_score=generalist_review_result.score,
        generalist_review_issues=generalist_review_result.issues,
        generalist_review_prompt_sha256=generalist_review_result.prompt_sha256,
        policyengine_pass=(
            policyengine_result.passed if policyengine_result is not None else None
        ),
        policyengine_score=(
            policyengine_result.score if policyengine_result is not None else None
        ),
        policyengine_issues=(
            policyengine_result.issues if policyengine_result is not None else []
        ),
        taxsim_pass=taxsim_result.passed if taxsim_result is not None else None,
        taxsim_score=taxsim_result.score if taxsim_result is not None else None,
        taxsim_issues=taxsim_result.issues if taxsim_result is not None else [],
    )


def _run_single_eval(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    axiom_rules_path: Path,
    xml_root: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
) -> EvalResult:
    source_text = find_citation_text(citation, xml_root)
    if not source_text:
        raise ValueError(f"No statute text found for {citation}")

    workspace = prepare_eval_workspace(
        citation=citation,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=axiom_rules_path,
        mode=mode,
        extra_context_paths=extra_context_paths,
    )

    relative_output = citation_to_relative_rulespec_path(citation)
    prompt = _build_eval_prompt(
        citation,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        include_tests=False,
        runner_backend=runner.backend,
    )
    generation_prompt_sha256 = _sha256_text(prompt)
    response = _run_prompt_eval(runner, workspace, prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrote_artifact = _materialize_eval_artifact(
        response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
    )
    if wrote_artifact:
        _hydrate_eval_root(output_file.parents[1], workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(citation)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rulespec_file=output_file,
            policy_repo_root=output_file.parents[1],
            axiom_rules_path=axiom_rules_path,
            source_text=source_text,
        )

    tokens = response.tokens
    result = EvalResult(
        citation=citation,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned"),
        generation_prompt_sha256=generation_prompt_sha256,
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
    )
    emit_eval_result(result, response.trace)
    return result


def _run_single_source_eval(
    source_id: str,
    source_text: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    policy_path: Path,
    source_path: Path | None,
    source_metadata_path: Path | None,
    source_metadata_payload: dict[str, object] | None,
    runtime_axiom_rules_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    oracle: EvalOracleMode,
    policyengine_country: str,
    policyengine_rule_hint: str | None,
) -> EvalResult:
    """Run one eval on an arbitrary source-backed text unit rather than a USC citation."""
    workspace = prepare_eval_workspace(
        citation=source_id,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=policy_path,
        mode=mode,
        source_path=source_path,
        source_metadata_path=source_metadata_path,
        source_metadata_payload=source_metadata_payload,
        extra_context_paths=extra_context_paths,
    )

    relative_output = _source_identifier_to_relative_rulespec_path(source_id)
    prompt = _build_eval_prompt(
        source_id,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        include_tests=True,
        runner_backend=runner.backend,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    generation_prompt_sha256 = _sha256_text(prompt)
    response = _run_prompt_eval(runner, workspace, prompt)
    output_file = Path(output_root) / runner.name / relative_output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    wrote_artifact = _materialize_eval_artifact(
        response.text,
        output_file,
        source_text=source_text,
        workspace_root=workspace.root,
        policyengine_rule_hint=policyengine_rule_hint,
    )
    if wrote_artifact:
        _hydrate_eval_root(output_file.parents[1], workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(source_id)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rulespec_file=output_file,
            policy_repo_root=output_file.parents[1],
            axiom_rules_path=runtime_axiom_rules_path,
            source_text=source_text,
            oracle=oracle,
            policyengine_country=policyengine_country,
            policyengine_rule_hint=policyengine_rule_hint,
        )

    tokens = response.tokens
    result = EvalResult(
        citation=source_id,
        runner=runner.name,
        backend=runner.backend,
        model=runner.model,
        mode=mode,
        output_file=str(output_file),
        trace_file=str(trace_file),
        context_manifest_file=str(workspace.manifest_file),
        duration_ms=response.duration_ms,
        success=wrote_artifact and response.error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned"),
        generation_prompt_sha256=generation_prompt_sha256,
        input_tokens=tokens.input_tokens if tokens else 0,
        output_tokens=tokens.output_tokens if tokens else 0,
        cache_read_tokens=tokens.cache_read_tokens if tokens else 0,
        cache_creation_tokens=tokens.cache_creation_tokens if tokens else 0,
        reasoning_output_tokens=tokens.reasoning_output_tokens if tokens else 0,
        estimated_cost_usd=response.estimated_cost_usd,
        actual_cost_usd=response.actual_cost_usd,
        retrieved_files=[item.source_path for item in workspace.context_files],
        unexpected_accesses=response.unexpected_accesses,
        metrics=metrics,
    )
    emit_eval_result(result, response.trace)
    return result


def _source_identifier_to_relative_rulespec_path(source_id: str) -> Path:
    """Map an arbitrary source identifier to a stable eval artifact path."""
    return Path("source") / f"{_slugify(source_id)}.yaml"


def _rulespec_test_path(path: Path) -> Path:
    """Return the companion RuleSpec test path for a programme file."""
    return path.with_name(f"{path.stem}.test.yaml")


def _build_rulespec_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    include_tests: bool,
    runner_backend: str,
    policyengine_rule_hint: str | None,
) -> str:
    """Build the RuleSpec authoring prompt used by current evals."""
    source_text = workspace.source_file.read_text().strip()
    backend_section = ""
    if runner_backend == "openai":
        backend_section = (
            "You do not have filesystem tool access in this eval; rely on the "
            "inline source and any inline context copies in this prompt.\n"
        )
    scaffold_dates = _collect_scaffold_dates(workspace, context_files)
    scaffold_dates_section = ""
    if scaffold_dates:
        scaffold_dates_section = f"""
Temporal scaffold dates visible in copied context:
{", ".join(f"`{item}`" for item in scaffold_dates)}
Prefer the earliest scaffold date that is relevant to the copied precedent when `./source.txt` lacks its own effective date.
"""

    source_metadata_section = ""
    if workspace.source_metadata is not None:
        source_metadata_section = f"""
Structured source metadata is available in `./source-metadata.json` and copied below.
If a metadata relation says this source `sets` a canonical target, model this artifact as setting the effective jurisdiction-specific value for that delegated slot and record the absolute target path under `metadata.sets`. This is not an `amends` relationship unless the source itself amends another source.
For state option/source-slice metadata, do not add a top-level `imports:` entry to the bare canonical `cfr/...#...` or `usc/...#...` path unless a copied context file actually provides that import target.
If the canonical target is an option/applies/uses-style slot such as `...#*_applies` or `...#*_uses_*`, encode the canonical boolean slot as a direct dated constant `true` or `false` when the source text itself sets that option.
Do not invent jurisdiction guards like `*_is_in_state` or `*_is_in_jurisdiction` unless `./source.txt` states them; for a jurisdiction-specific source slice, use only positive/continuity cases rather than a fabricated out-of-jurisdiction false case.
For a jurisdiction-specific setting slice, omit an inapplicable false test unless `./source.txt` itself states a narrower in-jurisdiction condition.

=== BEGIN SOURCE-METADATA.JSON ===
{json.dumps(workspace.source_metadata, indent=2, sort_keys=True)}
=== END SOURCE-METADATA.JSON ===
"""

    context_section = ""
    if context_files:
        listings = "\n".join(
            _format_context_file_listing(item) for item in context_files
        )
        inline_context = ""
        if runner_backend == "openai":
            inline_context = f"""

You do not have filesystem tool access in this eval, so the relevant context files are also copied inline below.
Inline context copies:
{_format_inline_context_snippets(workspace, context_files)}
"""
        definition_items = [
            item for item in context_files if item.kind == "definition_stub"
        ]
        canonical_items = [
            item for item in context_files if item.kind == "canonical_concept"
        ]
        resolved_guidance = ""
        if definition_items:
            labels = "\n".join(
                f"- {item.label or item.import_path}" for item in definition_items
            )
            resolved_guidance += f"""
Resolved definition files are available below.
{labels}
import that canonical definition instead of inventing a leaf-local helper. Do not replace that import with a local deferred stub.
Do not encode such local factual predicates as placeholder constants like `true` or `false`.
Do not encode such local factual predicates as `status: deferred`.
"""
        if canonical_items:
            labels = "\n".join(
                f"- {item.label or item.import_path}" for item in canonical_items
            )
            resolved_guidance += f"""
Resolved canonical concept files from this corpus are available below.
{labels}
import or re-export that exact canonical concept instead of duplicating it locally.
"""
        context_section = f"""
Context mode: `{mode}`.
Context files are precedent and dependency context, not independent legal authority for new values:
{listings}
{inline_context}
{resolved_guidance}
Import and context rules:
- Use the listed import target rather than the `./context/...` inspection path.
- do not wrap import targets in quotes.
- Every import path must point to a file that is actually copied into the workspace.
- If a copied context file already defines the exact symbol you need, import that exact symbol instead of inventing renamed locals that overlap with the copied file.
- Do not fabricate same-instrument imports or `statute/...#symbol` paths unless that exact `path#symbol` import target is listed.
- do not fabricate sibling-file imports for uncopied same-instrument provisions.
- When a copied chart or parameter file supplies values, keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file; do not guess contradictory expectations for those imported values.
- Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0` unless that row is visible in the copied chart file.
- Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file.
{scaffold_dates_section}
"""

    test_file_name = _rulespec_test_path(Path(target_file_name)).name
    if include_tests:
        oracle_rule = ""
        if policyengine_rule_hint:
            oracle_rule = (
                f"- Every non-empty test `output:` mapping must assert "
                f"`{policyengine_rule_hint}` directly.\n"
            )
        output_rules = f"""
Return exactly this two-file bundle and nothing else:
=== FILE: {target_file_name} ===
<RuleSpec YAML>
=== FILE: {test_file_name} ===
<YAML list of test cases>

Test file rules:
- `{test_file_name}` must be a YAML list beginning with `- name:` entries.
- Use `period`, `input`, and `output` keys. Use concrete scalar values, not formula strings.
- Do not use bare year periods like `2024`; they are ambiguous across jurisdictions.
- For monthly outputs, use `period: YYYY-MM`.
- For annual or non-calendar periods, use an explicit mapping such as `period: {{period_kind: custom, name: calendar_year, start: '2024-01-01', end: '2024-12-31'}}` or `period: {{period_kind: tax_year, start: '2024-04-06', end: '2025-04-05'}}`.
- Emit 1-4 cases unless `module.status` is `deferred` or `entity_not_supported`, in which case the test file may be empty.
- The test file must contain YAML only; do not put prose or markdown fences in it.
- Use factual predicates or quantities in `input:`, not the output variable being asserted.
- Do not add speculative future-period tests that rely on uprating or amendments not stated in `./source.txt`.
{oracle_rule.rstrip()}
"""
    else:
        output_rules = f"""
Return ONLY raw RuleSpec YAML for `{target_file_name}`. Do not include fences or explanation.
"""

    target_hint = ""
    if policyengine_rule_hint:
        target_hint = f"""
Preferred principal output:
- Name the main derived rule `{policyengine_rule_hint}` unless the source clearly defines a different canonical concept.
- Keep oracle-comparable tests at that named semantic level; do not assert only helper parameters or documentary scalars.
- Keep `.test.yaml` inputs oracle-comparable: prefer the oracle's direct component facts over inverted household proxy inputs, preserve direct component surfaces when available, and assert `{policyengine_rule_hint}` directly in every non-empty `output:` mapping.
- Prefer a contemporary monthly `.test.yaml` period like `2022-01` or `2024-01` when the source is current-effective and lacks a better effective date; avoid pre-2015 historical periods that PolicyEngine US cannot evaluate.
- If a copied downstream output named by the oracle hint is available, assert that copied downstream output named by the oracle hint rather than replacing it with a helper-only local test.
"""

    guidance_parts = [render_uk_legislation_guidance()]
    if _is_single_amount_table_slice(source_text):
        guidance_parts.append(render_single_amount_row_guidance())
    if not re.search(r"\b\d{4}-\d{2}-\d{2}\b", source_text) and not scaffold_dates:
        guidance_parts.append(render_date_silent_scaffold_guidance())
    additional_guidance = "\n".join(
        part.strip() for part in guidance_parts if part.strip()
    )

    inline_source = f"""
=== BEGIN SOURCE.TXT ===
{source_text}
=== END SOURCE.TXT ===
"""

    return f"""You are participating in an encoding eval for {citation}.

Author the output in Axiom RuleSpec YAML.

Primary legal authority:
- `./source.txt` contains the complete source text for this target source unit.
- Treat that source text as the only source of legal truth for this artifact.
{inline_source}
{source_metadata_section}{context_section}
{backend_section}

RuleSpec requirements:
- The programme file must begin with `format: rulespec/v1`.
- Include `module.summary: |-` containing the exact operative source text or an exact compact excerpt sufficient to audit all encoded rules.
- Use `rules:` as a list of rule objects. The filepath is the ID; do not add an `id:` field.
- Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`.
- Rule kinds are `parameter`, `derived`, or `relation`. Use `parameter` for named source scalars and `derived` for entity-scoped outputs.
- Do not invent new entities, periods, or dtypes.
- Allowed `entity:` values are {", ".join(f"`{entity}`" for entity in SUPPORTED_EVAL_ENTITIES)}.
- Allowed `period:` values are {", ".join(f"`{period}`" for period in SUPPORTED_EVAL_PERIODS)}.
- Allowed `dtype:` values are {", ".join(f"`{dtype}`" for dtype in SUPPORTED_EVAL_DTYPES)}, or `Enum[Name]`.
- Use `unit: USD`, `unit: GBP`, or another explicit unit for money outputs when the source states a currency.
- Put each rule's formulas under `versions: - effective_from: 'YYYY-MM-DD'` and `formula: |-`.
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==` for equality, `and`/`or` for booleans, decimal ratios for percentages, and no Python inline ternary syntax.
- Do not use Python inline ternaries like `x if cond else y`.
- Use chained `if condition: value else: other_value` expressions; do not use YAML-style `if:` / `then:` / `else:` blocks.
- Do not append a multiline conditional directly onto another expression, and do not use inline assignment syntax like `:=` inside formula blocks.
- For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals.
- Use concrete ISO calendar dates like `2025-03-21` for day-level tests; do not use ISO week strings like `2025-W13`.
- Any substantive numeric literal in a formula must either appear in `./source.txt` or be one of -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec when it is a legal amount, rate, threshold, cap, or limit.
- Represent every substantive source amount, rate, threshold, cap, or limit as a named `parameter` rule, then reference that parameter from derived formulas.
- If the same numeric value appears twice in materially different legal roles, give those roles distinct named scalars; otherwise reuse that named scalar everywhere the rule compares against or computes with that number.
- If `./source.txt` says someone is "aged 18 or over", "under 25", or similar, model the legal age predicate instead of inventing documentary age constants.
- Do not create scalar variables for citation numbers, paragraph numbers, branch numbers, or source line labels.
- Do not invent `dtype: String` variables just to restate the effective date.
- Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables.
- Do not create named `parameter` rules for structural table row labels, household-size row indexes, or branch numbers unless the source actually sets that value as a legal amount, rate, threshold, cap, or limit; use those structural comparisons inline instead.
- If the source cannot be represented faithfully with the supported schema, emit `module.status: deferred` or `module.status: entity_not_supported` with `rules: []`; do not invent unsupported ontology.
- For deferred or entity-not-supported artifacts, leave the companion `.test.yaml` empty and do not create assertions against deferred symbols.
- If metadata or context names an absolute canonical target that this source `sets`, store that absolute path in the relevant rule's `metadata.sets` list.
- When the source says a value is determined `in accordance with section X`, emit the upstream import instead of restating the concept locally when that import target is available.
- Do not fabricate sibling-file imports, do not guess unavailable import targets, and do not invent `import` statements or `imports:` blocks for uncopied same-instrument provisions.
{target_hint}
Additional encoding guidance:
{additional_guidance}

Minimal RuleSpec shape:
```yaml
format: rulespec/v1
module:
  summary: |-
    <exact source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
  - name: example_output
    kind: derived
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          if example_condition: example_amount else: 0
```

{output_rules}
Do not respond with summaries, markdown prose, or file-write confirmations.
"""


def _build_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    include_tests: bool = False,
    runner_backend: str = "codex",
    policyengine_rule_hint: str | None = None,
) -> str:
    """Build a prompt-only eval request with explicit provenance rules."""
    return _build_rulespec_eval_prompt(
        citation=citation,
        mode=mode,
        workspace=workspace,
        context_files=context_files,
        target_file_name=target_file_name,
        include_tests=include_tests,
        runner_backend=runner_backend,
        policyengine_rule_hint=policyengine_rule_hint,
    )


def _is_single_amount_table_slice(source_text: str) -> bool:
    """Return True when source text is a single amount-bearing table row slice."""
    marker = "Structured table:"
    if marker in source_text:
        table_section = source_text.split(marker, 1)[1]
        lines = [line.strip() for line in table_section.splitlines() if line.strip()]
        amount_rows = [line for line in lines if "|" in line and re.search(r"\d", line)]
        if len(amount_rows) == 1:
            return True

    money_matches = re.findall(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", source_text)
    if len(money_matches) != 1:
        return False
    if _CONDITIONAL_AMOUNT_SLICE_PATTERN.search(source_text):
        return False

    money_lines = [
        line.strip()
        for line in source_text.splitlines()
        if re.search(r"[£$€]\s*\d[\d,]*(?:\.\d+)?", line)
    ]
    return len(money_lines) == 1


def _format_inline_context_snippets(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    max_chars_per_file: int = 6000,
) -> str:
    """Inline copied precedent files for non-tool backends like Responses API."""
    snippets: list[str] = []
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            content = path.read_text().strip()
        except OSError:
            continue
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file].rstrip() + "\n... [truncated]"
        snippets.append(f"=== FILE: {item.workspace_path} ===\n{content}")
    return "\n\n".join(snippets)


def _format_context_file_listing(
    item: EvalContextFile,
    *,
    include_label: bool = False,
) -> str:
    """Format one copied context file for prompt display."""
    details = f": {item.label or item.source_path}" if include_label else ""
    if item.workspace_path == item.import_path:
        return f"- `{item.workspace_path}`{details}"
    return (
        f"- inspect `{item.workspace_path}`; import target `{item.import_path}`"
        f"{details}"
    )


def _collect_scaffold_dates(
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
) -> list[str]:
    """Collect usable temporal scaffold dates from copied precedent files."""
    dates: set[str] = set()
    for item in context_files:
        path = workspace.root / item.workspace_path
        try:
            text = path.read_text()
        except OSError:
            continue
        dates.update(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    return sorted(dates)


def _expand_context_files(
    selected_paths: list[Path],
    statute_root: Path,
    target_rel: Path | None,
) -> list[tuple[Path, str]]:
    """Expand selected precedent files with their transitive canonical imports."""
    expanded: list[tuple[Path, str]] = []
    pending: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for path in selected_paths:
        kind = (
            "implementation_precedent"
            if _is_under_root(path, statute_root)
            else "implementation_external"
        )
        pending.append((path, kind))

    while pending:
        source_path, kind = pending.pop(0)
        resolved = source_path.resolve()
        if resolved in seen:
            continue
        if (
            target_rel is not None
            and _relative_to_root(source_path, statute_root) == target_rel
        ):
            continue
        seen.add(resolved)
        expanded.append((source_path, kind))

        if not _is_under_root(source_path, statute_root):
            continue

        for dependency in _resolve_context_imports(source_path, statute_root):
            if dependency.resolve() in seen:
                continue
            pending.append((dependency, "implementation_dependency"))

    return expanded


def _resolve_context_imports(source_path: Path, statute_root: Path) -> list[Path]:
    """Resolve canonical import targets for one copied precedent file."""
    dependencies: list[Path] = []
    for import_target in _extract_import_targets(source_path.read_text()):
        target_path = _import_target_to_path(import_target)
        candidates = [statute_root / target_path]
        if target_path.parts:
            first = target_path.parts[0]
            if first == statute_root.name:
                candidates.append(statute_root / Path(*target_path.parts[1:]))
            if first in _LOCAL_IMPORT_ROOT_TOKENS:
                candidates.append(statute_root.parent / target_path)

        for candidate in candidates:
            if candidate.exists():
                dependencies.append(candidate)
                break
    return dependencies


def _extract_import_targets(content: str) -> list[str]:
    """Extract file-level import targets from RuleSpec imports blocks."""
    targets: list[str] = []
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
            targets.append(import_target)

    return targets


def _import_target_to_path(import_target: str) -> Path:
    """Convert an import target like 26/24/c#name into 26/24/c.yaml."""
    normalized = import_target.strip().strip('"').strip("'")
    if normalized.endswith((".yaml", ".yml")):
        return Path(normalized)
    return Path(f"{normalized}.yaml")


def _is_under_root(path: Path, root: Path) -> bool:
    """Return True when a path is within the given root."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _relative_to_root(path: Path, root: Path) -> Path | None:
    """Return the path relative to the root when possible."""
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def _hydrate_eval_root(eval_root: Path, workspace: EvalWorkspace) -> None:
    """Copy allowed precedent files into the eval root so imports resolve."""
    for item in workspace.context_files:
        workspace_path = Path(item.workspace_path)
        if not workspace_path.parts or workspace_path.parts[0] != "context":
            continue

        target_relative = _import_target_to_path(item.import_path)
        target = eval_root / target_relative
        if target.exists():
            continue

        source = workspace.root / workspace_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run one prompt-only eval through the selected local CLI."""
    if runner.backend == "claude":
        return _run_claude_prompt_eval(runner, workspace, prompt)
    if runner.backend == "codex":
        return _run_codex_prompt_eval(runner, workspace, prompt)
    if runner.backend == "openai":
        return _run_openai_prompt_eval(runner, workspace, prompt)
    raise ValueError(f"Unsupported backend: {runner.backend}")


def _run_claude_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Claude CLI."""
    cmd = [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        runner.model,
        "-p",
        prompt,
    ]

    start = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=workspace.root,
        timeout=600,
    )
    duration_ms = int((time.time() - start) * 1000)

    trace: dict = {}
    text = result.stdout + result.stderr
    tokens = None
    actual_cost = None
    error = None

    try:
        payload = json.loads(text)
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "json_result": payload,
        }
        usage = payload.get("usage", {}) or {}
        tokens = TokenUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        )
        actual_cost = payload.get("total_cost_usd")
        text = payload.get("result", "") or ""
        if payload.get("is_error"):
            error = text or "Claude eval returned an error"
    except json.JSONDecodeError:
        trace = {
            "provider": "anthropic",
            "backend": "claude-print",
            "model": runner.model,
            "raw_output": result.stdout + result.stderr,
        }
        if result.returncode != 0:
            error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    if result.returncode != 0 and not error:
        error = (result.stdout + result.stderr).strip() or "Claude eval failed"

    return EvalPromptResponse(
        text=text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        actual_cost_usd=actual_cost,
        trace=trace,
        error=error,
    )


def _run_codex_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via Codex CLI."""
    codex_idle_timeout_seconds = 300
    last_message_file = workspace.root / ".codex-last-message.txt"
    if last_message_file.exists():
        last_message_file.unlink()

    cmd = [
        resolve_codex_cli(),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "-o",
        str(last_message_file),
        "-m",
        runner.model,
        "-c",
        'reasoning_effort="low"',
        "-C",
        str(workspace.root),
        "-s",
        "read-only",
        prompt,
    ]

    start = time.time()
    terminated_after_output = False
    timed_out = False
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
            cwd=workspace.root,
        )
        try:
            terminated_after_output = _wait_for_codex_process(
                process,
                last_message_file=last_message_file,
                timeout=600,
                heartbeat_paths=[stdout_path, stderr_path],
                max_idle_seconds=codex_idle_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()

    stdout_text = stdout_path.read_text()
    stderr_text = stderr_path.read_text()
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    duration_ms = int((time.time() - start) * 1000)

    events: list[dict] = []
    assistant_messages: list[str] = []
    usage_payload: dict | None = None
    unexpected_accesses: list[str] = []
    error = None

    for line in (stdout_text + stderr_text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        events.append(payload)
        if payload.get("type") == "item.completed":
            item = payload.get("item", {}) or {}
            if item.get("type") == "agent_message" and item.get("text"):
                assistant_messages.append(item["text"])
            if item.get("type") == "command_execution":
                command = item.get("command", "")
                if _command_looks_out_of_bounds(command, workspace.root):
                    unexpected_accesses.append(command)
        elif payload.get("type") == "turn.completed":
            usage_payload = payload.get("usage") or {}
        elif payload.get("type") == "error":
            error = payload.get("message") or "Codex eval error"

    tokens = None
    if usage_payload is not None:
        tokens = TokenUsage(
            input_tokens=int(usage_payload.get("input_tokens", 0) or 0),
            output_tokens=int(usage_payload.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage_payload.get("cached_input_tokens", 0) or 0),
            reasoning_output_tokens=extract_reasoning_output_tokens(
                {
                    "provider": "openai",
                    "events": events,
                }
            ),
        )

    final_text = "\n".join(assistant_messages).strip()
    if last_message_file.exists():
        file_text = last_message_file.read_text().strip()
        if file_text:
            final_text = file_text

    if timed_out and not error and not final_text:
        error = "Codex eval timed out"

    if (
        process.returncode != 0
        and not error
        and not ((terminated_after_output and final_text) or (timed_out and final_text))
    ):
        error = (stdout_text + stderr_text).strip() or "Codex eval failed"

    return EvalPromptResponse(
        text=final_text,
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace={
            "provider": "openai",
            "backend": "codex-exec",
            "model": runner.model,
            "events": events,
        },
        unexpected_accesses=unexpected_accesses,
        error=error,
    )


def _wait_for_codex_process(
    process: subprocess.Popen[str],
    last_message_file: Path,
    timeout: int,
    *,
    heartbeat_paths: Sequence[Path] | None = None,
    settle_seconds: float = 5.0,
    max_output_wait_seconds: float = 30.0,
    max_idle_seconds: float = 120.0,
    poll_interval: float = 0.5,
) -> bool:
    """Wait for Codex CLI, terminating it once output is stable or persistent."""
    start = time.time()
    last_snapshot: tuple[int, int] | None = None
    stable_since: float | None = None
    output_seen_at: float | None = None
    last_activity_at = start
    heartbeat_snapshot: tuple[tuple[int, int, int], ...] | None = None

    def _snapshot_activity() -> tuple[tuple[int, int, int], ...]:
        files = [last_message_file, *(heartbeat_paths or [])]
        snapshot: list[tuple[int, int, int]] = []
        for path in files:
            if not path.exists():
                snapshot.append((0, 0, 0))
                continue
            try:
                stat = path.stat()
            except OSError:
                snapshot.append((0, 0, 0))
                continue
            snapshot.append((1, stat.st_size, stat.st_mtime_ns))
        return tuple(snapshot)

    while True:
        if process.poll() is not None:
            return False

        now = time.time()
        if now - start > timeout:
            raise subprocess.TimeoutExpired(process.args, timeout)

        current_heartbeat_snapshot = _snapshot_activity()
        if current_heartbeat_snapshot != heartbeat_snapshot:
            heartbeat_snapshot = current_heartbeat_snapshot
            last_activity_at = now
        elif now - last_activity_at >= max_idle_seconds:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise subprocess.TimeoutExpired(process.args, max_idle_seconds)

        if last_message_file.exists():
            try:
                text = last_message_file.read_text().strip()
                stat = last_message_file.stat()
            except OSError:
                text = ""
                stat = None

            if text and stat is not None:
                output_seen_at = output_seen_at or now
                snapshot = (stat.st_size, stat.st_mtime_ns)
                if snapshot == last_snapshot:
                    stable_since = stable_since or now
                    if now - stable_since >= settle_seconds:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True
                else:
                    last_snapshot = snapshot
                    stable_since = None
                    if (
                        output_seen_at is not None
                        and now - output_seen_at >= max_output_wait_seconds
                    ):
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        return True

        time.sleep(poll_interval)


def _extract_openai_response_text(payload: dict) -> str:
    """Flatten a Responses API payload into assistant text."""
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            continue
        if item.get("type") == "message":
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n\n".join(texts).strip()


def _run_openai_prompt_eval(
    runner: EvalRunnerSpec,
    workspace: EvalWorkspace,
    prompt: str,
) -> EvalPromptResponse:
    """Run prompt-only eval via the OpenAI Responses API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return EvalPromptResponse(
            text="",
            duration_ms=0,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
            },
            error="OPENAI_API_KEY is not set",
        )

    body = {
        "model": runner.model,
        "input": prompt,
        "max_output_tokens": 16384,
        "reasoning": {
            "effort": "low",
            "summary": "auto",
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.time()
    try:
        response = _post_openai_eval_request(headers=headers, body=body)
    except requests.RequestException as exc:
        duration_ms = int((time.time() - start) * 1000)
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace={
                "provider": "openai",
                "backend": "responses",
                "model": runner.model,
                "request_body": body,
            },
            error=str(exc),
        )
    duration_ms = int((time.time() - start) * 1000)

    request_id = response.headers.get("x-request-id")
    try:
        payload = response.json()
    except ValueError:
        payload = {
            "error": {
                "message": response.text or f"HTTP {response.status_code}",
            }
        }

    trace = {
        "provider": "openai",
        "backend": "responses",
        "model": runner.model,
        "request_id": request_id,
        "request_body": body,
        "json_result": payload,
        "status_code": response.status_code,
    }

    if response.status_code >= 400:
        error = payload.get("error") or {}
        return EvalPromptResponse(
            text="",
            duration_ms=duration_ms,
            trace=trace,
            error=error.get("message") or response.text or "OpenAI eval failed",
        )

    usage = payload.get("usage") or {}
    input_details = usage.get("input_tokens_details") or {}
    tokens = TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_read_tokens=int(input_details.get("cached_tokens", 0) or 0),
    )
    tokens.reasoning_output_tokens = int(
        ((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    )

    return EvalPromptResponse(
        text=_extract_openai_response_text(payload),
        duration_ms=duration_ms,
        tokens=tokens,
        estimated_cost_usd=estimate_usage_cost_usd(runner.model, tokens),
        trace=trace,
    )


def _post_openai_eval_request(
    headers: dict[str, str],
    body: dict[str, object],
    attempts: int = 6,
) -> requests.Response:
    """POST a Responses API eval request with transient retry handling."""
    last_response: requests.Response | None = None
    last_error: requests.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
                timeout=(30, 180),
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 10))
            continue

        last_response = response
        if response.status_code not in {429, 500, 502, 503, 504} or attempt == attempts:
            return response
        time.sleep(min(2 ** (attempt - 1), 10))

    if last_response is not None:
        return last_response
    if last_error is not None:
        raise last_error
    raise requests.RequestException("OpenAI eval request failed without response")


def _command_looks_out_of_bounds(command: str, workspace_root: Path) -> bool:
    """Heuristic for whether a Codex shell command accessed paths outside workspace."""
    if not command:
        return False

    if re.search(r"(^|[\s'\"`])\.\.(?:/|[\s'\"`]|$)", command):
        return True

    abs_paths = re.findall(r"(?:(?<=^)|(?<=[\s'\"`]))(/[^\s'\"`]+)", command)
    for raw_path in abs_paths:
        path = Path(raw_path)
        try:
            path.resolve().relative_to(workspace_root.resolve())
        except ValueError:
            return True
        except FileNotFoundError:
            return True

    return False


def _extract_rulespec_content(llm_response: str) -> str | None:
    """Extract raw RuleSpec content from a model response."""
    if not llm_response or not llm_response.strip():
        return None

    cleaned = re.sub(r"\x1b\[[0-9;]*m", "", llm_response)

    fence_match = re.search(r"```(?:yaml|text)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()
        if content.startswith("format: rulespec/v1"):
            return content + "\n"

    stripped = cleaned.strip()
    if re.match(r"^(format:\s*rulespec/v1|schema:\s*axiom\.rules\.)", stripped):
        return stripped + ("\n" if not stripped.endswith("\n") else "")

    lines = stripped.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(format:\s*rulespec/v1|schema:\s*axiom\.rules\.)", line):
            return "\n".join(lines[i:]).strip() + "\n"

    return None


def _normalize_rulespec_content(content: str) -> str:
    """Normalize generated RuleSpec without rewriting embedded source prose."""
    stripped = content.strip()
    if stripped and not stripped.startswith("format: rulespec/v1"):
        raise ValueError("RuleSpec content must start with `format: rulespec/v1`")
    return stripped + ("\n" if stripped else "")


def _normalize_main_eval_content(
    content: str,
    *,
    target_path: Path,
    single_amount_table_slice: bool,
) -> str:
    """Normalize generated main artifacts according to their format."""
    if target_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("RuleSpec artifacts must use .yaml or .yml paths")
    return _normalize_rulespec_content(content)


def _extract_generated_file_bundle(llm_response: str) -> dict[str, str]:
    """Extract a small multi-file bundle emitted by eval backends."""
    if not llm_response or "=== FILE:" not in llm_response:
        return {}

    pattern = re.compile(r"^=== FILE:\s*(?P<name>.+?)\s*===\s*$", re.MULTILINE)
    matches = list(pattern.finditer(llm_response))
    if not matches:
        return {}

    files: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(llm_response)
        )
        content = llm_response[start:end].strip()
        if content:
            files[match.group("name").strip()] = _clean_generated_file_content(content)
    return files


def _clean_generated_file_content(content: str) -> str:
    """Strip common wrapper noise from bundled file content."""
    stripped = content.strip()
    fenced = re.match(
        r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```(?:\s|$)",
        stripped,
        re.DOTALL,
    )
    if fenced:
        stripped = fenced.group(1).strip()
    stripped = re.sub(
        r"(?<![\d.])(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+(GBP|USD|EUR)\b",
        lambda match: match.group(1),
        stripped,
    )
    return stripped + ("\n" if stripped and not stripped.endswith("\n") else "")


def _normalize_comma_numeric_literals(content: str) -> str:
    """Strip thousands separators from numeric literals without touching prose."""
    return re.sub(
        r"(?<![\d.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d.])",
        lambda match: match.group(0).replace(",", ""),
        content,
    )


def _format_safe_numeric_expression(expression: str) -> str | None:
    """Evaluate a numeric literal or simple arithmetic expression safely."""
    try:
        value = _evaluate_safe_numeric_expression(expression.strip())
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None

    if float(value).is_integer():
        return str(int(value))
    return format(value, "g")


def _evaluate_safe_numeric_expression(expression: str) -> float:
    """Return the numeric value of a simple arithmetic expression."""
    node = ast.parse(expression, mode="eval")

    def visit(current: ast.AST) -> float:
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.Constant) and isinstance(
            current.value, (int, float)
        ):
            return float(current.value)
        if isinstance(current, ast.UnaryOp) and isinstance(
            current.op, (ast.UAdd, ast.USub)
        ):
            operand = visit(current.operand)
            return operand if isinstance(current.op, ast.UAdd) else -operand
        if isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = visit(current.left)
            right = visit(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"Unsupported numeric expression: {expression}")

    return visit(node)


def _normalize_single_amount_row_test_content(
    content: str,
    rulespec_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Drop alternate-branch tests for one-row fixed-amount slices."""
    normalized = _normalize_comma_numeric_literals(content)
    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    annual_period = bool(
        rulespec_content
        and re.search(r"^\s*period:\s*Year\s*$", rulespec_content, flags=re.MULTILINE)
    )
    effective_date = _extract_effective_date_for_tests(
        rulespec_content=rulespec_content,
        source_text=source_text,
    )

    def should_keep(case_name: str | None) -> bool:
        if not case_name:
            return True
        lowered = case_name.lower()
        if "alternate" in lowered:
            return False
        if annual_period and "effective_date_boundary" in lowered:
            return False
        return True

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = dict(case)
        if annual_period and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        for key in ("input", "inputs", "output"):
            if key in normalized_case and isinstance(normalized_case[key], dict):
                normalized_case[key] = {
                    child_key: _normalize_test_case_value(child_value)
                    for child_key, child_value in normalized_case[key].items()
                }
        output = normalized_case.get("output")
        if isinstance(output, dict) and len(output) > 1:
            numeric_output = {
                key: value
                for key, value in output.items()
                if value is None
                or (isinstance(value, (int, float)) and not isinstance(value, bool))
            }
            if numeric_output:
                normalized_case["output"] = numeric_output
        return normalized_case

    cases = _coerce_test_payload_to_case_list(payload)
    if cases is not None:
        filtered = [
            normalize_case(case)
            for case in cases
            if not isinstance(case, dict) or should_keep(case.get("name"))
        ]
        return yaml.safe_dump(filtered, sort_keys=False).strip() + "\n"

    return normalized


def _extract_effective_date_for_tests(
    rulespec_content: str | None,
    source_text: str | None,
) -> date | None:
    """Return the earliest explicit effective date available for test normalization."""
    if rulespec_content:
        if from_match := re.search(
            r"\beffective_from:\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
            rulespec_content,
        ):
            parsed = date.fromisoformat(from_match.group(1))
            if parsed != date(1, 1, 1):
                return parsed
    if source_text and (
        source_match := re.search(
            r"\b(?:text|current text)\s+valid\s+from\s+(\d{4}-\d{2}-\d{2})\b",
            source_text,
            flags=re.IGNORECASE,
        )
    ):
        parsed = date.fromisoformat(source_match.group(1))
        if parsed != date(1, 1, 1):
            return parsed
    return None


def _normalize_annual_test_period_value(
    period: object,
    effective_date: date,
) -> object:
    """Convert bare annual periods into concrete dates on or after the effective date."""
    year: int | None = None
    if period is None:
        year = effective_date.year
    elif isinstance(period, int):
        year = period
    elif isinstance(period, str) and re.fullmatch(r"\d{4}", period):
        year = int(period)

    if year is None:
        return period
    if year < effective_date.year:
        return period
    if year == effective_date.year:
        return effective_date.isoformat()
    return f"{year}-01-01"


def _extract_rulespec_period_granularity(rulespec_content: str | None) -> str | None:
    """Return the first declared RuleSpec period name."""
    if rulespec_content is None:
        return None
    match = re.search(
        r"^\s*period:\s*(Year|Month|Week|Day)\s*$",
        rulespec_content,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _default_test_period_for_granularity(
    granularity: str | None,
    effective_date: date,
) -> str:
    """Return a concrete test period compatible with the test runner."""
    return effective_date.isoformat()


def _normalize_nonannual_test_period_value(
    period: object,
    effective_date: date,
    granularity: str | None = None,
) -> object:
    """Normalize non-annual test periods while preserving explicit monthly boundary tests."""
    if granularity == "Month":
        effective_month = effective_date.strftime("%Y-%m")
        if period is None:
            return effective_month
        if isinstance(period, date):
            period_month = period.strftime("%Y-%m")
            return period_month
        if isinstance(period, int):
            if period == effective_date.year:
                return effective_month
            if period > effective_date.year:
                return f"{period}-01"
            return period
        if isinstance(period, str):
            if re.fullmatch(r"\d{4}", period):
                year = int(period)
                if year == effective_date.year:
                    return effective_month
                if year > effective_date.year:
                    return f"{year}-01"
                return period
            if _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
                week_year = int(period[:4])
                if week_year == effective_date.year:
                    return effective_month
                if week_year > effective_date.year:
                    return f"{week_year}-01"
                return period
            if re.fullmatch(r"\d{4}-\d{2}", period):
                return period
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
                try:
                    parsed = date.fromisoformat(period)
                except ValueError:
                    return period
                return parsed.strftime("%Y-%m")

    if period is None:
        return effective_date.isoformat()
    if isinstance(period, date):
        if period < effective_date:
            return effective_date.isoformat()
        return period.isoformat()
    if isinstance(period, int):
        if period == effective_date.year:
            return effective_date.isoformat()
        if period > effective_date.year:
            return f"{period}-01-01"
        return period
    if isinstance(period, str):
        if re.fullmatch(r"\d{4}", period):
            year = int(period)
            if year == effective_date.year:
                return effective_date.isoformat()
            if year > effective_date.year:
                return f"{year}-01-01"
            return period
        if _ISO_WEEK_PERIOD_PATTERN.fullmatch(period):
            week_year = int(period[:4])
            if week_year == effective_date.year:
                return effective_date.isoformat()
            if week_year > effective_date.year:
                return f"{week_year}-01-01"
            return period
        if re.fullmatch(r"\d{4}-\d{2}", period):
            if period == effective_date.strftime("%Y-%m"):
                return effective_date.isoformat()
            return period
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            try:
                parsed = date.fromisoformat(period)
            except ValueError:
                return period
            if parsed < effective_date:
                return effective_date.isoformat()
            return period
    return period


def _normalize_placeholder_monthly_test_period_value(period: object) -> object:
    """Replace placeholder month periods with a contemporary comparable month."""
    contemporary_month = "2024-01"
    if period is None:
        return contemporary_month
    if isinstance(period, date) and period.year <= 1:
        return contemporary_month
    if isinstance(period, int) and period <= 1:
        return contemporary_month
    if isinstance(period, str):
        stripped = period.strip()
        if stripped in {"", "1", "0001"}:
            return contemporary_month
        if re.fullmatch(r"0001-\d{2}", stripped):
            return contemporary_month
        if re.fullmatch(r"0001-\d{2}-\d{2}", stripped):
            return contemporary_month
    return period


def _coerce_test_payload_to_case_list(payload: object) -> list[object] | None:
    """Return test payloads as a plain list of case objects when recognizable."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    tests_payload = payload.get("tests")
    if isinstance(tests_payload, list):
        return tests_payload

    case_like_keys = {"name", "period", "input", "inputs", "output", "expect"}
    if case_like_keys & set(payload):
        return [payload]

    if not payload or not all(isinstance(value, dict) for value in payload.values()):
        return None

    cases: list[object] = []
    for key, value in payload.items():
        case = dict(value)
        if isinstance(key, str) and "name" not in case:
            case["name"] = key
        cases.append(case)
    return cases


def _normalize_test_case_value(value: object) -> object:
    """Collapse entity/time wrappers in generated RuleSpec test values to scalars."""
    if isinstance(value, list):
        return [_normalize_test_case_value(item) for item in value]
    if isinstance(value, str):
        expression = value.strip()
        if _PURE_NUMERIC_EXPRESSION_PATTERN.fullmatch(expression):
            try:
                return yaml.safe_load(_format_safe_numeric_expression(expression))
            except (TypeError, ValueError, yaml.YAMLError):
                return value
        return value
    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_test_case_value(inner) for key, inner in value.items()
    }
    lowered_keys = {str(key).lower() for key in normalized.keys()}
    metadata_keys = {
        "entity",
        "period",
        "dtype",
        "unit",
        "label",
        "description",
        "default",
    }

    values_entries = [
        inner for key, inner in normalized.items() if str(key).lower() == "values"
    ]
    if len(values_entries) == 1 and lowered_keys.issubset(metadata_keys | {"values"}):
        values_entry = values_entries[0]
        if isinstance(values_entry, dict) and len(values_entry) == 1:
            return _normalize_test_case_value(next(iter(values_entry.values())))
        return _normalize_test_case_value(values_entry)

    from_entries = [
        inner
        for key, inner in normalized.items()
        if str(key).lower().startswith("from ")
    ]
    if from_entries and lowered_keys.issubset(
        metadata_keys
        | {
            str(key).lower()
            for key in normalized.keys()
            if str(key).lower().startswith("from ")
        }
    ):
        if len(from_entries) == 1:
            return _normalize_test_case_value(from_entries[0])

    if len(normalized) == 1:
        only_key, only_value = next(iter(normalized.items()))
        if re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", str(only_key)):
            return _normalize_test_case_value(only_value)
        if not isinstance(only_value, (dict, list)):
            return only_value

    return normalized


def _period_precedes_effective_month(period: object, effective_date: date) -> bool:
    """Return True when an explicit period falls before the effective month."""
    effective_month = effective_date.strftime("%Y-%m")
    if isinstance(period, date):
        return period.strftime("%Y-%m") < effective_month
    if isinstance(period, str):
        if re.fullmatch(r"\d{4}-\d{2}", period):
            return period < effective_month
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            with contextlib.suppress(ValueError):
                return date.fromisoformat(period).strftime("%Y-%m") < effective_month
    return False


def _case_outputs_only_zero_values(case: object) -> bool:
    """Return True when a test case only asserts zero/false-like outputs."""
    if not isinstance(case, dict):
        return False
    output = case.get("output")
    if not isinstance(output, dict) or not output:
        return False
    saw_value = False
    for value in output.values():
        if value is None:
            continue
        saw_value = True
        if isinstance(value, bool):
            if value:
                return False
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value) != 0.0:
                return False
            continue
        return False
    return saw_value


def _case_output_keys(case: object) -> set[str]:
    if not isinstance(case, dict):
        return set()
    output = case.get("output")
    if not isinstance(output, dict):
        return set()
    return {str(key) for key in output.keys()}


def _normalize_test_periods_to_effective_dates(
    content: str,
    rulespec_content: str | None = None,
    source_text: str | None = None,
) -> str:
    """Normalize annual test periods to concrete dates that survive effective-date compilation."""
    normalized = _normalize_comma_numeric_literals(content)
    granularity = _extract_rulespec_period_granularity(rulespec_content)
    effective_date = _extract_effective_date_for_tests(
        rulespec_content=rulespec_content,
        source_text=source_text,
    )

    try:
        payload = yaml.safe_load(normalized)
    except yaml.YAMLError:
        return normalized

    if payload is None:
        return normalized

    cases = _coerce_test_payload_to_case_list(payload)
    positive_output_keys: set[str] = set()
    if cases is not None:
        for case in cases:
            if not isinstance(case, dict):
                continue
            output = case.get("output")
            if not isinstance(output, dict):
                continue
            if any(
                (isinstance(value, bool) and value)
                or (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) != 0.0
                )
                for value in output.values()
                if value is not None
            ):
                positive_output_keys.update(_case_output_keys(case))

    def normalize_case(case: object) -> object:
        if not isinstance(case, dict):
            return case
        normalized_case = dict(case)
        if granularity == "Year" and effective_date is not None:
            normalized_case["period"] = _normalize_annual_test_period_value(
                normalized_case.get("period"),
                effective_date,
            )
        elif effective_date is not None:
            normalized_case["period"] = _normalize_nonannual_test_period_value(
                normalized_case.get("period"),
                effective_date,
                granularity=granularity,
            )
        elif granularity == "Month":
            normalized_case["period"] = (
                _normalize_placeholder_monthly_test_period_value(
                    normalized_case.get("period")
                )
            )

        for key in ("input", "inputs", "output"):
            if key in normalized_case and isinstance(normalized_case[key], dict):
                normalized_case[key] = {
                    child_key: _normalize_test_case_value(child_value)
                    for child_key, child_value in normalized_case[key].items()
                }
        if "expect" in normalized_case:
            normalized_case["expect"] = _normalize_test_case_value(
                normalized_case["expect"]
            )
        return normalized_case

    if cases is not None:
        filtered_cases: list[object] = []
        for case in cases:
            if (
                granularity == "Month"
                and effective_date is not None
                and isinstance(case, dict)
                and "pre_effective" in str(case.get("name", "")).lower()
                and _period_precedes_effective_month(case.get("period"), effective_date)
                and _case_outputs_only_zero_values(case)
                and (_case_output_keys(case) & positive_output_keys)
            ):
                continue
            filtered_cases.append(case)
        return (
            yaml.safe_dump(
                [normalize_case(case) for case in filtered_cases],
                sort_keys=False,
            ).strip()
            + "\n"
        )

    return normalized


def _parse_simple_rulespec_literal(value: str) -> object | None:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped in {"true", "false"}:
        return stripped == "true"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        return yaml.safe_load(stripped)
    return None


def _extract_simple_rulespec_constant(
    rulespec_content: str | None,
    var_name: str | None,
) -> object | None:
    if not rulespec_content or not var_name:
        return None
    with contextlib.suppress(yaml.YAMLError, TypeError):
        payload = yaml.safe_load(rulespec_content)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            for rule in payload["rules"]:
                if not isinstance(rule, dict) or rule.get("name") != var_name:
                    continue
                versions = rule.get("versions")
                if not isinstance(versions, list):
                    continue
                for version in versions:
                    if not isinstance(version, dict):
                        continue
                    formula = version.get("formula")
                    if isinstance(formula, (int, float, bool)):
                        return formula
                    if isinstance(formula, str):
                        literal = _parse_simple_rulespec_literal(formula)
                        if literal is not None:
                            return literal
    return None


def _complete_oracle_hint_test_outputs(
    content: str,
    rulespec_content: str | None,
    policyengine_rule_hint: str | None,
) -> str:
    if not policyengine_rule_hint:
        return content
    hint_value = _extract_simple_rulespec_constant(
        rulespec_content, policyengine_rule_hint
    )
    if hint_value is None:
        return content
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError:
        return content
    if not isinstance(payload, list):
        return content

    changed = False
    normalized_cases: list[object] = []
    for case in payload:
        if not isinstance(case, dict):
            normalized_cases.append(case)
            continue
        output = case.get("output")
        if not isinstance(output, dict) or policyengine_rule_hint in output:
            normalized_cases.append(case)
            continue
        normalized_case = dict(case)
        normalized_output = dict(output)
        normalized_output[policyengine_rule_hint] = hint_value
        normalized_case["output"] = normalized_output
        normalized_cases.append(normalized_case)
        changed = True
    if not changed:
        return content
    return yaml.safe_dump(normalized_cases, sort_keys=False).strip() + "\n"


def _materialize_eval_artifact(
    llm_response: str,
    expected_path: Path,
    source_text: str | None = None,
    workspace_root: Path | None = None,
    policyengine_rule_hint: str | None = None,
) -> bool:
    """Write an eval artifact and optional companion test file from model output."""
    single_amount_table_slice = bool(
        source_text and _is_single_amount_table_slice(source_text)
    )
    expected_test_path = _rulespec_test_path(expected_path)

    if workspace_root is not None:
        wrote_from_workspace = _materialize_workspace_artifacts(
            expected_path=expected_path,
            expected_test_path=expected_test_path,
            workspace_root=workspace_root,
            single_amount_table_slice=single_amount_table_slice,
            source_text=source_text,
            policyengine_rule_hint=policyengine_rule_hint,
        )
        if wrote_from_workspace:
            return True

    bundle = _extract_generated_file_bundle(llm_response)
    if bundle:
        wrote_main = False
        bundle_by_candidate_name = {
            Path(file_name).name: content for file_name, content in bundle.items()
        }
        for file_name, content in bundle.items():
            candidate_name = Path(file_name).name
            if candidate_name == expected_path.name:
                target_path = expected_path
            elif candidate_name == expected_test_path.name:
                target_path = expected_test_path
            else:
                continue
            if target_path == expected_path:
                try:
                    content = _normalize_main_eval_content(
                        content,
                        target_path=target_path,
                        single_amount_table_slice=single_amount_table_slice,
                    )
                except ValueError:
                    continue
            elif target_path == expected_test_path:
                if single_amount_table_slice:
                    content = _normalize_single_amount_row_test_content(
                        content,
                        rulespec_content=bundle_by_candidate_name.get(
                            expected_path.name
                        ),
                        source_text=source_text,
                    )
                else:
                    content = _normalize_test_periods_to_effective_dates(
                        content,
                        rulespec_content=bundle_by_candidate_name.get(
                            expected_path.name
                        ),
                        source_text=source_text,
                    )
                    content = _complete_oracle_hint_test_outputs(
                        content,
                        rulespec_content=bundle_by_candidate_name.get(
                            expected_path.name
                        ),
                        policyengine_rule_hint=policyengine_rule_hint,
                    )
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)
            if target_path == expected_path:
                wrote_main = True
        if wrote_main or expected_path.exists():
            return True

    rulespec_content = _extract_rulespec_content(llm_response)
    if not rulespec_content:
        return False
    try:
        rulespec_content = _normalize_main_eval_content(
            rulespec_content,
            target_path=expected_path,
            single_amount_table_slice=single_amount_table_slice,
        )
    except ValueError:
        return False

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(rulespec_content)
    return True


def _materialize_workspace_artifacts(
    expected_path: Path,
    expected_test_path: Path,
    workspace_root: Path,
    single_amount_table_slice: bool,
    source_text: str | None,
    policyengine_rule_hint: str | None = None,
) -> bool:
    """Salvage eval artifacts that a model wrote directly into the workspace."""
    workspace_main = workspace_root / expected_path.name
    workspace_test = workspace_root / expected_test_path.name
    if not workspace_main.exists():
        return False

    main_content = workspace_main.read_text()
    try:
        main_content = _normalize_main_eval_content(
            main_content,
            target_path=expected_path,
            single_amount_table_slice=single_amount_table_slice,
        )
    except ValueError:
        return False

    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_text(main_content)

    if workspace_test.exists():
        test_content = workspace_test.read_text()
        if single_amount_table_slice:
            test_content = _normalize_single_amount_row_test_content(
                test_content,
                rulespec_content=main_content,
                source_text=source_text,
            )
        else:
            test_content = _normalize_test_periods_to_effective_dates(
                test_content,
                rulespec_content=main_content,
                source_text=source_text,
            )
            test_content = _complete_oracle_hint_test_outputs(
                test_content,
                rulespec_content=main_content,
                policyengine_rule_hint=policyengine_rule_hint,
            )
        expected_test_path.write_text(test_content)

    return True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
