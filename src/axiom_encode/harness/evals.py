"""Model comparison evals for statute and corpus-backed policy encoding."""

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
from typing import Iterator, Literal, Sequence

import requests
import yaml

from axiom_encode.codex_cli import resolve_codex_cli
from axiom_encode.constants import DEFAULT_OPENAI_MODEL
from axiom_encode.statute import (
    CitationParts,
    citation_to_citation_path,
    citation_to_relative_rulespec_path,
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
    _candidate_local_corpus_provision_files,
    _fetch_supabase_corpus_source_text,
    _read_local_corpus_provision_file,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_grounding_source_text,
    extract_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    numeric_value_is_grounded,
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
_LOCAL_IMPORT_ROOT_TOKENS = {"legislation", "statutes", "regulation"}


def _matching_numeric_occurrence_count(
    occurrences: Counter[float],
    value: float,
) -> int:
    """Count occurrences whose float value closely matches the source value."""
    return sum(
        count
        for occurrence_value, count in occurrences.items()
        if numeric_value_is_grounded(occurrence_value, {value})
    )


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
    source_text_file: Path
    manifest_file: Path
    source_metadata_file: Path | None = None
    source_metadata: dict[str, object] | None = None
    context_files: list[EvalContextFile] = field(default_factory=list)


@dataclass(frozen=True)
class CorpusSourceUnit:
    """A normalized source unit resolved from corpus.provisions."""

    requested: str
    citation_path: str
    body: str
    source: Literal["local", "supabase"]


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
    corpus_citation_path: str | None = None
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
    policy_path: Path,
    runtime_axiom_rules_path: Path,
    corpus_path: Path,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    include_tests: bool = False,
) -> list[EvalResult]:
    """Run a deterministic comparison over one or more citations."""
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        for citation in citations:
            results.append(
                _run_single_eval(
                    citation=citation,
                    runner=runner,
                    output_root=output_root,
                    policy_path=policy_path,
                    runtime_axiom_rules_path=runtime_axiom_rules_path,
                    corpus_path=corpus_path,
                    mode=mode,
                    extra_context_paths=extra_context_paths or [],
                    include_tests=include_tests,
                )
            )

    return results


def run_source_eval(
    source_id: str,
    source_text: str,
    runner_specs: list[str],
    output_root: Path,
    policy_path: Path,
    source_metadata_payload: dict[str, object] | None = None,
    runtime_axiom_rules_path: Path | None = None,
    mode: EvalMode = "repo-augmented",
    extra_context_paths: list[Path] | None = None,
    oracle: EvalOracleMode = "none",
    policyengine_country: str = "auto",
    policyengine_rule_hint: str | None = None,
) -> list[EvalResult]:
    """Run a deterministic comparison over one corpus-backed source unit."""
    results: list[EvalResult] = []

    for runner in [parse_runner_spec(spec) for spec in runner_specs]:
        results.append(
            _run_single_source_eval(
                source_id=source_id,
                source_text=source_text,
                runner=runner,
                output_root=output_root,
                policy_path=policy_path,
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
        if "source_file" in item:
            raise ValueError(
                "Eval suite source cases must use 'corpus_citation_path'; "
                f"'source_file' is no longer supported in {path}"
            )
        if "metadata_file" in item:
            raise ValueError(
                "Eval suite source metadata now comes from corpus.provisions; "
                f"'metadata_file' is no longer supported in {path}"
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
            corpus_citation_path=(
                str(item.get("corpus_citation_path")).strip()
                if item.get("corpus_citation_path") is not None
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
    corpus_path: Path | None = None,
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
                        if corpus_path is None:
                            raise ValueError(
                                "corpus_path is required for citation eval suite cases"
                            )
                        case_results = run_model_eval(
                            citations=[case.citation or ""],
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            policy_path=(
                                axiom_rules_path.parent / "rulespec-us"
                                if axiom_rules_path.name == "axiom-rules-engine"
                                else axiom_rules_path
                            ),
                            runtime_axiom_rules_path=axiom_rules_path,
                            corpus_path=corpus_path,
                            mode=case.mode,
                            extra_context_paths=extra_context,
                        )
                    elif case.kind == "source":
                        if corpus_path is None:
                            raise ValueError(
                                "corpus_path is required for corpus-backed "
                                "source eval suite cases"
                            )
                        source_unit = resolve_corpus_source_unit(
                            case.corpus_citation_path or "",
                            corpus_path,
                        )
                        source_text = source_unit.body
                        policy_repo_root = _policy_repo_root_for_corpus_source(
                            source_unit.citation_path,
                            axiom_rules_path,
                        )
                        source_metadata_payload = {
                            "corpus_citation_path": source_unit.citation_path,
                            "corpus_source": source_unit.source,
                            "requested_source": source_unit.requested,
                        }
                        case_results = run_source_eval(
                            source_id=case.source_id or case.name,
                            source_text=source_text,
                            runner_specs=resolved_runners,
                            output_root=case_output_root,
                            policy_path=policy_repo_root,
                            source_metadata_payload=source_metadata_payload,
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
        if not case.corpus_citation_path:
            raise ValueError(
                f"Eval suite case #{index} is missing 'corpus_citation_path'"
            )


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
    policy_root: Path,
    max_files: int = 6,
) -> list[Path]:
    """Select canonical implementation precedent files for repo-augmented evals."""
    parts = (
        citation
        if isinstance(citation, CitationParts)
        else parse_usc_citation(citation)
    )
    repo_root = Path(policy_root)
    statutes_root = repo_root / "statutes"
    section_root = statutes_root / parts.title / parts.section
    target_rel = citation_to_relative_rulespec_path(parts)
    target_path = repo_root / target_rel

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
        title_root = statutes_root / parts.title
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
            len(path.relative_to(statutes_root).parts),
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

    source_text_file = workspace_root / "source.txt"
    source_text_file.write_text(source_text.strip() + "\n")
    source_metadata = dict(source_metadata_payload or {})
    if not source_metadata:
        source_metadata = None
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
        selected.extend(
            _select_child_fragment_context_files(citation, context_corpus_root)
        )
        selected.extend(
            _select_same_section_subsection_context_files(
                citation,
                source_text,
                context_corpus_root,
            )
        )
        selected.extend(
            _select_cross_section_context_files(
                citation,
                source_text,
                context_corpus_root,
            )
        )
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
                    import_path=_context_import_target(source_path, relative_target),
                    kind=kind,
                )
            )

    manifest_file = workspace_root / "context-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "citation": citation,
                "mode": mode,
                "source_text_file": str(source_text_file.relative_to(workspace_root)),
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
        source_text_file=source_text_file,
        manifest_file=manifest_file,
        source_metadata_file=source_metadata_file,
        source_metadata=source_metadata,
        context_files=context_files,
    )


def resolve_corpus_source_unit(
    identifier: str,
    corpus_path: Path,
) -> CorpusSourceUnit:
    """Resolve an encode target to normalized corpus.provisions text.

    The identifier may already be a corpus citation path, or it may be a USC
    citation that can be normalized to one. For USC child fragments, the
    resolver falls back to the nearest available section-level corpus provision
    and slices that body to the requested fragment when the parent text carries
    structural markers such as ``(a)``, ``(2)``, or ``(C)``.
    """
    candidates = _candidate_corpus_citation_paths(identifier)
    primary = candidates[0] if candidates else ""
    for citation_path in candidates:
        local_text = _fetch_local_corpus_source_text_from_repo(
            citation_path,
            corpus_path,
        )
        if local_text is not None:
            body = _slice_parent_corpus_text_for_requested_path(
                local_text,
                requested_path=primary,
                resolved_path=citation_path,
            )
            return CorpusSourceUnit(
                requested=identifier,
                citation_path=citation_path,
                body=body,
                source="local",
            )
        supabase_text = _fetch_supabase_corpus_source_text(citation_path)
        if supabase_text is not None:
            body = _slice_parent_corpus_text_for_requested_path(
                supabase_text,
                requested_path=primary,
                resolved_path=citation_path,
            )
            return CorpusSourceUnit(
                requested=identifier,
                citation_path=citation_path,
                body=body,
                source="supabase",
            )

    candidates = ", ".join(_candidate_corpus_citation_paths(identifier)[:4])
    raise ValueError(
        "No corpus.provisions source text found for "
        f"{identifier!r}. Tried: {candidates}"
    )


def _slice_parent_corpus_text_for_requested_path(
    text: str,
    *,
    requested_path: str,
    resolved_path: str,
) -> str:
    """Slice section-granular USC source text to the requested child fragment."""
    requested_parts = requested_path.strip("/").split("/")
    resolved_parts = resolved_path.strip("/").split("/")
    if (
        len(requested_parts) <= len(resolved_parts)
        or requested_parts[: len(resolved_parts)] != resolved_parts
        or resolved_parts[:2] != ["us", "statute"]
    ):
        return text
    missing_fragments = tuple(requested_parts[len(resolved_parts) :])
    sliced = _slice_legal_text_by_parenthetical_fragments(text, missing_fragments)
    return sliced if sliced is not None else text


def _slice_legal_text_by_parenthetical_fragments(
    text: str,
    fragments: tuple[str, ...],
) -> str | None:
    current = text
    for depth, fragment in enumerate(fragments):
        current = _slice_legal_text_by_parenthetical_fragment(
            current,
            fragment,
            top_level=depth == 0,
        )
        if current is None:
            return None
    return current.strip()


def _slice_legal_text_by_parenthetical_fragment(
    text: str,
    fragment: str,
    *,
    top_level: bool,
) -> str | None:
    escaped = re.escape(fragment)
    if top_level:
        marker_pattern = re.compile(rf"(?:^|\n\s*\n)(\({escaped}\)\s+)")
    else:
        marker_pattern = re.compile(rf"(?<![A-Za-z0-9])(\({escaped}\)\s+)")
    marker_match = next(
        (
            match
            for match in marker_pattern.finditer(text)
            if _parenthetical_marker_context_is_structural(text, match.start(1))
        ),
        None,
    )
    if marker_match is None:
        return None

    start = marker_match.start(1)
    body_start = marker_match.end(1)
    sibling_pattern = _sibling_parenthetical_marker_pattern(fragment, top_level)
    end = len(text)
    for sibling_match in sibling_pattern.finditer(text, body_start):
        if sibling_match.start(
            1
        ) > start and _parenthetical_marker_context_is_structural(
            text,
            sibling_match.start(1),
        ):
            end = sibling_match.start(1)
            break
    return text[start:end]


_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL = (
    r"(?:paragraphs?|subparagraphs?|clauses?|subclauses?|sections?|"
    r"subsections?|chapters?|titles?|parts?|items?|sentences?|regulations?)"
)
_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX = re.compile(
    rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\s+$",
    re.IGNORECASE,
)


def _parenthetical_marker_context_is_structural(text: str, marker_start: int) -> bool:
    prefix = text[max(0, marker_start - 60) : marker_start]
    previous = prefix.rstrip()[-1:] if prefix.rstrip() else ""
    if previous == ")":
        return False
    if _NONSTRUCTURAL_PARENTHETICAL_REFERENCE_PREFIX.search(prefix):
        return False
    return not _parenthetical_marker_is_in_reference_list(prefix)


def _parenthetical_marker_is_in_reference_list(prefix: str) -> bool:
    segment = re.split(r"(?:[.;]\s+|\n+)", prefix)[-1]
    if not re.search(
        rf"\b{_NONSTRUCTURAL_PARENTHETICAL_REFERENCE_LABEL}\b",
        segment,
        flags=re.IGNORECASE,
    ):
        return False
    if not re.search(r"\([A-Za-z0-9]+\)", segment):
        return False
    return re.search(r"(?:,\s*|\b(?:or|and)\s+)$", segment) is not None


def _sibling_parenthetical_marker_pattern(
    fragment: str,
    top_level: bool,
) -> re.Pattern[str]:
    if fragment.isdigit():
        marker = r"\(\d+\)"
    elif len(fragment) == 1 and fragment.isalpha() and fragment.isupper():
        marker = r"\([A-Z]\)"
    elif len(fragment) == 1 and fragment.isalpha() and fragment.islower():
        marker = r"\([a-z]\)"
    elif re.fullmatch(r"[ivxlcdm]+", fragment, re.IGNORECASE):
        marker = r"\([ivxlcdm]+\)"
    else:
        marker = r"\([A-Za-z0-9]+\)"
    if top_level:
        return re.compile(rf"\n\s*\n({marker}\s+)")
    return re.compile(rf"(?<![A-Za-z0-9])({marker}\s+)")


def _candidate_corpus_citation_paths(identifier: str) -> tuple[str, ...]:
    """Return exact and nearest-parent corpus citation path candidates."""
    normalized = identifier.strip().strip("/")
    if not normalized:
        return ()

    try:
        primary = (
            normalized
            if _looks_like_corpus_citation_path(normalized)
            else citation_to_citation_path(normalized)
        )
    except ValueError:
        primary = normalized

    candidates: list[str] = []

    def add(candidate: str) -> None:
        cleaned = candidate.strip().strip("/")
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(primary)
    parts = primary.split("/")
    for end in range(len(parts) - 1, 2, -1):
        add("/".join(parts[:end]))
    return tuple(candidates)


def _looks_like_corpus_citation_path(identifier: str) -> bool:
    parts = identifier.split("/")
    return len(parts) >= 3 and parts[1] in {
        "guidance",
        "policy",
        "regulation",
        "statute",
        "statutes",
    }


def _fetch_local_corpus_source_text_from_repo(
    citation_path: str,
    corpus_path: Path,
) -> str | None:
    normalized_path = citation_path.strip().strip("/")
    if not normalized_path:
        return None
    provisions_root = _corpus_provisions_root(corpus_path)
    if provisions_root is None:
        return None
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


def _corpus_provisions_root(corpus_path: Path) -> Path | None:
    root = Path(corpus_path).expanduser()
    candidates = (
        root,
        root / "provisions",
        root / "data" / "corpus",
        root / "data" / "corpus" / "provisions",
    )
    for candidate in candidates:
        provisions_root = (
            candidate if candidate.name == "provisions" else candidate / "provisions"
        )
        with contextlib.suppress(OSError):
            resolved = provisions_root.resolve()
            if resolved.is_dir():
                return resolved
    return None


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
    shutil.copy2(resolved_concept.rulespec_file, target)
    return EvalContextFile(
        source_path=str(resolved_concept.rulespec_file),
        workspace_path=str(relative_target),
        import_path=_relative_rulespec_path_to_import_target(
            relative_target.relative_to("context")
        ),
        kind="canonical_concept",
        label=resolved_concept.label,
    )


def _auto_select_context_files(citation: str, policy_root: Path) -> list[Path]:
    """Best-effort auto-context selection for target files and nearby precedent."""
    selected: list[Path] = []
    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is not None:
        target_path = policy_root / target_rel
        if target_path.exists():
            selected.append(target_path)
        if target_path.parent.exists():
            for sibling in sorted(target_path.parent.glob("*.yaml")):
                if sibling.name.endswith(".test.yaml") or sibling == target_path:
                    continue
                selected.append(sibling)
                if len(selected) >= 6:
                    break
        if selected:
            return selected[:6]

    try:
        return select_context_files(citation, policy_root)
    except Exception:
        return []


def _select_same_section_subsection_context_files(
    citation: str,
    source_text: str,
    policy_root: Path,
) -> list[Path]:
    """Select existing RuleSpecs for same-section subsections cited by source text."""
    try:
        parts = parse_usc_citation(citation)
    except Exception:
        return []

    target_rel = citation_to_relative_rulespec_path(parts)
    selected: list[Path] = []
    seen: set[Path] = set()
    for match in re.finditer(
        r"\bsubsection\s+\((?P<subsection>[A-Za-z0-9]+)\)\s+of\s+this\s+section\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        subsection = match.group("subsection")
        if subsection in parts.fragments:
            continue
        candidate_rel = citation_to_relative_rulespec_path(
            CitationParts(parts.title, parts.section, (subsection,))
        )
        if candidate_rel == target_rel:
            continue
        candidate = policy_root / candidate_rel
        resolved = candidate.resolve()
        if candidate.exists() and resolved not in seen:
            selected.append(candidate)
            seen.add(resolved)
    return selected


def _select_cross_section_context_files(
    citation: str,
    source_text: str,
    policy_root: Path,
) -> list[Path]:
    """Select existing RuleSpecs for cited USC sections outside this section."""
    try:
        parts = parse_usc_citation(citation)
    except Exception:
        return []

    target_rel = citation_to_relative_rulespec_path(parts)
    selected: list[Path] = []
    seen: set[Path] = set()
    for match in re.finditer(
        r"\bsection\s+"
        r"(?P<section>[0-9][A-Za-z0-9.-]*)"
        r"(?P<fragments>(?:\([A-Za-z0-9]+\))*)",
        source_text,
        flags=re.IGNORECASE,
    ):
        section = match.group("section").rstrip(".")
        if section == parts.section:
            continue
        fragments = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("fragments")))
        candidate_rel = citation_to_relative_rulespec_path(
            CitationParts(parts.title, section, fragments)
        )
        if candidate_rel == target_rel:
            continue
        candidate = policy_root / candidate_rel
        resolved = candidate.resolve()
        if candidate.exists() and resolved not in seen:
            selected.append(candidate)
            seen.add(resolved)
    return selected


def _select_child_fragment_context_files(
    citation: str, policy_root: Path
) -> list[Path]:
    """Select existing child RuleSpecs when encoding an aggregate parent fragment."""
    target_rel = _target_rel_for_eval_identifier(citation)
    if target_rel is None:
        return []
    child_root = policy_root / target_rel.with_suffix("")
    if not child_root.is_dir():
        return []
    return sorted(
        path
        for path in child_root.rglob("*.yaml")
        if not path.name.endswith(".test.yaml")
    )


def _repo_augmented_context_root(policy_path: Path) -> Path:
    """Resolve the corpus root used for automatic repo-augmented context selection."""
    resolved = Path(policy_path).resolve()
    if resolved.name == "axiom-rules-engine":
        fallback = resolved.parent / "rulespec-us"
        if fallback.exists():
            return fallback
    return resolved


def _context_import_relative_target(source_path: Path, policy_path: Path) -> Path:
    """Prefer canonical repo-relative import targets for copied precedent files."""
    repo_parent = policy_path.parent.resolve()
    resolved_source = source_path.resolve()

    for candidate in sorted(repo_parent.glob("rulespec-*")):
        if not candidate.is_dir():
            continue
        resolved_candidate = candidate.resolve()
        with contextlib.suppress(ValueError):
            relative = resolved_source.relative_to(resolved_candidate)
            return relative

    return Path("external") / resolved_source.name


def _context_import_target(source_path: Path, relative_target: Path) -> str:
    """Return the canonical RuleSpec import target for a copied context file."""
    prefix = _rulespec_repo_import_prefix(source_path)
    return _relative_rulespec_path_to_import_target(relative_target, prefix=prefix)


def _rulespec_repo_import_prefix(source_path: Path) -> str | None:
    """Infer the absolute RuleSpec import prefix from a `rulespec-*` repo path."""
    for parent in (source_path, *source_path.parents):
        if parent.name.startswith("rulespec-") and len(parent.name) > len("rules-"):
            return parent.name.removeprefix("rulespec-")
    return None


def _relative_rulespec_path_to_import_target(
    path: Path,
    *,
    prefix: str | None = None,
) -> str:
    """Convert a relative RuleSpec file path into an import target."""
    normalized = path.with_suffix("") if path.suffix in {".yaml", ".yml"} else path
    target = normalized.as_posix()
    return f"{prefix}:{target}" if prefix else target


def _target_rel_for_eval_identifier(citation: str) -> Path | None:
    """Return the canonical RuleSpec target path for corpus or USC citations."""
    if _looks_like_corpus_citation_path(citation.strip().strip("/")):
        return _source_identifier_to_relative_rulespec_path(citation)
    try:
        return citation_to_relative_rulespec_path(citation)
    except Exception:
        return None


def _policy_repo_root_for_corpus_source(
    corpus_citation_path: str,
    axiom_rules_path: Path,
) -> Path:
    """Choose the jurisdiction RuleSpec repo for a corpus citation path."""
    jurisdiction = corpus_citation_path.strip().split("/", 1)[0] or "us"
    repo_name = "rulespec-us" if jurisdiction == "us" else f"rulespec-{jurisdiction}"
    if axiom_rules_path.name == repo_name:
        return axiom_rules_path
    candidate = axiom_rules_path.parent / repo_name
    return candidate if candidate.exists() else axiom_rules_path


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
    with _rulespec_validation_target(
        rulespec_file, policy_repo_root
    ) as validation_file:
        validation_policy_repo_root = _validation_policy_repo_root(
            validation_file, policy_repo_root
        )
        pipeline = ValidatorPipeline(
            policy_repo_path=validation_policy_repo_root,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=oracle != "none",
            policyengine_country=policyengine_country,
            policyengine_rule_hint=policyengine_rule_hint,
            require_policy_proofs=True,
        )
        compile_result = pipeline._run_compile_check(validation_file)
        ci_result = pipeline._run_ci(validation_file)

        policyengine_result = None
        taxsim_result = None
        if oracle in ("policyengine", "all"):
            try:
                policyengine_result = pipeline._run_policyengine(validation_file)
            except Exception as exc:
                policyengine_result = ValidationResult(
                    validator_name="policyengine",
                    passed=False,
                    error=str(exc),
                    issues=[str(exc)],
                )
        if oracle == "all":
            try:
                taxsim_result = pipeline._run_taxsim(validation_file)
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
                validation_file,
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
    numeric_source_text = extract_numeric_grounding_source_text(content)
    if not numeric_source_text and source_text:
        numeric_source_text = source_text
    numeric_validation_source_text = embedded_source or numeric_source_text
    source_numbers = extract_numbers_from_text(numeric_validation_source_text or "")
    source_numeric_occurrences = Counter(
        extract_numeric_occurrences_from_text(numeric_validation_source_text or "")
    )
    named_scalar_occurrences = Counter(
        item.value for item in extract_named_scalar_occurrences(content)
    )
    named_scalar_occurrences.update(
        _imported_named_scalar_occurrences(content, policy_repo_root)
    )

    grounding_metrics: list[GroundingMetric] = []
    for line, raw, value in extract_grounding_values(content):
        grounding_metrics.append(
            GroundingMetric(
                line=line,
                raw=raw,
                value=value,
                grounded=numeric_value_is_grounded(value, source_numbers),
            )
        )

    numeric_occurrence_issues: list[str] = []
    covered_source_numeric_occurrence_count = 0
    missing_source_numeric_occurrence_count = 0
    for value, expected_count in sorted(source_numeric_occurrences.items()):
        covered_count = min(
            expected_count,
            _matching_numeric_occurrence_count(named_scalar_occurrences, value),
        )
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
        numeric_validation_source_text,
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


def _validation_policy_repo_root(validation_file: Path, policy_repo_root: Path) -> Path:
    """Return the repo root that contains the validation copy."""
    repo_name = policy_repo_root.name
    for parent in validation_file.parents:
        if parent.name == repo_name:
            return parent
    return policy_repo_root


def _imported_named_scalar_occurrences(
    content: str,
    policy_repo_root: Path,
) -> Counter[float]:
    """Count direct scalar definitions from imported RuleSpec files."""
    occurrences: Counter[float] = Counter()
    seen: set[Path] = set()
    for import_target in _extract_import_targets(content):
        for path in _candidate_import_rule_files(import_target, policy_repo_root):
            resolved = path.resolve()
            if resolved in seen or not path.exists():
                continue
            seen.add(resolved)
            with contextlib.suppress(OSError):
                occurrences.update(
                    item.value
                    for item in extract_named_scalar_occurrences(path.read_text())
                )
            break
    return occurrences


def _candidate_import_rule_files(
    import_target: str,
    policy_repo_root: Path,
) -> list[Path]:
    """Return possible local files for an import target."""
    target_path = _import_target_to_path(import_target)
    candidates = [policy_repo_root / target_path]
    import_prefix = _import_target_prefix(import_target)
    if import_prefix:
        candidates.append(
            policy_repo_root.parent / f"rulespec-{import_prefix}" / target_path
        )
    return candidates


@contextlib.contextmanager
def _rulespec_validation_target(
    rulespec_file: Path, policy_repo_root: Path
) -> Iterator[Path]:
    """Yield a validation path whose ancestors expose canonical repo identity."""
    if _is_under_root(rulespec_file, policy_repo_root):
        yield rulespec_file
        return
    relative = _relative_rulespec_source_path(rulespec_file)
    if relative is None or not policy_repo_root.name.startswith("rulespec-"):
        yield rulespec_file
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        overlay_parent = Path(tmpdir)
        for sibling in policy_repo_root.parent.glob("rulespec-*"):
            if sibling.resolve() == policy_repo_root.resolve() or not sibling.is_dir():
                continue
            sibling_target = overlay_parent / sibling.name
            try:
                sibling_target.symlink_to(sibling.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(sibling, sibling_target, dirs_exist_ok=True)
        overlay_repo = overlay_parent / policy_repo_root.name
        shutil.copytree(policy_repo_root, overlay_repo)
        validation_file = overlay_repo / relative
        validation_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rulespec_file, validation_file)
        companion_test = rulespec_file.with_name(f"{rulespec_file.stem}.test.yaml")
        if companion_test.exists():
            validation_test = validation_file.with_name(
                f"{validation_file.stem}.test.yaml"
            )
            shutil.copy2(companion_test, validation_test)
        yield validation_file


def _relative_rulespec_source_path(path: Path) -> Path | None:
    """Return the path beginning at the RuleSpec source-root directory."""
    parts = path.parts
    for index, part in enumerate(parts):
        if part in {"policies", "regulations", "statutes"}:
            return Path(*parts[index:])
    return None


def _run_single_eval(
    citation: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    policy_path: Path,
    runtime_axiom_rules_path: Path,
    corpus_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    include_tests: bool = False,
) -> EvalResult:
    source_unit = resolve_corpus_source_unit(citation, corpus_path)
    source_text = source_unit.body

    workspace = prepare_eval_workspace(
        citation=citation,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=policy_path,
        mode=mode,
        source_metadata_payload={
            "corpus_citation_path": source_unit.citation_path,
            "corpus_source": source_unit.source,
            "requested_source": source_unit.requested,
        },
        extra_context_paths=extra_context_paths,
    )

    relative_output = (
        _source_identifier_to_relative_rulespec_path(source_unit.citation_path)
        if _looks_like_corpus_citation_path(citation)
        else citation_to_relative_rulespec_path(citation)
    )
    prompt = _build_eval_prompt(
        citation,
        mode,
        workspace,
        workspace.context_files,
        target_file_name=relative_output.name,
        target_ref_prefix=_canonical_target_ref_prefix(
            source_unit.citation_path, relative_output
        ),
        include_tests=include_tests,
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
        eval_root = Path(output_root) / runner.name
        _hydrate_eval_root(eval_root, workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(citation)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rulespec_file=output_file,
            policy_repo_root=policy_path,
            axiom_rules_path=runtime_axiom_rules_path,
            source_text=source_text,
        )
    validation_error = _eval_artifact_validation_error(metrics)

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
        success=wrote_artifact and response.error is None and validation_error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned")
        or validation_error,
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


def _eval_artifact_validation_error(metrics: EvalArtifactMetrics | None) -> str | None:
    if metrics is None:
        return None
    if not metrics.compile_pass:
        return "Generated RuleSpec failed compile validation"
    if not metrics.ci_pass:
        return "Generated RuleSpec failed CI validation"
    return None


def _run_single_source_eval(
    source_id: str,
    source_text: str,
    runner: EvalRunnerSpec,
    output_root: Path,
    policy_path: Path,
    source_metadata_payload: dict[str, object] | None,
    runtime_axiom_rules_path: Path,
    mode: EvalMode,
    extra_context_paths: list[Path],
    oracle: EvalOracleMode,
    policyengine_country: str,
    policyengine_rule_hint: str | None,
) -> EvalResult:
    """Run one eval on a corpus-backed source unit rather than a USC citation."""
    workspace = prepare_eval_workspace(
        citation=source_id,
        runner=runner,
        output_root=output_root,
        source_text=source_text,
        axiom_rules_path=policy_path,
        mode=mode,
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
        target_ref_prefix=_canonical_target_ref_prefix(source_id, relative_output),
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
        eval_root = Path(output_root) / runner.name
        _hydrate_eval_root(eval_root, workspace)

    trace_file = (
        Path(output_root) / "traces" / runner.name / f"{_slugify(source_id)}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(response.trace or {}, indent=2, sort_keys=True))

    metrics = None
    if output_file.exists():
        metrics = evaluate_artifact(
            rulespec_file=output_file,
            policy_repo_root=policy_path,
            axiom_rules_path=runtime_axiom_rules_path,
            source_text=source_text,
            oracle=oracle,
            policyengine_country=policyengine_country,
            policyengine_rule_hint=policyengine_rule_hint,
        )
    validation_error = _eval_artifact_validation_error(metrics)

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
        success=wrote_artifact and response.error is None and validation_error is None,
        error=response.error
        or (None if wrote_artifact else "No RuleSpec content returned")
        or validation_error,
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
    parts = [part for part in source_id.strip().strip("/").split("/") if part]
    if len(parts) >= 3:
        document_roots = {
            "guidance": "guidance",
            "policies": "policies",
            "policy": "policies",
            "regulation": "regulations",
            "regulations": "regulations",
            "statute": "statutes",
            "statutes": "statutes",
        }
        root = document_roots.get(parts[1])
        if root is not None:
            tail = parts[2:]
            if parts[0] == "us" and parts[1] in {"regulation", "regulations"}:
                tail = _canonical_us_regulation_tail(tail)
            if tail:
                return (Path(root) / Path(*tail)).with_suffix(".yaml")
    return Path("source") / f"{_slugify(source_id)}.yaml"


def _canonical_us_regulation_tail(tail: list[str]) -> list[str]:
    """Map federal regulation corpus paths to canonical RuleSpec repo paths."""
    if not tail:
        return tail
    title = tail[0].strip()
    if title.isdigit():
        return [f"{title}-cfr", *tail[1:]]
    return tail


def _canonical_target_ref_prefix(source_id: str, relative_path: Path) -> str | None:
    parts = [part for part in source_id.strip().strip("/").split("/") if part]
    if len(parts) < 3:
        return None
    if relative_path.parts and relative_path.parts[0] == "source":
        return None
    return f"{parts[0]}:{_relative_rulespec_path_to_import_target(relative_path)}"


def _rulespec_test_path(path: Path) -> Path:
    """Return the companion RuleSpec test path for a RuleSpec file."""
    return path.with_name(f"{path.stem}.test.yaml")


def _build_rulespec_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    target_ref_prefix: str | None,
    include_tests: bool,
    runner_backend: str,
    policyengine_rule_hint: str | None,
) -> str:
    """Build the RuleSpec authoring prompt used by current evals."""
    source_text = workspace.source_text_file.read_text().strip()
    corpus_citation_path = _workspace_corpus_citation_path(workspace)
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
If a metadata relation says this source `sets`, `amends`, `implements`, or `restates` a canonical target, record that legal/provenance edge as a separate `kind: source_relation` rule with `source_relation.type` and the absolute target path under `source_relation.target`. This is not an `amends` relationship unless the source itself amends another source.
For state option/source-slice metadata, do not add a top-level `imports:` entry to the absolute canonical target path such as `us:regulation/...#...` or `us:statutes/...#...` unless a copied context file actually provides that import target.
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
        branch_child_naming_section = _format_branch_child_naming_guidance(
            context_files,
            target_file_name=target_file_name,
            target_ref_prefix=target_ref_prefix,
        )
        cited_context_imports_section = _format_cited_context_import_guidance(
            source_text,
            context_files,
        )
        partial_extent_child_schema_section = (
            _format_partial_extent_child_schema_limit_guidance(
                source_text,
                context_files,
                target_ref_prefix=target_ref_prefix,
            )
        )
        context_section = f"""
Context mode: `{mode}`.
Context files are precedent and dependency context, not independent legal authority for new values:
{listings}
{inline_context}
{resolved_guidance}
{branch_child_naming_section}
{cited_context_imports_section}
{partial_extent_child_schema_section}
Import and context rules:
- Use the listed import target rather than the `./context/...` inspection path.
- do not wrap import targets in quotes.
- Every import path must point to a file that is actually copied into the workspace.
- Top-level `imports:` entries must be scalar strings, never map entries like
  `- target:` plus `symbols:`. Import a copied export as one exact string such
  as `us:statutes/26/45A/a#base_year_1993_indian_employment_costs`.
- If a copied context file already defines the exact symbol you need, import that exact symbol instead of inventing renamed locals that overlap with the copied file.
- Copied context listings include exported symbols as `import_target#name`; use
  those exact references in `imports:` and proof atoms when composing from context.
- In formulas, reference imported exports by their bare local rule name after adding an `imports:` entry; never write an absolute `us:...#rule_name` reference inside a formula.
- If a copied current target file already has executable `parameter`, `derived`, or `data_relation` rules, do not replace it with `module.status: deferred`, `module.status: entity_not_supported`, or `rules: []`. Preserve the executable scope and make the smallest source-faithful repair.
- If a copied current target file is present, treat it as the baseline to repair. Preserve existing rule names, imports, companion-test output keys, and public formulas unless `./source.txt` proves a specific rule legally wrong. Do not rewrite the whole file or rename established outputs just to improve style.
- For every retained executable output in a copied current target file, preserve its public executable surface: local `name`, `kind`, `entity`, `dtype`, `period`, `unit`, `indexed_by`, and every existing `versions[].effective_from`. Do not change `Employer` to `Business`, `TaxUnit` to another entity, or alter period/unit/indexing just to match a preferred modeling style. Changing an existing `dtype: Money` output to `dtype: Judgment`, or vice versa, is a forbidden public-surface migration.
- Preserve the existing factual input surface used by copied executable formulas and companion tests. Do not replace established local inputs such as `long_term_capital_gains` or `qualified_dividend_income` with newly invented upstream-sounding input names unless the task is an explicit source-grounded migration that also updates downstream tests, imports, and oracle mappings.
- When source text cites a section or subsection and a copied context file for
  that citation is listed, import and use the listed exported symbol from that
  context instead of creating a local `section_...` or `subsection_...`
  placeholder. For example, if a source references a deduction allowed by
  section 163(a) and context lists `us:statutes/26/163/a#interest_deduction`,
  import that exact export and use `interest_deduction`; a local fact such as
  `section_163_a_deduction_attributable_to_section_163_h_4_A_exception` is
  invalid.
- If this target is an aggregate parent provision and copied child-fragment files
  already encode subparagraphs, import those child outputs and compose them.
  Do not redefine the child parameters, helper rules, or copied executable
  outputs in the parent file.
- If a copied context file for this target or a same-program sibling contains a `kind: source_relation` record, preserve the legal/provenance edge unless `./source.txt` proves it wrong; executable formula changes are not a reason to drop source graph context.
- Do not fabricate same-instrument imports or `statutes/...#symbol` paths unless that exact `path#symbol` import target is listed.
- do not fabricate sibling-file imports for uncopied same-instrument provisions.
- When a copied chart or parameter file supplies values, keep `.test.yaml` inputs and expected outputs consistent with the rows visible in that imported file; do not guess contradictory expectations for those imported values.
- Do not invent degenerate placeholder rows like `number_of_children_in_assistance_unit: 0` plus `number_of_caretakers_in_assistance_unit: 0` unless that row is visible in the copied chart file.
- Do not assert an exact zero imported standard, grant, or threshold unless that exact imported row is visible in the copied chart file.
{scaffold_dates_section}
"""

    missing_cited_source_section = _format_missing_cited_source_guidance(
        citation,
        source_text,
        context_files,
    )

    test_file_name = _rulespec_test_path(Path(target_file_name)).name
    if include_tests:
        oracle_rule = ""
        if policyengine_rule_hint:
            oracle_rule = (
                "- Every non-empty test `output:` mapping must assert the "
                f"canonical RuleSpec output whose local name is "
                f"`{policyengine_rule_hint}`; use its full "
                "`jurisdiction:path#rule` id when the artifact has a legal "
                "pointer.\n"
            )
        canonical_target_rule = ""
        if target_ref_prefix:
            canonical_target_rule = f"""
	- The canonical RuleSpec reference prefix for `{target_file_name}` is `{target_ref_prefix}`.
	- In tests for this file, use `{target_ref_prefix}#input.<fact>` for local factual inputs and `{target_ref_prefix}#<rule>` for outputs. Never use `{target_file_name}#...` keys.
	"""
        proration_test_guidance = _format_proration_test_guidance(source_text)
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
- Supported mapping `period_kind` values are `tax_year` and `custom`; never use `period_kind: calendar_year`.
- For annual tax tests, use an explicit tax-year mapping such as `period: {{period_kind: tax_year, start: '2024-01-01', end: '2024-12-31'}}`.
- For non-tax annual periods, use `period: {{period_kind: custom, name: calendar_year, start: '2024-01-01', end: '2024-12-31'}}`.
- For `period: Day` outputs, use a custom day mapping such as `period: {{period_kind: custom, name: day, start: '2024-01-15', end: '2024-01-15'}}`; never use bare `YYYY-MM-DD` shorthand.
- Emit 1-4 cases unless a source-driven coverage rule below requires more. If
  `module.status` is `deferred` or `entity_not_supported`, the test file may be
  empty.
- The test file must contain YAML only; do not put prose or markdown fences in it.
- Use factual predicates or quantities in `input:`, not the output variable being asserted.
- Never assign an imported module's computed `#rule_name` output in `input:`. If this file imports that rule, the compiled program computes it. To make an imported output true, false, or equal a value, mirror the imported file's companion test pattern by setting its underlying `#input.<fact>` and `#relation.<name>` keys.
- Never turn an imported derived rule into a fabricated `#input.<same_rule_name>` key. For example, use `us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds` or `not_holds`, not `us:statutes/7/2012/j#input.snap_household_has_elderly_or_disabled_member`.
- Do not invent `#input` keys for imported files. Use only the bare fact names that the imported file's formulas actually reference, or mirror the imported file's companion `.test.yaml` input pattern when it is supplied in context. If that imported output is driven by an upstream structural relation, set the upstream `#relation.<name>` rows used by the companion test instead of creating a local input under the imported file.
- A `#relation.<name>` input value must be a YAML list of row mappings. Never use a scalar row such as `- true`. Bad: `us:statutes/7/2012/j#relation.member_of_household: [- true]`. Good: `us:statutes/7/2012/j#relation.member_of_household:` followed by `- us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true`.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If a module defines both `Person` and `TaxUnit` outputs, create separate cases: `Person` cases set person facts at the top level and assert person outputs; `TaxUnit` cases use relation rows to supply person facts and assert only tax-unit outputs. Do not assert relation-child outputs in the parent entity's case.
- Use `holds` and `not_holds` for actual `dtype: Judgment` rule keys in test inputs and outputs; do not use YAML booleans for Judgment rule values.
- Use YAML booleans `true` and `false` for local factual `#input.<fact>` keys referenced directly by formulas.
- For proration tests with a source-stated denominator, choose input amounts divisible by that denominator so expected outputs are exact decimals, not rounded approximations. For example, if the denominator is 365, use a base amount like 36500 so `36500 * 182 / 365 = 18200`.
- Every test case for a local derived formula must assign every local factual
  `#input.<fact>` referenced by that formula, including facts that are false in
  the case. Missing false inputs make the executable test invalid.
- For every encoded `except`, `unless`, or `notwithstanding` carve-out, include
  companion tests for the positive path and the carve-out path so exclusions
  cannot be silently dropped.
- If a formula negates multiple exception predicates, include a separate companion test for each predicate that sets that exception input true and expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the same output rule where only the exception input changes from `false` to `true`; do not combine the exception test with another branch change. For example, an IRC section 24(h)(4)(B) noncitizen exception test must keep the same dependent/qualifying-child facts as its positive companion and flip only `noncitizen_exception_to_other_dependent_credit_applies`.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into one aggregate fact such as `sections_..._do_not_preclude...`. Encode or import each cited exception separately, then combine them in a helper if useful.
- If context files import this target file or reference this target file's outputs, preserve this file's public output names unless the source text proves the old interface was legally wrong. Do not rename an exported value just because a clearer friendly name is possible.
- For existing executable outputs in a copied target file, preserve the whole public executable surface for each retained output: local `name`, `kind`, `entity`, `dtype`, `period`, `unit`, `indexed_by`, and `versions[].effective_from`. Do not change the entity or period to a preferred modeling style when the existing file compiles. Never change an existing output from `dtype: Money` to `dtype: Judgment` just because the name sounds like an allowance/applicability decision.
- Preserve existing factual input slots referenced by copied formulas and companion tests. Do not swap a working local input surface for new friendly names or upstream abstractions unless the generated bundle performs a full, source-grounded surface migration.
- For repo-backed artifacts, every `input:` and `output:` key must be a canonical
  legal RuleSpec reference that resolves to an actual file and fragment; do not
  use bare friendly keys or absolute-looking placeholders.
- Do not add speculative future-period tests that rely on uprating or amendments not stated in `./source.txt`.
{proration_test_guidance.rstrip()}
{canonical_target_rule.rstrip()}
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
- Keep `.test.yaml` inputs oracle-comparable: prefer the oracle's direct component facts over inverted household proxy inputs, preserve direct component surfaces when available, and assert the canonical RuleSpec output whose local name is `{policyengine_rule_hint}` in every non-empty `output:` mapping.
- Prefer a contemporary monthly `.test.yaml` period like `2022-01` or `2024-01` when the source is current-effective and lacks a better effective date; avoid pre-2015 historical periods that PolicyEngine US cannot evaluate.
- If that output has a durable `jurisdiction:path#rule` id, key the test by that id rather than the friendly local name.
- Key inputs by their resolving legal RuleSpec target too, e.g. `jurisdiction:path#input.fact`, `jurisdiction:path#relation.name`, or `jurisdiction:path#upstream_rule`.
- If a copied downstream output with the oracle hint's local name is available, assert that canonical copied output rather than replacing it with a helper-only local test.
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
    corpus_source_section = ""
    corpus_rulespec_requirement = ""
    if corpus_citation_path:
        corpus_source_section = f"""
- This source text was read from `corpus.provisions` at `{corpus_citation_path}`.
"""
        corpus_rulespec_requirement = f"""
- Include `module.source_verification.corpus_citation_path: {corpus_citation_path}` exactly.
"""

    return f"""You are participating in an encoding eval for {citation}.

Author the output in Axiom RuleSpec YAML.

Primary legal authority:
- `./source.txt` contains the complete source text for this target source unit.
- Treat that source text as the only source of legal truth for this artifact.
{corpus_source_section.rstrip()}
{inline_source}
{source_metadata_section}{context_section}{missing_cited_source_section}
{backend_section}

RuleSpec requirements:
- The RuleSpec file must begin with `format: rulespec/v1`.
- Include `module.summary: |-` containing the exact operative source text or an exact compact excerpt sufficient to audit all encoded rules.
- Do not emit `source_url`; RuleSpec source verification reads `corpus.provisions`, not raw PDFs or web pages.
{corpus_rulespec_requirement.rstrip()}
- Include `module.proof_validation.required: true` and add
  `metadata.proof.atoms` to every `parameter` and `derived` rule. Each atom
  must point to the corpus source, an accepted claim, or an explicit imported
  RuleSpec export supporting that rule's formula/value.
- For imported proof support, put `import:` at the proof atom top level
  (for example `kind: import` plus `import.target: us:statutes/...#symbol`);
  do not put imported RuleSpec targets under `source:`. Import proof atoms must
  include `import.target`, `import.output`, and `import.hash` with the listed
  context file `sha256:` hash. If no `sha256:` hash is listed for that import,
  do not emit an import proof atom. When the imported proof target is in the
  same RuleSpec file, use `hash: sha256:local`; never use `sha256:self`.
- Proof atom `kind` must be one of: `amount`, `condition`, `definition`,
  `default`, `effective_period`, `exception`, `formula`, `import`, `ordering`,
  `parameter`, `parameter_table`, `predicate`, `table_cell`, or `unit`.
- Use `rules:` as a list of rule objects. The filepath is the ID; do not add an `id:` field.
- Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`.
- Rule kinds are `parameter`, `derived`, `data_relation`, or `source_relation`. Use `parameter` for named source scalars, `derived` for entity-scoped outputs, `data_relation` for runtime predicates, and `source_relation` for non-executable legal/provenance edges.
- A `kind: table_cell` proof atom must include `source.table.header`, `source.table.row`, and `source.table.column`. A `kind: parameter_table` proof atom with `source.table` must include `source.table.header`, `source.table.row_key`, and `source.table.column_key`; header-only `parameter_table` proof atoms are invalid. Example: `source: {{table: {{header: "credit percentage table", row_key: "qualifying_child_count", column_key: "credit_percentage"}}}}`. If you cannot identify table coordinates, use a direct proof kind such as `amount`, `parameter`, or `formula` instead of `table_cell` or `parameter_table`.
- Every executable `parameter` and `derived` rule must include a `source:`
  field with the legal citation/span that directly supports that rule. Keep
  `source:` short and local to the rule; use `module.source_verification` for
  the corpus locator.
- If `./source.txt` is a broad application, furnishing, administrative duty, or purpose clause without a computable policy condition, preserve it in `module.summary` but do not create an executable derived output just to paraphrase it. Encode only the concrete conditions, exceptions, parameters, and relations that affect computation.
- Do not create an output for administrative clauses like "assistance shall be furnished to all eligible households who make application." Unless the source defines a calculable benefit, amount, condition, or exception, keep that text documentary in `module.summary`.
- Do not encode a pure pass-through rule whose formula is only one local fact. If the source only names a preexisting fact without changing it, reference the upstream rule when available or leave the phrase documentary.
- If a copied child-fragment file encodes a limitation, branch, amount, or
  predicate needed by the requested parent provision, import the child output
  and compose it. Do not copy the child formula or its factual inputs into the
  parent file. For example, IRC section 63(c) should import
  `us:statutes/26/63/c/5#dependent_standard_deduction` rather than reconstruct
  the dependent earned-income limitation in `c.yaml`.
- Do not create standalone small-number parameters just to restate prose such as "one-time" or "more than one consecutive month" when the number only qualifies a local factual condition. Encode the whole source-stated condition as a fact predicate or derived condition unless the scalar is an independent reusable amount, rate, threshold, cap, or limit.
- Do not append citation or file suffixes like `_2014_a` to new local rule names; the file path is already the legal ID. Keep names concise and semantic unless a copied public interface must be preserved.
- Rule names ending in the current path fragments, such as `_2_C`, `_b_1`,
  `_d_2_C`, or `_2014_a`, are invalid.
- If an existing copied output name violates the no-citation/path-suffix rule,
  do not preserve it. Rename it to a concise semantic name and update the
  companion tests.
- Rule names must not collide with copied sibling files. For subparagraph/list
  item child files, make the principal output name semantic to that branch
  (for example `care_responsibility_exemption_applies`), not only the shared
  parent consequence like `person_exempt_from_paragraph_1_work_requirements`.
- When a child provision substitutes, increases, caps, or otherwise modifies a
  sibling or parent output, give the replacement a branch-specific name such as
  `_under_subsection_h`, `_after_2017`, or another source-stated modifier. For
  IRC section 24(h), do not reuse sibling 24(d) names like
  `ctc_refundable_phase_in_threshold`; use a subsection-h-specific name such as
  `ctc_refundable_phase_in_threshold_under_subsection_h`.
- Choose structural relations at the narrow legal subject stated by the source.
  If the source grants an amount to the taxpayer, spouse, claimant, child, or
  other role-limited person, do not aggregate over a broader household/tax-unit
  relation unless the source says every member counts. Name the relation for the
  role set that is legally counted, such as `taxpayer_or_spouse`, not merely for
  the container entity. If a copied relation is legally too broad for the
  requested source, rename it; relation names are not stable public outputs.
  Never preserve or create `*_member_of_tax_unit` or `member_of_tax_unit` for a
  source that counts only the taxpayer, spouse, qualified individual, claimant,
  child, or dependent. For IRC section 22, count qualified individuals over a
  relation like `taxpayer_or_spouse_of_tax_unit`, not
  `elderly_disabled_member_of_tax_unit`.
- For child tax credit, dependent credit, or any source that says "qualifying
  child", "dependent of the taxpayer", or "with respect to such child", do not
  use `member_of_tax_unit`. Define a role-scoped relation such as
  `dependent_of_tax_unit`, `qualifying_child_of_tax_unit`, or
  `child_or_dependent_of_tax_unit`, and aggregate over that relation. For IRC
  section 24(h), count `ctc_qualifying_child` and `ctc_other_dependent` over a
  dependent/child relation, not over `member_of_tax_unit`.
- If a generic role-scoped relation name is already exported by a copied
  sibling file, do not reuse it. Make the relation source-specific, such as
  `ctc_qualifying_child_of_tax_unit` for section 24 rather than a sibling's
  `qualifying_child_of_tax_unit`.
- If the source computes an amount by reference to an entitlement, status,
  amount, or test "under" another section, subsection, paragraph, regulation, or
  document, do not inline that cross-reference's mechanics into this file unless
  that cross-referenced source text is included and this file is the canonical
  home for those mechanics. Import the existing RuleSpec target when present. If
  the cross-reference is not yet encoded, expose a semantic input/count named
  for the cross-reference itself, such as
  `additional_standard_deduction_entitlement_count_under_subsection_f`, rather
  than inventing the cross-referenced age, blindness, household, or membership
  tests locally. For example, IRC section 63(c)(3) should not count
  `is_aged_65_or_over` or `is_blind` over `member_of_tax_unit`; those are
  subsection 63(f) mechanics.
- When an unencoded cross-reference must be represented as a semantic local
  input, name it after the legal status with an `_under_section_<section>` or
  `_under_subsection_<subsection>` suffix. Do not start a local input with
  `section_<section>_` or `subsection_<subsection>_`; those names are reserved
  for imported legal outputs and will be treated as missing imports.
  Cross-reference local inputs such as `_under_section_<section>`,
  `_provided_in_section_<section>`, `_allowed_under_section_<section>`,
  `_deduction_under_section_<section>`, or `_credit_allowed_under_section_<section>`
  are only allowed for non-exception factual interfaces when the cited source is
  not available as RuleSpec. If the citation appears in definition,
  same-meaning, treated-as, rules-similar, exception, exclusion, `unless`,
  `notwithstanding`, shall-not-apply, or not-treated-as logic and the cited
  source is unavailable, emit `module.status: deferred` or
  `module.status: entity_not_supported` with `rules: []` instead of inventing a
  local cross-reference fact. If that section is present in repo context, import
  it and use its exported output instead.
- When a copied context file encodes a cited upstream source on a different
  entity, import that upstream output and bridge entities with a structural
  relation instead of replacing the import with a local cross-reference amount.
  For example, if IRC section 22 excludes amounts described in section
  104(a)(4), import
  `us:statutes/26/104/a/4#service_injury_pension_excluded_amount` and aggregate
  it over a TaxUnit-to-Payment relation; do not create local inputs named
  `section_104_a_4_amounts` or `section_104_a_4_veterans_affairs_benefits`.
- Do not encode simple unary factual inputs as `kind: data_relation` rules. If a formula needs a local true/false fact, reference a descriptive bare fact name in the formula and put that fact in tests as `{target_ref_prefix + "#input.<fact>" if target_ref_prefix else "<jurisdiction>:<path>#input.<fact>"}`.
- Use `kind: data_relation` only for structural runtime predicates with explicit `data_relation.predicate`, `data_relation.arity`, and `data_relation.arguments`.
- If the requested source text includes a limitation, cap, exception, or
  cross-referenced subparagraph that changes the final exported amount, the
  final exported amount must apply that limitation. If a copied sibling/context
  file already encodes the limitation, import it and compose with it instead of
  duplicating or ignoring it.
- If the copied target file is already executable, do not replace it with
  `module.status: deferred` merely because upstream cross-references are not
  fully encoded yet. Preserve the executable public surface and improve the
  source-faithful formulas/tests; only defer an already executable target when
  the existing executable surface is legally impossible to preserve.
- If the requested source itself defines a legal status or test through
  relationship, age, abode/residence, support, filing, income, or tie-breaker
  conditions, encode those conditions as executable predicates with boundary
  inputs for facts not defined in the source. Do not emit
  `module.status: deferred` merely because some facts must be supplied by the
  caller or because tie-breaker facts require relation inputs. For example, IRC
  section 152(c) should export qualifying-child predicates over a
  taxpayer-child relation with inputs for relationship, abode, age/student or
  disability, support, joint-return, and competing-claimant facts.
- If the requested source defines an exclusion, inclusion, deduction, or credit
  amount but depends on externally determined facts such as executive-order
  designations, military status, hospitalization, missing status, monthly pay
  grades, or source-document classifications, encode the amount with boundary
  inputs for those facts instead of deferring. For example, IRC section 112
  should export an executable amount excluded from gross income by reason of
  section 112; it should not be `module.status: deferred` solely because combat
  zone designation or military pay facts are supplied by the caller.
- Hard requirement for IRC section 112: do not emit `module.status: deferred`.
  Export `amount_excluded_from_gross_income_by_reason_of_section_112` as an
  executable TaxUnit/Year Money output, using boundary inputs for qualifying annual
  compensation, commissioned-officer status, maximum enlisted amount,
  combat-zone service, hospitalization, and Vietnam missing-status facts.
  Do not create Person helper outputs or a `data_relation` aggregate for this
  Section 112 encoding; use direct TaxUnit/Year annual input amounts and
  conditions so downstream tax-unit rules can import the result directly.
- Importing a child rate or threshold is not enough when the child file already
  exports the executable tax, benefit, deduction, or eligibility result. For
  aggregate parent sections, import the child result output itself and sum,
  cap, select, or otherwise compose those imported results. Do not recompute a
  child result locally from the child rate and the child factual inputs.
- Do not create parallel statutory-dollar executable parameters when a copied
  current-year authority already provides the applicable inflation-adjusted
  parameter. Import the current-year authority unless the task is to encode the
  inflation adjustment formula itself.
- If a copied current-year authority exports the same concept or output name
  that the requested statute formula would otherwise create, do not emit a
  local executable duplicate with that name. Import and use the current-year
  authority's output, keeping only statute-specific conditions or non-executable
  `source_relation` records in the statute file. For IRC section 63(c)(5), if
  Rev. Proc. context already exports `dependent_standard_deduction_limit`, do
  not recreate it in the statute file.
- If a current-year authority provides a directly rounded final amount table,
  use that table for the final amount instead of recomputing the amount from
  related rates and thresholds. For example, if an IRS revenue procedure exports
  an EITC maximum-credit table, `eitc_maximum` must select that imported maximum
  table, not multiply the phase-in rate by the earned-income amount and keep an
  unrounded decimal.
- When IRC section 32(c)(2) uses "net earnings from self-employment (within
  the meaning of section 1402(a))" and then says those net earnings are
  determined with regard to Section 164(f), do not import Section 1402(a)'s
  final `net_earnings_from_self_employment` output. Section 1402(a)(12)
  substitutes a rate-based deduction in lieu of Section 164(f). For Section
  32(c)(2), create a local self-employment component from Section 1402(a)'s
  pre-paragraph-12 net earnings minus the imported Section 164(f) deduction.
- When source text says an exemption, exclusion, or adjustment applies
  `to the extent` of an amount, do not model it as all-or-nothing zeroing such as
  `if exempt_amount > 0: 0 else: tax`. Subtract or apportion the stated amount.
  If imported child calculations cannot receive the adjusted basis faithfully
  under the current executable schema, emit `module.status: entity_not_supported`
  or `deferred` instead of an approximate executable formula.
- Do not repair that case by importing child rates or thresholds and rebuilding
  the child branch locally with an adjusted basis. That still re-encodes the
  child branch and is invalid unless the schema can explicitly wire the
  adjusted basis into the imported child result.
- When the statute states pre-inflation base dollars that a current-year
  authority adjusts, any local statute output must be named as a statutory/base
  concept, not as the current-year value. For IRC section 63(c)(5), use a name
  like `dependent_basic_standard_deduction_statutory_limit`, not
  `dependent_standard_deduction_limit`.
- When the source rounds an inflation or cost-of-living increase, round the
  increase before adding it to the base amount unless the source explicitly
  says to round the final total. Companion tests must assert the rounded
  increase plus the base, not the unrounded total. For example, with base
  15750, adjustment 0.1, and a next-lower $50 multiple, the increase is 1550
  and the total is 17300, not 17325.
- Do not invent new entities, periods, or dtypes.
- Allowed `entity:` values are {", ".join(f"`{entity}`" for entity in SUPPORTED_EVAL_ENTITIES)}.
- Allowed `period:` values are {", ".join(f"`{period}`" for period in SUPPORTED_EVAL_PERIODS)}.
- Allowed `dtype:` values are {", ".join(f"`{dtype}`" for dtype in SUPPORTED_EVAL_DTYPES)}, or `Enum[Name]`.
- Use `dtype: Judgment`, not `dtype: Boolean`, for legal eligibility, availability, applicability, entitlement, and other holds/not-holds style outputs, especially when the formula contains `not`.
- Do not create derived `dtype: Boolean` helper rules with logical formulas. Use `dtype: Judgment` for derived legal predicates, or leave simple local facts as factual `{target_ref_prefix + "#input.<fact>" if target_ref_prefix else "<jurisdiction>:<path>#input.<fact>"}` keys consumed by formulas and tests.
- Use `unit: USD`, `unit: GBP`, or another explicit unit for money outputs when the source states a currency.
- Put each rule's formulas under `versions: - effective_from: 'YYYY-MM-DD'` and `formula: |-`.
- Do not encode legal effective dates as `dtype: String` parameters or date
  literal formulas such as `2025-01-01`. Axiom formulas have no date literal type.
  Use `effective_from` metadata for version timing, or use a
  source-stated semantic boolean predicate when a date window is a runtime
  condition. Do not put the date or year value in the fact name; use names like
  `taxable_year_begins_after_termination_date` or
  `taxable_year_is_in_temporary_effective_window`, not
  `taxable_year_begins_after_2024_and_before_2029` or
  `taxable_year_begins_after_december_31_2021`.
- Do not emit more than one `versions:` entry for `kind: derived`; the runtime does not yet support period-selecting versioned formulas. Use a single source-faithful conditional formula when the provision itself defines a temporal branch, or encode only the currently applicable provision after resolving the source context.
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==` for equality, `and`/`or` for booleans, decimal ratios for percentages, and no Python inline ternary syntax.
- Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and `ceil(x)`. Do not use Python-only functions such as `round(...)`; express nearest-multiple rounding as `floor((x / multiple) + 0.5) * multiple` for nonnegative amounts.
- Benefit, allotment, credit, deduction, allowance, and subsidy formulas must never emit negative money. When subtracting an income, contribution, or other reduction from a maximum amount, floor the result with `max(0, ...)` before applying downstream minimum-benefit or issuance branches. When a nonnegative credit, deduction, allowance, subsidy, or benefit is a percentage of `min(income, cap)` or similar, floor the income base at zero: use `rate * min(max(0, earned_income), cap)`, not `rate * min(earned_income, cap)`.
- Outputs named `taxable_income` or ending in `_taxable_income` must also never be negative. Wrap the final selected branch at zero, including both sides of conditionals: use `if condition: max(0, branch_a) else: max(0, branch_b)`, not `if condition: branch_a else: branch_b`.
- If that reduction has rounding alternatives, every branch must be floored: use `if round_up: max(0, maximum - ceil(reduction)) else: max(0, floor(maximum - reduction))`, never `if round_up: maximum - ceil(reduction) else: floor(maximum - reduction)`.
- US tax filing status is a derived legal classification, not a downstream
  boundary fact. Do not create local `#input.filing_status` facts in a rule or
  test. Encode the upstream filing-status source first, then import its absolute
  RuleSpec output into downstream threshold, phaseout, deduction, and credit
  rules. If an already-encoded upstream filing-status output is unavailable,
  stop and encode that upstream source rather than synthesizing a local input.
  Do not preserve existing `#input.filing_status` or `#input.tax_filing_status`
  surfaces from copied target files; migrate them to upstream imports or
  source-backed non-status leaf facts such as whether a joint or separate return
  was actually made.
- When source text says a person is `entitled to a deduction under section 151`
  or that a section 151 deduction is `allowed` or `allowable`, do not use the
  monetary `us:statutes/26/151#section_151_exemption_deduction` amount as a
  proxy. The post-2017 exemption amount can be zero while entitlement still
  matters. Import a source-backed eligibility/judgment output such as
  `us:statutes/26/151#exemption_individual_eligible`, or encode the missing
  upstream 151 entitlement predicate first.
- Hard requirement for IRC sections 2, 6013, and 7703, 5 USC section 5566,
  and 37 USC section 556: do not emit
  `module.status: deferred` or `module.status: entity_not_supported`. These
  sources are the upstream filing-status source chain. Encode executable legal
  predicates with boundary facts for marital, household, abode, support, death,
  separation, and return-election facts not defined in the source.
- For IRC section 6013(a), expose a source-backed joint-return eligibility
  output before applying subsection (a)(3)'s decedent-return-maker limitation,
  and then a final output that applies (a)(3). Downstream IRC section 2(a)(2)(B)
  says "without regard to subsection (a)(3)", so it must be able to import the
  pre-(a)(3) output instead of using a local `under_section_6013` placeholder.
- For IRC section 151 repairs, preserve existing output IDs exactly while
  removing filing-status inputs. In particular, do not rename
  `senior_deduction_base_amount`, `senior_deduction_phaseout_threshold_other`,
  `senior_deduction_phaseout_threshold_joint`,
  `senior_deduction_phaseout_threshold`,
  `senior_deduction_amount_per_qualified_individual`,
  `senior_deduction_eligible`, or `exemption_amount`.
- The shared US tax filing-status output remains a structural enum: 0 single,
  1 joint return, 2 married filing separately, 3 head of household, and
  4 surviving spouse / qualifying widow(er). Never encode US tax filing status
  as string literals such as `"married_filing_jointly"` or as separate boolean
  facts such as `married_filing_jointly`, `head_of_household`, or
  `surviving_spouse`. Use the imported numeric filing-status output in formulas,
  e.g. `match filing_status: 1 => joint_amount; 4 => joint_amount; ...`. If the
  source groups surviving spouse with joint return, every branch or match that
  handles status 1 must also handle status 4 in that same branch with the same
  result.
- Supported relation aggregators are `len(relation)`,
  `count_where(relation, predicate_fact)`, `sum(relation.amount_fact)`, and
  `sum_where(relation, amount_fact_or_derived, predicate_fact)`. Do not write
  `sum(relation, expression)` or put arithmetic inside a relation field access.
  Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact
  supplied directly on each relation row. Do not use `sum(relation.local_output)`
  for a `parameter` or `derived` rule defined in the same file; for a computed
  per-related-entity amount, write
  `sum_where(relation, local_output, source_stated_predicate_fact)` instead.
  To count two boolean conditions over the same relation, write two
  `count_where(...)` calls and add them.
- If a conditional is embedded inside arithmetic or another larger expression, wrap the whole conditional in parentheses, such as `amount + (if condition: extra else: 0)`. Do not write `amount + if condition: extra else: 0`.
- Formula strings must use bare identifiers only. If an imported rule is listed
  as `us:statutes/...#example_rule`, add that exact target to `imports:` but
  reference `example_rule` inside formula text.
- Axiom conditionals are expression syntax, not YAML syntax. Money/scalar formulas may use `if condition: value else: other`; do not use Python ternary syntax.
- `dtype: Judgment` formulas must not use `if ... else ...`. Write them as boolean expressions using `and`, `or`, `not`, comparisons, and parentheses. For example, encode `if exempt: net_ok else: net_ok and gross_ok` as `net_ok and (exempt or gross_ok)`.
- When using negated conjuncts, write them as a multiline formula with each `not <predicate>` term on its own line joined by `and`, rather than one compact `not A and not B` line.
- Do not use Python inline ternaries like `x if cond else y`.
- Use chained `if condition: value else: other_value` expressions; do not use YAML-style `if:` / `then:` / `else:` blocks.
- Do not append a multiline conditional directly onto another expression, and do not use inline assignment syntax like `:=` inside formula blocks.
- For `dtype: Rate`, encode percentages as decimal ratios like `0.60` or `0.40`, never as `%` literals.
- Do not simplify source-stated ratios or fractions into new decimal literals.
  If the source states `20/200`, encode grounded numerator and denominator
  parameters and compare with `20 / 200` or with those named parameters; do not
  emit an ungrounded decimal such as `0.10`.
- Use concrete ISO calendar dates like `2025-03-21` for day-level tests; do not use ISO week strings like `2025-W13`.
- Any substantive numeric literal in a formula must either appear in `./source.txt` or be one of -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by a named scalar definition in RuleSpec when it is a legal amount, rate, threshold, cap, or limit.
- If you encode a substantive numeric literal, `module.summary` or the rule's proof excerpt
  must include the exact source phrase containing that number. Do not omit a
  subsection, table row, or clause that grounds an encoded
  numeric amount, rate, threshold, cap, or limit.
- Represent every substantive source amount, rate, threshold, cap, or limit as a named `parameter` rule, then reference that parameter from derived formulas.
- If the same numeric value appears twice in materially different legal roles, including separate numbered exceptions or subparagraphs, give those roles distinct named scalars; otherwise reuse that named scalar everywhere the rule compares against or computes with that number.
- Adjacent bracket thresholds repeated as both an upper bound and the next bracket's lower bound are separate source-stated legal roles; define distinct semantic scalars for those occurrences and use them in the branch conditions.
- If a formula negates multiple exception predicates, include a separate companion test for each predicate that sets that exception input true and expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the same output rule where only the exception input changes from `false` to `true`; do not combine the exception test with another branch change. For example, an IRC section 24(h)(4)(B) noncitizen exception test must keep the same dependent/qualifying-child facts as its positive companion and flip only `noncitizen_exception_to_other_dependent_credit_applies`.
- Every local executable `kind: parameter` and `kind: derived` rule must appear
  at least once under an `output:` block in the companion `.test.yaml`; do not
  leave scalar parameters, helper parameters, or helper derived rules
  unasserted.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If
  a module defines both `Person` and `TaxUnit` outputs, create separate cases:
  `Person` cases set person facts at the top level and assert person outputs;
  `TaxUnit` cases use relation rows to supply person facts and assert only
  tax-unit outputs. Do not assert relation-child outputs in the parent entity's
  case.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into one aggregate fact such as `sections_..._do_not_preclude...`. Encode or import each cited exception separately, then combine them in a helper if useful.
- If `./source.txt` says someone is "aged 18 or over", "under 25", or similar, model the legal age predicate instead of inventing documentary age constants.
- When source text uses amendment markup like `[old] new`, treat the bracketed value as superseded text. Encode the current unbracketed value/effective date unless the task explicitly asks for historical text.
- If `./source.txt` makes an allowance, deduction, exemption, or eligibility branch conditional on billed, paid, incurred, anticipated, or other cost/expense facts, encode a positive fact predicate for that source-stated condition. Do not model availability solely as `not` other categories.
- When the cost/expense fact only matters after exclusion predicates, exported amount/quantity formulas consumed by dependent modules must guard the exclusions before referencing the branch-specific fact, so excluded cases do not require that fact as an input. For example, the amount should use `if other_allowance_eligible: 0 else: if household_has_telephone_cost: amount else: 0` rather than `if telephone_eligible: amount else: 0` when `telephone_eligible` itself references the branch-specific telephone-cost input.
- Phrases like `consists of the cost for X` or `available to households with X costs` require a positive fact for that cost/service. For example, a telephone allowance must depend on a fact for the household having or incurring the basic telephone-service cost before applying exclusions for other allowances.
- In a jurisdiction-specific repo, phrases like `residing in New York State` usually describe the document's scope, not a new input variable. Do not add a state-residency input unless the provision itself is encoding a residency eligibility test.
- If an encoded child paragraph depends on an operative parent condition, include the parent condition in `module.summary` and include both child and parent corpus paths under `module.source_verification.corpus_citation_paths` when those corpus paths are available.
- Do not create scalar variables for citation numbers, paragraph numbers, branch numbers, or source line labels.
- Do not invent `dtype: String` variables just to restate the effective date.
- Do not decompose legal dates into numeric `year`, `month`, or `day` scalar variables.
- Do not create named `parameter` rules for structural table row labels, household-size row indexes, or branch numbers unless the source actually sets that value as a legal amount, rate, threshold, cap, or limit; use those structural comparisons inline instead.
- If the source cannot be represented faithfully with the supported schema, emit `module.status: deferred` or `module.status: entity_not_supported` with `rules: []`; do not invent unsupported ontology.
- Never emit `rules: []` without an explicit non-executable `module.status`. If the source has operative text, encode at least one source-backed rule instead of silently returning an empty module.
- For deferred or entity-not-supported artifacts, leave the companion `.test.yaml` empty and do not create assertions against deferred symbols.
- If metadata or context names an absolute canonical target that this source `sets`, `amends`, `implements`, or `restates`, add a separate `kind: source_relation` record with `source_relation.type` and `source_relation.target`. Do not put source graph edges in executable rule metadata.
- Preserve existing or copied `kind: source_relation` records unless `./source.txt` proves the legal/provenance edge is wrong.
- For state-set standards, allowances, thresholds, or options implementing federal delegation, include `source_relation.value` pointing to the local executable RuleSpec output and `source_relation.basis.delegation` when context identifies the upstream delegated slot.
- When the source says a value is determined `in accordance with section X`, emit the upstream import instead of restating the concept locally when that import target is available.
- When the source uses `except`, `unless`, or `notwithstanding` with cited
  sections or same-section subsections, do not create local `section_...` or
  `subsection_...` inputs for those cited sources. Import the cited RuleSpec
  source when it exists; if the target source is needed but unavailable, stop
  with an explicit missing-upstream/dependency request instead of encoding an
  opaque placeholder.
- If the cited same-section subsection is supplied in context as a RuleSpec file, add an `imports:` entry for that file and reference its exported rule; do not summarize the cited subsection into a local fact like `person_meets_...requirements`.
- Do not copy the body of a cited cross-reference provision into this module's `summary` or re-encode that cited provision locally. Keep this module scoped to the requested citation and import the cited provision instead.
- Do not fabricate sibling-file imports, do not guess unavailable import targets, and do not invent `import` statements or `imports:` blocks for uncopied same-instrument provisions.
- Before finalizing, do this self-check:
  1. Numeric inventory: every source-stated legal amount, rate, threshold, cap,
     or limit has a named `parameter`, and derived formulas reference the name
     rather than an inline literal.
  2. Test input inventory: for every local factual identifier referenced by a
     local derived formula, every companion test case assigns the corresponding
     `#input.<fact>` explicitly, including false facts. Do not rely on implicit
     defaults. If a test asserts an indexed `parameter` table output directly,
     the test must assign every `indexed_by` key as `#input.<key>`; otherwise
     assert the derived lookup output instead of the raw table. In ordinary
     end-to-end tests, do not output raw indexed parameter tables at all.
     For imported modules, only assign imported `#input` or `#relation` keys
     that exist in the current imported RuleSpec context. Do not preserve stale
     imported test inputs from copied files. Do not stub imported derived
     outputs as test inputs; imported programs are computed. If the downstream
     rule depends on an imported output, assign all current upstream factual
     inputs and relations needed by that imported output, including false facts.
  3. Proof inventory: every proof atom uses only an allowed `kind`; imported
     proof atoms include `import.target`, `import.output`, and `import.hash`;
     textual claim support is either direct corpus source support or a claim ID
     listed under `module.source_claims`.
  4. Import inventory: every `imports:` entry is an exact copied/importable
     RuleSpec target. Top-level `imports:` entries must be scalar strings; never
     map entries like `- target:` plus `symbols:`. Do not guess sibling paths; if
     required upstream context is missing, emit a typed missing-upstream/dependency
     request instead.
{target_hint}
Additional encoding guidance:
{additional_guidance}

Minimal RuleSpec shape:
```yaml
format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: <corpus citation path from this prompt>
  summary: |-
    <exact source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    source: <legal citation/span>
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: <corpus citation path from this prompt>
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
    source: <legal citation/span>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          if example_condition: example_amount else: 0
```

{output_rules}
Do not respond with summaries, markdown prose, or file-write confirmations.
"""


def _workspace_corpus_citation_path(workspace: EvalWorkspace) -> str | None:
    source_metadata = workspace.source_metadata
    if not isinstance(source_metadata, dict):
        return None
    raw_value = source_metadata.get("corpus_citation_path")
    if not isinstance(raw_value, str):
        return None
    citation_path = raw_value.strip()
    return citation_path or None


def _build_eval_prompt(
    citation: str,
    mode: EvalMode,
    workspace: EvalWorkspace,
    context_files: list[EvalContextFile],
    target_file_name: str,
    target_ref_prefix: str | None = None,
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
        target_ref_prefix=target_ref_prefix,
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
    kind = f" (kind: {item.kind})"
    context_hash = _context_file_hash(item.source_path)
    hash_detail = f"; context hash `{context_hash}`" if context_hash else ""
    export_detail = _context_file_export_detail(item)
    if item.workspace_path == item.import_path:
        return f"- `{item.workspace_path}`{hash_detail}{export_detail}{details}{kind}"
    return (
        f"- inspect `{item.workspace_path}`; import target `{item.import_path}`"
        f"{hash_detail}{export_detail}{details}{kind}"
    )


def _format_partial_extent_child_schema_limit_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
    *,
    target_ref_prefix: str | None,
) -> str:
    """Return target-specific guidance for unsupported partial child rewiring."""
    if not target_ref_prefix:
        return ""
    if not re.search(r"\bto\s+the\s+extent\b", source_text, flags=re.IGNORECASE):
        return ""
    child_prefix = f"{target_ref_prefix.rstrip('/')}/"
    child_terminal_refs: list[str] = []
    for item in context_files:
        if not item.import_path.startswith(child_prefix):
            continue
        for export in _context_file_terminal_exports(item.source_path):
            child_terminal_refs.append(f"{item.import_path}#{export}")
    if not child_terminal_refs:
        return ""

    refs = ", ".join(f"`{ref}`" for ref in child_terminal_refs[:6])
    if len(child_terminal_refs) > 6:
        refs += ", ..."
    return f"""
Target-specific schema limit:
- `./source.txt` uses `to the extent` and copied child-fragment files already
  export executable results ({refs}). Under the current executable schema, this
  parent cannot faithfully recompute those child results using a locally
  adjusted basis, and it cannot wire that adjusted basis into imported child
  results. Emit `module.status: entity_not_supported` or `deferred` with
  `rules: []` and an empty companion test file. Do not create a
  `*_before_exemption` executable output or an adjusted local wage/base helper
  for this parent.
"""


def _format_branch_child_naming_guidance(
    context_files: list[EvalContextFile],
    *,
    target_file_name: str,
    target_ref_prefix: str | None,
) -> str:
    """Warn child-fragment encoders about reserved sibling export names."""
    target_key = _context_import_path_key(target_ref_prefix or target_file_name)
    if target_key is None:
        return ""
    target_parent = _context_import_parent_key(target_key)

    target_exports: set[str] = set()
    sibling_exports: dict[str, str] = {}
    sibling_count = 0
    for item in context_files:
        item_key = _context_import_path_key(item.import_path)
        if item_key is None or _context_import_parent_key(item_key) != target_parent:
            continue
        exports = _context_file_exports(item.source_path)
        if not exports:
            continue
        if item_key == target_key:
            target_exports.update(exports)
            continue
        sibling_count += 1
        for name in exports:
            sibling_exports.setdefault(name, item.import_path)

    if not sibling_exports or sibling_count == 0:
        return ""

    reserved = _format_rule_name_list(sorted(sibling_exports))
    colliding_exports = target_exports & sibling_exports.keys()
    colliding = _format_rule_name_list(sorted(colliding_exports))
    collision_note = ""
    if colliding_exports:
        collision_note = (
            "\n- The copied target currently exports invalid colliding names: "
            f"{colliding}; do not preserve those names."
        )

    return f"""
Sibling export naming for this target:
- Copied sibling files already reserve these exported names: {reserved}.
{collision_note}
- Do not export any local rule with a copied sibling's name. If a suggested
  generic relation name is already reserved, use a source-specific semantic
  name, such as `ctc_qualifying_child_of_tax_unit` for section 24 rather than
  `qualifying_child_of_tax_unit`.
- If this target is a child branch and the source states a shared parent
  consequence such as "shall be exempt if (A) ...", define the condition in this
  branch, not the shared parent consequence. Use a concise semantic output
  name based on this branch's condition.
- If the copied target exports a name that mainly describes the shared parent outcome rather than this branch's source-stated condition, treat that name as stale and rename it even when no sibling currently collides with it.
"""


def _format_cited_context_import_guidance(
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Highlight copied context files directly cited by the source text."""
    lines: list[str] = []
    for item in context_files:
        citation = _import_target_to_statute_citation(item.import_path)
        if citation is None:
            continue
        if not _source_text_cites_statute(source_text, citation):
            continue
        exports = _context_file_exports(item.source_path)
        if not exports:
            continue
        references = ", ".join(
            f"`{item.import_path}#{name}`" for name in exports[:8]
        )
        if len(exports) > 8:
            references += ", ..."
        preferred_exports = _preferred_exports_for_cited_reference(
            source_text,
            citation,
            exports,
        )
        preferred_note = ""
        if preferred_exports:
            preferred = ", ".join(
                f"`{item.import_path}#{name}`" for name in preferred_exports
            )
            preferred_note = (
                " For the cited deduction/exemption/credit reference, prefer "
                f"the final imported output {preferred} over any local "
                "`*_provided_in_section_*`, `*_under_section_*`, or similar "
                "placeholder."
            )
        lines.append(
            f"- Source cites `{citation.label}`; copied context target "
            f"`{item.import_path}` exports {references}.{preferred_note}"
        )
    if not lines:
        return ""
    return f"""
Mandatory cited RuleSpec imports detected from source text:
{chr(10).join(lines)}
When this source computes by reference to one of these cited targets, add the
appropriate listed `imports:` entry and use the imported bare rule name in the
formula. Do not keep a local `_under_section_...` or `_under_subsection_...`
input, or a local `_provided_in_section_...`, `_allowed_under_section_...`,
`_deduction_under_section_...`, or `_credit_allowed_under_section_...` input,
for an already copied RuleSpec context target.
"""


def _format_missing_cited_source_guidance(
    citation: str,
    source_text: str,
    context_files: list[EvalContextFile],
) -> str:
    """Warn when exception-like logic cites upstream RuleSpec not in context."""
    if not _source_text_has_cross_reference_dependency(source_text):
        return ""

    missing_targets = _missing_cited_statute_targets(
        citation,
        source_text,
        context_files,
    )
    if not missing_targets:
        return ""

    lines = [
        f"- Source cites `{label}`, but no copied context provides `{target}`."
        for label, target in missing_targets
    ]
    example_suffix = _citation_example_suffix(missing_targets[0][1])
    return f"""
Missing cited RuleSpec sources detected:
{chr(10).join(lines)}
For definition, same-meaning, treated-as, rules-similar, exception, exclusion,
`unless`, `notwithstanding`, shall-not-apply, not-treated-as,
carryback/carryover, or special-rule logic, these citations are upstream legal
dependencies. Do not create local facts such as
`section_{example_suffix}...`, `transaction_to_which_section_{example_suffix}_applies`,
`*_under_section_{example_suffix}`, or `*_provided_in_section_{example_suffix}`.
If an executable output would depend on any missing target above, emit
`module.status: deferred` or `module.status: entity_not_supported` with
`rules: []`, preserve the source text in `module.summary`, and leave the
companion `.test.yaml` empty. Encode the upstream cited source first, then retry
this provision.
"""


def _source_text_has_cross_reference_dependency(source_text: str) -> bool:
    """Return whether text has dependency phrasing that should not be stubbed."""
    return bool(
        re.search(
            r"\b(?:except|unless|notwithstanding)\b"
            r"|shall\s+not\s+apply"
            r"|not\s+be\s+treated"
            r"|carrybacks?\s+and\s+carryovers?\s+under\s+section"
            r"|tax\s+imposed\s+by\s+section",
            source_text,
            flags=re.IGNORECASE,
        )
        or re.search(
            r"\bsame\s+meaning\b.*\bsection\s+[0-9]"
            r"|\btreated\s+as\b.*\bunder\s+section\s+[0-9]"
            r"|\brules\s+similar\s+to\b.*\bsection\s+[0-9]"
            r"|\bin\s+accordance\s+with\s+section\s+[0-9]",
            source_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )


def _missing_cited_statute_targets(
    citation: str,
    source_text: str,
    context_files: list[EvalContextFile],
) -> list[tuple[str, str]]:
    """Return cited statute targets that are not available as copied context."""
    try:
        parts = parse_usc_citation(citation)
    except Exception:
        return []

    available = {
        _normalize_prompt_import_target(item.import_path) for item in context_files
    }
    missing: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\bsection\s+"
        r"(?P<section>[0-9][A-Za-z0-9.-]*)"
        r"(?P<fragments>(?:\([A-Za-z0-9]+\))*)",
        source_text,
        flags=re.IGNORECASE,
    ):
        if _citation_match_points_to_other_act(source_text, match.end()):
            continue
        section = match.group("section").rstrip(".")
        if section == parts.section:
            continue
        fragments = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("fragments")))
        cited_parts = CitationParts(parts.title, section, fragments)
        target = _relative_rulespec_path_to_import_target(
            citation_to_relative_rulespec_path(cited_parts).with_suffix(""),
            prefix="us",
        )
        normalized = _normalize_prompt_import_target(target)
        if any(
            _prompt_import_covers(available_target, normalized)
            for available_target in available
        ):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        label = "section " + section + "".join(f"({fragment})" for fragment in fragments)
        missing.append((label, target))
    return missing


def _normalize_prompt_import_target(import_target: str) -> str:
    """Normalize an import target for prompt-context availability checks."""
    normalized = import_target.strip().strip("\"'")
    normalized = normalized.split("#", 1)[0]
    if ":" in normalized:
        _, normalized = normalized.split(":", 1)
    return normalized.strip("/")


def _prompt_import_covers(available: str, expected: str) -> bool:
    """Return whether an available prompt import covers an expected target."""
    return (
        available == expected
        or available.startswith(expected + "/")
        or expected.startswith(available + "/")
    )


def _citation_match_points_to_other_act(source_text: str, match_end: int) -> bool:
    """Heuristically skip `section X of the Other Act` references."""
    following = source_text[match_end : match_end + 80]
    return bool(
        re.match(
            r"\s+of\s+(?!this\s+(?:section|title)\b)(?!the\s+Internal\s+Revenue\s+Code\b)",
            following,
            flags=re.IGNORECASE,
        )
    )


def _citation_example_suffix(import_target: str) -> str:
    """Return identifier-style suffix for a cited import target."""
    normalized = _normalize_prompt_import_target(import_target)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) >= 3 and parts[0] == "statutes":
        return "_".join(parts[2:])
    return "_".join(parts[-2:]) if len(parts) >= 2 else "cited"


def _format_proration_test_guidance(source_text: str) -> str:
    """Return source-specific guidance for exact proration companion tests."""
    if not re.search(
        r"\bfraction\b.*\bdenominator\b.*\b[0-9]",
        source_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return ""
    return """
Source-specific proration test guidance:
- For proration tests with a source-stated denominator, choose input amounts
  divisible by that denominator so expected outputs are exact decimals, not
  rounded approximations. For denominator 365, prefer values like 36500 with
  182 days, so `36500 * 182 / 365 = 18200`, rather than 100000 with 182 days.
"""


@dataclass(frozen=True)
class _StatuteCitation:
    label: str
    section: str
    fragments: tuple[str, ...]


def _import_target_to_statute_citation(import_path: str) -> _StatuteCitation | None:
    """Infer a statute section citation from a RuleSpec import target."""
    normalized = import_path.strip().strip("\"'")
    normalized = normalized.split("#", 1)[0].strip().strip("/")
    if ":" in normalized:
        maybe_prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", maybe_prefix) and rest:
            normalized = rest.strip("/")
    if normalized.endswith((".yaml", ".yml")):
        normalized = normalized.rsplit(".", 1)[0]
    parts = [part for part in normalized.split("/") if part]
    try:
        statutes_index = parts.index("statutes")
    except ValueError:
        return None
    if len(parts) <= statutes_index + 2:
        return None
    section = parts[statutes_index + 2]
    fragments = tuple(parts[statutes_index + 3 :])
    if not re.fullmatch(r"[0-9][A-Za-z0-9.-]*", section):
        return None
    if not all(_is_citation_fragment(fragment) for fragment in fragments):
        return None
    label = section + "".join(f"({fragment})" for fragment in fragments)
    return _StatuteCitation(label=label, section=section, fragments=fragments)


def _source_text_cites_statute(
    source_text: str,
    citation: _StatuteCitation,
) -> bool:
    """Return whether source text cites the statute section or fragment."""
    citation_pattern = re.escape(citation.section)
    for fragment in citation.fragments:
        citation_pattern += rf"\s*\(\s*{re.escape(fragment)}\s*\)"
    if re.search(
        rf"\bsection\s+{citation_pattern}(?:\b|\s|\)|,|;|\.)",
        source_text,
        flags=re.IGNORECASE,
    ):
        return True
    if citation.fragments:
        return bool(
            re.search(
                rf"\b{citation_pattern}(?:\b|\s|\)|,|;|\.)",
                source_text,
                flags=re.IGNORECASE,
            )
        )
    return False


def _preferred_exports_for_cited_reference(
    source_text: str,
    citation: _StatuteCitation,
    exports: list[str],
) -> list[str]:
    """Suggest likely final outputs for source references to cited deductions."""
    window = _source_text_citation_window(source_text, citation).lower()
    if not any(
        token in window
        for token in (
            "deduction",
            "deduct",
            "exemption",
            "credit",
            "allowance",
            "allowed",
            "allowable",
        )
    ):
        return []

    scored: list[tuple[tuple[int, int, int, str], str]] = []
    for export in exports:
        name = export.lower()
        score = 0
        if "exemption" in window and "exemption" in name:
            score += 50
        if "credit" in window and "credit" in name:
            score += 50
        if "deduction" in window and "deduction" in name:
            score += 50
        if "allowance" in window and "allowance" in name:
            score += 50
        if name.endswith(("_deduction", "_credit", "_allowance")):
            score += 35
        if name.startswith("section_"):
            score += 10
        if any(
            marker in name
            for marker in (
                "_before_",
                "_cap",
                "_eligible",
                "_eligibility",
                "_increment",
                "_phaseout",
                "_rate",
                "_threshold",
                "_base",
                "_amount_per",
                "_modified_adjusted",
            )
        ):
            score -= 30
        if score <= 0:
            continue
        scored.append(((-score, len(export), exports.index(export), export), export))
    return [export for _sort_key, export in sorted(scored)[:3]]


def _source_text_citation_window(
    source_text: str,
    citation: _StatuteCitation,
) -> str:
    citation_pattern = re.escape(citation.section)
    for fragment in citation.fragments:
        citation_pattern += rf"\s*\(\s*{re.escape(fragment)}\s*\)"
    match = re.search(
        rf"\bsection\s+{citation_pattern}(?:\b|\s|\)|,|;|\.)",
        source_text,
        flags=re.IGNORECASE,
    )
    if match is None and citation.fragments:
        match = re.search(
            rf"\b{citation_pattern}(?:\b|\s|\)|,|;|\.)",
            source_text,
            flags=re.IGNORECASE,
        )
    if match is None:
        return ""
    start = max(0, match.start() - 120)
    end = min(len(source_text), match.end() + 80)
    return source_text[start:end]


def _is_citation_fragment(fragment: str) -> bool:
    """Return whether an import path fragment can represent a citation part."""
    return bool(re.fullmatch(r"[A-Za-z0-9]+", fragment))


def _context_import_path_key(import_path: str) -> tuple[str | None, str] | None:
    """Normalize an import target for sibling/target comparisons."""
    normalized = import_path.strip().strip("\"'")
    if not normalized:
        return None
    normalized = normalized.split("#", 1)[0].strip().strip("/")
    prefix: str | None = None
    if ":" in normalized:
        maybe_prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", maybe_prefix) and rest:
            prefix = maybe_prefix
            normalized = rest.strip("/")
    if normalized.endswith((".yaml", ".yml")):
        normalized = normalized.rsplit(".", 1)[0]
    normalized = normalized.strip("/")
    return (prefix, normalized) if normalized else None


def _context_import_parent_key(
    key: tuple[str | None, str],
) -> tuple[str | None, str]:
    prefix, target = key
    return prefix, Path(target).parent.as_posix()


def _format_rule_name_list(names: list[str], *, limit: int = 12) -> str:
    """Format a compact backticked list of rule names."""
    visible = ", ".join(f"`{name}`" for name in names[:limit])
    if len(names) > limit:
        visible += ", ..."
    return visible or "`<none>`"


def _context_file_hash(source_path: str) -> str | None:
    """Return the sha256 hash for a context file when it is readable."""
    try:
        digest = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    except OSError:
        return None
    return f"sha256:{digest}"


def _context_file_export_detail(item: EvalContextFile) -> str:
    """Return a compact list of exported symbols for a context RuleSpec file."""
    exports = _context_file_exports(item.source_path)
    if not exports:
        return ""
    references = ", ".join(f"`{item.import_path}#{name}`" for name in exports[:8])
    if len(exports) > 8:
        references += ", ..."
    terminal_exports = _context_file_terminal_exports(item.source_path)
    terminal_detail = ""
    if terminal_exports:
        terminal_references = ", ".join(
            f"`{item.import_path}#{name}`" for name in terminal_exports[:5]
        )
        if len(terminal_exports) > 5:
            terminal_references += ", ..."
        terminal_detail = f"; terminal exports {terminal_references}"
    return f"; exports {references}{terminal_detail}"


def _context_file_exports(source_path: str) -> list[str]:
    """Extract RuleSpec rule names exported by a copied context file."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    exports: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        name = str(rule.get("name") or "").strip()
        if name:
            exports.append(name)
    return exports


_CONTEXT_FORMULA_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CONTEXT_FORMULA_BUILTINS = {
    "and",
    "ceil",
    "else",
    "false",
    "floor",
    "if",
    "match",
    "max",
    "min",
    "not",
    "or",
    "true",
}


def _context_file_terminal_exports(source_path: str) -> list[str]:
    """Return executable exports not consumed by another local executable rule."""
    try:
        payload = yaml.safe_load(Path(source_path).read_text())
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    exports: list[str] = []
    formulas: list[tuple[str, str]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() not in {
            "parameter",
            "derived",
        }:
            continue
        name = str(rule.get("name") or "").strip()
        if not name:
            continue
        exports.append(name)
        versions = rule.get("versions")
        if not isinstance(versions, list):
            continue
        for version in versions:
            if not isinstance(version, dict):
                continue
            formula = version.get("formula")
            if isinstance(formula, (int, float)) and not isinstance(formula, bool):
                formulas.append((name, str(formula)))
            elif isinstance(formula, str) and formula.strip():
                formulas.append((name, formula))
    if not exports:
        return []

    export_names = set(exports)
    referenced: set[str] = set()
    for owner, formula in formulas:
        referenced.update(
            identifier
            for identifier in _CONTEXT_FORMULA_IDENTIFIER.findall(formula)
            if identifier in export_names
            and identifier != owner
            and identifier not in _CONTEXT_FORMULA_BUILTINS
        )
    terminal = [name for name in exports if name not in referenced]
    return terminal or exports


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
    policy_root: Path,
    target_rel: Path | None,
) -> list[tuple[Path, str]]:
    """Expand selected precedent files with their transitive canonical imports."""
    expanded: list[tuple[Path, str]] = []
    pending: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    for path in selected_paths:
        is_target = (
            target_rel is not None
            and _relative_to_root(path, policy_root) == target_rel
        )
        kind = (
            "existing_target"
            if is_target
            else (
                "implementation_precedent"
                if _is_under_root(path, policy_root)
                else "implementation_external"
            )
        )
        pending.append((path, kind))

    while pending:
        source_path, kind = pending.pop(0)
        resolved = source_path.resolve()
        if resolved in seen:
            continue
        if (
            target_rel is not None
            and _relative_to_root(source_path, policy_root) == target_rel
            and kind != "existing_target"
        ):
            continue
        seen.add(resolved)
        expanded.append((source_path, kind))
        if (
            source_path.suffix in {".yaml", ".yml"}
            and not source_path.name.endswith(".test.yaml")
        ):
            test_path = _rulespec_test_path(source_path)
            resolved_test = test_path.resolve()
            if test_path.exists() and resolved_test not in seen:
                seen.add(resolved_test)
                test_kind = (
                    "existing_target_test_context"
                    if kind == "existing_target"
                    else "implementation_test_context"
                )
                expanded.append((test_path, test_kind))

        if not _is_under_root(source_path, policy_root):
            continue

        for dependency in _resolve_context_imports(source_path, policy_root):
            if dependency.resolve() in seen:
                continue
            pending.append((dependency, "implementation_dependency"))

    return expanded


def _resolve_context_imports(source_path: Path, policy_root: Path) -> list[Path]:
    """Resolve canonical import targets for one copied precedent file."""
    dependencies: list[Path] = []
    for import_target in _extract_import_targets(source_path.read_text()):
        import_prefix = _import_target_prefix(import_target)
        target_path = _import_target_to_path(import_target)
        candidates = [policy_root / target_path]
        if import_prefix:
            candidates.append(
                policy_root.parent / f"rulespec-{import_prefix}" / target_path
            )
        if target_path.parts:
            first = target_path.parts[0]
            if first == policy_root.name:
                candidates.append(policy_root / Path(*target_path.parts[1:]))
            if first in _LOCAL_IMPORT_ROOT_TOKENS:
                candidates.append(policy_root.parent / target_path)

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
    normalized = normalized.split("#", 1)[0].strip()
    if ":" in normalized:
        prefix, rest = normalized.split(":", 1)
        if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and rest:
            normalized = rest
    if normalized.endswith((".yaml", ".yml")):
        return Path(normalized)
    return Path(f"{normalized}.yaml")


def _import_target_prefix(import_target: str) -> str | None:
    """Return the repo prefix from an absolute RuleSpec import target."""
    normalized = import_target.strip().strip('"').strip("'")
    normalized = normalized.split("#", 1)[0].strip()
    if ":" not in normalized:
        return None
    prefix, rest = normalized.split(":", 1)
    if re.fullmatch(r"[a-z][a-z0-9-]*", prefix) and rest:
        return prefix
    return None


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


def _rulespec_declares_rule(
    rulespec_content: str | None,
    var_name: str | None,
) -> bool:
    if not rulespec_content or not var_name:
        return False
    with contextlib.suppress(yaml.YAMLError, TypeError):
        payload = yaml.safe_load(rulespec_content)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            return any(
                isinstance(rule, dict) and rule.get("name") == var_name
                for rule in payload["rules"]
            )
    return False


def _canonical_rulespec_target_for_path(rulespec_path: Path | None) -> str | None:
    if rulespec_path is None:
        return None
    components = [
        str(component)
        for component in rulespec_path.expanduser().resolve(strict=False).parts
    ]
    repo_index = next(
        (
            index
            for index in range(len(components) - 1, -1, -1)
            if components[index].startswith("rulespec-")
        ),
        None,
    )
    if repo_index is None or repo_index + 1 >= len(components):
        return None
    prefix = components[repo_index].removeprefix("rulespec-")
    if not prefix:
        return None
    relative = Path(*components[repo_index + 1 :])
    if relative.suffix in {".yaml", ".yml"}:
        relative = relative.with_suffix("")
    return f"{prefix}:{relative.as_posix()}"


def _policyengine_hint_test_output_key(
    rulespec_content: str | None,
    policyengine_rule_hint: str | None,
    rulespec_path: Path | None,
) -> str | None:
    if not _rulespec_declares_rule(rulespec_content, policyengine_rule_hint):
        return None
    canonical_target = _canonical_rulespec_target_for_path(rulespec_path)
    if canonical_target and policyengine_rule_hint:
        return f"{canonical_target}#{policyengine_rule_hint}"
    return policyengine_rule_hint


def _complete_oracle_hint_test_outputs(
    content: str,
    rulespec_content: str | None,
    policyengine_rule_hint: str | None,
    rulespec_path: Path | None = None,
) -> str:
    if not policyengine_rule_hint:
        return content
    output_key = _policyengine_hint_test_output_key(
        rulespec_content, policyengine_rule_hint, rulespec_path
    )
    if not output_key:
        return content
    hint_value = _extract_simple_rulespec_constant(
        rulespec_content, policyengine_rule_hint
    )
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
        if not isinstance(output, dict) or output_key in output:
            normalized_cases.append(case)
            continue
        normalized_case = dict(case)
        normalized_output = dict(output)
        if output_key != policyengine_rule_hint and policyengine_rule_hint in output:
            normalized_output[output_key] = normalized_output.pop(
                policyengine_rule_hint
            )
        elif hint_value is None:
            normalized_cases.append(case)
            continue
        else:
            normalized_output[output_key] = hint_value
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
                        rulespec_path=expected_path,
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
                rulespec_path=expected_path,
            )
        expected_test_path.write_text(test_content)

    return True


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "eval"
