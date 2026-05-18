"""
Axiom Encode CLI - command line interface for RuleSpec encoding.

Primary workflow:
  1. axiom-encode encode "26 USC 21" emits Axiom RuleSpec
  2. axiom-encode validate <file.yaml> runs standalone validation
  3. axiom-encode eval-suite <manifest.yaml> runs batch encoding evals
  4. axiom-encode log records encoding runs
  5. axiom-encode stats shows patterns for improvement

Self-contained -- no external plugin dependencies.
"""

import argparse
import contextlib
import copy
import csv
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from axiom_encode import __version__

from .concepts import (
    audit_corpus as audit_concept_corpus,
)
from .concepts import (
    load_concept_registry,
    validate_generated_against_registry,
)
from .concepts.jurisdiction import jurisdiction_prefix as _repo_jurisdiction_prefix
from .constants import DEFAULT_OPENAI_MODEL
from .harness.encoding_db import (
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    ReviewResult,
    ReviewResults,
)
from .harness.evals import (
    _eval_result_from_payload,
    evaluate_artifact,
    load_eval_suite_manifest,
    resolve_corpus_source_unit,
    run_eval_suite,
    run_model_eval,
    run_source_eval,
    summarize_readiness,
)
from .harness.proof_validator import validate_rulespec_proofs
from .harness.validator_pipeline import (
    ValidatorPipeline,
    find_proof_import_reference_issues,
    find_tax_filing_status_local_input_issues,
    find_tax_status_component_local_input_issues,
    find_unused_import_issues,
    repair_current_year_final_amount_tables,
    repair_nonnegative_amount_reductions,
)
from .oracles.policyengine.coverage import (
    build_policyengine_candidate_report,
    build_policyengine_coverage_report,
)
from .oracles.policyengine.ecps_snap import (
    configure_parser as configure_snap_ecps_compare_parser,
)
from .oracles.policyengine.ecps_snap import (
    main as run_snap_ecps_compare,
)
from .oracles.policyengine.ecps_tax import (
    configure_parser as configure_tax_ecps_compare_parser,
)
from .oracles.policyengine.ecps_tax import (
    main as run_tax_ecps_compare,
)
from .oracles.policyengine.registry import load_policyengine_registry
from .oracles.policyengine.snap_readiness import build_snap_readiness_report
from .repo_routing import canonical_rulespec_repo_name, find_policy_repo_root

# Default DB path - can be overridden with --db
DEFAULT_DB = Path.home() / "TheAxiomFoundation" / "axiom-encode" / "encodings.db"
DEFAULT_GPT_RUNNER = f"codex:{DEFAULT_OPENAI_MODEL}"
RULESPEC_SOURCE_ROOTS = {"policies", "regulations", "statutes"}
APPLIED_ENCODING_MANIFEST_DIR = Path(".axiom") / "encoding-manifests"
APPLIED_ENCODING_MANIFEST_SCHEMA = "axiom-encode/applied-rulespec/v1"
APPLIED_ENCODING_SIGNING_KEY_ENV = "AXIOM_ENCODE_APPLY_SIGNING_KEY"
APPLIED_ENCODING_SIGNATURE_ALGORITHM = "hmac-sha256"
APPLIED_ENCODING_SIGNATURE_KEY_ID = "axiom-encode-apply-v1"


def _resolve_repo_checkout(name: str) -> Path:
    """Resolve sibling foundation repos before falling back to home checkouts."""
    workspace_root = Path(__file__).resolve().parents[3]
    candidates = [
        workspace_root / name,
        Path.home() / "TheAxiomFoundation" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _resolve_runtime_axiom_rules_checkout(
    policy_repo_root: Path | None = None,
) -> Path:
    """Resolve the Axiom rules engine checkout, preferring a sibling repo."""
    if policy_repo_root is not None:
        sibling = Path(policy_repo_root).resolve().parent / "axiom-rules-engine"
        if sibling.exists():
            return sibling.resolve()
    return _resolve_repo_checkout("axiom-rules-engine")


def _resolve_policy_repo_for_corpus_source(
    corpus_citation_path: str,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return override
    jurisdiction = corpus_citation_path.strip().split("/", 1)[0] or "us"
    repo_name = f"rulespec-{jurisdiction}"
    return _resolve_repo_checkout(repo_name)


def _resolve_validation_repo_roots(rulespec_file: Path) -> tuple[Path, Path]:
    """Resolve the policy repo root plus the Axiom rules engine for validation."""
    policy_repo_root = find_policy_repo_root(rulespec_file) or _resolve_repo_checkout(
        "rulespec-us"
    )
    return policy_repo_root, _resolve_runtime_axiom_rules_checkout(policy_repo_root)


def _reviewer_score_map(scores) -> dict[str, float | None]:
    """Normalize reviewer scores from flat attrs or ReviewResults.reviews."""
    reviewer_names = [
        "rulespec_reviewer",
        "formula_reviewer",
        "parameter_reviewer",
        "integration_reviewer",
    ]
    values = {name: getattr(scores, name, None) for name in reviewer_names}

    for review in getattr(scores, "reviews", []) or []:
        name = getattr(review, "reviewer", "")
        if name not in values:
            continue
        checked = getattr(review, "items_checked", 0) or 0
        passed = getattr(review, "items_passed", 0) or 0
        if checked > 0:
            values[name] = round(passed / checked * 10, 1)
        else:
            values[name] = 10.0 if getattr(review, "passed", False) else 0.0

    return values


def _add_gpt_backend_argument(parser: argparse.ArgumentParser) -> None:
    """Add a GPT backend override for local-vs-API runner selection."""
    parser.add_argument(
        "--gpt-backend",
        choices=["codex", "openai"],
        default=None,
        help=(
            "Override GPT runner backend for evals. "
            "Use 'codex' locally to route gpt-* runners through Codex CLI/ChatGPT, "
            "or 'openai' to force API-backed Responses runs. "
            "Defaults to 'codex' locally, or the AXIOM_ENCODE_GPT_BACKEND env var when set."
        ),
    )


def _resolved_gpt_backend(args) -> str | None:
    """Resolve the requested GPT backend override from args/env."""
    return (
        getattr(args, "gpt_backend", None)
        or os.getenv("AXIOM_ENCODE_GPT_BACKEND")
        or "codex"
    )


def _rewrite_gpt_runner_backend(spec: str, backend: str | None) -> str:
    """Rewrite gpt-* runner specs onto the requested backend, preserving aliases."""
    if backend not in {"codex", "openai"}:
        return spec

    alias = ""
    target = spec
    if "=" in spec:
        alias, target = spec.split("=", 1)

    if ":" not in target:
        return spec

    current_backend, model = target.split(":", 1)
    current_backend = current_backend.strip()
    model = model.strip()
    if current_backend not in {"codex", "openai"}:
        return spec
    if not model.startswith("gpt-"):
        return spec

    rewritten = f"{backend}:{model}"
    return f"{alias}={rewritten}" if alias else rewritten


def _effective_runner_specs(specs: list[str], args) -> list[str]:
    """Apply GPT backend override to a runner list."""
    backend = _resolved_gpt_backend(args)
    return [_rewrite_gpt_runner_backend(spec, backend) for spec in specs]


def _utc_now_iso() -> str:
    """Render an RFC3339 UTC timestamp without fractional seconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_rulespec_inventory_root() -> Path:
    """Resolve the workspace root that contains rulespec-* repos."""
    workspace_root = Path(__file__).resolve().parents[3]
    if any(workspace_root.glob("rulespec-*")):
        return workspace_root
    return Path.home() / "TheAxiomFoundation"


def _is_rulespec_yaml(path: Path) -> bool:
    """Return whether path is a non-test RuleSpec YAML file."""
    if path.suffix not in {".yaml", ".yml"}:
        return False
    if path.name.endswith(".test.yaml") or path.name.endswith(".test.yml"):
        return False
    parts = set(path.parts)
    return bool(parts & RULESPEC_SOURCE_ROOTS)


def _load_rulespec_payload(path: Path) -> dict:
    """Load a RuleSpec YAML file and normalize missing/empty content."""
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _is_composition_rulespec(payload: dict) -> bool:
    """Detect composition modules without depending on one hard-coded path."""
    module = payload.get("module") if isinstance(payload.get("module"), dict) else {}
    marker_values = [
        module.get("kind"),
        module.get("type"),
        module.get("category"),
        module.get("summary"),
    ]
    marker_text = " ".join(str(value) for value in marker_values if value)
    if re.search(r"\bcomposition\b", marker_text, flags=re.IGNORECASE):
        return True

    for rule in payload.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        source = rule.get("source")
        if isinstance(source, str) and re.search(
            r"\bcomposition\b", source, flags=re.IGNORECASE
        ):
            return True
    return False


def build_rulespec_inventory(root: Path | None = None) -> dict:
    """Build a RuleSpec inventory across sibling rulespec-* repos."""
    inventory_root = (root or _default_rulespec_inventory_root()).resolve()
    repo_summaries = []
    total_kind_counts: Counter[str] = Counter()
    total_files = 0
    total_source_provision_files = 0
    total_composition_files = 0
    total_rules = 0

    for repo in sorted(inventory_root.glob("rulespec-*")):
        if not repo.is_dir():
            continue

        repo_kind_counts: Counter[str] = Counter()
        repo_root_counts: Counter[str] = Counter()
        repo_files = 0
        repo_source_provision_files = 0
        repo_composition_files = 0
        repo_rules = 0

        for rulespec_file in sorted(repo.rglob("*.y*ml")):
            if not _is_rulespec_yaml(rulespec_file):
                continue

            payload = _load_rulespec_payload(rulespec_file)
            rules = payload.get("rules", []) or []
            if not isinstance(rules, list):
                rules = []
            is_composition = _is_composition_rulespec(payload)
            relative = rulespec_file.relative_to(repo)
            root_name = relative.parts[0]

            repo_files += 1
            repo_root_counts[root_name] += 1
            if is_composition:
                repo_composition_files += 1
            else:
                repo_source_provision_files += 1

            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                kind = str(rule.get("kind") or "missing")
                repo_kind_counts[kind] += 1
                repo_rules += 1

        if repo_files == 0:
            continue

        total_files += repo_files
        total_source_provision_files += repo_source_provision_files
        total_composition_files += repo_composition_files
        total_rules += repo_rules
        total_kind_counts.update(repo_kind_counts)

        repo_summaries.append(
            {
                "repo": repo.name,
                "files": repo_files,
                "source_provision_files": repo_source_provision_files,
                "composition_files": repo_composition_files,
                "rules": repo_rules,
                "roots": dict(sorted(repo_root_counts.items())),
                "kinds": dict(sorted(repo_kind_counts.items())),
            }
        )

    return {
        "root": str(inventory_root),
        "total_files": total_files,
        "source_provision_files": total_source_provision_files,
        "composition_files": total_composition_files,
        "total_rules": total_rules,
        "kind_counts": dict(sorted(total_kind_counts.items())),
        "repos": repo_summaries,
    }


def _format_counter(counter: dict[str, int]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.items())


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Axiom Encode - AI-assisted RuleSpec encoding infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a RuleSpec YAML file (CI + reviewer agents)"
    )
    validate_parser.add_argument("file", type=Path, help="Path to RuleSpec YAML file")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    validate_parser.add_argument("--skip-reviewers", action="store_true")
    validate_parser.add_argument(
        "--oracle",
        choices=["policyengine", "taxsim", "all"],
        help="Run external validation against oracles",
    )
    validate_parser.add_argument(
        "--min-match",
        type=float,
        default=0.95,
        help="Minimum match rate for oracle validation (default: 0.95)",
    )
    validate_parser.add_argument(
        "--require-oracle-classification",
        action="store_true",
        help="Fail oracle validation when oracle coverage reports unclassified legal IDs",
    )

    proof_validate_parser = subparsers.add_parser(
        "proof-validate",
        help="Validate explicit RuleSpec proof trees without reviewers or oracles",
    )
    proof_validate_parser.add_argument(
        "file", type=Path, help="Path to RuleSpec YAML file"
    )
    proof_validate_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # log command
    log_parser = subparsers.add_parser("log", help="Log an encoding run to encoding DB")
    log_parser.add_argument("--citation", required=True, help="Legal citation")
    log_parser.add_argument(
        "--file", type=Path, required=True, help="Path to RuleSpec YAML file"
    )
    log_parser.add_argument(
        "--iterations", type=int, default=1, help="Number of iterations"
    )
    log_parser.add_argument(
        "--errors", type=str, default="[]", help="Errors as JSON array"
    )
    log_parser.add_argument(
        "--duration", type=int, default=0, help="Total duration in ms"
    )
    log_parser.add_argument(
        "--scores",
        type=str,
        help="Reviewer scores as JSON {rulespec,formula,param,integration}",
    )
    log_parser.add_argument(
        "--session", type=str, help="Session ID to link this run to"
    )
    log_parser.add_argument("--db", type=Path, default=Path("encodings.db"))

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show encoding statistics")
    stats_parser.add_argument("--db", type=Path, default=Path("encodings.db"))

    # inventory command
    inventory_parser = subparsers.add_parser(
        "inventory", help="Show RuleSpec inventory across rulespec-* repos"
    )
    inventory_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-* repos",
    )
    inventory_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # oracle-coverage command
    oracle_coverage_parser = subparsers.add_parser(
        "oracle-coverage",
        help="Classify RuleSpec executable outputs against oracle registries",
    )
    oracle_coverage_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-* repos",
    )
    oracle_coverage_parser.add_argument(
        "--oracle",
        choices=["policyengine"],
        default="policyengine",
        help="Oracle registry to inspect",
    )
    oracle_coverage_parser.add_argument(
        "--program",
        default=None,
        help="Restrict report to a program label such as snap or tax",
    )
    oracle_coverage_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum unmapped outputs to print in text mode",
    )
    oracle_coverage_parser.add_argument(
        "--fail-on-unmapped",
        action="store_true",
        help="Exit non-zero when any executable output is unmapped",
    )
    oracle_coverage_parser.add_argument(
        "--fail-on-untested-comparable",
        action="store_true",
        help="Exit non-zero when a comparable output is absent from companion tests",
    )
    oracle_coverage_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    oracle_candidates_parser = subparsers.add_parser(
        "oracle-candidates",
        help="Show candidate RuleSpec outputs for expanding oracle coverage",
    )
    oracle_candidates_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-* repos",
    )
    oracle_candidates_parser.add_argument(
        "--oracle",
        choices=["policyengine"],
        default="policyengine",
        help="Oracle registry to inspect",
    )
    oracle_candidates_parser.add_argument(
        "--program",
        default=None,
        help="Restrict report to a program label such as snap or tax",
    )
    oracle_candidates_parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum candidates to print in text mode",
    )
    oracle_candidates_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    snap_ecps_compare_parser = subparsers.add_parser(
        "snap-ecps-compare",
        help="Compare SNAP RuleSpec output against PolicyEngine ECPS",
    )
    configure_snap_ecps_compare_parser(snap_ecps_compare_parser)

    tax_ecps_compare_parser = subparsers.add_parser(
        "tax-ecps-compare",
        help="Compare federal tax RuleSpec output against PolicyEngine ECPS",
    )
    configure_tax_ecps_compare_parser(tax_ecps_compare_parser)

    snap_readiness_parser = subparsers.add_parser(
        "snap-readiness",
        help="Report SNAP corpus, RuleSpec, and ECPS oracle readiness by state repo",
    )
    snap_readiness_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-us-* repos",
    )
    snap_readiness_parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Axiom corpus checkout containing data/corpus/provisions",
    )
    snap_readiness_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # calibration command
    calibration_parser = subparsers.add_parser(
        "calibration", help="Show review pass-rate metrics"
    )
    calibration_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    calibration_parser.add_argument("--limit", type=int, default=50)

    # runs command
    runs_parser = subparsers.add_parser("runs", help="List recent encoding runs")
    runs_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    runs_parser.add_argument("--limit", type=int, default=20)

    concepts_audit_parser = subparsers.add_parser(
        "concepts-audit",
        help="Report canonical-concept drift across sibling rules/rulespec repos",
    )
    concepts_audit_parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        required=True,
        help="Roots to walk (e.g. ~/rulespec-us ~/rulespec-us-co ~/rules-us-co)",
    )
    concepts_audit_parser.add_argument(
        "--path-filter",
        default=None,
        help="Optional regex applied to file paths to narrow the walk",
    )
    concepts_audit_parser.add_argument(
        "--name-prefix",
        action="append",
        default=None,
        help="Restrict findings to identifiers with this prefix (repeatable)",
    )
    concepts_audit_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit findings as JSON instead of human-readable text",
    )

    guard_generated_parser = subparsers.add_parser(
        "guard-generated",
        help="Reject RuleSpec changes that were not installed by axiom-encode --apply",
    )
    guard_generated_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository to inspect",
    )
    guard_generated_parser.add_argument(
        "--base-ref",
        default=None,
        help="Git base ref/SHA for changed-file detection",
    )
    guard_generated_parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Git head ref/SHA for changed-file detection",
    )
    guard_generated_parser.add_argument(
        "--roots",
        default=" ".join(sorted(RULESPEC_SOURCE_ROOTS)),
        help="Space-separated RuleSpec roots to protect",
    )
    guard_generated_parser.add_argument(
        "--all",
        action="store_true",
        help="Verify every existing RuleSpec YAML file, not only git changes",
    )
    guard_generated_parser.add_argument(
        "--json", action="store_true", help="Output guard result as JSON"
    )

    repair_floor_parser = subparsers.add_parser(
        "repair-nonnegative-floors",
        help="Apply signed deterministic repairs for nonnegative amount reductions",
    )
    repair_floor_parser.add_argument("file", type=Path, help="RuleSpec YAML file")
    repair_floor_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_floor_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_final_amount_parser = subparsers.add_parser(
        "repair-current-year-final-amounts",
        help="Apply signed deterministic repairs for imported final amount tables",
    )
    repair_final_amount_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_final_amount_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_final_amount_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_zero_tests_parser = subparsers.add_parser(
        "repair-zero-branch-tests",
        help="Apply signed deterministic repairs for missing zero-branch tests",
    )
    repair_zero_tests_parser.add_argument("file", type=Path, help="RuleSpec YAML file")
    repair_zero_tests_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_zero_tests_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_unused_imports_parser = subparsers.add_parser(
        "repair-unused-imports",
        help="Apply signed deterministic repairs for unused RuleSpec imports",
    )
    repair_unused_imports_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_unused_imports_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_unused_imports_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_proof_hash_parser = subparsers.add_parser(
        "repair-proof-import-hashes",
        help="Apply signed deterministic repairs for proof import hashes",
    )
    repair_proof_hash_parser.add_argument("file", type=Path, help="RuleSpec YAML file")
    repair_proof_hash_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_proof_hash_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_proof_reference_parser = subparsers.add_parser(
        "repair-unreferenced-proof-imports",
        help="Apply signed deterministic repairs for stale proof import atoms",
    )
    repair_proof_reference_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_proof_reference_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_proof_reference_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_imported_test_inputs_parser = subparsers.add_parser(
        "repair-imported-test-inputs",
        help="Apply signed deterministic repairs for missing imported test inputs",
    )
    repair_imported_test_inputs_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_imported_test_inputs_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_imported_test_inputs_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_oracle_parameter_tests_parser = subparsers.add_parser(
        "repair-oracle-parameter-tests",
        help="Apply signed deterministic repairs for missing oracle parameter tests",
    )
    repair_oracle_parameter_tests_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_oracle_parameter_tests_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_oracle_parameter_tests_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_tax_filing_parser = subparsers.add_parser(
        "repair-tax-filing-status-branches",
        help="Apply signed deterministic repairs for US tax filing-status branches",
    )
    repair_tax_filing_parser.add_argument("file", type=Path, help="RuleSpec YAML file")
    repair_tax_filing_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_tax_filing_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_tax_status_component_parser = subparsers.add_parser(
        "repair-tax-status-components",
        help="Apply signed deterministic repairs for local tax status component facts",
    )
    repair_tax_status_component_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_tax_status_component_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_tax_status_component_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    repair_source_proofs_parser = subparsers.add_parser(
        "repair-missing-source-proofs",
        help="Apply signed deterministic repairs for missing RuleSpec proof atoms",
    )
    repair_source_proofs_parser.add_argument(
        "file", type=Path, help="RuleSpec YAML file"
    )
    repair_source_proofs_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used for manifest signing",
    )
    repair_source_proofs_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    # test command
    test_parser = subparsers.add_parser(
        "test", help="Execute RuleSpec companion .test.yaml cases"
    )
    test_parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="RuleSpec .test.yaml files or directories to test",
    )
    test_parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Rules repository root used when no paths are supplied",
    )
    test_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    test_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # compile command
    compile_parser = subparsers.add_parser(
        "compile", help="Compile a RuleSpec YAML file with the Axiom rules engine"
    )
    compile_parser.add_argument("file", type=Path, help="Path to RuleSpec YAML file")
    compile_parser.add_argument(
        "--as-of",
        default=None,
        help="Date for temporal resolution (YYYY-MM-DD, default: today)",
    )
    compile_parser.add_argument("--json", action="store_true", help="Output as JSON")
    compile_parser.add_argument(
        "--execute",
        action="store_true",
        help="Deprecated; execution is not part of RuleSpec compile",
    )
    compile_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )

    # encode command - run the current RuleSpec generation path
    encode_parser = subparsers.add_parser(
        "encode", help="Encode a corpus-backed citation as Axiom RuleSpec"
    )
    encode_parser.add_argument(
        "citation",
        help=(
            "USC citation or corpus citation path "
            "(e.g., '26 USC 1(j)(2)' or 'us/statute/26/1')"
        ),
    )
    encode_parser.add_argument(
        "--source-id",
        default=None,
        help=(
            "Optional canonical RuleSpec source identifier to write while reading "
            "the requested corpus citation as source text. Use when a corpus "
            "page backs a logical policy file."
        ),
    )
    encode_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/axiom-encode-encodings"),
        help="Output root for generated RuleSpec YAML",
    )
    encode_parser.add_argument(
        "--model",
        default=None,
        help=f"Model to use for encoding (default: {DEFAULT_OPENAI_MODEL})",
    )
    encode_parser.add_argument(
        "--backend",
        choices=["codex", "openai", "claude"],
        default="codex",
        help=(
            "Backend: 'codex' uses the Codex CLI/ChatGPT path, "
            "'openai' uses OpenAI Responses API, "
            "'claude' uses Claude CLI"
        ),
    )
    encode_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to axiom-corpus repo (defaults to sibling checkout)",
    )
    encode_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    encode_parser.add_argument(
        "--policy-repo-path",
        type=Path,
        default=None,
        help="Path to jurisdiction RuleSpec repo (defaults to sibling rulespec-us checkout)",
    )
    encode_parser.add_argument(
        "--mode",
        choices=["cold", "repo-augmented"],
        default="repo-augmented",
        help="Whether the encoding gets only source text or repo precedent context",
    )
    encode_parser.add_argument(
        "--allow-context",
        action="append",
        default=[],
        help="Extra file path to copy into the repo-augmented workspace (repeatable)",
    )
    encode_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Local encoding history database",
    )
    encode_parser.add_argument(
        "--no-sync",
        dest="sync",
        action="store_false",
        default=True,
        help="Do not sync the completed run to Supabase even when credentials are configured",
    )
    encode_parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "After successful validation, install the generated RuleSpec and "
            "companion test into the policy repo path. This is the non-manual "
            "path for updating live RuleSpec files."
        ),
    )
    encode_parser.add_argument(
        "--apply-target-only",
        action="store_true",
        help=(
            "With --apply, validate and install only the generated target file. "
            "Use for clean breaking migrations where direct dependents will be "
            "re-encoded in the same change set before final repository validation."
        ),
    )

    # eval command - run deterministic model comparisons on one or more citations
    eval_parser = subparsers.add_parser(
        "eval", help="Compare model runners on one or more citations"
    )
    eval_parser.add_argument("citations", nargs="+", help="Citation(s) to encode")
    eval_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help=f"Runner spec [name=]backend:model. Defaults to claude:opus and {DEFAULT_GPT_RUNNER}",
    )
    _add_gpt_backend_argument(eval_parser)
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/axiom_encode-evals"),
        help="Directory for eval artifacts and traces",
    )
    eval_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to axiom-corpus repo (defaults to sibling repo checkout)",
    )
    eval_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    eval_parser.add_argument(
        "--policy-repo-path",
        type=Path,
        default=None,
        help="Path to jurisdiction RuleSpec repo (defaults to sibling rulespec-us checkout)",
    )
    eval_parser.add_argument(
        "--mode",
        choices=["cold", "repo-augmented"],
        default="repo-augmented",
        help="Whether the eval gets only source text or a logged bundle of repo precedent files",
    )
    eval_parser.add_argument(
        "--allow-context",
        action="append",
        default=[],
        help="Extra file path to copy into the repo-augmented eval workspace (repeatable)",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )

    eval_source_parser = subparsers.add_parser(
        "eval-source",
        help="Compare model runners on one corpus-backed source unit",
    )
    eval_source_parser.add_argument(
        "corpus_citation_path",
        help="Corpus citation path for the authoritative source unit",
    )
    eval_source_parser.add_argument(
        "--source-id",
        default=None,
        help="Optional logical output identifier; defaults to the corpus citation path",
    )
    eval_source_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help=f"Runner spec [name=]backend:model. Defaults to claude:opus and {DEFAULT_GPT_RUNNER}",
    )
    _add_gpt_backend_argument(eval_source_parser)
    eval_source_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/axiom_encode-evals"),
        help="Directory for eval artifacts and traces",
    )
    eval_source_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    eval_source_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to axiom-corpus repo (defaults to sibling repo checkout)",
    )
    eval_source_parser.add_argument(
        "--policy-repo-path",
        type=Path,
        default=None,
        help="Path to jurisdiction RuleSpec repo (defaults from corpus jurisdiction)",
    )
    eval_source_parser.add_argument(
        "--mode",
        choices=["cold", "repo-augmented"],
        default="repo-augmented",
        help="Whether the eval gets only source text or a logged bundle of explicit precedent files",
    )
    eval_source_parser.add_argument(
        "--allow-context",
        action="append",
        default=[],
        help="Extra file path to copy into the repo-augmented eval workspace (repeatable)",
    )
    eval_source_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    eval_source_parser.add_argument(
        "--policyengine-rule-hint",
        dest="policyengine_rule_hint",
        default=None,
        help="Canonical rule name to use as the PolicyEngine oracle target for this source slice",
    )

    eval_suite_parser = subparsers.add_parser(
        "eval-suite",
        help="Run a manifest-driven benchmark suite and evaluate readiness gates",
    )
    eval_suite_parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a YAML manifest describing the benchmark suite",
    )
    eval_suite_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Override manifest runners with [name=]backend:model (repeatable)",
    )
    _add_gpt_backend_argument(eval_suite_parser)
    eval_suite_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/axiom_encode-suite-evals"),
        help="Directory for suite artifacts and traces",
    )
    eval_suite_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partially completed suite in the same output directory",
    )
    eval_suite_parser.add_argument(
        "--auto-resume-attempts",
        type=int,
        default=0,
        help="Retry and resume a suite after unexpected failures up to this many times",
    )
    eval_suite_parser.add_argument(
        "--auto-resume-delay-seconds",
        type=int,
        default=5,
        help="Seconds to wait before an automatic eval-suite resume attempt",
    )
    eval_suite_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to axiom-corpus repo (needed for citation cases; defaults to sibling repo checkout)",
    )
    eval_suite_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    eval_suite_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )

    eval_suite_revalidate_parser = subparsers.add_parser(
        "eval-suite-revalidate",
        help="Re-run validators for an existing eval-suite output without regenerating artifacts",
    )
    eval_suite_revalidate_parser.add_argument(
        "source_output",
        type=Path,
        help="Path to an existing eval-suite output directory containing suite-results.jsonl",
    )
    eval_suite_revalidate_parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest override (defaults to the manifest path recorded in suite-run.json)",
    )
    eval_suite_revalidate_parser.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        metavar="AXIOM_RULES_ENGINE_PATH",
        type=Path,
        default=None,
        help="Path to axiom-rules-engine repo (defaults to sibling checkout)",
    )
    eval_suite_revalidate_parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to axiom-corpus repo (defaults to sibling repo checkout)",
    )
    eval_suite_revalidate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )

    eval_suite_report_parser = subparsers.add_parser(
        "eval-suite-report",
        help="Render a paper-ready comparison report from eval-suite JSON output",
    )
    eval_suite_report_parser.add_argument(
        "result_json",
        type=Path,
        help="Path to JSON emitted by `axiom_encode eval-suite --json`",
    )
    eval_suite_report_parser.add_argument(
        "--left-runner",
        default=None,
        help="Runner name to treat as the left column (defaults to first runner in the payload)",
    )
    eval_suite_report_parser.add_argument(
        "--right-runner",
        default=None,
        help="Runner name to treat as the right column (defaults to second runner in the payload)",
    )
    eval_suite_report_parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional path to write the rendered Markdown report",
    )
    eval_suite_report_parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write a case-level comparison CSV",
    )
    eval_suite_report_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary instead of Markdown",
    )

    eval_suite_archive_parser = subparsers.add_parser(
        "eval-suite-archive",
        help="Copy an eval-suite output tree into the durable local archive registry",
    )
    eval_suite_archive_parser.add_argument(
        "source_output",
        type=Path,
        help="Path to an eval-suite output directory containing suite-run.json",
    )
    eval_suite_archive_parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help=(
            "Destination root for archived suite outputs "
            "(defaults to AXIOM_ENCODE_EVAL_ARCHIVE_ROOT or ./artifacts/eval-suites)"
        ),
    )
    eval_suite_archive_parser.add_argument(
        "--name",
        default=None,
        help="Optional archive directory name (defaults to the source directory name)",
    )
    eval_suite_archive_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable archive metadata",
    )

    # =========================================================================
    # Session logging commands (for hooks)
    # =========================================================================

    # session-start command
    session_start_parser = subparsers.add_parser(
        "session-start", help="Start a new session (called by SessionStart hook)"
    )
    session_start_parser.add_argument("--model", default="", help="Model name")
    session_start_parser.add_argument("--cwd", default="", help="Working directory")
    session_start_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # session-end command
    session_end_parser = subparsers.add_parser(
        "session-end", help="End a session (called by SessionEnd hook)"
    )
    session_end_parser.add_argument("--session", required=True, help="Session ID")
    session_end_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # log-event command
    log_event_parser = subparsers.add_parser(
        "log-event", help="Log an event to a session (called by hooks)"
    )
    log_event_parser.add_argument("--session", required=True, help="Session ID")
    log_event_parser.add_argument("--type", required=True, help="Event type")
    log_event_parser.add_argument(
        "--tool", default=None, help="Tool name (for tool events)"
    )
    log_event_parser.add_argument("--content", default="", help="Event content")
    log_event_parser.add_argument("--metadata", default="{}", help="Metadata as JSON")
    log_event_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # sessions command
    sessions_parser = subparsers.add_parser("sessions", help="List recent sessions")
    sessions_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sessions_parser.add_argument("--limit", type=int, default=20)

    # session-show command
    session_show_parser = subparsers.add_parser(
        "session-show", help="Show a session transcript"
    )
    session_show_parser.add_argument("session_id", help="Session ID")
    session_show_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    session_show_parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    # session-stats command
    session_stats_parser = subparsers.add_parser(
        "session-stats", help="Show session statistics"
    )
    session_stats_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # =========================================================================
    # Transcript sync commands
    # =========================================================================

    # sync-transcripts command
    sync_transcripts_parser = subparsers.add_parser(
        "sync-transcripts", help="Sync local transcripts to Supabase"
    )
    sync_transcripts_parser.add_argument(
        "--session", default=None, help="Only sync specific session"
    )

    # transcript-stats command
    subparsers.add_parser(
        "transcript-stats", help="Show local transcript database stats"
    )

    # sync-agent-sessions command
    sync_agent_parser = subparsers.add_parser(
        "sync-agent-sessions", help="Sync agent sessions to Supabase"
    )
    sync_agent_parser.add_argument(
        "--session", default=None, help="Only sync specific session"
    )
    sync_agent_parser.add_argument(
        "--all",
        action="store_true",
        help="Sync every local session, including broad agent history",
    )

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "proof-validate":
        cmd_proof_validate(args)
    elif args.command == "compile":
        cmd_compile(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "inventory":
        cmd_inventory(args)
    elif args.command == "oracle-coverage":
        cmd_oracle_coverage(args)
    elif args.command == "oracle-candidates":
        cmd_oracle_candidates(args)
    elif args.command == "snap-ecps-compare":
        sys.exit(run_snap_ecps_compare(args))
    elif args.command == "tax-ecps-compare":
        sys.exit(run_tax_ecps_compare(args))
    elif args.command == "snap-readiness":
        cmd_snap_readiness(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "concepts-audit":
        cmd_concepts_audit(args)
    elif args.command == "guard-generated":
        cmd_guard_generated(args)
    elif args.command == "repair-nonnegative-floors":
        cmd_repair_nonnegative_floors(args)
    elif args.command == "repair-current-year-final-amounts":
        cmd_repair_current_year_final_amounts(args)
    elif args.command == "repair-zero-branch-tests":
        cmd_repair_zero_branch_tests(args)
    elif args.command == "repair-unused-imports":
        cmd_repair_unused_imports(args)
    elif args.command == "repair-proof-import-hashes":
        cmd_repair_proof_import_hashes(args)
    elif args.command == "repair-unreferenced-proof-imports":
        cmd_repair_unreferenced_proof_imports(args)
    elif args.command == "repair-imported-test-inputs":
        cmd_repair_imported_test_inputs(args)
    elif args.command == "repair-oracle-parameter-tests":
        cmd_repair_oracle_parameter_tests(args)
    elif args.command == "repair-tax-filing-status-branches":
        cmd_repair_tax_filing_status_branches(args)
    elif args.command == "repair-tax-status-components":
        cmd_repair_tax_status_components(args)
    elif args.command == "repair-missing-source-proofs":
        cmd_repair_missing_source_proofs(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "eval-source":
        cmd_eval_source(args)
    elif args.command == "eval-suite":
        cmd_eval_suite(args)
    elif args.command == "eval-suite-revalidate":
        cmd_eval_suite_revalidate(args)
    elif args.command == "eval-suite-report":
        cmd_eval_suite_report(args)
    elif args.command == "eval-suite-archive":
        cmd_eval_suite_archive(args)
    elif args.command == "session-start":
        cmd_session_start(args)
    elif args.command == "session-end":
        cmd_session_end(args)
    elif args.command == "log-event":
        cmd_log_event(args)
    elif args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "session-show":
        cmd_session_show(args)
    elif args.command == "session-stats":
        cmd_session_stats(args)
    elif args.command == "sync-transcripts":
        cmd_sync_transcripts(args)
    elif args.command == "transcript-stats":
        cmd_transcript_stats(args)
    elif args.command == "sync-agent-sessions":
        cmd_sync_agent_sessions(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_validate(args):
    """Validate a RuleSpec YAML file."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    rulespec_file = args.file.resolve()

    policy_repo_root, axiom_rules_path = _resolve_validation_repo_roots(rulespec_file)

    # Enable oracles if --oracle flag is set
    enable_oracles = args.oracle is not None
    oracle_validators = None
    if args.oracle == "policyengine":
        oracle_validators = ("policyengine",)
    elif args.oracle == "taxsim":
        oracle_validators = ("taxsim",)

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo_root,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=enable_oracles,
        oracle_validators=oracle_validators,
    )

    result = pipeline.validate(rulespec_file, skip_reviewers=args.skip_reviewers)
    scores = result.to_review_results()
    review_scores = (
        {
            "rulespec_reviewer": None,
            "formula_reviewer": None,
            "parameter_reviewer": None,
            "integration_reviewer": None,
        }
        if args.skip_reviewers
        else _reviewer_score_map(scores)
    )

    errors = []
    for name, vr in result.results.items():
        if vr.error:
            errors.append(f"{name}: {vr.error}")
    oracle_issues = None
    if args.oracle:
        oracle_issues = {}
        for name, vr in result.results.items():
            issues = getattr(vr, "issues", None)
            if name in {"policyengine", "taxsim"} and isinstance(issues, list):
                if issues:
                    oracle_issues[name] = issues
        if not oracle_issues:
            oracle_issues = None
    oracle_coverage = None
    if args.oracle:
        oracle_coverage = {}
        for name, vr in result.results.items():
            coverage = getattr(vr, "details", {}).get("coverage")
            if name in {"policyengine", "taxsim"} and isinstance(coverage, dict):
                oracle_coverage[name] = coverage
        if not oracle_coverage:
            oracle_coverage = None

    # Check oracle results against minimum match rate
    oracle_passed = True
    if args.oracle:
        min_match = args.min_match
        if args.oracle in ("policyengine", "all"):
            pe_result = result.results.get("policyengine")
            if pe_result and pe_result.score is not None:
                if pe_result.score < min_match:
                    oracle_passed = False
                    errors.append(
                        f"PolicyEngine: {pe_result.score:.1%} < {min_match:.0%} required"
                    )
        if args.oracle in ("taxsim", "all"):
            ts_result = result.results.get("taxsim")
            if ts_result and ts_result.score is not None:
                if ts_result.score < min_match:
                    oracle_passed = False
                    errors.append(
                        f"TAXSIM: {ts_result.score:.1%} < {min_match:.0%} required"
                    )
        if getattr(args, "require_oracle_classification", False) is True:
            if not oracle_coverage:
                oracle_passed = False
                errors.append(
                    "Oracle classification required but no oracle coverage was reported"
                )
            else:
                for name, coverage in oracle_coverage.items():
                    unmapped = int(coverage.get("unmapped", 0) or 0)
                    if unmapped:
                        oracle_passed = False
                        errors.append(
                            f"{name}: {unmapped} unclassified oracle output(s)"
                        )

    # Overall pass requires regular checks AND oracle checks (if enabled)
    all_passed = result.all_passed and oracle_passed

    if args.json:
        output = {
            "file": str(rulespec_file),
            "ci_pass": result.ci_pass,
            "scores": {
                "rulespec_reviewer": review_scores["rulespec_reviewer"],
                "formula_reviewer": review_scores["formula_reviewer"],
                "parameter_reviewer": review_scores["parameter_reviewer"],
                "integration_reviewer": review_scores["integration_reviewer"],
            },
            "oracle_scores": {
                "policyengine": scores.policyengine_match,
                "taxsim": scores.taxsim_match,
            }
            if args.oracle
            else None,
            "oracle_passed": oracle_passed if args.oracle else None,
            "oracle_issues": oracle_issues,
            "oracle_coverage": oracle_coverage,
            "all_passed": all_passed,
            "errors": errors,
            "duration_ms": result.total_duration_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {rulespec_file}")
        print(f"CI: {'✓' if result.ci_pass else '✗'}")
        if not args.skip_reviewers:
            print(
                "Scores: "
                f"RuleSpec {review_scores['rulespec_reviewer']}/10 | "
                f"Formula {review_scores['formula_reviewer']}/10 | "
                f"Param {review_scores['parameter_reviewer']}/10 | "
                f"Integration {review_scores['integration_reviewer']}/10"
            )
        if args.oracle:
            pe_score = scores.policyengine_match
            ts_score = scores.taxsim_match
            min_match = args.min_match
            if args.oracle in ("policyengine", "all") and pe_score is not None:
                status = "✓" if pe_score >= min_match else "✗"
                print(f"PolicyEngine: {status} {pe_score:.1%} (min: {min_match:.0%})")
            if args.oracle in ("taxsim", "all") and ts_score is not None:
                status = "✓" if ts_score >= min_match else "✗"
                print(f"TAXSIM: {status} {ts_score:.1%} (min: {min_match:.0%})")
            if oracle_coverage:
                for name, coverage in oracle_coverage.items():
                    print(
                        f"{name} coverage: comparable={coverage.get('comparable', 0)} "
                        f"passed={coverage.get('passed', 0)} "
                        f"failed={coverage.get('failed', 0)} "
                        f"unmapped={coverage.get('unmapped', 0)} "
                        f"unsupported={coverage.get('unsupported', 0)}"
                    )
        print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        if errors:
            for err in errors:
                print(f"  - {err}")
        if oracle_issues:
            for name, issues in oracle_issues.items():
                for issue in issues[:10]:
                    print(f"  - {name}: {issue}")
                if len(issues) > 10:
                    print(f"  - {name}: ... {len(issues) - 10} more oracle issue(s)")

    sys.exit(0 if all_passed else 1)


def cmd_proof_validate(args):
    """Validate explicit RuleSpec proof trees."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    rulespec_file = args.file.resolve()
    result = validate_rulespec_proofs(
        rulespec_file.read_text(encoding="utf-8"),
        validate_claim_records=True,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "file": str(rulespec_file),
                    "passed": result.passed,
                    "proof_required": result.proof_required,
                    "atoms_checked": result.atoms_checked,
                    "issues": result.issues,
                },
                indent=2,
            )
        )
    else:
        print(f"File: {rulespec_file}")
        print(f"Proofs required: {'yes' if result.proof_required else 'no'}")
        print(f"Atoms checked: {result.atoms_checked}")
        print(f"Result: {'✓ PASSED' if result.passed else '✗ FAILED'}")
        for issue in result.issues:
            print(f"  - {issue}")

    sys.exit(0 if result.passed else 1)


def cmd_test(args):
    """Execute RuleSpec companion .test.yaml cases."""
    root = Path(args.root).resolve()
    test_files = _discover_rulespec_test_files(args.paths, root=root)
    if not test_files:
        message = f"No RuleSpec companion tests found under {root}"
        if args.json:
            print(
                json.dumps(
                    {
                        "success": False,
                        "root": str(root),
                        "test_files": 0,
                        "cases": 0,
                        "failures": [{"file": None, "case": None, "message": message}],
                    },
                    indent=2,
                )
            )
        else:
            print(message)
        sys.exit(1)

    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(root)
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
    )
    binary = pipeline._axiom_rules_binary()
    rulespec_env = pipeline._rulespec_compile_env()

    failures: list[dict[str, str | None]] = []
    case_count = 0
    compiled_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        compiled_cache: dict[Path, tuple[Path, dict]] = {}
        for test_file in test_files:
            result = _execute_rulespec_test_file(
                test_file,
                binary=binary,
                axiom_rules_path=Path(axiom_rules_path),
                env=rulespec_env,
                tmp_path=tmp_path,
                compiled_cache=compiled_cache,
            )
            case_count += result["cases"]
            compiled_count += result["compiled"]
            failures.extend(result["failures"])

    payload = {
        "success": not failures,
        "root": str(root),
        "test_files": len(test_files),
        "cases": case_count,
        "compiled_programs": compiled_count,
        "failures": failures,
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    elif failures:
        print(
            f"RuleSpec companion tests failed: "
            f"{len(failures)} failure(s) across {case_count} case(s)"
        )
        for failure in failures[:50]:
            label = failure.get("file") or "<unknown file>"
            case_name = failure.get("case") or "<unknown case>"
            print(f"- {label} :: {case_name} :: {failure.get('message')}")
        if len(failures) > 50:
            print(f"- ... {len(failures) - 50} more failure(s)")
    else:
        print(
            f"RuleSpec companion tests passed: "
            f"{len(test_files)} file(s), {case_count} case(s)"
        )
    sys.exit(0 if not failures else 1)


_RULESPEC_TEST_DISCOVERY_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "_axiom",
        "axiom-rules-engine",
        "node_modules",
    }
)


def _discover_rulespec_test_files(paths: list[Path], *, root: Path) -> list[Path]:
    candidates = paths or [root]
    test_files: list[Path] = []
    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.is_dir():
            test_files.extend(
                file
                for file in path.rglob("*.test.yaml")
                if not _is_ignored_rulespec_test_discovery_path(file, root=root)
            )
            test_files.extend(
                file
                for file in path.rglob("*.test.yml")
                if not _is_ignored_rulespec_test_discovery_path(file, root=root)
            )
        elif _is_rulespec_test_file(
            path
        ) and not _is_ignored_rulespec_test_discovery_path(path, root=root):
            test_files.append(path)
    return sorted(set(test_files))


def _is_ignored_rulespec_test_discovery_path(path: Path, *, root: Path) -> bool:
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        parts = path.parts
    return any(part in _RULESPEC_TEST_DISCOVERY_IGNORED_PARTS for part in parts)


def _is_rulespec_test_file(path: Path) -> bool:
    return path.name.endswith(".test.yaml") or path.name.endswith(".test.yml")


def _rulespec_program_for_test_file(path: Path) -> Path:
    if path.name.endswith(".test.yaml"):
        return path.with_name(path.name.removesuffix(".test.yaml") + ".yaml")
    if path.name.endswith(".test.yml"):
        return path.with_name(path.name.removesuffix(".test.yml") + ".yml")
    raise ValueError(f"{path} is not a RuleSpec companion test file")


def _execute_rulespec_test_file(
    test_file: Path,
    *,
    binary: Path,
    axiom_rules_path: Path,
    env: dict[str, str],
    tmp_path: Path,
    compiled_cache: dict[Path, tuple[Path, dict]],
) -> dict:
    failures: list[dict[str, str | None]] = []
    program_file = _rulespec_program_for_test_file(test_file)
    if not program_file.exists():
        return {
            "cases": 0,
            "compiled": 0,
            "failures": [
                {
                    "file": str(test_file),
                    "case": None,
                    "message": f"missing adjacent RuleSpec file {program_file}",
                }
            ],
        }

    compiled = compiled_cache.get(program_file)
    compiled_count = 0
    if compiled is None:
        compiled_path = tmp_path / (_safe_artifact_stem(program_file) + ".json")
        result = subprocess.run(
            [
                str(binary),
                "compile",
                "--program",
                str(program_file),
                "--output",
                str(compiled_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(axiom_rules_path) if axiom_rules_path.exists() else None,
            env=env,
        )
        if result.returncode != 0:
            return {
                "cases": 0,
                "compiled": 0,
                "failures": [
                    {
                        "file": str(test_file),
                        "case": None,
                        "message": result.stderr.strip() or result.stdout.strip(),
                    }
                ],
            }
        artifact = json.loads(compiled_path.read_text())
        compiled_cache[program_file] = (compiled_path, artifact)
        compiled = (compiled_path, artifact)
        compiled_count = 1

    compiled_path, artifact = compiled
    cases = _load_rulespec_test_cases(test_file)
    parameter_by_id = {
        parameter.get("id"): parameter
        for parameter in artifact.get("program", {}).get("parameters", [])
        if isinstance(parameter, dict) and parameter.get("id")
    }
    derived_ids = {
        derived.get("id")
        for derived in artifact.get("program", {}).get("derived", [])
        if isinstance(derived, dict) and derived.get("id")
    }
    derived_by_id = {
        str(derived.get("id")): derived
        for derived in artifact.get("program", {}).get("derived", [])
        if isinstance(derived, dict) and derived.get("id")
    }

    for index, case in enumerate(cases):
        case_name = str(case.get("name") or f"case_{index}")
        try:
            failures.extend(
                _execute_rulespec_test_case(
                    test_file,
                    case,
                    case_name=case_name,
                    compiled_path=compiled_path,
                    binary=binary,
                    axiom_rules_path=axiom_rules_path,
                    env=env,
                    parameter_by_id=parameter_by_id,
                    derived_ids=derived_ids,
                    derived_by_id=derived_by_id,
                )
            )
        except Exception as error:
            failures.append(
                {"file": str(test_file), "case": case_name, "message": str(error)}
            )

    return {"cases": len(cases), "compiled": compiled_count, "failures": failures}


def _load_rulespec_test_cases(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    if isinstance(data, list):
        cases = data
    elif isinstance(data, dict) and isinstance(data.get("cases"), list):
        cases = data["cases"]
    else:
        raise ValueError(f"{path} must contain a case list or a mapping with `cases`")
    invalid = [index for index, case in enumerate(cases) if not isinstance(case, dict)]
    if invalid:
        raise ValueError(f"{path} contains non-mapping test case(s): {invalid}")
    return cases


def _execute_rulespec_test_case(
    test_file: Path,
    case: dict,
    *,
    case_name: str,
    compiled_path: Path,
    binary: Path,
    axiom_rules_path: Path,
    env: dict[str, str],
    parameter_by_id: dict[str, dict],
    derived_ids: set[str],
    derived_by_id: dict[str, dict],
) -> list[dict[str, str | None]]:
    failures: list[dict[str, str | None]] = []
    period = _rulespec_period_spec(case.get("period", "2026-01"))
    interval = {"start": period["start"], "end": period["end"]}
    root_entity_id = "case"
    inputs: list[dict] = []
    relations: list[dict] = []
    flat_inputs: dict[str, object] = {}

    for key, value in (case.get("input") or {}).items():
        key = str(key)
        flat_inputs[key] = value
        if isinstance(value, list) and "#relation." in key:
            for row_index, row in enumerate(value):
                related_id = f"related_{row_index}"
                # The current relation slot convention is related entity first,
                # enclosing entity second.
                relations.append(
                    {
                        "name": key,
                        "tuple": [related_id, root_entity_id],
                        "interval": interval,
                    }
                )
                if not isinstance(row, dict):
                    failures.append(
                        {
                            "file": str(test_file),
                            "case": case_name,
                            "message": f"relation row for {key} is not a mapping",
                        }
                    )
                    continue
                for row_key, row_value in row.items():
                    inputs.append(
                        {
                            "name": str(row_key),
                            "entity": "Entity",
                            "entity_id": related_id,
                            "interval": interval,
                            "value": _rulespec_scalar_value(row_value),
                        }
                    )
        else:
            inputs.append(
                {
                    "name": key,
                    "entity": "Entity",
                    "entity_id": root_entity_id,
                    "interval": interval,
                    "value": _rulespec_scalar_value(value),
                }
            )

    table_rows_by_entity: dict[str, list[tuple[str, dict]]] = {}
    tables = case.get("tables")
    if tables is not None:
        if not isinstance(tables, dict):
            failures.append(
                {
                    "file": str(test_file),
                    "case": case_name,
                    "message": "`tables` must be a mapping",
                }
            )
            return failures
        for table_entity, rows in tables.items():
            table_entity = str(table_entity)
            if not isinstance(rows, list):
                failures.append(
                    {
                        "file": str(test_file),
                        "case": case_name,
                        "message": f"`tables.{table_entity}` must be a list",
                    }
                )
                return failures
            resolved_rows: list[tuple[str, dict]] = []
            for row_index, row in enumerate(rows, 1):
                if not isinstance(row, dict):
                    failures.append(
                        {
                            "file": str(test_file),
                            "case": case_name,
                            "message": (
                                f"`tables.{table_entity}` row #{row_index} "
                                "must be a mapping"
                            ),
                        }
                    )
                    return failures
                row_id = _rulespec_table_row_entity_id(table_entity, row, row_index)
                resolved_rows.append((row_id, row))
                for row_key, row_value in row.items():
                    key = str(row_key)
                    flat_inputs[key] = row_value
                    inputs.append(
                        {
                            "name": key,
                            "entity": table_entity,
                            "entity_id": row_id,
                            "interval": interval,
                            "value": _rulespec_scalar_value(row_value),
                        }
                    )
            table_rows_by_entity[table_entity] = resolved_rows

    expected = case.get("output") or {}
    parameter_expected = {
        str(key): value
        for key, value in expected.items()
        if str(key) in parameter_by_id
    }
    derived_expected = {
        str(key): value
        for key, value in expected.items()
        if str(key) not in parameter_by_id
    }

    for output, expected_value in parameter_expected.items():
        try:
            actual_value = _rulespec_parameter_value(
                parameter_by_id[output], flat_inputs, period["start"]
            )
        except Exception as error:
            failures.append(
                {
                    "file": str(test_file),
                    "case": case_name,
                    "message": f"{output}: {error}",
                }
            )
            continue
        if not _rulespec_scalar_matches(actual_value, expected_value):
            failures.append(
                {
                    "file": str(test_file),
                    "case": case_name,
                    "message": (
                        f"{output}: expected {expected_value!r}, got {actual_value!r}"
                    ),
                }
            )

    unknown = [output for output in derived_expected if output not in derived_ids]
    for output in unknown:
        failures.append(
            {
                "file": str(test_file),
                "case": case_name,
                "message": f"unknown executable output {output}",
            }
        )
    derived_expected = {
        output: value
        for output, value in derived_expected.items()
        if output in derived_ids
    }
    if not derived_expected:
        return failures

    output_entities = {
        output: str(derived_by_id.get(output, {}).get("entity") or "Case")
        for output in derived_expected
    }
    list_outputs = {
        output
        for output, expected_value in derived_expected.items()
        if isinstance(expected_value, list)
    }
    query_entity = (
        output_entities[next(iter(list_outputs))]
        if list_outputs
        else output_entities[next(iter(derived_expected))]
    )
    if any(output_entities[output] != query_entity for output in list_outputs):
        failures.append(
            {
                "file": str(test_file),
                "case": case_name,
                "message": "row-ordered outputs must use one entity type",
            }
        )
        return failures
    table_rows = table_rows_by_entity.get(query_entity, [])
    query_entity_ids = [root_entity_id]
    if table_rows:
        query_entity_ids = [row_id for row_id, _row in table_rows]
        for output in list_outputs:
            expected_value = derived_expected[output]
            if len(expected_value) != len(table_rows):
                failures.append(
                    {
                        "file": str(test_file),
                        "case": case_name,
                        "message": (
                            f"{output}: expected {len(expected_value)} row "
                            f"value(s), but tables.{query_entity} has "
                            f"{len(table_rows)} row(s)"
                        ),
                    }
                )
                return failures
    elif list_outputs:
        failures.append(
            {
                "file": str(test_file),
                "case": case_name,
                "message": (
                    f"row-ordered output list requires `tables.{query_entity}` rows"
                ),
            }
        )
        return failures

    request = {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": relations},
        "queries": [
            {
                "entity_id": entity_id,
                "period": period,
                "outputs": list(derived_expected.keys()),
            }
            for entity_id in query_entity_ids
        ],
    }
    result = subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(compiled_path)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(axiom_rules_path) if axiom_rules_path.exists() else None,
        env=env,
    )
    if result.returncode != 0:
        failures.append(
            {
                "file": str(test_file),
                "case": case_name,
                "message": result.stderr.strip() or result.stdout.strip(),
            }
        )
        return failures

    results = json.loads(result.stdout)["results"]
    for output, expected_value in derived_expected.items():
        if isinstance(expected_value, list):
            for row_index, row_expected_value in enumerate(expected_value):
                outputs = results[row_index]["outputs"]
                actual = outputs.get(output)
                if actual is None:
                    failures.append(
                        {
                            "file": str(test_file),
                            "case": case_name,
                            "message": (
                                f"missing output {output}; got {sorted(outputs)}"
                            ),
                        }
                    )
                elif not _rulespec_output_matches(actual, row_expected_value):
                    failures.append(
                        {
                            "file": str(test_file),
                            "case": case_name,
                            "message": (
                                f"{output}[{row_index}]: expected "
                                f"{row_expected_value!r}, got {actual!r}"
                            ),
                        }
                    )
            continue
        outputs = results[0]["outputs"]
        actual = outputs.get(output)
        if actual is None:
            failures.append(
                {
                    "file": str(test_file),
                    "case": case_name,
                    "message": f"missing output {output}; got {sorted(outputs)}",
                }
            )
        elif not _rulespec_output_matches(actual, expected_value):
            failures.append(
                {
                    "file": str(test_file),
                    "case": case_name,
                    "message": f"{output}: expected {expected_value!r}, got {actual!r}",
                }
            )
    return failures


def _safe_artifact_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(path.resolve()))


def _rulespec_table_row_entity_id(
    table_entity: str,
    row: dict,
    index: int,
) -> str:
    entity_key = f"{_snake_case(table_entity)}_id"
    for key in ("entity_id", "id", entity_key):
        if key in row:
            return str(row[key])
    return f"{_snake_case(table_entity)}-{index}"


def _snake_case(value: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    second = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first)
    return re.sub(r"[^a-zA-Z0-9]+", "_", second).strip("_").lower()


def _rulespec_period_spec(value) -> dict[str, str]:
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        period = {
            "period_kind": str(value.get("period_kind", "period")),
            "start": start.isoformat() if hasattr(start, "isoformat") else str(start),
            "end": end.isoformat() if hasattr(end, "isoformat") else str(end),
        }
        if period["period_kind"] == "custom" and value.get("name") is not None:
            period["name"] = str(value["name"])
        return period
    if isinstance(value, int) or (
        isinstance(value, str) and len(value) == 4 and value.isdigit()
    ):
        year = int(value)
        return {
            "period_kind": "tax_year",
            "start": f"{year:04d}-01-01",
            "end": f"{year:04d}-12-31",
        }
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value):
        year = int(value[:4])
        month = int(value[5:])
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        next_start = date(next_year, next_month, 1)
        end = date.fromordinal(next_start.toordinal() - 1)
        return {
            "period_kind": "month",
            "start": f"{year:04d}-{month:02d}-01",
            "end": end.isoformat(),
        }
    if isinstance(value, date):
        return {
            "period_kind": "day",
            "start": value.isoformat(),
            "end": value.isoformat(),
        }
    raise ValueError(f"unsupported period shorthand: {value!r}")


def _rulespec_scalar_value(value) -> dict:
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return {"kind": "date", "value": value}
        return {"kind": "text", "value": value}
    raise ValueError(f"unsupported scalar value: {value!r}")


def _rulespec_parameter_value(
    parameter: dict, case_inputs: dict[str, object], period_start: str
) -> dict:
    versions = sorted(
        parameter.get("versions") or [], key=lambda item: str(item["effective_from"])
    )
    chosen = None
    for version in versions:
        if str(version["effective_from"]) <= period_start:
            chosen = version
    if chosen is None:
        if not versions:
            raise KeyError("no parameter versions")
        chosen = versions[0]
    values = chosen.get("values") or {}
    key = "0"
    indexed_by = parameter.get("indexed_by")
    if indexed_by:
        raw_index = _rulespec_case_input_value(case_inputs, indexed_by)
        if raw_index is None:
            raise KeyError(f"missing parameter index input {indexed_by}")
        if isinstance(raw_index, (int, float)) and not isinstance(raw_index, bool):
            key = str(int(raw_index))
        else:
            key = str(raw_index)
    if key not in values:
        raise KeyError(f"missing parameter value key {key}")
    return values[key]


def _rulespec_case_input_value(
    case_inputs: dict[str, object], reference: str
) -> object | None:
    reference = str(reference)
    candidates = [reference]
    fragment = reference.split("#", 1)[-1]
    if not fragment.startswith("input."):
        candidates.append(f"dummy#input.{fragment}")
    for candidate in candidates:
        if candidate in case_inputs:
            return case_inputs[candidate]
    suffix = f"#input.{fragment.removeprefix('input.')}"
    matches = [value for key, value in case_inputs.items() if str(key).endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


def _rulespec_scalar_matches(actual_value: dict, expected) -> bool:
    kind = actual_value.get("kind")
    value = actual_value.get("value")
    if (
        kind in {"integer", "decimal"}
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        from decimal import Decimal

        return Decimal(str(value)) == Decimal(str(expected))
    if kind == "date" and isinstance(expected, date):
        return value == expected.isoformat()
    if isinstance(expected, date):
        expected = expected.isoformat()
    return value == expected


def _rulespec_output_matches(actual: dict, expected) -> bool:
    if actual.get("kind") == "judgment":
        return actual.get("outcome") == expected
    return _rulespec_scalar_matches(actual.get("value") or {}, expected)


def cmd_compile(args):
    """Compile a RuleSpec YAML file with the Axiom rules engine."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    try:
        axiom_rules_path = (
            getattr(args, "axiom_rules_path", None)
            or (getattr(args, "axiom_rules_path", None))
            or _resolve_runtime_axiom_rules_checkout(find_policy_repo_root(args.file))
        )
        pipeline = ValidatorPipeline(
            policy_repo_path=find_policy_repo_root(args.file) or args.file.parent,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=False,
        )
        binary = pipeline._axiom_rules_binary()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled.json"
            result = subprocess.run(
                [
                    str(binary),
                    "compile",
                    "--program",
                    str(args.file),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(axiom_rules_path) if Path(axiom_rules_path).exists() else None,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            compiled = json.loads(output_path.read_text())

        program = compiled.get("program") if isinstance(compiled, dict) else {}
        if not isinstance(program, dict):
            program = {}
        rule_names = [
            item.get("name")
            for key in ("parameters", "derived", "relations")
            for item in (program.get(key) or [])
            if isinstance(item, dict) and item.get("name")
        ]

        if args.json:
            output = {
                "success": True,
                "file": str(args.file),
                "axiom_rules_path": str(axiom_rules_path),
                "rules": rule_names,
                "rule_count": len(rule_names),
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"Compiled: {args.file}")
            print(f"Axiom rules engine: {axiom_rules_path}")
            print(f"Rules: {len(rule_names)}")
            for name in rule_names:
                print(f"  - {name}")
            print("\nResult: compiled successfully")

        sys.exit(0)

    except Exception as e:
        if args.json:
            output = {
                "success": False,
                "file": str(args.file),
                "error": str(e),
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Compilation failed: {e}")

        sys.exit(1)


def cmd_log(args):
    """Log an encoding run."""
    db = EncodingDB(args.db)

    # Parse errors
    errors_data = json.loads(args.errors) if args.errors else []
    iteration_errors = [
        IterationError(
            error_type=e.get("type", "other"),
            message=e.get("message", ""),
            variable=e.get("variable"),
            fix_applied=e.get("fix"),
        )
        for e in errors_data
    ]

    # Build iterations (simplified: all errors in iteration 1, success in last)
    iterations = []
    for i in range(1, args.iterations + 1):
        is_last = i == args.iterations
        iterations.append(
            Iteration(
                attempt=i,
                duration_ms=args.duration // args.iterations,
                errors=iteration_errors if i == 1 else [],
                success=is_last,
            )
        )

    # Parse review results from --scores.
    review_results = None
    if args.scores:
        s = json.loads(args.scores)
        reviews = []
        for reviewer_name, key in [
            ("rulespec_reviewer", "rulespec"),
            ("formula_reviewer", "formula"),
            ("parameter_reviewer", "param"),
            ("integration_reviewer", "integration"),
        ]:
            score = float(s.get(key, 0))
            reviews.append(
                ReviewResult(
                    reviewer=reviewer_name,
                    passed=score >= 7.0,
                    items_checked=10,
                    items_passed=int(score),
                )
            )
        review_results = ReviewResults(reviews=reviews)

    # Read RuleSpec content.
    rulespec_content = ""
    if args.file.exists():
        rulespec_content = args.file.read_text()

    run = EncodingRun(
        citation=args.citation,
        file_path=str(args.file),
        review_results=review_results,
        iterations=iterations,
        total_duration_ms=args.duration,
        rulespec_content=rulespec_content,
        session_id=args.session,
    )

    db.log_run(run)

    print(f"Logged: {run.id}")
    print(f"  Citation: {args.citation}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Duration: {args.duration}ms")
    if args.session:
        print(f"  Session: {args.session}")
    if review_results:
        passed = sum(1 for r in review_results.reviews if r.passed)
        total = len(review_results.reviews)
        print(f"  Reviews: {passed}/{total} passed")


def cmd_stats(args):
    """Show encoding statistics."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        print("Run some encodings first to collect data.")
        sys.exit(1)

    db = EncodingDB(args.db)

    # Iteration stats
    iter_stats = db.get_iteration_stats()
    print("=== Iteration Statistics ===")
    print(f"Total runs: {iter_stats['total_runs']}")
    print(f"Average iterations: {iter_stats['average']:.1f}")
    print(f"First-try success rate: {iter_stats['first_try_rate']:.0f}%")
    print(f"Distribution: {iter_stats['distribution']}")
    print()

    # Error stats
    error_stats = db.get_error_stats()
    print("=== Error Statistics ===")
    print(f"Total errors: {error_stats['total_errors']}")
    if error_stats["counts"]:
        print("By type:")
        for error_type, count in sorted(
            error_stats["counts"].items(), key=lambda x: -x[1]
        ):
            pct = error_stats["percentages"][error_type]
            print(f"  {error_type}: {count} ({pct:.0f}%)")
    print()

    # Improvement suggestions
    print("=== Improvement Suggestions ===")
    if error_stats["counts"]:
        top_error = max(error_stats["counts"].items(), key=lambda x: x[1])
        print(f"Focus on: {top_error[0]} errors ({top_error[1]} occurrences)")
        if top_error[0] == "test":
            print("  → Add more RuleSpec test examples")
        elif top_error[0] == "parse":
            print("  → Clarify RuleSpec syntax")
        elif top_error[0] == "import":
            print("  → Document import patterns better")
    else:
        print("Not enough data yet. Run more encodings.")


def cmd_inventory(args):
    """Show RuleSpec inventory across rulespec-* repos."""
    inventory = build_rulespec_inventory(args.root)

    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return

    print("RuleSpec inventory")
    print(f"Root: {inventory['root']}")
    print(
        "Files: "
        f"{inventory['total_files']} total; "
        f"{inventory['source_provision_files']} source/provision; "
        f"{inventory['composition_files']} composition"
    )
    print(f"Rules: {inventory['total_rules']}")
    print(f"Kinds: {_format_counter(inventory['kind_counts'])}")
    print()

    for repo in inventory["repos"]:
        print(
            f"{repo['repo']}: "
            f"files={repo['files']} "
            f"source/provision={repo['source_provision_files']} "
            f"composition={repo['composition_files']} "
            f"rules={repo['rules']} "
            f"roots={_format_counter(repo['roots'])}"
        )


def cmd_oracle_coverage(args):
    """Show oracle coverage classification across rulespec-* repos."""
    if args.oracle != "policyengine":
        print(f"Unsupported oracle: {args.oracle}")
        sys.exit(2)

    root = (args.root or _default_rulespec_inventory_root()).resolve()
    report = build_policyengine_coverage_report(root, program=args.program)

    unmapped = int(report["status_counts"].get("unmapped", 0))
    untested_comparable = int(report.get("untested_comparable", 0))
    should_fail = (args.fail_on_unmapped and unmapped) or (
        args.fail_on_untested_comparable and untested_comparable
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        sys.exit(1 if should_fail else 0)

    print("PolicyEngine oracle coverage")
    print(f"Root: {report['root']}")
    if args.program:
        print(f"Program: {args.program}")
    print(f"Executable outputs: {report['total_outputs']}")
    print(f"Status: {_format_counter(report['status_counts'])}")
    print(f"Untested comparable outputs: {untested_comparable}")
    print(f"Programs: {_format_counter(report['program_counts'])}")
    print()

    for repo in report["repos"]:
        print(
            f"{repo['repo']}: "
            f"outputs={repo['total_outputs']} "
            f"status={_format_counter(repo['status_counts'])}"
        )

    unmapped_items = [
        item for item in report["items"] if item.get("status") == "unmapped"
    ]
    if unmapped_items:
        print()
        print(f"Unmapped outputs (first {args.limit}):")
        for item in unmapped_items[: args.limit]:
            print(f"  - {item['legal_id']}")
        if len(unmapped_items) > args.limit:
            print(f"  - ... {len(unmapped_items) - args.limit} more")

    untested_items = [
        item
        for item in report["items"]
        if item.get("status") == "comparable" and not item.get("tested")
    ]
    if untested_items:
        print()
        print(f"Untested comparable outputs (first {args.limit}):")
        for item in untested_items[: args.limit]:
            print(f"  - {item['legal_id']}")
        if len(untested_items) > args.limit:
            print(f"  - ... {len(untested_items) - args.limit} more")

    sys.exit(1 if should_fail else 0)


def cmd_oracle_candidates(args):
    """Show candidate outputs for expanding oracle coverage."""
    if args.oracle != "policyengine":
        print(f"Unsupported oracle: {args.oracle}")
        sys.exit(2)

    root = (args.root or _default_rulespec_inventory_root()).resolve()
    report = build_policyengine_candidate_report(root, program=args.program)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        sys.exit(0)

    print("PolicyEngine oracle candidates")
    print(f"Root: {report['root']}")
    if args.program:
        print(f"Program: {args.program}")
    print(f"Candidates: {report['total_candidates']}")
    print(f"Categories: {_format_counter(report['category_counts'])}")
    print(f"Priorities: {_format_counter(report['priority_counts'])}")
    print(f"Coverage status: {_format_counter(report['coverage_status_counts'])}")
    if not report.get("policyengine_variables_available"):
        print("PolicyEngine variables: unavailable; exact-variable hints omitted")

    items = report["items"]
    if items:
        print()
        print(f"Top candidates (first {args.limit}):")
        for item in items[: args.limit]:
            target = item.get("policyengine_variable") or item.get(
                "policyengine_parameter"
            )
            target_suffix = f" -> {target}" if target else ""
            tested = "tested" if item.get("tested") else "untested"
            print(
                f"  - [{item['priority']}] {item['category']} "
                f"{item['legal_id']}{target_suffix} ({tested})"
            )
            print(f"    {item['recommendation']}")
            if item.get("rationale"):
                print(f"    current rationale: {item['rationale']}")
        if len(items) > args.limit:
            print(f"  - ... {len(items) - args.limit} more")

    sys.exit(0)


def cmd_snap_readiness(args):
    """Report SNAP corpus, RuleSpec, and ECPS oracle readiness."""
    root = (args.root or _default_rulespec_inventory_root()).resolve()
    corpus_root = (
        args.corpus_root.resolve() if args.corpus_root else root / "axiom-corpus"
    )
    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("SNAP encoding readiness")
    print(f"Root: {report['root']}")
    print(f"Corpus: {report['corpus_root']}")
    print(f"Repos: {report['total_repos']}")
    print(f"Status: {_format_counter(report['status_counts'])}")
    print()

    for item in report["items"]:
        blockers = item.get("blockers") or []
        blocker_suffix = f" blockers={'; '.join(blockers)}" if blockers else ""
        print(
            f"{item['repo']}: "
            f"status={item['status']} "
            f"corpus={item['corpus_snap_provisions']} "
            f"files={item['rulespec_files']} "
            f"outputs={item['executable_outputs']} "
            f"tests={item['companion_test_files']} "
            f"ecps_config={str(item['policyengine_ecps_configured']).lower()} "
            f"program_module={str(item['program_module_exists']).lower()}"
            f"{blocker_suffix}"
        )


def cmd_calibration(args):
    """Show review results summary across recent runs."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = EncodingDB(args.db)
    runs = db.get_recent_runs(limit=args.limit)

    # Filter to runs with review results
    runs_with_reviews = [r for r in runs if r.review_results]

    if not runs_with_reviews:
        print("No runs with review results yet.")
        print("Use --scores when logging runs to enable review summaries.")
        return

    print("=== Calibration Report ===\n")
    print(f"Runs with reviews: {len(runs_with_reviews)}")
    print()

    # Per-reviewer pass rates
    reviewer_stats: dict[str, list[bool]] = {}
    for run in runs_with_reviews:
        for review in run.review_results.reviews:
            reviewer_stats.setdefault(review.reviewer, []).append(review.passed)

    print("Reviewer Pass Rates:")
    print("-" * 50)
    print(f"{'Reviewer':<25} {'Passed':>8} {'Total':>8} {'Rate':>8}")
    print("-" * 50)

    for reviewer, results in sorted(reviewer_stats.items()):
        passed = sum(1 for r in results if r)
        total = len(results)
        rate = passed / total * 100 if total > 0 else 0
        print(f"{reviewer:<25} {passed:>8} {total:>8} {rate:>7.0f}%")

    print()

    # Per-run breakdown
    print("Per-Run Breakdown:")
    print("-" * 70)
    print(f"{'Citation':<25} {'Passed':>8} {'Total':>8} {'Crit':>8} {'Iter':>6}")
    print("-" * 70)

    for run in runs_with_reviews[-10:]:  # Last 10
        rr = run.review_results
        passed = sum(1 for r in rr.reviews if r.passed)
        total = len(rr.reviews)
        critical = rr.total_critical_issues
        citation = run.citation[:25]
        print(
            f"{citation:<25} {passed:>8} {total:>8} {critical:>8} {run.iterations_needed:>6}"
        )


def cmd_runs(args):
    """List recent runs."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = EncodingDB(args.db)
    runs = db.get_recent_runs(limit=args.limit)

    if not runs:
        print("No encoding runs found.")
        return

    print(f"{'ID':<10} {'Citation':<30} {'Iter':<5} {'Time':<8} {'Result'}")
    print("-" * 70)

    for run in runs:
        result = "✓" if run.success else "✗"
        time_s = run.total_duration_ms / 1000
        print(
            f"{run.id:<10} {run.citation:<30} {run.iterations_needed:<5} {time_s:>6.1f}s {result}"
        )


# =========================================================================
# Canonical Concept Audit
# =========================================================================


def cmd_concepts_audit(args):
    """Walk one or more rules/rulespec roots and report concept drift."""
    registry = load_concept_registry()
    path_filter = (
        re.compile(args.path_filter, re.IGNORECASE) if args.path_filter else None
    )
    findings = audit_concept_corpus(
        [Path(r).expanduser() for r in args.roots],
        registry,
        path_filter=path_filter,
        name_prefixes=tuple(args.name_prefix) if args.name_prefix else None,
    )
    if args.json:
        payload = [
            {
                "kind": f.kind,
                "name": f.name,
                "anchor": f.anchor,
                "site_paths": [str(p) for p in f.site_paths],
                "detail": f.detail,
                "nearby_producers": list(f.nearby_producers),
            }
            for f in findings
        ]
        print(json.dumps({"findings": payload, "count": len(findings)}, indent=2))
        return
    by_kind: dict[str, list] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)
    print(f"Concept-drift findings: {len(findings)}")
    for kind in (
        "blocked_synonym",
        "canonical_conflict",
        "anchored_ref_miss",
        "missing_producer",
    ):
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"\n[{kind}] {len(items)}")
        for f in items:
            anchor = f" @ {f.anchor}" if f.anchor else ""
            nearby = (
                f" (nearby: {', '.join(f.nearby_producers[:4])})"
                if f.nearby_producers
                else ""
            )
            print(f"  - {f.name}{anchor} — {len(f.site_paths)} site(s){nearby}")


# =========================================================================
# Generated RuleSpec Guard
# =========================================================================


def cmd_guard_generated(args):
    """Reject protected RuleSpec changes that lack an apply manifest."""
    repo_path = Path(args.repo).resolve()
    roots = tuple(str(args.roots).split())
    all_files = bool(getattr(args, "all", False))
    issues = guard_generated_change_issues(
        repo_path,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        roots=roots,
        all_files=all_files,
    )
    payload = {"repo": str(repo_path), "passed": not issues, "issues": issues}
    if args.json:
        print(json.dumps(payload, indent=2))
    elif issues:
        print("Manual RuleSpec changes are not allowed.")
        for issue in issues:
            print(f"- {issue}")
    else:
        if all_files:
            print("All RuleSpec files have encoder apply manifests.")
        else:
            print("All changed RuleSpec files have encoder apply manifests.")
    sys.exit(0 if not issues else 1)


def cmd_repair_nonnegative_floors(args):
    """Apply a signed deterministic zero-floor repair to a RuleSpec file."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    original_content = rules_file.read_text()
    test_file = _rulespec_test_path(rules_file)
    original_test_content = test_file.read_text() if test_file.exists() else None
    repaired_content, repaired_rules = repair_nonnegative_amount_reductions(
        original_content
    )
    if repaired_content == original_content:
        print("No nonnegative floor repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(repaired_content)

        rules_file.write_text(repaired_content)
        applied_files = [rules_file]
        repaired_test_cases = _append_generated_zero_branch_tests_if_missing(
            rules_file=rules_file,
            test_file=test_file,
            repo_path=repo_path,
            relative_output=relative_output,
        )
        if repaired_test_cases:
            applied_files.append(test_file)
        validation = ValidatorPipeline(
            policy_repo_path=repo_path,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=False,
            require_policy_proofs=True,
        ).validate(rules_file, skip_reviewers=True)
        if not validation.all_passed:
            rules_file.write_text(original_content)
            if original_test_content is not None:
                test_file.write_text(original_test_content)
            issues = [
                result.error for result in validation.results.values() if result.error
            ]
            print("Repair failed validation; restored original file.")
            for issue in issues:
                print(f"- {issue}")
            sys.exit(1)

        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="nonnegative-floor-v1",
            tool="axiom-encode repair-nonnegative-floors",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=applied_files,
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied nonnegative floor repair to "
        f"{relative_output}: {', '.join(repaired_rules)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_current_year_final_amounts(args):
    """Apply signed deterministic repairs for imported final amount tables."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    original_content = rules_file.read_text()
    test_file = _rulespec_test_path(rules_file)
    original_test_content = test_file.read_text() if test_file.exists() else None
    repaired_content, repaired_rules = repair_current_year_final_amount_tables(
        original_content,
        rules_file=rules_file,
        policy_repo_path=repo_path,
    )
    if repaired_content == original_content:
        print("No current-year final amount repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(repaired_content)

        rules_file.write_text(repaired_content)
        applied_files = [rules_file]
        if _repair_current_year_final_amount_test_expectations(
            rules_file=rules_file,
            test_file=test_file,
            repo_path=repo_path,
            relative_output=relative_output,
        ):
            applied_files.append(test_file)
        validation = ValidatorPipeline(
            policy_repo_path=repo_path,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=False,
            require_policy_proofs=True,
        ).validate(rules_file, skip_reviewers=True)
        validation_issues = [
            result.error for result in validation.results.values() if result.error
        ]
        if (
            not validation.all_passed
            and not _only_pending_nonnegative_amount_reduction_issues(validation_issues)
        ):
            rules_file.write_text(original_content)
            if original_test_content is not None:
                test_file.write_text(original_test_content)
            print("Repair failed validation; restored original file.")
            for issue in validation_issues:
                print(f"- {issue}")
            sys.exit(1)

        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="current-year-final-amount-v1",
            tool="axiom-encode repair-current-year-final-amounts",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=applied_files,
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied current-year final amount repair to "
        f"{relative_output}: {', '.join(repaired_rules)}"
    )
    print(f"manifest={manifest_path}")


def _repair_current_year_final_amount_test_expectations(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
) -> bool:
    if not test_file.exists():
        return False
    try:
        payload = yaml.safe_load(rules_file.read_text())
    except (OSError, yaml.YAMLError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False

    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    specs = _current_year_final_amount_test_specs(payload, repo_path=repo_path)
    if not specs:
        return False

    original = test_file.read_text()
    lines = original.splitlines(keepends=True)
    case_starts = [
        index for index, line in enumerate(lines) if re.match(r"^- name:\s*", line)
    ]
    if not case_starts:
        return False

    changed = False
    output = lines[:]
    for case_start, next_case_start in zip(
        case_starts,
        case_starts[1:] + [len(output)],
        strict=False,
    ):
        block = "".join(output[case_start:next_case_start])
        for spec in specs:
            index_match = re.search(
                rf"^\s*{re.escape(target_base)}#{re.escape(spec['index_rule'])}:\s*(?P<index>[0-9]+)\s*$",
                block,
                flags=re.MULTILINE,
            )
            if index_match is None:
                continue
            expected = spec["values"].get(index_match.group("index"))
            if expected is None:
                continue
            rule_key = f"{target_base}#{spec['rule_name']}"
            for line_index in range(case_start, next_case_start):
                line = output[line_index]
                match = re.match(
                    rf"^(?P<prefix>\s*{re.escape(rule_key)}:\s*)"
                    r"(?P<value>-?[0-9]+(?:\.[0-9]+)?)"
                    r"(?P<suffix>\s*)$",
                    line,
                )
                if match is None or match.group("value") == expected:
                    continue
                newline = "\n" if line.endswith("\n") else ""
                output[line_index] = (
                    f"{match.group('prefix')}{expected}{match.group('suffix').rstrip()}"
                    f"{newline}"
                )
                changed = True

    if changed:
        test_file.write_text("".join(output))
    return changed


def _current_year_final_amount_test_specs(
    payload: dict,
    *,
    repo_path: Path,
) -> list[dict[str, object]]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    specs: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name") or "").strip()
        versions = rule.get("versions")
        if not rule_name or not isinstance(versions, list) or not versions:
            continue
        first_version = versions[0]
        if not isinstance(first_version, dict):
            continue
        formula = first_version.get("formula")
        if not isinstance(formula, str):
            continue
        match_header = re.match(
            r"^match\s+(?P<index>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*\n(?P<body>.*)\Z",
            formula.strip(),
            flags=re.DOTALL,
        )
        if match_header is None:
            continue
        table_names = {
            table_name
            for _key, table_name in re.findall(
                r"^\s*([0-9]+)\s*=>\s*([A-Za-z_][A-Za-z0-9_]*)\[\1\]\s*$",
                match_header.group("body"),
                flags=re.MULTILINE,
            )
        }
        if len(table_names) != 1:
            continue
        table_name = next(iter(table_names))
        values = _imported_parameter_values(payload, table_name, repo_path=repo_path)
        if values:
            specs.append(
                {
                    "rule_name": rule_name,
                    "index_rule": match_header.group("index"),
                    "values": values,
                }
            )
    return specs


def _imported_parameter_values(
    payload: dict,
    table_name: str,
    *,
    repo_path: Path,
) -> dict[str, str]:
    imports = payload.get("imports")
    if not isinstance(imports, list):
        return {}

    jurisdiction = _repo_jurisdiction_prefix(repo_path)
    for raw_import in imports:
        if not isinstance(raw_import, str):
            continue
        target = raw_import.split("#", 1)[0].strip().strip("/")
        if target.startswith(f"{jurisdiction}:"):
            target = target.split(":", 1)[1].strip().strip("/")
        if not target:
            continue
        relative = Path(
            target if target.endswith((".yaml", ".yml")) else f"{target}.yaml"
        )
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            continue
        target_file = repo_path / relative
        if not target_file.exists():
            continue
        try:
            imported_payload = yaml.safe_load(target_file.read_text())
        except (OSError, yaml.YAMLError, ValueError):
            continue
        if not isinstance(imported_payload, dict):
            continue
        rules = imported_payload.get("rules")
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if str(rule.get("name") or "").strip() != table_name:
                continue
            versions = rule.get("versions")
            if not isinstance(versions, list) or not versions:
                return {}
            first_version = versions[0]
            if not isinstance(first_version, dict):
                return {}
            values = first_version.get("values")
            if not isinstance(values, dict):
                return {}
            return {str(key): str(value) for key, value in values.items()}
    return {}


def cmd_repair_zero_branch_tests(args):
    """Apply signed deterministic zero-branch companion test repairs."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    test_file = _rulespec_test_path(rules_file)
    if not test_file.exists():
        print(f"Companion test file not found: {test_file}")
        sys.exit(1)

    original_test_content = test_file.read_text()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    initial_validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    initial_issues = [
        result.error for result in initial_validation.results.values() if result.error
    ]
    repaired_test_cases = _append_generated_zero_branch_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=repo_path,
        relative_output=relative_output,
        issues=initial_issues,
    )
    if not repaired_test_cases:
        print("No zero-branch test repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()

    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        test_file.write_text(original_test_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original test file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="zero-branch-test-v1",
            tool="axiom-encode repair-zero-branch-tests",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file, test_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied zero-branch test repair to "
        f"{relative_output}: {', '.join(repaired_test_cases)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_unused_imports(args):
    """Apply signed deterministic unused-import repairs."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    original_content = rules_file.read_text()
    repaired_content, removed_imports = _prune_unused_imports(original_content)
    test_file = _rulespec_test_path(rules_file)
    manifest_needs_refresh = test_file.exists() and _applied_manifest_missing_file(
        repo_path,
        relative_output,
        test_file,
    )
    if not removed_imports and not manifest_needs_refresh:
        print("No unused imports found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    if removed_imports:
        rules_file.write_text(repaired_content)

    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        rules_file.write_text(original_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original RuleSpec file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="unused-import-prune-v1",
            tool="axiom-encode repair-unused-imports",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=(
                [rules_file, test_file] if test_file.exists() else [rules_file]
            ),
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    if removed_imports:
        print(
            "Applied unused-import repair to "
            f"{relative_output}: {', '.join(removed_imports)}"
        )
    else:
        print(f"Refreshed unused-import repair manifest for {relative_output}")
    print(f"manifest={manifest_path}")


def _applied_manifest_missing_file(
    repo_path: Path,
    relative_output: Path,
    applied_file: Path,
) -> bool:
    manifest_path = repo_path / _applied_encoding_manifest_path(relative_output)
    if not manifest_path.exists():
        return True
    with contextlib.suppress(json.JSONDecodeError, OSError, ValueError):
        payload = json.loads(manifest_path.read_text())
        applied_files = payload.get("applied_files")
        if not isinstance(applied_files, list):
            return True
        relative = applied_file.relative_to(repo_path).as_posix()
        return not any(
            isinstance(item, dict) and item.get("path") == relative
            for item in applied_files
        )
    return True


def _prune_unused_imports(content: str) -> tuple[str, list[str]]:
    unused_imports = _unused_import_items(content)
    if not unused_imports:
        return content, []

    unused = set(unused_imports)
    repaired_lines: list[str] = []
    in_imports = False
    imports_indent = 0
    removed: list[str] = []

    for line in content.splitlines(keepends=True):
        imports_match = re.match(r"^(\s*)imports:\s*$", line)
        if imports_match:
            in_imports = True
            imports_indent = len(imports_match.group(1))
            repaired_lines.append(line)
            continue

        if in_imports:
            stripped = line.strip()
            if stripped:
                indent = len(line) - len(line.lstrip())
                if indent <= imports_indent:
                    in_imports = False
                else:
                    item_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
                    if item_match:
                        item = _strip_yaml_scalar_quotes(item_match.group(1).strip())
                        if item in unused:
                            removed.append(item)
                            continue

        repaired_lines.append(line)

    return "".join(repaired_lines), removed


def _unused_import_items(content: str) -> list[str]:
    items: list[str] = []
    for issue in find_unused_import_issues(content):
        match = re.search(r"Unused import `([^`]+)`", issue)
        if match:
            items.append(match.group(1))
    return items


def _strip_yaml_scalar_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def cmd_repair_proof_import_hashes(args):
    """Apply signed deterministic repairs for proof import hashes."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    test_file = _rulespec_test_path(rules_file)
    original_test_content = test_file.read_text() if test_file.exists() else None
    original_content = rules_file.read_text()
    repaired_content, repair_count = _repair_proof_import_hashes(
        original_content,
        target_base=target_base,
        rules_file=rules_file,
        repo_path=repo_path,
    )
    if repaired_content == original_content:
        print("No proof hash repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    rules_file.write_text(repaired_content)
    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=False,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        if _complete_missing_imported_test_inputs(
            rules_file=rules_file,
            test_file=test_file,
            repo_path=repo_path,
            validation=validation,
        ):
            validation = ValidatorPipeline(
                policy_repo_path=repo_path,
                axiom_rules_path=axiom_rules_path,
                enable_oracles=False,
                require_policy_proofs=False,
            ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        if _only_pending_zero_branch_coverage_issues(issues):
            print(
                "Applied proof import hash repair with pending zero-branch coverage "
                "repair still required."
            )
        else:
            rules_file.write_text(original_content)
            if original_test_content is not None:
                test_file.write_text(original_test_content)
            print("Repair failed validation; restored original file.")
            for issue in issues:
                print(f"- {issue}")
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(repaired_content)
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="proof-import-hash-v1",
            tool="axiom-encode repair-proof-import-hashes",
            citation=target_base,
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        applied_files = [rules_file]
        if test_file.exists():
            applied_files.append(test_file)
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=applied_files,
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(f"Applied proof hash repair to {relative_output}: {repair_count}")
    print(f"manifest={manifest_path}")


def cmd_repair_unreferenced_proof_imports(args):
    """Apply signed deterministic repairs for stale proof import atoms."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    original_content = rules_file.read_text()
    issues = find_proof_import_reference_issues(original_content)
    if not issues:
        print("No unreferenced proof import repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    removed_refs = _remove_unreferenced_proof_import_atoms(
        rules_file=rules_file,
        issues=issues,
    )
    if not removed_refs:
        print("No unreferenced proof import repairs found.")
        return

    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        rules_file.write_text(original_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original RuleSpec file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="proof-import-reference-v1",
            tool="axiom-encode repair-unreferenced-proof-imports",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied unreferenced proof import repair to "
        f"{relative_output}: {', '.join(removed_refs)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_imported_test_inputs(args):
    """Apply signed deterministic repairs for missing imported test inputs."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    test_file = _rulespec_test_path(rules_file)
    if not test_file.exists():
        print(f"Companion test file not found: {test_file}")
        sys.exit(1)

    original_test_content = test_file.read_text()
    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    pipeline = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=False,
    )
    validation = pipeline.validate(rules_file, skip_reviewers=True)
    if validation.all_passed:
        print("No imported test input repairs found.")
        return

    repaired = _complete_missing_imported_test_inputs(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=repo_path,
        validation=validation,
    )
    if not repaired:
        print("No imported test input repairs found.")
        return

    validation = pipeline.validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        test_file.write_text(original_test_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original test file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="imported-test-inputs-v1",
            tool="axiom-encode repair-imported-test-inputs",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file, test_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(f"Applied imported test input repair to {relative_output}")
    print(f"manifest={manifest_path}")


def cmd_repair_oracle_parameter_tests(args):
    """Apply signed deterministic companion tests for mapped parameter outputs."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    test_file = _rulespec_test_path(rules_file)
    original_test_content = test_file.read_text() if test_file.exists() else ""
    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    repaired_test_cases = _append_oracle_parameter_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        target_base=target_base,
    )
    if not repaired_test_cases:
        print("No oracle parameter test repairs found.")
        return

    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        test_file.write_text(original_test_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original test file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    test_failures = _rulespec_companion_test_failures(
        test_file,
        root=repo_path,
        axiom_rules_path=axiom_rules_path,
    )
    if test_failures:
        test_file.write_text(original_test_content)
        print("Repair failed companion tests; restored original test file.")
        for failure in test_failures[:20]:
            case_name = failure.get("case") or "<unknown case>"
            print(f"- {case_name}: {failure.get('message')}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="oracle-parameter-test-v1",
            tool="axiom-encode repair-oracle-parameter-tests",
            citation=target_base,
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file, test_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied oracle parameter test repair to "
        f"{relative_output}: {', '.join(repaired_test_cases)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_tax_filing_status_branches(args):
    """Apply signed deterministic repairs for US tax filing-status enum branches."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    test_file = _rulespec_test_path(rules_file)
    original_content = rules_file.read_text()
    repaired_content, repaired_rules = _repair_tax_filing_status_branches(
        original_content
    )
    original_test_content = test_file.read_text() if test_file.exists() else None
    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()

    if repaired_content == original_content:
        print("No tax filing-status branch repairs found.")
        return

    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    rules_file.write_text(repaired_content)
    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        rules_file.write_text(original_content)
        if original_test_content is not None:
            test_file.write_text(original_test_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(repaired_content)
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="tax-filing-status-branch-v1",
            tool="axiom-encode repair-tax-filing-status-branches",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied tax filing-status branch repair to "
        f"{relative_output}: {', '.join(repaired_rules)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_tax_status_components(args):
    """Apply signed deterministic repairs for local US tax status component facts."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    test_file = _rulespec_test_path(rules_file)
    original_content = rules_file.read_text()
    original_test_content = test_file.read_text() if test_file.exists() else None
    repaired_content, repaired_components = _repair_tax_status_component_local_inputs(
        original_content
    )
    repaired_test_content, removed_test_refs = _remove_tax_status_component_test_inputs(
        original_test_content,
        repaired_components=repaired_components,
    )
    if repaired_content == original_content and not removed_test_refs:
        print("No tax status component repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    rules_file.write_text(repaired_content)
    if (
        repaired_test_content is not None
        and repaired_test_content != original_test_content
    ):
        test_file.write_text(repaired_test_content)

    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        rules_file.write_text(original_content)
        if original_test_content is not None:
            test_file.write_text(original_test_content)
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        print("Repair failed validation; restored original RuleSpec file.")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(rules_file.read_text())
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="tax-status-component-v1",
            tool="axiom-encode repair-tax-status-components",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        applied_files = [rules_file]
        if (
            repaired_test_content is not None
            and repaired_test_content != original_test_content
        ):
            applied_files.append(test_file)
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=applied_files,
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    repairs = [
        f"{identifier}->{expression}"
        for identifier, expression in sorted(repaired_components.items())
    ]
    repairs.extend(removed_test_refs)
    print(
        "Applied tax status component repair to "
        f"{relative_output}: {', '.join(repairs)}"
    )
    print(f"manifest={manifest_path}")


def cmd_repair_missing_source_proofs(args):
    """Apply signed deterministic repairs for missing source proof atoms."""
    repo_path = Path(args.repo).resolve()
    rules_file = Path(args.file)
    if not rules_file.is_absolute():
        rules_file = repo_path / rules_file
    rules_file = rules_file.resolve()
    if not rules_file.exists():
        print(f"RuleSpec file not found: {rules_file}")
        sys.exit(1)
    try:
        relative_output = rules_file.relative_to(repo_path)
    except ValueError:
        print(f"RuleSpec file {rules_file} is not under repo {repo_path}")
        sys.exit(1)

    original_content = rules_file.read_text()
    repaired_content, repaired_rules = _repair_missing_source_proof_atoms(
        original_content
    )
    if repaired_content == original_content:
        print("No missing source proof repairs found.")
        return

    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()
    axiom_rules_path = getattr(
        args, "axiom_rules_path", None
    ) or _resolve_runtime_axiom_rules_checkout(repo_path)

    rules_file.write_text(repaired_content)
    validation = ValidatorPipeline(
        policy_repo_path=repo_path,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
        require_policy_proofs=True,
    ).validate(rules_file, skip_reviewers=True)
    if not validation.all_passed:
        issues = [
            result.error for result in validation.results.values() if result.error
        ]
        if _only_pending_tax_filing_status_branch_issues(
            issues
        ) or _only_pending_nonnegative_amount_reduction_issues(issues):
            print(
                "Applied missing source proof repair with pending deterministic "
                "validation repair still required."
            )
        else:
            rules_file.write_text(original_content)
            print("Repair failed validation; restored original file.")
            for issue in issues:
                print(f"- {issue}")
            sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_root = Path(tmpdir)
        generated_output = output_root / "deterministic-repair" / relative_output
        generated_output.parent.mkdir(parents=True, exist_ok=True)
        generated_output.write_text(repaired_content)
        result = argparse.Namespace(
            output_file=str(generated_output),
            runner="deterministic-repair",
            backend="deterministic",
            model="source-proof-atom-v1",
            tool="axiom-encode repair-missing-source-proofs",
            citation=(
                f"{_repo_jurisdiction_prefix(repo_path)}:"
                f"{_relative_rulespec_import_target(relative_output)}"
            ),
            generation_prompt_sha256=None,
            trace_file=None,
            context_manifest_file=None,
        )
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=repo_path,
            relative_output=relative_output,
            applied_files=[rules_file],
            run_id="deterministic-repair",
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )

    print(
        "Applied missing source proof repair to "
        f"{relative_output}: {', '.join(repaired_rules)}"
    )
    print(f"manifest={manifest_path}")


def _repair_tax_filing_status_branches(content: str) -> tuple[str, list[str]]:
    if (
        "surviving spouse" not in content.lower()
        and "qualifying widow" not in content.lower()
        and "any other case" not in content.lower()
    ):
        return content, []

    lines = content.splitlines(keepends=True)
    repaired: list[str] = []
    repaired_rules: list[str] = []
    needs_surviving_spouse = (
        "surviving spouse" in content.lower() or "qualifying widow" in content.lower()
    )
    needs_other_case = "any other case" in content.lower()
    current_rule_name: str | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("- name: "):
            current_rule_name = stripped.removeprefix("- name: ").strip()
        if stripped == "match filing_status:":
            match_block = [line]
            index += 1
            while index < len(lines):
                block_line = lines[index]
                block_stripped = block_line.strip()
                if block_stripped and "=>" not in block_stripped:
                    break
                match_block.append(block_line)
                index += 1
            repaired_block, block_repairs = _repair_tax_filing_status_match_block(
                match_block,
                needs_surviving_spouse=needs_surviving_spouse,
                needs_other_case=needs_other_case,
            )
            repaired.extend(repaired_block)
            repaired_rules.extend(
                current_rule_name or "<unknown>" for _ in block_repairs
            )
            continue

        repaired.append(line)
        index += 1

    return "".join(repaired), repaired_rules


def _repair_tax_filing_status_match_block(
    block: list[str],
    *,
    needs_surviving_spouse: bool,
    needs_other_case: bool,
) -> tuple[list[str], list[str]]:
    arms: dict[int, tuple[str, str]] = {}
    for line in block:
        arm_match = re.match(r"(\s*)(\d+)\s*=>\s*(.+)$", line.rstrip("\n"))
        if arm_match is None:
            continue
        indent, code, expression = arm_match.groups()
        arms[int(code)] = (indent, expression)

    should_add_surviving_spouse = needs_surviving_spouse and 1 in arms and 4 not in arms
    should_add_other_case = needs_other_case and 0 in arms and 3 not in arms

    repaired: list[str] = []
    repairs: list[str] = []
    for line in block:
        repaired.append(line)
        arm_match = re.match(r"(\s*)(\d+)\s*=>\s*(.+)$", line.rstrip("\n"))
        if arm_match is None:
            continue
        indent, code, expression = arm_match.groups()
        if code == "1" and should_add_surviving_spouse:
            repaired.append(f"{indent}4 => {expression}\n")
            repairs.append("surviving_spouse")
        if code == "0" and should_add_other_case:
            repaired.append(f"{indent}3 => {expression}\n")
            repairs.append("other_case")

    return repaired, repairs


_TAX_STATUS_COMPONENT_ISSUE_RE = re.compile(
    r"Tax filing-status component is a derived legal classification, "
    r"not a local factual input: `(?P<rule>[^`]+)` references "
    r"`(?P<identifier>[^`]+)`"
)


def _repair_tax_status_component_local_inputs(
    content: str,
) -> tuple[str, dict[str, str]]:
    """Replace local filing-status component facts with enum predicates."""
    issues = find_tax_status_component_local_input_issues(content)
    if not issues:
        return content, {}

    replacements: dict[str, str] = {}
    for issue in issues:
        match = _TAX_STATUS_COMPONENT_ISSUE_RE.search(str(issue))
        if not match:
            continue
        identifier = match["identifier"].strip()
        expression = _tax_status_component_expression(identifier)
        if expression:
            replacements[identifier] = expression
    if not replacements:
        return content, {}

    repaired = content
    applied: dict[str, str] = {}
    for identifier, expression in sorted(replacements.items()):
        updated = _replace_tax_status_component_identifier(
            repaired,
            identifier=identifier,
            expression=expression,
        )
        if updated != repaired:
            repaired = updated
            applied[identifier] = expression
    return repaired, applied


def _tax_status_component_expression(identifier: str) -> str | None:
    normalized = identifier.lower()
    if (
        "unmarried" in normalized or "not_married" in normalized
    ) and "surviving_spouse" in normalized:
        return "filing_status == 0 or filing_status == 3"
    if "head_of_household" in normalized:
        return "filing_status == 3"
    if "surviving_spouse" in normalized or "qualifying_widow" in normalized:
        if (
            "joint" in normalized
            or "filing_jointly" in normalized
            or "married_filing_jointly" in normalized
        ):
            return "filing_status == 1 or filing_status == 4"
        return "filing_status == 4"
    if "married_filing_separately" in normalized or "filing_separately" in normalized:
        return "filing_status == 2"
    if (
        "married_filing_jointly" in normalized
        or "filing_jointly" in normalized
        or "joint_return" in normalized
    ):
        return "filing_status == 1"
    if "is_married" in normalized:
        return "filing_status == 1 or filing_status == 2"
    return None


def _replace_tax_status_component_identifier(
    content: str, *, identifier: str, expression: str
) -> str:
    token = re.escape(identifier)
    repaired = re.sub(rf"\bif\s+{token}\s*:", f"if {expression}:", content)
    repaired = re.sub(rf"\bnot\s+{token}\b", f"not ({expression})", repaired)
    return re.sub(rf"\b{token}\b", f"({expression})", repaired)


def _remove_tax_status_component_test_inputs(
    test_content: str | None,
    *,
    repaired_components: dict[str, str],
) -> tuple[str | None, list[str]]:
    if test_content is None or not repaired_components:
        return test_content, []
    try:
        test_payload = yaml.safe_load(test_content) or []
    except (OSError, ValueError, yaml.YAMLError):
        return test_content, []
    if not isinstance(test_payload, list):
        return test_content, []

    input_suffixes = {
        f"#input.{identifier}" for identifier in repaired_components.keys()
    }
    removed: list[str] = []
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        if not isinstance(inputs, dict):
            continue
        for key in list(inputs.keys()):
            key_string = str(key)
            if any(key_string.endswith(suffix) for suffix in input_suffixes):
                removed.append(key_string)
                del inputs[key]

    if not removed:
        return test_content, []
    return yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False), sorted(
        set(removed)
    )


def _repair_missing_source_proof_atoms(content: str) -> tuple[str, list[str]]:
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, ValueError):
        return content, []
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return content, []

    module = payload.get("module")
    source_paths = _rulespec_module_source_paths(payload)
    if not source_paths:
        return content, []

    rules = payload.get("rules")
    if not isinstance(rules, list):
        return content, []

    repairs_by_rule: dict[str, dict[str, str]] = {}
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        if not _parsed_rule_requires_generated_source_proof(rule):
            continue
        rule_name = str(rule.get("name") or f"rules[{index}]").strip()
        if not rule_name:
            continue
        repairs_by_rule[rule_name] = {
            "kind": _generated_proof_atom_kind(rule),
            "corpus_citation_path": source_paths[0],
            "span": _generated_proof_atom_span(rule),
        }

    if not repairs_by_rule:
        return content, []

    lines = content.splitlines(keepends=True)
    lines = _ensure_module_proof_validation_required(lines, module)
    repaired_lines, repaired_rules = _insert_missing_source_proof_atoms(
        lines, repairs_by_rule
    )
    return "".join(repaired_lines), repaired_rules


def _only_pending_tax_filing_status_branch_issues(issues: list[str]) -> bool:
    return bool(issues) and all(
        issue.startswith("Filing status branch missing surviving spouse:")
        for issue in issues
    )


def _only_pending_nonnegative_amount_reduction_issues(issues: list[str]) -> bool:
    return bool(issues) and all(
        "Nonnegative amount reduction missing floor:" in issue
        or "Nonnegative amount income base missing floor:" in issue
        or "Nonnegative taxable income missing floor:" in issue
        for issue in issues
    )


def _rulespec_module_source_paths(payload: dict[str, object]) -> list[str]:
    module = payload.get("module")
    source_verification: object = None
    if isinstance(module, dict):
        source_verification = module.get("source_verification")
    if not isinstance(source_verification, dict):
        source_verification = payload.get("source_verification")
    if not isinstance(source_verification, dict):
        return []

    paths: list[str] = []
    single = str(source_verification.get("corpus_citation_path") or "").strip()
    if single:
        paths.append(single)
    many = source_verification.get("corpus_citation_paths")
    if isinstance(many, list):
        for item in many:
            value = str(item or "").strip()
            if value:
                paths.append(value)
    return paths


def _parsed_rule_requires_generated_source_proof(rule: dict[str, object]) -> bool:
    kind = str(rule.get("kind") or "").strip()
    if kind not in {"parameter", "derived"}:
        return False
    versions = rule.get("versions")
    if not isinstance(versions, list) or not versions:
        return False
    first_version = versions[0]
    if not isinstance(first_version, dict) or "formula" not in first_version:
        return False
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict):
        return True
    proof = metadata.get("proof")
    if not isinstance(proof, dict):
        return True
    atoms = proof.get("atoms")
    return not isinstance(atoms, list) or not atoms


def _generated_proof_atom_kind(rule: dict[str, object]) -> str:
    if str(rule.get("kind") or "").strip() == "parameter":
        return "parameter"
    return "formula"


def _generated_proof_atom_span(rule: dict[str, object]) -> str:
    source = rule.get("source")
    if isinstance(source, str):
        return source.strip()
    if isinstance(source, dict):
        for key in ("span", "ref", "citation"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _ensure_module_proof_validation_required(
    lines: list[str],
    module: object,
) -> list[str]:
    if isinstance(module, dict):
        proof_validation = module.get("proof_validation")
        if (
            isinstance(proof_validation, dict)
            and proof_validation.get("required") is True
        ):
            return lines

    repaired: list[str] = []
    inserted = False
    for line in lines:
        repaired.append(line)
        if not inserted and line == "module:\n":
            repaired.append("  proof_validation:\n")
            repaired.append("    required: true\n")
            inserted = True
    return repaired


def _insert_missing_source_proof_atoms(
    lines: list[str],
    repairs_by_rule: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    rule_starts = [
        index for index, line in enumerate(lines) if re.match(r"^  - name:\s*", line)
    ]
    if not rule_starts:
        return lines, []

    output = lines[:]
    offset = 0
    repaired_rules: list[str] = []
    for rule_start, next_rule_start in zip(
        rule_starts,
        rule_starts[1:] + [len(lines)],
        strict=False,
    ):
        adjusted_start = rule_start + offset
        adjusted_end = next_rule_start + offset
        name_match = re.match(r"^  - name:\s*(.+?)\s*$", output[adjusted_start])
        if name_match is None:
            continue
        rule_name = name_match.group(1).strip()
        repair = repairs_by_rule.get(rule_name)
        if repair is None:
            continue

        block = output[adjusted_start:adjusted_end]
        insertion_index, inserted_lines = _source_proof_insertion(
            block,
            proof_kind=repair["kind"],
            corpus_citation_path=repair["corpus_citation_path"],
            span=repair["span"],
        )
        absolute_insertion_index = adjusted_start + insertion_index
        output[absolute_insertion_index:absolute_insertion_index] = inserted_lines
        offset += len(inserted_lines)
        repaired_rules.append(rule_name)

    return output, repaired_rules


def _source_proof_insertion(
    rule_block: list[str],
    *,
    proof_kind: str,
    corpus_citation_path: str,
    span: str,
) -> tuple[int, list[str]]:
    metadata_index = next(
        (
            index
            for index, line in enumerate(rule_block)
            if re.match(r"^    metadata:\s*$", line)
        ),
        None,
    )
    proof_lines = _generated_source_proof_atom_lines(
        proof_kind=proof_kind,
        corpus_citation_path=corpus_citation_path,
        span=span,
        include_metadata=metadata_index is None,
    )
    if metadata_index is not None:
        return metadata_index + 1, proof_lines

    versions_index = next(
        (
            index
            for index, line in enumerate(rule_block)
            if re.match(r"^    versions:\s*$", line)
        ),
        len(rule_block),
    )
    return versions_index, proof_lines


def _generated_source_proof_atom_lines(
    *,
    proof_kind: str,
    corpus_citation_path: str,
    span: str,
    include_metadata: bool,
) -> list[str]:
    lines: list[str] = []
    if include_metadata:
        lines.append("    metadata:\n")
    lines.extend(
        [
            "      proof:\n",
            "        atoms:\n",
            "          - path: versions[0].formula\n",
            f"            kind: {proof_kind}\n",
            "            source:\n",
            f"              corpus_citation_path: {corpus_citation_path}\n",
        ]
    )
    if span:
        lines.append(f"              span: {_quote_yaml_single_line(span)}\n")
    return lines


def _quote_yaml_single_line(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _repair_proof_import_hashes(
    content: str,
    *,
    target_base: str,
    rules_file: Path,
    repo_path: Path,
) -> tuple[str, int]:
    lines = content.splitlines(keepends=True)
    repaired_lines: list[str] = []
    pending_expected_hash: str | None = None
    repair_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("target: "):
            target = stripped.removeprefix("target: ").strip().strip("'\"")
            pending_expected_hash = _expected_proof_import_hash(
                target,
                target_base=target_base,
                rules_file=rules_file,
                repo_path=repo_path,
            )
        elif pending_expected_hash is not None and stripped.startswith("hash: "):
            expected_line = f"hash: {pending_expected_hash}"
            if stripped != expected_line:
                newline = "\n" if line.endswith("\n") else ""
                prefix = line[: len(line) - len(line.lstrip())]
                line = f"{prefix}{expected_line}{newline}"
                repair_count += 1
            pending_expected_hash = None
        repaired_lines.append(line)
    return "".join(repaired_lines), repair_count


def _expected_proof_import_hash(
    target: str,
    *,
    target_base: str,
    rules_file: Path,
    repo_path: Path,
) -> str | None:
    if target.startswith(f"{target_base}#"):
        return "sha256:local"

    normalized = target.strip().strip("'\"")
    match = re.match(r"^(?P<prefix>[a-z][a-z0-9_-]*):(?P<path>[^#]+)", normalized)
    if match is None:
        return None
    if match.group("prefix") != _repo_jurisdiction_prefix(repo_path):
        return None

    target_path = match.group("path").strip().strip("/")
    if not target_path:
        return None
    target_relative = Path(target_path)
    if target_relative.is_absolute() or any(
        part in {"", ".", ".."} for part in target_relative.parts
    ):
        return None
    if not target_path.endswith((".yaml", ".yml")):
        target_relative = Path(f"{target_path}.yaml")

    target_file = (repo_path / target_relative).resolve()
    if not target_file.exists():
        return None
    if target_file == rules_file.resolve():
        return "sha256:local"
    return f"sha256:{hashlib.sha256(target_file.read_bytes()).hexdigest()}"


def _append_oracle_parameter_tests_if_missing(
    *,
    rules_file: Path,
    test_file: Path,
    target_base: str,
) -> list[str]:
    payload = yaml.safe_load(rules_file.read_text()) or {}
    rules = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(rules, list):
        return []

    existing_outputs = _rulespec_test_output_keys(test_file)
    registry = load_policyengine_registry()
    appended_cases: list[dict] = []
    repaired: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip() != "parameter":
            continue
        rule_name = str(rule.get("name") or "").strip()
        if not rule_name:
            continue
        legal_id = f"{target_base}#{rule_name}"
        if legal_id in existing_outputs:
            continue
        mapping = registry.mapping_for_legal_id(legal_id, country="us")
        if mapping is None or mapping.mapping_type != "parameter_value":
            continue

        index_name = _single_parameter_index_name(rule)
        if index_name is None:
            continue
        sample = _first_scalar_parameter_value(rule)
        if sample is None:
            continue
        key, value = sample
        appended_cases.append(
            {
                "name": f"oracle_parameter_{_safe_test_name(rule_name)}_{_safe_test_name(str(key))}",
                "period": {
                    "period_kind": "tax_year",
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                "input": {f"{target_base}#input.{index_name}": key},
                "output": {legal_id: value},
            }
        )
        repaired.append(rule_name)

    if not appended_cases:
        return []

    existing_content = test_file.read_text() if test_file.exists() else ""
    test_file.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(appended_cases, sort_keys=False)
    separator = "" if not existing_content or existing_content.endswith("\n") else "\n"
    test_file.write_text(f"{existing_content}{separator}{rendered}")
    return repaired


def _rulespec_test_output_keys(test_file: Path) -> set[str]:
    if not test_file.exists():
        return set()
    try:
        cases = _load_rulespec_test_cases(test_file)
    except (OSError, ValueError, yaml.YAMLError):
        return set()
    outputs: set[str] = set()
    for case in cases:
        output = case.get("output")
        if not isinstance(output, dict):
            continue
        outputs.update(str(key) for key in output)
    return outputs


def _single_parameter_index_name(rule: dict) -> str | None:
    indexed_by = rule.get("indexed_by")
    if isinstance(indexed_by, str) and indexed_by.strip():
        return indexed_by.strip()
    if (
        isinstance(indexed_by, list)
        and len(indexed_by) == 1
        and isinstance(indexed_by[0], str)
        and indexed_by[0].strip()
    ):
        return indexed_by[0].strip()
    return None


def _first_scalar_parameter_value(rule: dict) -> tuple[object, object] | None:
    versions = rule.get("versions")
    if not isinstance(versions, list):
        return None
    for version in versions:
        if not isinstance(version, dict):
            continue
        values = version.get("values")
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                return key, value
    return None


def _rulespec_companion_test_failures(
    test_file: Path,
    *,
    root: Path,
    axiom_rules_path: Path,
) -> list[dict[str, str | None]]:
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=axiom_rules_path,
        enable_oracles=False,
    )
    binary = pipeline._axiom_rules_binary()
    rulespec_env = pipeline._rulespec_compile_env()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _execute_rulespec_test_file(
            test_file,
            binary=binary,
            axiom_rules_path=Path(axiom_rules_path),
            env=rulespec_env,
            tmp_path=Path(tmpdir),
            compiled_cache={},
        )
    return list(result["failures"])


def _safe_test_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return safe or "value"


def _only_pending_zero_branch_coverage_issues(issues: list[str]) -> bool:
    return bool(issues) and all(
        "Zero branch test coverage missing: " in issue for issue in issues
    )


def _only_pending_exception_test_coverage_issues(issues: list[str]) -> bool:
    return bool(issues) and all(
        "Exception test coverage missing: " in issue for issue in issues
    )


def _append_exception_positive_companion_tests_if_missing(
    *,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
    issues: list[str],
) -> list[str]:
    """Clone generated negative exception tests into paired positive companions."""
    if not issues or not test_file.exists():
        return []
    repair_specs = _exception_test_repair_specs_from_issues(issues)
    if not repair_specs:
        return []

    try:
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(test_payload, list):
        return []

    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    repaired: list[str] = []
    existing_case_names = {
        str(test_case.get("name") or "").strip()
        for test_case in test_payload
        if isinstance(test_case, dict)
    }
    for rule_name, exception_input in repair_specs:
        case_name = (
            f"auto_positive_{_safe_test_name(rule_name)}_"
            f"{_safe_test_name(exception_input)}"
        )
        if case_name in existing_case_names:
            continue
        if _has_exception_positive_companion_case(
            test_payload,
            rule_name=rule_name,
            exception_input=exception_input,
        ):
            continue
        companion = _build_exception_positive_companion_case(
            test_payload,
            target_base=target_base,
            case_name=case_name,
            rule_name=rule_name,
            exception_input=exception_input,
        )
        if companion is None:
            continue
        test_payload.append(companion)
        existing_case_names.add(case_name)
        repaired.append(case_name)

    if not repaired:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return repaired


def _exception_test_repair_specs_from_issues(
    issues: list[str],
) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        match = re.search(
            r"Exception test coverage missing:\s*`(?P<rule>[^`]+)`\s+"
            r"negates\s+`(?P<input>[^`]+)`",
            str(issue),
        )
        if not match:
            continue
        spec = (match["rule"].strip(), match["input"].strip())
        if not all(spec) or spec in seen:
            continue
        seen.add(spec)
        specs.append(spec)
    return specs


def _build_exception_positive_companion_case(
    test_payload: list[object],
    *,
    target_base: str,
    case_name: str,
    rule_name: str,
    exception_input: str,
) -> dict[str, object] | None:
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        outputs = test_case.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if not _case_sets_exception_input(
            inputs, exception_input=exception_input, value=True
        ):
            continue
        if not _case_asserts_rule_value(
            outputs, rule_name=rule_name, normalized_value="not_holds"
        ):
            continue
        companion = copy.deepcopy(test_case)
        companion["name"] = case_name
        companion_inputs = companion.get("input")
        companion_outputs = companion.get("output")
        if not isinstance(companion_inputs, dict) or not isinstance(
            companion_outputs, dict
        ):
            return None
        _set_exception_input_value(
            companion_inputs,
            target_base=target_base,
            exception_input=exception_input,
            value=False,
        )
        _set_rule_output_value(
            companion_outputs,
            target_base=target_base,
            rule_name=rule_name,
            value="holds",
        )
        return companion
    return None


def _has_exception_positive_companion_case(
    test_payload: list[object],
    *,
    rule_name: str,
    exception_input: str,
) -> bool:
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        outputs = test_case.get("output")
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            continue
        if not _case_sets_exception_input(
            inputs, exception_input=exception_input, value=False
        ):
            continue
        if _case_asserts_rule_value(
            outputs, rule_name=rule_name, normalized_value="holds"
        ):
            return True
    return False


def _case_sets_exception_input(
    inputs: dict[object, object],
    *,
    exception_input: str,
    value: bool,
) -> bool:
    expected_fragment = f"input.{exception_input}"
    return any(
        _rulespec_test_key_fragment(key) == expected_fragment
        and _is_boolean_test_value(raw_value, value)
        for key, raw_value in inputs.items()
    )


def _case_asserts_rule_value(
    outputs: dict[object, object],
    *,
    rule_name: str,
    normalized_value: str,
) -> bool:
    return any(
        _rulespec_test_key_fragment(key) == rule_name
        and _normalized_generated_test_scalar(raw_value) == normalized_value
        for key, raw_value in outputs.items()
    )


def _set_exception_input_value(
    inputs: dict[object, object],
    *,
    target_base: str,
    exception_input: str,
    value: bool,
) -> None:
    expected_fragment = f"input.{exception_input}"
    for key in list(inputs):
        if _rulespec_test_key_fragment(key) == expected_fragment:
            inputs[key] = value
            return
    inputs[f"{target_base}#{expected_fragment}"] = value


def _set_rule_output_value(
    outputs: dict[object, object],
    *,
    target_base: str,
    rule_name: str,
    value: str,
) -> None:
    for key in list(outputs):
        if _rulespec_test_key_fragment(key) == rule_name:
            outputs[key] = value
            return
    outputs[f"{target_base}#{rule_name}"] = value


def _rulespec_test_key_fragment(key: object) -> str:
    raw = str(key)
    return raw.split("#", 1)[1] if "#" in raw else raw


def _is_boolean_test_value(value: object, expected: bool) -> bool:
    if isinstance(value, bool):
        return value is expected
    normalized = _normalized_generated_test_scalar(value)
    return normalized == ("true" if expected else "false")


def _normalized_generated_test_scalar(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _append_generated_zero_branch_tests_if_missing(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
    issues: list[str] | None = None,
) -> list[str]:
    return _append_generic_zero_branch_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=repo_path,
        relative_output=relative_output,
        issues=issues or [],
    )


def _append_generic_zero_branch_tests_if_missing(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
    issues: list[str],
) -> list[str]:
    """Append deterministic zero-output tests named by validator issues."""
    if not issues or not test_file.exists() or not rules_file.exists():
        return []
    output_names = _zero_branch_output_names_from_issues(issues)
    if not output_names:
        return []

    try:
        rules_content = rules_file.read_text()
        rules_payload = yaml.safe_load(rules_content) or {}
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(rules_payload, dict) or not isinstance(test_payload, list):
        return []

    rules = rules_payload.get("rules")
    if not isinstance(rules, list):
        return []
    rule_names = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    factual_inputs = _local_factual_input_names_from_rules_content(rules_content)
    input_defaults = {
        f"{target_base}#input.{input_name}": _default_generated_test_input_value(
            input_name,
            rules_payload=rules_payload,
        )
        for input_name in sorted(factual_inputs)
    }

    repaired: list[str] = []
    existing_case_names = {
        str(test_case.get("name") or "").strip()
        for test_case in test_payload
        if isinstance(test_case, dict)
    }
    for output_name in output_names:
        if output_name not in rule_names:
            continue
        target = f"{target_base}#{output_name}"
        if _has_zero_output_test(test_payload, target):
            continue
        case_name = f"auto_zero_{_safe_test_name(output_name)}"
        if case_name in existing_case_names:
            continue
        test_payload.append(
            {
                "name": case_name,
                "period": {
                    "period_kind": "tax_year",
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                "input": dict(input_defaults),
                "output": {target: 0},
            }
        )
        existing_case_names.add(case_name)
        repaired.append(case_name)

    if not repaired:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return repaired


def _append_generated_derived_output_tests_if_missing(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
    issues: list[str],
) -> list[str]:
    """Append deterministic companion cases for local derived outputs."""
    if not issues or not test_file.exists() or not rules_file.exists():
        return []
    output_targets = _missing_derived_output_targets_from_issues(issues)
    if not output_targets:
        return []

    try:
        rules_content = rules_file.read_text()
        rules_payload = yaml.safe_load(rules_content) or {}
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(rules_payload, dict) or not isinstance(test_payload, list):
        return []

    rules = rules_payload.get("rules")
    if not isinstance(rules, list):
        return []

    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    derived_rules_by_target = {
        f"{target_base}#{rule_name}": rule
        for rule in rules
        if isinstance(rule, dict)
        and str(rule.get("kind") or "").strip().lower() == "derived"
        and (rule_name := str(rule.get("name") or "").strip())
    }
    existing_outputs = _rulespec_test_output_keys(test_file)
    factual_inputs = _local_factual_input_names_from_rules_content(rules_content)
    input_defaults = {
        f"{target_base}#input.{input_name}": _default_generated_test_input_value(
            input_name,
            rules_payload=rules_payload,
        )
        for input_name in sorted(factual_inputs)
    }

    repaired: list[str] = []
    existing_case_names = {
        str(test_case.get("name") or "").strip()
        for test_case in test_payload
        if isinstance(test_case, dict)
    }
    for target in output_targets:
        if target in existing_outputs:
            continue
        rule = derived_rules_by_target.get(target)
        if rule is None:
            continue
        output_name = target.rsplit("#", 1)[-1]
        case_name = f"auto_output_{_safe_test_name(output_name)}"
        if case_name in existing_case_names:
            continue
        test_payload.append(
            {
                "name": case_name,
                "period": {
                    "period_kind": "tax_year",
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                "input": dict(input_defaults),
                "output": {
                    target: _default_generated_test_output_value(rule),
                },
            }
        )
        existing_case_names.add(case_name)
        repaired.append(case_name)

    if not repaired:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return repaired


def _missing_derived_output_targets_from_issues(issues: list[str]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        match = re.search(
            r"Derived rule missing companion output coverage:\s*`(?P<target>[^`]+)`",
            str(issue),
        )
        if not match:
            continue
        target = match["target"].strip()
        if not target or target in seen:
            continue
        seen.add(target)
        targets.append(target)
    return targets


def _default_generated_test_output_value(rule: dict[str, object]) -> object:
    dtype = str(rule.get("dtype") or "").strip().lower()
    if dtype == "judgment":
        return "not_holds"
    if dtype in {"boolean", "bool"}:
        return False
    if dtype in {"list", "array"}:
        return []
    return 0


def _remove_generated_import_output_input_placeholders(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
    issues: list[str],
) -> list[str]:
    if not rules_file.exists() or not test_file.exists():
        return []
    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    imported_outputs = _imported_output_names(rules_file)
    if not imported_outputs:
        return []
    invalid_refs = _invalid_input_refs_from_issues(issues)
    removable_refs = {
        ref
        for ref in invalid_refs
        if ref.startswith(f"{target_base}#input.")
        and ref.split("#input.", 1)[1] in imported_outputs
    }
    if not removable_refs:
        return []

    try:
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(test_payload, list):
        return []

    changed = False
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        if _remove_mapping_keys_recursive(inputs, removable_refs):
            changed = True
    if not changed:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return sorted(removable_refs)


def _remove_invalid_test_input_refs(
    *,
    test_file: Path,
    issues: list[str],
) -> list[str]:
    if not test_file.exists():
        return []
    invalid_refs = {
        ref for ref in _invalid_input_refs_from_issues(issues) if "#input." in ref
    }
    if not invalid_refs:
        return []

    try:
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(test_payload, list):
        return []

    changed = False
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        inputs = test_case.get("input")
        if _remove_mapping_keys_recursive(inputs, invalid_refs):
            changed = True
    if not changed:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return sorted(invalid_refs)


_UNREFERENCED_PROOF_IMPORT_RE = re.compile(
    r"Proof import not referenced:\s*`(?P<rule>[^`]+)`\s+"
    r"proof imports\s+`(?P<symbol>[^`]+)`"
)


def _remove_unreferenced_proof_import_atoms(
    *,
    rules_file: Path,
    issues: list[str],
) -> list[str]:
    if not rules_file.exists():
        return []
    stale_pairs: set[tuple[str, str]] = set()
    for issue in issues:
        for match in _UNREFERENCED_PROOF_IMPORT_RE.finditer(str(issue)):
            stale_pairs.add((match["rule"].strip(), match["symbol"].strip()))
    if not stale_pairs:
        return []

    original_content = rules_file.read_text()
    try:
        payload = yaml.safe_load(original_content) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return []
    if not isinstance(payload, dict):
        return []
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []

    repaired_content, removed = _remove_unreferenced_proof_import_atom_blocks(
        original_content,
        stale_pairs=stale_pairs,
    )
    if not removed:
        return []
    repaired_content, pruned_imports = _prune_unused_imports(repaired_content)
    rules_file.write_text(repaired_content)
    return sorted([*removed, *[f"unused_import:{item}" for item in pruned_imports]])


def _remove_unreferenced_proof_import_atom_blocks(
    content: str,
    *,
    stale_pairs: set[tuple[str, str]],
) -> tuple[str, list[str]]:
    lines = content.splitlines(keepends=True)
    repaired_lines: list[str] = []
    removed: list[str] = []
    current_rule = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        rule_match = re.match(r"^\s*-\s+name:\s*(.+?)\s*$", line)
        if rule_match:
            current_rule = _strip_yaml_scalar_quotes(rule_match.group(1).strip())

        item_match = re.match(r"^(\s*)-\s+path:\s*", line)
        if current_rule and item_match:
            block_end = _yaml_list_item_block_end(
                lines,
                start=index,
                start_indent=len(item_match.group(1)),
            )
            block = "".join(lines[index:block_end])
            imported_symbol = _proof_import_atom_block_imported_symbol(block)
            if imported_symbol and (current_rule, imported_symbol) in stale_pairs:
                removed.append(f"{current_rule}:{imported_symbol}")
                index = block_end
                continue

        repaired_lines.append(line)
        index += 1

    return "".join(repaired_lines), removed


def _yaml_list_item_block_end(
    lines: list[str],
    *,
    start: int,
    start_indent: int,
) -> int:
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= start_indent:
                break
        index += 1
    return index


def _proof_import_atom_block_imported_symbol(block: str) -> str:
    if not re.search(r"(?m)^\s*kind:\s*import\s*$", block):
        return ""
    output_match = re.search(r"(?m)^\s*output:\s*(.+?)\s*$", block)
    if output_match:
        return _strip_yaml_scalar_quotes(output_match.group(1).strip())
    target_match = re.search(r"(?m)^\s*target:\s*(.+?)\s*$", block)
    if not target_match:
        return ""
    target = _strip_yaml_scalar_quotes(target_match.group(1).strip())
    if "#" not in target:
        return ""
    return target.rsplit("#", 1)[1].strip()


def _imported_output_names(rules_file: Path) -> set[str]:
    try:
        payload = yaml.safe_load(rules_file.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return set()
    return _imported_output_names_from_payload(payload)


def _imported_output_names_from_payload(payload: object) -> set[str]:
    imports = payload.get("imports") if isinstance(payload, dict) else None
    if not isinstance(imports, list):
        return set()
    names: set[str] = set()
    for raw_import in imports:
        if not isinstance(raw_import, str) or "#" not in raw_import:
            continue
        fragment = raw_import.rsplit("#", 1)[1].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fragment):
            names.add(fragment)
    return names


def _input_ref_is_import_output_placeholder(
    input_ref: str,
    *,
    repo_path: Path,
) -> bool:
    if "#input." not in input_ref:
        return False
    import_base, input_name = input_ref.split("#input.", 1)
    import_base = import_base.strip()
    input_name = input_name.strip()
    if not import_base or not input_name:
        return False
    import_file = _import_base_to_repo_file(import_base, repo_path=repo_path)
    if import_file is None or not import_file.exists():
        return False
    return input_name in _imported_output_names(import_file)


def _invalid_input_refs_from_issues(issues: list[str]) -> set[str]:
    refs: set[str] = set()
    patterns = (
        r"dataset input `(?P<input>[^`]+)` must use an absolute legal RuleSpec reference",
        r"input `(?P<input>[^`]+)` does not resolve to an input slot",
    )
    for issue in issues:
        text = str(issue)
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                refs.add(match.group("input").strip())
    return refs


def _remove_mapping_keys_recursive(value: object, keys: set[str]) -> bool:
    changed = False
    if isinstance(value, dict):
        for key in list(value.keys()):
            if str(key) in keys:
                del value[key]
                changed = True
                continue
            if _remove_mapping_keys_recursive(value[key], keys):
                changed = True
    elif isinstance(value, list):
        for item in value:
            if _remove_mapping_keys_recursive(item, keys):
                changed = True
    return changed


def _zero_branch_output_names_from_issues(issues: list[str]) -> list[str]:
    output_names: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        match = re.search(
            r"Zero branch test coverage missing:\s*`(?P<name>[^`]+)`",
            str(issue),
        )
        if not match:
            continue
        name = match["name"].strip()
        if not name or name in seen:
            continue
        seen.add(name)
        output_names.append(name)
    return output_names


def _local_factual_input_names_from_rules_content(rules_content: str) -> set[str]:
    try:
        payload = yaml.safe_load(rules_content) or {}
    except yaml.YAMLError:
        return set()
    if not isinstance(payload, dict):
        return set()
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()

    defined_symbols = {
        str(rule.get("name") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and str(rule.get("name") or "").strip()
    }
    defined_symbols.update(_imported_output_names_from_payload(payload))
    dsl_symbols = {
        "abs",
        "and",
        "ceil",
        "count_where",
        "else",
        "false",
        "floor",
        "if",
        "len",
        "match",
        "max",
        "min",
        "not",
        "or",
        "round",
        "sum",
        "sum_where",
        "true",
    }
    factual_inputs: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "derived":
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
            identifiers = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", formula))
            factual_inputs.update(identifiers - defined_symbols - dsl_symbols)
    return factual_inputs


def _has_zero_output_test(test_cases: object, target: str) -> bool:
    if not isinstance(test_cases, list):
        return False
    normalized_target = target.strip()
    for test_case in test_cases:
        if not isinstance(test_case, dict):
            continue
        outputs = test_case.get("output")
        if not isinstance(outputs, dict):
            continue
        for key, value in outputs.items():
            if str(key).strip() != normalized_target:
                continue
            if _is_generated_zero_expected_value(value):
                return True
    return False


def _is_generated_zero_expected_value(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, list | tuple):
        return any(_is_generated_zero_expected_value(item) for item in value)
    if isinstance(value, int | float):
        return float(value) == 0.0
    if isinstance(value, str):
        with contextlib.suppress(ValueError):
            return float(value.strip()) == 0.0
        return False
    if isinstance(value, dict):
        raw_value = value.get("value")
        if isinstance(raw_value, dict):
            return _is_generated_zero_expected_value(raw_value.get("value"))
        return _is_generated_zero_expected_value(raw_value)
    return False


def guard_generated_change_issues(
    repo_path: Path,
    *,
    base_ref: str | None = None,
    head_ref: str = "HEAD",
    roots: tuple[str, ...] = tuple(sorted(RULESPEC_SOURCE_ROOTS)),
    changed_files: list[str] | None = None,
    all_files: bool = False,
) -> list[str]:
    """Return issues for RuleSpec changes that do not match apply manifests."""
    repo_path = Path(repo_path)
    if all_files:
        changed = []
        protected = _all_protected_rulespec_yaml_paths(repo_path, roots=roots)
    else:
        changed = (
            changed_files
            if changed_files is not None
            else _git_changed_files(repo_path, base_ref=base_ref, head_ref=head_ref)
        )
        protected = [
            path
            for path in changed
            if _is_protected_rulespec_yaml_path(Path(path), roots=roots)
        ]
    if not protected:
        return []

    manifest_paths = (
        _all_applied_encoding_manifest_paths(repo_path)
        if all_files
        else [
            path
            for path in changed
            if Path(path).parts[:2] == APPLIED_ENCODING_MANIFEST_DIR.parts
            and path.endswith(".json")
        ]
    )
    if not manifest_paths:
        if all_files:
            return [
                f"{path} is missing a matching {APPLIED_ENCODING_MANIFEST_DIR.as_posix()} manifest"
                for path in protected
            ]
        return [
            f"{path} changed without a matching {APPLIED_ENCODING_MANIFEST_DIR.as_posix()} manifest"
            for path in protected
        ]

    manifest_entries, manifest_issues = _load_applied_encoding_manifest_entries(
        repo_path, manifest_paths
    )
    if manifest_issues:
        return manifest_issues

    issues: list[str] = []
    for path in protected:
        expected = manifest_entries.get(path)
        if expected is None:
            if all_files:
                issues.append(
                    f"{path} is missing a matching {APPLIED_ENCODING_MANIFEST_DIR.as_posix()} manifest"
                )
            else:
                issues.append(
                    f"{path} changed but is not listed in a changed encoder apply manifest"
                )
            continue
        current_path = repo_path / path
        if not current_path.exists():
            issues.append(f"{path} changed but does not exist in the working tree")
            continue
        current_hash = _sha256_file(current_path)
        if current_hash != expected:
            issues.append(
                f"{path} content does not match the encoder apply manifest sha256"
            )
    return issues


def _all_protected_rulespec_yaml_paths(
    repo_path: Path, *, roots: tuple[str, ...]
) -> list[str]:
    paths: set[str] = set()
    for root in roots:
        base = repo_path / root
        if not base.exists():
            continue
        for suffix in ("*.yaml", "*.yml"):
            for path in base.rglob(suffix):
                if not path.is_file():
                    continue
                relative = path.relative_to(repo_path)
                if _is_protected_rulespec_yaml_path(relative, roots=roots):
                    paths.add(relative.as_posix())
    return sorted(paths)


def _all_applied_encoding_manifest_paths(repo_path: Path) -> list[str]:
    manifest_root = repo_path / APPLIED_ENCODING_MANIFEST_DIR
    if not manifest_root.exists():
        return []
    return sorted(
        path.relative_to(repo_path).as_posix()
        for path in manifest_root.rglob("*.json")
        if path.is_file()
    )


def _git_changed_files(
    repo_path: Path, *, base_ref: str | None, head_ref: str
) -> list[str]:
    if base_ref:
        diff_range = f"{base_ref}...{head_ref}"
        command = ["git", "diff", "--name-only", "--diff-filter=ACDMRT", diff_range]
    else:
        command = ["git", "diff", "--name-only", "--diff-filter=ACDMRT", head_ref]
    completed = subprocess.run(
        command,
        cwd=repo_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 and base_ref:
        completed = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACDMRT", base_ref, head_ref],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git diff failed")
    changed = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if base_ref is None:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if untracked.returncode != 0:
            raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
        changed.extend(
            line.strip() for line in untracked.stdout.splitlines() if line.strip()
        )
    return sorted(set(changed))


def _is_protected_rulespec_yaml_path(path: Path, *, roots: tuple[str, ...]) -> bool:
    parts = path.parts
    if len(parts) < 2:
        return False
    if parts[0] not in roots:
        return False
    return path.suffix in {".yaml", ".yml"}


def _load_applied_encoding_manifest_entries(
    repo_path: Path, manifest_paths: list[str]
) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    issues: list[str] = []
    signing_key = _applied_encoding_manifest_signing_key()
    if not signing_key:
        return (
            entries,
            [
                f"{APPLIED_ENCODING_SIGNING_KEY_ENV} is required to verify encoder apply manifests"
            ],
        )

    for manifest_path in manifest_paths:
        manifest_label = Path(manifest_path).as_posix()
        path = repo_path / manifest_path
        if not path.exists():
            issues.append(f"{manifest_label} does not exist in the working tree")
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            issues.append(f"{manifest_label} is not valid JSON")
            continue
        if payload.get("schema_version") != APPLIED_ENCODING_MANIFEST_SCHEMA:
            issues.append(f"{manifest_label} is not an encoder apply manifest")
            continue
        signature_issue = _applied_encoding_manifest_signature_issue(
            payload, signing_key
        )
        if signature_issue:
            issues.append(f"{manifest_label} {signature_issue}")
            continue
        applied_files = payload.get("applied_files")
        if not isinstance(applied_files, list):
            issues.append(f"{manifest_label} does not list applied files")
            continue
        for item in applied_files:
            if not isinstance(item, dict):
                continue
            file_path = item.get("path")
            file_hash = item.get("sha256")
            if isinstance(file_path, str) and isinstance(file_hash, str):
                entries[file_path] = file_hash
    return entries, issues


def _applied_encoding_manifest_signing_key() -> str | None:
    key = os.getenv(APPLIED_ENCODING_SIGNING_KEY_ENV, "")
    return key if key else None


def _require_applied_encoding_manifest_signing_key() -> str:
    key = _applied_encoding_manifest_signing_key()
    if not key:
        raise RuntimeError(
            f"{APPLIED_ENCODING_SIGNING_KEY_ENV} is required to apply generated RuleSpec changes"
        )
    return key


def _unsigned_applied_encoding_manifest_bytes(payload: dict) -> bytes:
    unsigned_payload = dict(payload)
    unsigned_payload.pop("signature", None)
    return json.dumps(
        unsigned_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _applied_encoding_manifest_signature(payload: dict, signing_key: str) -> str:
    return hmac.new(
        signing_key.encode(),
        _unsigned_applied_encoding_manifest_bytes(payload),
        hashlib.sha256,
    ).hexdigest()


def _sign_applied_encoding_manifest(payload: dict, signing_key: str) -> None:
    payload["signature"] = {
        "algorithm": APPLIED_ENCODING_SIGNATURE_ALGORITHM,
        "key_id": APPLIED_ENCODING_SIGNATURE_KEY_ID,
        "value": _applied_encoding_manifest_signature(payload, signing_key),
    }


def _applied_encoding_manifest_signature_issue(
    payload: dict, signing_key: str
) -> str | None:
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        return "is missing an encoder apply manifest signature"
    if signature.get("algorithm") != APPLIED_ENCODING_SIGNATURE_ALGORITHM:
        return "uses an unsupported encoder apply manifest signature algorithm"
    if signature.get("key_id") != APPLIED_ENCODING_SIGNATURE_KEY_ID:
        return "uses an unknown encoder apply manifest signing key"
    expected = _applied_encoding_manifest_signature(payload, signing_key)
    actual = signature.get("value")
    if not isinstance(actual, str) or not hmac.compare_digest(actual, expected):
        return "has an invalid encoder apply manifest signature"
    return None


def _axiom_encode_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_repo_provenance(path: Path) -> dict[str, object] | None:
    """Return exact git provenance for manifest auditing."""
    repo = Path(path)
    try:
        root = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "root": root,
        "commit": commit,
        "dirty_tracked": bool(status.strip()),
    }


_AXIOM_ENCODE_VERSION_FILES = (
    "pyproject.toml",
    "src/axiom_encode/__init__.py",
)
_AXIOM_ENCODE_VERSIONED_PREFIXES = ("src/axiom_encode/",)
_AXIOM_ENCODE_VERSIONED_FILES = {
    "pyproject.toml",
    "uv.lock",
}


def _require_axiom_encode_version_provenance(
    repo_root: Path,
) -> dict[str, str]:
    """Require encoder-affecting code changes to be behind a version bump."""
    repo = Path(repo_root)
    current_versions = _axiom_encode_versions_at_ref(repo, "HEAD")
    pyproject_version = current_versions.get("pyproject")
    package_version = current_versions.get("package")
    if not pyproject_version or not package_version:
        raise RuntimeError(
            "Cannot apply generated RuleSpec: axiom-encode version metadata is "
            "incomplete; pyproject.toml and src/axiom_encode/__init__.py must "
            "both declare the encoder version."
        )
    if pyproject_version != package_version or pyproject_version != __version__:
        raise RuntimeError(
            "Cannot apply generated RuleSpec: axiom-encode version metadata is "
            f"inconsistent (pyproject={pyproject_version}, "
            f"package={package_version}, runtime={__version__})."
        )

    version_commit = _latest_axiom_encode_version_commit(repo)
    if not version_commit:
        raise RuntimeError(
            "Cannot apply generated RuleSpec: no committed axiom-encode version "
            "bump was found; bump the encoder version before using --apply."
        )

    changed_since_version = _git_name_only(
        repo,
        "diff",
        "--name-only",
        "--diff-filter=ACMRT",
        f"{version_commit}..HEAD",
    )
    unversioned_encoder_changes = [
        path for path in changed_since_version if _is_axiom_encode_versioned_path(path)
    ]
    if unversioned_encoder_changes:
        display = ", ".join(f"`{path}`" for path in unversioned_encoder_changes[:8])
        if len(unversioned_encoder_changes) > 8:
            display += f", and {len(unversioned_encoder_changes) - 8} more"
        raise RuntimeError(
            "Cannot apply generated RuleSpec: encoder-affecting files changed "
            "after the latest axiom-encode version bump "
            f"({version_commit[:12]}): {display}. Bump the encoder version "
            "and commit that bump before using --apply."
        )

    return {
        "version": pyproject_version,
        "version_commit": version_commit,
    }


def _latest_axiom_encode_version_commit(repo: Path) -> str | None:
    commits = _git_name_only(
        repo,
        "log",
        "--format=%H",
        "--",
        *_AXIOM_ENCODE_VERSION_FILES,
    )
    for commit in commits:
        current = _axiom_encode_versions_at_ref(repo, commit)
        if not current.get("pyproject") or not current.get("package"):
            continue
        parent = _git_first_parent(repo, commit)
        if parent is None:
            return commit
        previous = _axiom_encode_versions_at_ref(repo, parent)
        if current != previous:
            return commit
    return None


def _git_first_parent(repo: Path, commit: str) -> str | None:
    try:
        line = subprocess.run(
            ["git", "-C", str(repo), "rev-list", "--parents", "-n", "1", commit],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    parts = line.split()
    return parts[1] if len(parts) > 1 else None


def _axiom_encode_versions_at_ref(repo: Path, ref: str) -> dict[str, str | None]:
    return {
        "pyproject": _extract_pyproject_version(
            _git_show_text(repo, ref, "pyproject.toml") or ""
        ),
        "package": _extract_package_init_version(
            _git_show_text(repo, ref, "src/axiom_encode/__init__.py") or ""
        ),
    }


def _git_show_text(repo: Path, ref: str, path: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{path}"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_name_only(repo: Path, *args: str) -> list[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(stderr.strip() or "git provenance command failed") from exc
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _extract_pyproject_version(content: str) -> str | None:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', content)
    return match.group(1) if match else None


def _extract_package_init_version(content: str) -> str | None:
    match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$', content)
    return match.group(1) if match else None


def _is_axiom_encode_versioned_path(path: str) -> bool:
    return path in _AXIOM_ENCODE_VERSIONED_FILES or path.startswith(
        _AXIOM_ENCODE_VERSIONED_PREFIXES
    )


def _require_clean_axiom_encode_git_provenance() -> dict[str, object]:
    provenance = _git_repo_provenance(_axiom_encode_repo_root())
    if provenance is None or not provenance.get("commit"):
        raise RuntimeError(
            "Cannot apply generated RuleSpec: axiom-encode git provenance is "
            "unavailable; commit the encoder build before using --apply."
        )
    if provenance.get("dirty_tracked"):
        raise RuntimeError(
            "Cannot apply generated RuleSpec from a dirty axiom-encode checkout; "
            "commit or discard encoder changes before using --apply."
        )
    version_provenance = _require_axiom_encode_version_provenance(
        Path(str(provenance["root"]))
    )
    provenance.update(version_provenance)
    return provenance


# =========================================================================
# Encode Command
# =========================================================================


def cmd_encode(args):
    """Encode a corpus-backed source unit through the RuleSpec eval pipeline."""
    model = args.model or DEFAULT_OPENAI_MODEL
    runner = f"{args.backend}:{model}"
    corpus_path = args.corpus_path or _resolve_repo_checkout("axiom-corpus")
    axiom_rules_path = args.axiom_rules_path or _resolve_repo_checkout(
        "axiom-rules-engine"
    )
    policy_repo_path = args.policy_repo_path or _resolve_repo_checkout("rulespec-us")

    if not corpus_path.exists():
        print(f"Axiom Corpus repo not found: {corpus_path}")
        sys.exit(1)
    if not axiom_rules_path.exists():
        print(f"axiom-rules-engine repo not found: {axiom_rules_path}")
        sys.exit(1)
    if not policy_repo_path.exists():
        print(f"Policy repo not found: {policy_repo_path}")
        sys.exit(1)

    source_id = getattr(args, "source_id", None)
    source_unit = None
    if source_id:
        source_unit = resolve_corpus_source_unit(args.citation, corpus_path)
        results = run_source_eval(
            source_id=source_id,
            source_text=source_unit.body,
            runner_specs=[runner],
            output_root=args.output,
            policy_path=policy_repo_path,
            source_metadata_payload={
                "corpus_citation_path": source_unit.citation_path,
                "corpus_source": source_unit.source,
                "requested_source": source_unit.requested,
            },
            runtime_axiom_rules_path=axiom_rules_path,
            mode=args.mode,
            extra_context_paths=[Path(path) for path in args.allow_context],
        )
    else:
        results = run_model_eval(
            citations=[args.citation],
            runner_specs=[runner],
            output_root=args.output,
            policy_path=policy_repo_path,
            runtime_axiom_rules_path=axiom_rules_path,
            corpus_path=corpus_path,
            mode=args.mode,
            extra_context_paths=[Path(path) for path in args.allow_context],
            include_tests=True,
        )

    result = results[0]
    print(f"Output root: {args.output}")
    print(f"Axiom Corpus: {corpus_path}")
    print(f"Axiom rules engine: {axiom_rules_path}")
    print(f"Policy repo: {policy_repo_path}")
    print(f"Runner: {runner}")
    print(f"Mode: {args.mode}")
    if source_unit is not None:
        print(f"Corpus source: {source_unit.citation_path} ({source_unit.source})")
        print(f"RuleSpec source id: {source_id}")
    print()
    print(f"{result.citation} [{result.runner}]")
    apply_requested = getattr(args, "apply", False) is True
    success_label = "standalone_success" if apply_requested else "success"
    print(
        f"  {success_label}={result.success} duration_ms={result.duration_ms} cost_est=${result.estimated_cost_usd or 0:.4f}"
    )
    print(
        f"  tokens in={result.input_tokens} out={result.output_tokens} cache_read={result.cache_read_tokens} reasoning_out={result.reasoning_output_tokens}"
    )
    print(f"  retrieved_files={len(result.retrieved_files)}")
    if result.unexpected_accesses:
        print(f"  unexpected_accesses={len(result.unexpected_accesses)}")
    _print_eval_metrics(result)
    if result.error:
        print(f"  error={result.error}")
    print(f"  file={result.output_file}")
    print(f"  trace={result.trace_file}")
    print(f"  manifest={result.context_manifest_file}")
    db_path = getattr(args, "db", DEFAULT_DB)
    logged_run = _log_eval_result(
        result,
        db_path=db_path,
        end_session=False,
        log_issue=False,
    )
    print(f"  run_id={logged_run.id}")
    outcome = _initial_encode_outcome(result, apply_requested=apply_requested)
    repair_manifest = None
    apply_passed = False
    if apply_requested:
        if not _can_attempt_apply(result):
            detail = str(getattr(result, "error", None) or "generation failed")
            outcome["status"] = "apply_blocked_generation"
            outcome["apply_error"] = detail
            outcome["final_success"] = False
            print(f"  apply=blocked_generation:{detail}")
        else:
            can_apply, apply_issues, supplemental_files = (
                _validate_generated_encoding_in_policy_overlay(
                    result,
                    output_root=args.output,
                    policy_repo_path=policy_repo_path,
                    axiom_rules_path=axiom_rules_path,
                    validate_dependents=not bool(
                        getattr(args, "apply_target_only", False)
                    ),
                )
            )
            outcome["overlay_validation_success"] = bool(can_apply)
            if not can_apply:
                repaired_rules = _try_repair_generated_nonnegative_floors_for_apply(
                    result,
                    output_root=args.output,
                    policy_repo_path=policy_repo_path,
                    issues=apply_issues,
                )
                if repaired_rules:
                    outcome["auto_repaired_nonnegative_floors"] = repaired_rules
                    print(
                        "  apply=auto_repaired_nonnegative_floors:"
                        + ",".join(repaired_rules)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
            if not can_apply:
                repaired_test_cases = _try_repair_generated_zero_branch_tests_for_apply(
                    result,
                    output_root=args.output,
                    policy_repo_path=policy_repo_path,
                    issues=apply_issues,
                )
                if repaired_test_cases:
                    outcome["auto_repaired_zero_branch_tests"] = repaired_test_cases
                    print(
                        "  apply=auto_repaired_zero_branch_tests:"
                        + ",".join(repaired_test_cases)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
            if not can_apply:
                repaired_test_cases = _try_repair_generated_exception_tests_for_apply(
                    result,
                    output_root=args.output,
                    policy_repo_path=policy_repo_path,
                    issues=apply_issues,
                )
                if repaired_test_cases:
                    outcome["auto_repaired_exception_tests"] = repaired_test_cases
                    print(
                        "  apply=auto_repaired_exception_tests:"
                        + ",".join(repaired_test_cases)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
            if not can_apply:
                repaired_proof_imports: list[str] = []
                while not can_apply:
                    repaired_refs = (
                        _try_repair_generated_unreferenced_proof_imports_for_apply(
                            result,
                            output_root=args.output,
                            issues=apply_issues,
                        )
                    )
                    if not repaired_refs:
                        break
                    repaired_proof_imports.extend(repaired_refs)
                    outcome["auto_repaired_unreferenced_proof_imports"] = (
                        repaired_proof_imports
                    )
                    print(
                        "  apply=auto_repaired_unreferenced_proof_imports:"
                        + ",".join(repaired_refs)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
                if repaired_proof_imports:
                    outcome["auto_repaired_unreferenced_proof_imports"] = (
                        repaired_proof_imports
                    )
            if not can_apply:
                repaired_derived_cases: list[str] = []
                while not can_apply:
                    repaired_test_cases = (
                        _try_repair_generated_derived_output_tests_for_apply(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            issues=apply_issues,
                        )
                    )
                    if not repaired_test_cases:
                        break
                    repaired_derived_cases.extend(repaired_test_cases)
                    outcome["auto_repaired_derived_output_tests"] = (
                        repaired_derived_cases
                    )
                    print(
                        "  apply=auto_repaired_derived_output_tests:"
                        + ",".join(repaired_test_cases)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
                if repaired_derived_cases:
                    outcome["auto_repaired_derived_output_tests"] = (
                        repaired_derived_cases
                    )
            if not can_apply:
                repaired_input_refs: list[str] = []
                while not can_apply:
                    repaired_refs = (
                        _try_repair_generated_import_output_inputs_for_apply(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            issues=apply_issues,
                        )
                    )
                    if not repaired_refs:
                        break
                    repaired_input_refs.extend(repaired_refs)
                    outcome["auto_repaired_import_output_inputs"] = repaired_input_refs
                    print(
                        "  apply=auto_repaired_import_output_inputs:"
                        + ",".join(repaired_refs)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
                if repaired_input_refs:
                    outcome["auto_repaired_import_output_inputs"] = repaired_input_refs
            if not can_apply:
                repaired_invalid_input_refs: list[str] = []
                while not can_apply:
                    repaired_refs = _try_repair_generated_invalid_test_inputs_for_apply(
                        result,
                        output_root=args.output,
                        issues=apply_issues,
                    )
                    if not repaired_refs:
                        break
                    repaired_invalid_input_refs.extend(repaired_refs)
                    outcome["auto_repaired_invalid_test_inputs"] = (
                        repaired_invalid_input_refs
                    )
                    print(
                        "  apply=auto_repaired_invalid_test_inputs:"
                        + ",".join(repaired_refs)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
                if repaired_invalid_input_refs:
                    outcome["auto_repaired_invalid_test_inputs"] = (
                        repaired_invalid_input_refs
                    )
            if not can_apply:
                repaired_input_cases: list[str] = []
                while not can_apply:
                    repaired_test_cases = (
                        _try_repair_generated_test_input_assignments_for_apply(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            issues=apply_issues,
                        )
                    )
                    if not repaired_test_cases:
                        break
                    repaired_input_cases.extend(repaired_test_cases)
                    outcome["auto_repaired_test_input_assignments"] = (
                        repaired_input_cases
                    )
                    print(
                        "  apply=auto_repaired_test_input_assignments:"
                        + ",".join(repaired_test_cases)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
                if repaired_input_cases:
                    outcome["auto_repaired_test_input_assignments"] = (
                        repaired_input_cases
                    )
            if not can_apply:
                repaired_test_cases = _try_repair_generated_zero_branch_tests_for_apply(
                    result,
                    output_root=args.output,
                    policy_repo_path=policy_repo_path,
                    issues=apply_issues,
                )
                if repaired_test_cases:
                    prior_repairs = outcome.get("auto_repaired_zero_branch_tests")
                    if not isinstance(prior_repairs, list):
                        prior_repairs = []
                    outcome["auto_repaired_zero_branch_tests"] = [
                        *prior_repairs,
                        *repaired_test_cases,
                    ]
                    print(
                        "  apply=auto_repaired_zero_branch_tests:"
                        + ",".join(repaired_test_cases)
                    )
                    can_apply, apply_issues, supplemental_files = (
                        _validate_generated_encoding_in_policy_overlay(
                            result,
                            output_root=args.output,
                            policy_repo_path=policy_repo_path,
                            axiom_rules_path=axiom_rules_path,
                            validate_dependents=not bool(
                                getattr(args, "apply_target_only", False)
                            ),
                        )
                    )
                    outcome["overlay_validation_success"] = bool(can_apply)
            if not can_apply:
                detail = (
                    apply_issues[0]
                    if apply_issues
                    else f"standalone_failed: {result.error or 'validation failed'}"
                )
                outcome["status"] = "apply_blocked_validation"
                outcome["apply_error"] = detail
                outcome["final_success"] = False
                print(f"  apply=blocked_validation:{detail}")
            if can_apply:
                try:
                    applied = _apply_generated_encoding_result(
                        result,
                        output_root=args.output,
                        policy_repo_path=policy_repo_path,
                        run_id=logged_run.id,
                        supplemental_files=supplemental_files,
                    )
                except RuntimeError as exc:
                    outcome["status"] = "apply_blocked_manifest"
                    outcome["apply_error"] = str(exc)
                    outcome["final_success"] = False
                    print(f"  apply=blocked_manifest:{exc}")
                else:
                    print("  apply=" + ",".join(str(path) for path in applied))
                    apply_passed = True
                    outcome["status"] = "apply_applied"
                    outcome["apply_success"] = True
                    outcome["final_success"] = True
                    outcome["applied_files"] = [str(path) for path in applied]
    repair_manifest = _record_encode_outcome(
        db_path=db_path,
        result=result,
        run=logged_run,
        outcome=outcome,
    )
    if apply_requested:
        print(f"  outcome={outcome['status']} final_success={outcome['final_success']}")
    if repair_manifest and repair_manifest.exists():
        print(f"  repair_manifest={repair_manifest}")
    if getattr(args, "sync", True) is True:
        sync_result = _sync_run_to_supabase_if_configured(logged_run, db_path=db_path)
        if not sync_result["configured"]:
            print("  supabase_sync=skipped")
        elif sync_result["run"] and sync_result["session"]:
            print("  supabase_sync=run+session")
        elif sync_result["run"]:
            print("  supabase_sync=run")
        else:
            print("  supabase_sync=failed")

    if apply_requested:
        sys.exit(0 if apply_passed else 1)
    sys.exit(0 if result.success else 1)


def _can_attempt_apply(result) -> bool:
    """Allow apply for successful runs or standalone artifact-validation failures."""
    if bool(getattr(result, "success", False)):
        return True
    error = str(getattr(result, "error", "") or "")
    return error in {
        "Generated RuleSpec failed compile validation",
        "Generated RuleSpec failed CI validation",
    }


def _try_repair_generated_nonnegative_floors_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Apply deterministic nonnegative-floor repairs to the generated candidate."""
    if not _only_pending_nonnegative_amount_reduction_issues(issues):
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    if not rules_file.exists():
        return []

    try:
        original_content = rules_file.read_text()
    except OSError:
        return []

    repaired_content, repaired_rules = repair_nonnegative_amount_reductions(
        original_content
    )
    if repaired_content == original_content:
        return []

    rules_file.write_text(repaired_content)
    test_file = _rulespec_test_path(rules_file)
    _append_generated_zero_branch_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=policy_repo_path,
        relative_output=relative_output,
    )
    return repaired_rules


def _try_repair_generated_zero_branch_tests_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Append deterministic generated zero-branch tests before applying."""
    if not _only_pending_zero_branch_coverage_issues(issues):
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _append_generated_zero_branch_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=policy_repo_path,
        relative_output=relative_output,
        issues=issues,
    )


def _try_repair_generated_exception_tests_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Append deterministic positive companions for generated exception tests."""
    if not _only_pending_exception_test_coverage_issues(issues):
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _append_exception_positive_companion_tests_if_missing(
        test_file=test_file,
        repo_path=policy_repo_path,
        relative_output=relative_output,
        issues=issues,
    )


def _try_repair_generated_derived_output_tests_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Append deterministic coverage tests for unasserted local derived outputs."""
    if not _only_pending_missing_derived_output_coverage_issues(issues):
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _append_generated_derived_output_tests_if_missing(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=policy_repo_path,
        relative_output=relative_output,
        issues=issues,
    )


def _only_pending_missing_derived_output_coverage_issues(
    issues: list[str],
) -> bool:
    return bool(issues) and all(
        "Derived rule missing companion output coverage:" in str(issue)
        for issue in issues
    )


def _try_repair_generated_import_output_inputs_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Remove local input placeholders for imported computed outputs."""
    if not issues:
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _remove_generated_import_output_input_placeholders(
        rules_file=rules_file,
        test_file=test_file,
        repo_path=policy_repo_path,
        relative_output=relative_output,
        issues=issues,
    )


def _try_repair_generated_unreferenced_proof_imports_for_apply(
    result,
    *,
    output_root: Path,
    issues: list[str],
) -> list[str]:
    """Remove proof-import atoms that validator proved are stale."""
    if not issues:
        return []

    try:
        _relative_generated_output_path(result, output_root=output_root)
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    return _remove_unreferenced_proof_import_atoms(
        rules_file=rules_file,
        issues=issues,
    )


def _try_repair_generated_invalid_test_inputs_for_apply(
    result,
    *,
    output_root: Path,
    issues: list[str],
) -> list[str]:
    """Remove generated-test input refs that validator proved are invalid."""
    if not issues:
        return []

    try:
        _relative_generated_output_path(result, output_root=output_root)
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _remove_invalid_test_input_refs(test_file=test_file, issues=issues)


def _try_repair_generated_test_input_assignments_for_apply(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    issues: list[str],
) -> list[str]:
    """Fill deterministic explicit defaults for generated tests missing facts."""
    if not _only_pending_test_input_assignment_issues(issues):
        return []

    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError:
        return []

    rules_file = Path(str(getattr(result, "output_file", "") or ""))
    test_file = _rulespec_test_path(rules_file)
    return _fill_missing_test_input_assignments(
        rules_file=rules_file,
        test_file=test_file,
        policy_repo_path=policy_repo_path,
        relative_output=relative_output,
        issues=issues,
    )


def _only_pending_test_input_assignment_issues(issues: list[str]) -> bool:
    return bool(issues) and all(
        "Test input assignment missing:" in str(issue) for issue in issues
    )


def _fill_missing_test_input_assignments(
    *,
    rules_file: Path,
    test_file: Path,
    policy_repo_path: Path,
    relative_output: Path,
    issues: list[str],
) -> list[str]:
    if not rules_file.exists() or not test_file.exists():
        return []

    try:
        rules_payload = yaml.safe_load(rules_file.read_text()) or {}
        test_payload = yaml.safe_load(test_file.read_text()) or []
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(rules_payload, dict) or not isinstance(test_payload, list):
        return []

    imported_outputs = _imported_output_names_from_payload(rules_payload)
    repairs_by_case: dict[str, set[str]] = {}
    for issue in issues:
        parsed = _parse_test_input_assignment_issue(str(issue))
        if parsed is None:
            return []
        case_name, missing_inputs = parsed
        repairs_by_case.setdefault(case_name, set()).update(
            missing_inputs - imported_outputs
        )
    if not repairs_by_case:
        return []

    anchor = _relative_output_to_anchor(
        relative_output, policy_repo_path=policy_repo_path
    )
    repaired_cases: list[str] = []
    for test_case in test_payload:
        if not isinstance(test_case, dict):
            continue
        case_name = str(test_case.get("name") or "").strip()
        missing_inputs = repairs_by_case.get(case_name)
        if not missing_inputs:
            continue
        inputs = test_case.get("input")
        if not isinstance(inputs, dict):
            inputs = {}
            test_case["input"] = inputs
        changed = False
        for input_name in sorted(missing_inputs):
            key = f"{anchor}#input.{input_name}"
            if key in inputs:
                continue
            inputs[key] = _default_generated_test_input_value(
                input_name, rules_payload=rules_payload
            )
            changed = True
        if changed:
            repaired_cases.append(case_name)

    if not repaired_cases:
        return []
    test_file.write_text(
        yaml.safe_dump(test_payload, sort_keys=False, allow_unicode=False)
    )
    return repaired_cases


def _parse_test_input_assignment_issue(issue: str) -> tuple[str, set[str]] | None:
    match = re.search(
        r"Test input assignment missing:\s*`(?P<case>[^`]+)`\s+"
        r"does not assign\s+(?P<inputs>.+?)\.\s+Every test",
        issue,
    )
    if not match:
        return None
    missing_inputs = {
        name.strip()
        for name in re.findall(r"#input\.([A-Za-z_][A-Za-z0-9_]*)", match["inputs"])
    }
    if not missing_inputs:
        return None
    return match["case"].strip(), missing_inputs


def _default_generated_test_input_value(
    input_name: str, *, rules_payload: dict[str, object]
) -> bool | int:
    if _factual_input_appears_numeric(input_name, rules_payload=rules_payload):
        return 0
    return False


def _factual_input_appears_numeric(
    input_name: str, *, rules_payload: dict[str, object]
) -> bool:
    if input_name == "filing_status":
        return True
    if _factual_input_name_looks_boolean(input_name):
        return False

    rules = rules_payload.get("rules")
    if isinstance(rules, list):
        pattern = re.compile(rf"\b{re.escape(input_name)}\b")
        numeric_context = re.compile(
            rf"(\b{re.escape(input_name)}\b\s*(?:[+\-*/<>]=?|==|!=)"
            rf"|(?:[+\-*/]\s*)\b{re.escape(input_name)}\b)"
        )
        boolean_context = re.compile(
            rf"(\bif\s+{re.escape(input_name)}\s*:"
            rf"|\bnot\s+{re.escape(input_name)}\b"
            rf"|\b{re.escape(input_name)}\b\s+(?:and|or)\b"
            rf"|\b(?:and|or)\s+{re.escape(input_name)}\b)"
        )
        formula_hits: list[str] = []
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
                if not isinstance(formula, str) or not pattern.search(formula):
                    continue
                formula_hits.append(formula)
        if any(numeric_context.search(formula) for formula in formula_hits):
            return True
        if any(boolean_context.search(formula) for formula in formula_hits):
            return False

    input_tokens = set(input_name.split("_"))
    numeric_name_fragments = (
        "amount",
        "base",
        "income",
        "wage",
        "wages",
        "remuneration",
        "salary",
        "age",
        "threshold",
        "rate",
        "value",
        "count",
        "cost",
        "expense",
        "deduction",
        "credit",
        "tax",
        "gross",
        "net",
    )
    if input_tokens.intersection(numeric_name_fragments):
        return True
    return False


def _factual_input_name_looks_boolean(input_name: str) -> bool:
    normalized = input_name.lower()
    tokens = set(normalized.split("_"))
    boolean_prefixes = (
        "any_",
        "all_",
        "is_",
        "are_",
        "has_",
        "have_",
        "had_",
        "was_",
        "were_",
        "can_",
        "must_",
        "does_",
        "do_",
        "did_",
    )
    boolean_tokens = {
        prefix.strip("_")
        for prefix in boolean_prefixes
        if prefix not in {"any_", "all_"}
    }
    boolean_suffixes = (
        "_applies",
        "_applicable",
        "_eligible",
        "_ineligible",
        "_qualified",
        "_unqualified",
        "_allowable",
        "_disallowed",
        "_exempt",
        "_excluded",
        "_included",
        "_true",
        "_false",
    )
    return (
        normalized.startswith(boolean_prefixes)
        or normalized.endswith(boolean_suffixes)
        or bool(tokens.intersection(boolean_tokens))
    )


def _relative_generated_output_path(
    result,
    *,
    output_root: Path,
) -> Path:
    output_file = Path(str(getattr(result, "output_file", "") or ""))
    runner = str(getattr(result, "runner", "") or "")
    if not output_file.exists():
        raise RuntimeError(f"Generated output file not found: {output_file}")
    generated_root = Path(output_root) / runner
    try:
        relative_output = output_file.resolve().relative_to(generated_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Generated output {output_file} is not under {generated_root}"
        ) from exc
    return relative_output


def _relative_output_to_anchor(
    relative_output: Path, *, policy_repo_path: Path | None = None
) -> str:
    """Convert a generated file's relative path to its `us:...` anchor."""
    rel = relative_output.with_suffix("")
    jurisdiction = (
        _repo_jurisdiction_prefix(policy_repo_path) if policy_repo_path else "us"
    )
    return f"{jurisdiction}:{rel.as_posix()}"


def _enforce_canonical_concept_registry(
    *,
    candidate_files: list[Path],
    relative_output: Path,
    policy_repo_path: Path | None = None,
) -> None:
    """Refuse to apply a generated encoding that violates the concept registry.

    Raises RuntimeError listing every violation so the encoder caller can
    surface them. Run before any file copy so a bad encode never lands in
    the live rules repo.
    """
    registry = load_concept_registry()
    apply_anchor = _relative_output_to_anchor(
        relative_output,
        policy_repo_path=policy_repo_path,
    )
    files = [f for f in candidate_files if f and f.exists()]
    violations = validate_generated_against_registry(
        files, registry, apply_anchor=apply_anchor
    )
    if not violations:
        return
    lines = [str(v) for v in violations]
    raise RuntimeError(
        "Canonical concept registry violations — refusing to apply:\n  "
        + "\n  ".join(lines)
    )


def _apply_generated_encoding_result(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    run_id: str | None = None,
    supplemental_files: dict[Path, str] | None = None,
) -> list[Path]:
    """Install a successful generated encoding into the target policy repo."""
    output_file = Path(str(getattr(result, "output_file", "") or ""))
    relative_output = _relative_generated_output_path(result, output_root=output_root)
    signing_key = _require_applied_encoding_manifest_signing_key()
    axiom_encode_git = _require_clean_axiom_encode_git_provenance()

    _enforce_canonical_concept_registry(
        candidate_files=[output_file, _rulespec_test_path(output_file)],
        relative_output=relative_output,
        policy_repo_path=policy_repo_path,
    )

    applied: list[Path] = []
    for source in (output_file, _rulespec_test_path(output_file)):
        if not source.exists():
            continue
        relative_source = (
            relative_output
            if source == output_file
            else _rulespec_test_path(relative_output)
        )
        target = policy_repo_path / relative_source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        applied.append(target)
    for relative_path, content in (supplemental_files or {}).items():
        target = policy_repo_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        applied.append(target)
    if applied:
        manifest_path = _write_applied_encoding_manifest(
            result,
            output_root=Path(output_root),
            policy_repo_path=Path(policy_repo_path),
            relative_output=relative_output,
            applied_files=applied,
            run_id=run_id,
            signing_key=signing_key,
            axiom_encode_git=axiom_encode_git,
        )
        applied.append(manifest_path)
    return applied


def _write_applied_encoding_manifest(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    relative_output: Path,
    applied_files: list[Path],
    axiom_encode_git: dict[str, object],
    signing_key: str,
    run_id: str | None = None,
) -> Path:
    """Record that live RuleSpec files were installed by the encoder."""
    manifest_path = policy_repo_path / _applied_encoding_manifest_path(relative_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    context_manifest_raw = str(getattr(result, "context_manifest_file", "") or "")
    trace_file_raw = str(getattr(result, "trace_file", "") or "")
    output_file_raw = str(getattr(result, "output_file", "") or "")
    context_manifest = Path(context_manifest_raw) if context_manifest_raw else None
    trace_file = Path(trace_file_raw) if trace_file_raw else None
    output_file = Path(output_file_raw) if output_file_raw else None
    tool = getattr(result, "tool", None)
    if not isinstance(tool, str) or not tool.strip():
        tool = "axiom-encode encode --apply"
    unique_applied_files: list[Path] = []
    seen_applied_paths: set[str] = set()
    for path in applied_files:
        relative_path = path.relative_to(policy_repo_path).as_posix()
        if relative_path in seen_applied_paths:
            continue
        seen_applied_paths.add(relative_path)
        unique_applied_files.append(path)
    payload = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "axiom_encode_version": __version__,
        "axiom_encode_git": axiom_encode_git,
        "generation_prompt_sha256": getattr(result, "generation_prompt_sha256", None),
        "run_id": run_id,
        "citation": str(getattr(result, "citation", "") or ""),
        "runner": str(getattr(result, "runner", "") or ""),
        "backend": str(getattr(result, "backend", "") or ""),
        "model": str(getattr(result, "model", "") or ""),
        "generated_output_root": str(output_root),
        "generated_output_file": str(output_file) if output_file is not None else None,
        "generated_output_sha256": _sha256_file(output_file)
        if output_file is not None and output_file.is_file()
        else None,
        "trace_file": str(trace_file) if trace_file is not None else None,
        "trace_sha256": _sha256_file(trace_file)
        if trace_file is not None and trace_file.is_file()
        else None,
        "context_manifest_file": str(context_manifest)
        if context_manifest is not None
        else None,
        "context_manifest_sha256": _sha256_file(context_manifest)
        if context_manifest is not None and context_manifest.is_file()
        else None,
        "applied_files": [
            {
                "path": path.relative_to(policy_repo_path).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in unique_applied_files
        ],
    }
    _sign_applied_encoding_manifest(payload, signing_key)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _applied_encoding_manifest_path(relative_output: Path) -> Path:
    return (APPLIED_ENCODING_MANIFEST_DIR / relative_output).with_suffix(".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_relation_preservation_issues(
    existing_content: str,
    generated_content: str,
) -> list[str]:
    """Return issues for source_relation edges dropped by generated content."""
    existing_records = _source_relation_records(existing_content)
    if not existing_records:
        return []
    generated_signatures = {
        record["signature"] for record in _source_relation_records(generated_content)
    }
    issues: list[str] = []
    for record in existing_records:
        if record["signature"] in generated_signatures:
            continue
        issues.append(
            "Generated RuleSpec dropped existing source_relation "
            f"`{record['name']}` (`{record['relation_type']}` -> "
            f"`{record['target']}`). Regenerate with the source_relation "
            "preserved; do not remove provenance edges by manual edit."
        )
    return issues


def _source_relation_records(content: str) -> list[dict[str, object]]:
    try:
        document = yaml.safe_load(content) or {}
    except yaml.YAMLError:
        return []
    rules = document.get("rules") if isinstance(document, dict) else []
    if not isinstance(rules, list):
        return []
    records: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "source_relation":
            continue
        source_relation = rule.get("source_relation")
        if not isinstance(source_relation, dict):
            continue
        basis = source_relation.get("basis")
        if not isinstance(basis, dict):
            basis = {}
        relation_type = _normalize_source_relation_field(source_relation.get("type"))
        target = _normalize_source_relation_field(source_relation.get("target"))
        signature = (
            relation_type,
            target,
            _normalize_source_relation_field(source_relation.get("authority")),
            _normalize_source_relation_field(source_relation.get("value")),
            _normalize_source_relation_field(basis.get("delegation")),
        )
        records.append(
            {
                "name": _normalize_source_relation_field(rule.get("name"))
                or "<unnamed>",
                "relation_type": relation_type or "<missing type>",
                "target": target or "<missing target>",
                "signature": signature,
            }
        )
    return records


def _normalize_source_relation_field(value: object) -> str:
    return str(value or "").strip()


def _validate_generated_encoding_in_policy_overlay(
    result,
    *,
    output_root: Path,
    policy_repo_path: Path,
    axiom_rules_path: Path,
    validate_dependents: bool = True,
) -> tuple[bool, list[str], dict[Path, str]]:
    """Validate generated artifacts in a temporary policy-repo overlay."""
    output_file = Path(str(getattr(result, "output_file", "") or ""))
    if not output_file.exists():
        return False, [f"Generated output file not found: {output_file}"], {}
    try:
        relative_output = _relative_generated_output_path(
            result, output_root=output_root
        )
    except RuntimeError as exc:
        return False, [str(exc)], {}

    generated_content = output_file.read_text()
    output_test = _rulespec_test_path(output_file)
    generated_test_cases: Any | None = None
    if output_test.exists():
        try:
            loaded_tests = yaml.safe_load(output_test.read_text())
        except (yaml.YAMLError, ValueError):
            loaded_tests = None
        if isinstance(loaded_tests, dict) and isinstance(
            loaded_tests.get("cases"), list
        ):
            generated_test_cases = loaded_tests["cases"]
        elif isinstance(loaded_tests, list):
            generated_test_cases = loaded_tests
    filing_status_issues = find_tax_filing_status_local_input_issues(
        generated_content,
        generated_test_cases,
    )
    if filing_status_issues:
        return (
            False,
            [f"{relative_output}: {issue}" for issue in filing_status_issues],
            {},
        )

    existing_output = policy_repo_path / relative_output
    if existing_output.exists():
        existing_content = existing_output.read_text()
        preservation_issues = _source_relation_preservation_issues(
            existing_content,
            generated_content,
        )
        if preservation_issues:
            return (
                False,
                [f"{relative_output}: {issue}" for issue in preservation_issues],
                {},
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        overlay_parent = Path(tmpdir)
        for sibling in policy_repo_path.parent.glob("rulespec-*"):
            if sibling.resolve() == policy_repo_path.resolve() or not sibling.is_dir():
                continue
            sibling_target = overlay_parent / sibling.name
            try:
                sibling_target.symlink_to(sibling.resolve(), target_is_directory=True)
            except OSError:
                shutil.copytree(sibling, sibling_target, dirs_exist_ok=True)

        overlay_repo_name = (
            canonical_rulespec_repo_name(policy_repo_path) or policy_repo_path.name
        )
        overlay_repo = overlay_parent / overlay_repo_name
        shutil.copytree(policy_repo_path, overlay_repo)
        overlay_target = overlay_repo / relative_output
        overlay_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_file, overlay_target)
        if output_test.exists():
            shutil.copy2(output_test, _rulespec_test_path(overlay_target))
        if not validate_dependents:
            _suppress_rulespec_ancestor_targets_for_subsection_overlay(
                overlay_repo,
                relative_output,
            )

        pipeline = ValidatorPipeline(
            policy_repo_path=overlay_repo,
            axiom_rules_path=axiom_rules_path,
            enable_oracles=False,
            require_policy_proofs=True,
        )
        dependents = (
            _find_rulespec_dependents(overlay_repo, relative_output)
            if validate_dependents
            else []
        )
        dependent_pipeline = (
            ValidatorPipeline(
                policy_repo_path=overlay_repo,
                axiom_rules_path=axiom_rules_path,
                enable_oracles=False,
                enforce_repository_layout=False,
            )
            if dependents
            else pipeline
        )
        supplemental_files: dict[Path, str] = {}
        changed_proof_hash_files = _repair_dependent_proof_import_hashes(
            overlay_repo=overlay_repo,
            dependents=dependents,
        )
        for path in changed_proof_hash_files:
            supplemental_files[path.relative_to(overlay_repo)] = path.read_text()
        validations = _validate_overlay_files(
            pipeline,
            dependent_pipeline=dependent_pipeline,
            overlay_target=overlay_target,
            dependents=dependents,
        )
        for _ in range(10):
            if all(validation.all_passed for _, validation in validations):
                return True, [], supplemental_files
            mixed_scalar_repairs = _repair_mixed_scalar_output_tests(
                rules_file=overlay_target,
                test_file=_rulespec_test_path(overlay_target),
                repo_path=overlay_repo,
                relative_output=relative_output,
            )
            if mixed_scalar_repairs:
                test_path = _rulespec_test_path(overlay_target)
                supplemental_files[test_path.relative_to(overlay_repo)] = (
                    test_path.read_text()
                )
                validations = _validate_overlay_files(
                    pipeline,
                    dependent_pipeline=dependent_pipeline,
                    overlay_target=overlay_target,
                    dependents=dependents,
                )
                continue
            target_validation = next(
                (
                    validation
                    for validated_file, validation in validations
                    if validated_file == overlay_target
                ),
                None,
            )
            target_test_path = _rulespec_test_path(overlay_target)
            if target_validation is not None and _complete_missing_imported_test_inputs(
                rules_file=overlay_target,
                test_file=target_test_path,
                repo_path=overlay_repo,
                validation=target_validation,
            ):
                supplemental_files[target_test_path.relative_to(overlay_repo)] = (
                    target_test_path.read_text()
                )
                validations = _validate_overlay_files(
                    pipeline,
                    dependent_pipeline=dependent_pipeline,
                    overlay_target=overlay_target,
                    dependents=dependents,
                )
                continue
            removed_invalid_inputs = _remove_invalid_dependent_test_inputs(
                overlay_repo=overlay_repo,
                relative_output=relative_output,
                validations=validations,
            )
            if removed_invalid_inputs:
                for path in removed_invalid_inputs:
                    supplemental_files[path.relative_to(overlay_repo)] = (
                        path.read_text()
                    )
                validations = _validate_overlay_files(
                    pipeline,
                    dependent_pipeline=dependent_pipeline,
                    overlay_target=overlay_target,
                    dependents=dependents,
                )
                continue
            changed_tests = _complete_missing_dependent_test_inputs(
                overlay_repo=overlay_repo,
                relative_output=relative_output,
                validations=validations,
            )
            if not changed_tests:
                break
            for path in changed_tests:
                supplemental_files[path.relative_to(overlay_repo)] = path.read_text()
            validations = _validate_overlay_files(
                pipeline,
                dependent_pipeline=dependent_pipeline,
                overlay_target=overlay_target,
                dependents=dependents,
            )
        issues: list[str] = []
        for validated_file, validation in validations:
            for validator_result in validation.results.values():
                if validator_result.error:
                    relative_file = validated_file.relative_to(overlay_repo)
                    issues.append(
                        f"{relative_file}: {validator_result.validator_name}: {validator_result.error}"
                    )
        return False, issues, {}


def _suppress_rulespec_ancestor_targets_for_subsection_overlay(
    overlay_repo: Path,
    relative_output: Path,
) -> list[Path]:
    """Remove ancestor RuleSpec files from a temporary subsection overlay."""
    suppressed: list[Path] = []
    preserved_imports = _same_repo_imported_rulespec_paths(
        overlay_repo / relative_output,
        policy_repo_path=overlay_repo,
    )
    for relative_path in _rulespec_ancestor_target_paths(relative_output):
        if relative_path in preserved_imports:
            continue
        for candidate in (
            overlay_repo / relative_path,
            _rulespec_test_path(overlay_repo / relative_path),
        ):
            if not candidate.exists() or not candidate.is_file():
                continue
            candidate.unlink()
            suppressed.append(candidate.relative_to(overlay_repo))
    return suppressed


def _same_repo_imported_rulespec_paths(
    rules_file: Path,
    *,
    policy_repo_path: Path,
) -> set[Path]:
    """Return same-repo RuleSpec import file paths used by a generated target."""
    try:
        payload = yaml.safe_load(rules_file.read_text())
    except (OSError, yaml.YAMLError, ValueError):
        return set()
    if not isinstance(payload, dict):
        return set()
    imports = payload.get("imports")
    if not isinstance(imports, list):
        return set()

    jurisdiction = _repo_jurisdiction_prefix(policy_repo_path)
    imported_paths: set[Path] = set()
    for raw_import in imports:
        if not isinstance(raw_import, str):
            continue
        target = raw_import.strip().strip('"').strip("'").split("#", 1)[0].strip()
        target = target.strip("/")
        if not target:
            continue
        if ":" in target:
            prefix, target = target.split(":", 1)
            if prefix != jurisdiction:
                continue
            target = target.strip().strip("/")
        relative = Path(
            target if target.endswith((".yaml", ".yml")) else f"{target}.yaml"
        )
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            continue
        imported_paths.add(relative)
    return imported_paths


def _rulespec_ancestor_target_paths(relative_output: Path) -> list[Path]:
    """Return possible ancestor RuleSpec files for a generated subsection."""
    target = relative_output.with_suffix("")
    source_root = next(
        (
            Path(root)
            for root in sorted(RULESPEC_SOURCE_ROOTS)
            if target == Path(root) or Path(root) in target.parents
        ),
        None,
    )
    if source_root is None:
        return []

    ancestors: list[Path] = []
    current = target.parent
    while (
        current != Path(".")
        and current.as_posix()
        and current.as_posix() != source_root.as_posix()
    ):
        ancestors.append(Path(f"{current.as_posix()}.yaml"))
        current = current.parent
    return ancestors


def _repair_mixed_scalar_output_tests(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    relative_output: Path,
) -> list[str]:
    """Split scalar parameter outputs out of mixed entity companion tests."""
    if not test_file.exists():
        return []
    try:
        rules_document = yaml.safe_load(rules_file.read_text()) or {}
        test_cases = yaml.safe_load(test_file.read_text()) or []
    except (OSError, yaml.YAMLError, ValueError):
        return []
    if not isinstance(rules_document, dict) or not isinstance(test_cases, list):
        return []

    target_base = (
        f"{_repo_jurisdiction_prefix(repo_path)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    scalar_outputs: set[str] = set()
    for rule in rules_document.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip().lower() != "parameter":
            continue
        name = str(rule.get("name") or "").strip()
        if name:
            scalar_outputs.add(f"{target_base}#{name}")
    if not scalar_outputs:
        return []

    repaired_cases: list[object] = []
    repaired_names: list[str] = []
    existing_names = {
        str(case.get("name") or "") for case in test_cases if isinstance(case, dict)
    }
    for case in test_cases:
        if not isinstance(case, dict):
            repaired_cases.append(case)
            continue
        output = case.get("output")
        if not isinstance(output, dict):
            repaired_cases.append(case)
            continue
        scalar_items = {
            key: value for key, value in output.items() if str(key) in scalar_outputs
        }
        if not scalar_items or len(scalar_items) == len(output):
            repaired_cases.append(case)
            continue

        entity_items = {
            key: value
            for key, value in output.items()
            if str(key) not in scalar_outputs
        }
        repaired_case = dict(case)
        repaired_case["output"] = entity_items
        repaired_cases.append(repaired_case)

        case_name = str(case.get("name") or "case").strip() or "case"
        scalar_case_name = _unique_test_case_name(
            f"{case_name}_scalar_outputs",
            existing_names,
        )
        existing_names.add(scalar_case_name)
        scalar_case = {
            "name": scalar_case_name,
            "period": copy.deepcopy(case.get("period")),
            "input": copy.deepcopy(case.get("input", {})),
            "output": scalar_items,
        }
        repaired_cases.append(scalar_case)
        repaired_names.append(case_name)

    if not repaired_names:
        return []
    test_file.write_text(yaml.safe_dump(repaired_cases, sort_keys=False))
    return repaired_names


def _unique_test_case_name(base: str, existing_names: set[str]) -> str:
    if base not in existing_names:
        return base
    index = 2
    while f"{base}_{index}" in existing_names:
        index += 1
    return f"{base}_{index}"


def _validate_overlay_files(
    pipeline: ValidatorPipeline,
    *,
    dependent_pipeline: ValidatorPipeline,
    overlay_target: Path,
    dependents: list[Path],
) -> list[tuple[Path, object]]:
    validations = [
        (overlay_target, pipeline.validate(overlay_target, skip_reviewers=True))
    ]
    if validations[0][1].all_passed:
        for dependent in dependents:
            validations.append(
                (
                    dependent,
                    dependent_pipeline.validate(dependent, skip_reviewers=True),
                )
            )
    return validations


def _repair_dependent_proof_import_hashes(
    *,
    overlay_repo: Path,
    dependents: list[Path],
) -> list[Path]:
    """Refresh proof import hashes in overlay dependents after target replacement."""
    changed: list[Path] = []
    jurisdiction = _repo_jurisdiction_prefix(overlay_repo)
    for dependent in dependents:
        try:
            relative_dependent = dependent.relative_to(overlay_repo)
            content = dependent.read_text()
        except (OSError, ValueError):
            continue
        target_base = (
            f"{jurisdiction}:{_relative_rulespec_import_target(relative_dependent)}"
        )
        repaired, repair_count = _repair_proof_import_hashes(
            content,
            target_base=target_base,
            rules_file=dependent,
            repo_path=overlay_repo,
        )
        if repair_count <= 0 or repaired == content:
            continue
        dependent.write_text(repaired)
        changed.append(dependent)
    return changed


_MISSING_INPUT_RE = re.compile(
    r"Test case `(?P<case>[^`]+)` execution failed: missing input `(?P<input>[^`]+)`"
)

_INVALID_INPUT_REF_RE = re.compile(
    r"input `(?P<input>[^`]+)` does not resolve to an input slot"
)


def _remove_invalid_dependent_test_inputs(
    *,
    overlay_repo: Path,
    relative_output: Path,
    validations: list[tuple[Path, object]],
) -> list[Path]:
    """Remove obsolete generated-target input refs from dependent tests."""
    target_ref = (
        f"{_repo_jurisdiction_prefix(overlay_repo)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    changed: list[Path] = []
    for validated_file, validation in validations:
        if validated_file == overlay_repo / relative_output:
            continue
        test_path = _rulespec_test_path(validated_file)
        if not test_path.exists():
            continue
        invalid_refs: set[str] = set()
        for validator_result in validation.results.values():
            error = validator_result.error or ""
            for input_ref in _invalid_input_refs_from_issues([error]):
                if input_ref.startswith(
                    f"{target_ref}#input."
                ) or _input_ref_is_import_output_placeholder(
                    input_ref,
                    repo_path=overlay_repo,
                ):
                    invalid_refs.add(input_ref)
        if not invalid_refs:
            continue
        content = test_path.read_text()
        updated = _remove_input_refs_from_test_cases(content, invalid_refs)
        if updated == content:
            continue
        test_path.write_text(updated)
        changed.append(test_path)
    return changed


def _remove_input_refs_from_test_cases(content: str, input_refs: set[str]) -> str:
    lines = content.splitlines(keepends=True)
    blocks = _find_yaml_input_blocks(lines)
    if not blocks:
        return content

    remove_indices: set[int] = set()
    for start, end in blocks:
        for index in range(start + 1, end):
            line = lines[index]
            for input_ref in input_refs:
                if re.match(rf"^\s*{re.escape(input_ref)}\s*:", line):
                    remove_indices.add(index)
                    break
                if re.match(rf"^\s*\?\s*{re.escape(input_ref)}\s*$", line):
                    remove_indices.add(index)
                    if index + 1 < end and re.match(r"^\s*:\s*", lines[index + 1]):
                        remove_indices.add(index + 1)
                    break
    if not remove_indices:
        return content
    return "".join(
        line for index, line in enumerate(lines) if index not in remove_indices
    )


def _complete_missing_dependent_test_inputs(
    *,
    overlay_repo: Path,
    relative_output: Path,
    validations: list[tuple[Path, object]],
) -> list[Path]:
    """Fill missing generated target inputs in dependent tests."""
    target_ref = (
        f"{_repo_jurisdiction_prefix(overlay_repo)}:"
        f"{_relative_rulespec_import_target(relative_output)}"
    )
    baseline_inputs = _load_test_input_baseline(
        _rulespec_test_path(overlay_repo / relative_output)
    )
    changed: list[Path] = []
    for validated_file, validation in validations:
        if validated_file == overlay_repo / relative_output:
            continue
        test_path = _rulespec_test_path(validated_file)
        if not test_path.exists():
            continue
        missing_inputs: set[str] = set()
        for validator_result in validation.results.values():
            error = validator_result.error or ""
            for match in _MISSING_INPUT_RE.finditer(error):
                missing_inputs.add(match.group("input"))
        if not missing_inputs:
            continue
        content = test_path.read_text()
        updated = content
        for input_name in sorted(missing_inputs):
            for input_ref, value in _default_refs_for_missing_input(
                input_name,
                target_ref=target_ref,
                baseline_inputs=baseline_inputs,
            ):
                updated = _insert_input_default_in_test_cases(
                    updated,
                    input_ref,
                    value,
                )
        if updated != content:
            test_path.write_text(updated)
            changed.append(test_path)
    return changed


def _complete_missing_imported_test_inputs(
    *,
    rules_file: Path,
    test_file: Path,
    repo_path: Path,
    validation: object,
) -> bool:
    """Fill missing input slots for imported modules in this file's tests."""
    if not test_file.exists():
        return False
    missing_inputs = _missing_input_names_from_validation(validation)
    if not missing_inputs:
        return False
    imported_inputs = _imported_input_refs_by_name(rules_file, repo_path=repo_path)
    if not imported_inputs:
        return False

    content = test_file.read_text()
    updated = content
    for input_name in sorted(missing_inputs):
        for input_ref in imported_inputs.get(input_name, []):
            updated = _insert_input_default_in_test_cases(
                updated,
                input_ref,
                _infer_missing_input_default(input_name),
            )
    if updated == content:
        return False
    test_file.write_text(updated)
    return True


def _missing_input_names_from_validation(validation: object) -> set[str]:
    missing_inputs: set[str] = set()
    results = getattr(validation, "results", {})
    if not isinstance(results, dict):
        return missing_inputs
    for validator_result in results.values():
        error = getattr(validator_result, "error", "") or ""
        for match in _MISSING_INPUT_RE.finditer(str(error)):
            missing_inputs.add(match.group("input"))
    return missing_inputs


def _imported_input_refs_by_name(
    rules_file: Path,
    *,
    repo_path: Path,
) -> dict[str, list[str]]:
    try:
        payload = yaml.safe_load(rules_file.read_text()) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    imports = payload.get("imports") if isinstance(payload, dict) else None
    if not isinstance(imports, list):
        return {}

    refs_by_name: dict[str, list[str]] = {}
    jurisdiction = _repo_jurisdiction_prefix(repo_path)
    for raw_import in imports:
        if not isinstance(raw_import, str):
            continue
        import_base = raw_import.split("#", 1)[0].strip().strip("/")
        if not import_base:
            continue
        import_file = _import_base_to_repo_file(import_base, repo_path=repo_path)
        if import_file is None or not import_file.exists():
            continue
        input_names = _local_factual_input_names_from_rules_content(
            import_file.read_text()
        )
        canonical_base = (
            import_base if ":" in import_base else f"{jurisdiction}:{import_base}"
        )
        for input_name in sorted(input_names):
            refs_by_name.setdefault(input_name, []).append(
                f"{canonical_base}#input.{input_name}"
            )
    return refs_by_name


def _import_base_to_repo_file(import_base: str, *, repo_path: Path) -> Path | None:
    normalized = import_base.strip().strip('"').strip("'").strip("/")
    if not normalized:
        return None
    if ":" in normalized:
        _, normalized = normalized.split(":", 1)
    relative = Path(normalized)
    if relative.suffix not in {".yaml", ".yml"}:
        relative = relative.with_suffix(".yaml")
    return repo_path / relative


def _load_test_input_baseline(test_path: Path) -> dict[str, object]:
    """Return the first companion-test input block for generated defaults."""
    try:
        cases = yaml.safe_load(test_path.read_text()) or []
    except (OSError, yaml.YAMLError, ValueError):
        return {}
    if not isinstance(cases, list):
        return {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        inputs = case.get("input")
        if isinstance(inputs, dict):
            return {str(key): value for key, value in inputs.items()}
    return {}


def _default_refs_for_missing_input(
    input_name: str,
    *,
    target_ref: str,
    baseline_inputs: dict[str, object],
) -> list[tuple[str, object]]:
    suffix = f"#input.{input_name}"
    matches = [
        (reference, value)
        for reference, value in baseline_inputs.items()
        if reference.endswith(suffix)
    ]
    if matches:
        return matches
    return [
        (f"{target_ref}#input.{input_name}", _infer_missing_input_default(input_name))
    ]


def _infer_missing_input_default(input_name: str) -> object:
    normalized = input_name.lower()
    numeric_markers = (
        "amount",
        "count",
        "deduction",
        "income",
        "number",
        "num_",
        "wage",
        "earning",
    )
    if any(marker in normalized for marker in numeric_markers):
        return 0
    return False


def _insert_input_default_in_test_cases(
    content: str, input_ref: str, value: object
) -> str:
    """Insert an input default into every concrete test input block that needs it."""
    lines = content.splitlines(keepends=True)
    blocks = _find_yaml_input_blocks(lines)
    if not blocks:
        return content

    anchored_blocks = [
        block for block in blocks if re.search(r"\s&\S+", lines[block[0]])
    ]
    target_blocks = [anchored_blocks[0]] if anchored_blocks else list(blocks)
    if anchored_blocks:
        for block in blocks:
            if block in target_blocks:
                continue
            block_text = "".join(lines[block[0] : block[1]])
            if "<<:" in block_text:
                continue
            target_blocks.append(block)

    rendered = _format_yaml_scalar(value)
    for start, end in sorted(target_blocks, reverse=True):
        block_text = "".join(lines[start:end])
        if re.search(rf"^\s*{re.escape(input_ref)}\s*:", block_text, re.MULTILINE):
            continue
        match = re.match(r"^(?P<indent>\s*)input:\s*", lines[start])
        if not match:
            continue
        indent = match.group("indent") + "  "
        newline = "\n" if lines[start].endswith("\n") else ""
        lines.insert(start + 1, f"{indent}{input_ref}: {rendered}{newline}")
    lines = _insert_input_default_in_relation_rows(lines, input_ref, rendered)
    return "".join(lines)


def _insert_input_default_in_relation_rows(
    lines: list[str], input_ref: str, rendered_value: str
) -> list[str]:
    """Insert an input default into relation entity rows using the same module."""
    if "#input." not in input_ref:
        return lines
    input_base = input_ref.split("#input.", 1)[0]

    insertions: dict[int, list[str]] = {}
    remove_indices: set[int] = set()
    for input_start, input_end in _find_yaml_input_blocks(lines):
        index = input_start + 1
        while index < input_end:
            match = re.match(r"^(?P<indent>\s*)-\s+", lines[index])
            if not match:
                index += 1
                continue
            item_indent = len(match.group("indent"))
            item_end = index + 1
            while item_end < input_end:
                candidate = lines[item_end]
                if (
                    candidate.strip()
                    and len(candidate) - len(candidate.lstrip(" ")) <= item_indent
                ):
                    break
                item_end += 1
            item_text = "".join(lines[index:item_end])
            if input_ref not in item_text and f"{input_base}#input." in item_text:
                carried_value, obsolete_line_indices = (
                    _similar_relation_row_input_value(
                        lines=lines,
                        start=index,
                        end=item_end,
                        input_base=input_base,
                        input_ref=input_ref,
                    )
                )
                if carried_value is not None:
                    rendered = carried_value
                    remove_indices.update(obsolete_line_indices)
                else:
                    rendered = rendered_value
                indent = " " * (item_indent + 2)
                newline = "\n" if lines[index].endswith("\n") else ""
                insertions.setdefault(index + 1, []).append(
                    f"{indent}{input_ref}: {rendered}{newline}"
                )
            index = item_end

    if not insertions and not remove_indices:
        return lines

    updated: list[str] = []
    for index, line in enumerate(lines):
        updated.extend(insertions.get(index, []))
        if index not in remove_indices:
            updated.append(line)
    updated.extend(insertions.get(len(lines), []))
    return updated


def _similar_relation_row_input_value(
    *,
    lines: list[str],
    start: int,
    end: int,
    input_base: str,
    input_ref: str,
) -> tuple[str | None, set[int]]:
    missing_name = input_ref.split("#input.", 1)[1]
    best_score = 0.0
    best_value: str | None = None
    best_indices: set[int] = set()
    tied = False
    pattern = re.compile(
        rf"^\s*(?P<ref>{re.escape(input_base)}#input\.(?P<name>[A-Za-z0-9_]+))\s*:\s*(?P<value>[^#\n]+)"
    )
    for index in range(start + 1, end):
        match = pattern.match(lines[index])
        if not match:
            continue
        candidate_name = match.group("name")
        score = _input_name_similarity(missing_name, candidate_name)
        if score < 0.5:
            continue
        if score == best_score:
            tied = True
            continue
        if score > best_score:
            best_score = score
            best_value = match.group("value").strip()
            best_indices = {index}
            tied = False
    if tied or best_value is None:
        return None, set()
    return best_value, best_indices


_INPUT_NAME_STOPWORDS = {
    "a",
    "an",
    "for",
    "individual",
    "is",
    "of",
    "person",
    "section",
    "the",
    "under",
}


def _input_name_similarity(left: str, right: str) -> float:
    left_tokens = _input_name_tokens(left)
    right_tokens = _input_name_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    if len(overlap) < 2:
        return 0.0
    return len(overlap) / len(left_tokens | right_tokens)


def _input_name_tokens(name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", name.lower())
        if token and not token.isdigit() and token not in _INPUT_NAME_STOPWORDS
    }


def _find_yaml_input_blocks(lines: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = re.match(r"^(?P<indent>\s*)input:\s*(?:&\S+\s*)?(?:#.*)?$", line)
        if not match:
            continue
        block_indent = len(match.group("indent"))
        end = index + 1
        while end < len(lines):
            candidate = lines[end]
            if (
                candidate.strip()
                and len(candidate) - len(candidate.lstrip(" ")) <= block_indent
            ):
                break
            end += 1
        blocks.append((index, end))
    return blocks


def _format_yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int | float):
        return str(value)
    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
    if dumped.endswith("\n..."):
        dumped = dumped.removesuffix("\n...").strip()
    return dumped


def _insert_false_input_default(content: str, input_ref: str) -> str:
    """Insert a false input default into test input blocks."""
    return _insert_input_default_in_test_cases(content, input_ref, False)


def _rulespec_test_path(path: Path) -> Path:
    """Return the companion RuleSpec test path."""
    return path.with_name(f"{path.stem}.test.yaml")


def _find_rulespec_dependents(
    policy_repo_path: Path, relative_output: Path
) -> list[Path]:
    """Find RuleSpec files that directly import the generated output."""
    target = _relative_rulespec_import_target(relative_output)
    jurisdiction = _repo_jurisdiction_prefix(policy_repo_path)
    dependents: list[Path] = []
    for root in sorted(RULESPEC_SOURCE_ROOTS):
        root_path = policy_repo_path / root
        if not root_path.exists():
            continue
        for candidate in sorted(root_path.rglob("*.yaml")):
            if candidate.name.endswith(".test.yaml"):
                continue
            try:
                relative_candidate = candidate.relative_to(policy_repo_path)
            except ValueError:
                continue
            if relative_candidate == relative_output:
                continue
            if _rulespec_file_imports_target(
                candidate, target=target, jurisdiction=jurisdiction
            ):
                dependents.append(candidate)
    return dependents


def _relative_rulespec_import_target(relative_output: Path) -> str:
    return relative_output.with_suffix("").as_posix()


def _rulespec_file_imports_target(
    path: Path, *, target: str, jurisdiction: str
) -> bool:
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    imports = payload.get("imports")
    if not isinstance(imports, list):
        return False
    canonical = f"{jurisdiction}:{target}"
    for raw_import in imports:
        if not isinstance(raw_import, str):
            continue
        import_target = raw_import.split("#", 1)[0].strip().strip("/")
        if import_target == target or import_target == canonical:
            return True
    return False


def _log_eval_result(
    result,
    *,
    db_path: Path,
    end_session: bool = True,
    log_issue: bool = True,
) -> EncodingRun:
    """Persist an eval-backed encode run in the local run history DB."""
    rulespec_content = _read_optional_text(getattr(result, "output_file", ""))
    source_text = _read_eval_source_text(getattr(result, "context_manifest_file", ""))
    error = getattr(result, "error", None)
    iteration_errors = []
    if isinstance(error, str) and error:
        iteration_errors.append(
            IterationError(
                error_type="validation",
                message=error,
            )
        )

    run = EncodingRun(
        citation=result.citation,
        file_path=str(result.output_file),
        source_text=source_text,
        iterations=[
            Iteration(
                attempt=1,
                duration_ms=int(result.duration_ms or 0),
                errors=iteration_errors,
                success=bool(result.success),
            )
        ],
        total_duration_ms=int(result.duration_ms or 0),
        agent_type=f"{result.backend}:encoder",
        agent_model=str(result.model or ""),
        rulespec_content=rulespec_content,
        review_results=_review_results_from_eval_metrics(result.metrics),
        axiom_encode_version=__version__,
    )
    run.session_id = f"encode-{run.id}"
    db = EncodingDB(db_path)
    db.log_run(run)
    repair_manifest = _write_eval_repair_manifest(result, run) if log_issue else None
    _log_eval_session(
        db,
        result,
        run,
        repair_manifest=repair_manifest,
        log_issue=log_issue,
        end_session=end_session,
    )
    return run


def _log_eval_session(
    db: EncodingDB,
    result,
    run: EncodingRun,
    *,
    repair_manifest: Path | None = None,
    log_issue: bool = True,
    end_session: bool = True,
) -> None:
    """Persist a minimal SDK-style session for eval-backed encode runs."""
    session = db.start_session(
        model=str(getattr(result, "model", "") or ""),
        cwd=os.getcwd(),
        session_id=run.session_id,
        run_id=run.id,
        axiom_encode_version=__version__,
    )
    db.update_session_tokens(
        session.id,
        input_tokens=int(getattr(result, "input_tokens", 0) or 0),
        output_tokens=int(getattr(result, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(result, "cache_read_tokens", 0) or 0),
        cache_creation_tokens=int(getattr(result, "cache_creation_tokens", 0) or 0),
    )
    db.log_event(
        session.id,
        "encode_request",
        content=f"Encode {getattr(result, 'citation', run.citation)} with {getattr(result, 'runner', '')}",
        tool_name="axiom-encode",
        metadata={
            "run_id": run.id,
            "citation": run.citation,
            "backend": getattr(result, "backend", ""),
            "model": getattr(result, "model", ""),
            "mode": str(getattr(result, "mode", "")),
        },
    )
    result_summary = {
        "run_id": run.id,
        "citation": run.citation,
        "success": bool(getattr(result, "success", False)),
        "standalone_validation_success": bool(getattr(result, "success", False)),
        "duration_ms": int(getattr(result, "duration_ms", 0) or 0),
        "estimated_cost_usd": getattr(result, "estimated_cost_usd", None),
        "input_tokens": int(getattr(result, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(result, "output_tokens", 0) or 0),
        "cache_read_tokens": int(getattr(result, "cache_read_tokens", 0) or 0),
        "cache_creation_tokens": int(getattr(result, "cache_creation_tokens", 0) or 0),
        "reasoning_output_tokens": int(
            getattr(result, "reasoning_output_tokens", 0) or 0
        ),
        "retrieved_file_count": len(getattr(result, "retrieved_files", []) or []),
        "unexpected_access_count": len(
            getattr(result, "unexpected_accesses", []) or []
        ),
        "output_file": str(getattr(result, "output_file", "") or ""),
        "trace_file": str(getattr(result, "trace_file", "") or ""),
        "context_manifest_file": str(
            getattr(result, "context_manifest_file", "") or ""
        ),
        "repair_manifest": str(repair_manifest) if repair_manifest else None,
    }
    db.log_event(
        session.id,
        "encode_result",
        content=json.dumps(result_summary, indent=2, sort_keys=True),
        tool_name="axiom-encode",
        metadata={"run_id": run.id, "success": result_summary["success"]},
    )
    error = getattr(result, "error", None)
    if log_issue and isinstance(error, str) and error:
        db.log_event(
            session.id,
            "encode_issue",
            content=error,
            tool_name="validator",
            metadata={
                "run_id": run.id,
                "repair_manifest": str(repair_manifest) if repair_manifest else None,
            },
        )
    if end_session:
        db.end_session(session.id)


def _initial_encode_outcome(result, *, apply_requested: bool) -> dict:
    """Build the workflow-level outcome object for an eval-backed encode run."""
    standalone_success = bool(getattr(result, "success", False))
    return {
        "standalone_validation_success": standalone_success,
        "apply_requested": bool(apply_requested),
        "overlay_validation_success": None,
        "apply_success": False if apply_requested else None,
        "final_success": standalone_success if not apply_requested else False,
        "status": "standalone_validated" if standalone_success else "standalone_failed",
        "primary_error": getattr(result, "error", None),
        "apply_error": None,
        "applied_files": [],
    }


def _record_encode_outcome(
    *,
    db_path: Path,
    result,
    run: EncodingRun,
    outcome: dict,
) -> Path | None:
    """Persist final encode workflow status and close the encode session."""
    db = EncodingDB(db_path)
    run.outcome = dict(outcome)
    db.update_run_outcome(run.id, run.outcome)
    db.log_event(
        run.session_id,
        "encode_outcome",
        content=json.dumps(run.outcome, indent=2, sort_keys=True),
        tool_name="axiom-encode",
        metadata={
            "run_id": run.id,
            "status": run.outcome.get("status"),
            "final_success": run.outcome.get("final_success"),
            "standalone_validation_success": run.outcome.get(
                "standalone_validation_success"
            ),
            "apply_requested": run.outcome.get("apply_requested"),
            "overlay_validation_success": run.outcome.get("overlay_validation_success"),
            "apply_success": run.outcome.get("apply_success"),
        },
    )

    repair_manifest = None
    if run.outcome.get("final_success") is not True:
        if run.outcome.get("standalone_validation_success") is not True:
            repair_manifest = _write_eval_repair_manifest(result, run)
        issue = _encode_outcome_issue(result, run.outcome)
        if issue:
            db.log_event(
                run.session_id,
                "encode_issue",
                content=issue,
                tool_name="validator",
                metadata={
                    "run_id": run.id,
                    "status": run.outcome.get("status"),
                    "repair_manifest": str(repair_manifest)
                    if repair_manifest
                    else None,
                },
            )
    db.end_session(run.session_id)
    return repair_manifest


def _encode_outcome_issue(result, outcome: dict) -> str:
    for key in ("apply_error", "primary_error"):
        value = outcome.get(key)
        if isinstance(value, str) and value:
            return value
    error = getattr(result, "error", None)
    if isinstance(error, str) and error:
        return error
    status = outcome.get("status")
    if isinstance(status, str) and status:
        return status
    return ""


def _write_eval_repair_manifest(result, run: EncodingRun) -> Path | None:
    """Write a small action manifest for failed eval-backed encode runs."""
    if bool(getattr(result, "success", False)):
        return None

    manifest_path = _eval_repair_manifest_path(result)
    if manifest_path is None:
        return None
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "axiom-encode/repair-manifest/v1",
            "created_at": _utc_now_iso(),
            "run_id": run.id,
            "session_id": run.session_id,
            "citation": run.citation,
            "runner": getattr(result, "runner", ""),
            "backend": getattr(result, "backend", ""),
            "model": getattr(result, "model", ""),
            "mode": str(getattr(result, "mode", "")),
            "error": getattr(result, "error", None),
            "files": {
                "output": str(getattr(result, "output_file", "") or ""),
                "trace": str(getattr(result, "trace_file", "") or ""),
                "context_manifest": str(
                    getattr(result, "context_manifest_file", "") or ""
                ),
            },
            "actions": [
                {
                    "id": "inspect_trace",
                    "label": "Inspect the model trace and validation output",
                    "path": str(getattr(result, "trace_file", "") or ""),
                },
                {
                    "id": "rerun_encode",
                    "label": "Rerun axiom-encode encode for the same citation",
                    "citation": run.citation,
                    "backend": getattr(result, "backend", ""),
                    "model": getattr(result, "model", ""),
                },
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except OSError:
        return None
    return manifest_path


def _eval_repair_manifest_path(result) -> Path | None:
    output_file = getattr(result, "output_file", None)
    if not output_file:
        return None
    return Path(str(output_file)).with_suffix(".repair.json")


def _sync_run_to_supabase_if_configured(
    run: EncodingRun, *, db_path: Path = DEFAULT_DB
) -> dict[str, bool]:
    if not (
        os.environ.get("AXIOM_ENCODE_SUPABASE_URL")
        and os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY")
    ):
        return {"configured": False, "run": False, "session": False}
    from .supabase_sync import sync_agent_sessions_to_supabase, sync_run_to_supabase

    run_synced = sync_run_to_supabase(run, "reviewer_agent")
    session_synced = False
    if run_synced and run.session_id:
        session_stats = sync_agent_sessions_to_supabase(
            session_id=run.session_id,
            db_path=db_path,
        )
        session_synced = (
            session_stats.get("synced", 0) > 0 and session_stats.get("failed", 0) == 0
        )
    return {"configured": True, "run": run_synced, "session": session_synced}


def _read_optional_text(path_value) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text()


def _read_eval_source_text(context_manifest_file) -> str | None:
    if not context_manifest_file:
        return None
    manifest_path = Path(str(context_manifest_file))
    if not manifest_path.exists() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    source_text_file = manifest.get("source_text_file")
    if not isinstance(source_text_file, str) or not source_text_file:
        return None
    source_text_path = manifest_path.parent / source_text_file
    source_text = _read_optional_text(source_text_path)
    return source_text or None


def _review_results_from_eval_metrics(metrics) -> ReviewResults | None:
    if metrics is None:
        return None

    reviews = [
        ReviewResult(
            reviewer="rulespec_reviewer",
            passed=bool(metrics.compile_pass),
            items_checked=10,
            items_passed=10 if metrics.compile_pass else 0,
            critical_issues=list(metrics.compile_issues or []),
        ),
        ReviewResult(
            reviewer="formula_reviewer",
            passed=bool(metrics.ci_pass),
            items_checked=10,
            items_passed=10 if metrics.ci_pass else 0,
            critical_issues=list(metrics.ci_issues or []),
        ),
        ReviewResult(
            reviewer="parameter_reviewer",
            passed=metrics.ungrounded_numeric_count == 0,
            items_checked=10,
            items_passed=10 if metrics.ungrounded_numeric_count == 0 else 0,
            important_issues=[
                item.raw
                for item in (metrics.grounding or [])
                if not getattr(item, "grounded", False)
            ],
        ),
    ]

    if metrics.generalist_review_score is not None:
        score = max(0, min(10, int(round(metrics.generalist_review_score))))
        reviews.append(
            ReviewResult(
                reviewer="integration_reviewer",
                passed=bool(metrics.generalist_review_pass),
                items_checked=10,
                items_passed=score,
                important_issues=list(metrics.generalist_review_issues or []),
            )
        )

    return ReviewResults(
        reviews=reviews,
        policyengine_match=metrics.policyengine_score,
        taxsim_match=metrics.taxsim_score,
    )


def _print_eval_metrics(result) -> None:
    """Print human-readable eval metrics when present."""
    if not result.metrics:
        return

    print(
        f"  compile={'yes' if result.metrics.compile_pass else 'no'} ci={'yes' if result.metrics.ci_pass else 'no'}"
    )
    print(
        f"  grounded={result.metrics.grounded_numeric_count} ungrounded={result.metrics.ungrounded_numeric_count} embedded_source={'yes' if result.metrics.embedded_source_present else 'no'}"
    )
    if result.metrics.generalist_review_score is not None:
        print(
            f"  generalist_review={'yes' if result.metrics.generalist_review_pass else 'no'} score={result.metrics.generalist_review_score:.1f}/10"
        )
    if result.metrics.policyengine_score is not None:
        print(
            f"  policyengine={'yes' if result.metrics.policyengine_pass else 'no'} score={result.metrics.policyengine_score:.1%}"
        )
    if result.metrics.taxsim_score is not None:
        print(
            f"  taxsim={'yes' if result.metrics.taxsim_pass else 'no'} score={result.metrics.taxsim_score:.1%}"
        )
    if result.metrics.ungrounded_numeric_count:
        offenders = [item.raw for item in result.metrics.grounding if not item.grounded]
        print(f"  ungrounded_values={', '.join(offenders[:10])}")


def cmd_eval(args):
    """Run deterministic model comparisons on one or more citations."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", DEFAULT_GPT_RUNNER], args
    )
    corpus_path = args.corpus_path or _resolve_repo_checkout("axiom-corpus")
    axiom_rules_path = args.axiom_rules_path or _resolve_repo_checkout(
        "axiom-rules-engine"
    )
    policy_repo_path = args.policy_repo_path or _resolve_repo_checkout("rulespec-us")

    if not corpus_path.exists():
        print(f"Axiom Corpus repo not found: {corpus_path}")
        sys.exit(1)
    if not axiom_rules_path.exists():
        print(f"axiom-rules-engine repo not found: {axiom_rules_path}")
        sys.exit(1)
    if not policy_repo_path.exists():
        print(f"Policy repo not found: {policy_repo_path}")
        sys.exit(1)

    results = run_model_eval(
        citations=args.citations,
        runner_specs=runners,
        output_root=args.output,
        policy_path=policy_repo_path,
        runtime_axiom_rules_path=axiom_rules_path,
        corpus_path=corpus_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"Axiom Corpus: {corpus_path}")
    print(f"Axiom rules engine: {axiom_rules_path}")
    print(f"Policy repo: {policy_repo_path}")
    print(f"Mode: {args.mode}")
    print()

    for result in results:
        print(f"{result.citation} [{result.runner}]")
        print(
            f"  success={result.success} duration_ms={result.duration_ms} cost_est=${result.estimated_cost_usd or 0:.4f}"
        )
        print(
            f"  tokens in={result.input_tokens} out={result.output_tokens} cache_read={result.cache_read_tokens} reasoning_out={result.reasoning_output_tokens}"
        )
        print(f"  retrieved_files={len(result.retrieved_files)}")
        if result.unexpected_accesses:
            print(f"  unexpected_accesses={len(result.unexpected_accesses)}")
        _print_eval_metrics(result)
        if result.error:
            print(f"  error={result.error}")
        print(f"  file={result.output_file}")
        print(f"  trace={result.trace_file}")
        print(f"  manifest={result.context_manifest_file}")
        print()


def cmd_eval_source(args):
    """Run deterministic model comparisons on one corpus-backed source unit."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", DEFAULT_GPT_RUNNER], args
    )
    corpus_path = args.corpus_path or _resolve_repo_checkout("axiom-corpus")
    if not corpus_path.exists():
        print(f"Axiom Corpus repo not found: {corpus_path}")
        sys.exit(1)

    source_unit = resolve_corpus_source_unit(args.corpus_citation_path, corpus_path)
    source_id = args.source_id or source_unit.citation_path
    policy_repo_path = _resolve_policy_repo_for_corpus_source(
        source_unit.citation_path,
        args.policy_repo_path,
    )
    runtime_axiom_rules_path = args.axiom_rules_path
    runtime_axiom_rules_path = (
        runtime_axiom_rules_path
        or _resolve_runtime_axiom_rules_checkout(policy_repo_path)
    )

    if not policy_repo_path.exists():
        print(f"Policy repo not found: {policy_repo_path}")
        sys.exit(1)
    if not runtime_axiom_rules_path.exists():
        print(f"axiom-rules-engine repo not found: {runtime_axiom_rules_path}")
        sys.exit(1)

    results = run_source_eval(
        source_id=source_id,
        source_text=source_unit.body,
        runner_specs=runners,
        output_root=args.output,
        policy_path=policy_repo_path,
        source_metadata_payload={
            "corpus_citation_path": source_unit.citation_path,
            "corpus_source": source_unit.source,
            "requested_source": source_unit.requested,
        },
        runtime_axiom_rules_path=runtime_axiom_rules_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
        policyengine_rule_hint=args.policyengine_rule_hint,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"Axiom Corpus: {corpus_path}")
    print(f"Policy repo: {policy_repo_path}")
    print(f"Axiom rules engine: {runtime_axiom_rules_path}")
    print(f"Corpus source: {source_unit.citation_path} ({source_unit.source})")
    if args.policyengine_rule_hint:
        print(f"PolicyEngine rule hint: {args.policyengine_rule_hint}")
    print(f"Mode: {args.mode}")
    print()

    for result in results:
        print(f"{result.citation} [{result.runner}]")
        print(
            f"  success={result.success} duration_ms={result.duration_ms} cost_est=${result.estimated_cost_usd or 0:.4f}"
        )
        print(
            f"  tokens in={result.input_tokens} out={result.output_tokens} cache_read={result.cache_read_tokens} reasoning_out={result.reasoning_output_tokens}"
        )
        print(f"  retrieved_files={len(result.retrieved_files)}")
        if result.unexpected_accesses:
            print(f"  unexpected_accesses={len(result.unexpected_accesses)}")
        _print_eval_metrics(result)
        if result.error:
            print(f"  error={result.error}")
        print(f"  file={result.output_file}")
        print(f"  trace={result.trace_file}")
        print(f"  manifest={result.context_manifest_file}")
        print()


def _format_gate_result(gate) -> str:
    """Format one readiness gate for human-readable output."""
    relation = ">=" if gate.comparator == "min" else "<="
    actual = "n/a" if gate.actual is None else f"{gate.actual}"
    return (
        f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: "
        f"{actual} {relation} {gate.threshold}"
    )


def _serialize_eval_result(result) -> dict:
    """Return a JSON-serializable eval result payload."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    return {
        "citation": getattr(result, "citation", None),
        "runner": getattr(result, "runner", None),
        "backend": getattr(result, "backend", None),
        "model": getattr(result, "model", None),
        "mode": getattr(result, "mode", None),
        "output_file": getattr(result, "output_file", None),
        "trace_file": getattr(result, "trace_file", None),
        "context_manifest_file": getattr(result, "context_manifest_file", None),
        "duration_ms": getattr(result, "duration_ms", None),
        "success": getattr(result, "success", None),
        "error": getattr(result, "error", None),
        "generation_prompt_sha256": getattr(result, "generation_prompt_sha256", None),
        "metrics": getattr(result, "metrics", None),
    }


def _serialize_gate_result(gate) -> dict:
    """Return a JSON-serializable readiness gate payload."""
    if is_dataclass(gate):
        return asdict(gate)
    if isinstance(gate, dict):
        return gate
    return {
        "name": getattr(gate, "name", None),
        "comparator": getattr(gate, "comparator", None),
        "threshold": getattr(gate, "threshold", None),
        "actual": getattr(gate, "actual", None),
        "passed": getattr(gate, "passed", None),
    }


def _serialize_readiness_summary(summary) -> dict:
    """Return a JSON-serializable readiness summary payload."""
    if is_dataclass(summary):
        return asdict(summary)
    if isinstance(summary, dict):
        return summary
    return {
        "total_cases": getattr(summary, "total_cases", None),
        "success_rate": getattr(summary, "success_rate", None),
        "compile_pass_rate": getattr(summary, "compile_pass_rate", None),
        "ci_pass_rate": getattr(summary, "ci_pass_rate", None),
        "zero_ungrounded_rate": getattr(summary, "zero_ungrounded_rate", None),
        "generalist_review_pass_rate": getattr(
            summary, "generalist_review_pass_rate", None
        ),
        "mean_generalist_review_score": getattr(
            summary, "mean_generalist_review_score", None
        ),
        "policyengine_case_count": getattr(summary, "policyengine_case_count", None),
        "policyengine_pass_rate": getattr(summary, "policyengine_pass_rate", None),
        "mean_policyengine_score": getattr(summary, "mean_policyengine_score", None),
        "mean_estimated_cost_usd": getattr(summary, "mean_estimated_cost_usd", None),
        "gate_results": [
            _serialize_gate_result(gate)
            for gate in getattr(summary, "gate_results", []) or []
        ],
        "ready": getattr(summary, "ready", None),
    }


def _build_eval_suite_payload(
    manifest, effective_runners, results, readiness, all_ready
):
    """Build the persisted eval-suite payload shared by text and JSON output."""
    return {
        "manifest": {
            "name": manifest.name,
            "path": str(manifest.path),
            "runners": manifest.runners,
            "effective_runners": effective_runners,
        },
        "results": [_serialize_eval_result(result) for result in results],
        "readiness": {
            runner: _serialize_readiness_summary(summary)
            for runner, summary in readiness.items()
        },
        "all_ready": all_ready,
    }


def _load_eval_suite_run_state(output_root: Path) -> dict | None:
    """Load persisted suite lifecycle state when available."""
    state_path = output_root / "suite-run.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _suite_auto_resume_reason(state: dict | None, total_cases: int) -> str | None:
    """Return a human-readable reason to auto-resume, if the suite is incomplete."""
    if not state:
        return None
    status = str(state.get("status") or "").strip().lower()
    completed_cases = int(state.get("completed_cases", 0) or 0)
    if completed_cases >= total_cases or status == "completed":
        return None
    error = str(state.get("error") or "").strip()
    if "usage limit" in error.lower():
        return None
    if status not in {"running", "failed", "interrupted"}:
        return None
    active_case = state.get("active_case") or {}
    active_name = active_case.get("name")
    if active_name:
        return f"{status} while running case '{active_name}'"
    if error:
        return f"{status}: {error}"
    return status or "incomplete state"


def cmd_eval_suite(args):
    """Run a manifest-driven benchmark suite and evaluate readiness gates."""
    manifest = load_eval_suite_manifest(args.manifest)
    effective_runners = _effective_runner_specs(args.runner or manifest.runners, args)
    axiom_rules_path = args.axiom_rules_path or _resolve_repo_checkout(
        "axiom-rules-engine"
    )
    corpus_path = args.corpus_path or _resolve_repo_checkout("axiom-corpus")

    if not axiom_rules_path.exists():
        print(f"axiom-rules-engine repo not found: {axiom_rules_path}")
        sys.exit(1)

    has_corpus_case = any(
        case.kind in {"citation", "source"} for case in manifest.cases
    )
    if has_corpus_case and not corpus_path.exists():
        print(f"Axiom Corpus repo not found: {corpus_path}")
        sys.exit(1)

    auto_resume_attempts = max(getattr(args, "auto_resume_attempts", 0), 0)
    auto_resume_delay_seconds = max(getattr(args, "auto_resume_delay_seconds", 0), 0)
    resume_existing = getattr(args, "resume", False)
    recovery_count = 0

    while True:
        try:
            results = run_eval_suite(
                manifest=manifest,
                output_root=args.output,
                axiom_rules_path=axiom_rules_path,
                corpus_path=corpus_path if has_corpus_case else None,
                runner_specs=effective_runners,
                resume_existing=resume_existing,
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            if recovery_count >= auto_resume_attempts:
                raise
            recovery_count += 1
            detail = str(exc).strip() or exc.__class__.__name__
            print(
                "eval-suite exited unexpectedly; "
                f"auto-resuming {recovery_count}/{auto_resume_attempts} "
                f"after {auto_resume_delay_seconds}s ({detail})",
                file=sys.stderr,
            )
            if auto_resume_delay_seconds:
                time.sleep(auto_resume_delay_seconds)
            resume_existing = True
            continue

        auto_resume_reason = _suite_auto_resume_reason(
            _load_eval_suite_run_state(args.output),
            total_cases=len(manifest.cases),
        )
        if auto_resume_reason and recovery_count < auto_resume_attempts:
            recovery_count += 1
            print(
                "eval-suite stopped before completion; "
                f"auto-resuming {recovery_count}/{auto_resume_attempts} "
                f"after {auto_resume_delay_seconds}s ({auto_resume_reason})",
                file=sys.stderr,
            )
            if auto_resume_delay_seconds:
                time.sleep(auto_resume_delay_seconds)
            resume_existing = True
            continue
        break

    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.runner, []).append(result)

    readiness = {
        runner: summarize_readiness(runner_results, manifest.gates)
        for runner, runner_results in grouped.items()
    }
    all_ready = all(summary.ready for summary in readiness.values())
    payload = _build_eval_suite_payload(
        manifest=manifest,
        effective_runners=effective_runners,
        results=results,
        readiness=readiness,
        all_ready=all_ready,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "manifest": payload["manifest"],
                "readiness": payload["readiness"],
                "all_ready": all_ready,
            },
            indent=2,
        )
        + "\n"
    )

    if args.json:
        print(json.dumps(payload, indent=2))
        sys.exit(0 if all_ready else 1)

    print(f"Manifest: {manifest.path}")
    print(f"Suite: {manifest.name}")
    print(f"Output root: {args.output}")
    print(f"Runners: {', '.join(effective_runners)}")
    print(f"Axiom rules engine: {axiom_rules_path}")
    if has_corpus_case:
        print(f"Axiom Corpus: {corpus_path}")
    print()

    for runner, summary in readiness.items():
        print(f"{runner}: {'READY' if summary.ready else 'NOT READY'}")
        print(
            f"  cases={summary.total_cases} success={summary.success_rate:.1%} "
            f"compile={summary.compile_pass_rate:.1%} ci={summary.ci_pass_rate:.1%} "
            f"zero_ungrounded={summary.zero_ungrounded_rate:.1%} "
            f"generalist_review={summary.generalist_review_pass_rate:.1%}"
        )
        if summary.mean_generalist_review_score is not None:
            print(
                f"  mean_generalist_review_score={summary.mean_generalist_review_score:.2f}/10"
            )
        if summary.policyengine_case_count:
            print(
                f"  policyengine_cases={summary.policyengine_case_count} "
                f"pass_rate={(summary.policyengine_pass_rate or 0):.1%} "
                f"mean_score={(summary.mean_policyengine_score or 0):.1%}"
            )
        if summary.mean_estimated_cost_usd is not None:
            print(f"  mean_estimated_cost=${summary.mean_estimated_cost_usd:.4f}")
        for gate in summary.gate_results:
            print(_format_gate_result(gate))

        notable_failures = [
            result
            for result in grouped[runner]
            if (
                not result.success
                or result.error
                or result.metrics is None
                or not result.metrics.compile_pass
                or not result.metrics.ci_pass
                or result.metrics.ungrounded_numeric_count > 0
                or result.metrics.generalist_review_pass is False
            )
        ]
        if notable_failures:
            print("  notable_failures:")
            for result in notable_failures[:5]:
                print(
                    f"    - {result.citation}: success={result.success} "
                    f"compile={getattr(result.metrics, 'compile_pass', None)} "
                    f"ci={getattr(result.metrics, 'ci_pass', None)} "
                    f"ungrounded={getattr(result.metrics, 'ungrounded_numeric_count', None)} "
                    f"generalist={getattr(result.metrics, 'generalist_review_pass', None)}"
                )
        print()

    sys.exit(0 if all_ready else 1)


def cmd_eval_suite_revalidate(args):
    """Re-run validators for an existing eval-suite output in place."""
    source_output = Path(args.source_output)
    if not source_output.exists():
        print(f"Suite output not found: {source_output}")
        sys.exit(1)

    run_state = _load_eval_suite_run_state(source_output)
    manifest_path = args.manifest
    if manifest_path is None and run_state:
        manifest_meta = run_state.get("manifest") or {}
        path_str = str(manifest_meta.get("path") or "").strip()
        if path_str:
            manifest_path = Path(path_str)
    if manifest_path is None:
        print(
            "Could not determine manifest path for suite revalidation. "
            "Pass --manifest explicitly."
        )
        sys.exit(1)

    manifest = load_eval_suite_manifest(manifest_path)
    axiom_rules_path = args.axiom_rules_path or _resolve_repo_checkout(
        "axiom-rules-engine"
    )
    corpus_path = args.corpus_path or _resolve_repo_checkout("axiom-corpus")
    if not axiom_rules_path.exists():
        print(f"axiom-rules-engine repo not found: {axiom_rules_path}")
        sys.exit(1)
    if not corpus_path.exists():
        print(f"Axiom Corpus repo not found: {corpus_path}")
        sys.exit(1)

    ledger_path = source_output / "suite-results.jsonl"
    if not ledger_path.exists():
        print(f"suite-results.jsonl not found: {ledger_path}")
        sys.exit(1)

    ledger_entries: list[dict] = []
    results = []
    for line in ledger_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entry = json.loads(stripped)
        ledger_entries.append(entry)
        results.append(_eval_result_from_payload(entry.get("result") or {}))

    for entry, result in zip(ledger_entries, results):
        case_index = int(entry.get("case_index", 0) or 0)
        if case_index <= 0 or case_index > len(manifest.cases):
            continue
        case = manifest.cases[case_index - 1]
        if not result.success or not result.output_file:
            continue

        rulespec_file = Path(result.output_file)
        if not rulespec_file.exists():
            continue

        source_text = ""
        if case.kind == "source" and case.corpus_citation_path:
            source_text = resolve_corpus_source_unit(
                case.corpus_citation_path,
                corpus_path,
            ).body

        result.metrics = evaluate_artifact(
            rulespec_file=rulespec_file,
            policy_repo_root=rulespec_file.parents[1],
            axiom_rules_path=axiom_rules_path,
            source_text=source_text,
            oracle=case.oracle,
            policyengine_country=case.policyengine_country,
            policyengine_rule_hint=case.policyengine_rule_hint,
        )
        entry["result"] = result.to_dict()

    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.runner, []).append(result)

    effective_runners = manifest.runners
    if run_state:
        manifest_meta = run_state.get("manifest") or {}
        effective_runners = list(
            manifest_meta.get("effective_runners") or effective_runners
        )

    readiness = {
        runner: summarize_readiness(runner_results, manifest.gates)
        for runner, runner_results in grouped.items()
    }
    all_ready = all(summary.ready for summary in readiness.values())
    payload = _build_eval_suite_payload(
        manifest=manifest,
        effective_runners=effective_runners,
        results=results,
        readiness=readiness,
        all_ready=all_ready,
    )

    ledger_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in ledger_entries)
    )
    (source_output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (source_output / "summary.json").write_text(
        json.dumps(
            {
                "manifest": payload["manifest"],
                "readiness": payload["readiness"],
                "all_ready": all_ready,
            },
            indent=2,
        )
        + "\n"
    )

    if run_state:
        now_iso = _utc_now_iso()
        run_state["updated_at"] = now_iso
        run_state["revalidated_at"] = now_iso
        (source_output / "suite-run.json").write_text(
            json.dumps(run_state, indent=2, sort_keys=True) + "\n"
        )

    if args.json:
        print(json.dumps(payload, indent=2))
        sys.exit(0 if all_ready else 1)

    print(f"Manifest: {manifest.path}")
    print(f"Suite: {manifest.name}")
    print(f"Output root: {source_output}")
    print(f"Revalidated cases: {len(results)}")
    print()
    for runner, summary in readiness.items():
        print(f"{runner}: {'READY' if summary.ready else 'NOT READY'}")
        print(
            f"  cases={summary.total_cases} success={summary.success_rate:.1%} "
            f"compile={summary.compile_pass_rate:.1%} ci={summary.ci_pass_rate:.1%} "
            f"zero_ungrounded={summary.zero_ungrounded_rate:.1%} "
            f"generalist_review={summary.generalist_review_pass_rate:.1%}"
        )
        if summary.mean_generalist_review_score is not None:
            print(
                f"  mean_generalist_review_score={summary.mean_generalist_review_score:.2f}/10"
            )
        if summary.policyengine_case_count:
            print(
                f"  policyengine_cases={summary.policyengine_case_count} "
                f"pass_rate={(summary.policyengine_pass_rate or 0):.1%} "
                f"mean_score={(summary.mean_policyengine_score or 0):.1%}"
            )
        if summary.mean_estimated_cost_usd is not None:
            print(f"  mean_estimated_cost=${summary.mean_estimated_cost_usd:.4f}")
        for gate in summary.gate_results:
            print(_format_gate_result(gate))
    sys.exit(0 if all_ready else 1)


def _axiom_encode_repo_root() -> Path:
    """Return the repository root for the current axiom_encode checkout."""
    return Path(__file__).resolve().parents[2]


def _default_eval_suite_archive_root() -> Path:
    """Resolve the durable local archive root for suite outputs."""
    configured = os.getenv("AXIOM_ENCODE_EVAL_ARCHIVE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (_axiom_encode_repo_root() / "artifacts" / "eval-suites").resolve()


def _slugify_archive_name(value: str) -> str:
    """Convert an arbitrary archive label into a filesystem-safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "eval-suite"


def _unique_archive_dir(archive_root: Path, base_name: str) -> Path:
    """Return the first available archive directory under the root."""
    candidate = archive_root / base_name
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = archive_root / f"{base_name}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _rewrite_archive_paths_in_payload(payload, source_root: Path, archive_root: Path):
    """Rewrite absolute paths that pointed at the source suite root."""
    source_root_str = str(source_root)
    archive_root_str = str(archive_root)
    source_prefix = source_root_str + os.sep

    if isinstance(payload, dict):
        return {
            key: _rewrite_archive_paths_in_payload(value, source_root, archive_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rewrite_archive_paths_in_payload(value, source_root, archive_root)
            for value in payload
        ]
    if isinstance(payload, str):
        if payload == source_root_str:
            return archive_root_str
        if payload.startswith(source_prefix):
            return archive_root_str + payload[len(source_root_str) :]
    return payload


def _rewrite_archived_eval_suite_json_files(
    archive_dir: Path, source_root: Path
) -> list[str]:
    """Rewrite archived JSON payloads so artifact paths point at the archive."""
    rewritten_files: list[str] = []

    for filename in ["suite-run.json", "results.json", "summary.json"]:
        path = archive_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rewritten = _rewrite_archive_paths_in_payload(payload, source_root, archive_dir)
        if rewritten != payload:
            path.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")
            rewritten_files.append(filename)

    ledger_path = archive_dir / "suite-results.jsonl"
    if ledger_path.exists():
        rows = []
        changed = False
        for line in ledger_path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            rewritten = _rewrite_archive_paths_in_payload(
                payload, source_root, archive_dir
            )
            rows.append(rewritten)
            changed = changed or rewritten != payload
        if changed:
            ledger_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            )
            rewritten_files.append("suite-results.jsonl")

    return rewritten_files


def _build_eval_suite_archive_metadata(
    archive_dir: Path, source_root: Path, rewritten_files: list[str]
) -> dict:
    """Build a compact metadata record for an archived suite snapshot."""
    state_path = archive_dir / "suite-run.json"
    summary_path = archive_dir / "summary.json"
    results_path = archive_dir / "results.json"

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    manifest = (
        state.get("manifest")
        or summary.get("manifest")
        or results.get("manifest")
        or {}
    )

    return {
        "archived_at": _utc_now_iso(),
        "source_output": str(source_root),
        "archive_dir": str(archive_dir),
        "manifest": manifest,
        "status": state.get("status"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "total_cases": state.get("total_cases"),
        "completed_cases": state.get("completed_cases"),
        "result_count": state.get(
            "result_count", len(results.get("results", []) or [])
        ),
        "all_ready": summary.get("all_ready"),
        "rewritten_files": rewritten_files,
    }


def cmd_eval_suite_archive(args):
    """Archive an eval-suite output tree into the durable local registry."""
    source_output = args.source_output.expanduser().resolve()
    if not source_output.exists() or not source_output.is_dir():
        print(f"Eval suite output directory not found: {source_output}")
        sys.exit(1)

    state_path = source_output / "suite-run.json"
    if not state_path.exists():
        print(
            f"Not an eval-suite output directory (missing suite-run.json): {source_output}"
        )
        sys.exit(1)

    archive_root = (
        args.archive_root.expanduser().resolve()
        if args.archive_root is not None
        else _default_eval_suite_archive_root()
    )
    archive_root.mkdir(parents=True, exist_ok=True)

    base_name = _slugify_archive_name(args.name or source_output.name)
    archive_dir = _unique_archive_dir(archive_root, base_name)
    shutil.copytree(source_output, archive_dir)

    rewritten_files = _rewrite_archived_eval_suite_json_files(
        archive_dir, source_output
    )
    metadata = _build_eval_suite_archive_metadata(
        archive_dir=archive_dir,
        source_root=source_output,
        rewritten_files=rewritten_files,
    )

    metadata_path = archive_dir / "archive-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    index_path = archive_root / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")

    if args.json:
        print(json.dumps(metadata, indent=2))
        return

    print(f"Archived eval suite to {archive_dir}")
    print(f"Source: {source_output}")
    print(f"Archive root: {archive_root}")
    if rewritten_files:
        print(f"Rewrote archived paths in: {', '.join(rewritten_files)}")
    if metadata.get("status"):
        print(f"Status: {metadata['status']}")
    if (
        metadata.get("completed_cases") is not None
        and metadata.get("total_cases") is not None
    ):
        print(
            f"Progress: {metadata['completed_cases']}/{metadata['total_cases']} cases"
        )


def _ordered_runner_names(payload: dict) -> list[str]:
    """Preserve runner order from results/readiness payloads."""
    ordered: list[str] = []
    for result in payload.get("results", []) or []:
        runner = result.get("runner")
        if runner and runner not in ordered:
            ordered.append(runner)
    for runner in (payload.get("readiness") or {}).keys():
        if runner not in ordered:
            ordered.append(runner)
    return ordered


def _mean_numeric(values: list[float | int | None]) -> float | None:
    """Return the arithmetic mean for present numeric values."""
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(sum(filtered) / len(filtered), 6)


def _format_percent(value: float | None) -> str:
    """Format optional fractions as percentages."""
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _format_money(value: float | None) -> str:
    """Format optional dollar values for reports."""
    if value is None:
        return "n/a"
    return f"${value:.4f}"


def _format_duration_seconds(value_ms: float | None) -> str:
    """Format optional millisecond durations as seconds."""
    if value_ms is None:
        return "n/a"
    return f"{(value_ms / 1000):.1f}"


def _format_generalist_score(value: float | None) -> str:
    """Format optional 0-10 reviewer scores."""
    if value is None:
        return "n/a"
    return f"{value:.2f}/10"


def _build_eval_suite_report(
    payload: dict, left_runner: str, right_runner: str
) -> dict:
    """Build a structured pairwise report from eval-suite JSON output."""
    results = payload.get("results", []) or []
    readiness = payload.get("readiness", {}) or {}

    by_case: dict[str, dict[str, dict]] = {}
    for result in results:
        citation = result.get("citation")
        runner = result.get("runner")
        if not citation or not runner:
            continue
        by_case.setdefault(str(citation), {})[str(runner)] = result

    case_rows: list[dict] = []
    both_present = 0
    left_success_only = 0
    right_success_only = 0
    left_compile_only = 0
    right_compile_only = 0
    left_ci_only = 0
    right_ci_only = 0
    left_zero_ungrounded_only = 0
    right_zero_ungrounded_only = 0
    left_lower_cost = 0
    right_lower_cost = 0
    tied_cost = 0
    left_higher_pe = 0
    right_higher_pe = 0
    tied_pe = 0

    for citation in sorted(by_case):
        left = by_case[citation].get(left_runner)
        right = by_case[citation].get(right_runner)
        left_metrics = (left or {}).get("metrics") or {}
        right_metrics = (right or {}).get("metrics") or {}

        row = {
            "case": citation,
            "left_runner": left_runner,
            "right_runner": right_runner,
            "left_success": left.get("success") if left else None,
            "right_success": right.get("success") if right else None,
            "left_compile_pass": left_metrics.get("compile_pass"),
            "right_compile_pass": right_metrics.get("compile_pass"),
            "left_ci_pass": left_metrics.get("ci_pass"),
            "right_ci_pass": right_metrics.get("ci_pass"),
            "left_zero_ungrounded": (
                left_metrics.get("ungrounded_numeric_count") == 0
                if left is not None and left_metrics
                else None
            ),
            "right_zero_ungrounded": (
                right_metrics.get("ungrounded_numeric_count") == 0
                if right is not None and right_metrics
                else None
            ),
            "left_policyengine_score": left_metrics.get("policyengine_score"),
            "right_policyengine_score": right_metrics.get("policyengine_score"),
            "left_estimated_cost_usd": left.get("estimated_cost_usd") if left else None,
            "right_estimated_cost_usd": right.get("estimated_cost_usd")
            if right
            else None,
            "left_duration_ms": left.get("duration_ms") if left else None,
            "right_duration_ms": right.get("duration_ms") if right else None,
            "left_output_file": left.get("output_file") if left else None,
            "right_output_file": right.get("output_file") if right else None,
        }
        case_rows.append(row)

        if left is not None and right is not None:
            both_present += 1
            if row["left_success"] and not row["right_success"]:
                left_success_only += 1
            elif row["right_success"] and not row["left_success"]:
                right_success_only += 1

            if row["left_compile_pass"] and not row["right_compile_pass"]:
                left_compile_only += 1
            elif row["right_compile_pass"] and not row["left_compile_pass"]:
                right_compile_only += 1

            if row["left_ci_pass"] and not row["right_ci_pass"]:
                left_ci_only += 1
            elif row["right_ci_pass"] and not row["left_ci_pass"]:
                right_ci_only += 1

            if row["left_zero_ungrounded"] and not row["right_zero_ungrounded"]:
                left_zero_ungrounded_only += 1
            elif row["right_zero_ungrounded"] and not row["left_zero_ungrounded"]:
                right_zero_ungrounded_only += 1

            left_cost = row["left_estimated_cost_usd"]
            right_cost = row["right_estimated_cost_usd"]
            if left_cost is not None and right_cost is not None:
                if left_cost < right_cost:
                    left_lower_cost += 1
                elif right_cost < left_cost:
                    right_lower_cost += 1
                else:
                    tied_cost += 1

            left_pe = row["left_policyengine_score"]
            right_pe = row["right_policyengine_score"]
            if left_pe is not None and right_pe is not None:
                if left_pe > right_pe:
                    left_higher_pe += 1
                elif right_pe > left_pe:
                    right_higher_pe += 1
                else:
                    tied_pe += 1

    runner_summaries: dict[str, dict] = {}
    for runner in [left_runner, right_runner]:
        runner_results = [
            result for result in results if result.get("runner") == runner
        ]
        summary = dict(readiness.get(runner) or {})
        summary["mean_duration_ms"] = _mean_numeric(
            [result.get("duration_ms") for result in runner_results]
        )
        summary["case_count"] = len(runner_results)
        runner_summaries[runner] = summary

    return {
        "manifest": payload.get("manifest") or {},
        "left_runner": left_runner,
        "right_runner": right_runner,
        "runner_summaries": runner_summaries,
        "pairwise": {
            "paired_case_count": both_present,
            "left_success_only_count": left_success_only,
            "right_success_only_count": right_success_only,
            "left_compile_only_count": left_compile_only,
            "right_compile_only_count": right_compile_only,
            "left_ci_only_count": left_ci_only,
            "right_ci_only_count": right_ci_only,
            "left_zero_ungrounded_only_count": left_zero_ungrounded_only,
            "right_zero_ungrounded_only_count": right_zero_ungrounded_only,
            "left_lower_cost_count": left_lower_cost,
            "right_lower_cost_count": right_lower_cost,
            "tied_cost_count": tied_cost,
            "left_higher_policyengine_score_count": left_higher_pe,
            "right_higher_policyengine_score_count": right_higher_pe,
            "tied_policyengine_score_count": tied_pe,
        },
        "case_rows": case_rows,
    }


def _render_eval_suite_report_markdown(report: dict) -> str:
    """Render a human-readable pairwise report suitable for a paper appendix."""
    manifest = report.get("manifest") or {}
    left_runner = report["left_runner"]
    right_runner = report["right_runner"]
    left_summary = report["runner_summaries"].get(left_runner) or {}
    right_summary = report["runner_summaries"].get(right_runner) or {}
    pairwise = report.get("pairwise") or {}
    case_rows = report.get("case_rows") or []

    lines = [
        f"# {manifest.get('name', 'Eval suite')} model comparison",
        "",
        f"- Manifest: `{manifest.get('path', 'n/a')}`",
        f"- Left runner: `{left_runner}`",
        f"- Right runner: `{right_runner}`",
        "",
        "| Metric | " + left_runner + " | " + right_runner + " |",
        "| --- | ---: | ---: |",
        f"| Cases | {left_summary.get('total_cases', left_summary.get('case_count', 'n/a'))} | {right_summary.get('total_cases', right_summary.get('case_count', 'n/a'))} |",
        f"| Success rate | {_format_percent(left_summary.get('success_rate'))} | {_format_percent(right_summary.get('success_rate'))} |",
        f"| Compile pass rate | {_format_percent(left_summary.get('compile_pass_rate'))} | {_format_percent(right_summary.get('compile_pass_rate'))} |",
        f"| CI pass rate | {_format_percent(left_summary.get('ci_pass_rate'))} | {_format_percent(right_summary.get('ci_pass_rate'))} |",
        f"| Zero-ungrounded rate | {_format_percent(left_summary.get('zero_ungrounded_rate'))} | {_format_percent(right_summary.get('zero_ungrounded_rate'))} |",
        f"| Generalist review pass rate | {_format_percent(left_summary.get('generalist_review_pass_rate'))} | {_format_percent(right_summary.get('generalist_review_pass_rate'))} |",
        f"| Mean generalist review score | {_format_generalist_score(left_summary.get('mean_generalist_review_score'))} | {_format_generalist_score(right_summary.get('mean_generalist_review_score'))} |",
        f"| PolicyEngine pass rate | {_format_percent(left_summary.get('policyengine_pass_rate'))} | {_format_percent(right_summary.get('policyengine_pass_rate'))} |",
        f"| Mean PolicyEngine score | {_format_percent(left_summary.get('mean_policyengine_score'))} | {_format_percent(right_summary.get('mean_policyengine_score'))} |",
        f"| Mean estimated cost | {_format_money(left_summary.get('mean_estimated_cost_usd'))} | {_format_money(right_summary.get('mean_estimated_cost_usd'))} |",
        f"| Mean duration (s) | {_format_duration_seconds(left_summary.get('mean_duration_ms'))} | {_format_duration_seconds(right_summary.get('mean_duration_ms'))} |",
        "",
        "## Pairwise counts",
        "",
        "| Outcome | Count |",
        "| --- | ---: |",
        f"| Paired cases | {pairwise.get('paired_case_count', 0)} |",
        f"| {left_runner} success-only advantages | {pairwise.get('left_success_only_count', 0)} |",
        f"| {right_runner} success-only advantages | {pairwise.get('right_success_only_count', 0)} |",
        f"| {left_runner} compile-only advantages | {pairwise.get('left_compile_only_count', 0)} |",
        f"| {right_runner} compile-only advantages | {pairwise.get('right_compile_only_count', 0)} |",
        f"| {left_runner} CI-only advantages | {pairwise.get('left_ci_only_count', 0)} |",
        f"| {right_runner} CI-only advantages | {pairwise.get('right_ci_only_count', 0)} |",
        f"| {left_runner} lower-cost cases | {pairwise.get('left_lower_cost_count', 0)} |",
        f"| {right_runner} lower-cost cases | {pairwise.get('right_lower_cost_count', 0)} |",
        f"| Tied-cost cases | {pairwise.get('tied_cost_count', 0)} |",
        f"| {left_runner} higher-PE-score cases | {pairwise.get('left_higher_policyengine_score_count', 0)} |",
        f"| {right_runner} higher-PE-score cases | {pairwise.get('right_higher_policyengine_score_count', 0)} |",
        f"| Tied-PE-score cases | {pairwise.get('tied_policyengine_score_count', 0)} |",
        "",
        "## Case-level appendix",
        "",
        "| Case | "
        + left_runner
        + " compile | "
        + right_runner
        + " compile | "
        + left_runner
        + " PE | "
        + right_runner
        + " PE | "
        + left_runner
        + " cost | "
        + right_runner
        + " cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in case_rows:
        left_pe = row["left_policyengine_score"]
        right_pe = row["right_policyengine_score"]
        lines.append(
            "| "
            + row["case"]
            + " | "
            + (
                "pass"
                if row["left_compile_pass"]
                else "fail"
                if row["left_compile_pass"] is not None
                else "n/a"
            )
            + " | "
            + (
                "pass"
                if row["right_compile_pass"]
                else "fail"
                if row["right_compile_pass"] is not None
                else "n/a"
            )
            + " | "
            + (_format_percent(left_pe) if left_pe is not None else "n/a")
            + " | "
            + (_format_percent(right_pe) if right_pe is not None else "n/a")
            + " | "
            + _format_money(row["left_estimated_cost_usd"])
            + " | "
            + _format_money(row["right_estimated_cost_usd"])
            + " |"
        )

    return "\n".join(lines) + "\n"


def cmd_eval_suite_report(args):
    """Render a pairwise comparison report from eval-suite JSON output."""
    payload = json.loads(args.result_json.read_text())
    available_runners = _ordered_runner_names(payload)
    if not available_runners:
        print(f"No runner results found in {args.result_json}")
        sys.exit(1)

    left_runner = args.left_runner or (
        available_runners[0] if available_runners else None
    )
    right_runner = args.right_runner or (
        available_runners[1] if len(available_runners) > 1 else None
    )
    if not left_runner or not right_runner:
        print(
            "Need two runners to compare. Pass --left-runner and --right-runner or provide a two-runner suite JSON."
        )
        sys.exit(1)
    if left_runner not in available_runners or right_runner not in available_runners:
        print(
            f"Requested runners must exist in the suite JSON. Available: {', '.join(available_runners)}"
        )
        sys.exit(1)

    report = _build_eval_suite_report(payload, left_runner, right_runner)

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as fh:
            fieldnames = (
                list(report["case_rows"][0].keys()) if report["case_rows"] else []
            )
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(report["case_rows"])

    if args.json:
        rendered = json.dumps(report, indent=2)
    else:
        rendered = _render_eval_suite_report_markdown(report)

    if args.markdown_out and not args.json:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(rendered)

    print(rendered)


# =========================================================================
# Session Commands
# =========================================================================


def cmd_session_start(args):
    """Start a new session."""
    db = EncodingDB(args.db)
    from . import __version__

    session = db.start_session(
        model=args.model,
        cwd=args.cwd or str(Path.cwd()),
        axiom_encode_version=__version__,
    )

    # Output just the session ID for hooks to capture
    print(session.id)


def cmd_session_end(args):
    """End a session."""
    db = EncodingDB(args.db)
    db.end_session(args.session)
    print(f"Session {args.session} ended")


def cmd_log_event(args):
    """Log an event to a session."""
    db = EncodingDB(args.db)

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            pass

    event = db.log_event(
        session_id=args.session,
        event_type=args.type,
        tool_name=args.tool,
        content=args.content,
        metadata=metadata,
    )

    print(f"Event {event.sequence}: {event.event_type}")


def cmd_sessions(args):
    """List recent sessions."""
    db = EncodingDB(args.db)
    sessions = db.get_recent_sessions(limit=args.limit)

    if not sessions:
        print("No sessions found.")
        return

    print(
        f"{'ID':<10} {'Started':<20} {'Events':<8} {'Model':<15} {'Version':<10} {'Status'}"
    )
    print("-" * 82)

    for s in sessions:
        started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "?"
        status = "ended" if s.ended_at else "active"
        model = s.model[:15] if s.model else "-"
        version = s.axiom_encode_version[:10] if s.axiom_encode_version else "-"
        print(
            f"{s.id:<10} {started:<20} {s.event_count:<8} {model:<15} {version:<10} {status}"
        )


def cmd_session_show(args):
    """Show a session transcript."""
    db = EncodingDB(args.db)

    session = db.get_session(args.session_id)
    if not session:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    events = db.get_session_events(args.session_id)

    if args.json:
        output = {
            "session": {
                "id": session.id,
                "started_at": session.started_at.isoformat()
                if session.started_at
                else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "model": session.model,
                "cwd": session.cwd,
                "axiom_encode_version": session.axiom_encode_version,
                "event_count": session.event_count,
            },
            "events": [
                {
                    "sequence": e.sequence,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "type": e.event_type,
                    "tool": e.tool_name,
                    "content": e.content[:500]
                    if e.content
                    else "",  # Truncate long content
                    "metadata": e.metadata,
                }
                for e in events
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Session: {session.id}")
        print(f"Model: {session.model}")
        print(f"Axiom Encode: {session.axiom_encode_version or '-'}")
        print(f"Started: {session.started_at}")
        print(f"Ended: {session.ended_at or 'active'}")
        print(f"Events: {session.event_count}")
        print("-" * 60)

        for e in events:
            time_str = e.timestamp.strftime("%H:%M:%S") if e.timestamp else "?"
            tool_str = f" [{e.tool_name}]" if e.tool_name else ""
            content_preview = (
                (e.content[:80] + "...")
                if e.content and len(e.content) > 80
                else (e.content or "")
            )

            print(f"{e.sequence:3}. [{time_str}] {e.event_type}{tool_str}")
            if content_preview:
                print(f"     {content_preview}")


def cmd_session_stats(args):
    """Show session statistics."""
    db = EncodingDB(args.db)
    stats = db.get_session_stats()

    print("=== Session Statistics ===")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Avg events/session: {stats['avg_events_per_session']}")
    print()

    if stats["event_type_counts"]:
        print("Event types:")
        for event_type, count in sorted(
            stats["event_type_counts"].items(), key=lambda x: -x[1]
        ):
            print(f"  {event_type}: {count}")
        print()

    if stats["tool_usage"]:
        print("Top tools:")
        for tool, count in list(stats["tool_usage"].items())[:10]:
            print(f"  {tool}: {count}")


# =========================================================================
# Transcript Sync Commands
# =========================================================================


def cmd_sync_transcripts(args):
    """Sync local transcripts to Supabase."""
    from .supabase_sync import sync_transcripts_to_supabase

    print(
        f"Syncing transcripts{f' for session {args.session}' if args.session else ''}..."
    )

    try:
        stats = sync_transcripts_to_supabase(session_id=args.session)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(
            "Set AXIOM_ENCODE_SUPABASE_URL and "
            "AXIOM_ENCODE_SUPABASE_SECRET_KEY environment variables"
        )
        sys.exit(1)


def cmd_transcript_stats(args):
    """Show local transcript database stats."""
    from .supabase_sync import get_local_transcript_stats

    stats = get_local_transcript_stats()

    if not stats.get("exists"):
        print("No local transcript database found")
        print("Transcripts are created automatically when subagents complete")
        return

    print("=== Local Transcript Stats ===")
    print(f"Total transcripts: {stats['total']}")
    print(f"Unsynced: {stats['unsynced']}")
    print(f"Synced: {stats['synced']}")
    print()

    if stats.get("by_type"):
        print("By agent type:")
        for agent_type, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
            print(f"  {agent_type}: {count}")


def cmd_sync_agent_sessions(args):
    """Sync agent sessions to Supabase."""
    from .supabase_sync import sync_agent_sessions_to_supabase

    print(f"Syncing agent sessions{f' for {args.session}' if args.session else ''}...")

    try:
        stats = sync_agent_sessions_to_supabase(
            session_id=args.session,
            include_all=getattr(args, "all", False) is True,
        )
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )
    except ValueError as e:
        print(f"Error: {e}")
        print(
            "Set AXIOM_ENCODE_SUPABASE_URL and "
            "AXIOM_ENCODE_SUPABASE_SECRET_KEY environment variables"
        )
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
