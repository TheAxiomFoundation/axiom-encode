"""
AutoRAC CLI - Command line interface for statute encoding.

Primary workflow:
  1. autorac encode "26 USC 21" runs the full pipeline
  2. Orchestrator encodes, validates, reviews, and logs
  3. autorac validate <file.rac> runs standalone validation
  4. autorac log records encoding runs
  5. autorac stats shows patterns for improvement

Self-contained -- no external plugin dependencies.
"""

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .harness.encoding_db import (
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    ReviewResult,
    ReviewResults,
)
from .harness.evals import (
    load_eval_suite_manifest,
    run_akn_section_eval,
    run_eval_suite,
    run_legislation_gov_uk_section_eval,
    run_model_eval,
    run_source_eval,
    summarize_readiness,
)
from .harness.validator_pipeline import ValidatorPipeline
from .statute import parse_usc_citation

# Default DB path - can be overridden with --db
DEFAULT_DB = Path.home() / "RulesFoundation" / "autorac" / "encodings.db"


def _resolve_repo_checkout(name: str) -> Path:
    """Resolve sibling foundation repos before falling back to legacy defaults."""
    workspace_root = Path(__file__).resolve().parents[3]
    candidates = [
        workspace_root / name,
        Path.home() / "TheAxiomFoundation" / name,
        Path.home() / "RulesFoundation" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1]


def _reviewer_score_map(scores) -> dict[str, float | None]:
    """Normalize reviewer scores from either legacy flat attrs or ReviewResults.reviews."""
    reviewer_names = [
        "rac_reviewer",
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
            "Defaults to the AUTORAC_GPT_BACKEND env var when set."
        ),
    )


def _resolved_gpt_backend(args) -> str | None:
    """Resolve the requested GPT backend override from args/env."""
    return getattr(args, "gpt_backend", None) or os.getenv("AUTORAC_GPT_BACKEND") or None


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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAC - AI-assisted RAC encoding infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a .rac file (CI + reviewer agents)"
    )
    validate_parser.add_argument("file", type=Path, help="Path to .rac file")
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

    # log command
    log_parser = subparsers.add_parser("log", help="Log an encoding run to encoding DB")
    log_parser.add_argument("--citation", required=True, help="Legal citation")
    log_parser.add_argument(
        "--file", type=Path, required=True, help="Path to .rac file"
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
        help="Actual scores as JSON {rac,formula,param,integration}",
    )
    log_parser.add_argument(
        "--predicted",
        type=str,
        help="Predicted scores as JSON {rac,formula,param,integration,iterations,time}",
    )
    log_parser.add_argument(
        "--session", type=str, help="Session ID to link this run to"
    )
    log_parser.add_argument("--db", type=Path, default=Path("encodings.db"))

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show encoding statistics")
    stats_parser.add_argument("--db", type=Path, default=Path("encodings.db"))

    # calibration command
    calibration_parser = subparsers.add_parser(
        "calibration", help="Show calibration metrics (predicted vs actual)"
    )
    calibration_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    calibration_parser.add_argument("--limit", type=int, default=50)

    # statute command - extract from local USC XML
    statute_parser = subparsers.add_parser(
        "statute", help="Extract statute text from local USC XML (e.g., '26 USC 25B')"
    )
    statute_parser.add_argument(
        "citation", help="Citation like '26 USC 25B' or '26/25B'"
    )
    statute_parser.add_argument(
        "--xml-path",
        type=Path,
        default=Path.home() / "RulesFoundation" / "atlas" / "data" / "uscode",
        help="Path to USC XML files",
    )

    # runs command
    runs_parser = subparsers.add_parser("runs", help="List recent encoding runs")
    runs_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    runs_parser.add_argument("--limit", type=int, default=20)

    # init command - create stubs for all subsections
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize encoding: create stubs for all subsections with text from atlas",
    )
    init_parser.add_argument("citation", help="Citation like '26 USC 1' or '26/1'")
    init_parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "RulesFoundation" / "rac-us" / "statute",
        help="Output directory for .rac files",
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )

    # coverage command - check all subsections have been examined
    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Check encoding coverage: verify no subsections remain unexamined",
    )
    coverage_parser.add_argument("citation", help="Citation like '26 USC 1' or '26/1'")
    coverage_parser.add_argument(
        "--path",
        type=Path,
        default=Path.home() / "RulesFoundation" / "rac-us" / "statute",
        help="Path to statute directory",
    )

    # compile command
    compile_parser = subparsers.add_parser(
        "compile", help="Compile a .rac file to engine IR"
    )
    compile_parser.add_argument("file", type=Path, help="Path to .rac file")
    compile_parser.add_argument(
        "--as-of",
        default=None,
        help="Date for temporal resolution (YYYY-MM-DD, default: today)",
    )
    compile_parser.add_argument("--json", action="store_true", help="Output as JSON")
    compile_parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the compiled IR after compilation",
    )

    # benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark execution speed of a .rac file"
    )
    benchmark_parser.add_argument("file", type=Path, help="Path to .rac file")
    benchmark_parser.add_argument(
        "--as-of",
        default=None,
        help="Date for temporal resolution (YYYY-MM-DD, default: today)",
    )
    benchmark_parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations (default: 100)",
    )
    benchmark_parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of entity rows to generate (default: 1000)",
    )

    # encode command - run SDK orchestrator with full logging
    encode_parser = subparsers.add_parser(
        "encode", help="Encode a statute using SDK orchestrator with full logging"
    )
    encode_parser.add_argument(
        "citation", help="Statute citation (e.g., '26 USC 1(j)(2)')"
    )
    encode_parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "RulesFoundation" / "rac-us" / "statute",
        help="Output directory for .rac files",
    )
    encode_parser.add_argument(
        "--model",
        default=None,
        help="Model to use for encoding (default: autorac.DEFAULT_MODEL)",
    )
    encode_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    encode_parser.add_argument(
        "--backend",
        choices=["cli", "api", "openai"],
        default="cli",
        help=(
            "Backend: 'cli' uses Claude Code CLI (no API key), "
            "'api' uses Anthropic API (requires ANTHROPIC_API_KEY), "
            "'openai' uses OpenAI Responses API (requires OPENAI_API_KEY)"
        ),
    )
    encode_parser.add_argument(
        "--atlas-path",
        type=Path,
        default=None,
        help="Path to atlas repo (default: ATLAS_PATH env var or ~/RulesFoundation/atlas)",
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
        help="Runner spec [name=]backend:model. Defaults to claude:opus and codex:gpt-5.4",
    )
    _add_gpt_backend_argument(eval_parser)
    eval_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/autorac-evals"),
        help="Directory for eval artifacts and traces",
    )
    eval_parser.add_argument(
        "--atlas-path",
        type=Path,
        default=None,
        help="Path to atlas repo (defaults to sibling repo checkout)",
    )
    eval_parser.add_argument(
        "--rac-path",
        type=Path,
        default=None,
        help="Path to rac repo (defaults to sibling repo checkout)",
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
        help="Compare model runners on one arbitrary source-text slice",
    )
    eval_source_parser.add_argument("source_id", help="Logical identifier for the source slice")
    eval_source_parser.add_argument(
        "source_file",
        type=Path,
        help="Path to a text file containing the authoritative source text",
    )
    eval_source_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Runner spec [name=]backend:model. Defaults to claude:opus and codex:gpt-5.4",
    )
    _add_gpt_backend_argument(eval_source_parser)
    eval_source_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/autorac-evals"),
        help="Directory for eval artifacts and traces",
    )
    eval_source_parser.add_argument(
        "--rac-path",
        type=Path,
        default=None,
        help="Path to rac repo (defaults to sibling repo checkout)",
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
        "--policyengine-rac-var-hint",
        default=None,
        help="Canonical RAC variable name to use as the PolicyEngine oracle target for this source slice",
    )

    eval_akn_section_parser = subparsers.add_parser(
        "eval-akn-section",
        help="Compare model runners on one section extracted from Akoma Ntoso XML",
    )
    eval_akn_section_parser.add_argument(
        "source_id",
        help="Logical identifier for the source section",
    )
    eval_akn_section_parser.add_argument(
        "akn_file",
        type=Path,
        help="Path to an Akoma Ntoso XML document",
    )
    eval_akn_section_parser.add_argument(
        "section_eid",
        help="eId of the AKN hcontainer section to extract",
    )
    eval_akn_section_parser.add_argument(
        "--allow-parent",
        action="store_true",
        help="Allow encoding a parent section even when atomic child sections exist",
    )
    eval_akn_section_parser.add_argument(
        "--table-row-query",
        default=None,
        help="Filter extracted section tables down to the matching row plus local table context",
    )
    eval_akn_section_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Runner spec [name=]backend:model. Defaults to claude:opus and codex:gpt-5.4",
    )
    _add_gpt_backend_argument(eval_akn_section_parser)
    eval_akn_section_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/autorac-evals"),
        help="Directory for eval artifacts and traces",
    )
    eval_akn_section_parser.add_argument(
        "--rac-path",
        type=Path,
        default=None,
        help="Path to rac repo (defaults to sibling repo checkout)",
    )
    eval_akn_section_parser.add_argument(
        "--mode",
        choices=["cold", "repo-augmented"],
        default="repo-augmented",
        help="Whether the eval gets only extracted section text or a logged bundle of explicit precedent files",
    )
    eval_akn_section_parser.add_argument(
        "--allow-context",
        action="append",
        default=[],
        help="Extra file path to copy into the repo-augmented eval workspace (repeatable)",
    )
    eval_akn_section_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    eval_akn_section_parser.add_argument(
        "--policyengine-rac-var-hint",
        default=None,
        help="Canonical RAC variable name to use as the PolicyEngine oracle target for this section",
    )

    eval_uk_legislation_parser = subparsers.add_parser(
        "eval-uk-legislation-section",
        help="Fetch official legislation.gov.uk XML and compare model runners on one UK AKN section",
    )
    eval_uk_legislation_parser.add_argument(
        "source_ref",
        help="legislation.gov.uk URL or site-relative path",
    )
    eval_uk_legislation_parser.add_argument(
        "--section-eid",
        default=None,
        help="eId of the AKN section to extract. If omitted, use the sole top-level section from the fetched AKN document.",
    )
    eval_uk_legislation_parser.add_argument(
        "--allow-parent",
        action="store_true",
        help="Allow encoding a parent section even when atomic child sections exist",
    )
    eval_uk_legislation_parser.add_argument(
        "--table-row-query",
        default=None,
        help="Filter extracted section tables down to the matching row plus local table context",
    )
    eval_uk_legislation_parser.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Runner spec [name=]backend:model. Defaults to claude:opus and codex:gpt-5.4",
    )
    _add_gpt_backend_argument(eval_uk_legislation_parser)
    eval_uk_legislation_parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/autorac-evals"),
        help="Directory for fetched sources, eval artifacts, and traces",
    )
    eval_uk_legislation_parser.add_argument(
        "--rac-path",
        type=Path,
        default=None,
        help="Path to rac repo (defaults to sibling repo checkout)",
    )
    eval_uk_legislation_parser.add_argument(
        "--mode",
        choices=["cold", "repo-augmented"],
        default="repo-augmented",
        help="Whether the eval gets only extracted section text or a logged bundle of explicit precedent files",
    )
    eval_uk_legislation_parser.add_argument(
        "--allow-context",
        action="append",
        default=[],
        help="Extra file path to copy into the repo-augmented eval workspace (repeatable)",
    )
    eval_uk_legislation_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    eval_uk_legislation_parser.add_argument(
        "--policyengine-rac-var-hint",
        default=None,
        help="Canonical RAC variable name to use as the PolicyEngine oracle target for this section",
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
        default=Path("/tmp/autorac-suite-evals"),
        help="Directory for suite artifacts and traces",
    )
    eval_suite_parser.add_argument(
        "--atlas-path",
        type=Path,
        default=None,
        help="Path to atlas repo (needed for citation cases; defaults to sibling repo checkout)",
    )
    eval_suite_parser.add_argument(
        "--rac-path",
        type=Path,
        default=None,
        help="Path to rac repo (defaults to sibling repo checkout)",
    )
    eval_suite_parser.add_argument(
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
        help="Path to JSON emitted by `autorac eval-suite --json`",
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

    # sync-sdk-sessions command
    sync_sdk_parser = subparsers.add_parser(
        "sync-sdk-sessions", help="Sync SDK orchestrator sessions to Supabase"
    )
    sync_sdk_parser.add_argument(
        "--session", default=None, help="Only sync specific session"
    )

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "compile":
        cmd_compile(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "statute":
        cmd_statute(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "coverage":
        cmd_coverage(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "eval-source":
        cmd_eval_source(args)
    elif args.command == "eval-akn-section":
        cmd_eval_akn_section(args)
    elif args.command == "eval-uk-legislation-section":
        cmd_eval_uk_legislation_section(args)
    elif args.command == "eval-suite":
        cmd_eval_suite(args)
    elif args.command == "eval-suite-report":
        cmd_eval_suite_report(args)
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
    elif args.command == "sync-sdk-sessions":
        cmd_sync_sdk_sessions(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_validate(args):
    """Validate a .rac file."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    rac_file = args.file.resolve()

    # Find rac paths
    rac_us = rac_file.parent
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent

    if rac_us.name != "rac-us":
        rac_us = _resolve_repo_checkout("rac-us")
        rac_path = _resolve_repo_checkout("rac")
    else:
        rac_path = rac_us.parent / "rac"

    # Enable oracles if --oracle flag is set
    enable_oracles = args.oracle is not None

    pipeline = ValidatorPipeline(
        rac_us_path=rac_us,
        rac_path=rac_path,
        enable_oracles=enable_oracles,
    )

    result = pipeline.validate(rac_file, skip_reviewers=args.skip_reviewers)
    scores = result.to_actual_scores()
    review_scores = (
        {
            "rac_reviewer": None,
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

    # Overall pass requires regular checks AND oracle checks (if enabled)
    all_passed = result.all_passed and oracle_passed

    if args.json:
        output = {
            "file": str(rac_file),
            "ci_pass": result.ci_pass,
            "scores": {
                "rac_reviewer": review_scores["rac_reviewer"],
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
            "all_passed": all_passed,
            "errors": errors,
            "duration_ms": result.total_duration_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {rac_file}")
        print(f"CI: {'✓' if result.ci_pass else '✗'}")
        if not args.skip_reviewers:
            print(
                "Scores: "
                f"RAC {review_scores['rac_reviewer']}/10 | "
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
        print(f"Result: {'✓ PASSED' if all_passed else '✗ FAILED'}")
        if errors:
            for err in errors:
                print(f"  - {err}")

    sys.exit(0 if all_passed else 1)


def cmd_compile(args):
    """Compile a .rac file to engine IR."""
    from datetime import date as date_type

    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    # Parse as_of date
    if args.as_of:
        as_of = date_type.fromisoformat(args.as_of)
    else:
        as_of = date_type.today()

    try:
        from rac import compile as rac_compile
        from rac import execute as rac_execute
        from rac import parse

        # Step 1: Parse
        rac_content = args.file.read_text()
        module = parse(rac_content, path=str(args.file))

        # Step 2: Compile
        ir = rac_compile([module], as_of=as_of)

        var_names = list(ir.variables.keys())

        if args.execute:
            # Execute with empty data
            result = rac_execute(ir, {})
            scalars = dict(result.scalars) if hasattr(result, "scalars") else {}

        if args.json:
            output = {
                "success": True,
                "file": str(args.file),
                "as_of": str(as_of),
                "variables": var_names,
                "variable_count": len(var_names),
            }
            if args.execute:
                output["scalars"] = scalars
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"Compiled: {args.file}")
            print(f"Date: {as_of}")
            print(f"Variables: {len(var_names)}")
            for name in var_names:
                print(f"  - {name}")
            if args.execute:
                print("\nExecution results:")
                for k, v in scalars.items():
                    print(f"  {k} = {v}")
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


def cmd_benchmark(args):
    """Benchmark execution speed of a .rac file."""
    import time
    from datetime import date as date_type

    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    if args.as_of:
        as_of = date_type.fromisoformat(args.as_of)
    else:
        as_of = date_type.today()

    try:
        from rac import compile as rac_compile
        from rac import execute as rac_execute
        from rac import parse

        # Parse and compile once
        rac_content = args.file.read_text()
        module = parse(rac_content, path=str(args.file))
        ir = rac_compile([module], as_of=as_of)

        # Build test data with specified number of rows
        # Detect entity types from IR
        entity_data = {}
        for var in ir.variables.values():
            if hasattr(var, "entity") and var.entity:
                if var.entity not in entity_data:
                    entity_data[var.entity] = [
                        {"id": i, "income": 1000 * (i + 1)} for i in range(args.rows)
                    ]

        # If no entities, just use empty data
        data = entity_data if entity_data else {}

        # Warmup
        for _ in range(min(5, args.iterations)):
            rac_execute(ir, data)

        # Benchmark
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            rac_execute(ir, data)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        total_ms = sum(times)
        rows_per_sec = (
            (args.rows * args.iterations) / (total_ms / 1000)
            if total_ms > 0 and args.rows > 0
            else 0
        )

        print(f"Benchmark: {args.file}")
        print(f"Date: {as_of}")
        print(f"Variables: {len(ir.variables)}")
        print(f"Rows: {args.rows}")
        print(f"Iterations: {args.iterations}")
        print("")
        print(f"Avg: {avg_ms:.3f} ms/iteration")
        print(f"Min: {min_ms:.3f} ms")
        print(f"Max: {max_ms:.3f} ms")
        print(f"Total: {total_ms:.1f} ms")
        if rows_per_sec > 0:
            print(f"Throughput: {rows_per_sec:,.0f} rows/sec")

        sys.exit(0)

    except Exception as e:
        print(f"Benchmark failed: {e}")
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

    # Parse review results from --scores (backward compat: convert scores to checklist)
    review_results = None
    if args.scores:
        s = json.loads(args.scores)
        reviews = []
        for reviewer_name, key in [
            ("rac_reviewer", "rac"),
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

    # Read RAC content
    rac_content = ""
    if args.file.exists():
        rac_content = args.file.read_text()

    run = EncodingRun(
        citation=args.citation,
        file_path=str(args.file),
        review_results=review_results,
        iterations=iterations,
        total_duration_ms=args.duration,
        rac_content=rac_content,
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
            print("  → Add more test examples to RAC_SPEC.md")
        elif top_error[0] == "parse":
            print("  → Clarify syntax in RAC_SPEC.md")
        elif top_error[0] == "import":
            print("  → Document import patterns better")
    else:
        print("Not enough data yet. Run more encodings.")


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
        print("No runs with both predictions and actual scores yet.")
        print("Use --predicted flag when logging runs to enable calibration.")
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


def cmd_statute(args):
    """Extract statute text from local USC XML."""
    import html
    import re

    # Parse citation: "26 USC 25B" or "26/25B" or "26 25B"
    citation = args.citation.upper().replace("USC", "").replace("§", "").strip()
    parts = re.split(r"[\s/]+", citation)

    if len(parts) < 2:
        print(f"Error: Could not parse citation '{args.citation}'")
        print("Expected format: '26 USC 25B' or '26/25B'")
        sys.exit(1)

    title = int(parts[0])
    section = parts[1]

    xml_file = args.xml_path / f"usc{title}.xml"

    if not xml_file.exists():
        print(f"Error: USC Title {title} XML not found at {xml_file}")
        print(
            f"Available titles: {sorted([f.stem.replace('usc', '') for f in args.xml_path.glob('usc*.xml')])}"
        )
        sys.exit(1)

    content = xml_file.read_text()

    # Find the section
    identifier = f"/us/usc/t{title}/s{section}"
    start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
    start_match = re.search(start_pattern, content)

    if not start_match:
        print(f"Error: Section {identifier} not found in {xml_file.name}")
        sys.exit(1)

    # Find matching closing tag (handle nesting)
    start_pos = start_match.start()
    depth = 0
    end_pos = start_pos
    i = start_pos

    while i < len(content):
        if content[i : i + 8] == "<section":
            depth += 1
        elif content[i : i + 10] == "</section>":
            depth -= 1
            if depth == 0:
                end_pos = i + 10
                break
        i += 1

    xml_section = content[start_pos:end_pos]

    # Simple text extraction: strip all tags, preserve structure via identifiers
    def clean(text):
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)
        return " ".join(text.split()).strip()

    # Extract heading
    sec_head = re.search(r"<heading[^>]*>(.*?)</heading>", xml_section, re.DOTALL)

    print(f"=== {title} USC § {section} ===")
    if sec_head:
        print(f"{clean(sec_head.group(1))}")
    print()

    # Process the XML structure iteratively
    def extract_element(xml, tag, depth=0):
        """Extract elements of given tag type with proper nesting."""
        results = []
        pattern = rf'<{tag}[^>]*identifier="([^"]+)"[^>]*>'

        for match in re.finditer(pattern, xml):
            ident = match.group(1)

            # Find closing tag
            open_tag = f"<{tag}"
            close_tag = f"</{tag}>"
            d = 1
            j = match.end()
            while j < len(xml) and d > 0:
                if xml[j : j + len(open_tag)] == open_tag:
                    d += 1
                elif xml[j : j + len(close_tag)] == close_tag:
                    d -= 1
                j += 1
            end = j

            elem_xml = xml[match.end() : end - len(close_tag)]
            results.append((ident, elem_xml))

        return results

    # Extract subsections
    for sub_id, sub_xml in extract_element(xml_section, "subsection"):
        sub_letter = sub_id.split("/")[-1]
        sub_head = re.search(r"<heading[^>]*>(.*?)</heading>", sub_xml, re.DOTALL)
        sub_content = re.search(r"<content>(.*?)</content>", sub_xml, re.DOTALL)

        line = f"({sub_letter})"
        if sub_head:
            line += f" {clean(sub_head.group(1))}"
        print(line)

        if sub_content:
            text = clean(sub_content.group(1))
            if text and text not in line:
                print(f"    {text}")

        # Extract paragraphs
        for para_id, para_xml in extract_element(sub_xml, "paragraph"):
            para_num = para_id.split("/")[-1]
            para_head = re.search(r"<heading[^>]*>(.*?)</heading>", para_xml, re.DOTALL)
            para_content = re.search(r"<content>(.*?)</content>", para_xml, re.DOTALL)

            pline = f"  ({para_num})"
            if para_head:
                pline += f" {clean(para_head.group(1))}"
            elif para_content:
                pline += f" {clean(para_content.group(1))}"
            print(pline)

            # Extract subparagraphs
            for subp_id, subp_xml in extract_element(para_xml, "subparagraph"):
                subp_letter = subp_id.split("/")[-1]
                subp_head = re.search(
                    r"<heading[^>]*>(.*?)</heading>", subp_xml, re.DOTALL
                )
                subp_content = re.search(
                    r"<content>(.*?)</content>", subp_xml, re.DOTALL
                )

                sline = f"    ({subp_letter})"
                if subp_head:
                    sline += f" {clean(subp_head.group(1))}"
                elif subp_content:
                    sline += f" {clean(subp_content.group(1))}"
                print(sline)

        print()


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
# Init and Coverage Commands
# =========================================================================

SUPABASE_URL = "https://nsupqhfchdtqclomlrgs.supabase.co/rest/v1"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5zdXBxaGZjaGR0cWNsb21scmdzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY5MzExMDgsImV4cCI6MjA4MjUwNzEwOH0.BPdUadtBCdKfWZrKbfxpBQUqSGZ4hd34Dlor8kMBrVI"


def _extract_subsections_from_xml(xml_path: Path, section: str) -> list[dict]:
    """Extract all subsections from USC XML file.

    Returns list of dicts with: path, heading, body, line_count, depth
    """
    import html as html_module
    import re

    content = xml_path.read_text()

    # Find the section
    title = xml_path.stem.replace("usc", "")
    identifier = f"/us/usc/t{title}/s{section}"
    start_pattern = rf'<section[^>]*identifier="{re.escape(identifier)}"[^>]*>'
    start_match = re.search(start_pattern, content)

    if not start_match:
        return []

    # Find matching closing tag
    start_pos = start_match.start()
    depth = 0
    end_pos = start_pos
    i = start_pos

    while i < len(content):
        if content[i : i + 8] == "<section":
            depth += 1
        elif content[i : i + 10] == "</section>":
            depth -= 1
            if depth == 0:
                end_pos = i + 10
                break
        i += 1

    xml_section = content[start_pos:end_pos]

    def clean(text):
        text = re.sub(r"<[^>]+>", "", text)
        text = html_module.unescape(text)
        return " ".join(text.split()).strip()

    def extract_elements_recursive(xml, parent_path, depth=0):
        """Recursively extract all nested elements."""
        results = []

        # Tags to look for at each level
        tag_order = ["subsection", "paragraph", "subparagraph", "clause", "subclause"]
        if depth >= len(tag_order):
            return results

        tag = tag_order[depth]
        pattern = rf'<{tag}[^>]*identifier="([^"]+)"[^>]*>'

        for match in re.finditer(pattern, xml):
            ident = match.group(1)

            # Find closing tag
            open_tag = f"<{tag}"
            close_tag = f"</{tag}>"
            d = 1
            j = match.end()
            while j < len(xml) and d > 0:
                if xml[j : j + len(open_tag)] == open_tag:
                    d += 1
                elif xml[j : j + len(close_tag)] == close_tag:
                    d -= 1
                j += 1

            elem_xml = xml[match.end() : j - len(close_tag)]

            # Extract heading and content
            heading_match = re.search(
                r"<heading[^>]*>(.*?)</heading>", elem_xml, re.DOTALL
            )
            content_match = re.search(r"<content>(.*?)</content>", elem_xml, re.DOTALL)

            heading = clean(heading_match.group(1)) if heading_match else ""
            body = clean(content_match.group(1)) if content_match else ""

            # Build path from identifier (e.g., /us/usc/t26/s1/h/1/E -> 1/h/1/E)
            path_parts = ident.split("/")
            # Find section and take everything after
            try:
                sec_idx = next(i for i, p in enumerate(path_parts) if p.startswith("s"))
                local_path = "/".join(
                    [path_parts[sec_idx][1:]] + path_parts[sec_idx + 1 :]
                )
            except StopIteration:
                local_path = path_parts[-1]

            results.append(
                {
                    "source_path": f"usc/{title}/{local_path}",
                    "heading": heading,
                    "body": body,
                    "line_count": len(body.split("\n")) if body else 0,
                }
            )

            # Recurse into children
            children = extract_elements_recursive(elem_xml, local_path, depth + 1)
            results.extend(children)

        return results

    # Start extraction
    rules = extract_elements_recursive(xml_section, section, 0)

    # Also add the section itself if it has content
    sec_heading = re.search(
        r"<heading[^>]*>(.*?)</heading>", xml_section[:500], re.DOTALL
    )
    sec_content = re.search(r"<chapeau>(.*?)</chapeau>", xml_section, re.DOTALL)
    if sec_heading:
        rules.insert(
            0,
            {
                "source_path": f"usc/{title}/{section}",
                "heading": clean(sec_heading.group(1)),
                "body": clean(sec_content.group(1)) if sec_content else "",
                "line_count": 0,
            },
        )

    return rules


def cmd_init(args):
    """Initialize encoding: create stubs for all subsections with text from atlas.

    Uses local USC XML as primary source (faster, more reliable).
    """
    # Parse citation to get title/section
    citation = args.citation.replace(" ", "").upper()
    if "USC" in citation:
        parts = citation.split("USC")
        title = parts[0]
        section = parts[1]
    else:
        # Assume format like "26/1"
        parts = args.citation.split("/")
        title = parts[0]
        section = "/".join(parts[1:])

    # Use local USC XML
    xml_path = (
        Path.home()
        / "RulesFoundation"
        / "atlas"
        / "data"
        / "uscode"
        / f"usc{title}.xml"
    )
    if not xml_path.exists():
        print(f"USC XML not found: {xml_path}")
        print("Run: cd ~/RulesFoundation/atlas && python scripts/download_usc.py")
        sys.exit(1)

    print(f"Parsing {xml_path}...")
    rules = _extract_subsections_from_xml(xml_path, section)

    if not rules:
        print(f"No subsections found for section {section} in {xml_path}")
        sys.exit(1)

    print(f"Found {len(rules)} subsections for {args.citation}")

    # Build tree structure to determine depth and encoding order
    tree = {}
    for rule in rules:
        path = rule["source_path"].replace(f"usc/{title}/", "")
        parts = path.split("/")
        tree[path] = {
            "depth": len(parts),
            "heading": rule.get("heading", ""),
            "body": rule.get("body", ""),
            "line_count": rule.get("line_count", 0),
            "citation_path": rule.get("citation_path", ""),
        }

    # Sort by depth (deepest first), then alphabetically for encoding order
    sorted_paths = sorted(tree.keys(), key=lambda p: (-tree[p]["depth"], p))

    # Generate encoding sequence
    sequence = []
    for i, path in enumerate(sorted_paths, 1):
        info = tree[path]
        sequence.append(
            {
                "order": i,
                "path": path,
                "depth": info["depth"],
                "heading": info["heading"],
                "line_count": info["line_count"],
            }
        )

    # Create .rac stub files
    created = 0
    skipped = 0
    output_base = args.output / title

    for path, info in tree.items():
        # Convert path to filesystem path
        rac_path = output_base / f"{path}.rac"

        if rac_path.exists() and not args.force:
            skipped += 1
            continue

        # Create parent directories
        rac_path.parent.mkdir(parents=True, exist_ok=True)

        # Build stub content
        heading = info["heading"] or path.split("/")[-1]
        body = info["body"] or ""

        # Clean body text (remove HTML tags if present)
        import html
        import re

        body = html.unescape(body)
        body = re.sub(r"<[^>]+>", "", body)

        stub_content = f'''# {title} USC Section {path} - {heading}
# Status: unexamined - encoder must set disposition

status: unexamined

"""
{body[:3000]}{"..." if len(body) > 3000 else ""}
"""

# Encoder: Set status to one of:
#   encoded - has formula (replace this stub entirely)
#   stub - interface only, needs future work
#   skip - with skip_reason: "administrative" | "superseded" | "boilerplate"
#   consolidated - captured in parent/sibling file
'''
        rac_path.write_text(stub_content)
        created += 1

    # Write encoding sequence file
    sequence_path = output_base / section / "_encoding_sequence.yaml"
    sequence_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml

    sequence_content = {
        "citation": args.citation,
        "total_subsections": len(rules),
        "encoding_order": "leaf-first (deepest to shallowest)",
        "sequence": sequence,
    }
    sequence_path.write_text(
        yaml.dump(sequence_content, default_flow_style=False, sort_keys=False)
    )

    print(f"Created {created} stub files, skipped {skipped} existing")
    print(f"Encoding sequence written to: {sequence_path}")
    print("\nEncoding order (first 10):")
    for item in sequence[:10]:
        print(
            f"  {item['order']:3}. {item['path']} (depth={item['depth']}, {item['line_count']} lines)"
        )
    if len(sequence) > 10:
        print(f"  ... and {len(sequence) - 10} more")


def cmd_coverage(args):
    """Check encoding coverage: verify no subsections remain unexamined."""
    import re

    # Handle absolute paths directly
    if args.citation.startswith("/"):
        search_path = Path(args.citation)
    else:
        # Parse citation to get path
        citation = args.citation.replace(" ", "").upper()
        if "USC" in citation:
            parts = citation.split("USC")
            title = parts[0]
            section = parts[1]
        else:
            parts = args.citation.split("/")
            title = parts[0]
            section = "/".join(parts[1:])

        # Find all .rac files
        search_path = args.path / title / section
    if not search_path.exists():
        print(f"Path not found: {search_path}")
        sys.exit(1)

    rac_files = list(search_path.rglob("*.rac"))

    if not rac_files:
        print(f"No .rac files found in {search_path}")
        sys.exit(1)

    # Check each file for status
    unexamined = []
    examined = {"encoded": [], "stub": [], "skip": [], "consolidated": []}
    errors = []

    for rac_file in rac_files:
        if rac_file.name.startswith("_"):
            continue  # Skip sequence files

        try:
            content = rac_file.read_text()

            # Extract status
            status_match = re.search(r"^status:\s*(\w+)", content, re.MULTILINE)
            if not status_match:
                errors.append(f"{rac_file}: no status field")
                continue

            status = status_match.group(1).lower()

            if status == "unexamined":
                unexamined.append(rac_file)
            elif status in examined:
                examined[status].append(rac_file)
            else:
                errors.append(f"{rac_file}: unknown status '{status}'")

        except Exception as e:
            errors.append(f"{rac_file}: {e}")

    # Print summary
    total = len(rac_files)
    print(f"Coverage for {search_path}:")
    print(f"  Total files: {total}")
    print(f"  Encoded:     {len(examined['encoded'])}")
    print(f"  Stub:        {len(examined['stub'])}")
    print(f"  Skip:        {len(examined['skip'])}")
    print(f"  Consolidated:{len(examined['consolidated'])}")
    print(f"  Unexamined:  {len(unexamined)}")
    if errors:
        print(f"  Errors:      {len(errors)}")

    # Show unexamined files
    if unexamined:
        print(f"\nUnexamined files ({len(unexamined)}):")
        for f in unexamined[:20]:
            rel_path = f.relative_to(search_path)
            print(f"  - {rel_path}")
        if len(unexamined) > 20:
            print(f"  ... and {len(unexamined) - 20} more")

    if errors:
        print("\nErrors:")
        for e in errors[:10]:
            print(f"  - {e}")

    # Exit with error if any unexamined
    if unexamined:
        print(f"\n✗ INCOMPLETE: {len(unexamined)} subsections not examined")
        sys.exit(1)
    else:
        print("\n✓ COMPLETE: All subsections examined")
        sys.exit(0)


# =========================================================================
# Encode Command
# =========================================================================


def cmd_encode(args):
    """Encode a statute using the SDK orchestrator with full logging."""
    import asyncio
    from datetime import datetime

    from . import __version__
    from .harness.orchestrator import Orchestrator

    citation_parts = parse_usc_citation(args.citation)
    output_path = args.output / citation_parts.title / citation_parts.section
    if citation_parts.fragments[:-1]:
        output_path /= Path(*citation_parts.fragments[:-1])

    print(f"=== Encoding: {args.citation} ===")
    print(f"Output: {output_path}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"AutoRAC: {__version__}")
    print(f"DB: {args.db}")
    print()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator with chosen backend
    args.db.parent.mkdir(parents=True, exist_ok=True)
    orchestrator = Orchestrator(
        model=args.model,
        db_path=args.db,
        backend=args.backend,
        atlas_path=getattr(args, "atlas_path", None),
    )

    print(f"Starting encoding at {datetime.now().strftime('%H:%M:%S')}...")
    print("-" * 60)

    # Run encoding
    async def run_encode():
        return await orchestrator.encode(
            citation=args.citation,
            output_path=output_path,
        )

    run = asyncio.run(run_encode())

    print("-" * 60)
    print()

    # Print report
    report = orchestrator.print_report(run)
    print(report)

    # Show session ID for later lookup
    print()
    print(f"Session logged: {run.session_id}")
    print(f"View with: autorac session-show {run.session_id}")

    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"Files created: {len(run.files_created)}")
    if run.total_tokens:
        print(
            f"Total tokens: {run.total_tokens.input_tokens:,} in + {run.total_tokens.output_tokens:,} out"
        )
        cost = getattr(run, "total_cost_usd", None)
        if isinstance(cost, (int, float)) and cost > 0:
            print(f"Estimated cost: ${cost:.2f}")
        else:
            print(f"Estimated cost: ${run.total_tokens.estimated_cost_usd:.2f}")
    if run.oracle_pe_match is not None:
        print(f"PE match: {run.oracle_pe_match}%")
    if run.oracle_taxsim_match is not None:
        print(f"TAXSIM match: {run.oracle_taxsim_match}%")

    # Auto-sync to Supabase (skip silently if credentials not set)
    try:
        from .supabase_sync import sync_sdk_sessions_to_supabase

        stats = sync_sdk_sessions_to_supabase(session_id=run.session_id)
        print(f"Synced to Supabase: {stats['synced']} sessions")
    except ValueError:
        pass  # No Supabase credentials — skip sync
    except Exception as e:
        print(f"Supabase sync failed: {e}")

    # Return exit code based on success
    has_errors = any(a.error for a in run.agent_runs)
    sys.exit(1 if has_errors else 0)


def _default_repo_checkout(name: str) -> Path:
    """Resolve sibling repo checkouts relative to this autorac repo."""
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / name


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
        offenders = [
            item.raw for item in result.metrics.grounding if not item.grounded
        ]
        print(f"  ungrounded_values={', '.join(offenders[:10])}")


def cmd_eval(args):
    """Run deterministic model comparisons on one or more citations."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", "codex:gpt-5.4"], args
    )
    atlas_path = args.atlas_path or _default_repo_checkout("atlas")
    rac_path = args.rac_path or _default_repo_checkout("rac")

    if not atlas_path.exists():
        print(f"Atlas repo not found: {atlas_path}")
        sys.exit(1)
    if not rac_path.exists():
        print(f"rac repo not found: {rac_path}")
        sys.exit(1)

    results = run_model_eval(
        citations=args.citations,
        runner_specs=runners,
        output_root=args.output,
        rac_path=rac_path,
        atlas_path=atlas_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"Atlas: {atlas_path}")
    print(f"rac: {rac_path}")
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
    """Run deterministic model comparisons on one arbitrary source slice."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", "codex:gpt-5.4"], args
    )
    rac_path = args.rac_path or _default_repo_checkout("rac")

    if not rac_path.exists():
        print(f"rac repo not found: {rac_path}")
        sys.exit(1)
    if not args.source_file.exists():
        print(f"Source file not found: {args.source_file}")
        sys.exit(1)

    source_text = args.source_file.read_text()
    results = run_source_eval(
        source_id=args.source_id,
        source_text=source_text,
        runner_specs=runners,
        output_root=args.output,
        rac_path=rac_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
        policyengine_rac_var_hint=args.policyengine_rac_var_hint,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"rac: {rac_path}")
    print(f"Source: {args.source_file}")
    if args.policyengine_rac_var_hint:
        print(f"PolicyEngine RAC var hint: {args.policyengine_rac_var_hint}")
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


def cmd_eval_akn_section(args):
    """Run deterministic model comparisons on one AKN section."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", "codex:gpt-5.4"], args
    )
    rac_path = args.rac_path or _default_repo_checkout("rac")

    if not rac_path.exists():
        print(f"rac repo not found: {rac_path}")
        sys.exit(1)
    if not args.akn_file.exists():
        print(f"AKN file not found: {args.akn_file}")
        sys.exit(1)

    results = run_akn_section_eval(
        source_id=args.source_id,
        akn_file=args.akn_file,
        section_eid=args.section_eid,
        runner_specs=runners,
        output_root=args.output,
        rac_path=rac_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
        allow_parent=args.allow_parent,
        table_row_query=args.table_row_query,
        policyengine_rac_var_hint=args.policyengine_rac_var_hint,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"rac: {rac_path}")
    print(f"AKN file: {args.akn_file}")
    print(f"Section: {args.section_eid}")
    if args.table_row_query:
        print(f"Table row query: {args.table_row_query}")
    if args.policyengine_rac_var_hint:
        print(f"PolicyEngine RAC var hint: {args.policyengine_rac_var_hint}")
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


def cmd_eval_uk_legislation_section(args):
    """Run deterministic model comparisons on official UK legislation XML."""
    runners = _effective_runner_specs(
        args.runner or ["claude:opus", "codex:gpt-5.4"], args
    )
    rac_path = args.rac_path or _default_repo_checkout("rac")

    if not rac_path.exists():
        print(f"rac repo not found: {rac_path}")
        sys.exit(1)

    results = run_legislation_gov_uk_section_eval(
        source_ref=args.source_ref,
        section_eid=args.section_eid,
        runner_specs=runners,
        output_root=args.output,
        rac_path=rac_path,
        mode=args.mode,
        extra_context_paths=[Path(path) for path in args.allow_context],
        allow_parent=args.allow_parent,
        table_row_query=args.table_row_query,
        policyengine_rac_var_hint=args.policyengine_rac_var_hint,
    )

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return

    print(f"Output root: {args.output}")
    print(f"rac: {rac_path}")
    print(f"Source: {args.source_ref}")
    if args.section_eid:
        print(f"Section: {args.section_eid}")
    if args.table_row_query:
        print(f"Table row query: {args.table_row_query}")
    if args.policyengine_rac_var_hint:
        print(f"PolicyEngine RAC var hint: {args.policyengine_rac_var_hint}")
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


def cmd_eval_suite(args):
    """Run a manifest-driven benchmark suite and evaluate readiness gates."""
    manifest = load_eval_suite_manifest(args.manifest)
    effective_runners = _effective_runner_specs(args.runner or manifest.runners, args)
    rac_path = args.rac_path or _default_repo_checkout("rac")
    atlas_path = args.atlas_path or _default_repo_checkout("atlas")

    if not rac_path.exists():
        print(f"rac repo not found: {rac_path}")
        sys.exit(1)

    has_citation_case = any(case.kind == "citation" for case in manifest.cases)
    if has_citation_case and not atlas_path.exists():
        print(f"Atlas repo not found: {atlas_path}")
        sys.exit(1)

    results = run_eval_suite(
        manifest=manifest,
        output_root=args.output,
        rac_path=rac_path,
        atlas_path=atlas_path if has_citation_case else None,
        runner_specs=effective_runners,
    )

    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.runner, []).append(result)

    readiness = {
        runner: summarize_readiness(runner_results, manifest.gates)
        for runner, runner_results in grouped.items()
    }
    all_ready = all(summary.ready for summary in readiness.values())

    if args.json:
        print(
            json.dumps(
                {
                    "manifest": {
                        "name": manifest.name,
                        "path": str(manifest.path),
                        "runners": manifest.runners,
                        "effective_runners": effective_runners,
                    },
                    "results": [result.to_dict() for result in results],
                    "readiness": {
                        runner: asdict(summary)
                        for runner, summary in readiness.items()
                    },
                    "all_ready": all_ready,
                },
                indent=2,
            )
        )
        sys.exit(0 if all_ready else 1)

    print(f"Manifest: {manifest.path}")
    print(f"Suite: {manifest.name}")
    print(f"Output root: {args.output}")
    print(f"Runners: {', '.join(effective_runners)}")
    print(f"rac: {rac_path}")
    if has_citation_case:
        print(f"Atlas: {atlas_path}")
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


def _build_eval_suite_report(payload: dict, left_runner: str, right_runner: str) -> dict:
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
            "right_estimated_cost_usd": right.get("estimated_cost_usd") if right else None,
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
        runner_results = [result for result in results if result.get("runner") == runner]
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
        "| Metric | "
        + left_runner
        + " | "
        + right_runner
        + " |",
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
            + ("pass" if row["left_compile_pass"] else "fail" if row["left_compile_pass"] is not None else "n/a")
            + " | "
            + ("pass" if row["right_compile_pass"] else "fail" if row["right_compile_pass"] is not None else "n/a")
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

    left_runner = args.left_runner or (available_runners[0] if available_runners else None)
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
            fieldnames = list(report["case_rows"][0].keys()) if report["case_rows"] else []
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
        autorac_version=__version__,
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
        version = s.autorac_version[:10] if s.autorac_version else "-"
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
                "autorac_version": session.autorac_version,
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
        print(f"AutoRAC: {session.autorac_version or '-'}")
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
        print("Set RAC_SUPABASE_URL and RAC_SUPABASE_SECRET_KEY environment variables")
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


def cmd_sync_sdk_sessions(args):
    """Sync SDK orchestrator sessions to Supabase."""
    from .supabase_sync import sync_sdk_sessions_to_supabase

    print(f"Syncing SDK sessions{f' for {args.session}' if args.session else ''}...")

    try:
        stats = sync_sdk_sessions_to_supabase(session_id=args.session)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Set RAC_SUPABASE_URL and RAC_SUPABASE_SECRET_KEY environment variables")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
