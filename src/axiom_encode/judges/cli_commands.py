"""CLI surface for the judge stages.

Kept in the judges package (not inline in the 1.8MB ``cli.py``) so the encoder's
public command surface only needs a two-line hook: call
:func:`register_judge_subparsers` after building the subparsers, and route via
:func:`dispatch` for commands in :data:`COMMANDS`.

Commands:

* ``preclassify``       — worklist triage (bulk-encode.yml routing workflow).
* ``judge-fidelity``    — corpus-bound Stage 1 referee.
* ``judge-grid``        — Stage 2 grid-adequacy judge.
* ``judge-disposition`` — Stage 3 disposition referee.
* ``drift-check``       — golden-regeneration drift check.
* ``publish-drift-report`` — file GitHub issues from an isolated JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from axiom_encode.constants import RULESPEC_ATOMIC_MODULE_ROOTS
from axiom_encode.corpus_resolver import (
    LocalCorpusRelease,
    ResolvedCorpusSource,
    require_canonical_corpus_citation_path,
    resolve_local_corpus_source,
)
from axiom_encode.repo_routing import is_composition_policy_repo_root
from axiom_encode.toolchain import load_rulespec_local_corpus_release

from .regeneration import (
    SUPPORTED_BACKENDS,
    NoReplayableManifestError,
    RegenerationInputError,
    UnsafeRegenerationPath,
    VerificationSupervisorConfig,
    read_citation,
    read_resolvable_citation,
    redact_sensitive_text,
    require_citation_derived_output_path,
    validate_module_path,
    verification_supervisor_command,
)

_MAX_DRIFT_REPORT_BYTES = 20 * 1024 * 1024
_RULESPEC_ATOMIC_MODULE_ROOTS = RULESPEC_ATOMIC_MODULE_ROOTS

COMMANDS = frozenset(
    {
        "preclassify",
        "judge-fidelity",
        "judge-grid",
        "judge-disposition",
        "drift-check",
        "publish-drift-report",
    }
)


def _add_run_log_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--run-id``/``--log-dir`` so a stage folds into the per-run run log.

    When ``--run-id`` is given the emitted judge verdict is appended (as a
    canonical ``axiom_encode.run_log.v1`` ``judge`` event) to the same
    ``run_id.jsonl`` as the encode run, so the per-run DAG and funnel fold it.
    """

    parser.add_argument(
        "--run-id",
        help="Encode run id; append the judge event to that run's run log.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Run-log directory (defaults to the standard run-log dir).",
    )


def _add_corpus_binding_args(parser: argparse.ArgumentParser) -> None:
    """Require one canonical RuleSpec checkout and its pinned corpus release."""

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="exact canonical rulespec-<country> checkout with a release pin",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        required=True,
        help="local axiom-corpus checkout containing the pinned named release",
    )


def register_judge_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add the judge-stage subcommands to an existing subparsers object."""

    pre = subparsers.add_parser(
        "preclassify",
        help="Triage a worklist before generation (amendment/xref -> skip).",
    )
    _add_corpus_binding_args(pre)
    pre_source = pre.add_mutually_exclusive_group(required=True)
    pre_source.add_argument(
        "--worklist",
        type=Path,
        help="JSON list of canonical corpus_citation_path strings",
    )
    pre_source.add_argument(
        "--corpus-citation-path",
        help="single exact canonical corpus citation path",
    )
    pre.add_argument(
        "--no-llm", action="store_true", help="Heuristics only, no arbiter"
    )
    _add_run_log_args(pre)
    pre.add_argument("--json", action="store_true")

    fid = subparsers.add_parser(
        "judge-fidelity",
        help="Corpus-bound statutory-fidelity referee.",
    )
    _add_corpus_binding_args(fid)
    fid.add_argument("--corpus-citation-path", required=True)
    fid.add_argument("--rule-file", type=Path, required=True)
    fid.add_argument("--rule-path")
    _add_run_log_args(fid)
    fid.add_argument("--json", action="store_true")

    grid = subparsers.add_parser(
        "judge-grid", help="Grid-adequacy judge for an oracle suite."
    )
    _add_corpus_binding_args(grid)
    grid.add_argument("--corpus-citation-path", required=True)
    grid.add_argument(
        "--suite-file", type=Path, required=True, help="JSON list of cases"
    )
    grid.add_argument("--suite-name")
    grid.add_argument("--emit-cells", action="store_true", help="Print follow-up cells")
    _add_run_log_args(grid)
    grid.add_argument("--json", action="store_true")

    disp = subparsers.add_parser(
        "judge-disposition", help="Disposition referee (>=3 sampled-record arithmetic)."
    )
    disp.add_argument("--disposition-file", type=Path, required=True)
    disp.add_argument("--tolerance", type=float, default=0.01)
    _add_run_log_args(disp)
    disp.add_argument("--json", action="store_true")

    drift = subparsers.add_parser(
        "drift-check", help="Golden-regeneration drift check over merged modules."
    )
    drift.add_argument(
        "--root",
        type=Path,
        help="exact canonical rulespec-<country> checkout to sample modules from",
    )
    drift.add_argument("--verification-supervisor", type=Path)
    drift.add_argument("--verification-trusted-roots", type=Path)
    drift.add_argument(
        "--verification-runtime-root", type=Path, action="append", default=[]
    )
    drift.add_argument(
        "--verification-import-root", type=Path, action="append", default=[]
    )
    drift.add_argument("--verification-package-root", type=Path)
    drift.add_argument("--verification-launcher", type=Path)
    drift.add_argument(
        "--corpus-path",
        type=Path,
        help="local axiom-corpus checkout used to preflight regeneration sources",
    )
    drift.add_argument(
        "--axiom-rules-engine-path",
        dest="axiom_rules_path",
        type=Path,
        help=(
            "exact axiom-rules-engine checkout used by live regeneration "
            "(required with --regenerate)"
        ),
    )
    drift.add_argument(
        "--modules-file", type=Path, help="JSON map module_id -> merged yaml path"
    )
    drift.add_argument("--k", type=int, default=3)
    drift.add_argument("--seed", type=int)
    mode = drift.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate with the packaged encoder using a fixed argv.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Identity regeneration (wiring check only; never reports drift)",
    )
    drift.add_argument(
        "--regenerate-backend",
        choices=SUPPORTED_BACKENDS,
        default="openai",
        help="Encoder backend for live regeneration (default: openai).",
    )
    drift.add_argument(
        "--report-file",
        type=Path,
        help="write the sanitized JSON report for an isolated publisher job",
    )
    drift.add_argument("--json", action="store_true")

    publish = subparsers.add_parser(
        "publish-drift-report",
        help="Create GitHub issues from a sanitized golden-drift report.",
    )
    publish.add_argument("--report-file", type=Path, required=True)
    publish.add_argument("--repo", default="TheAxiomFoundation/axiom-encode")
    publish.add_argument("--json", action="store_true")


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "preclassify":
        return cmd_preclassify(args)
    if args.command == "judge-fidelity":
        return cmd_judge_fidelity(args)
    if args.command == "judge-grid":
        return cmd_judge_grid(args)
    if args.command == "judge-disposition":
        return cmd_judge_disposition(args)
    if args.command == "drift-check":
        return cmd_drift_check(args)
    if args.command == "publish-drift-report":
        return cmd_publish_drift_report(args)
    raise ValueError(f"not a judge command: {args.command}")


# -- commands -------------------------------------------------------------


def cmd_preclassify(args: argparse.Namespace) -> int:
    from . import preclassifier as pc

    sources = _load_worklist_sources(args)
    if sources is None:
        return 2
    entries = [
        {"citation": source.requested, "source_text": source.body} for source in sources
    ]
    results = pc.classify_batch(entries, use_llm=not args.no_llm)
    for result, source in zip(results, sources, strict=True):
        result.event.extra["source_attestation"] = source.to_attestation()
    if getattr(args, "run_id", None):
        from axiom_encode.run_log import RunLogWriter

        writer = RunLogWriter(args.run_id, log_dir=getattr(args, "log_dir", None))
        for r in results:
            r.event.emit(writer)
    generate = [r for r in results if r.route == pc.GENERATE]
    skip = [r for r in results if r.route == pc.SKIP]
    payload = {
        "total": len(results),
        "generate": len(generate),
        "skip": len(skip),
        "results": [
            {
                "entry_ref": r.entry_ref,
                "classification": r.classification.value,
                "route": r.route,
                "reason": r.reason,
                "confidence": r.confidence,
                "method": r.method,
                "event": r.event.to_dict(),
            }
            for r in results
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"Pre-classified {len(results)} worklist entries: "
            f"{len(generate)} generate, {len(skip)} skip"
        )
        for r in results:
            print(
                f"  [{r.route:8s}] {r.classification.value:15s} {r.entry_ref}: {r.reason}"
            )
    # Advisory pre-step: never fails the workflow; the entries are routed, not
    # dropped. Enforce that invariant with a hard check (not an assert, which -O
    # would strip).
    if len(results) != len(entries):
        raise RuntimeError(
            f"pre-classifier returned {len(results)} results for "
            f"{len(entries)} entries; entries must never be dropped"
        )
    return 0


def cmd_judge_fidelity(args: argparse.Namespace) -> int:
    from . import statutory_fidelity as sf

    source = _load_bound_source(args, args.corpus_citation_path)
    if source is None:
        return 2
    event = sf.run(
        source.body,
        _read(args.rule_file),
        citation=source.requested,
        rule_path=args.rule_path,
    )
    event.extra["source_attestation"] = source.to_attestation()
    _emit_event(event, args.json, args)
    return 0


def cmd_judge_grid(args: argparse.Namespace) -> int:
    from . import grid_adequacy as ga

    source = _load_bound_source(args, args.corpus_citation_path)
    if source is None:
        return 2
    case_grid = json.loads(_read(args.suite_file))
    event = ga.run(source.body, case_grid, suite_name=args.suite_name)
    event.extra["source_attestation"] = source.to_attestation()
    if args.emit_cells and not args.json:
        for cell in ga.gaps_to_cells(event):
            print(json.dumps(cell, default=str))
        return 0
    _emit_event(event, args.json, args)
    return 0


def cmd_judge_disposition(args: argparse.Namespace) -> int:
    from . import disposition as dp

    data = json.loads(_read(args.disposition_file))
    disposition = dp.Disposition(
        disposition_id=str(data.get("disposition_id", "")),
        claim=str(data.get("claim", "")),
        residual=float(data.get("residual", 0.0)),
        variable=data.get("variable"),
        records=list(data.get("records", [])),
    )
    event = dp.run(disposition, tolerance=args.tolerance)
    _emit_event(event, args.json, args)
    return 0


def cmd_drift_check(args: argparse.Namespace) -> int:
    from . import drift

    delegation_values = (
        getattr(args, "verification_supervisor", None),
        getattr(args, "verification_trusted_roots", None),
        tuple(getattr(args, "verification_runtime_root", ())),
        tuple(getattr(args, "verification_import_root", ())),
        getattr(args, "verification_package_root", None),
        getattr(args, "verification_launcher", None),
    )
    if args.regenerate and (
        args.root is None
        or args.modules_file is not None
        or args.corpus_path is None
        or getattr(args, "axiom_rules_path", None) is None
        or not all(delegation_values)
    ):
        print(
            "live drift regeneration requires --root, --corpus-path, and "
            "--axiom-rules-engine-path, and complete verification-supervisor "
            "delegation paths and does not accept --modules-file",
            file=sys.stderr,
        )
        return 2
    verification_supervisor = None
    if args.regenerate:
        verification_supervisor = VerificationSupervisorConfig(
            executable=args.verification_supervisor,
            trusted_roots=args.verification_trusted_roots,
            runtime_roots=tuple(args.verification_runtime_root),
            import_roots=tuple(args.verification_import_root),
            package_root=args.verification_package_root,
            launcher=args.verification_launcher,
        )
        try:
            verification_supervisor_command(verification_supervisor, [])
        except RegenerationInputError as exc:
            print(f"invalid verification-supervisor delegation: {exc}", file=sys.stderr)
            return 2
    modules = _load_drift_modules(args)
    if modules is None:
        return 2
    if args.root:
        modules = _select_corpus_backed_modules(args, modules)
        if modules is None:
            return 2
    if args.dry_run:

        def regenerate(_module: str, merged: str) -> str:
            return merged  # identity — wiring check only

        note = "DRY RUN (identity regeneration; drift never reported)"
    elif args.regenerate:
        from .regeneration import regenerate_module

        def regenerate(module: str, merged: str) -> str:
            return regenerate_module(
                module,
                merged,
                root=args.root,
                corpus_path=args.corpus_path,
                axiom_rules_path=args.axiom_rules_path,
                verification_supervisor=verification_supervisor,
                backend=args.regenerate_backend,
            )

        note = f"packaged regenerator ({args.regenerate_backend})"
    else:
        raise RuntimeError("argparse must select --regenerate or --dry-run")

    report = drift.run_drift_check(modules, regenerate, k=args.k, seed=args.seed)
    out = report.to_dict()
    try:
        drift.DriftReport.from_dict(out)
    except ValueError as exc:
        print(f"generated invalid drift report: {exc}", file=sys.stderr)
        return 2
    out["note"] = note
    serialized = redact_sensitive_text(json.dumps(out, indent=2, default=str))
    if len((serialized + "\n").encode("utf-8")) > _MAX_DRIFT_REPORT_BYTES:
        print(
            f"generated drift report exceeds the {_MAX_DRIFT_REPORT_BYTES}-byte "
            "safety limit",
            file=sys.stderr,
        )
        return 2
    if args.report_file:
        Path(args.report_file).write_text(serialized + "\n", encoding="utf-8")

    if args.json:
        print(serialized)
    else:
        print(
            f"Drift check ({note}): {report.to_dict()['n_checked']} checked, "
            f"{len(report.drifted)} drifted, {len(report.errors)} errors"
        )
        for r in report.drifted:
            print(f"  DRIFT {r.module}: {r.diff_count} diffs")
        for r in report.errors:
            print(f"  ERROR {r.module}: {redact_sensitive_text(r.error or '')}")
    # Drift or a regeneration error is a signal worth a non-zero exit for CI.
    return 1 if (report.drifted or report.errors) else 0


def cmd_publish_drift_report(args: argparse.Namespace) -> int:
    """Publish a sanitized report in a process that has no model credential."""

    from . import drift

    try:
        report_path = Path(args.report_file)
        if report_path.is_symlink() or not report_path.is_file():
            raise ValueError("report is not a regular file")
        if report_path.stat().st_size > _MAX_DRIFT_REPORT_BYTES:
            raise ValueError(
                f"report exceeds the {_MAX_DRIFT_REPORT_BYTES}-byte safety limit"
            )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = drift.DriftReport.from_dict(payload)
    except (OSError, ValueError, RecursionError) as exc:
        print(f"invalid drift report {args.report_file}: {exc}", file=sys.stderr)
        return 2

    expected = len(report.drifted) + len(report.errors)
    created = _create_drift_issues(report, args.repo) if expected else []
    output = report.to_dict()
    output["issues_created"] = created
    serialized = redact_sensitive_text(json.dumps(output, indent=2, default=str))
    if args.json:
        print(serialized)
    else:
        print(
            f"Published {len(created)} of {expected} golden-regeneration issues "
            f"to {args.repo}"
        )
    if len(created) != expected:
        return 2
    return 1 if expected else 0


# -- helpers --------------------------------------------------------------


def _read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _write_run_log(event, args: argparse.Namespace) -> None:
    """Append a judge event to the per-run run log when ``--run-id`` is given.

    Never raises: the canonical writer captures any IO error internally so a
    logging problem can never abort a judge run.
    """

    run_id = getattr(args, "run_id", None)
    if not run_id:
        return
    from axiom_encode.run_log import RunLogWriter

    writer = RunLogWriter(run_id, log_dir=getattr(args, "log_dir", None))
    event.emit(writer)


def _emit_event(event, as_json: bool, args: argparse.Namespace | None = None) -> None:
    if args is not None:
        _write_run_log(event, args)
    if as_json:
        print(json.dumps(event.to_dict(), indent=2, default=str))
        return
    print(
        f"stage={event.stage.value} verdict={event.verdict.value} "
        f"confidence={event.confidence} model={event.model} "
        f"escalated={event.escalated}"
    )
    for f in event.findings:
        print(f"  - [{f.kind}] {f.clause_ref} @ {f.rule_path}: {f.explanation}")
    if event.judge_error:
        print(f"  judge_error: {event.judge_error.type}: {event.judge_error.message}")


def _load_bound_corpus_release(args: argparse.Namespace) -> LocalCorpusRelease | None:
    try:
        root = Path(args.root).resolve(strict=True)
        if not is_composition_policy_repo_root(root):
            raise ValueError(
                "root must be an exact canonical rulespec-<country> checkout"
            )
        return load_rulespec_local_corpus_release(root, Path(args.corpus_path))
    except (OSError, ValueError) as exc:
        print(f"invalid corpus binding: {exc}", file=sys.stderr)
        return None


def _resolve_bound_source(
    citation_path: object,
    release: LocalCorpusRelease,
) -> ResolvedCorpusSource:
    if not isinstance(citation_path, str):
        raise ValueError("corpus_citation_path must be a string")
    canonical = require_canonical_corpus_citation_path(citation_path)
    return resolve_local_corpus_source(canonical, release)


def _load_bound_source(
    args: argparse.Namespace,
    citation_path: object,
) -> ResolvedCorpusSource | None:
    release = _load_bound_corpus_release(args)
    if release is None:
        return None
    try:
        return _resolve_bound_source(citation_path, release)
    except (OSError, ValueError) as exc:
        print(f"invalid corpus source: {exc}", file=sys.stderr)
        return None


def _load_worklist_sources(
    args: argparse.Namespace,
) -> list[ResolvedCorpusSource] | None:
    raw_citation_paths: object
    if args.worklist:
        try:
            raw_citation_paths = json.loads(_read(args.worklist))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"invalid worklist JSON: {exc}", file=sys.stderr)
            return None
    elif args.corpus_citation_path:
        raw_citation_paths = [args.corpus_citation_path]
    else:
        print(
            "preclassify needs --worklist or --corpus-citation-path",
            file=sys.stderr,
        )
        return None
    if not isinstance(raw_citation_paths, list) or not raw_citation_paths:
        print(
            "worklist JSON must be a nonempty list of canonical corpus citation "
            "path strings",
            file=sys.stderr,
        )
        return None
    if all(isinstance(item, str) for item in raw_citation_paths) and len(
        set(raw_citation_paths)
    ) != len(raw_citation_paths):
        print("worklist corpus citation paths must be unique", file=sys.stderr)
        return None
    release = _load_bound_corpus_release(args)
    if release is None:
        return None
    try:
        return [
            _resolve_bound_source(citation_path, release)
            for citation_path in raw_citation_paths
        ]
    except (OSError, ValueError) as exc:
        print(f"invalid worklist corpus source: {exc}", file=sys.stderr)
        return None


def _load_drift_modules(args: argparse.Namespace) -> dict[str, str] | None:
    if args.modules_file:
        mapping = json.loads(_read(args.modules_file))
        return {str(k): _read(Path(v)) for k, v in mapping.items()}
    if args.root:
        try:
            root = Path(args.root).resolve(strict=True)
        except OSError as exc:
            print(f"invalid drift root {args.root}: {exc}", file=sys.stderr)
            return None
        if not is_composition_policy_repo_root(root):
            print(
                "drift root must be an exact canonical "
                f"rulespec-<country> checkout: {root}",
                file=sys.stderr,
            )
            return None
        modules: dict[str, str] = {}
        skipped = 0
        for path in sorted(root.rglob("*.yaml")):
            name = path.name
            relative_parts = path.relative_to(root).parts
            if (
                name.endswith(".test.yaml")
                or ".axiom" in relative_parts
                or len(relative_parts) < 3
                or relative_parts[1] not in _RULESPEC_ATOMIC_MODULE_ROOTS
            ):
                continue
            module = path.relative_to(root).as_posix()
            try:
                relative = validate_module_path(root, module)
                # Sampling only replayable, current, hash-bound encodes prevents
                # deterministic/manual repairs and unmanifested support YAML from
                # becoming guaranteed regeneration errors.
                citation = read_citation(root, relative)
                require_citation_derived_output_path(relative, citation)
            except UnsafeRegenerationPath as exc:
                print(f"unsafe drift candidate {module}: {exc}", file=sys.stderr)
                return None
            except NoReplayableManifestError:
                skipped += 1
                continue
            except RegenerationInputError as exc:
                print(f"invalid drift candidate {module}: {exc}", file=sys.stderr)
                return None
            modules[module] = path.read_text(encoding="utf-8")
        if not modules:
            print(f"no replayable merged modules found under {root}", file=sys.stderr)
            return None
        if skipped:
            print(
                f"skipped {skipped} non-replayable drift candidates under {root}",
                file=sys.stderr,
            )
        return modules
    print("drift-check needs --root or --modules-file", file=sys.stderr)
    return None


def _select_corpus_backed_modules(
    args: argparse.Namespace,
    modules: dict[str, str],
) -> dict[str, str] | None:
    """Uniformly sample locally regenerable modules without scanning them all."""

    from . import drift

    corpus_path = getattr(args, "corpus_path", None)
    if corpus_path is None:
        print("drift-check with --root requires --corpus-path", file=sys.stderr)
        return None
    try:
        corpus_release = load_rulespec_local_corpus_release(args.root, corpus_path)
    except (OSError, RegenerationInputError) as exc:
        print(f"invalid corpus path {corpus_path}: {exc}", file=sys.stderr)
        return None
    except ValueError as exc:
        print(f"invalid corpus release {corpus_path}: {exc}", file=sys.stderr)
        return None

    target = min(
        len(modules),
        max(drift.MIN_SAMPLE, min(drift.MAX_SAMPLE, args.k)),
    )
    candidates = list(modules)
    random.Random(args.seed).shuffle(candidates)
    selected: dict[str, str] = {}
    skipped = 0
    for module in candidates:
        try:
            relative = validate_module_path(args.root, module)
            read_resolvable_citation(
                args.root,
                relative,
                corpus_release=corpus_release,
            )
        except UnsafeRegenerationPath as exc:
            print(f"unsafe drift candidate {module}: {exc}", file=sys.stderr)
            return None
        except RegenerationInputError:
            skipped += 1
            continue
        selected[module] = modules[module]
        if len(selected) == target:
            break

    if len(selected) != target:
        print(
            f"found only {len(selected)} of {target} required locally "
            f"corpus-resolvable drift candidates under {args.root}",
            file=sys.stderr,
        )
        return None
    if skipped:
        print(
            f"skipped {skipped} sampled drift candidates whose manifest citation "
            "does not resolve in the local corpus",
            file=sys.stderr,
        )
    return selected


def _create_drift_issues(report, repo: str) -> list[str]:
    from . import drift

    github_env = _github_cli_environment()

    # Ensure the label exists so `gh issue create --label drift` doesn't fail on
    # a fresh repo. Best-effort and idempotent.
    subprocess.run(
        [
            "gh",
            "label",
            "create",
            "drift",
            "-R",
            repo,
            "--color",
            "B60205",
            "--description",
            "Golden-regeneration drift",
            "--force",
        ],
        capture_output=True,
        text=True,
        env=github_env,
        shell=False,
    )

    # File an issue for every drifted module AND every regeneration error — a
    # regeneration error is the least-visible outcome otherwise, and the drift
    # check's whole point is that failures are never silent.
    items = [(result, False) for result in report.drifted]
    items.extend((result, True) for result in report.errors)

    created: list[str] = []
    for result, regeneration_error in items:
        body_path: str | None = None
        try:
            if regeneration_error:
                body = drift.drift_error_issue_body(result)
                title = drift.drift_issue_title(result, regeneration_error=True)
            else:
                body = drift.drift_issue_body(result)
                title = drift.drift_issue_title(result)
            body = drift.enforce_github_issue_body_budget(redact_sensitive_text(body))
            title = drift.enforce_github_issue_title_budget(
                redact_sensitive_text(title)
            )
            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".md",
                delete=False,
                encoding="utf-8",
            ) as fh:
                fh.write(body)
                body_path = fh.name
            proc = subprocess.run(
                [
                    "gh",
                    "issue",
                    "create",
                    "-R",
                    repo,
                    "--title",
                    title,
                    "--label",
                    "drift",
                    "--body-file",
                    body_path,
                ],
                capture_output=True,
                text=True,
                env=github_env,
                shell=False,
            )
            if proc.returncode == 0:
                created.append(proc.stdout.strip())
            else:
                print(
                    "failed to create issue for "
                    f"{title}: {redact_sensitive_text(proc.stderr)}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001 - isolate every publication
            diagnostic = ascii(redact_sensitive_text(str(exc)))
            print(
                f"failed to create issue for {ascii(result.module)}: {diagnostic}",
                file=sys.stderr,
            )
        finally:
            if body_path is not None:
                try:
                    os.unlink(body_path)
                except OSError as exc:
                    diagnostic = ascii(redact_sensitive_text(str(exc)))
                    print(
                        "failed to remove temporary issue body for "
                        f"{ascii(result.module)}: {diagnostic}",
                        file=sys.stderr,
                    )
    return created


def _github_cli_environment() -> dict[str, str]:
    """Give GitHub CLI its auth context without exposing model credentials."""

    allowed = (
        "GH_CONFIG_DIR",
        "GH_ENTERPRISE_TOKEN",
        "GH_HOST",
        "GH_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
        "GITHUB_TOKEN",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TEMP",
        "TMP",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    )
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.setdefault("PATH", os.defpath)
    return env
