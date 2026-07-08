"""CLI surface for the judge stages.

Kept in the judges package (not inline in the 1.8MB ``cli.py``) so the encoder's
public command surface only needs a two-line hook: call
:func:`register_judge_subparsers` after building the subparsers, and route via
:func:`dispatch` for commands in :data:`COMMANDS`.

Commands:

* ``preclassify``       — worklist triage (bulk-encode.yml cheap pre-step).
* ``judge-fidelity``    — Stage 1 referee, plus ``--calibrate`` replay.
* ``judge-grid``        — Stage 2 grid-adequacy judge.
* ``judge-disposition`` — Stage 3 disposition referee.
* ``drift-check``       — golden-regeneration drift check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

COMMANDS = frozenset(
    {
        "preclassify",
        "judge-fidelity",
        "judge-grid",
        "judge-disposition",
        "drift-check",
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


def register_judge_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add the judge-stage subcommands to an existing subparsers object."""

    pre = subparsers.add_parser(
        "preclassify",
        help="Triage a worklist before generation (amendment/xref -> skip).",
    )
    pre.add_argument(
        "--worklist", type=Path, help="JSON list of {citation, source_text}"
    )
    pre.add_argument("--citation", help="Single-entry citation (with --source-file)")
    pre.add_argument(
        "--source-file", type=Path, help="Single-entry provision text file"
    )
    pre.add_argument(
        "--no-llm", action="store_true", help="Heuristics only, no arbiter"
    )
    _add_run_log_args(pre)
    pre.add_argument("--json", action="store_true")

    fid = subparsers.add_parser(
        "judge-fidelity",
        help="Statutory-fidelity referee (or --calibrate to replay over encodings.db).",
    )
    fid.add_argument("--provision-file", type=Path)
    fid.add_argument("--rule-file", type=Path)
    fid.add_argument("--citation")
    fid.add_argument("--rule-path")
    fid.add_argument("--calibrate", action="store_true")
    fid.add_argument("--db", type=Path, default=Path("encodings.db"))
    fid.add_argument("--n", type=int, default=30)
    fid.add_argument("--seed", type=int, default=0)
    _add_run_log_args(fid)
    fid.add_argument("--json", action="store_true")

    grid = subparsers.add_parser(
        "judge-grid", help="Grid-adequacy judge for an oracle suite."
    )
    grid.add_argument("--provision-file", type=Path, required=True)
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
    drift.add_argument("--root", type=Path, help="rulespec root to sample modules from")
    drift.add_argument(
        "--modules-file", type=Path, help="JSON map module_id -> merged yaml path"
    )
    drift.add_argument("--k", type=int, default=3)
    drift.add_argument("--seed", type=int)
    drift.add_argument(
        "--regenerate-cmd",
        help=(
            "Shell template to regenerate a module; {module} {merged} {output} "
            "placeholders. Writes regenerated YAML to {output}."
        ),
    )
    drift.add_argument(
        "--dry-run",
        action="store_true",
        help="Identity regeneration (wiring check only; never reports drift)",
    )
    drift.add_argument("--create-issues", action="store_true")
    drift.add_argument("--repo", default="TheAxiomFoundation/axiom-encode")
    drift.add_argument("--json", action="store_true")


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
    raise ValueError(f"not a judge command: {args.command}")


# -- commands -------------------------------------------------------------


def cmd_preclassify(args: argparse.Namespace) -> int:
    from . import preclassifier as pc

    entries = _load_worklist(args)
    if entries is None:
        return 2
    results = pc.classify_batch(entries, use_llm=not args.no_llm)
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
    if args.calibrate:
        from . import calibration

        report = calibration.calibrate(args.db, n=args.n, seed=args.seed)
        if args.json:
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(report.summary())
        return 0

    from . import statutory_fidelity as sf

    if not args.provision_file or not args.rule_file:
        print("judge-fidelity needs --provision-file and --rule-file", file=sys.stderr)
        return 2
    event = sf.run(
        _read(args.provision_file),
        _read(args.rule_file),
        citation=args.citation,
        rule_path=args.rule_path,
    )
    _emit_event(event, args.json, args)
    return 0


def cmd_judge_grid(args: argparse.Namespace) -> int:
    from . import grid_adequacy as ga

    case_grid = json.loads(_read(args.suite_file))
    event = ga.run(_read(args.provision_file), case_grid, suite_name=args.suite_name)
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

    modules = _load_drift_modules(args)
    if modules is None:
        return 2
    if args.dry_run:

        def regenerate(_module: str, merged: str) -> str:
            return merged  # identity — wiring check only

        note = "DRY RUN (identity regeneration; drift never reported)"
    elif args.regenerate_cmd:

        def regenerate(module: str, merged: str) -> str:
            return _subprocess_regenerate(args.regenerate_cmd, module, merged)

        note = f"regenerate-cmd: {args.regenerate_cmd}"
    else:
        print(
            "drift-check needs --regenerate-cmd or --dry-run",
            file=sys.stderr,
        )
        return 2

    report = drift.run_drift_check(modules, regenerate, k=args.k, seed=args.seed)
    created = []
    if args.create_issues and (report.drifted or report.errors):
        created = _create_drift_issues(report, args.repo)

    if args.json:
        out = report.to_dict()
        out["note"] = note
        out["issues_created"] = created
        print(json.dumps(out, indent=2, default=str))
    else:
        print(
            f"Drift check ({note}): {report.to_dict()['n_checked']} checked, "
            f"{len(report.drifted)} drifted, {len(report.errors)} errors"
        )
        for r in report.drifted:
            print(f"  DRIFT {r.module}: {len(r.diffs)} diffs")
        for r in report.errors:
            print(f"  ERROR {r.module}: {r.error}")
    # Drift or a regeneration error is a signal worth a non-zero exit for CI.
    return 1 if (report.drifted or report.errors) else 0


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


def _load_worklist(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    if args.worklist:
        data = json.loads(_read(args.worklist))
        if not isinstance(data, list):
            print("worklist JSON must be a list", file=sys.stderr)
            return None
        return data
    if args.citation and args.source_file:
        return [{"citation": args.citation, "source_text": _read(args.source_file)}]
    print(
        "preclassify needs --worklist or (--citation and --source-file)",
        file=sys.stderr,
    )
    return None


def _load_drift_modules(args: argparse.Namespace) -> dict[str, str] | None:
    if args.modules_file:
        mapping = json.loads(_read(args.modules_file))
        return {str(k): _read(Path(v)) for k, v in mapping.items()}
    if args.root:
        root = Path(args.root)
        modules: dict[str, str] = {}
        for path in sorted(root.rglob("*.yaml")):
            name = path.name
            if name.endswith(".test.yaml") or ".axiom" in path.parts:
                continue
            modules[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        if not modules:
            print(f"no merged modules found under {root}", file=sys.stderr)
            return None
        return modules
    print("drift-check needs --root or --modules-file", file=sys.stderr)
    return None


def _subprocess_regenerate(template: str, module: str, merged: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        merged_path = Path(tmp) / "merged.yaml"
        output_path = Path(tmp) / "regenerated.yaml"
        merged_path.write_text(merged, encoding="utf-8")
        cmd = template.format(
            module=module, merged=str(merged_path), output=str(output_path)
        )
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return output_path.read_text(encoding="utf-8")


def _create_drift_issues(report, repo: str) -> list[str]:
    from . import drift

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
    )

    # File an issue for every drifted module AND every regeneration error — a
    # regeneration error is the least-visible outcome otherwise, and the drift
    # check's whole point is that failures are never silent.
    items = [
        (r, drift.drift_issue_body(r), f"drift: {r.module}") for r in report.drifted
    ]
    items += [
        (
            r,
            f"Golden regeneration could not check `{r.module}`:\n\n> {r.error}\n",
            f"regeneration error: {r.module}",
        )
        for r in report.errors
    ]

    created: list[str] = []
    for _result, body, title in items:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
            fh.write(body)
            body_path = fh.name
        try:
            proc = subprocess.run(
                [
                    "gh",
                    "issue",
                    "create",
                    "-R",
                    repo,
                    "--title",
                    f"Golden-regeneration {title}",
                    "--label",
                    "drift",
                    "--body-file",
                    body_path,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                created.append(proc.stdout.strip())
            else:
                print(
                    f"failed to create issue for {title}: {proc.stderr}",
                    file=sys.stderr,
                )
        finally:
            os.unlink(body_path)
    return created
