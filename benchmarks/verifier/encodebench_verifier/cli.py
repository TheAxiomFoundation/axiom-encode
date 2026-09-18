"""Command surface: build suites, run judges, fold boards."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from axiom_encode.judges.client import DEFAULT_PROVISION_CHARS, truncate_provision

from . import DEFECT_KINDS
from .board import (
    DEFAULT_FALSE_ALARM_CEILING,
    VerifierBoardError,
    board_to_json,
    fold_verifier_board,
    render_board_markdown,
    render_board_text,
)
from .cases import CaseSuite, SuiteError
from .judges import make_runner
from .pricing import load_pricing, price_for
from .results import ResultsError, run_suite
from .sources import encodings_db, eval_suite, real, synthetic


def _eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def cmd_build_synthetic(args: argparse.Namespace) -> int:
    if bool(args.from_encodings_db) == bool(args.from_eval_suite):
        _eprint(
            "build-synthetic needs exactly one of --from-encodings-db / --from-eval-suite"
        )
        return 2
    corpus_release: Optional[str] = None
    if args.from_encodings_db:
        artifacts, identity = encodings_db.load_known_good(
            Path(args.from_encodings_db),
            generator_model=args.generator_model,
            citation_prefix=args.citation_prefix,
            max_artifact_chars=args.max_artifact_chars,
        )
        source_kind = "encodings_db"
    else:
        artifacts, identity = eval_suite.load_known_good(
            [Path(p) for p in args.from_eval_suite]
        )
        source_kind = "eval_suite"
        releases = identity.get("corpus_releases") or []
        corpus_release = releases[0] if len(releases) == 1 else None
    _eprint(f"known-good artifacts: {len(artifacts)} from {source_kind}")
    kinds = tuple(args.kinds) if args.kinds else DEFECT_KINDS
    suite, report = synthetic.build_synthetic_suite(
        artifacts,
        name=args.name,
        source_kind=source_kind,
        source_identity=identity,
        provision_chars=args.provision_chars,
        truncate=truncate_provision,
        per_kind=args.per_kind,
        seed=args.seed,
        kinds=kinds,
        corpus_release=corpus_release,
    )
    suite_path, manifest_path = suite.write(Path(args.out))
    _eprint(json.dumps(report, indent=1))
    _eprint(f"suite sha256 {suite.sha256}")
    _eprint(f"wrote {suite_path} and {manifest_path}")
    if report["short_of_quota"]:
        _eprint(f"warning: short of quota for {report['short_of_quota']}")
    return 0


def cmd_build_real(args: argparse.Namespace) -> int:
    suite = real.build_real_suite(
        Path(args.dir),
        provision_chars=args.provision_chars,
        truncate=truncate_provision,
        name=args.name,
    )
    suite_path, manifest_path = suite.write(Path(args.out))
    _eprint(json.dumps(suite.summary(), indent=1))
    for note in suite.notes:
        _eprint(f"note: {note}")
    _eprint(f"wrote {suite_path} and {manifest_path}")
    return 0


def cmd_filter_suite(args: argparse.Namespace) -> int:
    suite = CaseSuite.load(Path(args.suite))
    keep = tuple(args.keep_citation_prefix or ())
    drop = tuple(args.drop_citation_prefix or ())
    drop_pairs = set(args.drop_pair or ())
    if not keep and not drop and not drop_pairs:
        _eprint(
            "filter-suite needs --keep-citation-prefix, --drop-citation-prefix "
            "and/or --drop-pair"
        )
        return 2
    known_pairs = {case.pair_id for case in suite.cases}
    unknown = sorted(drop_pairs - known_pairs)
    if unknown:
        _eprint(f"error: --drop-pair names pairs not in the suite: {unknown}")
        return 2
    if drop_pairs and not args.reason:
        _eprint("error: --drop-pair needs --reason (recorded in the child suite)")
        return 2

    def keep_pair(case) -> bool:
        citation = case.citation
        if case.pair_id in drop_pairs:
            return False
        if keep and not citation.startswith(keep):
            return False
        return not (drop and citation.startswith(drop))

    child = suite.filtered(
        name=args.name,
        keep_pair=keep_pair,
        description={
            "keep_citation_prefix": list(keep),
            "drop_citation_prefix": list(drop),
            "drop_pairs": sorted(drop_pairs),
            "reason": args.reason,
        },
    )
    suite_path, manifest_path = child.write(Path(args.out))
    _eprint(json.dumps(child.summary(), indent=1))
    dropped = child.source_identity["derived_from"]["dropped_pairs"]
    _eprint(f"dropped {len(dropped)} pair(s): {dropped}")
    _eprint(f"wrote {suite_path} and {manifest_path}")
    return 0


def cmd_show_suite(args: argparse.Namespace) -> int:
    suite = CaseSuite.load(Path(args.suite))
    print(json.dumps(suite.summary(), indent=1))
    if args.locators:
        for index, case in enumerate(suite.cases, 1):
            if case.is_defective and case.locator:
                print(
                    f"{index:04d} {case.pair_id}: {case.locator.detail} @ {case.locator.path}"
                )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suite = CaseSuite.load(Path(args.suite))
    runner = make_runner(
        args.judge,
        name=args.name,
        provision_chars=suite.provision_chars,
        max_attempts=args.max_attempts,
        retry_seconds=args.retry_seconds,
        timeout=args.timeout,
        # JudgeClient counts attempts; the TypeSafe RetryPolicy counts retries
        # after the first attempt. Keep both families on the same budget.
        max_retries=max(0, args.max_attempts - 1),
    )
    prices = load_pricing(Path(args.pricing)) if args.pricing else load_pricing()
    price = price_for(runner.model, prices)
    if price is None:
        _eprint(f"no published price recorded for {runner.model}; cost will be blank")
    try:
        payload = run_suite(
            suite,
            runner,
            Path(args.out),
            price=price,
            workers=args.workers,
            resume=not args.fresh,
            limit=args.limit,
            progress=_eprint if not args.quiet else None,
        )
    except KeyboardInterrupt:
        _eprint(
            "interrupted: queued cases cancelled, finished rows kept; re-run to resume"
        )
        return 130
    coverage = payload["coverage"]
    _eprint(
        f"{runner.name}: scored {coverage['scored']} errors {coverage['errors']} "
        f"of {coverage['expected']} in suite; complete={coverage['complete']}"
    )
    return 0 if coverage["errors"] == 0 else 1


def cmd_board(args: argparse.Namespace) -> int:
    try:
        board = fold_verifier_board(
            [Path(p) for p in args.inputs],
            false_alarm_ceiling=args.false_alarm_ceiling,
            allow_partial=args.allow_partial,
        )
    except VerifierBoardError as exc:
        _eprint(f"error: {exc}")
        return 2
    print(render_board_text(board))
    if args.markdown_out:
        Path(args.markdown_out).write_text(render_board_markdown(board))
        _eprint(f"wrote {args.markdown_out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(board_to_json(board), indent=1))
        _eprint(f"wrote {args.json_out}")
    if args.csv_out:
        with Path(args.csv_out).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "judge",
                    "model",
                    "rank_status",
                    "kind",
                    "pairs",
                    "kind_auc",
                    "verdict_auc",
                    "paired_rise",
                    "detection_at_ceiling",
                    "native_detection",
                    "native_false_alarm",
                    "localization",
                    "errors",
                    "channel",
                ]
            )
            for stats in board.ordered_runners():
                for kind in board.kinds:
                    ks = stats.kinds[kind]
                    writer.writerow(
                        [
                            stats.runner,
                            stats.model,
                            stats.rank_status,
                            kind,
                            ks.complete_pairs,
                            ks.kind_auc,
                            ks.verdict_auc,
                            ks.kind_rise_rate,
                            ks.detection_at_ceiling,
                            ks.native_detection_rate,
                            ks.native_false_alarm_rate,
                            ks.localization_rate,
                            ks.errors,
                            ks.channel,
                        ]
                    )
        _eprint(f"wrote {args.csv_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifier",
        description="EncodeBench verifier track: benchmark fidelity judges.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-synthetic", help="plant defects in known-good artifacts")
    p.add_argument("--from-encodings-db", help="path to a read-only encodings.db")
    p.add_argument("--from-eval-suite", nargs="+", help="eval-suite output dir(s)")
    p.add_argument("--generator-model", default="gpt-5.5")
    p.add_argument("--citation-prefix", default=None, help="e.g. uk/ or us")
    p.add_argument("--max-artifact-chars", type=int, default=12_000)
    p.add_argument("--per-kind", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--kinds", nargs="*", choices=DEFECT_KINDS, default=None)
    p.add_argument("--provision-chars", type=int, default=DEFAULT_PROVISION_CHARS)
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_synthetic)

    p = sub.add_parser("build-real", help="load recorded real defect pairs")
    p.add_argument("--dir", required=True)
    p.add_argument("--provision-chars", type=int, default=DEFAULT_PROVISION_CHARS)
    p.add_argument("--name", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_build_real)

    p = sub.add_parser(
        "filter-suite",
        help="derive a child suite by citation prefix (records parent digest)",
    )
    p.add_argument("--suite", required=True)
    p.add_argument("--keep-citation-prefix", nargs="*", default=None)
    p.add_argument("--drop-citation-prefix", nargs="*", default=None)
    p.add_argument(
        "--drop-pair",
        nargs="*",
        default=None,
        help="pair ids to drop; the criterion must not depend on judge outputs",
    )
    p.add_argument("--reason", default=None, help="why pairs were dropped (recorded)")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_filter_suite)

    p = sub.add_parser("show-suite", help="summarise a suite")
    p.add_argument("suite")
    p.add_argument("--locators", action="store_true")
    p.set_defaults(func=cmd_show_suite)

    p = sub.add_parser("run", help="judge a suite with one runner")
    p.add_argument("--suite", required=True)
    p.add_argument(
        "--judge", required=True, help="referee:<model> | jev[:<model>] | replay:<file>"
    )
    p.add_argument("--name", default=None, help="runner name on the board")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge only the first N suite cases (rows already done are kept)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="rotate cases.jsonl and results.json to .bak files and start over",
    )
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--retry-seconds", type=float, default=15.0)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--pricing", default=None)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("board", help="fold results into a leaderboard")
    p.add_argument("inputs", nargs="+")
    p.add_argument(
        "--false-alarm-ceiling", type=float, default=DEFAULT_FALSE_ALARM_CEILING
    )
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--markdown-out", default=None)
    p.add_argument("--json-out", default=None)
    p.add_argument("--csv-out", default=None)
    p.set_defaults(func=cmd_board)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        SuiteError,
        real.RealDefectsError,
        eval_suite.EvalSuiteSourceError,
        ResultsError,
        ValueError,
        KeyError,
        OSError,
        sqlite3.Error,
    ) as exc:
        _eprint(f"error: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
