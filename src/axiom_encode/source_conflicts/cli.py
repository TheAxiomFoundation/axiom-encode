"""``axiom-encode compare-sources``: the source-conflict compare step."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .compare import CompareError, compare, render_record
from .engine import Engine, EngineError, bind_engine, pinned_engine_commit
from .gates import LinkGateError
from .link import LinkError, load_link
from .question import compare_question, load_question

EXIT_RAN = 0
EXIT_LINK = 1
EXIT_ENGINE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom-encode compare-sources",
        description=(
            "Run two linked RuleSpec modules on generated households and write a "
            "conflict record with the households that get different results. "
            "Exit 0 when the comparison ran (conflict or not), 1 when the link is "
            "invalid or fails a gate, 3 when the engine cannot be bound or fails."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--link", type=Path, help="Link file: two modules (.json, .yaml)"
    )
    source.add_argument(
        "--question",
        type=Path,
        help="Question file: two or more answers; every pair is compared",
    )
    parser.add_argument(
        "--rulespec-root",
        type=Path,
        required=True,
        help="RuleSpec checkout root; the link's module paths are relative to it",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "With --link: the conflict record to write (JSON). With --question: "
            "a directory for summary.json and one record per pair"
        ),
    )
    parser.add_argument(
        "--axiom-rules-engine-path",
        type=Path,
        required=True,
        help="Exact axiom-rules-engine checkout (no sibling discovery)",
    )
    parser.add_argument(
        "--axiom-rules-engine-ref",
        help=(
            "Full engine commit SHA to bind; default: the commit the RuleSpec "
            "checkout pins in .axiom/workflow-toolchain.toml"
        ),
    )
    return parser


def run(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(list(argv))
    try:
        link = load_link(args.link) if args.link else None
        question = load_question(args.question) if args.question else None
    except LinkError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_LINK
    if question is not None and (
        args.out.is_symlink() or (args.out.exists() and not args.out.is_dir())
    ):
        print(
            f"error: with --question, --out is a directory; {args.out} is a file "
            "or a link",
            file=sys.stderr,
        )
        return EXIT_LINK
    rulespec_root = args.rulespec_root.resolve()
    if not rulespec_root.is_dir():
        print(
            f"error: --rulespec-root is not a directory: {rulespec_root}",
            file=sys.stderr,
        )
        return EXIT_LINK
    try:
        commit = args.axiom_rules_engine_ref or pinned_engine_commit(rulespec_root)[0]
        engine = Engine(
            bind_engine(args.axiom_rules_engine_path.resolve(), commit), commit
        )
        with tempfile.TemporaryDirectory(prefix="compare-sources-") as workdir:
            if question is not None:
                summary, records = compare_question(
                    question,
                    rulespec_root=rulespec_root,
                    engine=engine,
                    workdir=Path(workdir),
                )
            else:
                record = compare(
                    link,
                    rulespec_root=rulespec_root,
                    engine=engine,
                    workdir=Path(workdir),
                )
    except LinkGateError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_LINK
    except (EngineError, CompareError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ENGINE
    if question is not None:
        return _write_question(args.out, summary, records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_record(record), encoding="utf-8")
    for warning in record["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    counts = record["counts"]
    print(
        f"{record['id']}: {counts['combinations']} combinations"
        f"{' (sampled from ' + str(counts['grid']) + ')' if record['sampled'] else ''}, "
        f"{counts['removed_by_constraints']} removed by constraints, {counts['run']} run, "
        f"{counts['skipped_by_guard']} skipped by guard, {counts['differ']} differ, "
        f"{counts['witness_groups']} witness groups"
    )
    for group in record["witness_groups"]:
        shown = {
            name: value
            for name, value in group["witness"]["facts"].items()
            if value is not False and str(value) != "0"
        }
        print(
            f"  a={group['direction']['a']} b={group['direction']['b']} "
            f"({group['count']} households) simplest: {shown}"
        )
    print(f"record: {args.out} (review_state: unreviewed)")
    return EXIT_RAN


def _write_question(out: Path, summary: dict, records: dict[str, dict]) -> int:
    try:
        out.mkdir(parents=True, exist_ok=True)
        # Remove the pair records that an earlier summary in this directory
        # lists, so the directory matches the new summary. Nothing else is
        # touched, and no file is written through a link.
        stale: list[str] = []
        try:
            earlier = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            stale = [
                pair["record"]
                for pair in earlier.get("pairs", [])
                if isinstance(pair, dict) and isinstance(pair.get("record"), str)
            ]
        except (OSError, ValueError, AttributeError):
            pass
        for name in (*records, "summary.json"):
            path = out / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise OSError(f"{path} exists and is not a regular file")
        for name in stale:
            path = out / name
            if (
                Path(name).name == name
                and name.endswith(".json")
                and "~" in name
                and name not in records
                and path.is_file()
                and not path.is_symlink()
            ):
                path.unlink()
        for name, record in records.items():
            (out / name).write_text(render_record(record), encoding="utf-8")
        (out / "summary.json").write_text(render_record(summary), encoding="utf-8")
    except OSError as error:
        print(f"error: cannot write the records into {out}: {error}", file=sys.stderr)
        return EXIT_LINK
    for warning in summary["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    counts = summary["counts"]
    print(
        f"{summary['id']}: {counts['answers']} answers, {counts['pairs']} pairs, "
        f"{counts['pairs_that_differ']} pairs differ"
    )
    for pair in summary["pairs"]:
        print(
            f"  {pair['a']} ~ {pair['b']}: {pair['run']} run, {pair['differ']} differ, "
            f"{len(pair['witness_groups'])} witness groups"
        )
        for group in pair["witness_groups"]:
            conditions = ", ".join(
                f"{item['fact']} = {str(item['value']).lower()}"
                for item in group["conditions"]
            )
            print(
                f"    a={group['direction']['a']} b={group['direction']['b']} "
                f"({group['count']} households)"
                + (f" when {conditions}" if conditions else "")
            )
    print(f"records: {out} (review_state: unreviewed)")
    return EXIT_RAN
