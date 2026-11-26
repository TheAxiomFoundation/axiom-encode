"""``axiom-encode compare-sources``: the source-conflict compare step."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .compare import CompareError, compare, render_record
from .engine import Engine, EngineError, bind_engine, pinned_engine_commit
from .gates import LinkGateError
from .link import LinkError, load_link

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
    parser.add_argument(
        "--link", type=Path, required=True, help="Link file (.json, .yaml)"
    )
    parser.add_argument(
        "--rulespec-root",
        type=Path,
        required=True,
        help="RuleSpec checkout root; the link's module paths are relative to it",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Conflict record to write (JSON)"
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
        link = load_link(args.link)
    except LinkError as error:
        print(f"error: {error}", file=sys.stderr)
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
            record = compare(
                link, rulespec_root=rulespec_root, engine=engine, workdir=Path(workdir)
            )
    except LinkGateError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_LINK
    except (EngineError, CompareError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ENGINE
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
