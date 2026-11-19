#!/usr/bin/env python3
"""Run the independent (blind) test author on one RuleSpec module.

A SHADOW TOOL. It reports where a second model's source-derived test cases
disagree with a candidate rule. It gates nothing: it declares no validation
check, the encode loop does not call it, and it exits 0 whatever the cases
say (2 only when the tool itself cannot run). See docs/blind-test-author.md.

Example (local claude.ai subscription login, no API key):

    uv run python scripts/blind_test_author.py \\
        --rules-file <rulespec-us>/us-az/policies/.../block-13.yaml \\
        --axiom-rules-engine-path <axiom-rules-engine checkout> \\
        --corpus-path <axiom-corpus checkout> \\
        --corpus-release-public-key "$KEY" \\
        --out-dir ~/axiom-encode-local-ab/<date>-blind-test-author/block-13

The second model sees the corpus source text and the module's declared
interface only. ``--print-prompt`` writes the exact prompt and stops before
any model call. The tool stops, with exit 2 and no model call, when the
candidate does not compile, when the prompt would carry module text, or when
a name or path carries a formula number or an effective date
(``--allow-interface-leaks`` continues and records the leak in the report).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from axiom_encode import blind_test_author as bta
from axiom_encode.constants import REVIEWER_CLI_MODEL

_ENCODER_CHECKOUT = Path(__file__).resolve().parents[1]

# The real model path is a subscription CLI login. A provider key in the
# environment is refused, as in the local A/B driver, so a model call can
# never be billed to an organization API key.
_FORBIDDEN_ENV = re.compile(
    r"(_API_KEY|_AUTH_TOKEN|_OAUTH_TOKEN)$|^(OPENAI|ANTHROPIC|AZURE_OPENAI|"
    r"GEMINI|GOOGLE|OPENROUTER)_.*(KEY|TOKEN)$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rules-file", type=Path)
    parser.add_argument(
        "--policy-repo-path",
        type=Path,
        help="Jurisdiction root, e.g. <rulespec-us>/us-az. Default: derived "
        "from --rules-file.",
    )
    parser.add_argument("--axiom-rules-engine-path", type=Path)
    parser.add_argument("--axiom-rules-engine-ref")
    parser.add_argument(
        "--citation",
        help="Corpus citation path. Default: the module's "
        "module.source_verification.corpus_citation_path.",
    )
    source = parser.add_argument_group("source text (one of)")
    source.add_argument(
        "--corpus-path",
        type=Path,
        help="axiom-corpus checkout; the citation resolves through the "
        "RuleSpec checkout's pinned corpus release.",
    )
    source.add_argument("--corpus-release-public-key")
    source.add_argument(
        "--corpus-release-retired-public-key", action="append", default=[]
    )
    source.add_argument(
        "--source-text-file",
        type=Path,
        help="The exact source text, when no corpus checkout is at hand.",
    )
    parser.add_argument(
        "--source-as-of", help="YYYY-MM-DD; default from the corpus row"
    )
    parser.add_argument(
        "--backend", choices=["claude-cli", "fake"], default="claude-cli"
    )
    parser.add_argument("--model", default=REVIEWER_CLI_MODEL)
    parser.add_argument("--timeout", type=int, default=bta.DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--fake-response-file",
        type=Path,
        help="Answer returned by --backend fake (tests and dry runs).",
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="Reuse the stored answer of an earlier report instead of calling "
        "a model, to run the same blind cases against another candidate.",
    )
    parser.add_argument("--period-hint", help="YYYY-MM")
    parser.add_argument("--max-cases", type=int, default=bta.DEFAULT_MAX_CASES)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--allow-interface-leaks",
        action="store_true",
        help="Proceed when a name or path in the prompt carries a formula "
        "number or an effective date. The report records each leak. Read the "
        "refusal first; use this for a coincidence or an accepted leak.",
    )
    parser.add_argument(
        "--list-defect-cases",
        type=Path,
        metavar="DIR",
        help="Print id, fix commit, parent commit, module path and provision "
        "file of the judgeable US cases of benchmarks/verifier/real_defects_v0 "
        "(tab-separated) and stop.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Write prompt.txt and stop. No model call, no case run.",
    )
    return parser


def _policy_repo_path(rules_file: Path, explicit: Path | None) -> Path:
    from axiom_encode.repo_routing import canonical_rulespec_root_identity

    if explicit is not None:
        return explicit.resolve()
    for parent in rules_file.resolve().parents:
        if canonical_rulespec_root_identity(parent) is not None:
            return parent
    raise bta.BlindTestAuthorError(
        "--rules-file is not inside <rulespec-country>/<jurisdiction>/; pass "
        "--policy-repo-path"
    )


def _inside(path: Path, roots: tuple[Path, ...]) -> Path | None:
    resolved = path.resolve()
    return next(
        (root for root in roots if resolved.is_relative_to(root.resolve())), None
    )


def _resolve_source(
    args: argparse.Namespace, citation: str | None, checkout: Path
) -> tuple[str, str | None]:
    """Return (source text, source as-of date)."""
    if args.source_text_file is not None:
        return args.source_text_file.read_text(encoding="utf-8"), args.source_as_of
    if args.corpus_path is None or not args.corpus_release_public_key:
        raise bta.BlindTestAuthorError(
            "give --source-text-file, or --corpus-path with --corpus-release-public-key"
        )
    if not citation:
        raise bta.BlindTestAuthorError(
            "the module names no corpus_citation_path; pass --citation"
        )
    from axiom_encode import toolchain
    from axiom_encode.corpus_resolver import resolve_local_corpus_source

    # The encoder's own path: the checkout's pinned release, verified against
    # the public trust root, then one release-bound row.
    with toolchain.local_corpus_release_verification(
        args.corpus_release_public_key,
        retired_public_keys=tuple(args.corpus_release_retired_public_key),
    ):
        release = toolchain.load_rulespec_local_corpus_release(
            checkout, args.corpus_path
        )
        resolved = resolve_local_corpus_source(citation, release)
    return resolved.body, args.source_as_of or resolved.row.source_as_of


def _backend(args: argparse.Namespace) -> bta.BlindAuthorBackend:
    if args.backend == "fake":
        if args.fake_response_file is None and args.replay_report is None:
            raise bta.BlindTestAuthorError("--backend fake needs --fake-response-file")
        response = (
            args.fake_response_file.read_text(encoding="utf-8")
            if args.fake_response_file is not None
            else ""
        )
        return bta.FakeBlindAuthorBackend(response)
    forbidden = sorted(name for name in os.environ if _FORBIDDEN_ENV.search(name))
    if forbidden and args.replay_report is None and not args.print_prompt:
        raise bta.BlindTestAuthorError(
            "refusing to call a model with provider credentials in the "
            f"environment: {', '.join(forbidden)}. Use the subscription CLI login."
        )
    return bta.ClaudeCliBlindAuthorBackend(model=args.model, timeout=args.timeout)


def _run(args: argparse.Namespace) -> int:
    if args.list_defect_cases is not None:
        for row in bta.real_defect_cases(args.list_defect_cases):
            print(
                row["id"],
                row["fix_commit"],
                row["parent_commit"],
                row["module_path"],
                row["provision"],
                sep="\t",
            )
        return 0
    missing = [
        flag
        for flag, value in (
            ("--rules-file", args.rules_file),
            ("--axiom-rules-engine-path", args.axiom_rules_engine_path),
            ("--out-dir", args.out_dir),
        )
        if value is None
    ]
    if missing:
        raise bta.BlindTestAuthorError(f"missing {', '.join(missing)}")
    out_dir = args.out_dir.expanduser()
    rules_file = args.rules_file.expanduser().resolve()
    policy_repo_path = _policy_repo_path(rules_file, args.policy_repo_path)
    # Measurement results never go into this repository or the RuleSpec checkout.
    repository = _inside(out_dir, (_ENCODER_CHECKOUT, policy_repo_path.parent))
    if repository is not None:
        raise bta.BlindTestAuthorError(
            f"--out-dir {out_dir} is inside {repository}; results stay out of git"
        )
    rulespec_payload = bta.load_yaml_mapping(rules_file)
    module_block = rulespec_payload.get("module")
    verification = (
        module_block.get("source_verification")
        if isinstance(module_block, dict)
        else None
    )
    citation = args.citation or (
        verification.get("corpus_citation_path")
        if isinstance(verification, dict)
        else None
    )
    source_text, source_as_of = _resolve_source(args, citation, policy_repo_path.parent)
    companion_file = rules_file.with_name(f"{rules_file.stem}.test.yaml")
    companion_cases = None
    if companion_file.is_file():
        import yaml

        loaded = yaml.safe_load(companion_file.read_text(encoding="utf-8"))
        companion_cases = loaded if isinstance(loaded, list) else None
    authored_response = None
    if args.replay_report is not None:
        authored_response = json.loads(args.replay_report.read_text(encoding="utf-8"))[
            "raw_response"
        ]
    backend = _backend(args)
    with tempfile.TemporaryDirectory(prefix="blind-test-author-") as work_dir:
        runner = bta.EngineCaseRunner(
            rules_file=rules_file,
            policy_repo_path=policy_repo_path,
            axiom_rules_path=args.axiom_rules_engine_path.expanduser(),
            work_dir=Path(work_dir),
            axiom_rules_engine_ref=args.axiom_rules_engine_ref,
        )
        if runner.compile_error:
            # The interface of a module that does not compile is not known,
            # and its cases could not run: stop before any model call.
            raise bta.BlindTestAuthorError(
                "the candidate does not compile; no model call was made: "
                f"{runner.compile_error}"
            )
        common = dict(
            rulespec_payload=rulespec_payload,
            module=runner.module,
            source_text=source_text,
            compiled_payload=runner.compiled_payload,
            companion_cases=companion_cases,
            citation_path=citation,
            source_as_of=source_as_of,
            period_hint=args.period_hint,
            max_cases=args.max_cases,
            allow_interface_leaks=args.allow_interface_leaks,
        )
        if args.print_prompt:
            prepared = bta.prepare_blind_prompt(**common)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "prompt.txt").write_text(prepared.prompt.text, encoding="utf-8")
            for leak in prepared.interface_leaks:
                print(f"interface leak (allowed): {leak}", file=sys.stderr)
            for dropped in prepared.interface.dropped:
                print(f"declaration dropped, not a declaration word: {dropped}")
            print(f"prompt written: {out_dir / 'prompt.txt'} (no model call)")
            return 0
        report = bta.run_blind_test_author(
            backend=backend,
            runner=runner,
            authored_response=authored_response,
            **common,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
    report["rules_file"] = str(rules_file)
    summary = bta.render_text_summary(report)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    sys.stdout.write(summary)
    print(f"report: {out_dir / 'report.json'}")
    # Shadow tool: a disagreement is a finding to read, never a failure.
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as exc:  # a shadow tool reports; it never raises into a caller
        print(f"blind test author could not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
