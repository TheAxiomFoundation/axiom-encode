"""Read the review records of a promotion from its candidate pull request.

Runs in the signing job of ``targeted-signed-reencode.yml`` before the
supervisor is provisioned, and only for a promotion whose dispatch wrapper
carries ``"require_review_record": true``. It holds no token and signs nothing.

One anonymous read of the candidate pull request gives the head commit and the
body in the same response. The body is used only when that head is the exact
``rulespec_ref`` of the dispatch. The body is read once: its SHA-256 goes into
the run summary and into the document this script writes, and the supervised
promotion reads that document, never GitHub, for its records.

It refuses (exit 1) unless every unit of the promotion has exactly one record
with the status ``passed`` whose citation, module path and SHA-256 of the
module and of the companion equal the files committed at the checked-out head.
The supervised ``promote-reviewed-candidate --require-review-record`` repeats
the check on the bytes it signs.

A body can be edited by the pull request author and by every account with
write access. The record is not signed: see docs/reviewed-pr-promotion.md,
"Review record", for what this check does and does not prove.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMOTION_SCHEMA = "axiom-encode/reviewed-pr-promotion/v1"
RECORDS_NAME = "candidate-review-records.json"
BODY_NAME = "candidate-pull-request-body.md"
SUMMARY_NAME = "candidate-review-records.md"


def _committed(repo: Path, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"HEAD:{path}"],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )


def review_record_required(dispatch: dict) -> bool:
    """Whether this dispatch requires review records: the one parsed value.

    True only for a promotion wrapper that carries ``"require_review_record":
    true``, as ``split_atomic_source_input`` reads it. Every other dispatch,
    also one whose source input cannot be parsed, is False here; the workflow
    refuses a malformed input in its own later step, as before. The workflow
    gates the reader step on this value and refuses the promotion when its
    own parse of the wrapper disagrees with it.
    """

    raw = dispatch.get("source_bundle_json", "[]")
    try:
        outer = json.loads(raw)
    except (TypeError, ValueError, RecursionError):
        return False
    if not isinstance(outer, dict) or outer.get("schema") != PROMOTION_SCHEMA:
        return False
    # Imported only for a promotion wrapper: every other run decides with
    # the standard library alone.
    from axiom_encode.prepare_signed_backfill import split_atomic_source_input

    try:
        transaction = split_atomic_source_input(raw)
    except ValueError:
        return False
    return transaction.get("promotion_review_record_required") is True


def read_records(
    *,
    dispatch: dict,
    repo: Path,
    out_dir: Path,
    run_id: str,
    run_attempt: int,
    fetch: Callable[[str], object],
) -> tuple[bool, str]:
    """Return (required, summary). Raises ``ValueError`` on a refusal."""

    from axiom_encode import __version__, candidate_review_record
    from axiom_encode.prepare_signed_backfill import (
        citation_rulespec_path,
        split_atomic_source_input,
    )

    if not review_record_required(dispatch):
        return False, ""
    transaction = split_atomic_source_input(dispatch["source_bundle_json"])

    country = dispatch["country"]
    head = dispatch["rulespec_ref"]
    if not isinstance(country, str) or re.fullmatch(r"[a-z]{2}", country) is None:
        raise ValueError("the country is not a two-letter code")
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ValueError("the reviewed ref is not a full commit SHA")
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != head:
        raise ValueError("the RuleSpec checkout is not at the reviewed head")
    number = transaction["rulespec_pr"]
    citations = transaction.get("promotion_units") or [dispatch["citation"]]

    try:
        pull = fetch(f"/repos/TheAxiomFoundation/rulespec-{country}/pulls/{number}")
    except Exception as exc:  # noqa: BLE001 - every read failure refuses
        raise ValueError(
            f"the candidate pull request #{number} could not be read "
            f"({type(exc).__name__}). The read is anonymous: GitHub allows 60 "
            f"requests an hour for one address, and a run that requires review "
            f"records reads 35 + 4k times for k units ({35 + 4 * len(citations)} "
            "here). A second run from the same address in the same hour can "
            "exhaust the limit; dispatch again later"
        ) from exc
    if not isinstance(pull, dict) or pull.get("number") != number:
        raise ValueError("the candidate pull request could not be read")
    pull_head = pull.get("head")
    if not isinstance(pull_head, dict) or pull_head.get("sha") != head:
        raise ValueError(
            "the candidate pull request head is not the reviewed ref; its body "
            "is not read"
        )
    body = pull.get("body")
    body = "" if body is None else body
    if not isinstance(body, str):
        raise ValueError("the candidate pull request body is not text")

    out_dir.mkdir(mode=0o700)
    (out_dir / BODY_NAME).write_bytes(body.encode("utf-8"))
    records = candidate_review_record.extract_records(body)
    document = candidate_review_record.build_read_document(
        rulespec_pr=number,
        head_sha=head,
        body=body,
        records=records,
        run_id=run_id,
        run_attempt=run_attempt,
    )
    (out_dir / RECORDS_NAME).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    units = []
    for citation in citations:
        path = str(citation_rulespec_path(citation))
        companion = path[: -len(".yaml")] + ".test.yaml"
        units.append(
            (
                citation,
                path,
                hashlib.sha256(_committed(repo, path)).hexdigest(),
                hashlib.sha256(_committed(repo, companion)).hexdigest(),
            )
        )
    try:
        admitted = candidate_review_record.require_passed_unit_records(
            records, units, encoder_version=__version__
        )
    except candidate_review_record.ReviewRecordError as exc:
        raise ValueError(
            f"{exc} (candidate pull request #{number}, body sha256 "
            f"{document['body_sha256']})"
        ) from exc
    summary = candidate_review_record.render_read_summary(document, admitted)
    (out_dir / SUMMARY_NAME).write_text(summary, encoding="utf-8")
    return True, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["switch", "read"])
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    def summarize(text: str) -> None:
        if summary_path:
            with open(summary_path, "a", encoding="utf-8") as stream:
                stream.write(text + "\n")

    dispatch = json.loads(os.environ["DISPATCH_JSON"])
    if args.command == "switch":
        # Every run states the switch, also when it is off.
        required = isinstance(dispatch, dict) and review_record_required(dispatch)
        summarize(f"Review record required: {'yes' if required else 'no'}")
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"required={'true' if required else 'false'}\n")
        return 0

    from axiom_encode import reviewed_pr_admission

    if args.repo is None or args.out_dir is None:
        parser.error("read requires --repo and --out-dir")
    try:
        required, summary = read_records(
            dispatch=dispatch,
            repo=args.repo,
            out_dir=args.out_dir,
            run_id=os.environ["GITHUB_RUN_ID"],
            run_attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
            fetch=reviewed_pr_admission.github_json,
        )
        if not required:
            # The step runs only when the switch step said yes.
            raise ValueError(
                "the reader step ran for a dispatch that does not require "
                "review records"
            )
    except Exception as exc:  # noqa: BLE001 - every failure refuses the run
        message = f"Review record refused: {exc}"
        print(message, file=sys.stderr)
        summarize("## Review record\n\nReview record required: yes\n\n" + message)
        return 1
    summarize(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
