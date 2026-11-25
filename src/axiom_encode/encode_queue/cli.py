"""``axiom-encode queue``: the command surface of the central encode queue.

Machine commands print one JSON document on stdout. Exit codes: 0 done,
1 refused (the queue is unchanged, the reason is on stderr), 2 usage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from axiom_encode.encode_queue import build as queue_build
from axiom_encode.encode_queue import model
from axiom_encode.encode_queue.model import QueueError
from axiom_encode.encode_queue.service import EncodeQueue
from axiom_encode.encode_queue.store import (
    GitHubApiError,
    GitHubBranchStore,
    GitHubUnknownResult,
    LocalFileStore,
    QueueStore,
)

ENV_DIR = "AXIOM_ENCODE_QUEUE_DIR"
ENV_REPO = "AXIOM_ENCODE_QUEUE_REPO"
ENV_MACHINE = "AXIOM_ENCODE_QUEUE_MACHINE"
ENV_PERSON = "AXIOM_ENCODE_QUEUE_PERSON"
ENV_CACHE = "AXIOM_ENCODE_QUEUE_CACHE_DIR"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom-encode queue",
        description="Central encode queue: priorities, atomic claims, outcomes.",
    )
    parser.add_argument(
        "--store-dir",
        default=os.environ.get(ENV_DIR),
        help=f"Local directory backend (or ${ENV_DIR}).",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get(ENV_REPO),
        help=f"GitHub backend: OWNER/NAME of the repository (or ${ENV_REPO}).",
    )
    parser.add_argument("--branch", default="encoding-queue-state")
    parser.add_argument("--prefix", default="central-queue")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(ENV_CACHE),
        help=f"GitHub backend: directory for the blob cache (or ${ENV_CACHE}).",
    )
    commands = parser.add_subparsers(dest="queue_command", required=True)

    def person(sub: argparse.ArgumentParser, required: bool = True) -> None:
        default = os.environ.get(ENV_PERSON)
        sub.add_argument(
            "--person",
            default=default,
            required=required and default is None,
            help=f"Who acts, for example a GitHub login (or ${ENV_PERSON}).",
        )

    add = commands.add_parser("add", help="Add a package to the priority list.")
    add.add_argument("--program", required=True)
    add.add_argument("--jurisdiction", required=True, help="For example us-nc.")
    add.add_argument("--before", help="Package id to go in front of.")
    add.add_argument("--note", default="")
    person(add)

    move = commands.add_parser("move", help="Reorder a package.")
    move.add_argument("package")
    move.add_argument("--before", help="Package id to go in front of; else last.")
    person(move)

    package = commands.add_parser("package", help="Set a package status.")
    package.add_argument("package")
    package.add_argument("--status", required=True, choices=model.PACKAGE_STATUSES)
    package.add_argument("--unblock", default="", help="What unblocks it.")
    person(package)

    build = commands.add_parser(
        "build", help="Derive the units of the packages from local checkouts."
    )
    build.add_argument("--corpus", required=True, type=Path)
    build.add_argument("--rulespec", required=True, type=Path)
    build.add_argument(
        "--open-prs",
        type=Path,
        help="JSON from `gh pr list --state open --json number,files`.",
    )
    build.add_argument("--package", action="append", help="Default: all.")
    build.add_argument("--corpus-ref")
    build.add_argument("--rulespec-ref")
    person(build)

    status = commands.add_parser("status", help="Show the queue.")
    status.add_argument("--json", action="store_true")
    status.add_argument("--encoder-version")

    def owner(sub: argparse.ArgumentParser, required: bool = True) -> None:
        default = os.environ.get(ENV_MACHINE)
        sub.add_argument(
            "--machine",
            default=default,
            required=required and default is None,
            help=f"A label for this machine, not its host name (or ${ENV_MACHINE}).",
        )
        person(sub, required=required)

    claim = commands.add_parser("claim", help="Reserve the next ready units.")
    owner(claim)
    claim.add_argument(
        "--request-id",
        help="Send the same id again after no answer: the same units come back.",
    )
    claim.add_argument("--count", type=int, default=1)
    claim.add_argument("--package")
    claim.add_argument("--lease-seconds", type=int, default=model.DEFAULT_LEASE_SECONDS)
    claim.add_argument("--encoder-version")

    heartbeat = commands.add_parser("heartbeat", help="Extend leases.")
    heartbeat.add_argument(
        "--lease",
        action="append",
        required=True,
        metavar="CITATION=LEASE_ID",
    )
    heartbeat.add_argument(
        "--encoding", action="store_true", help="Mark the units as encoding."
    )
    owner(heartbeat)

    release = commands.add_parser("release", help="Give a unit back.")
    release.add_argument("citation")
    release.add_argument("--lease-id", required=True)
    owner(release)

    batch = commands.add_parser(
        "apply",
        help="Run a JSON list of claim, heartbeat, report and release "
        "operations in one commit.",
    )
    batch.add_argument("--ops", required=True, help="A JSON file, or - for stdin.")

    report = commands.add_parser("report", help="Record an outcome or a step.")
    report.add_argument("citation")
    report.add_argument(
        "--state",
        required=True,
        choices=[s for s in model.STATES if s not in (model.RESERVED, model.ENCODING)],
    )
    report.add_argument("--lease-id")
    owner(report, required=False)
    report.add_argument("--category")
    report.add_argument("--unblock")
    report.add_argument("--candidate-pr", type=int)
    report.add_argument("--signing-run", type=int)
    report.add_argument("--train-pr", type=int)
    report.add_argument("--note", default="")
    return parser


def _store(args: argparse.Namespace) -> QueueStore:
    if bool(args.store_dir) == bool(args.repo):
        raise QueueError(
            f"choose one backend: --store-dir (or ${ENV_DIR}) or --repo (or ${ENV_REPO})"
        )
    if args.store_dir:
        return LocalFileStore(Path(args.store_dir))
    return GitHubBranchStore(
        args.repo, branch=args.branch, prefix=args.prefix, cache_dir=args.cache_dir
    )


def _git_head(checkout: Path) -> str:
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _encoder_version(explicit: str | None) -> str:
    if explicit:
        return explicit
    from axiom_encode import __version__

    return __version__


def _build(queue: EncodeQueue, args: argparse.Namespace) -> dict[str, Any]:
    for label, path in (("corpus", args.corpus), ("rulespec", args.rulespec)):
        if not path.is_dir():
            raise QueueError(f"--{label} is not a directory")
    status = queue.status()
    packages = [
        p for p in status["packages"] if not args.package or p["id"] in args.package
    ]
    missing = sorted(set(args.package or []) - {p["id"] for p in packages})
    if missing:
        raise QueueError(f"packages are not in the queue: {missing}")
    if not packages:
        raise QueueError("the queue has no package; add one first")
    open_prs: dict[str, int] = {}
    if args.open_prs:
        open_prs = queue_build.open_pr_paths(_read_json(args.open_prs))
    bundles = {
        program: queue_build.load_bundle(args.corpus, program)
        for program in sorted({p["program"] for p in packages})
    }
    jurisdictions: set[str] = set()
    for entry in packages:
        for document in queue_build.in_scope_documents(
            bundles[entry["program"]], entry["jurisdiction"]
        ):
            root = document.get("citation_path") or ""
            if root:
                jurisdictions.add(root.split("/")[0])
    index = queue_build.index_corpus(args.corpus, jurisdictions)
    modules, cited = queue_build.scan_rulespec(args.rulespec)
    inputs = {
        "corpus_ref": args.corpus_ref or _git_head(args.corpus),
        "rulespec_ref": args.rulespec_ref or _git_head(args.rulespec),
        "open_prs_listed": bool(args.open_prs),
    }
    now = datetime.fromisoformat(status["at"].replace("Z", "+00:00"))
    built = {
        entry["id"]: queue_build.derive_package(
            entry["program"],
            entry["jurisdiction"],
            bundle=bundles[entry["program"]],
            corpus_index=index,
            main_modules=modules,
            main_cited=cited,
            open_prs=open_prs,
            now=now,
            inputs={
                **inputs,
                "bundle_as_of": str(bundles[entry["program"]].get("as_of", "")),
            },
        )
        for entry in packages
    }
    return queue.apply_build(built, person=args.person)


def _status_text(status: dict[str, Any]) -> str:
    lines = [f"Central encode queue at {status['at']} (revision {status['revision']})"]
    lines.append("")
    lines.append("Encoding now:")
    if not status["active"]:
        lines.append("  nothing")
    for row in status["active"]:
        owner = row["owner"]
        lines.append(
            f"  {row['state']:<9} {owner['machine']}/{owner['person']}  "
            f"{row['citation']}  (lease to {row['lease_expires_at']})"
        )
    lines.append("")
    header = (
        f"{'#':>3} {'package':<22} {'status':<8} "
        + " ".join(f"{s[:8]:>8}" for s in model.STATES)
        + f" {'on main':>8}"
    )
    lines.append(header)
    for package in status["packages"]:
        counts = package["counts"]
        lines.append(
            f"{package['rank']:>3} {package['id']:<22} {package['status']:<8} "
            + " ".join(f"{counts[s]:>8}" for s in model.STATES)
            + f" {package['landed_on_main']:>8}"
        )
        if not package["built"]:
            lines.append("      not built yet: run `axiom-encode queue build`")
        elif package["unblock"]:
            lines.append(f"      blocked: {package['unblock']}")
    return "\n".join(lines)


def _read_json(source: Path | str) -> Any:
    try:
        text = sys.stdin.read() if str(source) == "-" else Path(source).read_text()
        return json.loads(text)
    except (OSError, ValueError) as error:
        raise QueueError(f"cannot read JSON from {source}: {error}") from None


def _parse_leases(values: list[str]) -> dict[str, str]:
    leases = {}
    for value in values:
        citation, separator, lease_id = value.rpartition("=")
        if not separator or not citation or not lease_id:
            raise QueueError("--lease must be CITATION=LEASE_ID")
        leases[citation] = lease_id
    return leases


def _run(args: argparse.Namespace) -> Any:
    queue = EncodeQueue(_store(args))
    command = args.queue_command
    if command == "add":
        return queue.add_package(
            args.program,
            args.jurisdiction,
            person=args.person,
            before=args.before,
            note=args.note,
        )
    if command == "move":
        return {
            "order": queue.move_package(
                args.package, person=args.person, before=args.before
            )
        }
    if command == "package":
        return queue.set_package_status(
            args.package, args.status, person=args.person, unblock=args.unblock
        )
    if command == "build":
        return _build(queue, args)
    if command == "status":
        status = queue.status(encoder_version=args.encoder_version)
        return status if args.json else _status_text(status)
    if command == "claim":
        return {
            "claimed": queue.claim(
                machine=args.machine,
                person=args.person,
                encoder_version=_encoder_version(args.encoder_version),
                count=args.count,
                package=args.package,
                lease_seconds=args.lease_seconds,
                request_id=args.request_id,
            )
        }
    if command == "heartbeat":
        return {
            "leases": queue.heartbeat(
                _parse_leases(args.lease),
                machine=args.machine,
                person=args.person,
                encoding=args.encoding,
            )
        }
    if command == "release":
        return queue.release(
            args.citation, args.lease_id, machine=args.machine, person=args.person
        )
    if command == "apply":
        return {"results": queue.apply(_read_json(args.ops))}
    if command == "report":
        return queue.report(
            args.citation,
            args.state,
            lease_id=args.lease_id,
            machine=args.machine if args.lease_id else None,
            person=args.person,
            category=args.category,
            unblock=args.unblock,
            candidate_pr=args.candidate_pr,
            signing_run=args.signing_run,
            train_pr=args.train_pr,
            note=args.note,
        )
    raise QueueError(f"unknown queue command {command!r}")


def run_queue(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except (QueueError, GitHubApiError, GitHubUnknownResult) as error:
        print(f"queue: refused: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - one line, never a traceback
        print(
            f"queue: refused: unexpected {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=1, sort_keys=True))
    return 0
