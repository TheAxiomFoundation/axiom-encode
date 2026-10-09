"""Receipt-bound publication helpers for the model-free metadata migration.

These helpers do not sign or invoke an encoder. The protected workflow runs
the migration through the apply signer, verifies it with guard-generated,
then uses this inventory to package and commit exactly the guarded change set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping

PLAN_SCHEMA = "axiom-encode/retired-source-metadata-migration-plan/v1"
CHANGES_SCHEMA = "axiom-encode/retired-source-metadata-migration-changes/v1"
RECEIPT_ROOT = Path(".axiom/retired-source-metadata-migrations")
MAX_JSON_BYTES = 16 * 1024 * 1024
EXCLUSIVE_INPUTS = (
    "DEPENDENT_CITATION",
    "DEPENDENT_REVIEW_FINDING",
    "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
    "QUEUE_DISPATCHER_RUN_ID",
    "QUEUE_ID",
    "QUEUE_ITEM_GENERATION_SHA256",
    "QUEUE_ITEM_ID",
    "QUEUE_MANIFEST_SHA256",
    "REPAIR_RUN_ID",
    "REPLACE_LEGACY_RULESPEC_PATH",
    "REPLACE_RULESPEC_PATH",
    "REVIEW_FINDING",
    "SECOND_DEPENDENT_CITATION",
    "SECOND_DEPENDENT_REVIEW_FINDING",
    "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
)
EXCLUSIVE_ARRAY_INPUTS = (
    "EXISTING_SIGNED_IMPORTS_JSON",
    "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
)


def _json(raw: str | bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("migration JSON exceeds its byte bound")
    return json.loads(raw, object_pairs_hook=pairs)


def resolve_request(
    source_bundle_json: str, *, base_ref: str, environment: Mapping[str, str]
) -> dict[str, object] | None:
    """Select an exact migration plan and refuse every other transaction input."""

    from axiom_encode.retired_source_metadata import load_plan_bytes

    payload = _json(source_bundle_json)
    if not isinstance(payload, dict) or payload.get("schema_version") != PLAN_SCHEMA:
        return None
    plan = load_plan_bytes(source_bundle_json.encode("utf-8"))
    if plan.base_commit != base_ref:
        raise ValueError(
            "migration plan base_commit must equal dispatched rulespec_ref"
        )
    mixed = [name for name in EXCLUSIVE_INPUTS if environment.get(name, "").strip()]
    for name in EXCLUSIVE_ARRAY_INPUTS:
        if _json(environment.get(name, "") or "[]") != []:
            mixed.append(name)
    if mixed:
        raise ValueError(
            "retired source metadata migrations cannot mix transaction inputs: "
            + ", ".join(mixed)
        )
    return payload


def _git(repo: Path, *args: str) -> bytes:
    from axiom_encode.cli import _rulespec_migration_git_environment

    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        env=_rulespec_migration_git_environment(),
    )


def _statuses(repo: Path, base_ref: str) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{40}", base_ref) is None:
        raise ValueError("migration base must be a full lowercase commit SHA")
    fields = _git(
        repo, "diff", "--name-status", "--no-renames", "-z", base_ref, "--"
    ).split(b"\0")
    result: dict[str, str] = {}
    for index in range(0, len(fields) - 1, 2):
        status, path = fields[index].decode("ascii"), fields[index + 1].decode("utf-8")
        if status not in {"A", "M"}:
            raise ValueError(f"migration may not delete or rename a file: {path}")
        result[path] = status
    for raw in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(
        b"\0"
    ):
        if raw:
            path = raw.decode("utf-8")
            if path in result:
                raise ValueError(f"ambiguous migration change: {path}")
            result[path] = "A"
    return result


def _read(repo: Path, relative: str) -> bytes:
    from axiom_encode.corpus_resolver import read_bounded_regular_file

    return read_bounded_regular_file(
        repo,
        repo / relative,
        label="receipt-bound migration publication file",
        max_bytes=MAX_JSON_BYTES,
    )


def migration_changes(repo: Path, base_ref: str, *, request: Path) -> dict[str, object]:
    """Replay the receipt and bind every diff path and postimage to one request."""

    from axiom_encode.cli import retired_source_metadata_change_set
    from axiom_encode.retired_source_metadata import load_plan_bytes

    dispatched_raw = request.read_bytes()
    dispatched = _json(dispatched_raw)
    parsed = load_plan_bytes(dispatched_raw)
    if parsed.base_commit != base_ref:
        raise ValueError("migration request is not for the dispatched base commit")
    statuses = _statuses(repo, base_ref)
    receipts = sorted(path for path in statuses if Path(path).parent == RECEIPT_ROOT)
    if len(receipts) != 1 or statuses[receipts[0]] != "A":
        raise ValueError("migration must add exactly one new receipt")
    receipt_path = receipts[0]
    receipt = _json(_read(repo, receipt_path))
    if not isinstance(receipt, dict) or receipt.get("plan") != dispatched:
        raise ValueError("migration receipt is not for the dispatched plan")
    evidence = retired_source_metadata_change_set(repo, Path(receipt_path))
    expected = set(evidence["changed_paths"])
    unexpected, missing = (
        sorted(set(statuses) - expected),
        sorted(expected - set(statuses)),
    )
    if unexpected or missing:
        raise ValueError(
            "migration change set is not exactly its receipt: "
            f"unexpected={unexpected} missing={missing}"
        )
    file_digests = {item["path"]: item["after_sha256"] for item in evidence["files"]}
    changes = []
    for path in sorted(expected):
        raw = _read(repo, path)
        digest = hashlib.sha256(raw).hexdigest()
        if path in file_digests and digest != file_digests[path]:
            raise ValueError(f"migration file is not its receipt postimage: {path}")
        if path in file_digests and statuses[path] != "M":
            raise ValueError(f"migration postimage must modify a base file: {path}")
        changes.append({"path": path, "status": statuses[path], "sha256": digest})
    return {
        "schema": CHANGES_SCHEMA,
        "base": base_ref,
        "plan_sha256": receipt["plan_sha256"],
        "modules": dispatched["modules"],
        "receipt_path": receipt_path,
        "receipt_sha256": evidence["receipt_sha256"],
        "changes": changes,
    }


def verify_inventory(
    repo: Path, *, request: Path, inventory: Path
) -> dict[str, object]:
    """Recheck packaged bytes immediately before staging or publication."""

    expected = _json(inventory.read_bytes())
    if not isinstance(expected, dict) or expected.get("schema") != CHANGES_SCHEMA:
        raise ValueError("migration inventory has an unsupported schema")
    actual = migration_changes(repo, expected["base"], request=request)
    if actual != expected:
        raise ValueError("migration changed after packaging")
    return actual


def stage_inventory(repo: Path, *, request: Path, inventory: Path) -> None:
    """Stage literal receipt paths, then prove the complete staged path set."""

    from axiom_encode.cli import _rulespec_migration_git_environment

    verified = verify_inventory(repo, request=request, inventory=inventory)
    paths = [item["path"] for item in verified["changes"]]
    subprocess.run(
        ["git", "--literal-pathspecs", "-C", str(repo), "add", "--", *paths],
        check=True,
        env=_rulespec_migration_git_environment(),
    )
    staged = _git(
        repo,
        "diff",
        "--cached",
        "--name-status",
        "--no-renames",
        "-z",
        verified["base"],
    )
    expected = b"".join(
        f"{item['status']}\0{item['path']}\0".encode("utf-8")
        for item in verified["changes"]
    )
    if staged != expected:
        raise ValueError("staged migration is not exactly the packaged inventory")
    if _git(repo, "diff", "--name-only", "--no-renames", "-z"):
        raise ValueError("migration has unstaged changes")
    if _git(repo, "ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("migration has untracked changes")


def verify_committed_inventory(repo: Path, *, request: Path, inventory: Path) -> None:
    """Require the committed tree itself to have every packaged path and digest."""

    verified = verify_inventory(repo, request=request, inventory=inventory)
    if _git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("committed migration checkout is not clean")
    committed = _git(
        repo, "diff", "--name-status", "--no-renames", "-z", verified["base"], "HEAD"
    )
    expected = b"".join(
        f"{item['status']}\0{item['path']}\0".encode("utf-8")
        for item in verified["changes"]
    )
    if committed != expected:
        raise ValueError("committed migration differs from packaged inventory")
    for item in verified["changes"]:
        blob = _git(repo, "show", f"HEAD:{item['path']}")
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"committed migration digest differs: {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    resolve = commands.add_parser("resolve-request")
    resolve.add_argument("source_bundle_json")
    resolve.add_argument("--base-ref", required=True)
    for command in ("changes", "verify", "stage", "verify-committed"):
        child = commands.add_parser(command)
        child.add_argument("--repo", type=Path, required=True)
        child.add_argument("--request", type=Path, required=True)
        if command == "changes":
            child.add_argument("--base-ref", required=True)
        else:
            child.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "resolve-request":
            payload = resolve_request(
                args.source_bundle_json, base_ref=args.base_ref, environment=os.environ
            )
            if payload is not None:
                print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
        elif args.command == "changes":
            print(
                json.dumps(
                    migration_changes(args.repo, args.base_ref, request=args.request),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            operation = {
                "verify": verify_inventory,
                "stage": stage_inventory,
                "verify-committed": verify_committed_inventory,
            }[args.command]
            operation(args.repo, request=args.request, inventory=args.inventory)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"retired source metadata publication refused: {exc}") from exc


if __name__ == "__main__":
    main()
