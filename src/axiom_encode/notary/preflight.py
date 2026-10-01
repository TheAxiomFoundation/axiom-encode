"""Read-only lineage diagnostics over explicit commits, never a merge check."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from .lineage import (
    POLICY_PATH,
    STORE_PREFIX,
    LineageClassification,
    StoreFile,
    classify_lineage,
    parse_path_policy,
)
from .manifest import Manifest, build_tree_manifest, fsck_clean
from .refusal import Refusal
from .registry import REGISTRY_PATH, parse_registry


def _git_blob(repo: Path, commit: str, path: str) -> bytes | Refusal:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repo),
                "cat-file",
                "blob",
                f"{commit}:{path}",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return Refusal("structural", path, "git_unavailable")
    if result.returncode != 0:
        return Refusal("structural", path, "blob_unreadable")
    return result.stdout


def _store(
    repo: Path, commit: str, manifest: Manifest
) -> dict[str, StoreFile] | Refusal:
    result = {}
    for path, mode, _ in manifest:
        if not path.startswith(STORE_PREFIX):
            continue
        raw = _git_blob(repo, commit, path)
        if isinstance(raw, Refusal):
            return raw
        result[path[len(STORE_PREFIX) :]] = StoreFile(raw, mode)
    return result


def inspect_lineage(
    repo: Path,
    *,
    base: str,
    subject: str,
    lane: str,
    epoch_sha256: str,
    notary_spki_sha256: str,
    legacy_apply_root: str,
    legacy_eval_root: str,
) -> LineageClassification | Refusal:
    """Classify committed lineage under explicit, caller-selected public pins.

    This does not reconstruct a finalized chain, check coverage/gates, authenticate
    the invoker, or implement the full trusted preflight. The inputs are not an
    authority claim. Dirty checkout files and candidate registry edits are ignored
    as trust inputs; production trust-surface refusal remains a separate stage.
    """
    # Accept only fully resolved pilot commit ids, not moving refs or rev syntax.
    for commit in (base, subject):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            return Refusal("subject-unresolvable", None, "full_commit_oid_required")
        try:
            resolved = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repo),
                    "cat-file",
                    "-t",
                    commit,
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return Refusal("subject-unresolvable", None, "git_unavailable")
        if resolved.returncode != 0 or resolved.stdout != b"commit\n":
            return Refusal("subject-unresolvable", None, "commit_required")
    clean = fsck_clean(repo)
    if isinstance(clean, Refusal):
        return clean
    if not clean:
        return Refusal("structural", None, "git_fsck_failed")
    base_manifest = build_tree_manifest(repo, base)
    if isinstance(base_manifest, Refusal):
        return base_manifest
    subject_manifest = build_tree_manifest(repo, subject)
    if isinstance(subject_manifest, Refusal):
        return subject_manifest
    base_paths = {entry[0] for entry in base_manifest}
    for path in (REGISTRY_PATH, POLICY_PATH):
        if path not in base_paths:
            return Refusal("policy-invalid", path, "missing_base_policy")
    registry_raw = _git_blob(repo, base, REGISTRY_PATH)
    if isinstance(registry_raw, Refusal):
        return registry_raw
    registry = parse_registry(
        registry_raw,
        lane=lane,
        notary_spki_sha256=notary_spki_sha256,
        legacy_apply_root=legacy_apply_root,
        legacy_eval_root=legacy_eval_root,
    )
    if isinstance(registry, Refusal):
        return registry
    policy_raw = _git_blob(repo, base, POLICY_PATH)
    if isinstance(policy_raw, Refusal):
        return policy_raw
    policy = parse_path_policy(policy_raw, lane=lane)
    if isinstance(policy, Refusal):
        return policy
    base_store = _store(repo, base, base_manifest)
    if isinstance(base_store, Refusal):
        return base_store
    subject_store = _store(repo, subject, subject_manifest)
    if isinstance(subject_store, Refusal):
        return subject_store
    return classify_lineage(
        base_store,
        subject_store,
        lane=lane,
        epoch_sha256=epoch_sha256,
        registry=registry,
        path_policy=policy,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--base", required=True, help="full base commit oid (SHA-1)")
    parser.add_argument(
        "--subject", required=True, help="full candidate commit oid (SHA-1)"
    )
    parser.add_argument("--lane", required=True)
    parser.add_argument("--epoch-sha256", required=True)
    parser.add_argument("--notary-spki-sha256", required=True)
    parser.add_argument("--legacy-apply-root", required=True)
    parser.add_argument("--legacy-eval-root", required=True)
    args = parser.parse_args(argv)
    result = inspect_lineage(
        args.repository,
        base=args.base,
        subject=args.subject,
        lane=args.lane,
        epoch_sha256=args.epoch_sha256,
        notary_spki_sha256=args.notary_spki_sha256,
        legacy_apply_root=args.legacy_apply_root,
        legacy_eval_root=args.legacy_eval_root,
    )
    output = {
        "scope": "lineage-diagnostic",
        "admission": "not-evaluated",
        "authority": "caller-supplied-base-and-pins",
        "lane": args.lane,
        "base_commit_git_oid": args.base,
        "subject_commit_git_oid": args.subject,
        "epoch_sha256": args.epoch_sha256,
    }
    if isinstance(result, Refusal):
        output["refusal"] = asdict(result)
    else:
        output["eligible_records"] = [record.body_sha256 for record in result.eligible]
        output["ineligible_records"] = [asdict(record) for record in result.ineligible]
    print(json.dumps(output, sort_keys=True, ensure_ascii=True))
    # Zero means diagnostic execution succeeded, never that the candidate passed.
    # Ineligible new records may coexist with full coverage in the future verifier.
    return 1 if isinstance(result, Refusal) else 0


if __name__ == "__main__":
    raise SystemExit(main())
