#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")
RULESPEC_ATOMIC_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
REVIEWED_RULESPEC_REFS = frozenset(
    {
        (
            "us",
            "8645fb934cd02dbf730cf980507bbb2d07731bd1",
        ),
        (
            "ca",
            "f60f7a84c30e38c7d4961d70647eb0457e7d76c2",
        ),
    }
)


def validate_country(value: str) -> str:
    if COUNTRY_PATTERN.fullmatch(value) is None:
        raise ValueError("country must be a two-letter lowercase country code")
    return value


def branch_name(country: str, run_id: str, run_attempt: str) -> str:
    validate_country(country)
    if not run_id.isdecimal() or not run_attempt.isdecimal():
        raise ValueError("run id and attempt must be decimal integers")
    return f"axiom/signed-backfill-{country}-{run_id}-{run_attempt}"


def validate_rulespec_base(
    repo: Path,
    country: str,
    requested_ref: str,
    *,
    open_pr: bool,
) -> str:
    """Admit main ancestry or one exact independently reviewed migration head."""

    validate_country(country)
    if COMMIT_PATTERN.fullmatch(requested_ref) is None:
        raise ValueError("rulespec ref must be a full lowercase commit SHA")
    actual_ref = _git(repo, "rev-parse", "HEAD").decode().strip()
    if actual_ref != requested_ref:
        raise ValueError("rulespec checkout does not match the requested ref")
    main_ancestor = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                "HEAD",
                "refs/remotes/origin/main",
            ],
            check=False,
        ).returncode
        == 0
    )
    if main_ancestor:
        return "main"
    if (country, requested_ref) not in REVIEWED_RULESPEC_REFS:
        raise ValueError(
            "rulespec ref is neither on main nor an approved reviewed head"
        )
    if open_pr:
        raise ValueError(
            "reviewed-head runs are artifact-only and cannot open a pull request"
        )
    return "reviewed-head-artifact"


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _changed_paths(repo: Path) -> set[PurePosixPath]:
    output = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    paths: set[PurePosixPath] = set()
    fields = output.split(b"\0")
    index = 0
    while index < len(fields) and fields[index]:
        entry = fields[index]
        status = entry[:2]
        if len(entry) < 4 or status[:1] in {b"R", b"C"} or status[1:] in {b"R", b"C"}:
            raise ValueError(
                "renamed/copied or malformed changed paths are not publishable"
            )
        paths.add(PurePosixPath(entry[3:].decode("utf-8")))
        index += 1
    return paths


def _safe_relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"{label} is not a safe repository-relative path")
    return path


def _validate_rulespec_path(repo: Path, path: PurePosixPath, *, label: str) -> None:
    repo_prefix = "rulespec-"
    if not repo.name.startswith(repo_prefix):
        raise ValueError("repository directory must use the rulespec-<country> name")
    country = validate_country(repo.name.removeprefix(repo_prefix))
    if (
        len(path.parts) < 3
        or (path.parts[0] != country and not path.parts[0].startswith(f"{country}-"))
        or path.parts[1] not in RULESPEC_ATOMIC_ROOTS
        or path.suffix != ".yaml"
    ):
        raise ValueError(f"{label} is not a canonical RuleSpec YAML path")


def authorized_changed_paths(repo: Path) -> set[PurePosixPath]:
    changed = _changed_paths(repo)
    manifests = {
        path
        for path in changed
        if path.is_relative_to(MANIFEST_ROOT) and path.suffix == ".json"
    }
    if not manifests:
        raise ValueError(
            "no changed signed apply manifest is available to authorize publication"
        )

    authorized = set(manifests)
    for relative in manifests:
        manifest_path = repo / relative
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"changed manifest is not a regular file: {relative}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "axiom-encode/applied-rulespec/v5":
            raise ValueError(f"changed manifest has an unsupported schema: {relative}")
        applied_files = payload.get("applied_files")
        if not isinstance(applied_files, list) or not applied_files:
            raise ValueError(
                f"changed manifest has no applied_files authorization: {relative}"
            )
        for index, entry in enumerate(applied_files):
            if not isinstance(entry, dict):
                raise ValueError(f"{relative} applied_files[{index}] is malformed")
            label = f"{relative} applied_files[{index}].path"
            applied_path = _safe_relative_path(entry.get("path"), label=label)
            _validate_rulespec_path(repo, applied_path, label=label)
            authorized.add(applied_path)

    unexpected = changed - authorized
    missing = authorized - changed
    if unexpected:
        raise ValueError(
            "publication found changed paths outside signed manifest authorization: "
            + ", ".join(map(str, sorted(unexpected)))
        )
    if missing:
        raise ValueError(
            "signed manifest authorizes paths that are not changed: "
            + ", ".join(map(str, sorted(missing)))
        )
    return authorized


def stage_authorized_changes(repo: Path) -> None:
    authorized = authorized_changed_paths(repo)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "core.hooksPath=/dev/null",
            "add",
            "--",
            *map(str, sorted(authorized)),
        ],
        check=True,
    )
    staged = {
        PurePosixPath(value.decode("utf-8"))
        for value in _git(repo, "diff", "--cached", "--name-only", "-z").split(b"\0")
        if value
    }
    if staged != authorized:
        raise ValueError("staged paths differ from signed manifest authorization")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    country_parser = subparsers.add_parser("validate-country")
    country_parser.add_argument("country")
    branch_parser = subparsers.add_parser("branch-name")
    branch_parser.add_argument("country")
    branch_parser.add_argument("run_id")
    branch_parser.add_argument("run_attempt")
    base_parser = subparsers.add_parser("validate-rulespec-base")
    base_parser.add_argument("repo", type=Path)
    base_parser.add_argument("country")
    base_parser.add_argument("requested_ref")
    base_parser.add_argument("open_pr", choices=("true", "false"))
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate-country":
            print(validate_country(args.country))
        elif args.command == "branch-name":
            print(branch_name(args.country, args.run_id, args.run_attempt))
        elif args.command == "validate-rulespec-base":
            print(
                validate_rulespec_base(
                    args.repo,
                    args.country,
                    args.requested_ref,
                    open_pr=args.open_pr == "true",
                )
            )
        else:
            stage_authorized_changes(args.repo)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
