#!/usr/bin/env python3
"""Fail-closed helpers for publishing targeted signed RuleSpec backfills."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

COUNTRY_PATTERN = re.compile(r"[a-z]{2}")
MANIFEST_ROOT = PurePosixPath(".axiom/encoding-manifests")


def validate_country(value: str) -> str:
    if COUNTRY_PATTERN.fullmatch(value) is None:
        raise ValueError("country must be a two-letter lowercase country code")
    return value


def branch_name(country: str, run_id: str, run_attempt: str) -> str:
    validate_country(country)
    if not run_id.isdecimal() or not run_attempt.isdecimal():
        raise ValueError("run id and attempt must be decimal integers")
    return f"axiom/signed-backfill-{country}-{run_id}-{run_attempt}"


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
            raise ValueError("renamed/copied or malformed changed paths are not publishable")
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


def authorized_changed_paths(repo: Path) -> set[PurePosixPath]:
    changed = _changed_paths(repo)
    manifests = {
        path
        for path in changed
        if path.is_relative_to(MANIFEST_ROOT) and path.suffix == ".json"
    }
    if not manifests:
        raise ValueError("no changed signed apply manifest is available to authorize publication")

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
            raise ValueError(f"changed manifest has no applied_files authorization: {relative}")
        for index, entry in enumerate(applied_files):
            if not isinstance(entry, dict):
                raise ValueError(f"{relative} applied_files[{index}] is malformed")
            authorized.add(
                _safe_relative_path(
                    entry.get("path"), label=f"{relative} applied_files[{index}].path"
                )
            )

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
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("repo", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate-country":
            print(validate_country(args.country))
        elif args.command == "branch-name":
            print(branch_name(args.country, args.run_id, args.run_attempt))
        else:
            stage_authorized_changes(args.repo)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
