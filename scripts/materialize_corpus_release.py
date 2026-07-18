#!/usr/bin/env python3
"""Validate and materialize a registry-backed corpus release object."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tomllib
from pathlib import Path
from typing import Any

MAX_REGISTRY_RESPONSE_BYTES = 16 * 1024 * 1024
_RELEASE_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReleaseAcquisitionError(ValueError):
    """The release pin, registry response, or destination is unsafe."""


def load_release_pin(toolchain_path: Path) -> tuple[str, str]:
    """Read and validate the corpus release pin from a RuleSpec toolchain file."""

    try:
        document = tomllib.loads(toolchain_path.read_text(encoding="utf-8"))
        toolchain = document["toolchain"]
        release_name = toolchain["axiom_corpus_release"]
        release_sha = toolchain["axiom_corpus_release_content_sha256"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseAcquisitionError(f"invalid RuleSpec toolchain: {exc}") from exc
    if (
        not isinstance(release_name, str)
        or _RELEASE_NAME_RE.fullmatch(release_name) is None
    ):
        raise ReleaseAcquisitionError("RuleSpec corpus release name is invalid")
    if not isinstance(release_sha, str) or _SHA256_RE.fullmatch(release_sha) is None:
        raise ReleaseAcquisitionError("RuleSpec corpus release digest is invalid")
    return release_name, release_sha


def _read_bounded_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ReleaseAcquisitionError("this platform cannot reject response symlinks")
    try:
        descriptor = os.open(path, flags | nofollow)
    except OSError as exc:
        raise ReleaseAcquisitionError(
            f"could not open registry response: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseAcquisitionError("registry response is not a regular file")
        if metadata.st_size > MAX_REGISTRY_RESPONSE_BYTES:
            raise ReleaseAcquisitionError("registry response exceeds the 16 MiB limit")
        chunks: list[bytes] = []
        remaining = MAX_REGISTRY_RESPONSE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_REGISTRY_RESPONSE_BYTES:
            raise ReleaseAcquisitionError("registry response exceeds the 16 MiB limit")
        return payload
    finally:
        os.close(descriptor)


def _validate_registry_response(
    raw_response: bytes,
    *,
    expected_name: str,
    expected_sha: str,
) -> tuple[dict[str, Any], str]:
    try:
        rows = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAcquisitionError(f"registry returned invalid JSON: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ReleaseAcquisitionError(
            "registry did not return exactly one release object"
        )
    payload = rows[0].get("release_object")
    if not isinstance(payload, dict) or payload.get("release") != expected_name:
        raise ReleaseAcquisitionError("registry returned the wrong release object")
    content = payload.get("content")
    if not isinstance(content, dict):
        raise ReleaseAcquisitionError("registry returned invalid release content")
    canonical_content = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    actual_sha = hashlib.sha256(canonical_content).hexdigest()
    if payload.get("content_sha256") != actual_sha or actual_sha != expected_sha:
        raise ReleaseAcquisitionError("registry returned a content-address mismatch")
    git = content.get("git")
    release_commit = git.get("commit") if isinstance(git, dict) else None
    if (
        not isinstance(release_commit, str)
        or _COMMIT_RE.fullmatch(release_commit) is None
    ):
        raise ReleaseAcquisitionError("registry returned invalid Git provenance")
    return payload, release_commit


def _write_contained_release(
    corpus_root: Path,
    *,
    release_name: str,
    release_sha: str,
    payload: dict[str, Any],
) -> Path:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ReleaseAcquisitionError("this platform cannot enforce safe corpus paths")
    directory_flags = os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        root_descriptor = os.open(corpus_root, directory_flags)
    except OSError as exc:
        raise ReleaseAcquisitionError(
            f"corpus root is not a safe directory: {exc}"
        ) from exc
    descriptor = root_descriptor
    try:
        for segment in ("releases", release_name):
            try:
                os.mkdir(segment, mode=0o755, dir_fd=descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(segment, directory_flags, dir_fd=descriptor)
            except OSError as exc:
                raise ReleaseAcquisitionError(
                    f"corpus release destination is not a safe directory: {exc}"
                ) from exc
            if descriptor != root_descriptor:
                os.close(descriptor)
            descriptor = child

        filename = f"{release_sha}.json"
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            output = os.open(filename, file_flags, 0o600, dir_fd=descriptor)
        except OSError as exc:
            raise ReleaseAcquisitionError(
                f"could not exclusively create corpus release object: {exc}"
            ) from exc
        try:
            materialized = (
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            with os.fdopen(output, "wb", closefd=True) as stream:
                stream.write(materialized)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                os.unlink(filename, dir_fd=descriptor)
            except OSError:
                pass
            raise
    finally:
        if descriptor != root_descriptor:
            os.close(descriptor)
        os.close(root_descriptor)
    return corpus_root / "releases" / release_name / f"{release_sha}.json"


def materialize_registry_response(
    response_path: Path,
    corpus_root: Path,
    *,
    release_name: str,
    release_sha: str,
) -> tuple[Path, str]:
    """Validate one registry response and write its object inside the corpus root."""

    if _RELEASE_NAME_RE.fullmatch(release_name) is None:
        raise ReleaseAcquisitionError("corpus release name is invalid")
    if _SHA256_RE.fullmatch(release_sha) is None:
        raise ReleaseAcquisitionError("corpus release digest is invalid")
    payload, release_commit = _validate_registry_response(
        _read_bounded_regular_file(response_path),
        expected_name=release_name,
        expected_sha=release_sha,
    )
    destination = _write_contained_release(
        corpus_root,
        release_name=release_name,
        release_sha=release_sha,
        payload=payload,
    )
    return destination, release_commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pin = subparsers.add_parser("pin", help="print a validated RuleSpec release pin")
    pin.add_argument("--toolchain", type=Path, required=True)
    materialize = subparsers.add_parser(
        "materialize", help="validate and materialize a registry response"
    )
    materialize.add_argument("--toolchain", type=Path, required=True)
    materialize.add_argument("--response", type=Path, required=True)
    materialize.add_argument("--corpus-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    release_name, release_sha = load_release_pin(args.toolchain)
    if args.command == "pin":
        print(release_name)
        print(release_sha)
        return 0
    _, release_commit = materialize_registry_response(
        args.response,
        args.corpus_root,
        release_name=release_name,
        release_sha=release_sha,
    )
    print(release_commit)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseAcquisitionError as exc:
        raise SystemExit(f"Corpus release acquisition error: {exc}") from exc
