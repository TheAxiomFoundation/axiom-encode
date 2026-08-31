#!/usr/bin/env python3
"""Copy a verifier-owned RuleSpec worktree without consulting its Git metadata.

The targeted signing workflow deliberately treats the model-owned ``.git``
directory as hostile.  This helper copies only the bounded filesystem payload
into a fresh root-created clone, rejects indirection and special files, and
never follows a source or destination symlink.  The protected clone's own
``.git`` directory remains untouched and becomes the sole Git authority for
the publication stages that follow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

MAX_FILES = 100_000
MAX_ENTRIES = 100_000
MAX_DIRECTORY_ENTRIES = 10_000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_DEPTH = 32
MAX_PATH_BYTES = 4096


def _exact_directory(path: Path, *, label: str, owner: int) -> Path:
    if not path.is_absolute() or Path(os.path.normpath(path)) != path:
        raise SystemExit(f"{label} must be an absolute normalized path")
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise SystemExit(f"{label} must be a realpath-identical directory")
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != owner:
        raise SystemExit(f"{label} has the wrong type or owner")
    return path


def _safe_name(name: str) -> None:
    encoded = os.fsencode(name)
    if (
        not name
        or name in {".", "..", ".git"}
        or len(encoded) > 255
        or b"\\" in encoded
        or any(byte < 0x20 or byte == 0x7F for byte in encoded)
    ):
        raise SystemExit(f"unsafe RuleSpec worktree entry name: {name!r}")


class Copier:
    def __init__(self, source_owner: int, staging: Path) -> None:
        self.source_owner = source_owner
        self.staging = staging
        self.files = 0
        self.entries = 0
        self.total_bytes = 0
        self.inventory: list[dict[str, object]] = []

    def _record(
        self,
        relative: Path,
        *,
        kind: str,
        mode: int,
        size: int | None = None,
        sha256: str | None = None,
    ) -> None:
        path = relative.as_posix()
        if len(path.encode("utf-8")) > MAX_PATH_BYTES:
            raise SystemExit(f"RuleSpec worktree path is too long: {path}")
        item: dict[str, object] = {"kind": kind, "mode": mode, "path": path}
        if size is not None:
            item["size"] = size
        if sha256 is not None:
            item["sha256"] = sha256
        self.inventory.append(item)

    def copy_directory(
        self,
        source_fd: int,
        destination: Path,
        relative: Path,
        depth: int,
    ) -> None:
        if depth > MAX_DEPTH:
            raise SystemExit("RuleSpec worktree exceeds the directory depth limit")
        unordered: list[os.DirEntry[str]] = []
        with os.scandir(source_fd) as entries:
            for entry in entries:
                unordered.append(entry)
                if len(unordered) > MAX_DIRECTORY_ENTRIES:
                    raise SystemExit(
                        "RuleSpec worktree directory exceeds its entry limit"
                    )
        ordered = sorted(unordered, key=lambda entry: os.fsencode(entry.name))
        for entry in ordered:
            if not relative.parts and entry.name == ".git":
                continue
            self.entries += 1
            if self.entries > MAX_ENTRIES:
                raise SystemExit("RuleSpec worktree exceeds its total entry limit")
            _safe_name(entry.name)
            child_relative = relative / entry.name
            metadata = entry.stat(follow_symlinks=False)
            if metadata.st_uid != self.source_owner:
                raise SystemExit(
                    f"RuleSpec worktree entry has the wrong owner: {child_relative}"
                )
            child_destination = destination / entry.name
            if stat.S_ISDIR(metadata.st_mode):
                child_destination.mkdir(mode=0o700)
                child_fd = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=source_fd,
                )
                try:
                    opened = os.fstat(child_fd)
                    if (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_uid,
                        opened.st_mode,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    ) != (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_uid,
                        metadata.st_mode,
                        metadata.st_mtime_ns,
                        metadata.st_ctime_ns,
                    ):
                        raise SystemExit(
                            f"RuleSpec directory changed while opening: {child_relative}"
                        )
                    self._record(child_relative, kind="directory", mode=0o755)
                    self.copy_directory(
                        child_fd,
                        child_destination,
                        child_relative,
                        depth + 1,
                    )
                    after = os.fstat(child_fd)
                    if (
                        after.st_dev,
                        after.st_ino,
                        after.st_uid,
                        after.st_mode,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ) != (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_uid,
                        opened.st_mode,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    ):
                        raise SystemExit(
                            f"RuleSpec directory changed during copy: {child_relative}"
                        )
                finally:
                    os.close(child_fd)
                child_destination.chmod(0o755)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise SystemExit(
                    f"RuleSpec worktree contains indirection or a special file: "
                    f"{child_relative}"
                )
            if metadata.st_size > MAX_FILE_BYTES:
                raise SystemExit(
                    f"RuleSpec worktree file is too large: {child_relative}"
                )
            self.files += 1
            self.total_bytes += metadata.st_size
            if self.files > MAX_FILES or self.total_bytes > MAX_TOTAL_BYTES:
                raise SystemExit("RuleSpec worktree exceeds its bounded copy budget")
            source_file = os.open(
                entry.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=source_fd,
            )
            try:
                opened = os.fstat(source_file)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_uid,
                        opened.st_size,
                    )
                    != (
                        metadata.st_dev,
                        metadata.st_ino,
                        metadata.st_uid,
                        metadata.st_size,
                    )
                ):
                    raise SystemExit(
                        f"RuleSpec file changed while opening: {child_relative}"
                    )
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                remaining = opened.st_size
                while remaining:
                    chunk = os.read(source_file, min(1024 * 1024, remaining))
                    if not chunk:
                        raise SystemExit(
                            f"RuleSpec file truncated during copy: {child_relative}"
                        )
                    chunks.append(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
                if os.read(source_file, 1):
                    raise SystemExit(
                        f"RuleSpec file grew during copy: {child_relative}"
                    )
                after = os.fstat(source_file)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_uid,
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_uid,
                    opened.st_mode,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                ):
                    raise SystemExit(
                        f"RuleSpec file changed during copy: {child_relative}"
                    )
            finally:
                os.close(source_file)
            mode = 0o755 if metadata.st_mode & 0o111 else 0o644
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
            destination_file = os.open(child_destination, flags, mode)
            try:
                for chunk in chunks:
                    os.write(destination_file, chunk)
                os.fsync(destination_file)
            finally:
                os.close(destination_file)
            self._record(
                child_relative,
                kind="file",
                mode=mode,
                size=opened.st_size,
                sha256=digest.hexdigest(),
            )


def _remove_protected_tree(
    path: Path,
    *,
    owner: int,
    keep_git: bool = False,
) -> None:
    with os.scandir(path) as entries:
        ordered = sorted(
            entries, key=lambda entry: os.fsencode(entry.name), reverse=True
        )
    for entry in ordered:
        if keep_git and entry.name == ".git":
            continue
        metadata = entry.stat(follow_symlinks=False)
        if metadata.st_uid != owner or entry.is_symlink():
            raise SystemExit(f"protected clone contains an unsafe entry: {entry.path}")
        entry_path = Path(entry.path)
        if stat.S_ISDIR(metadata.st_mode):
            _remove_protected_tree(entry_path, owner=owner)
            entry_path.rmdir()
        elif stat.S_ISREG(metadata.st_mode):
            entry_path.unlink()
        else:
            raise SystemExit(f"protected clone contains a special file: {entry.path}")


def copy_worktree(
    source: Path,
    destination: Path,
    *,
    source_owner: int,
    protected_owner: int = 0,
) -> dict:
    source = _exact_directory(source, label="source worktree", owner=source_owner)
    destination = _exact_directory(
        destination,
        label="protected clone",
        owner=protected_owner,
    )
    for root, label, owner in (
        (source / ".git", "source Git metadata", source_owner),
        (destination / ".git", "protected Git metadata", protected_owner),
    ):
        _exact_directory(root, label=label, owner=owner)

    staging = destination.parent / f".{destination.name}.worktree-copy-{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise SystemExit("worktree staging path already exists")
    staging.mkdir(mode=0o700)
    copier = Copier(source_owner, staging)
    source_fd = os.open(
        source,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    try:
        copier.copy_directory(source_fd, staging, Path(), 0)
    except BaseException:
        _remove_protected_tree(staging, owner=protected_owner)
        staging.rmdir()
        raise
    finally:
        os.close(source_fd)

    _remove_protected_tree(destination, owner=protected_owner, keep_git=True)
    with os.scandir(staging) as entries:
        ordered = sorted(entries, key=lambda entry: os.fsencode(entry.name))
    for entry in ordered:
        os.replace(entry.path, destination / entry.name)
    staging.rmdir()
    directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    inventory = sorted(
        copier.inventory, key=lambda item: os.fsencode(str(item["path"]))
    )
    canonical = json.dumps(
        inventory,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "bytes": copier.total_bytes,
        "files": copier.files,
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "schema": "axiom-encode/untrusted-worktree-copy/v1",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-owner-uid", type=int, required=True)
    args = parser.parse_args(argv)
    if os.geteuid() != 0:
        raise SystemExit("untrusted RuleSpec worktree copy must run as root")
    payload = copy_worktree(
        args.source,
        args.destination,
        source_owner=args.source_owner_uid,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
