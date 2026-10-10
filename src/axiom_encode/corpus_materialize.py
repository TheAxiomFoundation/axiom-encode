"""Place a verified corpus release's artifacts in a corpus checkout.

Once ``axiom-corpus`` keeps corpus bytes outside git, a checkout carries one
lock file per scope under ``.axiom/corpus-locks/`` and no longer tracks
``data/corpus/{sources,inventory,provisions,coverage}`` (axiom-corpus
``docs/corpus-storage.md``). A verified release object already names every
artifact's path, sha256 and byte count, so a consumer can put the exact bytes
back without trusting where they came from.

:func:`materialize_release_artifacts` places each missing artifact from the
first source whose bytes hash to the release's sha256 and byte count:

1. the content cache ``axiom-corpus-ingest corpus fetch`` fills
   (``$AXIOM_CORPUS_CACHE``, default ``~/.axiom/corpus-cache``);
2. the checkout's git objects: the file at the release's provenance commit,
   then the ``git_blob`` its scope's lock entry names;
3. the R2 object ``objects/sha256/<xx>/<sha256>`` in the release's bucket,
   when R2 read credentials are configured.

It places an artifact only where the checkout's own scope lock pins that path
to the release's sha256 and size. axiom-corpus treats the bytes at a protected
path as current when they match that path's lock entry, and a later
``sign-ingest-manifest --lock`` signs whatever is there; a protected path may
therefore hold only its lock's bytes or fresh extractor output. Where the
lock pins other bytes (the scope was re-ingested after the release), the
artifact is skipped, and reading that scope fails as it did before the
switch, when a checkout tracked different bytes at that path.

Bytes land at their path only after they verify, through a hard link or a
no-replace rename from a temporary file in ``data/corpus/.corpus-fetch-tmp/``
(the staging directory axiom-corpus uses), so a reader never sees partial or
unverified bytes and no temporary file ever sits inside a scope. An existing
file is never replaced, and no symlink is followed or created under the
corpus root. Only single-file artifact classes are placed: axiom-corpus
fetches a scope's ``sources/`` directory all or nothing.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import quote, urlsplit

from axiom_encode.corpus_release import (
    CorpusReleaseObjectError,
    VerifiedCorpusReleaseObject,
    VerifiedReleaseArtifact,
    verify_pinned_release_object_content,
)

LOCK_ROOT = PurePosixPath(".axiom/corpus-locks")
LOCK_SCHEMA_VERSION = "axiom-corpus/corpus-lock/v1"
# axiom-corpus stages fetched bytes here (corpus_locks.FETCH_TEMP_DIR and
# FETCH_TEMP_MARKER): the checkout's filesystem, outside every scope and
# ignored by git. It deletes marker files here that are more than a day old,
# and `corpus lock` names any marker file it meets inside a scope.
FETCH_TEMP_DIR = PurePosixPath("data/corpus/.corpus-fetch-tmp")
FETCH_TEMP_MARKER = ".corpus-fetch-"
STALE_FETCH_TEMP_SECONDS = 24 * 3600
NO_FETCH_ENV = "AXIOM_CORPUS_NO_FETCH"
CACHE_ENV = "AXIOM_CORPUS_CACHE"
DEFAULT_CACHE_ROOT = "~/.axiom/corpus-cache"
R2_CREDENTIAL_PATH = "~/.config/axiom-foundation/r2-credentials.json"
DEFAULT_R2_ACCOUNT_ID = "011fb8d44f0e4d9832265ac9f748bc6b"
ARTIFACT_CLASSES = ("provisions", "inventory", "coverage", "sources")
# One file per scope. A scope's sources/ directory is fetched all or nothing
# (code lists it to find the scope's sources), which only axiom-corpus does.
PLACEABLE_ARTIFACT_CLASSES = ("provisions", "inventory", "coverage")
MAX_LOCK_FILE_BYTES = 64 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
# axiom-corpus corpus_locks._SCOPE_COMPONENT_RE
_SCOPE_COMPONENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,255}$")
_ASCII_CONTROL = frozenset(chr(code) for code in (*range(0x20), 0x7F))
_EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
# link(2) errors that mean "this filesystem has no hard links" (axiom-corpus
# content_store._NO_HARDLINK_ERRNOS, less EXDEV, which is refused here).
_NO_HARDLINK_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK}
)
_LOCK_KEYS = frozenset(
    {"schema_version", "jurisdiction", "document_class", "version", "files"}
)
_LOCK_ENTRY_KEYS = frozenset({"path", "sha256", "size", "git_blob"})
_LOCK_ENTRY_REQUIRED_KEYS = frozenset({"path", "sha256", "size"})

_REMOTE_FETCH_DISABLED: ContextVar[bool] = ContextVar(
    "axiom_corpus_remote_fetch_disabled", default=False
)


class CorpusMaterializationError(ValueError):
    """A release artifact's path cannot safely receive its verified bytes."""


class SourceError(Exception):
    """One source could not supply one artifact's exact bytes."""


class ObjectSource(Protocol):
    """A place that may hold an artifact's bytes, addressed by its release entry."""

    label: str

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        """Stream the artifact's bytes, return ``None`` if absent, or raise."""


SourceFactory = Callable[[], AbstractContextManager[Sequence[ObjectSource]]]


def corpus_uses_lock_files(root: Path) -> bool:
    """True when a corpus checkout keeps its bytes in lock files, not git."""

    try:
        mode = os.lstat(Path(root) / LOCK_ROOT).st_mode
    except OSError:
        return False
    return stat.S_ISDIR(mode)


def fetch_disabled(environ: Mapping[str, str] = os.environ) -> bool:
    """True when ``AXIOM_CORPUS_NO_FETCH`` turns materialization off.

    The values axiom-corpus's ``resolver.fetch_disabled`` accepts: ``1``,
    ``true`` or ``yes``, in any case.
    """

    return environ.get(NO_FETCH_ENV, "").strip().lower() in {"1", "true", "yes"}


@contextmanager
def remote_fetch_disabled() -> Iterator[None]:
    """Use only local sources (cache and git) inside this context."""

    token = _REMOTE_FETCH_DISABLED.set(True)
    try:
        yield
    finally:
        _REMOTE_FETCH_DISABLED.reset(token)


@dataclass
class MaterializationReport:
    """What one materialization pass found, placed, left alone and could not place.

    ``skipped``: the checkout's lock does not pin the release's bytes at that
    path, so nothing is written there. ``modified``: the lock pins the
    release's bytes but the file there holds others (a local edit, or fresh
    extractor output not yet locked), so it is left untouched. ``notes``: a
    file present by size whose lock pins other bytes. None of these is a
    failure; a read of that scope fails instead, as before the switch.
    ``failed`` holds unsafe paths and artifacts the lock pins that no source
    could place.
    """

    selected: int = 0
    present: list[str] = field(default_factory=list)
    materialized: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    modified: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    bytes_materialized: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def unplaced(self) -> dict[str, str]:
        """Why a read may not find the release's bytes, by artifact path."""

        return {**self.notes, **self.skipped, **self.modified}

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.materialized.values():
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def describe_failures(self, limit: int = 10) -> str:
        return _describe(self.failed, limit)

    def describe_skipped(self, limit: int = 10) -> str:
        return _describe(self.skipped, limit)

    def describe_modified(self, limit: int = 10) -> str:
        return _describe(self.modified, limit)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "present": len(self.present),
            "materialized": len(self.materialized),
            "bytes_materialized": self.bytes_materialized,
            "sources": self.source_counts(),
            "skipped": dict(sorted(self.skipped.items())),
            "modified": dict(sorted(self.modified.items())),
            "failed": dict(sorted(self.failed.items())),
            "notes": dict(sorted(self.notes.items())),
        }


def _describe(reasons: Mapping[str, str], limit: int) -> str:
    items = sorted(reasons.items())
    lines = [f"{path}: {reason}" for path, reason in items[:limit]]
    if len(items) > limit:
        lines.append(f"... and {len(items) - limit} more")
    return "\n".join(lines)


class _DestinationDiffersError(CorpusMaterializationError):
    """A file at the artifact's path holds bytes other than the release's."""


class _LockChangedError(CorpusMaterializationError):
    """The checkout's lock stopped pinning the release's bytes mid-placement."""


def materialize_release_artifacts(
    root: Path,
    artifacts: Iterable[VerifiedReleaseArtifact],
    *,
    sources: SourceFactory,
    verify: bool = False,
    release_commit: str = "",
    note_present: bool = True,
) -> MaterializationReport:
    """Place every missing artifact the checkout's lock pins, from the first source that verifies.

    A present regular file of the listed size counts as present; ``verify``
    also hashes it. Readers hash what they read (``corpus_resolver``), so the
    size check only decides whether to fetch. Any other artifact is placed
    only when the checkout's own scope lock lists its path with the release's
    sha256 and size; otherwise it is skipped (``report.skipped``) and its path
    left as it is. A present file of another size or hash that the lock does
    pin to the release's bytes is left untouched (``report.modified``).
    ``release_commit`` names the release's ``git.commit`` in the reasons. A
    symlink anywhere on an artifact's path, or a path that cannot be
    inspected, is a failure. ``note_present`` also reads the lock of every
    present file and notes one the lock pins to other bytes (``report.notes``;
    not under ``verify``, which has hashed it). ``sources`` is opened only
    when an artifact is to be placed, so a fully materialized checkout starts
    no process and reads no credentials.
    """

    root = _require_root(root)
    report = MaterializationReport()
    locks = CheckoutLocks(root)
    missing: list[VerifiedReleaseArtifact] = []
    for artifact in artifacts:
        report.selected += 1
        try:
            _require_canonical_artifact_path(artifact.path)
            if artifact.artifact_class not in PLACEABLE_ARTIFACT_CLASSES:
                raise CorpusMaterializationError(
                    f"axiom-encode does not place {artifact.artifact_class} "
                    "artifacts; a scope's sources directory is fetched all or "
                    "nothing by `axiom-corpus-ingest corpus fetch`"
                )
            state, difference = _inspect_destination(root, artifact, verify=verify)
        except CorpusMaterializationError as exc:
            report.failed[artifact.path] = str(exc)
            continue
        if state == "present":
            report.present.append(artifact.path)
            note = (
                _present_note(locks, artifact, release_commit)
                if note_present and not verify
                else None
            )
            if note is not None:
                report.notes[artifact.path] = note
            continue
        unpinned = locks.release_bytes_unpinned(artifact)
        if unpinned is not None:
            report.skipped[artifact.path] = _reason(
                "not placed:", unpinned, release_commit
            )
        elif state == "different":
            report.modified[artifact.path] = _modified_reason(difference)
        else:
            missing.append(artifact)
    if not missing:
        return report
    with sources() as opened:
        for artifact in missing:
            _materialize_one(root, artifact, opened, report, locks, release_commit)
    return report


def present_file_note(
    root: Path, artifact: VerifiedReleaseArtifact, release_commit: str = ""
) -> str | None:
    """Why a file present by size may hold other bytes: its lock pins others."""

    return _present_note(CheckoutLocks(root), artifact, release_commit)


def _present_note(
    locks: CheckoutLocks, artifact: VerifiedReleaseArtifact, release_commit: str
) -> str | None:
    # Same size, but the lock pins other bytes: most likely the file holds
    # those (a re-ingest can keep a file's size), and a read will fail on them.
    if not isinstance(locks.entry(artifact.path), LockEntry):
        return None
    unpinned = locks.release_bytes_unpinned(artifact)
    if unpinned is None:
        return None
    return _reason("present by size, but", unpinned, release_commit)


def _reason(prefix: str, unpinned: str, release_commit: str) -> str:
    where = (
        f"the release's git.commit {release_commit}"
        if release_commit
        else "the release's git.commit"
    )
    return (
        f"{prefix} {unpinned}; to read this scope at this release, use a "
        f"corpus worktree at {where} (docs/corpus-bytes-outside-git.md)"
    )


def _modified_reason(difference: str) -> str:
    return (
        f"{difference}: the checkout's lock pins the release's bytes here, so "
        "the file is a local edit or extractor output not yet locked"
    )


def destination_state(
    root: Path,
    artifact: VerifiedReleaseArtifact,
    *,
    verify: bool = False,
) -> str:
    """Return ``"present"`` or ``"missing"``; raise for anything unsafe or different."""

    state, difference = _inspect_destination(root, artifact, verify=verify)
    if state == "different":
        raise _DestinationDiffersError(difference)
    return state


def _inspect_destination(
    root: Path,
    artifact: VerifiedReleaseArtifact,
    *,
    verify: bool = False,
) -> tuple[str, str]:
    """``("present" | "missing" | "different", why different)``; raise if unsafe."""

    parts = PurePosixPath(artifact.path).parts
    cursor = root
    for part in parts[:-1]:
        cursor = cursor / part
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            return "missing", ""
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot inspect {cursor}: {exc.strerror}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise CorpusMaterializationError(f"path contains a symlink: {cursor}")
        if not stat.S_ISDIR(mode):
            raise CorpusMaterializationError(
                f"path component is not a directory: {cursor}"
            )
    path = cursor / parts[-1]
    try:
        file_stat = os.lstat(path)
    except FileNotFoundError:
        return "missing", ""
    except OSError as exc:
        raise CorpusMaterializationError(
            f"cannot inspect {path}: {exc.strerror}"
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise CorpusMaterializationError(f"artifact is a symlink: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise CorpusMaterializationError(f"artifact is not a regular file: {path}")
    if file_stat.st_size != artifact.byte_count:
        return "different", (
            f"existing file holds {file_stat.st_size} bytes but the release lists "
            f"{artifact.byte_count}; left untouched"
        )
    if verify:
        try:
            digest = _sha256_path(path)
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot read {path}: {exc.strerror}"
            ) from exc
        if digest != artifact.sha256:
            return "different", (
                "existing file's sha256 differs from the release; left untouched"
            )
    return "present", ""


def _materialize_one(
    root: Path,
    artifact: VerifiedReleaseArtifact,
    sources: Sequence[ObjectSource],
    report: MaterializationReport,
    locks: CheckoutLocks,
    release_commit: str,
) -> None:
    reasons: list[str] = []
    created: list[tuple[int, str]] = []
    descriptors: list[int] = []
    settled = False
    relative_parent = PurePosixPath(artifact.path).parent
    try:
        parent = _OpenDirectory(root, relative_parent, created)
        descriptors.append(parent.fd)
        # The staging directory stays, as axiom-corpus leaves it (it is
        # ignored by git): removing it could pull it from under another
        # process's placement.
        staging_fd = _open_directories(root, FETCH_TEMP_DIR, [])
        descriptors.append(staging_fd)
        _prune_stale_fetch_temporaries(root / FETCH_TEMP_DIR, staging_fd)
        for source in sources:
            try:
                chunks = source.open(artifact)
            except SourceError as exc:
                reasons.append(f"{source.label}: {exc}")
                continue
            if chunks is None:
                reasons.append(f"{source.label}: absent")
                continue
            try:
                placed = _place_verified(
                    root, staging_fd, parent, chunks, artifact, locks
                )
            except SourceError as exc:
                reasons.append(f"{source.label}: {exc}")
                continue
            finally:
                descriptors[0] = parent.fd
            settled = True
            if placed:
                report.materialized[artifact.path] = source.label
                report.bytes_materialized += artifact.byte_count
            else:
                # Another process placed the release's bytes first.
                report.present.append(artifact.path)
            return
        report.failed[artifact.path] = "; ".join(reasons) or "no source is configured"
    except _LockChangedError as exc:
        report.skipped[artifact.path] = _reason(
            "not placed: during placement", str(exc), release_commit
        )
    except _DestinationDiffersError as exc:
        # Another process wrote other bytes there while ours were on the way.
        report.modified[artifact.path] = _modified_reason(str(exc))
    except CorpusMaterializationError as exc:
        report.failed[artifact.path] = str(exc)
    finally:
        if not settled:
            # Nothing was placed: remove the scope directories this attempt
            # created, so a failed run leaves the scopes as it found them
            # (rmdir refuses non-empty ones).
            for parent_fd, name in reversed(created):
                with suppress(OSError):
                    os.rmdir(name, dir_fd=parent_fd)
        for parent_fd, _name in created:
            with suppress(OSError):
                os.close(parent_fd)
        for descriptor in descriptors:
            with suppress(OSError):
                os.close(descriptor)


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


_DESCRIPTOR_OPERATIONS_SUPPORTED = (
    bool(getattr(os, "O_NOFOLLOW", 0))
    and bool(getattr(os, "O_DIRECTORY", 0))
    and all(
        function in os.supports_dir_fd
        for function in (os.open, os.mkdir, os.link, os.stat, os.unlink, os.rmdir)
    )
    and os.link in os.supports_follow_symlinks
)


def _require_descriptor_operations() -> None:
    if not _DESCRIPTOR_OPERATIONS_SUPPORTED:
        raise CorpusMaterializationError(
            "this platform cannot place corpus files without following symlinks"
        )


def _open_directories(
    root: Path,
    relative: PurePosixPath,
    created: list[tuple[int, str]],
    *,
    create: bool = True,
) -> int:
    """Open ``root/relative`` a component at a time, creating missing ones.

    Every component is opened relative to its parent's descriptor with
    ``O_NOFOLLOW``, so a symlink anywhere below the root is refused and a
    component swapped for one after it was checked is never followed. Each
    directory made here is recorded as ``(parent descriptor, name)`` so a
    failed attempt can remove it; the caller closes those descriptors and the
    returned one.
    """

    _require_descriptor_operations()
    try:
        descriptor = os.open(root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise CorpusMaterializationError(
            f"corpus root is not a real directory: {root}"
        ) from exc
    cursor = root
    try:
        for part in relative.parts:
            cursor = cursor / part
            try:
                if create:
                    os.mkdir(part, 0o755, dir_fd=descriptor)
                    created.append((os.dup(descriptor), part))
            except FileExistsError:
                pass
            except OSError as exc:
                raise CorpusMaterializationError(
                    f"cannot create directory {cursor}: {exc}"
                ) from exc
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise CorpusMaterializationError(
                    f"path component is not a real directory: {cursor}"
                ) from exc
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


class _OpenDirectory:
    """An artifact's parent directory, held open; reopened if it vanishes."""

    def __init__(
        self, root: Path, relative: PurePosixPath, created: list[tuple[int, str]]
    ):
        self.root = root
        self.relative = relative
        self.created = created
        self.fd = _open_directories(root, relative, created)

    def reopen(self) -> None:
        with suppress(OSError):
            os.close(self.fd)
        self.fd = _open_directories(self.root, self.relative, self.created)


def _place_verified(
    root: Path,
    staging_fd: int,
    parent: _OpenDirectory,
    chunks: Iterator[bytes],
    artifact: VerifiedReleaseArtifact,
    locks: CheckoutLocks,
) -> bool:
    """Stream into a staging file, verify, then give it its name without replacing.

    Returns True when this call placed the file, False when another process
    had already placed the release's bytes there. The temporary file lives in
    ``data/corpus/.corpus-fetch-tmp/`` and carries the ``.corpus-fetch-``
    marker, so a process killed mid-stream leaves it where axiom-corpus looks
    for and prunes leftovers, never inside a scope. Every step goes through
    directory descriptors without following symlinks, and the placed file
    must be the temporary file's own inode, reachable from the root by its
    path. Just before publishing, the lock is checked again, in case another
    process rewrote it meanwhile.
    """

    name = PurePosixPath(artifact.path).name
    temporary = f"{FETCH_TEMP_MARKER}{name[:40]}.{os.getpid()}.{os.urandom(6).hex()}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        digest = hashlib.sha256()
        size = 0
        try:
            descriptor = os.open(temporary, flags, 0o644, dir_fd=staging_fd)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                for chunk in chunks:
                    size += len(chunk)
                    if size > artifact.byte_count:
                        raise SourceError(
                            f"supplied more than the {artifact.byte_count} bytes "
                            "the release lists"
                        )
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            written = os.fstat(descriptor)
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot write {root / FETCH_TEMP_DIR / temporary}: {exc}"
            ) from exc
        if size != artifact.byte_count:
            raise SourceError(
                f"supplied {size} bytes but the release lists {artifact.byte_count}"
            )
        if digest.hexdigest() != artifact.sha256:
            raise SourceError("sha256 does not match the release")
        changed = locks.recheck(artifact)
        if changed is not None:
            raise _LockChangedError(changed)
        for attempt in range(2):
            try:
                published = _publish_no_replace_at(
                    staging_fd,
                    temporary,
                    parent.fd,
                    name,
                    written,
                    root / artifact.path,
                )
                break
            except _ParentVanishedError:
                if attempt:
                    raise CorpusMaterializationError(
                        f"{root / parent.relative} keeps disappearing"
                    ) from None
                # Another process's failed attempt removed this empty
                # directory after it was opened here; make it again, once.
                parent.reopen()
        if not published:
            # Another process placed a file first; accept only the release's bytes.
            _verify_release_bytes_at(parent.fd, name, artifact, root / artifact.path)
            return False
        if _identity_at_path(root, PurePosixPath(artifact.path)) != _identity(written):
            # A directory on the path was moved or swapped while the file was
            # placed: take back the name this call made and fail.
            if _identity_in(parent.fd, name) == _identity(written):
                with suppress(OSError):
                    os.unlink(name, dir_fd=parent.fd)
            raise CorpusMaterializationError(
                f"{root / parent.relative} changed while the file was placed; "
                "nothing was placed"
            )
        return True
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        with suppress(OSError):
            os.unlink(temporary, dir_fd=staging_fd)


class _ParentVanishedError(Exception):
    """The destination directory was removed after it was opened."""


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    return (file_stat.st_dev, file_stat.st_ino)


def _identity_in(directory_fd: int, name: str) -> tuple[int, int] | None:
    try:
        return _identity(os.stat(name, dir_fd=directory_fd, follow_symlinks=False))
    except OSError:
        return None


def _identity_at_path(root: Path, relative: PurePosixPath) -> tuple[int, int] | None:
    """The (device, inode) at ``root/relative``, walked without following symlinks."""

    try:
        descriptor = _open_directories(root, relative.parent, [], create=False)
    except CorpusMaterializationError:
        return None
    try:
        return _identity_in(descriptor, relative.name)
    finally:
        os.close(descriptor)


def _verify_release_bytes_at(
    directory_fd: int, name: str, artifact: VerifiedReleaseArtifact, shown: Path
) -> None:
    """Accept a file someone else placed only if it holds the release's bytes."""

    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise CorpusMaterializationError(f"cannot inspect {shown}: {exc}") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise CorpusMaterializationError(f"artifact is not a regular file: {shown}")
        if file_stat.st_size != artifact.byte_count:
            raise _DestinationDiffersError(
                f"existing file holds {file_stat.st_size} bytes but the release "
                f"lists {artifact.byte_count}; left untouched"
            )
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CorpusMaterializationError(f"cannot read {shown}: {exc}") from exc
    finally:
        os.close(descriptor)
    if digest.hexdigest() != artifact.sha256:
        raise _DestinationDiffersError(
            "existing file's sha256 differs from the release; left untouched"
        )


def _publish_no_replace_at(
    source_dir_fd: int,
    source: str,
    target_dir_fd: int,
    target: str,
    written: os.stat_result,
    shown: Path,
) -> bool:
    """Give the staged file the name ``target`` unless that name exists.

    Returns False, leaving ``target`` untouched, when something already
    exists there. Uses ``link(2)`` without following symlinks, or where the
    filesystem has no hard links a no-replace rename
    (``renameatx_np(RENAME_EXCL)`` on macOS, ``renameat2(RENAME_NOREPLACE)``
    on Linux), as axiom-corpus's ``content_store.publish_no_replace`` does.
    Unlike that function it never falls back to a checked plain rename or to
    a copy beside ``target``: it raises instead, because either could replace
    a file or leave a temporary file inside a scope. The staged name must
    reach ``written``'s inode before each publication call. If the new name
    reaches another inode afterward, placement fails and leaves it untouched.
    """

    def check_staged_file() -> None:
        if _identity_in(source_dir_fd, source) != _identity(written):
            raise CorpusMaterializationError(
                f"the staged file for {shown} was replaced before it was published; "
                "nothing was placed"
            )

    check_staged_file()
    try:
        os.link(
            source,
            target,
            src_dir_fd=source_dir_fd,
            dst_dir_fd=target_dir_fd,
            follow_symlinks=False,
        )
    except FileExistsError:
        return False
    except OSError as exc:
        if exc.errno == errno.ENOENT and _identity_in(source_dir_fd, source):
            raise _ParentVanishedError() from exc
        if exc.errno == errno.EXDEV:
            raise CorpusMaterializationError(
                f"{FETCH_TEMP_DIR} and {shown.parent} are on different "
                "filesystems; place this file with `axiom-corpus-ingest corpus fetch`"
            ) from exc
        if exc.errno not in _NO_HARDLINK_ERRNOS:
            raise CorpusMaterializationError(
                f"cannot link verified bytes into place: {shown}: {exc}"
            ) from exc
        rename_noreplace = _rename_noreplace()
        if rename_noreplace is None:
            raise CorpusMaterializationError(
                f"cannot place {shown}: the filesystem has no hard links and "
                "this platform has no no-replace rename"
            ) from exc
        check_staged_file()
        error = rename_noreplace(source_dir_fd, source, target_dir_fd, target)
        if error == errno.EEXIST:
            return False
        if error == errno.ENOENT and _identity_in(source_dir_fd, source):
            raise _ParentVanishedError() from exc
        if error != 0:
            raise CorpusMaterializationError(
                f"cannot place {shown}: the filesystem has no hard links and a "
                f"no-replace rename failed: {os.strerror(error)}"
            ) from exc
    if _identity_in(target_dir_fd, target) != _identity(written):
        raise CorpusMaterializationError(
            f"{shown} changed after verified bytes were published; left untouched"
        )
    return True


@cache
def _rename_noreplace() -> Callable[[int, str, int, str], int] | None:
    """An atomic "rename unless the target exists" from the C library, if any.

    Takes (source directory fd, source name, target directory fd, target
    name) and returns 0 on success or an errno. Loaded on first use: only a
    filesystem without hard links needs it.
    """

    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None
    if sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        renameatx = libc.renameatx_np
        renameatx.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx.restype = ctypes.c_int

        def rename_excl(
            source_fd: int, source: str, target_fd: int, target: str
        ) -> int:
            result = renameatx(
                source_fd, os.fsencode(source), target_fd, os.fsencode(target), 0x4
            )
            return 0 if result == 0 else ctypes.get_errno()  # RENAME_EXCL

        return rename_excl
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int

        def rename_noreplace(
            source_fd: int, source: str, target_fd: int, target: str
        ) -> int:
            result = renameat2(
                source_fd, os.fsencode(source), target_fd, os.fsencode(target), 1
            )
            return 0 if result == 0 else ctypes.get_errno()  # RENAME_NOREPLACE

        return rename_noreplace
    return None


_PRUNED_STAGING_DIRECTORIES: set[Path] = set()
_PRUNE_LOCK = threading.Lock()


def _prune_stale_fetch_temporaries(
    staging: Path, staging_fd: int | None = None
) -> None:
    """Delete marker files an interrupted placement left more than a day ago.

    The same rule as axiom-corpus's ``_prune_stale_fetch_tmp``, once per
    process: a live placement's file is never that old. ctime counts because
    a clone keeps its source's old mtime. With ``staging_fd`` the directory is
    listed and pruned through that descriptor, never through a symlink.
    """

    with _PRUNE_LOCK:
        if staging in _PRUNED_STAGING_DIRECTORIES:
            return
        _PRUNED_STAGING_DIRECTORIES.add(staging)
    cutoff = time.time() - STALE_FETCH_TEMP_SECONDS
    try:
        descriptor = (
            os.dup(staging_fd)
            if staging_fd is not None
            else os.open(staging, _DIRECTORY_FLAGS)
        )
    except OSError:
        return
    try:
        with os.scandir(descriptor) as entries:
            for item in entries:
                if FETCH_TEMP_MARKER not in item.name:
                    continue
                with suppress(OSError):
                    item_stat = os.stat(
                        item.name, dir_fd=descriptor, follow_symlinks=False
                    )
                    if not stat.S_ISREG(item_stat.st_mode):
                        continue
                    if max(item_stat.st_mtime, item_stat.st_ctime) < cutoff:
                        os.unlink(item.name, dir_fd=descriptor)
    except OSError:
        pass
    finally:
        with suppress(OSError):
            os.close(descriptor)


def _require_root(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute():
        raise CorpusMaterializationError(f"corpus root must be absolute: {root}")
    try:
        mode = os.lstat(root).st_mode
    except OSError as exc:
        raise CorpusMaterializationError(f"corpus root does not exist: {root}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CorpusMaterializationError(f"corpus root is not a real directory: {root}")
    return root


def _require_canonical_artifact_path(path: str) -> None:
    parts = path.split("/")
    if (
        len(parts) < 6
        or parts[:2] != ["data", "corpus"]
        or parts[2] not in ARTIFACT_CLASSES
        or any(part in {"", ".", ".."} for part in parts)
        or any(character in path for character in "\\\n\r\0")
    ):
        raise CorpusMaterializationError(f"non-canonical artifact path: {path!r}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(
        path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_handle(handle: BinaryIO) -> Iterator[bytes]:
    try:
        while True:
            try:
                chunk = handle.read(_CHUNK_BYTES)
            except (OSError, http.client.HTTPException) as exc:
                raise SourceError(f"read failed: {exc}") from exc
            if not chunk:
                return
            yield chunk
    finally:
        handle.close()


# --------------------------------------------------------------------- sources


class ContentCacheSource:
    """The shared cache ``axiom-corpus-ingest corpus fetch`` writes."""

    label = "cache"

    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] = os.environ
    ) -> ContentCacheSource | None:
        try:
            root = Path(environ.get(CACHE_ENV) or DEFAULT_CACHE_ROOT).expanduser()
        except RuntimeError:  # no HOME and no passwd entry
            return None
        return cls(root) if root.is_dir() else None

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        path = self.root / "objects" / "sha256" / artifact.sha256[:2] / artifact.sha256
        try:
            # O_NONBLOCK: a FIFO in the cache must not hang the open.
            descriptor = os.open(
                path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SourceError(f"cannot read {path}: {exc.strerror}") from exc
        handle = os.fdopen(descriptor, "rb")
        file_stat = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != artifact.byte_count
        ):
            handle.close()
            raise SourceError(
                f"cache object holds {file_stat.st_size} bytes but the release lists "
                f"{artifact.byte_count}"
            )
        return _iter_handle(handle)


class GitObjectStore:
    """One ``git cat-file --batch`` process over a checkout's object database.

    ``cat-file --batch`` is one of the commands the protected verification
    supervisor's trusted git wrapper allows, so supervised validation can
    read corpus bytes from git history without network or credentials.
    """

    def __init__(self, root: Path, *, git: str | None = None):
        self.root = Path(root)
        self.git = git if git is not None else shutil.which("git")
        self._process: subprocess.Popen[bytes] | None = None
        self._unavailable: str | None = None if self.git else "git is not on PATH"
        # True while a returned stream has not read its whole object; the
        # next request then restarts the process instead of reading stale bytes.
        self._in_flight = False

    def open(self, name: str, byte_count: int) -> Iterator[bytes] | None:
        if any(character in name for character in "\n\r\0"):
            return None
        if self._unavailable is not None:
            raise SourceError(self._unavailable)
        if self._in_flight:
            self._stop()
        try:
            process = self._start()
        except OSError as exc:
            self._unavailable = f"cannot run git: {exc}"
            raise SourceError(self._unavailable) from exc
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(name.encode("utf-8") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline()
        except OSError:
            header = b""
        if not header:
            self._stop()
            self._unavailable = f"git cat-file cannot read objects in {self.root}"
            raise SourceError(self._unavailable)
        encoded = name.encode("utf-8")
        if header in (encoded + b" missing\n", encoded + b" ambiguous\n"):
            return None
        fields = header.split()
        if len(fields) != 3 or not fields[2].isdigit():
            self._stop()
            raise SourceError(f"unexpected git cat-file reply {header[:80]!r}")
        object_type, size = fields[1].decode("ascii", "replace"), int(fields[2])
        if object_type != "blob" or size != byte_count:
            # Restart rather than drain an object we will not use.
            self._stop()
            if object_type != "blob":
                raise SourceError(f"{name} names a {object_type}, not a blob")
            raise SourceError(
                f"git blob holds {size} bytes but the release lists {byte_count}"
            )
        self._in_flight = True
        return self._stream(process, size)

    def _stream(self, process: subprocess.Popen[bytes], size: int) -> Iterator[bytes]:
        assert process.stdout is not None
        complete = False
        try:
            remaining = size
            while remaining:
                try:
                    chunk = process.stdout.read(min(_CHUNK_BYTES, remaining))
                except OSError as exc:
                    raise SourceError(f"git cat-file read failed: {exc}") from exc
                if not chunk:
                    raise SourceError("git cat-file stopped mid-object")
                remaining -= len(chunk)
                yield chunk
            try:
                terminator = process.stdout.read(1)
            except OSError as exc:
                raise SourceError(f"git cat-file read failed: {exc}") from exc
            if terminator != b"\n":
                raise SourceError("git cat-file reply is not terminated")
            complete = True
            if self._process is process:
                self._in_flight = False
        finally:
            # A stream abandoned after a newer request restarted git must not
            # stop that newer process.
            if not complete and self._process is process:
                self._stop()

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            assert self.git is not None
            self._process = subprocess.Popen(
                [self.git, "-C", str(self.root), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=_git_environment(os.environ),
            )
        return self._process

    def _stop(self) -> None:
        self._in_flight = False
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            with suppress(OSError):
                if stream is not None:
                    stream.close()
        with suppress(OSError):
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)

    def close(self) -> None:
        self._stop()


def _git_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The caller's environment without ``GIT_*`` overrides, reading local objects only.

    An inherited ``GIT_DIR`` or ``GIT_OBJECT_DIRECTORY`` (from a git hook, say)
    would read another repository; ``GIT_NO_LAZY_FETCH`` keeps a partial clone
    from fetching missing blobs, so ``--offline`` and ``--no-remote`` stay
    local. The supervisor's trusted git wrapper sets the same.
    """

    clean = {
        name: value
        for name, value in environ.items()
        if not name.startswith("GIT_") and name != "SSH_ASKPASS"
    }
    clean.update(
        GIT_NO_LAZY_FETCH="1",
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_TERMINAL_PROMPT="0",
    )
    return clean


class GitReleaseTreeSource:
    """The file at the release's provenance commit, while git still tracks it."""

    label = "git"

    def __init__(self, store: GitObjectStore, commit: str):
        self.store = store
        self.commit = commit

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        if re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            return None
        return self.store.open(f"{self.commit}:{artifact.path}", artifact.byte_count)


class GitLockBlobSource:
    """The ``git_blob`` a scope's lock file records for a file moved out of git."""

    label = "git-lock"

    def __init__(self, store: GitObjectStore, root: Path):
        self.store = store
        self.locks = CheckoutLocks(root)

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        entry = self.locks.entry(artifact.path)
        if (
            not isinstance(entry, LockEntry)
            or entry.content != (artifact.sha256, artifact.byte_count)
            or entry.git_blob is None
        ):
            return None
        return self.store.open(entry.git_blob, artifact.byte_count)


def lock_path_for(artifact_path: str) -> PurePosixPath | None:
    """Lock file that lists one protected corpus path, by the path's scope."""

    parts = artifact_path.split("/")
    if len(parts) < 6 or parts[:2] != ["data", "corpus"]:
        return None
    artifact_class, jurisdiction, document_class = parts[2], parts[3], parts[4]
    suffix = {"provisions": ".jsonl", "inventory": ".json", "coverage": ".json"}
    if artifact_class == "sources" and len(parts) >= 7:
        version = parts[5]
    elif artifact_class in suffix and len(parts) == 6:
        if not parts[5].endswith(suffix[artifact_class]):
            return None
        version = parts[5].removesuffix(suffix[artifact_class])
    else:
        return None
    if not version:
        return None
    return LOCK_ROOT / jurisdiction / document_class / f"{version}.json"


@dataclass(frozen=True)
class LockEntry:
    """One path's entry in a checkout's scope lock."""

    sha256: str
    size: int
    git_blob: str | None = None

    @property
    def content(self) -> tuple[str, int]:
        return (self.sha256, self.size)


class CheckoutLocks:
    """A corpus checkout's scope lock files, each read and validated once.

    These are the worktree's locks, the ones ``axiom-corpus-ingest corpus
    status``/``fetch`` compare files against. A lock that is missing,
    indirect or malformed pins nothing, so nothing is placed in its scope.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self._scopes: dict[PurePosixPath, dict[str, LockEntry] | str] = {}
        self._digests: dict[PurePosixPath, str | None] = {}

    def entry(self, artifact_path: str) -> LockEntry | str:
        """The lock entry for ``artifact_path``, or why the checkout has none."""

        lock_path = lock_path_for(artifact_path)
        if lock_path is None:
            return "the path belongs to no scope lock"
        if lock_path not in self._scopes:
            raw = _read_lock_bytes(self.root, lock_path)
            if isinstance(raw, str):
                self._digests[lock_path] = None
                self._scopes[lock_path] = raw
            else:
                self._digests[lock_path] = hashlib.sha256(raw).hexdigest()
                self._scopes[lock_path] = _parse_lock_or_reason(raw, lock_path)
        entries = self._scopes[lock_path]
        if isinstance(entries, str):
            return entries
        entry = entries.get(artifact_path)
        if entry is None:
            return f"the checkout's lock {lock_path} does not list this path"
        return entry

    def release_bytes_unpinned(self, artifact: VerifiedReleaseArtifact) -> str | None:
        """None when the lock pins exactly the release's bytes; otherwise why not."""

        entry = self.entry(artifact.path)
        if isinstance(entry, str):
            return entry
        if entry.content != (artifact.sha256, artifact.byte_count):
            return (
                f"the checkout's lock {lock_path_for(artifact.path)} pins other "
                f"bytes at this path (sha256 {entry.sha256[:12]}, {entry.size} "
                f"bytes; the release lists sha256 {artifact.sha256[:12]}, "
                f"{artifact.byte_count} bytes)"
            )
        return None

    def recheck(self, artifact: VerifiedReleaseArtifact) -> str | None:
        """Like :meth:`release_bytes_unpinned`, after re-reading a lock whose bytes changed."""

        lock_path = lock_path_for(artifact.path)
        if lock_path is not None and lock_path in self._scopes:
            raw = _read_lock_bytes(self.root, lock_path)
            digest = None if isinstance(raw, str) else hashlib.sha256(raw).hexdigest()
            if digest is None or digest != self._digests.get(lock_path):
                del self._scopes[lock_path]
        return self.release_bytes_unpinned(artifact)


def _read_lock_bytes(root: Path, lock_path: PurePosixPath) -> bytes | str:
    """One scope lock's bytes, read without following a symlink, or why not."""

    cursor, mode = root, 0
    for part in lock_path.parts:
        cursor = cursor / part
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            scope = "/".join((*lock_path.parts[-3:-1], lock_path.stem))
            return f"the checkout has no lock for scope {scope} ({lock_path})"
        except OSError as exc:
            return f"the checkout's lock {lock_path} cannot be read: {exc.strerror}"
        if stat.S_ISLNK(mode):
            return f"the checkout's lock path {cursor} is a symlink"
    if not stat.S_ISREG(mode):
        return f"the checkout's lock {lock_path} is not a regular file"
    try:
        descriptor = os.open(cursor, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read(MAX_LOCK_FILE_BYTES + 1)
    except OSError as exc:
        return f"the checkout's lock {lock_path} cannot be read: {exc.strerror}"
    if len(raw) > MAX_LOCK_FILE_BYTES:
        return f"the checkout's lock {lock_path} exceeds {MAX_LOCK_FILE_BYTES} bytes"
    return raw


def _read_scope_lock(
    root: Path, lock_path: PurePosixPath
) -> dict[str, LockEntry] | str:
    """``path -> entry`` from one scope lock, or why it pins nothing."""

    raw = _read_lock_bytes(root, lock_path)
    if isinstance(raw, str):
        return raw
    return _parse_lock_or_reason(raw, lock_path)


def _parse_lock_or_reason(
    raw: bytes, lock_path: PurePosixPath
) -> dict[str, LockEntry] | str:
    """Apply axiom-corpus's ``parse_lock`` checks: schema v1, this scope,
    canonical repository paths inside it, sha256, size, an optional 40-hex
    ``git_blob``, and the canonical encoding."""

    try:
        return _parse_scope_lock(raw, lock_path)
    except (ValueError, RecursionError) as exc:
        return f"the checkout's lock {lock_path} is not a valid lock: {exc}"


def _is_canonical_repository_path(path: str) -> bool:
    """axiom-corpus corpus_locks._validate_corpus_path, as a predicate."""

    if not path:
        return False
    if path.isascii():
        bad_characters = any(character in _ASCII_CONTROL for character in path)
    else:
        bad_characters = unicodedata.normalize("NFC", path) != path or any(
            unicodedata.category(character).startswith("C") for character in path
        )
    return not (
        path.startswith("/")
        or "\\" in path
        or bad_characters
        or any(part in {"", ".", ".."} for part in path.split("/"))
    )


def _parse_scope_lock(raw: bytes, lock_path: PurePosixPath) -> dict[str, LockEntry]:
    scope = (*lock_path.parts[-3:-1], lock_path.stem)
    if not all(_SCOPE_COMPONENT_RE.fullmatch(part) for part in scope):
        raise ValueError(f"invalid scope component in {lock_path}")
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict) or set(payload) != _LOCK_KEYS:
        raise ValueError(f"lock keys must be {sorted(_LOCK_KEYS)}")
    if payload["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError(f"unsupported lock schema {payload['schema_version']!r}")
    if (
        payload["jurisdiction"],
        payload["document_class"],
        payload["version"],
    ) != scope:
        raise ValueError("the lock's scope does not match its path")
    files = payload["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("lock files must be a non-empty list")
    entries: dict[str, LockEntry] = {}
    for item in files:
        if not isinstance(item, dict) or not (
            _LOCK_ENTRY_REQUIRED_KEYS <= set(item) <= _LOCK_ENTRY_KEYS
        ):
            raise ValueError("each lock entry needs path, sha256, size [, git_blob]")
        path, sha256, size = item["path"], item["sha256"], item["size"]
        git_blob = item.get("git_blob")
        if not isinstance(path, str) or not _is_canonical_repository_path(path):
            raise ValueError(f"lock entry path is not a canonical path: {path!r}")
        if lock_path_for(path) != lock_path:
            raise ValueError(f"lock entry path is outside the scope: {path!r}")
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise ValueError(f"lock entry sha256 is not a sha256: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"lock entry size is not a byte count: {path}")
        if "git_blob" in item and (
            not isinstance(git_blob, str) or _GIT_BLOB_RE.fullmatch(git_blob) is None
        ):
            raise ValueError(f"lock entry git_blob is not an object id: {path}")
        if path in entries:
            raise ValueError(f"lock lists a path twice: {path}")
        entries[path] = LockEntry(sha256, size, git_blob)
    if _canonical_lock_bytes(scope, entries) != raw:
        # Unsorted entries, another layout, duplicate JSON keys: axiom-corpus
        # rejects every lock that is not in its own serialize_lock form.
        raise ValueError("lock bytes are not in canonical form")
    return entries


def _canonical_lock_bytes(
    scope: tuple[str, ...], entries: Mapping[str, LockEntry]
) -> bytes:
    """A lock as axiom-corpus's ``serialize_lock`` writes it."""

    header = [
        f'  "schema_version": {json.dumps(LOCK_SCHEMA_VERSION)},',
        f'  "jurisdiction": {json.dumps(scope[0])},',
        f'  "document_class": {json.dumps(scope[1])},',
        f'  "version": {json.dumps(scope[2])},',
        '  "files": [',
    ]
    lines = []
    for path in sorted(entries):
        entry = entries[path]
        mapping: dict[str, object] = {
            "path": path,
            "sha256": entry.sha256,
            "size": entry.size,
        }
        if entry.git_blob is not None:
            mapping["git_blob"] = entry.git_blob
        lines.append(
            "    " + json.dumps(mapping, ensure_ascii=True, separators=(", ", ": "))
        )
    body = ",\n".join(lines)
    return ("{\n" + "\n".join(header) + "\n" + body + "\n  ]\n}\n").encode("ascii")


@dataclass(frozen=True)
class R2Credentials:
    endpoint: str
    access_key_id: str
    secret_access_key: str = field(repr=False)


def r2_credentials_from_environment(
    environ: Mapping[str, str] = os.environ,
    *,
    credential_path: str | Path | None = None,
) -> R2Credentials | None:
    """R2 read credentials, or ``None``.

    ``R2_ACCESS_KEY_ID``/``R2_SECRET_ACCESS_KEY`` and ``R2_ENDPOINT`` or
    ``R2_ACCOUNT_ID`` win over ``~/.config/axiom-foundation/r2-credentials.json``,
    in ``axiom-corpus-ingest``'s order. Unlike it, the ``AWS_*`` variables are
    not read, so credentials meant for another service are never sent to R2.
    """

    stored: dict[str, Any] = {}
    with suppress(OSError, ValueError, RuntimeError):
        path = Path(credential_path or R2_CREDENTIAL_PATH).expanduser()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            stored = loaded

    def pick(env_name: str, *keys: str) -> str | None:
        value = environ.get(env_name)
        if value:
            return value
        for key in keys:
            if stored.get(key):
                return str(stored[key])
        return None

    access_key_id = pick(
        "R2_ACCESS_KEY_ID", "access_key_id", "accessKeyId", "accessKey"
    )
    secret_access_key = pick(
        "R2_SECRET_ACCESS_KEY", "secret_access_key", "secretAccessKey", "secretKey"
    )
    if not access_key_id or not secret_access_key:
        return None
    account_id = (
        pick("R2_ACCOUNT_ID", "account_id", "accountId") or DEFAULT_R2_ACCOUNT_ID
    )
    endpoint = pick("R2_ENDPOINT", "endpoint", "endpoint_url", "endpointUrl") or (
        f"https://{account_id}.r2.cloudflarestorage.com"
    )
    if urlsplit(endpoint).scheme != "https":
        return None
    return R2Credentials(endpoint.rstrip("/"), access_key_id, secret_access_key)


def sigv4_headers(
    method: str,
    url: str,
    *,
    access_key_id: str,
    secret_access_key: str,
    now: datetime,
    region: str = "auto",
    service: str = "s3",
    headers: Mapping[str, str] | None = None,
    payload_sha256: str = _EMPTY_PAYLOAD_SHA256,
) -> dict[str, str]:
    """AWS Signature Version 4 headers for one request with no query string."""

    parts = urlsplit(url)
    if parts.query:
        raise ValueError("sigv4_headers signs requests without a query string")
    amz_date = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    date_stamp = amz_date[:8]
    signed: dict[str, str] = {
        "host": parts.netloc,
        "x-amz-content-sha256": payload_sha256,
        "x-amz-date": amz_date,
    }
    for name, value in (headers or {}).items():
        signed[name.lower()] = value
    names = sorted(signed)
    canonical_headers = "".join(
        f"{name}:{' '.join(signed[name].strip().split())}\n" for name in names
    )
    signed_headers = ";".join(names)
    canonical_request = "\n".join(
        [
            method,
            quote(parts.path or "/", safe="/-_.~"),
            "",
            canonical_headers,
            signed_headers,
            payload_sha256,
        ]
    )
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    key = f"AWS4{secret_access_key}".encode("utf-8")
    for component in (date_stamp, region, service, "aws4_request"):
        key = hmac.new(key, component.encode("utf-8"), hashlib.sha256).digest()
    signature = hmac.new(
        key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    result = {name: signed[name] for name in names if name != "host"}
    result["Authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    return result


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect would resend the signed request elsewhere; treat it as an error."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_R2_OPENER = urllib.request.build_opener(_RefuseRedirects)


class R2ObjectSource:
    """Content-addressed objects in the release's R2 bucket, over the S3 API."""

    label = "r2"

    def __init__(
        self,
        credentials: R2Credentials,
        bucket: str,
        *,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 3,
        timeout: float = 60.0,
    ):
        self.credentials = credentials
        self.bucket = bucket
        self._opener = opener or _R2_OPENER.open
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._attempts = max(1, attempts)
        self._timeout = timeout

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        if not self.bucket or _SHA256_RE.fullmatch(artifact.sha256) is None:
            return None
        key = f"objects/sha256/{artifact.sha256[:2]}/{artifact.sha256}"
        url = f"{self.credentials.endpoint}/{quote(self.bucket, safe='')}/{key}"
        failure = "no attempt made"
        for attempt in range(self._attempts):
            if attempt:
                self._sleep(0.5 * 2 ** (attempt - 1))
            request = urllib.request.Request(
                url,
                method="GET",
                headers=sigv4_headers(
                    "GET",
                    url,
                    access_key_id=self.credentials.access_key_id,
                    secret_access_key=self.credentials.secret_access_key,
                    now=self._clock(),
                ),
            )
            try:
                response = self._opener(request, timeout=self._timeout)
            except urllib.error.HTTPError as exc:
                with suppress(Exception):
                    exc.close()
                if exc.code == 404:
                    return None
                failure = f"HTTP {exc.code} for {key}"
                if exc.code in _RETRYABLE_HTTP_STATUS:
                    continue
                raise SourceError(failure) from exc
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                TimeoutError,
                OSError,
            ) as exc:
                failure = f"request for {key} failed: {exc!r}"
                continue
            length = response.headers.get("Content-Length")
            if length is not None and (
                not length.isdigit() or int(length) != artifact.byte_count
            ):
                response.close()
                raise SourceError(
                    f"object holds {length} bytes but the release lists "
                    f"{artifact.byte_count}"
                )
            return _iter_handle(response)
        raise SourceError(failure)


@contextmanager
def release_sources(
    root: Path,
    *,
    git_commit: str,
    r2_bucket: str,
    environ: Mapping[str, str] = os.environ,
    remote: bool = True,
) -> Iterator[list[ObjectSource]]:
    """Cache, git and (with credentials) R2 sources for one release, in order."""

    sources: list[ObjectSource] = []
    cache = ContentCacheSource.from_environment(environ)
    if cache is not None:
        sources.append(cache)
    store = GitObjectStore(root)
    sources.append(GitReleaseTreeSource(store, git_commit))
    sources.append(GitLockBlobSource(store, root))
    if remote and not _REMOTE_FETCH_DISABLED.get():
        credentials = r2_credentials_from_environment(environ)
        if credentials is not None:
            sources.append(R2ObjectSource(credentials, r2_bucket))
    try:
        yield sources
    finally:
        store.close()


# ------------------------------------------------------------------------- CLI


def run_corpus_fetch(argv: Sequence[str] | None = None) -> int:
    """CLI for ``axiom-encode corpus-fetch``."""

    parser = argparse.ArgumentParser(
        prog="axiom-encode corpus-fetch",
        description=(
            "Place a pinned corpus release's artifacts in a corpus checkout that "
            "keeps its bytes outside git. A file is placed only where the "
            "checkout's own scope lock pins the release's bytes; every placed "
            "file hashes to the release's sha256; existing files are never "
            "replaced. Exit 0 when every artifact is present, placed or skipped "
            "(skipped scopes fail when read); 1 when an artifact the lock pins "
            "could not be placed or its file differs from the lock; 2 when the "
            "release cannot be loaded."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        required=True,
        type=Path,
        help="Root of the axiom-corpus checkout.",
    )
    pin = parser.add_mutually_exclusive_group(required=True)
    pin.add_argument(
        "--rulespec-root",
        type=Path,
        help="RuleSpec checkout whose .axiom/toolchain.toml pins the release.",
    )
    pin.add_argument("--release", help="Release name (requires --content-sha256).")
    parser.add_argument("--content-sha256", help="Release content sha256.")
    parser.add_argument(
        "--artifact-class",
        action="append",
        choices=PLACEABLE_ARTIFACT_CLASSES,
        help=(
            "Artifact class to place (repeatable; default: provisions). Sources "
            "are fetched all or nothing by `axiom-corpus-ingest corpus fetch`."
        ),
    )
    parser.add_argument("--verify", action="store_true", help="Hash present files too.")
    parser.add_argument(
        "--no-remote", action="store_true", help="Use only the cache and git objects."
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.rulespec_root is not None:
            if args.content_sha256:
                parser.error("--content-sha256 goes with --release")
            from axiom_encode.toolchain import load_rulespec_corpus_release_pin

            name, content_sha256 = load_rulespec_corpus_release_pin(args.rulespec_root)
        else:
            if not args.content_sha256:
                parser.error("--release requires --content-sha256")
            name, content_sha256 = args.release, args.content_sha256
        root, verified = load_pinned_release(args.corpus_path, name, content_sha256)
        classes = set(args.artifact_class or ("provisions",))
        artifacts = [a for a in verified.artifacts if a.artifact_class in classes]
        report = materialize_release_artifacts(
            root,
            artifacts,
            sources=lambda: release_sources(
                root,
                git_commit=verified.git_commit,
                r2_bucket=verified.r2_bucket,
                remote=not args.no_remote,
            ),
            verify=args.verify,
            release_commit=verified.git_commit,
        )
    except (CorpusMaterializationError, CorpusReleaseObjectError, ValueError) as exc:
        print(f"axiom-encode corpus-fetch: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = {
            "release": name,
            "content_sha256": content_sha256,
            **report.to_mapping(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{name}: {report.selected} artifact(s): {len(report.present)} present, "
            f"{len(report.materialized)} placed ({report.bytes_materialized:,} bytes), "
            f"{len(report.skipped)} skipped, {len(report.modified)} modified, "
            f"{len(report.failed)} failed."
        )
        if report.materialized:
            print(
                "sources: "
                + ", ".join(f"{k} {v}" for k, v in report.source_counts().items())
            )
        if report.skipped:
            print(
                "skipped (the checkout's locks do not pin the release's bytes; "
                "reading these scopes fails):\n" + report.describe_skipped(limit=20),
                file=sys.stderr,
            )
        if report.modified:
            print(
                "left untouched (the files differ from the release bytes their "
                "locks pin):\n" + report.describe_modified(limit=20),
                file=sys.stderr,
            )
        if report.failed:
            print(report.describe_failures(limit=20), file=sys.stderr)
    return 0 if report.ok and not report.modified else 1


def load_pinned_release(
    corpus_path: Path, name: str, content_sha256: str
) -> tuple[Path, VerifiedCorpusReleaseObject]:
    """Read ``releases/<name>/<sha>.json`` and bind it to the pinned digest.

    This checks the schema and the content digest, not the signature: every
    artifact sha256 sits inside the pinned content, so the pin alone binds the
    bytes placed here. Consumers still verify the signature before reading.
    """

    from axiom_encode.corpus_resolver import (
        MAX_RELEASE_OBJECT_BYTES,
        read_bounded_regular_file,
        validate_corpus_release_name,
    )

    name = validate_corpus_release_name(name)
    if _SHA256_RE.fullmatch(content_sha256 or "") is None:
        raise ValueError("content sha256 must be a lowercase sha256 digest")
    raw_root = Path(os.path.abspath(Path(corpus_path).expanduser()))
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError(f"corpus root must be a real directory: {raw_root}")
    root = raw_root.resolve(strict=True)
    release_object_path = root / "releases" / name / f"{content_sha256}.json"
    if not release_object_path.is_file():
        raise ValueError(f"Corpus release object not found: {release_object_path}")
    raw = read_bounded_regular_file(
        root,
        release_object_path,
        label="corpus release object",
        max_bytes=MAX_RELEASE_OBJECT_BYTES,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Corpus release object is not valid UTF-8 JSON: {release_object_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Corpus release object must be a JSON object")
    verified = verify_pinned_release_object_content(
        payload, name=name, content_sha256=content_sha256
    )
    return root, verified
