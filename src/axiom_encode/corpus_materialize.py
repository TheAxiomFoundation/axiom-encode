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

Bytes land at their path only after they verify, through a hard link from a
temporary file, so a reader never sees partial or unverified bytes. An
existing file is never replaced, and no symlink is followed or created under
the corpus root.
"""

from __future__ import annotations

import argparse
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
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
NO_FETCH_ENV = "AXIOM_CORPUS_NO_FETCH"
CACHE_ENV = "AXIOM_CORPUS_CACHE"
DEFAULT_CACHE_ROOT = "~/.axiom/corpus-cache"
R2_CREDENTIAL_PATH = "~/.config/axiom-foundation/r2-credentials.json"
DEFAULT_R2_ACCOUNT_ID = "011fb8d44f0e4d9832265ac9f748bc6b"
ARTIFACT_CLASSES = ("provisions", "inventory", "coverage", "sources")
MAX_LOCK_FILE_BYTES = 64 * 1024 * 1024
MAX_RELEASE_OBJECT_BYTES = 16 * 1024 * 1024
_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})

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
    """True when ``AXIOM_CORPUS_NO_FETCH`` turns materialization off."""

    return environ.get(NO_FETCH_ENV, "").strip() not in {"", "0"}


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
    """What one materialization pass found, placed, and could not place."""

    selected: int = 0
    present: list[str] = field(default_factory=list)
    materialized: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    bytes_materialized: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.materialized.values():
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def describe_failures(self, limit: int = 10) -> str:
        items = sorted(self.failed.items())
        lines = [f"{path}: {reason}" for path, reason in items[:limit]]
        if len(items) > limit:
            lines.append(f"... and {len(items) - limit} more")
        return "\n".join(lines)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "present": len(self.present),
            "materialized": len(self.materialized),
            "bytes_materialized": self.bytes_materialized,
            "sources": self.source_counts(),
            "failed": dict(sorted(self.failed.items())),
        }


def materialize_release_artifacts(
    root: Path,
    artifacts: Iterable[VerifiedReleaseArtifact],
    *,
    sources: SourceFactory,
    verify: bool = False,
) -> MaterializationReport:
    """Place every missing artifact from the first source whose bytes verify.

    A present regular file of the listed size counts as present; ``verify``
    also hashes it. Readers hash what they read (``corpus_resolver``), so the
    size check only decides whether to fetch. A present file of another size
    or hash, or a symlink anywhere on an artifact's path, is reported as a
    failure and left untouched. ``sources`` is opened only when an artifact
    is missing, so a fully materialized checkout starts no process and reads
    no credentials.
    """

    root = _require_root(root)
    report = MaterializationReport()
    missing: list[VerifiedReleaseArtifact] = []
    for artifact in artifacts:
        report.selected += 1
        try:
            _require_canonical_artifact_path(artifact.path)
            state = destination_state(root, artifact, verify=verify)
        except CorpusMaterializationError as exc:
            report.failed[artifact.path] = str(exc)
            continue
        if state == "present":
            report.present.append(artifact.path)
        else:
            missing.append(artifact)
    if not missing:
        return report
    with sources() as opened:
        for artifact in missing:
            _materialize_one(root, artifact, opened, report)
    return report


def destination_state(
    root: Path,
    artifact: VerifiedReleaseArtifact,
    *,
    verify: bool = False,
) -> str:
    """Return ``"present"`` or ``"missing"``; raise for anything unsafe or different."""

    parts = PurePosixPath(artifact.path).parts
    cursor = root
    for part in parts[:-1]:
        cursor = cursor / part
        try:
            mode = os.lstat(cursor).st_mode
        except FileNotFoundError:
            return "missing"
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
        return "missing"
    if stat.S_ISLNK(file_stat.st_mode):
        raise CorpusMaterializationError(f"artifact is a symlink: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise CorpusMaterializationError(f"artifact is not a regular file: {path}")
    if file_stat.st_size != artifact.byte_count:
        raise CorpusMaterializationError(
            f"existing file holds {file_stat.st_size} bytes but the release lists "
            f"{artifact.byte_count}; left untouched"
        )
    if verify and _sha256_path(path) != artifact.sha256:
        raise CorpusMaterializationError(
            "existing file's sha256 differs from the release; left untouched"
        )
    return "present"


def _materialize_one(
    root: Path,
    artifact: VerifiedReleaseArtifact,
    sources: Sequence[ObjectSource],
    report: MaterializationReport,
) -> None:
    reasons: list[str] = []
    created: list[Path] = []
    try:
        parent = _ensure_parent(root, artifact.path, created)
        name = PurePosixPath(artifact.path).name
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
                _place_verified(root, parent, name, chunks, artifact)
            except SourceError as exc:
                reasons.append(f"{source.label}: {exc}")
                continue
            report.materialized[artifact.path] = source.label
            report.bytes_materialized += artifact.byte_count
            return
        report.failed[artifact.path] = "; ".join(reasons) or "no source is configured"
    except CorpusMaterializationError as exc:
        report.failed[artifact.path] = str(exc)
    # Nothing was placed: remove the directories this attempt created, so a
    # failed run leaves the tree as it found it (rmdir refuses non-empty ones).
    for directory in reversed(created):
        with suppress(OSError):
            os.rmdir(directory)


def _place_verified(
    root: Path,
    parent: Path,
    name: str,
    chunks: Iterator[bytes],
    artifact: VerifiedReleaseArtifact,
) -> None:
    """Stream into a private temporary file, verify, then link it into place."""

    temporary = parent / f".{name}.{os.getpid()}.{os.urandom(6).hex()}.part"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        digest = hashlib.sha256()
        size = 0
        try:
            descriptor = os.open(temporary, flags, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
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
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot write {temporary}: {exc}"
            ) from exc
        if size != artifact.byte_count:
            raise SourceError(
                f"supplied {size} bytes but the release lists {artifact.byte_count}"
            )
        if digest.hexdigest() != artifact.sha256:
            raise SourceError("sha256 does not match the release")
        destination = parent / name
        try:
            os.link(temporary, destination)
        except FileExistsError:
            # Another process placed it first; accept only the listed size.
            destination_state(root, artifact)
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot link verified bytes into place: {destination}: {exc}"
            ) from exc
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _ensure_parent(root: Path, relative: str, created: list[Path]) -> Path:
    cursor = root
    for part in PurePosixPath(relative).parts[:-1]:
        cursor = cursor / part
        try:
            os.mkdir(cursor, 0o755)
            created.append(cursor)
        except FileExistsError:
            pass
        except OSError as exc:
            raise CorpusMaterializationError(
                f"cannot create directory {cursor}: {exc}"
            ) from exc
        mode = os.lstat(cursor).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise CorpusMaterializationError(
                f"path component is not a real directory: {cursor}"
            )
    return cursor


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
    with open(path, "rb") as handle:
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
        root = Path(environ.get(CACHE_ENV) or DEFAULT_CACHE_ROOT).expanduser()
        return cls(root) if root.is_dir() else None

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        path = self.root / "objects" / "sha256" / artifact.sha256[:2] / artifact.sha256
        try:
            handle = open(path, "rb")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SourceError(f"cannot read {path}: {exc.strerror}") from exc
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
        fields = header.split()
        if len(fields) == 2 and fields[1] in {b"missing", b"ambiguous"}:
            return None
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
            self._in_flight = False
        finally:
            if not complete:
                self._stop()

    def _start(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            assert self.git is not None
            self._process = subprocess.Popen(
                [self.git, "-C", str(self.root), "cat-file", "--batch"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
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
        self.root = Path(root)
        self._locks: dict[PurePosixPath, dict[str, tuple[str, str]]] = {}

    def open(self, artifact: VerifiedReleaseArtifact) -> Iterator[bytes] | None:
        lock_path = lock_path_for(artifact.path)
        if lock_path is None:
            return None
        if lock_path not in self._locks:
            self._locks[lock_path] = _read_lock_blobs(self.root / lock_path)
        sha256, blob = self._locks[lock_path].get(artifact.path, ("", ""))
        if sha256 != artifact.sha256 or not blob:
            return None
        return self.store.open(blob, artifact.byte_count)


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


def _read_lock_blobs(path: Path) -> dict[str, tuple[str, str]]:
    """``path -> (sha256, git_blob)`` hints from one lock file; bytes still verify."""

    try:
        file_stat = os.lstat(path)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size > MAX_LOCK_FILE_BYTES
        ):
            return {}
        payload = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return {}
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        return {}
    hints: dict[str, tuple[str, str]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            continue
        entry_path, sha256, blob = (
            entry.get("path"),
            entry.get("sha256"),
            entry.get("git_blob"),
        )
        if (
            isinstance(entry_path, str)
            and isinstance(sha256, str)
            and isinstance(blob, str)
            and _OBJECT_ID_RE.fullmatch(blob)
        ):
            hints[entry_path] = (sha256, blob)
    return hints


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
    """R2 credentials the way ``axiom-corpus-ingest`` finds them, or ``None``.

    ``R2_ACCESS_KEY_ID``/``R2_SECRET_ACCESS_KEY`` and ``R2_ENDPOINT`` or
    ``R2_ACCOUNT_ID`` win over ``~/.config/axiom-foundation/r2-credentials.json``.
    """

    path = Path(credential_path or R2_CREDENTIAL_PATH).expanduser()
    stored: dict[str, Any] = {}
    with suppress(OSError, ValueError):
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
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                failure = f"request for {key} failed: {exc}"
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
            "keeps its bytes outside git. Every placed file hashes to the "
            "release's sha256; existing files are never replaced."
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
        choices=ARTIFACT_CLASSES,
        help="Artifact class to place (repeatable; default: provisions).",
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
            f"{len(report.failed)} failed."
        )
        if report.materialized:
            print(
                "sources: "
                + ", ".join(f"{k} {v}" for k, v in report.source_counts().items())
            )
        if report.failed:
            print(report.describe_failures(limit=20), file=sys.stderr)
    return 0 if report.ok else 1


def load_pinned_release(
    corpus_path: Path, name: str, content_sha256: str
) -> tuple[Path, VerifiedCorpusReleaseObject]:
    """Read ``releases/<name>/<sha>.json`` and bind it to the pinned digest.

    This checks the schema and the content digest, not the signature: every
    artifact sha256 sits inside the pinned content, so the pin alone binds the
    bytes placed here. Consumers still verify the signature before reading.
    """

    from axiom_encode.corpus_resolver import (
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
