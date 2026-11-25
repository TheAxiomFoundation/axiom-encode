"""Storage backends of the central encode queue.

A backend holds a small set of JSON documents and one revision for all of
them. ``commit`` is a compare-and-swap: it applies the changes only when the
revision is still the one the caller read. That one primitive gives the
atomic claim. Two workers that claim the same unit both read revision R; the
first commit moves the revision, the second gets :class:`RevisionConflict`,
reads again and sees the unit reserved.

A snapshot also carries the time of the store (``now``). Lease decisions
use that time, never the clock of the worker.

``LocalFileStore`` keeps the documents in a directory (tests, single-machine
use). ``GitHubBranchStore`` keeps them on a branch of a GitHub repository.
A write is one GraphQL ``createCommitOnBranch`` request with
``expectedHeadOid``: GitHub applies it only when the branch still points at
the commit the caller read. A worker needs only the ``gh`` login it already
has.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from axiom_encode.encode_queue.model import QueueError, utc_now

Document = dict[str, Any]


class RevisionConflict(Exception):
    """The store moved after the caller read it. Read again and retry."""


class Snapshot(Protocol):
    """The documents of the store at one revision."""

    revision: str
    now: datetime

    def paths(self) -> list[str]: ...

    def read(self, path: str) -> Document | None: ...


class QueueStore(Protocol):
    def snapshot(self) -> Snapshot: ...

    def commit(
        self,
        base_revision: str,
        changes: dict[str, Document | None],
        message: str,
    ) -> str:
        """Write ``changes`` (``None`` deletes) when the store is still at
        ``base_revision``. Return the new revision or raise
        :class:`RevisionConflict`."""
        ...


def _check_path(path: str) -> str:
    pure = PurePosixPath(path)
    if (
        not path
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.as_posix() != path
        or pure.suffix != ".json"
        or any(part.startswith(".") for part in pure.parts)
    ):
        raise QueueError(f"invalid queue document path: {path!r}")
    return path


def _dump(document: Document) -> str:
    return json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _load(text: str, path: str) -> Document:
    try:
        document = json.loads(text)
    except ValueError:
        raise QueueError(f"queue document {path} is not JSON") from None
    if not isinstance(document, dict):
        raise QueueError(f"queue document {path} is not a mapping")
    return document


# --------------------------------------------------------------------------
# Local directory
# --------------------------------------------------------------------------


class _LocalSnapshot:
    def __init__(self, revision: str, now: datetime, texts: dict[str, str]):
        self.revision = revision
        self.now = now
        self._texts = texts

    def paths(self) -> list[str]:
        return sorted(self._texts)

    def read(self, path: str) -> Document | None:
        text = self._texts.get(path)
        return None if text is None else _load(text, path)


class LocalFileStore:
    """Queue documents in a directory, guarded by one lock file.

    The revision is a counter in ``.revision``. Readers and writers take the
    same ``flock``, so a snapshot never mixes two revisions. The time of the
    store is the clock of this one machine.
    """

    def __init__(self, root: Path | str, clock: Callable[[], datetime] = utc_now):
        self.root = Path(root)
        self._clock = clock

    def _locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        handle = open(self.root / ".lock", "a+")  # noqa: SIM115 - closed by caller
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _revision(self) -> str:
        try:
            return (self.root / ".revision").read_text().strip() or "0"
        except FileNotFoundError:
            return "0"

    def snapshot(self) -> Snapshot:
        with self._locked():
            texts: dict[str, str] = {}
            for file in sorted(self.root.rglob("*.json")):
                relative = file.relative_to(self.root).as_posix()
                texts[relative] = file.read_text()
            return _LocalSnapshot(self._revision(), self._clock(), texts)

    def commit(
        self,
        base_revision: str,
        changes: dict[str, Document | None],
        message: str,
    ) -> str:
        del message  # a directory keeps no history
        for path in changes:
            _check_path(path)
        with self._locked():
            current = self._revision()
            if current != base_revision:
                raise RevisionConflict(
                    f"store is at revision {current}, not {base_revision}"
                )
            for path, document in changes.items():
                target = self.root / path
                if document is None:
                    target.unlink(missing_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write_atomic(target, _dump(document))
            new_revision = str(int(current) + 1)
            self._write_atomic(self.root / ".revision", new_revision + "\n")
            return new_revision

    @staticmethod
    def _write_atomic(target: Path, text: str) -> None:
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


# --------------------------------------------------------------------------
# GitHub branch
# --------------------------------------------------------------------------


@dataclass
class ApiResponse:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)  # lower-case names


class GitHubApiError(Exception):
    """GitHub answered, and the answer is a refusal. Nothing was written."""

    def __init__(
        self, status: int, message: str, headers: dict[str, str] | None = None
    ):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status
        self.message = message
        self.headers = headers or {}


class GitHubUnknownResult(Exception):
    """No usable answer (timeout, lost connection, 5xx).

    A write may or may not have been applied.
    """


# (method, path below the API root, JSON payload or None) -> response.
# Raises GitHubApiError for a 4xx answer and GitHubUnknownResult when there
# is no usable answer.
GitHubApi = Callable[[str, str, dict[str, Any] | None], ApiResponse]


def gh_cli_api(method: str, path: str, payload: dict[str, Any] | None) -> ApiResponse:
    """Call the GitHub API through the ``gh`` login of this machine.

    No token is read, stored or passed by this module.
    """

    command = ["gh", "api", "--method", method, path, "--include"]
    if payload is not None:
        command += ["--input", "-"]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            input=None if payload is None else json.dumps(payload),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise QueueError("the GitHub queue backend needs the gh CLI") from error
    except subprocess.TimeoutExpired as error:
        raise GitHubUnknownResult("gh api timed out") from error
    head, _, text = result.stdout.replace("\r\n", "\n").partition("\n\n")
    lines = head.splitlines()
    parts = lines[0].split() if lines else []
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    headers = {}
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if separator:
            headers[name.strip().lower()] = value.strip()
    try:
        body = json.loads(text) if text.strip() else None
    except ValueError:
        body = None
    if status == 0 or status >= 500:
        raise GitHubUnknownResult(
            f"no usable answer from GitHub (status {status}): "
            f"{result.stderr.strip()[:200]}"
        )
    if not 200 <= status < 300:
        message = body.get("message") if isinstance(body, dict) else None
        raise GitHubApiError(status, str(message or text.strip())[:300], headers)
    return ApiResponse(status, body, headers)


_COMMIT_MUTATION = (
    "mutation($input: CreateCommitOnBranchInput!) {"
    " createCommitOnBranch(input: $input) { commit { oid } } }"
)
REQUEST_TRAILER = "Queue-Request"
MAX_RATE_LIMIT_WAITS = 4
MAX_RATE_LIMIT_SECONDS = 300


def git_blob_sha(data: bytes) -> str:
    header = b"blob %d\x00" % len(data)
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


class _GitHubSnapshot:
    def __init__(
        self,
        store: GitHubBranchStore,
        revision: str,
        now: datetime,
        blobs: dict[str, str],
    ):
        self.revision = revision
        self.now = now
        self._store = store
        self._blobs = blobs

    def paths(self) -> list[str]:
        return sorted(self._blobs)

    def read(self, path: str) -> Document | None:
        if path not in self._blobs:
            return None
        return _load(self._store._blob_text(self._blobs[path]), path)


class GitHubBranchStore:
    """Queue documents under ``prefix/`` on one branch of a repository.

    Read: the branch head (its ``Date`` header is the time of the store),
    the tree, and the blobs that are not in the cache. Blobs are cached by
    their SHA (in memory, and in ``cache_dir`` when given), so a read after
    another machine wrote fetches only the documents that changed.

    Write: one ``createCommitOnBranch`` request with ``expectedHeadOid``.

    - A stale head is :class:`RevisionConflict`: the caller reads again.
    - Any other refusal (for example a branch rule) is a
      :class:`GitHubApiError` at once. It is not retried.
    - A rate limit (403 or 429 with the hints of the server) waits as the
      server says and sends the same request again. Nothing was written.
    - No usable answer: the store reads the branch again and looks for the
      request id of this write in the new commits. Found means done.

    The branch must exist. A missing branch is an error, never an empty
    queue.
    """

    def __init__(
        self,
        repo: str,
        branch: str = "encoding-queue-state",
        prefix: str = "central-queue",
        api: GitHubApi = gh_cli_api,
        *,
        cache_dir: Path | str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        owner, _, name = repo.partition("/")
        if not owner or not name or "/" in name:
            raise QueueError("repo must be OWNER/NAME")
        if not branch or branch.startswith("-") or ".." in branch:
            raise QueueError("invalid branch")
        self._repo = repo
        self._repo_path = f"repos/{repo}"
        self._branch = branch
        self._prefix = prefix.strip("/")
        self._api = api
        self._sleep = sleep
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._memory: dict[str, str] = {}

    # ------------------------------------------------------------- requests

    @staticmethod
    def _rate_limit_wait(error: GitHubApiError, attempt: int) -> float | None:
        """Seconds to wait when ``error`` is a rate limit, else ``None``."""

        headers = error.headers
        limited = error.status == 429 or (
            error.status == 403
            and (
                "retry-after" in headers
                or headers.get("x-ratelimit-remaining") == "0"
                or "rate limit" in error.message.lower()
            )
        )
        if not limited:
            return None
        wait = 60.0 * (attempt + 1)
        if headers.get("retry-after", "").isdigit():
            wait = float(headers["retry-after"])
        elif headers.get("x-ratelimit-reset", "").isdigit():
            wait = float(headers["x-ratelimit-reset"]) - time.time()
        return min(max(wait, 1.0), MAX_RATE_LIMIT_SECONDS)

    def _call(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> ApiResponse:
        for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
            try:
                return self._api(method, path, payload)
            except GitHubApiError as error:
                wait = self._rate_limit_wait(error, attempt)
                if wait is None or attempt == MAX_RATE_LIMIT_WAITS:
                    raise
                self._sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    def _read(self, path: str) -> ApiResponse:
        """A read is safe to repeat when there was no usable answer."""

        for attempt in range(3):
            try:
                return self._call("GET", path, None)
            except GitHubUnknownResult:
                if attempt == 2:
                    raise
                self._sleep(2.0 * (attempt + 1))
        raise AssertionError("unreachable")  # pragma: no cover

    # ---------------------------------------------------------------- reads

    def _head(self) -> tuple[str, datetime]:
        try:
            response = self._read(f"{self._repo_path}/git/ref/heads/{self._branch}")
        except GitHubApiError as error:
            if error.status == 404:
                raise QueueError(
                    f"queue branch {self._branch} of {self._repo} was not found. "
                    "The queue never creates its branch: check the name, or ask "
                    "a maintainer to restore it."
                ) from None
            raise
        try:
            now = parsedate_to_datetime(response.headers["date"])
        except (KeyError, TypeError, ValueError):
            raise QueueError(
                "GitHub gave no Date header; the queue has no clock"
            ) from None
        return response.body["object"]["sha"], now

    def _blob_text(self, sha: str) -> str:
        if sha in self._memory:
            return self._memory[sha]
        cached = self._cache_dir / sha if self._cache_dir else None
        data: bytes | None = None
        if cached is not None and cached.is_file():
            data = cached.read_bytes()
            if git_blob_sha(data) != sha:
                data = None  # a damaged or edited cache file is not used
        if data is None:
            blob = self._read(f"{self._repo_path}/git/blobs/{sha}").body
            data = base64.b64decode(blob["content"])
            if git_blob_sha(data) != sha:
                raise QueueError(f"GitHub returned other content for blob {sha}")
            fetched = True
        else:
            fetched = False
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise QueueError(f"queue blob {sha} is not UTF-8 text") from None
        if fetched and cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            LocalFileStore._write_atomic(cached, text)
        self._memory[sha] = text
        return text

    def snapshot(self) -> Snapshot:
        head, now = self._head()
        tree = self._read(f"{self._repo_path}/git/trees/{head}?recursive=1").body
        if tree.get("truncated"):
            raise QueueError("the queue branch tree is too large to list")
        blobs: dict[str, str] = {}
        lead = self._prefix + "/"
        for item in tree.get("tree", []):
            path = item.get("path", "")
            if item.get("type") == "blob" and path.startswith(lead):
                relative = path[len(lead) :]
                if relative.endswith(".json"):
                    blobs[relative] = item["sha"]
        return _GitHubSnapshot(self, head, now, blobs)

    # --------------------------------------------------------------- writes

    def commit(
        self,
        base_revision: str,
        changes: dict[str, Document | None],
        message: str,
    ) -> str:
        additions = []
        deletions = []
        for path, document in changes.items():
            _check_path(path)
            full = f"{self._prefix}/{path}"
            if document is None:
                deletions.append({"path": full})
            else:
                content = base64.b64encode(_dump(document).encode()).decode()
                additions.append({"path": full, "contents": content})
        request_id = uuid.uuid4().hex
        payload = {
            "query": _COMMIT_MUTATION,
            "variables": {
                "input": {
                    "branch": {
                        "repositoryNameWithOwner": self._repo,
                        "branchName": self._branch,
                    },
                    "expectedHeadOid": base_revision,
                    "message": {
                        "headline": message[:200],
                        "body": f"{REQUEST_TRAILER}: {request_id}",
                    },
                    "fileChanges": {"additions": additions, "deletions": deletions},
                }
            },
        }
        for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
            try:
                response = self._call("POST", "graphql", payload)
            except GitHubUnknownResult:
                return self._recover(base_revision, request_id)
            body = response.body if isinstance(response.body, dict) else {}
            errors = body.get("errors") or []
            first = errors[0] if errors and isinstance(errors[0], dict) else {}
            if first.get("type") == "RATE_LIMITED" and attempt < MAX_RATE_LIMIT_WAITS:
                # GraphQL reports a rate limit with status 200. Not written.
                limited = GitHubApiError(429, "rate limited", response.headers)
                self._sleep(self._rate_limit_wait(limited, attempt) or 60.0)
                continue
            break
        if errors:
            self._raise_for(errors, response.headers)
        try:
            return body["data"]["createCommitOnBranch"]["commit"]["oid"]
        except (KeyError, TypeError):
            return self._recover(base_revision, request_id)

    @staticmethod
    def _raise_for(errors: list[Any], headers: dict[str, str]) -> None:
        first = errors[0] if isinstance(errors[0], dict) else {}
        kind = str(first.get("type", ""))
        message = str(first.get("message", "unknown error"))[:300]
        if kind == "STALE_DATA" or "Expected branch to point to" in message:
            raise RevisionConflict(message)
        # Everything else is a refusal that a retry does not change: a branch
        # rule, a missing branch, no permission, a bad path.
        raise GitHubApiError(
            422, f"queue write refused ({kind or 'error'}): {message}", headers
        )

    def _recover(self, base_revision: str, request_id: str) -> str:
        """Decide what an unanswered write did, from the branch itself."""

        head, _now = self._head()
        if head == base_revision:
            # Not applied (yet). The caller reads again and decides again;
            # request ids in the queue make that repeat safe.
            raise RevisionConflict("the write got no answer and is not on the branch")
        commits = self._read(f"{self._repo_path}/commits?sha={head}&per_page=30").body
        needle = f"{REQUEST_TRAILER}: {request_id}"
        for commit in commits or []:
            if needle in str((commit.get("commit") or {}).get("message", "")):
                return commit["sha"]
        raise RevisionConflict("the write got no answer and is not on the branch")
