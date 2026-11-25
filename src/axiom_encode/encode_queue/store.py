"""Storage backends of the central encode queue.

A backend holds a small set of JSON documents and one revision for all of
them. ``commit`` is a compare-and-swap: it applies the changes only when the
revision is still the one the caller read. That one primitive gives the
atomic claim. Two workers that claim the same unit both read revision R; the
first commit moves the revision, the second gets :class:`RevisionConflict`,
reads again and sees the unit reserved.

``LocalFileStore`` keeps the documents in a directory (tests, single-machine
use). ``GitHubBranchStore`` keeps them on a branch of a GitHub repository and
moves the branch with a fast-forward-only reference update, so a worker needs
only the ``gh`` login it already has.
"""

from __future__ import annotations

import base64
import fcntl
import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from axiom_encode.encode_queue.model import QueueError

Document = dict[str, Any]


class RevisionConflict(Exception):
    """The store moved after the caller read it. Read again and retry."""


class Snapshot(Protocol):
    """The documents of the store at one revision."""

    revision: str

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


# --------------------------------------------------------------------------
# Local directory
# --------------------------------------------------------------------------


class _LocalSnapshot:
    def __init__(self, revision: str, documents: dict[str, Document]):
        self.revision = revision
        self._documents = documents

    def paths(self) -> list[str]:
        return sorted(self._documents)

    def read(self, path: str) -> Document | None:
        document = self._documents.get(path)
        return None if document is None else json.loads(json.dumps(document))


class LocalFileStore:
    """Queue documents in a directory, guarded by one lock file.

    The revision is a counter in ``.revision``. Readers and writers take the
    same ``flock``, so a snapshot never mixes two revisions.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

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
            documents: dict[str, Document] = {}
            for file in sorted(self.root.rglob("*.json")):
                relative = file.relative_to(self.root).as_posix()
                documents[relative] = json.loads(file.read_text())
            return _LocalSnapshot(self._revision(), documents)

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


class GitHubApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status
        self.message = message


# (method, path below the API root, JSON payload or None) -> decoded JSON.
GitHubApi = Callable[[str, str, dict[str, Any] | None], Any]


def gh_cli_api(method: str, path: str, payload: dict[str, Any] | None) -> Any:
    """Call the GitHub REST API through the ``gh`` login of this machine.

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
        raise GitHubApiError(0, "gh api timed out") from error
    head, _, body = result.stdout.replace("\r\n", "\n").partition("\n\n")
    status_line = head.splitlines()[0] if head else ""
    parts = status_line.split()
    status = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    if result.returncode != 0 or not 200 <= status < 300:
        detail = body.strip() or result.stderr.strip()
        try:
            detail = json.loads(body).get("message", detail)
        except (ValueError, AttributeError):
            pass
        raise GitHubApiError(status, str(detail)[:300])
    return json.loads(body) if body.strip() else None


class _GitHubSnapshot:
    def __init__(
        self,
        store: GitHubBranchStore,
        revision: str,
        blobs: dict[str, str],
    ):
        self.revision = revision
        self._store = store
        self._blobs = blobs
        self._cache: dict[str, Document] = {}

    def paths(self) -> list[str]:
        return sorted(self._blobs)

    def read(self, path: str) -> Document | None:
        if path not in self._blobs:
            return None
        if path not in self._cache:
            blob = self._store._api(
                "GET", f"{self._store._repo_path}/git/blobs/{self._blobs[path]}", None
            )
            text = base64.b64decode(blob["content"]).decode("utf-8")
            self._cache[path] = json.loads(text)
        return json.loads(json.dumps(self._cache[path]))


class GitHubBranchStore:
    """Queue documents under ``prefix/`` on one branch of a repository.

    A commit is built with the Git Data API on top of the commit the caller
    read, then the branch is moved with ``force: false``. GitHub refuses that
    update when the branch moved in between, which is the compare-and-swap.
    The revision is the commit SHA; an absent branch has revision ``""`` and
    the first commit creates it (creation of an existing reference fails, so
    two first commits cannot both win).
    """

    def __init__(
        self,
        repo: str,
        branch: str = "encoding-queue-state",
        prefix: str = "central-queue",
        api: GitHubApi = gh_cli_api,
    ):
        owner, _, name = repo.partition("/")
        if not owner or not name or "/" in name:
            raise QueueError("repo must be OWNER/NAME")
        if not branch or branch.startswith("-") or ".." in branch:
            raise QueueError("invalid branch")
        self._repo_path = f"repos/{repo}"
        self._branch = branch
        self._prefix = prefix.strip("/")
        self._api = api

    def _head(self) -> str:
        try:
            ref = self._api(
                "GET", f"{self._repo_path}/git/ref/heads/{self._branch}", None
            )
        except GitHubApiError as error:
            if error.status == 404:
                return ""
            raise
        return ref["object"]["sha"]

    def snapshot(self) -> Snapshot:
        head = self._head()
        blobs: dict[str, str] = {}
        if head:
            tree = self._api(
                "GET", f"{self._repo_path}/git/trees/{head}?recursive=1", None
            )
            if tree.get("truncated"):
                raise QueueError("the queue branch tree is too large to list")
            lead = self._prefix + "/"
            for item in tree.get("tree", []):
                path = item.get("path", "")
                if item.get("type") == "blob" and path.startswith(lead):
                    relative = path[len(lead) :]
                    if relative.endswith(".json"):
                        blobs[relative] = item["sha"]
        return _GitHubSnapshot(self, head, blobs)

    def commit(
        self,
        base_revision: str,
        changes: dict[str, Document | None],
        message: str,
    ) -> str:
        entries = []
        for path, document in changes.items():
            _check_path(path)
            entry: dict[str, Any] = {
                "path": f"{self._prefix}/{path}",
                "mode": "100644",
                "type": "blob",
            }
            if document is None:
                entry["sha"] = None
            else:
                entry["content"] = _dump(document)
            entries.append(entry)
        tree_request: dict[str, Any] = {"tree": entries}
        if base_revision:
            tree_request["base_tree"] = base_revision
        else:
            # A new branch has no base tree, and a deletion needs one.
            tree_request["tree"] = [e for e in entries if "content" in e]
        tree = self._api("POST", f"{self._repo_path}/git/trees", tree_request)
        commit = self._api(
            "POST",
            f"{self._repo_path}/git/commits",
            {
                "message": message,
                "tree": tree["sha"],
                "parents": [base_revision] if base_revision else [],
            },
        )
        try:
            if base_revision:
                self._api(
                    "PATCH",
                    f"{self._repo_path}/git/refs/heads/{self._branch}",
                    {"sha": commit["sha"], "force": False},
                )
            else:
                self._api(
                    "POST",
                    f"{self._repo_path}/git/refs",
                    {"ref": f"refs/heads/{self._branch}", "sha": commit["sha"]},
                )
        except GitHubApiError as error:
            # 422: not a fast-forward, or the reference already exists.
            # 409: the reference is being updated by another request.
            if error.status in (409, 422):
                raise RevisionConflict(error.message) from error
            raise
        return commit["sha"]
