"""Read-only views of a RuleSpec checkout and of a local corpus checkout.

The triage probes ask two questions of recorded state: does a RuleSpec module
export a symbol at the run's RuleSpec ref, and does a corpus row exist. Both
views only read local files and local git objects. Nothing is fetched, no
credential is read, and nothing outside a private temporary directory is
written.

``RulespecSnapshot`` copies the few files a probe asks about from a view into
a temporary directory laid out as a canonical ``rulespec-<country>`` checkout,
so the encoder's own import resolvers can run on it unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ..corpus_resolver import (
    MAX_CORPUS_PROVISION_BYTES,
    CorpusResolutionError,
    UnsafeCorpusPathError,
    read_bounded_regular_file,
    require_canonical_corpus_citation_path,
)
from ..repo_routing import monorepo_checkout_name

MAX_RULESPEC_FILE_BYTES = 4 * 1024 * 1024
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SAFE_REF_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._/-]{0,199}")
_GIT_ENV_KEEP = ("HOME", "PATH", "TMPDIR", "LANG", "LC_ALL")


def _git_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _GIT_ENV_KEEP if key in os.environ}
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            # A partial clone must never fetch a missing object.
            "GIT_NO_LAZY_FETCH": "1",
            # The recorded commit, not a local replacement of its objects.
            "GIT_NO_REPLACE_OBJECTS": "1",
            # No transport at all: these commands only read local objects.
            "GIT_ALLOW_PROTOCOL": "none",
        }
    )
    return env


def _run_git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes | None:
    """Run one read-only local git plumbing command; ``None`` on failure."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            input=input_bytes,
            capture_output=True,
            check=False,
            env=_git_env(),
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_head(repo: Path) -> str | None:
    """Return the HEAD commit of a local git checkout, or ``None``."""

    # Only the checkout's own repository: a plain directory inside some other
    # repository must not take that repository's HEAD.
    top = _run_git(repo, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    try:
        same = Path(top.decode().strip()).resolve() == Path(repo).resolve()
    except (OSError, UnicodeError):
        return None
    if not same:
        return None
    out = _run_git(repo, "rev-parse", "--verify", "--quiet", "HEAD^{commit}")
    if out is None:
        return None
    sha = out.decode("ascii", errors="replace").strip()
    return sha if _SHA_RE.fullmatch(sha) else None


class UnreadableRulespecFile(LookupError):
    """A file the view lists but whose content cannot be read."""


class RulespecView:
    """Read-only file access to one RuleSpec country checkout state.

    ``read`` returns ``None`` for a file the view does not have and raises
    :class:`UnreadableRulespecFile` for one it has but cannot read.
    """

    def identity(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def has_file(self, relative: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self, relative: str) -> bytes | None:  # pragma: no cover - interface
        raise NotImplementedError

    def first_yaml_under(self, relative_dir: str) -> str | None:  # pragma: no cover
        raise NotImplementedError


class GitRefRulespecView(RulespecView):
    """A RuleSpec repository at one exact commit, read from local git objects."""

    def __init__(self, repo: Path, ref: str) -> None:
        self.repo = Path(repo)
        if _SAFE_REF_RE.fullmatch(ref) is None or ref.startswith("-"):
            raise ValueError(f"unsafe RuleSpec ref: {ref!r}")
        out = _run_git(
            self.repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"
        )
        sha = out.decode("ascii", errors="replace").strip() if out else ""
        if not _SHA_RE.fullmatch(sha):
            raise LookupError(f"RuleSpec ref {ref} is not in the local repository")
        self.commit = sha
        listing = _run_git(self.repo, "ls-tree", "-r", "-z", "--full-tree", sha)
        if listing is None:
            raise LookupError(f"cannot list RuleSpec ref {sha}")
        regular: set[str] = set()
        special: set[str] = set()
        for entry in listing.split(b"\0"):
            if not entry:
                continue
            meta, _tab, raw_name = entry.partition(b"\t")
            name = raw_name.decode("utf-8", errors="surrogateescape")
            mode = meta.split(b" ", 1)[0]
            # A tracked symlink or submodule is not a regular module file.
            (regular if mode in {b"100644", b"100755"} else special).add(name)
        self._files = frozenset(regular)
        self._special = frozenset(special)
        self._dirs: dict[str, str] = {}
        # Regular modules first, so a readable descendant is preferred; a
        # directory with only a linked module still names it (unreadable).
        for name in [*sorted(self._files), *sorted(self._special)]:
            if not name.endswith(".yaml"):
                continue
            parent = PurePosixPath(name).parent
            while str(parent) not in {"", "."}:
                self._dirs.setdefault(str(parent), name)
                parent = parent.parent

    def identity(self) -> dict[str, Any]:
        return {"kind": "git_ref", "commit": self.commit}

    def has_file(self, relative: str) -> bool:
        return relative in self._files

    def read(self, relative: str) -> bytes | None:
        if relative in self._special:
            raise UnreadableRulespecFile(relative)
        if relative not in self._files:
            return None
        size = _run_git(self.repo, "cat-file", "-s", f"{self.commit}:{relative}")
        try:
            too_large = size is None or int(size.strip()) > MAX_RULESPEC_FILE_BYTES
        except ValueError:
            too_large = True
        data = (
            None
            if too_large
            else _run_git(self.repo, "cat-file", "blob", f"{self.commit}:{relative}")
        )
        if data is None:
            raise UnreadableRulespecFile(relative)
        return data

    def first_yaml_under(self, relative_dir: str) -> str | None:
        return self._dirs.get(relative_dir.strip("/"))


class CheckoutRulespecView(RulespecView):
    """A RuleSpec country checkout directory as it is on disk."""

    def __init__(self, root: Path) -> None:
        raw = Path(os.path.abspath(root))
        if raw.is_symlink() or not raw.is_dir():
            raise ValueError(f"RuleSpec checkout is not a directory: {raw}")
        self.root = raw
        self.head = git_head(raw)

    def identity(self) -> dict[str, Any]:
        return {"kind": "checkout", "head": self.head}

    def _path(self, relative: str) -> Path | None:
        parts = PurePosixPath(relative).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            return None
        return self.root.joinpath(*parts)

    def has_file(self, relative: str) -> bool:
        path = self._path(relative)
        return path is not None and path.is_file() and not path.is_symlink()

    def read(self, relative: str) -> bytes | None:
        path = self._path(relative)
        if path is None:
            return None
        if path.is_symlink() or (path.exists() and not path.is_file()):
            # Present, but not a regular file the resolver would accept.
            raise UnreadableRulespecFile(relative)
        if not self.has_file(relative):
            return None
        try:
            return read_bounded_regular_file(
                self.root,
                path,
                label="RuleSpec module",
                max_bytes=MAX_RULESPEC_FILE_BYTES,
            )
        except (OSError, CorpusResolutionError) as exc:
            raise UnreadableRulespecFile(relative) from exc

    def first_yaml_under(self, relative_dir: str) -> str | None:
        path = self._path(relative_dir)
        if path is None or path.is_symlink() or not path.is_dir():
            return None
        candidates = sorted(path.rglob("*.yaml"))
        for candidate in candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate.relative_to(self.root).as_posix()
        # Only linked or special entries: name one so it reads as unreadable.
        for candidate in candidates:
            return candidate.relative_to(self.root).as_posix()
        return None


class RulespecSnapshot:
    """A private, sparse copy of a RuleSpec view laid out as a checkout.

    Files are copied only on request, so the encoder's resolvers see exactly
    the files that exist in the view and nothing else.
    """

    def __init__(self, view: RulespecView, country: str) -> None:
        self.view = view
        self._tmp = tempfile.TemporaryDirectory(prefix="axiom-triage-rulespec-")
        self.checkout_root = Path(self._tmp.name).resolve() / monorepo_checkout_name(
            country
        )
        self.checkout_root.mkdir()
        self._done: set[str] = set()
        # Files the view lists but could not hand over: a probe must not
        # report them as absent.
        self.unreadable: set[str] = set()

    def close(self) -> None:
        self._tmp.cleanup()

    def __enter__(self) -> RulespecSnapshot:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def content_root(self, prefix: str) -> Path:
        root = self.checkout_root / prefix
        root.mkdir(parents=True, exist_ok=True)
        return root

    def relative(self, path: Path) -> str | None:
        try:
            return (
                Path(os.path.abspath(path)).relative_to(self.checkout_root).as_posix()
            )
        except ValueError:
            return None

    def materialize(self, relative: str) -> bool:
        """Copy one file from the view; return whether it exists there."""

        if relative in self._done:
            return (self.checkout_root / relative).is_file()
        self._done.add(relative)
        try:
            data = self.view.read(relative)
        except UnreadableRulespecFile:
            self.unreadable.add(relative)
            return False
        if data is None:
            return False
        target = self.checkout_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return True

    def materialize_path(self, path: Path) -> bool:
        relative = self.relative(path)
        return relative is not None and self.materialize(relative)

    def materialize_dir_marker(self, relative_dir: str) -> bool:
        """Copy one YAML file under a directory, when the view has one."""

        first = self.view.first_yaml_under(relative_dir)
        return first is not None and self.materialize(first)

    def remove_tree(self) -> None:
        shutil.rmtree(self.checkout_root, ignore_errors=True)


@dataclass(frozen=True)
class CorpusRowInfo:
    version: str
    provision_file: str
    status: str | None
    has_body: bool
    heading: str | None
    citation_label: str | None
    title: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "provision_file": self.provision_file,
            "status": self.status,
            "has_body": self.has_body,
            "heading": self.heading,
            "citation_label": self.citation_label,
            "title": self.title,
        }


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


class LocalCorpusIndex:
    """Unverified index of the provision rows in one local corpus checkout.

    This is not a signed release: rows are read from the checkout's
    ``data/corpus/provisions/<jurisdiction>/<document class>/*.jsonl`` files
    as they are on disk. Probe findings that use it say so.
    """

    MAX_BUCKETS = 6

    def __init__(self, root: Path) -> None:
        raw = Path(os.path.abspath(root))
        provisions = raw / "data" / "corpus" / "provisions"
        for path in (raw, raw / "data", raw / "data" / "corpus", provisions):
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"not a local corpus checkout: {raw}")
        self.root = raw
        self.provisions = provisions
        self.head = git_head(raw)
        self._buckets: OrderedDict[tuple[str, str], dict[str, list[CorpusRowInfo]]]
        self._buckets = OrderedDict()
        # Buckets with a file or a row that could not be read or parsed: an
        # absent citation there is unknown, not absent.
        self._incomplete: set[tuple[str, str]] = set()

    def identity(self) -> dict[str, Any]:
        return {"kind": "local_checkout_unverified", "head": self.head}

    def _bucket_dir(self, jurisdiction: str, document_class: str) -> Path | None:
        for segment in (jurisdiction, document_class):
            if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", segment) is None:
                return None
        parent = self.provisions / jurisdiction
        directory = parent / document_class
        if any(
            path.is_symlink() or (path.exists() and not path.is_dir())
            for path in (parent, directory)
        ):
            # Present but not inspectable: absence there is unknown.
            self._incomplete.add((jurisdiction, document_class))
            return None
        if not directory.is_dir():
            return None
        return directory

    def version_present(
        self, jurisdiction: str, document_class: str, version: str
    ) -> bool:
        directory = self._bucket_dir(jurisdiction, document_class)
        if (
            directory is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version) is None
        ):
            return False
        path = directory / f"{version}.jsonl"
        return path.is_file() and not path.is_symlink()

    def _load(
        self, jurisdiction: str, document_class: str
    ) -> dict[str, list[CorpusRowInfo]]:
        key = (jurisdiction, document_class)
        if key in self._buckets:
            self._buckets.move_to_end(key)
            return self._buckets[key]
        rows: dict[str, list[CorpusRowInfo]] = {}
        directory = self._bucket_dir(jurisdiction, document_class)
        files = sorted(directory.glob("*.jsonl")) if directory is not None else []
        for path in files:
            if path.is_symlink() or not path.is_file():
                self._incomplete.add(key)
                continue
            try:
                raw = read_bounded_regular_file(
                    self.root,
                    path,
                    label="corpus provision file",
                    max_bytes=MAX_CORPUS_PROVISION_BYTES,
                )
            except (OSError, CorpusResolutionError, UnsafeCorpusPathError):
                self._incomplete.add(key)
                continue
            relative = path.relative_to(self.root).as_posix()
            version = path.name.removesuffix(".jsonl")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._incomplete.add(key)
                text = raw.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, RecursionError):
                    self._incomplete.add(key)
                    continue
                if not isinstance(record, dict):
                    self._incomplete.add(key)
                    continue
                citation = record.get("citation_path")
                if not isinstance(citation, str):
                    self._incomplete.add(key)
                    continue
                metadata = record.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                status = metadata.get("status")
                body = record.get("body")
                rows.setdefault(citation, []).append(
                    CorpusRowInfo(
                        version=str(record.get("version") or version),
                        provision_file=relative,
                        status=status if isinstance(status, str) else None,
                        has_body=isinstance(body, str) and bool(body.strip()),
                        heading=_optional_text(record.get("heading")),
                        citation_label=_optional_text(record.get("citation_label")),
                        title=_optional_text(metadata.get("title")),
                    )
                )
        self._buckets[key] = rows
        while len(self._buckets) > self.MAX_BUCKETS:
            self._buckets.popitem(last=False)
        return rows

    def rows(self, citation_path: str) -> tuple[CorpusRowInfo, ...]:
        try:
            canonical = require_canonical_corpus_citation_path(citation_path)
        except (CorpusResolutionError, ValueError):
            return ()
        jurisdiction, document_class = canonical.split("/")[:2]
        found = self._load(jurisdiction, document_class).get(canonical, [])
        return tuple(sorted(found, key=lambda row: (row.version, row.provision_file)))

    def complete_for(self, citation_path: str) -> bool:
        """Whether every row of the citation's bucket was read and parsed."""

        try:
            canonical = require_canonical_corpus_citation_path(citation_path)
        except (CorpusResolutionError, ValueError):
            return False
        jurisdiction, document_class = canonical.split("/")[:2]
        self._load(jurisdiction, document_class)
        return (jurisdiction, document_class) not in self._incomplete

    def iter_loaded(self) -> Iterator[tuple[str, str]]:  # pragma: no cover - debug
        yield from self._buckets
