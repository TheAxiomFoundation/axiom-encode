"""Admission of retired apply signatures against supervisor-owned frozen trust.

This module grants no signing authority. Its inventory argument must originate
from the authenticated signing broker, never from a command-line manifest or a
repository file. Direct admission binds live files and their complete history;
nested admission proves only that an exact signed payload was frozen evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RETIRED_APPLY_INVENTORY_SCHEMA = "axiom-encode/retired-apply-inventory/v1"
_REPOSITORY = re.compile(r"TheAxiomFoundation/rulespec-[a-z0-9]+(?:-[a-z0-9]+)*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _inventory_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(
            ord(character) < 32
            or 127 <= ord(character) <= 159
            or 0xD800 <= ord(character) <= 0xDFFF
            for character in value
        )
        or value.startswith(("/", "-", ":"))
        or any(character in value for character in "\\*?[")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("retired apply inventory path must be canonical UTF-8 POSIX")
    return value


def _exact_fields(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"retired apply inventory {label} has unsupported fields")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            "retired apply inventory sha256 must be 64 lowercase hex digits"
        )
    return value


def validate_retired_apply_inventory(value: object) -> dict[str, object]:
    """Validate and copy the exact inventory schema shared by trust consumers.

    Digests bind raw bytes, including formatting and signatures. No timestamp,
    branch name, ambient environment variable, or repository file is authority.
    """

    root = _exact_fields(value, {"schema", "repositories"}, "root")
    if root["schema"] != RETIRED_APPLY_INVENTORY_SCHEMA:
        raise ValueError("retired apply inventory schema is unsupported")
    repositories = root["repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ValueError("retired apply inventory repositories must be nonempty")
    canonical_repositories: list[dict[str, object]] = []
    identities: set[str] = set()
    for raw_repository in repositories:
        repository = _exact_fields(
            raw_repository, {"repository", "cutoff_commit", "artifacts"}, "repository"
        )
        identity = repository["repository"]
        if not isinstance(identity, str) or _REPOSITORY.fullmatch(identity) is None:
            raise ValueError("retired apply inventory repository must be canonical")
        if identity in identities:
            raise ValueError("retired apply inventory has a duplicate repository")
        identities.add(identity)
        cutoff = repository["cutoff_commit"]
        if not isinstance(cutoff, str) or _COMMIT.fullmatch(cutoff) is None:
            raise ValueError("retired apply inventory cutoff must be a full commit SHA")
        artifacts = repository["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("retired apply inventory artifacts must be nonempty")
        canonical_artifacts: list[dict[str, object]] = []
        artifact_paths: set[str] = set()
        for raw_artifact in artifacts:
            artifact = _exact_fields(
                raw_artifact, {"path", "sha256", "covered_files"}, "artifact"
            )
            path = _inventory_path(artifact["path"])
            if path in artifact_paths:
                raise ValueError(
                    "retired apply inventory has a duplicate artifact path"
                )
            artifact_paths.add(path)
            covered_files = artifact["covered_files"]
            if not isinstance(covered_files, list):
                raise ValueError("retired apply inventory covered_files must be a list")
            canonical_covered: list[dict[str, object]] = []
            covered_paths: set[tuple[str, str]] = set()
            for raw_covered in covered_files:
                if not isinstance(raw_covered, dict) or set(raw_covered) not in (
                    {"path", "sha256"},
                    {"path", "sha256", "source_commit"},
                ):
                    raise ValueError(
                        "retired apply inventory covered file has unsupported fields"
                    )
                covered = raw_covered
                covered_path = _inventory_path(covered["path"])
                source = covered.get("source_commit", cutoff)
                if not isinstance(source, str) or _COMMIT.fullmatch(source) is None:
                    raise ValueError(
                        "retired apply covered source must be a full commit SHA"
                    )
                if covered_path == path or (covered_path, source) in covered_paths:
                    raise ValueError(
                        "retired apply inventory has duplicate covered paths"
                    )
                covered_paths.add((covered_path, source))
                copied: dict[str, object] = {
                    "path": covered_path,
                    "sha256": _digest(covered["sha256"]),
                }
                if "source_commit" in covered:
                    copied["source_commit"] = source
                canonical_covered.append(copied)
            canonical_artifacts.append(
                {
                    "path": path,
                    "sha256": _digest(artifact["sha256"]),
                    "covered_files": canonical_covered,
                }
            )
        canonical_repositories.append(
            {
                "repository": identity,
                "cutoff_commit": cutoff,
                "artifacts": canonical_artifacts,
            }
        )
    return {
        "schema": RETIRED_APPLY_INVENTORY_SCHEMA,
        "repositories": canonical_repositories,
    }


def _frozen_mapping_data(value: Mapping[str, Any]) -> tuple:
    """Build an immutable open-addressed hash table, including all nested data."""

    def freeze(item: Any) -> tuple:
        if isinstance(item, dict):
            return ("mapping", _frozen_mapping_data(item))
        if isinstance(item, list):
            return ("sequence", tuple(freeze(child) for child in item))
        return ("scalar", item)

    entries = tuple((key, freeze(item)) for key, item in value.items())
    capacity = 2
    while capacity < 2 * len(entries):
        capacity *= 2
    slots = [None] * capacity
    for number, (key, _) in enumerate(entries):
        slot = hash(key) & (capacity - 1)
        while slots[slot] is not None:
            slot = (slot + 1) & (capacity - 1)
        slots[slot] = number
    return entries, tuple(slots)


def _frozen_value_view(value: tuple) -> Any:
    kind, data = value
    if kind == "mapping":
        return tuple.__new__(_FrozenMapping, data)
    if kind == "sequence":
        return tuple(_frozen_value_view(item) for item in data)
    return data


class _FrozenMapping(tuple, Mapping[str, Any]):
    """Mapping views over tuple-only storage, with expected O(1) lookup.

    MappingProxyType cannot be used here: reflected comparison/union methods
    can receive its mutable backing dict. Nested views are created on access;
    only native tuples and scalar values are retained in the authority.
    """

    __slots__ = ()
    __hash__ = None
    __contains__ = Mapping.__contains__
    __eq__ = Mapping.__eq__

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __getitem__(self, key: str) -> Any:
        entries, slots = tuple.__iter__(self)
        slot = hash(key) & (len(slots) - 1)
        while (number := slots[slot]) is not None:
            candidate, value = entries[number]
            if candidate == key:
                return _frozen_value_view(value)
            slot = (slot + 1) & (len(slots) - 1)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in tuple.__getitem__(self, 0))

    def __len__(self) -> int:
        return len(tuple.__getitem__(self, 0))

    def __setstate__(self, state: object) -> None:
        raise TypeError("retired apply inventory cannot restore state")

    def __reduce__(self) -> Any:
        raise TypeError("retired apply inventory cannot be pickled")

    def __reduce_ex__(self, protocol: int) -> Any:
        return self.__reduce__()

    def __copy__(self) -> _FrozenMapping:
        return self

    def __deepcopy__(self, memo: dict) -> _FrozenMapping:
        return self


class RetiredApplyInventory(_FrozenMapping):
    """Detached repository/path index with immutable tuple-only storage.

    Only broker attachment constructs this view in production. The wire-schema
    validator remains shared with the trust-file consumers; admission never
    revalidates an instance of this class.
    """

    __slots__ = ()

    def __new__(cls, value: object) -> RetiredApplyInventory:
        validated = validate_retired_apply_inventory(value)
        repositories = {}
        for repository in validated["repositories"]:
            repositories[repository["repository"]] = {
                **repository,
                "artifacts": {
                    artifact["path"]: artifact for artifact in repository["artifacts"]
                },
            }
        return tuple.__new__(cls, _frozen_mapping_data(repositories))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("retired apply inventory cannot be subclassed")


@dataclass
class _RetiredApplyOperation:
    # Raw inputs are supported for standalone readers/tests. Hold the input as
    # well as its id, so Python cannot recycle a key during this operation.
    inventories: dict[int, tuple[object, RetiredApplyInventory]] = field(
        default_factory=dict
    )
    cutoffs: dict[
        tuple[int, Path, str, bool],
        dict[str, tuple[Path, str, str, Mapping[str, Any]]],
    ] = field(default_factory=dict)
    sources: set[tuple[Path, str, str]] = field(default_factory=set)


_operation: ContextVar[_RetiredApplyOperation | None] = ContextVar(
    "retired_apply_operation", default=None
)


@contextmanager
def retired_apply_operation() -> Iterator[None]:
    """Scope frozen trust/history snapshots to exactly one guard invocation.

    Nested operations get independent caches and restore the caller's scope.
    Successful repository checks are reused only at the same checkout and
    resolved HEAD. Live bytes and path history are always checked per artifact.
    No cache survives return or exception from the operation.
    """

    token = _operation.set(_RetiredApplyOperation())
    try:
        yield
    finally:
        _operation.reset(token)


def _inventory_index(inventory: object) -> RetiredApplyInventory:
    operation = _operation.get()
    if type(inventory) is RetiredApplyInventory:
        if operation is not None:
            operation.inventories.setdefault(id(inventory), (inventory, inventory))
        return inventory
    # isinstance honors a reassigned __class__, so a broker view re-classed to
    # pose as a dict could otherwise be revalidated as a replacement inventory.
    if type(inventory) is not dict:
        raise ValueError("retired apply inventory must be a broker index or raw dict")
    if operation is None:
        return RetiredApplyInventory(inventory)
    key = id(inventory)
    if key not in operation.inventories:
        operation.inventories[key] = (inventory, RetiredApplyInventory(inventory))
    indexed = operation.inventories[key][1]
    if type(indexed) is not RetiredApplyInventory:
        raise ValueError("retired apply inventory cached index type changed")
    return indexed


@dataclass(frozen=True)
class RetiredApplyContext:
    """Explicit artifact context; the frozen inventory is supplied separately."""

    repo_path: Path
    artifact_path: str | Path
    artifact_bytes: bytes
    head_ref: str = "HEAD"
    retrospective: bool = False


def _git_environment() -> dict[str, str]:
    # Use the broker's immutable subprocess environment when supervised. Avoid
    # importing the large eval module from the trust-file schema validator.
    from .signing_broker import scrub_private_signing_environment

    environment = {
        key: value
        for key, value in scrub_private_signing_environment().items()
        if not key.startswith("GIT_")
    }
    environment.update(
        GIT_GRAFT_FILE=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="core.commitGraph",
        GIT_CONFIG_VALUE_0="false",
        GIT_TERMINAL_PROMPT="0",
        GIT_LITERAL_PATHSPECS="1",
        GIT_OPTIONAL_LOCKS="0",
    )
    return environment


def _git(repo: Path, *arguments: str, ancestry: bool = False) -> bytes | bool:
    # Keep -C first, matching the provisioned supervisor's trusted Git wrapper.
    environment = _git_environment()
    executable = shutil.which("git", path=environment.get("PATH") or None)
    if executable is None:
        raise ValueError(
            "retired apply verification requires the trusted Git executable"
        )
    result = subprocess.run(
        [executable, "-C", str(repo), *arguments],
        capture_output=True,
        env=environment,
        timeout=30,
        check=False,
    )
    if ancestry and result.returncode in {0, 1}:
        return result.returncode == 0
    if result.returncode != 0:
        raise ValueError(f"retired apply history query failed: git {arguments[0]}")
    return result.stdout


def _git_text(repo: Path, *arguments: str) -> str:
    result = _git(repo, *arguments)
    assert isinstance(result, bytes)
    return result.decode("utf-8").strip()


def _regular_path(repo: Path, relative: str) -> Path:
    path = repo
    for component in relative.split("/"):
        path /= component
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("retired apply artifact paths must not contain symlinks")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("retired apply artifact must be a regular file")
    return path


def _live_bytes(repo: Path, relative: str) -> bytes:
    # Keep the existing path diagnostics, then atomically anchor the actual
    # read so a concurrent symlink swap cannot escape the checkout.
    from .corpus_resolver import read_bounded_regular_file

    path = _regular_path(repo, relative)
    return read_bounded_regular_file(
        repo,
        path,
        label="retired apply live artifact",
        max_bytes=16 * 1024 * 1024,
    )


def _blob(repo: Path, commit: str, path: str) -> bytes:
    tree = _git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path)
    assert isinstance(tree, bytes)
    entries = tree.split(b"\0")
    if len(entries) != 2 or entries[1] or b"\t" not in entries[0]:
        raise ValueError("retired apply frozen artifact is absent or ambiguous")
    metadata, listed_path = entries[0].split(b"\t", 1)
    mode, kind, oid = metadata.split()
    if (
        mode != b"100644"
        or kind != b"blob"
        or listed_path.decode("utf-8") != path
        or _COMMIT.fullmatch(oid.decode("ascii")) is None
    ):
        raise ValueError("retired apply frozen artifact must be a regular 0644 blob")
    result = _git(repo, "cat-file", "blob", oid.decode("ascii"))
    assert isinstance(result, bytes)
    return result


def _repository_identity(origin: str) -> str | None:
    patterns = (
        r"https://github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?",
        r"git@github\.com:(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?",
        r"ssh://git@github\.com/(?P<slug>[^/\s]+/[^/\s]+?)(?:\.git)?/?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, origin)
        if match is not None:
            identity = match.group("slug")
            return identity if _REPOSITORY.fullmatch(identity) else None
    return None


def _repository_cutoff(
    inventory: object,
    repo_path: Path,
    head_ref: str,
    retrospective: bool,
) -> tuple[Path, str, str, Mapping[str, Any]]:
    trusted = _inventory_index(inventory)
    checkout = Path(os.path.abspath(repo_path))
    if checkout.resolve(strict=True) != checkout:
        raise ValueError(
            "retired apply checkout must be canonical and contain no symlink"
        )
    operation = _operation.get()
    key = (id(trusted), checkout, head_ref, retrospective)
    cached = operation.cutoffs.get(key) if operation is not None else None
    if cached:
        # HEAD is mutable even within a guard. Never key a snapshot by the
        # symbolic string "HEAD", or reuse a different checkout's preflight.
        head = _git_text(checkout, "rev-parse", "--verify", "HEAD^{commit}")
        if head in cached:
            return cached[head]
    repo = Path(_git_text(checkout, "rev-parse", "--show-toplevel"))
    if repo.resolve(strict=True) != repo:
        raise ValueError("retired apply Git root must be canonical")
    identity = _repository_identity(_git_text(repo, "remote", "get-url", "origin"))
    if identity is None or repo.name != identity.split("/", 1)[1]:
        raise ValueError("retired apply repository identity is not canonical")
    repository = trusted.get(identity)
    if repository is None:
        raise ValueError("retired apply repository is not frozen in trusted inventory")
    if head_ref != "HEAD" and _COMMIT.fullmatch(head_ref) is None:
        raise ValueError(
            "retired apply historical head must be HEAD or a full commit SHA"
        )
    head = _git_text(repo, "rev-parse", "--verify", "HEAD^{commit}")
    if _COMMIT.fullmatch(head) is None or head_ref not in {"HEAD", head}:
        raise ValueError(
            "retired apply historical head must equal the checked-out HEAD"
        )
    cutoff = repository["cutoff_commit"]
    if _git_text(repo, "rev-parse", "--verify", f"{cutoff}^{{commit}}") != cutoff:
        raise ValueError("retired apply cutoff commit is unavailable")
    if _git_text(repo, "rev-parse", "--is-shallow-repository") != "false":
        raise ValueError("retired apply verification requires complete Git history")
    common = Path(
        _git_text(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    )
    grafts = common / "info/grafts"
    if grafts.exists() or grafts.is_symlink():
        raise ValueError("retired apply history contains forbidden graft metadata")
    if _git_text(repo, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        raise ValueError("retired apply history contains forbidden replacement refs")
    older = bool(_git(repo, "merge-base", "--is-ancestor", head, cutoff, ancestry=True))
    newer = bool(_git(repo, "merge-base", "--is-ancestor", cutoff, head, ancestry=True))
    if retrospective:
        if not older:
            raise ValueError(
                "retired apply retrospective head is newer than the frozen cutoff"
            )
    elif not newer:
        raise ValueError(
            "retired apply historical head must descend from the frozen cutoff"
        )
    result = repo, head, cutoff, repository
    if operation is not None:
        operation.cutoffs.setdefault(key, {})[head] = result
    return result


def retrospective_head_issue(
    inventory: object, repo_path: Path, head_ref: str = "HEAD"
) -> str | None:
    """Preflight an explicit retrospective guard even when no artifact is selected."""

    try:
        _repository_cutoff(inventory, repo_path, head_ref, True)
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)


def _frozen_artifact(
    inventory: object, context: RetiredApplyContext
) -> tuple[Path, str, str, Mapping[str, Any], bytes]:
    repo, head, cutoff, repository = _repository_cutoff(
        inventory, context.repo_path, context.head_ref, context.retrospective
    )
    absolute = Path(context.artifact_path)
    if not absolute.is_absolute():
        absolute = Path(os.path.abspath(context.repo_path)) / absolute
    # Preserve the lexical path, so relocated and symlinked files cannot turn
    # into a frozen artifact by resolving filesystem indirection.
    relative = _inventory_path(absolute.relative_to(repo).as_posix())
    artifact = repository["artifacts"].get(relative)
    if artifact is None:
        raise ValueError(
            "retired apply artifact path is not frozen in trusted inventory"
        )
    frozen_bytes = _blob(repo, cutoff, relative)
    if hashlib.sha256(frozen_bytes).hexdigest() != artifact["sha256"]:
        raise ValueError(
            "retired apply frozen artifact bytes disagree with trusted inventory"
        )
    for covered in artifact["covered_files"]:
        source = covered.get("source_commit", cutoff)
        operation = _operation.get()
        source_key = repo, source, cutoff
        if operation is None or source_key not in operation.sources:
            if (
                _git_text(repo, "rev-parse", "--verify", f"{source}^{{commit}}")
                != source
            ):
                raise ValueError("retired apply covered source commit is unavailable")
            if not _git(
                repo, "merge-base", "--is-ancestor", source, cutoff, ancestry=True
            ):
                raise ValueError(
                    "retired apply covered source must be an ancestor of the cutoff"
                )
            if operation is not None:
                operation.sources.add(source_key)
        frozen_covered = _blob(repo, source, covered["path"])
        if hashlib.sha256(frozen_covered).hexdigest() != covered["sha256"]:
            raise ValueError(
                "retired apply frozen covered-file bytes disagree with trusted inventory"
            )
    frozen = json.loads(frozen_bytes)
    prefix = relative.split(".axiom/", 1)[0] if ".axiom/" in relative else ""
    frozen_coverage = {
        (entry["path"], entry["sha256"]) for entry in artifact["covered_files"]
    }
    for path, digest in _payload_file_hashes(frozen, prefix):
        if (path, digest) not in frozen_coverage:
            raise ValueError(
                "retired apply inventory does not bind every generated-file byte hash"
            )
    return repo, head, cutoff, artifact, frozen_bytes


def _payload_file_hashes(value: object, prefix: str) -> list[tuple[str, str]]:
    covered: list[tuple[str, str]] = []
    if isinstance(value, dict):
        applied = value.get("applied_files")
        if isinstance(applied, list):
            for entry in applied:
                if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
                    covered.append(
                        (
                            _inventory_path(
                                prefix + _inventory_path(entry.get("path"))
                            ),
                            _digest(entry["sha256"]),
                        )
                    )
        for child in value.values():
            covered.extend(_payload_file_hashes(child, prefix))
    elif isinstance(value, list):
        for child in value:
            covered.extend(_payload_file_hashes(child, prefix))
    return covered


def retired_apply_admission_issue(
    inventory: object, context: RetiredApplyContext
) -> str | None:
    """Require an unchanged frozen live artifact, including no restored history."""

    try:
        repo, head, cutoff, artifact, frozen_bytes = _frozen_artifact(
            inventory, context
        )
        if (
            not isinstance(context.artifact_bytes, bytes)
            or context.artifact_bytes != frozen_bytes
        ):
            raise ValueError(
                "retired apply manifest/receipt bytes are not exactly frozen"
            )
        if _live_bytes(repo, artifact["path"]) != frozen_bytes:
            raise ValueError(
                "retired apply live manifest/receipt bytes are not exactly frozen"
            )
        live_covered = [
            entry
            for entry in artifact["covered_files"]
            if entry.get("source_commit", cutoff) == cutoff
        ]
        frozen = json.loads(frozen_bytes)
        prefix = (
            artifact["path"].split(".axiom/", 1)[0]
            if ".axiom/" in artifact["path"]
            else ""
        )
        if isinstance(frozen, dict) and isinstance(frozen.get("applied_files"), list):
            root_live = {
                "applied_files": [
                    entry
                    for entry in frozen["applied_files"]
                    if isinstance(entry, dict) and entry.get("deleted") is not True
                ]
            }
            live_hashes = {(entry["path"], entry["sha256"]) for entry in live_covered}
            if any(
                entry not in live_hashes
                for entry in _payload_file_hashes(root_live, prefix)
            ):
                raise ValueError(
                    "retired apply inventory live coverage must bind cutoff files"
                )
        files = [
            {"path": artifact["path"], "sha256": artifact["sha256"]},
            *live_covered,
        ]
        for covered in files:
            if (
                hashlib.sha256(_blob(repo, head, covered["path"])).hexdigest()
                != covered["sha256"]
            ):
                raise ValueError("retired apply HEAD file bytes are not exactly frozen")
            if (
                hashlib.sha256(_live_bytes(repo, covered["path"])).hexdigest()
                != covered["sha256"]
            ):
                raise ValueError(
                    "retired apply live covered-file bytes are not exactly frozen"
                )
        if head != cutoff and not context.retrospective:
            history = _git_text(
                repo,
                "log",
                "--full-history",
                "-m",
                "--format=%H",
                "--no-renames",
                f"{cutoff}..{head}",
                "--",
                *(entry["path"] for entry in files),
            )
            if history:
                raise ValueError(
                    "retired apply artifact was changed or superseded after the frozen cutoff"
                )
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)


def _contains_payload(value: object, payload: object) -> bool:
    if isinstance(value, dict):
        return value == payload or any(
            _contains_payload(child, payload) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_payload(child, payload) for child in value)
    return False


def retired_apply_payload_admission_issue(
    inventory: object,
    context: RetiredApplyContext,
    payload: object,
    *,
    nested: bool = False,
) -> str | None:
    """Admit direct signatures or exact nested frozen historical evidence.

    Nested success authenticates evidence only. A new outer artifact still
    requires the active root, and its existing receipt chain must bind the
    source path and original manifest hash before it invokes this function.
    """

    try:
        if not nested:
            if (
                not isinstance(payload, dict)
                or json.loads(context.artifact_bytes) != payload
            ):
                raise ValueError(
                    "retired apply payload does not match its exact manifest/receipt bytes"
                )
            return retired_apply_admission_issue(inventory, context)
        _, _, _, _, frozen_bytes = _frozen_artifact(inventory, context)
        frozen = json.loads(frozen_bytes)
        if not isinstance(payload, dict) or not _contains_payload(frozen, payload):
            raise ValueError(
                "retired apply nested payload is not exact frozen evidence at this path"
            )
        return None
    except (OSError, ValueError, subprocess.SubprocessError, UnicodeError) as exc:
        return str(exc)
