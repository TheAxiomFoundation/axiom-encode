"""Sealed identity for the one admitted PolicyEngine oracle runtime.

PolicyEngine output is trust-bearing oracle evidence.  Admission therefore
binds a canonical RuleSpec checkout to a committed runtime pin, proves that pin
against a literal official HTTPS repository in a fresh Git object database,
and executes only a root-owned self-contained Python closure.  The caller's
Git configuration, remotes, interpreter, environment, and import paths are not
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from axiom_encode.repo_routing import canonical_rulespec_root_identity

POLICYENGINE_RUNTIME_SCHEMA = "axiom-policyengine-runtime/v2"
POLICYENGINE_RUNTIME_PIN_SCHEMA = "axiom-policyengine-runtime-pin/v1"
POLICYENGINE_RUNTIME_PIN_PATH = Path(".axiom/policyengine-runtime.toml")

_COUNTRIES = frozenset({"us", "uk"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PYTHON_LIBRARY_RE = re.compile(r"^python\d+\.\d+$")
_MAX_IDENTITY_FILE_BYTES = 512 * 1024 * 1024
_MAX_TREE_FILES = 250_000
_MAX_TREE_BYTES = 8 * 1024 * 1024 * 1024
_MAX_PIN_BYTES = 16 * 1024
_PROBE_PREFIX = "AXIOM_POLICYENGINE_RUNTIME:"
_ORACLE_TIMEOUT_SECONDS = 300
_GIT_TIMEOUT_SECONDS = 300
_GIT = Path("/usr/bin/git")


class PolicyEngineRuntimeError(ValueError):
    """An explicit PolicyEngine runtime failed sealed admission."""


@dataclass(frozen=True, slots=True)
class _RuntimePin:
    country: str
    commit: str
    path: Path
    sha256: str
    rulespec_checkout_root: Path


@dataclass(frozen=True, slots=True)
class _OfficialFile:
    path: str
    sha256: str
    size: int
    executable: bool


@dataclass(frozen=True, slots=True)
class _OfficialTree:
    commit: str
    files: tuple[_OfficialFile, ...]
    content_sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class _TreeIdentity:
    sha256: str
    file_count: int
    directory_count: int
    byte_count: int
    files: tuple[_OfficialFile, ...]
    directories: tuple[str, ...]


def rulespec_country_from_root(policy_repo_root: Path) -> str:
    """Derive country only from an exact canonical jurisdiction root."""

    identity = canonical_rulespec_root_identity(Path(policy_repo_root))
    if identity is None:
        raise PolicyEngineRuntimeError(
            "PolicyEngine oracle requires an exact canonical active RuleSpec "
            "jurisdiction root (rulespec-<country>/<jurisdiction>)"
        )
    checkout_name, _jurisdiction = identity.split("/", 1)
    country = checkout_name.removeprefix("rulespec-")
    if country not in _COUNTRIES:
        raise PolicyEngineRuntimeError(
            f"PolicyEngine oracle does not support RuleSpec country '{country}'"
        )
    return country


def _rulespec_checkout_root(policy_repo_root: Path) -> Path:
    jurisdiction_root = Path(policy_repo_root)
    country = rulespec_country_from_root(jurisdiction_root)
    try:
        resolved = jurisdiction_root.resolve(strict=True)
    except OSError as exc:
        raise PolicyEngineRuntimeError(
            f"Canonical RuleSpec jurisdiction root does not exist: {jurisdiction_root}"
        ) from exc
    checkout = resolved.parent
    if checkout.name != f"rulespec-{country}":
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime pin must belong to the canonical RuleSpec "
            f"country checkout: {checkout}"
        )
    return checkout


def _absolute_unaliased_directory(raw_root: Path) -> Path:
    root = Path(raw_root)
    if not root.is_absolute():
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime root must be an explicit absolute path"
        )
    lexical = Path(os.path.abspath(root))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime root must not contain symlink components: {cursor}"
            )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime root does not exist: {lexical}"
        ) from exc
    if not resolved.is_dir():
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime root is not a directory: {resolved}"
        )
    return resolved


def _git_environment() -> dict[str, str]:
    """Return a Git environment with no caller configuration or credentials."""

    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "XDG_CONFIG_HOME": "/nonexistent",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_CEILING_DIRECTORIES": "/",
    }


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = _GIT_TIMEOUT_SECONDS,
) -> bytes:
    if not _GIT.is_file():
        raise PolicyEngineRuntimeError(
            "Sealed PolicyEngine admission requires /usr/bin/git"
        )
    try:
        result = subprocess.run(
            [str(_GIT), *args],
            check=False,
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyEngineRuntimeError(
            f"Could not prove PolicyEngine Git identity: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", "replace")
        lines = detail.strip().splitlines()
        suffix = f": {lines[-1]}" if lines else ""
        raise PolicyEngineRuntimeError(
            f"PolicyEngine official Git proof failed ({' '.join(args)}){suffix}"
        )
    return result.stdout


def _official_https_remote(country: str) -> str:
    """Return the code-owned official repository URL; callers cannot override it."""

    if country == "us":
        return "https://github.com/PolicyEngine/policyengine-us.git"
    if country == "uk":
        return "https://github.com/PolicyEngine/policyengine-uk.git"
    raise PolicyEngineRuntimeError(f"Unsupported PolicyEngine country '{country}'")


def _read_regular_bytes(
    path: Path,
    *,
    label: str,
    max_bytes: int = _MAX_IDENTITY_FILE_BYTES,
) -> bytes:
    if path.is_symlink():
        raise PolicyEngineRuntimeError(f"{label} must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise PolicyEngineRuntimeError(f"Could not read {label}: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PolicyEngineRuntimeError(f"{label} must be a regular file: {path}")
    if info.st_size > max_bytes:
        raise PolicyEngineRuntimeError(f"{label} is unexpectedly large: {path}")
    flags = os.O_RDONLY
    if not hasattr(os, "O_NOFOLLOW"):
        raise PolicyEngineRuntimeError("Sealed runtime admission requires O_NOFOLLOW")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PolicyEngineRuntimeError(f"Could not open {label}: {path}") from exc
    try:
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > max_bytes:
                raise PolicyEngineRuntimeError(
                    f"{label} exceeds its admission bound: {path}"
                )
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise PolicyEngineRuntimeError(f"{label} changed while it was being read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise PolicyEngineRuntimeError(f"{label} changed while it was being read")
    return raw


def _committed_rulespec_pin_bytes(checkout: Path, pin_path: Path) -> bytes:
    """Read the pin from RuleSpec HEAD and require identical worktree bytes."""

    config_dir = checkout / ".axiom"
    if config_dir.is_symlink() or not config_dir.is_dir():
        raise PolicyEngineRuntimeError(
            "RuleSpec runtime pin must be inside a regular checkout-owned .axiom directory"
        )
    try:
        relative = pin_path.relative_to(checkout).as_posix()
    except ValueError as exc:
        raise PolicyEngineRuntimeError(
            "RuleSpec runtime pin escaped its canonical checkout"
        ) from exc
    committed = _run_git(["-C", str(checkout), "show", f"HEAD:{relative}"])
    if len(committed) > _MAX_PIN_BYTES:
        raise PolicyEngineRuntimeError("Committed RuleSpec runtime pin is too large")
    observed = _read_regular_bytes(
        pin_path,
        label="RuleSpec PolicyEngine runtime pin",
        max_bytes=_MAX_PIN_BYTES,
    )
    if observed != committed:
        raise PolicyEngineRuntimeError(
            "RuleSpec PolicyEngine runtime pin must be committed and unchanged at HEAD"
        )
    return observed


def _load_runtime_pin_from_checkout(checkout: Path, country: str) -> _RuntimePin:
    path = checkout / POLICYENGINE_RUNTIME_PIN_PATH
    raw = _committed_rulespec_pin_bytes(checkout, path)
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PolicyEngineRuntimeError(
            f"RuleSpec PolicyEngine runtime pin is not valid UTF-8 TOML: {path}"
        ) from exc
    if set(payload) != {"policyengine_runtime"} or not isinstance(
        payload["policyengine_runtime"], dict
    ):
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime pin must contain exactly one "
            "[policyengine_runtime] table"
        )
    contract = payload["policyengine_runtime"]
    if set(contract) != {"schema", "git_commit"}:
        raise PolicyEngineRuntimeError(
            "[policyengine_runtime] must contain exactly: git_commit, schema"
        )
    if contract.get("schema") != POLICYENGINE_RUNTIME_PIN_SCHEMA:
        raise PolicyEngineRuntimeError(
            "RuleSpec PolicyEngine runtime pin has an unsupported schema"
        )
    commit = contract.get("git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise PolicyEngineRuntimeError(
            "[policyengine_runtime].git_commit must be one lowercase 40-character SHA"
        )
    return _RuntimePin(
        country=country,
        commit=commit,
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        rulespec_checkout_root=checkout,
    )


def _load_runtime_pin(policy_repo_root: Path) -> _RuntimePin:
    country = rulespec_country_from_root(policy_repo_root)
    checkout = _rulespec_checkout_root(policy_repo_root)
    return _load_runtime_pin_from_checkout(checkout, country)


def _canonical_official_files(files: tuple[_OfficialFile, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    for item in sorted(files, key=lambda value: value.path):
        encoded = item.path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(b"\x01" if item.executable else b"\x00")
        digest.update(item.size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(item.sha256))
        byte_count += item.size
    return digest.hexdigest(), byte_count


def _official_tree_from_object_database(
    object_root: Path,
    commit: str,
) -> _OfficialTree:
    """Read every remote tree blob directly; archive attributes cannot hide files."""

    listing = _run_git(
        [
            "--git-dir",
            str(object_root),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit,
        ]
    )
    objects: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for record in listing.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split(" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine Git tree is malformed"
            ) from exc
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or ".." in pure.parts:
            raise PolicyEngineRuntimeError(
                f"Official PolicyEngine tree contains unsafe path {relative!r}"
            )
        if relative in seen:
            raise PolicyEngineRuntimeError(
                f"Official PolicyEngine tree repeats path {relative!r}"
            )
        seen.add(relative)
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine tree contains a symlink, submodule, or "
                f"special entry: {relative}"
            )
        if _GIT_COMMIT_RE.fullmatch(object_id) is None:
            raise PolicyEngineRuntimeError(
                f"Official PolicyEngine blob identity is malformed: {relative}"
            )
        objects.append((object_id, relative, mode == "100755"))
    if len(objects) > _MAX_TREE_FILES:
        raise PolicyEngineRuntimeError(
            "Official PolicyEngine tree exceeds file-count bounds"
        )
    batch_input = b"".join(
        object_id.encode("ascii") + b"\n" for object_id, _path, _mode in objects
    )
    if not _GIT.is_file():
        raise PolicyEngineRuntimeError(
            "Sealed PolicyEngine admission requires /usr/bin/git"
        )
    try:
        result = subprocess.run(
            [str(_GIT), "--git-dir", str(object_root), "cat-file", "--batch"],
            input=batch_input,
            check=False,
            capture_output=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyEngineRuntimeError(
            f"Could not read official PolicyEngine objects: {exc}"
        ) from exc
    if result.returncode != 0:
        raise PolicyEngineRuntimeError(
            "Could not read every official PolicyEngine Git object"
        )
    output = memoryview(result.stdout)
    cursor = 0
    files: list[_OfficialFile] = []
    byte_count = 0
    for expected_id, relative, executable in objects:
        newline = result.stdout.find(b"\n", cursor)
        if newline < 0:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine object batch ended unexpectedly"
            )
        try:
            observed_id, object_type, raw_size = (
                result.stdout[cursor:newline].decode("ascii").split(" ")
            )
            size = int(raw_size)
        except (UnicodeError, ValueError) as exc:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine object batch header is malformed"
            ) from exc
        cursor = newline + 1
        end = cursor + size
        if (
            observed_id != expected_id
            or object_type != "blob"
            or size > _MAX_IDENTITY_FILE_BYTES
            or end >= len(output)
            or output[end] != 0x0A
        ):
            raise PolicyEngineRuntimeError(
                f"Official PolicyEngine object is malformed: {relative}"
            )
        raw = output[cursor:end]
        cursor = end + 1
        byte_count += size
        if byte_count > _MAX_TREE_BYTES:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine tree exceeds byte-count bounds"
            )
        files.append(
            _OfficialFile(
                path=relative,
                sha256=hashlib.sha256(raw).hexdigest(),
                size=size,
                executable=executable,
            )
        )
    if cursor != len(output):
        raise PolicyEngineRuntimeError(
            "Official PolicyEngine object batch contains trailing data"
        )
    ordered = tuple(sorted(files, key=lambda value: value.path))
    digest, canonical_bytes = _canonical_official_files(ordered)
    if canonical_bytes != byte_count:
        raise PolicyEngineRuntimeError("Official PolicyEngine tree accounting failed")
    return _OfficialTree(
        commit=commit,
        files=ordered,
        content_sha256=digest,
        byte_count=byte_count,
    )


def _fetch_official_tree(country: str, commit: str) -> _OfficialTree:
    """Fetch exactly one pin from literal HTTPS into a fresh object namespace."""

    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise PolicyEngineRuntimeError("PolicyEngine trusted commit is malformed")
    remote = _official_https_remote(country)
    with tempfile.TemporaryDirectory(prefix="axiom-policyengine-proof-") as raw_temp:
        object_root = Path(raw_temp) / "objects.git"
        _run_git(["init", "--bare", "--quiet", str(object_root)])
        reference = "refs/axiom/runtime-pin"
        _run_git(
            [
                "--git-dir",
                str(object_root),
                "-c",
                "protocol.version=2",
                "fetch",
                "--force",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                remote,
                f"+{commit}:{reference}",
            ]
        )
        observed = (
            _run_git(
                [
                    "--git-dir",
                    str(object_root),
                    "rev-parse",
                    "--verify",
                    f"{reference}^{{commit}}",
                ]
            )
            .decode("ascii", "strict")
            .strip()
        )
        if observed != commit:
            raise PolicyEngineRuntimeError(
                "Official PolicyEngine fetch did not resolve the exact trusted commit"
            )
        official = _official_tree_from_object_database(object_root, commit)
    return official


def _trusted_owner_uids() -> frozenset[int]:
    """Return trusted filesystem owners (overridden only by isolated tests)."""

    return frozenset({0})


def _caller_can_write(path: Path) -> bool:
    try:
        return os.access(path, os.W_OK, effective_ids=True)
    except TypeError:  # pragma: no cover - unsupported Unix implementation
        return os.access(path, os.W_OK)


def _require_unprivileged_process() -> None:
    if not all(hasattr(os, name) for name in ("getuid", "geteuid")):
        raise PolicyEngineRuntimeError(
            "Sealed PolicyEngine runtime admission requires Unix credentials"
        )
    uid = os.getuid()
    euid = os.geteuid()
    if uid == 0 or euid == 0 or uid != euid:
        raise PolicyEngineRuntimeError(
            "PolicyEngine oracle supervisor must execute unprivileged with uid=euid"
        )


def _require_protected_metadata(
    path: Path, info: os.stat_result, *, label: str
) -> None:
    if info.st_uid not in _trusted_owner_uids():
        raise PolicyEngineRuntimeError(f"{label} must be root-owned: {path}")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PolicyEngineRuntimeError(
            f"{label} must not be group- or other-writable: {path}"
        )
    if info.st_mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
        raise PolicyEngineRuntimeError(
            f"{label} must not have set-id or sticky mode bits: {path}"
        )
    if _caller_can_write(path):
        raise PolicyEngineRuntimeError(
            f"{label} must not be caller- or ACL-writable: {path}"
        )


def _require_protected_ancestor_chain(root: Path) -> None:
    chain = [root, *root.parents]
    for path in reversed(chain):
        if path.is_symlink():
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime ancestor must not be a symlink: {path}"
            )
        try:
            info = path.stat()
        except OSError as exc:
            raise PolicyEngineRuntimeError(
                f"Could not inspect PolicyEngine runtime ancestor: {path}"
            ) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime ancestor is not a directory: {path}"
            )
        _require_protected_metadata(path, info, label="PolicyEngine runtime ancestor")


def _stable_regular_file(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    raw = _read_regular_bytes(path, label=label)
    info = path.stat()
    _require_protected_metadata(path, info, label=label)
    return raw, info


def _hash_protected_tree(
    root: Path,
    *,
    label: str,
    excluded_top_level: frozenset[str] = frozenset(),
) -> _TreeIdentity:
    """Parent-hash one complete immutable tree without following links."""

    try:
        root_info = root.stat()
    except OSError as exc:
        raise PolicyEngineRuntimeError(f"Could not inspect {label}: {root}") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
        raise PolicyEngineRuntimeError(f"{label} must be a real directory: {root}")
    _require_protected_metadata(root, root_info, label=label)

    digest = hashlib.sha256()
    file_records: list[_OfficialFile] = []
    directories: list[str] = []
    byte_count = 0

    def record(kind: bytes, relative: str, info: os.stat_result) -> None:
        encoded = relative.encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(stat.S_IMODE(info.st_mode).to_bytes(4, "big"))
        digest.update(info.st_uid.to_bytes(8, "big"))
        digest.update(info.st_gid.to_bytes(8, "big"))

    record(b"R", "", root_info)

    def walk(directory: Path, relative_root: PurePosixPath | None) -> None:
        nonlocal byte_count
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise PolicyEngineRuntimeError(
                f"Could not enumerate {label}: {directory}"
            ) from exc
        for entry in entries:
            if relative_root is None and entry.name in excluded_top_level:
                continue
            relative = (
                PurePosixPath(entry.name)
                if relative_root is None
                else relative_root / entry.name
            )
            relative_text = relative.as_posix()
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise PolicyEngineRuntimeError(
                    f"Could not inspect {label} entry: {path}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                raise PolicyEngineRuntimeError(f"{label} contains symlink: {path}")
            _require_protected_metadata(path, info, label=f"{label} entry")
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative_text)
                record(b"D", relative_text, info)
                walk(path, relative)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise PolicyEngineRuntimeError(f"{label} contains special file: {path}")
            raw, stable_info = _stable_regular_file(path, label=f"{label} file")
            if len(file_records) >= _MAX_TREE_FILES:
                raise PolicyEngineRuntimeError(f"{label} exceeds file-count bounds")
            byte_count += len(raw)
            if byte_count > _MAX_TREE_BYTES:
                raise PolicyEngineRuntimeError(f"{label} exceeds byte-count bounds")
            record(b"F", relative_text, stable_info)
            file_digest = hashlib.sha256(raw).hexdigest()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(bytes.fromhex(file_digest))
            file_records.append(
                _OfficialFile(
                    path=relative_text,
                    sha256=file_digest,
                    size=len(raw),
                    executable=bool(stable_info.st_mode & 0o111),
                )
            )

    walk(root, None)
    return _TreeIdentity(
        sha256=digest.hexdigest(),
        file_count=len(file_records),
        directory_count=len(directories),
        byte_count=byte_count,
        files=tuple(file_records),
        directories=tuple(directories),
    )


def _validate_git_head(root: Path, trusted_commit: str) -> None:
    """Require a protected detached HEAD; local refs and remotes are not authority."""

    git_dir = root / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime must use a self-contained .git directory"
        )
    git_info = git_dir.stat()
    _require_protected_metadata(git_dir, git_info, label="PolicyEngine .git directory")
    head_path = git_dir / "HEAD"
    raw, _info = _stable_regular_file(head_path, label="PolicyEngine detached HEAD")
    try:
        head = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime HEAD is malformed"
        ) from exc
    if _GIT_COMMIT_RE.fullmatch(head) is None:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime must be checked out at a detached 40-character HEAD"
        )
    if head != trusted_commit:
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime HEAD {head} does not match trusted pin {trusted_commit}"
        )


def _expected_directories(files: tuple[_OfficialFile, ...]) -> set[str]:
    result: set[str] = set()
    for item in files:
        path = PurePosixPath(item.path)
        for parent in path.parents:
            if parent != PurePosixPath("."):
                result.add(parent.as_posix())
    return result


def _validate_checkout_against_official_tree(
    checkout: _TreeIdentity,
    official: _OfficialTree,
) -> None:
    observed_files = {item.path: item for item in checkout.files}
    expected_files = {item.path: item for item in official.files}
    if set(observed_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(observed_files))
        extra = sorted(set(observed_files) - set(expected_files))
        detail = f"missing {missing[0]!r}" if missing else f"extra {extra[0]!r}"
        raise PolicyEngineRuntimeError(
            f"PolicyEngine checkout does not match official pinned tree: {detail}"
        )
    expected_directories = _expected_directories(official.files)
    observed_directories = set(checkout.directories)
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        extra = sorted(observed_directories - expected_directories)
        detail = (
            f"missing directory {missing[0]!r}"
            if missing
            else f"extra directory {extra[0]!r}"
        )
        raise PolicyEngineRuntimeError(
            f"PolicyEngine checkout does not match official pinned tree: {detail}"
        )
    for relative, expected in expected_files.items():
        observed = observed_files[relative]
        if (
            observed.sha256 != expected.sha256
            or observed.size != expected.size
            or observed.executable != expected.executable
        ):
            raise PolicyEngineRuntimeError(
                "PolicyEngine checkout file does not match official pinned tree: "
                f"{relative}"
            )


def _validate_startup_surface(root: Path, venv: Path) -> tuple[Path, Path]:
    """Locate one self-contained stdlib/site-packages pair and reject hooks."""

    forbidden_names = {"sitecustomize.py", "usercustomize.py", "pyvenv.cfg"}
    for tree in (root, venv):
        for candidate in tree.rglob("*"):
            if (
                candidate.name in forbidden_names
                or candidate.suffix in {".pth", "._pth"}
                or candidate.name.startswith("sitecustomize.")
                or candidate.name.startswith("usercustomize.")
            ):
                raise PolicyEngineRuntimeError(
                    "PolicyEngine runtime contains forbidden Python startup "
                    f"component: {candidate}"
                )
    library_root = venv / "lib"
    candidates = []
    if library_root.is_dir() and not library_root.is_symlink():
        for candidate in library_root.iterdir():
            if candidate.is_dir() and _PYTHON_LIBRARY_RE.fullmatch(candidate.name):
                site_packages = candidate / "site-packages"
                if site_packages.is_dir() and not site_packages.is_symlink():
                    candidates.append((candidate, site_packages))
    if len(candidates) != 1:
        raise PolicyEngineRuntimeError(
            "PolicyEngine .venv must contain exactly one self-contained "
            "lib/pythonX.Y/site-packages tree"
        )
    stdlib_root, site_packages = candidates[0]
    os_path = stdlib_root / "os.py"
    if os_path.is_symlink() or not os_path.is_file():
        raise PolicyEngineRuntimeError(
            "PolicyEngine .venv must contain its own complete Python standard library"
        )
    return stdlib_root, site_packages


_TRUSTED_BOOTSTRAP = r"""
import pathlib
import sys

if sys.flags.isolated != 1 or sys.flags.no_site != 1:
    raise RuntimeError("PolicyEngine runtime requires Python -I -S")
if len(sys.argv) < 4:
    raise RuntimeError("invalid sealed PolicyEngine bootstrap arguments")
checkout = str(pathlib.Path(sys.argv[1]).resolve(strict=True))
site_packages = str(pathlib.Path(sys.argv[2]).resolve(strict=True))
script = sys.argv[3]
script_arguments = sys.argv[4:]
initial_sys_path = tuple(sys.path)
sys.path[:0] = [checkout, site_packages]
sys.argv[:] = ["<axiom-policyengine-oracle>", *script_arguments]
scope = {
    "__name__": "__main__",
    "__file__": "<axiom-policyengine-oracle>",
    "__axiom_initial_sys_path__": initial_sys_path,
}
exec(compile(script, "<axiom-policyengine-oracle>", "exec"), scope, scope)
"""


_RUNTIME_PROBE_SCRIPT = r"""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
country = sys.argv[2]
packages = {}
for distribution, module_name in (
    (f"policyengine-{country}", f"policyengine_{country}"),
    ("policyengine-core", "policyengine_core"),
):
    module = importlib.import_module(module_name)
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise RuntimeError(f"{module_name} has no module origin")
    distribution_object = importlib.metadata.distribution(distribution)
    packages[distribution] = {
        "distribution": distribution,
        "version": distribution_object.version,
        "module_origin": str(Path(raw_origin).resolve(strict=True)),
        "metadata_root": str(Path(distribution_object.locate_file("")).resolve(strict=True)),
    }

payload = {
    "python_version": "{}.{}.{}".format(*sys.version_info[:3]),
    "python_implementation": sys.implementation.name,
    "python_executable": str(Path(sys.executable).resolve(strict=True)),
    "python_prefix": str(Path(sys.prefix).resolve(strict=True)),
    "python_base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
    "python_exec_prefix": str(Path(sys.exec_prefix).resolve(strict=True)),
    "python_base_exec_prefix": str(Path(sys.base_exec_prefix).resolve(strict=True)),
    "initial_sys_path": list(__axiom_initial_sys_path__),
    "effective_sys_path": list(sys.path),
    "isolated": sys.flags.isolated,
    "no_site": sys.flags.no_site,
    "packages": packages,
}
print("AXIOM_POLICYENGINE_RUNTIME:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


def policyengine_subprocess_environment() -> dict[str, str]:
    """Return the entire environment allowed inside a PolicyEngine process."""

    return {
        "PATH": "/nonexistent",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }


def _bootstrap_command(
    python_path: Path,
    root: Path,
    site_packages: Path,
    script: str,
    *script_args: str,
) -> list[str]:
    return [
        str(python_path),
        "-I",
        "-S",
        "-B",
        "-c",
        _TRUSTED_BOOTSTRAP,
        str(root),
        str(site_packages),
        script,
        *script_args,
    ]


def _probe_runtime(
    root: Path,
    country: str,
    python_path: Path,
    site_packages: Path,
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            _bootstrap_command(
                python_path,
                root,
                site_packages,
                _RUNTIME_PROBE_SCRIPT,
                str(root),
                country,
            ),
            check=False,
            capture_output=True,
            text=True,
            timeout=_ORACLE_TIMEOUT_SECONDS,
            cwd=str(root),
            env=policyengine_subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime probe could not execute: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise PolicyEngineRuntimeError(f"PolicyEngine runtime probe failed{suffix}")
    lines = [
        line for line in result.stdout.splitlines() if line.startswith(_PROBE_PREFIX)
    ]
    if len(lines) != 1:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime probe did not emit exactly one identity payload"
        )
    try:
        payload = json.loads(lines[0].removeprefix(_PROBE_PREFIX))
    except json.JSONDecodeError as exc:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime probe emitted malformed identity JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime probe identity is malformed"
        )
    return payload


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_probe_payload(
    payload: dict[str, Any],
    *,
    root: Path,
    venv: Path,
    country: str,
    python_path: Path,
    site_packages: Path,
    locked_versions: Mapping[str, str],
) -> dict[str, Any]:
    expected_keys = {
        "python_version",
        "python_implementation",
        "python_executable",
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
        "initial_sys_path",
        "effective_sys_path",
        "isolated",
        "no_site",
        "packages",
    }
    if set(payload) != expected_keys:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime probe identity has unexpected fields"
        )
    if payload.get("python_implementation") != "cpython":
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime interpreter must be CPython"
        )
    python_version = payload.get("python_version")
    if (
        not isinstance(python_version, str)
        or re.fullmatch(r"\d+\.\d+\.\d+", python_version) is None
    ):
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime Python version is malformed"
        )
    if payload.get("isolated") != 1 or payload.get("no_site") != 1:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime probe did not execute under -I -S"
        )
    if payload.get("python_executable") != str(python_path):
        raise PolicyEngineRuntimeError(
            "PolicyEngine probe executed an unexpected Python interpreter"
        )
    for field in (
        "python_prefix",
        "python_base_prefix",
        "python_exec_prefix",
        "python_base_exec_prefix",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not _path_inside(Path(value), venv):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {field} escapes checkout-owned .venv"
            )
    initial = payload.get("initial_sys_path")
    effective = payload.get("effective_sys_path")
    if not isinstance(initial, list) or not all(
        isinstance(value, str) and value and Path(value).is_absolute()
        for value in initial
    ):
        raise PolicyEngineRuntimeError("PolicyEngine initial sys.path is malformed")
    resolved_initial = [Path(value).resolve(strict=False) for value in initial]
    if not all(_path_inside(value, venv) for value in resolved_initial):
        raise PolicyEngineRuntimeError(
            "PolicyEngine initial sys.path contains an out-of-root component"
        )
    expected_effective = [str(root), str(site_packages), *initial]
    if effective != expected_effective:
        raise PolicyEngineRuntimeError(
            "PolicyEngine trusted bootstrap produced an unexpected import path"
        )
    packages = payload.get("packages")
    expected_packages = {f"policyengine-{country}", "policyengine-core"}
    if not isinstance(packages, dict) or set(packages) != expected_packages:
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime package identity is incomplete"
        )
    for distribution in sorted(expected_packages):
        package = packages.get(distribution)
        if not isinstance(package, dict) or set(package) != {
            "distribution",
            "version",
            "module_origin",
            "metadata_root",
        }:
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {distribution} identity is malformed"
            )
        if package.get("distribution") != distribution:
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {distribution} identity is inconsistent"
            )
        if package.get("version") != locked_versions[distribution]:
            raise PolicyEngineRuntimeError(
                f"Installed {distribution} version {package.get('version')!r} does not "
                f"match uv.lock version {locked_versions[distribution]!r}"
            )
        origin = package.get("module_origin")
        metadata_root = package.get("metadata_root")
        if not isinstance(origin, str) or not isinstance(metadata_root, str):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {distribution} paths are malformed"
            )
        expected_root = (
            root if distribution == f"policyengine-{country}" else site_packages
        )
        if not _path_inside(Path(origin), expected_root):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {distribution} module escapes its sealed root"
            )
        if not _path_inside(Path(metadata_root), site_packages):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime {distribution} metadata escapes site-packages"
            )
    return payload


def _locked_versions(lock_bytes: bytes, country: str) -> dict[str, str]:
    try:
        payload = tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PolicyEngineRuntimeError(
            "PolicyEngine uv.lock is not valid UTF-8 TOML"
        ) from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise PolicyEngineRuntimeError("PolicyEngine uv.lock has no package inventory")
    expected_names = (f"policyengine-{country}", "policyengine-core")
    result: dict[str, str] = {}
    for expected_name in expected_names:
        versions = {
            item.get("version")
            for item in packages
            if isinstance(item, dict) and item.get("name") == expected_name
        }
        if len(versions) != 1 or not all(
            isinstance(version, str) and version for version in versions
        ):
            raise PolicyEngineRuntimeError(
                f"PolicyEngine uv.lock must pin exactly one {expected_name} version"
            )
        result[expected_name] = next(iter(versions))
    return result


def _runtime_snapshot(
    root: Path,
    pin: _RuntimePin,
    official: _OfficialTree,
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Build a parent-computed identity for one sealed local closure."""

    if official.commit != pin.commit:
        raise PolicyEngineRuntimeError(
            "Official PolicyEngine tree does not match the trusted runtime pin"
        )
    _require_unprivileged_process()
    _require_protected_ancestor_chain(root)
    if root.name != f"policyengine-{pin.country}":
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime root must be named policyengine-{pin.country}: {root}"
        )
    _validate_git_head(root, pin.commit)
    checkout_tree = _hash_protected_tree(
        root,
        label="PolicyEngine checkout execution tree",
        excluded_top_level=frozenset({".git", ".venv"}),
    )
    _validate_checkout_against_official_tree(checkout_tree, official)

    venv = root / ".venv"
    if venv.is_symlink() or not venv.is_dir():
        raise PolicyEngineRuntimeError(
            f"PolicyEngine runtime requires a checkout-owned .venv: {venv}"
        )
    stdlib_root, site_packages = _validate_startup_surface(root, venv)
    venv_tree = _hash_protected_tree(venv, label="PolicyEngine .venv execution tree")
    python_path = venv / "bin" / "python"
    if python_path.is_symlink() or not python_path.is_file():
        raise PolicyEngineRuntimeError(
            "PolicyEngine runtime requires a regular in-root .venv/bin/python"
        )
    python_info = python_path.stat()
    if not python_info.st_mode & stat.S_IXUSR:
        raise PolicyEngineRuntimeError(
            "PolicyEngine .venv/bin/python must be executable by its owner"
        )
    pyproject_bytes = _read_regular_bytes(
        root / "pyproject.toml", label="PolicyEngine pyproject.toml"
    )
    lock_bytes = _read_regular_bytes(root / "uv.lock", label="PolicyEngine uv.lock")
    locked_versions = _locked_versions(lock_bytes, pin.country)
    probe = _validated_probe_payload(
        _probe_runtime(root, pin.country, python_path, site_packages),
        root=root,
        venv=venv,
        country=pin.country,
        python_path=python_path,
        site_packages=site_packages,
        locked_versions=locked_versions,
    )
    # Recompute the complete trees after executing imports.  The before/after
    # identities must be exact; child-reported hashes are never accepted.
    checkout_after = _hash_protected_tree(
        root,
        label="PolicyEngine checkout execution tree",
        excluded_top_level=frozenset({".git", ".venv"}),
    )
    venv_after = _hash_protected_tree(venv, label="PolicyEngine .venv execution tree")
    _validate_git_head(root, pin.commit)
    if checkout_after != checkout_tree or venv_after != venv_tree:
        raise PolicyEngineRuntimeError(
            "PolicyEngine execution closure changed during admission probe"
        )
    identity: dict[str, Any] = {
        "schema": POLICYENGINE_RUNTIME_SCHEMA,
        "country": pin.country,
        "official_repository_url": _official_https_remote(pin.country),
        "trusted_git_commit": pin.commit,
        "official_tree_sha256": official.content_sha256,
        "official_tree_file_count": len(official.files),
        "official_tree_byte_count": official.byte_count,
        "rulespec_runtime_pin_path": str(pin.path),
        "rulespec_runtime_pin_schema": POLICYENGINE_RUNTIME_PIN_SCHEMA,
        "rulespec_runtime_pin_sha256": pin.sha256,
        "repository_root": str(root),
        "checkout_execution_tree_sha256": checkout_tree.sha256,
        "checkout_execution_file_count": checkout_tree.file_count,
        "checkout_execution_byte_count": checkout_tree.byte_count,
        "venv_root": str(venv),
        "venv_execution_tree_sha256": venv_tree.sha256,
        "venv_execution_file_count": venv_tree.file_count,
        "venv_execution_byte_count": venv_tree.byte_count,
        "stdlib_root": str(stdlib_root),
        "site_packages_root": str(site_packages),
        "pyproject_sha256": hashlib.sha256(pyproject_bytes).hexdigest(),
        "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "locked_versions": dict(sorted(locked_versions.items())),
        **probe,
    }
    return identity, python_path, venv, site_packages


@dataclass(frozen=True, slots=True, init=False)
class PolicyEngineRuntime:
    """Content-bound identity for one pinned, sealed PolicyEngine checkout."""

    root: Path
    country: str
    python_path: Path
    site_packages_path: Path
    rulespec_checkout_root: Path
    identity: dict[str, Any]
    identity_sha256: str
    _pin: _RuntimePin
    _official_tree: _OfficialTree

    @classmethod
    def for_rulespec_root(
        cls,
        runtime_root: Path,
        *,
        policy_repo_root: Path,
    ) -> PolicyEngineRuntime:
        """Admit the runtime pinned by one canonical RuleSpec country checkout."""

        pin = _load_runtime_pin(policy_repo_root)
        canonical_root = _absolute_unaliased_directory(Path(runtime_root))
        official = _fetch_official_tree(pin.country, pin.commit)
        identity, python_path, _venv, site_packages = _runtime_snapshot(
            canonical_root, pin, official
        )
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        runtime = object.__new__(cls)
        object.__setattr__(runtime, "root", canonical_root)
        object.__setattr__(runtime, "country", pin.country)
        object.__setattr__(runtime, "python_path", python_path)
        object.__setattr__(runtime, "site_packages_path", site_packages)
        object.__setattr__(
            runtime, "rulespec_checkout_root", pin.rulespec_checkout_root
        )
        object.__setattr__(runtime, "identity", identity)
        object.__setattr__(
            runtime,
            "identity_sha256",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(runtime, "_pin", pin)
        object.__setattr__(runtime, "_official_tree", official)
        return runtime

    def canonical_identity(self) -> dict[str, Any]:
        """Return a detached JSON-compatible identity payload."""

        return json.loads(json.dumps(self.identity, sort_keys=True))

    def assert_matches_rulespec_root(self, policy_repo_root: Path) -> None:
        """Reject cross-country or cross-toolchain oracle use."""

        country = rulespec_country_from_root(policy_repo_root)
        checkout = _rulespec_checkout_root(policy_repo_root)
        if country != self.country:
            raise PolicyEngineRuntimeError(
                f"PolicyEngine runtime country '{self.country}' does not match "
                f"active RuleSpec country '{country}'"
            )
        if checkout != self.rulespec_checkout_root:
            raise PolicyEngineRuntimeError(
                "PolicyEngine runtime is bound to a different RuleSpec toolchain checkout"
            )

    def oracle_command(self, script: str) -> list[str]:
        """Return the sole isolated command shape allowed for PE oracle code."""

        if not isinstance(script, str) or not script:
            raise PolicyEngineRuntimeError(
                "PolicyEngine oracle script must be a non-empty string"
            )
        return _bootstrap_command(
            self.python_path,
            self.root,
            self.site_packages_path,
            script,
        )

    def assert_unchanged(self) -> None:
        """Parent-rehash and re-probe the exact closure around every oracle run."""

        current_pin = _load_runtime_pin_from_checkout(
            self.rulespec_checkout_root, self.country
        )
        if current_pin != self._pin:
            raise PolicyEngineRuntimeError(
                "RuleSpec PolicyEngine runtime pin changed after admission"
            )
        current, python_path, _venv, site_packages = _runtime_snapshot(
            self.root, self._pin, self._official_tree
        )
        canonical = json.dumps(current, sort_keys=True, separators=(",", ":"))
        current_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if (
            current_sha256 != self.identity_sha256
            or current != self.identity
            or python_path != self.python_path
            or site_packages != self.site_packages_path
        ):
            raise PolicyEngineRuntimeError(
                "PolicyEngine runtime identity changed after it was admitted"
            )
