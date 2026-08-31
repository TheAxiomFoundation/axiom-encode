"""Strict RuleSpec toolchain loading for corpus-bound encoder commands."""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from base64 import b64decode, b64encode
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from .corpus_resolver import (
    InvalidCorpusReleaseError,
    LocalCorpusRelease,
    UnsafeCorpusPathError,
    read_bounded_regular_file,
    validate_corpus_release_name,
)
from .repo_routing import is_composition_policy_repo_root
from .signing_broker import SigningBrokerError, get_signing_broker

MAX_RULESPEC_TOOLCHAIN_BYTES = 64 * 1024
MAX_VALIDATION_WAIVER_SET_BYTES = 2_000_000
CORPUS_RELEASE_FIELD = "axiom_corpus_release"
CORPUS_RELEASE_CONTENT_SHA256_FIELD = "axiom_corpus_release_content_sha256"
VALIDATION_WAIVER_SET_SHA256_FIELD = "validation_waiver_set_sha256"
VALIDATION_WAIVER_SET_PATH = "known-validation-gaps.yaml"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_WAIVER_DIGEST_ASSIGNMENT_RE = re.compile(
    rb"(?m)^(?P<prefix>[ \t]*validation_waiver_set_sha256[ \t]*=[ \t]*\")"
    rb"(?P<digest>[0-9a-f]{64})"
    rb"(?P<suffix>\"[^\r\n]*)(?P<newline>\r?\n|$)"
)
_LOCAL_CORPUS_RELEASE_PUBLIC_KEY: ContextVar[str | None] = ContextVar(
    "axiom_ci_corpus_release_public_key", default=None
)


class RuleSpecToolchainError(ValueError):
    """A RuleSpec checkout does not declare one canonical toolchain contract."""


def _parse_rulespec_toolchain_bytes(
    raw: bytes,
    *,
    source: Path | str,
) -> tuple[str, str, str]:
    """Parse one strict toolchain document without resolving a checkout."""

    if len(raw) > MAX_RULESPEC_TOOLCHAIN_BYTES:
        raise RuleSpecToolchainError(
            f"RuleSpec toolchain exceeds {MAX_RULESPEC_TOOLCHAIN_BYTES} bytes: {source}"
        )
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuleSpecToolchainError(
            f"RuleSpec toolchain is not valid UTF-8 TOML: {source}"
        ) from exc
    if set(payload) != {"toolchain"}:
        raise RuleSpecToolchainError(
            "RuleSpec toolchain must contain exactly one top-level [toolchain] table"
        )
    toolchain = payload["toolchain"]
    if not isinstance(toolchain, dict):
        raise RuleSpecToolchainError("[toolchain] must be a TOML table")
    expected_fields = {
        CORPUS_RELEASE_FIELD,
        CORPUS_RELEASE_CONTENT_SHA256_FIELD,
        VALIDATION_WAIVER_SET_SHA256_FIELD,
    }
    if set(toolchain) != expected_fields:
        raise RuleSpecToolchainError(
            "[toolchain] must contain exactly: " + ", ".join(sorted(expected_fields))
        )
    release_name = toolchain.get(CORPUS_RELEASE_FIELD)
    if not isinstance(release_name, str) or not release_name:
        raise RuleSpecToolchainError(
            f"[toolchain].{CORPUS_RELEASE_FIELD} must be a non-empty string"
        )
    if release_name != release_name.strip():
        raise RuleSpecToolchainError(
            f"[toolchain].{CORPUS_RELEASE_FIELD} must not contain "
            "surrounding whitespace"
        )
    try:
        release_name = validate_corpus_release_name(release_name)
    except InvalidCorpusReleaseError as exc:
        raise RuleSpecToolchainError(str(exc)) from exc
    content_sha256 = toolchain.get(CORPUS_RELEASE_CONTENT_SHA256_FIELD)
    if (
        not isinstance(content_sha256, str)
        or _SHA256_RE.fullmatch(content_sha256) is None
    ):
        raise RuleSpecToolchainError(
            f"[toolchain].{CORPUS_RELEASE_CONTENT_SHA256_FIELD} must be a "
            "lowercase sha256 digest"
        )
    waiver_digest = toolchain.get(VALIDATION_WAIVER_SET_SHA256_FIELD)
    if (
        not isinstance(waiver_digest, str)
        or _SHA256_RE.fullmatch(waiver_digest) is None
    ):
        raise RuleSpecToolchainError(
            f"[toolchain].{VALIDATION_WAIVER_SET_SHA256_FIELD} must be a "
            "lowercase sha256 digest"
        )
    return release_name, content_sha256, waiver_digest


def validation_waiver_digest_transition_issues(
    *,
    base_toolchain: bytes,
    head_toolchain: bytes,
    base_waivers: bytes,
    head_waivers: bytes,
) -> tuple[str, ...]:
    """Validate the exact digest rebind needed by a waiver-state transition.

    Both toolchains must bind their corresponding waiver bytes. The head
    toolchain must then equal the base toolchain byte-for-byte except for the
    one digest value. This deliberately rejects formatting, comments, key
    spelling, and every unrelated semantic edit in the approval PR.
    """

    issues: list[str] = []
    if len(base_waivers) > MAX_VALIDATION_WAIVER_SET_BYTES:
        issues.append("protected-base validation waiver set exceeds the maximum size")
    if len(head_waivers) > MAX_VALIDATION_WAIVER_SET_BYTES:
        issues.append("head validation waiver set exceeds the maximum size")
    if issues:
        return tuple(issues)

    try:
        base_fields = _parse_rulespec_toolchain_bytes(
            base_toolchain,
            source="protected-base .axiom/toolchain.toml",
        )
    except RuleSpecToolchainError as exc:
        issues.append(f"protected-base toolchain is invalid: {exc}")
        base_fields = None
    try:
        head_fields = _parse_rulespec_toolchain_bytes(
            head_toolchain,
            source="head .axiom/toolchain.toml",
        )
    except RuleSpecToolchainError as exc:
        issues.append(f"head toolchain is invalid: {exc}")
        head_fields = None
    if base_fields is None or head_fields is None:
        return tuple(issues)

    base_waiver_digest = hashlib.sha256(base_waivers).hexdigest()
    head_waiver_digest = hashlib.sha256(head_waivers).hexdigest()
    if base_fields[2] != base_waiver_digest:
        issues.append(
            "protected-base toolchain does not bind the exact protected-base "
            f"waiver bytes: {base_fields[2]} != {base_waiver_digest}"
        )
    if head_fields[2] != head_waiver_digest:
        issues.append(
            "head toolchain does not bind the exact head waiver bytes: "
            f"{head_fields[2]} != {head_waiver_digest}"
        )
    if base_waiver_digest == head_waiver_digest:
        issues.append("validation-waiver transition did not change the waiver-set bytes")

    if base_fields[:2] != head_fields[:2]:
        issues.append(
            "validation-waiver transition may not change the corpus release name or "
            "content digest"
        )

    matches = list(_VALIDATION_WAIVER_DIGEST_ASSIGNMENT_RE.finditer(base_toolchain))
    if len(matches) != 1:
        issues.append(
            "protected-base toolchain must contain exactly one canonical "
            f"{VALIDATION_WAIVER_SET_SHA256_FIELD} assignment"
        )
        return tuple(issues)
    assignment = matches[0]
    encoded_base_digest = assignment.group("digest").decode("ascii")
    if encoded_base_digest != base_fields[2]:
        issues.append(
            "protected-base toolchain digest assignment does not match its parsed value"
        )
        return tuple(issues)

    expected_head = (
        base_toolchain[: assignment.start("digest")]
        + head_waiver_digest.encode("ascii")
        + base_toolchain[assignment.end("digest") :]
    )
    if head_toolchain != expected_head:
        issues.append(
            "head toolchain bytes may change only the canonical "
            f"{VALIDATION_WAIVER_SET_SHA256_FIELD} value"
        )
    return tuple(issues)


def _require_canonical_country_checkout(root: Path) -> Path:
    """Reject flat, aliased, workspace, and otherwise noncanonical roots."""

    if not is_composition_policy_repo_root(root):
        raise RuleSpecToolchainError(
            "RuleSpec toolchain root must be the exact canonical "
            f"rulespec-<country> checkout: {root}"
        )
    return root


@dataclass(frozen=True, slots=True)
class RuleSpecToolchain:
    """Evidence identities declared by one canonical RuleSpec checkout."""

    root: Path
    corpus_release: str
    corpus_release_content_sha256: str
    validation_waiver_set_sha256: str


def _canonical_rulespec_root(raw_root: Path) -> Path:
    raw = Path(os.path.abspath(Path(raw_root).expanduser()))
    if raw.is_symlink():
        raise RuleSpecToolchainError(f"RuleSpec root must not be a symlink: {raw}")
    try:
        root = raw.resolve(strict=True)
    except OSError as exc:
        raise RuleSpecToolchainError(f"RuleSpec root does not exist: {raw}") from exc
    if root.is_file():
        search_root = root.parent
    elif root.is_dir():
        search_root = root
    else:
        raise RuleSpecToolchainError(
            f"RuleSpec path is not a regular file or directory: {raw}"
        )
    search_chain = (search_root, *search_root.parents)
    repository_root: Path | None = None
    repository_root_index: int | None = None
    for index, candidate in enumerate(search_chain):
        git_marker = candidate / ".git"
        if git_marker.is_symlink():
            raise RuleSpecToolchainError(
                f"RuleSpec checkout .git marker must not be a symlink: {git_marker}"
            )
        if git_marker.exists():
            repository_root = candidate
            repository_root_index = index
            break

    scoped_chain = (
        search_chain
        if repository_root_index is None
        else search_chain[: repository_root_index + 1]
    )
    configured_roots = [
        candidate
        for candidate in scoped_chain
        if (candidate / ".axiom" / "toolchain.toml").exists()
        or (candidate / ".axiom" / "toolchain.toml").is_symlink()
    ]
    if repository_root is not None:
        if configured_roots == [repository_root]:
            return _require_canonical_country_checkout(repository_root)
        if not configured_roots:
            raise RuleSpecToolchainError(
                "RuleSpec checkout root does not contain .axiom/toolchain.toml: "
                f"{repository_root}"
            )
        raise RuleSpecToolchainError(
            "RuleSpec checkout must have exactly one .axiom/toolchain.toml at "
            f"its root {repository_root}; found configuration under: "
            + ", ".join(str(path) for path in configured_roots)
        )
    if len(configured_roots) == 1:
        return _require_canonical_country_checkout(configured_roots[0])
    if not configured_roots:
        raise RuleSpecToolchainError(
            f"No .axiom/toolchain.toml found at or above RuleSpec path: {root}"
        )
    raise RuleSpecToolchainError(
        "RuleSpec path has multiple ancestor .axiom/toolchain.toml files: "
        + ", ".join(str(path) for path in configured_roots)
    )


def load_rulespec_toolchain(rulespec_root: Path) -> RuleSpecToolchain:
    """Load the immutable corpus and waiver identities for one RuleSpec repo."""

    root = _canonical_rulespec_root(rulespec_root)
    config_dir = root / ".axiom"
    config_path = config_dir / "toolchain.toml"
    if config_dir.is_symlink() or not config_dir.is_dir():
        raise RuleSpecToolchainError(
            f"RuleSpec checkout must contain a regular .axiom directory: {root}"
        )
    if config_path.is_symlink():
        raise RuleSpecToolchainError(
            f"RuleSpec toolchain file must not be a symlink: {config_path}"
        )
    try:
        raw = read_bounded_regular_file(
            root,
            config_path,
            label="RuleSpec toolchain file",
            max_bytes=MAX_RULESPEC_TOOLCHAIN_BYTES,
        )
    except UnsafeCorpusPathError as exc:
        raise RuleSpecToolchainError(str(exc)) from exc
    release_name, content_sha256, waiver_digest = _parse_rulespec_toolchain_bytes(
        raw,
        source=config_path,
    )
    return RuleSpecToolchain(
        root=root,
        corpus_release=release_name,
        corpus_release_content_sha256=content_sha256,
        validation_waiver_set_sha256=waiver_digest,
    )


def load_rulespec_corpus_release_pin(rulespec_root: Path) -> tuple[str, str]:
    """Return the exact named corpus release object pinned by one RuleSpec repo."""

    toolchain = load_rulespec_toolchain(rulespec_root)
    return toolchain.corpus_release, toolchain.corpus_release_content_sha256


def _verify_rulespec_validation_waiver_set(toolchain: RuleSpecToolchain) -> str:
    waiver_path = toolchain.root / VALIDATION_WAIVER_SET_PATH
    try:
        raw = read_bounded_regular_file(
            toolchain.root,
            waiver_path,
            label="RuleSpec validation waiver set",
            max_bytes=MAX_VALIDATION_WAIVER_SET_BYTES,
        )
    except UnsafeCorpusPathError as exc:
        raise RuleSpecToolchainError(str(exc)) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != toolchain.validation_waiver_set_sha256:
        raise RuleSpecToolchainError(
            f"{VALIDATION_WAIVER_SET_PATH} sha256 does not match "
            f"[toolchain].{VALIDATION_WAIVER_SET_SHA256_FIELD}: "
            f"{actual} != {toolchain.validation_waiver_set_sha256}"
        )
    return actual


def verify_rulespec_validation_waiver_set(rulespec_root: Path) -> str:
    """Verify and return the toolchain-bound waiver-set byte digest."""

    return _verify_rulespec_validation_waiver_set(
        load_rulespec_toolchain(rulespec_root)
    )


def _local_corpus_release_public_keys() -> tuple[str, ...]:
    public_key = _LOCAL_CORPUS_RELEASE_PUBLIC_KEY.get()
    if public_key is not None:
        return (public_key,)
    try:
        broker = get_signing_broker()
    except SigningBrokerError as exc:
        raise RuleSpecToolchainError(
            "A protected signing broker is required to verify the pinned "
            "corpus release object"
        ) from exc
    public_keys_raw = broker.corpus_release_public_keys_raw
    if not public_keys_raw or any(
        len(candidate) != 32 for candidate in public_keys_raw
    ):
        raise RuleSpecToolchainError(
            "The protected signing broker has no valid corpus release public keyring"
        )
    return tuple(b64encode(candidate).decode("ascii") for candidate in public_keys_raw)


def load_rulespec_local_corpus_release_snapshot(
    rulespec_root: Path,
    corpus_root: Path,
    *,
    toolchain_bytes: bytes,
    validation_waiver_bytes: bytes,
) -> LocalCorpusRelease:
    """Bind a corpus release to one already captured RuleSpec evidence pair."""

    root = _canonical_rulespec_root(rulespec_root)
    release_name, content_sha256, waiver_digest = _parse_rulespec_toolchain_bytes(
        toolchain_bytes,
        source=root / ".axiom/toolchain.toml",
    )
    actual_waiver_digest = hashlib.sha256(validation_waiver_bytes).hexdigest()
    if waiver_digest != actual_waiver_digest:
        raise RuleSpecToolchainError(
            f"{VALIDATION_WAIVER_SET_PATH} sha256 does not match "
            f"[toolchain].{VALIDATION_WAIVER_SET_SHA256_FIELD}: "
            f"{actual_waiver_digest} != {waiver_digest}"
        )
    return LocalCorpusRelease(
        corpus_root,
        release_name,
        content_sha256,
        _local_corpus_release_public_keys(),
    )


def load_rulespec_local_corpus_release(
    rulespec_root: Path,
    corpus_root: Path,
) -> LocalCorpusRelease:
    """Bind a RuleSpec checkout to its one configured local corpus release."""

    toolchain = load_rulespec_toolchain(rulespec_root)
    _verify_rulespec_validation_waiver_set(toolchain)
    return LocalCorpusRelease(
        corpus_root,
        toolchain.corpus_release,
        toolchain.corpus_release_content_sha256,
        _local_corpus_release_public_keys(),
    )


@contextmanager
def local_corpus_release_verification(public_key: str):
    """Temporarily provide a verification-only corpus public key.

    This exists solely for ``axiom-encode ci``.  It conveys no signing
    capability, is never populated from the environment, and keeps the trusted
    bootstrap's prohibition on environment-supplied public roots intact.
    """

    # Construct once so malformed keys fail before any gate starts.
    try:
        raw = b64decode(public_key, validate=True)
    except Exception as exc:
        raise RuleSpecToolchainError(
            "--corpus-release-public-key must be canonical base64"
        ) from exc
    if len(raw) != 32 or b64encode(raw).decode("ascii") != public_key:
        raise RuleSpecToolchainError(
            "--corpus-release-public-key must encode exactly 32 bytes"
        )
    token = _LOCAL_CORPUS_RELEASE_PUBLIC_KEY.set(public_key)
    try:
        yield
    finally:
        _LOCAL_CORPUS_RELEASE_PUBLIC_KEY.reset(token)
