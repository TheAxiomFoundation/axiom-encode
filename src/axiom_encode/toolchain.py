"""Strict RuleSpec toolchain loading for corpus-bound encoder commands."""

from __future__ import annotations

import hashlib
import os
import re
import tomllib
from base64 import b64encode
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


class RuleSpecToolchainError(ValueError):
    """A RuleSpec checkout does not declare one canonical toolchain contract."""


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
        payload = tomllib.loads(raw.decode("utf-8"))
    except UnsafeCorpusPathError as exc:
        raise RuleSpecToolchainError(str(exc)) from exc
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise RuleSpecToolchainError(
            f"RuleSpec toolchain is not valid UTF-8 TOML: {config_path}"
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


def load_rulespec_local_corpus_release(
    rulespec_root: Path,
    corpus_root: Path,
) -> LocalCorpusRelease:
    """Bind a RuleSpec checkout to its one configured local corpus release."""

    toolchain = load_rulespec_toolchain(rulespec_root)
    _verify_rulespec_validation_waiver_set(toolchain)
    try:
        broker = get_signing_broker()
    except SigningBrokerError as exc:
        raise RuleSpecToolchainError(
            "A protected signing broker is required to verify the pinned "
            "corpus release object"
        ) from exc
    public_keys_raw = broker.corpus_release_public_keys_raw
    if not public_keys_raw or any(
        len(public_key) != 32 for public_key in public_keys_raw
    ):
        raise RuleSpecToolchainError(
            "The protected signing broker has no valid corpus release public keyring"
        )
    public_keys = tuple(
        b64encode(public_key).decode("ascii") for public_key in public_keys_raw
    )
    return LocalCorpusRelease(
        corpus_root,
        toolchain.corpus_release,
        toolchain.corpus_release_content_sha256,
        public_keys,
    )
