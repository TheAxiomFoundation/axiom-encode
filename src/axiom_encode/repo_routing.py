"""Helpers for resolving jurisdiction RuleSpec repos versus the rules engine."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path


def is_policy_repo_root(path: Path) -> bool:
    """Return True when a path is the root of a jurisdiction RuleSpec repo."""
    name = Path(path).resolve().name
    return name.startswith("rulespec-")


def find_policy_repo_root(path: Path) -> Path | None:
    """Walk upward from a file or directory to the enclosing policy repo root."""
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if is_policy_repo_root(candidate):
            return candidate
    return None


def canonical_rulespec_repo_name(path: Path) -> str | None:
    """Return the canonical `rulespec-*` repo name for a checkout when known."""
    current = Path(path).resolve()
    if current.is_file():
        current = current.parent

    root: Path | None = None
    for candidate in (current, *current.parents):
        if candidate.name.startswith("rulespec-"):
            root = candidate
            break
    root = root or current

    origin_name = _git_origin_repo_name(str(root))
    if origin_name and origin_name.startswith("rulespec-"):
        return origin_name
    return root.name if root.name.startswith("rulespec-") else None


@lru_cache(maxsize=256)
def _git_origin_repo_name(root: str) -> str | None:
    """Best-effort repository basename from Git origin."""
    try:
        completed = subprocess.run(
            ["git", "-C", root, "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    remote = completed.stdout.strip().rstrip("/")
    if not remote:
        return None
    name = remote.rsplit("/", 1)[-1]
    return name.removesuffix(".git") or None
