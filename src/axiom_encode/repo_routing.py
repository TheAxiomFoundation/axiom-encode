"""Helpers for resolving jurisdiction RuleSpec repos versus the rules engine."""

from __future__ import annotations

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
