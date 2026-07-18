"""Canonical country-checkout routing for RuleSpec content.

RuleSpec content has one supported on-disk shape::

    rulespec-<country>/<jurisdiction>/...

For example, ``us-ca:regulations/mpp/63-300/1`` resolves beneath
``rulespec-us/us-ca``.  A flat ``rulespec-us-ca`` checkout, a workspace that
merely contains a matching checkout, or a checkout alias inferred from Git
origin is not a RuleSpec content root.  Callers must authorize the exact
country checkout (for dependencies) or its direct jurisdiction child (for the
active content root).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from .constants import RULESPEC_COMPOSITION_SPEC_ROOT, RULESPEC_FILESYSTEM_ROOTS


class _GitProbeError(RuntimeError):
    """Git identity could not be checked for an observed repository boundary."""


class CanonicalRuleSpecCheckoutInspection(NamedTuple):
    """Canonical checkout identity and a stable rejection code."""

    name: str | None
    rejection: str | None


def jurisdiction_country(prefix: str) -> str:
    """Return the country portion of a jurisdiction repo prefix (us-ca -> us)."""
    return prefix.split("-", 1)[0]


def monorepo_checkout_name(prefix: str) -> str:
    """Return the country monorepo repository name for a jurisdiction prefix."""
    return f"rulespec-{jurisdiction_country(prefix)}"


def _lexical_rulespec_path(path: Path) -> Path | None:
    """Return one absolute path only when no component is caller-controlled indirection."""

    raw = Path(os.path.abspath(Path(path).expanduser()))
    if sys.platform == "darwin":
        for alias, expected_target in (
            (Path("/var"), Path("/private/var")),
            (Path("/tmp"), Path("/private/tmp")),
            (Path("/etc"), Path("/private/etc")),
        ):
            try:
                relative = raw.relative_to(alias)
            except ValueError:
                continue
            try:
                if alias.is_symlink() and alias.resolve(strict=True) == expected_target:
                    raw = expected_target / relative
            except OSError:
                pass
            break

    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            return None
    return raw


def _canonical_country_checkout_name(
    path: Path,
    *,
    allow_composition_specs: bool = False,
) -> str | None:
    """Return the exact canonical country-checkout name for ``path``."""

    return inspect_canonical_rulespec_checkout(
        path,
        allow_composition_specs=allow_composition_specs,
    ).name


def inspect_canonical_rulespec_checkout(
    path: Path,
    *,
    allow_composition_specs: bool = False,
) -> CanonicalRuleSpecCheckoutInspection:
    """Inspect one exact country checkout without weakening route acceptance."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return CanonicalRuleSpecCheckoutInspection(None, "lexical-path-rejected")
    if not lexical.is_dir():
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-not-directory")
    checkout = lexical.resolve()
    name = checkout.name
    if not name.startswith("rulespec-"):
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-name-prefix")
    country = name.removeprefix("rulespec-")
    if re.fullmatch(r"[a-z]{2}", country) is None:
        return CanonicalRuleSpecCheckoutInspection(None, "checkout-country-name")
    blocked_roots = RULESPEC_FILESYSTEM_ROOTS
    if allow_composition_specs:
        composition_root = checkout / RULESPEC_COMPOSITION_SPEC_ROOT
        if (composition_root.exists() or composition_root.is_symlink()) and (
            composition_root.is_symlink() or not composition_root.is_dir()
        ):
            return CanonicalRuleSpecCheckoutInspection(
                None, "composition-root-not-directory"
            )
        blocked_roots = blocked_roots - {RULESPEC_COMPOSITION_SPEC_ROOT}
    if any(
        (checkout / root_name).exists() or (checkout / root_name).is_symlink()
        for root_name in blocked_roots
    ):
        return CanonicalRuleSpecCheckoutInspection(None, "atomic-root-at-checkout")
    try:
        git_boundary = _nearest_git_boundary(checkout)
    except _GitProbeError:
        return CanonicalRuleSpecCheckoutInspection(None, "git-boundary-probe-error")
    if git_boundary is not None and git_boundary != checkout:
        return CanonicalRuleSpecCheckoutInspection(None, "git-boundary-mismatch")
    if git_boundary is None:
        return CanonicalRuleSpecCheckoutInspection(name, None)
    try:
        git_top_level = _git_top_level(str(checkout))
    except _GitProbeError:
        return CanonicalRuleSpecCheckoutInspection(None, "git-top-level-probe-error")
    if git_top_level is None:
        return CanonicalRuleSpecCheckoutInspection(None, "git-top-level-unavailable")
    if git_top_level is not None and git_top_level != checkout:
        return CanonicalRuleSpecCheckoutInspection(None, "git-top-level-mismatch")
    try:
        origin_name = _git_origin_repo_name(str(checkout))
    except _GitProbeError:
        return CanonicalRuleSpecCheckoutInspection(None, "git-origin-probe-error")
    if origin_name is not None and origin_name != name:
        return CanonicalRuleSpecCheckoutInspection(None, "git-origin-name-mismatch")
    return CanonicalRuleSpecCheckoutInspection(name, None)


def canonical_rulespec_root_identity(path: Path) -> str | None:
    """Return a stable identity for an exact canonical jurisdiction root.

    ``rulespec-us/us-co`` is returned for the direct ``us-co`` child of the
    canonical ``rulespec-us`` checkout.  Checkout roots, flat legacy roots,
    nested files, workspace directories, and aliased checkout names return
    ``None``.
    """

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    content_root = lexical.resolve()
    if not content_root.is_dir():
        return None
    jurisdiction = content_root.name
    checkout = content_root.parent
    expected_checkout = monorepo_checkout_name(jurisdiction)
    if checkout.name != expected_checkout:
        return None
    if (
        _canonical_country_checkout_name(checkout, allow_composition_specs=True)
        != expected_checkout
    ):
        return None
    country = jurisdiction_country(jurisdiction)
    if not _is_jurisdiction_dir_name(jurisdiction, country):
        return None
    return f"{expected_checkout}/{jurisdiction}"


def is_policy_repo_root(path: Path) -> bool:
    """Return True for an exact canonical country checkout root."""

    return _canonical_country_checkout_name(Path(path)) is not None


def is_composition_policy_repo_root(path: Path) -> bool:
    """Return True for an exact country checkout that may contain ProgramSpecs."""

    return (
        _canonical_country_checkout_name(Path(path), allow_composition_specs=True)
        is not None
    )


def is_jurisdiction_content_root(path: Path) -> bool:
    """Return True for an exact direct jurisdiction child of a country checkout."""

    return canonical_rulespec_root_identity(path) is not None


def find_policy_repo_root(path: Path) -> Path | None:
    """Return the canonical jurisdiction content root containing ``path``."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    current = lexical.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if canonical_rulespec_root_identity(candidate) is not None:
            return candidate
    return None


def canonical_rulespec_repo_name(path: Path) -> str | None:
    """Return the canonical import-repository name for ``path`` when known."""

    lexical = _lexical_rulespec_path(path)
    if lexical is None:
        return None
    current = lexical.resolve()
    content_root = find_policy_repo_root(current)
    if content_root is not None:
        return f"rulespec-{content_root.name}"
    return _canonical_country_checkout_name(current, allow_composition_specs=True)


def jurisdiction_content_dir(checkout: Path, prefix: str) -> Path:
    """Return ``prefix``'s direct root inside one canonical country checkout."""

    candidates = candidate_jurisdiction_content_dirs(checkout, prefix)
    if len(candidates) != 1:
        raise ValueError(
            f"{checkout} is not the canonical RuleSpec root for {prefix!r}"
        )
    return candidates[0]


def candidate_jurisdiction_content_dirs(base: Path, prefix: str) -> list[Path]:
    """Return the one canonical content-root candidate authorized by ``base``.

    ``base`` must be either the exact country checkout or the exact matching
    jurisdiction content root.  Workspace and sibling-checkout probing is not
    performed.  The returned direct child is not existence-checked.
    """

    lexical = _lexical_rulespec_path(base)
    if lexical is None:
        return []
    base = lexical.resolve()
    expected_checkout = monorepo_checkout_name(prefix)
    if (
        base.name == expected_checkout
        and _canonical_country_checkout_name(base, allow_composition_specs=True)
        == expected_checkout
    ):
        return [base / prefix]
    if canonical_rulespec_root_identity(base) == f"{expected_checkout}/{prefix}":
        return [base]
    return []


def resolve_jurisdiction_content_dir(
    bases: Iterable[Path],
    prefix: str,
) -> Path | None:
    """Return the first existing content root for ``prefix`` across ``bases``."""
    for base in bases:
        for candidate in candidate_jurisdiction_content_dirs(base, prefix):
            if candidate.is_dir():
                return candidate
    return None


def jurisdiction_subdir_names(
    checkout: Path,
    *,
    allow_composition_specs: bool = False,
) -> set[str]:
    """Return direct jurisdiction children of one canonical country checkout."""

    lexical = _lexical_rulespec_path(checkout)
    if lexical is None:
        return set()
    checkout = lexical.resolve()
    checkout_name = _canonical_country_checkout_name(
        checkout,
        allow_composition_specs=allow_composition_specs,
    )
    if checkout_name is None or not checkout.is_dir():
        return set()
    country = checkout_name.removeprefix("rulespec-")
    return {
        child.name
        for child in checkout.iterdir()
        if not child.is_symlink()
        and child.is_dir()
        and _is_jurisdiction_dir_name(child.name, country)
    }


def iter_jurisdiction_content_dirs(workspace_root: Path) -> list[tuple[str, Path]]:
    """Enumerate direct jurisdiction roots of one explicit country checkout."""

    lexical = _lexical_rulespec_path(workspace_root)
    if lexical is None:
        return []
    checkout = lexical.resolve()
    return [
        (name, checkout / name)
        for name in sorted(
            jurisdiction_subdir_names(checkout, allow_composition_specs=True)
        )
    ]


def _is_jurisdiction_dir_name(name: str, suffix: str) -> bool:
    """Return whether a directory name is a jurisdiction dir of ``rulespec-<suffix>``."""
    if re.fullmatch(r"[a-z]{2}", suffix) is None:
        return False
    return re.fullmatch(rf"{re.escape(suffix)}(?:-[a-z0-9]+)*", name) is not None


def _nearest_git_boundary(path: Path) -> Path | None:
    """Return the nearest lexical `.git` boundary without invoking Git."""

    for candidate in (path, *path.parents):
        marker = candidate / ".git"
        if marker.is_symlink():
            raise _GitProbeError(f"Git boundary is a symlink: {marker}")
        if marker.exists():
            return candidate
    return None


def _git_top_level(root: str) -> Path | None:
    """Return the exact Git worktree root containing ``root``, when present."""

    try:
        completed = subprocess.run(
            ["git", "-C", root, "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(f"Could not inspect Git top-level for {root}") from exc
    if completed.returncode != 0:
        return None
    top_level = completed.stdout.strip()
    if not top_level:
        return None
    return Path(top_level).resolve()


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
    except (OSError, subprocess.SubprocessError) as exc:
        raise _GitProbeError(f"Could not inspect Git origin for {root}") from exc
    if completed.returncode != 0:
        return None
    remote = completed.stdout.strip().rstrip("/")
    if not remote:
        return None
    name = remote.rsplit("/", 1)[-1]
    return name.removesuffix(".git") or None
