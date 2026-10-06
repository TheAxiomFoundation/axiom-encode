"""Map a canonical RuleSpec jurisdiction root to its anchor prefix."""

from __future__ import annotations

from pathlib import Path

from axiom_encode.repo_routing import canonical_rulespec_root_identity


def jurisdiction_prefix(repo_path: Path) -> str:
    """Return the jurisdiction for ``rulespec-<country>/<jurisdiction>``."""

    identity = canonical_rulespec_root_identity(Path(repo_path))
    if identity is None:
        raise ValueError(
            "jurisdiction roots must use the canonical "
            "rulespec-<country>/<jurisdiction> layout: "
            f"{repo_path}"
        )
    return identity.split("/", 1)[1]
