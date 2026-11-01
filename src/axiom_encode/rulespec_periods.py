"""Shared period semantics for RuleSpec companion runners.

A companion assertion on a compiled parameter is answered here, not by the
engine, so these readings must never answer a period the engine would refuse.
"""

from datetime import date
from typing import Any


def rulespec_engine_date(value: object) -> date:
    """Parse the one date form both Python and the engine read as the same day.

    `date.fromisoformat` alone also reads `20260101` and `2026-W01-1`, and the
    engine refuses both.
    """
    text = str(value)
    parsed = date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError(f"Invalid isoformat string: {text!r}")
    return parsed


def rulespec_parameter_version_at(
    parameter: dict[str, Any], period_start: object
) -> dict[str, Any]:
    """Return the version the engine reads at a period start, or refuse.

    Mirrors the engine's `lookup_parameter`: of the versions that apply at the
    start, it reads the latest `effective_from`. When none applies it refuses,
    so a version that starts later or has ended never stands in.
    """
    at = rulespec_engine_date(period_start)
    applicable = []
    for version in parameter.get("versions") or []:
        if not isinstance(version, dict):
            continue
        effective_to = version.get("effective_to")
        if date.fromisoformat(str(version["effective_from"])) <= at and (
            effective_to is None or at <= date.fromisoformat(str(effective_to))
        ):
            applicable.append(version)
    if not applicable:
        raise ValueError(
            f"parameter `{parameter.get('name')}` has no version at {period_start}"
        )
    # Rust's `max_by_key` keeps the last of equal keys; Python's `max` the first.
    return max(
        reversed(applicable),
        key=lambda version: date.fromisoformat(str(version["effective_from"])),
    )
