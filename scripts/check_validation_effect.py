"""Require each encoder pull request to declare its validation effect (#2094).

The RuleSpec drift run uses the declaration to choose its scope: ``loosen``
re-checks waived files, ``tighten`` re-checks every file, and ``none`` starts
no drift run. A weekly full run catches a wrong declaration.

The pull request body must contain exactly one line of the form::

    Validation effect: none | loosen | tighten

A ``loosen`` or ``tighten`` declaration must also name the changed check ids
on one ``Changed checks:`` line (a comma-separated list, or ``unknown``).
"""

from __future__ import annotations

import os
import re
import sys

EFFECTS = ("none", "loosen", "tighten")
_EFFECT_RE = re.compile(r"^\s*validation effect\s*:\s*(\S+)\s*$", re.I | re.M)
_CHECKS_RE = re.compile(r"^\s*changed checks\s*:\s*(.+?)\s*$", re.I | re.M)
_CHECK_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")


def validation_effect_issues(body: str | None) -> tuple[str | None, list[str]]:
    """Return the declared effect and the reasons the declaration is invalid."""

    # Template guidance sits in HTML comments; it is not a declaration.
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    effects = _EFFECT_RE.findall(text)
    if len(effects) != 1:
        return None, [
            "the pull request body needs exactly one line "
            "'Validation effect: none|loosen|tighten'"
            + (f" (found {len(effects)})" if effects else "")
        ]
    effect = effects[0].strip("`*_").lower()
    if effect not in EFFECTS:
        return None, [f"unknown validation effect {effects[0]!r}"]
    issues: list[str] = []
    if effect != "none":
        checks = _CHECKS_RE.findall(text)
        if len(checks) != 1:
            issues.append(
                f"a '{effect}' change needs exactly one 'Changed checks:' line "
                "with check ids or 'unknown'"
            )
        else:
            values = [item.strip(" `").lower() for item in checks[0].split(",")]
            if not values or any(
                not value or _CHECK_ID_RE.fullmatch(value) is None for value in values
            ):
                issues.append("'Changed checks:' must list check ids or 'unknown'")
    return effect, issues


def main() -> int:
    effect, issues = validation_effect_issues(os.environ.get("PR_BODY"))
    for issue in issues:
        print(f"::error::{issue}")
    if issues:
        return 1
    print(f"Validation effect: {effect}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Validation effect: `{effect}`\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
