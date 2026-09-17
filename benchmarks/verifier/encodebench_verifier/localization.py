"""Does a finding point at the planted defect?

A judge *localises* a defect when one of its findings names the mutated rule
(by ``rule_path`` naming the rule, or by ``rules[i]`` index) or mentions the
distinctive token of the edit (the new amount, the dropped identifier, the
operand beside the flipped boundary, the new entity or date). Judges that
return only probabilities (Jev) score blank here by construction.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .cases import Locator


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def localize(
    locator: Optional[Locator], findings: list[dict[str, Any]]
) -> tuple[Optional[bool], Optional[str]]:
    """Return ``(localized, evidence)``; ``(None, None)`` when not applicable."""

    if locator is None:
        return None, None
    if not findings:
        return False, None
    rule_name = _norm(locator.rule_name)
    index_path = (
        f"rules[{locator.rule_index}]" if locator.rule_index is not None else None
    )
    token = _norm(locator.token)
    if token and (len(token) < 3 and not token.isdigit()):
        token = ""
    for position, finding in enumerate(findings):
        rule_path = _norm(finding.get("rule_path"))
        explanation = _norm(finding.get("explanation"))
        if rule_name and rule_name in rule_path:
            return True, f"finding[{position}].rule_path names rule {locator.rule_name}"
        if index_path and index_path in rule_path:
            return True, f"finding[{position}].rule_path names {index_path}"
        if token and (token in rule_path or token in explanation):
            return True, f"finding[{position}] mentions token {locator.token!r}"
    return False, None
