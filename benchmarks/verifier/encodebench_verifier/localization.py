"""Does a finding point at the planted defect?

A judge *localises* a defect when one of its findings names the mutated rule
(``rule_path`` contains the rule's name as a whole identifier, or its
``rules[i]`` index) or mentions the distinctive token of the edit as a whole
word (the new amount, the dropped identifier, the operand beside the flipped
boundary, the new date). A rule named ``income`` does not match the path
``net_income_limit``, and the amount ``75`` does not match ``750``. Entity
and period edits localise by rule only: their tokens (``person``, ``month``)
are ordinary vocabulary that most explanations use, so a mention is no
evidence. Judges that return only probabilities (Jev) score blank here by
construction.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .cases import Locator

# Edits whose token is generic vocabulary rather than a distinctive value.
_RULE_ONLY_SUFFIXES = (".entity", ".period")


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _whole(token: str) -> re.Pattern[str]:
    """``token`` as a whole identifier, word or number, never a fragment.

    A number must not sit inside a longer number (``75`` in ``750`` or
    ``1.75``); an identifier must not sit inside a longer identifier, though
    it may follow a dot (``module.rule_name``).
    """

    escaped = re.escape(token)
    if re.fullmatch(r"[\d.,]+", token):
        return re.compile(rf"(?<![\w.]){escaped}(?![\w]|\.\d)")
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def localize(
    locator: Optional[Locator], findings: list[dict[str, Any]]
) -> tuple[Optional[bool], Optional[str]]:
    """Return ``(localized, evidence)``; ``(None, None)`` when not applicable."""

    if locator is None:
        return None, None
    if not findings:
        return False, None
    rule_name = _norm(locator.rule_name)
    rule_pattern = _whole(rule_name) if rule_name else None
    index_path = (
        f"rules[{locator.rule_index}]" if locator.rule_index is not None else None
    )
    token = _norm(locator.token)
    if token and (len(token) < 3 and not token.isdigit()):
        token = ""
    if (locator.path or "").endswith(_RULE_ONLY_SUFFIXES):
        token = ""
    token_pattern = _whole(token) if token else None
    for position, finding in enumerate(findings):
        rule_path = _norm(finding.get("rule_path"))
        explanation = _norm(finding.get("explanation"))
        if rule_pattern and rule_pattern.search(rule_path):
            return True, f"finding[{position}].rule_path names rule {locator.rule_name}"
        if index_path and index_path in rule_path:
            return True, f"finding[{position}].rule_path names {index_path}"
        if token_pattern and (
            token_pattern.search(rule_path) or token_pattern.search(explanation)
        ):
            return True, f"finding[{position}] mentions token {locator.token!r}"
    return False, None
