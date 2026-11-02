"""Require generated derived formulas and their dates to live in versions."""

from __future__ import annotations

import yaml


def find_rule_version_form_issues(content: str) -> list[str]:
    """Reject rule-level shorthand before generated content enters validation."""
    try:
        payload = yaml.safe_load(content)
    except (yaml.YAMLError, TypeError, ValueError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
        return []
    keys = {"formula", "from", "to", "effective_from", "effective_to"}
    return [
        f"[rule-version-form] Rule `{rule.get('name') or '<unnamed>'}` must put "
        "`formula` and its dates under `versions:`."
        for rule in payload["rules"]
        if isinstance(rule, dict)
        and str(rule.get("kind") or "").strip().lower()
        in {"derived", "derived_relation"}
        and keys.intersection(rule)
    ]
