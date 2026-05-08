"""PolicyEngine oracle coverage reports for RuleSpec repositories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .registry import PolicyEngineMapping, load_policyengine_registry

EXECUTABLE_RULE_KINDS = {"parameter", "derived"}
ORACLE_COVERAGE_STATUSES = {"comparable", "known_not_comparable", "unmapped"}


@dataclass(frozen=True)
class PolicyEngineCoverageItem:
    """One executable RuleSpec output classified against PolicyEngine."""

    legal_id: str
    repo: str
    file: str
    rule_name: str
    kind: str
    status: str
    program: str
    mapping_type: str | None = None
    policyengine_variable: str | None = None
    rationale: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "legal_id": self.legal_id,
            "repo": self.repo,
            "file": self.file,
            "rule_name": self.rule_name,
            "kind": self.kind,
            "status": self.status,
            "program": self.program,
            "mapping_type": self.mapping_type,
            "policyengine_variable": self.policyengine_variable,
            "rationale": self.rationale,
        }


def build_policyengine_coverage_report(
    root: Path,
    *,
    program: str | None = None,
) -> dict[str, Any]:
    """Classify executable RuleSpec outputs against the PolicyEngine registry."""
    registry = load_policyengine_registry()
    root = root.resolve()
    items = sorted(
        _iter_policyengine_coverage_items(root, registry),
        key=lambda item: item.legal_id,
    )
    if program:
        items = [item for item in items if item.program == program]

    status_counts = Counter(item.status for item in items)
    program_counts = Counter(item.program for item in items)
    repo_counts: dict[str, Counter[str]] = {}
    for item in items:
        repo_counts.setdefault(item.repo, Counter())[item.status] += 1

    return {
        "oracle": "policyengine",
        "root": str(root),
        "total_outputs": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "program_counts": dict(sorted(program_counts.items())),
        "repos": [
            {
                "repo": repo,
                "total_outputs": sum(counter.values()),
                "status_counts": dict(sorted(counter.items())),
            }
            for repo, counter in sorted(repo_counts.items())
        ],
        "items": [item.as_dict() for item in items],
    }


def _iter_policyengine_coverage_items(
    root: Path,
    registry,
) -> list[PolicyEngineCoverageItem]:
    items: list[PolicyEngineCoverageItem] = []
    for repo in sorted(root.glob("rules-*")):
        if not repo.is_dir():
            continue
        prefix = repo.name.removeprefix("rules-")
        for rulespec_file in sorted(repo.rglob("*.y*ml")):
            if rulespec_file.name.endswith(".test.yaml"):
                continue
            payload = _load_rulespec_payload(rulespec_file)
            if not payload:
                continue
            rules = payload.get("rules")
            if not isinstance(rules, list):
                continue
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                kind = str(rule.get("kind") or "").strip()
                if kind not in EXECUTABLE_RULE_KINDS:
                    continue
                rule_name = str(rule.get("name") or "").strip()
                if not rule_name:
                    continue
                legal_id = _canonical_rulespec_legal_id(
                    prefix=prefix,
                    repo=repo,
                    rulespec_file=rulespec_file,
                    rule_name=rule_name,
                )
                mapping = registry.mapping_for_legal_id(legal_id, country="us")
                items.append(
                    _coverage_item_from_mapping(
                        legal_id=legal_id,
                        repo=repo,
                        root=root,
                        rulespec_file=rulespec_file,
                        rule_name=rule_name,
                        kind=kind,
                        mapping=mapping,
                    )
                )
    return items


def _load_rulespec_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return None
    return payload


def _canonical_rulespec_legal_id(
    *,
    prefix: str,
    repo: Path,
    rulespec_file: Path,
    rule_name: str,
) -> str:
    relative = rulespec_file.resolve().relative_to(repo.resolve())
    if relative.suffix in {".yaml", ".yml"}:
        relative = relative.with_suffix("")
    return f"{prefix}:{relative.as_posix()}#{rule_name}"


def _coverage_item_from_mapping(
    *,
    legal_id: str,
    repo: Path,
    root: Path,
    rulespec_file: Path,
    rule_name: str,
    kind: str,
    mapping: PolicyEngineMapping | None,
) -> PolicyEngineCoverageItem:
    if mapping is None:
        status = "unmapped"
        program = _infer_program_from_legal_id(legal_id)
        mapping_type = None
    elif mapping.comparable:
        status = "comparable"
        program = mapping.program or _infer_program_from_legal_id(legal_id)
        mapping_type = mapping.mapping_type
    else:
        status = "known_not_comparable"
        program = mapping.program or _infer_program_from_legal_id(legal_id)
        mapping_type = mapping.mapping_type

    return PolicyEngineCoverageItem(
        legal_id=legal_id,
        repo=repo.name,
        file=str(rulespec_file.resolve().relative_to(root)),
        rule_name=rule_name,
        kind=kind,
        status=status,
        program=program,
        mapping_type=mapping_type,
        policyengine_variable=mapping.policyengine_variable if mapping else None,
        rationale=mapping.rationale if mapping else None,
    )


def _infer_program_from_legal_id(legal_id: str) -> str:
    lowered = legal_id.lower()
    if "snap" in lowered:
        return "snap"
    if lowered.startswith("us:statutes/26/") or lowered.startswith("us:policies/irs/"):
        return "tax"
    return "unknown"
