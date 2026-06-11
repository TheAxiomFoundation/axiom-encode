"""PolicyEngine oracle coverage reports for RuleSpec repositories."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.repo_routing import (
    canonical_rulespec_repo_name,
    iter_jurisdiction_content_dirs,
    jurisdiction_subdir_names,
    legacy_checkout_name,
)

from .registry import PolicyEngineMapping, load_policyengine_registry

EXECUTABLE_RULE_KINDS = {"parameter", "derived", "derived_relation"}
ORACLE_COVERAGE_STATUSES = {"comparable", "known_not_comparable", "unmapped"}
_PROGRAM_TOKEN_RE = {
    token: re.compile(rf"(^|[^a-z0-9]){token}([^a-z0-9]|$)")
    for token in ("medicaid", "chip", "aca")
}
_HEALTH_PROGRAMS = frozenset({"medicaid", "chip", "aca_ptc"})
_INTERVAL_BOUND_HELPER_RE = re.compile(
    r"(^|_)(tier|band|bracket|interval|row)_(lower|upper)_bound$"
)


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
    policyengine_parameter: str | None = None
    rationale: str | None = None
    candidate_priority: str | None = None
    tested: bool = False
    test_output_count: int = 0

    def as_dict(self) -> dict[str, str | int | bool | None]:
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
            "policyengine_parameter": self.policyengine_parameter,
            "rationale": self.rationale,
            "candidate_priority": self.candidate_priority,
            "tested": self.tested,
            "test_output_count": self.test_output_count,
        }


@dataclass(frozen=True)
class PolicyEngineCandidateItem:
    """One actionable oracle mapping candidate or review item."""

    legal_id: str
    repo: str
    file: str
    rule_name: str
    status: str
    program: str
    category: str
    priority: str
    recommendation: str
    policyengine_variable: str | None = None
    policyengine_parameter: str | None = None
    rationale: str | None = None
    tested: bool = False
    test_output_count: int = 0

    def as_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "legal_id": self.legal_id,
            "repo": self.repo,
            "file": self.file,
            "rule_name": self.rule_name,
            "status": self.status,
            "program": self.program,
            "category": self.category,
            "priority": self.priority,
            "recommendation": self.recommendation,
            "policyengine_variable": self.policyengine_variable,
            "policyengine_parameter": self.policyengine_parameter,
            "rationale": self.rationale,
            "tested": self.tested,
            "test_output_count": self.test_output_count,
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
        programs = _program_filter_values(program)
        items = [item for item in items if item.program in programs]

    status_counts = Counter(item.status for item in items)
    untested_comparable = [
        item for item in items if item.status == "comparable" and not item.tested
    ]
    program_counts = Counter(item.program for item in items)
    repo_counts: dict[str, Counter[str]] = {}
    for item in items:
        repo_counts.setdefault(item.repo, Counter())[item.status] += 1

    return {
        "oracle": "policyengine",
        "root": str(root),
        "total_outputs": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "untested_comparable": len(untested_comparable),
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


def build_policyengine_candidate_report(
    root: Path,
    *,
    program: str | None = None,
    policyengine_variables: set[str] | None = None,
) -> dict[str, Any]:
    """Build a triage queue for expanding PolicyEngine oracle coverage."""
    coverage_report = build_policyengine_coverage_report(root, program=program)
    pe_variables = (
        policyengine_variables
        if policyengine_variables is not None
        else _load_policyengine_variable_names()
    )
    raw_candidates = [
        _candidate_from_coverage_item(
            item,
            policyengine_variables=pe_variables,
        )
        for item in coverage_report["items"]
    ]
    candidates = sorted(
        (candidate for candidate in raw_candidates if candidate is not None),
        key=lambda item: (
            _candidate_priority_rank(item.priority),
            item.category,
            item.legal_id,
        ),
    )
    category_counts = Counter(item.category for item in candidates)
    priority_counts = Counter(item.priority for item in candidates)
    return {
        "oracle": "policyengine",
        "root": coverage_report["root"],
        "program": program,
        "policyengine_variables_available": pe_variables is not None,
        "total_candidates": len(candidates),
        "category_counts": dict(sorted(category_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "coverage_status_counts": coverage_report["status_counts"],
        "items": [item.as_dict() for item in candidates],
    }


def _iter_policyengine_coverage_items(
    root: Path,
    registry,
) -> list[PolicyEngineCoverageItem]:
    items: list[PolicyEngineCoverageItem] = []
    # Enumerate each jurisdiction's content root under the workspace, handling
    # both layouts: a legacy ``rulespec-<prefix>`` checkout contributes itself
    # under its prefix; a country monorepo ``rulespec-<country>`` contributes
    # each first-level jurisdiction directory (``us``, ``us-al``, ...,
    # ``uk-kingston-upon-thames``). ``prefix`` is the jurisdiction and
    # ``content_dir`` is the directory whose repo-relative paths form the
    # ``<prefix>:...`` portion of each output's legal ID, so a file at
    # ``<rulespec-us>/us-al/policies/X.yaml`` yields ``us-al:policies/X#name``
    # in either layout instead of a jurisdiction-doubled ``us:us-al/...`` ID.
    for prefix, content_dir in iter_jurisdiction_content_dirs(root):
        repo_name = canonical_rulespec_repo_name(content_dir) or legacy_checkout_name(
            prefix
        )
        # In a partially migrated monorepo a country root can be both the
        # content_dir for the country prefix and the parent of sibling
        # jurisdiction directories; skip those sibling subtrees here so each
        # output is attributed to exactly one (prefix, content_dir) pair.
        nested_subdirs = jurisdiction_subdir_names(content_dir)
        for rulespec_file in sorted(content_dir.rglob("*.y*ml")):
            if rulespec_file.name.endswith(".test.yaml"):
                continue
            rel_parts = rulespec_file.relative_to(content_dir).parts
            if "_axiom" in rel_parts:
                continue
            if rel_parts and rel_parts[0] in nested_subdirs:
                continue
            payload = _load_rulespec_payload(rulespec_file)
            if not payload:
                continue
            rules = payload.get("rules")
            if not isinstance(rules, list):
                continue
            test_output_counts = _rulespec_test_output_counts(rulespec_file)
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
                    content_dir=content_dir,
                    rulespec_file=rulespec_file,
                    rule_name=rule_name,
                )
                mapping = registry.mapping_for_legal_id(
                    legal_id,
                    country=_country_from_rulespec_prefix(prefix),
                )
                test_output_count = _mapping_test_output_count(
                    legal_id,
                    mapping=mapping,
                    test_output_counts=test_output_counts,
                )
                items.append(
                    _coverage_item_from_mapping(
                        legal_id=legal_id,
                        repo_name=repo_name,
                        root=root,
                        rulespec_file=rulespec_file,
                        rule_name=rule_name,
                        rule=rule,
                        kind=kind,
                        mapping=mapping,
                        test_output_count=test_output_count,
                    )
                )
    return items


def _candidate_from_coverage_item(
    item: dict[str, Any],
    *,
    policyengine_variables: set[str] | None,
) -> PolicyEngineCandidateItem | None:
    status = str(item.get("status") or "")
    rule_name = str(item.get("rule_name") or "")
    legal_id = str(item.get("legal_id") or "")
    tested = bool(item.get("tested"))
    pe_variable = item.get("policyengine_variable")
    pe_parameter = item.get("policyengine_parameter")
    exact_pe_variable = bool(
        policyengine_variables and rule_name in policyengine_variables
    )
    pe_looking = _is_policyengine_looking_rule_name(rule_name)

    if status == "comparable" and not tested:
        return _candidate_item(
            item,
            category="comparable_untested",
            priority=_candidate_priority(item, "P1"),
            recommendation=(
                "Add this comparable output to the companion .test.yaml so CI "
                "continues to exercise the oracle mapping."
            ),
        )

    if status == "unmapped":
        if exact_pe_variable:
            return _candidate_item(
                item,
                category="exact_variable_unmapped",
                priority=_candidate_priority(item, "P1" if tested else "P2"),
                policyengine_variable=rule_name,
                recommendation=(
                    f"Review whether `{legal_id}` has the same legal boundary as "
                    f"PolicyEngine variable `{rule_name}`; if exact, add a "
                    "direct_variable mapping."
                ),
            )
        if tested and pe_looking:
            return _candidate_item(
                item,
                category="tested_unmapped_pe_like",
                priority=_candidate_priority(item, "P2"),
                recommendation=(
                    "This tested output looks oracle-relevant but has no explicit "
                    "classification. Add an exact mapping, an adapter-backed "
                    "mapping, or a not_comparable rationale."
                ),
            )
        return _candidate_item(
            item,
            category="unmapped",
            priority=_candidate_priority(item, "P3"),
            recommendation=(
                "Classify this output explicitly before relying on full oracle "
                "coverage gates."
            ),
        )

    if status == "known_not_comparable":
        if pe_variable or pe_parameter:
            priority = _candidate_priority(item, "P2" if tested else "P3")
            return _candidate_item(
                item,
                category="known_adjacent_target",
                priority=priority,
                recommendation=(
                    "A nearby PolicyEngine target is recorded but currently marked "
                    "not comparable. Revisit only if a small adapter can make the "
                    "RuleSpec tests compare without changing the legal boundary."
                    if priority != "P4"
                    else "A nearby PolicyEngine target is recorded, but this "
                    "non-comparable classification has already been reviewed. "
                    "Leave it alone unless the source or oracle semantics change."
                ),
            )
        if exact_pe_variable:
            priority = _candidate_priority(item, "P2" if tested else "P3")
            return _candidate_item(
                item,
                category="pe_name_but_not_comparable",
                priority=priority,
                policyengine_variable=rule_name,
                recommendation=(
                    f"`{rule_name}` exists in PolicyEngine, but the registry "
                    "currently classifies this Axiom output as not comparable. "
                    "Review the rationale before adding an exact override."
                    if priority != "P4"
                    else f"`{rule_name}` exists in PolicyEngine, but this "
                    "non-comparable classification has already been reviewed. "
                    "Leave it alone unless the source or oracle semantics change."
                ),
            )
        return _candidate_item(
            item,
            category="prefix_not_comparable",
            priority=_candidate_priority(item, "P4"),
            recommendation=(
                "Covered by a broad not_comparable prefix. Leave it alone unless "
                "this output becomes tested or a known exact oracle target."
            ),
        )

    return None


def _candidate_item(
    item: dict[str, Any],
    *,
    category: str,
    priority: str,
    recommendation: str,
    policyengine_variable: str | None = None,
    policyengine_parameter: str | None = None,
) -> PolicyEngineCandidateItem:
    return PolicyEngineCandidateItem(
        legal_id=str(item.get("legal_id") or ""),
        repo=str(item.get("repo") or ""),
        file=str(item.get("file") or ""),
        rule_name=str(item.get("rule_name") or ""),
        status=str(item.get("status") or ""),
        program=str(item.get("program") or "unknown"),
        category=category,
        priority=priority,
        recommendation=recommendation,
        policyengine_variable=policyengine_variable
        if policyengine_variable is not None
        else item.get("policyengine_variable"),
        policyengine_parameter=policyengine_parameter
        if policyengine_parameter is not None
        else item.get("policyengine_parameter"),
        rationale=item.get("rationale"),
        tested=bool(item.get("tested")),
        test_output_count=int(item.get("test_output_count") or 0),
    )


def _candidate_priority_rank(priority: str) -> int:
    return {"P1": 0, "P2": 1, "P3": 2, "P4": 3}.get(priority, 99)


def _candidate_priority(item: dict[str, Any], default: str) -> str:
    priority = item.get("candidate_priority")
    if priority in {"P1", "P2", "P3", "P4"}:
        return str(priority)
    return default


def _is_policyengine_looking_rule_name(rule_name: str) -> bool:
    lowered = rule_name.lower()
    return lowered.startswith(("snap_", "is_snap_", "meets_snap_")) or any(
        token in lowered
        for token in (
            "tax",
            "deduction",
            "credit",
            "income",
            "allotment",
            "eligible",
            "eligibility",
            "threshold",
            "limit",
            "rate",
        )
    )


def _load_policyengine_variable_names() -> set[str] | None:
    try:
        from policyengine_us import CountryTaxBenefitSystem
    except Exception:
        return None
    try:
        return set(CountryTaxBenefitSystem().variables)
    except Exception:
        return None


def _load_rulespec_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return None
    return payload


def _rulespec_test_output_counts(path: Path) -> Counter[str]:
    """Count canonical output IDs emitted by the companion `.test.yaml` file."""
    test_path = path.with_name(path.stem + ".test.yaml")
    if not test_path.exists():
        return Counter()
    try:
        payload = yaml.safe_load(test_path.read_text(encoding="utf-8")) or []
    except (OSError, yaml.YAMLError, ValueError):
        return Counter()
    if isinstance(payload, dict):
        cases = payload.get("cases", payload.get("tests", []))
    else:
        cases = payload
    output_counts: Counter[str] = Counter()
    if not isinstance(cases, list):
        return output_counts
    for case in cases:
        if not isinstance(case, dict):
            continue
        outputs = case.get("output", case.get("expect"))
        if not isinstance(outputs, dict):
            continue
        output_counts.update(str(key) for key in outputs)
    return output_counts


def _mapping_test_output_count(
    legal_id: str,
    *,
    mapping: PolicyEngineMapping | None,
    test_output_counts: Counter[str],
) -> int:
    count = test_output_counts.get(legal_id, 0)
    if mapping is None:
        return count
    return count + sum(test_output_counts.get(alias, 0) for alias in mapping.aliases)


def _canonical_rulespec_legal_id(
    *,
    prefix: str,
    content_dir: Path,
    rulespec_file: Path,
    rule_name: str,
) -> str:
    """Derive an output's canonical legal ID from its jurisdiction content root.

    ``content_dir`` is the jurisdiction's content root (a ``rulespec-<prefix>``
    checkout root in the legacy layout, or the ``<prefix>`` directory inside a
    country monorepo). Paths are taken relative to that root so both layouts
    yield identical IDs: ``<rulespec-us>/us-al/policies/X.yaml`` and
    ``<rulespec-us-al>/policies/X.yaml`` both produce ``us-al:policies/X#name``.
    """
    relative = rulespec_file.relative_to(content_dir)
    if relative.suffix in {".yaml", ".yml"}:
        relative = relative.with_suffix("")
    return f"{prefix}:{relative.as_posix()}#{rule_name}"


def _country_from_rulespec_prefix(prefix: str) -> str:
    """Infer PolicyEngine country from a RuleSpec repository prefix."""
    return prefix.split("-", 1)[0]


def _display_file_path(rulespec_file: Path, root: Path) -> str:
    """Return a file path relative to the workspace ``root`` for reporting.

    Prefers the unresolved path so a file reached through a sibling-checkout
    symlink keeps its symlink-name prefix (``rulespec-us/us-al/...`` rather
    than ``_axiom/rulespec-us/us-al/...``); CI matches changed files against
    ``<consumer-repo-name>/<path>`` keys built from that symlink name.
    """
    for candidate in (rulespec_file, rulespec_file.resolve()):
        try:
            return str(candidate.relative_to(root))
        except ValueError:
            continue
    return str(rulespec_file)


def _coverage_item_from_mapping(
    *,
    legal_id: str,
    repo_name: str,
    root: Path,
    rulespec_file: Path,
    rule_name: str,
    rule: dict[str, Any],
    kind: str,
    mapping: PolicyEngineMapping | None,
    test_output_count: int,
) -> PolicyEngineCoverageItem:
    file_path = _display_file_path(rulespec_file, root)
    if mapping is None:
        program = _infer_program_from_legal_id(legal_id, rule_name=rule_name)
        if _is_interval_bound_helper_rule(rule_name, rule):
            return PolicyEngineCoverageItem(
                legal_id=legal_id,
                repo=repo_name,
                file=file_path,
                rule_name=rule_name,
                kind=kind,
                status="known_not_comparable",
                program=program,
                mapping_type="not_comparable",
                rationale=(
                    "Indexed interval-bound helper used to select or interpolate "
                    "a source table row; PolicyEngine generally stores final "
                    "parameter scales or computed outputs, not these generated "
                    "RuleSpec row-bound helpers as one-to-one oracle targets."
                ),
                candidate_priority="P4",
                tested=test_output_count > 0,
                test_output_count=test_output_count,
            )
        status = "unmapped"
        mapping_type = None
    elif mapping.comparable:
        status = "comparable"
        program = mapping.program or _infer_program_from_legal_id(
            legal_id, rule_name=rule_name
        )
        mapping_type = mapping.mapping_type
    else:
        status = "known_not_comparable"
        program = mapping.program or _infer_program_from_legal_id(
            legal_id, rule_name=rule_name
        )
        mapping_type = mapping.mapping_type

    return PolicyEngineCoverageItem(
        legal_id=legal_id,
        repo=repo_name,
        file=file_path,
        rule_name=rule_name,
        kind=kind,
        status=status,
        program=program,
        mapping_type=mapping_type,
        policyengine_variable=mapping.policyengine_variable if mapping else None,
        policyengine_parameter=mapping.policyengine_parameter if mapping else None,
        rationale=mapping.rationale if mapping else None,
        candidate_priority=mapping.candidate_priority if mapping else None,
        tested=test_output_count > 0,
        test_output_count=test_output_count,
    )


def _is_interval_bound_helper_rule(rule_name: str, rule: dict[str, Any]) -> bool:
    """Identify generated indexed table-bound helpers that are not PE outputs."""
    if str(rule.get("kind") or "").strip() != "parameter":
        return False
    if not rule.get("indexed_by"):
        return False
    return bool(_INTERVAL_BOUND_HELPER_RE.search(rule_name))


def _program_filter_values(program: str) -> frozenset[str]:
    if program == "health":
        return _HEALTH_PROGRAMS | {"health"}
    return frozenset({program})


def _infer_program_from_legal_id(legal_id: str, *, rule_name: str = "") -> str:
    lowered = legal_id.lower()
    rule = rule_name.lower()
    if lowered.startswith(
        (
            "uk:statutes/ukpga/2007/3/23",
            "uk:statutes/ukpga/2007/3/35",
        )
    ):
        return "tax"
    if lowered.startswith("uk:regulations/uksi/2006/965/2"):
        return "child_benefit"
    if lowered.startswith("uk:policies/govuk/child-benefit"):
        return "child_benefit"
    if lowered.startswith("uk:policies/govuk/state-pension"):
        return "state_pension"
    if lowered.startswith("uk:regulations/uksi/2002/1792/6"):
        return "pension_credit"
    if lowered.startswith("uk:regulations/uksi/2013/376/36"):
        return "universal_credit"
    if _is_combined_health_source(lowered):
        return _infer_health_program_from_rule_name(rule)
    if _PROGRAM_TOKEN_RE["medicaid"].search(lowered):
        return "medicaid"
    if _PROGRAM_TOKEN_RE["chip"].search(lowered):
        return "chip"
    if _PROGRAM_TOKEN_RE["aca"].search(lowered) or "premium_tax_credit" in lowered:
        return "aca_ptc"
    if lowered.startswith("us:statutes/26/36b"):
        return "aca_ptc"
    if "snap" in lowered:
        return "snap"
    if (
        lowered.startswith("us:statutes/26/")
        or lowered.startswith("us-co:statutes/39/")
        or lowered.startswith("us:policies/irs/")
        or lowered.startswith("us:policies/ssa/")
    ):
        return "tax"
    return "unknown"


def _is_combined_health_source(lowered_legal_id: str) -> bool:
    return bool(
        "medicaid-chip" in lowered_legal_id
        or "medicaid_chip" in lowered_legal_id
        or (
            _PROGRAM_TOKEN_RE["medicaid"].search(lowered_legal_id)
            and _PROGRAM_TOKEN_RE["chip"].search(lowered_legal_id)
            and ("bhp" in lowered_legal_id or "eligibility" in lowered_legal_id)
        )
    )


def _infer_health_program_from_rule_name(rule_name: str) -> str:
    if _PROGRAM_TOKEN_RE["chip"].search(rule_name):
        return "chip"
    if _PROGRAM_TOKEN_RE["medicaid"].search(rule_name):
        return "medicaid"
    if _PROGRAM_TOKEN_RE["aca"].search(rule_name) or "premium_tax_credit" in rule_name:
        return "aca_ptc"
    return "health"
