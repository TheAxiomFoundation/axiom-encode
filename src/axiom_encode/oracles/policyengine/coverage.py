"""PolicyEngine oracle coverage reports for RuleSpec repositories."""

from __future__ import annotations

import ast
import importlib.util
import os
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
PROGRAM_SURFACE_STATUSES = {
    "deferred_jurisdiction",
    "input_only",
    "known_not_comparable",
    "out_of_scope",
    "pe_in_progress",
    "pending_oracle_mapping",
    "pending_rulespec_encoding",
    "pending_source_ingestion",
    "wired",
}
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


@dataclass(frozen=True)
class PolicyEngineProgramSurfaceItem:
    """One PolicyEngine program variable classified against Axiom wiring."""

    country: str
    program_id: str
    program_name: str
    category: str
    policyengine_status: str
    coverage: str
    variable: str
    axiom_status: str
    source_type: str = "program"
    agency: str | None = None
    state: str | None = None
    priority: str | None = None
    rationale: str | None = None
    mapping_count: int = 0
    comparable_mapping_count: int = 0
    legal_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, str | int | list[str] | None]:
        return {
            "country": self.country,
            "program_id": self.program_id,
            "program_name": self.program_name,
            "category": self.category,
            "policyengine_status": self.policyengine_status,
            "coverage": self.coverage,
            "variable": self.variable,
            "axiom_status": self.axiom_status,
            "source_type": self.source_type,
            "agency": self.agency,
            "state": self.state,
            "priority": self.priority,
            "rationale": self.rationale,
            "mapping_count": self.mapping_count,
            "comparable_mapping_count": self.comparable_mapping_count,
            "legal_ids": list(self.legal_ids),
        }


def build_policyengine_coverage_report(
    root: Path,
    *,
    program: str | None = None,
    include_program_surfaces: bool = False,
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

    report = {
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
    if include_program_surfaces:
        report["program_surfaces"] = build_policyengine_program_surface_report(
            program=program,
            registry=registry,
        )
    return report


def build_policyengine_program_surface_report(
    *,
    country: str = "us",
    program: str | None = None,
    manifest_path: Path | None = None,
    registry=None,
) -> dict[str, Any]:
    """Classify PolicyEngine program variables against the Axiom registry."""
    registry = registry or load_policyengine_registry()
    payload = _load_policyengine_program_surface_manifest(
        country=country,
        manifest_path=manifest_path,
    )
    surfaces = [
        _program_surface_item_from_payload(raw_surface, registry=registry)
        for raw_surface in payload["surfaces"]
    ]
    if program:
        surfaces = [
            surface
            for surface in surfaces
            if _program_surface_matches_filter(surface, program)
        ]
    status_counts = Counter(surface.axiom_status for surface in surfaces)
    priority_counts = Counter(
        surface.priority for surface in surfaces if surface.priority
    )
    unwired_statuses = {
        "deferred_jurisdiction",
        "pending_oracle_mapping",
        "pending_rulespec_encoding",
        "pending_source_ingestion",
    }
    pending_surfaces = [
        surface for surface in surfaces if surface.axiom_status in unwired_statuses
    ]
    return {
        "oracle": "policyengine",
        "country": country,
        "program": program,
        "source": payload.get("source", {}),
        "total_surfaces": len(surfaces),
        "status_counts": dict(sorted(status_counts.items())),
        "priority_counts": dict(sorted(priority_counts.items())),
        "pending_surfaces": len(pending_surfaces),
        "items": [surface.as_dict() for surface in surfaces],
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
    items.extend(_iter_program_spec_coverage_items(root, registry))
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


def _iter_program_spec_coverage_items(
    root: Path,
    registry,
) -> list[PolicyEngineCoverageItem]:
    """Enumerate outputs declared by monorepo-native ``programs/`` specs."""
    items: list[PolicyEngineCoverageItem] = []
    for programs_dir, checkout_dir in _program_spec_dirs(root):
        repo_name = canonical_rulespec_repo_name(checkout_dir) or checkout_dir.name
        for spec_file in sorted(programs_dir.glob("*/*/*.y*ml")):
            items.extend(
                _program_spec_coverage_items_for_file(
                    root=root,
                    repo_name=repo_name,
                    spec_file=spec_file,
                    registry=registry,
                )
            )
    return items


def _program_spec_dirs(root: Path) -> list[tuple[Path, Path]]:
    """Return ``(programs_dir, checkout_dir)`` pairs under a checkout/workspace."""
    candidates: list[tuple[Path, Path]] = []
    direct = root / "programs"
    if direct.is_dir():
        candidates.append((direct, root))
    for checkout in sorted(root.glob("rulespec-*")):
        programs_dir = checkout / "programs"
        if programs_dir.is_dir():
            candidates.append((programs_dir, checkout))

    seen: set[Path] = set()
    out: list[tuple[Path, Path]] = []
    for programs_dir, checkout_dir in candidates:
        resolved = programs_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append((programs_dir, checkout_dir))
    return out


def _program_spec_coverage_items_for_file(
    *,
    root: Path,
    repo_name: str,
    spec_file: Path,
    registry,
) -> list[PolicyEngineCoverageItem]:
    payload = _load_program_spec_payload(spec_file)
    if not payload:
        return []
    raw_program = payload.get("program")
    raw_outputs = payload.get("outputs")
    if not isinstance(raw_program, str) or not raw_program.strip():
        return []
    if not isinstance(raw_outputs, list):
        return []
    program_id = raw_program.strip()
    prefix, _, program_path = program_id.partition("/")
    if not prefix or not program_path:
        return []
    period_stem = spec_file.with_suffix("").name
    spec_legal_path = f"programs/{program_path}/{period_stem}"
    country = _country_from_rulespec_prefix(prefix)
    file_path = _display_file_path(spec_file, root)
    items: list[PolicyEngineCoverageItem] = []
    for raw_output in raw_outputs:
        if not isinstance(raw_output, str) or not raw_output.strip():
            continue
        output_name = raw_output.strip()
        legal_id = f"{prefix}:{spec_legal_path}#{output_name}"
        mapping = registry.mapping_for_legal_id(legal_id, country=country)
        items.append(
            _coverage_item_from_program_spec_mapping(
                legal_id=legal_id,
                repo_name=repo_name,
                file_path=file_path,
                rule_name=output_name,
                program=program_path,
                mapping=mapping,
            )
        )
    return items


def _coverage_item_from_program_spec_mapping(
    *,
    legal_id: str,
    repo_name: str,
    file_path: str,
    rule_name: str,
    program: str,
    mapping: PolicyEngineMapping | None,
) -> PolicyEngineCoverageItem:
    if mapping is None:
        status = "unmapped"
        mapping_type = None
        report_program = program
    elif mapping.comparable:
        status = "comparable"
        mapping_type = mapping.mapping_type
        report_program = mapping.program or program
    else:
        status = "known_not_comparable"
        mapping_type = mapping.mapping_type
        report_program = mapping.program or program

    return PolicyEngineCoverageItem(
        legal_id=legal_id,
        repo=repo_name,
        file=file_path,
        rule_name=rule_name,
        kind="program_output",
        status=status,
        program=report_program,
        mapping_type=mapping_type,
        policyengine_variable=mapping.policyengine_variable if mapping else None,
        policyengine_parameter=mapping.policyengine_parameter if mapping else None,
        rationale=mapping.rationale if mapping else None,
        candidate_priority=mapping.candidate_priority if mapping else None,
        tested=False,
        test_output_count=0,
    )


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
    source_names = _load_policyengine_variable_names_from_source()
    if source_names is not None:
        return source_names
    try:
        from policyengine_us import CountryTaxBenefitSystem
    except Exception:
        return None
    try:
        return set(CountryTaxBenefitSystem().variables)
    except Exception:
        return None


def _load_policyengine_variable_names_from_source() -> set[str] | None:
    """Load PolicyEngine-US variable names without importing its model package.

    Candidate triage only needs to know whether a RuleSpec output name exists
    as a PolicyEngine variable. Importing ``policyengine_us`` can execute the
    full tax-benefit system and fail before triage starts, so prefer a source
    tree scan when the package or a local checkout is available.
    """
    source_dirs = _policyengine_variable_source_dirs()
    if not source_dirs:
        return None
    names: set[str] = set()
    for source_dir in source_dirs:
        for source_file in sorted(source_dir.rglob("*.py")):
            names.update(_policyengine_variable_names_from_file(source_file))
    return names


def _policyengine_variable_source_dirs() -> list[Path]:
    source_dirs: list[Path] = []
    for env_name in ("AXIOM_POLICYENGINE_US_ROOT", "POLICYENGINE_US_ROOT"):
        raw_path = os.environ.get(env_name)
        if raw_path:
            source_dirs.extend(_policyengine_variable_dirs_for_path(Path(raw_path)))

    spec = importlib.util.find_spec("policyengine_us")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            source_dirs.extend(_policyengine_variable_dirs_for_path(Path(location)))

    seen: set[Path] = set()
    out: list[Path] = []
    for source_dir in source_dirs:
        resolved = source_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(source_dir)
    return out


def _policyengine_variable_dirs_for_path(path: Path) -> list[Path]:
    candidates = (
        path,
        path / "variables",
        path / "policyengine_us" / "variables",
    )
    return [
        candidate
        for candidate in candidates
        if candidate.is_dir() and candidate.name == "variables"
    ]


def _policyengine_variable_names_from_file(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(_is_policyengine_variable_base(base) for base in node.bases):
            names.add(node.name)
    return names


def _is_policyengine_variable_base(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Variable"
    if isinstance(node, ast.Attribute):
        return node.attr == "Variable"
    return False


def _load_policyengine_program_surface_manifest(
    *,
    country: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    path = (
        manifest_path
        if manifest_path is not None
        else Path(__file__).with_name("program_surfaces") / f"{country}.yaml"
    )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(
            f"Unable to load PolicyEngine surface manifest {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"PolicyEngine surface manifest must be an object: {path}")
    raw_surfaces = payload.get("surfaces")
    if not isinstance(raw_surfaces, list):
        raise ValueError(
            f"PolicyEngine surface manifest surfaces must be a list: {path}"
        )
    for index, raw_surface in enumerate(raw_surfaces):
        if not isinstance(raw_surface, dict):
            raise ValueError(
                f"PolicyEngine surface manifest entry {index} must be an object: {path}"
            )
        missing = [
            key
            for key in (
                "country",
                "program_id",
                "program_name",
                "category",
                "policyengine_status",
                "coverage",
                "variable",
                "axiom_status",
            )
            if not raw_surface.get(key)
        ]
        if missing:
            raise ValueError(
                f"PolicyEngine surface manifest entry {index} missing "
                f"{', '.join(missing)}: {path}"
            )
        status = str(raw_surface.get("axiom_status"))
        if status not in PROGRAM_SURFACE_STATUSES - {"wired"}:
            raise ValueError(
                f"Unsupported PolicyEngine surface status for "
                f"{raw_surface.get('variable')}: {status}"
            )
    return payload


def _program_surface_item_from_payload(
    payload: dict[str, Any],
    *,
    registry,
) -> PolicyEngineProgramSurfaceItem:
    variable = str(payload["variable"])
    country = str(payload.get("country") or "us")
    mappings = registry.mappings_for_policyengine_variable(variable, country=country)
    comparable_mapping_count = sum(1 for mapping in mappings if mapping.comparable)
    manifest_status = str(payload["axiom_status"])
    if comparable_mapping_count:
        axiom_status = "wired"
    elif mappings and manifest_status in {
        "pending_oracle_mapping",
        "pending_rulespec_encoding",
    }:
        axiom_status = "known_not_comparable"
    else:
        axiom_status = manifest_status
    return PolicyEngineProgramSurfaceItem(
        country=country,
        program_id=str(payload["program_id"]),
        program_name=str(payload["program_name"]),
        category=str(payload["category"]),
        policyengine_status=str(payload["policyengine_status"]),
        coverage=str(payload["coverage"]),
        variable=variable,
        axiom_status=axiom_status,
        source_type=str(payload.get("source_type") or "program"),
        agency=payload.get("agency"),
        state=payload.get("state"),
        priority=payload.get("priority"),
        rationale=payload.get("rationale"),
        mapping_count=len(mappings),
        comparable_mapping_count=comparable_mapping_count,
        legal_ids=tuple(mapping.legal_id for mapping in mappings),
    )


def _program_surface_matches_filter(
    surface: PolicyEngineProgramSurfaceItem,
    program: str,
) -> bool:
    programs = _program_filter_values(program)
    normalized = program.lower()
    if surface.program_id in programs or surface.variable in programs:
        return True
    if normalized == "tax" and surface.category.lower() == "taxes":
        return True
    if normalized == "health" and surface.category.lower() == "healthcare":
        return True
    if normalized in {"ccap", "ccdf", "child_care"}:
        return surface.program_id in {"ccdf", "ne_childcare"} or any(
            token in surface.variable
            for token in ("ccap", "ccdf", "child_care", "childcare", "caps")
        )
    return False


def _load_rulespec_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
        return None
    return payload


def _load_program_spec_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


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
    if lowered.startswith("nz:statutes/income_tax/"):
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
    if lowered.startswith(("us:statutes/42/1382", "us:statutes/42/1382f")):
        return "ssi"
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
