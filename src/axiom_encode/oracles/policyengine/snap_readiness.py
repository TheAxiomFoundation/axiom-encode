"""SNAP encoding readiness report for the canonical rulespec-us monorepo."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from axiom_oracles.bridges.snap_populace import JURISDICTION_CONFIGS

from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILE_SUFFIX,
    RULESPEC_TEST_FILE_SUFFIX,
)
from axiom_encode.corpus_resolver import (
    ActiveCorpusBodyRow,
    LocalCorpusRelease,
    iter_active_local_corpus_rows,
)
from axiom_encode.repo_routing import (
    canonical_rulespec_root_identity,
    is_composition_policy_repo_root,
)
from axiom_encode.toolchain import load_rulespec_local_corpus_release

EXECUTABLE_RULE_KINDS = {"parameter", "derived", "derived_relation"}
SNAP_MARKERS = (
    "snap",
    "supplemental nutrition assistance",
    "food stamp",
    "food stamps",
)
STATE_MODULE_PATTERN = re.compile(r"us-[a-z]{2}")


class SnapReadinessConfigurationError(ValueError):
    """The canonical US RuleSpec checkout is absent or malformed."""


@dataclass(frozen=True)
class SnapReadinessItem:
    jurisdiction: str
    module: str
    module_path: str
    rulespec_files: int
    companion_test_files: int
    executable_outputs: int
    corpus_snap_provisions: int
    policyengine_populace_configured: bool
    program_module: str | None
    program_module_exists: bool
    status: str
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "module": self.module,
            "module_path": self.module_path,
            "rulespec_files": self.rulespec_files,
            "companion_test_files": self.companion_test_files,
            "executable_outputs": self.executable_outputs,
            "corpus_snap_provisions": self.corpus_snap_provisions,
            "policyengine_populace_configured": self.policyengine_populace_configured,
            "program_module": self.program_module,
            "program_module_exists": self.program_module_exists,
            "status": self.status,
            "blockers": list(self.blockers),
        }


def build_snap_readiness_report(
    root: Path,
    *,
    corpus_root: Path,
) -> dict[str, Any]:
    """Report whether each state module is ready for SNAP encoding/oracles."""
    rulespec_repo = _resolve_rulespec_us_repo(Path(root).expanduser())
    resolved_corpus_root = Path(corpus_root).expanduser().resolve()
    state_modules = _iter_state_rules_modules(rulespec_repo)
    corpus_release = load_rulespec_local_corpus_release(
        rulespec_repo,
        resolved_corpus_root,
    )
    items = [
        build_snap_readiness_item(module, corpus_release=corpus_release)
        for module in state_modules
    ]
    status_counts = Counter(item.status for item in items)
    return {
        "program": "snap",
        "root": str(rulespec_repo),
        "rulespec_repo": str(rulespec_repo),
        "corpus_root": str(resolved_corpus_root),
        "total_modules": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "items": [item.as_dict() for item in items],
    }


def build_snap_readiness_item(
    module: Path,
    *,
    corpus_release: LocalCorpusRelease,
) -> SnapReadinessItem:
    module = Path(module).expanduser().resolve()
    jurisdiction = module.name
    if STATE_MODULE_PATTERN.fullmatch(jurisdiction) is None:
        raise SnapReadinessConfigurationError(
            f"Invalid rulespec-us state module name: {module}"
        )
    rulespec_repo = module.parent
    rulespec_payloads = [
        (rulespec_file, payload)
        for rulespec_file, payload in _iter_snap_rulespec_payloads(module)
    ]
    rulespec_files = len(rulespec_payloads)
    executable_outputs = sum(
        _count_executable_outputs(payload) for _, payload in rulespec_payloads
    )
    companion_test_files = sum(
        1
        for rulespec_file, _ in rulespec_payloads
        if rulespec_file.with_name(
            rulespec_file.name.removesuffix(".yaml") + ".test.yaml"
        ).exists()
    )
    corpus_snap_provisions = count_snap_corpus_provisions(
        corpus_release,
        jurisdiction=jurisdiction,
    )
    populace_config = JURISDICTION_CONFIGS.get(jurisdiction)
    program_module = (
        populace_config.program_relative_path.as_posix() if populace_config else None
    )
    program_module_exists = (
        bool(populace_config)
        and (module / populace_config.program_relative_path).exists()
    )
    status, blockers = _classify_status(
        rulespec_files=rulespec_files,
        corpus_snap_provisions=corpus_snap_provisions,
        populace_configured=populace_config is not None,
        program_module_exists=program_module_exists,
    )
    return SnapReadinessItem(
        jurisdiction=jurisdiction,
        module=f"{rulespec_repo.name}/{module.name}",
        module_path=str(module),
        rulespec_files=rulespec_files,
        companion_test_files=companion_test_files,
        executable_outputs=executable_outputs,
        corpus_snap_provisions=corpus_snap_provisions,
        policyengine_populace_configured=populace_config is not None,
        program_module=program_module,
        program_module_exists=program_module_exists,
        status=status,
        blockers=tuple(blockers),
    )


def count_snap_corpus_provisions(
    corpus_release: LocalCorpusRelease,
    *,
    jurisdiction: str,
) -> int:
    """Count active, unambiguous corpus rows that appear to be SNAP sources."""
    rows = iter_active_local_corpus_rows(
        corpus_release,
        jurisdiction=jurisdiction,
    )
    return sum(1 for row in rows if _corpus_row_is_snap(row))


def _resolve_rulespec_us_repo(root: Path) -> Path:
    if root.name != "rulespec-us" or not is_composition_policy_repo_root(root):
        raise SnapReadinessConfigurationError(
            f"SNAP readiness requires the exact canonical rulespec-us checkout: {root}"
        )
    return root.resolve(strict=True)


def _iter_state_rules_modules(rulespec_repo: Path) -> list[Path]:
    modules: list[Path] = []
    for candidate in sorted(rulespec_repo.iterdir()):
        if STATE_MODULE_PATTERN.fullmatch(candidate.name) is None:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise SnapReadinessConfigurationError(
                f"rulespec-us state modules must be regular directories: {candidate}"
            )
        resolved = candidate.resolve(strict=True)
        if canonical_rulespec_root_identity(resolved) is None:
            raise SnapReadinessConfigurationError(
                f"Noncanonical rulespec-us state module: {candidate}"
            )
        modules.append(resolved)
    if not modules:
        raise SnapReadinessConfigurationError(
            f"Canonical rulespec-us checkout contains no us-xx state modules: "
            f"{rulespec_repo}"
        )
    return modules


def _iter_snap_rulespec_payloads(repo: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for rulespec_file in sorted(repo.rglob("*")):
        relative = rulespec_file.relative_to(repo)
        if relative.parts and relative.parts[0] == RULESPEC_COMPOSITION_SPEC_ROOT:
            continue
        if rulespec_file.is_symlink():
            raise SnapReadinessConfigurationError(
                f"RuleSpec content must not contain symlinks: {rulespec_file}"
            )
        if not rulespec_file.is_file():
            continue
        if rulespec_file.suffix == ".yml":
            raise SnapReadinessConfigurationError(
                "RuleSpec modules must use the canonical .yaml extension: "
                f"{rulespec_file}"
            )
        if rulespec_file.suffix != RULESPEC_FILE_SUFFIX:
            continue
        if not relative.parts or relative.parts[0] not in RULESPEC_ATOMIC_MODULE_ROOTS:
            raise SnapReadinessConfigurationError(
                "Atomic RuleSpec YAML must be under one canonical module root "
                f"{sorted(RULESPEC_ATOMIC_MODULE_ROOTS)}: {rulespec_file}"
            )
        if rulespec_file.name.endswith(RULESPEC_TEST_FILE_SUFFIX):
            continue
        payload = _load_yaml_mapping(rulespec_file)
        if _is_snap_rulespec(rulespec_file, repo=repo, payload=payload):
            payloads.append((rulespec_file, payload))
    return payloads


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SnapReadinessConfigurationError(
            f"Could not read canonical RuleSpec YAML {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SnapReadinessConfigurationError(
            f"Canonical RuleSpec YAML must be a mapping: {path}"
        )
    return payload


def _is_snap_rulespec(
    rulespec_file: Path,
    *,
    repo: Path,
    payload: dict[str, Any],
) -> bool:
    relative_text = rulespec_file.relative_to(repo).as_posix().lower()
    if "/snap/" in f"/{relative_text}" or "snap" in rulespec_file.stem.lower():
        return True
    imports = payload.get("imports")
    if isinstance(imports, list) and any(
        "snap" in str(item).lower() for item in imports
    ):
        return True
    rules = payload.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and "snap" in str(rule.get("name") or "").lower():
                return True
    return False


def _count_executable_outputs(payload: dict[str, Any]) -> int:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return 0
    count = 0
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("kind") or "").strip() in EXECUTABLE_RULE_KINDS:
            count += 1
    return count


def _corpus_row_is_snap(row: ActiveCorpusBodyRow) -> bool:
    metadata = row.metadata
    program = str(metadata.get("program") or "").lower()
    if program == "snap":
        return True
    haystack = " ".join(
        str(value or "")
        for value in (
            row.row.citation_path,
            row.citation_label,
            row.heading,
            row.body,
            metadata.get("title"),
            metadata.get("document_title"),
            metadata.get("source_authority"),
        )
    ).lower()
    return any(marker in haystack for marker in SNAP_MARKERS)


def _classify_status(
    *,
    rulespec_files: int,
    corpus_snap_provisions: int,
    populace_configured: bool,
    program_module_exists: bool,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if corpus_snap_provisions == 0:
        blockers.append("no SNAP corpus provisions found")
    if rulespec_files == 0:
        if corpus_snap_provisions > 0:
            return "ready_to_encode", blockers
        return "needs_corpus", blockers
    if corpus_snap_provisions == 0:
        return "rules_without_corpus", blockers
    if not populace_configured:
        blockers.append("missing PolicyEngine Populace jurisdiction config")
        return "needs_populace_config", blockers
    if not program_module_exists:
        blockers.append("missing configured SNAP program module")
        return "missing_program_module", blockers
    return "populace_ready", blockers
