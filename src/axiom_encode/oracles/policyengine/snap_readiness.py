"""SNAP encoding readiness report for rulespec-us-* repositories."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .ecps_snap import JURISDICTION_CONFIGS

EXECUTABLE_RULE_KINDS = {"parameter", "derived"}
SNAP_MARKERS = (
    "snap",
    "supplemental nutrition assistance",
    "food stamp",
    "food stamps",
)


@dataclass(frozen=True)
class SnapReadinessItem:
    jurisdiction: str
    repo: str
    repo_path: str
    rulespec_files: int
    companion_test_files: int
    executable_outputs: int
    corpus_snap_provisions: int
    policyengine_ecps_configured: bool
    program_module: str | None
    program_module_exists: bool
    status: str
    blockers: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "repo": self.repo,
            "repo_path": self.repo_path,
            "rulespec_files": self.rulespec_files,
            "companion_test_files": self.companion_test_files,
            "executable_outputs": self.executable_outputs,
            "corpus_snap_provisions": self.corpus_snap_provisions,
            "policyengine_ecps_configured": self.policyengine_ecps_configured,
            "program_module": self.program_module,
            "program_module_exists": self.program_module_exists,
            "status": self.status,
            "blockers": list(self.blockers),
        }


def build_snap_readiness_report(
    root: Path,
    *,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """Report whether each state rules repo is ready for SNAP encoding/oracles."""
    root = Path(root).expanduser().resolve()
    resolved_corpus_root = (
        Path(corpus_root).expanduser().resolve()
        if corpus_root
        else root / "axiom-corpus"
    )
    items = [
        build_snap_readiness_item(repo, corpus_root=resolved_corpus_root)
        for repo in _iter_state_rules_repos(root)
    ]
    status_counts = Counter(item.status for item in items)
    return {
        "program": "snap",
        "root": str(root),
        "corpus_root": str(resolved_corpus_root),
        "total_repos": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "items": [item.as_dict() for item in items],
    }


def build_snap_readiness_item(
    repo: Path,
    *,
    corpus_root: Path,
) -> SnapReadinessItem:
    repo = Path(repo).expanduser().resolve()
    jurisdiction = repo.name.removeprefix("rulespec-")
    rulespec_payloads = [
        (rulespec_file, payload)
        for rulespec_file, payload in _iter_snap_rulespec_payloads(repo)
    ]
    rulespec_files = len(rulespec_payloads)
    executable_outputs = sum(
        _count_executable_outputs(payload) for _, payload in rulespec_payloads
    )
    companion_test_files = sum(
        1
        for rulespec_file, _ in rulespec_payloads
        if rulespec_file.with_name(
            rulespec_file.name.removesuffix(".yaml").removesuffix(".yml") + ".test.yaml"
        ).exists()
    )
    corpus_snap_provisions = count_snap_corpus_provisions(
        corpus_root,
        jurisdiction=jurisdiction,
    )
    ecps_config = JURISDICTION_CONFIGS.get(jurisdiction)
    program_module = (
        ecps_config.program_relative_path.as_posix() if ecps_config else None
    )
    program_module_exists = (
        bool(ecps_config) and (repo / ecps_config.program_relative_path).exists()
    )
    status, blockers = _classify_status(
        rulespec_files=rulespec_files,
        corpus_snap_provisions=corpus_snap_provisions,
        ecps_configured=ecps_config is not None,
        program_module_exists=program_module_exists,
    )
    return SnapReadinessItem(
        jurisdiction=jurisdiction,
        repo=repo.name,
        repo_path=str(repo),
        rulespec_files=rulespec_files,
        companion_test_files=companion_test_files,
        executable_outputs=executable_outputs,
        corpus_snap_provisions=corpus_snap_provisions,
        policyengine_ecps_configured=ecps_config is not None,
        program_module=program_module,
        program_module_exists=program_module_exists,
        status=status,
        blockers=tuple(blockers),
    )


def count_snap_corpus_provisions(corpus_root: Path, *, jurisdiction: str) -> int:
    """Count ingested corpus.provisions rows that appear to be SNAP sources."""
    provisions_root = _corpus_provisions_root(corpus_root)
    if provisions_root is None:
        return 0
    jurisdiction_root = provisions_root / jurisdiction
    if not jurisdiction_root.exists():
        return 0
    count = 0
    for provision_file in sorted(jurisdiction_root.rglob("*.jsonl")):
        for row in _iter_jsonl_rows(provision_file):
            if _corpus_row_is_snap(row):
                count += 1
    return count


def _iter_state_rules_repos(root: Path) -> list[Path]:
    return [
        repo
        for repo in sorted(root.glob("rulespec-us-*"))
        if repo.is_dir() and repo.name != "rulespec-us"
    ]


def _iter_snap_rulespec_payloads(repo: Path) -> list[tuple[Path, dict[str, Any]]]:
    payloads: list[tuple[Path, dict[str, Any]]] = []
    for rulespec_file in sorted(repo.rglob("*.y*ml")):
        if rulespec_file.name.endswith(".test.yaml"):
            continue
        if ".axiom" in rulespec_file.relative_to(repo).parts:
            continue
        payload = _load_yaml_mapping(rulespec_file)
        if payload and _is_snap_rulespec(rulespec_file, repo=repo, payload=payload):
            payloads.append((rulespec_file, payload))
    return payloads


def _load_yaml_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text())
    except OSError, yaml.YAMLError:
        return None
    return payload if isinstance(payload, dict) else None


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


def _corpus_provisions_root(corpus_root: Path) -> Path | None:
    root = Path(corpus_root).expanduser()
    candidates = (
        root / "data" / "corpus" / "provisions",
        root / "data" / "corpus",
        root,
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _iter_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _corpus_row_is_snap(row: dict[str, Any]) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    program = str(metadata.get("program") or "").lower()
    if program == "snap":
        return True
    haystack = " ".join(
        str(value or "")
        for value in (
            row.get("citation_path"),
            row.get("citation_label"),
            row.get("heading"),
            row.get("body"),
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
    ecps_configured: bool,
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
    if not ecps_configured:
        blockers.append("missing PolicyEngine ECPS jurisdiction config")
        return "needs_ecps_config", blockers
    if not program_module_exists:
        blockers.append("missing configured SNAP program module")
        return "missing_program_module", blockers
    return "ecps_ready", blockers
