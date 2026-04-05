"""Autoresearch-style pilot utilities for harness tuning.

This module defines a narrow outer-loop optimization surface:
- one editable prompt file
- one frozen benchmark set
- one scalar score

It is designed to match Karpathy-style autoresearch loops more closely than a
general agent-harness framework.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .eval_prompt_surface import AUTOAGENT_PILOT_EDITABLE_FILES

AUTORESEARCH_PROGRAM_PATH = "autoresearch/program.md"
AUTORESEARCH_PILOT_MANIFESTS = (
    "benchmarks/uk_wave18_remaining_repair.yaml",
    "benchmarks/uk_wave19_failure_repair.yaml",
    "benchmarks/uk_wave19_branch_conjunction_repair.yaml",
)
LEGISLATION_CACHE_DIR_NAMES = ("_legislation_gov_uk", "_legislation_gov_uk_cache")


def autorac_repo_root() -> Path:
    """Return the repository root for the current autorac checkout."""
    return Path(__file__).resolve().parents[3]


def program_path(repo_root: Path | None = None) -> Path:
    """Resolve the autoresearch program file."""
    root = repo_root or autorac_repo_root()
    return (root / AUTORESEARCH_PROGRAM_PATH).resolve()


def shared_legislation_cache_root() -> Path:
    """Return the persistent local cache root for legislation.gov.uk payloads."""
    override = os.getenv("AUTORAC_SHARED_LEGISLATION_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "tmp" / "autorac-shared-legislation-cache").resolve()


def pilot_manifest_paths(repo_root: Path | None = None) -> list[Path]:
    """Resolve the frozen pilot manifest set."""
    root = repo_root or autorac_repo_root()
    return [(root / rel).resolve() for rel in AUTORESEARCH_PILOT_MANIFESTS]


def pilot_editable_paths(repo_root: Path | None = None) -> list[Path]:
    """Resolve the files the pilot is allowed to edit."""
    root = repo_root or autorac_repo_root()
    return [(root / rel).resolve() for rel in AUTOAGENT_PILOT_EDITABLE_FILES]


def score_readiness_summary(summary: dict) -> float:
    """Score one runner summary for outer-loop prompt tuning.

    This intentionally weights readiness much more heavily than cost. Cost only
    breaks ties once the deterministic and semantic gates are green.
    """
    ready = bool(summary.get("ready"))
    compile_pass_rate = float(summary.get("compile_pass_rate") or 0.0)
    ci_pass_rate = float(summary.get("ci_pass_rate") or 0.0)
    zero_ungrounded_rate = float(summary.get("zero_ungrounded_rate") or 0.0)
    generalist_review_pass_rate = float(
        summary.get("generalist_review_pass_rate") or 0.0
    )
    mean_estimated_cost_usd = float(summary.get("mean_estimated_cost_usd") or 0.0)

    score = 100.0 if ready else 0.0
    score -= 40.0 * (1.0 - compile_pass_rate)
    score -= 30.0 * (1.0 - ci_pass_rate)
    score -= 30.0 * (1.0 - generalist_review_pass_rate)
    score -= 10.0 * (1.0 - zero_ungrounded_rate)
    score -= 0.1 * mean_estimated_cost_usd
    return round(score, 6)


def extract_primary_runner_summary(payload: dict) -> tuple[str, dict]:
    """Return the first readiness summary from a suite payload."""
    readiness = payload.get("readiness") or {}
    if not readiness:
        raise ValueError("Missing readiness data in eval-suite payload")
    runner, summary = next(iter(readiness.items()))
    return str(runner), dict(summary)


def load_suite_summary(path: Path) -> dict:
    """Load a persisted eval-suite summary or full result payload."""
    payload = json.loads(path.read_text())
    if "readiness" in payload:
        return payload
    raise ValueError(f"Unsupported eval-suite payload: {path}")


def _candidate_legislation_cache_sources(
    *,
    dir_name: str,
    output_root: Path,
    shared_root: Path,
    search_root: Path | None = None,
) -> list[Path]:
    """Return local cache directories worth copying into a pilot run root."""
    candidates: list[Path] = []
    shared_dir = shared_root / dir_name
    if shared_dir.exists():
        candidates.append(shared_dir)

    search_base = (search_root or (Path.home() / "tmp")).resolve()
    if search_base.exists():
        run_dirs = sorted(
            (
                path
                for path in search_base.glob("autorac-*")
                if path.is_dir() and path.resolve() != output_root.resolve()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs:
            candidate = run_dir / dir_name
            if candidate.exists():
                candidates.append(candidate)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _merge_tree(source: Path, destination: Path) -> int:
    """Copy files from source into destination when they are missing."""
    copied = 0
    if not source.exists():
        return copied
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        if source_path.is_dir():
            continue
        relative = source_path.relative_to(source)
        target_path = destination / relative
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def seed_legislation_cache(
    output_root: Path,
    *,
    shared_root: Path | None = None,
    search_root: Path | None = None,
) -> dict[str, int]:
    """Prepopulate a pilot run root with reusable legislation.gov.uk caches."""
    root = output_root.resolve()
    cache_root = (shared_root or shared_legislation_cache_root()).resolve()
    copied: dict[str, int] = {}

    for dir_name in LEGISLATION_CACHE_DIR_NAMES:
        target_dir = root / dir_name
        copied[dir_name] = 0
        for candidate in _candidate_legislation_cache_sources(
            dir_name=dir_name,
            output_root=root,
            shared_root=cache_root,
            search_root=search_root,
        ):
            copied[dir_name] += _merge_tree(candidate, target_dir)
            if any(target_dir.rglob("*")):
                break
    return copied


def sync_legislation_cache(
    output_root: Path,
    *,
    shared_root: Path | None = None,
) -> dict[str, int]:
    """Promote any newly fetched legislation.gov.uk files into the shared cache."""
    root = output_root.resolve()
    cache_root = (shared_root or shared_legislation_cache_root()).resolve()
    synced: dict[str, int] = {}
    for dir_name in LEGISLATION_CACHE_DIR_NAMES:
        source_dir = root / dir_name
        target_dir = cache_root / dir_name
        synced[dir_name] = _merge_tree(source_dir, target_dir)
    return synced
