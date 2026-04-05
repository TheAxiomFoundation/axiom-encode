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
from pathlib import Path

from .eval_prompt_surface import AUTOAGENT_PILOT_EDITABLE_FILES

AUTORESEARCH_PROGRAM_PATH = "autoresearch/program.md"
AUTORESEARCH_PILOT_MANIFESTS = (
    "benchmarks/uk_wave18_remaining_repair.yaml",
    "benchmarks/uk_wave19_failure_repair.yaml",
    "benchmarks/uk_wave19_branch_conjunction_repair.yaml",
)


def autorac_repo_root() -> Path:
    """Return the repository root for the current autorac checkout."""
    return Path(__file__).resolve().parents[3]


def program_path(repo_root: Path | None = None) -> Path:
    """Resolve the autoresearch program file."""
    root = repo_root or autorac_repo_root()
    return (root / AUTORESEARCH_PROGRAM_PATH).resolve()


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
