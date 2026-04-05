"""Backward-compatible alias for the autoresearch pilot helpers."""

from .autoresearch_pilot import (
    AUTORESEARCH_PILOT_MANIFESTS as AUTOAGENT_PILOT_MANIFESTS,
)
from .autoresearch_pilot import (
    autorac_repo_root,
    extract_primary_runner_summary,
    load_suite_summary,
    pilot_editable_paths,
    pilot_manifest_paths,
    program_path,
    score_readiness_summary,
)

__all__ = [
    "AUTOAGENT_PILOT_MANIFESTS",
    "autorac_repo_root",
    "extract_primary_runner_summary",
    "load_suite_summary",
    "pilot_editable_paths",
    "pilot_manifest_paths",
    "program_path",
    "score_readiness_summary",
]
