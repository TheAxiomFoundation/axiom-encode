"""Failure triage (docs/design/encode-failure-triage.md).

Stage 3, PR 1: the deterministic evidence bundle and its five probes. No
model is called anywhere in this package.
"""

from .evidence import (
    EVIDENCE_SCHEMA,
    EvidenceInputError,
    build_evidence,
    evidence_sha256,
    load_any,
    load_local_encode,
    load_local_run,
    load_workflow_bundle,
    resolve_ref,
)
from .material import AttemptMaterial, CandidateMaterial, TriageMaterial

__all__ = [
    "EVIDENCE_SCHEMA",
    "AttemptMaterial",
    "CandidateMaterial",
    "EvidenceInputError",
    "TriageMaterial",
    "build_evidence",
    "evidence_sha256",
    "load_any",
    "load_local_encode",
    "load_local_run",
    "load_workflow_bundle",
    "resolve_ref",
]
