"""The normalized record of one failed run that the evidence builder reads.

Loaders in :mod:`axiom_encode.triage.evidence` fill it from a local encode,
a local pilot run directory or a workflow failure bundle. Every field is
plain data read from the run's own files; nothing here is inferred.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateMaterial:
    """The best (final) rejected candidate of the run."""

    module_path: str
    rulespec: str
    tests: str | None = None
    runner: str | None = None
    sibling_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptMaterial:
    """One generation attempt; ``issues`` is ``None`` when none were recorded."""

    attempt: int
    model: str | None = None
    error: str | None = None
    issues: tuple[str, ...] | None = None
    issue_count: int | None = None


@dataclass(frozen=True)
class TriageMaterial:
    input_kind: str
    input_name: str
    citation: str | None
    run_id: str | None = None
    lane: str | None = None
    policy_prefix: str | None = None
    rulespec_ref: str | None = None
    rulespec_ref_source: str | None = None
    corpus_ref: str | None = None
    encoder_version: str | None = None
    encoder_commit: str | None = None
    outcome: Mapping[str, Any] = field(default_factory=dict)
    source_text: str | None = None
    source_attestation: Mapping[str, Any] | None = None
    provision_metadata: Mapping[str, Any] | None = None
    # The original request when it was routed to its principal provision.
    amendment_request: str | None = None
    amendment_documents: tuple[str, ...] = ()
    dropped_amendment_documents: tuple[str, ...] = ()
    review_findings: tuple[str, ...] = ()
    candidate: CandidateMaterial | None = None
    final_issues: tuple[str, ...] = ()
    attempts: tuple[AttemptMaterial, ...] = ()
    attempt_count: int | None = None
    # ``prefix:path`` modules the generation workspace provided as context.
    workspace_context: tuple[str, ...] = ()
    # ``prefix:path`` modules the run's own runners wrote.
    run_outputs: tuple[str, ...] = ()
    replacement_target: bool = False
    notes: tuple[str, ...] = ()

    @property
    def citation_path(self) -> str | None:
        attestation = self.source_attestation or {}
        resolved = attestation.get("resolved_corpus_citation_path")
        if isinstance(resolved, str) and resolved:
            return resolved
        return self.citation

    @property
    def target_key(self) -> str | None:
        if self.candidate is None or not self.policy_prefix:
            return None
        return module_key(self.candidate.module_path, self.policy_prefix)


_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)*")


def split_module_path(
    relative: str, policy_prefix: str | None
) -> tuple[str, str] | None:
    """``(prefix, path)`` of a module path, with or without a jurisdiction dir.

    ``statutes/7/a.yaml`` takes ``policy_prefix``; ``us/statutes/7/a.yaml``
    names its own prefix, as the encoder's candidate locator accepts.
    """

    from ..constants import RULESPEC_ATOMIC_MODULE_ROOTS

    parts = relative.split("/")
    if parts and parts[0] in RULESPEC_ATOMIC_MODULE_ROOTS and policy_prefix:
        return policy_prefix, relative
    if (
        len(parts) >= 2
        and _JURISDICTION.fullmatch(parts[0])
        and parts[1] in RULESPEC_ATOMIC_MODULE_ROOTS
    ):
        return parts[0], "/".join(parts[1:])
    return None


def module_key(relative: str, policy_prefix: str | None) -> str | None:
    split = split_module_path(relative, policy_prefix)
    if split is None:
        return None
    prefix, path = split
    return f"{prefix}:{path.removesuffix('.yaml')}"
