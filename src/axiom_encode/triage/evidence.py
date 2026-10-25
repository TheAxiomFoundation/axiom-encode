"""Failure-triage evidence bundle (``axiom-encode/failure-triage-evidence/v1``).

See docs/design/encode-failure-triage.md (Stage 3, PR 1). The bundle is the
deterministic, bounded input a later triage step reads for one failed run:

- the target: citation, corpus row identity, source text, amendment context
  and the dispatch's review findings;
- the final candidate RuleSpec and its companion tests;
- every final-attempt issue with its file, gate and bracket tag (text only);
- the attempt history, with per-attempt issue signatures where recorded;
- the five probes of :mod:`axiom_encode.triage.probes`.

The builder never calls a model, the network or the encoder's validation and
never changes an encode. Inputs are untrusted files: they are only parsed
(JSON, safe YAML, text), read through contained bounded reads that refuse
symlinks, and never executed.

``evidence_sha256`` is the SHA-256 of the canonical JSON of the bundle without
``input`` (where the files were read from) and without the digest itself, so
the same recorded run gives the same digest wherever it is stored.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from ..constants import RULESPEC_ATOMIC_MODULE_ROOTS
from ..corpus_resolver import CorpusResolutionError, read_bounded_regular_file
from .issues import MAX_SIGNATURE_CHARS, issue_signature, parse_issue
from .material import AttemptMaterial, CandidateMaterial, TriageMaterial
from .probes import (
    PROBE_NAMES,
    probe_corpus,
    probe_gate,
    probe_import,
    probe_persistence,
    probe_registry,
)
from .views import LocalCorpusIndex, RulespecSnapshot, RulespecView

EVIDENCE_SCHEMA = "axiom-encode/failure-triage-evidence/v1"
WORKFLOW_BUNDLE_SCHEMA = "axiom-encode/failed-reencode-diagnostics/v1"
FAILED_CANDIDATE_SCHEMA = "axiom-encode/failed-encode-candidate/v1"
INPUT_LOCAL_ENCODE = "local_encode"
INPUT_LOCAL_RUN = "local_run"
INPUT_WORKFLOW_BUNDLE = "workflow_bundle"

MAX_INPUT_FILE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 512 * 1024
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)*")
_REF_RE = re.compile(
    r"^(?:(?P<plain>target|source|candidate|tests)"
    r"|(?P<lined>source|candidate|tests):L(?P<line>[1-9][0-9]*)"
    r"|issue:final:(?P<issue>[0-9]+)"
    r"|attempt:(?P<attempt>[0-9]+)(?::issue:(?P<attempt_issue>[0-9]+))?"
    r"|probe:(?P<probe>[a-z]+):(?P<finding>[0-9]+))$"
)


class EvidenceInputError(ValueError):
    """A run directory or bundle that the builder cannot read."""


@dataclass(frozen=True)
class EvidenceLimits:
    source_chars: int = 48_000
    candidate_chars: int = 48_000
    tests_chars: int = 24_000
    issues: int = 120
    issue_chars: int = 1_200
    attempts: int = 8
    attempt_issues: int = 60
    review_findings: int = 4
    review_finding_chars: int = 4_000
    findings_per_probe: int = 64
    # Free-form metadata (outcome, attestation, provision metadata).
    meta_chars: int = 8_000
    error_chars: int = 2_000
    list_items: int = 64


DEFAULT_LIMITS = EvidenceLimits()
# Tighter tiers, tried in order when a bundle exceeds MAX_BUNDLE_BYTES.
_SHRINK_TIERS = (
    EvidenceLimits(
        source_chars=24_000,
        candidate_chars=24_000,
        tests_chars=12_000,
        issues=60,
        issue_chars=800,
        attempt_issues=30,
        review_finding_chars=2_000,
        findings_per_probe=32,
    ),
    EvidenceLimits(
        source_chars=8_000,
        candidate_chars=8_000,
        tests_chars=4_000,
        issues=30,
        issue_chars=400,
        attempts=4,
        attempt_issues=10,
        review_findings=1,
        review_finding_chars=1_000,
        findings_per_probe=12,
        meta_chars=4_000,
        error_chars=600,
        list_items=16,
    ),
    EvidenceLimits(
        source_chars=2_000,
        candidate_chars=2_000,
        tests_chars=1_000,
        issues=10,
        issue_chars=300,
        attempts=2,
        attempt_issues=5,
        review_findings=1,
        review_finding_chars=500,
        findings_per_probe=4,
        meta_chars=1_000,
        error_chars=200,
        list_items=4,
    ),
)
# Room for ``size_bytes`` and ``evidence_sha256``, added after the size check,
# and for ``input`` (a name of at most MAX_INPUT_NAME_CHARS characters). The
# allowance is fixed, so where the input was read never changes the limits.
_DIGEST_OVERHEAD_BYTES = 128
MAX_INPUT_NAME_CHARS = 512
# A JSON-escaped control character takes six bytes.
_INPUT_ALLOWANCE_BYTES = 64 + 6 * MAX_INPUT_NAME_CHARS
_RESERVED_BYTES = _DIGEST_OVERHEAD_BYTES + _INPUT_ALLOWANCE_BYTES


# --------------------------------------------------------------------------
# Safe reading
# --------------------------------------------------------------------------


def _read_bytes(root: Path, relative: str | PurePosixPath) -> bytes | None:
    """Read one contained regular file; ``None`` when it does not exist."""

    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise EvidenceInputError(f"unsafe relative path: {relative}")
    path = root.joinpath(*parts)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        return read_bounded_regular_file(
            root, path, label="triage input file", max_bytes=MAX_INPUT_FILE_BYTES
        )
    except (OSError, CorpusResolutionError) as exc:
        raise EvidenceInputError(f"cannot read {relative}: {exc}") from exc


def _read_text(root: Path, relative: str | PurePosixPath) -> str | None:
    data = _read_bytes(root, relative)
    return None if data is None else data.decode("utf-8", errors="replace")


def _read_json(root: Path, relative: str | PurePosixPath) -> Any:
    text = _read_text(root, relative)
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceInputError(f"{relative} is not valid JSON") from exc


def reject_symlinks(base: Path) -> None:
    """Refuse an input tree that holds any symlink.

    Workflow bundles never contain links (the packager refuses them), and a
    link anywhere below the input root could point a later read outside it.
    """

    base = Path(os.path.abspath(base))
    if base.is_symlink():
        raise EvidenceInputError(f"input root is a symlink: {base}")
    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        for name in (*dirnames, *filenames):
            if (Path(directory) / name).is_symlink():
                relative = (Path(directory) / name).relative_to(base).as_posix()
                raise EvidenceInputError(f"input holds a symlink: {relative}")


def _safe_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _safe_children(path: Path) -> list[Path]:
    if not _safe_dir(path):
        return []
    return sorted(
        child
        for child in path.iterdir()
        if _SAFE_SEGMENT.fullmatch(child.name) and not child.is_symlink()
    )


def _yaml_modules(root: Path) -> list[str]:
    """Relative RuleSpec module paths (not tests) under one directory."""

    if not _safe_dir(root):
        return []
    found = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames if not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            path = Path(directory) / name
            if (
                name.endswith(".yaml")
                and not name.endswith(".test.yaml")
                and not path.is_symlink()
            ):
                found.append(path.relative_to(root).as_posix())
    return sorted(found)


def _dashed(citation: str) -> str:
    """The encoder's workspace slug for a citation (``evals._slugify``)."""

    from ..harness.evals import _slugify

    return _slugify(citation)


def _module_key(relative: str, policy_prefix: str | None) -> str | None:
    parts = PurePosixPath(relative.removesuffix(".yaml")).parts
    if not parts:
        return None
    if parts[0] in RULESPEC_ATOMIC_MODULE_ROOTS and policy_prefix:
        return f"{policy_prefix}:{'/'.join(parts)}"
    if (
        len(parts) >= 2
        and _JURISDICTION.fullmatch(parts[0])
        and parts[1] in RULESPEC_ATOMIC_MODULE_ROOTS
    ):
        return f"{parts[0]}:{'/'.join(parts[1:])}"
    return None


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Workspace:
    source_text: str | None
    source_attestation: Mapping[str, Any] | None
    provision_metadata: Mapping[str, Any] | None
    policy_prefix: str | None
    context_keys: tuple[str, ...]
    amendment_documents: tuple[str, ...]
    dropped_amendment_documents: tuple[str, ...]
    review_findings: tuple[str, ...]


def _read_workspace(workspace: Path) -> _Workspace:
    source_text = _read_text(workspace, "source.txt")
    metadata = _read_json(workspace, "source-metadata.json")
    attestation = (
        metadata.get("source_attestation") if isinstance(metadata, Mapping) else None
    )
    provision_raw = _read_text(workspace, "provision-metadata.txt")
    provision: Mapping[str, Any] | None = None
    if provision_raw:
        try:
            parsed = json.loads(provision_raw)
        except (json.JSONDecodeError, RecursionError):
            parsed = None
        provision = parsed if isinstance(parsed, Mapping) else None
    manifest = _read_json(workspace, "context-manifest.json")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    prefix = manifest.get("policy_prefix")
    prefix = (
        prefix if isinstance(prefix, str) and _JURISDICTION.fullmatch(prefix) else None
    )
    context_keys: list[str] = []
    amendments: list[str] = []
    for item in manifest.get("context_files") or []:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("kind") or "")
        if "amendment" in kind and isinstance(item.get("citation_path"), str):
            amendments.append(item["citation_path"])
        import_path = item.get("import_path")
        workspace_path = str(item.get("workspace_path") or "")
        if (
            isinstance(import_path, str)
            and workspace_path.startswith("context/")
            and workspace_path.endswith(".yaml")
            and not workspace_path.endswith(".test.yaml")
        ):
            from .probes import _target_key

            key = _target_key(import_path, prefix)
            if key:
                context_keys.append(key)
    dropped = [
        str(item.get("citation_path"))
        for item in manifest.get("dropped_amendment_documents") or []
        if isinstance(item, Mapping) and isinstance(item.get("citation_path"), str)
    ]
    findings = [
        str(item.get("content"))
        for item in manifest.get("review_findings_files") or []
        if isinstance(item, Mapping) and isinstance(item.get("content"), str)
    ]
    return _Workspace(
        source_text=source_text,
        source_attestation=attestation if isinstance(attestation, Mapping) else None,
        provision_metadata=provision,
        policy_prefix=prefix,
        context_keys=tuple(sorted(set(context_keys))),
        amendment_documents=tuple(sorted(set(amendments))),
        dropped_amendment_documents=tuple(sorted(set(dropped))),
        review_findings=tuple(findings),
    )


def _runner_dirs(lane_root: Path) -> list[Path]:
    reserved = {"_eval_workspaces", "traces", "final-rejected-candidate"}
    return [child for child in _safe_children(lane_root) if child.name not in reserved]


def _find_workspace(lane_root: Path, citation: str, runner: str | None) -> Path | None:
    workspaces = lane_root / "_eval_workspaces"
    if not _safe_dir(workspaces):
        return None
    dashed = _dashed(citation)
    if not _SAFE_SEGMENT.fullmatch(dashed):
        return None
    found = [
        child / dashed / "workspace"
        for child in _safe_children(workspaces)
        if _safe_dir(child / dashed / "workspace")
    ]
    if not found:
        # A routed amendment request is encoded under its principal
        # provision's slug; the provision metadata names the request.
        found = [
            workspace
            for workspace in _all_workspaces(workspaces)
            if _workspace_amendment_request(workspace) == citation
        ]
    for workspace in found:
        if runner is not None and workspace.parent.parent.name == runner:
            return workspace
    return found[-1] if found else None


def _all_workspaces(workspaces: Path) -> list[Path]:
    return [
        slug / "workspace"
        for runner_dir in _safe_children(workspaces)
        for slug in _safe_children(runner_dir)
        if _safe_dir(slug / "workspace")
    ]


def _workspace_amendment_request(workspace: Path) -> str | None:
    try:
        text = _read_text(workspace, "provision-metadata.txt")
    except EvidenceInputError:
        return None
    try:
        metadata = json.loads(text or "")
    except (json.JSONDecodeError, RecursionError):
        return None
    value = metadata.get("amendment_request") if isinstance(metadata, Mapping) else None
    return value if isinstance(value, str) else None


def _final_candidate(
    lane_root: Path, candidate_dir: Path | None
) -> tuple[Mapping[str, Any] | None, CandidateMaterial | None, list[str]]:
    notes: list[str] = []
    if candidate_dir is None or not _safe_dir(candidate_dir):
        return None, None, ["no final rejected candidate was recorded"]
    recorded = _read_json(candidate_dir, "issues.json")
    if not isinstance(recorded, Mapping):
        return None, None, ["final rejected candidate has no issues.json"]
    if recorded.get("schema") != FAILED_CANDIDATE_SCHEMA:
        notes.append(f"unexpected candidate schema {recorded.get('schema')!r}")
    modules = _yaml_modules(candidate_dir)
    module_path = recorded.get("path")
    if not isinstance(module_path, str) or module_path not in modules:
        module_path = modules[0] if len(modules) == 1 else None
    if module_path is None:
        return recorded, None, [*notes, "final candidate module is not identifiable"]
    rulespec = _read_text(candidate_dir, module_path) or ""
    tests = _read_text(candidate_dir, module_path.removesuffix(".yaml") + ".test.yaml")
    runner = None
    candidate_bytes = rulespec.encode()
    for runner_dir in _runner_dirs(lane_root):
        try:
            copy = _read_bytes(runner_dir, module_path)
        except EvidenceInputError:
            continue
        if copy == candidate_bytes:
            runner = runner_dir.name
    return (
        recorded,
        CandidateMaterial(
            module_path=module_path,
            rulespec=rulespec,
            tests=tests,
            runner=runner,
            sibling_modules=tuple(m for m in modules if m != module_path),
        ),
        notes,
    )


def _run_outputs(
    lane_root: Path, policy_prefix: str | None, citation: str | None
) -> tuple[str, ...]:
    """Modules a runner wrote, without the context modules it was given.

    A repo-augmented runner's output tree also holds copies of its workspace
    context; a file whose bytes equal the runner's own ``context/`` copy is
    context, not authored.
    """

    keys = set()
    for runner_dir in _runner_dirs(lane_root):
        workspace = (
            lane_root
            / "_eval_workspaces"
            / runner_dir.name
            / _dashed(citation)
            / "workspace"
            if citation
            else None
        )
        for relative in _yaml_modules(runner_dir):
            key = _module_key(relative, policy_prefix)
            if not key:
                continue
            if workspace is not None and _safe_dir(workspace / "context"):
                try:
                    context_copy = _read_bytes(workspace / "context", relative)
                    authored = _read_bytes(runner_dir, relative)
                except EvidenceInputError:
                    context_copy = authored = None
                if context_copy is not None and context_copy == authored:
                    continue
            keys.add(key)
    return tuple(sorted(keys))


def _issues_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


def material_from_output_root(
    lane_root: Path,
    *,
    citation: str | None,
    input_kind: str,
    input_name: str,
    candidate_dir: Path | None = None,
    attempts: Sequence[AttemptMaterial] = (),
    **fields: Any,
) -> TriageMaterial:
    """Build material from an encode output root (``--output``) layout."""

    lane_root = Path(os.path.abspath(lane_root))
    recorded, candidate, notes = _final_candidate(lane_root, candidate_dir)
    if citation is None and recorded is not None:
        value = recorded.get("citation")
        citation = value if isinstance(value, str) else None
    workspace = (
        _find_workspace(lane_root, citation, candidate.runner if candidate else None)
        if citation
        else None
    )
    ws = _read_workspace(workspace) if workspace is not None else None
    if ws is None:
        notes.append("no generation workspace was found for the citation")
    amendment_request = None
    principal = (
        (ws.source_attestation or {}).get("requested_corpus_citation_path")
        if ws
        else None
    )
    routed = (ws.provision_metadata or {}).get("amendment_request") if ws else None
    if (
        citation
        and routed == citation
        and isinstance(principal, str)
        and principal != citation
    ):
        # A routed amendment request is encoded as its principal provision:
        # the gate, outputs and paths follow the principal.
        amendment_request, citation = citation, principal
        notes.append(f"amendment request {amendment_request} encoded as {principal}")
    final_issues = _issues_list((recorded or {}).get("issues"))
    attempt_count = (recorded or {}).get("attempt_count")
    attempt_count = attempt_count if isinstance(attempt_count, int) else None
    attempts = list(attempts)
    # The final rejected candidate is the selected (best) attempt, whose
    # number is not recorded: its issues stay ``final_issues`` and are never
    # assigned to an attempt number.
    counts = [attempt_count, len(attempts) or None]
    attempt_count = max((c for c in counts if c), default=None)
    policy_prefix = ws.policy_prefix if ws else None
    encoder_version = fields.pop("encoder_version", None) or (recorded or {}).get(
        "encoder_version"
    )
    return TriageMaterial(
        input_kind=input_kind,
        input_name=input_name,
        citation=citation,
        policy_prefix=policy_prefix,
        encoder_version=encoder_version if isinstance(encoder_version, str) else None,
        source_text=ws.source_text if ws else None,
        source_attestation=ws.source_attestation if ws else None,
        provision_metadata=ws.provision_metadata if ws else None,
        amendment_request=amendment_request,
        amendment_documents=ws.amendment_documents if ws else (),
        dropped_amendment_documents=ws.dropped_amendment_documents if ws else (),
        review_findings=ws.review_findings if ws else (),
        candidate=candidate,
        final_issues=final_issues,
        attempts=tuple(attempts),
        attempt_count=attempt_count,
        workspace_context=ws.context_keys if ws else (),
        run_outputs=_run_outputs(lane_root, policy_prefix, citation),
        notes=tuple(notes),
        **fields,
    )


def load_local_encode(
    output_root: Path,
    *,
    citation: str,
    outcome: Mapping[str, Any] | None = None,
    attempts: Sequence[Mapping[str, Any]] = (),
    final_rejected_candidate: Path | None = None,
    run_id: str | None = None,
    encoder_version: str | None = None,
    rulespec_ref: str | None = None,
    rulespec_ref_source: str | None = None,
    replacement_target: bool = False,
) -> TriageMaterial:
    """Material for one local encode from its outcome and output workspace.

    ``attempts`` are the encode loop's failed attempts in order, each a
    mapping with ``error``, ``validation_issues`` (or
    ``full_validation_issues``), ``validation_issue_count`` and ``model``.
    """

    for root in (output_root, final_rejected_candidate):
        if root is not None and Path(root).exists():
            reject_symlinks(Path(root))
    attempt_records = []
    for number, attempt in enumerate(attempts, start=1):
        issues = _issues_list(
            attempt.get("full_validation_issues") or attempt.get("validation_issues")
        )
        count = attempt.get("validation_issue_count")
        attempt_records.append(
            AttemptMaterial(
                attempt=number,
                model=str(attempt["model"]) if attempt.get("model") else None,
                error=str(attempt["error"]) if attempt.get("error") else None,
                issues=issues if ("validation_issues" in attempt or issues) else None,
                issue_count=count if isinstance(count, int) else len(issues),
            )
        )
    return material_from_output_root(
        Path(output_root),
        citation=citation,
        input_kind=INPUT_LOCAL_ENCODE,
        input_name=Path(output_root).name,
        candidate_dir=final_rejected_candidate,
        attempts=attempt_records,
        run_id=run_id,
        encoder_version=encoder_version,
        rulespec_ref=rulespec_ref,
        rulespec_ref_source=rulespec_ref_source,
        outcome=_outcome_summary(outcome or {}),
        replacement_target=replacement_target,
    )


def _outcome_summary(outcome: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "final_success",
        "primary_error",
        "apply_error",
        "validation_failure_counts",
        "failure_class",
        "pre_encode_gate",
        "retry_stopped",
    )
    return {key: outcome[key] for key in keys if key in outcome}


def _read_local_db(
    run_dir: Path, citation: str
) -> tuple[dict[str, Any], list[Any], str | None, str | None]:
    db = run_dir / "encodings.db"
    if not db.is_file() or db.is_symlink():
        return {}, [], None, None
    try:
        # ``as_uri`` percent-encodes ``#`` and ``?`` in the path.
        connection = sqlite3.connect(f"{db.as_uri()}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error:
        return {}, [], None, None
    try:
        row = connection.execute(
            "select id, outcome_json, iterations_json, axiom_encode_version "
            "from encoding_runs where citation = ? order by timestamp desc, id desc limit 1",
            (citation,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    finally:
        connection.close()
    if row is None:
        return {}, [], None, None
    try:
        outcome = json.loads(row[1] or "{}")
        iterations = json.loads(row[2] or "[]")
    except (json.JSONDecodeError, RecursionError):
        outcome, iterations = {}, []
    return (
        outcome if isinstance(outcome, dict) else {},
        iterations if isinstance(iterations, list) else [],
        str(row[0]) if row[0] else None,
        str(row[3]) if row[3] else None,
    )


def is_local_run_dir(path: Path) -> bool:
    return (path / "run.json").is_file() and _safe_dir(path / "generated")


def load_local_run(
    path: Path,
    *,
    rulespec_ref: str | None = None,
) -> list[TriageMaterial]:
    """Material for one local pilot or A/B run directory.

    Returns an empty list when the run did not fail.
    """

    run_dir = Path(os.path.abspath(path))
    if run_dir.is_symlink():
        raise EvidenceInputError(f"input root is a symlink: {run_dir}")
    # Only these entries are read; the run's own tmp/ may hold links.
    for name in ("run.json", "failed_attempts.jsonl", "encodings.db"):
        if (run_dir / name).is_symlink():
            raise EvidenceInputError(f"input holds a symlink: {name}")
    run = _read_json(run_dir, "run.json")
    if not isinstance(run, Mapping) or not isinstance(run.get("citation"), str):
        raise EvidenceInputError("run.json has no citation")
    citation = run["citation"]
    outcome, iterations, run_id, version = _read_local_db(run_dir, citation)
    timed_out = run.get("timeout") is True
    if outcome.get("final_success") is True and not timed_out:
        return []
    if not outcome and run.get("rc") == 0 and not timed_out:
        return []
    attempts: list[dict[str, Any]] = []
    raw = _read_text(run_dir, "failed_attempts.jsonl") or ""
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            record = None
        # A malformed record still was an attempt: keep its place, with no
        # recorded issues, so the history reads as incomplete.
        attempts.append(record if isinstance(record, dict) else {})
    for index, attempt in enumerate(attempts):
        if index < len(iterations) and isinstance(iterations[index], Mapping):
            attempt.setdefault("model", iterations[index].get("model"))
    summary = dict(outcome)
    if timed_out:
        summary.setdefault("timed_out", True)
    material = load_local_encode(
        run_dir / "generated",
        citation=citation,
        outcome=summary,
        attempts=attempts,
        final_rejected_candidate=run_dir / "final-rejected-candidate",
        run_id=run_id,
        encoder_version=version,
        rulespec_ref=rulespec_ref,
        rulespec_ref_source="argument" if rulespec_ref else None,
    )
    outcome_fields = dict(material.outcome)
    if timed_out:
        outcome_fields["timed_out"] = True
    return [
        replace(
            material,
            input_kind=INPUT_LOCAL_RUN,
            input_name=run_dir.name,
            encoder_commit=run.get("encoder_commit")
            if isinstance(run.get("encoder_commit"), str)
            else None,
            outcome=outcome_fields,
            # Every source of the attempt count is kept: a count larger than
            # the recorded attempts marks the history incomplete.
            attempt_count=max(
                (c for c in (material.attempt_count, len(iterations)) if c),
                default=None,
            ),
        )
    ]


_REPLACEMENT_LANES = frozenset({"target", "target-preflight"})


def workflow_bundle_root(path: Path) -> Path | None:
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        return None
    for root in (path / "x", path):
        if (root / "metadata.json").is_file() and not root.is_symlink():
            return root
    return None


def load_workflow_bundle(path: Path) -> list[TriageMaterial]:
    """Material for each lane with a rejected candidate in one failure bundle.

    A bundle with no rejected candidate gives one material for its target
    lane (or its first lane) with the workspace, when one exists.
    """

    root = workflow_bundle_root(path)
    if root is None:
        raise EvidenceInputError("no metadata.json in the bundle")
    reject_symlinks(root)
    metadata = _read_json(root, "metadata.json")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != WORKFLOW_BUNDLE_SCHEMA
    ):
        raise EvidenceInputError("metadata.json is not a failed re-encode bundle")
    citation = (
        metadata.get("citation") if isinstance(metadata.get("citation"), str) else None
    )
    name = Path(os.path.abspath(path)).name
    lanes = _safe_children(root / "generated")
    with_candidate = [
        lane
        for lane in lanes
        if (lane / "final-rejected-candidate" / "issues.json").is_file()
    ]
    chosen = (
        with_candidate
        or sorted(lanes, key=lambda lane: (lane.name != "target", lane.name))[:1]
    )
    outcome = {
        "failed_steps": sorted(
            str(step)
            for step in metadata.get("failed_steps") or []
            if isinstance(step, str)
        ),
    }
    replacement = bool(
        metadata.get("replace_rulespec_path")
        or metadata.get("replace_legacy_rulespec_path")
    )
    common = {
        "run_id": str(metadata.get("workflow_run_id") or "") or None,
        "rulespec_ref": metadata.get("rulespec_ref")
        if isinstance(metadata.get("rulespec_ref"), str)
        else None,
        "rulespec_ref_source": "bundle_metadata",
        "corpus_ref": metadata.get("corpus_ref")
        if isinstance(metadata.get("corpus_ref"), str)
        else None,
        "encoder_commit": metadata.get("encoder_commit")
        if isinstance(metadata.get("encoder_commit"), str)
        else None,
        "replacement_target": replacement,
    }
    if not chosen:
        return [
            TriageMaterial(
                input_kind=INPUT_WORKFLOW_BUNDLE,
                input_name=name,
                citation=citation,
                outcome=outcome,
                notes=("the bundle has no generated lanes",),
                **common,
            )
        ]
    materials = []
    for lane in chosen:
        lane_citation = citation
        if lane.name not in _REPLACEMENT_LANES:
            dependent = metadata.get("dependent_citation")
            recorded = (
                _read_json(lane / "final-rejected-candidate", "issues.json")
                if (lane / "final-rejected-candidate" / "issues.json").is_file()
                else None
            )
            if isinstance(recorded, Mapping) and isinstance(
                recorded.get("citation"), str
            ):
                lane_citation = recorded["citation"]
            elif isinstance(dependent, str):
                lane_citation = dependent
        material = material_from_output_root(
            lane,
            citation=lane_citation,
            input_kind=INPUT_WORKFLOW_BUNDLE,
            input_name=name,
            candidate_dir=lane / "final-rejected-candidate",
            outcome=outcome,
            **common,
        )
        # The replace flags belong to the target invocation only; dependent
        # and source lanes run without them.
        materials.append(
            replace(
                material,
                lane=lane.name,
                replacement_target=replacement and lane.name in _REPLACEMENT_LANES,
            )
        )
    return materials


def load_any(path: Path, *, rulespec_ref: str | None = None) -> list[TriageMaterial]:
    """Load a workflow bundle or a local run directory."""

    if Path(os.path.abspath(path)).is_symlink():
        raise EvidenceInputError(f"input root is a symlink: {path}")
    if workflow_bundle_root(path) is not None:
        return load_workflow_bundle(path)
    if is_local_run_dir(Path(path)):
        return load_local_run(path, rulespec_ref=rulespec_ref)
    raise EvidenceInputError(
        "not a workflow failure bundle (metadata.json) or a local run (run.json)"
    )


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_sha256(bundle: Mapping[str, Any]) -> str:
    hashed = {k: v for k, v in bundle.items() if k not in {"evidence_sha256", "input"}}
    return _sha256_text(canonical_json(hashed))


def _bounded(
    field_name: str,
    text: str | None,
    limit: int,
    truncation: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if text is None:
        return None
    kept = text[:limit]
    if len(kept) < len(text):
        truncation.append(
            {
                "field": field_name,
                "original_chars": len(text),
                "kept_chars": len(kept),
                "original_sha256": _sha256_text(text),
            }
        )
    return {"text": kept, "sha256": _sha256_text(text), "chars": len(text)}


def _bounded_value(
    field_name: str,
    value: Any,
    limit: int,
    truncation: list[dict[str, Any]],
) -> Any:
    """A JSON value kept whole when small, else replaced by a bounded preview."""

    if value is None:
        return None
    text = canonical_json(value)
    if len(text) <= limit:
        return value
    truncation.append(
        {
            "field": field_name,
            "original_chars": len(text),
            "kept_chars": limit,
            "original_sha256": _sha256_text(text),
        }
    )
    return {"truncated": True, "sha256": _sha256_text(text), "preview": text[:limit]}


def _bounded_list(
    field_name: str,
    items: Sequence[Any],
    limit: int,
    truncation: list[dict[str, Any]],
) -> list[Any]:
    if len(items) > limit:
        truncation.append(
            {"field": field_name, "original_items": len(items), "kept_items": limit}
        )
    return list(items[:limit])


def _country(prefix: str | None) -> str | None:
    return prefix.split("-", 1)[0] if prefix else None


RulespecViewFactory = Callable[[TriageMaterial], RulespecView | None]


def _assemble(
    material: TriageMaterial,
    probes: Mapping[str, Mapping[str, Any]],
    *,
    views: Mapping[str, Any],
    limits: EvidenceLimits,
) -> dict[str, Any]:
    truncation: list[dict[str, Any]] = []
    attestation = dict(material.source_attestation or {})
    attestation.pop("rulespec_root", None)  # an absolute runner path
    candidate = None
    if material.candidate is not None:
        candidate = {
            "module_path": material.candidate.module_path,
            "runner": material.candidate.runner,
            "rulespec": _bounded(
                "candidate.rulespec",
                material.candidate.rulespec,
                limits.candidate_chars,
                truncation,
            ),
            "tests": _bounded(
                "candidate.tests",
                material.candidate.tests,
                limits.tests_chars,
                truncation,
            ),
            "sibling_modules": _bounded_list(
                "candidate.sibling_modules",
                material.candidate.sibling_modules,
                limits.list_items,
                truncation,
            ),
        }
    issues = []
    for index, text in enumerate(
        _bounded_list("final_issues", material.final_issues, limits.issues, truncation)
    ):
        parsed = parse_issue(text)
        kept = text[: limits.issue_chars]
        if len(kept) < len(text):
            truncation.append(
                {
                    "field": f"final_issues[{index}]",
                    "original_chars": len(text),
                    "kept_chars": len(kept),
                    "original_sha256": _sha256_text(text),
                }
            )
        issues.append(
            {
                "index": index,
                "text": kept,
                "file": parsed.file,
                "gate": parsed.gate,
                "tag": parsed.tag,
            }
        )
    attempts = []
    for attempt in _bounded_list(
        "attempts", material.attempts, limits.attempts, truncation
    ):
        signatures = None
        if attempt.issues is not None:
            full = [
                issue_signature(text, None)
                for text in _bounded_list(
                    f"attempts[{attempt.attempt}].issues",
                    attempt.issues,
                    limits.attempt_issues,
                    truncation,
                )
            ]
            signatures = [item[:MAX_SIGNATURE_CHARS] for item in full]
            clipped = [len(item) for item in full if len(item) > MAX_SIGNATURE_CHARS]
            if clipped:
                truncation.append(
                    {
                        "field": f"attempts[{attempt.attempt}].issue_signatures",
                        "clipped_items": len(clipped),
                        "kept_chars": MAX_SIGNATURE_CHARS,
                        "max_original_chars": max(clipped),
                        "original_sha256": _sha256_text(canonical_json(full)),
                    }
                )
        attempts.append(
            {
                "attempt": attempt.attempt,
                "model": attempt.model,
                "error": _bounded_value(
                    f"attempts[{attempt.attempt}].error",
                    attempt.error,
                    limits.error_chars,
                    truncation,
                ),
                "issue_count": attempt.issue_count,
                "issues_recorded": attempt.issues is not None,
                "issue_signatures": signatures,
            }
        )
    review_findings = [
        _bounded(
            f"target.review_findings[{index}]",
            text,
            limits.review_finding_chars,
            truncation,
        )
        for index, text in enumerate(
            _bounded_list(
                "target.review_findings",
                material.review_findings,
                limits.review_findings,
                truncation,
            )
        )
    ]
    bundle: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "input": {
            "kind": material.input_kind,
            "name": bounded_input_name(material.input_name),
        },
        "run": {
            "run_id": material.run_id,
            "lane": material.lane,
            "encoder_version": material.encoder_version,
            "encoder_commit": material.encoder_commit,
            "rulespec_ref": material.rulespec_ref,
            "rulespec_ref_source": material.rulespec_ref_source,
            "corpus_ref": material.corpus_ref,
            "attempt_count": material.attempt_count,
            "outcome": _bounded_value(
                "run.outcome", dict(material.outcome), limits.meta_chars, truncation
            ),
            "replacement_target": material.replacement_target,
            "notes": _bounded_list(
                "run.notes", material.notes, limits.list_items, truncation
            ),
        },
        "target": {
            "citation": material.citation,
            "citation_path": material.citation_path,
            "policy_prefix": material.policy_prefix,
            "source_attestation": _bounded_value(
                "target.source_attestation",
                attestation or None,
                limits.meta_chars,
                truncation,
            ),
            "provision_metadata": _bounded_value(
                "target.provision_metadata",
                dict(material.provision_metadata or {}) or None,
                limits.meta_chars,
                truncation,
            ),
            "source": _bounded(
                "target.source", material.source_text, limits.source_chars, truncation
            ),
            "amendment_context": {
                "request": material.amendment_request,
                "documents": _bounded_list(
                    "target.amendment_context.documents",
                    material.amendment_documents,
                    limits.list_items,
                    truncation,
                ),
                "dropped_documents": _bounded_list(
                    "target.amendment_context.dropped_documents",
                    material.dropped_amendment_documents,
                    limits.list_items,
                    truncation,
                ),
            },
            "review_findings": review_findings,
        },
        "candidate": candidate,
        "final_issues": issues,
        "attempts": attempts,
        "views": dict(views),
        "probes": {},
        "truncation": truncation,
    }
    for name in PROBE_NAMES:
        result = dict(probes[name])
        findings = list(result.get("findings") or [])
        kept = findings[: limits.findings_per_probe]
        result["truncated_findings"] = int(result.get("truncated_findings") or 0) + (
            len(findings) - len(kept)
        )
        result["findings"] = [
            {"id": f"probe:{name}:{index}", **finding}
            for index, finding in enumerate(kept)
        ]
        bundle["probes"][name] = result
    dropped = 0
    for result in bundle["probes"].values():
        for finding in result["findings"]:
            refs = [ref for ref in finding.get("refs", []) if resolve_ref(bundle, ref)]
            dropped += len(finding.get("refs", [])) - len(refs)
            finding["refs"] = refs
    if dropped:
        truncation.append(
            {"field": "probes.refs", "dropped_unresolvable_refs": dropped}
        )
    return bundle


def run_probes(
    material: TriageMaterial,
    *,
    rulespec_view: RulespecView | None = None,
    corpus: LocalCorpusIndex | None = None,
    registry: Any = None,
) -> dict[str, dict[str, Any]]:
    country = _country(material.policy_prefix)

    def snapshot_factory() -> RulespecSnapshot | None:
        if rulespec_view is None or country is None:
            return None
        return RulespecSnapshot(rulespec_view, country)

    return {
        "gate": probe_gate(material, snapshot_factory=snapshot_factory, corpus=corpus),
        "registry": probe_registry(
            material, snapshot_factory=snapshot_factory, registry=registry
        ),
        "corpus": probe_corpus(
            material, corpus=corpus, snapshot_factory=snapshot_factory
        ),
        "import": probe_import(material),
        "persistence": probe_persistence(material),
    }


def build_evidence(
    material: TriageMaterial,
    *,
    rulespec_view: RulespecView | None = None,
    corpus: LocalCorpusIndex | None = None,
    registry: Any = None,
    limits: EvidenceLimits = DEFAULT_LIMITS,
    max_bytes: int = MAX_BUNDLE_BYTES,
    rulespec_unavailable_reason: str | None = None,
) -> dict[str, Any]:
    """Build the bounded evidence bundle for one failed run."""

    probes = run_probes(
        material, rulespec_view=rulespec_view, corpus=corpus, registry=registry
    )
    views = {
        "rulespec": rulespec_view.identity()
        if rulespec_view is not None
        else {"kind": "none", "reason": rulespec_unavailable_reason},
        "corpus": corpus.identity() if corpus is not None else {"kind": "none"},
        "concept_registry": (
            {"kind": "packaged", "concepts": len(registry.concepts_by_id)}
            if registry is not None
            else {"kind": "none"}
        ),
    }
    bundle = _assemble(material, probes, views=views, limits=limits)
    for tier in _SHRINK_TIERS:
        if _hashed_size(bundle) + _RESERVED_BYTES <= max_bytes:
            break
        bundle = _assemble(material, probes, views=views, limits=tier)
        bundle["truncation"].append(
            {"field": "bundle", "shrunk_to_fit_bytes": max_bytes}
        )
    size = _hashed_size(bundle) + _RESERVED_BYTES
    if size > max_bytes:
        raise EvidenceInputError(
            f"evidence bundle needs {size} bytes even at the tightest limits "
            f"(bound {max_bytes})"
        )
    bundle["size_bytes"] = _hashed_size(bundle)
    bundle["evidence_sha256"] = evidence_sha256(bundle)
    return bundle


def bounded_input_name(name: str) -> str:
    return str(name)[:MAX_INPUT_NAME_CHARS]


def _hashed_size(bundle: Mapping[str, Any]) -> int:
    """Size of the location-independent part (what the digest covers)."""

    hashed = {k: v for k, v in bundle.items() if k not in {"evidence_sha256", "input"}}
    return len(canonical_json(hashed).encode())


# --------------------------------------------------------------------------
# Refs
# --------------------------------------------------------------------------


def _line_count(section: Mapping[str, Any] | None) -> int:
    if not isinstance(section, Mapping) or not isinstance(section.get("text"), str):
        return 0
    text = section["text"]
    return text.count("\n") + (0 if text.endswith("\n") else 1) if text else 0


def resolve_ref(bundle: Mapping[str, Any], ref: str) -> bool:
    """Return whether ``ref`` points at something present in the bundle."""

    match = _REF_RE.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        return False
    target = bundle.get("target") or {}
    candidate = bundle.get("candidate") or {}
    sections = {
        "source": target.get("source"),
        "candidate": candidate.get("rulespec"),
        "tests": candidate.get("tests"),
    }
    if match.group("plain"):
        name = match.group("plain")
        return bool(target) if name == "target" else sections[name] is not None
    if match.group("lined"):
        section = sections[match.group("lined")]
        line = int(match.group("line"))
        count = _line_count(section)
        if line > count:
            return False
        # The last kept line of a truncated text may be cut mid-line.
        text = section["text"]
        cut = int(section.get("chars") or 0) > len(text)
        return not (cut and line == count and not text.endswith("\n"))
    if match.group("issue") is not None:
        return int(match.group("issue")) < len(bundle.get("final_issues") or [])
    if match.group("attempt") is not None:
        number = int(match.group("attempt"))
        for attempt in bundle.get("attempts") or []:
            if attempt.get("attempt") != number:
                continue
            if match.group("attempt_issue") is None:
                return True
            signatures = attempt.get("issue_signatures") or []
            return int(match.group("attempt_issue")) < len(signatures)
        return False
    probe = (bundle.get("probes") or {}).get(match.group("probe"))
    return isinstance(probe, Mapping) and int(match.group("finding")) < len(
        probe.get("findings") or []
    )


# --------------------------------------------------------------------------
# Batch discovery
# --------------------------------------------------------------------------

_PRUNE = {
    "generated",
    "final-rejected-candidate",
    "_eval_workspaces",
    "tmp",
    "corpus",
    "engine",
    "tools",
    "node_modules",
    ".git",
    "x",
}


def iter_run_inputs(directory: Path, *, max_depth: int = 6) -> Iterator[Path]:
    """Yield every workflow bundle or local run directory under ``directory``."""

    base = Path(os.path.abspath(directory))
    for current, dirnames, _files in os.walk(base, followlinks=False):
        here = Path(current)
        depth = len(here.relative_to(base).parts)
        if here != base and (workflow_bundle_root(here) or is_local_run_dir(here)):
            dirnames[:] = []
            yield here
            continue
        dirnames[:] = sorted(
            name
            for name in dirnames
            if name not in _PRUNE
            and not name.startswith("rulespec-")
            and not (here / name).is_symlink()
            and depth < max_depth
        )
