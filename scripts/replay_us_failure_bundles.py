#!/usr/bin/env python3
"""Replay the complete-source-unit analyzer over US failure fixtures, offline.

The committed benchmark (``tests/fixtures/us_replay_benchmark``) holds real
rejected US candidates from ``targeted-reencode-failure`` artifacts: the
candidate RuleSpec, its companion tests, the exact workspace ``source.txt``,
the workspace ``source-metadata.json``, the recorded pipeline ``issues.json``
and the context modules the candidate imports or names as deferral providers.
Replaying them needs no model, network, rules engine or corpus checkout, so
every validator or prompt PR can report a before/after number.

Commands::

    # Replay the committed fixtures with their context (the default).
    python scripts/replay_us_failure_bundles.py replay --out after.json

    # Replay without context: no imports, deferral providers or same-act aliases.
    python scripts/replay_us_failure_bundles.py replay --no-context --out raw.json

    # Replay extracted failure bundles (``<dir>/<run_id>/x`` or ``<dir>/<run_id>``).
    python -I scripts/replay_us_failure_bundles.py replay --bundles bundles/ \
        [--latest-per-citation] --out all.json

    # Compare two replay reports (for example main versus a PR branch).
    python scripts/replay_us_failure_bundles.py compare before.json after.json

    # Check or rewrite the committed snapshot (the pytest ratchet).
    python scripts/replay_us_failure_bundles.py snapshot --check
    python scripts/replay_us_failure_bundles.py snapshot [--allow-increase]

    # Add one extracted failure bundle as a committed fixture.
    python -I scripts/replay_us_failure_bundles.py extract bundles/<run_id>/x \
        [--rulespec-root <rulespec checkout at the bundle's rulespec_ref>]

Bundles are untrusted data. This script only parses them (JSON, safe YAML and
text); it never executes anything they contain, refuses symlinks, and keeps
every path inside the bundle it came from.

Only ``analyze_complete_source_unit`` runs. Compilation, companion-test
execution, oracles, proof-excerpt checks, reviewers and signing are out of
scope; see ``docs/us-replay-benchmark.md``.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/us_replay_benchmark"
SNAPSHOT_FILE = "snapshot.json"
FIXTURE_FILE = "fixture.json"
FIXTURE_SCHEMA = "axiom-encode/us-replay-fixture/v1"
REPORT_SCHEMA = "axiom-encode/us-replay-report/v1"
SNAPSHOT_SCHEMA = "axiom-encode/us-replay-snapshot/v1"
RUN_URL = "https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/{run_id}"
MAX_FILE_BYTES = 8 * 1024 * 1024
RULE_PATTERN = re.compile(r"\[([a-z][a-z0-9-]*:[a-z0-9-]+)\]")
DEFERRAL_TARGET_PATTERN = re.compile(
    r"[a-z]{2}(?:-[a-z0-9-]+)?:[A-Za-z0-9_./-]+#[A-Za-z_][A-Za-z0-9_]*"
)
PREFIX_PATTERN = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)*")
RUN_ID_PATTERN = re.compile(r"[0-9]{1,20}")
SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CONTEXT_WORKSPACE = "workspace"
CONTEXT_NONE = "none"
ORIGIN_RULESPEC_REF = "rulespec_ref"
UNLABELED = "unlabeled"
# Failure classes v1 (design: encode failure classes and the pre-encode gate).
# A fixture may carry a hand label: ``expected_class``, an optional
# ``expected_subclass``, ``class_evidence`` and ``class_source``.
FAILURE_CLASSES: Mapping[str, frozenset[str]] = {
    "encoder": frozenset({"omission", "formula", "test", "harness"}),
    "validator_fp": frozenset(),
    "corpus": frozenset({"content", "metadata"}),
    "dependency": frozenset(),
    "selection": frozenset(),
    "engine": frozenset(),
    "migration": frozenset({"schema", "elided_excerpt"}),
    "process": frozenset(),
    "infra": frozenset(),
    "unknown": frozenset(),
}
CLASS_SOURCES = frozenset({"hand", "rule", "triage"})


class UnsafeBundlePath(ValueError):
    """A bundle path escapes its root, is a symlink, or is not a regular file."""


# --------------------------------------------------------------------------
# Issue classification
# --------------------------------------------------------------------------


def issue_rule(issue: str) -> str:
    """Bracketed rule tag of an analyzer issue, or ``other``."""

    match = RULE_PATTERN.search(issue)
    return match.group(1) if match else "other"


def recorded_issue_rule(issue: str) -> str:
    """Classify a recorded pipeline issue, including untagged CI failures."""

    match = RULE_PATTERN.search(issue)
    if match:
        return match.group(1)
    lowered = issue.lower()
    if "proof source evidence not found" in lowered:
        return "proof:excerpt-not-found"
    if re.search(r"test case .* expected .* got", lowered):
        return "tests:wrong-expected-value"
    if "execution failed" in lowered:
        return "tests:execution-failed"
    if "yaml parse failed" in lowered:
        return "tests:yaml-parse"
    if "test input assignment missing" in lowered:
        return "tests:missing-input"
    if "compile" in lowered and "failed" in lowered:
        return "compile"
    if "ungrounded" in lowered or ("embedded" in lowered and "literal" in lowered):
        return "grounding:literal"
    if "oracle" in lowered:
        return "oracle"
    return "other"


def rule_counts(issues: Iterable[str], classify=issue_rule) -> dict[str, int]:
    counts = collections.Counter(classify(issue) for issue in issues)
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# Safe reading
# --------------------------------------------------------------------------


def _read_inside(root: Path, relative: str | PurePosixPath) -> bytes:
    """Read a regular, non-symlink file that stays under ``root``."""

    relative_path = PurePosixPath(str(relative))
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or any(part in {"", ".", ".."} for part in relative_path.parts)
    ):
        raise UnsafeBundlePath(f"unsafe relative path: {relative}")
    if root.is_symlink():
        raise UnsafeBundlePath(f"symlinked root: {root}")
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise UnsafeBundlePath(f"symlink in bundle path: {current}")
    if not current.is_file():
        raise UnsafeBundlePath(f"not a regular file: {current}")
    if current.stat().st_size > MAX_FILE_BYTES:
        raise UnsafeBundlePath(f"file exceeds {MAX_FILE_BYTES} bytes: {current}")
    return current.read_bytes()


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _rulespec_text(data: bytes) -> str:
    """Decode RuleSpec bytes as ``Path.read_text`` does in the CI validator."""

    return data.decode().replace("\r\n", "\n").replace("\r", "\n")


def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _declared_citations(content: str) -> list[str]:
    try:
        payload = yaml.safe_load(content)
    except yaml.YAMLError:
        return []
    module = payload.get("module") if isinstance(payload, dict) else None
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    paths = (
        verification.get("corpus_citation_path")
        if isinstance(verification, dict)
        else None
    )
    if isinstance(paths, str):
        return [paths]
    if not isinstance(paths, list):
        return []
    return [path for path in paths if isinstance(path, str)]


def _dashed(citation: str) -> str:
    return citation.replace("/", "-")


# --------------------------------------------------------------------------
# Replay cases
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayCase:
    """One candidate module with everything the analyzer replay needs."""

    case_id: str
    run_id: str
    citation: str
    module_path: str
    candidate: str
    tests: str | None
    source: str
    source_metadata: Mapping[str, Any] | None
    policy_prefix: str | None
    context: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _import_items(content: str) -> list[str]:
    from axiom_encode.harness.validator_pipeline import (
        _extract_import_items_from_content,
    )

    return _extract_import_items_from_content(content)


def _deferral_targets(content: str) -> list[str]:
    """``blocked_by`` targets, parsed the way the CI validator parses them."""

    from axiom_encode.harness.validator_pipeline import _safe_load_unique_keys

    try:
        payload = _safe_load_unique_keys(content)
    except (ValueError, yaml.YAMLError):
        return []
    module = payload.get("module", {}) if isinstance(payload, dict) else {}
    records = module.get("deferred_outputs", []) if isinstance(module, dict) else []
    if not isinstance(records, list):
        return []
    return sorted(
        {
            target
            for record in records
            if isinstance(record, dict) and isinstance(record.get("blocked_by"), list)
            for target in record["blocked_by"]
            if isinstance(target, str) and DEFERRAL_TARGET_PATTERN.fullmatch(target)
        }
    )


def _context_key(target: str, policy_prefix: str | None) -> str | None:
    """Canonical ``prefix:path`` key for an import or deferral target.

    Returns ``None`` where the CI resolver would not resolve the target: an
    unsafe path, or a path outside the atomic RuleSpec module roots.
    """

    from axiom_encode.constants import RULESPEC_ATOMIC_MODULE_ROOTS

    target = target.split("#", 1)[0].strip().strip("\"'")
    if ":" not in target:
        # Unprefixed imports resolve inside the active checkout.
        if not policy_prefix:
            return None
        prefix, path = policy_prefix, target.strip().strip("/")
    else:
        # Prefixed targets follow ``_parse_rulespec_target``.
        prefix, _, path = target.partition(":")
        if path != path.strip() or path.startswith("/") or path.endswith("/"):
            return None
    if path.endswith(".yaml"):
        path = path[: -len(".yaml")]
    parts = path.split("/")
    if (
        not PREFIX_PATTERN.fullmatch(prefix)
        or not path
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0] not in RULESPEC_ATOMIC_MODULE_ROOTS
    ):
        return None
    return f"{prefix}:{path}"


def _lacks_policy_prefix(target: str, policy_prefix: str | None) -> bool:
    """An unprefixed target the replay cannot place without a policy prefix."""

    return not policy_prefix and ":" not in target.split("#", 1)[0]


def referenced_context_keys(content: str, policy_prefix: str | None) -> list[str]:
    """Context modules a candidate needs: its imports and deferral providers."""

    targets = [item.rpartition("#")[0] for item in _import_items(content)]
    targets.extend(_deferral_targets(content))
    keys = [_context_key(target, policy_prefix) for target in targets if target]
    return sorted({key for key in keys if key is not None})


def import_symbol_contents(case: ReplayCase) -> tuple[list[tuple[str, str]], list[str]]:
    """Bind each directly imported symbol to its context module.

    Mirrors ``ValidatorPipeline._complete_source_unit_import_symbol_contents``:
    one binding per (resolved module, symbol); unresolved items are skipped
    (and reported here so a replay can say how much it could not resolve).
    """

    bindings: list[tuple[str, str]] = []
    unresolved: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in _import_items(case.candidate):
        target, separator, symbol = item.rpartition("#")
        symbol = symbol.strip()
        if not separator or not symbol:
            continue
        if _lacks_policy_prefix(target, case.policy_prefix):
            unresolved.append(item)
            continue
        key = _context_key(target, case.policy_prefix)
        if key is None:
            continue
        content = case.context.get(key)
        if content is None:
            unresolved.append(item)
            continue
        if (key, symbol) in seen:
            continue
        seen.add((key, symbol))
        bindings.append((symbol, content))
    return bindings, unresolved


def resolved_deferral_outputs(case: ReplayCase) -> tuple[tuple[str, ...], list[str]]:
    """Mirror ``ValidatorPipeline._complete_source_unit_deferred_outputs``.

    Returns the resolved targets and the targets whose provider module is not
    in the case context (production may have resolved those).
    """

    from axiom_encode.harness.source_completeness import (
        _IRS_REV_PROC_2025_32_BEHAVIOR_EXPORTS,
    )
    from axiom_encode.harness.validator_pipeline import _safe_load_unique_keys

    resolved: list[str] = []
    unresolved: list[str] = []
    for target in _deferral_targets(case.candidate):
        key = _context_key(target, case.policy_prefix)
        if key is None:
            continue
        content = case.context.get(key)
        if content is None:
            unresolved.append(target)
            continue
        try:
            provider = _safe_load_unique_keys(content)
        except (ValueError, yaml.YAMLError):
            continue
        if not isinstance(provider, dict) or provider.get("format") != "rulespec/v1":
            continue
        rules = provider.get("rules")
        if not isinstance(rules, list):
            continue
        symbol = target.partition("#")[2]
        exports = [
            rule
            for rule in rules
            if isinstance(rule, dict) and rule.get("name") == symbol
        ]
        allowed_kinds = (
            {"derived"}
            if target in _IRS_REV_PROC_2025_32_BEHAVIOR_EXPORTS
            else {"parameter", "derived", "relation", "derived_relation"}
        )
        if len(exports) == 1 and exports[0].get("kind") in allowed_kinds:
            resolved.append(target)
    return tuple(resolved), unresolved


def _companion_cases(tests: str | None) -> list[object] | None:
    """Load companion tests the way the CI validator hands them to the analyzer."""

    from axiom_encode.harness.validator_pipeline import _safe_load_unique_keys

    if tests is None:
        return None
    try:
        payload = _safe_load_unique_keys(tests)
    except (ValueError, yaml.YAMLError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    return payload if isinstance(payload, list) else None


def replay_case(case: ReplayCase, *, use_context: bool = True) -> dict[str, Any]:
    """Run ``analyze_complete_source_unit`` on one case as the CI validator does.

    ``use_context=False`` leaves out imported symbol contents, resolved
    deferral providers and same-act aliases from the source metadata.
    """

    from axiom_encode.harness.source_completeness import (
        analyze_complete_source_unit,
        collect_artifact_numeric_bindings,
    )
    from axiom_encode.harness.validator_pipeline import (
        _authenticated_same_act_aliases_from_metadata,
        _numeric_profile_for_citation_path,
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    if use_context:
        bindings, unresolved_imports = import_symbol_contents(case)
        deferral_outputs, unresolved_deferrals = resolved_deferral_outputs(case)
        aliases = _authenticated_same_act_aliases_from_metadata(
            dict(case.source_metadata) if case.source_metadata else None
        )
    else:
        bindings, unresolved_imports = [], []
        deferral_outputs, unresolved_deferrals = (), []
        aliases = ()
    result: dict[str, Any] = {
        "id": case.case_id,
        "run_id": case.run_id,
        "citation": case.citation,
        "module": case.module_path,
        "imports_resolved": len(bindings),
        "imports_unresolved": unresolved_imports,
        "deferral_providers_resolved": list(deferral_outputs),
        "deferral_providers_unresolved": unresolved_deferrals,
        "expected_class": case.provenance.get("expected_class"),
        "expected_subclass": case.provenance.get("expected_subclass"),
    }
    try:
        profile = _numeric_profile_for_citation_path(case.citation)
        numeric_bindings = collect_artifact_numeric_bindings(
            case.candidate,
            extract_named_scalars=extract_named_scalar_occurrences,
            imported_symbol_contents=bindings,
        )
        analysis = analyze_complete_source_unit(
            case.candidate,
            case.source,
            corpus_citation_path=case.citation,
            test_cases=_companion_cases(case.tests),
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text, profile=profile
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile=profile
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            artifact_numeric_values=tuple(value for _name, value in numeric_bindings),
            artifact_numeric_bindings=numeric_bindings,
            imported_symbol_contents=bindings,
            resolved_dependency_outputs=deferral_outputs,
            authenticated_same_act_aliases=aliases,
        )
    except Exception as exc:  # report, never hide
        result.update(issues=[], rules={}, error=f"{type(exc).__name__}: {exc}"[:500])
        return result
    issues = list(analysis.issues)
    result.update(issues=issues, rules=rule_counts(issues), error=None)
    return result


# --------------------------------------------------------------------------
# Committed fixtures
# --------------------------------------------------------------------------


def fixture_label_errors(metadata: Mapping[str, Any]) -> list[str]:
    """Problems with the optional failure-class label of one fixture."""

    label_keys = ("expected_subclass", "class_evidence", "class_source")
    if "expected_class" not in metadata:
        present = [key for key in label_keys if key in metadata]
        return [f"{key} without expected_class" for key in present]
    errors: list[str] = []
    expected_class = metadata["expected_class"]
    if expected_class not in FAILURE_CLASSES:
        return [f"unknown expected_class {expected_class!r}"]
    if "expected_subclass" in metadata:
        subclass = metadata["expected_subclass"]
        if subclass not in FAILURE_CLASSES[expected_class]:
            errors.append(
                f"expected_subclass {subclass!r} is not a subclass of "
                f"{expected_class!r}"
            )
    evidence = metadata.get("class_evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        errors.append("expected_class needs a non-empty class_evidence")
    if metadata.get("class_source") not in CLASS_SOURCES:
        errors.append(f"class_source must be one of {sorted(CLASS_SOURCES)}")
    return errors


def load_fixture(directory: Path) -> ReplayCase:
    """Load and authenticate one committed fixture directory."""

    metadata = _json_object(_read_inside(directory, FIXTURE_FILE), FIXTURE_FILE)
    if metadata.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"{directory}: unexpected fixture schema")
    label_errors = fixture_label_errors(metadata)
    if label_errors:
        raise ValueError(f"{directory}: {'; '.join(label_errors)}")
    files: Mapping[str, str] = metadata["files"]
    raw: dict[str, bytes] = {}
    for relative, digest in files.items():
        data = _read_inside(directory, relative)
        if _sha256(data) != digest:
            raise ValueError(f"{directory}/{relative}: sha256 mismatch")
        raw[relative] = data
    on_disk = sorted(
        relative
        for path in directory.rglob("*")
        if path.is_file()
        and (relative := path.relative_to(directory).as_posix()) != FIXTURE_FILE
    )
    if on_disk != sorted(files):
        raise ValueError(
            f"{directory}: files on disk differ from {FIXTURE_FILE}: "
            f"{sorted(set(on_disk) ^ set(files))}"
        )
    source_metadata = (
        _json_object(raw["source-metadata.json"], "source-metadata.json")
        if "source-metadata.json" in raw
        else None
    )
    context = {
        key: _rulespec_text(raw[relative])
        for key, relative in metadata["context_imports"].items()
    }
    return ReplayCase(
        case_id=directory.name,
        run_id=metadata["run_id"],
        citation=metadata["citation"],
        module_path=metadata["module_path"],
        candidate=_rulespec_text(raw["candidate.yaml"]),
        tests=(
            _rulespec_text(raw["candidate.test.yaml"])
            if "candidate.test.yaml" in raw
            else None
        ),
        source=raw["source.txt"].decode(),
        source_metadata=source_metadata,
        policy_prefix=metadata.get("policy_prefix"),
        context=context,
        provenance=metadata,
    )


def fixture_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob(f"*/{FIXTURE_FILE}"))


def load_fixtures(root: Path) -> list[ReplayCase]:
    return [load_fixture(directory) for directory in fixture_directories(root)]


# --------------------------------------------------------------------------
# Extracted failure bundles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleModule:
    """A rejected candidate module located inside an extracted bundle."""

    bundle_root: Path
    run_id: str
    bundle_citation: str
    citation: str
    lane: str
    module_path: str
    runner: str | None
    workspace: str
    metadata: Mapping[str, Any]
    recorded: Mapping[str, Any]

    @property
    def candidate_dir(self) -> str:
        return f"generated/{self.lane}/final-rejected-candidate"


def _bundle_root(path: Path) -> Path:
    root = path / "x" if (path / "x" / "metadata.json").is_file() else path
    if path.is_symlink() or root.is_symlink():
        raise UnsafeBundlePath(f"symlinked bundle root: {root}")
    return root


def _bundle_run_id(root: Path, metadata: Mapping[str, Any]) -> str:
    run_id = str(metadata.get("workflow_run_id") or "")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        name = root.parent.name if root.name == "x" else root.name
        run_id = name if RUN_ID_PATTERN.fullmatch(name) else ""
    if not run_id:
        raise ValueError(f"{root}: cannot determine the workflow run id")
    return run_id


def discover_bundle_modules(path: Path) -> list[BundleModule]:
    """Find every rejected candidate module and its workspace in one bundle."""

    root = _bundle_root(path)
    metadata = _json_object(_read_inside(root, "metadata.json"), "metadata.json")
    run_id = _bundle_run_id(root, metadata)
    bundle_citation = str(metadata.get("citation") or "")
    found: list[BundleModule] = []
    generated = root / "generated"
    if not generated.is_dir() or generated.is_symlink():
        return found
    for lane_dir in sorted(generated.iterdir()):
        if lane_dir.is_symlink() or not SAFE_SEGMENT.fullmatch(lane_dir.name):
            continue
        candidate_dir = lane_dir / "final-rejected-candidate"
        if candidate_dir.is_symlink() or not (candidate_dir / "issues.json").is_file():
            continue
        recorded = _json_object(
            _read_inside(candidate_dir, "issues.json"), "issues.json"
        )
        workspaces_root = lane_dir / "_eval_workspaces"
        for module in sorted(candidate_dir.rglob("*.yaml")):
            if module.name.endswith(".test.yaml") or module.is_symlink():
                continue
            module_path = module.relative_to(candidate_dir).as_posix()
            content = _rulespec_text(_read_inside(candidate_dir, module_path))
            declared = _declared_citations(content)

            def workspace_for(citation: str) -> list[Path]:
                dashed = _dashed(citation)
                if (
                    not SAFE_SEGMENT.fullmatch(dashed)
                    or workspaces_root.is_symlink()
                    or not workspaces_root.is_dir()
                ):
                    return []
                return sorted(
                    runner_dir / dashed / "workspace"
                    for runner_dir in workspaces_root.iterdir()
                    if SAFE_SEGMENT.fullmatch(runner_dir.name)
                    and not runner_dir.is_symlink()
                    and (runner_dir / dashed / "workspace" / "source.txt").is_file()
                )

            ordered = ([bundle_citation] if bundle_citation in declared else []) + [
                citation for citation in declared if citation != bundle_citation
            ]
            citation = next((c for c in ordered if workspace_for(c)), None)
            if citation is None:
                continue
            workspaces = workspace_for(citation)
            runner = None
            candidate_bytes = _read_inside(candidate_dir, module_path)
            for workspace in workspaces:
                runner_name = workspace.parent.parent.name
                try:
                    generated_copy = _read_inside(
                        lane_dir, f"{runner_name}/{module_path}"
                    )
                except UnsafeBundlePath:
                    continue
                if generated_copy == candidate_bytes:
                    runner = runner_name
                    break
            chosen = next(
                (w for w in workspaces if w.parent.parent.name == runner),
                workspaces[0],
            )
            found.append(
                BundleModule(
                    bundle_root=root,
                    run_id=run_id,
                    bundle_citation=bundle_citation,
                    citation=citation,
                    lane=lane_dir.name,
                    module_path=module_path,
                    runner=runner,
                    workspace=chosen.relative_to(root).as_posix(),
                    metadata=metadata,
                    recorded=recorded,
                )
            )
    return found


def _bundle_context_map(module: BundleModule) -> tuple[dict[str, str], str | None]:
    """Map ``prefix:path`` keys to workspace-relative context files."""

    root = module.bundle_root
    manifest_path = f"{module.workspace}/context-manifest.json"
    if not (root / manifest_path).is_file():
        return {}, None
    manifest = _json_object(_read_inside(root, manifest_path), "context-manifest.json")
    policy_prefix = manifest.get("policy_prefix")
    if not isinstance(policy_prefix, str) or not PREFIX_PATTERN.fullmatch(
        policy_prefix
    ):
        policy_prefix = None
    context_files = manifest.get("context_files", [])
    if not isinstance(context_files, list):
        raise ValueError("context-manifest.json context_files is not a list")
    mapping: dict[str, str] = {}
    for item in context_files:
        if not isinstance(item, dict):
            continue
        import_path = item.get("import_path")
        workspace_path = item.get("workspace_path")
        if not isinstance(import_path, str) or not isinstance(workspace_path, str):
            continue
        if not workspace_path.startswith("context/") or not workspace_path.endswith(
            ".yaml"
        ):
            continue
        key = _context_key(import_path, policy_prefix)
        if key is not None:
            mapping.setdefault(key, f"{module.workspace}/{workspace_path}")
    return mapping, policy_prefix


def _bundle_context_bytes(
    module: BundleModule, candidate: str, rulespec_root: Path | None
) -> tuple[dict[str, tuple[str, bytes]], str | None]:
    """Referenced context modules as ``key -> (fixture path, bytes)``.

    The workspace context comes first. ``rulespec_root`` (a RuleSpec checkout
    at the bundle's ``rulespec_ref``, laid out as ``<prefix>/<path>.yaml``)
    fills only keys that the workspace context lacks.
    """

    mapping, policy_prefix = _bundle_context_map(module)
    found: dict[str, tuple[str, bytes]] = {}
    for key in referenced_context_keys(candidate, policy_prefix):
        relative = mapping.get(key)
        if relative is not None:
            found[key] = (
                relative[len(module.workspace) + 1 :],
                _read_inside(module.bundle_root, relative),
            )
            continue
        if rulespec_root is None:
            continue
        prefix, _, path = key.partition(":")
        provider = rulespec_root / prefix / f"{path}.yaml"
        if not provider.exists() and not provider.is_symlink():
            continue
        # Raises on symlinks or non-files instead of silently skipping them.
        data = _read_inside(rulespec_root, f"{prefix}/{path}.yaml")
        found[key] = (f"providers/{prefix}/{path}.yaml", data)
    return found, policy_prefix


def bundle_case(
    module: BundleModule, *, rulespec_root: Path | None = None
) -> ReplayCase:
    """Build a replay case straight from an extracted bundle."""

    root = module.bundle_root
    candidate = _rulespec_text(
        _read_inside(root, f"{module.candidate_dir}/{module.module_path}")
    )
    tests_path = (
        f"{module.candidate_dir}/{module.module_path[: -len('.yaml')]}.test.yaml"
    )
    tests = (
        _rulespec_text(_read_inside(root, tests_path))
        if (root / tests_path).is_file()
        else None
    )
    source = _read_inside(root, f"{module.workspace}/source.txt").decode()
    source_metadata = (
        _json_object(
            _read_inside(root, f"{module.workspace}/source-metadata.json"),
            "source-metadata.json",
        )
        if (root / module.workspace / "source-metadata.json").is_file()
        else None
    )
    context, policy_prefix = _bundle_context_bytes(module, candidate, rulespec_root)
    return ReplayCase(
        case_id=f"{module.run_id}:{module.lane}:{module.module_path}",
        run_id=module.run_id,
        citation=module.citation,
        module_path=module.module_path,
        candidate=candidate,
        tests=tests,
        source=source,
        source_metadata=source_metadata,
        policy_prefix=policy_prefix,
        context={key: _rulespec_text(data) for key, (_p, data) in context.items()},
    )


def bundle_paths(directory: Path) -> list[Path]:
    paths = []
    for child in sorted(directory.iterdir()):
        # A symlinked ``x`` is kept so that discovery reports it as a load error.
        if child.is_symlink() or not child.is_dir():
            continue
        if (child / "x" / "metadata.json").is_file() or (
            child / "metadata.json"
        ).is_file():
            paths.append(child)
    return paths


BUNDLE_ERRORS = (OSError, ValueError, UnicodeDecodeError, TypeError, AttributeError)


def load_bundle_cases(
    directory: Path, *, latest_per_citation: bool = False
) -> tuple[list[ReplayCase], list[dict[str, Any]]]:
    modules: list[BundleModule] = []
    errors: list[dict[str, Any]] = []
    for path in bundle_paths(directory):
        try:
            modules.extend(discover_bundle_modules(path))
        except BUNDLE_ERRORS as exc:
            errors.append(
                {"bundle": path.name, "error": f"{type(exc).__name__}: {exc}"}
            )
    if latest_per_citation:
        latest: dict[str, str] = {}
        for module in modules:
            if int(module.run_id) > int(latest.get(module.citation, "0")):
                latest[module.citation] = module.run_id
        modules = [m for m in modules if latest[m.citation] == m.run_id]
    cases = []
    for module in modules:
        try:
            cases.append(bundle_case(module))
        except BUNDLE_ERRORS as exc:
            errors.append(
                {
                    "bundle": module.run_id,
                    "module": module.module_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return cases, errors


def extract_fixture(
    module: BundleModule,
    fixture_root: Path,
    *,
    name: str | None = None,
    rulespec_root: Path | None = None,
) -> Path:
    """Copy one bundle module into a self-authenticating committed fixture."""

    root = module.bundle_root
    target = fixture_root / (name or module.run_id)
    if target.exists():
        raise FileExistsError(f"fixture already exists: {target}")
    candidate_bytes = _read_inside(root, f"{module.candidate_dir}/{module.module_path}")
    candidate = _rulespec_text(candidate_bytes)
    files: dict[str, bytes] = {
        "candidate.yaml": candidate_bytes,
        "source.txt": _read_inside(root, f"{module.workspace}/source.txt"),
        "issues.json": _read_inside(root, f"{module.candidate_dir}/issues.json"),
    }
    tests_path = (
        f"{module.candidate_dir}/{module.module_path[: -len('.yaml')]}.test.yaml"
    )
    if (root / tests_path).is_file():
        files["candidate.test.yaml"] = _read_inside(root, tests_path)
    if (root / module.workspace / "source-metadata.json").is_file():
        files["source-metadata.json"] = _read_inside(
            root, f"{module.workspace}/source-metadata.json"
        )
    context, policy_prefix = _bundle_context_bytes(module, candidate, rulespec_root)
    context_imports: dict[str, str] = {}
    context_origins: dict[str, str] = {}
    for key, (relative, data) in sorted(context.items()):
        files[relative] = data
        context_imports[key] = relative
        context_origins[key] = (
            ORIGIN_RULESPEC_REF
            if relative.startswith("providers/")
            else CONTEXT_WORKSPACE
        )
    inventory = {
        item["path"]: item["sha256"]
        for item in module.metadata.get("files", [])
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    candidate_inventory_sha = inventory.get(
        f"{module.lane}/final-rejected-candidate/{module.module_path}"
    )
    recorded_issues = [
        issue for issue in module.recorded.get("issues", []) if isinstance(issue, str)
    ]
    module_issues = [
        issue for issue in recorded_issues if issue.startswith(f"{module.module_path}:")
    ] or recorded_issues
    source_metadata = (
        _json_object(files["source-metadata.json"], "source-metadata.json")
        if "source-metadata.json" in files
        else {}
    )
    attestation = source_metadata.get("source_attestation")
    resolved_text_sha = (
        attestation.get("resolved_text_sha256")
        if isinstance(attestation, dict)
        else None
    )
    tests_sha = (
        _sha256(files["candidate.test.yaml"])
        if "candidate.test.yaml" in files
        else None
    )
    provenance = {
        "schema": FIXTURE_SCHEMA,
        "run_id": module.run_id,
        "run_url": RUN_URL.format(run_id=module.run_id),
        "bundle_citation": module.bundle_citation,
        "citation": module.citation,
        "lane": module.lane,
        "runner": module.runner,
        "module_path": module.module_path,
        "policy_prefix": policy_prefix,
        "encoder_commit": module.metadata.get("encoder_commit"),
        "encoder_version": module.recorded.get("encoder_version"),
        "attempt_count": module.recorded.get("attempt_count"),
        "corpus_ref": module.metadata.get("corpus_ref"),
        "rulespec_ref": module.metadata.get("rulespec_ref"),
        "failed_steps": module.metadata.get("failed_steps"),
        "candidate_matches_bundle_inventory": (
            candidate_inventory_sha == _sha256(candidate_bytes)
            if candidate_inventory_sha
            else None
        ),
        "candidate_matches_recorded_issues": (
            module.recorded.get("path") == module.module_path
            and module.recorded.get("rulespec_sha256") == _sha256(candidate_bytes)
            and module.recorded.get("tests_sha256") == tests_sha
        ),
        "source_matches_attestation": (
            resolved_text_sha == _sha256(files["source.txt"])
            if resolved_text_sha
            else None
        ),
        "recorded_rules": rule_counts(module_issues, classify=recorded_issue_rule),
        "context_imports": context_imports,
        "context_origins": context_origins,
        # Git blob ids let a reviewer check provider bytes against rulespec_ref
        # (``gh api .../contents/<path>?ref=<rulespec_ref>`` returns ``sha``).
        "provider_git_blob_sha1": {
            relative: _git_blob_sha1(files[relative])
            for key, relative in context_imports.items()
            if context_origins[key] == ORIGIN_RULESPEC_REF
        },
        "files": {path: _sha256(data) for path, data in sorted(files.items())},
    }
    for relative, data in files.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (target / FIXTURE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return target


# --------------------------------------------------------------------------
# Reports, comparison and the snapshot ratchet
# --------------------------------------------------------------------------


def build_report(
    cases: Sequence[ReplayCase],
    *,
    use_context: bool = True,
    errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    from axiom_encode import __version__

    results = []
    for case in cases:
        started = time.perf_counter()
        result = replay_case(case, use_context=use_context)
        result["seconds"] = round(time.perf_counter() - started, 3)
        results.append(result)
    rule_totals: collections.Counter[str] = collections.Counter()
    cases_with_rule: collections.Counter[str] = collections.Counter()
    expected_classes: collections.Counter[str] = collections.Counter()
    for result in results:
        rule_totals.update(result["rules"])
        cases_with_rule.update(result["rules"].keys())
        expected_classes[result.get("expected_class") or UNLABELED] += 1
    summary = {
        "cases": len(results),
        "clean": sum(1 for r in results if not r["issues"] and not r["error"]),
        "errors": sum(1 for r in results if r["error"]) + len(errors),
        "issues": sum(len(r["issues"]) for r in results),
        "imports_resolved": sum(r["imports_resolved"] for r in results),
        "imports_unresolved": sum(len(r["imports_unresolved"]) for r in results),
        "deferral_providers_unresolved": sum(
            len(r["deferral_providers_unresolved"]) for r in results
        ),
        "rules": dict(sorted(rule_totals.items())),
        "cases_with_rule": dict(sorted(cases_with_rule.items())),
        "expected_classes": dict(sorted(expected_classes.items())),
    }
    return {
        "schema": REPORT_SCHEMA,
        "encoder_version": __version__,
        "context": CONTEXT_WORKSPACE if use_context else CONTEXT_NONE,
        "summary": summary,
        "cases": results,
        "load_errors": list(errors),
    }


def format_summary(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"replayed {summary['cases']} modules (context: {report['context']}): "
        f"{summary['clean']} clean, {summary['issues']} issues, "
        f"{summary['errors']} errors",
        f"imports resolved {summary['imports_resolved']}, unresolved "
        f"{summary['imports_unresolved']}; deferral providers unresolved "
        f"{summary['deferral_providers_unresolved']}",
        f"  {'rule':48} {'issues':>7} {'modules':>8}",
    ]
    for rule, count in sorted(summary["rules"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {rule:48} {count:7} {summary['cases_with_rule'][rule]:8}")
    expected_classes = summary.get("expected_classes") or {}
    if set(expected_classes) - {UNLABELED}:
        lines.append(f"  {'expected class (hand label)':48} {'modules':>7}")
        for name, count in sorted(
            expected_classes.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"  {name:48} {count:7}")
    return "\n".join(lines)


def compare_reports(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Issue-level difference between two replay reports of the same cases.

    A case that errors on either side is listed under ``errored`` and left out
    of every count, so a crash never reads as removed issues.
    """

    before_cases = {r["id"]: r for r in before["cases"]}
    after_cases = {r["id"]: r for r in after["cases"]}
    shared = sorted(before_cases.keys() & after_cases.keys())
    errored = [
        {
            "id": case_id,
            "before": before_cases[case_id]["error"],
            "after": after_cases[case_id]["error"],
        }
        for case_id in shared
        if before_cases[case_id]["error"] or after_cases[case_id]["error"]
    ]
    compared = [
        case_id
        for case_id in shared
        if not before_cases[case_id]["error"] and not after_cases[case_id]["error"]
    ]
    removed: collections.Counter[str] = collections.Counter()
    added: collections.Counter[str] = collections.Counter()
    newly_clean, newly_failing, changed = [], [], []
    for case_id in compared:
        old = collections.Counter(before_cases[case_id]["issues"])
        new = collections.Counter(after_cases[case_id]["issues"])
        gone, came = old - new, new - old
        removed.update(issue_rule(issue) for issue in gone.elements())
        added.update(issue_rule(issue) for issue in came.elements())
        if old and not new:
            newly_clean.append(case_id)
        if new and not old:
            newly_failing.append(case_id)
        if gone or came:
            changed.append(
                {
                    "id": case_id,
                    "citation": after_cases[case_id]["citation"],
                    "before": sum(old.values()),
                    "after": sum(new.values()),
                    "removed": sorted(gone.elements()),
                    "added": sorted(came.elements()),
                }
            )
    return {
        "compared": len(compared),
        "errored": errored,
        "only_before": sorted(before_cases.keys() - after_cases.keys()),
        "only_after": sorted(after_cases.keys() - before_cases.keys()),
        "issues_before": sum(len(before_cases[c]["issues"]) for c in compared),
        "issues_after": sum(len(after_cases[c]["issues"]) for c in compared),
        "clean_before": sum(1 for c in compared if not before_cases[c]["issues"]),
        "clean_after": sum(1 for c in compared if not after_cases[c]["issues"]),
        "removed_by_rule": dict(removed.most_common()),
        "added_by_rule": dict(added.most_common()),
        "newly_clean": newly_clean,
        "newly_failing": newly_failing,
        "changed": changed,
    }


def format_comparison(diff: Mapping[str, Any]) -> str:
    lines = [
        f"compared modules: {diff['compared']}",
        f"issues: {diff['issues_before']} -> {diff['issues_after']}; "
        f"clean modules: {diff['clean_before']} -> {diff['clean_after']}",
        f"removed issues by rule: {diff['removed_by_rule']}",
        f"added issues by rule:   {diff['added_by_rule']}",
        f"newly clean: {len(diff['newly_clean'])}  newly failing: "
        f"{len(diff['newly_failing'])}",
    ]
    if diff["errored"]:
        lines.append(f"ERRORED (not compared): {len(diff['errored'])}")
        lines.extend(
            f"  {item['id']}: before={item['before']!r} after={item['after']!r}"
            for item in diff["errored"]
        )
    if diff["only_before"] or diff["only_after"]:
        lines.append(
            f"not compared: {len(diff['only_before'])} only before, "
            f"{len(diff['only_after'])} only after"
        )
    for change in diff["changed"]:
        lines.append(
            f"  {change['id']} ({change['citation']}): "
            f"{change['before']} -> {change['after']}"
        )
    return "\n".join(lines)


def snapshot_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report["context"] != CONTEXT_WORKSPACE:
        raise ValueError("the snapshot pins the replay with workspace context")
    return {
        "schema": SNAPSHOT_SCHEMA,
        "context": report["context"],
        "totals": {
            "cases": report["summary"]["cases"],
            "clean": report["summary"]["clean"],
            "issues": report["summary"]["issues"],
            "rules": report["summary"]["rules"],
        },
        "cases": {
            result["id"]: {
                "citation": result["citation"],
                "issues": len(result["issues"]),
                "rules": result["rules"],
                "error": result["error"],
            }
            for result in sorted(report["cases"], key=lambda r: r["id"])
        },
    }


@dataclass(frozen=True)
class SnapshotDeviations:
    """How a replay differs from the pinned snapshot."""

    increases: list[str]
    decreases: list[str]
    unpinned: list[str]

    def __bool__(self) -> bool:
        return bool(self.increases or self.decreases or self.unpinned)


def snapshot_deviations(
    snapshot: Mapping[str, Any], report: Mapping[str, Any]
) -> SnapshotDeviations:
    """Compare per-case, per-rule counts with the pinned snapshot.

    Increases are rises, new replay errors and pinned fixtures that are gone.
    Unpinned fixtures are new fixtures that the snapshot does not list yet.
    """

    if snapshot.get("schema") != SNAPSHOT_SCHEMA or not isinstance(
        snapshot.get("cases"), dict
    ):
        raise ValueError("not a US replay snapshot")
    increases: list[str] = []
    decreases: list[str] = []
    unpinned: list[str] = []
    pinned = snapshot["cases"]
    current = snapshot_from_report(report)["cases"]
    for case_id in sorted(pinned.keys() | current.keys()):
        if case_id not in current:
            increases.append(f"{case_id}: pinned fixture is missing from the replay")
            continue
        if case_id not in pinned:
            unpinned.append(f"{case_id}: fixture is not pinned in the snapshot")
            continue
        old, new = pinned[case_id], current[case_id]
        if new["error"] != old["error"]:
            message = f"{case_id}: replay error changed to {new['error']!r}"
            (increases if new["error"] else decreases).append(message)
        for rule in sorted(old["rules"].keys() | new["rules"].keys()):
            before, after = old["rules"].get(rule, 0), new["rules"].get(rule, 0)
            message = f"{case_id} ({new['citation']}) {rule}: {before} -> {after}"
            if after > before:
                increases.append(message)
            elif after < before:
                decreases.append(message)
    return SnapshotDeviations(increases, decreases, unpinned)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_json(path: str | None, payload: Mapping[str, Any]) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=1) + "\n")


def _cmd_replay(args: argparse.Namespace) -> int:
    if args.bundles:
        cases, errors = load_bundle_cases(
            Path(args.bundles), latest_per_citation=args.latest_per_citation
        )
    else:
        cases, errors = load_fixtures(Path(args.fixtures)), []
    report = build_report(cases, use_context=not args.no_context, errors=errors)
    _write_json(args.out, report)
    print(format_summary(report))
    for error in errors:
        print(f"  load error: {error}", file=sys.stderr)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    diff = compare_reports(before, after)
    if args.json:
        print(json.dumps(diff, indent=1))
    else:
        print(format_comparison(diff))
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    root = Path(args.fixtures)
    report = build_report(load_fixtures(root))
    snapshot_path = root / SNAPSHOT_FILE
    snapshot = (
        json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
    )
    deviations = (
        snapshot_deviations(snapshot, report)
        if snapshot is not None
        else SnapshotDeviations([], [], [])
    )
    for line in deviations.increases:
        print(f"INCREASE {line}")
    for line in deviations.decreases:
        print(f"decrease {line}")
    for line in deviations.unpinned:
        print(f"new      {line}")
    if args.check:
        if snapshot is None:
            print(f"no snapshot at {snapshot_path}", file=sys.stderr)
            return 1
        return 1 if deviations else 0
    if deviations.increases and not args.allow_increase:
        print(
            "refusing to raise pinned counts; rerun with --allow-increase and "
            "justify each increase in the PR",
            file=sys.stderr,
        )
        return 1
    snapshot_path.write_text(json.dumps(snapshot_from_report(report), indent=1) + "\n")
    print(format_summary(report))
    print(f"wrote {snapshot_path}")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    modules = discover_bundle_modules(Path(args.bundle))
    if args.module:
        modules = [m for m in modules if m.module_path == args.module]
    if not modules:
        print("no replayable candidate module found", file=sys.stderr)
        return 1
    if len(modules) > 1 and not args.module:
        print(
            "bundle has several candidate modules; pass --module: "
            + ", ".join(m.module_path for m in modules),
            file=sys.stderr,
        )
        return 1
    rulespec_root = Path(args.rulespec_root) if args.rulespec_root else None
    if rulespec_root is not None and (
        rulespec_root.is_symlink() or not rulespec_root.is_dir()
    ):
        print(
            f"--rulespec-root is not a real directory: {rulespec_root}", file=sys.stderr
        )
        return 1
    target = extract_fixture(
        modules[0], Path(args.fixtures), name=args.name, rulespec_root=rulespec_root
    )
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"wrote {target} ({size} bytes)")
    case = load_fixture(target)
    unresolved = import_symbol_contents(case)[1] + resolved_deferral_outputs(case)[1]
    for item in unresolved:
        print(f"WARNING unresolved import or deferral provider: {item}")
    if unresolved:
        print(
            "delete the fixture and rerun extract with --rulespec-root <RuleSpec "
            "checkout at the run's rulespec_ref>",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay", help="replay fixtures or bundles")
    replay.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_ROOT))
    replay.add_argument("--bundles", help="directory of extracted failure bundles")
    replay.add_argument("--latest-per-citation", action="store_true")
    replay.add_argument(
        "--no-context",
        action="store_true",
        help="leave out imports, deferral providers and same-act aliases",
    )
    replay.add_argument("--out", help="write the full JSON report here")
    replay.set_defaults(func=_cmd_replay)

    compare = commands.add_parser("compare", help="compare two replay reports")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=_cmd_compare)

    snapshot = commands.add_parser("snapshot", help="check or rewrite the snapshot")
    snapshot.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_ROOT))
    snapshot.add_argument("--check", action="store_true")
    snapshot.add_argument(
        "--allow-increase",
        action="store_true",
        help="also record rises, new errors and removed fixtures",
    )
    snapshot.set_defaults(func=_cmd_snapshot)

    extract = commands.add_parser("extract", help="add a bundle as a fixture")
    extract.add_argument("bundle", help="extracted bundle (<run_id>/x or <run_id>)")
    extract.add_argument("--fixtures", default=str(DEFAULT_FIXTURE_ROOT))
    extract.add_argument("--module", help="candidate module path to extract")
    extract.add_argument("--name", help="fixture directory name (default: run id)")
    extract.add_argument(
        "--rulespec-root",
        help=(
            "RuleSpec checkout at the bundle's rulespec_ref (<prefix>/<path>.yaml); "
            "fills imports and deferral providers missing from the workspace context"
        ),
    )
    extract.set_defaults(func=_cmd_extract)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
