"""Score candidates through the signed apply overlay without installing them."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from .. import __version__
from ..corpus_resolver import LocalCorpusRelease, ResolvedCorpusSource

COMPLETE_SOURCE_UNIT_CATEGORIES = (
    "authoritative-source",
    "structure",
    "deferral",
    "formula-output",
    "source-explicit-conditions",
    "tests",
    "numeric-recall",
)
# Infrastructure classes a caller may report for a generation that left no
# artifact. The scorer never infers one: the harness passes it from evidence
# of its own (see ``evals._eval_result_generation_infrastructure_failure``).
GENERATION_INFRASTRUCTURE_FAILURES = ("authentication", "quota-exhaustion")
_CONTEXT_CATEGORIES = {
    "temporal-dependency-coverage",
    "existing-target-oracle-contract",
    "existing-target-naming-contract",
}
_ISSUE_FAMILIES = (
    (
        "ci_ungrounded_numeric",
        ("ungrounded numeric", "ungrounded generated numeric", "ungrounded scalar"),
    ),
    ("ci_embedded_literal", ("embedded scalar literal",)),
    ("ci_proof_claim_unsupported", ("unsupported proof", "proof claim")),
    (
        "ci_proof_anchor",
        ("proof anchor", "proof excerpt", "proof atom", "proof source"),
    ),
    ("ci_source_scope", ("source scope", "source verification", "source claim")),
    (
        "ci_deferral_location",
        ("deferral location", "deferred location", "`deferred_outputs` is misplaced"),
    ),
    (
        "ci_deferral_reason",
        ("deferral reason", "deferred reason", ".reason is required"),
    ),
    (
        "ci_empty_module",
        (
            "empty module",
            "no rules",
        ),
    ),
    (
        "ci_source_subparagraph_coverage",
        (
            "source subparagraph",
            "subparagraph coverage",
            "source sub-paragraph coverage",
        ),
    ),
    ("ci_legacy_scalar_contract", ("legacy scalar", "parameter formula")),
    ("ci_formula_date", ("formula date", "date literal")),
    ("ci_temporal_formula_missing", ("temporal formula", "no formula version")),
    (
        "ci_input_reference_invalid",
        ("input reference", "unknown input", "dataset input"),
    ),
    ("ci_missing_input", ("missing input", "input assignment")),
    ("ci_nonnegative_income_floor", ("nonnegative", "non-negative")),
    (
        "ci_judgment_output_coverage",
        ("judgment positive", "judgment rule missing positive"),
    ),
    ("ci_output_coverage", ("output coverage", "companion output", "zero branch test")),
    (
        "ci_tests_missing",
        (
            "no tests found",
            "companion test file",
        ),
    ),
    (
        "ci_tests_yaml",
        (
            "tests must be",
            "test yaml",
        ),
    ),
    ("ci_test_division_zero", ("division by zero",)),
    ("ci_test_type_mismatch", ("test type", "type mismatch")),
    ("ci_test_assertion", ("test failed", "expected", "assertion")),
    (
        "compile_relation_typing",
        (
            "relation typing",
            "relation type",
            "relation entity typing",
        ),
    ),
    (
        "compile_atomic_kind",
        (
            "atomic kind",
            "invalid rule kind",
            "atomic rulespec module",
        ),
    ),
    ("compile_formula_version_missing", ("formula version",)),
    ("compile_validation_sum_where_argument", ("sum_where",)),
    (
        "compile_yaml_type_error",
        (
            "yaml parse",
            "invalid yaml",
            "yaml type",
        ),
    ),
)


def issue_category(issue: str) -> str:
    """Assign exactly one stable family, retaining production's gate labels."""

    prefix = re.match(
        r"^(?:[^\s:]+\.yaml:\s+)?(?:(?:ci|compile):\s+)?"
        r"\[([A-Za-z][A-Za-z0-9:_.-]+)\]",
        issue,
    )
    if prefix:
        name = prefix.group(1)
        if name.startswith("complete-source-unit:"):
            suffix = name.split(":", 1)[1]
            if suffix in COMPLETE_SOURCE_UNIT_CATEGORIES:
                return name
        elif name in _CONTEXT_CATEGORIES:
            return name
        return "unknown-prefix"
    lowered = issue.lower()
    if ".test.yaml yaml parse failed" in lowered:
        return "ci_tests_yaml"
    compile_stage = ": compile:" in lowered
    families = sorted(
        _ISSUE_FAMILIES,
        key=lambda item: item[0].startswith("compile_") != compile_stage,
    )
    for category, markers in families:
        if any(marker in lowered for marker in markers):
            return category
    if ": compile:" in lowered or "compile failed" in lowered:
        return "compile-other"
    if ": ci:" in lowered:
        return "ci-other"
    return "overlay-other"


@dataclass(frozen=True)
class AdmissionResult:
    """Production verdict and independently reported admission measurements."""

    admitted: bool
    compile_pass: bool | None
    ci_pass: bool | None
    issues: list[str]
    issue_categories: list[str]
    refusal_categories: list[str]
    prerequisite_categories: list[str]
    prerequisite_failure: bool
    failure_kind: str | None
    total_authoritative_occurrences: int | None
    covered_authoritative_occurrences: int | None
    missing_authoritative_occurrences: int | None
    authoritative_recall_percentage: float | None
    identity: dict[str, object]
    unknown_issues: list[str] = field(default_factory=list)
    scorer_error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD^{commit}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tracked_digest(root: Path) -> str | None:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if listing.returncode:
        return None
    digest = hashlib.sha256()
    for raw_name in sorted(filter(None, listing.stdout.split(b"\0"))):
        path = root / raw_name.decode()
        digest.update(raw_name + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0" + str(path.readlink()).encode())
        elif path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        else:
            digest.update(b"missing\0")
    return digest.hexdigest()


def _tool_identity(root: Path | None, binary_name: str) -> dict[str, object] | None:
    if root is None:
        return None
    binary = (
        root
        if root.is_file()
        else next(
            (
                path
                for path in (
                    root / "target" / "release" / binary_name,
                    root / "target" / "debug" / binary_name,
                    root / binary_name,
                )
                if path.is_file()
            ),
            None,
        )
    )
    checkout = root.parent if root.is_file() else root
    return {
        "commit": _git_commit(checkout),
        "tracked_sha256": _tracked_digest(checkout),
        "binary_sha256": hashlib.sha256(binary.resolve().read_bytes()).hexdigest()
        if binary is not None
        else None,
    }


def _policy_identity(root: Path) -> dict[str, object]:
    from .. import cli

    checkout = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    checkout_root = Path(checkout.stdout.strip()) if checkout.returncode == 0 else root
    content_digest = hashlib.sha256(b"axiom-admission-policy-v1\0")
    for directory, directory_names, file_names in os.walk(
        checkout_root, followlinks=False
    ):
        ignored = cli._apply_overlay_copy_ignore(
            directory, directory_names + file_names
        )
        directory_names[:] = sorted(
            name for name in directory_names if name not in ignored
        )
        for name in sorted(set(file_names) - ignored):
            path = Path(directory) / name
            relative = path.relative_to(checkout_root).as_posix()
            content_digest.update(relative.encode() + b"\0")
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                content_digest.update(b"symlink\0" + str(path.readlink()).encode())
            elif stat.S_ISREG(mode):
                content_digest.update(hashlib.sha256(path.read_bytes()).digest())
            else:
                raise ValueError(
                    f"Policy context artifact is not a regular file: {path}"
                )
    return {
        "commit": _git_commit(checkout_root),
        "tracked_sha256": _tracked_digest(checkout_root),
        "content_sha256": content_digest.hexdigest(),
    }


def admission_identity(
    *,
    policy_repo_path: Path,
    axiom_rules_path: Path,
    axiom_compose_path: Path | None = None,
    local_corpus_release: LocalCorpusRelease | None = None,
    rulespec_dependency_roots: Sequence[Path] = (),
    validate_dependents: bool = True,
) -> dict[str, object]:
    """Bind the score to source, runtime bytes and the production options."""

    runtime_package = Path(__file__).resolve().parents[1]
    encoder_root = runtime_package.parent.parent
    source_checkout = (
        runtime_package.parent.name == "src"
        and (encoder_root / "pyproject.toml").is_file()
        and (encoder_root / "uv.lock").is_file()
    )
    from .evals import _deterministic_tree_identity

    package = _deterministic_tree_identity(
        runtime_package,
        excluded_directory_names=frozenset({"__pycache__"}),
    )
    return {
        "encoder": {
            "commit": _git_commit(encoder_root) if source_checkout else None,
            "version": __version__,
            "package_sha256": package["tree_sha256"],
            "pyproject_sha256": hashlib.sha256(
                (encoder_root / "pyproject.toml").read_bytes()
            ).hexdigest()
            if source_checkout
            else None,
            "lock_sha256": hashlib.sha256(
                (encoder_root / "uv.lock").read_bytes()
            ).hexdigest()
            if source_checkout
            else None,
        },
        "engine": _tool_identity(Path(axiom_rules_path), "axiom-rules-engine"),
        "compose": _tool_identity(
            Path(axiom_compose_path) if axiom_compose_path is not None else None,
            "axiom-compose",
        ),
        "corpus": {
            "release_name": local_corpus_release.name,
            "content_sha256": local_corpus_release.content_sha256,
            "selector_sha256": local_corpus_release.selector_sha256,
        }
        if isinstance(local_corpus_release, LocalCorpusRelease)
        else None,
        "policy_repo": _policy_identity(Path(policy_repo_path)),
        "dependencies": [
            _policy_identity(Path(root)) for root in rulespec_dependency_roots
        ],
        "options": {
            "enable_oracles": False,
            "require_policy_proofs": True,
            "require_complete_source_unit": True,
            "skip_reviewers": True,
            "validate_dependents": validate_dependents,
        },
    }


def prerequisite_admission_result(
    issues: Sequence[str],
    *,
    identity: Mapping[str, object] | None = None,
    categories: Sequence[str] | None = None,
) -> AdmissionResult:
    return AdmissionResult(
        admitted=False,
        compile_pass=None,
        ci_pass=None,
        issues=list(issues),
        issue_categories=[issue_category(issue) for issue in issues],
        refusal_categories=[],
        prerequisite_categories=list(dict.fromkeys(categories or ["infrastructure"])),
        prerequisite_failure=True,
        failure_kind="prerequisite",
        total_authoritative_occurrences=None,
        covered_authoritative_occurrences=None,
        missing_authoritative_occurrences=None,
        authoritative_recall_percentage=None,
        identity=dict(identity or {}),
    )


def _candidate_failure_result(
    issues: Sequence[str],
    *,
    identity: Mapping[str, object],
    categories: Sequence[str],
) -> AdmissionResult:
    return AdmissionResult(
        admitted=False,
        compile_pass=None,
        ci_pass=None,
        issues=list(issues),
        issue_categories=list(categories),
        refusal_categories=list(dict.fromkeys(categories)),
        prerequisite_categories=[],
        prerequisite_failure=False,
        failure_kind="candidate",
        total_authoritative_occurrences=None,
        covered_authoritative_occurrences=None,
        missing_authoritative_occurrences=None,
        authoritative_recall_percentage=None,
        identity=dict(identity),
    )


def _scorer_error_result(
    exc: Exception, *, identity: Mapping[str, object], message: str
) -> AdmissionResult:
    result = _candidate_failure_result(
        [f"{type(exc).__name__}: {message}"], identity=identity, categories=[]
    )
    return AdmissionResult(
        **{
            **result.to_dict(),
            "failure_kind": "scorer-error",
            "scorer_error": {"type": type(exc).__name__, "message": message},
        }
    )


class _FrozenContextError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


def _read_frozen_artifact(
    path: Path, *, label: str, category: str, expected_sha256: object = None
) -> bytes:
    """Check frozen directory entries before reading their regular-file bytes."""

    try:
        entry = path.lstat()
    except FileNotFoundError as exc:
        raise _FrozenContextError(
            f"missing-{category}", f"Frozen {label} is missing: {path}"
        ) from exc
    except OSError as exc:
        raise _FrozenContextError(
            f"unreadable-{category}", f"Cannot inspect frozen {label}: {path}"
        ) from exc
    if not stat.S_ISREG(entry.st_mode) or not os.access(path, os.R_OK):
        raise _FrozenContextError(
            f"unreadable-{category}",
            f"Frozen {label} must be a readable regular file: {path}",
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _FrozenContextError(
            f"unreadable-{category}", f"Cannot read frozen {label}: {path}"
        ) from exc
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise _FrozenContextError(
            f"{category}-hash-mismatch", f"Frozen {label} sha256 mismatch: {path}"
        )
    return raw


def _generation_failed_without_workspace(result) -> bool:
    """Whether the row itself records a generation that left no workspace.

    The eval harness writes this row when its runner raises: a failed result
    with no output, trace or context manifest. Nothing else may omit the
    manifest.
    """

    return (
        getattr(result, "success", None) is False
        and getattr(result, "failure_kind", None) in ("timeout", "error")
        and not getattr(result, "output_file", "")
        and not getattr(result, "trace_file", "")
        and not getattr(result, "context_manifest_file", "")
    )


def _no_artifact_category(result) -> str:
    """Name why a row has no candidate, from the row's own failure record."""

    if (
        getattr(result, "timed_out", False) is True
        or getattr(result, "failure_kind", None) == "timeout"
    ):
        return "generation-timeout"
    if _generation_failed_without_workspace(result):
        return "generation-error"
    return "no-artifact"


def _canonical_context_manifest_sha256(context: Mapping[str, object]) -> str:
    """Digest a manifest without the checkout locations its items came from.

    An item's workspace path and packaged bytes identify it. An absolute origin
    path only records where one checkout kept those bytes, so the same context
    packaged from two checkouts gets one digest.
    """

    canonical = copy.deepcopy(dict(context))
    for key in ("context_files", "review_findings_files"):
        for item in canonical.get(key, []):
            origin = item.get("source_path")
            if isinstance(origin, str) and (
                PurePosixPath(origin).is_absolute()
                or PureWindowsPath(origin).is_absolute()
            ):
                del item["source_path"]
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _preflight_frozen_context(
    result,
    source: ResolvedCorpusSource,
    identity: dict[str, object],
    expected_context: Mapping[str, object] | None,
    *,
    allow_missing_manifest: bool = False,
) -> None:
    """Check harness-owned inputs without consulting candidate provenance."""

    from .. import cli

    normalized_body = source.body.replace("\r\n", "\n").replace("\r", "\n")
    generation_digest = hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()
    resolved_attestation = source.to_attestation()
    identity["source"] = {
        **resolved_attestation,
        "body_sha256": hashlib.sha256(source.body.encode("utf-8")).hexdigest(),
        "generation_input_sha256": generation_digest,
    }
    resolved_context = {
        "source_body_sha256": identity["source"]["body_sha256"],
        "policy_repo_commit": identity["policy_repo"]["commit"],
        "dependency_content_sha256": [
            item["content_sha256"] for item in identity["dependencies"]
        ],
        "corpus_release_content_sha256": identity["corpus"]["content_sha256"],
    }
    for key, expected in (expected_context or {}).items():
        if key not in resolved_context or expected != resolved_context[key]:
            raise _FrozenContextError(
                "expected-context-mismatch",
                f"Frozen suite context does not match {key}",
            )

    manifest_name = str(getattr(result, "context_manifest_file", "") or "")
    if not manifest_name:
        if allow_missing_manifest:
            # The row records a generation failure that left no workspace.
            return
        raise _FrozenContextError(
            "missing-context-manifest", "Frozen eval context manifest is not named"
        )
    manifest = Path(manifest_name)
    raw = _read_frozen_artifact(
        manifest,
        label="eval context manifest",
        category="context-manifest",
        expected_sha256=getattr(result, "context_manifest_sha256", None),
    )
    context_identity = {"context_manifest_sha256": hashlib.sha256(raw).hexdigest()}
    identity["context"] = context_identity
    try:
        context = json.loads(raw)
        # Reuse production's metadata shape and row-attestation equality checks.
        metadata = cli._generated_result_source_metadata(result)
    except (UnicodeError, ValueError, RuntimeError, RecursionError) as exc:
        raise _FrozenContextError("invalid-context-manifest", str(exc)) from exc
    for key in ("source_metadata_file", "provision_metadata_file"):
        name = context.get(key)
        if name is not None and (not isinstance(name, str) or not name):
            raise _FrozenContextError(
                "invalid-context-manifest", f"Frozen manifest {key} must name a file"
            )
    for key in ("context_files", "review_findings_files"):
        items = context.get(key, [])
        if not isinstance(items, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("workspace_path"), str)
            or not item["workspace_path"]
            for item in items
        ):
            raise _FrozenContextError(
                "invalid-context-manifest",
                f"Frozen manifest {key} must name workspace files",
            )
    # The raw digest is evidence of one packaging. Runs compare this one.
    context_identity["context_manifest_canonical_sha256"] = (
        _canonical_context_manifest_sha256(context)
    )
    source_name = context.get("source_text_file")
    if source_name is None or source_name == "":
        raise _FrozenContextError(
            "missing-generation-input", "Frozen manifest does not name source_text_file"
        )
    if not isinstance(source_name, str):
        raise _FrozenContextError(
            "invalid-context-manifest",
            "Frozen manifest source_text_file must name a file",
        )
    generation_path = manifest.parent / source_name
    generation_bytes = _read_frozen_artifact(
        generation_path,
        label="generation input record",
        category="generation-input",
        expected_sha256=context.get("source_text_sha256"),
    )
    actual_generation_digest = hashlib.sha256(generation_bytes).hexdigest()
    context_identity["generation_input_sha256"] = actual_generation_digest
    frozen_attestation = metadata.get("source_attestation") if metadata else None
    row_attestation = getattr(result, "source_attestation", None)
    for attestation in (frozen_attestation, row_attestation):
        if attestation is None:
            continue
        if not isinstance(attestation, dict):
            raise _FrozenContextError(
                "frozen-source-mismatch", "Frozen source_attestation must be an object"
            )
        for key, resolved in resolved_attestation.items():
            if key in attestation and attestation[key] != resolved:
                raise _FrozenContextError(
                    "frozen-source-mismatch",
                    f"Frozen source_attestation.{key} differs from the resolved source",
                )
        if (
            "generation_input_sha256" in attestation
            and attestation["generation_input_sha256"] != actual_generation_digest
        ):
            raise _FrozenContextError(
                "generation-input-hash-mismatch",
                "Frozen source_attestation.generation_input_sha256 differs from "
                "the generation input record",
            )
    if generation_bytes != normalized_body.encode("utf-8"):
        raise _FrozenContextError(
            "frozen-source-mismatch",
            "Frozen generation input record differs from the resolved source body",
        )
    row_generation_digest = getattr(result, "generation_input_sha256", None)
    if (
        row_generation_digest is not None
        and row_generation_digest != actual_generation_digest
    ):
        raise _FrozenContextError(
            "generation-input-hash-mismatch",
            "Frozen row generation input sha256 mismatch",
        )
    row_generation_file = getattr(result, "generation_input_file", None)
    if row_generation_file:
        _read_frozen_artifact(
            Path(row_generation_file),
            label="row generation input record",
            category="generation-input",
            expected_sha256=actual_generation_digest,
        )
    backend = str(getattr(result, "backend", "") or "").strip().lower()
    if backend in cli.APPLIED_ENCODING_ENCODER_BACKENDS:
        try:
            attestation = cli._generated_result_source_attestation(result)
            issues = cli._source_attestation_structure_issues(attestation)
            if issues:
                raise RuntimeError("; ".join(issues))
        except RuntimeError as exc:
            raise _FrozenContextError("frozen-source-mismatch", str(exc)) from exc

    # The metadata loader uses inline metadata. Check the additional packaged
    # context records named by the manifest as frozen inputs as well.
    artifacts = {}
    for key in ("source_metadata_file", "provision_metadata_file"):
        name = context.get(key)
        if name:
            artifact = _read_frozen_artifact(
                manifest.parent / name,
                label=key,
                category="context-artifact",
                expected_sha256=context.get(key.removesuffix("_file") + "_sha256"),
            )
            artifacts[key] = hashlib.sha256(artifact).hexdigest()
    for key in ("context_files", "review_findings_files"):
        for item in context.get(key, []):
            artifact = _read_frozen_artifact(
                manifest.parent / item["workspace_path"],
                label=f"{key} {item['workspace_path']}",
                category="context-artifact",
                expected_sha256=item.get("sha256"),
            )
            artifacts[item["workspace_path"]] = hashlib.sha256(artifact).hexdigest()
    context_identity["artifacts"] = artifacts


def _is_composition(path: Path) -> bool:
    from .. import cli

    if path.is_symlink() or not path.is_file():
        return False
    try:
        payload = cli._safe_load_unique_keys(path.read_text())
    except (OSError, UnicodeError, ValueError, RecursionError, cli.yaml.YAMLError):
        # Malformed candidate content belongs to production's refusal path.
        return False
    module = payload.get("module") if isinstance(payload, dict) else None
    return isinstance(module, dict) and module.get("kind") == "composition"


def score_admission(
    result,
    *,
    policy_repo_path: Path,
    axiom_rules_path: Path,
    local_corpus_release: LocalCorpusRelease | None,
    output_root: Path | None = None,
    axiom_compose_path: Path | None = None,
    citation: str | None = None,
    companion_test_path: Path | None = None,
    rulespec_dependency_roots: Sequence[Path] = (),
    validate_dependents: bool = True,
    expected_context: Mapping[str, object] | None = None,
    generation_infrastructure_failure: str | None = None,
) -> AdmissionResult:
    """Run signed apply's repairs and validation on disposable candidate bytes.

    Reject original artifact kinds before reading or staging their bytes.
    Production stamps source provenance before this guard and can report an
    earlier refusal or block on a FIFO. The scorer reports the artifact-kind
    refusal; both verdicts reject it. Production's operation order is unchanged.

    A non-regular artifact is evidence about the candidate whatever else
    failed, so that refusal is scored even when a prerequisite is also
    missing. An absent artifact is not: it is what an infrastructure failure
    looks like. A row with no artifact is scored only after every prerequisite
    holds, and is then named from the row's own failure record:
    ``generation-timeout``, ``generation-error`` (the runner raised and
    recorded no workspace) or ``no-artifact``. Only a row that records such a
    runner failure may omit its context manifest.

    ``generation_infrastructure_failure`` is the caller's own evidence that
    infrastructure ended a generation: one of
    ``GENERATION_INFRASTRUCTURE_FAILURES``. A row with no artifact is then a
    prerequisite failure (``generation-authentication`` or
    ``generation-quota-exhaustion``), not the model's. The scorer never derives
    it from diagnostic text, and it does not excuse an artifact that exists.

    ``expected_context`` optionally pins ``source_body_sha256``,
    ``policy_repo_commit``, the ordered ``dependency_content_sha256`` list and
    ``corpus_release_content_sha256`` from a suite manifest. Resolved identities
    are recorded even when these expected values are not supplied.
    """

    from .. import cli
    from ..corpus_resolver import resolve_local_corpus_source
    from ..engine_binding import (
        _engine_binary_candidates,
        load_declared_engine_pin,
        resolve_pinned_engine_binary,
    )
    from .evals import _resolve_eval_output_path
    from .validator_pipeline import _rulespec_dependencies_for_active_root

    if (
        generation_infrastructure_failure is not None
        and generation_infrastructure_failure not in GENERATION_INFRASTRUCTURE_FAILURES
    ):
        raise ValueError(
            "Unknown generation infrastructure failure: "
            f"{generation_infrastructure_failure!r}"
        )
    candidate_name = str(getattr(result, "output_file", "") or "")
    candidate = Path(candidate_name)
    companion = (
        Path(companion_test_path)
        if companion_test_path is not None
        else cli._rulespec_test_path(candidate)
        if candidate.name
        else Path("missing.test.yaml")
    )
    runner = str(getattr(result, "runner", "") or "admission-score")
    unsafe_runner = Path(runner).is_absolute() or ".." in Path(runner).parts
    generated_root = (
        Path(output_root) / runner
        if output_root is not None and not unsafe_runner
        else candidate.parent
    )
    canonical_citation = str(citation or getattr(result, "citation", ""))
    try:
        relative = _resolve_eval_output_path(canonical_citation)
    except ValueError:
        relative = Path(candidate.name)
    if output_root is not None:
        try:
            relative = candidate.resolve().relative_to(generated_root.resolve())
        except (OSError, ValueError, RuntimeError):
            try:
                relative = candidate.absolute().relative_to(generated_root.absolute())
            except ValueError:
                pass
    # The shared guard examines directory entries, never candidate contents.
    artifact_issue = (
        cli._generated_artifact_guard_issue(
            candidate, companion, relative, generated_root
        )
        if candidate_name
        else None
    )

    try:
        identity = admission_identity(
            policy_repo_path=policy_repo_path,
            axiom_rules_path=axiom_rules_path,
            axiom_compose_path=axiom_compose_path,
            local_corpus_release=local_corpus_release,
            rulespec_dependency_roots=rulespec_dependency_roots,
            validate_dependents=validate_dependents,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if artifact_issue is not None:
            return _candidate_failure_result(
                [artifact_issue],
                identity={},
                categories=[issue_category(artifact_issue)],
            )
        return prerequisite_admission_result(
            [str(exc)], categories=["identity-unavailable"]
        )
    no_artifact = not candidate_name or not os.path.lexists(candidate)

    def artifact_failure() -> AdmissionResult:
        if artifact_issue is not None:
            issue, category = artifact_issue, issue_category(artifact_issue)
        else:
            category = _no_artifact_category(result)
            issue = {
                "generation-timeout": "No candidate artifact: generation timed out",
                "generation-error": (
                    "No candidate artifact: generation failed before a "
                    "workspace was recorded"
                ),
            }.get(category, f"No candidate artifact: {candidate_name}")
        return _candidate_failure_result(
            [issue], identity=identity, categories=[category]
        )

    if not isinstance(local_corpus_release, LocalCorpusRelease):
        if artifact_issue is not None:
            return artifact_failure()
        return prerequisite_admission_result(
            ["A verified LocalCorpusRelease is required"],
            identity=identity,
            categories=["unverifiable-corpus-release"],
        )
    prerequisite = "missing-policy-context"
    try:
        if not Path(policy_repo_path).is_dir():
            raise ValueError(
                f"Policy repository context is missing: {policy_repo_path}"
            )
        prerequisite = "unresolvable-ref"
        if identity["encoder"]["commit"] is None:
            raise RuntimeError(
                "Cannot resolve ref HEAD for the running encoder source checkout"
            )
        if identity["policy_repo"]["commit"] is None:
            raise RuntimeError("Cannot resolve ref HEAD for the policy repository")
        if any(item["commit"] is None for item in identity["dependencies"]):
            raise RuntimeError("Cannot resolve ref HEAD for a RuleSpec dependency")
        prerequisite = "missing-corpus-text"
        source = resolve_local_corpus_source(canonical_citation, local_corpus_release)
        if not source.body or not source.body.strip():
            raise ValueError(
                f"Authoritative source body is empty: {canonical_citation}"
            )
        _preflight_frozen_context(
            result,
            source,
            identity,
            expected_context,
            allow_missing_manifest=_generation_failed_without_workspace(result),
        )
        prerequisite = "missing-policy-context"
        if unsafe_runner:
            raise ValueError("Eval runner path escapes output root")
        # Frozen repository context belongs to the citation, not a path the
        # candidate can redirect. Production validates the candidate's path.
        context_relative = _resolve_eval_output_path(canonical_citation)
        content_root = cli._rulespec_apply_content_root(
            policy_repo_path, context_relative
        )
        for root in rulespec_dependency_roots:
            if not Path(root).is_dir():
                raise ValueError(f"RuleSpec dependency context is missing: {root}")
        prerequisite = "invalid-dependency-context"
        dependencies = cli._normalize_rulespec_dependency_roots(
            rulespec_dependency_roots
        )
        _rulespec_dependencies_for_active_root(content_root, dependencies)
        prerequisite = "unresolvable-ref"
        pin = load_declared_engine_pin(content_root)
        if pin is not None:
            identity["engine"]["declared_ref"] = pin.sha
        if artifact_issue is not None:
            # Runtime receipt verification reads binary bytes. A refused
            # artifact can occupy that path, so only enrich frozen metadata.
            return artifact_failure()
        if pin is not None:
            binary = resolve_pinned_engine_binary(
                axiom_rules_path, pin, allow_build=False
            )
            identity["engine"]["binary_sha256"] = hashlib.sha256(
                binary.read_bytes()
            ).hexdigest()
        else:
            binary = next(
                (
                    path
                    for path in _engine_binary_candidates(axiom_rules_path)
                    if path.exists()
                ),
                None,
            )
            if binary is None:
                raise RuntimeError(
                    "Axiom rules engine binary not found in the supplied checkout"
                )
        prerequisite = "runtime-unavailable"
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError(f"Axiom rules engine binary is not executable: {binary}")
        if _is_composition(content_root / context_relative):
            if axiom_compose_path is None:
                raise ValueError(
                    "composition RuleSpec validation requires an explicit "
                    "axiom-compose executable"
                )
        if axiom_compose_path is not None:
            cli._resolve_optional_axiom_compose_path(axiom_compose_path)
    except _FrozenContextError as exc:
        if artifact_issue is not None:
            return artifact_failure()
        return prerequisite_admission_result(
            [str(exc)], identity=identity, categories=[exc.category]
        )
    except (OSError, ValueError, RuntimeError) as exc:
        if artifact_issue is not None:
            return artifact_failure()
        return prerequisite_admission_result(
            [str(exc)], identity=identity, categories=[prerequisite]
        )
    if no_artifact:
        if generation_infrastructure_failure is not None:
            return prerequisite_admission_result(
                [
                    "Generation ended on an infrastructure failure: "
                    f"{generation_infrastructure_failure}"
                ],
                identity=identity,
                categories=[f"generation-{generation_infrastructure_failure}"],
            )
        # Every prerequisite held, so the absence is the chain's own outcome.
        return artifact_failure()
    latest: dict[str, object] = {}

    def observe(validations: Sequence[tuple[Path, object]]) -> None:
        if validations:
            latest.clear()
            latest.update(getattr(validations[0][1], "results", {}))

    normalization_roots: list[tuple[str, str]] = []

    def normalize(text: str) -> str:
        for root, replacement in normalization_roots:
            text = text.replace(root, replacement)
        return text

    try:
        with tempfile.TemporaryDirectory(prefix="axiom-admission-") as directory:
            staged_result = copy.deepcopy(result)
            staging_container = Path(directory).resolve()
            staged_root = staging_container / "output"
            staged_result.runner = runner
            staged_candidate = staged_root / runner / relative
            diagnostic_root = (
                str(Path(output_root).absolute())
                if output_root is not None
                else "<admission-staging-root>"
            )
            normalization_roots = [
                (str(staged_candidate), str(candidate)),
                (str(staged_root), diagnostic_root),
                (str(staging_container), "<admission-staging-root>"),
            ]
            staged_candidate.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, staged_candidate)
            staged_result.output_file = str(staged_candidate)
            if companion.exists():
                shutil.copyfile(companion, cli._rulespec_test_path(staged_candidate))
            ok, issues, _supplemental = (
                cli._validate_generated_encoding_candidate_in_policy_overlay_with_release(
                    staged_result,
                    output_root=staged_root,
                    policy_repo_path=policy_repo_path,
                    axiom_rules_path=axiom_rules_path,
                    axiom_compose_path=axiom_compose_path,
                    local_corpus_release=local_corpus_release,
                    rulespec_dependency_roots=rulespec_dependency_roots,
                    validate_dependents=validate_dependents,
                    require_complete_source_unit=True,
                    validation_observer=observe,
                    normalize_staging_paths=True,
                )
            )
            # Early production refusals can name the disposable candidate root.
            issues = [normalize(issue) for issue in issues]
    except Exception as exc:
        return _scorer_error_result(exc, identity=identity, message=normalize(str(exc)))
    if not ok and not issues:
        issues = [f"{relative}: candidate validation refused without a diagnostic"]
    categories = [issue_category(issue) for issue in issues]
    ci = latest.get("ci")
    compile_result = latest.get("compile")
    recall = getattr(ci, "details", {}).get("complete_source_unit_recall", {})
    total, covered, missing = (
        recall.get(key) for key in ("total", "covered", "missing")
    )
    return AdmissionResult(
        admitted=ok,
        compile_pass=getattr(compile_result, "passed", None),
        ci_pass=getattr(ci, "passed", None),
        issues=issues,
        issue_categories=categories,
        refusal_categories=[] if ok else list(dict.fromkeys(categories)),
        prerequisite_categories=[],
        prerequisite_failure=False,
        failure_kind=None if ok else "candidate",
        total_authoritative_occurrences=total,
        covered_authoritative_occurrences=covered,
        missing_authoritative_occurrences=missing,
        authoritative_recall_percentage=100 * covered / total if total else None,
        identity=identity,
        unknown_issues=[
            issue
            for issue, category in zip(issues, categories, strict=True)
            if category
            in {"unknown-prefix", "ci-other", "compile-other", "overlay-other"}
        ],
    )
