"""Purpose-built module regeneration for golden drift checks.

Module names and encoding manifests come from a repository checkout, so both
are untrusted. Regeneration exposes no command-template surface: it validates
and binds those inputs, requires the module path derived from the manifest's
corpus citation, and invokes this installed encoder with a fixed argv and a
minimal environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from axiom_encode.constants import RULESPEC_ATOMIC_MODULE_ROOTS
from axiom_encode.corpus_release import RELEASE_OBJECT_PUBLIC_KEY_ENV
from axiom_encode.corpus_resolver import LocalCorpusRelease
from axiom_encode.repo_routing import (
    canonical_rulespec_root_identity,
    is_policy_repo_root,
)
from axiom_encode.toolchain import load_rulespec_local_corpus_release

SUPPORTED_BACKENDS = ("openai",)
REGENERATION_TIMEOUT_SECONDS = 20 * 60

_SAFE_CHILD_ENV = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "TEMP",
    "TMP",
    "TMPDIR",
)
_SAFE_PATH_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")
_SOURCE_DIRECTORIES = RULESPEC_ATOMIC_MODULE_ROOTS
_MAX_MODULE_BYTES = 10 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_GENERATED_BYTES = 10 * 1024 * 1024


class RegenerationInputError(ValueError):
    """A module cannot be safely or faithfully regenerated."""


class NoReplayableManifestError(RegenerationInputError):
    """A module has no current model-encoder manifest to replay."""


class UnsafeRegenerationPath(RegenerationInputError):
    """A repository path attempts traversal, indirection, or command syntax."""


def redact_sensitive_text(
    text: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Replace exact credential values before diagnostics cross a trust boundary."""

    source = os.environ if environ is None else environ
    sensitive_values = {
        value
        for name, value in source.items()
        if value
        and len(value) >= 8
        and any(
            marker in name.upper()
            for marker in ("AUTH", "CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN")
        )
    }
    redacted = str(text)
    for value in sorted(sensitive_values, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
        json_escaped = json.dumps(value)[1:-1]
        if json_escaped != value:
            redacted = redacted.replace(json_escaped, "[REDACTED]")
    return redacted


def _resolved_regular_file(
    root: Path,
    candidate: Path,
    *,
    label: str,
    max_bytes: int,
) -> Path | None:
    """Resolve a non-symlink regular file contained by root, if it exists."""

    raw_root = Path(root).absolute()
    resolved_root = Path(root).resolve(strict=True)
    candidate = Path(candidate).absolute()
    try:
        relative = candidate.relative_to(raw_root)
    except ValueError as exc:
        try:
            relative = candidate.resolve(strict=False).relative_to(resolved_root)
        except ValueError:
            raise UnsafeRegenerationPath(
                f"{label} is outside {resolved_root}: {candidate}"
            ) from exc

    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise UnsafeRegenerationPath(f"{label} contains a symlink: {candidate}")
    if not cursor.exists():
        return None

    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeRegenerationPath(
            f"{label} escapes {resolved_root}: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise UnsafeRegenerationPath(f"{label} is not a regular file: {candidate}")
    if resolved.stat().st_size > max_bytes:
        raise UnsafeRegenerationPath(
            f"{label} exceeds the {max_bytes}-byte safety limit: {candidate}"
        )
    return resolved


def validate_module_path(root: Path, module: str) -> PurePosixPath:
    """Return a normalized, contained RuleSpec YAML module path."""

    if not module or "\\" in module or any(ord(char) < 32 for char in module):
        raise UnsafeRegenerationPath(f"unsafe module path: {module!r}")
    if any(token in module for token in (";", "$", "`")):
        raise UnsafeRegenerationPath(f"module path contains command syntax: {module!r}")

    relative = PurePosixPath(module)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise UnsafeRegenerationPath(
            f"module path must be relative without traversal: {module!r}"
        )
    if relative.suffix != ".yaml":
        raise RegenerationInputError(f"module path must end in .yaml: {module!r}")
    if any(_SAFE_PATH_PART.fullmatch(part) is None for part in relative.parts):
        raise RegenerationInputError(
            f"module path contains unsupported characters: {module!r}"
        )
    if relative.as_posix() != module:
        raise UnsafeRegenerationPath(f"module path is not normalized: {module!r}")

    resolved_root = Path(root).resolve(strict=True)
    if not is_policy_repo_root(resolved_root):
        raise RegenerationInputError(
            "regeneration root must be the exact canonical "
            f"rulespec-<country> checkout: {resolved_root}"
        )
    if (
        len(relative.parts) < 3
        or relative.parts[1] not in _SOURCE_DIRECTORIES
        or relative.name.endswith(".test.yaml")
    ):
        raise RegenerationInputError(
            "module path must be "
            "<jurisdiction>/<atomic-module-root>/.../<module>.yaml: "
            f"{module!r}"
        )
    jurisdiction_root = resolved_root / relative.parts[0]
    if canonical_rulespec_root_identity(jurisdiction_root) is None:
        raise RegenerationInputError(
            f"module has a non-canonical jurisdiction directory: {module!r}"
        )
    candidate = resolved_root.joinpath(*relative.parts)
    resolved = _resolved_regular_file(
        resolved_root,
        candidate,
        label="module",
        max_bytes=_MAX_MODULE_BYTES,
    )
    if resolved is None:
        raise RegenerationInputError(
            f"module file does not exist under {resolved_root}: {module}"
        )
    return relative


def generated_subpath(module: PurePosixPath) -> Path:
    """Return the runner-relative output path for a validated module."""

    parts = module.parts
    if len(parts) < 3 or parts[1] not in _SOURCE_DIRECTORIES:
        raise RegenerationInputError(
            "module path must be "
            f"<jurisdiction>/<atomic-module-root>/.../<module>.yaml: {module}"
        )
    content = PurePosixPath(*parts[1:])
    return Path(*content.parts).with_suffix(".yaml")


def require_citation_derived_output_path(
    module: PurePosixPath,
    citation: str,
) -> Path:
    """Require ``module`` to occupy the output path derived from ``citation``."""

    from axiom_encode.harness.evals import _resolve_eval_output_path
    from axiom_encode.statute import citation_to_relative_rulespec_path

    expected = generated_subpath(module)
    try:
        derived = _resolve_eval_output_path(
            citation,
            fallback=citation_to_relative_rulespec_path,
        )
    except ValueError as exc:
        raise RegenerationInputError(
            f"manifest citation {citation!r} cannot derive a canonical RuleSpec "
            "output path; migrate or re-encode this module"
        ) from exc
    if derived != expected:
        raise RegenerationInputError(
            f"manifest citation {citation!r} canonically encodes to {derived}, "
            f"but the replay manifest binds {expected}; migrate or re-encode "
            "the module at its citation-derived path"
        )
    return derived


def _manifest_candidates(root: Path, module: PurePosixPath) -> list[Path]:
    """Return the single checkout-root mirror manifest path."""

    resolved_root = Path(root).resolve(strict=True)
    full_relative = Path(*module.parts).with_suffix(".json")
    return [
        resolved_root / ".axiom" / "encoding-manifests" / full_relative,
    ]


def _manifest_binds_module(
    manifest: dict[str, object],
    *,
    module: PurePosixPath,
    module_sha256: str,
) -> bool:
    expected_path = module.as_posix()
    applied_files = manifest.get("applied_files")
    if not isinstance(applied_files, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("path") == expected_path
        and item.get("sha256") == module_sha256
        for item in applied_files
    )


def _manifest_is_replayable_encode(manifest: dict[str, object]) -> bool:
    return (
        manifest.get("backend")
        in {
            "codex",
            "openai",
            "claude",
        }
        and manifest.get("tool") == "axiom-encode encode --apply"
    )


def load_replay_manifest(root: Path, module: PurePosixPath) -> dict[str, object]:
    """Load a current, hash-bound manifest produced by the encoder apply path."""

    resolved_root = Path(root).resolve(strict=True)
    module = validate_module_path(resolved_root, module.as_posix())
    module_file = resolved_root.joinpath(*module.parts)
    module_sha256 = hashlib.sha256(module_file.read_bytes()).hexdigest()

    for candidate in _manifest_candidates(resolved_root, module):
        path = _resolved_regular_file(
            resolved_root,
            candidate,
            label="encoding manifest",
            max_bytes=_MAX_MANIFEST_BYTES,
        )
        if path is None:
            continue
        from axiom_encode.cli import _load_verified_applied_encoding_manifest_payload

        payload, _root_prefix, _manifest_sha256, issues = (
            _load_verified_applied_encoding_manifest_payload(
                resolved_root,
                path.relative_to(resolved_root).as_posix(),
            )
        )
        if issues:
            raise RegenerationInputError(
                f"invalid encoding manifest {path}: " + "; ".join(issues)
            )
        if payload is None:
            continue
        if not _manifest_binds_module(
            payload,
            module=module,
            module_sha256=module_sha256,
        ):
            continue
        if _manifest_is_replayable_encode(payload):
            return payload

    raise NoReplayableManifestError(
        f"no current replayable encoding manifest binds module {module}"
    )


def _validate_citation(citation: object) -> str:
    if not isinstance(citation, str) or not citation.strip():
        raise RegenerationInputError("encoding manifest has no citation field")
    if citation != citation.strip() or len(citation) > 2048:
        raise RegenerationInputError(f"invalid manifest citation: {citation!r}")
    if (
        citation.startswith("/")
        or "\\" in citation
        or "`" in citation
        or "$(" in citation
        or any(ord(char) < 32 for char in citation)
        or any(part == ".." for part in citation.split("/"))
    ):
        raise UnsafeRegenerationPath(f"unsafe manifest citation: {citation!r}")
    return citation


def read_citation(root: Path, module: PurePosixPath) -> str:
    """Read and validate the citation from a replayable encoding manifest."""

    return _validate_citation(load_replay_manifest(root, module).get("citation"))


def resolve_local_citation(
    citation: str,
    corpus_release: LocalCorpusRelease,
) -> str | None:
    """Return the single release-bound source selected by the shared resolver."""

    from axiom_encode.corpus_resolver import (
        CorpusSourceNotFoundError,
        normalize_corpus_identifier,
        resolve_local_corpus_source,
    )

    normalized = normalize_corpus_identifier(citation)
    try:
        resolved = resolve_local_corpus_source(normalized, corpus_release)
    except CorpusSourceNotFoundError:
        return None
    return resolved.citation_path


def citation_resolves_locally(
    citation: str,
    corpus_release: LocalCorpusRelease,
) -> bool:
    """Return whether ``citation`` has an exact or parent local candidate."""

    return resolve_local_citation(citation, corpus_release) is not None


def read_resolvable_citation(
    root: Path,
    module: PurePosixPath,
    *,
    corpus_release: LocalCorpusRelease,
) -> str:
    """Read a replay manifest citation that is available in the local corpus."""

    manifest = load_replay_manifest(root, module)
    citation = _validate_citation(manifest.get("citation"))
    from axiom_encode.corpus_resolver import (
        CorpusSourceNotFoundError,
        normalize_corpus_identifier,
        resolve_local_corpus_source,
    )

    try:
        resolved = resolve_local_corpus_source(
            normalize_corpus_identifier(citation),
            corpus_release,
        )
    except CorpusSourceNotFoundError:
        raise RegenerationInputError(
            f"manifest citation does not resolve in the local corpus: {citation!r}"
        ) from None
    attestation = manifest.get("source_attestation")
    if not isinstance(attestation, dict):
        raise RegenerationInputError("replay manifest has no source attestation")
    current_attestation = resolved.to_attestation()
    for field, expected in current_attestation.items():
        if attestation.get(field) != expected:
            raise RegenerationInputError(
                "replay manifest source attestation does not match the active "
                f"corpus release at {field}"
            )
    return citation


def child_environment(backend: str) -> dict[str, str]:
    """Build a secret-minimized environment for the encoder child process."""

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported regeneration backend: {backend!r}")
    credential = os.environ.get("OPENAI_API_KEY")
    if not credential:
        raise RuntimeError("OPENAI_API_KEY is required for drift regeneration")
    release_public_key = os.environ.get(RELEASE_OBJECT_PUBLIC_KEY_ENV)
    if not release_public_key:
        raise RuntimeError(
            f"{RELEASE_OBJECT_PUBLIC_KEY_ENV} is required for drift regeneration"
        )

    env = {name: os.environ[name] for name in _SAFE_CHILD_ENV if name in os.environ}
    env.setdefault("PATH", os.defpath)
    env["OPENAI_API_KEY"] = credential
    env[RELEASE_OBJECT_PUBLIC_KEY_ENV] = release_public_key
    return env


def _read_generated_module(output_root: Path, expected: Path) -> str:
    candidates: list[Path] = []
    for runner_root in output_root.iterdir():
        if runner_root.is_symlink() or not runner_root.is_dir():
            continue
        candidate = runner_root / expected
        resolved = _resolved_regular_file(
            output_root,
            candidate,
            label="generated module",
            max_bytes=_MAX_GENERATED_BYTES,
        )
        if resolved is not None:
            candidates.append(resolved)
    if len(candidates) != 1:
        raise RuntimeError(
            f"encode produced {len(candidates)} files for expected module {expected} "
            f"under {output_root}"
        )
    return candidates[0].read_text(encoding="utf-8")


def regenerate_module(
    module: str,
    _merged: str,
    *,
    root: Path,
    corpus_path: Path,
    axiom_rules_path: Path,
    backend: str = "openai",
) -> str:
    """Regenerate one merged module with a fixed argv and isolated secrets."""

    resolved_root = Path(root).resolve(strict=True)
    relative = validate_module_path(resolved_root, module)
    corpus_release = load_rulespec_local_corpus_release(
        resolved_root,
        corpus_path,
    )
    citation = read_resolvable_citation(
        resolved_root,
        relative,
        corpus_release=corpus_release,
    )
    expected = require_citation_derived_output_path(relative, citation)
    env = child_environment(backend)
    resolved_axiom_rules_path = Path(axiom_rules_path).resolve(strict=True)
    if not resolved_axiom_rules_path.is_dir():
        raise RegenerationInputError(
            f"axiom-rules-engine path is not a directory: {resolved_axiom_rules_path}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            sys.executable,
            "-m",
            "axiom_encode.cli",
            "encode",
            "--output",
            tmp,
            "--backend",
            backend,
            "--corpus-path",
            str(corpus_release.root),
            "--axiom-rules-engine-path",
            str(resolved_axiom_rules_path),
            "--policy-repo-path",
            str(resolved_root),
            "--no-sync",
            "--skip-reviewers",
            "--",
            citation,
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            shell=False,
            timeout=REGENERATION_TIMEOUT_SECONDS,
        )
        if process.returncode != 0:
            diagnostic = redact_sensitive_text(process.stderr or "")[-2000:]
            raise RuntimeError(f"encode failed for {citation!r}: {diagnostic}")
        return _read_generated_module(Path(tmp), expected)
