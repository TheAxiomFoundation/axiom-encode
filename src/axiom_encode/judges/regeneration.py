"""Purpose-built module regeneration for golden drift checks.

Module names and encoding manifests come from a repository checkout, so both
are untrusted. Regeneration exposes no command-template surface: it validates
and binds those inputs, derives a safe source id, and invokes this installed
encoder with a fixed argv and a minimal environment.
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
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*\Z")
_SOURCE_DIRECTORIES = frozenset(
    {"forms", "guidance", "manuals", "policies", "regulations", "statutes"}
)
_MAX_MODULE_BYTES = 10 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_GENERATED_BYTES = 10 * 1024 * 1024
_MAX_CORPUS_PROVISION_BYTES = 64 * 1024 * 1024


class RegenerationInputError(ValueError):
    """A module cannot be safely or faithfully regenerated."""


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
    if parts[0] in _SOURCE_DIRECTORIES:
        content_parts = parts
    elif len(parts) >= 2:
        content_parts = parts[1:]
    else:
        raise RegenerationInputError(f"module has no source-root path: {module}")
    content = PurePosixPath(*content_parts)
    if content.parts[0] not in _SOURCE_DIRECTORIES:
        raise RegenerationInputError(
            f"module is outside a RuleSpec source root: {module}"
        )
    return Path(*content.parts).with_suffix(".yaml")


def source_id_for_module(root: Path, module: PurePosixPath) -> str:
    """Derive a canonical source id whose output is exactly this module path."""

    if module.parts[0] in _SOURCE_DIRECTORIES:
        repo_name = Path(root).resolve(strict=True).name
        if not repo_name.startswith("rulespec-"):
            raise RegenerationInputError(
                f"cannot infer jurisdiction from repository root: {repo_name}"
            )
        jurisdiction = repo_name.removeprefix("rulespec-")
    else:
        jurisdiction = module.parts[0]
    if _JURISDICTION.fullmatch(jurisdiction) is None:
        raise RegenerationInputError(f"invalid module jurisdiction: {jurisdiction!r}")

    expected = generated_subpath(module)
    source_id = f"{jurisdiction}:{expected.with_suffix('').as_posix()}"
    from axiom_encode.harness.evals import _resolve_eval_output_path

    resolved = _resolve_eval_output_path(source_id)
    if resolved != expected:
        raise RegenerationInputError(
            f"source id {source_id!r} resolves to {resolved}, expected {expected}"
        )
    return source_id


def _manifest_candidates(root: Path, module: PurePosixPath) -> list[Path]:
    """Return supported root-mirror, content-root, and legacy manifest paths."""

    resolved_root = Path(root).resolve(strict=True)
    full_relative = Path(*module.parts).with_suffix(".json")
    content_relative = generated_subpath(module).with_suffix(".json")
    candidates = [
        resolved_root / ".axiom" / "encoding-manifests" / full_relative,
    ]
    if module.parts[0] not in _SOURCE_DIRECTORIES:
        candidates.append(
            resolved_root
            / module.parts[0]
            / ".axiom"
            / "encoding-manifests"
            / content_relative
        )
    candidates.append(
        resolved_root / ".axiom" / "encoding-manifests" / content_relative
    )
    return list(dict.fromkeys(candidates))


def _manifest_binds_module(
    manifest: dict[str, object],
    *,
    module: PurePosixPath,
    module_sha256: str,
) -> bool:
    expected_paths = {module.as_posix(), generated_subpath(module).as_posix()}
    applied_files = manifest.get("applied_files")
    if not isinstance(applied_files, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("path") in expected_paths
        and item.get("sha256") == module_sha256
        for item in applied_files
    )


def _manifest_is_replayable_encode(manifest: dict[str, object]) -> bool:
    return manifest.get("backend") in {"codex", "openai", "claude"} and str(
        manifest.get("tool", "")
    ).startswith("axiom-encode encode")


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
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegenerationInputError(
                f"invalid encoding manifest {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RegenerationInputError(f"encoding manifest is not an object: {path}")
        if not _manifest_binds_module(
            payload,
            module=module,
            module_sha256=module_sha256,
        ):
            continue
        if _manifest_is_replayable_encode(payload):
            return payload

    raise RegenerationInputError(
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


def validate_corpus_path(corpus_path: Path) -> Path:
    """Return a corpus checkout with a locally readable provisions tree."""

    resolved_corpus = Path(corpus_path).resolve(strict=True)
    if not resolved_corpus.is_dir():
        raise RegenerationInputError(
            f"corpus path is not a directory: {resolved_corpus}"
        )

    raw_candidates = (
        resolved_corpus / "data" / "corpus" / "provisions",
        (
            resolved_corpus
            if resolved_corpus.name == "provisions"
            else resolved_corpus / "provisions"
        ),
    )
    provisions_root: Path | None = None
    for candidate in dict.fromkeys(raw_candidates):
        cursor = resolved_corpus
        for part in candidate.relative_to(resolved_corpus).parts:
            cursor /= part
            if cursor.is_symlink():
                raise UnsafeRegenerationPath(
                    f"corpus provisions path contains a symlink: {cursor}"
                )
        if not cursor.exists():
            continue
        if not cursor.is_dir():
            raise UnsafeRegenerationPath(
                f"corpus provisions path is not a directory: {cursor}"
            )
        resolved_provisions = cursor.resolve(strict=True)
        try:
            resolved_provisions.relative_to(resolved_corpus)
        except ValueError as exc:
            raise UnsafeRegenerationPath(
                f"corpus provisions path escapes {resolved_corpus}: {cursor}"
            ) from exc
        provisions_root = resolved_provisions
        break

    if provisions_root is None:
        raise RegenerationInputError(
            f"corpus checkout has no provisions directory: {resolved_corpus}"
        )

    provision_files = 0
    for candidate in provisions_root.rglob("*"):
        if candidate.is_symlink():
            raise UnsafeRegenerationPath(
                f"corpus provisions tree contains a symlink: {candidate}"
            )
        if candidate.suffix != ".jsonl":
            continue
        provision_files += 1
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(resolved_corpus)
        except ValueError as exc:
            raise UnsafeRegenerationPath(
                f"corpus provision escapes {resolved_corpus}: {candidate}"
            ) from exc
        if not resolved.is_file():
            raise UnsafeRegenerationPath(
                f"corpus provision is not a regular file: {candidate}"
            )
        if resolved.stat().st_size > _MAX_CORPUS_PROVISION_BYTES:
            raise UnsafeRegenerationPath(
                "corpus provision exceeds the "
                f"{_MAX_CORPUS_PROVISION_BYTES}-byte safety limit: {candidate}"
            )
    if provision_files == 0:
        raise RegenerationInputError(
            f"corpus checkout has no provision JSONL files: {resolved_corpus}"
        )
    return resolved_corpus


def citation_resolves_locally(citation: str, corpus_path: Path) -> bool:
    """Return whether the encoder can resolve ``citation`` from this checkout.

    The live resolver falls back to Supabase, which would make a scheduled run
    depend on mutable remote state.  Use its exact normalization and
    nearest-parent candidates, but deliberately stop after the local corpus
    lookup.
    """

    resolved_corpus = validate_corpus_path(corpus_path)

    from axiom_encode.harness.evals import (
        _candidate_corpus_citation_paths,
        _fetch_local_corpus_source_text_from_repo,
    )

    return any(
        _fetch_local_corpus_source_text_from_repo(candidate, resolved_corpus)
        is not None
        for candidate in _candidate_corpus_citation_paths(citation)
    )


def read_resolvable_citation(
    root: Path,
    module: PurePosixPath,
    *,
    corpus_path: Path,
) -> str:
    """Read a replay manifest citation that is available in the local corpus."""

    citation = read_citation(root, module)
    if not citation_resolves_locally(citation, corpus_path):
        raise RegenerationInputError(
            f"manifest citation does not resolve in the local corpus: {citation!r}"
        )
    return citation


def child_environment(backend: str) -> dict[str, str]:
    """Build a secret-minimized environment for the encoder child process."""

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported regeneration backend: {backend!r}")
    credential = os.environ.get("OPENAI_API_KEY")
    if not credential:
        raise RuntimeError("OPENAI_API_KEY is required for drift regeneration")

    env = {name: os.environ[name] for name in _SAFE_CHILD_ENV if name in os.environ}
    env.setdefault("PATH", os.defpath)
    env["OPENAI_API_KEY"] = credential
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
    backend: str = "openai",
) -> str:
    """Regenerate one merged module with a fixed argv and isolated secrets."""

    resolved_root = Path(root).resolve(strict=True)
    resolved_corpus = validate_corpus_path(corpus_path)
    relative = validate_module_path(resolved_root, module)
    citation = read_resolvable_citation(
        resolved_root,
        relative,
        corpus_path=resolved_corpus,
    )
    source_id = source_id_for_module(resolved_root, relative)
    expected = generated_subpath(relative)
    env = child_environment(backend)
    policy_root = (
        resolved_root
        if relative.parts[0] in _SOURCE_DIRECTORIES
        else resolved_root / relative.parts[0]
    )

    with tempfile.TemporaryDirectory() as tmp:
        command = [
            sys.executable,
            "-m",
            "axiom_encode.cli",
            "encode",
            "--source-id",
            source_id,
            "--output",
            tmp,
            "--backend",
            backend,
            "--corpus-path",
            str(resolved_corpus),
            "--policy-repo-path",
            str(policy_root),
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
