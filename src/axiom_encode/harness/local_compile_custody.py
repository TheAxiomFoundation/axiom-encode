"""Invocation-local compile provenance for the local-relation evidence route.

This is not source evidence. Only the snapshot materializer registers owners;
only the actual compile subprocess boundary can record a successful generation.
The old imported/floor capture paths do not consume this state.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class _Owner:
    capture: Any
    binding: tuple[Any, ...]
    generation: int = 0
    receipt: _Receipt | None = None
    consumed: bool = False
    case_records: Any = None


@dataclass(frozen=True)
class _Receipt:
    pipeline: Any
    generation: int
    output: Path
    raw: bytes
    command: tuple[str, ...]
    cwd: str | None
    environment_sha256: str
    stdout_sha256: str
    stderr_sha256: str


@dataclass(frozen=True)
class _Attempt:
    owner: _Owner
    generation: int
    pipeline: Any
    program: Path
    output: Path
    roots: tuple[Path, ...]
    cwd: str | None
    environment_sha256: str


_owners: dict[int, _Owner] = {}
_attempt: ContextVar[_Attempt | None] = ContextVar(
    "local_compile_attempt", default=None
)


def _owner(capture: Any) -> _Owner | None:
    state = _owners.get(id(capture))
    return state if state is not None and state.capture is capture else None


def _binding(capture: Any) -> tuple[Any, ...]:
    return (
        capture.original_root,
        capture.snapshot_root,
        capture.root_relative,
        capture.files,
        capture.binary,
        capture.binary_sha256,
        capture.engine_pin,
        capture.git_head,
        capture.overlay_origin,
    )


def _fresh(state: _Owner) -> bool:
    from .same_case_execution import ExecutionCapture

    try:
        return (
            type(state.capture) is ExecutionCapture
            and _binding(state.capture) == state.binding
            and ExecutionCapture.fresh(state.capture)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _materialize_registered(
    pipe: Any, rules_file: Path, plan: tuple, directory: str
) -> Any:
    """Materialize first; no API accepts a preconstructed capture for registration."""
    from .same_case_execution import ExecutionCapture, _materialize_snapshot, _payload

    capture = _materialize_snapshot(pipe, rules_file, plan, directory)
    if type(capture) is not ExecutionCapture:
        return capture
    try:
        payload = _payload((capture.snapshot_root / capture.root_relative).read_bytes())
        state = _Owner(capture, _binding(capture))
        if not payload.get("imports", []) and _fresh(state):
            _owners[id(capture)] = state
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return capture


def _expire(capture: Any) -> None:
    state = _owner(capture)
    if state is not None:
        from .local_case_custody import _forget_seal

        _forget_seal(state)
        state.generation += 1
        state.receipt = None
        state.case_records = None
        _owners.pop(id(capture), None)


@contextmanager
def compile_scope(
    capture: Any, pipeline: Any, program: Path, output: Path
) -> Iterator[None]:
    state = _owner(capture)
    attempt = None
    if state is not None:
        from .local_case_custody import _forget_seal

        _forget_seal(state)
        state.generation += 1
        state.receipt = None
        state.case_records = None
        state.consumed = False
        try:
            roots = tuple(Path(p).resolve() for p in pipeline._rulespec_compile_roots())
            expected = state.binding[1] / state.binding[2]
            env = dict(pipeline._rulespec_engine_env())
            env.pop("AXIOM_RULESPEC_REPO_ROOTS", None)
            env.pop("AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE", None)
            cwd = (
                str(pipeline.axiom_rules_path)
                if pipeline.axiom_rules_path.exists()
                else None
            )
            if (
                program == expected
                and program.resolve() == expected
                and _fresh(state)
                and not output.exists()
                and not output.is_symlink()
                and roots
                and all(
                    p == state.binding[1] or state.binding[1] in p.parents
                    for p in roots
                )
                and all(type(k) is str and type(v) is str for k, v in env.items())
            ):
                attempt = _Attempt(
                    state,
                    state.generation,
                    pipeline,
                    program,
                    output,
                    roots,
                    cwd,
                    _environment_digest(env),
                )
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            # Observation is optional. The ordinary compiler still owns its errors.
            attempt = None
    token = _attempt.set(attempt)
    try:
        yield
    except BaseException:
        if state is not None:
            state.receipt = None
            state.case_records = None
        raise
    finally:
        _attempt.reset(token)


def _environment_digest(env: Any) -> str | None:
    if not isinstance(env, dict) or not all(
        type(k) is str and type(v) is str for k, v in env.items()
    ):
        return None
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate artifact key")
        result[key] = value
    return result


def run_observed_compile(
    command: list[str], *, execution_owner: Any, composed: bool, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Run unchanged subprocess arguments, optionally retaining exact provenance."""
    attempt = _attempt.get()
    eligible = False
    if attempt is not None:
        state = attempt.owner
        state.receipt = None
        state.case_records = None
        capture = state.capture
        expected = [
            str(state.binding[4]),
            "compile",
            "--program",
            str(attempt.program),
            *(
                part
                for root in attempt.roots
                for part in ("--rulespec-root", str(root))
            ),
            "--output",
            str(attempt.output),
        ]
        eligible = (
            _owner(capture) is state
            and state.generation == attempt.generation
            and execution_owner is attempt.pipeline
            and not composed
            and command == expected
            and set(kwargs) == {"capture_output", "text", "timeout", "cwd", "env"}
            and kwargs.get("capture_output") is True
            and kwargs.get("text") is True
            and type(kwargs.get("timeout")) is int
            and kwargs.get("timeout") == 60
            and kwargs.get("cwd") == attempt.cwd
            and _environment_digest(kwargs.get("env")) == attempt.environment_sha256
            and not attempt.output.exists()
            and not attempt.output.is_symlink()
            and _fresh(state)
        )
    result = subprocess.run(command, **kwargs)
    if eligible and attempt is not None:
        state = attempt.owner
        capture = state.capture
        try:
            raw = attempt.output.read_bytes()
            parsed = json.loads(raw, object_pairs_hook=_unique_object)
            if (
                result.returncode == 0
                and _owner(capture) is state
                and state.generation == attempt.generation
                and _fresh(state)
                and attempt.output.is_file()
                and not attempt.output.is_symlink()
                and isinstance(parsed, dict)
                and attempt.output.read_bytes() == raw
            ):
                state.receipt = _Receipt(
                    attempt.pipeline,
                    attempt.generation,
                    attempt.output,
                    raw,
                    tuple(command),
                    attempt.cwd,
                    attempt.environment_sha256,
                    hashlib.sha256(result.stdout.encode()).hexdigest(),
                    hashlib.sha256(result.stderr.encode()).hexdigest(),
                )
        except (OSError, ValueError):
            state.receipt = None
        state.case_records = None
    return result


def consume_compilation(capture: Any, artifact: Path) -> bytes | None:
    """Consume once; bare capture.compiled(path) never creates this receipt."""
    state = _owner(capture)
    if state is None or state.receipt is None or state.consumed:
        return None
    receipt = state.receipt
    generation, output, raw = receipt.generation, receipt.output, receipt.raw
    state.consumed = True
    try:
        if (
            generation != state.generation
            or output != artifact
            or artifact.is_symlink()
            or artifact.read_bytes() != raw
            or not _fresh(state)
            or capture.artifact != artifact
            or capture.artifact_sha256 != hashlib.sha256(raw).hexdigest()
        ):
            return None
    except OSError:
        return None
    return raw
