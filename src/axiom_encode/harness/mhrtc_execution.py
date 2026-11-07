"""Fresh pinned-engine comparisons for MHRtC; not source or capture authority.

The private owner must independently bind the source, reconstruct the request
from the frozen companion, check expected outputs and register complete coverage.
These helpers neither mint that owner nor trust a reported execution as evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine_binding import EnginePin, resolve_pinned_engine_binary
from ..rules_engine_compat import run_rulespec_compile

ENGINE_PIN = "89571cc2a938707fd60a5489b951135345697296"


def strict_json(raw: str | bytes) -> Any:
    """Reject duplicate keys and floating JSON tokens, including overflow.

    This engine's exact decimal values are wire strings. Accepting JSON floats
    would lose precision before comparing two otherwise distinct executions.
    """

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"unsupported floating JSON number: {value}")

    value = json.loads(
        raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=invalid
    )
    _typed(value)
    return value


def _typed(value: Any) -> Any:
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("JSON string keys required")
        return (dict, tuple(sorted((key, _typed(v)) for key, v in value.items())))
    if type(value) is list:
        return (list, tuple(_typed(v) for v in value))
    if value is None or type(value) in (str, bool, int):
        return (type(value), value)
    raise ValueError("exact primitive JSON required; decimals must be wire strings")


def same_typed_json(left: Any, right: Any) -> bool:
    """Object order is immaterial; array order and primitive types are not."""
    return _typed(left) == _typed(right)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class MhrtcAttempt:
    """Diagnostic failure evidence only; never source authority."""

    operation: str
    returncode: int
    stdout: str
    stderr: str
    command: tuple[str, ...]
    request_json: str | None
    binary_sha256: str
    artifact_sha256: str
    rebuilt_ir: bytes | None = None


class MhrtcExecutionError(ValueError):
    def __init__(self, message: str, attempt: MhrtcAttempt):
        super().__init__(message)
        self.attempt = attempt


def _verified_binary(capture, pipe) -> Path:
    from .same_case_execution import _materialized_capture

    if not _materialized_capture(capture) or not capture.fresh():
        raise ValueError("fresh ordinary materialized capture required")
    declared = pipe._declared_engine_pin()
    if (
        declared is None
        or declared.sha != ENGINE_PIN
        or capture.engine_pin != ENGINE_PIN
        or capture.artifact is None
    ):
        raise ValueError("exact MHRtC engine pin and compiled artifact required")
    binary = resolve_pinned_engine_binary(
        pipe.axiom_rules_path,
        EnginePin(ENGINE_PIN, declared.source),
        allow_build=False,
    )
    if binary.resolve() != capture.binary.resolve() or (
        _digest(binary) != capture.binary_sha256
    ):
        raise ValueError("capture binary does not match verified engine")
    return binary


def verify_mhrtc_compiled(capture, pipe) -> bytes:
    """Recompile to a separate file and compare the entire typed IR."""
    binary = _verified_binary(capture, pipe)
    original = capture.artifact.read_bytes()
    original_payload = strict_json(original)
    root = capture.snapshot_root.resolve()
    program = (root / capture.root_relative).resolve()
    if not program.is_relative_to(root) or not program.is_file():
        raise ValueError("captured program must be within snapshot root")
    with tempfile.TemporaryDirectory(prefix="axiom-mhrtc-compile-") as directory:
        output = Path(directory) / "compiled.json"
        result = run_rulespec_compile(
            binary=binary,
            program=program,
            rulespec_roots=(root,),
            output=output,
            cwd=pipe.axiom_rules_path,
            env=pipe._rulespec_engine_env(),
            timeout=60,
        )
        rebuilt = output.read_bytes() if output.is_file() else None
        attempt = MhrtcAttempt(
            "compile",
            result.returncode,
            result.stdout,
            result.stderr,
            tuple(map(str, result.args)),
            None,
            capture.binary_sha256,
            hashlib.sha256(original).hexdigest(),
            rebuilt,
        )
        try:
            if result.returncode != 0 or rebuilt is None:
                raise ValueError("MHRtC independent compilation failed")
            if not same_typed_json(original_payload, strict_json(rebuilt)):
                raise ValueError(
                    "MHRtC compiled artifact differs from fresh compilation"
                )
            _verified_binary(capture, pipe)
            if capture.artifact.read_bytes() != original:
                raise ValueError("compiled artifact changed during verification")
        except (OSError, ValueError) as exc:
            raise MhrtcExecutionError(str(exc), attempt) from exc
    return rebuilt


@dataclass(frozen=True)
class MhrtcFreshExecution:
    """Public diagnostic record; constructing one confers no authority."""

    request_json: str
    stdout: str
    stderr: str
    binary_sha256: str
    artifact_sha256: str


def _run_mhrtc_request(capture, pipe, reconstructed):
    """Run once under the existing capture, receipt and freshness checks."""
    if type(reconstructed) is not dict or reconstructed.get("mode") != "explain":
        raise ValueError("Explain request required")
    binary = _verified_binary(capture, pipe)
    artifact = capture.artifact.read_bytes()
    request = json.dumps(reconstructed, allow_nan=False, separators=(",", ":"))
    result = subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(capture.artifact)],
        input=request,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=pipe.axiom_rules_path,
        env=pipe._rulespec_engine_env(),
    )
    attempt = MhrtcAttempt(
        "run-compiled",
        result.returncode,
        result.stdout,
        result.stderr,
        tuple(map(str, result.args)),
        request,
        capture.binary_sha256,
        hashlib.sha256(artifact).hexdigest(),
    )
    try:
        if result.returncode != 0:
            raise ValueError("MHRtC fresh execution failed")
        actual = strict_json(result.stdout)
        _verified_binary(capture, pipe)
        if capture.artifact.read_bytes() != artifact:
            raise ValueError("compiled artifact changed during execution")
    except (OSError, ValueError) as exc:
        raise MhrtcExecutionError(str(exc), attempt) from exc
    receipt = MhrtcFreshExecution(
        request,
        result.stdout,
        result.stderr,
        capture.binary_sha256,
        hashlib.sha256(artifact).hexdigest(),
    )
    return receipt, attempt, actual


def reexecute_mhrtc_case(capture, pipe, reconstructed, reported_request, response):
    """Rerun an independently reconstructed request; never replay arithmetic."""
    if not same_typed_json(reconstructed, reported_request):
        raise ValueError("reported request differs from reconstructed companion")
    receipt, attempt, actual = _run_mhrtc_request(capture, pipe, reconstructed)
    try:
        if not same_typed_json(actual, response):
            raise ValueError("reported response differs from fresh engine execution")
    except ValueError as exc:
        raise MhrtcExecutionError(str(exc), attempt) from exc
    return receipt


def execute_mhrtc_auxiliary_request(capture, pipe, reconstructed):
    """Obtain a real auxiliary receipt without inventing a reported response.

    The private source observer must reconstruct the fixed OtherClaim request,
    bind it to the principal case, and validate its typed result. This helper's
    public return value alone cannot certify coverage or mint source evidence.
    """
    receipt, _, _ = _run_mhrtc_request(capture, pipe, reconstructed)
    return receipt
