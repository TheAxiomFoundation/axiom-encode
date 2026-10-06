"""Fail-closed compatibility for explicit RuleSpec engine root contracts."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

_UNKNOWN_EXPLICIT_ROOT_FLAG = "unknown compile argument `--rulespec-root`"
_LEGACY_EXCLUSIVE_ROOT_FLAG = "--exclusive-rulespec-roots"


def run_rulespec_compile(
    *,
    binary: Path,
    program: Path,
    rulespec_roots: Sequence[Path],
    output: Path,
    cwd: Path | None,
    env: Mapping[str, str],
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Compile with the current root contract or its prior exclusive form."""

    roots = tuple(Path(root) for root in rulespec_roots)
    if not roots:
        raise ValueError("RuleSpec engine compilation requires an explicit root")
    base_command = [
        str(binary),
        "compile",
        "--program",
        str(program),
    ]
    current_command = [
        *base_command,
        *(item for root in roots for item in ("--rulespec-root", str(root))),
        "--output",
        str(output),
    ]
    clean_env = dict(env)
    clean_env.pop("AXIOM_RULESPEC_REPO_ROOTS", None)
    clean_env.pop("AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE", None)
    result = subprocess.run(
        current_command,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=clean_env,
    )
    output_text = result.stdout + result.stderr
    if not (
        result.returncode != 0
        and _UNKNOWN_EXPLICIT_ROOT_FLAG in output_text
        and _LEGACY_EXCLUSIVE_ROOT_FLAG in output_text
    ):
        return result

    unrepresentable_roots = [root for root in roots if os.pathsep in str(root)]
    if unrepresentable_roots:
        rendered = ", ".join(map(str, unrepresentable_roots))
        raise ValueError(
            "Legacy RuleSpec engine roots cannot contain the platform path "
            f"separator {os.pathsep!r}: {rendered}"
        )

    legacy_env = dict(clean_env)
    legacy_env["AXIOM_RULESPEC_REPO_ROOTS"] = os.pathsep.join(map(str, roots))
    legacy_env["AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE"] = "1"
    return subprocess.run(
        [
            *base_command,
            _LEGACY_EXCLUSIVE_ROOT_FLAG,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd) if cwd is not None else None,
        env=legacy_env,
    )
