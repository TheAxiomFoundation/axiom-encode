"""The rules engine, as the compare step uses it.

This module is the only place where the compare step touches the rest of
axiom-encode: it resolves the engine commit that the RuleSpec repository pins
and binds a receipted engine binary to it (``engine_binding``), and compiles
through the shared compile helper (``rules_engine_compat``). Everything else
in the package talks to the small :class:`Engine` surface, so the package can
move to another repository with this one file rewritten.
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

WORKFLOW_TOOLCHAIN = Path(".axiom") / "workflow-toolchain.toml"
ENGINE_PIN_FIELD = "axiom_rules_engine_ref"
COMPILE_TIMEOUT_SECONDS = 300
RUN_TIMEOUT_SECONDS = 600
_PASSED_ENVIRONMENT = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")


class EngineError(RuntimeError):
    """The engine could not be resolved, or a compile or run failed."""


def pinned_engine_commit(rulespec_root: Path) -> tuple[str, str]:
    """The engine commit the RuleSpec repository pins, and the file that pins it.

    ``.axiom/workflow-toolchain.toml`` (``[workflow_toolchain]``) is read
    first; ``.axiom/toolchain.toml`` (``[toolchain]``) is the fallback.
    """

    from axiom_encode.engine_binding import (
        EngineBindingError,
        load_declared_engine_pin,
        require_engine_ref_sha,
    )

    path = Path(rulespec_root) / WORKFLOW_TOOLCHAIN
    try:
        if path.is_file() and not path.is_symlink():
            table = tomllib.loads(path.read_text(encoding="utf-8")).get(
                "workflow_toolchain"
            )
            if isinstance(table, dict) and ENGINE_PIN_FIELD in table:
                sha = require_engine_ref_sha(
                    table[ENGINE_PIN_FIELD],
                    description=f"[workflow_toolchain].{ENGINE_PIN_FIELD} in {path}",
                )
                return sha, str(WORKFLOW_TOOLCHAIN)
        pin = load_declared_engine_pin(Path(rulespec_root))
    except (
        OSError,
        UnicodeError,
        tomllib.TOMLDecodeError,
        EngineBindingError,
    ) as error:
        raise EngineError(
            f"cannot read the engine pin of {rulespec_root}: {error}"
        ) from error
    if pin is None:
        raise EngineError(
            f"{rulespec_root} pins no engine commit ({WORKFLOW_TOOLCHAIN} or "
            ".axiom/toolchain.toml); pass --axiom-rules-engine-ref"
        )
    return pin.sha, ".axiom/toolchain.toml"


def bind_engine(engine_checkout: Path, commit: str) -> Path:
    """A binary in ``engine_checkout`` with a verified receipt for ``commit``."""

    from axiom_encode.engine_binding import (
        EngineBindingError,
        EnginePin,
        require_engine_ref_sha,
        resolve_pinned_engine_binary,
    )

    try:
        require_engine_ref_sha(commit, description="engine ref")
        return resolve_pinned_engine_binary(
            Path(engine_checkout),
            EnginePin(commit, Path("<compare-sources>")),
            allow_build=False,
        )
    except EngineBindingError as error:
        raise EngineError(
            f"{error}\nBuild and receipt the pinned engine with: axiom-encode engine-bind "
            f"--axiom-rules-engine-path {engine_checkout} --pin {commit}"
        ) from error


class Engine:
    """Compile modules and run compiled artifacts with one engine binary."""

    def __init__(self, binary: Path, commit: str):
        self.binary = Path(binary)
        self.commit = commit
        self._env = {
            key: os.environ[key] for key in _PASSED_ENVIRONMENT if key in os.environ
        }

    def compile(
        self, module: Path, rulespec_root: Path, output: Path
    ) -> dict[str, Any]:
        from axiom_encode.rules_engine_compat import run_rulespec_compile

        try:
            result = run_rulespec_compile(
                binary=self.binary,
                program=module,
                rulespec_roots=[rulespec_root],
                output=output,
                cwd=None,
                env=self._env,
                timeout=COMPILE_TIMEOUT_SECONDS,
            )
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            raise EngineError(
                f"the engine could not compile {module}: {error}"
            ) from error
        if result.returncode:
            detail = (result.stderr.strip() or result.stdout.strip())[-4000:]
            raise EngineError(f"the engine could not compile {module}:\n{detail}")
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise EngineError(
                f"the engine wrote no readable artifact for {module}"
            ) from error

    def run(self, artifact: Path, request: dict[str, Any]) -> dict[str, Any]:
        try:
            result = subprocess.run(
                [str(self.binary), "run-compiled", "--artifact", str(artifact)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_SECONDS,
                env=self._env,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise EngineError(
                f"the engine could not run {artifact.name}: {error}"
            ) from error
        if result.returncode:
            detail = (result.stderr.strip() or result.stdout.strip())[-4000:]
            raise EngineError(f"the engine refused a request:\n{detail}")
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise EngineError("the engine returned output that is not JSON") from error
