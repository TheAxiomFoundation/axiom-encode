"""Select an actual local test engine without changing another checkout."""

import os
import subprocess
from pathlib import Path

from axiom_encode.engine_binding import bind_clean_engine_checkout

DEFAULT_ENGINE_CHECKOUTS = (
    Path(
        "/Users/maxghenis/TheAxiomFoundation/_worktrees/"
        "axiom-rules-engine-canonical-loader-hard-cut"
    ),
    Path("/Users/maxghenis/TheAxiomFoundation/axiom-rules-engine"),
)


def local_engine_paths() -> tuple[Path, Path | None]:
    """Honor explicit selection and probe the executable production will use.

    An explicit checkout is required to work; it never falls back or skips.
    An optional ref verifies its clean source and existing binary receipt without
    rebuilding. Unconfigured local integration remains optional, as in CI.
    """
    configured = os.environ.get("AXIOM_TEST_ENGINE_CHECKOUT")
    engine_ref = os.environ.get("AXIOM_TEST_ENGINE_REF")
    for name, value in (
        ("AXIOM_TEST_ENGINE_CHECKOUT", configured),
        ("AXIOM_TEST_ENGINE_REF", engine_ref),
    ):
        if value is not None and not value.strip():
            raise RuntimeError(f"{name} must not be blank")
    if engine_ref is not None and configured is None:
        raise RuntimeError("AXIOM_TEST_ENGINE_REF requires AXIOM_TEST_ENGINE_CHECKOUT")
    candidates = (Path(configured),) if configured else DEFAULT_ENGINE_CHECKOUTS
    for checkout in candidates:
        # Match ValidatorPipeline's release/debug/bare resolution order. Never
        # probe debug and then execute an incompatible release from that checkout.
        binary = next(
            (
                candidate
                for candidate in (
                    checkout / "target/release/axiom-rules-engine",
                    checkout / "target/debug/axiom-rules-engine",
                    checkout / "axiom-rules-engine",
                )
                if candidate.exists()
            ),
            None,
        )
        if binary is None:
            continue
        if not binary.is_file():
            if configured is not None:
                raise RuntimeError(f"Explicit test engine is not a file: {binary}")
            continue
        if engine_ref is not None:
            binding = bind_clean_engine_checkout(
                checkout, engine_ref, allow_build=False
            )
            if Path(binding["binary"]) != binary.resolve():
                raise RuntimeError(
                    "Explicit test engine binding selects a different binary"
                )
        try:
            probe = subprocess.run(
                [str(binary), "compile", "--help"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            if configured:
                raise
            continue
        if probe.returncode == 0 and "--rulespec-root" in probe.stdout + probe.stderr:
            return checkout, binary
    if configured:
        raise RuntimeError(
            "Explicit test engine has no compatible executable: " + configured
        )
    return DEFAULT_ENGINE_CHECKOUTS[-1], None
