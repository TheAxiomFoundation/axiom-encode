#!/usr/bin/env python3
"""Regenerate one merged RuleSpec module for the golden-regeneration drift check.

Invoked by ``axiom-encode drift-check --regenerate-cmd`` with three positional
arguments — ``{module} {merged} {output}`` — plus the rulespec root in the
``AXIOM_DRIFT_ROOT`` environment variable.

It derives the module's citation from its encoding manifest
(``<root>/.axiom/encoding-manifests/<module-without-jurisdiction-prefix>.json``,
``citation`` field), re-runs ``axiom-encode encode`` with the current encoder
into a temporary output root, and writes the regenerated YAML to ``{output}``.

Failures raise (non-zero exit) so the drift check records a visible error rather
than a silent "no drift".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def manifest_path(root: Path, module_rel: str) -> Path:
    parts = Path(module_rel).parts
    # Drop the leading jurisdiction segment (us/, us-ak/, uk/, ...).
    without_prefix = Path(*parts[1:]) if len(parts) > 1 else Path(module_rel)
    return root / ".axiom" / "encoding-manifests" / without_prefix.with_suffix(".json")


def read_citation(root: Path, module_rel: str) -> str:
    mpath = manifest_path(root, module_rel)
    if not mpath.exists():
        raise SystemExit(f"no encoding manifest for {module_rel} at {mpath}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    citation = manifest.get("citation")
    if not citation:
        raise SystemExit(f"manifest {mpath} has no citation field")
    return citation


def generated_subpath(root: Path, module_rel: str) -> Path:
    parts = Path(module_rel).parts
    without_prefix = Path(*parts[1:]) if len(parts) > 1 else Path(module_rel)
    return without_prefix.with_suffix(".yaml")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: drift_regenerate.py <module> <merged> <output>")
    module_rel, _merged, output = argv
    root = Path(os.environ.get("AXIOM_DRIFT_ROOT", ".")).resolve()
    backend = os.environ.get("AXIOM_DRIFT_BACKEND", "openai")

    citation = read_citation(root, module_rel)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "axiom-encode",
            "encode",
            citation,
            "--output",
            tmp,
            "--backend",
            backend,
            "--no-sync",
            "--skip-reviewers",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"encode failed for {citation!r}: {proc.stderr[-2000:]}"
            )
        generated = Path(tmp) / generated_subpath(root, module_rel)
        if not generated.exists():
            raise SystemExit(
                f"encode produced no file at {generated} for {citation!r}"
            )
        Path(output).write_text(generated.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
