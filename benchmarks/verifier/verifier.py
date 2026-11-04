#!/usr/bin/env python
"""Entry point for the EncodeBench verifier track.

Run from the repository root with the project environment, for example::

    uv run --with anthropic --with 'typesafe-sdk==0.6.0' \
        python benchmarks/verifier/verifier.py --help

This shim puts ``benchmarks/verifier`` (the package) and ``src`` (for
``axiom_encode.judges``) on ``sys.path``; the code lives in
``encodebench_verifier``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent / "src"
for entry in (str(_HERE), str(_SRC)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from encodebench_verifier.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
