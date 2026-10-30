#!/usr/bin/env python3
"""Fail a workflow step with a typed infra issue (see axiom_encode.infra_checks)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_implementation_path = (
    Path(__file__).resolve().parents[1] / "src" / "axiom_encode" / "infra_checks.py"
)
_spec = importlib.util.spec_from_file_location(
    "_axiom_encode_infra_checks",
    _implementation_path,
)
if _spec is None or _spec.loader is None:
    print("could not load axiom_encode.infra_checks", file=sys.stderr)
    raise SystemExit(1)
_implementation = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _implementation
_spec.loader.exec_module(_implementation)

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
