"""Helpers for locating the Codex CLI used by AutoRAC."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_codex_cli() -> str:
    """Return the Codex executable, preferring the Desktop-bundled CLI."""
    override = os.getenv("AUTORAC_CODEX_BIN")
    if override:
        return override

    app_binary = Path("/Applications/Codex.app/Contents/Resources/codex")
    if app_binary.exists():
        return str(app_binary)

    return shutil.which("codex") or "codex"
