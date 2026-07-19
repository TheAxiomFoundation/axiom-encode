"""Helpers for locating the Codex CLI used by Axiom Encode."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_codex_cli() -> str:
    """Return the Codex executable, preferring the Desktop-bundled CLI."""
    # The trusted supervisor admits exactly one hash-verified Codex directory
    # to PATH. Never escape that allowlist through an override or Desktop app.
    if os.getenv("AXIOM_ENCODE_TRUSTED_RUNTIME") == "1" and os.getenv("CODEX_HOME"):
        return shutil.which("codex") or "codex"

    override = os.getenv("AXIOM_ENCODE_CODEX_BIN")
    if override:
        return override

    app_binary = Path("/Applications/Codex.app/Contents/Resources/codex")
    if app_binary.exists():
        return str(app_binary)

    return shutil.which("codex") or "codex"


def codex_auth_json_path() -> Path:
    """Return the Codex CLI auth file, honoring the CODEX_HOME override."""
    override = os.getenv("CODEX_HOME")
    home = Path(override) if override else Path.home() / ".codex"
    return home / "auth.json"


def codex_auth_error() -> str | None:
    """Return a clear error when the Codex CLI has no usable auth file.

    ``axiom-encode encode`` defaults to the Codex backend (gpt-5.6-terra), which
    authenticates through the Codex CLI's ``auth.json`` (ChatGPT sign-in or
    an ``OPENAI_API_KEY`` recorded by ``codex login``). When neither that
    file nor ``OPENAI_API_KEY`` is present, encoding fails deep inside the
    subprocess with an opaque message; surface an actionable one instead.
    Returns ``None`` when auth is available.
    """
    if os.getenv("OPENAI_API_KEY"):
        return None
    auth_path = codex_auth_json_path()
    if auth_path.is_file():
        return None
    return (
        f"Codex backend requires authentication but {auth_path} was not found. "
        "Run `codex login` (ChatGPT sign-in) to create it, set OPENAI_API_KEY, "
        "or pass an explicit backend such as `--backend claude`."
    )
