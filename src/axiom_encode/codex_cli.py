"""Helpers for locating the Codex CLI used by Axiom Encode."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_codex_cli() -> str:
    """Return the Codex executable, preferring the Desktop-bundled CLI."""
    # The trusted supervisor supplies the exact hash-verified executable. Never
    # rediscover it through PATH, an override, or the Desktop app.
    if os.getenv("AXIOM_ENCODE_TRUSTED_RUNTIME") == "1" and os.getenv("CODEX_HOME"):
        executable = os.getenv("AXIOM_ENCODE_TRUSTED_CODEX_BIN")
        if not executable or not Path(executable).is_absolute():
            raise RuntimeError("Trusted runtime did not bind an absolute Codex CLI")
        return executable

    override = os.getenv("AXIOM_ENCODE_CODEX_BIN")
    if override:
        return override

    app_binary = Path("/Applications/Codex.app/Contents/Resources/codex")
    if app_binary.exists():
        return str(app_binary)

    return shutil.which("codex") or "codex"


_CHATGPT_ACCOUNT_MODEL_REJECTION = (
    "not supported when using Codex with a ChatGPT account"
)


def with_codex_model_availability_hint(error: str | None) -> str | None:
    """Name the explicit-model workaround when ChatGPT-account Codex rejects a model.

    ChatGPT-account Codex answers an unserved model with HTTP 400 "The '<model>'
    model is not supported when using Codex with a ChatGPT account" (observed for
    gpt-6-luna and gpt-6-sol on 2026-09-24). That is not a validator rejection,
    so encode escalation never retries it. Keyed on the returned text rather than
    the auth mode, so the hint stops appearing once that auth path serves the model.
    """
    if not error or _CHATGPT_ACCOUNT_MODEL_REJECTION not in error:
        return error
    return (
        f"{error}\nChatGPT-account Codex does not serve this model. Pass a model "
        "it serves explicitly: `encode --model gpt-5.6-terra --escalation-model "
        "gpt-5.6-sol`, `eval --runner claude:opus --runner codex:gpt-5.6-terra`, "
        "or AXIOM_ENCODE_REVIEWER_CODEX_MODEL=gpt-5.6-terra for the Codex "
        "reviewer."
    )


def codex_auth_json_path() -> Path:
    """Return the Codex CLI auth file, honoring the CODEX_HOME override."""
    override = os.getenv("CODEX_HOME")
    home = Path(override) if override else Path.home() / ".codex"
    return home / "auth.json"


def codex_auth_error() -> str | None:
    """Return a clear error when the Codex CLI has no usable auth file.

    ``axiom-encode encode`` defaults to the Codex backend (gpt-6-luna), which
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
