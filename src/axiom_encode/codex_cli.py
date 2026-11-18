"""Helpers for locating the Codex CLI used by Axiom Encode."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

DEFAULT_CODEX_REASONING_EFFORT = "low"


def validate_codex_reasoning_effort(effort: str) -> str:
    """Validate Codex's extensible, nonempty reasoning-effort string."""
    if not isinstance(effort, str) or not effort or effort != effort.strip():
        raise ValueError(
            "Codex reasoning effort must be a nonempty string without surrounding whitespace"
        )
    return effort


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
    """Suggest account-specific checks only when Codex actually rejects a model.

    ChatGPT-account Codex answers an unserved model with HTTP 400 "The '<model>'
    model is not supported when using Codex with a ChatGPT account". That is not
    a validator rejection, so encode escalation never retries it. Keyed on the
    returned text rather than a model family or auth mode, so successful GPT-6
    calls receive no availability warning.
    """
    if not error or _CHATGPT_ACCOUNT_MODEL_REJECTION not in error:
        return error
    return (
        f"{error}\nThe signed-in Codex account rejected the requested model. "
        "Update the Codex CLI and check the models available to that account. "
        "Select an available model with `encode --model MODEL "
        "--escalation-model MODEL`, `eval --runner claude:opus "
        "--runner codex:MODEL`, or AXIOM_ENCODE_REVIEWER_CODEX_MODEL=MODEL "
        "for the Codex reviewer."
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


# Short values (``null``-like flags, auth modes) are not credentials, and
# replacing them would mangle ordinary text. Tokens and keys are far longer.
_MIN_REDACTED_SECRET_LENGTH = 16
CODEX_AUTH_REDACTION = "[redacted-codex-auth]"


def codex_auth_secrets(codex_home: Path | str | None) -> frozenset[str]:
    """Return every long string value in ``<codex_home>/auth.json``.

    ``auth.json`` holds the subscription's ID, access and refresh tokens, or a
    recorded API key. The file's layout is the Codex CLI's, so every string
    leaf long enough to be a credential is collected rather than named keys.
    A missing or unreadable file yields nothing; this never raises.
    """
    if not codex_home:
        return frozenset()
    try:
        payload = json.loads((Path(codex_home) / "auth.json").read_text())
    except (OSError, ValueError, RecursionError):
        return frozenset()
    secrets: set[str] = set()
    pending: list[object] = [payload]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, str) and len(value) >= _MIN_REDACTED_SECRET_LENGTH:
            secrets.add(value)
    return frozenset(secrets)


def redact_codex_auth_secrets(text: str, secrets: frozenset[str]) -> str:
    """Replace each secret, verbatim and JSON-escaped, before text is kept.

    Codex's ``--json`` event stream carries model output inside JSON strings,
    so a token can appear escaped there and verbatim once decoded.
    """
    if not text or not secrets:
        return text
    forms = set()
    for secret in secrets:
        forms.add(secret)
        forms.add(json.dumps(secret)[1:-1])
        forms.add(json.dumps(secret, ensure_ascii=False)[1:-1])
    # Longest first, so a secret containing another is replaced whole.
    for form in sorted(forms, key=len, reverse=True):
        if form:
            text = text.replace(form, CODEX_AUTH_REDACTION)
    return text
