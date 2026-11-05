"""Shared Messages API output budget and encoder effort configuration."""

import os
from typing import Any

from axiom_encode.constants import DEFAULT_API_ENCODER_EFFORT

# Shared output-token budget for streaming RuleSpec responses.
ANTHROPIC_MESSAGES_MAX_TOKENS = 64_000
ANTHROPIC_API_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def resolve_api_encoder_effort(
    effort: str | None = None,
    *,
    efforts: frozenset[str] = ANTHROPIC_API_EFFORTS,
) -> str:
    """Resolve and validate the existing API encoder effort convention."""
    # A blank argument defers to the environment, and a blank value there to
    # the default. Fail before asking the API to reject every request.
    if effort and effort.strip():
        source, raw_effort = "effort", effort
    else:
        source = "AXIOM_API_ENCODER_EFFORT"
        raw_effort = os.environ.get("AXIOM_API_ENCODER_EFFORT") or ""
    resolved = raw_effort.strip().lower() or DEFAULT_API_ENCODER_EFFORT
    if resolved not in efforts:
        raise ValueError(
            f"{source}={raw_effort!r} is not a valid effort; use one of "
            f"{sorted(efforts)}"
        )
    return resolved


def anthropic_messages_options(
    *,
    model: str,
    prompt: str,
    effort: str,
    max_tokens: int = ANTHROPIC_MESSAGES_MAX_TOKENS,
) -> dict[str, Any]:
    """Build the shared request, omitting ``thinking`` and ``fallbacks``."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "output_config": {"effort": effort},
    }
