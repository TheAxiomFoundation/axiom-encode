"""Judge runners and the ``--judge`` spec parser.

Specs: ``referee:<claude model id>``, ``jev`` or ``jev:<jev model id>``,
``replay:<response file>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import (
    CHANNEL_NATIVE,
    CHANNEL_VERDICT_FALLBACK,
    VERDICT_ERROR,
    VERDICT_FLAG,
    VERDICT_PASS,
    JudgeResponse,
    JudgeRunner,
    error_response,
)
from .jev import DEFAULT_JEV_MODEL, JevRunner
from .referee import REFEREE_KIND_MAP, RefereeRunner
from .replay import ReplayRunner

__all__ = [
    "CHANNEL_NATIVE",
    "CHANNEL_VERDICT_FALLBACK",
    "DEFAULT_JEV_MODEL",
    "JudgeResponse",
    "JudgeRunner",
    "JevRunner",
    "REFEREE_KIND_MAP",
    "RefereeRunner",
    "ReplayRunner",
    "VERDICT_ERROR",
    "VERDICT_FLAG",
    "VERDICT_PASS",
    "error_response",
    "make_runner",
]


def make_runner(
    spec: str,
    *,
    name: Optional[str] = None,
    provision_chars: int = 24_000,
    **options: Any,
) -> JudgeRunner:
    family, _, argument = spec.partition(":")
    family = family.strip().lower()
    if family == "referee":
        if not argument:
            raise ValueError("referee spec needs a model id: referee:<model>")
        return RefereeRunner(
            argument,
            name=name,
            provision_chars=provision_chars,
            **{
                k: v
                for k, v in options.items()
                if k in ("api_key", "max_attempts", "retry_seconds")
            },
        )
    if family == "jev":
        return JevRunner(
            argument or DEFAULT_JEV_MODEL,
            name=name,
            provision_chars=provision_chars,
            **{k: v for k, v in options.items() if k in ("timeout", "max_retries")},
        )
    if family == "replay":
        if not argument:
            raise ValueError("replay spec needs a file: replay:<responses.json>")
        return ReplayRunner(Path(argument), name=name or "replay")
    raise ValueError(f"unknown judge family {family!r} in spec {spec!r}")
