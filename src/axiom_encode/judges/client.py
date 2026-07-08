"""Cross-family Anthropic judge client for the LLM judge stages.

The generator is ``gpt-5.5``; the judges MUST run on a Claude-family model so a
judge's errors do not correlate with the generator's (the 9/9 identical
hardcoded-600,000 incident is the cautionary tale). This module enforces that
guard and the fail-closed contract:

* Any failure — missing ``ANTHROPIC_API_KEY``, missing ``anthropic`` SDK, API
  error after retries, JSON parse failure, or a cross-family guard trip — returns
  a :class:`JudgeCall` with a populated :attr:`JudgeCall.error`. Fail-open is
  banned; the caller turns that into a ``verdict == "error"`` event.
* Low-confidence verdicts escalate once from Haiku to Sonnet.
* Provision windows are truncated to a bounded budget; token counts are logged.

The client is deliberately generic: it takes a JSON schema and returns the
parsed payload plus call metadata. Each stage owns the prompt and the mapping
from payload to :class:`~axiom_encode.judges.run_log.JudgeEvent`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from axiom_encode.constants import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OPENAI_MODEL,
    JUDGE_ESCALATION_MODEL,
)

from .run_log import JudgeError, TokenCounts

# Provision windows are truncated to this many characters (~6k tokens) unless
# overridden. Head+tail are kept so both the operative opening and any closing
# boundary clauses survive.
DEFAULT_PROVISION_CHARS = 24_000
DEFAULT_MAX_TOKENS = 2048
DEFAULT_ESCALATE_BELOW = 0.6
DEFAULT_RETRY_SECONDS = 90.0
DEFAULT_MAX_ATTEMPTS = 2

_TRUNCATION_NOTE = "\n\n[... provision window truncated for judge token budget ...]\n\n"


def model_family(model: str) -> str:
    """Classify a model id into a provider family for the cross-family guard."""

    m = (model or "").lower()
    if m.startswith("claude") or m.startswith("anthropic."):
        return "anthropic"
    if (
        m.startswith("gpt")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
        or m.startswith("codex")
        or m.startswith("chatgpt")
    ):
        return "openai"
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "google"
    return "unknown"


def truncate_provision(text: str, max_chars: int = DEFAULT_PROVISION_CHARS) -> str:
    """Truncate a provision window to a bounded budget, keeping head and tail.

    Boundary inequalities and residual clauses often live at the end of a
    provision, so a naive head-only truncation would blind the fidelity referee
    to exactly the errors it hunts for.
    """

    if text is None:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    budget = max_chars - len(_TRUNCATION_NOTE)
    if budget <= 0:
        return text[:max_chars]
    head = budget * 2 // 3
    tail = budget - head
    return text[:head] + _TRUNCATION_NOTE + text[-tail:]


@dataclass
class JudgeCall:
    """Result of one (possibly escalated) judge call.

    ``payload`` is the parsed JSON matching the requested schema, or ``None`` on
    error. ``error`` is ``None`` on success and populated on any failure.
    """

    payload: Optional[dict[str, Any]]
    model: str
    escalated: bool
    tokens: TokenCounts
    error: Optional[JudgeError] = None
    raw_text: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.payload is not None


class CrossFamilyError(RuntimeError):
    """Raised (internally) when a judge model shares the generator's family."""


class JudgeClient:
    """Thin, fail-closed wrapper over the Anthropic Messages API for judging."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        escalation_model: Optional[str] = None,
        api_key: Optional[str] = None,
        generator_model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        provision_chars: Optional[int] = None,
        escalate_below: Optional[float] = None,
        retry_seconds: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> None:
        self.model = model or os.environ.get("AXIOM_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.escalation_model = escalation_model or os.environ.get(
            "AXIOM_JUDGE_ESCALATION_MODEL", JUDGE_ESCALATION_MODEL
        )
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.generator_model = generator_model or os.environ.get(
            "AXIOM_GENERATOR_MODEL", DEFAULT_OPENAI_MODEL
        )
        self.max_tokens = int(
            max_tokens
            if max_tokens is not None
            else os.environ.get("AXIOM_JUDGE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
        self.provision_chars = int(
            provision_chars
            if provision_chars is not None
            else os.environ.get("AXIOM_JUDGE_PROVISION_CHARS", DEFAULT_PROVISION_CHARS)
        )
        self.escalate_below = float(
            escalate_below
            if escalate_below is not None
            else os.environ.get("AXIOM_JUDGE_ESCALATE_BELOW", DEFAULT_ESCALATE_BELOW)
        )
        self.retry_seconds = float(
            retry_seconds
            if retry_seconds is not None
            else os.environ.get("AXIOM_JUDGE_RETRY_SECONDS", DEFAULT_RETRY_SECONDS)
        )
        self.max_attempts = int(
            max_attempts
            if max_attempts is not None
            else os.environ.get("AXIOM_JUDGE_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
        )

    # -- guards -----------------------------------------------------------

    def cross_family_problem(self, model: str) -> Optional[str]:
        """Return an error string if ``model`` violates the cross-family rule."""

        gen_family = model_family(self.generator_model)
        judge_family = model_family(model)
        if gen_family == "unknown":
            return (
                f"generator model {self.generator_model!r} has an unrecognized "
                "family; refusing to judge (cannot confirm the judge is "
                "cross-family with it)"
            )
        if judge_family == "unknown":
            return (
                f"judge model {model!r} has an unrecognized family; refusing to "
                "judge (cannot confirm it is cross-family with the generator)"
            )
        if judge_family == gen_family:
            return (
                f"judge model {model!r} shares the generator family "
                f"{gen_family!r} ({self.generator_model!r}); same-family "
                "self-review correlates errors and is banned by default"
            )
        if judge_family != "anthropic":
            return (
                f"judge model {model!r} is not a Claude-family model; the judge "
                "stages call the Anthropic API"
            )
        return None

    # -- core call --------------------------------------------------------

    def call(
        self,
        *,
        system: str,
        user_prompt: str,
        schema: dict[str, Any],
        escalate: bool = True,
    ) -> JudgeCall:
        """Run a judge call, escalating once on low confidence.

        Never raises for an operational failure — returns a :class:`JudgeCall`
        with ``error`` set instead (fail-closed).
        """

        problem = self.cross_family_problem(self.model)
        if problem:
            return JudgeCall(
                payload=None,
                model=self.model,
                escalated=False,
                tokens=TokenCounts(),
                error=JudgeError(type="cross_family_guard", message=problem),
            )
        if not self.api_key:
            return JudgeCall(
                payload=None,
                model=self.model,
                escalated=False,
                tokens=TokenCounts(),
                error=JudgeError(
                    type="missing_api_key",
                    message="ANTHROPIC_API_KEY is not set; cannot run judge",
                ),
            )

        first = self._one_model_call(
            model=self.model, system=system, user_prompt=user_prompt, schema=schema
        )
        if not first.ok:
            return first

        confidence = _payload_confidence(first.payload)
        should_escalate = (
            escalate
            and confidence is not None
            and confidence < self.escalate_below
            and self.escalation_model
            and self.escalation_model != self.model
        )
        if not should_escalate:
            return first

        esc_problem = self.cross_family_problem(self.escalation_model)
        if esc_problem:
            # The escalation model is misconfigured; keep the low-confidence
            # first verdict rather than silently dropping the escalation intent.
            return first

        second = self._one_model_call(
            model=self.escalation_model,
            system=system,
            user_prompt=user_prompt,
            schema=schema,
        )
        if not second.ok:
            # Escalation failed operationally; return the low-confidence but
            # valid first verdict, carrying the combined token spend.
            first.tokens = first.tokens + second.tokens
            return first
        second.escalated = True
        second.tokens = first.tokens + second.tokens
        return second

    def _one_model_call(
        self,
        *,
        model: str,
        system: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> JudgeCall:
        try:
            import anthropic
        except ImportError as exc:
            return JudgeCall(
                payload=None,
                model=model,
                escalated=False,
                tokens=TokenCounts(),
                error=JudgeError(
                    type="sdk_missing",
                    message=(
                        "anthropic SDK not installed; install axiom-encode[api] "
                        f"({exc})"
                    ),
                ),
            )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001 - never raise for an operational failure
            return JudgeCall(
                payload=None,
                model=model,
                escalated=False,
                tokens=TokenCounts(),
                error=JudgeError(
                    type="client_init_error",
                    message=f"could not construct Anthropic client: {exc}",
                ),
            )
        # Belt-and-suspenders: constrain the output to JSON *and* instruct the
        # model, so we degrade gracefully if a configured model lacks
        # output_config.format support.
        json_system = (
            system + "\n\nRespond with a single JSON object only — no prose, no code "
            "fences. It must conform to the provided schema."
        )
        messages = [{"role": "user", "content": user_prompt}]

        last_error: Optional[JudgeError] = None
        for attempt in range(self.max_attempts):
            try:
                text, tokens = self._invoke(
                    client, model, json_system, messages, schema, anthropic
                )
            except Exception as exc:  # noqa: BLE001 - normalized below
                retryable, err = _classify_exception(anthropic, exc)
                last_error = err
                if retryable and attempt < self.max_attempts - 1:
                    time.sleep(self.retry_seconds)
                    continue
                return JudgeCall(
                    payload=None,
                    model=model,
                    escalated=False,
                    tokens=TokenCounts(),
                    error=err,
                )

            payload = _extract_json(text)
            if payload is None:
                # A parse failure is a fail-closed error, never a pass.
                return JudgeCall(
                    payload=None,
                    model=model,
                    escalated=False,
                    tokens=tokens,
                    error=JudgeError(
                        type="parse_error",
                        message="judge response was not valid JSON",
                    ),
                    raw_text=text,
                )
            missing = [k for k in schema.get("required", []) if k not in payload]
            if missing:
                # Valid JSON that omits required keys is still not a usable
                # verdict — treat it as an error, never let it fall through to a
                # stage that would read a missing verdict as a pass (fail-open).
                return JudgeCall(
                    payload=None,
                    model=model,
                    escalated=False,
                    tokens=tokens,
                    error=JudgeError(
                        type="schema_error",
                        message=f"judge response missing required keys: {missing}",
                    ),
                    raw_text=text,
                )
            return JudgeCall(
                payload=payload,
                model=model,
                escalated=False,
                tokens=tokens,
                raw_text=text,
            )

        return JudgeCall(
            payload=None,
            model=model,
            escalated=False,
            tokens=TokenCounts(),
            error=last_error or JudgeError(type="unknown", message="judge call failed"),
        )

    def _invoke(
        self,
        client: Any,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        anthropic_mod: Any,
    ) -> tuple[str, TokenCounts]:
        """Make one Messages API request; return (text, tokens).

        Tries structured outputs first, falls back to a plain request if the
        SDK/model rejects ``output_config``.
        """

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        # Bind defensively — an SDK without BadRequestError must not raise an
        # AttributeError while handling an unrelated exception.
        bad_request = getattr(anthropic_mod, "BadRequestError", ())
        try:
            response = client.messages.create(
                output_config={"format": {"type": "json_schema", "schema": schema}},
                **kwargs,
            )
        except TypeError:
            # SDK too old for output_config; plain request + prompt-guided JSON.
            response = client.messages.create(**kwargs)
        except bad_request:
            # Model rejected the schema/format; retry plain.
            response = client.messages.create(**kwargs)

        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        usage = getattr(response, "usage", None)
        tokens = TokenCounts(
            input=getattr(usage, "input_tokens", 0) or 0,
            output=getattr(usage, "output_tokens", 0) or 0,
        )
        return text, tokens


def _payload_confidence(payload: Optional[dict[str, Any]]) -> Optional[float]:
    if not payload:
        return None
    value = payload.get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _classify_exception(anthropic_mod: Any, exc: Exception) -> tuple[bool, JudgeError]:
    """Map an SDK exception to (retryable, JudgeError)."""

    rate_limit = getattr(anthropic_mod, "RateLimitError", ())
    conn = getattr(anthropic_mod, "APIConnectionError", ())
    server = getattr(anthropic_mod, "InternalServerError", ())
    status = getattr(anthropic_mod, "APIStatusError", ())
    if isinstance(exc, rate_limit):
        return True, JudgeError(type="rate_limit", message=str(exc))
    if isinstance(exc, conn):
        return True, JudgeError(type="connection_error", message=str(exc))
    if isinstance(exc, server):
        return True, JudgeError(type="server_error", message=str(exc))
    if isinstance(exc, status):
        code = getattr(exc, "status_code", None)
        retryable = code is not None and code >= 500
        return retryable, JudgeError(type=f"api_status_{code}", message=str(exc))
    return False, JudgeError(type="unexpected", message=f"{type(exc).__name__}: {exc}")


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Parse a JSON object from a model response, tolerating fences/prose."""

    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        # drop a leading language tag like ``json``
        newline = candidate.find("\n")
        if newline != -1 and " " not in candidate[:newline]:
            candidate = candidate[newline + 1 :]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None
