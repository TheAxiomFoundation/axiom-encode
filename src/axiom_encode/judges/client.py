"""Cross-family Anthropic judge client for the LLM judge stages.

The generator is ``gpt-6-luna``; the judges MUST run on a Claude-family
model so a judge's errors do not correlate with the generator's (the 9/9 identical
hardcoded-600,000 incident is the cautionary tale). This module enforces that
guard and the fail-closed contract:

* Any failure — missing ``ANTHROPIC_API_KEY``, missing ``anthropic`` SDK, API
  error after retries, JSON parse failure, or a cross-family guard trip — returns
  a :class:`JudgeCall` with a populated :attr:`JudgeCall.error`. Fail-open is
  banned; the caller turns that into a ``verdict == "error"`` event.
* Low-confidence verdicts escalate once from Sonnet 5.5 to Opus 5.5.
* Provision windows are truncated to a bounded budget; token counts are logged.
* A reply cut off by the output budget (``stop_reason == "max_tokens"``) is a
  ``max_tokens`` error naming the budget, not a generic parse error.
* A safety refusal (``stop_reason == "refusal"``) is its own fail-closed
  ``refusal`` error, never a parse failure. There is deliberately no
  server-side refusal fallback: rerouting declined requests to another model
  would swap the judge on exactly the subset of provisions that trigger
  refusals, a selection bias that calibration could not see. A refusal is not
  low confidence, so it never escalates either.
* Current Claude models think adaptively by default and ``max_tokens`` caps
  thinking plus the JSON verdict, so :data:`DEFAULT_MAX_TOKENS` leaves room for
  both under the SDK's non-streaming ceiling (:data:`NONSTREAMING_MAX_TOKENS`).
* ``AXIOM_JUDGE_EFFORT`` optionally sets ``output_config.effort`` for the first
  call; the escalation call uses ``AXIOM_JUDGE_ESCALATION_EFFORT`` (default
  ``high``), so escalating never means thinking less. An effort the request
  cannot honor fails closed (``effort_rejected``) instead of silently running
  at the model default; an invalid value is a ``config_error``.
* An escalation that fails is recorded on :attr:`JudgeCall.escalation_error`
  (surfaced in the event's ``extra``) while the first verdict stands.

The client is deliberately generic: it takes a JSON schema and returns the
parsed payload plus call metadata. Each stage owns the prompt and the mapping
from payload to :class:`~axiom_encode.judges.run_log.JudgeEvent`.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
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
# The anthropic SDK (0.83.0, ``_calculate_nonstreaming_timeout``) refuses a
# non-streaming request whose max_tokens implies more than ten minutes
# (3,600 * max_tokens / 128,000 > 600), i.e. above 21,333 tokens. It also caps
# the models in its ``MODEL_NONSTREAMING_TOKENS`` table (Opus 4 and 4.1) at
# 8,192; neither judge default is in that table. A refusal is reported as a
# named ``max_tokens_config`` judge error.
NONSTREAMING_MAX_TOKENS = 21_333
# Output budget for one judge call. 2,048 truncated the statutory-fidelity
# referee's findings JSON on large modules (about 30k input tokens) and 8,192
# still truncated one Haiku 4.5 reply on a 34k-token module (EncodeBench verifier
# track, 2026-09-19). 16,000 leaves room under NONSTREAMING_MAX_TOKENS. Output is
# billed per token generated, so the higher ceiling costs nothing on replies
# that finish sooner. On current Claude models adaptive thinking counts against
# this budget too, so it must cover thinking plus the JSON verdict.
DEFAULT_MAX_TOKENS = 16_000
# Stop reasons meaning the reply was cut off before it finished, mapped to the
# judge error type reported when that leaves no complete JSON payload.
# ``model_context_window_exceeded`` is a beta stop reason in anthropic 0.83.0
# (``BetaStopReason``); listing it is harmless on calls that never return it.
TRUNCATION_STOP_REASONS = {
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "context_window_exceeded",
}
# The API's effort levels on current Claude models.
VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
DEFAULT_ESCALATION_EFFORT = "high"
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
    # Which request succeeded: "structured" (schema + effort), "effort_only"
    # (schema rejected, effort kept) or "plain" (no output_config at all).
    request_shape: Optional[str] = None
    # Set when a low-confidence verdict tried to escalate and the escalation
    # call failed; the first verdict stands.
    escalation_error: Optional[JudgeError] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.payload is not None


def call_diagnostics(call: JudgeCall) -> dict[str, Any]:
    """Event ``extra`` entries a stage should record for a successful call.

    Empty in the normal case (structured request, no failed escalation), so
    events only grow when something calibration should know about happened.
    """

    extra: dict[str, Any] = {}
    if call.request_shape and call.request_shape != "structured":
        extra["request_shape"] = call.request_shape
    if call.escalation_error is not None:
        extra["escalation_error"] = {
            "type": call.escalation_error.type,
            "message": call.escalation_error.message,
        }
    return extra


def with_call_diagnostics(event: Any, call: JudgeCall) -> Any:
    """Merge :func:`call_diagnostics` into ``event.extra`` and return the event."""

    event.extra.update(call_diagnostics(call))
    return event


def _normalize_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    return value or None


class CrossFamilyError(RuntimeError):
    """Raised (internally) when a judge model shares the generator's family."""


class EffortRejectedError(RuntimeError):
    """Raised (internally) when no request shape can carry the configured effort."""


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
        effort: Optional[str] = None,
        escalation_effort: Optional[str] = None,
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
        # None means the model's own default effort.
        self.effort = _normalize_effort(
            effort if effort is not None else os.environ.get("AXIOM_JUDGE_EFFORT")
        )
        # Escalation defaults to "high": Opus 5.5's own default is medium, which
        # would make the escalation think less than a Sonnet 5.5 first call.
        # Set AXIOM_JUDGE_ESCALATION_EFFORT="" to use the model default.
        self.escalation_effort = _normalize_effort(
            escalation_effort
            if escalation_effort is not None
            else os.environ.get(
                "AXIOM_JUDGE_ESCALATION_EFFORT", DEFAULT_ESCALATION_EFFORT
            )
        )

    # -- guards -----------------------------------------------------------

    def config_problem(self) -> Optional[str]:
        """Return an error string if the client is configured to fail.

        A budget above the SDK's non-streaming ceiling is not checked here: the
        SDK refuses it before sending, reported as ``max_tokens_config``.
        """

        for name, value in (
            ("AXIOM_JUDGE_EFFORT", self.effort),
            ("AXIOM_JUDGE_ESCALATION_EFFORT", self.escalation_effort),
        ):
            if value is not None and value not in VALID_EFFORTS:
                return (
                    f"{name}={value!r} is not a valid effort; use one of "
                    f"{sorted(VALID_EFFORTS)}"
                )
        if self.max_tokens < 1:
            return f"AXIOM_JUDGE_MAX_TOKENS={self.max_tokens} must be at least 1"
        return None

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
        config_problem = self.config_problem()
        if config_problem:
            return JudgeCall(
                payload=None,
                model=self.model,
                escalated=False,
                tokens=TokenCounts(),
                error=JudgeError(type="config_error", message=config_problem),
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
            model=self.model,
            system=system,
            user_prompt=user_prompt,
            schema=schema,
            effort=self.effort,
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
            effort=self.escalation_effort,
        )
        if not second.ok:
            # Escalation failed; return the low-confidence but valid first
            # verdict with the combined token spend, and record why so the
            # failure is visible in the event rather than only in a token total.
            first.tokens = first.tokens + second.tokens
            first.escalation_error = second.error or JudgeError(
                type="unknown", message="escalation call failed"
            )
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
        effort: Optional[str] = None,
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
                text, tokens, stop_reason, refusal_category, shape = self._invoke(
                    client, model, json_system, messages, schema, anthropic, effort
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

            if stop_reason == "refusal":
                # A safety decline carries no verdict, even if its text parses.
                # Name it so it isn't misread as a malformed response; still
                # fail-closed.
                return JudgeCall(
                    payload=None,
                    model=model,
                    escalated=False,
                    tokens=tokens,
                    error=JudgeError(
                        type="refusal",
                        message=(
                            "judge model declined the request"
                            + (
                                f" (category: {refusal_category})"
                                if refusal_category
                                else ""
                            )
                        ),
                    ),
                    raw_text=text or None,
                )

            payload = _extract_json(text)
            missing = (
                [k for k in schema.get("required", []) if k not in payload]
                if payload is not None
                else None
            )
            truncation = TRUNCATION_STOP_REASONS.get(stop_reason or "")
            if truncation and (payload is None or missing):
                # The reply was cut off before its JSON closed. It is a
                # fail-closed error like a parse failure, but named, so the
                # cause (a budget, not the model) is visible in the run log.
                if truncation == "max_tokens":
                    message = (
                        f"judge reply hit the {self.max_tokens}-token output "
                        f"budget before its JSON closed ({tokens.output} output "
                        "tokens); raise AXIOM_JUDGE_MAX_TOKENS (non-streaming judge "
                        f"calls allow at most {NONSTREAMING_MAX_TOKENS:,})"
                    )
                else:
                    message = (
                        "judge reply stopped at the model's context window before "
                        f"its JSON closed ({tokens.input} input tokens); lower "
                        "AXIOM_JUDGE_PROVISION_CHARS"
                    )
                return JudgeCall(
                    payload=None,
                    model=model,
                    escalated=False,
                    tokens=tokens,
                    error=JudgeError(type=truncation, message=message),
                    raw_text=text,
                )
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
                request_shape=shape,
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
        effort: Optional[str] = None,
    ) -> tuple[str, TokenCounts, Optional[str], Optional[str], str]:
        """Make one Messages API request.

        Returns ``(text, tokens, stop_reason, refusal_category, request_shape)``.
        Thinking blocks are skipped; only ``text`` blocks form the verdict.
        Tries structured outputs first. If the SDK or model rejects
        ``output_config``, retries without the schema but keeps ``effort`` when
        one is configured; if effort itself cannot be sent, raises
        :class:`EffortRejectedError` (fail-closed) instead of silently judging
        at the model default.
        """

        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema}
        }
        if effort:
            output_config["effort"] = effort
        # Bind defensively — an SDK without BadRequestError must not raise an
        # AttributeError while handling an unrelated exception.
        bad_request = getattr(anthropic_mod, "BadRequestError", None)
        rejected: tuple[type[BaseException], ...] = (
            (TypeError, bad_request) if bad_request else (TypeError,)
        )
        shape = "structured"
        try:
            response = client.messages.create(output_config=output_config, **kwargs)
        except rejected as first_exc:
            # TypeError: SDK too old for output_config. BadRequestError: the
            # model rejected the schema/format (or the effort). Fall back to
            # prompt-guided JSON, but never drop a configured effort silently.
            if effort:
                try:
                    response = client.messages.create(
                        output_config={"effort": effort}, **kwargs
                    )
                    shape = "effort_only"
                except rejected as exc:
                    raise EffortRejectedError(
                        f"judge effort {effort!r} could not be sent to {model} "
                        f"(structured request: {first_exc}; effort-only request: "
                        f"{exc})"
                    ) from exc
            else:
                response = client.messages.create(**kwargs)
                shape = "plain"

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
        stop_reason = getattr(response, "stop_reason", None)
        # stop_details is populated only on refusals; guard before reading.
        stop_details = getattr(response, "stop_details", None)
        # SDKs that predate the typed field keep it as an extra pydantic field,
        # which arrives as a plain dict.
        if isinstance(stop_details, Mapping):
            refusal_category = stop_details.get("category")
        elif stop_details is not None:
            refusal_category = getattr(stop_details, "category", None)
        else:
            refusal_category = None
        return (
            text,
            tokens,
            stop_reason if isinstance(stop_reason, str) else None,
            refusal_category,
            shape,
        )


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

    if isinstance(exc, EffortRejectedError):
        return False, JudgeError(type="effort_rejected", message=str(exc))

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
    if isinstance(exc, ValueError) and "Streaming is required" in str(exc):
        # The SDK refused the non-streaming request before sending it because
        # the output budget is too large for it (see NONSTREAMING_MAX_TOKENS).
        return False, JudgeError(
            type="max_tokens_config",
            message=(
                "the anthropic SDK refused a non-streaming judge call at this "
                f"output budget (at most {NONSTREAMING_MAX_TOKENS:,} tokens, and "
                "8,192 for the Opus 4 and 4.1 models); lower "
                f"AXIOM_JUDGE_MAX_TOKENS. SDK said: {exc}"
            ),
        )
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
