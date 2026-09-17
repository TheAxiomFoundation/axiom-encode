"""TypeSafe System One client for the statutory-fidelity screen.

System One (TypeSafe's ``jev-*`` models) answers typed questions about a JSON
state with calibrated probabilities: a ``Choice`` returns a label with a
probability per label, a ``Noul`` returns a single probability in [0, 1]. It
returns no prose, no clause reference and no artifact locus, so it cannot fill
a fidelity :class:`~axiom_encode.judges.run_log.Finding` on its own. The screen
stage (:mod:`~axiom_encode.judges.statutory_fidelity_screen`) uses it as a
cheap, fast pre-screen before the LLM referee.

This client mirrors :class:`~axiom_encode.judges.client.JudgeClient`'s
contract:

* Fail-closed: a missing ``TYPESAFE_API_KEY``, a missing ``typesafe-sdk``,
  any SDK exception, an unexpected response shape, or a cross-family guard
  trip returns a :class:`SystemOneCall` with :attr:`SystemOneCall.error`
  set. The caller turns that into a ``verdict == "error"`` event.
* Cross-family: the generator's family is checked before the call and the
  responding model's family after it. Both must be known and must differ,
  and the responding model must be TypeSafe's (``typesafe``).
* Secrets never reach a run log: SDK exceptions are mapped by class name
  only (their text, which may echo request context, is dropped) and the key
  is never placed on the call result or the event.
* Model id, latency and token usage are recorded on every call.

The SDK is imported lazily so the package works without the optional
``typesafe`` extra installed (``pip install axiom-encode[typesafe]`` pins
``typesafe-sdk==0.6.0``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union

from axiom_encode.constants import DEFAULT_OPENAI_MODEL

from .client import DEFAULT_PROVISION_CHARS, model_family
from .run_log import JudgeError, TokenCounts, coerce_confidence

TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
SYSTEM_ONE_FAMILY = "typesafe"
SDK_DISTRIBUTION = "typesafe-sdk"

# Per-request HTTP timeout and SDK-side retry budget. The SDK's own default
# timeout is 10 s; the pilot's median latency was 0.19 s, so 30 s is generous.
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2


@dataclass(frozen=True)
class ChoiceQuestion:
    """A ``Choice`` question: pick one label; probabilities per label."""

    instructions: str
    criteria: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True)
class NoulQuestion:
    """A ``Noul`` question: a single yes-probability in [0, 1]."""

    instructions: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "noul", "instructions": self.instructions}


Question = Union[ChoiceQuestion, NoulQuestion]


@dataclass(frozen=True)
class ChoiceAnswer:
    """The mapped answer to a :class:`ChoiceQuestion`."""

    choice: str
    confidence: Optional[float]
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass
class SystemOneCall:
    """Result of one System One request.

    ``answers`` maps each requested question name to a :class:`ChoiceAnswer`
    or a float (for a Noul), or is ``None`` on error. ``error`` is ``None`` on
    success and populated on any failure. The key is never stored here.
    """

    answers: Optional[dict[str, Union[ChoiceAnswer, float]]]
    model: Optional[str]
    family: Optional[str]
    tokens: TokenCounts
    latency_ms: int
    error: Optional[JudgeError] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.answers is not None


def questions_sha256(
    state: Mapping[str, Any], questions: Mapping[str, Question]
) -> str:
    """Stable digest of one request (state + questions) for reproducibility."""

    payload = {
        "state": dict(state),
        "questions": {name: q.to_dict() for name, q in questions.items()},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


class SystemOneClient:
    """Fail-closed wrapper over ``typesafe_sdk.TypeSafeClient.system_one``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        generator_model: Optional[str] = None,
        model: Optional[str] = None,
        provision_chars: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.environ.get(TYPESAFE_API_KEY_ENV)
        )
        self.generator_model = generator_model or os.environ.get(
            "AXIOM_GENERATOR_MODEL", DEFAULT_OPENAI_MODEL
        )
        # ``None`` lets the SDK pick its default model (``jev-latest``); the
        # responding model id is recorded on every call regardless.
        self.model = model or os.environ.get("AXIOM_JUDGE_SCREEN_MODEL") or None
        self.provision_chars = int(
            provision_chars
            if provision_chars is not None
            else os.environ.get("AXIOM_JUDGE_PROVISION_CHARS", DEFAULT_PROVISION_CHARS)
        )
        self.timeout = (
            float(timeout)
            if timeout is not None
            else _env_float(
                "AXIOM_JUDGE_SCREEN_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
            )
        )
        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else _env_int("AXIOM_JUDGE_SCREEN_MAX_RETRIES", DEFAULT_MAX_RETRIES)
        )

    def __repr__(self) -> str:  # never echo the key
        return (
            f"SystemOneClient(model={self.model!r}, "
            f"generator_model={self.generator_model!r}, "
            f"api_key={'set' if self.api_key else 'unset'})"
        )

    # -- guards -----------------------------------------------------------

    def cross_family_problem(self, model: Optional[str]) -> Optional[str]:
        """Return an error string if the screen would violate the family rule.

        ``model`` is the responding model id (``None`` before the call, when
        only the generator side can be checked).
        """

        gen_family = model_family(self.generator_model)
        if gen_family == "unknown":
            return (
                f"generator model {self.generator_model!r} has an unrecognized "
                "family; refusing to screen (cannot confirm the screen is "
                "cross-family with it)"
            )
        if gen_family == SYSTEM_ONE_FAMILY:
            return (
                f"generator model {self.generator_model!r} shares the screen "
                f"family {SYSTEM_ONE_FAMILY!r}; same-family self-review "
                "correlates errors and is banned by default"
            )
        if model is None:
            return None
        judge_family = model_family(model)
        if judge_family == "unknown":
            return (
                f"screen model {model!r} has an unrecognized family; refusing "
                "the verdict (cannot confirm it is cross-family with the generator)"
            )
        if judge_family == gen_family:
            return (
                f"screen model {model!r} shares the generator family "
                f"{gen_family!r} ({self.generator_model!r}); same-family "
                "self-review correlates errors and is banned by default"
            )
        if judge_family != SYSTEM_ONE_FAMILY:
            return (
                f"screen model {model!r} is not a TypeSafe System One model; "
                "the screen calls the TypeSafe service"
            )
        return None

    # -- core call --------------------------------------------------------

    def call(
        self,
        *,
        state: Mapping[str, Any],
        questions: Mapping[str, Question],
    ) -> SystemOneCall:
        """Ask ``questions`` about ``state``; never raises for an operational failure."""

        def failed(error: JudgeError, **kwargs: Any) -> SystemOneCall:
            return SystemOneCall(
                answers=None,
                model=kwargs.get("model"),
                family=kwargs.get("family"),
                tokens=kwargs.get("tokens", TokenCounts()),
                latency_ms=kwargs.get("latency_ms", 0),
                error=error,
            )

        if not questions:
            return failed(
                JudgeError(type="empty_questions", message="no questions to ask")
            )
        problem = self.cross_family_problem(None)
        if problem:
            return failed(JudgeError(type="cross_family_guard", message=problem))
        if not self.api_key:
            return failed(
                JudgeError(
                    type="missing_api_key",
                    message=f"{TYPESAFE_API_KEY_ENV} is not set; cannot run the screen",
                )
            )

        try:
            import typesafe_sdk
        except ImportError:
            return failed(
                JudgeError(
                    type="sdk_missing",
                    message=(
                        f"{SDK_DISTRIBUTION} not installed; install "
                        "axiom-encode[typesafe]"
                    ),
                )
            )

        try:
            sdk_questions = _to_sdk_questions(typesafe_sdk, questions)
            sdk_client = typesafe_sdk.TypeSafeClient(
                api_key=self.api_key,
                model=self.model,
                retry=typesafe_sdk.RetryPolicy(max_retries=self.max_retries),
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001 - normalized, by class name only
            return failed(_class_only_error(exc, phase="client construction"))

        started = time.perf_counter()
        try:
            response = sdk_client.system_one(dict(state), sdk_questions)
        except Exception as exc:  # noqa: BLE001 - normalized, by class name only
            latency_ms = int((time.perf_counter() - started) * 1000)
            return failed(
                _class_only_error(exc, phase="request"), latency_ms=latency_ms
            )
        finally:
            close = getattr(sdk_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - closing is best effort
                    pass
        latency_ms = int((time.perf_counter() - started) * 1000)

        model_id = getattr(response, "model", None)
        model_id = str(model_id) if model_id else None
        family = model_family(model_id or "")
        usage = getattr(response, "usage", None)
        tokens = TokenCounts(
            input=int(getattr(usage, "input_tokens", 0) or 0),
            output=int(getattr(usage, "output_tokens", 0) or 0),
        )
        problem = self.cross_family_problem(model_id or "")
        if problem:
            return failed(
                JudgeError(type="cross_family_guard", message=problem),
                model=model_id,
                family=family,
                tokens=tokens,
                latency_ms=latency_ms,
            )
        answers, error = _map_answers(getattr(response, "answers", None), questions)
        if error is not None:
            return failed(
                error,
                model=model_id,
                family=family,
                tokens=tokens,
                latency_ms=latency_ms,
            )
        return SystemOneCall(
            answers=answers,
            model=model_id,
            family=family,
            tokens=tokens,
            latency_ms=latency_ms,
        )


def _to_sdk_questions(sdk: Any, questions: Mapping[str, Question]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, question in questions.items():
        if isinstance(question, ChoiceQuestion):
            out[name] = sdk.Choice(
                instructions=question.instructions,
                criteria=dict(question.criteria),
            )
        elif isinstance(question, NoulQuestion):
            out[name] = sdk.Noul(instructions=question.instructions)
        else:
            raise TypeError(f"unsupported question type for {name!r}")
    return out


def _class_only_error(exc: BaseException, *, phase: str) -> JudgeError:
    """Map an exception to a JudgeError by class name only.

    Exception text is dropped on purpose: SDK errors may echo request context,
    and nothing from the request (or the key) may reach a run log.
    """

    name = type(exc).__name__
    return JudgeError(
        type=name,
        message=f"TypeSafe System One {phase} raised {name}",
    )


def _map_answers(
    raw: Any, questions: Mapping[str, Question]
) -> tuple[Optional[dict[str, Union[ChoiceAnswer, float]]], Optional[JudgeError]]:
    """Map SDK answers onto our types; any missing or malformed answer is an error."""

    if not isinstance(raw, Mapping):
        return None, JudgeError(
            type="schema_error", message="System One response carried no answers"
        )
    mapped: dict[str, Union[ChoiceAnswer, float]] = {}
    for name, question in questions.items():
        answer = raw.get(name)
        if answer is None:
            return None, JudgeError(
                type="schema_error",
                message=f"System One response missing answer {name!r}",
            )
        if isinstance(question, ChoiceQuestion):
            choice = getattr(answer, "choice", None)
            probabilities = getattr(answer, "probabilities", None)
            if not isinstance(choice, str) or not isinstance(probabilities, Mapping):
                return None, JudgeError(
                    type="schema_error",
                    message=f"System One answer {name!r} is not a choice answer",
                )
            clean: dict[str, float] = {}
            for label, value in probabilities.items():
                prob = coerce_confidence(value)
                if prob is None:
                    return None, JudgeError(
                        type="schema_error",
                        message=(
                            f"System One answer {name!r} has a non-numeric "
                            f"probability for {label!r}"
                        ),
                    )
                clean[str(label)] = prob
            mapped[name] = ChoiceAnswer(
                choice=choice,
                confidence=coerce_confidence(getattr(answer, "confidence", None)),
                probabilities=clean,
            )
        else:
            value = coerce_confidence(getattr(answer, "noul", None))
            if value is None:
                return None, JudgeError(
                    type="schema_error",
                    message=f"System One answer {name!r} is not a noul probability",
                )
            mapped[name] = value
    return mapped, None
