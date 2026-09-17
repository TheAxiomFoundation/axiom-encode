"""Jev (TypeSafe) runner: one Choice verdict plus one Noul per defect kind.

Jev answers named questions about a JSON state with calibrated probabilities.
It returns no clause reference, rule path or explanation, so ``findings`` is
always empty and localisation is blank by construction. The provision window
is truncated exactly the way the referee truncates it, so both judge
families read identical text.

Requires ``typesafe-sdk`` and ``TYPESAFE_API_KEY`` (read by the SDK from the
environment; this module never prints or stores it).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from axiom_encode.judges.client import truncate_provision

from .. import DEFECT_KINDS
from ..canonical import canonical_json_sha256
from ..cases import VerifierCase
from .base import (
    CHANNEL_NATIVE,
    VERDICT_FLAG,
    JudgeResponse,
    clamp_unit,
    error_response,
)

DEFAULT_JEV_MODEL = "jev-1.13.0"

VERDICT_QUESTION = {
    "instructions": (
        "Judge whether the generated RuleSpec artifact faithfully encodes the "
        "provision. Judge fidelity only, not style."
    ),
    "criteria": {
        "pass": (
            "Every branch traces to provision language, no operative clause is "
            "unrepresented, boundary inequalities point the right direction, "
            "every amount, rate and threshold matches the source, effective "
            "dates and periods match the source, and each rule applies to the "
            "entity the provision addresses."
        ),
        "flag": (
            "At least one branch has no source basis, or an operative clause is "
            "unrepresented, or a boundary inequality is reversed or off by its "
            "boundary, or an amount, rate, threshold, effective date, period or "
            "entity does not match the source."
        ),
    },
}

KIND_NOULS: dict[str, str] = {
    "amount_changed": (
        "At least one amount, rate or threshold stated in the artifact does not "
        "match the provision text."
    ),
    "boundary_flipped": (
        "At least one boundary inequality in the artifact is reversed or off by "
        "its boundary relative to the provision (for example > where the "
        "provision means at least, or <= where it means less than)."
    ),
    "conjunct_dropped": (
        "At least one condition the provision requires is missing from the "
        "artifact's conditions."
    ),
    "polarity_swapped": (
        "The artifact joins conditions with 'or' where the provision requires "
        "all of them, or with 'and' where the provision requires any of them."
    ),
    "date_or_period_wrong": (
        "An effective date or a rule period in the artifact does not match the "
        "date or period the provision states."
    ),
    "entity_wrong": (
        "At least one rule in the artifact applies to a different kind of "
        "entity (person, household, tax unit, employer, ...) than the one the "
        "provision addresses."
    ),
}

assert tuple(KIND_NOULS) == DEFECT_KINDS


def build_state(case: VerifierCase, provision_chars: int) -> dict[str, str]:
    return {
        "citation": case.citation,
        "provision_text_verbatim": truncate_provision(
            case.provision_text, provision_chars
        ),
        "generated_rulespec_artifact": case.artifact_text,
    }


def build_questions() -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul

    questions: dict[str, Any] = {
        "verdict": Choice(
            instructions=VERDICT_QUESTION["instructions"],
            criteria=dict(VERDICT_QUESTION["criteria"]),
        )
    }
    for kind in DEFECT_KINDS:
        questions[kind] = Noul(instructions=KIND_NOULS[kind])
    return questions


def question_set_sha256() -> str:
    return canonical_json_sha256(
        {"verdict": VERDICT_QUESTION, "nouls": KIND_NOULS, "kinds": list(DEFECT_KINDS)}
    )


class JevRunner:
    family = "jev"
    supports_localization = False

    def __init__(
        self,
        model: str = DEFAULT_JEV_MODEL,
        *,
        name: Optional[str] = None,
        provision_chars: int = 24_000,
        client: Any = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.name = name or "jev"
        self.provision_chars = provision_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._questions: Optional[dict[str, Any]] = None

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self._client = TypeSafeClient(
                model=self.model,
                timeout=self.timeout,
                retry=RetryPolicy(max_retries=self.max_retries, timeout=None),
            )
        return self._client

    def identity(self) -> dict[str, Any]:
        sdk_version = None
        try:
            import typesafe_sdk

            sdk_version = getattr(typesafe_sdk, "__version__", None)
        except ImportError:
            pass
        return {
            "family": self.family,
            "model": self.model,
            "sdk": "typesafe-sdk",
            "sdk_version": sdk_version,
            "question_set_sha256": question_set_sha256(),
            "provision_chars": self.provision_chars,
            "supports_localization": self.supports_localization,
            "kind_channel": "one Noul per defect kind",
        }

    def judge(self, case: VerifierCase) -> JudgeResponse:
        try:
            client = self._get_client()
            if self._questions is None:
                self._questions = build_questions()
        except ImportError as exc:
            return error_response(self.model, "sdk_missing", str(exc))
        state = build_state(case, self.provision_chars)
        started = time.perf_counter()
        try:
            result = client.system_one(state=state, questions=self._questions)
        except Exception as exc:  # noqa: BLE001 - fail closed, never a pass
            latency_ms = int((time.perf_counter() - started) * 1000)
            return error_response(
                self.model, type(exc).__name__, str(exc)[:500], latency_ms=latency_ms
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        answers = getattr(result, "answers", {}) or {}
        verdict_answer = answers.get("verdict")
        if verdict_answer is None or not hasattr(verdict_answer, "choice"):
            return error_response(
                self.model, "malformed_response", "no Choice verdict in answers"
            )
        choice = str(verdict_answer.choice)
        probabilities = dict(getattr(verdict_answer, "probabilities", {}) or {})
        p_flag = clamp_unit(probabilities.get(VERDICT_FLAG))
        if p_flag is None:
            p_flag = 1.0 if choice == VERDICT_FLAG else 0.0
        kind_scores: dict[str, Optional[float]] = {}
        for kind in DEFECT_KINDS:
            answer = answers.get(kind)
            kind_scores[kind] = clamp_unit(getattr(answer, "noul", None))
        usage = getattr(result, "usage", None)
        served_model = str(getattr(result, "model", "") or self.model)
        return JudgeResponse(
            verdict=choice if choice in ("pass", "flag") else "error",
            verdict_score=p_flag,
            kind_scores=kind_scores,
            kind_score_channels={kind: CHANNEL_NATIVE for kind in DEFECT_KINDS},
            findings=[],
            latency_ms=latency_ms,
            tokens_input=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_output=int(getattr(usage, "output_tokens", 0) or 0),
            model=served_model,
            error=(
                None
                if choice in ("pass", "flag")
                else {"type": "unrecognized_verdict", "message": choice}
            ),
            raw={
                "confidence": clamp_unit(getattr(verdict_answer, "confidence", None)),
                "probabilities": {k: clamp_unit(v) for k, v in probabilities.items()},
                "served_model": served_model,
                "requested_model": self.model,
            },
        )
