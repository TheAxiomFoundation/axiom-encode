"""Jev (TypeSafe) runner: one Choice verdict plus one Noul per defect kind.

Jev answers named questions about a JSON state with calibrated probabilities.
It returns no clause reference, rule path or explanation, so ``findings`` is
always empty and localisation is blank by construction. The provision window
is truncated exactly the way the referee truncates it, so both judge
families read identical text.

Fail closed: a missing SDK or key, any exception from the SDK, a response
without the Choice verdict or without every kind's Noul, a verdict outside
pass/flag, and a served model other than the pinned one all become error
responses, never passes and never partial scores. A model alias ending in
``-latest`` is the one case where the served id may differ; it is recorded.

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
    VERDICT_PASS,
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


def _count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


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
        questions: Optional[dict[str, Any]] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.name = name or "jev"
        self.provision_chars = provision_chars
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._questions = questions

    @property
    def _is_alias(self) -> bool:
        return self.model.endswith("-latest")

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
            return error_response(self.model, "sdk_missing", str(exc)[:500])
        except Exception as exc:  # noqa: BLE001 - e.g. no API key configured
            return error_response(self.model, type(exc).__name__, str(exc)[:500])
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
        usage = getattr(result, "usage", None)
        tokens_in = _count(getattr(usage, "input_tokens", None))
        tokens_out = _count(getattr(usage, "output_tokens", None))
        served_model = str(getattr(result, "model", "") or "")
        provenance = {"served_model": served_model, "requested_model": self.model}

        def fail(error_type: str, message: str) -> JudgeResponse:
            return error_response(
                served_model or self.model,
                error_type,
                message,
                latency_ms=latency_ms,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                raw=provenance,
            )

        if served_model and served_model != self.model and not self._is_alias:
            return fail(
                "served_model_mismatch",
                f"requested {self.model}, served {served_model}",
            )
        answers = getattr(result, "answers", None) or {}
        verdict_answer = answers.get("verdict")
        choice = str(getattr(verdict_answer, "choice", "") or "")
        if choice not in (VERDICT_PASS, VERDICT_FLAG):
            return fail("unrecognized_verdict", f"Choice verdict was {choice!r}")
        probabilities = dict(getattr(verdict_answer, "probabilities", None) or {})
        p_flag = clamp_unit(probabilities.get(VERDICT_FLAG))
        if p_flag is None:
            return fail("malformed_response", "Choice verdict carries no P(flag)")
        kind_scores: dict[str, Optional[float]] = {}
        for kind in DEFECT_KINDS:
            score = clamp_unit(getattr(answers.get(kind), "noul", None))
            if score is None:
                return fail("malformed_response", f"no Noul answer for {kind}")
            kind_scores[kind] = score
        return JudgeResponse(
            verdict=choice,
            verdict_score=p_flag,
            kind_scores=kind_scores,
            kind_score_channels={kind: CHANNEL_NATIVE for kind in DEFECT_KINDS},
            findings=[],
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            model=served_model or self.model,
            raw={
                "confidence": clamp_unit(getattr(verdict_answer, "confidence", None)),
                "probabilities": {k: clamp_unit(v) for k, v in probabilities.items()},
                **provenance,
            },
        )
