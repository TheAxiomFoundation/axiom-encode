"""Incumbent referee: the statutory-fidelity prompt and schema, one model.

This runner imports (never copies) the referee's system prompt, questions,
JSON schema and provision truncation from ``axiom_encode.judges`` and sends
them through the repo's own :class:`JudgeClient`. Differences from the
production wiring, all deliberate:

* no escalation — a benchmark of judges scores one model per runner, so the
  call passes ``escalate=False`` and the escalation model is pinned equal to
  the judged model;
* retry pacing is shorter than the production 90 s, and configurable;
* ``max_tokens`` is passed explicitly and recorded, so an
  ``AXIOM_JUDGE_MAX_TOKENS`` in the environment cannot silently change a run;
* the production cross-family guard is a pipeline policy ("a judge must not
  share the generator's family"), not a measurement rule. The client is built
  with the repo's declared generator so the guard is satisfied, and the case's
  true generator plus whether it shares the judge's family are recorded on
  every row instead of refusing the case.

The runner calls ``JudgeClient.call`` itself rather than
``statutory_fidelity.run`` so it can record the model's *raw* verdict and
ignore finding kinds outside the referee's taxonomy. The verdict it reports
and scores is the production verdict, derived exactly as
``statutory_fidelity.run`` derives it (a test asserts the two agree payload
for payload): a raw ``pass`` that still lists findings is a flag, because the
incumbent's own contract says a faithful artifact gets an empty findings
list. Such an answer is self-contradictory and its confidence is ambiguous;
the row records ``raw_verdict`` and ``verdict_coerced_by_findings`` and the
board counts coercions per judge rather than guessing a probability.

The digests in :meth:`RefereeRunner.identity` cover the stage's system prompt
plus user template, and the JSON schema. ``JudgeClient`` appends a fixed
"respond with a single JSON object" instruction to the system prompt; that
suffix lives inside the client and is not part of the digest.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from axiom_encode.constants import DEFAULT_OPENAI_MODEL
from axiom_encode.judges import statutory_fidelity
from axiom_encode.judges.client import (
    DEFAULT_MAX_TOKENS,
    JudgeClient,
    model_family,
    truncate_provision,
)
from axiom_encode.judges.run_log import coerce_confidence

from .. import DEFECT_KINDS
from ..canonical import canonical_json_sha256, text_sha256
from ..cases import VerifierCase
from .base import (
    CHANNEL_NATIVE,
    CHANNEL_VERDICT_FALLBACK,
    VERDICT_FLAG,
    VERDICT_PASS,
    JudgeResponse,
    error_response,
)

# How the referee's four finding kinds map onto the planted-defect taxonomy.
# Kinds with no referee question fall back to the verdict score and are
# marked as such in every result row. Only kinds the model actually wrote
# count: production clamps an unrecognised kind to ``untraceable_branch``,
# which must not be credited as a polarity detection.
REFEREE_KIND_MAP: dict[str, tuple[str, ...]] = {
    "amount_changed": ("amount_mismatch",),
    "boundary_flipped": ("boundary_direction",),
    "conjunct_dropped": ("unrepresented_clause",),
    "polarity_swapped": ("untraceable_branch", "unrepresented_clause"),
    "date_or_period_wrong": (),
    "entity_wrong": (),
}

assert tuple(REFEREE_KIND_MAP) == DEFECT_KINDS

_REFEREE_FINDING_KINDS = frozenset(statutory_fidelity._KINDS)


def referee_prompt_sha256() -> str:
    """Digest of the stage prompt (system prompt + user template)."""

    sample = statutory_fidelity.build_prompt("P", "A", citation="C")
    return text_sha256(statutory_fidelity._SYSTEM + "\n" + sample)


def referee_schema_sha256() -> str:
    return canonical_json_sha256(statutory_fidelity._SCHEMA)


def verdict_score_from(verdict: str, confidence: float) -> float:
    """P(defective) from the production verdict and the model's confidence.

    The referee reports ``confidence`` as P(its verdict is correct), so a flag
    at confidence c is P(defective) = c and a pass at confidence c is 1 - c.
    """

    conf = min(1.0, max(0.0, float(confidence)))
    return round(conf if verdict == VERDICT_FLAG else 1.0 - conf, 6)


class RefereeRunner:
    family = "referee"
    supports_localization = True

    def __init__(
        self,
        model: str,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        provision_chars: int = 24_000,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_attempts: int = 4,
        retry_seconds: float = 15.0,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.model = model
        self.name = name or model
        self.provision_chars = provision_chars
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.retry_seconds = retry_seconds
        self._api_key = api_key
        self._client_factory = client_factory or self._default_client
        self._client: Any = None

    def _default_client(self) -> JudgeClient:
        return JudgeClient(
            model=self.model,
            escalation_model=self.model,
            api_key=self._api_key,
            generator_model=DEFAULT_OPENAI_MODEL,
            max_tokens=self.max_tokens,
            provision_chars=self.provision_chars,
            max_attempts=self.max_attempts,
            retry_seconds=self.retry_seconds,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def identity(self) -> dict[str, Any]:
        sdk_version = None
        try:
            import anthropic

            sdk_version = getattr(anthropic, "__version__", None)
        except ImportError:
            pass
        return {
            "family": self.family,
            "model": self.model,
            "prompt": "axiom_encode.judges.statutory_fidelity",
            "judge_prompt_sha256": referee_prompt_sha256(),
            "judge_schema_sha256": referee_schema_sha256(),
            "escalation": False,
            "max_tokens": self.max_tokens,
            "provision_chars": self.provision_chars,
            "supports_localization": self.supports_localization,
            "anthropic_sdk_version": sdk_version,
            "cross_family_guard": (
                "satisfied with the repo's declared generator; each row records "
                "the case's true generator and same-family status"
            ),
            "finding_kind_map": {k: list(v) for k, v in REFEREE_KIND_MAP.items()},
        }

    def judge(self, case: VerifierCase) -> JudgeResponse:
        generator = case.origin.get("generator_model")
        generator = str(generator) if generator else None
        same_family: Optional[bool] = None
        if generator and model_family(generator) != "unknown":
            same_family = model_family(generator) == model_family(self.model)
        provenance = {
            "generator_model": generator,
            "same_family_as_generator": same_family,
        }

        started = time.perf_counter()
        try:
            client = self._get_client()
            prompt = statutory_fidelity.build_prompt(
                truncate_provision(case.provision_text, client.provision_chars),
                case.artifact_text,
                citation=case.citation or None,
            )
            call = client.call(
                system=statutory_fidelity._SYSTEM,
                user_prompt=prompt,
                schema=statutory_fidelity._SCHEMA,
                escalate=False,
            )
        except Exception as exc:  # noqa: BLE001 - fail closed, never a pass
            return error_response(
                self.model,
                type(exc).__name__,
                str(exc)[:500],
                latency_ms=int((time.perf_counter() - started) * 1000),
                raw=provenance,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        tokens_in = int(call.tokens.input)
        tokens_out = int(call.tokens.output)

        def fail(error_type: str, message: str) -> JudgeResponse:
            return error_response(
                call.model or self.model,
                error_type,
                message,
                latency_ms=latency_ms,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                raw=provenance,
            )

        if not call.ok:
            err = call.error
            return fail(
                err.type if err else "unknown",
                err.message if err else "unknown judge failure",
            )
        payload = call.payload or {}
        raw_findings = [f for f in payload.get("findings", []) if isinstance(f, dict)]
        raw_verdict = str(payload.get("verdict", "")).lower()
        if raw_verdict not in (VERDICT_PASS, VERDICT_FLAG):
            return fail("unrecognized_verdict", f"unrecognized verdict {raw_verdict!r}")
        confidence = coerce_confidence(payload.get("confidence"))
        if confidence is None:
            return fail(
                "confidence_unparseable",
                f"confidence {payload.get('confidence')!r} is not a number",
            )
        # Production verdict, exactly as statutory_fidelity.run derives it.
        verdict = (
            VERDICT_FLAG
            if (raw_verdict == VERDICT_FLAG or raw_findings)
            else VERDICT_PASS
        )
        findings = [
            {
                "kind": str(f.get("kind", "")),
                "rule_path": str(f.get("rule_path", "")),
                "clause_ref": str(f.get("clause_ref", "")),
                "explanation": str(f.get("explanation", "")),
            }
            for f in raw_findings
        ]
        named = {f["kind"] for f in findings if f["kind"] in _REFEREE_FINDING_KINDS}
        unknown_kinds = sorted(
            {f["kind"] for f in findings if f["kind"] not in _REFEREE_FINDING_KINDS}
        )
        score = verdict_score_from(verdict, confidence)
        kind_scores: dict[str, Optional[float]] = {}
        channels: dict[str, str] = {}
        for kind, referee_kinds in REFEREE_KIND_MAP.items():
            if referee_kinds:
                kind_scores[kind] = 1.0 if named.intersection(referee_kinds) else 0.0
                channels[kind] = CHANNEL_NATIVE
            else:
                kind_scores[kind] = score
                channels[kind] = CHANNEL_VERDICT_FALLBACK
        return JudgeResponse(
            verdict=verdict,
            verdict_score=score,
            kind_scores=kind_scores,
            kind_score_channels=channels,
            findings=findings,
            latency_ms=latency_ms,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            model=call.model or self.model,
            raw={
                "raw_verdict": raw_verdict,
                "verdict_coerced_by_findings": verdict != raw_verdict,
                "confidence": confidence,
                "escalated": bool(call.escalated),
                "n_findings": len(findings),
                "unknown_finding_kinds": unknown_kinds,
                **provenance,
            },
        )
