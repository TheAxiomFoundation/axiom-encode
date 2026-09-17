"""Incumbent referee: the statutory-fidelity prompt and schema, one model.

This runner imports (never copies) the referee's system prompt, questions,
JSON schema and provision truncation from ``axiom_encode.judges`` and calls
the referee through the repo's own :class:`JudgeClient`. Two deliberate
differences from production wiring:

* no escalation — a benchmark of judges scores one model per runner, so the
  escalation model is pinned equal to the judged model;
* retry pacing is shorter than the production 90 s, and configurable.

The prompt's sha256 (system + questions) and the schema's sha256 are part of
the runner identity so a prompt change under the judges package shows up as
a different runner, not a silently different board.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from axiom_encode.judges import statutory_fidelity
from axiom_encode.judges.client import JudgeClient
from axiom_encode.judges.run_log import Verdict

from .. import DEFECT_KINDS
from ..canonical import text_sha256
from ..cases import VerifierCase
from .base import (
    CHANNEL_NATIVE,
    CHANNEL_VERDICT_FALLBACK,
    VERDICT_FLAG,
    VERDICT_PASS,
    JudgeResponse,
    clamp_unit,
    error_response,
)

# How the referee's four finding kinds map onto the planted-defect taxonomy.
# Kinds with no referee question fall back to the verdict score and are
# marked as such in every result row.
REFEREE_KIND_MAP: dict[str, tuple[str, ...]] = {
    "amount_changed": ("amount_mismatch",),
    "boundary_flipped": ("boundary_direction",),
    "conjunct_dropped": ("unrepresented_clause",),
    "polarity_swapped": ("untraceable_branch", "unrepresented_clause"),
    "date_or_period_wrong": (),
    "entity_wrong": (),
}

assert tuple(REFEREE_KIND_MAP) == DEFECT_KINDS

DEFAULT_GENERATOR_MODEL = "gpt-5.5"


def referee_prompt_sha256() -> str:
    """Digest of the exact prompt text the referee sends (system + questions)."""

    sample = statutory_fidelity.build_prompt("P", "A", citation="C")
    return text_sha256(statutory_fidelity._SYSTEM + "\n" + sample)


def referee_schema_sha256() -> str:
    from ..canonical import canonical_json_sha256

    return canonical_json_sha256(statutory_fidelity._SCHEMA)


def verdict_score_from(verdict: str, confidence: Optional[float]) -> float:
    """P(defective) from a binary verdict plus its self-reported confidence.

    The referee reports ``confidence`` as P(verdict is correct), so a flag at
    confidence c is P(defective) = c and a pass at confidence c is 1 - c. A
    missing confidence is treated as 1.0 (a bare verdict).
    """

    conf = clamp_unit(confidence)
    if conf is None:
        conf = 1.0
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
        max_attempts: int = 4,
        retry_seconds: float = 15.0,
        client_factory: Optional[Callable[[str], JudgeClient]] = None,
    ) -> None:
        self.model = model
        self.name = name or model
        self.provision_chars = provision_chars
        self.max_attempts = max_attempts
        self.retry_seconds = retry_seconds
        self._api_key = api_key
        self._client_factory = client_factory or self._default_client
        self._clients: dict[str, JudgeClient] = {}

    def _default_client(self, generator_model: str) -> JudgeClient:
        return JudgeClient(
            model=self.model,
            escalation_model=self.model,
            api_key=self._api_key,
            generator_model=generator_model,
            provision_chars=self.provision_chars,
            max_attempts=self.max_attempts,
            retry_seconds=self.retry_seconds,
        )

    def _client_for(self, case: VerifierCase) -> JudgeClient:
        generator = str(case.origin.get("generator_model") or DEFAULT_GENERATOR_MODEL)
        client = self._clients.get(generator)
        if client is None:
            client = self._client_factory(generator)
            self._clients[generator] = client
        return client

    def identity(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "model": self.model,
            "prompt": "axiom_encode.judges.statutory_fidelity",
            "judge_prompt_sha256": referee_prompt_sha256(),
            "judge_schema_sha256": referee_schema_sha256(),
            "escalation": False,
            "provision_chars": self.provision_chars,
            "supports_localization": self.supports_localization,
            "finding_kind_map": {k: list(v) for k, v in REFEREE_KIND_MAP.items()},
        }

    def judge(self, case: VerifierCase) -> JudgeResponse:
        client = self._client_for(case)
        started = time.perf_counter()
        event = statutory_fidelity.run(
            case.provision_text,
            case.artifact_text,
            citation=case.citation or None,
            run_id=case.case_id,
            client=client,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if event.verdict == Verdict.ERROR or event.judge_error is not None:
            err = event.judge_error
            response = error_response(
                event.model or self.model,
                err.type if err else "judge_error",
                err.message if err else "referee returned an error verdict",
                latency_ms=latency_ms,
            )
            response.tokens_input = event.tokens.input
            response.tokens_output = event.tokens.output
            return response
        verdict = VERDICT_FLAG if event.verdict == Verdict.FLAG else VERDICT_PASS
        findings = [
            {
                "kind": f.kind,
                "rule_path": f.rule_path,
                "clause_ref": f.clause_ref,
                "explanation": f.explanation,
            }
            for f in event.findings
        ]
        named = {f["kind"] for f in findings}
        score = verdict_score_from(verdict, event.confidence)
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
            tokens_input=event.tokens.input,
            tokens_output=event.tokens.output,
            model=event.model or self.model,
            raw={
                "confidence": event.confidence,
                "escalated": event.escalated,
                "n_findings": len(findings),
            },
        )
