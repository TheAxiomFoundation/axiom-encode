"""Stage 3 — disposition referee (per new disposition).

A *disposition* records a causal claim that explains a residual — the gap
between the engine's value and an oracle's value on some records ("the $42
difference is because we round the standard deduction before applying the
rate"). Before a disposition is trusted, this referee:

1. Runs the sampled arithmetic deterministically on >=3 sampled records (pure
   Python, no model): does ``oracle_value - engine_value`` actually reproduce
   the claimed residual on each?
2. Cross-family reads the claim (Claude): does the stated cause plausibly
   produce that residual given the records' arithmetic?

A disposition passes only when both hold on at least three records. Advisory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .client import JudgeClient
from .run_log import (
    Finding,
    JudgeEvent,
    JudgeStage,
    Verdict,
    coerce_confidence,
    error_event,
)

MIN_RECORDS = 3
DEFAULT_TOLERANCE = 0.01  # absolute, in the residual's units


@dataclass
class Disposition:
    """A causal claim about a residual, with sampled records to check it on.

    Each record must carry ``engine_value`` and ``oracle_value``; the observed
    residual is ``oracle_value - engine_value``. Optional ``record_id`` labels it.
    """

    disposition_id: str
    claim: str
    residual: float
    variable: Optional[str] = None
    records: list[dict[str, Any]] = field(default_factory=list)


def _observed_residual(record: dict[str, Any]) -> Optional[float]:
    try:
        return float(record["oracle_value"]) - float(record["engine_value"])
    except (KeyError, TypeError, ValueError):
        return None


def check_arithmetic(
    disposition: Disposition, *, tolerance: float = DEFAULT_TOLERANCE
) -> dict[str, Any]:
    """Deterministically check that the claimed residual reproduces on records.

    Returns a machine-readable report: per-record observed residual and whether
    it matches the claim within ``tolerance``, plus the count reproduced.
    """

    per_record = []
    reproduced = 0
    for record in disposition.records:
        observed = _observed_residual(record)
        matches = (
            observed is not None and abs(observed - disposition.residual) <= tolerance
        )
        if matches:
            reproduced += 1
        per_record.append(
            {
                "record_id": record.get("record_id"),
                "engine_value": record.get("engine_value"),
                "oracle_value": record.get("oracle_value"),
                "observed_residual": observed,
                "claimed_residual": disposition.residual,
                "matches": bool(matches),
            }
        )
    return {
        "records": per_record,
        "n_records": len(disposition.records),
        "reproduced": reproduced,
        "tolerance": tolerance,
        "min_records": MIN_RECORDS,
        "arithmetic_ok": len(disposition.records) >= MIN_RECORDS
        and reproduced >= MIN_RECORDS,
    }


_SYSTEM = (
    "You are a disposition referee for encoded-law microsimulation. A "
    "disposition is a causal claim explaining why an engine value differs from "
    "an oracle value on sampled records. You are given the claim, the claimed "
    "residual, and the per-record arithmetic already computed deterministically. "
    "Judge only whether the stated cause plausibly produces that residual given "
    "the arithmetic — not whether the residual is acceptable. Be skeptical of "
    "claims the arithmetic does not support."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "consistent": {"type": "boolean"},
        "confidence": {"type": "number"},
        "explanation": {"type": "string"},
    },
    "required": ["consistent", "confidence", "explanation"],
}


def build_prompt(disposition: Disposition, arithmetic: dict[str, Any]) -> str:
    return (
        f"Disposition id: {disposition.disposition_id}\n"
        f"Variable: {disposition.variable or '(unspecified)'}\n"
        f"Claimed residual (oracle - engine): {disposition.residual}\n\n"
        f"Causal claim:\n{disposition.claim}\n\n"
        "Per-record arithmetic (already computed):\n"
        f"{json.dumps(arithmetic['records'], default=str, indent=2)}\n\n"
        f"Reproduced on {arithmetic['reproduced']} of {arithmetic['n_records']} "
        "records.\n\n"
        "Return JSON: consistent (does the claim explain this residual given the "
        "arithmetic?), confidence (0..1), explanation."
    )


def run(
    disposition: Disposition,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    run_id: Optional[str] = None,
    client: Optional[JudgeClient] = None,
) -> JudgeEvent:
    """Referee one disposition; deterministic arithmetic first, then claim read."""

    client = client or JudgeClient()
    arithmetic = check_arithmetic(disposition, tolerance=tolerance)

    if arithmetic["n_records"] < MIN_RECORDS:
        return JudgeEvent(
            stage=JudgeStage.DISPOSITION,
            verdict=Verdict.FLAG,
            confidence=1.0,
            advisory=True,
            findings=[
                Finding(
                    clause_ref=disposition.variable or "",
                    rule_path=disposition.disposition_id,
                    kind="insufficient_records",
                    explanation=(
                        f"only {arithmetic['n_records']} sampled record(s); the "
                        f"referee requires at least {MIN_RECORDS}"
                    ),
                )
            ],
            model=None,
            generator_model=client.generator_model,
            run_id=run_id,
            subject_ref=disposition.disposition_id,
            extra={"arithmetic": arithmetic},
        )

    call = client.call(
        system=_SYSTEM,
        user_prompt=build_prompt(disposition, arithmetic),
        schema=_SCHEMA,
    )
    if not call.ok:
        ev = error_event(
            JudgeStage.DISPOSITION,
            call.error.message if call.error else "unknown judge failure",
            error_type=call.error.type if call.error else "unknown",
            model=call.model,
            generator_model=client.generator_model,
            tokens=call.tokens,
            escalated=call.escalated,
            run_id=run_id,
            subject_ref=disposition.disposition_id,
        )
        ev.extra["arithmetic"] = arithmetic
        return ev

    payload = call.payload or {}
    claim_consistent = bool(payload.get("consistent"))
    arithmetic_ok = bool(arithmetic["arithmetic_ok"])
    passed = arithmetic_ok and claim_consistent

    findings: list[Finding] = []
    if not passed:
        reasons = []
        if not arithmetic_ok:
            reasons.append(
                f"arithmetic reproduces the residual on only "
                f"{arithmetic['reproduced']}/{arithmetic['n_records']} records"
            )
        if not claim_consistent:
            reasons.append(
                "claim not consistent with the arithmetic: "
                + str(payload.get("explanation", ""))
            )
        findings.append(
            Finding(
                clause_ref=disposition.variable or "",
                rule_path=disposition.disposition_id,
                kind="claim_mismatch",
                explanation="; ".join(reasons),
            )
        )

    confidence = payload.get("confidence")
    return JudgeEvent(
        stage=JudgeStage.DISPOSITION,
        verdict=Verdict.PASS if passed else Verdict.FLAG,
        confidence=coerce_confidence(confidence),
        advisory=True,
        findings=findings,
        model=call.model,
        generator_model=client.generator_model,
        escalated=call.escalated,
        tokens=call.tokens,
        run_id=run_id,
        subject_ref=disposition.disposition_id,
        extra={"arithmetic": arithmetic, "claim_consistent": claim_consistent},
    )
