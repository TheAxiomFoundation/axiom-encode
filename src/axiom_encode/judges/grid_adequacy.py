"""Stage 2 — grid-adequacy judge (per oracle suite).

Input: the provision text plus the case grid of an oracle suite. Output: named
untested boundaries/regions as structured gaps (e.g. "no case within the 35%
band", "phase-out start untested"), each with a suggested cell. A deterministic
follow-up hook (:func:`gaps_to_cells`) turns those gaps into machine-readable
cell specs a lane or workflow can consume to add coverage.

Advisory: emits an event + suite-report annotation; it does not fail a suite.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from .client import JudgeClient, truncate_provision
from .run_log import (
    Finding,
    JudgeEvent,
    JudgeStage,
    Verdict,
    coerce_confidence,
    error_event,
)

_MAX_GRID_CHARS = 8_000

_SYSTEM = (
    "You are a grid-adequacy judge for oracle test suites over encoded law. "
    "You receive a legal provision and the grid of test cases that exercise its "
    "encoding. Identify the boundaries and regions the provision defines "
    "(thresholds, phase-in/phase-out starts and ends, rate-band edges, caps, "
    "eligibility cliffs) and report which are NOT exercised by any case in the "
    "grid. For each untested region propose one concrete case that would cover "
    "it. Do not restate covered regions."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "confidence": {"type": "number"},
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "region": {"type": "string"},
                    "boundary": {"type": "string"},
                    "clause_ref": {"type": "string"},
                    "suggested_case": {"type": "object", "additionalProperties": True},
                    "explanation": {"type": "string"},
                },
                "required": ["region", "explanation"],
            },
        },
    },
    "required": ["confidence", "gaps"],
}


def _grid_view(case_grid: Any) -> str:
    text = json.dumps(case_grid, default=str, sort_keys=True)
    if len(text) > _MAX_GRID_CHARS:
        text = text[:_MAX_GRID_CHARS] + " …[grid truncated]"
    return text


def build_prompt(provision_text: str, case_grid: Any, *, suite_name: str | None) -> str:
    header = f"Suite: {suite_name}\n\n" if suite_name else ""
    return (
        f"{header}Identify untested boundaries/regions in this suite.\n\n"
        "=== PROVISION TEXT ===\n"
        f"{provision_text}\n\n"
        "=== CASE GRID (JSON list of cases) ===\n"
        f"{_grid_view(case_grid)}\n\n"
        "Return JSON: confidence (0..1) and gaps[]. Each gap: region (short "
        "name, e.g. '35% credit band'), boundary (the specific edge, if any), "
        "clause_ref (provision locus), suggested_case (an input dict that would "
        "land in the region), explanation."
    )


def run(
    provision_text: str,
    case_grid: Any,
    *,
    suite_name: Optional[str] = None,
    run_id: Optional[str] = None,
    client: Optional[JudgeClient] = None,
) -> JudgeEvent:
    """Judge one oracle suite's grid adequacy."""

    client = client or JudgeClient()
    prompt = build_prompt(
        truncate_provision(provision_text, client.provision_chars),
        case_grid,
        suite_name=suite_name,
    )
    call = client.call(system=_SYSTEM, user_prompt=prompt, schema=_SCHEMA)
    if not call.ok:
        return error_event(
            JudgeStage.GRID_ADEQUACY,
            call.error.message if call.error else "unknown judge failure",
            error_type=call.error.type if call.error else "unknown",
            model=call.model,
            generator_model=client.generator_model,
            tokens=call.tokens,
            escalated=call.escalated,
            run_id=run_id,
            subject_ref=suite_name,
        )

    payload = call.payload or {}
    gaps = [g for g in payload.get("gaps", []) if isinstance(g, dict)]
    findings = [
        Finding(
            clause_ref=str(g.get("clause_ref", "")),
            rule_path=str(suite_name or ""),
            kind="untested_region",
            explanation=(
                f"{g.get('region', '')}"
                + (f" [{g.get('boundary')}]" if g.get("boundary") else "")
                + f": {g.get('explanation', '')}"
            ).strip(),
        )
        for g in gaps
    ]
    # Deterministic follow-up hook payload: one cell spec per gap.
    cells = [
        {
            "region": g.get("region"),
            "boundary": g.get("boundary"),
            "clause_ref": g.get("clause_ref"),
            "suggested_case": g.get("suggested_case") or {},
        }
        for g in gaps
    ]
    verdict = Verdict.FLAG if gaps else Verdict.PASS
    confidence = payload.get("confidence")
    return JudgeEvent(
        stage=JudgeStage.GRID_ADEQUACY,
        verdict=verdict,
        confidence=coerce_confidence(confidence),
        advisory=True,
        findings=findings,
        model=call.model,
        generator_model=client.generator_model,
        escalated=call.escalated,
        tokens=call.tokens,
        run_id=run_id,
        subject_ref=suite_name,
        extra={"cells": cells},
    )


def gaps_to_cells(event: JudgeEvent) -> list[dict[str, Any]]:
    """Deterministic follow-up hook: the cell specs a lane/workflow adds.

    Consumes only the structured ``extra["cells"]`` payload; a lane can feed
    these straight into suite generation. Returns an empty list for a pass or an
    error event (nothing to add, and an error must not masquerade as coverage).
    """

    if event.stage != JudgeStage.GRID_ADEQUACY or event.is_error:
        return []
    return list(event.extra.get("cells", []))
