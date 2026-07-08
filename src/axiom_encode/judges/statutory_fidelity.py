"""Stage 1 — statutory-fidelity referee (per generation, post-gates pre-apply).

Cross-family read of a generated RuleSpec artifact against the pinned provision
text. Structured questions:

* Does every rule branch trace to provision language?
* Is any provision clause unrepresented?
* Are boundary inequalities the right direction ("exceeding" vs "not less than")?
* Do parameterized amounts match the verbatim source atoms?

Wired as advisory first: a ``needs-review`` label + run-log event, NOT a hard
gate. Promotion to a hard gate is a later decision on the calibration evidence
(see :mod:`~axiom_encode.judges.calibration`).
"""

from __future__ import annotations

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

_KINDS = [
    "unrepresented_clause",
    "untraceable_branch",
    "boundary_direction",
    "amount_mismatch",
]

_SYSTEM = (
    "You are a statutory-fidelity referee for encoded law. You receive the "
    "verbatim text of a legal provision and a machine-generated RuleSpec YAML "
    "artifact claimed to encode it. You judge fidelity only — not style. Be "
    "precise and conservative: a finding must name a specific clause and a "
    "specific place in the artifact. Do not invent provision text. If the "
    "artifact is faithful, return an empty findings list and verdict 'pass'."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "flag"]},
        "confidence": {"type": "number"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "clause_ref": {"type": "string"},
                    "rule_path": {"type": "string"},
                    "kind": {"type": "string", "enum": _KINDS},
                    "explanation": {"type": "string"},
                },
                "required": ["clause_ref", "rule_path", "kind", "explanation"],
            },
        },
    },
    "required": ["verdict", "confidence", "findings"],
}

_QUESTIONS = (
    "1. Does every branch / condition in the artifact trace to specific "
    "provision language? Flag branches with no source basis (untraceable_branch).\n"
    "2. Is any operative provision clause left unrepresented by the artifact? "
    "Flag it (unrepresented_clause).\n"
    "3. Are boundary inequalities the correct direction? 'exceeding X' is a "
    "strict '> X'; 'not less than X' is '>= X'; 'up to X' includes X. Flag "
    "reversed or off-by-boundary comparisons (boundary_direction).\n"
    "4. Do parameterized amounts, rates, and thresholds match the verbatim "
    "source atoms exactly? Flag any value the artifact states that the "
    "provision does not (amount_mismatch)."
)


def build_prompt(
    provision_text: str, generated_rule: str, *, citation: str | None
) -> str:
    header = f"Citation: {citation}\n\n" if citation else ""
    return (
        f"{header}Answer these fidelity questions about the artifact below.\n\n"
        f"{_QUESTIONS}\n\n"
        "=== PROVISION TEXT (verbatim) ===\n"
        f"{provision_text}\n\n"
        "=== GENERATED RULESPEC ARTIFACT ===\n"
        f"{generated_rule}\n\n"
        "Return JSON: verdict ('pass' or 'flag'), confidence (0..1 that your "
        "verdict is correct), and findings[]. Each finding: clause_ref (the "
        "provision locus), rule_path (the artifact locus, e.g. a variable or "
        "key path), kind (one of "
        f"{_KINDS}), explanation."
    )


def _clamp_kind(kind: str) -> str:
    return kind if kind in _KINDS else "untraceable_branch"


def run(
    provision_text: str,
    generated_rule: str,
    *,
    citation: Optional[str] = None,
    rule_path: Optional[str] = None,
    run_id: Optional[str] = None,
    client: Optional[JudgeClient] = None,
) -> JudgeEvent:
    """Judge one generation and return a structured :class:`JudgeEvent`."""

    client = client or JudgeClient()
    subject = citation or rule_path
    prompt = build_prompt(
        truncate_provision(provision_text, client.provision_chars),
        generated_rule,
        citation=citation,
    )
    call = client.call(system=_SYSTEM, user_prompt=prompt, schema=_SCHEMA)
    if not call.ok:
        return error_event(
            JudgeStage.STATUTORY_FIDELITY,
            call.error.message if call.error else "unknown judge failure",
            error_type=call.error.type if call.error else "unknown",
            model=call.model,
            generator_model=client.generator_model,
            tokens=call.tokens,
            escalated=call.escalated,
            run_id=run_id,
            subject_ref=subject,
        )

    payload = call.payload or {}
    findings = [
        Finding(
            clause_ref=str(f.get("clause_ref", "")),
            rule_path=str(f.get("rule_path", rule_path or "")),
            kind=_clamp_kind(str(f.get("kind", ""))),
            explanation=str(f.get("explanation", "")),
        )
        for f in payload.get("findings", [])
        if isinstance(f, dict)
    ]
    raw_verdict = str(payload.get("verdict", "")).lower()
    if raw_verdict not in ("pass", "flag"):
        # An unrecognized (or absent) verdict is not a usable answer. Fail closed
        # — never let it default to PASS (the fail-open the module bans). Only
        # reachable on the prompt-guided fallback path; the json_schema enum
        # constrains the happy path.
        return error_event(
            JudgeStage.STATUTORY_FIDELITY,
            f"unrecognized verdict {raw_verdict!r} from judge",
            error_type="unrecognized_verdict",
            model=call.model,
            generator_model=client.generator_model,
            tokens=call.tokens,
            escalated=call.escalated,
            run_id=run_id,
            subject_ref=subject,
        )
    verdict = Verdict.FLAG if (raw_verdict == "flag" or findings) else Verdict.PASS
    return JudgeEvent(
        stage=JudgeStage.STATUTORY_FIDELITY,
        verdict=verdict,
        confidence=coerce_confidence(payload.get("confidence")),
        advisory=True,
        findings=findings,
        model=call.model,
        generator_model=client.generator_model,
        escalated=call.escalated,
        tokens=call.tokens,
        run_id=run_id,
        subject_ref=subject,
    )


def needs_review_label(event: JudgeEvent) -> Optional[str]:
    """Advisory wiring: the label to apply when the referee flags a generation.

    Returns ``None`` for a pass. An *error* event also returns the label — a
    judge that could not run is itself a reason for a human to look, and must
    never be dropped as if it were a pass.
    """

    if event.stage != JudgeStage.STATUTORY_FIDELITY:
        return None
    if event.verdict in (Verdict.FLAG, Verdict.ERROR):
        return "needs-review"
    return None
