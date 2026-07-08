"""Stage 4 — worklist pre-classifier (dispatcher-side, before generation).

Classifies each pending worklist entry so the dispatcher does not burn a
generation the compile gate will reject:

* ``self_contained`` — route to generation.
* ``amendment_act`` (#1079 class) — an amend-in-place instruction; skip.
* ``xref_heavy`` (#1058 class) — reference-dominated, self-import risk; skip.
* ``needs_container`` — cannot stand alone without a parent container; skip.

Cheap by design: deterministic heuristics decide the clear cases; only ambiguous
entries pay for a Claude arbiter call. The load-bearing guarantee: an entry is
**never dropped silently**. Every entry returns a result with a route; on any
uncertainty or judge error the entry is marked skip-with-reason, never dropped
and never silently sent to generation.

Wired into ``bulk-encode.yml`` as a cheap pre-step.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .client import JudgeClient, truncate_provision
from .run_log import (
    Finding,
    JudgeEvent,
    JudgeStage,
    TokenCounts,
    Verdict,
    coerce_confidence,
)


class WorklistClass(str, Enum):
    SELF_CONTAINED = "self_contained"
    AMENDMENT_ACT = "amendment_act"
    XREF_HEAVY = "xref_heavy"
    NEEDS_CONTAINER = "needs_container"


GENERATE = "generate"
SKIP = "skip"

# Amendment-instruction markers (#1079). Deliberately excludes the past
# participle "as amended", which merely describes existing law.
_AMENDMENT_MARKERS = [
    re.compile(r"\bis(?:\s+hereby)?\s+amended\b", re.I),
    re.compile(r"\bare\s+(?:hereby\s+)?amended\b", re.I),
    re.compile(r"\bamended\s+by\b", re.I),
    re.compile(r"\bby\s+(?:inserting|adding|striking|redesignating|repealing)\b", re.I),
    re.compile(r"\bstriking\s+(?:out\s+)?(?:the\s+)?", re.I),
    re.compile(r"\bis\s+repealed\b", re.I),
    re.compile(r"\bamended\s+to\s+read\b", re.I),
    re.compile(
        r"\bthe\s+following\s+new\s+(?:section|subsection|paragraph|subparagraph)\b",
        re.I,
    ),
    re.compile(r"\binserting\s+after\b", re.I),
]

_XREF_MARKERS = [
    re.compile(r"\bsections?\s+\d+", re.I),
    re.compile(r"§+\s*\d+"),
    re.compile(r"\bparagraphs?\s+\(\w+\)", re.I),
    re.compile(r"\bsubsections?\s+\(\w+\)", re.I),
    re.compile(r"\bsubparagraphs?\s+\(\w+\)", re.I),
    re.compile(r"\bclauses?\s+\(\w+\)", re.I),
    re.compile(r"\bU\.?\s?S\.?\s?C\.?", re.I),
    re.compile(r"\btitle\s+\d+", re.I),
    re.compile(r"\bchapter\s+\d+", re.I),
]

# Heuristic thresholds. Tuned conservatively — a false "self_contained" only
# costs one generation; a false skip is worse, so heuristics only skip on strong
# signals and defer everything else to the arbiter.
AMENDMENT_MIN_HITS = 2
XREF_MIN_HITS = 6
XREF_MIN_DENSITY = 5.0  # cross-refs per 100 words


@dataclass
class PreclassifyResult:
    entry_ref: str
    classification: WorklistClass
    route: str
    reason: str
    confidence: float
    method: str
    event: JudgeEvent
    llm_error: Optional[dict[str, str]] = None


def _entry_ref(entry: dict[str, Any]) -> str:
    return str(entry.get("citation") or entry.get("legal_id") or entry.get("ref") or "")


def _count(markers: list[re.Pattern[str]], text: str) -> int:
    return sum(len(p.findall(text)) for p in markers)


def heuristic_classify(text: str) -> Optional[tuple[WorklistClass, float, str]]:
    """Deterministic pass. Returns None when the entry is ambiguous."""

    if not text or not text.strip():
        return None
    amendment_hits = _count(_AMENDMENT_MARKERS, text)
    if amendment_hits >= AMENDMENT_MIN_HITS:
        return (
            WorklistClass.AMENDMENT_ACT,
            0.9,
            f"amend-in-place instruction ({amendment_hits} markers); the compile "
            "gate rejects amendment acts without a source_relation.amendment "
            "operation (#1079 class)",
        )
    xref_hits = _count(_XREF_MARKERS, text)
    words = max(len(text.split()), 1)
    density = xref_hits / words * 100.0
    if xref_hits >= XREF_MIN_HITS and density >= XREF_MIN_DENSITY:
        return (
            WorklistClass.XREF_HEAVY,
            0.8,
            f"reference-dominated ({xref_hits} cross-refs, {density:.1f}/100 "
            "words); self-import / upstream-placement risk (#1058 class)",
        )
    return None


_SYSTEM = (
    "You are a worklist pre-classifier for an encoded-law pipeline. Given a "
    "legal provision, classify how the encoder should treat it, so the "
    "dispatcher does not waste a generation the compile gate will reject. "
    "Classes: self_contained (a standalone operative rule — encode it); "
    "amendment_act (an instruction to amend existing law in place); xref_heavy "
    "(mostly cross-references, little operative content, self-import risk); "
    "needs_container (cannot stand alone without a parent container/definitions). "
    "Prefer self_contained only when the provision states an operative rule on "
    "its own."
)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "classification": {
            "type": "string",
            "enum": [c.value for c in WorklistClass],
        },
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["classification", "confidence", "reason"],
}


def _build_event(
    entry_ref: str,
    classification: WorklistClass,
    route: str,
    reason: str,
    confidence: float,
    *,
    method: str,
    model: Optional[str],
    generator_model: str,
    escalated: bool,
    tokens,
    llm_error: Optional[dict[str, str]] = None,
) -> JudgeEvent:
    extra: dict[str, Any] = {
        "classification": classification.value,
        "route": route,
        "reason": reason,
        "method": method,
    }
    if llm_error is not None:
        # Keep the judge failure visible in the run-log even though the entry is
        # safely skipped rather than dropped.
        extra["judge_error"] = llm_error
    return JudgeEvent(
        stage=JudgeStage.WORKLIST_PRECLASSIFY,
        verdict=Verdict.PASS if route == GENERATE else Verdict.SKIP,
        confidence=confidence,
        advisory=True,
        findings=[
            Finding(
                clause_ref=entry_ref,
                rule_path=entry_ref,
                kind="routing",
                explanation=f"{classification.value} -> {route}: {reason}",
            )
        ],
        model=model,
        generator_model=generator_model,
        escalated=escalated,
        tokens=tokens,
        subject_ref=entry_ref,
        extra=extra,
    )


def classify(
    entry: dict[str, Any],
    *,
    client: Optional[JudgeClient] = None,
    use_llm: bool = True,
) -> PreclassifyResult:
    """Classify one worklist entry. Never drops — always returns a route."""

    client = client or JudgeClient()
    entry_ref = _entry_ref(entry)
    text = str(entry.get("source_text") or entry.get("text") or "")

    # No provision text to classify: skip-with-reason rather than pay for an
    # arbiter call on empty input or silently route it to generation.
    if not text.strip():
        return _skip_unclassified(entry_ref, client, reason="no source text provided")

    heur = heuristic_classify(text)
    if heur is not None:
        classification, confidence, reason = heur
        route = GENERATE if classification == WorklistClass.SELF_CONTAINED else SKIP
        return PreclassifyResult(
            entry_ref=entry_ref,
            classification=classification,
            route=route,
            reason=reason,
            confidence=confidence,
            method="heuristic",
            event=_build_event(
                entry_ref,
                classification,
                route,
                reason,
                confidence,
                method="heuristic",
                model=None,
                generator_model=client.generator_model,
                escalated=False,
                tokens=TokenCounts(),
            ),
        )

    # Ambiguous. If we cannot consult the arbiter, skip-with-reason — never drop,
    # never silently generate an entry we could not confirm is self-contained.
    if not use_llm:
        return _skip_unclassified(
            entry_ref, client, reason="classification deferred (llm disabled)"
        )

    call = client.call(
        system=_SYSTEM,
        user_prompt=(
            "Classify this provision.\n\n=== PROVISION TEXT ===\n"
            f"{truncate_provision(text, client.provision_chars)}\n\n"
            "Return JSON: classification, confidence (0..1), reason."
        ),
        schema=_SCHEMA,
    )
    if not call.ok:
        err = (
            {"type": call.error.type, "message": call.error.message}
            if call.error
            else {"type": "unknown", "message": "judge failed"}
        )
        return _skip_unclassified(
            entry_ref,
            client,
            reason=f"classification unavailable ({err['type']}: {err['message']})",
            llm_error=err,
            tokens=call.tokens,
            model=call.model,
            escalated=call.escalated,
        )

    payload = call.payload or {}
    try:
        classification = WorklistClass(str(payload.get("classification")))
    except ValueError:
        # Unrecognized label — treat as needs_container and skip, never drop.
        return _skip_unclassified(
            entry_ref,
            client,
            reason=f"arbiter returned an unrecognized class "
            f"{payload.get('classification')!r}",
            tokens=call.tokens,
            model=call.model,
            escalated=call.escalated,
        )
    confidence = coerce_confidence(payload.get("confidence")) or 0.0
    reason = str(payload.get("reason", ""))
    route = GENERATE if classification == WorklistClass.SELF_CONTAINED else SKIP
    return PreclassifyResult(
        entry_ref=entry_ref,
        classification=classification,
        route=route,
        reason=reason,
        confidence=confidence,
        method="llm",
        event=_build_event(
            entry_ref,
            classification,
            route,
            reason,
            confidence,
            method="llm",
            model=call.model,
            generator_model=client.generator_model,
            escalated=call.escalated,
            tokens=call.tokens,
        ),
    )


def _skip_unclassified(
    entry_ref: str,
    client: JudgeClient,
    *,
    reason: str,
    llm_error: Optional[dict[str, str]] = None,
    tokens=None,
    model: Optional[str] = None,
    escalated: bool = False,
) -> PreclassifyResult:
    classification = WorklistClass.NEEDS_CONTAINER
    return PreclassifyResult(
        entry_ref=entry_ref,
        classification=classification,
        route=SKIP,
        reason=reason,
        confidence=0.0,
        method="fallback-skip",
        llm_error=llm_error,
        event=_build_event(
            entry_ref,
            classification,
            SKIP,
            reason,
            0.0,
            method="fallback-skip",
            model=model,
            generator_model=client.generator_model,
            escalated=escalated,
            tokens=tokens or TokenCounts(),
            llm_error=llm_error,
        ),
    )


def classify_batch(
    entries: list[dict[str, Any]],
    *,
    client: Optional[JudgeClient] = None,
    use_llm: bool = True,
) -> list[PreclassifyResult]:
    """Classify a worklist. Output length always equals input length (no drops)."""

    client = client or JudgeClient()
    return [classify(e, client=client, use_llm=use_llm) for e in entries]
