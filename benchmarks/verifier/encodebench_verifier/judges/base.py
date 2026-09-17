"""The one interface every judge runner implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .. import DEFECT_KINDS
from ..cases import VerifierCase

VERDICT_PASS = "pass"
VERDICT_FLAG = "flag"
VERDICT_ERROR = "error"

CHANNEL_NATIVE = "native"
CHANNEL_VERDICT_FALLBACK = "verdict_fallback"


@dataclass
class JudgeResponse:
    """What a judge said about one case, normalised across judge families.

    ``verdict_score`` and every ``kind_scores`` entry are "probability the
    artifact is defective" in [0, 1]. ``kind_score_channels`` records whether
    a kind score came from a kind-specific answer (``native``) or fell back to
    the verdict score because the judge has no question for that kind
    (``verdict_fallback``). ``findings`` follow the referee's shape
    (``kind``, ``rule_path``, ``clause_ref``, ``explanation``); judges that
    return probabilities only leave it empty.
    """

    verdict: str
    verdict_score: Optional[float]
    kind_scores: dict[str, Optional[float]]
    kind_score_channels: dict[str, str]
    findings: list[dict[str, Any]]
    latency_ms: int
    tokens_input: int
    tokens_output: int
    model: str
    error: Optional[dict[str, str]] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in (VERDICT_PASS, VERDICT_FLAG) and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "verdict_score": self.verdict_score,
            "kind_scores": dict(self.kind_scores),
            "kind_score_channels": dict(self.kind_score_channels),
            "findings": list(self.findings),
            "latency_ms": self.latency_ms,
            "tokens": {"input": self.tokens_input, "output": self.tokens_output},
            "model": self.model,
            "error": self.error,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JudgeResponse":
        tokens = payload.get("tokens") or {}
        return cls(
            verdict=str(payload.get("verdict", VERDICT_ERROR)),
            verdict_score=payload.get("verdict_score"),
            kind_scores=dict(payload.get("kind_scores") or {}),
            kind_score_channels=dict(payload.get("kind_score_channels") or {}),
            findings=list(payload.get("findings") or []),
            latency_ms=int(payload.get("latency_ms") or 0),
            tokens_input=int(tokens.get("input") or 0),
            tokens_output=int(tokens.get("output") or 0),
            model=str(payload.get("model", "")),
            error=payload.get("error"),
            raw=dict(payload.get("raw") or {}),
        )


def error_response(
    model: str, error_type: str, message: str, *, latency_ms: int = 0
) -> JudgeResponse:
    """A fail-closed response: never a pass, never a score."""

    return JudgeResponse(
        verdict=VERDICT_ERROR,
        verdict_score=None,
        kind_scores={kind: None for kind in DEFECT_KINDS},
        kind_score_channels={},
        findings=[],
        latency_ms=latency_ms,
        tokens_input=0,
        tokens_output=0,
        model=model,
        error={"type": error_type, "message": message},
    )


def clamp_unit(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(min(1.0, max(0.0, number)), 6)


class JudgeRunner(Protocol):
    """A judge that reads one case and returns a :class:`JudgeResponse`."""

    name: str
    family: str
    model: str
    # False for judges that return probabilities only: their localisation
    # column is blank by construction rather than scored as a miss.
    supports_localization: bool

    def identity(self) -> dict[str, Any]:
        """Score-affecting configuration recorded in results.json."""

    def judge(self, case: VerifierCase) -> JudgeResponse: ...
