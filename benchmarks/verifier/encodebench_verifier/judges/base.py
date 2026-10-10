"""The one interface every judge runner implements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .. import DEFECT_KINDS
from ..cases import VerifierCase

VERDICT_PASS = "pass"
VERDICT_FLAG = "flag"
VERDICT_ERROR = "error"
VERDICTS = (VERDICT_PASS, VERDICT_FLAG, VERDICT_ERROR)

CHANNEL_NATIVE = "native"
CHANNEL_VERDICT_FALLBACK = "verdict_fallback"
CHANNELS = (CHANNEL_NATIVE, CHANNEL_VERDICT_FALLBACK)


def _unit_or_none(value: Any) -> bool:
    if value is None:
        return True
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


@dataclass
class JudgeResponse:
    """What a judge said about one case, normalised across judge families.

    ``verdict_score`` and every ``kind_scores`` entry are "probability the
    artifact is defective" in [0, 1]. ``kind_score_channels`` records whether
    a kind score came from a kind-specific answer (``native``) or fell back to
    the verdict score because the judge has no question for that kind
    (``verdict_fallback``). ``findings`` follow the referee's shape
    (``kind``, ``rule_path``, ``clause_ref``, ``explanation``); judges that
    return probabilities only leave it empty. Token counts are ``None`` when
    the provider did not report usage: unknown is never rendered as zero.
    """

    verdict: str
    verdict_score: Optional[float]
    kind_scores: dict[str, Optional[float]]
    kind_score_channels: dict[str, str]
    findings: list[dict[str, Any]]
    latency_ms: int
    tokens_input: Optional[int]
    tokens_output: Optional[int]
    model: str
    error: Optional[dict[str, str]] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in (VERDICT_PASS, VERDICT_FLAG) and self.error is None

    def problems(self) -> list[str]:
        """Contract violations; an empty list means the response is usable.

        A scored response must carry a verdict score and one score per defect
        kind (falling back to the verdict score where the judge has no
        kind-specific answer), so no case silently drops out of a kind's
        sample. An error response carries its error and no scores.
        """

        found: list[str] = []
        if self.verdict not in VERDICTS:
            found.append(f"unknown verdict {self.verdict!r}")
            return found
        if self.verdict == VERDICT_ERROR:
            if not self.error:
                found.append("error verdict without an error")
            if self.verdict_score is not None or any(
                value is not None for value in self.kind_scores.values()
            ):
                found.append("error verdict carries scores")
            return found
        if self.error:
            found.append("error present on a non-error verdict")
        if self.verdict_score is None or not _unit_or_none(self.verdict_score):
            found.append("verdict_score missing or outside [0, 1]")
        for kind in DEFECT_KINDS:
            value = self.kind_scores.get(kind)
            if value is None or not _unit_or_none(value):
                found.append(f"kind score for {kind} missing or outside [0, 1]")
            channel = self.kind_score_channels.get(kind)
            if channel not in CHANNELS:
                found.append(f"kind score channel for {kind} is {channel!r}")
        for name, count in (
            ("input", self.tokens_input),
            ("output", self.tokens_output),
        ):
            if count is not None and (not isinstance(count, int) or count < 0):
                found.append(f"{name} token count is not a non-negative integer")
        return found

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

        def count(value: Any) -> Optional[int]:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value

        return cls(
            verdict=str(payload.get("verdict", VERDICT_ERROR)),
            verdict_score=payload.get("verdict_score"),
            kind_scores=dict(payload.get("kind_scores") or {}),
            kind_score_channels=dict(payload.get("kind_score_channels") or {}),
            findings=list(payload.get("findings") or []),
            latency_ms=int(payload.get("latency_ms") or 0),
            tokens_input=count(tokens.get("input")),
            tokens_output=count(tokens.get("output")),
            model=str(payload.get("model", "")),
            error=payload.get("error"),
            raw=dict(payload.get("raw") or {}),
        )


def error_response(
    model: str,
    error_type: str,
    message: str,
    *,
    latency_ms: int = 0,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    raw: Optional[dict[str, Any]] = None,
) -> JudgeResponse:
    """A fail-closed response: never a pass, never a score."""

    return JudgeResponse(
        verdict=VERDICT_ERROR,
        verdict_score=None,
        kind_scores={kind: None for kind in DEFECT_KINDS},
        kind_score_channels={},
        findings=[],
        latency_ms=latency_ms,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        model=model,
        error={"type": error_type, "message": message},
        raw=dict(raw or {}),
    )


def checked(response: JudgeResponse) -> JudgeResponse:
    """Return ``response`` if it honours the contract, else a fail-closed error."""

    found = response.problems()
    if not found:
        return response
    return error_response(
        response.model,
        "invalid_response",
        "; ".join(found)[:500],
        latency_ms=response.latency_ms,
        tokens_input=response.tokens_input,
        tokens_output=response.tokens_output,
    )


def clamp_unit(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
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
