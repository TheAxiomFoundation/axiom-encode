"""Stage 1 pre-screen: the statutory-fidelity screen on TypeSafe System One.

Runs before the LLM referee (:mod:`~axiom_encode.judges.statutory_fidelity`)
on the same provision window and artifact. It asks one ``Choice`` pass/flag
verdict plus one ``Noul`` probability per fidelity finding kind
(``unrepresented_clause``, ``untraceable_branch``, ``boundary_direction``,
``amount_mismatch``) and emits a :class:`~axiom_encode.judges.run_log.JudgeEvent`
under :attr:`~axiom_encode.judges.run_log.JudgeStage.STATUTORY_FIDELITY_SCREEN`.

What the screen is and is not:

* It is advisory and never a gate. ``advisory`` is hard-wired on; a flag maps
  to ``status == "passed"`` in the run log, and the screen never produces the
  ``needs-review`` label (that stays with the referee).
* Its findings carry a kind and a probability and **no locators**: System One
  returns probabilities only, so ``clause_ref`` and ``rule_path`` are left
  empty rather than fabricated, and the explanation says so.
* It can decide whether to *request* the referee. The cascade policy
  (:class:`ScreenPolicy`) is advisory by default (the referee always runs); in
  ``cascade`` mode (``AXIOM_JUDGE_SCREEN_MODE=cascade``) a screen whose
  configured kinds all sit below their thresholds records the screen and skips
  the referee. An errored or incomplete screen always requests the referee
  (fail closed).

Question text and the state shape are verbatim from the 2026-09-17 pilot
(``_axiom-runs/jev-judge-pilot-2026-09-17``); changing either invalidates the
pilot evidence behind the placeholder threshold. The default thresholds are a
PLACEHOLDER (see :data:`~axiom_encode.constants.DEFAULT_JUDGE_SCREEN_THRESHOLD`);
the verifier track under ``benchmarks/verifier/`` is the authority for the
real value. By default only the two pilot-validated kinds (amount, boundary)
carry a threshold; the other two are recorded but never reported as findings
or used by the cascade unless a per-kind threshold is configured.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from axiom_encode.constants import (
    DEFAULT_JUDGE_SCREEN_MODE,
    DEFAULT_JUDGE_SCREEN_THRESHOLD,
    JUDGE_SCREEN_MODE_ADVISORY,
    JUDGE_SCREEN_MODE_CASCADE,
)

from .client import truncate_provision
from .run_log import (
    Finding,
    JudgeEvent,
    JudgeStage,
    Verdict,
    error_event,
)
from .statutory_fidelity import FIDELITY_KINDS
from .system_one import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulQuestion,
    Question,
    SystemOneClient,
    questions_sha256,
)

MODES = frozenset({JUDGE_SCREEN_MODE_ADVISORY, JUDGE_SCREEN_MODE_CASCADE})

# Kinds the 2026-09-17 planted-defect pilot validated (amount AUC 1.000,
# boundary AUC 0.953). They carry the placeholder threshold by default.
# ``unrepresented_clause`` (dropped-conjunct AUC 0.603) and
# ``untraceable_branch`` are recorded only unless configured per kind.
CASCADE_KINDS: tuple[str, ...] = ("amount_mismatch", "boundary_direction")

SCREEN_MODE_ENV = "AXIOM_JUDGE_SCREEN_MODE"
SCREEN_THRESHOLD_ENV = "AXIOM_JUDGE_SCREEN_THRESHOLD"
SCREEN_THRESHOLD_KIND_ENV_PREFIX = "AXIOM_JUDGE_SCREEN_THRESHOLD_"

# Identifies the question set the recorded probabilities were produced under.
QUESTIONS_VERSION = "jev-judge-pilot-2026-09-17"

_VERDICT_INSTRUCTIONS = (
    "Judge whether the generated RuleSpec artifact faithfully encodes the "
    "provision. Judge fidelity only, not style."
)
_VERDICT_CRITERIA = {
    "pass": (
        "Every branch traces to provision language, no operative clause is "
        "unrepresented, boundary inequalities point the right direction, and "
        "every amount, rate and threshold matches the source."
    ),
    "flag": (
        "At least one branch has no source basis, or an operative clause is "
        "unrepresented, or a boundary inequality is reversed or off by its "
        "boundary, or an amount, rate or threshold does not match the source."
    ),
}
_KIND_INSTRUCTIONS = {
    "untraceable_branch": (
        "The artifact contains at least one branch or condition with no basis "
        "in the provision text."
    ),
    "unrepresented_clause": (
        "At least one operative clause of the provision is not represented in "
        "the artifact."
    ),
    "boundary_direction": (
        "At least one boundary inequality in the artifact is reversed or off "
        "by its boundary relative to the provision."
    ),
    "amount_mismatch": (
        "At least one amount, rate or threshold in the artifact does not match "
        "the provision text."
    ),
}
_FAITHFUL_INSTRUCTIONS = (
    "The generated RuleSpec artifact is a faithful encoding of the provision text."
)


def build_state(
    provision_text: str, generated_rule: str, *, citation: Optional[str]
) -> dict[str, str]:
    """The JSON state System One judges; keys verbatim from the pilot."""

    return {
        "citation": citation or "",
        "provision_text_verbatim": provision_text,
        "generated_rulespec_artifact": generated_rule,
    }


def build_questions() -> dict[str, Question]:
    """One Choice verdict, one Noul per fidelity kind, plus the pilot's faithful Noul."""

    questions: dict[str, Question] = {
        "verdict": ChoiceQuestion(
            instructions=_VERDICT_INSTRUCTIONS, criteria=_VERDICT_CRITERIA
        ),
        # Kept from the pilot so recorded probabilities stay comparable; it is
        # recorded (``screen.p_faithful``) and never used for a decision.
        "faithful": NoulQuestion(instructions=_FAITHFUL_INSTRUCTIONS),
    }
    for kind in FIDELITY_KINDS:
        questions[kind] = NoulQuestion(instructions=_KIND_INSTRUCTIONS[kind])
    return questions


# -- policy ---------------------------------------------------------------


def _validate_threshold(kind: str, value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"screen threshold for {kind!r} is not numeric") from exc
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"screen threshold for {kind!r} must be in [0, 1]")
    return threshold


@dataclass(frozen=True)
class ScreenPolicy:
    """Cascade policy: mode plus per-kind thresholds.

    ``thresholds`` holds only the kinds that report findings and can trigger
    the referee; a kind absent from it is recorded but never acted on.
    ``source`` records whether the thresholds came from configuration
    (``env``) or the placeholder default.
    """

    mode: str = DEFAULT_JUDGE_SCREEN_MODE
    thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            kind: DEFAULT_JUDGE_SCREEN_THRESHOLD for kind in CASCADE_KINDS
        }
    )
    source: str = "placeholder"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"unknown screen mode {self.mode!r}; expected one of {sorted(MODES)}"
            )
        clean: dict[str, float] = {}
        for kind, value in dict(self.thresholds).items():
            if kind not in FIDELITY_KINDS:
                raise ValueError(f"unknown fidelity kind {kind!r} in screen thresholds")
            clean[kind] = _validate_threshold(kind, value)
        if self.mode == JUDGE_SCREEN_MODE_CASCADE and not clean:
            # With nothing to trigger on, a cascade would skip the referee on
            # every artifact. Refuse the configuration rather than fail open.
            raise ValueError("cascade mode requires at least one screen threshold")
        object.__setattr__(self, "thresholds", clean)

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "ScreenPolicy":
        """Read ``AXIOM_JUDGE_SCREEN_MODE`` and the threshold variables.

        ``AXIOM_JUDGE_SCREEN_THRESHOLD`` replaces the placeholder for the
        cascade kinds; ``AXIOM_JUDGE_SCREEN_THRESHOLD_<KIND>`` (upper-case
        kind) sets or overrides one kind, and may enable a record-only kind.
        """

        env = os.environ if environ is None else environ
        mode = (
            env.get(SCREEN_MODE_ENV) or ""
        ).strip().lower() or DEFAULT_JUDGE_SCREEN_MODE
        thresholds: dict[str, float] = {}
        source = "placeholder"
        shared = (env.get(SCREEN_THRESHOLD_ENV) or "").strip()
        base = DEFAULT_JUDGE_SCREEN_THRESHOLD
        if shared:
            base = _validate_threshold("*", shared)
            source = "env"
        for kind in CASCADE_KINDS:
            thresholds[kind] = base
        for kind in FIDELITY_KINDS:
            raw = (
                env.get(SCREEN_THRESHOLD_KIND_ENV_PREFIX + kind.upper()) or ""
            ).strip()
            if raw:
                thresholds[kind] = _validate_threshold(kind, raw)
                source = "env"
        return cls(mode=mode, thresholds=thresholds, source=source)

    def with_mode(self, mode: str) -> "ScreenPolicy":
        return ScreenPolicy(mode=mode, thresholds=self.thresholds, source=self.source)

    def threshold_for(self, kind: str) -> Optional[float]:
        return self.thresholds.get(kind)

    @property
    def cascade(self) -> bool:
        return self.mode == JUDGE_SCREEN_MODE_CASCADE

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "thresholds": dict(self.thresholds),
            "threshold_source": self.source,
        }


@dataclass(frozen=True)
class CascadeDecision:
    """Whether the LLM referee should run after this screen, and why."""

    request_referee: bool
    reason: str
    triggered: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_referee": self.request_referee,
            "reason": self.reason,
            "triggered": list(self.triggered),
        }


def cascade_decision(event: JudgeEvent, policy: ScreenPolicy) -> CascadeDecision:
    """Decide whether the referee runs. Fail closed: errors always request it."""

    if event.stage != JudgeStage.STATUTORY_FIDELITY_SCREEN:
        raise ValueError("cascade_decision expects a statutory_fidelity_screen event")
    if event.is_error:
        return CascadeDecision(True, "screen_error")
    screen = event.extra.get("screen") or {}
    probabilities = screen.get("probabilities") or {}
    triggered: list[str] = []
    for kind, threshold in policy.thresholds.items():
        probability = probabilities.get(kind)
        if probability is None:
            # A configured kind without a probability is an incomplete screen;
            # never let it read as "below threshold".
            return CascadeDecision(True, "screen_incomplete", tuple(triggered))
        if float(probability) >= threshold:
            triggered.append(kind)
    if not policy.cascade:
        return CascadeDecision(True, "advisory_mode", tuple(triggered))
    if triggered:
        return CascadeDecision(True, "threshold_exceeded", tuple(triggered))
    return CascadeDecision(False, "below_threshold")


# -- stage ----------------------------------------------------------------


def _finding(kind: str, probability: float, threshold: float) -> Finding:
    return Finding(
        clause_ref="",
        rule_path="",
        kind=kind,
        explanation=(
            f"screen probability {probability:.2f} of {kind} (threshold "
            f"{threshold:.2f}); the screen returns probabilities only and "
            "does not locate the clause or the artifact locus"
        ),
        probability=probability,
    )


def run(
    provision_text: str,
    generated_rule: str,
    *,
    citation: Optional[str] = None,
    rule_path: Optional[str] = None,
    run_id: Optional[str] = None,
    client: Optional[SystemOneClient] = None,
    policy: Optional[ScreenPolicy] = None,
) -> JudgeEvent:
    """Screen one generation and return an advisory :class:`JudgeEvent`.

    The returned event's ``extra["screen"]`` carries the policy, the recorded
    probabilities, latency, and the cascade decision; ``extra["judge_model_family"]``
    records the responding model's family.
    """

    client = client or SystemOneClient()
    policy = policy or ScreenPolicy.from_env()
    subject = citation or rule_path
    state = build_state(
        truncate_provision(provision_text, client.provision_chars),
        generated_rule,
        citation=citation,
    )
    questions = build_questions()
    prompt_sha = questions_sha256(state, questions)
    call = client.call(state=state, questions=questions)

    screen: dict[str, Any] = {
        **policy.to_dict(),
        "questions_version": QUESTIONS_VERSION,
        "latency_ms": call.latency_ms,
    }

    def finish(event: JudgeEvent) -> JudgeEvent:
        event.judge_prompt_sha256 = prompt_sha
        if call.family:
            event.extra["judge_model_family"] = call.family
        event.extra["screen"] = screen
        screen["cascade"] = cascade_decision(event, policy).to_dict()
        return event

    if not call.ok:
        return finish(
            error_event(
                JudgeStage.STATUTORY_FIDELITY_SCREEN,
                call.error.message if call.error else "unknown screen failure",
                error_type=call.error.type if call.error else "unknown",
                model=call.model,
                generator_model=client.generator_model,
                tokens=call.tokens,
                run_id=run_id,
                subject_ref=subject,
            )
        )

    answers = call.answers or {}
    verdict_answer = answers.get("verdict")
    if not isinstance(verdict_answer, ChoiceAnswer):
        return finish(
            error_event(
                JudgeStage.STATUTORY_FIDELITY_SCREEN,
                "screen returned no choice verdict",
                error_type="schema_error",
                model=call.model,
                generator_model=client.generator_model,
                tokens=call.tokens,
                run_id=run_id,
                subject_ref=subject,
            )
        )
    raw_verdict = verdict_answer.choice.strip().lower()
    if raw_verdict not in ("pass", "flag"):
        # Fail closed on an unexpected label, never default to PASS.
        return finish(
            error_event(
                JudgeStage.STATUTORY_FIDELITY_SCREEN,
                f"unrecognized verdict {raw_verdict!r} from screen",
                error_type="unrecognized_verdict",
                model=call.model,
                generator_model=client.generator_model,
                tokens=call.tokens,
                run_id=run_id,
                subject_ref=subject,
            )
        )

    probabilities: dict[str, float] = {}
    for kind in FIDELITY_KINDS:
        value = answers.get(kind)
        if not isinstance(value, float):
            return finish(
                error_event(
                    JudgeStage.STATUTORY_FIDELITY_SCREEN,
                    f"screen returned no probability for {kind}",
                    error_type="schema_error",
                    model=call.model,
                    generator_model=client.generator_model,
                    tokens=call.tokens,
                    run_id=run_id,
                    subject_ref=subject,
                )
            )
        probabilities[kind] = value

    findings: list[Finding] = []
    for kind in FIDELITY_KINDS:
        threshold = policy.threshold_for(kind)
        if threshold is not None and probabilities[kind] >= threshold:
            findings.append(_finding(kind, probabilities[kind], threshold))

    verdict = Verdict.FLAG if (raw_verdict == "flag" or findings) else Verdict.PASS
    faithful = answers.get("faithful")
    screen.update(
        {
            "verdict_probabilities": dict(verdict_answer.probabilities),
            "probabilities": probabilities,
            "p_faithful": faithful if isinstance(faithful, float) else None,
        }
    )
    return finish(
        JudgeEvent(
            stage=JudgeStage.STATUTORY_FIDELITY_SCREEN,
            verdict=verdict,
            confidence=verdict_answer.confidence,
            # Never a gate: the screen cannot be promoted. Promotion decisions
            # belong to the referee and the verifier track.
            advisory=True,
            findings=findings,
            model=call.model,
            generator_model=client.generator_model,
            escalated=False,
            tokens=call.tokens,
            run_id=run_id,
            subject_ref=subject,
        )
    )
