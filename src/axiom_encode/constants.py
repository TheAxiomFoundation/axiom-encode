# Model constants — single source of truth for all model references.
# Change the model in ONE place, it changes everywhere.

# Full model ID for Agent SDK (API) backend
DEFAULT_MODEL = "claude-opus-4-6"

# Short-form model names for Claude Code CLI backend
DEFAULT_CLI_MODEL = "opus"
REVIEWER_CLI_MODEL = "opus"

# Default model for direct OpenAI Responses and Codex-backed GPT evals.
# This is also the *generator* family the LLM judges must never share (see
# below): same-model self-review correlates errors (the 9/9 identical
# hardcoded-600,000 incident is the cautionary tale).
DEFAULT_OPENAI_MODEL = "gpt-5.6-terra"
DEFAULT_OPENAI_ESCALATION_MODEL = "gpt-5.6-sol"
DEFAULT_OPENAI_ESCALATE_AFTER = 2

# LLM judge models (maximum-traceability part 2). Cross-family by design: the
# generator is a GPT model, so judges run on a Claude-family model. Volume runs
# on Haiku; low-confidence verdicts escalate to Sonnet. Both are overridable via
# AXIOM_JUDGE_MODEL / AXIOM_JUDGE_ESCALATION_MODEL.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_ESCALATION_MODEL = "claude-sonnet-4-5"

# Statutory-fidelity screen (judges/statutory_fidelity_screen.py): a typed
# probability pre-screen on TypeSafe System One (model family ``typesafe``,
# model ids ``jev-*``) that runs before the LLM referee. Advisory by default;
# ``AXIOM_JUDGE_SCREEN_MODE=cascade`` lets a below-threshold screen skip the
# referee. The screen never gates the pipeline.
JUDGE_SCREEN_MODE_ADVISORY = "advisory"
JUDGE_SCREEN_MODE_CASCADE = "cascade"
JUDGE_SCREEN_MODES = frozenset({JUDGE_SCREEN_MODE_ADVISORY, JUDGE_SCREEN_MODE_CASCADE})
DEFAULT_JUDGE_SCREEN_MODE = JUDGE_SCREEN_MODE_ADVISORY

# PLACEHOLDER THRESHOLD. The 2026-09-17 planted-defect pilot (30 pairs per
# kind, synthetic single-edit defects in known-good artifacts) is the only
# evidence behind this value: at 0.25 the cascade rule (amount or boundary
# probability at or above the threshold) sent every planted amount change and
# every flipped boundary to the referee while 32 of 90 clean originals crossed
# it, so a cascade would still send about a third of clean artifacts on. It is
# deliberately low because a skipped referee on a real defect costs far more
# than a referee call. The calibration harness's apply/blocked labels cannot
# validate a fidelity judge, so the verifier track (axiom-encode#1657, landing
# under ``benchmarks/verifier/``) is the authority for setting the real
# threshold; override with ``AXIOM_JUDGE_SCREEN_THRESHOLD`` (or per kind with
# ``AXIOM_JUDGE_SCREEN_THRESHOLD_<KIND>``) until it does.
DEFAULT_JUDGE_SCREEN_THRESHOLD = 0.25

# Canonical RuleSpec filesystem contract. ``programs`` is canonical content,
# but it contains declarative axiom-compose ProgramSpecs rather than atomic
# ``rulespec/v1`` modules. Every encoder, validator, signer, manifest, import,
# proof, waiver, concept, judge, and source-hash surface must use the atomic
# four-root set; layout/routing checks alone use all five filesystem roots.
RULESPEC_COMPOSITION_SPEC_ROOT = "programs"
RULESPEC_ATOMIC_MODULE_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
RULESPEC_FILESYSTEM_ROOTS = frozenset(
    {*RULESPEC_ATOMIC_MODULE_ROOTS, RULESPEC_COMPOSITION_SPEC_ROOT}
)
RULESPEC_FILE_SUFFIX = ".yaml"
RULESPEC_TEST_FILE_SUFFIX = ".test.yaml"
