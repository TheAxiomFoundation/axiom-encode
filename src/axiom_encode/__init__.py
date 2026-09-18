__version__ = "0.2.2007"
# Axiom Encode - AI-assisted RuleSpec encoding
# Self-contained encoding infrastructure -- no external plugin dependencies.


from .attempt_evidence import (
    AttemptArtifact,
    AttemptEvidence,
    AttemptRef,
    RepairTriple,
    iter_attempt_evidence,
    iter_repair_triples,
)
from .constants import (
    DEFAULT_CLI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_ESCALATE_AFTER,
    DEFAULT_OPENAI_ESCALATION_MODEL,
    DEFAULT_OPENAI_MODEL,
    REVIEWER_CLI_MODEL,
)
from .harness import (
    AgentSDKBackend,
    # Calibration Metrics
    CalibrationMetrics,
    CalibrationSnapshot,
    ClaudeCodeBackend,
    CodexCLIBackend,
    ComplexityFactors,
    # Encoder Backends
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
    # Encoding DB
    EncodingDB,
    EncodingRun,
    EvalResult,
    EvalRunnerSpec,
    Iteration,
    IterationError,
    PipelineResult,
    PredictionScores,
    ProofValidationResult,
    ReviewResult,
    ReviewResults,
    ValidationResult,
    # Validator Pipeline
    ValidatorPipeline,
    compute_calibration,
    create_run,
    evaluate_artifact,
    find_rulespec_proof_issues,
    get_calibration_trend,
    parse_runner_spec,
    print_calibration_report,
    run_model_eval,
    save_calibration_snapshot,
    validate_rulespec_proofs,
)
from .harness.validation_issues import ValidationIssue

__all__ = [
    "__version__",
    "DEFAULT_MODEL",
    "DEFAULT_CLI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_OPENAI_ESCALATION_MODEL",
    "DEFAULT_OPENAI_ESCALATE_AFTER",
    "REVIEWER_CLI_MODEL",
    "EncodingDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "ReviewResult",
    "ReviewResults",
    "create_run",
    # Attempt evidence: read-only (run_id, attempt, artifact, issues, parent)
    "AttemptArtifact",
    "AttemptEvidence",
    "AttemptRef",
    "RepairTriple",
    "ValidationIssue",
    "iter_attempt_evidence",
    "iter_repair_triples",
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "ProofValidationResult",
    "find_rulespec_proof_issues",
    "validate_rulespec_proofs",
    # Encoder Backends
    "EncoderBackend",
    "ClaudeCodeBackend",
    "CodexCLIBackend",
    "AgentSDKBackend",
    "EncoderRequest",
    "EncoderResponse",
    "PredictionScores",
    "EvalRunnerSpec",
    "EvalResult",
    "parse_runner_spec",
    "evaluate_artifact",
    "run_model_eval",
    # Calibration Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
