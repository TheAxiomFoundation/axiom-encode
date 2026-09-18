# Encoding harness
# Feedback loop for AI-assisted RuleSpec encoding
# Self-contained -- no external plugin dependencies.

from .backends import (
    AgentSDKBackend,
    ClaudeCodeBackend,
    CodexCLIBackend,
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
    PredictionScores,
)
from .encoding_db import (
    ArtifactVersion,
    ComplexityFactors,
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    JudgeEventRow,
    ParentRunRef,
    ReviewResult,
    ReviewResults,
    RunArtifact,
    create_run,
)
from .evals import (
    EvalResult,
    EvalRunnerSpec,
    evaluate_artifact,
    parse_runner_spec,
    run_model_eval,
)
from .metrics import (
    CalibrationMetrics,
    CalibrationSnapshot,
    compute_calibration,
    get_calibration_trend,
    print_calibration_report,
    save_calibration_snapshot,
)
from .policyengine_runtime import (
    PolicyEngineRuntime,
    PolicyEngineRuntimeError,
)
from .proof_validator import (
    ProofValidationResult,
    find_rulespec_proof_issues,
    validate_rulespec_proofs,
)
from .validation_issues import (
    ValidationIssue,
    structure_attempt_issues,
    structure_issue,
)
from .validator_pipeline import (
    PipelineResult,
    ValidationResult,
    ValidatorPipeline,
)

__all__ = [
    # Encoding DB
    "EncodingDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "ReviewResult",
    "ReviewResults",
    "create_run",
    # Attempt evidence (artifact versions, judge events, structured issues)
    "ArtifactVersion",
    "RunArtifact",
    "JudgeEventRow",
    "ParentRunRef",
    "ValidationIssue",
    "structure_issue",
    "structure_attempt_issues",
    # Validator Pipeline
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "PolicyEngineRuntime",
    "PolicyEngineRuntimeError",
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
