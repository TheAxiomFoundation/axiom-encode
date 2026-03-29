# AutoRAC - AI-assisted RAC encoding
# Self-contained encoding infrastructure -- no external plugin dependencies.

__version__ = "0.2.5"

from .constants import (
    DEFAULT_CLI_MODEL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    REVIEWER_CLI_MODEL,
)
from .harness import (
    ActualScores,
    AgentSDKBackend,
    AgentSuggestion,
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
    ExperimentDB,  # backward compat
    FinalScores,
    Iteration,
    IterationError,
    # Orchestrator
    Orchestrator,
    PipelineResult,
    PredictedScores,
    PredictionScores,
    ReviewResult,
    ReviewResults,
    ValidationResult,
    # Validator Pipeline
    ValidatorPipeline,
    compute_calibration,
    create_run,
    evaluate_artifact,
    get_calibration_trend,
    parse_runner_spec,
    print_calibration_report,
    run_model_eval,
    save_calibration_snapshot,
    validate_file,
)

__all__ = [
    "__version__",
    "DEFAULT_MODEL",
    "DEFAULT_CLI_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "REVIEWER_CLI_MODEL",
    "EncodingDB",
    "ExperimentDB",  # backward compat
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "ReviewResult",
    "ReviewResults",
    "FinalScores",
    "PredictedScores",
    "ActualScores",
    "AgentSuggestion",
    "create_run",
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
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
    # Orchestrator
    "Orchestrator",
    # Calibration Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
