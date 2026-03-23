# Encoding Harness
# Feedback loop for AI-assisted statute encoding
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
    ActualScores,
    AgentSuggestion,
    ComplexityFactors,
    EncodingDB,
    EncodingRun,
    ExperimentDB,  # backward compat alias
    FinalScores,
    Iteration,
    IterationError,
    PredictedScores,
    ReviewResult,
    ReviewResults,
    create_run,
)
from .metrics import (
    CalibrationMetrics,
    CalibrationSnapshot,
    compute_calibration,
    get_calibration_trend,
    print_calibration_report,
    save_calibration_snapshot,
)
from .evals import EvalResult, EvalRunnerSpec, evaluate_artifact, parse_runner_spec, run_model_eval
from .orchestrator import Orchestrator
from .validator_pipeline import (
    PipelineResult,
    ValidationResult,
    ValidatorPipeline,
    validate_file,
)

__all__ = [
    # Encoding DB
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
    # Validator Pipeline
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
