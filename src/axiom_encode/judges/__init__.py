"""LLM judge stages for the encoding pipeline (maximum-traceability part 2).

Cross-family judging: the generator is ``gpt-5.6-terra``; judges run on a
Claude-family model. Every stage emits a structured
:class:`~axiom_encode.judges.run_log.JudgeEvent`
(``axiom_encode.run_log.v1``) and is fail-closed — a judge that cannot reach a
verdict emits a ``judge_error`` event, never a silent pass.

Stages:

* :mod:`~axiom_encode.judges.statutory_fidelity` — per generation, post-gates
  pre-apply. Advisory ``needs-review`` label + event.
* :mod:`~axiom_encode.judges.grid_adequacy` — per oracle suite. Names untested
  boundaries as structured gaps with a deterministic follow-up hook.
* :mod:`~axiom_encode.judges.disposition` — per new disposition. Verifies the
  causal claim reproduces the residual on >=3 sampled records.
* :mod:`~axiom_encode.judges.preclassifier` — dispatcher-side worklist triage
  before paying for generation; routes amendment/xref to skip-with-reason.
* :mod:`~axiom_encode.judges.drift` — golden-regeneration weekly drift check.
* :mod:`~axiom_encode.judges.calibration` — replays the fidelity referee over
  historical generations and reports FP/FN.
"""

from __future__ import annotations

from . import (
    calibration,
    disposition,
    drift,
    grid_adequacy,
    preclassifier,
    regeneration,
    statutory_fidelity,
)
from .client import JudgeCall, JudgeClient, model_family, truncate_provision
from .disposition import Disposition
from .preclassifier import PreclassifyResult, WorklistClass
from .run_log import (
    FINDING_KINDS,
    JUDGE_STAGE,
    SCHEMA_VERSION,
    Finding,
    JudgeError,
    JudgeEvent,
    JudgeStage,
    TokenCounts,
    Verdict,
    error_event,
    validate_event_dict,
)

__all__ = [
    "SCHEMA_VERSION",
    "JUDGE_STAGE",
    "FINDING_KINDS",
    "Finding",
    "JudgeError",
    "JudgeEvent",
    "JudgeStage",
    "TokenCounts",
    "Verdict",
    "error_event",
    "validate_event_dict",
    "JudgeClient",
    "JudgeCall",
    "model_family",
    "truncate_provision",
    # stage modules
    "statutory_fidelity",
    "grid_adequacy",
    "disposition",
    "preclassifier",
    "drift",
    "regeneration",
    "calibration",
    # stage types
    "Disposition",
    "PreclassifyResult",
    "WorklistClass",
]
