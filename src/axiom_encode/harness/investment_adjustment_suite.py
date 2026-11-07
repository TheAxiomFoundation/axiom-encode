"""Bind an entire ordinary positive suite to independently observed records.

This is a standalone observation, not proof of protected execution or source
coverage. The materializer must later bind its exact bytes and ordinal records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import yaml

from .investment_adjustment_execution import verify_investment_adjustment_compiled
from .investment_adjustment_records import (
    InvestmentAdjustmentRecordObservation,
    observe_investment_adjustment_record,
)


@dataclass(frozen=True)
class InvestmentAdjustmentSuiteObservation:
    content_sha256: str
    companion_sha256: str
    parent_sha256: str
    guide_sha256: str
    compiled_identity_sha256: str
    engine_pin: str
    caller: str
    records: tuple[InvestmentAdjustmentRecordObservation, ...]
    residuals: tuple[str, ...]
    upstream_boundaries: tuple[str, ...]


def observe_investment_adjustment_suite(
    bound, content, parent, guide, artifact, engine_pin, companion, calls, *, caller
):
    """Validate every raw ordinal; never silently keep a passing subset."""
    from .validator_pipeline import _safe_load_unique_keys

    verify_investment_adjustment_compiled(
        bound, content, parent, guide, artifact, engine_pin
    )
    if type(companion) is not str or type(calls) is not list:
        raise ValueError("raw companion and complete record list required")
    try:
        cases = _safe_load_unique_keys(companion)
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        raise ValueError("unique valid companion YAML required") from exc
    if type(cases) is not list or not cases or len(cases) != len(calls):
        raise ValueError("complete nonempty ordinal execution required")
    names = set()
    observed = []
    for case, call in zip(cases, calls, strict=True):
        if type(case) is not dict or type(case.get("name")) is not str:
            raise ValueError("case name required")
        if case["name"] in names:
            raise ValueError("duplicate case name is not distinct evidence")
        names.add(case["name"])
        observed.append(
            observe_investment_adjustment_record(
                bound,
                content,
                parent,
                guide,
                artifact,
                engine_pin,
                case,
                call,
                caller=caller,
            )
        )
    return InvestmentAdjustmentSuiteObservation(
        hashlib.sha256(content.encode()).hexdigest(),
        hashlib.sha256(companion.encode()).hexdigest(),
        hashlib.sha256(parent.encode()).hexdigest(),
        hashlib.sha256(guide.encode()).hexdigest(),
        hashlib.sha256(
            json.dumps(
                artifact, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        engine_pin,
        caller,
        tuple(observed),
        (
            *bound.residuals,
            "protected_capture_required",
            "source_witness_allocation_required",
        ),
        bound.upstream_boundaries,
    )
