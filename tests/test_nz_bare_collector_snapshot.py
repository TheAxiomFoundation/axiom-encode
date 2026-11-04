"""Keep intermediate numeric-suppression proofs bound to the pinned collector."""

import inspect
import sys
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import FunctionType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "nz_bare_structure_differential.py"
SPEC = spec_from_file_location("_nz_snapshot_invariant_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)

SOURCE = (
    "The tax applies from 1 October 1986 and 1 July 2007. "
    "Packaging contains 1 or more items. "
    "The value of the supply is less than or equal to $1,000. "
    "If more than 1 person is responsible, all must apply."
)


@pytest.fixture(scope="module")
def capture_inputs():
    baseline = AUDIT.load_baseline(
        "86a1f911a",
        "src/axiom_encode/harness/validator_pipeline.py",
        "axiom_encode.harness._nz_snapshot_invariant_baseline",
    )
    extract = baseline.extract_typed_numeric_inventory_occurrences_from_text
    normal = tuple(extract(SOURCE))
    old = tuple(AUDIT.anchored_occurrences(SOURCE, SOURCE, normal).values())
    current = AUDIT.anchored_occurrences(
        SOURCE,
        SOURCE,
        AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            SOURCE
        ),
    )
    target = next(item for item in current.values() if item["value"] == 1000)
    assert not any(item["value"] == 1000 for item in old)
    return baseline, extract, normal, old, target


def test_same_name_and_location_cannot_authorize_changed_runtime_code(
    capture_inputs, monkeypatch
):
    baseline, _, _, _, target = capture_inputs
    original = inspect.unwrap(baseline._tokenize_numeric_occurrences_from_text)
    constants = list(original.__code__.co_consts)
    position = next(
        index
        for index, value in enumerate(constants)
        if type(value) is int and value == 1
    )
    constants[position] = 2
    changed = FunctionType(
        original.__code__.replace(co_consts=tuple(constants)),
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    changed.__kwdefaults__ = original.__kwdefaults__
    assert changed.__code__.co_name == original.__code__.co_name
    assert changed.__code__.co_firstlineno == original.__code__.co_firstlineno
    monkeypatch.setattr(baseline, "_tokenize_numeric_occurrences_from_text", changed)

    def unexpected_extraction(_text):
        pytest.fail("changed code must be rejected before any baseline execution")

    assert (
        AUDIT.baseline_grouped_capture_snapshot(
            target, SOURCE, SOURCE, baseline, unexpected_extraction
        )
        is None
    )


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            "for match in percentage_context_number_matches:",
            "for match in unrelated_matches:",
        ),
        ("contextlib.suppress(ValueError)", "contextlib.suppress(RuntimeError)"),
        (
            "for occurrence in collector.inventory",
            "for occurrence in collector.grounding",
        ),
    ),
)
def test_capture_stage_rejects_another_loop_guard_or_suppression_stream(
    capture_inputs, before, after
):
    baseline, *_ = capture_inputs
    source = baseline._audit_baseline_source
    assert AUDIT.baseline_grouped_capture_stage(source) is not None
    assert before in source
    assert AUDIT.baseline_grouped_capture_stage(source.replace(before, after)) is None


def test_snapshot_restores_an_existing_trace_function(capture_inputs):
    baseline, extract, _, _, target = capture_inputs

    def previous_tracer(_frame, _event, _argument):
        return previous_tracer

    outer_tracer = sys.gettrace()
    try:
        sys.settrace(previous_tracer)
        snapshot = AUDIT.baseline_grouped_capture_snapshot(
            target, SOURCE, SOURCE, baseline, extract
        )
        assert sys.gettrace() is previous_tracer
    finally:
        sys.settrace(outer_tracer)

    assert snapshot is not None
    assert snapshot["traced_final_matches_normal"]
    assert any(item["has_temporal_context"] for item in snapshot["occurrences"])


def test_snapshot_restores_an_existing_trace_when_extraction_raises(
    capture_inputs, monkeypatch
):
    baseline, _, normal, _, target = capture_inputs

    class ForcedExtractionFailure(RuntimeError):
        pass

    def fail_during_extraction(_raw):
        raise ForcedExtractionFailure

    def previous_tracer(_frame, _event, _argument):
        return previous_tracer

    monkeypatch.setattr(
        baseline, "_parse_belgian_numeric_phrase", fail_during_extraction
    )
    outer_tracer = sys.gettrace()
    try:
        sys.settrace(previous_tracer)
        with pytest.raises(ForcedExtractionFailure):
            AUDIT.baseline_grouped_capture_snapshot(
                target, SOURCE, SOURCE, baseline, lambda _text: normal
            )
        assert sys.gettrace() is previous_tracer
    finally:
        sys.settrace(outer_tracer)


@pytest.mark.parametrize("mutation", ("raw", "source-span", "source-value", "flags"))
def test_snapshot_cannot_certify_another_target_or_numeric_meaning(
    capture_inputs, mutation
):
    baseline, extract, _, old, target = capture_inputs
    target = dict(target)
    if mutation == "raw":
        target["raw"] = "2,000"
    elif mutation == "source-span":
        target["source_start"] += 1
        target["source_end"] += 1
    elif mutation == "source-value":
        target["source_value"] = 1
    else:
        target["has_rate_context"] = not target["has_rate_context"]

    assert (
        AUDIT.cross_occurrence_restoration_evidence(
            target, SOURCE, SOURCE, old, baseline, extract
        )
        is None
    )


def test_snapshot_rejects_different_normal_output_metadata(capture_inputs):
    baseline, _, normal, _, target = capture_inputs
    changed = (
        replace(normal[0], has_rate_context=not normal[0].has_rate_context),
        *normal[1:],
    )
    assert (
        AUDIT.baseline_grouped_capture_snapshot(
            target, SOURCE, SOURCE, baseline, lambda _text: changed
        )
        is None
    )


@pytest.mark.parametrize("location", ("year-prefix", "target-overlap"))
def test_snapshot_candidates_still_require_complete_disjoint_operands(
    capture_inputs, location
):
    baseline, extract, _, _, target = capture_inputs
    snapshot = AUDIT.baseline_grouped_capture_snapshot(
        target, SOURCE, SOURCE, baseline, extract
    )
    assert snapshot is not None
    candidate = dict(snapshot["occurrences"][0])
    start = (
        SOURCE.index("1986") if location == "year-prefix" else target["source_start"]
    )
    candidate.update(
        raw="1",
        value=1,
        source_value=1,
        start=start,
        end=start + 1,
        source_start=start,
        source_end=start + 1,
    )

    def unexpected_replay(_text):
        pytest.fail("partial or overlapping operands must be rejected before replay")

    assert (
        AUDIT._cross_occurrence_restoration_from_candidates(
            target,
            SOURCE,
            SOURCE,
            (candidate,),
            baseline,
            unexpected_replay,
            stage_snapshot=snapshot,
        )
        is None
    )
