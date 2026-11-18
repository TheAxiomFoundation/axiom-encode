"""Private lifecycle tests do not mint materializer/source authority."""

from datetime import date
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_case_custody as c


@pytest.mark.parametrize(
    "left,right",
    [(1, True), (1, 1.0), (date(2025, 1, 1), "2025-01-01"), ([1, 2], [2, 1])],
)
def test_typed_companion_identity(left, right):
    assert c._typed(left) != c._typed(right)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), {1: "value"}])
def test_unsupported_companion_values(value):
    with pytest.raises(ValueError):
        c._typed(value)


def suite(tmp_path):
    owner = SimpleNamespace(case_records=None)
    result = c._Suite(
        owner, 1, object(), tmp_path / "artifact", b"{}", {1: ((), ())}, None, "digest"
    )
    result.fresh = lambda: True
    result.records[1] = c._Record(1, "{}", "{}")
    return result


def test_late_comparison_failure_cannot_be_resealed(tmp_path):
    value = suite(tmp_path)
    value.finish(["expected output mismatch"])
    value.finish([])
    assert value.failed and value.owner.case_records is None


def test_seal_is_one_way_and_invalidation_sticky(tmp_path):
    value = suite(tmp_path)
    value.finish([])
    assert value.owner.case_records
    value.invalidate()
    assert value.owner.case_records is None
    value.finish([])
    assert value.owner.case_records is None


def test_missing_case_prevents_complete_suite_claim(tmp_path):
    value = suite(tmp_path)
    value.expected[2] = ((), ())
    value.finish([])
    assert value.failed and value.owner.case_records is None


@pytest.mark.parametrize("which", ["cases", "compiled"])
def test_late_input_mutation_prevents_sealing(tmp_path, which):
    value = suite(tmp_path)
    value.cases_input = [{"output": {"amount": 10}}]
    value.cases_identity = c._typed(value.cases_input)
    value.compiled_input = {"program": {"derived": []}}
    value.compiled_identity = c._typed(value.compiled_input)
    if which == "cases":
        value.cases_input[0]["output"]["amount"] = 11
    else:
        value.compiled_input["program"]["derived"].append({"name": "invented"})
    value.finish([])
    assert value.failed and value.owner.case_records is None


def sealed_suite(tmp_path):
    value = suite(tmp_path)
    value.owner.receipt = object()
    token = c._current.set(value)
    try:
        value.finish([])
    finally:
        c._current.reset(token)
    assert c._live_seal(value.owner, value.pipeline) is not None
    return value


@pytest.mark.parametrize(
    "mutation", ["records", "receipt", "expected", "failed", "unfinished", "freshness"]
)
def test_immutable_seal_rejects_replaced_state(tmp_path, mutation):
    value = sealed_suite(tmp_path)
    try:
        if mutation == "records":
            value.owner.case_records = tuple(list(value.owner.case_records))
        elif mutation == "receipt":
            value.owner.receipt = object()
        elif mutation == "expected":
            value.expected[2] = ((), ())
        elif mutation == "failed":
            value.failed = True
        elif mutation == "unfinished":
            value.finished = False
        else:
            value.fresh = lambda: False
        assert c._live_seal(value.owner, value.pipeline) is None
    finally:
        c._forget_seal(value.owner)


def test_finish_outside_current_process_suite_does_not_seal(tmp_path):
    value = suite(tmp_path)
    value.finish([])
    assert value.owner.case_records
    assert c._live_seal(value.owner, value.pipeline) is None


def test_late_invalidation_removes_seal(tmp_path):
    value = sealed_suite(tmp_path)
    value.invalidate()
    assert id(value.owner) not in c._sealed
