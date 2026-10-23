from __future__ import annotations

import pytest

from axiom_encode.failure_classes import (
    FAILURE_CLASSES,
    FailureClassRecord,
    classify_outcome,
    failure_class_payload,
)
from axiom_encode.harness.pre_encode_gate import GateSource, evaluate_pre_encode_gate


def test_the_ten_v1_classes_and_their_subclasses():
    assert FAILURE_CLASSES == {
        "encoder": ("omission", "formula", "test", "harness"),
        "validator_fp": (),
        "corpus": ("content", "metadata"),
        "dependency": (),
        "selection": (),
        "engine": (),
        "migration": ("schema", "elided_excerpt"),
        "process": (),
        "infra": (),
        "unknown": (),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_class": "nope"},
        {"failure_class": "encoder", "subclass": "schema"},
        {"source": "guess"},
        {"confidence": "maybe"},
    ],
)
def test_record_rejects_values_outside_the_taxonomy(kwargs):
    fields = {
        "failure_class": "unknown",
        "subclass": None,
        "source": "rule",
        "confidence": "certain",
        **kwargs,
    }
    with pytest.raises(ValueError):
        FailureClassRecord(**fields)


def test_record_is_frozen_and_serializes():
    record = FailureClassRecord("encoder", "harness", "rule", "certain", ("x",))

    with pytest.raises(AttributeError):
        record.failure_class = "infra"  # type: ignore[misc]
    assert record.to_dict() == {
        "schema": "axiom-encode/failure-class/v1",
        "class": "encoder",
        "subclass": "harness",
        "source": "rule",
        "confidence": "certain",
        "evidence": ["x"],
    }


def test_success_has_no_failure_class():
    assert classify_outcome({"status": "apply_applied", "final_success": True}) is None
    assert failure_class_payload({"final_success": True}) is None


def test_pre_encode_block_takes_the_enforced_finding_class():
    gate = evaluate_pre_encode_gate(
        GateSource(requested="us/x/y/z", citation_path="us/x/y/z", body="")
    )
    record = classify_outcome(
        {
            "status": "pre_encode_blocked",
            "final_success": False,
            "pre_encode_gate": gate.to_dict(),
        }
    )

    assert record == FailureClassRecord(
        "corpus",
        "content",
        "rule",
        "certain",
        ("status:pre_encode_blocked", "pre_encode_gate:no_operative_text"),
    )


def test_pre_encode_block_on_a_heading_only_body_is_selection():
    gate = evaluate_pre_encode_gate(
        GateSource(
            requested="us/x/y/z",
            citation_path="us/x/y/z",
            body="Eligibility Standards",
            labels=("Eligibility Standards",),
        )
    )
    record = classify_outcome(
        {
            "status": "pre_encode_blocked",
            "final_success": False,
            "pre_encode_gate": gate.to_dict(),
        }
    )

    assert (record.failure_class, record.subclass) == ("selection", None)


def test_pre_encode_block_without_an_enforced_finding_is_unknown():
    record = classify_outcome(
        {
            "status": "pre_encode_blocked",
            "final_success": False,
            "pre_encode_gate": {"findings": [{"check": "x", "enforce": False}]},
        }
    )

    assert record.failure_class == "unknown"


def test_report_findings_never_classify_a_later_failure():
    gate = evaluate_pre_encode_gate(
        GateSource(requested="us/x/y/z", citation_path="us/x/y/z", body=""),
        mode="report",
    )
    record = classify_outcome(
        {
            "status": "standalone_failed",
            "final_success": False,
            "pre_encode_gate": gate.to_dict(),
        }
    )

    assert record.failure_class == "unknown"


@pytest.mark.parametrize(
    "outcome",
    [
        {"status": "standalone_failed", "timed_out": True},
        {"status": "standalone_failed", "failure_kind": "timeout"},
    ],
)
def test_timeouts_are_infra(outcome):
    assert classify_outcome(outcome).failure_class == "infra"


@pytest.mark.parametrize(
    ("key", "detail"),
    [
        ("apply_error", "[generated-yaml:io] unable to read generated RuleSpec"),
        ("primary_error", "[generated-yaml:decode] generated RuleSpec text is not"),
    ],
)
def test_unreadable_generated_yaml_is_an_encoder_harness_failure(key, detail):
    record = classify_outcome(
        {"status": "apply_blocked_validation", "final_success": False, key: detail}
    )

    assert (record.failure_class, record.subclass) == ("encoder", "harness")
    assert record.confidence == "certain"
    assert detail in record.evidence


@pytest.mark.parametrize(
    "detail",
    [
        "[generated-yaml:resource] generated YAML nesting exceeds safe parser capacity",
        "[complete-source-unit:tests] branch has no witnessing test",
        "Generated RuleSpec failed CI validation",
        "compile failed: unknown function",
        "validator false positive suspected",
        "corpus row is missing its heading",
    ],
)
def test_issue_text_alone_is_unknown(detail):
    record = classify_outcome(
        {"status": "standalone_failed", "final_success": False, "primary_error": detail}
    )

    assert record.failure_class == "unknown"
    assert record.source == "rule"


def test_non_mapping_outcome_is_unknown():
    assert classify_outcome(None).failure_class == "unknown"  # type: ignore[arg-type]
