from __future__ import annotations

import pytest

from axiom_encode.failure_classes import (
    FAILURE_CLASSES,
    FailureClassRecord,
    classify_issue,
    classify_outcome,
    failure_class_payload,
    retry_stop_record,
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


ROOT_LAYOUT = (
    "Axiom rules engine compile failed: invalid RuleSpec repository root "
    "configuration: root-level `programs/` is forbidden in "
    "`/tmp/tmpd78gxt4f/rulespec-us`; content belongs below a direct "
    "jurisdiction directory"
)
EMPTY_BODY = (
    "[complete-source-unit:authoritative-source] The authoritative corpus body "
    "is required; `module.summary` is never completeness evidence."
)
CONTRACT_CITATION = (
    "[required-deferred-output-contract] generated citation does not match "
    "the signed dispatch review contract"
)
CONTRACT_PATH = (
    "[required-deferred-output-contract] generated RuleSpec path does not "
    "match the signed dispatch review contract"
)
ENCODER_ISSUE = (
    "statutes/7/2015/f.yaml: ci: [complete-source-unit:tests] Source-stated "
    "exceptions or applicability conditions require paired positive/blocking cases"
)


@pytest.mark.parametrize(
    ("issue", "expected"),
    [
        (ROOT_LAYOUT, ("migration", "schema")),
        (f"statutes/7/2015/f.yaml: compile: {ROOT_LAYOUT}", ("migration", "schema")),
        (f"statutes/7/2015/f.yaml: ci: {ROOT_LAYOUT}", ("migration", "schema")),
        (EMPTY_BODY, ("corpus", "content")),
        (f"statutes/26/1.yaml: ci: {EMPTY_BODY}", ("corpus", "content")),
        (CONTRACT_CITATION, ("process", None)),
        (f"statutes/26/1.yaml: {CONTRACT_PATH}", ("process", None)),
    ],
)
def test_classify_issue_maps_only_certain_non_encoder_signals(issue, expected):
    record = classify_issue(issue)
    assert (record.failure_class, record.subclass) == expected
    assert (record.source, record.confidence) == ("rule", "certain")
    assert record.evidence and record.evidence[0].startswith("issue:")


@pytest.mark.parametrize(
    "issue",
    [
        ENCODER_ISSUE,
        # A compile error in an imported legacy module: a later candidate in
        # the recorded data stopped importing it and compiled.
        "statutes/42/402/a.yaml: compile: Axiom rules engine compile failed: "
        "failed to load RuleSpec module `/tmp/x/rulespec-us/us/statutes/42/402/a.yaml`: "
        "RuleSpec module `us:statutes/42/416/l` declares removed plural "
        "`corpus_citation_paths`",
        "Axiom rules engine compile failed: invalid RuleSpec repository root "
        "configuration: root `/tmp/x/rulespec-us` is an alias",
        "[required-deferred-output-contract] expected exactly one "
        "module.deferred_outputs entry with output `us:x#y`; found 0",
        "[required-test-case-contract] expected exactly one companion case named 'a'",
        "[temporal-dependency-coverage] Derived rule `a` version 1 covers ...",
        "[generated-yaml:io] unable to read generated RuleSpec",
        "Retained repair candidate is unusable; emit a complete replacement: x",
        # The rule text must start the issue body, not occur inside it.
        f"Test case `a` failed: {ROOT_LAYOUT}",
        "",
        None,
    ],
)
def test_classify_issue_leaves_everything_else_unknown(issue):
    record = classify_issue(issue)
    assert record.failure_class == "unknown"


def test_retry_stops_only_when_every_issue_is_certain_non_encoder():
    record = retry_stop_record(
        (
            f"statutes/7/2015/f.yaml: compile: {ROOT_LAYOUT}",
            f"statutes/7/2015/f.yaml: ci: {ROOT_LAYOUT}",
        )
    )
    assert record is not None
    assert record["schema"] == "axiom-encode/retry-stopped/v1"
    assert (record["class"], record["subclass"]) == ("migration", "schema")
    assert record["issue_count"] == 2
    assert "migration/schema" in record["reason"]
    assert record["evidence"][0] == "issue:rulespec-root-layout"


@pytest.mark.parametrize(
    "issues",
    [
        (),
        [],
        (ENCODER_ISSUE,),
        (ROOT_LAYOUT, ENCODER_ISSUE),
        (ENCODER_ISSUE, EMPTY_BODY),
        (ROOT_LAYOUT, "validator rejected generated section"),
        (ROOT_LAYOUT, None),
        ROOT_LAYOUT,
        None,
    ],
)
def test_any_uncertain_or_missing_issue_keeps_the_retry(issues):
    assert retry_stop_record(issues) is None


def test_mixed_non_encoder_classes_still_stop():
    record = retry_stop_record((CONTRACT_CITATION, EMPTY_BODY))
    assert record is not None
    assert (record["class"], record["subclass"]) == ("process", None)
    assert "process, corpus/content" in record["reason"]


def test_failure_class_carries_retry_stopped():
    stopped = retry_stop_record((ROOT_LAYOUT,))
    payload = failure_class_payload(
        {
            "final_success": False,
            "status": "apply_blocked_validation",
            "retry_stopped": {**stopped, "attempt": 1, "next_model": "m"},
        }
    )
    assert payload["class"] == "migration"
    assert payload["subclass"] == "schema"
    assert payload["confidence"] == "certain"
    assert payload["evidence"][:2] == [
        "status:apply_blocked_validation",
        "retry_stopped",
    ]


@pytest.mark.parametrize(
    "stopped",
    [
        {"class": "encoder", "subclass": None},
        {"class": "unknown", "subclass": None},
        {"class": "migration", "subclass": "nope"},
        "migration",
    ],
)
def test_invalid_retry_stopped_never_classifies(stopped):
    record = classify_outcome(
        {
            "final_success": False,
            "status": "standalone_failed",
            "retry_stopped": stopped,
        }
    )
    assert record.failure_class == "unknown"


def test_success_ignores_retry_stopped():
    assert (
        classify_outcome(
            {"final_success": True, "retry_stopped": {"class": "migration"}}
        )
        is None
    )
