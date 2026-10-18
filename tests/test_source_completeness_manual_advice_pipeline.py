"""Manual advice reaches existing repairs without participating in any gate."""

import contextlib
import subprocess
from dataclasses import asdict

import pytest

from axiom_encode import cli
from axiom_encode.harness import evals, validator_pipeline
from axiom_encode.harness.evals import EvalArtifactMetrics, EvalResult
from axiom_encode.harness.source_completeness import manual_outcome_exception_advice
from axiom_encode.harness.validator_pipeline import ValidationResult, ValidatorPipeline

CITATION = "us-or/manual/odhs/open/page-644"
SOURCE = "Households are eligible unless a member is disqualified."
CONTENT = f"""format: rulespec/v1
module:
  status: deferred
  summary: An explicitly deferred source unit.
  source_verification:
    corpus_citation_path: {CITATION}
rules: []
"""


def _pipeline(tmp_path, *, citation=CITATION):
    policy_repo = tmp_path / "rulespec-us" / "us-or"
    policy_repo.mkdir(parents=True, exist_ok=True)
    return ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        source_citation_path=citation,
        enable_oracles=False,
        enforce_repository_layout=False,
    )


def _rules_file(pipeline):
    rules_file = pipeline.policy_repo_path / "policies/guidance/candidate.yaml"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(CONTENT)
    return rules_file


def _metrics(advice=(), *, issues=()):
    return EvalArtifactMetrics(
        compile_pass=True,
        compile_issues=[],
        ci_pass=not issues,
        ci_issues=list(issues),
        embedded_source_present=False,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
        manual_outcome_exception_advice=list(advice),
    )


def _result(metrics):
    return EvalResult(
        citation=CITATION,
        runner="codex:test",
        backend="codex",
        model="test",
        mode="cold",
        output_file="candidate.yaml",
        trace_file="candidate.trace.json",
        context_manifest_file="candidate.context.json",
        generated_output_sha256="a" * 64,
        trace_sha256="b" * 64,
        context_manifest_sha256="c" * 64,
        duration_ms=0,
        success=True,
        error=None,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        reasoning_output_tokens=0,
        estimated_cost_usd=None,
        actual_cost_usd=None,
        retrieved_files=[],
        unexpected_accesses=[],
        metrics=metrics,
    )


@pytest.mark.parametrize("compile_pass", [True, False])
def test_ci_advice_preserves_pass_fail_and_all_blocking_issues(
    tmp_path, monkeypatch, compile_pass
):
    pipeline = _pipeline(tmp_path)
    rules_file = _rules_file(pipeline)
    monkeypatch.setattr(
        validator_pipeline, "_fetch_corpus_source_text", lambda _citation: SOURCE
    )
    monkeypatch.setattr(
        pipeline,
        "_source_texts_for_rulespec_content",
        lambda _content: {CITATION: SOURCE},
    )
    monkeypatch.setattr(
        pipeline,
        "_compile_rulespec_to_artifact",
        lambda _path, _output: (
            subprocess.CompletedProcess(
                ["test-engine"], 0 if compile_pass else 1, "", "compile failure"
            ),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        ),
    )
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            validator_pipeline, "manual_outcome_exception_advice", lambda *_a, **_k: ()
        )
        baseline = pipeline._run_rulespec_ci(rules_file)
    advised = pipeline._run_rulespec_ci(rules_file)

    assert advised.passed == baseline.passed == compile_pass
    assert advised.issues == baseline.issues
    assert advised.error == baseline.error
    assert advised.raw_output == baseline.raw_output
    assert advised.details["manual_outcome_exception_advice"] == list(
        manual_outcome_exception_advice(SOURCE, corpus_citation_path=CITATION)
    )
    assert advised.details["manual_outcome_exception_advice"]


def test_advice_uses_trusted_requested_source_over_model_metadata(tmp_path):
    pipeline = _pipeline(tmp_path)
    declared = "us-ut/manual/dws/eligibility/770-1"
    advice = pipeline._manual_outcome_exception_advice(
        CONTENT.replace(CITATION, declared),
        validation_source_texts={CITATION: SOURCE, declared: "Update the address."},
    )
    assert advice == manual_outcome_exception_advice(
        SOURCE, corpus_citation_path=CITATION
    )


def test_nonmanual_advice_never_resolves_an_extra_source(tmp_path, monkeypatch):
    pipeline = _pipeline(tmp_path, citation="us/statute/7/2014")

    def unexpected_source_lookup(**_kwargs):
        pytest.fail("Optional manual advice resolved a nonmanual source")

    monkeypatch.setattr(
        validator_pipeline, "_source_verification_text", unexpected_source_lookup
    )
    assert not pipeline._manual_outcome_exception_advice(
        CONTENT, validation_source_texts={CITATION: SOURCE}
    )


@pytest.mark.parametrize("failure_stage", ["source_lookup", "advice_parser"])
def test_optional_advice_failure_cannot_become_a_gate_failure(
    tmp_path, monkeypatch, failure_stage
):
    pipeline = _pipeline(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("optional advice unavailable")

    monkeypatch.setattr(
        validator_pipeline,
        "_source_verification_text"
        if failure_stage == "source_lookup"
        else "manual_outcome_exception_advice",
        unavailable,
    )
    assert not pipeline._manual_outcome_exception_advice(
        CONTENT, validation_source_texts={CITATION: SOURCE}
    )


def test_evaluation_carries_advice_to_result_json_and_round_trips(
    tmp_path, monkeypatch
):
    advice = list(
        manual_outcome_exception_advice(SOURCE, corpus_citation_path=CITATION)
    )
    pipeline = _pipeline(tmp_path)
    rules_file = _rules_file(pipeline)
    monkeypatch.setattr(
        validator_pipeline, "_fetch_corpus_source_text", lambda _citation: SOURCE
    )
    monkeypatch.setattr(evals, "ValidatorPipeline", lambda **_kwargs: pipeline)
    monkeypatch.setattr(
        evals,
        "_rulespec_validation_target",
        lambda *_args, **_kwargs: contextlib.nullcontext(rules_file),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_compile_check",
        lambda _path: ValidationResult("compile", passed=True),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_ci",
        lambda _path: ValidationResult(
            "ci", passed=True, details={"manual_outcome_exception_advice": advice}
        ),
    )
    metrics = evals._evaluate_artifact_in_scope(
        rulespec_file=rules_file,
        policy_repo_root=pipeline.policy_repo_path,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        source_text=SOURCE,
        source_citation_path=CITATION,
        local_corpus_release=None,
        skip_reviewers=True,
    )
    assert metrics.ci_pass
    assert metrics.ci_issues == []
    assert metrics.manual_outcome_exception_advice == advice

    result = _result(metrics)
    payload = result.to_dict()
    assert payload["metrics"]["manual_outcome_exception_advice"] == advice
    restored = evals._eval_result_from_payload(payload)
    assert asdict(restored.metrics) == asdict(metrics)
    assert restored.success
    assert not cli._encode_attempt_was_validator_rejected(restored, {})
    assert not cli._encode_attempt_was_validator_rejected(
        restored, {"apply_requested": True, "status": "applied"}
    )


def test_selected_failed_candidate_advice_reaches_optional_repair_section():
    advice = manual_outcome_exception_advice(SOURCE, corpus_citation_path=CITATION)
    old = cli._FailedEncodeAttempt(
        result=_result(_metrics(advice, issues=("missing companion case",))),
        error="Generated RuleSpec failed CI validation",
        validation_issues=("missing companion case",),
        validation_issue_count=1,
        full_validation_issues=("missing companion case",),
    )
    latest = cli._FailedEncodeAttempt(
        result=_result(_metrics()),
        error="Generated RuleSpec failed CI validation",
        validation_issues=("regression one", "regression two"),
        validation_issue_count=2,
        full_validation_issues=("regression one", "regression two"),
    )
    feedback = cli._encode_validation_retry_feedback([old, latest])
    assert feedback == (old.error, *old.validation_issues, advice[0]["message"])
    assert old.full_validation_issues == ("missing companion case",)
    assert cli._best_validation_retry_attempt([old, latest]) is old

    rendered = evals._format_validation_retry_feedback(feedback)
    required, optional = rendered.split("Optional source advice", 1)
    assert "missing companion case" in required
    assert SOURCE not in required
    assert SOURCE in optional
    assert "not required" in optional
    assert "Correct every listed issue" not in optional


def test_empty_advice_preserves_existing_serialized_metrics_shape():
    result = _result(_metrics())
    payload = result.to_dict()
    assert "manual_outcome_exception_advice" not in payload["metrics"]
    restored = evals._eval_result_from_payload(payload)
    assert restored.metrics.manual_outcome_exception_advice == []


def test_advice_never_displaces_existing_bounded_blocking_feedback():
    issues = tuple(f"blocking issue {index}" for index in range(20))
    result = _result(_metrics())
    failure = cli._FailedEncodeAttempt(
        result=result,
        error="Generated RuleSpec failed CI validation",
        validation_issues=issues,
    )
    baseline = cli._encode_validation_retry_feedback([failure])
    result.metrics.manual_outcome_exception_advice = list(
        manual_outcome_exception_advice(SOURCE, corpus_citation_path=CITATION)
    )
    assert cli._encode_validation_retry_feedback([failure]) == baseline
    assert failure.validation_issues == issues


def test_optional_only_feedback_does_not_request_mandatory_correction():
    advice = manual_outcome_exception_advice(SOURCE, corpus_citation_path=CITATION)
    rendered = evals._format_validation_retry_feedback([advice[0]["message"]])
    assert "Optional source advice" in rendered
    assert "Correct every listed issue" not in rendered
    legacy = "[non-blocking] Existing generalist advice"
    assert (
        legacy
        in evals._format_validation_retry_feedback([legacy]).split(
            "=== BEGIN PRIOR VALIDATION FEEDBACK ===", 1
        )[1]
    )
