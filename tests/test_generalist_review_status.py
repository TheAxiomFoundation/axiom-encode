"""Reviewer availability is durable telemetry, separate from encode acceptance."""

from types import SimpleNamespace

from axiom_encode.cli import (
    _initial_encode_outcome,
    _log_eval_result,
    _print_eval_metrics,
    _record_encode_outcome,
)
from axiom_encode.harness.encoding_db import EncodingDB
from axiom_encode.harness.evals import EvalArtifactMetrics


def metrics(**overrides):
    values = dict(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        embedded_source_present=True,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
        generalist_review_pass=False,
        generalist_review_score=None,
        generalist_review_issues=["Reviewer response could not be parsed"],
        generalist_review_prompt_sha256="a" * 64,
    )
    values.update(overrides)
    return EvalArtifactMetrics(**values)


def result(review_metrics):
    return SimpleNamespace(
        citation="us/statute/99/1",
        backend="codex",
        model="synthetic",
        runner="synthetic",
        mode="source",
        output_file="",
        trace_file="",
        context_manifest_file="",
        success=True,
        error=None,
        metrics=review_metrics,
    )


def test_unscored_review_failure_survives_successful_encode_db_outcome(tmp_path):
    evaluated = result(metrics())
    db_path = tmp_path / "encodings.db"
    run = _log_eval_result(evaluated, db_path=db_path, end_session=False)
    outcome = _initial_encode_outcome(evaluated, apply_requested=False)
    _record_encode_outcome(db_path=db_path, result=evaluated, run=run, outcome=outcome)
    saved = EncodingDB(db_path).get_run(run.id)
    assert saved.success is True  # Advisory review does not change acceptance.
    review = saved.outcome["generalist_review"]
    assert review["status"] == "unavailable"
    assert review["passed"] is False
    assert review["score"] is None
    assert review["issues"] == ["Reviewer response could not be parsed"]
    assert review["prompt_sha256"] == "a" * 64
    assert saved.review_results.oracle_context["generalist_review"] == review


def test_unscored_review_failure_is_visible_in_console(capsys):
    _print_eval_metrics(result(metrics()))
    console = capsys.readouterr().out
    assert "generalist_review=unavailable" in console
    assert "Reviewer response could not be parsed" in console
    assert "prompt_sha256=" + "a" * 64 in console
    assert "generalist_review=yes" not in console


def evaluated_metrics(tmp_path, monkeypatch, *, reviewer=None, **options):
    """Exercise real metric collection with synthetic, isolated validators."""
    from contextlib import nullcontext
    from unittest.mock import Mock

    from axiom_encode.harness import evals
    from axiom_encode.harness.validator_pipeline import ValidationResult

    artifact = tmp_path / "generated/statutes/example.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        "format: rulespec/v1\nmodule:\n  summary: Synthetic review.\nrules: []\n"
    )
    root = tmp_path / "rulespec-us/us"
    root.mkdir(parents=True)
    ci_pass = options.pop("ci_pass", True)
    pipeline = SimpleNamespace(
        _run_compile_check=Mock(return_value=ValidationResult("compile", True)),
        _run_ci=Mock(return_value=ValidationResult("ci", ci_pass)),
        _run_reviewer=Mock(return_value=reviewer),
        _numeric_source_texts_for_rulespec_content=Mock(return_value={}),
    )
    monkeypatch.setattr(evals, "ValidatorPipeline", lambda **kw: pipeline)
    monkeypatch.setattr(
        evals,
        "_rulespec_validation_target",
        lambda *a, **kw: nullcontext(artifact),
    )
    monkeypatch.setattr(evals, "_validation_policy_repo_root", lambda *a: root)
    monkeypatch.setattr(evals, "_validation_rulespec_dependency_roots", lambda **kw: ())
    measured = evals._evaluate_artifact_in_scope(
        rulespec_file=artifact,
        policy_repo_root=root,
        axiom_rules_path=tmp_path / "engine",
        source_text="Synthetic review source.",
        local_corpus_release=None,
        **options,
    )
    return measured, pipeline


import pytest


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"skip_reviewers": True}, "user_requested"),
        (
            {
                "skip_reviewers": True,
                "reviewer_skip_reason": "retained_candidate_preflight",
            },
            "retained_candidate_preflight",
        ),
        (
            {"skip_reviewers": True, "reviewer_skip_reason": "persisted_revalidation"},
            "persisted_revalidation",
        ),
        (
            {"reviewers_require_deterministic_pass": True, "ci_pass": False},
            "deterministic_rejection",
        ),
    ],
)
def test_explicit_skip_has_reason_without_approval_or_prompt(
    tmp_path, monkeypatch, capsys, options, reason
):
    from axiom_encode.harness.evals import generalist_review_snapshot

    measured, pipeline = evaluated_metrics(tmp_path, monkeypatch, **options)
    pipeline._run_reviewer.assert_not_called()
    assert measured.generalist_review_pass is None
    assert generalist_review_snapshot(measured) == {
        "status": "skipped",
        "passed": None,
        "score": None,
        "issues": [],
        "prompt_sha256": None,
        "skip_reason": reason,
    }
    _print_eval_metrics(result(measured))
    console = capsys.readouterr().out
    assert "generalist_review=skipped" in console
    assert f"generalist_review_skip_reason={reason}" in console
    assert "prompt_sha256=" not in console


@pytest.mark.parametrize(
    "passed,score,status",
    [(True, 9.0, "passed"), (False, 4.5, "failed"), (False, None, "unavailable")],
)
def test_actual_reviewer_metric_collection_retains_scored_and_unscored_evidence(
    tmp_path, monkeypatch, capsys, passed, score, status
):
    from axiom_encode.cli import _review_results_from_eval_metrics
    from axiom_encode.harness.evals import _eval_artifact_validation_error
    from axiom_encode.harness.validator_pipeline import ValidationResult

    reviewer = ValidationResult(
        "generalist-reviewer",
        passed,
        score=score,
        issues=["Synthetic reviewer finding"],
        prompt_sha256="b" * 64,
    )
    measured, pipeline = evaluated_metrics(tmp_path, monkeypatch, reviewer=reviewer)
    pipeline._run_reviewer.assert_called_once()
    assert measured.generalist_review_status == status
    assert measured.generalist_review_pass is passed
    assert measured.generalist_review_score == score
    assert measured.generalist_review_prompt_sha256 == "b" * 64
    assert measured.generalist_review_issues == reviewer.issues
    assert _eval_artifact_validation_error(measured) is None
    reviews = _review_results_from_eval_metrics(measured)
    assert reviews.passed is passed
    integration = [r for r in reviews.reviews if r.reviewer == "integration_reviewer"]
    if score is None:
        assert integration == []  # No synthetic checklist score.
    else:
        assert integration[0].items_checked == 10
        assert integration[0].items_passed == round(score)
    _print_eval_metrics(result(measured))
    console = capsys.readouterr().out
    if score is not None:
        assert (
            f"generalist_review={'yes' if passed else 'no'} score={score:.1f}/10"
            in console
        )


def test_skipped_and_unavailable_review_survive_live_and_db_backfill(tmp_path):
    from axiom_encode.run_log import StageStatus
    from axiom_encode.run_log_export import (
        _gate_events_from_metrics,
        synthesize_backfill_events,
    )

    for status, expected in [
        ("unavailable", StageStatus.error),
        ("skipped", StageStatus.skipped),
    ]:
        measured = (
            metrics()
            if status == "unavailable"
            else metrics(
                generalist_review_pass=None,
                generalist_review_issues=[],
                generalist_review_prompt_sha256=None,
                generalist_review_status="skipped",
                generalist_review_skip_reason="user_requested",
            )
        )
        evaluated = result(measured)
        db_path = tmp_path / f"{status}.db"
        run = _log_eval_result(evaluated, db_path=db_path)
        _record_encode_outcome(
            db_path=db_path,
            result=evaluated,
            run=run,
            outcome=_initial_encode_outcome(evaluated, apply_requested=False),
        )
        saved = EncodingDB(db_path).get_run(run.id)
        assert saved.success is True
        assert saved.review_results.passed is False
        assert saved.outcome["generalist_review"]["status"] == status
        live = next(
            e
            for e in _gate_events_from_metrics(measured)
            if e["stage"] == "gate.review"
        )
        backfill = next(
            e
            for e in synthesize_backfill_events(saved, None)
            if e.stage == "gate.review"
        )
        assert live["status"] == backfill.status == expected
        assert live["attrs"] == backfill.attrs
        assert [
            finding.message for finding in live["findings"]
        ] == measured.generalist_review_issues


def test_legacy_missing_status_is_not_inferred_as_approval():
    from axiom_encode.harness.evals import generalist_review_snapshot

    ambiguous = metrics(
        generalist_review_pass=True,
        generalist_review_score=None,
        generalist_review_issues=[],
        generalist_review_prompt_sha256=None,
    )
    assert generalist_review_snapshot(ambiguous)["status"] == "not_recorded"
    assert generalist_review_snapshot(ambiguous)["passed"] is None
    assert generalist_review_snapshot(metrics())["status"] == "unavailable"


@pytest.mark.parametrize("recorded", [True, False])
def test_persisted_eval_payload_restores_status_with_legacy_compatibility(
    recorded, tmp_path
):
    from axiom_encode.harness import evals

    measured = metrics(generalist_review_status="unavailable")
    payload = bound_payload(measured, tmp_path)
    if not recorded:
        payload["metrics"].pop("generalist_review_status")
        payload["metrics"].pop("generalist_review_skip_reason")
    restored = evals._eval_result_from_payload(evals._bind_eval_result_payload(payload))
    assert restored.metrics.generalist_review_status == (
        "unavailable" if recorded else None
    )
    assert evals.generalist_review_snapshot(restored.metrics)["status"] == "unavailable"


def test_reviewer_metadata_does_not_affect_deterministic_revalidation():
    from axiom_encode.harness.evals import _reviewer_independent_metrics

    old = metrics()
    skipped = metrics(
        generalist_review_pass=None,
        generalist_review_score=None,
        generalist_review_issues=[],
        generalist_review_prompt_sha256=None,
        generalist_review_status="skipped",
        generalist_review_skip_reason="persisted_revalidation",
    )
    assert _reviewer_independent_metrics(old) == _reviewer_independent_metrics(skipped)


def bound_payload(measured, tmp_path):
    from dataclasses import asdict
    from hashlib import sha256

    payload = {
        "success": True,
        "metrics": asdict(measured),
        "admission": {"synthetic": True},
    }
    for path_field, digest_field in [
        ("output_file", "generated_output_sha256"),
        ("trace_file", "trace_sha256"),
        ("context_manifest_file", "context_manifest_sha256"),
    ]:
        file = tmp_path / path_field
        file.write_text("synthetic review-status fixture")
        payload[path_field] = str(file)
        payload[digest_field] = sha256(file.read_bytes()).hexdigest()
    return payload


def test_historical_authenticated_metric_shape_survives_roundtrip(tmp_path):
    from axiom_encode.harness import evals

    original = evals._eval_result_from_payload(
        evals._bind_eval_result_payload(bound_payload(metrics(), tmp_path))
    )
    historical = original.to_dict()
    assert "generalist_review_status" not in historical["metrics"]
    assert "generalist_review_skip_reason" not in historical["metrics"]
    restored = evals._eval_result_from_payload(historical)
    assert restored.to_dict() == historical
    expected = evals._eval_result_verdict_evidence_payload(original, original.admission)
    actual = evals._eval_result_verdict_evidence_payload(restored, restored.admission)
    assert actual == expected
    assert actual["validation"]["metrics"] == historical["metrics"]


def test_legacy_null_review_context_keeps_reporting_behavior():
    from axiom_encode.harness.encoding_db import (
        EncodingRun,
        ReviewResult,
        ReviewResults,
    )
    from axiom_encode.run_log import StageStatus
    from axiom_encode.run_log_export import synthesize_backfill_events

    legacy = ReviewResults(
        reviews=[ReviewResult(reviewer="legacy", passed=True)], oracle_context=None
    )
    assert legacy.passed is True
    run = EncodingRun(review_results=legacy)
    review = next(
        event
        for event in synthesize_backfill_events(run, None)
        if event.stage == "gate.review"
    )
    assert review.status == StageStatus.passed
