"""Admission scoring stays separate from the EncodeBench four gates."""

import copy
import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals
from axiom_encode.harness.admission import admission_identity, score_admission
from axiom_encode.harness.eval_board import (
    EvalBoardError,
    eval_board_to_json,
    fold_eval_board,
    render_eval_board_markdown,
    render_eval_board_text,
    result_gate_pass,
)
from tests.eval_evidence_fixtures import install_test_eval_evidence_keys
from tests.test_admission import _context_with_packaged_file, _short_timeout
from tests.test_admission import context as _admission_context
from tests.test_eval_board import (
    CASE_IDENTITIES,
    _admission_comparison_paths,
    _admission_comparison_rows,
    _execution_identity,
    _payload,
    _result,
    _write_payload,
)
from tests.test_evals import (
    _bind_fake_source_results,
    _fake_eval_result,
    _strict_eval_suite_manifest_payload,
    _write_test_corpus_provision,
)


@pytest.fixture
def admission_context(tmp_path, monkeypatch):
    return _admission_context.__wrapped__(tmp_path, monkeypatch)


def _score(*, admitted=False, prerequisite=False, identity=None):
    return {
        "admitted": admitted,
        "compile_pass": admitted,
        "ci_pass": admitted,
        "issues": [] if admitted else ["fixture admission issue"],
        "refusal_categories": [] if admitted or prerequisite else ["ci-other"],
        "issue_categories": [] if admitted else ["ci-other"],
        "prerequisite_failure": prerequisite,
        "prerequisite_categories": ["missing-corpus-text"] if prerequisite else [],
        "failure_kind": (
            "prerequisite" if prerequisite else "candidate" if not admitted else None
        ),
        "identity": identity or _bound_identity("1"),
        "numeric_recall": {"total": 0, "covered": 0, "missing": 0, "percentage": None},
    }


def _bound_identity(encoder_digit):
    return {
        "encoder": {"commit": encoder_digit * 40, "version": "test"},
        "source": {"body_sha256": "4" * 64},
        "context": {
            "context_manifest_sha256": "5" * 64,
            "context_manifest_canonical_sha256": "7" * 64,
        },
    }


def _scored_payload(runner, scores, *, declared=True):
    rows = [_result(runner, case) for case in CASE_IDENTITIES]
    for row, score in zip(rows, scores, strict=True):
        if score is not None:
            row["admission_score"] = score
    execution_identity = _execution_identity()
    if declared:
        execution_identity["admission_score"] = {
            "enabled": True,
            "axiom_compose": None,
        }
    return _payload(
        [(runner, "codex", "gpt-5.6-terra")],
        rows,
        execution_identity=execution_identity,
    )


def test_admission_suite_flag_preserves_seeded_four_gate_rows(tmp_path, monkeypatch):
    install_test_eval_evidence_keys(monkeypatch)
    rng = random.Random(76021)
    generated = [
        {
            "success": rng.choice([True, False]),
            "compile_pass": rng.choice([True, False]),
            "ci_pass": rng.choice([True, False]),
            "ungrounded_numeric_count": rng.randrange(3),
            "total": rng.randrange(10),
        }
        for _ in range(40)
    ]
    payload = _strict_eval_suite_manifest_payload()
    payload["cases"] = [
        {
            "kind": "source",
            "name": f"case-{index}",
            "corpus_citation_path": "us/statute/7/2017",
        }
        for index in range(len(generated))
    ]
    manifest_path = tmp_path / "suite.yaml"
    # JSON is valid YAML and preserves the generated cases verbatim.
    manifest_path.write_text(json.dumps(payload))
    manifest = evals.load_eval_suite_manifest(manifest_path)
    release = _write_test_corpus_provision(tmp_path)
    engine = tmp_path / "axiom-rules-engine"
    engine.mkdir()
    compose = tmp_path / "axiom-compose"
    compose.write_bytes(b"fixture compose executable")
    results_by_mode = []
    calls = []

    def fake_score(result, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: _score())

    for enabled in (False, True):
        pending = iter(generated)

        def fake_run(**kwargs):
            values = next(pending)
            result = _fake_eval_result(
                "openai-gpt-5.4",
                "fixture",
                compile_pass=values["compile_pass"],
                ci_pass=values["ci_pass"],
                ungrounded_numeric_count=values["ungrounded_numeric_count"],
            )
            result.success = values["success"]
            result.error = None if result.success else "seeded validation refusal"
            result.failure_kind = None if result.success else "validation"
            total = values["total"]
            covered = total // 2
            result.metrics.source_numeric_occurrence_count = total
            result.metrics.covered_source_numeric_occurrence_count = covered
            result.metrics.missing_source_numeric_occurrence_count = total - covered
            return _bind_fake_source_results([result], kwargs)

        with (
            patch.object(evals, "run_source_eval", side_effect=fake_run),
            patch(
                "axiom_encode.harness.admission.score_admission", side_effect=fake_score
            ),
        ):
            results_by_mode.append(
                evals.run_eval_suite(
                    manifest,
                    tmp_path / f"out-{enabled}",
                    engine,
                    tmp_path / "rulespec-us",
                    release,
                    suite_retry_attempts=0,
                    admission_score=enabled,
                    axiom_compose_path=compose,
                )
            )

    for baseline, scored in zip(*results_by_mode, strict=True):
        assert (baseline.success, baseline.error) == (scored.success, scored.error)
        assert baseline.metrics == scored.metrics
        assert result_gate_pass(baseline.to_dict()) == result_gate_pass(
            scored.to_dict()
        )
        assert "admission_score" not in baseline.to_dict()
        assert scored.to_dict()["admission_score"] == _score()
        assert scored.admission["schema"] == "axiom-encode/eval-result-admission/v2"
    assert len(calls) == 40
    assert all(call["local_corpus_release"] is release for call in calls)
    assert all(call["axiom_compose_path"] == compose for call in calls)
    assert all(call["citation"] == "us/statute/7/2017" for call in calls)
    state = json.loads((tmp_path / "out-True" / "suite-run.json").read_text())
    assert state["execution_identity"]["admission_score"]["enabled"] is True
    assert (
        state["execution_identity"]["admission_score"]["axiom_compose"]["binary_sha256"]
        == hashlib.sha256(compose.read_bytes()).hexdigest()
    )
    row = json.loads(
        (tmp_path / "out-True" / "suite-results.jsonl").read_text().splitlines()[0]
    )
    verdict = json.loads(Path(row["result"]["verdict_file"]).read_text())
    assert verdict["admission_score"] == row["result"]["admission_score"]


def _run_suite_with_failing_runner(tmp_path, monkeypatch, admission_context, raised):
    """Run a one-case suite whose runner raises, with admission scoring on."""

    install_test_eval_evidence_keys(monkeypatch)
    _, options = admission_context()
    checkout = options["policy_repo_path"].parent
    release = options["local_corpus_release"]
    waiver = b"validate_failures: {}\n"
    (checkout / "known-validation-gaps.yaml").write_bytes(waiver)
    toolchain = checkout / ".axiom/toolchain.toml"
    toolchain.parent.mkdir()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release.name}"\n'
        f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{hashlib.sha256(waiver).hexdigest()}"\n'
    )
    payload = _strict_eval_suite_manifest_payload()
    payload["cases"][0]["corpus_citation_path"] = "us/statute/26/1"
    manifest_path = tmp_path / "empty-suite.yaml"
    manifest_path.write_text(json.dumps(payload))
    manifest = evals.load_eval_suite_manifest(manifest_path)

    def runner_failure(**_kwargs):
        raise raised

    monkeypatch.setattr(evals, "run_source_eval", runner_failure)
    output_root = tmp_path / "empty-suite-output"
    results = evals.run_eval_suite(
        manifest,
        output_root,
        options["axiom_rules_path"],
        checkout,
        release,
        suite_retry_attempts=0,
        admission_score=True,
    )
    assert len(results) == 1
    assert results[0].output_file == ""
    assert results[0].context_manifest_file == ""
    score = results[0].admission_score
    row = json.loads((output_root / "suite-results.jsonl").read_text())
    assert row["result"]["admission_score"] == score
    verdict = json.loads(Path(row["result"]["verdict_file"]).read_text())
    assert verdict["admission_score"] == score
    return results[0], score


@pytest.mark.parametrize(
    "raised,category",
    [
        (RuntimeError("Runner failed before writing a workspace"), "generation-error"),
        (subprocess.TimeoutExpired(["codex"], 5), "generation-timeout"),
    ],
)
def test_admission_suite_runner_failure_is_persisted_as_candidate_failure(
    tmp_path, monkeypatch, admission_context, raised, category
):
    # The backend has credentials, so nothing proves an infrastructure failure.
    monkeypatch.setattr(evals, "codex_auth_error", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    _, score = _run_suite_with_failing_runner(
        tmp_path, monkeypatch, admission_context, raised
    )
    assert score["admitted"] is False
    assert score["failure_kind"] == "candidate"
    assert score["refusal_categories"] == [category]
    assert score["prerequisite_failure"] is False
    assert "context" not in score["identity"]


@pytest.mark.parametrize(
    "credentials,raised,category",
    [
        (
            False,
            RuntimeError("Runner failed before writing a workspace"),
            "generation-authentication",
        ),
        (
            True,
            RuntimeError("Usage limit reached for this account"),
            "generation-quota-exhaustion",
        ),
    ],
)
def test_admission_suite_proven_infrastructure_failure_is_a_prerequisite(
    tmp_path, monkeypatch, admission_context, credentials, raised, category
):
    """The fourth review's probe: infrastructure is not the model's failure."""

    monkeypatch.setattr(
        evals,
        "codex_auth_error",
        lambda: None if credentials else "Codex backend requires authentication",
    )
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    if credentials:
        monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result, score = _run_suite_with_failing_runner(
        tmp_path, monkeypatch, admission_context, raised
    )
    assert score["admitted"] is False
    assert score["failure_kind"] == "prerequisite"
    assert score["prerequisite_failure"] is True
    assert score["prerequisite_categories"] == [category]
    assert score["refusal_categories"] == []
    # The four-gate row is untouched: it still records a failed encode.
    assert result.success is False
    assert result.failure_kind == "error"


def _generation_row(**overrides):
    row = SimpleNamespace(
        success=False,
        output_file="",
        failure_kind="error",
        backend="codex",
        error="Codex eval failed",
        metrics=None,
    )
    for name, value in overrides.items():
        setattr(row, name, value)
    return row


@pytest.mark.parametrize(
    "overrides,codex_credentials,openai_key,expected",
    [
        ({}, False, False, "authentication"),
        ({}, True, False, None),
        ({"backend": "openai"}, True, False, "authentication"),
        ({"backend": "openai"}, False, True, None),
        ({"backend": "claude"}, False, False, None),
        # Reported model tokens show a model was reached, whatever the check says.
        ({"output_tokens": 12}, False, False, None),
        ({"input_tokens": 900}, False, False, None),
        ({"backend": "openai", "reasoning_output_tokens": 3}, True, False, None),
        ({"error": "You have hit your usage limit"}, True, True, "quota-exhaustion"),
        (
            {"backend": "claude", "error": "Usage limit reached"},
            False,
            False,
            "quota-exhaustion",
        ),
        # A row that is not a no-artifact backend error is never infrastructure.
        ({"success": True}, False, False, None),
        ({"output_file": "statutes/26/1.yaml"}, False, False, None),
        ({"failure_kind": "timeout"}, False, False, None),
        ({"failure_kind": "validation"}, False, False, None),
        (
            {"output_file": "statutes/26/1.yaml", "error": "usage limit"},
            True,
            True,
            None,
        ),
    ],
)
def test_admission_generation_infrastructure_needs_harness_evidence(
    monkeypatch, overrides, codex_credentials, openai_key, expected
):
    monkeypatch.setattr(
        evals,
        "codex_auth_error",
        lambda: None if codex_credentials else "Codex backend requires authentication",
    )
    if openai_key:
        monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    row = _generation_row(**overrides)
    assert evals._eval_result_generation_infrastructure_failure(row) == expected
    if expected == "authentication" and row.backend == "codex":
        # Another Codex credential the canonical check does not read.
        monkeypatch.setenv("CODEX_API_KEY", "fixture-key")
        assert evals._eval_result_generation_infrastructure_failure(row) is None


def test_admission_candidate_text_cannot_claim_a_usage_limit(monkeypatch):
    """Metrics issues quote candidate text; with an artifact they prove nothing."""

    monkeypatch.setattr(evals, "codex_auth_error", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    row = _generation_row(
        output_file="statutes/26/1.yaml",
        failure_kind="validation",
        error="Generated RuleSpec failed CI validation",
        metrics=SimpleNamespace(
            compile_issues=[],
            ci_issues=["rule usage limit reached is ungrounded"],
            generalist_review_issues=[],
            policyengine_issues=[],
        ),
    )
    assert evals._eval_result_indicates_usage_limit(row)
    assert evals._eval_result_generation_infrastructure_failure(row) is None


@pytest.mark.parametrize("recorded", ["authentication", "quota-exhaustion", None])
@pytest.mark.parametrize("credentials_now", [False, True])
def test_admission_resume_reuses_the_recorded_infrastructure_class(
    monkeypatch, admission_context, recorded, credentials_now
):
    """Re-scoring a persisted row must not depend on today's environment."""

    result, options = admission_context(backend="codex")
    Path(result.output_file).unlink()
    result.output_file = ""
    result.success = False
    result.failure_kind = "error"
    result.error = "Codex eval failed"
    result.metrics = None
    case = SimpleNamespace(corpus_citation_path="us/statute/26/1", citation=None)
    arguments = {
        "output_root": options["output_root"],
        "policy_repo_path": options["policy_repo_path"],
        "axiom_rules_path": options["axiom_rules_path"],
        "axiom_compose_path": None,
        "corpus_release": options["local_corpus_release"],
    }

    monkeypatch.delenv("CODEX_API_KEY", raising=False)

    def credentials(present):
        monkeypatch.setattr(
            evals,
            "codex_auth_error",
            lambda: None if present else "Codex backend requires authentication",
        )
        if present:
            monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
        else:
            monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Generation time: the environment and the row's error decide the class.
    credentials(recorded != "authentication")
    if recorded == "quota-exhaustion":
        result.error = "Usage limit reached"
    evals._score_eval_suite_case_admission(case, [result], **arguments)
    persisted = copy.deepcopy(result.admission_score)
    expected = [f"generation-{recorded}"] if recorded else []
    assert persisted["prerequisite_categories"] == expected
    assert persisted["refusal_categories"] == ([] if recorded else ["no-artifact"])

    # Resume: whatever the environment is now, the recorded class is reused.
    credentials(credentials_now)
    evals._score_eval_suite_case_admission(
        case, [result], recorded_scores=[persisted], **arguments
    )
    assert result.admission_score == persisted


def test_admission_eval_board_counts_prerequisites_separately(tmp_path):
    payload = _scored_payload(
        "scored",
        [_score(admitted=True), _score(), _score(prerequisite=True)],
    )
    path = _write_payload(tmp_path, "scored.json", payload)
    board = fold_eval_board([path])
    stats = board.runners[0]
    assert stats.gate_pass_count == 3
    assert stats.admission_case_count == 3
    assert stats.admitted_count == 1
    assert stats.admission_refusal_count == 1
    assert stats.admission_prerequisite_count == 1
    assert stats.admission_scorer_error_count == 0
    assert "| gate pass | admission | prerequisites |" in render_eval_board_markdown(
        board
    )
    assert "admission 1/3  prerequisites 1" in render_eval_board_text(board)
    rendered = eval_board_to_json(board)
    assert rendered["admission_scored"] is True
    assert rendered["runners"][0]["admission_prerequisite_count"] == 1


def test_admission_eval_board_counts_scorer_errors_separately(
    tmp_path, monkeypatch, admission_context
):
    from axiom_encode import cli
    from axiom_encode.harness.admission import score_admission

    result, options = admission_context()

    def scorer_error(*_args, **_kwargs):
        raise RuntimeError("injected production validation error")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        scorer_error,
    )
    score = score_admission(
        result,
        **{
            name: value
            for name, value in options.items()
            if name != "require_complete_source_unit"
        },
    ).to_dict()
    assert score["failure_kind"] == "scorer-error"
    assert score["refusal_categories"] == []
    assert score["prerequisite_failure"] is False
    assert score["scorer_error"] == {
        "type": "RuntimeError",
        "message": "injected production validation error",
    }
    payload = _scored_payload(
        "scored",
        [score, _score(), _score(prerequisite=True)],
    )
    board = fold_eval_board([_write_payload(tmp_path, "scored.json", payload)])
    stats = board.runners[0]
    assert stats.gate_pass_count == 3
    assert stats.admission_case_count == 3
    assert stats.admitted_count == 0
    assert stats.admission_refusal_count == 1
    assert stats.admission_prerequisite_count == 1
    assert stats.admission_scorer_error_count == 1
    assert "| prerequisites | scorer errors |" in render_eval_board_markdown(board)
    assert "prerequisites 1  scorer errors 1" in render_eval_board_text(board)
    assert eval_board_to_json(board)["runners"][0]["admission_scorer_error_count"] == 1


@pytest.mark.parametrize("split_payloads", [False, True])
def test_admission_scores_packaged_from_two_checkouts_fold(
    tmp_path, admission_context, split_payloads
):
    """The reviewer's repro, through the real scorer and the real fold."""

    result, options = admission_context()
    options = {
        name: value
        for name, value in options.items()
        if name != "require_complete_source_unit"
    }
    identities = []
    for origin in ("/checkouts/left/statutes/26/2.yaml", "/work/right/26/2.yaml"):
        _context_with_packaged_file(result, origin)
        score = score_admission(result, **options)
        assert score.admitted, score.issues
        identities.append(json.loads(score.to_json())["identity"])
    assert (
        identities[0]["context"]["context_manifest_sha256"]
        != identities[1]["context"]["context_manifest_sha256"]
    )
    runners_and_rows = [
        (runner, _admission_comparison_rows(runner, identity))
        for runner, identity in zip(("left", "right"), identities, strict=True)
    ]
    for directory in ("fold", "refused"):
        (tmp_path / directory).mkdir()
    board = fold_eval_board(
        _admission_comparison_paths(
            tmp_path / "fold", runners_and_rows, split_payloads=split_payloads
        )
    )
    assert [runner.admitted_count for runner in board.runners] == [3, 3]
    payload = json.loads(json.dumps(eval_board_to_json(board)))
    for identity in payload["admission_identities"].values():
        assert "context_manifest_sha256" not in identity["context"]
        assert (
            identity["context"]["context_manifest_canonical_sha256"]
            == identities[0]["context"]["context_manifest_canonical_sha256"]
        )

    # Different packaged bytes are a different exam, and still refuse to fold.
    manifest = Path(result.context_manifest_file)
    (manifest.parent / "context/precedent.yaml").write_text("format: other\n")
    changed = json.loads(score_admission(result, **options).to_json())["identity"]
    with pytest.raises(EvalBoardError, match="admission identity"):
        fold_eval_board(
            _admission_comparison_paths(
                tmp_path / "refused",
                [
                    ("left", _admission_comparison_rows("left", identities[0])),
                    ("changed", _admission_comparison_rows("changed", changed)),
                ],
                split_payloads=split_payloads,
            )
        )


@pytest.mark.parametrize("split_payloads", [False, True])
def test_admission_real_prerequisite_score_folds_beside_an_admitted_one(
    tmp_path, admission_context, split_payloads
):
    """A real missing-manifest score binds no context and must not block a fold."""

    result, options = admission_context()
    options = {
        name: value
        for name, value in options.items()
        if name != "require_complete_source_unit"
    }
    admitted = score_admission(result, **options)
    assert admitted.admitted, admitted.issues
    Path(result.context_manifest_file).unlink()
    excluded = score_admission(result, **options)
    assert excluded.prerequisite_categories == ["missing-context-manifest"]
    assert "context" not in excluded.identity
    excluded_rows = _admission_comparison_rows(
        "excluded", json.loads(excluded.to_json())["identity"]
    )
    for row in excluded_rows:
        row["admission_score"].update(
            admitted=False,
            prerequisite_failure=True,
            failure_kind="prerequisite",
            refusal_categories=[],
        )
    board = fold_eval_board(
        _admission_comparison_paths(
            tmp_path,
            [
                (
                    "regular",
                    _admission_comparison_rows(
                        "regular", json.loads(admitted.to_json())["identity"]
                    ),
                ),
                ("excluded", excluded_rows),
            ],
            split_payloads=split_payloads,
        )
    )
    stats = {runner.runner: runner for runner in board.runners}
    assert stats["regular"].admitted_count == 3
    assert stats["excluded"].admission_prerequisite_count == 3
    json.dumps(eval_board_to_json(board))


def test_admission_eval_board_counts_no_artifact_as_candidate_refusal(tmp_path):
    row = _result(
        "scored",
        CASE_IDENTITIES[0],
        success=False,
        error="Model returned no artifact",
        metrics=None,
        failure_kind="error",
    )
    row["admission_score"] = _score()
    row["admission_score"]["refusal_categories"] = ["no-artifact"]
    rows = [row, *[_result("scored", case) for case in CASE_IDENTITIES[1:]]]
    for admitted_row in rows[1:]:
        admitted_row["admission_score"] = _score(admitted=True)
    execution_identity = _execution_identity()
    execution_identity["admission_score"] = {
        "enabled": True,
        "axiom_compose": None,
    }
    payload = _payload(
        [("scored", "codex", "gpt-5.6-terra")],
        rows,
        execution_identity=execution_identity,
    )
    board = fold_eval_board([_write_payload(tmp_path, "empty.json", payload)])
    stats = board.runners[0]
    assert stats.gate_pass_count == 2
    assert stats.admission_case_count == 3
    assert stats.admitted_count == 2
    assert stats.admission_refusal_count == 1
    assert stats.admission_prerequisite_count == 0
    assert stats.admission_scorer_error_count == 0


def test_admission_eval_board_accepts_bound_compose_executable(tmp_path):
    compose = tmp_path / "axiom-compose"
    compose.write_bytes(b"fixture compose executable")
    execution = _execution_identity()
    execution["admission_score"] = evals._eval_suite_admission_score_options(compose)
    rows = [_result("compose", case) for case in CASE_IDENTITIES]
    for row in rows:
        row["admission_score"] = _score(admitted=True)
    payload = _payload(
        [("compose", "codex", "gpt-5.6-terra")], rows, execution_identity=execution
    )
    board = fold_eval_board([_write_payload(tmp_path, "compose.json", payload)])
    assert board.runners[0].admitted_count == 3


@pytest.mark.parametrize("allow_mixed_toolchains", [False, True])
def test_admission_eval_board_refuses_mixed_scoring(tmp_path, allow_mixed_toolchains):
    scored = _scored_payload("scored", [_score()] * 3)
    plain = _scored_payload("plain", [None] * 3, declared=False)
    paths = [
        _write_payload(tmp_path, "scored.json", scored),
        _write_payload(tmp_path, "plain.json", plain),
    ]
    with pytest.raises(EvalBoardError, match="admission scoring presence"):
        fold_eval_board(paths, allow_mixed_toolchains=allow_mixed_toolchains)


@pytest.mark.parametrize("allow_mixed_toolchains", [False, True])
def test_admission_eval_board_refuses_identity_mismatch(
    tmp_path, allow_mixed_toolchains
):
    other = _score(identity=_bound_identity("2"))
    paths = [
        _write_payload(
            tmp_path, "first.json", _scored_payload("first", [_score()] * 3)
        ),
        _write_payload(tmp_path, "second.json", _scored_payload("second", [other] * 3)),
    ]
    with pytest.raises(EvalBoardError, match="admission identity"):
        fold_eval_board(paths, allow_mixed_toolchains=allow_mixed_toolchains)


def test_admission_eval_board_refuses_partial_scoring_within_run(tmp_path):
    payload = _scored_payload("mixed", [_score(), None, _score()])
    path = _write_payload(tmp_path, "mixed.json", payload)
    with pytest.raises(EvalBoardError, match="admission scoring presence"):
        fold_eval_board([path])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_identity",
        "admitted_prerequisite",
        "admitted_scorer_error",
        "bad_kind",
        "bad_kind_list",
    ],
)
def test_admission_eval_board_refuses_malformed_score(tmp_path, mutation):
    score = copy.deepcopy(_score())
    if mutation == "missing_identity":
        score.pop("identity")
    elif mutation == "admitted_prerequisite":
        score.update(admitted=True, prerequisite_failure=True)
    elif mutation == "admitted_scorer_error":
        score.update(admitted=True, failure_kind="scorer-error")
    elif mutation == "bad_kind_list":
        score["failure_kind"] = []
    else:
        score["failure_kind"] = "infrastructure"
    path = _write_payload(tmp_path, "bad.json", _scored_payload("bad", [score] * 3))
    with pytest.raises(EvalBoardError, match="admission"):
        fold_eval_board([path])


def _context_score(result, options, *, expected_context=None):
    return score_admission(
        result,
        **{
            name: value
            for name, value in options.items()
            if name != "require_complete_source_unit"
        },
        expected_context=expected_context,
    )


def _write_frozen_manifest(result, manifest):
    path = Path(result.context_manifest_file)
    path.write_text(json.dumps(manifest, sort_keys=True))
    result.context_manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "mutation,category",
    [
        ("missing-manifest", "missing-context-manifest"),
        ("unreadable-manifest", "unreadable-context-manifest"),
        ("manifest-hash", "context-manifest-hash-mismatch"),
        ("invalid-manifest", "invalid-context-manifest"),
        ("missing-generation-input", "missing-generation-input"),
        ("missing-row-generation-input", "missing-generation-input"),
        ("unreadable-generation-input", "unreadable-generation-input"),
        ("generation-input-hash", "generation-input-hash-mismatch"),
        ("row-generation-input-hash", "generation-input-hash-mismatch"),
        ("resolved-body-hash", "frozen-source-mismatch"),
        ("rebound-generation-input", "frozen-source-mismatch"),
    ],
)
def test_admission_frozen_context_preflight_refuses_before_production(
    admission_context, backend, mutation, category
):
    result, options = admission_context(backend=backend)
    manifest_path = Path(result.context_manifest_file)
    manifest = json.loads(manifest_path.read_text())
    source_path = manifest_path.parent / manifest["source_text_file"]
    protected = [
        Path(result.output_file),
        cli._rulespec_test_path(Path(result.output_file)),
    ]
    original_bytes = [path.read_bytes() for path in protected]
    unreadable = None
    if mutation == "missing-manifest":
        manifest_path.unlink()
    elif mutation == "unreadable-manifest":
        unreadable = manifest_path
        unreadable.chmod(0)
    elif mutation == "manifest-hash":
        manifest_path.write_text(manifest_path.read_text() + "\n")
    elif mutation == "invalid-manifest":
        _write_frozen_manifest(result, [])
    elif mutation == "missing-generation-input":
        source_path.unlink()
    elif mutation == "missing-row-generation-input":
        result.generation_input_file = str(source_path.with_name("missing-source.txt"))
    elif mutation == "unreadable-generation-input":
        unreadable = source_path
        unreadable.chmod(0)
    elif mutation == "generation-input-hash":
        source_path.write_text("The allowance is $99.")
    elif mutation == "row-generation-input-hash":
        row_source = source_path.with_name("wrong-source.txt")
        row_source.write_text("The allowance is $99.")
        result.generation_input_file = str(row_source)
    elif mutation == "resolved-body-hash":
        result.source_attestation["resolved_text_sha256"] = "f" * 64
        manifest["source_metadata"]["source_attestation"] = result.source_attestation
        _write_frozen_manifest(result, manifest)
    elif mutation == "rebound-generation-input":
        source_path.write_text("The allowance is $99.")
        changed_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest["source_text_sha256"] = changed_digest
        result.generation_input_sha256 = changed_digest
        result.source_attestation["generation_input_sha256"] = changed_digest
        manifest["source_metadata"]["source_attestation"] = result.source_attestation
        _write_frozen_manifest(result, manifest)
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        try:
            score = _context_score(result, options)
        finally:
            if unreadable is not None:
                unreadable.chmod(0o644)
    assert score.admitted is False
    assert score.failure_kind == "prerequisite", score.issues
    assert score.prerequisite_categories == [category]
    assert score.prerequisite_failure is True
    assert score.refusal_categories == []
    production.assert_not_called()
    assert [path.read_bytes() for path in protected] == original_bytes


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "record",
    [
        "source_metadata_file",
        "provision_metadata_file",
        "context_files",
        "review_findings_files",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "unreadable", "hash", "valid"])
def test_admission_frozen_manifest_artifacts_are_preflighted(
    admission_context, backend, record, mutation
):
    result, options = admission_context(backend=backend)
    manifest_path = Path(result.context_manifest_file)
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest_path.parent / "frozen-context.txt"
    artifact.write_text("Frozen generation context\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if record.endswith("_file"):
        manifest[record] = artifact.name
        manifest[record.removesuffix("_file") + "_sha256"] = digest
        identity_key = record
    else:
        manifest[record] = [{"workspace_path": artifact.name, "sha256": digest}]
        identity_key = artifact.name
    _write_frozen_manifest(result, manifest)
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "unreadable":
        artifact.chmod(0)
    elif mutation == "hash":
        artifact.write_text("Different frozen generation context\n")
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        try:
            score = _context_score(result, options)
        finally:
            if mutation == "unreadable":
                artifact.chmod(0o644)
    if mutation == "valid":
        production.assert_called_once()
        assert score.admitted is True, score.issues
        assert score.identity["context"]["artifacts"][identity_key] == digest
    else:
        production.assert_not_called()
        assert score.failure_kind == "prerequisite", score.issues
        category = (
            "context-artifact-hash-mismatch"
            if mutation == "hash"
            else f"{mutation}-context-artifact"
        )
        assert score.prerequisite_categories == [category]


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("source_text_file", 23),
        ("source_metadata_file", {"name": "source-metadata.json"}),
        ("provision_metadata_file", 23),
        ("context_files", "missing context"),
        ("context_files", [None]),
        ("context_files", [{}]),
        ("context_files", [{"workspace_path": []}]),
        ("review_findings_files", {}),
        ("review_findings_files", [{"workspace_path": ""}]),
    ],
)
def test_admission_malformed_frozen_manifest_records_are_prerequisites(
    admission_context, backend, field, value
):
    result, options = admission_context(backend=backend)
    manifest_path = Path(result.context_manifest_file)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    _write_frozen_manifest(result, manifest)
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options)
    production.assert_not_called()
    assert score.failure_kind == "prerequisite", score.issues
    assert score.prerequisite_categories == ["invalid-context-manifest"]


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_review_missing_resolver_manifest_probe_is_prerequisite(
    admission_context, backend
):
    result, options = admission_context(backend=backend)
    Path(result.context_manifest_file).unlink()
    with pytest.raises(
        RuntimeError,
        match="Cannot read explicit source metadata from the eval context manifest",
    ):
        cli._generated_result_source_metadata(result)
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options)
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_categories == ["missing-context-manifest"]
    assert score.refusal_categories == []
    production.assert_not_called()


def _expected_context(result, options):
    identity = admission_identity(
        **{
            name: value
            for name, value in options.items()
            if name
            not in {
                "output_root",
                "local_corpus_release",
                "require_complete_source_unit",
            }
        },
        local_corpus_release=options["local_corpus_release"],
    )
    return {
        "source_body_sha256": result.source_attestation["resolved_text_sha256"],
        "policy_repo_commit": identity["policy_repo"]["commit"],
        "dependency_content_sha256": [
            dependency["content_sha256"] for dependency in identity["dependencies"]
        ],
        "corpus_release_content_sha256": identity["corpus"]["content_sha256"],
    }


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "field",
    [
        "source_body_sha256",
        "policy_repo_commit",
        "dependency_content_sha256",
        "corpus_release_content_sha256",
    ],
)
def test_admission_expected_suite_context_mismatch_is_prerequisite(
    admission_context, backend, field
):
    result, options = admission_context(backend=backend)
    expected = _expected_context(result, options)
    expected[field] = ["f" * 64] if field == "dependency_content_sha256" else "f" * 64
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options, expected_context=expected)
    assert score.failure_kind == "prerequisite", score.issues
    assert score.prerequisite_categories == ["expected-context-mismatch"]
    assert field in score.issues[0]
    production.assert_not_called()


def _dependency_checkout(tmp_path, name="rulespec-ca", jurisdiction_name="ca"):
    dependency = tmp_path / name
    dependency.mkdir(parents=True)
    jurisdiction = dependency / jurisdiction_name
    jurisdiction.mkdir()
    record = jurisdiction / "README.md"
    record.write_text("Frozen dependency context\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(dependency), *args],
            capture_output=True,
            check=True,
        )
    return dependency, record


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_expected_dependency_digest_detects_changed_context(
    admission_context, tmp_path, backend
):
    result, options = admission_context(backend=backend)
    dependency, record = _dependency_checkout(tmp_path)
    options["rulespec_dependency_roots"] = (dependency,)
    expected = _expected_context(result, options)
    assert len(expected["dependency_content_sha256"]) == 1
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        baseline = _context_score(result, options, expected_context=expected)
        production.assert_called_once()
        record.write_text("Changed dependency context\n")
        changed = _context_score(result, options, expected_context=expected)
        production.assert_called_once()
    assert baseline.admitted is True, baseline.issues
    assert (
        baseline.identity["dependencies"][0]["content_sha256"]
        == (expected["dependency_content_sha256"][0])
    )
    assert changed.failure_kind == "prerequisite", changed.issues
    assert changed.prerequisite_categories == ["expected-context-mismatch"]
    assert "dependency_content_sha256" in changed.issues[0]


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize("kind", ["invalid-name", "active-namespace"])
def test_admission_invalid_dependency_context_is_a_prerequisite(
    admission_context, tmp_path, backend, kind
):
    result, options = admission_context(backend=backend)
    dependency, _ = _dependency_checkout(
        tmp_path / "other-checkout",
        name="dependency" if kind == "invalid-name" else "rulespec-us",
        jurisdiction_name="ca" if kind == "invalid-name" else "us",
    )
    options["rulespec_dependency_roots"] = (dependency,)
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options)
    production.assert_not_called()
    assert score.failure_kind == "prerequisite", score.issues
    assert score.prerequisite_categories == ["invalid-dependency-context"]


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize("directory", [".", "/"])
def test_admission_nameless_candidate_directory_is_refused_before_production(
    admission_context, backend, directory
):
    result, options = admission_context(backend=backend)
    original_candidate = Path(result.output_file)
    original_bytes = original_candidate.read_bytes()
    result.output_file = directory
    with (
        patch.object(
            cli,
            "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
            wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
        ) as production,
        _short_timeout(),
    ):
        score = _context_score(result, options)
    production.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate", score.issues
    assert score.refusal_categories
    assert "generated artifact must be a regular file, not a link" in score.issues[0]
    assert original_candidate.read_bytes() == original_bytes


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize("supply_expected", [False, True])
def test_admission_context_identity_records_frozen_inputs(
    admission_context, backend, supply_expected
):
    result, options = admission_context(backend=backend)
    expected = _expected_context(result, options)
    score = _context_score(
        result, options, expected_context=expected if supply_expected else None
    )
    assert score.admitted is True, score.issues
    assert score.identity["source"]["body_sha256"] == expected["source_body_sha256"]
    assert (
        score.identity["source"]["generation_input_sha256"]
        == (result.source_attestation["generation_input_sha256"])
    )
    assert score.identity["context"]["context_manifest_sha256"] == (
        result.context_manifest_sha256
    )
    assert (
        score.identity["context"]["generation_input_sha256"]
        == (result.source_attestation["generation_input_sha256"])
    )


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize("mutation", ["unbound-path", "module", "source-verification"])
def test_admission_candidate_owned_source_binding_remains_candidate_failure(
    admission_context, backend, mutation
):
    result, options = admission_context(backend=backend)
    candidate = Path(result.output_file)
    payload = yaml.safe_load(candidate.read_text())
    if mutation == "unbound-path":
        payload["module"]["source_verification"]["corpus_citation_path"] = (
            "us/statute/26/999"
        )
    elif mutation == "module":
        payload["module"] = "Cannot read explicit source metadata: missing manifest"
    else:
        payload["module"]["source_verification"] = (
            "Cannot recompute generation input digest: missing frozen context"
        )
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
    original_bytes = candidate.read_bytes()
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options)
    production.assert_called_once()
    assert score.admitted is False
    assert score.failure_kind == "candidate", score.issues
    assert score.refusal_categories
    assert score.prerequisite_categories == []
    assert score.prerequisite_failure is False
    assert candidate.read_bytes() == original_bytes


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_standalone_candidate_uses_canonical_citation_target(
    admission_context, tmp_path, backend
):
    result, options = admission_context(backend=backend)
    candidate = Path(result.output_file)
    companion = cli._rulespec_test_path(candidate)
    draft = tmp_path / "draft.yaml"
    draft_test = cli._rulespec_test_path(draft)
    candidate.rename(draft)
    companion.rename(draft_test)
    result.output_file = str(draft)
    options.pop("output_root")
    original_bytes = (draft.read_bytes(), draft_test.read_bytes())
    policy = options["policy_repo_path"].parent
    original_policy = evals._deterministic_tree_identity(
        policy, excluded_directory_names=frozenset({".git"})
    )
    with patch.object(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        wraps=cli._validate_generated_encoding_candidate_in_policy_overlay_with_release,
    ) as production:
        score = _context_score(result, options)
    production.assert_called_once()
    staged_result = production.call_args.args[0]
    staged_root = production.call_args.kwargs["output_root"] / staged_result.runner
    assert Path(staged_result.output_file).relative_to(staged_root) == Path(
        "statutes/26/1.yaml"
    )
    assert score.admitted is True, score.issues
    assert (draft.read_bytes(), draft_test.read_bytes()) == original_bytes
    assert (
        evals._deterministic_tree_identity(
            policy, excluded_directory_names=frozenset({".git"})
        )
        == original_policy
    )
