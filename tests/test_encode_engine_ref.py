"""Synthetic coverage for explicit, per-job encode engine binding."""

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from axiom_encode import cli
from axiom_encode.engine_binding import (
    EngineBindingError,
    bind_clean_engine_checkout,
    engine_ref_arguments,
    write_engine_binding_receipt,
)
from axiom_encode.harness import evals
from tests.test_engine_binding import _drift_commit, _git


def _bound_engine(base: Path) -> tuple[Path, str, Path]:
    """Bind inert bytes in a new synthetic Git checkout; never execute them."""
    checkout = base / "axiom-rules-engine"
    checkout.mkdir(parents=True, exist_ok=True)
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "config", "user.name", "Synthetic test")
    (checkout / ".gitignore").write_text("target/\n")
    (checkout / "src.rs").write_text("fn main() {}\n")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "synthetic engine")
    ref = _git(checkout, "rev-parse", "HEAD")
    binary = checkout / "target/release/axiom-rules-engine"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic engine bytes")
    binary.chmod(0o755)
    write_engine_binding_receipt(binary, engine_commit=ref, cargo_profile="release")
    return checkout, ref, binary


@pytest.mark.parametrize("ref", ["e" * 40, "E" * 40, "e" * 39])
def test_encode_parser_accepts_only_full_lowercase_ref(tmp_path, monkeypatch, ref):
    captured = Mock()
    monkeypatch.setattr(cli, "cmd_encode", captured)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "encode",
            "us/statute/99/1",
            "--corpus-path",
            str(tmp_path),
            "--axiom-rules-engine-path",
            str(tmp_path),
            "--policy-repo-path",
            str(tmp_path),
            "--axiom-rules-engine-ref",
            ref,
        ],
    )
    if ref == "e" * 40:
        cli.main()
        assert captured.call_args.args[0].axiom_rules_engine_ref == ref
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        captured.assert_not_called()


@pytest.mark.parametrize("drift", ["commit", "dirty", "nested"])
def test_encode_binding_refuses_source_mismatch_even_with_valid_receipt(
    tmp_path, drift
):
    checkout, ref, binary = _bound_engine(tmp_path)
    if drift == "commit":
        _drift_commit(checkout)
    elif drift == "dirty":
        (checkout / "src.rs").write_text("changed source")
    else:
        nested = checkout / "nested"
        nested.mkdir()
        (checkout / ".gitignore").write_text("target/\nnested/\n")
        _git(checkout, "add", ".gitignore")
        _git(checkout, "commit", "-qm", "ignore nested")
        ref = _git(checkout, "rev-parse", "HEAD")
        checkout = nested
    assert binary.exists()
    with pytest.raises(EngineBindingError, match="clean checkout|Git checkout root"):
        bind_clean_engine_checkout(checkout, ref, allow_build=False)


def test_mismatched_ref_stops_model_eval_before_corpus_or_generation(
    tmp_path, monkeypatch
):
    checkout, ref, _ = _bound_engine(tmp_path)
    _drift_commit(checkout)
    resolve = Mock(side_effect=AssertionError("must not resolve a source"))
    generate = Mock(side_effect=AssertionError("must not generate"))
    monkeypatch.setattr(evals, "resolve_corpus_source_unit", resolve)
    monkeypatch.setattr(evals, "_run_single_eval", generate)
    with pytest.raises(EngineBindingError, match="HEAD exactly"):
        evals.run_model_eval(
            citations=["us/statute/99/1"],
            runner_specs=["codex:synthetic"],
            output_root=tmp_path / "out",
            policy_path=tmp_path / "rulespec-us/us",
            runtime_axiom_rules_path=checkout,
            corpus_release=object(),
            axiom_rules_engine_ref=ref,
        )
    resolve.assert_not_called()
    generate.assert_not_called()


def test_model_eval_pins_each_job_without_changing_unpinned_calls(
    tmp_path, monkeypatch
):
    checkout, ref, _ = _bound_engine(tmp_path)
    monkeypatch.setattr(evals, "resolve_corpus_source_unit", lambda *_: object())
    run = Mock(side_effect=lambda **kw: kw.get("axiom_rules_engine_ref"))
    monkeypatch.setattr(evals, "_run_single_eval", run)
    args = dict(
        citations=["us/statute/99/1", "us/statute/99/2"],
        runner_specs=["codex:first", "codex:second"],
        output_root=tmp_path / "out",
        policy_path=tmp_path / "rulespec-us/us",
        runtime_axiom_rules_path=checkout,
        corpus_release=object(),
    )
    assert evals.run_model_eval(**args, axiom_rules_engine_ref=ref) == [ref] * 4
    assert all(
        call.kwargs["axiom_rules_engine_ref"] == ref for call in run.call_args_list
    )
    run.reset_mock()
    assert evals.run_model_eval(**args) == [None] * 4
    assert all(
        "axiom_rules_engine_ref" not in call.kwargs for call in run.call_args_list
    )
    assert engine_ref_arguments(None) == {}


def test_eval_repairs_preserve_pin_on_every_validation_round(tmp_path, monkeypatch):
    rules = tmp_path / "synthetic.yaml"
    rules.write_text("synthetic initial candidate")
    metrics = SimpleNamespace(ci_issues=["synthetic repair"])
    evaluate = Mock(return_value=metrics)
    monkeypatch.setattr(evals, "evaluate_artifact", evaluate)

    def repair(**kwargs):
        assert kwargs["axiom_rules_engine_ref"] == "d" * 40
        if repair_call.call_count == 1:
            rules.write_text("synthetic repaired candidate")
            return ["repaired"]
        return []

    repair_call = Mock(side_effect=repair)
    monkeypatch.setattr(evals, "_apply_generated_eval_repairs", repair_call)
    assert (
        evals._evaluate_generated_artifact_with_repairs(
            rulespec_file=rules,
            policy_repo_root=tmp_path,
            axiom_rules_path=tmp_path,
            source_text="synthetic",
            local_corpus_release=object(),
            axiom_rules_engine_ref="d" * 40,
        )
        is metrics
    )
    assert evaluate.call_count == repair_call.call_count == 2
    assert all(
        call.kwargs["axiom_rules_engine_ref"] == "d" * 40
        for call in evaluate.call_args_list
    )


@pytest.mark.parametrize("entry", ["evaluate", "companion"])
def test_validation_entry_reaches_pipeline_with_explicit_pin(
    tmp_path, monkeypatch, entry
):
    class ReachedPipeline(Exception):
        pass

    def pipeline(**kwargs):
        assert kwargs["axiom_rules_engine_ref"] == "d" * 40
        raise ReachedPipeline

    monkeypatch.setattr(cli, "ValidatorPipeline", pipeline)
    monkeypatch.setattr(evals, "ValidatorPipeline", pipeline)
    monkeypatch.setattr(evals, "_authoritative_corpus_scope", lambda *_: nullcontext())
    monkeypatch.setattr(evals, "_relative_rulespec_source_path", lambda *_: None)
    monkeypatch.setattr(
        evals, "_rulespec_validation_target", lambda path, *a, **kw: nullcontext(path)
    )
    monkeypatch.setattr(evals, "_validation_policy_repo_root", lambda *_: tmp_path)
    monkeypatch.setattr(evals, "_validation_rulespec_dependency_roots", lambda **_: ())
    with pytest.raises(ReachedPipeline):
        if entry == "evaluate":
            evals.evaluate_artifact(
                rulespec_file=tmp_path / "synthetic.yaml",
                policy_repo_root=tmp_path,
                axiom_rules_path=tmp_path,
                source_text="synthetic",
                local_corpus_release=object(),
                axiom_rules_engine_ref="d" * 40,
            )
        else:
            cli._rulespec_companion_test_failures(
                tmp_path / "synthetic.test.yaml",
                root=tmp_path,
                axiom_rules_path=tmp_path,
                axiom_rules_engine_ref="d" * 40,
            )


def _isolate_apply_metadata(monkeypatch, root):
    """Exercise execution binding while replacing unrelated corpus/artifact I/O."""
    monkeypatch.setattr(cli, "_rulespec_apply_content_root", lambda *_: root)
    monkeypatch.setattr(
        cli, "_apply_encoder_execution_identity", lambda: {"synthetic": True}
    )
    monkeypatch.setattr(
        cli, "_rulespec_root_execution_identity", lambda _: {"synthetic": True}
    )
    monkeypatch.setattr(cli, "_apply_policy_content_files", lambda _: {})
    monkeypatch.setattr(cli, "load_rulespec_local_corpus_release", lambda *_: object())
    monkeypatch.setattr(
        cli,
        "_build_apply_validation_snapshot",
        lambda *a, **kw: {"validation_execution": kw["validation_execution_identity"]},
    )


@pytest.mark.parametrize("stage", ["before_apply", "post_install"])
@pytest.mark.parametrize("rewrite_receipt", [False, True])
def test_apply_rechecks_actual_binary_even_when_source_is_unchanged(
    tmp_path,
    monkeypatch,
    stage,
    rewrite_receipt,
):
    checkout, ref, binary = _bound_engine(tmp_path)
    _isolate_apply_metadata(monkeypatch, tmp_path)
    baseline = bind_clean_engine_checkout(checkout, ref, allow_build=False)
    result = SimpleNamespace()
    kwargs = dict(
        output_root=tmp_path,
        policy_repo_path=tmp_path,
        relative_output=Path("synthetic.yaml"),
        supplemental_files={},
        local_corpus_release=object(),
    )
    cli._record_successful_apply_validation(
        result,
        **kwargs,
        axiom_rules_path=checkout,
        axiom_rules_engine_ref=ref,
        validated_engine_binding=baseline,
    )
    snapshot = getattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR)
    assert json.loads(json.dumps(snapshot)) == snapshot
    assert snapshot["validation_execution"]["axiom_rules_engine_binding"] == baseline
    cli._require_unchanged_successful_apply_validation(result, **kwargs)
    binary.write_bytes(b"different engine bytes")
    if rewrite_receipt:
        write_engine_binding_receipt(binary, engine_commit=ref, cargo_profile="release")
    with pytest.raises(
        (EngineBindingError, RuntimeError), match="changed|verified|build|binding"
    ):
        if stage == "before_apply":
            cli._require_unchanged_successful_apply_validation(result, **kwargs)
        else:
            del kwargs["local_corpus_release"]
            cli._require_apply_post_install_closure(
                result,
                **kwargs,
                corpus_path=tmp_path,
                planned={},
                manifest_path=tmp_path / "manifest.json",
                manifest_bytes=b"{}",
            )
    assert snapshot["validation_execution"]["axiom_rules_engine_binding"] == baseline


def test_record_rejects_binary_changed_during_overlay(tmp_path, monkeypatch):
    checkout, ref, binary = _bound_engine(tmp_path)
    _isolate_apply_metadata(monkeypatch, tmp_path)
    baseline = bind_clean_engine_checkout(checkout, ref, allow_build=False)
    binary.write_bytes(b"changed during validation")
    write_engine_binding_receipt(binary, engine_commit=ref, cargo_profile="release")
    result = SimpleNamespace()
    with pytest.raises(RuntimeError, match="changed during overlay"):
        cli._record_successful_apply_validation(
            result,
            output_root=tmp_path,
            policy_repo_path=tmp_path,
            relative_output=Path("synthetic.yaml"),
            supplemental_files={},
            local_corpus_release=object(),
            axiom_rules_path=checkout,
            axiom_rules_engine_ref=ref,
            validated_engine_binding=baseline,
        )
    assert not hasattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR)
    assert baseline["binary_sha256"] != hashlib.sha256(binary.read_bytes()).hexdigest()


@pytest.mark.parametrize("binding", [None, {}, {"engine_ref": "d" * 40}])
def test_present_malformed_snapshot_binding_cannot_downgrade(binding):
    with pytest.raises(RuntimeError, match="malformed engine binding"):
        cli._apply_validation_engine_ref({"axiom_rules_engine_binding": binding})
    assert cli._apply_validation_engine_ref({}) is None


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("mutation", [None, "source", "binary"])
def test_single_job_rechecks_binding_before_emitting_success(
    tmp_path,
    monkeypatch,
    retained,
    mutation,
):
    from tests.test_evals import _write_test_source_unit

    checkout, ref, binary = _bound_engine(tmp_path)
    release, source = _write_test_source_unit(
        tmp_path,
        "Synthetic test source.",
        citation_path="us/statute/99/1",
    )
    policy = tmp_path / "rulespec-us/us"
    policy.mkdir(parents=True)
    calls = []
    metrics = evals.EvalArtifactMetrics(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        embedded_source_present=False,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
    )

    def evaluate(**kwargs):
        calls.append(kwargs)
        assert kwargs["axiom_rules_engine_ref"] == ref
        if mutation == "source":
            (checkout / "src.rs").write_text("dirty while evaluating")
        elif mutation == "binary":
            binary.write_bytes(b"changed while evaluating")
            write_engine_binding_receipt(
                binary, engine_commit=ref, cargo_profile="release"
            )
        return metrics

    def generate(**kwargs):
        output = kwargs["output_file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("format: rulespec/v1\nrules: []\n")
        return (
            evals.EvalPromptResponse(text="synthetic", duration_ms=1),
            True,
            0,
            frozenset({output}),
        )

    monkeypatch.setattr(evals, "_evaluate_generated_artifact_with_repairs", evaluate)
    generation = Mock(side_effect=generate)
    monkeypatch.setattr(evals, "_run_prompt_eval_with_empty_artifact_retry", generation)
    # Exercise both retained-candidate preflights, including the proof-rebind branch.
    monkeypatch.setattr(
        evals,
        "_rebind_retained_candidate_proof_import_hashes",
        lambda **_: ["synthetic"],
    )
    emit = Mock()
    monkeypatch.setattr(evals, "emit_eval_result", emit)
    kwargs = dict(
        citations=[source.requested],
        runner_specs=["codex:synthetic"],
        output_root=tmp_path / "out",
        policy_path=policy,
        runtime_axiom_rules_path=checkout,
        corpus_release=release,
        mode="cold",
        axiom_rules_engine_ref=ref,
    )
    if retained:
        kwargs.update(
            accept_valid_retry_candidate=True,
            validation_retry_candidate=evals.ValidationRetryCandidate(
                rulespec="format: rulespec/v1\nrules: []\n",
                tests="[]\n",
            ),
        )
    if mutation is None:
        result = evals.run_model_eval(**kwargs)[0]
        assert result.success
        emit.assert_called_once()
    else:
        with pytest.raises(
            (EngineBindingError, RuntimeError), match="clean checkout|changed"
        ):
            evals.run_model_eval(**kwargs)
        emit.assert_not_called()
    assert len(calls) == (2 if retained else 1)
    assert generation.call_count == (0 if retained else 1)


@pytest.mark.parametrize(
    "source_identity",
    [
        {"kind": "git", "commit": "f" * 40, "dirty": False},
        {"kind": "git", "commit": None, "dirty": True},
    ],
)
def test_captured_source_identity_must_match_actual_binding(
    tmp_path, monkeypatch, source_identity
):
    checkout, ref, _ = _bound_engine(tmp_path)
    _isolate_apply_metadata(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "_git_checkout_execution_identity", lambda _: source_identity
    )
    with pytest.raises(RuntimeError, match="source identity changed"):
        cli._apply_validation_execution_identity(
            axiom_rules_path=checkout,
            policy_repo_path=tmp_path,
            relative_output=Path("synthetic.yaml"),
            rulespec_dependency_roots=(),
            axiom_rules_engine_ref=ref,
        )
