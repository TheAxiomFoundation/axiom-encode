"""CLI policy and reviewer switches for the prompt-only Messages runner."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import axiom_encode.cli as cli
from axiom_encode.constants import DEFAULT_MODEL, DEFAULT_OPENAI_ESCALATION_MODEL


def _arguments(command, positional):
    return [
        "axiom-encode",
        command,
        positional,
        "--corpus-path",
        "/synthetic/axiom-corpus",
        "--axiom-rules-engine-path",
        "/synthetic/axiom-rules-engine",
        "--policy-repo-path",
        "/synthetic/rulespec-us",
    ]


def test_encode_parser_accepts_claude_api_and_defaults_to_opus(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(cli, "cmd_encode", handler)
    monkeypatch.setattr(
        "sys.argv",
        _arguments("encode", "26 USC 1") + ["--backend", "claude-api"],
    )

    cli.main()

    args = handler.call_args.args[0]
    assert args.backend == "claude-api"
    assert args.escalation_model == DEFAULT_MODEL
    config = cli._resolve_encode_escalation_config(args)
    assert config.initial_model == DEFAULT_MODEL
    assert config.escalation_model == DEFAULT_MODEL


def test_claude_api_encode_config_uses_opus_escalation_default():
    config = cli._resolve_encode_escalation_config(
        SimpleNamespace(backend="claude-api", model="claude-sonnet-5-5")
    )

    assert config.initial_model == "claude-sonnet-5-5"
    assert config.escalation_model == "claude-opus-5-5"


def test_claude_api_encode_keeps_explicit_escalation_model(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(cli, "cmd_encode", handler)
    monkeypatch.setattr(
        "sys.argv",
        _arguments("encode", "26 USC 1")
        + [
            "--backend",
            "claude-api",
            "--escalation-model",
            "claude-sonnet-5-5",
        ],
    )

    cli.main()

    assert handler.call_args.args[0].escalation_model == "claude-sonnet-5-5"


def test_claude_api_apply_rejected_before_trust_generation_or_signing(monkeypatch):
    touched = {
        name: Mock(side_effect=AssertionError(f"unexpected call: {name}"))
        for name in (
            "_local_verification_encode_roots",
            "_cmd_encode_under_current_trust",
            "_isolated_apply_manifest_signer",
            "run_model_eval",
        )
    }
    for name, mock in touched.items():
        monkeypatch.setattr(cli, name, mock)

    with pytest.raises(ValueError, match=r"claude-api.*signed.*encoding rule"):
        cli.cmd_encode(SimpleNamespace(backend="claude-api", apply=True))

    for mock in touched.values():
        mock.assert_not_called()


def test_claude_api_rejects_gpt_escalation_model():
    with pytest.raises(ValueError, match=r"GPT escalation model.*claude-api"):
        cli._require_compatible_encode_retry_model(
            backend="claude-api", model=DEFAULT_OPENAI_ESCALATION_MODEL
        )


@pytest.mark.parametrize(
    ("command", "positional", "handler_name"),
    [
        ("eval", "26 USC 1", "cmd_eval"),
        ("eval-source", "us/statute/26/1", "cmd_eval_source"),
        ("eval-suite", "synthetic.yaml", "cmd_eval_suite"),
    ],
)
def test_eval_commands_accept_skip_reviewers(
    monkeypatch, command, positional, handler_name
):
    handler = Mock()
    monkeypatch.setattr(cli, handler_name, handler)
    monkeypatch.setattr(
        "sys.argv", _arguments(command, positional) + ["--skip-reviewers"]
    )

    cli.main()

    assert handler.call_args.args[0].skip_reviewers is True


@pytest.mark.parametrize("command", ["eval", "eval_source", "eval_suite"])
def test_eval_commands_forward_skip_reviewers(monkeypatch, tmp_path, command):
    args = SimpleNamespace(
        runner=["codex:gpt-test"],
        citations=["26 USC 1"],
        corpus_citation_path="us/statute/26/1",
        corpus_path=tmp_path,
        axiom_rules_path=tmp_path,
        policy_repo_path=tmp_path,
        output=tmp_path / "out",
        manifest=tmp_path / "suite.yaml",
        mode="cold",
        allow_context=[],
        policyengine_rule_hint=None,
        skip_reviewers=True,
    )
    monkeypatch.setattr(cli, "_effective_runner_specs", lambda runners, _args: runners)
    monkeypatch.setattr(cli, "_rulespec_dependency_roots_from_args", lambda _: ())
    monkeypatch.setattr(cli, "_eval_policy_repo_for_citations", lambda *_: tmp_path)
    monkeypatch.setattr(
        cli, "_resolve_explicit_policy_repo_for_corpus_source", lambda *_: tmp_path
    )
    monkeypatch.setattr(cli, "load_rulespec_local_corpus_release", lambda *_: object())
    monkeypatch.setattr(cli, "resolve_corpus_source_unit", lambda *_: object())
    monkeypatch.setattr(
        cli,
        "load_eval_suite_manifest",
        lambda _: SimpleNamespace(runners=["codex:gpt-test"], cases=[]),
    )
    monkeypatch.setattr(cli, "_eval_suite_corpus_release", lambda *_, **__: object())
    monkeypatch.setattr(cli, "_eval_suite_policyengine_runtime", lambda *_: None)

    class RunReached(Exception):
        pass

    run = Mock(side_effect=RunReached)
    run_name = {
        "eval": "run_model_eval",
        "eval_source": "run_source_eval",
        "eval_suite": "run_eval_suite",
    }[command]
    monkeypatch.setattr(cli, run_name, run)

    with pytest.raises(RunReached):
        getattr(cli, f"cmd_{command}")(args)

    assert run.call_args.kwargs["skip_reviewers"] is True


@pytest.mark.parametrize("persisted_skip", [True, False])
def test_suite_verification_recovers_persisted_reviewer_state(
    monkeypatch, tmp_path, persisted_skip
):
    identity = {"skip_reviewers": persisted_skip}
    state = {
        "execution_identity": identity,
        "execution_identity_sha256": cli._eval_suite_execution_identity_sha256(
            identity
        ),
    }
    monkeypatch.setattr(
        cli,
        "_read_eval_suite_artifact_bytes",
        lambda *_args, **_kwargs: json.dumps(state).encode(),
    )
    monkeypatch.setattr(cli, "_build_eval_suite_manifest_identity", lambda _: {})
    monkeypatch.setattr(cli, "_eval_suite_rulespec_roots", lambda *_: ())

    class IdentityReached(Exception):
        pass

    build = Mock(side_effect=IdentityReached)
    monkeypatch.setattr(cli, "_build_eval_suite_execution_identity", build)

    with pytest.raises(IdentityReached):
        cli._load_verified_eval_suite_artifacts(
            output_root=tmp_path,
            manifest=SimpleNamespace(cases=[]),
            effective_runners=["codex:gpt-test"],
            axiom_rules_path=tmp_path,
            policy_repo_path=tmp_path,
            corpus_release=object(),
            require_complete=False,
            suite_retry_attempts=0,
            admission_score=False,
        )

    assert build.call_args.kwargs.get("skip_reviewers", False) is persisted_skip


def test_suite_verification_rejects_nonboolean_reviewer_state(monkeypatch, tmp_path):
    identity = {"skip_reviewers": 1}
    state = {
        "execution_identity": identity,
        "execution_identity_sha256": cli._eval_suite_execution_identity_sha256(
            identity
        ),
    }
    monkeypatch.setattr(
        cli,
        "_read_eval_suite_artifact_bytes",
        lambda *_args, **_kwargs: json.dumps(state).encode(),
    )
    monkeypatch.setattr(cli, "_build_eval_suite_manifest_identity", lambda _: {})
    monkeypatch.setattr(cli, "_eval_suite_rulespec_roots", lambda *_: ())
    monkeypatch.setattr(
        cli,
        "_build_eval_suite_execution_identity",
        Mock(
            side_effect=AssertionError(
                "invalid reviewer state reached identity builder"
            )
        ),
    )

    with pytest.raises(ValueError, match="skip_reviewers.*boolean"):
        cli._load_verified_eval_suite_artifacts(
            output_root=tmp_path,
            manifest=SimpleNamespace(cases=[]),
            effective_runners=["codex:gpt-test"],
            axiom_rules_path=tmp_path,
            policy_repo_path=tmp_path,
            corpus_release=object(),
            require_complete=False,
            suite_retry_attempts=0,
            admission_score=False,
        )


def test_eval_suite_verification_receives_current_reviewer_state(monkeypatch, tmp_path):
    args = SimpleNamespace(
        manifest=tmp_path / "suite.yaml",
        corpus_path=tmp_path,
        axiom_rules_path=tmp_path,
        policy_repo_path=tmp_path,
        output=tmp_path / "out",
        skip_reviewers=True,
    )
    monkeypatch.setattr(
        cli,
        "load_eval_suite_manifest",
        lambda _: SimpleNamespace(runners=["codex:gpt-test"], cases=[]),
    )
    monkeypatch.setattr(cli, "_eval_suite_corpus_release", lambda *_, **__: object())
    monkeypatch.setattr(cli, "_eval_suite_policyengine_runtime", lambda *_: None)
    monkeypatch.setattr(cli, "_load_eval_suite_run_state", lambda *_: {})
    monkeypatch.setattr(cli, "run_eval_suite", Mock(return_value=[]))

    class VerificationReached(Exception):
        pass

    verify = Mock(side_effect=VerificationReached)
    monkeypatch.setattr(cli, "_load_verified_eval_suite_artifacts", verify)

    with pytest.raises(VerificationReached):
        cli.cmd_eval_suite(args)

    assert verify.call_args.kwargs["skip_reviewers"] is True


def test_eval_suite_reports_context_overflow_as_infrastructure(
    monkeypatch, tmp_path, capsys
):
    args = SimpleNamespace(
        manifest=tmp_path / "suite.yaml",
        corpus_path=tmp_path,
        axiom_rules_path=tmp_path,
        policy_repo_path=tmp_path,
        output=tmp_path / "out",
        json=False,
    )
    manifest = SimpleNamespace(
        runners=["claude-api:claude-opus-5-5"],
        cases=[object()],
        name="overflow",
        path=args.manifest,
        gates=object(),
    )
    result = SimpleNamespace(
        runner="claude-api-claude-opus-5-5",
        citation="us/statute/26/1",
        success=False,
        error="Context overflow (prompt_chars=1200000)",
        metrics=None,
    )
    summary = SimpleNamespace(
        ready=False,
        total_cases=0,
        artifact_case_count=0,
        timeout_count=0,
        infrastructure_error_count=1,
        success_rate=0.0,
        compile_pass_rate=None,
        ci_pass_rate=None,
        zero_ungrounded_rate=None,
        generalist_review_pass_rate=None,
        mean_generalist_review_score=None,
        policyengine_case_count=0,
        mean_estimated_cost_usd=None,
        gate_results=[],
    )
    monkeypatch.setattr(cli, "load_eval_suite_manifest", lambda _: manifest)
    monkeypatch.setattr(
        cli,
        "_eval_suite_corpus_release",
        lambda *_, **__: SimpleNamespace(name="synthetic-release"),
    )
    monkeypatch.setattr(cli, "_eval_suite_policyengine_runtime", lambda *_: None)
    monkeypatch.setattr(cli, "_load_eval_suite_run_state", lambda *_: {})
    monkeypatch.setattr(cli, "_suite_auto_resume_reason", lambda *_, **__: None)
    monkeypatch.setattr(cli, "run_eval_suite", Mock(return_value=[result]))
    monkeypatch.setattr(
        cli,
        "_load_verified_eval_suite_artifacts",
        lambda **_: {
            "results": [result],
            "completed_case_indexes": {1},
            "run_state": {},
        },
    )
    monkeypatch.setattr(cli, "summarize_readiness", lambda *_: summary)
    monkeypatch.setattr(
        cli, "_build_eval_suite_payload", lambda **_: {"all_ready": False}
    )
    monkeypatch.setattr(cli, "_eval_suite_summary_payload", lambda payload: payload)
    monkeypatch.setattr(cli, "_atomic_write_eval_suite_artifact", Mock())

    with pytest.raises(SystemExit) as stopped:
        cli.cmd_eval_suite(args)

    assert stopped.value.code == 1
    output = capsys.readouterr().out
    assert "infrastructure_errors=1" in output
    assert "context overflow" in output.lower()
