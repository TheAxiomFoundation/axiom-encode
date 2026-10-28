"""Reviewer CLI environment for local runs versus protected runs."""

from __future__ import annotations

import os
import pwd
from types import SimpleNamespace

import pytest

from axiom_encode import signing_broker
from axiom_encode.cli import (
    _print_eval_metrics,
    _review_results_from_eval_metrics,
)
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.eval_evidence import scrub_attestation_signing_keys

_AMBIENT_SECRETS = {
    "OPENAI_API_KEY": "sk-ambient-openai",
    "ANTHROPIC_API_KEY": "sk-ambient-anthropic",
    "GITHUB_TOKEN": "ghp-ambient",
    "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY": "private-eval",
    "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY": "private-apply",
    "SUPABASE_SERVICE_ROLE_KEY": "service-role",
}


@pytest.fixture
def local_context(monkeypatch):
    """No broker, no trusted runtime, no CI: a plain local run."""
    monkeypatch.setattr(signing_broker, "_trusted_subprocess_base_environment", None)
    monkeypatch.setattr(signing_broker, "_active_broker", None)
    for name in ("AXIOM_ENCODE_TRUSTED_RUNTIME", "GITHUB_ACTIONS", "CI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_REVIEWER_CLI", raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_CODEX_BIN", raising=False)
    for name, value in _AMBIENT_SECRETS.items():
        monkeypatch.setenv(name, value)


def _fake_bin(tmp_path, name):
    directory = tmp_path / f"{name}-bin"
    directory.mkdir()
    executable = directory / name
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    return executable


def _capture_runs(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return validator_pipeline._SubprocessRunResult(
            output='{"passed": true}', returncode=0
        )

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", fake_run
    )
    return calls


def test_local_claude_reviewer_gets_allowlisted_environment(
    tmp_path, monkeypatch, local_context
):
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", f"{claude.parent}{os.pathsep}/usr/bin")
    calls = _capture_runs(monkeypatch)

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert (output, returncode) == ('{"passed": true}', 0)
    cmd, kwargs = calls[0]
    assert cmd[0] == str(claude)
    assert kwargs["env_is_final"] is True
    env = kwargs["env"]
    account = pwd.getpwuid(os.getuid())
    assert env["HOME"] == account.pw_dir
    assert env["USER"] == account.pw_name
    assert env["PATH"].split(os.pathsep)[0] == str(claude.parent)
    assert set(env["PATH"].split(os.pathsep)) <= {
        str(claude.parent),
        os.path.dirname(os.path.realpath(claude)),
        *validator_pipeline._LOCAL_REVIEWER_SYSTEM_PATH,
    }
    for name in _AMBIENT_SECRETS:
        assert name not in env
    assert all(value not in env.values() for value in _AMBIENT_SECRETS.values())
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "XDG_CONFIG_HOME" not in env
    allowed = set(scrub_attestation_signing_keys()) | {
        "USER",
        "LOGNAME",
        "TMPDIR",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
    }
    assert set(env) <= allowed


def test_local_codex_fallback_gets_absolute_binary_and_environment(
    tmp_path, monkeypatch, local_context
):
    codex = _fake_bin(tmp_path, "codex")
    monkeypatch.setenv("PATH", str(codex.parent))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    validator_pipeline.run_claude_code("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == str(codex)
    assert cmd[1] == "exec"
    assert kwargs["env_is_final"] is True
    assert kwargs["env"]["HOME"] == pwd.getpwuid(os.getuid()).pw_dir
    assert "OPENAI_API_KEY" not in kwargs["env"]


def test_local_missing_reviewer_clis_is_an_explicit_error(
    tmp_path, monkeypatch, local_context
):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert calls == []
    assert returncode == 1
    assert output.startswith("reviewer_unavailable:")


def test_missing_reviewer_is_a_failed_unscored_review(
    tmp_path, monkeypatch, local_context
):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    rulespec = tmp_path / "rule.yaml"
    rulespec.write_text("format: rulespec/v1\n")
    pipeline = object.__new__(validator_pipeline.ValidatorPipeline)
    pipeline.policy_repo_path = tmp_path

    result = pipeline._run_reviewer("generalist-reviewer", rulespec)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_unavailable" in (result.error or "")


@pytest.mark.parametrize(
    "protect",
    ["broker_base", "active_broker", "trusted_runtime", "github_actions", "ci"],
)
def test_protected_reviewer_environment_is_unchanged(
    tmp_path, monkeypatch, local_context, protect
):
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", str(claude.parent))
    if protect == "broker_base":
        monkeypatch.setattr(
            signing_broker,
            "_trusted_subprocess_base_environment",
            {"HOME": "/trusted-home", "PATH": "/trusted/bin"},
        )
    elif protect == "active_broker":
        monkeypatch.setattr(signing_broker, "_active_broker", object())
    elif protect == "trusted_runtime":
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
    elif protect == "github_actions":
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    else:
        monkeypatch.setenv("CI", "true")
    expected_env = scrub_attestation_signing_keys()
    calls = _capture_runs(monkeypatch)

    validator_pipeline.run_claude_code("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == "claude"
    assert "env_is_final" not in kwargs
    assert kwargs["env"] == expected_env


def test_protected_codex_reviewer_call_is_unchanged(
    tmp_path, monkeypatch, local_context
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    validator_pipeline._run_codex_reviewer_cli("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == "codex"
    assert set(kwargs) == {"timeout", "idle_timeout", "cwd"}


def test_unscrubbed_environment_requires_explicit_env():
    with pytest.raises(ValueError):
        validator_pipeline._run_subprocess_with_idle_timeout(
            ["true"], timeout=1, idle_timeout=1, env_is_final=True
        )


def _metrics(**overrides):
    values = dict(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        ungrounded_numeric_count=0,
        grounded_numeric_count=1,
        embedded_source_present=True,
        grounding=[],
        generalist_review_pass=False,
        generalist_review_score=None,
        generalist_review_issues=["Reviewer error: reviewer_unavailable: missing"],
        policyengine_score=None,
        policyengine_pass=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reviewer_error_is_recorded_and_printed(capsys):
    metrics = _metrics()

    review_results = _review_results_from_eval_metrics(metrics)
    _print_eval_metrics(SimpleNamespace(metrics=metrics))

    integration = [
        review
        for review in review_results.reviews
        if review.reviewer == "integration_reviewer"
    ]
    assert len(integration) == 1
    assert integration[0].passed is False
    assert integration[0].critical_issues == metrics.generalist_review_issues
    captured = capsys.readouterr()
    assert "generalist_review=error Reviewer error: reviewer_unavailable" in (
        captured.out
    )
    assert "was not reviewed" in captured.err


def test_skipped_reviewer_is_not_reported_as_error(capsys):
    metrics = _metrics(generalist_review_pass=True, generalist_review_issues=[])

    review_results = _review_results_from_eval_metrics(metrics)
    _print_eval_metrics(SimpleNamespace(metrics=metrics))

    assert all(
        review.reviewer != "integration_reviewer" for review in review_results.reviews
    )
    captured = capsys.readouterr()
    assert "generalist_review" not in captured.out
    assert captured.err == ""
