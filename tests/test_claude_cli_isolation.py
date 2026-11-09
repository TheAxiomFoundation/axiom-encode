"""Claude CLI children must use subscription auth, even with an API key upstream."""

from types import SimpleNamespace

import pytest

from axiom_encode.harness import backends, evals, validator_pipeline


def _upstream_environment():
    # Synthetic input only: never inspect the caller's credentials.
    return {"ANTHROPIC_API_KEY": "synthetic-credit-key", "KEEP_ME": "unchanged"}


def test_claude_eval_cli_removes_only_anthropic_api_key(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(evals, "scrub_attestation_signing_keys", _upstream_environment)

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout='{"result": "ok"}', stderr="", returncode=0)

    monkeypatch.setattr(evals.subprocess, "run", fake_run)
    response = evals._run_claude_prompt_eval(
        SimpleNamespace(model="claude-sonnet-5-5"),
        SimpleNamespace(root=tmp_path),
        "synthetic prompt",
    )

    assert response.error is None
    assert calls[0][0][0] == "claude"
    assert calls[0][1]["env"] == {"KEEP_ME": "unchanged"}


@pytest.mark.parametrize("local_context", [False, True])
def test_claude_reviewer_cli_removes_only_anthropic_api_key(
    monkeypatch, tmp_path, local_context
):
    calls = []
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLI", "claude")
    monkeypatch.setattr(
        validator_pipeline, "_local_reviewer_context_active", lambda: local_context
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_resolve_local_reviewer_executable",
        lambda name: "/synthetic/bin/claude",
    )
    monkeypatch.setattr(
        validator_pipeline, "scrub_attestation_signing_keys", _upstream_environment
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_local_reviewer_environment",
        lambda executables: _upstream_environment(),
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(output="ok", returncode=0)

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", fake_run
    )
    assert validator_pipeline.run_claude_code("synthetic review", cwd=tmp_path) == (
        "ok",
        0,
    )
    assert calls[0][0][0].endswith("claude")
    assert calls[0][1]["env"] == {"KEEP_ME": "unchanged"}


def test_claude_encoder_cli_removes_only_anthropic_api_key(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        backends, "scrub_attestation_signing_keys", _upstream_environment
    )

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(backends.subprocess, "run", fake_run)
    assert backends.ClaudeCodeBackend(cwd=tmp_path)._run_claude_code(
        "synthetic prompt"
    ) == ("ok", 0)
    assert calls[0][0][0] == "claude"
    assert calls[0][1]["env"] == {"KEEP_ME": "unchanged"}
