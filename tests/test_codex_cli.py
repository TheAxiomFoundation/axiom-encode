"""Codex CLI helper behavior."""

import json
from types import SimpleNamespace

import pytest

from axiom_encode.codex_cli import with_codex_model_availability_hint
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import _extract_codex_text_output

# Verbatim shape of the 400 ChatGPT-account Codex returned for gpt-6-luna on
# 2026-09-24 (codex-cli 0.153.3).
CHATGPT_REJECTION = (
    '{"type":"error","status":400,"error":{"type":"invalid_request_error",'
    '"message":"The \'gpt-6-luna\' model is not supported when using Codex '
    'with a ChatGPT account."}}'
)


def test_hint_leaves_absent_and_unrelated_errors_alone():
    assert with_codex_model_availability_hint(None) is None
    assert with_codex_model_availability_hint("") == ""
    assert with_codex_model_availability_hint("Codex eval timed out") == (
        "Codex eval timed out"
    )


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"])
def test_hint_leaves_successful_gpt6_output_alone(model):
    output = f"Generation completed with {model}"
    assert with_codex_model_availability_hint(output) == output


def test_hint_names_the_explicit_model_workaround():
    hinted = with_codex_model_availability_hint(CHATGPT_REJECTION)
    assert hinted.startswith(CHATGPT_REJECTION)
    assert "Update the Codex CLI" in hinted
    assert "--model MODEL --escalation-model MODEL" in hinted
    assert "--runner claude:opus --runner codex:MODEL" in hinted
    assert "AXIOM_ENCODE_REVIEWER_CODEX_MODEL=MODEL" in hinted
    assert "gpt-5.6" not in hinted


def test_codex_reviewer_output_carries_the_hint():
    stream = json.dumps({"type": "error", "message": CHATGPT_REJECTION})
    text = _extract_codex_text_output(stream + "\n")
    assert CHATGPT_REJECTION in text
    assert "--model MODEL" in text


def test_codex_reviewer_defaults_to_the_encoder_model(monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_REVIEWER_CODEX_MODEL", raising=False)
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return SimpleNamespace(output="", returncode=0)

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", fake_run
    )
    validator_pipeline._run_codex_reviewer_cli("review this")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--model") + 1] == "gpt-6-luna"
