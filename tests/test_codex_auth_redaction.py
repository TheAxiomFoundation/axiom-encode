"""Codex children can read auth.json; nothing they print may keep its secrets."""

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from axiom_encode import signing_broker
from axiom_encode.codex_cli import (
    CODEX_AUTH_REDACTION,
    codex_auth_secrets,
    redact_codex_auth_secrets,
)
from axiom_encode.harness import evals, validator_pipeline

ACCESS = "eyJhbGciOiJSUzI1NiJ9.invented-access-token.signature"
REFRESH = 'rt_invented"refresh\\token/value'
ROTATED = "rt_invented-rotated-refresh-token-value"


def write_auth(home: Path, **tokens: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {"account_id": "acct", **tokens},
            }
        )
    )


def assert_clean(text: str, *secrets: str) -> None:
    for secret in secrets:
        assert secret not in text
        assert json.dumps(secret)[1:-1] not in text


def test_secrets_are_every_long_string_leaf_and_never_raise(tmp_path):
    write_auth(tmp_path, access_token=ACCESS, refresh_token=REFRESH)
    assert codex_auth_secrets(tmp_path) == {ACCESS, REFRESH}
    assert codex_auth_secrets(tmp_path / "missing") == frozenset()
    assert codex_auth_secrets(None) == frozenset()
    (tmp_path / "auth.json").write_text("{not json")
    assert codex_auth_secrets(tmp_path) == frozenset()


def test_redaction_covers_verbatim_and_json_escaped_forms():
    text = f"plain {REFRESH} escaped {json.dumps(REFRESH)} done"
    redacted = redact_codex_auth_secrets(text, frozenset({REFRESH}))
    assert_clean(redacted, REFRESH)
    assert redacted.count(CODEX_AUTH_REDACTION) == 2
    assert redact_codex_auth_secrets(text, frozenset()) == text


@given(
    secret=st.text(min_size=16, max_size=40).filter(
        lambda value: value not in CODEX_AUTH_REDACTION
    ),
    before=st.text(max_size=20),
    after=st.text(max_size=20),
)
def test_no_form_of_a_secret_survives_redaction(secret, before, after):
    secrets = frozenset({secret})
    for form in (
        secret,
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
    ):
        redacted = redact_codex_auth_secrets(before + form + after, secrets)
        assert secret not in redacted
        assert json.dumps(secret)[1:-1] not in redacted


def test_reviewer_output_and_logs_carry_no_auth_token(tmp_path, monkeypatch, caplog):
    """A real child prints the token and rotates it; neither form survives."""
    home = tmp_path / "supervisor-home"
    write_auth(home, access_token=ACCESS, refresh_token=REFRESH)
    child = tmp_path / "codex"
    child.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "auth = pathlib.Path(os.environ['CODEX_HOME']) / 'auth.json'\n"
        "data = json.loads(auth.read_text())\n"
        "old = data['tokens']['refresh_token']\n"
        f"data['tokens']['refresh_token'] = {ROTATED!r}\n"
        "auth.write_text(json.dumps(data))\n"
        "text = 'leaked ' + data['tokens']['access_token'] + ' ' + old\n"
        "text += ' ' + data['tokens']['refresh_token']\n"
        "print(json.dumps({'type': 'item.completed',\n"
        "    'item': {'type': 'agent_message', 'text': text}}))\n"
        "sys.stderr.write('stderr ' + old + '\\n')\n"
    )
    child.chmod(0o755)
    monkeypatch.setattr(signing_broker, "_trusted_codex_home", str(home))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: str(child))

    with caplog.at_level(logging.DEBUG):
        text, returncode = validator_pipeline._run_codex_reviewer_cli(
            "review this", timeout=30
        )

    assert returncode == 0
    assert "leaked" in text
    assert CODEX_AUTH_REDACTION in text
    assert_clean(text, ACCESS, REFRESH, ROTATED)
    assert_clean(caplog.text, ACCESS, REFRESH, ROTATED)


def test_generation_output_carries_no_auth_token(tmp_path, monkeypatch):
    """Generation's temporary home links auth.json; its secrets are redacted too."""
    source_home = tmp_path / "source-home"
    write_auth(source_home, access_token=ACCESS, refresh_token=REFRESH)
    runner = replace(evals.parse_runner_spec("codex:test-model"))
    workspace = evals.EvalWorkspace(
        root=tmp_path,
        source_text_file=tmp_path / "source.txt",
        manifest_file=tmp_path / "manifest.json",
    )

    class FakePopen:
        returncode = 1

        def __init__(self, cmd, **kwargs):
            home = Path(kwargs["env"]["CODEX_HOME"])
            data = json.loads((home / "auth.json").read_text())
            old = data["tokens"]["refresh_token"]
            data["tokens"]["refresh_token"] = ROTATED
            # Write through to the copy or symlink target, as a refresh would.
            (home / "auth.json").write_text(json.dumps(data))
            message = f"format: rulespec/v1 # {ACCESS} {old} {ROTATED}"
            kwargs["stdout"].write(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": message},
                    }
                )
                + "\n"
            )
            kwargs["stdout"].flush()
            kwargs["stderr"].write(f"error with {old}\n")
            kwargs["stderr"].flush()
            last = cmd[cmd.index("-o") + 1]
            Path(last).write_text(message)

    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: "/explicit/codex")
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (20, 10))
    monkeypatch.setattr(evals.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(evals, "_wait_for_codex_process", lambda *a, **kw: False)

    response = evals._run_codex_prompt_eval(runner, workspace, "encode source")

    assert response.text.startswith("format: rulespec/v1")
    kept = json.dumps(
        {"text": response.text, "error": response.error, "trace": response.trace},
        default=str,
    )
    assert CODEX_AUTH_REDACTION in kept
    assert_clean(kept, ACCESS, REFRESH, ROTATED)
