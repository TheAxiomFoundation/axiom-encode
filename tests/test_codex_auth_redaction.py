"""Codex children can read auth.json; nothing they print may keep its secrets."""

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
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
        assert json.dumps(secret, ensure_ascii=False)[1:-1] not in redacted


def test_unicode_json_form_is_redacted():
    """JSON written without ASCII escaping keeps non-ASCII characters raw but
    still escapes quotes and backslashes; that form is neither verbatim nor
    ASCII-escaped, so it needs its own replacement."""
    secret = 'tok\u00e9n-"quoted"\\-value-0123'
    form = json.dumps(secret, ensure_ascii=False)[1:-1]
    assert form not in (secret, json.dumps(secret)[1:-1])
    assert redact_codex_auth_secrets(f"x {form} y", frozenset({secret})) == (
        f"x {CODEX_AUTH_REDACTION} y"
    )


def test_a_secret_containing_another_is_replaced_whole():
    """An account ID inside a longer token must not split the longer one."""
    inner = "acct-0123456789abcdef"
    outer = f"eyJ.{inner}.signature-part"
    redacted = redact_codex_auth_secrets(f"[{outer}]", frozenset({inner, outer}))
    assert redacted == f"[{CODEX_AUTH_REDACTION}]"


def _token_printing_child(tmp_path: Path) -> Path:
    """A real child that rotates the refresh token and prints every token."""
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
    return child


def test_reviewer_output_and_logs_carry_no_auth_token(tmp_path, monkeypatch, caplog):
    """Through the real review path: the reply isn't JSON, so _run_reviewer
    logs 500 characters of it. Neither that warning nor the result may carry
    a token, including one rotated during the run."""
    home = tmp_path / "supervisor-home"
    write_auth(home, access_token=ACCESS, refresh_token=REFRESH)
    child = _token_printing_child(tmp_path)
    monkeypatch.setattr(signing_broker, "_trusted_codex_home", str(home))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: str(child))
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLI", "codex")
    policy = tmp_path / "rulespec-us" / "us"
    rules = policy / "statutes" / "1.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text("format: rulespec/v1\nrules: []\n")
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=policy,
        axiom_rules_path=tmp_path / "missing-engine",
        local_corpus_release=None,
        enable_oracles=False,
    )

    with caplog.at_level(logging.WARNING):
        result = pipeline._run_reviewer("generalist-reviewer", rules)

    assert "failed to parse as JSON" in caplog.text
    assert "leaked" in caplog.text and CODEX_AUTH_REDACTION in caplog.text
    assert "reviewer_parse_failed" in result.issues
    kept = json.dumps(
        {"raw": result.raw_output, "issues": result.issues, "error": result.error}
    )
    for text in (caplog.text, result.raw_output, kept):
        assert_clean(text, ACCESS, REFRESH, ROTATED)


def _fake_reviewer_run(monkeypatch, tmp_path, *, output, extractor):
    home = tmp_path / "supervisor-home"
    write_auth(home, access_token=ACCESS, refresh_token=REFRESH)
    monkeypatch.setattr(signing_broker, "_trusted_codex_home", str(home))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    monkeypatch.setattr(
        validator_pipeline,
        "_run_subprocess_with_idle_timeout",
        lambda *a, **k: SimpleNamespace(output=output, returncode=0),
    )
    monkeypatch.setattr(validator_pipeline, "_extract_codex_text_output", extractor)
    return validator_pipeline._run_codex_reviewer_cli("review", timeout=30)


def test_raw_event_stream_is_redacted_before_extraction(tmp_path, monkeypatch):
    """Layer one, on its own: the extractor never sees a token, verbatim or
    JSON-escaped, so nothing it might log or keep can carry one."""
    seen = []
    raw = json.dumps({"text": f"{ACCESS} {REFRESH}"})
    _fake_reviewer_run(
        monkeypatch,
        tmp_path,
        output=raw,
        extractor=lambda text: seen.append(text) or "clean",
    )
    assert seen and CODEX_AUTH_REDACTION in seen[0]
    assert_clean(seen[0], ACCESS, REFRESH)


def test_extracted_text_is_redacted_again(tmp_path, monkeypatch):
    """Layer two, on its own: whatever extraction yields is redacted too, so
    a token the extractor assembles from parts can't be returned."""
    text, returncode = _fake_reviewer_run(
        monkeypatch,
        tmp_path,
        output="nothing secret here",
        extractor=lambda _text: f"reassembled {ACCESS}",
    )
    assert returncode == 0
    assert text == f"reassembled {CODEX_AUTH_REDACTION}"


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
    # The raw last-message file holds the unredacted text and sits in the
    # workspace under --output, which failure diagnostics upload. It must go.
    assert not (tmp_path / ".codex-last-message.txt").exists()
    for path in tmp_path.rglob("*"):
        # Only the Codex home itself may hold the credentials.
        if path.is_file() and source_home not in path.parents:
            assert_clean(path.read_text(errors="replace"), ACCESS, ROTATED)


def test_unreadable_last_message_file_is_still_deleted(tmp_path, monkeypatch):
    """A read that raises must not leave the raw file in the workspace."""
    source_home = tmp_path / "source-home"
    write_auth(source_home, access_token=ACCESS, refresh_token=REFRESH)
    workspace = evals.EvalWorkspace(
        root=tmp_path / "workspace",
        source_text_file=tmp_path / "source.txt",
        manifest_file=tmp_path / "manifest.json",
    )
    workspace.root.mkdir()
    last_message = workspace.root / ".codex-last-message.txt"

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            # Not valid UTF-8, with a token in it: the text read raises.
            Path(cmd[cmd.index("-o") + 1]).write_bytes(
                b"\xff\xfe " + ACCESS.encode() + b" \xc3\x28"
            )

    monkeypatch.setenv("CODEX_HOME", str(source_home))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: "/explicit/codex")
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (20, 10))
    monkeypatch.setattr(evals.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(evals, "_wait_for_codex_process", lambda *a, **kw: False)

    with pytest.raises(UnicodeDecodeError):
        evals._run_codex_prompt_eval(
            evals.parse_runner_spec("codex:test-model"), workspace, "encode source"
        )

    assert not last_message.exists()
