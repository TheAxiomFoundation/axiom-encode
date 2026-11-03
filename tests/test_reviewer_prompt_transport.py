"""Complete reviewer prompts travel on stdin, outside the OS argv limits."""

import hashlib
import json
import subprocess
import sys

import pytest

from axiom_encode.harness import validator_pipeline


@pytest.fixture
def long_prompt():
    prompt = "Review the complete artifact.\n" + "λ-policy\r\n" * 20_000
    assert len(prompt.encode("utf-8")) > 128 * 1024
    return prompt


@pytest.mark.parametrize("reviewer_cli", ["claude", "codex"])
def test_reviewer_cli_passes_complete_prompt_on_stdin(
    tmp_path, monkeypatch, long_prompt, reviewer_cli
):
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLI", reviewer_cli)
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CODEX_MODEL", "codex-test-model")
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLAUDE_IDLE_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CODEX_IDLE_TIMEOUT_SECONDS", "7")
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex-test")
    calls = []

    def capture_run(command, **kwargs):
        calls.append((command, kwargs))
        output = (
            '{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"complete review"}}'
            if reviewer_cli == "codex"
            else "complete review"
        )
        return validator_pipeline._SubprocessRunResult(output=output, returncode=0)

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", capture_run
    )
    result = validator_pipeline.run_claude_code(
        long_prompt, model="claude-test-model", timeout=30, cwd=tmp_path
    )

    assert result == ("complete review", 0)
    assert len(calls) == 1
    command, options = calls[0]
    assert long_prompt not in command
    assert options["input_text"] == long_prompt
    assert options["cwd"] == tmp_path
    assert options["timeout"] == 30
    assert options["idle_timeout"] == 7
    if reviewer_cli == "codex":
        assert command[-1] == "-"
        assert command[command.index("--model") + 1] == "codex-test-model"
        assert command[command.index("--sandbox") + 1] == "read-only"
    else:
        assert command[-1] == "-p"
        assert command[command.index("--model") + 1] == "claude-test-model"
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--permission-mode") + 1] == "dontAsk"


def test_missing_claude_fallback_preserves_stdin_prompt(
    tmp_path, monkeypatch, long_prompt
):
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLI", "claude")
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex-test")
    calls = []

    def capture_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "claude":
            raise FileNotFoundError("synthetic missing Claude")
        return validator_pipeline._SubprocessRunResult(
            output='{"type":"item.completed","item":{"type":"agent_message",'
            '"text":"fallback review"}}',
            returncode=0,
        )

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", capture_run
    )
    result = validator_pipeline.run_claude_code(long_prompt, cwd=tmp_path)

    assert result == ("fallback review", 0)
    assert [command[0] for command, _ in calls] == ["claude", "codex-test"]
    for command, options in calls:
        assert long_prompt not in command
        assert options["input_text"] == long_prompt


def test_subprocess_receives_exact_large_utf8_stdin(long_prompt):
    script = (
        "import hashlib,json,sys; data=sys.stdin.buffer.read(); "
        "print(json.dumps({'sha256':hashlib.sha256(data).hexdigest(),"
        "'bytes':len(data)}))"
    )
    result = validator_pipeline._run_subprocess_with_idle_timeout(
        [sys.executable, "-c", script],
        input_text=long_prompt,
        timeout=10,
        idle_timeout=10,
        poll_interval=0.001,
    )

    expected = long_prompt.encode("utf-8")
    assert result.returncode == 0
    assert json.loads(result.output) == {
        "sha256": hashlib.sha256(expected).hexdigest(),
        "bytes": len(expected),
    }


def test_subprocess_without_input_still_receives_eof():
    result = validator_pipeline._run_subprocess_with_idle_timeout(
        [sys.executable, "-c", "import sys; print(len(sys.stdin.buffer.read()))"],
        timeout=10,
        idle_timeout=10,
        poll_interval=0.001,
    )
    assert result.returncode == 0
    assert result.output == "0\n"


def test_unread_large_stdin_does_not_block_timeout(long_prompt):
    with pytest.raises(subprocess.TimeoutExpired):
        validator_pipeline._run_subprocess_with_idle_timeout(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            input_text=long_prompt,
            timeout=0,
            idle_timeout=10,
            poll_interval=0.001,
        )
