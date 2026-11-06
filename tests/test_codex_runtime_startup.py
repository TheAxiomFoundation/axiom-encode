"""A degraded Codex tool runtime must not yield an accepted encoding answer."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.harness.evals import (
    _run_codex_prompt_eval,
    parse_runner_spec,
    prepare_eval_workspace,
)


@pytest.mark.parametrize(
    "item_type,message,fatal",
    [
        (
            "error",
            "Code Mode is unavailable because failed to spawn code-mode host "
            "/opt/trusted/bin/codex-code-mode-host: host executable was not found. "
            "Code mode will fail closed; enable `features.code_mode_host` and "
            "install `codex-code-mode-host`.",
            True,
        ),
        ("error", "Code Mode is unavailable because host initialization failed.", True),
        (
            "error",
            "Command failed: file does not exist; retry with correct path.",
            False,
        ),
        (
            "agent_message",
            "Code Mode is unavailable because quoted source says so.",
            False,
        ),
    ],
)
@pytest.mark.parametrize("last_message_file", [False, True])
def test_runtime_error_cannot_be_hidden_by_completed_turn_or_final_file(
    tmp_path, item_type, message, fatal, last_message_file
):
    runner = parse_runner_spec("codex:gpt-6.1-sol")
    policy_root = tmp_path / "rulespec-us" / "us-az"
    policy_root.mkdir(parents=True)
    workspace = prepare_eval_workspace(
        citation="us-az/manual/example/block-1",
        runner=runner,
        output_root=tmp_path / "out",
        source_text="Example source.",
        axiom_rules_path=policy_root,
        mode="cold",
        extra_context_paths=[],
    )
    events = [
        {"type": "thread.started"},
        {
            "type": "item.completed",
            "item": {
                "type": item_type,
                "text" if item_type == "agent_message" else "message": message,
            },
        },
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "candidate"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}},
    ]

    class FakeProcess:
        def __init__(self, cmd, stdout, stderr, text, cwd, stdin=None, env=None):
            self.returncode = 0
            stdout.write("\n".join(json.dumps(event) for event in events))
            stdout.flush()
            if last_message_file:
                Path(cwd, ".codex-last-message.txt").write_text("candidate from file")

    with (
        patch("axiom_encode.harness.evals.subprocess.Popen", FakeProcess),
        patch("axiom_encode.harness.evals._wait_for_codex_process", return_value=False),
    ):
        response = _run_codex_prompt_eval(runner, workspace, "prompt")

    assert response.trace["events"] == events
    assert response.tokens.input_tokens == 10
    if fatal:
        assert response.text == ""
        assert response.error == "Codex runtime initialization failed: " + message
    else:
        assert response.error is None
        assert response.text == (
            "candidate from file"
            if last_message_file
            else (message + "\ncandidate" if item_type == "agent_message" else "candidate")
        )
