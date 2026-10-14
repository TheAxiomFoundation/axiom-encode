"""Codex reasoning-effort selection from encode CLI to subprocess and trace."""

import json
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.harness import evals


@pytest.mark.parametrize("effort", [None, "low", "high", "ultra"])
def test_encode_cli_reasoning_effort_defaults_to_low_and_accepts_override(effort):
    argv = [
        "axiom-encode",
        "encode",
        "us/statute/26/1",
        "--corpus-path",
        "/explicit/corpus",
        "--axiom-rules-engine-path",
        "/explicit/engine",
        "--policy-repo-path",
        "/explicit/rulespec-us",
    ]
    if effort is not None:
        argv.extend(["--codex-reasoning-effort", effort])

    with patch("sys.argv", argv), patch.object(cli, "cmd_encode") as encode:
        cli.main()

    assert encode.call_args.args[0].codex_reasoning_effort == (effort or "low")


@pytest.mark.parametrize("effort", ["", " ", " high", "high\n"])
def test_encode_cli_rejects_blank_or_padded_reasoning_effort(effort):
    argv = [
        "axiom-encode",
        "encode",
        "us/statute/26/1",
        "--corpus-path",
        "/explicit/corpus",
        "--axiom-rules-engine-path",
        "/explicit/engine",
        "--policy-repo-path",
        "/explicit/rulespec-us",
        "--codex-reasoning-effort",
        effort,
    ]
    with (
        patch("sys.argv", argv),
        patch.object(cli, "cmd_encode") as encode,
        pytest.raises(SystemExit) as error,
    ):
        cli.main()

    assert error.value.code == 2
    encode.assert_not_called()


@pytest.mark.parametrize("effort", [None, "high", "ultra"])
def test_run_model_eval_applies_effort_only_to_codex_runners(tmp_path, effort):
    kwargs = {} if effort is None else {"codex_reasoning_effort": effort}
    with (
        patch.object(evals, "_validate_eval_oracle_runtime"),
        patch.object(evals, "resolve_corpus_source_unit"),
        patch.object(
            evals,
            "_authoritative_rulespec_dependency_scope",
            return_value=nullcontext(),
        ),
        patch.object(evals, "_run_single_eval") as run_single,
    ):
        evals.run_model_eval(
            citations=["us/statute/26/1"],
            runner_specs=[
                "chosen=codex:test-model",
                "openai:test-model",
                "claude:opus",
            ],
            output_root=tmp_path / "output",
            policy_path=tmp_path / "policy",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=object(),
            **kwargs,
        )

    runners = [call.kwargs["runner"] for call in run_single.call_args_list]
    assert [runner.name for runner in runners] == [
        "chosen",
        "openai-test-model",
        "claude-opus",
    ]
    assert [runner.codex_reasoning_effort for runner in runners] == [
        effort or "low",
        "low",
        "low",
    ]


@pytest.mark.parametrize("effort", ["", " high", None, 1])
def test_run_model_eval_rejects_invalid_effort_before_resolving_sources(
    tmp_path, effort
):
    with (
        patch.object(evals, "resolve_corpus_source_unit") as resolve_source,
        pytest.raises(ValueError, match="Codex reasoning effort"),
    ):
        evals.run_model_eval(
            citations=["us/statute/26/1"],
            runner_specs=["codex:test-model"],
            output_root=tmp_path / "output",
            policy_path=tmp_path / "policy",
            runtime_axiom_rules_path=tmp_path / "engine",
            corpus_release=object(),
            codex_reasoning_effort=effort,
        )
    resolve_source.assert_not_called()


@pytest.mark.parametrize("effort", [None, "high", "ultra", 'future"effort'])
def test_codex_subprocess_uses_model_reasoning_effort_and_records_trace(
    tmp_path, monkeypatch, effort
):
    runner = evals.parse_runner_spec("codex:test-model")
    if effort is not None:
        runner = replace(runner, codex_reasoning_effort=effort)
    workspace = evals.EvalWorkspace(
        root=tmp_path,
        source_text_file=tmp_path / "source.txt",
        manifest_file=tmp_path / "manifest.json",
    )
    observed = {}

    class FakePopen:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            observed["cmd"] = cmd
            kwargs["stdout"].write(
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "generated output"},
                    }
                )
                + "\n"
            )
            kwargs["stdout"].flush()

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: "/explicit/codex")
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (20, 10))
    monkeypatch.setattr(evals.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(evals, "_wait_for_codex_process", lambda *a, **kw: False)

    response = evals._run_codex_prompt_eval(runner, workspace, "encode source")

    config_overrides = [
        observed["cmd"][index + 1]
        for index, arg in enumerate(observed["cmd"])
        if arg == "-c"
    ]
    assert config_overrides == [f"model_reasoning_effort={json.dumps(effort or 'low')}"]
    assert response.trace["reasoning_effort"] == (effort or "low")
    assert response.text == "generated output"
    assert response.error is None


def test_direct_codex_runner_rejects_invalid_effort_before_starting_process(tmp_path):
    runner = replace(
        evals.parse_runner_spec("codex:test-model"), codex_reasoning_effort=""
    )
    with (
        patch.object(evals.subprocess, "Popen") as popen,
        pytest.raises(ValueError, match="Codex reasoning effort"),
    ):
        evals._run_codex_prompt_eval(runner, None, "encode source")
    popen.assert_not_called()
