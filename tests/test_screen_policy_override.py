"""CLI mode overrides replace inherited modes before policy validation."""

import argparse
import json
import os

import pytest

from axiom_encode.judges import (
    JudgeEvent,
    JudgeStage,
    Verdict,
    cli_commands,
    statutory_fidelity,
    statutory_fidelity_screen,
)
from axiom_encode.judges.statutory_fidelity import FIDELITY_KINDS


@pytest.fixture
def screen_cli(monkeypatch, tmp_path):
    for name in (
        "AXIOM_JUDGE_SCREEN_MODE",
        "AXIOM_JUDGE_SCREEN_THRESHOLD",
        *(f"AXIOM_JUDGE_SCREEN_THRESHOLD_{kind.upper()}" for kind in FIDELITY_KINDS),
    ):
        monkeypatch.delenv(name, raising=False)

    class Source:
        body = "authoritative source"
        requested = "us/statute/26/1"

        def to_attestation(self):
            return {"requested_corpus_citation_path": self.requested}

    monkeypatch.setattr(cli_commands, "_load_bound_source", lambda *_: Source())
    state = {"policies": [], "referee_calls": 0}

    def screen_run(*_, policy, **kwargs):
        state["policies"].append(policy)
        return JudgeEvent(
            stage=JudgeStage.STATUTORY_FIDELITY_SCREEN,
            verdict=Verdict.PASS,
            extra={
                "screen": {"probabilities": {kind: 0.05 for kind in FIDELITY_KINDS}}
            },
        )

    def referee_run(*_, **kwargs):
        state["referee_calls"] += 1
        return JudgeEvent(stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.PASS)

    monkeypatch.setattr(statutory_fidelity_screen, "run", screen_run)
    monkeypatch.setattr(statutory_fidelity, "run", referee_run)
    rule_file = tmp_path / "rule.yaml"
    rule_file.write_text("rules: []\n", encoding="utf-8")
    args = argparse.Namespace(
        corpus_citation_path=Source.requested,
        rule_file=rule_file,
        rule_path=None,
        run_id=None,
        log_dir=None,
        json=True,
        screen=True,
        screen_mode=None,
    )
    return args, state


@pytest.mark.parametrize("command", ["judge-fidelity", "judge-fidelity-screen"])
@pytest.mark.parametrize("mode", ["advisory", "cascade"])
def test_explicit_mode_overrides_invalid_inherited_mode(
    command, mode, screen_cli, monkeypatch, capsys
):
    args, state = screen_cli
    args.command = command
    args.screen_mode = mode
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_MODE", "bad")
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_THRESHOLD", "0.42")
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_THRESHOLD_AMOUNT_MISMATCH", "0.31")

    assert cli_commands.dispatch(args) == 0
    assert len(state["policies"]) == 1
    policy = state["policies"][0]
    assert policy.mode == mode
    assert policy.thresholds == {"amount_mismatch": 0.31, "boundary_direction": 0.42}
    assert policy.source == "env"
    assert state["referee_calls"] == int(
        command == "judge-fidelity" and mode == "advisory"
    )
    assert os.environ["AXIOM_JUDGE_SCREEN_MODE"] == "bad"
    captured = capsys.readouterr()
    assert captured.err == ""
    json.loads(captured.out)


@pytest.mark.parametrize("command", ["judge-fidelity", "judge-fidelity-screen"])
@pytest.mark.parametrize("invalid_setting", ["mode", "threshold"])
def test_invalid_effective_policy_still_exits_before_judges_run(
    command, invalid_setting, screen_cli, monkeypatch, capsys
):
    args, state = screen_cli
    args.command = command
    if invalid_setting == "mode":
        monkeypatch.setenv("AXIOM_JUDGE_SCREEN_MODE", "bad")
    else:
        args.screen_mode = "advisory"
        monkeypatch.setenv("AXIOM_JUDGE_SCREEN_THRESHOLD", "very high")

    assert cli_commands.dispatch(args) == 2
    assert state["policies"] == []
    assert state["referee_calls"] == 0
    captured = capsys.readouterr()
    assert "invalid screen policy" in captured.err
    assert captured.out == ""
