from pathlib import Path
from subprocess import CompletedProcess

import pytest

from axiom_encode.rules_engine_compat import run_rulespec_compile


def _run(monkeypatch, responses):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return responses.pop(0)

    monkeypatch.setattr("axiom_encode.rules_engine_compat.subprocess.run", fake_run)
    result = run_rulespec_compile(
        binary=Path("/engine/axiom-rules-engine"),
        program=Path("/rulespec-us/us/policies/program.yaml"),
        rulespec_roots=(Path("/rulespec-us"),),
        output=Path("/tmp/program.json"),
        cwd=Path("/engine"),
        env={"PATH": "/bin", "AXIOM_RULESPEC_REPO_ROOTS": "/ambient"},
    )
    return result, calls


def test_current_explicit_root_contract_is_preferred(monkeypatch):
    success = CompletedProcess([], 0, "ok", "")

    result, calls = _run(monkeypatch, [success])

    assert result is success
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--rulespec-root") + 1] == "/rulespec-us"
    assert "--exclusive-rulespec-roots" not in command
    assert "AXIOM_RULESPEC_REPO_ROOTS" not in kwargs["env"]


def test_legacy_contract_requires_exact_unknown_flag_evidence(monkeypatch):
    unsupported = CompletedProcess(
        [],
        1,
        "",
        "unknown compile argument `--rulespec-root`\n"
        "usage: compile [--exclusive-rulespec-roots]",
    )
    success = CompletedProcess([], 0, "ok", "")

    result, calls = _run(monkeypatch, [unsupported, success])

    assert result is success
    assert len(calls) == 2
    command, kwargs = calls[1]
    assert "--rulespec-root" not in command
    assert "--exclusive-rulespec-roots" in command
    assert kwargs["env"]["AXIOM_RULESPEC_REPO_ROOTS"] == "/rulespec-us"
    assert kwargs["env"]["AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE"] == "1"


def test_other_compile_failures_do_not_fallback(monkeypatch):
    failure = CompletedProcess([], 1, "", "invalid RuleSpec")

    result, calls = _run(monkeypatch, [failure])

    assert result is failure
    assert len(calls) == 1


def test_unknown_flag_without_legacy_advertisement_does_not_fallback(monkeypatch):
    failure = CompletedProcess(
        [], 1, "", "unknown compile argument `--rulespec-root`"
    )

    result, calls = _run(monkeypatch, [failure])

    assert result is failure
    assert len(calls) == 1


def test_compile_requires_an_explicit_root():
    with pytest.raises(
        ValueError, match="RuleSpec engine compilation requires an explicit root"
    ):
        run_rulespec_compile(
            binary=Path("/engine/axiom-rules-engine"),
            program=Path("/rulespec-us/us/policies/program.yaml"),
            rulespec_roots=(),
            output=Path("/tmp/program.json"),
            cwd=Path("/engine"),
            env={"PATH": "/bin"},
        )
