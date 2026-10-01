import os
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from axiom_encode.rules_engine_compat import run_rulespec_compile


def _run(monkeypatch, responses, *, composed=False):
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
        composed=composed,
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


def test_composed_program_uses_dedicated_surface_without_legacy_fallback(monkeypatch):
    failure = CompletedProcess(
        [],
        1,
        "",
        "unknown compile argument `--rulespec-root`\n"
        "usage: compile [--exclusive-rulespec-roots]",
    )

    result, calls = _run(monkeypatch, [failure], composed=True)

    assert result is failure
    assert len(calls) == 1
    assert calls[0][0][1] == "compile-composed"


def test_real_engine_accepts_nested_composed_program(tmp_path):
    raw_binary = os.environ.get("AXIOM_RULES_ENGINE_BINARY")
    if not raw_binary:
        pytest.skip("set AXIOM_RULES_ENGINE_BINARY for the real engine contract test")
    binary = Path(raw_binary).resolve(strict=True)
    rulespec_root = (tmp_path / "rulespec-us").resolve()
    atomic = rulespec_root / "us/policies/base.yaml"
    atomic.parent.mkdir(parents=True)
    atomic.write_text(
        """format: rulespec/v1
rules:
  - name: base_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: 2026-01-01
        formula: "10"
"""
    )
    nested = rulespec_root / "us-az/policies/state-program.yaml"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        """format: rulespec/v1
module:
  kind: composition
  summary: State composition imported by a generated program.
imports:
  - us:policies/base
rules:
  - name: adjusted_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: 2026-01-01
        formula: base_amount
"""
    )
    composed = (tmp_path / "generated/composition.yaml").resolve()
    composed.parent.mkdir()
    composed.write_text(
        """format: rulespec/v1
module:
  kind: composition
  summary: Real compile-composed contract test.
imports:
  - us-az:policies/state-program
"""
    )
    output = tmp_path / "compiled.json"

    result = run_rulespec_compile(
        binary=binary,
        program=composed,
        rulespec_roots=(rulespec_root,),
        output=output,
        cwd=binary.parent,
        env={"PATH": os.environ.get("PATH", "")},
        composed=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file()


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
    failure = CompletedProcess([], 1, "", "unknown compile argument `--rulespec-root`")

    result, calls = _run(monkeypatch, [failure])

    assert result is failure
    assert len(calls) == 1


def test_legacy_contract_rejects_roots_containing_path_separator(monkeypatch):
    unsupported = CompletedProcess(
        [],
        1,
        "",
        "unknown compile argument `--rulespec-root`\n"
        "usage: compile [--exclusive-rulespec-roots]",
    )
    monkeypatch.setattr(
        "axiom_encode.rules_engine_compat.subprocess.run",
        lambda command, **kwargs: unsupported,
    )
    root = Path(f"/rulespec{os.pathsep}other/rulespec-us")

    with pytest.raises(ValueError, match="platform path separator"):
        run_rulespec_compile(
            binary=Path("/engine/axiom-rules-engine"),
            program=Path("/rulespec-us/us/policies/program.yaml"),
            rulespec_roots=(root,),
            output=Path("/tmp/program.json"),
            cwd=Path("/engine"),
            env={"PATH": "/bin"},
        )


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
