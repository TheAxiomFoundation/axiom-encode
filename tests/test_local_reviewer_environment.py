"""Reviewer CLI environment for verification-only local runs versus protected runs."""

from __future__ import annotations

import os
import pwd
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import signing_broker
from axiom_encode.cli import (
    _print_eval_metrics,
    _review_results_from_eval_metrics,
)
from axiom_encode.cli import main as cli_main
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.eval_evidence import scrub_attestation_signing_keys
from axiom_encode.toolchain import (
    RuleSpecToolchainError,
    local_corpus_release_verification,
    local_verification_active,
)
from tests.release_object_fixtures import TEST_RELEASE_PUBLIC_KEY
from tests.test_cli import _bind_test_corpus_release

# Credential sentinels: none of these names or values may reach a local
# reviewer child.
_AMBIENT_SECRETS = {
    "OPENAI_API_KEY": "sk-ambient-openai",
    "CODEX_API_KEY": "sk-ambient-codex",
    "ANTHROPIC_API_KEY": "sk-ambient-anthropic",
    "GITHUB_TOKEN": "ghp-ambient",
    "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY": "private-eval",
    "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY": "private-apply",
    "SUPABASE_SERVICE_ROLE_KEY": "service-role",
    "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "supabase-secret",
    "AXIOM_ENCODE_SUPABASE_URL": "https://supabase.invalid",
    "R2_ACCESS_KEY_ID": "r2-access-id",
    "R2_SECRET_ACCESS_KEY": "r2-secret",
    "R2_ENDPOINT": "https://r2.invalid",
}
_PROTECTED_SIGNALS = (
    "broker_base",
    "active_broker",
    "trusted_runtime",
    "github_actions",
    "ci",
)


@pytest.fixture
def plain_process(monkeypatch):
    """No broker, no trusted runtime, no CI, and credential sentinels set."""
    monkeypatch.setattr(signing_broker, "_trusted_subprocess_base_environment", None)
    monkeypatch.setattr(signing_broker, "_active_broker", None)
    for name in ("AXIOM_ENCODE_TRUSTED_RUNTIME", "GITHUB_ACTIONS", "CI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_REVIEWER_CLI", raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_CODEX_BIN", raising=False)
    for name, value in _AMBIENT_SECRETS.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def local_context(plain_process):
    """A verification-only local run: the context encode and ci enter."""
    with local_corpus_release_verification(TEST_RELEASE_PUBLIC_KEY):
        yield


def _fake_bin(tmp_path, name, body="#!/bin/sh\n"):
    directory = tmp_path / f"{name}-bin"
    directory.mkdir()
    executable = directory / name
    executable.write_text(body)
    executable.chmod(0o755)
    return executable


_ENV_SHEBANGS = (
    "#!/usr/bin/env fake-node",
    "#!/usr/bin/env -S -u NODE_OPTIONS fake-node --no-warnings",
)


def _env_shebang_reviewer(tmp_path, shebang=_ENV_SHEBANGS[0]):
    """Return a ``claude`` launcher whose ``env`` shebang names an interpreter.

    The interpreter lives in another directory, as with an npm-installed CLI
    whose Node runtime is elsewhere. It prints the environment it received.
    """
    interpreter = _fake_bin(tmp_path, "fake-node", "#!/bin/sh\nexec /usr/bin/env\n")
    launcher = _fake_bin(tmp_path, "claude", f"{shebang}\nconsole.log('cli')\n")
    return launcher, interpreter


def _child_environment(output: str) -> dict[str, str]:
    environment = {}
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator:
            environment[name] = value
    return environment


def _capture_runs(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return validator_pipeline._SubprocessRunResult(
            output='{"passed": true}', returncode=0
        )

    monkeypatch.setattr(
        validator_pipeline, "_run_subprocess_with_idle_timeout", fake_run
    )
    return calls


def _set_protected_signal(monkeypatch, protect):
    if protect == "broker_base":
        monkeypatch.setattr(
            signing_broker,
            "_trusted_subprocess_base_environment",
            {"HOME": "/trusted-home", "PATH": "/trusted/bin"},
        )
    elif protect == "active_broker":
        monkeypatch.setattr(signing_broker, "_active_broker", object())
    elif protect == "trusted_runtime":
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
    elif protect == "github_actions":
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
    else:
        monkeypatch.setenv("CI", "true")


def _assert_local_environment_allowlist(env):
    account = pwd.getpwuid(os.getuid())
    assert env["HOME"] == account.pw_dir
    assert env["USER"] == account.pw_name
    for name in _AMBIENT_SECRETS:
        assert name not in env
    assert all(value not in env.values() for value in _AMBIENT_SECRETS.values())
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "XDG_CONFIG_HOME" not in env


def test_local_claude_reviewer_gets_allowlisted_environment(
    tmp_path, monkeypatch, local_context
):
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", f"{claude.parent}{os.pathsep}/usr/bin")
    calls = _capture_runs(monkeypatch)

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert (output, returncode) == ('{"passed": true}', 0)
    cmd, kwargs = calls[0]
    assert cmd[0] == str(claude)
    assert kwargs["env_is_final"] is True
    env = kwargs["env"]
    _assert_local_environment_allowlist(env)
    assert env["PATH"].split(os.pathsep)[0] == str(claude.parent)
    assert set(env["PATH"].split(os.pathsep)) <= {
        str(claude.parent),
        os.path.dirname(os.path.realpath(claude)),
        *validator_pipeline._LOCAL_REVIEWER_SYSTEM_PATH,
    }
    allowed = set(scrub_attestation_signing_keys()) | {
        "USER",
        "LOGNAME",
        "TMPDIR",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
    }
    assert set(env) <= allowed


def test_local_codex_fallback_gets_absolute_binary_and_environment(
    tmp_path, monkeypatch, local_context
):
    codex = _fake_bin(tmp_path, "codex")
    monkeypatch.setenv("PATH", str(codex.parent))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    validator_pipeline.run_claude_code("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == str(codex)
    assert cmd[1] == "exec"
    assert kwargs["env_is_final"] is True
    assert kwargs["env"]["HOME"] == pwd.getpwuid(os.getuid()).pw_dir
    assert "OPENAI_API_KEY" not in kwargs["env"]


def test_local_missing_reviewer_clis_is_an_explicit_error(
    tmp_path, monkeypatch, local_context
):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert calls == []
    assert returncode == 1
    assert output.startswith("reviewer_unavailable:")


def test_missing_reviewer_is_a_failed_unscored_review(
    tmp_path, monkeypatch, local_context
):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    rulespec = tmp_path / "rule.yaml"
    rulespec.write_text("format: rulespec/v1\n")
    pipeline = object.__new__(validator_pipeline.ValidatorPipeline)
    pipeline.policy_repo_path = tmp_path

    result = pipeline._run_reviewer("generalist-reviewer", rulespec)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_unavailable" in (result.error or "")


@pytest.mark.parametrize("shebang", _ENV_SHEBANGS)
def test_local_reviewer_child_process_keeps_only_the_allowlist(
    tmp_path, monkeypatch, local_context, shebang
):
    """A real child (no mocked subprocess helper) sees only the allowlist."""
    launcher, interpreter = _env_shebang_reviewer(tmp_path, shebang)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(launcher.parent), str(interpreter.parent), "/usr/bin"]),
    )

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert returncode == 0, output
    child = _child_environment(output)
    _assert_local_environment_allowlist(child)
    child_path = child["PATH"].split(os.pathsep)
    assert str(interpreter.parent) in child_path
    assert str(launcher.parent) in child_path
    for value in _AMBIENT_SECRETS.values():
        assert value not in output


def _marked_interpreters(tmp_path):
    first = tmp_path / "first-bin"
    second = tmp_path / "second-bin"
    for directory, marker in ((first, "first"), (second, "second")):
        directory.mkdir()
        interpreter = directory / "fake-node"
        interpreter.write_text(f"#!/bin/sh\necho MARKER={marker}\nexec /usr/bin/env\n")
        interpreter.chmod(0o755)
    decoy = second / "decoy"
    decoy.write_text("#!/bin/sh\n")
    decoy.chmod(0o755)
    return first, second


def test_relative_caller_path_entry_keeps_its_place(
    tmp_path, monkeypatch, local_context
):
    """A relative PATH entry resolves against the current directory, in place."""
    first, second = _marked_interpreters(tmp_path)
    # The decoy word puts second-bin on the child PATH too.
    launcher = _fake_bin(
        tmp_path, "claude", "#!/usr/bin/env -S -u decoy fake-node\nconsole.log(1)\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(launcher.parent), first.name, str(second), "/usr/bin"]),
    )

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert returncode == 0, output
    assert "MARKER=first" in output
    child_path = _child_environment(output)["PATH"].split(os.pathsep)
    assert all(os.path.isabs(entry) for entry in child_path)
    assert child_path.index(os.path.abspath(first.name)) < child_path.index(
        os.path.abspath(second)
    )


def test_shebang_word_collision_keeps_the_caller_interpreter(
    tmp_path, monkeypatch, local_context
):
    """An option operand that resolves elsewhere cannot reorder the PATH."""
    first, second = _marked_interpreters(tmp_path)
    launcher = _fake_bin(
        tmp_path, "claude", "#!/usr/bin/env -S -u decoy fake-node\nconsole.log(1)\n"
    )
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(launcher.parent), str(first), str(second), "/usr/bin"]),
    )

    output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert returncode == 0, output
    assert "MARKER=first" in output
    child_path = _child_environment(output)["PATH"].split(os.pathsep)
    assert child_path.index(str(first)) < child_path.index(str(second))


def test_env_shebang_interpreter_directory_is_load_bearing(
    tmp_path, monkeypatch, local_context
):
    """Without the shebang interpreter's directory the launcher cannot start."""
    launcher, interpreter = _env_shebang_reviewer(tmp_path)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(launcher.parent), str(interpreter.parent), "/usr/bin"]),
    )
    monkeypatch.setattr(
        validator_pipeline, "_shebang_env_program_names", lambda _executable: ()
    )

    _output, returncode = validator_pipeline.run_claude_code("review", cwd=tmp_path)

    assert returncode != 0


@pytest.mark.parametrize(
    ("first_line", "expected"),
    [
        ("#!/usr/bin/env node\n", ("node",)),
        ("#!/usr/bin/env -S node --no-warnings\n", ("node",)),
        ("#!/usr/bin/env NODE_OPTIONS=--x node\n", ("node",)),
        ("#!/usr/bin/env -S -u NODE_OPTIONS node\n", ("NODE_OPTIONS", "node")),
        ("#!/usr/bin/env -S 'node --x' -C /tmp\n", ("node",)),
        ("#!/usr/bin/env -S 'unbalanced node\n", ("'unbalanced", "node")),
        ("#!/bin/sh\n", ()),
        ("#!/usr/bin/env ./relative\n", ()),
        ("not a shebang\n", ()),
    ],
)
def test_shebang_env_program_names(tmp_path, first_line, expected):
    script = tmp_path / "launcher"
    script.write_text(first_line)
    assert validator_pipeline._shebang_env_program_names(str(script)) == expected


def test_no_verification_context_keeps_the_protected_environment(
    tmp_path, monkeypatch, plain_process
):
    """Fail closed: without the verification-only context nothing changes."""
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", str(claude.parent))
    expected_env = scrub_attestation_signing_keys()
    calls = _capture_runs(monkeypatch)

    validator_pipeline.run_claude_code("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == "claude"
    assert "env_is_final" not in kwargs
    assert kwargs["env"] == expected_env


@pytest.mark.parametrize("protect", _PROTECTED_SIGNALS)
def test_protected_reviewer_environment_is_unchanged(
    tmp_path, monkeypatch, local_context, protect
):
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", str(claude.parent))
    _set_protected_signal(monkeypatch, protect)
    expected_env = scrub_attestation_signing_keys()
    calls = _capture_runs(monkeypatch)

    validator_pipeline.run_claude_code("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == "claude"
    assert "env_is_final" not in kwargs
    assert kwargs["env"] == expected_env


def test_protected_codex_reviewer_call_is_unchanged(
    tmp_path, monkeypatch, local_context
):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    calls = _capture_runs(monkeypatch)

    validator_pipeline._run_codex_reviewer_cli("review", cwd=tmp_path)

    cmd, kwargs = calls[0]
    assert cmd[0] == "codex"
    assert set(kwargs) == {"timeout", "idle_timeout", "cwd"}


def test_unscrubbed_environment_requires_explicit_env():
    with pytest.raises(ValueError):
        validator_pipeline._run_subprocess_with_idle_timeout(
            ["true"], timeout=1, idle_timeout=1, env_is_final=True
        )


# ---------------------------------------------------------------------------
# The documented local entry point: encode --corpus-release-public-key
# ---------------------------------------------------------------------------

_CITATION = "us/statute/26/1"


class _ReachedModelEval(Exception):
    """Raised by the stubbed model eval once the reviewer probe has run."""


def _encode_argv(tmp_path, *extra):
    policy_repo = tmp_path / "rulespec-us"
    (policy_repo / "us").mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "axiom-corpus"
    corpus.mkdir(exist_ok=True)
    engine = tmp_path / "axiom-rules-engine"
    engine.mkdir(exist_ok=True)
    _bind_test_corpus_release(policy_repo, corpus, citation_path=_CITATION)
    return [
        "axiom-encode",
        "encode",
        _CITATION,
        "--corpus-path",
        str(corpus),
        "--axiom-rules-engine-path",
        str(engine),
        "--policy-repo-path",
        str(policy_repo),
        "--output",
        str(tmp_path / "out"),
        "--db",
        str(tmp_path / "encodings.db"),
        "--no-sync",
        *extra,
    ]


def _run_encode_to_reviewer_probe(monkeypatch, argv, cwd):
    """Run the real encode entry point; probe the reviewer at model eval."""
    seen = {}

    def probe_reviewer(**kwargs):
        seen["corpus_release"] = kwargs["corpus_release"]
        seen["local_verification_active"] = local_verification_active()
        seen["reviewer"] = validator_pipeline.run_claude_code("review", cwd=cwd)
        raise _ReachedModelEval

    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("axiom_encode.cli.codex_auth_error", return_value=None),
        patch("axiom_encode.cli.run_model_eval", side_effect=probe_reviewer),
        pytest.raises(_ReachedModelEval),
    ):
        cli_main()
    return seen


def test_local_encode_entry_reaches_local_reviewer_environment(
    tmp_path, monkeypatch, plain_process
):
    """encode --corpus-release-public-key loads the corpus, then reviews locally."""
    launcher, interpreter = _env_shebang_reviewer(tmp_path)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            [str(launcher.parent), str(interpreter.parent), "/usr/bin", "/bin"]
        ),
    )
    argv = _encode_argv(
        tmp_path, "--corpus-release-public-key", TEST_RELEASE_PUBLIC_KEY
    )

    seen = _run_encode_to_reviewer_probe(monkeypatch, argv, tmp_path)

    assert seen["local_verification_active"] is True
    assert tuple(seen["corpus_release"].public_key) == (TEST_RELEASE_PUBLIC_KEY,)
    output, returncode = seen["reviewer"]
    assert returncode == 0, output
    _assert_local_environment_allowlist(_child_environment(output))
    # The context ends with the command.
    assert local_verification_active() is False


def test_plain_encode_without_broker_stops_before_model_eval(
    tmp_path, monkeypatch, plain_process
):
    """Without the flag or a broker, corpus loading fails closed as before."""
    argv = _encode_argv(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("axiom_encode.cli.codex_auth_error", return_value=None),
        patch("axiom_encode.cli.run_model_eval") as model_eval,
        pytest.raises(RuleSpecToolchainError, match="protected signing broker"),
    ):
        cli_main()
    model_eval.assert_not_called()


def test_encode_under_broker_keeps_the_protected_reviewer_environment(
    tmp_path, monkeypatch
):
    """With the verification broker the conftest attaches, nothing changes."""
    for name in ("AXIOM_ENCODE_TRUSTED_RUNTIME", "GITHUB_ACTIONS", "CI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_REVIEWER_CLI", raising=False)
    claude = _fake_bin(tmp_path, "claude")
    monkeypatch.setenv("PATH", os.pathsep.join([str(claude.parent), "/usr/bin"]))
    calls = _capture_runs(monkeypatch)
    expected_env = scrub_attestation_signing_keys()

    seen = _run_encode_to_reviewer_probe(monkeypatch, _encode_argv(tmp_path), tmp_path)

    assert seen["local_verification_active"] is False
    cmd, kwargs = calls[0]
    assert cmd[0] == "claude"
    assert "env_is_final" not in kwargs
    assert kwargs["env"] == expected_env


def test_local_encode_entry_refuses_apply(tmp_path, monkeypatch, plain_process):
    argv = _encode_argv(
        tmp_path,
        "--corpus-release-public-key",
        TEST_RELEASE_PUBLIC_KEY,
        "--apply",
    )
    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("axiom_encode.cli.run_model_eval") as model_eval,
        pytest.raises(ValueError, match="cannot --apply"),
    ):
        cli_main()
    model_eval.assert_not_called()


@pytest.mark.parametrize("protect", _PROTECTED_SIGNALS)
def test_local_encode_entry_refuses_a_protected_process(
    tmp_path, monkeypatch, plain_process, protect
):
    _set_protected_signal(monkeypatch, protect)
    argv = _encode_argv(
        tmp_path, "--corpus-release-public-key", TEST_RELEASE_PUBLIC_KEY
    )
    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("axiom_encode.cli.run_model_eval") as model_eval,
        pytest.raises(ValueError, match="verification-only local run"),
    ):
        cli_main()
    model_eval.assert_not_called()
    assert local_verification_active() is False


def test_local_roots_require_the_corpus_release_key(
    tmp_path, monkeypatch, plain_process
):
    argv = _encode_argv(tmp_path, "--apply-public-key", TEST_RELEASE_PUBLIC_KEY)
    monkeypatch.setattr(sys, "argv", argv)
    with (
        patch("axiom_encode.cli.run_model_eval") as model_eval,
        pytest.raises(ValueError, match="require --corpus-release-public-key"),
    ):
        cli_main()
    model_eval.assert_not_called()


def _metrics(**overrides):
    values = dict(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        ungrounded_numeric_count=0,
        grounded_numeric_count=1,
        embedded_source_present=True,
        grounding=[],
        generalist_review_pass=False,
        generalist_review_score=None,
        generalist_review_issues=["Reviewer error: reviewer_unavailable: missing"],
        policyengine_score=None,
        policyengine_pass=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reviewer_error_is_recorded_and_printed(capsys):
    metrics = _metrics()

    review_results = _review_results_from_eval_metrics(metrics)
    _print_eval_metrics(SimpleNamespace(metrics=metrics))

    integration = [
        review
        for review in review_results.reviews
        if review.reviewer == "integration_reviewer"
    ]
    assert len(integration) == 1
    assert integration[0].passed is False
    assert integration[0].critical_issues == metrics.generalist_review_issues
    captured = capsys.readouterr()
    assert "generalist_review=error Reviewer error: reviewer_unavailable" in (
        captured.out
    )
    assert "was not reviewed" in captured.err


def test_skipped_reviewer_is_not_reported_as_error(capsys):
    metrics = _metrics(generalist_review_pass=True, generalist_review_issues=[])

    review_results = _review_results_from_eval_metrics(metrics)
    _print_eval_metrics(SimpleNamespace(metrics=metrics))

    assert all(
        review.reviewer != "integration_reviewer" for review in review_results.reviews
    )
    captured = capsys.readouterr()
    assert "generalist_review" not in captured.out
    assert captured.err == ""
