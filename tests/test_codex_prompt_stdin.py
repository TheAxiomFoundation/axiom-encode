"""The Codex prompt travels on stdin, not argv (#1205).

The prompt embeds the whole source text. As one argv element it exceeds the
kernel's argument limits (131,072 bytes per argument on Linux, ARG_MAX in
total on macOS), so exec fails with E2BIG before Codex starts. These tests run
a real child process with a prompt larger than both limits.
"""

import hashlib
import json
import shlex
import stat
import sys
from pathlib import Path

from axiom_encode.harness import backends, evals

# Larger than Linux MAX_ARG_STRLEN (131,072) and macOS ARG_MAX (1,048,576).
_LARGE_PROMPT = "Encode this source.\n" + ("statute text " * 120_000)

_FAKE_CODEX = """import hashlib, json, sys

argv = sys.argv[1:]
# Raw bytes, so the record does not depend on how this child decodes stdin.
prompt = sys.stdin.buffer.read()
record = {{
    "argv": argv,
    "stdin_bytes": len(prompt),
    "stdin_sha256": hashlib.sha256(prompt).hexdigest(),
}}
with open({record_path!r}, "w") as handle:
    json.dump(record, handle)
if "-o" in argv:
    with open(argv[argv.index("-o") + 1], "w") as handle:
        handle.write("generated output")
print(json.dumps({{
    "type": "item.completed",
    "item": {{"type": "agent_message", "text": "generated output"}},
}}))
"""


def _install_fake_codex(tmp_path: Path) -> tuple[Path, Path]:
    record_path = tmp_path / "codex-invocation.json"
    script = tmp_path / "fake_codex.py"
    script.write_text(_FAKE_CODEX.format(record_path=str(record_path)))
    # A shell wrapper, not a Python shebang: a shebang line breaks on an
    # interpreter path that is long or contains a space.
    fake = tmp_path / "fake-codex"
    fake.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "
        '"$@"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, record_path


# Non-ASCII text plus a lone surrogate, which stands for the raw byte 0xFF.
_NON_ASCII_PROMPT = "26 U.S.C. § 32(b) — crédito, £1,000 \udcff"


def _assert_prompt_arrived_on_stdin(record_path: Path, prompt: str) -> None:
    expected = prompt.encode("utf-8", "surrogateescape")
    record = json.loads(record_path.read_text())
    assert record["argv"][-1] == "-"
    assert max(len(arg) for arg in record["argv"]) < 4096
    assert record["stdin_bytes"] == len(expected)
    assert record["stdin_sha256"] == hashlib.sha256(expected).hexdigest()


def _eval_workspace(tmp_path: Path) -> evals.EvalWorkspace:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    return evals.EvalWorkspace(
        root=workspace_root,
        source_text_file=workspace_root / "source.txt",
        manifest_file=workspace_root / "manifest.json",
    )


def test_prompt_exceeds_both_kernel_argument_limits():
    assert len(_LARGE_PROMPT.encode("utf-8")) > 1_048_576


def test_backend_codex_exec_sends_large_prompt_on_stdin(tmp_path, monkeypatch):
    fake, record_path = _install_fake_codex(tmp_path)
    monkeypatch.setattr(backends, "resolve_codex_cli", lambda: str(fake))
    workdir = tmp_path / "work"
    workdir.mkdir()

    output, returncode = backends.CodexCLIBackend()._run_codex_exec(
        _LARGE_PROMPT, "gpt-test", 60, workdir
    )

    assert returncode == 0, output[:500]
    assert "generated output" in output
    _assert_prompt_arrived_on_stdin(record_path, _LARGE_PROMPT)


def test_eval_codex_prompt_sends_large_prompt_on_stdin(tmp_path, monkeypatch):
    fake, record_path = _install_fake_codex(tmp_path)
    workspace = _eval_workspace(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: str(fake))
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (60, 30))

    response = evals._run_codex_prompt_eval(
        evals.parse_runner_spec("codex:test-model"), workspace, _LARGE_PROMPT
    )

    assert response.error is None
    assert response.text == "generated output"
    _assert_prompt_arrived_on_stdin(record_path, _LARGE_PROMPT)


def test_backend_codex_exec_sends_prompt_as_utf8_bytes(tmp_path, monkeypatch):
    fake, record_path = _install_fake_codex(tmp_path)
    monkeypatch.setattr(backends, "resolve_codex_cli", lambda: str(fake))

    output, returncode = backends.CodexCLIBackend()._run_codex_exec(
        _NON_ASCII_PROMPT, "gpt-test", 60, tmp_path
    )

    assert returncode == 0, output[:500]
    _assert_prompt_arrived_on_stdin(record_path, _NON_ASCII_PROMPT)


def test_eval_codex_prompt_sends_prompt_as_utf8_bytes(tmp_path, monkeypatch):
    fake, record_path = _install_fake_codex(tmp_path)
    workspace = _eval_workspace(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: str(fake))
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (60, 30))

    response = evals._run_codex_prompt_eval(
        evals.parse_runner_spec("codex:test-model"), workspace, _NON_ASCII_PROMPT
    )

    assert response.error is None
    _assert_prompt_arrived_on_stdin(record_path, _NON_ASCII_PROMPT)
