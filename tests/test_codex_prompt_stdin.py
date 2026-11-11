"""The Codex prompt travels on stdin, not argv (#1205).

The prompt embeds the whole source text. As one argv element it exceeds the
kernel's argument limits (131,072 bytes per argument on Linux, ARG_MAX in
total on macOS), so exec fails with E2BIG before Codex starts. These tests run
a real child process with a prompt larger than both limits.
"""

import hashlib
import json
import stat
import sys
from pathlib import Path

from axiom_encode.harness import backends, evals

# Larger than Linux MAX_ARG_STRLEN (131,072) and macOS ARG_MAX (1,048,576).
_LARGE_PROMPT = "Encode this source.\n" + ("statute text " * 120_000)

_FAKE_CODEX = """#!{python}
import hashlib, json, sys

argv = sys.argv[1:]
prompt = sys.stdin.read()
record = {{
    "argv": argv,
    "stdin_chars": len(prompt),
    "stdin_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
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
    fake = tmp_path / "fake-codex"
    fake.write_text(
        _FAKE_CODEX.format(python=sys.executable, record_path=str(record_path))
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return fake, record_path


def _assert_prompt_arrived_on_stdin(record_path: Path) -> None:
    record = json.loads(record_path.read_text())
    assert record["argv"][-1] == "-"
    assert max(len(arg) for arg in record["argv"]) < 4096
    assert record["stdin_chars"] == len(_LARGE_PROMPT)
    assert (
        record["stdin_sha256"]
        == hashlib.sha256(_LARGE_PROMPT.encode("utf-8")).hexdigest()
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
    _assert_prompt_arrived_on_stdin(record_path)


def test_eval_codex_prompt_sends_large_prompt_on_stdin(tmp_path, monkeypatch):
    fake, record_path = _install_fake_codex(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = evals.EvalWorkspace(
        root=workspace_root,
        source_text_file=workspace_root / "source.txt",
        manifest_file=workspace_root / "manifest.json",
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "isolated-home"))
    monkeypatch.setattr(evals, "resolve_codex_cli", lambda: str(fake))
    monkeypatch.setattr(evals, "_codex_prompt_timeouts", lambda _: (60, 30))

    response = evals._run_codex_prompt_eval(
        evals.parse_runner_spec("codex:test-model"), workspace, _LARGE_PROMPT
    )

    assert response.error is None
    assert response.text == "generated output"
    _assert_prompt_arrived_on_stdin(record_path)
