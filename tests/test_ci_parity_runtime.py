"""Runtime helpers, release acquisition and reporting of ``axiom-encode ci``.

Expected values come from the pinned validate-rulespec workflows packaged in
``axiom_encode/ci_parity_workflows`` (the "Fetch pinned signed corpus release
object" step, the heredoc/``python3 -c`` programs every gate executes, and the
protected supervisor's child environment), not from the implementation.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from argparse import Namespace
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
import yaml

from axiom_encode import __version__, ci_parity
from axiom_encode import toolchain as toolchain_module
from axiom_encode.ci_parity import (
    _INPUT_TYPE_DEFAULTS,
    _ISOLATED_CLI_BOOTSTRAP,
    DEFAULT_RELEASE_BASE_URL,
    MAX_RELEASE_DOWNLOAD_BYTES,
    RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS,
    SUPERVISOR_FORWARDED_ENVIRONMENT,
    SUPPORTED_WORKFLOW_PINS,
    WAIVER_AUDIT_WORKERS_ENV,
    WORKFLOW_DIRECTORY,
    CallerConfig,
    CallerOverrides,
    CliInvocation,
    DependencyMismatch,
    GateResult,
    PinnedWorkflow,
    PullRequestSimulation,
    ShardPlan,
    WorkflowRun,
    _fetch_https,
    _git_blob_id,
    _heredoc_bodies,
    _HttpsOnlyRedirects,
    _isolated_cli_main,
    _run_cli,
    _run_cli_batch,
    _run_cli_isolated,
    _run_embedded_python,
    _single_quoted_python,
    acquire_release_object,
    acquire_workflow_release_object,
    register_ci_parser,
    run_ci,
    verify_ambient_encoder,
    verify_dependency_checkout,
)
from axiom_encode.corpus_resolver import MAX_RELEASE_OBJECT_BYTES
from axiom_encode.toolchain import (
    RuleSpecToolchain,
    RuleSpecToolchainError,
    local_corpus_release_verification,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
PIN_0EFFA6A5 = "0effa6a5b05e7fac53902df7d523e909bd7fc48a"
PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
LEGACY_PIN = "615c1df9b9ace7deea84da65efd137f46f8bad2b"
EMBEDDED_PINS = (PIN_0EFFA6A5, PIN_6F11BE26)
CURRENT_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
APPLY_KEY = base64.b64encode(bytes(range(128, 160))).decode("ascii")
RETIRED_KEY = base64.b64encode(bytes(range(32, 64))).decode("ascii")
OTHER_RETIRED_KEY = base64.b64encode(bytes(range(64, 96))).decode("ascii")
RELEASE = "us-rulespec-2026-09-01"
REFS = {
    "encode": "1" * 40,
    "engine": "2" * 40,
    "corpus": "3" * 40,
    "rulespec_us": "4" * 40,
}
FETCH_STEP = "Fetch pinned signed corpus release object"
REGISTRY_HEADERS = {
    "apikey": "anon-key",
    "Authorization": "Bearer anon-key",
    "Accept-Profile": "corpus",
}

# Every (job, step, extraction) ci_parity.py executes from a pinned workflow:
# PinnedWorkflow.python(...) always takes occurrence 0 and
# PinnedWorkflow.inline_python(...) the one `python3 -c '...'` program.
EMBEDDED_SCRIPT_STEPS_0EFFA6A5: tuple[tuple[str, str, str], ...] = (
    ("shards", "Reject unsupported tracked paths", "inline"),
    ("shards", "Compute validation shards", "heredoc"),
    ("validate", "Authorize exact reviewed migration", "heredoc"),
    ("validate", "Resolve RuleSpec toolchain", "heredoc"),
    ("validate", "Verify immutable retired-schema freeze", "heredoc"),
    ("validate", FETCH_STEP, "heredoc"),
    ("validate", "Reject disallowed repository layout", "heredoc"),
    ("validate", "Enforce validation waiver ratchet", "heredoc"),
    ("validate", "Select RuleSpec validation targets", "heredoc"),
    ("validate", "Validate RuleSpec YAML", "heredoc"),
    ("validate", "Validate RuleSpec proofs and claims", "heredoc"),
    (
        "validate",
        "Validate changed PolicyEngine oracle coverage classification",
        "heredoc",
    ),
)
EMBEDDED_SCRIPT_STEPS = {
    PIN_0EFFA6A5: EMBEDDED_SCRIPT_STEPS_0EFFA6A5,
    PIN_6F11BE26: (
        *EMBEDDED_SCRIPT_STEPS_0EFFA6A5,
        ("shards", "Reject unmanifested RuleSpec content", "heredoc"),
    ),
}
HEREDOC_MARKER_RE = re.compile(r"<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*")


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _commit_all(path: Path, message: str) -> str:
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", message)
    return _git(path, "rev-parse", "HEAD")


def _bash_available() -> bool:
    return shutil.which("bash") is not None


needs_bash = pytest.mark.skipif(not _bash_available(), reason="bash is required")


@pytest.fixture(autouse=True)
def _hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's Git configuration and hook context out of tests."""

    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# PinnedWorkflow
# ---------------------------------------------------------------------------


def test_git_blob_id_is_git_hash_object() -> None:
    # Well-known git blob ids anchor the object-header format.
    assert _git_blob_id(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert _git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.parametrize("sha", list(SUPPORTED_WORKFLOW_PINS), ids=lambda s: s[:8])
def test_every_packaged_workflow_is_its_pinned_git_blob(sha: str) -> None:
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    path = WORKFLOW_DIRECTORY / pin.fixture
    data = path.read_bytes()
    hashed = subprocess.run(
        ["git", "hash-object", "--stdin"],
        input=data,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode()

    assert hashed.strip() == pin.workflow_blob
    assert _git_blob_id(data) == pin.workflow_blob

    workflow = PinnedWorkflow(sha)

    assert workflow.sha == sha
    assert workflow.pin is pin
    assert workflow.path == path
    assert workflow.payload == yaml.safe_load(data)


def _tamper_append_newline(data: bytes) -> bytes:
    return data + b"\n"


def _tamper_crlf(data: bytes) -> bytes:
    return data.replace(b"\n", b"\r\n")


def _tamper_flip_middle_byte(data: bytes) -> bytes:
    middle = len(data) // 2
    return data[:middle] + bytes([data[middle] ^ 0x01]) + data[middle + 1 :]


@pytest.mark.parametrize("sha", list(SUPPORTED_WORKFLOW_PINS), ids=lambda s: s[:8])
@pytest.mark.parametrize(
    "tamper",
    [_tamper_append_newline, _tamper_crlf, _tamper_flip_middle_byte],
    ids=["appended-newline", "crlf", "flipped-byte"],
)
def test_tampered_packaged_workflow_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sha: str,
    tamper: Callable[[bytes], bytes],
) -> None:
    for pin in SUPPORTED_WORKFLOW_PINS.values():
        shutil.copyfile(WORKFLOW_DIRECTORY / pin.fixture, tmp_path / pin.fixture)
    monkeypatch.setattr(ci_parity, "WORKFLOW_DIRECTORY", tmp_path)
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    target = tmp_path / pin.fixture

    # The untampered copy is accepted from the substituted directory.
    assert PinnedWorkflow(sha).path == target

    tampered = tamper(target.read_bytes())
    target.write_bytes(tampered)

    with pytest.raises(ValueError) as error:
        PinnedWorkflow(sha)

    message = str(error.value)
    assert str(target) in message
    assert _git_blob_id(tampered) in message
    assert pin.workflow_blob in message
    assert f"validate-rulespec@{sha}" in message


@pytest.mark.parametrize("sha", EMBEDDED_PINS, ids=lambda s: s[:8])
def test_step_lookup_rejects_unknown_and_misplaced_step_names(sha: str) -> None:
    workflow = PinnedWorkflow(sha)

    with pytest.raises(ValueError) as error:
        workflow.step("validate", "No such step")
    assert str(error.value) == (
        f"validate-rulespec@{sha[:8]} job validate has 0 steps named 'No such step'"
    )
    # A real step name looked up in the wrong job is not silently found.
    with pytest.raises(ValueError, match="job validate has 0 steps named"):
        workflow.python("validate", "Compute validation shards")
    with pytest.raises(ValueError, match="job shards has 0 steps named"):
        workflow.inline_python("shards", "Validate RuleSpec YAML")
    with pytest.raises(KeyError):
        workflow.step("no-such-job", "Validate RuleSpec YAML")


def test_step_lookup_rejects_ambiguous_step_names() -> None:
    workflow = PinnedWorkflow(PIN_0EFFA6A5)
    steps = workflow.payload["jobs"]["validate"]["steps"]
    duplicate = next(step for step in steps if step.get("name") == FETCH_STEP)
    workflow.payload["jobs"]["validate"]["steps"] = [*steps, dict(duplicate)]

    with pytest.raises(ValueError, match=f"has 2 steps named '{FETCH_STEP}'"):
        workflow.step("validate", FETCH_STEP)


def test_unmanifested_precheck_exists_only_in_the_pin_that_runs_it() -> None:
    name = "Reject unmanifested RuleSpec content"

    assert PinnedWorkflow(PIN_6F11BE26).step("shards", name)["name"] == name
    with pytest.raises(ValueError, match="0 steps named"):
        PinnedWorkflow(PIN_0EFFA6A5).step("shards", name)


# ---------------------------------------------------------------------------
# Script extraction
# ---------------------------------------------------------------------------


def test_heredoc_bodies_returns_every_quoted_heredoc_in_order() -> None:
    run = (
        "set -euo pipefail\n"
        "python - <<'PY' >> \"$GITHUB_OUTPUT\"\n"
        "print('one')\n"
        "PY\n"
        "echo between\n"
        'python - "$a" <<\'SKIP\' > "$skip_list"\n'
        "print('two')\n"
        "  PY\n"
        "SKIP\n"
        "echo done\n"
    )

    assert _heredoc_bodies(run) == ["print('one')\n", "print('two')\n  PY\n"]


def test_heredoc_marker_on_a_backslash_continued_line() -> None:
    run = (
        'python - "$protected_base" known-validation-gaps.yaml \\\n'
        '  "$protected_base_toolchain" .axiom/toolchain.toml "$changed_paths" \\\n'
        "  \"$audit_changed_paths\" <<'PY'\n"
        "import sys\n"
        "print(sys.argv[1:])\n"
        "PY\n"
        "/opt/axiom-verification/axiom-encode-signing-supervisor \\\n"
        "  --flag\n"
    )

    assert _heredoc_bodies(run) == ["import sys\nprint(sys.argv[1:])\n"]


def test_heredoc_body_text_is_not_rescanned_for_markers() -> None:
    run = "cat <<'A'\necho <<'B'\nA\n"

    assert _heredoc_bodies(run) == ["echo <<'B'\n"]


def test_only_quoted_heredocs_are_extracted() -> None:
    assert _heredoc_bodies("cat <<EOF\nexpanded $HOME\nEOF\n") == []


@pytest.mark.parametrize(
    "run",
    [
        "python - <<'PY'\nprint(1)\n",
        "python - <<'PY'\nprint(1)\n  PY\nPY \nPYX\n",
        "python - <<'PY'\nprint(1)\nPY\ncat <<'SKIP'\nx\n",
    ],
    ids=["no-terminator", "near-miss-terminators", "second-unterminated"],
)
def test_unterminated_heredoc_raises(run: str) -> None:
    with pytest.raises(ValueError, match=r"Unterminated heredoc (PY|SKIP)"):
        _heredoc_bodies(run)


@needs_bash
def test_heredoc_extraction_matches_what_bash_feeds_the_interpreter() -> None:
    body_lines = [
        "import os",
        "print(f'$HOME `date` \\\\ {os.getcwd()!r}')",
        "  PY",
        "PY ",
        "PYX",
        "    # indented comment",
    ]
    run = "cat \\\n  <<'PY'\n" + "\n".join(body_lines) + "\nPY\necho tail >&2\n"
    fed = subprocess.run(
        ["bash", "-c", run], check=True, stdout=subprocess.PIPE
    ).stdout.decode()

    assert _heredoc_bodies(run) == [fed]


def test_single_quoted_python_extracts_the_program() -> None:
    run = (
        "{\n"
        "  git ls-files -z\n"
        "} | python3 -c '\n"
        "import sys\n"
        "print(sys.stdin.read())\n"
        "'\n"
        "echo after\n"
    )

    assert _single_quoted_python(run) == "\nimport sys\nprint(sys.stdin.read())\n"
    with pytest.raises(ValueError):
        _single_quoted_python("python - <<'PY'\nprint(1)\nPY\n")


@pytest.mark.parametrize(
    ("sha", "job", "name", "kind"),
    [
        (sha, job, name, kind)
        for sha, steps in EMBEDDED_SCRIPT_STEPS.items()
        for job, name, kind in steps
    ],
    ids=lambda value: value[:8] if re.fullmatch(r"[0-9a-f]{40}", str(value)) else None,
)
def test_every_executed_workflow_script_is_found_and_compiles(
    sha: str, job: str, name: str, kind: str
) -> None:
    workflow = PinnedWorkflow(sha)
    run = workflow.step(job, name)["run"]

    if kind == "inline":
        script = workflow.inline_python(job, name)
        marker = "python3 -c '"
        start = run.index(marker) + len(marker)
        # bash ends a single-quoted word at the next quote, so the program
        # must contain none for the extractor and bash to agree.
        assert "'" not in script
        assert run[start : start + len(script)] == script
        assert run[start + len(script)] == "'"
    else:
        script = workflow.python(job, name)
        bodies = _heredoc_bodies(run)
        # Occurrence 0 is unambiguous, and no unquoted heredoc was skipped.
        assert len(bodies) == 1
        assert len(HEREDOC_MARKER_RE.findall(run)) == 1
        assert script == bodies[0]
        marker_line = next(
            line for line in run.split("\n") if HEREDOC_MARKER_RE.search(line)
        )
        tag = re.search(r"<<'(\w+)'", marker_line).group(1)
        assert f"{marker_line}\n{script}{tag}\n" in run

    assert script.strip()
    compile(script, f"<validate-rulespec@{sha[:8]} {job}/{name}>", "exec")


def test_us_caller_resolver_script_compiles_and_reads_workflow_toolchain(
    tmp_path: Path,
) -> None:
    caller = yaml.safe_load((FIXTURES / "us-caller.yml").read_text())
    resolver = caller["jobs"]["workflow-toolchain"]["steps"][1]
    digest = hashlib.sha256(resolver["run"].encode("utf-8")).hexdigest()

    assert digest in RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS
    bodies = _heredoc_bodies(resolver["run"])
    assert len(bodies) == 1
    compile(bodies[0], "<rulespec-us workflow-toolchain resolver>", "exec")

    (tmp_path / ".axiom").mkdir()
    shutil.copyfile(
        FIXTURES / "us-workflow-toolchain.toml",
        tmp_path / ".axiom" / "workflow-toolchain.toml",
    )
    declared = tomllib.loads((FIXTURES / "us-workflow-toolchain.toml").read_text())[
        "workflow_toolchain"
    ]

    code, stdout, stderr = _run_embedded_python(bodies[0], cwd=tmp_path)

    assert (code, stderr) == (0, "")
    assert stdout.splitlines() == [
        f"{key}={declared[key]}"
        for key in (
            "axiom_encode_ref",
            "axiom_rules_engine_ref",
            "axiom_corpus_ref",
            "rulespec_us_ref",
        )
    ]


# ---------------------------------------------------------------------------
# _run_embedded_python
# ---------------------------------------------------------------------------

_PROBE_SCRIPT = """
import json
import os
import sys

print(json.dumps({
    "argv": sys.argv[1:],
    "stdin": sys.stdin.buffer.read().decode("utf-8"),
    "cwd": os.getcwd(),
    "executable": sys.executable,
    "github_runner": sorted(
        name for name in os.environ if name.startswith(("GITHUB_", "RUNNER_"))
    ),
    "env": {
        name: os.environ.get(name)
        for name in (
            "GITHUB_SHA", "GITHUB_OUTPUT", "GITHUB_TOKEN", "GITHUB_EVENT_NAME",
            "RUNNER_TEMP", "CI_PARITY_KEEP", "NOT_GITHUB_PREFIXED", "EXTRA_STEP_ENV",
        )
    },
}))
print("diagnostic on stderr", file=sys.stderr)
raise SystemExit(7)
"""


def test_run_embedded_python_withholds_ambient_actions_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    monkeypatch.setenv("GITHUB_OUTPUT", "/tmp/ambient-output")
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-token")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("RUNNER_TEMP", "/tmp/ambient-runner")
    monkeypatch.setenv("CI_PARITY_KEEP", "kept")
    monkeypatch.setenv("NOT_GITHUB_PREFIXED", "kept-too")
    before = dict(os.environ)

    code, stdout, stderr = _run_embedded_python(
        _PROBE_SCRIPT,
        cwd=tmp_path,
        argv=("first arg", "--flag", ""),
        environment={"GITHUB_EVENT_NAME": "pull_request", "EXTRA_STEP_ENV": "1"},
        stdin=b"one\0two\n",
    )

    assert code == 7
    assert stderr == "diagnostic on stderr\n"
    probe = json.loads(stdout)
    assert probe["argv"] == ["first arg", "--flag", ""]
    assert probe["stdin"] == "one\0two\n"
    assert os.path.realpath(probe["cwd"]) == os.path.realpath(tmp_path)
    assert probe["executable"] == sys.executable
    # Only the explicitly passed step context crosses into the child.
    assert probe["github_runner"] == ["GITHUB_EVENT_NAME"]
    assert probe["env"] == {
        "GITHUB_SHA": None,
        "GITHUB_OUTPUT": None,
        "GITHUB_TOKEN": None,
        "GITHUB_EVENT_NAME": "pull_request",
        "RUNNER_TEMP": None,
        "CI_PARITY_KEEP": "kept",
        "NOT_GITHUB_PREFIXED": "kept-too",
        "EXTRA_STEP_ENV": "1",
    }
    assert dict(os.environ) == before


def test_run_embedded_python_defaults_and_undecodable_output(tmp_path: Path) -> None:
    code, stdout, stderr = _run_embedded_python(
        "import sys\n"
        "data = sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(b'argv=%d stdin=%d \\xff\\n' % (len(sys.argv) - 1, len(data)))\n"
        "sys.stderr.buffer.write(b'\\xfe')\n",
        cwd=tmp_path,
    )

    assert code == 0
    assert stdout == "argv=0 stdin=0 \ufffd\n"
    assert stderr == "\ufffd"


# ---------------------------------------------------------------------------
# acquire_workflow_release_object
# ---------------------------------------------------------------------------


def _content_sha256(content: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _release_payload(
    release: str = RELEASE, **overrides: Any
) -> tuple[dict[str, Any], str]:
    content = {
        "git": {"commit": "a" * 40},
        "artifacts": {"b": 2, "a": 1},
        "label": "caf\u00e9",
    }
    digest = _content_sha256(content)
    payload: dict[str, Any] = {
        "release": release,
        "content_sha256": digest,
        "content": content,
        "signature": {"key_id": "corpus-release", "value": "c2ln"},
    }
    payload.update(overrides)
    return payload, digest


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _caller(tmp_path: Path, sha: str = PIN_0EFFA6A5, **overrides: Any) -> CallerConfig:
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    inputs: dict[str, Any] = {
        name: (
            declaration.default
            if declaration.default is not None
            else _INPUT_TYPE_DEFAULTS[declaration.type]
        )
        for name, declaration in pin.inputs.items()
    }
    inputs.update(
        {
            "axiom-encode-ref": REFS["encode"],
            "axiom-rules-engine-ref": REFS["engine"],
            "axiom-corpus-ref": REFS["corpus"],
            "rulespec-us-ref": REFS["rulespec_us"],
        }
    )
    inputs.update(overrides)
    return CallerConfig(
        path=tmp_path / "rulespec-us" / ".github" / "workflows" / "checks.yml",
        workflow_sha=sha,
        refs=dict(REFS),
        validate_roots=str(inputs["validate-roots"]),
        run_generated_guard=bool(inputs["run-generated-guard"]),
        guard_programs_root=bool(inputs["guard-programs-root"]),
        release_base_url=str(inputs["corpus-release-base-url"]),
        inputs=inputs,
    )


def _registry_caller(tmp_path: Path, sha: str, url: str) -> CallerConfig:
    return _caller(
        tmp_path,
        sha,
        **{
            "corpus-release-registry-url": url,
            "corpus-release-registry-anon-key": "anon-key",
            # The registry is used exclusively when set, even beside a mirror.
            "corpus-release-base-url": "https://mirror.example/unused",
        },
    )


def _release_toolchain(tmp_path: Path, digest: str) -> RuleSpecToolchain:
    return RuleSpecToolchain(tmp_path / "rulespec-us", RELEASE, digest, "0" * 64)


class _Fetcher:
    def __init__(self, *responses: bytes | BaseException) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: Any) -> bytes:
        self.calls.append((url, dict(headers)))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _never_fetch(url: str, headers: Any) -> bytes:
    raise AssertionError(f"unexpected fetch of {url}")


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "axiom-corpus"
    corpus.mkdir()
    return corpus


def _tmp_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.tmp"))


def _bash_strip_one_slash(value: str) -> str:
    return subprocess.run(
        ["bash", "-c", 'v="$1"; printf "%s" "${v%/}"', "_", value],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


@pytest.mark.parametrize("sha", EMBEDDED_PINS, ids=lambda s: s[:8])
def test_registry_acquisition_requests_the_workflow_url_and_rewrites_canonically(
    tmp_path: Path, sha: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    fetcher = _Fetcher(json.dumps([{"release_object": payload}]).encode())

    destination, note = acquire_workflow_release_object(
        PinnedWorkflow(sha),
        _release_toolchain(tmp_path, digest),
        corpus,
        _registry_caller(tmp_path, sha, "https://registry.example"),
        offline=False,
        fetcher=fetcher,
    )

    assert fetcher.calls == [
        (
            "https://registry.example/rest/v1/release_objects?select=release_object"
            f"&release_name=eq.{RELEASE}&content_sha256=eq.{digest}&limit=2",
            REGISTRY_HEADERS,
        )
    ]
    assert destination == corpus / "releases" / RELEASE / f"{digest}.json"
    assert destination.read_bytes() == _canonical(payload)
    assert note == (
        f"corpus release {RELEASE}@{digest[:12]} from registry https://registry.example"
    )
    assert [path.name for path in destination.parent.iterdir()] == [f"{digest}.json"]


def _large_release_payload() -> tuple[dict[str, Any], str]:
    # Several megabytes, like a real union object, so that concurrent writes
    # to one shared temporary file would interleave.
    content = {
        "git": {"commit": "a" * 40},
        "artifacts": {f"file-{index:05d}": "x" * 400 for index in range(8000)},
    }
    digest = _content_sha256(content)
    return {
        "release": RELEASE,
        "content_sha256": digest,
        "content": content,
        "signature": {"key_id": "corpus-release", "value": "c2ln"},
    }, digest


def test_release_verification_never_uses_a_shared_temporary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    shared = corpus / "releases" / RELEASE / f"{digest}.json.tmp"
    seen: list[str] = []
    real = ci_parity._run_embedded_python

    def spy(script: str, **kwargs: Any) -> tuple[int, str, str]:
        seen.append(kwargs["argv"][0])
        return real(script, **kwargs)

    monkeypatch.setattr(ci_parity, "_run_embedded_python", spy)
    for _ in range(2):
        destination, _ = acquire_workflow_release_object(
            PinnedWorkflow(PIN_6F11BE26),
            _release_toolchain(tmp_path, digest),
            corpus,
            _registry_caller(tmp_path, PIN_6F11BE26, "https://registry.example"),
            offline=False,
            fetcher=_Fetcher(json.dumps([{"release_object": payload}]).encode()),
        )

    assert len(seen) == 2 and len(set(seen)) == 2
    assert str(shared) not in seen
    assert all(Path(path).name == shared.name for path in seen)
    assert destination.read_bytes() == _canonical(payload)
    assert [path.name for path in destination.parent.iterdir()] == [f"{digest}.json"]


@pytest.mark.parametrize("sha", EMBEDDED_PINS, ids=lambda s: s[:8])
def test_concurrent_runs_sharing_a_corpus_checkout_verify_the_release_object(
    tmp_path: Path, sha: str
) -> None:
    # Attempt 6 of the rulespec-us end-to-end run started five ci runs on one
    # corpus checkout; their verifiers truncated and appended to one
    # "<object>.json.tmp", leaving invalid JSON ("Extra data") or no file.
    corpus = _corpus(tmp_path)
    payload, digest = _large_release_payload()
    body = json.dumps([{"release_object": payload}]).encode()

    def acquire(_: int) -> bytes:
        destination, _ = acquire_workflow_release_object(
            PinnedWorkflow(sha),
            _release_toolchain(tmp_path, digest),
            corpus,
            _registry_caller(tmp_path, sha, "https://registry.example"),
            offline=False,
            fetcher=_Fetcher(body),
        )
        return destination.read_bytes()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(acquire, range(8)))

    assert results == [_canonical(payload)] * 8
    releases = corpus / "releases" / RELEASE
    assert [path.name for path in releases.iterdir()] == [f"{digest}.json"]


def test_concurrent_legacy_acquisitions_share_a_corpus_checkout(
    tmp_path: Path,
) -> None:
    payload, digest = _large_release_payload()
    raw = json.dumps(payload).encode()
    toolchain = RuleSpecToolchain(tmp_path / "rulespec-us", RELEASE, digest, "0" * 64)

    def acquire(_: int) -> bytes:
        return acquire_release_object(
            toolchain,
            tmp_path,
            "https://objects.example",
            offline=False,
            fetcher=lambda url: raw,
        ).read_bytes()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(acquire, range(8)))

    assert results == [raw] * 8
    releases = tmp_path / "releases" / RELEASE
    assert [path.name for path in releases.iterdir()] == [f"{digest}.json"]


@needs_bash
@pytest.mark.parametrize(
    "registry",
    [
        "https://registry.example/",
        "https://registry.example//",
        "https://registry.example/base",
    ],
)
def test_registry_url_strips_exactly_one_trailing_slash_like_bash(
    tmp_path: Path, registry: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    fetcher = _Fetcher(json.dumps([{"release_object": payload}]).encode())
    stripped = _bash_strip_one_slash(registry)

    _, note = acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _registry_caller(tmp_path, PIN_0EFFA6A5, registry),
        offline=False,
        fetcher=fetcher,
    )

    assert fetcher.calls[0][0] == (
        f"{stripped}/rest/v1/release_objects?select=release_object"
        f"&release_name=eq.{RELEASE}&content_sha256=eq.{digest}&limit=2"
    )
    assert note.endswith(f"from registry {stripped}")


def _registry_rows(rows: Any) -> bytes:
    return json.dumps(rows).encode()


@pytest.mark.parametrize(
    ("make_response", "message"),
    [
        (lambda payload: _registry_rows([]), "registry did not return exactly one row"),
        (
            lambda payload: _registry_rows(
                [{"release_object": payload}, {"release_object": payload}]
            ),
            "registry did not return exactly one row",
        ),
        (
            lambda payload: _registry_rows({"release_object": payload}),
            "registry did not return exactly one row",
        ),
        (
            lambda payload: _registry_rows(["not-a-row"]),
            "registry did not return exactly one row",
        ),
        (
            lambda payload: _registry_rows([{"other": payload}]),
            "missing release object",
        ),
        (
            lambda payload: _registry_rows(
                [{"release_object": {**payload, "release": "us-rulespec-other"}}]
            ),
            "release name mismatch",
        ),
        (
            lambda payload: _registry_rows(
                [
                    {
                        "release_object": {
                            **payload,
                            "content": {**payload["content"], "label": "tampered"},
                        }
                    }
                ]
            ),
            "content sha256 mismatch",
        ),
        (
            lambda payload: _registry_rows(
                [{"release_object": {**payload, "content_sha256": "0" * 64}}]
            ),
            "content sha256 mismatch",
        ),
        (
            lambda payload: _registry_rows(
                [{"release_object": {**payload, "content": ["not", "a", "dict"]}}]
            ),
            "missing content object",
        ),
        (lambda payload: b"<html>502</html>", "invalid JSON"),
    ],
    ids=[
        "zero-rows",
        "two-rows",
        "object-not-list",
        "row-not-object",
        "row-without-release-object",
        "wrong-release-name",
        "wrong-content",
        "wrong-declared-content-sha",
        "content-not-object",
        "invalid-json",
    ],
)
def test_registry_verifier_failures_fail_closed_without_leftovers(
    tmp_path: Path, make_response: Callable[[dict[str, Any]], bytes], message: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    destination = corpus / "releases" / RELEASE / f"{digest}.json"

    with pytest.raises(ValueError) as error:
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_0EFFA6A5),
            _release_toolchain(tmp_path, digest),
            corpus,
            _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example"),
            offline=False,
            fetcher=_Fetcher(make_response(payload)),
        )

    assert str(error.value).startswith("Corpus release acquisition error: ")
    assert message in str(error.value)
    assert not destination.exists()
    assert _tmp_files(corpus) == []


@pytest.mark.parametrize("sha", EMBEDDED_PINS, ids=lambda s: s[:8])
def test_mirror_acquisition_when_registry_url_is_empty(
    tmp_path: Path, sha: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    compact = json.dumps(payload, separators=(",", ":")).encode()
    fetcher = _Fetcher(compact)
    caller = _caller(
        tmp_path, sha, **{"corpus-release-base-url": "https://objects.example/base"}
    )

    destination, note = acquire_workflow_release_object(
        PinnedWorkflow(sha),
        _release_toolchain(tmp_path, digest),
        corpus,
        caller,
        offline=False,
        fetcher=fetcher,
    )

    url = f"https://objects.example/base/releases/{RELEASE}/{digest}.json"
    assert fetcher.calls == [(url, {})]
    assert destination.read_bytes() == _canonical(payload)
    assert note == f"corpus release {RELEASE}@{digest[:12]} from mirror {url}"
    assert _tmp_files(corpus) == []


def test_mirror_acquisition_uses_the_workflow_default_base_url(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    fetcher = _Fetcher(json.dumps(payload).encode())

    acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _caller(tmp_path),
        offline=False,
        fetcher=fetcher,
    )

    assert fetcher.calls[0][0] == (
        f"{DEFAULT_RELEASE_BASE_URL}/releases/{RELEASE}/{digest}.json"
    )


@needs_bash
@pytest.mark.parametrize(
    "base", ["https://objects.example/base/", "https://objects.example/base//"]
)
def test_mirror_base_url_strips_exactly_one_trailing_slash_like_bash(
    tmp_path: Path, base: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    fetcher = _Fetcher(json.dumps(payload).encode())

    acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _caller(tmp_path, **{"corpus-release-base-url": base}),
        offline=False,
        fetcher=fetcher,
    )

    assert fetcher.calls[0][0] == (
        f"{_bash_strip_one_slash(base)}/releases/{RELEASE}/{digest}.json"
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            lambda payload: json.dumps({**payload, "release": "other"}).encode(),
            "release name mismatch",
        ),
        (
            lambda payload: json.dumps(
                {**payload, "content": {**payload["content"], "extra": 1}}
            ).encode(),
            "content sha256 mismatch",
        ),
        # A registry-shaped response is not a release object on the mirror.
        (
            lambda payload: json.dumps([{"release_object": payload}]).encode(),
            "missing release object",
        ),
    ],
    ids=["wrong-release-name", "wrong-content", "registry-shape"],
)
def test_mirror_verifier_failures_fail_closed_without_leftovers(
    tmp_path: Path, response: Callable[[dict[str, Any]], bytes], message: str
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()

    with pytest.raises(ValueError, match=message):
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_6F11BE26),
            _release_toolchain(tmp_path, digest),
            corpus,
            _caller(tmp_path, PIN_6F11BE26),
            offline=False,
            fetcher=_Fetcher(response(payload)),
        )

    assert not (corpus / "releases" / RELEASE / f"{digest}.json").exists()
    assert _tmp_files(corpus) == []


def test_offline_cached_release_object_is_reverified_and_rewritten(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    destination = corpus / "releases" / RELEASE / f"{digest}.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(json.dumps(payload, separators=(",", ":")).encode())

    returned, note = acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example"),
        offline=True,
        fetcher=_never_fetch,
    )

    assert returned == destination
    assert destination.read_bytes() == _canonical(payload)
    assert note.startswith(
        f"corpus release {RELEASE}@{digest[:12]} from cached {destination}"
    )
    assert "--offline" in note
    assert _tmp_files(corpus) == []


def test_cached_release_object_is_fetched_afresh_online_as_ci_does(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    destination = corpus / "releases" / RELEASE / f"{digest}.json"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(_canonical(payload))
    fetcher = _Fetcher(_registry_rows([{"release_object": payload}]))

    returned, note = acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example"),
        offline=False,
        fetcher=fetcher,
    )

    # CI downloads the object on every run, so a cache never hides a release
    # that is missing from the registry.
    assert len(fetcher.calls) == 1
    assert returned == destination
    assert destination.read_bytes() == _canonical(payload)
    assert "from registry https://registry.example" in note
    assert _tmp_files(corpus) == []


def _write_tampered_cache(corpus: Path, payload: dict[str, Any], digest: str) -> Path:
    destination = corpus / "releases" / RELEASE / f"{digest}.json"
    destination.parent.mkdir(parents=True)
    tampered = {**payload, "content": {**payload["content"], "tampered": True}}
    destination.write_bytes(_canonical(tampered))
    return destination


def test_stale_cached_release_object_with_failing_fetch_fails_closed(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    destination = _write_tampered_cache(corpus, payload, digest)

    with pytest.raises(ValueError, match="release name mismatch"):
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_0EFFA6A5),
            _release_toolchain(tmp_path, digest),
            corpus,
            _caller(tmp_path),
            offline=False,
            fetcher=_Fetcher(json.dumps({**payload, "release": "other"}).encode()),
        )

    # A failed download never replaces or deletes what is cached.
    assert destination.exists()
    assert _tmp_files(corpus) == []


def test_stale_cached_release_object_offline_fails_closed(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    destination = _write_tampered_cache(corpus, payload, digest)

    with pytest.raises(ValueError) as error:
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_0EFFA6A5),
            _release_toolchain(tmp_path, digest),
            corpus,
            _caller(tmp_path),
            offline=True,
            fetcher=_never_fetch,
        )

    assert "Cached corpus release object failed verification" in str(error.value)
    assert "content sha256 mismatch" in str(error.value)
    assert destination.exists()
    assert _tmp_files(corpus) == []


def test_offline_without_cached_release_object_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    _, digest = _release_payload()
    destination = corpus / "releases" / RELEASE / f"{digest}.json"

    with pytest.raises(ValueError) as error:
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_0EFFA6A5),
            _release_toolchain(tmp_path, digest),
            corpus,
            _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example"),
            offline=True,
            fetcher=_never_fetch,
        )

    assert str(error.value) == (
        f"--offline requires pinned corpus release object: {destination}"
    )
    assert not (corpus / "releases").exists()


@pytest.mark.parametrize(
    "failure",
    [
        OSError("connection reset"),
        urllib.error.URLError("name resolution failed"),
        urllib.error.HTTPError(
            "http://evil.example/x",
            302,
            "refusing a non-HTTPS redirect",
            Message(),
            None,
        ),
        ValueError("Release object download exceeds 64 bytes"),
    ],
    ids=["oserror", "urlerror", "http-redirect-refused", "oversized"],
)
@pytest.mark.parametrize("registry", [True, False], ids=["registry", "mirror"])
def test_fetch_errors_become_acquisition_errors(
    tmp_path: Path, failure: BaseException, registry: bool
) -> None:
    corpus = _corpus(tmp_path)
    _, digest = _release_payload()
    caller = (
        _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example")
        if registry
        else _caller(tmp_path)
    )
    fetcher = _Fetcher(failure)

    with pytest.raises(ValueError) as error:
        acquire_workflow_release_object(
            PinnedWorkflow(PIN_0EFFA6A5),
            _release_toolchain(tmp_path, digest),
            corpus,
            caller,
            offline=False,
            fetcher=fetcher,
        )

    url = fetcher.calls[0][0]
    assert str(error.value) == f"Corpus release acquisition error: {url}: {failure}"
    assert error.value.__cause__ is failure
    assert not (corpus / "releases").exists()


def test_default_fetcher_is_the_bounded_https_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    payload, digest = _release_payload()
    calls = []

    def fake_fetch(url: str, headers: Any) -> bytes:
        calls.append((url, dict(headers)))
        return json.dumps([{"release_object": payload}]).encode()

    monkeypatch.setattr(ci_parity, "_fetch_https", fake_fetch)

    acquire_workflow_release_object(
        PinnedWorkflow(PIN_0EFFA6A5),
        _release_toolchain(tmp_path, digest),
        corpus,
        _registry_caller(tmp_path, PIN_0EFFA6A5, "https://registry.example"),
        offline=False,
    )

    assert [headers for _, headers in calls] == [REGISTRY_HEADERS]


# ---------------------------------------------------------------------------
# _fetch_https and redirects
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads: list[int] = []

    def read(self, amount: int = -1) -> bytes:
        self.reads.append(amount)
        return self.data if amount < 0 else self.data[:amount]

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class _FakeOpener:
    def __init__(self, handlers: tuple[Any, ...], data: bytes) -> None:
        self.handlers = handlers
        self.response = _FakeResponse(data)
        self.requests: list[tuple[urllib.request.Request, Any]] = []

    def open(self, request: urllib.request.Request, timeout: Any = None) -> Any:
        self.requests.append((request, timeout))
        return self.response


def _install_fake_opener(
    monkeypatch: pytest.MonkeyPatch, data: bytes
) -> list[_FakeOpener]:
    openers: list[_FakeOpener] = []

    def build_opener(*handlers: Any) -> _FakeOpener:
        opener = _FakeOpener(handlers, data)
        openers.append(opener)
        return opener

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    return openers


@pytest.mark.parametrize(
    "url",
    [
        "http://objects.example/releases/x.json",
        "HTTPS://objects.example/releases/x.json",
        "file:///etc/passwd",
        "ftp://objects.example/x.json",
        " https://objects.example/x.json",
    ],
)
def test_fetch_https_refuses_non_https_urls(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    openers = _install_fake_opener(monkeypatch, b"{}")

    with pytest.raises(ValueError, match="Refusing to fetch a non-HTTPS URL"):
        _fetch_https(url, {})

    assert openers == []


def test_fetch_https_identifies_itself_and_uses_https_only_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openers = _install_fake_opener(monkeypatch, b'{"ok": true}')
    url = "https://registry.example/rest/v1/release_objects?limit=2"

    data = _fetch_https(url, REGISTRY_HEADERS)

    assert data == b'{"ok": true}'
    (opener,) = openers
    assert opener.handlers == (_HttpsOnlyRedirects,)
    ((request, timeout),) = opener.requests
    assert timeout == 60
    assert request.full_url == url
    assert request.get_method() == "GET"
    assert request.get_header("User-agent") == f"axiom-encode/{__version__}"
    assert request.get_header("Apikey") == "anon-key"
    assert request.get_header("Authorization") == "Bearer anon-key"
    assert request.get_header("Accept-profile") == "corpus"
    assert opener.response.reads == [MAX_RELEASE_DOWNLOAD_BYTES + 1]


def test_fetch_https_bounds_the_download(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MAX_RELEASE_DOWNLOAD_BYTES == 4 * MAX_RELEASE_OBJECT_BYTES
    monkeypatch.setattr(ci_parity, "MAX_RELEASE_DOWNLOAD_BYTES", 8)
    openers = _install_fake_opener(monkeypatch, b"123456789")

    with pytest.raises(ValueError, match="Release object download exceeds 8 bytes"):
        _fetch_https("https://objects.example/x.json", {})
    assert openers[0].response.reads == [9]

    _install_fake_opener(monkeypatch, b"12345678")
    assert _fetch_https("https://objects.example/x.json", {}) == b"12345678"


def test_https_only_redirects_refuse_an_http_target() -> None:
    handler = _HttpsOnlyRedirects()
    request = urllib.request.Request(
        "https://objects.example/x.json",
        headers={"User-Agent": f"axiom-encode/{__version__}"},
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        handler.redirect_request(
            request, None, 302, "Found", Message(), "http://evil.example/x.json"
        )
    assert error.value.code == 302
    assert error.value.msg == "refusing a non-HTTPS redirect"
    assert error.value.filename == "http://evil.example/x.json"

    followed = handler.redirect_request(
        request, None, 302, "Found", Message(), "https://cdn.example/x.json"
    )
    assert followed.full_url == "https://cdn.example/x.json"
    assert followed.get_header("User-agent") == f"axiom-encode/{__version__}"


def test_https_only_redirects_refuse_a_location_header_to_http() -> None:
    handler = _HttpsOnlyRedirects()
    request = urllib.request.Request("https://objects.example/x.json")
    headers = Message()
    headers["Location"] = "http://evil.example/x.json"

    with pytest.raises(urllib.error.HTTPError, match="non-HTTPS redirect"):
        handler.http_error_302(request, None, 302, "Found", headers)


def test_https_only_redirects_replace_the_default_redirect_handler() -> None:
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)

    assert [
        type(handler)
        for handler in opener.handlers
        if isinstance(handler, urllib.request.HTTPRedirectHandler)
    ] == [_HttpsOnlyRedirects]


# ---------------------------------------------------------------------------
# _run_cli
# ---------------------------------------------------------------------------


def _install_fake_main(
    monkeypatch: pytest.MonkeyPatch,
    record: dict[str, Any],
    then: Callable[[], None] | None = None,
) -> None:
    from axiom_encode import cli

    def main() -> None:
        record["environ"] = dict(os.environ)
        record["cwd"] = os.getcwd()
        record["argv"] = list(sys.argv)
        print("stdout line")
        print("stderr line", file=sys.stderr)
        os.environ["CI_PARITY_LEAKED"] = "leak"
        sys.argv.append("--mutated")
        if then is not None:
            then()

    monkeypatch.setattr(cli, "main", main)


@pytest.fixture
def ambient_axiom_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_TERMINAL_PROMPT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CI_PARITY_LEAKED", raising=False)
    values = {
        "AXIOM_SECRET_TOKEN": "withheld",
        "AXIOM_ENCODE_WAIVER_AUDIT_WORKERS": "9",
        "AXIOM_RULESPEC_REPO_ROOTS": "/ambient/roots",
        "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY": "never-forwarded",
        "CI_PARITY_OTHER": "kept",
        **{name: f"forwarded-{name}" for name in SUPERVISOR_FORWARDED_ENVIRONMENT},
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_supervised_run_cli_mirrors_the_supervisor_child_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_axiom_environment: dict[str, str],
) -> None:
    record: dict[str, Any] = {}
    _install_fake_main(monkeypatch, record)
    before_environ = dict(os.environ)
    before_cwd = os.getcwd()
    before_argv = list(sys.argv)

    code, output = _run_cli(
        ["validation-waivers", "audit", "--root", "x"], cwd=tmp_path, supervised=True
    )

    assert (code, output) == (0, "stdout line\nstderr line\n")
    seen = record["environ"]
    assert {name for name in seen if name.startswith("AXIOM_")} == set(
        SUPERVISOR_FORWARDED_ENVIRONMENT
    )
    for name in SUPERVISOR_FORWARDED_ENVIRONMENT:
        assert seen[name] == f"forwarded-{name}"
    assert seen["GIT_CONFIG_GLOBAL"] == os.devnull
    assert seen["GIT_CONFIG_NOSYSTEM"] == "1"
    assert seen["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["CI_PARITY_OTHER"] == "kept"
    assert os.path.realpath(record["cwd"]) == os.path.realpath(tmp_path)
    assert record["argv"] == [
        "axiom-encode",
        "validation-waivers",
        "audit",
        "--root",
        "x",
    ]
    assert dict(os.environ) == before_environ
    assert os.getcwd() == before_cwd
    assert sys.argv == before_argv


def test_unsupervised_run_cli_keeps_ambient_axiom_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_axiom_environment: dict[str, str],
) -> None:
    record: dict[str, Any] = {}
    _install_fake_main(monkeypatch, record)
    before_environ = dict(os.environ)

    code, _ = _run_cli(
        ["test", "--root", "us"],
        environment={"AXIOM_RULESPEC_REPO_ROOTS": "/repo:/rulespec-us"},
    )

    assert code == 0
    seen = record["environ"]
    assert seen["AXIOM_SECRET_TOKEN"] == "withheld"
    assert seen["AXIOM_ENCODE_WAIVER_AUDIT_WORKERS"] == "9"
    assert seen["AXIOM_RULESPEC_REPO_ROOTS"] == "/repo:/rulespec-us"
    assert "GIT_CONFIG_GLOBAL" not in seen
    assert "GIT_TERMINAL_PROMPT" not in seen
    # Without cwd the working directory is left as it is.
    assert record["cwd"] == os.getcwd()
    assert dict(os.environ) == before_environ


def test_environment_overrides_apply_after_supervised_withholding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_axiom_environment: dict[str, str],
) -> None:
    record: dict[str, Any] = {}
    _install_fake_main(monkeypatch, record)

    _run_cli(
        ["validation-waivers", "audit"],
        environment={
            WAIVER_AUDIT_WORKERS_ENV: "2",
            "AXIOM_RULESPEC_REPO_ROOTS": "/repo:/rulespec-us",
            "GIT_TERMINAL_PROMPT": "1",
        },
        cwd=tmp_path,
        supervised=True,
    )

    seen = record["environ"]
    assert seen[WAIVER_AUDIT_WORKERS_ENV] == "2"
    assert seen["AXIOM_RULESPEC_REPO_ROOTS"] == "/repo:/rulespec-us"
    assert seen["GIT_TERMINAL_PROMPT"] == "1"
    assert "AXIOM_SECRET_TOKEN" not in seen


@pytest.mark.parametrize("scenario", ["ambient-git-dir", "env-injected-config"])
def test_supervised_run_cli_git_sees_no_ambient_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    from axiom_encode import cli

    target = _init_repo(tmp_path / "target")
    (target / "f").write_text("target\n")
    target_head = _commit_all(target, "target")
    other = _init_repo(tmp_path / "other")
    (other / "f").write_text("other\n")
    _commit_all(other, "other")
    if scenario == "ambient-git-dir":
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        probe = ["git", "-C", str(target), "rev-parse", "HEAD"]
        expected = (0, target_head)
    else:
        monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
        monkeypatch.setenv("GIT_CONFIG_KEY_0", "axiom.probe")
        monkeypatch.setenv("GIT_CONFIG_VALUE_0", "leaked")
        probe = ["git", "-C", str(target), "config", "--get", "axiom.probe"]
        expected = (1, "")
    record: dict[str, Any] = {}

    def main() -> None:
        result = subprocess.run(probe, capture_output=True, text=True, check=False)
        record["result"] = (result.returncode, result.stdout.strip())

    monkeypatch.setattr(cli, "main", main)

    _run_cli(["guard-generated"], cwd=target, supervised=True)

    assert record["result"] == expected


def test_run_cli_reports_exceptions_and_restores_process_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_axiom_environment: dict[str, str],
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def explode() -> None:
        os.chdir(elsewhere)
        raise RuntimeError("boom")

    _install_fake_main(monkeypatch, {}, explode)
    before_environ = dict(os.environ)
    before_cwd = os.getcwd()
    before_argv = list(sys.argv)

    code, output = _run_cli(["validate", "x"], cwd=tmp_path, supervised=True)

    assert code == 1
    assert output == "stdout line\nstderr line\nRuntimeError: boom\n"
    assert dict(os.environ) == before_environ
    assert os.getcwd() == before_cwd
    assert sys.argv == before_argv


def test_run_cli_restores_process_state_when_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ambient_axiom_environment: dict[str, str],
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    _install_fake_main(monkeypatch, {}, interrupt)
    before_environ = dict(os.environ)
    before_cwd = os.getcwd()
    before_argv = list(sys.argv)

    with pytest.raises(KeyboardInterrupt):
        _run_cli(["validate", "x"], cwd=tmp_path, supervised=True)

    assert dict(os.environ) == before_environ
    assert os.getcwd() == before_cwd
    assert sys.argv == before_argv


@pytest.mark.parametrize(
    ("exit_code", "expected_code", "expected_tail"),
    [
        ("program-scope-sync failed: nope", 1, "program-scope-sync failed: nope\n"),
        (None, 0, "stderr line\n"),
        (0, 0, "stderr line\n"),
        (2, 2, "stderr line\n"),
        (3, 3, "stderr line\n"),
    ],
    ids=["message", "none", "zero", "two", "three"],
)
def test_run_cli_maps_system_exit_like_a_process(
    monkeypatch: pytest.MonkeyPatch,
    exit_code: object,
    expected_code: int,
    expected_tail: str,
) -> None:
    def leave() -> None:
        raise SystemExit(exit_code)

    _install_fake_main(monkeypatch, {}, leave)

    code, output = _run_cli(["validate"])

    assert code == expected_code
    assert output.startswith("stdout line\nstderr line\n")
    assert output.endswith(expected_tail)


def test_run_cli_bare_system_exit_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def leave() -> None:
        raise SystemExit

    _install_fake_main(monkeypatch, {}, leave)

    assert _run_cli(["validate"]) == (0, "stdout line\nstderr line\n")


# ---------------------------------------------------------------------------
# _run_cli_batch, _run_cli_isolated and _isolated_cli_main
# ---------------------------------------------------------------------------


def _invocations(tmp_path: Path, count: int) -> list[CliInvocation]:
    return [
        CliInvocation(
            ("validate", f"file-{index}.yaml"),
            cwd=tmp_path / f"cwd-{index}",
            supervised=index % 2 == 0,
            environment=None
            if index == 0
            else {"AXIOM_RULESPEC_REPO_ROOTS": str(index)},
        )
        for index in range(count)
    ]


def _refuse_isolation(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("unexpected isolated worker")


@pytest.mark.parametrize(("jobs", "count"), [(1, 3), (0, 3), (-2, 3), (4, 1), (4, 0)])
def test_run_cli_batch_runs_in_process_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jobs: int, count: int
) -> None:
    calls = []

    def fake_run_cli(arguments, *, environment, cwd, supervised):
        calls.append((tuple(arguments), environment, cwd, supervised))
        return len(calls) - 1, f"out {arguments[1]}\n"

    monkeypatch.setattr(ci_parity, "_run_cli", fake_run_cli)
    monkeypatch.setattr(ci_parity, "_run_cli_isolated", _refuse_isolation)
    invocations = _invocations(tmp_path, count)

    results = _run_cli_batch(invocations, jobs=jobs, keyring=(CURRENT_KEY,))

    assert calls == [
        (
            invocation.arguments,
            dict(invocation.environment or {}),
            invocation.cwd,
            invocation.supervised,
        )
        for invocation in invocations
    ]
    assert results == [(index, f"out file-{index}.yaml\n") for index in range(count)]


def test_run_cli_batch_isolates_concurrent_workers_and_keeps_input_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = 3
    invocations = _invocations(tmp_path, 5)
    barrier = threading.Barrier(jobs, timeout=10)
    lock = threading.Lock()
    active = {"now": 0, "max": 0}
    calls = []

    def fake_isolated(invocation, keyring, worker_cap, signing_roots=None):
        index = invocations.index(invocation)
        with lock:
            calls.append((index, tuple(keyring), worker_cap))
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        if index < jobs:
            # The first `jobs` invocations must be in flight together.
            barrier.wait()
        # Later inputs finish first; results must still follow input order.
        time.sleep(0.01 * (len(invocations) - index))
        with lock:
            active["now"] -= 1
        return index, f"isolated {index}\n"

    monkeypatch.setattr(ci_parity, "_run_cli_isolated", fake_isolated)
    monkeypatch.setattr(ci_parity, "_run_cli", _refuse_isolation)
    monkeypatch.setattr(ci_parity.os, "cpu_count", lambda: 8)

    results = _run_cli_batch(invocations, jobs=jobs, keyring=(CURRENT_KEY, RETIRED_KEY))

    assert results == [(index, f"isolated {index}\n") for index in range(5)]
    assert sorted(calls) == [
        (index, (CURRENT_KEY, RETIRED_KEY), 8 // jobs) for index in range(5)
    ]
    assert active["max"] == jobs


@pytest.mark.parametrize(
    ("cpu_count", "jobs", "worker_cap"),
    [(8, 3, 2), (18, 2, 9), (2, 4, 1), (None, 2, 1), (4, 4, 1)],
)
def test_run_cli_batch_worker_cap_is_cpu_count_over_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cpu_count: int | None,
    jobs: int,
    worker_cap: int,
) -> None:
    caps = []

    def fake_isolated(invocation, keyring, cap, signing_roots=None):
        caps.append(cap)
        return 0, ""

    monkeypatch.setattr(ci_parity, "_run_cli_isolated", fake_isolated)
    monkeypatch.setattr(ci_parity.os, "cpu_count", lambda: cpu_count)

    _run_cli_batch(_invocations(tmp_path, 2), jobs=jobs, keyring=(CURRENT_KEY,))

    assert caps == [worker_cap, worker_cap]


def test_run_cli_isolated_passes_keyring_on_stdin_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("PYTHONPATH", "/existing/path")

    def fake_run(argv, *, input, stdout, stderr, cwd, env, check):
        captured.update(argv=argv, input=input, env=env, check=check, cwd=cwd)
        return subprocess.CompletedProcess(argv, 4, b"worker out\n", b"worker err\n")

    monkeypatch.setattr(ci_parity.subprocess, "run", fake_run)
    invocation = CliInvocation(
        ("validation-waivers", "audit", "--partition-key", "us"),
        cwd=tmp_path,
        supervised=True,
        environment={
            "AXIOM_RULESPEC_REPO_ROOTS": "/a:/b",
            WAIVER_AUDIT_WORKERS_ENV: "32",
        },
    )

    code, output = _run_cli_isolated(invocation, (CURRENT_KEY, RETIRED_KEY), 3)

    assert (code, output) == (4, "worker out\nworker err\n")
    # -I: neither cwd nor PYTHONPATH can put another encoder on sys.path.
    assert captured["argv"] == [sys.executable, "-I", "-c", _ISOLATED_CLI_BOOTSTRAP]
    assert captured["check"] is False
    assert captured["cwd"] == tmp_path
    module_file = str(Path(ci_parity.__file__).resolve())
    assert json.loads(captured["input"]) == {
        "package_root": str(Path(module_file).parents[1]),
        "module_file": module_file,
        "keyring": [CURRENT_KEY, RETIRED_KEY],
        "signing_roots": {},
        "arguments": ["validation-waivers", "audit", "--partition-key", "us"],
        "cwd": str(tmp_path),
        "supervised": True,
        # The batch worker cap wins over any per-invocation value.
        "environment": {
            "AXIOM_RULESPEC_REPO_ROOTS": "/a:/b",
            WAIVER_AUDIT_WORKERS_ENV: "3",
        },
    }
    assert not any(name.startswith("GIT_") for name in captured["env"])
    assert not any(
        key in value
        for key in (CURRENT_KEY, RETIRED_KEY)
        for value in captured["env"].values()
    )


@pytest.mark.parametrize("supervised", [True, False])
def test_isolated_cli_main_runs_under_the_local_keyring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    supervised: bool,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_cli(arguments, *, environment, cwd, supervised):
        seen.update(
            keys=toolchain_module._LOCAL_CORPUS_RELEASE_PUBLIC_KEYS.get(),
            arguments=arguments,
            environment=environment,
            cwd=cwd,
            supervised=supervised,
        )
        return 6, "worker output\n"

    monkeypatch.setattr(ci_parity, "_run_cli", fake_run_cli)

    code = _isolated_cli_main(
        {
            "keyring": [CURRENT_KEY, RETIRED_KEY],
            "arguments": ["validate", "a.yaml"],
            "cwd": str(tmp_path),
            "supervised": supervised,
            "environment": {WAIVER_AUDIT_WORKERS_ENV: "1"},
        }
    )

    assert code == 6
    assert capsys.readouterr().out == "worker output\n"
    assert seen == {
        "keys": (CURRENT_KEY, RETIRED_KEY),
        "arguments": ["validate", "a.yaml"],
        "environment": {WAIVER_AUDIT_WORKERS_ENV: "1"},
        "cwd": tmp_path,
        "supervised": supervised,
    }
    assert toolchain_module._LOCAL_CORPUS_RELEASE_PUBLIC_KEYS.get() is None


def test_isolated_cli_main_rejects_a_malformed_keyring_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ci_parity, "_run_cli", _refuse_isolation)

    with pytest.raises(RuleSpecToolchainError, match="--corpus-release-retired"):
        _isolated_cli_main(
            {
                "keyring": [CURRENT_KEY, "not base64"],
                "arguments": ["validate"],
                "cwd": str(tmp_path),
                "supervised": True,
                "environment": {},
            }
        )


def test_real_isolated_workers_bootstrap_the_cli(tmp_path: Path) -> None:
    results = _run_cli_batch(
        [
            CliInvocation(("ci", "--help"), cwd=tmp_path, supervised=True),
            CliInvocation(("ci", "--json"), cwd=tmp_path),
        ],
        jobs=2,
        keyring=(CURRENT_KEY, RETIRED_KEY),
    )

    (help_code, help_output), (usage_code, usage_output) = results
    assert help_code == 0, help_output
    assert help_output.startswith("usage: axiom-encode ci ")
    assert "--corpus-release-retired-public-key" in help_output
    assert usage_code == 2, usage_output
    assert (
        "the following arguments are required: --repo, --corpus-release-public-key"
        in usage_output
    )


def test_real_isolated_worker_fails_closed_on_a_malformed_keyring(
    tmp_path: Path,
) -> None:
    code, output = _run_cli_isolated(
        CliInvocation(("ci", "--help"), cwd=tmp_path), ("not-base64",), 1
    )

    assert code == 1
    assert "RuleSpecToolchainError" in output
    assert "--corpus-release-public-key must be canonical base64" in output
    assert "usage:" not in output


# ---------------------------------------------------------------------------
# local_corpus_release_verification
# ---------------------------------------------------------------------------


def _active_keyring() -> tuple[str, ...] | None:
    return toolchain_module._LOCAL_CORPUS_RELEASE_PUBLIC_KEYS.get()


def test_local_keyring_is_current_then_retired_and_resets() -> None:
    assert _active_keyring() is None
    with local_corpus_release_verification(CURRENT_KEY):
        assert _active_keyring() == (CURRENT_KEY,)
    with local_corpus_release_verification(
        CURRENT_KEY, retired_public_keys=[RETIRED_KEY, OTHER_RETIRED_KEY]
    ):
        assert _active_keyring() == (CURRENT_KEY, RETIRED_KEY, OTHER_RETIRED_KEY)
        with local_corpus_release_verification(RETIRED_KEY):
            assert _active_keyring() == (RETIRED_KEY,)
        assert _active_keyring() == (CURRENT_KEY, RETIRED_KEY, OTHER_RETIRED_KEY)
    assert _active_keyring() is None


def test_local_keyring_resets_when_the_body_raises() -> None:
    with pytest.raises(RuntimeError, match="gate crashed"):
        with local_corpus_release_verification(
            CURRENT_KEY, retired_public_keys=(RETIRED_KEY,)
        ):
            raise RuntimeError("gate crashed")

    assert _active_keyring() is None


@pytest.mark.parametrize(
    "retired",
    [[CURRENT_KEY], [RETIRED_KEY, RETIRED_KEY], [RETIRED_KEY, CURRENT_KEY]],
    ids=["retired-equals-current", "repeated-retired", "current-among-retired"],
)
def test_local_keyring_rejects_duplicate_keys(retired: list[str]) -> None:
    entered = False
    with pytest.raises(RuleSpecToolchainError, match="must be distinct"):
        with local_corpus_release_verification(
            CURRENT_KEY, retired_public_keys=retired
        ):
            entered = True

    assert entered is False
    assert _active_keyring() is None


_SHORT_KEY = base64.b64encode(bytes(31)).decode("ascii")
_LONG_KEY = base64.b64encode(bytes(33)).decode("ascii")
# Same 32 bytes, but non-zero padding bits: decodes, is not canonical.
_NONCANONICAL_KEY = CURRENT_KEY[:-2] + "9="


@pytest.mark.parametrize(
    ("key", "problem"),
    [
        ("not base64!", "must be canonical base64"),
        (CURRENT_KEY.rstrip("="), "must be canonical base64"),
        (_SHORT_KEY, "must encode exactly 32 bytes"),
        (_LONG_KEY, "must encode exactly 32 bytes"),
        (_NONCANONICAL_KEY, "must encode exactly 32 bytes"),
        ("", "must encode exactly 32 bytes"),
    ],
    ids=["garbage", "unpadded", "31-bytes", "33-bytes", "noncanonical", "empty"],
)
def test_malformed_keys_name_their_flag(key: str, problem: str) -> None:
    assert base64.b64decode(_NONCANONICAL_KEY) == base64.b64decode(CURRENT_KEY)

    with pytest.raises(RuleSpecToolchainError) as retired_error:
        with local_corpus_release_verification(
            CURRENT_KEY, retired_public_keys=[RETIRED_KEY, key]
        ):
            pass
    assert str(retired_error.value) == f"--corpus-release-retired-public-key {problem}"

    with pytest.raises(RuleSpecToolchainError) as current_error:
        with local_corpus_release_verification(key):
            pass
    assert str(current_error.value) == f"--corpus-release-public-key {problem}"
    assert _active_keyring() is None


# ---------------------------------------------------------------------------
# verify_dependency_checkout / verify_ambient_encoder
# ---------------------------------------------------------------------------


def _corpus_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    repo = _init_repo(tmp_path / "axiom-corpus")
    (repo / "README.md").write_text("corpus\n")
    (repo / "sources").mkdir()
    (repo / "sources" / "a.txt").write_text("a\n")
    pin = _commit_all(repo, "corpus")
    _git(repo, "update-ref", "refs/remotes/origin/main", pin)
    _, digest = _release_payload()
    return repo, pin, f"releases/{RELEASE}/{digest}.json"


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_release_cache_is_the_only_untracked_exemption(tmp_path: Path) -> None:
    repo, pin, cache = _corpus_checkout(tmp_path)
    caller = tmp_path / "caller.yml"
    # The whole releases/ tree is untracked, so this relies on
    # --untracked-files=all listing the file rather than the directory.
    _write(repo / cache, "{}\n")

    assert (
        verify_dependency_checkout(
            "corpus",
            repo,
            pin,
            caller,
            allow_ref_mismatch=False,
            ignored_untracked=frozenset({cache}),
        )
        is None
    )
    with pytest.raises(ValueError, match="dirty worktree"):
        verify_dependency_checkout(
            "corpus", repo, pin, caller, allow_ref_mismatch=False
        )


def _dirty_untracked_sibling(repo: Path, cache: str) -> None:
    _write(repo / Path(cache).with_name("0" * 64 + ".json"))


def _dirty_leftover_tmp(repo: Path, cache: str) -> None:
    _write(repo / f"{cache}.tmp")


def _dirty_untracked_file(repo: Path, cache: str) -> None:
    _write(repo / "notes.txt")


def _dirty_modified(repo: Path, cache: str) -> None:
    (repo / "README.md").write_text("changed\n")


def _dirty_staged(repo: Path, cache: str) -> None:
    _write(repo / "sources" / "b.txt")
    _git(repo, "add", "sources/b.txt")


def _dirty_deleted(repo: Path, cache: str) -> None:
    (repo / "sources" / "a.txt").unlink()


def _dirty_untracked_with_config_hiding_it(repo: Path, cache: str) -> None:
    _git(repo, "config", "status.showUntrackedFiles", "no")
    _write(repo / "sources" / "stray.txt")


@pytest.mark.parametrize(
    "make_dirty",
    [
        _dirty_untracked_sibling,
        _dirty_leftover_tmp,
        _dirty_untracked_file,
        _dirty_modified,
        _dirty_staged,
        _dirty_deleted,
        _dirty_untracked_with_config_hiding_it,
    ],
    ids=[
        "untracked-sibling-release",
        "leftover-tmp",
        "untracked-file",
        "modified",
        "staged",
        "deleted",
        "untracked-hidden-by-config",
    ],
)
def test_other_dirt_is_still_a_mismatch_beside_the_release_cache(
    tmp_path: Path, make_dirty: Callable[[Path, str], None]
) -> None:
    repo, pin, cache = _corpus_checkout(tmp_path)
    caller = tmp_path / "caller.yml"
    _write(repo / cache, "{}\n")
    make_dirty(repo, cache)

    with pytest.raises(ValueError) as error:
        verify_dependency_checkout(
            "corpus",
            repo,
            pin,
            caller,
            allow_ref_mismatch=False,
            ignored_untracked=frozenset({cache}),
        )
    assert "REF MISMATCH" in str(error.value)
    assert f"{pin} (dirty worktree)" in str(error.value)

    mismatch = verify_dependency_checkout(
        "corpus",
        repo,
        pin,
        caller,
        allow_ref_mismatch=True,
        ignored_untracked=frozenset({cache}),
    )
    assert mismatch == DependencyMismatch("corpus", f"{pin} (dirty worktree)", pin)


def test_a_tracked_release_path_is_not_exempt_when_modified(tmp_path: Path) -> None:
    repo, _, cache = _corpus_checkout(tmp_path)
    _write(repo / cache, "{}\n")
    pin = _commit_all(repo, "track the release object")
    _git(repo, "update-ref", "refs/remotes/origin/main", pin)
    (repo / cache).write_text('{"changed": true}\n')

    mismatch = verify_dependency_checkout(
        "corpus",
        repo,
        pin,
        tmp_path / "caller.yml",
        allow_ref_mismatch=True,
        ignored_untracked=frozenset({cache}),
    )

    assert mismatch == DependencyMismatch("corpus", f"{pin} (dirty worktree)", pin)


def _canned_git(
    responses: dict[tuple[str, ...], tuple[int, str]], calls: list[Any]
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def fake(repo: Path, *arguments: str, check: bool = True):
        calls.append((repo, arguments, check))
        if arguments[:2] == ("ls-files", "-v") and arguments not in responses:
            # No assume-unchanged or skip-worktree entries.
            return subprocess.CompletedProcess(
                ["git", "-C", str(repo), *arguments], 0, "", ""
            )
        if arguments == AMBIENT_TOPLEVEL and arguments not in responses:
            # The ambient package is a real src/ checkout; answer as git would.
            return subprocess.CompletedProcess(
                ["git", "-C", str(repo), *arguments], 0, f"{repo}\n", ""
            )
        code, stdout = responses[arguments]
        return subprocess.CompletedProcess(
            ["git", "-C", str(repo), *arguments], code, stdout, ""
        )

    return fake


AMBIENT_STATUS = (
    "status",
    "--porcelain",
    "--untracked-files=all",
    "--",
    "src",
    "pyproject.toml",
    "uv.lock",
)
AMBIENT_TOPLEVEL = ("rev-parse", "--show-toplevel")


def test_ambient_encoder_clean_checkout_at_the_pin_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = "1" * 40
    calls: list[Any] = []
    monkeypatch.setattr(
        ci_parity,
        "_git",
        _canned_git(
            {("rev-parse", "HEAD"): (0, pin + "\n"), AMBIENT_STATUS: (0, "")}, calls
        ),
    )

    assert (
        verify_ambient_encoder(
            pin, "0.2.2006", tmp_path / "caller.yml", allow_encoder_mismatch=False
        )
        is None
    )
    checkout = Path(ci_parity.__file__).resolve().parents[2]
    assert calls == [
        (checkout, AMBIENT_TOPLEVEL, False),
        (checkout, ("rev-parse", "HEAD"), False),
        (checkout, AMBIENT_STATUS, False),
        (checkout, ("ls-files", "-v", "--", "src", "pyproject.toml", "uv.lock"), False),
    ]


@pytest.mark.parametrize(
    "status",
    [" M src/axiom_encode/cli.py\n", "?? src/yaml.py\n", "M  uv.lock\n"],
    ids=["modified-src", "untracked-src", "staged-lock"],
)
def test_ambient_encoder_dirty_sources_at_the_pinned_head_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    pin = "1" * 40
    caller = tmp_path / "caller.yml"
    monkeypatch.setattr(
        ci_parity,
        "_git",
        _canned_git(
            {("rev-parse", "HEAD"): (0, pin + "\n"), AMBIENT_STATUS: (0, status)}, []
        ),
    )

    with pytest.raises(ValueError) as error:
        verify_ambient_encoder(pin, "0.2.2006", caller, allow_encoder_mismatch=False)
    message = str(error.value)
    assert message.startswith("ENCODER MISMATCH: ambient-encoder: HEAD ")
    assert f"{pin} (dirty worktree)" in message
    assert str(caller) in message
    assert "--allow-encoder-mismatch" in message

    assert verify_ambient_encoder(
        pin, "0.2.2006", caller, allow_encoder_mismatch=True
    ) == DependencyMismatch("ambient-encoder", f"{pin} (dirty worktree)", pin)


def test_ambient_encoder_other_head_and_dirty_names_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin, head = "1" * 40, "2" * 40
    monkeypatch.setattr(
        ci_parity,
        "_git",
        _canned_git(
            {
                ("rev-parse", "HEAD"): (0, head + "\n"),
                AMBIENT_STATUS: (0, " M src/x.py\n"),
            },
            [],
        ),
    )

    assert verify_ambient_encoder(
        pin, "0.2.2006", tmp_path / "caller.yml", allow_encoder_mismatch=True
    ) == DependencyMismatch("ambient-encoder", f"{head} (dirty worktree)", pin)


def test_ambient_encoder_non_sha_head_is_unresolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = "1" * 40
    calls: list[Any] = []
    monkeypatch.setattr(
        ci_parity, "_git", _canned_git({("rev-parse", "HEAD"): (0, "HEAD\n")}, calls)
    )

    mismatch = verify_ambient_encoder(
        pin, "0.2.2006", tmp_path / "caller.yml", allow_encoder_mismatch=True
    )

    assert mismatch == DependencyMismatch(
        "ambient-encoder", f"unresolvable ({__version__})", pin
    )
    assert [arguments for _, arguments, _ in calls] == [
        AMBIENT_TOPLEVEL,
        ("rev-parse", "HEAD"),
    ]


def _encoder_checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = _init_repo(tmp_path / "axiom-encode")
    _write(repo / "src" / "axiom_encode" / "ci_parity.py", "# stand-in\n")
    _write(repo / "pyproject.toml", '[project]\nversion = "0.2.2006"\n')
    _write(repo / "uv.lock", "version = 1\n")
    _write(repo / "docs" / "readme.md", "docs\n")
    _write(repo / ".gitignore", "__pycache__/\n.venv/\n")
    return repo, _commit_all(repo, "encoder")


def _as_ambient(monkeypatch: pytest.MonkeyPatch, module_path: Path) -> None:
    monkeypatch.setattr(ci_parity, "__file__", str(module_path))


def test_ambient_encoder_from_a_real_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _encoder_checkout(tmp_path)
    _as_ambient(monkeypatch, repo / "src" / "axiom_encode" / "ci_parity.py")
    caller = tmp_path / "caller.yml"

    def check() -> DependencyMismatch | None:
        return verify_ambient_encoder(
            pin, "0.2.2006", caller, allow_encoder_mismatch=True
        )

    assert check() is None
    # Files outside the importable sources and lock do not change the encoder.
    (repo / "docs" / "readme.md").write_text("edited\n")
    assert check() is None
    _write(repo / "src" / "axiom_encode" / "__pycache__" / "x.pyc", "ignored\n")
    assert check() is None

    _write(repo / "src" / "yaml.py", "raise SystemExit('shadowed')\n")
    assert check() == DependencyMismatch(
        "ambient-encoder", f"{pin} (dirty worktree)", pin
    )
    (repo / "src" / "yaml.py").unlink()
    (repo / "uv.lock").write_text("version = 2\n")
    assert check() == DependencyMismatch(
        "ambient-encoder", f"{pin} (dirty worktree)", pin
    )


def test_ambient_encoder_untracked_source_is_dirty_regardless_of_git_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _encoder_checkout(tmp_path)
    _as_ambient(monkeypatch, repo / "src" / "axiom_encode" / "ci_parity.py")
    _git(repo, "config", "status.showUntrackedFiles", "no")
    _write(repo / "src" / "yaml.py", "raise SystemExit('shadowed')\n")

    assert verify_ambient_encoder(
        pin, "0.2.2006", tmp_path / "caller.yml", allow_encoder_mismatch=True
    ) == DependencyMismatch("ambient-encoder", f"{pin} (dirty worktree)", pin)


def test_ambient_encoder_installed_wheel_inside_a_checkout_is_not_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, pin = _encoder_checkout(tmp_path)
    installed = repo / ".venv" / "lib" / "python3.14" / "site-packages" / "axiom_encode"
    _write(installed / "ci_parity.py", "# a different, installed encoder\n")
    # Even edited pinned sources go unnoticed from there.
    (repo / "src" / "axiom_encode" / "ci_parity.py").write_text("# edited\n")
    _as_ambient(monkeypatch, installed / "ci_parity.py")

    assert (
        verify_ambient_encoder(
            pin, "0.2.2006", tmp_path / "caller.yml", allow_encoder_mismatch=True
        )
        is not None
    )


# ---------------------------------------------------------------------------
# register_ci_parser
# ---------------------------------------------------------------------------


def _ci_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="axiom-encode")
    subparsers = parser.add_subparsers(dest="command")
    register_ci_parser(subparsers)
    return parser


def test_ci_parser_defaults() -> None:
    args = _ci_parser().parse_args(
        ["ci", "--repo", "rulespec-us", "--corpus-release-public-key", CURRENT_KEY]
    )

    assert args.repo == Path("rulespec-us")
    assert args.corpus_release_public_key == CURRENT_KEY
    assert args.corpus_release_retired_public_key == []
    assert args.corpus_release_registry_url is None
    assert args.corpus_release_registry_anon_key is None
    assert args.pull_request is None
    assert args.jobs == 1
    assert args.base_ref == "origin/main"
    assert args.roots is None
    assert (args.offline, args.json) == (False, False)
    assert (args.allow_ref_mismatch, args.allow_encoder_mismatch) == (False, False)


def test_ci_parser_repeatable_and_typed_options() -> None:
    parser = _ci_parser()
    base = ["ci", "--repo", "r", "--corpus-release-public-key", CURRENT_KEY]

    args = parser.parse_args(
        [
            *base,
            "--corpus-release-retired-public-key",
            RETIRED_KEY,
            "--corpus-release-retired-public-key",
            OTHER_RETIRED_KEY,
            "--corpus-release-registry-url",
            "https://registry.example",
            "--corpus-release-registry-anon-key",
            "anon-key",
            "--pull-request",
            "911",
            "--jobs",
            "4",
        ]
    )

    assert args.corpus_release_retired_public_key == [RETIRED_KEY, OTHER_RETIRED_KEY]
    assert args.corpus_release_registry_url == "https://registry.example"
    assert args.corpus_release_registry_anon_key == "anon-key"
    assert args.pull_request == 911
    assert args.jobs == 4
    # The append default is not shared between parses.
    assert parser.parse_args(base).corpus_release_retired_public_key == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--pull-request", "abc"],
        ["--jobs", "two"],
        ["--jobs", "1.5"],
        ["--corpus-release-retired-public-key"],
    ],
)
def test_ci_parser_rejects_malformed_options(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        _ci_parser().parse_args(
            ["ci", "--repo", "r", "--corpus-release-public-key", CURRENT_KEY, *extra]
        )
    assert error.value.code == 2


def test_ci_parser_requires_an_explicit_public_key() -> None:
    with pytest.raises(SystemExit) as error:
        _ci_parser().parse_args(["ci", "--repo", "r"])
    assert error.value.code == 2


# ---------------------------------------------------------------------------
# run_ci reporting
# ---------------------------------------------------------------------------

RESOLUTIONS = (
    "axiom-encode-ref: ${{ needs.workflow-toolchain.outputs.axiom_encode_ref }} -> "
    + REFS["encode"],
    "corpus-release-registry-url: ${{ vars.NEXT_PUBLIC_SUPABASE_URL }} -> "
    "https://registry.example (--corpus-release-registry-url)",
)
RELEASE_DIGEST = "c" * 64
RELEASE_NOTE = (
    f"corpus release {RELEASE}@{RELEASE_DIGEST[:12]} from registry "
    "https://registry.example"
)
PLAN = ShardPlan(
    matrix=("us", "us-ca"),
    first="us",
    roots="us us-ca",
    allowed_extra="sources programs",
    matrix_json='["us", "us-ca"]',
    scope="shard-on-demand: us us-ca",
)


class _CiHarness:
    """Monkeypatched run_ci seams that record how they were called."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        workflow_sha: str = PIN_0EFFA6A5,
        gate_statuses: tuple[str, ...] = ("PASS", "PASS"),
        mismatches: dict[str, DependencyMismatch] | None = None,
        encoder_mismatch: DependencyMismatch | None = None,
        base_is_ancestor: bool = True,
        acquire_error: Exception | None = None,
    ) -> None:
        self.tmp_path = tmp_path
        self.repo = tmp_path / "rulespec-us"
        self.paths = {
            "encode": tmp_path / "axiom-encode",
            "engine": tmp_path / "axiom-rules-engine",
            "corpus": tmp_path / "axiom-corpus",
            "rulespec_us": tmp_path / "canonical-rulespec-us",
        }
        for path in (self.repo, *self.paths.values()):
            path.mkdir()
        self.workflow_sha = workflow_sha
        embedded = SUPPORTED_WORKFLOW_PINS[
            workflow_sha
        ].gate_parameters.embedded_scripts
        overrides: dict[str, Any] = (
            {
                "validate-roots": "auto",
                "validation-workers": 4,
                "corpus-release-registry-url": "https://registry.example",
                "corpus-release-registry-anon-key": "anon-key",
            }
            if embedded
            else {"validate-roots": "statutes regulations"}
        )
        caller = _caller(tmp_path, workflow_sha, **overrides)
        self.caller = CallerConfig(
            path=caller.path,
            workflow_sha=caller.workflow_sha,
            refs=caller.refs,
            validate_roots=caller.validate_roots,
            run_generated_guard=caller.run_generated_guard,
            guard_programs_root=caller.guard_programs_root,
            release_base_url=caller.release_base_url,
            inputs=caller.inputs,
            resolutions=RESOLUTIONS,
        )
        self.toolchain = RuleSpecToolchain(
            self.repo.resolve(), RELEASE, RELEASE_DIGEST, "d" * 64
        )
        self.release_path = (
            self.paths["corpus"].resolve()
            / "releases"
            / RELEASE
            / f"{RELEASE_DIGEST}.json"
        )
        self.results = [
            GateResult(
                f"gate_{index}",
                f"Gate {index}",
                status,
                ["cmd", str(index)],
                [] if status == "PASS" else [f"gate {index} failed"],
                f"output {index}\n",
                None,
            )
            for index, status in enumerate(gate_statuses)
        ]
        self.mismatches = mismatches or {}
        self.encoder_mismatch = encoder_mismatch
        self.base_is_ancestor = base_is_ancestor
        self.acquire_error = acquire_error
        self.calls: dict[str, list[Any]] = {}
        self.run: WorkflowRun | None = None
        self.simulation: PullRequestSimulation | None = None
        for name in (
            "find_caller_workflow",
            "simulate_pull_request",
            "resolve_workflow_toolchain",
            "verify_toolchain_base_binding",
            "load_rulespec_toolchain",
            "verify_rulespec_validation_waiver_set",
            "verify_dependency_checkout",
            "encoder_version_at_pin",
            "verify_ambient_encoder",
            "committed_checkout",
            "resolve_commit",
            "uncommitted_changes_note",
            "verify_python_version",
            "acquire_workflow_release_object",
            "acquire_release_object",
            "authenticate_release_provenance",
            "compute_shard_plan",
            "load_rulespec_local_corpus_release",
            "execute_workflow_gates",
            "execute_gates",
        ):
            monkeypatch.setattr(ci_parity, name, getattr(self, name))

    def _record(self, name: str, *arguments: Any) -> None:
        self.calls.setdefault(name, []).append(arguments)

    def find_caller_workflow(self, repo, overrides=None):
        self._record("find_caller_workflow", repo, overrides)
        return self.caller

    def simulate_pull_request(self, repo, base_ref, number=None):
        self._record("simulate_pull_request", repo, base_ref, number)
        self.simulation = PullRequestSimulation(
            base_sha="a" * 40,
            head_sha="b" * 40,
            base_branch="main",
            repository="TheAxiomFoundation/rulespec-us",
            number=number,
            base_is_ancestor=self.base_is_ancestor,
        )
        return self.simulation

    def resolve_workflow_toolchain(self, workflow, repo, simulation):
        self._record("resolve_workflow_toolchain", workflow.sha, repo, simulation)

    def verify_toolchain_base_binding(self, repo, base_ref):
        self._record("verify_toolchain_base_binding", repo, base_ref)

    def load_rulespec_toolchain(self, repo):
        self._record("load_rulespec_toolchain", repo)
        return self.toolchain

    def verify_rulespec_validation_waiver_set(self, repo):
        self._record("verify_rulespec_validation_waiver_set", repo)
        return "d" * 64

    def verify_dependency_checkout(
        self,
        name,
        path,
        pin,
        caller,
        *,
        allow_ref_mismatch,
        ignored_untracked=frozenset(),
    ):
        self._record(
            "verify_dependency_checkout",
            name,
            path,
            pin,
            caller,
            allow_ref_mismatch,
            ignored_untracked,
        )
        return self.mismatches.get(name)

    def encoder_version_at_pin(self, path, pin):
        self._record("encoder_version_at_pin", path, pin)
        return "0.2.2006"

    def verify_ambient_encoder(self, pin, version, caller, *, allow_encoder_mismatch):
        self._record("verify_ambient_encoder", pin, version, allow_encoder_mismatch)
        return self.encoder_mismatch

    @contextlib.contextmanager
    def committed_checkout(self, source, head=None, base=None):
        # The harness validates the source path itself.
        self._record("committed_checkout", source, head)
        yield source

    def resolve_commit(self, repo, ref):
        # The harness repository is not a git checkout; refs pass through.
        self._record("resolve_commit", repo, ref)
        return ref

    def uncommitted_changes_note(self, source):
        self._record("uncommitted_changes_note", source)
        return None

    def verify_python_version(self, requested, caller, *, allow_encoder_mismatch):
        self._record("verify_python_version", requested, allow_encoder_mismatch)
        return None

    def acquire_workflow_release_object(
        self, workflow, toolchain, corpus, caller, *, offline
    ):
        self._record(
            "acquire_workflow_release_object",
            workflow.sha,
            toolchain,
            corpus,
            caller,
            offline,
        )
        if self.acquire_error is not None:
            raise self.acquire_error
        return self.release_path, RELEASE_NOTE

    def acquire_release_object(self, toolchain, corpus, base_url, *, offline):
        self._record("acquire_release_object", toolchain, corpus, base_url, offline)
        return self.release_path

    def authenticate_release_provenance(self, release_path, corpus, pin, caller):
        self._record("authenticate_release_provenance", release_path, corpus, pin)
        return "e" * 40

    def compute_shard_plan(
        self, workflow, repo, simulation, validate_roots_input, guard_programs_root
    ):
        self._record(
            "compute_shard_plan",
            workflow.sha,
            repo,
            simulation,
            validate_roots_input,
            guard_programs_root,
        )
        return PLAN

    def load_rulespec_local_corpus_release(self, repo, corpus):
        self._record(
            "load_rulespec_local_corpus_release", repo, corpus, _active_keyring()
        )

    def execute_workflow_gates(self, run):
        self._record("execute_workflow_gates", _active_keyring())
        self.run = run
        run.notes.append("note recorded by a gate")
        return self.results

    def execute_gates(self, args, caller, paths, roots):
        self._record("execute_gates", _active_keyring(), paths, roots)
        return self.results

    def args(self, *extra: str, json_output: bool = True) -> Namespace:
        argv = [
            "ci",
            "--repo",
            str(self.repo),
            "--corpus-release-public-key",
            CURRENT_KEY,
            "--apply-public-key",
            APPLY_KEY,
            "--encode-path",
            str(self.paths["encode"]),
            "--engine-path",
            str(self.paths["engine"]),
            "--corpus-path",
            str(self.paths["corpus"]),
            "--rulespec-us-path",
            str(self.paths["rulespec_us"]),
            *extra,
        ]
        if json_output:
            argv.append("--json")
        return _ci_parser().parse_args(argv)


def _report(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def test_run_ci_embedded_pass_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)

    assert run_ci(harness.args()) == 0

    report = _report(capsys)
    assert report == {
        "passed": True,
        "verdict": "PASS",
        "caller": str(harness.caller.path),
        # The harness repository is not a git checkout, so no commit resolves.
        "validated_commit": "",
        "workflow_sha": PIN_0EFFA6A5,
        "dependency_mismatches": [],
        "resolutions": list(RESOLUTIONS),
        "notes": [RELEASE_NOTE, "note recorded by a gate"],
        "roots": ["us", "us-ca"],
        "gates": [
            {
                "gate": result.gate,
                "name": result.name,
                "status": "PASS",
                "command": result.command,
                "failures": [],
                "output": result.output,
                "note": None,
            }
            for result in harness.results
        ],
        "shards": {
            "matrix": ["us", "us-ca"],
            "first": "us",
            "scope": "shard-on-demand: us us-ca",
        },
    }
    repo = harness.repo.resolve()
    assert harness.calls["find_caller_workflow"] == [(repo, CallerOverrides())]
    assert harness.calls["simulate_pull_request"] == [(repo, "origin/main", None)]
    assert harness.calls["resolve_workflow_toolchain"] == [
        (PIN_0EFFA6A5, repo, harness.simulation)
    ]
    assert "verify_toolchain_base_binding" not in harness.calls
    assert "acquire_release_object" not in harness.calls
    assert "execute_gates" not in harness.calls
    assert harness.calls["compute_shard_plan"] == [
        (PIN_0EFFA6A5, repo, harness.simulation, "auto", False)
    ]
    run = harness.run
    assert run is not None
    assert run.caller is harness.caller
    assert run.workflow.sha == PIN_0EFFA6A5
    assert run.repo == repo
    assert run.paths == {name: path.resolve() for name, path in harness.paths.items()}
    assert run.simulation is harness.simulation
    assert run.plan is PLAN
    assert run.validate_roots_input == "auto"
    assert run.keyring == (CURRENT_KEY,)
    assert run.jobs == 1
    # The release is authenticated under the keyring before any gate runs,
    # and the gates run inside the same keyring.
    assert harness.calls["load_rulespec_local_corpus_release"] == [
        (repo, harness.paths["corpus"].resolve(), (CURRENT_KEY,))
    ]
    assert harness.calls["execute_workflow_gates"] == [((CURRENT_KEY,),)]
    assert _active_keyring() is None


def test_run_ci_wires_release_cache_pin_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)

    assert run_ci(harness.args("--offline", "--allow-ref-mismatch")) == 0

    caller_path = harness.caller.path
    cache = f"releases/{RELEASE}/{RELEASE_DIGEST}.json"
    assert harness.calls["verify_dependency_checkout"] == [
        (
            name,
            harness.paths[name].resolve(),
            REFS[name],
            caller_path,
            True,
            frozenset({cache}) if name == "corpus" else frozenset(),
        )
        for name in ("encode", "engine", "corpus", "rulespec_us")
    ]
    assert harness.calls["encoder_version_at_pin"] == [
        (harness.paths["encode"].resolve(), REFS["encode"])
    ]
    assert harness.calls["verify_ambient_encoder"] == [
        (REFS["encode"], "0.2.2006", False)
    ]
    assert harness.calls["acquire_workflow_release_object"] == [
        (
            PIN_0EFFA6A5,
            harness.toolchain,
            harness.paths["corpus"].resolve(),
            harness.caller,
            True,
        )
    ]
    assert harness.calls["authenticate_release_provenance"] == [
        (harness.release_path, harness.paths["corpus"].resolve(), REFS["corpus"])
    ]


def test_run_ci_forwards_registry_overrides_pull_request_roots_jobs_and_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=PIN_6F11BE26)

    code = run_ci(
        harness.args(
            "--corpus-release-registry-url",
            "https://other-registry.example",
            "--corpus-release-registry-anon-key",
            "local-anon",
            "--pull-request",
            "911",
            "--roots",
            "us",
            "--jobs",
            "3",
            "--base-ref",
            "upstream/main",
            "--corpus-release-retired-public-key",
            RETIRED_KEY,
            "--corpus-release-retired-public-key",
            OTHER_RETIRED_KEY,
        )
    )

    assert code == 0
    repo = harness.repo.resolve()
    keyring = (CURRENT_KEY, RETIRED_KEY, OTHER_RETIRED_KEY)
    assert harness.calls["find_caller_workflow"] == [
        (
            repo,
            CallerOverrides(
                registry_url="https://other-registry.example",
                registry_anon_key="local-anon",
            ),
        )
    ]
    assert harness.calls["simulate_pull_request"] == [(repo, "upstream/main", 911)]
    assert harness.calls["compute_shard_plan"][0][3] == "us"
    assert harness.run is not None
    assert harness.run.validate_roots_input == "us"
    assert harness.run.jobs == 3
    assert harness.run.keyring == keyring
    assert harness.calls["load_rulespec_local_corpus_release"][0][2] == keyring
    assert harness.calls["execute_workflow_gates"] == [(keyring,)]
    assert _report(capsys)["workflow_sha"] == PIN_6F11BE26


@pytest.mark.parametrize("base_ref", ["origin/main", "upstream/release"])
def test_run_ci_notes_when_head_does_not_contain_the_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_ref: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, base_is_ancestor=False)

    assert run_ci(harness.args("--base-ref", base_ref)) == 0

    assert _report(capsys)["notes"] == [
        RELEASE_NOTE,
        f"HEAD does not contain {base_ref}; CI validates the pull request's "
        "merge commit, so rebase for exact parity",
        "note recorded by a gate",
    ]


def test_run_ci_resolves_head_and_base_in_the_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)

    assert run_ci(harness.args("--base-ref", "@{upstream}")) == 0

    source = harness.repo.resolve()
    assert harness.calls["resolve_commit"] == [
        (source, "HEAD"),
        (source, "@{upstream}"),
    ]
    assert harness.calls["committed_checkout"] == [(source, "HEAD")]
    capsys.readouterr()


def test_run_ci_mismatched_dependencies_exit_3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = DependencyMismatch("engine", "9" * 40, REFS["engine"])
    ambient = DependencyMismatch(
        "ambient-encoder", f"{REFS['encode']} (dirty worktree)", REFS["encode"]
    )
    harness = _CiHarness(
        tmp_path,
        monkeypatch,
        mismatches={"engine": engine},
        encoder_mismatch=ambient,
    )

    code = run_ci(harness.args("--allow-ref-mismatch", "--allow-encoder-mismatch"))

    assert code == 3
    report = _report(capsys)
    assert report["verdict"] == "PASS-WITH-MISMATCHED-DEPS"
    assert report["passed"] is False
    assert report["dependency_mismatches"] == [
        {"name": "engine", "head_sha": "9" * 40, "pinned_sha": REFS["engine"]},
        {
            "name": "ambient-encoder",
            "head_sha": f"{REFS['encode']} (dirty worktree)",
            "pinned_sha": REFS["encode"],
        },
    ]
    assert harness.calls["verify_ambient_encoder"] == [
        (REFS["encode"], "0.2.2006", True)
    ]


@pytest.mark.parametrize(
    "mismatches",
    [{}, {"corpus": DependencyMismatch("corpus", "9" * 40, REFS["corpus"])}],
    ids=["clean", "with-mismatch"],
)
def test_run_ci_failed_gate_exit_1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mismatches: dict[str, DependencyMismatch],
) -> None:
    harness = _CiHarness(
        tmp_path, monkeypatch, gate_statuses=("PASS", "FAIL"), mismatches=mismatches
    )

    assert run_ci(harness.args("--allow-ref-mismatch")) == 1

    report = _report(capsys)
    assert report["verdict"] == "FAIL"
    assert report["passed"] is False
    assert [gate["status"] for gate in report["gates"]] == ["PASS", "FAIL"]
    assert report["gates"][1]["failures"] == ["gate 1 failed"]
    assert len(report["dependency_mismatches"]) == len(mismatches)
    assert "resolution_error" not in report


def test_run_ci_resolution_failure_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = DependencyMismatch("engine", "9" * 40, REFS["engine"])
    harness = _CiHarness(
        tmp_path,
        monkeypatch,
        mismatches={"engine": engine},
        acquire_error=ValueError(
            "Corpus release acquisition error: registry did not return exactly one row"
        ),
    )

    assert run_ci(harness.args("--allow-ref-mismatch")) == 1

    assert _report(capsys) == {
        "passed": False,
        "verdict": "FAIL",
        "dependency_mismatches": [
            {"name": "engine", "head_sha": "9" * 40, "pinned_sha": REFS["engine"]}
        ],
        "resolutions": list(RESOLUTIONS),
        "notes": [],
        "resolution_error": (
            "Corpus release acquisition error: registry did not return exactly one row"
        ),
        "gates": [],
    }
    assert "execute_workflow_gates" not in harness.calls
    assert "load_rulespec_local_corpus_release" not in harness.calls


def test_run_ci_resolution_failure_text_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = DependencyMismatch("engine", "9" * 40, REFS["engine"])
    harness = _CiHarness(
        tmp_path,
        monkeypatch,
        mismatches={"engine": engine},
        acquire_error=ValueError("Corpus release acquisition error: offline"),
    )

    assert run_ci(harness.args("--allow-ref-mismatch", json_output=False)) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        f"WARNING: {engine.banner_line()}",
        "axiom-encode ci resolution failed: Corpus release acquisition error: offline",
    ]


@pytest.mark.parametrize("jobs", ["-1", "-8"])
def test_run_ci_negative_jobs_is_a_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    jobs: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)

    assert run_ci(harness.args("--jobs", jobs)) == 1

    report = _report(capsys)
    assert report["resolution_error"] == "--jobs must be at least 1"
    assert report["gates"] == []
    assert "execute_workflow_gates" not in harness.calls


def test_run_ci_zero_jobs_is_a_resolution_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)

    assert run_ci(harness.args("--jobs", "0")) == 1

    assert _report(capsys)["resolution_error"] == "--jobs must be at least 1"
    assert "execute_workflow_gates" not in harness.calls


@pytest.mark.parametrize(
    ("current", "retired", "message"),
    [
        ("not-a-key", (), "--corpus-release-public-key must be canonical base64"),
        (
            CURRENT_KEY,
            ("not-a-key",),
            "--corpus-release-retired-public-key must be canonical base64",
        ),
        (
            CURRENT_KEY,
            (CURRENT_KEY,),
            "corpus release, apply and eval public keys must be distinct",
        ),
    ],
    ids=["bad-current", "bad-retired", "duplicate"],
)
def test_run_ci_bad_keyring_fails_before_any_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    current: str,
    retired: tuple[str, ...],
    message: str,
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch)
    args = harness.args(
        *(
            item
            for key in retired
            for item in ("--corpus-release-retired-public-key", key)
        )
    )
    args.corpus_release_public_key = current

    assert run_ci(args) == 1

    assert _report(capsys)["resolution_error"] == message
    assert "load_rulespec_local_corpus_release" not in harness.calls
    assert "execute_workflow_gates" not in harness.calls


def test_run_ci_legacy_pin_ignores_retired_keys_with_a_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=LEGACY_PIN)

    code = run_ci(harness.args("--corpus-release-retired-public-key", RETIRED_KEY))

    assert code == 0
    report = _report(capsys)
    assert report["notes"] == [
        f"validate-rulespec@{LEGACY_PIN[:8]} provisions only the current corpus "
        "release key; --corpus-release-retired-public-key is ignored"
    ]
    assert report["roots"] == ["statutes", "regulations"]
    assert "shards" not in report
    repo = harness.repo.resolve()
    assert harness.calls["verify_toolchain_base_binding"] == [(repo, "origin/main")]
    assert harness.calls["acquire_release_object"] == [
        (
            harness.toolchain,
            harness.paths["corpus"].resolve(),
            DEFAULT_RELEASE_BASE_URL,
            False,
        )
    ]
    for embedded_only in (
        "simulate_pull_request",
        "resolve_workflow_toolchain",
        "acquire_workflow_release_object",
        "compute_shard_plan",
        "execute_workflow_gates",
    ):
        assert embedded_only not in harness.calls
    assert harness.calls["load_rulespec_local_corpus_release"][0][2] == (CURRENT_KEY,)
    assert harness.calls["execute_gates"] == [
        (
            (CURRENT_KEY,),
            {name: path.resolve() for name, path in harness.paths.items()},
            ("statutes", "regulations"),
        )
    ]


def test_run_ci_legacy_pin_without_retired_keys_has_no_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=LEGACY_PIN)

    assert run_ci(harness.args()) == 0

    assert _report(capsys)["notes"] == []


def test_run_ci_text_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = DependencyMismatch("engine", "9" * 40, REFS["engine"])
    harness = _CiHarness(
        tmp_path,
        monkeypatch,
        mismatches={"engine": engine},
        base_is_ancestor=False,
    )

    assert run_ci(harness.args("--allow-ref-mismatch", json_output=False)) == 3

    captured = capsys.readouterr()
    assert captured.err.splitlines() == [f"WARNING: {engine.banner_line()}"]
    assert captured.out.splitlines() == [
        f"validate-rulespec caller: {harness.caller.path} @ {PIN_0EFFA6A5}",
        "  validated commit: ",
        *(f"  resolved {resolution}" for resolution in RESOLUTIONS),
        "  shards: 2 (shard-on-demand: us us-ca); first us",
        f"  note: {RELEASE_NOTE}",
        "  note: HEAD does not contain origin/main; CI validates the pull "
        "request's merge commit, so rebase for exact parity",
        "  note: note recorded by a gate",
        "PASS Gate 0",
        "PASS Gate 1",
        "ci parity result: PASS-WITH-MISMATCHED-DEPS",
        "MISMATCHED DEPENDENCIES:",
        f"  - {engine.banner_line()}",
    ]
