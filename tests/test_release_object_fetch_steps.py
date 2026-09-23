"""Run the workflow release-object fetch steps against a recording curl double.

Both protected workflows query the Supabase release registry first and fall
back to the public r2.dev mirror only when the registry has no row for the pin
or is unavailable. Ambiguous or mismatched registry answers must fail closed,
and every fetch must keep the pin check and the resolver's byte cap.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from axiom_encode.corpus_resolver import MAX_RELEASE_OBJECT_BYTES

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
RELEASE_NAME = "us-rulespec-2026-09-14-wave4-r2-union"
REGISTRY_URL = "https://swocpijqqahhuwtuahwc.supabase.co"
MIRROR_URL = "https://pub-a8952f8657fc49fda358146ac001366c.r2.dev"
ANON_KEY = "public-anon-key"

CURL_DOUBLE = textwrap.dedent(
    """\
    #!{python}
    import json, os, shutil, sys

    args = sys.argv[1:]
    call = {{"headers": [], "max_filesize": [], "flags": []}}
    positional = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in ("--output", "--header", "--max-filesize", "--proto", "--proto-redir"):
            value = args[index + 1]
            if arg == "--output":
                call["output"] = value
            elif arg == "--header":
                call["headers"].append(value)
            elif arg == "--max-filesize":
                call["max_filesize"].append(value)
            else:
                call["flags"].append(f"{{arg}}={{value}}")
            index += 2
            continue
        if arg.startswith("-"):
            call["flags"].append(arg)
        else:
            positional.append(arg)
        index += 1
    call["url"] = positional[-1]
    with open(os.environ["CURL_DOUBLE_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(call) + "\\n")
    routes = json.load(open(os.environ["CURL_DOUBLE_ROUTES"], encoding="utf-8"))
    route = routes.get(call["url"])
    if route is None:
        print("curl: (6) Could not resolve host", file=sys.stderr)
        raise SystemExit(6)
    if "status" in route:
        print(
            f"curl: (22) The requested URL returned error: {{route['status']}}",
            file=sys.stderr,
        )
        raise SystemExit(22)
    shutil.copyfile(route["body"], call["output"])
    """
)


def _step(workflow: str, job: str, name: str) -> dict:
    payload = yaml.load(
        (WORKFLOWS / workflow).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    return next(step for step in payload["jobs"][job]["steps"] if step["name"] == name)


def _canonical_sha256(content: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def _release_object(commit: str) -> tuple[dict, str]:
    content = {"git": {"commit": commit}, "scopes": [{"id": "us/statute"}]}
    digest = _canonical_sha256(content)
    return {"release": RELEASE_NAME, "content_sha256": digest, "content": content}, (
        digest
    )


def _bash_with_mapfile() -> str | None:
    for candidate in (shutil.which("bash"), "/bin/bash"):
        if candidate and (
            subprocess.run(
                [candidate, "-c", "type mapfile"], capture_output=True
            ).returncode
            == 0
        ):
            return candidate
    return None


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _pinned_signed_apply_materializer() -> str:
    payload = yaml.load(
        (WORKFLOWS / "signed-apply-reusable.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    ref = payload["env"]["AXIOM_ENCODE_REF"]
    shown = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{ref}:scripts/materialize_corpus_release.py",
        ],
        capture_output=True,
        text=True,
    )
    if shown.returncode:
        pytest.skip(f"signed-apply encoder pin {ref} is not in this checkout's history")
    return shown.stdout


@dataclass
class Harness:
    root: Path
    workdir: Path
    corpus: Path
    release_object: dict
    digest: str
    env: dict[str, str]
    routes: dict[str, dict]

    @property
    def registry_query(self) -> str:
        return (
            f"{REGISTRY_URL}/rest/v1/release_objects?select=release_object"
            f"&release_name=eq.{RELEASE_NAME}&content_sha256=eq.{self.digest}&limit=2"
        )

    @property
    def mirror_object(self) -> str:
        return f"{MIRROR_URL}/releases/{RELEASE_NAME}/{self.digest}.json"

    @property
    def materialized(self) -> Path:
        return self.corpus / "releases" / RELEASE_NAME / f"{self.digest}.json"

    def serve(self, url: str, payload: object | bytes) -> None:
        body = self.root / f"body-{len(self.routes)}"
        body.write_bytes(
            payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        )
        self.routes[url] = {"body": str(body)}

    def fail(self, url: str, status: int) -> None:
        self.routes[url] = {"status": status}

    def registry_rows(self, *rows: dict) -> None:
        self.serve(self.registry_query, [{"release_object": row} for row in rows])

    def run(self, script: str, bash: str) -> subprocess.CompletedProcess[str]:
        routes = self.root / "routes.json"
        routes.write_text(json.dumps(self.routes), encoding="utf-8")
        log = self.root / "curl.jsonl"
        log.write_text("", encoding="utf-8")
        return subprocess.run(
            [bash, "-c", script],
            cwd=self.workdir,
            env={
                **self.env,
                "CURL_DOUBLE_ROUTES": str(routes),
                "CURL_DOUBLE_LOG": str(log),
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self) -> list[dict]:
        log = self.root / "curl.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()]


def _harness(tmp_path: Path, workdir: Path, corpus: Path) -> Harness:
    bin_dir = tmp_path / "bin"
    _executable(bin_dir / "curl", CURL_DOUBLE.format(python=sys.executable))
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    corpus.mkdir(parents=True, exist_ok=True)
    _git(corpus, "init", "-q", "-b", "main")
    _git(corpus, "config", "user.email", "test@example.com")
    _git(corpus, "config", "user.name", "Test")
    (corpus / "README.md").write_text("corpus\n", encoding="utf-8")
    _git(corpus, "add", "README.md")
    _git(corpus, "commit", "-q", "-m", "corpus")
    release_object, digest = _release_object(_git(corpus, "rev-parse", "HEAD"))
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
    }
    return Harness(tmp_path, workdir, corpus, release_object, digest, env, {})


def _toolchain(path: Path, digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{RELEASE_NAME}"\n'
        f'axiom_corpus_release_content_sha256 = "{digest}"\n'
        f'validation_waiver_set_sha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )


def _targeted_step(tmp_path: Path):
    bash = _bash_with_mapfile()
    if bash is None:
        pytest.skip("targeted re-encode steps use mapfile, which needs bash 4+")
    step = _step(
        "targeted-signed-reencode.yml",
        "encode",
        "Fetch pinned signed corpus release object",
    )
    workdir = tmp_path / "workspace"
    harness = _harness(tmp_path, workdir, workdir / "axiom-corpus")
    scripts = workdir / "axiom-encode" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "scripts" / "materialize_corpus_release.py",
        scripts / "materialize_corpus_release.py",
    )
    _executable(
        workdir / "axiom-encode" / ".venv" / "bin" / "python",
        f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
    )
    _toolchain(workdir / "rulespec-us" / ".axiom" / "toolchain.toml", harness.digest)
    env = step["env"]
    assert env["RELEASE_REGISTRY_URL"] == REGISTRY_URL
    assert env["RELEASE_BASE_URL"] == MIRROR_URL
    harness.env.update(
        RELEASE_REGISTRY_URL=env["RELEASE_REGISTRY_URL"],
        RELEASE_REGISTRY_ANON_KEY=ANON_KEY,
        RELEASE_BASE_URL=env["RELEASE_BASE_URL"],
        RULESPEC_CHECKOUT="rulespec-us",
        QUEUE_ID="",
        QUEUE_MANIFEST_SHA256="",
    )
    assert "${{" not in step["run"]
    return harness, step["run"], bash


def _signed_apply_step(tmp_path: Path, encoder: str):
    step = _step(
        "signed-apply-reusable.yml",
        "encode",
        "Fetch and verify the signed release object",
    )
    workdir = tmp_path / "rulespec-de"
    harness = _harness(tmp_path, workdir, workdir / "_axiom" / "axiom-corpus")
    materializer = (
        workdir
        / "_axiom"
        / "axiom-encode"
        / "scripts"
        / "materialize_corpus_release.py"
    )
    materializer.parent.mkdir(parents=True)
    materializer.write_text(
        _pinned_signed_apply_materializer()
        if encoder == "pinned-encoder"
        else (ROOT / "scripts" / "materialize_corpus_release.py").read_text(),
        encoding="utf-8",
    )
    _toolchain(workdir / ".axiom" / "toolchain.toml", harness.digest)
    env = step["env"]
    assert env["RELEASE_REGISTRY_URL"] == (
        "${{ inputs['corpus-release-registry-url'] }}"
    )
    assert env["RELEASE_REGISTRY_ANON_KEY"] == (
        "${{ inputs['corpus-release-registry-anon-key'] }}"
    )
    assert env["RELEASE_BASE_URL"] == MIRROR_URL
    assert env["RELEASE_NAME"] == "${{ steps.pins.outputs.release }}"
    assert env["RELEASE_SHA"] == "${{ steps.pins.outputs.release_sha }}"
    assert env["PYBIN"] == "${{ steps.py.outputs.python-path }}"
    harness.env.update(
        RELEASE_REGISTRY_URL=REGISTRY_URL,
        RELEASE_REGISTRY_ANON_KEY=ANON_KEY,
        RELEASE_BASE_URL=env["RELEASE_BASE_URL"],
        RELEASE_NAME=RELEASE_NAME,
        RELEASE_SHA=harness.digest,
        PYBIN=sys.executable,
    )
    assert "${{" not in step["run"]
    return harness, step["run"], "/bin/bash"


@pytest.fixture
def targeted(tmp_path: Path):
    return _targeted_step(tmp_path)


@pytest.fixture(params=["pinned-encoder", "current-encoder"])
def signed_apply(tmp_path: Path, request: pytest.FixtureRequest):
    return _signed_apply_step(tmp_path, request.param)


@pytest.fixture(
    params=[
        "targeted-reencode",
        "signed-apply-pinned-encoder",
        "signed-apply-current-encoder",
    ]
)
def fetch_step(tmp_path: Path, request: pytest.FixtureRequest):
    if request.param == "targeted-reencode":
        return _targeted_step(tmp_path)
    return _signed_apply_step(tmp_path, request.param.removeprefix("signed-apply-"))


def _assert_capped_https(calls: list[dict]) -> None:
    for call in calls:
        assert call["max_filesize"] == [str(MAX_RELEASE_OBJECT_BYTES)], call
        assert "--proto==https" in call["flags"], call
        assert "--proto-redir==https" in call["flags"], call
        assert "--fail" in call["flags"], call


def _assert_registry_headers(call: dict) -> None:
    assert call["headers"] == [
        f"apikey: {ANON_KEY}",
        f"Authorization: Bearer {ANON_KEY}",
        "Accept-Profile: corpus",
    ]


def _assert_materialized(harness: Harness) -> None:
    assert json.loads(harness.materialized.read_text()) == harness.release_object


def test_registry_row_is_materialized_without_touching_the_mirror(fetch_step):
    harness, script, bash = fetch_step
    harness.registry_rows(harness.release_object)
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode == 0, result.stderr
    calls = harness.calls()
    assert [call["url"] for call in calls] == [harness.registry_query]
    _assert_registry_headers(calls[0])
    _assert_capped_https(calls)
    _assert_materialized(harness)
    assert "source: registry" in result.stdout


def test_registry_without_a_row_falls_back_to_the_mirror(fetch_step):
    harness, script, bash = fetch_step
    harness.serve(harness.registry_query, [])
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode == 0, result.stderr
    calls = harness.calls()
    assert [call["url"] for call in calls] == [
        harness.registry_query,
        harness.mirror_object,
    ]
    assert calls[1]["headers"] == []
    _assert_capped_https(calls)
    _assert_materialized(harness)
    assert "source: public mirror" in result.stdout
    assert "::warning::" not in result.stdout


@pytest.mark.parametrize("status", [402, 401, 503])
def test_unavailable_registry_falls_back_to_the_mirror(fetch_step, status):
    harness, script, bash = fetch_step
    harness.fail(harness.registry_query, status)
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode == 0, result.stderr
    calls = harness.calls()
    assert [call["url"] for call in calls] == [
        harness.registry_query,
        harness.mirror_object,
    ]
    _assert_capped_https(calls)
    _assert_materialized(harness)
    assert (
        "::warning::Corpus release registry is unavailable; trying the public "
        "release mirror"
    ) in result.stdout
    assert f"returned error: {status}" in result.stderr


@pytest.mark.parametrize(
    ("registry_answer", "status"),
    [
        ("two rows", "ambiguous"),
        ({"message": "not a row list"}, "invalid"),
        (b"<html>maintenance</html>", "invalid"),
    ],
)
def test_ambiguous_or_malformed_registry_answers_fail_closed(
    fetch_step, registry_answer, status
):
    harness, script, bash = fetch_step
    if registry_answer == "two rows":
        harness.registry_rows(harness.release_object, harness.release_object)
    else:
        harness.serve(harness.registry_query, registry_answer)
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode == 1
    assert f"Corpus release registry returned an {status} response" in result.stderr
    assert [call["url"] for call in harness.calls()] == [harness.registry_query]
    assert not harness.materialized.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row["content"].update(extra=True),
        lambda row: row.update(release="us-rulespec-other"),
        lambda row: row.update(content_sha256="0" * 64),
    ],
)
def test_mismatched_registry_rows_fail_closed_without_fallback(fetch_step, mutate):
    harness, script, bash = fetch_step
    row = json.loads(json.dumps(harness.release_object))
    mutate(row)
    harness.registry_rows(row)
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode != 0
    assert "Corpus release acquisition error" in result.stderr
    assert [call["url"] for call in harness.calls()] == [harness.registry_query]
    assert not harness.materialized.exists()


def test_mismatched_mirror_objects_fail_closed(fetch_step):
    harness, script, bash = fetch_step
    tampered = json.loads(json.dumps(harness.release_object))
    tampered["content"]["extra"] = True
    harness.serve(harness.registry_query, [])
    harness.serve(harness.mirror_object, tampered)

    result = harness.run(script, bash)

    assert result.returncode != 0
    assert "content-address mismatch" in result.stderr
    assert not harness.materialized.exists()


@pytest.mark.parametrize(
    ("registry", "reason"), [("missing", "missing"), ("unavailable", "unavailable")]
)
def test_pin_missing_from_both_sources_names_both(fetch_step, registry, reason):
    harness, script, bash = fetch_step
    if registry == "missing":
        harness.serve(harness.registry_query, [])
    else:
        harness.fail(harness.registry_query, 402)
    harness.fail(harness.mirror_object, 404)

    result = harness.run(script, bash)

    assert result.returncode == 1
    assert (
        f"Signed corpus release {RELEASE_NAME}@{harness.digest} is unavailable: "
        f"registry {reason}, public mirror fetch failed"
    ) in result.stderr
    assert not harness.materialized.exists()


def test_targeted_reencode_requires_the_registry_anon_key(targeted):
    harness, script, bash = targeted
    harness.env["RELEASE_REGISTRY_ANON_KEY"] = ""

    result = harness.run(script, bash)

    assert result.returncode == 1
    assert harness.calls() == []


def test_signed_apply_without_a_registry_uses_only_the_mirror(signed_apply):
    harness, script, bash = signed_apply
    harness.env.update(RELEASE_REGISTRY_URL="", RELEASE_REGISTRY_ANON_KEY="")
    harness.serve(harness.mirror_object, harness.release_object)

    result = harness.run(script, bash)

    assert result.returncode == 0, result.stderr
    calls = harness.calls()
    assert [call["url"] for call in calls] == [harness.mirror_object]
    assert calls[0]["headers"] == []
    _assert_capped_https(calls)
    _assert_materialized(harness)
    assert "source: public mirror" in result.stdout


@pytest.mark.parametrize(
    ("registry_url", "anon_key", "message"),
    [
        (REGISTRY_URL, "", "anon-key is required with corpus-release-registry-url"),
        ("", ANON_KEY, "anon-key requires corpus-release-registry-url"),
        ("http://registry.example", ANON_KEY, "must use HTTPS"),
    ],
)
def test_signed_apply_rejects_incomplete_registry_inputs(
    signed_apply, registry_url, anon_key, message
):
    harness, script, bash = signed_apply
    harness.env.update(
        RELEASE_REGISTRY_URL=registry_url, RELEASE_REGISTRY_ANON_KEY=anon_key
    )

    result = harness.run(script, bash)

    assert result.returncode == 1
    assert message in result.stderr
    assert harness.calls() == []


@pytest.mark.parametrize(
    ("name", "sha", "message"),
    [
        ("../etc", None, "release name is invalid"),
        ("us-rulespec$(touch pwned)", None, "release name is invalid"),
        (None, "A" * 64, "release digest is invalid"),
        (None, "a" * 63, "release digest is invalid"),
    ],
)
def test_signed_apply_rejects_unsafe_release_pins_before_fetching(
    signed_apply, name, sha, message
):
    harness, script, bash = signed_apply
    if name is not None:
        harness.env["RELEASE_NAME"] = name
    if sha is not None:
        harness.env["RELEASE_SHA"] = sha

    result = harness.run(script, bash)

    assert result.returncode == 1
    assert message in result.stderr
    assert harness.calls() == []
    assert not (harness.workdir / "pwned").exists()


def test_signed_apply_registry_inputs_mirror_validate_rulespec():
    payload = yaml.load(
        (WORKFLOWS / "signed-apply-reusable.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    inputs = payload["on"]["workflow_call"]["inputs"]

    for name in ("corpus-release-registry-url", "corpus-release-registry-anon-key"):
        assert inputs[name]["required"] == "false"
        assert inputs[name]["type"] == "string"
        assert inputs[name]["default"] == ""
