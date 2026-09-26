"""Caller parsing and resolution for ``axiom-encode ci``.

Expected values come from the callers' own workflow semantics: rulespec-us's
real ``workflow-toolchain`` job and the pinned reusable workflow's
``workflow_call`` declarations and "Validate immutable dependency inputs" step.
Where a caller or workflow step is a script, the real script is run as an
oracle (bash, the heredoc, ``$GITHUB_OUTPUT``) and ci must agree with it.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from axiom_encode import ci_parity
from axiom_encode.ci_parity import (
    DEFAULT_RELEASE_BASE_URL,
    DEFAULT_RELEASE_REGISTRY_URL,
    DEPENDENCY_INPUTS,
    LEGACY_WORKFLOW_INPUTS,
    RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS,
    RELEASE_REGISTRY_ANON_KEY_ENV,
    SUPPORTED_WORKFLOW_PINS,
    WORKFLOW_INPUTS_0EFFA6A5,
    CallerConfig,
    CallerOverrides,
    PinnedWorkflow,
    WorkflowInput,
    _caller_overrides,
    find_caller_workflow,
    parse_caller_workflow,
    register_ci_parser,
    run_ci,
    verify_dependency_inputs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
US_CALLER_TEXT = (FIXTURES / "us-caller.yml").read_text(encoding="utf-8")
US_TOOLCHAIN_TEXT = (FIXTURES / "us-workflow-toolchain.toml").read_text(
    encoding="utf-8"
)
PIN_0EFFA6A5 = "0effa6a5b05e7fac53902df7d523e909bd7fc48a"
PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
PIN_615C1DF9 = "615c1df9b9ace7deea84da65efd137f46f8bad2b"
PIN_34BCFAB2 = "34bcfab235c585c47292c95f51be1a4f4f91d29e"
EMBEDDED_PINS = (PIN_0EFFA6A5, PIN_6F11BE26)
LEGACY_PINS = (PIN_615C1DF9, PIN_34BCFAB2)
WORKFLOW_USES = "TheAxiomFoundation/.github/.github/workflows/validate-rulespec.yml@"
TOOLCHAIN_PATH = ".axiom/workflow-toolchain.toml"
# The four checkout identities in us-workflow-toolchain.toml.
ENCODE = "f856cfcb886d9bd050b228aa60aeb4b96939f739"
ENGINE = "af6e4ea2920b0c0a97bf6a6f45b0c6643e93c0ca"
CORPUS = "8f7d60aaced28ee4252b9237f9d6e02360dc34bc"
RULESPEC_US = "38d5c8c516f9243cedf8e07a22f96dde9ac66fe3"
US_REFS = {
    "encode": ENCODE,
    "engine": ENGINE,
    "corpus": CORPUS,
    "rulespec_us": RULESPEC_US,
}
# Caller input name -> workflow-toolchain job output it reads.
REF_OUTPUTS = {
    "axiom-encode-ref": "axiom_encode_ref",
    "axiom-rules-engine-ref": "axiom_rules_engine_ref",
    "axiom-corpus-ref": "axiom_corpus_ref",
    "rulespec-us-ref": "rulespec_us_ref",
}
ANON_KEY = "anon-key-value-for-tests-7f3a"
WITH_ANON_KEY = CallerOverrides(registry_anon_key=ANON_KEY)
REGISTRY_URL_EXPRESSION = "${{ vars.NEXT_PUBLIC_SUPABASE_URL }}"
ANON_KEY_EXPRESSION = "${{ vars.NEXT_PUBLIC_SUPABASE_ANON_KEY }}"
INPUT_EXPRESSION = re.compile(r"\$\{\{\s*inputs\.(?P<name>[A-Za-z0-9_-]+)\s*\}\}")
DELETE = object()


@pytest.fixture(autouse=True)
def _no_ambient_registry_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RELEASE_REGISTRY_ANON_KEY_ENV, raising=False)


@pytest.fixture
def embedded_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every embedded-script execution, then run the real script."""

    calls: list[dict[str, Any]] = []
    real = ci_parity._run_embedded_python

    def spy(script: str, **kwargs: Any) -> tuple[int, str, str]:
        calls.append({"script": script, **kwargs})
        return real(script, **kwargs)

    monkeypatch.setattr(ci_parity, "_run_embedded_python", spy)
    return calls


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _us_caller() -> dict[str, Any]:
    return yaml.safe_load(US_CALLER_TEXT)


def _resolver_job(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["jobs"]["workflow-toolchain"]


def _resolver_step(payload: dict[str, Any]) -> dict[str, Any]:
    return _resolver_job(payload)["steps"][1]


def _validate_job(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["jobs"]["validate"]


def _with(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate_job(payload)["with"]


def _rules_repo(
    tmp_path: Path,
    *,
    caller: dict[str, Any] | str | None = US_CALLER_TEXT,
    toolchain: str | None = US_TOOLCHAIN_TEXT,
    workflow_name: str = "repository-checks.yml",
) -> Path:
    """A committed rules checkout with a caller workflow and toolchain file."""

    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("rules\n")
    if caller is not None:
        workflows = repo / ".github" / "workflows"
        workflows.mkdir(parents=True)
        text = (
            caller
            if isinstance(caller, str)
            else yaml.safe_dump(caller, sort_keys=False)
        )
        (workflows / workflow_name).write_text(text, encoding="utf-8")
    if toolchain is not None:
        (repo / ".axiom").mkdir()
        (repo / TOOLCHAIN_PATH).write_text(toolchain, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "rules")
    _git(
        repo, "update-ref", "refs/remotes/origin/main", _git(repo, "rev-parse", "HEAD")
    )
    return repo


def _parse(repo: Path, overrides: CallerOverrides = WITH_ANON_KEY) -> CallerConfig:
    return find_caller_workflow(repo, overrides=overrides)


def _with_updates(updates: dict[str, Any]) -> dict[str, Any]:
    payload = _us_caller()
    inputs = _with(payload)
    for name, value in updates.items():
        if value is DELETE:
            del inputs[name]
        else:
            inputs[name] = value
    return payload


def _set_toolchain_key(text: str, key: str, raw: str | None) -> str:
    pattern = re.compile(rf"(?m)^{re.escape(key)} = .*\n")
    assert pattern.search(text), key
    return pattern.sub("" if raw is None else f"{key} = {raw}\n", text)


def _resolver_body(run: str) -> str:
    """The quoted heredoc that bash feeds to ``python -`` in the resolver step."""

    head, rest = run.split("<<'PY' >> \"$GITHUB_OUTPUT\"\n", 1)
    assert head == "set -euo pipefail\npython - "
    body, tail = rest.split("\nPY\n", 1)
    assert tail == ""
    return body + "\n"


def _python_shim(tmp_path: Path) -> Path:
    shim = tmp_path / "python-shim"
    shim.mkdir(exist_ok=True)
    python = shim / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    return shim


def _step_environment(tmp_path: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("GITHUB_", "RUNNER_"))
    }
    environment["PATH"] = f"{_python_shim(tmp_path)}{os.pathsep}{os.environ['PATH']}"
    return environment


def _bash(run: str, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    # GitHub writes the run block to a file and executes `shell: bash` as
    # `bash --noprofile --norc -eo pipefail {0}`.
    with tempfile.TemporaryDirectory(prefix="step-") as temp_name:
        script = Path(temp_name) / "step.sh"
        script.write_text(run, encoding="utf-8")
        return subprocess.run(
            [
                shutil.which("bash") or "bash",
                "--noprofile",
                "--norc",
                "-eo",
                "pipefail",
                str(script),
            ],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )


@functools.cache
def _pinned_workflow(sha: str) -> PinnedWorkflow:
    return PinnedWorkflow(sha)


def _run_caller_resolver_job(repo: Path, tmp_path: Path) -> tuple[int, str, dict]:
    """Run rulespec-us's real "Read protected workflow toolchain" step."""

    output = tmp_path / "github-output"
    output.write_text("")
    environment = _step_environment(tmp_path)
    environment["GITHUB_OUTPUT"] = str(output)
    result = _bash(_resolver_step(_us_caller())["run"], cwd=repo, env=environment)
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    return result.returncode, result.stderr, outputs


# ---------------------------------------------------------------------------
# rulespec-us's real caller
# ---------------------------------------------------------------------------


def test_real_us_resolver_script_is_the_reviewed_one() -> None:
    run = _resolver_step(_us_caller())["run"]
    digest = hashlib.sha256(run.encode("utf-8")).hexdigest()

    assert digest in RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS
    assert "rulespec-us" in RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS[digest]


def test_rulespec_us_caller_resolves_every_input(
    tmp_path: Path, embedded_calls: list[dict[str, Any]]
) -> None:
    repo = _rules_repo(tmp_path)

    caller = _parse(repo)

    assert caller.path == repo / ".github" / "workflows" / "repository-checks.yml"
    assert caller.workflow_sha == PIN_0EFFA6A5
    assert caller.refs == US_REFS
    assert caller.validate_roots == "auto"
    assert caller.run_generated_guard is True
    assert caller.guard_programs_root is False
    assert caller.run_pytest is True
    assert caller.run_money_atom_check is True
    assert caller.release_base_url == DEFAULT_RELEASE_BASE_URL
    # Every declared input: the caller's values, else the workflow defaults.
    assert dict(caller.inputs) == {
        "python-version": "3.14",
        "axiom-encode-ref": ENCODE,
        "axiom-rules-engine-ref": ENGINE,
        "axiom-corpus-ref": CORPUS,
        "rulespec-us-ref": RULESPEC_US,
        "corpus-release-base-url": DEFAULT_RELEASE_BASE_URL,
        "corpus-release-registry-url": DEFAULT_RELEASE_REGISTRY_URL,
        "corpus-release-registry-anon-key": ANON_KEY,
        "validate-roots": "auto",
        "validation-workers": 4,
        "run-pytest": True,
        "run-generated-guard": True,
        "migration-authorization-path": "",
        "retired-schema-bootstrap-sha256": "",
        "allow-retired-schema-prefreeze": True,
        "validation-waiver-bootstrap-sha256": "",
        "guard-programs-root": False,
        "run-money-atom-check": True,
    }
    assert type(caller.inputs["validation-workers"]) is int
    assert caller.resolutions == (
        "axiom-encode-ref: ${{ needs.workflow-toolchain.outputs.axiom_encode_ref }}"
        f" -> {ENCODE}",
        "axiom-rules-engine-ref: "
        "${{ needs.workflow-toolchain.outputs.axiom_rules_engine_ref }}"
        f" -> {ENGINE}",
        "axiom-corpus-ref: ${{ needs.workflow-toolchain.outputs.axiom_corpus_ref }}"
        f" -> {CORPUS}",
        "rulespec-us-ref: ${{ needs.workflow-toolchain.outputs.rulespec_us_ref }}"
        f" -> {RULESPEC_US}",
        f"corpus-release-registry-url: {REGISTRY_URL_EXPRESSION} -> "
        f"{DEFAULT_RELEASE_REGISTRY_URL} (the organization registry default)",
        f"corpus-release-registry-anon-key: {ANON_KEY_EXPRESSION} -> "
        "<--corpus-release-registry-anon-key>",
    )
    # The resolver job ran once, unchanged, from the rules checkout.
    assert len(embedded_calls) == 1
    (call,) = embedded_calls
    assert call["script"] == _resolver_body(_resolver_step(_us_caller())["run"])
    assert call["cwd"] == repo
    assert call["environment"] == {"GITHUB_OUTPUT": os.devnull}
    # And the resolved caller satisfies "Validate immutable dependency inputs".
    verify_dependency_inputs(caller)


def test_round_tripped_us_caller_is_still_the_reviewed_resolver(
    tmp_path: Path,
) -> None:
    # Guards the dict-mutation helpers below: a yaml round trip keeps the
    # resolver's run script byte-identical, so only the mutation differs.
    repo = _rules_repo(tmp_path, caller=_us_caller())

    assert _parse(repo).refs == US_REFS


@pytest.mark.parametrize("pin", EMBEDDED_PINS, ids=lambda sha: sha[:8])
def test_us_caller_parses_at_each_embedded_script_pin(tmp_path: Path, pin: str) -> None:
    payload = _us_caller()
    _validate_job(payload)["uses"] = WORKFLOW_USES + pin
    repo = _rules_repo(tmp_path, caller=payload)

    caller = _parse(repo)

    assert caller.workflow_sha == pin
    assert caller.refs == US_REFS
    assert caller.inputs["validation-workers"] == 4
    assert caller.inputs["allow-retired-schema-prefreeze"] is True


def test_parse_caller_workflow_reads_toolchain_from_the_given_checkout(
    tmp_path: Path,
) -> None:
    repo = _rules_repo(tmp_path, caller=None)

    caller = parse_caller_workflow(
        FIXTURES / "us-caller.yml", repo=repo, overrides=WITH_ANON_KEY
    )

    assert caller.path == FIXTURES / "us-caller.yml"
    assert caller.refs == US_REFS


def test_parse_caller_workflow_infers_checkout_from_workflows_directory(
    tmp_path: Path,
) -> None:
    repo = _rules_repo(tmp_path)

    caller = parse_caller_workflow(
        repo / ".github" / "workflows" / "repository-checks.yml",
        overrides=WITH_ANON_KEY,
    )

    assert caller.refs == US_REFS


def test_needs_refs_without_a_checkout_fail_closed(
    embedded_calls: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="no checkout to read its workflow toolchain"):
        parse_caller_workflow(FIXTURES / "us-caller.yml", overrides=WITH_ANON_KEY)
    assert embedded_calls == []


# ---------------------------------------------------------------------------
# corpus-release-registry inputs passed as ${{ vars.* }}
# ---------------------------------------------------------------------------


def test_registry_url_override_replaces_the_organization_default(
    tmp_path: Path,
) -> None:
    repo = _rules_repo(tmp_path)

    caller = _parse(
        repo,
        CallerOverrides(
            registry_url="https://registry.example.test", registry_anon_key=ANON_KEY
        ),
    )

    assert caller.inputs["corpus-release-registry-url"] == (
        "https://registry.example.test"
    )
    assert (
        f"corpus-release-registry-url: {REGISTRY_URL_EXPRESSION} -> "
        "https://registry.example.test (--corpus-release-registry-url)"
    ) in caller.resolutions


def test_literal_registry_inputs_pass_through_unchanged(tmp_path: Path) -> None:
    # CI passes a literal with: value as-is; overrides only stand in for vars.*.
    repo = _rules_repo(
        tmp_path,
        caller=_with_updates(
            {
                "corpus-release-registry-url": "https://literal.example.test",
                "corpus-release-registry-anon-key": "literal-anon-key",
            }
        ),
    )

    caller = _parse(
        repo,
        CallerOverrides(
            registry_url="https://override.example.test",
            registry_anon_key="override-key",
        ),
    )

    assert caller.inputs["corpus-release-registry-url"] == (
        "https://literal.example.test"
    )
    assert caller.inputs["corpus-release-registry-anon-key"] == "literal-anon-key"
    assert not any(
        line.startswith("corpus-release-registry") for line in caller.resolutions
    )


def test_anon_key_override_wins_over_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELEASE_REGISTRY_ANON_KEY_ENV, "environment-key-9c1d")
    repo = _rules_repo(tmp_path)

    caller = _parse(repo, CallerOverrides(registry_anon_key=ANON_KEY))

    assert caller.inputs["corpus-release-registry-anon-key"] == ANON_KEY
    assert (
        f"corpus-release-registry-anon-key: {ANON_KEY_EXPRESSION} -> "
        "<--corpus-release-registry-anon-key>"
    ) in caller.resolutions
    assert not any(ANON_KEY in line for line in caller.resolutions)


def test_anon_key_falls_back_to_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(RELEASE_REGISTRY_ANON_KEY_ENV, "environment-key-9c1d")
    repo = _rules_repo(tmp_path)

    caller = _parse(repo, CallerOverrides())

    assert caller.inputs["corpus-release-registry-anon-key"] == "environment-key-9c1d"
    assert (
        f"corpus-release-registry-anon-key: {ANON_KEY_EXPRESSION} -> "
        f"<${RELEASE_REGISTRY_ANON_KEY_ENV}>"
    ) in caller.resolutions
    assert not any("environment-key-9c1d" in line for line in caller.resolutions)


@pytest.mark.parametrize("environment", [None, ""], ids=["unset", "empty"])
def test_anon_key_without_override_or_environment_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environment: str | None
) -> None:
    if environment is not None:
        monkeypatch.setenv(RELEASE_REGISTRY_ANON_KEY_ENV, environment)
    repo = _rules_repo(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        _parse(repo, CallerOverrides())

    message = str(excinfo.value)
    assert "--corpus-release-registry-anon-key" in message
    assert RELEASE_REGISTRY_ANON_KEY_ENV in message
    assert "corpus-release-registry-anon-key comes from" in message
    assert ANON_KEY_EXPRESSION in message


def test_ci_parser_flags_become_caller_overrides(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    register_ci_parser(parser.add_subparsers(dest="command"))
    base = ["ci", "--repo", str(tmp_path), "--corpus-release-public-key", "k"]

    assert _caller_overrides(parser.parse_args(base)) == CallerOverrides()
    assert _caller_overrides(
        parser.parse_args(
            [
                *base,
                "--corpus-release-registry-url",
                "https://registry.example.test",
                "--corpus-release-registry-anon-key",
                ANON_KEY,
            ]
        )
    ) == CallerOverrides("https://registry.example.test", ANON_KEY)


def _run_ci_json(
    repo: Path, capsys: pytest.CaptureFixture[str], *extra: str
) -> tuple[int, dict[str, Any], str]:
    parser = argparse.ArgumentParser()
    register_ci_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "ci",
            "--repo",
            str(repo),
            "--corpus-release-public-key",
            "test-public-key",
            "--json",
            *extra,
        ]
    )
    code = run_ci(args)
    out = capsys.readouterr().out
    return code, json.loads(out), out


def test_run_ci_reports_missing_anon_key_as_resolution_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _rules_repo(tmp_path)

    code, report, _ = _run_ci_json(repo, capsys)

    assert code == 1
    assert report["passed"] is False
    assert report["verdict"] == "FAIL"
    assert report["gates"] == []
    assert report["resolutions"] == []
    assert "--corpus-release-registry-anon-key" in report["resolution_error"]


def test_run_ci_applies_registry_flags_before_dependency_input_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _rules_repo(tmp_path)

    code, report, out = _run_ci_json(
        repo,
        capsys,
        "--corpus-release-registry-url",
        "http://registry.example.test",
        "--corpus-release-registry-anon-key",
        ANON_KEY,
    )

    assert code == 1
    assert report["verdict"] == "FAIL"
    assert report["resolution_error"] == "corpus-release-registry-url must use HTTPS"
    assert (
        f"corpus-release-registry-url: {REGISTRY_URL_EXPRESSION} -> "
        "http://registry.example.test (--corpus-release-registry-url)"
    ) in report["resolutions"]
    assert ANON_KEY not in out


# ---------------------------------------------------------------------------
# The workflow-toolchain resolver mirrors rulespec-us's job exactly
# ---------------------------------------------------------------------------

TOOLCHAIN_CASES: dict[str, tuple[Callable[[str], str | None], str | None]] = {
    "fixture": (lambda text: text, None),
    # Keys the job does not check are not validated either.
    "unchecked-key-invalid": (
        lambda text: _set_toolchain_key(text, "axiom_compose_ref", '"not-a-sha"'),
        None,
    ),
    "encode-invalid": (
        lambda text: _set_toolchain_key(text, "axiom_encode_ref", '"not-a-sha"'),
        f"{TOOLCHAIN_PATH}: axiom_encode_ref must be a full lowercase commit SHA",
    ),
    "engine-uppercase": (
        lambda text: _set_toolchain_key(
            text, "axiom_rules_engine_ref", f'"{ENGINE.upper()}"'
        ),
        f"{TOOLCHAIN_PATH}: axiom_rules_engine_ref must be a full lowercase commit SHA",
    ),
    "corpus-short": (
        lambda text: _set_toolchain_key(text, "axiom_corpus_ref", f'"{CORPUS[:39]}"'),
        f"{TOOLCHAIN_PATH}: axiom_corpus_ref must be a full lowercase commit SHA",
    ),
    "rulespec-us-missing": (
        lambda text: _set_toolchain_key(text, "rulespec_us_ref", None),
        f"{TOOLCHAIN_PATH}: rulespec_us_ref must be a full lowercase commit SHA",
    ),
    "rulespec-us-integer": (
        lambda text: _set_toolchain_key(text, "rulespec_us_ref", "42"),
        f"{TOOLCHAIN_PATH}: rulespec_us_ref must be a full lowercase commit SHA",
    ),
    "table-missing": (
        lambda text: text.replace("[workflow_toolchain]", "[toolchain]"),
        "KeyError: 'workflow_toolchain'",
    ),
    "invalid-toml": (lambda text: text + "this is not toml\n", "TOMLDecodeError"),
    "file-missing": (lambda text: None, "FileNotFoundError"),
}


@pytest.mark.parametrize("case", sorted(TOOLCHAIN_CASES))
def test_workflow_toolchain_resolution_matches_the_caller_job(
    tmp_path: Path, case: str
) -> None:
    mutate, expected_error = TOOLCHAIN_CASES[case]
    repo = _rules_repo(tmp_path, toolchain=mutate(US_TOOLCHAIN_TEXT))

    job_code, job_stderr, job_outputs = _run_caller_resolver_job(repo, tmp_path)

    if expected_error is None:
        assert job_code == 0, job_stderr
        caller = _parse(repo)
        assert caller.refs == {
            dependency: job_outputs[REF_OUTPUTS[name]]
            for dependency, name in DEPENDENCY_INPUTS.items()
        }
        assert caller.refs == US_REFS
        return
    assert job_code != 0
    assert expected_error in job_stderr
    with pytest.raises(ValueError) as excinfo:
        _parse(repo)
    message = str(excinfo.value)
    assert "workflow-toolchain job workflow-toolchain fails" in message
    # The job's own final diagnostic, verbatim.
    assert job_stderr.strip().splitlines()[-1] in message
    assert expected_error in message


@pytest.mark.parametrize("key", sorted(REF_OUTPUTS.values()))
def test_any_invalid_toolchain_ref_fails_even_when_the_caller_reads_one(
    tmp_path: Path, key: str
) -> None:
    # Only axiom-encode-ref reads the job; the job still validates all four
    # keys, and CI skips the validate job when its needs job fails.
    payload = _with_updates(
        {
            "axiom-rules-engine-ref": ENGINE,
            "axiom-corpus-ref": CORPUS,
            "rulespec-us-ref": RULESPEC_US,
        }
    )
    toolchain = _set_toolchain_key(US_TOOLCHAIN_TEXT, key, '"0000"')
    repo = _rules_repo(tmp_path, caller=payload, toolchain=toolchain)

    with pytest.raises(
        ValueError,
        match=re.escape(f"{TOOLCHAIN_PATH}: {key} must be a full lowercase commit SHA"),
    ):
        _parse(repo)


def test_caller_reading_one_output_resolves_only_that_input(tmp_path: Path) -> None:
    payload = _with_updates(
        {
            "axiom-rules-engine-ref": "1" * 40,
            "axiom-corpus-ref": "2" * 40,
            "rulespec-us-ref": "3" * 40,
        }
    )
    repo = _rules_repo(tmp_path, caller=payload)

    caller = _parse(repo)

    assert caller.refs == {
        "encode": ENCODE,
        "engine": "1" * 40,
        "corpus": "2" * 40,
        "rulespec_us": "3" * 40,
    }
    assert [line.split(":", 1)[0] for line in caller.resolutions] == [
        "axiom-encode-ref",
        "corpus-release-registry-url",
        "corpus-release-registry-anon-key",
    ]


def test_needs_may_be_a_list(tmp_path: Path) -> None:
    payload = _us_caller()
    _validate_job(payload)["needs"] = ["workflow-toolchain"]
    repo = _rules_repo(tmp_path, caller=payload)

    assert _parse(repo).refs == US_REFS


def test_failing_needs_resolver_fails_even_when_no_input_reads_it(
    tmp_path: Path,
) -> None:
    payload = _with_updates(
        {
            "axiom-encode-ref": ENCODE,
            "axiom-rules-engine-ref": ENGINE,
            "axiom-corpus-ref": CORPUS,
            "rulespec-us-ref": RULESPEC_US,
        }
    )
    toolchain = _set_toolchain_key(US_TOOLCHAIN_TEXT, "axiom_encode_ref", '"bad"')
    repo = _rules_repo(tmp_path, caller=payload, toolchain=toolchain)

    with pytest.raises(ValueError, match="axiom_encode_ref must be a full lowercase"):
        _parse(repo)


# ---------------------------------------------------------------------------
# Resolver recognition fails closed
# ---------------------------------------------------------------------------


def _modify_script(payload: dict[str, Any]) -> None:
    step = _resolver_step(payload)
    step["run"] = step["run"].replace(
        'print(f"{key}={value}")', 'print(f"{key}={value}")  # adjusted'
    )


def _add_step(payload: dict[str, Any]) -> None:
    _resolver_job(payload)["steps"].append({"name": "Extra", "run": "true"})


def _drop_checkout(payload: dict[str, Any]) -> None:
    del _resolver_job(payload)["steps"][0]


def _swap_steps(payload: dict[str, Any]) -> None:
    _resolver_job(payload)["steps"].reverse()


def _checkout_with_ref(payload: dict[str, Any]) -> None:
    _resolver_job(payload)["steps"][0]["with"] = {"ref": "refs/heads/other"}


def _foreign_checkout(payload: dict[str, Any]) -> None:
    _resolver_job(payload)["steps"][0]["uses"] = "someone/checkout@v1"


def _job_key(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        _resolver_job(payload)[key] = value

    return mutate


def _resolver_key(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        if value is DELETE:
            del _resolver_step(payload)[key]
        else:
            _resolver_step(payload)[key] = value

    return mutate


def _drop_outputs(payload: dict[str, Any]) -> None:
    del _resolver_job(payload)["outputs"]


def _resolver_calls_workflow(payload: dict[str, Any]) -> None:
    payload["jobs"]["workflow-toolchain"] = {
        "uses": "TheAxiomFoundation/.github/.github/workflows/toolchain.yml@main"
    }


UNRECOGNIZED_BEFORE_EXECUTION: dict[str, tuple[Callable[..., None], str]] = {
    "extra-step": (_add_step, "expected a checkout step and one resolver step"),
    "no-checkout": (_drop_checkout, "expected a checkout step and one resolver step"),
    "steps-swapped": (_swap_steps, "the first step is not a plain rules-repository"),
    "checkout-with-ref": (
        _checkout_with_ref,
        "the first step is not a plain rules-repository checkout",
    ),
    "foreign-checkout": (
        _foreign_checkout,
        "the first step is not a plain rules-repository checkout",
    ),
    "job-if": (_job_key("if", "github.event_name == 'push'"), "['if']"),
    "job-env": (_job_key("env", {"PYTHONPATH": "evil"}), "['env']"),
    "job-needs": (_job_key("needs", "setup"), "['needs']"),
    "job-defaults": (
        _job_key("defaults", {"run": {"working-directory": "elsewhere"}}),
        "['defaults']",
    ),
    "job-continue-on-error": (
        _job_key("continue-on-error", True),
        "['continue-on-error']",
    ),
    "job-strategy": (_job_key("strategy", {"matrix": {"x": [1]}}), "['strategy']"),
    "resolver-env": (
        _resolver_key("env", {"PYTHONPATH": "evil"}),
        "the resolver step has an unexpected shape",
    ),
    "resolver-working-directory": (
        _resolver_key("working-directory", "elsewhere"),
        "the resolver step has an unexpected shape",
    ),
    "resolver-continue-on-error": (
        _resolver_key("continue-on-error", True),
        "the resolver step has an unexpected shape",
    ),
    "resolver-shell-sh": (
        _resolver_key("shell", "sh"),
        "the resolver step has an unexpected shape",
    ),
    "resolver-no-shell": (
        _resolver_key("shell", DELETE),
        "the resolver step has an unexpected shape",
    ),
    "resolver-no-id": (
        _resolver_key("id", DELETE),
        "the resolver step has an unexpected shape",
    ),
    "no-outputs": (_drop_outputs, "(no outputs)"),
    "reusable-workflow-job": (_resolver_calls_workflow, "(no such job)"),
}


@pytest.mark.parametrize("case", sorted(UNRECOGNIZED_BEFORE_EXECUTION))
def test_unrecognized_resolver_fails_closed_without_running_it(
    tmp_path: Path, case: str, embedded_calls: list[dict[str, Any]]
) -> None:
    mutate, fragment = UNRECOGNIZED_BEFORE_EXECUTION[case]
    payload = _us_caller()
    mutate(payload)
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError) as excinfo:
        _parse(repo)

    message = str(excinfo.value)
    assert "job workflow-toolchain is not a recognized workflow-toolchain resolver" in (
        message
    )
    assert fragment in message
    assert embedded_calls == []


def test_modified_resolver_script_is_not_reviewed_and_never_runs(
    tmp_path: Path, embedded_calls: list[dict[str, Any]]
) -> None:
    payload = _us_caller()
    _modify_script(payload)
    digest = hashlib.sha256(_resolver_step(payload)["run"].encode()).hexdigest()
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(
        ValueError, match=f"resolver script sha256 {digest} is not reviewed"
    ):
        _parse(repo)

    assert digest not in RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS
    assert embedded_calls == []


def _output_source(output: str, value: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        _resolver_job(payload)["outputs"][output] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "output"),
    [
        (
            _output_source(
                "axiom_encode_ref", "${{ steps.other.outputs.axiom_encode_ref }}"
            ),
            "axiom_encode_ref",
        ),
        # The job's script never prints axiom_compose_ref.
        (
            _output_source(
                "rulespec_us_ref", "${{ steps.pins.outputs.axiom_compose_ref }}"
            ),
            "rulespec_us_ref",
        ),
        (_output_source("axiom_corpus_ref", CORPUS), "axiom_corpus_ref"),
        (
            _output_source(
                "axiom_rules_engine_ref", "${{ env.AXIOM_RULES_ENGINE_REF }}"
            ),
            "axiom_rules_engine_ref",
        ),
    ],
    ids=["other-step", "unprinted-output", "literal", "env-expression"],
)
def test_outputs_must_map_to_the_resolver_steps_outputs(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None], output: str
) -> None:
    payload = _us_caller()
    mutate(payload)
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(
        ValueError, match=re.escape(f"output {output} is not a resolver step output")
    ):
        _parse(repo)


def test_validate_job_must_list_the_resolver_in_needs(
    tmp_path: Path, embedded_calls: list[dict[str, Any]]
) -> None:
    payload = _us_caller()
    del _validate_job(payload)["needs"]
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(
        ValueError,
        match=(
            "axiom-encode-ref reads needs.workflow-toolchain, but the "
            "validate-rulespec job does not list workflow-toolchain in needs"
        ),
    ):
        _parse(repo)
    assert embedded_calls == []


def test_needs_reference_to_an_unlisted_job_fails(tmp_path: Path) -> None:
    payload = _with_updates(
        {"axiom-corpus-ref": "${{ needs.toolchain.outputs.axiom_corpus_ref }}"}
    )
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="does not list toolchain in needs") as excinfo:
        _parse(repo)
    assert "axiom-corpus-ref reads needs.toolchain" in str(excinfo.value)


def test_needs_reference_to_an_undefined_job_fails(
    tmp_path: Path, embedded_calls: list[dict[str, Any]]
) -> None:
    payload = _with_updates(
        {
            "axiom-encode-ref": "${{ needs.ghost.outputs.axiom_encode_ref }}",
        }
    )
    _validate_job(payload)["needs"] = ["ghost", "workflow-toolchain"]
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(
        ValueError, match=r"job ghost is not a recognized .*no such job"
    ):
        _parse(repo)
    assert embedded_calls == []


def test_needs_reference_to_an_unknown_output_fails(tmp_path: Path) -> None:
    payload = _with_updates(
        {"rulespec-us-ref": "${{ needs.workflow-toolchain.outputs.nope }}"}
    )
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="job workflow-toolchain has no output nope"):
        _parse(repo)


# ---------------------------------------------------------------------------
# Typed inputs against the pin declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sha", sorted(SUPPORTED_WORKFLOW_PINS), ids=lambda s: s[:8])
def test_input_tables_are_the_pinned_workflow_call_declarations(sha: str) -> None:
    declared = _pinned_workflow(sha).payload[True]["workflow_call"]["inputs"]
    expected = {
        name: WorkflowInput(
            declaration["type"],
            required=bool(declaration.get("required", False)),
            default=declaration.get("default"),
        )
        for name, declaration in declared.items()
    }

    assert dict(SUPPORTED_WORKFLOW_PINS[sha].inputs) == expected
    assert SUPPORTED_WORKFLOW_PINS[sha].inputs is (
        LEGACY_WORKFLOW_INPUTS if sha in LEGACY_PINS else WORKFLOW_INPUTS_0EFFA6A5
    )


REJECTED_INPUTS: dict[str, tuple[dict[str, Any], str]] = {
    "unknown-input": (
        {"bogus-input": "x"},
        "validate-rulespec@0effa6a5 declares no input bogus-input",
    ),
    "unknown-inputs-sorted": (
        {"zeta-input": "1", "alpha-input": "2"},
        "declares no input alpha-input, zeta-input",
    ),
    "boolean-quoted-false": (
        {"guard-programs-root": "false"},
        "input guard-programs-root must be a boolean, got 'false'",
    ),
    "boolean-quoted-true": (
        {"allow-retired-schema-prefreeze": "true"},
        "input allow-retired-schema-prefreeze must be a boolean, got 'true'",
    ),
    "boolean-number": ({"run-pytest": 1}, "input run-pytest must be a boolean, got 1"),
    "number-string": (
        {"validation-workers": "4"},
        "input validation-workers must be a number, got '4'",
    ),
    "number-boolean": (
        {"validation-workers": True},
        "input validation-workers must be a number, got True",
    ),
    "string-boolean": (
        {"validate-roots": True},
        "input validate-roots must be a string, got True",
    ),
    "required-ref-missing": (
        {"rulespec-us-ref": DELETE},
        "required input rulespec-us-ref is missing",
    ),
    "expression-on-boolean": (
        {"guard-programs-root": "${{ vars.GUARD_PROGRAMS_ROOT }}"},
        "input guard-programs-root uses an expression ci cannot reproduce",
    ),
    "expression-on-number": (
        {"validation-workers": "${{ vars.VALIDATION_WORKERS }}"},
        "input validation-workers uses an expression ci cannot reproduce",
    ),
    "github-context": (
        {"validate-roots": "${{ github.sha }}"},
        "input validate-roots uses an expression ci cannot reproduce",
    ),
    "steps-context": (
        {"axiom-encode-ref": "${{ steps.pins.outputs.axiom_encode_ref }}"},
        "input axiom-encode-ref uses an expression ci cannot reproduce",
    ),
    "partial-string": (
        {"validate-roots": "x-${{ vars.Y }}"},
        "input validate-roots uses an expression ci cannot reproduce",
    ),
    "two-expressions": (
        {"validate-roots": "${{ vars.A }} ${{ vars.B }}"},
        "input validate-roots uses an expression ci cannot reproduce",
    ),
    "expression-operator": (
        {
            "corpus-release-registry-url": (
                "${{ vars.NEXT_PUBLIC_SUPABASE_URL || 'https://x.test' }}"
            )
        },
        "input corpus-release-registry-url uses an expression ci cannot reproduce",
    ),
    "vars-on-validate-roots": (
        {"validate-roots": "${{ vars.VALIDATE_ROOTS }}"},
        "input validate-roots reads ${{ vars.VALIDATE_ROOTS }}; ci resolves "
        "repository variables only for the corpus release registry inputs",
    ),
    "vars-on-base-url": (
        {"corpus-release-base-url": "${{ vars.RELEASE_BASE_URL }}"},
        "input corpus-release-base-url reads ${{ vars.RELEASE_BASE_URL }}",
    ),
    "vars-on-ref": (
        {"axiom-encode-ref": "${{ vars.AXIOM_ENCODE_REF }}"},
        "input axiom-encode-ref reads ${{ vars.AXIOM_ENCODE_REF }}",
    ),
    "literal-uppercase-ref": (
        {"axiom-corpus-ref": CORPUS.upper()},
        "axiom-corpus-ref must be a full lowercase SHA",
    ),
}


@pytest.mark.parametrize("case", sorted(REJECTED_INPUTS))
def test_caller_inputs_are_type_checked_against_the_pin(
    tmp_path: Path, case: str
) -> None:
    updates, fragment = REJECTED_INPUTS[case]
    repo = _rules_repo(tmp_path, caller=_with_updates(updates))

    with pytest.raises(ValueError) as excinfo:
        _parse(repo)

    assert fragment in str(excinfo.value)


def test_caller_without_with_inputs_fails(tmp_path: Path) -> None:
    payload = _us_caller()
    del _validate_job(payload)["with"]
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="has no with-inputs"):
        _parse(repo)


def test_expression_without_spaces_is_still_resolved(tmp_path: Path) -> None:
    repo = _rules_repo(
        tmp_path,
        caller=_with_updates(
            {
                "axiom-encode-ref": "${{needs.workflow-toolchain.outputs.axiom_encode_ref}}"
            }
        ),
    )

    assert _parse(repo).refs["encode"] == ENCODE


@pytest.mark.parametrize("literal", ["yes", "no", "on", "off"])
def test_yaml_1_1_boolean_words_are_strings_to_github(
    tmp_path: Path, literal: str
) -> None:
    text = US_CALLER_TEXT.replace(
        "guard-programs-root: false", f"guard-programs-root: {literal}"
    )
    assert text != US_CALLER_TEXT
    repo = _rules_repo(tmp_path, caller=text)

    with pytest.raises(ValueError, match="guard-programs-root must be a boolean"):
        _parse(repo)


def test_placeholder_pin_is_rejected(tmp_path: Path) -> None:
    payload = _us_caller()
    _validate_job(payload)["uses"] = WORKFLOW_USES + "<pin-me>"
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="placeholder pin <pin-me>"):
        _parse(repo)


def test_unsupported_pin_lists_supported_pins(tmp_path: Path) -> None:
    unsupported = "a" * 40
    payload = _us_caller()
    _validate_job(payload)["uses"] = WORKFLOW_USES + unsupported
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError) as excinfo:
        _parse(repo)

    message = str(excinfo.value)
    assert f"Unsupported validate-rulespec workflow pin {unsupported}" in message
    for pin in (PIN_615C1DF9, PIN_34BCFAB2, PIN_0EFFA6A5, PIN_6F11BE26):
        assert pin in message


@pytest.mark.parametrize(
    "ref", ["main", PIN_0EFFA6A5.upper(), PIN_0EFFA6A5[:12]], ids=str
)
def test_non_sha_pins_are_not_counted_as_callers(tmp_path: Path, ref: str) -> None:
    payload = _us_caller()
    _validate_job(payload)["uses"] = WORKFLOW_USES + ref
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="found 0"):
        _parse(repo)


def _legacy_caller(pin: str, extra: str = "") -> str:
    return (
        "name: Repository Checks\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  validate:\n"
        f"    uses: {WORKFLOW_USES}{pin}\n"
        "    with:\n"
        f"      axiom-encode-ref: {ENCODE}\n"
        f"      axiom-rules-engine-ref: {ENGINE}\n"
        f"      axiom-corpus-ref: {CORPUS}\n"
        f"      rulespec-us-ref: {RULESPEC_US}\n" + extra
    )


@pytest.mark.parametrize("pin", LEGACY_PINS, ids=lambda sha: sha[:8])
def test_legacy_caller_defaults_come_from_the_workflow(
    tmp_path: Path, pin: str
) -> None:
    repo = _rules_repo(tmp_path, caller=_legacy_caller(pin), toolchain=None)

    caller = _parse(repo, CallerOverrides())

    assert caller.validate_roots == "statutes regulations policies"
    assert caller.run_generated_guard is True
    assert caller.guard_programs_root is False
    assert caller.run_pytest is True
    assert caller.run_money_atom_check is True
    assert caller.release_base_url == DEFAULT_RELEASE_BASE_URL
    assert caller.resolutions == ()
    assert dict(caller.inputs) == {
        "python-version": "3.14",
        "axiom-encode-ref": ENCODE,
        "axiom-rules-engine-ref": ENGINE,
        "axiom-corpus-ref": CORPUS,
        "rulespec-us-ref": RULESPEC_US,
        "corpus-release-base-url": DEFAULT_RELEASE_BASE_URL,
        "validate-roots": "statutes regulations policies",
        "run-pytest": True,
        "run-generated-guard": True,
        "guard-programs-root": False,
        "run-money-atom-check": True,
    }
    verify_dependency_inputs(caller)


@pytest.mark.parametrize(
    "extra",
    [
        "      validation-workers: 4\n",
        f"      corpus-release-registry-url: {REGISTRY_URL_EXPRESSION}\n",
        "      allow-retired-schema-prefreeze: true\n",
    ],
    ids=["validation-workers", "registry-url", "prefreeze"],
)
@pytest.mark.parametrize("pin", LEGACY_PINS, ids=lambda sha: sha[:8])
def test_legacy_pins_reject_inputs_they_do_not_declare(
    tmp_path: Path, pin: str, extra: str
) -> None:
    repo = _rules_repo(tmp_path, caller=_legacy_caller(pin, extra), toolchain=None)
    name = extra.strip().split(":", 1)[0]

    with pytest.raises(
        ValueError, match=f"validate-rulespec@{pin[:8]} declares no input {name}"
    ):
        _parse(repo)


def test_legacy_caller_may_read_refs_from_the_resolver(tmp_path: Path) -> None:
    payload = _us_caller()
    _validate_job(payload)["uses"] = WORKFLOW_USES + PIN_615C1DF9
    inputs = _with(payload)
    for name in (
        "corpus-release-registry-url",
        "corpus-release-registry-anon-key",
        "validation-workers",
        "allow-retired-schema-prefreeze",
    ):
        del inputs[name]
    repo = _rules_repo(tmp_path, caller=payload)

    caller = _parse(repo, CallerOverrides())

    assert caller.workflow_sha == PIN_615C1DF9
    assert caller.refs == US_REFS
    assert len(caller.resolutions) == 4


def test_real_lane_callers_still_parse_with_the_same_values() -> None:
    de = parse_caller_workflow(FIXTURES / "de-caller.yml")
    dk = parse_caller_workflow(FIXTURES / "dk-caller.yml")

    assert (de.workflow_sha, dk.workflow_sha) == (PIN_615C1DF9, PIN_34BCFAB2)
    assert de.refs == {
        "encode": "164abed93f6df7f4bc52af533a9989159ec58113",
        "engine": "05eac9d2f89dabe5c6673176260762cef3a58f47",
        "corpus": "2dee8d9021ff54583b1314bd029a0e4eafa468aa",
        "rulespec_us": "0f291b367bf7e15555f9973112278c5cbf221653",
    }
    assert dk.refs == {
        "encode": "f4b952b7ba83d8382d0074b64b27ba6ea9a7637b",
        "engine": "05eac9d2f89dabe5c6673176260762cef3a58f47",
        "corpus": "f7be113cede06e0619011cdc231fd2fa744271e5",
        "rulespec_us": "0f291b367bf7e15555f9973112278c5cbf221653",
    }
    for caller, guard in ((de, True), (dk, False)):
        assert caller.validate_roots == "auto"
        assert caller.run_generated_guard is guard
        assert caller.guard_programs_root is False
        assert caller.release_base_url == DEFAULT_RELEASE_BASE_URL
        assert caller.run_pytest is True
        assert caller.run_money_atom_check is True
        assert caller.resolutions == ()
        assert caller.inputs["python-version"] == "3.14"
        verify_dependency_inputs(caller)


# ---------------------------------------------------------------------------
# verify_dependency_inputs mirrors "Validate immutable dependency inputs"
# ---------------------------------------------------------------------------


def _declared_inputs(sha: str, updates: dict[str, Any]) -> dict[str, Any]:
    declared = _pinned_workflow(sha).payload[True]["workflow_call"]["inputs"]
    inputs = {
        name: declaration["default"]
        for name, declaration in declared.items()
        if "default" in declaration
    }
    inputs |= {
        "axiom-encode-ref": ENCODE,
        "axiom-rules-engine-ref": ENGINE,
        "axiom-corpus-ref": CORPUS,
        "rulespec-us-ref": RULESPEC_US,
    }
    return inputs | updates


def _caller_from_inputs(sha: str, inputs: dict[str, Any]) -> CallerConfig:
    return CallerConfig(
        path=Path("caller.yml"),
        workflow_sha=sha,
        refs={
            dependency: inputs[name] for dependency, name in DEPENDENCY_INPUTS.items()
        },
        validate_roots=str(inputs["validate-roots"]),
        run_generated_guard=bool(inputs["run-generated-guard"]),
        guard_programs_root=bool(inputs["guard-programs-root"]),
        release_base_url=str(inputs["corpus-release-base-url"]),
        inputs=inputs,
    )


def _run_dependency_input_step(
    sha: str, inputs: dict[str, Any], tmp_path: Path
) -> tuple[int, str]:
    step = _pinned_workflow(sha).step(
        "validate", "Validate immutable dependency inputs"
    )
    environment = _step_environment(tmp_path)
    for name, expression in step["env"].items():
        match = INPUT_EXPRESSION.fullmatch(expression)
        assert match is not None, (name, expression)
        environment[name] = str(inputs[match.group("name")])
    result = _bash(step["run"], cwd=tmp_path, env=environment)
    return result.returncode, result.stderr.strip()


PYTHON_VERSION_ERROR = "python-version must be a bare major.minor like 3.14: "
REGISTRY = {
    "corpus-release-registry-url": "https://registry.example.test",
    "corpus-release-registry-anon-key": "anon",
}
DEPENDENCY_INPUT_CASES: dict[str, tuple[dict[str, Any], str | None]] = {
    "defaults": ({}, None),
    "python-3.13": ({"python-version": "3.13"}, None),
    "python-3.100": ({"python-version": "3.100"}, None),
    "python-patch": ({"python-version": "3.14.1"}, PYTHON_VERSION_ERROR + "3.14.1"),
    "python-major": ({"python-version": "3"}, PYTHON_VERSION_ERROR + "3"),
    "python-word": ({"python-version": "x"}, PYTHON_VERSION_ERROR + "x"),
    "python-2": ({"python-version": "2.7"}, PYTHON_VERSION_ERROR + "2.7"),
    "python-shell": (
        {"python-version": "3.14; true"},
        PYTHON_VERSION_ERROR + "3.14; true",
    ),
    "uppercase-ref": (
        {"axiom-rules-engine-ref": ENGINE.upper()},
        "Dependency inputs must be exact 40-character lowercase commit SHAs: "
        + ENGINE.upper(),
    ),
    "short-ref": (
        {"rulespec-us-ref": RULESPEC_US[:7]},
        "Dependency inputs must be exact 40-character lowercase commit SHAs: "
        + RULESPEC_US[:7],
    ),
    "ref-before-python": (
        {"axiom-corpus-ref": "x", "python-version": "3"},
        "Dependency inputs must be exact 40-character lowercase commit SHAs: x",
    ),
    "http-base": (
        {"corpus-release-base-url": "http://mirror.example.test"},
        "corpus-release-base-url must use HTTPS",
    ),
    "https-prefix-base": ({"corpus-release-base-url": "https://"}, None),
    "registry": (REGISTRY, None),
    "registry-skips-base-check": (
        REGISTRY | {"corpus-release-base-url": "http://mirror.example.test"},
        None,
    ),
    "registry-http": (
        REGISTRY | {"corpus-release-registry-url": "http://registry.example.test"},
        "corpus-release-registry-url must use HTTPS",
    ),
    "registry-uppercase-scheme": (
        REGISTRY | {"corpus-release-registry-url": "HTTPS://registry.example.test"},
        "corpus-release-registry-url must use HTTPS",
    ),
    "registry-without-key": (
        {"corpus-release-registry-url": "https://registry.example.test"},
        "corpus-release-registry-anon-key is required with corpus-release-registry-url",
    ),
    "key-without-registry": (
        {"corpus-release-registry-anon-key": "anon"},
        "corpus-release-registry-anon-key requires corpus-release-registry-url",
    ),
    "key-without-registry-before-base": (
        {
            "corpus-release-registry-anon-key": "anon",
            "corpus-release-base-url": "http://mirror.example.test",
        },
        "corpus-release-registry-anon-key requires corpus-release-registry-url",
    ),
    "python-before-registry": (
        {"python-version": "3", "corpus-release-registry-url": "http://x.test"},
        PYTHON_VERSION_ERROR + "3",
    ),
}
LEGACY_DEPENDENCY_INPUT_CASES = (
    "defaults",
    "python-patch",
    "python-word",
    "uppercase-ref",
    "http-base",
    "https-prefix-base",
)


def _assert_dependency_input_parity(sha: str, case: str, tmp_path: Path) -> None:
    updates, expected = DEPENDENCY_INPUT_CASES[case]
    inputs = _declared_inputs(sha, updates)
    step_code, step_error = _run_dependency_input_step(sha, inputs, tmp_path)
    caller = _caller_from_inputs(sha, inputs)

    if expected is None:
        assert step_code == 0, step_error
        verify_dependency_inputs(caller)
        return
    assert step_code != 0
    assert step_error == expected
    with pytest.raises(ValueError) as excinfo:
        verify_dependency_inputs(caller)
    assert str(excinfo.value) == step_error


@pytest.mark.parametrize("case", sorted(DEPENDENCY_INPUT_CASES))
@pytest.mark.parametrize("sha", EMBEDDED_PINS, ids=lambda sha: sha[:8])
def test_dependency_inputs_match_the_workflow_step(
    tmp_path: Path, sha: str, case: str
) -> None:
    _assert_dependency_input_parity(sha, case, tmp_path)


@pytest.mark.parametrize("case", LEGACY_DEPENDENCY_INPUT_CASES)
@pytest.mark.parametrize("sha", LEGACY_PINS, ids=lambda sha: sha[:8])
def test_legacy_dependency_inputs_match_the_workflow_step(
    tmp_path: Path, sha: str, case: str
) -> None:
    _assert_dependency_input_parity(sha, case, tmp_path)


# ---------------------------------------------------------------------------
# find_caller_workflow over .github/workflows
# ---------------------------------------------------------------------------

UNRELATED_WORKFLOW = (
    "name: Lint\non:\n  pull_request:\njobs:\n  lint:\n"
    "    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"
)


def test_no_workflows_directory_means_no_caller(tmp_path: Path) -> None:
    repo = _rules_repo(tmp_path, caller=None)

    with pytest.raises(ValueError, match="Expected exactly one .* found 0"):
        _parse(repo)


def test_unrelated_workflows_only_means_no_caller(tmp_path: Path) -> None:
    repo = _rules_repo(tmp_path, caller=UNRELATED_WORKFLOW, workflow_name="lint.yml")

    with pytest.raises(ValueError, match="found 0"):
        _parse(repo)


def test_two_caller_files_are_ambiguous(tmp_path: Path) -> None:
    repo = _rules_repo(tmp_path)
    shutil.copy(
        repo / ".github" / "workflows" / "repository-checks.yml",
        repo / ".github" / "workflows" / "nightly.yaml",
    )

    with pytest.raises(ValueError, match="found 2"):
        _parse(repo)


def test_two_caller_jobs_in_one_file_are_ambiguous(tmp_path: Path) -> None:
    payload = _us_caller()
    payload["jobs"]["validate-again"] = dict(_validate_job(payload))
    repo = _rules_repo(tmp_path, caller=payload)

    with pytest.raises(ValueError, match="found 2"):
        _parse(repo)


@pytest.mark.parametrize("extension", ["yml", "yaml"])
def test_exactly_one_caller_is_found_with_either_extension(
    tmp_path: Path, extension: str
) -> None:
    repo = _rules_repo(tmp_path, workflow_name=f"repository-checks.{extension}")
    workflows = repo / ".github" / "workflows"
    (workflows / "lint.yml").write_text(UNRELATED_WORKFLOW)
    # GitHub reads only top-level workflow files with these two extensions.
    (workflows / "archive").mkdir()
    (workflows / "archive" / "old.yml").write_text(US_CALLER_TEXT)
    (workflows / "disabled.yml.off").write_text(US_CALLER_TEXT)
    (workflows / "notes.txt").write_text(US_CALLER_TEXT)

    caller = _parse(repo)

    assert caller.path == workflows / f"repository-checks.{extension}"
    assert caller.refs == US_REFS


def test_malformed_workflow_file_fails_caller_discovery_closed(
    tmp_path: Path,
) -> None:
    # An unparseable file could be the caller itself, so discovery refuses to
    # guess which caller CI would run (a false FAIL, never a false PASS).
    repo = _rules_repo(tmp_path)
    broken = repo / ".github" / "workflows" / "broken.yml"
    broken.write_text("jobs: [unclosed\n")

    with pytest.raises(ValueError, match="Invalid caller workflow") as error:
        _parse(repo)
    assert str(broken) in str(error.value)
