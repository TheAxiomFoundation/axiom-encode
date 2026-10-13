"""Regression tests for the pinned workflow's execution environment."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity
from axiom_encode.ci_parity import GateSpec
from axiom_encode.engine_binding import EnginePin, resolve_pinned_engine_binary
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

PIN = "6f11be2655f79dd0a3b582db46525f58332ca120"
ENGINE_PIN = "a" * 40
ORACLES_PIN = "b" * 40


def _run(tmp_path: Path, *, mode: str = "changed", offline: bool = True):
    repo = tmp_path / "rulespec-zz"
    temp = tmp_path / "runner-temp"
    paths = {
        name: tmp_path / name for name in ("engine", "encode", "corpus", "rulespec_us")
    }
    for path in (repo, temp, *paths.values()):
        path.mkdir(exist_ok=True)
    subprocess.run(
        ["git", "init", "--quiet", "--template=", str(repo)],
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        },
        check=True,
    )
    (paths["encode"] / "pyproject.toml").write_text(
        '[project]\nname="axiom-encode"\nversion="0.1"\ndependencies=['
        '"axiom-oracles @ git+https://github.com/TheAxiomFoundation/'
        f'axiom-oracles@{ORACLES_PIN}"]\n'
    )
    return ci_parity.WorkflowRun(
        caller=ci_parity.CallerConfig(
            tmp_path / "caller.yml",
            PIN,
            {"encode": "c" * 40, "engine": ENGINE_PIN},
            "auto",
            True,
            False,
            inputs={"python-version": "3.14", "run-pytest": True},
        ),
        workflow=ci_parity.PinnedWorkflow(PIN),
        paths=paths,
        repo=repo,
        simulation=ci_parity.PullRequestSimulation(
            "d" * 40,
            "e" * 40,
            "main",
            "TheAxiomFoundation/rulespec-zz",
            12,
            True,
        ),
        plan=ci_parity.ShardPlan(
            ("zz",), "zz", "zz", "sources programs", '["zz"]', "test"
        ),
        validate_roots_input="auto",
        keyring=(),
        temp=temp,
        mode=mode,
        offline=offline,
    )


def _spec(key: str) -> GateSpec:
    return next(
        spec for spec in ci_parity.gate_registry_for_pin(PIN) if spec.key == key
    )


def _binary(path: Path, text: str = "fresh debug") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{text}'\n")
    path.chmod(0o755)
    return path


def _committed_engine_fixture(monkeypatch: pytest.MonkeyPatch, engine: Path) -> str:
    from tests.test_ci_parity_review_c10 import _c10_committed_engine

    home = engine.parent / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    return _c10_committed_engine(engine)


def _mock_engine_commands(monkeypatch: pytest.MonkeyPatch, cargo) -> None:
    actual_run = subprocess.run

    def child(command, **kwargs):
        executable = Path(command[0]).name
        if executable == "git":
            return actual_run(command, **kwargs)
        if executable == "rustup":
            return subprocess.CompletedProcess(
                command, 0, "stable-test-host (default)\n", ""
            )
        if executable == "rustc":
            return subprocess.CompletedProcess(
                command, 0, "rustc 1.94.1\nrelease: 1.94.1\nhost: test-host\n", ""
            )
        assert executable == "cargo", command
        return cargo(command, **kwargs)

    monkeypatch.setattr(ci_parity.subprocess, "run", child)


def _distribution(pin: str, root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        read_text=lambda _: json.dumps({"vcs_info": {"commit_id": pin}}),
        locate_file=lambda name: root / name,
    )


def test_installed_oracles_pin_rejects_shadowed_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = "1" * 40
    distribution = _distribution(pin, tmp_path / "site-packages")
    origin = tmp_path / "shadow" / "axiom_oracles" / "__init__.py"
    monkeypatch.setattr(
        ci_parity.importlib.metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(
        ci_parity.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=origin)
    )
    monkeypatch.setitem(
        sys.modules, "axiom_oracles", SimpleNamespace(__file__=str(origin))
    )

    assert ci_parity._installed_oracles_pin() is None


@given(pin=st.text("0123456789abcdef", min_size=40, max_size=40), shadow=st.booleans())
@settings(deadline=None)
def test_oracles_pin_is_trusted_only_for_the_distribution_module(
    pin: str, shadow: bool
) -> None:
    """For every exact pin, a module outside its distribution is untrusted."""

    distribution = _distribution(pin, Path("/pinned/site-packages"))
    origin = Path("/shadow" if shadow else "/pinned/site-packages") / (
        "axiom_oracles/__init__.py"
    )
    with (
        mock.patch.object(
            ci_parity.importlib.metadata, "distribution", return_value=distribution
        ),
        mock.patch.object(
            ci_parity.importlib.util,
            "find_spec",
            return_value=SimpleNamespace(origin=origin),
        ),
        mock.patch.dict(
            sys.modules, {"axiom_oracles": SimpleNamespace(__file__=str(origin))}
        ),
    ):
        assert ci_parity._installed_oracles_pin() == (None if shadow else pin)


def _classifier_archive(shadow: str) -> bytes:
    files = {
        "src/axiom_encode/__init__.py": "",
        "src/axiom_encode/cli.py": "print('classifier ran')\n",
    }
    if shadow != "none":
        files["src/axiom_oracles/__init__.py"] = ""
    if shadow == "metadata":
        files["src/axiom_oracles-0.0.dist-info/METADATA"] = (
            "Name: axiom-oracles\nVersion: 0.0\n"
        )
        files["src/axiom_oracles-0.0.dist-info/direct_url.json"] = json.dumps(
            {"vcs_info": {"commit_id": "2" * 40}}
        )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, text in files.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


@pytest.mark.parametrize("shadow", ["none", "module", "metadata"])
def test_pinned_classifier_reports_and_verifies_the_child_oracles_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shadow: str
) -> None:
    run_process = subprocess.run
    archive = _classifier_archive(shadow)

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if arguments[0] == "git":
            return subprocess.CompletedProcess(arguments, 0, archive)
        return run_process(arguments, **kwargs)

    monkeypatch.setattr(ci_parity.subprocess, "run", run)
    monkeypatch.setattr(ci_parity, "_oracles_pin_mismatch", lambda *_: None)

    code, stdout, stderr = ci_parity._run_pinned_process(
        tmp_path, "1" * 40, ["oracle-coverage"], stderr=subprocess.PIPE
    )

    assert "ci oracle dependency: axiom_oracles.__file__=" in stderr
    if shadow != "none":
        assert code == 1
        assert stdout == ""
        assert "expected" in stderr
    else:
        assert code == 0
        assert stdout == "classifier ran\n"


def test_full_oracle_coverage_uses_the_isolated_pinned_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = SimpleNamespace(
        mode="full-toolchain-bump",
        repo=tmp_path / "rulespec-us",
        paths={"encode": tmp_path / "encode"},
        caller=SimpleNamespace(refs={"encode": "1" * 40}),
    )
    calls: list[tuple[Path, str, list[str]]] = []

    def pinned(path: Path, pin: str, arguments: list[str]) -> tuple[int, str]:
        calls.append((path, pin, arguments))
        return 0, "pinned coverage\n"

    monkeypatch.setattr(ci_parity, "_run_pinned_cli", pinned)
    monkeypatch.setattr(ci_parity, "_oracles_pin_mismatch", lambda *_: None)
    monkeypatch.setattr(ci_parity, "_run_cli", lambda *_, **__: (1, "ambient\n"))
    spec = GateSpec("oracle_coverage", "Oracle coverage", "subcommand", (), "2502-2511")

    result = ci_parity._gate_oracle_coverage(run, spec)

    assert result.status == "PASS"
    assert result.output == "pinned coverage\n"
    assert calls == [(run.paths["encode"], run.caller.refs["encode"], result.command)]
    assert f"encode pin {run.caller.refs['encode']}" in result.note


@pytest.mark.parametrize("code", [0, 1])
@pytest.mark.parametrize("json_output", [False, True])
def test_pinned_cli_preserves_json_stdout_and_failure_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    json_output: bool,
) -> None:
    stdout = '{"items": []}\n'
    diagnostics = "ci oracle dependency: axiom_oracles.__file__=/pinned/oracles.py\n"
    seen = []

    def child(path, pin, arguments, *, stderr):
        seen.append(stderr)
        if stderr == subprocess.STDOUT:
            return code, diagnostics + stdout, ""
        return code, stdout, diagnostics

    monkeypatch.setattr(ci_parity, "_run_pinned_process", child)
    arguments = ["oracle-coverage", *(["--json"] if json_output else [])]

    result_code, output = ci_parity._run_pinned_cli(tmp_path, "1" * 40, arguments)

    assert result_code == code
    assert seen == [subprocess.PIPE if json_output else subprocess.STDOUT]
    if json_output:
        assert output == stdout + (diagnostics if code else "")
        if not code:
            assert json.loads(output) == {"items": []}
    else:
        assert output == diagnostics + stdout


def test_legacy_classifier_accepts_json_with_child_origin_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "rulespec-us"
    selection = ci_parity.Selection(
        mode="changed",
        rulespec_files=(repo / "us" / "statutes" / "a.yaml",),
        test_files=(),
    )

    def child(path, pin, arguments, *, stderr):
        stdout = '{"items": []}\n'
        diagnostic = "ci oracle dependency: axiom_oracles.__file__=/pinned/oracles.py\n"
        if stderr == subprocess.STDOUT:
            return 0, diagnostic + stdout, ""
        return 0, stdout, diagnostic

    monkeypatch.setattr(ci_parity, "_run_pinned_process", child)

    code, output = ci_parity._classifier_gate(repo, selection, tmp_path, "1" * 40)

    assert code == 0
    assert "passed for 0 output(s)" in output


@given(
    stdout=st.text(max_size=80),
    diagnostics=st.text(max_size=80),
    code=st.integers(min_value=0, max_value=255),
)
@settings(deadline=None)
def test_json_stdout_is_preserved_for_every_child_output(
    stdout: str, diagnostics: str, code: int
) -> None:
    """For all child outputs, successful JSON stdout excludes diagnostics."""

    with mock.patch.object(
        ci_parity, "_run_pinned_process", return_value=(code, stdout, diagnostics)
    ) as child:
        result = ci_parity._run_pinned_cli(Path("/encode"), "1" * 40, ["--json"])
    assert child.call_args.kwargs["stderr"] == subprocess.PIPE
    assert result == (code, stdout + (diagnostics if code else ""))


def test_changed_classifier_reports_origin_without_contaminating_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    run.selections["zz"] = ci_parity.ShardSelection(
        "zz", ("zz",), ("zz/statutes/a.yaml",), ()
    )
    diagnostic = "ci oracle dependency: axiom_oracles.__file__=/pinned/oracles.py\n"
    monkeypatch.setattr(
        ci_parity,
        "_run_pinned_process",
        lambda *_, **__: (0, '{"items": []}\n', diagnostic),
    )

    def filter_json(script, *, argv, **kwargs):
        assert json.loads(Path(argv[0]).read_text()) == {"items": []}
        return 0, "Changed PolicyEngine oracle coverage passed for 0 output(s).\n", ""

    monkeypatch.setattr(ci_parity, "_run_embedded_python", filter_json)

    result = ci_parity._gate_changed_oracle_coverage(
        run, _spec("changed_oracle_coverage")
    )

    assert result.status == "PASS"
    assert diagnostic in result.output
    assert "passed for 0 output(s)" in result.output


@pytest.mark.parametrize("offline", [False, True])
def test_engine_build_binds_fresh_debug_despite_stale_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    run = _run(tmp_path)
    engine = run.paths["engine"]
    _binary(engine / "target" / "release" / "axiom-rules-engine", "stale release")
    _binary(engine / "target" / "debug" / "axiom-rules-engine", "stale debug")
    pin = _committed_engine_fixture(monkeypatch, engine)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(tmp_path / "foreign-target"))
    monkeypatch.setenv("RUSTFLAGS", "--cfg foreign")
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    calls = []

    def cargo(command, **kwargs):
        calls.append((command, kwargs))
        snapshot = Path(kwargs["cwd"])
        _binary(snapshot / "target" / "debug" / "axiom-rules-engine")
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"name": "axiom-rules-engine", "kind": ["bin"]},
                    "executable": str(
                        snapshot / "target" / "debug" / "axiom-rules-engine"
                    ),
                }
            ),
        )

    _mock_engine_commands(monkeypatch, cargo)
    view = ci_parity._build_workflow_engine(
        engine, run.temp, verified_pin=pin, offline=offline
    )

    assert calls[0][0] == [
        "cargo",
        "build",
        "--verbose",
        "--message-format=json-render-diagnostics",
        "--config",
        "profile.dev.overflow-checks=true",
        "--config",
        "profile.dev.debug-assertions=true",
        "--config",
        "build.rustflags=[]",
        *(["--offline"] if offline else []),
    ]
    snapshot = Path(calls[0][1]["cwd"])
    assert snapshot != engine
    assert snapshot.is_relative_to(run.temp)
    assert calls[0][1]["env"]["CARGO_TARGET_DIR"] == str(snapshot / "target")
    assert "RUSTFLAGS" not in calls[0][1]["env"]
    assert "GIT_DIR" not in calls[0][1]["env"]
    pipeline = SimpleNamespace(axiom_rules_path=view, _log_event=lambda *_: None)
    bound = ValidatorPipeline._unverified_axiom_rules_binary(pipeline)
    assert bound.resolve() == view / "target" / "debug" / "axiom-rules-engine"
    assert "fresh debug" in bound.read_text()
    assert resolve_pinned_engine_binary(view, EnginePin(pin, run.repo)) == bound
    assert not (view / "target" / "release").exists()


@pytest.mark.parametrize(
    "failure", ["exit", "missing", "not-executable", "unavailable", "redirected"]
)
def test_engine_build_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    run = _run(tmp_path)
    pin = _committed_engine_fixture(monkeypatch, run.paths["engine"])

    def cargo(command, **kwargs):
        if failure == "unavailable":
            raise FileNotFoundError("cargo missing")
        snapshot = Path(kwargs["cwd"])
        artifact = snapshot / "target" / "debug" / "axiom-rules-engine"
        if failure == "redirected":
            artifact = snapshot / "target" / "foreign" / "axiom-rules-engine"
        if failure in ("not-executable", "redirected"):
            binary = _binary(artifact)
            if failure == "not-executable":
                binary.chmod(0o644)
        return subprocess.CompletedProcess(
            command,
            7 if failure == "exit" else 0,
            "x" * 3000 + "cargo tail"
            if failure == "exit"
            else json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"name": "axiom-rules-engine", "kind": ["bin"]},
                    "executable": str(artifact),
                }
            ),
        )

    _mock_engine_commands(monkeypatch, cargo)
    with pytest.raises(ValueError) as error:
        ci_parity._build_workflow_engine(
            run.paths["engine"], run.temp, verified_pin=pin, offline=True
        )
    if failure == "exit":
        assert str(error.value).endswith("cargo tail")
        assert len(str(error.value)) < 2300


@given(release=st.binary(max_size=30), bare=st.binary(max_size=30))
@settings(deadline=None)
def test_debug_binding_is_independent_of_every_shadow_candidate(
    release: bytes, bare: bytes
) -> None:
    """For all release/bare bytes, discovery binds only the built debug binary."""

    with tempfile.TemporaryDirectory(prefix="axiom-env-property-") as directory:
        root = Path(directory).resolve()
        engine = root / "engine"
        _binary(engine / "target" / "debug" / "axiom-rules-engine")
        for candidate, payload in (
            (engine / "target" / "release" / "axiom-rules-engine", release),
            (engine / "axiom-rules-engine", bare),
        ):
            candidate.parent.mkdir(exist_ok=True, parents=True)
            candidate.write_bytes(payload)

        def cargo(command, **kwargs):
            binary = _binary(
                Path(kwargs["cwd"]) / "target" / "debug" / "axiom-rules-engine"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "reason": "compiler-artifact",
                        "target": {"name": "axiom-rules-engine", "kind": ["bin"]},
                        "executable": str(binary),
                    }
                ),
            )

        with pytest.MonkeyPatch.context() as patch:
            _committed_engine_fixture(patch, engine)
            _mock_engine_commands(patch, cargo)
            view = ci_parity._build_workflow_engine(
                engine, root / "view", verified_pin=None, offline=True
            )
        pipeline = SimpleNamespace(axiom_rules_path=view, _log_event=lambda *_: None)
        assert (
            ValidatorPipeline._unverified_axiom_rules_binary(pipeline).resolve()
            == view / "target" / "debug" / "axiom-rules-engine"
        )


def test_engine_view_and_sibling_agree_with_shell_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh unavailable")
    run = _run(tmp_path)
    binary = _binary(run.paths["engine"] / "target" / "debug" / "axiom-rules-engine")
    _committed_engine_fixture(monkeypatch, run.paths["engine"])

    def cargo(command, **kwargs):
        artifact = _binary(Path(kwargs["cwd"]) / "target" / "debug" / binary.name)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "reason": "compiler-artifact",
                    "target": {"name": "axiom-rules-engine", "kind": ["bin"]},
                    "executable": str(artifact),
                }
            ),
        )

    with monkeypatch.context() as patch:
        _mock_engine_commands(patch, cargo)
        view = ci_parity._build_workflow_engine(
            run.paths["engine"], run.temp, verified_pin=None, offline=True
        )
    run.paths["engine"] = view
    ci_parity._link_workflow_checkouts(run.repo, run.paths)
    environment = ci_parity._repository_test_environment(run)
    result = subprocess.run(
        [shell, "-c", "command -v axiom-rules-engine; axiom-rules-engine"],
        cwd=run.repo,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    location, output = result.stdout.splitlines()
    assert Path(location).resolve() == view / "target" / "debug" / binary.name
    assert output == "fresh debug"
    assert (run.repo.parent / "axiom-rules-engine").resolve() == view
    # No link inside the rules checkout: the pinned encoder rejects symlinks
    # in an active RuleSpec checkout, and CI's gates prune _axiom.
    assert not (run.repo / "_axiom").exists()


def test_run_ci_builds_debug_engine_once_before_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.test_ci_parity_runtime import _CiHarness

    harness = _CiHarness(tmp_path, monkeypatch)
    builder = ci_parity._build_workflow_engine
    events = []
    rules_checkouts = []

    def build(engine, temp, *, verified_pin, offline, notes=None, rules_checkout=None):
        events.append(("build", engine, verified_pin, offline))
        # CI builds the engine under the rules checkout (cycle 12).
        rules_checkouts.append(rules_checkout)
        view = temp / "view"
        _binary(view / "target" / "debug" / "axiom-rules-engine")
        if notes is not None:
            notes.append("engine binary bound: mocked debug binary")
        return view

    def gates(run):
        events.append(("gates", run.paths["engine"]))
        assert rules_checkouts == [run.repo]
        assert run.offline
        assert (
            run.paths["engine"] / "target" / "debug" / "axiom-rules-engine"
        ).is_file()
        assert os.environ["PATH"].split(os.pathsep)[0] == str(
            run.paths["engine"] / "target" / "debug"
        )
        return harness.results

    # _CiHarness mocks dependency identity and Cargo; restore this seam with
    # a recorder so the orchestration is exercised without any network.
    assert builder is not None
    monkeypatch.setattr(ci_parity, "_build_workflow_engine", build)
    monkeypatch.setattr(ci_parity, "_link_workflow_checkouts", lambda *_: None)
    monkeypatch.setattr(ci_parity, "execute_workflow_gates", gates)
    assert ci_parity.run_ci(harness.args("--offline")) == 0
    assert [event[0] for event in events] == ["build", "gates"]
    assert events[0][1:] == (
        harness.paths["engine"],
        harness.caller.refs["engine"],
        True,
    )
    report = json.loads(capsys.readouterr().out)
    assert any("engine binary bound:" in note for note in report["notes"])
    assert not events[1][1].exists()


def _mock_uv(run, monkeypatch: pytest.MonkeyPatch, *, failure_at: int | None = None):
    calls = []
    actual_run = subprocess.run

    def uv(command, **kwargs):
        if command[0] == "git":
            return actual_run(command, **kwargs)
        assert command[0] == "uv", command
        calls.append((command, kwargs))
        if command[1] == "venv":
            _binary(Path(command[-1]) / "bin" / "python", "venv")
        failed = failure_at == len(calls)
        return subprocess.CompletedProcess(
            command, 2 if failed else 0, "offline cache miss" if failed else "installed"
        )

    monkeypatch.setattr(ci_parity.subprocess, "run", uv)
    # The venv's interpreter is identified after creation; mock it as the
    # requested setup-python build (release CPython with the GIL).
    monkeypatch.setattr(
        ci_parity,
        "_python_build_of",
        lambda python, environment: str(run.caller.inputs["python-version"]),
    )
    return calls


def test_repository_venv_with_a_non_ci_build_is_a_recorded_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uv's resolution is checked: a free-threaded venv is not setup-python's."""
    run = _run(tmp_path, mode="changed", offline=False)
    _mock_uv(run, monkeypatch)
    monkeypatch.setattr(
        ci_parity, "_python_build_of", lambda python, environment: "3.14t"
    )
    assert ci_parity._repository_test_python(run) == sys.executable
    [mismatch] = run.dependency_mismatches
    assert mismatch.name == "repository-test-environment"
    assert "Python 3.14t interpreter" in mismatch.head_sha


def test_repository_venv_probe_failure_is_a_recorded_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unidentifiable venv interpreter qualifies the verdict, never PASSes."""
    run = _run(tmp_path, mode="changed", offline=False)
    _mock_uv(run, monkeypatch)

    def unidentifiable(python, environment):
        raise ValueError(f"cannot identify interpreter {python}: garbage")

    monkeypatch.setattr(ci_parity, "_python_build_of", unidentifiable)
    assert ci_parity._repository_test_python(run) == sys.executable
    [mismatch] = run.dependency_mismatches
    assert mismatch.name == "repository-test-environment"
    assert "cannot identify interpreter" in mismatch.head_sha


@pytest.mark.parametrize(
    "output", ["", "garbage", "cpython 3 x 0 0", "cpython 3 14 0 0 extra"]
)
def test_python_build_probe_rejects_unparseable_output(
    tmp_path: Path, output: str
) -> None:
    fake = tmp_path / "python"
    fake.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
    fake.chmod(0o755)
    with pytest.raises(ValueError):
        ci_parity._python_build_of(fake, {"PATH": "/usr/bin:/bin"})


def test_python_build_probe_timeout_is_a_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def hang(command, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(command, 60)

    monkeypatch.setattr(ci_parity.subprocess, "run", hang)
    with pytest.raises(ValueError, match="did not answer"):
        ci_parity._python_build_of(tmp_path / "python", {})


def test_run_ci_engine_build_failure_is_resolution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.test_ci_parity_runtime import _CiHarness

    harness = _CiHarness(tmp_path, monkeypatch)

    def build(*_, **__):
        raise ValueError("Build axiom-rules-engine failed: cargo tail")

    monkeypatch.setattr(ci_parity, "_build_workflow_engine", build)
    assert ci_parity.run_ci(harness.args()) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL"
    assert report["resolution_error"].endswith("cargo tail")
    assert "execute_workflow_gates" not in harness.calls


def test_repository_environment_fallback_reaches_run_ci_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.test_ci_parity_runtime import _CiHarness

    harness = _CiHarness(tmp_path, monkeypatch)
    tests = harness.repo / "tests"
    tests.mkdir()
    (tests / "test_ci.py").write_text("def test_ci(): pass\n")
    (harness.paths["encode"] / "pyproject.toml").write_text(
        '[project]\nname="axiom-encode"\n'
    )

    def gates(run):
        run.temp = tmp_path / "pytest-temp"
        run.temp.mkdir()
        with (
            mock.patch.object(
                ci_parity,
                "_workflow_internal_directory",
                return_value=run.repo / "_axiom",
            ),
            mock.patch.object(
                ci_parity.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 2, "offline interpreter unavailable"
                ),
            ),
        ):
            return [ci_parity._gate_repository_tests(run, _spec("repository_tests"))]

    monkeypatch.setattr(ci_parity, "execute_workflow_gates", gates)
    monkeypatch.setattr(ci_parity, "_run_process", lambda *_, **__: (0, "1 passed\n"))
    assert ci_parity.run_ci(harness.args("--offline")) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "PASS-WITH-MISMATCHED-DEPS"
    assert report["dependency_mismatches"][0]["name"] == "repository-test-environment"
    assert (
        "offline interpreter unavailable"
        in report["dependency_mismatches"][0]["head_sha"]
    )


@pytest.mark.parametrize("mode", ["changed", "full-toolchain-bump"])
@pytest.mark.parametrize("offline", [False, True])
def test_repository_pytest_installs_only_ci_runtime_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, offline: bool
) -> None:
    run = _run(tmp_path, mode=mode, offline=offline)
    monkeypatch.setenv("UV_NO_DEPS", "true")
    monkeypatch.setenv("PIP_NO_DEPS", "true")
    monkeypatch.setenv("GIT_DIR", "/foreign/.git")
    monkeypatch.setenv("PYTHONPATH", "/foreign")
    calls = _mock_uv(run, monkeypatch)
    interpreter = ci_parity._repository_test_python(run)
    commands = [command for command, _ in calls]
    assert commands[0] == [
        "uv",
        "venv",
        "--python",
        "3.14+gil",
        "--no-config",
        *(["--offline"] if offline else []),
        str(run.temp / "repository-test-venv"),
    ]
    expected = [
        ["--upgrade", "pip"],
        ["pytest", "pyyaml"],
        ["-e", str(run.paths["encode"])],
    ]
    if mode == "changed":
        expected += [
            # CI checks the classifier out at the same encode pin under _axiom;
            # ci installs that same source without linking it into the checkout.
            ["-e", str(run.paths["encode"])],
            [
                "--force-reinstall",
                "--no-deps",
                f"axiom-oracles @ git+https://github.com/TheAxiomFoundation/axiom-oracles@{ORACLES_PIN}",
            ],
        ]
    prefix = [
        "uv",
        "pip",
        "install",
        "--python",
        interpreter,
        "--no-config",
        *(["--offline"] if offline else []),
    ]
    assert commands[1:] == [prefix + packages for packages in expected]
    assert all("--no-config" in command for command in commands)
    assert interpreter != sys.executable
    # The pinned encoder rejects any symlink in an active RuleSpec checkout.
    assert not (run.repo / "_axiom").exists()
    assert not run.dependency_mismatches
    assert all(
        not any(
            name.startswith(("PYTHON", "PYTEST_", "GIT_", "UV_", "PIP_"))
            for name in kwargs["env"]
        )
        for _, kwargs in calls
    )


@pytest.mark.parametrize("failure_at", range(1, 7))
def test_repository_pytest_provisioning_failure_qualifies_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int
) -> None:
    run = _run(tmp_path)
    calls = _mock_uv(run, monkeypatch, failure_at=failure_at)
    assert ci_parity._repository_test_python(run) == sys.executable
    assert len(calls) == failure_at
    assert len(run.dependency_mismatches) == 1
    mismatch = run.dependency_mismatches[0]
    assert mismatch.name == "repository-test-environment"
    assert "offline cache miss" in mismatch.head_sha
    assert (
        ci_parity.ci_verdict(True, run.dependency_mismatches)
        == "PASS-WITH-MISMATCHED-DEPS"
    )
    assert ci_parity.ci_verdict(False, run.dependency_mismatches) == "FAIL"


def test_repository_pytest_gate_uses_venv_and_ci_step_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _run(tmp_path)
    tests = run.repo / "tests"
    tests.mkdir()
    (tests / "test_ci.py").write_text("def test_ci(): pass\n")
    run.selections["zz"] = ci_parity.ShardSelection(
        "zz", ("zz",), ("zz/statutes/a.yaml",), ("zz/statutes/a.test.yaml",)
    )
    calls = _mock_uv(run, monkeypatch)
    seen = {}

    def pytest_process(command, cwd, *, environment):
        seen.update(command=command, cwd=cwd, env=environment)
        return 0, "1 passed\n"

    for name in (
        "AXIOM_RULESPEC_REPO_ROOTS",
        "DRAIN_BASE",
        "RULESPEC_SHADOW",
        "RULESPEC_VALIDATION_MODE",
        "GITHUB_SHA",
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
        "GIT_DIR",
    ):
        monkeypatch.setenv(name, "foreign")
    monkeypatch.setattr(ci_parity, "_run_process", pytest_process)
    result = ci_parity._gate_repository_tests(run, _spec("repository_tests"))
    assert result.status == "PASS"
    assert len(calls) == 6
    assert seen["command"] == [
        str(run.temp / "repository-test-venv" / "bin" / "python"),
        "-m",
        "pytest",
        "-q",
        "tests",
    ]
    assert seen["cwd"] == run.repo
    environment = seen["env"]
    assert environment["CI"] == environment["GITHUB_ACTIONS"] == "true"
    assert environment["GITHUB_SHA"] == run.simulation.github_sha
    assert environment["GITHUB_REF"] == "refs/pull/12/merge"
    assert environment["RULESPEC_VALIDATION_MODE"] == "changed"
    assert Path(environment["RULESPEC_FILE_LIST"]).read_text() == "zz/statutes/a.yaml\n"
    assert (
        Path(environment["RULESPEC_TEST_FILE_LIST"]).read_text()
        == "zz/statutes/a.test.yaml\n"
    )
    assert "DRAIN_BASE" not in environment and "RULESPEC_SHADOW" not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert not any(
        name.startswith(("AXIOM_", "PYTEST_", "PYTHON", "GIT_"))
        for name in environment
        if name not in {"GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM"}
    )


@given(
    ambient=st.dictionaries(
        st.from_regex(
            r"(?:AXIOM_|DRAIN_|RULESPEC_|PYTHON|PYTEST_|GIT_)[A-Z]{1,8}", fullmatch=True
        ),
        st.text(alphabet="abc123", max_size=10),
        max_size=20,
    )
)
@settings(deadline=None)
def test_repository_environment_never_inherits_ambient_overrides(
    ambient: dict[str, str],
) -> None:
    """For all ambient variables, pytest's environment is fixed by run inputs."""

    run = SimpleNamespace(
        paths={"engine": Path("/pinned/engine")},
        temp=Path("/runner/temp"),
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "Org/rulespec-zz", 7, True
        ),
        repo=Path("/runner/rulespec-zz"),
        mode="changed",
    )
    with mock.patch.dict(os.environ, {"HOME": "/home/runner"}, clear=True):
        expected = ci_parity._repository_test_environment(
            run, interpreter="/venv/bin/python"
        )
        os.environ.update(ambient)
        assert (
            ci_parity._repository_test_environment(run, interpreter="/venv/bin/python")
            == expected
        )


@given(
    cpu_count=st.integers(min_value=1, max_value=128),
    jobs=st.integers(min_value=1, max_value=32),
)
@settings(deadline=None)
def test_explicit_audit_workers_are_invariant_under_local_cpu_and_jobs(
    cpu_count: int, jobs: int
) -> None:
    """For every local CPU count and job cap, the audit still receives four."""

    seen = {}
    invocation = ci_parity.CliInvocation(
        ("validation-waivers", "audit"),
        cwd=Path("."),
        environment={ci_parity.WAIVER_AUDIT_WORKERS_ENV: "4"},
        supervised=True,
    )

    def worker(command, **kwargs):
        seen.update(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with mock.patch.object(ci_parity.subprocess, "run", worker):
        ci_parity._run_cli_isolated(invocation, (), max(1, cpu_count // jobs))
    assert seen["environment"][ci_parity.WAIVER_AUDIT_WORKERS_ENV] == "4"


def test_waiver_gate_uses_ci_workers_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axiom_encode import cli
    from tests.test_ci_parity_workflow_gates import _run as workflow_run
    from tests.test_ci_parity_workflow_gates import _waiver_repo

    repo, _ = _waiver_repo(tmp_path, {})
    run = workflow_run(repo, tmp_path, sha=PIN)
    seen = []

    def audit():
        seen.append(cli._waiver_audit_worker_count(1941))

    monkeypatch.setattr(cli, "main", audit)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 18)
    monkeypatch.setenv(ci_parity.WAIVER_AUDIT_WORKERS_ENV, "1")
    result = ci_parity._gate_validation_waivers(run, _spec("validation_waivers"))
    assert result.status == "PASS", result.output
    assert seen == [4] * len(run.plan.matrix)
    assert os.environ[ci_parity.WAIVER_AUDIT_WORKERS_ENV] == "1"


@pytest.mark.parametrize("zone,hour", [("Etc/GMT+12", 1), ("Etc/GMT-14", 23)])
@pytest.mark.parametrize("expiry_days,expected", [(0, 1), (1, 0), (90, 0), (91, 1)])
def test_waiver_expiry_and_horizon_follow_utc_under_extreme_zones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    zone: str,
    hour: int,
    expiry_days: int,
    expected: int,
) -> None:
    from axiom_encode import cli
    from axiom_encode.validation_waivers import load_validation_waivers
    from tests.test_ci_parity_runtime import _CiHarness

    if not hasattr(time, "tzset"):
        pytest.skip("tzset unavailable")
    harness = _CiHarness(tmp_path, monkeypatch)
    instant = datetime(2026, 12, 21, hour, tzinfo=timezone.utc)
    ledger = harness.repo / "known-validation-gaps.yaml"
    ledger.write_text(
        json.dumps(
            {
                "validate_failures": {
                    "us/statutes/a.yaml": {
                        "active": {
                            "fingerprint": "sha256:" + "a" * 64,
                            "owner": "@reviewer",
                            "issue": "https://github.com/TheAxiomFoundation/rulespec-us/issues/1",
                            "expires": str(
                                instant.date() + timedelta(days=expiry_days)
                            ),
                        }
                    }
                }
            }
        )
    )
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    monkeypatch.setattr(time, "time", lambda: instant.timestamp())
    assert date.today() != instant.date()

    def audit():
        load_validation_waivers(ledger, require_paths=False)

    def gates(run):
        code, output = ci_parity._run_cli(
            ["validation-waivers", "audit"], cwd=run.repo, supervised=True
        )
        assert date.today() == instant.date()
        return [ci_parity._result(_spec("validation_waivers"), code, output, [])]

    monkeypatch.setattr(cli, "main", audit)
    monkeypatch.setattr(ci_parity, "execute_workflow_gates", gates)
    try:
        assert ci_parity.run_ci(harness.args()) == expected
        assert os.environ["TZ"] == zone
        assert date.today() != instant.date()
        capsys.readouterr()
    finally:
        monkeypatch.undo()
        time.tzset()


def test_runner_locale_and_timezone_reach_every_execution_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from axiom_encode import cli
    from tests.test_ci_parity_runtime import _CiHarness

    harness = _CiHarness(tmp_path, monkeypatch)
    original = {name: os.environ.get(name) for name in ("TZ", "LANG", "LC_ALL")}
    monkeypatch.setenv("TZ", "Etc/GMT-14")
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("LC_ALL", "C")
    seen = []

    def probe():
        seen.append({name: os.environ[name] for name in ("TZ", "LANG", "LC_ALL")})

    def child(command, **kwargs):
        seen.append({name: kwargs["env"][name] for name in ("TZ", "LANG", "LC_ALL")})
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def gates(run):
        probe()
        ci_parity._run_cli(["validation-waivers", "audit"], supervised=True)
        with mock.patch.object(ci_parity.subprocess, "run", child):
            ci_parity._run_embedded_python("pass", cwd=run.repo)
            ci_parity._run_cli_isolated(
                ci_parity.CliInvocation(("validate",), cwd=run.repo), (), 1
            )

        def pinned_child(command, **kwargs):
            if command[0] == "git":
                return subprocess.CompletedProcess(
                    command, 0, _classifier_archive("none")
                )
            seen.append(
                {name: kwargs["env"][name] for name in ("TZ", "LANG", "LC_ALL")}
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            mock.patch.object(ci_parity.subprocess, "run", pinned_child),
            mock.patch.object(ci_parity, "_oracles_pin_mismatch", return_value=None),
            mock.patch.object(
                ci_parity, "_installed_oracles_pin", return_value=ORACLES_PIN
            ),
        ):
            ci_parity._run_pinned_process(
                run.paths["encode"],
                "c" * 40,
                ["oracle-coverage"],
                stderr=subprocess.PIPE,
            )
        seen.append(
            {
                name: ci_parity._repository_test_environment(run)[name]
                for name in ("TZ", "LANG", "LC_ALL")
            }
        )
        return harness.results

    monkeypatch.setattr(cli, "main", probe)
    monkeypatch.setattr(ci_parity, "execute_workflow_gates", gates)
    try:
        assert ci_parity.run_ci(harness.args()) == 0
        assert seen == [{"TZ": "UTC", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}] * 6
        assert os.environ["TZ"] == "Etc/GMT-14"
        assert os.environ["LANG"] == os.environ["LC_ALL"] == "C"
        capsys.readouterr()
    finally:
        monkeypatch.undo()
        time.tzset()
    assert {name: os.environ.get(name) for name in original} == original


@given(
    zone=st.sampled_from(["UTC", "Etc/GMT+12", "Etc/GMT-14"]),
    language=st.sampled_from(["C", "POSIX", "en_US.UTF-8"]),
    locale=st.sampled_from(["C", "POSIX", "de_DE.UTF-8"]),
)
@settings(deadline=None)
def test_runner_clock_and_locale_are_invariant_and_restored(
    zone: str, language: str, locale: str
) -> None:
    """For all ambient clock/locale choices, every run uses UTC and C.UTF-8."""

    if not hasattr(time, "tzset"):
        pytest.skip("tzset unavailable")
    ambient = {"TZ": zone, "LANG": language, "LC_ALL": locale, "PATH": os.defpath}
    seen = {}

    def resolve(*_):
        seen.update({name: os.environ[name] for name in ("TZ", "LANG", "LC_ALL")})
        assert time.localtime(1797814800).tm_gmtoff == 0
        raise ValueError("stop after inspecting the runner environment")

    try:
        with (
            mock.patch.dict(os.environ, ambient, clear=True),
            mock.patch.object(ci_parity, "resolve_commit", resolve),
            mock.patch("sys.stdout", io.StringIO()),
        ):
            time.tzset()
            assert (
                ci_parity.run_ci(
                    SimpleNamespace(
                        repo=Path("/unavailable"), base_ref="base", json=True
                    )
                )
                == 1
            )
            assert seen == {"TZ": "UTC", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
            assert dict(os.environ) == ambient
    finally:
        time.tzset()
