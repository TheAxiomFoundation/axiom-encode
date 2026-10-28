"""Regressions for the independently reproduced cycle-10 parity review."""


def _c10_engine_binary(path, payload=b"verified debug"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _c10_cargo_artifact(path):
    import json

    return (
        json.dumps(
            {
                "reason": "compiler-artifact",
                "target": {"name": "axiom-rules-engine", "kind": ["bin"]},
                "executable": str(path),
            }
        )
        + "\n"
    )


def _c10_committed_engine(engine):
    import json
    import os
    import stat
    import subprocess

    engine.mkdir(parents=True, exist_ok=True)
    (engine / "src").mkdir(exist_ok=True)
    for name, content in {
        "Cargo.toml": '[package]\nname="axiom-rules-engine"\nversion="0.0.0"\n',
        "Cargo.lock": "# committed lock\n",
        "src/main.rs": "fn main() {}\n",
        ".gitignore": "target/\naxiom-rules-engine\n",
    }.items():
        path = engine / name
        if not path.exists():
            path.write_text(content)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(engine), *args],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    files = []
    for path in sorted(engine.rglob("*")):
        relative = path.relative_to(engine)
        if relative.parts[0] in {".git", "target", "axiom-rules-engine"}:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            files.append((relative.as_posix(), "120000", os.readlink(path).encode()))
        elif stat.S_ISREG(mode):
            files.append(
                (
                    relative.as_posix(),
                    "100755" if mode & 0o111 else "100644",
                    path.read_bytes(),
                )
            )
    git("init", "--quiet", "--template=")
    stream = bytearray()
    for index, (_, _, content) in enumerate(files, 1):
        stream.extend(f"blob\nmark :{index}\ndata {len(content)}\n".encode())
        stream.extend(content + b"\n")
    stream.extend(
        b"commit refs/heads/main\ncommitter CI <ci@example.com> 1700000000 +0000\n"
        b"data 10\nengine pin\n"
    )
    for index, (name, mode, _) in enumerate(files, 1):
        stream.extend(f"M {mode} :{index} {json.dumps(name)}\n".encode())
    stream.extend(b"\ndone\n")
    subprocess.run(
        ["git", "-C", str(engine), "fast-import", "--quiet"],
        input=bytes(stream),
        env=environment,
        capture_output=True,
        check=True,
    )
    git("checkout", "--quiet", "--force", "main")
    return git("rev-parse", "HEAD")


def _c10_mock_engine(
    monkeypatch,
    engine,
    *,
    artifact=None,
    status="",
    payload=b"verified debug",
    during_build=None,
):
    import subprocess
    from pathlib import Path

    from axiom_encode import ci_parity

    home = engine.parent / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    pin = _c10_committed_engine(engine)
    actual_run = subprocess.run

    @hides_environment
    def child(command, **kwargs):
        executable = Path(command[0]).name
        if executable == "git":
            return actual_run(command, **kwargs)
        if executable == "rustc":
            output = "rustc 1.94.1\nrelease: 1.94.1\nhost: c10-test-host\n"
        elif executable == "rustup":
            output = "stable-c10-test-host (default)\n"
        else:
            snapshot = Path(kwargs["cwd"])
            binary = _c10_engine_binary(
                snapshot / "target" / "debug" / "axiom-rules-engine", payload
            )
            if "Cargo.lock" in status:
                (snapshot / "Cargo.lock").write_text("# cargo rewrote snapshot lock\n")
            if during_build is not None:
                during_build(snapshot)
            output = _c10_cargo_artifact(binary if artifact is None else artifact)
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(ci_parity.subprocess, "run", child)
    return pin


def test_c10_engine_rejects_redirected_artifact_despite_stale_debug(
    tmp_path, monkeypatch
):
    import pytest

    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    stale = _c10_engine_binary(
        engine / "target" / "debug" / "axiom-rules-engine", b"stale"
    )
    redirected = _c10_engine_binary(
        engine / "target" / "c10-target" / "debug" / "axiom-rules-engine", b"fresh"
    )
    pin = _c10_mock_engine(monkeypatch, engine, artifact=redirected)
    with pytest.raises(ValueError, match="artifact.*expected"):
        ci_parity._build_workflow_engine(
            engine, tmp_path / "view", verified_pin=pin, offline=True
        )
    assert stale.read_bytes() == b"stale"


def test_c10_engine_view_freezes_verified_binary_bytes(tmp_path, monkeypatch):
    from axiom_encode import ci_parity
    from axiom_encode.engine_binding import EnginePin, resolve_pinned_engine_binary

    engine = tmp_path / "engine"
    binary = _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    pin = _c10_mock_engine(monkeypatch, engine)
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=pin, offline=True
    )
    bound = view / "target" / "debug" / "axiom-rules-engine"
    binary.write_bytes(b"different checkout build")
    assert not bound.is_symlink()
    assert bound.read_bytes() == b"verified debug"
    assert resolve_pinned_engine_binary(view, EnginePin(pin, engine)) == bound


def test_c10_engine_snapshot_lockfile_rewrite_keeps_receipt(tmp_path, monkeypatch):
    from axiom_encode import ci_parity
    from axiom_encode.engine_binding import engine_binding_receipt_path

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    pin = _c10_mock_engine(monkeypatch, engine, status=" M Cargo.lock\n")
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=pin, offline=True
    )
    assert engine_binding_receipt_path(
        view / "target" / "debug" / "axiom-rules-engine"
    ).exists()


def test_c10_engine_accepts_original_source_change_during_snapshot_build(
    tmp_path, monkeypatch
):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    pin = _c10_mock_engine(
        monkeypatch,
        engine,
        during_build=lambda _: (engine / "src" / "main.rs").write_text("edited\n"),
    )
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=pin, offline=True
    )
    assert (view / "target" / "debug" / "axiom-rules-engine").read_bytes() == (
        b"verified debug"
    )


def test_c10_nonfirst_freeze_failure_survives_in_run_ci_report(
    tmp_path, monkeypatch, capsys
):
    import json

    from axiom_encode import ci_parity
    from tests.test_ci_parity_runtime import PIN_0EFFA6A5, PIN_6F11BE26, _CiHarness

    # Both workflow pins set IS_FIRST_SHARD per matrix leg (6f11be26 L1146;
    # 0effa6a5 L793). A failure belongs to that gate, not dependency resolution.
    for pin in (PIN_0EFFA6A5, PIN_6F11BE26):
        with monkeypatch.context() as patch:
            home = tmp_path / pin[:8]
            home.mkdir()
            execute = ci_parity.execute_workflow_gates
            harness = _CiHarness(home, patch, workflow_sha=pin)
            patch.setattr(ci_parity, "execute_workflow_gates", execute)
            patch.setattr(ci_parity, "_link_workflow_checkouts", lambda *_: None)
            for key in ci_parity.SUPPORTED_WORKFLOW_PINS[pin].gates:
                if key == "retired_schema_freeze":
                    continue
                patch.setattr(
                    ci_parity,
                    f"_gate_{key}",
                    lambda run, spec: ci_parity.GateResult(
                        spec.key, spec.name, "PASS", output=f"completed {spec.key}"
                    ),
                )

            @hides_environment
            def worker(_script, *, cwd, argv, environment):
                if environment["IS_FIRST_SHARD"] == "true":
                    return 0, "first leg passed\n", ""
                return 1, "", "non-first worker failed\n"

            patch.setattr(ci_parity, "_run_embedded_python", worker)
            assert ci_parity.run_ci(harness.args()) == 1
            report = json.loads(capsys.readouterr().out)
            assert report["verdict"] == "FAIL"
            assert "resolution_error" not in report
            gates = {gate["gate"]: gate for gate in report["gates"]}
            assert gates["migration_authorization"]["status"] == "PASS"
            assert gates["retired_schema_freeze"]["status"] == "FAIL"
            assert gates["retired_schema_freeze"]["failures"] == [
                "[us-ca] non-first worker failed"
            ]
            assert gates["repository_tests"]["status"] == "PASS"
            assert "continued for diagnostics" in gates["repository_tests"]["note"]
            assert gates["compute_shards"]["status"] == "PASS"


def test_c10_freeze_failure_property():
    from pathlib import Path
    from types import SimpleNamespace
    from unittest import mock

    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    # For every first/non-first exit code and matrix size, the gate fails iff
    # one execution fails, and every failing leg is labelled in its failures.
    @settings(deadline=None, max_examples=30)
    @given(first=st.integers(0, 3), other=st.integers(0, 3), size=st.integers(1, 5))
    def check(first, other, size):
        matrix = tuple(f"shard-{i}" for i in range(size))
        run = SimpleNamespace(
            caller=SimpleNamespace(
                inputs={
                    "retired-schema-bootstrap-sha256": "",
                    "run-generated-guard": False,
                    "allow-retired-schema-prefreeze": False,
                }
            ),
            workflow=SimpleNamespace(python=lambda *_: "pass"),
            simulation=ci_parity.PullRequestSimulation(
                "a" * 40, "b" * 40, "main", "Org/rulespec-zz", 1, True
            ),
            temp=Path("/unused"),
            repo=Path("/unused"),
            plan=SimpleNamespace(matrix=matrix, first=matrix[0]),
            retired_skip_by_shard={},
            migration_authorized="",
        )
        spec = ci_parity.GateSpec("retired_schema_freeze", "Freeze", "inline", (), "")

        @hides_environment
        def worker(*_args, **kwargs):
            first_leg = kwargs["environment"]["IS_FIRST_SHARD"] == "true"
            return first if first_leg else other, "first" if first_leg else "other", ""

        with (
            mock.patch.object(ci_parity, "_run_embedded_python", worker),
            mock.patch.object(Path, "is_file", return_value=False),
        ):
            result = ci_parity._gate_retired_schema_freeze(run, spec)
        assert (result.status == "FAIL") == bool(first or (size > 1 and other))
        failing = ([matrix[0]] if first else []) + (list(matrix[1:]) if other else [])
        for shard in failing:
            assert any(
                line.startswith(f"[{shard}] ") if size > 1 else bool(line)
                for line in result.failures
            )
        assert len(result.failures) == len(failing)

    check()


def _c10_environment_run(tmp_path, *, mode="changed"):
    import os
    import shutil
    import subprocess
    from types import SimpleNamespace

    import pytest

    from axiom_encode import ci_parity

    if shutil.which("git") is None:
        pytest.skip("git unavailable")
    repo = tmp_path / "rulespec-c10"
    temp = tmp_path / "runner-temp"
    paths = {
        name: tmp_path / name for name in ("encode", "engine", "corpus", "rulespec_us")
    }
    for path in (repo, temp, *paths.values()):
        path.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--template=", str(repo)],
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
    )
    (paths["encode"] / "pyproject.toml").write_text('[project]\nname="axiom-encode"\n')
    return SimpleNamespace(
        repo=repo,
        temp=temp,
        paths=paths,
        mode=mode,
        offline=True,
        caller=SimpleNamespace(inputs={"python-version": "3.14"}),
        dependency_mismatches=[],
        notes=[],
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "Org/rulespec-c10", 7, True
        ),
    )


def test_c10_repository_git_config_matches_empty_runner(tmp_path, monkeypatch):
    import os
    import shutil
    import subprocess

    import pytest

    from axiom_encode import ci_parity

    git = shutil.which("git")
    if git is None:
        pytest.skip("git unavailable")
    run = _c10_environment_run(tmp_path)
    home = tmp_path / "developer-home"
    home.mkdir()
    (home / ".gitconfig").write_text("[commit]\n    gpgsign = true\n")
    monkeypatch.setenv("HOME", str(home))
    ambient = subprocess.run(
        [git, "config", "--get", "commit.gpgsign"],
        cwd=run.repo,
        env={"PATH": os.defpath, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert ambient.stdout.strip() == "true"
    environment = ci_parity._repository_test_environment(run)
    isolated = subprocess.run(
        [git, "config", "--get", "commit.gpgsign"],
        cwd=run.repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert isolated.returncode == 1
    assert isolated.stdout == ""
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["HOME"] == str(run.temp / "repository-test-home")


def test_c10_repository_environment_is_invariant_for_all_home_values():
    import os
    from pathlib import Path
    from types import SimpleNamespace
    from unittest import mock

    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    run = SimpleNamespace(
        paths={"engine": Path("/pinned/engine")},
        temp=Path("/runner/temp"),
        simulation=ci_parity.PullRequestSimulation(
            "a" * 40, "b" * 40, "main", "Org/rulespec-c10", 7, True
        ),
        repo=Path("/runner/rulespec-c10"),
        mode="changed",
    )

    @given(home=st.text(alphabet="abc123/._-", max_size=100))
    @settings(deadline=None)
    def invariant(home):
        with mock.patch.dict(os.environ, {"HOME": "/home/runner"}, clear=True):
            expected = ci_parity._repository_test_environment(run)
            os.environ["HOME"] = home
            changed = changed_names(
                expected, ci_parity._repository_test_environment(run)
            )
            assert changed == []

    invariant()


def test_c10_committed_axiom_symlink_cannot_create_outside_links(tmp_path):
    import os
    import subprocess

    import pytest

    from axiom_encode import ci_parity

    run = _c10_environment_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.repo / "_axiom").symlink_to(outside, target_is_directory=True)
    environment = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    subprocess.run(["git", "add", "_axiom"], cwd=run.repo, env=environment, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "committed symlink",
        ],
        cwd=run.repo,
        env=environment,
        check=True,
    )
    with ci_parity.committed_checkout(run.repo) as checkout:
        with pytest.raises(ValueError, match="_axiom"):
            ci_parity._link_workflow_checkouts(checkout, run.paths)
    assert list(outside.iterdir()) == []


def test_c10_tracked_axiom_directory_is_rejected_before_links(tmp_path):
    import subprocess

    import pytest

    from axiom_encode import ci_parity

    run = _c10_environment_run(tmp_path)
    internal = run.repo / "_axiom"
    internal.mkdir()
    (internal / "tracked.txt").write_text("PR-controlled checkout content\n")
    subprocess.run(["git", "add", "_axiom"], cwd=run.repo, check=True)
    with pytest.raises(ValueError, match="tracked"):
        ci_parity._link_workflow_checkouts(run.repo, run.paths)
    assert list(internal.iterdir()) == [internal / "tracked.txt"]


def test_e2e_links_leave_no_symlink_inside_the_rules_checkout(tmp_path, monkeypatch):
    """Attempt-10 e2e regression: _axiom links inside the checkout made the
    pinned encoder fail every engine gate ("active RuleSpec checkout contains a
    symlink"), a false FAIL on a pull request CI passed."""
    import os

    from axiom_encode import ci_parity

    run = _c10_environment_run(tmp_path, mode="changed")
    ci_parity._link_workflow_checkouts(run.repo, run.paths)
    monkeypatch.setattr(
        ci_parity.subprocess,
        "run",
        hides_environment(
            lambda command, **kwargs: ci_parity.subprocess.CompletedProcess(
                command, 0, "installed"
            )
        ),
    )
    monkeypatch.setattr(
        ci_parity,
        "_python_build_of",
        hides_environment(
            lambda python, environment: run.caller.inputs["python-version"]
        ),
    )
    (run.temp / "repository-test-venv" / "bin").mkdir(parents=True)
    (run.temp / "repository-test-venv" / "bin" / "python").write_text("")
    ci_parity._repository_test_python(run)
    links = [
        os.path.join(root, name)
        for root, dirs, files in os.walk(run.repo)
        if ".git" not in root.split(os.sep)
        for name in dirs + files
        if os.path.islink(os.path.join(root, name))
    ]
    assert links == []
    # The sibling links CI also makes stay outside the checkout.
    assert (run.repo.parent / "axiom-rules-engine").is_symlink()


def test_c10_repository_uv_ignores_actual_repository_config(tmp_path, monkeypatch):
    import shutil
    import subprocess
    import sys

    import pytest

    from axiom_encode import ci_parity

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv unavailable")
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    found = subprocess.run(
        [uv, "python", "find", "--no-config", f"{version}+gil"],
        capture_output=True,
        check=False,
    )
    if found.returncode:
        pytest.skip(f"no GIL CPython {version} installed for uv to find")
    run = _c10_environment_run(tmp_path, mode="full-toolchain-bump")
    # A bare major.minor, as the workflow requires; with --no-python-downloads
    # uv resolves it (plus "+gil") to an installed interpreter or fails.
    run.caller.inputs["python-version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    (run.repo / "uv.toml").write_text("this is invalid TOML = [\n")
    actual_run = subprocess.run
    commands = []

    venv_python = str(run.temp / "repository-test-venv" / "bin" / "python")

    @hides_environment
    def invoke(command, **kwargs):
        if command[0] == venv_python:
            # The post-creation identity probe runs the real venv interpreter.
            return actual_run(command, **kwargs)
        commands.append(command)
        if "venv" in command:
            # Use the existing interpreter and forbid downloads: this check
            # exercises real uv config discovery without touching the network.
            return actual_run(
                [
                    uv,
                    *command[1:],
                    "--no-python-downloads",
                    "--cache-dir",
                    str(run.temp / "uv-cache"),
                ],
                **kwargs,
            )
        return subprocess.CompletedProcess(command, 0, "installed")

    monkeypatch.setattr(ci_parity.subprocess, "run", invoke)
    interpreter = ci_parity._repository_test_python(run)
    assert not run.dependency_mismatches, run.notes
    assert interpreter == str(run.temp / "repository-test-venv" / "bin" / "python")
    assert all("--no-config" in command for command in commands)


def test_c10_engine_records_compiler_and_lock_rewrite_notes(tmp_path, monkeypatch):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    pin = _c10_mock_engine(monkeypatch, engine, status=" M Cargo.lock\n")
    notes = []
    ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=pin, offline=True, notes=notes
    )
    bound_note = next(note for note in notes if note.startswith("engine binary bound:"))
    assert "rustc release: 1.94.1; host: c10-test-host" in bound_note
    assert any("Cargo.lock rewritten" in note and "snapshot" in note for note in notes)


def test_c10_engine_rejects_unsupported_cargo_config(tmp_path, monkeypatch):
    import pytest

    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    binary = _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    config = tmp_path / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text('[build]\nrustc-wrapper="unexpected-wrapper"\n')
    _c10_mock_engine(monkeypatch, engine)
    with pytest.raises(ValueError, match="unsupported Cargo config"):
        ci_parity._build_workflow_engine(
            engine, tmp_path / "view", verified_pin=None, offline=True
        )
    assert binary.read_bytes() == b"verified debug"


import pytest as _c11_pytest  # noqa: E402

from tests.environment_assertions import changed_names, hides_environment


@_c11_pytest.mark.parametrize(
    "text",
    [
        "",
        "# comments only\n",
        '[term]\nverbose = true\ncolor = "never"\n',
        "[net]\nretry = 5\ngit-fetch-with-cli = true\n",
        "[http]\ntimeout = 60\n",
        '[alias]\nb = "build"\n',
        '[registries.internal]\nindex = "sparse+https://example.invalid/"\n',
        '[cargo-new]\nvcs = "none"\n',
    ],
)
def test_c11_engine_accepts_inert_ambient_cargo_config(tmp_path, monkeypatch, text):
    """Config that cannot change the build must not fail resolution (cycle 11)."""
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    config = tmp_path / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text(text)
    _c10_mock_engine(monkeypatch, engine)
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=None, offline=True
    )
    assert (view / "target" / "debug" / "axiom-rules-engine").read_bytes() == (
        b"verified debug"
    )


def test_c11_engine_accepts_empty_checkout_cargo_config(tmp_path, monkeypatch):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    config = engine / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text("# nothing configured\n")
    _c10_mock_engine(monkeypatch, engine)
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=None, offline=True
    )
    assert (view / "target" / "debug" / "axiom-rules-engine").is_file()


def test_c11_inert_sections_do_not_mask_a_build_changing_key(tmp_path, monkeypatch):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    config = tmp_path / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text('[term]\nverbose = true\n[env]\nRUSTC_BOOTSTRAP = "1"\n')
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="unsupported Cargo config.*env"):
        ci_parity._build_workflow_engine(
            engine, tmp_path / "view", verified_pin=None, offline=True
        )


def _c12_rules_checkout(tmp_path):
    rules = tmp_path / "rulespec-us"
    rules.mkdir()
    return rules


@_c11_pytest.mark.parametrize(
    "text",
    [
        '[build]\nrustc-wrapper = "/bin/false"\n',
        '[env]\nRUSTC_BOOTSTRAP = "1"\n',
        '[build]\ntarget = "x86_64-unknown-linux-musl"\n',
        "[profile.dev]\noverflow-checks = false\n",
    ],
)
def test_c12_rules_checkout_cargo_config_fails_closed(tmp_path, monkeypatch, text):
    """CI's engine build under _axiom reads the PR's .cargo config; ci cannot."""
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    (rules / ".cargo").mkdir()
    (rules / ".cargo" / "config.toml").write_text(text)
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="committed Cargo config"):
        ci_parity._build_workflow_engine(
            engine,
            tmp_path / "view",
            verified_pin=None,
            offline=True,
            rules_checkout=rules,
        )


@_c11_pytest.mark.parametrize("checkout", ["rules-empty", "engine"])
def test_c12_committed_inert_cargo_config_builds(tmp_path, monkeypatch, checkout):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    if checkout == "rules-empty":
        (rules / ".cargo").mkdir()
        (rules / ".cargo" / "config.toml").write_text("# nothing\n")
        root = engine
    else:
        root = engine
    (root / ".cargo").mkdir()
    (root / ".cargo" / "config").write_text('[term]\ncolor = "never"\n')
    _c10_mock_engine(monkeypatch, engine)
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=None, offline=True, rules_checkout=rules
    )
    assert (view / "target" / "debug" / "axiom-rules-engine").is_file()


@_c11_pytest.mark.parametrize("name", ["rust-toolchain", "rust-toolchain.toml"])
@_c11_pytest.mark.parametrize("where", ["rules", "above-snapshot"])
def test_c12_rust_toolchain_overrides_fail_closed(tmp_path, monkeypatch, name, where):
    from axiom_encode import ci_parity

    engine = tmp_path / "work" / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    target = rules if where == "rules" else engine.parent
    (target / name).write_text('[toolchain]\nchannel = "0.0.0-missing"\n')
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="Rust toolchain override"):
        ci_parity._build_workflow_engine(
            engine,
            engine.parent / "view",
            verified_pin=None,
            offline=True,
            rules_checkout=rules,
        )


@_c11_pytest.mark.parametrize(
    "text",
    ['[term.progress]\nwidth = "bogus"\n', '[term]\ncolor = "never"\n'],
)
def test_c13_rules_checkout_inert_cargo_config_fails_closed(
    tmp_path, monkeypatch, text
):
    """Only CI's cargo reads it, so even inert values would go unvalidated."""
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    (rules / ".cargo").mkdir()
    (rules / ".cargo" / "config.toml").write_text(text)
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="committed Cargo config.*term"):
        ci_parity._build_workflow_engine(
            engine,
            tmp_path / "view",
            verified_pin=None,
            offline=True,
            rules_checkout=rules,
        )


@_c11_pytest.mark.parametrize(
    "text", ["[workspace]\nmembers = []\n", '[package]\nname = "x"\n', ""]
)
def test_c13_rules_checkout_cargo_manifest_fails_closed(tmp_path, monkeypatch, text):
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    (rules / "Cargo.toml").write_text(text)
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="committed Cargo manifest"):
        ci_parity._build_workflow_engine(
            engine,
            tmp_path / "view",
            verified_pin=None,
            offline=True,
            rules_checkout=rules,
        )


@_c11_pytest.mark.parametrize(
    "text", ["[workspace]\nmembers = []\n", '[package]\nname = "other"\n']
)
def test_c13_ambient_cargo_manifest_above_snapshot_fails_closed(
    tmp_path, monkeypatch, text
):
    """Cargo parses every ancestor manifest; the runner's engine has none above."""
    from axiom_encode import ci_parity

    engine = tmp_path / "work" / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    (engine.parent / "Cargo.toml").write_text(text)
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="ambient Cargo manifest"):
        ci_parity._build_workflow_engine(
            engine, engine.parent / "view", verified_pin=None, offline=True
        )


@_c11_pytest.mark.parametrize(
    "ancestor", ["[workspace]\nmembers = []\n", '[package]\nname = "other"\n']
)
def test_c13_real_cargo_reads_ancestor_manifests(tmp_path, ancestor):
    """Differential: why an ancestor Cargo.toml must fail closed.

    The engine manifest declares no workspace, so real cargo parses each
    ancestor manifest; on the runner the rules checkout root is one.
    """
    import shutil
    import subprocess

    cargo = shutil.which("cargo")
    if cargo is None:
        _c11_pytest.skip("cargo is not installed")
    crate = tmp_path / "rules" / "_axiom" / "engine"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(
        '[package]\nname = "engine"\nversion = "0.1.0"\nedition = "2021"\n'
    )
    (crate / "src" / "main.rs").write_text("fn main() {}\n")

    def metadata() -> int:
        return subprocess.run(
            [cargo, "metadata", "--offline", "--no-deps", "--format-version", "1"],
            cwd=crate,
            capture_output=True,
            check=False,
        ).returncode

    assert metadata() == 0
    (tmp_path / "rules" / "Cargo.toml").write_text(ancestor)
    assert metadata() != 0


@_c11_pytest.mark.parametrize("checkout", ["rules", "engine"])
@_c11_pytest.mark.parametrize("link", ["config", "config.toml", ".cargo"])
def test_c14_symlinked_committed_cargo_config_fails_closed(
    tmp_path, monkeypatch, checkout, link
):
    """A dangling link in CI's layout resolves into _axiom (cycle 14)."""
    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    rules = _c12_rules_checkout(tmp_path)
    root = rules if checkout == "rules" else engine
    if link == ".cargo":
        (root / ".cargo").symlink_to("_axiom/missing")
    else:
        (root / ".cargo").mkdir()
        (root / ".cargo" / link).symlink_to("../_axiom/axiom-rules-engine/src/main.rs")
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="symlinked committed Cargo config"):
        ci_parity._build_workflow_engine(
            engine,
            tmp_path / "view",
            verified_pin=None,
            offline=True,
            rules_checkout=rules,
        )


def test_c14_dangling_ambient_cargo_config_fails_to_inspect(tmp_path, monkeypatch):
    from axiom_encode import ci_parity

    engine = tmp_path / "work" / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    (engine.parent / ".cargo").mkdir()
    (engine.parent / ".cargo" / "config.toml").symlink_to(tmp_path / "missing")
    _c10_mock_engine(monkeypatch, engine)
    with _c11_pytest.raises(ValueError, match="Cannot inspect Cargo config"):
        ci_parity._build_workflow_engine(
            engine, engine.parent / "view", verified_pin=None, offline=True
        )


def test_c20_dependency_tracking_a_symlink_fails_closed(tmp_path):
    """CI's encoder audits symlinks under _axiom/; ci refuses one up front."""
    import subprocess

    from axiom_encode import ci_parity

    clean = tmp_path / "clean"
    linked = tmp_path / "linked"
    for repo in (clean, linked):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "a.txt").write_text("a\n")
    (linked / "link").symlink_to("a.txt")
    for repo in (clean, linked):
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    ci_parity._reject_tracked_dependency_symlinks({"encode": clean})
    with _c11_pytest.raises(ValueError, match="tracks a symlink .*link"):
        ci_parity._reject_tracked_dependency_symlinks(
            {"encode": clean, "corpus": linked}
        )


def test_c10_engine_profile_and_rustflags_match_real_cargo(tmp_path, monkeypatch):
    """Differential: external config cannot change cargo's manifest/default build."""
    import os
    import shutil
    import subprocess

    import pytest

    from axiom_encode import ci_parity

    cargo = shutil.which("cargo")
    rustc = shutil.which("rustc")
    if cargo is None or rustc is None:
        pytest.skip("cargo and rustc are required for the offline differential")
    # Use installed toolchain executables so fresh HOME never asks rustup to
    # discover or download a default compiler.
    from pathlib import Path

    if Path(cargo).resolve().name == "rustup":
        rustup = shutil.which("rustup")
        if rustup is None:
            pytest.skip("installed Rust toolchain unavailable")
        tool = subprocess.run(
            [rustup, "which", "cargo"], text=True, capture_output=True, check=False
        )
        if tool.returncode:
            pytest.skip("installed Rust toolchain unavailable")
        cargo = tool.stdout.strip()
        monkeypatch.setenv(
            "PATH", str(Path(cargo).parent) + os.pathsep + os.environ["PATH"]
        )
    engine = tmp_path / "engine"
    (engine / "src").mkdir(parents=True)
    (engine / "Cargo.toml").write_text(
        '[package]\nname="axiom-rules-engine"\nversion="0.0.0"\nedition="2021"\n'
    )
    (engine / "src" / "main.rs").write_text(
        "fn main() { let n = std::hint::black_box(u8::MAX); "
        "let checked = std::panic::catch_unwind(|| n + 1).is_err(); "
        'println!("{} {} {}", cfg!(debug_assertions), cfg!(c10_probe), checked); }\n'
    )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CARGO_HOME", str(home / ".cargo"))
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("CARGO_", "RUST"))
    }
    environment["CARGO_TARGET_DIR"] = str(engine / "target")
    subprocess.run(
        [cargo, "build", "--verbose", "--offline"],
        cwd=engine,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    binary = engine / "target" / "debug" / "axiom-rules-engine"
    baseline = subprocess.run(
        [str(binary)], text=True, capture_output=True, check=True
    ).stdout
    config = tmp_path / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        '[build]\nrustflags=["--cfg", "c10_probe"]\n'
        "[profile.dev]\noverflow-checks=false\ndebug-assertions=false\n"
    )
    _c10_committed_engine(engine)
    actual_run = subprocess.run

    @hides_environment
    def subprocess_with_default_toolchain(command, **kwargs):
        if Path(command[0]).name == "rustup":
            return subprocess.CompletedProcess(
                command, 0, "stable-test-host (default)\n", ""
            )
        return actual_run(command, **kwargs)

    monkeypatch.setattr(ci_parity.subprocess, "run", subprocess_with_default_toolchain)
    notes = []
    view = ci_parity._build_workflow_engine(
        engine, tmp_path / "view", verified_pin=None, offline=True, notes=notes
    )
    observed = subprocess.run(
        [str(view / "target" / "debug" / "axiom-rules-engine")],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert observed == baseline == "true false true\n"
    assert any("rustc release:" in note and "host:" in note for note in notes)


def test_c10_engine_binding_snapshot_for_all_checkout_mutations():
    """For all build and later checkout bytes, the copied binding keeps build bytes."""
    import tempfile
    from pathlib import Path

    import pytest
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    @given(built=st.binary(max_size=64), replacement=st.binary(max_size=64))
    @settings(max_examples=30, deadline=None)
    def property_(built, replacement):
        with tempfile.TemporaryDirectory(prefix="axiom-c10-snapshot-") as directory:
            root = Path(directory)
            engine = root / "engine"
            binary = _c10_engine_binary(
                engine / "target" / "debug" / "axiom-rules-engine", b"stale"
            )
            with pytest.MonkeyPatch.context() as patch:
                _c10_mock_engine(patch, engine, payload=built)
                view = ci_parity._build_workflow_engine(
                    engine, root / "view", verified_pin=None, offline=True
                )
            binary.write_bytes(replacement)
            assert (view / "target" / "debug" / binary.name).read_bytes() == built

    property_()


def test_c10_engine_artifact_binding_for_all_other_target_paths():
    """For every distinct cargo target, a stale default binary cannot be bound."""
    import tempfile
    from pathlib import Path

    import pytest
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    @given(
        target=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=20
        )
    )
    @settings(max_examples=30, deadline=None)
    def property_(target):
        with tempfile.TemporaryDirectory(prefix="axiom-c10-target-") as directory:
            root = Path(directory)
            engine = root / "engine"
            _c10_engine_binary(
                engine / "target" / "debug" / "axiom-rules-engine", b"stale"
            )
            artifact = _c10_engine_binary(
                engine / "target" / target / "debug" / "axiom-rules-engine"
            )
            with pytest.MonkeyPatch.context() as patch:
                _c10_mock_engine(patch, engine, artifact=artifact)
                with pytest.raises(ValueError, match="artifact.*expected"):
                    ci_parity._build_workflow_engine(
                        engine, root / "view", verified_pin=None, offline=True
                    )

    property_()


def test_c10_engine_rejects_checkout_cargo_config_instead_of_overriding_ci_input(
    tmp_path, monkeypatch
):
    import pytest

    from axiom_encode import ci_parity

    engine = tmp_path / "engine"
    _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")
    config = engine / ".cargo" / "config.toml"
    config.parent.mkdir()
    config.write_text("[profile.dev]\noverflow-checks=false\n")
    _c10_mock_engine(monkeypatch, engine)
    with pytest.raises(ValueError, match="committed Cargo config"):
        ci_parity._build_workflow_engine(
            engine, tmp_path / "view", verified_pin=None, offline=True
        )


def test_c10_all_axiom_symlink_targets_are_rejected_before_writes(tmp_path):
    import pytest
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    case_number = 0

    @given(
        target_name=st.from_regex(r"[a-z]{1,8}", fullmatch=True), exists=st.booleans()
    )
    @settings(deadline=None, max_examples=20)
    def invariant(target_name, exists):
        nonlocal case_number
        case_number += 1
        case = tmp_path / str(case_number)
        case.mkdir()
        run = _c10_environment_run(case)
        outside = case / f"outside-{target_name}"
        if exists:
            outside.mkdir()
        (run.repo / "_axiom").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="_axiom"):
            ci_parity._link_workflow_checkouts(run.repo, run.paths)
        assert (list(outside.iterdir()) if exists else []) == []
        assert not (run.repo.parent / "axiom-rules-engine").exists()

    invariant()


def test_c10_all_repository_uv_commands_disable_configuration(tmp_path):
    import subprocess
    from unittest import mock

    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity

    case_number = 0
    actual_run = subprocess.run

    @given(
        offline=st.booleans(),
        mode=st.sampled_from(["changed", "full-toolchain-bump"]),
        version=st.text(alphabet="0123456789.", min_size=1, max_size=8),
    )
    @settings(deadline=None, max_examples=20)
    def invariant(offline, mode, version):
        nonlocal case_number
        case_number += 1
        case = tmp_path / str(case_number)
        case.mkdir()
        run = _c10_environment_run(case, mode=mode)
        run.offline = offline
        run.caller.inputs["python-version"] = version
        (run.paths["encode"] / "pyproject.toml").write_text(
            '[project]\nname="axiom-encode"\ndependencies=["axiom-oracles @ '
            "git+https://github.com/TheAxiomFoundation/axiom-oracles@"
            + "a" * 40
            + '"]\n'
        )
        commands = []

        @hides_environment
        def invoke(command, **kwargs):
            if command[0] == "git":
                return actual_run(command, **kwargs)
            commands.append(command)
            if "venv" in command:
                interpreter = run.temp / "repository-test-venv" / "bin" / "python"
                interpreter.parent.mkdir(parents=True)
                interpreter.write_text("mock interpreter\n")
            return subprocess.CompletedProcess(command, 0, "installed")

        with (
            mock.patch.object(ci_parity.subprocess, "run", invoke),
            mock.patch.object(
                ci_parity,
                "_python_build_of",
                hides_environment(
                    lambda python, environment: run.caller.inputs["python-version"]
                ),
            ),
        ):
            ci_parity._repository_test_python(run)
        assert not run.dependency_mismatches
        assert len(commands) == (6 if mode == "changed" else 4)
        assert all(command.count("--no-config") == 1 for command in commands)
        assert all(("--offline" in command) == offline for command in commands)

    invariant()


def test_c10_engine_build_is_independent_of_all_original_checkout_changes():
    """For all original checkout edits during a build, the snapshot stays bound."""
    import tempfile
    from pathlib import Path

    import pytest
    from hypothesis import given, settings
    from hypothesis import strategies as st

    from axiom_encode import ci_parity
    from axiom_encode.engine_binding import engine_binding_receipt_path

    @given(
        changed=st.lists(
            st.sampled_from(["Cargo.lock", "Cargo.toml", "src/main.rs", "new.txt"]),
            max_size=4,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def property_(changed):
        with tempfile.TemporaryDirectory(prefix="axiom-c10-lock-") as directory:
            root = Path(directory)
            engine = root / "engine"
            _c10_engine_binary(engine / "target" / "debug" / "axiom-rules-engine")

            def mutate_original(_snapshot):
                for name in changed:
                    (engine / name).write_text(
                        "original checkout edited during build\n"
                    )

            with pytest.MonkeyPatch.context() as patch:
                pin = _c10_mock_engine(patch, engine, during_build=mutate_original)
                view = ci_parity._build_workflow_engine(
                    engine, root / "view", verified_pin=pin, offline=True
                )
            bound = view / "target" / "debug" / "axiom-rules-engine"
            assert bound.read_bytes() == b"verified debug"
            assert engine_binding_receipt_path(bound).exists()

    property_()
