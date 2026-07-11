"""Security contract tests for sealed PolicyEngine oracle runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode.harness import policyengine_runtime as runtime_module
from axiom_encode.harness.policyengine_runtime import (
    POLICYENGINE_RUNTIME_PIN_SCHEMA,
    POLICYENGINE_RUNTIME_SCHEMA,
    PolicyEngineRuntime,
    PolicyEngineRuntimeError,
    _canonical_official_files,
    _official_https_remote,
    _OfficialFile,
    _OfficialTree,
    _probe_runtime,
    policyengine_subprocess_environment,
    rulespec_country_from_root,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_distribution_metadata(
    site_packages: Path,
    *,
    distribution: str,
    version: str,
) -> None:
    metadata_dir = (
        site_packages / f"{distribution.replace('-', '_')}-{version}.dist-info"
    )
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n"
    )


def _runtime_repo(
    tmp_path: Path,
    *,
    country: str = "us",
    origin: str = "https://example.invalid/caller-controlled.git",
    country_version: str = "1.2.3",
    core_version: str = "3.4.5",
) -> Path:
    root = tmp_path / f"policyengine-{country}"
    root.mkdir()
    module_name = f"policyengine_{country}"
    module_root = root / module_name
    module_root.mkdir()
    (module_root / "__init__.py").write_text("class Simulation:\n    pass\n")
    (root / ".gitignore").write_text(".venv/\n")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "policyengine-{country}"\nversion = "{country_version}"\n'
    )
    (root / "uv.lock").write_text(
        "version = 1\n"
        "[[package]]\n"
        f'name = "policyengine-{country}"\n'
        f'version = "{country_version}"\n'
        'source = { editable = "." }\n'
        "[[package]]\n"
        'name = "policyengine-core"\n'
        f'version = "{core_version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )

    venv = root / ".venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python_path = bin_dir / "python"
    shutil.copy2(sys._base_executable, python_path)
    python_path.chmod(0o755)
    stdlib = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = stdlib / "site-packages"
    site_packages.mkdir(parents=True)
    (stdlib / "os.py").write_text("# sealed fixture stdlib\n")
    core_root = site_packages / "policyengine_core"
    core_root.mkdir()
    (core_root / "__init__.py").write_text("CORE = True\n")
    _write_distribution_metadata(
        site_packages,
        distribution=f"policyengine-{country}",
        version=country_version,
    )
    _write_distribution_metadata(
        site_packages,
        distribution="policyengine-core",
        version=core_version,
    )

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "user.name", "Axiom tests")
    _git(root, "remote", "add", "origin", origin)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    _git(root, "checkout", "--detach", "-q")
    return root


def _rulespec_root(tmp_path: Path, *, country: str, commit: str) -> Path:
    checkout = tmp_path / f"rulespec-{country}"
    jurisdiction = checkout / country
    jurisdiction.mkdir(parents=True)
    (jurisdiction / ".gitkeep").write_text("")
    pin_path = checkout / ".axiom" / "policyengine-runtime.toml"
    pin_path.parent.mkdir()
    pin_path.write_text(
        "[policyengine_runtime]\n"
        f'schema = "{POLICYENGINE_RUNTIME_PIN_SCHEMA}"\n'
        f'git_commit = "{commit}"\n'
    )
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "tests@example.invalid")
    _git(checkout, "config", "user.name", "Axiom tests")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "runtime pin")
    return jurisdiction


def _rewrite_rulespec_pin(jurisdiction: Path, commit: str) -> None:
    checkout = jurisdiction.parent
    pin = checkout / ".axiom" / "policyengine-runtime.toml"
    pin.write_text(
        "[policyengine_runtime]\n"
        f'schema = "{POLICYENGINE_RUNTIME_PIN_SCHEMA}"\n'
        f'git_commit = "{commit}"\n'
    )
    _git(checkout, "add", ".axiom/policyengine-runtime.toml")
    _git(checkout, "commit", "-qm", "update runtime pin")


def _official_tree(root: Path, commit: str) -> _OfficialTree:
    files: list[_OfficialFile] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if relative.parts[0] in {".git", ".venv"} or not path.is_file():
            continue
        raw = path.read_bytes()
        files.append(
            _OfficialFile(
                path=relative.as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
                size=len(raw),
                executable=bool(path.stat().st_mode & 0o111),
            )
        )
    ordered = tuple(files)
    digest, byte_count = _canonical_official_files(ordered)
    return _OfficialTree(
        commit=commit,
        files=ordered,
        content_sha256=digest,
        byte_count=byte_count,
    )


def _fake_probe(
    root: Path,
    country: str,
    python_path: Path,
    site_packages: Path,
) -> dict[str, object]:
    venv = root / ".venv"
    stdlib = site_packages.parent
    initial = [str(stdlib), str(stdlib / "lib-dynload")]
    return {
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        ),
        "python_implementation": "cpython",
        "python_executable": str(python_path),
        "python_prefix": str(venv),
        "python_base_prefix": str(venv),
        "python_exec_prefix": str(venv),
        "python_base_exec_prefix": str(venv),
        "initial_sys_path": initial,
        "effective_sys_path": [str(root), str(site_packages), *initial],
        "isolated": 1,
        "no_site": 1,
        "packages": {
            f"policyengine-{country}": {
                "distribution": f"policyengine-{country}",
                "version": "1.2.3",
                "module_origin": str(root / f"policyengine_{country}" / "__init__.py"),
                "metadata_root": str(site_packages),
            },
            "policyengine-core": {
                "distribution": "policyengine-core",
                "version": "3.4.5",
                "module_origin": str(
                    site_packages / "policyengine_core" / "__init__.py"
                ),
                "metadata_root": str(site_packages),
            },
        },
    }


def _install_fixture_trust(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    allowed_commit: str,
) -> _OfficialTree:
    official = _official_tree(root, allowed_commit)
    monkeypatch.setattr(runtime_module, "_require_unprivileged_process", lambda: None)
    monkeypatch.setattr(
        runtime_module, "_require_protected_ancestor_chain", lambda _root: None
    )
    monkeypatch.setattr(
        runtime_module,
        "_trusted_owner_uids",
        lambda: frozenset({0, os.getuid()}),
    )
    monkeypatch.setattr(runtime_module, "_caller_can_write", lambda _path: False)
    monkeypatch.setattr(runtime_module, "_probe_runtime", _fake_probe)

    def fetch(country: str, observed_commit: str) -> _OfficialTree:
        assert _official_https_remote(country) == (
            f"https://github.com/PolicyEngine/policyengine-{country}.git"
        )
        if observed_commit != allowed_commit:
            raise PolicyEngineRuntimeError(
                "official HTTPS repository does not contain trusted commit"
            )
        return official

    monkeypatch.setattr(runtime_module, "_fetch_official_tree", fetch)
    return official


def _admit_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PolicyEngineRuntime, Path, Path]:
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)
    runtime = PolicyEngineRuntime.for_rulespec_root(
        root,
        policy_repo_root=jurisdiction,
    )
    return runtime, root, jurisdiction


def test_runtime_loads_deterministically_from_committed_pin_and_official_tree(
    tmp_path,
    monkeypatch,
):
    runtime, root, jurisdiction = _admit_fixture(tmp_path, monkeypatch)
    second = PolicyEngineRuntime.for_rulespec_root(
        root,
        policy_repo_root=jurisdiction,
    )

    assert runtime.identity == second.identity
    assert runtime.identity_sha256 == second.identity_sha256
    assert runtime.identity["schema"] == POLICYENGINE_RUNTIME_SCHEMA
    assert runtime.identity["official_repository_url"] == (
        "https://github.com/PolicyEngine/policyengine-us.git"
    )
    assert runtime.identity["trusted_git_commit"] == _git(root, "rev-parse", "HEAD")
    assert runtime.identity["locked_versions"] == {
        "policyengine-core": "3.4.5",
        "policyengine-us": "1.2.3",
    }
    assert "package_tree_sha256" not in json.dumps(runtime.identity)


def test_arbitrary_local_commit_with_official_config_url_is_rejected(
    tmp_path,
    monkeypatch,
):
    root = _runtime_repo(
        tmp_path,
        origin="https://github.com/PolicyEngine/policyengine-us.git",
    )
    approved = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=approved)
    _install_fixture_trust(monkeypatch, root, allowed_commit=approved)

    (root / "policyengine_us" / "attacker.py").write_text("VALUE = 1\n")
    _git(root, "add", "policyengine_us/attacker.py")
    _git(root, "commit", "-qm", "arbitrary local commit")
    arbitrary = _git(root, "rev-parse", "HEAD")
    _rewrite_rulespec_pin(jurisdiction, arbitrary)

    with pytest.raises(PolicyEngineRuntimeError, match="official HTTPS repository"):
        PolicyEngineRuntime.for_rulespec_root(
            root,
            policy_repo_root=jurisdiction,
        )


def test_pin_schema_rejects_caller_supplied_repository_url(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    checkout = jurisdiction.parent
    pin = checkout / ".axiom" / "policyengine-runtime.toml"
    pin.write_text(
        "[policyengine_runtime]\n"
        f'schema = "{POLICYENGINE_RUNTIME_PIN_SCHEMA}"\n'
        f'git_commit = "{commit}"\n'
        'repository = "https://github.com/PolicyEngine/policyengine-us.git"\n'
    )
    _git(checkout, "add", ".axiom/policyengine-runtime.toml")
    _git(checkout, "commit", "-qm", "invalid caller URL")
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="exactly: git_commit, schema"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_pin_must_be_committed_and_unchanged(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    pin = jurisdiction.parent / ".axiom" / "policyengine-runtime.toml"
    pin.write_text(pin.read_text() + "\n")
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="committed and unchanged"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_post_admission_pth_is_rejected(tmp_path, monkeypatch):
    runtime, root, _jurisdiction = _admit_fixture(tmp_path, monkeypatch)
    site_packages = runtime.site_packages_path
    (site_packages / "late-injection.pth").write_text("/attacker\n")

    with pytest.raises(PolicyEngineRuntimeError, match="forbidden Python startup"):
        runtime.assert_unchanged()


@pytest.mark.parametrize("target", ["dependency", "stdlib"])
def test_dependency_and_stdlib_mutation_are_parent_detected(
    tmp_path,
    monkeypatch,
    target,
):
    runtime, root, _jurisdiction = _admit_fixture(tmp_path, monkeypatch)
    if target == "dependency":
        path = runtime.site_packages_path / "policyengine_core" / "__init__.py"
    else:
        path = Path(runtime.identity["stdlib_root"]) / "os.py"
    path.write_text(path.read_text() + "# mutation\n")

    with pytest.raises(PolicyEngineRuntimeError, match="identity changed"):
        runtime.assert_unchanged()


def test_runtime_rejects_symlink_component(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    core = next((root / ".venv").glob("lib/python*/site-packages/policyengine_core"))
    shutil.rmtree(core)
    core.symlink_to(root / "policyengine_us", target_is_directory=True)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="contains symlink"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_runtime_rejects_writable_component(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    dependency = next(
        (root / ".venv").glob("lib/python*/site-packages/policyengine_core/__init__.py")
    )
    dependency.chmod(0o666)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="group- or other-writable"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


@pytest.mark.parametrize(
    "startup_name",
    ["sitecustomize.py", "usercustomize.py", "pyvenv.cfg"],
)
def test_runtime_rejects_startup_customization_components(
    tmp_path,
    monkeypatch,
    startup_name,
):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    (root / ".venv" / startup_name).write_text("# forbidden\n")
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="forbidden Python startup"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_oracle_command_hard_codes_isolated_no_site_bootstrap(tmp_path, monkeypatch):
    runtime, _root, _jurisdiction = _admit_fixture(tmp_path, monkeypatch)
    command = runtime.oracle_command("print('ok')")

    assert command[1:4] == ["-I", "-S", "-B"]
    assert command[4] == "-c"
    assert command[-1] == "print('ok')"
    assert str(runtime.root) in command
    assert str(runtime.site_packages_path) in command


def test_probe_itself_launches_with_i_and_s_and_scrubs_environment(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "probe-root"
    root.mkdir()
    (root / "policyengine_us").mkdir()
    (root / "policyengine_us" / "__init__.py").write_text("class Simulation: pass\n")
    site_packages = root / "site-packages"
    (site_packages / "policyengine_core").mkdir(parents=True)
    (site_packages / "policyengine_core" / "__init__.py").write_text("CORE=True\n")
    _write_distribution_metadata(
        site_packages, distribution="policyengine-us", version="1.2.3"
    )
    _write_distribution_metadata(
        site_packages, distribution="policyengine-core", version="3.4.5"
    )
    observed: dict[str, object] = {}
    real_run = subprocess.run

    def capture_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs.get("env")
        return real_run(command, **kwargs)

    monkeypatch.setenv("PYTHONPATH", "/attacker/modules")
    monkeypatch.setenv("TOKEN", "secret")
    monkeypatch.setattr(runtime_module.subprocess, "run", capture_run)

    payload = _probe_runtime(root, "us", Path(sys.executable), site_packages)

    command = observed["command"]
    assert isinstance(command, list)
    assert command[1:4] == ["-I", "-S", "-B"]
    assert observed["env"] == policyengine_subprocess_environment()
    assert "PYTHONPATH" not in observed["env"]
    assert "TOKEN" not in observed["env"]
    assert payload["isolated"] == 1
    assert payload["no_site"] == 1


@pytest.mark.parametrize("root_name", ["rulespec-us/us", "rulespec-us/us-co"])
def test_rulespec_country_is_derived_only_from_canonical_active_root(
    tmp_path,
    root_name,
):
    root = tmp_path / root_name
    root.mkdir(parents=True)
    assert rulespec_country_from_root(root) == "us"


@pytest.mark.parametrize(
    "root_name",
    ["rulespec-us", "rulespec-us-co", "workspace/us-co"],
)
def test_rulespec_country_rejects_checkout_flat_and_workspace_roots(
    tmp_path,
    root_name,
):
    root = tmp_path / root_name
    root.mkdir(parents=True)
    with pytest.raises(PolicyEngineRuntimeError, match="exact canonical active"):
        rulespec_country_from_root(root)


def test_runtime_identity_digest_is_canonical_json_sha256(tmp_path, monkeypatch):
    runtime, _root, _jurisdiction = _admit_fixture(tmp_path, monkeypatch)
    canonical = json.dumps(runtime.identity, sort_keys=True, separators=(",", ":"))
    assert runtime.identity_sha256 == hashlib.sha256(canonical.encode()).hexdigest()


def test_policyengine_runtime_root_must_be_absolute(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PolicyEngineRuntimeError, match="explicit absolute path"):
        PolicyEngineRuntime.for_rulespec_root(
            Path("policyengine-us"),
            policy_repo_root=jurisdiction,
        )


def test_production_trust_has_no_current_user_or_environment_relaxation():
    assert runtime_module._trusted_owner_uids() == frozenset({0})
    assert not any(
        name.startswith("AXIOM") or "TRUST" in name
        for name in policyengine_subprocess_environment()
    )


def test_policyengine_subprocess_environment_is_fixed():
    assert policyengine_subprocess_environment() == {
        "PATH": "/nonexistent",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    assert not set(policyengine_subprocess_environment()) & set(os.environ) - {
        "PATH",
        "HOME",
        "LC_ALL",
        "LANG",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
    }


def test_writable_mode_check_is_independent_of_acl_probe(tmp_path, monkeypatch):
    path = tmp_path / "dependency.py"
    path.write_text("VALUE = 1\n")
    path.chmod(0o666)
    monkeypatch.setattr(
        runtime_module,
        "_trusted_owner_uids",
        lambda: frozenset({path.stat().st_uid}),
    )
    monkeypatch.setattr(runtime_module, "_caller_can_write", lambda _path: False)

    with pytest.raises(PolicyEngineRuntimeError, match="group- or other-writable"):
        runtime_module._require_protected_metadata(
            path,
            path.stat(),
            label="runtime dependency",
        )


def test_python_executable_must_be_regular_not_symlink(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    python = root / ".venv" / "bin" / "python"
    python.unlink()
    python.symlink_to(sys.executable)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="contains symlink"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_python_executable_must_not_be_setid(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    python = root / ".venv" / "bin" / "python"
    python.chmod(python.stat().st_mode | stat.S_ISUID)
    _install_fixture_trust(monkeypatch, root, allowed_commit=commit)

    with pytest.raises(PolicyEngineRuntimeError, match="set-id or sticky"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)


def test_official_checkout_content_mode_is_enforced(tmp_path, monkeypatch):
    root = _runtime_repo(tmp_path)
    commit = _git(root, "rev-parse", "HEAD")
    jurisdiction = _rulespec_root(tmp_path, country="us", commit=commit)
    official = _install_fixture_trust(monkeypatch, root, allowed_commit=commit)
    target = root / "policyengine_us" / "__init__.py"
    target.chmod(target.stat().st_mode | stat.S_IXUSR)

    with pytest.raises(PolicyEngineRuntimeError, match="file does not match"):
        PolicyEngineRuntime.for_rulespec_root(root, policy_repo_root=jurisdiction)
    assert official.commit == commit
