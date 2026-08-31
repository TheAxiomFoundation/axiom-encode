import json
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.provision_verification_supervisor import _signing_trust_roots_payload


def test_provision_replaces_base_runtime_site_packages(tmp_path: Path) -> None:
    system_git = Path("/usr/bin/git")
    git = system_git if system_git.is_file() else shutil.which("git")
    if git is None:
        raise AssertionError("Git is required for the provisioner test")
    git = Path(git).resolve()
    source_site_packages = (
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert (source_site_packages / "pip").is_dir()
    assert (source_site_packages / "README.txt").is_file()

    trusted_packages = tmp_path / "trusted-packages"
    intended_package = trusted_packages / "intended_package"
    intended_package.mkdir(parents=True)
    (intended_package / "__init__.py").write_text("TRUSTED = True\n")
    supervisor = tmp_path / "supervisor"
    supervisor.write_text("supervisor\n")
    destination = tmp_path / "provisioned"

    subprocess.run(
        [
            sys.executable,
            "scripts/provision_verification_supervisor.py",
            "--destination",
            str(destination),
            "--supervisor",
            str(supervisor),
            "--site-packages",
            str(trusted_packages),
            "--apply-root",
            "apply-root",
            "--eval-root",
            "eval-root",
            "--corpus-release-root",
            "corpus-release-root",
            "--git",
            str(git),
        ],
        check=True,
    )

    runtime = destination / "python"
    interpreter = runtime / Path(sys.executable).resolve().relative_to(
        Path(sys.base_prefix).resolve()
    )
    assert (destination / "axiom-encode").read_text().splitlines()[0] == (
        f"#!{interpreter} -I"
    )
    git_wrapper = interpreter.parent / "git"
    assert git_wrapper.read_text().splitlines()[0] == f"#!{interpreter} -I"
    assert not (destination / "git").exists()
    repository = tmp_path / "rulespec-us"
    subprocess.run([str(git), "init", "--quiet", str(repository)], check=True)
    top_level = subprocess.run(
        [str(git_wrapper), "-C", str(repository), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(runtime),
            "PATH": str(interpreter.parent),
        },
    )
    assert Path(top_level.stdout.strip()) == repository
    wrapper_text = git_wrapper.read_text(encoding="utf-8")
    assert "'-c', f'safe.directory={repository}'" in wrapper_text
    assert "f'--git-dir={repository}/.git'" in wrapper_text
    assert "f'--work-tree={repository}'" in wrapper_text
    assert "'-c', f'core.worktree={repository}'" in wrapper_text
    assert "'-c', 'commit.gpgSign=false'" in wrapper_text
    relative_repository = subprocess.run(
        [str(git_wrapper), "-C", "rulespec-us", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(runtime),
            "PATH": str(interpreter.parent),
        },
    )
    assert relative_repository.returncode != 0
    assert "requires an exact repository after -C" in relative_repository.stderr
    repository_alias = tmp_path / "rulespec-us-alias"
    repository_alias.symlink_to(repository, target_is_directory=True)
    symlinked_repository = subprocess.run(
        [
            str(git_wrapper),
            "-C",
            str(repository_alias),
            "rev-parse",
            "--show-toplevel",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(runtime),
            "PATH": str(interpreter.parent),
        },
    )
    assert symlinked_repository.returncode != 0
    assert "requires an exact repository after -C" in symlinked_repository.stderr

    tracked = repository / "tracked.txt"
    attributes = repository / ".gitattributes"
    manifests = repository / ".axiom" / "encoding-manifests"
    manifests.mkdir(parents=True)
    modified_manifest = manifests / "modified.json"
    deleted_manifest = manifests / "deleted.json"
    tracked.write_text("initial\n", encoding="utf-8")
    attributes.write_text("tracked.txt diff=evil\n", encoding="utf-8")
    modified_manifest.write_text("initial\n", encoding="utf-8")
    deleted_manifest.write_text("initial\n", encoding="utf-8")
    subprocess.run([str(git), "-C", str(repository), "add", "--", "."], check=True)
    subprocess.run(
        [
            str(git),
            "-C",
            str(repository),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        check=True,
    )
    marker = tmp_path / "tainted-git-executed"
    payload = tmp_path / "tainted-git-payload"
    payload.write_text(f"#!/bin/sh\nprintf tainted > {marker}\n", encoding="utf-8")
    payload.chmod(0o755)
    hook = repository / ".git" / "hooks" / "pre-commit"
    hook.write_text(payload.read_text(encoding="utf-8"), encoding="utf-8")
    hook.chmod(0o755)
    poisoned_worktree = tmp_path / "poisoned-worktree"
    poisoned_worktree.mkdir()
    for name, value in (
        ("commit.gpgSign", "true"),
        ("gpg.program", str(payload)),
        ("core.fsmonitor", str(payload)),
        ("core.hooksPath", str(hook.parent)),
        ("core.worktree", str(poisoned_worktree)),
        ("diff.evil.textconv", str(payload)),
    ):
        subprocess.run(
            [str(git), "-C", str(repository), "config", "--local", name, value],
            check=True,
        )
    tracked.write_text("checkpoint\n", encoding="utf-8")
    modified_manifest.write_text("changed manifest\n", encoding="utf-8")
    deleted_manifest.unlink()
    wrapper_env = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(runtime),
        "PATH": str(interpreter.parent),
    }
    subprocess.run(
        [str(git_wrapper), "-C", str(repository), "add", "--", "tracked.txt"],
        check=True,
        env=wrapper_env,
    )
    subprocess.run(
        [str(git_wrapper), "-C", str(repository), "diff", "--check"],
        check=True,
        env=wrapper_env,
    )
    parent = subprocess.check_output(
        [str(git_wrapper), "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        env=wrapper_env,
    ).strip()
    modified_paths = subprocess.check_output(
        [
            str(git_wrapper),
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "--diff-filter=AM",
            "-z",
            parent,
            "--",
            ".axiom/encoding-manifests",
        ],
        env=wrapper_env,
    )
    deleted_paths = subprocess.check_output(
        [
            str(git_wrapper),
            "-C",
            str(repository),
            "diff",
            "--name-only",
            "--diff-filter=D",
            "-z",
            parent,
            "--",
            ".axiom/encoding-manifests",
        ],
        env=wrapper_env,
    )
    assert modified_paths == b".axiom/encoding-manifests/modified.json\0"
    assert deleted_paths == b".axiom/encoding-manifests/deleted.json\0"
    for refused_options in (
        [
            "--name-only",
            "--diff-filter=AM",
            "-z",
            "HEAD",
            "--",
            ".axiom/encoding-manifests",
        ],
        ["--name-only", "--diff-filter=D", "-z", parent, "--", ".axiom"],
    ):
        refused = subprocess.run(
            [str(git_wrapper), "-C", str(repository), "diff", *refused_options],
            check=False,
            capture_output=True,
            env=wrapper_env,
        )
        assert refused.returncode != 0
        assert b"trusted git wrapper refused arguments for diff" in refused.stderr
    tree = subprocess.check_output(
        [str(git_wrapper), "-C", str(repository), "write-tree"],
        text=True,
        env=wrapper_env,
    ).strip()
    checkpoint = subprocess.check_output(
        [
            str(git_wrapper),
            "-C",
            str(repository),
            "commit-tree",
            tree,
            "-p",
            parent,
            "-m",
            "Axiom encoder checkpoint",
        ],
        text=True,
        env=wrapper_env,
    ).strip()
    subprocess.run(
        [
            str(git_wrapper),
            "-C",
            str(repository),
            "update-ref",
            "HEAD",
            checkpoint,
            parent,
        ],
        check=True,
        env=wrapper_env,
    )
    assert not marker.exists()
    assert (
        subprocess.run(
            [str(git), "-C", str(repository), "log", "-1", "--format=%s"],
            check=True,
            capture_output=True,
            text=True,
            env={
                **wrapper_env,
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.worktree",
                "GIT_CONFIG_VALUE_0": str(repository),
            },
        ).stdout.strip()
        == "Axiom encoder checkpoint"
    )
    assert json.loads((destination / "signing-trust-roots.json").read_text()) == {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": "apply-root",
        "eval_ed25519_public_key": "eval-root",
        "corpus_release_ed25519_public_key": "corpus-release-root",
    }

    provisioned_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert sorted(
        path.relative_to(provisioned_site_packages)
        for path in provisioned_site_packages.rglob("*")
    ) == [
        Path("intended_package"),
        Path("intended_package/__init__.py"),
    ]
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    forbidden_patterns = (
        "*.pth",
        "*.egg-link",
        "sitecustomize.py",
        "usercustomize.py",
        "pyvenv.cfg",
        "__editable__*",
    )
    assert not any(
        path.is_file()
        for pattern in forbidden_patterns
        for path in destination.rglob(pattern)
    )


def test_provision_writes_v3_corpus_release_keyring() -> None:
    assert _signing_trust_roots_payload(
        "apply",
        "eval",
        "current",
        ("retired-one", "retired-two"),
    ) == {
        "schema": "axiom-encode/signing-trust-roots/v3",
        "apply_ed25519_public_key": "apply",
        "eval_ed25519_public_key": "eval",
        "corpus_release_ed25519_public_keys": [
            "current",
            "retired-one",
            "retired-two",
        ],
    }
