import os
import subprocess
from pathlib import Path

import pytest

import scripts.copy_untrusted_rulespec_worktree as worktree_copy
from scripts.copy_untrusted_rulespec_worktree import copy_worktree, main


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)


def test_copy_worktree_replaces_payload_without_consulting_source_git(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rulespec-us-source"
    protected = tmp_path / "rulespec-us-protected"
    _git_init(source)
    _git_init(protected)
    protected_head = (protected / ".git" / "HEAD").read_bytes()

    rule = source / "us" / "policies" / "example.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    executable = source / "script"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    (protected / "stale.txt").write_text("remove me\n", encoding="utf-8")

    payload = tmp_path / "must-not-run"
    hook = source / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {payload}\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "config",
            "--local",
            "core.fsmonitor",
            str(hook),
        ],
        check=True,
    )

    owner = os.geteuid()
    result = copy_worktree(
        source,
        protected,
        source_owner=owner,
        protected_owner=owner,
    )

    assert result["schema"] == "axiom-encode/untrusted-worktree-copy/v1"
    assert result["files"] == 2
    assert len(result["inventory_sha256"]) == 64
    assert (protected / "us" / "policies" / "example.yaml").read_bytes() == (
        rule.read_bytes()
    )
    assert (protected / "script").stat().st_mode & 0o777 == 0o755
    assert not (protected / "stale.txt").exists()
    assert (protected / ".git" / "HEAD").read_bytes() == protected_head
    assert not payload.exists()


@pytest.mark.parametrize("indirection", ["symlink", "hardlink"])
def test_copy_worktree_rejects_indirection(
    tmp_path: Path,
    indirection: str,
) -> None:
    source = tmp_path / "rulespec-us-source"
    protected = tmp_path / "rulespec-us-protected"
    _git_init(source)
    _git_init(protected)
    target = source / "target.yaml"
    target.write_text("rules: []\n", encoding="utf-8")
    if indirection == "symlink":
        (source / "unsafe.yaml").symlink_to(target)
    else:
        os.link(target, source / "unsafe.yaml")

    owner = os.geteuid()
    with pytest.raises(SystemExit, match="indirection or a special file"):
        copy_worktree(
            source,
            protected,
            source_owner=owner,
            protected_owner=owner,
        )


def test_copy_worktree_cli_requires_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 501)
    with pytest.raises(SystemExit, match="must run as root"):
        main(["/tmp/source", "/tmp/destination", "--source-owner-uid", "502"])


def test_copy_worktree_bounds_empty_directory_fanout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "rulespec-us-source"
    protected = tmp_path / "rulespec-us-protected"
    _git_init(source)
    _git_init(protected)
    for index in range(3):
        (source / f"empty-{index}").mkdir()
    monkeypatch.setattr(worktree_copy, "MAX_DIRECTORY_ENTRIES", 2)

    owner = os.geteuid()
    with pytest.raises(SystemExit, match="directory exceeds its entry limit"):
        copy_worktree(
            source,
            protected,
            source_owner=owner,
            protected_owner=owner,
        )
