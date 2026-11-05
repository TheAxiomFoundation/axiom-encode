"""Preparation must not turn candidate files into signing authority."""

import hashlib
import subprocess

import pytest

from axiom_encode.reviewed_replacement import inventory


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()


PRIMARY = "us-az/policies/example.yaml"
COMPANION = "us-az/policies/example.test.yaml"


@pytest.fixture(params=["root", "nested"])
def candidate(tmp_path, request):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    owner = (
        ".axiom/encoding-manifests/us-az/policies/example.json"
        if request.param == "root"
        else "us-az/.axiom/encoding-manifests/policies/example.json"
    )
    for name, content in [
        (PRIMARY, "old"),
        (COMPANION, "test"),
        (owner, "unverified evidence"),
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / PRIMARY).write_text("new")
    git(tmp_path, "commit", "-qam", "candidate")
    head = git(tmp_path, "rev-parse", "HEAD")
    return tmp_path, base, head, owner


def test_inventory_binds_committed_bytes_not_worktree(candidate):
    repo, base, head, owner = candidate
    (repo / PRIMARY).write_text("dirty unrelated working bytes")
    result = inventory(repo, base, head)
    assert result["authorizes_signing"] is False
    assert len(result["unchecked"]) == 4
    entry = result["replacements"][0]
    assert entry["candidate_rulespec_sha256"] == hashlib.sha256(b"new").hexdigest()
    assert entry["base_rulespec_sha256"] == hashlib.sha256(b"old").hexdigest()
    assert entry["predecessor_manifest"]["path"] == owner
    assert (repo / PRIMARY).read_text() == "dirty unrelated working bytes"


@pytest.mark.parametrize(
    "defect",
    [
        "manifest-edit",
        "manifest-delete",
        "unrelated",
        "new-file",
        "symlink",
        "executable",
        "missing-companion",
        "second-owner",
    ],
)
def test_inventory_refuses_unsupported_candidate(candidate, defect):
    repo, base, _, owner = candidate
    if defect == "manifest-edit":
        (repo / owner).write_text("new evidence")
    elif defect == "manifest-delete":
        (repo / owner).unlink()
    elif defect == "unrelated":
        (repo / "README.md").write_text("unrelated")
    elif defect == "new-file":
        (repo / "us-az/policies/new.yaml").write_text("new")
    elif defect == "symlink":
        (repo / PRIMARY).unlink()
        (repo / PRIMARY).symlink_to("example.test.yaml")
    elif defect == "executable":
        (repo / PRIMARY).chmod(0o755)
    elif defect == "missing-companion":
        (repo / COMPANION).unlink()
    else:
        other = (
            "us-az/.axiom/encoding-manifests/policies/example.json"
            if owner.startswith(".axiom")
            else ".axiom/encoding-manifests/us-az/policies/example.json"
        )
        path = repo / other
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("duplicate")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", defect)
    with pytest.raises(ValueError):
        inventory(repo, base, git(repo, "rev-parse", "HEAD"))


def test_inventory_refuses_symbolic_refs_and_empty_diff(candidate):
    repo, base, head, _ = candidate
    with pytest.raises(ValueError, match="full lowercase"):
        inventory(repo, base, "HEAD")
    with pytest.raises(ValueError, match="changes no"):
        inventory(repo, head, head)


def test_inventory_refuses_nonancestor(candidate):
    repo, base, head, _ = candidate
    with pytest.raises(subprocess.CalledProcessError):
        inventory(repo, head, base)


def test_inventory_cannot_hide_changes_with_relative_diff(candidate):
    repo, base, _, _ = candidate
    (repo / "README.md").write_text("unrelated")
    git(repo, "add", "README.md")
    git(repo, "commit", "-qm", "unrelated root change")
    head = git(repo, "rev-parse", "HEAD")
    git(repo, "config", "diff.relative", "true")
    with pytest.raises(ValueError, match="README.md"):
        inventory(repo / "us-az", base, head)


def test_inventory_cannot_hide_gitlink_changes(candidate):
    repo, base, head, _ = candidate
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{base},vendor")
    git(repo, "commit", "-qm", "base with gitlink")
    link_base = git(repo, "rev-parse", "HEAD")
    (repo / PRIMARY).write_text("replacement with unrelated gitlink change")
    git(repo, "add", PRIMARY)
    git(repo, "update-index", "--cacheinfo", f"160000,{head},vendor")
    git(repo, "commit", "-qm", "changed gitlink")
    link_head = git(repo, "rev-parse", "HEAD")
    git(repo, "config", "diff.ignoreSubmodules", "all")
    with pytest.raises(ValueError, match="vendor"):
        inventory(repo, link_base, link_head)


@pytest.mark.parametrize("ambient", [True, False])
def test_inventory_ignores_grafts(candidate, monkeypatch, ambient):
    repo, base, head, _ = candidate
    tree = git(repo, "rev-parse", f"{base}^{{tree}}")
    sibling = git(repo, "commit-tree", tree, "-p", base, "-m", "sibling")
    graft = repo / ("forged-grafts" if ambient else ".git/info/grafts")
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text(f"{head} {sibling}\n")
    if ambient:
        monkeypatch.setenv("GIT_GRAFT_FILE", str(graft))
    with pytest.raises(subprocess.CalledProcessError):
        inventory(repo, sibling, head)


def test_inventory_rejects_ambiguous_owners_already_in_base(candidate):
    repo, _, _, owner = candidate
    other = (
        "us-az/.axiom/encoding-manifests/policies/example.json"
        if owner.startswith(".axiom")
        else ".axiom/encoding-manifests/us-az/policies/example.json"
    )
    path = repo / other
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("duplicate owner")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "ambiguous base")
    base = git(repo, "rev-parse", "HEAD")
    (repo / PRIMARY).write_text("second candidate")
    git(repo, "commit", "-qam", "second candidate")
    with pytest.raises(ValueError, match="exactly one predecessor"):
        inventory(repo, base, git(repo, "rev-parse", "HEAD"))


def test_inventory_ignores_local_git_replacement_refs(candidate):
    repo, base, head, _ = candidate
    (repo / PRIMARY).write_text("substituted tree")
    git(repo, "commit", "-qam", "replacement tree")
    replacement = git(repo, "rev-parse", "HEAD")
    git(repo, "replace", head, replacement)
    assert git(repo, "show", f"{head}:{PRIMARY}") == "substituted tree"
    result = inventory(repo, base, head)
    assert result["candidate_sha"] == head
    assert (
        result["replacements"][0]["candidate_rulespec_sha256"]
        == hashlib.sha256(b"new").hexdigest()
    )
