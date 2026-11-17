"""Actual production Git broker regression; no wrapper policy is relaxed."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import reviewed_repair_patch as patch
from tests.test_provision_supervisor import provisioner
from tests.test_reviewed_repair_patch import lineage_fixture
from tests.test_reviewed_repair_patch import registered as registered_fixture


@pytest.fixture(name="registered")
def protected_registration(tmp_path, monkeypatch):
    # Keep this fixture local even when pytest collected the donor module first.
    return registered_fixture.__wrapped__(tmp_path, monkeypatch)


def install_wrapper(tmp_path, monkeypatch):
    tools = tmp_path / "protected-tools"
    tools.mkdir()
    real_git = Path(shutil.which("git")).resolve()
    provisioner._install_trusted_git_wrapper(
        tools,
        Path(sys.executable).resolve(),
        provisioner._resolve_trusted_git(real_git),
    )
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])


def test_registered_package_through_actual_protected_wrapper(
    registered, tmp_path, monkeypatch
):
    root, _, reg = registered
    expected = patch.load_registered_patch(root, "fixture")
    install_wrapper(tmp_path, monkeypatch)
    for args in (
        ["ls-tree", "-r", "-z", reg.commit, "--", "reviewed-repair-patches/fixture"],
        ["cat-file", "-s", reg.members[0][0]],
    ):
        refused = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
        assert refused.returncode == 126
        assert b"trusted git wrapper refused" in refused.stderr
    assert patch.load_registered_patch(root, "fixture") == expected


def test_predecessor_and_consumer_through_actual_protected_wrapper(
    tmp_path, monkeypatch
):
    repo, package, payload, canonical = lineage_fixture(tmp_path, monkeypatch)
    expected = patch.verify_predecessor(repo, package)
    install_wrapper(tmp_path, monkeypatch)
    assert patch.verify_predecessor(repo, package) == expected
    patch.verify_manifest_reconstruction(payload, repo, canonical)
    # Live tampering is still rejected independently of the immutable listing.
    target = repo / package.manifest["rulespec_path"]
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="strictly clean"):
        patch.verify_predecessor(repo, package)
    historical = repo / package.manifest["predecessor"]["path"]
    historical.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="historical predecessor changed"):
        patch.verify_manifest_reconstruction(payload, repo, canonical)


@pytest.mark.parametrize("name", ["extra", "nested/extra", "rules.patch.yaml"])
def test_package_membership_and_modes_still_refused(registered, monkeypatch, name):
    root, git, reg = registered
    path = root / "reviewed-repair-patches/fixture" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "rules.patch.yaml":
        path.unlink()
        path.symlink_to("patch.json")
    else:
        path.write_text("extra")
    git("add", ".")
    git("commit", "-qm", "malformed fixture")
    monkeypatch.setattr(
        patch,
        "REVIEWED_REPAIR_PATCHES",
        {
            "fixture": patch.ReviewedPatchRegistration(
                git("rev-parse", "HEAD"), reg.members
            )
        },
    )
    with pytest.raises(ValueError, match="missing or extra|mode or Git blob"):
        patch.load_registered_patch(root, "fixture")


def test_registered_size_refused_before_blob_read(registered, monkeypatch):
    root, _, reg = registered
    members = list(reg.members)
    blob, digest, size = members[0]
    members[0] = (blob, digest, size + 1)
    monkeypatch.setattr(
        patch,
        "REVIEWED_REPAIR_PATCHES",
        {"fixture": patch.ReviewedPatchRegistration(reg.commit, tuple(members))},
    )

    def unexpected(*args):
        pytest.fail("must refuse tree size before reading a blob")

    monkeypatch.setattr(patch, "_immutable_blob", unexpected)
    with pytest.raises(ValueError, match="size mismatch"):
        patch.load_registered_patch(root, "fixture")


@pytest.mark.parametrize(
    "raw",
    [
        b"100644 blob " + b"a" * 40 + b" 1\tx",  # truncated
        (b"100644 blob " + b"a" * 40 + b" 1\tx\0") * 2,
        b"100644 blob " + b"a" * 40 + b" -\tx\0",
    ],
)
def test_invalid_tree_listing_fails_closed(tmp_path, monkeypatch, raw):
    monkeypatch.setattr(patch, "_bounded_git_read", lambda *a, **k: raw)
    with pytest.raises(ValueError):
        patch._immutable_tree(tmp_path, "a" * 40)


def test_tree_and_blob_output_bounds(registered, monkeypatch):
    root, _, reg = registered
    monkeypatch.setattr(patch, "MAX_TREE_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds bound"):
        patch._immutable_tree(root, reg.commit)
    with pytest.raises(ValueError, match="exceeds bound"):
        patch._immutable_blob(root, reg.members[0][0], 1)


def test_committed_empty_canonical_tree_remains_present(tmp_path, monkeypatch):
    from dataclasses import replace

    repo, package, payload, canonical = lineage_fixture(tmp_path, monkeypatch)

    def git(*args, data=None):
        return subprocess.run(
            ["git", "-C", str(repo), *args], input=data, check=True, capture_output=True
        ).stdout

    empty = git("mktree", data=b"").decode().strip()

    def with_empty(tree, parts):
        records = {}
        for record in git("ls-tree", "-z", tree).split(b"\0"):
            if record:
                header, name = record.split(b"\t", 1)
                records[name] = header
        name = parts[0].encode()
        if len(parts) == 1:
            child = empty
        else:
            prior = (
                records.get(name, f"040000 tree {empty}".encode()).split()[2].decode()
            )
            child = with_empty(prior, parts[1:])
        records[name] = f"040000 tree {child}".encode()
        return (
            git(
                "mktree",
                "-z",
                data=b"".join(
                    header + b"\t" + name + b"\0"
                    for name, header in sorted(records.items())
                ),
            )
            .decode()
            .strip()
        )

    base = package.manifest["rulespec_base_commit"]
    tree = with_empty(
        git("rev-parse", "HEAD^{tree}").decode().strip(), canonical.split("/")
    )
    commit = (
        git("commit-tree", tree, "-p", base, data=b"empty canonical tree\n")
        .decode()
        .strip()
    )
    git("reset", "--hard", commit)
    declared = package.manifest
    declared["rulespec_base_commit"] = commit
    package = replace(package, manifest_bytes=patch.canonical_json(declared))
    monkeypatch.setattr(
        patch,
        "REVIEWED_REPAIR_PATCHES",
        {
            "fixture": patch.ReviewedPatchRegistration(
                package.commit,
                tuple(
                    (patch.git_blob_id(raw), patch.sha256(raw), len(raw))
                    for raw in (
                        package.manifest_bytes,
                        package.rules_bytes,
                        package.tests_bytes,
                    )
                ),
            )
        },
    )
    target = repo / declared["rulespec_path"]
    payload["reviewed_repair"] = patch.build_repair_binding(
        package,
        rulespec=target.read_bytes(),
        tests=target.with_suffix(".test.yaml").read_bytes(),
    )
    install_wrapper(tmp_path, monkeypatch)
    with pytest.raises(
        ValueError, match="canonical manifest unexpectedly present at base"
    ):
        patch.verify_predecessor(repo, package)
    with pytest.raises(ValueError, match="canonical manifest was not absent at base"):
        patch.verify_manifest_reconstruction(payload, repo, canonical)


def test_tree_entry_count_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(patch, "MAX_TREE_ENTRIES", 1)
    raw = b"".join(
        b"100644 blob " + b"a" * 40 + b" 1\t" + name + b"\0"
        for name in (b"one", b"two")
    )
    monkeypatch.setattr(patch, "_bounded_git_read", lambda *a, **k: raw)
    with pytest.raises(ValueError, match="entry bound"):
        patch._immutable_tree(tmp_path, "a" * 40)


def test_git_read_timeout_reaps_process(tmp_path, monkeypatch):
    # A local stalled executable isolates the reader's timeout, without network.
    script = tmp_path / "git"
    script.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(patch, "GIT_READ_TIMEOUT", 0.05)
    with pytest.raises(ValueError, match="timed out"):
        patch._bounded_git_read(tmp_path, "unused", limit=10)
