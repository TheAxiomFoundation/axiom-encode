"""Compare dispatcher file admission with the publication reader's Git modes."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.prepare_signed_backfill import _publication_git, _regular_blob_at
from tests import test_publication_base_advance as local
from tests.test_snap_dispatch_queue import RULESPEC_REPO, q

TOOLCHAIN = ".axiom/workflow-toolchain.toml"


class LocalObjects:
    def __init__(self, repo: Path, *, fault: str = ""):
        self.repo = repo
        self.fault = fault
        self.calls: list[str] = []

    def get(self, path, params=None):
        assert params is None
        self.calls.append(path)
        kind, sha = path.removeprefix(f"repos/{RULESPEC_REPO}/git/").split("/")
        assert q.SHA_PATTERN.fullmatch(sha)
        if kind == "commits":
            if self.fault == "missing-commit":
                raise q.GitHubError(404, "missing")
            tree = local._git(self.repo, "rev-parse", sha + "^{tree}")
            return {
                "sha": "0" * 40 if self.fault == "wrong-commit-sha" else sha,
                "tree": {"sha": tree},
            }
        if kind == "trees":
            raw = local._git(self.repo, "ls-tree", "-z", sha)
            entries = []
            for record in raw.split("\0"):
                if not record:
                    continue
                metadata, name = record.split("\t", 1)
                mode, object_type, object_sha = metadata.split()
                entries.append(
                    {"mode": mode, "type": object_type, "sha": object_sha, "path": name}
                )
            directory = any(
                entry["path"] == "workflow-toolchain.toml" for entry in entries
            )
            if self.fault == ("missing-directory" if directory else "missing-root"):
                raise q.GitHubError(404, "missing")
            payload = {"sha": sha, "truncated": False, "tree": entries}
            if not directory and self.fault == "symlink-parent":
                parent = next(entry for entry in entries if entry["path"] == ".axiom")
                parent.update(mode="120000", type="blob")
            if directory:
                target = next(
                    entry
                    for entry in entries
                    if entry["path"] == "workflow-toolchain.toml"
                )
                if self.fault == "missing-entry":
                    payload["tree"] = []
                elif self.fault == "duplicate-entry":
                    payload["tree"] = entries + entries
                elif self.fault == "unknown-mode":
                    target["mode"] = "100664"
                elif self.fault == "wrong-type":
                    target["type"] = "commit"
                elif self.fault == "invalid-object-sha":
                    target["sha"] = "not-an-object"
            if self.fault == ("truncated-directory" if directory else "truncated-root"):
                payload["truncated"] = True
            if self.fault == "wrong-tree-sha":
                payload["sha"] = "0" * 40
            return payload
        assert kind == "blobs"
        if self.fault == "missing-blob":
            raise q.GitHubError(404, "missing")
        raw = _publication_git(self.repo, "cat-file", "blob", sha)
        payload = {
            "sha": sha,
            "encoding": "base64",
            "content": base64.encodebytes(raw).decode(),
        }
        if self.fault == "unknown-encoding":
            payload["encoding"] = "none"
        elif self.fault == "invalid-base64":
            payload["content"] = "!"
        elif self.fault == "changed-bytes":
            payload["content"] = base64.b64encode(raw + b"changed").decode()
        elif self.fault == "wrong-blob-sha":
            payload["sha"] = "0" * 40
        return payload


def _mode_fixture(parent: Path, mode: str) -> tuple[Path, str]:
    repo, base = local._rulespec_repo(parent)
    path = repo / TOOLCHAIN
    if mode == "100755":
        path.chmod(0o755)
    elif mode == "120000":
        local._write(repo, "tools/workflow-pins.toml", local._toolchain())
        path.unlink()
        path.symlink_to("../tools/workflow-pins.toml")
    elif mode == "160000":
        local._git(repo, "update-index", "--cacheinfo", f"160000,{base},{TOOLCHAIN}")
        local._git(repo, "commit", "-q", "-m", "Gitlink fixture")
        return repo, local._git(repo, "rev-parse", "HEAD")
    elif mode == "040000":
        path.unlink()
        local._write(repo, TOOLCHAIN + "/child", "directory fixture\n")
    if mode != "100644":
        return repo, local._commit(repo, "Mode fixture")
    return repo, base


@given(mode=st.sampled_from(["100644", "100755", "120000", "160000", "040000"]))
@settings(deadline=None)
def test_git_object_api_matches_publication_file_mode_admission(mode):
    with tempfile.TemporaryDirectory(prefix="axiom-dispatch-git-mode-") as temporary:
        repo, ref = _mode_fixture(Path(temporary), mode)
        github = LocalObjects(repo)
        expected = _regular_blob_at(repo, ref, PurePosixPath(TOOLCHAIN))

        assert q._regular_file_bytes(github, RULESPEC_REPO, TOOLCHAIN, ref) == expected
        assert (expected is not None) == (mode == "100644")
        assert len(github.calls) <= 4


@pytest.mark.parametrize(
    "fault",
    [
        "missing-commit",
        "wrong-commit-sha",
        "symlink-parent",
        "missing-root",
        "missing-directory",
        "missing-blob",
        "missing-entry",
        "duplicate-entry",
        "unknown-mode",
        "wrong-type",
        "invalid-object-sha",
        "truncated-root",
        "truncated-directory",
        "wrong-tree-sha",
        "unknown-encoding",
        "invalid-base64",
        "changed-bytes",
        "wrong-blob-sha",
    ],
)
def test_incomplete_git_object_proof_refuses_file_bytes(tmp_path, fault):
    repo, ref = _mode_fixture(tmp_path, "100644")
    github = LocalObjects(repo, fault=fault)

    assert q._regular_file_bytes(github, RULESPEC_REPO, TOOLCHAIN, ref) is None
    assert len(github.calls) <= 4
