"""Bind the Schedule N/T1170 local captures to their published producer inputs."""

from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY / "tests/fixtures/us_replay_benchmark"
PRODUCER = "6df4522a6ff20c78d8d70991c55f0353d49338b5"
PRODUCER_TREE = "d7746713dd0823cdf7983ce2edff73d2ec422a5f"
CAPTURES = ["local-schedule-n-20261113", "local-t1170-source-20261113"]
OLD_PRODUCERS = {
    CAPTURES[0]: (
        "e5c4b24d0fcb83291b0d684af21a0c289c1f6cb5",
        "54673baca7068aa23454d443703f21027e5838c9",
        "60f0949c1a87222826b8306294f811b6d81da412",
    ),
    CAPTURES[1]: (
        "fb376ce573872568b0a843a0d02f99bca798498b",
        "ebca52a843b9eb790598985ea82f727efaa74cef",
        "179437fa09beded785cb9d6af9247374e4e92d8d",
    ),
}


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=REPOSITORY, capture_output=True, check=False
    )
    if result.returncode:
        raise ValueError("capture provenance names an unavailable Git object")
    return result.stdout


def _assert_published_capture_inputs(directory: Path, metadata: dict) -> None:
    # These two captures use inputs already present at their producer commit.
    # Other fixtures may declare external public exports with their own proof.
    assert directory.name in CAPTURES
    producer = metadata["encoder_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", producer)
    tree = _git_bytes("rev-parse", "--verify", producer + "^{tree}").decode().strip()
    assert metadata["encoder_tree"] == tree
    assert set(metadata["input_origins"]) == {
        "candidate.yaml",
        "candidate.test.yaml",
        "source.txt",
    }
    assert set(metadata["input_git_blob_sha1"]) == set(metadata["input_origins"])
    for name, origin in metadata["input_origins"].items():
        blob = _git_bytes("rev-parse", "--verify", producer + ":" + origin)
        assert metadata["input_git_blob_sha1"][name] == blob.decode().strip()
        original = _git_bytes("cat-file", "blob", producer + ":" + origin)
        assert original == (REPOSITORY / origin).read_bytes()
        assert original == (directory / name).read_bytes()


@pytest.fixture(scope="module", autouse=True)
def full_history_capture_context():
    # Full test shards fetch all history. Draft quick checks may fetch only two
    # commits; neither the real-engine nor replay job runs these unmarked tests.
    if _git_bytes("rev-parse", "--is-shallow-repository").strip() == b"true":
        pytest.skip("published-producer provenance needs the full-history test shard")
    assert _git_bytes(
        "rev-parse", "--verify", PRODUCER + "^{tree}"
    ).decode().strip() == (PRODUCER_TREE)


@pytest.mark.parametrize("capture", CAPTURES)
def test_local_capture_binds_published_producer_tree_and_inputs(capture):
    directory = FIXTURES / capture
    metadata = json.loads((directory / "fixture.json").read_text())
    assert metadata["encoder_commit"] == PRODUCER
    assert metadata["encoder_tree"] == PRODUCER_TREE
    _assert_published_capture_inputs(directory, metadata)


@pytest.mark.parametrize("capture", CAPTURES)
@pytest.mark.parametrize(
    "mutation",
    [
        "old_pair",
        "old_producer_real_tree",
        "wrong_tree",
        "missing_id",
        "missing_path",
        "wrong_blob",
        "empty_inputs",
    ],
)
def test_local_capture_refuses_wrong_producer_tree_or_input_origin(capture, mutation):
    directory = FIXTURES / capture
    metadata = copy.deepcopy(json.loads((directory / "fixture.json").read_text()))
    old_producer, old_tree, actual_old_tree = OLD_PRODUCERS[capture]
    if mutation == "old_pair":
        metadata.update(encoder_commit=old_producer, encoder_tree=old_tree)
    elif mutation == "old_producer_real_tree":
        metadata.update(encoder_commit=old_producer, encoder_tree=actual_old_tree)
    elif mutation == "wrong_tree":
        metadata["encoder_tree"] = "f" * 40
    elif mutation == "missing_id":
        metadata["encoder_commit"] = "0" * 40
    elif mutation == "missing_path":
        metadata["input_origins"]["candidate.yaml"] = (
            "tests/fixtures/nonexistent-2110-provenance/candidate.yaml"
        )
    elif mutation == "wrong_blob":
        metadata["input_git_blob_sha1"]["candidate.yaml"] = "0" * 40
    else:
        metadata["input_origins"] = {}
        metadata["input_git_blob_sha1"] = {}
    with pytest.raises((AssertionError, ValueError)):
        _assert_published_capture_inputs(directory, metadata)
