"""CI-only decrement ratchet for the frozen v1 compatibility inventory."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

INVENTORY = Path("src/axiom_encode/data/v1_source_attestation_rollout_inventory.json")


def _entries(payload: dict[str, object]) -> dict[tuple[str, str], str]:
    repositories = payload.get("repositories")
    assert isinstance(repositories, dict)
    return {
        (str(repository), str(entry["path"])): str(entry["sha256"])
        for repository, raw_entries in repositories.items()
        if isinstance(raw_entries, list)
        for entry in raw_entries
        if isinstance(entry, dict)
    }


def test_v1_rollout_inventory_is_removal_only_against_pr_merge_base() -> None:
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if not base_ref:
        pytest.skip("GITHUB_BASE_REF is only available in pull-request CI")
    repo = Path(__file__).resolve().parents[1]
    candidates = (f"origin/{base_ref}", base_ref)
    merge_base = None
    for candidate in candidates:
        completed = subprocess.run(
            ["git", "merge-base", "HEAD", candidate],
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            merge_base = completed.stdout.strip()
            break
    if not merge_base:
        pytest.skip(f"base ref {base_ref!r} is unavailable in this checkout")
    previous = subprocess.run(
        ["git", "show", f"{merge_base}:{INVENTORY.as_posix()}"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    old_entries = _entries(json.loads(previous.stdout))
    new_entries = _entries(json.loads((repo / INVENTORY).read_text()))
    added_or_modified = {
        key: digest
        for key, digest in new_entries.items()
        if old_entries.get(key) != digest
    }
    assert not added_or_modified, (
        "the frozen v1 rollout inventory is decrement-only; entries may only be removed: "
        f"{added_or_modified}"
    )
