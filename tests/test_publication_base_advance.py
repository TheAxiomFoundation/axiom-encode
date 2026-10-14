"""Signed RuleSpec publications on a main tip that advanced after dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.prepare_signed_backfill import (
    PUBLICATION_BASE_SCHEMA,
    STALE_PUBLICATION_BASE_ERROR,
    advance_publication_base,
    publication_base_advance_paths,
    validate_rulespec_base,
)
from scripts.prepare_signed_backfill import (
    main as prepare_signed_backfill_main,
)

LANE_BRANCH = "axiom/signed-backfill-us-123-1"
WORKFLOW_TOOLCHAIN = """# Immutable checkout identities.
[workflow_toolchain]
axiom_encode_version = "0.2.2160"
axiom_compose_ref = "{compose}"
axiom_encode_ref = "{encode}"
axiom_rules_engine_ref = "{engine}"
axiom_artifact_rules_engine_ref = "{artifact_engine}"
source_staleness_axiom_encode_ref = "{staleness}"
axiom_corpus_ref = "{corpus}"
rulespec_us_ref = "{rulespec}"
"""
PINS = {
    "compose": "a" * 40,
    "encode": "b" * 40,
    "engine": "c" * 40,
    "artifact_engine": "d" * 40,
    "staleness": "e" * 40,
    "corpus": "f" * 40,
    "rulespec": "1" * 40,
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str, *paths: str) -> str:
    _git(repo, "add", "-A", "--", *(paths or (".",)))
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _rulespec_repo(tmp_path: Path) -> tuple[Path, str]:
    """A rulespec-us checkout whose recorded base is the current main tip."""

    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    _write(repo, "README.md", "base\n")
    _write(repo, ".github/workflows/repository-checks.yml", "name: checks\n")
    _write(repo, "tests/test_repository_layout.py", "def test_layout():\n    pass\n")
    _write(repo, "tests/test_encoding_manifests.py", "MANIFESTS = []\n")
    _write(repo, "tools/engine_root_load.py", "print('load')\n")
    _write(repo, ".axiom/workflow-toolchain.toml", WORKFLOW_TOOLCHAIN.format(**PINS))
    _write(repo, ".axiom/toolchain.toml", '[toolchain]\naxiom_corpus_release = "r1"\n')
    _write(repo, ".axiom/index/provisions_to_rules.json", '{"provisions": {}}\n')
    _write(repo, "known-validation-gaps.yaml", "waivers: []\n")
    _write(repo, "us/statutes/7/2014/a.yaml", "format: rulespec/v1\nrules: []\n")
    _write(repo, "us/statutes/7/2014/old.yaml", "format: rulespec/v1\nrules: []\n")
    base = _commit(repo, "base")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    return repo, base


def _advance_main(repo: Path, changes: dict[str, str | None], message: str) -> str:
    """Commit ``changes`` on main and move origin/main, then restore HEAD."""

    previous = _git(repo, "rev-parse", "HEAD")
    previous_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    tip = _git(repo, "rev-parse", "refs/remotes/origin/main")
    _git(repo, "checkout", "-q", "--detach", tip)
    for relative, text in changes.items():
        if text is None:
            (repo / relative).unlink()
        else:
            _write(repo, relative, text)
    new_tip = _commit(repo, message)
    _git(repo, "update-ref", "refs/remotes/origin/main", new_tip)
    if previous_branch == "HEAD":
        _git(repo, "checkout", "-q", "--detach", previous)
    else:
        _git(repo, "checkout", "-q", previous_branch)
    return new_tip


def _signed_lane(repo: Path, base: str) -> str:
    """Commit one signed-apply-shaped lane on the recorded base."""

    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [f]\n")
    _write(repo, "us/statutes/7/2015/f.test.yaml", "- name: f\n")
    _write(
        repo,
        ".axiom/encoding-manifests/us/statutes/7/2015/f.json",
        json.dumps({"tool": "axiom-encode encode --apply"}) + "\n",
    )
    _git(repo, "add", "-A")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=axiom-encode-bot",
            "-c",
            "user.email=axiom-encode-bot@users.noreply.github.com",
            "commit",
            "-q",
            "-m",
            "Add signed encoding manifest for us/statute/7/2015/f\n\nbody line\n",
        ],
        check=True,
    )
    return _git(repo, "rev-parse", "HEAD")


def _blob(repo: Path, commit: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "-q", f"{commit}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _commit_message_bytes(repo: Path, commit: str) -> bytes:
    raw = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "commit", commit],
        capture_output=True,
        check=True,
    ).stdout
    return raw.partition(b"\n\n")[2]


def _toolchain(**overrides: str) -> str:
    return WORKFLOW_TOOLCHAIN.format(**{**PINS, **overrides})


# --- publication_base_advance_paths -------------------------------------------


def test_exact_tip_has_no_advance_paths(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)

    assert publication_base_advance_paths(repo, base, base) == ()


def test_plumbing_only_advance_is_admitted(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {".github/workflows/repository-checks.yml": "name: v2\n"}, "ci")
    _advance_main(repo, {"tests/test_repository_layout.py": "# v2\n"}, "tests")
    _advance_main(
        repo,
        {"tools/engine_root_load.py": None, "README.md": "advanced\n"},
        "tools and docs",
    )
    tip = _git(repo, "rev-parse", "refs/remotes/origin/main")

    assert [
        path.as_posix() for path in publication_base_advance_paths(repo, base, tip)
    ] == [
        ".github/workflows/repository-checks.yml",
        "README.md",
        "tests/test_repository_layout.py",
        "tools/engine_root_load.py",
    ]


def test_net_zero_advance_is_admitted(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {"known-validation-gaps.yaml": "waivers: [x]\n"}, "stage")
    tip = _advance_main(repo, {"known-validation-gaps.yaml": "waivers: []\n"}, "revert")

    assert publication_base_advance_paths(repo, base, tip) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"known-validation-gaps.yaml": "waivers: [x]\n"},
        {".axiom/toolchain.toml": '[toolchain]\naxiom_corpus_release = "r2"\n'},
        {".axiom/index/provisions_to_rules.json": '{"provisions": {"x": []}}\n'},
        {".axiom/pending-validation-fingerprints.json": "{}\n"},
        {"oracle-coverage-pending.yaml": "pending: []\n"},
        {"known-dangling.yaml": "[]\n"},
        {"us/statutes/7/2014/a.yaml": "format: rulespec/v1\nrules: [changed]\n"},
        {"us-nc/statutes/1/2.yaml": "format: rulespec/v1\nrules: []\n"},
        {".axiom/encoding-manifests/us/statutes/7/2014/a.json": "{}\n"},
        {".axiom/legacy-replacements/receipt.json": "{}\n"},
        {"programs/snap.yaml": "program: snap\n"},
        {"tests/test_encoding_manifests.py": "MANIFESTS = ['x']\n"},
        {"tests/test_legacy_rulespec_freeze.py": "COUNT = 1\n"},
        {".gitattributes": "* text=auto eol=crlf\n"},
        {".gitignore": "*.yaml\n"},
        {"sources/us/x.txt": "x\n"},
    ],
)
def test_binding_advance_keeps_the_exact_tip_demand(
    tmp_path: Path, changes: dict[str, str]
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {"README.md": "docs\n"}, "docs")
    tip = _advance_main(repo, changes, "binding change")

    with pytest.raises(ValueError) as excinfo:
        publication_base_advance_paths(repo, base, tip)
    message = str(excinfo.value)
    assert message.startswith(STALE_PUBLICATION_BASE_ERROR)
    assert next(iter(changes)) in message
    assert "README.md" not in message


def test_workflow_toolchain_nonbinding_pin_change_is_admitted(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    tip = _advance_main(
        repo,
        {
            ".axiom/workflow-toolchain.toml": _toolchain(
                artifact_engine="9" * 40,
                staleness="8" * 40,
            )
        },
        "Pin the artifact engine",
    )

    assert [
        path.as_posix() for path in publication_base_advance_paths(repo, base, tip)
    ] == [".axiom/workflow-toolchain.toml"]


@pytest.mark.parametrize(
    "text",
    [
        _toolchain(encode="9" * 40),
        _toolchain(compose="9" * 40),
        _toolchain(engine="9" * 40),
        _toolchain(corpus="9" * 40),
        _toolchain(rulespec="9" * 40),
        _toolchain().replace('"0.2.2160"', '"0.2.2161"'),
        _toolchain() + 'new_pin = "x"\n',
        _toolchain() + "\n[other]\nkey = 1\n",
        _toolchain(artifact_engine="9" * 40).replace(
            "axiom_compose_ref", "axiom_compose_ref_renamed"
        ),
        "[workflow_toolchain\nbroken",
        "workflow_toolchain = 1\n",
    ],
)
def test_workflow_toolchain_binding_change_keeps_the_exact_tip_demand(
    tmp_path: Path, text: str
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    tip = _advance_main(repo, {".axiom/workflow-toolchain.toml": text}, "pins")

    with pytest.raises(ValueError, match=r"workflow-toolchain\.toml"):
        publication_base_advance_paths(repo, base, tip)


def test_workflow_toolchain_removal_keeps_the_exact_tip_demand(
    tmp_path: Path,
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    tip = _advance_main(repo, {".axiom/workflow-toolchain.toml": None}, "drop")

    with pytest.raises(ValueError, match=r"workflow-toolchain\.toml"):
        publication_base_advance_paths(repo, base, tip)


def test_side_branch_base_is_not_a_first_parent_ancestor(tmp_path: Path) -> None:
    repo, root = _rulespec_repo(tmp_path)
    _git(repo, "switch", "-q", "-c", "side")
    _write(repo, "README.md", "side\n")
    side = _commit(repo, "side")
    _git(repo, "switch", "-q", "main")
    _write(repo, "tests/test_repository_layout.py", "# main\n")
    _commit(repo, "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "side")
    tip = _git(repo, "rev-parse", "HEAD")
    assert _git(repo, "merge-base", "--is-ancestor", side, tip) == ""
    assert root != side

    with pytest.raises(ValueError, match="not a first-parent ancestor"):
        publication_base_advance_paths(repo, side, tip)


def test_diverged_base_is_rejected(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    tip = _advance_main(repo, {"README.md": "main\n"}, "main")
    _git(repo, "checkout", "-q", "--detach", base)
    _write(repo, "README.md", "elsewhere\n")
    diverged = _commit(repo, "elsewhere")

    with pytest.raises(ValueError, match="not a first-parent ancestor"):
        publication_base_advance_paths(repo, diverged, tip)


def test_base_ahead_of_tip_is_rejected(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    tip = _advance_main(repo, {"README.md": "main\n"}, "main")

    with pytest.raises(ValueError, match="does not descend from it"):
        publication_base_advance_paths(repo, tip, base)


def test_malformed_refs_are_rejected(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        publication_base_advance_paths(repo, "main", base)
    with pytest.raises(ValueError, match="not an available commit"):
        publication_base_advance_paths(repo, "0" * 40, base)


# --- validate_rulespec_base start gate ------------------------------------------


def test_start_gate_admits_plumbing_only_main_advance(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {".github/workflows/repository-checks.yml": "v2\n"}, "ci")
    _git(repo, "checkout", "-q", "--detach", base)

    assert validate_rulespec_base(repo, "us", base, open_pr=True) == "main"


def test_start_gate_rejects_binding_main_advance(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {"known-validation-gaps.yaml": "waivers: [x]\n"}, "waive")
    _git(repo, "checkout", "-q", "--detach", base)

    with pytest.raises(ValueError, match=STALE_PUBLICATION_BASE_ERROR):
        validate_rulespec_base(repo, "us", base, open_pr=True)


def test_start_gate_artifact_only_runs_ignore_the_tip(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _advance_main(repo, {"known-validation-gaps.yaml": "waivers: [x]\n"}, "waive")
    _git(repo, "checkout", "-q", "--detach", base)

    assert validate_rulespec_base(repo, "us", base, open_pr=False) == "main"


# --- advance_publication_base publish step --------------------------------------


def test_exact_tip_publication_is_unchanged(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    head = _signed_lane(repo, base)

    evidence = advance_publication_base(repo, base)

    assert evidence == {
        "schema": PUBLICATION_BASE_SCHEMA,
        "pr_base_branch": "main",
        "recorded_base": base,
        "published_base": base,
        "lane_head": head,
        "published_head": head,
        "mode": "exact",
        "base_advance_paths": [],
        "lane_paths": [],
    }
    assert _git(repo, "rev-parse", "HEAD") == head


def test_lane_is_replayed_onto_plumbing_advance_with_signed_bytes_intact(
    tmp_path: Path,
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    head = _signed_lane(repo, base)
    _advance_main(repo, {".github/workflows/repository-checks.yml": "v2\n"}, "ci")
    tip = _advance_main(
        repo,
        {
            "tests/test_repository_layout.py": "# v2\n",
            ".axiom/workflow-toolchain.toml": _toolchain(artifact_engine="9" * 40),
        },
        "tests and artifact engine",
    )
    lane_paths = [
        ".axiom/encoding-manifests/us/statutes/7/2015/f.json",
        "us/statutes/7/2015/f.test.yaml",
        "us/statutes/7/2015/f.yaml",
    ]
    original_blobs = {path: _blob(repo, head, path) for path in lane_paths}
    original_message = _git(repo, "log", "-1", "--format=%B", head)
    original_author = _git(repo, "log", "-1", "--format=%an <%ae> %at %ai", head)

    evidence = advance_publication_base(repo, base)

    published = _git(repo, "rev-parse", "HEAD")
    assert published != head
    assert evidence["mode"] == "advanced"
    assert evidence["recorded_base"] == base
    assert evidence["published_base"] == tip
    assert evidence["lane_head"] == head
    assert evidence["published_head"] == published
    assert evidence["lane_paths"] == lane_paths
    assert evidence["base_advance_paths"] == [
        ".axiom/workflow-toolchain.toml",
        ".github/workflows/repository-checks.yml",
        "tests/test_repository_layout.py",
    ]
    assert _git(repo, "rev-parse", f"{published}^") == tip
    assert _git(repo, "symbolic-ref", "HEAD") == f"refs/heads/{LANE_BRANCH}"
    assert _git(repo, "rev-parse", LANE_BRANCH) == published
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert {path: _blob(repo, published, path) for path in lane_paths} == (
        original_blobs
    )
    assert _git(
        repo, "diff", "--name-only", "--no-renames", tip, published
    ).splitlines() == sorted(lane_paths)
    assert (
        _git(repo, "diff", "--name-only", "--no-renames", head, published).splitlines()
        == evidence["base_advance_paths"]
    )
    assert _git(repo, "log", "-1", "--format=%B", published) == original_message
    assert _commit_message_bytes(repo, published) == _commit_message_bytes(repo, head)
    assert (
        _git(repo, "log", "-1", "--format=%an <%ae> %at %ai", published)
        == original_author
    )
    assert _git(repo, "log", "-1", "--format=%cn <%ce>", published) == (
        "axiom-encode-bot <axiom-encode-bot@users.noreply.github.com>"
    )
    assert (repo / "tests/test_repository_layout.py").read_text() == "# v2\n"


def test_multi_commit_lane_and_deletions_are_replayed_in_order(
    tmp_path: Path,
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [f]\n")
    first = _commit(repo, "Add signed source module for us/statute/7/2015/f")
    (repo / "us/statutes/7/2014/old.yaml").unlink()
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [g]\n")
    second = _commit(repo, "Compose signed source bundle")
    tip = _advance_main(repo, {"tools/engine_root_load.py": "print(2)\n"}, "tools")

    evidence = advance_publication_base(repo, base)

    replayed = _git(repo, "rev-list", "--reverse", f"{tip}..HEAD").split()
    assert len(replayed) == 2
    assert [_git(repo, "log", "-1", "--format=%s", commit) for commit in replayed] == [
        "Add signed source module for us/statute/7/2015/f",
        "Compose signed source bundle",
    ]
    assert _blob(repo, replayed[0], "us/statutes/7/2015/f.yaml") == _blob(
        repo, first, "us/statutes/7/2015/f.yaml"
    )
    assert _blob(repo, replayed[0], "us/statutes/7/2014/old.yaml") is not None
    assert _blob(repo, replayed[1], "us/statutes/7/2014/old.yaml") is None
    assert _blob(repo, replayed[1], "us/statutes/7/2015/f.yaml") == _blob(
        repo, second, "us/statutes/7/2015/f.yaml"
    )
    assert _blob(repo, replayed[0], "tools/engine_root_load.py") == _blob(
        repo, tip, "tools/engine_root_load.py"
    )
    assert evidence["lane_paths"] == [
        "us/statutes/7/2014/old.yaml",
        "us/statutes/7/2015/f.yaml",
    ]
    assert not (repo / "us/statutes/7/2014/old.yaml").exists()


def test_replay_preserves_verbatim_commit_messages(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [f]\n")
    _git(repo, "add", "-A")
    _git(
        repo,
        "commit",
        "-q",
        "--cleanup=verbatim",
        "-m",
        "Subject with trailing space \n\n# not a comment\n\n\n",
    )
    head = _git(repo, "rev-parse", "HEAD")
    _advance_main(repo, {"README.md": "docs\n"}, "docs")

    advance_publication_base(repo, base)

    assert _commit_message_bytes(repo, "HEAD") == _commit_message_bytes(repo, head)
    assert _commit_message_bytes(repo, head) == (
        b"Subject with trailing space \n\n# not a comment\n\n\n"
    )


def test_binding_advance_fails_closed_without_moving_the_lane(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    head = _signed_lane(repo, base)
    _advance_main(
        repo,
        {".axiom/toolchain.toml": '[toolchain]\naxiom_corpus_release = "r2"\n'},
        "repin corpus",
    )

    with pytest.raises(ValueError) as excinfo:
        advance_publication_base(repo, base)

    assert str(excinfo.value).startswith(STALE_PUBLICATION_BASE_ERROR)
    assert ".axiom/toolchain.toml" in str(excinfo.value)
    assert _git(repo, "rev-parse", "HEAD") == head


def test_advance_touching_a_lane_path_fails_closed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [f]\n")
    _write(repo, "tests/test_repository_layout.py", "# lane\n")
    head = _commit(repo, "lane writes a plumbing path")
    _advance_main(repo, {"tests/test_repository_layout.py": "# main\n"}, "tests")

    with pytest.raises(ValueError, match="paths the signed run changed"):
        advance_publication_base(repo, base)
    assert _git(repo, "rev-parse", "HEAD") == head


def test_intermediate_lane_path_overlap_fails_closed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "README.md", "lane\n")
    _commit(repo, "touch")
    _write(repo, "README.md", "base\n")
    _write(repo, "us/statutes/7/2015/f.yaml", "format: rulespec/v1\nrules: [f]\n")
    _commit(repo, "restore and encode")
    _advance_main(repo, {"README.md": "main\n"}, "docs")

    with pytest.raises(ValueError, match="paths the signed run changed: README.md"):
        advance_publication_base(repo, base)


def test_reviewed_branch_publication_keeps_the_exact_tip_demand(
    tmp_path: Path,
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)
    _git(repo, "update-ref", "refs/remotes/origin/reviewed", base)
    assert advance_publication_base(repo, base, pr_base_branch="reviewed")["mode"] == (
        "exact"
    )
    _advance_main(repo, {"README.md": "docs\n"}, "docs")
    _git(
        repo,
        "update-ref",
        "refs/remotes/origin/reviewed",
        _git(repo, "rev-parse", "refs/remotes/origin/main"),
    )

    with pytest.raises(ValueError, match=f"^{STALE_PUBLICATION_BASE_ERROR}$"):
        advance_publication_base(repo, base, pr_base_branch="reviewed")


def test_unavailable_base_branch_is_rejected(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)

    with pytest.raises(ValueError, match="base branch is unavailable"):
        advance_publication_base(repo, base, pr_base_branch="missing")


def test_dirty_checkout_is_not_replayed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)
    _advance_main(repo, {"README.md": "docs\n"}, "docs")
    _write(repo, "us/statutes/7/2015/stray.yaml", "x\n")

    with pytest.raises(ValueError, match="not clean"):
        advance_publication_base(repo, base)


def test_detached_head_is_not_replayed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    head = _signed_lane(repo, base)
    _advance_main(repo, {"README.md": "docs\n"}, "docs")
    _git(repo, "checkout", "-q", "--detach", head)

    with pytest.raises(ValueError, match="not a lane branch"):
        advance_publication_base(repo, base)


def test_lane_with_a_merge_commit_is_not_replayed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)
    _git(repo, "switch", "-q", "-c", "other", base)
    _write(repo, "us/statutes/7/2015/g.yaml", "format: rulespec/v1\nrules: [g]\n")
    _commit(repo, "other")
    _git(repo, "switch", "-q", LANE_BRANCH)
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", "other")
    _advance_main(repo, {"README.md": "docs\n"}, "docs")

    with pytest.raises(ValueError, match="linear chain of plain commits"):
        advance_publication_base(repo, base)


def test_empty_lane_is_not_replayed(tmp_path: Path) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _advance_main(repo, {"README.md": "docs\n"}, "docs")

    with pytest.raises(ValueError, match="no lane commits"):
        advance_publication_base(repo, base)


def test_cli_emits_publication_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)
    tip = _advance_main(repo, {".github/workflows/x.yml": "x\n"}, "ci")
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_signed_backfill.py", "advance-publication-base", str(repo), base],
    )

    prepare_signed_backfill_main()

    evidence = json.loads(capsys.readouterr().out)
    assert evidence["mode"] == "advanced"
    assert evidence["published_base"] == tip
    assert evidence["published_head"] == _git(repo, "rev-parse", "HEAD")


def test_cli_reports_binding_advance_as_stale_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base = _rulespec_repo(tmp_path)
    _signed_lane(repo, base)
    _advance_main(repo, {"known-validation-gaps.yaml": "waivers: [x]\n"}, "waive")
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_signed_backfill.py", "advance-publication-base", str(repo), base],
    )

    with pytest.raises(SystemExit) as excinfo:
        prepare_signed_backfill_main()

    assert excinfo.value.code == 1
    assert STALE_PUBLICATION_BASE_ERROR in capsys.readouterr().err


# --- the workflow publish step, end to end --------------------------------------

ROOT = Path(__file__).resolve().parents[1]
GH_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$GH_STUB_LOG"
if [ "$1" = "auth" ]; then
  exit 0
fi
if [ "$1 $2 $3" = "api --method PATCH" ]; then
  touch "$GH_STUB_CLOSED"
  exit 0
fi
if [ "$1 $2 $3" = "api --method POST" ]; then
  shift 4
  base=""
  head=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -f)
        case "$2" in
          base=*) base="${2#base=}" ;;
          head=*) head="${2#head=}" ;;
          body=*) printf '%s' "${2#body=}" > "$GH_STUB_BODY" ;;
        esac
        shift 2
        ;;
      *) shift ;;
    esac
  done
  branch="${head#TheAxiomFoundation:}"
  base_sha="${GH_STUB_BASE_SHA:-$(git --git-dir "$GH_STUB_REMOTE" rev-parse "refs/heads/$base")}"
  head_sha="$(git --git-dir "$GH_STUB_REMOTE" rev-parse "refs/heads/$branch")"
  printf '{"number": 7, "base": {"ref": "%s", "sha": "%s"}, "head": {"sha": "%s"}}\n' \
    "$base" "$base_sha" "$head_sha"
  exit 0
fi
exit 1
"""


def _publish_step() -> dict[str, object]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    return next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("id") == "publish_lane_pull_request"
    )


def _publish_fixture(tmp_path: Path) -> dict[str, Any]:
    """A GitHub-shaped remote, the runner checkout, and a committed lane."""

    remote = tmp_path / "github" / "TheAxiomFoundation" / "rulespec-us.git"
    remote.parent.mkdir(parents=True)
    seed_root = tmp_path / "seed-root"
    seed_root.mkdir()
    seed, base = _rulespec_repo(seed_root)
    _git(seed, "remote", "remove", "origin")
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(remote)], check=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "axiom-encode").symlink_to(ROOT)
    checkout = workspace / "rulespec-us"
    subprocess.run(["git", "clone", "-q", str(remote), str(checkout)], check=True)
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "config", "user.email", "test@example.com")
    head = _signed_lane(checkout, base)

    runner_temp = tmp_path / "runner"
    (runner_temp / "targeted-reencode").mkdir(parents=True)
    for name in (
        "source-bundle.json",
        "canonical-refresh-bundle.json",
        "existing-signed-imports.json",
    ):
        (runner_temp / name).write_text("[]\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH_STUB)
    gh.chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        f'[url "file://{remote.parent}/"]\n'
        "\tinsteadOf = https://github.com/TheAxiomFoundation/\n"
    )
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("GIT_", "GH_", "GITHUB_"))
        },
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "AXIOM_TEST_PYTHON": sys.executable,
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GH_TOKEN": "test-token",
        "GH_STUB_LOG": str(tmp_path / "gh.log"),
        "GH_STUB_BODY": str(tmp_path / "gh-body.txt"),
        "GH_STUB_CLOSED": str(tmp_path / "gh-closed"),
        "GH_STUB_REMOTE": str(remote),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
        "RUNNER_TEMP": str(runner_temp),
        "COUNTRY": "us",
        "CITATION": "us/statute/7/2015/f",
        "DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_CITATION": "",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
        "RULESPEC_REF": base,
        "PR_BASE_BRANCH": "main",
        "QUEUE_ID": "",
        "QUEUE_ITEM_ID": "",
        "QUEUE_ITEM_GENERATION_SHA256": "",
        "QUEUE_MANIFEST_SHA256": "",
        "REPAIR_RUN_ID": "",
        "RULESPEC_CHECKOUT": "rulespec-us",
    }
    return {
        "remote": remote,
        "workspace": workspace,
        "checkout": checkout,
        "base": base,
        "head": head,
        "runner_temp": runner_temp,
        "env": env,
        "tmp_path": tmp_path,
    }


def _advance_remote(fixture: dict[str, Any], changes: dict[str, str]) -> str:
    pusher = fixture["tmp_path"] / "pusher"
    if not pusher.exists():
        subprocess.run(
            ["git", "clone", "-q", str(fixture["remote"]), str(pusher)], check=True
        )
        _git(pusher, "config", "user.name", "Maintainer")
        _git(pusher, "config", "user.email", "maintainer@example.com")
    _git(pusher, "pull", "-q", "--ff-only", "origin", "main")
    for relative, text in changes.items():
        _write(pusher, relative, text)
    tip = _commit(pusher, "advance main")
    _git(pusher, "push", "-q", "origin", "HEAD:refs/heads/main")
    return tip


def _run_publish(
    fixture: dict[str, Any], **extra_env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", str(_publish_step()["run"])],
        cwd=fixture["workspace"],
        env={**fixture["env"], **extra_env},
        capture_output=True,
        text=True,
        check=False,
    )


def _remote_rev(fixture: dict[str, Any], revision: str) -> str | None:
    result = subprocess.run(
        [
            "git",
            "--git-dir",
            str(fixture["remote"]),
            "rev-parse",
            "--verify",
            "-q",
            revision,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _remote_lane(fixture: dict[str, Any]) -> str | None:
    return _remote_rev(fixture, "refs/heads/axiom/signed-backfill-us-123-1")


def _publication_evidence(fixture: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (fixture["runner_temp"] / "targeted-reencode/publication-base.json").read_text()
    )


def test_publish_step_keeps_exact_tip_lane(tmp_path: Path) -> None:
    fixture = _publish_fixture(tmp_path)

    result = _run_publish(fixture)

    assert result.returncode == 0, result.stderr
    assert _remote_lane(fixture) == fixture["head"]
    assert _publication_evidence(fixture)["mode"] == "exact"
    body = (tmp_path / "gh-body.txt").read_text()
    assert f"Base commit: `{fixture['base']}`" in body
    assert f"Published base commit: `{fixture['base']}`" in body
    assert "Base advance paths: `[]`" in body
    assert (
        fixture["runner_temp"] / "targeted-reencode/lane-pull-request.json"
    ).exists()


def test_publish_step_lands_signed_lane_on_plumbing_advanced_main(
    tmp_path: Path,
) -> None:
    fixture = _publish_fixture(tmp_path)
    tip = _advance_remote(
        fixture,
        {
            ".github/workflows/repository-checks.yml": "name: v2\n",
            "tests/test_repository_layout.py": "# v2\n",
        },
    )

    result = _run_publish(fixture)

    assert result.returncode == 0, result.stderr
    published = _remote_lane(fixture)
    assert published is not None and published != fixture["head"]
    assert _remote_rev(fixture, f"{published}^") == tip
    for path in (
        ".axiom/encoding-manifests/us/statutes/7/2015/f.json",
        "us/statutes/7/2015/f.test.yaml",
        "us/statutes/7/2015/f.yaml",
    ):
        assert _remote_rev(fixture, f"{published}:{path}") == _blob(
            fixture["checkout"], fixture["head"], path
        )
    evidence = _publication_evidence(fixture)
    assert evidence["mode"] == "advanced"
    assert evidence["recorded_base"] == fixture["base"]
    assert evidence["published_base"] == tip
    assert evidence["published_head"] == published
    body = (tmp_path / "gh-body.txt").read_text()
    assert f"Base commit: `{fixture['base']}`" in body
    assert f"Published base commit: `{tip}`" in body
    assert (
        'Base advance paths: `[".github/workflows/repository-checks.yml",'
        '"tests/test_repository_layout.py"]`'
    ) in body
    assert not (tmp_path / "gh-closed").exists()


def test_publish_step_fails_closed_on_binding_advance_without_pushing(
    tmp_path: Path,
) -> None:
    fixture = _publish_fixture(tmp_path)
    _advance_remote(fixture, {"known-validation-gaps.yaml": "waivers: [x]\n"})

    result = _run_publish(fixture)

    assert result.returncode != 0
    assert STALE_PUBLICATION_BASE_ERROR in result.stderr
    assert "known-validation-gaps.yaml" in result.stderr
    assert _remote_lane(fixture) is None
    assert not (tmp_path / "gh-body.txt").exists()
    assert _git(fixture["checkout"], "rev-parse", "HEAD") == fixture["head"]


def test_publish_step_closes_pr_when_main_moves_after_the_proof(
    tmp_path: Path,
) -> None:
    fixture = _publish_fixture(tmp_path)
    _advance_remote(fixture, {"README.md": "docs\n"})

    result = _run_publish(fixture, GH_STUB_BASE_SHA="9" * 40)

    assert result.returncode != 0
    assert "created pull request does not target the reviewed base SHA" in (
        result.stderr
    )
    assert (tmp_path / "gh-closed").exists()
    assert _remote_lane(fixture) is None
