from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_signed_backfill import (
    REVIEWED_RULESPEC_REFS,
    branch_name,
    stage_authorized_changes,
    validate_country,
    validate_rulespec_base,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    return repo


def _add_origin_main(repo: Path) -> str:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "remote", "add", "origin", str(repo))
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    return base


def _write_signed_change(repo: Path) -> tuple[Path, Path]:
    rule = repo / "us/regulations/example.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text("rules: []\n", encoding="utf-8")
    manifest = repo / ".axiom/encoding-manifests/us/regulations/example.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "applied_files": [{"path": "us/regulations/example.yaml"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rule, manifest


@pytest.mark.parametrize("country", ["../x", "us/x", "${{ inputs.x }}", "us\n"])
def test_validate_country_rejects_adversarial_values(country: str) -> None:
    with pytest.raises(ValueError, match="two-letter lowercase"):
        validate_country(country)


def test_stage_authorized_changes_stages_only_manifest_and_applied_files(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    rule, manifest = _write_signed_change(repo)

    stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only").splitlines() == sorted(
        [str(manifest.relative_to(repo)), str(rule.relative_to(repo))]
    )


def test_stage_authorized_changes_rejects_unexpected_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write_signed_change(repo)
    (repo / "encoder-surprise.txt").write_text("must not publish\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside signed manifest authorization"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_stage_rejects_manifest_authorizing_non_rulespec_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _rule, manifest = _write_signed_change(repo)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["applied_files"] = [{"path": ".github/workflows/pwn.yml"}]
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    unexpected = repo / ".github/workflows/pwn.yml"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("name: unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a canonical RuleSpec YAML path"):
        stage_authorized_changes(repo)

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_rerun_attempt_uses_recoverable_distinct_branch() -> None:
    first = branch_name("us", "12345", "1")
    rerun = branch_name("us", "12345", "2")

    assert first == "axiom/signed-backfill-us-12345-1"
    assert rerun == "axiom/signed-backfill-us-12345-2"
    assert rerun != first


def test_validate_rulespec_base_accepts_main_ancestor(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    base = _add_origin_main(repo)

    assert validate_rulespec_base(repo, "us", base, open_pr=True) == "main"


@pytest.mark.parametrize(
    ("country", "reviewed_ref"),
    [
        ("us", "8645fb934cd02dbf730cf980507bbb2d07731bd1"),
        ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
    ],
)
def test_validate_rulespec_base_accepts_exact_reviewed_head_artifact_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    country: str,
    reviewed_ref: str,
) -> None:
    repo = tmp_path / f"rulespec-{country}"
    assert REVIEWED_RULESPEC_REFS == frozenset(
        {
            ("us", "8645fb934cd02dbf730cf980507bbb2d07731bd1"),
            ("ca", "f60f7a84c30e38c7d4961d70647eb0457e7d76c2"),
        }
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{reviewed_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    assert (
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=False)
        == "reviewed-head-artifact"
    )

    with pytest.raises(ValueError, match="artifact-only"):
        validate_rulespec_base(repo, country, reviewed_ref, open_pr=True)


def test_validate_rulespec_base_rejects_retired_reviewed_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "rulespec-us"
    retired_ref = "991f5375b92dffca57b08069093c24a463365cbc"
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill._git",
        lambda _repo, *_args: f"{retired_ref}\n".encode(),
    )
    monkeypatch.setattr(
        "scripts.prepare_signed_backfill.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1),
    )

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", retired_ref, open_pr=False)


def test_validate_rulespec_base_rejects_unreviewed_non_main_head(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _add_origin_main(repo)
    _git(repo, "commit", "--allow-empty", "-m", "unreviewed")
    head = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(ValueError, match="neither on main nor an approved"):
        validate_rulespec_base(repo, "us", head, open_pr=False)
