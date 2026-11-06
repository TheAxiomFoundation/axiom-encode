"""RuleSpec-repository admissions registry (#2095)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode import rulespec_admission_registry as registry
from axiom_encode.harness import coverage_exemptions as ce

ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "Fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.com",
    "GIT_COMMITTER_NAME": "Fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.com",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=ENV,
    ).stdout.strip()


def _packaged_us_entry() -> dict:
    entry = next(
        e
        for e in ce._load_packaged_coverage_exemptions()
        if e.citation_path.split("/", 1)[0].startswith("us")
    )
    raw = {field: getattr(entry, field) for field in entry.__dataclass_fields__}
    raw["requested_scope"] = (
        list(raw["requested_scope"]) if raw["requested_scope"] else None
    )
    raw["owner_evidence"] = dict(raw["owner_evidence"])
    return raw


def _new_us_entry() -> dict:
    raw = _packaged_us_entry()
    raw["physical_start"] += 100000  # a key that no encoder record uses
    return raw


def _repo(tmp_path: Path, name: str = "rulespec-us") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("fixture\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _commit_registry(repo: Path, lines: list[dict], name=None) -> str:
    directory = repo / ".axiom/admissions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or registry.COVERAGE_EXEMPTIONS_FILE)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "registry")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch):
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)
    ce._load_coverage_exemptions_with_registry.cache_clear()
    yield
    ce._load_coverage_exemptions_with_registry.cache_clear()


def test_exported_registry_extends_encoder_exemptions(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    entry = _new_us_entry()
    commit = _commit_registry(repo, [entry])
    exported = registry.export_registry(repo, commit, tmp_path / "out")
    assert exported.country == "us" and exported.commit == commit
    assert len(exported.coverage_exemptions) == 1
    packaged = ce.load_coverage_exemptions()
    monkeypatch.setenv(registry.REGISTRY_ENV, str(tmp_path / "out"))
    merged = ce.load_coverage_exemptions()
    assert len(merged) == len(packaged) + 1
    assert exported.coverage_exemptions[0] in merged


def test_export_reads_the_protected_commit_not_later_or_working_bytes(tmp_path):
    repo = _repo(tmp_path)
    base = _commit_registry(repo, [_new_us_entry()])
    second = _new_us_entry()
    second["physical_start"] += 1
    _commit_registry(repo, [_new_us_entry(), second])
    (repo / ".axiom/admissions/coverage-exemptions.jsonl").write_text("garbage\n")
    exported = registry.export_registry(repo, base, tmp_path / "out")
    assert len(exported.coverage_exemptions) == 1


def test_absent_registry_exports_empty(tmp_path):
    repo = _repo(tmp_path)
    exported = registry.export_registry(
        repo, _git(repo, "rev-parse", "HEAD"), tmp_path / "out"
    )
    assert exported.coverage_exemptions == ()


def test_foreign_country_exemption_fails(tmp_path):
    repo = _repo(tmp_path, "rulespec-ca")
    commit = _commit_registry(repo, [_new_us_entry()])
    with pytest.raises(registry.AdmissionRegistryError, match="outside the ca"):
        registry.export_registry(repo, commit, tmp_path / "out")


def test_unknown_registry_file_fails(tmp_path):
    repo = _repo(tmp_path)
    commit = _commit_registry(repo, [], name="reviewed-heads.json")
    with pytest.raises(registry.AdmissionRegistryError, match="unknown registry file"):
        registry.export_registry(repo, commit, tmp_path / "out")


def test_symlinked_registry_file_fails(tmp_path):
    repo = _repo(tmp_path)
    directory = repo / ".axiom/admissions"
    directory.mkdir(parents=True)
    (repo / "elsewhere.jsonl").write_text("")
    (directory / registry.COVERAGE_EXEMPTIONS_FILE).symlink_to("../../elsewhere.jsonl")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "symlink")
    with pytest.raises(registry.AdmissionRegistryError, match="not a regular file"):
        registry.export_registry(repo, _git(repo, "rev-parse", "HEAD"), tmp_path / "o")


def test_conflicting_key_with_an_encoder_record_fails(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    conflicting = _packaged_us_entry()
    conflicting["reviewer"] = "someone-else"
    commit = _commit_registry(repo, [conflicting])
    registry.export_registry(repo, commit, tmp_path / "out")
    monkeypatch.setenv(registry.REGISTRY_ENV, str(tmp_path / "out"))
    with pytest.raises(registry.AdmissionRegistryError, match="conflicts"):
        ce.load_coverage_exemptions()


def test_identical_copy_of_an_encoder_record_is_accepted(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    commit = _commit_registry(repo, [_packaged_us_entry()])
    registry.export_registry(repo, commit, tmp_path / "out")
    packaged = ce.load_coverage_exemptions()
    monkeypatch.setenv(registry.REGISTRY_ENV, str(tmp_path / "out"))
    assert ce.load_coverage_exemptions() == packaged


@pytest.mark.parametrize("commit", ["HEAD", "main", "a" * 39])
def test_export_requires_a_full_commit(tmp_path, commit):
    repo = _repo(tmp_path)
    with pytest.raises(registry.AdmissionRegistryError, match="full commit"):
        registry.export_registry(repo, commit, tmp_path / "out")


def test_check_command_validates_head_and_refuses_uncommitted_edits(tmp_path, capsys):
    repo = _repo(tmp_path)
    _commit_registry(repo, [_new_us_entry()])
    args = type("Args", (), {"admission_registry_command": "check", "repo": repo})
    assert cli.cmd_admission_registry(args) == 0
    assert "1 coverage exemptions" in capsys.readouterr().out
    (repo / ".axiom/admissions/coverage-exemptions.jsonl").write_text("{}\n")
    assert cli.cmd_admission_registry(args) == 1
