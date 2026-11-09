"""Smoke coverage for unreceipted modules under UK council jurisdiction roots."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import cli
from tests import test_cli as fixtures
from tests.test_unreceipted_retirement import (
    retirement_test_signer as retirement_test_signer,
)


@pytest.mark.parametrize("council", ["arun", "leeds"])
def test_unreceipted_uk_council_round_trip(tmp_path, council):
    fixture = fixtures.TestCmdRetire()
    repo, primary, companion, prior_manifest = fixture._repo_with_encoded_rule(tmp_path)
    fixture._add_second_encoded_rule(repo, prior_manifest)

    relative = Path(f"uk-{council}/policies/{council}/council-tax-reduction.yaml")
    target = repo / relative
    target_test = target.with_name("council-tax-reduction.test.yaml")
    target.parent.mkdir(parents=True)
    primary.rename(target)
    companion.rename(target_test)
    prior_manifest.unlink()
    expected_entries = [
        {
            "path": path.relative_to(repo).as_posix(),
            "deleted": True,
            "sha256": fixtures._sha256_file(path),
        }
        for path in (target, target_test)
    ]

    fixtures._git(repo, "init")
    fixtures._git(repo, "config", "user.email", "test@example.com")
    fixtures._git(repo, "config", "user.name", "Test User")
    fixtures._git(repo, "add", "-A")
    fixtures._git(repo, "commit", "-m", "unreceipted UK council module")
    base = fixtures._git(repo, "rev-parse", "HEAD").stdout.strip()

    cli.cmd_retire(
        SimpleNamespace(
            paths=[relative.as_posix()],
            policy_repo_path=repo,
            corpus_path=fixture.corpus_path,
            reason="replace merged council module with section encodings",
            unreceipted=True,
        )
    )
    manifest = repo / cli._applied_encoding_manifest_path(relative)
    payload = json.loads(manifest.read_text())
    assert payload["tool"] == cli.APPLIED_ENCODING_UNRECEIPTED_RETIRE_TOOL
    assert payload["applied_files"] == expected_entries
    assert not target.exists() and not target_test.exists()
    fixtures._git(repo, "add", "-A")
    fixtures._git(repo, "commit", "-m", "retire unreceipted UK council module")

    assert (
        cli.guard_generated_change_issues(
            repo,
            corpus_path=fixture.corpus_path,
            base_ref=base,
            head_ref="HEAD",
        )
        == []
    )
    assert (
        cli.guard_generated_change_issues(
            repo,
            corpus_path=fixture.corpus_path,
            base_ref="HEAD",
            head_ref="HEAD",
            all_files=True,
        )
        == []
    )


@pytest.mark.parametrize("unreceipted", [False, True])
def test_cli_retirement_requires_explicit_opt_in(tmp_path, monkeypatch, unreceipted):
    calls = []
    monkeypatch.setattr(cli, "cmd_retire", calls.append)
    arguments = [
        "axiom-encode",
        "retire",
        "uk-arun/policies/arun/council-tax-reduction.yaml",
        "--policy-repo-path",
        str(tmp_path / "rulespec-uk"),
        "--corpus-path",
        str(tmp_path / "axiom-corpus"),
        "--reason",
        "replace merged council module",
    ]
    if unreceipted:
        arguments.append("--unreceipted")
    monkeypatch.setattr("sys.argv", arguments)
    cli.main()
    assert len(calls) == 1
    assert calls[0].unreceipted is unreceipted
