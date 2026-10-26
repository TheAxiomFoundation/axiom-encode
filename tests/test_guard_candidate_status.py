"""guard-generated --candidate-status classifies change sets (#1628, B3)."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import cli
from tests.test_cli import _bind_test_corpus_release, _git, _init_test_git_repo

MODULE = "us/statutes/26/21.yaml"
COMPANION = "us/statutes/26/21.test.yaml"
MANIFEST = ".axiom/encoding-manifests/us/statutes/26/21.json"
OTHER_MODULE = "us/statutes/26/22.yaml"
OTHER_COMPANION = "us/statutes/26/22.test.yaml"


def _module_text(citation: str) -> str:
    return (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation}\n"
        "rules: []\n"
    )


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "--allow-empty", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    checkout = tmp_path / "rulespec-us"
    _init_test_git_repo(checkout)
    _write(checkout, "known-validation-gaps.yaml", "validate_failures: []\n")
    _write(checkout, "README.md", "rules\n")
    _commit(checkout, "base")
    return checkout


def _report(repo: Path, base: str) -> dict:
    return cli.guard_candidate_status_report(repo, base_ref=base, head_ref="HEAD")


def _candidate(repo: Path) -> None:
    _write(repo, MODULE, _module_text("us/statute/26/21"))
    _write(repo, COMPANION, "- name: case\n")


def _base(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_single_module_and_companion_is_unadmitted_candidate(repo):
    base = _base(repo)
    _candidate(repo)
    head = _commit(repo, "candidate")

    report = _report(repo, base)

    assert report["status"] == "unadmitted_reviewed_candidate"
    assert report["reasons"] == []
    assert report["schema"] == "axiom-encode/guard-candidate-status/v1"
    assert (report["country"], report["base_sha"], report["head_sha"]) == (
        "us",
        base,
        head,
    )
    assert report["candidate"] == {
        "citation": "us/statute/26/21",
        "rulespec_path": MODULE,
        "rulespec_sha256": hashlib.sha256((repo / MODULE).read_bytes()).hexdigest(),
        "companion_path": COMPANION,
        "companion_sha256": hashlib.sha256((repo / COMPANION).read_bytes()).hexdigest(),
    }
    assert [entry["path"] for entry in report["changed_files"]] == [
        COMPANION,
        MODULE,
    ]
    json.dumps(report)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("module_without_companion", f"companion test {COMPANION} did not change"),
        ("companion_without_module", "expected exactly one RuleSpec module, found 0"),
        ("two_modules", "expected exactly one RuleSpec module, found 2"),
        ("noncanonical_path", "is not at the canonical path us/statutes/26/21.yaml"),
        ("no_citation", "declares no module.source_verification"),
        ("manifest_only", "manifest-only change set"),
        ("metadata_change", "provenance metadata changed: .axiom/"),
        ("waiver_change", "validation waiver set changed: known-validation-gaps"),
        ("other_file_change", "non-RuleSpec file changed: README.md"),
        ("manifest_missing_for_module", f"without its matching manifest {MANIFEST}"),
        ("deleted_module", f"{MODULE} is deleted at head"),
        ("deleted_manifest", f"manifest {MANIFEST} is deleted at head"),
        ("symlink_manifest", f"manifest {MANIFEST} is not a regular 0644 file"),
        ("waiver_only", "validation waiver set changed without a RuleSpec change"),
        (
            "jurisdiction_tree_manifest_only",
            "provenance metadata changed without a RuleSpec change: us/.axiom/",
        ),
        ("double_slash_citation", "has no canonical path"),
        ("trailing_slash_citation", "has no canonical path"),
        ("plural_class_citation", "has no canonical path"),
        ("foreign_jurisdiction", "is outside the rulespec-us jurisdiction"),
        ("invalid_yaml_date", "declares no module.source_verification"),
        ("deeply_nested_yaml", "declares no module.source_verification"),
        ("duplicate_citation_key", "declares no module.source_verification"),
        (
            "case_alias_metadata_only",
            "provenance metadata changed without a RuleSpec change: .AXIOM/",
        ),
    ],
)
def test_invalid_change_sets_report_reasons(repo, case, expected_reason):
    if case in {"manifest_only", "deleted_module", "deleted_manifest"}:
        _candidate(repo)
        if case in {"manifest_only", "deleted_manifest"}:
            _write(repo, MANIFEST, "{}\n")
        _commit(repo, "existing module")
    base = _base(repo)
    if case == "module_without_companion":
        _write(repo, MODULE, _module_text("us/statute/26/21"))
    elif case == "companion_without_module":
        _write(repo, COMPANION, "- name: case\n")
    elif case == "two_modules":
        _candidate(repo)
        _write(repo, OTHER_MODULE, _module_text("us/statute/26/22"))
        _write(repo, OTHER_COMPANION, "- name: case\n")
    elif case == "noncanonical_path":
        _write(repo, OTHER_MODULE, _module_text("us/statute/26/21"))
        _write(repo, OTHER_COMPANION, "- name: case\n")
    elif case == "no_citation":
        _write(repo, MODULE, "format: rulespec/v1\nrules: []\n")
        _write(repo, COMPANION, "- name: case\n")
    elif case == "manifest_only":
        _write(repo, MANIFEST, '{"changed": true}\n')
    elif case == "metadata_change":
        _candidate(repo)
        _write(repo, ".axiom/retired-source-metadata/receipt.json", "{}\n")
    elif case == "waiver_change":
        _candidate(repo)
        _write(repo, "known-validation-gaps.yaml", "validate_failures: [x]\n")
    elif case == "other_file_change":
        _candidate(repo)
        _write(repo, "README.md", "changed\n")
    elif case == "manifest_missing_for_module":
        _candidate(repo)
        _write(repo, OTHER_MODULE, _module_text("us/statute/26/22"))
        _write(
            repo,
            ".axiom/encoding-manifests/us/statutes/26/22.json",
            "{}\n",
        )
    elif case == "deleted_module":
        (repo / MODULE).unlink()
        _write(repo, COMPANION, "- name: changed\n")
    elif case == "deleted_manifest":
        _write(repo, MODULE, _module_text("us/statute/26/21") + "# edit\n")
        (repo / MANIFEST).unlink()
    elif case == "symlink_manifest":
        _candidate(repo)
        _write(repo, "manifest-target.json", "{}\n")
        (repo / MANIFEST).parent.mkdir(parents=True)
        (repo / MANIFEST).symlink_to(repo / "manifest-target.json")
    elif case == "waiver_only":
        _write(repo, "known-validation-gaps.yaml", "validate_failures: [x]\n")
    elif case == "jurisdiction_tree_manifest_only":
        _write(repo, "us/.axiom/encoding-manifests/statutes/26/21.json", "{}\n")
    elif case in {
        "double_slash_citation",
        "trailing_slash_citation",
        "plural_class_citation",
    }:
        citation = {
            "double_slash_citation": "us//statute/26/21",
            "trailing_slash_citation": "us/statute/26/21/",
            "plural_class_citation": "us/statutes/26/21",
        }[case]
        _write(repo, MODULE, _module_text(citation))
        _write(repo, COMPANION, "- name: case\n")
    elif case == "foreign_jurisdiction":
        _write(repo, "ca/statutes/26/21.yaml", _module_text("ca/statute/26/21"))
        _write(repo, "ca/statutes/26/21.test.yaml", "- name: case\n")
    elif case == "invalid_yaml_date":
        _write(repo, MODULE, "invalid: 2026-99-99\n")
        _write(repo, COMPANION, "- name: case\n")
    elif case == "duplicate_citation_key":
        _write(
            repo,
            MODULE,
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            "    corpus_citation_path: ca/statute/26/21\n"
            "    corpus_citation_path: us/statute/26/21\n"
            "rules: []\n",
        )
        _write(repo, COMPANION, "- name: case\n")
    elif case == "case_alias_metadata_only":
        _write(repo, ".AXIOM/encoding-manifests/us/statutes/26/21.json", "{}\n")
    elif case == "deeply_nested_yaml":
        _write(repo, MODULE, "[" * 5000 + "]" * 5000 + "\n")
        _write(repo, COMPANION, "- name: case\n")
    _commit(repo, case)

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert report["candidate"] is None
    assert any(expected_reason in reason for reason in report["reasons"]), report[
        "reasons"
    ]


def test_change_set_with_matching_manifest_is_admitted(repo):
    base = _base(repo)
    _candidate(repo)
    _write(repo, MANIFEST, "{}\n")
    _write(repo, "known-validation-gaps.yaml", "validate_failures: []\n# -\n")
    _commit(repo, "applied")

    report = _report(repo, base)

    assert report["status"] == "admitted"
    assert report["reasons"] == []
    assert report["candidate"] is None
    assert {entry["path"]: entry["kind"] for entry in report["changed_files"]} == {
        COMPANION: "companion",
        MODULE: "module",
        MANIFEST: "manifest",
        "known-validation-gaps.yaml": "waiver",
    }


def test_symlinked_module_is_invalid(repo):
    base = _base(repo)
    _write(repo, "target.yaml", _module_text("us/statute/26/21"))
    (repo / MODULE).parent.mkdir(parents=True)
    (repo / MODULE).symlink_to(repo / "target.yaml")
    _write(repo, COMPANION, "- name: case\n")
    _commit(repo, "symlink")

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert any("not a regular 0644 file" in reason for reason in report["reasons"])


def test_uncommitted_change_and_missing_base_are_invalid(repo):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    (repo / "scratch.txt").write_text("untracked\n")

    dirty = _report(repo, base)
    no_base = cli.guard_candidate_status_report(repo, base_ref=None)

    assert dirty["status"] == "invalid"
    assert any("scratch.txt" in reason for reason in dirty["reasons"])
    assert no_base["status"] == "invalid"
    assert no_base["reasons"] == ["candidate status requires --base-ref"]


def test_replacement_refs_cannot_hide_changes(repo):
    base = _base(repo)
    _candidate(repo)
    _write(repo, ".axiom/retired-source-metadata/receipt.json", "{}\n")
    _commit(repo, "candidate with metadata")
    _git(repo, "checkout", "-q", "-b", "decoy", base)
    _write(repo, ".axiom/retired-source-metadata/receipt.json", "{}\n")
    decoy = _commit(repo, "decoy base")
    _git(repo, "checkout", "-q", "-")
    _git(repo, "replace", base, decoy)

    report = _report(repo, base)

    assert report["base_sha"] == base
    assert report["status"] == "invalid"
    assert any(
        "provenance metadata changed: .axiom/retired-source-metadata" in reason
        for reason in report["reasons"]
    )


def test_ignored_submodule_settings_cannot_hide_gitlink_changes(repo):
    first, second = "1" * 40, "2" * 40
    _write(
        repo,
        ".gitmodules",
        '[submodule "x"]\n\tpath = .axiom/vendor\n\turl = ./x\n\tignore = all\n',
    )
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{first},.axiom/vendor")
    _git(repo, "add", ".gitmodules")
    _git(repo, "commit", "-q", "-m", "gitlink")
    base = _base(repo)
    _git(repo, "config", "diff.ignoreSubmodules", "all")
    _candidate(repo)
    _git(repo, "add", MODULE, COMPANION)
    _git(repo, "update-index", "--cacheinfo", f"160000,{second},.axiom/vendor")
    _git(repo, "commit", "-q", "-m", "candidate with gitlink update")

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert any(
        "provenance metadata changed: .axiom/vendor" in reason
        for reason in report["reasons"]
    ), report["reasons"]


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_cannot_hide_worktree_changes(repo, flag):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    _git(repo, "update-index", flag, "known-validation-gaps.yaml")
    _write(repo, "known-validation-gaps.yaml", "validate_failures: [hidden]\n")

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert any(
        "known-validation-gaps.yaml (index flag hides worktree changes)" in reason
        for reason in report["reasons"]
    ), report["reasons"]


def test_fsmonitor_valid_entries_cannot_hide_worktree_changes(repo):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    _git(repo, "config", "core.fsmonitor", "true")
    _write(repo, "known-validation-gaps.yaml", "validate_failures: [hidden]\n")
    _git(repo, "update-index", "--fsmonitor-valid", "known-validation-gaps.yaml")

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert any(
        "known-validation-gaps.yaml" in reason for reason in report["reasons"]
    ), report["reasons"]


def test_staged_change_restored_in_worktree_is_invalid(repo):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    original = (repo / "README.md").read_text()
    _write(repo, "README.md", "staged edit\n")
    _git(repo, "add", "README.md")
    _write(repo, "README.md", original)

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert any("README.md" in reason for reason in report["reasons"])


def test_subjurisdiction_citation_in_country_checkout_is_candidate(repo):
    base = _base(repo)
    _write(repo, "us-tn/statutes/1/2.yaml", _module_text("us-tn/statute/1/2"))
    _write(repo, "us-tn/statutes/1/2.test.yaml", "- name: case\n")
    _commit(repo, "subjurisdiction")

    report = _report(repo, base)

    assert report["status"] == "unadmitted_reviewed_candidate", report["reasons"]
    assert report["candidate"]["citation"] == "us-tn/statute/1/2"


def test_change_set_without_rulespec_is_not_classified_as_candidate(repo):
    base = _base(repo)
    _write(repo, "README.md", "docs only\n")
    _commit(repo, "docs")

    assert _report(repo, base)["status"] == "no_rulespec_change"


def test_report_does_no_corpus_engine_or_broker_work(repo, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("candidate status must stay cheap")

    for name in (
        "get_signing_broker",
        "load_rulespec_local_corpus_release",
        "load_rulespec_toolchain",
        "verify_rulespec_validation_waiver_set",
        "resolve_corpus_source_unit",
        "_load_applied_encoding_manifest_entries",
        "_guard_manifest_waiver_set_identity",
        "guard_generated_change_issues",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")

    assert _report(repo, base)["status"] == "unadmitted_reviewed_candidate"


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize(
    "change", ["candidate", "docs_only", "invalid_yaml", "nested_yaml"]
)
def test_guard_exit_code_and_issues_are_unchanged_by_flag(
    repo, tmp_path, capsys, as_json, change
):
    release = _bind_test_corpus_release(repo, tmp_path / "axiom-corpus")
    base = _commit(repo, "bind corpus release")
    if change == "candidate":
        _candidate(repo)
    elif change == "invalid_yaml":
        _write(repo, MODULE, "invalid: 2026-99-99\n")
        _write(repo, COMPANION, "- name: case\n")
    elif change == "nested_yaml":
        _write(repo, MODULE, "[" * 5000 + "]" * 5000 + "\n")
        _write(repo, COMPANION, "- name: case\n")
    else:
        _write(repo, "README.md", "docs only\n")
    _commit(repo, change)

    def run(candidate_status: bool) -> tuple[int, str]:
        args = SimpleNamespace(
            repo=repo,
            corpus_path=release.root,
            base_ref=base,
            head_ref="HEAD",
            json=as_json,
            all=False,
            candidate_status=candidate_status,
        )
        with pytest.raises(SystemExit) as exc_info:
            cli.cmd_guard_generated(args)
        return exc_info.value.code, capsys.readouterr().out

    plain_code, plain_out = run(False)
    flagged_code, flagged_out = run(True)

    assert flagged_code == plain_code == (0 if change == "docs_only" else 1)
    expected_status = {
        "candidate": "unadmitted_reviewed_candidate",
        "docs_only": "no_rulespec_change",
    }.get(change, "invalid")
    if as_json:
        plain, flagged = json.loads(plain_out), json.loads(flagged_out)
        report = flagged.pop("candidate_status")
        assert flagged == plain
    else:
        decoder = json.JSONDecoder()
        report, end = decoder.raw_decode(flagged_out)
        assert flagged_out[end:].lstrip("\n") == plain_out
    assert report["status"] == expected_status
    if change == "candidate":
        assert "Manual RuleSpec changes are not allowed." in plain_out or as_json
        assert f"{MODULE} changed without a matching" in plain_out
