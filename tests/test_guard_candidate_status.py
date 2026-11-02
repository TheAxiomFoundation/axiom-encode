"""guard-generated --candidate-status classifies change sets (#1628, B3)."""

import hashlib
import json
import os
import time
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


def _messages(report: dict) -> list[str]:
    return [reason["message"] for reason in report["reasons"]]


def _codes(report: dict) -> set[str]:
    return {reason["code"] for reason in report["reasons"]}


def _merge_key_bomb(levels: int) -> str:
    lines = ["a0: &a0 {x: 1}"]
    lines.extend(
        f"a{level}: &a{level} {{<<: [*a{level - 1}, *a{level - 1}]}}"
        for level in range(1, levels + 1)
    )
    lines.append(f"<<: *a{levels}")
    return "\n".join(lines) + "\n"


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
    ("case", "expected_code", "expected_reason"),
    [
        (
            "module_without_companion",
            "companion_missing",
            f"companion test {COMPANION} did not change",
        ),
        (
            "companion_without_module",
            "module_count",
            "expected exactly one RuleSpec module, found 0",
        ),
        (
            "two_modules",
            "module_count",
            "expected exactly one RuleSpec module, found 2",
        ),
        (
            "noncanonical_path",
            "module_path_noncanonical",
            "is not at the canonical path us/statutes/26/21.yaml",
        ),
        ("no_citation", "citation_missing", "declares no module.source_verification"),
        ("manifest_only", "manifest_only", "manifest-only change set"),
        ("metadata_change", "metadata_changed", "provenance metadata changed: .axiom/"),
        (
            "waiver_change",
            "waiver_changed",
            "validation waiver set changed: known-validation-gaps",
        ),
        (
            "other_file_change",
            "other_file_changed",
            "non-RuleSpec file changed: README.md",
        ),
        (
            "manifest_missing_for_module",
            "manifest_missing",
            f"without its matching manifest {MANIFEST}",
        ),
        ("deleted_module", "module_not_regular", f"{MODULE} is deleted at head"),
        (
            "deleted_manifest",
            "manifest_not_regular",
            f"manifest {MANIFEST} is deleted at head",
        ),
        (
            "symlink_manifest",
            "manifest_not_regular",
            f"manifest {MANIFEST} is not a regular 0644 file",
        ),
        (
            "symlink_module_with_manifest",
            "module_not_regular",
            f"{MODULE} is not a regular 0644 file (mode 120000) at head",
        ),
        (
            "executable_module_with_manifest",
            "module_not_regular",
            f"{MODULE} is not a regular 0644 file (mode 100755) at head",
        ),
        (
            "deleted_module_with_manifest",
            "module_not_regular",
            f"{MODULE} is deleted at head",
        ),
        (
            "deleted_companion_with_manifest",
            "module_not_regular",
            f"{COMPANION} is deleted at head",
        ),
        (
            "unmatched_manifest",
            "manifest_unmatched",
            "manifest .axiom/encoding-manifests/us/statutes/26/99.json changed "
            "without a change to its RuleSpec module or companion",
        ),
        (
            "waiver_only",
            "waiver_changed",
            "validation waiver set changed without a RuleSpec change",
        ),
        (
            "jurisdiction_tree_manifest_only",
            "metadata_changed",
            "provenance metadata changed without a RuleSpec change: us/.axiom/",
        ),
        ("double_slash_citation", "citation_noncanonical", "has no canonical path"),
        ("trailing_slash_citation", "citation_noncanonical", "has no canonical path"),
        ("plural_class_citation", "citation_noncanonical", "has no canonical path"),
        (
            "foreign_jurisdiction",
            "citation_outside_jurisdiction",
            "is outside the rulespec-us jurisdiction",
        ),
        (
            "invalid_yaml_date",
            "citation_missing",
            "declares no module.source_verification",
        ),
        (
            "deeply_nested_yaml",
            "citation_missing",
            "(YAML rejected: candidate YAML exceeds the 128-level limit)",
        ),
        (
            "merge_key_bomb",
            "citation_missing",
            "(YAML rejected: YAML merge keys are not permitted in RuleSpec scans)",
        ),
        (
            "duplicate_citation_key",
            "citation_missing",
            "declares no module.source_verification",
        ),
        (
            "case_alias_metadata_only",
            "metadata_changed",
            "provenance metadata changed without a RuleSpec change: .AXIOM/",
        ),
        (
            "case_alias_toolchain_only",
            "metadata_changed",
            "provenance metadata changed without a RuleSpec change: "
            ".AXIOM/toolchain.toml",
        ),
    ],
)
def test_invalid_change_sets_report_reasons(repo, case, expected_code, expected_reason):
    if case in {
        "manifest_only",
        "deleted_module",
        "deleted_manifest",
        "deleted_module_with_manifest",
        "deleted_companion_with_manifest",
    }:
        _candidate(repo)
        if case != "deleted_module":
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
    elif case == "symlink_module_with_manifest":
        _write(repo, "target.yaml", _module_text("us/statute/26/21"))
        (repo / MODULE).parent.mkdir(parents=True)
        (repo / MODULE).symlink_to(repo / "target.yaml")
        _write(repo, COMPANION, "- name: case\n")
        _write(repo, MANIFEST, "{}\n")
    elif case == "executable_module_with_manifest":
        _candidate(repo)
        (repo / MODULE).chmod(0o755)
        _write(repo, MANIFEST, "{}\n")
    elif case == "deleted_module_with_manifest":
        (repo / MODULE).unlink()
        _write(repo, MANIFEST, '{"changed": true}\n')
    elif case == "deleted_companion_with_manifest":
        (repo / COMPANION).unlink()
        _write(repo, MANIFEST, '{"changed": true}\n')
    elif case == "unmatched_manifest":
        _candidate(repo)
        _write(repo, MANIFEST, "{}\n")
        _write(repo, ".axiom/encoding-manifests/us/statutes/26/99.json", "{}\n")
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
    elif case == "case_alias_toolchain_only":
        _write(repo, ".AXIOM/toolchain.toml", "version = 1\n")
    elif case == "deeply_nested_yaml":
        _write(repo, MODULE, "[" * 5000 + "]" * 5000 + "\n")
        _write(repo, COMPANION, "- name: case\n")
    elif case == "merge_key_bomb":
        # Small enough that a regression fails fast instead of exhausting memory.
        _write(repo, MODULE, _merge_key_bomb(16))
        _write(repo, COMPANION, "- name: case\n")
    _commit(repo, case)

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert report["candidate"] is None
    matching = [
        reason for reason in report["reasons"] if expected_reason in reason["message"]
    ]
    assert matching, report["reasons"]
    assert {reason["code"] for reason in matching} == {expected_code}
    assert all(
        reason["category"] == cli.GUARD_CANDIDATE_REASON_CATEGORIES[reason["code"]]
        for reason in report["reasons"]
    )


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
    assert any("not a regular 0644 file" in message for message in _messages(report))


def test_uncommitted_change_and_missing_base_are_invalid(repo):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    (repo / "scratch.txt").write_text("untracked\n")

    dirty = _report(repo, base)
    no_base = cli.guard_candidate_status_report(repo, base_ref=None)

    assert dirty["status"] == "invalid"
    assert any("scratch.txt" in message for message in _messages(dirty))
    assert _codes(dirty) == {"checkout_dirty"}
    assert no_base["status"] == "invalid"
    assert no_base["reasons"] == [
        {
            "code": "base_ref_missing",
            "category": "environment",
            "message": "candidate status requires --base-ref",
        }
    ]


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
        "provenance metadata changed: .axiom/retired-source-metadata" in message
        for message in _messages(report)
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
        "provenance metadata changed: .axiom/vendor" in message
        for message in _messages(report)
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
        "known-validation-gaps.yaml (index flag hides worktree changes)" in message
        for message in _messages(report)
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
        "known-validation-gaps.yaml" in message for message in _messages(report)
    ), report["reasons"]


@pytest.mark.parametrize("rewrite", ["in_place", "new_inode"])
def test_minimal_stat_config_cannot_hide_same_size_edit(repo, rewrite):
    base = _base(repo)
    _candidate(repo)
    _commit(repo, "candidate")
    readme = repo / "README.md"
    past = time.time_ns() - 1000 * 10**9
    os.utime(readme, ns=(past, past))
    # The index now caches the old mtime and is newer than it, so Git trusts
    # the cached stat data instead of treating the entry as racily clean.
    _git(repo, "update-index", "--refresh")
    _git(repo, "config", "core.checkStat", "minimal")
    _git(repo, "config", "core.trustctime", "false")
    if rewrite == "in_place":
        time.sleep(1.1)  # the edit changes ctime in a later whole second
        readme.write_text("RULES\n")
    else:
        (repo / "README.next").write_text("RULES\n")
        os.replace(repo / "README.next", readme)
    os.utime(readme, ns=(past, past))
    # With the repository's own configuration, Git cannot see the edit.
    assert _git(repo, "diff", "--name-only", "HEAD").stdout == ""

    report = _report(repo, base)

    assert report["status"] == "invalid"
    assert report["candidate"] is None
    assert _codes(report) == {"checkout_dirty"}
    assert any("README.md" in message for message in _messages(report))


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
    assert any("README.md" in message for message in _messages(report))


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


def test_reason_codes_have_one_known_category():
    assert set(cli.GUARD_CANDIDATE_REASON_CATEGORIES.values()) == {
        "rulespec",
        "maintenance",
        "environment",
    }
    with pytest.raises(KeyError):
        cli._guard_candidate_reason([], "free_text", "unknown codes are refused")


def test_git_work_does_not_grow_with_changed_paths(repo, monkeypatch):
    calls: list[tuple[str, ...]] = []
    kept: list[set[str]] = []
    real_run, real_popen = cli.subprocess.run, cli.subprocess.Popen
    real_hash = cli._guard_candidate_hash_blobs

    def run(command, *args, **kwargs):
        calls.append(tuple(command))
        return real_run(command, *args, **kwargs)

    def popen(command, *args, **kwargs):
        calls.append(tuple(command))
        return real_popen(command, *args, **kwargs)

    def hash_blobs(repo_path, object_ids, *, keep):
        kept.append(set(keep))
        return real_hash(repo_path, object_ids, keep=keep)

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli, "_guard_candidate_hash_blobs", hash_blobs)

    def count(files: int) -> tuple[int, dict]:
        base = _base(repo)
        _candidate(repo)
        for index in range(files):
            _write(repo, f"docs/{files}/{index}.md", f"{index}\n")
        _commit(repo, f"{files} files")
        calls.clear()
        report = _report(repo, base)
        _git(repo, "reset", "-q", "--hard", base)
        return len(calls), report

    few_calls, few = count(1)
    many_calls, many = count(40)

    assert len(few["changed_files"]) == 3
    assert len(many["changed_files"]) == 42
    assert all(entry["sha256"] for entry in many["changed_files"])
    assert few_calls == many_calls
    module_entry = next(e for e in many["changed_files"] if e["path"] == MODULE)
    assert (
        module_entry["sha256"]
        == hashlib.sha256(_module_text("us/statute/26/21").encode()).hexdigest()
    )
    # Only the single module's blob is kept in memory for parsing.
    assert [len(keep) for keep in kept] == [1, 1]


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
    "change",
    ["candidate", "docs_only", "invalid_yaml", "nested_yaml", "merge_key_bomb"],
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
    elif change == "merge_key_bomb":
        # Canary: a regression fails here, before the 1 KB module below can
        # exhaust memory (expanding its merge keys would need tens of GB).
        assert cli._guard_candidate_citation(_merge_key_bomb(16).encode(), MODULE) == (
            None,
            "YAML merge keys are not permitted in RuleSpec scans",
        )
        _write(repo, MODULE, _merge_key_bomb(30))
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
    if change == "merge_key_bomb":
        assert {reason["code"] for reason in report["reasons"]} == {"citation_missing"}
        assert "YAML merge keys are not permitted" in report["reasons"][0]["message"]
    if change == "candidate":
        assert "Manual RuleSpec changes are not allowed." in plain_out or as_json
        assert f"{MODULE} changed without a matching" in plain_out


@pytest.mark.parametrize("as_json", [False, True])
def test_report_failure_keeps_guard_output_and_report_shape(
    repo, tmp_path, capsys, monkeypatch, as_json
):
    release = _bind_test_corpus_release(repo, tmp_path / "axiom-corpus")
    base = _commit(repo, "bind corpus release")
    _candidate(repo)
    _commit(repo, "candidate")
    normal = _report(repo, base)

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

    def broken(*_args, **_kwargs):
        raise LookupError("report bug")

    plain_code, plain_out = run(False)
    monkeypatch.setattr(cli, "guard_candidate_status_report", broken)
    flagged_code, flagged_out = run(True)

    assert flagged_code == plain_code == 1
    if as_json:
        plain, flagged = json.loads(plain_out), json.loads(flagged_out)
        report = flagged.pop("candidate_status")
        assert flagged == plain
    else:
        report, end = json.JSONDecoder().raw_decode(flagged_out)
        assert flagged_out[end:].lstrip("\n") == plain_out
    assert set(report) == set(normal)
    assert (report["status"], report["repository"], report["country"]) == (
        "invalid",
        "rulespec-us",
        "us",
    )
    assert report["reasons"] == [
        {
            "code": "report_failed",
            "category": "environment",
            "message": "candidate status failed: LookupError: report bug",
        }
    ]
