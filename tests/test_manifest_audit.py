"""Tests for the committed-manifest audit (axiom-encode#1282)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from axiom_encode import manifest_audit
from axiom_encode.manifest_audit import (
    AMBIGUOUS,
    DUPLICATE,
    LEDGER_INVALID,
    LEDGER_UNUSED,
    MISMATCH,
    MISSING,
    RESURRECTED,
    UNREADABLE,
    LedgerSchemaError,
    audit_repository,
    load_ledger,
    plan_prune,
)

MANIFEST_DIR = ".axiom/encoding-manifests"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_rule(repo: Path, relpath: str, content: str) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return sha256(content)


def write_manifest(
    repo: Path,
    relpath: str,
    applied: list[dict[str, object]],
    *,
    generated_at: str = "2026-06-01T00:00:00+00:00",
) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "axiom-encode/applied-rulespec/v5",
                "generated_at": generated_at,
                "applied_files": applied,
                "backend": "codex",
            },
            indent=2,
        )
        + "\n"
    )
    return relpath


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A country monorepo with one jurisdiction root and both manifest trees."""
    (tmp_path / "us" / "statutes").mkdir(parents=True)
    return tmp_path


def test_clean_repo_passes(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    result = audit_repository(repo)
    assert result.passed, result.findings
    assert result.matched == 1
    assert result.manifests == 1


def test_country_relative_key_under_root_tree_resolves(repo: Path) -> None:
    """The pre-#1078 checkout-root record keys files country-relative.

    Reading those repo-relative instead reported 828 phantom missing files on
    rulespec-us; the jurisdiction fallback is what makes the audit truthful.
    """
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]


def test_repo_relative_key_under_root_tree_resolves(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": digest}],
    )
    assert audit_repository(repo).passed


def test_mismatch_is_reported(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [changed]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: []\n")}],
    )
    result = audit_repository(repo)
    assert not result.passed
    assert [f.kind for f in result.findings] == [MISMATCH]


def test_stale_duplicate_is_reported_even_when_a_good_record_exists(repo: Path) -> None:
    """The #1282 defect: one matching record must not excuse a contradictory one."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    result = audit_repository(repo)
    kinds = sorted(f.kind for f in result.findings)
    assert MISMATCH in kinds
    assert DUPLICATE in kinds


def test_agreeing_duplicate_still_violates_uniqueness(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [DUPLICATE]


def test_one_manifest_listing_a_path_twice_is_not_a_duplicate(repo: Path) -> None:
    """Uniqueness is over records, not entries — rulespec-us has such a manifest."""
    digest = write_rule(repo, "us/statutes/26/1401.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/1401.json",
        [
            {"path": "statutes/26/1401.yaml", "sha256": digest},
            {"path": "statutes/26/1401.yaml", "sha256": digest},
        ],
    )
    assert audit_repository(repo).passed


def test_missing_file_is_reported(repo: Path) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("gone")}],
    )
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [MISSING]


def test_deletion_marker_matches_an_absent_file(repo: Path) -> None:
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    result = audit_repository(repo)
    assert result.passed
    assert result.matched == 1


def test_resurrected_file_is_reported(repo: Path) -> None:
    write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "deleted": True}],
    )
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [RESURRECTED]


def test_ambiguous_key_is_reported_not_guessed(repo: Path) -> None:
    (repo / "us-co" / "statutes").mkdir(parents=True)
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [a]\n")
    write_rule(repo, "us-co/statutes/26/32.yaml", "rules: [b]\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [a]\n")}],
    )
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [AMBIGUOUS]


def test_unreadable_manifest_is_reported(repo: Path) -> None:
    path = repo / MANIFEST_DIR / "broken.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [UNREADABLE]


def test_non_manifest_json_is_ignored(repo: Path) -> None:
    path = repo / MANIFEST_DIR / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": "not an apply manifest"}))
    assert audit_repository(repo).passed


# --------------------------------------------------------------------------
# Supersession ledger
# --------------------------------------------------------------------------


def write_ledger(repo: Path, entries: list[dict[str, object]]) -> None:
    import yaml

    path = repo / manifest_audit.LEDGER_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": manifest_audit.LEDGER_SCHEMA,
                "superseded": entries,
            },
            sort_keys=False,
        )
    )


def _superseded_fixture(repo: Path) -> tuple[str, str]:
    """A legacy record left stale by a re-generation, plus its successor."""
    current = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    successor = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": current}],
    )
    return stale, successor


def test_superseded_entry_clears_the_finding(repo: Path) -> None:
    stale, successor = _superseded_fixture(repo)
    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "statutes/26/32.yaml",
                "attested_sha256": sha256("rules: [old]\n"),
                "superseded_by": successor,
                "reason": "regenerated after the #1078 relocation",
            }
        ],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]
    assert result.disclosed == 1


def test_superseded_entry_rejected_when_successor_absent(repo: Path) -> None:
    stale, successor = _superseded_fixture(repo)
    (repo / successor).unlink()
    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "statutes/26/32.yaml",
                "attested_sha256": sha256("rules: [old]\n"),
                "superseded_by": successor,
                "reason": "regenerated",
            }
        ],
    )
    result = audit_repository(repo)
    assert any(f.kind == LEDGER_INVALID for f in result.findings)


def test_superseded_entry_rejected_when_successor_disagrees(repo: Path) -> None:
    """A ledger entry cannot launder a file nobody attests correctly."""
    stale, successor = _superseded_fixture(repo)
    (repo / "us/statutes/26/32.yaml").write_text("rules: [tampered]\n")
    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "statutes/26/32.yaml",
                "attested_sha256": sha256("rules: [old]\n"),
                "superseded_by": successor,
                "reason": "regenerated",
            }
        ],
    )
    result = audit_repository(repo)
    assert any(
        f.kind == LEDGER_INVALID and "current content" in f.detail
        for f in result.findings
    ), [str(f) for f in result.findings]


def test_entry_rejected_when_attested_hash_does_not_match_the_manifest(
    repo: Path,
) -> None:
    stale, successor = _superseded_fixture(repo)
    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "statutes/26/32.yaml",
                "attested_sha256": sha256("something else"),
                "superseded_by": successor,
                "reason": "regenerated",
            }
        ],
    )
    result = audit_repository(repo)
    assert any(
        f.kind == LEDGER_INVALID and "actual claim" in f.detail for f in result.findings
    )


def test_unused_entry_is_reported(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    stale = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "statutes/26/32.yaml",
                "attested_sha256": sha256("rules: [old]\n"),
                "superseded_by": stale,
                "reason": "obsolete disclosure",
            }
        ],
    )
    result = audit_repository(repo)
    assert [f.kind for f in result.findings] == [LEDGER_UNUSED]


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"manifest": "m.json"}, "missing keys"),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
            },
            "exactly one of superseded_by or retired_in",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "nothex",
                "reason": "r",
                "superseded_by": "n.json",
            },
            "sha256 hex digest",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
                "superseded_by": "n.json",
                "retired_in": ["a" * 40],
            },
            "exactly one of superseded_by or retired_in",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
                "retired_in": ["short"],
            },
            "40-character commit sha1",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
                "retired_in": ["a" * 40],
                "current_path": "b.yaml",
            },
            "must be given together",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
                "superseded_by": "n.json",
                "current_path": "b.yaml",
                "current_sha256": "0" * 64,
            },
            "belong to retired_in entries",
        ),
        (
            {
                "manifest": "m.json",
                "attested_path": "a.yaml",
                "attested_sha256": "0" * 64,
                "reason": "r",
                "superseded_by": "n.json",
                "surprise": 1,
            },
            "unknown keys",
        ),
    ],
)
def test_ledger_schema_is_strict(repo: Path, entry: dict, expected: str) -> None:
    write_ledger(repo, [entry])
    with pytest.raises(LedgerSchemaError, match=expected):
        load_ledger(repo)


def test_ledger_rejects_duplicate_entries(repo: Path) -> None:
    entry = {
        "manifest": "m.json",
        "attested_path": "a.yaml",
        "attested_sha256": "0" * 64,
        "reason": "r",
        "superseded_by": "n.json",
    }
    write_ledger(repo, [entry, dict(entry)])
    with pytest.raises(LedgerSchemaError, match="duplicate entry"):
        load_ledger(repo)


def test_absent_ledger_is_not_an_error(repo: Path) -> None:
    assert load_ledger(repo) == {}


# --------------------------------------------------------------------------
# retired_in entries verify against real git history
# --------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def git_repo(repo: Path) -> Path:
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def test_retired_in_entry_verifies_against_history(git_repo: Path) -> None:
    repo = git_repo
    original = "rules: [original]\n"
    write_rule(repo, "us/statutes/26/32.yaml", original)
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": sha256(original)}],
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "encode")

    migrated = "rules: [migrated]\n"
    (repo / "us/statutes/26/32.yaml").write_text(migrated)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "mechanical migration")
    commit = _git(repo, "rev-parse", "HEAD")

    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "us/statutes/26/32.yaml",
                "attested_sha256": sha256(original),
                "retired_in": [commit],
                "current_path": "us/statutes/26/32.yaml",
                "current_sha256": sha256(migrated),
                "reason": "schema migration",
            }
        ],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]
    assert result.disclosed == 1


def test_retired_in_entry_rejected_when_history_does_not_support_it(
    git_repo: Path,
) -> None:
    """A commit that never held the attested content cannot excuse the claim."""
    repo = git_repo
    write_rule(repo, "us/statutes/26/32.yaml", "rules: [only ever this]\n")
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": sha256("never committed\n")}],
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "first")
    (repo / "unrelated.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    commit = _git(repo, "rev-parse", "HEAD")

    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "us/statutes/26/32.yaml",
                "attested_sha256": sha256("never committed\n"),
                "retired_in": [commit],
                "current_path": "us/statutes/26/32.yaml",
                "current_sha256": sha256("rules: [only ever this]\n"),
                "reason": "unsupported claim",
            }
        ],
    )
    result = audit_repository(repo)
    assert any(
        f.kind == LEDGER_INVALID
        and "does not begin at the attested content" in f.detail
        for f in result.findings
    ), [str(f) for f in result.findings]


def test_retired_in_entry_covers_a_relocation(git_repo: Path) -> None:
    repo = git_repo
    content = "spec: {}\n"
    write_rule(repo, "programs/gh/income-tax/fy-2026.yaml", content)
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/gh/programs/gh/income-tax/fy-2026.json",
        [{"path": "programs/gh/income-tax/fy-2026.yaml", "sha256": sha256(content)}],
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "encode programspec")

    (repo / "gh" / "programs" / "income-tax").mkdir(parents=True)
    (repo / "programs/gh/income-tax/fy-2026.yaml").rename(
        repo / "gh/programs/income-tax/fy-2026.yaml"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "move composition specs under the jurisdiction root")
    commit = _git(repo, "rev-parse", "HEAD")

    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "programs/gh/income-tax/fy-2026.yaml",
                "attested_sha256": sha256(content),
                "retired_in": [commit],
                "current_path": "gh/programs/income-tax/fy-2026.yaml",
                "current_sha256": sha256(content),
                "reason": "relocated; content unchanged",
            }
        ],
    )
    result = audit_repository(repo)
    assert result.passed, [str(f) for f in result.findings]


def test_retired_in_entry_rejected_when_current_content_moved_again(
    git_repo: Path,
) -> None:
    repo = git_repo
    original = "rules: [original]\n"
    write_rule(repo, "us/statutes/26/32.yaml", original)
    stale = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [{"path": "us/statutes/26/32.yaml", "sha256": sha256(original)}],
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "encode")
    (repo / "us/statutes/26/32.yaml").write_text("rules: [migrated]\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "migrate")
    commit = _git(repo, "rev-parse", "HEAD")
    # Drift again after the ledger entry was written.
    (repo / "us/statutes/26/32.yaml").write_text("rules: [drifted again]\n")

    write_ledger(
        repo,
        [
            {
                "manifest": stale,
                "attested_path": "us/statutes/26/32.yaml",
                "attested_sha256": sha256(original),
                "retired_in": [commit],
                "current_path": "us/statutes/26/32.yaml",
                "current_sha256": sha256("rules: [migrated]\n"),
                "reason": "schema migration",
            }
        ],
    )
    result = audit_repository(repo)
    assert any(
        f.kind == LEDGER_INVALID and "current_sha256 does not match" in f.detail
        for f in result.findings
    ), [str(f) for f in result.findings]


# --------------------------------------------------------------------------
# Prune-on-supersede
# --------------------------------------------------------------------------


def test_plan_prune_retires_a_fully_superseded_record(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == (old,)
    assert plan.disclose == ()


def test_plan_prune_discloses_a_partly_superseded_record(repo: Path) -> None:
    """The rulespec-us case: the old record still covers a companion test."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    old = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [
            {"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")},
            {"path": "statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.retire == ()
    assert plan.disclose == ((old, "us/statutes/26/32.yaml"),)


def test_plan_prune_ignores_unrelated_records(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    other = write_rule(repo, "us/statutes/26/24.yaml", "rules: []\n")
    write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/24.json",
        [{"path": "statutes/26/24.yaml", "sha256": other}],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = plan_prune(repo, new_manifest=new, attested_paths=["us/statutes/26/32.yaml"])
    assert plan.empty


def test_apply_prune_deletes_and_discloses(repo: Path) -> None:
    """The write-path fix: a re-generation must not leave a second record."""
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: [new]\n")
    write_rule(repo, "us/statutes/26/32.test.yaml", "cases: []\n")
    fully = write_manifest(
        repo,
        f"{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": sha256("rules: [old]\n")}],
    )
    partly = write_manifest(
        repo,
        f"{MANIFEST_DIR}/us/statutes/26/32.json",
        [
            {"path": "us/statutes/26/32.yaml", "sha256": sha256("rules: [older]\n")},
            {"path": "us/statutes/26/32.test.yaml", "sha256": sha256("cases: []\n")},
        ],
    )
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )

    plan = manifest_audit.apply_prune(
        repo,
        new_manifest=new,
        attested_paths=["us/statutes/26/32.yaml"],
        reason="superseded by a re-generation",
    )
    assert plan.retire == (fully,)
    assert plan.disclose == ((partly, "us/statutes/26/32.yaml"),)
    assert not (repo / fully).exists()
    assert (repo / partly).exists()

    ledger = load_ledger(repo)
    assert list(ledger) == [(partly, "us/statutes/26/32.yaml")]
    assert audit_repository(repo).passed, [
        str(f) for f in audit_repository(repo).findings
    ]


def test_apply_prune_is_a_no_op_without_predecessors(repo: Path) -> None:
    digest = write_rule(repo, "us/statutes/26/32.yaml", "rules: []\n")
    new = write_manifest(
        repo,
        f"us/{MANIFEST_DIR}/statutes/26/32.json",
        [{"path": "statutes/26/32.yaml", "sha256": digest}],
    )
    plan = manifest_audit.apply_prune(
        repo,
        new_manifest=new,
        attested_paths=["us/statutes/26/32.yaml"],
        reason="none",
    )
    assert plan.empty
    assert not (repo / manifest_audit.LEDGER_RELATIVE_PATH).exists()
