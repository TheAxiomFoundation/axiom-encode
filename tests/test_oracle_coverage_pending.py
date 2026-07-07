"""Tests for the declared oracle-coverage pending lane.

Covers the report reclassification, the both-ways ratchet, the dispatcher
sync, and — the load-bearing guarantee — that an output declared in neither
the oracle mappings nor the pending file still fails the gate.
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest

from axiom_encode.oracles.policyengine.coverage import (
    build_policyengine_coverage_report,
)
from axiom_encode.oracles.policyengine.pending import (
    PENDING_STATUS,
    PendingDeclarationError,
    apply_pending_to_report,
    declarations_from_files,
    load_pending_files,
    parse_pending_payload,
    ratchet_problems,
    sync_repo_pending,
)

UNMAPPED = "uk:statutes/ukpga/9999/1/1#pending_test_only_output"
OTHER_UNMAPPED = "uk:statutes/ukpga/9999/1/1#pending_test_second_output"
MAPPED = "uk:statutes/ukpga/9999/1/1#pending_test_mapped_output"


def _report(*items: dict) -> dict:
    from collections import Counter

    return {
        "items": [dict(item) for item in items],
        "status_counts": dict(Counter(item["status"] for item in items)),
    }


def _entry(legal_id: str, source: str = "bulk", since: str = "2026-07-07") -> dict:
    return {"legal_id": legal_id, "source": source, "since": since}


def _pending_file(tmp_path: Path, *entries: dict, ceiling=None, repo="rulespec-uk"):
    payload: dict = {"version": 1, "entries": list(entries)}
    if ceiling is not None:
        payload["ceiling"] = ceiling
    return parse_pending_payload(
        payload, path=tmp_path / "oracle-coverage-pending.yaml", repo=repo
    )


# --- reclassification ------------------------------------------------------


def test_apply_reclassifies_declared_unmapped():
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"}
    )
    declared = declarations_from_files([_pending_file(Path("."), _entry(UNMAPPED))])
    summary = apply_pending_to_report(report, declared)
    assert report["items"][0]["status"] == PENDING_STATUS
    assert report["items"][0]["pending"]["source"] == "bulk"
    assert report["status_counts"] == {PENDING_STATUS: 1}
    assert summary["applied"] == [UNMAPPED]
    assert summary["stale"] == []


def test_apply_leaves_undeclared_unmapped():
    """Negative guarantee: an undeclared unmapped output stays unmapped."""
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"}
    )
    apply_pending_to_report(report, declarations_from_files([_pending_file(Path("."))]))
    assert report["items"][0]["status"] == "unmapped"
    assert report["status_counts"].get("unmapped") == 1


def test_apply_marks_stale_when_declared_but_already_classified():
    report = _report(
        {"legal_id": MAPPED, "status": "comparable", "file": "rulespec-uk/x"}
    )
    summary = apply_pending_to_report(
        report, declarations_from_files([_pending_file(Path("."), _entry(MAPPED))])
    )
    assert report["items"][0]["status"] == "comparable"  # untouched
    assert [row["legal_id"] for row in summary["stale"]] == [MAPPED]
    assert "already classified" in summary["stale"][0]["reason"]


def test_apply_marks_stale_when_output_absent():
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"}
    )
    summary = apply_pending_to_report(
        report,
        declarations_from_files(
            [_pending_file(Path("."), _entry(UNMAPPED), _entry(MAPPED))]
        ),
    )
    stale = {row["legal_id"] for row in summary["stale"]}
    assert stale == {MAPPED}
    assert "not found" in next(r["reason"] for r in summary["stale"])


# --- ratchet ---------------------------------------------------------------


def test_ratchet_flags_undeclared_stale_and_ceiling(tmp_path):
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"},
        {"legal_id": OTHER_UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"},
        {"legal_id": MAPPED, "status": "comparable", "file": "rulespec-uk/x"},
    )
    files = [
        _pending_file(
            tmp_path,
            _entry(UNMAPPED),  # valid
            _entry(MAPPED),  # stale (comparable)
            ceiling=1,  # 2 entries > ceiling 1
        )
    ]
    apply_pending_to_report(report, declarations_from_files(files))
    problems = ratchet_problems(report, files)
    joined = "\n".join(problems)
    assert OTHER_UNMAPPED in joined  # undeclared unmapped
    assert "remove it" in joined and MAPPED in joined  # stale
    assert "ceiling" in joined  # overflow


def test_ratchet_passes_on_exact_match(tmp_path):
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"}
    )
    files = [_pending_file(tmp_path, _entry(UNMAPPED), ceiling=1)]
    apply_pending_to_report(report, declarations_from_files(files))
    assert ratchet_problems(report, files) == []


def test_ratchet_repo_scope_ignores_other_repo(tmp_path):
    """A rulespec-uk check must not fail on rulespec-us's undeclared output."""
    report = _report(
        {"legal_id": UNMAPPED, "status": "unmapped", "file": "rulespec-uk/x"},
        {
            "legal_id": "us:statutes/x#y",
            "status": "unmapped",
            "file": "rulespec-us/x",
        },
    )
    files = [_pending_file(tmp_path, _entry(UNMAPPED))]
    apply_pending_to_report(report, declarations_from_files(files))
    # Unscoped: both repos' undeclared outputs surface.
    assert any("us:statutes/x#y" in p for p in ratchet_problems(report, files))
    # Scoped to rulespec-uk: only this repo's outputs are enforced.
    scoped = ratchet_problems(report, files, repo="rulespec-uk")
    assert scoped == []


# --- schema validation -----------------------------------------------------


@pytest.mark.parametrize(
    "payload, needle",
    [
        ({"version": 2, "entries": []}, "version: 1"),
        (
            {
                "version": 1,
                "entries": [{"legal_id": "x", "source": "nope", "since": "2026-07-07"}],
            },
            "source",
        ),
        (
            {
                "version": 1,
                "entries": [{"legal_id": "x", "source": "bulk", "since": "not-a-date"}],
            },
            "ISO date",
        ),
        (
            {"version": 1, "entries": [{"source": "bulk", "since": "2026-07-07"}]},
            "legal_id",
        ),
        ({"version": 1, "ceiling": -1, "entries": []}, "ceiling"),
        (
            {"version": 1, "entries": [_entry("dup"), _entry("dup")]},
            "duplicate",
        ),
    ],
)
def test_parse_rejects_malformed(payload, needle):
    with pytest.raises(PendingDeclarationError) as exc:
        parse_pending_payload(payload, path=Path("p.yaml"), repo="rulespec-uk")
    assert needle in str(exc.value)


def test_parse_accepts_yaml_date_object():
    """YAML parses an unquoted date into a datetime.date; accept it."""
    payload = {
        "version": 1,
        "entries": [
            {"legal_id": "x", "source": "bulk", "since": datetime.date(2026, 7, 7)}
        ],
    }
    parsed = parse_pending_payload(payload, path=Path("p.yaml"), repo="rulespec-uk")
    assert parsed.entries[0].since == "2026-07-07"


# --- dispatcher sync -------------------------------------------------------


def test_sync_writes_idempotently_and_drains(tmp_path):
    repo = tmp_path / "rulespec-uk"
    repo.mkdir()
    first = sync_repo_pending(
        repo_root=repo,
        repo_name="rulespec-uk",
        unmapped_legal_ids=[UNMAPPED, OTHER_UNMAPPED],
        source="bulk",
        since="2026-07-07",
    )
    assert first["count"] == 2 and first["changed"] is True and len(first["added"]) == 2
    # idempotent
    again = sync_repo_pending(
        repo_root=repo,
        repo_name="rulespec-uk",
        unmapped_legal_ids=[UNMAPPED, OTHER_UNMAPPED],
        source="bulk",
        since="2026-07-07",
    )
    assert again["changed"] is False
    # drain one: it disappears from the file
    drained = sync_repo_pending(
        repo_root=repo,
        repo_name="rulespec-uk",
        unmapped_legal_ids=[UNMAPPED],
        source="bulk",
        since="2026-07-08",
    )
    assert drained["dropped"] == [OTHER_UNMAPPED] and drained["count"] == 1
    files = load_pending_files(tmp_path)
    assert {e.legal_id for e in files[0].entries} == {UNMAPPED}
    # sync preserves the original since for entries that persist
    assert files[0].entries[0].since == "2026-07-07"


# --- registry-backed integration: the real gate path ----------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace_with_unmapped_output(tmp_path: Path) -> str:
    """Create a rulespec-uk checkout with one guaranteed-unmapped output."""
    _write(
        tmp_path / "rulespec-uk" / "uk/statutes/ukpga/9999/1/1.yaml",
        """format: rulespec/v1
rules:
  - name: pending_lane_integration_unmapped_output
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )
    return "uk:statutes/ukpga/9999/1/1#pending_lane_integration_unmapped_output"


def test_integration_undeclared_output_stays_unmapped(tmp_path):
    legal_id = _workspace_with_unmapped_output(tmp_path)
    report = build_policyengine_coverage_report(tmp_path)
    apply_pending_to_report(report, {})
    statuses = {it["legal_id"]: it["status"] for it in report["items"]}
    assert statuses[legal_id] == "unmapped"


def test_integration_declared_output_reclassified(tmp_path):
    legal_id = _workspace_with_unmapped_output(tmp_path)
    _write(
        tmp_path / "rulespec-uk" / "oracle-coverage-pending.yaml",
        f"""version: 1
entries:
  - legal_id: {legal_id}
    source: bulk
    since: 2026-07-07
""",
    )
    report = build_policyengine_coverage_report(tmp_path)
    apply_pending_to_report(
        report, declarations_from_files(load_pending_files(tmp_path))
    )
    statuses = {it["legal_id"]: it["status"] for it in report["items"]}
    assert statuses[legal_id] == PENDING_STATUS


def _run_cli(monkeypatch, *argv: str) -> int:
    from axiom_encode.cli import main

    monkeypatch.setattr(sys, "argv", ["axiom-encode", *argv])
    try:
        main()
    except SystemExit as exit_error:
        return int(exit_error.code or 0)
    return 0


def test_cli_gate_fails_on_undeclared_unmapped(tmp_path, monkeypatch):
    """End-to-end exit-code negative test through the real CLI."""
    _workspace_with_unmapped_output(tmp_path)
    code = _run_cli(
        monkeypatch, "oracle-coverage", "--root", str(tmp_path), "--fail-on-unmapped"
    )
    assert code == 1


def test_cli_gate_passes_when_declared(tmp_path, monkeypatch):
    legal_id = _workspace_with_unmapped_output(tmp_path)
    _write(
        tmp_path / "rulespec-uk" / "oracle-coverage-pending.yaml",
        f"version: 1\nentries:\n  - legal_id: {legal_id}\n    source: bulk\n    since: 2026-07-07\n",
    )
    code = _run_cli(
        monkeypatch,
        "oracle-coverage",
        "--root",
        str(tmp_path),
        "--fail-on-unmapped",
        "--fail-on-stale-pending",
    )
    assert code == 0
