"""Tests for the declared oracle-coverage pending lane.

Covers the report reclassification, the both-ways ratchet, the read-only
checkout boundary, and — the load-bearing guarantee — that an output declared in neither
the oracle mappings nor the pending file still fails the gate.
"""

from __future__ import annotations

import datetime
import stat
import sys
from pathlib import Path

import pytest
from axiom_oracles.bridges.coverage import (
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


def test_ratchet_has_no_cross_repo_scope_bypass(tmp_path):
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
    assert any("us:statutes/x#y" in p for p in ratchet_problems(report, files))


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


# --- exact checkout boundary ----------------------------------------------


def test_pending_loader_rejects_workspace_and_does_not_scan_siblings(tmp_path):
    checkout = tmp_path / "rulespec-uk"
    checkout.mkdir()
    _write(
        checkout / "oracle-coverage-pending.yaml",
        "version: 1\nentries: []\n",
    )

    with pytest.raises(PendingDeclarationError, match="exact canonical"):
        load_pending_files(tmp_path)

    assert [pending.path for pending in load_pending_files(checkout)] == [
        checkout / "oracle-coverage-pending.yaml"
    ]


def test_pending_loader_reads_exact_nested_github_actions_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    nested_checkout = checkout / "rulespec-us"
    nested_checkout.mkdir(parents=True)
    _write(
        nested_checkout / "oracle-coverage-pending.yaml",
        "version: 1\nentries: []\n",
    )

    assert [pending.path for pending in load_pending_files(checkout)] == [
        nested_checkout / "oracle-coverage-pending.yaml"
    ]


def test_pending_loader_rejects_ambiguous_direct_and_nested_files(tmp_path):
    checkout = tmp_path / "rulespec-us"
    nested_checkout = checkout / "rulespec-us"
    nested_checkout.mkdir(parents=True)
    for path in (
        checkout / "oracle-coverage-pending.yaml",
        nested_checkout / "oracle-coverage-pending.yaml",
    ):
        _write(path, "version: 1\nentries: []\n")

    with pytest.raises(PendingDeclarationError, match="ambiguous"):
        load_pending_files(checkout)


def test_pending_loader_rejects_symlinked_nested_checkout(tmp_path):
    checkout = tmp_path / "rulespec-us"
    checkout.mkdir()
    outside_checkout = tmp_path / "outside"
    outside_checkout.mkdir()
    (checkout / "rulespec-us").symlink_to(outside_checkout, target_is_directory=True)

    with pytest.raises(PendingDeclarationError, match="regular directory"):
        load_pending_files(checkout)


@pytest.mark.parametrize("nested", [False, True])
def test_pending_loader_rejects_symlinked_pending_file(tmp_path, nested):
    checkout = tmp_path / "rulespec-us"
    declaration_root = checkout / "rulespec-us" if nested else checkout
    declaration_root.mkdir(parents=True)
    outside_file = tmp_path / "outside-pending.yaml"
    _write(outside_file, "version: 1\nentries: []\n")
    (declaration_root / "oracle-coverage-pending.yaml").symlink_to(outside_file)

    with pytest.raises(PendingDeclarationError, match="regular file"):
        load_pending_files(checkout)


# --- registry-backed integration: the real gate path ----------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checkout_with_unmapped_output(tmp_path: Path) -> tuple[Path, str]:
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
    return (
        tmp_path / "rulespec-uk",
        "uk:statutes/ukpga/9999/1/1#pending_lane_integration_unmapped_output",
    )


def test_integration_undeclared_output_stays_unmapped(tmp_path):
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    report = build_policyengine_coverage_report(checkout)
    apply_pending_to_report(report, {})
    statuses = {it["legal_id"]: it["status"] for it in report["items"]}
    assert statuses[legal_id] == "unmapped"


def test_integration_declared_output_reclassified(tmp_path):
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    _write(
        tmp_path / "rulespec-uk" / "oracle-coverage-pending.yaml",
        f"""version: 1
entries:
  - legal_id: {legal_id}
    source: bulk
    since: 2026-07-07
""",
    )
    report = build_policyengine_coverage_report(checkout)
    apply_pending_to_report(
        report, declarations_from_files(load_pending_files(checkout))
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
    checkout, _legal_id = _checkout_with_unmapped_output(tmp_path)
    code = _run_cli(
        monkeypatch,
        "oracle-coverage",
        "--root",
        str(checkout),
        "--fail-on-unmapped",
    )
    assert code == 1


def test_cli_gate_passes_when_declared(tmp_path, monkeypatch):
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    _write(
        tmp_path / "rulespec-uk" / "oracle-coverage-pending.yaml",
        f"version: 1\nentries:\n  - legal_id: {legal_id}\n    source: bulk\n    since: 2026-07-07\n",
    )
    code = _run_cli(
        monkeypatch,
        "oracle-coverage",
        "--root",
        str(checkout),
        "--fail-on-unmapped",
        "--fail-on-stale-pending",
    )
    assert code == 0


def test_cli_gate_passes_with_nested_github_actions_checkout(tmp_path, monkeypatch):
    outer_checkout = tmp_path / "rulespec-uk"
    nested_checkout, legal_id = _checkout_with_unmapped_output(outer_checkout)
    _write(
        nested_checkout / "oracle-coverage-pending.yaml",
        f"version: 1\nentries:\n  - legal_id: {legal_id}\n    source: bulk\n    since: 2026-07-07\n",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage",
        "--root",
        str(outer_checkout),
        "--fail-on-unmapped",
        "--fail-on-stale-pending",
    )

    assert code == 0


def test_cli_pending_sync_declares_unmapped_output_idempotently(tmp_path, monkeypatch):
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )
    assert code == 0
    pending = load_pending_files(checkout)[0]
    assert [entry.legal_id for entry in pending.entries] == [legal_id]
    assert pending.ceiling == 1

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )
    assert code == 0
    assert load_pending_files(checkout)[0] == pending


def test_cli_pending_sync_excludes_nested_foreign_checkout(tmp_path, monkeypatch):
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    foreign_id = "us:statutes/9999/1#foreign_pending_output"
    _write(
        checkout / "rulespec-us" / "us/statutes/9999/1.yaml",
        """format: rulespec/v1
rules:
  - name: foreign_pending_output
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )

    assert code == 0
    declared = [entry.legal_id for entry in load_pending_files(checkout)[0].entries]
    assert declared == [legal_id]
    assert foreign_id not in declared


def test_cli_pending_sync_preserves_declared_legacy_root_output(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "rulespec-us"
    legal_id = "us-mo:manual/dss/snap/example#legacy_manual_output"
    _write(
        checkout / "us-mo/manual/dss/snap/example.yaml",
        """format: rulespec/v1
rules:
  - name: legacy_manual_output
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )
    _write(
        checkout / "oracle-coverage-pending.yaml",
        f"""version: 1
ceiling: 1
entries:
  - legal_id: {legal_id}
    source: manual
    since: 2026-07-07
""",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )

    assert code == 0
    assert [entry.legal_id for entry in load_pending_files(checkout)[0].entries] == [
        legal_id
    ]
    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage-pending",
            "check",
            "--root",
            str(checkout),
        )
        == 0
    )


def test_cli_pending_sync_includes_country_monorepo_state_output(tmp_path, monkeypatch):
    checkout = tmp_path / "rulespec-us"
    legal_id = "us-hi:statutes/235-54#individual_personal_exemption_deduction"
    _write(
        checkout / "us-hi" / "statutes/235-54.yaml",
        """format: rulespec/v1
rules:
  - name: individual_personal_exemption_deduction
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )

    assert code == 0
    declared = [entry.legal_id for entry in load_pending_files(checkout)[0].entries]
    assert declared == [legal_id]


def test_cli_pending_sync_targets_nested_actions_checkout(tmp_path, monkeypatch):
    outer_checkout = tmp_path / "rulespec-us"
    nested_checkout = outer_checkout / "rulespec-us"
    state_legal_id = "us-hi:statutes/235-54#individual_personal_exemption_deduction"
    program_legal_id = "us-zz:programs/example/fy-2099#brand_new_program_output_xyz"
    cross_country_program_id = "uk-zz:programs/example/fy-2099#cross_country_output_xyz"
    foreign_legal_id = "us:foreign/statutes/y#foreign_output_xyz"
    _write(
        nested_checkout / "us-hi" / "statutes/235-54.yaml",
        """format: rulespec/v1
rules:
  - name: individual_personal_exemption_deduction
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )
    _write(
        nested_checkout / "programs/us-zz/example/fy-2099.yaml",
        """program: us-zz/example
period: 2099
outputs:
  - brand_new_program_output_xyz
""",
    )
    _write(
        nested_checkout / "foreign/statutes/y.yaml",
        """format: rulespec/v1
rules:
  - name: foreign_output_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )
    _write(
        nested_checkout / "programs/uk-zz/example/fy-2099.yaml",
        """program: uk-zz/example
period: 2099
outputs:
  - cross_country_output_xyz
""",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(outer_checkout),
        "--source",
        "bulk",
    )

    assert code == 0
    pending = load_pending_files(outer_checkout)[0]
    assert pending.path.parent == nested_checkout
    assert [entry.legal_id for entry in pending.entries] == [
        state_legal_id,
        program_legal_id,
    ]
    assert foreign_legal_id not in {entry.legal_id for entry in pending.entries}
    assert cross_country_program_id not in {entry.legal_id for entry in pending.entries}


def test_cli_pending_sync_supports_direct_checkout_programs(tmp_path, monkeypatch):
    checkout = tmp_path / "rulespec-us"
    state_legal_id = "us-hi:statutes/235-54#direct_state_output_xyz"
    program_legal_id = "us-hi:programs/snap/fy-2099#direct_program_output_xyz"
    _write(
        checkout / "us-hi/statutes/235-54.yaml",
        """format: rulespec/v1
rules:
  - name: direct_state_output_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
    )
    _write(
        checkout / "programs/us-hi/snap/fy-2099.yaml",
        """program: us-hi/snap
period: 2099
outputs:
  - direct_program_output_xyz
""",
    )

    code = _run_cli(
        monkeypatch,
        "oracle-coverage-pending",
        "sync",
        "--root",
        str(checkout),
        "--source",
        "bulk",
    )

    assert code == 0
    assert [entry.legal_id for entry in load_pending_files(checkout)[0].entries] == [
        program_legal_id,
        state_legal_id,
    ]


def test_pending_sync_preserves_provenance_and_drains_fixed_entries(tmp_path):
    checkout = tmp_path / "rulespec-uk"
    checkout.mkdir()
    first = sync_repo_pending(
        repo_root=checkout,
        unmapped_legal_ids=[UNMAPPED, OTHER_UNMAPPED],
        source="manual",
        since="2026-07-01",
        issue="https://example.test/issues/1",
    )
    assert first["added"] == [UNMAPPED, OTHER_UNMAPPED]

    second = sync_repo_pending(
        repo_root=checkout,
        unmapped_legal_ids=[UNMAPPED],
        source="bulk",
        since="2026-07-13",
    )
    assert second["dropped"] == [OTHER_UNMAPPED]
    pending = load_pending_files(checkout)[0]
    assert pending.ceiling == 1
    assert pending.issue == "https://example.test/issues/1"
    assert pending.entries[0].source == "manual"
    assert pending.entries[0].since == "2026-07-01"


def test_pending_sync_quotes_yaml_sensitive_metadata_and_preserves_mode(tmp_path):
    checkout = tmp_path / "rulespec-uk"
    checkout.mkdir()
    path = checkout / "oracle-coverage-pending.yaml"
    path.write_text(
        "version: 1\n"
        'issue: "tracking: issue"\n'
        "ceiling: 1\n"
        "entries:\n"
        f"  - legal_id: {UNMAPPED}\n"
        "    source: manual\n"
        "    since: 2026-07-01\n"
        '    note: "blocked: needs mapping #42"\n',
        encoding="utf-8",
    )
    path.chmod(0o640)

    sync_repo_pending(
        repo_root=checkout,
        unmapped_legal_ids=[UNMAPPED],
        source="bulk",
        since="2026-07-13",
    )

    pending = load_pending_files(checkout)[0]
    assert pending.issue == "tracking: issue"
    assert pending.entries[0].note == "blocked: needs mapping #42"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_pending_sync_targets_nested_actions_checkout(tmp_path):
    outer = tmp_path / "rulespec-uk"
    nested = outer / "rulespec-uk"
    nested.mkdir(parents=True)

    result = sync_repo_pending(
        repo_root=outer,
        unmapped_legal_ids=[UNMAPPED],
        source="bulk",
        since="2026-07-13",
    )

    assert Path(result["path"]) == nested / "oracle-coverage-pending.yaml"
    assert not (outer / "oracle-coverage-pending.yaml").exists()
