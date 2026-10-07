"""Tests for the declared oracle-coverage pending lane.

Covers the report reclassification, the both-ways ratchet, the read-only
checkout boundary, and — the load-bearing guarantee — that an output declared in neither
the oracle mappings nor the pending file still fails the gate.
"""

from __future__ import annotations

import copy
import datetime
import json
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest
from axiom_oracles.bridges.coverage import (
    build_policyengine_coverage_report,
)
from hypothesis import given, settings
from hypothesis import strategies as st

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
from axiom_encode.repo_routing import is_composition_policy_repo_root

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


def _name_shaped_zero_output_root(tmp_path: Path, kind: str) -> Path:
    """Build a ``rulespec-<country>``-named directory that clears the name gate
    but classifies to zero executable outputs.

    ``is_composition_policy_repo_root`` accepts a directory by name when no
    ``.git`` boundary sits at or above it, so each of these clears it — the
    shapes the auditor showed still produced ``total_outputs: 0`` with exit 0
    even behind a filesystem-evidence guard (axiom-encode#1113).
    """
    root = tmp_path / kind / "rulespec-dk"
    root.mkdir(parents=True)
    if kind == "empty_axiom_dir":
        (root / ".axiom").mkdir()
    elif kind == "axiom_file":
        (root / ".axiom").write_text("", encoding="utf-8")
    elif kind == "empty_pending_ledger":
        _write(root / "oracle-coverage-pending.yaml", "version: 1\nentries: []\n")
    elif kind == "empty_jurisdiction_dir":
        (root / "dk").mkdir()
    elif kind == "fake_nested_git":
        (root / "rulespec-dk" / ".git").mkdir(parents=True)
    else:  # pragma: no cover - guard against a typo in the parametrization
        raise AssertionError(f"unknown shape {kind!r}")
    return root


ZERO_OUTPUT_SHAPES = [
    "empty_axiom_dir",
    "axiom_file",
    "empty_pending_ledger",
    "empty_jurisdiction_dir",
    "fake_nested_git",
]


@pytest.mark.parametrize("kind", ZERO_OUTPUT_SHAPES)
def test_cli_gate_fail_on_empty_rejects_zero_output_shapes(tmp_path, monkeypatch, kind):
    """``--fail-on-empty`` fails loudly on every name-shaped root that classifies
    to zero outputs — the fix for the axiom-encode#1113 vacuous pass.

    Each shape clears the ``is_composition_policy_repo_root`` name gate, so a
    filesystem-evidence guard was satisfiable: an empty ``.axiom/``, an
    ``.axiom`` file, an empty pending ledger, an empty jurisdiction directory,
    and a fake nested ``.git`` all still produced ``total_outputs: 0`` with exit
    0 under ``--fail-on-unmapped``. The fix gates on the *report*, not on
    filesystem evidence: with ``--fail-on-empty`` a zero-output report is a loud
    failure, while without it the same report still passes so a legitimately
    empty new repo is not broken.
    """
    root = _name_shaped_zero_output_root(tmp_path, kind)
    assert is_composition_policy_repo_root(root)  # the name gate accepts it

    assert (
        _run_cli(
            monkeypatch, "oracle-coverage", "--root", str(root), "--fail-on-unmapped"
        )
        == 0
    )

    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage",
            "--root",
            str(root),
            "--fail-on-unmapped",
            "--fail-on-empty",
        )
        != 0
    )


def test_cli_gate_empty_repo_passes_without_fail_on_empty(tmp_path, monkeypatch):
    """A legitimately-empty new jurisdiction repo must pass its coverage gate.

    rulespec-tz classified to zero outputs on its push-to-main coverage gate
    (``--fail-on-unmapped --fail-on-untested-comparable``) this week and passed;
    that is why the zero-output rejection is opt-in via ``--fail-on-empty`` and
    not the default under ``--fail-on-unmapped``.
    """
    checkout = tmp_path / "rulespec-tz"
    _write(checkout / ".axiom/known-validation-gaps.yaml", "version: 1\n")

    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage",
            "--root",
            str(checkout),
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
        )
        == 0
    )


def test_cli_gate_fail_on_empty_passes_with_declared_outputs(tmp_path, monkeypatch):
    """``--fail-on-empty`` does not false-fail a checkout that has outputs: a
    declared-pending output leaves a non-empty report with zero unmapped."""
    checkout, legal_id = _checkout_with_unmapped_output(tmp_path)
    _write(
        checkout / "oracle-coverage-pending.yaml",
        f"version: 1\nentries:\n  - legal_id: {legal_id}\n    source: bulk\n    since: 2026-07-07\n",
    )

    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage",
            "--root",
            str(checkout),
            "--fail-on-unmapped",
            "--fail-on-stale-pending",
            "--fail-on-empty",
        )
        == 0
    )


def test_cli_gate_fail_on_empty_rejects_empty_root_with_program_surfaces(
    tmp_path, monkeypatch
):
    """Global program surfaces must not vouch for root content.

    ``--include-program-surfaces`` attaches the fixed PolicyEngine program-surface
    manifest (~189 surfaces, independent of ``--root``), so an empty root reports
    ``total_outputs: 0`` with a nonzero surface count. ``--fail-on-empty`` must
    still fail — the emptiness decision keys off the root's executable outputs
    only — or global surfaces reopen the axiom-encode#1113 vacuous pass. This is
    the zero-output / nonzero-surface case.
    """
    root = tmp_path / "rulespec-dk"
    (root / ".axiom").mkdir(parents=True)
    assert is_composition_policy_repo_root(root)  # cleared the name gate

    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage",
            "--root",
            str(root),
            "--include-program-surfaces",
            "--fail-on-empty",
        )
        != 0
    )


def test_cli_oracle_coverage_accepts_programspec_only_tree(
    tmp_path, monkeypatch, capsys
):
    """A supported non-Git, ProgramSpec-only tree is accepted and classifies its
    own executable outputs — not rejected as a non-checkout.

    Its ``outputs:`` list yields executable outputs (``total_outputs > 0``), so it
    is not a zero-output shape: ``--fail-on-empty`` passes on the tree's real
    content, independently of the global program-surface manifest. This is the
    nonzero-output counterpart to the zero-output shapes above.
    """
    checkout = tmp_path / "rulespec-dk"
    _write(
        checkout / "programs/dk-zz/example/fy-2099.yaml",
        "program: dk-zz/example\nperiod: 2099\noutputs:\n  - programspec_only_output_xyz\n",
    )

    # Accepted and classified, not rejected as a non-checkout, and it produced
    # real executable outputs (so it is genuinely non-empty).
    assert (
        _run_cli(monkeypatch, "oracle-coverage", "--root", str(checkout), "--json") == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["total_outputs"] > 0

    # --fail-on-empty passes on the tree's own outputs (no --include-program-surfaces).
    assert (
        _run_cli(
            monkeypatch,
            "oracle-coverage",
            "--root",
            str(checkout),
            "--fail-on-empty",
        )
        == 0
    )


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


def test_cli_pending_sync_preserves_declared_legacy_root_output(tmp_path, monkeypatch):
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


def test_cli_pending_sync_drops_declared_nested_same_country_checkout(
    tmp_path, monkeypatch
):
    checkout = tmp_path / "rulespec-us"
    legal_id = "us-mo:policies/demo/x#nested_checkout_output"
    _write(
        checkout / "us/statutes/placeholder.yaml",
        """format: rulespec/v1
rules: []
""",
    )
    _write(
        checkout / "rulespec-us-mo/policies/demo/x.yaml",
        """format: rulespec/v1
rules:
  - name: nested_checkout_output
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
    assert load_pending_files(checkout)[0].entries == ()


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


# --- status aggregates after reclassification ----------------------------
#
# Invariant: the report's per-repo breakdown partitions its items, so after
# ``apply_pending_to_report`` the sum over ``repos[].status_counts`` equals the
# top-level ``status_counts``, each row's ``total_outputs`` equals its own
# status sum, and the rows' totals sum to ``total_outputs``. Reclassification
# moves outputs from ``unmapped`` to ``pending_classification`` and changes no
# other aggregate.

_REPO_AGGREGATE_KEYS = (
    "total_outputs",
    "status_counts",
    "untested_comparable",
    "program_counts",
    "repos",
)


def _builder_shaped_report(items: list[dict]) -> dict:
    """Aggregate ``items`` the way ``build_policyengine_coverage_report`` does.

    ``test_builder_shaped_report_matches_real_builder`` pins this helper to the
    real builder, so the property test runs on report shapes it can produce.
    """
    items = sorted((dict(item) for item in items), key=lambda it: it["legal_id"])
    repo_counts: dict[str, Counter] = {}
    for item in items:
        repo_counts.setdefault(item["repo"], Counter())[item["status"]] += 1
    return {
        "total_outputs": len(items),
        "status_counts": dict(sorted(Counter(it["status"] for it in items).items())),
        "untested_comparable": sum(
            1 for it in items if it["status"] == "comparable" and not it["tested"]
        ),
        "program_counts": dict(sorted(Counter(it["program"] for it in items).items())),
        "repos": [
            {
                "repo": repo,
                "total_outputs": sum(counter.values()),
                "status_counts": dict(sorted(counter.items())),
            }
            for repo, counter in sorted(repo_counts.items())
        ],
        "items": items,
    }


def _assert_repo_rows_partition_report(report: dict) -> None:
    rows = report["repos"]
    summed = Counter()
    for row in rows:
        summed.update(row["status_counts"])
        assert row["total_outputs"] == sum(row["status_counts"].values()), row
    assert dict(summed) == report["status_counts"]
    assert sum(row["total_outputs"] for row in rows) == report["total_outputs"]
    assert report["status_counts"] == dict(
        Counter(item["status"] for item in report["items"])
    )
    assert [row["repo"] for row in rows] == sorted(row["repo"] for row in rows)


def _item(legal_id: str, repo: str, status: str, **extra) -> dict:
    return {
        "legal_id": legal_id,
        "repo": repo,
        "status": status,
        "program": extra.pop("program", "tax"),
        "tested": extra.pop("tested", False),
        "file": f"{repo}/x.yaml",
        **extra,
    }


def test_apply_recounts_repo_rows_for_declared_outputs():
    """Regression: per-repo rows used to keep the pre-reclassification counts.

    rulespec-et on 2026-09-28 reported top-level
    ``{pending_classification: 49, unmapped: 5}`` beside the row
    ``rulespec-et {unmapped: 54}``.
    """
    report = _builder_shaped_report(
        [
            _item("et:a#one", "rulespec-et", "unmapped"),
            _item("et:a#two", "rulespec-et", "unmapped"),
            _item("et:a#three", "rulespec-et", "comparable", tested=True),
            _item("et-aa:b#four", "rulespec-et-aa", "unmapped"),
        ]
    )
    declared = declarations_from_files(
        [_pending_file(Path("."), _entry("et:a#one"), _entry("et-aa:b#four"), repo="x")]
    )

    apply_pending_to_report(report, declared)

    assert report["status_counts"] == {
        "comparable": 1,
        PENDING_STATUS: 2,
        "unmapped": 1,
    }
    assert report["repos"] == [
        {
            "repo": "rulespec-et",
            "total_outputs": 3,
            "status_counts": {"comparable": 1, PENDING_STATUS: 1, "unmapped": 1},
        },
        {
            "repo": "rulespec-et-aa",
            "total_outputs": 1,
            "status_counts": {PENDING_STATUS: 1},
        },
    ]
    _assert_repo_rows_partition_report(report)


_FEDERAL_UNMAPPED = "us:statutes/26/9999#brand_new_federal_helper_xyz"
_FEDERAL_UNDECLARED = "us:statutes/26/9998#brand_new_federal_other_xyz"


def _us_checkout_with_outputs_in_three_repos(tmp_path: Path) -> Path:
    """A rulespec-us checkout whose items land in three ``repos`` rows."""
    checkout = tmp_path / "rulespec-us"
    for relative, name in [
        ("us/statutes/26/9999.yaml", "brand_new_federal_helper_xyz"),
        ("us/statutes/26/9998.yaml", "brand_new_federal_other_xyz"),
        ("us-al/statutes/40/18/9999.yaml", "brand_new_state_helper_xyz"),
        ("us-ks/statutes/79/9999.yaml", "brand_new_ks_helper_xyz"),
    ]:
        _write(
            checkout / relative,
            f"""format: rulespec/v1
rules:
  - name: {name}
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: some_input
""",
        )
    return checkout


def _declare(checkout: Path, *legal_ids: str) -> None:
    _write(
        checkout / "oracle-coverage-pending.yaml",
        "version: 1\nentries:\n"
        + "".join(
            f"  - legal_id: '{legal_id}'\n    source: bulk\n    since: 2026-09-28\n"
            for legal_id in legal_ids
        ),
    )


def test_builder_shaped_report_matches_real_builder(tmp_path):
    """Differential: the test helper aggregates exactly as axiom-oracles does."""
    report = build_policyengine_coverage_report(
        _us_checkout_with_outputs_in_three_repos(tmp_path)
    )
    assert len(report["repos"]) == 3
    rebuilt = _builder_shaped_report(report["items"])
    for key in _REPO_AGGREGATE_KEYS:
        assert rebuilt[key] == report[key], key


def test_apply_without_declarations_is_identity_on_builder_report(tmp_path):
    """Differential: the recount reproduces every builder aggregate exactly.

    If axiom-oracles adds a per-repo field, this fails on the pin bump instead
    of the recount silently dropping it.
    """
    report = build_policyengine_coverage_report(
        _us_checkout_with_outputs_in_three_repos(tmp_path)
    )
    before = copy.deepcopy(report)

    apply_pending_to_report(report, {})

    assert report.pop("pending")["applied"] == []
    assert report == before


def test_integration_declared_output_recounts_repo_rows(tmp_path):
    checkout = _us_checkout_with_outputs_in_three_repos(tmp_path)
    _declare(checkout, _FEDERAL_UNMAPPED)
    report = build_policyengine_coverage_report(checkout)
    federal_before = next(r for r in report["repos"] if r["repo"] == "rulespec-us")
    assert federal_before["status_counts"] == {"unmapped": 2}
    state_rows_before = [r for r in report["repos"] if r["repo"] != "rulespec-us"]

    apply_pending_to_report(
        report, declarations_from_files(load_pending_files(checkout))
    )

    rows = {row["repo"]: row for row in report["repos"]}
    assert rows["rulespec-us"] == {
        "repo": "rulespec-us",
        "total_outputs": 2,
        "status_counts": {PENDING_STATUS: 1, "unmapped": 1},
    }
    assert [r for r in report["repos"] if r["repo"] != "rulespec-us"] == (
        state_rows_before
    )
    _assert_repo_rows_partition_report(report)


def test_cli_text_repo_line_matches_top_level_status(tmp_path, monkeypatch, capsys):
    """The printed per-repo line no longer contradicts the ``Status:`` line."""
    checkout = _us_checkout_with_outputs_in_three_repos(tmp_path)
    _declare(checkout, _FEDERAL_UNMAPPED)

    code = _run_cli(monkeypatch, "oracle-coverage", "--root", str(checkout))

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    status_line = next(line for line in lines if line.startswith("Status: "))
    federal_line = next(line for line in lines if line.startswith("rulespec-us: "))
    assert f"{PENDING_STATUS}=1" in status_line
    assert federal_line == (
        f"rulespec-us: outputs=2 status={PENDING_STATUS}=1, unmapped=1"
    )


def test_cli_json_repo_rows_sum_to_top_level_status(tmp_path, monkeypatch, capsys):
    checkout = _us_checkout_with_outputs_in_three_repos(tmp_path)
    _declare(checkout, _FEDERAL_UNMAPPED, _FEDERAL_UNDECLARED)

    code = _run_cli(monkeypatch, "oracle-coverage", "--root", str(checkout), "--json")

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status_counts"][PENDING_STATUS] == 2
    assert "unmapped" not in report["status_counts"]
    _assert_repo_rows_partition_report(report)


_PROPERTY_REPOS = ("rulespec-et", "rulespec-et-aa", "rulespec-us", "rulespec-us-al")
_PROPERTY_STATUSES = (
    "unmapped",
    "comparable",
    "incomplete_comparable",
    "known_not_comparable",
    PENDING_STATUS,
)


@st.composite
def _items_and_declarations(draw):
    count = draw(st.integers(min_value=0, max_value=30))
    items = [
        _item(
            f"x:statutes/{index}#output_{index}",
            draw(st.sampled_from(_PROPERTY_REPOS)),
            draw(st.sampled_from(_PROPERTY_STATUSES)),
            program=draw(st.sampled_from(("tax", "snap", "medicaid"))),
            tested=draw(st.booleans()),
        )
        for index in range(count)
    ]
    present = [item["legal_id"] for item in items]
    absent = [f"x:statutes/absent#output_{n}" for n in range(5)]
    declared_ids = draw(
        st.lists(st.sampled_from(present + absent), unique=True, max_size=35)
    )
    entries = [
        _entry(
            legal_id,
            source=draw(st.sampled_from(("bulk", "manual", "migration", "backfill"))),
        )
        for legal_id in declared_ids
    ]
    return items, entries


@settings(max_examples=300, deadline=None)
@given(_items_and_declarations())
def test_property_repo_rows_partition_top_level_after_apply(case):
    items, entries = case
    report = _builder_shaped_report(items)
    before = copy.deepcopy(report)
    declared = declarations_from_files([_pending_file(Path("."), *entries)])

    summary = apply_pending_to_report(report, declared)

    _assert_repo_rows_partition_report(report)
    # Only unmapped -> pending_classification moves; every other aggregate holds.
    applied = set(summary["applied"])
    moved = Counter(
        item["repo"] for item in before["items"] if item["legal_id"] in applied
    )
    before_status = Counter(before["status_counts"])
    after_status = Counter(report["status_counts"])
    assert after_status[PENDING_STATUS] - before_status[PENDING_STATUS] == len(applied)
    assert before_status["unmapped"] - after_status["unmapped"] == len(applied)
    for status in set(before_status) | set(after_status):
        if status not in {PENDING_STATUS, "unmapped"}:
            assert after_status[status] == before_status[status], status
    before_rows = {row["repo"]: row for row in before["repos"]}
    after_rows = {row["repo"]: row for row in report["repos"]}
    assert after_rows.keys() == before_rows.keys()
    for repo, row in after_rows.items():
        assert row["total_outputs"] == before_rows[repo]["total_outputs"]
        row_before = Counter(before_rows[repo]["status_counts"])
        row_after = Counter(row["status_counts"])
        assert row_after[PENDING_STATUS] - row_before[PENDING_STATUS] == moved[repo]
        assert row_before["unmapped"] - row_after["unmapped"] == moved[repo]
    for key in ("total_outputs", "untested_comparable", "program_counts"):
        assert report[key] == before[key], key
    # The recount agrees with a from-scratch aggregation of the new items.
    rebuilt = _builder_shaped_report(report["items"])
    for key in _REPO_AGGREGATE_KEYS:
        assert report[key] == rebuilt[key], key
