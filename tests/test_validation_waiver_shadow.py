"""Report-only shadow comparison of the incremental waiver audit with the full audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.validation_waiver_scope import WaiverAuditScope
from axiom_encode.validation_waiver_shadow import (
    SHADOW_SCHEMA,
    append_shadow_summary,
    compare_incremental_shadow,
    shadow_summary_line,
    write_shadow_record,
)
from tests.test_validation_waiver_audit_cli import _entry, _metadata, _waiver_set
from tests.test_validation_waiver_incremental_cli import _run_audit, _signed_change
from tests.test_validation_waiver_incremental_cli import (
    audit_repo as _incremental_audit_repo,
)

audit_repo = _incremental_audit_repo

PATHS = ("us-ca/statutes/a.yaml", "us-ca/statutes/b.yaml", "us-ny/statutes/z.yaml")


def _full_report(flagged=(), *, paths=PATHS, partition=None):
    results = []
    errors = []
    for path in paths:
        result = {"path": path, "kind": "active", "success": path not in flagged}
        if path in flagged:
            result["error"] = f"{path}: validation failure fingerprint drifted"
            errors.append(result["error"])
        results.append(result)
    report = {
        "success": not errors,
        "checked": len(results),
        "results": results,
        "errors": errors,
        "scope": "full",
    }
    if partition is not None:
        report["partition"] = partition
    return report


def _selection(paths, scope="incremental", fallback_reason=None):
    selection = {"scope": scope, "paths": list(paths)}
    if fallback_reason is not None:
        selection["fallback_reason"] = fallback_reason
    return selection


# The comparison itself.


def test_equal_results_agree():
    record = compare_incremental_shadow(
        full_report=_full_report(flagged=[PATHS[0]]),
        selection=_selection([PATHS[0], PATHS[1]]),
        selection_seconds=1.23456,
    )

    assert record["schema"] == SHADOW_SCHEMA
    assert record["report_only"] is True
    assert record["verdict"] == "agree"
    assert record["unsafe"] is False
    assert record["missed"] == record["missed_unselected"] == record["extra"] == []
    assert record["errors"] == []
    assert record["full"] == {"checked": 3, "flagged": 1, "success": False}
    assert record["incremental"] == {
        "scope": "incremental",
        "selected": 2,
        "unselected": 1,
        "derived": True,
        "flagged": 1,
    }
    assert record["selection_seconds"] == 1.235


def test_both_pass_with_nothing_selected_agrees():
    record = compare_incremental_shadow(
        full_report=_full_report(), selection=_selection([])
    )

    assert record["verdict"] == "agree"
    assert record["unsafe"] is False
    assert record["incremental"]["unselected"] == 3


def test_incremental_misses_a_waiver_the_full_audit_flags():
    """The dangerous direction: the incremental gate would pass, the full fails."""

    record = compare_incremental_shadow(
        full_report=_full_report(flagged=[PATHS[2]]),
        selection=_selection([PATHS[0]]),
    )

    assert record["verdict"] == "incremental_missed"
    assert record["unsafe"] is True
    assert record["missed"] == record["missed_unselected"] == [PATHS[2]]
    assert record["extra"] == []
    assert record["incremental"]["flagged"] == 0


def test_a_selected_waiver_the_separate_incremental_run_did_not_flag_is_missed():
    full = _full_report(flagged=[PATHS[0]])
    incremental = _full_report(paths=[PATHS[0]])
    incremental["scope"] = "incremental"

    record = compare_incremental_shadow(
        full_report=full,
        selection=_selection([PATHS[0]]),
        incremental_report=incremental,
    )

    assert record["verdict"] == "incremental_missed"
    assert record["unsafe"] is True
    assert record["missed"] == [PATHS[0]]
    assert record["missed_unselected"] == []
    assert record["incremental"]["derived"] is False


def test_incremental_flags_more_than_the_full_audit():
    incremental = _full_report(flagged=[PATHS[1]], paths=[PATHS[0], PATHS[1]])
    incremental["scope"] = "incremental"

    record = compare_incremental_shadow(
        full_report=_full_report(),
        selection=_selection([PATHS[0], PATHS[1]]),
        incremental_report=incremental,
    )

    assert record["verdict"] == "incremental_extra"
    # Failing more is a disagreement to investigate, not an unsound pass.
    assert record["unsafe"] is False
    assert record["extra"] == [PATHS[1]]
    assert record["missed"] == []


def test_missed_wins_over_extra():
    incremental = _full_report(flagged=[PATHS[1]], paths=[PATHS[0], PATHS[1]])
    incremental["scope"] = "incremental"

    record = compare_incremental_shadow(
        full_report=_full_report(flagged=[PATHS[2]]),
        selection=_selection([PATHS[0], PATHS[1]]),
        incremental_report=incremental,
    )

    assert record["verdict"] == "incremental_missed"
    assert record["missed"] == [PATHS[2]]
    assert record["extra"] == [PATHS[1]]


def test_a_full_fallback_selection_agrees_and_keeps_its_reason():
    record = compare_incremental_shadow(
        full_report=_full_report(flagged=[PATHS[1]]),
        selection=_selection(PATHS, scope="full", fallback_reason="ledger changed"),
    )

    assert record["verdict"] == "agree"
    assert record["incremental"]["scope"] == "full"
    assert record["incremental"]["fallback_reason"] == "ledger changed"


def _without_success(report):
    del report["results"][0]["success"]
    return report


def _with_duplicate(report):
    report["results"].append(dict(report["results"][0]))
    return report


def _with_unattributed_error(report):
    report["errors"].append("systemic audit discrepancy: 30 of 38 modules")
    report["success"] = False
    return report


def _with_wrong_success(report):
    report["success"] = False
    return report


def _with_incremental_scope(report):
    report["scope"] = "incremental"
    return report


@pytest.mark.parametrize(
    ("full_report", "selection", "message"),
    [
        (None, _selection([]), "full report is not an object"),
        ([], _selection([]), "full report is not an object"),
        (_with_incremental_scope(_full_report()), _selection([]), "scope is not full"),
        ({"scope": "full", "errors": [], "success": True}, _selection([]), "results"),
        (_without_success(_full_report()), _selection([]), "success is not a boolean"),
        (_with_duplicate(_full_report()), _selection([]), "occurs twice"),
        (_with_unattributed_error(_full_report()), _selection([]), "belong to no"),
        (_with_wrong_success(_full_report()), _selection([]), "disagrees"),
        (_full_report(), None, "was not computed"),
        (_full_report(), "everything", "selection is not an object"),
        (_full_report(), {"scope": "some", "paths": []}, "full or incremental"),
        (_full_report(), {"scope": "incremental", "paths": "a"}, "list of paths"),
        (_full_report(), _selection([PATHS[0], PATHS[0]]), "duplicate path"),
        (_full_report(), _selection(["us-tx/statutes/q.yaml"]), "did not check"),
        (_full_report(), _selection([PATHS[0]], scope="full"), "every checked"),
    ],
)
def test_malformed_input_is_inconclusive_and_unsafe(full_report, selection, message):
    record = compare_incremental_shadow(full_report=full_report, selection=selection)

    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert len(record["errors"]) == 1
    assert message in record["errors"][0]


def test_a_failed_selection_is_inconclusive_even_when_the_full_audit_passes():
    record = compare_incremental_shadow(
        full_report=_full_report(),
        selection=None,
        selection_error="OSError: permission denied",
    )

    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert "permission denied" in record["errors"][0]
    assert record["full"]["success"] is True


def test_a_separate_incremental_report_must_check_exactly_the_selection():
    incremental = _full_report(paths=[PATHS[0]])

    record = compare_incremental_shadow(
        full_report=_full_report(),
        selection=_selection([PATHS[0], PATHS[1]]),
        incremental_report=incremental,
    )

    assert record["verdict"] == "inconclusive"
    assert "exactly the selected paths" in record["errors"][0]


def test_summary_line_and_files(tmp_path: Path):
    record = compare_incremental_shadow(
        full_report=_full_report(
            flagged=[PATHS[2]], partition={"key": "us-ny", "index": 0, "count": 2}
        ),
        selection=_selection([PATHS[0]]),
    )
    line = shadow_summary_line(record)

    assert line.startswith(
        "incremental waiver audit shadow (report only): incremental_missed"
    )
    assert "partition us-ny" in line
    assert "selected 1 of 3" in line
    assert f"missed 1, first {PATHS[2]}" in line

    target = tmp_path / "shadow.json"
    write_shadow_record(target, record)
    assert json.loads(target.read_text()) == record
    assert [path.name for path in tmp_path.iterdir()] == ["shadow.json"]

    summary = tmp_path / "summary.md"
    summary.write_text("# Audit\n")
    append_shadow_summary(summary, record)
    assert summary.read_text() == f"# Audit\n- {line}\n"


# The audit command: the shadow never changes the report or the exit code.


def _run_with_shadow(repo, capsys, tmp_path, *, select=None, **kwargs):
    record_path = tmp_path / "shadow" / "record.json"
    record_path.parent.mkdir(exist_ok=True)
    summary_path = tmp_path / "shadow" / "summary.md"
    original = cli._cmd_validation_waivers_audit

    def audit_with_flags(args):
        args.incremental_shadow_report = record_path
        args.incremental_shadow_summary = summary_path
        return original(args)

    with (
        patch.object(cli, "_cmd_validation_waivers_audit", audit_with_flags),
        patch.object(
            cli,
            "select_validation_waiver_scope",
            side_effect=select or AssertionError("selection must not run"),
        ),
    ):
        code, report, executed = _run_audit(repo, capsys, **kwargs)
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    summary = summary_path.read_text() if summary_path.exists() else None
    return code, report, executed, record, summary


def _selector(selected):
    def select(paths, **kwargs):
        chosen = tuple(path for path in paths if path in selected)
        return WaiverAuditScope(
            "incremental", chosen, {path: ["test"] for path in chosen}
        )

    return select


def _plain_report(repo, capsys, **kwargs):
    with patch.object(
        cli,
        "select_validation_waiver_scope",
        side_effect=AssertionError("full mode must not select"),
    ):
        return _run_audit(repo, capsys, **kwargs)


def test_shadow_leaves_a_passing_full_audit_unchanged(audit_repo, capsys, tmp_path):
    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    code, report, executed, record, summary = _run_with_shadow(
        audit_repo,
        capsys,
        tmp_path,
        select=_selector({audit_repo.paths[0]}),
        changed=[audit_repo.paths[0]],
    )

    assert (code, report, executed) == plain
    assert record["verdict"] == "agree"
    assert record["incremental"]["selected"] == 1
    assert record["full"] == {"checked": 4, "flagged": 0, "success": True}
    assert summary.startswith("- incremental waiver audit shadow (report only): agree")


def _drifted_head(repo, drifted=None):
    drifted = drifted or repo.paths[3]
    head = _waiver_set(
        {
            path: (
                _entry(path, active=_metadata("sha256:expected-before-drift"))
                if path == drifted
                else entry
            )
            for path, entry in repo.waivers.entries.items()
        }
    )
    return drifted, head


def _confirming_recheck(repo):
    def recheck(paths, **kwargs):
        # The isolated recheck confirms the drift.
        return [
            {
                "path": path.as_posix(),
                "passed": False,
                "fingerprint": repo.waivers.entries[path.as_posix()].active.fingerprint,
                "outcome": {},
            }
            for path in paths
        ]

    return recheck


def test_shadow_records_a_miss_without_changing_a_failing_exit_code(
    audit_repo, capsys, tmp_path
):
    """The full audit flags a waiver that the selection does not hold.

    The module's result drifted while the ledger stayed the protected base
    ledger (the case an under-selection produces), so the ledger guard does
    not widen the selection.
    """

    drifted, head = _drifted_head(audit_repo)
    with (
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            side_effect=_confirming_recheck(audit_repo),
        ),
        patch.object(cli, "_validation_waiver_ledger_differs", return_value=False),
    ):
        plain = _plain_report(
            audit_repo, capsys, changed=[audit_repo.paths[0]], head=head
        )
        code, report, executed, record, summary = _run_with_shadow(
            audit_repo,
            capsys,
            tmp_path,
            select=_selector({audit_repo.paths[0]}),
            changed=[audit_repo.paths[0]],
            head=head,
        )

    assert plain[0] == 1
    assert (code, report, executed) == plain
    assert record["verdict"] == "incremental_missed"
    assert record["unsafe"] is True
    assert record["missed"] == [drifted]
    assert record["full"]["flagged"] == 1
    assert "incremental_missed" in summary


def test_shadow_selection_widens_when_the_ledger_left_the_protected_base(
    audit_repo, capsys, tmp_path
):
    """The shadow uses the same ledger guard as the incremental scope."""

    drifted, head = _drifted_head(audit_repo)
    with patch.object(
        cli,
        "_fingerprint_validation_waiver_modules",
        side_effect=_confirming_recheck(audit_repo),
    ):
        code, _report, _executed, record, _summary = _run_with_shadow(
            audit_repo,
            capsys,
            tmp_path,
            select=_selector({audit_repo.paths[0]}),
            changed=[audit_repo.paths[0]],
            head=head,
        )

    assert code == 1
    assert record["verdict"] == "agree"
    assert record["incremental"]["scope"] == "full"
    assert "protected base ledger" in record["incremental"]["fallback_reason"]


@pytest.mark.parametrize(
    ("drifted_index", "verdict"),
    [(3, "agree"), (0, "incremental_missed")],
)
def test_shadow_with_the_real_selector_on_a_signed_change(
    audit_repo, capsys, tmp_path, drifted_index, verdict
):
    """A signed us-ny change selects the us-ny waiver only.

    A drift of that waiver agrees; a drift of a us-ca waiver, which the
    selector rightly leaves out, is what an under-selection would look like.
    """

    changed = _signed_change(audit_repo, "us-ny/statutes/new.yaml")
    drifted, head = _drifted_head(audit_repo, audit_repo.paths[drifted_index])
    real_selector = cli.select_validation_waiver_scope
    with (
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            side_effect=_confirming_recheck(audit_repo),
        ),
        patch.object(cli, "_validation_waiver_ledger_differs", return_value=False),
    ):
        code, report, _executed, record, _summary = _run_with_shadow(
            audit_repo,
            capsys,
            tmp_path,
            select=real_selector,
            changed=changed,
            head=head,
        )

    assert code == 1
    assert report["scope"] == "full"
    assert record["incremental"]["scope"] == "incremental"
    assert record["incremental"]["selected"] == 1
    assert record["verdict"] == verdict
    assert record["missed"] == ([] if verdict == "agree" else [drifted])


def test_shadow_selection_error_is_recorded_and_the_exit_code_kept(
    audit_repo, capsys, tmp_path
):
    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    code, report, executed, record, _summary = _run_with_shadow(
        audit_repo,
        capsys,
        tmp_path,
        select=RuntimeError("selector broke"),
        changed=[audit_repo.paths[0]],
    )

    assert (code, report, executed) == plain
    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert record["errors"] == [
        "incremental selection failed: RuntimeError: selector broke"
    ]
    # The full audit's side of the comparison is still recorded.
    assert record["full"] == {"checked": 4, "flagged": 0, "success": True}
    assert "selection_seconds" in record


def test_shadow_with_a_verifier_restate_is_inconclusive(audit_repo, capsys, tmp_path):
    original = cli._cmd_validation_waivers_audit

    def audit_with_restate(args):
        args.verifier_restate = True
        return original(args)

    with patch.object(cli, "_cmd_validation_waivers_audit", audit_with_restate):
        code, report, executed, record, _summary = _run_with_shadow(
            audit_repo,
            capsys,
            tmp_path,
            changed=[".axiom/workflow-toolchain.toml"],
        )

    assert code == 0
    assert report["scope"] == "full"
    assert executed == list(audit_repo.paths)
    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert any("verifier restate" in error for error in record["errors"])


def test_shadow_after_transition_failure_is_inconclusive(audit_repo, capsys, tmp_path):
    code, report, executed, record, _summary = _run_with_shadow(
        audit_repo,
        capsys,
        tmp_path,
        changed=[audit_repo.paths[0]],
        transition_issues=["active waiver removed without a fix"],
    )

    assert code == 1
    assert executed == []
    assert report["errors"] == ["active waiver removed without a fix"]
    assert record["verdict"] == "inconclusive"
    assert any("transition checks failed" in error for error in record["errors"])


def test_shadow_with_incremental_scope_is_inconclusive(audit_repo, capsys, tmp_path):
    code, report, executed, record, _summary = _run_with_shadow(
        audit_repo,
        capsys,
        tmp_path,
        select=_selector({audit_repo.paths[0]}),
        changed=[audit_repo.paths[0]],
        scope="incremental",
    )

    assert code == 0
    assert executed == [audit_repo.paths[0]]
    assert report["scope"] == "incremental"
    assert record["verdict"] == "inconclusive"
    assert any("--scope full" in error for error in record["errors"])


def test_shadow_partition_selects_only_its_own_waivers(audit_repo, capsys, tmp_path):
    seen = []

    def select(paths, **kwargs):
        seen.append(list(paths))
        return WaiverAuditScope("incremental", (), {})

    code, report, executed, record, _summary = _run_with_shadow(
        audit_repo,
        capsys,
        tmp_path,
        select=select,
        changed=["README.md"],
        partition_key="shard-a",
    )

    assert code == 0
    assert seen == [executed]
    assert record["partition"]["key"] == "shard-a"
    assert record["full"]["checked"] == len(executed)
    assert record["verdict"] == "agree"


def test_unwritable_record_keeps_the_exit_code(audit_repo, capsys, tmp_path):
    original = cli._cmd_validation_waivers_audit

    def audit_with_flags(args):
        args.incremental_shadow_report = tmp_path / "missing-dir" / "record.json"
        return original(args)

    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    lines = []
    with (
        patch.object(cli, "_cmd_validation_waivers_audit", audit_with_flags),
        patch.object(
            cli,
            "select_validation_waiver_scope",
            side_effect=_selector(set()),
        ),
        patch.object(
            cli._waiver_partition, "write_audit_line", side_effect=lines.append
        ),
    ):
        result = _run_audit(audit_repo, capsys, changed=[audit_repo.paths[0]])

    assert result == plain
    summary = [line for line in lines if "audit shadow (report only)" in line]
    assert len(summary) == 1
    assert "agree" in summary[0]
    assert "record not written" in summary[0]
    prefix = "validation-waivers incremental shadow record: "
    logged = [line for line in lines if line.startswith(prefix)]
    assert len(logged) == 1
    record = json.loads(logged[0].removeprefix(prefix))
    assert record["verdict"] == "agree"
    assert record["incremental"]["selected"] == 0


def _parsed_audit_args(monkeypatch, tmp_path, *extra):
    argv = [
        "axiom-encode",
        "validation-waivers",
        "audit",
        "--root",
        str(tmp_path / "absent"),
        "--corpus-path",
        str(tmp_path),
        "--protected-base",
        str(tmp_path / "base.yaml"),
        "--changed-paths",
        str(tmp_path / "changed.txt"),
        "--axiom-rules-engine-path",
        str(tmp_path),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with patch.object(cli, "cmd_validation_waivers") as call:
        cli.main()
    return call.call_args.args[0]


def test_audit_error_writes_an_inconclusive_record(tmp_path, capsys, monkeypatch):
    record_path = tmp_path / "record.json"
    args = _parsed_audit_args(
        monkeypatch, tmp_path, "--incremental-shadow-report", str(record_path)
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_validation_waivers(args)

    assert exit_info.value.code == 2
    record = json.loads(record_path.read_text())
    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert any("did not complete" in error for error in record["errors"])
    assert "incremental waiver audit shadow" in capsys.readouterr().err


def test_shadow_flags_default_to_off(tmp_path, monkeypatch):
    args = _parsed_audit_args(monkeypatch, tmp_path)

    assert args.scope == "full"
    assert args.incremental_shadow_report is None
    assert args.incremental_shadow_summary is None


def test_args_reused_from_an_earlier_run_still_write_a_record(
    audit_repo, capsys, tmp_path
):
    original = cli._cmd_validation_waivers_audit

    def audit_with_earlier_run(args):
        # The marker an earlier run on the same args object left behind.
        args._validation_waiver_shadow_done = True
        return original(args)

    with patch.object(cli, "_cmd_validation_waivers_audit", audit_with_earlier_run):
        _code, _report, _executed, record, _summary = _run_with_shadow(
            audit_repo,
            capsys,
            tmp_path,
            select=_selector({audit_repo.paths[0]}),
            changed=[audit_repo.paths[0]],
        )

    assert record is not None
    assert record["verdict"] == "agree"
