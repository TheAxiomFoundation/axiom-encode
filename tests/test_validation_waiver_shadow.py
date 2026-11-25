"""Report-only shadow comparison of the incremental waiver audit with the full audit."""

from __future__ import annotations

import json
import os
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
        str(tmp_path / "corpus"),
        "--protected-base",
        str(tmp_path / "base.yaml"),
        "--changed-paths",
        str(tmp_path / "changed.txt"),
        "--axiom-rules-engine-path",
        str(tmp_path / "engine"),
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


# Review round 1: partition in every record, output path safety, fail-closed
# branches, run binding, argument checks.

RECORD_PREFIX = "validation-waivers incremental shadow record: "


def _run_with_paths(
    repo,
    capsys,
    record_path,
    *,
    summary_path=None,
    select=None,
    audit_line=None,
    **kwargs,
):
    """Run the audit with shadow outputs; return its result and the logged record."""

    original = cli._cmd_validation_waivers_audit
    lines: list[str] = []

    def audit_with_flags(args):
        args.incremental_shadow_report = record_path
        args.incremental_shadow_summary = summary_path
        return original(args)

    def write_line(line):
        if audit_line is not None:
            audit_line(line)
        lines.append(line)

    with (
        patch.object(cli, "_cmd_validation_waivers_audit", audit_with_flags),
        patch.object(
            cli,
            "select_validation_waiver_scope",
            side_effect=select or _selector(set()),
        ),
        patch.object(cli._waiver_partition, "write_audit_line", side_effect=write_line),
    ):
        result = _run_audit(repo, capsys, **kwargs)
    logged = [
        json.loads(line.removeprefix(RECORD_PREFIX))
        for line in lines
        if line.startswith(RECORD_PREFIX)
    ]
    return result, lines, logged


def test_every_record_names_its_partition(audit_repo, capsys, tmp_path):
    whole, _lines, _logged = _run_with_paths(
        audit_repo, capsys, tmp_path / "whole.json", changed=[audit_repo.paths[0]]
    )
    assert whole[0] == 0
    record = json.loads((tmp_path / "whole.json").read_text())
    assert record["partition"] == {"key": None, "count": 1}

    _result, _lines, _logged = _run_with_paths(
        audit_repo,
        capsys,
        tmp_path / "transition.json",
        changed=[audit_repo.paths[0]],
        partition_key="shard-a",
        transition_issues=["active waiver removed without a fix"],
    )
    record = json.loads((tmp_path / "transition.json").read_text())
    assert record["verdict"] == "inconclusive"
    assert record["partition"] == {"key": "shard-a", "count": 2}


def test_audit_exception_record_names_its_partition(tmp_path, capsys, monkeypatch):
    record_path = tmp_path / "record.json"
    args = _parsed_audit_args(
        monkeypatch,
        tmp_path,
        "--incremental-shadow-report",
        str(record_path),
        "--partition-key",
        "us-ny",
        "--partition-keys-json",
        '["us-ca", "us-ny", "us-tx"]',
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_validation_waivers(args)

    assert exit_info.value.code == 2
    record = json.loads(record_path.read_text())
    assert record["verdict"] == "inconclusive"
    assert record["partition"] == {"key": "us-ny", "count": 3}


def test_a_failing_comparison_still_writes_an_unsafe_record(
    audit_repo, capsys, tmp_path
):
    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    with patch.object(
        cli, "compare_incremental_shadow", side_effect=RuntimeError("compare broke")
    ):
        result, _lines, logged = _run_with_paths(
            audit_repo,
            capsys,
            tmp_path / "record.json",
            changed=[audit_repo.paths[0]],
            partition_key="shard-b",
        )
    plain_partitioned = _plain_report(
        audit_repo, capsys, changed=[audit_repo.paths[0]], partition_key="shard-b"
    )

    assert plain[0] == 0
    assert result == plain_partitioned
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["schema"] == SHADOW_SCHEMA
    assert record["verdict"] == "inconclusive"
    assert record["unsafe"] is True
    assert record["partition"] == {"key": "shard-b", "count": 2}
    assert record["errors"] == ["shadow comparison failed: RuntimeError: compare broke"]
    assert logged == [record]


def test_a_failing_summary_append_keeps_the_exit_code(audit_repo, capsys, tmp_path):
    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    with patch.object(cli, "append_shadow_summary", side_effect=OSError("disk full")):
        result, lines, _logged = _run_with_paths(
            audit_repo,
            capsys,
            tmp_path / "record.json",
            summary_path=tmp_path / "summary.md",
            changed=[audit_repo.paths[0]],
        )

    assert result == plain
    summary = [line for line in lines if "audit shadow (report only)" in line]
    assert summary and "summary not written: disk full" in summary[0]
    assert json.loads((tmp_path / "record.json").read_text())["verdict"] == "agree"


def test_a_broken_stderr_keeps_the_exit_code(audit_repo, capsys, tmp_path):
    def broken_pipe(line):
        if "shadow" in line:
            raise BrokenPipeError("stderr closed")

    plain = _plain_report(audit_repo, capsys, changed=[audit_repo.paths[0]])
    result, _lines, _logged = _run_with_paths(
        audit_repo,
        capsys,
        tmp_path / "record.json",
        audit_line=broken_pipe,
        changed=[audit_repo.paths[0]],
    )

    assert result == plain
    assert json.loads((tmp_path / "record.json").read_text())["verdict"] == "agree"


def _assert_refused(repo, capsys, record_path, *, summary_path=None, reason):
    plain = _plain_report(repo, capsys, changed=[repo.paths[0]])
    result, lines, logged = _run_with_paths(
        repo,
        capsys,
        record_path,
        summary_path=summary_path,
        changed=[repo.paths[0]],
    )
    assert result == plain
    assert len(logged) == 1
    assert logged[0]["verdict"] == "inconclusive"
    assert logged[0]["unsafe"] is True
    assert any(reason in error for error in logged[0]["errors"])
    assert any("output path refused" in line for line in lines)


def test_record_path_inside_the_repository_is_refused(audit_repo, capsys):
    ledger = audit_repo.root / "known-validation-gaps.yaml"
    module = audit_repo.root / audit_repo.paths[0]
    before = {path: path.read_bytes() for path in (ledger, module)}

    _assert_refused(audit_repo, capsys, ledger, reason="inside the checked")
    _assert_refused(audit_repo, capsys, module, reason="inside the checked")

    assert {path: path.read_bytes() for path in before} == before
    assert not [path for path in audit_repo.root.rglob(".*.tmp") if path.is_file()]


def test_audit_inputs_are_never_overwritten(audit_repo, capsys):
    before = audit_repo.base_file.read_bytes()

    _assert_refused(
        audit_repo, capsys, audit_repo.base_file, reason="input of the audit"
    )
    _assert_refused(
        audit_repo,
        capsys,
        audit_repo.base_file.parent / "record.json",
        summary_path=audit_repo.changed_file,
        reason="input of the audit",
    )

    assert audit_repo.base_file.read_bytes() == before
    assert not (audit_repo.base_file.parent / "record.json").exists()


def test_symlink_and_directory_paths_are_refused(audit_repo, capsys, tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    directory = tmp_path / "a-directory"
    directory.mkdir()

    _assert_refused(audit_repo, capsys, link, reason="symbolic link")
    _assert_refused(audit_repo, capsys, directory, reason="not a regular file")
    _assert_refused(
        audit_repo,
        capsys,
        tmp_path / "record.json",
        summary_path=link,
        reason="symbolic link",
    )

    assert target.read_text() == "keep\n"
    assert link.is_symlink()
    assert list(directory.iterdir()) == []
    assert not (tmp_path / "record.json").exists()


def test_write_failure_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "record.json"
    with (
        patch(
            "axiom_encode.validation_waiver_shadow.os.replace",
            side_effect=OSError("rename failed"),
        ),
        pytest.raises(OSError, match="rename failed"),
    ):
        write_shadow_record(target, {"schema": SHADOW_SCHEMA})

    assert list(tmp_path.iterdir()) == []


def test_temporary_file_does_not_follow_a_planted_link(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n")
    target_dir = tmp_path / "out"
    target_dir.mkdir()
    # The name the first version predicted from the process id.
    (target_dir / f".record.json.{os.getpid()}.tmp").symlink_to(outside)

    write_shadow_record(target_dir / "record.json", {"schema": SHADOW_SCHEMA})

    assert outside.read_text() == "keep\n"
    assert json.loads((target_dir / "record.json").read_text()) == {
        "schema": SHADOW_SCHEMA
    }


def test_record_binds_the_run(audit_repo, capsys, tmp_path):
    import hashlib

    _result, _lines, _logged = _run_with_paths(
        audit_repo, capsys, tmp_path / "record.json", changed=[audit_repo.paths[0]]
    )
    record = json.loads((tmp_path / "record.json").read_text())

    binding = record["binding"]
    assert binding["encoder_version"] == cli.__version__
    assert binding["protected_base_sha256"] == (
        "sha256:" + hashlib.sha256(audit_repo.base_file.read_bytes()).hexdigest()
    )
    assert binding["changed_paths_sha256"] == (
        "sha256:" + hashlib.sha256(audit_repo.changed_file.read_bytes()).hexdigest()
    )
    # The fixture repository is not a Git checkout of its own.
    assert binding["head_commit"] is None


def test_failed_write_removes_only_an_earlier_record(audit_repo, capsys, tmp_path):
    earlier = tmp_path / "earlier.json"
    earlier.write_text(json.dumps({"schema": SHADOW_SCHEMA, "verdict": "agree"}))
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"schema": "something-else"}))

    for path in (earlier, foreign):
        with patch.object(
            cli, "write_shadow_record", side_effect=OSError("write failed")
        ):
            result, lines, _logged = _run_with_paths(
                audit_repo, capsys, path, changed=[audit_repo.paths[0]]
            )
        assert result[0] == 0
        assert any("record not written: write failed" in line for line in lines)

    assert not earlier.exists()
    assert foreign.exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--incremental-shadow-summary", "s.md"], "requires"),
        (
            [
                "--incremental-shadow-report",
                "same.json",
                "--incremental-shadow-summary",
                "same.json",
            ],
            "different files",
        ),
        (
            [
                "--incremental-shadow-report",
                "R.json",
                "--incremental-shadow-summary",
                "r.json",
            ],
            "different files",
        ),
    ],
)
def test_shadow_argument_misuse_is_an_argument_error(
    tmp_path, capsys, monkeypatch, extra, message
):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        _parsed_audit_args(monkeypatch, tmp_path, *extra)

    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# Review round 2: identity instead of names, every path input, the Git probe.


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.write_text("")
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.unlink()


def test_other_spellings_of_the_root_and_inputs_are_refused(
    audit_repo, capsys, tmp_path
):
    ledger = audit_repo.root / "known-validation-gaps.yaml"
    before = (ledger.read_bytes(), audit_repo.base_file.read_bytes())
    alias = tmp_path / "alias"
    alias.symlink_to(audit_repo.root, target_is_directory=True)
    hard_link = tmp_path / "hard-link-to-base.json"
    os.link(audit_repo.base_file, hard_link)

    _assert_refused(
        audit_repo, capsys, alias / "record.json", reason="inside the checked"
    )
    _assert_refused(audit_repo, capsys, hard_link, reason="input of the audit")
    if _case_insensitive(tmp_path):
        upper_root = Path(str(audit_repo.root).upper())
        _assert_refused(
            audit_repo,
            capsys,
            upper_root / "known-validation-gaps.yaml",
            reason="inside the checked",
        )
        _assert_refused(
            audit_repo,
            capsys,
            audit_repo.base_file.with_name(audit_repo.base_file.name.upper()),
            reason="input of the audit",
        )

    assert (ledger.read_bytes(), audit_repo.base_file.read_bytes()) == before


@pytest.mark.parametrize(
    "name",
    [
        "corpus_path",
        "axiom_rules_path",
        "axiom_compose_path",
        "rulespec_dependency_root",
    ],
)
def test_paths_inside_any_audit_input_are_refused(audit_repo, capsys, tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    original = cli._cmd_validation_waivers_audit

    def audit_with_input(args):
        setattr(
            args,
            name,
            [directory] if name == "rulespec_dependency_root" else directory,
        )
        return original(args)

    with (
        patch.object(cli, "_cmd_validation_waivers_audit", audit_with_input),
        patch.object(cli, "_resolve_optional_axiom_compose_path", return_value=None),
        patch.object(cli, "_rulespec_dependency_roots_from_args", return_value=()),
    ):
        _assert_refused(
            audit_repo, capsys, directory / "record.json", reason="input of the audit"
        )
    assert list(directory.iterdir()) == []


def test_every_path_argument_of_the_audit_is_protected(tmp_path, monkeypatch):
    """A new path-typed audit argument must be added to the protected inputs."""

    import argparse

    captured = []
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured.append(self)
        return original(self, *args, **kwargs)

    with patch.object(argparse.ArgumentParser, "parse_args", capture):
        _parsed_audit_args(monkeypatch, tmp_path)

    def subparser(parser, name):
        for action in parser._actions:
            if (
                isinstance(action, argparse._SubParsersAction)
                and name in action.choices
            ):
                return action.choices[name]
        raise AssertionError(name)

    audit = subparser(subparser(captured[0], "validation-waivers"), "audit")
    path_arguments = {action.dest for action in audit._actions if action.type is Path}
    shadow_outputs = {"incremental_shadow_report", "incremental_shadow_summary"}
    assert path_arguments - shadow_outputs - {"root"} == set(
        cli._VALIDATION_WAIVER_AUDIT_PATH_INPUTS
    )


def _git(cwd: Path, *arguments: str) -> str:
    import subprocess

    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "commit", "-q", "--allow-empty", "-m", path.name)
    return _git(path, "rev-parse", "HEAD")


def _binding(root: Path, tmp_path: Path):
    from types import SimpleNamespace

    return cli._validation_waiver_shadow_binding(
        SimpleNamespace(
            root=root,
            protected_base=tmp_path / "absent-base",
            changed_paths=tmp_path / "absent-changed",
        )
    )


def test_head_commit_is_the_roots_own_checkout_only(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own_head = _repository(own)
    other_head = _repository(tmp_path / "other")
    outer = tmp_path / "outer"
    _repository(outer)
    nested = outer / "rulespec-us"
    nested.mkdir()

    assert _binding(own, tmp_path)["head_commit"] == own_head
    # A root inside another checkout has no commit of its own.
    assert _binding(nested, tmp_path)["head_commit"] is None

    # GIT_DIR and GIT_WORK_TREE must not redirect the probe.
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "other" / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "other"))
    assert _binding(own, tmp_path)["head_commit"] == own_head
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _binding(plain, tmp_path)["head_commit"] is None
    assert other_head != own_head


def test_undecodable_git_output_loses_only_the_head_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    with patch.object(
        cli.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout=b"\xff\xfe\n" + b"a" * 40),
    ):
        binding = _binding(tmp_path, tmp_path)

    assert binding["head_commit"] is None
    assert binding["encoder_version"] == cli.__version__


def test_record_carries_github_sha(audit_repo, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    _run_with_paths(
        audit_repo, capsys, tmp_path / "record.json", changed=[audit_repo.paths[0]]
    )

    record = json.loads((tmp_path / "record.json").read_text())
    assert record["binding"]["github_sha"] == "a" * 40


def test_dot_dot_through_a_link_is_judged_by_the_filesystem(
    audit_repo, capsys, tmp_path
):
    """`link/..` names the link target's parent, not the link's directory."""

    statutes = audit_repo.root / "us-ca" / "statutes"
    link = tmp_path / "statutes-link"
    link.symlink_to(statutes, target_is_directory=True)
    sneaky = link / ".." / "record.json"

    _assert_refused(audit_repo, capsys, sneaky, reason="inside the checked")
    assert not (audit_repo.root / "us-ca" / "record.json").exists()


def test_two_names_of_one_file_are_the_same_output(tmp_path, capsys, monkeypatch):
    report = tmp_path / "report.json"
    report.write_text("")
    summary = tmp_path / "summary.md"
    os.link(report, summary)

    with pytest.raises(SystemExit) as exit_info:
        _parsed_audit_args(
            monkeypatch,
            tmp_path,
            "--incremental-shadow-report",
            str(report),
            "--incremental-shadow-summary",
            str(summary),
        )

    assert exit_info.value.code == 2
    assert "different files" in capsys.readouterr().err
