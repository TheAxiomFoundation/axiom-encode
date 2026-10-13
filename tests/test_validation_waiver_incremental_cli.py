"""Incremental audit CLI reports, defaults, partitioning, and strict transitions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from axiom_encode import cli
from tests.test_validation_waiver_audit_cli import (
    _audit_args,
    _entry,
    _metadata,
    _waiver_set,
)
from tests.test_validation_waiver_scope import _module


@pytest.fixture
def audit_repo(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    paths = (
        "us-ca/statutes/a.yaml",
        "us-ca/statutes/b.yaml",
        "us-ca/statutes/c.yaml",
        "us-ny/statutes/z.yaml",
    )
    for path in paths:
        _module(root, path)
    (root / "known-validation-gaps.yaml").write_text("validate_failures: {}\n")
    base_file = tmp_path / "protected-base.yaml"
    base_file.write_text("validate_failures: {}\n")
    changed_file = tmp_path / "changed.txt"
    changed_file.write_text("")
    entries = {
        path: _entry(path, active=_metadata(f"sha256:{index}"))
        for index, path in enumerate(paths)
    }
    return SimpleNamespace(
        root=root,
        paths=paths,
        base_file=base_file,
        changed_file=changed_file,
        waivers=_waiver_set(entries),
    )


def _run_audit(
    repo,
    capsys,
    *,
    changed=(),
    scope=None,
    partition_key=None,
    head=None,
    transition_issues=(),
    as_json=True,
    execution_inputs=None,
):
    repo.changed_file.write_text("".join(f"{path}\n" for path in changed))
    args = _audit_args(
        repo.root, repo.base_file, repo.changed_file, repo.root / "corpus"
    )
    for name, value in (execution_inputs or {}).items():
        setattr(args, name, value)
    if scope is not None:
        args.scope = scope
    if partition_key is not None:
        args.partition_key = partition_key
        args.partition_keys_json = '["shard-b", "shard-a"]'
    args.json = as_json
    executed = []

    def fingerprint(paths, **kwargs):
        executed.extend(path.as_posix() for path in paths)
        return [
            {
                "path": path.as_posix(),
                "passed": False,
                "fingerprint": repo.waivers.entries[path.as_posix()].active.fingerprint,
                "outcome": {},
            }
            for path in paths
        ]

    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            side_effect=[head or repo.waivers, repo.waivers],
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=transition_issues,
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules_parallel",
            side_effect=fingerprint,
        ),
    ):
        code = cli._cmd_validation_waivers_audit(args)
    output = capsys.readouterr()
    report = json.loads(output.out) if as_json else output.out
    return code, report, executed


def test_default_full_report_keeps_existing_payload_and_audits_every_path(
    audit_repo,
    capsys,
):
    with patch.object(
        cli,
        "select_validation_waiver_scope",
        side_effect=AssertionError("full mode must not select incrementally"),
    ):
        code, report, executed = _run_audit(audit_repo, capsys, changed=["README.md"])

    assert code == 0
    assert executed == list(audit_repo.paths)
    assert report.pop("scope") == "full"
    assert report == {
        "success": True,
        "root": str(audit_repo.root),
        "waivers": "known-validation-gaps.yaml",
        "checked": 4,
        "results": [
            {
                "path": path,
                "kind": "active",
                "passed": False,
                "fingerprint": f"sha256:{index}",
                "outcome": {},
                "expected_fingerprint": f"sha256:{index}",
                "success": True,
            }
            for index, path in enumerate(audit_repo.paths)
        ],
        "errors": [],
    }


def test_explicit_full_and_default_reports_are_identical(audit_repo, capsys):
    default = _run_audit(audit_repo, capsys, changed=[audit_repo.paths[0]])
    explicit = _run_audit(
        audit_repo, capsys, changed=[audit_repo.paths[0]], scope="full"
    )

    assert default == explicit


@pytest.mark.parametrize("changed", [[], ["README.md"], ["docs/guide.md"]])
def test_incremental_inert_changes_report_zero_without_fingerprinting(
    audit_repo,
    capsys,
    changed,
):
    for relative in changed:
        document = audit_repo.root / relative
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text("# Documentation\n", encoding="utf-8")
    code, report, executed = _run_audit(
        audit_repo, capsys, changed=changed, scope="incremental"
    )

    assert code == 0
    assert executed == []
    assert report["scope"] == "incremental"
    assert report["checked"] == report["selected"] == 0
    assert report["total_waivers"] == 4
    assert report["selection_reasons"] == {}
    assert "fallback_reason" not in report


@pytest.mark.parametrize(
    "input_argument", ["corpus_path", "axiom_rules_path", "axiom_compose_path"]
)
def test_incremental_configured_execution_input_change_reports_full_fallback(
    audit_repo, capsys, input_argument
):
    changed = "docs/execution/compose.md"
    candidate = audit_repo.root / changed
    candidate.parent.mkdir(parents=True)
    candidate.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    candidate.chmod(0o700)
    configured = (
        candidate if input_argument == "axiom_compose_path" else candidate.parent
    )

    code, report, executed = _run_audit(
        audit_repo,
        capsys,
        changed=[changed],
        scope="incremental",
        execution_inputs={input_argument: str(configured)},
    )

    assert code == 0
    assert executed == list(audit_repo.paths)
    assert report["scope"] == "full"
    assert report["selected"] == report["checked"] == report["total_waivers"] == 4
    assert report["fallback_reason"] == f"configured execution input changed: {changed}"
    assert set(report["selection_reasons"]) == set(audit_repo.paths)


@pytest.mark.parametrize(
    "input_argument", ["corpus_path", "axiom_rules_path", "axiom_compose_path"]
)
def test_incremental_execution_input_case_alias_reports_full_fallback(
    audit_repo, capsys, input_argument
):
    changed = "docs/execution/compose.md"
    actual = audit_repo.root / changed
    actual.parent.mkdir(parents=True)
    actual.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    actual.chmod(0o700)
    configured = audit_repo.root / "docs/EXECUTION"
    if input_argument == "axiom_compose_path":
        configured /= "COMPOSE.md"

    code, report, executed = _run_audit(
        audit_repo,
        capsys,
        changed=[changed],
        scope="incremental",
        execution_inputs={input_argument: str(configured)},
    )

    assert code == 0
    assert executed == list(audit_repo.paths)
    assert report["scope"] == "full"
    assert report["selected"] == report["checked"] == report["total_waivers"] == 4
    assert report["fallback_reason"] == f"configured execution input changed: {changed}"
    assert set(report["selection_reasons"]) == set(audit_repo.paths)


def test_incremental_ignores_mock_execution_arguments_and_does_not_admit_unused_inputs(
    audit_repo, capsys
):
    document = audit_repo.root / "docs/guide.md"
    document.parent.mkdir()
    document.write_text("# Documentation\n", encoding="utf-8")
    with patch.object(
        cli,
        "_resolve_optional_axiom_compose_path",
        side_effect=AssertionError("empty audit must not admit or execute a composer"),
    ):
        code, report, executed = _run_audit(
            audit_repo,
            capsys,
            changed=["docs/guide.md"],
            scope="incremental",
            execution_inputs={
                "corpus_path": None,
                "axiom_rules_path": Mock(),
                "axiom_compose_path": Mock(),
            },
        )

    assert code == 0
    assert executed == []
    assert report["scope"] == "incremental"
    assert report["selected"] == 0
    assert "fallback_reason" not in report


def test_incremental_companion_change_reports_selection_reason(audit_repo, capsys):
    companion = audit_repo.paths[0].removesuffix(".yaml") + ".test.yaml"
    (audit_repo.root / companion).write_text("cases: []\n", encoding="utf-8")
    code, report, executed = _run_audit(
        audit_repo, capsys, changed=[companion], scope="incremental"
    )

    assert code == 0
    assert executed == [audit_repo.paths[0]]
    assert report["scope"] == "incremental"
    assert report["selected"] == report["checked"] == 1
    assert report["total_waivers"] == 4
    assert set(report["selection_reasons"]) == {audit_repo.paths[0]}
    assert companion in "; ".join(report["selection_reasons"][audit_repo.paths[0]])


def test_partitioning_applies_after_selection_and_is_exhaustive(audit_repo, capsys):
    reports = []
    executed_parts = []
    for key in ("shard-a", "shard-b"):
        code, report, executed = _run_audit(
            audit_repo,
            capsys,
            changed=[audit_repo.paths[0]],
            scope="incremental",
            partition_key=key,
        )
        assert code == 0
        assert report["scope"] == "incremental"
        assert report["selected"] == len(executed) == report["checked"]
        assert report["total_waivers"] == 4
        assert report["partition"]["total_paths"] == 3
        assert set(report["selection_reasons"]) == set(executed)
        reports.append(report)
        executed_parts.append(set(executed))

    assert reports[0]["selected"] == 2
    assert reports[1]["selected"] == 1
    assert executed_parts[0].isdisjoint(executed_parts[1])
    assert set.union(*executed_parts) == set(audit_repo.paths[:3])


@pytest.mark.parametrize(
    "changed",
    [
        "known-validation-gaps.yaml",
        ".axiom/toolchain.toml",
        ".github/workflows/validate.yml",
        "corpus.lock",
        "programs/us/snap/program.yaml",
        "unknown.txt",
    ],
)
def test_incremental_full_fallback_keeps_all_paths_and_explains_why(
    audit_repo,
    capsys,
    changed,
):
    code, report, executed = _run_audit(
        audit_repo, capsys, changed=[changed], scope="incremental"
    )

    assert code == 0
    assert executed == list(audit_repo.paths)
    assert report["scope"] == "full"
    assert report["selected"] == report["checked"] == report["total_waivers"] == 4
    assert changed in report["fallback_reason"]
    assert set(report["selection_reasons"]) == set(audit_repo.paths)


def test_full_fallback_still_partitions(audit_repo, capsys):
    code, report, executed = _run_audit(
        audit_repo,
        capsys,
        changed=["known-validation-gaps.yaml"],
        scope="incremental",
        partition_key="shard-b",
    )

    assert code == 0
    assert report["scope"] == "full"
    assert report["selected"] == 2
    assert report["total_waivers"] == report["partition"]["total_paths"] == 4
    assert executed == [audit_repo.paths[1], audit_repo.paths[3]]


def test_incremental_cannot_skip_protected_base_transition_checks(audit_repo, capsys):
    code, report, executed = _run_audit(
        audit_repo,
        capsys,
        changed=["docs/guide.md"],
        scope="incremental",
        transition_issues=("unapproved active waiver mutation",),
    )

    assert code == 1
    assert executed == []
    assert report["success"] is False
    assert report["checked"] == 0
    assert report["errors"] == ["unapproved active waiver mutation"]


def test_deleted_removed_waiver_is_reported_without_fingerprinting(audit_repo, capsys):
    deleted = audit_repo.paths[0]
    (audit_repo.root / deleted).unlink()
    remaining = _waiver_set(
        {
            path: entry
            for path, entry in audit_repo.waivers.entries.items()
            if path != deleted
        }
    )

    code, report, executed = _run_audit(
        audit_repo,
        capsys,
        changed=[deleted],
        scope="incremental",
        head=remaining,
    )

    assert code == 0
    assert report["scope"] == "full"
    assert report["selected"] == report["total_waivers"] == 4
    assert report["checked"] == 4
    assert executed == list(audit_repo.paths[1:])
    assert deleted in report["fallback_reason"]
    assert report["total_waivers"] == 4
    assert deleted not in executed
    assert deleted in report["selection_reasons"]
    assert next(item for item in report["results"] if item["path"] == deleted) == {
        "path": deleted,
        "kind": "removed",
        "success": True,
        "module_deleted": True,
    }


def test_incremental_text_report_includes_scope_counts_and_reason(audit_repo, capsys):
    changed = ".axiom/toolchain.toml"
    code, text_report, _ = _run_audit(
        audit_repo,
        capsys,
        changed=[changed],
        scope="incremental",
        as_json=False,
    )

    assert code == 0
    assert "scope: full" in text_report.lower()
    assert "4" in text_report
    assert changed in text_report
    assert "fallback" in text_report.lower()


@pytest.mark.parametrize("as_json", [True, False])
@pytest.mark.parametrize(
    ("changed", "expected_scope", "expected_paths"),
    [
        (
            "us-ca/statutes/a.yaml",
            "incremental",
            ["us-ca/statutes/b.yaml"],
        ),
        (
            ".axiom/toolchain.toml",
            "full",
            ["us-ca/statutes/b.yaml", "us-ny/statutes/z.yaml"],
        ),
    ],
)
def test_fingerprint_exception_preserves_partitioned_selection_report(
    audit_repo,
    capsys,
    changed,
    expected_scope,
    expected_paths,
    as_json,
):
    audit_repo.changed_file.write_text(f"{changed}\n", encoding="utf-8")
    args = _audit_args(
        audit_repo.root,
        audit_repo.base_file,
        audit_repo.changed_file,
        audit_repo.root / "corpus",
    )
    args.validation_waivers_command = "audit"
    args.scope = "incremental"
    args.partition_key = "shard-b"
    args.partition_keys_json = '["shard-b", "shard-a"]'
    args.json = as_json

    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            return_value=audit_repo.waivers,
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=(),
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules_parallel",
            side_effect=RuntimeError("fingerprint worker failed"),
        ) as fingerprint,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.cmd_validation_waivers(args)

    assert exc_info.value.code == 2
    assert [path.as_posix() for path in fingerprint.call_args.args[0]] == expected_paths
    output = capsys.readouterr()
    if as_json:
        report = json.loads(output.out)
        assert report["success"] is False
        assert report["errors"] == ["fingerprint worker failed"]
        assert report["scope"] == expected_scope
        assert report["selected"] == len(expected_paths)
        assert report["total_waivers"] == 4
        assert set(report["selection_reasons"]) == set(expected_paths)
        for path in expected_paths:
            assert changed in "; ".join(report["selection_reasons"][path])
        if expected_scope == "full":
            assert changed in report["fallback_reason"]
        else:
            assert "fallback_reason" not in report
    else:
        assert output.out == ""
        assert (
            f"Audit scope: {expected_scope}; selected {len(expected_paths)} "
            "of 4 waiver path(s)"
        ) in output.err
        assert "fingerprint worker failed" in output.err
        assert changed in output.err
        for path in expected_paths:
            assert path in output.err
        assert ("Full audit fallback:" in output.err) == (expected_scope == "full")


def test_reused_args_default_full_clears_incremental_exception_metadata(
    audit_repo, capsys
):
    audit_repo.changed_file.write_text(".axiom/toolchain.toml\n", encoding="utf-8")
    args = _audit_args(
        audit_repo.root,
        audit_repo.base_file,
        audit_repo.changed_file,
        audit_repo.root / "corpus",
    )
    args.validation_waivers_command = "audit"
    args.scope = "incremental"

    with (
        patch.object(
            cli._validation_waivers,
            "load_validation_waivers",
            return_value=audit_repo.waivers,
        ),
        patch.object(
            cli._validation_waivers,
            "protected_base_transition_issues",
            return_value=(),
        ),
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules_parallel",
            side_effect=RuntimeError("fingerprint worker failed"),
        ),
    ):
        with pytest.raises(SystemExit) as incremental_exit:
            cli.cmd_validation_waivers(args)
        assert incremental_exit.value.code == 2
        incremental_report = json.loads(capsys.readouterr().out)
        assert incremental_report["selected"] == 4
        assert incremental_report["selection_reasons"]
        assert ".axiom/toolchain.toml" in incremental_report["fallback_reason"]

        del args.scope
        with pytest.raises(SystemExit) as full_exit:
            cli.cmd_validation_waivers(args)

    assert full_exit.value.code == 2
    assert not hasattr(args, "_validation_waiver_audit_selection")
    assert json.loads(capsys.readouterr().out) == {
        "success": False,
        "errors": ["fingerprint worker failed"],
        "scope": "full",
    }


@pytest.mark.parametrize("scope", [None, "full", "incremental"])
def test_cli_registers_scope_and_preserves_full_default(tmp_path: Path, scope):
    argv = [
        "axiom-encode",
        "validation-waivers",
        "audit",
        "--root",
        str(tmp_path),
        "--protected-base",
        str(tmp_path / "base.yaml"),
        "--changed-paths",
        str(tmp_path / "changed.txt"),
        "--corpus-path",
        str(tmp_path),
        "--axiom-rules-engine-path",
        str(tmp_path / "engine"),
    ]
    if scope is not None:
        argv.extend(["--scope", scope])
    with (
        patch("sys.argv", argv),
        patch.object(cli, "cmd_validation_waivers") as command,
    ):
        cli.main()

    assert command.call_args.args[0].scope == (scope or "full")


def test_cli_rejects_unknown_scope(tmp_path: Path):
    argv = [
        "axiom-encode",
        "validation-waivers",
        "audit",
        "--root",
        str(tmp_path),
        "--protected-base",
        str(tmp_path / "base.yaml"),
        "--changed-paths",
        str(tmp_path / "changed.txt"),
        "--corpus-path",
        str(tmp_path),
        "--axiom-rules-engine-path",
        str(tmp_path / "engine"),
        "--scope",
        "changed",
    ]
    with patch("sys.argv", argv), pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
