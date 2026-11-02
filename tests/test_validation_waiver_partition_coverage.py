"""Partition coverage must survive different discovery outcomes on each shard."""

from __future__ import annotations

import io
import json
import tempfile
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode import validation_waiver_scope as scope
from tests.test_validation_waiver_audit_cli import (
    _audit_args,
    _entry,
    _metadata,
    _waiver_set,
)
from tests.test_validation_waiver_scope import _module


def _partition_audits(root, paths, affected, outcomes):
    """Run real selection and CLI partitioning with deterministic fingerprints."""
    for path in paths:
        _module(root, path)
        (root / path.replace(".yaml", ".test.yaml")).write_text("cases: []\n")
    waivers = _waiver_set(
        {path: _entry(path, active=_metadata("sha256:base")) for path in paths}
    )
    base = root / "base.yaml"
    base.write_text("validate_failures: {}\n")
    changed = root / "changed.txt"
    changed.write_text(
        "".join(path.replace(".yaml", ".test.yaml") + "\n" for path in affected)
    )
    keys = [f"shard-{index:03}" for index in range(len(outcomes))]

    def fingerprint(modules, **kwargs):
        return [
            {
                "path": module.as_posix(),
                "passed": False,
                "fingerprint": (
                    "sha256:drift" if module.as_posix() in affected else "sha256:base"
                ),
                "outcome": {},
            }
            for module in modules
        ]

    reports = []
    for key, outcome in zip(keys, outcomes, strict=True):
        args = _audit_args(root, base, changed, root / "corpus")
        args.scope = "incremental"
        args.partition_key = key
        # Ownership must also be independent of manifest ordering.
        args.partition_keys_json = json.dumps(list(reversed(keys)))
        if outcome == "fallback":
            selection_context = patch.object(
                cli,
                "select_validation_waiver_scope",
                return_value=scope.WaiverAuditScope(
                    "full",
                    tuple(paths),
                    {path: ["full audit fallback: uncertain input"] for path in paths},
                    "uncertain input",
                ),
            )
        elif outcome == "discovery-error":
            selection_context = patch.object(
                scope._ReadSets,
                "read_set",
                side_effect=OSError(24, "Too many open files"),
            )
        else:
            selection_context = nullcontext()
        output = io.StringIO()
        with (
            patch.object(
                cli._validation_waivers, "load_validation_waivers", return_value=waivers
            ),
            patch.object(
                cli._validation_waivers,
                "protected_base_transition_issues",
                return_value=(),
            ),
            patch.object(
                cli,
                "_fingerprint_validation_waiver_modules_parallel",
                side_effect=fingerprint,
            ),
            patch.object(
                cli, "_fingerprint_validation_waiver_modules", side_effect=fingerprint
            ),
            selection_context,
            redirect_stdout(output),
        ):
            code = cli._cmd_validation_waivers_audit(args)
        reports.append((code, json.loads(output.getvalue())))
    return reports


def test_shard_local_discovery_error_cannot_hide_changed_waiver(tmp_path):
    paths = (
        "us-ca/statutes/1/1.yaml",
        "us-ca/statutes/1/2.yaml",
        "us-ca/statutes/1/3.yaml",
        "us-ny/statutes/1/4.yaml",
    )
    affected = {paths[-1]}
    reports = _partition_audits(
        tmp_path / "rulespec-us", paths, affected, ("discovery-error", "incremental")
    )

    assert reports[0][1]["scope"] == "full"
    assert "Too many open files" in reports[0][1]["fallback_reason"]
    assert reports[1][1]["scope"] == "incremental"
    audited = {item["path"] for _, report in reports for item in report["results"]}
    assert affected <= audited, reports
    assert any(code != 0 for code, _ in reports), reports
    assert any(
        item.get("isolated_recheck") and "fingerprint drifted" in item.get("error", "")
        for _, report in reports
        for item in report["results"]
    )


@st.composite
def _partition_cases(draw):
    size = draw(st.integers(min_value=1, max_value=10))
    affected = draw(st.sets(st.integers(min_value=0, max_value=size - 1)))
    outcomes = draw(
        st.lists(
            st.sampled_from(("incremental", "fallback", "discovery-error")),
            min_size=1,
            max_size=12,
        )
    )
    return size, affected, outcomes


@settings(max_examples=100, deadline=None, database=None)
@example((4, {3}, ["discovery-error", "incremental"]))
@example((4, {3}, ["fallback", "incremental"]))
@example((4, set(range(4)), ["incremental"] * 12))
@given(_partition_cases())
def test_partition_union_covers_affected_waivers_or_fails(case):
    size, changed_indices, outcomes = case
    paths = tuple(f"us-ca/statutes/1/{index}.yaml" for index in range(size))
    affected = {paths[index] for index in changed_indices}
    with tempfile.TemporaryDirectory() as temporary:
        reports = _partition_audits(
            Path(temporary) / "rulespec-us", paths, affected, outcomes
        )

    audited = {item["path"] for _, report in reports for item in report["results"]}
    assert affected <= audited or any(code != 0 for code, _ in reports), reports
    # Stable ledger ownership actually guarantees coverage even when drift
    # already makes one shard fail; a failure must not conceal other omissions.
    assert affected <= audited, reports
    assert all(report["checked"] == report["selected"] for _, report in reports)
    assert all(
        set(report["selection_reasons"]) == {item["path"] for item in report["results"]}
        for _, report in reports
    )
