"""Partition coverage must survive process-local discovery failures."""

import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from axiom_encode.validation_waiver_scope import _ReadSets
from tests import test_validation_waiver_incremental_invariant as invariant


def test_shard_local_discovery_error_cannot_hide_real_fingerprint_drift(real_engine):
    """A changed real verdict is checked, or at least one shard must fail.

    The complete ledger contains three California waivers before the New York
    waiver. Only New York's companion changes; only shard-a loses a discovery
    read. Validation, fingerprints and import resolution use the real engine.
    """

    case = invariant._RepositoryCase(
        jurisdictions=("us-ca", "us-ca", "us-ca", "us-ca", "us-ny"),
        values=(1, 1, 1, 1, 1),
        imports=((), (), (), (), ()),
        edit="companion",
        target=4,
    )
    target = case.module_path(case.target)
    original_audit = invariant._full_audit
    original_read_set = _ReadSets.read_set
    observed = {}

    def check_partition_coverage(root, corpus, engine, protected_base, changed_file):
        report = original_audit(root, corpus, engine, protected_base, changed_file)
        if "base" not in observed:
            observed["base"] = report
            assert report["success"], report
            return report

        observed["head"] = report
        base_verdicts = invariant._verdicts(observed["base"])
        head_verdicts = invariant._verdicts(report)
        affected = {
            path for path in base_verdicts if base_verdicts[path] != head_verdicts[path]
        }
        assert affected == {target}, (observed["base"], report)
        assert not report["success"], report
        assert any(
            "fingerprint drifted" in error and target in error
            for error in report["errors"]
        ), report

        def run(shard=None):
            args = SimpleNamespace(
                root=root,
                corpus_path=corpus,
                axiom_rules_path=engine,
                protected_base=protected_base,
                changed_paths=changed_file,
                scope="incremental",
                json=True,
            )
            if shard is not None:
                args.partition_key = shard
                args.partition_keys_json = '["shard-a", "shard-b"]'
            invariant._rulespec_executable_index_for_roots.cache_clear()
            output = io.StringIO()
            with redirect_stdout(output):
                code = invariant.cli._cmd_validation_waivers_audit(args)
            return code, json.loads(output.getvalue())

        unpartitioned_code, unpartitioned = run()
        assert unpartitioned_code != 0, unpartitioned
        assert affected <= {entry["path"] for entry in unpartitioned["results"]}

        discovery_errors = []

        def transient(reader, path, **kwargs):
            if path == target:
                discovery_errors.append(path)
                raise OSError(24, "Too many open files")
            return original_read_set(reader, path, **kwargs)

        with patch.object(_ReadSets, "read_set", transient):
            code_a, report_a = run("shard-a")
        code_b, report_b = run("shard-b")
        assert discovery_errors == [target]
        assert report_a["scope"] == "full", report_a
        assert report_b["scope"] == "incremental", report_b
        audited = {
            entry["path"]
            for partition in (report_a, report_b)
            for entry in partition["results"]
        }
        assert affected <= audited or code_a != 0 or code_b != 0, (
            affected,
            code_a,
            report_a,
            code_b,
            report_b,
        )
        observed["partitions"] = (report_a, report_b)
        return report

    with patch.object(invariant, "_full_audit", check_partition_coverage):
        invariant._check_invariant(case, real_engine.root)
    assert "partitions" in observed
