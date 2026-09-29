"""Large bounded diagnostics survive producer and independent verifier."""

import json
from types import SimpleNamespace

import pytest

from axiom_encode.cli import _emit_final_rejected_candidate
from axiom_encode.harness.evals import ValidationRetryCandidate
from axiom_encode.repair_candidate_contract import (
    FAILED_ENCODE_CANDIDATE_MAX_ISSUES_BYTES,
)
from scripts.verify_failed_encode_candidate import verify_candidate_directory


def emit(tmp_path, issue):
    output_root = tmp_path / "out"
    result = SimpleNamespace(
        runner="test", output_file=output_root / "test/statutes/example.yaml"
    )
    return _emit_final_rejected_candidate(
        result,
        output_root=output_root,
        destination=tmp_path / "failed",
        citation="us/statute/example",
        validation_issues=[issue],
        attempt_count=1,
        candidate_override=ValidationRetryCandidate(
            rulespec="format: rulespec/v1\nrules: []\n", tests="[]\n"
        ),
    )


def test_large_diagnostics_round_trip_without_truncating(tmp_path):
    issue = "source diagnostic " * 40000
    root = emit(tmp_path, issue)
    assert (root / "issues.json").stat().st_size > 512 * 1024
    assert json.loads((root / "issues.json").read_text())["issues"] == [issue]
    assert (
        verify_candidate_directory(root, citation="us/statute/example")["path"]
        == "statutes/example.yaml"
    )


def test_producer_rejects_diagnostics_over_shared_bound(tmp_path):
    with pytest.raises(ValueError, match="exceeds its size limit"):
        emit(tmp_path, "x" * FAILED_ENCODE_CANDIDATE_MAX_ISSUES_BYTES)
    assert not (tmp_path / "failed/issues.json").exists()


def test_independent_verifier_rejects_diagnostics_over_shared_bound(tmp_path):
    root = emit(tmp_path, "small")
    payload = json.loads((root / "issues.json").read_text())
    payload["issues"] = ["x" * FAILED_ENCODE_CANDIDATE_MAX_ISSUES_BYTES]
    (root / "issues.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="size limit"):
        verify_candidate_directory(root)
