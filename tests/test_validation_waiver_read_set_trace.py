"""Optional replay of recorded real-module reads against computed dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from axiom_encode.harness.validator_pipeline import _rulespec_resolution_cache_scope
from axiom_encode.repo_routing import _rulespec_routing_cache_scope
from axiom_encode.validation_waiver_scope import _ReadSets

_SAMPLE = json.loads(
    (Path(__file__).parent / "fixtures/validation_waiver_read_set_modules.json").read_text(
        encoding="utf-8"
    )
)
_TRACE_ROOT_ENV = "AXIOM_VALIDATION_WAIVER_TRACE_ROOT"
_TRACE_DIR_ENV = "AXIOM_VALIDATION_WAIVER_TRACE_DIR"


@pytest.fixture(scope="module")
def recorded_trace_evidence():
    root_value = os.environ.get(_TRACE_ROOT_ENV)
    trace_value = os.environ.get(_TRACE_DIR_ENV)
    if not root_value and not trace_value:
        pytest.skip(f"provide {_TRACE_ROOT_ENV} and {_TRACE_DIR_ENV} to replay traces")
    assert root_value and trace_value, "both trace environment variables are required"
    root = Path(root_value).resolve()
    trace_dir = Path(trace_value).resolve()
    assert root.is_dir(), "trace repository root must already exist"
    assert trace_dir.is_dir(), "recorded trace directory must already exist"
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    assert revision == _SAMPLE["source"]["revision"], "trace checkout revision differs"
    # Incremental selection shares the same production reader and caches
    # across modules in one immutable tree. Reuse that path for trace replay.
    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        yield root, trace_dir, _ReadSets(root, ())


@pytest.mark.parametrize(
    ("index", "module"),
    enumerate(_SAMPLE["modules"]),
    ids=_SAMPLE["modules"],
)
def test_recorded_module_reads_are_subset_of_computed_read_set(
    recorded_trace_evidence, index: int, module: str
):
    _root, trace_dir, reader = recorded_trace_evidence
    artifact = trace_dir / f"trace-{index:02}.json"
    assert artifact.is_file(), f"missing recorded trace for {module}"
    artifact_bytes = artifact.read_bytes()
    trace = json.loads(artifact_bytes)
    assert trace["module"] == module
    assert "error" not in trace, f"fingerprinting did not complete for {module}"
    assert trace["source_mutated"] == [], f"trace inputs changed for {module}"
    if trace.get("requires_chunk_directory_verification"):
        begin = trace["chunk_begin"]
        end = trace["chunk_end"]
        assert isinstance(begin, int) and not isinstance(begin, bool)
        assert isinstance(end, int) and not isinstance(end, bool)
        assert 0 <= begin <= index < end <= len(_SAMPLE["modules"])
        completion_name = f"input-snapshot-check-{begin:02}-{end:02}.json"
        assert trace["input_snapshot_check"] == completion_name
        completion = trace_dir / completion_name
        assert completion.is_file(), f"missing completed trace chunk for {module}"
        chunk = json.loads(completion.read_text(encoding="utf-8"))
        assert chunk.get("success") is True, f"trace chunk failed for {module}"
        assert chunk.get("begin") == begin and chunk.get("end") == end
        assert isinstance(trace["source_identity"], dict)
        assert chunk.get("source_identity") == trace["source_identity"]
        assert chunk.get("mutated") == [], f"trace chunk inputs changed for {module}"
        directory_check = chunk.get("directory_metadata_verification")
        assert isinstance(directory_check, dict)
        assert directory_check.get("mutated") == [], (
            f"trace chunk directory inputs changed for {module}"
        )
        bindings = chunk.get("trace_artifacts")
        assert isinstance(bindings, dict)
        assert bindings.get(artifact.name) == hashlib.sha256(artifact_bytes).hexdigest(), (
            f"trace artifact is not bound to its completed chunk for {module}"
        )
    result = trace.get("result")
    assert isinstance(result, dict), f"missing fingerprint result for {module}"
    assert result["path"] == module
    assert result["fingerprint"].startswith("sha256:")

    observed: set[str] = set()
    for field in (
        "python_open_paths",
        "native_open_paths",
        "child_native_open_paths",
        "index_open_paths",
    ):
        reads = trace[field]
        assert isinstance(reads, list)
        for path in reads:
            assert isinstance(path, str)
            relative = PurePosixPath(path)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert relative.as_posix() == path
        observed.update(reads)
    assert module in observed, f"trace did not record opening {module}"

    computed, _upstream_roots, composition = reader.read_set(module)
    native_evidence = trace["native_engine_evidence"]
    assert isinstance(native_evidence, dict)
    assert native_evidence["module_read_required"] is (not composition)
    if native_evidence["module_read_required"]:
        assert module in trace["child_native_open_paths"], (
            f"engine child did not open atomic module {module}"
        )
        assert native_evidence["module_opened_by_native_child"] is True
    else:
        assert native_evidence["limitation"], (
            f"missing composition engine tracing limitation for {module}"
        )
    assert observed <= computed, f"uncovered reads for {module}: {sorted(observed - computed)}"
