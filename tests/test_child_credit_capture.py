"""Transport unit tests use actual records but synthetic local provenance.

The separate real pipeline run establishes internal resolver/snapshot minting.
These tests do not authenticate a binary, checkout, or source release.
"""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.harness.child_credit_execution import _capture_source_binding
from axiom_encode.harness.same_case_execution import (
    ClosureFile,
    ExecutionCapture,
    digest,
)

ROOT = "ca/policies/cra/t1-2025/ns479-credits.yaml"
CITATION = "ca/policy/cra/t1-2025/ns479-credits"
FIXTURE = Path(__file__).parent / "fixtures/child_credit/observed-graph.json"
COMPANION = FIXTURE.with_name("cases.yaml")


def data():
    return json.loads(FIXTURE.read_text())


def capture(tmp_path, d, *, seed=True, companion_raw=None):
    original = tmp_path / "original"
    snapshot = tmp_path / "snapshot"
    files = []
    if companion_raw is None:
        companion_raw = (
            COMPANION.read_bytes()
            if d["cases"] == data()["cases"]
            else json.dumps(d["cases"]).encode()
        )
    for relative, raw in [
        (ROOT, d["content"].encode()),
        (ROOT.replace(".yaml", ".test.yaml"), companion_raw),
    ]:
        for root in (original, snapshot):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        files.append(ClosureFile(relative, original / relative, raw))
    binary = tmp_path / "engine"
    binary.write_bytes(b"synthetic-unit-test-binary")
    c = ExecutionCapture(
        original,
        snapshot,
        ROOT,
        tuple(files),
        binary,
        digest(binary.read_bytes()),
        d["engine_pin"],
        0,
        "a" * 40,
    )
    if seed:
        c._child_credit_seed = _capture_source_binding(
            d["content"], {CITATION: d["source"]}, ROOT
        )
        assert c._child_credit_seed is not None
    artifact = tmp_path / "compiled.json"
    artifact.write_text(json.dumps(d["artifact"]))
    c.compiled(artifact)
    return c


def stage(c, d, index=0):
    r = d["records"][index]
    c.stage(
        d["cases"][index],
        r["request"],
        r["response"],
        c.binary,
        c.artifact,
        case_index=r["case_index"],
    )


def test_all_actual17_transport_records_and_recompile_reset(tmp_path):
    d = data()
    c = capture(tmp_path, d)
    assert c.files[1].raw == COMPANION.read_bytes()
    assert b"&id" in c.files[1].raw and b"*id" in c.files[1].raw
    assert c._child_credit_context is not None
    for i in range(17):
        stage(c, d, i)
    assert len(c.records) == 17
    assert c.evidence() is not None
    c.compiled(c.artifact)
    assert c.records == [] and c._child_credit_context is not None


def test_old_strict_route_unchanged_without_private_seed(tmp_path):
    d = data()
    c = capture(tmp_path, d, seed=False)
    stage(c, d, 0)
    assert not c.records
    stage(c, d, 7)
    assert len(c.records) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "compiled_slot",
        "compiled_catalog",
        "compiled_unit",
        "compiled_format",
        "unsupported_pin",
        "forged_seed",
        "missing_companion",
        "duplicate_cases",
    ],
)
def test_invalid_context_never_falls_back_to_same_entity_case(tmp_path, defect):
    d = data()
    if defect == "compiled_slot":
        d["artifact"]["program"]["relations"][0]["slot_entities"].reverse()
    elif defect == "compiled_catalog":
        d["artifact"]["metadata"]["input_catalog"][0]["request_names"].append("foreign")
    elif defect == "compiled_unit":
        d["artifact"]["program"]["units"][-1]["minor_units"] = 0
    elif defect == "compiled_format":
        d["artifact"]["artifact_format_version"] = True
    elif defect == "unsupported_pin":
        d["engine_pin"] = "0" * 40
    elif defect == "duplicate_cases":
        d["cases"].append(copy.deepcopy(d["cases"][0]))
    c = capture(tmp_path, d)
    if defect == "forged_seed":
        c._child_credit_seed = replace(c._child_credit_seed, content_sha256="0" * 64)
        c.compiled(c.artifact)
    elif defect == "missing_companion":
        c.files = c.files[:1]
        c.compiled(c.artifact)
    assert c._child_credit_context is None
    stage(c, d, 7)
    assert not c.records and c.evidence() is None


@pytest.mark.parametrize(
    "defect",
    [
        "case_name",
        "case_index",
        "case_bool_index",
        "actual_kind",
        "foreign_trace",
        "relation_reverse",
        "pins",
        "assessment",
        "binary",
        "artifact",
        "original",
        "snapshot",
        "companion",
        "context_root",
        "context_pin",
        "context_hash",
    ],
)
def test_capture_boundaries_refuse(tmp_path, defect):
    d = data()
    c = capture(tmp_path, d)
    r = d["records"][0]
    if defect == "case_name":
        d["cases"][0]["name"] += "changed"
    elif defect == "case_index":
        r["case_index"] = 2
    elif defect == "case_bool_index":
        r["case_index"] = True
    elif defect == "actual_kind":
        next(
            f for f in r["request"]["dataset"]["inputs"] if f["entity"] == "SourceEntry"
        )["entity"] = "Person"
    elif defect == "foreign_trace":
        trace = r["response"]["results"][0]["trace"]
        key = next(k for k in trace if "@entity:" in k)
        trace[key]["entity_id"] = "unrelated"
    elif defect == "relation_reverse":
        r["request"]["dataset"]["relations"][0]["tuple"].reverse()
    elif defect == "pins":
        r["request"]["pins"] = []
    elif defect == "assessment":
        r["request"]["queries"][0]["assessment_date"] = "2025-12-31"
        r["response"]["results"][0]["assessment_date"] = "2025-12-31"
    elif defect == "binary":
        c.binary.write_bytes(b"changed")
    elif defect == "artifact":
        c.artifact.write_text(c.artifact.read_text() + " ")
    elif defect == "original":
        c.files[0].original.write_text(d["content"] + "\n")
    elif defect == "snapshot":
        (c.snapshot_root / ROOT).write_text(d["content"] + "\n")
    elif defect == "companion":
        c.files[1].original.write_bytes(b"[]")
    elif defect == "context_root":
        c._child_credit_context = replace(
            c._child_credit_context, root_relative="ca/policies/other.yaml"
        )
    elif defect == "context_pin":
        c._child_credit_context = replace(c._child_credit_context, engine_pin="0" * 40)
    elif defect == "context_hash":
        c._child_credit_context = replace(
            c._child_credit_context, artifact_sha256="0" * 64
        )
    stage(c, d)
    assert not c.records


@pytest.mark.parametrize(
    "defect", ["absent", "wrong_hash", "wrong_path", "alias_path", "imports"]
)
def test_no_source_or_path_metadata_grants_eligibility(defect):
    d = data()
    source = {CITATION: d["source"]}
    path = ROOT
    content = d["content"]
    if defect == "absent":
        source = {}
    elif defect == "wrong_hash":
        source[CITATION] += "\n"
    elif defect == "wrong_path":
        path = ROOT.replace("ns479-credits", "other")
    elif defect == "alias_path":
        path = "ca//policies/cra/t1-2025/ns479-credits.yaml"
    elif defect == "imports":
        content = content.replace("imports: []", "imports: [foreign]")
    assert _capture_source_binding(content, source, path) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"- &cycle {name: cycle, input: {self: *cycle}}\n",
        b"- name: one\n  name: two\n",
        b"- name: bad\n  input: {x: .nan}\n",
    ],
)
def test_ordinary_companion_unsafe_structures_refuse_context(tmp_path, raw):
    d = data()
    c = capture(tmp_path, d, companion_raw=raw)
    assert c._child_credit_context is None
    stage(c, d, 7)
    assert not c.records
