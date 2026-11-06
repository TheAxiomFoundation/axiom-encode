"""Synthetic snapshot construction around immutable actual82 records.

These tests mock source resolution, not a signature or trusted runtime claim.
The ordinary genuine-checkout replay remains a separate integration requirement.
"""

import copy
import gc
import json
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import capital_gift_capture as c
from axiom_encode.harness.capital_gift_execution import ENGINE_PIN
from axiom_encode.harness.same_case_execution import (
    ClosureFile,
    ExecutionCapture,
    digest,
)

F = Path(__file__).parent / "fixtures"
STATIC = json.loads((F / "t1170-static.json").read_text())
ACTUAL = json.loads((F / "t1170-execution.json").read_text())
CASES = yaml.safe_load(ACTUAL["companion_yaml"])
if isinstance(CASES, dict):
    CASES = CASES["cases"]


@pytest.fixture
def capture(tmp_path, monkeypatch):
    original, snapshot = tmp_path / "original", tmp_path / "snapshot"
    relative = "ca/policies/cra/t1-2025/capital-gains-on-gifts.yaml"
    files = []
    for name, raw in [
        (relative, STATIC["policy_yaml"].encode()),
        (relative.replace(".yaml", ".test.yaml"), ACTUAL["companion_yaml"].encode()),
    ]:
        for root in (original, snapshot):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        files.append(ClosureFile(name, original / name, raw))
    binary = tmp_path / "binary"
    binary.write_bytes(b"explicit synthetic binary")
    artifact = tmp_path / "compiled.json"
    artifact.write_text(json.dumps(ACTUAL["artifact"]))
    cap = ExecutionCapture(
        original,
        snapshot,
        relative,
        tuple(files),
        binary,
        digest(binary.read_bytes()),
        ENGINE_PIN,
        0,
        "synthetic-head",
    )
    source = (
        (F / "t1170-authenticated-body.txt").read_text(),
        STATIC["p113_body"],
        ("explicit synthetic resolver identities",),
    )
    monkeypatch.setattr(c, "_sources", lambda release: source)
    pipe = SimpleNamespace(local_corpus_release=object())
    # Explicit synthetic materializer adapter authorizes only this exact fixture.
    from axiom_encode.harness import same_case_execution

    monkeypatch.setattr(
        same_case_execution,
        "_materialized_capture",
        lambda x, ref=weakref.ref(cap): x is ref(),
    )
    assert c.activate(pipe, cap)
    cap._capital_gift_owned = True
    cap.compiled(artifact)
    assert cap._capital_gift_context is not None
    return cap


def stage(cap, index):
    call = ACTUAL["calls"][index]
    cap.stage(
        CASES[index],
        call["request"],
        json.loads(call["stdout"]),
        cap.binary,
        cap.artifact,
        case_index=index + 1,
    )


def test_actual82_private_capture_and_actual_consumer(capture):
    for i in range(82):
        stage(capture, i)
    evidence = capture.evidence()
    assert len(evidence.records) == 82
    assert evidence.values(CASES[0], STATIC["policy_yaml"]) is None
    observed = c.observation(evidence, STATIC["policy_yaml"])
    assert observed is not None
    assert sum(len(r.outputs) for r in observed.records) == 361
    # Actual consumer refuses equal caller-created evidence, not just bad hashes.
    assert c.observation(replace(evidence), STATIC["policy_yaml"]) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "ordinal",
        "case_name",
        "coherent_other_case",
        "Date_text",
        "duplicate",
        "source",
        "companion",
        "artifact",
        "binary",
        "forged_context",
        "wrong_kind",
        "metadata",
    ],
)
def test_private_capture_refuses_before_any_generic_fallback(
    capture, monkeypatch, mutation
):
    index = 0
    if mutation == "Date_text":
        index = next(
            i
            for i, x in enumerate(CASES)
            if any(hasattr(v, "isoformat") for v in x["output"].values())
        )
    case = copy.deepcopy(CASES[index])
    call = copy.deepcopy(ACTUAL["calls"][index])
    response = json.loads(call["stdout"])
    ordinal = index + 1
    if mutation == "ordinal":
        ordinal = 2
    elif mutation == "case_name":
        case["name"] = "replaced-raw-case-name"
    elif mutation == "coherent_other_case":
        case = copy.deepcopy(CASES[1])
        call = copy.deepcopy(ACTUAL["calls"][1])
        response = json.loads(call["stdout"])
    elif mutation == "Date_text":
        case["output"] = {
            k: v.isoformat() if hasattr(v, "isoformat") else v
            for k, v in case["output"].items()
        }
    elif mutation == "duplicate":
        stage(capture, index)
    elif mutation == "source":
        monkeypatch.setattr(c, "_sources", lambda release: ("wrong", "wrong", ()))
    elif mutation == "companion":
        next(
            f.original for f in capture.files if f.relative.endswith(".test.yaml")
        ).write_text("[]")
    elif mutation == "artifact":
        capture.artifact.write_text("{}")
    elif mutation == "binary":
        capture.binary.write_bytes(b"changed")
    elif mutation == "forged_context":
        capture._capital_gift_context = replace(capture._capital_gift_context)
    elif mutation == "wrong_kind":
        call["request"]["dataset"]["inputs"][0]["entity"] = "Person"
    elif mutation == "metadata":
        response["metadata"]["unknown_override"] = True
    before = len(capture.records)
    capture.stage(
        case,
        call["request"],
        response,
        capture.binary,
        capture.artifact,
        case_index=ordinal,
    )
    assert len(capture.records) == before


def test_caller_constructed_capture_cannot_mint(capture):
    forged = replace(capture)
    forged._capital_gift_owned = True
    forged.compiled(capture.artifact)
    assert forged._capital_gift_context is None
    stage(forged, 0)
    assert not forged.records


def test_actual_consumer_rechecks_fresh_binary_artifact_and_closure(capture):
    for i in range(82):
        stage(capture, i)
    evidence = capture.evidence()
    assert c.observation(evidence, STATIC["policy_yaml"]) is not None
    for path in (
        capture.binary,
        capture.artifact,
        capture.files[0].original,
        capture.snapshot_root / capture.files[1].relative,
    ):
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        assert c.observation(evidence, STATIC["policy_yaml"]) is None
        path.write_bytes(original)
    assert c.observation(evidence, STATIC["policy_yaml"]) is not None


@pytest.mark.parametrize("raw", ["cases: [", "cases: &x [*x]", "cases: []\ncases: []"])
def test_companion_malformed_duplicate_cycle_refuses_mint(capture, raw):
    companion = next(f for f in capture.files if f.relative.endswith(".test.yaml"))
    companion.original.write_text(raw)
    (capture.snapshot_root / companion.relative).write_text(raw)
    capture.files = tuple(
        replace(f, raw=raw.encode()) if f is companion else f for f in capture.files
    )
    capture.compiled(capture.artifact)
    assert capture._capital_gift_context is None


def test_caller_activation_replay_cannot_grant_actual_consumer(capture):
    forged = replace(capture)
    forged.records = []
    forged._capital_gift_owned = True
    if c.activate(SimpleNamespace(local_corpus_release=object()), forged):
        forged.compiled(capture.artifact)
        for i in range(82):
            stage(forged, i)
    evidence = forged.evidence()
    assert evidence is None or c.observation(evidence, STATIC["policy_yaml"]) is None


def test_borrowed_context_cannot_grant_equal_capture_consumer(capture):
    for i in range(82):
        stage(capture, i)
    forged = replace(capture)
    forged.records = list(capture.records)
    forged._capital_gift_owned = True
    forged._capital_gift_context = capture._capital_gift_context
    evidence = forged.evidence()
    assert evidence is None or c.observation(evidence, STATIC["policy_yaml"]) is None


def test_evidence_survives_helper_return_but_registry_releases_on_gc(
    capture, monkeypatch
):
    from axiom_encode.harness import same_case_execution

    def helper():
        local = replace(capture)
        local.records = []
        ref = weakref.ref(local)
        monkeypatch.setattr(
            same_case_execution, "_materialized_capture", lambda x: x is ref()
        )
        assert c.activate(SimpleNamespace(local_corpus_release=object()), local)
        local._capital_gift_owned = True
        local.compiled(capture.artifact)
        for i in range(82):
            stage(local, i)
        return local.evidence(), ref

    evidence, capture_ref = helper()
    gc.collect()
    assert capture_ref() is not None
    assert c.observation(evidence, STATIC["policy_yaml"]) is not None
    key = id(evidence)
    assert key in c._EVIDENCE
    # Evidence does not confer life beyond the byte-bound snapshot.
    snapshot_file = capture.snapshot_root / capture.files[0].relative
    raw = snapshot_file.read_bytes()
    snapshot_file.unlink()
    assert c.observation(evidence, STATIC["policy_yaml"]) is None
    snapshot_file.write_bytes(raw)
    del evidence
    gc.collect()
    assert key not in c._EVIDENCE
    assert capture_ref() is None


def test_coherent_closure_replacement_cannot_remint_same_invocation(capture):
    raw_cases = copy.deepcopy(CASES)
    raw_cases[0]["name"] = "caller-coherently-replaced-case"
    raw = yaml.safe_dump(raw_cases, sort_keys=False).encode()
    companion = next(f for f in capture.files if f.relative.endswith(".test.yaml"))
    companion.original.write_bytes(raw)
    (capture.snapshot_root / companion.relative).write_bytes(raw)
    capture.files = tuple(
        replace(f, raw=raw) if f is companion else f for f in capture.files
    )
    assert capture.fresh()
    assert not c.activate(SimpleNamespace(local_corpus_release=object()), capture)
    capture.compiled(capture.artifact)
    assert capture._capital_gift_context is None


def test_registered_other_capture_cannot_borrow_context(capture, monkeypatch):
    from axiom_encode.harness import same_case_execution

    for i in range(82):
        stage(capture, i)
    second = replace(capture)
    second.records = list(capture.records)
    release = c._CAPTURES[id(capture)][3]
    monkeypatch.setattr(
        same_case_execution,
        "_materialized_capture",
        lambda x: x is capture or x is second,
    )
    assert c.activate(SimpleNamespace(local_corpus_release=release), second)
    second._capital_gift_owned = True
    second._capital_gift_context = capture._capital_gift_context
    assert second.evidence() is None
