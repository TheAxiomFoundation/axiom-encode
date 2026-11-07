"""Scoped dispatch controls with mocked private boundaries; no engine calls."""

from types import SimpleNamespace

import pytest

from axiom_encode.harness import mhrtc_capture as m
from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import scoped_execution as s
from axiom_encode.harness.validator_pipeline import ValidatorPipeline


@pytest.fixture
def capture(tmp_path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"unit fixture")
    artifact = tmp_path / "compiled.json"
    artifact.write_bytes(b"{}")
    cap = e.ExecutionCapture(
        tmp_path,
        tmp_path,
        "m.yaml",
        (),
        binary,
        e.digest(binary.read_bytes()),
        "pin",
        0.0,
        "head",
    )
    cap.artifact = artifact
    cap.artifact_sha256 = e.digest(artifact.read_bytes())
    cap._mhrtc_owned = True
    cap._mhrtc_context = object()
    return cap


def test_exact_owner_and_ambiguity(capture):
    assert s.adapter(capture) == (m, "_mhrtc")
    capture._investment_adjustment_owned = True
    assert s.adapter(capture) is None


def test_compile_invalidates_before_remint(capture, monkeypatch):
    calls = []
    monkeypatch.setattr(m, "clear", lambda c: calls.append(("clear", c)))
    context = object()

    def mint(c):
        calls.append(("mint", c))
        return context

    monkeypatch.setattr(m, "mint", mint)
    capture.compiled(capture.artifact)
    assert calls == [("clear", capture), ("mint", capture)]
    assert capture._mhrtc_context is context


@pytest.mark.parametrize("ambiguous", [False, True])
def test_stage_owned_never_generic_fallback(capture, monkeypatch, ambiguous):
    calls = []
    monkeypatch.setattr(e, "strict_explain_metadata", lambda *a: True)
    monkeypatch.setattr(m, "stage", lambda *a: calls.append(a))
    capture._investment_adjustment_owned = ambiguous
    capture.stage({}, {}, {}, capture.binary, capture.artifact, case_index=1)
    assert len(calls) == (0 if ambiguous else 1)
    assert capture.records == []


@pytest.mark.parametrize("accepted", [False, True])
def test_evidence_requires_private_registration(capture, monkeypatch, accepted):
    capture.records.append(e.CaseExecution("key", "{}", "{}", 1))
    calls = []

    def register(context, actual, evidence):
        calls.append((context, actual, evidence))
        return accepted

    monkeypatch.setattr(m, "register_evidence", register)
    result = capture.evidence()
    assert len(calls) == 1
    assert (result is not None) is accepted


def test_clear_all_ambiguous_owners(capture, monkeypatch):
    capture._investment_adjustment_owned = True
    capture._investment_adjustment_context = object()
    capture._investment_adjustment_refusals.append("retained")
    calls = []
    monkeypatch.setattr(m, "clear", lambda c: calls.append(c))
    s.clear_private(capture)
    assert calls == [capture]
    assert capture._investment_adjustment_context is None
    assert capture._investment_adjustment_refusals == []


@pytest.mark.parametrize(
    "owned,complete", [(True, False), (True, True), (False, False)]
)
def test_normal_pipeline_requires_complete_owned_evidence(
    tmp_path, monkeypatch, owned, complete
):
    rules = tmp_path / "m.yaml"
    rules.write_text("policy")
    cap = SimpleNamespace(
        _mhrtc_owned=owned, records=["record"], evidence=lambda: object()
    )
    calls = []
    clears = []

    def observe(*a):
        calls.append(a)
        return object() if complete else None

    monkeypatch.setattr(s, "observation", observe)
    monkeypatch.setattr(s, "clear_private", lambda c: clears.append(c))
    pipe = ValidatorPipeline(
        tmp_path, tmp_path, local_corpus_release=None, enable_oracles=False
    )
    issues = pipe._run_rulespec_test_cases(
        rules_file=rules,
        compiled_path=tmp_path / "compiled",
        compiled_payload={"program": {}},
        cases=[],
        execution_capture=cap,
    )
    assert len(calls) == int(owned)
    assert bool(issues) is (owned and not complete)
    assert bool(clears) is (owned and not complete)
    if issues:
        assert cap.records == []


@pytest.mark.parametrize("detached", [False, True])
def test_no_import_evidence_cannot_fall_back_after_owner_expiry(
    capture, monkeypatch, detached
):
    import json
    from dataclasses import replace

    case = {"name": "local", "period": "2025", "input": {}, "output": {"local": 0}}
    capture._mhrtc_owned = False
    capture.records.append(
        e.CaseExecution(
            e.case_key(case),
            json.dumps({"queries": [{"entity_id": "p"}]}),
            json.dumps({"results": [{"trace": {}}]}),
            1,
        )
    )
    result = capture.evidence()
    assert result is not None
    if detached:
        result = replace(result)
    monkeypatch.setattr(s, "owns_evidence", lambda _: False)
    assert (
        result.values(
            case,
            "format: rulespec/v1\nmodule: {}\nimports: []\nrules: []\ninputs: []\n",
        )
        is None
    )
