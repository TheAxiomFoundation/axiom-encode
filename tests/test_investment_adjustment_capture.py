"""Synthetic custody around saved actual records, not fresh execution authority."""

import copy
import json
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import investment_adjustment_capture as capture_module
from tests.test_investment_adjustment_refusal import (
    ARTIFACT,
    CONTENT,
    DATA,
    ENGINE_PIN,
    GUIDE,
    PARENT,
)


@pytest.fixture
def n_capture(tmp_path, monkeypatch):
    from axiom_encode.harness import same_case_execution as s

    cases = [copy.deepcopy(DATA["positive_case"]), copy.deepcopy(DATA["negative_case"])]
    calls = [copy.deepcopy(DATA["positive_call"]), copy.deepcopy(DATA["negative_call"])]
    original, snapshot = tmp_path / "rulespec-ca", tmp_path / "snapshot" / "rulespec-ca"
    relative = "ca/policies/revenu-quebec/tp1-2025/schedule-n-investment-expense-adjustment.yaml"
    files = []
    for path, raw in [
        (relative, CONTENT.encode()),
        (
            relative.replace(".yaml", ".test.yaml"),
            yaml.safe_dump(cases, sort_keys=False).encode(),
        ),
    ]:
        for root in (original, snapshot):
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        files.append(s.ClosureFile(path, original / path, raw))
    binary, artifact = tmp_path / "binary", tmp_path / "compiled.json"
    binary.write_bytes(b"synthetic: no actual runtime authority")
    artifact.write_text(json.dumps(ARTIFACT))
    cap = s.ExecutionCapture(
        original,
        snapshot,
        relative,
        tuple(files),
        binary,
        s.digest(binary.read_bytes()),
        ENGINE_PIN,
        0,
        "synthetic-head",
    )
    monkeypatch.setattr(
        capture_module,
        "_sources",
        lambda release: (PARENT, GUIDE, ("synthetic named source resolver",)),
    )
    monkeypatch.setattr(
        s, "_materialized_capture", lambda obj, ref=weakref.ref(cap): obj is ref()
    )
    assert capture_module.activate(SimpleNamespace(local_corpus_release=object()), cap)
    cap._investment_adjustment_owned = True
    cap.compiled(artifact)
    assert cap._investment_adjustment_context is not None
    return cap, cases, calls


@pytest.fixture
def t_capture(tmp_path, monkeypatch):
    from tests.test_capital_gift_refusal_capture import mixed

    return mixed.__wrapped__(tmp_path / "t1170-isolated", monkeypatch)


def stage_all(cap, cases, calls):
    for i, (case, call) in enumerate(zip(cases, calls, strict=True), 1):
        if "expected_refusal" in case:
            assert cap.stage_refusal(
                case,
                call["request"],
                call["returncode"],
                call["stdout"],
                call["stderr"],
                cap.binary,
                cap.artifact,
                i,
            )
        else:
            cap.stage(
                case,
                call["request"],
                json.loads(call["stdout"]),
                cap.binary,
                cap.artifact,
                i,
            )


def test_private_n_partition_has_no_generic_or_complete_source_authority(n_capture):
    cap, cases, calls = n_capture
    stage_all(cap, cases, calls)
    evidence = cap.evidence()
    observed = capture_module.observation(evidence, CONTENT)
    assert observed is not None
    assert (
        len(observed.positive.records)
        == len(observed.refusals)
        == len(observed.pairs)
        == 1
    )
    assert len(observed.numeric_owners) == 83
    assert observed.coverage.missing
    assert capture_module.observation(replace(evidence), CONTENT) is None
    assert evidence.values(cases[0], CONTENT) is None


@pytest.mark.parametrize(
    "mutation",
    ["case", "ordinal", "error", "status", "compiled", "source", "foreign_owner"],
)
def test_refusal_and_source_context_cannot_be_borrowed_or_stale(n_capture, mutation):
    cap, cases, calls = n_capture
    case, call = copy.deepcopy(cases[1]), copy.deepcopy(calls[1])
    ordinal = 2
    if mutation == "case":
        case["name"] += "mutant"
    elif mutation == "ordinal":
        ordinal = 1
    elif mutation == "error":
        call["stderr"] += "extra\n"
    elif mutation == "status":
        call["returncode"] = 0
    elif mutation == "compiled":
        cap.artifact.write_text("{}")
    elif mutation == "source":
        (cap.snapshot_root / cap.root_relative).write_text(CONTENT + "\n")
    elif mutation == "foreign_owner":
        cap._capital_gift_owned = True
    assert not cap.stage_refusal(
        case,
        call["request"],
        call["returncode"],
        call["stdout"],
        call["stderr"],
        cap.binary,
        cap.artifact,
        ordinal,
    )


def test_missing_positive_pair_or_refusal_never_yields_complete_observation(n_capture):
    cap, cases, calls = n_capture
    stage_all(cap, cases[:1], calls[:1])
    assert capture_module.observation(cap.evidence(), CONTENT) is None


def test_cross_owner_context_cannot_authorize_n_or_t(n_capture, t_capture):
    from axiom_encode.harness import capital_gift_capture
    from tests.test_capital_gift_execution_capture import STATIC

    n, n_cases, n_calls = n_capture
    t, _, _ = t_capture
    assert capture_module.valid(n._investment_adjustment_context, n)
    assert capital_gift_capture.valid(t._capital_gift_context, t)
    original_n = n._investment_adjustment_context
    n._investment_adjustment_context = t._capital_gift_context
    assert not capture_module.valid(n._investment_adjustment_context, n)
    t._capital_gift_context = original_n
    assert not capital_gift_capture.valid(t._capital_gift_context, t)
    assert capture_module.observation(t.evidence(), CONTENT) is None
    assert capital_gift_capture.observation(n.evidence(), STATIC["policy_yaml"]) is None


@pytest.mark.parametrize(
    "seed_name", ["_child_credit_seed", "_retirement_transfer_seed"]
)
def test_recompile_clears_scoped_contexts_on_cross_family_overlap(n_capture, seed_name):
    cap, cases, calls = n_capture
    stage_all(cap, cases, calls)
    assert cap._investment_adjustment_context is not None
    cap._capital_gift_context = object()
    cap._child_credit_context = object()
    cap._retirement_transfer_context = object()
    cap._capital_gift_refusals.append(object())
    cap._investment_adjustment_refusals.append(object())
    setattr(cap, seed_name, object())
    cap.compiled(cap.artifact)
    assert cap.records == []
    assert cap._capital_gift_refusals == []
    assert cap._investment_adjustment_refusals == []
    assert cap._capital_gift_context is None
    assert cap._investment_adjustment_context is None
    assert cap._child_credit_context is None
    assert cap._retirement_transfer_context is None
    assert cap.evidence() is None
