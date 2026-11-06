"""Source records and private execution authority remain separate."""

import copy
import json
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness.bound_capital_gift_operands import PARENT
from axiom_encode.harness.capital_gift_evidence import (
    _NUMERIC_SOURCE_ROLES,
    CapitalGiftEvidence,
    _branch_owners,
    _gate_spans,
    certify_capital_gift,
)
from axiom_encode.harness.source_completeness import SourceStructureBranch
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests.test_capital_gift_execution_capture import STATIC
from tests.test_capital_gift_refusal_capture import mixed, stage_all

SOURCE = (Path(__file__).parent / "fixtures/t1170-authenticated-body.txt").read_text()


def branch(start, end, kind="exception-clause"):
    return SourceStructureBranch((), kind, "source unit", SOURCE[start:end], start, end)


def test_exact_closed_record_compositions_no_global_offset_or_trim_credit():
    context = certify_capital_gift(STATIC["policy_yaml"], SOURCE, PARENT, [], None)
    assert _gate_spans(context.source) == ((0, 899), (250, 1063))
    assert set(_branch_owners(context.source)) == {
        (0, 899),
        (900, 1013),
        (1014, 1063),
        (1064, 1328),
        (1329, 1510),
        (1850, 2028),
    }
    for start, end in _branch_owners(context.source):
        actual = branch(start, end)
        assert context.owns_branch(actual, "control")
        assert not context.owns_branch(replace(actual, start=start + 1), "control")
        assert not context.owns_branch(
            replace(actual, text=actual.text + " "), "control"
        )
    assert context.missing()
    assert not CapitalGiftEvidence(context.source, True).owns_branch(
        branch(0, 899), "control"
    )
    assert not replace(context).owns_branch(branch(0, 899), "control")
    for changed in (
        SOURCE + " Do not claim this amount.",
        SOURCE.replace("within 30 days", "within 31 days"),
    ):
        invalid = certify_capital_gift(STATIC["policy_yaml"], changed, PARENT, [], None)
        assert invalid.source is None and invalid.missing()
        assert not invalid.live()
        assert not invalid.owns_branch(branch(0, 899), "control")
        ordinary = certify_capital_gift(
            STATIC["policy_yaml"], changed, "unknown/source", [], None
        )
        assert ordinary.source is None and not ordinary.missing()


def test_unobserved_and_duplicate_owned_obligations_never_generic_fallback():
    context = certify_capital_gift(STATIC["policy_yaml"], SOURCE, PARENT, [], None)
    actual = branch(0, 899)
    other = branch(900, 1013)
    remaining, missing = context.partition(
        (actual, copy.deepcopy(actual), other), "control"
    )
    assert remaining == ()
    assert len(missing) == 3


def test_complete_typed_numeric_inventory_has_distinct_closed_record_roles():
    context = certify_capital_gift(STATIC["policy_yaml"], SOURCE, PARENT, [], None)
    records = {r.role: r for r in context.source.records}
    expected = {
        (
            records[role].span[0] + a,
            records[role].span[0] + b,
            raw,
            value,
            source_value,
            rate,
        )
        for role, a, b, raw, value, source_value, rate in _NUMERIC_SOURCE_ROLES
    }
    actual = extract_typed_numeric_inventory_occurrences_from_text(SOURCE)
    assert {
        (o.start, o.end, o.raw, o.value, o.source_value, o.has_rate_context)
        for o in actual
    } == expected
    assert len(expected) == 52
    assert all(context.numeric_owner(o) is None for o in actual)


def test_historical_direct_only_capture_cannot_certify_provider_or_new_principal_roles(
    tmp_path, monkeypatch
):
    from axiom_encode.harness import capital_gift_capture as capture

    cap, cases, calls = mixed.__wrapped__(tmp_path, monkeypatch)
    stage_all(cap, cases, calls)
    evidence = cap.evidence()
    observed = capture.observation(evidence, STATIC["policy_yaml"])
    assert observed is not None and len(observed.refusals) == 18
    context = certify_capital_gift(
        STATIC["policy_yaml"], SOURCE, PARENT, cases, evidence
    )
    assert context.source is not None and not context.live() and context.missing()


@pytest.fixture
def complete(tmp_path, monkeypatch):
    """All124 actual records in original order, synthetic custody only."""
    from axiom_encode.harness import capital_gift_capture as capture
    from axiom_encode.harness import same_case_execution as execution
    from tests.test_capital_gift_execution_capture import ACTUAL, CASES, ENGINE_PIN
    from tests.test_capital_gift_refusals import REFUSALS

    extra = json.loads(
        (Path(__file__).parent / "fixtures/t1170-phase-c-additions.json").read_text()
    )
    cases = copy.deepcopy(
        CASES
        + [REFUSALS["separately_executed_positive"]]
        + REFUSALS["cases"]
        + extra["cases"]
    )
    calls = copy.deepcopy(
        ACTUAL["calls"]
        + [REFUSALS["separately_executed_positive_call"]]
        + REFUSALS["calls"]
        + extra["calls"]
    )
    original, snapshot = tmp_path / "rulespec-ca", tmp_path / "snapshot/rulespec-ca"
    relative = "ca/policies/cra/t1-2025/capital-gains-on-gifts.yaml"
    files = []
    for name, raw in (
        (relative, STATIC["policy_yaml"].encode()),
        (
            relative.replace(".yaml", ".test.yaml"),
            yaml.safe_dump(cases, sort_keys=False).encode(),
        ),
    ):
        for root in (original, snapshot):
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(raw)
        files.append(execution.ClosureFile(name, original / name, raw))
    binary, artifact = tmp_path / "binary", tmp_path / "compiled.json"
    binary.write_bytes(b"synthetic custody, no engine or signature authority")
    artifact.write_text(json.dumps(ACTUAL["artifact"]))
    cap = execution.ExecutionCapture(
        original,
        snapshot,
        relative,
        tuple(files),
        binary,
        execution.digest(binary.read_bytes()),
        ENGINE_PIN,
        0,
        "synthetic-head",
    )
    monkeypatch.setattr(
        capture,
        "_sources",
        lambda release: (SOURCE, STATIC["p113_body"], ("synthetic resolver",)),
    )
    monkeypatch.setattr(
        execution,
        "_materialized_capture",
        lambda obj, ref=weakref.ref(cap): obj is ref(),
    )
    assert capture.activate(SimpleNamespace(local_corpus_release=object()), cap)
    cap._capital_gift_owned = True
    cap.compiled(artifact)
    stage_all(cap, cases, calls)
    return cap, cases


def test_complete_actual_records_mint_only_through_private_source_factory(complete):
    cap, cases = complete
    evidence = cap.evidence()
    context = certify_capital_gift(
        STATIC["policy_yaml"], SOURCE, PARENT, cases, evidence
    )
    assert context.live() and not context.missing()
    for start, end in _branch_owners(context.source):
        _, missing = context.partition((branch(start, end),), "control")
        assert not missing
    for occurrence in extract_typed_numeric_inventory_occurrences_from_text(SOURCE):
        assert context.numeric_owner(occurrence) is not None
        assert (
            context.numeric_owner(replace(occurrence, start=occurrence.start + 1))
            is None
        )
    from axiom_encode.harness.bound_capital_gift_operands import _load

    rules = _load(STATIC["policy_yaml"])["rules"]
    gate_names = {
        "federal_2025_t1170_gift_capital_gain_column_5",
        "t1170_exchange_extension_conditions_met",
        "federal_2025_t1170_taxable_capital_gain_from_gift",
        "t1170_no_advantage_in_respect_of_gift",
        *(
            "t1170_" + category + "_table_" + output
            for category in ("shares_units", "bonds_other", "ecological_land")
            for output in ("proceeds", "zero_inclusion", "other_gain")
        ),
    }
    for rule in rules:
        if rule["name"] not in gate_names:
            continue
        arithmetic = rule["name"] in {
            "federal_2025_t1170_taxable_capital_gain_from_gift",
            "t1170_no_advantage_in_respect_of_gift",
        }
        start, end = _gate_spans(context.source)[int(arithmetic)]
        clause = SimpleNamespace(
            start=start, end=end, branch_path=(), text=SOURCE[start:end]
        )
        formula = rule["versions"][0]["formula"]
        args = (rule["name"], rule, 0, formula, "2025-01-01", "2025-12-31", clause)
        assert context.covers_gate(*args)
        assert not context.covers_gate(*args[:2], True, *args[3:])
        assert not context.covers_gate(*args[:3], formula + " + 0", *args[4:])
        changed_rule = copy.deepcopy(rule)
        changed_rule["versions"][0]["effective_to"] = "2026-12-31"
        assert not context.covers_gate(rule["name"], changed_rule, *args[2:])
        shifted = SimpleNamespace(
            start=start + 1, end=end, branch_path=(), text=clause.text
        )
        assert not context.covers_gate(*args[:-1], shifted)
        assert not context.covers_gate(*args[:5], "2026-12-31", clause)
    for occurrence in extract_typed_numeric_inventory_occurrences_from_text(SOURCE):
        if occurrence.raw not in {"50", "2025", "13286"}:
            continue
        for mutation in (
            {"raw": occurrence.raw + " "},
            {"value": occurrence.value + 1.0},
            {"source_value": occurrence.source_value + 1.0},
            {"has_rate_context": not occurrence.has_rate_context},
            {"requires_rate_context": not occurrence.requires_rate_context},
            {"has_structural_context": True},
            {"alternative_values": (occurrence.value,)},
        ):
            assert context.numeric_owner(replace(occurrence, **mutation)) is None
    assert not replace(context).live()
    duplicate = branch(0, 899)
    _, missing = context.partition((duplicate, copy.deepcopy(duplicate)), "control")
    assert len(missing) == 2
    changed = copy.deepcopy(cases)
    changed[0]["name"] += " changed"
    assert not certify_capital_gift(
        STATIC["policy_yaml"], SOURCE, PARENT, changed, evidence
    ).live()


@pytest.fixture(scope="module")
def complete_raw():
    from tests.test_capital_gift_execution_capture import ACTUAL, CASES
    from tests.test_capital_gift_refusals import REFUSALS, bound

    extra = json.loads(
        (Path(__file__).parent / "fixtures/t1170-phase-c-additions.json").read_text()
    )
    return (
        bound.__wrapped__(),
        CASES
        + [REFUSALS["separately_executed_positive"]]
        + REFUSALS["cases"]
        + extra["cases"],
        ACTUAL["calls"]
        + [REFUSALS["separately_executed_positive_call"]]
        + REFUSALS["calls"]
        + extra["calls"],
        ACTUAL["artifact"],
    )


def raw_observation(bound, cases, calls, artifact):
    """Recompute records after coherent omissions, without minting custody."""
    from axiom_encode.harness.capital_gift_capture import _mixed_observation
    from axiom_encode.harness.capital_gift_execution import capital_gift_case_key

    keys = tuple(capital_gift_case_key(case, bound) for case in cases)
    positive, negative = [], []
    for index, (case, call) in enumerate(zip(cases, calls, strict=True), 1):
        record = SimpleNamespace(
            case_sha256=keys[index - 1],
            case_index=index,
            request_json=json.dumps(call["request"]),
            response_json=call["stdout"],
            stdout=call["stdout"],
            stderr=call["stderr"],
            returncode=call["returncode"],
        )
        (negative if "expected_refusal" in case else positive).append(record)
    context = SimpleNamespace(
        binding=bound,
        cases=cases,
        keys=keys,
        artifact=json.dumps(artifact),
        companion=yaml.safe_dump(cases, sort_keys=False),
    )
    return _mixed_observation(context, positive, negative)


@pytest.mark.parametrize(
    "target",
    [
        "t1170_shares_units_zero_inclusion_line_68230",
        "t1170_bonds_other_zero_inclusion_line_68250",
        "t1170_ecological_land_zero_inclusion_line_68252",
        "t1170_partnership_gain_for_schedule3_line17400",
    ],
)
def test_coherent_provider_family_omission_cannot_borrow_direct_refusals(
    complete_raw, target
):
    from axiom_encode.harness.capital_gift_evidence import source_role_witnesses
    from axiom_encode.harness.capital_gift_execution import PREFIX

    bound, cases, calls, artifact = complete_raw
    selected = [
        (case, call)
        for case, call in zip(cases, calls, strict=True)
        if case.get("expected_refusal", {}).get("output") != PREFIX + target
    ]
    assert len(selected) == len(cases) - 2
    observed = raw_observation(
        bound, [c for c, _ in selected], [r for _, r in selected], artifact
    )
    assert observed is not None and observed.refusals
    with pytest.raises(
        ValueError, match="missing source-owned role: provider:" + target
    ):
        source_role_witnesses(bound, observed)
