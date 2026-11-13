"""Lifecycle unit controls; these fixtures do not create engine/source authority."""

from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as s
from axiom_encode.harness.local_case_custody import _typed


@pytest.mark.parametrize("capture", [None, object(), SimpleNamespace(case_records=())])
def test_unregistered_capture_never_mints(capture):
    before = dict(s._contexts)
    with s.local_relation_source_scope(
        capture, object(), "policy", [], {"c": "body"}
    ) as token:
        assert token is None
    assert s._contexts == before


@pytest.mark.parametrize("token", [None, object(), {}, []])
def test_plain_or_forged_tokens_decline(token):
    assert (
        s.observed_dependencies(
            token,
            content="policy",
            case={},
            principal="amount",
            source_text="body",
            citation="c",
            principal_rules={},
        )
        is None
    )


@pytest.fixture
def bound(tmp_path, monkeypatch):
    artifact = tmp_path / "compiled.json"
    artifact.write_bytes(b"{}")
    receipt = SimpleNamespace(output=artifact, raw=b"{}", pipeline=object())
    records = (object(),)
    owner = SimpleNamespace(
        capture=object(),
        generation=1,
        receipt=receipt,
        consumed=True,
        case_records=records,
    )
    monkeypatch.setattr(
        s.custody, "_owner", lambda capture: owner if capture is owner.capture else None
    )
    monkeypatch.setattr(s.custody, "_fresh", lambda value: value is owner)
    seal = object()
    monkeypatch.setattr(
        s, "_live_seal", lambda value, pipeline: seal if value is owner else None
    )
    cases = [{"input": {"x": 1}}]
    sources = {"c": "body"}
    return s._Bound(
        owner,
        1,
        receipt,
        records,
        "policy",
        cases,
        _typed(cases),
        sources,
        _typed(sources),
        {},
        (),
        seal,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "generation",
        "receipt",
        "records",
        "consumed",
        "artifact",
        "cases",
        "source",
        "symlink",
    ],
)
def test_each_lifecycle_or_content_change_expires(bound, mutation, tmp_path):
    assert bound.fresh()
    if mutation == "generation":
        bound.owner.generation += 1
    elif mutation == "receipt":
        bound.owner.receipt = SimpleNamespace(**vars(bound.receipt))
    elif mutation == "records":
        bound.owner.case_records = tuple(list(bound.records))
    elif mutation == "consumed":
        bound.owner.consumed = False
    elif mutation == "artifact":
        bound.receipt.output.write_bytes(b'{"changed":true}')
    elif mutation == "cases":
        bound.cases[0]["input"]["x"] = True
    elif mutation == "source":
        bound.sources["c"] = "different body"
    else:
        other = tmp_path / "other"
        other.write_bytes(b"{}")
        bound.receipt.output.unlink()
        bound.receipt.output.symlink_to(other)
    assert not bound.fresh()


@pytest.mark.parametrize("assertion", ["qualified", "friendly", "foreign", "absent"])
def test_asserted_principal_identity_in_observer_only(assertion):
    # Injected private fixture tests name routing only, not factory admission.
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values
    from tests.test_actual_local_relation_evidence import _actual_case

    policy, request, response = _actual_case()
    ref = request["queries"][0]["outputs"][0]
    name = ref.split("#", 1)[1]
    output_key = {
        "qualified": ref,
        "friendly": name,
        "foreign": "other#" + name,
        "absent": "different",
    }[assertion]
    case = {"output": {output_key: 0}}
    principal = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    token = object()
    s._contexts[token] = SimpleNamespace(
        fresh=lambda: True,
        content="policy",
        sources={"c": "body"},
        policy=policy,
        observations=(
            (
                frozenset({_typed(case)}),
                request,
                response,
                _replay_values(policy, request, response),
            ),
        ),
    )
    try:
        result = s.observed_dependencies(
            token,
            content="policy",
            case=case,
            principal=name,
            source_text="body",
            citation="c",
            principal_rules=principal,
        )
        assert (result is not None) == (assertion in {"qualified", "friendly"})
    finally:
        s._contexts.pop(token)


def test_existing_analyzer_normalization_has_exact_aliases_only():
    from datetime import date
    from decimal import Decimal

    policy = {
        "inputs": [{"name": "when", "dtype": "Date"}],
        "rules": [{"name": "amount", "dtype": "Money"}],
    }
    raw = [
        {
            "input": {"when": "2025-01-01", "other": "2025-01-01"},
            "output": {"amount": "1.00"},
        }
    ]
    _, normalized = s._normalized_source_inputs(policy, raw)
    assert normalized[0]["input"] == {"when": date(2025, 1, 1), "other": "2025-01-01"}
    assert normalized[0]["output"]["amount"] == Decimal("1.00")
    assert raw[0]["output"]["amount"] == "1.00"
    assert s._source_key(normalized) != s._source_key(raw)
    assert s._source_key(True) != s._source_key(1)
    assert s._source_key(Decimal("1.00")) != s._source_key(1)
    assert s._source_key(Decimal("1.00")) != s._source_key(Decimal("1.0"))


def test_normalized_duplicate_cases_stay_ambiguous():
    from datetime import date

    policy = {"inputs": [{"name": "when", "dtype": "Date"}], "rules": []}
    raw = [{"input": {"when": "2025-01-01"}}, {"input": {"when": date(2025, 1, 1)}}]
    _, normalized = s._normalized_source_inputs(policy, raw)
    keys = [
        frozenset({s._source_key(r), s._source_key(n)}) for r, n in zip(raw, normalized)
    ]
    assert len([k for k in keys if s._source_key(normalized[0]) in k]) == 2


def test_rounding_wrapper_identity_requires_exact_resolved_stamp():
    from axiom_encode.harness.source_completeness import (
        _CurrencyRounding,
        _CurrencyRoundingRule,
    )

    policy = {
        "units": [{"name": "CAD", "kind": "currency", "minor_units": 2}],
        "rules": [
            {"name": "amount", "kind": "derived", "unit": "CAD", "rounding": "half_up"}
        ],
    }
    normalized, _ = s._normalized_source_inputs(policy, [])
    rule = normalized["rules"][0]
    assert s._source_key(rule) != s._source_key(dict(rule))
    other = _CurrencyRoundingRule(dict(rule), _CurrencyRounding("half_up", 3))
    assert s._source_key(other) != s._source_key(rule)

    class Forged(_CurrencyRoundingRule):
        pass

    with pytest.raises(ValueError):
        s._source_key(Forged(dict(rule), rule.resolved_currency_rounding))


@pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_normalized_numbers_decline(text):
    from decimal import Decimal

    with pytest.raises(ValueError):
        s._source_key(Decimal(text))


@pytest.mark.parametrize("token", [None, object(), {}, []])
def test_forged_branch_corroboration_has_no_values(token):
    assert s.corroborated_dependencies(token, {}) is None
    assert not s.corroborates(token, {}, "amount", 10)
    assert s.corroborated_execution(token, {"name": "amount"}, {}, {}) is None


def test_verified_sum_execution_retains_authored_sum_leaf():
    from axiom_encode.harness import source_completeness as source
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values
    from tests.test_actual_local_relation_evidence import _actual_case

    policy, request, response = _actual_case()
    values = _replay_values(policy, request, response)
    query = request["queries"][0]
    root = query["entity_id"]
    ref = query["outputs"][0]
    name = ref.split("#", 1)[1]
    case = {
        "input": {},
        "output": {ref: values[(name, root)]},
        "period": query["period"],
    }
    rules = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    context, witness = object(), object()
    # Private routing fixture only: successful process custody is tested separately.
    s._contexts[context] = SimpleNamespace(
        fresh=lambda: True,
        content="policy",
        sources={"c": "body"},
        policy=policy,
        observations=((frozenset({s._source_key(case)}), request, response, values),),
    )
    s._witnesses[witness] = s._Witness(
        context, "policy", case, name, "body", "c", rules
    )
    try:
        reached = s.corroborated_dependencies(witness, case)
        assert reached
        sum_rule = next(
            r
            for n, r in rules.items()
            if n in reached
            and str(r["versions"][0]["formula"]).strip().startswith("sum(")
        )
        execution = s.corroborated_execution(
            witness, sum_rule, case, source._constant_rule_environment(policy)
        )
        assert execution is not None
        assert execution.leaf == sum_rule["versions"][0]["formula"].strip()
        assert (
            source._formula_execution_runtime_value(execution)
            == reached[sum_rule["name"]]
        )
        changed = {**sum_rule, "unit": "USD"}
        assert s.corroborated_execution(witness, changed, case, {}) is None
        s._contexts.pop(context)
        assert s.corroborated_execution(witness, sum_rule, case, {}) is None
    finally:
        s._contexts.pop(context, None)
        s._witnesses.pop(witness, None)


def test_same_value_wrong_cutoff_cannot_change_verified_selected_path():
    import json
    from dataclasses import replace
    from datetime import date
    from pathlib import Path

    from axiom_encode.harness import source_completeness as source
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values
    from tests.test_actual_local_relation_evidence import _actual_case

    data = json.loads(
        (Path(__file__).parent / "fixtures/t936_provider_date_replay.json").read_text()
    )
    policy = _actual_case()[0]
    policy, cases = s._normalized_source_inputs(policy, [data["case"]])
    case = cases[0]
    request, response = data["request"], data["response"]
    values = _replay_values(policy, request, response)
    rules = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    name = "t936_ordinary_vehicle_cost_excluded"
    context, witness = object(), object()
    s._contexts[context] = SimpleNamespace(
        fresh=lambda: True,
        content="policy",
        sources={"c": "body"},
        policy=policy,
        observations=((frozenset({s._source_key(case)}), request, response, values),),
    )
    s._witnesses[witness] = s._Witness(
        context, "policy", case, name, "body", "c", rules
    )
    try:
        environment = source._constant_rule_environment(policy)
        original = environment["t936_start_2022"]
        changed = replace(
            original,
            versions=tuple(
                (lo, hi, replace(value, cells=((2025, date(2021, 1, 1)),)))
                for lo, hi, value in original.versions
            ),
        )
        baseline = s.corroborated_execution(witness, rules[name], case, environment)
        supplied = s.corroborated_execution(
            witness, rules[name], case, {**environment, "t936_start_2022": changed}
        )
        assert baseline is not None and supplied is not None
        assert supplied.trace == baseline.trace
        assert supplied.leaf == baseline.leaf
        assert source._formula_execution_runtime_value(supplied) is False
        altered_case = {**case, "period": {**case["period"], "start": "2025-02-01"}}
        assert (
            s.corroborated_execution(witness, rules[name], altered_case, environment)
            is None
        )
    finally:
        s._contexts.pop(context, None)
        s._witnesses.pop(witness, None)


@pytest.mark.parametrize("outcome", ["fresh", "expired", "exception"])
def test_formula_coverage_commits_only_with_live_observation(monkeypatch, outcome):
    # Private seam test of transaction rollback, not engine evidence admission.
    from axiom_encode.harness import source_completeness as source

    case = {"output": {"amount": 1}}
    live = True
    monkeypatch.setattr(
        s, "corroborated_dependencies", lambda *args: {"amount": 1} if live else None
    )
    monkeypatch.setattr(s, "corroborated_execution", lambda *args: object())
    monkeypatch.setattr(s, "_canonical_formula_context", lambda *args: ({}, case))
    monkeypatch.setattr(
        source, "_asserted_formula_runtime_values_equal", lambda *args: True
    )
    keys = (
        "represented_annual_spans",
        "represented_measurement_spans",
        "measurement_witnesses",
    )
    originals = {key: {("existing", key)} for key in keys}

    def check(branch, **kwargs):
        nonlocal live
        for key in keys:
            assert kwargs[key] is not originals[key]
            kwargs[key].add(("new", key))
        if outcome == "exception":
            raise ValueError("declined witness")
        live = outcome != "expired"
        return {("amount", "witness")}

    monkeypatch.setattr(source, "_formula_branch_test_witnesses", check)
    before = dict(s._witnesses)
    result = s.formula_witnesses(
        object(),
        object(),
        content="policy",
        principal_rules={"amount": {"name": "amount"}},
        rule_names={"amount"},
        asserted_by_rule={"amount": [case]},
        source_text="source",
        corpus_citation_path="citation",
        formula_environment={},
        **originals,
    )
    assert result == ({("amount", "witness")} if outcome == "fresh" else set())
    for key in keys:
        assert originals[key] == (
            {("existing", key), ("new", key)}
            if outcome == "fresh"
            else {("existing", key)}
        )
    assert s._witnesses == before


@pytest.mark.parametrize("expired", [False, True])
def test_site_replacement_preserves_one_claim_per_operation(bound, expired):
    # Private registry seam: test graph conservation, not evidence admission.
    from axiom_encode.harness.source_completeness import _unmatched_evidence_obligations

    context = object()
    s._contexts[context] = bound
    mul1, mul2, total = object(), object(), object()
    old = ("amount", "case:1")
    site = ("amount", "case:1|site:left/right|node:abc")
    graph = {mul1: {old}, mul2: {old}, total: {old}}
    try:
        if expired:
            bound.owner.generation += 1
        for branch in (mul1, mul2):
            token = object()
            s._site_replacements[token] = (context, branch, old, site)
            s.apply_operation_replacements(context, branch, graph[branch], [token])
            assert token not in s._site_replacements
        assert graph[total] == {old}
        assert graph[mul1] == graph[mul2] == ({old} if expired else {site})
        # Two distinct operations can cover two clauses, never three claims.
        assert len(_unmatched_evidence_obligations(graph)) == (2 if expired else 1)
    finally:
        s._contexts.pop(context, None)


def test_unregistered_replacement_and_wrong_context_cannot_change_graph(bound):
    context = object()
    other = object()
    branch = object()
    old, new = ("rule", "case"), ("rule", "site")
    s._contexts[context] = bound
    token = object()
    s._site_replacements[token] = (other, branch, old, new)
    try:
        candidates = {old}
        s.apply_operation_replacements(context, branch, candidates, [object(), token])
        assert candidates == {old}
        assert token not in s._site_replacements
    finally:
        s._contexts.pop(context, None)
