import copy
import dataclasses
import functools

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_composition_evidence as comp
from axiom_encode.harness import worksheet_exception_evidence as eq
from axiom_encode.harness import worksheet_operation_evidence as op
from tests.test_worksheet_exception_evidence import C, S
from tests.test_worksheet_exception_evidence import data as equality_data


@pytest.fixture(scope="module")
def data():
    bound, equality, witnesses, rules, asserted = equality_data.__wrapped__()
    composition = comp.bind_worksheet_composition(bound)
    assert not composition.evidence.unresolved
    branches = sc.recognize_source_structure(S)
    worksheet = next(
        b
        for b in sc._source_exception_branches(
            S, branches=branches, active_branches=branches, deferred_paths=set()
        )
        if (b.start, b.end) == comp.SPAN
    )
    return bound, composition, equality, worksheet, witnesses, rules, asserted


def kwargs(d):
    b, c, e, w, ws, r, a = d
    return dict(
        source_text=S,
        corpus_citation_path=C,
        principal_rules=r,
        principal_rule_paths={},
        asserted_by_rule=a,
        toggled_exception_selectors=ws,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )


def test_combined_real_consumer_allocation_and_annual(data):
    b, c, e, w, *_ = data
    legacy_annual = {(17, 29)}
    new_annual = set(legacy_annual)
    baseline = sc._unwitnessed_exception_branches(
        (e, w), represented_annual_spans=legacy_annual, **kwargs(data)
    )
    assert set(baseline) == {e, w}
    assert not sc._unwitnessed_exception_branches(
        (e, w),
        represented_annual_spans=new_annual,
        certified_replacement=b,
        certified_composition=c,
        **kwargs(data),
    )
    assert legacy_annual == new_annual == {(17, 29)}
    branches = tuple(
        dataclasses.replace(e, path=("eq", str(i))) for i in range(4)
    ) + tuple(dataclasses.replace(w, path=("ws", str(i))) for i in range(2))
    missing = sc._unwitnessed_exception_branches(
        branches, certified_replacement=b, certified_composition=c, **kwargs(data)
    )
    assert len(missing) == 2
    assert sum(x.start == e.start for x in missing) == 1
    assert sum(x.start == w.start for x in missing) == 1


@pytest.mark.parametrize(
    "owner", [op._LINE5, op._LINE2, "net_income_for_multiple_jurisdictions"]
)
def test_shared_bound_private_metadata_invalidation(data, owner):
    b, c, e, w, ws, r, a = copy.deepcopy(data)
    # Remap existing witness IDs to actual copied cases, avoiding incidental
    # clone rejection as the reason the equality route fails.
    idmap = {id(x): id(y) for x, y in zip(data[0].cases, b.cases)}
    ws = {
        dataclasses.replace(
            x, case_pair_identity=tuple(idmap[i] for i in x.case_pair_identity)
        )
        for x in ws
    }
    assert (
        len(eq.existing_witnesses(b, e, ws, principal_rules=r, asserted_by_rule=a)) == 3
    )
    assert c.resource_for(w, r)
    i = next(i for i, x in enumerate(b.payload["rules"]) if x["name"] == owner)
    original = b.payload["rules"][i]
    b.payload["rules"][i] = sc._CurrencyRoundingRule(
        original, sc._CurrencyRounding("floor", 0)
    )
    r = op.paths._index(b.payload["rules"])
    assert not b.evidence.matches(b.payload, b.cases, b.source, b.context, b.bundle)
    assert not eq.existing_witnesses(b, e, ws, principal_rules=r, asserted_by_rule=a)
    assert c.resource_for(w, r) is None
