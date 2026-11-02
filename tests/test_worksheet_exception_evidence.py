"""Portable existing-witness controls using real producer and consumer APIs."""

import copy
import dataclasses
import functools
from pathlib import Path

import pytest
import yaml

from tests.test_worksheet_transfer_evidence import CONTEXT
from tests.test_worksheet_transfer_evidence import SOURCE as S

C = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness import worksheet_replacement_evidence as r
from axiom_encode.harness import worksheet_transfer_evidence as tr
from axiom_encode.harness.worksheet_exception_evidence import SPAN, existing_witnesses


@pytest.fixture(scope="module")
def data():
    fixtures = Path(__file__).parent / "fixtures" / "worksheet_exceptions"
    p = sc._bind_currency_rounding_rules(
        yaml.safe_load((fixtures / "policy.yaml").read_text())
    )
    cases = sc._typed_date_cases(
        sc._typed_numeric_expected_cases(
            yaml.safe_load((fixtures / "cases.yaml").read_text()),
            {rule["name"]: rule for rule in p["rules"]},
        ),
        p,
    )
    bundle = tr.certify_worksheet_transfers(
        p,
        source_text=S,
        corpus_citation_path=C,
        source_context=CONTEXT,
        test_cases=cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )
    assert not bundle.unresolved
    bound = r.bind_worksheet_replacement(p, cases, S, CONTEXT, bundle)
    assert not bound.evidence.unresolved
    rules = op.paths._index(p["rules"])
    asserted = {
        n: [c for c in cases if n in sc._test_case_output_names(c)] for n in rules
    }
    witnesses = sc._toggled_formula_numeric_selectors(
        rules,
        asserted_by_rule=asserted,
        formula_environment=sc._constant_rule_environment(p),
    )
    branches = sc.recognize_source_structure(S)
    branch = next(
        b
        for b in sc._source_exception_branches(
            S, branches=branches, active_branches=branches, deferred_paths=set()
        )
        if (b.start, b.end) == SPAN
    )
    return bound, branch, witnesses, rules, asserted


def accepted(d):
    b, branch, witnesses, rules, asserted = d
    return existing_witnesses(
        b, branch, witnesses, principal_rules=rules, asserted_by_rule=asserted
    )


def test_actual_existing_resources_and_allocation(data):
    found = accepted(data)
    assert len(found) == 3
    assert all(any(x is y for y in data[2]) for x in found)
    witness = next(iter(found))
    branch = data[1]
    duplicate = dataclasses.replace(branch, path=("duplicate",))
    assert (
        len(
            sc._unmatched_evidence_obligations(
                {branch: {witness}, duplicate: {witness}}
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "kind", ["context", "case", "source", "alias", "proof", "date"]
)
def test_stale_bound(data, kind):
    b, branch, witnesses, rules, asserted = data
    # Change original referenced data in a fully self-consistent deepcopy, so
    # master membership alone cannot mask stale identity guards.
    b, branch, witnesses, rules, asserted = copy.deepcopy(data)
    # IDs in existing witnesses need current copied master objects.
    old_to_new = {id(a): id(c) for a, c in zip(data[0].cases, b.cases)}
    witnesses = {
        dataclasses.replace(
            w, case_pair_identity=tuple(old_to_new[i] for i in w.case_pair_identity)
        )
        for w in witnesses
    }
    if kind == "context":
        b.context[op._SECONDARY] += "\nchanged"
    elif kind == "case":
        b.cases[6]["output"] = {}
    elif kind == "source":
        b = dataclasses.replace(b, source=b.source + "\n")
    elif kind == "alias":
        next(x for x in b.payload["rules"] if x["name"] == op._LINE1)["versions"][0][
            "formula"
        ] += " + 1"
    elif kind == "proof":
        next(x for x in b.payload["rules"] if x["name"] == op._LINE5)["metadata"][
            "proof"
        ]["atoms"] = []
    else:
        next(x for x in b.payload["rules"] if x["name"] == op._LINE5)["versions"][0][
            "effective_to"
        ] = "2025-06-30"
    assert not accepted((b, branch, witnesses, rules, asserted))


def test_extra_partial_claimant_not_borrowed(data):
    b, branch, witnesses, rules, asserted = data
    rules = dict(rules)
    extra = copy.deepcopy(rules[op._LINE5])
    extra["name"] = "unrelated_credit"
    extra["versions"][0]["formula"] = "0"
    extra["metadata"]["proof"]["atoms"] = [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {"corpus_citation_path": C, "excerpt": branch.text},
        }
    ]
    rules[extra["name"]] = extra
    assert not accepted((b, branch, witnesses, rules, asserted))


def test_supplied_alias_mismatch_not_ignored(data):
    b, branch, witnesses, rules, asserted = data
    rules = dict(rules)
    alias = copy.deepcopy(rules[op._LINE1])
    alias["versions"][0]["formula"] += " + 1"
    rules[op._LINE1] = alias
    assert not accepted((b, branch, witnesses, rules, asserted))


@pytest.mark.parametrize(
    "kind", ["no_asserted", "cloned_asserted", "other_span", "wrong_relation"]
)
def test_pair_or_clause_guards(data, kind):
    b, branch, witnesses, rules, asserted = data
    if kind == "no_asserted":
        asserted = {}
    elif kind == "cloned_asserted":
        asserted = {**asserted, op._LINE5: copy.deepcopy(asserted[op._LINE5])}
    elif kind == "other_span":
        branch = dataclasses.replace(branch, start=branch.start + 1)
    else:
        witnesses = {
            dataclasses.replace(
                w, relational_transitions=((op._LINE1, ">=", op._CREDIT),)
            )
            for w in witnesses
        }
    assert not accepted((b, branch, witnesses, rules, asserted))


@pytest.mark.parametrize("kind", ["selector", "transition"])
def test_actual_changed_source_operand_matches_witness(data, kind):
    b, branch, witnesses, rules, asserted = data
    if kind == "selector":
        witnesses = {dataclasses.replace(w, selector_name=op._PAID) for w in witnesses}
    else:
        witnesses = {
            dataclasses.replace(w, numeric_transition=(999.0, 1000.0))
            for w in witnesses
        }
    assert not accepted((b, branch, witnesses, rules, asserted))


def test_current_payload_partial_claimant(data):
    b, branch, witnesses, rules, asserted = data
    p = copy.deepcopy(b.payload)
    extra = copy.deepcopy(rules[op._LINE5])
    extra["name"] = "unrelated_credit"
    extra["versions"][0]["formula"] = "0"
    atoms = extra["metadata"]["proof"]["atoms"]
    extra["metadata"]["proof"]["atoms"] = [
        a for a in atoms if a["path"].endswith(("effective_from", "effective_to"))
    ] + [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {"corpus_citation_path": C, "excerpt": branch.text},
        }
    ]
    p["rules"].append(extra)
    bundle = tr.certify_worksheet_transfers(
        p,
        source_text=S,
        corpus_citation_path=C,
        source_context=CONTEXT,
        test_cases=b.cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )
    fresh = r.bind_worksheet_replacement(p, b.cases, S, CONTEXT, bundle)
    fresh_rules = op.paths._index(p["rules"])
    result = accepted((fresh, branch, witnesses, fresh_rules, asserted))
    assert not result


def test_unmodified_deepcopy_remaps_actual_case_ids(data):
    original = data
    b, branch, witnesses, rules, asserted = copy.deepcopy(data)
    old_to_new = {id(a): id(c) for a, c in zip(original[0].cases, b.cases)}
    witnesses = {
        dataclasses.replace(
            w, case_pair_identity=tuple(old_to_new[i] for i in w.case_pair_identity)
        )
        for w in witnesses
    }
    assert len(accepted((b, branch, witnesses, rules, asserted))) == 3


@pytest.mark.parametrize(
    "transition",
    [
        (),
        (3000,),
        (3000, 5000, 6000),
        ("3000", "5000"),
        (float("nan"), 5000),
        (3000, float("inf")),
        (True, 5000),
        1,
    ],
)
def test_malformed_transition_rejected(data, transition):
    b, branch, witnesses, rules, asserted = data
    witnesses = {
        dataclasses.replace(w, numeric_transition=transition) for w in witnesses
    }
    assert not accepted((b, branch, witnesses, rules, asserted))


@pytest.mark.parametrize("kind", ["calendar", "superseded_alias", "superseded_owner"])
def test_temporal_evidence_rejected(data, kind):
    b, branch, witnesses, rules, asserted = data
    if kind == "calendar":
        witnesses = {
            dataclasses.replace(w, calendar_attainment_age=18) for w in witnesses
        }
    else:
        rules = dict(rules)
        name = op._LINE1 if kind == "superseded_alias" else op._LINE5
        rule = copy.deepcopy(rules[name])
        version = copy.deepcopy(rule["versions"][0])
        version["effective_from"] = "2025-07-01"
        rule["versions"].append(version)
        rules[name] = rule
    assert not accepted((b, branch, witnesses, rules, asserted))


def test_real_exception_consumer_keeps_annual_allocation_separate(data):
    bound, branch, witnesses, rules, asserted = data
    kwargs = dict(
        source_text=S,
        corpus_citation_path=C,
        principal_rules=rules,
        principal_rule_paths={},
        asserted_by_rule=asserted,
        toggled_exception_selectors=witnesses,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )
    baseline_annual = {(17, 29)}
    enhanced_annual = set(baseline_annual)
    assert sc._unwitnessed_exception_branches(
        (branch,), represented_annual_spans=baseline_annual, **kwargs
    ) == (branch,)
    assert not sc._unwitnessed_exception_branches(
        (branch,),
        represented_annual_spans=enhanced_annual,
        certified_replacement=bound,
        **kwargs,
    )
    assert enhanced_annual == baseline_annual == {(17, 29)}
    # Three existing pair resources cannot satisfy four distinct obligations.
    duplicate_branches = tuple(
        dataclasses.replace(branch, path=(str(index),)) for index in range(4)
    )
    assert (
        len(
            sc._unwitnessed_exception_branches(
                duplicate_branches, certified_replacement=bound, **kwargs
            )
        )
        == 1
    )


def test_changed_private_rounding_on_bound_payload_is_stale(data):
    bound, branch, witnesses, rules, asserted = copy.deepcopy(data)
    old_to_new = {id(a): id(b) for a, b in zip(data[0].cases, bound.cases)}
    witnesses = {
        dataclasses.replace(
            w, case_pair_identity=tuple(old_to_new[i] for i in w.case_pair_identity)
        )
        for w in witnesses
    }
    assert len(accepted((bound, branch, witnesses, rules, asserted))) == 3
    index = next(
        i for i, rule in enumerate(bound.payload["rules"]) if rule["name"] == op._LINE5
    )
    original = bound.payload["rules"][index]
    changed = sc._CurrencyRoundingRule(original, sc._CurrencyRounding("floor", 0))
    assert dict(original) == dict(changed)
    bound.payload["rules"][index] = changed
    rules = op.paths._index(bound.payload["rules"])
    assert not bound.evidence.matches(
        bound.payload, bound.cases, bound.source, bound.context, bound.bundle
    )
    assert not accepted((bound, branch, witnesses, rules, asserted))
