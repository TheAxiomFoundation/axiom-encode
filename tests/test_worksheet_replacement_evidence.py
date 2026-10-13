"""Portable tests against real producer APIs; no saved certificates/instrumentation."""

import copy
import dataclasses
import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness import worksheet_replacement_evidence as r
from tests.test_worksheet_transfer_evidence import CONTEXT, SOURCE, certify, inputs


@pytest.fixture(scope="module")
def scope():
    payload, cases = inputs()
    # Test-only equivalent constant fold: no new facts/expected values.
    rules = {x["name"]: x for x in payload["rules"]}
    threshold = rules["multiple_foreign_country_count_threshold"]
    assert threshold["versions"][0]["formula"] == 1
    helper = rules[r.OWNER]["versions"][0]
    assert (
        "jurisdictions_tax_paid_count > multiple_foreign_country_count_threshold"
        in helper["formula"]
    )
    helper["formula"] = helper["formula"].replace(
        "jurisdictions_tax_paid_count > multiple_foreign_country_count_threshold",
        "jurisdictions_tax_paid_count > 1",
    )
    payload = sc._bind_currency_rounding_rules(payload)
    cases = sc._typed_date_cases(
        sc._typed_numeric_expected_cases(
            cases, {x["name"]: x for x in payload["rules"]}
        ),
        payload,
    )
    bundle = certify(payload, cases)
    assert not bundle.unresolved
    bound = r.bind_worksheet_replacement(payload, cases, SOURCE, CONTEXT, bundle)
    assert not bound.evidence.unresolved
    return bound


def args(bound):
    branches = sc.recognize_source_structure(SOURCE)
    branch = next(
        b
        for b in sc._source_formula_branches(
            SOURCE, branches=branches, active_branches=branches, deferred_paths=set()
        )
        if (b.start, b.end) == r.SPAN
    )
    rules = op.paths._index(bound.payload["rules"])
    constants = sc._constant_rule_environment(bound.payload)
    case = bound.cases[bound.evidence.records[0].master_index]
    rule = rules[r.OWNER]
    env = sc._case_asserted_dependency_environment(
        rules, case, formula_environment=constants
    )
    execution = sc._case_formula_execution(
        rule, case, formula_environment=constants, dependency_environment=env
    )
    return dict(rule=rule, case=case, branch=branch, execution=execution)


def edges(bound):
    a = args(bound)
    rules = op.paths._index(bound.payload["rules"])
    kw = dict(
        source_text=SOURCE,
        corpus_citation_path=op._PRIMARY,
        principal_rules=rules,
        asserted_by_rule={
            r.OWNER: [
                c
                for c in bound.cases
                if sc._test_case_asserted_output_value(c, r.OWNER)
                is not sc._UNRESOLVED_CONDITION_VALUE
            ]
        },
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        formula_environment=sc._constant_rule_environment(bound.payload),
        dependency_cache={},
        execution_cache={},
        certified_replacement=bound,
    )
    return (
        a["branch"],
        kw,
        sc._formula_branch_test_witnesses(a["branch"], rule_names={r.OWNER}, **kw),
    )


def test_actual_producer_and_consumer(scope):
    assert scope.operation_kinds(**args(scope)) == frozenset({"subtract"})
    branch, kw, witnesses = edges(scope)
    assert witnesses
    assert all(
        (sc._case_input_formula_environment(scope.cases[x.master_index]) or {})[
            "jurisdictions_tax_paid_count"
        ]
        > 1
        for x in scope.evidence.records
    )
    # Existing branching semantic leaf keys are retained; no certificate IDs.
    assert all(owner == r.OWNER and key.startswith("leaf:") for owner, key in witnesses)
    duplicates = [
        dataclasses.replace(branch, path=branch.path + (str(i),))
        for i in range(len(witnesses) + 1)
    ]
    missing = sc._unwitnessed_formula_branches(
        duplicates,
        principal_formula_clause_rules={b: {r.OWNER} for b in duplicates},
        **kw,
    )
    assert len(missing) == 1


@pytest.mark.parametrize(
    "field",
    [
        "payload",
        "cases",
        "context",
        "source",
        "projection",
        "core_input",
        "core_context",
    ],
)
def test_bound_reference_mutations(scope, field):
    b = copy.deepcopy(scope)
    if field == "payload":
        b.payload["rules"][0]["source"] = "changed"
    elif field == "cases":
        b.cases[0]["name"] = "changed"
    elif field == "context":
        b.context[op._SECONDARY] += "\nchanged"
    elif field == "source":
        b = dataclasses.replace(b, source=SOURCE[:-1])
    elif field == "projection":
        b = dataclasses.replace(
            b,
            bundle=dataclasses.replace(
                b.bundle, flat_case_indices=b.bundle.flat_case_indices[::-1]
            ),
        )
    else:
        attr = "input_identity" if field == "core_input" else "context_identity"
        b = dataclasses.replace(
            b,
            bundle=dataclasses.replace(
                b.bundle,
                scalar_operations=dataclasses.replace(
                    b.bundle.scalar_operations, **{attr: "stale"}
                ),
            ),
        )
    assert b.operation_kinds(**args(b)) is None


@pytest.mark.parametrize("field", ["rule", "case", "branch", "execution"])
def test_malformed_supplied_objects(scope, field):
    a = args(scope)
    a[field] = []
    assert scope.operation_kinds(**a) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_rule_type",
        "wrong_version",
        "wrong_case",
        "expanded_leaf",
        "different_span",
    ],
)
def test_supplied_identity_and_expanded_retry(scope, mutation):
    a = args(scope)
    if mutation == "wrong_rule_type":
        a["rule"] = copy.deepcopy(a["rule"])
        a["rule"]["unit"] = "USD"
    elif mutation == "wrong_version":
        a["rule"] = copy.deepcopy(a["rule"])
        a["rule"]["versions"][0]["effective_to"] = "2025-06-30"
    elif mutation == "wrong_case":
        a["case"] = copy.deepcopy(a["case"])
        a["case"]["output"] = {}
    elif mutation == "expanded_leaf":
        a["execution"] = dataclasses.replace(
            a["execution"], leaf=a["execution"].leaf + " + 1"
        )
    else:
        a["branch"] = dataclasses.replace(a["branch"], start=a["branch"].start + 1)
    assert scope.operation_kinds(**a) is None


@pytest.mark.parametrize("mutation", ["value", "trace", "rounding"])
def test_supplied_execution_identity_probe(scope, mutation):
    a = args(scope)
    e = a["execution"]
    if mutation == "value":
        e = dataclasses.replace(e, evaluated_value=("numeric", "999999"))
    elif mutation == "trace":
        e = dataclasses.replace(e, trace=())
    else:
        e = dataclasses.replace(
            e, unrounded_value=__import__("decimal").Decimal("999999")
        )
    assert e != a["execution"]
    a["execution"] = e
    assert scope.operation_kinds(**a) is None


def test_identical_nonmaster_clone_rejected(scope):
    a = args(scope)
    original = a["case"]
    a["case"] = copy.deepcopy(original)
    assert a["case"] == original and a["case"] is not original
    assert scope.operation_kinds(**a) is None


@pytest.mark.parametrize("missing_secondary", [False, True])
def test_actual_complete_source_consumer(scope, missing_secondary):
    content = yaml.safe_dump(
        {**scope.payload, "rules": [dict(rule) for rule in scope.payload["rules"]]}
    )
    bindings = sc.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    result = sc.analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=op._PRIMARY,
        source_context=({op._PRIMARY: SOURCE} if missing_secondary else CONTEXT),
        test_cases=scope.cases,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_values=tuple(x[1] for x in bindings),
        artifact_numeric_bindings=bindings,
    )
    if missing_secondary:
        failed = r.bind_worksheet_replacement(
            scope.payload, scope.cases, SOURCE, {op._PRIMARY: SOURCE}, scope.bundle
        )
        assert failed.evidence.unresolved
        assert failed.operation_kinds(**args(scope)) is None
        assert any("2025 has no named scalar" in issue for issue in result.issues)
        assert any(
            "The amount on line 5 should not be more" in issue
            for issue in result.issues
        )
    else:
        assert not any("4334:4582" in issue for issue in result.issues)


def test_unmodified_deepcopy_has_matching_master(scope):
    b = copy.deepcopy(scope)
    assert b.operation_kinds(**args(b)) == frozenset({"subtract"})
