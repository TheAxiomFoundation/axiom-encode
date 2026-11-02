import copy
import dataclasses
import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_replacement_evidence as replacement
from axiom_encode.harness import worksheet_transfer_evidence as tr
from tests.test_worksheet_transfer_evidence import CONTEXT
from tests.test_worksheet_transfer_evidence import SOURCE as S

FIXTURES = Path(__file__).parent / "fixtures" / "worksheet_exceptions"
PAYLOAD = yaml.safe_load((FIXTURES / "policy.yaml").read_text())
CASES = yaml.safe_load((FIXTURES / "cases.yaml").read_text())

from axiom_encode.harness import (
    source_completeness as sc,
)
from axiom_encode.harness import (
    worksheet_operation_evidence as op,
)
from axiom_encode.harness.worksheet_composition_evidence import SPAN, certify


def build(payload=PAYLOAD, cases=CASES, source=S, context=CONTEXT):
    p = sc._bind_currency_rounding_rules(payload)
    cs = sc._typed_date_cases(
        sc._typed_numeric_expected_cases(cases, {r["name"]: r for r in p["rules"]}), p
    )
    bundle = tr.certify_worksheet_transfers(
        p,
        source_text=source,
        corpus_citation_path=op._PRIMARY,
        source_context=context,
        test_cases=cs,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )
    return replacement.bind_worksheet_replacement(p, cs, source, context, bundle)


@pytest.fixture(scope="module")
def data():
    b = build(copy.deepcopy(PAYLOAD), copy.deepcopy(CASES))
    c = certify(b)
    assert not c.unresolved
    branches = sc.recognize_source_structure(S)
    branch = next(
        x
        for x in sc._source_exception_branches(
            S, branches=branches, active_branches=branches, deferred_paths=set()
        )
        if (x.start, x.end) == SPAN
    )
    return b, c, branch


def test_positive_ordinary_and_amt_executions(data):
    b, c, branch = data
    assert c.matches(b, branch, op.paths._index(b.payload["rules"]))
    assert c.ordinary_paths == (0,)
    assert len(c.execution_records) == 127
    assert {x[0] for x in c.execution_records} == set(op._OUTPUTS)
    duplicate = dataclasses.replace(branch, path=("duplicate",))
    resource = (1, c.resource)
    assert (
        len(
            sc._unmatched_evidence_obligations(
                {branch: {resource}, duplicate: {resource}}
            )
        )
        == 1
    )


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "context",
        "case",
        "projection",
        "core",
        "owner",
        "span",
        "boolspan",
        "missingowner",
    ],
)
def test_stale_accessor(data, kind):
    b, c, branch = copy.deepcopy(data)
    rules = op.paths._index(b.payload["rules"])
    if kind == "source":
        b = dataclasses.replace(b, source=b.source + "\n")
    elif kind == "context":
        b.context[op._SECONDARY] += "\n"
    elif kind == "case":
        b.cases[0]["output"] = {}
    elif kind == "projection":
        b = dataclasses.replace(
            b,
            bundle=dataclasses.replace(
                b.bundle, flat_case_indices=tuple(reversed(b.bundle.flat_case_indices))
            ),
        )
    elif kind == "core":
        b = dataclasses.replace(
            b,
            bundle=dataclasses.replace(
                b.bundle, scalar_operations=op.WorksheetOperationCoverage()
            ),
        )
    elif kind == "owner":
        rules = dict(rules)
        rules[op._LINE2] = copy.deepcopy(rules[op._LINE2])
        rules[op._LINE2]["versions"][0]["formula"] = "0"
    elif kind == "span":
        branch = dataclasses.replace(branch, end=branch.end - 1)
    elif kind == "boolspan":
        branch = dataclasses.replace(branch, start=True)
    else:
        rules = {k: v for k, v in rules.items() if k != op._LINE2}
    assert not c.matches(b, branch, rules)


def test_unmodified_copy(data):
    b, c, branch = copy.deepcopy(data)
    assert c.matches(b, branch, op.paths._index(b.payload["rules"]))


@pytest.mark.parametrize(
    "kind",
    [
        "ordinary_proof",
        "partial_claimant",
        "ordinary_formula",
        "ratio",
        "date",
        "wrongtype",
        "dead_amt",
        "missingordinary",
    ],
)
def test_actual_mutated_candidate(data, kind):
    p = copy.deepcopy(PAYLOAD)
    cases = copy.deepcopy(CASES)
    rules = {r["name"]: r for r in p["rules"]}
    if kind == "ordinary_proof":
        atoms = rules[op._LINE2]["metadata"]["proof"]["atoms"]
        rules[op._LINE2]["metadata"]["proof"]["atoms"] = [
            a
            for a in atoms
            if S[1223:1305] not in a.get("source", {}).get("excerpt", "")
        ]
    elif kind == "partial_claimant":
        extra = copy.deepcopy(rules[op._LINE5])
        extra["name"] = "unrelated_credit"
        extra["versions"][0]["formula"] = "0"
        extra["metadata"]["proof"]["atoms"] = [
            a
            for a in extra["metadata"]["proof"]["atoms"]
            if a["path"].endswith(("effective_from", "effective_to"))
        ] + [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": op._PRIMARY,
                    "excerpt": S[1310:1333],
                },
            }
        ]
        p["rules"].append(extra)
    elif kind == "ordinary_formula":
        rules[op._LINE2]["versions"][0]["formula"] = rules[op._LINE2]["versions"][0][
            "formula"
        ].replace(op._CREDIT, op._CREDIT + " + 1", 1)
    elif kind == "ratio":
        rules[op._LINE4]["versions"][0]["formula"] = "1"
    elif kind == "date":
        rules[op._LINE2]["versions"][0]["effective_to"] = "2025-06-30"
    elif kind == "wrongtype":
        rules[op._LINE2]["unit"] = "USD"
    elif kind == "dead_amt":
        rules[op._LINE2]["versions"][0]["formula"] = op._CREDIT
    else:
        for case in cases:
            env = sc._case_input_formula_environment(case) or {}
            if env.get("minimum_tax_is_payable") is False:
                output = case.get("output", {})
                for key in list(output):
                    if key.rsplit("#", 1)[-1] == op._LINE2:
                        output.pop(key)
    b = build(p, cases)
    c = certify(b)
    assert c.unresolved and not c.resource


@pytest.mark.parametrize("kind", ["prefix", "suffix", "quote", "row_tail"])
def test_changed_actual_source_with_updated_hash(data, kind):
    import hashlib

    p = copy.deepcopy(PAYLOAD)
    source = (
        "Restriction: residents only.\n" + S
        if kind == "prefix"
        else S + "\nOnly disabled claimants may claim."
        if kind == "suffix"
        else S[:1119] + '"' + S[1119:1492] + '"' + S[1492:]
        if kind == "quote"
        else S[:1305] + " only for residents" + S[1305:]
    )
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    context = {**CONTEXT, op._PRIMARY: source}
    b = build(p, copy.deepcopy(CASES), source, context)
    assert certify(b).unresolved


@pytest.mark.parametrize("kind", ["spans", "paths", "dependency"])
def test_mutated_actual_amt_record(data, kind):
    b, c, branch = copy.deepcopy(data)
    records = list(b.bundle.scalar_paths.obligations)
    i = next(i for i, r in enumerate(records) if r.identity == "bc_amt")
    record = records[i]
    if kind == "spans":
        records[i] = dataclasses.replace(record, source_spans=())
    elif kind == "paths":
        records[i] = dataclasses.replace(record, compatible_paths=(0,))
    else:
        deps = list(record.selected_dependencies)
        deps[0] = dataclasses.replace(deps[0], formula="0")
        records[i] = dataclasses.replace(record, selected_dependencies=tuple(deps))
    b = dataclasses.replace(
        b,
        bundle=dataclasses.replace(
            b.bundle,
            scalar_paths=dataclasses.replace(
                b.bundle.scalar_paths, obligations=tuple(records)
            ),
        ),
    )
    assert certify(b).unresolved


def test_private_rounding_identity(data):
    b, c, branch = copy.deepcopy(data)
    rules = op.paths._index(b.payload["rules"])
    rule = rules[op._LINE5]

    class RuleWithPrivateRounding(dict):
        pass

    changed = RuleWithPrivateRounding(rule)
    changed.resolved_currency_rounding = ("mutated",)
    index = next(i for i, r in enumerate(b.payload["rules"]) if r["name"] == op._LINE5)
    b.payload["rules"][index] = changed
    rules = op.paths._index(b.payload["rules"])
    assert not c.matches(b, branch, rules)


def test_closed_resource_enters_real_matcher_without_annual_credit(data):
    from axiom_encode.harness.worksheet_composition_evidence import (
        bind_worksheet_composition,
    )

    bound, certificate, branch = data
    rules = op.paths._index(bound.payload["rules"])
    kwargs = dict(
        source_text=S,
        corpus_citation_path=op._PRIMARY,
        principal_rules=rules,
        principal_rule_paths={},
        asserted_by_rule={},
        toggled_exception_selectors=set(),
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
    )
    annual = {(17, 29)}
    assert sc._unwitnessed_exception_branches((branch,), **kwargs) == (branch,)
    closed = bind_worksheet_composition(bound)
    assert closed.evidence == certificate
    assert not sc._unwitnessed_exception_branches(
        (branch,),
        certified_composition=closed,
        represented_annual_spans=annual,
        **kwargs,
    )
    duplicate = dataclasses.replace(branch, path=("duplicate",))
    assert (
        len(
            sc._unwitnessed_exception_branches(
                (branch, duplicate),
                certified_composition=closed,
                represented_annual_spans=annual,
                **kwargs,
            )
        )
        == 1
    )
    assert annual == {(17, 29)}
