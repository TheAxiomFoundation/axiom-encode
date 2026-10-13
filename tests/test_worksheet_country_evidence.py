"""Source-owned count and monetary conjunction; no income-polarity bypass."""

import copy
import hashlib
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import worksheet_country_evidence as c

SOURCE = (Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt").read_text()
PREFIX = "ca:policies/cra/t1-2025/provincial-territorial-foreign-tax-credit#"
HEADER = (
    "Use this form to calculate the foreign non-business income tax credit for 2025"
)


def atom(path, excerpt, kind="condition"):
    return {
        "path": path,
        "kind": kind,
        "source": {"corpus_citation_path": c.CITE, "excerpt": excerpt},
    }


def rule(name, formula, parameter=False):
    result = {
        "name": name,
        "kind": "parameter" if parameter else "derived",
        "dtype": "Money" if parameter else "Judgment",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": formula,
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    atom(
                        "versions[0].formula",
                        c.ANTECEDENT if parameter else c.CLAUSE,
                        "amount" if parameter else "condition",
                    ),
                    *[
                        atom("versions[0]." + field, HEADER, "definition")
                        for field in ("effective_from", "effective_to")
                    ],
                ]
            }
        },
    }
    if parameter:
        result["unit"] = "CAD"
    else:
        result.update(entity="Person", period="Year")
    return result


def fixture():
    payload = {
        "module": {
            "source_verification": {
                "corpus_citation_path": c.CITE,
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            }
        },
        "inputs": [
            {"name": c.COUNT, "dtype": "Integer", "entity": "Person", "period": "Year"},
            {
                "name": c.AMOUNT,
                "dtype": "Money",
                "unit": "CAD",
                "entity": "Person",
                "period": "Year",
            },
        ],
        "rules": [
            rule(c.OWNER, f"{c.COUNT} > 1 and {c.AMOUNT} > {c.LIMIT}"),
            rule(c.LIMIT, 200, True),
        ],
    }
    cases = [
        {
            "name": str(i),
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                PREFIX + "input." + c.COUNT: n,
                PREFIX + "input." + c.AMOUNT: amount,
            },
            "output": {PREFIX + c.OWNER: expected},
        }
        for i, (n, amount, expected) in enumerate(
            ((1, 201, "not_holds"), (2, 200, "not_holds"), (2, 201, "holds"))
        )
    ]
    return payload, cases


def certify(payload, cases, source=SOURCE):
    return c.certify_worksheet_country(
        payload, test_cases=cases, source_text=source, corpus_citation_path=c.CITE
    )


def owns(evidence, payload, cases, **changes):
    args = {
        "payload": payload,
        "test_cases": cases,
        "source_text": SOURCE,
        "corpus_citation_path": c.CITE,
        "owner": c.OWNER,
        "version": 0,
        "interval_start": "2025-01-01",
        "interval_end": "2025-12-31",
        "start": 887,
        "end": 1118,
    }
    return evidence.owns_condition(**(args | changes))


def test_actual_source_three_boundary_cases_and_exact_projection():
    payload, cases = fixture()
    evidence = certify(payload, cases)
    assert not evidence.unresolved
    assert len(evidence.pairs) == 2
    assert owns(evidence, payload, cases)
    assert owns(evidence, payload, cases, start=886)


@pytest.mark.parametrize(
    "mutation",
    [
        "count2",
        "inclusive",
        "or",
        "negated",
        "dead",
        "extra",
        "missing_owner_proof",
        "antecedent_only",
        "ontario_only_threshold",
        "threshold201",
        "threshold_bool",
        "threshold_nan",
        "parameter_entity",
        "parameter_period",
        "owner_unit",
        "owner_entity",
        "owner_period",
        "input_usd",
        "input_count_alias",
        "input_entity",
        "input_period",
        "missing_date",
        "partial_year",
        "superseded",
        "duplicate_input",
        "duplicate_rule",
        "import",
        "description_income",
        "description_no_tax",
        "description_payable",
        "description_jurisdiction",
        "renamed_income",
        "renamed_payable",
    ],
)
def test_candidate_mutations_fail_closed(mutation):
    payload, cases = fixture()
    owner, parameter = payload["rules"]
    count, amount = payload["inputs"]
    formula = owner["versions"][0]["formula"]
    if mutation in {"count2", "inclusive", "or", "negated", "dead", "extra"}:
        owner["versions"][0]["formula"] = {
            "count2": formula.replace("> 1", "> 2"),
            "inclusive": formula.replace("> 1", ">= 1"),
            "or": formula.replace(" and ", " or "),
            "negated": "not (" + formula + ")",
            "dead": "False and (" + formula + ")",
            "extra": "unknown and (" + formula + ")",
        }[mutation]
    elif mutation == "missing_owner_proof":
        owner["metadata"]["proof"]["atoms"] = owner["metadata"]["proof"]["atoms"][1:]
    elif mutation == "antecedent_only":
        owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = c.ANTECEDENT
    elif mutation == "ontario_only_threshold":
        parameter["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "If the total non-business income taxes you paid to all foreign countries is $200 or less"
        )
    elif mutation.startswith("threshold"):
        parameter["versions"][0]["formula"] = {
            "threshold201": 201,
            "threshold_bool": True,
            "threshold_nan": float("nan"),
        }[mutation]
    elif mutation == "parameter_entity":
        parameter["entity"] = "Person"
    elif mutation == "parameter_period":
        parameter["period"] = "Year"
    elif mutation == "owner_unit":
        owner["unit"] = "CAD"
    elif mutation == "owner_entity":
        owner["entity"] = "Household"
    elif mutation == "owner_period":
        owner["period"] = "Month"
    elif mutation == "input_usd":
        amount["unit"] = "USD"
    elif mutation == "input_count_alias":
        count["dtype"] = "Count"
    elif mutation == "input_entity":
        count["entity"] = "Household"
    elif mutation == "input_period":
        count["period"] = "Month"
    elif mutation == "missing_date":
        owner["metadata"]["proof"]["atoms"].pop()
    elif mutation == "partial_year":
        owner["versions"][0]["effective_to"] = "2025-06-30"
    elif mutation == "superseded":
        owner["versions"].append(
            {
                "effective_from": "2025-07-01",
                "effective_to": "2025-12-31",
                "formula": False,
            }
        )
    elif mutation == "duplicate_input":
        payload["inputs"].append(copy.deepcopy(count))
    elif mutation == "duplicate_rule":
        payload["rules"].append(copy.deepcopy(owner))
    elif mutation == "import":
        payload["imports"] = ["foreign#" + c.COUNT]
    elif mutation == "description_jurisdiction":
        count["description"] = "Canadian jurisdictions in which tax was paid"
    elif mutation.startswith("description"):
        amount["description"] = {
            "description_income": "has income",
            "description_no_tax": "no income tax paid",
            "description_payable": "income tax payable",
        }[mutation]
    elif mutation.startswith("renamed"):
        amount["name"] = {
            "renamed_income": "has_income",
            "renamed_payable": "income_tax_payable",
        }[mutation]
    assert certify(payload, cases).unresolved


@pytest.mark.parametrize(
    "mutation", ["quoted", "negative", "has_income", "before", "after", "duplicate"]
)
def test_source_mutations_with_fresh_source_hash_reject(mutation):
    payload, cases = fixture()
    modified = {
        "quoted": '"' + c.CLAUSE + '"',
        "negative": c.CLAUSE.replace(
            "income taxes that you paid", "income taxes that you did not pay"
        ),
        "has_income": c.CLAUSE.replace(
            "income taxes that you paid", "income that you had"
        ),
        "before": "Only residents qualify.\n" + c.CLAUSE,
        "after": c.CLAUSE + " Only residents qualify.",
        "duplicate": c.CLAUSE + "\n" + c.CLAUSE,
    }[mutation]
    source = SOURCE.replace(c.CLAUSE, modified)
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    assert certify(payload, cases, source).unresolved


@pytest.mark.parametrize(
    "bad",
    [
        None,
        0,
        [],
        {"input": [], "output": {}},
        {"input": {}, "output": []},
        {"input": {1: 0}, "output": {}},
    ],
)
def test_malformed_case_rejects_without_crash(bad):
    payload, cases = fixture()
    assert certify(payload, [bad]).unresolved
    evidence = certify(payload, cases)
    assert not owns(evidence, payload, [bad])


@pytest.mark.parametrize("field", ["input", "output"])
def test_foreign_prefix_and_duplicate_suffix_reject(field):
    payload, cases = fixture()
    original = next(iter(cases[0][field]))
    cases[0][field]["foreign#" + original.split("#")[1]] = cases[0][field][original]
    assert certify(payload, cases).unresolved


@pytest.mark.parametrize(
    "change",
    [
        {"owner": "other"},
        {"version": 1},
        {"interval_start": "2025-02-01"},
        {"interval_end": "2026-12-31"},
        {"start": 885},
        {"start": 888},
        {"end": 1117},
        {"end": 1119},
        {"source_text": SOURCE + " "},
        {"corpus_citation_path": "foreign"},
    ],
)
def test_exact_consumer_binding(change):
    payload, cases = fixture()
    evidence = certify(payload, cases)
    assert not owns(evidence, payload, cases, **change)


def test_stale_payload_cases_and_64_bound():
    payload, cases = fixture()
    evidence = certify(payload, cases)
    assert not owns(evidence, payload, cases[:-1])
    stale = copy.deepcopy(payload)
    stale["module"]["summary"] = "different"
    assert not owns(evidence, stale, cases)
    many = cases + [copy.deepcopy(cases[0]) for _ in range(61)]
    assert len(many) == 64 and not certify(payload, many).unresolved
    assert certify(payload, many + [copy.deepcopy(cases[0])]).unresolved
    assert not owns(evidence, payload, many + [copy.deepcopy(cases[0])])


def test_income_polarity_mechanisms_remain_distinct_and_unchanged():
    assert sc._source_gate_predicate_polarities("has income") == {"income": False}
    assert sc._source_gate_predicate_polarities("has no income") == {"income": True}
    payload, cases = fixture()
    payload["inputs"][1]["description"] = "has no income"
    assert certify(payload, cases).unresolved


def test_case_assertion_period_and_same_world_controls():
    payload, cases = fixture()
    cases[0]["output"][PREFIX + c.OWNER] = "holds"
    assert certify(payload, cases).unresolved
    payload, cases = fixture()
    cases[0]["period"]["end"] = "2026-12-31"
    assert certify(payload, cases).unresolved
    payload, cases = fixture()
    cases[0]["output"][PREFIX + "unrelated"] = 0
    assert certify(payload, cases).unresolved


@pytest.mark.parametrize("source", ["", "short", SOURCE[:887], SOURCE[:1117]])
def test_shortened_source_accessor_fails_closed(source):
    payload, cases = fixture()
    evidence = certify(payload, cases)
    assert not owns(evidence, payload, cases, source_text=source)


def test_boolean_version_accessor_fails_closed():
    payload, cases = fixture()
    assert not owns(certify(payload, cases), payload, cases, version=False)


@pytest.mark.parametrize(
    "field,value", [("start", []), ("start", 887.0), ("end", {}), ("end", 1118.0)]
)
def test_malformed_span_accessor_fails_closed(field, value):
    payload, cases = fixture()
    assert not owns(certify(payload, cases), payload, cases, **{field: value})


@pytest.mark.parametrize("anchor", ["This form does not apply", c.PREDECESSOR])
def test_new_preceding_restriction_remains_unresolved(anchor):
    payload, cases = fixture()
    source = SOURCE.replace(anchor, "Only residents qualify. " + anchor, 1)
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    assert certify(payload, cases, source).unresolved
