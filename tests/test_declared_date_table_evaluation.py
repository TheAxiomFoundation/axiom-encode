import ast
from copy import deepcopy
from datetime import date

import pytest

from axiom_encode.harness import source_completeness as s


def payload():
    citation = "ca/policy/example"
    atoms = [
        {
            "path": "versions[0]." + field,
            "kind": "definition",
            "source": {
                "corpus_citation_path": citation,
                "excerpt": "Applicable in 2025; acquired before 2023.",
            },
        }
        for field in ("values", "effective_from", "effective_to")
    ]
    return {
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "inputs": [],
        "rules": [
            {
                "name": "year",
                "kind": "parameter",
                "dtype": "Integer",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "2025",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {**a, "path": a["path"].replace(".values", ".formula")}
                            for a in atoms
                        ]
                    }
                },
            },
            {
                "name": "cutoff",
                "kind": "parameter",
                "dtype": "Date",
                "indexed_by": "year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "values": {2025: "2023-01-01"},
                    }
                ],
                "metadata": {"proof": {"atoms": atoms}},
            },
        ],
    }


def environment(p=None, case=None):
    return s._formula_environment_for_case(
        s._constant_rule_environment(p or payload()),
        {"period": "2025-01-01"} if case is None else case,
    )


def evaluate(text, env):
    actual = s._evaluate_condition_expression(ast.parse(text, mode="eval").body, env)
    ordinary = s._evaluate_rulespec_formula(text, environment=env)
    assert ordinary == actual
    return actual


@pytest.mark.parametrize(
    ("value", "expected"),
    [(date(2022, 12, 31), True), (date(2023, 1, 1), False), (date(2023, 1, 2), False)],
)
def test_typed_date_boundary(value, expected):
    env = environment()
    env["acquired"] = value
    assert evaluate("acquired < cutoff[year]", env) is expected
    table = env["cutoff"]
    assert {a[0] for a in table.source_atoms} == {
        "versions[0].values",
        "versions[0].effective_from",
        "versions[0].effective_to",
    }


@pytest.mark.parametrize(
    "case",
    [{}, {"period": "2024-12-31"}, {"period": "2026-01-01"}, {"period": "invalid"}],
)
def test_no_missing_or_outside_period_fallback(case):
    assert (
        evaluate("cutoff[year]", environment(case=case))
        is s._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_dtype",
        "wrong_index_dtype",
        "input_collision",
        "duplicate_rule",
        "missing_proof",
        "formula_proof",
        "wrong_citation",
        "invalid_date",
        "bool_key",
        "str_key",
        "mixed_formula",
        "default",
        "rounding",
        "entity",
        "unit",
        "overlap",
        "bad_period",
    ],
)
def test_malformed_or_ambiguous_table_refuses(mutation):
    p = payload()
    r = p["rules"][1]
    v = r["versions"][0]
    if mutation == "wrong_dtype":
        r["dtype"] = "String"
    elif mutation == "wrong_index_dtype":
        p["rules"][0]["dtype"] = "Boolean"
    elif mutation == "input_collision":
        p["inputs"].append({"name": "year", "dtype": "Integer"})
    elif mutation == "duplicate_rule":
        p["rules"].append(deepcopy(r))
    elif mutation == "missing_proof":
        r["metadata"]["proof"]["atoms"].pop(0)
    elif mutation == "formula_proof":
        r["metadata"]["proof"]["atoms"][0]["path"] = "versions[0].formula"
    elif mutation == "wrong_citation":
        r["metadata"]["proof"]["atoms"][0]["source"]["corpus_citation_path"] = (
            "elsewhere"
        )
    elif mutation == "invalid_date":
        v["values"] = {2025: "2023-02-30"}
    elif mutation == "bool_key":
        v["values"] = {True: "2023-01-01"}
    elif mutation == "str_key":
        v["values"] = {"2025": "2023-01-01"}
    elif mutation == "mixed_formula":
        v["formula"] = "0"
    elif mutation in ("default", "rounding", "entity", "unit"):
        r[mutation] = "unexpected"
    elif mutation == "overlap":
        r["versions"].append(deepcopy(v))
        r["metadata"]["proof"]["atoms"] += [
            {**a, "path": a["path"].replace("[0]", "[1]")}
            for a in r["metadata"]["proof"]["atoms"]
        ]
    elif mutation == "bad_period":
        v["effective_from"] = "2026-01-01"
    assert evaluate("cutoff[year]", environment(p)) is s._UNRESOLVED_CONDITION_VALUE


@pytest.mark.parametrize(
    "expr",
    ["cutoff[2025]", "cutoff[other]", "cutoff[year + 0]", "cutoff[:]"],
)
def test_no_arbitrary_indexing(expr):
    env = environment()
    env.update(other=2025, raw={2025: date(2023, 1, 1)})
    assert evaluate(expr, env) is s._UNRESOLVED_CONDITION_VALUE


@pytest.mark.parametrize("key", [True, 2025.0, "2025", 2024])
def test_index_must_be_exact_integer_present(key):
    env = environment()
    env["year"] = key
    assert evaluate("cutoff[year]", env) is s._UNRESOLVED_CONDITION_VALUE


def test_multiple_disjoint_versions_select_without_fallback():
    p = payload()
    table = p["rules"][1]
    old = deepcopy(table["versions"][0])
    old.update(
        effective_from="2024-01-01",
        effective_to="2024-12-31",
        values={2025: "2022-01-01"},
    )
    table["versions"].append(old)
    table["metadata"]["proof"]["atoms"] += [
        {**a, "path": a["path"].replace("[0]", "[1]")}
        for a in table["metadata"]["proof"]["atoms"]
    ]
    constants = s._constant_rule_environment(p)
    prior = s._formula_environment_for_case(constants, {"period": "2024-06-01"})
    assert "cutoff" not in prior
    assert (
        evaluate("cutoff[year]", prior) is s._UNRESOLVED_CONDITION_VALUE
    )  # index not applicable
    assert evaluate("cutoff[year]", environment(p)) == date(2023, 1, 1)


def test_existing_scalar_missing_period_behavior_unchanged():
    assert environment(case={})["year"] == 2025
    assert "cutoff" not in environment(case={})


@pytest.mark.parametrize(
    "mutation", ["overlap", "default", "period", "missing_lineage", "wrong_lineage"]
)
def test_index_contract_refuses(mutation):
    p = payload()
    index = p["rules"][0]
    if mutation == "overlap":
        v = deepcopy(index["versions"][0])
        v["effective_from"] = "2025-06-01"
        index["versions"].append(v)
        index["metadata"]["proof"]["atoms"] += [
            {**a, "path": a["path"].replace("[0]", "[1]")}
            for a in index["metadata"]["proof"]["atoms"]
        ]
    elif mutation == "default":
        index["versions"][0]["default"] = 2025
    elif mutation == "period":
        index["period"] = "Month"
    elif mutation == "missing_lineage":
        index.pop("metadata")
    else:
        index["metadata"]["proof"]["atoms"][0]["source"]["corpus_citation_path"] = (
            "wrong"
        )
    assert evaluate("cutoff[year]", environment(p)) is s._UNRESOLVED_CONDITION_VALUE


def test_selected_index_lineage_retained():
    table = environment()["cutoff"]
    assert {a[0] for a in table.selected_index_atoms} == {
        "versions[0].formula",
        "versions[0].effective_from",
        "versions[0].effective_to",
    }


@pytest.mark.parametrize(
    "period",
    [
        {"start": "2025-01-01"},
        {"period_kind": "nonsense", "start": "2025-01-01", "end": "2025-12-31"},
        {"period_kind": "tax_year", "start": "2025-06-01", "end": "2025-01-01"},
        2025,
    ],
)
def test_date_table_refuses_invalid_complete_period(period):
    assert (
        evaluate("cutoff[year]", environment(case={"period": period}))
        is s._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize(
    "period",
    [
        "2025",
        "2025-01",
        "2025-01-01",
        date(2025, 1, 1),
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    ],
)
def test_supported_complete_period_representations(period):
    assert evaluate("cutoff[year]", environment(case={"period": period})) == date(
        2023, 1, 1
    )
