"""Static source-site controls; these never confer runtime provenance."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as source
from axiom_encode.harness.actual_local_relation_evidence import _wire_value
from axiom_encode.harness.local_fraction_site_evidence import _PROFILES, fraction_site

DATA = json.loads(
    (Path(__file__).parent / "fixtures/t936_fraction_sites.json").read_text()
)
BODY = (Path(__file__).parent / "fixtures/multipage_t936_source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/cumulative-net-investment-loss"
NAMES = (
    "federal_2025_t936_total_investment_expenses_line_a",
    "federal_2025_t936_total_investment_income_line_b",
)


def fixture(index):
    d = deepcopy(DATA)
    rule = next(r for r in d["policy"]["rules"] if r["name"] == NAMES[index])
    row = _PROFILES[index][0]
    start = BODY.index(row)
    branch = source.SourceStructureBranch(
        (), "formula-clause", "fraction", row, start, start + len(row)
    )
    facts = {n: _wire_value(v) for n, v in d["facts"].items()}
    return d, rule, branch, facts


@pytest.mark.parametrize("index", [0, 1])
def test_exact_distinct_site_and_no_source_ordinal_identity(index):
    d, rule, branch, facts = fixture(index)
    args = (
        d["policy"],
        rule,
        d["case"],
        branch,
        BODY,
        CITATION,
        d["traces"][NAMES[index]],
        facts,
    )
    site = fraction_site(*args)
    assert site is not None and site[0] == 0
    assert len(site[2]) == 64
    changed_label = source.SourceStructureBranch(
        (), "formula-clause", "different-label", branch.text, branch.start, branch.end
    )
    assert (
        fraction_site(
            d["policy"],
            rule,
            d["case"],
            changed_label,
            BODY,
            CITATION,
            d["traces"][NAMES[index]],
            facts,
        )
        == site
    )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "change",
    [
        "wrong_line",
        "generic_description",
        "after_half",
        "duplicate_contract",
        "wrong_unit",
        "default",
        "missing_row_proof",
        "wrong_rate",
        "wrong_read",
        "wrong_read_version",
        "wrong_period",
        "wrong_header",
        "duplicate_term",
        "swapped_operand",
        "wrong_position",
        "wrong_operator",
        "source_drift",
        "missing_fact",
    ],
)
def test_conflicting_source_or_operation_declines(index, change):
    d, rule, branch, facts = fixture(index)
    amount = next(i for i in d["policy"]["inputs"] if i["name"] == _PROFILES[index][2])
    rate = next(r for r in d["policy"]["rules"] if r["kind"] == "parameter")
    trace = d["traces"][NAMES[index]]
    body = BODY
    if change == "wrong_line":
        amount["name"] += "_wrong"
    elif change == "generic_description":
        amount["description"] = "Full amount."
    elif change == "after_half":
        amount["description"] = amount["description"].replace("before", "after")
    elif change == "duplicate_contract":
        d["policy"]["inputs"].append(dict(amount, name="unrelated"))
    elif change == "wrong_unit":
        amount["unit"] = "USD"
    elif change == "default":
        amount["default"] = 0
    elif change == "missing_row_proof":
        rate["metadata"]["proof"]["atoms"] = [
            a
            for a in rate["metadata"]["proof"]["atoms"]
            if a["source"]["excerpt"] != branch.text
        ]
    elif change == "wrong_rate":
        rate["versions"][0]["formula"] = "0.6"
    elif change == "wrong_read":
        trace["parameter_reads"][0]["value"]["value"] = "0.6"
    elif change == "wrong_read_version":
        trace["parameter_reads"][0]["index"] = 1
    elif change == "wrong_period":
        d["case"]["period"]["start"] = "2026-01-01"
    elif change == "wrong_header":
        rule["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = "2025"
    elif change == "duplicate_term":
        rule["versions"][0]["formula"] += " + " + amount["name"]
    elif change == "swapped_operand":
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            amount["name"], _PROFILES[1 - index][2]
        )
    elif change == "wrong_position":
        terms = rule["versions"][0]["formula"].split(" + ")
        terms[3], terms[4] = terms[4], terms[3]
        rule["versions"][0]["formula"] = " + ".join(terms)
    elif change == "wrong_operator":
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            " * ", " + "
        )
    elif change == "source_drift":
        body = body.replace("50% of exploration", "60% of exploration")
    elif change == "missing_fact":
        facts.pop(amount["name"])
    assert (
        fraction_site(
            d["policy"], rule, d["case"], branch, body, CITATION, trace, facts
        )
        is None
    )


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "change",
    [
        "whole_body",
        "shifted",
        "truncated",
        "crossed",
        "missing_total",
        "overlap_rate",
        "overlap_principal",
    ],
)
def test_exact_group_and_selected_version_boundaries(index, change):
    d, rule, branch, facts = fixture(index)
    atoms = rule["metadata"]["proof"]["atoms"]
    if change == "whole_body":
        atoms[0]["source"]["excerpt"] = BODY
    elif change == "shifted":
        a, b = (1193, 1647) if index == 0 else (2014, 2562)
        atoms[0]["source"]["excerpt"] = BODY[a + 1 : b + 1]
    elif change == "truncated":
        atoms[0]["source"]["excerpt"] = atoms[0]["source"]["excerpt"][:-3]
    elif change == "crossed":
        a, b = (2014, 2562) if index == 0 else (1193, 1647)
        atoms[0]["source"]["excerpt"] = BODY[a:b]
    elif change == "missing_total":
        atoms[:] = [a for a in atoms if a["source"]["excerpt"] != _PROFILES[index][1]]
    elif change == "overlap_rate":
        rate = next(r for r in d["policy"]["rules"] if r["kind"] == "parameter")
        rate["versions"].append(deepcopy(rate["versions"][0]))
    else:
        rule["versions"].append(deepcopy(rule["versions"][0]))
    assert (
        fraction_site(
            d["policy"],
            rule,
            d["case"],
            branch,
            BODY,
            CITATION,
            d["traces"][NAMES[index]],
            facts,
        )
        is None
    )
