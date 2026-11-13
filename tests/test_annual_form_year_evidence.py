"""Reporting years are source-owned only after the complete formula witness."""

import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.printed_source_rows import recognize_printed_rows

SOURCE = (Path(__file__).parent / "fixtures/multipage_t936_source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/cumulative-net-investment-loss"
DATA = json.loads(
    (Path(__file__).parent / "fixtures/t936_annuity_evidence.json").read_text()
)
EXTRACT = partial(
    v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)


def evaluate(rule, case, source=SOURCE, start=2239, end=2465):
    branch = c.SourceStructureBranch(
        (),
        "formula-clause",
        "annuity",
        source[start:end],
        start,
        end,
        structural_numeric_spans=tuple(
            (a - start, b - start)
            for a, b in c.form_numeric_locator_spans(source)
            if start <= a < b <= end
        ),
    )
    execution = c._case_formula_execution(
        rule, case, formula_environment={}, dependency_environment={}
    )
    tentative = c._annual_form_source_evidence(
        rule,
        case=case,
        execution=execution,
        branch=branch,
        source_text=source,
        corpus_citation_path=CITATION,
    )
    covered = set()
    found = c._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules={rule["name"]: rule},
        rule_names={rule["name"]},
        asserted_by_rule={rule["name"]: [case]},
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=source,
        represented_annual_spans=covered,
    )
    return tentative, found, covered


def test_actual_case61_reporting_year_and_existing85():
    r, case = deepcopy((DATA["rule"], DATA["case"]))
    tentative, found, covered = evaluate(r, case)
    assert tentative
    assert found
    assert covered == {(2277, 2281)}


def test_six_roles_only_on_exact_source_opt_in():
    assert recognize_printed_rows(SOURCE) is None
    roles = recognize_printed_rows(SOURCE, allow_multipage=True).annual_form_year_roles
    assert [r.year_span for r in roles] == [
        (1548, 1552),
        (1890, 1894),
        (2277, 2281),
        (2772, 2776),
        (5654, 5658),
        (6052, 6056),
    ]
    assert all(r.header_span == (0, SOURCE.index("\n")) for r in roles)
    assert recognize_printed_rows(SOURCE + "invented", allow_multipage=True) is None


@pytest.mark.parametrize(
    "change",
    [
        "wrong_expected",
        "missing_expected",
        "listed_expected",
        "missing_period",
        "numeric_period",
        "bad_month",
        "bad_dict",
        "reversed_dict",
        "outside",
        "crossyear_dict",
        "wrong_window",
        "overlap",
        "no_date_proof",
        "wrong_date_excerpt",
        "wrong_citation",
        "other_row_proof",
        "enclosing_proof",
        "source_drift",
        "missing_operand",
    ],
)
def test_missing_or_conflicting_evidence_refused(change):
    r, case = deepcopy((DATA["rule"], DATA["case"]))
    atoms = r["metadata"]["proof"]["atoms"]
    source = SOURCE
    output = next(k for k in case["output"] if k.endswith("#" + r["name"]))
    if change == "wrong_expected":
        case["output"][output] = 86
    elif change == "missing_expected":
        del case["output"][output]
    elif change == "listed_expected":
        case["output"][output] = [85]
    elif change == "missing_period":
        case.pop("period")
    elif change == "numeric_period":
        case["period"] = 2025
    elif change == "bad_month":
        case["period"] = "2025-13"
    elif change == "bad_dict":
        case["period"] = {"start": "2025-01-01"}
    elif change == "reversed_dict":
        case["period"]["end"] = "2024-12-31"
    elif change == "outside":
        case["period"] = "2026-01"
    elif change == "crossyear_dict":
        case["period"]["end"] = "2026-01-01"
    elif change == "wrong_window":
        r["versions"][0]["effective_from"] = "2025-02-01"
    elif change == "overlap":
        r["versions"].append(dict(r["versions"][0], effective_from="2025-03-01"))
    elif change == "no_date_proof":
        atoms.pop()
    elif change == "wrong_date_excerpt":
        atoms[-1]["source"]["excerpt"] = "2025"
    elif change == "wrong_citation":
        atoms[-1]["source"]["corpus_citation_path"] = "other"
    elif change == "other_row_proof":
        atoms[0]["source"]["excerpt"] = SOURCE[1507:1647]
    elif change == "enclosing_proof":
        atoms[0]["source"]["excerpt"] = SOURCE[2120:2466]
    elif change == "source_drift":
        source = source.replace("acquired", "obtained", 1)
    elif change == "missing_operand":
        case["input"].pop(
            next(
                k
                for k in case["input"]
                if k.endswith("#input.t936_annuity_capital_deduction_60_a")
            )
        )
    tentative, found, covered = evaluate(r, case, source)
    assert not tentative
    assert not covered


@pytest.mark.parametrize("period", ["2025", "2025-06", "2025-06-01"])
def test_explicit_normalized_periods(period):
    r, case = deepcopy((DATA["rule"], DATA["case"]))
    case["period"] = period
    assert evaluate(r, case)[2] == {(2277, 2281)}


def test_matching_result_cannot_hide_wrong_operation():
    r, case = deepcopy((DATA["rule"], DATA["case"]))
    r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(" - ", " + ")
    output = next(k for k in case["output"] if k.endswith("#" + r["name"]))
    case["output"][output] = 205
    tentative, found, covered = evaluate(r, case)
    assert tentative  # The annual role is provisional, never sufficient.
    assert not found
    assert not covered


def as_of_fixture():
    # Source-role unit fixture; this does not mint runtime provenance.
    header = SOURCE[: SOURCE.index("\n")]
    owner = SOURCE[6332:6408]
    rule = {
        "name": "cnil",
        "kind": "derived",
        "dtype": "Money",
        "unit": "CAD",
        "entity": "Person",
        "period": "Year",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": "max(0, expenses - income)",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0]." + field,
                        "kind": kind,
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                    for field, kind, excerpt in (
                        ("formula", "formula", owner),
                        ("effective_from", "effective_period", header),
                        ("effective_to", "effective_period", header),
                    )
                ]
            }
        },
    }
    return rule, {
        "period": {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
        "input": {"expenses": 200, "income": 100},
        "output": {"cnil": 100},
    }


def as_of_evaluate(rule, case, source=SOURCE):
    start, end = 6332, 6408
    branch = c.SourceStructureBranch(
        (), "formula-clause", "cnil", source[start:end], start, end
    )
    execution = c._case_formula_execution(
        rule, case, formula_environment={}, dependency_environment={}
    )
    tentative = c._annual_form_source_evidence(
        rule,
        case=case,
        execution=execution,
        branch=branch,
        source_text=source,
        corpus_citation_path=CITATION,
    )
    covered = set()
    found = c._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules={"cnil": rule},
        rule_names={"cnil"},
        asserted_by_rule={"cnil": [case]},
        extract_numeric_occurrences=partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=source,
        represented_annual_spans=covered,
    )
    return tentative, found, covered


@pytest.mark.parametrize(
    "period",
    [
        "2025",
        "2025-01",
        "2025-01-01",
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
    ],
)
def test_as_of_reporting_role_requires_full_annual_query(period):
    rule, case = as_of_fixture()
    case["period"] = period
    tentative, found, covered = as_of_evaluate(rule, case)
    assert tentative and found
    assert covered == {(6400, 6404)}
    structure = recognize_printed_rows(SOURCE, allow_multipage=True)
    assert len(structure.annual_form_year_roles) == 6
    assert len(structure.annual_form_as_of_roles) == 1
    assert structure.annual_form_as_of_roles[0].as_of_date == "2025-12-31"


@pytest.mark.parametrize(
    "change",
    [
        "midyear",
        "month_kind",
        "partial_year",
        "wrong_day",
        "wrong_month",
        "different_year",
        "monthly_rule",
        "wrong_expected",
        "wrong_operation",
        "wrong_owner",
        "wrong_header",
    ],
)
def test_as_of_role_does_not_hide_changed_cutoff_or_calculation(change):
    rule, case = as_of_fixture()
    source = SOURCE
    if change == "midyear":
        case["period"] = "2025-06-01"
    elif change == "month_kind":
        case["period"]["period_kind"] = "month"
    elif change == "partial_year":
        case["period"]["end"] = "2025-06-30"
    elif change == "wrong_day":
        source = source.replace("CNIL to December 31", "CNIL to December 30")
    elif change == "wrong_month":
        source = source.replace("CNIL to December 31", "CNIL to November 30")
    elif change == "different_year":
        source = source.replace(
            "CNIL to December 31, 2025", "CNIL to December 31, 2024"
        )
    elif change == "monthly_rule":
        rule["period"] = "Month"
    elif change == "wrong_expected":
        case["output"]["cnil"] = 101
    elif change == "wrong_operation":
        rule["versions"][0]["formula"] = "max(0, expenses + income)"
        case["output"]["cnil"] = 300
    elif change == "wrong_owner":
        rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = SOURCE[1835:1898]
    else:
        rule["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = "2025"
    _, found, covered = as_of_evaluate(rule, case, source)
    assert not found and not covered
