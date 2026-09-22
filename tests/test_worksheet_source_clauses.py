"""Worksheet line labels must not become required arithmetic constants."""

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "ca/policy/worksheet"
ROW = "Closing balance: line 1 minus line 2 65220 = 3"
NOTICE = 'If the amount on line 3 is more than "0," show it on the notice.'
SOURCE = ROW + "\n" + NOTICE
EXTRACT = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)


def rule(name, formula, excerpt, dtype="Money"):
    return {
        "name": name,
        "kind": "derived",
        "entity": "Person",
        "dtype": dtype,
        "period": "Year",
        "versions": [{"effective_from": "2025-01-01", "formula": formula}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
    }


def fixture():
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": name, "entity": "Person", "dtype": "Money", "period": "Year"}
            for name in ("opening_balance_line_1", "benefits_line_2")
        ],
        "rules": [
            rule("amount_on_line_3", "opening_balance_line_1 - benefits_line_2", ROW),
            rule("show_on_notice", "amount_on_line_3 > 0", NOTICE, "Judgment"),
        ],
    }
    cases = [
        {
            "name": f"balance_{opening}",
            "period": "2025-01-01",
            "input": {"opening_balance_line_1": opening, "benefits_line_2": 4000},
            "output": {
                "amount_on_line_3": opening - 4000,
                "show_on_notice": "holds" if opening > 4000 else "not_holds",
            },
        }
        for opening in (10000, 4000, 3000)
    ]
    return payload, cases


def analyze(payload, cases):
    return completeness.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=(0.0,),
    )


def test_completed_worksheet_row_preserves_offsets_and_notice_threshold():
    clauses = list(completeness._source_clause_spans(SOURCE, branches=()))
    assert [text for _, _, text in clauses] == [ROW, NOTICE]
    assert all(SOURCE[start:end] == text for start, end, text in clauses)
    assert {
        o.value for o in EXTRACT(completeness.authoritative_numeric_recall_text(SOURCE))
    } == {0}


def test_subtraction_is_covered_and_derived_notice_blocker_remains_explicit():
    payload, cases = fixture()
    issues = analyze(payload, cases).issues
    assert len(issues) == 1
    assert "paired positive/blocking cases" in issues[0]
    assert NOTICE in issues[0]


@pytest.mark.parametrize(
    "mutation",
    ["addition", "no_arithmetic_assertion", "no_notice_assertion"],
)
def test_incorrect_or_unasserted_computation_is_not_credited(mutation):
    payload, cases = fixture()
    if mutation == "addition":
        payload["rules"][0]["versions"][0]["formula"] = (
            "opening_balance_line_1 + benefits_line_2"
        )
    else:
        key = (
            "amount_on_line_3"
            if mutation == "no_arithmetic_assertion"
            else "show_on_notice"
        )
        for case in cases:
            del case["output"][key]
    issues = analyze(payload, cases).issues
    if mutation == "addition":
        assert any("do not demonstrate formula branch" in issue for issue in issues)
    else:
        assert any(
            f"Principal output `{key}` is never asserted" in issue for issue in issues
        )


@pytest.mark.parametrize(
    "source",
    [
        ROW,
        ROW + "\n" + NOTICE.replace("line 3", "line 4"),
        SOURCE.replace("65220 = 3", "= 3"),
        SOURCE.replace("= 3", "= 3.5"),
        "The payment equals 3. If the amount on line 3 is more than 0, show it.",
    ],
)
def test_unconfirmed_labels_and_substantive_constants_are_not_masked(source):
    assert not completeness._worksheet_arithmetic_rows(source)
    assert any(
        o.value in (3, 3.5)
        for o in EXTRACT(completeness.authoritative_numeric_recall_text(source))
    )
