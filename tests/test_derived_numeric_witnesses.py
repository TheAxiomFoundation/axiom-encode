"""Derived threshold evidence requires replay and assertions at both endpoints."""

import pytest

from axiom_encode.harness import source_completeness as completeness
from tests.test_worksheet_source_clauses import EXTRACT, ROW, analyze, fixture, rule


def witnesses(payload, cases):
    rules = {r["name"]: r for r in payload["rules"]}
    asserted = {
        name: [case for case in cases if name in case["output"]] for name in rules
    }
    return {
        w
        for w in completeness._toggled_formula_numeric_selectors(
            rules,
            asserted_by_rule=asserted,
            formula_environment={},
        )
        if w.rule_name == "show_on_notice"
    }


def test_witness_uses_recomputed_balance_not_changed_opening_input():
    payload, cases = fixture()
    found = witnesses(payload, cases[:2])
    assert {w.selector_name for w in found} == {"amount_on_line_3"}
    assert {w.numeric_transition for w in found} == {(6000.0, 0.0), (0.0, 6000.0)}
    assert not analyze(payload, cases).issues


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_left_assertion",
        "missing_right_assertion",
        "wrong_left_assertion",
        "wrong_right_assertion",
        "no_notice_assertion",
        "wrong_notice_formula",
        "unresolved_dependency",
        "cycle",
        "two_inputs_changed",
        "different_periods",
        "unchanged_computed_selector",
        "boolean_dependency",
        "forged_input_alias",
    ],
)
def test_invalid_pairs_do_not_become_derived_witnesses(mutation):
    payload, all_cases = fixture()
    cases = all_cases[:2]
    if mutation.startswith("missing_"):
        del cases[0 if "left" in mutation else 1]["output"]["amount_on_line_3"]
    elif mutation.startswith("wrong_left") or mutation.startswith("wrong_right"):
        cases[0 if "left" in mutation else 1]["output"]["amount_on_line_3"] = 1
    elif mutation == "no_notice_assertion":
        for case in cases:
            del case["output"]["show_on_notice"]
    elif mutation == "wrong_notice_formula":
        payload["rules"][1]["versions"][0]["formula"] = "amount_on_line_3 < 0"
    elif mutation == "unresolved_dependency":
        payload["rules"][0]["versions"][0]["formula"] = "external_unresolved_value"
    elif mutation == "cycle":
        payload["rules"][0]["versions"][0]["formula"] = "amount_on_line_3 + 1"
    elif mutation == "two_inputs_changed":
        cases[1]["input"]["benefits_line_2"] = 0
        cases[1]["output"]["amount_on_line_3"] = 4000
        cases[1]["output"]["show_on_notice"] = "holds"
    elif mutation == "different_periods":
        cases[1]["period"] = "2026-01-01"
    elif mutation == "unchanged_computed_selector":
        payload["rules"][0]["versions"][0]["formula"] = "benefits_line_2"
        for case in cases:
            case["output"]["amount_on_line_3"] = 4000
            case["output"]["show_on_notice"] = "holds"
    elif mutation == "boolean_dependency":
        payload["rules"][0]["versions"][0]["formula"] = (
            "opening_balance_line_1 > benefits_line_2"
        )
        for case in cases:
            case["output"]["amount_on_line_3"] = (
                case["input"]["opening_balance_line_1"] > 4000
            )
    elif mutation == "forged_input_alias":
        # Only the fake alias changes; the raw opening balance is stable.
        cases[1]["input"]["opening_balance_line_1"] = cases[0]["input"][
            "opening_balance_line_1"
        ]
        for case in cases:
            case["input"]["amount_on_line_3"] = case["output"]["amount_on_line_3"]
    assert not witnesses(payload, cases)


def test_intermediate_dependency_must_be_asserted_and_replayed_too():
    payload, all_cases = fixture()
    cases = all_cases[:2]
    payload["rules"].insert(
        0, rule("intermediate_balance", "opening_balance_line_1 - benefits_line_2", ROW)
    )
    payload["rules"][1]["versions"][0]["formula"] = "intermediate_balance + 0"
    assert not witnesses(payload, cases)
    for case in cases:
        case["output"]["intermediate_balance"] = case["output"]["amount_on_line_3"]
    assert witnesses(payload, cases)
    cases[0]["output"]["intermediate_balance"] = 1
    assert not witnesses(payload, cases)


@pytest.mark.parametrize("quoted", ['"0,"', '"0"', "“0,”", "“0”"])
def test_quoted_integer_threshold_keeps_value_and_exclusivity(quoted):
    text = f"If the amount is more than {quoted}, show it."
    interval = completeness._formula_interval_from_text(
        text, extract_numeric_occurrences=EXTRACT
    )
    assert interval is not None
    assert not completeness._interval_contains(interval, 0)
    assert completeness._interval_contains(interval, 1)
    masked = completeness._mask_quoted_integer_delimiters(text)
    assert len(masked) == len(text)
    assert [o.value for o in EXTRACT(masked)] == [0]


@pytest.mark.parametrize("quoted", ['"0', '0"', '“0"', '"zero"', '"0 or 1"'])
def test_unsupported_or_unbalanced_quotes_are_not_normalized(quoted):
    assert completeness._mask_quoted_integer_delimiters(quoted) == quoted


def test_opaque_import_assertions_are_not_independent_execution_evidence():
    payload, all_cases = fixture()
    cases = all_cases[:2]
    payload["rules"][0]["versions"][0]["formula"] = "external_balance"
    payload["rules"][0]["metadata"]["proof"]["atoms"].append(
        {
            "path": "versions[0].formula",
            "kind": "import",
            "import": {
                "target": "ca:policies/external#external_balance",
                "output": "external_balance",
                "hash": "sha256:" + "a" * 64,
            },
        }
    )
    rules = {r["name"]: r for r in payload["rules"]}
    for case in cases:
        asserted = completeness._case_asserted_dependency_environment(
            rules, case, formula_environment={}
        )
        replayed = completeness._case_dependency_environment(
            rules, case, formula_environment={}, require_asserted_value=False
        )
        assert "amount_on_line_3" in asserted
        assert "amount_on_line_3" not in replayed
    assert not witnesses(payload, cases)
