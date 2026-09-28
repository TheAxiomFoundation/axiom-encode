"""Regressions from the rejected Rev. Proc. 2025-32 page-14 encoder candidate."""

from __future__ import annotations

import ast
import copy
import functools
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _rewrite_judgment_conditional_formulas
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURE = Path(__file__).parent / "fixtures/source_completeness/irs_rev_proc_2025_32"
CITATION = "us/guidance/irs/rev-proc-2025-32/page-14"
REFERENCE = "us:policies/irs/rev-proc-2025-32/child-tax-credit#"
SELECTOR = "earned_income_is_greater_than_adjusted_gross_income"
EXTRACT = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)
OUTPUTS = (
    (
        "earned_income_credit_maximum_amount_begins_to_phase_out",
        "threshold_phaseout_amount",
        23890,
        ">",
    ),
    (
        "earned_income_credit_is_fully_phased_out",
        "completed_phaseout_amount",
        51593,
        ">=",
    ),
)


def _candidate():
    folder = FIXTURE / "max_formula"
    return (
        yaml.safe_load((folder / "child-tax-credit.yaml").read_text()),
        yaml.safe_load((folder / "child-tax-credit.test.yaml").read_text()),
    )


def _binding_pair(rule_name, threshold_name, threshold, *, named=False):
    inputs = {
        REFERENCE + "input.adjusted_gross_income": threshold - 1,
        REFERENCE + "input.earned_income": threshold - 2,
        REFERENCE + "input." + threshold_name: threshold,
    }
    if named:
        inputs[REFERENCE + "input.earned_income"] = threshold + 1
        inputs[REFERENCE + "input." + SELECTOR] = "not_holds"
    ordinary = {
        "name": rule_name + "_ordinary",
        "period": {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        "input": inputs,
        "output": {REFERENCE + rule_name: "not_holds"},
    }
    alternative = copy.deepcopy(ordinary)
    alternative["name"] = rule_name + "_greater_earned_income"
    alternative["input"][
        REFERENCE + "input." + (SELECTOR if named else "earned_income")
    ] = "holds" if named else threshold + 1
    alternative["output"][REFERENCE + rule_name] = "holds"
    return [ordinary, alternative]


def _paired_issues(payload, cases):
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        json.loads((FIXTURE / "page-14.json").read_text())["body"],
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [
        issue for issue in result.issues if "require paired positive/blocking" in issue
    ]


def test_original_max_candidate_does_not_switch_the_binding_operand():
    payload, cases = _candidate()
    assert _paired_issues(payload, cases)


@pytest.mark.parametrize("reverse_arguments", [False, True])
@pytest.mark.parametrize("guarded", [False, True])
def test_real_candidate_accepts_pairs_switching_the_max_operand(
    reverse_arguments, guarded
):
    payload, cases = _candidate()
    if reverse_arguments:
        for rule in payload["rules"]:
            rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
                "max(adjusted_gross_income, earned_income)",
                "max(earned_income, adjusted_gross_income)",
            )
    for name, threshold_name, threshold, _ in OUTPUTS:
        if guarded:
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            formula = rule["versions"][0]["formula"]
            rule["versions"][0]["formula"] = (
                f"if {threshold_name} > 0: {formula} else: false"
            )
        cases.extend(_binding_pair(name, threshold_name, threshold))
    assert not _paired_issues(payload, cases)


def test_real_candidate_named_judgment_selector_survives_auto_repair(tmp_path):
    payload, cases = _candidate()
    payload["inputs"].append(
        {"name": SELECTOR, "entity": "TaxUnit", "dtype": "Judgment", "period": "Year"}
    )
    for name, threshold_name, threshold, comparison in OUTPUTS:
        rule = next(rule for rule in payload["rules"] if rule["name"] == name)
        rule["versions"][0]["formula"] = (
            f"if {SELECTOR}: earned_income {comparison} {threshold_name} "
            f"else: adjusted_gross_income {comparison} {threshold_name}"
        )
        cases = [case for case in cases if REFERENCE + name not in case["output"]]
        cases.extend(_binding_pair(name, threshold_name, threshold, named=True))
    rules_file = tmp_path / "candidate.yaml"
    rules_file.write_text(yaml.safe_dump(payload))
    assert set(_rewrite_judgment_conditional_formulas(rules_file)) == {
        item[0] for item in OUTPUTS
    }
    repaired = yaml.safe_load(rules_file.read_text())
    assert not _paired_issues(repaired, cases)


@pytest.mark.parametrize(
    "invalid_pair",
    [
        "same_binding",
        "same_output",
        "wrong_assertion",
        "different_period",
        "two_inputs",
        "inactive_max",
        "cancelled_max",
        "unreached_chained_max",
        "dead_numeric_comparison",
        "duplicated_cancelled_max",
    ],
)
def test_max_binding_evidence_must_be_executed_isolated_and_change_output(invalid_pair):
    payload, cases = _candidate()
    for name, threshold_name, threshold, _ in OUTPUTS:
        pair = _binding_pair(name, threshold_name, threshold)
        if invalid_pair == "same_binding":
            for case in pair:
                case["input"][REFERENCE + "input.adjusted_gross_income"] = threshold - 3
        elif invalid_pair == "same_output":
            pair[0]["input"][REFERENCE + "input." + threshold_name] = threshold + 2
            pair[1]["input"][REFERENCE + "input." + threshold_name] = threshold + 2
            pair[1]["output"][REFERENCE + name] = "not_holds"
        elif invalid_pair == "wrong_assertion":
            pair[1]["output"][REFERENCE + name] = "not_holds"
        elif invalid_pair == "different_period":
            pair[1]["period"]["start"] = "2027-01-01"
            pair[1]["period"]["end"] = "2027-12-31"
        elif invalid_pair == "two_inputs":
            pair[1]["input"][REFERENCE + "input.adjusted_gross_income"] -= 1
        elif invalid_pair == "inactive_max":
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            rule["versions"][0]["formula"] = (
                "(false and max(adjusted_gross_income, earned_income) > "
                f"{threshold_name}) or earned_income > {threshold_name}"
            )
        elif invalid_pair == "cancelled_max":
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            rule["versions"][0]["formula"] = (
                "0 * max(adjusted_gross_income, earned_income) + "
                f"earned_income > {threshold_name}"
            )
        elif invalid_pair == "unreached_chained_max":
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            rule["versions"][0]["formula"] = (
                "(0 > 1 > max(adjusted_gross_income, earned_income)) or "
                f"earned_income > {threshold_name}"
            )
        elif invalid_pair == "dead_numeric_comparison":
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            rule["versions"][0]["formula"] = (
                "(false and earned_income > adjusted_gross_income) or "
                f"earned_income > {threshold_name}"
            )
        elif invalid_pair == "duplicated_cancelled_max":
            rule = next(rule for rule in payload["rules"] if rule["name"] == name)
            rule["versions"][0]["formula"] = (
                "max(adjusted_gross_income, earned_income) - "
                "max(earned_income, adjusted_gross_income) + "
                f"earned_income > {threshold_name}"
            )
        # Isolate new cases from the original pairs to prevent accidental cross-pairs.
        cases = [case for case in cases if REFERENCE + name not in case["output"]]
        cases.extend(pair)
    assert _paired_issues(payload, cases)


@pytest.mark.parametrize("first_operand, reached", [(0, False), (2, True)])
def test_maximum_reachability_respects_chained_comparison(first_operand, reached):
    expression = ast.parse(
        f"({first_operand} > 1 > max(adjusted_gross_income, earned_income)) "
        "or earned_income > threshold",
        mode="eval",
    ).body
    maxima = list(
        sc._formula_maximum_operands(
            expression,
            environment={
                "adjusted_gross_income": 10,
                "earned_income": 11,
                "threshold": 10,
            },
        )
    )
    assert bool(maxima) is reached


def test_max_pair_does_not_cover_another_source_condition():
    payload, cases = _candidate()
    name, threshold_name, threshold, _ = OUTPUTS[0]
    cases.extend(_binding_pair(name, threshold_name, threshold))
    issues = _paired_issues(payload, cases)
    assert issues
    assert 'The "completed phaseout amount"' in issues[0]
    assert 'The "threshold phaseout amount"' not in issues[0]


def test_max_binding_may_switch_when_adjusted_gross_income_decreases():
    payload, cases = _candidate()
    for name, threshold_name, threshold, _ in OUTPUTS:
        pair = _binding_pair(name, threshold_name, threshold)
        for index, case in enumerate(pair):
            case["input"][REFERENCE + "input.earned_income"] = threshold - 1
            case["input"][REFERENCE + "input.adjusted_gross_income"] = (
                threshold + 1 if index == 0 else threshold - 2
            )
            case["output"][REFERENCE + name] = "holds" if index == 0 else "not_holds"
        cases = [case for case in cases if REFERENCE + name not in case["output"]]
        cases.extend(pair)
    assert not _paired_issues(payload, cases)


@settings(max_examples=30, deadline=None)
@given(
    threshold=st.integers(min_value=10, max_value=1_000_000),
    below=st.integers(min_value=1, max_value=9),
    above=st.integers(min_value=1, max_value=100),
    reverse_arguments=st.booleans(),
)
def test_max_binding_witness_tracks_the_compared_operands(
    threshold, below, above, reverse_arguments
):
    payload, _ = _candidate()
    name, threshold_name, _, _ = OUTPUTS[0]
    rule = next(rule for rule in payload["rules"] if rule["name"] == name)
    if reverse_arguments:
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            "max(adjusted_gross_income, earned_income)",
            "max(earned_income, adjusted_gross_income)",
        )
    pair = _binding_pair(name, threshold_name, threshold)
    pair[0]["input"][REFERENCE + "input.adjusted_gross_income"] = threshold
    pair[1]["input"][REFERENCE + "input.adjusted_gross_income"] = threshold
    pair[0]["input"][REFERENCE + "input.earned_income"] = threshold - below
    pair[1]["input"][REFERENCE + "input.earned_income"] = threshold + above
    source = next(
        atom["source"]["excerpt"]
        for atom in rule["metadata"]["proof"]["atoms"]
        if atom["path"] == "versions[0].formula"
    )
    branch = sc.SourceStructureBranch(
        (), "exception-clause", "source unit", source, 0, len(source)
    )
    witnesses = sc._toggled_formula_boolean_selectors(
        {name: rule}, asserted_by_rule={name: pair}, formula_environment={}
    )
    assert any(
        sc._numeric_exception_witness_matches_source(
            branch, witness, extract_numeric_occurrences=EXTRACT
        )
        for witness in witnesses
    )
    # Similar terminology is insufficient: both actual operands must match.
    unrelated = sc.SourceStructureBranch(
        (),
        "exception-clause",
        "source unit",
        source.replace("earned income", "investment income"),
        0,
        len(source),
    )
    assert not any(
        sc._numeric_exception_witness_matches_source(
            unrelated, witness, extract_numeric_occurrences=EXTRACT
        )
        for witness in witnesses
    )
