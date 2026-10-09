"""Regressions from the rejected Rev. Proc. 2025-32 page-14 encoder candidate."""

from __future__ import annotations

import ast
import copy
import functools
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _rewrite_judgment_conditional_formulas
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
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


def _paired_issues(payload, cases, *, source=None):
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source or json.loads((FIXTURE / "page-14.json").read_text())["body"],
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


def test_max_operand_pairs_do_not_cover_an_independent_residency_condition():
    payload, cases = _candidate()
    for name, threshold_name, threshold, _ in OUTPUTS:
        cases.extend(_binding_pair(name, threshold_name, threshold))
    source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
    assert not _paired_issues(payload, cases, source=source)
    source = source.replace(
        "above which the maximum amount of the credit begins to phase out.",
        "above which the maximum amount of the credit begins to phase out, "
        "and only when the applicant is resident.",
    )
    assert "resident" not in yaml.safe_dump(payload)
    assert "resident" not in yaml.safe_dump(cases)
    assert _paired_issues(payload, cases, source=source)


@settings(max_examples=30, deadline=None)
@given(
    threshold=st.integers(min_value=10, max_value=1_000_000),
    reverse_arguments=st.booleans(),
    condition=st.sampled_from(
        [
            ", and only when the applicant is resident",
            ", but only if the applicant is resident",
            ", unless the applicant is nonresident",
        ]
    ),
)
def test_max_witness_credit_is_limited_to_its_isolated_condition(
    threshold, reverse_arguments, condition
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
    source = next(
        atom["source"]["excerpt"]
        for atom in rule["metadata"]["proof"]["atoms"]
        if atom["path"] == "versions[0].formula"
    )
    witnesses = sc._toggled_formula_boolean_selectors(
        {name: rule}, asserted_by_rule={name: pair}, formula_environment={}
    )
    branch = sc.SourceStructureBranch(
        (), "exception-clause", "source unit", source, 0, len(source)
    )
    assert any(
        sc._numeric_exception_witness_matches_source(
            branch, witness, extract_numeric_occurrences=EXTRACT
        )
        for witness in witnesses
    )
    mixed_source = source.rstrip(".") + condition + "."
    mixed = sc.SourceStructureBranch(
        (), "exception-clause", "source unit", mixed_source, 0, len(mixed_source)
    )
    assert not any(
        sc._numeric_exception_witness_matches_source(
            mixed, witness, extract_numeric_occurrences=EXTRACT
        )
        for witness in witnesses
    )


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


def test_named_max_selector_cannot_cover_an_independent_condition(tmp_path):
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
    source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
    assert not _paired_issues(repaired, cases, source=source)
    source = source.replace(
        "above which the maximum amount of the credit begins to phase out.",
        "above which the maximum amount of the credit begins to phase out, "
        "and only when the applicant is resident.",
    )
    assert "resident" not in yaml.safe_dump(repaired)
    assert "resident" not in yaml.safe_dump(cases)
    assert _paired_issues(repaired, cases, source=source)


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


@pytest.mark.parametrize("alternative_count", [1, 2])
@pytest.mark.parametrize("incorrect_completed_formula", [False, True])
def test_max_pair_does_not_cover_another_source_condition(
    alternative_count, incorrect_completed_formula
):
    payload, cases = _candidate()
    name, threshold_name, threshold, _ = OUTPUTS[0]
    pair = _binding_pair(name, threshold_name, threshold)
    cases.extend(pair)
    if alternative_count == 2:
        alternative = copy.deepcopy(pair[1])
        alternative["name"] = "another_threshold_alternative"
        alternative["input"][REFERENCE + "input.earned_income"] = threshold + 2
        cases.append(alternative)
    if incorrect_completed_formula:
        completed_rule = next(
            rule for rule in payload["rules"] if rule["name"] == OUTPUTS[1][0]
        )
        completed_rule["versions"][0]["formula"] = (
            "adjusted_gross_income >= completed_phaseout_amount"
        )
    issues = _paired_issues(payload, cases)
    assert issues
    assert 'The "completed phaseout amount"' in issues[0]
    assert 'The "threshold phaseout amount"' not in issues[0]


def test_max_threshold_pairs_with_split_exact_atoms_leave_completed_unwitnessed():
    payload, cases = _candidate()
    name, threshold_name, threshold, _ = OUTPUTS[0]
    pair = _binding_pair(name, threshold_name, threshold)
    alternative = copy.deepcopy(pair[1])
    alternative["name"] = "another_threshold_alternative"
    alternative["input"][REFERENCE + "input.earned_income"] = threshold + 2
    cases.extend(pair + [alternative])
    threshold_rule, completed_rule = [
        next(rule for rule in payload["rules"] if rule["name"] == output[0])
        for output in OUTPUTS
    ]
    completed_rule["versions"][0]["formula"] = (
        "adjusted_gross_income >= completed_phaseout_amount"
    )
    threshold_rule["metadata"]["proof"]["atoms"].insert(
        1, copy.deepcopy(completed_rule["metadata"]["proof"]["atoms"][0])
    )
    source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={CITATION: source},
    )
    assert proof.passed
    assert proof.atoms_checked == 23
    assert not proof.issues
    environment = {
        "adjusted_gross_income": 51590,
        "earned_income": 52000,
        "completed_phaseout_amount": 51593,
    }
    assert (
        sc._evaluate_rulespec_formula(
            completed_rule["versions"][0]["formula"], environment=environment
        )
        is False
    )
    assert (
        sc._evaluate_rulespec_formula(
            "max(adjusted_gross_income, earned_income) >= completed_phaseout_amount",
            environment=environment,
        )
        is True
    )
    issues = _paired_issues(payload, cases)
    assert any('The "completed phaseout amount"' in issue for issue in issues)


@pytest.mark.parametrize("output_index", [0, 1])
def test_max_clause_witness_stays_with_defined_output_when_formula_and_atom_move(
    output_index,
):
    payload, cases = _candidate()
    definition_rules = [
        next(rule for rule in payload["rules"] if rule["name"] == output[0])
        for output in OUTPUTS
    ]
    missing_index = 1 - output_index
    witnessed_rule, missing_rule = (
        definition_rules[output_index],
        definition_rules[missing_index],
    )
    # Copying the comparison and proof cannot change which output the source
    # clause defines: these companion pairs still assert only its sibling.
    witnessed_rule["metadata"]["proof"]["atoms"][0] = copy.deepcopy(
        missing_rule["metadata"]["proof"]["atoms"][0]
    )
    witnessed_rule["versions"][0]["formula"] = missing_rule["versions"][0]["formula"]
    _, missing_threshold, threshold, comparison = OUTPUTS[missing_index]
    missing_rule["versions"][0]["formula"] = (
        f"adjusted_gross_income {comparison} {missing_threshold}"
    )
    pair = _binding_pair(OUTPUTS[output_index][0], missing_threshold, threshold)
    alternative = copy.deepcopy(pair[1])
    alternative["name"] = OUTPUTS[output_index][0] + "_another_alternative"
    alternative["input"][REFERENCE + "input.earned_income"] = threshold + 2
    cases.extend(pair + [alternative])
    source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={CITATION: source},
    )
    assert proof.passed
    assert proof.atoms_checked == 22
    assert not proof.issues
    missing_term = ("threshold", "completed")[missing_index]
    issues = _paired_issues(payload, cases)
    assert any(f'The "{missing_term} phaseout amount"' in issue for issue in issues)


@settings(
    max_examples=30,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@example(
    output_index=0,
    atom_placement=(1 << 7, (1 << 7) | (1 << 8)),
    incorrect_unwitnessed_formula=True,
    witnessed_formula_defines_missing_output=False,
)
@example(
    output_index=0,
    atom_placement=(0, 1 << 7),
    incorrect_unwitnessed_formula=True,
    witnessed_formula_defines_missing_output=False,
)
@example(
    output_index=1,
    atom_placement=(1 << 8, 0),
    incorrect_unwitnessed_formula=True,
    witnessed_formula_defines_missing_output=False,
)
@example(
    output_index=0,
    atom_placement=(0, 1 << 7),
    incorrect_unwitnessed_formula=True,
    witnessed_formula_defines_missing_output=True,
)
@example(
    output_index=1,
    atom_placement=(1 << 8, 0),
    incorrect_unwitnessed_formula=True,
    witnessed_formula_defines_missing_output=True,
)
@given(
    output_index=st.integers(min_value=0, max_value=1),
    atom_placement=st.tuples(
        st.integers(min_value=0, max_value=(1 << 10) - 1),
        st.integers(min_value=0, max_value=(1 << 10) - 1),
    ),
    incorrect_unwitnessed_formula=st.booleans(),
    witnessed_formula_defines_missing_output=st.booleans(),
)
def test_max_clause_obligations_survive_arbitrary_exact_atom_placement(
    output_index,
    atom_placement,
    incorrect_unwitnessed_formula,
    witnessed_formula_defines_missing_output,
):
    payload, cases = _candidate()
    definition_rules = [
        next(rule for rule in payload["rules"] if rule["name"] == output[0])
        for output in OUTPUTS
    ]
    exact_atoms = [
        copy.deepcopy(rule["metadata"]["proof"]["atoms"][0])
        for rule in definition_rules
    ]
    assert len(payload["rules"]) == 10
    # Place either clause's exact atom on any subset of the fixture's rules.
    # Moving proof text cannot provide a pair on the clause's affected output.
    for rule_index, rule in enumerate(payload["rules"]):
        atoms = rule["metadata"]["proof"]["atoms"]
        atoms[:] = [atom for atom in atoms if atom not in exact_atoms]
        for atom, placement in zip(exact_atoms, atom_placement, strict=True):
            if placement & (1 << rule_index):
                atoms.append(copy.deepcopy(atom))
    missing_index = 1 - output_index
    name = OUTPUTS[output_index][0]
    pair_index = (
        missing_index if witnessed_formula_defines_missing_output else output_index
    )
    _, threshold_name, threshold, _ = OUTPUTS[pair_index]
    if witnessed_formula_defines_missing_output:
        definition_rules[output_index]["versions"][0]["formula"] = definition_rules[
            missing_index
        ]["versions"][0]["formula"]
    pair = _binding_pair(name, threshold_name, threshold)
    alternative = copy.deepcopy(pair[1])
    alternative["name"] = name + "_another_alternative"
    alternative["input"][REFERENCE + "input.earned_income"] = threshold + 2
    cases.extend(pair + [alternative])
    if incorrect_unwitnessed_formula:
        _, missing_threshold, _, comparison = OUTPUTS[missing_index]
        definition_rules[missing_index]["versions"][0]["formula"] = (
            f"adjusted_gross_income {comparison} {missing_threshold}"
        )
    missing_term = ("threshold", "completed")[missing_index]
    issues = _paired_issues(payload, cases)
    assert any(f'The "{missing_term} phaseout amount"' in issue for issue in issues)


@pytest.mark.parametrize(
    "extra_atom", ["same_clause", "metadata_clause", "inactive_version_clause"]
)
def test_max_witness_ownership_ignores_duplicate_or_unexecuted_atoms(extra_atom):
    payload, cases = _candidate()
    for name, threshold_name, threshold, _ in OUTPUTS:
        cases.extend(_binding_pair(name, threshold_name, threshold))
    threshold_rule, completed_rule = [
        next(rule for rule in payload["rules"] if rule["name"] == output[0])
        for output in OUTPUTS
    ]
    copied_rule = threshold_rule if extra_atom == "same_clause" else completed_rule
    atom = copy.deepcopy(copied_rule["metadata"]["proof"]["atoms"][0])
    if extra_atom == "metadata_clause":
        threshold_rule["metadata"]["summary"] = atom["source"]["excerpt"]
        atom["path"] = "metadata.summary"
    elif extra_atom == "inactive_version_clause":
        inactive = copy.deepcopy(threshold_rule["versions"][0])
        inactive["effective_from"] = "2027-01-01"
        inactive["effective_to"] = "2027-12-31"
        threshold_rule["versions"].append(inactive)
        atom["path"] = "versions[1].formula"
    threshold_rule["metadata"]["proof"]["atoms"].append(atom)
    assert not _paired_issues(payload, cases)


@settings(
    max_examples=30,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@example(output_index=0, threshold=23890, alternative_count=2, reverse_arguments=False)
@given(
    output_index=st.integers(min_value=0, max_value=1),
    threshold=st.integers(min_value=10, max_value=1_000_000),
    alternative_count=st.integers(min_value=2, max_value=5),
    reverse_arguments=st.booleans(),
)
def test_max_witnesses_cannot_certify_a_sibling_definition(
    output_index, threshold, alternative_count, reverse_arguments
):
    payload, cases = _candidate()
    name, threshold_name, _, _ = OUTPUTS[output_index]
    if reverse_arguments:
        for rule in payload["rules"]:
            rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
                "max(adjusted_gross_income, earned_income)",
                "max(earned_income, adjusted_gross_income)",
            )
    pair = _binding_pair(name, threshold_name, threshold)
    cases.extend(pair)
    for offset in range(2, alternative_count + 1):
        alternative = copy.deepcopy(pair[1])
        alternative["name"] = f"{name}_alternative_{offset}"
        alternative["input"][REFERENCE + "input.earned_income"] = threshold + offset
        cases.append(alternative)
    issues = _paired_issues(payload, cases)
    missing_term = ("completed", "threshold")[output_index]
    witnessed_term = ("threshold", "completed")[output_index]
    assert issues
    assert f'The "{missing_term} phaseout amount"' in issues[0]
    assert f'The "{witnessed_term} phaseout amount"' not in issues[0]


@pytest.mark.parametrize(
    "invalid_binding",
    [
        "shared_excerpt",
        "broad_excerpt",
        "sibling_excerpt",
        "different_citation",
        "different_version",
    ],
)
def test_max_witness_requires_its_own_executed_formula_source(invalid_binding):
    payload, cases = _candidate()
    for name, threshold_name, threshold, _ in OUTPUTS:
        cases.extend(_binding_pair(name, threshold_name, threshold))
    completed_rule = next(
        rule for rule in payload["rules"] if rule["name"] == OUTPUTS[1][0]
    )
    atom = completed_rule["metadata"]["proof"]["atoms"][0]
    if invalid_binding == "shared_excerpt":
        atom["source"]["excerpt"] = (
            "adjusted gross income (or, if greater, earned income)"
        )
    elif invalid_binding == "broad_excerpt":
        source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
        atom["source"]["excerpt"] = next(
            branch.text
            for branch in sc.recognize_source_structure(
                source, corpus_citation_path=CITATION
            )
            if branch.path == ("06", "1")
        )
    elif invalid_binding == "sibling_excerpt":
        threshold_rule = next(
            rule for rule in payload["rules"] if rule["name"] == OUTPUTS[0][0]
        )
        atom["source"]["excerpt"] = threshold_rule["metadata"]["proof"]["atoms"][0][
            "source"
        ]["excerpt"]
    elif invalid_binding == "different_citation":
        atom["source"]["corpus_citation_path"] = CITATION.replace("page-14", "page-15")
    elif invalid_binding == "different_version":
        atom["path"] = "versions[1].formula"
    issues = _paired_issues(payload, cases)
    assert issues
    assert 'The "completed phaseout amount"' in issues[0]


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
