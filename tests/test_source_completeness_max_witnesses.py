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


def _paired_issues(payload, cases, *, source=None, citation=CITATION):
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source or json.loads((FIXTURE / "page-14.json").read_text())["body"],
        corpus_citation_path=citation,
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


def test_numeric_max_guard_cannot_credit_a_comparison_of_the_lesser_income():
    payload, cases = _candidate()
    name, threshold_name, threshold, _ = OUTPUTS[0]
    rule = next(rule for rule in payload["rules"] if rule["name"] == name)
    rule["versions"][0]["formula"] = (
        "if max(adjusted_gross_income, earned_income) > adjusted_gross_income: "
        "adjusted_gross_income > threshold_phaseout_amount "
        "else: earned_income > threshold_phaseout_amount"
    )
    pair = _binding_pair(name, threshold_name, threshold)
    for index, case in enumerate(pair):
        case["input"][REFERENCE + "input.adjusted_gross_income"] = 23891
        case["input"][REFERENCE + "input.earned_income"] = (23890, 23893)[index]
        environment = {
            key.rsplit("input.", 1)[-1]: value for key, value in case["input"].items()
        }
        # Independent arithmetic oracle: the source maximum is above its
        # threshold in both executions, although the candidate switches output.
        required = (
            max(environment["adjusted_gross_income"], environment["earned_income"])
            > environment[threshold_name]
        )
        assert required is True
        execution = sc._execute_formula_text(
            rule["versions"][0]["formula"],
            environment=environment,
            constant_environment={},
        )
        assert sc._formula_execution_runtime_value(execution) is bool(index)
    cases = [case for case in cases if REFERENCE + name not in case["output"]]
    cases.extend(pair)
    cases.extend(_binding_pair(*OUTPUTS[1][:3]))
    source = json.loads((FIXTURE / "page-14.json").read_text())["body"]
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={CITATION: source},
    )
    assert proof.passed
    assert proof.atoms_checked == 22
    assert not proof.issues
    issues = _paired_issues(payload, cases)
    assert any('The "threshold phaseout amount"' in issue for issue in issues)
    assert not any('The "completed phaseout amount"' in issue for issue in issues)


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


@pytest.mark.parametrize("repaired", [False, True])
def test_named_max_selector_requires_the_source_arm_polarity(tmp_path, repaired):
    payload, cases = _candidate()
    payload["inputs"].append(
        {"name": SELECTOR, "entity": "TaxUnit", "dtype": "Judgment", "period": "Year"}
    )
    name, threshold_name, threshold, comparison = OUTPUTS[0]
    rule = next(rule for rule in payload["rules"] if rule["name"] == name)
    rule["versions"][0]["formula"] = (
        f"if {SELECTOR}: adjusted_gross_income {comparison} {threshold_name} "
        f"else: earned_income {comparison} {threshold_name}"
    )
    pair = _binding_pair(name, threshold_name, threshold, named=True)
    for index, case in enumerate(pair):
        # The declared selector authorizes alternative income in its true arm;
        # the swapped candidate instead changes from true to false.
        case["output"][REFERENCE + name] = "holds" if index == 0 else "not_holds"
    cases = [case for case in cases if REFERENCE + name not in case["output"]]
    cases.extend(pair)
    cases.extend(_binding_pair(*OUTPUTS[1][:3]))
    if repaired:
        rules_file = tmp_path / "candidate.yaml"
        rules_file.write_text(yaml.safe_dump(payload))
        assert _rewrite_judgment_conditional_formulas(rules_file) == [name]
        payload = yaml.safe_load(rules_file.read_text())
        rule = next(rule for rule in payload["rules"] if rule["name"] == name)
    for index, case in enumerate(pair):
        environment = {
            key.rsplit("input.", 1)[-1]: value for key, value in case["input"].items()
        }
        environment[SELECTOR] = bool(index)
        execution = sc._execute_formula_text(
            rule["versions"][0]["formula"],
            environment=environment,
            constant_environment={},
        )
        assert sc._formula_execution_runtime_value(execution) is (index == 0)
    issues = _paired_issues(payload, cases)
    assert any('The "threshold phaseout amount"' in issue for issue in issues)
    assert not any('The "completed phaseout amount"' in issue for issue in issues)


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


_SHARED_CREDIT_CITATION = "us/statute/26/32/a"
_SHARED_CREDIT_REFERENCE = "us:statutes/26/32/a#"
_SHARED_CREDIT_ALTERNATIVES = ("earned_income", "investment_income")
_SHARED_CREDIT_THRESHOLDS = ("first_threshold", "second_threshold")


def _shared_credit_candidate(
    clause_count=2,
    *,
    formula_specs=(("direct", False, False, True),) * 2,
    pair_specs=((True, "switch"),) * 2,
):
    """Make isolated pairs and an arithmetic oracle independent of the evaluator.

    A formula spec selects direct max or numeric guarded arms, max argument
    order, guard polarity, and whether the true arm compares alternative income.
    A pair spec chooses the intervened operand and whether the source result or
    only the candidate changes. Distinct years exclude unintended cross-pairs.
    """
    sentences = [
        f"The credit is {'also ' if index else ''}allowed for adjusted gross "
        f"income (or, if greater, {alternative.replace('_', ' ')}) above the "
        f"{_SHARED_CREDIT_THRESHOLDS[index].replace('_', ' ')}."
        for index, alternative in enumerate(_SHARED_CREDIT_ALTERNATIVES[:clause_count])
    ]
    terms = []
    for index, alternative in enumerate(_SHARED_CREDIT_ALTERNATIVES[:clause_count]):
        kind, reverse_arguments, reverse_guard, then_alternative = formula_specs[index]
        operands = ("adjusted_gross_income", alternative)
        if reverse_arguments:
            operands = operands[::-1]
        maximum = f"max({', '.join(operands)})"
        threshold = _SHARED_CREDIT_THRESHOLDS[index]
        if kind == "direct":
            terms.append(f"{maximum} > {threshold}")
        else:
            guard = f"{maximum} > adjusted_gross_income"
            if reverse_guard:
                guard = f"not ({guard})"
            true_operand = alternative if then_alternative else "adjusted_gross_income"
            false_operand = "adjusted_gross_income" if then_alternative else alternative
            terms.append(
                f"(({guard}) and {true_operand} > {threshold}) or "
                f"((not ({guard})) and {false_operand} > {threshold})"
            )
    formula = " or ".join(f"({term})" for term in terms)
    rule = {
        "name": "credit_allowed",
        "kind": "derived",
        "entity": "TaxUnit",
        "dtype": "Judgment",
        "period": "Year",
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": _SHARED_CREDIT_CITATION,
                            "excerpt": sentence,
                        },
                    }
                    for sentence in sentences
                ]
            }
        },
    }
    names = (
        "adjusted_gross_income",
        *_SHARED_CREDIT_ALTERNATIVES[:clause_count],
        *_SHARED_CREDIT_THRESHOLDS[:clause_count],
    )
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": _SHARED_CREDIT_CITATION}
        },
        "inputs": [
            {"name": name, "entity": "TaxUnit", "dtype": "Money", "period": "Year"}
            for name in names
        ],
        "rules": [rule],
    }

    def candidate_clause_result(values, index):
        kind, _, reverse_guard, then_alternative = formula_specs[index]
        ordinary = values["adjusted_gross_income"]
        alternative = values[_SHARED_CREDIT_ALTERNATIVES[index]]
        if kind == "direct":
            operand = max(ordinary, alternative)
        else:
            guard = max(ordinary, alternative) > ordinary
            if reverse_guard:
                guard = not guard
            chooses_alternative = then_alternative if guard else not then_alternative
            operand = alternative if chooses_alternative else ordinary
        return operand > values[_SHARED_CREDIT_THRESHOLDS[index]]

    cases = []
    expected_credit = []
    for index in range(clause_count):
        changed_alternative, profile = pair_specs[index]
        fixed = 11 if profile == "source_constant" else 9
        varying = (7, 8) if profile == "same_binding" else (fixed - 1, fixed + 2)
        source_results = []
        candidate_results = []
        for alternate, varying_value in enumerate(varying):
            values = dict.fromkeys(names, 8)
            values["adjusted_gross_income"] = fixed
            for threshold in _SHARED_CREDIT_THRESHOLDS[:clause_count]:
                values[threshold] = 100
            values[_SHARED_CREDIT_THRESHOLDS[index]] = 10
            changed = (
                _SHARED_CREDIT_ALTERNATIVES[index]
                if changed_alternative
                else "adjusted_gross_income"
            )
            if not changed_alternative:
                values[_SHARED_CREDIT_ALTERNATIVES[index]] = fixed
            values[changed] = varying_value
            source_result = (
                max(
                    values["adjusted_gross_income"],
                    values[_SHARED_CREDIT_ALTERNATIVES[index]],
                )
                > values[_SHARED_CREDIT_THRESHOLDS[index]]
            )
            candidate_result = any(
                candidate_clause_result(values, candidate_index)
                for candidate_index in range(clause_count)
            )
            source_results.append(source_result)
            candidate_results.append(candidate_result)
            year = 2026 + index
            cases.append(
                {
                    "name": f"clause_{index}_{alternate}",
                    "period": {
                        "period_kind": "tax_year",
                        "start": f"{year}-01-01",
                        "end": f"{year}-12-31",
                    },
                    "input": {
                        _SHARED_CREDIT_REFERENCE + "input." + name: value
                        for name, value in values.items()
                    },
                    "output": {
                        _SHARED_CREDIT_REFERENCE + "credit_allowed": (
                            "holds" if candidate_result else "not_holds"
                        )
                    },
                }
            )
        expected_credit.append(
            source_results[0] != source_results[1]
            and candidate_results == source_results
        )
    return payload, cases, "(1) " + " ".join(sentences), sentences, expected_credit


def test_two_max_clauses_can_share_one_output_with_their_own_pairs():
    payload, cases, source, _, expected_credit = _shared_credit_candidate()
    # Match the review's positive control exactly: both pairs are in 2026.
    for case in cases:
        case["period"]["start"] = "2026-01-01"
        case["period"]["end"] = "2026-12-31"
    assert expected_credit == [True, True]
    formula = payload["rules"][0]["versions"][0]["formula"]
    for index, case in enumerate(cases):
        values = {
            key.rsplit("input.", 1)[-1]: value for key, value in case["input"].items()
        }
        assert sc._evaluate_rulespec_formula(formula, environment=values) is bool(
            index % 2
        )
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload),
        require_policy_proofs=True,
        source_texts={_SHARED_CREDIT_CITATION: source},
    )
    assert proof.passed
    assert proof.atoms_checked == 2
    assert not proof.issues
    assert not _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )


def test_numeric_max_guard_credits_a_correct_pair_changing_ordinary_income():
    payload, cases, source, _, expected_credit = _shared_credit_candidate(
        1,
        formula_specs=(("arms", False, False, True),) * 2,
        pair_specs=((False, "switch"),) * 2,
    )
    assert expected_credit == [True]
    formula = payload["rules"][0]["versions"][0]["formula"]
    for index, case in enumerate(cases):
        values = {
            key.rsplit("input.", 1)[-1]: value for key, value in case["input"].items()
        }
        assert (
            max(values["adjusted_gross_income"], values["earned_income"]) > 10
        ) is bool(index)
        assert sc._evaluate_rulespec_formula(formula, environment=values) is bool(index)
    assert not _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )


def test_two_named_max_repairs_can_share_one_output_with_their_own_pairs():
    payload, cases, source, _, _ = _shared_credit_candidate()
    selectors = [
        alternative + "_is_greater_than_adjusted_gross_income"
        for alternative in _SHARED_CREDIT_ALTERNATIVES
    ]
    payload["inputs"].extend(
        {"name": selector, "entity": "TaxUnit", "dtype": "Judgment", "period": "Year"}
        for selector in selectors
    )
    terms = [
        f"(({selector}) and {alternative} > {threshold}) or "
        f"((not ({selector})) and adjusted_gross_income > {threshold})"
        for selector, alternative, threshold in zip(
            selectors,
            _SHARED_CREDIT_ALTERNATIVES,
            _SHARED_CREDIT_THRESHOLDS,
            strict=True,
        )
    ]
    formula = " or ".join(f"({term})" for term in terms)
    payload["rules"][0]["versions"][0]["formula"] = formula
    for index, case in enumerate(cases):
        values = case["input"]
        for alternative in _SHARED_CREDIT_ALTERNATIVES:
            values[_SHARED_CREDIT_REFERENCE + "input." + alternative] = 11
        for selector_index, selector in enumerate(selectors):
            values[_SHARED_CREDIT_REFERENCE + "input." + selector] = (
                "holds" if selector_index == index // 2 and index % 2 else "not_holds"
            )
        environment = {
            key.rsplit("input.", 1)[-1]: value for key, value in values.items()
        }
        for selector in selectors:
            environment[selector] = environment[selector] == "holds"
        assert sc._evaluate_rulespec_formula(formula, environment=environment) is bool(
            index % 2
        )
    assert not _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )


@settings(
    max_examples=30,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@example(
    clause_count=2,
    formula_specs=(("direct", False, False, True),) * 2,
    pair_specs=((True, "switch"),) * 2,
)
@example(
    clause_count=1,
    formula_specs=(("arms", False, False, False),) * 2,
    pair_specs=((True, "source_constant"),) * 2,
)
@example(
    clause_count=2,
    formula_specs=(("arms", False, False, True), ("arms", True, True, False)),
    pair_specs=((True, "switch"), (False, "switch")),
)
@example(
    clause_count=2,
    formula_specs=(("direct", False, False, True), ("arms", False, False, False)),
    pair_specs=((True, "switch"), (True, "source_constant")),
)
@given(
    clause_count=st.integers(min_value=1, max_value=2),
    formula_specs=st.tuples(
        *[
            st.tuples(
                st.sampled_from(("direct", "arms")),
                st.booleans(),
                st.booleans(),
                st.booleans(),
            )
        ]
        * 2
    ),
    pair_specs=st.tuples(
        *[
            st.tuples(
                st.booleans(),
                st.sampled_from(("switch", "source_constant", "same_binding")),
            )
        ]
        * 2
    ),
)
def test_max_clause_credit_matches_source_and_executed_arm_results(
    clause_count, formula_specs, pair_specs
):
    payload, cases, source, sentences, expected_credit = _shared_credit_candidate(
        clause_count, formula_specs=formula_specs, pair_specs=pair_specs
    )
    formula = payload["rules"][0]["versions"][0]["formula"]
    for case in cases:
        values = {
            key.rsplit("input.", 1)[-1]: value for key, value in case["input"].items()
        }
        expected = (
            case["output"][_SHARED_CREDIT_REFERENCE + "credit_allowed"] == "holds"
        )
        assert sc._evaluate_rulespec_formula(formula, environment=values) is expected
    issues = _paired_issues(
        payload, cases, source=source, citation=_SHARED_CREDIT_CITATION
    )
    for sentence, credited in zip(sentences, expected_credit, strict=True):
        assert (not any(sentence in issue for issue in issues)) is credited
