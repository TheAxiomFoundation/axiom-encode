"""A companion row assertion must not corroborate another entity's facts."""

from __future__ import annotations

import copy
import functools
from decimal import Decimal

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us/guidance/example/benefit"
EXTRACT = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)
GROUND = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)


def _rule(name, formula, *, dtype="Money", source=None):
    rule = {
        "name": name,
        "kind": "derived",
        "dtype": dtype,
        "entity": "Person",
        "period": "Month",
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }
    if source is not None:
        rule["metadata"] = {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": source,
                        },
                    }
                ]
            }
        }
    return rule


def _analyze(payload, source, cases):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_numeric_grounding_occurrences=GROUND,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _rounding_fixture():
    source = (
        "The benefit equals 50 percent of income, rounded to the nearest whole dollar."
    )
    rate = _rule("rate", "0.5", dtype="Rate")
    rate["kind"] = "parameter"
    benefit = _rule("benefit", "income * rate", source=source)
    benefit.update(unit="USD", rounding="half_up")
    return source, {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "units": [{"name": "USD", "kind": "currency", "minor_units": 0}],
        "inputs": [
            {"name": "income", "entity": "Person", "dtype": "Money", "period": "Month"}
        ],
        "rules": [rate, benefit],
    }


@pytest.mark.parametrize(
    "income,row_income,expected", [(200.98, 200, 100), (201, 202, 101)]
)
def test_review_rounding_cases_do_not_exercise_fractional_queried_values(
    income, row_income, expected
):
    source, payload = _rounding_fixture()
    case = {
        "name": "wrong_row_rounding",
        "period": "2026-01",
        "input": {"income": income},
        "tables": {"Person": [{"income": row_income}]},
        "output": {"benefit": [expected]},
    }
    rules = {r["name"]: r for r in sc._bind_currency_rounding_rules(payload)["rules"]}
    queried = {**case, "input": case["tables"]["Person"][0]}
    execution = sc._case_formula_execution(
        rules["benefit"], queried, formula_environment={"rate": Decimal("0.5")}
    )
    assert execution.unrounded_value == Decimal(row_income) / 2
    assert execution.unrounded_value == execution.unrounded_value.to_integral_value()
    assert not _rounding_evidence(rules, case)


def _rounding_evidence(rules, case):
    return sc._fractional_rounding_case_witnesses(
        "benefit",
        rules["benefit"],
        principal_rules={"benefit": rules["benefit"]},
        parameter_rules={"rate": rules["rate"]},
        asserted_by_rule={"benefit": [case]},
        direction="nearest",
        formula_environment={"rate": Decimal("0.5")},
        source_formula_branches=(),
        rounding_refers_to_result=False,
        require_clause_binding=False,
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_review_rounding_pair_still_reports_missing_fractional_evidence():
    source, payload = _rounding_fixture()
    cases = [
        {
            "name": "wrong_row_rounding_" + str(index),
            "period": "2026-01",
            "input": {"income": income},
            "tables": {"Person": [{"income": row_income}]},
            "output": {"benefit": [expected]},
        }
        for index, (income, row_income, expected) in enumerate(
            [(200.98, 200, 100), (201, 202, 101)]
        )
    ]
    assert any("rounding" in issue for issue in _analyze(payload, source, cases).issues)


def _dependency_fixture(value=2, row_value=0):
    rules = {"benefit": _rule("benefit", "income * 2")}
    case = {
        "name": "other_entity_facts",
        "period": "2026-01",
        "input": {"income": value},
        "tables": {
            "Person": [{"income": row_value}],
            "OtherPerson": [{"income": value}],
        },
        "output": {"benefit": [value * 2]},
    }
    return rules, case


def test_asserted_dependency_environment_does_not_import_other_row_output():
    rules, case = _dependency_fixture()
    control = copy.deepcopy(case)
    control["output"]["benefit"] = 4
    assert sc._case_asserted_dependency_environment(
        rules, control, formula_environment={}
    )
    assert not sc._case_asserted_dependency_environment(
        rules, case, formula_environment={}
    )


def test_closed_fractional_operand_requires_scalar_intermediate_corroboration():
    rules = {
        "raw": _rule("raw", "0.5"),
        "benefit": _rule("benefit", "floor(raw + 0.5)"),
    }
    case = {
        "period": "2026-01",
        "input": {"income": 1},
        "tables": {"Person": [{"income": 0}], "OtherPerson": [{"income": 1}]},
        "output": {"raw": [Decimal("0.5")], "benefit": [1]},
    }
    environment = {"raw": Decimal("0.5"), "benefit": Decimal(1)}
    execution = sc._case_formula_execution(
        rules["benefit"], case, dependency_environment=environment
    )

    def accepted(case):
        return sc._closed_fractional_rounding_operand_witness(
            "raw",
            case=case,
            parameter_rules={},
            principal_rules=rules,
            dependency_environment=environment,
            operand_value=Decimal("0.5"),
            rule_name="benefit",
            execution=execution,
        )

    control = copy.deepcopy(case)
    control["output"] = {key: value[0] for key, value in control["output"].items()}
    assert accepted(control)
    assert not accepted(case)


def _annual_fixture(value=2, row_value=0):
    header = "Use this form to calculate the foreign tax credit for 2026."
    text = "The credit equals total taxes paid for 2026 divided by income."
    source = header + "\n" + text
    rule = _rule("benefit", "taxes / income", source=text)
    rule["versions"][0]["effective_to"] = "2026-12-31"
    rule["metadata"]["proof"]["atoms"].extend(
        {
            "path": "versions[0]." + field,
            "kind": "definition",
            "source": {"corpus_citation_path": CITATION, "excerpt": header},
        }
        for field in ("effective_from", "effective_to")
    )
    case = {
        "period": "2026-01",
        "input": {"income": 1, "taxes": value},
        "tables": {
            "Person": [{"income": 1, "taxes": row_value}],
            "OtherPerson": [{"income": 1, "taxes": value}],
        },
        "output": {"benefit": [value]},
    }
    branch = sc.SourceStructureBranch(
        path=(),
        kind="formula",
        label="",
        text=text,
        start=len(header) + 1,
        end=len(source),
    )
    return rule, case, branch, source


def _annual_evidence(rule, case, branch, source):
    return sc._annual_source_evidence(
        rule,
        case=case,
        execution=sc._case_formula_execution(rule, case),
        branch=branch,
        source_text=source,
        corpus_citation_path=CITATION,
    )


def test_annual_year_evidence_cannot_borrow_other_entity_execution():
    rule, case, branch, source = _annual_fixture()
    control = copy.deepcopy(case)
    control["output"]["benefit"] = 2
    assert _annual_evidence(rule, control, branch, source)
    assert not _annual_evidence(rule, case, branch, source)


def _boundary_fixture(limit=100):
    text = f"B.(1) The credit applies when income is at most {limit} dollars."
    branch = sc.SourceStructureBranch(
        ("b", "1"), "paragraph", "B.(1)", text, 0, len(text)
    )
    boundary = next(
        occurrence for occurrence in EXTRACT(text) if occurrence.value == limit
    )
    rules = {
        "low_income_applies": _rule(
            "low_income_applies", "income <= income_limit", dtype="Judgment"
        ),
        "credit_amount": _rule(
            "credit_amount", "if low_income_applies: base_credit else: 0"
        ),
    }
    case = {
        "name": "other_entity_boundary",
        "period": "2026-01",
        "input": {"income": limit, "base_credit": 50},
        "tables": {
            "Person": [{"income": limit + 1, "base_credit": 50}],
            "OtherPerson": [{"income": limit, "base_credit": 50}],
        },
        "output": {"low_income_applies": ["holds"], "credit_amount": [50]},
    }
    return branch, boundary, rules, case, limit


def _boundary_evidence(branch, boundary, rules, case, limit):
    return sc._branch_boundary_test_witnesses(
        branch,
        boundary,
        principal_rules=rules,
        principal_rule_paths={
            "low_income_applies": {("a", "1")},
            "credit_amount": {("b", "1")},
        },
        asserted_by_rule={name: [case] for name in rules},
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={"income_limit": limit},
        extract_numeric_occurrences=EXTRACT,
    )


def test_numeric_boundary_grounding_cannot_reach_other_entity_helper():
    branch, boundary, rules, case, limit = _boundary_fixture()
    control = copy.deepcopy(case)
    control["output"] = {key: value[0] for key, value in control["output"].items()}
    assert _boundary_evidence(branch, boundary, rules, control, limit)
    assert not _boundary_evidence(branch, boundary, rules, case, limit)


def _formula_fixture(value=2, row_value=0):
    source = "The benefit equals 50 percent of subtotal."
    rules = {
        "subtotal": _rule("subtotal", "income * 2"),
        "benefit": _rule("benefit", "subtotal * rate", source=source),
    }
    case = {
        "period": "2026-01",
        "input": {"income": value},
        "tables": {
            "Person": [{"income": row_value}],
            "OtherPerson": [{"income": value}],
        },
        "output": {"subtotal": [value * 2], "benefit": [value]},
    }
    branch = sc.SourceStructureBranch((), "formula", "", source, 0, len(source))
    return rules, case, branch, source


def _formula_evidence(rules, case, branch, source):
    return sc._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules=rules,
        rule_names={"benefit"},
        asserted_by_rule={"benefit": [case]},
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={"rate": Decimal("0.5")},
        dependency_cache={},
        execution_cache={},
        source_text=source,
    )


def test_formula_witness_cannot_reach_other_entity_helper():
    rules, case, branch, source = _formula_fixture()

    control = copy.deepcopy(case)
    control["output"] = {key: value[0] for key, value in control["output"].items()}
    assert _formula_evidence(rules, control, branch, source)
    assert not _formula_evidence(rules, case, branch, source)


def test_negative_clamp_witness_cannot_borrow_other_entity_negative_income():
    from tests.test_source_completeness import _al_direct_input_clamp_analysis

    case = {
        "name": "other_entity_negative_income",
        "period": "2026",
        "input": {"completed_taxable_income": -1},
        "output": {"taxable_income_boundary": 0, "schedule_before_credits": 0},
    }
    assert not any(
        "Direct nonnegative clamp" in issue
        for issue in _al_direct_input_clamp_analysis([case]).issues
    )
    case["tables"] = {
        "TaxUnit": [{"completed_taxable_income": 0}],
        "OtherTaxUnit": [copy.deepcopy(case["input"])],
    }
    case["output"] = {key: [value] for key, value in case["output"].items()}
    assert any(
        "Direct nonnegative clamp" in issue
        for issue in _al_direct_input_clamp_analysis([case]).issues
    )


def _different_entity_assertions(cases):
    cases = copy.deepcopy(cases)
    for case in cases:
        case["tables"] = {"Person": [{}], "OtherPerson": [copy.deepcopy(case["input"])]}
        case["output"] = {
            key: [value] if not isinstance(value, list) else value
            for key, value in case["output"].items()
        }
    return cases


def test_worksheet_operations_cannot_certify_other_entity_assertions():
    from axiom_encode.harness import worksheet_operation_evidence as operations
    from tests.test_worksheet_operation_evidence import certify, inputs

    payload, cases = inputs()
    assert not certify(payload, cases).unresolved
    wrong = copy.deepcopy(cases)
    first = wrong[0]
    key = next(k for k in first["output"] if k.endswith("#" + operations._LINE2))
    first["output"][key] = [first["output"][key]]
    row = copy.deepcopy(first["input"])
    credit = next(k for k in row if k.endswith("#input." + operations._CREDIT))
    row[credit] += 1
    first["tables"] = {"Person": [row]}
    result = certify(payload, wrong)
    assert result.unresolved
    assert not result.case_links


def test_worksheet_transfers_cannot_certify_other_entity_assertions():
    from axiom_encode.harness import worksheet_transfer_evidence as transfer
    from tests.test_worksheet_transfer_evidence import certify, inputs

    payload, cases = inputs()
    assert not certify(payload, cases).unresolved
    wrong = copy.deepcopy(cases)
    parent = wrong[-1]
    key = next(k for k in parent["output"] if k.endswith("#" + transfer.TOTAL))
    parent["output"][key] = [parent["output"][key]]
    row = {
        key: value
        for key, value in parent["input"].items()
        if not isinstance(value, (dict, list))
    }
    count = next(k for k in row if k.endswith("#input." + transfer.PAYABLE))
    row[count] += 1  # Queried scalar facts differ from the flat relation replay.
    entity = next(r["entity"] for r in payload["rules"] if r["name"] == transfer.TOTAL)
    parent["tables"] = {entity: [row]}
    result = certify(payload, wrong)
    assert result.unresolved
    assert not result.cases


def test_worksheet_zero_and_cap_evidence_cannot_borrow_other_entity_assertions():
    from axiom_encode.harness import worksheet_operation_evidence as operations
    from tests.test_worksheet_operation_evidence import certify, inputs

    payload, cases = inputs()
    certificate = certify(payload, cases)
    assert not certificate.unresolved
    assert not operations._zero_and_computation_witnesses(cases, certificate.case_links)
    wrong = copy.deepcopy(cases)
    for case in wrong:
        for key, value in list(case["output"].items()):
            if key.endswith("#" + operations._LINE5):
                case["output"][key] = [value]
                row = copy.deepcopy(case["input"])
                residence = next(
                    k for k in row if k.endswith("#input." + operations._RESIDENCE)
                )
                row[residence] = "Quebec" if row[residence] != "Quebec" else "Ontario"
                case["tables"] = {"Person": [row]}
    # Reuse genuine links to isolate the downstream reader. The public check
    # now refuses these assertions before creating links; head normalized them.
    missing = operations._zero_and_computation_witnesses(wrong, certificate.case_links)
    assert {
        "missing source zero pair: " + reason
        for reason in ("quebec", "manitoba", "equality")
    } <= set(missing)
    assert "no actual active cap execution" in missing


def test_worksheet_country_rejects_row_cases_before_reading_assertions():
    from tests.test_worksheet_country_evidence import certify, fixture

    payload, cases = fixture()
    assert not certify(payload, cases).unresolved
    # This closed scalar schema already rejects the tables key. Its reader is
    # unreachable for a row case even before singleton normalization is fixed.
    result = certify(payload, _different_entity_assertions(cases))
    assert result.unresolved
    assert not result.pairs


def test_shadow_links_do_not_corroborate_other_entity_assertions():
    from axiom_encode.harness import source_path_shadow as shadow
    from tests.test_source_path_shadow import FIXTURE, L2, both_owners

    payload = yaml.safe_load((FIXTURE / "candidate.yaml").read_text())
    cases = yaml.safe_load((FIXTURE / "cases.yaml").read_text())
    result = both_owners(payload)[L2]
    links = shadow.corroborated_case_links(payload, result, cases)
    assert links and all(link["corroborated"] for link in links)
    wrong = shadow.corroborated_case_links(
        payload, result, _different_entity_assertions(cases)
    )
    assert not any(link["corroborated"] for link in wrong)


@pytest.mark.parametrize(
    "consumer",
    [
        "dependency",
        "annual",
        "boundary",
        "formula",
        "rounding",
        "bound_reader",
        "raw_reader",
    ],
)
@settings(max_examples=30, deadline=None)
@given(
    value=st.integers(min_value=-100_000, max_value=100_000),
    difference=st.integers(min_value=1, max_value=1000),
)
def test_wrong_row_facts_never_supply_scalar_evidence(consumer, value, difference):
    """Vary both facts and mismatches across bound and legacy consumer paths."""
    rules, case = _dependency_fixture(value, value + difference)
    untouched = copy.deepcopy(case)
    if consumer == "dependency":
        assert not sc._case_asserted_dependency_environment(
            rules, case, formula_environment={}
        )
    elif consumer == "annual":
        rule, annual_case, branch, source = _annual_fixture(value, value + difference)
        assert not _annual_evidence(rule, annual_case, branch, source)
    elif consumer == "boundary":
        args = _boundary_fixture(abs(value) + 1)
        assert not _boundary_evidence(*args)
    elif consumer == "formula":
        assert not _formula_evidence(*_formula_fixture(value, value + difference))
    elif consumer == "rounding":
        _, payload = _rounding_fixture()
        rounding_rules = {
            rule["name"]: rule
            for rule in sc._bind_currency_rounding_rules(payload)["rules"]
        }
        income = Decimal(2 * value) + Decimal("0.98")
        rounding_case = {
            "period": "2026-01",
            "input": {"income": income},
            "tables": {"Person": [{"income": income + difference}]},
            "output": {"benefit": [value]},
        }
        assert not _rounding_evidence(rounding_rules, rounding_case)
    elif consumer == "bound_reader":
        assert (
            sc._test_case_asserted_output_value(case, "benefit", principal_rules=rules)
            is sc._UNRESOLVED_CONDITION_VALUE
        )
    else:
        assert sc._test_case_asserted_output_value(case, "benefit") == [value * 2]
    assert case == untouched
