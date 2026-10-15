"""One-row entity assertions must retain their exception-pair evidence."""

from __future__ import annotations

import copy
import dataclasses
import functools

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "nz/statute/act/synthetic/section/1"
REFERENCE = "nz:statutes/act/synthetic/section/1#"
OUTPUT = "payment_is_payable"
SOURCE = "(1) The payment is payable unless an exemption applies."
EXTRACT = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)


def _payload():
    # Synthetic harness input, reduced from the s75 Payment/table assertion shape.
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "exemption_applies",
                "entity": "Payment",
                "dtype": "Boolean",
                "period": "Day",
            }
        ],
        "rules": [
            {
                "name": OUTPUT,
                "kind": "derived",
                "entity": "Payment",
                "dtype": "Judgment",
                "period": "Day",
                "source": CITATION + "(1)",
                "versions": [{"formula": "not exemption_applies"}],
            }
        ],
    }


def _pair():
    cases = []
    for exempt in (False, True):
        inputs = {REFERENCE + "input.exemption_applies": exempt}
        cases.append(
            {
                "name": "blocking" if exempt else "ordinary",
                "period": {
                    "period_kind": "custom",
                    "name": "day",
                    "start": "2026-01-01",
                    "end": "2026-01-01",
                },
                "input": inputs,
                "tables": {"Payment": [copy.deepcopy(inputs)]},
                "output": {REFERENCE + OUTPUT: "not_holds" if exempt else "holds"},
            }
        )
    return cases


def _paired_issues(cases):
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(_payload()),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [
        issue for issue in result.issues if "require paired positive/blocking" in issue
    ]


def _witnesses(rules, cases):
    principal = {rule["name"]: rule for rule in rules}
    witnesses = sc._toggled_formula_boolean_selectors(
        principal,
        asserted_by_rule={
            name: [case for case in cases if name in sc._test_case_output_names(case)]
            for name in principal
        },
        formula_environment={"income_limit": 10},
    )
    # Production identities deliberately use object ids. Compare the same case
    # indices across independent copies without losing case-pair provenance.
    indices = {id(case): index for index, case in enumerate(cases)}
    return {
        dataclasses.replace(
            witness,
            case_pair_identity=tuple(
                sorted(indices[identity] for identity in witness.case_pair_identity)
            ),
        )
        for witness in witnesses
    }


@pytest.mark.parametrize("wrapped_indices", [(0, 1), (0,), (1,)])
def test_payment_table_one_row_lists_satisfy_required_exception_pair(wrapped_indices):
    scalar = _pair()
    assert _paired_issues(scalar[:1])
    assert not _paired_issues(scalar)
    listed = copy.deepcopy(scalar)
    for index in wrapped_indices:
        listed[index]["output"][REFERENCE + OUTPUT] = [
            listed[index]["output"][REFERENCE + OUTPUT]
        ]
    untouched = copy.deepcopy(listed)

    assert not _paired_issues(listed)
    scalar_witnesses = _witnesses(_payload()["rules"], scalar)
    assert len(scalar_witnesses) == 2
    assert _witnesses(_payload()["rules"], listed) == scalar_witnesses
    assert listed == untouched


@pytest.mark.parametrize("values", [[], ["holds", "not_holds"]])
def test_non_singleton_outputs_report_case_output_and_row_order(values):
    cases = _pair()
    cases[0]["output"][REFERENCE + OUTPUT] = values
    cases[0]["tables"]["Payment"] = [
        {REFERENCE + "input.exemption_applies": value == "not_holds"}
        for value in values
    ]

    assert not _witnesses(_payload()["rules"], cases)
    issues = _paired_issues(cases)
    assert issues
    feedback = " ".join(issues)
    assert "ordinary" in feedback
    assert REFERENCE + OUTPUT in feedback
    assert str(len(values)) in feedback
    assert "row-ordered" in feedback.lower()
    assert "one-row" in feedback


def test_nested_singleton_lists_do_not_supply_scalar_evidence():
    cases = _pair()
    cases[0]["output"][REFERENCE + OUTPUT] = [["holds"]]

    assert not _witnesses(_payload()["rules"], cases)
    assert _paired_issues(cases)


def test_non_singleton_extra_case_does_not_hide_an_existing_valid_pair():
    cases = _pair()
    extra = copy.deepcopy(cases[0])
    extra["name"] = "unpaired_rows"
    extra["tables"]["Payment"].extend(copy.deepcopy(cases[1]["tables"]["Payment"]))
    extra["output"][REFERENCE + OUTPUT] = ["holds", "not_holds"]

    assert not _paired_issues([*cases, extra])


def _generated_case_set(family, specs, legal_names):
    numeric = family.startswith("numeric")
    dependency = family.endswith("dependency")
    input_name = "income" if numeric else "exemption_applies"
    condition = "income <= income_limit" if numeric else "not exemption_applies"
    judgment = "low_income_applies" if numeric else "payment_is_payable"
    rules = [
        {
            "name": judgment,
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [{"formula": condition}],
        }
    ]
    if dependency:
        rules.append(
            {
                "name": "payment_amount",
                "kind": "derived",
                "dtype": "Money",
                "versions": [{"formula": f"if {judgment}: 10 else: 0"}],
            }
        )
    elif family == "boolean_money":
        rules = [
            {
                "name": "payment_amount",
                "kind": "derived",
                "dtype": "Money",
                "versions": [{"formula": "if exemption_applies: 0 else: 10"}],
            }
        ]
    cases = []
    for index, (value, period, wrong, omitted) in enumerate(specs):
        input_value = value if numeric else bool(value)
        holds = input_value <= 10 if numeric else not input_value
        expected_holds = not holds if wrong else holds
        inputs = {
            (REFERENCE + "input." if legal_names else "") + input_name: input_value
        }
        outputs = {}
        for rule in rules:
            name = rule["name"]
            if omitted and name == rules[-1]["name"]:
                continue
            outputs[(REFERENCE if legal_names else "") + name] = (
                ("holds" if expected_holds else "not_holds")
                if rule["dtype"] == "Judgment"
                else (10 if expected_holds else 0)
            )
        cases.append(
            {
                "name": f"case_{index}",
                "period": f"202{period + 5}",
                "input": inputs,
                "tables": {"Payment": [copy.deepcopy(inputs)]},
                "output": outputs,
            }
        )
    return rules, cases


@settings(max_examples=80, deadline=None)
@given(
    family=st.sampled_from(
        [
            "boolean",
            "boolean_money",
            "numeric",
            "boolean_dependency",
            "numeric_dependency",
        ]
    ),
    specs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=20),
            st.integers(min_value=0, max_value=1),
            st.booleans(),
            st.booleans(),
        ),
        max_size=8,
    ),
    legal_names=st.booleans(),
)
@example(
    family="boolean",
    specs=[(0, 0, False, False), (1, 0, False, False)],
    legal_names=True,
)
@example(
    family="boolean_money",
    specs=[(0, 0, False, False), (1, 0, False, False)],
    legal_names=False,
)
@example(
    family="numeric_dependency",
    specs=[(0, 0, False, False), (20, 0, False, False)],
    legal_names=True,
)
def test_wrapping_or_unwrapping_any_companion_case_set_preserves_witnesses(
    family, specs, legal_names
):
    rules, scalar = _generated_case_set(family, specs, legal_names)
    listed = copy.deepcopy(scalar)
    for case in listed:
        case["output"] = {name: [value] for name, value in case["output"].items()}
    untouched = copy.deepcopy(listed)
    expected = _witnesses(rules, scalar)

    assert _witnesses(rules, listed) == expected
    assert listed == untouched
    unwrapped = copy.deepcopy(listed)
    for case in unwrapped:
        case["output"] = {name: value[0] for name, value in case["output"].items()}
    assert _witnesses(rules, unwrapped) == expected
