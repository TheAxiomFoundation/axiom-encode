"""The public completeness gate requires the source's operative care action."""

from functools import partial

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us/statute/care-credit"
CARE_INPUT = "taxpayer_care_expense"


def analyze_care_action(
    source_fact, input_description, *, uppercase=False, space=" ", input_name=CARE_INPUT
):
    source = "The credit applies if the taxpayer is resident and " + source_fact + "."
    source = source.replace(" ", space)
    if uppercase:
        source = source.upper()
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "taxpayer_is_resident",
                "description": "The taxpayer is resident.",
                "entity": "TaxUnit",
                "dtype": "Boolean",
                "period": "Year",
            },
            {
                "name": input_name,
                "description": input_description,
                "entity": "TaxUnit",
                "dtype": "Boolean",
                "period": "Year",
            },
        ],
        "rules": [
            {
                "name": "care_credit_applies",
                "kind": "derived",
                "entity": "TaxUnit",
                "dtype": "Judgment",
                "period": "Year",
                "source": CITATION,
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": (
                            "if taxpayer_is_resident and "
                            + input_name
                            + ": true else: false"
                        ),
                    }
                ],
                "metadata": {
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
                },
            }
        ],
    }
    cases = [
        {
            "name": name,
            "period": "2026",
            "input": {"taxpayer_is_resident": resident, input_name: care},
            "output": {"care_credit_applies": resident and care},
        }
        for name, resident, care in (
            ("both facts hold", True, True),
            ("residence absent", False, True),
            ("care fact absent", True, False),
        )
    ]
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_public_gate_rejects_care_expenses_without_required_billing_action():
    result = analyze_care_action(
        "dependent care expense billed to the taxpayer",
        "The taxpayer has care expenses.",
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


def test_public_gate_rejects_expense_presence_when_source_requires_absence():
    result = analyze_care_action(
        "has no dependent care expenses", "The taxpayer has care expenses."
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


def test_public_gate_accepts_explicit_expense_absence_evidence():
    result = analyze_care_action(
        "has no dependent care expenses",
        "The taxpayer has no dependent care expenses.",
        input_name="taxpayer_has_no_dependent_care_expenses",
    )
    assert result.issues == ()


@pytest.mark.parametrize(
    "description",
    [
        "The spouse has no dependent care expenses.",
        "The taxpayer may have no dependent care expenses.",
        "The taxpayer has no dependent care expenses or is disabled.",
    ],
)
def test_public_gate_requires_actual_expense_absence_for_the_source_taxpayer(
    description,
):
    result = analyze_care_action("has no dependent care expenses", description)
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@given(uppercase=st.booleans(), space=st.sampled_from([" ", "\t", "  "]))
@settings(deadline=None, max_examples=6)
def test_public_gate_preserves_the_no_care_expense_fact(uppercase, space):
    missing = analyze_care_action(
        "has no dependent care expenses",
        "The taxpayer has care expenses.",
        uppercase=uppercase,
        space=space,
    )
    assert any("source-explicit-conditions" in issue for issue in missing.issues)
    proved = analyze_care_action(
        "has no dependent care expenses",
        "The taxpayer has no dependent care expenses.",
        input_name="taxpayer_has_no_dependent_care_expenses",
        uppercase=uppercase,
        space=space,
    )
    assert proved.issues == ()


def test_public_gate_requires_an_operative_care_report():
    result = analyze_care_action(
        "the taxpayer reports that dependent care expense is billed to the taxpayer",
        "Dependent care expense billed to the taxpayer.",
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("source_fact", "description"),
    [
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense is billed or documented for the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense is billed or estimated for the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense may be billed to the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense could be billed to the taxpayer.",
        ),
        (
            "the taxpayer pays dependent care expenses",
            "The taxpayer can pay dependent care expenses.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense billed to the spouse.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense billed to the taxpayer's spouse.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense billed to the taxpayer’s spouse.",
        ),
    ],
)
def test_public_gate_rejects_unproved_or_wrong_actor_care_actions(
    source_fact, description
):
    result = analyze_care_action(source_fact, description)
    assert any("source-explicit-conditions" in issue for issue in result.issues)


def test_public_gate_accepts_explicit_action_actor_in_the_input_name():
    result = analyze_care_action(
        "dependent care expense billed to the taxpayer",
        "The taxpayer has dependent care expenses.",
        input_name="dependent_care_expense_billed_to_taxpayer",
    )
    assert result.issues == ()


def test_public_gate_rejects_a_care_action_with_out_of_window_modality():
    result = analyze_care_action(
        "dependent care expense billed to the taxpayer",
        "The taxpayer might "
        + "qualifier " * 20
        + "have dependent care expense billed to the taxpayer.",
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "description",
    [
        "The taxpayer has paid dependent care expenses.",
        "The taxpayer has incurred dependent care expenses.",
        "Dependent care expenses billed to or paid by the taxpayer.",
        "The taxpayer care expense is billed or is paid by the taxpayer.",
        "The taxpayer care expense is billed or alternatively paid by the taxpayer.",
    ],
)
def test_public_gate_rejects_other_actions_for_required_billing(description):
    result = analyze_care_action(
        "dependent care expense billed to the taxpayer", description
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("source_fact", "description"),
    [
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense not billed to the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense isn't billed to the taxpayer.",
        ),
        (
            "the taxpayer pays dependent care expenses",
            "The taxpayer has never paid dependent care expenses.",
        ),
        (
            "the taxpayer pays dependent care expenses",
            "The taxpayer doesn't pay dependent care expenses.",
        ),
        (
            "the taxpayer incurred dependent care expenses",
            "The taxpayer has not incurred dependent care expenses.",
        ),
        (
            "the taxpayer incurred dependent care expenses",
            "The taxpayer hasn't incurred dependent care expenses.",
        ),
        (
            "dependent care expense not billed to the taxpayer",
            "Dependent care expense billed to the taxpayer.",
        ),
    ],
)
def test_public_gate_rejects_care_action_polarity_mismatch(source_fact, description):
    result = analyze_care_action(source_fact, description)
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("input_name", "description"),
    [
        (
            "taxpayer_care_expense_billed",
            "Dependent care expense not billed to the taxpayer.",
        ),
        (
            "taxpayer_care_expense_not_billed",
            "Dependent care expense billed to the taxpayer.",
        ),
        (
            CARE_INPUT,
            "Dependent care expense billed to or not paid by the taxpayer.",
        ),
        (
            CARE_INPUT,
            "Dependent care expense billed to or the taxpayer has not paid rent.",
        ),
        (
            CARE_INPUT,
            "The taxpayer care expense is billed to the taxpayer, or paid by the taxpayer.",
        ),
    ],
)
def test_public_gate_rejects_contradictory_or_unbound_action_evidence(
    input_name, description
):
    result = analyze_care_action(
        "dependent care expense billed to the taxpayer",
        description,
        input_name=input_name,
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    "description",
    [
        "The taxpayer pays rent. The taxpayer has dependent care expenses.",
        "The taxpayer pays rent and has dependent care expenses.",
    ],
)
def test_public_gate_rejects_payment_on_unrelated_expenses(description):
    result = analyze_care_action(
        "the taxpayer pays dependent care expenses", description
    )
    assert any("source-explicit-conditions" in issue for issue in result.issues)


@pytest.mark.parametrize(
    ("source_fact", "description"),
    [
        (
            "dependent care expense billed to the taxpayer",
            "Dependent care expense billed to the taxpayer.",
        ),
        (
            "the taxpayer pays dependent care expenses",
            "The taxpayer has paid dependent care expenses.",
        ),
        (
            "the taxpayer incurred dependent care expenses",
            "The taxpayer incurs dependent care expenses.",
        ),
        (
            "dependent care expense billed to or paid by the taxpayer",
            "Dependent care expense billed to the taxpayer.",
        ),
        (
            "dependent care expense billed to or paid by the taxpayer",
            "The taxpayer has paid dependent care expenses.",
        ),
        (
            "dependent care expense billed-or-paid by the taxpayer",
            "The taxpayer has paid dependent care expenses.",
        ),
        (
            "dependent care expense billed to the taxpayer, or paid by the taxpayer",
            "Dependent care expense billed to the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer, or paid by the taxpayer",
            "The taxpayer has paid dependent care expenses.",
        ),
        (
            "dependent care expense is billed or is paid by the taxpayer",
            "The taxpayer has paid dependent care expenses.",
        ),
        (
            "dependent care expense billed to or paid by the taxpayer",
            "The taxpayer care expense is billed to the taxpayer, or paid by the taxpayer.",
        ),
        (
            "dependent care expense not billed to the taxpayer",
            "Dependent care expense not billed to the taxpayer.",
        ),
        (
            "the taxpayer is unable to provide dependent care",
            "The taxpayer cannot provide dependent care.",
        ),
        (
            "the taxpayer reports that dependent care expense is billed to the taxpayer",
            "The taxpayer reports that dependent care expense is billed to the taxpayer.",
        ),
        (
            "dependent care expense billed to the taxpayer's spouse",
            "Dependent care expense billed to the spouse.",
        ),
        (
            "dependent care expense billed to the spouse",
            "Dependent care expense billed to the taxpayer’s spouse.",
        ),
    ],
)
def test_public_gate_accepts_action_bound_care_evidence(source_fact, description):
    assert analyze_care_action(source_fact, description).issues == ()


@given(
    fact=st.sampled_from(
        [
            (
                "dependent care expense billed to the taxpayer",
                "Dependent care expense billed to the taxpayer.",
            ),
            (
                "the taxpayer pays dependent care expenses",
                "The taxpayer has paid dependent care expenses.",
            ),
            (
                "the taxpayer incurred dependent care expenses",
                "The taxpayer incurs dependent care expenses.",
            ),
            (
                "dependent care expense billed to or paid by the taxpayer",
                "The taxpayer has paid dependent care expenses.",
            ),
        ]
    ),
    uppercase=st.booleans(),
    space=st.sampled_from([" ", "\t", "  "]),
)
@settings(deadline=None, max_examples=24)
def test_public_care_actions_require_action_evidence(fact, uppercase, space):
    source_fact, description = fact
    missing = analyze_care_action(
        source_fact, "The taxpayer has care expenses.", uppercase=uppercase, space=space
    )
    assert any("source-explicit-conditions" in issue for issue in missing.issues)
    proved = analyze_care_action(
        source_fact, description, uppercase=uppercase, space=space
    )
    assert proved.issues == ()
