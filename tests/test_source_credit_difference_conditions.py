"""Credit subtraction operands are computations, not claimant fact gates."""

from functools import partial

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CLAUSE = (
    "If the resident taxpayer has a federal tax liability, the taxpayer shall "
    "claim from the state not more than the difference between the amount of "
    "the state child care credit for which the taxpayer is eligible and the "
    "federal credit for child and dependent care expenses the taxpayer is able "
    "to deduct from federal tax liability for the same taxable year;"
)
CITATION = "us-nm/statute/7-2-18.1"


def _public_gate(payload, source, cases):
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
    ).issues


def _credit_candidate(
    *,
    state_name="state_credit",
    federal_name="federal_credit",
    state_description="The amount of the state child care credit for which the taxpayer is eligible.",
    federal_description="The federal credit for child and dependent care expenses deductible from federal tax liability.",
    source=CLAUSE,
):
    rule = {
        "name": "child_care_credit",
        "kind": "derived",
        "entity": "TaxUnit",
        "dtype": "Money",
        "unit": "USD",
        "period": "Year",
        "source": CITATION + "(e)",
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": f"if resident_has_federal_tax_liability: {state_name} - {federal_name} else: {state_name}",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": source},
                    }
                ]
            }
        },
    }
    payload = {
        "format": "rulespec/v1",
        "module": {"name": "child_care", "version": "1.0.0"},
        "inputs": [
            {
                "name": "resident_has_federal_tax_liability",
                "description": "The resident taxpayer has federal tax liability.",
                "dtype": "Boolean",
            },
            {"name": state_name, "description": state_description, "dtype": "Money"},
            {
                "name": federal_name,
                "description": federal_description,
                "dtype": "Money",
            },
        ],
        "rules": [rule],
    }
    cases = [
        {
            "name": f"credit_{index}",
            "period": "2026-01-01",
            "input": {
                "resident_has_federal_tax_liability": liability,
                state_name: state,
                federal_name: federal,
            },
            "output": {"child_care_credit": state - federal if liability else state},
        }
        for index, (liability, state, federal) in enumerate(
            [(True, 1000, 200), (False, 1000, 200), (True, 1200, 100)]
        )
    ]
    return payload, cases


def _credit_source_variant(source_case, whitespace=" ", *, source=CLAUSE):
    source = {
        "original": source,
        "upper": source.upper(),
        "mixed": source.title(),
    }[source_case]
    return source.replace(" ", whitespace)


@pytest.mark.parametrize(
    "state_description,federal_description",
    [
        ("An unrelated amount.", "Another unrelated amount."),
        (
            "The state child care credit amount.",
            "The federal child and dependent care credit amount.",
        ),
        (
            "The state child care credit for which the taxpayer is eligible.",
            "An unrelated amount.",
        ),
        (
            "An unrelated amount.",
            "The federal child and dependent care credit deductible from federal tax liability.",
        ),
        (
            "The state child care credit for which the taxpayer is never eligible.",
            "The federal child and dependent care credit deductible from federal tax liability.",
        ),
        (
            "The state child care credit for which the taxpayer is eligible.",
            "The federal child and dependent care credit never deductible from federal tax liability.",
        ),
        (
            "An unrelated amount; the taxpayer is eligible for the state child care credit.",
            "An unrelated amount; the federal child and dependent care credit is deductible from federal tax liability.",
        ),
        (
            "Whether the taxpayer is eligible for the state child care credit.",
            "Whether the federal child and dependent care credit is deductible from federal tax liability.",
        ),
        (
            "The state income eligible for the child care credit.",
            "The federal tax liability before the deductible child care credit.",
        ),
    ],
)
def test_public_credit_difference_rejects_unbound_operands(
    state_description, federal_description
):
    payload, cases = _credit_candidate(
        state_name="unrelated_a",
        federal_name="unrelated_b",
        state_description=state_description,
        federal_description=federal_description,
    )
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_public_credit_difference_accepts_bound_operands():
    payload, cases = _credit_candidate()
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()


@pytest.mark.parametrize("source_case", ["upper", "mixed"])
def test_public_credit_difference_rejects_wrong_operands_regardless_of_source_case(
    source_case,
):
    source = _credit_source_variant(source_case)
    payload, cases = _credit_candidate(
        state_name="unrelated_a",
        federal_name="unrelated_b",
        state_description="An unrelated amount.",
        federal_description="Another unrelated amount.",
        source=source,
    )
    issues = _public_gate(payload, "E. " + source, cases)
    assert issues, issues


@pytest.mark.parametrize("source_case", ["upper", "mixed"])
def test_public_credit_difference_accepts_bound_operands_regardless_of_source_case(
    source_case,
):
    source = _credit_source_variant(source_case)
    payload, cases = _credit_candidate(source=source)
    issues = _public_gate(payload, "E. " + source, cases)
    assert issues == (), issues


def test_public_credit_difference_rejects_unrelated_addition_to_bound_difference():
    payload, cases = _credit_candidate()
    payload["inputs"].append(
        {
            "name": "unrelated_bonus",
            "description": "An unrelated amount.",
            "dtype": "Money",
        }
    )
    payload["rules"][0]["versions"][0]["formula"] = (
        "if resident_has_federal_tax_liability: "
        "(state_credit - federal_credit) + unrelated_bonus else: state_credit"
    )
    for case in cases:
        case["input"]["unrelated_bonus"] = 50
        if case["input"]["resident_has_federal_tax_liability"]:
            case["output"]["child_care_credit"] += 50
    assert _public_gate(payload, "E. " + CLAUSE, cases)


@pytest.mark.parametrize(
    "mutation", ["missing_state", "missing_federal", "unused_bound_inputs", "reversed"]
)
def test_public_credit_difference_requires_the_reached_ordered_operands(mutation):
    payload, cases = _credit_candidate()
    if mutation.startswith("missing_"):
        name = mutation.removeprefix("missing_") + "_credit"
        payload["inputs"] = [item for item in payload["inputs"] if item["name"] != name]
    elif mutation == "unused_bound_inputs":
        for item in payload["inputs"][1:]:
            payload["inputs"].append({**item, "name": item["name"] + "_unused"})
            item["description"] = "An unrelated amount."
    else:
        payload["rules"][0]["versions"][0]["formula"] = (
            "if resident_has_federal_tax_liability: federal_credit - state_credit else: state_credit"
        )
        for case in cases:
            if case["input"]["resident_has_federal_tax_liability"]:
                case["output"]["child_care_credit"] *= -1
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def _derived_credit_candidate(binding):
    payload, cases = _credit_candidate()
    state_excerpt = (
        "the amount of the state child care credit for which the taxpayer is eligible"
    )
    federal_excerpt = (
        "the federal credit for child and dependent care expenses the taxpayer is able "
        "to deduct from federal tax liability for the same taxable year"
    )
    for index, excerpt in enumerate((state_excerpt, federal_excerpt), 1):
        declaration = payload["inputs"][index]
        dependency = {
            **declaration,
            "kind": "derived",
            "entity": "TaxUnit",
            "unit": "USD",
            "period": "Year",
            "source": CITATION + "(e)",
            "versions": [
                {"effective_from": "2026-01-01", "formula": f"amount_{index}"}
            ],
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
        if binding == "proof":
            dependency.pop("description")
        payload["rules"].append(dependency)
        payload["inputs"][index] = {
            **declaration,
            "name": f"amount_{index}",
            "description": "An upstream monetary amount.",
        }
        for case in cases:
            amount = case["input"].pop(declaration["name"])
            case["input"][f"amount_{index}"] = amount
            case["output"][declaration["name"]] = amount
    return payload, cases


@pytest.mark.parametrize("binding", ["description", "proof"])
def test_public_credit_difference_accepts_individually_bound_derived_operands(binding):
    payload, cases = _derived_credit_candidate(binding)
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()


@pytest.mark.parametrize(
    "description",
    [
        "The ineligible state child care credit.",
        "The state child care credit for which the taxpayer might be eligible.",
    ],
)
def test_public_credit_operand_proof_cannot_override_contradictory_description(
    description,
):
    payload, cases = _derived_credit_candidate("proof")
    payload["rules"][1]["description"] = description
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_public_credit_operands_cannot_claim_both_credit_roles():
    payload, cases = _derived_credit_candidate("proof")
    payload["rules"][1]["metadata"]["proof"]["atoms"].extend(
        payload["rules"][2]["metadata"]["proof"]["atoms"]
    )
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def _with_credit_result_alias(payload, cases):
    root = payload["rules"][0]
    difference = {
        **root,
        "name": "credit_difference",
        "versions": [
            {"effective_from": "2026-01-01", "formula": "state_credit - federal_credit"}
        ],
    }
    payload["rules"].append(difference)
    root["versions"][0]["formula"] = (
        "if resident_has_federal_tax_liability: credit_difference else: state_credit"
    )
    for case in cases:
        case["output"]["credit_difference"] = case["output"].get(
            "state_credit", case["input"].get("state_credit")
        ) - case["output"].get("federal_credit", case["input"].get("federal_credit"))
    return payload, cases


@pytest.mark.parametrize("bound", [True, False])
def test_public_credit_difference_follows_asserted_reached_result_alias(bound):
    payload, cases = _with_credit_result_alias(*_credit_candidate())
    if not bound:
        for item in payload["inputs"][1:]:
            item["description"] = "An unrelated monetary amount."
    issues = _public_gate(payload, "E. " + CLAUSE, cases)
    if bound:
        assert issues == ()
    else:
        assert issues


def test_public_credit_result_alias_preserves_individually_proved_derived_operands():
    payload, cases = _with_credit_result_alias(*_derived_credit_candidate("proof"))
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()


@pytest.mark.parametrize("source_case", ["upper", "mixed"])
def test_public_credit_result_alias_preserves_derived_operands_regardless_of_source_case(
    source_case,
):
    source = _credit_source_variant(source_case)
    payload, cases = _with_credit_result_alias(*_derived_credit_candidate("proof"))
    for rule in payload["rules"]:
        for atom in rule["metadata"]["proof"]["atoms"]:
            atom["source"]["excerpt"] = _credit_source_variant(
                source_case, source=atom["source"]["excerpt"]
            )
    issues = _public_gate(payload, "E. " + source, cases)
    assert issues == (), issues


@given(
    names=st.sampled_from(
        [
            ("amount_a", "amount_b"),
            ("x", "y"),
            ("eligible_state_credit", "deductible_federal_credit"),
        ]
    ),
    state=st.integers(min_value=1000, max_value=10000),
    federal=st.integers(min_value=1, max_value=500),
    missing=st.sampled_from(["state", "federal", "both"]),
    source_case=st.sampled_from(["original", "upper", "mixed"]),
    whitespace=st.sampled_from([" ", "\t", "  "]),
)
@example(
    names=("amount_a", "amount_b"),
    state=1000,
    federal=200,
    missing="both",
    source_case="upper",
    whitespace="\t",
)
@settings(deadline=None, max_examples=24)
def test_public_credit_subtraction_requires_each_operand_binding(
    names, state, federal, missing, source_case, whitespace
):
    source = _credit_source_variant(source_case, whitespace)
    payload, cases = _credit_candidate(
        state_name=names[0], federal_name=names[1], source=source
    )
    for index, case in enumerate(cases):
        liability = case["input"]["resident_has_federal_tax_liability"]
        varied_state = state + (2 if index == 2 else 0)
        case["input"].update({names[0]: varied_state, names[1]: federal})
        case["output"]["child_care_credit"] = (
            varied_state - federal if liability else varied_state
        )
    assert _public_gate(payload, "E. " + source, cases) == ()
    for index, role in ((1, "state"), (2, "federal")):
        if missing in {role, "both"}:
            payload["inputs"][index]["description"] = "An unrelated monetary amount."
    assert _public_gate(payload, "E. " + source, cases)


def test_nm_credit_difference_has_no_conjunctive_operand_facts():
    gates = sc._source_conjunctive_fact_gates(CLAUSE)
    assert gates == ()
    payload = {
        "inputs": [
            {
                "name": "resident_has_federal_tax_liability",
                "description": "The resident taxpayer has federal tax liability.",
            },
            {
                "name": "state_child_care_credit",
                "description": "The amount of the state child care credit for which the taxpayer is eligible.",
            },
            {
                "name": "federal_cdcc_able_to_deduct",
                "description": "Federal credit for child and dependent care expenses the taxpayer is able to deduct from federal tax liability.",
            },
        ]
    }
    assert sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in payload["inputs"]),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(payload),
    )


def test_exact_nm_credit_clause_corroborates_formula_at_explicit_conditions_gate():
    rule = {
        "name": "nm_cdcc",
        "kind": "derived",
        "entity": "TaxUnit",
        "dtype": "Money",
        "unit": "USD",
        "period": "Year",
        "source": CITATION + "(e)",
        "versions": [
            {
                "effective_from": "2026-01-01",
                "formula": "if resident_has_federal_tax_liability: state_child_care_credit - federal_cdcc_able_to_deduct else: state_child_care_credit",
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
                            "excerpt": CLAUSE,
                        },
                    }
                ]
            }
        },
    }
    source = "E. " + CLAUSE
    payload = {
        "inputs": [
            {"name": name, "description": description}
            for name, description in (
                (
                    "resident_has_federal_tax_liability",
                    "The resident taxpayer has federal tax liability.",
                ),
                ("state_child_care_credit", "The state child care credit amount."),
                (
                    "federal_cdcc_able_to_deduct",
                    "The federal credit for child and dependent care expenses deductible from federal tax liability.",
                ),
            )
        ],
        "rules": [rule],
    }
    assert (
        sc._opaque_same_source_condition_input_issues(
            payload,
            source_text=source,
            branches=sc.recognize_source_structure(
                source, corpus_citation_path=CITATION
            ),
            principal_rules={"nm_cdcc": rule},
            corpus_citation_path=CITATION,
        )
        == []
    )


def test_dependent_care_subject_matter_is_not_dependency_polarity():
    phrase = "the federal credit for child and dependent care expenses"
    assert "dependent" not in sc._source_gate_predicate_polarities(phrase)
    assert not {
        "dependent",
        "not_dependent",
        "unknown_dependent",
    } & sc._source_gate_semantic_tokens(phrase)


@pytest.mark.parametrize(
    ("clause", "expected_predicates"),
    [
        (
            "If a parent or guardian report that they are unable to provide care "
            "for the child and require dependent care, see Special Dependent Care Situations.",
            ({"unable", "provide", "care"}, {"require", "care"}),
        ),
        (
            "If a school, institution, program, or other grantor earmarks part of "
            "educational assistance for dependent care expenses, and this income "
            "is excluded for SNAP purposes, it cannot be allowed as a deduction.",
            ({"earmark", "educational", "assistance", "care"}, {"excluded", "snap"}),
        ),
    ],
)
def test_dependent_care_actions_retain_distinct_factual_gates(
    clause, expected_predicates
):
    gates = sc._source_conjunctive_fact_gates(clause)
    assert len(gates) == len(expected_predicates)
    assert not any("unknown_dependent" in predicates for _, predicates in gates)
    for (_, predicates), expected in zip(gates, expected_predicates, strict=True):
        assert expected <= predicates


def test_dependent_care_actions_require_distinct_corroborating_terminals():
    clause = (
        "If a parent or guardian report that they are unable to provide care "
        "for the child and require dependent care, see Special Dependent Care Situations."
    )
    gates = sc._source_conjunctive_fact_gates(clause)
    assert len(gates) == 2
    collapsed = {
        "inputs": [
            {
                "name": "parent_conditions",
                "description": "The parent is unable to provide care for the child and requires dependent care.",
            }
        ]
    }
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset({"parent_conditions"}),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(collapsed),
    )
    distinct = {
        "inputs": [
            {
                "name": "parent_unable_to_provide_child_care",
                "description": "The parent reports that they are unable to provide care for the child.",
            },
            {
                "name": "parent_requires_dependent_care",
                "description": "The parent requires dependent care.",
            },
        ]
    }
    assert sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in distinct["inputs"]),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(distinct),
    )


def test_dependent_care_billed_or_paid_expense_retains_its_factual_gate():
    clause = (
        "If the ineligible member has income and dependent care expense billed "
        "to or paid by the ineligible member(s), we determine the deduction by "
        "dividing the expenses evenly among eligible and ineligible members."
    )
    gates = sc._source_conjunctive_fact_gates(clause)
    assert len(gates) == 2
    assert "income" in gates[0][1]
    assert {"billed", "paid", "care", "expense"} <= gates[1][1]
    assert not any("unknown_dependent" in predicates for _, predicates in gates)


@pytest.mark.parametrize(
    ("clause", "expected_predicates"),
    [
        (
            "If a resident individual has federal adjusted gross income below "
            "sixty thousand dollars and the individual claims the federal "
            "dependent care credit;",
            ({"income"}, {"claim", "federal", "care"}),
        ),
        (
            "If the household reports a change in dependent care payments "
            "and the information is questionable;",
            ({"report", "change", "care", "payment"}, {"questionable"}),
        ),
    ],
)
def test_care_credit_claim_and_reported_change_keep_real_conjunctions(
    clause, expected_predicates
):
    # Reduced factual antecedents from CO credit eligibility and PA SAR
    # verification. Public-gate properties below also exercise these facts.
    gates = sc._source_conjunctive_fact_gates(clause)
    assert len(gates) == 2
    for (_, predicates), expected in zip(gates, expected_predicates, strict=True):
        assert expected <= predicates
    assert not any("unknown_dependent" in predicates for _, predicates in gates)


@given(
    fact=st.sampled_from(
        [
            ("dependent care expenses", frozenset({"care", "expense"})),
            (
                "the taxpayer claims the dependent care credit",
                frozenset({"claim", "care"}),
            ),
            (
                "the taxpayer reports a change in dependent care payments",
                frozenset({"report", "change", "care", "payment"}),
            ),
            (
                "dependent care expense billed to the taxpayer",
                frozenset({"billed", "care", "expense"}),
            ),
            (
                "dependent care expense paid by the taxpayer",
                frozenset({"paid", "care", "expense"}),
            ),
        ]
    ),
    whitespace=st.sampled_from([" ", "\t", "  "]),
    uppercase=st.booleans(),
)
@settings(deadline=None, max_examples=80)
def test_care_noun_retains_its_non_dependency_tokens_and_gate_count(
    fact, whitespace, uppercase
):
    wording, expected_predicates = fact
    source = "If the taxpayer is resident and " + wording.replace(" ", whitespace) + ";"
    if uppercase:
        source = source.upper()
    gates = sc._source_conjunctive_fact_gates(source)
    assert len(gates) == 2
    assert "resident" in gates[0][1]
    assert expected_predicates <= gates[1][1]
    assert not any("unknown_dependent" in predicates for _, predicates in gates)

    _assert_public_boolean_facts(
        source,
        (
            ("taxpayer_is_resident", "The taxpayer is resident.", True),
            ("taxpayer_care_fact", wording + ".", True),
        ),
    )


@given(
    action=st.sampled_from(
        ["requires", "required", "requiring", "earmarks", "earmarked", "earmarking"]
    ),
    subject=st.sampled_from(["parent", "taxpayer", "applicant"]),
    whitespace=st.sampled_from([" ", "\t", "  "]),
    uppercase=st.booleans(),
)
@settings(deadline=None, max_examples=80)
def test_dependent_care_noun_keeps_transitive_action_facts(
    action, subject, whitespace, uppercase
):
    source = (
        f"If the {subject} {action} dependent{whitespace}care expenses "
        "and the spouse is disabled, the credit applies."
    )
    if uppercase:
        source = source.upper()
    gates = sc._source_conjunctive_fact_gates(source)
    assert len(gates) == 2
    assert subject in gates[0][0]
    assert "care" in gates[0][1]
    assert any(token.startswith(("requir", "earmark")) for token in gates[0][1])
    assert "spouse" in gates[1][0]
    assert "disabled" in gates[1][1]
    assert not any("unknown_dependent" in predicates for _, predicates in gates)

    _assert_public_boolean_facts(
        source,
        (
            (
                f"{subject}_care_action",
                f"The {subject} {action} dependent care expenses.",
                True,
            ),
            ("spouse_is_disabled", "The spouse is disabled.", True),
        ),
    )


@pytest.mark.parametrize("punctuation", [",", ";", ":", ".", " —"])
def test_punctuation_keeps_dependency_status_separate_from_care(punctuation):
    phrase = f"the child is dependent{punctuation} care expenses are deductible"
    assert sc._source_gate_predicate_polarities(phrase) == {"dependent": False}
    assert "dependent" in sc._source_gate_semantic_tokens(phrase)


def test_credit_difference_retains_complete_computation_demand():
    source = CLAUSE
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    obligations = sc._source_formula_branches(
        source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(obligations) == 1
    assert obligations[0].text == CLAUSE
    assert source[obligations[0].start : obligations[0].end] == CLAUSE


def test_credit_difference_still_requires_a_formula_output():
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []}),
        "E. " + CLAUSE,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert any(
        "[complete-source-unit:formula-output]" in issue for issue in result.issues
    )


@given(
    st.sampled_from(
        [
            "the applicant is resident",
            "the taxpayer is disabled",
            "the spouse is employed",
        ]
    ),
    st.sampled_from(
        [
            "the spouse is blind",
            "the applicant has income",
            "the child is dependent on the taxpayer",
        ]
    ),
    st.sampled_from([" ", "\n", "\t", "  "]),
)
@settings(deadline=None, max_examples=80)
def test_credit_difference_preserves_all_antecedent_fact_demands(first, second, space):
    antecedent = f"If {first} and {second}"
    expected = sc._source_conjunctive_fact_gates(antecedent)
    assert len(expected) == 2
    consequence = CLAUSE.split(",", 1)[1].replace(" ", space)
    source = antecedent + "," + consequence
    assert sc._source_conjunctive_fact_gates(source) == expected
    assert sc.source_states_explicit_computation(source)

    payload, _ = _credit_candidate(source=source)
    # Generated antecedents are a single source unit without a recognized
    # statutory subsection, so their exact proof is owned by the unit root.
    payload["rules"][0]["source"] = CITATION
    selectors = tuple(
        wording.removeprefix("the ").replace(" ", "_") for wording in (first, second)
    )
    operands = payload["inputs"][1:]
    payload["inputs"] = [
        {"name": name, "description": wording + ".", "dtype": "Boolean"}
        for name, wording in zip(selectors, (first, second), strict=True)
    ] + operands
    payload["rules"][0]["versions"][0]["formula"] = (
        f"if {selectors[0]} and {selectors[1]}: state_credit - federal_credit else: 0"
    )
    cases = [
        {
            "period": "2026-01-01",
            "input": {
                selectors[0]: left,
                selectors[1]: right,
                "state_credit": state,
                "federal_credit": federal,
            },
            "output": {"child_care_credit": state - federal if left and right else 0},
        }
        for left, right, state, federal in (
            (True, True, 1000, 200),
            (False, True, 1000, 200),
            (True, False, 1000, 200),
            (True, True, 1200, 100),
        )
    ]
    issues = _public_gate(payload, "E. " + source, cases)
    assert issues == (), issues
    for missing in selectors:
        incomplete = yaml.safe_load(yaml.safe_dump(payload))
        incomplete["inputs"] = [
            item for item in incomplete["inputs"] if item["name"] != missing
        ]
        remaining = selectors[1] if missing == selectors[0] else selectors[0]
        incomplete["rules"][0]["versions"][0]["formula"] = (
            f"if {remaining}: state_credit - federal_credit else: 0"
        )
        incomplete_cases = [
            {
                "period": case["period"],
                "input": {
                    name: value
                    for name, value in case["input"].items()
                    if name != missing
                },
                "output": {
                    "child_care_credit": (
                        case["input"]["state_credit"] - case["input"]["federal_credit"]
                        if case["input"][remaining]
                        else 0
                    )
                },
            }
            for case in cases
        ]
        assert any(
            "[complete-source-unit:source-explicit-conditions]" in issue
            for issue in _public_gate(incomplete, "E. " + source, incomplete_cases)
        )


@pytest.mark.parametrize(
    "restriction",
    [
        "if the spouse is disabled",
        "and the spouse is disabled",
        "only when the applicant is employed",
        "provided that the taxpayer is resident",
    ],
)
def test_unrecognized_credit_consequence_conditions_remain_demanded(restriction):
    restricted = CLAUSE.rstrip(";") + " " + restriction + ";"
    assert sc._source_conjunctive_fact_gates(restricted)


@given(
    st.sampled_from(
        [
            "may be dependent",
            "is sometimes dependent",
            "is neither dependent nor independent",
        ]
    )
)
@settings(deadline=None, max_examples=10)
def test_dependent_care_does_not_hide_ambiguous_dependency_status(status):
    text = f"If the child {status} on the taxpayer and the spouse is disabled, claim the dependent care credit."
    gates = sc._source_conjunctive_fact_gates(text)
    assert any("unknown_dependent" in predicates for _, predicates in gates)
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset({"child_dependent_fact", "spouse_is_disabled"}),
        gates,
        terminal_evidence=sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": "child_dependent_fact"},
                    {"name": "spouse_is_disabled"},
                ]
            }
        ),
    )

    payload, cases = _property_boolean_candidate(
        text,
        (
            ("child_dependent_fact", "The child is dependent on the taxpayer.", True),
            ("spouse_is_disabled", "The spouse is disabled.", True),
        ),
    )
    assert any(
        "[complete-source-unit:source-explicit-conditions]" in issue
        for issue in _public_gate(payload, text, cases)
    )


@given(
    st.integers(min_value=0, max_value=20),
    st.sampled_from(["the taxpayer is dependent", "the taxpayer has income"]),
)
@settings(deadline=None, max_examples=80)
def test_care_subject_preserves_other_predicate_negation_windows(padding, predicate):
    prefix = "not eligible for a credit for {} care expenses "
    suffix = "qualifier " * padding + predicate
    neutral = prefix.format("ancillary") + suffix
    subject = prefix.format("dependent") + suffix
    assert sc._source_gate_predicate_polarities(
        subject
    ) == sc._source_gate_predicate_polarities(neutral)

    _assert_public_care_window_fact_preserved(neutral, subject, predicate)


@given(
    st.integers(min_value=0, max_value=120),
    st.sampled_from(["the spouse is dependent", "the spouse has income"]),
)
@example(51, "the spouse is dependent")
@settings(deadline=None, max_examples=80)
def test_care_subject_preserves_guarded_yet_character_windows(padding, predicate):
    prefix = "the applicant is disabled yet eligible for {} care expenses "
    suffix = "q" * padding + " " + predicate
    neutral = prefix.format("ancillary") + suffix
    subject = prefix.format("dependent") + suffix
    assert sc._source_gate_predicate_polarities(
        subject
    ) == sc._source_gate_predicate_polarities(neutral)

    _assert_public_care_window_fact_preserved(neutral, subject, predicate)


def _property_boolean_candidate(source, facts):
    expression = " and ".join(
        name if active else f"not {name}" for name, _, active in facts
    )
    formula = f"if {expression}: true else: false" if expression else "true"
    rule = {
        "name": "credit_applies",
        "kind": "derived",
        "dtype": "Judgment",
        "source": CITATION,
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
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
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": name, "description": description, "dtype": "Boolean"}
            for name, description, _ in facts
        ],
        "rules": [rule],
    }
    active_inputs = {name: active for name, _, active in facts}
    cases = [
        {
            "period": "2026-01-01",
            "input": active_inputs,
            "output": {"credit_applies": True},
        }
    ]
    for name, _, active in facts:
        cases.append(
            {
                "period": "2026-01-01",
                "input": {**active_inputs, name: not active},
                "output": {"credit_applies": False},
            }
        )
    return payload, cases


def _assert_public_boolean_facts(source, facts):
    payload, cases = _property_boolean_candidate(source, facts)
    issues = _public_gate(payload, source, cases)
    assert issues == (), issues
    for missing in range(len(facts)):
        payload, cases = _property_boolean_candidate(
            source, facts[:missing] + facts[missing + 1 :]
        )
        issues = _public_gate(payload, source, cases)
        assert any(
            "[complete-source-unit:source-explicit-conditions]" in issue
            for issue in issues
        ), issues


def _assert_public_care_window_fact_preserved(neutral, subject, predicate):
    complete_outcomes = []
    for fragment in (neutral, subject):
        source = f"If the taxpayer is resident and {fragment}, the credit applies."
        facts = (("taxpayer_is_resident", "The taxpayer is resident.", True),)
        care_description = fragment
        if fragment.startswith("the applicant is disabled yet "):
            facts += (("applicant_is_disabled", "The applicant is disabled.", True),)
            care_description = fragment.split(" yet ", 1)[1]
        facts += (("care_condition", care_description + ".", True),)
        payload, cases = _property_boolean_candidate(source, facts)
        issues = _public_gate(payload, source, cases)
        ambiguous = None in sc._source_gate_predicate_polarities(fragment).values()
        if ambiguous:
            assert any(
                "[complete-source-unit:source-explicit-conditions]" in issue
                for issue in issues
            ), issues
        else:
            assert issues == (), issues
        complete_outcomes.append(bool(issues))
        missing_facts = facts[:-1] + (
            (
                "care_condition",
                care_description.removesuffix(predicate) + ".",
                True,
            ),
        )
        payload, cases = _property_boolean_candidate(source, missing_facts)
        issues = _public_gate(payload, source, cases)
        assert any(
            "[complete-source-unit:source-explicit-conditions]" in issue
            for issue in issues
        ), issues
    assert complete_outcomes[0] is complete_outcomes[1]
