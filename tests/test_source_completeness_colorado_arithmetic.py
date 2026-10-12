"""Physical PDF wraps must not invent facts in the Colorado refund arithmetic."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "us-co/statute/39/39-22-2003"
REFUND_COMPARISON = """(6) Except as otherwise provided in this subsection (6), if the refund allowed
under this section exceeds the income taxes otherwise due on the claimant's
income, the amount of the refund shall be refunded to the claimant."""
CARRY_FORWARD_ELECTION = """The claimant
may elect to carry forward the amount of the refund not used as an offset against
income taxes or as an offset against subsequent years' income tax liability."""
FACTS_AND_ELECTION = (
    "A cash refund is allowed if the claimant is a resident and the claimant "
    "elects to receive cash."
)


def _rule(name: str, formula: str, excerpt: str) -> dict:
    return {
        "name": name,
        "kind": "derived",
        "dtype": "Money",
        "unit": "USD",
        "source": CITATION,
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
        "versions": [{"formula": formula}],
    }


def _condition_issues(source: str, rules: list[dict], inputs: list[dict]) -> list[str]:
    return sc._opaque_same_source_condition_input_issues(
        {"rules": rules, "inputs": inputs},
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=CITATION),
        principal_rules={rule["name"]: rule for rule in rules},
        corpus_citation_path=CITATION,
    )


def _execute(formula: str, environment: dict):
    execution = sc._execute_formula_text(
        formula, environment=environment, constant_environment={}
    )
    assert execution is not None
    return sc._formula_execution_runtime_value(execution)


def test_wrapped_refund_comparison_has_no_extra_conjunctive_income_gates():
    gates = sc._source_conjunctive_fact_gates(REFUND_COMPARISON)

    assert (
        gates
        == sc._source_conjunctive_fact_gates(" ".join(REFUND_COMPARISON.split()))
        == ()
    )
    assert sc._terminal_names_corroborate_source_gates(
        frozenset({"refund_amount", "income_tax_liability", "claimant_income"}),
        gates,
        terminal_evidence={},
    )


def test_refund_arithmetic_proof_excerpts_pass_explicit_condition_gate():
    rules = [
        _rule(
            "refund_tax_offset",
            "min(refund_amount, income_tax_liability)",
            REFUND_COMPARISON,
        ),
        _rule(
            "unused_refund_amount",
            "max(refund_amount - income_tax_liability, 0)",
            REFUND_COMPARISON,
        ),
        _rule(
            "cash_refund_amount",
            "if refund_amount > income_tax_liability: refund_amount - income_tax_liability else: 0",
            REFUND_COMPARISON,
        ),
    ]
    inputs = [
        {"name": name, "dtype": "Money", "description": name.replace("_", " ")}
        for name in ("refund_amount", "income_tax_liability", "claimant_income")
    ]

    assert not _condition_issues(REFUND_COMPARISON, rules, inputs)


@pytest.mark.parametrize(
    "refund,tax,offset,unused",
    [(150, 100, 100, 50), (100, 100, 100, 0), (50, 100, 50, 0)],
)
def test_refund_and_offset_comparison_is_executed_arithmetic(
    refund, tax, offset, unused
):
    environment = {"refund_amount": refund, "income_tax_liability": tax}
    assert _execute("min(refund_amount, income_tax_liability)", environment) == offset
    assert (
        _execute("max(refund_amount - income_tax_liability, 0)", environment) == unused
    )
    assert (
        _execute(
            "if refund_amount > income_tax_liability: refund_amount - income_tax_liability else: 0",
            environment,
        )
        == unused
    )


def test_claimant_carry_forward_election_changes_executed_refund_destination():
    source = REFUND_COMPARISON + " " + CARRY_FORWARD_ELECTION
    cash_formula = "if claimant_elects_carry_forward: 0 else: max(refund_amount - income_tax_liability, 0)"
    carry_formula = "if claimant_elects_carry_forward: max(refund_amount - income_tax_liability, 0) else: 0"
    rules = [
        _rule("cash_refund_amount", cash_formula, source),
        _rule("carry_forward_amount", carry_formula, CARRY_FORWARD_ELECTION),
    ]
    inputs = [
        {"name": name, "dtype": "Money"}
        for name in ("refund_amount", "income_tax_liability")
    ] + [
        {
            "name": "claimant_elects_carry_forward",
            "dtype": "Boolean",
            "description": "The claimant elects to carry forward the unused refund.",
        }
    ]
    assert not _condition_issues(source, rules, inputs)
    for elects, cash, carry in [(False, 50, 0), (True, 0, 50)]:
        environment = {
            "refund_amount": 150,
            "income_tax_liability": 100,
            "claimant_elects_carry_forward": elects,
        }
        assert _execute(cash_formula, environment) == cash
        assert _execute(carry_formula, environment) == carry


@pytest.mark.parametrize(
    "missing", ["claimant_is_resident", "claimant_elects_to_receive_cash"]
)
def test_actual_facts_and_claimant_election_remain_required(missing):
    names = ("claimant_is_resident", "claimant_elects_to_receive_cash")
    inputs = [
        {"name": name, "dtype": "Boolean", "description": name.replace("_", " ")}
        for name in names
    ]
    formula = "if claimant_is_resident and claimant_elects_to_receive_cash: refund_amount else: 0"
    rules = [_rule("cash_refund_amount", formula, FACTS_AND_ELECTION)]
    inputs.append({"name": "refund_amount", "dtype": "Money"})
    assert len(sc._source_conjunctive_fact_gates(FACTS_AND_ELECTION)) == 2
    assert not _condition_issues(FACTS_AND_ELECTION, rules, inputs)

    # Companion cases execute the real formula, holding the amount and the
    # other fact fixed while changing exactly one source condition.
    present = {**dict.fromkeys(names, True), "refund_amount": 50}
    absent = {**present, missing: False}
    assert _execute(formula, present) == 50
    assert _execute(formula, absent) == 0

    omitted_formula = formula.replace(missing + " and ", "").replace(
        " and " + missing, ""
    )
    rules[0]["versions"][0]["formula"] = omitted_formula
    assert _condition_issues(FACTS_AND_ELECTION, rules, inputs)
    assert _execute(omitted_formula, absent) == 50


@pytest.mark.parametrize("first,second", [("●", "●"), ("(a)", "(b)"), ("1.", "2.")])
def test_explicit_list_rows_still_demand_each_factual_condition(first, second):
    source = (
        "A refund is allowed if all of these conditions hold:\n"
        f"{first} The claimant is a resident.\n"
        f"{second} The claimant elects to receive cash."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    inputs = [
        {"name": name, "description": name.replace("_", " ")}
        for name in ("claimant_is_resident", "claimant_elects_to_receive_cash")
    ]
    evidence = sc._terminal_gate_evidence({"inputs": inputs})
    assert len(gates) == 2
    assert sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in inputs), gates, terminal_evidence=evidence
    )
    for input_item in inputs:
        assert not sc._terminal_names_corroborate_source_gates(
            frozenset({input_item["name"]}), gates, terminal_evidence=evidence
        )


def test_ambiguous_income_fact_still_fails_closed():
    source = "A credit applies if the claimant reports income and is a resident."
    gates = sc._source_conjunctive_fact_gates(source)
    inputs = [
        {"name": name, "description": name.replace("_", " ")}
        for name in ("claimant_reports_income", "claimant_is_resident")
    ]
    assert any("unknown_income" in predicates for _, predicates in gates)
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in inputs),
        gates,
        terminal_evidence=sc._terminal_gate_evidence({"inputs": inputs}),
    )


@pytest.mark.parametrize("separator", [" ", "\n"])
@pytest.mark.parametrize("oxford_comma", [False, True])
def test_comma_separated_fact_clauses_cannot_omit_residency(separator, oxford_comma):
    source = (
        "A credit applies if the taxpayer is a resident,"
        + separator
        + "the spouse has no income"
        + ("," if oxford_comma else "")
        + " and"
        + separator
        + "the taxpayer is blind."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert gates == (
        (frozenset({"taxpayer"}), frozenset({"resident"})),
        (frozenset({"spouse"}), frozenset({"no_income"})),
        (frozenset({"taxpayer"}), frozenset({"blind"})),
    )
    inputs = [
        {"name": name, "dtype": "Boolean", "description": name.replace("_", " ")}
        for name in (
            "taxpayer_is_resident",
            "spouse_has_no_income",
            "taxpayer_is_blind",
        )
    ] + [{"name": "refund_amount", "dtype": "Money"}]
    omitted = "if spouse_has_no_income and taxpayer_is_blind: refund_amount else: 0"
    complete = "if taxpayer_is_resident and spouse_has_no_income and taxpayer_is_blind: refund_amount else: 0"
    rule = _rule("conditional_credit", omitted, source)
    assert _condition_issues(source, [rule], inputs)
    rule["versions"][0]["formula"] = complete
    assert not _condition_issues(source, [rule], inputs)
    environment = {
        "taxpayer_is_resident": False,
        "spouse_has_no_income": True,
        "taxpayer_is_blind": True,
        "refund_amount": 50,
    }
    assert _execute(complete, environment) == 0
    assert _execute(omitted, environment) == 50


@pytest.mark.parametrize(
    "source",
    [
        "A refund applies if the taxpayer is resident, the claimant receives the refund.",
        "A refund applies if the taxpayer is resident, the claimant is entitled to the refund.",
    ],
)
def test_conditional_comma_does_not_make_a_refund_consequence_a_fact(source):
    assert sc._source_conjunctive_fact_gates(source) == ()


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_serial_fact_member_cannot_be_replaced_by_its_declarative_tail(separator):
    source = (
        "A credit applies if the taxpayer is resident,"
        + separator
        + "the spouse has no income, and"
        + separator
        + "the taxpayer is blind,"
        + separator
        + "the claimant receives cash."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    names = frozenset(
        {"taxpayer_is_resident", "spouse_has_no_income", "claimant_receives_cash"}
    )
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": name.replace("_", " ")} for name in names
            ]
        }
    )
    assert len(gates) == 4
    assert not sc._terminal_names_corroborate_source_gates(
        names, gates, terminal_evidence=evidence
    )


@pytest.mark.parametrize(
    "chapeau",
    ["the following conditions are met", "all of these conditions hold"],
)
def test_unbulleted_fact_list_keeps_each_explicit_condition(chapeau):
    source = (
        f"A credit applies if {chapeau}:\n"
        "The taxpayer is a resident\n"
        "The spouse is blind."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert (frozenset({"taxpayer"}), frozenset({"resident"})) in gates
    assert (frozenset({"spouse"}), frozenset({"blind"})) in gates
    formula = "if taxpayer_is_resident and spouse_is_blind: refund_amount else: 0"
    environment = {
        "taxpayer_is_resident": True,
        "spouse_is_blind": False,
        "refund_amount": 50,
    }
    assert _execute(formula, environment) == 0
    assert _execute("if taxpayer_is_resident: refund_amount else: 0", environment) == 50


@pytest.mark.parametrize(
    "spouse_conditions", [("has no income", "is blind"), ("is blind", "has no income")]
)
@pytest.mark.parametrize("include_taxpayer", [False, True])
@pytest.mark.parametrize(
    "chapeau",
    ["the following conditions are met", "all of these conditions hold"],
)
def test_unbulleted_fact_list_keeps_inherited_spouse_blindness(
    spouse_conditions, include_taxpayer, chapeau
):
    source = (
        f"A credit applies if {chapeau}:\n"
        + ("The taxpayer is resident\n" if include_taxpayer else "")
        + "The spouse "
        + spouse_conditions[0]
        + "\n"
        + spouse_conditions[1]
        + "."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert (frozenset({"spouse"}), frozenset({"blind"})) in gates
    names = frozenset({"following_conditions_are_met", "spouse_has_no_income"})
    if include_taxpayer:
        names |= {"taxpayer_is_resident"}
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": name.replace("_", " ")} for name in names
            ]
        }
    )
    assert not sc._terminal_names_corroborate_source_gates(
        names, gates, terminal_evidence=evidence
    )
    environment = {
        "taxpayer_is_resident": True,
        "spouse_has_no_income": True,
        "spouse_is_blind": False,
        "refund_amount": 50,
    }
    assert (
        _execute(
            "if taxpayer_is_resident and spouse_has_no_income and spouse_is_blind: refund_amount else: 0",
            environment,
        )
        == 0
    )
    assert (
        _execute(
            "if taxpayer_is_resident and spouse_has_no_income: refund_amount else: 0",
            environment,
        )
        == 50
    )


@settings(max_examples=25, deadline=None)
@example(
    breaks={9, 13, 18},
    reverse=False,
    income_fact="has no income",
    include_taxpayer=False,
    chapeau="the following conditions are met",
)
@given(
    breaks=st.sets(st.integers(min_value=1, max_value=20), max_size=15),
    reverse=st.booleans(),
    income_fact=st.sampled_from(
        ["has no income", "has income", "does not have income", "never has income"]
    ),
    include_taxpayer=st.booleans(),
    chapeau=st.sampled_from(
        ["the following conditions are met", "all of these conditions hold"]
    ),
)
def test_inherited_unbulleted_fact_list_wraps_keep_independent_conditions(
    breaks, reverse, income_fact, include_taxpayer, chapeau
):
    spouse_conditions = (
        "is blind " + income_fact if reverse else income_fact + " is blind"
    )
    source = (
        f"A credit applies if {chapeau}: "
        + ("The taxpayer is resident " if include_taxpayer else "")
        + "The spouse "
        + spouse_conditions
        + "."
    )
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(source.split(), start=1)
    ).strip()
    gates = sc._source_conjunctive_fact_gates(wrapped)
    assert gates == sc._source_conjunctive_fact_gates(source)
    income_predicate = "income" if income_fact == "has income" else "no_income"
    assert any(
        entities == frozenset({"spouse"}) and income_predicate in predicates
        for entities, predicates in gates
    )
    assert (frozenset({"spouse"}), frozenset({"blind"})) in gates


def test_inherited_fact_list_head_retains_its_or_alternative():
    source = (
        "A credit applies if the following conditions are met:\n"
        "The taxpayer is resident or is blind\nThe spouse has no income."
    )
    gates = tuple(gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0])
    assert gates == (
        (frozenset({"taxpayer"}), frozenset({"resident", "blind"})),
        (frozenset({"spouse"}), frozenset({"no_income"})),
    )


def test_inherited_fact_list_head_does_not_split_an_income_noun_complement():
    source = (
        "A credit applies if the following conditions are met:\n"
        "The taxpayer is resident\nThe spouse receives an amount that\nis income."
    )
    gates = tuple(gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0])
    assert gates == (
        (frozenset({"taxpayer"}), frozenset({"resident"})),
        (frozenset({"spouse"}), frozenset({"amount", "receive", "unknown_income"})),
    )


@pytest.mark.parametrize(
    "spouse_clause,expected",
    [
        ("is not allowed to have income", {"allowed", "income"}),
        ("may have income", {"may", "income"}),
        ("does not ever have income", {"does", "ever", "income"}),
        ("may potentially have income", {"may", "potentially", "income"}),
        (
            "receives an amount that generally is income",
            {"receive", "amount", "generally", "unknown_income"},
        ),
        (
            "receives an amount that really is income",
            {"receive", "amount", "really", "unknown_income"},
        ),
    ],
)
def test_inherited_list_heads_preserve_governing_verb_and_relative_scope(
    spouse_clause, expected
):
    source = (
        "A credit applies if all of these conditions hold:\n"
        "The taxpayer is resident\n"
        "The spouse " + spouse_clause.replace(" is income", "\nis income") + "."
    )
    gates = tuple(gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0])
    assert gates == (
        (frozenset({"taxpayer"}), frozenset({"resident"})),
        (frozenset({"spouse"}), frozenset(expected)),
    )


@pytest.mark.parametrize("negative", ["does not have income", "never has income"])
def test_named_auxiliary_income_fact_preserves_its_negation(negative):
    source = (
        "A credit applies if all of these conditions hold:\n"
        "The taxpayer is resident\nThe spouse " + negative + "."
    )
    gates = tuple(gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0])
    assert len(gates) == 2
    assert gates[0] == (frozenset({"taxpayer"}), frozenset({"resident"}))
    assert gates[1][0] == frozenset({"spouse"})
    assert "no_income" in gates[1][1]
    assert "income" not in gates[1][1]
    complete_names = frozenset({"taxpayer_is_resident", "spouse_has_no_income"})
    for names in (
        frozenset({"taxpayer_is_resident"}),
        frozenset({"taxpayer_is_resident", "spouse_has_income"}),
        complete_names,
    ):
        evidence = sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": name, "description": name.replace("_", " ")}
                    for name in names
                ]
            }
        )
        assert sc._terminal_names_corroborate_source_gates(
            names, gates, terminal_evidence=evidence
        ) == (names == complete_names)
    formula = "if taxpayer_is_resident and spouse_has_no_income: refund_amount else: 0"
    environment = {
        "taxpayer_is_resident": True,
        "spouse_has_no_income": False,
        "refund_amount": 50,
    }
    assert _execute(formula, environment) == 0
    assert _execute("if taxpayer_is_resident: refund_amount else: 0", environment) == 50


@settings(max_examples=25, deadline=None)
@given(
    modifier=st.sampled_from(
        ["really", "plainly", "apparently", "ordinarily", "typically", "normally"]
    ),
    breaks=st.sets(st.integers(min_value=1, max_value=30), max_size=15),
)
def test_relative_income_classification_keeps_full_scope_under_word_wraps(
    modifier, breaks
):
    source = (
        "A credit applies if all of these conditions hold: "
        "The taxpayer is resident The spouse receives an amount that "
        + modifier
        + " is income."
    )
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(source.split(), start=1)
    ).strip()
    gates = tuple(
        gate for gate in sc._source_conjunctive_fact_gates(wrapped) if gate[0]
    )
    assert gates == tuple(
        gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0]
    )
    assert len(gates) == 2
    assert gates[0] == (frozenset({"taxpayer"}), frozenset({"resident"}))
    assert gates[1][0] == frozenset({"spouse"})
    assert {"receive", "amount", "unknown_income"} <= gates[1][1]
    inputs = [
        {"name": name, "description": name.replace("_", " ")}
        for name in ("taxpayer_is_resident", "spouse_has_income")
    ]
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset(item["name"] for item in inputs),
        gates,
        terminal_evidence=sc._terminal_gate_evidence({"inputs": inputs}),
    )


@settings(max_examples=25, deadline=None)
@given(breaks=st.sets(st.integers(min_value=1, max_value=24), max_size=15))
def test_unbulleted_fact_list_word_wraps_keep_independent_conditions(breaks):
    source = (
        "A credit applies if the following conditions are met: "
        "The taxpayer is a resident The spouse is blind."
    )
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(source.split(), start=1)
    ).strip()
    gates = sc._source_conjunctive_fact_gates(wrapped)
    assert gates == sc._source_conjunctive_fact_gates(source)
    assert (frozenset({"taxpayer"}), frozenset({"resident"})) in gates
    assert (frozenset({"spouse"}), frozenset({"blind"})) in gates
    for omitted in ("taxpayer_is_resident", "spouse_is_blind"):
        names = frozenset({"taxpayer_is_resident", "spouse_is_blind"}) - {omitted}
        evidence = sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": name, "description": name.replace("_", " ")}
                    for name in names
                ]
            }
        )
        assert not sc._terminal_names_corroborate_source_gates(
            names, gates, terminal_evidence=evidence
        )


@pytest.mark.parametrize(
    "prefix",
    ["", "(3)(A) ", "Income classification. ", "Except as otherwise provided, "],
)
def test_receipt_classification_does_not_add_an_outcome_fact_gate(prefix):
    source = (
        prefix + "If it is not received by the client or landlord until July, it is a "
        "non-recurring lump sum payment and is exempt as income."
    )
    assert sc._source_conjunctive_fact_gates(source) == (
        (
            frozenset(),
            frozenset(
                {
                    "client",
                    "july",
                    "landlord",
                    "lump",
                    "non-recurring",
                    "payment",
                    "received",
                    "sum",
                    "until",
                }
            ),
        ),
        (frozenset(), frozenset({"exempt", "unknown_income"})),
    )


@settings(max_examples=20, deadline=None)
@given(
    month=st.sampled_from(["July", "August", "March"]),
    breaks=st.sets(st.integers(min_value=1, max_value=30), max_size=15),
)
def test_receipt_classification_word_wraps_preserve_original_fact_scope(month, breaks):
    source = (
        f"If it is not received by the client or landlord until {month}, it is a "
        "non-recurring lump sum payment and is exempt as income."
    )
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(source.split(), start=1)
    ).strip()
    gates = sc._source_conjunctive_fact_gates(wrapped)
    assert len(gates) == 2
    assert gates == sc._source_conjunctive_fact_gates(source)
    assert {"received", "lump", "payment"} <= gates[0][1]


@pytest.mark.parametrize("pronoun", ["he", "she"])
def test_preposed_result_preserves_human_pronoun_fact_lists(pronoun):
    source = (
        f"A credit applies if {pronoun} is resident, the spouse has no income "
        f"and {pronoun} is blind."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert len(gates) == 3
    assert [predicates for _, predicates in gates] == [
        frozenset({"resident"}),
        frozenset({"no_income"}),
        frozenset({"blind"}),
    ]


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_unbulleted_fact_list_preserves_an_alternative_inside_its_member(separator):
    source = (
        "A credit applies if the following conditions are met:"
        + separator
        + "The taxpayer is a resident or the spouse is blind."
        + separator
        + "The claimant elects to receive cash."
    )
    gates = tuple(gate for gate in sc._source_conjunctive_fact_gates(source) if gate[0])
    assert gates == (
        (frozenset({"taxpayer", "spouse"}), frozenset({"resident", "blind"})),
        (frozenset({"claimant"}), frozenset({"elect", "receive", "cash"})),
    )
    for alternative in ("taxpayer_is_resident", "spouse_is_blind"):
        names = frozenset({alternative, "claimant_elects_to_receive_cash"})
        evidence = sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": name, "description": name.replace("_", " ")}
                    for name in names
                ]
            }
        )
        assert sc._terminal_names_corroborate_source_gates(
            names, gates, terminal_evidence=evidence
        )
    formula = "if (taxpayer_is_resident or spouse_is_blind) and claimant_elects_to_receive_cash: refund_amount else: 0"
    for resident, blind in ((True, False), (False, True)):
        assert (
            _execute(
                formula,
                {
                    "taxpayer_is_resident": resident,
                    "spouse_is_blind": blind,
                    "claimant_elects_to_receive_cash": True,
                    "refund_amount": 50,
                },
            )
            == 50
        )


@settings(max_examples=30, deadline=None)
@example(
    profile=(
        "A credit applies if the taxpayer is a resident under 26 U.S.C. 1, the spouse has no income, and the taxpayer is blind.",
        ("taxpayer_is_resident", "spouse_has_no_income", "taxpayer_is_blind"),
    ),
    breaks={12},
    oxford_comma=False,
)
@given(
    profile=st.sampled_from(
        [
            (
                "A credit applies if the taxpayer is a resident, the spouse has no income, and the taxpayer is blind.",
                ("taxpayer_is_resident", "spouse_has_no_income", "taxpayer_is_blind"),
            ),
            (
                "A refund applies if the claimant elects to receive cash, the spouse has no income, and the claimant is a resident.",
                (
                    "claimant_elects_to_receive_cash",
                    "spouse_has_no_income",
                    "claimant_is_resident",
                ),
            ),
            (
                "A credit applies if the taxpayer is a resident, has no income, and is blind.",
                ("taxpayer_is_resident", "taxpayer_has_no_income", "taxpayer_is_blind"),
            ),
            (
                "A credit applies if the taxpayer is a resident under subsection (1.5), the spouse has no income, and the taxpayer is blind.",
                ("taxpayer_is_resident", "spouse_has_no_income", "taxpayer_is_blind"),
            ),
            (
                "A credit applies if the taxpayer is a resident under 26 U.S.C. 1, the spouse has no income, and the taxpayer is blind.",
                ("taxpayer_is_resident", "spouse_has_no_income", "taxpayer_is_blind"),
            ),
            (
                "A credit applies if the taxpayer is resident, blind, and married.",
                ("taxpayer_is_resident", "taxpayer_is_blind", "taxpayer_is_married"),
            ),
        ]
    ),
    breaks=st.sets(st.integers(min_value=1, max_value=30), max_size=20),
    oxford_comma=st.booleans(),
)
def test_serial_comma_factual_conditions_keep_three_distinct_demands_under_wraps(
    profile, breaks, oxford_comma
):
    source, names = profile
    if not oxford_comma:
        source = source.replace(", and", " and")
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(source.split(), start=1)
    ).strip()
    gates = sc._source_conjunctive_fact_gates(wrapped)
    assert gates == sc._source_conjunctive_fact_gates(source)
    assert len(gates) == 3
    for omitted in names:
        retained = frozenset(names) - {omitted}
        evidence = sc._terminal_gate_evidence(
            {
                "inputs": [
                    {"name": name, "description": name.replace("_", " ")}
                    for name in retained
                ]
            }
        )
        assert not sc._terminal_names_corroborate_source_gates(
            retained,
            gates,
            terminal_evidence=evidence,
        )


@settings(max_examples=15, deadline=None)
@given(
    status=st.sampled_from(["resident", "disabled", "citizen"]),
    tail=st.sampled_from(["", " Enter another amount.", " Only residents may claim."]),
)
def test_unadmitted_printed_rows_retain_claimant_facts_and_operand_demands(
    status, tail
):
    source = (
        Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt"
    ).read_text()
    start = source.index("Enter the amount from line 1 of Form T2209.")
    end = source.index("Form 428.", start) + len("Form 428.")
    # This intentionally lacks the complete publisher context required by the
    # existing row-partition admission rule. A printed claimant restriction
    # cannot be flattened into the worksheet's arithmetic operand prose.
    partial = (
        source[start:end].replace("Net foreign", f"The claimant is {status}") + tail
    )
    gates = sc._source_conjunctive_fact_gates(partial)
    assert any(status in predicates for _, predicates in gates)
    assert len(gates) >= 2
    inputs = [
        {"name": "minimum_tax_is_payable", "description": "Minimum tax is payable."}
    ]
    assert not sc._terminal_names_corroborate_source_gates(
        frozenset({"minimum_tax_is_payable"}),
        gates,
        terminal_evidence=sc._terminal_gate_evidence({"inputs": inputs}),
    )


@settings(max_examples=50, deadline=None)
@example(source=REFUND_COMPARISON, breaks={14, 27})
@given(
    source=st.sampled_from(
        [
            REFUND_COMPARISON,
            FACTS_AND_ELECTION,
            "A credit applies if the spouse is blind, yet the spouse has no income.",
            "A credit applies if the taxpayer has income and the spouse has no income.",
            "A refund applies if the claimant is a resident under subsection (6) of this section and the claimant elects to receive cash.",
        ]
    ),
    breaks=st.sets(st.integers(min_value=1, max_value=60), max_size=20),
)
def test_ordinary_word_wraps_preserve_source_fact_gates(source, breaks):
    words = source.split()
    flat = " ".join(words)
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(words, start=1)
    ).strip()
    assert sc._source_conjunctive_fact_gates(
        wrapped
    ) == sc._source_conjunctive_fact_gates(flat)
