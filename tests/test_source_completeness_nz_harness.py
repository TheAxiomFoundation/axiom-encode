"""Public-law regressions for outline, locator and clause ownership checks.

Excerpts: Social Security Act 2018 (2018/32); amending Act 2026/54, section 15.
Source: New Zealand Legislation (legislation.govt.nz), Crown copyright, CC BY 4.0
"""

from __future__ import annotations

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

NZ_ACT = "nz/statute/act/public/2018/0032"
NZ_AMENDMENT = "nz/statute/act/public/2026/0054/section/15"


def numeric_values(source: str, citation: str) -> list[float]:
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    return vp.extract_numeric_occurrences_from_text(cleaned)


def test_completed_prose_subsections_mask_only_structural_numbers():
    source = (
        "(1) The winter energy payment is not payable for 1 or more days.\n"
        "(2) The winter energy payment is payable for up to a maximum of 28 days.\n"
        "(3) Sections 219 and 436 do not apply to the winter energy payment.\n"
    )
    # The terminal prose "do not apply" does not authenticate a bare final
    # list item. Keep its conservative numeric obligation while retaining all
    # three positively identified subsection branches.
    assert numeric_values(source, f"{NZ_ACT}/section/220") == [1, 28, 436]
    assert {branch.path for branch in sc.recognize_source_structure(source)} == {
        ("1",),
        ("2",),
        ("3",),
    }


def test_unsupported_value_table_retains_numeric_clause_label_obligations():
    # Even a legal caption cannot authenticate extra value-cell ownership.
    source = (
        "Clause | Provision\n1 | The rates of winter energy payment are as follows:\n"
        "(a) | to a person who is single and has no dependent children | $ 450 "
        "per winter period (b) | to a person who is single and has 1 or more "
        "dependent children | $ 700 per winter period "
        "2 | For the purposes of clause 1(a) and (b), dependent child means "
        "any other person who is a child.\n"
    )
    assert numeric_values(source, f"{NZ_ACT}/schedule/4/part/8/clause/lms118454") == [
        1,
        450,
        1,
        700,
        2,
    ]


def test_unsupported_amendment_value_table_retains_numeric_row_obligations():
    source = (
        "In Schedule 4 , Part 7, replace clauses 4 to 6A with:\n"
        "4 | To a person | $305 a week\n"
        "4A | To a specified person | $220 a week\n"
        "5 | To any other person | $160 a week\n"
    )
    assert numeric_values(source, NZ_AMENDMENT) == [4, 305, 220, 5, 160]


def test_table_alternatives_have_independent_source_gate_ownership():
    source = (
        "In Schedule 4, Part 7, replace clauses 4 to 5 with:\n"
        # Synthetic marker-only rows have no ambiguous value-cell boundary.
        "4 | To a person who has dependent children and is in a relationship "
        "$305 a week, if the applicant resides in Area 1\n"
        "5 | To any other person $235 a week, if the applicant resides in Area 2\n"
    )
    clauses, ambiguous = sc._source_condition_clauses_owned_by_excerpt(
        "$305 a week",
        rule={"source": NZ_AMENDMENT},
        source_text=source,
        branches=sc.recognize_source_structure(
            source, corpus_citation_path=NZ_AMENDMENT
        ),
        corpus_citation_path=NZ_AMENDMENT,
    )
    assert not ambiguous
    assert len(clauses) == 1
    assert clauses[0].branch_path == ("4",)
    assert "$235" not in clauses[0].text
    assert sc._source_conjunctive_fact_gates(clauses[0].text) == ()


def test_counterfactual_rate_does_not_require_unchanged_circumstances_as_a_fact():
    source = (
        "The rate is the rate in clauses 1 to 6AA that would apply to them "
        "if they were single (and their other circumstances were unchanged)."
    )
    assert sc._source_conjunctive_fact_gates(source) == ()
    assert (
        len(
            sc._source_conjunctive_fact_gates(
                "The payment applies if the applicant is single and has no income."
            )
        )
        == 2
    )


def test_independent_table_row_still_rejects_a_missing_conjunctive_fact():
    source = (
        "In Schedule 4, Part 7, replace clauses 4 to 5 with:\n"
        "4 | The amount is $305 if the applicant is single and has no income.\n"
        "5 | The amount is $235 if the applicant is partnered.\n"
    )
    rule = {
        "name": "payment",
        "kind": "derived",
        "dtype": "Money",
        "source": NZ_AMENDMENT,
        "versions": [{"formula": "if applicant_is_single: 305 else: 0"}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": NZ_AMENDMENT,
                            "excerpt": "$305",
                        },
                    }
                ]
            }
        },
    }
    payload = {
        "rules": [rule],
        "inputs": [
            {
                "name": "applicant_is_single",
                "dtype": "Boolean",
                "description": "The applicant is single.",
            }
        ],
    }
    issues = sc._opaque_same_source_condition_input_issues(
        payload,
        source_text=source,
        branches=sc.recognize_source_structure(
            source, corpus_citation_path=NZ_AMENDMENT
        ),
        principal_rules={"payment": rule},
        corpus_citation_path=NZ_AMENDMENT,
    )
    assert len(issues) == 1
    assert "2 source gates" in issues[0]


@pytest.mark.parametrize(
    "citation", [NZ_AMENDMENT, "uk/statute/example/15", "us/statute/26/15"]
)
def test_non_german_clause_reference_never_creates_satz_branch(citation: str):
    # No complete outline: exercise the fallback independently of table parsing.
    source = "The rate in clauses 1 to 6AA applies. 6B identifies the specified person."
    assert not [
        branch
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path=citation
        )
        if branch.kind == "sentence"
    ]
    assert "6AA" in sc.authoritative_numeric_recall_text(
        "The category 6AA is unchanged.", corpus_citation_path=citation
    )


@pytest.mark.parametrize("jurisdiction", ["de", "at", "ch"])
def test_german_glued_sentence_fallback_stays_available(jurisdiction: str):
    source = "(1) 1Die Hauptregel gilt.2Die Sonderregel gilt."
    branches = sc.recognize_source_structure(
        source, corpus_citation_path=f"{jurisdiction}/statute/estg/32a"
    )
    assert {b.path for b in branches if b.kind == "sentence"} == {
        ("1", "satz-1"),
        ("1", "satz-2"),
    }


@pytest.mark.parametrize("jurisdiction", ["de", "at", "ch"])
@pytest.mark.parametrize(
    "subject",
    [
        "EU-Bürger",
        "EG-Verordnungen",
        "UN-Organisationen",
        "US-amerikanische Staatsangehörige",
        "EU-rechtliche Vorgaben",
        "SGB-II-Leistungen",
    ],
)
def test_german_glued_acronym_compounds_are_sentence_starts(
    subject: str, jurisdiction: str
):
    source = f"(1) 1{subject} erhalten die Zahlung.2{subject} erhalten die Leistung."
    branches = sc.recognize_source_structure(
        source, corpus_citation_path=f"{jurisdiction}/statute/example/1"
    )
    assert {b.path for b in branches if b.kind == "sentence"} == {
        ("1", "satz-1"),
        ("1", "satz-2"),
    }
    assert not numeric_values(source, f"{jurisdiction}/statute/example/1")


def test_german_source_preserves_bare_alphanumeric_clause_locators():
    source = "Die Zahlung gilt nach den Klauseln 6AA und 6B."
    assert not [
        branch
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path="de/statute/example/1"
        )
        if branch.kind == "sentence"
    ]


@pytest.mark.parametrize(
    "citation", [NZ_AMENDMENT, "at/statute/example/1", "ch/statute/example/1"]
)
def test_english_glued_prose_does_not_activate_german_sentence_fallback(citation):
    source = "6Accommodation costs are the sum of mortgage payments."
    assert not sc.recognize_source_structure(source, corpus_citation_path=citation)
    assert sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    ).startswith("6Accommodation")


@pytest.mark.parametrize(
    "citation",
    [
        "",
        "fr/statute/example/article/1",
        "at/statute/example/1",
        "ch/statute/example/1",
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        "Les prestations des Allocations Familiales sont versées. "
        "1Le montant est de 450 euros.",
        "1Des prestations sont versées. 2Le montant est de 450 euros.",
    ],
)
def test_french_article_des_does_not_authenticate_german_sentence_numbers(
    source: str, citation: str
):
    assert not [
        branch
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path=citation
        )
        if branch.kind == "sentence"
    ]


@pytest.mark.parametrize("citation", ["", NZ_AMENDMENT, "uk/statute/example/15"])
def test_person_name_does_not_authenticate_german_sentence_numbers(citation: str):
    source = "The beneficiary Van der Meer is entitled. 6Accommodation costs are paid."
    assert not [
        branch
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path=citation
        )
        if branch.kind == "sentence"
    ]


def test_german_grammatical_prose_authenticates_fallback_without_corpus_path():
    source = (
        "(1) 1Aus der Bescheinigung muss der Beginn der Behinderung hervorgehen, "
        "soweit das Kind das 25. Lebensjahr vollendet hat. "
        "2Wenn die Bescheinigung fehlt, ist der Nachweis nicht erbracht."
    )
    assert {
        b.path for b in sc.recognize_source_structure(source) if b.kind == "sentence"
    } == {("1", "satz-1"), ("1", "satz-2")}


def test_clause_reference_stays_in_owning_table_row():
    source = (
        "In Schedule 4, Part 7, replace clauses 6 to 6B with:\n"
        # Synthetic marker-only rows retain the independent clause owner.
        "6 | To any other person $165 a week\n"
        "6AA | To any other specified person $105 a week\n"
        "6A | The rate is the rate in clauses 1 to 6AA that would apply "
        "if they were single.\n"
        "6B | In clauses 4 to 6AA, specified person means a person receiving a benefit.\n"
    )
    branches = sc.recognize_source_structure(source, corpus_citation_path=NZ_AMENDMENT)
    row = next(branch for branch in branches if branch.path == ("6a",))
    assert "clauses 1 to 6AA" in row.text
    assert "specified person means" not in row.text
    assert not [branch for branch in branches if branch.kind == "sentence"]


def test_explicit_canonical_dependency_keeps_obligation_after_reversion():
    source = "The beneficiary's eligibility for the payment is under section 72."
    reason = f"Cannot be computed until {NZ_ACT}/section/72 is encoded."
    assert not sc._reason_dependency_is_source_bound(
        reason,
        source,
        corpus_citation_path=f"{NZ_ACT}/section/220",
        path=("2", "b"),
    )


@settings(deadline=None)
@given(year=st.integers(1000, 9999), section=st.integers(1, 999))
def test_canonical_corpus_dependency_year_never_binds_as_section(
    year: int, section: int
):
    instrument = f"nz/statute/act/public/{year}/0032"
    reason = f"Cannot be computed until {instrument}/section/{section} is encoded."
    assert not sc._reason_dependency_is_source_bound(
        reason,
        f"The payment depends on section {section}.",
        corpus_citation_path=f"{instrument}/section/1000",
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        f"The payment depends on section {year}.",
        corpus_citation_path=f"{instrument}/section/1000",
    )


def test_canonical_dependency_does_not_bind_another_instrument():
    assert not sc._reason_dependency_is_source_bound(
        "Cannot be computed until nz/statute/act/public/2019/0032/section/72 is encoded.",
        "The payment depends on section 72.",
        corpus_citation_path=f"{NZ_ACT}/section/220",
    )


@pytest.mark.parametrize(
    ("instrument", "descendant"),
    [
        ("de/statute/estg", "absatz-1"),
        ("de/statute/estg", "absatz-1/document-1"),
        ("uk/statute/example", "subsection/1"),
        ("uk/statute/example", "subsection/1/paragraph/a"),
    ],
)
@pytest.mark.parametrize("current_descendant", ["", "/document-1"])
def test_canonical_dependency_preserves_authenticated_descendant_sections(
    instrument: str, descendant: str, current_descendant: str
):
    reason = f"Cannot be computed until {instrument}/72/{descendant} is encoded."
    assert sc._reason_dependency_is_source_bound(
        reason,
        "The payment depends on section 72.",
        corpus_citation_path=f"{instrument}/220{current_descendant}",
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        "The payment depends on section 72.",
        corpus_citation_path=f"{instrument}-other/220{current_descendant}",
    )


@pytest.mark.parametrize("tail", ["2018/0032", "72/unknown-1"])
def test_canonical_dependency_numeric_metadata_does_not_establish_a_section(tail: str):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until uk/statute/example/{tail} is encoded.",
        "The payment depends on section 72 and section 2018.",
        corpus_citation_path="uk/statute/example/220",
    )


@settings(deadline=None)
@given(year=st.integers(1000, 9998))
def test_canonical_current_act_year_does_not_authenticate_an_implicit_section(
    year: int,
):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until uk/statute/ukpga/{year + 1} is encoded.",
        f"The payment depends on section {year + 1}.",
        corpus_citation_path=f"uk/statute/ukpga/{year}/32/section/220",
    )


def test_canonical_dependency_with_sentence_punctuation_is_not_its_own_blocker():
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {NZ_ACT}/section/220.",
        "The payment depends on section 220.",
        corpus_citation_path=f"{NZ_ACT}/section/220",
    )


@pytest.mark.parametrize(
    "dependency", [f"{NZ_ACT}", f"{NZ_ACT}/section", f"{NZ_ACT}/schedule/4"]
)
def test_canonical_dependency_requires_a_section_locator(dependency: str):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {dependency} is encoded.",
        "The payment depends on section 4 and section 2018.",
        corpus_citation_path=f"{NZ_ACT}/section/220",
    )


def test_complete_cross_reference_list_is_excluded_from_numeric_recall():
    assert (
        numeric_values(
            "This section overrides sections 297 , 311(1) , and 338 .",
            f"{NZ_ACT}/section/74",
        )
        == []
    )


def test_cross_reference_list_can_wrap_without_swallowing_a_following_amount():
    assert numeric_values(
        "This section overrides sections\n297 , 311(1) ,\nand 338 . "
        "The amount is $450.",
        f"{NZ_ACT}/section/74",
    ) == [450]


def test_abbreviated_subsection_continuation_stays_in_legal_citation_list():
    assert numeric_values(
        "This section overrides sections 297, 311(1), (2), and 338. "
        "The amount is $450.",
        f"{NZ_ACT}/section/74",
    ) == [450]


@settings(deadline=None)
@given(
    subsection=st.integers(1, 999),
    amount=st.integers(1, 999),
    separator=st.sampled_from([", ", " and ", ", and ", ",\nand "]),
)
def test_abbreviated_legal_list_keeps_policy_values_after_citation(
    subsection: int, amount: int, separator: str
):
    source = (
        "This section overrides sections 297, 311(1)"
        f"{separator}({subsection})(a), and 338. The amount is ${amount}."
    )
    assert numeric_values(source, f"{NZ_ACT}/section/74") == [amount]


@pytest.mark.parametrize("unit", ["dollars", "per week", "euros", "USD"])
@pytest.mark.parametrize("suffix", ["", "(a)", " (a)"])
def test_parenthetical_coordinated_amount_keeps_its_numeric_demand(
    unit: str, suffix: str
):
    assert numeric_values(
        "This section overrides sections 297, 311(1), "
        f"and (450){suffix} {unit} are payable.",
        f"{NZ_ACT}/section/74",
    ) == [450]


def test_parenthetical_list_without_legal_antecedent_stays_numeric():
    assert numeric_values("The payable values are (450), (700), and 338.", "") == [
        450,
        700,
        338,
    ]


@pytest.mark.parametrize("separator", [" and ", ", "])
@pytest.mark.parametrize(
    "unit", ["dollars", "dollars per week", "per week", "euros", "USD", "EUR"]
)
@pytest.mark.parametrize("wrapping", [" ", "\n"])
def test_cross_reference_list_does_not_consume_a_coordinated_policy_amount(
    separator: str, unit: str, wrapping: str
):
    source = (
        "This section overrides sections 297, 311(1), and 338"
        f"{separator}450{wrapping}{unit} are payable."
    )
    assert numeric_values(source, f"{NZ_ACT}/section/74") == [450]


@pytest.mark.parametrize("unit", ["per person", "per week", "dollars", "USD"])
def test_explicitly_named_locator_is_removed_even_when_a_unit_follows(unit: str):
    source = f"The amount is prescribed by section 72 {unit}."
    assert numeric_values(source, f"{NZ_ACT}/section/74") == []


@settings(deadline=None)
@given(sections=st.lists(st.integers(1, 999), min_size=3, max_size=8, unique=True))
def test_legal_citation_list_never_contributes_numeric_demands(sections: list[int]):
    references = [f"{section}(1)(a)" for section in sections]
    source = "This section overrides sections " + ", ".join(references[:-1])
    source += ", and " + references[-1] + ". The amount is $450."
    assert numeric_values(source, f"{NZ_ACT}/section/74") == [450]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "This section overrides sections 297, 311(1), and 338, and 450 a week is payable.",
            [450],
        ),
        (
            "The supplement is the sum of the allowance under clauses 1 to 6AA and 20% of earnings.",
            [0.2, 1],
        ),
    ],
)
def test_citation_cleanup_keeps_review_payment_and_percentage(source, expected):
    assert numeric_values(source, f"{NZ_ACT}/section/74") == expected


@settings(max_examples=100, deadline=None)
@given(
    amount=st.integers(1, 999),
    separator=st.sampled_from([", and ", " and ", ", ", ",\nand "]),
    suffix=st.sampled_from(
        [
            "a week",
            "each week",
            "per calendar month",
            "per annum",
            "a full fiscal year",
            "each full calendar week",
            "a year",
            "every fortnight",
            "per working day",
            "per cent of earnings",
            "% of earnings",
            "£",
            "€",
            "$",
            "dollars",
        ]
    ),
)
@example(amount=450, separator=", and ", suffix="per annum")
@example(amount=450, separator=", and ", suffix="a full fiscal year")
def test_real_amount_after_a_citation_list_survives_cleanup(amount, separator, suffix):
    source = f"The payment under sections 297, 311(1), and 338{separator}{amount} {suffix} is payable."
    expected = numeric_values(f"{amount} {suffix} is payable.", "")
    assert expected
    assert numeric_values(source, f"{NZ_ACT}/section/74") == expected


@pytest.mark.parametrize(
    "caption", ["Family size", "Age", "Number of children", "Years"]
)
def test_count_keyed_table_keeps_counts_and_payment_amounts(caption):
    source = (
        f"{caption} | Eligibility\n"
        "1 | The household is eligible if— (a) | its income is below $450.\n"
        "2 | The household is eligible if— (a) | its income is below $700."
    )
    assert numeric_values(source, f"{NZ_ACT}/section/74") == [1, 450, 2, 700]


@pytest.mark.parametrize("kind", ["article", "regulation"])
def test_unsupported_canonical_locator_kinds_fail_closed(kind):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {NZ_ACT}/{kind}/72 is encoded.",
        f"The payment depends on {kind} 72.",
        corpus_citation_path=f"{NZ_ACT}/{kind}/220",
    )


CITATION_CONTROL_INSTRUMENTS = [
    NZ_ACT,
    "uk/statute/example",
    "us/statute/example",
    "de/statute/example",
    "fr/statute/example",
]


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize(
    ("source", "amount"),
    [
        (
            "This section overrides sections 297, 311(1), and 338, and 450 is payable each week.",
            450,
        ),
        (
            "The supplement is the sum of the allowance under clauses 1 to 6AA and 20 times the weekly amount.",
            20,
        ),
        (
            "The supplement is the sum of the allowance under clauses 1 to 6AA and 20 basis points of earnings.",
            20,
        ),
    ],
)
def test_citation_cleanup_keeps_round_two_operative_quantities(
    instrument: str, source: str, amount: int
):
    from tests.test_source_completeness import (
        EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        _analyze,
    )

    citation = f"{instrument}/section/74"
    assert numeric_values(source, citation) == [amount]
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [
                {
                    "name": "payment",
                    "kind": "derived",
                    "dtype": "Money",
                    "source": citation,
                    "versions": [{"formula": "0"}],
                }
            ],
        }
    )
    result = _analyze(
        content,
        source,
        corpus_citation_path=citation,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    assert any(
        "numeric-recall" in issue and f"value {amount} has" in issue
        for issue in result.issues
    )


@settings(max_examples=100, deadline=None)
@given(
    amount=st.integers(1, 999),
    word=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=16),
)
def test_citation_continuation_never_consumes_a_number_before_a_nonlocator_word(
    amount: int, word: str
):
    source = (
        f"The payment under sections 297, 311(1), and 338, and {amount} {word} applies."
    )
    expected = numeric_values(f"{amount} {word} applies.", "")
    assert expected
    assert numeric_values(source, f"{NZ_ACT}/section/74") == expected


@pytest.mark.parametrize("terminal", [".", ";", ":", ")", ""])
def test_citation_continuation_requires_a_positive_terminal(terminal: str):
    assert (
        numeric_values(
            f"This section overrides sections 297, 311(1), and 338{terminal}",
            f"{NZ_ACT}/section/74",
        )
        == []
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("kind", ["article", "regulation"])
@pytest.mark.parametrize("prose_kind", ["section", "article", "regulation"])
def test_unsupported_locator_never_binds_section_prose_or_defers_a_branch(
    instrument: str, kind: str, prose_kind: str
):
    citation = f"{instrument}/{kind}/220"
    reason = f"Cannot be computed until {instrument}/{kind}/72 is encoded."
    source = f"(1) The payment depends on {prose_kind} 72."
    assert not sc._reason_dependency_is_source_bound(
        reason, source, corpus_citation_path=citation
    )
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": sc._rulespec_target_base(citation) + "/1#payment",
                        "reason": reason,
                    }
                ]
            }
        },
        corpus_citation_path=citation,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=citation),
    )
    assert covered == set()
    assert any("does not name an exact missing dependency" in issue for issue in issues)


@pytest.mark.parametrize("kind", ["article", "regulation"])
def test_unsupported_outer_locator_cannot_be_rescued_by_a_nested_section(kind: str):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {NZ_ACT}/{kind}/72/section/5 is encoded.",
        "The payment depends on section 5.",
        corpus_citation_path=f"{NZ_ACT}/{kind}/72/section/220",
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
def test_embedded_count_table_retains_actual_completeness_obligations(instrument: str):
    from tests.test_source_completeness import (
        EN_NUMERIC_OCCURRENCE_EXTRACTOR,
        _analyze,
    )

    citation = f"{instrument}/section/74"
    source = (
        "Clause | Provision\n"
        "1 | The rates are specified in the following table:\n"
        "Number of children | Eligibility\n"
        "2 | The household is eligible if— (a) | its income is below $450.\n"
        "3 | The household is eligible if— (a) | its income is below $700."
    )
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [
                {
                    "name": "payment",
                    "kind": "derived",
                    "dtype": "Money",
                    "source": citation,
                    "versions": [{"formula": "0"}],
                }
            ],
        }
    )
    result = _analyze(
        content,
        source,
        corpus_citation_path=citation,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    for count in (2, 3):
        assert any(
            "numeric-recall" in issue and f"value {count} has" in issue
            for issue in result.issues
        )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
def test_ambiguous_hyphenated_locator_retains_quantities_without_inventing_topology(
    instrument: str,
):
    source = "Section 40-18-2 shall apply. The payment is $450."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{instrument}/section/74"
    )
    assert "40-18-2" in cleaned
    assert 450 in numeric_values(source, f"{instrument}/section/74")
    assert sc._explicit_source_arithmetic_topology(cleaned) is None


@settings(max_examples=50, deadline=None)
@given(components=st.lists(st.integers(1, 999), min_size=2, max_size=8))
def test_ambiguous_hyphenated_locator_never_leaves_a_partial_arithmetic_tail(
    components,
):
    locator = "-".join(str(component) for component in components)
    source = f"Section {locator} shall apply. The payment is $450."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{NZ_ACT}/section/74"
    )
    assert locator in cleaned
    assert sc._explicit_source_arithmetic_topology(cleaned) is None


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("prose_kind", ["paragraph", "article", "regulation", "clause"])
def test_canonical_section_dependency_requires_the_same_prose_locator_kind(
    instrument: str, prose_kind: str
):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {instrument}/section/72 is encoded.",
        f"The payment depends on {prose_kind} 72.",
        corpus_citation_path=f"{instrument}/section/220",
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("alias", ["section", "§", "§§"])
def test_reverted_explicit_dependency_does_not_guess_from_typed_section_references(
    instrument, alias
):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {instrument}/section/72 is encoded.",
        f"The payment depends on {alias} 72.",
        corpus_citation_path=f"{instrument}/section/220",
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("tail", ["article/73", "regulation/73", "unknown/73"])
@pytest.mark.parametrize("location", ["dependency", "current"])
def test_explicit_canonical_section_uses_supported_descendant_grammar(
    instrument: str, tail: str, location: str
):
    dependency = f"{instrument}/section/72"
    current = f"{instrument}/section/220"
    if location == "dependency":
        dependency += f"/{tail}"
    else:
        current += f"/{tail}"
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {dependency} is encoded.",
        "The payment depends on section 72.",
        corpus_citation_path=current,
    )


@pytest.mark.parametrize(
    "source_kind", ["Articles", "Sections", "Regulations", "Paragraphs"]
)
def test_ambiguous_citation_number_cannot_authenticate_a_conversion_base(source_kind):
    source = f"Under {source_kind} 10 to 12, converted monthly, it is 6150 dollars."
    # Keep the conservative recall demand while refusing to infer that a
    # possible locator is the monetary base of the converted result.
    assert numeric_values(source, f"{NZ_ACT}/section/74") == [12, 6150]
    assert not sc.source_states_stated_conversion_result(source)


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize(
    "section_prose",
    ["Section 72 remains unaffected.", "Section 72 has a heading."],
)
def test_canonical_dependency_kind_and_link_must_belong_to_the_same_reference(
    instrument: str, section_prose: str
):
    source = f"(1) {section_prose} The payment depends on paragraph 72."
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {instrument}/section/72 is encoded.",
        source,
        corpus_citation_path=f"{instrument}/section/220",
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("suffix", ["times the weekly amount", "is payable each week"])
def test_unproven_attached_hyphen_range_retains_the_whole_identifier(
    instrument, suffix
):
    source = f"The payment under sections 1-20 {suffix}."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{instrument}/section/74"
    )
    assert "1-20" in cleaned


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize(
    "locator",
    [
        "72.5",
        "72-5",
        "72,5",
        "72/5",
        "72_5",
        "72.foo",
        "72-foo",
        "72(foo-bar)",
        "72 (5.1)",
        "72-A",
        "72/other",
        "72–5",
        "72:other",
        "72;other",
        "72!other",
    ],
)
def test_canonical_section_dependency_never_binds_a_locator_prefix(instrument, locator):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {instrument}/section/72 is encoded.",
        f"The payment depends on section {locator}.",
        corpus_citation_path=f"{instrument}/section/220",
    )


@pytest.mark.parametrize("locator", ["72", "72.", "72(5)", "72 (5)(a)"])
def test_canonical_section_atom_keeps_terminal_punctuation_and_supported_children(
    locator,
):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {NZ_ACT}/section/72 is encoded.",
        f"The payment depends on section {locator}",
        corpus_citation_path=f"{NZ_ACT}/section/220",
    )


@pytest.mark.parametrize("suffix", [".5", "_x", "/5", "-a"])
def test_parenthetical_locator_fragment_never_cleans_an_unsupported_atom_prefix(suffix):
    source = f"The payment under sections 297, 311(1), and (450){suffix} is payable."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{NZ_ACT}/section/74"
    )
    assert f"(450){suffix}" in cleaned


@settings(max_examples=50, deadline=None)
@given(
    amount=st.integers(1, 999),
    suffix=st.one_of(
        st.integers(1, 999).map(lambda value: f".{value}"),
        st.integers(1, 999).map(lambda value: f"/{value}"),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).map(
            lambda value: f"_{value}"
        ),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).map(
            lambda value: f"-{value}"
        ),
    ),
)
def test_generated_unsupported_parenthetical_atoms_abstain_from_prefix_cleanup(
    amount, suffix
):
    source = (
        f"The payment under sections 297, 311(1), and ({amount}){suffix} is payable."
    )
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{NZ_ACT}/section/74"
    )
    assert f"({amount}){suffix}" in cleaned


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("heading", ["Family size.", "Household size.", "Age."])
def test_detached_count_axes_retain_actual_missing_scalar_obligations(
    instrument, heading
):
    from tests.test_source_completeness import EN_NUMERIC_OCCURRENCE_EXTRACTOR, _analyze

    citation = f"{instrument}/section/220"
    source = f"{heading}\n1\nThe household is entitled to $450.\n2\nThe household is entitled to $700."
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": citation}},
            "rules": [
                {
                    "name": "payment",
                    "kind": "derived",
                    "dtype": "Money",
                    "source": citation,
                    "versions": [{"formula": "0"}],
                }
            ],
        }
    )
    result = _analyze(
        content,
        source,
        corpus_citation_path=citation,
        extract_numeric_occurrences=EN_NUMERIC_OCCURRENCE_EXTRACTOR,
    )
    assert numeric_values(source, citation) == [1, 450, 2, 700]
    for value in (1, 2, 450, 700):
        assert any(
            "numeric-recall" in issue and f"value {value} has" in issue
            for issue in result.issues
        )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize(
    "sentence",
    [
        "Section 72 has a heading and the conditions under paragraph 72 apply.",
        "Section 72 has a heading and conditions under section 73 apply.",
        "Section 72 has a heading and the payment is subject to paragraph 72.",
        "Section 72 has a heading and section 73 is required.",
        "The payment is not subject to section 72.",
        "The payment does not depend on section 72.",
        "No conditions under section 72 apply.",
        "The payment does not require section 72.",
        "The conditions under section 72 are described in a heading and the conditions under paragraph 73 apply.",
        "The conditions under section 72 are described in a heading and the conditions under section 73 apply.",
    ],
)
def test_same_sentence_dependency_cannot_borrow_another_references_predicate(
    instrument, sentence
):
    citation = f"{instrument}/section/220"
    source = "(1) " + sentence
    reason = f"Cannot be computed until {instrument}/section/72 is encoded."
    assert not sc._reason_dependency_is_source_bound(
        reason, source, corpus_citation_path=citation
    )
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": sc._rulespec_target_base(citation) + "/1#payment",
                        "reason": reason,
                    }
                ]
            }
        },
        corpus_citation_path=citation,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=citation),
    )
    assert not covered
    assert len(issues) == 1
    assert "does not name an exact missing dependency" in issues[0]


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize(
    "sentence",
    [
        "Section 73 has a heading and the conditions under section 72 apply.",
        "Section 73 has a heading and the payment depends on section 72.",
        "The payment is subject to section 72.",
        "The beneficiary’s eligibility for the payment under section 72.",
        "The beneficiary's eligibility for the payment is under section 72.",
        "Section 72 is required and section 73 has a heading.",
    ],
)
def test_explicit_canonical_dependency_relaxation_abstains_even_on_positive_prose(
    instrument, sentence
):
    assert not sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {instrument}/section/72 is encoded.",
        sentence,
        corpus_citation_path=f"{instrument}/section/220",
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("separator", [", ", " and ", ", and "])
@pytest.mark.parametrize("locator", ["40-18-2", "40 - 18 - 2"])
def test_partial_citation_list_preserves_percentage_formula_witness(
    instrument, separator, locator
):
    from tests.test_source_completeness import EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR

    source = f"The payment is 20% of earnings under sections 1{separator}{locator}."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{instrument}/section/74"
    )
    assert f"sections 1{separator}{locator}" in cleaned
    assert sc._explicit_source_arithmetic_topology(cleaned) is None
    branch = sc.SourceStructureBranch(
        (), "formula-clause", "source unit", source, 0, len(source)
    )
    execution = sc._FormulaExecution(
        trace=(),
        leaf="earnings * 0.2",
        evaluated_value=None,
        evaluates_to_zero=False,
        constant_environment={},
    )
    assert sc._formula_execution_matches_source_branch(
        execution,
        branch,
        interval=None,
        formula_environment={},
        execution_environment={"earnings": 1000},
        extract_numeric_occurrences=EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )


@pytest.mark.parametrize("instrument", CITATION_CONTROL_INSTRUMENTS)
@pytest.mark.parametrize("separator", [" ", "\u00a0", "\u202f"])
@pytest.mark.parametrize(
    "prefix",
    [
        "The payment under sections 297, 311(1), and 338, and ",
        "The supplement is the sum of the allowance under clauses 1 to 6AA and ",
    ],
)
def test_citation_cleanup_keeps_spaced_amount_whole(instrument, separator, prefix):
    source = prefix + f"450{separator}700 dollars is payable."
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=f"{instrument}/section/74"
    )
    assert f"450{separator}700 dollars" in cleaned
    assert numeric_values(source, f"{instrument}/section/74")[-2:] == [450, 700]


@pytest.mark.parametrize("other", ["paragraph 72", "section 73"])
def test_clause_wide_dependency_fallback_cannot_borrow_a_different_reference(other):
    clause = f"Section 72 has a heading and the conditions under {other} apply."
    assert not sc._source_clause_links_dependency(
        clause, reference_start=0, reference_end=len("Section 72")
    )


@pytest.mark.parametrize(
    ("clause", "reference"),
    [
        ("Die Voraussetzungen nach § 26 und § 32 vorliegen", "§ 32"),
        ("The conditions under section 26 and section 32 apply", "section 32"),
    ],
)
def test_coordinated_condition_predicate_owns_each_explicit_reference(
    clause, reference
):
    start = clause.index(reference)
    assert sc._source_clause_links_dependency(
        clause, reference_start=start, reference_end=start + len(reference)
    )


@pytest.mark.parametrize(
    "kind", ["sections", "clauses", "articles", "regulations", "paragraphs"]
)
@pytest.mark.parametrize("amount", [2, 20, 25, 2.5, 20.5])
def test_possible_citation_formula_mask_retains_independent_percentage(kind, amount):
    from tests.test_source_completeness import EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR

    source = f"The supplement is the sum of the allowance under {kind} 1 to 6 and {amount}% of earnings."
    assert sc._formula_operation_kinds(source) == {"add", "multiply"}
    branch = sc.SourceStructureBranch(
        (), "formula-clause", "source unit", source, 0, len(source)
    )
    for formula, expected in [
        (f"allowance + earnings * {amount / 100}", True),
        (f"allowance + {amount / 100}", False),
        ("allowance + 450", False),
    ]:
        execution = sc._FormulaExecution(
            trace=(),
            leaf=formula,
            evaluated_value=None,
            evaluates_to_zero=False,
            constant_environment={},
        )
        assert (
            sc._formula_execution_matches_source_branch(
                execution,
                branch,
                interval=None,
                formula_environment={},
                execution_environment={"allowance": 100, "earnings": 1000},
                extract_numeric_occurrences=EN_NUMERIC_GROUNDING_OCCURRENCE_EXTRACTOR,
                numeric_value_is_grounded=vp.numeric_value_is_grounded,
            )
            is expected
        )
