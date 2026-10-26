"""NZ report regressions for structure-aware coverage and numeric parsing.

Excerpts: Social Security Act 2018 (2018/32); amending Act 2026/54, section 15.
Source: New Zealand Legislation (legislation.govt.nz), Crown copyright, CC BY 4.0
"""

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as vp

TABLE_CITATION = "nz/statute/act/public/2018/0032/schedule/4/part/8/clause/lms118454"


@pytest.mark.parametrize("suffix", ["(a)", "(a) (b)x"])
def test_canonical_paragraph_fallback_retains_base_coverage_obligation(suffix):
    # Keep base's demand even for the formerly admitted exact citation.
    # A spaced unsupported continuation must not gain broader admission.
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": TABLE_CITATION}},
        "rules": [
            {
                "name": name,
                "kind": "derived",
                "source": TABLE_CITATION + suffix,
                "versions": [{"effective_from": "2018-01-01", "formula": "450"}],
            }
            for name in (
                "single_without_children_rate",
                "winter_energy_payment_per_winter_period",
            )
        ],
    }
    assert vp._source_subparagraph_citation_pattern(TABLE_CITATION) is None
    assert vp._rule_source_subparagraph_paths(payload, TABLE_CITATION) == set()
    issues = vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump(payload),
        source_texts={TABLE_CITATION: "(a) A person without children is eligible."},
    )
    assert len(issues) == 1
    assert TABLE_CITATION + "(a)" in issues[0]


@settings(deadline=None)
@given(
    jurisdiction=st.sampled_from(["nz", "uk", "de", "fr", "za", "gh"]),
    label=st.sampled_from(["a", "b", "c", "1", "2", "A", "ii"]),
)
def test_canonical_citation_expansion_is_deferred_to_keep_base_obligations(
    jurisdiction, label
):
    citation = f"{jurisdiction}/statute/example/section/72"
    expected = set()
    parse = vp._source_citation_subparagraph_paths
    assert parse(f"{citation}({label})", citation) == expected
    assert parse(f"{citation}0({label})", citation) == set()
    assert parse(f"other/{citation}({label})", citation) == set()
    assert parse(f"{citation}/other({label})", citation) == set()


@pytest.mark.parametrize(
    "source",
    [
        "4\nAn instalment must be paid on the relevant payment date.",
        "4 An instalment must be paid on the relevant payment date.",
        "4\nAn eligible person is entitled to payment.\n5\nAn instalment is due.",
    ],
)
def test_english_an_conversion_retains_base_behavior_pending_separate_fix(source):
    assert any(
        "An" in occurrence.raw
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )
    assert 48 in vp._extract_year_duration_month_values(source)


@settings(deadline=None)
@given(label=st.integers(min_value=1, max_value=999))
def test_english_subsection_labels_keep_base_conversion_until_followup(label):
    source = f"{label}\nAn instalment must be paid on the relevant payment date."
    assert vp._extract_year_duration_month_values(source) == {label * 12}
    assert any(
        occurrence.raw == f"{label}\nAn"
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )


@pytest.mark.parametrize(
    ("source", "months"),
    [
        ("Cette prolongation ne peut depasser quinze ans.", 180),
        ("La durée est de 4 ans.", 48),
        ("La durée est de 4\nans.", 48),
        ("La durée est de 1 an.", 12),
        ("Le délai ne dépasse pas 1 an.", 12),
        ("Les délais ne dépassent pas 1 an.", 12),
        ("La durée ne dépassera pas 1 an.", 12),
        ("La durée ne dépassait pas 1 an.", 12),
        ("Le délai ne peut excéder 1 an.", 12),
        ("Le délai n’excède pas 1 an.", 12),
        ("Le délai a dépassé 1 an.", 12),
        ("Les durées ont excédé 1 an.", 12),
        ("La durée ne dépasserait pas 1 an.", 12),
        ("La durée est de\n1 an.", 12),
        ("La protection est garantie pendant 1 an.", 12),
        ("Un délai de 1 an est prévu.", 12),
        ("Le délai expire après 1 an.", 12),
        ("Il est âgé de 1 an.", 12),
        ("Le paiement est dû pour 1 an.", 12),
        ("Le contrat de 1 an prend effet ce jour.", 12),
        ("Une garantie de 1 an couvre le produit.", 12),
        ("Durée: 1 an", 12),
        ("1 an est prévu pour la garantie.", 12),
        ("La durée est de deux ans.", 24),
        ("De leeftijdsvoorwaarde bedraagt vijfentwintig jaar.", 300),
    ],
)
def test_local_french_and_dutch_year_durations_keep_month_conversion(source, months):
    assert months in vp.extract_numeric_occurrences_from_text(source)


def test_mixed_table_cells_preserve_ordinary_paragraph_coverage():
    source = (
        "In Schedule 4, replace clauses 1 and 2 with:\n"
        "1 | A single person is eligible for payment as follows:\n"
        "(a) A single person is eligible for $450.\n"
        "(b) A sole parent is eligible for $700.\n"
        "2 | A dependent child means a person supported by a parent."
    )
    demands = vp._high_signal_top_level_coverage_demands(source)
    assert [demand.label for demand in demands] == ["a", "b"]
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": TABLE_CITATION}},
        "rules": [],
    }
    issues = vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump(payload), source_texts={TABLE_CITATION: source}
    )
    assert len(issues) == 2
    assert any(f"{TABLE_CITATION}(a)" in issue for issue in issues)
    assert any(f"{TABLE_CITATION}(b)" in issue for issue in issues)


@pytest.mark.parametrize(
    "source",
    [
        "4\nAn applicant resident in La Paz is eligible.",
        "4\nAn applicant wears a pendant and is eligible.",
        "A pendant is worth 4\nAn applicant is eligible.",
    ],
)
def test_incidental_english_vocabulary_keeps_base_an_conversion(source):
    assert vp._extract_year_duration_month_values(source) == {48}
    assert any(
        occurrence.value == 48
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )


def test_exact_canonical_paragraph_citation_never_matches_a_malformed_prefix():
    assert (
        vp._source_citation_subparagraph_paths(
            TABLE_CITATION + "(a)(b)x", TABLE_CITATION
        )
        == set()
    )


CANONICAL_BOUNDARY_CITATIONS = [
    TABLE_CITATION,
    "uk/statute/example/section/72",
    "us/guidance/example/section/72",
    "de/statute/example/section/72",
    "fr/statute/example/section/72",
]


@pytest.mark.parametrize("citation", CANONICAL_BOUNDARY_CITATIONS)
@pytest.mark.parametrize("continuation", [".5", "-other", ":other"])
def test_attached_canonical_suffix_retains_missing_paragraph_coverage(
    citation, continuation
):
    # The exact review input uses TABLE_CITATION + "(a).5". Its valid prefix
    # must not discharge the independent source paragraph's coverage demand.
    source = "(a) A person without children is eligible."
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "rules": [
            {
                "name": "single_without_children_rate",
                "kind": "derived",
                "source": citation + "(a)" + continuation,
                "versions": [{"effective_from": "2018-01-01", "formula": "450"}],
            }
        ],
    }
    assert vp._rule_source_subparagraph_paths(payload, citation) == set()
    issues = vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump(payload), source_texts={citation: source}
    )
    assert len(issues) == 1
    assert f"{citation}(a)" in issues[0]


@pytest.mark.parametrize("citation", CANONICAL_BOUNDARY_CITATIONS)
@pytest.mark.parametrize("terminal", ["", " ", "\n", ".", ". Text", ", text"])
def test_complete_canonical_atom_keeps_base_coverage_demand(citation, terminal):
    assert (
        vp._source_citation_subparagraph_paths(citation + "(a)" + terminal, citation)
        == set()
    )


@pytest.mark.parametrize(
    "source",
    [
        "Son contrat dure 1 an.",
        "Votre contrat dure 1 an.",
        "Notre garantie dure 1 an.",
        "La durée maximale : 1 an.",
        "Mon contrat dure 1 an.",
        "Ton contrat dure 1 an.",
        "Leur garantie dure 1 an.",
        "Le délai minimal : 1 an.",
    ],
)
def test_french_possessive_and_duration_labels_keep_singular_years(source):
    assert vp._extract_year_duration_month_values(source) == {12}
    assert any(
        occurrence.raw == "1 an"
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )


def test_singular_french_year_has_both_inventory_and_conversion():
    source = "Le délai ne dépasse pas 1 an."
    assert vp._extract_year_duration_month_values(source) == {12}
    assert any(
        occurrence.raw == "1 an"
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )


@pytest.mark.parametrize(
    "source",
    [
        "Notre garantie de 1 an couvre le produit.",
        "Votre contrat de 1 an prend effet ce jour.",
        "La durée légale : 1 an.",
    ],
)
def test_french_noun_phrase_and_label_forms_keep_base_month_conversion(
    source,
):
    # The reverted language gate must not delete genuine year obligations.
    assert vp._extract_year_duration_month_values(source) == {12}
    assert 12 in vp.extract_numeric_occurrences_from_text(source)
