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


def test_authenticated_canonical_paragraph_citations_satisfy_coverage():
    # The report's rejected candidate already cites the exact demanded (a)
    # on both of these rules; the citation parser used to return None.
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": TABLE_CITATION}},
        "rules": [
            {
                "name": name,
                "kind": "derived",
                "source": TABLE_CITATION + "(a)",
                "versions": [{"effective_from": "2018-01-01", "formula": "450"}],
            }
            for name in (
                "single_without_children_rate",
                "winter_energy_payment_per_winter_period",
            )
        ],
    }
    assert vp._source_subparagraph_citation_pattern(TABLE_CITATION) is not None
    assert vp._rule_source_subparagraph_paths(payload, TABLE_CITATION) == {("a",)}
    assert (
        vp.find_source_subparagraph_coverage_issues(
            yaml.safe_dump(payload),
            source_texts={TABLE_CITATION: "(a) A person without children is eligible."},
        )
        == []
    )


@settings(deadline=None)
@given(
    jurisdiction=st.sampled_from(["nz", "uk", "de", "fr", "za", "gh"]),
    label=st.sampled_from(["a", "b", "c", "1", "2", "A", "ii"]),
)
def test_exact_canonical_citations_bind_only_the_authenticated_identity(
    jurisdiction, label
):
    citation = f"{jurisdiction}/statute/example/section/72"
    expected = {(label.lower(),)}
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
def test_english_subsection_and_article_are_not_a_french_year_duration(source):
    assert all(
        "An" not in occurrence.raw
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )
    assert vp._extract_year_duration_month_values(source) == set()


@settings(deadline=None)
@given(label=st.integers(min_value=1, max_value=999))
def test_english_subsection_labels_never_generate_duration_conversions(label):
    source = f"{label}\nAn instalment must be paid on the relevant payment date."
    assert vp._extract_year_duration_month_values(source) == set()
    assert not any(
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


# Synthetic clause caption authenticates this positive layout control.
FLATTENED_RATE_TABLE = (
    "Clause | Provision\n1 | The rates of winter energy payment are as follows:\n"
    "(a) | to a person who is single and has no dependent children | $ 450 per winter period "
    "(b) | to a person who is single and has 1 or more dependent children | $ 700 per winter period "
    "(c) | to a person who is in a relationship, and who is the 1 spouse or partner "
    "who is entitled under section 72(3) | $ 700 per winter period "
    "2 | For the purposes of clause 1(a) and (b), dependent child means any other person who— "
    "(a) | is a child; and (b) | may be, but need not be, a child in respect of whom "
    "payments are made under section 363."
)


def test_flattened_table_coverage_preserves_row_and_enclosing_clause_boundaries():
    segments = vp._top_level_source_subparagraph_segments(FLATTENED_RATE_TABLE)
    assert [label for label, _ in segments] == ["a", "b", "c", "a", "b"]
    assert segments[0][1] == (
        "to a person who is single and has no dependent children | $ 450 per winter period"
    )
    assert all("dependent child means" not in body for _, body in segments)
    assert vp._high_signal_top_level_coverage_demands(FLATTENED_RATE_TABLE) == []


@settings(deadline=None)
@given(amount=st.integers(min_value=1, max_value=100_000))
def test_flattened_coverage_rows_never_acquire_a_sibling_definition(amount):
    source = (
        f"Clause | Provision\n1 | The rates of payment are as follows: (a) | A person receives $ {amount}. "
        "(b) | A second person receives $ 700. "
        "2 | For the purposes of clause 1, dependent child means another person."
    )
    segments = vp._top_level_source_subparagraph_segments(source)
    assert len(segments) == 2
    assert segments[0][1] == f"A person receives $ {amount}."
    assert segments[1][1] == "A second person receives $ 700."
    assert vp._high_signal_top_level_coverage_demands(source) == []


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


def test_flattened_high_signal_demands_retain_their_own_physical_marker():
    source = (
        "Clause | Provision\n1 | The rates of payment are as follows: "
        "(a) | A single person is eligible for $ 450. "
        "(b) | A sole parent is eligible for $ 700. "
        "2 | For the purposes of clause 1, dependent child means another person."
    )
    demands = vp._high_signal_top_level_coverage_demands(source)
    assert [
        (demand.label, demand.body, demand.physical_start) for demand in demands
    ] == [
        ("a", "A single person is eligible for $ 450.", source.index("(a)")),
        ("b", "A sole parent is eligible for $ 700.", source.index("(b)")),
    ]


def test_repeated_table_paragraph_citation_covers_only_its_own_clause():
    source = (
        "Clause | Provision\n1 | The rates of payment are as follows: "
        "(a) | A single person is eligible for $ 450. "
        "(b) | A sole parent receives $ 700. "
        "2 | For the purposes of clause 1, the definition is as follows: "
        "(a) | A child means a person supported by a parent. "
        "(b) | A parent means a person maintaining a child."
    )
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": TABLE_CITATION}},
        "rules": [
            {
                "name": "single_without_children_rate",
                "kind": "derived",
                "source": TABLE_CITATION + "(a)",
                "versions": [{"effective_from": "2018-01-01", "formula": "450"}],
            }
        ],
    }
    issues = vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump(payload), source_texts={TABLE_CITATION: source}
    )
    assert len(issues) == 2
    assert any(f"{TABLE_CITATION}(2)(a)" in issue for issue in issues)
    assert any(f"{TABLE_CITATION}(2)(b)" in issue for issue in issues)
    payload["rules"].append(
        {
            "name": "dependent_child_definition",
            "kind": "derived",
            "source": TABLE_CITATION + "(2)",
            "versions": [{"effective_from": "2018-01-01", "formula": "true"}],
        }
    )
    assert (
        len(
            vp.find_source_subparagraph_coverage_issues(
                yaml.safe_dump(payload), source_texts={TABLE_CITATION: source}
            )
        )
        == 2
    )
    payload["rules"][-1]["source"] = TABLE_CITATION + "(2)(a)(i)"
    payload["rules"].append(
        {
            "name": "dependent_parent_definition",
            "kind": "derived",
            "source": TABLE_CITATION + "(2)(b)(i)",
            "versions": [{"effective_from": "2018-01-01", "formula": "true"}],
        }
    )
    assert (
        vp.find_source_subparagraph_coverage_issues(
            yaml.safe_dump(payload), source_texts={TABLE_CITATION: source}
        )
        == []
    )


@pytest.mark.parametrize(
    "citation",
    [
        "nz/statute/act/public/2026/0054/section/15",
        "nz/statute/act/public/2026/0054/schedule/4/clause/5",
    ],
)
def test_table_source_does_not_drop_an_authenticated_numeric_subsection(citation):
    source = (
        "Clause | Provision\n1 | The rates of payment are as follows: "
        "(a) | A single person is eligible for $ 450. "
        "(b) | A sole parent receives $ 700. "
        "2 | For the purposes of subsection 1, a dependent child means another person."
    )
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "rules": [
            {
                "name": "single_without_children_rate",
                "kind": "derived",
                "source": citation + "(a)",
                "versions": [{"effective_from": "2026-01-01", "formula": "450"}],
            }
        ],
    }
    issues = vp.find_source_subparagraph_coverage_issues(
        yaml.safe_dump(payload), source_texts={citation: source}
    )
    assert len(issues) == 1
    assert f"{citation}(1)(a)" in issues[0]
    payload["rules"][0]["source"] = citation + "(1)(a)"
    assert (
        vp.find_source_subparagraph_coverage_issues(
            yaml.safe_dump(payload), source_texts={citation: source}
        )
        == []
    )


@pytest.mark.parametrize(
    "source",
    [
        "4\nAn applicant resident in La Paz is eligible.",
        "4\nAn applicant wears a pendant and is eligible.",
        "A pendant is worth 4\nAn applicant is eligible.",
    ],
)
def test_french_duration_language_evidence_is_not_incidental_english_vocabulary(source):
    assert vp._extract_year_duration_month_values(source) == set()
    assert not any(
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


def test_singular_french_year_has_both_inventory_and_conversion():
    source = "Le délai ne dépasse pas 1 an."
    assert vp._extract_year_duration_month_values(source) == {12}
    assert any(
        occurrence.raw == "1 an"
        for occurrence in vp.extract_typed_numeric_inventory_occurrences_from_text(
            source
        )
    )
