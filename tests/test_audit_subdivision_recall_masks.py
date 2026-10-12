"""The audit independently rejects operative numeric removals."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from axiom_encode.harness import source_completeness as completeness
from scripts.audit_subdivision_recall_masks import (
    entry_removals,
    independent_identifier_context,
    project_recall_removals,
    require_inventory_projections,
)


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2 or 20 or 30 percent of income.",
        "Under subdivision 3 or more eligible children qualify.",
        "Under subdivision 20 of every 100 applicants qualify.",
        "Under subdivision 3–5 of the taxpayer’s children qualify.",
        "Under subdivision 20.5 percent of income.",
        "The benefit is $48 contiguous states.",
        "The benefit is $ 48 contiguous states.",
        "The benefit is $\n48 contiguous states.",
        "The benefit is $\r\n48 contiguous states.",
        "The benefit is ₩48 contiguous states.",
        "The benefit is USD 48 contiguous states.",
        "The benefit is usd 48 contiguous states.",
        "The benefit is yuan 48 contiguous states.",
        "A heading\n48 contiguous states.",
        "For 48 contiguous states.",
        "The benefit is 148 contiguous states.",
        "The benefit is a48 contiguous states.",
        "The benefit is 48 contiguous stateside offices.",
    ),
)
def test_audit_does_not_classify_operative_values_as_identifiers(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 20, 30 percent of income.",
        "Under subdivisions 2, 20 and 30 children qualify.",
    ),
)
def test_intermediate_commas_do_not_prove_coordinated_quantities_are_locators(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 20, or 30 percent of income, the credit is $300.",
        "A land subdivision 3, or more acres, qualifies.",
    ),
)
def test_comma_fallback_cannot_prove_an_ambiguous_quantity_is_a_locator(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, 3 or more eligible children must qualify.",
        "Under subdivision 2, 20 of every 100 applicants qualify.",
        "Under subdivision 2, 3–5 of the taxpayer’s children must qualify.",
        "Under subdivision 2, the following applies: 20 percent of income.",
    ),
)
def test_a_complete_quantity_clause_independently_proves_the_locator_boundary(source):
    for match in re.finditer(r"\d+", source):
        if match.start() == source.index("2"):
            assert independent_identifier_context(source, *match.span()) is not None
        else:
            assert independent_identifier_context(source, *match.span()) is None


def test_a_quoted_term_definition_independently_proves_the_locator_boundary():
    source = (
        'For purposes of subdivision 15, "business interest" means the same as '
        "that term is defined under § 163(j) of the Internal Revenue Code."
    )
    start = source.index("15")
    assert independent_identifier_context(source, start, start + 2) is not None


@pytest.mark.parametrize("term", ('"20 percent"', "“20 percent”", "'20 percent'"))
def test_a_numeric_quoted_term_cannot_prove_a_locator_boundary(term):
    source = f"Under subdivision 2, {term} means the applicable rate."
    start = source.index("2")
    assert independent_identifier_context(source, start, start + 1) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 20 of this section’s 100 applicants qualify.",
        "Under subdivision 20 of this section's 100 applicants qualify.",
        "Under subdivision 20 of section 100 applicants qualify.",
        "Under subdivision 3m, the credit applies.",
        "Under subdivision 3 m of this section, the credit applies.",
        "Under subdivision 20 is greater than the threshold.",
        "A land subdivision 3 m is required.",
        "A land subdivision 3m.",
        "Under subdivisions 2, 20 and 30 children qualify.",
        "Under subdivision 20 hereof’s 100 applicants qualify.",
        "Under subdivision 20 for the same taxable year’s 100 applicants qualify.",
        "Under subdivision 3–5.",
        "Under subdivisions 3–5.",
    ),
)
def test_quantity_contexts_cannot_complete_the_audit_locator_proof(source):
    for match in re.finditer(r"\d+", source):
        assert independent_identifier_context(source, *match.span()) is None


def test_possessive_object_does_not_prove_a_bare_coordinated_value_is_a_locator():
    source = "Under subdivision 2, 20 of this section's 100 applicants qualify."
    for match in re.finditer(r"\d+", source):
        if match.group() != "2":
            assert independent_identifier_context(source, *match.span()) is None


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, the credit is $300.",
        "Under subdivision B 3 of this section.",
        "Under subdivisions 2, 3, and 5 of this section.",
        "Under subdivision 1 or 2 for the same taxable year.",
        "Under subdivision 2(b) of this section.",
        "Under subdivision 2(b).",
        "Under subdivision B3 is required for qualifying taxpayers.",
        "Under subdivisions 3 through 5 of this section.",
        "The guidelines for the 48 contiguous states apply.",
        "48 contiguous states.",
        "\n 48 contiguous states.",
    ),
)
def test_audit_classifies_complete_identifiers(source):
    for match in re.finditer(r"\d+", source):
        if match.group() == "300":
            continue
        assert independent_identifier_context(source, *match.span()) is not None


def test_entry_removal_audit_preserves_offsets_and_nonnumeric_text():
    source = "Under subdivision 2, the following applies: 20 percent."
    masked = "Under subdivision  , the following applies: 20 percent."
    removals = entry_removals(source, masked)
    assert len(removals) == 1
    assert removals[0]["raw"] == "2"
    assert removals[0]["kind"] == "legal-locator"
    with pytest.raises(ValueError, match="offsets"):
        entry_removals(source, masked[:-1])
    with pytest.raises(ValueError, match="nonnumeric"):
        entry_removals(source, masked.replace("applies", "appliex"))


def test_entry_removal_audit_rejects_deletion_of_operative_rate():
    source = "Under subdivision 2 or 20.5 percent of income."
    with pytest.raises(ValueError, match="Unclassified removal"):
        entry_removals(source, source.replace("20.5", "  . "))


def _citation_projection_example():
    source = (
        "Under § 58.1-1, the credit under subdivision 8 of § 58.1-322.02; "
        "8 eligible children qualify for a $300 credit."
    )
    start = source.index("8 of")
    masked = source[:start] + " " + source[start + 1 :]

    def inherited_citation_cleanup(text, *, corpus_citation_path):
        return completeness._GERMAN_LEGAL_CITATION.sub("", text)

    baseline = SimpleNamespace(
        authoritative_numeric_recall_text=inherited_citation_cleanup
    )
    before = inherited_citation_cleanup(source, corpus_citation_path="")
    after = inherited_citation_cleanup(masked, corpus_citation_path="")
    return source, baseline, before, after, entry_removals(source, masked)


def test_removed_section_object_keeps_exact_original_identifier_proof():
    source, baseline, before, after, removals = _citation_projection_example()
    start = before.index("8 of")
    assert independent_identifier_context(before, start, start + 1) is None
    projections = project_recall_removals(baseline, source, "", before, after, removals)
    assert len(projections) == 1
    assert projections[0]["original_start"] == source.index("8 of")
    assert projections[0]["start"] == start
    assert projections[0]["original_start"] != projections[0]["start"]
    require_inventory_projections(
        [{"start": start, "end": start + 1, "raw": "8"}], projections
    )


def test_equal_valued_operative_occurrence_cannot_borrow_identifier_proof():
    source, baseline, before, after, removals = _citation_projection_example()
    projections = project_recall_removals(baseline, source, "", before, after, removals)
    start = before.index("8 eligible")
    with pytest.raises(ValueError, match="exact original-span proof"):
        require_inventory_projections(
            [{"start": start, "end": start + 1, "raw": "8"}], projections
        )


def test_erasing_an_equal_valued_quantity_cannot_pass_recall_projection():
    source, baseline, before, _after, removals = _citation_projection_example()
    start = before.index("8 eligible")
    wrong_recall = before[:start] + " " + before[start + 1 :]
    with pytest.raises(ValueError, match="changes another token"):
        project_recall_removals(baseline, source, "", before, wrong_recall, removals)
