"""U.S. source structure recognition and English branch labels.

The flattened U.S. Code text puts each subsection on one line with run-in
paragraph, subparagraph, clause and subclause markers. Before this change the
recognizer missed markers after headings and list joiners, so the remaining
markers landed at the wrong depth (26 U.S.C. 32(c)(1)(B) was reported as
`(c)(i)(b)`), and a model citing the real designation could never cover it.
U.S. repair feedback also used German Absatz/Nummer/Buchstabe labels.
"""

from __future__ import annotations

import functools

from axiom_encode.cli import _validation_retry_branch_locator
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

USC_32 = "us/statute/26/32"

# Shortened from the 26 U.S.C. 32 corpus text (uscode.house.gov flattening):
# one line per subsection, run-in markers after headings, dashes and joiners,
# and cross references that are not branches.
FLATTENED_USC_SOURCE = (
    "(a) Allowance of credit (1) In general In the case of an eligible "
    "individual, there shall be allowed as a credit an amount equal to the "
    "credit percentage of the earned income amount. (2) Limitation The amount "
    "of the credit allowable to a taxpayer under paragraph (1) shall not "
    "exceed the excess (if any) of— (A) the credit percentage of the earned "
    "income amount, over (B) the phaseout percentage of so much of the "
    "adjusted gross income as exceeds the phaseout amount.\n\n"
    "(b) Percentages and amounts For purposes of subsection (a)— (1) "
    "Percentages The credit percentage is 34 percent. (2) Amounts The earned "
    "income amount is $6,330.\n\n"
    "(c) Definitions and special rules For purposes of this section— (1) "
    "Eligible individual (A) In general The term “eligible individual” "
    "means— (i) any individual who has a qualifying child for the taxable "
    "year, or (ii) any other individual who does not have a qualifying child "
    "for the taxable year, if— (I) such individual’s principal place of abode "
    "is in the United States for more than one-half of such taxable year, and "
    "(II) such individual has attained age 25 but not attained age 65 before "
    "the close of the taxable year. (B) Limitation on eligibility of "
    "nonresident aliens The term “eligible individual” shall not include any "
    "nonresident alien individual unless treated as a resident by reason of an "
    "election under subsection (g) or (h) of section 6013. (2) Earned income "
    "For purposes of clauses (i) and (ii) of subparagraph (A) of paragraph "
    "(1), the term “earned income” means wages.\n\n"
    "[(d) Repealed. Pub. L. 111–226, title II, § 219(a)(2) , Aug. 10, 2010 , "
    "124 Stat. 2403 ]\n\n"
    "(e) Denial of credit (1) In general No credit shall be allowed if the "
    "disqualified income exceeds $10,000. (2) Disqualified income The term "
    "means— (A) interest determined under paragraph (5)(D), (B) dividends, "
    "and (C) amounts described in subparagraphs (A), (B), and (C) of "
    "paragraph (6)."
)

EXPECTED_USC_PATHS = {
    ("a",),
    ("a", "1"),
    ("a", "2"),
    ("a", "2", "a"),
    ("a", "2", "b"),
    ("b",),
    ("b", "1"),
    ("b", "2"),
    ("c",),
    ("c", "1"),
    ("c", "1", "a"),
    ("c", "1", "a", "i"),
    ("c", "1", "a", "ii"),
    ("c", "1", "a", "ii", "i"),
    ("c", "1", "a", "ii", "ii"),
    ("c", "1", "b"),
    ("c", "2"),
    ("e",),
    ("e", "1"),
    ("e", "2"),
    ("e", "2", "a"),
    ("e", "2", "b"),
    ("e", "2", "c"),
}


def _analyze(content: str, source: str, citation: str):
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _structure_issues(result) -> list[str]:
    return [
        issue
        for issue in result.issues
        if issue.startswith("[complete-source-unit:structure]")
    ]


def _module(citation: str, *rule_sources: str) -> str:
    rules = "".join(
        f"""\
  - name: rule_{index}
    kind: derived
    dtype: Judgment
    source: {source}
    versions:
      - effective_from: '2025-01-01'
        formula: some_input_{index}
"""
        for index, source in enumerate(rule_sources)
    )
    return f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
{rules}"""


def test_flattened_us_code_restores_run_in_outline_at_true_depth():
    branches = recognize_source_structure(
        FLATTENED_USC_SOURCE, corpus_citation_path=USC_32
    )

    assert {branch.path for branch in branches} == EXPECTED_USC_PATHS
    by_path = {branch.path: branch for branch in branches}
    # (c)(1)(B) is no longer reported as `(c)(i)(b)`.
    assert by_path[("c", "1", "b")].text.startswith("(B) Limitation on eligibility")
    assert by_path[("c", "1", "a", "ii", "i")].display_path == (
        "c",
        "1",
        "A",
        "ii",
        "I",
    )
    # Branch text is read from the authoritative source, never the outline
    # (less the opening bracket of a following `[(d) Repealed ...]` note).
    for branch in branches:
        assert (
            branch.text
            == FLATTENED_USC_SOURCE[branch.start : branch.end]
            .strip()
            .removesuffix("[")
            .rstrip()
        )
    assert by_path[("c",)].text.endswith("the term “earned income” means wages.")


def test_us_code_cross_references_and_tombstones_are_not_branches():
    branches = recognize_source_structure(
        FLATTENED_USC_SOURCE, corpus_citation_path=USC_32
    )
    starts = {branch.start for branch in branches}

    for reference in (
        "under paragraph (1)",
        "subsection (g) or (h) of section 6013",
        "clauses (i) and (ii) of subparagraph (A) of paragraph (1)",
        "subparagraphs (A), (B), and (C) of paragraph (6)",
        "under paragraph (5)(D)",
    ):
        offset = FLATTENED_USC_SOURCE.index(reference)
        assert not any(offset <= start < offset + len(reference) for start in starts), (
            reference
        )
    # The bracketed `[(d) Repealed ...]` note keeps the sequence (the next
    # `(e)` is subsection (e)) but is still not a branch of its own.
    assert ("d",) not in {branch.path for branch in branches}


def test_us_code_item_after_a_cited_designation_is_still_a_branch():
    # `(5)(D), (B) dividends` continues the open (A)/(B)/(C) list; it is not a
    # continuation of the citation `paragraph (5)(D)`.
    branches = recognize_source_structure(
        FLATTENED_USC_SOURCE, corpus_citation_path=USC_32
    )
    item = next(branch for branch in branches if branch.path == ("e", "2", "b"))
    assert item.text.startswith("(B) dividends")


def test_us_code_correct_citations_cover_restored_branches():
    covered_sources = [
        f"26 USC 32{''.join(f'({part})' for part in path)}"
        for path in (
            ("a", "1"),
            ("a", "2", "A"),
            ("a", "2", "B"),
            ("c", "1", "A", "i"),
            ("c", "1", "A", "ii", "I"),
            ("c", "1", "A", "ii", "II"),
            ("c", "1", "B"),
        )
    ]
    result = _analyze(_module(USC_32, *covered_sources), FLATTENED_USC_SOURCE, USC_32)
    structure = _structure_issues(result)

    for covered in (
        "us/statute/26/32(a)(1) ",
        "us/statute/26/32(c)(1)(B) ",
        "us/statute/26/32(c)(1)(A)(ii)(I) ",
    ):
        assert not any(covered in issue for issue in structure), covered


def test_us_code_genuinely_omitted_branch_still_fails_with_english_label():
    # Negative control: the module encodes (a)(1) only; the operative
    # limitation in (a)(2) and its subparagraphs must still be rejected.
    result = _analyze(_module(USC_32, "26 USC 32(a)(1)"), FLATTENED_USC_SOURCE, USC_32)
    structure = _structure_issues(result)

    assert any(
        "Source branch (2) at us/statute/26/32(a)(2) [paragraph (a)(2)] is "
        "neither encoded nor precisely deferred." in issue
        for issue in structure
    ), structure
    assert any(
        "us/statute/26/32(a)(2)(B) [subparagraph (a)(2)(B)]" in issue
        for issue in structure
    )
    assert any(
        "us/statute/26/32(c)(1)(A)(ii)(II) [subclause (c)(1)(A)(ii)(II)]" in issue
        for issue in structure
    )
    assert any("us/statute/26/32(e) [subsection (e)]" in issue for issue in structure)
    assert not any("us/statute/26/32(a)(1) " in issue for issue in structure)


def test_cfr_omitted_branch_fails_with_cfr_paragraph_label():
    source = """\
(a) Household members meeting citizenship or alien status requirements. No person is eligible unless that person is:
(1) A U.S. citizen;
(6) An individual who is a qualified alien:
(i) A qualified alien is:
(A) An alien who is lawfully admitted for permanent residence;
(B) An alien who is granted asylum.
(b) Reporting illegal aliens. The State agency must inform USCIS.
"""
    citation = "us/regulation/7/273/4"
    result = _analyze(_module(citation, "7 CFR 273.4(a)(6)(i)(A)"), source, citation)
    structure = _structure_issues(result)

    assert any(
        "Source branch (B) at us/regulation/7/273/4(a)(6)(i)(B) "
        "[paragraph (a)(6)(i)(B)] is neither encoded nor precisely deferred." in issue
        for issue in structure
    ), structure
    assert any(
        "us/regulation/7/273/4(b) [paragraph (b)]" in issue for issue in structure
    )
    assert not any("(a)(6)(i)(A) " in issue for issue in structure)
    assert not any(
        word in issue
        for issue in result.issues
        for word in ("Absatz", "Nummer", "Buchstabe")
    )


def test_state_statute_labels_are_english_and_keep_printed_case():
    source = """\
(1) (a) For purposes of this section, qualified individual means:
(I) A natural person who is domiciled in this state; or
(II) A natural person who died during the taxable year.
(b) The refund is equal to the identical refund amount.
(2) The department shall pay the refund.
"""
    citation = "us-co/statute/39/39-22-2003"
    branches = recognize_source_structure(source, corpus_citation_path=citation)
    branch = next(branch for branch in branches if branch.path == ("1", "a", "ii"))

    assert (
        completeness._branch_citation(citation, branch)
        == "us-co/statute/39/39-22-2003(1)(a)(II) [provision (1)(a)(II)]"
    )


def test_german_labels_are_unchanged():
    branches = recognize_source_structure(
        "(1) Die Einkommensteuer bemisst sich nach dem Einkommen.\n"
        "(2) Bei Ehegatten gilt:\n"
        "1. der Splittingtarif;\n"
        "2. der Grundtarif.\n",
        corpus_citation_path="de/statute/estg/32a",
    )
    rendered = {
        completeness._branch_citation("de/statute/estg/32a", branch)
        for branch in branches
    }

    assert "de/statute/estg/32a(2) [Absatz 2, Nummer 1]" in rendered
    assert "de/statute/estg/32a(1) [Absatz 1]" in rendered


def test_us_sources_do_not_recognize_german_sentence_markers():
    # A Kansas K-40ES voucher: `K-40ES` was recognized as German "Satz 40".
    source = (
        "Schedule K-210 is available from our website at: ksrevenue.gov K-40ES "
        "2026 KANSAS INDIVIDUAL ESTIMATED INCOME TAX VOUCHER K-40ES 1810 "
        "Please use UPPER CASE letters to print the first four letters."
    )
    us_branches = recognize_source_structure(
        source,
        corpus_citation_path=(
            "us-ks/guidance/department-of-revenue/forms/2026/k-40es/document-1"
        ),
    )
    assert not [branch for branch in us_branches if branch.kind == "sentence"]

    german = recognize_source_structure(
        "(1) 1Die Steuer beträgt 10 Prozent. 2Sie ist jährlich zu entrichten.",
        corpus_citation_path="de/statute/estg/32a",
    )
    assert {branch.path for branch in german if branch.kind == "sentence"} == {
        ("1", "satz-1"),
        ("1", "satz-2"),
    }


def test_english_structure_and_formula_locators_still_pair_for_retry():
    structure = (
        "[complete-source-unit:structure] Source branch (ii) at "
        "us/regulation/7/273/4(a)(6)(ii) [paragraph (a)(6)(ii)] is neither "
        "encoded nor precisely deferred."
    )
    formula = (
        "[complete-source-unit:formula-output] Source-stated control (ii) in "
        "us/regulation/7/273/4(a)(6)(ii) [paragraph (a)(6)(ii)] has no "
        "principal derived/relation output (`derived` or `derived_relation`)."
    )
    other = (
        "[complete-source-unit:formula-output] Source-stated control (iii) in "
        "us/regulation/7/273/4(a)(6)(iii) [paragraph (a)(6)(iii)] has no "
        "principal derived/relation output (`derived` or `derived_relation`)."
    )

    assert _validation_retry_branch_locator(structure) == (
        _validation_retry_branch_locator(formula)
    )
    assert _validation_retry_branch_locator(structure) != (
        _validation_retry_branch_locator(other)
    )


def test_us_citation_lists_and_ranges_name_each_listed_branch():
    paths = completeness._paths_from_source_reference(
        "7 CFR 273.4(a)(4)(iii), (a)(5)(ii), and (c)(3)(vi)-(vii)",
        corpus_citation_path="us/regulation/7/273/4",
    )
    assert paths == {
        ("a", "4", "iii"),
        ("a", "5", "ii"),
        ("c", "3", "vi"),
        ("c", "3", "vii"),
    }
    assert completeness._paths_from_source_reference(
        "26 USC 2(a)(1)(A)-(C)", corpus_citation_path="us/statute/26/2/a"
    ) == {("1", "a"), ("1", "b"), ("1", "c")}
    assert completeness._paths_from_source_reference(
        "42 USC 416(l)(1), (3)", corpus_citation_path="us/statute/42/416/l"
    ) == {("1",), ("3",)}
    # A parenthetical gloss, a different section, and German citations are
    # not read as further designations.
    assert completeness._paths_from_source_reference(
        "7 CFR 273.4(a) (citizenship) and 7 CFR 273.10(c)",
        corpus_citation_path="us/regulation/7/273/4",
    ) == {("a",)}
    assert completeness._paths_from_source_reference(
        "de/statute/estg/32a(1), (2)", corpus_citation_path="de/statute/estg/32a"
    ) == {("1",)}


def test_cited_range_covers_only_its_own_branches():
    source = """\
(a) A person is eligible only if the person is:
(1) A U.S. citizen;
(2) A U.S. non-citizen national;
(3) A qualified alien; or
(4) A Hmong or Highland Laotian tribal member.
(b) Reporting requirements apply.
"""
    citation = "us/regulation/7/273/4"
    result = _analyze(_module(citation, "7 CFR 273.4(a)(1)-(3)"), source, citation)
    structure = _structure_issues(result)

    for covered in ("(a)(1) ", "(a)(2) ", "(a)(3) "):
        assert not any(f"273/4{covered}" in issue for issue in structure), covered
    # Negative control: the omitted sibling outside the range still fails.
    assert any(
        "Source branch (4) at us/regulation/7/273/4(a)(4) [paragraph (a)(4)]" in issue
        for issue in structure
    ), structure
    assert any("273/4(b) [paragraph (b)]" in issue for issue in structure)
