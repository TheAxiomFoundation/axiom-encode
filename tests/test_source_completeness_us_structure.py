"""U.S. structure findings: English branch labels, sentence markers, citations.

U.S. repair feedback used German Absatz/Nummer/Buchstabe labels, U.S. text
was read for German glued sentence numbers (`K-40ES` became "Satz 40"), and a
rule citing `7 CFR 273.4(a)(3)-(5)` covered only (a)(3).
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

CFR_273_4 = "us/regulation/7/273/4"
CFR_SOURCE = """\
(a) Household members meeting citizenship or alien status requirements. No person is eligible unless that person is:
(1) A U.S. citizen;
(2) A U.S. non-citizen national;
(3) A qualified alien; or
(4) A Hmong or Highland Laotian tribal member.
(6) An individual who is a qualified alien:
(i) A qualified alien is:
(A) An alien who is lawfully admitted for permanent residence;
(B) An alien who is granted asylum.
(b) Reporting illegal aliens. The State agency must inform USCIS.
"""


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


def test_cfr_omitted_branch_fails_with_english_paragraph_label():
    result = _analyze(
        _module(CFR_273_4, "7 CFR 273.4(a)(6)(i)(A)"), CFR_SOURCE, CFR_273_4
    )
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


def test_us_statute_omitted_branch_fails_with_english_provision_label():
    citation = "us/statute/7/2015/f"
    source = (
        "No individual may participate in the supplemental nutrition assistance "
        "program as a member of any household unless he or she is—\n\n"
        "(1) a resident of the United States; and\n\n"
        "(2) either—\n\n"
        "(A) a citizen or national of the United States;\n\n"
        "(B) an alien lawfully admitted for permanent residence; or\n\n"
        "(C) an individual who lawfully resides in the United States.\n"
    )
    structure = _structure_issues(
        _analyze(
            _module(citation, "7 U.S.C. 2015(f)(1)", "7 U.S.C. 2015(f)(2)(A)-(B)"),
            source,
            citation,
        )
    )

    assert any(
        "Source branch (C) at us/statute/7/2015/f(2)(C) [provision (2)(C)] is "
        "neither encoded nor precisely deferred." in issue
        for issue in structure
    ), structure
    assert not any("2015/f(2)(B) " in issue for issue in structure)


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
    # Corpus shapes: a Kansas voucher number, a CHIP plan's `(42CFR`, a
    # Virginia budget's `8VAC20`, a Georgia rule's `Form G-2RP`.
    for citation, source in (
        (
            "us-ks/guidance/department-of-revenue/forms/2026/k-40es/document-1",
            "Schedule K-210 is available at ksrevenue.gov K-40ES 2026 KANSAS "
            "INDIVIDUAL ESTIMATED INCOME TAX VOUCHER K-40ES 1810 Please print.",
        ),
        (
            "us-md/policy/cms/chip-state-plan/spa/md-23-0002/page-88",
            "Check all standards that will apply to the State plan. "
            "(42CFR 457.305(a) and 457.320(a)) Describe the verification.",
        ),
        (
            "us-va/statute/budget/2025/hb1600/chapter-725/page-198",
            "Notwithstanding 8VAC-20-100, the Board of Education shall issue loans.",
        ),
        (
            "us-ga/regulation/560-7/560-7-8-.33",
            "The buyer shall provide the seller a copy of the Form G-2RP as a "
            "statement of tax withheld.",
        ),
    ):
        branches = recognize_source_structure(source, corpus_citation_path=citation)
        assert not [branch for branch in branches if branch.kind == "sentence"], (
            citation
        )

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


def _listed(source: str, citation: str = CFR_273_4) -> set[tuple[str, ...]]:
    return completeness._paths_from_source_reference(
        source, corpus_citation_path=citation, list_continuations=True
    )


def test_us_citation_lists_and_ranges_name_each_listed_branch():
    assert _listed("7 CFR 273.4(a)(4)(iii), (a)(5)(ii), and (c)(3)(vi)-(vii)") == {
        ("a", "4", "iii"),
        ("a", "5", "ii"),
        ("c", "3", "vi"),
        ("c", "3", "vii"),
    }
    assert _listed("26 USC 2(a)(1)(A)-(C)", "us/statute/26/2/a") == {
        ("1", "a"),
        ("1", "b"),
        ("1", "c"),
    }
    assert _listed("42 USC 416(l)(1), (3)", "us/statute/42/416/l") == {
        ("1",),
        ("3",),
    }
    assert _listed("7 CFR 273.4(a)(1) and (c)(1) of this section") == {
        ("a", "1"),
        ("c", "1"),
    }
    # Without list reading (the default, and every non-principal rule), only
    # the first designation counts, as on main.
    assert completeness._paths_from_source_reference(
        "7 CFR 273.4(a)(3)-(5)", corpus_citation_path=CFR_273_4
    ) == {("a", "3")}


def test_us_citation_lists_never_claim_more_than_they_name():
    # A gloss, a different section, and German citations are not continued.
    assert _listed("7 CFR 273.4(a) (citizenship) and 7 CFR 273.10(c)") == {("a",)}
    assert _listed("de/statute/estg/32a(1), (2)", "de/statute/estg/32a") == {("1",)}
    # A list qualified by another source names that source's branches, so
    # only the first designation counts.
    for source in (
        "7 CFR 273.4(a)(1) and (c)(1) of 7 CFR 273.9",
        "7 CFR 273.4(a)(1), (c)(1)-(2) of 7 CFR 273.9",
        "7 CFR 273.4(a)(1), (c)(1), and (2) of section 273.9",
        "7 CFR 273.4(a)(1) and (c)(1) of paragraph (b)",
        "7 CFR 273.4(a)(1) and (c)(1) of Pub. L. 104-193",
        "7 CFR 273.4(a)(1) and (c)(1) of INA",
    ):
        assert _listed(source) == {("a", "1")}, source
    for qualifier in ("of section 152", "of § 152", "of sec. 152", "of 26 USC 152"):
        assert _listed(
            f"26 USC 32(c)(1)(A) and (B) {qualifier}", "us/statute/26/32"
        ) == {("c", "1", "a")}, qualifier
    # A range that replaces a non-final designation names only its endpoints:
    # neither all of (a)(4) nor subsection (b).
    assert _listed("7 CFR 273.4(a)(4)(iii)-(5)") == {("a", "4", "iii"), ("a", "5")}
    assert _listed("26 USC 32(a)(1)(A)-(c)", "us/statute/26/32") == {
        ("a", "1", "a"),
        ("c",),
    }
    # At most 25 siblings are expanded; a longer range names its endpoints.
    assert len(_listed("7 CFR 273.4(a)(1)-(25)")) == 25
    assert _listed("7 CFR 273.4(a)(1)-(26)") == {("a", "1"), ("a", "26")}
    assert _listed("7 CFR 273.4(a)(1)-(999999)") == {("a", "1"), ("a", "999999")}


def test_cited_range_covers_only_its_own_branches():
    result = _analyze(
        _module(CFR_273_4, "7 CFR 273.4(a)(1)-(3)"), CFR_SOURCE, CFR_273_4
    )
    structure = _structure_issues(result)

    for covered in ("(a)(1) ", "(a)(2) ", "(a)(3) "):
        assert not any(f"273/4{covered}" in issue for issue in structure), covered
    # Negative controls: omitted siblings outside the range still fail.
    assert any(
        "Source branch (4) at us/regulation/7/273/4(a)(4) [paragraph (a)(4)]" in issue
        for issue in structure
    ), structure
    assert any("273/4(a)(6) [paragraph (a)(6)]" in issue for issue in structure)
    assert any("273/4(b) [paragraph (b)]" in issue for issue in structure)


def test_parameter_citation_list_does_not_encode_the_branches_it_lists():
    # Replay shape (7 CFR 273.4 runs 35606543223 and 35611151132): a parameter
    # for the age 18 lists the deeming exemptions (c)(3)(vi)-(vii). It does not
    # encode them, so they must still be reported.
    source = """\
(a) Household members meeting citizenship or alien status requirements.
(b) Reporting illegal aliens.
(c) Households containing sponsored alien members.
(3) Exempt aliens. The provisions of paragraph (c)(2) do not apply to:
(i) An alien who is a member of the sponsor's household;
(ii) An alien who is sponsored by an organization;
(iii) An alien who is not required to have a sponsor;
(iv) An indigent alien;
(v) A battered alien spouse.
(vi) A sponsored alien child under 18 years of age of a sponsored alien.
(vii) A citizen child under age 18 of a sponsored alien.
"""
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CFR_273_4}
rules:
  - name: under_eighteen_age_limit
    kind: parameter
    dtype: Integer
    source: 7 CFR 273.4(c)(3)(v), and (c)(3)(vi)-(vii)
    versions:
      - effective_from: '2025-01-01'
        formula: '18'
"""
    structure = _structure_issues(_analyze(content, source, CFR_273_4))
    for branch in ("(c)(3)(vi) [paragraph", "(c)(3)(vii) [paragraph"):
        assert any(branch in issue for issue in structure), (branch, structure)

    # The same citation on a principal derived output does encode them.
    derived = content.replace("kind: parameter", "kind: derived").replace(
        "dtype: Integer", "dtype: Judgment"
    )
    structure = _structure_issues(_analyze(derived, source, CFR_273_4))
    for branch in ("(c)(3)(vi) [paragraph", "(c)(3)(vii) [paragraph"):
        assert not any(branch in issue for issue in structure), (branch, structure)


def test_descending_roman_continuation_never_reaches_a_top_level_letter():
    # `(i)` after `(ii)` is a roman sibling listed out of order, not the
    # letter subsection (i): the list stops instead of citing (i).
    assert _listed("7 CFR 273.4(a)(1)(ii), (i)") == {("a", "1", "ii")}
    assert _listed("7 CFR 273.4(c)(3)(vi), (v)") == {("c", "3", "vi")}
    assert _listed("26 USC 32(c)(1)(F)(ii), (i)", "us/statute/26/32") == {
        ("c", "1", "f", "ii")
    }
    assert _listed("7 CFR 273.4(a)(6)(ii)(B), (i)") == {("a", "6", "ii", "b")}
    # Positive controls: ascending romans and a letter after a letter-shaped
    # roman still read.
    assert _listed("7 CFR 273.4(a)(1)(i), (ii)") == {
        ("a", "1", "i"),
        ("a", "1", "ii"),
    }
    assert _listed("7 CFR 273.4(c)(3)(v), and (c)(3)(vi)") == {
        ("c", "3", "v"),
        ("c", "3", "vi"),
    }
    assert _listed("7 CFR 273.4(a)(1)(i), (c)") == {("a", "1", "i"), ("c",)}


def test_us_citation_list_counts_only_when_nothing_qualifies_it():
    for tail in (
        " in 7 CFR 273.9",
        " under 7 CFR 273.9",
        " as defined in 7 CFR 273.9",
        ", 7 CFR 273.9",
        " (7 CFR 273.9)",
        ", of 7 CFR 273.9",
        "of 7 CFR 273.9",
        " (relating to aliens) of 7 CFR 273.9",
        " are not implemented",
        " and 7 CFR 273.41(b)",
    ):
        assert _listed(f"7 CFR 273.4(a)(1) and (c)(1){tail}") == {("a", "1")}, tail
    assert _listed(
        "26 USC 32(c)(1)(A) and (B), both of section 152", "us/statute/26/32"
    ) == {("c", "1", "a")}
    # Positive controls: the list ends the citation, says `of this section`,
    # or is followed by another citation of this unit.
    for tail in ("", ".", "; 7 CFR 273.9", "\nNext line", " of this section"):
        assert _listed(f"7 CFR 273.4(a)(1) and (c)(1){tail}") == {
            ("a", "1"),
            ("c", "1"),
        }, tail
    assert _listed("7 CFR 273.4(a)(1) and (c)(1), and 7 CFR 273.4(b)") == {
        ("a", "1"),
        ("c", "1"),
        ("b",),
    }
    assert _listed(
        "26 USC 32(c)(1)(A) and (B), and 26 U.S.C. 32(d)", "us/statute/26/32"
    ) == {("c", "1", "a"), ("c", "1", "b"), ("d",)}


_LETTER_I_SOURCE = """\
(a) Eligibility. A household is eligible only if:
(1) The household meets both tests:
(i) The household lives in the State; and
(ii) The household applies for benefits.
(2) The household cooperates with the State agency.
(b) Applications. The State agency must accept applications.
(c) Interviews. The State agency must interview households.
(d) Verification. The State agency must verify income.
(e) Notices. The State agency must send notices.
(f) Hearings. The State agency must hold hearings.
(g) Records. The State agency must keep records.
(h) Reports. The State agency must file reports.
(i) Penalties. The State agency must apply penalties.
"""


def test_list_shapes_that_name_no_branch_here_leave_it_reported():
    def structure(rule_source: str) -> list[str]:
        return _structure_issues(
            _analyze(_module(CFR_273_4, rule_source), _LETTER_I_SOURCE, CFR_273_4)
        )

    top_level_i = "Source branch (i) at us/regulation/7/273/4(i) [paragraph (i)]"
    roman_ii = "273/4(a)(1)(ii) [paragraph (a)(1)(ii)]"
    issues = structure("7 CFR 273.4(a)(1)(ii), (i)")
    assert any(top_level_i in issue for issue in issues), issues
    issues = structure("7 CFR 273.4(a)(1)(i), (ii) in 7 CFR 273.9")
    assert any(roman_ii in issue for issue in issues), issues
    # Positive control: the plain list covers both romans.
    issues = structure("7 CFR 273.4(a)(1)(i), (ii)")
    assert not any(roman_ii in issue for issue in issues), issues
    assert not any("273/4(a)(1)(i) [" in issue for issue in issues), issues
