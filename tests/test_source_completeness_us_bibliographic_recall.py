"""US bibliographic locators are not demanded as named scalars.

Replays of the latest rejected US candidates (2026-10-14) still demanded:

* `62` and `65` from `(62 FR 61344)` and `(65 FR 58301)` in 7 CFR 273.4;
* `111`, `124`, `2403`, `107`, `115` and `55` from the U.S. Code repeal notes
  in 26 U.S.C. 32, and `116`, `134` and `3041` from the note in 26 U.S.C. 25A;
* `1972`, `1974`, `1980` and `1983` from act titles in 42 U.S.C. 415.

Each mask accepts one complete citation grammar and blanks only the digits
(and a public-law number's joining dash) of its locator numbers in place. The
differential tests below compare every input with the same pipeline with the
four new masks disabled: no occurrence may be added or changed, and every
removed occurrence must sit on blanked locator digits.
"""

from __future__ import annotations

import functools
import re

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

SNAP_REGULATION = "us/regulation/7/273/4"
EITC_STATUTE = "us/statute/26/32"
PIA_STATUTE = "us/statute/42/415/a"
NEW_LOCATORS = "_US_BIBLIOGRAPHIC_NUMERIC_RECALL_LOCATORS"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)


def _cleaned(text: str, citation: str) -> str:
    return completeness.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )


def _occurrences(text: str, citation: str):
    cleaned = _cleaned(text, citation)
    return [
        (item.value, item.start, item.end, cleaned[item.start : item.end])
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="legacy"
        )
    ]


def _values(text: str, citation: str) -> set[float]:
    return {value for value, *_ in _occurrences(text, citation)}


def _main_occurrences(text: str, citation: str):
    """The same pipeline with the four new locator masks disabled."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(completeness, NEW_LOCATORS, ())
        return _occurrences(text, citation)


def _assert_only_locator_digits_removed(
    text: str, citation: str, *, allow_restored: bool = False
):
    """Every occurrence main reads survives unless it sat on blanked digits.

    With `allow_restored`, the masked view may also read an operative token
    that main's inventory suppressed because a locator shared its leading
    digits (main drops `124,000` after a `124 Stat.`); such a token's text is
    untouched by the masks.
    """

    masked = _occurrences(text, citation)
    main = _main_occurrences(text, citation)
    cleaned = _cleaned(text, citation)
    added = set(masked) - set(main)
    if allow_restored:
        for _value, start, end, raw in added:
            assert raw.strip() and cleaned[start:end] == raw
    else:
        assert not added, "a mask added or changed an occurrence"
    removed = set(main) - set(masked)
    for _value, start, end, _raw in removed:
        assert not cleaned[start:end].strip(), (cleaned, start, end)
    return {value for value, *_ in removed}
    return {value for value, *_ in removed}


def _demanded(source: str, citation: str) -> set[str]:
    content = yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []})
    result = completeness.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        match.group(1)
        for issue in result.issues
        for match in [RECALL_VALUE.search(issue)]
        if match
    }


# Excerpts of the replayed corpus rows (7 CFR 273.4, 26 U.S.C. 32 and
# 42 U.S.C. 415), with operative sentences kept beside each locator.
SNAP_FOOTNOTE = (
    "1 For guidance, see the DOJ Interim Guidance published November 17, 1997 "
    "(62 FR 61344).\n\n"
    "(iii) An alien who is a Cuban or Haitian entrant is eligible for 12 months "
    "after the State agency determines eligibility.\n\n"
    "A Policy Notice providing guidance for compliance with PRWORA published on "
    "September 28, 2000 (65 FR 58301)."
)
EITC_REPEAL_NOTES = (
    "[(g) Repealed. Pub. L. 111–226, title II, § 219(a)(2) , Aug. 10, 2010 , "
    "124 Stat. 2403 ]\n\n"
    "[(h) Repealed. Pub. L. 107–16, title III, § 303(c) , June 7, 2001 , "
    "115 Stat. 55 ]\n\n"
    "(i) Denial of credit for individuals having excessive investment income "
    "(1) In general No credit shall be allowed under subsection (a) for the "
    "taxable year if the aggregate amount of disqualified income of the taxpayer "
    "for the taxable year exceeds $10,000."
)
PIA_TITLES = (
    "subject to section 215(j)(2) of the Social Security Amendments of 1972 "
    "(relating to the number of elapsed years), compensation under the Railroad "
    "Retirement Act of 1937 or 1974 which is creditable, and the Social Security "
    "Disability Amendments of 1980. The amount shall be $11.50 for each year of "
    "coverage in excess of 10, and not exceeding 30 years."
)


@pytest.mark.parametrize(
    "source, citation, main_only, still_demanded",
    (
        # The footnote marker `1` is a separate, unchanged demand.
        (SNAP_FOOTNOTE, SNAP_REGULATION, {"62", "65"}, {"1", "12"}),
        (
            EITC_REPEAL_NOTES,
            EITC_STATUTE,
            {"111", "124", "2403", "107", "115", "55"},
            {"10000"},
        ),
        (PIA_TITLES, PIA_STATUTE, {"1972", "1974", "1980"}, {"11.5", "10", "30"}),
    ),
)
def test_replayed_locators_are_not_demanded_but_operative_values_are(
    source, citation, main_only, still_demanded, monkeypatch
):
    assert _demanded(source, citation) == still_demanded
    monkeypatch.setattr(completeness, NEW_LOCATORS, ())
    assert _demanded(source, citation) == still_demanded | main_only


@pytest.mark.parametrize(
    "source",
    (
        # Federal Register: volume, marker and page on one line, with no
        # amount, decimal, range, percentage or sign syntax touching them.
        "62 FR",
        "62 FR\n61344",
        "$62 FR 61344",
        "62 FR 61344%",
        "62 FR 61344.5",
        "62 FR 100-2",
        "62 FR 100 - 2",
        "1.62 FR 100",
        "-62 FR 100",
        "62 fr 61344",
        "62 CFR 273",
        "62\u00adFR 100",
        # Public law: only a complete Congress-law pair.
        "Pub. L. 111",
        "Pub. L. 111–226.5 dollars",
        "Pub. L. 111–226 - 5 dollars",
        "Pub. L. 111–226%",
        # A bare Statutes at Large reference outside a complete note keeps its
        # values, as on main.
        "124 Stat. 2403",
        "Pub. L. 111–226 5 dollars",
        "Pub. L. 111–226, 5 dollars",
        "Pub. L. 111–226 and 5 dollars",
        "62 FR 61344 1 households",
        "90 Fed. Reg. 27074 1 households",
        "62 FR 61344 and 1 household",
        "62 FR 61344\n5 dollars",
        "5 62 FR 61344",
        "Amendments of 1983 and 1984",
        "Social Security Amendments of 1983 1 household",
        # Act titles: a capitalized title, the exact phrase, nothing numeric
        # touching the year.
        "the amendments of 1983 apply",
        "Social Security Amendments of 1983.5",
        "Social Security Amendments of 1983-1",
        "Social Security Amendments of 1983%",
        "Social Security Amendments in 1983",
        "an Act of 1937 or 1974",
        "Railroad Retirement Act of 1937 or 1974%",
        "Railroad Retirement Act of 1937 or 1974.5",
    ),
)
def test_partial_or_operative_locator_shapes_keep_every_value(source):
    assert _occurrences(source, EITC_STATUTE) == _main_occurrences(source, EITC_STATUTE)


@pytest.mark.parametrize(
    "source, kept",
    (
        # An incomplete repeal note keeps its Statutes at Large numbers and
        # every operative value; only the public-law identifier is a locator.
        (
            "Repealed. Pub. L. 111–226, title II, $500, Aug. 10, 2010, "
            "124 Stat. 2403 ]",
            {124, 2403, 500},
        ),
        (
            "Repealed. Pub. L. 111–226, title II, Aug. 10, 2010, 124 Stat. 2403 "
            "and $5 is allowed.",
            {124, 2403, 5},
        ),
        (
            "Repealed. Pub. L. 111–226, title II, Aug. 10, 2010, 124 Stat. 2403 "
            "dollars.",
            {124, 2403},
        ),
        ("Repealed. Pub. L. 111–226, 124 Stat. 2403 ]", {124, 2403}),
        ("Repealed. Pub. L. 111–226, Aug. 10, 2010, 124 Stat. 2403.5 ]", {124}),
    ),
)
def test_incomplete_repeal_notes_keep_statutes_at_large_values(source, kept):
    removed = _assert_only_locator_digits_removed(source, EITC_STATUTE)
    assert removed <= {111, 226}
    assert kept <= _values(source, EITC_STATUTE)


# The #1809 counterexample corpus, plus operative numbers equal to each
# locator component written beside the locator.
COUNTEREXAMPLES = (
    "Under section 7C, 35 per cent of earnings is disregarded.",
    "Under section 7C, 100(1 + rate) dollars shall be paid.",
    "Under section 7C, 100x-2 dollars shall be paid. A separate adjustment of "
    "-2 dollars applies.",
    "At least 236 of the National Housing Act applicants must qualify.",
    "A household is eligible if it contains Parolees Paroled into the U.S. "
    "under section 212(d)(5)(A) of the INA for a period of at least 1 year.",
    "The Social Security Amendments of 1983 set the limit at $1,983 and 1983 "
    "households.",
    "The Railroad Retirement Act of 1937 or 1974 pays 1974 dollars to 1937 people.",
    "See 62 FR 61344. The household of 62 members receives $61,344.",
    "[(g) Repealed. Pub. L. 111–226, title II, § 219(a)(2) , Aug. 10, 2010 , "
    "124 Stat. 2403 ] (h) The credit is $111 for 226 days and 124 or 2403 "
    "units.",
    "Pub. L. 104-193 allows -5 dollars, then 104 - 193 = -89.",
    "Repealed. Pub. L. 95–216, title II, § 201(f)(2) , Dec. 20, 1977 , "
    "91 Stat. 1521 .\n(4) A payment of 95 or 216 dollars, 91 and 1521.",
)


@pytest.mark.parametrize("source", COUNTEREXAMPLES)
def test_masks_remove_only_locator_digits(source):
    _assert_only_locator_digits_removed(source, EITC_STATUTE)
    _assert_only_locator_digits_removed(source, SNAP_REGULATION)


@pytest.mark.parametrize(
    "source, operative",
    (
        (COUNTEREXAMPLES[5], {1983}),
        (COUNTEREXAMPLES[6], {1937, 1974}),
        (COUNTEREXAMPLES[7], {62, 61344}),
        (COUNTEREXAMPLES[8], {111, 226, 124, 2403}),
        (COUNTEREXAMPLES[10], {95, 216, 91, 1521}),
    ),
)
def test_values_equal_to_masked_locators_stay_demanded(source, operative):
    assert operative <= _values(source, EITC_STATUTE)


def test_non_us_paths_are_unchanged():
    source = "(62 FR 61344) Pub. L. 111-226 and the Social Security Amendments of 1983"
    citation = "uk/legislation/ukpga/2012/5/section/1"
    assert _occurrences(source, citation) == _main_occurrences(source, citation)


# Verbatim excerpts from the US corpus (axiom-corpus ba210f4b) where a masked
# locator sits beside operative amounts, rates and ages. Federal guidance,
# CFR, U.S. Code and state statute/manual rows are covered.
CORPUS_EXCERPTS = (
    (
        "us/guidance/cms/part-a-deductible-coinsurance/2026/federal-register/block-1",
        "This deductible amount is determined by multiplying $1,676 (the "
        "inpatient hospital deductible for CY 2025 (89 FR 68986)) by the "
        "payment-weighted average increase in the payment rates of 1.0262 "
        "multiplied by the increase in real casemix of 1.009, which equals "
        "$1,735.39 and is rounded to $1,736",
        {89},
        {1676, 1.0262, 1.009, 1735.39, 1736},
    ),
    (
        "us/regulation/42/457/609",
        "$15,361,000,000 (appropriated in section 108 of Pub. L. 111-3, as amended "
        "by section 10203 of Pub. L. 111-148)).\n\n(2) $21,061,000,000, determined "
        "as the sum of the amount determined in paragraph (1) of this section, "
        "$18,211,000,000, and $2,850,000,",
        {3, 111, 148},
        {15_361_000_000, 21_061_000_000, 18_211_000_000, 2_850_000},
    ),
    (
        "us/statute/26/3121",
        "if the remuneration paid in such year by the organization to the employee "
        "for such service is less than $100; [(17) Repealed. Pub. L. 113–295, "
        "div. A, title II, § 221(a)(19)(B)(iv) , Dec. 19, 2014 , 128 Stat. 4040 ] "
        "(18) any payment made",
        {113, 128, 4040},
        {100},
    ),
    (
        "us/guidance/irs/rev-proc-2016-55/page-4",
        "enacted as part of the Consolidated Appropriations Act, 2016, Pub. L. "
        "114-113, div. Q, made permanent the earned income threshold of $3,000 "
        "used in § 24(d)(1)(B)(i)",
        {113, 114},
        {3000},
    ),
    (
        "us/statute/hts/9903.94.32",
        "[Compilers note: This heading is effective on or after June 30, 2025. "
        "For more information, see 90 Fed. Reg. 27851.]\nRates of duty "
        "(1-General): 10%\nRates of duty (1-Special): 10%",
        {90},
        {0.1},
    ),
    (
        "us-ri/manual/eohhs/medicaid/210-ricr-30-05-2/page-6",
        "for continued Medicaid coverage until the age of twenty-six (26) under the "
        "SUPPORT for Patients and Communities Act, Pub. L. No. 115-271. Members of "
        "this population are only eligible in Rhode Island",
        {115, 271},
        {26},
    ),
    (
        "us-co/statute/crs-2024/title-39/page-466",
        '(A) "CARES Act" means the March 2020 "Coronavirus Aid, Relief, and '
        'Economic Security Act", Pub.L. 116-136. (B) "Colorado taxable income" '
        "means federal taxable income",
        {116, 136},
        set(),
    ),
    (
        "us-ny/manual/doh/medicaid/categorical-factors/page-95",
        "Certified disabled widows or widowers who lost SSI and Medicaid benefits "
        "due to an actuarial adjustment authorized under the Social Security "
        "Amendments of 1983 are eligible for Medicaid",
        {1983},
        set(),
    ),
)


@pytest.mark.parametrize("citation, source, locators, operative", CORPUS_EXCERPTS)
def test_corpus_excerpts_drop_only_locators(citation, source, locators, operative):
    main_values = {value for value, *_ in _main_occurrences(source, citation)}
    values = _values(source, citation)
    assert main_values - values == locators
    assert operative <= values
    _assert_only_locator_digits_removed(source, citation)


LOCATORS = st.sampled_from(
    (
        "(62 FR 61344)",
        "90 Fed. Reg. 27074",
        "Pub. L. 111–226",
        "Pub. L. No. 92-603",
        "the Social Security Amendments of 1983",
        "the Railroad Retirement Act of 1937 or 1974",
        "[(g) Repealed. Pub. L. 111–226, title II, § 219(a)(2) , Aug. 10, 2010 , "
        "124 Stat. 2403 ]",
        "(3) Repealed. Pub. L. 95–216, title II, § 201(f)(2) , Dec. 20, 1977 , "
        "91 Stat. 1521 .",
    )
)
OPERATIVE = st.one_of(
    st.integers(min_value=1, max_value=999_999).map(lambda n: f"${n:,}"),
    st.integers(min_value=1, max_value=99).map(lambda n: f"{n} percent"),
    st.integers(min_value=1, max_value=9_999).map(lambda n: f"{n} households"),
    st.integers(min_value=1, max_value=9_999).map(lambda n: f"-{n} dollars"),
    st.integers(min_value=1, max_value=9_999).map(lambda n: f"{n}(1 + rate)"),
    st.decimals(min_value=1, max_value=9_999, places=2).map(str),
)
SEPARATORS = st.sampled_from(
    (
        " ",
        "",
        ". ",
        "; ",
        ": ",
        "\n",
        "\n\n",
        " and ",
        " or ",
        " to ",
        ", ",
        " under ",
        "(",
        ")",
        "/",
        " x ",
        "-",
        " - ",
        "\u2013",
        "\u00ad",
    )
)


@given(st.lists(st.tuples(LOCATORS, SEPARATORS, OPERATIVE, SEPARATORS), max_size=4))
@settings(deadline=None, max_examples=300)
def test_masks_never_remove_or_change_operative_tokens(parts):
    source = "".join(f"{a}{b}{c}{d}" for a, b, c, d in parts)
    for citation in (EITC_STATUTE, SNAP_REGULATION):
        _assert_only_locator_digits_removed(source, citation, allow_restored=True)


def test_removing_a_locator_restores_a_grouped_amount_main_suppressed():
    # Main's inventory drops `124,000` because `124` (the Statutes at Large
    # volume) appears earlier; with the note masked, the amount is demanded.
    source = (
        "[(g) Repealed. Pub. L. 111\u2013226, title II, \u00a7 219(a)(2) , Aug. 10, 2010 , "
        "124 Stat. 2403 ] (h) The limit is 124,000 dollars."
    )
    assert 124_000 not in {
        value for value, *_ in _main_occurrences(source, EITC_STATUTE)
    }
    assert 124_000 in _values(source, EITC_STATUTE)
    _assert_only_locator_digits_removed(source, EITC_STATUTE, allow_restored=True)


@given(st.integers(min_value=1, max_value=999_999))
@settings(deadline=None)
def test_new_masks_are_idempotent_and_keep_length(amount):
    source = (
        "(62 FR 61344) Pub. L. 111–226. The Social Security Amendments of 1983 "
        f"apply. The limit is ${amount}."
    )
    masked = _cleaned(source, EITC_STATUTE)
    assert len(masked) == len(_cleaned(source, "uk/legislation/ukpga/2012/5"))
    assert float(amount) in _values(source, EITC_STATUTE)
    assert _cleaned(masked, EITC_STATUTE) == masked
