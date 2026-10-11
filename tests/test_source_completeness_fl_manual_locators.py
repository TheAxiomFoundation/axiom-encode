"""Florida ESS manual pages: passage and appendix locators, folded 100% rates.

Two signed promotions of Florida ESS Program Policy Manual pages failed the
complete-source-unit gate on demands the source does not make. The sentences
below are verbatim from the pinned corpus rows
`us-fl/manual/dcf/ess-program-policy-manual/2000-coverage-groups/page-5` and
`.../2600-calculation-of-benefits/page-8`:

* numeric recall demanded the passage numbers 2010.0202 (an inline heading) and
  2610.0103 (`Refer to passage 2610.0103`) and the `1` of `Appendix A-1`;
* formula clause 11 of page 5 runs across the inline heading `2010.0202
  Noncategorically Eligible Assistance Groups (FS)`, so the heading number
  became an operand the executed formula had to contain;
* simplification folds a 100% rate out of `rate * base`, so the source's
  `100% of` could not be grounded by the rate that executes it.
"""

from __future__ import annotations

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

PAGE_5 = "us-fl/manual/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
PAGE_8 = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
NAMESPACE = "us-fl:policies/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
INVENTORY = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)
GROUNDING = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)

# Page 5, characters 2437:2932 (formula clause 11 as the gate segments it).
CLAUSE_ACROSS_HEADING = (
    "If the SFU does not meet the 200% of the federal poverty income limit, the "
    "SFU must meet the net income limit of 100% of the federal poverty level for "
    "the AG size and the asset limit of $4,500. 2010.0202 Noncategorically "
    "Eligible Assistance Groups (FS) If the standard filing unit does not contain "
    "all individuals who receive or are authorized to receive Temporary Cash "
    "Assistance, SSI, or services through Healthy Families Florida, the "
    "assistance group is not considered categorically eligible."
)


def _recall_values(source: str, citation: str) -> set[float]:
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    return {occurrence.value for occurrence in INVENTORY(cleaned)}


# --- numeric recall --------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        (CLAUSE_ACROSS_HEADING, PAGE_5, {2.0, 1.0, 4500.0}),
        (
            "The assistance group is responsible for providing information "
            "regarding the individual's inability to purchase and prepare meals "
            "since this is the key determining factor. Refer to Appendix A-1 for "
            "the 165% Need Standard Table.",
            PAGE_8,
            {1.65},
        ),
        (
            "Refer to passage 2610.0103 for assistance groups not tested against "
            "the gross income limit. The gross income limits can be found in the "
            "on-line reference table or in Appendix A-1.",
            PAGE_8,
            set(),
        ),
        (
            "Income tests for determining food stamp eligibility are presented in "
            "passages 2610.0104.02 through 2610.0104.04.",
            PAGE_8,
            set(),
        ),
    ),
)
def test_florida_passage_and_appendix_locators_are_not_values(
    source, citation, expected
):
    assert _recall_values(source, citation) == expected


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        # A four-by-four dotted number without a titled program-code heading
        # after it is a value.
        ("The rate is 1234.5678 dollars per unit.", PAGE_5, {1234.5678}),
        ("Pay 2010.0202 Noncategorically Eligible amounts.", PAGE_5, {2010.0202}),
        # Outside manuals the heading shape is not assumed.
        (
            "2010.0202 Noncategorically Eligible Assistance Groups (FS)",
            "us/guidance/usda/fns/snap-example",
            {2010.0202},
        ),
        ("Refer to passage 2610.0103 now.", "us/statute/7/2014", {2610.0103}),
        # Values beside a locator keep their value.
        ("Appendix A-1 lists the $4,500 asset limit.", PAGE_8, {4500.0}),
        ("Refer to passage 2610.0103 for income of $2,221.", PAGE_8, {2221.0}),
        ("See passage 12 of the act.", PAGE_8, {12.0}),
        ("Appendix A-12 shows 3 rates.", PAGE_8, {3.0}),
    ),
)
def test_values_beside_florida_locator_shapes_are_kept(source, citation, expected):
    assert _recall_values(source, citation) == expected


_TITLE_WORD = st.from_regex(r"[A-Z][a-z]{2,9}", fullmatch=True)
_PROGRAM = st.sampled_from(("FS", "TCA", "MA", "SFP", "RAP"))


@settings(max_examples=200, deadline=None)
@given(
    prefix=st.from_regex(r"[a-z ]{0,30}\.", fullmatch=True),
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    title=st.lists(_TITLE_WORD, min_size=1, max_size=6),
    program=_PROGRAM,
)
def test_heading_spans_cover_exactly_the_passage_number(
    prefix, major, minor, title, program
):
    number = f"{major}.{minor:04d}"
    text = f"{prefix} {number} {' '.join(title)} ({program}) If it applies."
    spans = sc._us_manual_passage_locator_spans(text, corpus_citation_path=PAGE_5)
    start = text.index(number)
    assert spans == ((start, start + len(number)),)
    masked = sc._mask_numeric_spans(text, spans)
    # Masking keeps the length and changes nothing outside the label.
    assert len(masked) == len(text)
    assert masked[:start] == text[:start]
    assert masked[start + len(number) :] == text[start + len(number) :]


@settings(max_examples=200, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    tail=st.from_regex(r"[a-z][a-z ,]{0,60}", fullmatch=True),
)
def test_untitled_four_by_four_numbers_never_mask(major, minor, tail):
    text = f"The limit is {major}.{minor:04d} {tail}"
    assert sc._us_manual_passage_locator_spans(text, corpus_citation_path=PAGE_5) == ()


@settings(max_examples=100, deadline=None)
@given(text=st.text(max_size=200))
def test_locator_spans_are_inert_outside_us_manuals(text):
    for citation in ("us/statute/7/2014", "de/statutes/estg/32a", ""):
        assert (
            sc._us_manual_passage_locator_spans(text, corpus_citation_path=citation)
            == ()
        )


# --- formula clauses -------------------------------------------------------


def _formula_payload(rate: str, excerpt: str, formula: str) -> str:
    def atom(kind: str, text: str) -> dict:
        return {
            "kind": kind,
            "path": "versions[0].formula",
            "source": {"corpus_citation_path": PAGE_5, "excerpt": text},
        }

    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": PAGE_5}},
            "inputs": [
                {
                    "name": name,
                    "entity": "Household",
                    "dtype": "Money",
                    "unit": "USD",
                    "period": "Month",
                }
                for name in (
                    "federal_poverty_level",
                    "sfu_net_income",
                    "sfu_countable_assets",
                )
            ],
            "rules": [
                {
                    "name": "net_income_ratio",
                    "kind": "parameter",
                    "dtype": "Rate",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                atom("amount", f"{rate} of the federal poverty level")
                            ]
                        }
                    },
                    "versions": [
                        {
                            "effective_from": "0001-01-01",
                            "formula": str(int(rate[:-1]) / 100),
                        }
                    ],
                },
                {
                    "name": "asset_limit",
                    "kind": "parameter",
                    "dtype": "Money",
                    "unit": "USD",
                    "metadata": {
                        "proof": {"atoms": [atom("amount", "asset limit of $4,500")]}
                    },
                    "versions": [{"effective_from": "0001-01-01", "formula": "4500"}],
                },
                {
                    "name": "meets_limits",
                    "kind": "derived",
                    "entity": "Household",
                    "dtype": "Judgment",
                    "period": "Month",
                    "metadata": {"proof": {"atoms": [atom("formula", excerpt)]}},
                    "versions": [{"effective_from": "0001-01-01", "formula": formula}],
                },
            ],
        },
        sort_keys=False,
    )


def _cases(limit_holds_at: float) -> list[dict]:
    return [
        {
            "name": "at_the_limits",
            "period": "2026-05",
            "input": {
                "federal_poverty_level": 1000,
                "sfu_net_income": limit_holds_at,
                "sfu_countable_assets": 4500,
            },
            "output": {f"{NAMESPACE}#meets_limits": "holds"},
        },
        {
            "name": "above_the_net_limit",
            "period": "2026-05",
            "input": {
                "federal_poverty_level": 1000,
                "sfu_net_income": limit_holds_at + 1,
                "sfu_countable_assets": 4500,
            },
            "output": {f"{NAMESPACE}#meets_limits": "not_holds"},
        },
    ]


def _formula_branch_issues(
    source: str, rate: str, excerpt: str, formula: str, at: float
):
    result = sc.analyze_complete_source_unit(
        _formula_payload(rate, excerpt, formula),
        source,
        corpus_citation_path=PAGE_5,
        test_cases=_cases(at),
        extract_numeric_occurrences=INVENTORY,
        extract_numeric_grounding_occurrences=GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [
        issue
        for issue in result.issues
        if "Companion tests do not demonstrate formula branch" in issue
    ]


_EXCERPT_130 = (
    "the SFU must meet the net income limit of 130% of the federal poverty level "
    "for the AG size and the asset limit of $4,500"
)
_FORMULA = (
    "sfu_net_income <= net_income_ratio * federal_poverty_level "
    "and sfu_countable_assets <= asset_limit"
)


def test_passage_heading_inside_a_formula_clause_is_not_an_operand():
    # The 130% rate is not an identity factor, so only the heading number
    # inside the clause can keep this branch undemonstrated.
    source = CLAUSE_ACROSS_HEADING.replace(
        "If the SFU does not meet the 200% of the federal poverty income limit, ",
        "If the SFU does not meet the gross income limit, ",
    ).replace("100%", "130%")
    assert _formula_branch_issues(source, "130%", _EXCERPT_130, _FORMULA, 1300) == []


def test_without_a_heading_the_same_clause_was_already_demonstrated():
    source = (
        "If the SFU does not meet the gross income limit, the SFU must meet the "
        "net income limit of 130% of the federal poverty level for the AG size "
        "and the asset limit of $4,500."
    )
    assert _formula_branch_issues(source, "130%", _EXCERPT_130, _FORMULA, 1300) == []


def test_a_folded_100_percent_rate_still_grounds_its_source_operand():
    source = (
        "If the SFU does not meet the gross income limit, the SFU must meet the "
        "net income limit of 100% of the federal poverty level for the AG size "
        "and the asset limit of $4,500."
    )
    excerpt = _EXCERPT_130.replace("130%", "100%")
    assert _formula_branch_issues(source, "100%", excerpt, _FORMULA, 1000) == []
