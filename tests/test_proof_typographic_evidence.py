"""Proof excerpts that differ from the source only typographically.

The cases come from US encoder runs and committed rulespec-us modules whose
proof excerpts quoted the printed source text but were rejected because the
source spells it with a ligature, a wrapped compound hyphen, an en dash, a
space after a currency sign or before a comma, curly quotation marks, or a
recased first letter. Each acceptance has a negative control of the same
shape that changes a word, number, or structure and must still be rejected.
"""

from __future__ import annotations

import pytest

from axiom_encode.harness.proof_validator import (
    _located_proof_evidence,
    _source_contains_proof_evidence,
    _source_proof_evidence_spans,
    source_contains_typographic_proof_evidence,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import (
    _rule_verified_source_excerpt_pairs_by_path,
    _source_evidence_fragment_is_body_bound,
)

CITATION = "us/guidance/example/page-1"


def _content(excerpt: str, *, rule_source: str | None = None) -> str:
    source_line = f"    source: {rule_source}\n" if rule_source else ""
    return f"""format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: {CITATION}
rules:
  - name: official_amount
    kind: parameter
    dtype: Money
    unit: USD
{source_line}    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: {excerpt!r}
    versions:
      - effective_from: '2025-10-01'
        formula: '25'
"""


def _evidence_issues(excerpt: str, source_text: str, **kwargs) -> list[str]:
    result = validate_rulespec_proofs(
        _content(excerpt, **kwargs),
        source_texts={CITATION: source_text},
    )
    return [issue for issue in result.issues if "Proof source evidence" in issue]


@pytest.mark.parametrize(
    ("source_text", "excerpt"),
    [
        # us-tn/manual/dhs/snap/24-21/page-1: ligature plus a compound word
        # wrapped after its hyphen.
        (
            "the income is from a new source and not more than twenty- ﬁve "
            "dollars ($25) from the new income source will be received",
            "not more than twenty-five dollars ($25)",
        ),
        (
            "not more than twenty-\nfive dollars ($25) will be received",
            "not more than twenty-five dollars ($25)",
        ),
        # A word hyphenated at a line end.
        (
            "income from self-employ-\nment is counted monthly",
            "income from self-employment is counted monthly",
        ),
        # us/statute/42/12709: en dash in a standard's number.
        (
            "the requirements of ASHRAE Standard 90.1–2004. (c) Revisions",
            "the requirements of ASHRAE Standard 90.1-2004",
        ),
        # us/statute/42/12709: a space set before a comma.
        (
            "If the Secretaries have not, by September 30, 2006 , established "
            "energy efficiency standards under subsection (a), all new",
            "If the Secretaries have not, by September 30, 2006, established "
            "energy efficiency standards under subsection (a)",
        ),
        # us-va/statute/58.1/58.1-322.03: a space after the currency sign.
        (
            "shall be limited to $ 4,000 per prepaid tuition contract or college "
            "savings trust account. No deduction",
            "$4,000 per prepaid tuition contract or college savings trust account",
        ),
        ("as provided in § 273.4 of this part", "as provided in §273.4"),
        # No-break and thin spaces after a sign; a soft hyphen inside a word.
        ("as provided in \u00a7\u00a0273.4 of this part", "as provided in §273.4"),
        ("limited to $\u2009 4,000 per account", "$4,000 per account"),
        ("income from self-employ\u00adment is counted", "income from self-employment"),
        # A non-ASCII hyphen or a soft hyphen at a line wrap.
        (
            "not more than twenty\u2010 \ufb01ve dollars ($25)",
            "twenty-five dollars ($25)",
        ),
        (
            "income from self\u2011\nemployment is counted",
            "income from self-employment",
        ),
        (
            "income from self\u2013\nemployment is counted",
            "income from self-employment",
        ),
        (
            "income from self-employ\u2010\nment is counted",
            "income from self-employment",
        ),
        (
            "income from self-employ\u00ad\nment is counted",
            "income from self-employment",
        ),
        # Curly and straight apostrophes and quotation marks.
        (
            "unless the taxpayer includes with the taxpayer’s return",
            "unless the taxpayer includes with the taxpayer's return",
        ),
        ('the term "earned income" means', "the term “earned income” means"),
        # us-ok/statute/68-2906: the excerpt wrapped in its own quotation marks.
        (
            "Added by Laws 1988, c. 162, § 106, eff. Jan. 1, 1992.",
            "“Laws 1988”",
        ),
        # us/regulation/7/273/4: first letter recased to fit the quoting text.
        (
            "(iii) An unmarried dependent child of such Hmong or Highland "
            "Laotian who is under the age of 18",
            "an unmarried dependent child of such Hmong or Highland Laotian",
        ),
    ],
)
def test_proof_evidence_accepts_typographic_spelling(source_text, excerpt):
    assert _source_contains_proof_evidence(
        source_text=source_text,
        evidence_text=excerpt,
    )
    assert _evidence_issues(excerpt, source_text) == []


@pytest.mark.parametrize(
    ("source_text", "excerpt"),
    [
        # Ellipsis-stitched excerpts are not verbatim source text.
        (
            "A Individual (A1E) $1,276 $1,305 $1,329 $1,356 B Individual",
            "A Individual (A1E) ... $1,356",
        ),
        # An inserted word (us-nm/statute/7-2-18.1).
        (
            "an amount equal to the amount of federal credit for child and "
            "dependent care expenses",
            "the amount of the federal credit for child and dependent care expenses",
        ),
        # A substituted word (us-wv income maintenance manual).
        (
            "enables certain SNAP, Supplemental Security Income (SSI) and WV "
            "WORKS clients to receive a 20% discount",
            "qualifying SNAP, Supplemental Security Income (SSI) and WV WORKS "
            "clients to receive a 20% discount",
        ),
        # A table header joined to a row label the source never prints so.
        (
            "Table 1: Maximum Monthly Allotment Household Size 48 States\n"
            "1 $298 $383\n2 $546 $703",
            "Household Size 1",
        ),
        # Column-interleaved PDF text rebuilt by the model.
        (
            "Household with at All other least 1 member households age 60+ or "
            "disabled $4,500 $3,000",
            "Household with at least 1 member age 60+ or disabled $4,500",
        ),
        # A different number behind the currency-sign space.
        ("limited to $ 14,000 per account", "$4,000 per account"),
        ("limited to $ 4,000.50 per account", "$4,000 per account"),
        # A hyphen the source does not print, or a split it does not wrap.
        ("not more than twenty five dollars", "not more than twenty-five dollars"),
        ("income from self-employ- ment", "income from self-employment"),
        # A hyphen wrap never crosses a paragraph gap or a record boundary.
        ("not more than twenty-\n\nfive dollars", "not more than twenty-five dollars"),
        (
            "not more than twenty-\n\x1e\nfive dollars",
            "not more than twenty-five dollars",
        ),
        # Recasing is for the first letter only.
        (
            "(iii) An unmarried dependent child of such Hmong",
            "An Unmarried dependent child of such Hmong",
        ),
        # Quotation marks are removed only when they wrap the whole excerpt.
        ("Added by Laws 1988, c. 162", "“Laws 1988”, c. 163"),
        # A comma the source does not print.
        ("by September 30 2006 established", "by September 30, 2006, established"),
        # A spaced colon is ratio notation, not a label to close up.
        ("The official ratio is 100 : 200 dollars.", "200 dollars"),
        # An em dash is punctuation, never the minus sign of a number.
        ("Group—5 percent of the amount", "-5 percent of the amount"),
        # A dash in sign position keeps its own spelling: no excerpt may add,
        # drop or swap the sign the source prints.
        ("The adjustment factor is -500 for each household.", "\u2212500 for each"),
        ("The adjustment factor is \u2013500 for each household.", "-500 for each"),
        ("The adjustment factor is \u2212500 for each household.", "-500 for each"),
        # A dash between a letter and a digit is not folded either.
        ("The adjustment is Group-500 for each household.", "Group\u2212500 for each"),
        ("The adjustment is Group\u2013500 for each household.", "Group-500 for each"),
    ],
)
def test_proof_evidence_rejects_changed_text(source_text, excerpt):
    assert not _source_contains_proof_evidence(
        source_text=source_text,
        evidence_text=excerpt,
    )
    assert _evidence_issues(excerpt, source_text)


def test_typographic_match_reports_exact_source_span():
    source_text = "Prefix.\nnot more than twenty- ﬁve dollars ($25) applies."

    spans = _source_proof_evidence_spans(
        source_text=source_text,
        evidence_text="twenty-five dollars ($25)",
    )

    assert [source_text[start:end] for start, end in spans] == [
        "twenty- ﬁve dollars ($25)"
    ]
    assert _located_proof_evidence(
        source_text=source_text,
        evidence_text="twenty-five dollars ($25)",
    ) == ("twenty- ﬁve dollars ($25)",)


def test_empty_excerpt_is_never_typographic_evidence():
    for excerpt in ("", "   "):
        assert not source_contains_typographic_proof_evidence(
            source_text="abc", evidence_text=excerpt
        )
        assert not _source_contains_proof_evidence(
            source_text="abc", evidence_text=excerpt
        )


def test_numeric_evidence_binding_reads_typographic_spelling():
    source_text = "shall be limited to $ 4,000 per prepaid tuition contract."

    assert _source_evidence_fragment_is_body_bound(
        "$4,000 per prepaid tuition contract", source_text
    )
    assert not _source_evidence_fragment_is_body_bound(
        "$5,000 per prepaid tuition contract", source_text
    )


@pytest.mark.parametrize(
    ("source_text", "excerpt"),
    [
        ("The adjustment factor is -500 for each household.", "\u2212500 for each"),
        ("The adjustment factor is \u2013500 for each household.", "-500 for each"),
        ("The adjustment factor is \u2212500 for each household.", "-500 for each"),
        ("The adjustment is Group-500 for each household.", "Group\u2212500 for each"),
    ],
)
def test_numeric_evidence_binding_keeps_the_printed_sign(source_text, excerpt):
    assert not _source_evidence_fragment_is_body_bound(excerpt, source_text)


def test_verified_excerpt_pairs_read_typographic_spelling():
    source_text = "shall be limited to $ 4,000 per prepaid tuition contract."
    rule = {
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": "$4,000 per prepaid tuition contract",
                        },
                    },
                    {
                        "path": "versions[1].formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": "$5,000 per prepaid tuition contract",
                        },
                    },
                ]
            }
        }
    }

    pairs = _rule_verified_source_excerpt_pairs_by_path(rule, {CITATION: source_text})

    assert pairs == {
        "versions[0].formula": (("$4,000 per prepaid tuition contract", source_text),)
    }


def test_proof_evidence_accepts_code_joint_return_parenthetical():
    # us/statute/26/25A(d)(1): the parenthetical names an alternative amount.
    source_text = (
        "(ii) $80,000 ($160,000 in the case of a joint return), bears to "
        "(B) $10,000 ($20,000 in the case of a joint return)."
    )

    for excerpt in (
        "$160,000 in the case of a joint return",
        "$20,000 in the case of a joint return",
    ):
        assert _evidence_issues(excerpt, source_text) == []


@pytest.mark.parametrize(
    "excerpt",
    [
        # The bare amount still omits accounting-style parentheses.
        "$160,000",
        # Other parenthesized numeric prose keeps its existing treatment.
        "$50 of total assets for the current period",
    ],
)
def test_proof_evidence_keeps_accounting_parenthesis_boundary(excerpt):
    source_text = (
        "(ii) $80,000 ($160,000) bears to; Statement ($50 of total assets for "
        "the current period) filed."
    )

    assert _evidence_issues(excerpt, source_text)


_ORS_SOURCE = (
    "(1)(a) A tax is imposed for each taxable year on the entire taxable income "
    "of every resident of this state.\n"
    "(b) For tax years beginning in each calendar year, the Department of "
    "Revenue shall adopt a table.\n"
    "(c) The cost-of-living adjustment is the percentage increase.\n"
    "(d) As used in this subsection, the index means the published index.\n"
    "(e) If any increase determined under paragraph (b) of this subsection is "
    "not a multiple of $50, the increase shall be rounded to the next lower "
    "multiple of $50.\n"
    "(2)(a) A tax is imposed for each taxable year upon the entire taxable "
    "income of every part-year resident of this state.\n"
    "(e) If the part-year amount is not a multiple of $25, it shall be rounded "
    "down.\n"
)


def test_proof_scope_reads_fused_numeric_and_alpha_markers():
    # us-or/statute/316.037: ORS prints "(1)(a)" on one line.
    excerpt = (
        "(e) If any increase determined under paragraph (b) of this subsection "
        "is not a multiple of $50"
    )

    assert _evidence_issues(excerpt, _ORS_SOURCE, rule_source="ORS 316.037(1)(e)") == []


def test_proof_scope_rejects_alpha_child_of_other_fused_numeric_parent():
    excerpt = "(e) If the part-year amount is not a multiple of $25"

    issues = _evidence_issues(excerpt, _ORS_SOURCE, rule_source="ORS 316.037(1)(e)")

    assert any(
        "outside the rule's declared subsection scope" in issue for issue in issues
    )


_ORS_TWO_PARAGRAPHS = (
    "(1)(a) A tax is imposed on every resident.\n"
    "(b) The department shall adopt a table.\n"
    "(c) The adjustment is the percentage increase.\n"
    "(2)(a) A tax is imposed on every part-year resident.\n"
    "(b) The part-year table applies.\n"
    "(c) If the part-year amount is not a multiple of $25, it is rounded down.\n"
    "(3)(a) A tax is imposed on every nonresident.\n"
)


def test_proof_scope_reads_later_fused_numeric_paragraphs():
    excerpt = "(c) If the part-year amount is not a multiple of $25"

    assert (
        _evidence_issues(excerpt, _ORS_TWO_PARAGRAPHS, rule_source="ORS 316.037(2)(c)")
        == []
    )
    assert any(
        "outside the rule's declared subsection scope" in issue
        for issue in _evidence_issues(
            excerpt, _ORS_TWO_PARAGRAPHS, rule_source="ORS 316.037(1)(c)"
        )
    )
    assert any(
        "outside the rule's declared subsection scope" in issue
        for issue in _evidence_issues(
            "(c) The adjustment is the percentage increase",
            _ORS_TWO_PARAGRAPHS,
            rule_source="ORS 316.037(2)(c)",
        )
    )


def test_quoted_excerpt_with_several_printed_forms_keeps_scope_check():
    source_text = (
        "(1)(a) A tax is imposed.\n"
        "(b) The base amount is $50.\n"
        "(2)(a) A second tax applies.\n"
        "(b) The second amount applies.\n"
        "(c) The amount is $ 25 per month.\n"
        "(3)(a) A third tax applies.\n"
        "(b) The third amount applies.\n"
        "(c) The amount is $25 per month.\n"
    )

    for excerpt in (
        "(c) The amount is $25 per month",
        "“(c) The amount is $25 per month”",
    ):
        assert any(
            "outside the rule's declared subsection scope" in issue
            for issue in _evidence_issues(
                excerpt, source_text, rule_source="ORS 316.037(1)(b)"
            )
        ), excerpt
