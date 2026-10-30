"""Proof excerpts that differ from the source only typographically.

The cases come from US encoder runs and committed rulespec-us modules whose
proof excerpts quoted the printed source text but spelled it differently: a
ligature, a wrapped compound hyphen, an en dash, a space after a currency sign
or before a comma, curly quotation marks, a recased first letter, an added end
mark, or parentheses the excerpt left out.

The validator stays exact: it rejects these spellings. ``align_proof_excerpt``
locates the printed text they quote, and the generated-artifact repair stores
that text, which the validator then accepts. Each alignment has a negative
control of the same shape that changes a word, number, sign or structure and
must be neither accepted nor aligned.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.corpus_resolver import (
    CorpusSourceNotFoundError,
    CorpusSourceSliceError,
)
from axiom_encode.harness.proof_validator import (
    _located_proof_evidence,
    _source_contains_proof_evidence,
    _source_proof_evidence_spans,
    _typographic_source_evidence_spans,
    align_proof_excerpt,
    source_contains_typographic_proof_evidence,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import (
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
              excerpt: {json.dumps(excerpt)}
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
        # A ligature right before a line-end hyphen.
        ("it applies speci\ufb01-\ncally to the", "it applies specifically to the"),
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
def test_typographic_excerpt_aligns_to_the_printed_text(source_text, excerpt):
    # The model's spelling is not accepted as written ...
    assert not _source_contains_proof_evidence(
        source_text=source_text,
        evidence_text=excerpt,
    )
    assert _evidence_issues(excerpt, source_text)
    # ... it is rewritten to the printed source text, which is accepted.
    printed = align_proof_excerpt(source_text=source_text, excerpt=excerpt)
    assert printed is not None
    assert printed in source_text
    assert _evidence_issues(printed, source_text) == []


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
        # Ranges and section numbers never collapse: a dash or soft hyphen
        # between digits is never removed, and a dot never moves.
        ("for household sizes 2-5 only", "for household sizes 25 only"),
        ("for household sizes 2\u20135 only", "for household sizes 25 only"),
        ("for ages 18\u00ad64 only", "for ages 1864 only"),
        ("for ages 18\u00ad\n64 only", "for ages 1864 only"),
        ("for household sizes 25 only", "for household sizes 2-5 only"),
        ("for household sizes 25 only", "for household sizes 2\u20135 only"),
        ("as provided in \u00a7 273.4 of this part", "as provided in \u00a7 273.14"),
        ("as provided in \u00a7 273.4 of this part", "as provided in \u00a7 2734"),
        ("as provided in \u00a7 273.4 of this part", "as provided in \u00a7 27.34"),
        ("as provided in \u00a7 2 73 of this part", "as provided in \u00a7273"),
        ("limited to $1 000 per account", "$1000 per account"),
        ("limited to $1,000 per account", "$1000 per account"),
        # Quotation marks that change which words are quoted.
        ('the term "earned income" means', '"the term" earned income means'),
        ('the term "earned income" means', 'the "term earned" income means'),
    ],
)
def test_proof_evidence_rejects_changed_text(source_text, excerpt):
    assert not _source_contains_proof_evidence(
        source_text=source_text,
        evidence_text=excerpt,
    )
    assert _evidence_issues(excerpt, source_text)
    assert align_proof_excerpt(source_text=source_text, excerpt=excerpt) is None


def test_typographic_match_reports_exact_source_span():
    source_text = "Prefix.\nnot more than twenty- ﬁve dollars ($25) applies."

    # Clause binding reads printed text only, as the proof check does.
    assert (
        _source_proof_evidence_spans(
            source_text=source_text,
            evidence_text="twenty-five dollars ($25)",
        )
        == ()
    )
    spans = _typographic_source_evidence_spans(
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
        assert align_proof_excerpt(source_text="abc", excerpt=excerpt) is None


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

    # The printed excerpt is in (3)(c), outside the declared (1)(b).
    assert any(
        "outside the rule's declared subsection scope" in issue
        for issue in _evidence_issues(
            "(c) The amount is $25 per month",
            source_text,
            rule_source="ORS 316.037(1)(b)",
        )
    )
    # The quoted spelling matches two printed texts ("$ 25" and "$25"): it is
    # neither accepted nor aligned to either one.
    quoted = "“(c) The amount is $25 per month”"
    assert _evidence_issues(quoted, source_text, rule_source="ORS 316.037(1)(b)")
    assert align_proof_excerpt(source_text=source_text, excerpt=quoted) is None


# Alignment beyond typography (us/statute/42/416/l, SNAP FY 2026 COLA memo).
@pytest.mark.parametrize(
    ("source_text", "excerpt", "printed"),
    [
        # An end mark the source does not print there.
        (
            "(2) 66 years of age plus the number of months in the age increase "
            "factor for the calendar year in which such individual attains "
            "early retirement age; and",
            "66 years of age plus the number of months in the age increase "
            "factor for the calendar year in which such individual attains "
            "early retirement age.",
            "66 years of age plus the number of months in the age increase "
            "factor for the calendar year in which such individual attains "
            "early retirement age",
        ),
        # The parentheses a token-boundary rule needs to read the number.
        (
            "Where Elderly Disabled Are a Separate Household (165 Percent of "
            "Poverty Level) Household Size",
            "165 Percent of Poverty Level",
            "(165 Percent of Poverty Level)",
        ),
        (
            "Table 2: Gross Monthly Income Limit (130% of Federal Poverty Level) "
            "Household Size",
            "130% of Federal Poverty Level",
            "(130% of Federal Poverty Level)",
        ),
        # Straight quotation marks for printed curly ones: the quoted words
        # found inside the quoted span are the same occurrence.
        (
            "the term \u201cmagic phrase\u201d means",
            '"magic phrase"',
            "\u201cmagic phrase\u201d",
        ),
        # One printed text wrapped two ways is one text.
        (
            "the \u201cearned income\u201d and the \u201cearned\nincome\u201d",
            '"earned income"',
            "\u201cearned income\u201d",
        ),
    ],
)
def test_excerpt_aligns_beyond_typography(source_text, excerpt, printed):
    assert _evidence_issues(excerpt, source_text)
    assert align_proof_excerpt(source_text=source_text, excerpt=excerpt) == printed
    assert _evidence_issues(printed, source_text) == []


# False-evidence shapes from the offline accuracy test: none may be aligned.
@pytest.mark.parametrize(
    ("source_text", "excerpt"),
    [
        # A changed digit.
        ("Each Additional Member $218 $282 $360", "Each Additional Member $219"),
        # A span that starts inside a number.
        ("Under $11,564 $445.00 $1,124.00", "1,564 $445.00"),
        ("Under $11,564 $445.00 $1,124.00", "564 $445.00"),
        # An inserted negation.
        (
            "The spouse can transfer to you $5,000 minus the amounts used.",
            "The spouse can not transfer to you $5,000",
        ),
        # A far stitch.
        (
            "Mit Wohnung sind stationäre Räumlichkeiten gemeint. "
            + "x " * 800
            + "Der Wohnsitz wird beibehalten.",
            "stationäre Räumlichkeiten ... Der Wohnsitz wird beibehalten",
        ),
        # A quote across a sentence-ending paragraph break.
        (
            "pr. støttemåned, jf. dog § 25, stk. 5.\n\nStk. 2. Det supplerende "
            "stipendium kan ikke udbetales",
            "§ 25, stk. 5. Stk. 2. Det supplerende stipendium",
        ),
        # A case change inside the quote.
        (
            "Nach Abschluss einer erstmaligen Berufsausbildung wird ein Kind",
            "Nach Abschluss einer Erstmaligen Berufsausbildung",
        ),
        # A number the parentheses do not belong to.
        ("(ii) $80,000 ($160,000) bears to", "160,000"),
        # A parenthesized bare number is an accounting negative: never widened.
        ("Loss (500) reported", "500"),
        ("(ii) $80,000 ($160,000) bears to", "$160,000"),
        ("the rate (2.5) applies", "2.5"),
        # A sign before the parenthesis belongs to the number.
        ("total -(500) due", "500"),
        ("total \u2212(500) due", "500"),
    ],
)
def test_false_evidence_is_never_aligned(source_text, excerpt):
    assert _evidence_issues(excerpt, source_text)
    assert align_proof_excerpt(source_text=source_text, excerpt=excerpt) is None


_REPAIR_ISSUE = (
    "Proof source evidence not found: rule `{rule}` proof atom 0 "
    "`source.excerpt` does not appear in `{path}`."
)


def _write_rules(tmp_path, *, path: str, excerpt: str, summary: str = "") -> object:
    output_root = tmp_path / "out"
    rules_file = output_root / "model" / "policies" / "page.yaml"
    rules_file.parent.mkdir(parents=True)
    module = f"module:\n  summary: |-\n    {summary}\n" if summary else ""
    rules_file.write_text(
        "format: rulespec/v1\n" + module + "rules:\n"
        "- name: official_amount\n"
        "  kind: parameter\n"
        "  dtype: Rate\n"
        "  metadata:\n"
        "    proof:\n"
        "      atoms:\n"
        "      - path: versions[0].formula\n"
        "        kind: parameter\n"
        "        source:\n"
        f"          corpus_citation_path: {path}\n"
        f"          excerpt: {json.dumps(excerpt)}\n"
        "  versions:\n"
        "  - effective_from: '2025-10-01'\n"
        "    formula: 1.65\n"
    )
    return rules_file


def test_repair_stores_the_printed_text_of_a_typographic_excerpt(tmp_path):
    rules_file = _write_rules(
        tmp_path,
        path="us/guidance/example/page-3",
        excerpt="165 Percent of Poverty Level",
        summary="Separate Household (165 Percent of Poverty Level) Household Size",
    )

    repaired = cli._try_repair_generated_nonexact_proof_excerpts(
        rules_file,
        issues=[
            _REPAIR_ISSUE.format(
                rule="official_amount", path="us/guidance/example/page-3"
            )
        ],
    )

    assert repaired == ["official_amount[0]"]
    source = yaml.safe_load(rules_file.read_text())["rules"][0]["metadata"]["proof"][
        "atoms"
    ][0]["source"]
    assert source == {
        "corpus_citation_path": "us/guidance/example/page-3",
        "excerpt": "(165 Percent of Poverty Level)",
    }


_MEMO = "us/guidance/usda/fns/example-memo"
_MEMO_PAGES = {
    f"{_MEMO}/page-3": "State agencies that implement after the deadline will not "
    "apply the exclusion. The exclusion will not apply if the State agency "
    "does not implement the new provision.",
    f"{_MEMO}/page-4": "Quality control reviews follow the usual schedule.",
    f"{_MEMO}/page-5": "Contact the regional office with questions.",
}
_QUOTE = "will not apply if the State agency does not implement the new provision"


def _with_document(pages: dict[str, str], document: str = _MEMO) -> dict[str, str]:
    """Pages plus the document's release text, which joins its pages."""
    joined = "\n\x1e\n".join(
        text
        for path, text in sorted(
            pages.items(), key=lambda item: int(item[0].rsplit("-", 1)[1])
        )
        if path.startswith(document + "/")
    )
    return {**pages, document: joined}


def _corpus(monkeypatch, texts: dict[str, str], *, raising: set[str] = frozenset()):
    """A fake release: page texts, and documents composed from their pages."""

    def lookup(path, *, corpus_release):
        if path in raising:
            raise CorpusSourceSliceError(f"slice for {path} is absent")
        return texts.get(path)

    def resolve(path, corpus_release):
        if path in raising or path not in texts:
            raise CorpusSourceNotFoundError(path)
        rows = tuple(
            SimpleNamespace(citation_path=child)
            for child in texts
            if child.startswith(path + "/")
        )
        return SimpleNamespace(proof_evidence_text=texts[path], component_rows=rows)

    monkeypatch.setattr(cli, "_local_source_text_for_corpus_path", lookup)
    monkeypatch.setattr(cli, "resolve_local_corpus_source", resolve)
    return SimpleNamespace(name="test-release")


@pytest.fixture
def memo_corpus(monkeypatch):
    return _corpus(monkeypatch, _with_document(_MEMO_PAGES))


def _repair(rules_file, release, path=f"{_MEMO}/page-4"):
    return cli._try_repair_generated_nonexact_proof_excerpts(
        rules_file,
        corpus_release=release,
        issues=[_REPAIR_ISSUE.format(rule="official_amount", path=path)],
    )


def _atom_source(rules_file):
    return yaml.safe_load(rules_file.read_text())["rules"][0]["metadata"]["proof"][
        "atoms"
    ][0]["source"]


def test_repair_moves_a_quote_to_the_one_page_of_its_document_that_prints_it(
    tmp_path, memo_corpus
):
    rules_file = _write_rules(tmp_path, path=f"{_MEMO}/page-4", excerpt=_QUOTE)

    assert _repair(rules_file, memo_corpus) == ["official_amount[0]"]
    moved = _atom_source(rules_file)
    assert moved == {"corpus_citation_path": f"{_MEMO}/page-3", "excerpt": _QUOTE}
    # The moved atom passes the full proof check against the new page.
    content = rules_file.read_text()
    result = validate_rulespec_proofs(
        content, source_texts={f"{_MEMO}/page-3": _MEMO_PAGES[f"{_MEMO}/page-3"]}
    )
    assert not [i for i in result.issues if "Proof source" in i], result.issues


def test_repair_does_not_move_a_quote_two_pages_print(tmp_path, monkeypatch):
    pages = dict(_MEMO_PAGES)
    pages[f"{_MEMO}/page-5"] = pages[f"{_MEMO}/page-3"]
    release = _corpus(monkeypatch, _with_document(pages))
    rules_file = _write_rules(tmp_path, path=f"{_MEMO}/page-4", excerpt=_QUOTE)
    original = rules_file.read_text()

    assert _repair(rules_file, release) == []
    assert rules_file.read_text() == original


def test_repair_does_not_move_past_a_page_gap_or_the_scan(tmp_path, monkeypatch):
    # Pages 11-13 do not exist; page 15 prints the quote a second time, and so
    # does a page beyond any scan limit. The document text shows both copies.
    pages = dict(_MEMO_PAGES)
    pages.update({f"{_MEMO}/page-{n}": f"Page {n} text." for n in (6, 7, 8, 9, 10, 14)})
    pages[f"{_MEMO}/page-15"] = pages[f"{_MEMO}/page-3"]
    for extra in (
        {},
        {
            f"{_MEMO}/page-15": "Page 15 text.",
            f"{_MEMO}/page-999": pages[f"{_MEMO}/page-3"],
        },
    ):
        release = _corpus(monkeypatch, _with_document({**pages, **extra}))
        rules_file_dir = tmp_path / str(len(extra))
        rules_file = _write_rules(
            rules_file_dir, path=f"{_MEMO}/page-4", excerpt=_QUOTE
        )
        original = rules_file.read_text()

        assert _repair(rules_file, release) == []
        assert rules_file.read_text() == original


def test_repair_page_probe_survives_resolver_errors(tmp_path, monkeypatch):
    release = _corpus(
        monkeypatch,
        _with_document(_MEMO_PAGES),
        raising={f"{_MEMO}/page-1", f"{_MEMO}/page-2"},
    )
    rules_file = _write_rules(tmp_path, path=f"{_MEMO}/page-4", excerpt=_QUOTE)

    assert _repair(rules_file, release) == ["official_amount[0]"]
    assert _atom_source(rules_file)["corpus_citation_path"] == f"{_MEMO}/page-3"


def test_repair_keeps_the_citation_when_the_document_does_not_resolve(
    tmp_path, monkeypatch
):
    release = _corpus(monkeypatch, dict(_MEMO_PAGES), raising={_MEMO})
    rules_file = _write_rules(tmp_path, path=f"{_MEMO}/page-4", excerpt=_QUOTE)
    original = rules_file.read_text()

    assert _repair(rules_file, release) == []
    assert rules_file.read_text() == original


def test_repair_does_not_move_a_pinned_or_out_of_scope_atom(tmp_path, memo_corpus):
    pinned = _write_rules(tmp_path / "pinned", path=f"{_MEMO}/page-4", excerpt=_QUOTE)
    pinned.write_text(
        pinned.read_text().replace(
            "          excerpt:", "          source_sha256: abc\n          excerpt:"
        )
    )
    original = pinned.read_text()
    assert _repair(pinned, memo_corpus) == []
    assert pinned.read_text() == original

    scoped = _write_rules(tmp_path / "scoped", path=f"{_MEMO}/page-4", excerpt=_QUOTE)
    original = scoped.read_text()
    out_of_scope = (
        "Proof source evidence not found: rule `official_amount` proof atom 0 "
        "`source.excerpt` appears outside the rule's declared subsection scope "
        "`x(1)` (excerpt begins at `(2)`)."
    )
    cli._try_repair_generated_nonexact_proof_excerpts(
        scoped, corpus_release=memo_corpus, issues=[out_of_scope]
    )
    assert _atom_source(scoped)["corpus_citation_path"] == f"{_MEMO}/page-4"


def test_repair_never_moves_a_quote_to_another_section(tmp_path, monkeypatch):
    texts = {
        "us/statute/26/25A": "the taxable year increased by any amount excluded",
        "us/statute/26/24": "adjusted gross income increased by any amount excluded",
    }
    release = _corpus(monkeypatch, texts)
    rules_file = _write_rules(
        tmp_path,
        path="us/statute/26/25A",
        excerpt="adjusted gross income increased by any amount excluded",
    )
    cli._try_repair_generated_nonexact_proof_excerpts(
        rules_file,
        corpus_release=release,
        issues=[_REPAIR_ISSUE.format(rule="official_amount", path="us/statute/26/25A")],
    )

    # The page move is for pages of one document only: the citation stays on
    # 25A even though 26 USC 24 prints the quote. (Any excerpt change here
    # comes from the existing closest-exact-span repair on 25A's own text.)
    source = yaml.safe_load(rules_file.read_text())["rules"][0]["metadata"]["proof"][
        "atoms"
    ][0]["source"]
    assert source["corpus_citation_path"] == "us/statute/26/25A"
