from __future__ import annotations

from pathlib import Path

import pytest

from axiom_encode.corpus_resolver import (
    InvalidActiveCorpusSourceError,
    RepealedCorpusSourceError,
    resolve_local_corpus_source,
)
from axiom_encode.harness.pre_encode_gate import (
    CHECK_EXISTING_MODULE,
    CHECK_NO_OPERATIVE_TEXT,
    CHECK_REPEALED_TARGET,
    CHECK_TOC_PAGE,
    CHECK_UNENCODED_REFERENCE,
    GateSource,
    check_no_operative_text,
    check_toc_page,
    evaluate_pre_encode_gate,
    explicit_federal_references,
)
from axiom_encode.source_text_signals import TOC_MIN_LEADERS, is_table_of_contents
from tests.release_object_fixtures import bind_test_corpus_release

CONTENTS_ENTRIES = "\n".join(
    f"SNAP section {number}{'.' * 40} {number}" for number in range(12)
)
TOC_BODY = "Table of Contents\n" + CONTENTS_ENTRIES
OPERATIVE_BODY = (
    "A household is eligible when its net income does not exceed 100 percent "
    "of the poverty guideline."
)


def _source(body: str, *, labels=(), citation="us-or/manual/snap/a/1") -> GateSource:
    return GateSource(
        requested=citation, citation_path=citation, body=body, labels=tuple(labels)
    )


def _checks(result) -> list[tuple[str, bool]]:
    return [(finding.check, finding.enforce) for finding in result.findings]


# -- toc_page ---------------------------------------------------------------


def test_toc_page_blocks_a_contents_page_with_running_heads():
    body = (
        "West Virginia Income Maintenance Manual Chapter 11 Page i\n"
        "Chapter 11 Benefit Repayment Table of Contents (cont'd)\n"
        f"11.1 Benefit Repayment{'.' * 30} 1\n" + CONTENTS_ENTRIES
    )
    labels = ("Page 1081", "West Virginia Income Maintenance Manual 1081")

    assert check_toc_page(body, labels) is not None
    # Without the row's labels the running head is unexplained text.
    assert check_toc_page(body) is None


def test_toc_page_reports_a_contents_page_without_blocking():
    # toc_page is report-only: a judgement heuristic never stops an encode.
    result = evaluate_pre_encode_gate(_source(TOC_BODY))

    assert _checks(result) == [(CHECK_TOC_PAGE, False)]
    assert not result.blocked
    assert result.findings[0].failure_class == "selection"


def test_toc_page_keeps_policy_text_with_a_few_leaders():
    body = OPERATIVE_BODY + "\n" + "\n".join(f"Item{'.' * 9} {n}" for n in range(3))

    assert check_toc_page(body) is None
    assert not evaluate_pre_encode_gate(_source(body)).blocked


def test_toc_threshold_is_the_queue_builder_threshold():
    just_below = " ".join(f"x{'.' * 8}" for _ in range(TOC_MIN_LEADERS - 1))
    at_threshold = " ".join(f"x{'.' * 8}" for _ in range(TOC_MIN_LEADERS))

    assert not is_table_of_contents(just_below)
    assert is_table_of_contents(at_threshold)
    # Seven dots are not a leader run.
    assert not is_table_of_contents(" ".join("x......." for _ in range(20)))


# -- no_operative_text ------------------------------------------------------


@pytest.mark.parametrize("body", ["", "   \n\t  "])
def test_no_operative_text_blocks_an_empty_body(body):
    result = evaluate_pre_encode_gate(_source(body))

    assert _checks(result) == [(CHECK_NO_OPERATIVE_TEXT, True)]
    # The row lost its text: a corpus extraction gap, not a selection error.
    assert (result.findings[0].failure_class, result.findings[0].subclass) == (
        "corpus",
        "content",
    )


@pytest.mark.parametrize("body", [".", " . \n", "--", "\x00\x01"])
def test_no_operative_text_blocks_a_punctuation_only_body_as_corpus(body):
    result = evaluate_pre_encode_gate(_source(body))

    assert _checks(result) == [(CHECK_NO_OPERATIVE_TEXT, True)]
    assert (result.findings[0].failure_class, result.findings[0].subclass) == (
        "corpus",
        "content",
    )


@pytest.mark.parametrize(
    ("body", "citation"),
    [
        ("Eligibility Standards", "us-or/manual/snap/a/1"),
        ("  ELIGIBILITY   standards \n", "us-or/manual/snap/a/1"),
        ("Eligibility Standards\n461-155-0190", "us-or/regulation/461-155-0190"),
        ("§ 273.9 Eligibility Standards", "us/regulation/7/273/9"),
        ("Section 5. Eligibility Standards.", "us-xx/statute/1/5"),
        ("Eligibility Standards (a)", "us/statute/7/2014/a"),
    ],
)
def test_no_operative_text_blocks_a_heading_only_body(body, citation):
    result = evaluate_pre_encode_gate(
        _source(body, labels=("Eligibility Standards",), citation=citation)
    )

    assert _checks(result) == [(CHECK_NO_OPERATIVE_TEXT, True)]
    assert (result.findings[0].failure_class, result.findings[0].subclass) == (
        "selection",
        None,
    )


@pytest.mark.parametrize(
    "body",
    [
        OPERATIVE_BODY,
        "Eligibility Standards\n\n" + OPERATIVE_BODY,
        # A heading plus one word that is not a locator is still text.
        "Eligibility Standards Reserved.",
        "Eligibility Standards\n(a) Repealed.",
        # A numeric table body has no letters but more than one value.
        "100 200 300",
    ],
)
def test_no_operative_text_keeps_bodies_with_text(body):
    assert check_no_operative_text(body, ("Eligibility Standards",)) is None
    result = evaluate_pre_encode_gate(_source(body, labels=("Eligibility Standards",)))
    assert not result.blocked


def test_no_operative_text_without_labels_keeps_a_one_line_body():
    assert check_no_operative_text("Eligibility Standards") is None


# -- repealed_target --------------------------------------------------------


def test_repealed_target_blocks_in_enforce_mode():
    result = evaluate_pre_encode_gate(
        None, repealed_reason="Active corpus source 'us/statute/7/1' is repealed"
    )

    assert _checks(result) == [(CHECK_REPEALED_TARGET, True)]
    assert result.findings[0].evidence == (
        "Active corpus source 'us/statute/7/1' is repealed",
    )


def _write_rows(corpus_root: Path, rows: list[dict]) -> object:
    import json

    version = "2026-gate-test"
    jurisdiction, document_class = rows[0]["citation_path"].split("/")[:2]
    path = (
        corpus_root
        / "data/corpus/provisions"
        / jurisdiction
        / document_class
        / f"{version}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"test:{row['citation_path']}",
                    "jurisdiction": jurisdiction,
                    "document_class": document_class,
                    "version": version,
                    "source_path": f"sources/{jurisdiction}/{document_class}/test",
                    "source_as_of": "2026-01-01",
                    "expression_date": "2026-01-01",
                    **row,
                }
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return bind_test_corpus_release(
        corpus_root,
        "rulespec-test-release",
        [(jurisdiction, document_class, version)],
    )


def test_resolver_types_a_repealed_row(tmp_path):
    release = _write_rows(
        tmp_path / "axiom-corpus",
        [
            {
                "citation_path": "us/statute/7/2015",
                "body": None,
                "metadata": {"status": "repealed"},
            }
        ],
    )

    with pytest.raises(RepealedCorpusSourceError) as raised:
        resolve_local_corpus_source("us/statute/7/2015", release)

    # Callers that catch the old type still see the same failure.
    assert isinstance(raised.value, InvalidActiveCorpusSourceError)


def test_a_partial_repeal_resolves_and_does_not_block(tmp_path):
    release = _write_rows(
        tmp_path / "axiom-corpus",
        [
            {"citation_path": "us/statute/7/2016", "body": None},
            {
                "citation_path": "us/statute/7/2016/a",
                "heading": "(a) Repealed",
                "body": None,
                "metadata": {"status": "repealed"},
            },
            {
                "citation_path": "us/statute/7/2016/b",
                "heading": "(b) Allotment",
                "body": OPERATIVE_BODY,
            },
        ],
    )

    resolved = resolve_local_corpus_source("us/statute/7/2016", release)
    result = evaluate_pre_encode_gate(
        GateSource(
            requested="us/statute/7/2016",
            citation_path=resolved.citation_path,
            body=resolved.body,
        )
    )

    assert OPERATIVE_BODY in resolved.body
    assert result.findings == ()


# -- report mode ------------------------------------------------------------


def test_report_mode_never_blocks():
    for source, reason in (
        (_source(TOC_BODY), None),
        (_source(""), None),
        (None, "Active corpus source 'us/statute/7/1' is repealed"),
    ):
        result = evaluate_pre_encode_gate(source, mode="report", repealed_reason=reason)
        assert result.findings
        assert not result.blocked
        assert all(finding.enforce is False for finding in result.findings)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        evaluate_pre_encode_gate(_source(OPERATIVE_BODY), mode="off")


# -- existing_module (report) -----------------------------------------------


def test_existing_module_reports_an_existing_canonical_module(tmp_path):
    content_root = tmp_path / "rulespec-us" / "us"
    module = content_root / "statutes/7/2014.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")
    source = _source(OPERATIVE_BODY, citation="us/statute/7/2014")

    result = evaluate_pre_encode_gate(source, policy_repo_path=content_root)

    assert _checks(result) == [(CHECK_EXISTING_MODULE, False)]
    assert result.findings[0].evidence == ("statutes/7/2014.yaml",)
    assert not result.blocked


def test_existing_module_is_silent_for_a_replacement_or_a_new_module(tmp_path):
    content_root = tmp_path / "rulespec-us" / "us"
    module = content_root / "statutes/7/2014.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\n")

    replacing = evaluate_pre_encode_gate(
        _source(OPERATIVE_BODY, citation="us/statute/7/2014"),
        policy_repo_path=content_root,
        replacement_target=True,
    )
    new_module = evaluate_pre_encode_gate(
        _source(OPERATIVE_BODY, citation="us/statute/7/2015"),
        policy_repo_path=content_root,
    )

    assert replacing.findings == ()
    assert new_module.findings == ()


# -- unencoded_reference (report) -------------------------------------------


def test_explicit_federal_references_reads_only_usc_and_cfr_sections():
    body = (
        "As defined in 7 U.S.C. 2014(e) and 7 CFR 273.9(d)(1), and under "
        "OAR 461-155-0190 and section 5 of this chapter, see 7 C.F.R. § 273.10."
    )

    assert explicit_federal_references(body) == (
        "7 USC 2014",
        "7 CFR 273.9",
        "7 CFR 273.10",
    )


def test_unencoded_reference_reports_a_cited_row_with_no_module(tmp_path):
    checkout = tmp_path / "rulespec-us"
    encoded = checkout / "us/regulations/7-cfr/273/10/a.yaml"
    encoded.parent.mkdir(parents=True)
    encoded.write_text("format: rulespec/v1\n")
    rows = {"us/regulation/7/273/9", "us/regulation/7/273/10"}
    looked_up: list[str] = []

    def corpus_row_exists(path: str) -> bool:
        looked_up.append(path)
        return path in rows

    body = (
        "Income is counted under 7 CFR 273.9, deductions under 7 CFR 273.10, "
        "and resources under 7 CFR 273.8."
    )
    result = evaluate_pre_encode_gate(
        _source(body, citation="us-or/manual/snap/a/1"),
        rulespec_checkout_root=checkout,
        corpus_row_exists=corpus_row_exists,
    )

    assert _checks(result) == [(CHECK_UNENCODED_REFERENCE, False)]
    assert result.findings[0].failure_class == "dependency"
    # 273.10 has a module under it; 273.8 has no corpus row.
    assert result.findings[0].evidence == ("us/regulation/7/273/9",)
    assert "us/regulation/7/273/10" not in looked_up
    assert not result.blocked


def test_unencoded_reference_ignores_the_target_itself_and_lookup_errors(tmp_path):
    def broken_lookup(path: str) -> bool:
        raise RuntimeError("corpus unavailable")

    self_cite = evaluate_pre_encode_gate(
        _source("See 7 CFR 273.9(a) for details.", citation="us/regulation/7/273/9"),
        rulespec_checkout_root=tmp_path,
        corpus_row_exists=lambda path: True,
    )
    broken = evaluate_pre_encode_gate(
        _source("See 7 CFR 273.9 for details."),
        rulespec_checkout_root=tmp_path,
        corpus_row_exists=broken_lookup,
    )

    assert self_cite.findings == ()
    assert broken.findings == ()


def test_gate_result_to_dict_records_findings():
    payload = evaluate_pre_encode_gate(_source(".")).to_dict()

    assert payload["schema"] == "axiom-encode/pre-encode-gate/v1"
    assert payload["mode"] == "enforce"
    assert payload["blocked"] is True
    assert payload["findings"][0]["check"] == CHECK_NO_OPERATIVE_TEXT
    assert payload["findings"][0]["class"] == "corpus"
    assert payload["findings"][0]["subclass"] == "content"


# -- review round 1 regressions ---------------------------------------------


def test_cross_reference_body_is_not_a_locator():
    # us/statute/26/408A/c/4/A is the operative text "Section 401(a)(9)(A)."

    body = "Section 401(a)(9)(A)."
    labels = ("26 U.S.C. § 408A(c)(4)(A)",)

    assert check_no_operative_text(body, labels, "us/statute/26/408A/c/4/A") is None
    assert check_no_operative_text(body, (), "us/statute/26/408A/c/4/A") is None


def test_heading_with_the_targets_own_locator_still_blocks():
    assert check_no_operative_text(
        "Eligibility Standards 273.9",
        ("Eligibility Standards",),
        "us/regulation/7/273/9",
    )
    assert (
        check_no_operative_text(
            "Eligibility Standards 273.10",
            ("Eligibility Standards",),
            "us/regulation/7/273/9",
        )
        is None
    )


NE_FILL_IN_FORM = (
    "The certificate shall be substantially in the following form: COUNTY "
    "TREASURER'S CERTIFICATE OF TAX SALE. State of Nebraska .............. "
    "County, ss: I, .............. treasurer of the county of .............., "
    "do hereby certify that the following described real estate was, on the "
    ".......... day of .......... 20...., duly sold by me for the delinquent "
    "taxes for the years .....(list years)..... thereon, amounting to "
    ".......... dollars, to ........... for the sum of .......... dollars. "
    "the .........., heirs or assigns will be entitled to a deed on and after "
    "the .......... day of .......... A.D. 20.... .\n"
    "(L.S.) ....................., Treasurer."
)
OH_TAX_FORM = "\n".join(
    f"{n}. Credit line {n} (include a copy of the worksheet){'.' * 40}{n}."
    for n in range(1, 13)
)
FL_APPROPRIATIONS = "These funds shall be allocated as follows: " + " ".join(
    f"College {n}{'.' * 30} {n},355,831" for n in range(12)
)


@pytest.mark.parametrize(
    "body",
    [
        NE_FILL_IN_FORM,
        OH_TAX_FORM,
        FL_APPROPRIATIONS,
        # A contents-shaped list that never calls itself contents.
        CONTENTS_ENTRIES,
        # A chapter that opens with its contents list and then has its text.
        TOC_BODY + "\n" + OPERATIVE_BODY * 20,
        # A contents list next to one short rule (review round 2).
        TOC_BODY + "\n" + OPERATIVE_BODY,
        # Short, uppercase and numeric rules next to a contents list (round 3).
        TOC_BODY + "\nHouseholds must report changes within ten days.",
        TOC_BODY + "\nHOUSEHOLDS MUST REPORT CHANGES.",
        TOC_BODY + "\nStandard deduction: $198.",
        TOC_BODY + "\nBenefit reduction rate 30%.",
        TOC_BODY + "\nNo household may receive more than one allotment.",
        # Capitals and title case prove nothing (round 5).
        TOC_BODY + "\nSUBMIT APPLICATIONS WITHIN 30 DAYS.",
        TOC_BODY + "\nMaximum Certification Period: 12 Months.",
        TOC_BODY + "\nSUBMIT APPLICATIONS PROMPTLY",
        # Numeric limits without symbols and short imperatives (round 4).
        TOC_BODY + "\nCertification period: 12 months.",
        TOC_BODY + "\nSubmit applications within 30 days.",
    ],
)
def test_dot_leader_forms_and_value_tables_are_not_contents_pages(body):
    assert check_toc_page(body) is None


def test_content_checks_only_report_when_amendments_supply_the_text():
    """A Reserved principal with an operative amendment (review round 5)."""

    for body, labels in (("Reserved", ("Reserved",)), ("", ()), (TOC_BODY, ())):
        result = evaluate_pre_encode_gate(
            GateSource(
                requested="us/statute/7/2016",
                citation_path="us/statute/7/2016",
                body=body,
                labels=labels,
                amendment_context=True,
            )
        )
        assert result.findings
        assert not result.blocked


def test_gate_source_marks_amendment_context_from_the_source_unit():
    from types import SimpleNamespace

    def unit(**fields):
        return SimpleNamespace(
            requested="us/statute/7/2016",
            citation_path="us/statute/7/2016",
            body="Reserved",
            resolved_source=None,
            provision_metadata={},
            **fields,
        )

    assert not GateSource.from_source_unit(unit()).amendment_context
    assert GateSource.from_source_unit(
        unit(amendment_documents=(object(),))
    ).amendment_context
    assert GateSource.from_source_unit(
        unit(amendment_request="us/statute/7/2016")
    ).amendment_context


def test_contents_titles_do_not_explain_a_numeric_limit():
    """Entry titles that overlap the rule's words (review round 6)."""

    entries = "\n".join(
        [
            f"Maximum Certification Period{'.' * 30} 4",
            f"Households Certified for 12 Months{'.' * 30} 5",
        ]
        + [f"SNAP section {number}{'.' * 40} {number}" for number in range(10)]
    )
    body = (
        "Table of Contents\n" + entries + "\nMaximum Certification Period: 12 Months."
    )

    assert check_toc_page(body) is None
    # Being on page 12 does not make the limit's 12 a page number (round 7).
    assert check_toc_page(body, ("Page 12", "Manual 12")) is None
    # The same page without the limit is a contents page.
    assert check_toc_page("Table of Contents\n" + entries) is not None


def test_contents_page_numbers_and_revision_dates_are_locators():
    body = "12 (07/2026) Chapter 3 Table of Contents Page 12\n" + CONTENTS_ENTRIES

    assert check_toc_page(body, ("Page 12", "Oregon Notebook 12")) is not None
